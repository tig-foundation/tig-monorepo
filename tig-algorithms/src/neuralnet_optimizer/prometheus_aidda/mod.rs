// Ported from aidda-swarm: best feasible neuralnet_optimizer algo
// (experiment 0f1e2968d010, agents u42-6/u42-7, swarm score 596,186.36).
// Role-scaled Cautious AdanW + cosine LR. Swarm-format hooks grafted under
// the tig-monorepo `solve_challenge`/`training_loop` entry point.
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let _ = hyperparameters; // tuned internally; hyperparameters ignored
    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init_state,
        optimizer_query_at_params,
        optimizer_step,
    )?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("Optimizer: role-scaled Cautious AdanW + cosine LR with warmup and plateau damping.");
    println!(" - Adan tracks EMAs of g, of gradient differences, and of the combined second moment.");
    println!(" - Linear weights, biases, output head, and BatchNorm affine params use separate LR/decay.");
    println!(" - Cautious sign mask: coordinates whose update opposes the raw gradient are damped to 0.25x.");
    println!(" - query_at_params is disabled to avoid extra full-model copies each batch.");
}

// ─── Hyper-constants ─────────────────────────────────────────────────────────
const LR_MAX: f32 = 3.4e-3;
const LR_MIN: f32 = 2e-5;
const WARMUP_EPOCHS: usize = 8;
const T_MAX_EPOCHS: usize = 700;
const EPS: f32 = 1e-8;
const HIDDEN_WEIGHT_DECAY: f32 = 1.6e-3;
const OUTPUT_WEIGHT_DECAY: f32 = 1e-5;
const PLATEAU_PATIENCE: usize = 12;
const PLATEAU_DECAY: f32 = 0.82;
const MIN_LR_SCALE: f32 = 0.35;
const VAL_IMPROVEMENT_EPS: f32 = 1e-7;
const GROUP_HIDDEN_WEIGHT: i32 = 0;
const GROUP_HIDDEN_BIAS: i32 = 1;
const GROUP_OUTPUT_WEIGHT: i32 = 2;
const GROUP_OUTPUT_BIAS: i32 = 3;
const GROUP_BN_WEIGHT: i32 = 4;
const GROUP_BN_BIAS: i32 = 5;
const GROUP_RUNNING_STAT: i32 = 6;

// ─── State ───────────────────────────────────────────────────────────────────
#[derive(Clone)]
struct OptimizerState {
    m: Vec<CudaSlice<f32>>, // first moment (EMA of g)
    v: Vec<CudaSlice<f32>>, // EMA of gradient differences (Adan)
    s: Vec<CudaSlice<f32>>, // Adan second moment n: EMA of (g + 0.92*g_diff)^2
    prev_grad: Vec<CudaSlice<f32>>,
    param_groups: Vec<i32>,
    step: usize,
    best_val_loss: f32,
    plateau_epochs: usize,
    lr_scale: f32,
    last_sched_epoch: usize,
}

impl OptimizerStateTrait for OptimizerState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

// Cosine-annealed LR schedule with a short linear warmup.
fn schedule_lr(epoch: usize) -> f32 {
    if epoch < WARMUP_EPOCHS {
        LR_MAX * ((epoch as f32) + 1.0) / (WARMUP_EPOCHS as f32).max(1.0)
    } else {
        let denom = (T_MAX_EPOCHS.saturating_sub(WARMUP_EPOCHS)).max(1) as f32;
        let p = (((epoch - WARMUP_EPOCHS) as f32) / denom).min(1.0);
        LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

fn infer_param_groups(param_count: usize) -> Vec<i32> {
    let hidden_layers = param_count.saturating_sub(2) / 6;
    let linear_layers = hidden_layers + 1;
    let mut groups = Vec::with_capacity(param_count);

    for layer_idx in 0..linear_layers {
        if layer_idx + 1 == linear_layers {
            groups.push(GROUP_OUTPUT_WEIGHT);
            groups.push(GROUP_OUTPUT_BIAS);
        } else {
            groups.push(GROUP_HIDDEN_WEIGHT);
            groups.push(GROUP_HIDDEN_BIAS);
        }
    }

    for _ in 0..hidden_layers {
        groups.push(GROUP_BN_WEIGHT);
        groups.push(GROUP_BN_BIAS);
        groups.push(GROUP_RUNNING_STAT);
        groups.push(GROUP_RUNNING_STAT);
    }

    groups.resize(param_count, GROUP_HIDDEN_WEIGHT);
    groups
}

fn group_hyperparams(group: i32, base_lr: f32) -> (f32, f32) {
    match group {
        GROUP_HIDDEN_WEIGHT => (base_lr, HIDDEN_WEIGHT_DECAY),
        GROUP_HIDDEN_BIAS => (base_lr * 1.25, 0.0),
        GROUP_OUTPUT_WEIGHT => (base_lr * 1.75, OUTPUT_WEIGHT_DECAY),
        GROUP_OUTPUT_BIAS => (base_lr * 2.0, 0.0),
        GROUP_BN_WEIGHT => (base_lr * 0.55, 0.0),
        GROUP_BN_BIAS => (base_lr * 0.8, 0.0),
        GROUP_RUNNING_STAT => (0.0, 0.0),
        _ => (base_lr, HIDDEN_WEIGHT_DECAY),
    }
}

// ─── Hooks ───────────────────────────────────────────────────────────────────
pub fn optimizer_init_state(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let mut m = Vec::with_capacity(param_sizes.len());
    let mut v = Vec::with_capacity(param_sizes.len());
    let mut s = Vec::with_capacity(param_sizes.len());
    let mut prev_grad = Vec::with_capacity(param_sizes.len());
    for &sz in param_sizes {
        m.push(stream.alloc_zeros::<f32>(sz)?);
        v.push(stream.alloc_zeros::<f32>(sz)?);
        s.push(stream.alloc_zeros::<f32>(sz)?);
        prev_grad.push(stream.alloc_zeros::<f32>(sz)?);
    }
    Ok(Box::new(OptimizerState {
        m,
        v,
        s,
        prev_grad,
        param_groups: infer_param_groups(param_sizes.len()),
        step: 0,
        best_val_loss: f32::INFINITY,
        plateau_epochs: 0,
        lr_scale: 1.0,
        last_sched_epoch: usize::MAX,
    }))
}

pub fn optimizer_query_at_params(
    _optimizer_state: &dyn OptimizerStateTrait,
    _model_params: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

pub fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    epoch: usize,
    _train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<OptimizerState>()
        .ok_or_else(|| anyhow!("Invalid optimizer state"))?;

    state.step += 1;
    if state.last_sched_epoch != epoch {
        state.last_sched_epoch = epoch;
        if let Some(v) = val_loss.filter(|v| v.is_finite()) {
            if v + VAL_IMPROVEMENT_EPS < state.best_val_loss {
                state.best_val_loss = v;
                state.plateau_epochs = 0;
                state.lr_scale = (state.lr_scale * 1.03).min(1.0);
            } else {
                state.plateau_epochs += 1;
                if state.plateau_epochs >= PLATEAU_PATIENCE {
                    state.lr_scale = (state.lr_scale * PLATEAU_DECAY).max(MIN_LR_SCALE);
                    state.plateau_epochs = 0;
                }
            }
        }
    }
    // On the very first step there is no previous gradient yet; the kernel
    // forces g_diff = 0 instead of treating prev_grad's zeros as a real value.
    let first_step: i32 = if state.step == 1 { 1 } else { 0 };
    let scheduled_lr = schedule_lr(epoch);
    let lr = LR_MIN + (scheduled_lr - LR_MIN) * state.lr_scale;
    let eps = EPS;

    let update_kernel = module.load_function("adabelief_update_kernel")?;
    let mut updates = Vec::with_capacity(gradients.len());

    for i in 0..gradients.len() {
        let n = gradients[i].len();
        let n_i = n as i32;
        let mut delta = stream.alloc_zeros::<f32>(n)?;
        let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            updates.push(delta);
            continue;
        }
        let (lr_i, wd_i) = group_hyperparams(group, lr);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            stream.launch_builder(&update_kernel)
                .arg(&gradients[i])
                .arg(&model_params[i])
                .arg(&mut state.m[i])
                .arg(&mut state.v[i])
                .arg(&mut state.s[i])
                .arg(&mut state.prev_grad[i])
                .arg(&lr_i)
                .arg(&eps)
                .arg(&wd_i)
                .arg(&first_step)
                .arg(&mut delta)
                .arg(&n_i)
                .launch(cfg)?;
        }
        updates.push(delta);
    }

    Ok(updates)
}
