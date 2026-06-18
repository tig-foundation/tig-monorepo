// prometheus_aidda_v2: per-track dispatcher.
//  - n_hidden=4  -> spectral-phase LR
//  - all others  -> role-scaled Cautious AdanW + cosine LR,
//                   with every tuned constant overridable via the
//                   hyperparameters JSON.
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

mod helpers;
mod track_n4;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    if challenge.num_hidden_layers == 4 {
        return track_n4::solve(
            challenge,
            save_solution,
            hyperparameters,
            module,
            stream,
            prop,
        );
    }
    CONFIG.with(|c| *c.borrow_mut() = Config::parse(hyperparameters));
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
    println!("prometheus_aidda_v2 - per-track optimizer:");
    println!(" - n_hidden=4: neural_extrem_v3 t29 track — spectral-phase LR with dual");
    println!("     consensus/Fisher and sign-EF kernels (hyperparameters: total_steps,");
    println!("     warmup_steps, noise_variance, spectral_boost, beta1, beta2, eps,");
    println!("     weight_decay, bn_layer_boost, output_layer_damping)");
    println!(" - other tracks: role-scaled Cautious AdanW + cosine LR with warmup and");
    println!("     plateau damping (hyperparameters: lr_max, lr_min, warmup_epochs,");
    println!("     t_max_epochs, hidden_wd, output_wd, plateau_patience, plateau_decay,");
    println!("     min_lr_scale, hidden_bias_lr_mult, output_weight_lr_mult,");
    println!("     output_bias_lr_mult, bn_weight_lr_mult, bn_bias_lr_mult, keep_damp,");
    println!("     beta_m, beta_v, beta_n, mix)");
}

// ─── Cautious AdanW config (defaults reproduce the tuned constants exactly) ──
const EPS: f32 = 1e-8;
const VAL_IMPROVEMENT_EPS: f32 = 1e-7;
const GROUP_HIDDEN_WEIGHT: i32 = 0;
const GROUP_HIDDEN_BIAS: i32 = 1;
const GROUP_OUTPUT_WEIGHT: i32 = 2;
const GROUP_OUTPUT_BIAS: i32 = 3;
const GROUP_BN_WEIGHT: i32 = 4;
const GROUP_BN_BIAS: i32 = 5;
const GROUP_RUNNING_STAT: i32 = 6;

#[derive(Clone, Copy)]
struct Config {
    lr_max: f32,
    lr_min: f32,
    warmup_epochs: usize,
    t_max_epochs: usize,
    hidden_wd: f32,
    output_wd: f32,
    plateau_patience: usize,
    plateau_decay: f32,
    min_lr_scale: f32,
    hidden_bias_lr_mult: f32,
    output_weight_lr_mult: f32,
    output_bias_lr_mult: f32,
    bn_weight_lr_mult: f32,
    bn_bias_lr_mult: f32,
    keep_damp: f32,
    beta_m: f32,
    beta_v: f32,
    beta_n: f32,
    mix: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lr_max: 3.4e-3,
            lr_min: 2e-5,
            warmup_epochs: 8,
            t_max_epochs: 700,
            hidden_wd: 1.6e-3,
            output_wd: 1e-5,
            plateau_patience: 12,
            plateau_decay: 0.82,
            min_lr_scale: 0.35,
            hidden_bias_lr_mult: 1.25,
            output_weight_lr_mult: 1.75,
            output_bias_lr_mult: 2.0,
            bn_weight_lr_mult: 0.55,
            bn_bias_lr_mult: 0.8,
            keep_damp: 0.25,
            beta_m: 0.98,
            beta_v: 0.92,
            beta_n: 0.99,
            mix: 0.92,
        }
    }
}

impl Config {
    fn parse(hyperparameters: &Option<Map<String, Value>>) -> Self {
        let mut cfg = Self::default();
        let Some(hp) = hyperparameters else {
            return cfg;
        };
        let f = |key: &str, default: f32| -> f32 {
            hp.get(key).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(default)
        };
        let u = |key: &str, default: usize| -> usize {
            hp.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default)
        };
        cfg.lr_max = f("lr_max", cfg.lr_max);
        cfg.lr_min = f("lr_min", cfg.lr_min);
        cfg.warmup_epochs = u("warmup_epochs", cfg.warmup_epochs);
        cfg.t_max_epochs = u("t_max_epochs", cfg.t_max_epochs);
        cfg.hidden_wd = f("hidden_wd", cfg.hidden_wd);
        cfg.output_wd = f("output_wd", cfg.output_wd);
        cfg.plateau_patience = u("plateau_patience", cfg.plateau_patience);
        cfg.plateau_decay = f("plateau_decay", cfg.plateau_decay);
        cfg.min_lr_scale = f("min_lr_scale", cfg.min_lr_scale);
        cfg.hidden_bias_lr_mult = f("hidden_bias_lr_mult", cfg.hidden_bias_lr_mult);
        cfg.output_weight_lr_mult = f("output_weight_lr_mult", cfg.output_weight_lr_mult);
        cfg.output_bias_lr_mult = f("output_bias_lr_mult", cfg.output_bias_lr_mult);
        cfg.bn_weight_lr_mult = f("bn_weight_lr_mult", cfg.bn_weight_lr_mult);
        cfg.bn_bias_lr_mult = f("bn_bias_lr_mult", cfg.bn_bias_lr_mult);
        cfg.keep_damp = f("keep_damp", cfg.keep_damp);
        cfg.beta_m = f("beta_m", cfg.beta_m);
        cfg.beta_v = f("beta_v", cfg.beta_v);
        cfg.beta_n = f("beta_n", cfg.beta_n);
        cfg.mix = f("mix", cfg.mix);
        cfg
    }
}

thread_local! {
    static CONFIG: RefCell<Config> = RefCell::new(Config::default());
}

// ─── State ───────────────────────────────────────────────────────────────────
#[derive(Clone)]
struct OptimizerState {
    m: Vec<CudaSlice<f32>>, // first moment (EMA of g)
    v: Vec<CudaSlice<f32>>, // EMA of gradient differences (Adan)
    s: Vec<CudaSlice<f32>>, // Adan second moment n: EMA of (g + mix*g_diff)^2
    prev_grad: Vec<CudaSlice<f32>>,
    param_groups: Vec<i32>,
    cfg: Config,
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
fn schedule_lr(cfg: &Config, epoch: usize) -> f32 {
    if epoch < cfg.warmup_epochs {
        cfg.lr_max * ((epoch as f32) + 1.0) / (cfg.warmup_epochs as f32).max(1.0)
    } else {
        let denom = (cfg.t_max_epochs.saturating_sub(cfg.warmup_epochs)).max(1) as f32;
        let p = (((epoch - cfg.warmup_epochs) as f32) / denom).min(1.0);
        cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1.0 + (std::f32::consts::PI * p).cos())
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

fn group_hyperparams(cfg: &Config, group: i32, base_lr: f32) -> (f32, f32) {
    match group {
        GROUP_HIDDEN_WEIGHT => (base_lr, cfg.hidden_wd),
        GROUP_HIDDEN_BIAS => (base_lr * cfg.hidden_bias_lr_mult, 0.0),
        GROUP_OUTPUT_WEIGHT => (base_lr * cfg.output_weight_lr_mult, cfg.output_wd),
        GROUP_OUTPUT_BIAS => (base_lr * cfg.output_bias_lr_mult, 0.0),
        GROUP_BN_WEIGHT => (base_lr * cfg.bn_weight_lr_mult, 0.0),
        GROUP_BN_BIAS => (base_lr * cfg.bn_bias_lr_mult, 0.0),
        GROUP_RUNNING_STAT => (0.0, 0.0),
        _ => (base_lr, cfg.hidden_wd),
    }
}

// ─── Hooks (non-n4 tracks) ───────────────────────────────────────────────────
pub fn optimizer_init_state(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let cfg = CONFIG.with(|c| *c.borrow());
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
        cfg,
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
    let cfg = state.cfg;

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
                if state.plateau_epochs >= cfg.plateau_patience {
                    state.lr_scale = (state.lr_scale * cfg.plateau_decay).max(cfg.min_lr_scale);
                    state.plateau_epochs = 0;
                }
            }
        }
    }
    // On the very first step there is no previous gradient yet; the kernel
    // forces g_diff = 0 instead of treating prev_grad's zeros as a real value.
    let first_step: i32 = if state.step == 1 { 1 } else { 0 };
    let scheduled_lr = schedule_lr(&cfg, epoch);
    let lr = cfg.lr_min + (scheduled_lr - cfg.lr_min) * state.lr_scale;
    let eps = EPS;
    let keep_damp = cfg.keep_damp;

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
        let (lr_i, wd_i) = group_hyperparams(&cfg, group, lr);
        let cfg_launch = LaunchConfig::for_num_elems(n as u32);
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
                .arg(&cfg.beta_m)
                .arg(&cfg.beta_v)
                .arg(&cfg.beta_n)
                .arg(&cfg.mix)
                .arg(&keep_damp)
                .arg(&first_step)
                .arg(&mut delta)
                .arg(&n_i)
                .launch(cfg_launch)?;
        }
        updates.push(delta);
    }

    Ok(updates)
}
