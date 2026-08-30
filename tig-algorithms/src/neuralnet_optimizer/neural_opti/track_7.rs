use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

const PROFILE: S3Profile = S3Profile {
    _name: "hidden_7",
    total_steps: 1200,
    warmup_steps: 40,
    lr: 3.0e-3,
    beta: 0.95,
    eps: 1.0e-8,
    weight_decay: 0.01,
    min_lr_ratio: 0.10,
    coherence_gain: 0.25,
    progress_horizon_epochs: 50,
    arch_mantissa_bits: 18,
    enable_consensus_denom_mix: true,
    enable_bn_specialised: true,
    enable_plateau_lr_restart: true,
};

const GROUP_HIDDEN_WEIGHT: i32 = 0;
const GROUP_HIDDEN_BIAS: i32 = 1;
const GROUP_OUTPUT_WEIGHT: i32 = 2;
const GROUP_OUTPUT_BIAS: i32 = 3;
const GROUP_BN_WEIGHT: i32 = 4;
const GROUP_BN_BIAS: i32 = 5;
const GROUP_RUNNING_STAT: i32 = 6;

const FULL_HIDDEN_ELEMENTS: usize = 256 * 256;

#[derive(Clone, Copy, Debug)]
struct S3Profile {
    _name: &'static str,
    total_steps: usize,
    warmup_steps: usize,
    lr: f32,
    beta: f32,
    eps: f32,
    weight_decay: f32,
    min_lr_ratio: f32,
    coherence_gain: f32,
    progress_horizon_epochs: usize,
    arch_mantissa_bits: u32,
    enable_consensus_denom_mix: bool,
    enable_bn_specialised: bool,
    enable_plateau_lr_restart: bool,
}

impl S3Profile {
    fn with_overrides(mut self, hp: &Option<Map<String, Value>>) -> Self {
        let Some(h) = hp.as_ref() else { return self; };

        self.total_steps = read_usize(h, &["s3_total_steps", "total_steps"], self.total_steps);
        self.warmup_steps = read_usize(h, &["s3_warmup_steps", "warmup_steps"], self.warmup_steps);
        self.lr = read_f32(h, &["s3_lr", "lr"], self.lr);
        self.beta = read_f32(h, &["s3_beta", "beta"], self.beta);
        self.eps = read_f32(h, &["s3_eps", "eps"], self.eps);
        self.weight_decay = read_f32(h, &["s3_weight_decay", "weight_decay"], self.weight_decay);
        self.min_lr_ratio = read_f32(h, &["s3_min_lr_ratio", "min_lr_ratio"], self.min_lr_ratio);
        self.coherence_gain = read_f32(h, &["s3_coherence_gain", "coherence_gain"], self.coherence_gain);
        self.progress_horizon_epochs = read_usize(
            h,
            &["s3_progress_horizon_epochs", "progress_horizon_epochs"],
            self.progress_horizon_epochs,
        );
        self.arch_mantissa_bits = read_u32(
            h,
            &["s3_arch_mantissa_bits", "arch_mantissa_bits"],
            self.arch_mantissa_bits,
        ).clamp(8, 23);
        self.enable_consensus_denom_mix = read_bool(
            h,
            &["s3_enable_consensus_denom_mix", "enable_consensus_denom_mix"],
            self.enable_consensus_denom_mix,
        );
        self.enable_bn_specialised = read_bool(
            h,
            &["s3_enable_bn_specialised", "enable_bn_specialised"],
            self.enable_bn_specialised,
        );
        self.enable_plateau_lr_restart = read_bool(
            h,
            &["s3_enable_plateau_lr_restart", "enable_plateau_lr_restart"],
            self.enable_plateau_lr_restart,
        );

        self
    }
}

fn read_f32(h: &Map<String, Value>, keys: &[&str], default: f32) -> f32 {
    keys.iter()
        .find_map(|k| h.get(*k).and_then(Value::as_f64))
        .map(|x| x as f32)
        .unwrap_or(default)
}

fn read_usize(h: &Map<String, Value>, keys: &[&str], default: usize) -> usize {
    keys.iter()
        .find_map(|k| h.get(*k).and_then(Value::as_u64))
        .map(|x| x as usize)
        .unwrap_or(default)
}

#[inline]
fn read_u32(h: &Map<String, Value>, keys: &[&str], default: u32) -> u32 {
    keys.iter()
        .find_map(|k| h.get(*k).and_then(Value::as_u64))
        .map(|x| x as u32)
        .unwrap_or(default)
}

#[inline]
fn read_bool(h: &Map<String, Value>, keys: &[&str], default: bool) -> bool {
    keys.iter()
        .find_map(|k| {
            h.get(*k).and_then(|v| {
                v.as_bool()
                    .or_else(|| v.as_u64().map(|x| x != 0))
                    .or_else(|| v.as_f64().map(|x| x != 0.0))
            })
        })
        .unwrap_or(default)
}

#[inline]
fn canonicalize_f32(value: f32, keep_mantissa_bits: u32) -> f32 {
    if !value.is_finite() || value == 0.0 || keep_mantissa_bits >= 23 {
        return value;
    }

    let keep = keep_mantissa_bits.clamp(1, 23);
    let drop = 23 - keep;
    if drop == 0 {
        return value;
    }

    let bits = value.to_bits();
    let sign = bits & 0x8000_0000;
    let mut mag = bits & 0x7fff_ffff;
    let exp = mag & 0x7f80_0000;
    if exp == 0x7f80_0000 {
        return value;
    }

    let mask = (1u32 << drop) - 1;
    let half = 1u32 << (drop - 1);
    let rem = mag & mask;
    let lsb = (mag >> drop) & 1;
    mag &= !mask;
    if rem > half || (rem == half && lsb != 0) {
        mag = mag.saturating_add(1u32 << drop);
    }
    if mag >= 0x7f80_0000 {
        mag = 0x7f7f_ffff & !mask;
    }

    f32::from_bits(sign | mag)
}

thread_local! {
    static ACTIVE_PROFILE: std::cell::RefCell<Option<S3Profile>> = const { std::cell::RefCell::new(None) };
}

#[derive(Clone)]
struct S3State {
    momentum: Vec<CudaSlice<f32>>,
    power_momentum: Vec<CudaSlice<f32>>,
    updates: Vec<CudaSlice<f32>>,
    prev_grad: Vec<CudaSlice<f32>>,
    prev_grad2: Vec<CudaSlice<f32>>,
    ema_g2: Vec<CudaSlice<f32>>,
    groups: Vec<i32>,
    step: usize,
    lr_step: usize,
    last_lr_restart_epoch: usize,
    profile: S3Profile,
    best_val_loss: f32,
    best_val_epoch: usize,
    progress_credit: f32,
}

impl OptimizerStateTrait for S3State {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

fn solve_with_profile(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    profile: S3Profile,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let profile = profile.with_overrides(hyperparameters);
    ACTIVE_PROFILE.with(|p| *p.borrow_mut() = Some(profile));

    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init,
        optimizer_query,
        optimizer_step,
    )?;
    Ok(())
}

fn optimizer_init(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let profile = ACTIVE_PROFILE
        .with(|p| *p.borrow())
        .ok_or_else(|| anyhow!("S3 profile was not initialized"))?;

    let mut momentum = Vec::with_capacity(param_sizes.len());
    let mut power_momentum = Vec::with_capacity(param_sizes.len());
    let mut updates = Vec::with_capacity(param_sizes.len());
    let mut prev_grad = Vec::with_capacity(param_sizes.len());
    let mut prev_grad2 = Vec::with_capacity(param_sizes.len());
    let mut ema_g2 = Vec::with_capacity(param_sizes.len());

    for &n in param_sizes {
        momentum.push(stream.alloc_zeros::<f32>(n)?);
        power_momentum.push(stream.alloc_zeros::<f32>(n)?);
        updates.push(unsafe { stream.alloc::<f32>(n)? });
        prev_grad.push(stream.alloc_zeros::<f32>(n)?);
        prev_grad2.push(stream.alloc_zeros::<f32>(n)?);
        ema_g2.push(stream.alloc_zeros::<f32>(n)?);
    }

    Ok(Box::new(S3State {
        momentum,
        power_momentum,
        updates,
        prev_grad,
        prev_grad2,
        ema_g2,
        groups: infer_param_groups(param_sizes.len()),
        step: 0,
        lr_step: 0,
        last_lr_restart_epoch: 0,
        profile,
        best_val_loss: f32::INFINITY,
        best_val_epoch: 0,
        progress_credit: 1.0,
    }))
}

fn optimizer_query(
    _state: &dyn OptimizerStateTrait,
    _params: &[CudaSlice<f32>],
    _epoch: usize,
    _train_loss: Option<f32>,
    _val_loss: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

fn optimizer_step(
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
        .downcast_mut::<S3State>()
        .ok_or_else(|| anyhow!("Invalid S3 optimizer state"))?;

    update_progress_credit(state, epoch, val_loss);

    state.step += 1;
    state.lr_step += 1;

    if state.profile.enable_plateau_lr_restart {
        let horizon = state.profile.progress_horizon_epochs.max(1);
        let stale = epoch.saturating_sub(state.best_val_epoch);
        let since_restart = epoch.saturating_sub(state.last_lr_restart_epoch);
        if stale >= horizon && since_restart >= horizon {
            state.lr_step = state.profile.warmup_steps;
            state.last_lr_restart_epoch = epoch;
        }
    }

    let lr = state.profile.lr * learning_rate_scale(state.lr_step, &state.profile);
    let effective_gain = (state.profile.coherence_gain * state.progress_credit).clamp(0.0, 1.0);

    let base_kernel = module.load_function("s3_p3_base_update_7")?;
    let headroom_kernel = module.load_function("s3_p3_headroom_sqrt_update_7")?;
    let bn_affine_kernel = module.load_function("s3_p3_bn_affine_update_7")?;

    let mut result = Vec::with_capacity(gradients.len());
    for i in 0..gradients.len() {
        let n = gradients[i].len();
        if n == 0 {
            result.push(stream.alloc_zeros::<f32>(0)?);
            continue;
        }

        let group = state.groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            result.push(stream.alloc_zeros::<f32>(n)?);
            continue;
        }

        let launch = launch_config(n);
        let wd = decay_for_group(group, &state.profile);
        let use_headroom = group == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS;
        let use_bn_affine = state.profile.enable_bn_specialised
            && (group == GROUP_BN_WEIGHT || group == GROUP_BN_BIAS);
        let mix_flag: i32 = i32::from(state.profile.enable_consensus_denom_mix);

        unsafe {
            if use_bn_affine {
                stream.launch_builder(&bn_affine_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut state.power_momentum[i])
                    .arg(&mut state.ema_g2[i])
                    .arg(&mut state.updates[i])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&state.profile.eps)
                    .arg(&wd)
                    .arg(&state.profile.arch_mantissa_bits)
                    .launch(launch)?;
            } else if use_headroom {
                stream.launch_builder(&headroom_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut state.momentum[i])
                    .arg(&mut state.power_momentum[i])
                    .arg(&mut state.prev_grad[i])
                    .arg(&mut state.prev_grad2[i])
                    .arg(&mut state.updates[i])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&state.profile.eps)
                    .arg(&wd)
                    .arg(&effective_gain)
                    .arg(&state.profile.arch_mantissa_bits)
                    .arg(&mix_flag)
                    .launch(launch)?;
            } else {
                stream.launch_builder(&base_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut state.momentum[i])
                    .arg(&mut state.power_momentum[i])
                    .arg(&mut state.updates[i])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&state.profile.eps)
                    .arg(&wd)
                    .arg(&state.profile.arch_mantissa_bits)
                    .launch(launch)?;
            }
        }

        result.push(state.updates[i].clone());
    }

    Ok(result)
}

#[inline]
fn update_progress_credit(state: &mut S3State, epoch: usize, val_loss: Option<f32>) {
    let Some(v_raw) = val_loss.filter(|x| x.is_finite()) else { return; };
    let v = canonicalize_f32(v_raw, state.profile.arch_mantissa_bits);

    if v < state.best_val_loss {
        state.best_val_loss = v;
        state.best_val_epoch = epoch;
    }

    let horizon = state.profile.progress_horizon_epochs.max(1);
    let stale = epoch.saturating_sub(state.best_val_epoch);
    let x = (stale as f32 / horizon as f32).clamp(0.0, 1.0);
    state.progress_credit = 0.5 * (1.0 + (std::f32::consts::PI * x).cos());
}

#[inline]
fn learning_rate_scale(step: usize, profile: &S3Profile) -> f32 {
    if step <= profile.warmup_steps {
        return (step as f32 / profile.warmup_steps.max(1) as f32).clamp(0.0, 1.0);
    }

    let decay_steps = profile.total_steps.saturating_sub(profile.warmup_steps).max(1) as f32;
    let progress = (step.saturating_sub(profile.warmup_steps) as f32 / decay_steps).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    profile.min_lr_ratio + (1.0 - profile.min_lr_ratio) * cosine
}

#[inline]
fn decay_for_group(group: i32, profile: &S3Profile) -> f32 {
    match group {
        GROUP_HIDDEN_WEIGHT | GROUP_OUTPUT_WEIGHT => profile.weight_decay,
        _ => 0.0,
    }
}

fn infer_param_groups(param_count: usize) -> Vec<i32> {
    let hidden_layers = param_count.saturating_sub(2) / 6;
    let linear_layers = hidden_layers + 1;
    let mut groups = Vec::with_capacity(param_count);

    for layer in 0..linear_layers {
        if layer + 1 == linear_layers {
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

#[inline]
fn launch_config(n: usize) -> LaunchConfig {
    let block = 256u32;
    let grid = ((n as u32 + block - 1) / block).max(1);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    solve_with_profile(challenge, save_solution, hyperparameters, PROFILE, module, stream, prop)
}
