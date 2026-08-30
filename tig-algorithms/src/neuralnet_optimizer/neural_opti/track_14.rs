use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

const PROFILE: S3Profile = S3Profile {
    _name: "hidden_14",
    total_steps: 1950,
    warmup_steps: 200,
    lr: 3.0e-3,
    beta: 0.95,
    eps: 1.0e-8,
    weight_decay: 0.01,
    min_lr_ratio: 0.10,
    coherence_gain: 0.25,
    progress_horizon_epochs: 50,
    arch_mantissa_bits: 18,
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

#[derive(Clone)]
struct S3State {
    momentum: Vec<CudaSlice<f32>>,
    power_momentum: Vec<Option<CudaSlice<f32>>>,
    max_power_momentum: Vec<Option<CudaSlice<f32>>>,
    previous_gradients: Vec<Option<CudaSlice<f32>>>,
    updates: Vec<CudaSlice<f32>>,
    groups: Vec<i32>,
    seed: [u8; 32],
    step: usize,
    profile: S3Profile,
    best_val_loss: f32,
    best_val_epoch: usize,
    progress_credit: f32,
    diversification_armed: bool,
    base_descriptors: Option<CudaSlice<u64>>,
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
    let _ = (hyperparameters, profile);

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
    seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let profile = PROFILE;

    let groups = infer_param_groups(param_sizes.len());
    let mut momentum = Vec::with_capacity(param_sizes.len());
    let mut power_momentum = Vec::with_capacity(param_sizes.len());
    let mut max_power_momentum = Vec::with_capacity(param_sizes.len());
    let mut previous_gradients = Vec::with_capacity(param_sizes.len());
    let mut updates = Vec::with_capacity(param_sizes.len());
    let base_descriptor_words = param_sizes.iter().enumerate()
        .filter(|(i, &n)| {
            groups[*i] != GROUP_RUNNING_STAT
                && !(groups[*i] == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS)
                && n != 0
        })
        .map(|(_, &n)| ((n + 255) / 256) * 6)
        .sum::<usize>();
    let base_descriptors = if base_descriptor_words == 0 {
        None
    } else {
        Some(stream.alloc_zeros::<u64>(base_descriptor_words)?)
    };

    for (i, &n) in param_sizes.iter().enumerate() {
        momentum.push(stream.alloc_zeros::<f32>(n)?);
        let uses_hidden_s3 = groups[i] == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS;
        if uses_hidden_s3 {
            power_momentum.push(Some(stream.alloc_zeros::<f32>(n)?));
            max_power_momentum.push(Some(stream.alloc_zeros::<f32>(n)?));
            previous_gradients.push(Some(stream.alloc_zeros::<f32>(n)?));
        } else {
            power_momentum.push(None);
            max_power_momentum.push(None);
            previous_gradients.push(None);
        }
        updates.push(stream.alloc_zeros::<f32>(n)?);
    }

    Ok(Box::new(S3State {
        momentum,
        power_momentum,
        max_power_momentum,
        previous_gradients,
        updates,
        groups,
        seed,
        step: 0,
        profile,
        best_val_loss: f32::INFINITY,
        best_val_epoch: 0,
        progress_credit: 1.0,
        diversification_armed: true,
        base_descriptors,
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

    let diversify = update_progress_credit(state, epoch, val_loss);

    state.step += 1;
    let lr = state.profile.lr * learning_rate_scale(state.step, &state.profile);
    let effective_gain = (state.profile.coherence_gain * state.progress_credit).clamp(0.0, 1.0);
    let seed_key = state.seed.iter().fold(0u32, |key, &byte| {
        key.rotate_left(5) ^ byte as u32
    });

    let base_dispatch_kernel = module.load_function("s3_p3_base_dispatch_update_14")?;
    let headroom_kernel = module.load_function("s3_p3_headroom_sqrt_update_14")?;
    let mut base_tile_words = Vec::new();

    for i in 0..gradients.len() {
        let n = gradients[i].len();
        if n == 0 {
            continue;
        }

        let group = state.groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            continue;
        }

        let wd = decay_for_group(group, &state.profile);
        let use_headroom = group == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS;
        if use_headroom {
            unsafe {
                stream.launch_builder(&headroom_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut state.momentum[i])
                    .arg(state.power_momentum[i].as_mut().ok_or_else(|| anyhow!("Missing hidden cubic momentum"))?)
                    .arg(state.max_power_momentum[i].as_mut().ok_or_else(|| anyhow!("Missing AMSGrad envelope"))?)
                    .arg(state.previous_gradients[i].as_mut().ok_or_else(|| anyhow!("Missing previous-gradient history"))?)
                    .arg(&mut state.updates[i])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&state.profile.eps)
                    .arg(&wd)
                    .arg(&effective_gain)
                    .arg(&state.profile.arch_mantissa_bits)
                    .arg(&(diversify as u32))
                    .arg(&seed_key)
                    .arg(&(i as u32))
                    .launch(launch_config(n))?;
            }
            continue;
        }

        let (gradient_ptr, _gradient_sync) = gradients[i].device_ptr(&stream);
        let (parameter_ptr, _parameter_sync) = model_params[i].device_ptr(&stream);
        let (momentum_ptr, _momentum_sync) = state.momentum[i].device_ptr(&stream);
        let (update_ptr, _update_sync) = state.updates[i].device_ptr(&stream);

        for tile_base in (0..n).step_by(256) {
            base_tile_words.extend_from_slice(&[
                gradient_ptr,
                parameter_ptr,
                momentum_ptr,
                update_ptr,
                n as u64,
                ((wd.to_bits() as u64) << 32) | tile_base as u64,
            ]);
        }
    }

    if !base_tile_words.is_empty() {
        let descriptor_count = (base_tile_words.len() / 6) as u32;
        let base_descriptors = state.base_descriptors.as_mut()
            .ok_or_else(|| anyhow!("Missing base descriptor storage"))?;
        stream.memcpy_htod(&base_tile_words, base_descriptors)?;
        unsafe {
            stream.launch_builder(&base_dispatch_kernel)
                .arg(base_descriptors)
                .arg(&descriptor_count)
                .arg(&lr)
                .arg(&state.profile.beta)
                .arg(&state.profile.arch_mantissa_bits)
                .launch(base_batch_launch_config(descriptor_count))?;
        }
    }

    let mut result = Vec::with_capacity(gradients.len());
    for (i, gradient) in gradients.iter().enumerate() {
        let n = gradient.len();
        let group = state.groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if n == 0 || group == GROUP_RUNNING_STAT {
            result.push(stream.alloc_zeros::<f32>(n)?);
        } else {
            result.push(state.updates[i].clone());
        }
    }
    Ok(result)
}

#[inline]
fn update_progress_credit(state: &mut S3State, epoch: usize, val_loss: Option<f32>) -> bool {
    let Some(v_raw) = val_loss.filter(|x| x.is_finite()) else { return false; };
    let v = canonicalize_f32(v_raw, state.profile.arch_mantissa_bits);

    if v < state.best_val_loss {
        state.best_val_loss = v;
        state.best_val_epoch = epoch;
        state.diversification_armed = true;
    }

    let horizon = state.profile.progress_horizon_epochs.max(1);
    let stale = epoch.saturating_sub(state.best_val_epoch);
    let x = (stale as f32 / horizon as f32).clamp(0.0, 1.0);
    state.progress_credit = 0.5 * (1.0 + (std::f32::consts::PI * x).cos());

    if stale >= horizon && state.diversification_armed {
        state.diversification_armed = false;
        true
    } else {
        false
    }
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
fn base_batch_launch_config(tile_count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (tile_count, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
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

