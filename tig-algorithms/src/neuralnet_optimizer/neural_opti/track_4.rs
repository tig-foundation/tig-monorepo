use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

const PROFILE: S3Profile = S3Profile {
    _name: "hidden_4",
    total_steps: 3500,
    warmup_steps: 16,
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
    directional_momentum: Vec<CudaSlice<f32>>,
    power_momentum: Vec<CudaSlice<f32>>,
    monotone_cubic_envelope: Vec<CudaSlice<f32>>,
    updates: Vec<CudaSlice<f32>>,
    groups: Vec<i32>,
    step: usize,
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
    let _ = hyperparameters;
    let _ = profile;

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
    let profile = PROFILE;

    let groups = infer_param_groups(param_sizes.len());
    let mut momentum = Vec::with_capacity(param_sizes.len());
    let mut directional_momentum = Vec::with_capacity(param_sizes.len());
    let mut power_momentum = Vec::with_capacity(param_sizes.len());
    let mut monotone_cubic_envelope = Vec::with_capacity(param_sizes.len());
    let mut updates = Vec::with_capacity(param_sizes.len());

    for (i, &n) in param_sizes.iter().enumerate() {
        let affine_weight = affine_hidden_pair_at(&groups, param_sizes, i);
        let affine_bias = i > 0 && affine_hidden_pair_at(&groups, param_sizes, i - 1);

        momentum.push(stream.alloc_zeros::<f32>(n)?);
        directional_momentum.push(stream.alloc_zeros::<f32>(n)?);
        power_momentum.push(stream.alloc_zeros::<f32>(
            if affine_weight {
                n / 256
            } else if affine_bias || groups[i] == GROUP_RUNNING_STAT {
                0
            } else {
                n
            },
        )?);
        monotone_cubic_envelope.push(stream.alloc_zeros::<f32>(
            if affine_weight || affine_bias || groups[i] == GROUP_RUNNING_STAT {
                0
            } else {
                n
            },
        )?);
        updates.push(stream.alloc_zeros::<f32>(n)?);
    }

    Ok(Box::new(S3State {
        momentum,
        directional_momentum,
        power_momentum,
        monotone_cubic_envelope,
        updates,
        groups,
        step: 0,
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
    let lr = state.profile.lr * learning_rate_scale(state.step, &state.profile);
    let effective_gain = (state.profile.coherence_gain * state.progress_credit).clamp(0.0, 1.0);
    let raw_nesterov_bias_correction =
        1.0 - state.profile.beta.powi(state.step.saturating_add(1) as i32);
    let nesterov_bias_correction = canonicalize_f32(
        if raw_nesterov_bias_correction.is_finite() && raw_nesterov_bias_correction > 0.0 {
            raw_nesterov_bias_correction
        } else {
            1.0
        },
        state.profile.arch_mantissa_bits,
    );

    let base_kernel = module.load_function("s3_p3_base_update_4")?;
    let headroom_kernel = module.load_function("s3_p3_headroom_sqrt_update_4")?;
    let affine_kernel = module.load_function("s3_p3_affine_row_update_4")?;

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

        let affine_weight = i + 1 < gradients.len()
            && group == GROUP_HIDDEN_WEIGHT
            && n == FULL_HIDDEN_ELEMENTS
            && state.groups.get(i + 1).copied() == Some(GROUP_HIDDEN_BIAS)
            && gradients[i + 1].len() == 256
            && model_params[i + 1].len() == 256;
        let affine_bias = i > 0
            && group == GROUP_HIDDEN_BIAS
            && state.groups.get(i - 1).copied() == Some(GROUP_HIDDEN_WEIGHT)
            && gradients[i - 1].len() == FULL_HIDDEN_ELEMENTS
            && n == 256;

        if affine_bias {
            result.push(state.updates[i].clone());
            continue;
        }

        let launch = launch_config(n);
        let wd = decay_for_group(group, &state.profile);
        let use_headroom =
            group == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS && !affine_weight;

        unsafe {
            if affine_weight {
                let bias_index = i + 1;
                let (weight_momentum, bias_momentum) = state.momentum.split_at_mut(bias_index);
                let (weight_directional_momentum, bias_directional_momentum) =
                    state.directional_momentum.split_at_mut(bias_index);
                let (weight_updates, bias_updates) = state.updates.split_at_mut(bias_index);

                stream.launch_builder(&affine_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&gradients[bias_index])
                    .arg(&model_params[bias_index])
                    .arg(&mut weight_momentum[i])
                    .arg(&mut bias_momentum[0])
                    .arg(&mut weight_directional_momentum[i])
                    .arg(&mut bias_directional_momentum[0])
                    .arg(&mut state.power_momentum[i])
                    .arg(&mut weight_updates[i])
                    .arg(&mut bias_updates[0])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&state.profile.eps)
                    .arg(&wd)
                    .arg(&effective_gain)
                    .arg(&state.profile.arch_mantissa_bits)
                    .launch(launch)?;
            } else if use_headroom {
                stream.launch_builder(&headroom_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut state.momentum[i])
                    .arg(&mut state.power_momentum[i])
                    .arg(&mut state.monotone_cubic_envelope[i])
                    .arg(&mut state.updates[i])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&state.profile.eps)
                    .arg(&wd)
                    .arg(&effective_gain)
                    .arg(&state.profile.arch_mantissa_bits)
                    .launch(launch)?;
            } else {
                stream.launch_builder(&base_kernel)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut state.momentum[i])
                    .arg(&mut state.power_momentum[i])
                    .arg(&mut state.monotone_cubic_envelope[i])
                    .arg(&mut state.updates[i])
                    .arg(&(n as u32))
                    .arg(&lr)
                    .arg(&state.profile.beta)
                    .arg(&nesterov_bias_correction)
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

fn affine_hidden_pair_at(groups: &[i32], param_sizes: &[usize], index: usize) -> bool {
    index + 1 < param_sizes.len()
        && groups.get(index).copied() == Some(GROUP_HIDDEN_WEIGHT)
        && param_sizes[index] == FULL_HIDDEN_ELEMENTS
        && groups.get(index + 1).copied() == Some(GROUP_HIDDEN_BIAS)
        && param_sizes[index + 1] == 256
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
