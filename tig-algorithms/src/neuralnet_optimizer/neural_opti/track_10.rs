use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

const PROFILE: S3Profile = S3Profile {
    _name: "hidden_10",
    total_steps: 2850,
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
struct S3Kernels {
    base_update: CudaFunction,
    fused_dual_loss_update: CudaFunction,
    base_update_sam: CudaFunction,
    fused_dual_loss_update_sam_row_trust: CudaFunction,
    reduce_sum_sq: CudaFunction,
    reduce_sum_sq_dual: CudaFunction,
    scale_updates: CudaFunction,
    row_trust_scale: CudaFunction,
}

#[derive(Clone)]
struct S3State {
    momentum: Vec<CudaSlice<f32>>,
    power_momentum: Vec<CudaSlice<f32>>,
    updates: Vec<CudaSlice<f32>>,
    reduce_scratch: CudaSlice<f32>,
    reduce_offsets: Vec<usize>,
    max_reduce_blocks: usize,
    groups: Vec<i32>,
    step: usize,
    profile: S3Profile,
    best_val_loss: f32,
    best_val_epoch: usize,
    best_train_loss: f32,
    headroom_allowed: bool,
    progress_credit: f32,
    prev_grad: Vec<CudaSlice<f32>>,
    kernels: S3Kernels,
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
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let profile = ACTIVE_PROFILE
        .with(|p| *p.borrow())
        .ok_or_else(|| anyhow!("S3 profile was not initialized"))?;

    let kernels = S3Kernels {
        base_update: module.load_function("s3_p3_base_update_10")?,
        fused_dual_loss_update: module.load_function("s3_p3_fused_dual_loss_update_10")?,
        base_update_sam: module.load_function("s3_p3_base_update_sam_10")?,
        fused_dual_loss_update_sam_row_trust: module
            .load_function("s3_p3_fused_dual_loss_update_sam_row_trust_10")?,
        reduce_sum_sq: module.load_function("s3_reduce_sum_sq_10")?,
        reduce_sum_sq_dual: module.load_function("s3_reduce_sum_sq_dual_10")?,
        scale_updates: module.load_function("s3_scale_updates_10")?,
        row_trust_scale: module.load_function("s3_row_trust_scale_10")?,
    };

    let mut momentum = Vec::with_capacity(param_sizes.len());
    let mut power_momentum = Vec::with_capacity(param_sizes.len());
    let mut updates = Vec::with_capacity(param_sizes.len());
    let mut prev_grad = Vec::with_capacity(param_sizes.len());

    for &n in param_sizes {
        momentum.push(stream.alloc_zeros::<f32>(n)?);
        power_momentum.push(stream.alloc_zeros::<f32>(n)?);
        updates.push(unsafe { stream.alloc::<f32>(n)? });
        prev_grad.push(stream.alloc_zeros::<f32>(n)?);
    }

    let mut reduce_offsets = Vec::with_capacity(param_sizes.len());
    let mut total_reduce_blocks = 0usize;
    let mut max_reduce_blocks = 1usize;
    for &n in param_sizes {
        let blocks = ((n + 255) / 256).max(1);
        reduce_offsets.push(total_reduce_blocks);
        total_reduce_blocks = total_reduce_blocks.saturating_add(blocks);
        max_reduce_blocks = max_reduce_blocks.max(blocks);
    }
    let scratch_len = total_reduce_blocks
        .max(max_reduce_blocks.saturating_mul(2))
        .max(2);
    let reduce_scratch = stream.alloc_zeros::<f32>(scratch_len)?;

    Ok(Box::new(S3State {
        momentum,
        power_momentum,
        updates,
        reduce_scratch,
        reduce_offsets,
        max_reduce_blocks,
        groups: infer_param_groups(param_sizes.len()),
        step: 0,
        profile,
        best_val_loss: f32::INFINITY,
        best_val_epoch: 0,
        best_train_loss: f32::INFINITY,
        headroom_allowed: true,
        progress_credit: 1.0,
        prev_grad,
        kernels,
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
    train_loss: Option<f32>,
    val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state
        .as_any_mut()
        .downcast_mut::<S3State>()
        .ok_or_else(|| anyhow!("Invalid S3 optimizer state"))?;

    update_dual_loss_policy(state, epoch, train_loss, val_loss);

    state.step += 1;
    let lr = state.profile.lr * learning_rate_scale(state.step, &state.profile);

    let base_kernel = state.kernels.base_update.clone();
    let fused_hidden_kernel = state.kernels.fused_dual_loss_update.clone();
    let base_sam_kernel = state.kernels.base_update_sam.clone();
    let fused_hidden_sam_row_trust_kernel =
        state.kernels.fused_dual_loss_update_sam_row_trust.clone();
    let reduce_kernel = state.kernels.reduce_sum_sq.clone();
    let reduce_dual_kernel = state.kernels.reduce_sum_sq_dual.clone();
    let scale_kernel = state.kernels.scale_updates.clone();
    let row_trust_kernel = state.kernels.row_trust_scale.clone();
    let _ = module;

    let headroom_flag: f32 = if state.headroom_allowed { 1.0 } else { 0.0 };
    let progress_credit = state.progress_credit.clamp(0.0, 1.0);
    let coherence_gain = state.profile.coherence_gain;
    let apply_corr: f32 = if state.step > 1 { 1.0 } else { 0.0 };

    #[derive(Clone, Copy)]
    struct SamReduceSlot {
        idx: usize,
        offset: usize,
        grid: u32,
    }
    let mut sam_slots: Vec<SamReduceSlot> = Vec::with_capacity(gradients.len());
    let mut g_norms: Vec<f32> = vec![0.0; gradients.len()];

    for i in 0..gradients.len() {
        let n = gradients[i].len();
        if n == 0 {
            continue;
        }
        let group = state.groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            continue;
        }
        let apply_sam = (group == GROUP_HIDDEN_WEIGHT || group == GROUP_OUTPUT_WEIGHT) && lr > 0.0;
        if apply_sam {
            let offset = state.reduce_offsets.get(i).copied().unwrap_or(0);
            let grid = launch_reduce_sum_sq_at(
                &stream,
                &reduce_kernel,
                &gradients[i],
                &mut state.reduce_scratch,
                offset,
                n,
            )?;
            sam_slots.push(SamReduceSlot { idx: i, offset, grid });
        }
    }

    if !sam_slots.is_empty() {
        let need = sam_slots
            .iter()
            .map(|s| s.offset + s.grid as usize)
            .max()
            .unwrap_or(0)
            .min(state.reduce_scratch.len());
        let partials: Vec<f32> = stream.memcpy_dtov(&state.reduce_scratch.slice(0..need))?;
        for slot in &sam_slots {
            let mut sum = 0.0f32;
            let end = slot.offset + slot.grid as usize;
            for j in slot.offset..end.min(partials.len()) {
                sum += partials[j];
            }
            g_norms[slot.idx] = sum.sqrt();
        }
    }

    let mut fused_row_trust_done = vec![false; gradients.len()];
    for i in 0..gradients.len() {
        let n = gradients[i].len();
        if n == 0 {
            continue;
        }
        let group = state.groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            continue;
        }

        let launch = launch_config(n);
        let wd = decay_for_group(group, &state.profile);
        let use_fused_hidden = group == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS;
        let apply_sam = (group == GROUP_HIDDEN_WEIGHT || group == GROUP_OUTPUT_WEIGHT) && lr > 0.0;

        if apply_sam {
            let g_norm = g_norms[i];
            unsafe {
                if use_fused_hidden {
                    let rows = 256u32;
                    let cols = 256u32;
                    let block = 256u32;
                    let shared_bytes =
                        (((block / 32) * 2) * std::mem::size_of::<f32>() as u32).max(64);
                    stream
                        .launch_builder(&fused_hidden_sam_row_trust_kernel)
                        .arg(&gradients[i])
                        .arg(&model_params[i])
                        .arg(&mut state.momentum[i])
                        .arg(&mut state.power_momentum[i])
                        .arg(&mut state.updates[i])
                        .arg(&mut state.prev_grad[i])
                        .arg(&rows)
                        .arg(&cols)
                        .arg(&lr)
                        .arg(&state.profile.beta)
                        .arg(&state.profile.eps)
                        .arg(&wd)
                        .arg(&coherence_gain)
                        .arg(&progress_credit)
                        .arg(&headroom_flag)
                        .arg(&g_norm)
                        .arg(&apply_corr)
                        .arg(&state.profile.arch_mantissa_bits)
                        .launch(LaunchConfig {
                            grid_dim: (rows, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        })?;
                    fused_row_trust_done[i] = true;
                } else {
                    stream
                        .launch_builder(&base_sam_kernel)
                        .arg(&gradients[i])
                        .arg(&model_params[i])
                        .arg(&mut state.momentum[i])
                        .arg(&mut state.power_momentum[i])
                        .arg(&mut state.updates[i])
                        .arg(&mut state.prev_grad[i])
                        .arg(&(n as u32))
                        .arg(&lr)
                        .arg(&state.profile.beta)
                        .arg(&state.profile.eps)
                        .arg(&wd)
                        .arg(&g_norm)
                        .arg(&apply_corr)
                        .arg(&state.profile.arch_mantissa_bits)
                        .launch(launch)?;
                }
            }
        } else {
            unsafe {
                if use_fused_hidden {
                    stream
                        .launch_builder(&fused_hidden_kernel)
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
                        .arg(&coherence_gain)
                        .arg(&progress_credit)
                        .arg(&headroom_flag)
                        .arg(&state.profile.arch_mantissa_bits)
                        .launch(launch)?;
                } else {
                    stream
                        .launch_builder(&base_kernel)
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
        }
    }

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
        let apply_trust = group == GROUP_HIDDEN_WEIGHT || group == GROUP_OUTPUT_WEIGHT;
        if apply_trust && lr > 0.0 && !fused_row_trust_done.get(i).copied().unwrap_or(false) {
            if group == GROUP_HIDDEN_WEIGHT && n == FULL_HIDDEN_ELEMENTS {
                let rows = 256u32;
                let cols = 256u32;
                let block = 256u32;
                let shared_bytes = (((block / 32) * 2) * std::mem::size_of::<f32>() as u32).max(64);
                unsafe {
                    stream
                        .launch_builder(&row_trust_kernel)
                        .arg(&model_params[i])
                        .arg(&mut state.updates[i])
                        .arg(&rows)
                        .arg(&cols)
                        .arg(&lr)
                        .arg(&state.profile.eps)
                        .arg(&state.profile.arch_mantissa_bits)
                        .launch(LaunchConfig {
                            grid_dim: (rows, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        })?;
                }
            } else {
                let (p_sq, u_sq) = reduce_sum_sq_dual(
                    &stream,
                    &reduce_dual_kernel,
                    &model_params[i],
                    &state.updates[i],
                    &mut state.reduce_scratch,
                    state.max_reduce_blocks,
                    n,
                )?;
                let p_norm = p_sq.sqrt();
                let u_norm = u_sq.sqrt();
                if p_norm > 0.0 && u_norm > 0.0 {
                    let v_norm = u_norm / lr;
                    let trust = p_norm / (v_norm + state.profile.eps);
                    unsafe {
                        stream
                            .launch_builder(&scale_kernel)
                            .arg(&mut state.updates[i])
                            .arg(&(n as u32))
                            .arg(&trust)
                            .arg(&state.profile.arch_mantissa_bits)
                            .launch(launch)?;
                    }
                }
            }
        }

        result.push(state.updates[i].clone());
    }

    Ok(result)
}

fn launch_reduce_sum_sq_at(
    stream: &Arc<CudaStream>,
    reduce_kernel: &CudaFunction,
    data: &CudaSlice<f32>,
    scratch: &mut CudaSlice<f32>,
    offset: usize,
    n: usize,
) -> Result<u32> {
    if n == 0 {
        return Ok(0);
    }
    let block = 256u32;
    let grid = ((n as u32 + block - 1) / block).max(1);
    let shared_bytes = ((block / 32) * std::mem::size_of::<f32>() as u32).max(32);
    let n_u32 = n as u32;
    let out_offset = offset as u32;
    unsafe {
        stream
            .launch_builder(reduce_kernel)
            .arg(data)
            .arg(&mut *scratch)
            .arg(&n_u32)
            .arg(&out_offset)
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared_bytes,
            })?;
    }
    Ok(grid)
}

fn reduce_sum_sq_dual(
    stream: &Arc<CudaStream>,
    reduce_dual_kernel: &CudaFunction,
    data_a: &CudaSlice<f32>,
    data_b: &CudaSlice<f32>,
    scratch: &mut CudaSlice<f32>,
    max_reduce_blocks: usize,
    n: usize,
) -> Result<(f32, f32)> {
    if n == 0 {
        return Ok((0.0, 0.0));
    }
    let block = 256u32;
    let grid = ((n as u32 + block - 1) / block).max(1);
    let shared_bytes = ((block / 32) * 2 * std::mem::size_of::<f32>() as u32).max(64);
    let n_u32 = n as u32;
    let partial_stride = max_reduce_blocks as u32;
    unsafe {
        stream
            .launch_builder(reduce_dual_kernel)
            .arg(data_a)
            .arg(data_b)
            .arg(&mut *scratch)
            .arg(&n_u32)
            .arg(&partial_stride)
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared_bytes,
            })?;
    }
    let need = (max_reduce_blocks + grid as usize).min(scratch.len());
    let partials: Vec<f32> = stream.memcpy_dtov(&scratch.slice(0..need))?;
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    let g = grid as usize;
    let off = max_reduce_blocks;
    for i in 0..g {
        sum_a += partials[i];
        sum_b += partials[off + i];
    }
    Ok((sum_a, sum_b))
}

#[inline]
fn update_dual_loss_policy(
    state: &mut S3State,
    epoch: usize,
    train_loss: Option<f32>,
    val_loss: Option<f32>,
) {
    let bits = state.profile.arch_mantissa_bits;
    let t = train_loss
        .filter(|x| x.is_finite())
        .map(|x| canonicalize_f32(x, bits));
    let v = val_loss
        .filter(|x| x.is_finite())
        .map(|x| canonicalize_f32(x, bits));

    let mut val_improved = false;
    if let Some(vv) = v {
        if vv < state.best_val_loss {
            state.best_val_loss = vv;
            state.best_val_epoch = epoch;
            val_improved = true;
            state.headroom_allowed = true;
        }
    }

    if let Some(tt) = t {
        if tt < state.best_train_loss {
            state.best_train_loss = tt;
            if !val_improved {
                state.headroom_allowed = false;
            }
        }
    }

    if v.is_some() {
        let horizon = state.profile.progress_horizon_epochs.max(1);
        let stale = epoch.saturating_sub(state.best_val_epoch);
        let x = (stale as f32 / horizon as f32).clamp(0.0, 1.0);
        state.progress_credit = 0.5 * (1.0 + (std::f32::consts::PI * x).cos());
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

