use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

// ─── Compile-time defaults ─────────────────────────────────────────────────────
// LR_MAX 3.4e-3 dead-listed for n_hidden=18 (i2 -4.3% Q). 2.8e-3 = top of safe range.
const LR_MAX_DEFAULT: f32 = 2.8e-3;
const LR_MIN: f32 = 2e-5;
const WARMUP_EPOCHS: usize = 8;
const T_MAX_DEFAULT: usize = 650;
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

// Runtime HP overrides: set in solve() from hp_json before calling training_loop.
// Thread-local safe: TIG runs single-threaded per nonce.
use std::cell::Cell;
thread_local! {
    static RT_LR_MAX:    Cell<f32>   = Cell::new(LR_MAX_DEFAULT);
    static RT_T_MAX:     Cell<usize> = Cell::new(T_MAX_DEFAULT);
    static RT_GHW_SCALE: Cell<f32>   = Cell::new(1.0);
}

// ─── Local state ─────────────────────────────────────────────────────────────
#[derive(Clone)]
struct AdaBeliefState {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    s: Vec<CudaSlice<f32>>,
    prev_grad: Vec<CudaSlice<f32>>,
    param_groups: Vec<i32>,
    step: usize,
    best_val_loss: f32,
    plateau_epochs: usize,
    lr_scale: f32,
    last_sched_epoch: usize,
}

impl OptimizerStateTrait for AdaBeliefState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

fn schedule_lr(epoch: usize) -> f32 {
    let lr_max = RT_LR_MAX.with(|c| c.get());
    let t_max  = RT_T_MAX.with(|c| c.get());
    if epoch < WARMUP_EPOCHS {
        lr_max * ((epoch as f32) + 1.0) / (WARMUP_EPOCHS as f32).max(1.0)
    } else {
        let denom = (t_max.saturating_sub(WARMUP_EPOCHS)).max(1) as f32;
        let p = (((epoch - WARMUP_EPOCHS) as f32) / denom).min(1.0);
        LR_MIN + 0.5 * (lr_max - LR_MIN) * (1.0 + (std::f32::consts::PI * p).cos())
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
    let ghw_scale = RT_GHW_SCALE.with(|c| c.get());
    match group {
        GROUP_HIDDEN_WEIGHT => (base_lr * ghw_scale, HIDDEN_WEIGHT_DECAY),
        GROUP_HIDDEN_BIAS   => (base_lr * 1.5, 0.0),
        GROUP_OUTPUT_WEIGHT => (base_lr * 1.75, OUTPUT_WEIGHT_DECAY),
        GROUP_OUTPUT_BIAS   => (base_lr * 2.0, 0.0),
        GROUP_BN_WEIGHT     => (base_lr * 0.55, 0.0),
        GROUP_BN_BIAS       => (base_lr * 0.8, 0.0),
        GROUP_RUNNING_STAT  => (0.0, 0.0),
        _                   => (base_lr * ghw_scale, HIDDEN_WEIGHT_DECAY),
    }
}

// ─── Hooks ────────────────────────────────────────────────────────────────────
fn optimizer_init(
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
    Ok(Box::new(AdaBeliefState {
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

fn optimizer_query(
    _state: &dyn OptimizerStateTrait,
    _params: &[CudaSlice<f32>],
    _epoch: usize,
    _train: Option<f32>,
    _val: Option<f32>,
    _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    Ok(None)
}

// P3 fused kernel: 1 launch/step replacing n_fused separate launches.

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
        .downcast_mut::<AdaBeliefState>()
        .ok_or_else(|| anyhow!("Invalid optimizer state for AdaBeliefState"))?;

    state.step += 1;
    if state.last_sched_epoch != epoch {
        state.last_sched_epoch = epoch;
        if let Some(vl) = val_loss.filter(|vl| vl.is_finite()) {
            if vl + VAL_IMPROVEMENT_EPS < state.best_val_loss {
                state.best_val_loss = vl;
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

    let first_step: i32 = if state.step == 1 { 1 } else { 0 };
    let scheduled_lr = schedule_lr(epoch);
    let lr = LR_MIN + (scheduled_lr - LR_MIN) * state.lr_scale;
    let eps = EPS;

    let n_params = gradients.len();

    // Separate RUNNING_STAT tensors (zeros) from active tensors (fused kernel).
    let mut updates: Vec<Option<CudaSlice<f32>>> = (0..n_params).map(|_| None).collect();

    // h_all_ptrs layout: [grad0..gradN, param0..paramN, m0..mN, v0..vN, s0..sN, pg0..pgN, delta0..deltaN]
    let mut h_all_ptrs: Vec<u64> = Vec::new();
    let mut h_lr_wd:    Vec<f32> = Vec::new();
    let mut h_offsets:  Vec<i32> = vec![0i32];
    let mut delta_ptrs_tmp: Vec<u64> = Vec::new();
    let mut n_fused = 0usize;

    for i in 0..n_params {
        let n = gradients[i].len();
        let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            updates[i] = Some(stream.alloc_zeros::<f32>(n)?);
            continue;
        }

        let (g_ptr, _sg)   = gradients[i].device_ptr(&stream);
        let (p_ptr, _sp)   = model_params[i].device_ptr(&stream);
        let (m_ptr, _sm)   = state.m[i].device_ptr(&stream);
        let (v_ptr, _sv)   = state.v[i].device_ptr(&stream);
        let (s_ptr, _ss)   = state.s[i].device_ptr(&stream);
        let (pg_ptr, _spg) = state.prev_grad[i].device_ptr(&stream);

        // delta is fully overwritten — skip memset
        let delta = unsafe { stream.alloc::<f32>(n) }?;
        let delta_raw: u64 = { let (p, _s) = delta.device_ptr(&stream); p };
        updates[i] = Some(delta);

        // Accumulate interleaved per-tensor, reorder to role-segmented after loop
        h_all_ptrs.push(g_ptr);
        h_all_ptrs.push(p_ptr);
        h_all_ptrs.push(m_ptr);
        h_all_ptrs.push(v_ptr);
        h_all_ptrs.push(s_ptr);
        h_all_ptrs.push(pg_ptr);
        delta_ptrs_tmp.push(delta_raw);

        let (lr_i, wd_i) = group_hyperparams(group, lr);
        h_lr_wd.push(lr_i);
        h_lr_wd.push(wd_i);

        let prev_off = *h_offsets.last().unwrap();
        h_offsets.push(prev_off + n as i32);
        n_fused += 1;
    }

    if n_fused == 0 {
        return Ok(updates.into_iter().map(|o| o.unwrap()).collect());
    }

    // Reorder from interleaved [g0,p0,m0,v0,s0,pg0, g1,...] to
    // role-segmented [g0..gN, p0..pN, m0..mN, v0..vN, s0..sN, pg0..pgN] + deltas (7th role)
    let mut final_ptrs: Vec<u64> = Vec::with_capacity(7 * n_fused);
    for role in 0..6usize {
        for t in 0..n_fused {
            final_ptrs.push(h_all_ptrs[t * 6 + role]);
        }
    }
    final_ptrs.extend_from_slice(&delta_ptrs_tmp);

    let total_elems = *h_offsets.last().unwrap();

    // 3 H2D transfers instead of n_fused separate kernel launches
    let d_all_ptrs = stream.memcpy_stod(&final_ptrs)?;
    let d_lr_wd    = stream.memcpy_stod(&h_lr_wd)?;
    let d_offsets  = stream.memcpy_stod(&h_offsets)?;

    let fused_kernel = module.load_function("adabelief_fused_kernel_t28")?;
    let total_i   = total_elems;
    let n_fused_i = n_fused as i32;
    let block_dim = 256u32;
    let grid_dim  = (total_elems as u32 + block_dim - 1) / block_dim;
    // Shared memory: (n_fused + 1) i32 for offset table used in binary search
    let smem_bytes = (n_fused + 1) as u32 * 4;
    let cfg = LaunchConfig {
        block_dim: (block_dim, 1, 1),
        grid_dim:  (grid_dim, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    unsafe {
        stream.launch_builder(&fused_kernel)
            .arg(&d_all_ptrs)
            .arg(&d_lr_wd)
            .arg(&d_offsets)
            .arg(&eps)
            .arg(&first_step)
            .arg(&total_i)
            .arg(&n_fused_i)
            .launch(cfg)?;
    }

    Ok(updates.into_iter().map(|o| o.unwrap()).collect())
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    
    let lr_max: f32 = hyperparameters.as_ref()
        .and_then(|m| m.get("lr_max"))
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(LR_MAX_DEFAULT);
    let t_max: usize = hyperparameters.as_ref()
        .and_then(|m| m.get("t_max_epochs"))
        .and_then(|v| v.as_u64())
        .map(|u| u as usize)
        .unwrap_or(T_MAX_DEFAULT);
    let ghw_scale: f32 = hyperparameters.as_ref()
        .and_then(|m| m.get("ghw_scale"))
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(1.0);

    RT_LR_MAX.with(|c| c.set(lr_max));
    RT_T_MAX.with(|c| c.set(t_max));
    RT_GHW_SCALE.with(|c| c.set(ghw_scale));

    training_loop(challenge, save_solution, module, stream, prop, optimizer_init, optimizer_query, optimizer_step)?;
    Ok(())
}
