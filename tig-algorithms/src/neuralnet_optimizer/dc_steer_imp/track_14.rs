use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

use super::helpers::dc;

thread_local! {
    static DC: std::cell::RefCell<dc::DcState> = std::cell::RefCell::new(dc::DcState::default());
}

const DC_ENABLE_DEFAULT_T27: u64 = 1;
const DC_AXPY_T27: &str = "dc_axpy_t27";

static EMA_GAMMA_BITS:        AtomicU32 = AtomicU32::new(0);
static EMA_BETA_BITS:         AtomicU32 = AtomicU32::new(0);
static MUON_LR_SCALE_BITS:    AtomicU32 = AtomicU32::new(0);
static HIDDEN_BIAS_SCALE_BITS: AtomicU32 = AtomicU32::new(0);
static QH_NU_BITS: AtomicU32 = AtomicU32::new(0);
static AB_BLEND_BITS: AtomicU32 = AtomicU32::new(0);
static SOPHIA_RHO_BITS: AtomicU32 = AtomicU32::new(0);
static T_MAX_EPOCHS_OVERRIDE: AtomicU32 = AtomicU32::new(850);
static MIN_LR_SCALE_BITS: AtomicU32 = AtomicU32::new(0);       
static PLATEAU_PATIENCE_OVERRIDE: AtomicU32 = AtomicU32::new(0); 
static MARS_GAMMA_BITS: AtomicU32 = AtomicU32::new(0); 
static MARS_CLIP_SCALE_BITS: AtomicU32 = AtomicU32::new(0x3f800000); 

const EMA_GAMMA_DEFAULT: f32 = 0.999;
const EMA_BETA_DEFAULT:  f32 = 0.8;
const QH_NU_DEFAULT:     f32 = 1.0;
const AB_BLEND_DEFAULT:  f32 = 1.0;

const LR_MAX: f32 = 3.4e-3;
const LR_MIN: f32 = 2e-5;
const WARMUP_EPOCHS: usize = 8;
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
const NS_MATRIX_SIZE: usize = 65536; 

#[derive(Clone)]
struct PrometheusState {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    s: Vec<CudaSlice<f32>>,
    prev_grad: Vec<CudaSlice<f32>>,
    ema: Vec<CudaSlice<f32>>, 
    envelope: Vec<CudaSlice<f32>>, 
    prev_update: Vec<CudaSlice<f32>>, 
    param_groups: Vec<i32>,
    step: usize,
    best_val_loss: f32,
    plateau_epochs: usize,
    lr_scale: f32,
    last_sched_epoch: usize,
    plateau_routing_epoch: bool,
    muon_ema: Vec<CudaSlice<f32>>, 
    ns_buf0: CudaSlice<f32>, 
    ns_buf1: CudaSlice<f32>, 
    ns_x:    CudaSlice<f32>, 
    ns_x2:   CudaSlice<f32>, 
    ns_poly: CudaSlice<f32>, 
    grokfast_mu:    Vec<CudaSlice<f32>>, 
    grokfast_g_aug: Vec<CudaSlice<f32>>, 
    dc_scratch:     Vec<CudaSlice<f32>>,
    dc_last_epoch:  usize,
    dc_sie:         usize,
}

impl OptimizerStateTrait for PrometheusState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

fn schedule_lr(epoch: usize) -> f32 {
    let t_max = T_MAX_EPOCHS_OVERRIDE.load(Ordering::Relaxed) as usize;
    if epoch < WARMUP_EPOCHS {
        LR_MAX * ((epoch as f32) + 1.0) / (WARMUP_EPOCHS as f32).max(1.0)
    } else {
        let denom = (t_max.saturating_sub(WARMUP_EPOCHS)).max(1) as f32;
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
    let hidden_bias_scale = f32::from_bits(HIDDEN_BIAS_SCALE_BITS.load(Ordering::Relaxed));
    match group {
        GROUP_HIDDEN_WEIGHT => (base_lr, HIDDEN_WEIGHT_DECAY),
        GROUP_HIDDEN_BIAS   => (base_lr * hidden_bias_scale, 0.0),
        GROUP_OUTPUT_WEIGHT => (base_lr * 1.75, OUTPUT_WEIGHT_DECAY),
        GROUP_OUTPUT_BIAS   => (base_lr * 2.0, 0.0),
        GROUP_BN_WEIGHT     => (base_lr * 0.55, 0.0),
        GROUP_BN_BIAS       => (base_lr * 0.8, 0.0),
        GROUP_RUNNING_STAT  => (0.0, 0.0),
        _                   => (base_lr, HIDDEN_WEIGHT_DECAY),
    }
}

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
    let mut ema = Vec::with_capacity(param_sizes.len());
    let mut envelope = Vec::with_capacity(param_sizes.len());
    let mut prev_update = Vec::with_capacity(param_sizes.len());
    for &sz in param_sizes {
        m.push(stream.alloc_zeros::<f32>(sz)?);
        v.push(stream.alloc_zeros::<f32>(sz)?);
        s.push(stream.alloc_zeros::<f32>(sz)?);
        prev_grad.push(stream.alloc_zeros::<f32>(sz)?);
        ema.push(stream.alloc_zeros::<f32>(sz)?);
        envelope.push(stream.alloc_zeros::<f32>(sz)?);
        prev_update.push(stream.alloc_zeros::<f32>(sz)?);
    }

    let pg_tmp = infer_param_groups(param_sizes.len());
    let n_hw_sq = param_sizes.iter().zip(pg_tmp.iter())
        .filter(|(&sz, &g)| g == GROUP_HIDDEN_WEIGHT && sz == NS_MATRIX_SIZE)
        .count();

    let mut muon_ema: Vec<CudaSlice<f32>> = Vec::with_capacity(n_hw_sq);
    for _ in 0..n_hw_sq {
        muon_ema.push(stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?);
    }

    let ns_buf0 = stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?;
    let ns_buf1 = stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?;
    let ns_x    = stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?;
    let ns_x2   = stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?;
    let ns_poly = stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?;

    let mut grokfast_mu:    Vec<CudaSlice<f32>> = Vec::with_capacity(param_sizes.len());
    let mut grokfast_g_aug: Vec<CudaSlice<f32>> = Vec::with_capacity(param_sizes.len());
    for &sz in param_sizes {
        grokfast_mu.push(stream.alloc_zeros::<f32>(sz)?);
        grokfast_g_aug.push(stream.alloc_zeros::<f32>(sz)?);
    }

    Ok(Box::new(PrometheusState {
        m,
        v,
        s,
        prev_grad,
        ema,
        envelope,
        prev_update,
        param_groups: pg_tmp,
        step: 0,
        best_val_loss: f32::INFINITY,
        plateau_epochs: 0,
        lr_scale: 1.0,
        last_sched_epoch: usize::MAX,
        plateau_routing_epoch: false,
        muon_ema,
        ns_buf0,
        ns_buf1,
        ns_x,
        ns_x2,
        ns_poly,
        grokfast_mu,
        grokfast_g_aug,
        dc_scratch: Vec::new(),
        dc_last_epoch: usize::MAX,
        dc_sie: 0,
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
        .downcast_mut::<PrometheusState>()
        .ok_or_else(|| anyhow!("Invalid optimizer state for PrometheusState"))?;

    state.step += 1;
    if state.dc_last_epoch != epoch {
        state.dc_sie = 0;
        state.dc_last_epoch = epoch;
    }
    state.dc_sie = state.dc_sie.saturating_add(1);
    if state.last_sched_epoch != epoch {
        state.last_sched_epoch = epoch;
        state.plateau_routing_epoch = false;
        let min_lr_scale_bits = MIN_LR_SCALE_BITS.load(Ordering::Relaxed);
        let eff_min_lr_scale = if min_lr_scale_bits == 0 { MIN_LR_SCALE }
                               else { f32::from_bits(min_lr_scale_bits) };
        let patience_raw = PLATEAU_PATIENCE_OVERRIDE.load(Ordering::Relaxed);
        let eff_patience = if patience_raw == 0 { PLATEAU_PATIENCE } else { patience_raw as usize };
        if let Some(vl) = val_loss.filter(|vl| vl.is_finite()) {
            if vl + VAL_IMPROVEMENT_EPS < state.best_val_loss {
                state.best_val_loss = vl;
                state.plateau_epochs = 0;
                state.lr_scale = (state.lr_scale * 1.03).min(1.0);
            } else {
                state.plateau_routing_epoch = true;
                state.plateau_epochs += 1;
                if state.plateau_epochs >= eff_patience {
                    state.lr_scale = (state.lr_scale * PLATEAU_DECAY).max(eff_min_lr_scale);
                    state.plateau_epochs = 0;
                }
            }
        }
    }

    let first_step: i32 = if state.step == 1 { 1 } else { 0 };
    let scheduled_lr = schedule_lr(epoch);
    let lr = LR_MIN + (scheduled_lr - LR_MIN) * state.lr_scale;
    let eps = EPS;

    let ema_gamma = f32::from_bits(EMA_GAMMA_BITS.load(Ordering::Relaxed));
    let nesterov_beta = f32::from_bits(EMA_BETA_BITS.load(Ordering::Relaxed));
    let lr_frac = (lr / LR_MAX).clamp(0.0, 1.0);
    let ema_beta: f32 = nesterov_beta * lr_frac;

    let muon_lr_scale = f32::from_bits(MUON_LR_SCALE_BITS.load(Ordering::Relaxed));
    let qh_nu = f32::from_bits(QH_NU_BITS.load(Ordering::Relaxed));
    let ab_blend = f32::from_bits(AB_BLEND_BITS.load(Ordering::Relaxed));
    let sophia_rho = f32::from_bits(SOPHIA_RHO_BITS.load(Ordering::Relaxed));

    let mars_gamma = f32::from_bits(MARS_GAMMA_BITS.load(Ordering::Relaxed));
    let mars_clip_scale = f32::from_bits(MARS_CLIP_SCALE_BITS.load(Ordering::Relaxed));

    let n_params = gradients.len();

    if mars_gamma > 1e-7 {
        let drift_kern = module.load_function("mars_tensor_drift_stats_kernel")?;
        let mars_kern = module.load_function("mars_correction_kernel")?;
        let block_dim = 256u32;
        let mut drift_stats: Vec<Option<CudaSlice<f32>>> =
            (0..n_params).map(|_| None).collect();
        let mut mars_routes = vec![0i32; n_params];

        unsafe {
            for i in 0..n_params {
                let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
                if group == GROUP_RUNNING_STAT { continue; }
                let n_i = gradients[i].len() as i32;
                let stats = stream.alloc_zeros::<f32>(3)?;
                let (raw_g, _sg) = gradients[i].device_ptr(&stream);
                let (gprev_p, _sp) = state.grokfast_mu[i].device_ptr(&stream);
                let (stats_p, _ss) = stats.device_ptr(&stream);
                stream.launch_builder(&drift_kern)
                    .arg(&raw_g)
                    .arg(&gprev_p)
                    .arg(&stats_p)
                    .arg(&n_i)
                    .launch(LaunchConfig {
                        block_dim: (block_dim, 1, 1),
                        grid_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                drop(_ss);
                drift_stats[i] = Some(stats);
            }

            for i in 0..n_params {
                if let Some(stats) = drift_stats[i].as_ref() {
                    let h: Vec<f32> = stream.memcpy_dtov(stats)?;
                    let cosine_sign = h[2] / (h[0].sqrt() * h[1].sqrt());
                    mars_routes[i] =
                        if h[0] > 0.0 && h[1] > 0.0 && cosine_sign > 0.0 { 1 } else { 0 };
                }
            }

            for i in 0..n_params {
                let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
                if group == GROUP_RUNNING_STAT { continue; }
                let n_i = gradients[i].len() as i32;
                let grid_dim = (n_i as u32 + block_dim - 1) / block_dim;
                let cfg_mars = LaunchConfig {
                    block_dim: (block_dim, 1, 1),
                    grid_dim: (grid_dim, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (raw_g, _sg)   = gradients[i].device_ptr(&stream);
                let (gprev_p, _sp) = state.grokfast_mu[i].device_ptr(&stream);
                let (c_p, _sc)     = state.grokfast_g_aug[i].device_ptr(&stream);
                let route = mars_routes[i];
                stream.launch_builder(&mars_kern)
                    .arg(&raw_g)
                    .arg(&gprev_p)
                    .arg(&c_p)
                    .arg(&mars_gamma)
                    .arg(&mars_clip_scale)
                    .arg(&route)
                    .arg(&first_step)
                    .arg(&n_i)
                    .launch(cfg_mars)?;
            }
        }
    }

    let mut updates: Vec<Option<CudaSlice<f32>>> = (0..n_params).map(|_| None).collect();

    let mut h_all_ptrs: Vec<u64> = Vec::new();
    let mut h_lr_wd:    Vec<f32> = Vec::new();
    let mut h_offsets:  Vec<i32> = vec![0i32];
    let mut h_geometry_mask: Vec<i32> = Vec::new();
    let mut h_rows: Vec<i32> = Vec::new();
    let mut h_column_offsets: Vec<i32> = vec![0i32];

    let mut delta_ptrs_tmp: Vec<u64> = Vec::new();
    let mut ema_ptrs_tmp:   Vec<u64> = Vec::new();
    let mut envelope_ptrs_tmp: Vec<u64> = Vec::new();
    let mut prev_update_ptrs_tmp: Vec<u64> = Vec::new();
    let mut n_fused = 0usize;

    for i in 0..n_params {
        let n = gradients[i].len();
        let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
        if group == GROUP_RUNNING_STAT {
            updates[i] = Some(stream.alloc_zeros::<f32>(n)?);
            continue;
        }

        let (g_ptr, _sg) = if mars_gamma > 1e-7 {
            state.grokfast_g_aug[i].device_ptr(&stream)
        } else {
            gradients[i].device_ptr(&stream)
        };
        let (p_ptr, _sp)     = model_params[i].device_ptr(&stream);
        let (m_ptr, _sm)     = state.m[i].device_ptr(&stream);
        let (v_ptr, _sv)     = state.v[i].device_ptr(&stream);
        let (s_ptr, _ss)     = state.s[i].device_ptr(&stream);
        let (pg_ptr, _spg)   = state.prev_grad[i].device_ptr(&stream);
        let (ema_ptr, _sema) = state.ema[i].device_ptr(&stream);
        let (env_ptr, _senv) = state.envelope[i].device_ptr(&stream);
        let (pu_ptr, _spu)   = state.prev_update[i].device_ptr(&stream);

        let delta = unsafe { stream.alloc::<f32>(n) }?;
        let delta_raw: u64 = { let (p, _s) = delta.device_ptr(&stream); p };
        updates[i] = Some(delta);

        h_all_ptrs.push(g_ptr);
        h_all_ptrs.push(p_ptr);
        h_all_ptrs.push(m_ptr);
        h_all_ptrs.push(v_ptr);
        h_all_ptrs.push(s_ptr);
        h_all_ptrs.push(pg_ptr);
        ema_ptrs_tmp.push(ema_ptr);
        delta_ptrs_tmp.push(delta_raw);
        envelope_ptrs_tmp.push(env_ptr);
        prev_update_ptrs_tmp.push(pu_ptr);

        let (lr_i, wd_i) = group_hyperparams(group, lr);
        h_lr_wd.push(lr_i);
        h_lr_wd.push(wd_i);
        h_geometry_mask.push(if group == GROUP_HIDDEN_WEIGHT && n == NS_MATRIX_SIZE { 1 } else { 0 });

        let inferred_rows = match group {
            GROUP_HIDDEN_WEIGHT
                if state.param_groups.get(i + 1).copied() == Some(GROUP_HIDDEN_BIAS) =>
            {
                gradients.get(i + 1).map(|bias| bias.len()).unwrap_or(0)
            }
            GROUP_OUTPUT_WEIGHT
                if state.param_groups.get(i + 1).copied() == Some(GROUP_OUTPUT_BIAS) =>
            {
                gradients.get(i + 1).map(|bias| bias.len()).unwrap_or(0)
            }
            _ => 0,
        };
        let row_count = if inferred_rows > 0 && n % inferred_rows == 0 {
            inferred_rows
        } else {
            0
        };
        h_rows.push(row_count as i32);

        let prev_off = *h_offsets.last().unwrap();
        h_offsets.push(prev_off + n as i32);
        let width = if row_count > 0 { n / row_count } else { 0 };
        let prev_column_off = *h_column_offsets.last().unwrap();
        h_column_offsets.push(prev_column_off + width as i32);
        n_fused += 1;
    }

    if n_fused == 0 {
        return Ok(updates.into_iter().map(|o| o.unwrap()).collect());
    }

    let mut final_ptrs: Vec<u64> = Vec::with_capacity(10 * n_fused);
    for role in 0..6usize {
        for t in 0..n_fused {
            final_ptrs.push(h_all_ptrs[t * 6 + role]);
        }
    }
    final_ptrs.extend_from_slice(&ema_ptrs_tmp);
    final_ptrs.extend_from_slice(&delta_ptrs_tmp);
    final_ptrs.extend_from_slice(&envelope_ptrs_tmp);
    final_ptrs.extend_from_slice(&prev_update_ptrs_tmp);

    let total_elems = *h_offsets.last().unwrap();

    let d_all_ptrs = stream.memcpy_stod(&final_ptrs)?;
    let d_lr_wd    = stream.memcpy_stod(&h_lr_wd)?;
    let d_offsets  = stream.memcpy_stod(&h_offsets)?;
    let d_geometry_mask = stream.memcpy_stod(&h_geometry_mask)?;
    let d_rows = stream.memcpy_stod(&h_rows)?;
    let d_column_offsets = stream.memcpy_stod(&h_column_offsets)?;
    let total_columns = *h_column_offsets.last().unwrap();

    let fused_kernel = module.load_function("adabelief_fused_kernel_t27")?;
    let matrix_energy_kernel = module.load_function("hidden_matrix_biaxial_energy_t27")?;
    let matrix_balance_kernel = module.load_function("hidden_matrix_biaxial_balance_t27")?;
    let total_i   = total_elems;
    let n_fused_i = n_fused as i32;
    let block_dim = 256u32;
    let grid_dim  = (total_elems as u32 + block_dim - 1) / block_dim;
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
            .arg(&ema_gamma)
            .arg(&ema_beta)
            .arg(&qh_nu)
            .arg(&ab_blend)
            .arg(&sophia_rho)
            .launch(cfg)?;
    }

    if state.plateau_routing_epoch {
        let agreement_kernel = module.load_function("ema_tensor_agreement_stats_t27")?;
        let route_kernel = module.load_function("ema_tensor_conservative_route_t27")?;
        let agreement_stats = stream.alloc_zeros::<f32>(n_fused * 3)?;
        let (agreement_p, _sa) = agreement_stats.device_ptr(&stream);
        unsafe {
            stream.launch_builder(&agreement_kernel)
                .arg(&d_all_ptrs)
                .arg(&d_offsets)
                .arg(&agreement_p)
                .arg(&n_fused_i)
                .launch(LaunchConfig {
                    block_dim: (256, 1, 1),
                    grid_dim: (n_fused as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        let h_agreement: Vec<f32> = stream.memcpy_dtov(&agreement_stats)?;
        let mut h_routes = vec![0i32; n_fused];
        for t in 0..n_fused {
            let base = t * 3;
            if h_agreement[base] > 0.0
                && h_agreement[base + 1] > 0.0
                && h_agreement[base + 2] <= 0.0
            {
                h_routes[t] = 1;
            }
        }
        let d_routes = stream.memcpy_stod(&h_routes)?;
        unsafe {
            stream.launch_builder(&route_kernel)
                .arg(&d_all_ptrs)
                .arg(&d_offsets)
                .arg(&d_routes)
                .arg(&ema_gamma)
                .arg(&ema_beta)
                .arg(&n_fused_i)
                .launch(LaunchConfig {
                    block_dim: (256, 1, 1),
                    grid_dim: (n_fused as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }

    let matrix_energies = stream.alloc_zeros::<f32>(n_fused * 512)?;
    let (matrix_energy_p, _se) = matrix_energies.device_ptr(&stream);
    let energy_cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: ((n_fused * 512) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream.launch_builder(&matrix_energy_kernel)
            .arg(&d_all_ptrs)
            .arg(&d_geometry_mask)
            .arg(&matrix_energy_p)
            .arg(&n_fused_i)
            .launch(energy_cfg)?;
    }

    let h_matrix_energies: Vec<f32> = stream.memcpy_dtov(&matrix_energies)?;
    let mut h_matrix_scales = vec![1.0f32; n_fused * 512];
    for t in 0..n_fused {
        if h_geometry_mask[t] == 0 {
            continue;
        }
        let base = t * 512;
        let mut row_total = 0.0f32;
        let mut col_total = 0.0f32;
        for axis in 0..256 {
            row_total += h_matrix_energies[base + axis];
            col_total += h_matrix_energies[base + 256 + axis];
        }
        let row_reference = row_total / 256.0f32;
        let col_reference = col_total / 256.0f32;
        if row_reference > 0.0 {
            for row in 0..256 {
                let energy = h_matrix_energies[base + row];
                if energy > 0.0 {
                    h_matrix_scales[base + row] = (row_reference / energy).sqrt();
                }
            }
        }
        if col_reference > 0.0 {
            for col in 0..256 {
                let energy = h_matrix_energies[base + 256 + col];
                if energy > 0.0 {
                    h_matrix_scales[base + 256 + col] = (col_reference / energy).sqrt();
                }
            }
        }
    }
    let d_matrix_scales = stream.memcpy_stod(&h_matrix_scales)?;
    let balance_cfg = LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: ((n_fused * 256) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream.launch_builder(&matrix_balance_kernel)
            .arg(&d_all_ptrs)
            .arg(&d_geometry_mask)
            .arg(&d_matrix_scales)
            .arg(&n_fused_i)
            .launch(balance_cfg)?;
    }

    if muon_lr_scale > 1e-7 {
        let k_frob  = module.load_function("ns_frob_kernel")?;
        let k_norm  = module.load_function("ns_copy_normalize")?;
        let k_atb   = module.load_function("ns_atb_256")?;
        let k_ab    = module.load_function("ns_ab_256")?;
        let k_poly  = module.load_function("ns_poly_256")?;
        let k_delta = module.load_function("muon_ema_delta_256")?;
        let k_stats = module.load_function("adan_muon_candidate_stats_256")?;
        let k_commit = module.load_function("muon_candidate_commit_256")?;

        let muon_lr     = lr * muon_lr_scale;
        let muon_beta   = nesterov_beta * lr_frac;
        let ns_a        = 3.4445f32;
        let ns_b        = -4.7750f32;
        let ns_c        = 2.0315f32;
        let eps_norm    = 1e-7f32;
        let n_65536: i32 = NS_MATRIX_SIZE as i32;
        let cfg1d = LaunchConfig { block_dim: (256, 1, 1), grid_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let cfg2d = LaunchConfig { block_dim: (16, 16, 1), grid_dim: (16, 16, 1), shared_mem_bytes: 0 };
        let cfg_one_block = LaunchConfig { block_dim: (256, 1, 1), grid_dim: (1, 1, 1), shared_mem_bytes: 0 };

        let mut muon_ema_idx = 0usize;

        for i in 0..n_params {
            let group = state.param_groups.get(i).copied().unwrap_or(GROUP_HIDDEN_WEIGHT);
            if group != GROUP_HIDDEN_WEIGHT || state.m[i].len() != NS_MATRIX_SIZE { continue; }

            let sq_partials = stream.alloc_zeros::<f32>(256)?;
            let (m_p, _gm) = state.m[i].device_ptr(&stream);
            let (sq_partial_p, _gsp) = sq_partials.device_ptr(&stream);
            unsafe {
                stream.launch_builder(&k_frob)
                    .arg(&m_p).arg(&sq_partial_p).arg(&n_65536)
                    .launch(cfg1d)?;
            }
            let h_sq_partials: Vec<f32> = stream.memcpy_dtov(&sq_partials)?;
            let mut sq_total = 0.0f32;
            for partial in h_sq_partials {
                sq_total += partial;
            }
            let sq_sum = stream.memcpy_stod(&[sq_total])?;
            let (sq_p, _gsq) = sq_sum.device_ptr(&stream);

            let (b0_p, _gb0) = state.ns_buf0.device_ptr(&stream);
            unsafe {
                stream.launch_builder(&k_norm)
                    .arg(&m_p).arg(&b0_p).arg(&sq_p).arg(&eps_norm).arg(&n_65536)
                    .launch(cfg1d)?;
            }

            let mut cur = 0usize;
            for _ns in 0..5 {
                let nxt = 1 - cur;
                let (cur_p, _gc) = if cur == 0 { state.ns_buf0.device_ptr(&stream) }
                                   else         { state.ns_buf1.device_ptr(&stream) };
                let (nxt_p, _gn) = if nxt == 0 { state.ns_buf0.device_ptr(&stream) }
                                   else         { state.ns_buf1.device_ptr(&stream) };
                let (x_p,  _gx)  = state.ns_x.device_ptr(&stream);
                let (x2_p, _gx2) = state.ns_x2.device_ptr(&stream);
                let (pl_p, _gpl) = state.ns_poly.device_ptr(&stream);

                unsafe { stream.launch_builder(&k_atb).arg(&cur_p).arg(&cur_p).arg(&x_p).launch(cfg2d)?; }
                unsafe { stream.launch_builder(&k_ab).arg(&x_p).arg(&x_p).arg(&x2_p).launch(cfg2d)?; }
                unsafe {
                    stream.launch_builder(&k_poly)
                        .arg(&x_p).arg(&x2_p).arg(&pl_p)
                        .arg(&ns_a).arg(&ns_b).arg(&ns_c).arg(&n_65536)
                        .launch(cfg1d)?;
                }
                unsafe { stream.launch_builder(&k_ab).arg(&cur_p).arg(&pl_p).arg(&nxt_p).launch(cfg2d)?; }
                cur = nxt;
            }

            let muon_candidate = stream.alloc_zeros::<f32>(NS_MATRIX_SIZE)?;
            let (ns_dir_p, _gdir) = if cur == 0 { state.ns_buf0.device_ptr(&stream) }
                                    else         { state.ns_buf1.device_ptr(&stream) };
            let (mema_p, _gme) = state.muon_ema[muon_ema_idx].device_ptr(&stream);
            let adan_raw = { let (p, _g) = updates[i].as_ref().unwrap().device_ptr(&stream); p };
            let (muon_raw, _gmu) = muon_candidate.device_ptr(&stream);
            unsafe {
                stream.launch_builder(&k_delta)
                    .arg(&ns_dir_p).arg(&mema_p).arg(&muon_raw)
                    .arg(&muon_lr).arg(&ema_gamma).arg(&muon_beta).arg(&n_65536)
                    .launch(cfg1d)?;
            }

            let candidate_stats = stream.alloc_zeros::<f32>(8)?;
            let (grad_p, _gg) = gradients[i].device_ptr(&stream);
            let (param_p, _gp) = model_params[i].device_ptr(&stream);
            let (stats_p, _gst) = candidate_stats.device_ptr(&stream);
            unsafe {
                stream.launch_builder(&k_stats)
                    .arg(&grad_p).arg(&param_p).arg(&adan_raw).arg(&muon_raw)
                    .arg(&stats_p).arg(&n_65536)
                    .launch(cfg_one_block)?;
            }
            let h: Vec<f32> = stream.memcpy_dtov(&candidate_stats)?;

            let choose_muon =
                h[0] > 0.0 && h[1] > 0.0 && h[2] > 0.0 && h[3] > 0.0
                && (h[5] / (h[0] * h[3]).sqrt()) < (h[4] / (h[0] * h[2]).sqrt())
                && (h[7] / (h[1] * h[3]).sqrt()) <= (h[6] / (h[1] * h[2]).sqrt());

            if choose_muon {
                unsafe {
                    stream.launch_builder(&k_commit)
                        .arg(&muon_raw).arg(&adan_raw).arg(&n_65536)
                        .launch(cfg1d)?;
                }
            }

            muon_ema_idx += 1;
        }
    }

    if total_columns > 0 {
        let polarize_kernel = module.load_function("matrix_update_polarize_t27")?;
        unsafe {
            stream.launch_builder(&polarize_kernel)
                .arg(&d_all_ptrs)
                .arg(&d_offsets)
                .arg(&d_rows)
                .arg(&d_column_offsets)
                .arg(&total_columns)
                .arg(&n_fused_i)
                .launch(LaunchConfig {
                    block_dim: (256, 1, 1),
                    grid_dim: (total_columns as u32, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }

    let mut out: Vec<CudaSlice<f32>> = updates.into_iter().map(|o| o.unwrap()).collect();

    {
        let sie = state.dc_sie;
        let res: Result<()> = DC.with(|cell| {
            let mut d = cell.borrow_mut();
            if !d.enable {
                return Ok(());
            }
            dc::step(
                &mut *d, sie, epoch, val_loss, model_params, &mut out,
                &stream, &module, &mut state.dc_scratch, DC_AXPY_T27,
            )
        });
        res?;
    }

    let record_kernel = module.load_function("record_applied_update_fused_t27")?;
    unsafe {
        stream.launch_builder(&record_kernel)
            .arg(&d_all_ptrs)
            .arg(&d_offsets)
            .arg(&total_i)
            .arg(&n_fused_i)
            .launch(cfg)?;
    }

    Ok(out)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let read_hp = |key: &str, default: f32| -> f32 {
        hyperparameters
            .as_ref()
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_f64())
            .map(|x| x as f32)
            .unwrap_or(default)
    };
    let ema_gamma = read_hp("ema_decay", EMA_GAMMA_DEFAULT).clamp(0.0, 0.9998);
    let nesterov_beta = read_hp("nesterov_beta", EMA_BETA_DEFAULT).clamp(0.0, 1.0);
    let muon_lr_scale = read_hp("muon_lr_scale", 0.0).clamp(0.0, 10.0);
    let hidden_bias_scale = read_hp("hidden_bias_scale", 1.25_f32).clamp(0.5, 3.0);
    let qh_nu = read_hp("qh_nu", QH_NU_DEFAULT).clamp(0.0, 1.0);
    let ab_blend = read_hp("ab_blend", AB_BLEND_DEFAULT).clamp(0.0, 1.0);
    let sophia_rho = read_hp("sophia_rho", 0.0_f32).clamp(0.0, 100.0);
    let t_max_epochs = read_hp("t_max_epochs", 850.0_f32).clamp(100.0, 5000.0) as u32;
    let min_lr_scale = read_hp("min_lr_scale", MIN_LR_SCALE).clamp(0.1, 1.0);
    let plateau_patience = read_hp("plateau_patience", PLATEAU_PATIENCE as f32).clamp(1.0, 200.0) as u32;
    let mars_gamma = read_hp("mars_gamma", 0.0_f32).clamp(0.0, 1.0);
    let mars_clip_scale = read_hp("mars_clip_scale", 1.0_f32).clamp(0.0, 2.0);
    EMA_GAMMA_BITS.store(ema_gamma.to_bits(), Ordering::Relaxed);
    EMA_BETA_BITS.store(nesterov_beta.to_bits(), Ordering::Relaxed);
    MUON_LR_SCALE_BITS.store(muon_lr_scale.to_bits(), Ordering::Relaxed);
    HIDDEN_BIAS_SCALE_BITS.store(hidden_bias_scale.to_bits(), Ordering::Relaxed);
    QH_NU_BITS.store(qh_nu.to_bits(), Ordering::Relaxed);
    AB_BLEND_BITS.store(ab_blend.to_bits(), Ordering::Relaxed);
    SOPHIA_RHO_BITS.store(sophia_rho.to_bits(), Ordering::Relaxed);
    T_MAX_EPOCHS_OVERRIDE.store(t_max_epochs, Ordering::Relaxed);
    MIN_LR_SCALE_BITS.store(min_lr_scale.to_bits(), Ordering::Relaxed);
    PLATEAU_PATIENCE_OVERRIDE.store(plateau_patience, Ordering::Relaxed);
    MARS_GAMMA_BITS.store(mars_gamma.to_bits(), Ordering::Relaxed);
    MARS_CLIP_SCALE_BITS.store(mars_clip_scale.to_bits(), Ordering::Relaxed);

    let dcfg = dc::config_from_hp(hyperparameters, DC_ENABLE_DEFAULT_T27);
    DC.with(|c| *c.borrow_mut() = dc::DcState::default());
    if dcfg.enable != 0 {
        let m = dc::target_mean(challenge, &stream)?;
        let nb = dc::batches_per_epoch(challenge);
        DC.with(|c| c.borrow_mut().arm(&dcfg, m, nb));
    }

    training_loop(challenge, save_solution, module, stream, prop, optimizer_init, optimizer_query, optimizer_step)?;
    Ok(())
}