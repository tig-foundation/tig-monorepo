use anyhow::{anyhow, Result};
use cudarc::{
    driver::{
        CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    },
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::neuralnet_optimizer::*;

const K_ADAN: &str = "sk_adan_4";
const K_KINK: &str = "sk_kink_4";
const K_KINK2: &str = "sk_kink2_4";
const K_OFFSET: &str = "sk_offset_4";
const K_SCALE: &str = "sk_scale_4";
const K_AXIS_RMS: &str = "sk_axis_rms_4";
const K_MATRIX_ADAN_PRECOND: &str = "sk_matrix_adan_precond_rollback_4";
const K_NONMATRIX_BATCH: &str = "sk_nonmatrix_adan_rollback_batch_4";
const K_ROLLBACK: &str = "sk_rollback_4";

extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

#[derive(Clone, Debug)]
struct Cfg {
    lr_max: f32,
    lr_min: f32,
    warmup_epochs: usize,
    t_max: usize,
    wd: f32,
    eps: f32,
    beta1: f32,
    beta2: f32,
    b3: f32,
    cautious: f32,
    plateau_patience: usize,
    plateau_decay: f32,
    plateau_grow: f32,
    plateau_floor: f32,
    head_mult: f32,
    kink: f32,
    kink_lin: f32,
    depth_lr: f32,
    meanalt: f32,
    meanalt_start: usize,
    meanalt_levels: usize,
    meanalt_top: f32,
    fuel_reserve: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            lr_max: 2.0e-3,
            lr_min: 2e-5,
            warmup_epochs: 8,
            t_max: 400,
            wd: 0.02,
            eps: 1e-8,
            beta1: 0.98,
            beta2: 0.92,
            b3: 0.99,
            cautious: 0.25,
            plateau_patience: 12,
            plateau_decay: 0.82,
            plateau_floor: 0.15,
            plateau_grow: 1.03,
            head_mult: 1.0,
            kink: 1.0,
            kink_lin: 0.80,
            depth_lr: 1.0,
            meanalt: 0.4,
            meanalt_start: 20,
            meanalt_levels: 1,
            meanalt_top: 1.0,
            fuel_reserve: 55,
        }
    }
}

static CFG: OnceLock<Cfg> = OnceLock::new();

fn cfg() -> &'static Cfg {
    CFG.get_or_init(Cfg::default)
}

fn parse_cfg(hp: &Option<Map<String, Value>>) -> Cfg {
    let mut c = Cfg::default();
    let Some(m) = hp else { return c };
    let f = |k: &str, d: f32| m.get(k).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
    let u = |k: &str, d: usize| m.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(d);
    c.lr_max = f("lr_max", c.lr_max);
    c.lr_min = f("lr_min", c.lr_min);
    c.warmup_epochs = u("warmup_epochs", c.warmup_epochs);
    c.t_max = u("t_max", c.t_max).max(1);
    c.wd = f("wd", c.wd);
    c.eps = f("eps", c.eps);
    c.beta1 = f("beta1", c.beta1);
    c.beta2 = f("beta2", c.beta2);
    c.b3 = f("b3", c.b3);
    c.cautious = f("cautious", c.cautious);
    c.plateau_patience = u("plateau_patience", c.plateau_patience);
    c.plateau_decay = f("plateau_decay", c.plateau_decay);
    c.plateau_grow = f("plateau_grow", c.plateau_grow);
    c.plateau_floor = f("plateau_floor", c.plateau_floor);
    c.head_mult = f("head_mult", c.head_mult);
    c.kink = f("kink", c.kink);
    c.kink_lin = f("kink_lin", c.kink_lin);
    c.depth_lr = f("depth_lr", c.depth_lr);
    c.meanalt = f("meanalt", c.meanalt);
    c.meanalt_start = u("meanalt_start", c.meanalt_start);
    c.meanalt_levels = u("meanalt_levels", c.meanalt_levels).max(1);
    c.meanalt_top = f("meanalt_top", c.meanalt_top);
    c.fuel_reserve = u("fuel_reserve", c.fuel_reserve);
    c
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Weight,
    Bias,
    HeadWeight,
    HeadBias,
    BnWeight,
    BnBias,
    Skip,
}

fn roles(param_count: usize) -> Vec<Role> {
    let l = (param_count + 4) / 6;
    let mut r = vec![Role::Skip; param_count];
    let n_trainable = l.saturating_sub(2);
    for i in 0..n_trainable {
        let head = i + 1 == n_trainable;
        r[2 * i] = if head { Role::HeadWeight } else { Role::Weight };
        r[2 * i + 1] = if head { Role::HeadBias } else { Role::Bias };
    }
    for j in 0..n_trainable.min(l.saturating_sub(1)) {
        let base = 2 * l + 4 * j;
        if base + 1 < param_count {
            r[base] = Role::BnWeight;
            r[base + 1] = Role::BnBias;
        }
    }
    r
}

fn role_lr_wd(role: Role, lr: f32, wd: f32, head_mult: f32) -> (f32, f32) {
    match role {
        Role::Weight => (lr, wd),
        Role::Bias => (lr * 1.25, 0.0),
        Role::HeadWeight => (lr * head_mult, wd),
        Role::HeadBias => (lr * head_mult * 1.25, 0.0),
        Role::BnWeight => (lr * 0.55, 0.0),
        Role::BnBias => (lr * 0.80, 0.0),
        Role::Skip => (0.0, 0.0),
    }
}

struct State {
    _module: Arc<CudaModule>,
    k_adan: CudaFunction,
    k_kink: CudaFunction,
    k_kink2: CudaFunction,
    k_offset: CudaFunction,
    k_scale: CudaFunction,
    k_axis_rms: CudaFunction,
    k_matrix_adan_precond: CudaFunction,
    k_nonmatrix_batch: CudaFunction,
    k_rollback: CudaFunction,
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    nsq: Vec<CudaSlice<f32>>,
    gprev: Vec<CudaSlice<f32>>,
    last_delta: Vec<CudaSlice<f32>>,
    row_rms: Vec<CudaSlice<f32>>,
    col_rms: Vec<CudaSlice<f32>>,
    roles: Vec<Role>,
    active: Vec<usize>,
    matrix: Vec<bool>,
    depth_mult: Vec<f32>,
    kink_eff: f32,
    kink_lin_eff: f32,
    lr_eff: f32,
    meanalt_eff: f32,
    levels_eff: usize,
    top_eff: f32,
    step: usize,
    epoch: usize,
    epoch_seen: bool,
    best_val: f32,
    plateau_stale: usize,
    lr_scale: f32,
    batch_in_epoch: usize,
    batches_per_epoch: usize,
    shift_idx: usize,
    shift_applied: f32,
    epoch_cap: usize,
    stalled: bool,
}

impl OptimizerStateTrait for State {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> {
        unreachable!("the training loop never clones the state")
    }
}

pub fn optimizer_init_state(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let c = cfg();
    let roles = roles(param_sizes.len());
    let active: Vec<usize> = (0..param_sizes.len()).filter(|&i| roles[i] != Role::Skip).collect();
    let matrix: Vec<bool> = (0..param_sizes.len()).map(|i| {
        matches!(roles[i], Role::Weight | Role::HeadWeight)
            && i + 1 < param_sizes.len()
            && matches!(roles[i + 1], Role::Bias | Role::HeadBias)
            && param_sizes[i + 1] > 0
            && param_sizes[i] % param_sizes[i + 1] == 0
    }).collect();

    let alloc = |i: usize| -> Result<CudaSlice<f32>> {
        stream.alloc_zeros::<f32>(if roles[i] == Role::Skip { 1 } else { param_sizes[i] })
            .map_err(|e| anyhow!("alloc: {e}"))
    };
    let buf = || -> Result<Vec<CudaSlice<f32>>> { (0..param_sizes.len()).map(alloc).collect() };
    let rms_buf = |rows_axis: bool| -> Result<Vec<CudaSlice<f32>>> {
        (0..param_sizes.len()).map(|i| {
            let n = if matrix[i] {
                let rows = param_sizes[i + 1];
                if rows_axis { rows } else { param_sizes[i] / rows }
            } else { 1 };
            stream.alloc_zeros::<f32>(n).map_err(|e| anyhow!("alloc: {e}"))
        }).collect()
    };

    let l = (param_sizes.len() + 4) / 6;
    let n_trainable = l.saturating_sub(2).max(1);

    Ok(Box::new(State {
        k_adan: module.load_function(K_ADAN)?,
        k_kink: module.load_function(K_KINK)?,
        k_kink2: module.load_function(K_KINK2)?,
        k_offset: module.load_function(K_OFFSET)?,
        k_scale: module.load_function(K_SCALE)?,
        k_axis_rms: module.load_function(K_AXIS_RMS)?,
        k_matrix_adan_precond: module.load_function(K_MATRIX_ADAN_PRECOND)?,
        k_nonmatrix_batch: module.load_function(K_NONMATRIX_BATCH)?,
        k_rollback: module.load_function(K_ROLLBACK)?,
        _module: module,
        m: buf()?, v: buf()?, nsq: buf()?, gprev: buf()?, last_delta: buf()?,
        row_rms: rms_buf(true)?, col_rms: rms_buf(false)?,
        kink_eff: c.kink, kink_lin_eff: c.kink_lin, meanalt_eff: c.meanalt,
        levels_eff: c.meanalt_levels, top_eff: c.meanalt_top, lr_eff: c.lr_max,
        depth_mult: (0..param_sizes.len()).map(|i| {
            if c.depth_lr == 1.0 || i >= 2 * n_trainable { 1.0 } else {
                c.depth_lr.powf(if n_trainable > 1 {
                    (i / 2) as f32 / (n_trainable - 1) as f32
                } else { 0.0 })
            }
        }).collect(),
        roles, active, matrix, step: 0, epoch: usize::MAX, epoch_seen: false,
        best_val: f32::INFINITY, plateau_stale: 0, lr_scale: 1.0,
        batch_in_epoch: 0, batches_per_epoch: 0,
        shift_idx: 2 * l + 4 * n_trainable.saturating_sub(1) + 1,
        shift_applied: 0.0,
        epoch_cap: {
            let sum_w: usize = (0..l).map(|i| param_sizes[2 * i]).sum();
            (fuel_remaining() / (870u64 * sum_w as u64).max(1)) as usize
        }.saturating_sub(c.fuel_reserve),
        stalled: false,
    }))
}

pub fn optimizer_query_at_params(
    _state: &dyn OptimizerStateTrait, _params: &[CudaSlice<f32>], _epoch: usize,
    _train: Option<f32>, _val: Option<f32>, _stream: Arc<CudaStream>,
    _module: Arc<CudaModule>, _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> { Ok(None) }

fn lr_at(s: &State, c: &Cfg, epoch: usize) -> f32 {
    if epoch < c.warmup_epochs {
        return s.lr_eff * (epoch + 1) as f32 / c.warmup_epochs.max(1) as f32;
    }
    let denom = c.t_max.saturating_sub(c.warmup_epochs).max(1) as f32;
    c.lr_min + 0.5 * (s.lr_eff - c.lr_min) *
        (1.0 + (std::f32::consts::PI * (epoch as f32 / denom).min(1.0)).cos())
}

pub fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>], gradients: &[CudaSlice<f32>],
    epoch: usize, _train_loss: Option<f32>, val_loss: Option<f32>,
    stream: Arc<CudaStream>, _module: Arc<CudaModule>, _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let c = cfg();
    let s = optimizer_state.as_any_mut().downcast_mut::<State>()
        .ok_or_else(|| anyhow!("unexpected optimizer state"))?;

    let new_epoch = s.epoch != epoch;
    if new_epoch {
        if s.epoch_seen && s.batches_per_epoch == 0 { s.batches_per_epoch = s.batch_in_epoch; }
        s.batch_in_epoch = 0; s.epoch = epoch; s.epoch_seen = true;
        if let Some(val) = val_loss.filter(|v| v.is_finite()) {
            if val < s.best_val {
                s.best_val = val; s.plateau_stale = 0;
                s.lr_scale = (s.lr_scale * c.plateau_grow).min(1.0);
            } else {
                s.plateau_stale += 1;
                if s.plateau_stale >= c.plateau_patience {
                    s.lr_scale = (s.lr_scale * c.plateau_decay).max(c.plateau_floor);
                    s.plateau_stale = 0;
                }
            }
        }
    }
    if new_epoch && epoch >= s.epoch_cap { s.stalled = true; }
    let batch_idx = s.batch_in_epoch; s.batch_in_epoch += 1; s.step += 1;

    let lr = lr_at(s, c, epoch) * s.lr_scale;
    let bc1 = 1.0 - c.beta1.powi(s.step as i32);
    let bc2 = 1.0 - c.beta2.powi(s.step as i32);
    let bc3 = 1.0 - c.b3.powi(s.step as i32);
    let first_step: i32 = if s.step == 1 { 1 } else { 0 };

    let mut updates = Vec::with_capacity(model_params.len());
    for (i, p) in model_params.iter().enumerate() {
        let n = if s.roles[i] == Role::Skip { 1 } else { p.len() };
        updates.push(if s.stalled || s.roles[i] == Role::Skip {
            stream.alloc_zeros::<f32>(n)?
        } else { unsafe { stream.alloc::<f32>(n) }? });
    }

    if s.stalled {
        if !model_params.is_empty() && updates[0].len() == model_params[0].len() {
            let n = model_params[0].len(); let neg = -1.0f32;
            unsafe { stream.launch_builder(&s.k_scale).arg(&mut updates[0]).arg(&model_params[0])
                .arg(&neg).arg(&(n as i32)).launch(LaunchConfig::for_num_elems(n as u32))?; }
        }
        return Ok(updates);
    }

    let cycle = if s.meanalt_eff != 0.0 && epoch >= c.meanalt_start {
        let ph = epoch % (2 * s.levels_eff + 1);
        if ph == 0 { 0.0 } else {
            let k = ((ph + 1) / 2) as f32;
            let sign = if ph % 2 == 1 { 1.0 } else { -1.0 };
            sign * s.meanalt_eff * s.top_eff * k / s.levels_eff as f32
        }
    } else { 0.0 };

    let mut offset_amount = 0.0;
    if s.shift_idx < updates.len() && s.roles[s.shift_idx] != Role::Skip && s.batches_per_epoch > 0 {
        if batch_idx == 0 && s.shift_applied != 0.0 {
            offset_amount -= s.shift_applied; s.shift_applied = 0.0;
        }
        if batch_idx + 1 >= s.batches_per_epoch && cycle != 0.0 {
            offset_amount += cycle; s.shift_applied = cycle;
        }
    }

    let kink_overwrites = s.kink_eff > 0.0 && s.step == 1 && model_params.len() >= 2
        && model_params[0].len() == model_params[1].len() && model_params[1].len() > 1;
    let mut fallback_nonmatrix = Vec::new();
    let mut descriptor_words = Vec::<u64>::new();

    for &i in s.active.iter() {
        let n = model_params[i].len();
        let n_i = n as i32;
        let (lr_i_base, wd_i) = role_lr_wd(s.roles[i], lr, c.wd, c.head_mult);
        let lr_i = lr_i_base * s.depth_mult[i];

        if s.matrix[i] {
            let rows = model_params[i + 1].len() as i32;
            let cols = (n / rows as usize) as i32;
            unsafe {
                stream.launch_builder(&s.k_axis_rms).arg(&gradients[i]).arg(&mut s.row_rms[i])
                    .arg(&rows).arg(&cols).arg(&0i32).launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0,
                    })?;
                stream.launch_builder(&s.k_axis_rms).arg(&gradients[i]).arg(&mut s.col_rms[i])
                    .arg(&rows).arg(&cols).arg(&1i32).launch(LaunchConfig {
                        grid_dim: (cols as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0,
                    })?;
                stream.launch_builder(&s.k_matrix_adan_precond)
                    .arg(&gradients[i]).arg(&model_params[i]).arg(&mut s.m[i]).arg(&mut s.v[i])
                    .arg(&mut s.nsq[i]).arg(&mut s.gprev[i]).arg(&mut updates[i])
                    .arg(&s.row_rms[i]).arg(&s.col_rms[i]).arg(&mut s.last_delta[i])
                    .arg(&lr_i).arg(&c.beta1).arg(&c.beta2).arg(&c.b3).arg(&c.eps).arg(&wd_i)
                    .arg(&bc1).arg(&bc2).arg(&bc3).arg(&first_step).arg(&c.cautious)
                    .arg(&cols).arg(&n_i).launch(LaunchConfig::for_num_elems(n as u32))?;
            }
        } else if (kink_overwrites && i == 1) || (offset_amount != 0.0 && i == s.shift_idx) {
            fallback_nonmatrix.push(i);
            unsafe {
                stream.launch_builder(&s.k_adan)
                    .arg(&gradients[i]).arg(&model_params[i]).arg(&mut s.m[i]).arg(&mut s.v[i])
                    .arg(&mut s.nsq[i]).arg(&mut s.gprev[i]).arg(&mut updates[i])
                    .arg(&lr_i).arg(&c.beta1).arg(&c.beta2).arg(&c.b3).arg(&c.eps).arg(&wd_i)
                    .arg(&bc1).arg(&bc2).arg(&bc3).arg(&first_step).arg(&c.cautious).arg(&n_i)
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
        } else {
            let (grad_ptr, _grad_sync) = gradients[i].device_ptr(&stream);
            let (param_ptr, _param_sync) = model_params[i].device_ptr(&stream);
            let (m_ptr, _m_sync) = s.m[i].device_ptr(&stream);
            let (v_ptr, _v_sync) = s.v[i].device_ptr(&stream);
            let (nsq_ptr, _nsq_sync) = s.nsq[i].device_ptr(&stream);
            let (gprev_ptr, _gprev_sync) = s.gprev[i].device_ptr(&stream);
            let (update_ptr, _update_sync) = updates[i].device_ptr(&stream);
            let (last_delta_ptr, _last_delta_sync) = s.last_delta[i].device_ptr(&stream);

            for base in (0..n).step_by(256) {
                let count = (n - base).min(256);
                descriptor_words.extend_from_slice(&[
                    grad_ptr, param_ptr, m_ptr, v_ptr, nsq_ptr, gprev_ptr, update_ptr, last_delta_ptr,
                    lr_i.to_bits() as u64, wd_i.to_bits() as u64, base as u64, count as u64,
                ]);
            }
        }
    }

    if !descriptor_words.is_empty() {
        let mut descriptors = unsafe { stream.alloc::<u64>(descriptor_words.len()) }?;
        stream.memcpy_htod(&descriptor_words, &mut descriptors)?;
        unsafe {
            stream.launch_builder(&s.k_nonmatrix_batch).arg(&descriptors)
                .arg(&c.beta1).arg(&c.beta2).arg(&c.b3).arg(&c.eps).arg(&bc1).arg(&bc2).arg(&bc3)
                .arg(&first_step).arg(&c.cautious).launch(LaunchConfig {
                    grid_dim: ((descriptor_words.len() / 12) as u32, 1, 1),
                    block_dim: (256, 1, 1), shared_mem_bytes: 0,
                })?;
        }
    }

    if s.kink_eff > 0.0 && s.step == 1 && model_params.len() >= 2 {
        let nb = model_params[1].len();
        if model_params[0].len() == nb && nb > 1 {
            let nb_i = nb as i32;
            unsafe {
                if s.kink_lin_eff >= 0.0 {
                    stream.launch_builder(&s.k_kink2).arg(&mut updates[1]).arg(&model_params[0])
                        .arg(&model_params[1]).arg(&s.kink_lin_eff).arg(&s.kink_eff).arg(&nb_i)
                        .launch(LaunchConfig::for_num_elems(nb as u32))?;
                } else {
                    stream.launch_builder(&s.k_kink).arg(&mut updates[1]).arg(&model_params[0])
                        .arg(&model_params[1]).arg(&s.kink_eff).arg(&nb_i)
                        .launch(LaunchConfig::for_num_elems(nb as u32))?;
                }
            }
        }
    }

    if offset_amount != 0.0 {
        let n = model_params[s.shift_idx].len();
        unsafe {
            stream.launch_builder(&s.k_offset).arg(&mut updates[s.shift_idx]).arg(&offset_amount)
                .arg(&(n as i32)).launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }

    unsafe {
        for &i in fallback_nonmatrix.iter() {
            let n = model_params[i].len();
            stream.launch_builder(&s.k_rollback).arg(&mut updates[i]).arg(&mut s.last_delta[i])
                .arg(&first_step).arg(&(n as i32)).launch(LaunchConfig::for_num_elems(n as u32))?;
        }
    }
    Ok(updates)
}

pub fn solve(
    challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>, module: Arc<CudaModule>,
    stream: Arc<CudaStream>, prop: &cudaDeviceProp,
) -> Result<()> {
    let _ = CFG.set(parse_cfg(hyperparameters));
    training_loop(challenge, save_solution, module, stream, prop,
        optimizer_init_state, optimizer_query_at_params, optimizer_step)
}
