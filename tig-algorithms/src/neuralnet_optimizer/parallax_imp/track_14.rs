use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::neuralnet_optimizer::*;

const K_ADAN: &str = "sk_adan_14";
const K_KINK: &str = "sk_kink_14";
const K_KINK2: &str = "sk_kink2_14";
const K_OFFSET_ROLLBACK: &str = "sk_offset_rollback_14";
const K_SCALE: &str = "sk_scale_14";
const K_AXIS_RMS: &str = "sk_axis_rms_14";
const K_MATRIX_PRECOND: &str = "sk_matrix_precond_14";
const K_BATCHED_ADAN_ROLLBACK: &str = "sk_batched_adan_rollback_14";
const K_MATRIX_ADAN_PRECOND_ROLLBACK: &str = "sk_matrix_adan_precond_rollback_14";
const K_ROLLBACK: &str = "sk_rollback_14";

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
            plateau_grow: 1.03,
            plateau_floor: 0.15,
            head_mult: 1.0,
            kink: 1.5,
            kink_lin: -1.0,
            depth_lr: 1.0,
            meanalt: 1.4,
            meanalt_start: 20,
            meanalt_levels: 3,
            meanalt_top: 2.0,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct AdanDescriptor {
    grad: u64,
    param: u64,
    m: u64,
    v: u64,
    nsq: u64,
    gprev: u64,
    delta: u64,
    previous: u64,
    lr: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    eps: f32,
    wd: f32,
    bc1: f32,
    bc2: f32,
    bc3: f32,
    cautious: f32,
    first_step: i32,
    n: i32,
    block_start: i32,
    block_end: i32,
}

unsafe impl DeviceRepr for AdanDescriptor {}

struct MatrixWorkspace {
    row_rms: CudaSlice<f32>,
    col_rms: CudaSlice<f32>,
    rows: i32,
    cols: i32,
}

struct State {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    nsq: Vec<CudaSlice<f32>>,
    gprev: Vec<CudaSlice<f32>>,
    last_delta: Vec<CudaSlice<f32>>,
    matrix_workspaces: Vec<Option<MatrixWorkspace>>,
    batched_desc: CudaSlice<AdanDescriptor>,
    batched_host: Vec<AdanDescriptor>,
    roles: Vec<Role>,
    active: Vec<usize>,
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
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> {
        unreachable!("the training loop never clones the state")
    }
}

pub fn optimizer_init_state(
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let c = cfg();
    let roles = roles(param_sizes.len());
    let active: Vec<usize> = (0..param_sizes.len())
        .filter(|&i| roles[i] != Role::Skip)
        .collect();

    let alloc = |i: usize| -> Result<CudaSlice<f32>> {
        let n = if roles[i] == Role::Skip { 1 } else { param_sizes[i] };
        stream.alloc_zeros::<f32>(n).map_err(|e| anyhow!("alloc: {e}"))
    };
    let buf = || -> Result<Vec<CudaSlice<f32>>> { (0..param_sizes.len()).map(alloc).collect() };

    let l = (param_sizes.len() + 4) / 6;
    let n_trainable = l.saturating_sub(2).max(1);
    let matrix_workspaces = (0..param_sizes.len())
        .map(|i| -> Result<Option<MatrixWorkspace>> {
            if matches!(roles[i], Role::Weight | Role::HeadWeight)
                && i + 1 < param_sizes.len()
                && matches!(roles[i + 1], Role::Bias | Role::HeadBias)
            {
                let rows = param_sizes[i + 1];
                if rows > 0 && param_sizes[i] % rows == 0 {
                    let cols = param_sizes[i] / rows;
                    return Ok(Some(MatrixWorkspace {
                        row_rms: stream.alloc_zeros::<f32>(rows)?,
                        col_rms: stream.alloc_zeros::<f32>(cols)?,
                        rows: rows as i32,
                        cols: cols as i32,
                    }));
                }
            }
            Ok(None)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Box::new(State {
        m: buf()?,
        v: buf()?,
        nsq: buf()?,
        gprev: buf()?,
        last_delta: buf()?,
        matrix_workspaces,
        batched_desc: unsafe { stream.alloc::<AdanDescriptor>(active.len().max(1))? },
        batched_host: Vec::with_capacity(active.len()),
        kink_eff: c.kink,
        kink_lin_eff: c.kink_lin,
        meanalt_eff: c.meanalt,
        levels_eff: c.meanalt_levels,
        top_eff: c.meanalt_top,
        lr_eff: c.lr_max,
        depth_mult: (0..param_sizes.len())
            .map(|i| {
                if c.depth_lr == 1.0 || i >= 2 * n_trainable {
                    1.0
                } else {
                    let f = if n_trainable > 1 {
                        (i / 2) as f32 / (n_trainable - 1) as f32
                    } else {
                        0.0
                    };
                    c.depth_lr.powf(f)
                }
            })
            .collect(),
        roles,
        active,
        step: 0,
        epoch: usize::MAX,
        epoch_seen: false,
        best_val: f32::INFINITY,
        plateau_stale: 0,
        lr_scale: 1.0,
        batch_in_epoch: 0,
        batches_per_epoch: 0,
        shift_idx: 2 * l + 4 * n_trainable.saturating_sub(1) + 1,
        shift_applied: 0.0,
        epoch_cap: {
            let sum_w: usize = (0..l).map(|i| param_sizes[2 * i]).sum();
            let per_epoch = 870u64 * sum_w as u64;
            let affordable = fuel_remaining() / per_epoch.max(1);
            (affordable as usize).saturating_sub(c.fuel_reserve)
        },
        stalled: false,
    }))
}

pub fn optimizer_query_at_params(
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

fn lr_at(s: &State, c: &Cfg, epoch: usize) -> f32 {
    if epoch < c.warmup_epochs {
        return s.lr_eff * ((epoch + 1) as f32) / (c.warmup_epochs.max(1) as f32);
    }
    let denom = c.t_max.saturating_sub(c.warmup_epochs).max(1) as f32;
    let p = (epoch as f32 / denom).min(1.0);
    c.lr_min + 0.5 * (s.lr_eff - c.lr_min) * (1.0 + (std::f32::consts::PI * p).cos())
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
    let c = cfg();
    let s = optimizer_state
        .as_any_mut()
        .downcast_mut::<State>()
        .ok_or_else(|| anyhow!("unexpected optimizer state"))?;

    let new_epoch = s.epoch != epoch;
    if new_epoch {
        if s.epoch_seen && s.batches_per_epoch == 0 {
            s.batches_per_epoch = s.batch_in_epoch;
        }
        s.batch_in_epoch = 0;
        s.epoch = epoch;
        s.epoch_seen = true;
        if let Some(val) = val_loss.filter(|v| v.is_finite()) {
            if val < s.best_val {
                s.best_val = val;
                s.plateau_stale = 0;
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
    if new_epoch && epoch >= s.epoch_cap {
        s.stalled = true;
    }

    let batch_idx = s.batch_in_epoch;
    s.batch_in_epoch += 1;
    s.step += 1;

    let lr = lr_at(s, c, epoch) * s.lr_scale;
    let bc1 = 1.0 - c.beta1.powi(s.step as i32);
    let bc2 = 1.0 - c.beta2.powi(s.step as i32);
    let bc3 = 1.0 - c.b3.powi(s.step as i32);
    let first_step: i32 = if s.step == 1 { 1 } else { 0 };

    let kink_applies = s.kink_eff > 0.0
        && s.step == 1
        && model_params.len() >= 2
        && model_params[0].len() == model_params[1].len()
        && model_params[1].len() > 1;

    let cycle = if s.meanalt_eff != 0.0 && epoch >= c.meanalt_start {
        let a = s.meanalt_eff;
        let lv = s.levels_eff;
        let ph = epoch % (2 * lv + 1);
        if ph == 0 {
            0.0
        } else {
            let k = ((ph + 1) / 2) as f32;
            let sign = if ph % 2 == 1 { 1.0 } else { -1.0 };
            sign * a * s.top_eff * k / lv as f32
        }
    } else {
        0.0
    };
    let mut shift_amount = 0.0;
    if s.shift_idx < model_params.len()
        && s.roles[s.shift_idx] != Role::Skip
        && s.batches_per_epoch > 0
    {
        if batch_idx == 0 && s.shift_applied != 0.0 {
            shift_amount -= s.shift_applied;
        }
        if batch_idx + 1 >= s.batches_per_epoch && cycle != 0.0 {
            shift_amount += cycle;
        }
    }

    let mut updates: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());
    for (i, p) in model_params.iter().enumerate() {
        let n = if s.roles[i] == Role::Skip { 1 } else { p.len() };
        let update = if s.stalled || s.roles[i] == Role::Skip {
            stream.alloc_zeros::<f32>(n)?
        } else {
            unsafe { stream.alloc::<f32>(n)? }
        };
        updates.push(update);
    }

    if s.stalled {
        if !model_params.is_empty() && updates[0].len() == model_params[0].len() {
            let n = model_params[0].len();
            let neg = -1.0f32;
            unsafe {
                stream
                    .launch_builder(&module.load_function(K_SCALE)?)
                    .arg(&mut updates[0])
                    .arg(&model_params[0])
                    .arg(&neg)
                    .arg(&(n as i32))
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
        }
        return Ok(updates);
    }

    let k_adan = module.load_function(K_ADAN)?;
    let k_axis_rms = module.load_function(K_AXIS_RMS)?;
    let k_matrix_precond = module.load_function(K_MATRIX_PRECOND)?;
    let k_batched_adan_rollback = module.load_function(K_BATCHED_ADAN_ROLLBACK)?;
    let k_matrix_adan_precond_rollback = module.load_function(K_MATRIX_ADAN_PRECOND_ROLLBACK)?;
    let k_rollback = module.load_function(K_ROLLBACK)?;
    let k_offset_rollback = module.load_function(K_OFFSET_ROLLBACK)?;
    s.batched_host.clear();
    let mut batched_blocks = 0i32;
    for &i in s.active.iter() {
        let n = model_params[i].len();
        let n_i = n as i32;
        let (lr_i, wd_i) = role_lr_wd(s.roles[i], lr, c.wd, c.head_mult);
        let lr_i = lr_i * s.depth_mult[i];

        let fuse_rollback = (i != 1 || !kink_applies)
            && (i != s.shift_idx || shift_amount == 0.0);

        if s.matrix_workspaces[i].is_none() && fuse_rollback {
            let blocks = ((n + 255) / 256) as i32;
            let (grad, _grad_guard) = gradients[i].device_ptr(&stream);
            let (param, _param_guard) = model_params[i].device_ptr(&stream);
            let (m, _m_guard) = s.m[i].device_ptr(&stream);
            let (v, _v_guard) = s.v[i].device_ptr(&stream);
            let (nsq, _nsq_guard) = s.nsq[i].device_ptr(&stream);
            let (gprev, _gprev_guard) = s.gprev[i].device_ptr(&stream);
            let (delta, _delta_guard) = updates[i].device_ptr(&stream);
            let (previous, _previous_guard) = s.last_delta[i].device_ptr(&stream);

            s.batched_host.push(AdanDescriptor {
                grad: grad as u64,
                param: param as u64,
                m: m as u64,
                v: v as u64,
                nsq: nsq as u64,
                gprev: gprev as u64,
                delta: delta as u64,
                previous: previous as u64,
                lr: lr_i,
                b1: c.beta1,
                b2: c.beta2,
                b3: c.b3,
                eps: c.eps,
                wd: wd_i,
                bc1,
                bc2,
                bc3,
                cautious: c.cautious,
                first_step,
                n: n_i,
                block_start: batched_blocks,
                block_end: batched_blocks + blocks,
            });
            batched_blocks += blocks;
            continue;
        }

        unsafe {
            if let Some(workspace) = s.matrix_workspaces[i].as_mut() {
                stream
                    .launch_builder(&k_axis_rms)
                    .arg(&gradients[i])
                    .arg(&mut workspace.row_rms)
                    .arg(&mut workspace.col_rms)
                    .arg(&workspace.rows)
                    .arg(&workspace.cols)
                    .launch(LaunchConfig {
                        grid_dim: ((workspace.rows + workspace.cols) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            if let Some(workspace) = s.matrix_workspaces[i].as_ref() {
                let row_rms = &workspace.row_rms;
                let col_rms = &workspace.col_rms;
                let cols = workspace.cols;
                if fuse_rollback {
                    stream
                        .launch_builder(&k_matrix_adan_precond_rollback)
                        .arg(&gradients[i])
                        .arg(&model_params[i])
                        .arg(&mut s.m[i])
                        .arg(&mut s.v[i])
                        .arg(&mut s.nsq[i])
                        .arg(&mut s.gprev[i])
                        .arg(&mut updates[i])
                        .arg(&mut s.last_delta[i])
                        .arg(row_rms)
                        .arg(col_rms)
                        .arg(&cols)
                        .arg(&lr_i)
                        .arg(&c.beta1)
                        .arg(&c.beta2)
                        .arg(&c.b3)
                        .arg(&c.eps)
                        .arg(&wd_i)
                        .arg(&bc1)
                        .arg(&bc2)
                        .arg(&bc3)
                        .arg(&first_step)
                        .arg(&c.cautious)
                        .arg(&n_i)
                        .launch(LaunchConfig::for_num_elems(n as u32))?;
                } else {
                    stream
                        .launch_builder(&k_adan)
                        .arg(&gradients[i])
                        .arg(&model_params[i])
                        .arg(&mut s.m[i])
                        .arg(&mut s.v[i])
                        .arg(&mut s.nsq[i])
                        .arg(&mut s.gprev[i])
                        .arg(&mut updates[i])
                        .arg(&lr_i)
                        .arg(&c.beta1)
                        .arg(&c.beta2)
                        .arg(&c.b3)
                        .arg(&c.eps)
                        .arg(&wd_i)
                        .arg(&bc1)
                        .arg(&bc2)
                        .arg(&bc3)
                        .arg(&first_step)
                        .arg(&c.cautious)
                        .arg(&n_i)
                        .launch(LaunchConfig::for_num_elems(n as u32))?;
                    stream
                        .launch_builder(&k_matrix_precond)
                        .arg(&mut updates[i])
                        .arg(row_rms)
                        .arg(col_rms)
                        .arg(&cols)
                        .arg(&c.eps)
                        .arg(&n_i)
                        .launch(LaunchConfig::for_num_elems(n as u32))?;
                }
            } else {
                stream
                    .launch_builder(&k_adan)
                    .arg(&gradients[i])
                    .arg(&model_params[i])
                    .arg(&mut s.m[i])
                    .arg(&mut s.v[i])
                    .arg(&mut s.nsq[i])
                    .arg(&mut s.gprev[i])
                    .arg(&mut updates[i])
                    .arg(&lr_i)
                    .arg(&c.beta1)
                    .arg(&c.beta2)
                    .arg(&c.b3)
                    .arg(&c.eps)
                    .arg(&wd_i)
                    .arg(&bc1)
                    .arg(&bc2)
                    .arg(&bc3)
                    .arg(&first_step)
                    .arg(&c.cautious)
                    .arg(&n_i)
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
        }
    }

    if !s.batched_host.is_empty() {
        stream.memcpy_htod(&s.batched_host, &mut s.batched_desc)?;
        let descriptor_count = s.batched_host.len() as i32;
        unsafe {
            stream
                .launch_builder(&k_batched_adan_rollback)
                .arg(&s.batched_desc)
                .arg(&descriptor_count)
                .launch(LaunchConfig {
                    grid_dim: (batched_blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }

    if s.kink_eff > 0.0 && s.step == 1 && model_params.len() >= 2 {
        let nb = model_params[1].len();
        if model_params[0].len() == nb && nb > 1 {
            let nb_i = nb as i32;
            let lc = LaunchConfig::for_num_elems(nb as u32);
            unsafe {
                if s.kink_lin_eff >= 0.0 {
                    stream
                        .launch_builder(&module.load_function(K_KINK2)?)
                        .arg(&mut updates[1])
                        .arg(&model_params[0])
                        .arg(&model_params[1])
                        .arg(&s.kink_lin_eff)
                        .arg(&s.kink_eff)
                        .arg(&nb_i)
                        .launch(lc)?;
                } else {
                    stream
                        .launch_builder(&module.load_function(K_KINK)?)
                        .arg(&mut updates[1])
                        .arg(&model_params[0])
                        .arg(&model_params[1])
                        .arg(&s.kink_eff)
                        .arg(&nb_i)
                        .launch(lc)?;
                }
            }
        }
    }

    let mut shift_postprocessed = false;
    if s.shift_idx < updates.len()
        && s.roles[s.shift_idx] != Role::Skip
        && s.batches_per_epoch > 0
    {
        let mut amount = 0.0;
        if batch_idx == 0 && s.shift_applied != 0.0 {
            amount -= s.shift_applied;
            s.shift_applied = 0.0;
        }
        if batch_idx + 1 >= s.batches_per_epoch && cycle != 0.0 {
            amount += cycle;
            s.shift_applied = cycle;
        }
        if amount != 0.0 {
            let n = model_params[s.shift_idx].len();
            let n_i = n as i32;
            unsafe {
                stream
                    .launch_builder(&k_offset_rollback)
                    .arg(&mut updates[s.shift_idx])
                    .arg(&mut s.last_delta[s.shift_idx])
                    .arg(&amount)
                    .arg(&first_step)
                    .arg(&n_i)
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
            shift_postprocessed = true;
        }
    }

    unsafe {
        for &i in s.active.iter() {
            if ((i == 1 && kink_applies) || (i == s.shift_idx && shift_amount != 0.0))
                && !shift_postprocessed
            {
                let n = model_params[i].len();
                stream
                    .launch_builder(&k_rollback)
                    .arg(&mut updates[i])
                    .arg(&mut s.last_delta[i])
                    .arg(&first_step)
                    .arg(&(n as i32))
                    .launch(LaunchConfig::for_num_elems(n as u32))?;
            }
        }
    }

    Ok(updates)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let _ = CFG.set(parse_cfg(hyperparameters));
    training_loop(
        challenge,
        save_solution,
        module,
        stream,
        prop,
        optimizer_init_state,
        optimizer_query_at_params,
        optimizer_step,
    )
}
