use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::neuralnet_optimizer::*;

const K_ADAN: &str = "sk_adan_7";                 
const K_ADAN_FS: &str = "sk_adan_fs_7";           
const K_ADAN_NR: &str = "sk_adan_nr_7";           
const K_ADAN_FS_NR: &str = "sk_adan_fs_nr_7";     
const K_ADAN_PRECOND: &str = "sk_adan_precond_7";
const K_ADAN_PRECOND_FS: &str = "sk_adan_precond_fs_7";
const K_ADAN_PRECOND_NR: &str = "sk_adan_precond_nr_7";
const K_ADAN_PRECOND_FS_NR: &str = "sk_adan_precond_fs_nr_7";
const K_KINK: &str = "sk_kink_7";
const K_KINK2: &str = "sk_kink2_7";
const K_OFFSET: &str = "sk_offset_7";
const K_SCALE: &str = "sk_scale_7";
const K_DUAL_AXIS_RMS: &str = "sk_dual_axis_rms_7";
const K_ROLLBACK: &str = "sk_rollback_7";

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
            kink: 1.5,
            kink_lin: -1.0,
            depth_lr: 1.0,
            meanalt: 0.7,
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

fn launch_cfg_f4(n: usize) -> LaunchConfig {
    let n8 = ((n as u32) + 7) / 8;
    if n8 == 0 {
        return LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
    }
    let block: u32 = if n8 <= 32 {
        32
    } else if n8 <= 128 {
        64
    } else if n8 <= 512 {
        128
    } else {
        256
    };
    let grid = (n8 + block - 1) / block;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_cfg_elem(n: usize) -> LaunchConfig {
    let n = n.max(1) as u32;
    let block: u32 = if n <= 32 {
        32
    } else if n <= 128 {
        64
    } else if n <= 512 {
        128
    } else if n <= 4096 {
        256
    } else {
        256
    };
    let grid = (n + block - 1) / block;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

struct State {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    nsq: Vec<CudaSlice<f32>>,
    gprev: Vec<CudaSlice<f32>>,
    last_delta: Vec<CudaSlice<f32>>,
    row_rms: Vec<Option<CudaSlice<f32>>>,
    col_rms: Vec<Option<CudaSlice<f32>>>,
    matrix_dims: Vec<Option<(i32, i32)>>,
    adan_cfg: Vec<LaunchConfig>,
    precond_cfg: Vec<Option<LaunchConfig>>,
    elem_cfg: Vec<LaunchConfig>,
    work_streams: Vec<Arc<CudaStream>>,
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

    let mut row_rms: Vec<Option<CudaSlice<f32>>> = Vec::with_capacity(param_sizes.len());
    let mut col_rms: Vec<Option<CudaSlice<f32>>> = Vec::with_capacity(param_sizes.len());
    let mut matrix_dims: Vec<Option<(i32, i32)>> = Vec::with_capacity(param_sizes.len());
    for i in 0..param_sizes.len() {
        let dims = if matches!(roles[i], Role::Weight | Role::HeadWeight)
            && i + 1 < param_sizes.len()
            && matches!(roles[i + 1], Role::Bias | Role::HeadBias)
        {
            let rows = param_sizes[i + 1];
            let n = param_sizes[i];
            if rows > 0 && n % rows == 0 {
                Some((rows as i32, (n / rows) as i32))
            } else {
                None
            }
        } else {
            None
        };
        if let Some((rows, cols)) = dims {
            row_rms.push(Some(stream.alloc_zeros::<f32>(rows as usize)?));
            col_rms.push(Some(stream.alloc_zeros::<f32>(cols as usize)?));
            matrix_dims.push(Some((rows, cols)));
        } else {
            row_rms.push(None);
            col_rms.push(None);
            matrix_dims.push(None);
        }
    }

    let adan_cfg: Vec<LaunchConfig> = (0..param_sizes.len())
        .map(|i| {
            let n = if roles[i] == Role::Skip {
                1
            } else {
                param_sizes[i]
            };
            launch_cfg_f4(n)
        })
        .collect();
    const TILE_R: u32 = 8;
    const TILE_C: u32 = 32;
    let precond_cfg: Vec<Option<LaunchConfig>> = matrix_dims
        .iter()
        .map(|d| {
            d.map(|(rows, cols)| {
                let gr = ((rows as u32) + TILE_R - 1) / TILE_R;
                let gc = ((cols as u32) + TILE_C - 1) / TILE_C;
                LaunchConfig {
                    grid_dim: (gc.max(1), gr.max(1), 1),
                    block_dim: (TILE_C, TILE_R, 1),
                    shared_mem_bytes: 0,
                }
            })
        })
        .collect();
    let elem_cfg: Vec<LaunchConfig> = (0..param_sizes.len())
        .map(|i| {
            let n = if roles[i] == Role::Skip {
                1
            } else {
                param_sizes[i]
            };
            launch_cfg_elem(n)
        })
        .collect();

    const N_WORK_STREAMS: usize = 4;
    let mut work_streams: Vec<Arc<CudaStream>> = Vec::with_capacity(N_WORK_STREAMS);
    work_streams.push(stream.clone());
    let ctx = stream.context();
    for _ in 1..N_WORK_STREAMS {
        work_streams.push(
            ctx.new_stream()
                .map_err(|e| anyhow!("work stream alloc: {e}"))?,
        );
    }

    Ok(Box::new(State {
        m: buf()?,
        v: buf()?,
        nsq: buf()?,
        gprev: buf()?,
        last_delta: buf()?,
        row_rms,
        col_rms,
        matrix_dims,
        adan_cfg,
        precond_cfg,
        elem_cfg,
        work_streams,
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

    let mut updates: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());
    if s.stalled {
        for (i, p) in model_params.iter().enumerate() {
            let n = if s.roles[i] == Role::Skip { 1 } else { p.len() };
            updates.push(stream.alloc_zeros::<f32>(n)?);
        }
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
    for (i, p) in model_params.iter().enumerate() {
        if s.roles[i] == Role::Skip {
            updates.push(stream.alloc_zeros::<f32>(1)?);
        } else {
            updates.push(unsafe { stream.alloc::<f32>(p.len()) }.map_err(|e| anyhow!("alloc: {e}"))?);
        }
    }

    let k_adan = module.load_function(K_ADAN)?;
    let k_adan_fs = module.load_function(K_ADAN_FS)?;
    let k_adan_nr = module.load_function(K_ADAN_NR)?;
    let k_adan_fs_nr = module.load_function(K_ADAN_FS_NR)?;
    let k_adan_precond = module.load_function(K_ADAN_PRECOND)?;
    let k_adan_precond_fs = module.load_function(K_ADAN_PRECOND_FS)?;
    let k_adan_precond_nr = module.load_function(K_ADAN_PRECOND_NR)?;
    let k_adan_precond_fs_nr = module.load_function(K_ADAN_PRECOND_FS_NR)?;
    let k_dual_axis_rms = module.load_function(K_DUAL_AXIS_RMS)?;
    let k_rollback = module.load_function(K_ROLLBACK)?;

    let kink_target: Option<usize> =
        if s.kink_eff > 0.0 && s.step == 1 && model_params.len() >= 2 {
            let nb = model_params[1].len();
            if model_params[0].len() == nb && nb > 1 {
                Some(1)
            } else {
                None
            }
        } else {
            None
        };

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

    let mut offset_amount = 0.0f32;
    if s.shift_idx < model_params.len()
        && s.roles[s.shift_idx] != Role::Skip
        && s.batches_per_epoch > 0
    {
        if batch_idx == 0 && s.shift_applied != 0.0 {
            offset_amount -= s.shift_applied;
            s.shift_applied = 0.0;
        }
        if batch_idx + 1 >= s.batches_per_epoch && cycle != 0.0 {
            offset_amount += cycle;
            s.shift_applied = cycle;
        }
    }
    let offset_target: Option<usize> = if offset_amount != 0.0 {
        Some(s.shift_idx)
    } else {
        None
    };

    let n_ws = s.work_streams.len().max(1);
    let is_fs = first_step != 0;
    for (slot, &i) in s.active.iter().enumerate() {
        let n = model_params[i].len();
        let n_i = n as i32;
        let (lr_i, wd_i) = role_lr_wd(s.roles[i], lr, c.wd, c.head_mult);
        let lr_i = lr_i * s.depth_mult[i];
        let apply_rb = !(Some(i) == kink_target || Some(i) == offset_target);
        let ws = &s.work_streams[slot % n_ws];
        let k_a = match (is_fs, apply_rb) {
            (false, true) => &k_adan,
            (true, true) => &k_adan_fs,
            (false, false) => &k_adan_nr,
            (true, false) => &k_adan_fs_nr,
        };
        let k_ap = match (is_fs, apply_rb) {
            (false, true) => &k_adan_precond,
            (true, true) => &k_adan_precond_fs,
            (false, false) => &k_adan_precond_nr,
            (true, false) => &k_adan_precond_fs_nr,
        };

        unsafe {
            if let (Some((rows, cols)), Some(row_rms), Some(col_rms)) = (
                s.matrix_dims[i],
                s.row_rms[i].as_mut(),
                s.col_rms[i].as_mut(),
            ) {
                ws.launch_builder(&k_dual_axis_rms)
                    .arg(&gradients[i])
                    .arg(&mut *row_rms)
                    .arg(&mut *col_rms)
                    .arg(&rows)
                    .arg(&cols)
                    .launch(LaunchConfig {
                        grid_dim: ((rows + cols) as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
                let pc = s.precond_cfg[i]
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing precond launch cfg"))?;
                ws.launch_builder(k_ap)
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
                    .arg(&c.cautious)
                    .arg(&*row_rms)
                    .arg(&*col_rms)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&mut s.last_delta[i])
                    .launch(*pc)?;
            } else {
                ws.launch_builder(k_a)
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
                    .arg(&c.cautious)
                    .arg(&mut s.last_delta[i])
                    .arg(&n_i)
                    .launch(s.adan_cfg[i])?;
            }
        }
    }

    for ws in s.work_streams.iter() {
        ws.synchronize()
            .map_err(|e| anyhow!("work stream sync: {e}"))?;
    }

    if let Some(1) = kink_target {
        let nb = model_params[1].len();
        let nb_i = nb as i32;
        let lc = s.elem_cfg[1];
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

    if let Some(idx) = offset_target {
        let n = model_params[idx].len();
        let n_i = n as i32;
        unsafe {
            stream
                .launch_builder(&module.load_function(K_OFFSET)?)
                .arg(&mut updates[idx])
                .arg(&offset_amount)
                .arg(&n_i)
                .launch(s.elem_cfg[idx])?;
        }
    }

    unsafe {
        if let Some(idx) = kink_target {
            let n = model_params[idx].len();
            stream
                .launch_builder(&k_rollback)
                .arg(&mut updates[idx])
                .arg(&mut s.last_delta[idx])
                .arg(&first_step)
                .arg(&(n as i32))
                .launch(s.elem_cfg[idx])?;
        }
        if let Some(idx) = offset_target {
            if kink_target != offset_target {
                let n = model_params[idx].len();
                stream
                    .launch_builder(&k_rollback)
                    .arg(&mut updates[idx])
                    .arg(&mut s.last_delta[idx])
                    .arg(&first_step)
                    .arg(&(n as i32))
                    .launch(s.elem_cfg[idx])?;
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
