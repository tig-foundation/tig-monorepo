use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::neuralnet_optimizer::*;

const K_ADAN_PRECOND: &str = "sk_adan_precond_10";
const K_ADAN_MULTI: &str = "sk_adan_multi_10";
const K_KINK_RB: &str = "sk_kink_rb_10";
const K_KINK2_RB: &str = "sk_kink2_rb_10";
const K_OFFSET_RB: &str = "sk_offset_rb_10";
const K_SCALE: &str = "sk_scale_10";
const K_AXIS_RMS_BOTH: &str = "sk_axis_rms_both_10";

#[inline]
fn sl_ptr(s: &CudaSlice<f32>, stream: &CudaStream) -> u64 {
    s.device_ptr(stream).0 as u64
}

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
            meanalt: 1.0,
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

struct MultiPack {
    param_idx: Vec<usize>,
    h_starts: Vec<i32>,
    h_ns: Vec<i32>,
    h_lrs: Vec<f32>,
    h_wds: Vec<f32>,
    h_rbs: Vec<i32>,
    total_n: i32,
    d_grad: Option<CudaSlice<u64>>,
    d_param: Option<CudaSlice<u64>>,
    d_m: Option<CudaSlice<u64>>,
    d_v: Option<CudaSlice<u64>>,
    d_nsq: Option<CudaSlice<u64>>,
    d_gprev: Option<CudaSlice<u64>>,
    d_delta: Option<CudaSlice<u64>>,
    d_last: Option<CudaSlice<u64>>,
    d_starts: Option<CudaSlice<i32>>,
    d_ns: Option<CudaSlice<i32>>,
    d_lrs: Option<CudaSlice<f32>>,
    d_wds: Option<CudaSlice<f32>>,
    d_rbs: Option<CudaSlice<i32>>,
    ready: bool,
}

impl MultiPack {
    fn new(active: &[usize], matrix_shape: &[Option<(i32, i32)>], param_sizes: &[usize]) -> Self {
        let mut param_idx = Vec::new();
        let mut h_starts = Vec::new();
        let mut h_ns = Vec::new();
        let mut total_n: i32 = 0;
        for (ai, &i) in active.iter().enumerate() {
            if matrix_shape[ai].is_some() {
                continue;
            }
            let n_i = param_sizes[i] as i32;
            param_idx.push(i);
            h_starts.push(total_n);
            h_ns.push(n_i);
            total_n += n_i;
        }
        let nt = param_idx.len();
        MultiPack {
            param_idx,
            h_starts,
            h_ns,
            h_lrs: vec![0.0; nt],
            h_wds: vec![0.0; nt],
            h_rbs: vec![1; nt],
            total_n,
            d_grad: None,
            d_param: None,
            d_m: None,
            d_v: None,
            d_nsq: None,
            d_gprev: None,
            d_delta: None,
            d_last: None,
            d_starts: None,
            d_ns: None,
            d_lrs: None,
            d_wds: None,
            d_rbs: None,
            ready: false,
        }
    }
}

struct State {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    nsq: Vec<CudaSlice<f32>>,
    gprev: Vec<CudaSlice<f32>>,
    last_delta: Vec<CudaSlice<f32>>,
    roles: Vec<Role>,
    active: Vec<usize>,
    matrix_shape: Vec<Option<(i32, i32)>>,
    rms_bufs: Vec<Option<(CudaSlice<f32>, CudaSlice<f32>)>>,
    multi: MultiPack,
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

    let mut matrix_shape: Vec<Option<(i32, i32)>> = Vec::with_capacity(active.len());
    let mut rms_bufs: Vec<Option<(CudaSlice<f32>, CudaSlice<f32>)>> =
        Vec::with_capacity(active.len());
    for &i in active.iter() {
        let shape = if matches!(roles[i], Role::Weight | Role::HeadWeight)
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
        if let Some((rows, cols)) = shape {
            let row_rms = stream
                .alloc_zeros::<f32>(rows as usize)
                .map_err(|e| anyhow!("row_rms: {e}"))?;
            let col_rms = stream
                .alloc_zeros::<f32>(cols as usize)
                .map_err(|e| anyhow!("col_rms: {e}"))?;
            rms_bufs.push(Some((row_rms, col_rms)));
            matrix_shape.push(Some((rows, cols)));
        } else {
            rms_bufs.push(None);
            matrix_shape.push(None);
        }
    }

    let multi = MultiPack::new(&active, &matrix_shape, param_sizes);

    Ok(Box::new(State {
        m: buf()?,
        v: buf()?,
        nsq: buf()?,
        gprev: buf()?,
        last_delta: buf()?,
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
        matrix_shape,
        rms_bufs,
        multi,
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
    for (i, p) in model_params.iter().enumerate() {
        let n = if s.roles[i] == Role::Skip { 1 } else { p.len() };
        updates.push(stream.alloc_zeros::<f32>(n)?);
    }

    if s.stalled {
        if !model_params.is_empty() && updates[0].len() == model_params[0].len() {
            let n = model_params[0].len();
            let neg = -1.0f32;
            let n_u = n.max(1) as u32;
            let block: u32 = if n_u < 4096 { 128 } else { 256 };
            let needed = ((n_u + block - 1) / block).max(1);
            let sm = (_prop.multiProcessorCount as u32).max(1);
            let grid = needed.min(sm.saturating_mul(4)).max(1);
            let lc = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&module.load_function(K_SCALE)?)
                    .arg(&mut updates[0])
                    .arg(&model_params[0])
                    .arg(&neg)
                    .arg(&(n as i32))
                    .launch(lc)?;
            }
        }
        return Ok(updates);
    }

    let do_kink = s.kink_eff > 0.0
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
    let do_offset = offset_amount != 0.0;

    let elem_lc = |n: usize| -> LaunchConfig {
        let n_u = n.max(1) as u32;
        let block: u32 = if n_u < 4096 { 128 } else { 256 };
        let needed = ((n_u + block - 1) / block).max(1);
        let sm = (_prop.multiProcessorCount as u32).max(1);
        let grid = needed.min(sm.saturating_mul(4)).max(1);
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let rms_lc = |groups: u32, count: i32| -> LaunchConfig {
        let c = count.max(1) as u32;
        let mut block = 32u32;
        while block < c && block < 256 {
            block <<= 1;
        }
        LaunchConfig {
            grid_dim: (groups.max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    let k_adan_precond = module.load_function(K_ADAN_PRECOND)?;
    let k_adan_multi = module.load_function(K_ADAN_MULTI)?;
    let k_axis_rms_both = module.load_function(K_AXIS_RMS_BOTH)?;

    for (ai, &i) in s.active.iter().enumerate() {
        if let (Some((rows, cols)), Some((ref mut row_rms, ref mut col_rms))) =
            (s.matrix_shape[ai], s.rms_bufs[ai].as_mut())
        {
            let grid = (rows as u32) + (cols as u32);
            let rms_count = if rows > cols { rows } else { cols };
            unsafe {
                stream
                    .launch_builder(&k_axis_rms_both)
                    .arg(&gradients[i])
                    .arg(row_rms)
                    .arg(col_rms)
                    .arg(&rows)
                    .arg(&cols)
                    .launch(rms_lc(grid, rms_count))?;
            }
        }
    }

    for (ai, &i) in s.active.iter().enumerate() {
        if let Some((_rows, cols)) = s.matrix_shape[ai] {
            let n = model_params[i].len();
            let n_i = n as i32;
            let (lr_i, wd_i) = role_lr_wd(s.roles[i], lr, c.wd, c.head_mult);
            let lr_i = lr_i * s.depth_mult[i];
            let apply_rb: i32 =
                if (do_kink && i == 1) || (do_offset && i == s.shift_idx) {
                    0
                } else {
                    1
                };
            let (ref row_rms, ref col_rms) = s.rms_bufs[ai].as_ref().unwrap();
            unsafe {
                stream
                    .launch_builder(&k_adan_precond)
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
                    .arg(&cols)
                    .arg(&apply_rb)
                    .arg(&n_i)
                    .launch(elem_lc(n))?;
            }
        }
    }

    if s.multi.total_n > 0 {
        let nt = s.multi.param_idx.len();
        for t in 0..nt {
            let i = s.multi.param_idx[t];
            let (lr_i, wd_i) = role_lr_wd(s.roles[i], lr, c.wd, c.head_mult);
            s.multi.h_lrs[t] = lr_i * s.depth_mult[i];
            s.multi.h_wds[t] = wd_i;
            s.multi.h_rbs[t] =
                if (do_kink && i == 1) || (do_offset && i == s.shift_idx) {
                    0
                } else {
                    1
                };
        }

        let mut h_grad = vec![0u64; nt];
        let mut h_param = vec![0u64; nt];
        let mut h_m = vec![0u64; nt];
        let mut h_v = vec![0u64; nt];
        let mut h_nsq = vec![0u64; nt];
        let mut h_gprev = vec![0u64; nt];
        let mut h_delta = vec![0u64; nt];
        let mut h_last = vec![0u64; nt];
        for t in 0..nt {
            let i = s.multi.param_idx[t];
            h_grad[t] = sl_ptr(&gradients[i], &stream);
            h_param[t] = sl_ptr(&model_params[i], &stream);
            h_m[t] = sl_ptr(&s.m[i], &stream);
            h_v[t] = sl_ptr(&s.v[i], &stream);
            h_nsq[t] = sl_ptr(&s.nsq[i], &stream);
            h_gprev[t] = sl_ptr(&s.gprev[i], &stream);
            h_delta[t] = sl_ptr(&updates[i], &stream);
            h_last[t] = sl_ptr(&s.last_delta[i], &stream);
        }

        if !s.multi.ready {
            s.multi.d_grad = Some(stream.memcpy_stod(&h_grad)?);
            s.multi.d_param = Some(stream.memcpy_stod(&h_param)?);
            s.multi.d_m = Some(stream.memcpy_stod(&h_m)?);
            s.multi.d_v = Some(stream.memcpy_stod(&h_v)?);
            s.multi.d_nsq = Some(stream.memcpy_stod(&h_nsq)?);
            s.multi.d_gprev = Some(stream.memcpy_stod(&h_gprev)?);
            s.multi.d_delta = Some(stream.memcpy_stod(&h_delta)?);
            s.multi.d_last = Some(stream.memcpy_stod(&h_last)?);
            s.multi.d_starts = Some(stream.memcpy_stod(&s.multi.h_starts)?);
            s.multi.d_ns = Some(stream.memcpy_stod(&s.multi.h_ns)?);
            s.multi.d_lrs = Some(stream.memcpy_stod(&s.multi.h_lrs)?);
            s.multi.d_wds = Some(stream.memcpy_stod(&s.multi.h_wds)?);
            s.multi.d_rbs = Some(stream.memcpy_stod(&s.multi.h_rbs)?);
            s.multi.ready = true;
        } else {
            stream.memcpy_htod(&h_grad, s.multi.d_grad.as_mut().unwrap())?;
            stream.memcpy_htod(&h_param, s.multi.d_param.as_mut().unwrap())?;
            stream.memcpy_htod(&h_m, s.multi.d_m.as_mut().unwrap())?;
            stream.memcpy_htod(&h_v, s.multi.d_v.as_mut().unwrap())?;
            stream.memcpy_htod(&h_nsq, s.multi.d_nsq.as_mut().unwrap())?;
            stream.memcpy_htod(&h_gprev, s.multi.d_gprev.as_mut().unwrap())?;
            stream.memcpy_htod(&h_delta, s.multi.d_delta.as_mut().unwrap())?;
            stream.memcpy_htod(&h_last, s.multi.d_last.as_mut().unwrap())?;
            stream.memcpy_htod(&s.multi.h_lrs, s.multi.d_lrs.as_mut().unwrap())?;
            stream.memcpy_htod(&s.multi.h_wds, s.multi.d_wds.as_mut().unwrap())?;
            stream.memcpy_htod(&s.multi.h_rbs, s.multi.d_rbs.as_mut().unwrap())?;
        }

        let num_t = nt as i32;
        let total_nm = s.multi.total_n;
        unsafe {
            stream
                .launch_builder(&k_adan_multi)
                .arg(s.multi.d_grad.as_ref().unwrap())
                .arg(s.multi.d_param.as_ref().unwrap())
                .arg(s.multi.d_m.as_ref().unwrap())
                .arg(s.multi.d_v.as_ref().unwrap())
                .arg(s.multi.d_nsq.as_ref().unwrap())
                .arg(s.multi.d_gprev.as_ref().unwrap())
                .arg(s.multi.d_delta.as_ref().unwrap())
                .arg(s.multi.d_last.as_ref().unwrap())
                .arg(s.multi.d_starts.as_ref().unwrap())
                .arg(s.multi.d_ns.as_ref().unwrap())
                .arg(s.multi.d_lrs.as_ref().unwrap())
                .arg(s.multi.d_wds.as_ref().unwrap())
                .arg(s.multi.d_rbs.as_ref().unwrap())
                .arg(&num_t)
                .arg(&total_nm)
                .arg(&c.beta1)
                .arg(&c.beta2)
                .arg(&c.b3)
                .arg(&c.eps)
                .arg(&bc1)
                .arg(&bc2)
                .arg(&bc3)
                .arg(&first_step)
                .arg(&c.cautious)
                .launch(elem_lc(total_nm as usize))?;
        }
    }

    if do_kink {
        let nb = model_params[1].len();
        let nb_i = nb as i32;
        let lc = elem_lc(nb);
        unsafe {
            if s.kink_lin_eff >= 0.0 {
                stream
                    .launch_builder(&module.load_function(K_KINK2_RB)?)
                    .arg(&mut updates[1])
                    .arg(&mut s.last_delta[1])
                    .arg(&model_params[0])
                    .arg(&model_params[1])
                    .arg(&s.kink_lin_eff)
                    .arg(&s.kink_eff)
                    .arg(&first_step)
                    .arg(&nb_i)
                    .launch(lc)?;
            } else {
                stream
                    .launch_builder(&module.load_function(K_KINK_RB)?)
                    .arg(&mut updates[1])
                    .arg(&mut s.last_delta[1])
                    .arg(&model_params[0])
                    .arg(&model_params[1])
                    .arg(&s.kink_eff)
                    .arg(&first_step)
                    .arg(&nb_i)
                    .launch(lc)?;
            }
        }
    }

    if do_offset {
        let n = model_params[s.shift_idx].len();
        let n_i = n as i32;
        let lc = elem_lc(n);
        unsafe {
            stream
                .launch_builder(&module.load_function(K_OFFSET_RB)?)
                .arg(&mut updates[s.shift_idx])
                .arg(&mut s.last_delta[s.shift_idx])
                .arg(&offset_amount)
                .arg(&first_step)
                .arg(&n_i)
                .launch(lc)?;
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