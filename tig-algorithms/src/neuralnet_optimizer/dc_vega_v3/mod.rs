// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
//
// dc_vega_v3: c006 neuralnet_optimizer.

use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;

// Set by the runtime to the fuel cap and decremented by the instrumentation.
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
    depth_lr: f32,
    fuel_reserve: usize,
    kink_set: bool,
    lr_set: bool,
    dc_start: usize,
    dc_refresh: usize,
    dc_max_abs: f32,
    dc_gain: f32,
    ab_blend: f32,
    wd_gate: i32,
    dc_gain_set: bool,
    save_eager: usize,
    bn_w: f32,
    bn_b: f32,
    // dc_vega_v3: bit l freezes the WEIGHT update of linear layer l (parameter index 2l); u64::MAX = per-depth default
    // (freeze_default): lin1 (bit 1) frozen where it measured Pareto-better than the unfrozen vega, 0 elsewhere.
    freeze_mask: u64,
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
            depth_lr: 1.0,
            fuel_reserve: 55,
            kink_set: false,
            lr_set: false,
            dc_start: 20,
            dc_refresh: 8,
            dc_max_abs: 2.0,
            dc_gain: 1.0,
            ab_blend: 1.0,
            wd_gate: 0,
            dc_gain_set: false,
            save_eager: 1,
            bn_w: 0.55,
            bn_b: 0.80,
            freeze_mask: u64::MAX,
        }
    }
}

thread_local! {
    static CFG: std::cell::RefCell<Cfg> = std::cell::RefCell::new(Cfg::default());
    static TARGET_MEAN: std::cell::Cell<[f64; 2]> = std::cell::Cell::new([0.0; 2]);
}

fn cfg() -> Cfg {
    CFG.with(|c| c.borrow().clone())
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
    c.depth_lr = f("depth_lr", c.depth_lr);
    c.fuel_reserve = u("fuel_reserve", c.fuel_reserve);
    c.kink_set = m.contains_key("kink");
    c.lr_set = m.contains_key("lr_max");
    c.dc_start = u("dc_start", c.dc_start);
    c.dc_refresh = u("dc_refresh", c.dc_refresh).max(1);
    c.dc_max_abs = f("dc_max_abs", c.dc_max_abs);
    c.dc_gain = f("dc_gain", c.dc_gain);
    c.ab_blend = f("ab_blend", c.ab_blend);
    c.wd_gate = m.get("wd_gate").and_then(|v| v.as_u64()).map(|v| v as i32).unwrap_or(c.wd_gate);
    c.dc_gain_set = m.contains_key("dc_gain");
    c.save_eager = u("save_eager", c.save_eager);
    c.bn_w = f("bn_w", c.bn_w);
    c.bn_b = f("bn_b", c.bn_b);
    c.freeze_mask = m.get("freeze_mask").and_then(|v| v.as_u64()).unwrap_or(c.freeze_mask);
    c
}

// Per-depth freeze default, keyed on the depth derived from the parameter layout (never on track_id).
// Measured same-batch vs parallax_vega (EXP-126/134, 8 bundles x 8 nonces): nh7 +2,698 at rt 0.868, nh18 +5,090 at rt 0.990
// (lin1 frozen = Pareto-better); nh4 -3,851 (frozen lin1 loses on the shallow net; mechanism not established). nh10 +2,098 at
// rt 0.942, nh14 +6,388 at rt 0.945 (EXP-134). Depths without a measurement stay unfrozen (bit-identical to parallax_vega).
fn freeze_default(n_hidden: usize) -> u64 {
    match n_hidden {
        7 => 2,
        10 => 2,
        14 => 2,
        18 => 2,
        _ => 0,
    }
}

fn target_mean() -> [f64; 2] {
    TARGET_MEAN.with(|t| t.get())
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

// Layout: 2L linear parameters, then L-1 BatchNorm groups of four (weight,
// bias, running mean, running var). The last two layers are frozen.
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

fn role_lr_wd(role: Role, lr: f32, wd: f32, c: &Cfg) -> (f32, f32) {
    match role {
        Role::Weight => (lr, wd),
        Role::Bias => (lr * 1.25, 0.0),
        Role::HeadWeight => (lr * c.head_mult, wd),
        Role::HeadBias => (lr * c.head_mult * 1.25, 0.0),
        Role::BnWeight => (lr * c.bn_w, 0.0),
        Role::BnBias => (lr * c.bn_b, 0.0),
        Role::Skip => (0.0, 0.0),
    }
}

// v1s (consult #15 L1): persistent dispatch metadata. Seven pointer planes (grad, param, delta are refreshed every
// step; m, v, nsq, gprev are invariant), per-epoch LR/WD tables, and the fused kernel handle loaded once.
struct FusedCache {
    ptrs_h: Vec<u64>,
    pg: CudaSlice<u64>, pp: CudaSlice<u64>, pd: CudaSlice<u64>,
    pm: CudaSlice<u64>, pv: CudaSlice<u64>, pn: CudaSlice<u64>, pgp: CudaSlice<u64>,
    lrs_h: Vec<f32>, wds_h: Vec<f32>,
    d_lr: CudaSlice<f32>, d_wd: CudaSlice<f32>,
    scalar_epoch: Option<usize>,
    kernel: CudaFunction,
}

struct State {
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    nsq: Vec<CudaSlice<f32>>,
    gprev: Vec<CudaSlice<f32>>,
    roles: Vec<Role>,
    active: Vec<usize>,
    depth_mult: Vec<f32>,
    kink_eff: f32,
    lr_eff: f32,
    step: usize,
    epoch: usize,
    epoch_seen: bool,
    best_val: f32,
    plateau_stale: usize,
    lr_scale: f32,
    batch_in_epoch: usize,
    batches_per_epoch: usize,
    shift_idx: usize,
    epoch_cap: usize,
    stalled: bool,
    dc_idx: Option<DcIdx>,
    dc_w3: Vec<f32>,
    dc_w4: Vec<f32>,
    dc_frozen_read: bool,
    dc_dev: Option<CudaSlice<f32>>,
    dc_ready: bool,
    dc_applied: f32,
    dc_next_refresh: usize,
    dc_gain_eff: f32,
    fz_prefix: Option<CudaSlice<i32>>,
    fz_total: usize,
    fz_cache: Option<FusedCache>,
    epoch_lr: f32,
    freeze_eff: u64,
}

#[derive(Clone, Copy)]
struct DcIdx {
    bias: usize,
    w3: usize,
    w4: usize,
    rv: usize,
}

// Indices of the four tensors the offset reads or writes.
fn dc_indices(param_count: usize) -> Option<DcIdx> {
    if param_count < 8 || (param_count + 4) % 6 != 0 {
        return None;
    }
    let l = (param_count + 4) / 6;
    if l < 4 {
        return None;
    }
    let base = 2 * l;
    let bn_frozen = l - 2;
    let bn_shifted = l - 3;
    Some(DcIdx {
        bias: base + 4 * bn_shifted + 1,
        w3: 2 * (l - 2),
        w4: 2 * (l - 1),
        rv: base + 4 * bn_frozen + 3,
    })
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

    // The harness never reads updates for skipped roles.
    let alloc = |i: usize| -> Result<CudaSlice<f32>> {
        if roles[i] == Role::Skip {
            unsafe { stream.alloc::<f32>(1) }.map_err(|e| anyhow!("alloc: {e}"))
        } else {
            stream.alloc_zeros::<f32>(param_sizes[i]).map_err(|e| anyhow!("alloc: {e}"))
        }
    };
    let buf = || -> Result<Vec<CudaSlice<f32>>> { (0..param_sizes.len()).map(alloc).collect() };

    let l = (param_sizes.len() + 4) / 6;
    let n_hidden = l.saturating_sub(1);
    let n_trainable = l.saturating_sub(2).max(1);

    Ok(Box::new(State {
        m: buf()?,
        v: buf()?,
        nsq: buf()?,
        gprev: buf()?,
        // Depth is read from param_sizes, not track_id, which TIG may change.
        kink_eff: if c.kink_set { c.kink } else { 1.5 },
        lr_eff: if c.lr_set || n_hidden < 18 {
            c.lr_max
        } else {
            1.0e-3
        },
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
        // Bias of the last trainable BatchNorm.
        shift_idx: 2 * l + 4 * n_trainable.saturating_sub(1) + 1,
        dc_idx: dc_indices(param_sizes.len()),
        dc_w3: Vec::new(),
        dc_w4: Vec::new(),
        dc_frozen_read: false,
        dc_dev: None,
        dc_ready: false,
        dc_applied: 0.0,
        dc_next_refresh: 0,
        dc_gain_eff: if c.dc_gain_set {
            c.dc_gain
        } else if n_hidden == 14 {
            1.0
        } else {
            1.2
        },
        fz_prefix: None,
        fz_total: 0,
        fz_cache: None,
        epoch_lr: 0.0,
        freeze_eff: if c.freeze_mask == u64::MAX { freeze_default(n_hidden) } else { c.freeze_mask },
        epoch_cap: {
            let sum_w: usize = (0..l).map(|i| param_sizes[2 * i]).sum();
            let per_epoch = 870u64 * sum_w as u64;
            let affordable = (fuel_remaining() / per_epoch.max(1)) as usize;
            if affordable <= c.fuel_reserve {
                usize::MAX
            } else {
                affordable - c.fuel_reserve
            }
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

// Solved in f64 in a fixed operation order.
fn dc_solve(s: &mut State, c: &Cfg, model_params: &[CudaSlice<f32>], stream: &Arc<CudaStream>) -> Result<bool> {
    const OD: usize = 2;
    const BN_EPS: f64 = 1e-5;
    let Some(ix) = s.dc_idx else { return Ok(false) };
    if ix.rv >= model_params.len() {
        return Ok(false);
    }
    let h = model_params[ix.bias].len();
    if h == 0
        || model_params[ix.w3].len() != h * h
        || model_params[ix.w4].len() != OD * h
        || model_params[ix.rv].len() != h
    {
        return Ok(false);
    }

    if !s.dc_frozen_read {
        s.dc_w3 = stream.memcpy_dtov(&model_params[ix.w3])?;
        s.dc_w4 = stream.memcpy_dtov(&model_params[ix.w4])?;
        stream.synchronize()?;
        s.dc_frozen_read = true;
    }
    let rv = stream.memcpy_dtov(&model_params[ix.rv])?;
    stream.synchronize()?;

    let mut c0 = vec![0.0f64; h];
    let mut c1 = vec![0.0f64; h];
    for i in 0..h {
        let sig = ((rv[i] as f64) + BN_EPS).max(1e-12).sqrt();
        let p = 0.5;
        c0[i] = (s.dc_w4[i * OD] as f64) * p / sig;
        c1[i] = (s.dc_w4[i * OD + 1] as f64) * p / sig;
    }

    let mut a0 = vec![0.0f64; h];
    let mut a1 = vec![0.0f64; h];
    for k in 0..h {
        let base = k * h;
        let mut t0 = 0.0f64;
        let mut t1 = 0.0f64;
        for i in 0..h {
            let w = s.dc_w3[base + i] as f64;
            t0 += c0[i] * w;
            t1 += c1[i] * w;
        }
        a0[k] = t0;
        a1[k] = t1;
    }

    let (mut g00, mut g01, mut g11) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..h {
        g00 += a0[k] * a0[k];
        g01 += a0[k] * a1[k];
        g11 += a1[k] * a1[k];
    }
    let lam = 1e-6 * (g00 + g11) + 1e-30;
    let (g00r, g11r) = (g00 + lam, g11 + lam);
    let det = g00r * g11r - g01 * g01;
    if !det.is_finite() || det.abs() < 1e-30 {
        return Ok(false);
    }
    let m = target_mean();
    let y0 = (g11r * m[0] - g01 * m[1]) / det;
    let y1 = (-g01 * m[0] + g00r * m[1]) / det;

    let mut d = vec![0.0f32; h];
    let mut mx = 0.0f64;
    for k in 0..h {
        let v = a0[k] * y0 + a1[k] * y1;
        if !v.is_finite() {
            return Ok(false);
        }
        mx = mx.max(v.abs());
        d[k] = v as f32;
    }
    if mx > c.dc_max_abs as f64 && mx > 0.0 {
        let sc = (c.dc_max_abs as f64) / mx;
        for x in d.iter_mut() {
            *x = ((*x as f64) * sc) as f32;
        }
    }

    if s.dc_dev.as_ref().map(|b| b.len()) != Some(h) {
        s.dc_dev = Some(unsafe { stream.alloc::<f32>(h)? }); // fully overwritten by the memcpy below
    }
    if let Some(buf) = s.dc_dev.as_mut() {
        stream.memcpy_htod(&d[..], buf)?;
    }
    s.dc_ready = true;
    Ok(true)
}

fn lr_at(s: &State, c: &Cfg, epoch: usize) -> f32 {
    if epoch < c.warmup_epochs {
        return s.lr_eff * ((epoch + 1) as f32) / (c.warmup_epochs.max(1) as f32);
    }
    let denom = c.t_max.saturating_sub(c.warmup_epochs).max(1) as f32;
    let p = (epoch as f32 / denom).min(1.0);
    c.lr_min + 0.5 * (s.lr_eff - c.lr_min) * (1.0 + (std::f32::consts::PI * p).cos())
}

// 256-thread blocks for the 256-element vector kernels (element-wise; the 1,024-thread default spawned 768 idle threads).
const SM_REF: u32 = 82; // RTX3090 SM count, pinned (grid sizing only)

fn vec_launch(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (((n as u32 + 255) / 256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
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
    if new_epoch && (epoch >= s.epoch_cap) {
        s.stalled = true;
    }

    let batch_idx = s.batch_in_epoch;
    s.batch_in_epoch += 1;
    s.step += 1;

    if new_epoch {
        s.epoch_lr = lr_at(s, &c, epoch) * s.lr_scale;
    }
    let lr = s.epoch_lr;
    const GSIGN: i32 = 0;

    let bc1 = 1.0 - c.beta1.powi(s.step as i32);
    let bc2 = 1.0 - c.beta2.powi(s.step as i32);
    let bc3 = 1.0 - c.b3.powi(s.step as i32);
    let first_step: i32 = if s.step == 1 { 1 } else { 0 };

    let mut updates: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());
    for (i, p) in model_params.iter().enumerate() {
        if s.roles[i] == Role::Skip {
            updates.push(unsafe { stream.alloc::<f32>(1)? }); // never read by the harness
        } else if s.stalled {
            updates.push(stream.alloc_zeros::<f32>(p.len())?);
        } else {
            updates.push(unsafe { stream.alloc::<f32>(p.len())? });
        }
    }

    // Late path; the checkpointed best model is untouched.
    if s.stalled {
        if !model_params.is_empty() && updates[0].len() == model_params[0].len() {
            let n = model_params[0].len();
            let neg = -1.0f32;
            unsafe {
                stream
                    .launch_builder(&module.load_function("sk_scale")?)
                    .arg(&mut updates[0])
                    .arg(&model_params[0])
                    .arg(&neg)
                    .arg(&(n as i32))
                    .launch(vec_launch(n))?;
            }
        }
        return Ok(updates);
    }

    if !s.active.is_empty() {
        let t = s.active.len();
        if s.fz_prefix.is_none() {
            let mut pre = Vec::with_capacity(t + 1);
            let mut acc = 0i32;
            for &i in s.active.iter() {
                pre.push(acc);
                acc += model_params[i].len() as i32;
            }
            pre.push(acc);
            s.fz_total = acc as usize;
            s.fz_prefix = Some(stream.memcpy_stod(&pre)?);
        }
        let ptr = |v: &[CudaSlice<f32>], idx: &[usize]| -> Vec<u64> {
            idx.iter()
                .map(|&i| {
                    let (p, _guard) = v[i].device_ptr(&stream);
                    p
                })
                .collect()
        };
        if s.fz_cache.is_none() {
            // Invariant planes uploaded once; the same values the per-step upload produced before.
            s.fz_cache = Some(FusedCache {
                ptrs_h: vec![0u64; t],
                pg: unsafe { stream.alloc::<u64>(t)? }, pp: unsafe { stream.alloc::<u64>(t)? }, pd: unsafe { stream.alloc::<u64>(t)? },
                pm: stream.memcpy_stod(&ptr(&s.m, &s.active))?,
                pv: stream.memcpy_stod(&ptr(&s.v, &s.active))?,
                pn: stream.memcpy_stod(&ptr(&s.nsq, &s.active))?,
                pgp: stream.memcpy_stod(&ptr(&s.gprev, &s.active))?,
                lrs_h: vec![0.0f32; t], wds_h: vec![0.0f32; t],
                d_lr: unsafe { stream.alloc::<f32>(t)? }, d_wd: unsafe { stream.alloc::<f32>(t)? },
                scalar_epoch: None,
                kernel: module.load_function("sk_adan_fused")?,
            });
        }
        let cache = s.fz_cache.as_mut().unwrap();
        // Dynamic planes: fresh gradient / parameter snapshots and the freshly allocated updates, every step.
        for (j, &i) in s.active.iter().enumerate() { cache.ptrs_h[j] = { let (p, _g) = gradients[i].device_ptr(&stream); p }; }
        stream.memcpy_htod(&cache.ptrs_h, &mut cache.pg)?;
        for (j, &i) in s.active.iter().enumerate() { cache.ptrs_h[j] = { let (p, _g) = model_params[i].device_ptr(&stream); p }; }
        stream.memcpy_htod(&cache.ptrs_h, &mut cache.pp)?;
        for (j, &i) in s.active.iter().enumerate() { cache.ptrs_h[j] = { let (p, _g) = updates[i].device_ptr(&stream); p }; }
        stream.memcpy_htod(&cache.ptrs_h, &mut cache.pd)?;
        // LR / WD tables change only with the epoch (lr and lr_scale are per-epoch values).
        if cache.scalar_epoch != Some(epoch) {
            for (j, &i) in s.active.iter().enumerate() {
                // Frozen weight: lr 0 and wd 0 -> sk_adan_fused writes delta = 0.0 exactly for that tensor (element-wise, no
                // cross-tensor reduction); the moments still advance. With freeze_eff == 0 this branch never runs.
                if matches!(s.roles[i], Role::Weight | Role::HeadWeight) && ((s.freeze_eff >> (i / 2)) & 1) == 1 {
                    cache.lrs_h[j] = 0.0;
                    cache.wds_h[j] = 0.0;
                    continue;
                }
                let (l, w) = role_lr_wd(s.roles[i], lr, c.wd, &c);
                cache.lrs_h[j] = l * s.depth_mult[i];
                cache.wds_h[j] = w;
            }
            stream.memcpy_htod(&cache.lrs_h, &mut cache.d_lr)?;
            stream.memcpy_htod(&cache.wds_h, &mut cache.d_wd)?;
            cache.scalar_epoch = Some(epoch);
        }
        let total = s.fz_total as i32;
        let nt = t as i32;
        unsafe {
            stream
                .launch_builder(&cache.kernel)
                .arg(&cache.pg).arg(&cache.pp).arg(&cache.pm).arg(&cache.pv).arg(&cache.pn).arg(&cache.pgp).arg(&cache.pd)
                .arg(s.fz_prefix.as_ref().unwrap())
                .arg(&cache.d_lr).arg(&cache.d_wd).arg(&nt)
                .arg(&c.beta1).arg(&c.beta2).arg(&c.b3).arg(&c.eps)
                .arg(&bc1).arg(&bc2).arg(&bc3)
                .arg(&first_step).arg(&c.cautious).arg(&c.ab_blend).arg(&c.wd_gate).arg(&GSIGN).arg(&total)
                .launch(LaunchConfig {
                    // v1g: 256-thread blocks, grid capped at SM_REF blocks (architecture-invariant grid; per-element
                    // results do not depend on the grid because elements are independent).
                    grid_dim: (((s.fz_total as u32 + 255) / 256).min(SM_REF), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
    }

    // Guarded on param_sizes so the stage disables itself if the input
    // dimension ever changes.
    if s.kink_eff > 0.0 && s.step == 1 && model_params.len() >= 2 {
        let nb = model_params[1].len();
        if model_params[0].len() == nb && nb > 1 {
            let nb_i = nb as i32;
            let lc = vec_launch(nb);
            unsafe {
                stream
                    .launch_builder(&module.load_function("sk_kink")?)
                    .arg(&mut updates[1])
                    .arg(&model_params[0])
                    .arg(&model_params[1])
                    .arg(&s.kink_eff)
                    .arg(&nb_i)
                    .launch(lc)?;
            }
        }
    }

    if s.dc_idx.is_some() && s.batches_per_epoch > 0 {
        let n = model_params[s.shift_idx].len();
        let lc = vec_launch(n);

        if batch_idx == 0 && s.dc_applied != 0.0 {
            let g = -s.dc_applied;
            s.dc_applied = 0.0;
            if let Some(buf) = s.dc_dev.as_ref() {
                unsafe {
                    stream
                        .launch_builder(&module.load_function("sk_offset_vec")?)
                        .arg(&mut updates[s.shift_idx])
                        .arg(buf)
                        .arg(&g)
                        .arg(&(n as i32))
                        .launch(lc)?;
                }
            }
        }

        if batch_idx == 0 && epoch + 1 >= c.dc_start && epoch >= s.dc_next_refresh {
            s.dc_next_refresh = epoch + c.dc_refresh;
            let _ = dc_solve(s, &c, model_params, &stream)?;
        }

        if batch_idx + 1 >= s.batches_per_epoch && epoch >= c.dc_start && s.dc_ready {
            let g = s.dc_gain_eff;
            if g != 0.0 {
                if let Some(buf) = s.dc_dev.as_ref() {
                    unsafe {
                        stream
                            .launch_builder(&module.load_function("sk_offset_vec")?)
                            .arg(&mut updates[s.shift_idx])
                            .arg(buf)
                            .arg(&g)
                            .arg(&(n as i32))
                            .launch(lc)?;
                    }
                }
                s.dc_applied = g;
            }
        }
        return Ok(updates);
    }

    Ok(updates)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let parsed = parse_cfg(hyperparameters);
    CFG.with(|c| *c.borrow_mut() = parsed);

    // All work runs on the stream the runtime provides; no buffer is shared and
    // no free is concurrent, so the contract of disable_event_tracking holds.
    unsafe {
        stream.context().disable_event_tracking();
    }

    // Changes no device state, so it cannot change a result.
    let _ = stream
        .context()
        .set_flags(cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_SPIN);

    {
        let od = challenge.dataset.output_dims;
        let n_tr = challenge.dataset.train_size;
        let mut mean = [0.0f64; 2];
        if od >= 1 && n_tr > 0 {
            let t = stream.memcpy_dtov(&challenge.dataset.train_targets_noisy())?;
            stream.synchronize()?;
            let lim = od.min(2);
            let mut acc = [0.0f64; 2];
            for i in 0..n_tr {
                for j in 0..lim {
                    acc[j] += t[i * od + j] as f64;
                }
            }
            for j in 0..2 {
                mean[j] = acc[j] / n_tr as f64;
            }
        }
        TARGET_MEAN.with(|t| t.set(mean));
    }

    // The last save is the one that counts; the first few are kept eager so an
    // interrupted run still returns a valid solution.
    let held: std::cell::RefCell<Option<Solution>> = std::cell::RefCell::new(None);
    let seen = std::cell::Cell::new(0usize);
    let eager = cfg().save_eager;
    let deferred = |sol: &Solution| -> Result<()> {
        let n = seen.get();
        seen.set(n + 1);
        if n < eager {
            return save_solution(sol);
        }
        *held.borrow_mut() = Some(sol.clone());
        Ok(())
    };

    let outcome = training_loop(
        challenge,
        &deferred,
        module,
        stream,
        prop,
        optimizer_init_state,
        optimizer_query_at_params,
        optimizer_step,
    );

    let last = held.borrow_mut().take();
    if let Some(sol) = last {
        save_solution(&sol)?;
    }
    outcome
}

pub fn help() {
    println!("dc_vega_v3: neural network optimizer.");
    println!("Deterministic: no thread, no clock, no filesystem access, no randomness.");
    println!("A valid solution is checkpointed as soon as one exists.");
    println!("Defaults are the intended operating point; every hyperparameter overrides them.");
    println!("HP: dc_start, dc_refresh, dc_gain, dc_max_abs, kink, save_eager, fuel_reserve,");
    println!("    lr_max, lr_min, warmup_epochs, t_max, wd, eps, beta1, beta2, b3, cautious,");
    println!("    plateau_patience, plateau_decay, plateau_grow, plateau_floor, head_mult,");
    println!("    depth_lr, bn_w, bn_b, ab_blend, wd_gate, freeze_mask (default: per-depth, lin1 frozen on n_hidden 7, 10, 14 and 18; 0 on 4)");
}

