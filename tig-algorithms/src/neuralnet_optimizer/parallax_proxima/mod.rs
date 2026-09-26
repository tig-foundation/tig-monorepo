// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
//
// parallax_proxima: c006 neuralnet_optimizer.

use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
mod config;
mod dc;
mod fused;
mod gp;
mod orth;
mod pstop;
mod schedule;
use config::{depth_defaults, parse_cfg, Cfg};
use dc::DcIdx;
use fused::FusedCache;

const VECTOR_THREADS: u32 = 256;
const FUEL_PER_WEIGHT_EPOCH: u64 = 870;

// Set by the runtime to the fuel cap and decremented by the instrumentation.
extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

thread_local! {
    static CFG: RefCell<Cfg> = RefCell::new(Cfg::default());
    static TARGET_MEAN: Cell<[f64; 2]> = Cell::new([0.0; 2]);
    static STOPPED: Cell<bool> = Cell::new(false);
    static REC_DELTA: Cell<f32> = Cell::new(1e-7);
    static PSTOP_PATIENCE: Cell<usize> = Cell::new(50);
}

fn cfg() -> Cfg {
    CFG.with(|c| c.borrow().clone())
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

struct State {
    pstop: Option<pstop::Stopper>,
    m: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    nsq: Vec<CudaSlice<f32>>,
    gprev: Vec<CudaSlice<f32>>,
    roles: Vec<Role>,
    active: Vec<usize>,
    depth_mult: Vec<f32>,
    warmup_eff: usize,
    step: usize,
    epoch: usize,
    epoch_seen: bool,
    best_val: f32,
    plateau_stale: usize,
    lr_scale: f32,
    batch_in_epoch: usize,
    batches_per_epoch: usize,
    epoch_cap: usize,
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
    stale_total: usize,
    cooling: bool,
    cool_start: usize,
    rec_lowest: f32,
    rec_stale: usize,
    stop_age_eff: usize,
    stop_stale_eff: usize,
    freeze_eff: u64,
    cos_fix_eff: bool,
    cool_stale_eff: usize,
    cool_decay_eff: f32,
    orth_eff: bool,
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

fn depth_multipliers(count: usize, n_trainable: usize, depth_lr: f32) -> Vec<f32> {
    (0..count)
        .map(|i| {
            if depth_lr == 1.0 || i >= 2 * n_trainable {
                1.0
            } else {
                let f = if n_trainable > 1 { (i / 2) as f32 / (n_trainable - 1) as f32 } else { 0.0 };
                depth_lr.powf(f)
            }
        })
        .collect()
}

fn affordable_epochs(param_sizes: &[usize], layers: usize, reserve: usize) -> usize {
    let weights: usize = (0..layers).map(|i| param_sizes[2 * i]).sum();
    let per_epoch = FUEL_PER_WEIGHT_EPOCH * weights as u64;
    let affordable = (fuel_remaining() / per_epoch.max(1)) as usize;
    affordable.saturating_sub(reserve).max(1)
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
    let active: Vec<usize> = (0..param_sizes.len()).filter(|&i| roles[i] != Role::Skip).collect();

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
    let defaults = depth_defaults(n_hidden, c.warmup_epochs);
    let n_trainable = l.saturating_sub(2).max(1);
    let pstop_z = c.pstop_z.unwrap_or(defaults.pstop_z);

    Ok(Box::new(State {
        m: buf()?,
        v: buf()?,
        nsq: buf()?,
        gprev: buf()?,
        warmup_eff: c.warmup_epochs.unwrap_or(defaults.warmup_epochs),
        depth_mult: depth_multipliers(param_sizes.len(), n_trainable, c.depth_lr),
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
        dc_idx: dc::indices(param_sizes.len()),
        dc_w3: Vec::new(),
        dc_w4: Vec::new(),
        dc_frozen_read: false,
        dc_dev: None,
        dc_ready: false,
        dc_applied: 0.0,
        dc_next_refresh: 0,
        dc_gain_eff: c.dc_gain.unwrap_or(defaults.dc_gain),
        fz_prefix: None,
        fz_total: 0,
        fz_cache: None,
        stale_total: 0,
        cooling: false,
        cool_start: 0,
        rec_lowest: f32::INFINITY,
        rec_stale: 0,
        stop_age_eff: c.stop_age.unwrap_or(defaults.stop_age),
        stop_stale_eff: c.stop_stale.unwrap_or(defaults.stop_stale),
        freeze_eff: c.freeze_mask.unwrap_or(defaults.freeze_mask),
        cos_fix_eff: c.cos_fix.unwrap_or(defaults.cos_fix),
        cool_stale_eff: c.cool_stale.unwrap_or(defaults.cool_stale),
        cool_decay_eff: c.cool_decay.unwrap_or(defaults.cool_decay),
        orth_eff: c.orth.unwrap_or(defaults.orth),
        epoch_cap: affordable_epochs(param_sizes, l, c.fuel_reserve),
        pstop: (pstop_z > 0.0).then(|| {
            pstop::Stopper::new(c.pstop_w, c.pstop_s0, pstop_z as f64, PSTOP_PATIENCE.with(|p| p.get()))
        }),
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

fn vec_launch(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32 + (VECTOR_THREADS - 1)) / VECTOR_THREADS, 1, 1),
        block_dim: (VECTOR_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn spread_first_biases(
    c: &Cfg, model_params: &[CudaSlice<f32>], updates: &mut [CudaSlice<f32>],
    stream: &Arc<CudaStream>, module: &Arc<CudaModule>,
) -> Result<()> {
    if c.kink <= 0.0 || model_params.len() < 2 {
        return Ok(());
    }
    let nb = model_params[1].len();
    if model_params[0].len() != nb || nb <= 1 {
        return Ok(());
    }
    let nb_i = nb as i32;
    unsafe {
        stream
            .launch_builder(&module.load_function("sk_kink")?)
            .arg(&mut updates[1])
            .arg(&model_params[0])
            .arg(&model_params[1])
            .arg(&c.kink)
            .arg(&nb_i)
            .launch(vec_launch(nb))?;
    }
    Ok(())
}

pub fn help() {
    config::help();
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
        schedule::advance_epoch(s, &c, epoch, val_loss)?;
        if epoch >= s.epoch_cap {
            return schedule::stop();
        }
    }

    let batch_idx = s.batch_in_epoch;
    s.batch_in_epoch += 1;
    s.step += 1;

    let lr = schedule::lr_at(s, &c, epoch) * s.lr_scale;

    let bc1 = 1.0 - c.beta1.powi(s.step as i32);
    let bc2 = 1.0 - c.beta2.powi(s.step as i32);
    let bc3 = 1.0 - c.b3.powi(s.step as i32);
    let first_step: i32 = if s.step == 1 { 1 } else { 0 };

    let mut updates: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());
    for (i, p) in model_params.iter().enumerate() {
        let len = if s.roles[i] == Role::Skip { 1 } else { p.len() };
        updates.push(unsafe { stream.alloc::<f32>(len)? });
    }

    fused::dispatch(s, &c, model_params, gradients, &mut updates, &stream, &module,
        lr, bc1, bc2, bc3, first_step)?;

    if s.step == 1 {
        spread_first_biases(&c, model_params, &mut updates, &stream, &module)?;
        if s.orth_eff {
            orth::apply(s, &c, model_params, &mut updates, &stream, &module)?;
        }
    }

    dc::apply(s, &c, epoch, batch_idx, model_params, &mut updates, &stream, &module)?;
    Ok(updates)
}

fn noisy_target_mean(challenge: &Challenge, stream: &Arc<CudaStream>) -> Result<[f64; 2]> {
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
    Ok(mean)
}

fn train_in_harness(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let held: RefCell<Option<Solution>> = RefCell::new(None);
    let seen = Cell::new(0usize);
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

    if let Some(sol) = held.into_inner() {
        save_solution(&sol)?;
    }
    if STOPPED.with(|f| f.get()) && seen.get() > 0 {
        return Ok(());
    }
    outcome
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    CFG.with(|c| *c.borrow_mut() = parse_cfg(hyperparameters));
    STOPPED.with(|f| f.set(false));
    REC_DELTA.with(|d| d.set(challenge.min_loss_delta));
    PSTOP_PATIENCE.with(|p| p.set(challenge.patience));

    unsafe {
        stream.context().disable_event_tracking();
    }
    let _ = stream
        .context()
        .set_flags(cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_SPIN);

    let gp_cfg = gp::parse(hyperparameters);
    if gp_cfg.enabled
        && gp::supported(challenge)
        && gp::try_solve(challenge, save_solution, &gp_cfg, module.clone(), stream.clone())?
    {
        return Ok(());
    }

    let mean = noisy_target_mean(challenge, &stream)?;
    TARGET_MEAN.with(|t| t.set(mean));
    train_in_harness(challenge, save_solution, module, stream, prop)
}
