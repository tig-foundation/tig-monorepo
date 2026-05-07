// neural_sam_lion_v1 — SAM (Sharpness-Aware Minimization) hybridised with Lion.
//
// SAM (Foret et al. 2021, https://arxiv.org/abs/2010.01412) seeks parameters
// θ such that the LOSS NEIGHBORHOOD around θ is flat — defined as the
// worst-case loss over a small ball of radius ρ around θ. This biases
// training toward generalising minima rather than sharp ones.
//
// Standard SAM is a 2-pass-per-step algorithm:
//   1. compute g₁ = ∇L(θ) at the current point
//   2. ascend to θ_adv = θ + ρ · g₁ / ‖g₁‖     ("worst-case perturbation")
//   3. compute g₂ = ∇L(θ_adv)                    ("SAM gradient")
//   4. step from θ using g₂
//
// TIG's optimizer trait gives us `optimizer_query_at_params` which is called
// BEFORE forward/backward. We exploit this to alternate batches:
//
//   odd batch  N:   query returns None. We receive g(θ_N), save as g_last.
//                   Apply Lion update using g(θ_N).
//   even batch N+1: query returns θ_{N+1} + ρ · sign(g_last) · ε_sam.
//                   We receive g at the perturbed (sharpness-probing) point.
//                   Apply Lion update using this SAM-style gradient.
//
// We use sign(g_last) rather than g_last/‖g_last‖ to avoid needing a per-
// tensor norm reduction kernel — sign-perturbation is a well-known SAM
// variant (Sign-SAM) with similar generalisation benefits and ~1/3 the
// kernel complexity.
//
// Why this might beat v4: per the all-28 recon, NO current submission uses
// query_at_params for sharpness probing. This is unexploited niche #2
// (after Sophia-H's Hessian probe). On c006's noisy-regression target
// (σ=0.2 added Gaussian noise on RFF labels), SAM-style flat-minima
// preference should produce better generalisation to the held-out test set.
//
// Per-track tuned learning rate (same scheme as Sophia-H + Lion-per-track).
// Memory: 2× param memory (momentum + g_last buffer).

use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Arc, Mutex};
use tig_challenges::neuralnet_optimizer::*;

const THREADS_PER_BLOCK: u32 = 1024;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hyperparameters {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub eps_sam: f32,                 // SAM perturbation magnitude
    pub sam_period: usize,            // probe every N batches (default 2 = SAM every other batch)
    pub lr_track_4: f32, pub lr_track_7: f32, pub lr_track_10: f32,
    pub lr_track_14: f32, pub lr_track_18: f32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            lr: 3e-4, beta1: 0.9, beta2: 0.99, weight_decay: 0.0,
            eps_sam: 5e-3,
            sam_period: 2,
            lr_track_4: 1.5,  lr_track_7: 1.2, lr_track_10: 1.0,
            lr_track_14: 0.8, lr_track_18: 0.6,
        }
    }
}

impl Hyperparameters {
    fn from_map(m: &Option<Map<String, Value>>) -> Self {
        let mut h = Self::default();
        if let Some(m) = m {
            macro_rules! pull_f { ($k:expr, $field:ident) => {
                if let Some(v) = m.get($k).and_then(|v| v.as_f64()) { h.$field = v as f32; }
            }; }
            pull_f!("lr", lr); pull_f!("beta1", beta1); pull_f!("beta2", beta2);
            pull_f!("weight_decay", weight_decay); pull_f!("eps_sam", eps_sam);
            pull_f!("lr_track_4", lr_track_4); pull_f!("lr_track_7", lr_track_7);
            pull_f!("lr_track_10", lr_track_10); pull_f!("lr_track_14", lr_track_14);
            pull_f!("lr_track_18", lr_track_18);
            if let Some(v) = m.get("sam_period").and_then(|v| v.as_u64()) { h.sam_period = v as usize; }
        }
        h
    }

    fn track_lr(&self, n_hidden: usize) -> f32 {
        let m = match n_hidden {
            n if n <= 4  => self.lr_track_4, n if n <= 7  => self.lr_track_7,
            n if n <= 10 => self.lr_track_10, n if n <= 14 => self.lr_track_14,
            _            => self.lr_track_18,
        };
        self.lr * m
    }
}

pub fn help() {
    println!("neural_sam_lion_v1 — SAM (sign-perturbation variant) hybridised with Lion");
    println!("Per-track lr, alternating SAM-probe batches via query_at_params");
}

#[derive(Clone)]
struct OptimizerState {
    momentum: Vec<CudaSlice<f32>>,
    g_last: Vec<CudaSlice<f32>>,                // gradient saved from last vanilla batch
    n_hidden: Arc<Mutex<usize>>,
    step: Arc<Mutex<usize>>,
    has_g_last: Arc<Mutex<bool>>,               // false until first vanilla batch fires
    hp: Hyperparameters,
}

impl OptimizerStateTrait for OptimizerState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

pub fn solve_challenge(
    challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp,
) -> Result<()> {
    let hp = Hyperparameters::from_map(hyperparameters);
    HYPERPARAMS.with(|h| *h.borrow_mut() = Some(hp));
    training_loop(challenge, save_solution, module, stream, prop,
                  optimizer_init_state, optimizer_query_at_params, optimizer_step)
}

thread_local! {
    static HYPERPARAMS: std::cell::RefCell<Option<Hyperparameters>> = std::cell::RefCell::new(None);
}

fn optimizer_init_state(
    _seed: [u8; 32], param_sizes: &[usize],
    stream: Arc<CudaStream>, _module: Arc<CudaModule>, _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let hp = HYPERPARAMS.with(|h| h.borrow().clone()).unwrap_or_default();
    let n_hidden = param_sizes.iter().filter(|&&n| n == 256 * 256).count().max(1);

    let mut alloc = || -> Result<Vec<CudaSlice<f32>>> {
        let mut v = Vec::with_capacity(param_sizes.len());
        for &n in param_sizes { v.push(stream.alloc_zeros::<f32>(n)?); }
        Ok(v)
    };

    Ok(Box::new(OptimizerState {
        momentum: alloc()?,
        g_last: alloc()?,
        n_hidden: Arc::new(Mutex::new(n_hidden)),
        step: Arc::new(Mutex::new(0)),
        has_g_last: Arc::new(Mutex::new(false)),
        hp,
    }))
}

fn optimizer_query_at_params(
    optimizer_state: &dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    _epoch: usize, _train_loss: Option<f32>, _val_loss: Option<f32>,
    stream: Arc<CudaStream>, module: Arc<CudaModule>, _prop: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> {
    let state = optimizer_state.as_any().downcast_ref::<OptimizerState>()
        .ok_or_else(|| anyhow!("downcast failed"))?;

    let step = *state.step.lock().unwrap();
    let has = *state.has_g_last.lock().unwrap();

    // SAM-probe only on (period | 1) batches AND only after we've captured
    // at least one g_last from a vanilla batch.
    let is_probe = state.hp.sam_period > 1 && (step % state.hp.sam_period == 1) && has;
    if !is_probe { return Ok(None); }

    let kernel = module.load_function("perturb_sign_sam")?;
    let mut perturbed: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());
    for (i, theta) in model_params.iter().enumerate() {
        let g_last = &state.g_last[i];
        let n = theta.len();
        let mut out = stream.alloc_zeros::<f32>(n)?;
        let grid_dim = (n as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&mut out).arg(theta).arg(g_last)
                .arg(&state.hp.eps_sam).arg(&(n as i32))
                .launch(cfg)?;
        }
        perturbed.push(out);
    }
    Ok(Some(perturbed))
}

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>], gradients: &[CudaSlice<f32>],
    _epoch: usize, _train_loss: Option<f32>, _val_loss: Option<f32>,
    stream: Arc<CudaStream>, module: Arc<CudaModule>, _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state.as_any_mut().downcast_mut::<OptimizerState>()
        .ok_or_else(|| anyhow!("downcast failed"))?;

    let step = { let mut s = state.step.lock().unwrap(); let cur = *s; *s += 1; cur };
    let n_hidden = *state.n_hidden.lock().unwrap();
    let lr = state.hp.track_lr(n_hidden);

    let is_probe = state.hp.sam_period > 1
        && (step % state.hp.sam_period == 1)
        && *state.has_g_last.lock().unwrap();
    let is_vanilla_to_save = !is_probe;

    let lion_kernel = module.load_function("lion_step")?;
    let capture     = module.load_function("capture_grad")?;
    let mut updates: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());

    for (i, theta) in model_params.iter().enumerate() {
        let grad = &gradients[i];
        let m    = &mut state.momentum[i];
        let n    = theta.len();
        let grid_dim = (n as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // On vanilla batches, capture grad as g_last for next probe step.
        if is_vanilla_to_save {
            let g_dst = &mut state.g_last[i];
            unsafe { stream.launch_builder(&capture).arg(g_dst).arg(grad).arg(&(n as i32)).launch(cfg)?; }
        }

        let mut update = stream.alloc_zeros::<f32>(n)?;
        unsafe {
            stream.launch_builder(&lion_kernel)
                .arg(&mut update).arg(m).arg(theta).arg(grad)
                .arg(&lr).arg(&state.hp.beta1).arg(&state.hp.beta2).arg(&state.hp.weight_decay)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        updates.push(update);
    }

    if is_vanilla_to_save { *state.has_g_last.lock().unwrap() = true; }

    Ok(updates)
}
