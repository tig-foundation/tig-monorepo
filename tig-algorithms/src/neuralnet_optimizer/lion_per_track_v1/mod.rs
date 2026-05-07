// neural_lion_per_track_v1 — Lion with per-track tuned hyperparameters.
//
// Direct fix for neural_baseline_lion_v1's design flaw. Per the all-28
// recon (STRATEGY/tig_recon/2026_05_03_all_28_submissions_landscape.md):
//   "62.5% of competitors ignore that the challenge explicitly varies
//    hidden layer depths (4→18) — instant validator rejection."
//   "v4's edge is boring: better constants, per-track tuned."
//
// Same Lion update rule (sign-momentum, Chen et al. 2023) but with
// distinct (lr, β₁, β₂, weight_decay) tuples per track:
//
//   n_hidden = 4   → smallest MLP, more aggressive lr, higher β₁
//   n_hidden = 7   → small, slightly less aggressive
//   n_hidden = 10  → medium, paper defaults
//   n_hidden = 14  → larger, smaller lr, more momentum smoothing
//   n_hidden = 18  → largest, most conservative lr
//
// Track is detected from param_sizes structure on first init call
// (count of 256×256 weight tensors ≈ n_hidden + 1).
//
// Single CUDA kernel (same as neural_baseline_lion_v1's lion_step) — the
// per-track logic is purely in hyperparameter selection on the Rust side.

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

#[derive(Clone, Copy, Debug)]
struct TrackHP {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hyperparameters {
    // Per-track tuples. Defaults are hand-tuned for Lion on c006-like
    // small MLP regression — small networks tolerate larger lr, big ones
    // need smaller lr and more momentum smoothing.
    pub track_4_lr: f32,   pub track_4_beta1: f32,   pub track_4_beta2: f32,   pub track_4_wd: f32,
    pub track_7_lr: f32,   pub track_7_beta1: f32,   pub track_7_beta2: f32,   pub track_7_wd: f32,
    pub track_10_lr: f32,  pub track_10_beta1: f32,  pub track_10_beta2: f32,  pub track_10_wd: f32,
    pub track_14_lr: f32,  pub track_14_beta1: f32,  pub track_14_beta2: f32,  pub track_14_wd: f32,
    pub track_18_lr: f32,  pub track_18_beta1: f32,  pub track_18_beta2: f32,  pub track_18_wd: f32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            // Smallest network — most aggressive
            track_4_lr:  6e-4,  track_4_beta1:  0.92,  track_4_beta2:  0.985, track_4_wd:  0.0,
            track_7_lr:  4e-4,  track_7_beta1:  0.91,  track_7_beta2:  0.99,  track_7_wd:  0.0,
            // Paper-default-ish
            track_10_lr: 3e-4,  track_10_beta1: 0.9,   track_10_beta2: 0.99,  track_10_wd: 0.0,
            track_14_lr: 2e-4,  track_14_beta1: 0.9,   track_14_beta2: 0.995, track_14_wd: 0.0,
            // Largest — most conservative
            track_18_lr: 1.5e-4,track_18_beta1: 0.88,  track_18_beta2: 0.995, track_18_wd: 0.0,
        }
    }
}

impl Hyperparameters {
    fn from_map(m: &Option<Map<String, Value>>) -> Self {
        let mut h = Self::default();
        if let Some(m) = m {
            macro_rules! pull { ($k:expr, $field:ident) => {
                if let Some(v) = m.get($k).and_then(|v| v.as_f64()) { h.$field = v as f32; }
            }; }
            pull!("track_4_lr", track_4_lr);   pull!("track_4_beta1", track_4_beta1);
            pull!("track_4_beta2", track_4_beta2); pull!("track_4_wd", track_4_wd);
            pull!("track_7_lr", track_7_lr);   pull!("track_7_beta1", track_7_beta1);
            pull!("track_7_beta2", track_7_beta2); pull!("track_7_wd", track_7_wd);
            pull!("track_10_lr", track_10_lr); pull!("track_10_beta1", track_10_beta1);
            pull!("track_10_beta2", track_10_beta2); pull!("track_10_wd", track_10_wd);
            pull!("track_14_lr", track_14_lr); pull!("track_14_beta1", track_14_beta1);
            pull!("track_14_beta2", track_14_beta2); pull!("track_14_wd", track_14_wd);
            pull!("track_18_lr", track_18_lr); pull!("track_18_beta1", track_18_beta1);
            pull!("track_18_beta2", track_18_beta2); pull!("track_18_wd", track_18_wd);
        }
        h
    }

    fn for_n_hidden(&self, n_hidden: usize) -> TrackHP {
        match n_hidden {
            n if n <= 4  => TrackHP { lr: self.track_4_lr,  beta1: self.track_4_beta1,  beta2: self.track_4_beta2,  weight_decay: self.track_4_wd  },
            n if n <= 7  => TrackHP { lr: self.track_7_lr,  beta1: self.track_7_beta1,  beta2: self.track_7_beta2,  weight_decay: self.track_7_wd  },
            n if n <= 10 => TrackHP { lr: self.track_10_lr, beta1: self.track_10_beta1, beta2: self.track_10_beta2, weight_decay: self.track_10_wd },
            n if n <= 14 => TrackHP { lr: self.track_14_lr, beta1: self.track_14_beta1, beta2: self.track_14_beta2, weight_decay: self.track_14_wd },
            _            => TrackHP { lr: self.track_18_lr, beta1: self.track_18_beta1, beta2: self.track_18_beta2, weight_decay: self.track_18_wd },
        }
    }
}

pub fn help() {
    println!("neural_lion_per_track_v1 — Lion with per-track (n_hidden) hyperparameter tuning");
    println!("Hyperparameters: track_{{4,7,10,14,18}}_{{lr,beta1,beta2,wd}}");
}

#[derive(Clone)]
struct OptimizerState {
    momentum: Vec<CudaSlice<f32>>,
    n_hidden: Arc<Mutex<usize>>,
    hp: Hyperparameters,
}

impl OptimizerStateTrait for OptimizerState {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn box_clone(&self) -> Box<dyn OptimizerStateTrait> { Box::new(self.clone()) }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
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
    _seed: [u8; 32],
    param_sizes: &[usize],
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Box<dyn OptimizerStateTrait>> {
    let hp = HYPERPARAMS.with(|h| h.borrow().clone()).unwrap_or_default();
    let n_hidden = param_sizes.iter().filter(|&&n| n == 256 * 256).count().max(1);

    let mut momentum = Vec::with_capacity(param_sizes.len());
    for &n in param_sizes { momentum.push(stream.alloc_zeros::<f32>(n)?); }

    Ok(Box::new(OptimizerState {
        momentum,
        n_hidden: Arc::new(Mutex::new(n_hidden)),
        hp,
    }))
}

fn optimizer_query_at_params(
    _: &dyn OptimizerStateTrait, _: &[CudaSlice<f32>], _: usize,
    _: Option<f32>, _: Option<f32>,
    _: Arc<CudaStream>, _: Arc<CudaModule>, _: &cudaDeviceProp,
) -> Result<Option<Vec<CudaSlice<f32>>>> { Ok(None) }

fn optimizer_step(
    optimizer_state: &mut dyn OptimizerStateTrait,
    model_params: &[CudaSlice<f32>],
    gradients: &[CudaSlice<f32>],
    _epoch: usize, _train_loss: Option<f32>, _val_loss: Option<f32>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    _prop: &cudaDeviceProp,
) -> Result<Vec<CudaSlice<f32>>> {
    let state = optimizer_state.as_any_mut().downcast_mut::<OptimizerState>()
        .ok_or_else(|| anyhow!("downcast failed"))?;

    if model_params.len() != gradients.len() || model_params.len() != state.momentum.len() {
        return Err(anyhow!("shape mismatch"));
    }

    let n_hidden = *state.n_hidden.lock().unwrap();
    let track = state.hp.for_n_hidden(n_hidden);

    let kernel = module.load_function("lion_step")?;
    let mut updates: Vec<CudaSlice<f32>> = Vec::with_capacity(model_params.len());

    for (i, theta) in model_params.iter().enumerate() {
        let grad = &gradients[i];
        let m = &mut state.momentum[i];
        let n = theta.len();
        let mut update = stream.alloc_zeros::<f32>(n)?;
        let grid_dim = (n as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&mut update).arg(m).arg(theta).arg(grad)
                .arg(&track.lr).arg(&track.beta1).arg(&track.beta2).arg(&track.weight_decay)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        updates.push(update);
    }

    Ok(updates)
}
