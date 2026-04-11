// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

mod helpers;
mod track_t26;
mod track_t27;
mod track_t28;
mod track_t29;
mod track_t30;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub total_steps: Option<usize>,
    pub warmup_steps: Option<usize>,
    pub spectral_boost: Option<f64>,
    pub noise_variance: Option<f64>,
    pub beta1: Option<f64>,
    pub beta2: Option<f64>,
    pub weight_decay: Option<f64>,
    pub bn_layer_boost: Option<f64>,
    pub output_layer_damping: Option<f64>,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    match challenge.num_hidden_layers {
        4 => track_t29::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        7 => track_t30::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        10 => track_t26::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        14 => track_t27::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        18 => track_t28::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        n => Err(anyhow!("Unsupported num_hidden_layers: {}. Valid values are 4, 7, 10, 14, 18", n)),
    }
}

pub fn help() {
    println!("Neural Extrem V3 - Dual-Phase Consensus Optimizer");
    println!("All tracks support HP tuning via JSON.");
    println!();
    println!("Tracks: T29(n=4) T30(n=7) T26(n=10) T27(n=14) T28(n=18)");
    println!("HP: total_steps, warmup_steps, spectral_boost, noise_variance,");
    println!("    beta1, beta2, weight_decay, bn_layer_boost, output_layer_damping");
}
