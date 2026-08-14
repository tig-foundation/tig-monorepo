use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

mod helpers;
mod track_4;
mod track_7;
mod track_10;
mod track_14;
mod track_18;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    match challenge.num_hidden_layers {
        4 => track_4::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        7 => track_7::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        10 => track_10::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        14 => track_14::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        18 => track_18::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        n => Err(anyhow!("Unsupported num_hidden_layers: {}. Valid values are 4, 7, 10, 14, 18", n)),
    }
}

pub fn help() {
    println!("Neural Advanced v4 - Dual-Phase Consensus Optimizer");
    println!("====================================================");
    println!("Further refined achieving higher qualities on n_hidden=4 and n_hidden=7");
    println!("Supported tracks: 4, 7, 10, 14, 18");
    println!("Supported hyperparameters for n_hidden 18 only: total_steps, warmup_steps, noise_variance, spectral_boost, beta1, beta2, eps, weight_decay, bn_layer_boost, output_layer_damping, threads_per_block, blocks_per_sm");
}
