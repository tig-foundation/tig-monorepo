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
    println!("Neural Supreme - Dual-Phase Consensus Optimizer");
    println!("================================================");
    println!();
    println!("Tracks (determined by challenge.num_hidden_layers):");
    println!("  - n_hidden=4:  Shallow network, fast convergence");
    println!("  - n_hidden=7:  Medium-shallow network");
    println!("  - n_hidden=10: Medium network");
    println!("  - n_hidden=14: Medium-deep network");
    println!("  - n_hidden=18: Deep network, slower convergence");
    println!();
    println!("Hyperparameters (all optional, JSON format):");
    println!("  total_steps:        Total training steps (default: 1000)");
    println!("  warmup_steps:       Warmup phase steps (default: 40-55)");
    println!("  noise_variance:     Target noise floor (default: 0.04-0.048)");
    println!("  spectral_boost:     Learning rate boost (default: 1.02-1.1)");
    println!("  beta1:              Momentum decay (default: 0.89-0.92)");
    println!("  beta2:              Velocity decay (default: 0.997-0.999)");
    println!("  eps:                Numerical stability (default: 1e-8)");
    println!("  weight_decay:       L2 regularization (default: 0.0025-0.0032)");
    println!("  bn_layer_boost:     Small layer LR boost (default: 1.25-1.35)");
    println!("  output_layer_damping: Output layer damping (default: 0.72-0.8)");
    println!();
    println!("Example usage:");
    println!("  test_algorithm neural_supreme n_hidden=4 null --nonces 10");
}
