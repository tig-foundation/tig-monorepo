use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value, json};

mod mod_a;
mod mod_b;
mod mod_c;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    match challenge.num_hidden_layers {
        4 => mod_c::track_t29::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        7 => mod_b::track_7::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        10 => mod_a::track_t26::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        14 => mod_b::track_14::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        18 => {
            let baked = json!({
                "total_steps": 1350,
                "warmup_steps": 60,
                "noise_variance": 0.04,
                "spectral_boost": 1.1,
                "output_layer_damping": 0.65
            });
            let hp_map = baked.as_object().unwrap().clone();
            mod_b::track_18::solve(challenge, save_solution, &Some(hp_map), module, stream, prop)
        }
        n => Err(anyhow!("Unsupported num_hidden_layers: {}", n)),
    }
}

pub fn help() {
    println!("neural_extrem_v5 - per-track neural net optimizer");
}
