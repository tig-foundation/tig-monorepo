use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::neuralnet_optimizer::*;
use serde_json::{Map, Value};

mod helpers;
mod track_t26;
mod track_t27;
mod track_t28;
mod track_t29;
mod track_t30;

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
        n => Err(anyhow!("Unsupported num_hidden_layers: {}", n)),
    }
}

pub fn help() {
    println!("Neural Extrem v4_f — Fisher-ASAM + Trust-Gated WD + Jacobian Trust Damping");
}
