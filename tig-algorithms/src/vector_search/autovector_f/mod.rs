
pub mod track_t16;
pub mod track_t17;
pub mod track_t18;
pub mod track_t19;
pub mod track_t20;

use anyhow::{anyhow, Result};
use cudarc::driver::safe::{CudaModule, CudaStream};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Hyperparameters {}

pub fn help() {
    println!("autovector_f");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    match challenge.num_queries {
        7000  => track_t19::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        9000  => track_t20::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        11000 => track_t16::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        13000 => track_t17::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        15000 => track_t18::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        n     => Err(anyhow!("autovector_final4: unknown track for num_queries={}", n)),
    }
}
