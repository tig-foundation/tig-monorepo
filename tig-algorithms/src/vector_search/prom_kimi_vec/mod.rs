mod track_shared;
mod track_t19;
mod track_t20;
mod sub_t19_alt;

use anyhow::Result;
use cudarc::driver::safe::{CudaModule, CudaStream};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Hyperparameters {}

pub fn help() {
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
        7000 => sub_t19_alt::solve_t19_alt(challenge, save_solution, hyperparameters, module, stream, prop),
        _    => track_t20::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}
