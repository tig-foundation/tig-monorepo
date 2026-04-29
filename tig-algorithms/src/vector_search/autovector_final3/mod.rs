
mod track_shared;
mod track_t19;
mod track_t20;

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
    println!("autovector_final3 - per-track GPU dispatch (T16/17/18 cuBLAS, T19 FP16 padded, T20 FP16 large-tile)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {

    match challenge.num_queries as usize {
        7000  => track_t19::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        9000  => track_t20::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        _     => track_shared::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}
