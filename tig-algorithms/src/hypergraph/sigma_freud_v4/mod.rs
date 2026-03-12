use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

mod track_10k;
mod track_20k;
mod track_50k;
mod track_100k;
mod track_200k;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {    
    let dummy_partition: Vec<u32> = (0..challenge.num_nodes as u32)
        .map(|i| i % challenge.num_parts as u32)
        .collect();
    save_solution(&Solution {
        partition: dummy_partition,
    })?;

    match challenge.num_hyperedges {
        10000 => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        20000 => track_20k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        50000 => track_50k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        100000 => track_100k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        200000 => track_200k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        _ => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("Hypergraph Partitioning - GPU-accelerated with adaptive clustering");
    println!();
    println!("Hyperparameters:");
    println!("  effort          Overall quality/runtime tradeoff (0-5, default: 2)");
    println!("                  Controls defaults for all algorithm phases");
    println!("                  0=fastest, 5=highest quality");
    println!();
    println!("  Advanced (override effort defaults):");
    println!("  refinement      Main refinement iterations (50-5000)");
    println!("  ils_iterations  Iterated local search restarts (1-10)");
    println!("  post_refinement Final balance passes (0-128)");
    println!("  move_limit      Max moves per refinement round (256-1000000)");
    println!("  clusters        Hyperedge clusters (4-256, must be multiple of 4)");
}
