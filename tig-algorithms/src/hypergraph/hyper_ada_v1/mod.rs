use anyhow::Result;
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::{Challenge, Solution};

mod track_10k;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    // Save immediate feasible partition as anytime fallback
    let dummy_partition: Vec<u32> = (0..challenge.num_nodes as u32)
        .map(|i| i % challenge.num_parts as u32)
        .collect();
    save_solution(&Solution {
        partition: dummy_partition,
    })?;

    // Dispatch to track solver
    match challenge.num_hyperedges {
        10000 => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        _ => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("hyper_ada_v1 - Ultra-Fast Bitmask GPU Hypergraph Partitioning");
    println!();
    println!("Architecture: Register-resident 64-bit integer bitmasks + zero-copy capacity-aware GPU refinement.");
    println!("Designed for NVIDIA Ada Lovelace (sm_89) and modern compute architectures.");
    println!();
    println!("=== HYPERPARAMETERS ===");
    println!("  rounds      Number of GPU refinement rounds (1-100, default: 15)");
    println!();
    println!("=== EXAMPLE USAGE ===");
    println!("  Default:    null");
    println!("  Fast:       {{\"rounds\": 10}}");
    println!("  Deep:       {{\"rounds\": 30}}");
}
