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
    println!("Sigma Freud V5 - GPU-accelerated Hypergraph Partitioning");
    println!();
    println!("Uses capacity-aware move selection with Iterated Local Search (ILS).");
    println!();
    println!("=== QUICK START ===");
    println!("  - Default settings (effort=2) work well for most cases");
    println!("  - For better quality at cost of runtime, increase effort to 3 or 4");
    println!("  - For faster runtime with slight quality loss, use effort=1 or 0");
    println!();
    println!("=== HYPERPARAMETERS ===");
    println!();
    println!("  effort          Overall effort level (0-5, default: 2)");
    println!("                  Controls refinement rounds, ILS iterations, and polish passes");
    println!("                  Higher = better quality, longer runtime");
    println!();
    println!("  tabu_tenure     Tabu search memory length (1-30, default: 10-14 depending on track)");
    println!("                  Higher values prevent cycling but may miss good moves");
    println!();
    println!("  refinement      Main refinement rounds (50-5000, default: 500)");
    println!("                  More rounds = better quality, longer runtime");
    println!();
    println!("  ils_iterations  Number of ILS perturbation cycles (1-10, default: 5)");
    println!("                  More iterations = explore more solution space");
    println!();
    println!("  ils_quick_refine Refinement rounds per ILS iteration (10-100, default: 30-50)");
    println!("                  Quick local search after each perturbation");
    println!();
    println!("  post_ils_polish Polish rounds after ILS (20-200, default: 60-150)");
    println!("                  Final refinement to improve solution quality");
    println!();
    println!("  move_limit      Max moves considered per round (256-1000000, auto-scaled)");
    println!("                  Lower = faster but may miss good moves");
    println!();
    println!("=== EFFORT PRESETS ===");
    println!("  effort=0: Fast mode - minimal refinement");
    println!("  effort=1: Light mode - reduced iterations");
    println!("  effort=2: Default - balanced quality/speed (RECOMMENDED)");
    println!("  effort=3: Quality mode - more thorough search");
    println!("  effort=4: High quality - extended refinement");
    println!("  effort=5: Maximum quality - longest runtime");
    println!();
    println!("=== EXAMPLE USAGE ===");
    println!("  Default:        null");
    println!("  Higher effort:  {{\"effort\": 4}}");
    println!("  Custom tuning:  {{\"effort\": 3, \"tabu_tenure\": 12}}");
}
