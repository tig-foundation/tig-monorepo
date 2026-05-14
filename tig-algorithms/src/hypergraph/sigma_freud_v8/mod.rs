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
    println!("Sigma Freud V8 - GPU-accelerated Hypergraph Partitioning");
    println!();
    println!("Uses capacity-aware move selection with Iterated Local Search (ILS) and swap phases.");
    println!();
    println!("=== QUICK START ===");
    println!("  - Default settings (effort=3) work well for most cases");
    println!("  - For better quality at cost of runtime, increase effort to a maximum of 5");
    println!("  - For faster runtime with slight quality loss, use effort=1 or 0");
    println!();
    println!("=== HYPERPARAMETERS ===");
    println!();
    println!("  effort           Overall effort level (0-5, default: 3)");
    println!("                   Controls base refinement, ILS passes, polish, and post-refinement");
    println!("                  Higher = better quality, longer runtime");
    println!();
    println!("  clusters         Hyperedge cluster count (4-256, default: 64)");
    println!("                   Rounded up to multiple of 4 internally");
    println!();
    println!("  tabu_tenure      Tabu memory length (1-30, default: 12 or 14 depending on track)");
    println!("                   Higher values reduce cycling but can block good revisits");
    println!();
    println!("  refinement       Main refinement rounds (50-50000, default from track effort preset)");
    println!("                   Overrides the effort preset's refinement count");
    println!();
    println!("  ils_iterations   Number of ILS cycles (1-500, default from track effort preset)");
    println!();
    println!("  ils_quick_refine Quick refine rounds per ILS cycle (10-500, default from effort)");
    println!();
    println!("  post_ils_polish  Polish rounds after ILS (20-500, default from effort)");
    println!();
    println!("  post_refinement  Post-balance refinement rounds (0-128, default from effort)");
    println!();
    println!("  move_limit       Max moves considered per round (256-1000000, auto-scaled)");
    println!("                   Lower = faster but may miss good moves");
    println!();
    println!("=== DEFAULT EFFORT=3 PRESETS ===");
    println!("  10k:        refine=8000, ils=5, quick=60, polish=150, post_ref=0");
    println!("  20k:        refine=3000, ils=5, quick=50, polish=150, post_ref=64");
    println!("  50k:        refine=4000, ils=5, quick=50, polish=150, post_ref=64");
    println!("  100k/200k:  refine=5000, ils=5, quick=50, polish=150, post_ref=64");
    println!("  Higher efforts increase the same budgets per track.");
    println!();
    println!("=== EXAMPLE USAGE ===");
    println!("  Default:         null");
    println!("  Higher effort:   {{\"effort\": 4}}");
    println!("  Max quality:     {{\"effort\": 5, \"refinement\": 50000}}");
    println!("  Custom tuning:   {{\"effort\": 3, \"tabu_tenure\": 14, \"post_refinement\": 64}}");
}