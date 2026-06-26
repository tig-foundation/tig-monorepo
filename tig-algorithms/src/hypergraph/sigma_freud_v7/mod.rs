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
    println!("Sigma Freud V7 - GPU-accelerated Hypergraph Partitioning");
    println!();
    println!("Uses capacity-aware move selection with Iterated Local Search (ILS) and swap phases.");
    println!();
    println!("=== QUICK START ===");
    println!("  - Default settings (effort=2) work well for most cases");
    println!("  - For better quality at cost of runtime, increase effort to 3 or 4");
    println!("  - For faster runtime with slight quality loss, use effort=1 or 0");
    println!();
    println!("=== HYPERPARAMETERS ===");
    println!();
    println!("  effort           Overall effort level (0-5, default: 2)");
    println!("                   Controls base refinement, ILS passes, polish, and post-refinement");
    println!("                  Higher = better quality, longer runtime");
    println!();
    println!("  clusters         Hyperedge cluster count (4-256, default: 64)");
    println!("                   Rounded up to multiple of 4 internally");
    println!();
    println!("  tabu_tenure      Tabu memory length (1-30, default: 12 or 14 depending on track)");
    println!("                   Higher values reduce cycling but can block good revisits");
    println!();
    println!("  refinement       Main refinement rounds (50-50000, default from effort preset)");
    println!("                   Effort presets map to 500..10000 base rounds");
    println!();
    println!("  ils_iterations   Number of ILS cycles (1-10, default from effort preset)");
    println!("                   Preset defaults are 3 or 5");
    println!();
    println!("  ils_quick_refine Quick refine rounds per ILS cycle (10-100, default from effort)");
    println!("                   Preset defaults are 20, 25, or 50");
    println!();
    println!("  post_ils_polish  Polish rounds after ILS (20-200, default from effort)");
    println!("                   Preset defaults are 30, 40, 100, or 150");
    println!();
    println!("  post_refinement  Post-balance refinement rounds (0-128, default from effort)");
    println!("                   Preset defaults are 32 or 64");
    println!();
    println!("  move_limit       Max moves considered per round (256-1000000, auto-scaled)");
    println!("                   Lower = faster but may miss good moves");
    println!();
    println!("=== EFFORT PRESETS ===");
    println!("  effort=0: refine=500,   ils=3, quick=20, polish=30,  post_ref=32");
    println!("  effort=1: refine=1000,  ils=3, quick=25, polish=40,  post_ref=32");
    println!("  effort=2: refine=2000,  ils=5, quick=50, polish=100, post_ref=64 (DEFAULT)");
    println!("  effort=3: refine=3000,  ils=5, quick=50, polish=150, post_ref=64");
    println!("  effort=4: refine=5000,  ils=5, quick=50, polish=200, post_ref=64");
    println!("  effort=5: refine=10000, ils=5, quick=50, polish=250, post_ref=64");
    println!();
    println!("=== EXAMPLE USAGE ===");
    println!("  Default:         null");
    println!("  Higher effort:   {{\"effort\": 4}}");
    println!("  Max quality:     {{\"effort\": 5, \"refinement\": 50000}}");
    println!("  Custom tuning:   {{\"effort\": 3, \"tabu_tenure\": 14, \"post_refinement\": 64}}");
}