use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

pub(crate) fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

mod hflow;
mod hjet;
mod hmulti;
mod hrefine;
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
    println!("Sigma_freud_v10 - GPU-accelerated Hypergraph Partitioning");
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
    println!("=== GPU LOOP (all tracks; output is bit-identical for any setting) ===");
    println!("  mlanes           Lanes per node in the moves kernel (2-32, power of two, default 4; 0 = original kernel)");
    println!("  mblocks          Grid size of the moves kernel (default: two nodes per lane group)");
    println!("  tiebreak         Equal-gain rule in the moves kernel: 0 lowest part, 1 smaller part+hash, 2 more pins (default 1 on 50k/200k, else 0)");
    println!("  gfloor           Main-loop kernel emits no key below this gain (default -3, bit-identical; -32768 disables)");
    println!("  gfilter          Main-loop scan + tabu/RNG filter on the GPU, host receives only survivors (default 1, bit-identical; 0 = host filter)");
    println!("  rfuse            whole main-loop round (flags -> moves -> filter -> sort -> quota fill) as ONE launch (default 1, bit-identical; 0 = separate launches / host selection; 20k+ needs hostflags 0; 10k uses its own kernel with the 10k move rules)");
    println!("  rmulti           rounds per launch (default 8, all tracks); the kernel applies the accepted moves and tabu marks on the device between rounds and returns early on stagnation / no candidates (20k+: needs gsort 1; batches stop at reheat cycle boundaries); 1 = one round per launch with the host replay (bit-identical)");
    println!("  gsort            20k+: round kernel also radix-sorts the candidates and runs the quota fill, host downloads only the accepted moves (default 1, bit-identical; 0 = host bucket/select/sort; 10k always sorts on the device)");
    println!("  reheat           Number of cooling cycles in the main loop; each restarts from the best partition so far (default 1 = unchanged)");
    println!("  reheat_temp      Initial acceptance of cycles after the first, as percent of the first cycle's (default 50)");
    println!("  sched_exp        Cooling exponent: low-gain acceptance = (rounds_left/R)^exp / (1 + pen_mul*p) (default 2; 1..4)");
    println!("  pen_mul          Gain penalty multiplier in the acceptance denominator (default 5)");
    println!("  tabu_exec        1 = mark only nodes that actually moved as tabu (default 0: head of accepted list)");
    println!("  swap_scale       Swap/cycle phase rounds in percent (default 100; 0 skips it)");
    println!("  hostflags        1 = maintain hyperedge flags on the host and upload them each round (default 1 on 10k, else 0; 1 disables rfuse)");
    println!("                   instead of launching precompute_edge_flags (default 1 at 10k-50k,");
    println!("                   0 at 100k/200k where the upload is ~break-even)");
    println!();
    println!("=== HOST REFINEMENT (exact FM / compound moves / flow / multilevel V-cycles) ===");
    println!("  href             0/1 enable the host stage after the GPU pipeline (default 1)");
    println!("  hfuel_budget     Fuel the host stage may spend (0 = unlimited; ~2G fuel per second).");
    println!("                   Defaults: 10k unlimited, 20k 10G, 50k 15G, 100k 20G, 200k 30G");
    println!("  hfuel_reserve    Stop the host stage when remaining fuel drops below this (default 5G)");
    println!("  hvcycles, hml_levels, hfm_*, hcm_*, hflow_*   see track source for the full list");
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
