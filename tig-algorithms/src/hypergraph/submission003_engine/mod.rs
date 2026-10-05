use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Number, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

// wave13_parity: the shared solver.  track_20k / 50k / 100k / 200k are now
// constant tables that call `round_engine::solve_with`; track_10k is NOT yet
// on the engine (wave14) and keeps its own implementation.
mod round_engine;

mod track_10k;
mod track_20k;
mod track_50k;
mod track_100k;
mod track_200k;

/// User hyperparameters win; `defaults` only fill in keys the user did not set.
fn merge_hp(
    user_hp: &Option<Map<String, Value>>,
    defaults: Vec<(&str, Value)>,
) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

fn n(v: u64) -> Value {
    Value::Number(Number::from(v))
}

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

    // Single stream, no cross-stream sharing: cudarc's per-memcpy event tracking (event create/record/wait/destroy
    // around every copy) is pure overhead in this loop (~46k blocking copies per nonce). Bit-identical results.
    unsafe { stream.context().disable_event_tracking(); }
    match challenge.num_hyperedges {
        10000 => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        20000 => {
            // Baked benchmarker defaults for the 20k track (user HPs override).
            // Same tuned point as the earlier multitrack submission, plus the
            // best-of-K knob `runs`.  Everything else on this track is the
            // unchanged fast solver.
            let hp = merge_hp(hyperparameters, vec![
                ("effort", n(5)),
                ("clusters", n(64)),
                ("move_limit", n(800000)),
                ("refinement", n(45000)),
                ("tabu_tenure", n(8)),
                ("ils_iterations", n(10)),
                ("post_ils_polish", n(200)),
                ("post_refinement", n(128)),
                ("ils_quick_refine", n(100)),
                ("runs", n(1)),
                ("run_flavor_mode", n(2)),
                ("run_refinement_pct", n(50)),
                ("run_ils_iters", n(1)),
                ("run0_ils_iters", n(0)),
                ("run_ils_quick_pct", n(5)),
                ("run0_ils_quick_pct", n(100)),
                ("run_polish_pct", n(5)),
                ("run0_polish_pct", n(100)),
            ]);
            track_20k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        50000 => {
            // Baked benchmarker defaults for the 50k track (user HPs override),
            // plus `runs` = 2 for the ruin-and-recreate best-of-K wrapper.
            let hp = merge_hp(hyperparameters, vec![
                ("clusters", n(64)),
                ("effort", n(5)),
                ("ils_iterations", n(10)),
                ("ils_quick_refine", n(100)),
                ("move_limit", n(800000)),
                ("post_ils_polish", n(200)),
                ("post_refinement", n(128)),
                ("refinement", n(32000)),
                ("tabu_tenure", n(8)),
                ("runs", n(1)),
                ("run_flavor_mode", n(2)),
                ("run_refinement_pct", n(50)),
                ("run_ils_iters", n(1)),
                ("run0_ils_iters", n(0)),
                ("run_ils_quick_pct", n(5)),
                ("run0_ils_quick_pct", n(100)),
                ("run_polish_pct", n(5)),
                ("fused_mode", n(2)),
                ("run0_polish_pct", n(100)),
            ]);
            track_50k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        100000 => {
            // Baked benchmarker defaults for the 100k track (user HPs override).
            // NOTE: no `move_limit` key -- the track's own size-dependent default
            // (200_000 here) is what the benchmarkers ran.
            let hp = merge_hp(hyperparameters, vec![
                ("clusters", n(72)),
                ("effort", n(5)),
                ("ils_iterations", n(13)),
                ("ils_quick_refine", n(220)),
                ("post_ils_polish", n(40)),
                ("post_refinement", n(0)),
                ("refinement", n(22000)),
                ("tabu_tenure", n(12)),
                ("runs", n(1)),
                ("run_flavor_mode", n(2)),
                ("run_refinement_pct", n(50)),
                ("run_ils_iters", n(1)),
                ("run0_ils_iters", n(0)),
                ("run_ils_quick_pct", n(5)),
                ("run0_ils_quick_pct", n(100)),
                ("run_polish_pct", n(5)),
                ("fused_mode", n(2)),
                ("run0_polish_pct", n(100)),
            ]);
            track_100k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        200000 => {
            // Baked benchmarker defaults for the 200k track (user HPs override).
            // `init_restart_id` is not read by this solver; it is kept so the
            // baked map is literally the benchmarkers' best-known HP set.
            let hp = merge_hp(hyperparameters, vec![
                ("clusters", n(72)),
                ("effort", n(5)),
                ("ils_iterations", n(10)),
                ("ils_quick_refine", n(80)),
                ("init_restart_id", n(0)),
                ("move_limit", n(250000)),
                ("post_ils_polish", n(40)),
                ("post_refinement", n(0)),
                ("refinement", n(20000)),
                ("tabu_tenure", n(13)),
                ("runs", n(1)),
                ("run_flavor_mode", n(2)),
                ("run_refinement_pct", n(50)),
                ("run_ils_iters", n(1)),
                ("run0_ils_iters", n(0)),
                ("run_ils_quick_pct", n(5)),
                ("run0_ils_quick_pct", n(100)),
                ("run_polish_pct", n(5)),
                ("fused_mode", n(2)),
                ("run0_polish_pct", n(100)),
            ]);
            track_200k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        _ => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("Freud Opt - GPU-accelerated Hypergraph Partitioning");
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
    println!("  runs             Best-of-K restarts of the post-construction solve (1-64,");
    println!("                   default: 2; 20k / 50k / 100k / 200k tracks). Each run repeats");
    println!("                   the whole refine/ILS/polish pipeline from the SAME construction");
    println!("                   with a different deterministic seed offset; the run with the");
    println!("                   lowest true connectivity (km1) is the one saved. runs=1");
    println!("                   reproduces the single-run solver bit for bit.");
    println!();
    println!("  --- start diversification for runs k >= 1 (20k/50k/100k/200k tracks) ---");
    println!("  These only ever affect runs 1..K-1; run 0 is always the plain solver,");
    println!("  so runs=1 stays bit-identical whatever they are set to.");
    println!();
    println!("  run_ruin_frac    Fraction of hyperedges considered by the ruin-and-recreate");
    println!("                   of run k's STARTING partition (default 0.02 = m/50).");
    println!("                   0 turns the ruin off (runs k>=1 then differ from run 0");
    println!("                   only by their RNG seed offset, i.e. the old behaviour).");
    println!();
    println!("  run_ruin_cap     Fraction of NODES the ruin is allowed to free");
    println!("                   (default 0.25). This is the real strength knob: the ruin");
    println!("                   walks hyperedges in descending lambda order and stops at");
    println!("                   this many freed nodes, so run_ruin_frac is nearly inert");
    println!("                   above ~0.02.");
    println!();
    println!("  run_ruin_growth_pct  Percent multiplier applied to run_ruin_cap for each");
    println!("                   extra run (default 100 = every run uses the same");
    println!("                   strength; 150 makes runs 1,2,3.. progressively wilder).");
    println!();
    println!("  run_clusters_delta   If non-zero, run k re-runs CONSTRUCTION with");
    println!("                   clusters + delta*k (rounded to a multiple of 4, clamped");
    println!("                   4..256) before the ruin, for a structurally different");
    println!("                   greedy start. Default 0 (off).");
    println!();
    println!("  run_tenure_delta If non-zero, run k uses tabu_tenure + delta*k (clamped");
    println!("                   1..30). Default 0 (off).");
    println!();
    println!("  run_ruin_run0    Set to 1 to ruin run 0's start as well. This BREAKS the");
    println!("                   bit-identity with the single-run solver on purpose and is");
    println!("                   off by default; {{\"runs\":1,\"run_ruin_run0\":1}} is the");
    println!("                   plain ruin-and-recreate construction, with no best-of-K.");
    println!();
    println!("=== EFFORT PRESETS ===");
    println!("  effort=0: refine=500,   ils=3, quick=20, polish=30,  post_ref=32");
    println!("  effort=1: refine=1000,  ils=3, quick=25, polish=40,  post_ref=32");
    println!("  effort=2: refine=2000,  ils=5, quick=50, polish=100, post_ref=64 (DEFAULT)");
    println!("  effort=3: refine=3000,  ils=5, quick=50, polish=150, post_ref=64");
    println!("  effort=4: refine=5000,  ils=5, quick=50, polish=200, post_ref=64");
    println!("  effort=5: refine=10000, ils=5, quick=50, polish=250, post_ref=64");
    println!("  (the 50k / 100k / 200k tracks use their own, larger, refine presets)");
    println!();
    println!("=== EXAMPLE USAGE ===");
    println!("  Default:         null");
    println!("  Higher effort:   {{\"effort\": 4}}");
    println!("  Max quality:     {{\"effort\": 5, \"refinement\": 50000}}");
    println!("  Custom tuning:   {{\"effort\": 3, \"tabu_tenure\": 14, \"post_refinement\": 64}}");
}
