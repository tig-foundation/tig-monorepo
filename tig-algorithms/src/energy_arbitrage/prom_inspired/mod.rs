use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod track_baseline;
mod track_capstone;
mod track_congested;
mod track_dense;
mod track_multiday;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15 => track_baseline::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 30 => track_congested::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 50 => track_multiday::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 80 => track_dense::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 150 => track_capstone::solve_challenge(challenge, save_solution, hyperparameters),
        n => Err(anyhow!("test: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("energy_arbitrage 'prom_inspired' - track-specialised per-track solver files");
    println!("Routes by battery count to baseline / congested / multiday / dense / capstone.");
    println!();
    println!("Hyperparameter guidance:");
    println!("  Defaults are the recommended starting point and are already track-tuned.");
    println!("  Override only for focused per-track testing or explicit quality/runtime/fuel trade-offs.");
    println!();
    println!("  baseline (s=baseline, <=15 batteries):");
    println!("    Default now enables anticipate_lmp with lmp_premium_scale=0.75.");
    println!("    use_lp/use_cg/use_sdp stay on by default; lp_refine_sweeps/cg_iters/asca_iters add more search depth.");
    println!("    soc_levels/action_grid/network_derating/flow_margin are secondary tuning knobs.");
    println!();
    println!("  congested (<=30 batteries):");
    println!("    Recent tuning moved grad_outer_iters to 120 and lookahead_horizon to 16.");
    println!("    grad_outer_iters remains the main quality/runtime knob; proj_max_iters and grad_ls_iters are secondary.");
    println!("    coord_polish_passes is runtime-heavy and showed little gain in recent tests.");
    println!();
    println!("  multiday (<=50 batteries):");
    println!("    Recent tuning moved grad_outer_iters to 60.");
    println!("    grad_outer_iters is the main search-depth knob; shorter lookahead helped less than outer-iters.");
    println!("    fuel_budget=0 keeps the solver's internal fuel-aware regime gating.");
    println!();
    println!("  dense (<=80 batteries):");
    println!("    Recent tuning moved grad_outer_iters to 40.");
    println!("    grad_outer_iters is by far the main dense-track knob; coord_polish_passes and lookahead_horizon had low impact.");
    println!("    dp/policy grid sizes are expensive and usually not the first place to tune.");
    println!();
    println!("  capstone (<=150 batteries):");
    println!("    Current defaults are still the conservative starting point.");
    println!("    If tuning capstone, start with grad_outer_iters first, then coord_polish_passes / lookahead_horizon / projection knobs.");
    println!("    fuel_budget can cap work; 0 uses the track's internal safety budget.");
}
