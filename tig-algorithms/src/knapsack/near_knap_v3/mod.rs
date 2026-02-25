pub mod construct;
pub mod ils;
pub mod local_search;
pub mod params;
pub mod refinement;
pub mod types;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

use ils::run_one_instance;
use params::Params;

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let params = Params::initialize(hyperparameters);
        let solution = run_one_instance(challenge, &params);
        Ok(Some(solution))
    }
}

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
        let _ = save_solution(&solution);
    }
    Ok(())
}

pub fn help() {
    println!("Quadratic Knapsack - Multi-Start ILS with Hybrid Basin Discovery");
    println!();
    println!("Hyperparameters:");
    println!("  effort (integer 1-6, default 1):");
    println!("    1 -> 15 rounds, strength 3  (default)");
    println!("    2 -> 22 rounds, strength 3");
    println!("    3 -> 30 rounds, strength 3");
    println!("    4 -> 40 rounds, strength 4");
    println!("    5 -> 50 rounds, strength 4");
    println!("    6 -> 60 rounds, strength 4  (best quality)");
    println!();
    println!("  n_perturbation_rounds        : overrides effort rounds directly");
    println!("  perturbation_strength_base   : overrides effort strength directly");
    println!();
    println!("  extra_starts (default 0)     : additional construction starts on top of the auto-computed count");
    println!("  max_frontier_swaps_override  : overrides the per-iteration frontier swap limit (default: 0 for n>=2500, 1 for n>=1500, 2 otherwise)");
    println!("  dp_passes_multiplier (def 1) : multiplies the number of DP refinement passes per call");
}
