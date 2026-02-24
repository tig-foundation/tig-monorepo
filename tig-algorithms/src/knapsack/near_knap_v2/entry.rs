use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
use super::params::Params;
use super::ils::run_one_instance;

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
    println!("Quadratic Knapsack Problem - Multi-Start ILS with Hybrid Basin Discovery");
    println!("  + support tracking, micro-QKP refinement, frontier-swap local search");
    println!();
    println!("Hyperparameters:");
    println!("  effort: 1-6 (default: 1)");
    println!("    Controls search intensity for n<2500 tracks only.");
    println!("    effort=1: perturbation_rounds=15, +0 starts");
    println!("    effort=2: perturbation_rounds=22");
    println!("    effort=3: perturbation_rounds=29, +1 start");
    println!("    effort=4: perturbation_rounds=36");
    println!("    effort=5: perturbation_rounds=43, +2 starts");
    println!("    effort=6: perturbation_rounds=50");
    println!("    Note: has no effect on n>=2500 tracks.");
    println!();
    println!("  stall_limit: 1-20 (default: 6)");
    println!("    Consecutive non-improving perturbation rounds before early exit.");
    println!("    Affects ALL tracks. Higher = more persistent search.");
    println!();
    println!("  perturbation_strength: 1-20 (default: auto)");
    println!("    Overrides the number of items removed per perturbation step.");
    println!("    Affects ALL tracks. Default scales with effort and instance size.");
    println!();
    println!("  perturbation_rounds: 1-100 (default: auto)");
    println!("    Overrides the maximum number of ILS perturbation rounds.");
    println!("    Affects ALL tracks. Default scales with effort and instance size.");
}
