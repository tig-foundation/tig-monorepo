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
    println!("No hyperparameters. All settings are hardcoded to defaults.");
}
