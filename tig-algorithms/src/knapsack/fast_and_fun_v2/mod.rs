use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
mod params;
mod state;
mod construct;
mod dp;
mod local_search;
mod solver;
pub use solver::Solver;
use tig_challenges::knapsack::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve(challenge, Some(save_solution), hyperparameters)? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Err(anyhow!("No feasible solution found")),
    }
}

pub fn help() {
    println!("No help information available.");
}
