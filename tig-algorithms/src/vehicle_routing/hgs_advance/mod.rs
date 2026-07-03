use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
mod constructive;
mod compression;
mod genetic;
mod individual;
mod loader_tig;
#[cfg(feature = "benchmark_io")]
mod loader_cvrp;
#[cfg(feature = "benchmark_io")]
mod loader_vrptw;
mod local_search;
mod params;
mod pred_queue;
mod population;
mod problem;
mod reverse_mode;
mod sequence;
mod solver;
pub use solver::Solver;
use tig_challenges::vehicle_routing::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Err(anyhow!("No feasible solution found")),
    }
}

#[allow(dead_code)]
pub fn help() {
    println!("No help information available.");
}
