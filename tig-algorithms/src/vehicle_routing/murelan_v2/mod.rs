use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
mod stopping;
mod progress;
mod baseline;
mod compression;
mod constructive;
mod genetic;
mod individual;
mod loader_tig;
mod local_search;
mod params;
mod population;
mod pred_queue;
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
    println!("Send {{\"target_quality\": T}}, T = the track's qualifying quality (199543 on chain = 0.199543).");
    println!("Update T when it is more than 0.003 from the line; missing = the README default.");
    println!("{{\"early_stop\": 0}} turns stopping off and gives the hgs_epure output.");
}
