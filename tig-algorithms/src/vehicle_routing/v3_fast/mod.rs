// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs` to identify algorithms
pub mod instance;
pub mod config;
pub mod route_eval;
pub mod builder;
pub mod solution;
pub mod gene_pool;
pub mod operators;
pub mod evolution;
pub mod runner;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() -> String {
    String::from("v3_fast: HGS-VRPTW speed-optimized. Reduced granularity, LS pass limit, no swap3.")
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    runner::Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))
}
