// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[path = "hybrid_engine_b.rs"]
mod engine_b;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Per-track dispatch by (num_variables, num_clauses); tuned defaults baked per track.
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();
    match (nv, nc) {
        (100000, 415000) => engine_b::solve(challenge, save_solution, hyperparameters),
        (5000, 21335) => engine_b::solve(challenge, save_solution, hyperparameters),
        (100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
        _ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
    }
}

#[allow(dead_code)]
pub fn help() {
    println!("sat_hybrid - per-track SAT solver");
}
