// r4200 route of sat_hybrid_sp128 (c001_a123), vendored unchanged: sat_hybrid_v4
// engine_b with the survey-propagation reinforcement seed and its early hand-off.
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[path = "sp128_engine_b.rs"]
mod engine_b;
#[path = "sp128_engine_spr.rs"]
mod engine_spr;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();
    match (nv, nc) {
        (100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
        _ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
    }
}
