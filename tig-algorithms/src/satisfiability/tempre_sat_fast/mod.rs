mod weighted;
mod track1;
mod track2;
mod track3;
mod track4;
mod track5;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {}

pub fn help() {
    println!("Per-track SAT solver: tempre_sat weighted ProbSAT on CSR storage.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();
    if nc == 0 || nv == 0 {
        save_solution(&Solution { variables: vec![false; nv] })?;
        return Ok(());
    }
    let ratio = (nc as f64 / nv as f64 * 1000.0).round() as u32;

    match (nv, ratio) {
        (5000, 4267)   => track1::solve(challenge, save_solution),
        (7500, 4267)   => track2::solve(challenge, save_solution),
        (10000, 4267)  => track3::solve(challenge, save_solution),
        (100000, 4150) => track4::solve(challenge, save_solution),
        (100000, 4200) => track5::solve(challenge, save_solution),
        _ => track1::solve(challenge, save_solution),
    }
}