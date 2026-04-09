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
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
}

pub fn help() {
    println!("SAT imp v1 - Fully modular per-track WalkSAT solver");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    });

    let nv = challenge.num_variables;
    let nc_raw = challenge.clauses.len();
    let density_raw = nc_raw as f64 / nv as f64;
    let ratio = (density_raw * 1000.0).round() as u32;

    match (nv, ratio) {
        (5000, 4267) => track1::solve(challenge, &hp, save_solution),
        (7500, 4267) => track2::solve(challenge, &hp, save_solution),
        (10000, 4267) => track3::solve(challenge, &hp, save_solution),
        (100000, 4150) => track4::solve(challenge, &hp, save_solution),
        (100000, 4200) => track5::solve(challenge, &hp, save_solution),
        _ => track2::solve(challenge, &hp, save_solution),
    }
}
