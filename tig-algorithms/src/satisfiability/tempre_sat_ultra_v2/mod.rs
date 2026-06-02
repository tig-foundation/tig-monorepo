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
    println!("Hybrid SAT solver: tempre_sat_ultra (tracks 1,5) + cambium (tracks 2,3,4).");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
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
        (7500, 4267)   => {
            let hp = parse_hp(hyperparameters);
            track2::solve(challenge, &hp, save_solution)
        }
        (10000, 4267)  => {
            let hp = parse_hp(hyperparameters);
            track3::solve(challenge, &hp, save_solution)
        }
        (100000, 4150) => {
            let hp = parse_hp(hyperparameters);
            track4::solve(challenge, &hp, save_solution)
        }
        (100000, 4200) => track5::solve(challenge, save_solution),
        _ => track1::solve(challenge, save_solution),
    }
}

fn parse_hp(hyperparameters: &Option<Map<String, Value>>) -> Option<Hyperparameters> {
    hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    })
}