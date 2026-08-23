// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::satisfiability::*;

mod engine_a;
mod engine_b;
mod engine_c;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn u(v: u64) -> Value { Value::Number(Number::from(v)) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Per-track dispatch by (num_variables, num_clauses); tuned defaults baked per track.
    let nv = challenge.num_variables;
    let nc = challenge.clauses.len();
    match (nv, nc) {
        (10000, 42670) => {
            let hp = merge_hp(hyperparameters, vec![("max_fuel_high", u(180000000000))]);
            engine_a::solve(challenge, save_solution, &hp)
        }
        (100000, 415000) => engine_b::solve(challenge, save_solution, hyperparameters),
        (5000, 21335) => engine_b::solve(challenge, save_solution, hyperparameters),
        (7500, 32002) => {
            let hp = merge_hp(hyperparameters, vec![("target_max_fuel", u(200000000000))]);
            engine_c::solve(challenge, save_solution, &hp)
        }
        (100000, 420000) => engine_b::solve(challenge, save_solution, hyperparameters),
        _ => Err(anyhow!("unknown track config (num_variables={}, num_clauses={})", nv, nc)),
    }
}

pub fn help() {
    println!("sat_hybrid - per-track SAT solver");
}
