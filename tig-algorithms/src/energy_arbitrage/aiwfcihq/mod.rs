use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod track_baseline;
mod track_capstone;
mod track_congested;
mod track_dense;
mod track_multiday;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15 => track_baseline::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 30 => track_congested::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 50 => track_multiday::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 80 => track_dense::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 150 => track_capstone::solve_challenge(challenge, save_solution, hyperparameters),
        n => Err(anyhow!("aycdicdb: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("aycdicdb - inspiration taken from the AI evolved titan_v2 and near_arb_v1");
}
