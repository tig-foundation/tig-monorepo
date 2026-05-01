use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tig_challenges::zk_optimization::{baselines::remove_aliases_and_scales, *};

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("No help information provided.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let circuit_star = remove_aliases_and_scales(&challenge.circuit_c0);
    let solution = challenge.build_solution(&circuit_star);
    save_solution(&solution.unwrap())?;
    Ok(())
}
