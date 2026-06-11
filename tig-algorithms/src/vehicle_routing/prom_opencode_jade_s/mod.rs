use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

mod instance;
mod config;
mod route_eval;
mod solution;
mod builder;
mod operators;
mod gene_pool;
mod evolution;
mod runner;

pub use runner::Solver;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

fn select_default_hyperparameters(challenge: &Challenge) -> Option<Map<String, Value>> {
    let n = challenge.num_nodes;
    if n == 0 {
        return None;
    }

    let total_demand: i64 = challenge.demands.iter().skip(1).map(|&d| d as i64).sum();
    let fleet_capacity = (challenge.max_capacity as i64).saturating_mul(challenge.fleet_size.max(1) as i64);
    let load_ratio = if fleet_capacity > 0 {
        total_demand as f64 / fleet_capacity as f64
    } else {
        0.0
    };

    let horizon = (challenge.due_times[0] - challenge.ready_times[0]).max(1) as f64;
    let mut width_sum = 0.0;
    let mut narrow_windows = 0usize;
    for i in 1..n {
        let width = (challenge.due_times[i] - challenge.ready_times[i]).max(0) as f64;
        width_sum += width;
        if width / horizon < 0.16 {
            narrow_windows += 1;
        }
    }
    let customers = n.saturating_sub(1).max(1);
    let avg_width = width_sum / customers as f64;
    let tw_tightness = 1.0 - (avg_width / horizon).clamp(0.0, 1.0);
    let narrow_ratio = narrow_windows as f64 / customers as f64;

    let exploration_level = if n <= 25 {
        1
    } else if n <= 80 {
        if tw_tightness > 0.72 || narrow_ratio > 0.35 || load_ratio > 0.82 {
            3
        } else {
            2
        }
    } else if n <= 250 {
        if tw_tightness > 0.68 || narrow_ratio > 0.30 || load_ratio > 0.86 {
            4
        } else {
            3
        }
    } else if tw_tightness > 0.76 || narrow_ratio > 0.38 || load_ratio > 0.92 {
        4
    } else {
        3
    };

    let mut map = Map::new();
    map.insert(
        "exploration_level".to_string(),
        Value::from(exploration_level as u64),
    );
    Some(map)
}

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let tuned_hyperparameters;
    let params = if hyperparameters.is_none() {
        tuned_hyperparameters = select_default_hyperparameters(challenge);
        &tuned_hyperparameters
    } else {
        hyperparameters
    };

    match Solver::solve_challenge_instance(challenge, params, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Ok(()),
    }
}

pub fn help() {
}
