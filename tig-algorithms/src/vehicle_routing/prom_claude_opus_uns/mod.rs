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
use std::cell::RefCell;
use std::time::Instant;

const WALL_BUDGET_SECS: f64 = 248.0;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let t0 = Instant::now();
    let best_feasible_cost: RefCell<i64> = RefCell::new(i64::MAX);

    // Feasibility-filtered save: only forward strictly improving FEASIBLE solutions
    // to the harness. Protects against (a) seed-perturbed runs producing solutions
    // valid for the perturbed seed only, (b) the GA's initial dummy-route save
    // overwriting a good solution from a prior portfolio entry, (c) regressions.
    let filtered_save = |sol: &Solution| -> Result<()> {
        match challenge.evaluate_total_distance(sol) {
            Ok(cost) => {
                let cost_i64 = cost as i64;
                let mut best = best_feasible_cost.borrow_mut();
                if cost_i64 < *best {
                    *best = cost_i64;
                    drop(best);
                    save_solution(sol)
                } else {
                    Ok(())
                }
            }
            Err(_) => Ok(()),
        }
    };

    let base_hp: Map<String, Value> = hyperparameters.clone().unwrap_or_else(Map::new);

    // Entry A: budget-absorbing primary run.
    // exploration_level=4 with allow_swap3 re-enabled (preset disables it at L4),
    // very high iteration caps so termination comes from stagnation / wall-kill,
    // not artificial limits. This entry is intended to use ~75% of the wall budget.
    let mut params_a = base_hp.clone();
    params_a.insert("exploration_level".to_string(), Value::from(4u64));
    params_a.insert("allow_swap3".to_string(), Value::Bool(true));
    params_a.insert("max_it_total".to_string(), Value::from(1_000_000u64));
    params_a.insert("max_it_noimprov".to_string(), Value::from(50_000u64));
    let _ = Solver::solve_challenge_instance(challenge, &Some(params_a), Some(&filtered_save));

    // Entry B: seed-perturbed insurance run, only if appreciable budget remains.
    // Drives the GA into a structurally distinct exploration trajectory via seed
    // XOR. Tighter no-improv cap so it terminates well before wall-kill if A used
    // most of the budget. The save filter validates against the ORIGINAL challenge,
    // so any solution that's only valid for the perturbed seed is dropped.
    if t0.elapsed().as_secs_f64() < WALL_BUDGET_SECS * 0.65 {
        let mut perturbed = challenge.clone();
        perturbed.seed[0] ^= 0x5A_u8;
        perturbed.seed[1] ^= 0xA5_u8;

        let mut params_b = base_hp.clone();
        params_b.insert("exploration_level".to_string(), Value::from(4u64));
        params_b.insert("allow_swap3".to_string(), Value::Bool(true));
        params_b.insert("max_it_total".to_string(), Value::from(1_000_000u64));
        params_b.insert("max_it_noimprov".to_string(), Value::from(20_000u64));

        let _ = Solver::solve_challenge_instance(&perturbed, &Some(params_b), Some(&filtered_save));
    }

    Ok(())
}

pub fn help() {
}
