// water_thimble_001 - energy_arbitrage solver.
//
// Dispatches each instance by battery count:
//   * 16..=50 batteries (the congested and multiday tracks): water_dp, a per-battery stochastic
//     water-value dynamic program with network-aware fleet allocation.
//   * every other size: gridflow's per-track engines (t1, t4, t5), unchanged apart from the
//     removal of their lookup of the private price seed.
//
// The gridflow engines and this file's hyperparameter merge are derived from gridflow by NVX
// (Copyright 2026 NVX, tig-algorithms/src/energy_arbitrage/gridflow), used under the TIG
// Innovator Outbound Game License.

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

pub mod hyperparameters;
pub mod track_t1;
pub mod track_t4;
pub mod track_t5;
pub mod water_dp;

fn merge_hp(
    user_hp: &Option<Map<String, Value>>,
    defaults: Vec<(&str, Value)>,
) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    user_hp: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if (16..=50).contains(&n) => water_dp::solve_challenge(challenge, save_solution, user_hp),
        n if n <= 15 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t1_defaults());
            track_t1::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t4_defaults());
            track_t4::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 150 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t5_defaults());
            track_t5::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("water_thimble_001: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("water_thimble_001 - energy_arbitrage solver dispatching by battery count:");
    println!("  16..=50 batteries: water-value dynamic program (water_dp)");
    println!("  otherwise: gridflow's per-track engines (by NVX)");
}
