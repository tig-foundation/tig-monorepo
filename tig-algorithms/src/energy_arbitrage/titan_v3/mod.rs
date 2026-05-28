use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_t49;
pub mod track_t50;
pub mod track_t51;
pub mod track_t52;
pub mod track_t53;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15 => {
            let hp = merge_hp(hyperparameters, vec![
                ("k_clusters", Value::Number(Number::from(80u64))),
            ]);
            track_t49::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 30 => {
            let hp = merge_hp(hyperparameters, vec![
                ("action_grid", Value::Number(Number::from(10u64))),
                ("lns_max_lines", Value::Number(Number::from(12u64))),
            ]);
            track_t50::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => track_t51::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 80 => track_t52::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 150 => track_t53::solve_challenge(challenge, save_solution, hyperparameters),
        n => Err(anyhow!("titan_v2: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("titan_v3");
}
