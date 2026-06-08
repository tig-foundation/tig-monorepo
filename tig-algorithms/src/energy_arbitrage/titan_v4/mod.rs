use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_t49;
pub mod track_t50;
pub mod track_t52;
pub mod t51_engine;
pub mod t53_engine;

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
                ("use_lp", Value::Bool(true)),
            ]);
            track_t50::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => t51_engine::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 80 => track_t52::solve_challenge(challenge, save_solution, hyperparameters),
        n if n <= 150 => {
            let hp = merge_hp(hyperparameters, vec![
                ("use_dp_seed", Value::Bool(false)),
                ("bisect_iters", Value::Number(Number::from(8u64))),
                ("dp_soc_levels", Value::Number(Number::from(5u64))),
                ("use_zero_seed", Value::Bool(false)),
                ("proj_max_iters", Value::Number(Number::from(12u64))),
                ("grad_outer_iters", Value::Number(Number::from(3u64))),
                ("lookahead_horizon", Value::Number(Number::from(4u64))),
                ("policy_action_levels", Value::Number(Number::from(5u64))),
            ]);
            t53_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("titan_v4: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("titan_v4");
}
