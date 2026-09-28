use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

pub mod hyperparameters;
pub mod track_t1;
pub mod track_t2;
pub mod track_t3;
pub mod track_t4;
pub mod track_t5;

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
        n if n <= 15 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t1_defaults());
            track_t1::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 30 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t2_defaults());
            track_t2::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t3_defaults());
            track_t3::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t4_defaults());
            track_t4::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 150 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t5_defaults());
            track_t5::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("energy_arbitrage: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("energy_arbitrage solver");
}
