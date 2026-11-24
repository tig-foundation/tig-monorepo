use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

pub mod baseline_policy;
pub mod hyperparameters;
pub mod kernels;
pub mod track_t1;
pub mod track_t3;
pub mod track_t5;
pub mod wasserwert;
pub mod wide;

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
        n if n <= 30 => wasserwert::solve_challenge(challenge, save_solution, user_hp),
        n if n <= 50 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t3_defaults());
            track_t3::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => wasserwert::solve_challenge(challenge, save_solution, user_hp),
        n if n <= 150 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t5_defaults());
            track_t5::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("harimox_v2: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("harimox_v2 - send {{}} on every track but capstone; see README.md.");
    println!("capstone: send {{\"target_quality\": T}}, T = the capstone qualifying quality (3806936 on chain = 3.806936), default 3.85.");
    println!("Update T when the line moves by more than 0.1; 0 turns the target off.");
}
