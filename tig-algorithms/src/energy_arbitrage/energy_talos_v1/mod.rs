use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

pub mod hyperparameters;
pub mod kernels;
pub mod track_t1;
pub mod track_t3;
pub mod track_t5;
pub mod wasserwert_t2;
pub mod wasserwert_t4;
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
        n if n <= 30 => {
            // Frozen T2 research configuration; explicit caller HP wins.
            let hp = merge_hp(user_hp, vec![
                ("grid", hyperparameters::n(28)),
                ("sdp_q", hyperparameters::n(5)),
                ("sdp_qj", hyperparameters::n(2)),
                ("cong_prem", hyperparameters::n(1)),
                ("lp_seg", hyperparameters::n(1)),
                ("samples", hyperparameters::n(4)),
                ("gate_k", hyperparameters::n(8)),
            ]);
            wasserwert_t2::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t3_defaults());
            track_t3::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => wasserwert_t4::solve_challenge(challenge, save_solution, user_hp),
        n if n <= 150 => {
            let hp = merge_hp(user_hp, hyperparameters::track_t5_defaults());
            track_t5::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("energy_talos_v1: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("energy_talos_v1: use null or {{}} on all five C008 tracks.");
    println!("T1 R17; T2 R20; T3 R20; T4 R13; T5 R15. Explicit HP overrides defaults.");
    println!("T2 defaults: grid=28, sdp_q=5, sdp_qj=2, cong_prem=1, lp_seg=1, samples=4, gate_k=8.");
    println!("See README.md for inherited HP interfaces, provenance and quality/speed tradeoffs.");
}
