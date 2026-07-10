use anyhow::{anyhow, Result};
use serde_json::{Map, Number, Value};
use tig_challenges::energy_arbitrage::*;

pub mod track_t49;
pub mod track_t50;
pub mod t51_engine;
pub mod t52_engine;
pub mod t53_engine;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

fn n(v: u64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }
fn b(v: bool) -> Value { Value::Bool(v) }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match challenge.num_batteries {
        n if n <= 15 => {
            let hp = merge_hp(hyperparameters, vec![
                ("soc_levels", self::n(2001)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(125)),
                ("ternary_iters", self::n(125)),
                ("convergence_tol", f(1e-15)),
                ("k_clusters", self::n(80)),
                ("deflator_iters", self::n(75)),
                ("lp_refine_sweeps", self::n(15)),
                ("cg_iters", self::n(100)),
                ("use_lp", b(true)),
                ("use_sdp", b(true)),
                ("use_cg", b(true)),
                ("network_derating", f(1.00)),
            ]);
            track_t49::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 30 => {
            let hp = merge_hp(hyperparameters, vec![
                ("soc_levels", self::n(201)),
                ("action_grid", self::n(40)),
                ("asca_iters", self::n(20)),
                ("ternary_iters", self::n(100)),
                ("deflator_iters", self::n(250)),
                ("network_derating", f(0.22)),
                ("max_admm_iters", self::n(50)),
                ("lp_total_pivots", self::n(75000)),
                ("dw_total_pivot_budget", self::n(40000)),
                ("lns_lp_pivots_total", self::n(30000)),
                ("use_lp", b(true)),
                ("use_dw", b(true)),
                ("use_lns", b(true)),
                ("use_kkt", b(true)),
            ]);
            track_t50::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 50 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(97)),
                ("dp_action_levels", self::n(17)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(400)),
                ("grad_outer_iters", self::n(500)),
                ("grad_ls_iters", self::n(30)),
                ("bisect_iters", self::n(150)),
                ("coord_polish_passes", self::n(10)),
                ("lookahead_horizon", self::n(24)),
            ]);
            t51_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 80 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(33)),
                ("dp_action_levels", self::n(17)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(400)),
                ("grad_outer_iters", self::n(125)),
                ("grad_ls_iters", self::n(30)),
                ("bisect_iters", self::n(150)),
                ("coord_polish_passes", self::n(5)),
                ("lookahead_horizon", self::n(24)),
            ]);
            t52_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n if n <= 150 => {
            let hp = merge_hp(hyperparameters, vec![
                ("dp_soc_levels", self::n(33)),
                ("dp_action_levels", self::n(17)),
                ("policy_action_levels", self::n(65)),
                ("proj_max_iters", self::n(80)),
                ("grad_outer_iters", self::n(25)),
                ("grad_ls_iters", self::n(6)),
                ("bisect_iters", self::n(30)),
                ("coord_polish_passes", self::n(1)),
                ("lookahead_horizon", self::n(24)),
            ]);
            t53_engine::solve_challenge(challenge, save_solution, &hp)
        }
        n => Err(anyhow!("titan_v5: unsupported num_batteries={}", n)),
    }
}

pub fn help() {
    println!("titan_v5");
}
