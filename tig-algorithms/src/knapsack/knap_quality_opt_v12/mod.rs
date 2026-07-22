// TIG's UI uses the pattern `tig-algorithms/src/<challenge>/<algo_name>/mod.rs`
mod track_t39;
mod track_t40;
mod track_t41;
mod track_t42;
mod track_t43;

use anyhow::Result;
use serde_json::{Map, Number, Value};
use tig_challenges::knapsack::*;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn u(v: u64) -> Value { Value::Number(Number::from(v)) }

fn compute_budget_pct(challenge: &Challenge) -> u32 {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    if sum_w > 0 { ((challenge.max_weight as u64) * 100 / sum_w) as u32 } else { 10 }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    // Per-track tuned defaults baked in; user-supplied hyperparameters override them.
    let n = challenge.num_items;
    let budget_pct = compute_budget_pct(challenge);
    match (n, budget_pct) {
        (1000, b) if b <= 7 => track_t41::solve_challenge(challenge, save, hp),
        (1000, b) if b <= 17 => track_t39::solve(challenge, save, hp),
        (1000, _) => track_t40::solve(challenge, save, hp),
        (5000, b) if b <= 17 => {
            let hp = merge_hp(hp, vec![
                ("sa_iter", u(0)), ("sa_rounds", u(1)), ("ils_rounds", u(115)),
                ("n_sa_members", u(0)), ("bounded_2_2_k", u(20)), ("ils_vnd_level", u(3)),
                ("n_crossover_gen", u(16)), ("n_random_starts", u(2)),
                ("perturb_max_frac", u(7)), ("perturb_base_frac", u(7)),
                ("ils_restart_interval", u(15)),
            ]);
            track_t42::solve_challenge(challenge, save, &hp)
        }
        (5000, _) => {
            let hp = merge_hp(hp, vec![
                ("sa_iter", u(0)), ("window_k", u(280)), ("sa_rounds", u(0)),
                ("ils_rounds", u(160)), ("core_half_dp", u(55)), ("bounded_2_2_k", u(20)),
                ("ils_vnd_level", u(1)), ("n_crossover_gen", u(8)), ("n_full_restarts", u(13)),
                ("n_random_starts", u(4)), ("perturb_max_frac", u(7)),
                ("perturb_base_frac", u(6)), ("ils_restart_interval", u(14)),
            ]);
            track_t43::solve_challenge(challenge, save, &hp)
        }
        _ => Err(anyhow::anyhow!("unsupported instance (num_items={}, budget_pct={})", n, budget_pct)),
    }
}

pub fn help() {
    println!("knap_quality_opt_v12 - per-track knapsack solver");
}
