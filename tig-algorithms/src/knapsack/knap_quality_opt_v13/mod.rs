use anyhow::Result;
use serde_json::{Map, Value, Number};
use tig_challenges::knapsack::*;

mod track_t39;
mod track_t40;
mod track_t41;
mod track_t42;
mod track_t43;

fn merge_hp(user: &Option<Map<String, Value>>, defs: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user.clone().unwrap_or_default();
    for (k, v) in defs { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn n(v: i64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    let n_items = challenge.num_items;
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 { ((challenge.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };
    match (n_items, budget_pct) {
        (1000, b) if b <= 7  => { let hp = merge_hp(hp, vec![("core_final_mode", n(2))]); track_t41::solve(challenge, save, &hp) }
        (1000, b) if b <= 17 => { let hp = merge_hp(hp, vec![("n_perturbation_rounds", n(16))]); track_t39::solve(challenge, save, &hp) }
        (1000, _)            => { let hp = merge_hp(hp, vec![("n_full_restarts", n(60))]); track_t40::solve(challenge, save, &hp) }
        (5000, b) if b <= 17 => { let hp = merge_hp(hp, vec![("lagr_mode", n(3))]); track_t42::solve(challenge, save, &hp) }
        (5000, _)            => {
            let hp = merge_hp(hp, vec![
                ("rc_mode", n(5)), ("rc_fraction", f(0.45)), ("refine_mode", n(5)), ("use_flat_iv", n(1)),
                ("base_profile", n(1)), ("use_rflip_screen", n(0)), ("use_screen_break", n(1)), ("use_light_fast_replace", n(1)),
            ]);
            track_t43::solve(challenge, save, &hp)
        }
        _ => Err(anyhow::anyhow!("unknown track config (n_items={}, budget_pct={})", n_items, budget_pct)),
    }
}
pub fn help() { println!("knap_quality_opt_v13 - per-track knapsack solver"); }
