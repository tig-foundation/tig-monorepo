mod track_t39;
mod track_t40;
mod track_t41;
mod track_t42;
mod track_t43;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

pub fn help() {
    println!("knap_quality_opt v10 - per-track files (T39/T40/T41/T42/T43)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_items;
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    };

    match (n, budget_pct) {
        (1000, b) if b <= 7  => track_t41::solve_challenge(challenge, save, hp),
        (1000, b) if b <= 17 => track_t39::solve_challenge(challenge, save, hp),
        (1000, _)            => track_t40::solve_challenge(challenge, save, hp),
        (5000, b) if b <= 17 => track_t42::solve_challenge(challenge, save, hp),
        (5000, _)            => track_t43::solve_challenge(challenge, save, hp),
        _ => Err(anyhow::anyhow!(
            "knap_quality_opt_v10: unknown track config (n_items={}, budget_pct={})",
            n, budget_pct
        )),
    }
}
