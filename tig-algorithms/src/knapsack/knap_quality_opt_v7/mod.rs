// TIG's UI uses the pattern `tig_challenges::knapsack` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

mod v71_solver;
mod basin_solver;
mod bs_construct;
mod bs_ils;
mod bs_local_search;
mod bs_params;
mod bs_refinement;
mod bs_types;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub window_k: Option<usize>,
    pub core_dp: Option<usize>,
    pub ils_rounds: Option<usize>,
    pub perturb_base: Option<usize>,
    pub n_starts: Option<usize>,
}

pub fn solve_challenge(
    ch: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp_map: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = ch.num_items;
    let sum_w: u64 = ch.weights.iter().map(|&w| w as u64).sum();
    let budget_pct = if sum_w > 0 { ((ch.max_weight as u64) * 100 / sum_w) as u32 } else { 10 };

    match (n, budget_pct) {
        (0..=1200, 0..=7)  => v71_solver::solve(ch, save, hp_map),
        (0..=1200, 8..=15) => v71_solver::solve(ch, save, hp_map),
        _                  => basin_solver::solve(ch, save),
    }
}

pub fn help() {
    println!("knap_quality_opt_v7: Per-track hybrid solver — sparse VND + basin discovery ILS");
}
