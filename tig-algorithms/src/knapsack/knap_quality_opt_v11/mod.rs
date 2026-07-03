pub mod track_t39;
pub mod track_t40;
pub mod track_t41;
pub mod track_t42;
pub mod track_t43;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

/// Centralized hyperparameters per track (Phase 1: empty placeholder to satisfy
/// V2 convention `mod_rs_no_hyperparameters`). HP overrides continue to flow
/// through the legacy `Option<Map<String, Value>>` channel inside each track,
/// guaranteeing bit-identical behavior with code_raw.
#[derive(Clone, Debug, Default)]
pub struct Hparams {}

impl Hparams {
    pub fn for_t39() -> Self { Self::default() }
    pub fn for_t40() -> Self { Self::default() }
    pub fn for_t41() -> Self { Self::default() }
    pub fn for_t42() -> Self { Self::default() }
    pub fn for_t43() -> Self { Self::default() }
}

pub fn help() {
    println!("knap_quality_opt v11 (refacto V2) - per-track files (T39/T40/T41/T42/T43)");
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

    // Dispatch identical to code_raw/mod.rs. Mapping from challenge params to
    // track_index is dictated by tracks.params in BDD V2:
    //   t39 = (n_items=1000, budget=10)  → 1000, mid budget
    //   t40 = (n_items=1000, budget=25)  → 1000, large budget
    //   t41 = (n_items=1000, budget=5)   → 1000, tight budget
    //   t42 = (n_items=5000, budget=10)  → 5000, mid budget
    //   t43 = (n_items=5000, budget=25)  → 5000, large budget
    match (n, budget_pct) {
        (1000, b) if b <= 7  => track_t41::solve(challenge, save, hp),
        (1000, b) if b <= 17 => track_t39::solve(challenge, save, hp),
        (1000, _)            => track_t40::solve(challenge, save, hp),
        (5000, b) if b <= 17 => track_t42::solve(challenge, save, hp),
        (5000, _)            => track_t43::solve(challenge, save, hp),
        _ => Err(anyhow::anyhow!(
            "knap_quality_opt_v11: unknown track config (n_items={}, budget_pct={})",
            n, budget_pct
        )),
    }
}
