//! mod.rs — QKBP Track Dispatcher
//!
//! Follows TIG V2 submission convention exactly, mirroring
//! the reference innovator's mod.rs structure:
//!   - pub mod declarations for each track
//!   - pub fn solve_challenge dispatches by (num_items, budget_pct)
//!   - Empty Hparams struct (V2 convention: mod_rs_no_hyperparameters)
//!   - HP overrides flow through Option<Map<String, Value>> per track
//!
//! Routing table (mirrors code_raw):
//!   n_items=1000, budget_pct<= 7  ->  track1  (tight budget,  ~5 budgets)
//!   n_items=1000, budget_pct<=17  ->  track2  (mid   budget, ~10 budgets)
//!   n_items=1000, budget_pct> 17  ->  track2  (large budget, ~25 budgets)
//!   n_items=5000, budget_pct<=17  ->  track4  (mid   budget, ~10 budgets)
//!   n_items=5000, budget_pct> 17  ->  track2  (large budget, ~25 budgets)

pub mod track1;
pub mod track2;
pub mod track3;
pub mod track4;
pub mod track5;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

/* -------------------------------------------------------
 * Hyperparameters — empty placeholder per TIG V2 convention
 * (mod_rs_no_hyperparameters).
 * Per-track HP overrides continue to flow through the legacy
 * Option<Map<String, Value>> channel inside each track,
 * guaranteeing bit-identical behaviour with code_raw.
 * ------------------------------------------------------- */
#[derive(Clone, Debug, Default)]
pub struct Hparams {}

impl Hparams {
    pub fn for_track1() -> Self { Self::default() }
    pub fn for_track2() -> Self { Self::default() }
    pub fn for_track4() -> Self { Self::default() }
    pub fn for_track3() -> Self { Self::default() }
    pub fn for_track5() -> Self { Self::default() }
}

pub fn help() {
    println!(
        "QKBP solver (TIG V2) — track1 / track2 / track4\n\
         \n\
         Routing by (num_items, budget_pct):\n\
           n_items=1000, budget_pct<= 7  ->  track1\n\
           n_items=1000, budget_pct<=17  ->  track2\n\
           n_items=1000, budget_pct> 17  ->  track3\n\
           n_items=5000, budget_pct<=17  ->  track4\n\
           n_items=5000, budget_pct> 17  ->  track5"
    );
}

/* -------------------------------------------------------
 * compute_budget_pct
 *
 * Identical to the reference submission:
 *   budget_pct = (max_weight * 100) / sum_of_all_weights
 * Safe default of 10 when sum_w == 0 (degenerate instance).
 * ------------------------------------------------------- */
fn compute_budget_pct(challenge: &Challenge) -> u32 {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    }
}

/* -------------------------------------------------------
 * solve_challenge — primary TIG entry point
 *
 * Signature is mandated by the TIG V2 challenge interface.
 * Dispatches to the correct track module based on
 * (num_items, budget_pct), identical to the reference
 * submission's match block.
 * ------------------------------------------------------- */
pub fn solve_challenge(
    challenge: &Challenge,
    save:      &dyn Fn(&Solution) -> Result<()>,
    hp:        &Option<Map<String, Value>>,
) -> Result<()> {

    let n          = challenge.num_items;
    let budget_pct = compute_budget_pct(challenge);

    match (n, budget_pct) {
        /* ---- n_items = 1000 ------------------------------------ */
        // tight budget (~5 budget values)
        (1000, b) if b <= 7  => track1::solve(challenge, save, hp),
        // mid budget (~10 budget values)
        (1000, b) if b <= 17 => track2::solve(challenge, save, hp),
        // large budget (~25 budget values)
        (1000, _)            => track3::solve(challenge, save, hp),

        /* ---- n_items = 5000 ------------------------------------ */
        // mid budget (~10 budget values)
        (5000, b) if b <= 17 => track4::solve(challenge, save, hp),
        // large budget (~25 budget values)
        (5000, _)            => track5::solve(challenge, save, hp),

        /* ---- Unknown — return descriptive error ---------------- */
        _ => Err(anyhow::anyhow!(
            "QKBP dispatcher: unknown track config \
             (num_items={}, budget_pct={}). \
             Expected num_items in {{1000, 5000}}.",
            n, budget_pct
        )),
    }

}