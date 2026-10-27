// satchel_cluster - quadratic knapsack solver (c003).
// satchel (c003_a154, unchanged in track1.rs..track5.rs) followed, on the tracks with budget <= 17 %, by an exact
// local-search polish of its packing (polish.rs): add, 1-1, 2-1, 1-2 and cluster moves. The generator's items form
// clusters with strong mutual ties and weak ties to the rest; exchanging a whole cluster for another of equal
// weight is invisible to per-item scoring, so the polish moves clusters as blocks. Every polish move is exact and
// kept only if the packing's value rises; the polished packing is saved only if strictly better, so the result is
// never below satchel's. Deterministic: no clock, no threads, no randomness.

pub mod track1;
pub mod track2;
pub mod track3;
pub mod track4;
pub mod track5;
mod polish;

use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;
use tig_challenges::knapsack::*;

/// Safety stop on polish moves (a local optimum is reached far earlier).
const MAX_MOVES: u32 = 10_000;

pub fn help() {
    println!(
        "satchel followed by an exact local-search polish with cluster moves\n\
         \n\
         Tracks with budget <= 17 % (n_items=1000 budget 5/10, n_items=5000 budget 10): satchel, then the polish.\n\
         Tracks with budget > 17 %: satchel unchanged.\n\
         \n\
         Hyperparameters: none needed (defaults are the operating point). satchel's hyperparameters are passed\n\
         through; the polish also reads polish_taus (array of cluster thresholds, default [150]),\n\
         polish_cluster_max (default 24) and polish_swap_tries (default 8)."
    );
}

fn compute_budget_pct(challenge: &Challenge) -> u32 {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    }
}

/// satchel's dispatcher, as in c003_a154.
fn satchel_solve(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_items;
    let budget_pct = compute_budget_pct(challenge);
    match (n, budget_pct) {
        (1000, b) if b <= 7 => track1::solve(challenge, save, hp),
        (1000, b) if b <= 17 => track2::solve(challenge, save, hp),
        (1000, _) => track3::solve(challenge, save, hp),
        (5000, b) if b <= 17 => track4::solve(challenge, save, hp),
        (5000, _) => track5::solve(challenge, save, hp),
        _ => Err(anyhow::anyhow!(
            "QKBP dispatcher: unknown track config (num_items={}, budget_pct={}). Expected num_items in {{1000, 5000}}.",
            n, budget_pct
        )),
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<Map<String, Value>>,
) -> Result<()> {
    if compute_budget_pct(challenge) > 17 {
        return satchel_solve(challenge, save, hp);
    }
    let mut cfg = polish::Cfg::default();
    if let Some(m) = hp {
        if let Some(t) = m.get("polish_taus").and_then(|v| v.as_array()) {
            cfg.taus = t.iter().filter_map(|x| x.as_i64()).map(|x| x as i32).collect();
        }
        if let Some(x) = m.get("polish_cluster_max").and_then(|v| v.as_u64()) {
            cfg.cluster_max = x as usize;
        }
        if let Some(x) = m.get("polish_swap_tries").and_then(|v| v.as_u64()) {
            cfg.swap_tries = x as usize;
        }
    }
    let last: RefCell<Option<Solution>> = RefCell::new(None);
    let first = |s: &Solution| -> Result<()> {
        *last.borrow_mut() = Some(s.clone());
        save(s)
    };
    satchel_solve(challenge, &first, hp)?;
    let sol = match last.into_inner() {
        Some(s) => s,
        None => return Ok(()),
    };
    let base = challenge.evaluate_total_value(&sol)? as i64;
    let mut p = polish::Polish::new(challenge, &sol.items, &cfg);
    if p.run(MAX_MOVES) > 0 && p.value > base {
        let mut out = sol.clone();
        out.items = p.items();
        save(&out)?;
    }
    Ok(())
}
