//! mod.rs — zero_knap_sa QKBP Track Dispatcher (self-contained bundle)
//!
//! Architecture: superfast_knap_v1 pipeline (HPF parametric min-cut
//! breakpoint seeds -> greedy construction -> DP core refinement -> VND
//! local search -> ILS) followed by a staged, fuel-gated multi-seed search
//! with Metropolis simulated-annealing acceptance in deep_polish on the
//! n_items=1000 mid-budget track (track2):
//!   Stage 1: breakpoint lambda sweep (16 grids, 200..4800)
//!   Stage 2: beta-variant greedy seeds (same breakpoints, beta weighting)
//!   Stage 3: crossover generations over the diverse elite pool
//!   Stage 4: final deep polish of the global best (SA acceptance,
//!            default sa_t0_bp=100, sa_decay_pm=970)
//! Every seed gets a deep polish (DP refinement + heavy VND + bounded
//! basin-local ILS). Improvements are persisted immediately via the save
//! callback. Effort is bounded by the runtime fuel counter.
//!
//! Tracks 1/3/4/5 (n_items=1000 tight/large, n_items=5000) use the proven
//! superfast_knap_v1 track pipelines, vendored into this bundle and
//! credited in README.md.
//!
//! Copyright 2026 AgentZero.
//! Identity of Submitter: AgentZero.
//! Licensed under the TIG Inbound Game License v2.0 or (at your option)
//! any later version.

pub mod track1;
pub mod track2;
pub mod track3;
pub mod track4;
pub mod track5;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Clone, Debug, Default)]
pub struct Hparams {}

pub fn help() {
    println!(
        "zero_knap_sa — QKBP solver: superfast pipeline + staged multi-seed SA search\n\nRouting by (num_items, budget_pct):\n  n_items=1000, budget_pct<= 7  ->  track1 (superfast pipeline)\n  n_items=1000, budget_pct<=17  ->  track2 (pipeline + staged multi-seed SA search)\n  n_items=1000, budget_pct> 17  ->  track3 (superfast pipeline)\n  n_items=5000, budget_pct<=17  ->  track4 (superfast pipeline)\n  n_items=5000, budget_pct> 17  ->  track5 (superfast pipeline)\n\ntrack2 hyperparameters:\n  ils_rounds, window_k, core_half_dp, n_lambda_values (pipeline, as superfast)\n  seed_lambdas  breakpoint lambda grids swept in stage 1 (default 16, max 16)\n  beta_seeds    beta-variant greedy seeds in stage 2 (default 4, max 4)\n  cross_gens   crossover generations in stage 3 (default 12)\n  mini_ils      stall-driven basin ILS rounds per seed polish (default 60)\n  elite_k       elite pool size (default 8)\n  sa_t0_bp      SA initial temperature in basis points of seed value (0=off, default 100)\n  sa_decay_pm   SA temperature decay per ILS round, per-mille (default 970)"
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

pub fn solve_challenge(
    challenge: &Challenge,
    save:      &dyn Fn(&Solution) -> Result<()>,
    hp:        &Option<Map<String, Value>>,
) -> Result<()> {
    let n          = challenge.num_items;
    let budget_pct = compute_budget_pct(challenge);

    match (n, budget_pct) {
        (1000, b) if b <= 7  => track1::solve(challenge, save, hp),
        (1000, b) if b <= 17 => track2::solve(challenge, save, hp),
        (1000, _)            => track3::solve(challenge, save, hp),
        (5000, b) if b <= 17 => track4::solve(challenge, save, hp),
        (5000, _)            => track5::solve(challenge, save, hp),
        _ => Err(anyhow::anyhow!(
            "zero_knap_sa dispatcher: unknown track config \
             (num_items={}, budget_pct={}). \
             Expected num_items in {{1000, 5000}}.",
            n, budget_pct
        )),
    }
}
