//! Per-scenario solver for `energy_arbitrage`.
//!
//! Five scenario configurations differ in network density, line tightness,
//! horizon length, and price volatility. `mod.rs` dispatches by inferring
//! the scenario from `(num_steps, num_batteries)`. Each `track_*.rs`
//! configures `vt_value_function_policy` with the scenario-appropriate
//! discretisation, derating, and dispatch parameters.
//!
//! | Scenario  | nodes | lines | batteries | steps |
//! |-----------|-------|-------|-----------|-------|
//! | BASELINE  |    20 |    30 |        10 |    96 |
//! | CONGESTED |    40 |    60 |        20 |    96 |
//! | MULTIDAY  |    80 |   120 |        40 |   192 |
//! | DENSE     |   100 |   200 |        60 |   192 |
//! | CAPSTONE  |   150 |   300 |       100 |   192 |
//!
//! Quality scoring is the shifted geomean across the five scenarios with
//! `QUALITY_PRECISION = 1,000,000` and `QUALITY_CLAMP = 10,000,000`. Per-
//! scenario specialisation is essential — one weak track tanks the
//! aggregate score.

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

pub mod helpers;
pub mod track_baseline;
pub mod track_congested;
pub mod track_dense;
pub mod track_multiday;
pub mod track_capstone;

/// The five known scenarios. Inferred at solve time from challenge
/// dimensions because the `Challenge` struct does not expose the
/// originating `Track` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    Baseline,
    Congested,
    Multiday,
    Dense,
    Capstone,
}

impl Scenario {
    pub fn from_challenge(challenge: &Challenge) -> Scenario {
        match (challenge.num_steps, challenge.num_batteries) {
            (96, b) if b <= 15 => Scenario::Baseline,
            (96, _) => Scenario::Congested,
            (192, b) if b <= 50 => Scenario::Multiday,
            (192, b) if b <= 80 => Scenario::Dense,
            (192, _) => Scenario::Capstone,
            // Defensive: fall back to the most general handler if a new
            // scenario is added rather than panic.
            _ => Scenario::Capstone,
        }
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let scenario = Scenario::from_challenge(challenge);
    let solution = match scenario {
        Scenario::Baseline => track_baseline::solve(challenge, hyperparameters),
        Scenario::Congested => track_congested::solve(challenge, hyperparameters),
        Scenario::Multiday => track_multiday::solve(challenge, hyperparameters),
        Scenario::Dense => track_dense::solve(challenge, hyperparameters),
        Scenario::Capstone => track_capstone::solve(challenge, hyperparameters),
    }?;
    save_solution(&solution)?;
    Ok(())
}

pub fn help() {
    println!("Per-scenario energy arbitrage solver");
}
