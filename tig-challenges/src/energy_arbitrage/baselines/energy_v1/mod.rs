pub mod baseline;
pub mod capstone;
pub mod congested;
pub mod dense;
pub mod multiday;

use crate::energy_arbitrage::{Challenge, State};
use anyhow::Result;

/// Determine which track we're on from the challenge parameters.
fn detect_track(challenge: &Challenge) -> &'static str {
    match (challenge.num_batteries, challenge.num_steps) {
        (10, 96) => "baseline",
        (20, 96) => "congested",
        (40, 192) => "multiday",
        (60, 192) => "dense",
        (100, 192) => "capstone",
        _ => "baseline", // fallback
    }
}

/// Dispatch to the correct energy_v1 track module.
pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    match detect_track(challenge) {
        "baseline" => baseline::policy(challenge, state),
        "congested" => congested::policy(challenge, state),
        "multiday" => multiday::policy(challenge, state),
        "dense" => dense::policy(challenge, state),
        "capstone" => capstone::policy(challenge, state),
        _ => baseline::policy(challenge, state),
    }
}
