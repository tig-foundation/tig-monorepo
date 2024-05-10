//! <Optional header: remove if not needed>
//!
//! # Authors
//! <Optional: remove if not needed>
//!
//! # Description
//! <Optional: remove if not needed>
//!
//! # References
//! <Optional: remove if not needed>
//!
//! # License
//! <Important: read TIG's Terms & Conditions before replacing>
//! <Optional: remove if not needed>

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::knapsack::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(None) if your algorithm finds no solution or needs to exit early
    // return Ok(Solution { .. }) if your algorithm finds a solution
    Err(anyhow!("Not implemented"))
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
