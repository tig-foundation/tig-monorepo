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

#[cfg(test)]
mod tests {
    use super::solve_challenge;
    use tig_challenges::{knapsack::*, *};

    // Write any personal tests you want to run against your algorithm in this module
    // All your tests must have #[ignore]
    // You can run ignored tests with `cargo test -p tig-algorithms -- --include-ignored`

    #[test]
    #[ignore]
    fn test_solve_challenge() {
        let difficulty = Difficulty {
            num_items: 50,
            better_than_baseline: 10,
        };
        let seed = 0;
        let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();
        assert!(solve_challenge(&challenge).is_ok());
    }
}
