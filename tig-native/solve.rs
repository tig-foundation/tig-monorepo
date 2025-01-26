extern crate tig_algorithms;

use {
    super::{Challenge, Solution},
};

pub fn solve(challenge: Challenge) -> Option<Solution>
{
    return tig_algorithms::{challenge_type}::{algorithm_name}::solve_challenge(&challenge).unwrap_or(None);
}