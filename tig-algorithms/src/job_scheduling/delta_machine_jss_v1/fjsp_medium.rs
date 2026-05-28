use anyhow::Result;
use tig_challenges::job_scheduling::*;

use super::types::*;
use super::our_search;

#[allow(unused_variables)]
pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _pre: &Pre,
    _greedy_sol: Solution,
    _greedy_mk: u32,
    _effort: &EffortConfig,
) -> Result<()> {
    our_search::solve_our(challenge, save_solution)
}
