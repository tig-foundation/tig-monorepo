use anyhow::Result;
use tig_challenges::job_scheduling::*;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    super::our_search::solve_our(challenge, save_solution)
}
