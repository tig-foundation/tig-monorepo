use anyhow::Result;
use tig_challenges::job_scheduling::*;

#[path = "ember_hfs_construct.rs"]
mod hfs_construct;
#[path = "ember_hybrid_engine.rs"]
mod engine;

pub fn solve(challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> Result<()>) -> Result<()> {
    engine::solve_challenge(challenge, save_solution)
}
