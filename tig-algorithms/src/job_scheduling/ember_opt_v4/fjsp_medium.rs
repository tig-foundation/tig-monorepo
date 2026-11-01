use anyhow::Result;
use tig_challenges::job_scheduling::*;

use super::types::{EffortConfig, Pre};

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _pre: &Pre,
    _effort: &EffortConfig,
) -> Result<()> {
    super::ember_medium::solve(challenge, save_solution)
}
