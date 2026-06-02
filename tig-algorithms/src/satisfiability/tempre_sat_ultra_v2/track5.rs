use anyhow::Result;
use tig_challenges::satisfiability::*;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    super::weighted::solve(challenge, save_solution, &super::weighted::Params {
        weighted_restarts: 80,
        flips_multiplier: 60,
        cb_exp: 26,
        cambium_interval_divisor: 4,
        smooth_every: 3,
        perturb_pct: 12,
        crossover_pct: 20,
        crossover_bias: 70,
        stagnation_factor: 15,
        fast_restarts: 0,
        fast_flips_multiplier: 0,
    })
}