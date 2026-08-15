use anyhow::Result;
use tig_challenges::satisfiability::*;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    super::weighted::solve(challenge, save_solution, &super::weighted::Params {
        weighted_restarts: 7,
        flips_multiplier: 3100,
        cb_exp: 25,
        cambium_interval_divisor: 15,
        smooth_every: 5,
        perturb_pct: 12,
        crossover_pct: 10,
        crossover_bias: 75,
        stagnation_factor: 12,
        fast_restarts: 0,
        fast_flips_multiplier: 2500,
    })
}