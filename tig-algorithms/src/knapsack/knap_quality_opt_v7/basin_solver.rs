use anyhow::Result;
use tig_challenges::knapsack::*;
use super::bs_ils::run_one_instance;
use super::bs_params::Params;

pub fn solve(ch: &Challenge, save: &dyn Fn(&Solution) -> Result<()>) -> Result<()> {
    let params = Params::initialize(&None);
    let solution = run_one_instance(ch, &params);
    let _ = save(&solution);
    Ok(())
}
