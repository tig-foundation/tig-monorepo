// TIG's UI uses the pattern `tig_challenges::energy_arbitrage` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

mod helpers;
mod ic_congested;
mod track_baseline;
mod track_capstone;
mod track_congested;
mod track_dense;
mod track_multiday;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub soc_levels: Option<usize>,
    pub action_grid: Option<usize>,
    pub asca_iters: Option<usize>,
    pub ternary_iters: Option<usize>,
    pub convergence_tol: Option<f64>,
    pub anticipate_lmp: Option<bool>,
    pub lmp_threshold: Option<f64>,
    pub lmp_premium_scale: Option<f64>,
    pub jump_premium: Option<f64>,
    pub prune_ratio: Option<f64>,
    pub deflator_iters: Option<usize>,
    pub flow_margin: Option<f64>,
    pub network_derating: Option<f64>,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Submission-safe entrypoint: do not reference sibling algorithms because
    // TIG dashboard compilation only receives this algorithm's uploaded files.
    ic_congested::solve_challenge(challenge, save_solution, hyperparameters)
}

pub fn help() {
    println!("ic_energy_v1");
}
