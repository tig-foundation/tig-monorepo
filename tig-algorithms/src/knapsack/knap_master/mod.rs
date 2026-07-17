pub mod track1;
pub mod track2;
pub mod track3;
pub mod track4;
pub mod track5;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[derive(Clone, Debug, Default)]
pub struct Hparams {}

impl Hparams {
    pub fn for_track1() -> Self { Self::default() }
    pub fn for_track2() -> Self { Self::default() }
    pub fn for_track4() -> Self { Self::default() }
    pub fn for_track3() -> Self { Self::default() }
    pub fn for_track5() -> Self { Self::default() }
}

pub fn help() {
    println!(
        "Quadratic Knapsack solver — per-track ILS, DP, and population search\n\
         \n\
         Routing by (num_items, budget_pct):\n\
           n_items=1000, budget_pct<= 7  ->  track1\n\
           n_items=1000, budget_pct<=17  ->  track2\n\
           n_items=1000, budget_pct> 17  ->  track3\n\
           n_items=5000, budget_pct<=17  ->  track4\n\
           n_items=5000, budget_pct> 17  ->  track5\n\
         \n\
         Hyperparameters accepted by all tracks:\n\
           ils_rounds            Number of iterated-local-search perturbation rounds.\n\
           window_k               Candidate-window size for local-search exchanges.\n\
           core_half_dp           Half-width of the DP refinement core.\n\
         \n\
         Additional hyperparameters accepted by tracks 1, 2, and 3:\n\
           n_lambda_values        Resolution of the parametric breakpoint seed search.\n\
         \n\
         Additional hyperparameters accepted by track3 (1000 items, budget>17%):\n\
           n_random_starts        Number of randomized construction seeds.\n\
           n_crossover_gen        Number of population crossover generations.\n\
           sa_rounds              Simulated-annealing temperature rounds per selected member.\n\
           sa_iter                Simulated-annealing moves attempted per temperature round.\n\
           n_sa_members           Population members improved by simulated annealing.\n\
           ils_restart_interval   Stalled ILS rounds before restarting from the population.\n\
           perturb_base_frac      Base fraction of selected items removed by perturbation.\n\
           perturb_max_frac       Upper bound on perturbation removal intensity.\n\
           ils_vnd_level          Local-search mode selector.\n\
           bounded_2_2_k          Candidate limit for bounded 2-for-2 exchanges.\n\
           n_full_restarts        Number of independent population/ILS restarts.\n\
         \n\
         Example: {\"ils_rounds\":100,\"window_k\":300,\"core_half_dp\":60}"
    );
}

fn compute_budget_pct(challenge: &Challenge) -> u32 {
    let sum_w: u64 = challenge.weights.iter().map(|&w| w as u64).sum();
    if sum_w > 0 {
        ((challenge.max_weight as u64) * 100 / sum_w) as u32
    } else {
        10
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save:      &dyn Fn(&Solution) -> Result<()>,
    hp:        &Option<Map<String, Value>>,
) -> Result<()> {

    let n          = challenge.num_items;
    let budget_pct = compute_budget_pct(challenge);

    match (n, budget_pct) {
        (1000, b) if b <= 7  => track1::solve(challenge, save, hp),
        (1000, b) if b <= 17 => track2::solve(challenge, save, hp),
        (1000, _)            => track3::solve(challenge, save, hp),

        (5000, b) if b <= 17 => track4::solve(challenge, save, hp),
        (5000, _)            => track5::solve(challenge, save, hp),

        _ => Err(anyhow::anyhow!(
            "QKBP dispatcher: unknown track config \
             (num_items={}, budget_pct={}). \
             Expected num_items in {{1000, 5000}}.",
            n, budget_pct
        )),
    }

}