use super::helpers::{solve_with_hp, TrackHp};
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::{Challenge, Solution};

fn defaults() -> TrackHp {
    TrackHp {
        soc_levels: 201,
        action_grid: 30,
        asca_iters: 15,
        ternary_iters: 15,
        convergence_tol: 1e-3,
        anticipate_lmp: false,
        lmp_threshold: 0.85,
        lmp_premium_scale: 0.45,
        jump_premium: 0.00,
        prune_ratio: 0.00,
        deflator_iters: 50,
        flow_margin: 1e-4,
        network_derating: 1.00,
    }
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut hp = defaults();
    hp.override_from_map(hyperparameters);
    solve_with_hp(challenge, save_solution, hp)
}
