//! DENSE scenario: 100 nodes, 200 lines, 60 batteries, H=192, line
//! limits ×0.50, high volatility.
//!
//! Densest network in the challenge set. Stochastic V_t captures the
//! option value of state-contingent decisions in the high-noise regime.
//! Derating 0.15 — the joint dispatch (ASCA + LP) handles the remaining
//! flow coordination across 60 batteries.

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

use super::helpers::{vt_value_function_policy, VtConfig};

pub fn solve(
    challenge: &Challenge,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<Solution> {
    let cfg = VtConfig {
        soc_levels: 201,
        action_grid: 30,
        action_scale: 1.0,
        shrink_factor: 0.5,
        use_sdp: true,
        jump_premium: 0.0,
        network_derating: 0.15,
        anticipate_lmp: true,
        lmp_threshold: 0.65,
        lmp_premium_scale: 1.0,
        use_asca: true,
        asca_iters: 25,
        convergence_tol: 1e-3,
        flow_margin: 1e-4,
        deflator_iters: 15,
        lp_iters: 200,
        lp_step_size: 0.125,
        lp_momentum: 0.0,
    };
    vt_value_function_policy(challenge, cfg)
}
