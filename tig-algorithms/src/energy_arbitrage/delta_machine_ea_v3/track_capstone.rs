//! CAPSTONE scenario: 150 nodes, 300 lines, 100 batteries, H=192, line
//! limits ×0.40, high volatility.
//!
//! Stochastic V_t value function + ASCA coordinate dispatch + LP flow
//! coordination, tuned for quality-per-fuel on the largest network:
//! soc_levels=64, no SDP, asca_iters=15, deflator_iters=20.

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

use super::helpers::{vt_value_function_policy, VtConfig};

pub fn solve(
    challenge: &Challenge,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<Solution> {
    let cfg = VtConfig {
        soc_levels: 64,
        action_grid: 30,
        action_scale: 1.0,
        shrink_factor: 0.5,
        use_sdp: false,
        jump_premium: 0.0,
        network_derating: 0.20,
        anticipate_lmp: true,
        lmp_threshold: 0.65,
        lmp_premium_scale: 1.0,
        use_asca: true,
        asca_iters: 15,
        convergence_tol: 1e-3,
        flow_margin: 1e-4,
        deflator_iters: 20,
        lp_iters: 25,
        lp_step_size: 0.125,
        lp_momentum: 0.0,
        use_joint_lp: false,
        use_history_seed: false,
        lookahead_horizon: 4,
        history_seed_drives: false,
    };
    vt_value_function_policy(challenge, cfg)
}
