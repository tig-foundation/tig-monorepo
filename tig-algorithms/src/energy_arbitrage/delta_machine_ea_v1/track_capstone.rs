//! CAPSTONE scenario: 150 nodes, 300 lines, 100 batteries, H=192, line
//! limits ×0.40, high volatility.
//!
//! Tightest network + most batteries. Stochastic V_t for option value
//! in the high-noise regime. Derating 0.20 keeps the per-battery plan
//! conservative; the joint dispatch (ASCA + LP) coordinates the 100
//! batteries to push beyond the planning envelope where the network
//! has slack. Reduced lp_iters (25) because the largest network is
//! where sub-gradient oscillation costs the most; more LP iters here
//! produces worse outcomes via dual-price drift.

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
        network_derating: 0.20,
        anticipate_lmp: true,
        lmp_threshold: 0.65,
        lmp_premium_scale: 1.0,
        use_asca: true,
        asca_iters: 25,
        convergence_tol: 1e-3,
        flow_margin: 1e-4,
        deflator_iters: 50,
        lp_iters: 25,
        lp_step_size: 0.125,
        lp_momentum: 0.0,
    };
    vt_value_function_policy(challenge, cfg)
}
