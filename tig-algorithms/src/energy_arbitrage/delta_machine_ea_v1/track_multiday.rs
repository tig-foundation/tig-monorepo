//! MULTIDAY scenario: 80 nodes, 120 lines, 40 batteries, H=192, line
//! limits ×0.60, medium-high volatility.
//!
//! Long horizon (H=192) means the V_t backward induction integrates
//! multi-day SOC dynamics. Deterministic V_t with a small jump_premium
//! works better here than stochastic SDP — the long horizon already
//! averages out per-step price noise, and the cleaner deterministic
//! signal lets the joint dispatch (ASCA + LP) coordinate the 40
//! batteries more sharply.

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
        use_sdp: false,
        jump_premium: 0.02,
        network_derating: 0.15,
        anticipate_lmp: true,
        lmp_threshold: 0.65,
        lmp_premium_scale: 1.0,
        use_asca: true,
        asca_iters: 25,
        convergence_tol: 1e-3,
        flow_margin: 1e-4,
        deflator_iters: 25,
        lp_iters: 200,
        lp_step_size: 0.125,
        lp_momentum: 0.0,
    };
    vt_value_function_policy(challenge, cfg)
}
