//! CONGESTED scenario: 40 nodes, 60 lines, 20 batteries, H=96, line
//! limits ×0.80, medium volatility.
//!
//! Lines bind frequently from exogenous load. Deterministic V_t (no
//! SDP) with a small jump_premium on the sell side performs better here
//! than stochastic DP because the binding-line dynamics dominate the
//! per-step price noise. Derating loosened to 0.30 because the
//! interleaved ASCA + Lagrangian LP dispatch can safely push individual
//! batteries beyond the planning envelope when per-line slack allows.

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
        network_derating: 0.30,
        anticipate_lmp: true,
        lmp_threshold: 0.65,
        lmp_premium_scale: 1.0,
        use_asca: true,
        asca_iters: 25,
        convergence_tol: 1e-3,
        flow_margin: 1e-4,
        deflator_iters: 50,
        lp_iters: 200,
        lp_step_size: 0.125,
        lp_momentum: 0.0,
        use_joint_lp: true,
    };
    vt_value_function_policy(challenge, cfg)
}
