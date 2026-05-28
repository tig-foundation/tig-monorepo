//! BASELINE scenario: 20 nodes, 30 lines, 10 batteries, H=96, nominal
//! line limits, low volatility.
//!
//! Loose network — flow constraints rarely bind from exogenous load
//! alone, so derating stays at 1.0 (use full nameplate power in DP
//! planning). Stochastic V_t (K=5 Gauss-Hermite) captures the option
//! value of state-contingent decisions in the low-noise regime. LMP
//! anticipation runs at a low threshold (0.3) to catch the few lines
//! that do reach high utilisation; the small per-node price shift this
//! produces is exploitable on a loose network.

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
        jump_premium: 0.0, // ignored when use_sdp = true
        network_derating: 1.0,
        anticipate_lmp: true,
        lmp_threshold: 0.3,
        lmp_premium_scale: 0.45,
        use_asca: true,
        asca_iters: 30,
        convergence_tol: 1e-4,
        flow_margin: 1e-4,
        deflator_iters: 5,
        lp_iters: 200,
        lp_step_size: 0.125,
        lp_momentum: 0.0,
        use_joint_lp: true,
    };
    vt_value_function_policy(challenge, cfg)
}
