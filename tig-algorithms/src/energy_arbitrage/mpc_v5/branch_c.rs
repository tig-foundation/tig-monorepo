use anyhow::Result;
use tig_challenges::energy_arbitrage::{Challenge, NextRTPrices, State};

use super::config::Config;

// H=1 MPC with DA price override for CAPSTONE (size > SHALLOW_MPC_THRESHOLD).
//
// RT prices at h=0 cause Pareto-spike misdirection; substituting DA[t] eliminates it.
// take_step advances SOC faithfully with DA[t+1] as next-step RT.
// Terminal SOC value prevents premature depletion.
// Coordinate descent in capacity order; binary-search flow enforcement.
//
// Candidate selection:
//   n_cand_c == 3 → [lo, 0.0, hi]  exact mpc_v3 behavior (ensures effort=2 identity)
//   n_cand_c != 3 → make_candidates(lo, hi, n_cand_c)

pub fn policy(challenge: &Challenge, state: &State, config: &Config) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_bat = challenge.num_batteries;
    let da = &challenge.market.day_ahead_prices;

    let mut bat_order: Vec<usize> = (0..n_bat).collect();
    bat_order.sort_by(|&a, &b| {
        challenge.batteries[b]
            .capacity_mwh
            .partial_cmp(&challenge.batteries[a].capacity_mwh)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // DA-overridden state for compute_profit scoring only; take_step uses original state
    let mut da_state = state.clone();
    da_state.rt_prices = da[t].clone();

    let next_da = da[(t + 1).min(da.len() - 1)].clone();

    let mut chosen = vec![0.0f64; n_bat];
    for &b in &bat_order {
        let (lo, hi) = state.action_bounds[b];

        let candidates: Vec<f64> = if config.n_cand_c == 3 {
            vec![lo, 0.0f64, hi]
        } else {
            super::make_candidates(lo, hi, config.n_cand_c)
        };

        let mut best_val = f64::NEG_INFINITY;
        let mut best_u = 0.0f64;

        for u in candidates {
            let u = u.clamp(lo, hi);
            let mut action = vec![0.0f64; n_bat];
            action[b] = u;

            let da_profit = challenge.compute_profit(&da_state, &action);

            match challenge.take_step(state, &action, NextRTPrices::Override(next_da.clone())) {
                Ok(next_sim) => {
                    let term = if !next_sim.socs.is_empty() {
                        super::terminal_soc_value(challenge, &next_sim, b, t + 1)
                    } else {
                        0.0
                    };
                    let val = da_profit + term;
                    if val > best_val {
                        best_val = val;
                        best_u = u;
                    }
                }
                Err(_) => {}
            }
        }

        chosen[b] = best_u;
    }

    Ok(super::apply_flow_scale(challenge, state, chosen))
}
