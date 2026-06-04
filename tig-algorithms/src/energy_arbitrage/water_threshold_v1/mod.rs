// TIG's UI uses the pattern `tig_challenges::energy_arbitrage` to detect the challenge.
//
// water_threshold_v1 — receding-horizon water-value threshold policy for battery arbitrage.
//
// The objective is to buy energy when cheap and sell when expensive, per battery, subject to
// grid line-flow limits and battery physics. Individual transaction/degradation costs are tiny
// relative to the diurnal price swing, so the problem is essentially near-linear arbitrage where
// *timing* and *state-of-charge management* dominate. We therefore drive each battery with a
// price-threshold rule:
//
//   - Per node, derive charge/discharge price thresholds from quantiles of the (fully known)
//     day-ahead price horizon at that node.
//   - At each step, act on the OBSERVED real-time price (so RT spikes/crashes are exploited):
//       price <= charge_threshold   -> charge  as hard as feasible
//       price >= discharge_threshold-> discharge as hard as feasible
//       otherwise idle.
//   - SOC feedback (discrete water value): when the battery is full, shift both thresholds down
//     (discharge eagerly, charge reluctantly); when empty, shift them up. This self-regulates the
//     daily cycle without an explicit horizon optimization.
//   - Finally, project the joint action onto the network-feasible set (reuse the baseline's
//     line-softening + global-bisection fallback) so no line limit is ever violated. Zeros are
//     always feasible, so we can never emit an invalid action.
//
// This beats the provided greedy baseline, which (a) uses node 0's price for every battery,
// (b) only looks 3 hours ahead, and (c) ignores SOC and nodal prices.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

#[derive(Serialize, Deserialize, Default)]
pub struct Hyperparameters {
    /// Lower price quantile (per node) below which we charge. Default 0.25.
    pub charge_q: Option<f64>,
    /// Upper price quantile (per node) above which we discharge. Default 0.75.
    pub discharge_q: Option<f64>,
    /// SOC-feedback strength: threshold shift = soc_bias * (soc_frac - 0.5) * spread. Default 0.0.
    pub soc_bias: Option<f64>,
}

struct Params {
    charge_q: f64,
    discharge_q: f64,
    soc_bias: f64,
}

pub fn help() {
    println!(
        "water_threshold_v1: per-node day-ahead price quantiles set charge/discharge thresholds; \
         act on observed real-time price with SOC-feedback (water-value) shifting; project actions \
         to network-feasible. Hyperparameters: charge_q (default 0.25), discharge_q (default 0.75), \
         soc_bias (default 0.0)."
    );
}

/// Internal safety margin on flow limits so floating-point slop never trips the strict verifier.
const FLOW_MARGIN: f64 = 0.999;
const EPS: f64 = 1e-12;

#[inline]
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn line_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    challenge.network.compute_flows(&injections)
}

/// Returns (line, signed_flow, overshoot_amount) for the most-violated line, against the margined limit.
fn most_violated(challenge: &Challenge, flows: &[f64]) -> Option<(usize, f64, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l] * FLOW_MARGIN;
        let amount = flow.abs() - limit;
        if amount > 0.0 {
            match best {
                Some((_, _, b)) if amount <= b => {}
                _ => best = Some((l, flow, amount)),
            }
        }
    }
    best
}

/// Scale down (toward zero) only the batteries that push the violated line in its overflow
/// direction, just enough to remove the overshoot. Returns false if it can't help.
fn soften(challenge: &Challenge, line: usize, flow: f64, amount: f64, action: &mut [f64]) -> bool {
    let dir = flow.signum();
    if dir.abs() <= EPS {
        return false;
    }
    let mut idxs = Vec::new();
    let mut strength = 0.0;
    for (i, bat) in challenge.batteries.iter().enumerate() {
        let contribution = challenge.network.ptdf[line][bat.node] * action[i];
        let signed = dir * contribution;
        if signed > EPS {
            strength += signed;
            idxs.push(i);
        }
    }
    if idxs.is_empty() || strength <= EPS {
        return false;
    }
    let keep = (1.0 - amount / strength).clamp(0.0, 1.0);
    if (1.0 - keep).abs() <= EPS {
        return false;
    }
    for i in idxs {
        action[i] *= keep;
    }
    true
}

fn is_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = line_flows(challenge, state, action);
    most_violated(challenge, &flows).is_none()
}

/// Project an action onto the network-feasible set. Never returns an infeasible action:
/// zeros are always feasible (exogenous injections are pre-scaled to leave line headroom).
fn enforce_flow_feasibility(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Vec<f64> {
    for _ in 0..64 {
        let flows = line_flows(challenge, state, &action);
        match most_violated(challenge, &flows) {
            None => return action,
            Some((l, flow, amount)) => {
                if !soften(challenge, l, flow, amount, &mut action) {
                    break;
                }
            }
        }
    }
    if is_feasible(challenge, state, &action) {
        return action;
    }
    // Global bisection on a scale factor in [0, 1].
    let base = action;
    let zero = vec![0.0; base.len()];
    if !is_feasible(challenge, state, &zero) {
        // Should not happen, but zeros are the safe floor.
        return zero;
    }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..32 {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| mid * u).collect();
        if is_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    base.into_iter().map(|u| low * u).collect()
}

fn resolve_params(hyperparameters: &Option<Map<String, Value>>) -> Params {
    let hp = match hyperparameters {
        Some(map) => serde_json::from_value::<Hyperparameters>(Value::Object(map.clone()))
            .unwrap_or_default(),
        None => Hyperparameters::default(),
    };
    Params {
        charge_q: hp.charge_q.unwrap_or(0.25),
        discharge_q: hp.discharge_q.unwrap_or(0.75),
        soc_bias: hp.soc_bias.unwrap_or(0.0),
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let params = resolve_params(hyperparameters);
    let n_nodes = challenge.network.num_nodes;
    let horizon = challenge.num_steps;
    let da = &challenge.market.day_ahead_prices; // [H][n]

    // Per-node charge/discharge price thresholds from the day-ahead horizon.
    let mut charge_th = vec![0.0f64; n_nodes];
    let mut discharge_th = vec![0.0f64; n_nodes];
    let mut col = Vec::with_capacity(horizon);
    for node in 0..n_nodes {
        col.clear();
        for t in 0..horizon {
            col.push(da[t][node]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        charge_th[node] = quantile(&col, params.charge_q);
        discharge_th[node] = quantile(&col, params.discharge_q);
    }

    let policy = |ch: &Challenge, st: &State| -> Result<Vec<f64>> {
        let m = ch.num_batteries;
        let mut action = vec![0.0f64; m];
        for b in 0..m {
            let bat = &ch.batteries[b];
            let node = bat.node;
            let price = st.rt_prices[node];
            let (u_min, u_max) = st.action_bounds[b]; // u_min<=0 (charge), u_max>=0 (discharge)

            let span = bat.soc_max_mwh - bat.soc_min_mwh;
            let soc_frac = if span > 0.0 {
                ((st.socs[b] - bat.soc_min_mwh) / span).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let spread = (discharge_th[node] - charge_th[node]).max(0.0);
            // Full -> shift thresholds down (discharge eager); empty -> shift up (charge eager).
            let shift = params.soc_bias * (soc_frac - 0.5) * spread;
            let ct = charge_th[node] - shift;
            let dt = discharge_th[node] - shift;

            if price <= ct {
                action[b] = u_min; // charge as hard as feasible (bound already respects SOC/power)
            } else if price >= dt {
                action[b] = u_max; // discharge as hard as feasible
            } else {
                action[b] = 0.0;
            }
        }
        Ok(enforce_flow_feasibility(ch, st, action))
    };

    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}
