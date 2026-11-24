use anyhow::{anyhow, Result};
use tig_challenges::energy_arbitrage::{Challenge, State};

const MAX_FLOW_ADJUST_ITERS: usize = 16;
const EPS: f64 = 1e-12;
const CHARGE_QUANTILE: f64 = 0.30;
const DISCHARGE_QUANTILE: f64 = 0.70;

pub struct Tables {
    curves: Vec<Vec<f64>>,
    cols: Vec<Vec<(usize, f64, f64)>>,
    limits: Vec<f64>,
}

pub fn tables(challenge: &Challenge) -> Tables {
    let net = &challenge.network;
    let eps_flow = tig_challenges::energy_arbitrage::constants::EPS_FLOW;
    let cols = challenge
        .batteries
        .iter()
        .map(|b| {
            (0..net.num_lines)
                .filter(|&l| net.ptdf[l][b.node].abs() > EPS)
                .map(|l| (l, net.ptdf[l][b.node], 1.0 / net.ptdf[l][b.node]))
                .collect()
        })
        .collect();
    let limits = net.flow_limits.iter().map(|&f| f * (1.0 + eps_flow)).collect();
    Tables { curves: continuation_curves(challenge), cols, limits }
}

pub fn day_ahead_profit(challenge: &Challenge, tables: &Tables) -> f64 {
    let mut socs: Vec<f64> = challenge.batteries.iter().map(|b| b.soc_initial_mwh).collect();
    let mut profit = 0.0;
    for t in 0..challenge.num_steps {
        let state = State {
            time_step: t,
            socs: socs.clone(),
            rt_prices: challenge.market.day_ahead_prices[t].clone(),
            exogenous_injections: challenge.exogenous_injections[t].clone(),
            action_bounds: challenge
                .batteries
                .iter()
                .zip(socs.iter())
                .map(|(b, &s)| action_bounds(b, s))
                .collect(),
            total_profit: 0.0,
        };
        let Ok(a) = policy(challenge, &state, tables, false) else {
            return f64::NAN;
        };
        profit += challenge.compute_profit(&state, &a);
        for (i, b) in challenge.batteries.iter().enumerate() {
            socs[i] = b.apply_action_to_soc(a[i], socs[i]);
        }
    }
    profit
}

fn action_bounds(b: &tig_challenges::energy_arbitrage::Battery, soc: f64) -> (f64, f64) {
    let dt = tig_challenges::energy_arbitrage::constants::DELTA_T;
    let headroom = (b.soc_max_mwh - soc).max(0.0);
    let available = (soc - b.soc_min_mwh).max(0.0);
    let max_charge = if b.efficiency_charge > 0.0 { headroom / (b.efficiency_charge * dt) } else { 0.0 };
    let max_discharge = if b.efficiency_discharge > 0.0 { available * b.efficiency_discharge / dt } else { 0.0 };
    (-max_charge.min(b.power_charge_mw).max(0.0), max_discharge.min(b.power_discharge_mw).max(0.0))
}

fn continuation_curves(challenge: &Challenge) -> Vec<Vec<f64>> {
    let node_thresholds = compute_node_thresholds_inner(challenge);
    let (suffix_max_env, suffix_min_env) = compute_suffix_price_bounds_inner(challenge);

    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let mut continuation_curves = vec![vec![0.0f64; 2 * num_steps]; num_nodes];

    for node in 0..num_nodes {
        let (charge_q, discharge_q) = node_thresholds[node];
        for t in 0..num_steps {
            let sell = suffix_max_env[node][t + 1].max(discharge_q);
            let buy = suffix_min_env[node][t + 1].min(charge_q);
            continuation_curves[node][2 * t] = sell;
            continuation_curves[node][2 * t + 1] = buy;
        }
    }
    continuation_curves
}

struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    (0..challenge.network.num_lines)
        .map(|l| {
            (0..challenge.network.num_nodes)
                .map(|k| challenge.network.ptdf[l][k] * injections[k])
                .sum::<f64>()
        })
        .collect()
}

fn most_violated_line(challenge: &Challenge, action: &[f64], flows: &[f64]) -> Option<Violation> {
    let mut best: Option<(Violation, f64, f64)> = None;

    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let amount = flow.abs() - limit;
        if amount <= tig_challenges::energy_arbitrage::constants::EPS_FLOW * limit {
            continue;
        }

        let signed_direction = flow.signum();
        if signed_direction.abs() <= EPS {
            continue;
        }

        let mut controllable_worsening = 0.0;
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let ptdf_val = challenge.network.ptdf[l][battery.node];
            let signed_contribution = signed_direction * (ptdf_val * action[i]);
            if signed_contribution > EPS {
                controllable_worsening += signed_contribution;
            }
        }

        let difficulty = amount / controllable_worsening.max(EPS);
        let candidate_v = Violation { line: l, flow, amount };

        match best {
            Some((_, best_difficulty, best_amount))
                if difficulty < best_difficulty
                    || (difficulty == best_difficulty && amount <= best_amount) => {}
            _ => best = Some((candidate_v, difficulty, amount)),
        }
    }

    best.map(|(v, _, _)| v)
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, action, &flows).is_none()
}

fn soften_most_violated_line(challenge: &Challenge, state: &State, violation: &Violation, action: &mut [f64]) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    let mut candidates: Vec<(usize, f64, f64)> = Vec::new();
    let mut total_worsening = 0.0;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let ptdf_val = challenge.network.ptdf[line][battery.node];
        let signed_contribution = signed_direction * (ptdf_val * action[i]);
        if signed_contribution > EPS {
            total_worsening += signed_contribution;

            let abs_ptdf = ptdf_val.abs().max(EPS);
            let price = state.rt_prices[battery.node];
            let marginal_loss = if action[i] >= 0.0 {
                price * battery.efficiency_discharge
            } else {
                -price / battery.efficiency_charge.max(EPS)
            };
            let loss_per_flow = marginal_loss / abs_ptdf;

            candidates.push((i, signed_contribution, loss_per_flow));
        }
    }

    if candidates.is_empty() || total_worsening <= EPS {
        return false;
    }

    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining_violation = violation.amount;
    let mut changed = false;

    for (idx, flow_contrib, _) in &candidates {
        if remaining_violation <= EPS {
            break;
        }
        let i = *idx;
        if *flow_contrib <= remaining_violation {
            action[i] = 0.0;
            remaining_violation -= flow_contrib;
            changed = true;
        } else {
            let fraction_to_remove = remaining_violation / flow_contrib;
            action[i] *= 1.0 - fraction_to_remove;
            changed = true;
            break;
        }
    }

    changed
}

fn enforce_flow_feasibility(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Result<Vec<f64>> {
    let min_valid: Vec<f64> = state.action_bounds.iter()
        .map(|&(lo, hi)| 0.0f64.clamp(lo, hi))
        .collect();

    let mut flows = compute_flows(challenge, state, &action);

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(violation) = most_violated_line(challenge, &action, &flows) else {
            return Ok(action);
        };
        let old_action = action.clone();
        if !soften_most_violated_line(challenge, state, &violation, &mut action) {
            break;
        }
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let delta = action[i] - old_action[i];
            if delta.abs() > EPS {
                let node = battery.node;
                for l in 0..challenge.network.num_lines {
                    flows[l] += challenge.network.ptdf[l][node] * delta;
                }
            }
            let (lo, hi) = state.action_bounds[i];
            action[i] = action[i].clamp(lo, hi);
        }
    }

    if most_violated_line(challenge, &action, &flows).is_none() {
        return Ok(action);
    }

    if !is_flow_feasible(challenge, state, &min_valid) {
        return Err(anyhow!("Grid infeasible even with minimum battery actions"));
    }

    let mut projected = action;
    let mut proj_flows = compute_flows(challenge, state, &projected);

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(violation) = most_violated_line(challenge, &projected, &proj_flows) else {
            return Ok(projected);
        };

        let line = violation.line;
        let signed_direction = violation.flow.signum();
        if signed_direction.abs() <= EPS { break; }

        let mut total_worsening = 0.0;
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let ptdf_val = challenge.network.ptdf[line][battery.node];
            let signed_contribution = signed_direction * (ptdf_val * projected[i]);
            if signed_contribution > EPS {
                total_worsening += signed_contribution;
            }
        }
        if total_worsening <= EPS { break; }

        let frac = (violation.amount / total_worsening).clamp(0.0, 1.0);

        for (i, battery) in challenge.batteries.iter().enumerate() {
            let ptdf_val = challenge.network.ptdf[line][battery.node];
            let signed_contribution = signed_direction * (ptdf_val * projected[i]);
            if signed_contribution > EPS {
                let old = projected[i];
                projected[i] *= 1.0 - frac;
                let (lo, hi) = state.action_bounds[i];
                projected[i] = projected[i].clamp(lo, hi);
                let delta = projected[i] - old;
                if delta.abs() > EPS {
                    for l in 0..challenge.network.num_lines {
                        proj_flows[l] += challenge.network.ptdf[l][battery.node] * delta;
                    }
                }
            }
        }
    }

    if most_violated_line(challenge, &projected, &proj_flows).is_none() {
        return Ok(projected);
    }

    Ok(min_valid)
}

fn compute_node_thresholds_inner(challenge: &Challenge) -> Vec<(f64, f64)> {
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let mut thresholds = vec![(0.0f64, 0.0f64); num_nodes];
    for node in 0..num_nodes {
        let mut prices: Vec<f64> = (0..num_steps)
            .map(|t| challenge.market.day_ahead_prices[t][node])
            .collect();
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = prices.len();
        let ci = ((n as f64 * CHARGE_QUANTILE) as usize).min(n.saturating_sub(1));
        let di = ((n as f64 * DISCHARGE_QUANTILE) as usize).min(n.saturating_sub(1));
        thresholds[node] = (prices[ci], prices[di]);
    }
    thresholds
}

fn compute_suffix_price_bounds_inner(challenge: &Challenge) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let mut suffix_max = vec![vec![f64::NEG_INFINITY; num_steps + 1]; num_nodes];
    let mut suffix_min = vec![vec![f64::INFINITY; num_steps + 1]; num_nodes];
    for node in 0..num_nodes {
        for t in (0..num_steps).rev() {
            let p = challenge.market.day_ahead_prices[t][node];
            suffix_max[node][t] = p.max(suffix_max[node][t + 1]);
            suffix_min[node][t] = p.min(suffix_min[node][t + 1]);
        }
    }
    (suffix_max, suffix_min)
}

fn compute_proactive_bounds(state: &State, tables: &Tables, baseline_flows: &[f64]) -> Vec<(f64, f64)> {
    let mut bounds: Vec<(f64, f64)> = state.action_bounds.clone();

    for (i, col) in tables.cols.iter().enumerate() {
        let (mut lo, mut hi) = bounds[i];

        for &(l, _, inv_p) in col.iter() {
            let limit = tables.limits[l];
            let f0 = baseline_flows[l];

            let mut lo_u = (-limit - f0) * inv_p;
            let mut hi_u = (limit - f0) * inv_p;
            if lo_u > hi_u {
                let tmp = lo_u;
                lo_u = hi_u;
                hi_u = tmp;
            }

            if lo_u > lo {
                lo = lo_u;
            }
            if hi_u < hi {
                hi = hi_u;
            }

            if hi < lo {
                let mid = 0.0f64.clamp(lo.min(hi), lo.max(hi));
                lo = mid;
                hi = mid;
                break;
            }
        }

        bounds[i] = (lo, hi);
    }

    bounds
}

pub fn policy(challenge: &Challenge, state: &State, tables: &Tables, with_flows: bool) -> Result<Vec<f64>> {
    let continuation_curves = &tables.curves;
    let t = state.time_step;
    let da_prices = &challenge.market.day_ahead_prices;

    if t >= da_prices.len() {
        return Err(anyhow!("Missing DA prices for time_step {}", t));
    }

    let baseline_action = state
        .action_bounds
        .iter()
        .map(|&(lo, hi)| 0.0f64.clamp(lo, hi))
        .collect::<Vec<f64>>();
    let baseline_flows = compute_flows(challenge, state, &baseline_action);
    let proactive_bounds = if with_flows {
        compute_proactive_bounds(state, tables, &baseline_flows)
    } else {
        state.action_bounds.clone()
    };

    let steps_remaining = challenge.num_steps.saturating_sub(t);
    let horizon_fraction = steps_remaining as f64 / challenge.num_steps as f64;

    let mut action = baseline_action;
    let mut flows = baseline_flows;

    #[derive(Clone, Copy)]
    struct Candidate {
        i: usize,
        target: f64,
        priority: f64,
    }

    let mut cands: Vec<Candidate> = Vec::with_capacity(challenge.batteries.len());

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (min_bound, max_bound) = proactive_bounds[i];

        let da_price = da_prices[t][node];

        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = (state.socs[i] - battery.soc_min_mwh) / soc_range;

        let (future_sell, future_buy) = if node < continuation_curves.len() {
            let curve = &continuation_curves[node];
            let j = 2 * t;
            if j + 1 < curve.len() {
                (curve[j], curve[j + 1])
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            }
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };

        let soc_surplus_mwh = (state.socs[i] - battery.soc_min_mwh).max(0.0);
        let steps_left = steps_remaining.max(1) as f64;
        let required_discharge_mw = if battery.efficiency_discharge > EPS {
            soc_surplus_mwh
                / (steps_left
                    * tig_challenges::energy_arbitrage::constants::DELTA_T
                    * battery.efficiency_discharge)
        } else {
            0.0
        };
        let max_discharge_mw = battery.power_discharge_mw.max(EPS);
        let liquidation_urgency = (required_discharge_mw / max_discharge_mw).clamp(0.0, 1.0);

        let near_end = horizon_fraction < 0.30;

        let eff_c = battery.efficiency_charge.max(EPS);
        let eff_d = battery.efficiency_discharge;

        let discharge_edge_base = da_price * eff_d - future_buy / eff_c;
        let charge_edge_base = future_sell * battery.efficiency_charge * eff_d - da_price;

        let discharge_terminal = liquidation_urgency * (future_buy.abs() / eff_c);
        let charge_terminal = liquidation_urgency * (future_sell.abs() * battery.efficiency_charge * eff_d);

        let discharge_edge = discharge_edge_base + discharge_terminal;
        let charge_edge = charge_edge_base - charge_terminal;

        let discharge_frac = (soc_level + liquidation_urgency * (1.0 - soc_level)).clamp(0.0, 1.0);
        let charge_frac = ((1.0 - soc_level) * (1.0 - liquidation_urgency)).clamp(0.0, 1.0);

        let u: f64 = if discharge_edge > EPS && max_bound > EPS {
            max_bound * discharge_frac
        } else if charge_edge > EPS && min_bound < -EPS && !near_end {
            min_bound * charge_frac
        } else {
            0.0f64
        }
        .clamp(min_bound, max_bound);

        let alpha = if u > EPS {
            discharge_edge
        } else if u < -EPS {
            charge_edge
        } else {
            0.0
        };

        let priority = alpha.abs() * (1.0 + liquidation_urgency);

        if priority > EPS && u.abs() > EPS {
            cands.push(Candidate {
                i,
                target: u,
                priority,
            });
        }
    }

    cands.sort_by(|a, b| {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for cand in cands {
        let i = cand.i;
        let (lo, hi) = proactive_bounds[i];

        let old_u = action[i];
        let desired = cand.target.clamp(lo, hi);
        let delta = desired - old_u;
        if delta.abs() <= EPS {
            continue;
        }

        let node = challenge.batteries[i].node;

        let mut s_low = 0.0f64;
        let mut s_high = 1.0f64;

        let inv_delta = 1.0 / delta;
        for &(l, p, inv_p) in tables.cols[i].iter() {
            let a = p * delta;
            if a.abs() <= EPS {
                continue;
            }

            let limit = tables.limits[l];
            let f0 = flows[l];

            let inv_a = inv_p * inv_delta;
            let mut lo_s = (-limit - f0) * inv_a;
            let mut hi_s = (limit - f0) * inv_a;
            if lo_s > hi_s {
                let tmp = lo_s;
                lo_s = hi_s;
                hi_s = tmp;
            }

            if lo_s > s_low {
                s_low = lo_s;
            }
            if hi_s < s_high {
                s_high = hi_s;
            }

            if s_high < s_low {
                break;
            }
        }

        let mut s = s_high;
        if s > 1.0 {
            s = 1.0;
        }
        if s <= 0.0 {
            continue;
        }

        let new_u = (old_u + delta * s).clamp(lo, hi);
        let applied = new_u - old_u;
        if applied.abs() <= EPS {
            continue;
        }

        action[i] = new_u;

        for &(l, p, _) in tables.cols[i].iter() {
            flows[l] += p * applied;
        }
    }

    if !with_flows {
        return Ok(action);
    }
    enforce_flow_feasibility(challenge, state, action)
}