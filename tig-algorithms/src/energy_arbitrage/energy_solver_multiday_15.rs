// energy_solver_multiday_15: STRUCTURAL — forward-simulation MPC. At each step, evaluate 5 structurally distinct candidate actions (zero, threshold, 1.5x, 0.5x, inverted) by simulating H steps ahead using DA as RT proxy + threshold tail policy. Pick candidate with highest projected total profit, then enforce_feasible. multiday_12's threshold policy serves as the tail/inner policy.
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub mpc_horizon: i64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self { mpc_horizon: 24 }
    }
}

pub fn help() {
    println!("HYPERPARAMETERS:");
    println!("  mpc_horizon: int range=[4, 48] default=24  # forward-simulation horizon (15-min steps)");
    println!();
    println!("Forward-simulation MPC: at each step, score 5 candidate actions (zero, threshold, 1.5x, 0.5x,");
    println!("inverted) by simulating mpc_horizon steps ahead with DA-as-RT proxy and the multiday_12");
    println!("threshold policy as the tail. Pick highest-projected-profit candidate, then enforce_feasible.");
    println!("All multiday_12 winners (ct=8, dt=1, ew=66, ls=66, sf=1.273, p=2.545, trend=0.6364) preserved.");
}

fn parse_hp(h: &Option<Map<String, Value>>) -> Hyperparameters {
    let mut out = Hyperparameters::default();
    if let Some(m) = h {
        if let Some(v) = m.get("mpc_horizon").and_then(|v| v.as_i64()) {
            out.mpc_horizon = v;
        }
    }
    out
}

const LOOKAHEAD_STEPS: usize = 66;
const END_WINDOW: usize = 66;
const END_FACTOR: f64 = 0.0;
const SLOPE_FACTOR: f64 = 1.273;
const CHARGE_THRESHOLD: f64 = 8.0;
const DISCHARGE_THRESHOLD: f64 = 1.0;
const TREND_FACTOR: f64 = 0.6364;
const PROFIT_WEIGHT_POWER: f64 = 2.545;

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;
const FLOW_EPS: f64 = 1e-6;
const PROFIT_EPS: f64 = 1e-3;

// From challenge::constants (Battery::compute_action_bounds is pub(crate); inline)
const DELTA_T: f64 = 0.25;

fn battery_action_bounds(battery: &Battery, soc: f64) -> (f64, f64) {
    let headroom = (battery.soc_max_mwh - soc).max(0.0);
    let available = (soc - battery.soc_min_mwh).max(0.0);
    let max_charge_from_soc = if battery.efficiency_charge > 0.0 {
        headroom / (battery.efficiency_charge * DELTA_T)
    } else {
        0.0
    };
    let max_discharge_from_soc = if battery.efficiency_discharge > 0.0 {
        available * battery.efficiency_discharge / DELTA_T
    } else {
        0.0
    };
    let max_charge = max_charge_from_soc.min(battery.power_charge_mw).max(0.0);
    let max_discharge = max_discharge_from_soc.min(battery.power_discharge_mw).max(0.0);
    (-max_charge, max_discharge)
}

#[derive(Clone, Copy)]
struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    challenge.network.compute_flows(&injections)
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > FLOW_EPS * limit {
            let cand = Violation { line: l, flow, amount: violation };
            match best {
                Some(cur) if cand.amount <= cur.amount => {}
                _ => best = Some(cand),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, &flows).is_none()
}

fn soften_line_weighted(
    challenge: &Challenge,
    v: Violation,
    action: &mut [f64],
    profit: &[f64],
    p: f64,
) -> bool {
    let line = v.line;
    let dir = v.flow.signum();
    if dir.abs() <= EPS {
        return false;
    }

    let mut worsening: Vec<(usize, f64)> = Vec::new();
    let mut total_signed = 0.0;
    for (i, b) in challenge.batteries.iter().enumerate() {
        let contrib = challenge.network.ptdf[line][b.node] * action[i];
        let signed = dir * contrib;
        if signed > EPS {
            total_signed += signed;
            worsening.push((i, signed));
        }
    }
    if worsening.is_empty() || total_signed <= EPS {
        return false;
    }

    if p == 0.0 {
        let keep = (1.0 - v.amount / total_signed).clamp(0.0, 1.0);
        if (1.0 - keep).abs() <= EPS {
            return false;
        }
        for (i, _) in &worsening {
            action[*i] *= keep;
        }
        return true;
    }

    let mut weights: Vec<f64> = worsening
        .iter()
        .map(|(i, signed)| {
            let pi = profit[*i].max(0.0).powf(p) + PROFIT_EPS;
            signed / pi
        })
        .collect();

    let mut residual = v.amount;
    let mut sum_w: f64 = weights.iter().sum();
    let mut zeroed = vec![false; worsening.len()];

    loop {
        if sum_w <= EPS || residual <= EPS {
            break;
        }
        let mut any_zeroed = false;
        for k in 0..worsening.len() {
            if zeroed[k] {
                continue;
            }
            let (_, signed) = worsening[k];
            let reduce = (residual / sum_w) * weights[k];
            if reduce >= signed {
                zeroed[k] = true;
                residual -= signed;
                sum_w -= weights[k];
                weights[k] = 0.0;
                any_zeroed = true;
            }
        }
        if !any_zeroed {
            break;
        }
    }

    let mut changed = false;
    for k in 0..worsening.len() {
        let (i, signed) = worsening[k];
        if zeroed[k] {
            if action[i].abs() > EPS {
                action[i] = 0.0;
                changed = true;
            }
        } else if sum_w > EPS && residual > EPS {
            let reduce = (residual / sum_w) * weights[k];
            let keep = (1.0 - reduce / signed).clamp(0.0, 1.0);
            if (1.0 - keep).abs() > EPS {
                action[i] *= keep;
                changed = true;
            }
        }
    }
    changed
}

fn enforce_feasible(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
    profit: &[f64],
    p: f64,
) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(v) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_line_weighted(challenge, v, &mut action, profit, p) {
            break;
        }
    }
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }
    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!("Grid infeasible even with zero battery actions"));
    }
    let base = action;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| mid * u).collect();
        if is_flow_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(base.into_iter().map(|u| low * u).collect())
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp = parse_hp(hyperparameters);
    let policy_fn = move |ch: &Challenge, st: &State| -> Result<Vec<f64>> {
        policy_with_hp(ch, st, &hp)
    };
    let solution = challenge.grid_optimize(&policy_fn)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    policy_with_hp(challenge, state, &Hyperparameters::default())
}

fn median_of(buf: &mut Vec<f64>) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = buf.len();
    if n % 2 == 1 {
        buf[n / 2]
    } else {
        0.5 * (buf[n / 2 - 1] + buf[n / 2])
    }
}

/// Threshold-policy action computation (multiday_12 logic) at a given (sim) time and price.
/// Returns raw action (no flow softening), to be enforce_feasible'd at the caller's discretion.
fn threshold_action(
    challenge: &Challenge,
    time_step: usize,
    rt_prices: &[f64],
    action_bounds: &[(f64, f64)],
) -> Vec<f64> {
    let end = (time_step + 1 + LOOKAHEAD_STEPS).min(challenge.num_steps);
    let da = &challenge.market.day_ahead_prices;

    let remaining = challenge.num_steps.saturating_sub(time_step);
    let in_end_window = END_WINDOW > 0 && remaining <= END_WINDOW;
    let (base_discharge_th, base_charge_th, suppress_charge) = if in_end_window {
        let progress = (remaining as f64 / END_WINDOW as f64).clamp(0.0, 1.0);
        let scale = END_FACTOR + (1.0 - END_FACTOR) * progress;
        (DISCHARGE_THRESHOLD * scale, CHARGE_THRESHOLD, true)
    } else {
        (DISCHARGE_THRESHOLD, CHARGE_THRESHOLD, false)
    };
    let slope = SLOPE_FACTOR.max(0.05);

    let mut action = vec![0.0; challenge.num_batteries];
    let mut buf: Vec<f64> = Vec::with_capacity(LOOKAHEAD_STEPS);
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        buf.clear();
        for tt in (time_step + 1)..end {
            buf.push(da[tt][node]);
        }
        if buf.is_empty() {
            let (_, u_max) = action_bounds[i];
            action[i] = u_max;
            continue;
        }

        let trend_signal = if buf.len() >= 2 {
            buf[buf.len() - 1] - buf[0]
        } else {
            0.0
        };

        let mut buf_for_median = buf.clone();
        let reference = median_of(&mut buf_for_median);
        let current = rt_prices[node];
        let diff = current - reference;

        let discharge_th_eff = (base_discharge_th + TREND_FACTOR * trend_signal).max(0.5);
        let charge_th_eff = (base_charge_th - TREND_FACTOR * trend_signal).max(0.5);

        let (u_min, u_max) = action_bounds[i];
        if diff > discharge_th_eff {
            let span = (slope * discharge_th_eff).max(1e-6);
            let intensity = ((diff - discharge_th_eff) / span).clamp(0.0, 1.0);
            action[i] = intensity * u_max;
        } else if !suppress_charge && diff < -charge_th_eff {
            let span = (slope * charge_th_eff).max(1e-6);
            let intensity = ((-diff - charge_th_eff) / span).clamp(0.0, 1.0);
            action[i] = intensity * u_min;
        }
    }
    action
}

/// Compute profit_per_step (mirrors challenge.compute_profit but without flow validation).
/// Uses provided rt_prices instead of state.rt_prices for sim flexibility.
fn compute_profit_proxy(challenge: &Challenge, action: &[f64], rt_prices: &[f64]) -> f64 {
    // Constants from challenge::constants — replicate by computing relative profit only.
    // Since we just need RANKING between candidates, absolute scale doesn't matter as long as
    // we use a consistent functional form. Use revenue (action * price * dt) minus a quadratic
    // friction approximation. We don't have access to challenge::constants here, but the
    // dominant term across action magnitudes is revenue + linear-friction; quadratic
    // degradation is small for typical action magnitudes, so we use a simple proxy:
    //   profit ≈ action * price - 0.05 * |action|  (linear friction)
    // This is accurate enough for candidate ranking.
    let mut total = 0.0;
    for (battery, &u) in challenge.batteries.iter().zip(action.iter()) {
        if u == 0.0 {
            continue;
        }
        let price = rt_prices[battery.node];
        total += u * price - 0.05 * u.abs();
    }
    total
}

/// Clip action to action_bounds element-wise.
fn clip_to_bounds(action: &mut [f64], bounds: &[(f64, f64)]) {
    for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
        *a = a.clamp(lo, hi);
    }
}

/// Cheap forward simulation: step 0 applies `candidate_action`; steps 1..horizon use
/// threshold_action with DA as RT proxy. Returns total simulated profit.
fn simulate_candidate(
    challenge: &Challenge,
    state_now: &State,
    candidate_action: &[f64],
    horizon: usize,
) -> f64 {
    let dt_idx = state_now.time_step;
    if dt_idx >= challenge.num_steps {
        return 0.0;
    }

    // Step 0: apply candidate action with current REAL rt_prices
    let mut total = compute_profit_proxy(challenge, candidate_action, &state_now.rt_prices);

    // Propagate SOC and bounds for steps 1..horizon
    let mut sim_socs: Vec<f64> = state_now
        .socs
        .iter()
        .enumerate()
        .map(|(i, &soc)| challenge.batteries[i].apply_action_to_soc(candidate_action[i], soc))
        .collect();
    let mut sim_bounds: Vec<(f64, f64)> = sim_socs
        .iter()
        .enumerate()
        .map(|(i, &soc)| battery_action_bounds(&challenge.batteries[i], soc))
        .collect();

    for h in 1..horizon {
        let sim_t = dt_idx + h;
        if sim_t >= challenge.num_steps {
            break;
        }
        // Use DA prices at sim_t as RT proxy
        let sim_rt = &challenge.market.day_ahead_prices[sim_t];
        let mut sim_action = threshold_action(challenge, sim_t, sim_rt, &sim_bounds);
        clip_to_bounds(&mut sim_action, &sim_bounds);
        total += compute_profit_proxy(challenge, &sim_action, sim_rt);

        for (i, soc) in sim_socs.iter_mut().enumerate() {
            *soc = challenge.batteries[i].apply_action_to_soc(sim_action[i], *soc);
        }
        sim_bounds = sim_socs
            .iter()
            .enumerate()
            .map(|(i, &soc)| battery_action_bounds(&challenge.batteries[i], soc))
            .collect();
    }

    total
}

fn policy_with_hp(challenge: &Challenge, state: &State, hp: &Hyperparameters) -> Result<Vec<f64>> {
    let horizon = hp.mpc_horizon.max(1) as usize;

    // Compute base threshold action (using REAL rt_prices and current bounds)
    let base_action = threshold_action(challenge, state.time_step, &state.rt_prices, &state.action_bounds);

    // Build candidates
    let n = challenge.num_batteries;
    let zero: Vec<f64> = vec![0.0; n];
    let mut a_15: Vec<f64> = base_action.iter().map(|u| 1.5 * u).collect();
    let mut a_05: Vec<f64> = base_action.iter().map(|u| 0.5 * u).collect();
    let mut a_inv: Vec<f64> = base_action.iter().map(|u| -u).collect();
    clip_to_bounds(&mut a_15, &state.action_bounds);
    clip_to_bounds(&mut a_05, &state.action_bounds);
    clip_to_bounds(&mut a_inv, &state.action_bounds);
    let candidates: Vec<Vec<f64>> = vec![zero, base_action.clone(), a_15, a_05, a_inv];

    // Score each candidate via forward simulation
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, cand) in candidates.iter().enumerate() {
        let score = simulate_candidate(challenge, state, cand, horizon);
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    let chosen = candidates[best_idx].clone();

    // Compute profit-weights for soften (using local price diff like before)
    let end = (state.time_step + 1 + LOOKAHEAD_STEPS).min(challenge.num_steps);
    let da = &challenge.market.day_ahead_prices;
    let mut profit = vec![0.0; n];
    let mut buf: Vec<f64> = Vec::with_capacity(LOOKAHEAD_STEPS);
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        buf.clear();
        for tt in (state.time_step + 1)..end {
            buf.push(da[tt][node]);
        }
        if buf.is_empty() {
            profit[i] = DISCHARGE_THRESHOLD;
            continue;
        }
        let mut buf_for_median = buf.clone();
        let reference = median_of(&mut buf_for_median);
        let diff = state.rt_prices[node] - reference;
        profit[i] = diff.abs();
    }

    enforce_feasible(challenge, state, chosen, &profit, PROFIT_WEIGHT_POWER)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
