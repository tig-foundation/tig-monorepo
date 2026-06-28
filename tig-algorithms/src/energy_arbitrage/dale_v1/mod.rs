// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
//
// `dale_v1` — energy arbitrage solver. Built on the prometheus_eb1 architecture (per-battery DP value
// function MPC + joint PTDF-projected gradient step + active-line coordinate polish) with targeted
// improvements:
//   * Finer DP grids (dp_soc_levels 33 -> 49, dp_action_levels 17 -> 33) so the value function and the
//     policy query grid are resolved at comparable resolution.
//   * More coordinate-polish passes (1 -> 4) to tighten the joint PTDF optimum.
//   * More diverse seeds for the joint projected-gradient ascent (3 -> 6), with a fuel-aware "light"
//     mode that drops back to 3 seeds as the runtime fuel counter approaches its floor.
//   * Holt's exponential smoothing (level + trend) for the RT-vs-DA residual forecast, replacing the
//     static 0.65/0.35 median blend; wider clamp; bounded ring buffers for the percentile bands.
//   * A feasible greedy-drain fallback at fuel exhaustion (instead of abandoning stored energy as zeros)
//     and a stronger horizon-based terminal drain.
//   * Asymmetric jump weighting (upward-skewed), single-counted transaction friction in the policy
//     thresholds, and a central-difference SOC shadow-value gradient.
//
// Fuel is the runtime instruction-budget counter (`__fuel_remaining`), not wall-clock time. We persist a
// valid zero solution before any optimization and self-limit before the counter runs out, so a kill never
// leaves the instance without a feasible `.solution`.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Mutex, OnceLock};
use tig_challenges::energy_arbitrage::*;
use tig_challenges::energy_arbitrage::constants::{
    DELTA_T, EPS_FLOW, ETA_CHARGE, ETA_DISCHARGE, KAPPA_DEG, KAPPA_TX,
};

/// Generic numerical tolerance.
const EPS: f64 = 1e-12;

// `__fuel_remaining` is initialized by the runtime to the fuel cap and decremented by the
// fuel-instrumentation pass as the algorithm executes; it is exported from the built `.so`.
extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

#[inline(always)]
fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
#[serde(default)]
pub struct Hyperparameters {
    /// SOC grid resolution of the per-battery DP value function.
    pub dp_soc_levels: usize,
    /// Action grid resolution used while building the DP value function.
    pub dp_action_levels: usize,
    /// Action grid resolution used when querying the DP at policy time.
    pub policy_action_levels: usize,
    /// Max alternating-projection iterations onto the PTDF feasibility polytope.
    pub proj_max_iters: usize,
    /// Outer iterations of the joint projected-gradient ascent.
    pub grad_outer_iters: usize,
    /// Backtracking line-search iterations per gradient step.
    pub grad_ls_iters: usize,
    /// Bisection iterations of the feasibility-scaling fallback.
    pub bisect_iters: usize,
    /// Passes of the PTDF-aware coordinate polish.
    pub coord_polish_passes: usize,
    /// Day-ahead lookahead window (steps) used for the quantile threshold policy.
    pub lookahead_horizon: usize,
    /// Max fuel (runtime fuel units) the optimization rollout may spend before it falls back to the
    /// drain policy for the remaining steps. 0 = spend all the fuel the runtime makes available
    /// (minus a safety reserve). Always capped so it cannot trigger an out-of-fuel exit.
    pub fuel_budget: u64,
    /// Holt smoothing level (alpha) for the residual forecast.
    pub holt_alpha: f64,
    /// Holt trend smoothing (beta) for the residual forecast.
    pub holt_beta: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            dp_soc_levels: 49,
            dp_action_levels: 33,
            policy_action_levels: 65,
            proj_max_iters: 80,
            grad_outer_iters: 25,
            grad_ls_iters: 6,
            bisect_iters: 30,
            coord_polish_passes: 4,
            lookahead_horizon: 24,
            fuel_budget: 0,
            holt_alpha: 0.22,
            holt_beta: 0.12,
        }
    }
}

impl Hyperparameters {
    fn parse(raw: &Option<Map<String, Value>>) -> Result<Self> {
        let mut hp: Self = match raw {
            Some(map) => serde_json::from_value(Value::Object(map.clone()))
                .map_err(|e| anyhow!("invalid hyperparameters: {}", e))?,
            None => Self::default(),
        };
        hp.dp_soc_levels = hp.dp_soc_levels.max(2);
        hp.dp_action_levels = hp.dp_action_levels.max(3);
        hp.policy_action_levels = hp.policy_action_levels.max(3);
        hp.proj_max_iters = hp.proj_max_iters.max(1);
        hp.grad_ls_iters = hp.grad_ls_iters.max(1);
        hp.bisect_iters = hp.bisect_iters.max(1);
        hp.lookahead_horizon = hp.lookahead_horizon.max(1);
        hp.coord_polish_passes = hp.coord_polish_passes.max(1);
        hp.holt_alpha = hp.holt_alpha.clamp(0.02, 0.8);
        hp.holt_beta = hp.holt_beta.clamp(0.02, 0.8);
        Ok(hp)
    }
}

pub fn help() {
    println!("energy_arbitrage 'dale_v1' — DP-value-function MPC with joint PTDF-projected gradient step,");
    println!("multi-seed ascent, multi-pass coordinate polish, Holt-smoothed residual forecast, and a");
    println!("feasible greedy-drain fallback at fuel exhaustion.");
    println!();
    println!("Hyperparameters (defaults in parentheses):");
    println!("  dp_soc_levels (49)         DP value-function SOC grid resolution");
    println!("  dp_action_levels (33)      action grid used to build the DP");
    println!("  policy_action_levels (65)  action grid used to query the DP at runtime");
    println!("  proj_max_iters (80)        PTDF polytope projection iterations");
    println!("  grad_outer_iters (25)      projected-gradient outer iterations");
    println!("  grad_ls_iters (6)          line-search iterations per gradient step");
    println!("  bisect_iters (30)          feasibility-scaling bisection iterations");
    println!("  coord_polish_passes (4)    PTDF-aware coordinate-polish passes");
    println!("  lookahead_horizon (24)     day-ahead quantile lookahead window (steps)");
    println!("  fuel_budget (0)            max fuel to spend optimizing; 0 = use all available");
    println!("  holt_alpha (0.22)          Holt level smoothing for residual forecast");
    println!("  holt_beta (0.12)           Holt trend smoothing for residual forecast");
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    challenge.network.compute_flows(&injections)
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    challenge.network.verify_flows(&flows).is_ok()
}

fn clamp_to_bounds(action: &mut [f64], bounds: &[(f64, f64)]) {
    for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
        if *a < lo {
            *a = lo;
        }
        if *a > hi {
            *a = hi;
        }
    }
}

fn edge_sized_fraction(edge: f64, price_band: f64) -> f64 {
    if edge <= 0.0 {
        0.0
    } else {
        let normalized = edge / price_band.max(5.0);
        // Lower floor than a fixed 0.35 so sub-threshold edges idle instead of paying friction.
        (0.15 + 0.85 * normalized).clamp(0.0, 1.0)
    }
}

fn relative_soc_pressure(battery: &Battery, soc: f64) -> f64 {
    let span = (battery.soc_max_mwh - battery.soc_min_mwh).max(1e-9);
    ((soc - battery.soc_min_mwh) / span).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// RT-price / residual history with Holt exponential smoothing and bounded
// ring buffers for the percentile bands.
// ---------------------------------------------------------------------------
const RING_CAP: usize = 64;

#[derive(Clone)]
struct RtHistory {
    num_nodes: usize,
    // Holt state per node: level and trend of the (RT - DA) residual.
    level: Vec<f64>,
    trend: Vec<f64>,
    init: Vec<bool>,
    // Bounded recent samples for percentile-based spike/dip bands.
    ring_price: Vec<Vec<f64>>,
    ring_resid: Vec<Vec<f64>>,
}

static RT_HISTORY: OnceLock<Mutex<RtHistory>> = OnceLock::new();

fn history_lock() -> &'static Mutex<RtHistory> {
    RT_HISTORY.get_or_init(|| {
        Mutex::new(RtHistory {
            num_nodes: 0,
            level: Vec::new(),
            trend: Vec::new(),
            init: Vec::new(),
            ring_price: Vec::new(),
            ring_resid: Vec::new(),
        })
    })
}

fn ring_push(buf: &mut Vec<f64>, val: f64) {
    if buf.len() >= RING_CAP {
        buf.remove(0);
    }
    buf.push(val);
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) * numerator) / denominator;
    sorted[idx]
}

struct BatteryDP {
    soc_lo: f64,
    soc_step_inv: f64,
    levels: usize,
    values: Vec<Vec<f64>>,
}

fn immediate_profit(battery: &Battery, action: f64, price: f64) -> f64 {
    let throughput = action.abs() * DELTA_T;
    action * price * DELTA_T
        - KAPPA_TX * throughput
        - KAPPA_DEG * (throughput / battery.capacity_mwh).powi(2)
}

fn interp_value(values: &[f64], soc: f64, lo: f64, step_inv: f64, last: usize) -> f64 {
    let pos = ((soc - lo) * step_inv).clamp(0.0, last as f64);
    let low = pos.floor() as usize;
    let high = (low + 1).min(last);
    let alpha = pos - low as f64;
    values[low] * (1.0 - alpha) + values[high] * alpha
}

fn adaptive_action_grid(
    battery: &Battery,
    charge_max: f64,
    discharge_min: f64,
    price: f64,
    levels: usize,
) -> Vec<f64> {
    if levels < 3 {
        return vec![0.0];
    }

    let mut actions = Vec::new();
    let base_charge = -battery.power_charge_mw;
    let base_discharge = battery.power_discharge_mw;

    actions.push(base_charge);
    actions.push(0.0);
    actions.push(base_discharge);

    let in_discharge_region = price > discharge_min;
    let in_charge_region = price < charge_max;

    let mut discharge_points = Vec::new();
    let mut charge_points = Vec::new();

    // Allocate the grid evenly between the two regions to avoid silently dropping candidates when
    // both regions are active.
    let half = ((levels as f64) * 0.5).round() as usize;

    if in_discharge_region {
        let discharge_levels = half.max(2);
        for i in 1..discharge_levels {
            let frac = i as f64 / (discharge_levels as f64);
            discharge_points.push(frac * base_discharge);
        }
    }

    if in_charge_region {
        let charge_levels = half.max(2);
        for i in 1..charge_levels {
            let frac = i as f64 / (charge_levels as f64);
            charge_points.push(-frac * battery.power_charge_mw);
        }
    }

    let total_points = actions.len() + discharge_points.len() + charge_points.len();
    if total_points < levels {
        let remaining = levels - total_points;
        for i in 1..remaining {
            let frac = -1.0 + 2.0 * (i as f64) / ((remaining - 1) as f64);
            let action = if frac >= 0.0 {
                frac * base_discharge
            } else {
                frac * battery.power_charge_mw
            };
            actions.push(action);
        }
    }

    actions.extend(discharge_points);
    actions.extend(charge_points);

    actions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    actions.dedup_by(|a, b| (*a - *b).abs() < EPS);

    if actions.len() > levels {
        let mut kept = vec![base_charge, 0.0, base_discharge];
        let mut candidates: Vec<(f64, f64)> = actions
            .iter()
            .filter(|&&a| ![base_charge, 0.0, base_discharge].contains(&a))
            .map(|&a| (a, (a - if price > discharge_min { base_discharge } else if price < charge_max { base_charge } else { 0.0 }).abs()))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        kept.extend(candidates.iter().take(levels - 3).map(|(a, _)| *a));
        kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        kept.dedup_by(|a, b| (*a - *b).abs() < EPS);
        kept
    } else {
        actions
    }
}

/// Feasible signed action bounds `(u_min, u_max)` for one battery. Mirrors
/// `Battery::compute_action_bounds` (crate-private) using only public fields.
fn compute_action_bounds(battery: &Battery, soc: f64) -> (f64, f64) {
    let dt = DELTA_T;

    let headroom = (battery.soc_max_mwh - soc).max(0.0);
    let available = (soc - battery.soc_min_mwh).max(0.0);

    let max_charge_from_soc = if battery.efficiency_charge > 0.0 {
        headroom / (battery.efficiency_charge * dt)
    } else {
        0.0
    };
    let max_discharge_from_soc = if battery.efficiency_discharge > 0.0 {
        available * battery.efficiency_discharge / dt
    } else {
        0.0
    };

    let max_charge = max_charge_from_soc.min(battery.power_charge_mw).max(0.0);
    let max_discharge = max_discharge_from_soc.min(battery.power_discharge_mw).max(0.0);

    (-max_charge, max_discharge)
}

fn build_battery_dp(
    battery: &Battery,
    da_at_node: &[f64],
    num_steps: usize,
    sigma: f64,
    p_jump: f64,
    mean_pareto: f64,
    second_pareto: f64,
    hp: &Hyperparameters,
) -> BatteryDP {
    let levels = hp.dp_soc_levels;
    let soc_lo = battery.soc_min_mwh;
    let span = (battery.soc_max_mwh - battery.soc_min_mwh).max(1e-9);
    let soc_step = span / (levels - 1) as f64;
    let soc_step_inv = 1.0 / soc_step;

    let mut bounds = Vec::with_capacity(levels);
    for s_idx in 0..levels {
        let soc = soc_lo + soc_step * s_idx as f64;
        let (lo, hi) = compute_action_bounds(battery, soc);
        bounds.push((lo, hi));
    }

    let mut values = vec![vec![0.0; levels]; num_steps + 1];
    let last = levels - 1;

    // Asymmetric jump weighting: energy prices have a strong upward skew (Pareto tail on the high
    // side), so we tilt the normal-mass split toward the high scenario to keep inventory for spikes.
    let w_jump = p_jump.clamp(0.0, 1.0);
    let w_normal = (1.0 - w_jump).max(0.0);
    let w_low = 0.4 * w_normal;
    let w_high = 0.6 * w_normal;
    let jump_floor = 1.0_f64;
    let jump_ceiling = if second_pareto.is_finite()
        && mean_pareto.is_finite()
        && mean_pareto > jump_floor + EPS
    {
        ((second_pareto - mean_pareto * jump_floor) / (mean_pareto - jump_floor))
            .max(mean_pareto)
            .min(80.0)
    } else {
        mean_pareto.max(jump_floor).min(80.0)
    };
    let w_jump_high = if jump_ceiling > jump_floor + EPS {
        w_jump * ((mean_pareto - jump_floor) / (jump_ceiling - jump_floor)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let w_jump_low = (w_jump - w_jump_high).max(0.0);

    // The DP's break-even friction keeps a margin for forecast error (round trip pays KAPPA_TX twice).
    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let friction = 2.0 * KAPPA_TX;

    for t in (0..num_steps).rev() {
        let da = da_at_node[t];
        let price_low = da * (1.0 - sigma);
        let price_high = da * (1.0 + sigma);
        let price_jump_low = da * (1.0 + jump_floor);
        let price_jump_high = da * (1.0 + jump_ceiling);

        let q_low = price_low;
        let q_high = price_high;
        let charge_max = q_high * eta_rt - friction;
        let discharge_min = q_low / eta_rt + friction;

        let (left, right) = values.split_at_mut(t + 1);
        let current = &mut left[t];
        let next = &right[0];

        for s_idx in 0..levels {
            let (lo, hi) = bounds[s_idx];
            let soc = soc_lo + soc_step * s_idx as f64;

            let actions = adaptive_action_grid(
                battery,
                charge_max,
                discharge_min,
                (price_low + price_high) * 0.5,
                hp.dp_action_levels,
            );

            let mut best_low = f64::NEG_INFINITY;
            let mut best_high = f64::NEG_INFINITY;
            let mut best_jump_low = f64::NEG_INFINITY;
            let mut best_jump_high = f64::NEG_INFINITY;

            for &raw in &actions {
                let action = raw.clamp(lo, hi);
                let future = {
                    let next_soc = battery.apply_action_to_soc(action, soc);
                    interp_value(next, next_soc, soc_lo, soc_step_inv, last)
                };

                best_low = best_low.max(immediate_profit(battery, action, price_low) + future);
                best_high = best_high.max(immediate_profit(battery, action, price_high) + future);
                best_jump_low = best_jump_low.max(immediate_profit(battery, action, price_jump_low) + future);
                best_jump_high = best_jump_high.max(immediate_profit(battery, action, price_jump_high) + future);
            }
            current[s_idx] = w_low * best_low
                + w_high * best_high
                + w_jump_low * best_jump_low
                + w_jump_high * best_jump_high;
        }
    }

    BatteryDP {
        soc_lo,
        soc_step_inv,
        levels,
        values,
    }
}

fn dp_action_value(
    dp: &BatteryDP,
    battery: &Battery,
    t: usize,
    soc: f64,
    price: f64,
    action: f64,
) -> f64 {
    let next_t = (t + 1).min(dp.values.len() - 1);
    let next_soc = battery.apply_action_to_soc(action, soc);
    immediate_profit(battery, action, price)
        + interp_value(
            &dp.values[next_t],
            next_soc,
            dp.soc_lo,
            dp.soc_step_inv,
            dp.levels - 1,
        )
}

/// Central-difference shadow value dV/dSOC (less biased than the one-sided difference).
fn dv_dsoc(dp: &BatteryDP, t: usize, soc: f64) -> f64 {
    let next_t = (t + 1).min(dp.values.len() - 1);
    let values = &dp.values[next_t];
    let last = dp.levels - 1;
    if last == 0 {
        return 0.0;
    }
    let pos = ((soc - dp.soc_lo) * dp.soc_step_inv).clamp(0.0, last as f64);
    let low = pos.floor() as usize;
    if low == 0 {
        (values[1] - values[0]) * dp.soc_step_inv
    } else if low >= last {
        (values[last] - values[last - 1]) * dp.soc_step_inv
    } else {
        (values[low + 1] - values[low - 1]) * (0.5 * dp.soc_step_inv)
    }
}

fn pick_dp_action(
    dp: &BatteryDP,
    battery: &Battery,
    t: usize,
    soc: f64,
    price: f64,
    bounds: (f64, f64),
    hp: &Hyperparameters,
) -> f64 {
    let (lo, hi) = bounds;
    let mut best_action = 0.0_f64.clamp(lo, hi);
    let mut best_value = dp_action_value(dp, battery, t, soc, price, best_action);

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    // Policy-time friction is single-counted: the true per-step profit pays KAPPA_TX once, so gating
    // on a single KAPPA_TX captures marginal-but-profitable trades the DP's round-trip margin skips.
    let friction = KAPPA_TX;
    let q_low = price;
    let q_high = price;
    let charge_max = q_high * eta_rt - friction;
    let discharge_min = q_low / eta_rt + friction;

    for raw in adaptive_action_grid(battery, charge_max, discharge_min, price, hp.policy_action_levels) {
        let action = raw.clamp(lo, hi);
        let value = dp_action_value(dp, battery, t, soc, price, action);
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }
    for action in [lo, hi] {
        let value = dp_action_value(dp, battery, t, soc, price, action);
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }

    best_action
}

// ---- Joint per-step optimization with PTDF projection ----

fn build_sensitivity(challenge: &Challenge) -> Vec<Vec<f64>> {
    let m = challenge.num_batteries;
    let n_lines = challenge.network.num_lines;
    let slack = challenge.network.slack_bus;
    let mut sens = vec![vec![0.0; m]; n_lines];
    for l in 0..n_lines {
        let ptdf_slack = challenge.network.ptdf[l][slack];
        for b in 0..m {
            let node = challenge.batteries[b].node;
            sens[l][b] = challenge.network.ptdf[l][node] - ptdf_slack;
        }
    }
    sens
}

#[inline]
fn line_flow(sens_row: &[f64], action: &[f64], base: f64) -> f64 {
    let mut f = base;
    for b in 0..action.len() {
        f += sens_row[b] * action[b];
    }
    f
}

fn project_polytope(
    action: &mut [f64],
    bounds: &[(f64, f64)],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    limits: &[f64],
    max_iters: usize,
) -> bool {
    let n_lines = sens.len();
    for _ in 0..max_iters {
        for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
            if *a < lo { *a = lo; }
            if *a > hi { *a = hi; }
        }
        let mut worst_l: usize = usize::MAX;
        let mut worst_excess: f64 = 0.0;
        let mut worst_sign: f64 = 0.0;
        let mut worst_limit: f64 = 1.0;
        for l in 0..n_lines {
            let f = line_flow(&sens[l], action, base_flows[l]);
            let limit = limits[l];
            let excess = f.abs() - limit;
            if excess > worst_excess {
                worst_excess = excess;
                worst_l = l;
                worst_sign = if f >= 0.0 { 1.0 } else { -1.0 };
                worst_limit = limit;
            }
        }
        if worst_l == usize::MAX || worst_excess <= EPS_FLOW * worst_limit.max(1.0) {
            return true;
        }
        let row = &sens[worst_l];
        let norm_sq: f64 = row.iter().map(|x| x * x).sum();
        if norm_sq < 1e-14 {
            return false;
        }
        let mu = worst_excess / norm_sq;
        for b in 0..action.len() {
            action[b] -= worst_sign * mu * row[b];
        }
    }
    for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
        if *a < lo { *a = lo; }
        if *a > hi { *a = hi; }
    }
    for l in 0..n_lines {
        let f = line_flow(&sens[l], action, base_flows[l]);
        if f.abs() > limits[l] * (1.0 + EPS_FLOW) + 1e-6 {
            return false;
        }
    }
    true
}

fn safe_project_to_feasible(
    challenge: &Challenge,
    state: &State,
    action: &mut Vec<f64>,
    sens: &[Vec<f64>],
    base_flows: &[f64],
    hp: &Hyperparameters,
) {
    let limits = &challenge.network.flow_limits;
    let ok = project_polytope(action, &state.action_bounds, sens, base_flows, limits, hp.proj_max_iters);
    if ok && is_flow_feasible(challenge, state, action) {
        return;
    }
    let original = action.clone();
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..hp.bisect_iters {
        let mid = 0.5 * (lo + hi);
        for b in 0..action.len() {
            action[b] = original[b] * mid;
        }
        clamp_to_bounds(action, &state.action_bounds);
        if is_flow_feasible(challenge, state, action) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    for b in 0..action.len() {
        action[b] = original[b] * lo;
    }
    clamp_to_bounds(action, &state.action_bounds);
    if !is_flow_feasible(challenge, state, action) {
        for a in action.iter_mut() { *a = 0.0; }
    }
}

/// Cheap bisection-scale-to-zero feasibility recovery used by the drain path. Fewer iterations than
/// `safe_project_to_feasible` (no halfspace projection) so it stays fuel-thrifty.
fn scale_to_feasible(
    challenge: &Challenge,
    state: &State,
    action: &mut Vec<f64>,
    iters: usize,
) {
    let original = action.clone();
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        for b in 0..action.len() {
            action[b] = original[b] * mid;
        }
        clamp_to_bounds(action, &state.action_bounds);
        if is_flow_feasible(challenge, state, action) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    for b in 0..action.len() {
        action[b] = original[b] * lo;
    }
    clamp_to_bounds(action, &state.action_bounds);
    if !is_flow_feasible(challenge, state, action) {
        for a in action.iter_mut() { *a = 0.0; }
    }
}

fn total_step_value(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    action: &[f64],
) -> f64 {
    let mut total = 0.0;
    for b in 0..challenge.num_batteries {
        let battery = &challenge.batteries[b];
        total += dp_action_value(
            &dps[b],
            battery,
            state.time_step,
            state.socs[b],
            state.rt_prices[battery.node],
            action[b],
        );
    }
    total
}

fn analytic_gradient(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    action: &[f64],
) -> Vec<f64> {
    let mut grad = vec![0.0_f64; action.len()];
    for b in 0..action.len() {
        let battery = &challenge.batteries[b];
        let price = state.rt_prices[battery.node];
        let u = action[b];
        let s = if u > EPS { 1.0 } else if u < -EPS { -1.0 } else { 0.0 };
        let cap2 = battery.capacity_mwh.powi(2).max(1e-9);
        let imm = price * DELTA_T
            - s * KAPPA_TX * DELTA_T
            - 2.0 * KAPPA_DEG * DELTA_T * DELTA_T * u / cap2;

        let next_soc = battery.apply_action_to_soc(u, state.socs[b]);
        let dsoc_du = if u > 0.0 {
            if next_soc <= battery.soc_min_mwh + EPS { 0.0 } else { -DELTA_T / ETA_DISCHARGE }
        } else if u < 0.0 {
            if next_soc >= battery.soc_max_mwh - EPS { 0.0 } else { -ETA_CHARGE * DELTA_T }
        } else {
            -0.5 * (DELTA_T / ETA_DISCHARGE + ETA_CHARGE * DELTA_T)
        };
        let dv = dv_dsoc(&dps[b], state.time_step, next_soc);
        grad[b] = imm + dv * dsoc_du;
    }
    grad
}

fn projected_gradient_ascent(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    seed: Vec<f64>,
    hp: &Hyperparameters,
) -> (Vec<f64>, f64) {
    let mut action = seed;
    safe_project_to_feasible(challenge, state, &mut action, sens, base_flows, hp);
    let mut best_value = total_step_value(challenge, state, dps, &action);
    let mut best_action = action.clone();

    let max_power: f64 = challenge
        .batteries
        .iter()
        .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
        .fold(1.0_f64, f64::max);

    let mut lr = max_power * 0.5;
    for _ in 0..hp.grad_outer_iters {
        let grad = analytic_gradient(challenge, state, dps, &action);
        let g_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm < 1e-9 {
            break;
        }

        let mut improved = false;
        let mut cur_lr = lr;
        for _ in 0..hp.grad_ls_iters {
            let step_scale = cur_lr / g_norm;
            let mut trial: Vec<f64> = action
                .iter()
                .zip(grad.iter())
                .map(|(a, g)| a + step_scale * g)
                .collect();
            safe_project_to_feasible(challenge, state, &mut trial, sens, base_flows, hp);
            let v = total_step_value(challenge, state, dps, &trial);
            if v > best_value + 1e-9 {
                action = trial.clone();
                best_value = v;
                best_action = trial;
                improved = true;
                lr = cur_lr * 1.4;
                break;
            }
            cur_lr *= 0.5;
        }
        if !improved {
            lr *= 0.4;
            if lr < max_power * 1e-4 {
                break;
            }
        }
    }
    (best_action, best_value)
}

fn joint_optimize_step(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    seeds: Vec<Vec<f64>>,
    hp: &Hyperparameters,
) -> Vec<f64> {
    let mut best_action = vec![0.0_f64; challenge.num_batteries];
    let mut best_value = total_step_value(challenge, state, dps, &best_action);

    for seed in seeds {
        let (a, v) = projected_gradient_ascent(challenge, state, dps, sens, base_flows, seed, hp);
        if v > best_value && is_flow_feasible(challenge, state, &a) {
            best_value = v;
            best_action = a;
        }
    }
    best_action
}

fn coordinate_polish_step(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    mut action: Vec<f64>,
    hp: &Hyperparameters,
) -> Vec<f64> {
    if !is_flow_feasible(challenge, state, &action) {
        return action;
    }

    let mut best_value = total_step_value(challenge, state, dps, &action);
    for _ in 0..hp.coord_polish_passes {
        let mut improved = false;
        for b in 0..challenge.num_batteries {
            let (lo, hi) = state.action_bounds[b];
            let cur = action[b];
            let current_flows = compute_flows(challenge, state, &action);
            let mut net_lo = lo;
            let mut net_hi = hi;
            for l in 0..challenge.network.num_lines {
                let coeff = sens[l][b];
                if coeff.abs() <= 1e-12 {
                    continue;
                }
                let without_b = current_flows[l] - coeff * cur;
                let limit = challenge.network.flow_limits[l];
                let low_at_line = (-limit - without_b) / coeff;
                let high_at_line = (limit - without_b) / coeff;
                let line_lo = low_at_line.min(high_at_line);
                let line_hi = low_at_line.max(high_at_line);
                net_lo = net_lo.max(line_lo);
                net_hi = net_hi.min(line_hi);
            }
            let span = (hi - lo).max(0.0);
            let net_span = net_hi - net_lo;
            if span <= EPS {
                continue;
            }

            let mut candidates = vec![
                0.0_f64.clamp(lo, hi),
                lo,
                hi,
                lo + 0.25 * span,
                lo + 0.50 * span,
                lo + 0.75 * span,
                (cur - 0.25 * span).clamp(lo, hi),
                (cur + 0.25 * span).clamp(lo, hi),
            ];
            if net_span > EPS {
                candidates.extend([
                    net_lo.clamp(lo, hi),
                    net_hi.clamp(lo, hi),
                    (net_lo + 0.25 * net_span).clamp(lo, hi),
                    (net_lo + 0.50 * net_span).clamp(lo, hi),
                    (net_lo + 0.75 * net_span).clamp(lo, hi),
                ]);
            }

            let mut best_b_action = cur;
            let mut best_b_value = best_value;
            for &candidate in candidates.iter() {
                if (candidate - cur).abs() <= EPS {
                    continue;
                }
                let mut trial = action.clone();
                trial[b] = candidate;
                if !is_flow_feasible(challenge, state, &trial) {
                    continue;
                }
                let value = total_step_value(challenge, state, dps, &trial);
                if value > best_b_value + 1e-9 {
                    best_b_value = value;
                    best_b_action = candidate;
                }
            }

            if (best_b_action - cur).abs() > EPS {
                action[b] = best_b_action;
                best_value = best_b_value;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    action
}

// ---------------------------------------------------------------------------
// Feasible greedy-drain fallback for when fuel is exhausted. Discharges stored
// energy at prices that beat the transaction cost and are not below the
// near-future mean, bisection-scaled to flow feasibility. Cheap (no gradient
// ascent, no halfspace projection) so the reserve covers the whole tail.
// ---------------------------------------------------------------------------
fn cheap_drain(challenge: &Challenge, state: &State, ctx: &Ctx) -> Vec<f64> {
    let t = state.time_step;
    let n_remaining = challenge.num_steps.saturating_sub(t);
    let mut action = vec![0.0_f64; challenge.num_batteries];

    for b in 0..challenge.num_batteries {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let price = state.rt_prices[node];
        let (u_min, u_max) = state.action_bounds[b];
        if u_max <= 0.0 {
            continue;
        }
        if price <= KAPPA_TX {
            // Discharging at or below the transaction cost is value-destructive; hold.
            continue;
        }
        let pressure = relative_soc_pressure(battery, state.socs[b]);
        if pressure <= 0.0 {
            continue;
        }

        // Steps of full discharge needed to reach soc_min.
        let withdrawable_mwh = (state.socs[b] - battery.soc_min_mwh).max(0.0);
        let mwh_per_step = u_max * DELTA_T / ETA_DISCHARGE;
        let discharge_steps_to_min = if mwh_per_step > 1e-9 {
            (withdrawable_mwh / mwh_per_step).ceil() as usize
        } else {
            usize::MAX
        };
        let must_empty = n_remaining <= discharge_steps_to_min.saturating_add(1);

        let fm = ctx.future_mean[t].get(node).copied().unwrap_or(price);
        let decent_price = price >= fm * 0.9;

        if must_empty {
            // No time left to wait for a better price; liquidate at full rate.
            action[b] = u_max;
        } else if decent_price {
            // Above-average price: discharge, more aggressively the fuller the battery.
            action[b] = u_max * (0.5 + 0.5 * pressure);
        }
        // otherwise hold for a better price later
        let _ = u_min;
    }

    clamp_to_bounds(&mut action, &state.action_bounds);
    scale_to_feasible(challenge, state, &mut action, 10);
    action
}

/// Shared context for the per-step policy.
struct Ctx<'a> {
    dps: &'a [BatteryDP],
    sens: &'a [Vec<f64>],
    hp: &'a Hyperparameters,
    fuel_floor: u64,
    future_mean: &'a [Vec<f64>],
}

fn policy(challenge: &Challenge, state: &State, ctx: &Ctx) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_steps = challenge.num_steps;
    let n_remaining = n_steps.saturating_sub(t);
    if n_remaining == 0 {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    // Fuel-aware dispatch. At the floor we can no longer afford the full optimization, so fall back
    // to the feasible greedy-drain (which still earns from stored energy). Near the floor, drop to
    // the light configuration (fewer seeds, one polish pass, half the gradient iterations) so a
    // single step cannot overshoot the remaining budget.
    let fr = fuel_remaining();
    if fr <= ctx.fuel_floor {
        return Ok(cheap_drain(challenge, state, ctx));
    }
    let light = fr <= ctx.fuel_floor + (ctx.fuel_floor / 4).max(1);

    let mut hp = *ctx.hp;
    if light {
        hp.coord_polish_passes = 1;
        hp.grad_outer_iters = (hp.grad_outer_iters / 2).max(5);
    }

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let horizon = hp.lookahead_horizon.min(n_remaining);
    let mut target = vec![0.0_f64; challenge.num_batteries];

    let friction = KAPPA_TX;
    let hours_left = (n_remaining as f64) * DELTA_T;
    let allow_charge = hours_left >= 1.5;

    let mut soc_ranks: Vec<(f64, usize)> = challenge
        .batteries
        .iter()
        .enumerate()
        .map(|(b, battery)| (relative_soc_pressure(battery, state.socs[b]), b))
        .collect();
    soc_ranks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut terminal_rank = vec![challenge.num_batteries; challenge.num_batteries];
    for (rank, &(_, b)) in soc_ranks.iter().enumerate() {
        terminal_rank[b] = rank;
    }

    // --- Holt exponential smoothing of the RT-DA residual (forecast bias + trend) ---
    let mut history = history_lock().lock().unwrap();
    if state.time_step == 0 || history.num_nodes != challenge.network.num_nodes {
        history.num_nodes = challenge.network.num_nodes;
        let n = challenge.network.num_nodes;
        history.level = vec![0.0; n];
        history.trend = vec![0.0; n];
        history.init = vec![false; n];
        history.ring_price = vec![Vec::new(); n];
        history.ring_resid = vec![Vec::new(); n];
    }
    let mut rt_bands = vec![None; challenge.network.num_nodes];
    let mut residual_shift = vec![0.0_f64; challenge.network.num_nodes];
    let alpha = hp.holt_alpha;
    let beta = hp.holt_beta;
    for node in 0..challenge.network.num_nodes {
        if history.ring_price[node].len() >= 12 {
            let mut sorted = history.ring_price[node].clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q15 = percentile(&sorted, 15, 100);
            let q85 = percentile(&sorted, 85, 100);
            if q85 - q15 > 2.0 {
                rt_bands[node] = Some((q15, q85));
            }
        }
        if history.init[node] {
            // Holt update: L = a*x + (1-a)(L_prev + T_prev); T = b*(L - L_prev) + (1-b)T_prev
            let prev_level = history.level[node];
            let prev_trend = history.trend[node];
            // x = last observed residual (most recent entry in the residual ring)
            let x = *history.ring_resid[node].last().unwrap_or(&0.0);
            let new_level = alpha * x + (1.0 - alpha) * (prev_level + prev_trend);
            let new_trend = beta * (new_level - prev_level) + (1.0 - beta) * prev_trend;
            history.level[node] = new_level;
            history.trend[node] = new_trend;
            let forecast = new_level + new_trend;
            residual_shift[node] = forecast.clamp(-100.0, 100.0);
        }
    }

    for (b, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let current_price = state.rt_prices[node];
        let (u_min, u_max) = state.action_bounds[b];

        let end = (t + horizon).min(n_steps);
        let mut future: Vec<f64> = Vec::with_capacity(end - t);
        let shift = residual_shift[node];
        for tau in t..end {
            future.push(challenge.market.day_ahead_prices[tau][node] + shift);
        }
        future.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = future.len();
        let q_low_idx = n / 4;
        let q_high_idx = ((3 * n) / 4).min(n - 1);
        let q_low = future[q_low_idx];
        let q_high = future[q_high_idx];
        let price_band = (q_high - q_low).abs();

        let charge_max = q_high * eta_rt - friction;
        let discharge_min = q_low / eta_rt + friction;

        let discharge_steps_to_min = if u_max > EPS {
            let withdrawable_mwh = (state.socs[b] - battery.soc_min_mwh).max(0.0);
            let mwh_per_step = u_max * DELTA_T / ETA_DISCHARGE;
            if mwh_per_step > 1e-9 {
                (withdrawable_mwh / mwh_per_step).ceil() as usize
            } else {
                usize::MAX
            }
        } else {
            usize::MAX
        };
        let terminal_drain = n_remaining <= discharge_steps_to_min.saturating_add(1);
        let rank_frac = (terminal_rank[b] as f64 + 1.0) / (challenge.num_batteries.max(1) as f64);
        // Broadened end-of-horizon drain: any battery with charge, within 48 steps of the end, at a
        // price that clears the transaction cost.
        let early_terminal_drain = n_remaining <= 48
            && u_max > 0.0
            && relative_soc_pressure(battery, state.socs[b]) > 0.2
            && current_price > KAPPA_TX;

        let mut a = 0.0_f64;
        if terminal_drain && u_max > 0.0 && current_price > KAPPA_TX {
            a = u_max;
        } else if early_terminal_drain {
            let urgency = (1.0 - n_remaining as f64 / 48.0).clamp(0.0, 1.0);
            let fullness = relative_soc_pressure(battery, state.socs[b]);
            let rank_boost = (0.65 - rank_frac).max(0.0);
            let fraction = (0.25 + 0.55 * urgency + 0.35 * fullness + 0.25 * rank_boost).clamp(0.35, 1.0);
            a = u_max * fraction;
        } else if u_max > 0.0 && current_price > discharge_min {
            let fraction = edge_sized_fraction(current_price - discharge_min, price_band);
            a = u_max * fraction;
        } else if allow_charge && u_min < 0.0 && current_price < charge_max {
            let fraction = edge_sized_fraction(charge_max - current_price, price_band);
            a = u_min * fraction;
        }

        if let Some((rt_low, rt_high)) = rt_bands[node] {
            let rt_band = (rt_high - rt_low).max(price_band).max(5.0);
            if u_max > 0.0 && current_price > rt_high + friction {
                let fraction = edge_sized_fraction(current_price - rt_high - friction, rt_band);
                let spike_action = u_max * fraction;
                if spike_action.abs() > a.abs() || a < 0.0 {
                    a = spike_action;
                }
            } else if allow_charge && u_min < 0.0 && current_price < rt_low * eta_rt - friction {
                let fraction =
                    edge_sized_fraction(rt_low * eta_rt - friction - current_price, rt_band);
                let dip_action = u_min * fraction;
                if dip_action.abs() > a.abs() || a > 0.0 {
                    a = dip_action;
                }
            }
        }

        let dp_action = pick_dp_action(
            &ctx.dps[b],
            battery,
            t,
            state.socs[b],
            current_price,
            state.action_bounds[b],
            &hp,
        );
        if dp_action_value(&ctx.dps[b], battery, t, state.socs[b], current_price, dp_action)
            > dp_action_value(&ctx.dps[b], battery, t, state.socs[b], current_price, a) + EPS
        {
            a = dp_action;
        }

        target[b] = a;
    }

    // Record this step's RT observations for the next call's forecast.
    for node in 0..challenge.network.num_nodes {
        let resid = state.rt_prices[node] - challenge.market.day_ahead_prices[t][node];
        if !history.init[node] {
            history.level[node] = resid;
            history.trend[node] = 0.0;
            history.init[node] = true;
        }
        ring_push(&mut history.ring_price[node], state.rt_prices[node]);
        ring_push(&mut history.ring_resid[node], resid);
    }
    drop(history);

    clamp_to_bounds(&mut target, &state.action_bounds);

    let dp_seed: Vec<f64> = (0..challenge.num_batteries)
        .map(|b| {
            let battery = &challenge.batteries[b];
            pick_dp_action(
                &ctx.dps[b],
                battery,
                t,
                state.socs[b],
                state.rt_prices[battery.node],
                state.action_bounds[b],
                &hp,
            )
        })
        .collect();

    let zero = vec![0.0_f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero);

    // Diverse seed set for the joint projected-gradient ascent: the heuristic target, the
    // independent DP seed, the all-zero baseline, and scaled variants to escape PTDF-induced local
    // maxima. The light (fuel-near-floor) configuration trims back to the three core seeds.
    let scale = |v: &[f64], s: f64| -> Vec<f64> {
        v.iter().map(|&x| x * s).collect()
    };
    let mut seeds: Vec<Vec<f64>> = Vec::with_capacity(6);
    seeds.push(target.clone());
    seeds.push(dp_seed.clone());
    seeds.push(zero.clone());
    if !light {
        seeds.push(scale(&target, 0.5));
        seeds.push(scale(&target, 1.5));
        seeds.push(scale(&dp_seed, 0.7));
    }

    let mut result = joint_optimize_step(challenge, state, ctx.dps, ctx.sens, &base_flows, seeds, &hp);
    result = coordinate_polish_step(challenge, state, ctx.dps, ctx.sens, result, &hp);

    if !is_flow_feasible(challenge, state, &result) {
        result = zero;
    }
    Ok(result)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp = Hyperparameters::parse(hyperparameters)?;

    let sigma = challenge.market.params.volatility.max(0.0);
    let p_jump = challenge.market.params.jump_probability.clamp(0.0, 1.0);
    let alpha = challenge.market.params.tail_index;
    let mean_pareto = if alpha > 1.0 {
        alpha / (alpha - 1.0)
    } else {
        50.0
    };
    let second_pareto = if alpha > 2.0 {
        alpha / (alpha - 2.0)
    } else {
        6400.0
    };

    let dps: Vec<BatteryDP> = challenge
        .batteries
        .iter()
        .map(|battery| {
            let node = battery.node;
            let da_at_node: Vec<f64> = (0..challenge.num_steps)
                .map(|t| challenge.market.day_ahead_prices[t][node])
                .collect();
            build_battery_dp(
                battery,
                &da_at_node,
                challenge.num_steps,
                sigma,
                p_jump,
                mean_pareto,
                second_pareto,
                &hp,
            )
        })
        .collect();

    let sens = build_sensitivity(challenge);

    // Per-node, per-step mean of the remaining day-ahead prices. Used by the drain fallback to
    // avoid dumping stored energy at a trough when a better price is expected later. Computed with
    // a backward running sum (O(num_steps * num_nodes)).
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let mut future_mean: Vec<Vec<f64>> = vec![vec![0.0; num_nodes]; num_steps];
    for node in 0..num_nodes {
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for t in (0..num_steps).rev() {
            sum += challenge.market.day_ahead_prices[t][node];
            cnt += 1;
            future_mean[t][node] = sum / cnt as f64;
        }
    }

    // Initial feasible solution: all zeros. Saved before any optimization so a kill or out-of-fuel
    // exit never leaves the instance without a valid `.solution`.
    let zero_solution = Solution {
        schedule: vec![vec![0.0; challenge.num_batteries]; challenge.num_steps],
    };
    save_solution(&zero_solution)?;

    // Budget the rollout's fuel spend. The reserve is slightly larger than prometheus's 1/28 so it
    // covers the (cheap but nonzero) drain tail once the floor is reached. `fuel_budget == 0` spends
    // all available fuel minus the reserve; a positive value caps spend lower to trade fuel for
    // quality. Budgeting off fuel (not wall time) keeps the degrade-to-drain fallback deterministic.
    let available = fuel_remaining();
    let reserve = available / 20;
    let max_spend = available.saturating_sub(reserve);
    let target_spend = if hp.fuel_budget == 0 {
        max_spend
    } else {
        hp.fuel_budget.min(max_spend)
    };
    let fuel_floor = available - target_spend;

    let ctx = Ctx {
        dps: &dps,
        sens: &sens,
        hp: &hp,
        fuel_floor,
        future_mean: &future_mean,
    };

    let solution = challenge.grid_optimize(&|c, s| policy(c, s, &ctx))?;
    save_solution(&solution)?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
