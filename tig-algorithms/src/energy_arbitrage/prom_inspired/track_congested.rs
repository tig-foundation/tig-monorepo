// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Mutex, OnceLock};
use tig_challenges::energy_arbitrage::*;
use tig_challenges::energy_arbitrage::constants::{
    DELTA_T, EPS_FLOW, ETA_CHARGE, ETA_DISCHARGE, KAPPA_DEG, KAPPA_TX,
};

/// Generic numerical tolerance. The swarm ran against the challenge's
/// `EPS_BASELINE` (a baseline-solver tunable = 1e-12); declared locally here
/// so the submission doesn't depend on a repo-internal constant.
const EPS: f64 = 1e-12;

// `__fuel_remaining` is initialized by the runtime to the fuel cap and decremented
// by the fuel-instrumentation pass as the algorithm executes; it is exported from
// the built `.so`. We budget against it instead of wall-clock time so the solver's
// degrade-to-zeros fallback triggers deterministically regardless of how fast the
// grading machine runs the (instrumented) binary.
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
    /// Passes of the PTDF-aware coordinate polish.
    pub coord_polish_passes: usize,
    /// Day-ahead lookahead window (steps) used for the quantile threshold policy.
    pub lookahead_horizon: usize,
    /// Max fuel (runtime fuel units) the optimization rollout may spend before it
    /// falls back to zero actions for the remaining steps. 0 = spend all the fuel
    /// the runtime makes available (minus a small safety reserve). Always capped so
    /// it cannot trigger an out-of-fuel exit.
    pub fuel_budget: u64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            dp_soc_levels: 28,
            dp_action_levels: 7,
            policy_action_levels: 9,
            proj_max_iters: 80,
            grad_outer_iters: 120,
            grad_ls_iters: 6,
            coord_polish_passes: 7,
            lookahead_horizon: 16,
            fuel_budget: 0,
        }
    }
}

impl Hyperparameters {
    /// Parse from the optional JSON map, falling back to defaults for any missing
    /// field, then clamp the values that would otherwise be able to panic the solver.
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
        hp.lookahead_horizon = hp.lookahead_horizon.max(1);
        Ok(hp)
    }
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    challenge.network.compute_flows(&injections)
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    challenge.network.verify_flows(&flows).is_ok()
}



#[derive(Clone)]
struct RtHistory {
    num_nodes: usize,
    values: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
}

static RT_HISTORY: OnceLock<Mutex<RtHistory>> = OnceLock::new();

fn history_lock() -> &'static Mutex<RtHistory> {
    RT_HISTORY.get_or_init(|| {
        Mutex::new(RtHistory {
            num_nodes: 0,
            values: Vec::new(),
            residuals: Vec::new(),
        })
    })
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) * numerator) / denominator;
    sorted[idx]
}

struct BatteryDP {
    /// Lower bound of SOC grid (same for near and far).
    soc_lo: f64,
    /// Near‑term DP: steps 0 .. near_horizon-1, full resolution.
    near_soc_step_inv: f64,
    near_levels: usize,
    near_values: Vec<Vec<f64>>, // [near_horizon][near_levels]
    near_dvalues: Vec<Vec<f64>>, // [near_horizon][near_levels] monotone-clamped slopes
    /// Far‑future DP: steps near_horizon .. num_steps, coarser resolution.
    far_soc_step_inv: Option<f64>,
    far_levels: Option<usize>,
    far_values: Option<Vec<Vec<f64>>>, // [num_steps - near_horizon + 1][far_levels]
    far_dvalues: Option<Vec<Vec<f64>>>, // [num_steps - near_horizon + 1][far_levels]
    near_horizon: usize,
    num_steps: usize,
}

fn immediate_profit(battery: &Battery, action: f64, price: f64) -> f64 {
    let throughput = action.abs() * DELTA_T;
    action * price * DELTA_T
        - KAPPA_TX * throughput
        - KAPPA_DEG * (throughput / battery.capacity_mwh).powi(2)
}

/// Monotone‑clamped derivatives for cubic Hermite interpolation using
/// Fritsch‑Carlson method.  Returns a vector of slopes dy/dx at each grid
/// point, with grid step h.
fn compute_clamped_derivatives(values: &[f64], h: f64) -> Vec<f64> {
    let n = values.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut d = vec![0.0; n];
    // interior points
    for i in 1..(n - 1) {
        let m1 = (values[i] - values[i - 1]) / h;
        let m2 = (values[i + 1] - values[i]) / h;
        if m1 * m2 <= 0.0 {
            d[i] = 0.0;
        } else {
            let central = (values[i + 1] - values[i - 1]) / (2.0 * h);
            let limit = 3.0 * m1.abs().min(m2.abs());
            d[i] = central.clamp(-limit, limit);
        }
    }
    // endpoints: one-sided differences, no extra clamping
    d[0] = (values[1] - values[0]) / h;
    d[n - 1] = (values[n - 1] - values[n - 2]) / h;
    d
}

/// Cubic Hermite interpolation using pre‑computed monotone‑clamped derivatives.
/// `values` and `dvalues` are the function values and slopes at the grid points.
fn interp_value_cubic(
    values: &[f64],
    dvalues: &[f64],
    soc: f64,
    lo: f64,
    step_inv: f64,
    last: usize,
) -> f64 {
    if last == 0 {
        return values[0];
    }
    let pos = ((soc - lo) * step_inv).clamp(0.0, last as f64);
    let low = pos.floor() as usize;
    let high = (low + 1).min(last);
    if low == high {
        return values[low];
    }
    let h = 1.0 / step_inv; // grid step
    let x0 = lo + low as f64 * h;
    let t = ((soc - x0) / h).clamp(0.0, 1.0);
    let f0 = values[low];
    let f1 = values[high];
    let d0 = dvalues[low] * h;
    let d1 = dvalues[high] * h;

    // Hermite basis functions
    let h00 = 2.0 * t.powi(3) - 3.0 * t.powi(2) + 1.0;
    let h10 = t.powi(3) - 2.0 * t.powi(2) + t;
    let h01 = -2.0 * t.powi(3) + 3.0 * t.powi(2);
    let h11 = t.powi(3) - t.powi(2);

    h00 * f0 + h10 * d0 + h01 * f1 + h11 * d1
}

/// Gradient‑adapted action grid that uses the DP shadow value `dv` (≈ dV_{t+1}/dSOC)
/// to place action points close to the unconstrained DP‑augmented optimum, plus
/// a few fixed fractions of the full range to guarantee global coverage.
///
/// The optimum action u* solves  d(immediate_profit)/du  +  dv · dsoc/du  = 0,
/// which is derived analytically (piecewise for charge / discharge).
/// Points are centred on u* and on the bounds; the total number of returned
/// actions is at most `levels`.
fn adaptive_action_grid(
    battery: &Battery,
    price: f64,
    dv: f64,
    levels: usize,
) -> Vec<f64> {
    let cap2 = battery.capacity_mwh.powi(2).max(1e-9);
    let dt = DELTA_T;
    let power_charge = battery.power_charge_mw;
    let power_discharge = battery.power_discharge_mw;

    let min_a = -power_charge;
    let max_a = power_discharge;

    let denom = 2.0 * KAPPA_DEG * dt * dt;
    let norm = cap2 / denom;

    // Unconstrained discharge optimum (u ≥ 0).
    // DSOC/du for u>0  →  -DELTA_T / ETA_DISCHARGE
    let u_discharge = norm
        * (price * dt - KAPPA_TX * dt - dv * dt / ETA_DISCHARGE)
        .clamp(0.0, max_a);

    // Unconstrained charge optimum (u ≤ 0).
    // DSOC/du for u<0  →  -ETA_CHARGE * DELTA_T
    let u_charge = norm
        * (price * dt + KAPPA_TX * dt - dv * ETA_CHARGE * dt)
        .clamp(min_a, 0.0);

    let mut actions = Vec::with_capacity(levels);
    actions.push(min_a);
    actions.push(0.0);
    actions.push(max_a);

    if u_discharge > EPS {
        actions.push(u_discharge);
        let lo = u_discharge * 0.5;
        if lo > EPS {
            actions.push(lo);
        }
        let hi = (u_discharge + max_a) * 0.5;
        if hi > u_discharge + EPS {
            actions.push(hi);
        }
    }
    if u_charge < -EPS {
        actions.push(u_charge);
        let lo = u_charge * 0.5; // closer to zero
        if lo < -EPS {
            actions.push(lo);
        }
        let hi = (u_charge + min_a) * 0.5; // between min and opt
        if hi < u_charge - EPS {
            actions.push(hi);
        }
    }

    // Fill remaining slots with fixed fractions of the full range.
    if actions.len() < levels {
        for &frac in &[0.25, 0.5, 0.75] {
            actions.push(-frac * power_charge);
            actions.push(frac * power_discharge);
        }
    }

    actions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    actions.dedup_by(|a, b| (*a - *b).abs() < EPS);
    actions.truncate(levels);
    actions
}

/// Feasible signed action bounds `(u_min, u_max)` for one battery.
///
/// Mirrors `Battery::compute_action_bounds` (crate-private in `tig-challenges`)
/// using only the public `Battery` fields, so the algorithm stays self-contained.
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
    let near_horizon = hp.lookahead_horizon.max(1).min(num_steps);
    let far_steps = num_steps - near_horizon;

    let soc_lo = battery.soc_min_mwh;
    let soc_max = battery.soc_max_mwh;
    let span = (soc_max - soc_lo).max(1e-9);

    // Build far DP if needed.
    let far_soc_step_inv;
    let far_levels;
    let far_values;
    let far_dvalues;
    if far_steps > 0 {
        far_levels = (hp.dp_soc_levels / 2).max(4);
        let far_soc_step = span / (far_levels - 1) as f64;
        far_soc_step_inv = 1.0 / far_soc_step;

        let mut values = vec![vec![0.0; far_levels]; far_steps + 1];
        let mut dvalues = vec![vec![0.0; far_levels]; far_steps + 1];
        // The last row (terminal) is all zeros, derivatives are zero.
        dvalues[far_steps] = vec![0.0; far_levels];
        let far_last = far_levels - 1;

        let w_jump = p_jump.clamp(0.0, 1.0);
        let w_normal = (1.0 - w_jump).max(0.0);
        let w_low = 0.5 * w_normal;
        let w_high = 0.5 * w_normal;
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
        let w_jump_low = w_jump - w_jump_high;

        for i in (0..far_steps).rev() {
            let t = near_horizon + i;
            let da = da_at_node[t];
            let price_low = da * (1.0 - sigma);
            let price_high = da * (1.0 + sigma);
            let price_jump_low = da * (1.0 + jump_floor);
            let price_jump_high = da * (1.0 + jump_ceiling);

            for s_idx in 0..far_levels {
                let soc = soc_lo + far_soc_step * s_idx as f64;
                let (lo, hi) = compute_action_bounds(battery, soc);

                // dv computed with a temporary borrow of values[i+1]
                let dv = {
                    let next_vals = &values[i + 1];
                    let pos = ((soc - soc_lo) * far_soc_step_inv).clamp(0.0, far_last as f64);
                    let low = pos.floor() as usize;
                    let high = (low + 1).min(far_last);
                    (next_vals[high] - next_vals[low]) * far_soc_step_inv
                };
                let actions = adaptive_action_grid(
                    battery,
                    (price_low + price_high) * 0.5,
                    dv,
                    hp.dp_action_levels,
                );

                let mut best_low = f64::NEG_INFINITY;
                let mut best_high = f64::NEG_INFINITY;
                let mut best_jump_low = f64::NEG_INFINITY;
                let mut best_jump_high = f64::NEG_INFINITY;

                for &raw in &actions {
                    let action = raw.clamp(lo, hi);
                    let next_soc = battery.apply_action_to_soc(action, soc);
                    // Borrow values[i+1] and dvalues[i+1] on each iteration
                    let future = interp_value_cubic(
                        &values[i + 1],
                        &dvalues[i + 1],
                        next_soc,
                        soc_lo,
                        far_soc_step_inv,
                        far_last,
                    );

                    best_low = best_low.max(immediate_profit(battery, action, price_low) + future);
                    best_high = best_high.max(immediate_profit(battery, action, price_high) + future);
                    best_jump_low = best_jump_low.max(immediate_profit(battery, action, price_jump_low) + future);
                    best_jump_high = best_jump_high.max(immediate_profit(battery, action, price_jump_high) + future);
                }
                values[i][s_idx] = w_low * best_low
                    + w_high * best_high
                    + w_jump_low * best_jump_low
                    + w_jump_high * best_jump_high;
            }
            // Compute derivatives for this step.
            dvalues[i] = compute_clamped_derivatives(&values[i], far_soc_step);
        }
        far_values = Some(values);
        far_dvalues = Some(dvalues);
    } else {
        far_levels = 0;
        far_soc_step_inv = 0.0;
        far_values = None;
        far_dvalues = None;
    }

    // Build near DP (full resolution).
    let near_levels = hp.dp_soc_levels;
    let near_soc_step = span / (near_levels - 1) as f64;
    let near_soc_step_inv = 1.0 / near_soc_step;
    let near_last = near_levels - 1;

    let mut near_values = vec![vec![0.0; near_levels]; near_horizon];
    let mut near_dvalues = vec![vec![0.0; near_levels]; near_horizon];

    let w_jump = p_jump.clamp(0.0, 1.0);
    let w_normal = (1.0 - w_jump).max(0.0);
    let w_low = 0.5 * w_normal;
    let w_high = 0.5 * w_normal;
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
    let w_jump_low = w_jump - w_jump_high;

    for t in (0..near_horizon).rev() {
        let da = da_at_node[t];
        let price_low = da * (1.0 - sigma);
        let price_high = da * (1.0 + sigma);
        let price_jump_low = da * (1.0 + jump_floor);
        let price_jump_high = da * (1.0 + jump_ceiling);

        let next_t = t + 1;
        let future_is_near = next_t < near_horizon;

        for s_idx in 0..near_levels {
            let soc = soc_lo + near_soc_step * s_idx as f64;
            let (lo, hi) = compute_action_bounds(battery, soc);

            // Approximate dv from future value (near or far).
            let dv = if future_is_near {
                let next_vals = &near_values[next_t];
                let pos = ((soc - soc_lo) * near_soc_step_inv).clamp(0.0, near_last as f64);
                let low = pos.floor() as usize;
                let high = (low + 1).min(near_last);
                (next_vals[high] - next_vals[low]) * near_soc_step_inv
            } else if let Some(ref far) = far_values {
                let idx = next_t - near_horizon;
                let far_last = far_levels - 1;
                let vals = &far[idx];
                let pos = ((soc - soc_lo) * far_soc_step_inv).clamp(0.0, far_last as f64);
                let low = pos.floor() as usize;
                let high = (low + 1).min(far_last);
                (vals[high] - vals[low]) * far_soc_step_inv
            } else {
                0.0
            };

            let actions = adaptive_action_grid(
                battery,
                (price_low + price_high) * 0.5,
                dv,
                hp.dp_action_levels,
            );

            let mut best_low = f64::NEG_INFINITY;
            let mut best_high = f64::NEG_INFINITY;
            let mut best_jump_low = f64::NEG_INFINITY;
            let mut best_jump_high = f64::NEG_INFINITY;

            for &raw in &actions {
                let action = raw.clamp(lo, hi);
                let next_soc = battery.apply_action_to_soc(action, soc);
                let future = if future_is_near {
                    interp_value_cubic(
                        &near_values[next_t],
                        &near_dvalues[next_t],
                        next_soc,
                        soc_lo,
                        near_soc_step_inv,
                        near_last,
                    )
                } else if let Some(ref far) = far_values {
                    let idx = next_t - near_horizon;
                    let far_dvals = far_dvalues.as_ref().unwrap();
                    interp_value_cubic(
                        &far[idx],
                        &far_dvals[idx],
                        next_soc,
                        soc_lo,
                        far_soc_step_inv,
                        far_levels - 1,
                    )
                } else {
                    0.0
                };

                best_low = best_low.max(immediate_profit(battery, action, price_low) + future);
                best_high = best_high.max(immediate_profit(battery, action, price_high) + future);
                best_jump_low = best_jump_low.max(immediate_profit(battery, action, price_jump_low) + future);
                best_jump_high = best_jump_high.max(immediate_profit(battery, action, price_jump_high) + future);
            }
            near_values[t][s_idx] = w_low * best_low
                + w_high * best_high
                + w_jump_low * best_jump_low
                + w_jump_high * best_jump_high;
        }
        // Compute derivatives for this step.
        near_dvalues[t] = compute_clamped_derivatives(&near_values[t], near_soc_step);
    }

    BatteryDP {
        soc_lo,
        near_soc_step_inv,
        near_levels,
        near_values,
        near_dvalues,
        far_soc_step_inv: if far_steps > 0 { Some(far_soc_step_inv) } else { None },
        far_levels: if far_steps > 0 { Some(far_levels) } else { None },
        far_values,
        far_dvalues,
        near_horizon,
        num_steps,
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
    let next_t = (t + 1).min(dp.num_steps);
    let next_soc = battery.apply_action_to_soc(action, soc);
    let future = if next_t < dp.near_horizon {
        interp_value_cubic(
            &dp.near_values[next_t],
            &dp.near_dvalues[next_t],
            next_soc,
            dp.soc_lo,
            dp.near_soc_step_inv,
            dp.near_levels - 1,
        )
    } else if let Some(ref far) = dp.far_values {
        let idx = next_t - dp.near_horizon;
        let far_dvals = dp.far_dvalues.as_ref().unwrap();
        interp_value_cubic(
            &far[idx],
            &far_dvals[idx],
            next_soc,
            dp.soc_lo,
            dp.far_soc_step_inv.unwrap(),
            dp.far_levels.unwrap() - 1,
        )
    } else {
        0.0
    };
    immediate_profit(battery, action, price) + future
}

fn dv_dsoc(dp: &BatteryDP, t: usize, soc: f64) -> f64 {
    let next_t = (t + 1).min(dp.num_steps);
    if next_t < dp.near_horizon {
        let values = &dp.near_values[next_t];
        let last = dp.near_levels - 1;
        if last == 0 {
            return 0.0;
        }
        let pos = ((soc - dp.soc_lo) * dp.near_soc_step_inv).clamp(0.0, last as f64);
        let mut low = pos.floor() as usize;
        if low >= last {
            low = last - 1;
        }
        (values[low + 1] - values[low]) * dp.near_soc_step_inv
    } else if let Some(ref far) = dp.far_values {
        let idx = next_t - dp.near_horizon;
        let last = dp.far_levels.unwrap() - 1;
        if last == 0 {
            return 0.0;
        }
        let values = &far[idx];
        let far_soc_step_inv = dp.far_soc_step_inv.unwrap();
        let pos = ((soc - dp.soc_lo) * far_soc_step_inv).clamp(0.0, last as f64);
        let mut low = pos.floor() as usize;
        if low >= last {
            low = last - 1;
        }
        (values[low + 1] - values[low]) * far_soc_step_inv
    } else {
        0.0
    }
}

/// Future-only DP value (ignores immediate profit).
fn dp_future_value(
    dp: &BatteryDP,
    battery: &Battery,
    t: usize,
    soc: f64,
    action: f64,
) -> f64 {
    let next_t = (t + 1).min(dp.num_steps);
    let next_soc = battery.apply_action_to_soc(action, soc);
    if next_t < dp.near_horizon {
        interp_value_cubic(
            &dp.near_values[next_t],
            &dp.near_dvalues[next_t],
            next_soc,
            dp.soc_lo,
            dp.near_soc_step_inv,
            dp.near_levels - 1,
        )
    } else if let Some(ref far) = dp.far_values {
        let idx = next_t - dp.near_horizon;
        let far_dvals = dp.far_dvalues.as_ref().unwrap();
        interp_value_cubic(
            &far[idx],
            &far_dvals[idx],
            next_soc,
            dp.soc_lo,
            dp.far_soc_step_inv.unwrap(),
            dp.far_levels.unwrap() - 1,
        )
    } else {
        0.0
    }
}

/// Pick top‑2 distinct actions that maximize future DP value (one‑step lookahead),
/// using price only for the adaptive action grid.  The returned Vec contains the
/// best action followed by the best action that differs from it by more than EPS,
/// providing seed diversity for the joint gradient‑ascent step at negligible extra
/// fuel cost (one extra GA run per step only when the second seed differs).
fn pick_dp_action_future_only(
    dp: &BatteryDP,
    battery: &Battery,
    t: usize,
    soc: f64,
    price: f64,
    bounds: (f64, f64),
    hp: &Hyperparameters,
) -> Vec<f64> {
    let (lo, hi) = bounds;

    // Collect all candidate actions with their future values.
    let mut candidates: Vec<(f64, f64)> = Vec::new();

    let dv = dv_dsoc(dp, t, soc);
    for raw in adaptive_action_grid(battery, price, dv, hp.policy_action_levels) {
        let action = raw.clamp(lo, hi);
        let value = dp_future_value(dp, battery, t, soc, action);
        candidates.push((action, value));
    }
    for action in [lo, hi] {
        let value = dp_future_value(dp, battery, t, soc, action);
        candidates.push((action, value));
    }

    // Sort by value descending, then pick top 2 distinct actions.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result: Vec<f64> = Vec::with_capacity(2);
    for (action, _value) in candidates {
        if result.is_empty() || (action - result[0]).abs() > EPS {
            result.push(action);
            if result.len() >= 2 {
                break;
            }
        }
    }
    // Fallback: at least one action (clamped zero).
    if result.is_empty() {
        result.push(0.0_f64.clamp(lo, hi));
    }
    result
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

/// Alternating projection onto box bounds and the most-violated halfspace.
/// Returns true if all constraints are satisfied within tolerance.
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
        // Project onto box.
        for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
            if *a < lo { *a = lo; }
            if *a > hi { *a = hi; }
        }
        // Find worst violation.
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
        // Project onto the halfspace worst_sign * (base + sens·u) <= limit.
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
    // Final clamp.
    for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
        if *a < lo { *a = lo; }
        if *a > hi { *a = hi; }
    }
    // Verify.
    for l in 0..n_lines {
        let f = line_flow(&sens[l], action, base_flows[l]);
        if f.abs() > limits[l] * (1.0 + EPS_FLOW) + 1e-6 {
            return false;
        }
    }
    true
}

/// Attempt to restore feasibility by adjusting one battery at a time
/// using flow sensitivity to stay within line limits.
fn restore_feasibility(
    challenge: &Challenge,
    state: &State,
    action: &mut [f64],
    sens: &[Vec<f64>],
) -> bool {
    let mut current_flows = compute_flows(challenge, state, action);
    const MAX_SWEEPS: usize = 3;
    for _ in 0..MAX_SWEEPS {
        let mut any_change = false;
        for b in 0..action.len() {
            let (lo, hi) = state.action_bounds[b];
            let cur = action[b];
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
            let u_low = net_lo.max(lo);
            let u_high = net_hi.min(hi);
            let new_u = if u_low <= u_high + EPS {
                (u_low + u_high) * 0.5
            } else {
                cur
            };
            let new_u = new_u.clamp(lo, hi);
            if (new_u - cur).abs() > EPS {
                let delta = new_u - cur;
                action[b] = new_u;
                for l in 0..challenge.network.num_lines {
                    current_flows[l] += sens[l][b] * delta;
                }
                any_change = true;
            }
        }
        if !any_change {
            break;
        }
    }
    is_flow_feasible(challenge, state, action)
}

/// Project + Gauss-Seidel feasibility sweep (instead of expensive bisection).
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
    // Gauss-Seidel style feasibility restoration.
    if !restore_feasibility(challenge, state, action, sens) {
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

/// Analytic gradient of immediate profit + DP shadow value w.r.t. u_b.
/// When the next SOC hits a bound (dsoc_du == 0), the analytic gradient's dv term
/// vanishes, which can give a misleading zero gradient despite actionable improvements.
/// In that case we compute the gradient using a forward finite‑difference of the
/// DP‑action value function, capturing the full effect near bounds. The perturbation
/// step is 0.1% of max_power, clamped to the battery's action bounds.
fn analytic_gradient(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    action: &[f64],
) -> Vec<f64> {
    let mut grad = vec![0.0_f64; action.len()];
    // Compute max absolute power across all batteries for finite‑difference step.
    let max_power: f64 = challenge
        .batteries
        .iter()
        .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
        .fold(1.0_f64, f64::max);
    let epsilon = (max_power * 0.001).max(EPS);

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
            // Subgradient at u=0: pick direction with steeper slope average.
            -0.5 * (DELTA_T / ETA_DISCHARGE + ETA_CHARGE * DELTA_T)
        };

        if dsoc_du.abs() < EPS {
            // Near SOC bound: analytic dv term unreliable → use finite difference.
            let dp = &dps[b];
            let soc = state.socs[b];
            let t = state.time_step;
            let (lo, hi) = state.action_bounds[b]; // (u_min, u_max) – note: lo ≤ 0, hi ≥ 0
            // Clamp trial action to bounds to avoid out‑of‑bounds evaluation.
            let u_trial = (u + epsilon).clamp(lo, hi);
            let value_current = dp_action_value(dp, battery, t, soc, price, u);
            let value_trial  = dp_action_value(dp, battery, t, soc, price, u_trial);
            grad[b] = (value_trial - value_current) / (u_trial - u).max(EPS);
        } else {
            let dv = dv_dsoc(&dps[b], state.time_step, next_soc);
            grad[b] = imm + dv * dsoc_du;
        }
    }
    grad
}

/// Projected gradient ascent with momentum and a single adaptive step per iteration.
/// The inner backtracking line search is replaced by a single trial with an
/// aggressively adapting learning rate, which saves fuel while retaining convergence
/// quality through the outer iteration count.
///
/// The momentum factor β is adapted each iteration based on the alignment
/// between the previous momentum direction and the current gradient (cosθ).
/// When they are aligned (cosθ > 0) β is increased (more momentum) to accelerate
/// in smooth regions; when they oppose (cosθ < 0) β is decreased to reduce
/// oscillations.  β is clamped to [0.5, 0.95] and adjusted multiplicatively
/// by (1 + α·cosθ) with α = 0.1, keeping fuel cost negligible.
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
    let mut momentum = vec![0.0_f64; action.len()];
    let mut beta = 0.8;
    let alpha_adapt = 0.1;
    for iter in 0..hp.grad_outer_iters {
        let grad = analytic_gradient(challenge, state, dps, &action);
        // Adapt β based on alignment between previous momentum and current gradient.
        if iter > 0 {
            let prev_norm: f64 = momentum.iter().map(|v| v * v).sum::<f64>().sqrt();
            let grad_norm: f64 = grad.iter().map(|v| v * v).sum::<f64>().sqrt();
            if prev_norm > EPS && grad_norm > EPS {
                let dot: f64 = momentum.iter().zip(grad.iter()).map(|(m, g)| m * g).sum();
                let cos_theta = (dot / (prev_norm * grad_norm)).clamp(-1.0, 1.0);
                beta = (beta * (1.0 + alpha_adapt * cos_theta)).clamp(0.5, 0.95);
            }
        }
        for i in 0..grad.len() {
            momentum[i] = beta * momentum[i] + (1.0 - beta) * grad[i];
        }
        let g_norm: f64 = momentum.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm < 1e-9 {
            break;
        }

        let step_scale = lr / g_norm;
        let mut trial: Vec<f64> = action
            .iter()
            .zip(momentum.iter())
            .map(|(a, m)| a + step_scale * m)
            .collect();
        safe_project_to_feasible(challenge, state, &mut trial, sens, base_flows, hp);
        let v = total_step_value(challenge, state, dps, &trial);
        if v > best_value + 1e-9 {
            action = trial.clone();
            best_value = v;
            best_action = trial;
            lr *= 1.4;   // increase step size
        } else {
            lr *= 0.5;   // decrease step size
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

/// Single‑battery coordinate polish with golden‑section search and
/// early termination when the total improvement of a pass falls below
/// a scale‑dependent threshold, saving fuel for later steps.
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

    // Per‑battery evaluation closure – only depends on the battery's own action.
    let compute_battery_value = |b: usize, u: f64| -> f64 {
        let bat = &challenge.batteries[b];
        dp_action_value(
            &dps[b],
            bat,
            state.time_step,
            state.socs[b],
            state.rt_prices[bat.node],
            u,
        )
    };

    // Scale‑dependent early‑exit threshold.
    let max_power: f64 = challenge
        .batteries
        .iter()
        .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
        .fold(1.0_f64, f64::max);
    let improvement_threshold = 1e-12 * (challenge.num_batteries as f64) * max_power;

    let mut improved_any_pass;
    for _ in 0..hp.coord_polish_passes {
        improved_any_pass = false;
        let mut pass_improvement = 0.0;
        let mut current_flows = compute_flows(challenge, state, &action);

        for bi in 0..challenge.num_batteries {
            let (lo, hi) = state.action_bounds[bi];
            let cur = action[bi];

            // Feasible interval [net_lo, net_hi] derived from current flows.
            let mut net_lo = lo;
            let mut net_hi = hi;
            for l in 0..challenge.network.num_lines {
                let coeff = sens[l][bi];
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

            let u_low = net_lo.max(lo);
            let u_high = net_hi.min(hi);
            if u_high - u_low < EPS {
                continue;
            }

            // Quadratic interpolation search within [u_low, u_high] using three points.
            let mid = (u_low + u_high) * 0.5;
            let f_low = compute_battery_value(bi, u_low);
            let f_mid = compute_battery_value(bi, mid);
            let f_high = compute_battery_value(bi, u_high);

            // Determine the best among the evaluated points and, if the parabola is
            // concave down, its vertex.
            let mut best_u = u_low;
            let mut best_f = f_low;
            if f_mid > best_f {
                best_f = f_mid;
                best_u = mid;
            }
            if f_high > best_f {
                best_f = f_high;
                best_u = u_high;
            }

            // Fit a quadratic to the three points and, if it opens downward, evaluate
            // the vertex within the interval.
            {
                let x1 = u_low;
                let x2 = mid;
                let x3 = u_high;
                let y1 = f_low;
                let y2 = f_mid;
                let y3 = f_high;
                let num = x1.powi(2) * (y2 - y3) + x2.powi(2) * (y3 - y1) + x3.powi(2) * (y1 - y2);
                let den = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
                if den.abs() > EPS {
                    // Estimate a (coefficient of x^2) via the second divided difference.
                    let h1 = x2 - x1;
                    let h2 = x3 - x2;
                    let a_est = ((y2 - y1) / h1 - (y3 - y2) / h2) / (x3 - x1);
                    // a = -a_est; parabola opens down when a < 0  => a_est > 0.
                    if a_est > EPS {
                        let vertex_x = (num / den).clamp(u_low, u_high);
                        let f_vertex = compute_battery_value(bi, vertex_x);
                        if f_vertex > best_f {
                            best_u = vertex_x;
                        }
                    }
                }
            }

            // Clamp to the interval to avoid floating‑point slip.
            best_u = best_u.clamp(u_low, u_high);

            let cur_val = compute_battery_value(bi, cur);
            let new_val = compute_battery_value(bi, best_u);
            if new_val > cur_val + 1e-12 {
                let delta = best_u - cur;
                action[bi] = best_u;
                // Update flows incrementally to keep subsequent batteries’ intervals correct.
                for l in 0..challenge.network.num_lines {
                    current_flows[l] += sens[l][bi] * delta;
                }
                pass_improvement += delta.abs();
                improved_any_pass = true;
            }
        }

        if !improved_any_pass {
            break;
        }
        if pass_improvement < improvement_threshold {
            break;
        }
    }

    action
}

fn policy(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_steps = challenge.num_steps;
    let n_remaining = n_steps.saturating_sub(t);
    if n_remaining == 0 {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    let friction = 2.0 * KAPPA_TX;
    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;

    let mut history = history_lock().lock().unwrap();
    if state.time_step == 0 || history.num_nodes != challenge.network.num_nodes {
        history.num_nodes = challenge.network.num_nodes;
        history.values = vec![Vec::new(); challenge.network.num_nodes];
        history.residuals = vec![Vec::new(); challenge.network.num_nodes];
    }
    let mut rt_bands = vec![None; challenge.network.num_nodes];
    for node in 0..challenge.network.num_nodes {
        if history.values[node].len() >= 16 {
            let mut sorted = history.values[node].clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q15 = percentile(&sorted, 15, 100);
            let q85 = percentile(&sorted, 85, 100);
            if q85 - q15 > 2.0 {
                rt_bands[node] = Some((q15, q85));
            }
        }
    }
    // Residual shift is no longer computed; residuals are kept for future use.

    // Independent per-battery DP-preferred seeds: collect top‑2 distinct
    // actions per battery to create two alternative seeds for GA, increasing
    // the chance of discovering a better combined action.
    let mut dp_seed_first = Vec::with_capacity(challenge.num_batteries);
    let mut dp_seed_second = Vec::with_capacity(challenge.num_batteries);
    let mut has_second = false;
    for b in 0..challenge.num_batteries {
        let battery = &challenge.batteries[b];
        let top_actions = pick_dp_action_future_only(
            &dps[b],
            battery,
            t,
            state.socs[b],
            state.rt_prices[battery.node],
            state.action_bounds[b],
            hp,
        );
        dp_seed_first.push(top_actions[0]);
        let second = if top_actions.len() > 1 { top_actions[1] } else { top_actions[0] };
        dp_seed_second.push(second);
        if (second - top_actions[0]).abs() > EPS {
            has_second = true;
        }
    }

    // Baseline flows depend on this step's exogenous injections.
    let zero = vec![0.0_f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero);

    // Build list of seeds for joint optimisation.
    // Include the second‑best DP seed only when it differs from the first across
    // at least one battery, keeping fuel cost minimal.
    let mut seeds = vec![dp_seed_first.clone(), zero.clone()];
    if has_second {
        seeds.push(dp_seed_second.clone());
    }

    // Lightweight spike/dip detection using historical RT bands.
    let mut spike_actions: Vec<f64> = vec![0.0; challenge.num_batteries];
    let mut has_spike = false;
    for (b, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let current_price = state.rt_prices[node];
        let (u_min, u_max) = state.action_bounds[b];
        if let Some((rt_low, rt_high)) = rt_bands[node] {
            if u_max > 0.0 && current_price > rt_high + friction {
                spike_actions[b] = u_max;
                has_spike = true;
            } else if u_min < 0.0 && current_price < rt_low * eta_rt - friction {
                spike_actions[b] = u_min;
                has_spike = true;
            }
        }
    }
    if has_spike {
        seeds.push(spike_actions);
    }

    // Joint projected gradient ascent over the seeds.
    let mut result = joint_optimize_step(challenge, state, dps, sens, &base_flows, seeds, hp);
    result = coordinate_polish_step(challenge, state, dps, sens, result, hp);

    // Record the observed RT prices and residuals into history.
    for node in 0..challenge.network.num_nodes {
        history.values[node].push(state.rt_prices[node]);
        history.residuals[node]
            .push(state.rt_prices[node] - challenge.market.day_ahead_prices[t][node]);
    }
    drop(history);

    if !is_flow_feasible(challenge, state, &result) {
        // Final safety net: zeros are guaranteed feasible.
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

    // Initial feasible solution: all zeros.
    let zero_solution = Solution {
        schedule: vec![vec![0.0; challenge.num_batteries]; challenge.num_steps],
    };
    save_solution(&zero_solution)?;

    // Decide how much fuel the optimization rollout may spend. Always reserve ~1/28
    // of the fuel left after setup so the rollout can finish the cheap zero-action
    // tail and save a valid solution (never an out-of-fuel exit). `fuel_budget == 0`
    // spends all available fuel minus that reserve; a positive value caps the spend
    // lower so fuel can be traded against quality. Budgeting off fuel (not wall time)
    // keeps the degrade-to-zeros fallback deterministic across grading machines.
    let available = fuel_remaining();
    let reserve = available / 28;
    let max_spend = available.saturating_sub(reserve);
    let target_spend = if hp.fuel_budget == 0 {
        max_spend
    } else {
        hp.fuel_budget.min(max_spend)
    };
    let fuel_floor = available - target_spend;
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        policy(c, s, &dps, &sens, &hp)
    })?;
    save_solution(&solution)?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected