use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};
use tig_challenges::energy_arbitrage::*;
use tig_challenges::energy_arbitrage::constants::{
    DELTA_T, EPS_FLOW, ETA_CHARGE, ETA_DISCHARGE, KAPPA_DEG, KAPPA_TX,
};

const EPS: f64 = 1e-12;

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
    pub dp_soc_levels: usize,
    pub dp_action_levels: usize,
    pub policy_action_levels: usize,
    pub proj_max_iters: usize,
    pub grad_outer_iters: usize,
    pub grad_ls_iters: usize,
    pub bisect_iters: usize,
    pub coord_polish_passes: usize,
    pub lookahead_horizon: usize,
    pub fuel_budget: u64,
    pub num_seeds: usize,
    pub use_momentum: bool,
    #[serde(default)]
    pub use_cosine_beta: bool,
    pub use_pair_polish: bool,
    #[serde(default)]
    pub anticipate_lmp: bool,
    #[serde(default)]
    pub use_joint_pair_polish: bool,
    #[serde(default = "default_joint_pair_budget")]
    pub joint_pair_budget: usize,
    #[serde(default)]
    pub use_lp_dispatch: bool,
    #[serde(default)]
    pub lp_max_lines: usize,
    #[serde(default)]
    pub lp_pivot_budget: usize,
    #[serde(default)]
    pub use_dual_dispatch: bool,
    #[serde(default = "default_admm_iters")]
    pub max_admm_iters: usize,
    #[serde(default = "default_admm_rho")]
    pub admm_rho: f64,
    #[serde(default)]
    pub use_mpc_lookahead: bool,
    #[serde(default = "default_mpc_horizon")]
    pub mpc_horizon: usize,
    #[serde(default = "default_mpc_n_cand")]
    pub mpc_n_cand: usize,
    #[serde(default)]
    pub mpc_pivot_threshold: f64,
    #[serde(default)]
    pub mpc_use_rt_gate: bool,
    #[serde(default)]
    pub use_sqdp: bool,
    #[serde(default)]
    pub use_coupling_cut: bool,
    #[serde(default)]
    pub use_aggregate_reg: bool,
    #[serde(default)]
    pub agg_reg_lambda: f64,
    #[serde(default)]
    pub use_ptdf_ct: bool,
    #[serde(default = "default_ct_step_eta")]
    pub ct_step_eta: f64,
    #[serde(default)]
    pub ct_ref_kappa: f64,
    #[serde(default)]
    pub ct_gdd_alpha: f64,
    #[serde(default)]
    pub use_dp_value_shift: bool,
    #[serde(default)]
    pub use_composite_wv: bool,
    #[serde(default = "default_cwv_lambda")]
    pub cwv_lambda: f64,
    #[serde(default = "default_cwv_agg_levels")]
    pub cwv_agg_levels: usize,
    #[serde(default = "default_cwv_clusters")]
    pub cwv_clusters: usize,
}

const MOMENTUM_BETA: f64 = 0.999;
const BETA_END: f64 = 0.7;
const PAIR_POLISH_ALPHA: f64 = 0.125;
const PAIR_POLISH_BUDGET: usize = 64;
const LMP_THRESHOLD: f64 = 0.5;
const LMP_PREMIUM_SCALE: f64 = 2.0;

fn default_joint_pair_budget() -> usize {
    64
}

fn default_admm_iters() -> usize {
    6
}

fn default_admm_rho() -> f64 {
    0.2
}

fn default_mpc_horizon() -> usize {
    12
}

fn default_mpc_n_cand() -> usize {
    5
}

fn default_ct_step_eta() -> f64 {
    0.25
}

fn default_cwv_lambda() -> f64 {
    0.25
}

fn default_cwv_agg_levels() -> usize {
    65
}

fn default_cwv_clusters() -> usize {
    1
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            dp_soc_levels: 33,
            dp_action_levels: 17,
            policy_action_levels: 65,
            proj_max_iters: 80,
            grad_outer_iters: 25,
            grad_ls_iters: 6,
            bisect_iters: 30,
            coord_polish_passes: 1,
            lookahead_horizon: 24,
            fuel_budget: 0,
            num_seeds: 3,
            use_momentum: false,
            use_cosine_beta: false,
            use_pair_polish: false,
            anticipate_lmp: false,
            use_joint_pair_polish: false,
            joint_pair_budget: 64,
            use_lp_dispatch: false,
            lp_max_lines: 0,
            lp_pivot_budget: 0,
            use_dual_dispatch: false,
            max_admm_iters: 6,
            admm_rho: 0.2,
            use_mpc_lookahead: false,
            mpc_horizon: 12,
            mpc_n_cand: 5,
            mpc_pivot_threshold: 0.0,
            mpc_use_rt_gate: false,
            use_sqdp: false,
            use_coupling_cut: false,
            use_aggregate_reg: false,
            agg_reg_lambda: 0.0,
            use_ptdf_ct: false,
            ct_step_eta: 0.25,
            ct_ref_kappa: 0.0,
            ct_gdd_alpha: 0.0,
            use_dp_value_shift: false,
            use_composite_wv: false,
            cwv_lambda: 0.25,
            cwv_agg_levels: 65,
            cwv_clusters: 1,
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
        hp.num_seeds = hp.num_seeds.max(1);
        hp.max_admm_iters = hp.max_admm_iters.max(1);
        hp.admm_rho = hp.admm_rho.max(1e-6);
        hp.mpc_n_cand = hp.mpc_n_cand.max(2);
        hp.mpc_horizon = hp.mpc_horizon.max(1);
        hp.cwv_agg_levels = hp.cwv_agg_levels.max(2);
        hp.cwv_clusters = hp.cwv_clusters.max(1);
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







fn iter_pool() -> &'static Mutex<i64> {
    static POOL: OnceLock<Mutex<i64>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(0))
}

#[inline(always)]
fn iter_pool_reset() {
    *iter_pool().lock().unwrap() = 0;
}

#[inline(always)]
fn iter_pool_claim(max: i64) -> i64 {
    let mut g = iter_pool().lock().unwrap();
    let take = (*g).min(max).max(0);
    *g -= take;
    take
}

#[inline(always)]
fn iter_pool_donate(savings: i64) {
    if savings > 0 {
        *iter_pool().lock().unwrap() += savings;
    }
}

fn prev_policy_store() -> &'static Mutex<Option<(Vec<f64>, Vec<f64>)>> {
    static STORE: OnceLock<Mutex<Option<(Vec<f64>, Vec<f64>)>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(None))
}



#[derive(Clone)]
struct BatteryDP {
    soc_lo: f64,
    soc_step_inv: f64,
    levels: usize,
    values: Vec<Vec<f64>>,
    use_shift: bool,
    pub cluster_grid: Option<Vec<f64>>,
}

#[inline(always)]
fn dp_eval_future(dp: &BatteryDP, t_next: usize, soc: f64) -> f64 {
    let t = t_next.min(dp.values.len() - 1);
    if let Some(ref grid) = dp.cluster_grid {
        // non-uniform grid
        interp_nonuniform(&dp.values[t], grid, soc)
    } else if dp.levels == 0 {
        dp.values[t][0] + dp.values[t][1] * soc + dp.values[t][2] * soc * soc
    } else if dp.use_shift {
        interp_value_q(&dp.values[t], soc, dp.soc_lo, dp.soc_step_inv, dp.levels - 1)
    } else {
        interp_value(&dp.values[t], soc, dp.soc_lo, dp.soc_step_inv, dp.levels - 1)
    }
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

#[inline(always)]
fn interp_value_q(values: &[f64], soc: f64, lo: f64, step_inv: f64, last: usize) -> f64 {
    let pos = ((soc - lo) * step_inv).clamp(0.0, last as f64);
    let low = pos.floor() as usize;
    let high = (low + 1).min(last);
    let alpha = pos - low as f64;
    let linear = values[low] * (1.0 - alpha) + values[high] * alpha;
    if high < last {
        let d2v = values[high + 1] - 2.0 * values[high] + values[low];
        linear + alpha * (alpha - 1.0) * 0.5 * d2v
    } else {
        linear
    }
}

#[inline(always)]
fn interp_nonuniform(values: &[f64], grid: &[f64], soc: f64) -> f64 {
    let n = grid.len();
    if n == 0 { return 0.0; }
    if soc <= grid[0] { return values[0]; }
    if soc >= grid[n-1] { return values[n-1]; }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        if grid[mid] <= soc {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let x0 = grid[lo];
    let x1 = grid[hi];
    let alpha = (soc - x0) / (x1 - x0);
    values[lo] * (1.0 - alpha) + values[hi] * alpha
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

    if in_discharge_region {
        let discharge_levels = (levels as f64 * 0.6).round() as usize;
        for i in 1..discharge_levels {
            let frac = i as f64 / (discharge_levels as f64);
            discharge_points.push(frac * base_discharge);
        }
    }

    if in_charge_region {
        let charge_levels = (levels as f64 * 0.6).round() as usize;
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

fn build_battery_sqdp(
    battery: &Battery,
    da_at_node: &[f64],
    num_steps: usize,
    sigma: f64,
    p_jump: f64,
    mean_pareto: f64,
    second_pareto: f64,
) -> BatteryDP {
    let soc_lo_b = battery.soc_min_mwh;
    let soc_hi_b = battery.soc_max_mwh;
    let soc_mid = 0.5 * (soc_lo_b + soc_hi_b);
    let half_span = (soc_hi_b - soc_lo_b) * 0.5;

    const S_NORM: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let soc_samples: [f64; 5] = [
        soc_mid + S_NORM[0] * half_span,
        soc_mid + S_NORM[1] * half_span,
        soc_mid + S_NORM[2] * half_span,
        soc_mid + S_NORM[3] * half_span,
        soc_mid + S_NORM[4] * half_span,
    ];

    let dt = DELTA_T;
    let eta_c = ETA_CHARGE;
    let eta_d = ETA_DISCHARGE;
    let cap2 = (battery.capacity_mwh * battery.capacity_mwh).max(1e-9);

    let w_jump = p_jump.clamp(0.0, 1.0);
    let w_normal = (1.0 - w_jump).max(0.0);
    let w_low_p = 0.5 * w_normal;
    let w_high_p = 0.5 * w_normal;
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
    } else { 0.0 };
    let w_jump_low = w_jump - w_jump_high;

    let mut values: Vec<Vec<f64>> = vec![vec![0.0_f64; 3]; num_steps + 1];

    for t in (0..num_steps).rev() {
        let da = da_at_node[t];
        let price_low = da * (1.0 - sigma);
        let price_high = da * (1.0 + sigma);
        let price_jump_low = da * (1.0 + jump_floor);
        let price_jump_high = da * (1.0 + jump_ceiling);

        let prices = [price_low, price_high, price_jump_low, price_jump_high];
        let weights = [w_low_p, w_high_p, w_jump_low, w_jump_high];

        let alpha_f = values[t + 1][0];
        let beta_f = values[t + 1][1];
        let gamma_f = values[t + 1][2];

        let c2 = dt * dt * (gamma_f / (eta_d * eta_d) - KAPPA_DEG / cap2);
        let d2 = dt * dt * (gamma_f * eta_c * eta_c - KAPPA_DEG / cap2);

        let mut v_samples = [0.0_f64; 5];
        for k in 0..5 {
            let soc = soc_samples[k];
            let (lo, hi) = compute_action_bounds(battery, soc);
            let mut v_total = 0.0_f64;

            for pi in 0..4 {
                let weight = weights[pi];
                if weight < 1e-12 { continue; }
                let price = prices[pi];

                let mut best = f64::NEG_INFINITY;

                {
                    let sn = battery.apply_action_to_soc(0.0, soc);
                    let v = alpha_f + beta_f * sn + gamma_f * sn * sn;
                    if v > best { best = v; }
                }

                if hi > 1e-9 {
                    let c1 = dt * (price - KAPPA_TX - (beta_f + 2.0 * gamma_f * soc) / eta_d);
                    let a_opt = if c2 < -1e-12 {
                        (-c1 / (2.0 * c2)).clamp(0.0_f64.max(lo), hi)
                    } else {
                        if c1 > 0.0 { hi } else { 0.0_f64.max(lo) }
                    };
                    for &a in &[a_opt, hi] {
                        let sn = battery.apply_action_to_soc(a, soc);
                        let v = immediate_profit(battery, a, price)
                            + alpha_f + beta_f * sn + gamma_f * sn * sn;
                        if v > best { best = v; }
                    }
                }

                if lo < -1e-9 {
                    let d1 = dt * (price + KAPPA_TX - eta_c * (beta_f + 2.0 * gamma_f * soc));
                    let a_opt = if d2 < -1e-12 {
                        (-d1 / (2.0 * d2)).clamp(lo, 0.0_f64.min(hi))
                    } else {
                        if d1 < 0.0 { lo } else { 0.0_f64.min(hi) }
                    };
                    for &a in &[a_opt, lo] {
                        let sn = battery.apply_action_to_soc(a, soc);
                        let v = immediate_profit(battery, a, price)
                            + alpha_f + beta_f * sn + gamma_f * sn * sn;
                        if v > best { best = v; }
                    }
                }

                v_total += weight * best;
            }
            v_samples[k] = v_total;
        }

        let sv: f64  = v_samples.iter().sum();
        let ssv: f64 = S_NORM.iter().zip(v_samples.iter()).map(|(&s, &v)| s * v).sum();
        let s2v: f64 = S_NORM.iter().zip(v_samples.iter()).map(|(&s, &v)| s * s * v).sum();

        let gamma_n = (5.0 * s2v - 2.5 * sv) / 4.375;
        let beta_n  = ssv / 2.5;
        let alpha_n = (sv - 2.5 * gamma_n) / 5.0;

        let hs = half_span;
        if hs < 1e-9 {
            values[t] = vec![v_samples[2], 0.0, 0.0];
        } else {
            let hs2 = hs * hs;
            let gamma_p = gamma_n / hs2;
            let beta_p  = beta_n / hs - 2.0 * gamma_p * soc_mid;
            let alpha_p = alpha_n - beta_n * soc_mid / hs + gamma_n * soc_mid * soc_mid / hs2;
            values[t] = vec![alpha_p, beta_p, gamma_p];
        }
    }

    BatteryDP { soc_lo: 0.0, soc_step_inv: 0.0, levels: 0, values, use_shift: false, cluster_grid: None }
}

fn build_battery_dp(
    battery: &Battery,
    da_at_node: &[f64],
    num_steps: usize,
    sigma: f64,
    p_jump: f64,
    mean_pareto: f64,
    second_pareto: f64,
    fleet_soc_norm: f64,
    hp: &Hyperparameters,
) -> BatteryDP {
    if hp.use_sqdp {
        return build_battery_sqdp(
            battery, da_at_node, num_steps, sigma, p_jump, mean_pareto, second_pareto,
        );
    }
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

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let friction = 2.0 * KAPPA_TX;

    let dt = DELTA_T;
    let cap2 = (battery.capacity_mwh * battery.capacity_mwh).max(1e-9);
    let tx_coeff = KAPPA_TX * dt;
    let deg_coeff = KAPPA_DEG * dt * dt / cap2;

    let mut best_low_vec = vec![0.0; levels];
    let mut best_high_vec = vec![0.0; levels];
    let mut best_jump_low_vec = vec![0.0; levels];
    let mut best_jump_high_vec = vec![0.0; levels];

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

        let actions = adaptive_action_grid(
            battery,
            charge_max,
            discharge_min,
            (price_low + price_high) * 0.5,
            hp.dp_action_levels,
        );

        best_low_vec.fill(f64::NEG_INFINITY);
        best_high_vec.fill(f64::NEG_INFINITY);
        best_jump_low_vec.fill(f64::NEG_INFINITY);
        best_jump_high_vec.fill(f64::NEG_INFINITY);

        let p_low_dt = price_low * dt;
        let p_high_dt = price_high * dt;
        let p_j_low_dt = price_jump_low * dt;
        let p_j_high_dt = price_jump_high * dt;

        for &raw in &actions {
            for s_idx in 0..levels {
                let (lo, hi) = bounds[s_idx];
                let a = raw.clamp(lo, hi);
                let a_abs = a.abs();
                let tx = tx_coeff * a_abs;
                let deg = deg_coeff * a_abs * a_abs;
                let soc = soc_lo + soc_step * s_idx as f64;
                let next_soc = battery.apply_action_to_soc(a, soc);
                let future_val = if hp.use_dp_value_shift {
                    interp_value_q(next, next_soc, soc_lo, soc_step_inv, last)
                } else {
                    interp_value(next, next_soc, soc_lo, soc_step_inv, last)
                };
                let base_profit = future_val - tx - deg;
                let val_low = base_profit + a * p_low_dt;
                let val_high = base_profit + a * p_high_dt;
                let val_jlo = base_profit + a * p_j_low_dt;
                let val_jhi = base_profit + a * p_j_high_dt;

                if val_low > best_low_vec[s_idx] { best_low_vec[s_idx] = val_low; }
                if val_high > best_high_vec[s_idx] { best_high_vec[s_idx] = val_high; }
                if val_jlo > best_jump_low_vec[s_idx] { best_jump_low_vec[s_idx] = val_jlo; }
                if val_jhi > best_jump_high_vec[s_idx] { best_jump_high_vec[s_idx] = val_jhi; }
            }
        }

        for s_idx in 0..levels {
            current[s_idx] = w_low * best_low_vec[s_idx]
                + w_high * best_high_vec[s_idx]
                + w_jump_low * best_jump_low_vec[s_idx]
                + w_jump_high * best_jump_high_vec[s_idx];
            if hp.use_aggregate_reg && hp.agg_reg_lambda > 0.0 {
                let soc_norm = s_idx as f64 / (levels - 1) as f64;
                let diff = soc_norm - fleet_soc_norm;
                current[s_idx] -= hp.agg_reg_lambda * diff * diff;
            }
        }
    }

    BatteryDP {
        soc_lo,
        soc_step_inv,
        levels,
        values,
        use_shift: hp.use_dp_value_shift,
        cluster_grid: None,
    }
}

fn build_battery_dp_nonuniform(
    battery: &Battery,
    da_at_node: &[f64],
    num_steps: usize,
    sigma: f64,
    p_jump: f64,
    mean_pareto: f64,
    second_pareto: f64,
    fleet_soc_norm: f64,
    hp: &Hyperparameters,
    grid: &[f64],
) -> BatteryDP {
    let levels = grid.len();
    let last = levels - 1;

    let mut bounds = Vec::with_capacity(levels);
    for s_idx in 0..levels {
        let soc = grid[s_idx];
        let (lo, hi) = compute_action_bounds(battery, soc);
        bounds.push((lo, hi));
    }

    let mut values = vec![vec![0.0; levels]; num_steps + 1];
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

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let friction = 2.0 * KAPPA_TX;

    let dt = DELTA_T;
    let cap2 = (battery.capacity_mwh * battery.capacity_mwh).max(1e-9);
    let tx_coeff = KAPPA_TX * dt;
    let deg_coeff = KAPPA_DEG * dt * dt / cap2;

    let mut best_low_vec = vec![0.0; levels];
    let mut best_high_vec = vec![0.0; levels];
    let mut best_jump_low_vec = vec![0.0; levels];
    let mut best_jump_high_vec = vec![0.0; levels];

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

        let actions = adaptive_action_grid(
            battery,
            charge_max,
            discharge_min,
            (price_low + price_high) * 0.5,
            hp.dp_action_levels,
        );

        best_low_vec.fill(f64::NEG_INFINITY);
        best_high_vec.fill(f64::NEG_INFINITY);
        best_jump_low_vec.fill(f64::NEG_INFINITY);
        best_jump_high_vec.fill(f64::NEG_INFINITY);

        let p_low_dt = price_low * dt;
        let p_high_dt = price_high * dt;
        let p_j_low_dt = price_jump_low * dt;
        let p_j_high_dt = price_jump_high * dt;

        for &raw in &actions {
            for s_idx in 0..levels {
                let (lo, hi) = bounds[s_idx];
                let a = raw.clamp(lo, hi);
                let a_abs = a.abs();
                let tx = tx_coeff * a_abs;
                let deg = deg_coeff * a_abs * a_abs;
                let soc = grid[s_idx];
                let next_soc = battery.apply_action_to_soc(a, soc);
                let future_val = interp_nonuniform(next, grid, next_soc);
                let base_profit = future_val - tx - deg;
                let val_low = base_profit + a * p_low_dt;
                let val_high = base_profit + a * p_high_dt;
                let val_jlo = base_profit + a * p_j_low_dt;
                let val_jhi = base_profit + a * p_j_high_dt;

                if val_low > best_low_vec[s_idx] { best_low_vec[s_idx] = val_low; }
                if val_high > best_high_vec[s_idx] { best_high_vec[s_idx] = val_high; }
                if val_jlo > best_jump_low_vec[s_idx] { best_jump_low_vec[s_idx] = val_jlo; }
                if val_jhi > best_jump_high_vec[s_idx] { best_jump_high_vec[s_idx] = val_jhi; }
            }
        }

        for s_idx in 0..levels {
            let soc = grid[s_idx];
            current[s_idx] = w_low * best_low_vec[s_idx]
                + w_high * best_high_vec[s_idx]
                + w_jump_low * best_jump_low_vec[s_idx]
                + w_jump_high * best_jump_high_vec[s_idx];
            if hp.use_aggregate_reg && hp.agg_reg_lambda > 0.0 {
                let soc_norm = (soc - battery.soc_min_mwh) / (battery.soc_max_mwh - battery.soc_min_mwh);
                let diff = soc_norm - fleet_soc_norm;
                current[s_idx] -= hp.agg_reg_lambda * diff * diff;
            }
        }
    }

    BatteryDP {
        soc_lo: grid[0],
        soc_step_inv: 0.0,
        levels,
        values,
        use_shift: false,
        cluster_grid: Some(grid.to_vec()),
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
    let next_soc = battery.apply_action_to_soc(action, soc);
    immediate_profit(battery, action, price) + dp_eval_future(dp, t + 1, next_soc)
}

fn dv_dsoc(dp: &BatteryDP, t: usize, soc: f64) -> f64 {
    let next_t = (t + 1).min(dp.values.len() - 1);
    if let Some(ref grid) = dp.cluster_grid {
        let values = &dp.values[next_t];
        let n = grid.len();
        if n <= 1 { return 0.0; }
        if soc <= grid[0] {
            return (values[1] - values[0]) / (grid[1] - grid[0]);
        }
        if soc >= grid[n-1] {
            return (values[n-1] - values[n-2]) / (grid[n-1] - grid[n-2]);
        }
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) >> 1;
            if grid[mid] <= soc {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (values[hi] - values[lo]) / (grid[hi] - grid[lo]);
    } else if dp.levels == 0 {
        return dp.values[next_t][1] + 2.0 * dp.values[next_t][2] * soc;
    }
    let values = &dp.values[next_t];
    let last = dp.levels - 1;
    if last == 0 {
        return 0.0;
    }
    let pos = ((soc - dp.soc_lo) * dp.soc_step_inv).clamp(0.0, last as f64);
    let mut low = pos.floor() as usize;
    if low >= last {
        low = last - 1;
    }
    (values[low + 1] - values[low]) * dp.soc_step_inv
}

fn build_aggregate_dp(
    batteries: &[Battery],
    da_prices_fleet: &[f64],
    num_steps: usize,
    sigma: f64,
    p_jump: f64,
    mean_pareto: f64,
    second_pareto: f64,
    e_levels: usize,
) -> BatteryDP {
    let e_agg_min: f64 = batteries.iter().map(|b| b.soc_min_mwh).sum();
    let e_agg_max: f64 = batteries.iter().map(|b| b.soc_max_mwh).sum();
    let total_charge_mw: f64 = batteries.iter().map(|b| b.power_charge_mw).sum();
    let total_discharge_mw: f64 = batteries.iter().map(|b| b.power_discharge_mw).sum();
    let total_cap = (e_agg_max - e_agg_min).max(1.0);

    let soc_lo = e_agg_min;
    let span = (e_agg_max - e_agg_min).max(1e-9);
    let levels = e_levels.max(2);
    let soc_step = span / (levels - 1) as f64;
    let soc_step_inv = 1.0 / soc_step;
    let last = levels - 1;

    let mut agg_bounds = Vec::with_capacity(levels);
    for s_idx in 0..levels {
        let soc = soc_lo + soc_step * s_idx as f64;
        let headroom = (e_agg_max - soc).max(0.0);
        let available = (soc - e_agg_min).max(0.0);
        let max_charge = (headroom / (ETA_CHARGE * DELTA_T)).min(total_charge_mw).max(0.0);
        let max_discharge = (available * ETA_DISCHARGE / DELTA_T).min(total_discharge_mw).max(0.0);
        agg_bounds.push((-max_charge, max_discharge));
    }

    let w_jump = p_jump.clamp(0.0, 1.0);
    let w_normal = (1.0 - w_jump).max(0.0);
    let w_low = 0.5 * w_normal;
    let w_high = 0.5 * w_normal;
    let jump_floor = 1.0_f64;
    let jump_ceiling = if second_pareto.is_finite() && mean_pareto.is_finite() && mean_pareto > jump_floor + EPS {
        ((second_pareto - mean_pareto * jump_floor) / (mean_pareto - jump_floor))
            .max(mean_pareto).min(80.0)
    } else {
        mean_pareto.max(jump_floor).min(80.0)
    };
    let w_jump_high = if jump_ceiling > jump_floor + EPS {
        w_jump * ((mean_pareto - jump_floor) / (jump_ceiling - jump_floor)).clamp(0.0, 1.0)
    } else { 0.0 };
    let w_jump_low = w_jump - w_jump_high;

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let friction = 2.0 * KAPPA_TX;

    let mut values = vec![vec![0.0; levels]; num_steps + 1];

    for t in (0..num_steps).rev() {
        let da = da_prices_fleet[t];
        let price_low = da * (1.0 - sigma);
        let price_high = da * (1.0 + sigma);
        let price_jump_low = da * (1.0 + jump_floor);
        let price_jump_high = da * (1.0 + jump_ceiling);

        let charge_max_low = price_low * eta_rt - friction;
        let discharge_min_low = price_low / eta_rt + friction;

        let (left, right) = values.split_at_mut(t + 1);
        let current = &mut left[t];
        let next = &right[0];

        let agg_actions = {
            let avg_price = (price_low + price_high) * 0.5;
            let in_dis = avg_price > discharge_min_low;
            let in_chg = avg_price < charge_max_low;
            let mut acts = vec![-total_charge_mw, 0.0, total_discharge_mw];
            if in_dis {
                for i in 1..5usize { acts.push(i as f64 / 5.0 * total_discharge_mw); }
            }
            if in_chg {
                for i in 1..5usize { acts.push(-(i as f64 / 5.0 * total_charge_mw)); }
            }
            acts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            acts.dedup_by(|a, b| (*a - *b).abs() < EPS);
            acts
        };

        for s_idx in 0..levels {
            let (lo, hi) = agg_bounds[s_idx];
            let soc = soc_lo + soc_step * s_idx as f64;

            let apply = |action: f64| -> f64 {
                if action < 0.0 {
                    soc + (-action) * ETA_CHARGE * DELTA_T
                } else if action > 0.0 {
                    soc - action * DELTA_T / ETA_DISCHARGE
                } else {
                    soc
                }
            };

            let imm_profit = |action: f64, price: f64| -> f64 {
                let throughput = action.abs() * DELTA_T;
                action * price * DELTA_T
                    - KAPPA_TX * throughput
                    - KAPPA_DEG * (throughput / total_cap).powi(2)
            };

            let mut best_low = f64::NEG_INFINITY;
            let mut best_high = f64::NEG_INFINITY;
            let mut best_jlo = f64::NEG_INFINITY;
            let mut best_jhi = f64::NEG_INFINITY;

            for &raw in &agg_actions {
                let action = raw.clamp(lo, hi);
                let ns = apply(action).clamp(soc_lo, e_agg_max);
                let future = interp_value(next, ns, soc_lo, soc_step_inv, last);
                best_low = best_low.max(imm_profit(action, price_low) + future);
                best_high = best_high.max(imm_profit(action, price_high) + future);
                best_jlo = best_jlo.max(imm_profit(action, price_jump_low) + future);
                best_jhi = best_jhi.max(imm_profit(action, price_jump_high) + future);
            }
            current[s_idx] = w_low * best_low + w_high * best_high
                + w_jump_low * best_jlo + w_jump_high * best_jhi;
        }
    }

    BatteryDP { soc_lo, soc_step_inv, levels, values, use_shift: false, cluster_grid: None }
}

#[inline(always)]
fn aggregate_dv_dsoc(agg_dp: &BatteryDP, t: usize, e_agg: f64) -> f64 {
    dv_dsoc(agg_dp, t, e_agg)
}

fn pick_dp_action(
    dp: &BatteryDP,
    battery: &Battery,
    t: usize,
    soc: f64,
    price: f64,
    bounds: (f64, f64),
    hp: &Hyperparameters,
    precomputed_grid: Option<&[f64]>,
) -> f64 {
    let (lo, hi) = bounds;

    if dp.levels == 0 {
        let next_t = (t + 1).min(dp.values.len() - 1);
        let beta_f = dp.values[next_t][1];
        let gamma_f = dp.values[next_t][2];
        let dt = DELTA_T;
        let cap2 = (battery.capacity_mwh * battery.capacity_mwh).max(1e-9);
        let c2 = dt * dt * (gamma_f / (ETA_DISCHARGE * ETA_DISCHARGE) - KAPPA_DEG / cap2);
        let c1 = dt * (price - KAPPA_TX - (beta_f + 2.0 * gamma_f * soc) / ETA_DISCHARGE);
        let a_d = if c2 < -1e-12 {
            (-c1 / (2.0 * c2)).clamp(0.0_f64.max(lo), hi)
        } else {
            if c1 > 0.0 { hi } else { 0.0_f64.max(lo) }
        };
        let d2 = dt * dt * (gamma_f * ETA_CHARGE * ETA_CHARGE - KAPPA_DEG / cap2);
        let d1 = dt * (price + KAPPA_TX - ETA_CHARGE * (beta_f + 2.0 * gamma_f * soc));
        let a_c = if d2 < -1e-12 {
            (-d1 / (2.0 * d2)).clamp(lo, 0.0_f64.min(hi))
        } else {
            if d1 < 0.0 { lo } else { 0.0_f64.min(hi) }
        };
        let mut best_action = 0.0_f64.clamp(lo, hi);
        let mut best_value = dp_action_value(dp, battery, t, soc, price, best_action);
        for &a in &[a_d, a_c, lo, hi] {
            let ac = a.clamp(lo, hi);
            let val = dp_action_value(dp, battery, t, soc, price, ac);
            if val > best_value { best_value = val; best_action = ac; }
        }
        return best_action;
    }

    let mut best_action = 0.0_f64.clamp(lo, hi);
    let mut best_value = dp_action_value(dp, battery, t, soc, price, best_action);

    let grid_vec: Vec<f64>;
    let grid: &[f64] = match precomputed_grid {
        Some(pre) => pre,
        None => {
            let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
            let friction = 2.0 * KAPPA_TX;
            let charge_max = price * eta_rt - friction;
            let discharge_min = price / eta_rt + friction;
            grid_vec = adaptive_action_grid(battery, charge_max, discharge_min, price, hp.policy_action_levels);
            &grid_vec
        }
    };
    for &raw in grid {
        let action = raw.clamp(lo, hi);
        let profit = immediate_profit(battery, action, price);
        let future = dp_eval_future(dp, t + 1, battery.apply_action_to_soc(action, soc));
        let value = profit + future;
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }
    for action in [lo, hi] {
        let a = action.clamp(lo, hi);
        let profit = immediate_profit(battery, a, price);
        let future = dp_eval_future(dp, t + 1, battery.apply_action_to_soc(a, soc));
        let value = profit + future;
        if value > best_value {
            best_value = value;
            best_action = a;
        }
    }

    best_action
}

fn admm_dispatch(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    init_actions: &[f64],
    hp: &Hyperparameters,
) -> Vec<f64> {
    let n_b = challenge.num_batteries;
    let n_l = sens.len();
    let rho = hp.admm_rho;
    const TOL: f64 = 0.05;

    let mut any_violated = false;
    for l in 0..n_l {
        let limit = challenge.network.flow_limits[l];
        let mut f = base_flows[l];
        for b in 0..n_b { f += sens[l][b] * init_actions[b]; }
        if f.abs() > limit + EPS_FLOW { any_violated = true; break; }
    }
    if !any_violated { return init_actions.to_vec(); }

    let mut actions = init_actions.to_vec();

    let zero = vec![0.0_f64; n_b];
    let mut best_feasible = zero.clone();
    let mut best_feasible_val = total_step_value(challenge, state, dps, &zero);

    let mut s: Vec<f64> = (0..n_l).map(|l| {
        let limit = challenge.network.flow_limits[l];
        let mut bat_f = 0.0_f64;
        for b in 0..n_b { bat_f += sens[l][b] * actions[b]; }
        (base_flows[l] + bat_f).clamp(-limit, limit)
    }).collect();

    let mut y = vec![0.0_f64; n_l];

    for _iter in 0..hp.max_admm_iters {
        let prev_actions = actions.clone();

        let mut bat_flow = vec![0.0_f64; n_l];
        for l in 0..n_l {
            for b in 0..n_b { bat_flow[l] += sens[l][b] * actions[b]; }
        }

        const GRID: usize = 65;
        for b in 0..n_b {
            let battery = &challenge.batteries[b];
            let soc = state.socs[b];
            let price = state.rt_prices[battery.node];
            let (lo, hi) = state.action_bounds[b];

            let offsets: Vec<(f64, f64)> = (0..n_l).filter_map(|l| {
                let imp = sens[l][b];
                if imp.abs() < 1e-12 { return None; }
                let off = s[l] - base_flows[l] + y[l] / rho
                    - (bat_flow[l] - imp * actions[b]);
                Some((off, imp))
            }).collect();

            let step = if hi > lo { (hi - lo) / GRID as f64 } else { 0.0 };
            let mut best_u = actions[b];
            let mut best_val = f64::NEG_INFINITY;

            for k in 0..=GRID {
                let u = (lo + k as f64 * step).clamp(lo, hi);
                let next_soc = battery.apply_action_to_soc(u, soc);
                let future = dp_eval_future(&dps[b], state.time_step + 1, next_soc);
                let profit = immediate_profit(battery, u, price) + future;
                let penalty: f64 = offsets.iter().map(|&(off, imp)| {
                    let err = off - imp * u;
                    (rho / 2.0) * err * err
                }).sum();
                let val = profit - penalty;
                if val > best_val { best_val = val; best_u = u; }
            }

            let delta = best_u - actions[b];
            for l in 0..n_l { bat_flow[l] += sens[l][b] * delta; }
            actions[b] = best_u;
        }

        for l in 0..n_l {
            let limit = challenge.network.flow_limits[l];
            s[l] = (bat_flow[l] + base_flows[l] - y[l] / rho).clamp(-limit, limit);
        }

        let mut max_resid = 0.0_f64;
        for l in 0..n_l {
            let resid = s[l] - bat_flow[l] - base_flows[l];
            y[l] += rho * resid;
            max_resid = max_resid.max(resid.abs());
        }
        let max_du = (0..n_b)
            .map(|b| (actions[b] - prev_actions[b]).abs())
            .fold(0.0_f64, f64::max);

        if is_flow_feasible(challenge, state, &actions) {
            let val = total_step_value(challenge, state, dps, &actions);
            if val > best_feasible_val {
                best_feasible_val = val;
                best_feasible = actions.clone();
            }
        }

        if max_resid < TOL && max_du < TOL { break; }
    }

    best_feasible
}

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

fn ct_simulate_flows(
    challenge: &Challenge,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    candidate_lines: &[usize],
    hp: &Hyperparameters,
) -> Vec<Vec<f64>> {
    let n_b = challenge.num_batteries;
    let n_t = challenge.num_steps;
    let n_l = sens.len();
    let mut socs: Vec<f64> = challenge.batteries.iter().map(|b| b.soc_initial_mwh).collect();
    let mut flows_all = Vec::with_capacity(n_t);
    for t in 0..n_t {
        let mut action = vec![0.0_f64; n_b];
        for b in 0..n_b {
            let battery = &challenge.batteries[b];
            let soc = socs[b];
            let (lo, hi) = compute_action_bounds(battery, soc);
            if hi - lo > EPS {
                let price = challenge.market.day_ahead_prices[t][battery.node];
                action[b] = pick_dp_action(&dps[b], battery, t, soc, price, (lo, hi), hp, None);
            }
        }
        let exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
        let mut flows_t = vec![0.0_f64; n_l];
        for &l in candidate_lines {
            flows_t[l] = exo[l] + sens[l].iter().zip(action.iter()).map(|(s, a)| s * a).sum::<f64>();
        }
        flows_all.push(flows_t);
        for b in 0..n_b {
            let battery = &challenge.batteries[b];
            socs[b] = battery.apply_action_to_soc(action[b], socs[b])
                .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        }
    }
    flows_all
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
    let n_l = limits.len();
    let ok = project_polytope(action, &state.action_bounds, sens, base_flows, limits, hp.proj_max_iters);
    if ok {
        let mut feasible = true;
        for l in 0..n_l {
            let f = line_flow(&sens[l], action, base_flows[l]);
            if f.abs() > limits[l] * (1.0 + EPS_FLOW) + 1e-6 {
                feasible = false;
                break;
            }
        }
        if feasible {
            return;
        }
    }
    let original = action.clone();
    let n_b = original.len();
    let mut lambda = 1.0_f64;
    for l in 0..n_l {
        let a = base_flows[l];
        let mut b = 0.0_f64;
        for bi in 0..n_b {
            b += sens[l][bi] * original[bi];
        }
        let lim = limits[l];
        if b.abs() <= EPS {
            if a.abs() > lim + EPS_FLOW {
                lambda = 0.0;
                break;
            }
            continue;
        }
        let upper = (lim - a) / b;
        let lower = (-lim - a) / b;
        let low_f = lower.min(upper);
        let high_f = lower.max(upper);
        let line_max = high_f.min(1.0);
        let line_min = low_f.max(0.0);
        if line_min > line_max + EPS {
            lambda = 0.0;
            break;
        }
        if line_max < lambda {
            lambda = line_max;
        }
    }
    if lambda < 0.0 { lambda = 0.0; }
    for b in 0..n_b {
        action[b] = original[b] * lambda;
    }
    clamp_to_bounds(action, &state.action_bounds);
    let mut feasible = true;
    for l in 0..n_l {
        let f = line_flow(&sens[l], action, base_flows[l]);
        if f.abs() > limits[l] * (1.0 + EPS_FLOW) + 1e-6 {
            feasible = false;
            break;
        }
    }
    if !feasible {
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
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
    out: &mut [f64],
) {
    let t = state.time_step;
    for b in 0..action.len() {
        let dc = delta_cong.get(b).and_then(|v| v.get(t)).copied().unwrap_or(0.0);
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
        let dv = dv_dsoc(&dps[b], state.time_step, next_soc) + cwv_lambda * dc;
        out[b] = imm + dv * dsoc_du;
    }
}

struct GradBuffer {
    velocity: Vec<f64>,
    grad: Vec<f64>,
    dir: Vec<f64>,
    trial: Vec<f64>,
    action: Vec<f64>,
    best_action: Vec<f64>,
}

impl GradBuffer {
    fn new(size: usize) -> Self {
        GradBuffer {
            velocity: Vec::with_capacity(size),
            grad: Vec::with_capacity(size),
            dir: Vec::with_capacity(size),
            trial: Vec::with_capacity(size),
            action: Vec::with_capacity(size),
            best_action: Vec::with_capacity(size),
        }
    }
}

fn projected_gradient_ascent(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    seed: Vec<f64>,
    hp: &Hyperparameters,
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
    buf: &mut GradBuffer,
) -> (Vec<f64>, f64) {
    let n = seed.len();
    buf.action.resize(n, 0.0);
    buf.action.copy_from_slice(&seed);
    safe_project_to_feasible(challenge, state, &mut buf.action, sens, base_flows, hp);
    let mut best_value = total_step_value(challenge, state, dps, &buf.action);
    buf.best_action.resize(n, 0.0);
    buf.best_action.copy_from_slice(&buf.action);

    let max_power: f64 = challenge
        .batteries
        .iter()
        .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
        .fold(1.0_f64, f64::max);

    let mut lr = max_power * 0.5;
    buf.velocity.resize(n, 0.0);
    buf.velocity.fill(0.0);
    buf.grad.resize(n, 0.0);
    buf.dir.resize(n, 0.0);
    buf.trial.resize(n, 0.0);

    let base_budget = hp.grad_outer_iters;
    let extra = iter_pool_claim(base_budget as i64) as usize;
    let total_limit = base_budget + extra;

    let mut iters_run = 0usize;
    let mut exited_early = false;
    for outer_iter in 0..total_limit {
        iters_run += 1;
        analytic_gradient(challenge, state, dps, &buf.action, delta_cong, cwv_lambda, &mut buf.grad);
        let g_norm: f64 = buf.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm < 1e-9 {
            exited_early = true;
            break;
        }

        let beta_t = if hp.use_cosine_beta && base_budget > 1 {
            let frac = outer_iter as f64 / (base_budget - 1) as f64;
            BETA_END + (MOMENTUM_BETA - BETA_END) * (1.0 + (std::f64::consts::PI * frac).cos()) * 0.5
        } else {
            MOMENTUM_BETA
        };

        if hp.use_momentum {
            for b in 0..n {
                buf.dir[b] = beta_t * buf.velocity[b] + buf.grad[b];
            }
        } else {
            buf.dir.copy_from_slice(&buf.grad);
        }

        let mut improved = false;
        let mut cur_lr = lr;
        for _ in 0..hp.grad_ls_iters {
            let step_scale = cur_lr / g_norm;
            for b in 0..n {
                buf.trial[b] = buf.action[b] + step_scale * buf.dir[b];
            }
            safe_project_to_feasible(challenge, state, &mut buf.trial, sens, base_flows, hp);
            let v = total_step_value(challenge, state, dps, &buf.trial);
            if v > best_value + 1e-9 {
                buf.action.copy_from_slice(&buf.trial);
                best_value = v;
                buf.best_action.copy_from_slice(&buf.trial);
                improved = true;
                lr = cur_lr * 1.4;
                if hp.use_momentum {
                    for b in 0..n {
                        buf.velocity[b] = beta_t * buf.velocity[b] + buf.grad[b];
                    }
                }
                break;
            }
            cur_lr *= 0.5;
        }
        if !improved {
            lr *= 0.4;
            if lr < max_power * 1e-4 {
                exited_early = true;
                break;
            }
        }
    }
    if exited_early {
        iter_pool_donate((total_limit - iters_run) as i64);
    }
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&buf.best_action);
    (result, best_value)
}

fn joint_optimize_step(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    seeds: Vec<Vec<f64>>,
    hp: &Hyperparameters,
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
    buf: &mut GradBuffer,
) -> Vec<f64> {
    let mut best_action = vec![0.0_f64; challenge.num_batteries];
    let mut best_value = total_step_value(challenge, state, dps, &best_action);

    for seed in seeds {
        let (a, v) = projected_gradient_ascent(
            challenge, state, dps, sens, base_flows, seed, hp, delta_cong, cwv_lambda, buf,
        );
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

    let mut flows = compute_flows(challenge, state, &action);
    let limits = &challenge.network.flow_limits;
    let mut best_value = total_step_value(challenge, state, dps, &action);
    for _ in 0..hp.coord_polish_passes {
        let mut improved = false;
        for b in 0..challenge.num_batteries {
            let (lo, hi) = state.action_bounds[b];
            let cur = action[b];
            let node_b = challenge.batteries[b].node;
            let cur_val = dp_action_value(
                &dps[b],
                &challenge.batteries[b],
                state.time_step,
                state.socs[b],
                state.rt_prices[node_b],
                cur,
            );
            let mut net_lo = lo;
            let mut net_hi = hi;
            for l in 0..challenge.network.num_lines {
                let coeff = sens[l][b];
                if coeff.abs() <= 1e-12 {
                    continue;
                }
                let without_b = flows[l] - coeff * cur;
                let limit = limits[l];
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
                let delta = candidate - cur;
                let mut feasible = true;
                for l in 0..flows.len() {
                    let coeff = sens[l][b];
                    if coeff.abs() <= 1e-12 {
                        continue;
                    }
                    let f_new = flows[l] + coeff * delta;
                    if f_new.abs() > limits[l] * (1.0 + EPS_FLOW) {
                        feasible = false;
                        break;
                    }
                }
                if !feasible {
                    continue;
                }
                let candidate_val = dp_action_value(
                    &dps[b],
                    &challenge.batteries[b],
                    state.time_step,
                    state.socs[b],
                    state.rt_prices[node_b],
                    candidate,
                );
                let new_value = best_value - cur_val + candidate_val;
                if new_value > best_b_value + 1e-9 {
                    best_b_value = new_value;
                    best_b_action = candidate;
                }
            }

            if (best_b_action - cur).abs() > EPS {
                let delta_change = best_b_action - cur;
                for l in 0..flows.len() {
                    flows[l] += sens[l][b] * delta_change;
                }
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

fn pairwise_perturb_step(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    mut action: Vec<f64>,
) -> Vec<f64> {
    let nb = challenge.num_batteries;
    let nl = challenge.network.num_lines;
    if nb < 2 || nl == 0 {
        return action;
    }
    let limits = &challenge.network.flow_limits;
    let t = state.time_step;

    let mut flows: Vec<f64> = (0..nl)
        .map(|l| {
            let mut f = base_flows[l];
            for b in 0..nb {
                f += sens[l][b] * action[b];
            }
            f
        })
        .collect();

    let mut tested = 0usize;
    let mut pass_improved = true;
    while pass_improved && tested < PAIR_POLISH_BUDGET {
        pass_improved = false;
        'outer: for i in 0..nb {
            let (lo_i, hi_i) = state.action_bounds[i];
            let span_i = hi_i - lo_i;
            if span_i < EPS {
                continue;
            }
            let battery_i = &challenge.batteries[i];
            let price_i = state.rt_prices[battery_i.node];
            let cur_i = action[i];
            let val_i = dp_action_value(&dps[i], battery_i, t, state.socs[i], price_i, cur_i);

            for j in (i + 1)..nb {
                if tested >= PAIR_POLISH_BUDGET {
                    break 'outer;
                }
                let (lo_j, hi_j) = state.action_bounds[j];
                let span_j = hi_j - lo_j;
                if span_j < EPS {
                    continue;
                }
                tested += 1;
                let battery_j = &challenge.batteries[j];
                let price_j = state.rt_prices[battery_j.node];
                let cur_j = action[j];
                let val_j = dp_action_value(&dps[j], battery_j, t, state.socs[j], price_j, cur_j);
                let base_pair_val = val_i + val_j;

                let mut best_pair_val = base_pair_val;
                let mut best_di = 0.0_f64;
                let mut best_dj = 0.0_f64;

                for &sign in &[1.0_f64, -1.0_f64] {
                    let cand_i = (cur_i + sign * PAIR_POLISH_ALPHA * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - sign * PAIR_POLISH_ALPHA * span_j).clamp(lo_j, hi_j);
                    let di = cand_i - cur_i;
                    let dj = cand_j - cur_j;
                    if di.abs() < EPS && dj.abs() < EPS {
                        continue;
                    }

                    let mut feasible = true;
                    for l in 0..nl {
                        let f_new = flows[l] + sens[l][i] * di + sens[l][j] * dj;
                        if f_new.abs() > limits[l] * (1.0 + EPS_FLOW) + 1e-6 {
                            feasible = false;
                            break;
                        }
                    }
                    if !feasible {
                        continue;
                    }

                    let pair_val =
                        dp_action_value(&dps[i], battery_i, t, state.socs[i], price_i, cand_i)
                            + dp_action_value(
                                &dps[j],
                                battery_j,
                                t,
                                state.socs[j],
                                price_j,
                                cand_j,
                            );
                    if pair_val > best_pair_val + 1e-9 {
                        best_pair_val = pair_val;
                        best_di = di;
                        best_dj = dj;
                    }
                }

                if best_di.abs() > EPS || best_dj.abs() > EPS {
                    for l in 0..nl {
                        flows[l] += sens[l][i] * best_di + sens[l][j] * best_dj;
                    }
                    action[i] = cur_i + best_di;
                    action[j] = cur_j + best_dj;
                    pass_improved = true;
                }
            }
        }
    }
    action
}

fn joint_pair_polish(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    actions: &mut Vec<f64>,
    hp: &Hyperparameters,
) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b < 2 {
        return;
    }
    let t = state.time_step;
    let limits = &challenge.network.flow_limits;
    let mut flows = vec![0.0_f64; num_l];
    for l in 0..num_l {
        let mut f = base_flows[l];
        for b in 0..num_b {
            f += sens[l][b] * actions[b];
        }
        flows[l] = f;
    }
    let pair_budget = hp.joint_pair_budget.max(1);
    let mut improved = true;
    while improved {
        improved = false;
        let mut tested = 0usize;
        'outer: for i in 0..num_b {
            let batt_i = &challenge.batteries[i];
            let price_i = state.rt_prices[batt_i.node];
            let soc_i = state.socs[i];
            let (lo_i, hi_i) = state.action_bounds[i];
            let cur_i = actions[i];
            let span_i = hi_i - lo_i;
            for j in (i + 1)..num_b {
                if tested >= pair_budget {
                    break 'outer;
                }
                tested += 1;
                let batt_j = &challenge.batteries[j];
                let price_j = state.rt_prices[batt_j.node];
                let soc_j = state.socs[j];
                let (lo_j, hi_j) = state.action_bounds[j];
                let cur_j = actions[j];
                let span_j = hi_j - lo_j;
                let base_val = dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cur_i)
                    + dp_action_value(&dps[j], batt_j, t, soc_j, price_j, cur_j);
                let mut best_val = base_val;
                let mut best_i = cur_i;
                let mut best_j = cur_j;
                for &alpha in &[-0.5_f64, -0.25, 0.25, 0.5] {
                    let cand_i = (cur_i + alpha * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - alpha * span_j).clamp(lo_j, hi_j);
                    let delta_i = cand_i - cur_i;
                    let delta_j = cand_j - cur_j;
                    let mut feasible = true;
                    for l in 0..num_l {
                        let limit = limits[l];
                        if limit <= 1e-6 {
                            continue;
                        }
                        let f_new = flows[l] + sens[l][i] * delta_i + sens[l][j] * delta_j;
                        if f_new.abs() > limit {
                            feasible = false;
                            break;
                        }
                    }
                    if feasible {
                        let val = dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cand_i)
                            + dp_action_value(&dps[j], batt_j, t, soc_j, price_j, cand_j);
                        if val > best_val + 1e-9 {
                            best_val = val;
                            best_i = cand_i;
                            best_j = cand_j;
                        }
                    }
                }
                if (best_i - cur_i).abs() > EPS || (best_j - cur_j).abs() > EPS {
                    let delta_i = best_i - cur_i;
                    let delta_j = best_j - cur_j;
                    actions[i] = best_i;
                    actions[j] = best_j;
                    for l in 0..num_l {
                        flows[l] += sens[l][i] * delta_i + sens[l][j] * delta_j;
                    }
                    improved = true;
                    break 'outer;
                }
            }
        }
    }
}







fn policy(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    coupling_prems: &[Vec<f64>],
    hp: &Hyperparameters,
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
    buf: &mut GradBuffer,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_steps = challenge.num_steps;
    let n_remaining = n_steps.saturating_sub(t);
    if n_remaining == 0 {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    let mut target = vec![0.0_f64; challenge.num_batteries];
    let mut dp_seed = vec![0.0_f64; challenge.num_batteries];

    for (b, battery) in challenge.batteries.iter().enumerate() {
        let current_price = state.rt_prices[battery.node];
        let eff_price = current_price + coupling_prems[t][b];

        let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
        let friction = 2.0 * KAPPA_TX;
        let charge_max = current_price * eta_rt - friction;
        let discharge_min = current_price / eta_rt + friction;
        let grid = adaptive_action_grid(battery, charge_max, discharge_min, current_price, hp.policy_action_levels);

        let a = pick_dp_action(
            &dps[b],
            battery,
            t,
            state.socs[b],
            eff_price,
            state.action_bounds[b],
            hp,
            Some(&grid),
        );
        target[b] = a;
        dp_seed[b] = pick_dp_action(
            &dps[b],
            battery,
            t,
            state.socs[b],
            current_price,
            state.action_bounds[b],
            hp,
            Some(&grid),
        );
    }

    clamp_to_bounds(&mut target, &state.action_bounds);

    let zero = vec![0.0_f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero);

    let mut result = if hp.use_dual_dispatch {
        admm_dispatch(challenge, state, dps, sens, &base_flows, &target, hp)
    } else {
        let mut seeds = vec![target, dp_seed, zero.clone()];
        seeds.truncate(hp.num_seeds.max(1));
        // Warm-start seed from previous step
        if t > 0 {
            let store = prev_policy_store().lock().unwrap();
            if let Some((ref prev_actions, ref prev_socs)) = *store {
                let mut warm = vec![0.0_f64; challenge.num_batteries];
                for b in 0..challenge.num_batteries {
                    let battery = &challenge.batteries[b];
                    let (prev_lo, prev_hi) = compute_action_bounds(battery, prev_socs[b]);
                    let (new_lo, new_hi) = state.action_bounds[b];
                    let prev_a = prev_actions[b];
                    let new_a = if prev_a >= 0.0 {
                        let frac = if prev_hi > 1e-9 { (prev_a / prev_hi).clamp(0.0, 1.0) } else { 0.0 };
                        frac * new_hi
                    } else {
                        let frac = if prev_lo < -1e-9 { (prev_a / prev_lo).clamp(0.0, 1.0) } else { 0.0 };
                        frac * new_lo
                    };
                    warm[b] = new_a.clamp(new_lo, new_hi);
                }
                seeds.push(warm);
            }
        }
        let pga_result = joint_optimize_step(
            challenge, state, dps, sens, &base_flows, seeds, hp, delta_cong, cwv_lambda, buf,
        );
        let pga_val = total_step_value(challenge, state, dps, &pga_result);
        let mut r = pga_result;

        if hp.use_lp_dispatch {
            if let Some(mut lp_act) = lp_dispatch_step(challenge, state, dps, sens, &base_flows, hp) {
                safe_project_to_feasible(challenge, state, &mut lp_act, sens, &base_flows, hp);
                if is_flow_feasible(challenge, state, &lp_act) {
                    let lp_val = total_step_value(challenge, state, dps, &lp_act);
                    if lp_val > pga_val {
                        r = lp_act;
                    }
                }
            }
        }
        r
    };

    if hp.use_pair_polish {
        result = pairwise_perturb_step(challenge, state, dps, sens, &base_flows, result);
    }
    result = coordinate_polish_step(challenge, state, dps, sens, result, hp);

    // Gradient polish: one projected gradient step to capture joint corrections
    {
        let base_val = total_step_value(challenge, state, dps, &result);
        let max_power: f64 = challenge
            .batteries
            .iter()
            .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
            .fold(1.0_f64, f64::max);
        buf.grad.resize(challenge.num_batteries, 0.0);
        analytic_gradient(challenge, state, dps, &result, delta_cong, cwv_lambda, &mut buf.grad);
        let g_norm: f64 = buf.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm > 1e-9 {
            buf.dir.resize(challenge.num_batteries, 0.0);
            buf.trial.resize(challenge.num_batteries, 0.0);
            let mut best_val = base_val;
            let mut best_step_scale = 0.0_f64;
            let mut cur_lr = max_power * 0.25;
            for _ in 0..3 {
                let step_scale = cur_lr / g_norm;
                for b in 0..challenge.num_batteries {
                    buf.trial[b] = result[b] + step_scale * buf.grad[b];
                }
                safe_project_to_feasible(challenge, state, &mut buf.trial, sens, &base_flows, hp);
                let v = total_step_value(challenge, state, dps, &buf.trial);
                if v > best_val + 1e-9 {
                    best_val = v;
                    best_step_scale = step_scale;
                    cur_lr *= 1.5;
                } else {
                    cur_lr *= 0.5;
                }
            }
            if best_step_scale > 0.0 {
                for b in 0..challenge.num_batteries {
                    result[b] += best_step_scale * buf.grad[b];
                }
                safe_project_to_feasible(challenge, state, &mut result, sens, &base_flows, hp);
            }
        }
    }

    if hp.use_joint_pair_polish {
        let pre_polish = result.clone();
        joint_pair_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_polish;
        }
    }

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

    let sens = build_sensitivity(challenge);

    let n_lines = challenge.network.flow_limits.len();
    let expected_premiums: Vec<Vec<f64>> = if hp.anticipate_lmp && n_lines > 0 {
        let base_premium = 20.0 * LMP_PREMIUM_SCALE;
        let threshold = LMP_THRESHOLD;
        let n_t = challenge.num_steps;
        let n_b = challenge.num_batteries;
        let mut prem = vec![vec![0.0_f64; n_b]; n_t];
        for t in 0..n_t {
            let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            for l in 0..n_lines {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let ratio = f_exo[l].abs() / limit;
                if ratio > threshold {
                    let proba = ((ratio - threshold) / (1.0 - threshold).max(1e-6))
                        .clamp(0.0, 1.0);
                    let premium = base_premium * proba;
                    let sign_f = if f_exo[l] >= 0.0 { 1.0_f64 } else { -1.0_f64 };
                    for b in 0..n_b {
                        let impact = sens[l][b];
                        if impact.abs() > 1e-6 {
                            prem[t][b] += -impact * sign_f * premium;
                        }
                    }
                }
            }
        }
        prem
    } else {
        vec![vec![0.0_f64; challenge.num_batteries]; challenge.num_steps]
    };

    let fleet_soc_norm: f64 = if hp.use_aggregate_reg && !challenge.batteries.is_empty() {
        let n_b = challenge.batteries.len() as f64;
        challenge.batteries.iter().map(|b| {
            let span = (b.soc_max_mwh - b.soc_min_mwh).max(1e-9);
            (b.soc_initial_mwh - b.soc_min_mwh) / span
        }).sum::<f64>() / n_b
    } else {
        0.0
    };

    let mut dps: Vec<BatteryDP> = challenge
        .batteries
        .iter()
        .enumerate()
        .map(|(b, battery)| {
            let node = battery.node;
            let da_at_node: Vec<f64> = (0..challenge.num_steps)
                .map(|t| challenge.market.day_ahead_prices[t][node] + expected_premiums[t][b])
                .collect();
            build_battery_dp(
                battery,
                &da_at_node,
                challenge.num_steps,
                sigma,
                p_jump,
                mean_pareto,
                second_pareto,
                fleet_soc_norm,
                &hp,
            )
        })
        .collect();

    let coupling_prems: Vec<Vec<f64>> = if hp.use_coupling_cut && n_lines > 0 {
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let base = 20.0 * LMP_PREMIUM_SCALE;
        let threshold = LMP_THRESHOLD;
        let mut action_est = vec![vec![0.0_f64; n_b]; n_t];
        for b in 0..n_b {
            let battery = &challenge.batteries[b];
            let soc_mid = battery.soc_min_mwh
                + (battery.soc_max_mwh - battery.soc_min_mwh) * 0.5;
            for t in 0..n_t {
                let node = battery.node;
                let price =
                    challenge.market.day_ahead_prices[t][node] + expected_premiums[t][b];
                let (lo, hi) = compute_action_bounds(battery, soc_mid);
                action_est[t][b] =
                    pick_dp_action(&dps[b], battery, t, soc_mid, price, (lo, hi), &hp, None);
            }
        }
        let mut c_prem = vec![vec![0.0_f64; n_b]; n_t];
        for t in 0..n_t {
            let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let f_endo: Vec<f64> = (0..n_lines)
                .map(|l| (0..n_b).map(|b| sens[l][b] * action_est[t][b]).sum::<f64>())
                .collect();
            for l in 0..n_lines {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 {
                    continue;
                }
                let f_total = f_exo[l] + f_endo[l];
                if f_total.abs() <= limit * threshold {
                    continue;
                }
                let delta_ratio = (f_total.abs() - f_exo[l].abs()).max(0.0) / limit;
                if delta_ratio < 1e-6 {
                    continue;
                }
                let coupling_p = base * delta_ratio.clamp(0.0, 1.0);
                let sign_f = if f_total >= 0.0 { 1.0 } else { -1.0 };
                for b in 0..n_b {
                    let impact = sens[l][b];
                    if impact.abs() > 1e-6 {
                        c_prem[t][b] += -impact * sign_f * coupling_p;
                    }
                }
            }
        }
        c_prem
    } else {
        vec![vec![0.0_f64; challenge.num_batteries]; challenge.num_steps]
    };

    // Curvature-based grid refinement: add points where DP value's second derivative is large
    if !hp.use_sqdp {
        let mut new_dps = Vec::with_capacity(challenge.num_batteries);
        for (b, battery) in challenge.batteries.iter().enumerate() {
            let dp = &dps[b];
            let base_levels = dp.levels;
            let soc_min = battery.soc_min_mwh;
            let soc_max = battery.soc_max_mwh;
            let span = soc_max - soc_min;
            let soc_step = span / (base_levels - 1) as f64;
            let mut grid_points: Vec<f64> = (0..base_levels)
                .map(|i| soc_min + soc_step * i as f64)
                .collect();

            if base_levels >= 3 {
                // compute max absolute second derivative across time for each interior SOC index
                let n_t = dp.values.len() - 1; // values dimension is (num_steps+1) x levels
                let mut curv = vec![0.0_f64; base_levels];
                for i in 1..base_levels - 1 {
                    let mut max_abs = 0.0_f64;
                    for t in 0..=n_t {
                        let v = &dp.values[t];
                        let d2 = (v[i - 1] - 2.0 * v[i] + v[i + 1]).abs();
                        if d2 > max_abs { max_abs = d2; }
                    }
                    curv[i] = max_abs;
                }
                // select top indices based on curvature magnitude; allocate up to base_levels extra points
                let mut indices: Vec<usize> = (1..base_levels - 1).collect();
                indices.sort_by(|&a, &b| curv[b].partial_cmp(&curv[a]).unwrap_or(std::cmp::Ordering::Equal));
                let max_extra = base_levels; // total grid points limited to 2*base_levels
                let mut added = 0usize;
                for &i in &indices {
                    if added + 2 > max_extra {
                        break;
                    }
                    if curv[i] <= 1e-9 {
                        break; // remaining curvature negligible
                    }
                    let soc_i = grid_points[i];
                    // add midpoints with neighbors
                    if i > 0 {
                        let left_mid = (grid_points[i - 1] + soc_i) * 0.5;
                        if !grid_points.iter().any(|&x| (x - left_mid).abs() < 1e-8) {
                            grid_points.push(left_mid);
                            added += 1;
                        }
                    }
                    if i < base_levels - 1 {
                        let right_mid = (soc_i + grid_points[i + 1]) * 0.5;
                        if !grid_points.iter().any(|&x| (x - right_mid).abs() < 1e-8) {
                            grid_points.push(right_mid);
                            added += 1;
                        }
                    }
                }
            }

            grid_points.sort_by(|a,b| a.partial_cmp(b).unwrap());
            grid_points.dedup_by(|a,b| (*a - *b).abs() < 1e-9);
            // enforce size cap (should already be satisfied)
            if grid_points.len() > base_levels * 2 {
                let keep = base_levels * 2;
                let step = grid_points.len() / keep;
                let mut sub = Vec::with_capacity(keep);
                for i in 0..keep {
                    sub.push(grid_points[i * step]);
                }
                sub.push(soc_max);
                sub.sort_by(|a,b| a.partial_cmp(b).unwrap());
                sub.dedup_by(|a,b| (*a - *b).abs() < 1e-9);
                grid_points = sub;
            }
            let node = battery.node;
            let da_at_node: Vec<f64> = (0..challenge.num_steps)
                .map(|t| challenge.market.day_ahead_prices[t][node] + expected_premiums[t][b])
                .collect();
            let clustered_dp = build_battery_dp_nonuniform(
                battery,
                &da_at_node,
                challenge.num_steps,
                sigma,
                p_jump,
                mean_pareto,
                second_pareto,
                fleet_soc_norm,
                &hp,
                &grid_points,
            );
            new_dps.push(clustered_dp);
        }
        dps = new_dps;
    }

    let dps = if hp.use_ptdf_ct && n_lines > 0 {
        let eta = hp.ct_step_eta;
        let ct_scale = 1.0 - hp.ct_ref_kappa;
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let limits = &challenge.network.flow_limits;

        const ACTIVE_SENS_THRESH: f64 = 1e-4;
        let candidate_lines: Vec<usize> = (0..n_lines)
            .filter(|&l| limits[l] > 1e-6 && sens[l].iter().any(|&s| s.abs() > ACTIVE_SENS_THRESH)) 
            .collect();

        let flows_all = ct_simulate_flows(challenge, &dps, &sens, &candidate_lines, &hp);

        let mut mu_oco = vec![vec![0.0_f64; n_lines]; n_t];
        for t in 0..n_t {
            for &l in &candidate_lines {
                let limit = limits[l];
                let viol = flows_all[t][l].abs() - limit;
                if viol > 0.0 {
                    mu_oco[t][l] = (eta * viol).min(limit * 0.5);
                }
            }
        }

        let mut ep_ct = expected_premiums.clone();
        let mut touched = vec![false; n_b];
        for t in 0..n_t {
            for &l in &candidate_lines {
                let mu_l = mu_oco[t][l];
                if mu_l <= 1e-12 { continue; }
                let sign = if flows_all[t][l] >= 0.0 { 1.0_f64 } else { -1.0_f64 };
                for b in 0..n_b {
                    let impact = sens[l][b];
                    if impact.abs() > 1e-6 {
                        ep_ct[t][b] -= impact * sign * mu_l * ct_scale;
                        touched[b] = true;
                    }
                }
            }
        }

        if hp.ct_gdd_alpha > 1e-12 {
            for t in 0..n_t {
                for b in 0..n_b {
                    let s_b: f64 = candidate_lines
                        .iter()
                        .map(|&l| {
                            let limit = limits[l];
                            if limit <= 1e-6 {
                                return 0.0;
                            }
                            let v_frac = (flows_all[t][l].abs() - limit).max(0.0) / limit;
                            v_frac * sens[l][b].abs()
                        })
                        .sum();
                    if s_b > 1e-9 {
                        ep_ct[t][b] -= (hp.ct_gdd_alpha * s_b).exp() - 1.0;
                        touched[b] = true;
                    }
                }
            }
        }

        let mut hp_oco = hp;
        hp_oco.dp_soc_levels = (hp.dp_soc_levels / 2).max(17);
        hp_oco.dp_action_levels = (hp.dp_action_levels / 2).max(5);
        challenge
            .batteries
            .iter()
            .enumerate()
            .map(|(b, battery)| {
                if !touched[b] {
                    dps[b].clone()
                } else {
                    let node = battery.node;
                    let da_ct: Vec<f64> = (0..n_t)
                        .map(|t| challenge.market.day_ahead_prices[t][node] + ep_ct[t][b])
                        .collect();
                    build_battery_dp(
                        battery,
                        &da_ct,
                        n_t,
                        sigma,
                        p_jump,
                        mean_pareto,
                        second_pareto,
                        fleet_soc_norm,
                        &hp_oco,
                    )
                }
            })
            .collect()
    } else {
        dps
    };

    // Reset warm-start previous policy store
    *prev_policy_store().lock().unwrap() = None;

    let cwv_lambda = if hp.use_composite_wv { hp.cwv_lambda } else { 0.0 };
    let delta_cong: Vec<Vec<f64>> = if hp.use_composite_wv {
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let k = hp.cwv_clusters.max(1).min(n_b.max(1));

        if k == 1 {
            let total_cap: f64 =
                challenge.batteries.iter().map(|b| b.capacity_mwh).sum::<f64>().max(1.0);
            let fleet_da: Vec<f64> = (0..n_t)
                .map(|t| {
                    let mut p = 0.0_f64;
                    for batt in challenge.batteries.iter() {
                        p += batt.capacity_mwh * challenge.market.day_ahead_prices[t][batt.node];
                    }
                    p / total_cap
                })
                .collect();
            let fleet_premium: Vec<f64> = (0..n_t)
                .map(|t| expected_premiums[t].iter().sum::<f64>() / (n_b as f64).max(1.0))
                .collect();
            let da_with_cong: Vec<f64> = fleet_da
                .iter()
                .zip(fleet_premium.iter())
                .map(|(da, prem)| da + prem)
                .collect();
            let agg_dp_cong = build_aggregate_dp(
                &challenge.batteries, &da_with_cong, n_t,
                sigma, p_jump, mean_pareto, second_pareto, hp.cwv_agg_levels.max(2),
            );
            let agg_dp_nocong = build_aggregate_dp(
                &challenge.batteries, &fleet_da, n_t,
                sigma, p_jump, mean_pareto, second_pareto, hp.cwv_agg_levels.max(2),
            );
            let e_mid: f64 = challenge
                .batteries
                .iter()
                .map(|b| (b.soc_min_mwh + b.soc_max_mwh) * 0.5)
                .sum();
            let fleet_delta: Vec<f64> = (0..n_t)
                .map(|t| aggregate_dv_dsoc(&agg_dp_cong, t, e_mid)
                        - aggregate_dv_dsoc(&agg_dp_nocong, t, e_mid))
                .collect();
            vec![fleet_delta; n_b]
        } else {
            let exposure: Vec<f64> = (0..n_b)
                .map(|b| expected_premiums.iter().map(|pt| pt[b]).sum::<f64>() / n_t.max(1) as f64)
                .collect();
            let mut sorted_idx: Vec<usize> = (0..n_b).collect();
            sorted_idx.sort_by(|&a, &bi| {
                exposure[a].partial_cmp(&exposure[bi]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cluster_id = vec![0usize; n_b];
            for (rank, &b_idx) in sorted_idx.iter().enumerate() {
                cluster_id[b_idx] = (rank * k) / n_b;
            }
            let mut cluster_deltas: Vec<Vec<f64>> = Vec::with_capacity(k);
            for ck in 0..k {
                let cluster_bats: Vec<Battery> = (0..n_b)
                    .filter(|&b| cluster_id[b] == ck)
                    .map(|b| challenge.batteries[b].clone())
                    .collect();
                if cluster_bats.is_empty() {
                    cluster_deltas.push(vec![0.0_f64; n_t]);
                    continue;
                }
                let cluster_cap: f64 =
                    cluster_bats.iter().map(|b| b.capacity_mwh).sum::<f64>().max(1.0);
                let cluster_da: Vec<f64> = (0..n_t)
                    .map(|t| {
                        cluster_bats
                            .iter()
                            .map(|b| b.capacity_mwh * challenge.market.day_ahead_prices[t][b.node])
                            .sum::<f64>()
                            / cluster_cap
                    })
                    .collect();
                let cluster_premium: Vec<f64> = (0..n_t)
                    .map(|t| {
                        let (num, denom) = (0..n_b)
                            .filter(|&b| cluster_id[b] == ck)
                            .fold((0.0_f64, 0.0_f64), |(p, w), b| {
                                let cap = challenge.batteries[b].capacity_mwh;
                                (p + cap * expected_premiums[t][b], w + cap)
                            });
                        num / denom.max(1.0)
                    })
                    .collect();
                let da_with_cong: Vec<f64> = cluster_da
                    .iter()
                    .zip(cluster_premium.iter())
                    .map(|(da, prem)| da + prem)
                    .collect();
                let agg_dp_cong = build_aggregate_dp(
                    &cluster_bats, &da_with_cong, n_t,
                    sigma, p_jump, mean_pareto, second_pareto, hp.cwv_agg_levels.max(2),
                );
                let agg_dp_nocong = build_aggregate_dp(
                    &cluster_bats, &cluster_da, n_t,
                    sigma, p_jump, mean_pareto, second_pareto, hp.cwv_agg_levels.max(2),
                );
                let e_mid_cluster: f64 = cluster_bats
                    .iter()
                    .map(|b| (b.soc_min_mwh + b.soc_max_mwh) * 0.5)
                    .sum();
                let cluster_delta: Vec<f64> = (0..n_t)
                    .map(|t| aggregate_dv_dsoc(&agg_dp_cong, t, e_mid_cluster)
                            - aggregate_dv_dsoc(&agg_dp_nocong, t, e_mid_cluster))
                    .collect();
                cluster_deltas.push(cluster_delta);
            }
            (0..n_b).map(|b| cluster_deltas[cluster_id[b]].clone()).collect()
        }
    } else {
        vec![vec![0.0_f64; challenge.num_steps]; challenge.num_batteries]
    };

    let zero_solution = Solution {
        schedule: vec![vec![0.0; challenge.num_batteries]; challenge.num_steps],
    };
    save_solution(&zero_solution)?;

    let available = fuel_remaining();
    let reserve = available / 28;
    let max_spend = available.saturating_sub(reserve);
    let target_spend = if hp.fuel_budget == 0 {
        max_spend
    } else {
        hp.fuel_budget.min(max_spend)
    };
    let fuel_floor = available - target_spend;
    let grad_buf = RefCell::new(GradBuffer::new(challenge.num_batteries));
    iter_pool_reset();
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        let mut buf = grad_buf.borrow_mut();
        let result = policy(c, s, &dps, &sens, &coupling_prems, &hp, &delta_cong, cwv_lambda, &mut *buf)?;
        // update previous policy state for warm-start
        {
            let mut guard = prev_policy_store().lock().unwrap();
            *guard = Some((result.clone(), s.socs.clone()));
        }
        Ok(result)
    })?;
    save_solution(&solution)?;
    Ok(())
}

fn lp_dispatch_step(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    hp: &Hyperparameters,
) -> Option<Vec<f64>> {
    let num_b = challenge.num_batteries;
    let n_lines_total = sens.len();
    let t = state.time_step;

    let line_indices: Vec<usize> = if hp.lp_max_lines > 0 && hp.lp_max_lines < n_lines_total {
        let limits = &challenge.network.flow_limits;
        let mut scored: Vec<(f64, usize)> = (0..n_lines_total)
            .map(|l| {
                let lim = limits[l];
                let ratio = if lim > 1e-6 { base_flows[l].abs() / lim } else { 0.0 };
                (ratio, l)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(hp.lp_max_lines);
        scored.into_iter().map(|(_, l)| l).collect()
    } else {
        (0..n_lines_total).collect()
    };

    let num_l = line_indices.len();
    let limits = &challenge.network.flow_limits;

    let n = 2 * num_b;
    let m = 4 * num_b + 2 * num_l;

    let mut c_obj = vec![0.0_f64; n];
    let mut a_mat = vec![vec![0.0_f64; n]; m];
    let mut b_vec = vec![0.0_f64; m];

    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let price = state.rt_prices[battery.node];
        let soc = state.socs[b];
        let dv = dv_dsoc(&dps[b], t, soc);

        c_obj[b] = (price - KAPPA_TX) * DELTA_T - dv * (DELTA_T / ETA_DISCHARGE);
        c_obj[num_b + b] = dv * ETA_CHARGE * DELTA_T - (price + KAPPA_TX) * DELTA_T;

        let (u_min, u_max) = state.action_bounds[b];
        let r = 4 * b;
        a_mat[r][b] = 1.0;
        b_vec[r] = u_max.max(0.0);
        a_mat[r + 1][num_b + b] = 1.0;
        b_vec[r + 1] = (-u_min).max(0.0);
        a_mat[r + 2][b] = DELTA_T / ETA_DISCHARGE;
        b_vec[r + 2] = (soc - battery.soc_min_mwh).max(0.0);
        a_mat[r + 3][num_b + b] = ETA_CHARGE * DELTA_T;
        b_vec[r + 3] = (battery.soc_max_mwh - soc).max(0.0);
    }

    let row_f = 4 * num_b;
    for (li, &l) in line_indices.iter().enumerate() {
        let limit = limits[l];
        let exo = base_flows[l];
        let rp = row_f + 2 * li;
        let rn = rp + 1;
        for b in 0..num_b {
            let ptdf = sens[l][b];
            a_mat[rp][b] += ptdf;
            a_mat[rp][num_b + b] -= ptdf;
            a_mat[rn][b] -= ptdf;
            a_mat[rn][num_b + b] += ptdf;
        }
        b_vec[rp] = (limit - exo).max(0.0);
        b_vec[rn] = (limit + exo).max(0.0);
    }

    let budget = if hp.lp_pivot_budget > 0 { hp.lp_pivot_budget } else { 2000 };
    let (opt_x, _) = lp_solver::lp_solve_with_budget(n, m, &c_obj, &a_mat, &b_vec, budget);
    let opt_x = opt_x?;

    let mut actions = vec![0.0_f64; num_b];
    for b in 0..num_b {
        let d = opt_x[b];
        let c = opt_x[num_b + b];
        let u = d - c;
        let (lo, hi) = state.action_bounds[b];
        actions[b] = u.clamp(lo, hi);
    }
    Some(actions)
}

mod lp_solver {
    const LP_EPS: f64 = 1e-9;

    pub fn lp_solve_with_budget(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize,
    ) -> (Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6) {
            return (None, 0);
        }

        let n_vars = n + m;
        let rhs_col = n_vars;
        let n_cols = n_vars + 1;

        let mut tab = vec![vec![0.0_f64; n_cols]; m + 1];
        for i in 0..m {
            for j in 0..n {
                tab[i][j] = a[i][j];
            }
            tab[i][n + i] = 1.0;
            tab[i][rhs_col] = b[i].max(0.0);
        }
        for j in 0..n {
            tab[m][j] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();
        let mut pivots_used = 0usize;

        for pivot in 0..max_pivots {
            pivots_used = pivot + 1;
            let entering = match (0..n_vars).find(|&j| tab[m][j] < -LP_EPS) {
                Some(j) => j,
                None => break,
            };
            let leaving_row = (0..m)
                .filter(|&i| tab[i][entering] > LP_EPS)
                .min_by(|&i1, &i2| {
                    let r1 = tab[i1][rhs_col] / tab[i1][entering];
                    let r2 = tab[i2][rhs_col] / tab[i2][entering];
                    r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
                });
            let leaving_row = match leaving_row {
                Some(r) => r,
                None => return (None, 0),
            };

            let pivot_val = tab[leaving_row][entering];
            if pivot_val.abs() < LP_EPS {
                return (None, 0);
            }
            for j in 0..n_cols {
                tab[leaving_row][j] /= pivot_val;
            }
            for i in 0..=m {
                if i != leaving_row {
                    let factor = tab[i][entering];
                    if factor.abs() > 1e-15 {
                        for j in 0..n_cols {
                            tab[i][j] -= factor * tab[leaving_row][j];
                        }
                    }
                }
            }
            basis[leaving_row] = entering;
        }

        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i][rhs_col].max(0.0);
            }
        }
        (Some(x), pivots_used)
    }
}