
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Mutex, OnceLock};
use tig_challenges::energy_arbitrage::*;
use tig_challenges::energy_arbitrage::constants::{
    DELTA_T, EPS_FLOW, ETA_CHARGE, ETA_DISCHARGE, KAPPA_DEG, KAPPA_TX,
};

const EPS: f64 = 1e-12;

const LP_EPS: f64 = 1e-9;

const RH_KKT_ITERS: usize = 5;
const RH_KKT_ALPHA0: f64 = 0.5;
const RH_KKT_ALPHA_MAX: f64 = 0.3;
const RH_BINDING_RATIO: f64 = 0.7;
const N_4VAR: usize = 4;
const M_4VAR: usize = 8;
const NV_4VAR: usize = N_4VAR + M_4VAR;
const NC_4VAR: usize = NV_4VAR + 1;

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
    #[serde(default)]
    pub use_bb_clamps: bool,
    #[serde(default)]
    pub use_momentum: bool,

    #[serde(default)]
    pub anticipate_lmp: bool,

    pub lmp_threshold: f64,
    pub lmp_premium_scale: f64,
    #[serde(default)]
    pub use_joint_pair_polish: bool,
    #[serde(default = "default_joint_pair_budget")]
    pub joint_pair_budget: usize,
    #[serde(default)]
    pub use_joint_triplet_polish: bool,
    #[serde(default = "default_joint_triplet_budget")]
    pub joint_triplet_budget: usize,

    #[serde(default = "default_joint_triplet_top_k")]
    pub joint_triplet_top_k: usize,

    #[serde(default)]
    pub use_admm_polish: bool,
    #[serde(default)]
    pub use_ejection_chain: bool,

    #[serde(default)]
    pub use_scvc: bool,
    #[serde(default = "default_scvc_alpha")]
    pub scvc_alpha: f64,
    #[serde(default)]
    pub use_rolling_horizon: bool,
    #[serde(default = "default_rh_stride")]
    pub rh_stride: usize,

    #[serde(default)]
    pub soc_ref_lambda: f64,

    #[serde(default = "default_soc_ref_dyn_stride")]
    pub soc_ref_dyn_stride: usize,
    #[serde(default)]
    pub use_cosine_beta: bool,

    #[serde(default = "default_pga_beta_end")]
    pub pga_beta_end: f64,
    #[serde(default)]
    pub use_admm_solver: bool,
    #[serde(default = "default_admm_rho")]
    pub admm_rho: f64,
    #[serde(default = "default_admm_iters")]
    pub admm_iters: usize,

    #[serde(default)]
    pub use_arb_seed: bool,

    #[serde(default)]
    pub arb_pct: u64,

    #[serde(default)]
    pub arb_inverse: bool,

    #[serde(default)]
    pub use_std_arb_third: bool,

    #[serde(default)]
    pub arb_pct_third: u64,
    #[serde(default)]
    pub use_obl_target_seed: bool,
    #[serde(default)]
    pub use_ramp_seed: bool,
    #[serde(default)]
    pub use_da_arb_third: bool,
    #[serde(default)]
    pub use_spread_arb_third: bool,

    #[serde(default)]
    pub use_water_value_third: bool,
    #[serde(default = "default_wv_kappa")]
    pub wv_kappa: f64,

    #[serde(default)]
    pub use_block_seed_third: bool,
    #[serde(default = "default_block_frac")]
    pub block_frac: f64,

    #[serde(default)]
    pub use_rollout_seed_third: bool,

    #[serde(default = "default_rollout_window")]
    pub rollout_window: u64,

    #[serde(default)]
    pub use_rollout_additive: bool,

    #[serde(default)]
    pub use_rollout_additive_2: bool,

    #[serde(default = "default_rollout_window_2")]
    pub rollout_window_2: u64,

    #[serde(default)]
    pub use_rollout_additive_3: bool,

    #[serde(default = "default_rollout_window_3")]
    pub rollout_window_3: u64,

    #[serde(default)]
    pub use_rollout_additive_4: bool,
    #[serde(default = "default_rollout_window_4")]
    pub rollout_window_4: u64,
    #[serde(default)]
    pub use_basin_hop: bool,
    #[serde(default = "default_basin_hop_scale")]
    pub basin_hop_scale: f64,
    #[serde(default = "default_basin_hop_k")]
    pub basin_hop_k: usize,
    #[serde(default)]
    pub pair_lahc_lh: usize,
    #[serde(default)]
    pub lahc_init_alpha_span: f64,

    #[serde(default)]
    pub use_lp_dispatch: bool,
    #[serde(default)]
    pub lp_max_lines: usize,
    #[serde(default)]
    pub lp_pivot_budget: usize,

    #[serde(default)]
    pub lp_flat_tableau: bool,

    #[serde(default)]
    pub lp_dantzig: bool,
    #[serde(default)]
    pub lp_drop_soc_rows: bool,
    #[serde(default)]
    pub lp_screen_mode: usize,
    #[serde(default)]
    pub lp_dv_mode: usize,
}

fn default_joint_pair_budget() -> usize {
    780
}

fn default_joint_triplet_budget() -> usize {
    150
}

fn default_joint_triplet_top_k() -> usize {
    15
}

fn default_scvc_alpha() -> f64 {
    0.5
}

fn default_rh_stride() -> usize {
    1
}

fn default_soc_ref_dyn_stride() -> usize {
    3
}

fn default_pga_beta_end() -> f64 {
    0.7
}

fn default_admm_rho() -> f64 {
    0.45
}

fn default_admm_iters() -> usize {
    9
}

fn default_wv_kappa() -> f64 {
    0.25
}

fn default_block_frac() -> f64 {
    0.25
}

fn default_rollout_window() -> u64 {
    12
}

fn default_rollout_window_2() -> u64 {
    4
}

fn default_rollout_window_3() -> u64 {
    2
}

fn default_rollout_window_4() -> u64 {
    1
}

fn default_basin_hop_scale() -> f64 {
    0.05
}

fn default_basin_hop_k() -> usize {
    4
}

fn default_pair_lahc_lh() -> usize {
    0
}

fn default_lahc_init_alpha_span() -> f64 {
    0.0
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            dp_soc_levels: 97,
            dp_action_levels: 17,
            policy_action_levels: 65,
            proj_max_iters: 80,
            grad_outer_iters: 100,
            grad_ls_iters: 6,
            bisect_iters: 30,
            coord_polish_passes: 2,
            lookahead_horizon: 24,
            fuel_budget: 0,
            use_bb_clamps: false,
            use_momentum: false,
            anticipate_lmp: false,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.0,
            use_joint_pair_polish: false,
            joint_pair_budget: 780,
            use_joint_triplet_polish: false,
            joint_triplet_budget: 150,
            joint_triplet_top_k: 15,
            use_admm_polish: false,
            use_ejection_chain: false,
            use_scvc: false,
            scvc_alpha: 0.5,
            use_rolling_horizon: false,
            rh_stride: 1,
            soc_ref_lambda: 0.0,
            soc_ref_dyn_stride: 3,
            use_cosine_beta: false,
            pga_beta_end: 0.7,
            use_admm_solver: false,
            admm_rho: 0.45,
            admm_iters: 9,
            use_arb_seed: false,
            arb_pct: 0,
            arb_inverse: false,
            use_std_arb_third: false,
            arb_pct_third: 0,
            use_obl_target_seed: false,
            use_ramp_seed: false,
            use_da_arb_third: false,
            use_spread_arb_third: false,
            use_water_value_third: false,
            wv_kappa: default_wv_kappa(),
            use_block_seed_third: false,
            block_frac: default_block_frac(),
            use_rollout_seed_third: false,
            rollout_window: default_rollout_window(),
            use_rollout_additive: false,
            use_rollout_additive_2: false,
            rollout_window_2: default_rollout_window_2(),
            use_rollout_additive_3: false,
            rollout_window_3: default_rollout_window_3(),
            use_rollout_additive_4: false,
            rollout_window_4: default_rollout_window_4(),
            use_basin_hop: false,
            basin_hop_scale: default_basin_hop_scale(),
            basin_hop_k: default_basin_hop_k(),
            pair_lahc_lh: 0,
            lahc_init_alpha_span: 0.0,
            use_lp_dispatch: false,
            lp_max_lines: 0,
            lp_pivot_budget: 0,
            lp_flat_tableau: false,
            lp_dantzig: false,
            lp_drop_soc_rows: false,
            lp_screen_mode: 0,
            lp_dv_mode: 0,
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
        hp.rh_stride = hp.rh_stride.max(1);
        hp.soc_ref_dyn_stride = hp.soc_ref_dyn_stride.max(1);
        if !(hp.admm_rho > 0.0) {
            hp.admm_rho = 0.45;
        }
        hp.admm_iters = hp.admm_iters.max(1);
        hp.lp_screen_mode = hp.lp_screen_mode.min(1);
        hp.lp_dv_mode = hp.lp_dv_mode.min(1);
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

fn edge_sized_fraction(edge: f64, price_band: f64) -> f64 {
    if edge <= 0.0 {
        0.0
    } else {
        let normalized = edge / price_band.max(5.0);
        (0.35 + 0.65 * normalized).clamp(0.35, 1.0)
    }
}

fn relative_soc_pressure(battery: &Battery, soc: f64) -> f64 {
    let span = (battery.soc_max_mwh - battery.soc_min_mwh).max(1e-9);
    ((soc - battery.soc_min_mwh) / span).clamp(0.0, 1.0)
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


static SOC_REF: OnceLock<Mutex<Vec<Vec<f64>>>> = OnceLock::new();
fn soc_ref_lock() -> &'static Mutex<Vec<Vec<f64>>> {
    SOC_REF.get_or_init(|| Mutex::new(Vec::new()))
}

fn compute_soc_reference_dynamic(
    challenge: &Challenge,
    current_socs: &[f64],
    residual_shift: &[f64],
    start_t: usize,
) -> Vec<Vec<f64>> {
    let n_steps = challenge.num_steps;
    let n_batt = challenge.num_batteries;
    let mut refs = vec![vec![0.0_f64; n_steps + 1]; n_batt];
    for b in 0..n_batt {
        if start_t >= n_steps {
            continue;
        }
        refs[b][start_t] = current_socs[b];
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let shift = residual_shift.get(node).copied().unwrap_or(0.0);
        let mut da: Vec<f64> = (start_t..n_steps)
            .map(|t| challenge.market.day_ahead_prices[t][node] + shift)
            .collect();
        da.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p25 = da[da.len() / 4];
        let p75 = da[(da.len() * 3) / 4];
        for t in start_t..n_steps {
            let soc = refs[b][t];
            let price = challenge.market.day_ahead_prices[t][node] + shift;
            let delta_soc = if price > p75 {
                let max_disch = (soc - battery.soc_min_mwh).max(0.0);
                let disch_mwh = (battery.power_discharge_mw * DELTA_T / ETA_DISCHARGE).min(max_disch);
                -disch_mwh
            } else if price < p25 {
                let max_chg = (battery.soc_max_mwh - soc).max(0.0);
                let chg_mwh = (battery.power_charge_mw * DELTA_T * ETA_CHARGE).min(max_chg);
                chg_mwh
            } else {
                0.0
            };
            refs[b][t + 1] = (soc + delta_soc).clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        }
    }
    refs
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

        for s_idx in 0..levels {
            let (lo, hi) = bounds[s_idx];
            let soc = soc_lo + soc_step * s_idx as f64;

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

fn dv_dsoc(dp: &BatteryDP, t: usize, soc: f64) -> f64 {
    let next_t = (t + 1).min(dp.values.len() - 1);
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


fn scvc_greedy_trajectory(dp: &BatteryDP, battery: &Battery, da_at_node: &[f64]) -> Vec<f64> {
    let num_steps = dp.values.len().saturating_sub(1);
    let soc_lo = dp.soc_lo;
    let soc_step_inv = dp.soc_step_inv;
    let last = dp.levels.saturating_sub(1);
    let soc_span = if last > 0 { last as f64 / soc_step_inv } else { 0.0 };
    let mut soc = soc_lo + soc_span * 0.5;

    let mut traj = Vec::with_capacity(num_steps + 1);
    traj.push(soc);

    for t in 0..num_steps {
        let da = da_at_node.get(t).copied().unwrap_or(0.0);
        let (lo, hi) = compute_action_bounds(battery, soc);
        let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
        let friction = 2.0 * KAPPA_TX;
        let charge_max = da * eta_rt - friction;
        let discharge_min = da / eta_rt + friction;
        let grid = adaptive_action_grid(battery, charge_max, discharge_min, da, 9);

        let mut best_a = 0.0_f64.clamp(lo, hi);
        {
            let nxt = battery.apply_action_to_soc(best_a, soc);
            let mut best_v = immediate_profit(battery, best_a, da)
                + interp_value(&dp.values[t + 1], nxt, soc_lo, soc_step_inv, last);
            for raw in grid {
                let a = raw.clamp(lo, hi);
                let next_soc = battery.apply_action_to_soc(a, soc);
                let v = immediate_profit(battery, a, da)
                    + interp_value(&dp.values[t + 1], next_soc, soc_lo, soc_step_inv, last);
                if v > best_v + EPS {
                    best_v = v;
                    best_a = a;
                }
            }
        }
        soc = battery.apply_action_to_soc(best_a, soc);
        traj.push(soc);
    }
    traj
}

fn apply_scvc_to_dp(dp: &mut BatteryDP, battery: &Battery, da_at_node: &[f64], alpha: f64) {
    let num_steps = dp.values.len().saturating_sub(1);
    if num_steps < 2 || dp.levels < 2 {
        return;
    }
    let soc_lo = dp.soc_lo;
    let soc_step = 1.0 / dp.soc_step_inv;
    let levels = dp.levels;

    let traj = scvc_greedy_trajectory(dp, battery, da_at_node);

    let marge_b = da_at_node.iter().take(num_steps).map(|p| p.abs()).sum::<f64>()
        / num_steps as f64;
    if marge_b < EPS {
        return;
    }

    let slope = alpha * marge_b;
    for t in 1..num_steps {
        let soc_ref = traj[t];
        let vals = &mut dp.values[t];
        for s_idx in 0..levels {
            let soc_s = soc_lo + soc_step * s_idx as f64;
            vals[s_idx] += slope * (soc_s - soc_ref);
        }
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
    let friction = 2.0 * KAPPA_TX;
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



#[inline(always)]
fn solve_battery_kkt(
    f0: f64, g0: f64, f1: f64, g1: f64,
    ub_d0: f64, ub_c0: f64, ub_d1: f64, ub_c1: f64,
    avail: f64, head: f64, d_f: f64, c_f: f64,
) -> [f64; N_4VAR] {
    let mut tab = [[0.0_f64; NC_4VAR]; M_4VAR + 1];

    tab[0][0] = 1.0; tab[0][N_4VAR] = 1.0; tab[0][NV_4VAR] = ub_d0;
    tab[1][1] = 1.0; tab[1][N_4VAR + 1] = 1.0; tab[1][NV_4VAR] = ub_c0;
    tab[2][2] = 1.0; tab[2][N_4VAR + 2] = 1.0; tab[2][NV_4VAR] = ub_d1;
    tab[3][3] = 1.0; tab[3][N_4VAR + 3] = 1.0; tab[3][NV_4VAR] = ub_c1;
    tab[4][0] = d_f; tab[4][1] = -c_f; tab[4][N_4VAR + 4] = 1.0; tab[4][NV_4VAR] = avail;
    tab[5][0] = -d_f; tab[5][1] = c_f; tab[5][N_4VAR + 5] = 1.0; tab[5][NV_4VAR] = head;
    tab[6][0] = d_f; tab[6][1] = -c_f; tab[6][2] = d_f; tab[6][3] = -c_f;
    tab[6][N_4VAR + 6] = 1.0; tab[6][NV_4VAR] = avail;
    tab[7][0] = -d_f; tab[7][1] = c_f; tab[7][2] = -d_f; tab[7][3] = c_f;
    tab[7][N_4VAR + 7] = 1.0; tab[7][NV_4VAR] = head;
    tab[M_4VAR][0] = -f0; tab[M_4VAR][1] = -g0;
    tab[M_4VAR][2] = -f1; tab[M_4VAR][3] = -g1;

    let mut basis = [
        N_4VAR, N_4VAR + 1, N_4VAR + 2, N_4VAR + 3,
        N_4VAR + 4, N_4VAR + 5, N_4VAR + 6, N_4VAR + 7,
    ];

    for _ in 0..(3 * N_4VAR + 2) {
        let mut entering = NV_4VAR;
        let mut min_c = -LP_EPS;
        for j in 0..NV_4VAR {
            if tab[M_4VAR][j] < min_c {
                min_c = tab[M_4VAR][j];
                entering = j;
            }
        }
        if entering == NV_4VAR { break; }

        let mut leaving = M_4VAR;
        let mut min_r = f64::MAX;
        for i in 0..M_4VAR {
            if tab[i][entering] > LP_EPS {
                let r = tab[i][NV_4VAR] / tab[i][entering];
                if r < min_r { min_r = r; leaving = i; }
            }
        }
        if leaving == M_4VAR { break; }

        let pv = tab[leaving][entering];
        for j in 0..NC_4VAR { tab[leaving][j] /= pv; }
        for i in 0..=M_4VAR {
            if i != leaving {
                let f = tab[i][entering];
                if f.abs() > 1e-15 {
                    for j in 0..NC_4VAR { tab[i][j] -= f * tab[leaving][j]; }
                }
            }
        }
        basis[leaving] = entering;
    }

    let mut sol = [0.0_f64; N_4VAR];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < N_4VAR { sol[bv] = tab[i][NV_4VAR].max(0.0); }
    }
    sol
}


fn rolling_horizon_lp_seed(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    lam_warm: &mut (Vec<f64>, Vec<f64>),
) -> Option<Vec<f64>> {
    let t = state.time_step;
    if t + 1 >= challenge.num_steps { return None; }
    let num_b = challenge.num_batteries;
    let limits = &challenge.network.flow_limits;
    let dt = DELTA_T;
    let d_f = dt / ETA_DISCHARGE;
    let c_f = ETA_CHARGE * dt;

    let mut f0 = vec![0.0_f64; num_b];
    let mut g0 = vec![0.0_f64; num_b];
    let mut f1 = vec![0.0_f64; num_b];
    let mut g1 = vec![0.0_f64; num_b];
    let mut available = vec![0.0_f64; num_b];
    let mut headroom = vec![0.0_f64; num_b];
    let mut ub_d0 = vec![0.0_f64; num_b];
    let mut ub_c0 = vec![0.0_f64; num_b];
    let mut ub_d1 = vec![0.0_f64; num_b];
    let mut ub_c1 = vec![0.0_f64; num_b];

    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let p0 = state.rt_prices[node];
        let p1 = challenge.market.day_ahead_prices[t + 1][node];
        let dv2 = if t + 1 < dps[b].values.len() {
            dv_dsoc(&dps[b], t + 1, state.socs[b])
        } else {
            0.0
        };
        f0[b] = (p0 - KAPPA_TX) * dt;
        g0[b] = -(p0 + KAPPA_TX) * dt;
        f1[b] = (p1 - KAPPA_TX) * dt - dv2 * dt / ETA_DISCHARGE;
        g1[b] = -(p1 + KAPPA_TX) * dt + dv2 * ETA_CHARGE * dt;
        let soc0 = state.socs[b];
        available[b] = (soc0 - battery.soc_min_mwh).max(0.0);
        headroom[b] = (battery.soc_max_mwh - soc0).max(0.0);
        ub_d0[b] = state.action_bounds[b].1.max(0.0);
        ub_c0[b] = (-state.action_bounds[b].0).max(0.0);
        ub_d1[b] = battery.power_discharge_mw;
        ub_c1[b] = battery.power_charge_mw;
    }

    let binding_lines: Vec<usize> = limits.iter().enumerate()
        .filter(|&(l, &lim)| {
            lim > 1e-6
                && base_flows.get(l).copied().unwrap_or(0.0).abs() / lim > RH_BINDING_RATIO
        })
        .map(|(l, _)| l)
        .collect();
    let n_binding = binding_lines.len();

    let mut lam_fwd: Vec<f64> = binding_lines.iter()
        .map(|&l| lam_warm.0.get(l).copied().unwrap_or(0.0))
        .collect();
    let mut lam_rev: Vec<f64> = binding_lines.iter()
        .map(|&l| lam_warm.1.get(l).copied().unwrap_or(0.0))
        .collect();

    let mut d0_sol = vec![0.0_f64; num_b];
    let mut c0_sol = vec![0.0_f64; num_b];

    for iter in 0..RH_KKT_ITERS {
        let alpha = (RH_KKT_ALPHA0 / ((iter + 1) as f64).sqrt()).min(RH_KKT_ALPHA_MAX);

        for b in 0..num_b {
            let ptdf_adj: f64 = binding_lines.iter().enumerate()
                .map(|(k, &l)| {
                    let s = sens.get(l).and_then(|r| r.get(b)).copied().unwrap_or(0.0);
                    (lam_fwd[k] - lam_rev[k]) * s
                })
                .sum();
            let sol = solve_battery_kkt(
                f0[b] - ptdf_adj, g0[b] + ptdf_adj,
                f1[b], g1[b],
                ub_d0[b], ub_c0[b], ub_d1[b], ub_c1[b],
                available[b], headroom[b], d_f, c_f,
            );
            d0_sol[b] = sol[0];
            c0_sol[b] = sol[1];
        }

        for (k, &l) in binding_lines.iter().enumerate() {
            let lim = limits[l];
            let bf = base_flows.get(l).copied().unwrap_or(0.0);
            let net_flow: f64 = sens[l].iter().zip(d0_sol.iter().zip(c0_sol.iter()))
                .map(|(&s, (&d, &c))| s * (d - c))
                .sum();
            let b_fwd = (lim - bf).max(0.0);
            let b_rev = (lim + bf).max(0.0);
            lam_fwd[k] = (lam_fwd[k] + alpha * (net_flow - b_fwd)).max(0.0);
            lam_rev[k] = (lam_rev[k] + alpha * (-net_flow - b_rev)).max(0.0);
        }
    }

    for (k, &l) in binding_lines.iter().enumerate() {
        if l < lam_warm.0.len() {
            lam_warm.0[l] = lam_fwd[k];
            lam_warm.1[l] = lam_rev[k];
        }
    }

    for b in 0..num_b {
        let ptdf_adj: f64 = binding_lines.iter().enumerate()
            .map(|(k, &l)| {
                let s = sens.get(l).and_then(|r| r.get(b)).copied().unwrap_or(0.0);
                (lam_fwd[k] - lam_rev[k]) * s
            })
            .sum();
        let sol = solve_battery_kkt(
            f0[b] - ptdf_adj, g0[b] + ptdf_adj,
            f1[b], g1[b],
            ub_d0[b], ub_c0[b], ub_d1[b], ub_c1[b],
            available[b], headroom[b], d_f, c_f,
        );
        d0_sol[b] = sol[0];
        c0_sol[b] = sol[1];
    }

    let actions: Vec<f64> = (0..num_b).map(|b| d0_sol[b] - c0_sol[b]).collect();
    Some(actions)
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
    hp: &Hyperparameters,
) -> Vec<f64> {
    let soc_ref_snapshot: Option<Vec<Vec<f64>>> = if hp.soc_ref_lambda > 0.0 {
        soc_ref_lock().lock().ok().map(|g| g.clone()).filter(|g| !g.is_empty())
    } else {
        None
    };
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

        if let Some(ref refs) = soc_ref_snapshot {
            if b < refs.len() {
                let t1 = (state.time_step + 1).min(refs[b].len().saturating_sub(1));
                let soc_ref_t1 = refs[b][t1];
                grad[b] -= hp.soc_ref_lambda * (next_soc - soc_ref_t1) * dsoc_du;
            }
        }
    }
    grad
}

fn admm_solver(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    seed: Vec<f64>,
    hp: &Hyperparameters,
) -> (Vec<f64>, f64) {
    let n = seed.len();
    let rho = hp.admm_rho;

    let h_diag: Vec<f64> = (0..n)
        .map(|b| {
            let cap2 = challenge.batteries[b].capacity_mwh.powi(2).max(1e-9);
            2.0 * KAPPA_DEG * DELTA_T * DELTA_T / cap2
        })
        .collect();

    let mut z = seed;
    safe_project_to_feasible(challenge, state, &mut z, sens, base_flows, hp);
    let mut x = z.clone();
    let mut w = vec![0.0_f64; n];

    let mut best_action = z.clone();
    let mut best_value = total_step_value(challenge, state, dps, &best_action);

    for _ in 0..hp.admm_iters {
        let g = analytic_gradient(challenge, state, dps, &x, hp);
        for b in 0..n {
            x[b] = (g[b] + h_diag[b] * x[b] + rho * (z[b] - w[b])) / (h_diag[b] + rho);
        }
        clamp_to_bounds(&mut x, &state.action_bounds);

        let mut z_new: Vec<f64> = x.iter().zip(w.iter()).map(|(xi, wi)| xi + wi).collect();
        safe_project_to_feasible(challenge, state, &mut z_new, sens, base_flows, hp);
        z = z_new;

        for b in 0..n {
            w[b] += x[b] - z[b];
        }

        let v = total_step_value(challenge, state, dps, &z);
        if v > best_value {
            best_value = v;
            best_action = z.clone();
        }
    }
    (best_action, best_value)
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
    if hp.use_admm_solver {
        return admm_solver(challenge, state, dps, sens, base_flows, seed, hp);
    }
    let mut action = seed;
    safe_project_to_feasible(challenge, state, &mut action, sens, base_flows, hp);
    let mut best_value = total_step_value(challenge, state, dps, &action);
    let mut best_action = action.clone();

    let max_power: f64 = challenge
        .batteries
        .iter()
        .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
        .fold(1.0_f64, f64::max);


    const LR_GROWTH_CAP: f64 = 1.05;
    const BB_DECAY_FACTOR: f64 = 0.85;
    const MOMENTUM_BETA: f64 = 0.99;
    let t_max = hp.grad_outer_iters.saturating_sub(1).max(1) as f64;

    let mut lr = max_power * 0.5;
    let mut velocity = vec![0.0_f64; action.len()];
    for outer_iter in 0..hp.grad_outer_iters {
        let beta = if hp.use_cosine_beta {
            let frac = outer_iter as f64 / t_max;
            hp.pga_beta_end + (MOMENTUM_BETA - hp.pga_beta_end) * (1.0 + (std::f64::consts::PI * frac).cos()) * 0.5
        } else {
            MOMENTUM_BETA
        };

        let grad = analytic_gradient(challenge, state, dps, &action, hp);
        let g_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm < 1e-9 {
            break;
        }

        let dir: Vec<f64> = if hp.use_momentum {
            grad.iter()
                .zip(velocity.iter())
                .map(|(g, v)| beta * v + g)
                .collect()
        } else {
            grad.clone()
        };

        let prev_lr = lr;
        let mut improved = false;
        let mut cur_lr = lr;
        for _ in 0..hp.grad_ls_iters {
            let step_scale = cur_lr / g_norm;
            let mut trial: Vec<f64> = action
                .iter()
                .zip(dir.iter())
                .map(|(a, d)| a + step_scale * d)
                .collect();
            safe_project_to_feasible(challenge, state, &mut trial, sens, base_flows, hp);
            let v = total_step_value(challenge, state, dps, &trial);
            if v > best_value + 1e-9 {
                action = trial.clone();
                best_value = v;
                best_action = trial;
                improved = true;
                lr = if hp.use_bb_clamps {
                    (cur_lr * 1.4).min(prev_lr * LR_GROWTH_CAP)
                } else {
                    cur_lr * 1.4
                };
                if hp.use_momentum {
                    for (vel, g) in velocity.iter_mut().zip(grad.iter()) {
                        *vel = beta * *vel + g;
                    }
                }
                break;
            }
            cur_lr *= 0.5;
        }
        if !improved {
            lr = if hp.use_bb_clamps {
                (lr * 0.4).max(prev_lr * BB_DECAY_FACTOR)
            } else {
                lr * 0.4
            };
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

#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn basin_hop_restart(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    actions: &mut Vec<f64>,
    hp: &Hyperparameters,
) {
    let num_b = challenge.num_batteries;
    if num_b == 0 {
        return;
    }

    let nonce_u64 = u64::from_le_bytes([
        challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
        challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
    ]);
    let mut rng: u64 = nonce_u64
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_mul(0xbf58476d1ce4e5b9)
        ^ ((state.time_step as u64).wrapping_mul(0x94d049bb133111eb));

    let k = hp.basin_hop_k.min(num_b);
    let mut spans: Vec<(f64, usize)> = (0..num_b)
        .map(|b| {
            let (lo, hi) = state.action_bounds[b];
            (hi - lo, b)
        })
        .collect();
    spans.sort_unstable_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let pre_hop = actions.clone();
    let current_val = total_step_value(challenge, state, dps, actions);
    let scale = hp.basin_hop_scale;

    for &(span_b, b) in spans.iter().take(k) {
        if span_b < 1e-9 {
            continue;
        }
        let (lo, hi) = state.action_bounds[b];
        let u1_raw = splitmix64(&mut rng);
        let u2_raw = splitmix64(&mut rng);
        let u1 = (u1_raw as f64 + 0.5) / 18446744073709551616.0_f64;
        let u2 = (u2_raw as f64 + 0.5) / 18446744073709551616.0_f64;
        let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        actions[b] = (actions[b] + z * scale * span_b).clamp(lo, hi);
    }

    joint_pair_polish(challenge, state, dps, sens, base_flows, actions, hp);

    if !is_flow_feasible(challenge, state, actions) {
        *actions = pre_hop;
        return;
    }
    let new_val = total_step_value(challenge, state, dps, actions);
    if new_val <= current_val + 1e-9 {
        *actions = pre_hop;
    }
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
    let lh = hp.pair_lahc_lh;
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

    let mut lahc_hist: Vec<f64> = Vec::new();
    let mut lahc_step: usize = 0;
    let mut current_total: f64 = 0.0;
    let mut best_total: f64 = f64::NEG_INFINITY;
    let mut best_actions: Vec<f64> = Vec::new();
    if lh > 0 {
        current_total = total_step_value(challenge, state, dps, actions);
        best_total = current_total;
        best_actions = actions.clone();
        let span = hp.lahc_init_alpha_span;
        if span > 0.0 {
            let nonce_u64 = u64::from_le_bytes([
                challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
                challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
            ]);
            let mut rng_lahc: u64 = nonce_u64
                .wrapping_add(0x6c62272e07bb0142)
                .wrapping_mul(0xbf58476d1ce4e5b9)
                ^ ((state.time_step as u64).wrapping_mul(0x94d049bb133111eb));
            lahc_hist = (0..lh).map(|_| {
                let u = (splitmix64(&mut rng_lahc) as f64 + 0.5) / 18446744073709551616.0_f64;
                let factor = 1.0 + (2.0 * u - 1.0) * span;
                current_total * factor
            }).collect();
        } else {
            lahc_hist = vec![current_total; lh];
        }
    }

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

                let hist_total = if lh > 0 {
                    let slot = lahc_step % lh;
                    let hv = lahc_hist[slot];
                    lahc_hist[slot] = current_total;
                    lahc_step = lahc_step.wrapping_add(1);
                    hv
                } else {
                    0.0
                };
                let late_pair_thr = base_val + (hist_total - current_total);

                let mut best_val = base_val;
                let mut best_i = cur_i;
                let mut best_j = cur_j;
                let mut late_val = late_pair_thr;
                let mut late_i = cur_i;
                let mut late_j = cur_j;

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
                        if lh > 0 && val > late_val + 1e-9 {
                            late_val = val;
                            late_i = cand_i;
                            late_j = cand_j;
                        }
                    }
                }

                let is_strict = (best_i - cur_i).abs() > EPS || (best_j - cur_j).abs() > EPS;
                let is_late = lh > 0
                    && !is_strict
                    && ((late_i - cur_i).abs() > EPS || (late_j - cur_j).abs() > EPS);

                if is_strict || is_late {
                    let (use_i, use_j, use_val) = if is_strict {
                        (best_i, best_j, best_val)
                    } else {
                        (late_i, late_j, late_val)
                    };
                    let delta_i = use_i - cur_i;
                    let delta_j = use_j - cur_j;
                    actions[i] = use_i;
                    actions[j] = use_j;
                    for l in 0..num_l {
                        flows[l] += sens[l][i] * delta_i + sens[l][j] * delta_j;
                    }
                    if lh > 0 {
                        current_total += use_val - base_val;
                        if current_total > best_total + 1e-12 {
                            best_total = current_total;
                            best_actions = actions.clone();
                        }
                    }
                    improved = true;
                    break 'outer;
                }
            }
        }
    }

    if lh > 0 && !best_actions.is_empty() {
        *actions = best_actions;
    }
}

fn joint_triplet_polish(
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
    if num_b < 3 {
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

    let top_k = hp.joint_triplet_top_k.max(3).min(num_b);
    let mut batt_scores: Vec<(f64, usize)> = (0..num_b)
        .map(|b| (actions[b].abs(), b))
        .collect();
    batt_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let active: Vec<usize> = batt_scores.iter().take(top_k).map(|&(_, b)| b).collect();

    let triplet_budget = hp.joint_triplet_budget.max(1);
    let mut tested = 0usize;
    'outer: for ii in 0..top_k {
        let i = active[ii];
        let batt_i = &challenge.batteries[i];
        let price_i = state.rt_prices[batt_i.node];
        let soc_i = state.socs[i];
        let (lo_i, hi_i) = state.action_bounds[i];
        let span_i = hi_i - lo_i;
        for jj in (ii + 1)..top_k {
            let j = active[jj];
            let batt_j = &challenge.batteries[j];
            let price_j = state.rt_prices[batt_j.node];
            let soc_j = state.socs[j];
            let (lo_j, hi_j) = state.action_bounds[j];
            let span_j = hi_j - lo_j;
            for kk in (jj + 1)..top_k {
                let k = active[kk];
                if tested >= triplet_budget {
                    break 'outer;
                }
                tested += 1;
                let batt_k = &challenge.batteries[k];
                let price_k = state.rt_prices[batt_k.node];
                let soc_k = state.socs[k];
                let (lo_k, hi_k) = state.action_bounds[k];
                let span_k = hi_k - lo_k;
                let cur_i = actions[i];
                let cur_j = actions[j];
                let cur_k = actions[k];
                let base_val =
                    dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cur_i)
                    + dp_action_value(&dps[j], batt_j, t, soc_j, price_j, cur_j)
                    + dp_action_value(&dps[k], batt_k, t, soc_k, price_k, cur_k);
                let mut best_val = base_val;
                let mut best_i = cur_i;
                let mut best_j = cur_j;
                let mut best_k = cur_k;
                for &alpha_ij in &[-0.5_f64, -0.25, 0.25, 0.5] {
                    let cand_i = (cur_i + alpha_ij * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - alpha_ij * span_j).clamp(lo_j, hi_j);
                    let delta_i = cand_i - cur_i;
                    let delta_j = cand_j - cur_j;
                    for &alpha_k in &[-0.25_f64, 0.0, 0.25] {
                        let cand_k = (cur_k + alpha_k * span_k).clamp(lo_k, hi_k);
                        let delta_k = cand_k - cur_k;
                        let mut feasible = true;
                        for l in 0..num_l {
                            let limit = limits[l];
                            if limit <= 1e-6 {
                                continue;
                            }
                            let f_new = flows[l]
                                + sens[l][i] * delta_i
                                + sens[l][j] * delta_j
                                + sens[l][k] * delta_k;
                            if f_new.abs() > limit {
                                feasible = false;
                                break;
                            }
                        }
                        if !feasible {
                            continue;
                        }
                        let val =
                            dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cand_i)
                            + dp_action_value(&dps[j], batt_j, t, soc_j, price_j, cand_j)
                            + dp_action_value(&dps[k], batt_k, t, soc_k, price_k, cand_k);
                        if val > best_val + 1e-9 {
                            best_val = val;
                            best_i = cand_i;
                            best_j = cand_j;
                            best_k = cand_k;
                        }
                    }
                }
                if (best_i - cur_i).abs() > EPS
                    || (best_j - cur_j).abs() > EPS
                    || (best_k - cur_k).abs() > EPS
                {
                    let delta_i = best_i - cur_i;
                    let delta_j = best_j - cur_j;
                    let delta_k = best_k - cur_k;
                    actions[i] = best_i;
                    actions[j] = best_j;
                    actions[k] = best_k;
                    for l in 0..num_l {
                        flows[l] += sens[l][i] * delta_i
                            + sens[l][j] * delta_j
                            + sens[l][k] * delta_k;
                    }
                }
            }
        }
    }
}


fn joint_ejection_chain_polish(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    actions: &mut Vec<f64>,
    hp: &Hyperparameters,
) {
    const EJECTION_MAX_DEPTH: usize = 3;
    const EJECTION_BUDGET: usize = 24;
    const EJECTION_ALPHAS: [f64; 4] = [-0.5, -0.25, 0.25, 0.5];

    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b < 3 {
        return;
    }
    let t = state.time_step;
    let limits = &challenge.network.flow_limits;

    let top_k = hp.joint_triplet_top_k.max(3).min(num_b);
    let mut batt_scores: Vec<(f64, usize)> =
        (0..num_b).map(|b| (actions[b].abs(), b)).collect();
    batt_scores
        .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let active: Vec<usize> = batt_scores.iter().take(top_k).map(|&(_, b)| b).collect();

    let mut work = actions.clone();
    let mut flows = vec![0.0_f64; num_l];
    for l in 0..num_l {
        let mut f = base_flows[l];
        for b in 0..num_b {
            f += sens[l][b] * work[b];
        }
        flows[l] = f;
    }

    let dp_val = |b: usize, u: f64| -> f64 {
        let battery = &challenge.batteries[b];
        dp_action_value(&dps[b], battery, t, state.socs[b], state.rt_prices[battery.node], u)
    };

    let mut chains_tested = 0usize;
    'seeds: for &seed in &active {
        for &eject_alpha in &EJECTION_ALPHAS {
            if chains_tested >= EJECTION_BUDGET {
                break 'seeds;
            }
            chains_tested += 1;

            let mut chain_act = work.clone();
            let mut chain_flows = flows.clone();
            let start_val = total_step_value(challenge, state, dps, &chain_act);
            let mut in_chain = vec![false; num_b];

            let (lo_s, hi_s) = state.action_bounds[seed];
            let span_s = hi_s - lo_s;
            let cand_s = (chain_act[seed] + eject_alpha * span_s).clamp(lo_s, hi_s);
            let delta_s = cand_s - chain_act[seed];
            if delta_s.abs() < EPS {
                continue;
            }
            let mut chain_val = start_val + (dp_val(seed, cand_s) - dp_val(seed, chain_act[seed]));
            for l in 0..num_l {
                chain_flows[l] += sens[l][seed] * delta_s;
            }
            chain_act[seed] = cand_s;
            in_chain[seed] = true;

            let mut best_val = start_val;
            let mut best_snapshot: Option<Vec<f64>> = None;

            for _depth in 1..EJECTION_MAX_DEPTH {
                let mut best_gain_val = f64::NEG_INFINITY;
                let mut best_m = usize::MAX;
                let mut best_cand_m = 0.0_f64;
                for &m in &active {
                    if in_chain[m] {
                        continue;
                    }
                    let (lo_m, hi_m) = state.action_bounds[m];
                    let span_m = hi_m - lo_m;
                    let cur_m = chain_act[m];
                    let base_m = dp_val(m, cur_m);
                    for &alpha_m in &EJECTION_ALPHAS {
                        let cand_m = (cur_m + alpha_m * span_m).clamp(lo_m, hi_m);
                        let delta_m = cand_m - cur_m;
                        if delta_m.abs() < EPS {
                            continue;
                        }
                        let mut feasible = true;
                        for l in 0..num_l {
                            let limit = limits[l];
                            if limit <= 1e-6 {
                                continue;
                            }
                            if (chain_flows[l] + sens[l][m] * delta_m).abs() > limit {
                                feasible = false;
                                break;
                            }
                        }
                        if !feasible {
                            continue;
                        }
                        let cand_val = chain_val + (dp_val(m, cand_m) - base_m);
                        if cand_val > best_gain_val {
                            best_gain_val = cand_val;
                            best_m = m;
                            best_cand_m = cand_m;
                        }
                    }
                }
                if best_m == usize::MAX {
                    break;
                }
                let delta_m = best_cand_m - chain_act[best_m];
                for l in 0..num_l {
                    chain_flows[l] += sens[l][best_m] * delta_m;
                }
                chain_act[best_m] = best_cand_m;
                chain_val = best_gain_val;
                in_chain[best_m] = true;

                if chain_val > best_val + 1e-9 {
                    let mut all_feasible = true;
                    for l in 0..num_l {
                        let limit = limits[l];
                        if limit > 1e-6 && chain_flows[l].abs() > limit + 1e-6 {
                            all_feasible = false;
                            break;
                        }
                    }
                    if all_feasible {
                        best_val = chain_val;
                        best_snapshot = Some(chain_act.clone());
                    }
                }
            }

            if let Some(snap) = best_snapshot {
                work = snap;
                for l in 0..num_l {
                    let mut f = base_flows[l];
                    for b in 0..num_b {
                        f += sens[l][b] * work[b];
                    }
                    flows[l] = f;
                }
            }
        }
    }

    *actions = work;
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


fn admm_consensus_polish(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    actions: &mut Vec<f64>,
) {
    const RHO: f64 = 0.1;
    const ADMM_ITERS: usize = 3;
    const ADMM_PROJ_ITERS: usize = 40;

    let num_b = challenge.num_batteries;
    let limits = &challenge.network.flow_limits;
    let t = state.time_step;

    let init_val = total_step_value(challenge, state, dps, actions);
    let mut best_val = init_val;
    let mut best_z = actions.clone();

    let mut z = actions.clone();
    let mut y = vec![0.0_f64; num_b];

    for _iter in 0..ADMM_ITERS {
        let mut u = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let battery = &challenge.batteries[b];
            let (lo, hi) = state.action_bounds[b];
            let soc = state.socs[b];
            let price = state.rt_prices[battery.node];
            let center = (z[b] - y[b] / RHO).clamp(lo, hi);

            let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
            let friction = 2.0 * KAPPA_TX;
            let charge_max = price * eta_rt - friction;
            let discharge_min = price / eta_rt + friction;

            let mut best_u = center;
            let mut best_v = dp_action_value(&dps[b], battery, t, soc, price, center)
                - (RHO / 2.0) * (center - z[b] + y[b] / RHO).powi(2);

            let fixed = [lo, hi, 0.0_f64.clamp(lo, hi), center,
                         (lo + center) * 0.5, (hi + center) * 0.5];
            for &a_raw in fixed.iter() {
                let a = a_raw.clamp(lo, hi);
                let v = dp_action_value(&dps[b], battery, t, soc, price, a)
                    - (RHO / 2.0) * (a - z[b] + y[b] / RHO).powi(2);
                if v > best_v + 1e-12 {
                    best_v = v;
                    best_u = a;
                }
            }
            for raw in adaptive_action_grid(battery, charge_max, discharge_min, price, 9) {
                let a = raw.clamp(lo, hi);
                let v = dp_action_value(&dps[b], battery, t, soc, price, a)
                    - (RHO / 2.0) * (a - z[b] + y[b] / RHO).powi(2);
                if v > best_v + 1e-12 {
                    best_v = v;
                    best_u = a;
                }
            }
            u[b] = best_u;
        }

        let mut new_z = u.clone();
        project_polytope(&mut new_z, &state.action_bounds, sens, base_flows, limits, ADMM_PROJ_ITERS);
        clamp_to_bounds(&mut new_z, &state.action_bounds);

        for b in 0..num_b {
            y[b] += RHO * (u[b] - new_z[b]);
        }
        z = new_z;

        if is_flow_feasible(challenge, state, &z) {
            let v = total_step_value(challenge, state, dps, &z);
            if v > best_val {
                best_val = v;
                best_z = z.clone();
            }
        }
    }

    if best_val > init_val {
        *actions = best_z;
    }
}

fn policy(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    rh_lam: &Mutex<(Vec<f64>, Vec<f64>)>,
    block_plan: &[Vec<i8>],
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_steps = challenge.num_steps;
    let n_remaining = n_steps.saturating_sub(t);
    if n_remaining == 0 {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let horizon = hp.lookahead_horizon.min(n_remaining);
    let mut target = vec![0.0_f64; challenge.num_batteries];

    let friction = 2.0 * KAPPA_TX;
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

    let mut history = history_lock().lock().unwrap();
    if state.time_step == 0 || history.num_nodes != challenge.network.num_nodes {
        history.num_nodes = challenge.network.num_nodes;
        history.values = vec![Vec::new(); challenge.network.num_nodes];
        history.residuals = vec![Vec::new(); challenge.network.num_nodes];
    }
    let mut rt_bands = vec![None; challenge.network.num_nodes];
    let mut residual_shift = vec![0.0_f64; challenge.network.num_nodes];
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
        if history.residuals[node].len() >= 8 {
            let mut sorted = history.residuals[node].clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = percentile(&sorted, 50, 100);
            let recent = *history.residuals[node].last().unwrap_or(&median);
            residual_shift[node] = (0.65 * median + 0.35 * recent).clamp(-25.0, 25.0);
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
            (withdrawable_mwh / mwh_per_step).ceil() as usize
        } else {
            usize::MAX
        };
        let terminal_drain = n_remaining <= discharge_steps_to_min.saturating_add(1);
        let rank_frac = (terminal_rank[b] as f64 + 1.0) / (challenge.num_batteries.max(1) as f64);
        let early_terminal_drain = n_remaining <= 48
            && u_max > 0.0
            && relative_soc_pressure(battery, state.socs[b]) > 0.35
            && rank_frac <= 0.55
            && current_price > KAPPA_TX;

        let mut a = 0.0_f64;
        if terminal_drain && u_max > 0.0 && current_price > friction {
            a = u_max;
        } else if early_terminal_drain {
            let urgency = (1.0 - n_remaining as f64 / 48.0).clamp(0.0, 1.0);
            let fullness = relative_soc_pressure(battery, state.socs[b]);
            let rank_boost = (0.65 - rank_frac).max(0.0);
            let fraction = (0.25 + 0.55 * urgency + 0.35 * fullness + 0.25 * rank_boost)
                .clamp(0.35, 1.0);
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
            &dps[b],
            battery,
            t,
            state.socs[b],
            current_price,
            state.action_bounds[b],
            hp,
        );
        if dp_action_value(&dps[b], battery, t, state.socs[b], current_price, dp_action)
            > dp_action_value(&dps[b], battery, t, state.socs[b], current_price, a) + EPS
        {
            a = dp_action;
        }

        target[b] = a;
    }

    for node in 0..challenge.network.num_nodes {
        history.values[node].push(state.rt_prices[node]);
        history.residuals[node]
            .push(state.rt_prices[node] - challenge.market.day_ahead_prices[t][node]);
    }
    drop(history);

    if hp.soc_ref_lambda > 0.0 && t % hp.soc_ref_dyn_stride == 0 {
        let refs = compute_soc_reference_dynamic(challenge, &state.socs, &residual_shift, t);
        let mut soc_ref = soc_ref_lock().lock().unwrap();
        *soc_ref = refs;
    }

    clamp_to_bounds(&mut target, &state.action_bounds);

    let dp_seed: Vec<f64> = (0..challenge.num_batteries)
        .map(|b| {
            let battery = &challenge.batteries[b];
            pick_dp_action(
                &dps[b],
                battery,
                t,
                state.socs[b],
                state.rt_prices[battery.node],
                state.action_bounds[b],
                hp,
            )
        })
        .collect();

    let zero = vec![0.0_f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero);



    let mut seeds = if hp.use_arb_seed {
        let prices = &state.rt_prices;
        let n = prices.len();
        let arb_threshold = if hp.arb_pct == 0 || n == 0 {
            if n > 0 { prices.iter().sum::<f64>() / n as f64 } else { 0.0 }
        } else {
            let mut sorted = prices.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            percentile(&sorted, hp.arb_pct as usize, 100)
        };
        let arb_seed: Vec<f64> = (0..challenge.num_batteries)
            .map(|b| {
                let battery = &challenge.batteries[b];
                let price = prices[battery.node];
                let (lo, hi) = state.action_bounds[b];

                let discharge_when_high = !hp.arb_inverse;
                if (price > arb_threshold) == discharge_when_high { hi } else { lo }
            })
            .collect();
        let third_seed: Vec<f64> = if hp.use_water_value_third {

            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let press = relative_soc_pressure(battery, state.socs[b]);
                    let price_scale = arb_threshold.abs();
                    let trigger = arb_threshold - hp.wv_kappa * (press - 0.5) * price_scale;
                    let price = prices[battery.node];
                    let (lo, hi) = state.action_bounds[b];
                    if price > trigger { hi } else { lo }
                })
                .collect()
        } else if hp.use_obl_target_seed {

            (0..challenge.num_batteries)
                .map(|b| {
                    let (lo, hi) = state.action_bounds[b];
                    (lo + hi - target[b]).clamp(lo, hi)
                })
                .collect()
        } else if hp.use_ramp_seed {

            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let node = battery.node;
                    let (lo, hi) = state.action_bounds[b];
                    let price_now = challenge.market.day_ahead_prices[t][node];
                    let price_prev = challenge.market.day_ahead_prices[t.saturating_sub(1)][node];
                    if price_now >= price_prev { hi } else { lo }
                })
                .collect()
        } else if hp.use_block_seed_third && !block_plan.is_empty() {

            (0..challenge.num_batteries)
                .map(|b| {
                    let (lo, hi) = state.action_bounds[b];
                    match block_plan.get(b).and_then(|p| p.get(t)).copied().unwrap_or(0) {
                        1 => hi,
                        -1 => lo,
                        _ => 0.0,
                    }
                })
                .collect()
        } else if hp.use_rollout_seed_third {

            let w = (hp.rollout_window as usize).max(2);
            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let node = battery.node;
                    let (lo, hi) = state.action_bounds[b];
                    let end = (t + w).min(n_steps);
                    if end <= t {
                        return 0.0_f64;
                    }
                    let mut sorted_w: Vec<f64> = (t..end)
                        .map(|tau| challenge.market.day_ahead_prices[tau][node])
                        .collect();
                    sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let local_threshold = percentile(&sorted_w, 75, 100);
                    let price_now = challenge.market.day_ahead_prices[t][node];
                    if price_now > local_threshold { hi } else { lo }
                })
                .collect()
        } else if hp.use_da_arb_third {

            let n_nodes = challenge.network.num_nodes;
            let da_threshold = {
                let mut da_cross_node: Vec<f64> = Vec::with_capacity(n_nodes);
                for node in 0..n_nodes {
                    da_cross_node.push(challenge.market.day_ahead_prices[t][node]);
                }
                da_cross_node.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                percentile(&da_cross_node, hp.arb_pct as usize, 100)
            };
            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let price_da = challenge.market.day_ahead_prices[t][battery.node];
                    let (lo, hi) = state.action_bounds[b];
                    if price_da > da_threshold { hi } else { lo }
                })
                .collect()
        } else if hp.use_spread_arb_third {

            let n_nodes = challenge.network.num_nodes;
            let spread_threshold = {
                let mut spread_cross_node: Vec<f64> = Vec::with_capacity(n_nodes);
                for node in 0..n_nodes {
                    let spread = state.rt_prices[node] - challenge.market.day_ahead_prices[t][node];
                    spread_cross_node.push(spread);
                }
                spread_cross_node.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                percentile(&spread_cross_node, hp.arb_pct as usize, 100)
            };
            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let spread_b = state.rt_prices[battery.node]
                        - challenge.market.day_ahead_prices[t][battery.node];
                    let (lo, hi) = state.action_bounds[b];
                    if spread_b > spread_threshold { hi } else { lo }
                })
                .collect()
        } else if hp.use_std_arb_third {

            let third_threshold = if hp.arb_pct_third == 0 {
                arb_threshold
            } else {
                let mut sorted3 = prices.to_vec();
                sorted3.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                percentile(&sorted3, hp.arb_pct_third as usize, 100)
            };
            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let price = prices[battery.node];
                    let (lo, hi) = state.action_bounds[b];
                    if price > third_threshold { hi } else { lo }
                })
                .collect()
        } else {
            zero.clone()
        };
        vec![target, arb_seed, third_seed]
    } else {
        vec![target, dp_seed, zero.clone()]
    };


    if hp.use_arb_seed && hp.use_rollout_additive {
        let w = (hp.rollout_window as usize).max(2);
        let rollout: Vec<f64> = (0..challenge.num_batteries)
            .map(|b| {
                let battery = &challenge.batteries[b];
                let node = battery.node;
                let (lo, hi) = state.action_bounds[b];
                let end = (t + w).min(n_steps);
                if end <= t {
                    return 0.0_f64;
                }
                let mut sorted_w: Vec<f64> = (t..end)
                    .map(|tau| challenge.market.day_ahead_prices[tau][node])
                    .collect();
                sorted_w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let local_threshold = percentile(&sorted_w, 75, 100);
                let price_now = challenge.market.day_ahead_prices[t][node];
                if price_now > local_threshold { hi } else { lo }
            })
            .collect();
        seeds.push(rollout);
    }


    if hp.use_arb_seed && hp.use_rollout_additive && hp.use_rollout_additive_2 {
        let w2 = (hp.rollout_window_2 as usize).max(2);
        let rollout2: Vec<f64> = (0..challenge.num_batteries)
            .map(|b| {
                let battery = &challenge.batteries[b];
                let node = battery.node;
                let (lo, hi) = state.action_bounds[b];
                let end = (t + w2).min(n_steps);
                if end <= t {
                    return 0.0_f64;
                }
                let mut sorted_w2: Vec<f64> = (t..end)
                    .map(|tau| challenge.market.day_ahead_prices[tau][node])
                    .collect();
                sorted_w2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let local_threshold2 = percentile(&sorted_w2, 75, 100);
                let price_now = challenge.market.day_ahead_prices[t][node];
                if price_now > local_threshold2 { hi } else { lo }
            })
            .collect();
        seeds.push(rollout2);
    }


    if hp.use_arb_seed && hp.use_rollout_additive && hp.use_rollout_additive_2 && hp.use_rollout_additive_3 {
        let w3 = (hp.rollout_window_3 as usize).max(2);
        let rollout3: Vec<f64> = (0..challenge.num_batteries)
            .map(|b| {
                let battery = &challenge.batteries[b];
                let node = battery.node;
                let (lo, hi) = state.action_bounds[b];
                let end = (t + w3).min(n_steps);
                if end <= t {
                    return 0.0_f64;
                }
                let mut sorted_w3: Vec<f64> = (t..end)
                    .map(|tau| challenge.market.day_ahead_prices[tau][node])
                    .collect();
                sorted_w3.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let local_threshold3 = percentile(&sorted_w3, 75, 100);
                let price_now = challenge.market.day_ahead_prices[t][node];
                if price_now > local_threshold3 { hi } else { lo }
            })
            .collect();
        seeds.push(rollout3);
    }


    if hp.use_arb_seed && hp.use_rollout_additive && hp.use_rollout_additive_2
        && hp.use_rollout_additive_3 && hp.use_rollout_additive_4
    {
        let rollout4: Vec<f64> = (0..challenge.num_batteries)
            .map(|b| {
                let battery = &challenge.batteries[b];
                let node = battery.node;
                let (lo, hi) = state.action_bounds[b];
                if t == 0 {
                    return lo;
                }
                let price_now = challenge.market.day_ahead_prices[t][node];
                let price_prev = challenge.market.day_ahead_prices[t - 1][node];
                if price_now > price_prev { hi } else { lo }
            })
            .collect();
        seeds.push(rollout4);
    }



    if hp.use_rolling_horizon && state.time_step % hp.rh_stride == 0 {
        let mut warm = rh_lam.lock().unwrap();
        if let Some(rh_raw) = rolling_horizon_lp_seed(
            challenge, state, dps, sens, &base_flows, &mut *warm
        ) {
            let mut rh = rh_raw;
            clamp_to_bounds(&mut rh, &state.action_bounds);
            seeds.push(rh);
        }
    }

    let mut result = joint_optimize_step(challenge, state, dps, sens, &base_flows, seeds, hp);

    if hp.use_lp_dispatch {
        let pga_val = total_step_value(challenge, state, dps, &result);
        if let Some(mut lp_act) = lp_dispatch_step(challenge, state, dps, sens, &base_flows, hp) {
            safe_project_to_feasible(challenge, state, &mut lp_act, sens, &base_flows, hp);
            if is_flow_feasible(challenge, state, &lp_act) {
                let lp_val = total_step_value(challenge, state, dps, &lp_act);
                if lp_val > pga_val {
                    result = lp_act;
                }
            }
        }
    }

    result = coordinate_polish_step(challenge, state, dps, sens, result, hp);

    if hp.use_joint_pair_polish {
        let pre_polish = result.clone();
        joint_pair_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_polish;
        }
    }


    if hp.use_joint_triplet_polish {
        let pre_triplet = result.clone();
        joint_triplet_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_triplet;
        }
    }

    if hp.use_basin_hop {
        basin_hop_restart(challenge, state, dps, sens, &base_flows, &mut result, hp);
    }

    if hp.use_admm_polish {
        let pre_admm = result.clone();
        admm_consensus_polish(challenge, state, dps, sens, &base_flows, &mut result);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_admm;
        }
    }

    if hp.use_ejection_chain {
        let pre_chain = result.clone();
        joint_ejection_chain_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_chain;
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
        let base_premium = 20.0 * hp.lmp_premium_scale;
        let threshold = hp.lmp_threshold;
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
                &hp,
            )
        })
        .collect();

    if hp.use_scvc {
        for (b, battery) in challenge.batteries.iter().enumerate() {
            let node = battery.node;
            let da_at_node: Vec<f64> = (0..challenge.num_steps)
                .map(|t| challenge.market.day_ahead_prices[t][node] + expected_premiums[t][b])
                .collect();
            apply_scvc_to_dp(&mut dps[b], battery, &da_at_node, hp.scvc_alpha);
        }
    }

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
    let n_lines = challenge.network.flow_limits.len();
    let rh_lam: Mutex<(Vec<f64>, Vec<f64>)> = Mutex::new((
        vec![0.0_f64; n_lines],
        vec![0.0_f64; n_lines],
    ));

    let block_plan: Vec<Vec<i8>> = if hp.use_block_seed_third {
        let n_t = challenge.num_steps;
        (0..challenge.num_batteries).map(|b| {
            let battery = &challenge.batteries[b];
            let node = battery.node;
            let cap_span = (battery.soc_max_mwh - battery.soc_min_mwh).max(0.0);
            let steps_to_full = if battery.power_charge_mw > 0.0 {
                (cap_span / (battery.power_charge_mw * DELTA_T)).round() as usize
            } else {
                n_t
            };
            let k_cap = (hp.block_frac * n_t as f64).round() as usize;
            let k = steps_to_full.min(k_cap).min(n_t / 2);
            let mut sorted_steps: Vec<usize> = (0..n_t).collect();
            sorted_steps.sort_unstable_by(|&a, &bb| {
                let pa = challenge.market.day_ahead_prices[a][node];
                let pb = challenge.market.day_ahead_prices[bb][node];
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut plan = vec![0i8; n_t];
            for &step in sorted_steps.iter().take(k) {
                plan[step] = -1;
            }
            for &step in sorted_steps.iter().rev().take(k) {
                plan[step] = 1;
            }
            plan
        }).collect()
    } else {
        Vec::new()
    };
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        policy(c, s, &dps, &sens, &hp, &rh_lam, &block_plan)
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
    let limits = &challenge.network.flow_limits;

    let line_indices: Vec<usize> = if hp.lp_screen_mode == 1 {
        (0..n_lines_total)
            .filter(|&l| {
                let row = &sens[l];
                let mut swing = 0.0_f64;
                for b in 0..num_b {
                    let coeff = row[b];
                    let (lo, hi) = state.action_bounds[b];
                    swing += (coeff * lo).abs().max((coeff * hi).abs());
                }
                base_flows[l].abs() + swing > limits[l]
            })
            .collect()
    } else if hp.lp_max_lines > 0 && hp.lp_max_lines < n_lines_total {
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

    let rows_per_b = if hp.lp_drop_soc_rows { 2 } else { 4 };
    let n = 2 * num_b;
    let m = rows_per_b * num_b + 2 * num_l;

    let mut c_obj = vec![0.0_f64; n];
    let mut a_mat = vec![vec![0.0_f64; n]; m];
    let mut b_vec = vec![0.0_f64; m];

    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let price = state.rt_prices[battery.node];
        let soc = state.socs[b];
        let (u_min, u_max) = state.action_bounds[b];

        let (dv_d, dv_c) = if hp.lp_dv_mode == 1 {
            let mid_d = (soc - 0.5 * u_max.max(0.0) * DELTA_T / ETA_DISCHARGE)
                .max(battery.soc_min_mwh);
            let mid_c = (soc + 0.5 * (-u_min).max(0.0) * ETA_CHARGE * DELTA_T)
                .min(battery.soc_max_mwh);
            (dv_dsoc(&dps[b], t, mid_d), dv_dsoc(&dps[b], t, mid_c))
        } else {
            let dv = dv_dsoc(&dps[b], t, soc);
            (dv, dv)
        };

        c_obj[b] = (price - KAPPA_TX) * DELTA_T - dv_d * (DELTA_T / ETA_DISCHARGE);
        c_obj[num_b + b] = dv_c * ETA_CHARGE * DELTA_T - (price + KAPPA_TX) * DELTA_T;

        let r = rows_per_b * b;
        a_mat[r][b] = 1.0;
        b_vec[r] = u_max.max(0.0);
        a_mat[r + 1][num_b + b] = 1.0;
        b_vec[r + 1] = (-u_min).max(0.0);
        if !hp.lp_drop_soc_rows {
            a_mat[r + 2][b] = DELTA_T / ETA_DISCHARGE;
            b_vec[r + 2] = (soc - battery.soc_min_mwh).max(0.0);
            a_mat[r + 3][num_b + b] = ETA_CHARGE * DELTA_T;
            b_vec[r + 3] = (battery.soc_max_mwh - soc).max(0.0);
        }
    }

    let row_f = rows_per_b * num_b;
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
    let (opt_x, _) = if hp.lp_flat_tableau || hp.lp_dantzig {
        lp_solver::lp_solve_flat(n, m, &c_obj, &a_mat, &b_vec, budget, hp.lp_dantzig)
    } else {
        lp_solver::lp_solve_with_budget(n, m, &c_obj, &a_mat, &b_vec, budget)
    };
    let opt_x = opt_x?;

    let mut actions = vec![0.0_f64; num_b];
    for b in 0..num_b {
        let d = opt_x[b];
        let c = opt_x[num_b + b];
        let u = d - c;
        let (lo, hi) = state.action_bounds[b];
        actions[b] = u.clamp(lo, hi);
    }

    if !actions.iter().all(|a| a.is_finite()) {
        return None;
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

    pub fn lp_solve_flat(
        n: usize, m: usize, c: &[f64], a: &[Vec<f64>], b: &[f64], max_pivots: usize, dantzig: bool,
    ) -> (Option<Vec<f64>>, usize) {
        if b.iter().any(|&x| x < -1e-6) {
            return (None, 0);
        }

        let n_vars = n + m;
        let rhs_col = n_vars;
        let n_cols = n_vars + 1;
        let obj_r = m * n_cols;

        let mut tab = vec![0.0_f64; (m + 1) * n_cols];
        for i in 0..m {
            let r = i * n_cols;
            tab[r..r + n].copy_from_slice(&a[i][..n]);
            tab[r + n + i] = 1.0;
            tab[r + rhs_col] = b[i].max(0.0);
        }
        for j in 0..n {
            tab[obj_r + j] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();
        let mut prow = vec![0.0_f64; n_cols];
        let mut pivots_used = 0usize;
        let mut stalls = 0usize;
        let mut last_obj = f64::NEG_INFINITY;
        const STALL_LIMIT: usize = 8;

        for pivot in 0..max_pivots {
            pivots_used = pivot + 1;

            let use_bland = !dantzig || stalls >= STALL_LIMIT;
            let entering = {
                let obj = &tab[obj_r..obj_r + n_vars];
                if use_bland {
                    obj.iter().position(|&v| v < -LP_EPS)
                } else {
                    let mut bj: Option<usize> = None;
                    let mut bv = -LP_EPS;
                    for (j, &v) in obj.iter().enumerate() {
                        if v < bv {
                            bv = v;
                            bj = Some(j);
                        }
                    }
                    bj
                }
            };
            let entering = match entering {
                Some(j) => j,
                None => break,
            };

            let tie_break = dantzig && use_bland;
            let mut leaving: usize = usize::MAX;
            let mut best_ratio = 0.0_f64;
            for i in 0..m {
                let ri = i * n_cols;
                let ai = tab[ri + entering];
                if !(ai > LP_EPS) {
                    continue;
                }
                let r = tab[ri + rhs_col] / ai;
                if leaving == usize::MAX || r < best_ratio {
                    leaving = i;
                    best_ratio = r;
                } else if tie_break && r == best_ratio && basis[i] < basis[leaving] {
                    leaving = i;
                }
            }
            if leaving == usize::MAX {
                return (None, 0);
            }
            let leaving_row = leaving;

            let lr = leaving_row * n_cols;
            let pivot_val = tab[lr + entering];
            if pivot_val.abs() < LP_EPS {
                return (None, 0);
            }
            for j in 0..n_cols {
                tab[lr + j] /= pivot_val;
            }
            prow.copy_from_slice(&tab[lr..lr + n_cols]);
            for (i, row) in tab.chunks_exact_mut(n_cols).enumerate() {
                if i == leaving_row {
                    continue;
                }
                let factor = row[entering];
                if factor.abs() > 1e-15 {
                    for (x, p) in row.iter_mut().zip(prow.iter()) {
                        *x -= factor * *p;
                    }
                }
            }
            basis[leaving_row] = entering;

            if dantzig {
                let o = tab[obj_r + rhs_col];
                if o <= last_obj + 1e-12 {
                    stalls += 1;
                } else {
                    stalls = 0;
                }
                last_obj = o;
            }
        }

        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[i * n_cols + rhs_col].max(0.0);
            }
        }
        (Some(x), pivots_used)
    }
}
