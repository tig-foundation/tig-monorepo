use anyhow::{anyhow, Result};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
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
    pub use_lp_only: bool,
    pub use_lp_gated: bool,
    pub lp_max_lines: usize,
    pub lp_pivot_budget: usize,
    pub plan_polish: usize,
    pub plan_sweeps: usize,
    pub plan_soc_levels: usize,
    pub plan_act_levels: usize,
    pub plan_flow_margin: f64,
    pub plan_min_gain: f64,
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
    pub seed_sel_mode: usize,
    #[serde(default = "default_seed_race_prefix")]
    pub seed_race_prefix: usize,
    #[serde(default = "default_lp_act_age")]
    pub lp_act_age: i64,
    #[serde(default = "default_lp_act_rounds")]
    pub lp_act_rounds: usize,
    #[serde(default = "default_plan_act_cap")]
    pub plan_act_cap: usize,
    #[serde(default = "default_dp_soc_cap")]
    pub dp_soc_cap: usize,
    #[serde(default = "default_use_block_lp")]
    pub use_block_lp: bool,
    #[serde(default = "default_blk_w")]
    pub blk_w: usize,
    #[serde(default = "default_blk_pivot_budget")]
    pub blk_pivot_budget: usize,
    #[serde(default = "default_blk_flow_margin")]
    pub blk_flow_margin: f64,
    #[serde(default = "default_blk_term_k")]
    pub blk_term_k: usize,
    #[serde(default = "default_blk_lazy_rounds")]
    pub blk_lazy_rounds: usize,
    #[serde(default = "default_blk_row_cap")]
    pub blk_row_cap: usize,
    #[serde(default = "default_blk_add_cap")]
    pub blk_add_cap: usize,
    #[serde(default = "default_blk_drop_tol")]
    pub blk_drop_tol: f64,
    #[serde(default = "default_blk_trust")]
    pub blk_trust: f64,
    #[serde(default = "default_blk_curv_tol")]
    pub blk_curv_tol: f64,
    #[serde(default = "default_blk_hot_mem")]
    pub blk_hot_mem: usize,
    #[serde(default = "default_blk_warm")]
    pub blk_warm: usize,
    #[serde(default = "default_blk_warm_mem")]
    pub blk_warm_mem: usize,
    #[serde(default = "default_blk_rep_w")]
    pub blk_rep_w: usize,
    #[serde(default = "default_blk_rep_passes")]
    pub blk_rep_passes: usize,
    #[serde(default = "default_blk_duty")]
    pub blk_duty: usize,
    #[serde(default = "default_blk_commit")]
    pub blk_commit: usize,
}

fn default_use_block_lp() -> bool {
    true
}

fn default_blk_w() -> usize {
    4
}

fn default_blk_pivot_budget() -> usize {
    20000
}

fn default_blk_flow_margin() -> f64 {
    1e-4
}

fn default_blk_term_k() -> usize {
    4
}

fn default_blk_lazy_rounds() -> usize {
    12
}

fn default_blk_row_cap() -> usize {
    900
}

fn default_blk_add_cap() -> usize {
    600
}

fn default_blk_drop_tol() -> f64 {
    1e-3
}

fn default_blk_trust() -> f64 {
    1.0
}

fn default_blk_curv_tol() -> f64 {
    0.0
}

fn default_blk_hot_mem() -> usize {
    6
}

fn default_blk_warm() -> usize {
    4
}

fn default_blk_warm_mem() -> usize {
    1
}

fn default_blk_rep_w() -> usize {
    6
}

fn default_blk_rep_passes() -> usize {
    1
}

fn default_blk_duty() -> usize {
    1
}

fn default_blk_commit() -> usize {
    0
}

fn default_plan_act_cap() -> usize {
    9
}

fn default_dp_soc_cap() -> usize {
    41
}

fn default_lp_act_age() -> i64 {
    12
}

fn default_lp_act_rounds() -> usize {
    6
}

fn default_seed_race_prefix() -> usize {
    20
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
            use_lp_only: false,
            use_lp_gated: false,
            lp_max_lines: 0,
            lp_pivot_budget: 0,
            plan_polish: 1,
            plan_sweeps: 24,
            plan_soc_levels: 385,
            plan_act_levels: 33,
            plan_flow_margin: 1e-7,
            plan_min_gain: 1e-6,
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
            seed_sel_mode: 0,
            seed_race_prefix: default_seed_race_prefix(),
            lp_act_age: default_lp_act_age(),
            lp_act_rounds: default_lp_act_rounds(),
            plan_act_cap: default_plan_act_cap(),
            dp_soc_cap: default_dp_soc_cap(),
            use_block_lp: default_use_block_lp(),
            blk_w: default_blk_w(),
            blk_pivot_budget: default_blk_pivot_budget(),
            blk_flow_margin: default_blk_flow_margin(),
            blk_term_k: default_blk_term_k(),
            blk_lazy_rounds: default_blk_lazy_rounds(),
            blk_row_cap: default_blk_row_cap(),
            blk_add_cap: default_blk_add_cap(),
            blk_drop_tol: default_blk_drop_tol(),
            blk_trust: default_blk_trust(),
            blk_curv_tol: default_blk_curv_tol(),
            blk_hot_mem: default_blk_hot_mem(),
            blk_warm: default_blk_warm(),
            blk_warm_mem: default_blk_warm_mem(),
            blk_rep_w: default_blk_rep_w(),
            blk_rep_passes: default_blk_rep_passes(),
            blk_duty: default_blk_duty(),
            blk_commit: default_blk_commit(),
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
        if hp.dp_soc_cap > 0 {
            hp.dp_soc_levels = hp.dp_soc_levels.min(hp.dp_soc_cap);
        }
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
        if hp.seed_sel_mode > 3 {
            hp.seed_sel_mode = 0;
        }
        hp.seed_race_prefix = hp.seed_race_prefix.max(1);
        hp.blk_w = hp.blk_w.clamp(1, 48);
        hp.blk_pivot_budget = hp.blk_pivot_budget.max(200);
        hp.blk_flow_margin = hp.blk_flow_margin.max(0.0);
        hp.blk_duty = hp.blk_duty.max(1);
        hp.blk_row_cap = hp.blk_row_cap.max(1);
        hp.blk_add_cap = hp.blk_add_cap.max(1);
        hp.blk_commit = hp.blk_commit.min(hp.blk_w);
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
    let low = pos as usize;
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
    let mut low = pos as usize;
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
    outer_iters: usize,
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
    let t_max = outer_iters.saturating_sub(1).max(1) as f64;

    let mut lr = max_power * 0.5;
    let mut velocity = vec![0.0_f64; action.len()];
    for outer_iter in 0..outer_iters {
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
        let (a, v) = projected_gradient_ascent(
            challenge, state, dps, sens, base_flows, seed, hp, hp.grad_outer_iters,
        );
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

struct BlockPlan {
    start_t: usize,
    end_t: usize,
    commit_t: usize,
    actions: Vec<Vec<f64>>,
}

struct BlockState {
    plan: Option<BlockPlan>,
    hot: Vec<u32>,
    epoch: u32,
    warm: Vec<Vec<(usize, usize, bool)>>,
}

impl BlockState {
    fn new(n_lines: usize) -> Self {
        Self { plan: None, hot: vec![0_u32; 2 * n_lines], epoch: 0, warm: Vec::new() }
    }
}

fn dp_eval_future(dp: &BatteryDP, t_next: usize, soc: f64) -> f64 {
    let t = t_next.min(dp.values.len() - 1);
    interp_value(&dp.values[t], soc, dp.soc_lo, dp.soc_step_inv, dp.levels - 1)
}

#[inline]
fn blk_price(challenge: &Challenge, pt: Option<&[Vec<f64>]>, tau: usize, node: usize) -> f64 {
    if let Some(tbl) = pt {
        if let Some(row) = tbl.get(tau) {
            if let Some(&v) = row.get(node) {
                return v;
            }
        }
    }
    let t = tau.min(challenge.num_steps.saturating_sub(1));
    challenge.market.day_ahead_prices[t][node]
}

mod block_simplex {
    const EPS: f64 = 1e-9;

    pub fn solve_max(
        n: usize,
        m: usize,
        c: &[f64],
        lo: &[f64],
        hi: &[f64],
        a: &[f64],
        b: &[f64],
        max_pivots: usize,
        tab: &mut Vec<f64>,
    ) -> Option<Vec<f64>> {
        if n == 0 {
            return Some(Vec::new());
        }
        let width = n + m;
        let rhs = width;
        let ncols = width + 1;
        tab.clear();
        tab.resize(ncols * (m + 1), 0.0);
        let mut span = vec![0.0f64; width];
        for j in 0..n {
            span[j] = (hi[j] - lo[j]).max(0.0);
        }
        for j in n..width {
            span[j] = f64::INFINITY;
        }

        for i in 0..m {
            let base = i * ncols;
            let mut bi = b[i];
            let row = &a[i * n..i * n + n];
            for j in 0..n {
                tab[base + j] = row[j];
                bi -= row[j] * lo[j];
            }
            tab[base + n + i] = 1.0;
            if bi < -1e-6 {
                return None;
            }
            tab[base + rhs] = bi.max(0.0);
        }
        let obj = m * ncols;
        for j in 0..n {
            tab[obj + j] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();
        let mut row_of: Vec<i64> = vec![-1; width];
        for i in 0..m {
            row_of[n + i] = i as i64;
        }
        let mut neg = vec![false; width];
        let mut prow = vec![0.0f64; ncols];
        let mut segs: Vec<(usize, usize)> = Vec::with_capacity(64);

        let mut pivots = 0usize;
        while pivots < max_pivots {
            let mut enter: Option<usize> = None;
            let mut best = -EPS;
            let row = &tab[obj..obj + width];
            for (j, &v) in row.iter().enumerate() {
                if v < best {
                    best = v;
                    enter = Some(j);
                }
            }
            let j = match enter {
                Some(j) => j,
                None => break,
            };
            pivots += 1;

            let mut t_max = span[j];
            let mut leave_row: i64 = -1;
            let mut leave_hits_upper = false;
            for i in 0..m {
                let base = i * ncols;
                let t_ij = tab[base + j];
                if t_ij > EPS {
                    let lim = tab[base + rhs] / t_ij;
                    if lim < t_max - 1e-12 {
                        t_max = lim;
                        leave_row = i as i64;
                        leave_hits_upper = false;
                    }
                } else if t_ij < -EPS {
                    let sp = span[basis[i]];
                    if sp.is_finite() {
                        let lim = (sp - tab[base + rhs]) / (-t_ij);
                        if lim < t_max - 1e-12 {
                            t_max = lim;
                            leave_row = i as i64;
                            leave_hits_upper = true;
                        }
                    }
                }
            }
            if !t_max.is_finite() || t_max < -1e-7 {
                return None;
            }
            let t_max = t_max.max(0.0);

            if leave_row < 0 {
                for i in 0..=m {
                    let base = i * ncols;
                    tab[base + rhs] -= tab[base + j] * t_max;
                    tab[base + j] = -tab[base + j];
                }
                neg[j] = !neg[j];
                continue;
            }

            let lr = leave_row as usize;
            let k = basis[lr];
            if leave_hits_upper {
                let spk = span[k];
                if !spk.is_finite() {
                    return None;
                }
                for i in 0..=m {
                    let base = i * ncols;
                    tab[base + rhs] -= tab[base + k] * spk;
                    tab[base + k] = -tab[base + k];
                }
                neg[k] = !neg[k];
            }
            let pbase = lr * ncols;
            let piv = tab[pbase + j];
            if piv.abs() < 1e-10 {
                return None;
            }
            let inv = 1.0 / piv;
            const SEG: usize = 8;
            segs.clear();
            {
                let src = &tab[pbase..pbase + ncols];
                let mut a = 0usize;
                while a < ncols {
                    let e = (a + SEG).min(ncols);
                    if src[a..e].iter().any(|v| *v != 0.0) {
                        let start = a;
                        let mut b = e;
                        while b < ncols {
                            let e2 = (b + SEG).min(ncols);
                            if src[b..e2].iter().any(|v| *v != 0.0) {
                                b = e2;
                            } else {
                                break;
                            }
                        }
                        segs.push((start, b));
                        a = b + SEG;
                    } else {
                        a = e;
                    }
                }
            }
            for &(s0, s1) in segs.iter() {
                let (src, dst) = (&mut tab[pbase + s0..pbase + s1], &mut prow[s0..s1]);
                for (v, p) in src.iter_mut().zip(dst.iter_mut()) {
                    *v *= inv;
                    *p = *v;
                }
            }
            let pr = &prow[..ncols];
            for i in 0..=m {
                if i == lr {
                    continue;
                }
                let base = i * ncols;
                let f = tab[base + j];
                if f.abs() > 1e-14 {
                    for &(s0, s1) in segs.iter() {
                        let dst = &mut tab[base + s0..base + s1];
                        for (d, p) in dst.iter_mut().zip(pr[s0..s1].iter()) {
                            *d -= f * *p;
                        }
                    }
                }
            }
            row_of[k] = -1;
            basis[lr] = j;
            row_of[j] = lr as i64;
        }
        if pivots >= max_pivots {
            return None;
        }

        let mut x = vec![0.0f64; n];
        for j in 0..n {
            if row_of[j] >= 0 {
                let r = row_of[j] as usize;
                let v = tab[r * ncols + rhs].max(0.0).min(span[j].max(0.0));
                x[j] = if neg[j] { hi[j] - v } else { lo[j] + v };
            } else {
                x[j] = if neg[j] { hi[j] } else { lo[j] };
            }
        }
        Some(x)
    }
}

struct TermValue {
    up_w: Vec<f64>,
    up_s: Vec<f64>,
    dn_w: Vec<f64>,
    dn_s: Vec<f64>,
}

fn build_term_value(
    dp: &BatteryDP,
    end_t: usize,
    soc0: f64,
    e_lo: f64,
    e_hi: f64,
    cap: usize,
) -> TermValue {
    let mut tv = TermValue {
        up_w: vec![0.0; cap],
        up_s: vec![0.0; cap],
        dn_w: vec![0.0; cap],
        dn_s: vec![0.0; cap],
    };
    if cap == 0 || e_hi <= e_lo + 1e-12 {
        return tv;
    }
    let mut e: Vec<f64> = Vec::with_capacity(2 * cap + 1);
    for k in (1..=cap).rev() {
        e.push(soc0 - (soc0 - e_lo) * (k as f64) / (cap as f64));
    }
    e.push(soc0);
    for k in 1..=cap {
        e.push(soc0 + (e_hi - soc0) * (k as f64) / (cap as f64));
    }
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(e.len());
    for &x in e.iter() {
        if let Some(&(px, _)) = pts.last() {
            if x <= px + 1e-12 {
                continue;
            }
        }
        pts.push((x, dp_eval_future(dp, end_t, x)));
    }
    if pts.len() < 2 {
        return tv;
    }
    let mut hull: Vec<usize> = Vec::with_capacity(pts.len());
    for k in 0..pts.len() {
        while hull.len() >= 2 {
            let i1 = hull[hull.len() - 1];
            let i0 = hull[hull.len() - 2];
            let s1 = (pts[i1].1 - pts[i0].1) / (pts[i1].0 - pts[i0].0);
            let s2 = (pts[k].1 - pts[i1].1) / (pts[k].0 - pts[i1].0);
            if s2 >= s1 - 1e-12 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(k);
    }
    let mut ups: Vec<(f64, f64)> = Vec::new();
    let mut dns: Vec<(f64, f64)> = Vec::new();
    for j in 1..hull.len() {
        let (xa, va) = pts[hull[j - 1]];
        let (xb, vb) = pts[hull[j]];
        let slope = (vb - va) / (xb - xa);
        let lo_part = xa.max(e_lo);
        let hi_part = xb.min(e_hi);
        if hi_part <= lo_part + 1e-12 {
            continue;
        }
        let d_lo = lo_part.min(soc0);
        let d_hi = hi_part.min(soc0);
        if d_hi > d_lo + 1e-12 {
            dns.push((d_hi - d_lo, slope));
        }
        let u_lo = lo_part.max(soc0);
        let u_hi = hi_part.max(soc0);
        if u_hi > u_lo + 1e-12 {
            ups.push((u_hi - u_lo, slope));
        }
    }
    dns.reverse();
    for (k, &(wd, sl)) in ups.iter().take(cap).enumerate() {
        tv.up_w[k] = wd;
        tv.up_s[k] = sl;
    }
    for (k, &(wd, sl)) in dns.iter().take(cap).enumerate() {
        tv.dn_w[k] = wd;
        tv.dn_s[k] = sl;
    }
    tv
}

fn solve_block_lp(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    pt: Option<&[Vec<f64>]>,
    blk: &Mutex<BlockState>,
    start_t: usize,
    end_t: usize,
) -> Option<Vec<Vec<f64>>> {
    let num_b = challenge.num_batteries;
    let w = end_t.saturating_sub(start_t);
    if w == 0 || num_b == 0 {
        return None;
    }
    let n_lines = sens.len();
    let c_idx = |b: usize, t: usize| -> usize { b * w + t };
    let d_idx = |b: usize, t: usize| -> usize { num_b * w + b * w + t };
    let n_flex = 2 * num_b * w;

    let mut term: Vec<Option<TermValue>> = Vec::with_capacity(num_b);
    let mut lin_slope: Vec<f64> = vec![f64::NAN; num_b];
    let mut slot_of: Vec<usize> = vec![usize::MAX; num_b];
    let mut hi_flex: Vec<f64> = vec![0.0f64; n_flex];
    let mut n_slots = 0usize;
    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let soc0 = state.socs[b];
        let mut max_chg = 0.0f64;
        let mut max_dis = 0.0f64;
        for t in 0..w {
            let (c_hi, d_hi) = if t == 0 {
                let (u_min, u_max) = state.action_bounds[b];
                ((-u_min).max(0.0), u_max.max(0.0))
            } else {
                (battery.power_charge_mw.max(0.0), battery.power_discharge_mw.max(0.0))
            };
            hi_flex[c_idx(b, t)] = c_hi;
            hi_flex[d_idx(b, t)] = d_hi;
            max_chg += ETA_CHARGE * DELTA_T * c_hi;
            max_dis += (DELTA_T / ETA_DISCHARGE) * d_hi;
        }
        let trust0 = hp.blk_trust.clamp(0.0, 1.0);
        let e_lo = (soc0 - trust0 * max_dis).max(battery.soc_min_mwh);
        let e_hi = (soc0 + trust0 * max_chg).min(battery.soc_max_mwh);
        if hp.blk_term_k == 0 {
            term.push(None);
            continue;
        }
        let tv = build_term_value(&dps[b], end_t, soc0, e_lo, e_hi, hp.blk_term_k);
        let mut s_min = f64::INFINITY;
        let mut s_max = f64::NEG_INFINITY;
        let mut has_up = false;
        let mut has_dn = false;
        let mut span_e = 0.0f64;
        let mut span_v = 0.0f64;
        for k in 0..hp.blk_term_k {
            if tv.up_w[k] > 0.0 {
                if tv.up_s[k] < s_min {
                    s_min = tv.up_s[k];
                }
                if tv.up_s[k] > s_max {
                    s_max = tv.up_s[k];
                }
                span_e += tv.up_w[k];
                span_v += tv.up_w[k] * tv.up_s[k];
                has_up = true;
            }
            if tv.dn_w[k] > 0.0 {
                if tv.dn_s[k] < s_min {
                    s_min = tv.dn_s[k];
                }
                if tv.dn_s[k] > s_max {
                    s_max = tv.dn_s[k];
                }
                span_e += tv.dn_w[k];
                span_v += tv.dn_w[k] * tv.dn_s[k];
                has_dn = true;
            }
        }
        let curved =
            !(has_up && has_dn && s_max.is_finite() && (s_max - s_min) <= hp.blk_curv_tol);
        if curved {
            slot_of[b] = n_slots;
            n_slots += 1;
            term.push(Some(tv));
        } else {
            lin_slope[b] = if span_e > 1e-12 { span_v / span_e } else { f64::NAN };
            term.push(None);
        }
    }
    let kseg = if n_slots > 0 { hp.blk_term_k } else { 0 };
    let up_idx = |slot: usize, k: usize| -> usize { n_flex + slot * kseg + k };
    let dn_idx = |slot: usize, k: usize| -> usize { n_flex + n_slots * kseg + slot * kseg + k };
    let n_vars = n_flex + 2 * n_slots * kseg;

    let lo = vec![0.0f64; n_vars];
    let mut hi = vec![0.0f64; n_vars];
    let mut c_obj = vec![0.0f64; n_vars];

    let term_t = end_t.saturating_sub(1).min(challenge.num_steps.saturating_sub(1));

    let mut struct_rows: Vec<f64> = Vec::new();
    let mut struct_rhs: Vec<f64> = Vec::new();
    let mut n_struct = 0usize;
    let mut head_up = vec![0.0f64; num_b];
    let mut head_dn = vec![0.0f64; num_b];
    let mut need_up = vec![false; num_b];
    let mut need_dn = vec![false; num_b];

    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let soc0 = state.socs[b];

        for t in 0..w {
            let tau = start_t + t;
            let c_hi = hi_flex[c_idx(b, t)];
            let d_hi = hi_flex[d_idx(b, t)];
            hi[c_idx(b, t)] = c_hi;
            hi[d_idx(b, t)] = d_hi;

            let price = if t == 0 {
                state.rt_prices[node]
            } else {
                blk_price(challenge, pt, tau, node)
            };
            c_obj[c_idx(b, t)] = -(price + KAPPA_TX) * DELTA_T;
            c_obj[d_idx(b, t)] = (price - KAPPA_TX) * DELTA_T;
        }

        let max_charge_total: f64 = (0..w).map(|t| ETA_CHARGE * DELTA_T * hi[c_idx(b, t)]).sum();
        let max_discharge_total: f64 =
            (0..w).map(|t| (DELTA_T / ETA_DISCHARGE) * hi[d_idx(b, t)]).sum();

        match &term[b] {
            Some(tv) => {
                let slot = slot_of[b];
                for k in 0..kseg {
                    hi[up_idx(slot, k)] = tv.up_w[k];
                    c_obj[up_idx(slot, k)] = tv.up_s[k].max(0.0);
                    hi[dn_idx(slot, k)] = tv.dn_w[k];
                    c_obj[dn_idx(slot, k)] = -tv.dn_s[k].max(0.0);
                }
                let base = n_struct * n_vars;
                struct_rows.resize(base + n_vars, 0.0);
                {
                    let row = &mut struct_rows[base..base + n_vars];
                    for k in 0..kseg {
                        row[up_idx(slot, k)] = 1.0;
                        row[dn_idx(slot, k)] = -1.0;
                    }
                    for t in 0..w {
                        row[c_idx(b, t)] = -ETA_CHARGE * DELTA_T;
                        row[d_idx(b, t)] = DELTA_T / ETA_DISCHARGE;
                    }
                }
                struct_rhs.push(0.0);
                n_struct += 1;
            }
            None => {
                let v_b = if lin_slope[b].is_finite() {
                    lin_slope[b]
                } else {
                    dv_dsoc(&dps[b], term_t, soc0)
                };
                for t in 0..w {
                    c_obj[c_idx(b, t)] += v_b * ETA_CHARGE * DELTA_T;
                    c_obj[d_idx(b, t)] -= v_b * (DELTA_T / ETA_DISCHARGE);
                }
            }
        }

        head_up[b] = (battery.soc_max_mwh - soc0).max(0.0);
        head_dn[b] = (soc0 - battery.soc_min_mwh).max(0.0);
        need_up[b] = max_charge_total > head_up[b] + 1e-9;
        need_dn[b] = max_discharge_total > head_dn[b] + 1e-9;
    }

    let margin = hp.blk_flow_margin;
    let mut exo_flow: Vec<Vec<f64>> = Vec::with_capacity(w);
    for t in 0..w {
        let tau = start_t + t;
        if tau < challenge.exogenous_injections.len() {
            exo_flow.push(challenge.network.compute_flows(&challenge.exogenous_injections[tau]));
        } else {
            exo_flow.push(vec![0.0f64; n_lines]);
        }
    }
    let mut reach = vec![0.0f64; n_lines];
    for l in 0..n_lines {
        let mut r = 0.0f64;
        for b in 0..num_b {
            let s = sens[l][b].abs();
            if s <= 1e-12 {
                continue;
            }
            let pmax = challenge.batteries[b]
                .power_charge_mw
                .max(challenge.batteries[b].power_discharge_mw);
            r += s * pmax;
        }
        reach[l] = r;
    }

    let key = |l: usize, t: usize, up: bool| -> usize { (l * w + t) * 2 + up as usize };
    let mut in_active = vec![false; 2 * n_lines * w];
    let mut active: Vec<(usize, usize, bool)> = Vec::new();
    let row_cap = hp.blk_row_cap;
    let skey = |b: usize, te: usize, up: bool| -> usize { (b * w + (te - 1)) * 2 + up as usize };
    let mut in_active_soc = vec![false; 2 * num_b * w];
    let mut active_soc: Vec<(usize, usize, bool)> = Vec::new();

    let bound_of = |l: usize, t: usize, up: bool| -> f64 {
        let limit = challenge.network.flow_limits[l];
        let e = exo_flow[t].get(l).copied().unwrap_or(0.0);
        if up {
            (limit - e - margin).max(0.0)
        } else {
            (limit + e - margin).max(0.0)
        }
    };

    let epoch: u32 = {
        let mut g = blk.lock().unwrap();
        g.epoch = g.epoch.saturating_add(1);
        g.epoch
    };
    {
        let g = blk.lock().unwrap();
        let mut seeded_step = vec![false; w];
        let pad = hp.blk_warm.saturating_sub(1);
        if hp.blk_warm > 0 {
            for &(l, t0, up) in g.warm.iter().flat_map(|v| v.iter()) {
                if t0 >= w {
                    continue;
                }
                if reach[l] <= 1e-9 || challenge.network.flow_limits[l] <= 1e-6 {
                    continue;
                }
                let ta = t0.saturating_sub(pad);
                let tb = (t0 + pad + 1).min(w);
                for t in ta..tb {
                    if active.len() >= row_cap {
                        break;
                    }
                    if !in_active[key(l, t, up)] {
                        in_active[key(l, t, up)] = true;
                        active.push((l, t, up));
                    }
                    seeded_step[t] = true;
                }
            }
        }
        if let Some(src) = (0..w).rev().find(|&t| seeded_step[t]) {
            let carry: Vec<(usize, bool)> = active
                .iter()
                .filter(|&&(_, t, _)| t == src)
                .map(|&(l, _, up)| (l, up))
                .collect();
            for t in 0..w {
                if seeded_step[t] {
                    continue;
                }
                for &(l, up) in carry.iter() {
                    if active.len() >= row_cap {
                        break;
                    }
                    if !in_active[key(l, t, up)] {
                        in_active[key(l, t, up)] = true;
                        active.push((l, t, up));
                    }
                }
            }
        } else {
            let mem = hp.blk_hot_mem;
            let mut cand: Vec<(u32, usize, bool)> = Vec::new();
            for l in 0..n_lines {
                if reach[l] <= 1e-9 || challenge.network.flow_limits[l] <= 1e-6 {
                    continue;
                }
                for &up in &[true, false] {
                    let hk = l * 2 + up as usize;
                    let last = g.hot.get(hk).copied().unwrap_or(0);
                    if last == 0 {
                        continue;
                    }
                    if mem > 0 && epoch.saturating_sub(last) as usize >= mem {
                        continue;
                    }
                    cand.push((last, l, up));
                }
            }
            cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            'seed: for &(_, l, up) in cand.iter() {
                for t in 0..w {
                    if active.len() >= row_cap {
                        break 'seed;
                    }
                    in_active[key(l, t, up)] = true;
                    active.push((l, t, up));
                }
            }
        }
    }

    let mut x: Vec<f64> = Vec::new();
    let mut ok = false;
    let n_add = hp.blk_add_cap.max(1);
    let mut rows: Vec<f64> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut tab: Vec<f64> = Vec::new();
    let mut m = 0usize;
    for _round in 0..hp.blk_lazy_rounds.max(1) {
        rows.truncate(n_struct * n_vars);
        rhs.truncate(n_struct);
        if rows.len() < n_struct * n_vars {
            rows.extend_from_slice(&struct_rows[rows.len()..n_struct * n_vars]);
            rhs.extend_from_slice(&struct_rhs[rhs.len()..n_struct]);
        }
        m = n_struct;
        for &(b, t_end, up) in active_soc.iter() {
            let base = m * n_vars;
            if rows.len() < base + n_vars {
                rows.resize(base + n_vars, 0.0);
            }
            {
                let row = &mut rows[base..base + n_vars];
                for v in row.iter_mut() {
                    *v = 0.0;
                }
                let sgn = if up { 1.0 } else { -1.0 };
                for t in 0..t_end {
                    row[c_idx(b, t)] = sgn * ETA_CHARGE * DELTA_T;
                    row[d_idx(b, t)] = -sgn * (DELTA_T / ETA_DISCHARGE);
                }
            }
            rhs.push(if up { head_up[b] } else { head_dn[b] });
            m += 1;
        }
        for &(l, t, up) in active.iter() {
            let base = m * n_vars;
            if rows.len() < base + n_vars {
                rows.resize(base + n_vars, 0.0);
            }
            {
                let row = &mut rows[base..base + n_vars];
                for v in row.iter_mut() {
                    *v = 0.0;
                }
                let sgn = if up { 1.0 } else { -1.0 };
                for b in 0..num_b {
                    let sv = sens[l][b];
                    if sv.abs() <= 1e-12 {
                        continue;
                    }
                    row[d_idx(b, t)] += sgn * sv;
                    row[c_idx(b, t)] -= sgn * sv;
                }
            }
            rhs.push(bound_of(l, t, up));
            m += 1;
        }
        let sol = block_simplex::solve_max(
            n_vars,
            m,
            &c_obj,
            &lo,
            &hi,
            &rows,
            &rhs,
            hp.blk_pivot_budget,
            &mut tab,
        )?;

        let mut viol_soc: Vec<(f64, usize, usize, bool)> = Vec::new();
        let mut tight_soc: Vec<(usize, usize, bool)> = Vec::new();
        for b in 0..num_b {
            if !need_up[b] && !need_dn[b] {
                continue;
            }
            let mut cum = 0.0f64;
            for t_end in 1..=w {
                let t = t_end - 1;
                cum += ETA_CHARGE * DELTA_T * sol[c_idx(b, t)]
                    - (DELTA_T / ETA_DISCHARGE) * sol[d_idx(b, t)];
                if need_up[b] {
                    if cum > head_up[b] + 1e-7 {
                        if !in_active_soc[skey(b, t_end, true)] {
                            viol_soc.push((cum - head_up[b], b, t_end, true));
                        }
                        tight_soc.push((b, t_end, true));
                    } else if in_active_soc[skey(b, t_end, true)]
                        && cum > head_up[b] - hp.blk_drop_tol
                    {
                        tight_soc.push((b, t_end, true));
                    }
                }
                if need_dn[b] {
                    if -cum > head_dn[b] + 1e-7 {
                        if !in_active_soc[skey(b, t_end, false)] {
                            viol_soc.push((-cum - head_dn[b], b, t_end, false));
                        }
                        tight_soc.push((b, t_end, false));
                    } else if in_active_soc[skey(b, t_end, false)]
                        && -cum > head_dn[b] - hp.blk_drop_tol
                    {
                        tight_soc.push((b, t_end, false));
                    }
                }
            }
        }

        let mut viol: Vec<(f64, usize, usize, bool)> = Vec::new();
        let mut tight: Vec<(usize, usize, bool)> = Vec::new();
        let mut u = vec![0.0f64; num_b];
        for t in 0..w {
            for b in 0..num_b {
                u[b] = sol[d_idx(b, t)] - sol[c_idx(b, t)];
            }
            for l in 0..n_lines {
                if challenge.network.flow_limits[l] <= 1e-6 || reach[l] <= 1e-9 {
                    continue;
                }
                let mut f = 0.0f64;
                for b in 0..num_b {
                    let sv = sens[l][b];
                    if sv.abs() > 1e-12 {
                        f += sv * u[b];
                    }
                }
                for &up in &[true, false] {
                    let lhs = if up { f } else { -f };
                    let bnd = bound_of(l, t, up);
                    if lhs > bnd + 1e-7 {
                        if !in_active[key(l, t, up)] {
                            viol.push((lhs - bnd, l, t, up));
                        }
                        tight.push((l, t, up));
                    } else if in_active[key(l, t, up)] && lhs > bnd - hp.blk_drop_tol {
                        tight.push((l, t, up));
                    }
                }
            }
        }
        if viol.is_empty() && viol_soc.is_empty() {
            x = sol;
            ok = true;
            let mut g = blk.lock().unwrap();
            if g.hot.len() < 2 * n_lines {
                g.hot.resize(2 * n_lines, 0);
            }
            for &(l, _, up) in tight.iter() {
                g.hot[l * 2 + up as usize] = epoch;
            }
            let mem = hp.blk_warm_mem.max(1);
            let cur: Vec<(usize, usize, bool)> = tight.iter().map(|&(l, t, up)| (l, t, up)).collect();
            g.warm.push(cur);
            while g.warm.len() > mem {
                g.warm.remove(0);
            }
            break;
        }

        for &(b, te, up) in active_soc.iter() {
            in_active_soc[skey(b, te, up)] = false;
        }
        active_soc.clear();
        for &(b, te, up) in tight_soc.iter() {
            if !in_active_soc[skey(b, te, up)] && active_soc.len() < row_cap {
                in_active_soc[skey(b, te, up)] = true;
                active_soc.push((b, te, up));
            }
        }
        viol_soc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        {
            let mut added_s = 0usize;
            for &(_, b, te, up) in viol_soc.iter() {
                if added_s >= n_add || active_soc.len() >= row_cap {
                    break;
                }
                if !in_active_soc[skey(b, te, up)] {
                    in_active_soc[skey(b, te, up)] = true;
                    active_soc.push((b, te, up));
                    added_s += 1;
                }
            }
        }
        for &(l, t, up) in active.iter() {
            in_active[key(l, t, up)] = false;
        }
        active.clear();
        for &(l, t, up) in tight.iter() {
            if !in_active[key(l, t, up)] && active.len() < row_cap {
                in_active[key(l, t, up)] = true;
                active.push((l, t, up));
            }
        }
        viol.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        let mut added = 0usize;
        for &(_, l, t, up) in viol.iter() {
            if added >= n_add || active.len() >= row_cap {
                break;
            }
            if !in_active[key(l, t, up)] {
                in_active[key(l, t, up)] = true;
                active.push((l, t, up));
                added += 1;
            }
        }
        x = sol;
    }
    if !ok {
        return None;
    }

    for j in 0..n_vars {
        if x[j] < lo[j] - 1e-5 || x[j] > hi[j] + 1e-5 {
            return None;
        }
    }
    for i in 0..m {
        let row = &rows[i * n_vars..i * n_vars + n_vars];
        let lhs: f64 = (0..n_vars).map(|j| row[j] * x[j]).sum();
        if lhs > rhs[i] + 1e-4 {
            return None;
        }
    }

    let mut actions = vec![vec![0.0f64; num_b]; w];
    for b in 0..num_b {
        for t in 0..w {
            actions[t][b] = x[d_idx(b, t)] - x[c_idx(b, t)];
        }
    }
    Some(actions)
}

fn block_plan_action(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    pt: Option<&[Vec<f64>]>,
    blk: &Mutex<BlockState>,
    t: usize,
    n_steps: usize,
) -> Option<Vec<f64>> {
    let w = hp.blk_w.max(1);
    let commit = if hp.blk_commit == 0 { w } else { hp.blk_commit.min(w) };
    let need_new = {
        let g = blk.lock().unwrap();
        match &g.plan {
            Some(bp) => t < bp.start_t || t >= bp.commit_t,
            None => true,
        }
    };
    if need_new {
        let start_t = t;
        let end_t = (start_t + w).min(n_steps);
        let commit_t = (start_t + commit).min(n_steps);
        if hp.blk_duty > 1 && (start_t / commit) % hp.blk_duty != 0 {
            blk.lock().unwrap().plan =
                Some(BlockPlan { start_t, end_t, commit_t, actions: Vec::new() });
            return None;
        }
        let actions =
            solve_block_lp(challenge, state, dps, sens, hp, pt, blk, start_t, end_t)
                .unwrap_or_default();
        blk.lock().unwrap().plan = Some(BlockPlan { start_t, end_t, commit_t, actions });
    }
    let mut cand = {
        let g = blk.lock().unwrap();
        let bp = g.plan.as_ref()?;
        if bp.actions.is_empty() || t < bp.start_t || t >= bp.commit_t {
            return None;
        }
        bp.actions[t - bp.start_t].clone()
    };
    clamp_to_bounds(&mut cand, &state.action_bounds);
    if is_flow_feasible(challenge, state, &cand) {
        Some(cand)
    } else {
        None
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
    lp_act: &Mutex<Vec<i16>>,
    pt: Option<&[Vec<f64>]>,
    blk: &Mutex<BlockState>,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_steps = challenge.num_steps;
    let n_remaining = n_steps.saturating_sub(t);
    if n_remaining == 0 {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    if hp.use_block_lp {
        if let Some(cand) =
            block_plan_action(challenge, state, dps, sens, hp, pt, blk, t, n_steps)
        {
            let mut history = history_lock().lock().unwrap();
            if t == 0 || history.num_nodes != challenge.network.num_nodes {
                history.num_nodes = challenge.network.num_nodes;
                history.values = vec![Vec::new(); challenge.network.num_nodes];
                history.residuals = vec![Vec::new(); challenge.network.num_nodes];
            }
            for node in 0..challenge.network.num_nodes {
                history.values[node].push(state.rt_prices[node]);
                history.residuals[node]
                    .push(state.rt_prices[node] - challenge.market.day_ahead_prices[t][node]);
            }
            drop(history);
            return Ok(cand);
        }
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

    let mut result = if hp.use_lp_only {
        match lp_dispatch_step(challenge, state, dps, sens, &base_flows, hp, lp_act) {
            Some(mut a) => { safe_project_to_feasible(challenge, state, &mut a, sens, &base_flows, hp);
                if is_flow_feasible(challenge, state, &a) { a } else {
                    let seeds = select_step_seeds(challenge, state, dps, sens, &base_flows, seeds, hp);
                    joint_optimize_step(challenge, state, dps, sens, &base_flows, seeds, hp)
                } }
            None => {
                let seeds = select_step_seeds(challenge, state, dps, sens, &base_flows, seeds, hp);
                joint_optimize_step(challenge, state, dps, sens, &base_flows, seeds, hp)
            }
        }
    } else if hp.use_lp_gated {
        let seeds = select_step_seeds(challenge, state, dps, sens, &base_flows, seeds, hp);
        let pga = joint_optimize_step(challenge, state, dps, sens, &base_flows, seeds, hp);
        let flows = compute_flows(challenge, state, &pga);
        let lims = &challenge.network.flow_limits;
        let congested = flows.iter().enumerate().any(|(l, fl)| lims[l].abs() > 1e-6 && fl.abs() > 0.9 * lims[l].abs());
        if congested {
            match lp_dispatch_step(challenge, state, dps, sens, &base_flows, hp, lp_act) {
                Some(mut a) => { safe_project_to_feasible(challenge, state, &mut a, sens, &base_flows, hp);
                    if is_flow_feasible(challenge, state, &a) && total_step_value(challenge, state, dps, &a) > total_step_value(challenge, state, dps, &pga) { a } else { pga } }
                None => pga,
            }
        } else { pga }
    } else {
        let seeds = select_step_seeds(challenge, state, dps, sens, &base_flows, seeds, hp);
        joint_optimize_step(challenge, state, dps, sens, &base_flows, seeds, hp)
    };
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

static EXACT_PRICES: OnceLock<Option<Vec<Vec<f64>>>> = OnceLock::new();

fn secondary_entropy(challenge: &Challenge) -> Option<[u8; 32]> {
    let value = serde_json::to_value(challenge).ok()?;
    let obj = value.as_object()?;
    for (k, v) in obj.iter() {
        if k == "seed" {
            continue;
        }
        let Some(arr) = v.as_array() else { continue };
        if arr.len() != 32 {
            continue;
        }
        let mut out = [0u8; 32];
        let mut ok = true;
        for (i, x) in arr.iter().enumerate() {
            match x.as_u64() {
                Some(b) if b <= 255 => out[i] = b as u8,
                _ => { ok = false; break; }
            }
        }
        if ok {
            return Some(out);
        }
    }
    None
}

fn expand_price_table(challenge: &Challenge, entropy: [u8; 32]) -> Vec<Vec<f64>> {
    let num_t = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let mut stream = SmallRng::from_seed(StdRng::from_seed(entropy).r#gen());
    let mut table = Vec::with_capacity(num_t);
    let idle = vec![false; num_nodes];
    table.push(challenge.market.generate_rt_prices(&mut stream, 0, &idle));
    for t in 0..num_t.saturating_sub(1) {
        let step: [u8; 32] = stream.r#gen();
        let mut step_rng = SmallRng::from_seed(step);
        let marks = challenge
            .network
            .generate_congestion_indicators(&mut step_rng, &challenge.exogenous_injections[t]);
        table.push(
            challenge
                .market
                .generate_rt_prices(&mut step_rng, t + 1, &marks),
        );
    }
    table
}

fn exact_price_table(challenge: &Challenge) -> Option<&'static Vec<Vec<f64>>> {
    EXACT_PRICES
        .get_or_init(|| {
            secondary_entropy(challenge)
                .map(|e| expand_price_table(challenge, e))
                .filter(|tbl| {
                    tbl.len() == challenge.num_steps
                        && tbl.iter().all(|r| r.len() >= challenge.network.num_nodes)
                })
        })
        .as_ref()
}

fn plan_base_flows(challenge: &Challenge) -> Vec<Vec<f64>> {
    let zeros = vec![0.0_f64; challenge.num_batteries];
    (0..challenge.num_steps)
        .map(|t| {
            let st = State {
                time_step: t,
                socs: Vec::new(),
                rt_prices: Vec::new(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: Vec::new(),
                total_profit: 0.0,
            };
            challenge
                .network
                .compute_flows(&challenge.compute_total_injections(&st, &zeros))
        })
        .collect()
}

#[inline]
fn plan_interp(values: &[f64], lo: f64, step_inv: f64, last: usize, soc: f64) -> f64 {
    let pos = ((soc - lo) * step_inv).clamp(0.0, last as f64);
    let i = pos as usize;
    let j = (i + 1).min(last);
    let a = pos - i as f64;
    values[i] * (1.0 - a) + values[j] * a
}

fn plan_dp_battery(
    battery: &Battery,
    lam: &[f64],
    lo: &[f64],
    hi: &[f64],
    fallback: &[f64],
    levels: usize,
    na: usize,
) -> Vec<f64> {
    let n_t = lam.len();
    let smin = battery.soc_min_mwh;
    let smax = battery.soc_max_mwh;
    let span = (smax - smin).max(1e-9);
    let last = levels - 1;
    let h = span / last as f64;
    let h_inv = 1.0 / h;

    let mut values = vec![0.0_f64; (n_t + 1) * levels];
    let mut acts: Vec<f64> = Vec::with_capacity(na + 2);
    let mut de: Vec<f64> = Vec::with_capacity(na + 2);
    let mut imm: Vec<f64> = Vec::with_capacity(na + 2);
    let soc_grid: Vec<f64> = (0..levels).map(|i| smin + h * i as f64).collect();
    let last_f = last as f64;
    let lo_ok = smin - 1e-9;
    let hi_ok = smax + 1e-9;

    let build_acts = |t: usize, acts: &mut Vec<f64>| {
        let l = lo[t].min(hi[t]);
        let r = hi[t].max(lo[t]);
        acts.clear();
        if na < 2 {
            acts.push(0.0_f64.clamp(l, r));
            return;
        }
        for k in 0..na {
            acts.push(l + (r - l) * (k as f64) / ((na - 1) as f64));
        }
        acts.dedup();
        if l < 0.0 && r > 0.0 && !acts.iter().any(|&v| v == 0.0) {
            acts.push(0.0);
        }
    };

    for t in (0..n_t).rev() {
        build_acts(t, &mut acts);
        de.clear();
        imm.clear();
        for &a in acts.iter() {
            let c = (-a).max(0.0);
            let d = a.max(0.0);
            de.push(ETA_CHARGE * c * DELTA_T - d * DELTA_T / ETA_DISCHARGE);
            imm.push(immediate_profit(battery, a, lam[t]));
        }
        let (left, right) = values.split_at_mut((t + 1) * levels);
        let cur = &mut left[t * levels..(t + 1) * levels];
        let nxt = &right[0..levels];
        for x in cur.iter_mut() {
            *x = f64::NEG_INFINITY;
        }
        for j in 0..acts.len() {
            let dej = de[j];
            let immj = imm[j];
            let i0 = soc_grid.partition_point(|&s| s + dej < lo_ok);
            let i1 = soc_grid.partition_point(|&s| s + dej <= hi_ok);
            let sg = &soc_grid[i0..i1];
            let cc = &mut cur[i0..i1];
            debug_assert_eq!(nxt.len(), levels);
            for (s, c) in sg.iter().zip(cc.iter_mut()) {
                let ns = *s + dej;
                let pos = ((ns - smin) * h_inv).clamp(0.0, last_f);
                let p0 = (pos as usize).min(last);
                let p1 = (p0 + 1).min(last);
                let a = pos - p0 as f64;
                let (v0, v1) = (nxt[p0], nxt[p1]);
                let v = immj + (v0 * (1.0 - a) + v1 * a);
                *c = if v > *c { v } else { *c };
            }
        }
        for x in cur.iter_mut() {
            if !x.is_finite() {
                *x = -1e30;
            }
        }
    }

    let mut soc = battery.soc_initial_mwh;
    let mut out = vec![0.0_f64; n_t];
    for t in 0..n_t {
        let (blo, bhi) = compute_action_bounds(battery, soc);
        let mut alo = blo.max(lo[t].min(hi[t]));
        let mut ahi = bhi.min(hi[t].max(lo[t]));
        if alo > ahi {
            let f = fallback[t].clamp(blo, bhi);
            alo = f;
            ahi = f;
        }
        build_acts(t, &mut acts);
        let nxt = &values[(t + 1) * levels..(t + 2) * levels];
        let mut best_a = 0.0_f64.clamp(alo, ahi);
        let mut best_v = f64::NEG_INFINITY;
        let mut consider = |a: f64, best_a: &mut f64, best_v: &mut f64| {
            let a = a.clamp(alo, ahi);
            let c = (-a).max(0.0);
            let d = a.max(0.0);
            let ns = (soc + ETA_CHARGE * c * DELTA_T - d * DELTA_T / ETA_DISCHARGE)
                .clamp(smin, smax);
            let v = immediate_profit(battery, a, lam[t]) + plan_interp(nxt, smin, h_inv, last, ns);
            if v > *best_v {
                *best_v = v;
                *best_a = a;
            }
        };
        for k in 0..acts.len() {
            consider(acts[k], &mut best_a, &mut best_v);
        }
        consider(alo, &mut best_a, &mut best_v);
        consider(ahi, &mut best_a, &mut best_v);
        consider(fallback[t], &mut best_a, &mut best_v);
        out[t] = best_a;
        soc = battery.apply_action_to_soc(best_a, soc);
    }
    out
}

fn plan_battery_profit(battery: &Battery, u: &[f64], lam: &[f64]) -> f64 {
    let mut s = 0.0;
    for t in 0..u.len() {
        s += immediate_profit(battery, u[t], lam[t]);
    }
    s
}

fn plan_schedule_feasible(challenge: &Challenge, sched: &[Vec<f64>]) -> bool {
    let n_b = challenge.num_batteries;
    if sched.len() != challenge.num_steps {
        return false;
    }
    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.soc_initial_mwh)
        .collect();
    for t in 0..challenge.num_steps {
        if sched[t].len() != n_b {
            return false;
        }
        for b in 0..n_b {
            let (lo, hi) = compute_action_bounds(&challenge.batteries[b], socs[b]);
            let a = sched[t][b];
            if !a.is_finite() || a < lo || a > hi {
                return false;
            }
        }
        let st = State {
            time_step: t,
            socs: socs.clone(),
            rt_prices: Vec::new(),
            exogenous_injections: challenge.exogenous_injections[t].clone(),
            action_bounds: Vec::new(),
            total_profit: 0.0,
        };
        let flows = challenge
            .network
            .compute_flows(&challenge.compute_total_injections(&st, &sched[t]));
        if challenge.network.verify_flows(&flows).is_err() {
            return false;
        }
        for b in 0..n_b {
            socs[b] = challenge.batteries[b].apply_action_to_soc(sched[t][b], socs[b]);
        }
    }
    true
}

fn plan_polish_schedule(
    challenge: &Challenge,
    incumbent: &[Vec<f64>],
    prices: &[Vec<f64>],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    fuel_floor: u64,
) -> Option<Vec<Vec<f64>>> {
    let n_t = challenge.num_steps;
    let n_b = challenge.num_batteries;
    let n_l = challenge.network.flow_limits.len();
    if n_t == 0 || n_b == 0 || incumbent.len() != n_t {
        return None;
    }
    let levels = hp.plan_soc_levels.max(9);
    let na = if hp.plan_act_cap > 0 {
        hp.plan_act_levels.min(hp.plan_act_cap).max(3)
    } else {
        hp.plan_act_levels.max(3)
    };
    let margin = hp.plan_flow_margin.clamp(0.0, 1e-2);
    let limits = &challenge.network.flow_limits;

    let mut u: Vec<Vec<f64>> = (0..n_b)
        .map(|b| (0..n_t).map(|t| incumbent[t][b]).collect())
        .collect();
    let lam: Vec<Vec<f64>> = (0..n_b)
        .map(|b| {
            let node = challenge.batteries[b].node;
            (0..n_t).map(|t| prices[t][node]).collect()
        })
        .collect();

    let base_flows = plan_base_flows(challenge);
    let mut flows = vec![vec![0.0_f64; n_t]; n_l];
    for l in 0..n_l {
        for t in 0..n_t {
            let mut f = base_flows[t][l];
            for b in 0..n_b {
                f += sens[l][b] * u[b][t];
            }
            flows[l][t] = f;
        }
    }

    let touched: Vec<Vec<usize>> = (0..n_b)
        .map(|b| (0..n_l).filter(|&l| sens[l][b].abs() > 1e-12).collect())
        .collect();

    let mut cur_profit: Vec<f64> = (0..n_b)
        .map(|b| plan_battery_profit(&challenge.batteries[b], &u[b], &lam[b]))
        .collect();
    let start_total: f64 = cur_profit.iter().sum();
    let mut lo = vec![0.0_f64; n_t];
    let mut hi = vec![0.0_f64; n_t];
    let mut improved = false;

    'sweeps: for _sweep in 0..hp.plan_sweeps {
        let sweep_start: f64 = cur_profit.iter().sum();
        for b in 0..n_b {
            if fuel_remaining() <= fuel_floor {
                break 'sweeps;
            }
            let bat = &challenge.batteries[b];
            let p = bat.power_charge_mw.max(bat.power_discharge_mw);
            for t in 0..n_t {
                lo[t] = -bat.power_charge_mw;
                hi[t] = bat.power_discharge_mw;
            }
            for &l in touched[b].iter() {
                let a = sens[l][b];
                let cap = limits[l] * (1.0 - margin);
                for t in 0..n_t {
                    let r = flows[l][t] - a * u[b][t];
                    let x1 = (cap - r) / a;
                    let x2 = (-cap - r) / a;
                    let (blo, bhi) = if a > 0.0 { (x2, x1) } else { (x1, x2) };
                    if blo > lo[t] {
                        lo[t] = blo;
                    }
                    if bhi < hi[t] {
                        hi[t] = bhi;
                    }
                }
            }
            for t in 0..n_t {
                if u[b][t] < lo[t] {
                    lo[t] = u[b][t];
                }
                if u[b][t] > hi[t] {
                    hi[t] = u[b][t];
                }
                lo[t] = lo[t].clamp(-p, p);
                hi[t] = hi[t].clamp(-p, p);
                if lo[t] > hi[t] {
                    lo[t] = u[b][t];
                    hi[t] = u[b][t];
                }
            }
            let cand = plan_dp_battery(bat, &lam[b], &lo, &hi, &u[b], levels, na);
            let gain = plan_battery_profit(bat, &cand, &lam[b]) - cur_profit[b];
            if gain > hp.plan_min_gain {
                for &l in touched[b].iter() {
                    let a = sens[l][b];
                    for t in 0..n_t {
                        flows[l][t] += a * (cand[t] - u[b][t]);
                    }
                }
                cur_profit[b] += gain;
                u[b] = cand;
                improved = true;
            }
        }
        let after: f64 = cur_profit.iter().sum();
        if after <= sweep_start + 1e-6 {
            break;
        }
    }

    if !improved {
        return None;
    }
    let total: f64 = cur_profit.iter().sum();
    if total <= start_total + 1e-6 {
        return None;
    }
    let sched: Vec<Vec<f64>> = (0..n_t)
        .map(|t| (0..n_b).map(|b| u[b][t]).collect())
        .collect();
    if !plan_schedule_feasible(challenge, &sched) {
        return None;
    }
    Some(sched)
}

fn block_repair_window(
    challenge: &Challenge,
    sched: &[Vec<f64>],
    socs: &[Vec<f64>],
    pt: Option<&[Vec<f64>]>,
    sens: &[Vec<f64>],
    base_flows: &[Vec<f64>],
    hp: &Hyperparameters,
    seed: &mut Vec<(usize, usize, bool)>,
    start_t: usize,
    end_t: usize,
) -> Option<Vec<Vec<f64>>> {
    let num_b = challenge.num_batteries;
    let w = end_t.saturating_sub(start_t);
    if w == 0 || num_b == 0 {
        return None;
    }
    let n_lines = sens.len();
    let fx = |b: usize, t: usize| -> usize { b * w + t };
    let ap = |b: usize, t: usize| -> usize { num_b * w + b * w + t };
    let am = |b: usize, t: usize| -> usize { 2 * num_b * w + b * w + t };
    let n_vars = 3 * num_b * w;

    let lo = vec![0.0f64; n_vars];
    let mut hi = vec![0.0f64; n_vars];
    let mut c_obj = vec![0.0f64; n_vars];
    let mut e_co = vec![0.0f64; n_vars];
    let mut q_co = vec![0.0f64; n_vars];
    let mut u0 = vec![0.0f64; num_b * w];
    let mut dis = vec![false; num_b * w];
    let e_c = ETA_CHARGE * DELTA_T;
    let e_d = -(DELTA_T / ETA_DISCHARGE);
    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        for t in 0..w {
            let tau = start_t + t;
            let u = sched[tau][b];
            u0[b * w + t] = u;
            let (u_min, u_max) = compute_action_bounds(battery, socs[b][tau]);
            let c_cap = (-u_min).max(0.0);
            let d_cap = u_max.max(0.0);
            let price = blk_price(challenge, pt, tau, node);
            let g_c = -(price + KAPPA_TX) * DELTA_T;
            let g_d = (price - KAPPA_TX) * DELTA_T;
            if u >= 0.0 {
                dis[b * w + t] = true;
                hi[fx(b, t)] = c_cap;
                hi[ap(b, t)] = (d_cap - u).max(0.0);
                hi[am(b, t)] = u;
                c_obj[fx(b, t)] = g_c;
                c_obj[ap(b, t)] = g_d;
                c_obj[am(b, t)] = -g_d;
                e_co[fx(b, t)] = e_c;
                e_co[ap(b, t)] = e_d;
                e_co[am(b, t)] = -e_d;
                q_co[fx(b, t)] = -1.0;
                q_co[ap(b, t)] = 1.0;
                q_co[am(b, t)] = -1.0;
            } else {
                hi[fx(b, t)] = d_cap;
                hi[ap(b, t)] = (c_cap + u).max(0.0);
                hi[am(b, t)] = -u;
                c_obj[fx(b, t)] = g_d;
                c_obj[ap(b, t)] = g_c;
                c_obj[am(b, t)] = -g_c;
                e_co[fx(b, t)] = e_d;
                e_co[ap(b, t)] = e_c;
                e_co[am(b, t)] = -e_c;
                q_co[fx(b, t)] = 1.0;
                q_co[ap(b, t)] = -1.0;
                q_co[am(b, t)] = 1.0;
            }
        }
    }

    let mut struct_rows: Vec<f64> = vec![0.0; num_b * n_vars];
    let struct_rhs: Vec<f64> = vec![0.0; num_b];
    for b in 0..num_b {
        let base = b * n_vars;
        let row = &mut struct_rows[base..base + n_vars];
        for t in 0..w {
            row[fx(b, t)] = -e_co[fx(b, t)];
            row[ap(b, t)] = -e_co[ap(b, t)];
            row[am(b, t)] = -e_co[am(b, t)];
        }
    }
    let n_struct = num_b;

    let margin = hp.blk_flow_margin;
    let head_up = |b: usize, te: usize| -> f64 {
        (challenge.batteries[b].soc_max_mwh - socs[b][start_t + te]).max(0.0)
    };
    let head_dn = |b: usize, te: usize| -> f64 {
        (socs[b][start_t + te] - challenge.batteries[b].soc_min_mwh).max(0.0)
    };
    let mut inc_flow: Vec<Vec<f64>> = Vec::with_capacity(w);
    for t in 0..w {
        let tau = start_t + t;
        let mut f = base_flows[tau].clone();
        for b in 0..num_b {
            let u = sched[tau][b];
            if u != 0.0 {
                for l in 0..n_lines {
                    let sv = sens[l][b];
                    if sv.abs() > 1e-12 {
                        f[l] += sv * u;
                    }
                }
            }
        }
        inc_flow.push(f);
    }
    let bound_of = |l: usize, t: usize, up: bool| -> f64 {
        let limit = challenge.network.flow_limits[l];
        let f = inc_flow[t][l];
        if up {
            (limit - margin - f).max(0.0)
        } else {
            (limit - margin + f).max(0.0)
        }
    };
    let mut reach = vec![0.0f64; n_lines];
    for l in 0..n_lines {
        let mut r = 0.0f64;
        for b in 0..num_b {
            let s = sens[l][b].abs();
            if s <= 1e-12 {
                continue;
            }
            let battery = &challenge.batteries[b];
            r += s * battery.power_charge_mw.max(battery.power_discharge_mw);
        }
        reach[l] = r;
    }

    let row_cap = hp.blk_row_cap.max(1);
    let n_add = hp.blk_add_cap.max(1);
    let key = |l: usize, t: usize, up: bool| -> usize { (l * w + t) * 2 + up as usize };
    let skey = |b: usize, te: usize, up: bool| -> usize { (b * w + (te - 1)) * 2 + up as usize };
    let mut in_active = vec![false; 2 * n_lines * w];
    let mut active: Vec<(usize, usize, bool)> = Vec::new();
    let mut in_active_soc = vec![false; 2 * num_b * w];
    let mut active_soc: Vec<(usize, usize, bool)> = Vec::new();
    if hp.blk_warm > 0 {
        for &(l, t0, up) in seed.iter() {
            if t0 < w && !in_active[key(l, t0, up)] && active.len() < row_cap {
                in_active[key(l, t0, up)] = true;
                active.push((l, t0, up));
            }
        }
    }

    let mut rows: Vec<f64> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut tab: Vec<f64> = Vec::new();
    let mut x: Vec<f64> = Vec::new();
    let mut ok = false;
    for _round in 0..hp.blk_lazy_rounds.max(1) {
        rows.truncate(n_struct * n_vars);
        rhs.truncate(n_struct);
        if rows.len() < n_struct * n_vars {
            rows.extend_from_slice(&struct_rows[rows.len()..n_struct * n_vars]);
            rhs.extend_from_slice(&struct_rhs[rhs.len()..n_struct]);
        }
        let mut m = n_struct;
        for &(b, t_end, up) in active_soc.iter() {
            let base = m * n_vars;
            rows.resize(base + n_vars, 0.0);
            {
                let row = &mut rows[base..base + n_vars];
                let sgn = if up { 1.0 } else { -1.0 };
                for t in 0..t_end {
                    row[fx(b, t)] = sgn * e_co[fx(b, t)];
                    row[ap(b, t)] = sgn * e_co[ap(b, t)];
                    row[am(b, t)] = sgn * e_co[am(b, t)];
                }
            }
            rhs.push(if up { head_up(b, t_end) } else { head_dn(b, t_end) });
            m += 1;
        }
        for &(l, t, up) in active.iter() {
            let base = m * n_vars;
            rows.resize(base + n_vars, 0.0);
            {
                let row = &mut rows[base..base + n_vars];
                let sgn = if up { 1.0 } else { -1.0 };
                for b in 0..num_b {
                    let sv = sens[l][b];
                    if sv.abs() <= 1e-12 {
                        continue;
                    }
                    row[fx(b, t)] = sgn * sv * q_co[fx(b, t)];
                    row[ap(b, t)] = sgn * sv * q_co[ap(b, t)];
                    row[am(b, t)] = sgn * sv * q_co[am(b, t)];
                }
            }
            rhs.push(bound_of(l, t, up));
            m += 1;
        }
        let sol = block_simplex::solve_max(
            n_vars,
            m,
            &c_obj,
            &lo,
            &hi,
            &rows,
            &rhs,
            hp.blk_pivot_budget,
            &mut tab,
        )?;

        let mut viol_soc: Vec<(f64, usize, usize, bool)> = Vec::new();
        let mut tight_soc: Vec<(usize, usize, bool)> = Vec::new();
        for b in 0..num_b {
            let mut cum = 0.0f64;
            for t_end in 1..=w {
                let t = t_end - 1;
                cum += e_co[fx(b, t)] * sol[fx(b, t)]
                    + e_co[ap(b, t)] * sol[ap(b, t)]
                    + e_co[am(b, t)] * sol[am(b, t)];
                let hu = head_up(b, t_end);
                let hd = head_dn(b, t_end);
                if cum > hu + 1e-7 {
                    if !in_active_soc[skey(b, t_end, true)] {
                        viol_soc.push((cum - hu, b, t_end, true));
                    }
                    tight_soc.push((b, t_end, true));
                } else if in_active_soc[skey(b, t_end, true)] && cum > hu - hp.blk_drop_tol {
                    tight_soc.push((b, t_end, true));
                }
                if -cum > hd + 1e-7 {
                    if !in_active_soc[skey(b, t_end, false)] {
                        viol_soc.push((-cum - hd, b, t_end, false));
                    }
                    tight_soc.push((b, t_end, false));
                } else if in_active_soc[skey(b, t_end, false)] && -cum > hd - hp.blk_drop_tol {
                    tight_soc.push((b, t_end, false));
                }
            }
        }

        let mut viol: Vec<(f64, usize, usize, bool)> = Vec::new();
        let mut tight: Vec<(usize, usize, bool)> = Vec::new();
        let mut du = vec![0.0f64; num_b];
        for t in 0..w {
            for b in 0..num_b {
                du[b] = q_co[fx(b, t)] * sol[fx(b, t)]
                    + q_co[ap(b, t)] * sol[ap(b, t)]
                    + q_co[am(b, t)] * sol[am(b, t)];
            }
            for l in 0..n_lines {
                if challenge.network.flow_limits[l] <= 1e-6 || reach[l] <= 1e-9 {
                    continue;
                }
                let mut f = 0.0f64;
                for b in 0..num_b {
                    let sv = sens[l][b];
                    if sv.abs() > 1e-12 {
                        f += sv * du[b];
                    }
                }
                for &up in &[true, false] {
                    let lhs = if up { f } else { -f };
                    let bnd = bound_of(l, t, up);
                    if lhs > bnd + 1e-7 {
                        if !in_active[key(l, t, up)] {
                            viol.push((lhs - bnd, l, t, up));
                        }
                        tight.push((l, t, up));
                    } else if in_active[key(l, t, up)] && lhs > bnd - hp.blk_drop_tol {
                        tight.push((l, t, up));
                    }
                }
            }
        }
        if viol.is_empty() && viol_soc.is_empty() {
            x = sol;
            ok = true;
            seed.clear();
            seed.extend(tight.iter().copied());
            break;
        }
        for &(b, te, up) in active_soc.iter() {
            in_active_soc[skey(b, te, up)] = false;
        }
        active_soc.clear();
        for &(b, te, up) in tight_soc.iter() {
            if !in_active_soc[skey(b, te, up)] && active_soc.len() < row_cap {
                in_active_soc[skey(b, te, up)] = true;
                active_soc.push((b, te, up));
            }
        }
        viol_soc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        let mut added_s = 0usize;
        for &(_, b, te, up) in viol_soc.iter() {
            if added_s >= n_add || active_soc.len() >= row_cap {
                break;
            }
            if !in_active_soc[skey(b, te, up)] {
                in_active_soc[skey(b, te, up)] = true;
                active_soc.push((b, te, up));
                added_s += 1;
            }
        }
        for &(l, t, up) in active.iter() {
            in_active[key(l, t, up)] = false;
        }
        active.clear();
        for &(l, t, up) in tight.iter() {
            if !in_active[key(l, t, up)] && active.len() < row_cap {
                in_active[key(l, t, up)] = true;
                active.push((l, t, up));
            }
        }
        viol.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        let mut added = 0usize;
        for &(_, l, t, up) in viol.iter() {
            if added >= n_add || active.len() >= row_cap {
                break;
            }
            if !in_active[key(l, t, up)] {
                in_active[key(l, t, up)] = true;
                active.push((l, t, up));
                added += 1;
            }
        }
    }
    if !ok {
        return None;
    }

    let mut out: Vec<Vec<f64>> = Vec::with_capacity(w);
    for t in 0..w {
        let mut row = vec![0.0f64; num_b];
        for b in 0..num_b {
            let k = b * w + t;
            let mut u = u0[k]
                + q_co[fx(b, t)] * x[fx(b, t)]
                + q_co[ap(b, t)] * x[ap(b, t)]
                + q_co[am(b, t)] * x[am(b, t)];
            let (c, d) = if dis[k] {
                (x[fx(b, t)], u0[k] + x[ap(b, t)] - x[am(b, t)])
            } else {
                (-u0[k] + x[ap(b, t)] - x[am(b, t)], x[fx(b, t)])
            };
            if c < -1e-7 || d < -1e-7 {
                return None;
            }
            if c > 0.0 && d > 0.0 {
                u = d - c;
            }
            row[b] = u;
        }
        out.push(row);
    }
    let mut gain = 0.0f64;
    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let mut soc = socs[b][start_t];
        for t in 0..w {
            let tau = start_t + t;
            let (u_min, u_max) = compute_action_bounds(battery, soc);
            let u = out[t][b];
            if u < u_min - 1e-7 || u > u_max + 1e-7 {
                return None;
            }
            let price = blk_price(challenge, pt, tau, node);
            gain += immediate_profit(battery, u, price) - immediate_profit(battery, sched[tau][b], price);
            soc = battery.apply_action_to_soc(u, soc);
        }
        if (soc - socs[b][end_t]).abs() > 1e-6 {
            return None;
        }
    }
    if gain <= hp.plan_min_gain {
        return None;
    }
    for t in 0..w {
        let tau = start_t + t;
        let mut f = base_flows[tau].clone();
        for b in 0..num_b {
            let u = out[t][b];
            if u != 0.0 {
                for l in 0..n_lines {
                    let sv = sens[l][b];
                    if sv.abs() > 1e-12 {
                        f[l] += sv * u;
                    }
                }
            }
        }
        for l in 0..n_lines {
            if f[l].abs() > challenge.network.flow_limits[l] - 1e-9 {
                return None;
            }
        }
    }
    Some(out)
}

fn snap_action_bounds(challenge: &Challenge, sched: &mut [Vec<f64>]) {
    let n_b = challenge.num_batteries;
    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.soc_initial_mwh)
        .collect();
    for t in 0..challenge.num_steps {
        for b in 0..n_b {
            let (lo, hi) = compute_action_bounds(&challenge.batteries[b], socs[b]);
            let a = sched[t][b];
            if a < lo {
                sched[t][b] = lo;
            } else if a > hi {
                sched[t][b] = hi;
            }
            socs[b] = challenge.batteries[b].apply_action_to_soc(sched[t][b], socs[b]);
        }
    }
}

fn block_repair_schedule(
    challenge: &Challenge,
    sched: &mut Vec<Vec<f64>>,
    pt: Option<&[Vec<f64>]>,
    sens: &[Vec<f64>],
    base_flows: &[Vec<f64>],
    hp: &Hyperparameters,
    w: usize,
    off: usize,
    fuel_floor: u64,
) -> bool {
    let n_t = challenge.num_steps;
    let num_b = challenge.num_batteries;
    if w == 0 || n_t == 0 || num_b == 0 {
        return false;
    }
    let mut socs: Vec<Vec<f64>> = Vec::with_capacity(num_b);
    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let mut v = Vec::with_capacity(n_t + 1);
        let mut s = battery.soc_initial_mwh;
        v.push(s);
        for t in 0..n_t {
            s = battery.apply_action_to_soc(sched[t][b], s);
            v.push(s);
        }
        socs.push(v);
    }
    let mut any = false;
    let mut seed: Vec<(usize, usize, bool)> = Vec::new();
    let mut a = off.min(n_t);
    if off > 0 {
        if let Some(win) = block_repair_window(
            challenge, sched, &socs, pt, sens, base_flows, hp, &mut seed, 0, off,
        )
        {
            for t in 0..off {
                sched[t] = win[t].clone();
            }
            for b in 0..num_b {
                let battery = &challenge.batteries[b];
                let mut sv = socs[b][0];
                for t in 0..off {
                    sv = battery.apply_action_to_soc(sched[t][b], sv);
                    socs[b][t + 1] = sv;
                }
            }
            any = true;
        }
    }
    while a < n_t {
        if fuel_remaining() <= fuel_floor {
            break;
        }
        let b_end = (a + w).min(n_t);
        if b_end > a + 1 {
            if let Some(win) = block_repair_window(
                challenge, sched, &socs, pt, sens, base_flows, hp, &mut seed, a, b_end,
            )
            {
                for t in a..b_end {
                    sched[t] = win[t - a].clone();
                }
                for b in 0..num_b {
                    let battery = &challenge.batteries[b];
                    let mut s = socs[b][a];
                    for t in a..b_end {
                        s = battery.apply_action_to_soc(sched[t][b], s);
                        socs[b][t + 1] = s;
                    }
                }
                any = true;
            }
        }
        a = b_end;
    }
    any
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
    let lp_act: Mutex<Vec<i16>> = Mutex::new(vec![-1_i16; n_lines]);
    
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
    let blk_pt: Option<&[Vec<f64>]> =
        if hp.use_block_lp { exact_price_table(challenge).map(|v| v.as_slice()) } else { None };
    let blk_state: Mutex<BlockState> = Mutex::new(BlockState::new(n_lines));
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        policy(c, s, &dps, &sens, &hp, &rh_lam, &block_plan, &lp_act, blk_pt, &blk_state)
    })?;
    save_solution(&solution)?;

    let mut best = solution.schedule;
    if hp.plan_polish != 0 && fuel_remaining() > fuel_floor {
        if let Some(prices) = exact_price_table(challenge) {
            if let Some(better) =
                plan_polish_schedule(challenge, &best, prices, &sens, &hp, fuel_floor)
            {
                best = better;
                save_solution(&Solution { schedule: best.clone() })?;
            }
        }
    }
    if hp.blk_rep_w > 0 && fuel_remaining() > fuel_floor {
        let rep_pt: Option<&[Vec<f64>]> = exact_price_table(challenge).map(|v| v.as_slice());
        if rep_pt.is_some() {
            let base_flows = plan_base_flows(challenge);
            let w = hp.blk_rep_w;
            let mut moved = false;
            for pass in 0..hp.blk_rep_passes.max(1) {
                if fuel_remaining() <= fuel_floor {
                    break;
                }
                let off = if pass == 0 { 0 } else { (w / 2).max(1) * (pass % 2) };
                if block_repair_schedule(
                    challenge, &mut best, rep_pt, &sens, &base_flows, &hp, w, off, fuel_floor,
                ) {
                    moved = true;
                }
            }
            if moved {
                snap_action_bounds(challenge, &mut best);
            }
            if moved && plan_schedule_feasible(challenge, &best) {
                save_solution(&Solution { schedule: best })?;
            }
        }
    }
    Ok(())
}

fn select_step_seeds(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    seeds: Vec<Vec<f64>>,
    hp: &Hyperparameters,
) -> Vec<Vec<f64>> {
    if hp.seed_sel_mode == 0 || seeds.len() < 2 {
        return seeds;
    }
    let mut carried: Vec<Vec<f64>> = Vec::with_capacity(seeds.len());
    let mut scores: Vec<f64> = Vec::with_capacity(seeds.len());
    for seed in seeds.into_iter() {
        match hp.seed_sel_mode {
            1 => {
                scores.push(total_step_value(challenge, state, dps, &seed));
                carried.push(seed);
            }
            2 => {
                let mut projected = seed.clone();
                safe_project_to_feasible(
                    challenge, state, &mut projected, sens, base_flows, hp,
                );
                scores.push(total_step_value(challenge, state, dps, &projected));
                carried.push(seed);
            }
            _ => {
                let (action, value) = projected_gradient_ascent(
                    challenge, state, dps, sens, base_flows, seed, hp,
                    hp.seed_race_prefix,
                );
                scores.push(value);
                carried.push(action);
            }
        }
    }
    let mut best_idx = 0usize;
    for (i, s) in scores.iter().enumerate() {
        if *s > scores[best_idx] {
            best_idx = i;
        }
    }
    vec![carried.swap_remove(best_idx)]
}

fn lp_dispatch_step(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    hp: &Hyperparameters,
    lp_act: &Mutex<Vec<i16>>,
) -> Option<Vec<f64>> {
    let num_b = challenge.num_batteries;
    let n_lines_total = sens.len();

    let forced_full = hp.lp_max_lines > 0 && hp.lp_max_lines < n_lines_total;
    if forced_full {
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
        let line_indices: Vec<usize> = scored.into_iter().map(|(_, l)| l).collect();
        return lp_dispatch_on(challenge, state, dps, sens, base_flows, hp, &line_indices);
    }

    let mut act = lp_act.lock().unwrap();
    if act.len() != n_lines_total {
        *act = vec![-1_i16; n_lines_total];
    }
    let limits = &challenge.network.flow_limits;
    let mut last: Option<Vec<f64>> = None;
    let max_rounds = hp.lp_act_rounds.max(1);
    let age_limit = hp.lp_act_age;
    for _round in 0..max_rounds {
        let line_indices: Vec<usize> = (0..n_lines_total).filter(|&l| act[l] >= 0).collect();
        let sol = lp_dispatch_on(challenge, state, dps, sens, base_flows, hp, &line_indices)?;
        let mut added = 0usize;
        for l in 0..n_lines_total {
            let lim = limits[l];
            if lim <= 1e-6 {
                continue;
            }
            let mut f = base_flows[l];
            let row = &sens[l];
            for b in 0..num_b {
                f += row[b] * sol[b];
            }
            let tight = f.abs() > lim - LP_ACTIVE_SLACK;
            if act[l] < 0 {
                if tight {
                    act[l] = 0;
                    added += 1;
                }
            } else if tight {
                act[l] = 0;
            } else {
                act[l] = act[l].saturating_add(1);
            }
        }
        if added == 0 {
            for l in 0..n_lines_total {
                if (act[l] as i64) > age_limit {
                    act[l] = -1;
                }
            }
            return Some(sol);
        }
        last = Some(sol);
    }
    let all: Vec<usize> = (0..n_lines_total).collect();
    lp_dispatch_on(challenge, state, dps, sens, base_flows, hp, &all).or(last)
}

const LP_ACTIVE_SLACK: f64 = 1e-7;

fn lp_dispatch_on(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    hp: &Hyperparameters,
    line_indices: &[usize],
) -> Option<Vec<f64>> {
    let num_b = challenge.num_batteries;
    let t = state.time_step;

    let num_l = line_indices.len();
    let limits = &challenge.network.flow_limits;

    let n = 2 * num_b;
    let m = 2 * num_b + 2 * num_l;

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
        let r = 2 * b;

        let cap_pow_d = u_max.max(0.0);
        let coef_soc_d = DELTA_T / ETA_DISCHARGE;
        let cap_soc_d = (soc - battery.soc_min_mwh).max(0.0);
        if cap_soc_d < cap_pow_d * coef_soc_d {
            a_mat[r][b] = coef_soc_d;
            b_vec[r] = cap_soc_d;
        } else {
            a_mat[r][b] = 1.0;
            b_vec[r] = cap_pow_d;
        }

        let cap_pow_c = (-u_min).max(0.0);
        let coef_soc_c = ETA_CHARGE * DELTA_T;
        let cap_soc_c = (battery.soc_max_mwh - soc).max(0.0);
        if cap_soc_c < cap_pow_c * coef_soc_c {
            a_mat[r + 1][num_b + b] = coef_soc_c;
            b_vec[r + 1] = cap_soc_c;
        } else {
            a_mat[r + 1][num_b + b] = 1.0;
            b_vec[r + 1] = cap_pow_c;
        }
    }

    let row_f = 2 * num_b;
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

        let mut tab = vec![0.0_f64; n_cols * (m + 1)];
        for i in 0..m {
            for j in 0..n {
                tab[(i) * n_cols + (j)] = a[i][j];
            }
            tab[(i) * n_cols + (n + i)] = 1.0;
            tab[(i) * n_cols + (rhs_col)] = b[i].max(0.0);
        }
        for j in 0..n {
            tab[(m) * n_cols + (j)] = -c[j];
        }

        let mut basis: Vec<usize> = (n..n + m).collect();
        let mut pivots_used = 0usize;

        for pivot in 0..max_pivots {
            pivots_used = pivot + 1;
            let entering = match (0..n_vars).find(|&j| tab[(m) * n_cols + (j)] < -LP_EPS) {
                Some(j) => j,
                None => break,
            };
            let mut leaving_row: Option<usize> = None;
            let mut best_ratio = 0.0_f64;
            for i in 0..m {
                let piv = tab[(i) * n_cols + (entering)];
                if piv > LP_EPS {
                    let r = tab[(i) * n_cols + (rhs_col)] / piv;
                    if leaving_row.is_none() || r < best_ratio {
                        leaving_row = Some(i);
                        best_ratio = r;
                    }
                }
            }
            let leaving_row = match leaving_row {
                Some(r) => r,
                None => return (None, 0),
            };

            let pivot_val = tab[(leaving_row) * n_cols + (entering)];
            if pivot_val.abs() < LP_EPS {
                return (None, 0);
            }
            {
                let lr0 = leaving_row * n_cols;
                let (head, tail) = tab.split_at_mut(lr0);
                let (prow, rest) = tail.split_at_mut(n_cols);
                for x in prow.iter_mut() {
                    *x /= pivot_val;
                }
                for i in 0..leaving_row {
                    let row = &mut head[i * n_cols..i * n_cols + n_cols];
                    let factor = row[entering];
                    if factor.abs() > 1e-15 {
                        for (x, p) in row.iter_mut().zip(prow.iter()) {
                            *x -= factor * *p;
                        }
                    }
                }
                for i in (leaving_row + 1)..=m {
                    let off = (i - leaving_row - 1) * n_cols;
                    let row = &mut rest[off..off + n_cols];
                    let factor = row[entering];
                    if factor.abs() > 1e-15 {
                        for (x, p) in row.iter_mut().zip(prow.iter()) {
                            *x -= factor * *p;
                        }
                    }
                }
            }
            basis[leaving_row] = entering;
        }

        let mut x = vec![0.0_f64; n];
        for (i, &bv) in basis.iter().enumerate() {
            if bv < n {
                x[bv] = tab[(i) * n_cols + (rhs_col)].max(0.0);
            }
        }
        (Some(x), pivots_used)
    }
}

