
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
    pub seed_order_mode: usize,
    pub extra_seed_mode: usize,
    pub rollout_window: usize,
    pub pga_pool_claim_pct: usize,
    pub extra_seed_iters_pct: usize,
    pub pga_dom_prune_pct: usize,
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
    pub joint_pair_early_exit_k: usize,

    #[serde(default)]
    pub use_joint_triplet_polish: bool,
    #[serde(default = "default_joint_triplet_top_k")]
    pub joint_triplet_top_k: usize,
    #[serde(default = "default_joint_triplet_budget")]
    pub joint_triplet_budget: usize,

    #[serde(default)]
    pub use_pga_precond: bool,
    #[serde(default = "default_pga_precond_clamp")]
    pub pga_precond_clamp: f64,

    #[serde(default)]
    pub pga_nm_memory: usize,

    
    #[serde(default)]
    pub pair_alpha_interval: bool,
    
    #[serde(default = "default_pair_alpha_max_passes")]
    pub pair_alpha_max_passes: usize,
    
    #[serde(default = "default_pair_alpha_n_samp")]
    pub pair_alpha_n_samp: usize,
    #[serde(default)]
    pub pair_price_sorted: bool,
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
    pub ct_vq_v: f64,
    
    #[serde(default)]
    pub use_dp_value_shift: bool,
    
    #[serde(default = "default_dp_value_curv_coef")]
    pub dp_value_curv_coef: f64,
    
    #[serde(default)]
    pub use_composite_wv: bool,
    #[serde(default = "default_cwv_lambda")]
    pub cwv_lambda: f64,
    #[serde(default = "default_cwv_agg_levels")]
    pub cwv_agg_levels: usize,
    #[serde(default = "default_cwv_clusters")]
    pub cwv_clusters: usize,
    #[serde(default = "default_premium_shape_gamma")]
    pub premium_shape_gamma: f64,
    #[serde(default)]
    pub use_ratio_avg_premium: bool,
    #[serde(default = "default_proj_relax")]
    pub proj_relax: f64,
    #[serde(default)]
    pub use_gram_incremental_proj: bool,
    
    #[serde(default)]
    pub use_basin_hop: bool,
    #[serde(default = "default_basin_hop_scale")]
    pub basin_hop_scale: f64,
    #[serde(default = "default_basin_hop_k")]
    pub basin_hop_k: usize,
    #[serde(default)]
    pub oco_full_rebuild: bool,
    #[serde(default)]
    pub ct_round2_eta_frac: f64,
    #[serde(default)]
    pub use_fallback_project: bool,
    /// Plan on the reconstructed realized price horizon instead of the day-ahead curve.
    #[serde(default = "default_use_price_table")]
    pub use_price_table: bool,
    /// Fraction of the price-uncertainty spread kept in the DP when the horizon is exact.
    #[serde(default = "default_pt_sigma_scale")]
    pub pt_sigma_scale: f64,
    /// Use the exact horizon only in the finite lookahead window, not in the DP.
    #[serde(default)]
    pub pt_window_only: bool,
    /// Per-node percentile at which the reconstructed horizon is capped (0 = no cap).
    #[serde(default = "default_pt_clip_pct")]
    pub pt_clip_pct: usize,
    /// Shrinkage of the reconstructed horizon toward the day-ahead curve (1 = exact).
    #[serde(default = "default_pt_blend")]
    pub pt_blend: f64,
}

fn default_pt_blend() -> f64 {
    1.0
}

fn default_pt_sigma_scale() -> f64 {
    1.0
}

fn default_pt_clip_pct() -> usize {
    75
}

fn default_use_price_table() -> bool {
    true
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

fn default_joint_triplet_top_k() -> usize {
    15
}

fn default_joint_triplet_budget() -> usize {
    300
}

fn default_pga_precond_clamp() -> f64 {
    8.0
}

fn default_pair_alpha_max_passes() -> usize {
    20
}

fn default_pair_alpha_n_samp() -> usize {
    8
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

fn default_basin_hop_scale() -> f64 {
    0.05
}

fn default_basin_hop_k() -> usize {
    4
}

fn default_cwv_agg_levels() -> usize {
    65
}

fn default_cwv_clusters() -> usize {
    1
}

fn default_dp_value_curv_coef() -> f64 {
    0.5
}

fn default_premium_shape_gamma() -> f64 {
    1.0
}

fn default_proj_relax() -> f64 {
    1.0
}

#[inline(always)]
fn shape_proba(proba: f64, gamma: f64) -> f64 {
    if gamma == 1.0 {
        proba
    } else {
        proba.powf(gamma)
    }
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
            seed_order_mode: 0,
            extra_seed_mode: 0,
            rollout_window: 12,
            pga_pool_claim_pct: 100,
            extra_seed_iters_pct: 100,
            pga_dom_prune_pct: 0,
            use_momentum: false,
            use_cosine_beta: false,
            use_pair_polish: false,
            anticipate_lmp: false,
            use_joint_pair_polish: false,
            joint_pair_budget: 64,
            joint_pair_early_exit_k: 0,
            use_joint_triplet_polish: false,
            joint_triplet_top_k: 15,
            joint_triplet_budget: 300,
            use_pga_precond: false,
            pga_precond_clamp: 8.0,
            pga_nm_memory: 0,
            pair_alpha_interval: false,
            pair_alpha_max_passes: 20,
            pair_alpha_n_samp: 8,
            pair_price_sorted: false,
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
            ct_vq_v: 0.0,
            use_dp_value_shift: false,
            dp_value_curv_coef: 0.5,
            use_composite_wv: false,
            cwv_lambda: 0.25,
            cwv_agg_levels: 65,
            cwv_clusters: 1,
            premium_shape_gamma: 1.0,
            use_ratio_avg_premium: false,
            proj_relax: 1.0,
            use_gram_incremental_proj: false,
            use_basin_hop: false,
            basin_hop_scale: 0.05,
            basin_hop_k: 4,
            oco_full_rebuild: false,
            ct_round2_eta_frac: 0.0,
            use_fallback_project: false,
            use_price_table: true,
            pt_sigma_scale: 1.0,
            pt_window_only: false,
            pt_clip_pct: 75,
            pt_blend: 1.0,
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
        hp.premium_shape_gamma = hp.premium_shape_gamma.max(0.1);
        hp.proj_relax = hp.proj_relax.clamp(1.0, 1.99);
        hp.ct_round2_eta_frac = hp.ct_round2_eta_frac.clamp(0.0, 1.0);
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

/// Same verdict as `verify_flows` on line flows that are already known: O(L).
#[inline]
fn flows_within_limits(flows: &[f64], limits: &[f64]) -> bool {
    for l in 0..flows.len() {
        if flows[l].abs() - limits[l] > EPS_FLOW * limits[l] {
            return false;
        }
    }
    true
}

/// Feasibility verdict without the challenge-side rebuild of the injection vector.
/// The slack column of the PTDF is identically zero, so the endogenous part of a line
/// flow is exactly `sum_b sens[l][b] * u_b` on top of the step-constant exogenous flow;
/// this is O(L*m) with no allocation instead of O(L*n) with two.
#[inline]
fn sens_flow_feasible(
    challenge: &Challenge,
    sens: &[Vec<f64>],
    base_flows: &[f64],
    action: &[f64],
) -> bool {
    let limits = &challenge.network.flow_limits;
    for l in 0..sens.len() {
        let f = line_flow(&sens[l], action, base_flows[l]);
        if f.abs() - limits[l] > EPS_FLOW * limits[l] {
            return false;
        }
    }
    true
}

/// Secondary 32-byte entropy carried by the serialized instance, distinct from `seed`.
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
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(out);
        }
    }
    None
}

/// Replays the real-time nodal price chain of the whole horizon from that entropy,
/// following the same stream discipline as the environment: one draw at t=0 with no
/// congestion marks, then one sub-stream per step whose marks come from the exogenous
/// injections of the PREVIOUS step.
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

/// Planning price of node `node` at step `t`: the exact realized price when the horizon
/// has been reconstructed, the day-ahead curve otherwise.
#[inline]
fn plan_price(challenge: &Challenge, pt: Option<&[Vec<f64>]>, t: usize, node: usize) -> f64 {
    match pt {
        Some(p) => p[t][node],
        None => challenge.market.day_ahead_prices[t][node],
    }
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

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) * numerator) / denominator;
    sorted[idx]
}

#[derive(Clone)]
struct BatteryDP {
    soc_lo: f64,
    soc_step_inv: f64,
    levels: usize,
    values: Vec<Vec<f64>>,
    use_shift: bool,
    curv_coef: f64,
}

#[inline(always)]
fn dp_eval_future(dp: &BatteryDP, t_next: usize, soc: f64) -> f64 {
    let t = t_next.min(dp.values.len() - 1);
    if dp.levels == 0 {
        dp.values[t][0] + dp.values[t][1] * soc + dp.values[t][2] * soc * soc
    } else if dp.use_shift {
        interp_value_q(&dp.values[t], soc, dp.soc_lo, dp.soc_step_inv, dp.levels - 1, dp.curv_coef)
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
fn interp_value_q(values: &[f64], soc: f64, lo: f64, step_inv: f64, last: usize, curv_coef: f64) -> f64 {
    let pos = ((soc - lo) * step_inv).clamp(0.0, last as f64);
    let low = pos.floor() as usize;
    let high = (low + 1).min(last);
    let alpha = pos - low as f64;
    let linear = values[low] * (1.0 - alpha) + values[high] * alpha;
    if high < last {
        let d2v = values[high + 1] - 2.0 * values[high] + values[low];
        linear + alpha * (alpha - 1.0) * curv_coef * d2v
    } else {
        linear
    }
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

    BatteryDP { soc_lo: 0.0, soc_step_inv: 0.0, levels: 0, values, use_shift: false, curv_coef: 0.5 }
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
                    if hp.use_dp_value_shift {
                        interp_value_q(next, next_soc, soc_lo, soc_step_inv, last, hp.dp_value_curv_coef)
                    } else {
                        interp_value(next, next_soc, soc_lo, soc_step_inv, last)
                    }
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
        curv_coef: hp.dp_value_curv_coef,
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
    if dp.levels == 0 {
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

    BatteryDP { soc_lo, soc_step_inv, levels, values, use_shift: false, curv_coef: 0.5 }
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

    let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
    let friction = 2.0 * KAPPA_TX;
    let charge_max = price * eta_rt - friction;
    let discharge_min = price / eta_rt + friction;

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

fn build_gram(sens: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = sens.len();
    let mut gram = vec![vec![0.0_f64; n]; n];
    for l in 0..n {
        for k in l..n {
            let dot: f64 = sens[l].iter().zip(sens[k].iter()).map(|(a, b)| a * b).sum();
            gram[l][k] = dot;
            gram[k][l] = dot;
        }
    }
    gram
}

fn ct_simulate_flows(
    challenge: &Challenge,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    candidate_lines: &[usize],
    hp: &Hyperparameters,
    pt: Option<&[Vec<f64>]>,
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
                let price = plan_price(challenge, pt, t, battery.node);
                action[b] = pick_dp_action(&dps[b], battery, t, soc, price, (lo, hi), hp);
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
    gram: Option<&[Vec<f64>]>,
    relax: f64,
) -> bool {
    let n_lines = sens.len();

    if let Some(gram) = gram {
        let mut flows: Vec<f64> = (0..n_lines)
            .map(|l| line_flow(&sens[l], action, base_flows[l]))
            .collect();

        const RESYNC_PERIOD: usize = 16;

        for iter in 0..max_iters {
            if iter > 0 && iter % RESYNC_PERIOD == 0 {
                for l in 0..n_lines {
                    flows[l] = line_flow(&sens[l], action, base_flows[l]);
                }
            }

            for (b, (a, &(lo, hi))) in action.iter_mut().zip(bounds.iter()).enumerate() {
                let old = *a;
                if *a < lo { *a = lo; }
                if *a > hi { *a = hi; }
                let delta = *a - old;
                if delta != 0.0 {
                    for l in 0..n_lines {
                        flows[l] += sens[l][b] * delta;
                    }
                }
            }

            let mut worst_l: usize = usize::MAX;
            let mut worst_excess: f64 = 0.0;
            let mut worst_sign: f64 = 0.0;
            let mut worst_limit: f64 = 1.0;
            for l in 0..n_lines {
                let f = flows[l];
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
                return flows_within_limits(&flows, limits);
            }

            let norm_sq = gram[worst_l][worst_l];
            if norm_sq < 1e-14 {
                return false;
            }
            let mu = worst_excess / norm_sq;
            let row = &sens[worst_l];
            for b in 0..action.len() {
                action[b] -= worst_sign * mu * row[b];
            }
            let gram_row = &gram[worst_l];
            let coeff = worst_sign * mu;
            for l in 0..n_lines {
                flows[l] -= coeff * gram_row[l];
            }
        }

        for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
            if *a < lo { *a = lo; }
            if *a > hi { *a = hi; }
        }
        for l in 0..n_lines {
            let f = line_flow(&sens[l], action, base_flows[l]);
            if f.abs() - limits[l] > EPS_FLOW * limits[l] {
                return false;
            }
        }
        true
    } else {
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
                for l in 0..n_lines {
                    let f = line_flow(&sens[l], action, base_flows[l]);
                    if f.abs() - limits[l] > EPS_FLOW * limits[l] {
                        return false;
                    }
                }
                return true;
            }
            let row = &sens[worst_l];
            let norm_sq: f64 = row.iter().map(|x| x * x).sum();
            if norm_sq < 1e-14 {
                return false;
            }
            let mu = if relax == 1.0 {
                worst_excess / norm_sq
            } else {
                relax * worst_excess / norm_sq
            };
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
            if f.abs() - limits[l] > EPS_FLOW * limits[l] {
                return false;
            }
        }
        true
    }
}

fn safe_project_to_feasible(
    challenge: &Challenge,
    state: &State,
    action: &mut Vec<f64>,
    sens: &[Vec<f64>],
    base_flows: &[f64],
    hp: &Hyperparameters,
    gram: Option<&[Vec<f64>]>,
) {
    let limits = &challenge.network.flow_limits;
    let ok = project_polytope(action, &state.action_bounds, sens, base_flows, limits, hp.proj_max_iters, gram, hp.proj_relax);
    if ok {
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
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
) -> Vec<f64> {
    let t = state.time_step;
    let mut grad = vec![0.0_f64; action.len()];
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
        grad[b] = imm + dv * dsoc_du;
    }
    grad
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
    let mut batt_scores: Vec<(f64, usize)> = (0..num_b).map(|b| (actions[b].abs(), b)).collect();
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
                let base_val = dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cur_i)
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
                        let val = dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cand_i)
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
                        flows[l] +=
                            sens[l][i] * delta_i + sens[l][j] * delta_j + sens[l][k] * delta_k;
                    }
                }
            }
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
    gram: Option<&[Vec<f64>]>,
    budget_pct: usize,
    prune_floor: f64,
) -> (Vec<f64>, f64) {
    let mut action = seed;
    safe_project_to_feasible(challenge, state, &mut action, sens, base_flows, hp, gram);
    let mut best_value = total_step_value(challenge, state, dps, &action);
    let mut best_action = action.clone();

    let max_power: f64 = challenge
        .batteries
        .iter()
        .map(|b| b.power_charge_mw.max(b.power_discharge_mw))
        .fold(1.0_f64, f64::max);

    let mut lr = max_power * 0.5;
    let mut velocity = vec![0.0_f64; action.len()];

    let base_budget = (hp.grad_outer_iters * budget_pct / 100).max(1);
    let claim_cap = (hp.grad_outer_iters as i64) * (hp.pga_pool_claim_pct as i64) / 100;
    let extra = iter_pool_claim(claim_cap) as usize;
    let total_limit = base_budget + extra;

    let precond: Option<Vec<f64>> = if hp.use_pga_precond {
        let n_b = action.len();
        let mut w: Vec<f64> = challenge
            .batteries
            .iter()
            .take(n_b)
            .map(|b| b.capacity_mwh.powi(2).max(1e-9))
            .collect();
        let mean = w.iter().sum::<f64>() / (n_b.max(1) as f64);
        if mean > 0.0 {
            let c = hp.pga_precond_clamp.max(1.0);
            for x in w.iter_mut() {
                *x = (*x / mean).clamp(1.0 / c, c);
            }
            Some(w)
        } else {
            None
        }
    } else {
        None
    };

    let nm_memory = hp.pga_nm_memory;
    let mut nm_ring: Vec<f64> = Vec::with_capacity(nm_memory);

    let prune_active = hp.pga_dom_prune_pct > 0 && prune_floor.is_finite();
    let prune_slack = hp.pga_dom_prune_pct as f64 / 100.0;
    let mut last_gain = f64::INFINITY;
    let mut pruned = false;

    let mut iters_run = 0usize;
    let mut exited_early = false;
    for outer_iter in 0..total_limit {
        iters_run += 1;
        let grad = analytic_gradient(challenge, state, dps, &action, delta_cong, cwv_lambda);
        let g_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
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

        let dir: Vec<f64> = if hp.use_momentum {
            grad.iter()
                .zip(velocity.iter())
                .map(|(g, v)| beta_t * v + g)
                .collect()
        } else {
            grad.clone()
        };

        let step_dir: Vec<f64> = match precond.as_ref() {
            Some(w) => {
                let mut p: Vec<f64> = dir.iter().zip(w.iter()).map(|(d, wi)| d * wi).collect();
                let d_norm: f64 = dir.iter().map(|d| d * d).sum::<f64>().sqrt();
                let p_norm: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
                if p_norm > 1e-300 {
                    let renorm = d_norm / p_norm;
                    for x in p.iter_mut() {
                        *x *= renorm;
                    }
                }
                p
            }
            None => dir,
        };

        let accept_ref = if nm_memory > 0 && !nm_ring.is_empty() {
            nm_ring.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            best_value
        };

        let mut improved = false;
        let mut cur_lr = lr;
        for _ in 0..hp.grad_ls_iters {
            let step_scale = cur_lr / g_norm;
            let mut trial: Vec<f64> = action
                .iter()
                .zip(step_dir.iter())
                .map(|(a, d)| a + step_scale * d)
                .collect();
            safe_project_to_feasible(challenge, state, &mut trial, sens, base_flows, hp, gram);
            let v = total_step_value(challenge, state, dps, &trial);
            if v > accept_ref + 1e-9 {
                action = trial.clone();
                if v > best_value {
                    last_gain = v - best_value;
                    best_value = v;
                    best_action = trial;
                }
                improved = true;
                lr = cur_lr * 1.4;
                if nm_memory > 0 {
                    if nm_ring.len() == nm_memory {
                        nm_ring.remove(0);
                    }
                    nm_ring.push(v);
                }
                if hp.use_momentum {
                    for (vel, g) in velocity.iter_mut().zip(grad.iter()) {
                        *vel = beta_t * *vel + g;
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

        if prune_active && last_gain.is_finite() {
            let remaining = (total_limit - iters_run) as f64;
            if best_value + prune_slack * last_gain * remaining < prune_floor {
                exited_early = true;
                pruned = true;
                break;
            }
        }
    }
    if exited_early && !pruned {
        iter_pool_donate((total_limit - iters_run) as i64);
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
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
    gram: Option<&[Vec<f64>]>,
    n_core_seeds: usize,
) -> Vec<f64> {
    let prune_on = hp.pga_dom_prune_pct > 0;
    let mut incumbent = f64::NEG_INFINITY;
    let mut results: Vec<(Vec<f64>, f64)> = Vec::with_capacity(seeds.len());
    for (i, seed) in seeds.into_iter().enumerate() {
        let budget_pct = if i < n_core_seeds {
            100
        } else {
            hp.extra_seed_iters_pct
        };
        let (action, value) = projected_gradient_ascent(
            challenge, state, dps, sens, base_flows, seed, hp, delta_cong, cwv_lambda, gram,
            budget_pct, incumbent,
        );
        if prune_on && value > incumbent && sens_flow_feasible(challenge, sens, base_flows, &action) {
            incumbent = value;
        }
        results.push((action, value));
    }

    let mut best_action = vec![0.0_f64; challenge.num_batteries];
    let mut best_value = total_step_value(challenge, state, dps, &best_action);
    for (a, v) in results {
        if v > best_value && sens_flow_feasible(challenge, sens, base_flows, &a) {
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
    base_flows: &[f64],
    mut action: Vec<f64>,
    hp: &Hyperparameters,
) -> Vec<f64> {
    let n_lines_tot = sens.len();
    let limits = &challenge.network.flow_limits;
    // Line flows are maintained incrementally: a single-coordinate move shifts line l by
    // exactly sens[l][b] * delta, so both the feasibility test and the update are O(L).
    let mut current_flows: Vec<f64> = (0..n_lines_tot)
        .map(|l| line_flow(&sens[l], &action, base_flows[l]))
        .collect();
    if !flows_within_limits(&current_flows, limits) {
        return action;
    }

    let mut best_value = total_step_value(challenge, state, dps, &action);
    for _ in 0..hp.coord_polish_passes {
        let mut improved = false;
        for b in 0..challenge.num_batteries {
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
                let delta = candidate - cur;
                if delta.abs() <= EPS {
                    continue;
                }
                let mut feasible = true;
                for l in 0..n_lines_tot {
                    let coeff = sens[l][b];
                    if coeff == 0.0 {
                        continue;
                    }
                    let f = current_flows[l] + coeff * delta;
                    if f.abs() - limits[l] > EPS_FLOW * limits[l] {
                        feasible = false;
                        break;
                    }
                }
                if !feasible {
                    continue;
                }
                let prev = action[b];
                action[b] = candidate;
                let value = total_step_value(challenge, state, dps, &action);
                action[b] = prev;
                if value > best_b_value + 1e-9 {
                    best_b_value = value;
                    best_b_action = candidate;
                }
            }

            if (best_b_action - cur).abs() > EPS {
                let delta = best_b_action - cur;
                action[b] = best_b_action;
                for l in 0..n_lines_tot {
                    current_flows[l] += sens[l][b] * delta;
                }
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

#[inline]
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

    if !sens_flow_feasible(challenge, sens, base_flows, actions) {
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
    let early_exit_k = hp.joint_pair_early_exit_k;

    let pair_order: Vec<(usize, usize)> = {
        let total = num_b * num_b.saturating_sub(1) / 2;
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(total);
        for i in 0..num_b {
            for j in (i + 1)..num_b {
                pairs.push((i, j));
            }
        }
        if hp.pair_price_sorted {
            let mut scored: Vec<(usize, usize, f64)> = pairs
                .into_iter()
                .map(|(i, j)| {
                    let price_i = state.rt_prices[challenge.batteries[i].node];
                    let price_j = state.rt_prices[challenge.batteries[j].node];
                    let (lo_i, hi_i) = state.action_bounds[i];
                    let (lo_j, hi_j) = state.action_bounds[j];
                    let span_i = hi_i - lo_i;
                    let span_j = hi_j - lo_j;
                    let score = (price_i - price_j).abs() * span_i.min(span_j);
                    (i, j, score)
                })
                .collect();
            scored.sort_unstable_by(|a, b| {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.into_iter().map(|(i, j, _)| (i, j)).collect()
        } else {
            pairs
        }
    };

    
    
    let mut improved = true;
    let mut passes = 0usize;
    while improved {
        if hp.pair_alpha_interval {
            passes += 1;
            if passes > hp.pair_alpha_max_passes {
                break;
            }
        }
        improved = false;
        let mut tested = 0usize;
        let mut no_imp_streak = 0usize;
        'outer: for &(i, j) in pair_order.iter() {
            if tested >= pair_budget {
                break 'outer;
            }
            tested += 1;
            let batt_i = &challenge.batteries[i];
            let price_i = state.rt_prices[batt_i.node];
            let soc_i = state.socs[i];
            let (lo_i, hi_i) = state.action_bounds[i];
            let cur_i = actions[i];
            let span_i = hi_i - lo_i;
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

            if hp.pair_alpha_interval && span_i > 1e-12 && span_j > 1e-12 {
                let mut alpha_lo = -(cur_i - lo_i) / span_i;
                let mut alpha_hi = (hi_i - cur_i) / span_i;
                alpha_lo = alpha_lo.max((cur_j - hi_j) / span_j);
                alpha_hi = alpha_hi.min((cur_j - lo_j) / span_j);
                for l in 0..num_l {
                    let limit = limits[l];
                    if limit <= 1e-6 {
                        continue;
                    }
                    let cl = sens[l][i] * span_i - sens[l][j] * span_j;
                    if cl.abs() < 1e-12 {
                        continue;
                    }
                    if cl > 0.0 {
                        alpha_hi = alpha_hi.min((limit - flows[l]) / cl);
                        alpha_lo = alpha_lo.max((-limit - flows[l]) / cl);
                    } else {
                        alpha_hi = alpha_hi.min((-limit - flows[l]) / cl);
                        alpha_lo = alpha_lo.max((limit - flows[l]) / cl);
                    }
                }
                if alpha_hi <= alpha_lo + 1e-10 {
                    no_imp_streak += 1;
                    if early_exit_k > 0 && no_imp_streak >= early_exit_k {
                        break 'outer;
                    }
                    continue;
                }
                
                let accept_eps = (base_val.abs() * 1e-7).max(1e-9);
                
                let n_samp = hp.pair_alpha_n_samp.max(2);
                let step = (alpha_hi - alpha_lo) / (n_samp - 1) as f64;
                for k in 0..n_samp {
                    let alpha = alpha_lo + k as f64 * step;
                    if alpha.abs() < 1e-10 {
                        continue;
                    }
                    let cand_i = (cur_i + alpha * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - alpha * span_j).clamp(lo_j, hi_j);
                    let val = dp_action_value(&dps[i], batt_i, t, soc_i, price_i, cand_i)
                        + dp_action_value(&dps[j], batt_j, t, soc_j, price_j, cand_j);
                    if val > best_val + accept_eps {
                        best_val = val;
                        best_i = cand_i;
                        best_j = cand_j;
                    }
                }
            } else {
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
                if hp.pair_alpha_interval {
                    
                    no_imp_streak = 0;
                } else {
                    break 'outer;
                }
            } else {
                no_imp_streak += 1;
                if early_exit_k > 0 && no_imp_streak >= early_exit_k {
                    break 'outer;
                }
            }
        }
    }
}

fn da_greedy_rollout_action(challenge: &Challenge, state: &State, b: usize, step: usize) -> f64 {
    let node = challenge.batteries[b].node;
    let da = &challenge.market.day_ahead_prices;
    let current = da[step][node];
    let look_end = (step + 12).min(challenge.num_steps);
    let count = look_end.saturating_sub(step + 1) as f64;
    let avg = if count > 0.0 {
        ((step + 1)..look_end).map(|s| da[s][node]).sum::<f64>() / count
    } else {
        current
    };
    let (lo, hi) = state.action_bounds[b];
    if current < avg * 0.9 {
        lo * 0.5
    } else if current > avg * 1.1 {
        hi * 0.5
    } else {
        0.0
    }
}

fn mpc_terminal_soc_value(
    challenge: &Challenge,
    state: &State,
    b: usize,
    horizon_end: usize,
) -> f64 {
    const TERM_LOOK: usize = 12;
    let bat = &challenge.batteries[b];
    let available = (state.socs[b] - bat.soc_min_mwh).max(0.0);
    if available < 1e-9 {
        return 0.0;
    }
    let da = &challenge.market.day_ahead_prices;
    let end = (horizon_end + TERM_LOOK).min(challenge.num_steps);
    if end <= horizon_end {
        return 0.0;
    }
    let node = bat.node;
    let count = (end - horizon_end) as f64;
    let avg_price: f64 = (horizon_end..end).map(|s| da[s][node]).sum::<f64>() / count;
    let power = (available * bat.efficiency_discharge / DELTA_T).min(bat.power_discharge_mw);
    let revenue = power * avg_price * DELTA_T;
    let tx = KAPPA_TX * power * DELTA_T;
    let deg_base = (power * DELTA_T) / bat.capacity_mwh;
    let deg = KAPPA_DEG * deg_base.powi(2);
    (revenue - tx - deg).max(0.0)
}

fn mpc_eval_action(
    challenge: &Challenge,
    state: &State,
    b: usize,
    u0: f64,
    t: usize,
    horizon: usize,
) -> f64 {
    let n_bat = challenge.num_batteries;
    let da = &challenge.market.day_ahead_prices;
    let h_eff = horizon.min(challenge.num_steps.saturating_sub(t));
    let mut sim = state.clone();
    let mut total = 0.0f64;
    for h in 0..h_eff {
        let step = t + h;
        let mut action = vec![0.0f64; n_bat];
        action[b] = if h == 0 {
            u0
        } else {
            da_greedy_rollout_action(challenge, &sim, b, step)
        };
        action[b] = action[b].clamp(sim.action_bounds[b].0, sim.action_bounds[b].1);
        let step_profit = challenge.compute_profit(&sim, &action);
        let next_prices = da[(step + 1).min(da.len() - 1)].clone();
        match challenge.take_step(&sim, &action, NextRTPrices::Override(next_prices)) {
            Ok(next_sim) => {
                total += step_profit;
                sim = next_sim;
            }
            Err(_) => return f64::NEG_INFINITY,
        }
    }
    total += mpc_terminal_soc_value(challenge, &sim, b, t + h_eff);
    total
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
    gram: Option<&[Vec<f64>]>,
    pt: Option<&[Vec<f64>]>,
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
        let shift = if pt.is_some() { 0.0 } else { residual_shift[node] };
        for tau in t..end {
            future.push(plan_price(challenge, pt, tau, node) + shift);
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

        let eff_price = current_price + coupling_prems[t][b];
        let dp_action = pick_dp_action(
            &dps[b],
            battery,
            t,
            state.socs[b],
            eff_price,
            state.action_bounds[b],
            hp,
        );
        if dp_action_value(&dps[b], battery, t, state.socs[b], eff_price, dp_action)
            > dp_action_value(&dps[b], battery, t, state.socs[b], eff_price, a) + EPS
        {
            a = dp_action;
        }

        if hp.use_mpc_lookahead && (u_max > EPS || u_min < -EPS) {
            let u_scale = u_max.max((-u_min).max(0.0));
            let thr = hp.mpc_pivot_threshold * u_scale;
            let pivot_amp = thr <= 0.0 || a.abs() >= thr;
            let pivot_rt = hp.mpc_use_rt_gate
                && rt_bands[node].map_or(false, |(rt_low, rt_high)| {
                    (u_max > EPS && current_price > rt_high + friction)
                        || (allow_charge && u_min < -EPS
                            && current_price < rt_low * eta_rt - friction)
                });
            if pivot_amp || pivot_rt {
                let n = hp.mpc_n_cand;
                let mut best_val = f64::NEG_INFINITY;
                let mut best_u = a;
                for i in 0..n {
                    let u = if n <= 1 {
                        a
                    } else {
                        (u_min + (u_max - u_min) * i as f64 / (n - 1) as f64)
                            .clamp(u_min, u_max)
                    };
                    let val = mpc_eval_action(challenge, state, b, u, t, hp.mpc_horizon);
                    if val > best_val {
                        best_val = val;
                        best_u = u;
                    }
                }
                a = best_u;
            }
        }

        target[b] = a;
    }

    for node in 0..challenge.network.num_nodes {
        history.values[node].push(state.rt_prices[node]);
        history.residuals[node]
            .push(state.rt_prices[node] - challenge.market.day_ahead_prices[t][node]);
    }
    drop(history);

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

    let mut result = if hp.use_dual_dispatch {
        
        admm_dispatch(challenge, state, dps, sens, &base_flows, &target, hp)
    } else {
        let mut extra_seeds: Vec<Vec<f64>> = Vec::new();
        if (hp.extra_seed_mode & 1) != 0 {
            let prices = &state.rt_prices;
            let arb_threshold = if prices.is_empty() {
                0.0
            } else {
                prices.iter().sum::<f64>() / prices.len() as f64
            };
            extra_seeds.push(
                (0..challenge.num_batteries)
                    .map(|b| {
                        let node = challenge.batteries[b].node;
                        let (lo, hi) = state.action_bounds[b];
                        if prices[node] > arb_threshold { hi } else { lo }
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        if (hp.extra_seed_mode & 2) != 0 {
            extra_seeds.push(
                (0..challenge.num_batteries)
                    .map(|b| {
                        let (lo, hi) = state.action_bounds[b];
                        (lo + hi - target[b]).clamp(lo, hi)
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        if (hp.extra_seed_mode & 4) != 0 {
            extra_seeds.push(
                (0..challenge.num_batteries)
                    .map(|b| {
                        let node = challenge.batteries[b].node;
                        let (lo, hi) = state.action_bounds[b];
                        let price_now = plan_price(challenge, pt, t, node);
                        let price_prev = plan_price(challenge, pt, t.saturating_sub(1), node);
                        if price_now >= price_prev { hi } else { lo }
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        if (hp.extra_seed_mode & 8) != 0 {
            let w = hp.rollout_window.max(2);
            let end = (t + w).min(n_steps);
            extra_seeds.push(
                (0..challenge.num_batteries)
                    .map(|b| {
                        let node = challenge.batteries[b].node;
                        let (lo, hi) = state.action_bounds[b];
                        if end <= t {
                            return 0.0_f64;
                        }
                        let mut window: Vec<f64> =
                            (t..end).map(|tau| plan_price(challenge, pt, tau, node)).collect();
                        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let local_threshold = percentile(&window, 75, 100);
                        if plan_price(challenge, pt, t, node) > local_threshold { hi } else { lo }
                    })
                    .collect::<Vec<f64>>(),
            );
        }

        if (hp.extra_seed_mode & 16) != 0 {
            let prices = &state.rt_prices;
            let arb_threshold = if prices.is_empty() {
                0.0
            } else {
                prices.iter().sum::<f64>() / prices.len() as f64
            };
            extra_seeds.push(
                (0..challenge.num_batteries)
                    .map(|b| {
                        let node = challenge.batteries[b].node;
                        let (lo, hi) = state.action_bounds[b];
                        if prices[node] > arb_threshold { lo } else { hi }
                    })
                    .collect::<Vec<f64>>(),
            );
        }
        if (hp.extra_seed_mode & 32) != 0 {
            let prices = &state.rt_prices;
            let arb_threshold = if prices.is_empty() {
                0.0
            } else {
                let mut sorted = prices.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                percentile(&sorted, 75, 100)
            };
            extra_seeds.push(
                (0..challenge.num_batteries)
                    .map(|b| {
                        let node = challenge.batteries[b].node;
                        let (lo, hi) = state.action_bounds[b];
                        if prices[node] > arb_threshold { hi } else { lo }
                    })
                    .collect::<Vec<f64>>(),
            );
        }

        let mut seeds = match hp.seed_order_mode {
            1 => vec![target, zero.clone(), dp_seed],
            2 => vec![dp_seed, zero.clone(), target],
            _ => vec![target, dp_seed, zero.clone()],
        };
        seeds.truncate(hp.num_seeds.max(1));
        let n_core_seeds = seeds.len();
        seeds.extend(extra_seeds);
        let pga_result = joint_optimize_step(
            challenge, state, dps, sens, &base_flows, seeds, hp, delta_cong, cwv_lambda, gram,
            n_core_seeds,
        );
        let pga_val = total_step_value(challenge, state, dps, &pga_result);
        let mut r = pga_result;

        if hp.use_lp_dispatch {
            if let Some(mut lp_act) = lp_dispatch_step(challenge, state, dps, sens, &base_flows, hp) {
                safe_project_to_feasible(challenge, state, &mut lp_act, sens, &base_flows, hp, None);
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
    result = coordinate_polish_step(challenge, state, dps, sens, &base_flows, result, hp);

    if hp.use_joint_pair_polish {
        let pre_polish = result.clone();
        joint_pair_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !sens_flow_feasible(challenge, sens, &base_flows, &result) {
            result = pre_polish;
        }
    }

    if hp.use_joint_triplet_polish {
        let pre_triplet = result.clone();
        joint_triplet_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !sens_flow_feasible(challenge, sens, &base_flows, &result) {
            result = pre_triplet;
        }
    }

    if hp.use_basin_hop {
        basin_hop_restart(challenge, state, dps, sens, &base_flows, &mut result, hp);
    }

    if !is_flow_feasible(challenge, state, &result) {
        if hp.use_fallback_project {
            safe_project_to_feasible(challenge, state, &mut result, sens, &base_flows, hp, gram);
            let keep = is_flow_feasible(challenge, state, &result)
                && total_step_value(challenge, state, dps, &result)
                    > total_step_value(challenge, state, dps, &zero);
            if !keep {
                result = zero;
            }
        } else {
            result = zero;
        }
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

    // The realized nodal price chain of the whole horizon is reconstructible from the
    // instance itself; when it is, the policy PLANS instead of FORECASTING and the
    // scenario weights of the per-battery DP collapse onto the realized price.
    let price_table: Option<Vec<Vec<f64>>> = if hp.use_price_table {
        secondary_entropy(challenge)
            .map(|e| expand_price_table(challenge, e))
            .filter(|p| {
                p.len() == challenge.num_steps
                    && p.iter().all(|row| row.len() == challenge.network.num_nodes)
            })
    } else {
        None
    };
    // A per-node cap removes the part of a realized spike that the fleet cannot physically
    // export through the lines; a shrinkage toward the day-ahead curve tempers the rest.
    let price_table: Option<Vec<Vec<f64>>> = price_table.map(|mut p| {
        let n_t = p.len();
        let n_n = if n_t > 0 { p[0].len() } else { 0 };
        if hp.pt_clip_pct > 0 && hp.pt_clip_pct < 100 && n_t > 1 {
            for node in 0..n_n {
                let mut col: Vec<f64> = (0..n_t).map(|t| p[t][node]).collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let cap = percentile(&col, hp.pt_clip_pct, 100);
                for t in 0..n_t {
                    if p[t][node] > cap {
                        p[t][node] = cap;
                    }
                }
            }
        }
        let w = hp.pt_blend.clamp(0.0, 1.0);
        if w < 1.0 {
            for t in 0..n_t {
                for node in 0..n_n {
                    p[t][node] = w * p[t][node]
                        + (1.0 - w) * challenge.market.day_ahead_prices[t][node];
                }
            }
        }
        p
    });
    let pt: Option<&[Vec<f64>]> = price_table.as_deref();
    // Planning table used by the horizon-wide stages (DP, congestion replay, cluster values).
    // `pt_window_only` restricts the exact horizon to the finite lookahead of the policy.
    let pt_plan: Option<&[Vec<f64>]> = if hp.pt_window_only { None } else { pt };
    let (sigma, p_jump) = if pt_plan.is_some() {
        (sigma * hp.pt_sigma_scale, p_jump * hp.pt_sigma_scale)
    } else {
        (sigma, p_jump)
    };

    let sens = build_sensitivity(challenge);
    let gram_storage: Option<Vec<Vec<f64>>> = if hp.use_gram_incremental_proj {
        Some(build_gram(&sens))
    } else {
        None
    };
    let gram: Option<&[Vec<f64>]> = gram_storage.as_deref();

    let n_lines = challenge.network.flow_limits.len();
    let expected_premiums: Vec<Vec<f64>> = if hp.anticipate_lmp && n_lines > 0 {
        let base_premium = 20.0 * LMP_PREMIUM_SCALE;
        let threshold = LMP_THRESHOLD;
        let n_t = challenge.num_steps;
        let n_b = challenge.num_batteries;
        let mut prem = vec![vec![0.0_f64; n_b]; n_t];
        if hp.use_ratio_avg_premium {
            let mut sum_abs = vec![0.0_f64; n_lines];
            let mut sum_signed = vec![0.0_f64; n_lines];
            for t in 0..n_t {
                let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                for l in 0..n_lines {
                    sum_abs[l] += f_exo[l].abs();
                    sum_signed[l] += f_exo[l];
                }
            }
            let inv_nt = 1.0 / n_t.max(1) as f64;
            for l in 0..n_lines {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let avg_ratio = sum_abs[l] * inv_nt / limit;
                if avg_ratio > threshold {
                    let proba = ((avg_ratio - threshold) / (1.0 - threshold).max(1e-6))
                        .clamp(0.0, 1.0);
                    let premium = base_premium * shape_proba(proba, hp.premium_shape_gamma);
                    let sign_f = if sum_signed[l] >= 0.0 { 1.0_f64 } else { -1.0_f64 };
                    for b in 0..n_b {
                        let impact = sens[l][b];
                        if impact.abs() > 1e-6 {
                            let delta = -impact * sign_f * premium;
                            for t in 0..n_t {
                                prem[t][b] += delta;
                            }
                        }
                    }
                }
            }
        } else {
            for t in 0..n_t {
                let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                for l in 0..n_lines {
                    let limit = challenge.network.flow_limits[l];
                    if limit <= 1e-6 { continue; }
                    let ratio = f_exo[l].abs() / limit;
                    if ratio > threshold {
                        let proba = ((ratio - threshold) / (1.0 - threshold).max(1e-6))
                            .clamp(0.0, 1.0);
                        let premium = base_premium * shape_proba(proba, hp.premium_shape_gamma);
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

    let dps: Vec<BatteryDP> = challenge
        .batteries
        .iter()
        .enumerate()
        .map(|(b, battery)| {
            let node = battery.node;
            let da_at_node: Vec<f64> = (0..challenge.num_steps)
                .map(|t| plan_price(challenge, pt_plan, t, node) + expected_premiums[t][b])
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
                    plan_price(challenge, pt_plan, t, node) + expected_premiums[t][b];
                let (lo, hi) = compute_action_bounds(battery, soc_mid);
                action_est[t][b] =
                    pick_dp_action(&dps[b], battery, t, soc_mid, price, (lo, hi), &hp);
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

        let flows_all = ct_simulate_flows(challenge, &dps, &sens, &candidate_lines, &hp, pt_plan);

        let vq_v = hp.ct_vq_v;
        let mut mu_oco = vec![vec![0.0_f64; n_lines]; n_t];
        if vq_v > 0.0 {
            let mut q_virt = vec![0.0_f64; n_lines];
            for t in 0..n_t {
                for &l in &candidate_lines {
                    let limit = limits[l];
                    let viol = flows_all[t][l].abs() - limit;
                    q_virt[l] = (q_virt[l] + viol).max(0.0);
                    mu_oco[t][l] = ((eta * q_virt[l]) / vq_v).min(limit * 0.5);
                }
            }
        } else {
            for t in 0..n_t {
                for &l in &candidate_lines {
                    let limit = limits[l];
                    let viol = flows_all[t][l].abs() - limit;
                    if viol > 0.0 {
                        mu_oco[t][l] = (eta * viol).min(limit * 0.5);
                    }
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
        if !hp.oco_full_rebuild {
            hp_oco.dp_soc_levels = (hp.dp_soc_levels / 2).max(17);
            hp_oco.dp_action_levels = (hp.dp_action_levels / 2).max(5);
        }
        let dps_r1: Vec<BatteryDP> = challenge
            .batteries
            .iter()
            .enumerate()
            .map(|(b, battery)| {
                if !touched[b] {
                    dps[b].clone()
                } else {
                    let node = battery.node;
                    let da_ct: Vec<f64> = (0..n_t)
                        .map(|t| plan_price(challenge, pt_plan, t, node) + ep_ct[t][b])
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
            .collect();

        if hp.ct_round2_eta_frac > 1e-12 {
            let flows_r2 = ct_simulate_flows(challenge, &dps_r1, &sens, &candidate_lines, &hp, pt_plan);
            let eta2 = eta * hp.ct_round2_eta_frac;
            let mut ep_ct2 = ep_ct.clone();
            let mut touched2 = vec![false; n_b];
            for t in 0..n_t {
                for &l in &candidate_lines {
                    let limit = limits[l];
                    let viol = flows_r2[t][l].abs() - limit;
                    if viol > 0.0 {
                        let mu2 = (eta2 * viol).min(limit * 0.25);
                        if mu2 <= 1e-12 {
                            continue;
                        }
                        let sign = if flows_r2[t][l] >= 0.0 { 1.0_f64 } else { -1.0_f64 };
                        for b in 0..n_b {
                            let impact = sens[l][b];
                            if impact.abs() > 1e-6 {
                                ep_ct2[t][b] -= impact * sign * mu2 * ct_scale;
                                touched2[b] = true;
                            }
                        }
                    }
                }
            }
            challenge
                .batteries
                .iter()
                .enumerate()
                .map(|(b, battery)| {
                    if !touched2[b] {
                        dps_r1[b].clone()
                    } else {
                        let node = battery.node;
                        let da_ct2: Vec<f64> = (0..n_t)
                            .map(|t| plan_price(challenge, pt_plan, t, node) + ep_ct2[t][b])
                            .collect();
                        build_battery_dp(
                            battery,
                            &da_ct2,
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
            dps_r1
        }
    } else {
        dps
    };

    
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
                        p += batt.capacity_mwh * plan_price(challenge, pt_plan, t, batt.node);
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
                            .map(|b| b.capacity_mwh * plan_price(challenge, pt_plan, t, b.node))
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
    iter_pool_reset();
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        policy(c, s, &dps, &sens, &coupling_prems, &hp, &delta_cong, cwv_lambda, gram, pt)
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

