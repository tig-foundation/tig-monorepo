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
    #[serde(default = "default_use_price_table")]
    pub use_price_table: bool,
    #[serde(default = "default_pt_sigma_scale")]
    pub pt_sigma_scale: f64,
    #[serde(default)]
    pub pt_window_only: bool,
    #[serde(default = "default_pt_clip_pct")]
    pub pt_clip_pct: usize,
    #[serde(default = "default_pt_blend")]
    pub pt_blend: f64,
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
    #[serde(default)]
    pub blk_prem: f64,
    #[serde(default = "default_blk_px_mode")]
    pub blk_px_mode: usize,
    #[serde(default = "default_blk_duty")]
    pub blk_duty: usize,
    #[serde(default = "default_blk_curv_tol")]
    pub blk_curv_tol: f64,
    #[serde(default = "default_pre_dp_cheap")]
    pub pre_dp_cheap: bool,
    #[serde(default = "default_blk_hot_mem")]
    pub blk_hot_mem: usize,
    #[serde(default = "default_ct_premium")]
    pub ct_premium: usize,
    #[serde(default = "default_blk_soc_lazy")]
    pub blk_soc_lazy: usize,
    #[serde(default = "default_blk_w_tail")]
    pub blk_w_tail: usize,
    #[serde(default = "default_blk_head_pct")]
    pub blk_head_pct: usize,
    #[serde(default)]
    pub rt_prem_scale: f64,
    #[serde(default)]
    pub blk_polish: usize,
    #[serde(default = "default_dp_gh_points")]
    pub dp_gh_points: usize,
    #[serde(default)]
    pub dp_jump_bins: usize,
    #[serde(default = "default_dp_power_scale")]
    pub dp_power_scale: f64,
    #[serde(default)]
    pub dp_power_scale_dis: f64,
    #[serde(default)]
    pub dp_derate_mode: usize,
    #[serde(default)]
    pub dp_derate_scale: f64,
    #[serde(default)]
    pub dp_derate_floor: f64,
    #[serde(default = "default_step_seg_k")]
    pub step_seg_k: usize,
    #[serde(default = "default_step_dual")]
    pub step_dual: usize,
    #[serde(default = "default_pilot_paths")]
    pub pilot_paths: usize,
    #[serde(default = "default_pilot_gamma")]
    pub pilot_gamma: f64,
    #[serde(default = "default_pilot_soc_levels")]
    pub pilot_soc_levels: usize,
    #[serde(default = "default_pilot_gh")]
    pub pilot_gh: usize,
    #[serde(default = "default_pilot_prem_mode")]
    pub pilot_prem_mode: usize,
    #[serde(default = "default_pilot_prem_scale")]
    pub pilot_prem_scale: f64,
    #[serde(default = "default_pilot_prem_win")]
    pub pilot_prem_win: usize,
    #[serde(default)]
    pub pilot_no_derate: usize,
    #[serde(default)]
    pub pilot_iters: usize,
    #[serde(default)]
    pub pilot_damp: f64,
    #[serde(default)]
    pub pilot_cheap_mid: usize,
    #[serde(default)]
    pub pilot_mean_path: usize,
    #[serde(default = "default_pilot_seg_k")]
    pub pilot_seg_k: usize,
}

fn default_blk_soc_lazy() -> usize {
    1
}

fn default_blk_w_tail() -> usize {
    3
}

fn default_blk_head_pct() -> usize {
    100
}

fn default_ct_premium() -> usize {
    1
}

fn default_blk_hot_mem() -> usize {
    2
}

fn default_pre_dp_cheap() -> bool {
    true
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

fn default_blk_px_mode() -> usize {
    2
}

fn default_blk_duty() -> usize {
    1
}

fn default_blk_curv_tol() -> f64 {
    20.0
}

fn default_dp_gh_points() -> usize {
    7
}
fn default_dp_power_scale() -> f64 {
    0.45
}
fn default_step_seg_k() -> usize {
    4
}
fn default_step_dual() -> usize {
    1
}
fn default_pilot_paths() -> usize {
    2
}
fn default_pilot_gamma() -> f64 {
    0.5
}
fn default_pilot_soc_levels() -> usize {
    33
}
fn default_pilot_gh() -> usize {
    5
}
fn default_pilot_prem_mode() -> usize {
    2
}
fn default_pilot_prem_scale() -> f64 {
    0.7
}
fn default_pilot_prem_win() -> usize {
    2
}
fn default_pilot_seg_k() -> usize {
    1
}
fn default_use_block_lp() -> bool {
    true
}

fn default_blk_w() -> usize {
    1
}

fn default_blk_pivot_budget() -> usize {
    20000
}

fn default_blk_flow_margin() -> f64 {
    1e-4
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
            use_block_lp: true,
            blk_w: 1,
            blk_pivot_budget: 20000,
            blk_flow_margin: 1e-4,
            blk_term_k: 4,
            blk_lazy_rounds: 12,
            blk_row_cap: 900,
            blk_add_cap: 600,
            blk_drop_tol: 1e-3,
            blk_trust: 1.0,
            blk_prem: 0.0,
            blk_px_mode: 2,
            blk_duty: 1,
            blk_curv_tol: 20.0,
            pre_dp_cheap: true,
            blk_hot_mem: 2,
            ct_premium: 1,
            blk_soc_lazy: 1,
            blk_w_tail: 3,
            blk_head_pct: 100,
            rt_prem_scale: 0.0,
            blk_polish: 0,
            dp_gh_points: 7,
            dp_jump_bins: 0,
            dp_power_scale: 0.45,
            dp_power_scale_dis: 0.0,
            dp_derate_mode: 0,
            dp_derate_scale: 0.0,
            dp_derate_floor: 0.0,
            step_seg_k: 4,
            step_dual: 1,
            pilot_paths: 2,
            pilot_gamma: 0.5,
            pilot_soc_levels: 33,
            pilot_gh: 5,
            pilot_prem_mode: 2,
            pilot_prem_scale: 0.7,
            pilot_prem_win: 2,
            pilot_no_derate: 0,
            pilot_iters: 1,
            pilot_damp: 0.0,
            pilot_cheap_mid: 0,
            pilot_mean_path: 0,
            pilot_seg_k: 1,
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
        hp.blk_w = hp.blk_w.clamp(1, 24);
        hp.blk_pivot_budget = hp.blk_pivot_budget.max(200);
        hp.blk_flow_margin = hp.blk_flow_margin.max(0.0);
        hp.blk_term_k = hp.blk_term_k.min(64);
        hp.blk_lazy_rounds = hp.blk_lazy_rounds.clamp(1, 32);
        hp.blk_row_cap = hp.blk_row_cap.clamp(0, 20000);
        hp.blk_add_cap = hp.blk_add_cap.clamp(1, 20000);
        hp.blk_drop_tol = hp.blk_drop_tol.max(0.0);
        hp.blk_trust = hp.blk_trust.clamp(0.0, 1.0);
        hp.blk_duty = hp.blk_duty.clamp(1, 64);
        hp.blk_curv_tol = hp.blk_curv_tol.max(0.0);
        hp.blk_soc_lazy = hp.blk_soc_lazy.min(1);
        hp.blk_w_tail = hp.blk_w_tail.min(24);
        hp.blk_head_pct = hp.blk_head_pct.min(100);
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

#[inline]
fn flows_within_limits(flows: &[f64], limits: &[f64]) -> bool {
    for l in 0..flows.len() {
        if flows[l].abs() - limits[l] > EPS_FLOW * limits[l] {
            return false;
        }
    }
    true
}

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
    let low = pos as usize;
    let high = (low + 1).min(last);
    let alpha = pos - low as f64;
    values[low] * (1.0 - alpha) + values[high] * alpha
}

#[inline(always)]
fn interp_value_q(values: &[f64], soc: f64, lo: f64, step_inv: f64, last: usize, curv_coef: f64) -> f64 {
    let pos = ((soc - lo) * step_inv).clamp(0.0, last as f64);
    let low = pos as usize;
    let high = (low + 1).min(last);
    let alpha = pos - low as f64;
    let linear = values[low] * (1.0 - alpha) + values[high] * alpha;
    let h2 = (high + 1).min(last);
    let d2v = values[h2] - 2.0 * values[high] + values[low];
    let on = if high < last { 1.0 } else { 0.0 };
    linear + on * (alpha * (alpha - 1.0) * curv_coef * d2v)
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
    der: Option<&[[f64; 2]]>,
    da_at_node: &[f64],
    num_steps: usize,
    sigma: f64,
    p_jump: f64,
    mean_pareto: f64,
    second_pareto: f64,
    fleet_soc_norm: f64,
    hp: &Hyperparameters,
) -> BatteryDP {
    if hp.dp_power_scale > 0.0 && hp.dp_power_scale < 1.0 {
        let mut bd = battery.clone();
        bd.power_charge_mw *= hp.dp_power_scale;
        bd.power_discharge_mw *= if hp.dp_power_scale_dis > 0.0 { hp.dp_power_scale_dis.min(1.0) } else { hp.dp_power_scale };
        let mut h2 = *hp;
        h2.dp_power_scale = 0.0;
        return build_battery_dp(
            &bd, der, da_at_node, num_steps, sigma, p_jump, mean_pareto, second_pareto,
            fleet_soc_norm, &h2,
        );
    }
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

    let gh: Vec<(f64, f64)> = match hp.dp_gh_points {
        3 => vec![(-1.7320508075688772, 1.0 / 6.0), (0.0, 2.0 / 3.0), (1.7320508075688772, 1.0 / 6.0)],
        5 => vec![
            (-2.8569700138728056, 0.011257411327720691),
            (-1.3556261799742657, 0.22207592200561266),
            (0.0, 0.5333333333333333),
            (1.3556261799742657, 0.22207592200561266),
            (2.8569700138728056, 0.011257411327720691),
        ],
        7 => vec![
            (-3.7504397177257425, 0.0005482688559722184),
            (-2.366759410734541, 0.030757123967586496),
            (-1.1544053947399682, 0.2401231786050127),
            (0.0, 0.45714285714285713),
            (1.1544053947399682, 0.2401231786050127),
            (2.366759410734541, 0.030757123967586496),
            (3.7504397177257425, 0.0005482688559722184),
        ],
        _ => Vec::new(),
    };
    let mut scen: Vec<(f64, f64)> = Vec::new();
    if !gh.is_empty() {
        for &(z, w) in gh.iter() {
            scen.push((1.0 + sigma * z, w_normal * w));
        }
        let nb = hp.dp_jump_bins;
        let a = if mean_pareto > 1.0 { mean_pareto / (mean_pareto - 1.0) } else { 0.0 };
        if nb >= 2 && a > 1.0 {
            let mut edges: Vec<f64> = vec![0.0];
            for k in 1..nb {
                edges.push(1.0 - 0.5_f64.powi(k as i32));
            }
            edges.push(1.0);
            let e = 1.0 - 1.0 / a;
            for k in 0..nb {
                let (q1, q2) = (edges[k], edges[k + 1]);
                if q2 <= q1 {
                    continue;
                }
                let m = ((1.0 - q1).powf(e) - (1.0 - q2).powf(e)) / (e * (q2 - q1));
                scen.push((1.0 + m.min(400.0), w_jump * (q2 - q1)));
            }
        } else {
            scen.push((1.0 + jump_floor, w_jump_low));
            scen.push((1.0 + jump_ceiling, w_jump_high));
        }
    }
    let mut gh_best: Vec<f64> = vec![0.0; scen.len()];
    let mut gh_price: Vec<f64> = vec![0.0; scen.len()];

    for t in (0..num_steps).rev() {
        let da = da_at_node[t];
        if !gh.is_empty() {
            for (k, &(mlt, _)) in scen.iter().enumerate() {
                gh_price[k] = da * mlt;
            }
        }
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
            for v in gh_best.iter_mut() {
                *v = f64::NEG_INFINITY;
            }

            for &raw0 in &actions {
                let raw = match der {
                    Some(d) => {
                        let dd = d[t.min(d.len() - 1)];
                        if raw0 < 0.0 { raw0 * dd[0] } else { raw0 * dd[1] }
                    }
                    None => raw0,
                };
                let action = raw.clamp(lo, hi);
                let future = {
                    let next_soc = battery.apply_action_to_soc(action, soc);
                    if hp.use_dp_value_shift {
                        interp_value_q(next, next_soc, soc_lo, soc_step_inv, last, hp.dp_value_curv_coef)
                    } else {
                        interp_value(next, next_soc, soc_lo, soc_step_inv, last)
                    }
                };

                let throughput = action.abs() * DELTA_T;
                let tx = KAPPA_TX * throughput;
                let deg = KAPPA_DEG * (throughput / battery.capacity_mwh).powi(2);

                if gh_best.is_empty() {
                best_low = best_low.max(action * price_low * DELTA_T - tx - deg + future);
                best_high = best_high.max(action * price_high * DELTA_T - tx - deg + future);
                best_jump_low =
                    best_jump_low.max(action * price_jump_low * DELTA_T - tx - deg + future);
                best_jump_high =
                    best_jump_high.max(action * price_jump_high * DELTA_T - tx - deg + future);
                }
                for k in 0..gh_best.len() {
                    let v = action * gh_price[k] * DELTA_T - tx - deg + future;
                    let b0 = gh_best[k];
                    gh_best[k] = if v > b0 { v } else { b0 };
                }
            }
            current[s_idx] = if gh.is_empty() {
                w_low * best_low
                    + w_high * best_high
                    + w_jump_low * best_jump_low
                    + w_jump_high * best_jump_high
            } else {
                let mut acc = 0.0_f64;
                for k in 0..scen.len() {
                    acc += scen[k].1 * gh_best[k];
                }
                acc
            };
            
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
    let mut low = pos as usize;
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

fn instance_hash(challenge: &Challenge) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for row in challenge.exogenous_injections.iter() {
        for v in row.iter() {
            h ^= v.to_bits();
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
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

    let nonce_u64 = instance_hash(challenge);
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
    pt_raw: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_steps = challenge.num_steps;
    let n_remaining = n_steps.saturating_sub(t);
    if n_remaining == 0 {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    let step_cand = if hp.use_block_lp && hp.step_seg_k > 0 {
        solve_step_lp(challenge, state, dps, sens, hp, None)
    } else {
        None
    };
    if hp.use_block_lp {
        if let Some(cand) = step_cand.or_else(|| if hp.step_seg_k > 0 { None } else { block_plan_action(
            challenge, state, dps, sens, coupling_prems, hp, pt, pt_raw, t, n_steps,
        ) }) {
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
            if hp.blk_polish == 0 {
                return Ok(cand);
            }
            let zero = vec![0.0_f64; challenge.num_batteries];
            let base_flows = compute_flows(challenge, state, &zero);
            let v0 = total_step_value(challenge, state, dps, &cand);
            let mut r = cand.clone();
            if (hp.blk_polish & 1) != 0 {
                let mut h = *hp;
                h.coord_polish_passes = h.coord_polish_passes.max(1);
                r = coordinate_polish_step(challenge, state, dps, sens, &base_flows, r, &h);
            }
            if (hp.blk_polish & 2) != 0 {
                let pre = r.clone();
                joint_pair_polish(challenge, state, dps, sens, &base_flows, &mut r, hp);
                if !sens_flow_feasible(challenge, sens, &base_flows, &r) {
                    r = pre;
                }
            }
            if (hp.blk_polish & 4) != 0 {
                basin_hop_restart(challenge, state, dps, sens, &base_flows, &mut r, hp);
            }
            if is_flow_feasible(challenge, state, &r)
                && total_step_value(challenge, state, dps, &r) > v0
            {
                return Ok(r);
            }
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

fn causal_price_table(challenge: &Challenge, scale: f64) -> Vec<Vec<f64>> {
    let n_t = challenge.num_steps;
    let net = &challenge.network;
    let n_n = net.num_nodes;
    let mean_half_normal = 0.398_942_280_401_432_7_f64;
    let mut table: Vec<Vec<f64>> = challenge.market.day_ahead_prices.clone();
    for t in 1..n_t {
        let flows = net.compute_flows(&challenge.exogenous_injections[t - 1]);
        let mut p_free = vec![1.0_f64; n_n];
        for (l, &f) in flows.iter().enumerate() {
            let lim = net.congestion_threshold * net.flow_limits[l];
            if lim <= 0.0 {
                continue;
            }
            let p = (f.abs() / lim).powf(10.0).min(1.0);
            if p <= 0.0 {
                continue;
            }
            let (a, b) = net.lines[l];
            p_free[a] *= 1.0 - p;
            p_free[b] *= 1.0 - p;
        }
        for i in 0..n_n {
            table[t][i] += scale * 20.0 * mean_half_normal * (1.0 - p_free[i]);
        }
    }
    table
}

fn build_dp_derate(challenge: &Challenge, sens: &[Vec<f64>], hp: &Hyperparameters) -> Vec<Vec<[f64; 2]>> {
    let n_b = challenge.num_batteries;
    let n_t = challenge.num_steps;
    let n_l = sens.len();
    let limits = &challenge.network.flow_limits;
    let scale = if hp.dp_derate_scale > 0.0 { hp.dp_derate_scale } else { 1.0 };
    let floor = hp.dp_derate_floor.clamp(0.0, 1.0);
    let mut reach = vec![0.0_f64; n_l];
    for l in 0..n_l {
        for b in 0..n_b {
            let bat = &challenge.batteries[b];
            reach[l] += sens[l][b].abs() * bat.power_charge_mw.max(bat.power_discharge_mw);
        }
    }
    let mut out = vec![vec![[1.0_f64; 2]; n_t]; n_b];
    if hp.dp_derate_mode == 4 || hp.dp_derate_mode == 5 {
        let mut agg = vec![0.0_f64; n_l];
        for l in 0..n_l {
            for b in 0..n_b {
                agg[l] += sens[l][b] * challenge.batteries[b].power_discharge_mw;
            }
        }
        let mut al_d_all = vec![0.0_f64; n_t];
        let mut al_c_all = vec![0.0_f64; n_t];
        for t in 0..n_t {
            let f = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut ad = f64::INFINITY;
            let mut ac = f64::INFINITY;
            for l in 0..n_l {
                let lim = limits[l];
                let g = agg[l];
                if lim <= 1e-6 || g.abs() <= 1e-9 {
                    continue;
                }
                let up = (lim - f[l]).max(0.0);
                let dn = (lim + f[l]).max(0.0);
                if g > 0.0 {
                    ad = ad.min(up / g);
                    ac = ac.min(dn / g);
                } else {
                    ad = ad.min(dn / -g);
                    ac = ac.min(up / -g);
                }
            }
            al_d_all[t] = ad;
            al_c_all[t] = ac;
        }
        if hp.dp_derate_scale < 0.0 {
            let gamma = -hp.dp_derate_scale;
            let med = |v: &Vec<f64>| -> f64 {
                let mut w: Vec<f64> = v.iter().map(|x| x.min(1e6)).collect();
                w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                w[w.len() / 2].max(1e-9)
            };
            let (md, mc) = (med(&al_d_all), med(&al_c_all));
            let level = if floor > 0.0 { floor } else { 0.55 };
            for t in 0..n_t {
                let dd = (level * (al_d_all[t].min(1e6) / md).powf(gamma)).clamp(0.15, 1.0);
                let dc = (level * (al_c_all[t].min(1e6) / mc).powf(gamma)).clamp(0.15, 1.0);
                for b in 0..n_b {
                    out[b][t] = [dc, dd];
                }
            }
            return out;
        }
        for t in 0..n_t {
            let (mut ad, mut ac) = (al_d_all[t], al_c_all[t]);
            if hp.dp_derate_mode == 5 {
                let m = 0.5 * (ad.min(1.0) + ac.min(1.0));
                ad = m;
                ac = m;
            }
            let dd = (ad * scale).clamp(floor, 1.0);
            let dc = (ac * scale).clamp(floor, 1.0);
            for b in 0..n_b {
                out[b][t] = [dc, dd];
            }
        }
        return out;
    }
    let mut room_pos = vec![1.0_f64; n_l];
    let mut room_neg = vec![1.0_f64; n_l];
    for t in 0..n_t {
        let f = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
        for l in 0..n_l {
            let lim = limits[l];
            if lim <= 1e-6 || reach[l] <= 1e-9 {
                room_pos[l] = 1.0;
                room_neg[l] = 1.0;
                continue;
            }
            if hp.dp_derate_mode >= 2 {
                room_pos[l] = ((lim - f[l]).max(0.0) / reach[l]).min(1.0);
                room_neg[l] = ((lim + f[l]).max(0.0) / reach[l]).min(1.0);
            } else {
                let r = ((lim - f[l].abs()).max(0.0) / reach[l]).min(1.0);
                room_pos[l] = r;
                room_neg[l] = r;
            }
        }
        for b in 0..n_b {
            let mut wsum = 0.0_f64;
            let mut acc_d = 0.0_f64;
            let mut acc_c = 0.0_f64;
            for l in 0..n_l {
                let w = sens[l][b].abs();
                if w <= 1e-6 || limits[l] <= 1e-6 {
                    continue;
                }
                wsum += w;
                if sens[l][b] >= 0.0 {
                    acc_d += w * room_pos[l];
                    acc_c += w * room_neg[l];
                } else {
                    acc_d += w * room_neg[l];
                    acc_c += w * room_pos[l];
                }
            }
            let (dc, dd) = if wsum > 1e-12 { (acc_c / wsum, acc_d / wsum) } else { (1.0, 1.0) };
            out[b][t] = [(dc * scale).clamp(floor, 1.0), (dd * scale).clamp(floor, 1.0)];
        }
    }
    if hp.dp_derate_mode >= 3 {
        for b in 0..n_b {
            let mut m = [0.0_f64; 2];
            for t in 0..n_t {
                m[0] += out[b][t][0];
                m[1] += out[b][t][1];
            }
            let k = n_t.max(1) as f64;
            let avg = [m[0] / k, m[1] / k];
            for t in 0..n_t {
                out[b][t] = avg;
            }
        }
    }
    out
}

fn solve_step_lp(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    prem_out: Option<&mut Vec<f64>>,
) -> Option<Vec<f64>> {
    let num_b = challenge.num_batteries;
    let t = state.time_step;
    let n_lines = sens.len();
    let kk = hp.step_seg_k.max(1);
    let mut col_b: Vec<usize> = Vec::new();
    let mut col_sign: Vec<f64> = Vec::new();
    let mut c_obj: Vec<f64> = Vec::new();
    let mut hi: Vec<f64> = Vec::new();
    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let soc0 = state.socs[b];
        let price = state.rt_prices[battery.node];
        let (u_min, u_max) = state.action_bounds[b];
        let c_hi = (-u_min).max(0.0);
        let d_hi = u_max.max(0.0);
        let e_lo = (soc0 - (DELTA_T / ETA_DISCHARGE) * d_hi).max(battery.soc_min_mwh);
        let e_hi = (soc0 + ETA_CHARGE * DELTA_T * c_hi).min(battery.soc_max_mwh);
        let tv = build_term_value(&dps[b], t + 1, soc0, e_lo, e_hi, kk);
        let mut used_c = 0.0f64;
        for k in 0..kk {
            if tv.up_w[k] <= 0.0 {
                continue;
            }
            let wdt = (tv.up_w[k] / (ETA_CHARGE * DELTA_T)).min(c_hi - used_c);
            if wdt <= 1e-12 {
                continue;
            }
            used_c += wdt;
            col_b.push(b);
            col_sign.push(-1.0);
            hi.push(wdt);
            c_obj.push(-(price + KAPPA_TX) * DELTA_T + tv.up_s[k].max(0.0) * ETA_CHARGE * DELTA_T);
        }
        let mut used_d = 0.0f64;
        for k in 0..kk {
            if tv.dn_w[k] <= 0.0 {
                continue;
            }
            let wdt = (tv.dn_w[k] * ETA_DISCHARGE / DELTA_T).min(d_hi - used_d);
            if wdt <= 1e-12 {
                continue;
            }
            used_d += wdt;
            col_b.push(b);
            col_sign.push(1.0);
            hi.push(wdt);
            c_obj.push((price - KAPPA_TX) * DELTA_T - tv.dn_s[k].max(0.0) * (DELTA_T / ETA_DISCHARGE));
        }
    }
    let n_vars = c_obj.len();
    let lo = vec![0.0f64; n_vars];
    let zero = vec![0.0f64; num_b];
    let base = compute_flows(challenge, state, &zero);
    let margin = hp.blk_flow_margin;
    let limits = &challenge.network.flow_limits;
    let bound_of = |l: usize, up: bool| -> f64 {
        if up {
            (limits[l] - base[l] - margin).max(0.0)
        } else {
            (limits[l] + base[l] - margin).max(0.0)
        }
    };
    let epoch: u32 = {
        let mut e = hot_epoch().lock().unwrap();
        *e = e.saturating_add(1);
        *e
    };
    let mut in_active = vec![false; 2 * n_lines];
    let mut active: Vec<(usize, bool)> = Vec::new();
    {
        let hot = hot_lines().lock().unwrap();
        let mem = hp.blk_hot_mem;
        let mut cand: Vec<(u32, usize, bool)> = Vec::new();
        for l in 0..n_lines {
            if limits[l] <= 1e-6 {
                continue;
            }
            for &up in &[true, false] {
                let last = hot.get(l * 2 + up as usize).copied().unwrap_or(0);
                if last == 0 || (mem > 0 && epoch.saturating_sub(last) as usize >= mem) {
                    continue;
                }
                cand.push((last, l, up));
            }
        }
        cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for &(_, l, up) in cand.iter() {
            if active.len() >= hp.blk_row_cap {
                break;
            }
            in_active[l * 2 + up as usize] = true;
            active.push((l, up));
        }
    }
    if hp.step_dual != 0 {
        if let Some(cand) = step_lp_warm(challenge, state, sens, hp, &c_obj, &hi, &col_b, &col_sign,
            &active, &bound_of, epoch, prem_out)
        {
            return Some(cand);
        }
    }
    let mut rows: Vec<f64> = Vec::new();
    let mut grows: Vec<f64> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut tab: Vec<f64> = Vec::new();
    let mut duals: Vec<f64> = Vec::new();
    let mut u = vec![0.0f64; num_b];
    for _round in 0..hp.blk_lazy_rounds.max(1) {
        let m = active.len();
        rhs.clear();
        for &(l, up) in active.iter() {
            rhs.push(bound_of(l, up));
        }
        let dense_rows = |rows: &mut Vec<f64>| {
            rows.clear();
            rows.resize(m * n_vars, 0.0);
            for (i, &(l, up)) in active.iter().enumerate() {
                let sgn = if up { 1.0 } else { -1.0 };
                let row = &mut rows[i * n_vars..(i + 1) * n_vars];
                for j in 0..n_vars {
                    row[j] = sgn * col_sign[j] * sens[l][col_b[j]];
                }
            }
        };
        let _ = &grows;
        dense_rows(&mut rows);
        let sol = block_simplex::solve_max(
            n_vars, m, &c_obj, &lo, &hi, &rows, &rhs, hp.blk_pivot_budget, &mut tab, &mut duals,
        )?;
        for b in 0..num_b {
            u[b] = 0.0;
        }
        for j in 0..n_vars {
            u[col_b[j]] += col_sign[j] * sol[j];
        }
        let mut viol: Vec<(f64, usize, bool)> = Vec::new();
        let mut tight: Vec<(usize, bool)> = Vec::new();
        for l in 0..n_lines {
            if limits[l] <= 1e-6 {
                continue;
            }
            let mut f = 0.0f64;
            for b in 0..num_b {
                f += sens[l][b] * u[b];
            }
            for &up in &[true, false] {
                let lhs = if up { f } else { -f };
                let bnd = bound_of(l, up);
                let k = l * 2 + up as usize;
                if lhs > bnd + 1e-7 {
                    if !in_active[k] {
                        viol.push((lhs - bnd, l, up));
                    }
                    tight.push((l, up));
                } else if in_active[k] && lhs > bnd - hp.blk_drop_tol {
                    tight.push((l, up));
                }
            }
        }
        if viol.is_empty() {
            let mut hot = hot_lines().lock().unwrap();
            if hot.len() < 2 * n_lines {
                hot.resize(2 * n_lines, 0);
            }
            for &(l, up) in tight.iter() {
                hot[l * 2 + up as usize] = epoch;
            }
            drop(hot);
            let mut cand = u.clone();
            clamp_to_bounds(&mut cand, &state.action_bounds);
            if is_flow_feasible(challenge, state, &cand) {
                return Some(cand);
            }
            return None;
        }
        for &(l, up) in active.iter() {
            in_active[l * 2 + up as usize] = false;
        }
        active.clear();
        for &(l, up) in tight.iter() {
            let k = l * 2 + up as usize;
            if !in_active[k] && active.len() < hp.blk_row_cap {
                in_active[k] = true;
                active.push((l, up));
            }
        }
        viol.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        let mut added = 0usize;
        for &(_, l, up) in viol.iter() {
            if added >= hp.blk_add_cap.max(1) || active.len() >= hp.blk_row_cap {
                break;
            }
            let k = l * 2 + up as usize;
            if !in_active[k] {
                in_active[k] = true;
                active.push((l, up));
                added += 1;
            }
        }
    }
    None
}

fn pilot_unif(rs: &mut u64) -> f64 {
    ((splitmix64(rs) >> 11) as f64) * (1.0 / 9007199254740992.0)
}

fn pilot_normal(rs: &mut u64) -> f64 {
    let u1 = pilot_unif(rs).max(1e-300);
    let u2 = pilot_unif(rs);
    (-2.0 * u1.ln()).sqrt() * (6.283185307179586 * u2).cos()
}

fn pilot_derate(
    challenge: &Challenge,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    prem_out: &mut Vec<Vec<f64>>,
) -> Vec<Vec<[f64; 2]>> {
    let n_b = challenge.num_batteries;
    let n_t = challenge.num_steps;
    let net = &challenge.network;
    let n_n = net.num_nodes;
    let n_l = net.flow_limits.len();
    let mkt = &challenge.market;
    let sigma = mkt.params.volatility;
    let pj = mkt.params.jump_probability;
    let alpha = mkt.params.tail_index;
    let rho: f64 = 0.70;
    let mut p_line = vec![vec![0.0f64; n_l]; n_t];
    for t in 1..n_t {
        let f = net.compute_flows(&challenge.exogenous_injections[t - 1]);
        for l in 0..n_l {
            let lim = net.congestion_threshold * net.flow_limits[l];
            if lim > 0.0 {
                p_line[t][l] = (f[l].abs() / lim).powf(10.0).min(1.0);
            }
        }
    }
    let mut rs: u64 = instance_hash(challenge) ^ 0x7069_6c6f_745f_7061;
    let mut des = vec![[0.0f64; 2]; n_b];
    let mut exe = vec![[0.0f64; 2]; n_b];
    let mut flag = vec![false; n_n];
    let mut prices = vec![0.0f64; n_n];
    let mut prem_step: Vec<f64> = Vec::new();
    let mut hp_lp = *hp;
    if hp.pilot_seg_k > 0 {
        hp_lp.step_seg_k = hp.pilot_seg_k;
    }
    let mut prem_acc = vec![vec![0.0f64; n_b]; n_t];
    for _path in 0..hp.pilot_paths {
        let mut socs: Vec<f64> = challenge.batteries.iter().map(|b| b.soc_initial_mwh).collect();
        for t in 0..n_t {
            for i in 0..n_n {
                flag[i] = false;
            }
            for l in 0..n_l {
                if p_line[t][l] > 0.0 && p_line[t][l] > pilot_unif(&mut rs) {
                    let (a, b) = net.lines[l];
                    flag[a] = true;
                    flag[b] = true;
                }
            }
            let zc = pilot_normal(&mut rs);
            let zeta = pilot_normal(&mut rs).max(0.0);
            if hp.pilot_mean_path != 0 && _path == 0 {
                let mp = if alpha > 1.0 { alpha / (alpha - 1.0) } else { 1.0 };
                for i in 0..n_n {
                    let da = mkt.day_ahead_prices[t][i];
                        prices[i] = da * (1.0 + pj * mp);
                }
                for l in 0..n_l {
                    let (a, b) = net.lines[l];
                    let pl = p_line[t][l];
                    prices[a] += 20.0 * 0.3989422804014327 * pl;
                    prices[b] += 20.0 * 0.3989422804014327 * pl;
                }
            } else {
            for i in 0..n_n {
                let da = mkt.day_ahead_prices[t][i];
                let e = pilot_normal(&mut rs);
                let xi = rho.sqrt() * zc + (1.0 - rho).sqrt() * e;
                let mut p = da * (1.0 + sigma * xi);
                if flag[i] {
                    p += 20.0 * zeta;
                }
                if pilot_unif(&mut rs) < pj {
                    let u = pilot_unif(&mut rs).max(1e-10);
                    p += da * (1.0 - u).powf(-1.0 / alpha);
                }
                prices[i] = p.clamp(-200.0, 5000.0);
            }
            }
            let bounds: Vec<(f64, f64)> = challenge
                .batteries
                .iter()
                .zip(socs.iter())
                .map(|(b, &s)| compute_action_bounds(b, s))
                .collect();
            let state = State {
                time_step: t,
                socs: socs.clone(),
                rt_prices: prices.clone(),
                exogenous_injections: challenge.exogenous_injections[t].clone(),
                action_bounds: bounds.clone(),
                total_profit: 0.0,
            };
            let u = solve_step_lp(challenge, &state, dps, sens, &hp_lp, Some(&mut prem_step))
                .unwrap_or_else(|| vec![0.0f64; n_b]);
            if prem_step.len() == n_b {
                for b in 0..n_b {
                    prem_acc[t][b] += prem_step[b];
                }
            }
            prem_step.clear();
            for b in 0..n_b {
                let bat = &challenge.batteries[b];
                let want = pick_dp_action(
                    &dps[b], bat, t, socs[b], prices[bat.node], bounds[b], hp,
                );
                if want > 1e-9 {
                    des[b][1] += want;
                    exe[b][1] += u[b].max(0.0).min(want);
                } else if want < -1e-9 {
                    des[b][0] += -want;
                    exe[b][0] += (-u[b]).max(0.0).min(-want);
                }
                socs[b] = bat.apply_action_to_soc(u[b], socs[b]);
            }
        }
    }
    let mut r = vec![[1.0f64; 2]; n_b];
    let mut mean = [0.0f64; 2];
    let mut cnt = [0usize; 2];
    for b in 0..n_b {
        for k in 0..2 {
            if des[b][k] > 1e-9 {
                r[b][k] = exe[b][k] / des[b][k];
                mean[k] += r[b][k];
                cnt[k] += 1;
            }
        }
    }
    for k in 0..2 {
        mean[k] = if cnt[k] > 0 { mean[k] / cnt[k] as f64 } else { 1.0 };
    }
    {
        let w = hp.pilot_prem_win.max(1) as i64;
        let paths = hp.pilot_paths.max(1) as f64;
        prem_out.clear();
        prem_out.resize(n_t, vec![0.0f64; n_b]);
        for t in 0..n_t as i64 {
            let lo = (t - w).max(0) as usize;
            let hi = ((t + w) as usize).min(n_t - 1);
            let k = (hi - lo + 1) as f64 * paths;
            for b in 0..n_b {
                let mut a = 0.0f64;
                for tt in lo..=hi {
                    a += prem_acc[tt][b];
                }
                prem_out[t as usize][b] = a / k;
            }
        }
    }
    let level = if hp.dp_power_scale > 0.0 && hp.dp_power_scale < 1.0 { hp.dp_power_scale } else { 1.0 };
    let g = hp.pilot_gamma;
    let mut out = vec![vec![[1.0f64; 2]; n_t]; n_b];
    for b in 0..n_b {
        let mut d = [1.0f64; 2];
        for k in 0..2 {
            d[k] = if g < 0.0 {
                (r[b][k] * (-g)).clamp(0.15, 1.0)
            } else {
                (level * (r[b][k] / mean[k].max(1e-9)).powf(g)).clamp(0.15, 1.0)
            };
        }
        for t in 0..n_t {
            out[b][t] = d;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn step_lp_warm(
    challenge: &Challenge,
    state: &State,
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    c_obj: &[f64],
    hi: &[f64],
    col_b: &[usize],
    col_sign: &[f64],
    seed: &[(usize, bool)],
    bound_of: &dyn Fn(usize, bool) -> f64,
    epoch: u32,
    prem_out: Option<&mut Vec<f64>>,
) -> Option<Vec<f64>> {
    let num_b = challenge.num_batteries;
    let n_lines = sens.len();
    let limits = &challenge.network.flow_limits;
    let mut solver = dual_simplex::Solver::new(c_obj, hi, col_b, col_sign, num_b);
    let mut in_model = vec![false; 2 * n_lines];
    let mut model: Vec<(usize, bool)> = Vec::new();
    let mut grow: Vec<f64> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let push_rows = |list: &[(usize, bool)], grow: &mut Vec<f64>, rhs: &mut Vec<f64>| {
        grow.clear();
        rhs.clear();
        for &(l, up) in list.iter() {
            let sg = if up { 1.0 } else { -1.0 };
            for b in 0..num_b {
                grow.push(sg * sens[l][b]);
            }
            rhs.push(bound_of(l, up));
        }
    };
    push_rows(seed, &mut grow, &mut rhs);
    solver.add_rows(&grow, &rhs);
    for &(l, up) in seed.iter() {
        in_model[l * 2 + up as usize] = true;
        model.push((l, up));
    }
    let mut u = vec![0.0f64; num_b];
    let mut viol: Vec<(f64, usize, bool)> = Vec::new();
    let mut add: Vec<(usize, bool)> = Vec::new();
    for _round in 0..hp.blk_lazy_rounds.max(1) {
        if !solver.solve(hp.blk_pivot_budget) {
            return None;
        }
        let sol = solver.solution();
        for b in 0..num_b {
            u[b] = 0.0;
        }
        for j in 0..sol.len() {
            u[col_b[j]] += col_sign[j] * sol[j];
        }
        viol.clear();
        let mut fl = vec![0.0f64; n_lines];
        for l in 0..n_lines {
            if limits[l] <= 1e-6 {
                continue;
            }
            let mut f = 0.0f64;
            for b in 0..num_b {
                f += sens[l][b] * u[b];
            }
            fl[l] = f;
            for &up in &[true, false] {
                let lhs = if up { f } else { -f };
                let bnd = bound_of(l, up);
                if lhs > bnd + 1e-7 && !in_model[l * 2 + up as usize] {
                    viol.push((lhs - bnd, l, up));
                }
            }
        }
        if viol.is_empty() {
            let mut hot = hot_lines().lock().unwrap();
            if hot.len() < 2 * n_lines {
                hot.resize(2 * n_lines, 0);
            }
            for &(l, up) in model.iter() {
                let lhs = if up { fl[l] } else { -fl[l] };
                if lhs > bound_of(l, up) - hp.blk_drop_tol {
                    hot[l * 2 + up as usize] = epoch;
                }
            }
            drop(hot);
            if let Some(po) = prem_out {
                let mu = solver.row_duals();
                po.clear();
                po.resize(num_b, 0.0);
                for (i, &(l, up)) in model.iter().enumerate() {
                    if mu[i] <= 0.0 {
                        continue;
                    }
                    let sg = if up { 1.0 } else { -1.0 };
                    for b in 0..num_b {
                        po[b] -= mu[i] * sg * sens[l][b] / DELTA_T;
                    }
                }
            }
            let mut cand = u.clone();
            clamp_to_bounds(&mut cand, &state.action_bounds);
            if is_flow_feasible(challenge, state, &cand) {
                return Some(cand);
            }
            return None;
        }
        viol.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        add.clear();
        for &(_, l, up) in viol.iter() {
            if add.len() >= hp.blk_add_cap.max(1) || model.len() + add.len() >= hp.blk_row_cap {
                break;
            }
            add.push((l, up));
        }
        if add.is_empty() {
            return None;
        }
        push_rows(&add, &mut grow, &mut rhs);
        solver.add_rows(&grow, &rhs);
        for &(l, up) in add.iter() {
            in_model[l * 2 + up as usize] = true;
            model.push((l, up));
        }
    }
    None
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

    let price_table: Option<Vec<Vec<f64>>> = None;
    let causal_table: Option<Vec<Vec<f64>>> = if hp.rt_prem_scale != 0.0 {
        Some(causal_price_table(challenge, hp.rt_prem_scale))
    } else {
        None
    };
    let price_table_raw: Option<Vec<Vec<f64>>> = price_table.clone();
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
    let pt: Option<&[Vec<f64>]> = price_table.as_deref().or(causal_table.as_deref());
    let pt_raw: Option<&[Vec<f64>]> = price_table_raw.as_deref().or(causal_table.as_deref());
    let pt_plan: Option<&[Vec<f64>]> = if hp.pt_window_only { None } else { pt };
    let (sigma, p_jump) = if price_table.is_some() {
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
    let mut pilot_tab: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut pilot_prem: Vec<Vec<f64>> = Vec::new();
    let mut hp_pass = hp;
    let hp_orig = hp;
    let hp_orig_with_level = hp;
    let mut pass = 0usize;
    if hp.pilot_paths > 0 {
        if hp.pilot_soc_levels > 0 {
            hp_pass.dp_soc_levels = hp.pilot_soc_levels.max(2);
        }
        if hp.pilot_gh > 0 {
            hp_pass.dp_gh_points = hp.pilot_gh;
        }
    }
    let (dps, coupling_prems, expected_premiums) = loop {
    let hp = hp_pass;
    let derate_tab: Vec<Vec<[f64; 2]>> = if !pilot_tab.is_empty() {
        pilot_tab.clone()
    } else if hp.dp_derate_mode > 0 && n_lines > 0 {
        build_dp_derate(challenge, &sens, &hp)
    } else {
        Vec::new()
    };
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

    
    let expected_premiums: Vec<Vec<f64>> = if !pilot_prem.is_empty() {
        let sc = if hp.pilot_prem_scale > 0.0 { hp.pilot_prem_scale } else { 1.0 };
        let mut e = expected_premiums;
        for t in 0..e.len().min(pilot_prem.len()) {
            for b in 0..e[t].len().min(pilot_prem[t].len()) {
                e[t][b] += sc * pilot_prem[t][b];
            }
        }
        e
    } else {
        expected_premiums
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

    let hp_pre = {
        let mut h = hp;
        if hp.pre_dp_cheap && hp.ct_premium > 0 && hp.use_ptdf_ct && n_lines > 0 {
            h.dp_soc_levels = (hp.dp_soc_levels / 2).max(17);
            h.dp_action_levels = (hp.dp_action_levels / 2).max(5);
        }
        h
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
                derate_tab.get(b).map(|v| v.as_slice()),
                &da_at_node,
                challenge.num_steps,
                sigma,
                p_jump,
                mean_pareto,
                second_pareto,
                fleet_soc_norm,
                &hp_pre,
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

    let dps = if hp.ct_premium > 0 && hp.use_ptdf_ct && n_lines > 0 {
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
                    if hp.pre_dp_cheap {
                        let node = battery.node;
                        let da_at_node: Vec<f64> = (0..n_t)
                            .map(|t| plan_price(challenge, pt_plan, t, node) + expected_premiums[t][b])
                            .collect();
                        build_battery_dp(
                            battery,
                            derate_tab.get(b).map(|v| v.as_slice()),
                            &da_at_node,
                            n_t,
                            sigma,
                            p_jump,
                            mean_pareto,
                            second_pareto,
                            fleet_soc_norm,
                            &hp,
                        )
                    } else {
                        dps[b].clone()
                    }
                } else {
                    let node = battery.node;
                    let da_ct: Vec<f64> = (0..n_t)
                        .map(|t| plan_price(challenge, pt_plan, t, node) + ep_ct[t][b])
                        .collect();
                    build_battery_dp(
                        battery,
                        derate_tab.get(b).map(|v| v.as_slice()),
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
                            derate_tab.get(b).map(|v| v.as_slice()),
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
    if pass >= 1 && pass < hp_orig.pilot_iters.max(1) && hp_orig.pilot_paths > 0 && n_lines > 0 {
        let mut newp: Vec<Vec<f64>> = Vec::new();
        let _ = pilot_derate(challenge, &dps, &sens, &hp_orig_with_level, &mut newp);
        let w = hp_orig.pilot_damp.clamp(0.0, 1.0);
        for t in 0..pilot_prem.len().min(newp.len()) {
            for b in 0..pilot_prem[t].len().min(newp[t].len()) {
                pilot_prem[t][b] = w * pilot_prem[t][b] + (1.0 - w) * newp[t][b];
            }
        }
        pass += 1;
        if pass >= hp_orig.pilot_iters.max(1) {
            hp_pass.dp_soc_levels = hp_orig.dp_soc_levels;
            hp_pass.dp_gh_points = hp_orig.dp_gh_points;
        }
        continue;
    }
    if pass == 0 && hp.pilot_paths > 0 && n_lines > 0 {
        pilot_tab = pilot_derate(challenge, &dps, &sens, &hp, &mut pilot_prem);
        hp_pass = hp_orig;
        if hp.pilot_no_derate != 0 {
            pilot_tab.clear();
        } else {
            hp_pass.dp_power_scale = 0.0;
        }
        if hp.pilot_prem_mode == 2 {
            hp_pass.ct_premium = 0;
        }
        if hp.pilot_prem_mode == 0 {
            pilot_prem.clear();
        }
        if hp_orig.pilot_iters > 1 && hp_orig.pilot_cheap_mid != 0 {
            hp_pass.dp_soc_levels = hp.dp_soc_levels;
            hp_pass.dp_gh_points = hp.dp_gh_points;
        }
        pass = 1;
        continue;
    }
    break (dps, coupling_prems, expected_premiums);
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
    block_lp_reset();
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        policy(c, s, &dps, &sens, &coupling_prems, &hp, &delta_cong, cwv_lambda, gram, pt, pt_raw)
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

struct BlockPlan {
    start_t: usize,
    end_t: usize,
    actions: Vec<Vec<f64>>,
}

fn block_lp_slot() -> &'static Mutex<Option<BlockPlan>> {
    static SLOT: OnceLock<Mutex<Option<BlockPlan>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

fn block_lp_reset() {
    *block_lp_slot().lock().unwrap() = None;
    hot_lines().lock().unwrap().clear();
    *hot_epoch().lock().unwrap() = 0;
}

fn hot_lines() -> &'static Mutex<Vec<u32>> {
    static HOT: OnceLock<Mutex<Vec<u32>>> = OnceLock::new();
    HOT.get_or_init(|| Mutex::new(Vec::new()))
}

fn hot_epoch() -> &'static Mutex<u32> {
    static EP: OnceLock<Mutex<u32>> = OnceLock::new();
    EP.get_or_init(|| Mutex::new(0))
}

mod dual_simplex {
    const TOL_P: f64 = 1e-9;
    const TOL_PIV: f64 = 1e-10;

    pub struct Solver<'a> {
        n: usize,
        ng: usize,
        pub m: usize,
        c: &'a [f64],
        hi: &'a [f64],
        grp: &'a [usize],
        sgn: &'a [f64],
        g: Vec<f64>,
        x: Vec<f64>,
        d: Vec<f64>,
        at_upper: Vec<bool>,
        basic_row: Vec<i64>,
        basis: Vec<usize>,
        binv: Vec<f64>,
    }

    impl<'a> Solver<'a> {
        pub fn new(c: &'a [f64], hi: &'a [f64], grp: &'a [usize], sgn: &'a [f64], ng: usize) -> Self {
            let n = c.len();
            let mut x = vec![0.0f64; n];
            let mut d = vec![0.0f64; n];
            let mut at_upper = vec![false; n];
            for j in 0..n {
                d[j] = -c[j];
                if c[j] > 0.0 {
                    at_upper[j] = true;
                    x[j] = hi[j];
                }
            }
            Solver {
                n, ng, m: 0, c, hi, grp, sgn, g: Vec::new(), x, d, at_upper,
                basic_row: vec![-1; n], basis: Vec::new(), binv: Vec::new(),
            }
        }

        pub fn add_rows(&mut self, rows: &[f64], rhs: &[f64]) {
            let ng = self.ng;
            let k = rhs.len();
            if k == 0 {
                return;
            }
            let n = self.n;
            let m0 = self.m;
            let m1 = m0 + k;
            let mut net = vec![0.0f64; ng];
            for j in 0..n {
                if self.x[j] != 0.0 {
                    net[self.grp[j]] += self.sgn[j] * self.x[j];
                }
            }
            let mut nb = vec![0.0f64; m1 * m1];
            for i in 0..m0 {
                nb[i * m1..i * m1 + m0].copy_from_slice(&self.binv[i * m0..i * m0 + m0]);
            }
            for r in 0..k {
                let gr = &rows[r * ng..r * ng + ng];
                let mut rb = vec![0.0f64; m0];
                for i in 0..m0 {
                    let j = self.basis[i];
                    if j < n {
                        rb[i] = self.sgn[j] * gr[self.grp[j]];
                    }
                }
                let row = m0 + r;
                for col in 0..m0 {
                    let mut acc = 0.0f64;
                    for i in 0..m0 {
                        if rb[i] != 0.0 {
                            acc += rb[i] * self.binv[i * m0 + col];
                        }
                    }
                    nb[row * m1 + col] = -acc;
                }
                nb[row * m1 + row] = 1.0;
                let mut v = rhs[r];
                for q in 0..ng {
                    v -= gr[q] * net[q];
                }
                self.x.push(v);
                self.d.push(0.0);
                self.at_upper.push(false);
                self.basic_row.push(row as i64);
                self.basis.push(n + row);
                self.g.extend_from_slice(gr);
            }
            self.binv = nb;
            self.m = m1;
        }

        pub fn row_duals(&self) -> Vec<f64> {
            (0..self.m)
                .map(|i| {
                    let j = self.n + i;
                    if self.basic_row[j] >= 0 { 0.0 } else { self.d[j].max(0.0) }
                })
                .collect()
        }

        pub fn solution(&self) -> Vec<f64> {
            (0..self.n).map(|j| self.x[j].max(0.0).min(self.hi[j])).collect()
        }

        pub fn solve(&mut self, max_pivots: usize) -> bool {
            let n = self.n;
            let m = self.m;
            let ng = self.ng;
            let nt = n + m;
            let hi = self.hi;
            let ub = |j: usize| -> f64 { if j < n { hi[j] } else { f64::INFINITY } };
            let mut alpha = vec![0.0f64; nt];
            let mut ga = vec![0.0f64; ng];
            let mut aq = vec![0.0f64; m];
            let mut colq = vec![0.0f64; m];
            let mut cand: Vec<(f64, usize)> = Vec::new();
            let mut flips: Vec<usize> = Vec::new();
            let mut dvec = vec![0.0f64; m];
            let mut dnet = vec![0.0f64; ng];
            let mut pivots = 0usize;
            let _ = self.c;
            loop {
                let mut r: i64 = -1;
                let mut worst = TOL_P;
                let mut below = true;
                for i in 0..m {
                    let k = self.basis[i];
                    let v = self.x[k];
                    if -v > worst {
                        worst = -v;
                        r = i as i64;
                        below = true;
                    }
                    let u = ub(k);
                    if v - u > worst {
                        worst = v - u;
                        r = i as i64;
                        below = false;
                    }
                }
                if r < 0 {
                    return true;
                }
                if pivots >= max_pivots {
                    return false;
                }
                pivots += 1;
                let r = r as usize;
                {
                    let rowv = &self.binv[r * m..r * m + m];
                    for q in 0..ng {
                        ga[q] = 0.0;
                    }
                    for i in 0..m {
                        let w = rowv[i];
                        if w == 0.0 {
                            continue;
                        }
                        let gr = &self.g[i * ng..i * ng + ng];
                        for q in 0..ng {
                            ga[q] += w * gr[q];
                        }
                    }
                    for j in 0..n {
                        alpha[j] = self.sgn[j] * ga[self.grp[j]];
                    }
                    alpha[n..nt].copy_from_slice(rowv);
                    for i in 0..m {
                        alpha[self.basis[i]] = 0.0;
                    }
                }
                let dir = if below { -1.0f64 } else { 1.0f64 };
                cand.clear();
                cand.resize(nt, (0.0, 0));
                let mut kc = 0usize;
                for j in 0..nt {
                    let uf = if self.at_upper[j] { -1.0f64 } else { 1.0f64 };
                    let al = alpha[j];
                    let ok = al * dir * uf > TOL_PIV;
                    let dj = self.d[j] * uf;
                    let dj = if dj > 0.0 { dj } else { 0.0 };
                    let aa = if al < 0.0 { -al } else { al };
                    cand[kc] = (dj / aa, j);
                    kc += ok as usize;
                }
                cand.truncate(kc);
                if cand.is_empty() {
                    return false;
                }
                let cmp = |p1: &(f64, usize), p2: &(f64, usize)| {
                    p1.0.partial_cmp(&p2.0).unwrap_or(core::cmp::Ordering::Equal).then(p1.1.cmp(&p2.1))
                };
                const HEAD: usize = 24;
                let split = if cand.len() > HEAD {
                    cand.select_nth_unstable_by(HEAD - 1, cmp);
                    HEAD
                } else {
                    cand.len()
                };
                cand[..split].sort_unstable_by(cmp);
                let leave = self.basis[r];
                let mut slope = if below { -self.x[leave] } else { self.x[leave] - ub(leave) };
                flips.clear();
                let mut q: i64 = -1;
                let mut idx = 0usize;
                while idx < cand.len() {
                    if idx == split && split < cand.len() {
                        cand[split..].sort_unstable_by(cmp);
                    }
                    let j = cand[idx].1;
                    idx += 1;
                    let span = ub(j);
                    let dec = alpha[j].abs() * span;
                    if span.is_finite() && slope - dec > TOL_P {
                        slope -= dec;
                        flips.push(j);
                    } else {
                        q = j as i64;
                        break;
                    }
                }
                if q < 0 {
                    q = flips.pop().unwrap() as i64;
                }
                if !flips.is_empty() {
                    for v in dvec.iter_mut() {
                        *v = 0.0;
                    }
                    for v in dnet.iter_mut() {
                        *v = 0.0;
                    }
                    let mut any_g = false;
                    for &j in flips.iter() {
                        let delta = if self.at_upper[j] { -ub(j) } else { ub(j) };
                        self.x[j] += delta;
                        self.at_upper[j] = !self.at_upper[j];
                        if j < n {
                            dnet[self.grp[j]] += self.sgn[j] * delta;
                            any_g = true;
                        } else {
                            dvec[j - n] += delta;
                        }
                    }
                    if any_g {
                        for i in 0..m {
                            let gr = &self.g[i * ng..i * ng + ng];
                            let mut acc = 0.0f64;
                            for k in 0..ng {
                                acc += gr[k] * dnet[k];
                            }
                            dvec[i] += acc;
                        }
                    }
                    for i2 in 0..m {
                        let bi = &self.binv[i2 * m..i2 * m + m];
                        let mut acc = 0.0f64;
                        for k in 0..m {
                            acc += bi[k] * dvec[k];
                        }
                        let kb = self.basis[i2];
                        self.x[kb] -= acc;
                    }
                }
                let q = q as usize;
                if q < n {
                    let gq = self.grp[q];
                    for k in 0..m {
                        colq[k] = self.sgn[q] * self.g[k * ng + gq];
                    }
                    for i in 0..m {
                        let bi = &self.binv[i * m..i * m + m];
                        let mut acc = 0.0f64;
                        for k in 0..m {
                            acc += bi[k] * colq[k];
                        }
                        aq[i] = acc;
                    }
                } else {
                    for i in 0..m {
                        aq[i] = self.binv[i * m + (q - n)];
                    }
                }
                let arq = aq[r];
                if arq.abs() <= TOL_PIV {
                    return false;
                }
                let target = if below { 0.0 } else { ub(leave) };
                let theta_p = (self.x[leave] - target) / arq;
                self.x[q] += theta_p;
                for i in 0..m {
                    let k = self.basis[i];
                    self.x[k] -= theta_p * aq[i];
                }
                self.x[leave] = target;
                let theta_d = self.d[q] / arq;
                for j in 0..nt {
                    self.d[j] -= theta_d * alpha[j];
                }
                self.d[q] = 0.0;
                self.d[leave] = -theta_d;
                self.at_upper[leave] = !below;
                self.at_upper[q] = false;
                self.basic_row[leave] = -1;
                self.basic_row[q] = r as i64;
                self.basis[r] = q;
                let inv = 1.0 / arq;
                for k in 0..m {
                    self.binv[r * m + k] *= inv;
                }
                for i in 0..m {
                    if i == r {
                        continue;
                    }
                    let f = aq[i];
                    if f == 0.0 {
                        continue;
                    }
                    for k in 0..m {
                        self.binv[i * m + k] -= f * self.binv[r * m + k];
                    }
                }
            }
        }
    }
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
        duals: &mut Vec<f64>,
    ) -> Option<Vec<f64>> {
        duals.clear();
        duals.resize(m, 0.0);
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
        let mut pivots = 0usize;
        while pivots < max_pivots {
            let mut enter: Option<usize> = None;
            {
                let mut best = -EPS;
                for j in 0..width {
                    if row_of[j] >= 0 {
                        continue;
                    }
                    let v = tab[obj + j];
                    if v < best {
                        best = v;
                        enter = Some(j);
                    }
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
            for kk in 0..ncols {
                tab[pbase + kk] *= inv;
            }
            prow.copy_from_slice(&tab[pbase..pbase + ncols]);
            for i in 0..=m {
                if i == lr {
                    continue;
                }
                let base = i * ncols;
                let f = tab[base + j];
                if f.abs() > 1e-14 {
                    let dst = &mut tab[base..base + ncols];
                    for kk in 0..ncols {
                        dst[kk] -= f * prow[kk];
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

        for i in 0..m {
            duals[i] = tab[obj + n + i];
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

fn block_plan_action(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    coupling_prems: &[Vec<f64>],
    hp: &Hyperparameters,
    pt: Option<&[Vec<f64>]>,
    pt_raw: Option<&[Vec<f64>]>,
    t: usize,
    n_steps: usize,
) -> Option<Vec<f64>> {
    let w_head = hp.blk_w.max(1);
    let w_tail = if hp.blk_w_tail == 0 { w_head } else { hp.blk_w_tail.max(1) };
    let head_end = if hp.blk_head_pct >= 100 {
        n_steps
    } else {
        let raw = n_steps * hp.blk_head_pct / 100;
        (raw / w_head) * w_head
    };
    let (w, tile_base) = if t < head_end { (w_head, 0usize) } else { (w_tail, head_end) };
    let need_new = {
        let guard = block_lp_slot().lock().unwrap();
        match &*guard {
            Some(bp) => t < bp.start_t || t >= bp.end_t,
            None => true,
        }
    };
    if need_new {
        let start_t = tile_base + ((t - tile_base) / w) * w;
        let end_t = (start_t + w).min(n_steps).min(if t < head_end { head_end.max(start_t + 1) } else { n_steps });
        if hp.blk_duty > 1 && ((start_t - tile_base) / w) % hp.blk_duty != 0 {
            *block_lp_slot().lock().unwrap() =
                Some(BlockPlan { start_t, end_t, actions: Vec::new() });
            return None;
        }
        let actions = solve_block_lp(
            challenge, state, dps, sens, coupling_prems, hp, pt, pt_raw, start_t, end_t,
        )
        .unwrap_or_default();
        *block_lp_slot().lock().unwrap() = Some(BlockPlan { start_t, end_t, actions });
    }
    let guard = block_lp_slot().lock().unwrap();
    let bp = guard.as_ref()?;
    if bp.actions.is_empty() || t < bp.start_t || t >= bp.end_t {
        return None;
    }
    let mut cand = bp.actions[t - bp.start_t].clone();
    drop(guard);
    clamp_to_bounds(&mut cand, &state.action_bounds);
    if is_flow_feasible(challenge, state, &cand) {
        Some(cand)
    } else {
        None
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
    coupling_prems: &[Vec<f64>],
    hp: &Hyperparameters,
    pt: Option<&[Vec<f64>]>,
    pt_raw: Option<&[Vec<f64>]>,
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
        let mut n_pos = 0usize;
        let mut has_up = false;
        let mut has_dn = false;
        let mut span_e = 0.0f64;
        let mut span_v = 0.0f64;
        for k in 0..hp.blk_term_k {
            if tv.up_w[k] > 0.0 {
                s_min = s_min.min(tv.up_s[k]);
                s_max = s_max.max(tv.up_s[k]);
                span_e += tv.up_w[k];
                span_v += tv.up_w[k] * tv.up_s[k];
                has_up = true;
                n_pos += 1;
            }
            if tv.dn_w[k] > 0.0 {
                s_min = s_min.min(tv.dn_s[k]);
                s_max = s_max.max(tv.dn_s[k]);
                span_e += tv.dn_w[k];
                span_v += tv.dn_w[k] * tv.dn_s[k];
                has_dn = true;
                n_pos += 1;
            }
        }
        let _ = n_pos;
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
    let prem_w = hp.blk_prem;

    let mut struct_rows: Vec<f64> = Vec::new();
    let mut struct_rhs: Vec<f64> = Vec::new();
    let mut n_struct = 0usize;
    let soc_key = |b: usize, t_end: usize, up: bool| -> usize { ((b * w + (t_end - 1)) * 2) + up as usize };
    let mut soc_elig = vec![false; 2 * num_b * w];
    let mut soc_in = vec![false; 2 * num_b * w];
    let mut soc_active: Vec<(usize, usize, bool)> = Vec::new();
    let mut head_up = vec![0.0f64; num_b];
    let mut head_dn = vec![0.0f64; num_b];

    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let soc0 = state.socs[b];

        for t in 0..w {
            let tau = start_t + t;
            let (c_hi, d_hi) = if t == 0 {
                let (u_min, u_max) = state.action_bounds[b];
                ((-u_min).max(0.0), u_max.max(0.0))
            } else {
                (battery.power_charge_mw.max(0.0), battery.power_discharge_mw.max(0.0))
            };
            hi[c_idx(b, t)] = c_hi;
            hi[d_idx(b, t)] = d_hi;

            let mut price = match hp.blk_px_mode {
                0 => plan_price(challenge, pt, tau, node),
                1 => {
                    if t == 0 {
                        state.rt_prices[node]
                    } else {
                        plan_price(challenge, pt, tau, node)
                    }
                }
                _ => {
                    if t == 0 {
                        state.rt_prices[node]
                    } else {
                        plan_price(challenge, pt_raw.or(pt), tau, node)
                    }
                }
            };
            if prem_w != 0.0 {
                price += prem_w
                    * coupling_prems.get(tau).and_then(|row| row.get(b)).copied().unwrap_or(0.0);
            }
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

        let headroom_up = battery.soc_max_mwh - soc0;
        let headroom_down = soc0 - battery.soc_min_mwh;
        head_up[b] = headroom_up.max(0.0);
        head_dn[b] = headroom_down.max(0.0);
        let lazy = hp.blk_soc_lazy != 0;
        if max_charge_total > headroom_up + 1e-9 {
            let mut cap = 0.0f64;
            for t_end in 1..=w {
                cap += ETA_CHARGE * DELTA_T * hi[c_idx(b, t_end - 1)];
                if cap <= headroom_up + 1e-9 {
                    continue;
                }
                if lazy {
                    soc_elig[soc_key(b, t_end, true)] = true;
                    continue;
                }
                let base = n_struct * n_vars;
                struct_rows.resize(base + n_vars, 0.0);
                {
                    let row = &mut struct_rows[base..base + n_vars];
                    for t in 0..t_end {
                        row[c_idx(b, t)] = ETA_CHARGE * DELTA_T;
                        row[d_idx(b, t)] = -(DELTA_T / ETA_DISCHARGE);
                    }
                }
                struct_rhs.push(headroom_up.max(0.0));
                n_struct += 1;
            }
        }
        if max_discharge_total > headroom_down + 1e-9 {
            let mut cap = 0.0f64;
            for t_end in 1..=w {
                cap += (DELTA_T / ETA_DISCHARGE) * hi[d_idx(b, t_end - 1)];
                if cap <= headroom_down + 1e-9 {
                    continue;
                }
                if lazy {
                    soc_elig[soc_key(b, t_end, false)] = true;
                    continue;
                }
                let base = n_struct * n_vars;
                struct_rows.resize(base + n_vars, 0.0);
                {
                    let row = &mut struct_rows[base..base + n_vars];
                    for t in 0..t_end {
                        row[d_idx(b, t)] = DELTA_T / ETA_DISCHARGE;
                        row[c_idx(b, t)] = -(ETA_CHARGE * DELTA_T);
                    }
                }
                struct_rhs.push(headroom_down.max(0.0));
                n_struct += 1;
            }
        }
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
        let mut e = hot_epoch().lock().unwrap();
        *e = e.saturating_add(1);
        *e
    };
    {
        let hot = hot_lines().lock().unwrap();
        let mem = hp.blk_hot_mem;
        let mut cand: Vec<(u32, usize, bool)> = Vec::new();
        for l in 0..n_lines {
            if reach[l] <= 1e-9 || challenge.network.flow_limits[l] <= 1e-6 {
                continue;
            }
            for &up in &[true, false] {
                let hk = l * 2 + up as usize;
                let last = hot.get(hk).copied().unwrap_or(0);
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

    let mut duals: Vec<f64> = Vec::new();
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
        for &(b, t_end, up) in soc_active.iter() {
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
            &mut duals,
        )?;

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
        let mut soc_viol = 0usize;
        {
            for b in 0..num_b {
                let mut e = 0.0f64;
                for t_end in 1..=w {
                    e += ETA_CHARGE * DELTA_T * sol[c_idx(b, t_end - 1)]
                        - (DELTA_T / ETA_DISCHARGE) * sol[d_idx(b, t_end - 1)];
                    let ku = soc_key(b, t_end, true);
                    if e > head_up[b] + 1e-7 && soc_elig[ku] && !soc_in[ku] {
                        soc_in[ku] = true;
                        soc_active.push((b, t_end, true));
                        soc_viol += 1;
                    }
                    let kd = soc_key(b, t_end, false);
                    if -e > head_dn[b] + 1e-7 && soc_elig[kd] && !soc_in[kd] {
                        soc_in[kd] = true;
                        soc_active.push((b, t_end, false));
                        soc_viol += 1;
                    }
                }
            }
        }
        if viol.is_empty() && soc_viol == 0 {
            let mut hot = hot_lines().lock().unwrap();
            if hot.len() < 2 * n_lines {
                hot.resize(2 * n_lines, 0);
            }
            for &(l, _, up) in tight.iter() {
                hot[l * 2 + up as usize] = epoch;
            }
            drop(hot);
            x = sol;
            ok = true;
            break;
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
