
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Mutex, OnceLock};
use tig_challenges::energy_arbitrage::*;
use tig_challenges::energy_arbitrage::constants::{
    DELTA_T, EPS_FLOW, ETA_CHARGE, ETA_DISCHARGE, KAPPA_DEG, KAPPA_TX,
};

/// Generic numerical tolerance. The implementation uses the challenge's
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
    /// Bisection iterations of the feasibility-scaling fallback.
    pub bisect_iters: usize,
    /// Passes of the PTDF-aware coordinate polish.
    pub coord_polish_passes: usize,
    /// Day-ahead lookahead window (steps) used for the quantile threshold policy.
    pub lookahead_horizon: usize,
    /// Max fuel (runtime fuel units) the optimization rollout may spend before it
    /// falls back to zero actions for the remaining steps. 0 = spend all the fuel
    /// the runtime makes available (minus a small safety reserve). Always capped so
    /// it cannot trigger an out-of-fuel exit.
    pub fuel_budget: u64,
    /// Number of multi-start seeds fed to the joint projected-gradient ascent, in
    /// priority order `[target, dp_seed, zero]`. The seed list is truncated to this
    /// count, so 3 keeps all seeds (baseline behaviour), 2 drops the least-favourable
    /// `zero` seed, and 1 keeps only the `target` warm-start.
    pub num_seeds: usize,
    /// Enable Polyak heavy-ball momentum on the projected-gradient ascent step
    /// direction (`dir = MOMENTUM_BETA*velocity + grad`). Default `false` keeps
    /// the pure backtracking gradient path (iso-baseline). The velocity is reset
    /// per seed, so the first step is identical to the no-momentum path.
    pub use_momentum: bool,
    /// i80: cosine annealing of heavy-ball β: 0.999→0.7 over go outer iters.
    /// β(t) = BETA_END + (0.999-BETA_END)*(1+cos(π·t/(T-1)))/2. Default false =
    /// constant MOMENTUM_BETA=0.999 (LLVM const-folds false branch away).
    #[serde(default)]
    pub use_cosine_beta: bool,
    /// Enable pairwise battery action perturbation after the PGA step (cross-poll
    /// titan_v3/t49/i87 VarA α=±0.125). Tries ±PAIR_POLISH_ALPHA × span on pairs
    /// of batteries, accepts if PTDF-feasible and improves decomposable DP value.
    /// Default `false` = iso-baseline.
    pub use_pair_polish: bool,
    /// Anticipate network congestion by adding PTDF-weighted premiums to DP prices
    /// (port P8 from t52/i15). `false` = no-op path byte-equivalent (all premiums
    /// zero). `#[serde(default)]` ensures old hp_json without this key parse cleanly.
    #[serde(default)]
    pub anticipate_lmp: bool,
    /// Enable joint pairwise battery-action exchange polish after coordinate polish
    /// (cross-poll P7 from t52/i13-i14). Alphas {±0.5,±0.25}×span, budget-reset-per-
    /// pass, rollback if infeasible. Default `false` = iso-baseline (no-op branch).
    #[serde(default)]
    pub use_joint_pair_polish: bool,
    /// Max distinct battery pairs probed per `joint_pair_polish` pass (reset each pass).
    /// Only read inside the `use_joint_pair_polish` branch. Default 64.
    #[serde(default = "default_joint_pair_budget")]
    pub joint_pair_budget: usize,
    /// CASSURE CF=15: LP dispatch oracle alongside PGA. LP linearizes the DP value
    /// function (dv/dsoc) and solves the per-timestep dispatch exactly on the PTDF
    /// polytope. If LP solution (after feasibility projection) beats PGA result,
    /// it wins. false = iso-baseline (PGA only). Evaluates ceiling of LP family.
    #[serde(default)]
    pub use_lp_dispatch: bool,
    /// Number of most-congested lines to include in LP (sorted by |base_flow|/limit).
    /// 0 = all lines. Smaller = faster LP. Default 0.
    #[serde(default)]
    pub lp_max_lines: usize,
    /// Max simplex pivots per LP call. 0 = default (2000).
    #[serde(default)]
    pub lp_pivot_budget: usize,
    /// CASSURE CF=17 i47: ADMM dispatch replaces Lagrangian subgradient.
    /// At each timestep, decomposes network dispatch per battery via ADMM
    /// augmented Lagrangian (port from t50 run_admm_dispatch, ρ-const 6it).
    /// false = iso-baseline (PGA path, byte-equivalent to i26/i44).
    #[serde(default)]
    pub use_dual_dispatch: bool,
    /// ADMM iterations per timestep (use_dual_dispatch=true). Default 6.
    /// Proven convergence at 6it with rho=0.2 (t50 production value).
    #[serde(default = "default_admm_iters")]
    pub max_admm_iters: usize,
    /// ADMM penalty parameter ρ. Constant (ρ-const → guaranteed convergence).
    /// Default 0.2 (t50 proven optimal: rho=0.20, iters=6, Hindsight a5918d91).
    #[serde(default = "default_admm_rho")]
    pub admm_rho: f64,
    /// CASSURE CF=19 i48: MPC receding-horizon rollout as per-battery target-setter.
    /// Replaces the myopic single-period threshold heuristic with H-step lookahead.
    /// false = exact i26 code path (iso-baseline with use_dual_dispatch=false).
    #[serde(default)]
    pub use_mpc_lookahead: bool,
    /// Rollout horizon (steps) for MPC evaluation. Default 12 (~3h quarter-hourly).
    #[serde(default = "default_mpc_horizon")]
    pub mpc_horizon: usize,
    /// Number of candidate actions evaluated per battery in MPC rollout. Default 5.
    #[serde(default = "default_mpc_n_cand")]
    pub mpc_n_cand: usize,
    /// Pivot gate: MPC fires only when |a| >= mpc_pivot_threshold * max(u_max, -u_min).
    /// 0.0 = every step (old every-step behavior). 0.5 = A1 moderate gate. 0.7 = A2 strict gate.
    #[serde(default)]
    pub mpc_pivot_threshold: f64,
    /// Also gate MPC on RT price band spike/dip (A1 variant). Requires rt_bands history.
    #[serde(default)]
    pub mpc_use_rt_gate: bool,
    /// CASSURE CF=21 i50: SQDP (Stochastic Quadratic DP) value function approximation.
    /// Replaces the 9-action × 33-SOC-level backward DP with a quadratic cut
    /// (α+β·soc+γ·soc²) per timestep, fitted from 5-point analytical backward pass.
    /// Forward policy uses O(5) analytical candidates instead of 65-grid scan.
    /// false = iso-baseline (grid DP, parity with i26/i49).
    #[serde(default)]
    pub use_sqdp: bool,
    /// SWAP #4 i51: fleet-coupling endogenous congestion-dual feedback.
    /// Before PGA, estimates joint battery actions at soc_mid from pre-built DPs,
    /// computes endogenous line flows, derives per-battery coupling premiums that
    /// discount/amplify the effective price to discourage congestion-increasing actions.
    /// These premiums are injected into pick_dp_action at policy time (no DP rebuild).
    /// false = iso-baseline (no coupling, byte-equivalent to i50 use_sqdp=false path).
    #[serde(default)]
    pub use_coupling_cut: bool,
    /// SWAP #5 i52: primal aggregate fleet SOC regularizer (convexify-the-sum, GDD insight
    /// `9064f5d7`/`6e60694b`). Per-battery DP convexifies each VF individually -> fleet
    /// duality gap. Fix: soft quadratic penalty -lambda*(soc_norm - fleet_soc_norm)^2 added
    /// to the VF at build-time for each SOC state. fleet_soc_norm = mean normalized initial
    /// SOC over all batteries. Term is PRIMAL (not a price premium), one-shot O(n_batteries),
    /// SOFT in VF (POCS still enforces hard feasibility at policy time).
    /// false = iso-baseline (byte-equivalent to i51 use_coupling_cut=false path).
    #[serde(default)]
    pub use_aggregate_reg: bool,
    /// lambda coefficient for the aggregate SOC regularizer (only when use_aggregate_reg=true).
    /// Sweep: A1=0.15, A2=0.30, B=0.0 (parity). Default 0.0.
    #[serde(default)]
    pub agg_reg_lambda: f64,
    /// i55 OCO constraint tracking (port from t50/i59 lines 1877-1963, Huang `78025f66`).
    /// After building initial DPs, runs a cheap greedy forward simulation to detect
    /// which lines are violated by the dispatch. Computes OCO Lagrange multipliers
    /// μ_l[t] = clamp(η*(|flow_l[t]|-limit_l), 0, limit*0.5) then adjusts DA premiums
    /// (ep_ct[t][b] -= PTDF_lb*sign_l*μ_l*(1-kappa)) and rebuilds DPs at half resolution.
    /// false = iso-baseline (byte-equivalent to i52 use_aggregate_reg=false path).
    #[serde(default)]
    pub use_ptdf_ct: bool,
    /// OCO step size η: μ ← clamp(η*viol, 0, limit*0.5). Default 0.25 (conservative for ~150 batt).
    #[serde(default = "default_ct_step_eta")]
    pub ct_step_eta: f64,
    /// Reference tracking damping κ: ct_scale = 1-κ. 0.0 = full CT. Default 0.0.
    #[serde(default)]
    pub ct_ref_kappa: f64,
    /// i87 GDD-EXP: port P_GDD_EXP from t50/i68 (Sinha-Vaze 2024 `48397995`).
    /// Augments the LINEAR OCO premium (mu_oco, bounded by limit*0.5 → attenuates →
    /// under-corrects dense co-congested clusters, root cause of i84 max-norm REVERT)
    /// with a normalized EXPONENTIAL premium that AMPLIFIES the most-exposed batteries:
    ///   v_frac_l = (|flow_l|-limit_l)/limit_l   (dimensionless O(0.01-0.5))
    ///   s_b      = Σ_l v_frac_l * |ptdf_{b,l}|  (per-battery normalized exposure)
    ///   ep_ct[t][b] -= exp(ct_gdd_alpha·s_b) - 1
    /// ct_gdd_alpha=0.0 → zero-cost branch, byte-identical to i85. Single-peak @1.0 on t50.
    #[serde(default)]
    pub ct_gdd_alpha: f64,
    /// L2 i78: quadratic Taylor curvature correction on DP backward and policy lookups.
    /// Reduces O(h²) linear-interpolation bias for concave value functions at O(1) cost.
    /// false = iso-baseline (identical code path).
    #[serde(default)]
    pub use_dp_value_shift: bool,
    /// L4 i85: composite water-value delta-congestion correction (port t52/i50, KEPT +590Q).
    /// Builds two aggregate (fleet-level) 1-D DPs — with and without the fleet-average
    /// congestion premium — and adds λ·(dv_agg_cong − dv_agg_nocong) to each battery's
    /// dv_dsoc inside `analytic_gradient`. Re-injects the PTDF shadow-price coupling that
    /// the per-battery decoupled DP misses. false = iso-baseline (delta_cong all-zeros,
    /// cwv_lambda forced to 0.0 → byte-identical to i81).
    #[serde(default)]
    pub use_composite_wv: bool,
    /// Blend weight λ for the delta-congestion correction. Only read when use_composite_wv=true.
    /// Peak on t52 ∈ [0.15, 0.30] (λ=0.25 KEPT +590Q; λ=0.50 REVERTED −1,412Q). Default 0.25.
    #[serde(default = "default_cwv_lambda")]
    pub cwv_lambda: f64,
    /// Resolution of the aggregate fleet DP grid (SOC levels). Default 65. Only when use_composite_wv=true.
    #[serde(default = "default_cwv_agg_levels")]
    pub cwv_agg_levels: usize,
    /// Number of congestion-exposure clusters for per-cluster delta_cong (i53 dé-dilution).
    /// 1 = fleet-average (parity gate). Sweep: {2,3,4}. Only when use_composite_wv=true.
    #[serde(default = "default_cwv_clusters")]
    pub cwv_clusters: usize,
}

/// Heavy-ball momentum coefficient for `projected_gradient_ascent`. Hardcoded
/// const (NOT a serde `f64` field) — adding a no-op `f64` HP field changes LLVM
/// const-prop / codegen and broke iso-binary parity at t52/i9 (+51% regression).
const MOMENTUM_BETA: f64 = 0.999;
/// i80 cosine annealing endpoint. Const → LLVM eliminates cosine path when use_cosine_beta=false.
const BETA_END: f64 = 0.7;
/// Action span fraction tried in pairwise perturbation step (cross-poll t49/i87 VarA).
const PAIR_POLISH_ALPHA: f64 = 0.125;
/// Max pair checks per pairwise_perturb_step call (fuel-safe budget).
const PAIR_POLISH_BUDGET: usize = 64;
/// Congestion ratio threshold: lines with |f_exo|/limit > LMP_THRESHOLD receive a
/// premium. Set to 0.5 (proven optimal on t52: threshold=0.5 > 0.65 by +8,465 Q).
const LMP_THRESHOLD: f64 = 0.5;
/// Scale factor applied to the 20 $/MWh base premium. Hardcoded (not serde f64).
/// P9 ext: 2.0 (scale=2.0 was BEST on t52/i17 +6,471Q vs scale=1.5; testing on t53).
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

/// Per-solve adaptive outer-iteration budget pool.
/// Early-converging timestep PGA calls donate unused iterations here;
/// iteration-bound calls claim extra before running past their base budget.
/// Reset at the start of each `solve_challenge` call so nonces are independent.
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
    // levels == 0  →  SQDP mode: values[t] = [alpha_t, beta_t, gamma_t]
    // levels > 0   →  grid mode: values[t] is the SOC-level value array
    levels: usize,
    values: Vec<Vec<f64>>,
    use_shift: bool,
}

// Evaluate the future value function V_{t_next}(soc).
// In SQDP mode (levels==0): quadratic α+β·soc+γ·soc².
// In grid mode: piecewise-linear (or quadratic-Taylor when use_shift) interpolation.
#[inline(always)]
fn dp_eval_future(dp: &BatteryDP, t_next: usize, soc: f64) -> f64 {
    let t = t_next.min(dp.values.len() - 1);
    if dp.levels == 0 {
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

/// Quadratic Taylor correction on top of linear interpolation.
/// Eliminates O(h²) bias for concave value functions at O(1) cost per lookup.
/// Uses Newton forward 2nd-difference: correction = α(α-1)/2 · (V[j+2]-2V[j+1]+V[j]).
/// Zero at both endpoints (α=0,1), maximal at midpoint; positive for concave VF.
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

// ----- SQDP backward pass (CASSURE CF=21 i50) -----
//
// Replaces the grid DP (9-action × 33-SOC) with a quadratic value-function cut
// α_t + β_t·soc + γ_t·soc² per timestep, computed analytically from 5 SOC samples
// at s ∈ {-1, -0.5, 0, 0.5, 1} (normalized). Analytical action solve replaces
// the 9-action grid scan in the backward pass and the 65-grid in pick_dp_action.
// Returns a BatteryDP with levels=0 (sentinel for SQDP mode); values[t] = [α,β,γ].
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

    // 5 symmetric normalized points s ∈ {-1, -0.5, 0, 0.5, 1}
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

    // Price weights (same as grid build_battery_dp)
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

    // SQDP mode: levels=0, values[t] = [alpha_t, beta_t, gamma_t]
    let mut values: Vec<Vec<f64>> = vec![vec![0.0_f64; 3]; num_steps + 1];
    // Terminal V_{num_steps} = 0 → all zeros already

    for t in (0..num_steps).rev() {
        let da = da_at_node[t];
        let price_low = da * (1.0 - sigma);
        let price_high = da * (1.0 + sigma);
        let price_jump_low = da * (1.0 + jump_floor);
        let price_jump_high = da * (1.0 + jump_ceiling);

        let prices = [price_low, price_high, price_jump_low, price_jump_high];
        let weights = [w_low_p, w_high_p, w_jump_low, w_jump_high];

        // Future VF coefficients α_{t+1}, β_{t+1}, γ_{t+1}
        let alpha_f = values[t + 1][0];
        let beta_f = values[t + 1][1];
        let gamma_f = values[t + 1][2];

        // Price-independent quadratic coefficients for analytical action solve
        let c2 = dt * dt * (gamma_f / (eta_d * eta_d) - KAPPA_DEG / cap2);
        let d2 = dt * dt * (gamma_f * eta_c * eta_c - KAPPA_DEG / cap2);

        // Compute expected V_t at each of 5 SOC sample points
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

                // a = 0 (always feasible)
                {
                    let sn = battery.apply_action_to_soc(0.0, soc);
                    let v = alpha_f + beta_f * sn + gamma_f * sn * sn;
                    if v > best { best = v; }
                }

                // Discharge region (a ≥ 0)
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

                // Charge region (a ≤ 0)
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

        // Fit quadratic α̃ + β̃·s + γ̃·s² via least squares on the 5 symmetric points.
        // Precomputed sums for s ∈ {-1,-0.5,0,0.5,1}:
        //   n=5, Σs=0, Σs²=2.5, Σs³=0, Σs⁴=2.125
        // Closed-form (symmetric decoupling):
        //   γ̃ = (5·Σs²v − 2.5·Σv) / 4.375
        //   β̃ = Σsv / 2.5
        //   α̃ = (Σv − 2.5·γ̃) / 5
        let sv: f64  = v_samples.iter().sum();
        let ssv: f64 = S_NORM.iter().zip(v_samples.iter()).map(|(&s, &v)| s * v).sum();
        let s2v: f64 = S_NORM.iter().zip(v_samples.iter()).map(|(&s, &v)| s * s * v).sum();

        let gamma_n = (5.0 * s2v - 2.5 * sv) / 4.375;
        let beta_n  = ssv / 2.5;
        let alpha_n = (sv - 2.5 * gamma_n) / 5.0;

        // Convert normalized-space (s ∈ [-1,1]) → physical-space (soc):
        //   γ = γ̃/hs²,  β = β̃/hs − 2γ·soc_mid,  α = α̃ − β̃·soc_mid/hs + γ̃·soc_mid²/hs²
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

    BatteryDP { soc_lo: 0.0, soc_step_inv: 0.0, levels: 0, values, use_shift: false }
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
                        interp_value_q(next, next_soc, soc_lo, soc_step_inv, last)
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
            // SWAP #5 i52: soft primal aggregate fleet SOC regularizer.
            // Penalizes deviation from fleet-average normalized SOC at build time.
            // SOFT in VF (not a hard constraint); POCS still enforces feasibility.
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
        // SQDP mode: analytical derivative d/d(soc) [α + β·soc + γ·soc²] = β + 2γ·soc
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

/// L4 i85 (port t52/i50): build a single aggregate (fleet- or cluster-level) 1-D DP over
/// the summed battery as if it were one big battery with total power / total capacity, given
/// a capacity-weighted DA price path. Used to derive the delta-congestion correction signal.
/// Grid mode only (use_shift=false) — never SQDP, never value-shift Taylor.
fn build_aggregate_dp(
    batteries: &[Battery],
    da_prices_fleet: &[f64], // fleet-capacity-weighted price per timestep (len = num_steps)
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

    // Precompute action bounds per grid point for the aggregate battery.
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

        // Adaptive action grid for the aggregate battery (coarser, 11 points).
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

            // Apply-action-to-soc for aggregate battery.
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

    BatteryDP { soc_lo, soc_step_inv, levels, values, use_shift: false }
}

/// Compute dV_agg/dE_agg at timestep t and aggregate SOC e_agg (reuses dv_dsoc).
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
        // SQDP mode: analytical action selection — O(5) candidates instead of 65-grid scan.
        let next_t = (t + 1).min(dp.values.len() - 1);
        let beta_f = dp.values[next_t][1];
        let gamma_f = dp.values[next_t][2];
        let dt = DELTA_T;
        let cap2 = (battery.capacity_mwh * battery.capacity_mwh).max(1e-9);
        // Discharge region (a ∈ [0, hi]): f(a) = c2·a² + c1·a + const
        let c2 = dt * dt * (gamma_f / (ETA_DISCHARGE * ETA_DISCHARGE) - KAPPA_DEG / cap2);
        let c1 = dt * (price - KAPPA_TX - (beta_f + 2.0 * gamma_f * soc) / ETA_DISCHARGE);
        let a_d = if c2 < -1e-12 {
            (-c1 / (2.0 * c2)).clamp(0.0_f64.max(lo), hi)
        } else {
            if c1 > 0.0 { hi } else { 0.0_f64.max(lo) }
        };
        // Charge region (a ∈ [lo, 0]): f(a) = d2·a² + d1·a + const
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

    // Grid mode: existing 65-point scan (unchanged).
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

// ---- ADMM dispatch (CASSURE CF=17 i47, PORT from t50 run_admm_dispatch) ----
//
// Replaces Lagrangian subgradient i46 (identification phase too slow, 40it+SLP=32.5s).
// ADMM augmented Lagrangian: per-battery grid scan minimizing profit - rho/2 penalty.
// Converges in 6 iterations ρ-const=0.2 (t50 proven: Hindsight a5918d91).
// Returns best feasible primal action tracked during ADMM (fallback = zero).

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

    // Early exit: if init already feasible, no ADMM needed.
    let mut any_violated = false;
    for l in 0..n_l {
        let limit = challenge.network.flow_limits[l];
        let mut f = base_flows[l];
        for b in 0..n_b { f += sens[l][b] * init_actions[b]; }
        if f.abs() > limit + EPS_FLOW { any_violated = true; break; }
    }
    if !any_violated { return init_actions.to_vec(); }

    let mut actions = init_actions.to_vec();

    // Best feasible tracker (zero is safe baseline — exogenous violations handled below).
    let zero = vec![0.0_f64; n_b];
    let mut best_feasible = zero.clone();
    let mut best_feasible_val = total_step_value(challenge, state, dps, &zero);

    // Slack z: initial projection of current battery flows to [-limit, limit].
    let mut s: Vec<f64> = (0..n_l).map(|l| {
        let limit = challenge.network.flow_limits[l];
        let mut bat_f = 0.0_f64;
        for b in 0..n_b { bat_f += sens[l][b] * actions[b]; }
        (base_flows[l] + bat_f).clamp(-limit, limit)
    }).collect();

    // Dual y (zero-init per timestep; ADMM doesn't need cross-timestep warm-start).
    let mut y = vec![0.0_f64; n_l];

    for _iter in 0..hp.max_admm_iters {
        let prev_actions = actions.clone();

        // Current battery flows per line.
        let mut bat_flow = vec![0.0_f64; n_l];
        for l in 0..n_l {
            for b in 0..n_b { bat_flow[l] += sens[l][b] * actions[b]; }
        }

        // Battery x-update: maximize profit - ADMM augmented penalty.
        // For each battery b: argmax_u { profit(u) - sum_l (rho/2) * (off_l - imp_l*u)^2 }
        // where off_l = s[l] - base_l + y[l]/rho - (bat_flow[l] - imp_l*actions[b]).
        const GRID: usize = 65;
        for b in 0..n_b {
            let battery = &challenge.batteries[b];
            let soc = state.socs[b];
            let price = state.rt_prices[battery.node];
            let (lo, hi) = state.action_bounds[b];

            // Precompute per-line constants (allocated once per battery, reused over grid).
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

            // Propagate delta to bat_flow (avoids full recompute next iter).
            let delta = best_u - actions[b];
            for l in 0..n_l { bat_flow[l] += sens[l][b] * delta; }
            actions[b] = best_u;
        }

        // Slack z-update: project total flow to [-limit, limit].
        for l in 0..n_l {
            let limit = challenge.network.flow_limits[l];
            s[l] = (bat_flow[l] + base_flows[l] - y[l] / rho).clamp(-limit, limit);
        }

        // Dual y-update + convergence check.
        let mut max_resid = 0.0_f64;
        for l in 0..n_l {
            let resid = s[l] - bat_flow[l] - base_flows[l];
            y[l] += rho * resid;
            max_resid = max_resid.max(resid.abs());
        }
        let max_du = (0..n_b)
            .map(|b| (actions[b] - prev_actions[b]).abs())
            .fold(0.0_f64, f64::max);

        // Track best feasible primal found so far.
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

/// Fast greedy forward simulation for OCO-CT (i55). Follows DP policy without PTDF
/// projection or PGA — O(n_t × n_b × policy_action_levels). Used ONLY to estimate
/// which lines would be violated by the unconstrained dispatch (not the actual solution).
/// Returns flows_all[t][l] = exo_flow_l[t] + Σ_b ptdf_lb * greedy_action_b[t].
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
                action[b] = pick_dp_action(&dps[b], battery, t, soc, price, (lo, hi), hp);
            }
        }
        let exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
        // Active-set: only compute battery-affected flows for candidate lines.
        // Non-candidate lines have near-zero sensitivity → OCO premium delta = 0 → safe to skip.
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

/// Project + bisection fallback (scale toward zero if projection fails).
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
    // Bisection fallback: scale toward zero (which is feasible by assumption).
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

/// Analytic gradient of immediate profit + DP shadow value w.r.t. u_b.
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
        // L4 i85: per-battery delta-congestion correction delta_cong[b][t] (zeros when off).
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
            // Subgradient at u=0: pick direction with steeper slope average.
            -0.5 * (DELTA_T / ETA_DISCHARGE + ETA_CHARGE * DELTA_T)
        };
        // dv_corrected = dv_decoupled + λ·delta_cong[b][t] (re-inject PTDF congestion shadow-price).
        let dv = dv_dsoc(&dps[b], state.time_step, next_soc) + cwv_lambda * dc;
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
    delta_cong: &[Vec<f64>],
    cwv_lambda: f64,
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
    // Heavy-ball velocity accumulator (reset per call = per seed). Stays all-zero
    // when `hp.use_momentum` is false, so the step direction is `grad` verbatim
    // (iso-baseline path); at iter 0 velocity≈0 ⇒ first step identical either way.
    let mut velocity = vec![0.0_f64; action.len()];

    // Adaptive budget: claim any surplus donated by earlier-converging PGA calls.
    // Capped at one full base budget to avoid a single call monopolising the pool.
    let base_budget = hp.grad_outer_iters;
    let extra = iter_pool_claim(base_budget as i64) as usize;
    let total_limit = base_budget + extra;

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

        // i80: cosine-anneal β 0.999→0.7 over base_budget iters when enabled.
        let beta_t = if hp.use_cosine_beta && base_budget > 1 {
            let frac = outer_iter as f64 / (base_budget - 1) as f64;
            BETA_END + (MOMENTUM_BETA - BETA_END) * (1.0 + (std::f64::consts::PI * frac).cos()) * 0.5
        } else {
            MOMENTUM_BETA
        };

        // Step direction: heavy-ball `β·v + grad` carries the search across
        // non-improving zones where pure-gradient `lr*=0.4` would collapse.
        let dir: Vec<f64> = if hp.use_momentum {
            grad.iter()
                .zip(velocity.iter())
                .map(|(g, v)| beta_t * v + g)
                .collect()
        } else {
            grad.clone()
        };

        let mut improved = false;
        let mut cur_lr = lr;
        for _ in 0..hp.grad_ls_iters {
            // step_scale uses the gradient norm (unchanged backtracking schedule);
            // momentum only re-orients the direction, not the line-search budget.
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
                lr = cur_lr * 1.4;
                // Accumulate velocity only on an accepted (improving) step.
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
    }
    // Return unused budget to pool so later iteration-bound timesteps can use it.
    if exited_early {
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
) -> Vec<f64> {
    let results: Vec<(Vec<f64>, f64)> = seeds
        .into_iter()
        .map(|seed| {
            projected_gradient_ascent(
                challenge, state, dps, sens, base_flows, seed, hp, delta_cong, cwv_lambda,
            )
        })
        .collect();

    let mut best_action = vec![0.0_f64; challenge.num_batteries];
    let mut best_value = total_step_value(challenge, state, dps, &best_action);
    for (a, v) in results {
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

/// Pairwise battery action perturbation polish (cross-poll titan_v3/t49/i87 VarA).
/// For each pair (i,j), tries ±PAIR_POLISH_ALPHA×span perturbations (opposing direction),
/// checks PTDF flow feasibility, accepts if per-pair DP value improves.
/// Budget-limited to PAIR_POLISH_BUDGET pairs per call (fuel-safe).
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

    // Current total flows = base (exogenous) + PTDF × action
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

                // Try (+α, -α) and (-α, +α)
                for &sign in &[1.0_f64, -1.0_f64] {
                    let cand_i = (cur_i + sign * PAIR_POLISH_ALPHA * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - sign * PAIR_POLISH_ALPHA * span_j).clamp(lo_j, hi_j);
                    let di = cand_i - cur_i;
                    let dj = cand_j - cur_j;
                    if di.abs() < EPS && dj.abs() < EPS {
                        continue;
                    }

                    // PTDF flow feasibility (linear model, exact for PTDF networks)
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

                    // Decomposable per-battery value (no coupling in DP value fn)
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

/// Joint pairwise cross-battery exchange polish (P7 port from t52/i13-i14).
/// Tries alphas {-0.5,-0.25,0.25,0.5}×span on opposing pairs (i,j), accepts the
/// best feasible candidate per pair. Budget resets each pass (first-improvement
/// restart). Criterion: per-pair dp_action_value (same as t52/i13 verbatim).
/// Called AFTER coordinate_polish_step with an explicit rollback at the call site.
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
    // Reconstruct current flows from base + actions.
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

        // Coupling premium: effective price includes endogenous congestion correction.
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
            // Pivot gate: only invoke the costly H-step rollout at price-action pivots.
            // thr=0.0 means every step (old i48 behavior). A1: 0.5+rt_gate. A2: 0.7 only.
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

    // Independent per-battery DP-preferred seed.
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

    // Baseline flows depend on this step's exogenous injections.
    let zero = vec![0.0_f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero);

    // Primary optimizer: ADMM dispatch (use_dual_dispatch=true, i47) OR
    // joint projected gradient ascent (use_dual_dispatch=false = iso-baseline i26).
    let mut result = if hp.use_dual_dispatch {
        // i47: ADMM dispatch, port from t50 run_admm_dispatch (rho-const 6it).
        // Warm-start from target (DP policy). No RefCell state needed.
        admm_dispatch(challenge, state, dps, sens, &base_flows, &target, hp)
    } else {
        // ISO-BASELINE path (parity with i44/i26 at use_dual_dispatch=false).
        let mut seeds = vec![target, dp_seed, zero.clone()];
        seeds.truncate(hp.num_seeds.max(1));
        let pga_result = joint_optimize_step(
            challenge, state, dps, sens, &base_flows, seeds, hp, delta_cong, cwv_lambda,
        );
        let pga_val = total_step_value(challenge, state, dps, &pga_result);
        let mut r = pga_result;

        // CASSURE CF=15: LP oracle (kept for backward compat, off by default).
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

    // P7: joint pairwise cross-battery exchange (port t52/i13-i14). Runs after
    // coordinate_polish (NO-OP at passes=0) with explicit rollback if infeasible.
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

    // Congestion-anticipation premiums added to DP prices (P8 port from t52/i15).
    // When anticipate_lmp=false (default), all premiums are zero → byte-equivalent
    // no-op path; the iso-binary parity bench (variant B) must confirm this.
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

    // SWAP #5 i52: fleet-average normalized initial SOC (one-shot, O(n_batteries)).
    // Normalized to [0,1] per battery: (soc_init - soc_min) / (soc_max - soc_min).
    // Used as regularization target in build_battery_dp when use_aggregate_reg=true.
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

    // Fleet-coupling premiums (SWAP #4 i51). One-pass endogenous correction:
    // estimate joint fleet dispatch at soc_mid, compute net endogenous line flows,
    // derive per-battery per-timestep coupling premiums injected at policy time.
    // No DP rebuild needed — correction applied in pick_dp_action price argument.
    let coupling_prems: Vec<Vec<f64>> = if hp.use_coupling_cut && n_lines > 0 {
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let base = 20.0 * LMP_PREMIUM_SCALE;
        let threshold = LMP_THRESHOLD;
        // Estimate action at representative SOC (midpoint of each battery's SOC range).
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
                    pick_dp_action(&dps[b], battery, t, soc_mid, price, (lo, hi), &hp);
            }
        }
        // Compute endogenous flows and coupling premiums.
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
                // Only add coupling where endogenous fleet increases congestion.
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

    // i55 OCO constraint tracking (port from t50/i59 lines 1877-1963, Huang 78025f66).
    // i57 TIME-CLAWBACK: active-set candidate_lines + selective DP rebuild (touched only).
    // Discriminant vs static P8: μ tracks REAL dispatch flow violations (not exo-only),
    // capturing dispatch-induced congestion missed by the one-shot exo premium.
    // When use_ptdf_ct=false → this block is skipped, dps unchanged = iso-baseline.
    let dps = if hp.use_ptdf_ct && n_lines > 0 {
        let eta = hp.ct_step_eta;
        let ct_scale = 1.0 - hp.ct_ref_kappa;
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let limits = &challenge.network.flow_limits;

        // Active-set: only lines where batteries have meaningful sensitivity.
        // Lines with all |sens[l][b]| < threshold can't be congested by battery dispatch
        // → OCO premium delta = 0 for those batteries → safe to skip entirely.
        const ACTIVE_SENS_THRESH: f64 = 1e-4;
        let candidate_lines: Vec<usize> = (0..n_lines)
            .filter(|&l| limits[l] > 1e-6 && sens[l].iter().any(|&s| s.abs() > ACTIVE_SENS_THRESH))
            .collect();

        // Step 1: fast greedy forward simulation restricted to candidate lines.
        let flows_all = ct_simulate_flows(challenge, &dps, &sens, &candidate_lines, &hp);

        // Step 2: OCO multipliers for candidate lines only.
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

        // Step 3: adjust premiums + track which batteries are actually touched.
        // A battery is touched iff its OCO premium differs from the baseline premium.
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

        // Step 3b (i87): GDD-EXP normalized exponential premium (port P_GDD_EXP, t50/i68).
        // The linear mu_oco above is bounded by limit*0.5 → ATTENUATES → under-corrects
        // dense co-congested clusters (the exact root cause of the i84 max-norm REVERT).
        // GDD-EXP AMPLIFIES the most-exposed batteries via Sinha-Vaze 2024 `48397995`:
        //   v_frac_l = (|flow_l|-limit_l)/limit_l  (dimensionless O(0.01-0.5))
        //   s_b      = Σ_l v_frac_l * |ptdf_{b,l}|  (per-battery normalized exposure)
        //   ep_ct[t][b] -= exp(ct_gdd_alpha·s_b) - 1   (commensurable with premium $/MWh)
        // Reuses the same `flows_all` already simulated above. Stacks ON TOP of the linear
        // OCO premium (additive). ct_gdd_alpha=0.0 → branch skipped → bit-exact i85 baseline.
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

        // Step 4: selective half-resolution DP rebuild.
        // Untouched batteries have identical premiums → reuse existing DP (Q-EXACT).
        // Only rebuild batteries whose ep_ct differs from expected_premiums.
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

    // L4 i85: composite water-value delta-congestion correction (port t52/i50).
    // delta_cong[b][t] = per-battery congestion shadow-price correction.
    // cwv_clusters=1: fleet-average path. cwv_clusters>1: K quantile-cluster path (i53 dé-dilution).
    // When use_composite_wv=false → all zeros + cwv_lambda forced to 0.0 → byte-identical to i81.
    let cwv_lambda = if hp.use_composite_wv { hp.cwv_lambda } else { 0.0 };
    let delta_cong: Vec<Vec<f64>> = if hp.use_composite_wv {
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let k = hp.cwv_clusters.max(1).min(n_b.max(1));

        if k == 1 {
            // Fleet-average path.
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
            // Per-cluster path (i53 dé-dilution): K quantile clusters by avg congestion exposure.
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
    // Reset per-solve adaptive budget pool so nonces are independent.
    iter_pool_reset();
    let solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        policy(c, s, &dps, &sens, &coupling_prems, &hp, &delta_cong, cwv_lambda)
    })?;
    save_solution(&solution)?;
    Ok(())
}

// ---- LP dispatch (CASSURE CF=15) ----

/// Exact LP dispatch per-timestep: maximize linearized value function subject to
/// battery bounds + PTDF flow constraints. Uses discharge (d) / charge (c) split
/// variables (d>=0, c>=0, action = d - c) so the simplex feasibility assumption
/// b >= 0 holds.
///
/// lp_max_lines: if > 0, only include the K most-loaded lines (by |exo_flow|/limit)
/// to bound the LP tableau size. Post-LP projection handles remaining constraints.
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

    // Select lines to include in LP.
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

    // LP: maximize c^T x, s.t. A x <= b, x >= 0.
    // Variables: x[0..num_b] = discharge (d_b), x[num_b..2*num_b] = charge magnitude (c_b).
    let n = 2 * num_b;
    let m = 4 * num_b + 2 * num_l;

    let mut c_obj = vec![0.0_f64; n];
    let mut a_mat = vec![vec![0.0_f64; n]; m];
    let mut b_vec = vec![0.0_f64; m];

    for b in 0..num_b {
        let battery = &challenge.batteries[b];
        let price = state.rt_prices[battery.node];
        let soc = state.socs[b];
        // Linear approx of future value: dv/dsoc at current soc.
        let dv = dv_dsoc(&dps[b], t, soc);

        // Discharge coefficient: price profit - tx cost + future value change.
        c_obj[b] = (price - KAPPA_TX) * DELTA_T - dv * (DELTA_T / ETA_DISCHARGE);
        // Charge coefficient: future value gain - price - tx cost.
        c_obj[num_b + b] = dv * ETA_CHARGE * DELTA_T - (price + KAPPA_TX) * DELTA_T;

        let (u_min, u_max) = state.action_bounds[b];
        let r = 4 * b;
        // d_b <= u_max
        a_mat[r][b] = 1.0;
        b_vec[r] = u_max.max(0.0);
        // c_b <= -u_min (u_min <= 0)
        a_mat[r + 1][num_b + b] = 1.0;
        b_vec[r + 1] = (-u_min).max(0.0);
        // d_b * dt / eta_d <= soc - soc_min  (SOC lower)
        a_mat[r + 2][b] = DELTA_T / ETA_DISCHARGE;
        b_vec[r + 2] = (soc - battery.soc_min_mwh).max(0.0);
        // c_b * eta_c * dt <= soc_max - soc  (SOC upper)
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
            // Positive flow: sum PTDF * (d - c) <= limit - exo
            a_mat[rp][b] += ptdf;
            a_mat[rp][num_b + b] -= ptdf;
            // Negative flow: sum -PTDF * (d - c) <= limit + exo
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

/// Minimal dense Bland's-rule simplex LP solver embedded in t53_engine for CASSURE.
/// Solves: maximize c^T x, s.t. A x <= b, x >= 0.  Assumes b_i >= 0.
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

// Important! Do not include any tests in this file, it will result in your submission being rejected
