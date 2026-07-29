
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tig_challenges::energy_arbitrage::*;
use tig_challenges::energy_arbitrage::constants::{
    DELTA_T, EPS_FLOW, ETA_CHARGE, ETA_DISCHARGE, KAPPA_DEG, KAPPA_TX,
};

// Profiling counters (fuel-based, reset per nonce in solve_challenge).
static PROF_PGA_OUTER_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROF_POLICY_CALLS: AtomicU64 = AtomicU64::new(0);

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
    /// Enable Barzilai-Borwein step clamping (growth cap + decay floor) inside the
    /// projected-gradient ascent. Default = false → hot-path identical to v1732
    /// (LLVM dead-code-eliminates the `false` branch; no `f64::MAX` on the hot path).
    #[serde(default)]
    pub use_bb_clamps: bool,
    /// BB-clamp growth cap: bounds the multiplicative growth of the step on an
    /// improving line-search hit, via `(cur_lr*1.4).min(prev_lr*lr_growth_cap)`.
    /// Only read inside the `use_bb_clamps` branch. Default 1.05 (= i10 baked const).
    /// NB: no field-level `#[serde(default)]` — the struct-level `#[serde(default)]`
    /// fills a missing field from `Self::default()` (1.05), whereas a field-level
    /// attr would use `f64::default()` = 0.0 and silently zero the clamp.
    pub lr_growth_cap: f64,
    /// BB-clamp decay floor: floors the multiplicative shrink of the step on a
    /// failed outer iteration, via `(lr*0.4).max(prev_lr*bb_decay_factor)`.
    /// Only read inside the `use_bb_clamps` branch. Default 0.85 (= i10 baked const).
    pub bb_decay_factor: f64,
    /// Include the zero-vector seed in the joint-optimize seed set (P27, ported
    
    /// zero seed is dropped → fewer PGA trajectories = the time lever (seed-drop).
    /// No field-level `#[serde(default)]`: the struct-level attr fills a missing
    /// field from `Self::default()` (true), not `bool::default()` (false).
    pub use_zero_seed: bool,
    /// Include the independent per-battery DP-preferred seed (P29). Default true =
    /// current behaviour. When false, `dp_seed` is not even constructed (skipped).
    pub use_dp_seed: bool,
    /// Run the pairwise cross-battery coordinate-exchange polish after the
    
    /// → no-op path byte-identical (the `joint_pair_polish` call is dead-eliminated).
    /// The Q lever: recovers inter-battery coupling the independent gradient misses.
    pub use_joint_pair_polish: bool,
    
    /// inside the `use_joint_pair_polish` branch. Default 64.
    pub joint_pair_budget: usize,
    /// Anticipate network congestion by adding PTDF-weighted premiums to DP prices.
    
    /// false → no-op path byte-equivalent (expected_premiums are all zeros).
    #[serde(default)]
    pub anticipate_lmp: bool,
    /// Congestion ratio threshold: only lines with |f_exo|/limit > threshold receive
    
    pub lmp_threshold: f64,
    /// Scale factor applied to the base premium (20 $/MWh × scale). Default 1.0.
    pub lmp_premium_scale: f64,
    /// Sort battery pairs by expected gain (|price_i - price_j| × min(span_i, span_j))
    
    /// 55d08db5: "most critical scenarios first"). Default false → sequential order
    /// (byte-identical to v1816 when false). Flag LLVM-safe (bool + branch, pattern i10).
    #[serde(default)]
    pub use_pair_priority: bool,
    /// Enable Polyak heavy-ball momentum on the PGA step direction
    /// (`dir = MOMENTUM_BETA*velocity + grad`). Default false = pure gradient (iso-baseline).
    /// Velocity reset per seed. Ported from t51/i22+i24 (P6-bis β=0.9, +1.138% Q).
    /// Convention: const MOMENTUM_BETA (NOT serde f64) to preserve LLVM const-fold.
    #[serde(default)]
    pub use_momentum: bool,
    /// Enable composite water-value delta-congestion correction (L6b-bis from t51/i54 lesson).
    /// Builds two aggregate 1-D DPs over E_agg = Σ soc_b (with/without fleet-avg congestion
    /// premium) and adds lambda × (dv_agg_cong − dv_agg_nocong) to each battery's dv_dsoc in
    /// analytic_gradient. Preserves per-battery price signal (additive delta, not blend).
    /// Default false = no-op (byte-identical to previous iters).
    #[serde(default)]
    pub use_composite_wv: bool,
    /// Blend weight λ for the delta-congestion correction. Only read when use_composite_wv=true.
    /// Sweep: 0.25 (A1), 0.50 (A2). Default 0.25.
    pub cwv_lambda: f64,
    /// Number of E_agg grid levels for the aggregate DP (L6b-bis). Finer grid = sharper
    /// congestion signal. Default 65 (= i50 behaviour). Sweep: {65, 101}.
    pub cwv_agg_levels: usize,
    /// Number of congestion-exposure clusters for per-cluster delta_cong (i53 dé-dilution).
    /// 1 = fleet-average (byte-identical to i50). K>1 = per-cluster: batteries partitioned by
    /// average expected_premium into K quantile clusters; each cluster gets its own DP pair and
    /// delta_cong vector. Battery b receives delta_cong[cluster(b)][t] instead of fleet average.
    /// Default 1 (parity gate). Sweep: {2, 3, 4}. Only read when use_composite_wv=true.
    pub cwv_clusters: usize,
    /// Enable 3-battery triplet exchange polish after joint_pair_polish (P8, ported from
    /// t51/i67 KEPT Q=2,811,442). Explores (i,j,k) triplets: ±α on i, ∓α on j, +γ on k.
    /// Single-pass greedy first-improvement, budget-capped. Default false.
    #[serde(default)]
    pub use_joint_triplet_polish: bool,
    /// Max distinct triplets probed in a single greedy pass. Default 150.
    #[serde(default = "default_joint_triplet_budget")]
    pub joint_triplet_budget: usize,
    /// Top-K batteries by |action| to form triplets from. Default 15.
    #[serde(default = "default_joint_triplet_top_k")]
    pub joint_triplet_top_k: usize,
    /// i64 OCO constraint tracking: after DPs are built, simulate dispatch flows and
    /// apply PTDF-weighted Lagrange multipliers to correct expected_premiums, then
    /// selectively rebuild only touched batteries' DPs at half resolution.
    /// false = iso-baseline (byte-equivalent to i63).
    #[serde(default)]
    pub use_ptdf_ct: bool,
    /// OCO step size η: μ_l[t] = clamp(η*max(|flow_l[t]|−limit_l, 0), 0, limit*0.5).
    /// Default 0.25 (conservative). Sweep A1=0.25, A2=0.5, A3=1.0.
    #[serde(default = "default_ct_step_eta")]
    pub ct_step_eta: f64,
    /// Reference tracking damping κ: ct_scale = 1-κ. 0.0 = full CT. Default 0.0.
    #[serde(default)]
    pub ct_ref_kappa: f64,
    /// i44 per-line sensitivity-adaptive dual cap: cap_l = (limit*0.5)/(1+beta*sens_agg[l]).
    /// beta=0.0 → cap_l = limit*0.5 byte-exact (CTRL=i37). beta>0 → high-|PTDF| lines get
    /// tighter cap (anti-overshoot i43); low-sensitivity lines unchanged. Recall 86ad812c.
    #[serde(default)]
    pub ct_cap_sens_beta: f64,
    /// i45 virtual-queue drift-plus-penalty (Zhou-Wang-Li 2024, recall 00cc7807/c55e83c8).
    /// Lyapunov weight V controlling the integral multiplier: μ_l[t] = min(η·Q_l/V, limit·0.5)
    /// where Q_l ← max(0, Q_l + viol_l[t]) accumulates violations across timesteps.
    /// 0.0 (sentinel) → disabled → proportional one-shot byte-EXACT CTRL (iso-i37).
    /// Sweep: {0.5, 1.0, 2.0}. Bounds overshoot vs i43 naïf via principled regret m·O(√T).
    #[serde(default)]
    pub ct_vq_v: f64,
    /// Number of forward accumulation passes for the virtual queue (ct_vq_v > 0 only).
    /// 1 (sentinel) → single forward scan. K>1 → K scans with persistent q_virt (stronger
    /// wind-up, equivalent to a smaller effective V). Default=1 (iso-i37 when vq_v=0).
    #[serde(default = "default_ct_vq_passes")]
    pub ct_vq_passes: usize,
    /// i65 cosine annealing of heavy-ball β over PGA outer iterations.
    /// β decays from MOMENTUM_BETA (0.99) at iter 0 to BETA_END (0.7) at iter go-1:
    ///   β(t) = BETA_END + (0.99 - BETA_END) × (1 + cos(π·t/(T-1))) / 2
    /// Default false = constant MOMENTUM_BETA=0.99 (iso-baseline, LLVM const-folds false branch).
    #[serde(default)]
    pub use_cosine_beta: bool,
    /// i76 L6 PRIMAL granularity: congestion-aware NON-UNIFORM redistribution of the
    /// `dp_action_levels` DP action grid (CONSTANT number of levels — no points added).
    /// For a battery with strong `|ptdf|` toward a congested line, the protective-arbitrage
    /// optimum sits at the *knee* `u* = limit_{l*} / |sens_{l*,b}|` (the largest own action
    /// before the most-sensitive line saturates). The uniform price-aware grid places no
    /// resolution there, so the DP rounds the binding action to a coarse level and loses
    /// protective arbitrage. When `alpha>0`, a fraction `alpha` of the `dp_action_levels`
    /// points is concentrated in a narrow band around the price-favored knee; the rest keep
    /// the existing price-aware allocation. `alpha=0.0` ⇒ congestion path skipped at the
    /// call site (knee=None) ⇒ BYTE-IDENTICAL to i71. Sweep alpha ∈ {0.25, 0.5, 1.0}.
    /// Grounding: 148da942 (μ_l identifies binding lines), 9cd77d91 (finer-where-it-matters),
    /// 572151fa (+7.56% economic from non-uniform allocation, NOT quantity).
    #[serde(default)]
    pub congestion_grid_alpha: f64,
    
    /// flows[] incrementally in O(n_lines)/iter instead of O(n_lines·n_b). Iso-geometry:
    /// identical Euclidean iterates, Q BIT-EXACT. Default false = original rescan path.
    #[serde(default)]
    pub use_gram_incremental_proj: bool,
    /// Resync cadence of the Gram-incremental projection: recompute flows[] exactly
    /// every `resync_period` iters to bound float drift. Larger = fewer O(n_lines·n_b)
    /// full rescans = less time; the incremental delta-flow update is exact per step so
    /// the resync only corrects accumulated FP round-off (iso-geometry). Default 16 =
    /// byte-exact sentinel reproducing the previous baked-const path. Only read on the
    /// `use_gram_incremental_proj` path. Clamped ≥1.
    #[serde(default = "default_resync_period")]
    pub resync_period: usize,
    /// i28 L6 résidu-gate: adaptive early-exit on the PGA outer loop when the gradient
    /// norm `g_norm` drops below this threshold. `0.0` = disabled (sentinel, iso-baseline
    /// to the hard `g_norm < 1e-9` exit already present). Per-nonce: nonces that converge
    /// early exit faster; hard nonces keep the full `grad_outer_iters` budget. Iso-Q by
    /// construction when threshold ≤ effective-convergence noise. Sweep after CTRL confirms
    /// iso-Q: {1e-6, 1e-5, 1e-4}. Gate-D on i25+i26 KEPT. Grounding: 87e6099b (g_norm
    
    /// residual adaptive restart). Only in i28 (gram base).
    #[serde(default)]
    pub outer_grad_tol: f64,
    /// i29 L6 sub-fallback: Barzilai-Borwein adaptive step-size in the PGA outer loop.
    /// `false` = disabled (sentinel; iso-baseline, byte-exact to i28 with outer_grad_tol=0.0).
    /// `true` = BB-2 step: at each outer iter k>0, estimate local Lipschitz constant L via
    /// L_est = ||y_k|| / ||s_k|| where s_k = action_k − action_{k-1}, y_k = grad_k − grad_{k-1}.
    /// Step = 1/L_est = ||s_k|| / ||y_k||, clamped to [max_power×1e-4, max_power×50]. Starts
    /// the backtracking LS from the local-curvature estimate instead of the carried `lr`,
    /// converging faster when L_global (used by lr) is overly conservative. CTRL=false → iso-i28.
    /// Grounding: fdb144cd (APDB: adaptive step essential when L_global overly large),
    /// a8e6205f (APDB/rAPDB-ada adaptive step-size non-monotone backtracking). Gate: bench
    /// after i28 résidu-gate null (or in parallel). Base: i28 code (gram+outer_grad_tol).
    #[serde(default)]
    pub use_bb_step: bool,
    /// i35 L7 disaggregation-order mode in coordinate_polish_step.
    /// 0 = natural order (CTRL iso-i31). 1 = high-PTDF-danger-first (max_l |sens[l][b]|/limit[l]).
    /// 2 = low-PTDF-danger-first. 3 = sum-PTDF-descending.
    /// Grounding: fcae28e1 (priority-based disaggregation heterogeneous fleet), 7e695deb (QCQP η≠).
    #[serde(default)]
    pub disagg_order_mode: u8,
    /// i36 RÈGLE D'OR: LMP premium curvature. CTRL=1.0 (linear, iso-i31). Port of P20(t50,+5262Q)
    /// and P28(t53,+8623Q): `proba^gamma` concentrates premium on near-binding lines (ratio→1).
    /// Optimal: gamma=2.0 on t50, gamma=1.5 on t53. Sweep {1.5,2.0,2.5} via hp_json.
    #[serde(default = "default_premium_shape_gamma")]
    pub premium_shape_gamma: f64,
    /// i51 temporal-flow dispatch: bypass per-step PGA with full-horizon greedy-DP extraction
    /// + temporal rescheduling (captures SOC-carryover × line congestion coupling, +91k gap
    /// prouvé i17 LP joint). false = CTRL byte-exact (iso-i46). Gate: Q>2,588,212 elapsed≤10.5s.
    #[serde(default)]
    pub use_temporal_flow: bool,
    
    /// Only read when use_temporal_flow=true. Default: 3.
    #[serde(default = "default_flow_max_sweeps")]
    pub flow_max_sweeps: usize,
    /// Fraction of line violation to shift per temporal correction step [0.1, 1.0].
    /// Only read when use_temporal_flow=true. Default: 1.0 (full correction per step).
    #[serde(default = "default_flow_eps_scale")]
    pub flow_eps_scale: f64,
    /// i53 AUGMENT-form: if true, apply step-2 temporal reschedule POST-PASS on the champion
    /// schedule (grid_optimize result) before save_solution. Seed = champion (not greedy).
    /// Q-MONOTONE-SAFE: guard score>0 → shifts only to higher-DA-price timesteps.
    /// false = CTRL byte-exact (iso-i46). Gate: Q>2,588,212 elapsed≤10.5s.
    #[serde(default)]
    pub use_temporal_flow_postpass: bool,
    
    /// injected into the joint PGA multi-start. Substitutes dp_seed with arb_seed: for each
    /// battery, discharge at max if rt_price > threshold, else charge at max. TIME-NEUTRAL:
    /// num_seeds stays 3 (dp_seed substituted, not added). Default false = byte-exact CTRL.
    #[serde(default)]
    pub use_arb_seed: bool,
    /// Percentile threshold for arb_seed (only read when use_arb_seed=true).
    /// 0 = use arithmetic mean (P36 baseline). Non-zero = use this percentile of rt_prices
    /// as discharge threshold (e.g. 75 → discharge when price > P75, charge otherwise).
    #[serde(default)]
    pub arb_pct: u64,
    
    /// 1.0 = byte-exact i55 (seed at lo/hi bounds, slow projection). <1.0 scales toward
    /// interior: reduces flow-constraint violation → Dykstra/POCS converges faster.
    /// Sweep {0.5, 0.7, 0.9}. Default 1.0 = byte-exact i55.
    #[serde(default = "default_arb_scale")]
    pub arb_scale: f64,
    /// i87 (combine t51 P40/P42 → t52): append ONE maximally-distant price-arbitrage seed
    /// to the joint PGA multi-start. t52 champion runs single-start `[target]` (dp_seed &
    /// zero dropped) → zero basin diversity. An extreme binary arb seed (charge/discharge at
    /// action bounds by rt-price percentile) starts PGA in a DISTINCT non-convex basin
    /// (Hindsight 813c100d/be379b88/4906d1ee). Appends (does NOT substitute target). Threshold
    /// = P75 (arb_pct if >0 else 75). TIME: +1 PGA trajectory. Default false = byte-exact CTRL.
    #[serde(default)]
    pub use_arb_diversity_seed: bool,
    /// Polarity of the arb-diversity seed. false = standard economic arb (discharge when
    /// price > threshold). true = anti-correlated / opposition seed (charge when expensive),
    /// MAXIMALLY distant from the DP-like `target` (t51 P40 arb_inverse = biggest arb win).
    /// Only read when use_arb_diversity_seed=true.
    #[serde(default)]
    pub arb_diversity_inverse: bool,
    /// i89 (combine t51 P42 → t52): append a SECOND arb-diversity seed of the OPPOSITE
    /// polarity to `arb_diversity_inverse`, giving 3 maximally-distant basins
    /// `[target, inverse-arb(P{arb_pct}), std-arb(P{arb_pct})]`. t52 champion i87 runs only
    /// 2 basins `[target, inverse-arb]`; t51 proved +4,809 Q (P42) by adding the standard-polarity
    /// std_arb_third seed as a distinct 3rd PGA basin (Hindsight a7b7d662/4906d1ee/d10e28dd:
    /// std_arb_third discharge-when-expensive = distinct 3rd basin → higher Q; inter-seed
    /// DISTANCE primes over economic quality 813c100d/756420a7). Same P{arb_pct} threshold as
    /// the inverse seed. TIME: +1 PGA trajectory (~+6s, financed by 12.5s slack under SOTA gate).
    /// Only read when use_arb_diversity_seed=true. Default false = byte-exact i87 sentinel.
    #[serde(default)]
    pub use_arb_diversity_pair: bool,
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
            use_bb_clamps: false,
            lr_growth_cap: 1.05,
            bb_decay_factor: 0.85,
            use_zero_seed: true,
            use_dp_seed: true,
            use_joint_pair_polish: false,
            joint_pair_budget: 64,
            anticipate_lmp: false,
            lmp_threshold: 0.65,
            lmp_premium_scale: 1.0,
            use_pair_priority: false,
            use_momentum: false,
            use_composite_wv: false,
            cwv_lambda: 0.25,
            cwv_agg_levels: 65,
            cwv_clusters: 1,
            use_joint_triplet_polish: false,
            joint_triplet_budget: 150,
            joint_triplet_top_k: 15,
            use_ptdf_ct: false,
            ct_step_eta: 0.25,
            ct_ref_kappa: 0.0,
            ct_cap_sens_beta: 0.0,
            ct_vq_v: 0.0,
            ct_vq_passes: 1,
            use_cosine_beta: false,
            congestion_grid_alpha: 0.0,
            use_gram_incremental_proj: false,
            resync_period: 16,
            outer_grad_tol: 0.0,
            use_bb_step: false,
            disagg_order_mode: 0,
            premium_shape_gamma: 1.0,
            use_temporal_flow: false,
            flow_max_sweeps: 3,
            flow_eps_scale: 1.0,
            use_temporal_flow_postpass: false,
            use_arb_seed: false,
            arb_pct: 0,
            arb_scale: 1.0,
            use_arb_diversity_seed: false,
            arb_diversity_inverse: false,
            use_arb_diversity_pair: false,
        }
    }
}

fn default_resync_period() -> usize { 16 }
fn default_premium_shape_gamma() -> f64 { 1.0 }
fn default_flow_max_sweeps() -> usize { 3 }
fn default_flow_eps_scale() -> f64 { 1.0 }
fn default_arb_scale() -> f64 { 1.0 }

fn default_joint_triplet_budget() -> usize { 150 }
fn default_joint_triplet_top_k() -> usize { 15 }
fn default_ct_step_eta() -> f64 { 0.25 }
fn default_ct_vq_passes() -> usize { 1 }

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
        
        // clamp [0.0, 1.0]; 0.0 = congestion path off (byte-iso i71).
        hp.congestion_grid_alpha = hp.congestion_grid_alpha.clamp(0.0, 1.0);
        // Resync cadence must be ≥1 (0 or below would panic on `iter % period`).
        hp.resync_period = hp.resync_period.max(1);
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
    
    // knee `u_sat` (action magnitude that saturates this battery's most-sensitive line),
    // concentrating a fraction `alpha` of the `levels` points near the price-favored knee.
    // None ⇒ pure price-aware grid (BYTE-IDENTICAL to i71 — non-DP callers always pass None).
    cong: Option<(f64, f64)>,
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

    let grid = if actions.len() > levels {
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
    };

    // i76 congestion-aware NON-UNIFORM redistribution. CONSTANT `levels`: we keep the 3
    // anchors {base_charge, 0, base_discharge}, allocate `n_dense = round(alpha·levels)`
    // points to a narrow band around the price-favored knee, and refill the remainder from
    // the price-aware `grid` (closest-to-favored-edge first, mirroring the existing trim).
    // `cong == None` ⇒ return `grid` unchanged ⇒ byte-identical to i71.
    let (u_sat, alpha) = match cong {
        Some(c) if c.1 > 0.0 && c.0 > EPS => c,
        _ => return grid,
    };

    // Knee in the price-favored direction (discharge if selling, charge if buying), clamped
    // to the battery's signed power range. If neither region is active, no knee → keep grid.
    let favored = if price > discharge_min {
        u_sat.min(base_discharge)
    } else if price < charge_max {
        (-u_sat).max(base_charge)
    } else {
        return grid;
    };

    // n_dense reserves the 3 anchors; at least 2 dense points to actually form a band.
    let n_dense = ((alpha * levels as f64).round() as usize)
        .min(levels.saturating_sub(3))
        .max(2);
    if n_dense < 2 {
        return grid;
    }

    // Narrow band (±12% of the full action span) centred on the knee, clamped to range.
    let span = (base_discharge - base_charge).max(1e-9);
    let half_w = 0.12 * span;
    let lo_b = (favored - half_w).max(base_charge);
    let hi_b = (favored + half_w).min(base_discharge);

    let mut kept: Vec<f64> = vec![base_charge, 0.0, base_discharge];
    for i in 0..n_dense {
        let f = i as f64 / ((n_dense - 1) as f64);
        kept.push(lo_b + (hi_b - lo_b) * f);
    }
    // Refill from the price-aware grid (favored-edge-first) until we hit `levels`.
    let favored_edge = if price > discharge_min {
        base_discharge
    } else {
        base_charge
    };
    let mut fillers: Vec<f64> = grid
        .iter()
        .copied()
        .filter(|&a| {
            (a - base_charge).abs() > EPS && a.abs() > EPS && (a - base_discharge).abs() > EPS
        })
        .collect();
    fillers.sort_by(|a, b| {
        (a - favored_edge)
            .abs()
            .partial_cmp(&(b - favored_edge).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for a in fillers {
        if kept.len() >= levels {
            break;
        }
        if kept.iter().any(|&k| (k - a).abs() < EPS) {
            continue;
        }
        kept.push(a);
    }

    kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    kept.dedup_by(|a, b| (*a - *b).abs() < EPS);
    kept
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
    
    // congestion_grid_alpha==0 or the battery has no sensitive line ⇒ byte-iso i71).
    cong_knee: Option<f64>,
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

        // Hoist: adaptive_action_grid args are t-level (not s_idx-level) — compute once per timestep, not once per soc level.
        
        // for high-|ptdf| batteries. cong_knee=None ⇒ pure price-aware grid (byte-iso i71).
        let actions = adaptive_action_grid(
            battery,
            charge_max,
            discharge_min,
            (price_low + price_high) * 0.5,
            hp.dp_action_levels,
            cong_knee.map(|u| (u, hp.congestion_grid_alpha)),
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

/// Build a 1-D aggregate DP over E_agg = Σ_b soc_b using fleet-aggregate parameters.
/// Used for the delta-congestion correction (L6b-bis): build once WITH congestion premium
/// in prices and once WITHOUT; the difference isolates the pure congestion-coupling value.
/// Returns a BatteryDP struct reusing the same layout (soc_lo=E_agg_min, span=E_agg_max-min).
fn build_aggregate_dp(
    batteries: &[Battery],
    da_prices_fleet: &[f64],  // fleet-capacity-weighted price per timestep (len = num_steps)
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

    BatteryDP { soc_lo, soc_step_inv, levels, values }
}

/// Compute dV_agg/dE_agg at timestep t and aggregate SOC e_agg (reuses dv_dsoc).
#[inline(always)]
fn aggregate_dv_dsoc(agg_dp: &BatteryDP, t: usize, e_agg: f64) -> f64 {
    dv_dsoc(agg_dp, t, e_agg)
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

    for raw in adaptive_action_grid(battery, charge_max, discharge_min, price, hp.policy_action_levels, None) {
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

/// Fast greedy forward simulation for OCO-CT (i64). Follows DP policy without PTDF
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

/// i51 temporal-flow dispatch — warm-startée par duals OCO-CT (Bertsekas ε-relaxation).
/// Extrait le schedule complet via greedy-DP forward-pass (optimal par batterie),
/// puis corrige les violations de lignes via temporal rescheduling (shifts d'énergie

/// congestion inter-temporel-spatial que le PGA per-step ne peut pas voir.
/// Faisabilité stricte : EPS_FLOW exact du framework, JAMAIS plus laxiste.
fn solve_temporal_flow(
    challenge: &Challenge,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    candidate_lines: &[usize],
    hp: &Hyperparameters,
) -> Vec<Vec<f64>> {
    let n_b = challenge.num_batteries;
    let n_t = challenge.num_steps;
    let n_l = sens.len();
    let limits = &challenge.network.flow_limits;
    let batteries = &challenge.batteries;
    let frac = hp.flow_eps_scale.clamp(0.1, 1.0);

    // Step 1: Greedy forward-pass extraction from DPs (DA price, consistent with ct_simulate_flows).
    // This gives the DP-optimal per-battery schedule without joint line constraints.
    let mut socs: Vec<f64> = batteries.iter().map(|b| b.soc_initial_mwh).collect();
    let mut schedule: Vec<Vec<f64>> = Vec::with_capacity(n_t);
    for t in 0..n_t {
        let mut action = vec![0.0_f64; n_b];
        for b in 0..n_b {
            let battery = &batteries[b];
            let soc = socs[b];
            let bounds = compute_action_bounds(battery, soc);
            if bounds.1 - bounds.0 > EPS {
                let price = challenge.market.day_ahead_prices[t][battery.node];
                action[b] = pick_dp_action(&dps[b], battery, t, soc, price, bounds, hp);
            }
        }
        // Project onto line constraints at this timestep (per-step feasibility).
        let exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
        let base: Vec<f64> = (0..n_l).map(|l| exo.get(l).copied().unwrap_or(0.0)).collect();
        let bounds_vec: Vec<(f64, f64)> = (0..n_b)
            .map(|b| compute_action_bounds(&batteries[b], socs[b]))
            .collect();
        project_polytope(&mut action, &bounds_vec, sens, &base, limits,
                         hp.proj_max_iters, None, hp.resync_period);
        schedule.push(action.clone());
        for b in 0..n_b {
            let battery = &batteries[b];
            socs[b] = battery.apply_action_to_soc(action[b], socs[b])
                .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        }
    }

    // Step 2: Temporal rescheduling — shift discharge between timesteps to relieve
    // line violations while exploiting price differentials (SOC-carryover physics).
    for _sweep in 0..hp.flow_max_sweeps.max(1) {
        // Recompute SOC trajectory from current schedule.
        let mut soc_traj = vec![vec![0.0_f64; n_b]; n_t + 1];
        for b in 0..n_b {
            soc_traj[0][b] = batteries[b].soc_initial_mwh;
        }
        for t in 0..n_t {
            for b in 0..n_b {
                let battery = &batteries[b];
                soc_traj[t + 1][b] = battery
                    .apply_action_to_soc(schedule[t][b], soc_traj[t][b])
                    .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
            }
        }
        // Compute current line flows for candidate lines.
        let mut flows: Vec<Vec<f64>> = Vec::with_capacity(n_t);
        for t in 0..n_t {
            let exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut ft = vec![0.0_f64; n_l];
            for &l in candidate_lines {
                ft[l] = exo.get(l).copied().unwrap_or(0.0)
                    + sens[l].iter().zip(schedule[t].iter()).map(|(s, a)| s * a).sum::<f64>();
            }
            flows.push(ft);
        }
        // i52 FIX: use raw DA prices (not OCO-CT-adjusted) for opportunity cost scoring.
        
        // congested high-DA-price timesteps look "cheap" → energy shifted FROM high-DA-price TO
        // low-DA-price uncongested timesteps → Q=-436k. Raw DA prices give the correct direction.
        let mut any_shift = false;
        for t in 0..n_t {
            for &l in candidate_lines {
                let flow_lt = flows[t][l];
                let viol = flow_lt.abs() - limits[l];
                if viol <= EPS { continue; }
                let sign_flow = if flow_lt >= 0.0 { 1.0_f64 } else { -1.0_f64 };
                // Find battery with largest contribution to this line's violation.
                let mut b_star = usize::MAX;
                let mut contrib_max = 0.0_f64;
                for b in 0..n_b {
                    let contrib = sens[l][b] * sign_flow * schedule[t][b];
                    if contrib > contrib_max {
                        contrib_max = contrib;
                        b_star = b;
                    }
                }
                if b_star == usize::MAX || contrib_max < EPS { continue; }
                let b = b_star;
                // Max discharge to shift away from t (fraction of violation).
                let s_lb = sens[l][b].abs().max(EPS);
                let max_shift = (viol / s_lb).min(schedule[t][b].max(0.0)) * frac;
                if max_shift < EPS { continue; }
                let battery = &batteries[b];
                // Raw DA price at t (not OCO-CT adjusted — we want actual revenue comparison).
                let price_t = challenge.market.day_ahead_prices[t][batteries[b].node];
                // Find best future timestep t' to compensate (store-and-discharge-later).
                // Only shift to t' where raw DA price is HIGHER than at t (score > 0 guard).
                let mut best_tp = usize::MAX;
                let mut best_score = 0.0_f64; // guard: only accept if revenue-improving
                for tp in (t + 1)..n_t {
                    // Only shift to tp if line l has capacity there.
                    let flow_ltp = flows[tp][l];
                    let slack_tp = limits[l] - flow_ltp.abs();
                    if slack_tp < EPS { continue; }
                    // SOC feasibility: tp must have room for extra discharge.
                    let soc_tp = soc_traj[tp][b];
                    let bounds_tp = compute_action_bounds(battery, soc_tp);
                    let room_tp = bounds_tp.1 - schedule[tp][b];
                    if room_tp < max_shift * 0.1 { continue; }
                    // Score: raw DA price differential (positive = more profitable at tp).
                    let score = challenge.market.day_ahead_prices[tp][batteries[b].node]
                        - price_t + 1e-3 * slack_tp;
                    // Guard: only accept revenue-improving shifts (score > 0 threshold).
                    if score > best_score {
                        best_score = score;
                        best_tp = tp;
                    }
                }
                if best_tp == usize::MAX { continue; }
                let tp = best_tp;
                let bounds_t = compute_action_bounds(battery, soc_traj[t][b]);
                let bounds_tp = compute_action_bounds(battery, soc_traj[tp][b]);
                // Actual delta: min of what we want to shift and what tp can absorb.
                let delta = max_shift.min(bounds_tp.1 - schedule[tp][b]).max(0.0);
                if delta < EPS { continue; }
                let new_at = (schedule[t][b] - delta).clamp(bounds_t.0, bounds_t.1);
                let actual_delta = schedule[t][b] - new_at;
                if actual_delta < EPS { continue; }
                let new_atp = (schedule[tp][b] + actual_delta).clamp(bounds_tp.0, bounds_tp.1);
                schedule[t][b] = new_at;
                schedule[tp][b] = new_atp;
                any_shift = true;
                // Propagate flow delta for t and tp (only candidate lines).
                for &ll in candidate_lines {
                    flows[t][ll] -= sens[ll][b] * actual_delta;
                    flows[tp][ll] += sens[ll][b] * actual_delta;
                }
            }
        }
        if !any_shift { break; }
    }
    schedule
}

/// i54 AUGMENT-form: apply step-2 temporal reschedule (from solve_temporal_flow L1041-1143)
/// as a POST-PASS on an existing schedule. Seed = champion (NOT greedy forward-pass).
/// After shifts, FULL SEQUENTIAL RE-PROJECTION from first-modified-t: clamp ALL timesteps
/// to new SOC bounds (ensures SOC-chain consistency) + project_polytope at modified_tp
/// (ensures EPS_FLOW feasibility). Fixes i53 nonces_invalid=2 (SOC chain inconsistency).
/// Single-thread: no spawn/rayon/threading — all loops are sequential .iter().
fn apply_temporal_postpass(
    challenge: &Challenge,
    schedule: &mut Vec<Vec<f64>>,
    sens: &[Vec<f64>],
    candidate_lines: &[usize],
    hp: &Hyperparameters,
    gram: Option<&[Vec<f64>]>,
) {
    let n_b = challenge.num_batteries;
    let n_t = challenge.num_steps;
    let n_l = sens.len();
    let limits = &challenge.network.flow_limits;
    let batteries = &challenge.batteries;
    let frac = hp.flow_eps_scale.clamp(0.1, 1.0);
    // Track tp timesteps (discharge INCREASED) + earliest t where any shift occurred.
    let mut modified_tp = vec![false; n_t];
    let mut first_shift_t: Option<usize> = None;

    for _sweep in 0..hp.flow_max_sweeps.max(1) {
        // Recompute SOC trajectory from current schedule.
        let mut soc_traj = vec![vec![0.0_f64; n_b]; n_t + 1];
        for b in 0..n_b {
            soc_traj[0][b] = batteries[b].soc_initial_mwh;
        }
        for t in 0..n_t {
            for b in 0..n_b {
                let battery = &batteries[b];
                soc_traj[t + 1][b] = battery
                    .apply_action_to_soc(schedule[t][b], soc_traj[t][b])
                    .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
            }
        }
        // Compute current line flows for candidate lines.
        let mut flows: Vec<Vec<f64>> = Vec::with_capacity(n_t);
        for t in 0..n_t {
            let exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            let mut ft = vec![0.0_f64; n_l];
            for &l in candidate_lines {
                ft[l] = exo.get(l).copied().unwrap_or(0.0)
                    + sens[l].iter().zip(schedule[t].iter()).map(|(s, a)| s * a).sum::<f64>();
            }
            flows.push(ft);
        }
        // Identical to step-2 of solve_temporal_flow (L1041-1143) except seed=champion.
        let mut any_shift = false;
        for t in 0..n_t {
            for &l in candidate_lines {
                let flow_lt = flows[t][l];
                let viol = flow_lt.abs() - limits[l];
                if viol <= EPS { continue; }
                let sign_flow = if flow_lt >= 0.0 { 1.0_f64 } else { -1.0_f64 };
                let mut b_star = usize::MAX;
                let mut contrib_max = 0.0_f64;
                for b in 0..n_b {
                    let contrib = sens[l][b] * sign_flow * schedule[t][b];
                    if contrib > contrib_max {
                        contrib_max = contrib;
                        b_star = b;
                    }
                }
                if b_star == usize::MAX || contrib_max < EPS { continue; }
                let b = b_star;
                let s_lb = sens[l][b].abs().max(EPS);
                let max_shift = (viol / s_lb).min(schedule[t][b].max(0.0)) * frac;
                if max_shift < EPS { continue; }
                let battery = &batteries[b];
                let price_t = challenge.market.day_ahead_prices[t][batteries[b].node];
                let mut best_tp = usize::MAX;
                let mut best_score = 0.0_f64;
                for tp in (t + 1)..n_t {
                    let flow_ltp = flows[tp][l];
                    let slack_tp = limits[l] - flow_ltp.abs();
                    if slack_tp < EPS { continue; }
                    let soc_tp = soc_traj[tp][b];
                    let bounds_tp = compute_action_bounds(battery, soc_tp);
                    let room_tp = bounds_tp.1 - schedule[tp][b];
                    if room_tp < max_shift * 0.1 { continue; }
                    let score = challenge.market.day_ahead_prices[tp][batteries[b].node]
                        - price_t + 1e-3 * slack_tp;
                    if score > best_score {
                        best_score = score;
                        best_tp = tp;
                    }
                }
                if best_tp == usize::MAX { continue; }
                let tp = best_tp;
                let bounds_t = compute_action_bounds(battery, soc_traj[t][b]);
                let bounds_tp = compute_action_bounds(battery, soc_traj[tp][b]);
                let delta = max_shift.min(bounds_tp.1 - schedule[tp][b]).max(0.0);
                if delta < EPS { continue; }
                let new_at = (schedule[t][b] - delta).clamp(bounds_t.0, bounds_t.1);
                let actual_delta = schedule[t][b] - new_at;
                if actual_delta < EPS { continue; }
                let new_atp = (schedule[tp][b] + actual_delta).clamp(bounds_tp.0, bounds_tp.1);
                schedule[t][b] = new_at;
                schedule[tp][b] = new_atp;
                modified_tp[tp] = true;
                if first_shift_t.map_or(true, |ft| t < ft) { first_shift_t = Some(t); }
                any_shift = true;
                for &ll in candidate_lines {
                    flows[t][ll] -= sens[ll][b] * actual_delta;
                    flows[tp][ll] += sens[ll][b] * actual_delta;
                }
            }
        }
        if !any_shift { break; }
    }

    // i54 fix: FULL SEQUENTIAL RE-PROJECTION from first_shift_t.
    // Problem (i53): only re-projecting modified_tp left SOC-chain inconsistent at unmodified
    // timesteps after tp (new SOC < old SOC → champion actions may violate battery bounds).
    // Fix: from first_shift_t, clamp ALL timestep actions to current SOC bounds (ensures
    // battery feasibility) + project_polytope at modified_tp (ensures EPS_FLOW line safety).
    if let Some(first_t) = first_shift_t {
        // Advance SOC to first_t using unmodified champion schedule (correct there).
        let mut soc = batteries.iter().map(|b| b.soc_initial_mwh).collect::<Vec<_>>();
        for t in 0..first_t {
            for b in 0..n_b {
                let battery = &batteries[b];
                soc[b] = battery.apply_action_to_soc(schedule[t][b], soc[b])
                    .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
            }
        }
        // From first_t onwards: clamp + (at modified_tp) project_polytope.
        for t in first_t..n_t {
            // Clamp to new SOC bounds (battery feasibility with shifted SOC trajectory).
            for b in 0..n_b {
                let bounds = compute_action_bounds(&batteries[b], soc[b]);
                schedule[t][b] = schedule[t][b].clamp(bounds.0, bounds.1);
            }
            // At modified tp: also run project_polytope for line feasibility.
            if modified_tp[t] {
                let exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
                let base_t: Vec<f64> = (0..n_l)
                    .map(|l| exo.get(l).copied().unwrap_or(0.0))
                    .collect();
                let bounds_t: Vec<(f64, f64)> = (0..n_b)
                    .map(|b| compute_action_bounds(&batteries[b], soc[b]))
                    .collect();
                project_polytope(
                    &mut schedule[t],
                    &bounds_t,
                    sens,
                    &base_t,
                    limits,
                    hp.proj_max_iters,
                    gram,
                    hp.resync_period,
                );
            }
            // Update SOC sequentially.
            for b in 0..n_b {
                let battery = &batteries[b];
                soc[b] = battery.apply_action_to_soc(schedule[t][b], soc[b])
                    .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
            }
        }
    }
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

/// Gram matrix G[l][k] = dot(sens[l], sens[k]): static per nonce (sens is PTDF-derived,
/// constant across all timesteps of a nonce). Used by the incremental projection path to
/// update flows[l] after each halfspace projection in O(n_lines) via G[l][worst_l].
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
/// When `gram` is Some (precomputed G=sens·sensᵀ), uses the incremental flow
/// tracking path: maintains flows[] in O(K_clamped·n_lines + n_lines) per iter
/// instead of O(n_lines·n_b). Iso-geometry: identical Euclidean iterates, same Q.
fn project_polytope(
    action: &mut [f64],
    bounds: &[(f64, f64)],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    limits: &[f64],
    max_iters: usize,
    gram: Option<&[Vec<f64>]>,
    resync_period: usize,
) -> bool {
    let n_lines = sens.len();

    if let Some(gram) = gram {
        // Incremental path: maintain flows[] rather than recomputing from scratch.
        // Init once: O(n_lines × n_b).
        let mut flows: Vec<f64> = (0..n_lines)
            .map(|l| line_flow(&sens[l], action, base_flows[l]))
            .collect();

        // Resync period: recompute flows exactly every N iters to bound float drift.
        // Sweepable via hp.resync_period (default 16 = byte-exact to the former const).
        let resync_period = resync_period.max(1);

        for iter in 0..max_iters {
            // Periodic resync (after iter 0 to skip the initial recompute).
            if iter > 0 && iter % resync_period == 0 {
                for l in 0..n_lines {
                    flows[l] = line_flow(&sens[l], action, base_flows[l]);
                }
            }

            // 1. Box clamp with delta-flow update (only clamped components contribute).
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

            // 2. Find worst violation — O(n_lines) read of cached flows.
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
                return true;
            }

            // 3. Project onto halfspace: action -= worst_sign * mu * sens[worst_l].
            //    norm_sq = G[worst_l][worst_l] (diagonal of gram, O(1) lookup).
            let norm_sq = gram[worst_l][worst_l];
            if norm_sq < 1e-14 {
                return false;
            }
            let mu = worst_excess / norm_sq;
            let row = &sens[worst_l];
            for b in 0..action.len() {
                action[b] -= worst_sign * mu * row[b];
            }
            // Update flows: flows[l] -= worst_sign * mu * G[worst_l][l]
            let gram_row = &gram[worst_l];
            let coeff = worst_sign * mu;
            for l in 0..n_lines {
                flows[l] -= coeff * gram_row[l];
            }
        }

        // Final clamp.
        for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
            if *a < lo { *a = lo; }
            if *a > hi { *a = hi; }
        }
        // Verify using exact line_flow (guards against drift).
        for l in 0..n_lines {
            let f = line_flow(&sens[l], action, base_flows[l]);
            if f.abs() > limits[l] * (1.0 + EPS_FLOW) + 1e-6 {
                return false;
            }
        }
        true
    } else {
        // Original rescan path (iso-baseline).
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
}

/// Project + bisection fallback (scale toward zero if projection fails).
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
    let ok = project_polytope(action, &state.action_bounds, sens, base_flows, limits, hp.proj_max_iters, gram, hp.resync_period);
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
/// delta_cong[b][t]: per-battery congestion correction (dv_agg_cong − dv_agg_nocong) at step t.
/// When cwv_clusters=1, all rows are identical (fleet-average) → byte-identical to i50.
/// When cwv_clusters>1, each battery receives its cluster's delta (i53 dé-dilution).
/// When use_composite_wv=false delta_cong rows are all-zeros → byte-identical to prior iters.
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
        // Per-battery delta: delta_cong[b][t] (cluster-specific when cwv_clusters>1).
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
        // dv_corrected = dv_decoupled + lambda × delta_cong[b][t] (i53 per-cluster, preserves per-battery price signal)
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
    gram: Option<&[Vec<f64>]>,
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

    // Barzilai-Borwein step-size clamps (gated by `use_bb_clamps`, ported from
    
    // (anti-overshoot, cf Hindsight 1b78ed87); the decay floor floors step decay on
    // failure (anti clamp-saturation/oscillation, cf Hindsight d240f498). i11 promotes
    // the two i10 baked consts to tunable HP fields (defaults 1.05 / 0.85) so the
    
    // (loop-invariant single load → no hot-path cost vs the consts).
    let lr_growth_cap = hp.lr_growth_cap;
    let bb_decay_factor = hp.bb_decay_factor;
    // Heavy-ball momentum coefficient. Hardcoded const (NOT serde f64) — a serde
    // f64 field breaks LLVM const-fold and caused +51% regression at t52/i9.
    const MOMENTUM_BETA: f64 = 0.99;
    // i65 cosine annealing endpoint. Const so LLVM eliminates cosine path when use_cosine_beta=false.
    const BETA_END: f64 = 0.7;
    let t_max = hp.grad_outer_iters.saturating_sub(1).max(1) as f64;

    let mut lr = max_power * 0.5;
    // Velocity accumulator reset per call (= per seed). All-zero when
    // `use_momentum=false` → step direction = grad verbatim (iso-baseline).
    let mut velocity = vec![0.0_f64; action.len()];
    // i29 BB adaptive step-size state. Allocated only when use_bb_step=true (empty vecs
    // are zero-cost: no alloc, no iter on the hot path). prev_action_for_bb stores the
    // action BEFORE the step taken in outer iter k-1 (= action_{k-1}); prev_grad_for_bb
    // stores the gradient computed at action_{k-1}. Used to compute Barzilai-Borwein
    // step estimate at iter k: step = ||s|| / ||y|| where s=action_k-action_{k-1},
    // y=grad_k-grad_{k-1}. CTRL: use_bb_step=false → both vecs empty → no hot-path cost.
    let mut prev_action_for_bb: Vec<f64> = if hp.use_bb_step { action.clone() } else { vec![] };
    let mut prev_grad_for_bb: Vec<f64> = if hp.use_bb_step { vec![0.0_f64; action.len()] } else { vec![] };
    let mut pga_iters_done = 0_u64;
    for outer_iter in 0..hp.grad_outer_iters {
        pga_iters_done += 1;
        // β decays 0.99→0.7 via cosine if use_cosine_beta, else constant 0.99.
        let beta = if hp.use_cosine_beta {
            let frac = outer_iter as f64 / t_max;
            BETA_END + (MOMENTUM_BETA - BETA_END) * (1.0 + (std::f64::consts::PI * frac).cos()) * 0.5
        } else {
            MOMENTUM_BETA
        };
        // Save action before this step for BB state update at end of outer iter.
        let saved_action_for_bb: Vec<f64> = if hp.use_bb_step { action.clone() } else { vec![] };
        let grad = analytic_gradient(challenge, state, dps, &action, delta_cong, cwv_lambda);
        let g_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm < 1e-9 {
            break;
        }
        // i28 L6 résidu-gate: per-nonce adaptive exit when gradient is negligible.
        // 0.0 = disabled (sentinel, iso-baseline). Only exits when PGA has converged
        // on THIS nonce; hard nonces keep the full grad_outer_iters budget.
        if hp.outer_grad_tol > 0.0 && g_norm < hp.outer_grad_tol {
            break;
        }

        // Step direction: heavy-ball β·v + g carries search across non-improving
        // zones where pure-gradient lr*=0.4 would collapse (Hindsight d0a700d7).
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
        // i29 BB step-size: at outer_iter > 0, estimate local Lipschitz L_local via
        // BB-2 formula: step = ||s_k|| / ||y_k|| (gradient-secant ratio). Clamped to
        // [max_power*1e-4, max_power*50] for safety. Falls back to lr when degenerate
        // (||y_k|| ≈ 0 = stagnation). When use_bb_step=false: cur_lr=lr (iso-i28).
        let mut cur_lr = if hp.use_bb_step && outer_iter > 0 {
            let dy_sq: f64 = grad.iter().zip(prev_grad_for_bb.iter())
                .map(|(g, p)| (g - p) * (g - p)).sum();
            let ds_sq: f64 = action.iter().zip(prev_action_for_bb.iter())
                .map(|(a, p)| (a - p) * (a - p)).sum();
            // BB-2 step = ||s|| / ||y||. Requires both norms > 0; degenerate cases
            // (no action change: ds≈0, or gradient stagnation: dy≈0) fall back to lr.
            if dy_sq > 1e-15 && ds_sq > 1e-15 {
                (ds_sq / dy_sq).sqrt().clamp(max_power * 1e-4, max_power * 50.0)
            } else {
                lr
            }
        } else {
            lr
        };
        for _ in 0..hp.grad_ls_iters {
            // step_scale uses grad norm (unchanged backtracking); momentum only
            // re-orients direction, not the line-search budget.
            let step_scale = cur_lr / g_norm;
            let mut trial: Vec<f64> = action
                .iter()
                .zip(dir.iter())
                .map(|(a, d)| a + step_scale * d)
                .collect();
            safe_project_to_feasible(challenge, state, &mut trial, sens, base_flows, hp, gram);
            let v = total_step_value(challenge, state, dps, &trial);
            if v > best_value + 1e-9 {
                action = trial.clone();
                best_value = v;
                best_action = trial;
                improved = true;
                lr = if hp.use_bb_clamps {
                    (cur_lr * 1.4).min(prev_lr * lr_growth_cap)
                } else {
                    cur_lr * 1.4
                };
                // Accumulate velocity only on accepted (improving) steps.
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
                (lr * 0.4).max(prev_lr * bb_decay_factor)
            } else {
                lr * 0.4
            };
            if lr < max_power * 1e-4 {
                break;
            }
        }
        // Update BB state: store the action and gradient FROM THIS OUTER ITER for use
        // in the BB step-size estimate of outer_iter+1. Enabled only when use_bb_step=true.
        if hp.use_bb_step {
            prev_action_for_bb = saved_action_for_bb;
            prev_grad_for_bb = grad;
        }
    }
    PROF_PGA_OUTER_TOTAL.fetch_add(pga_iters_done, Ordering::Relaxed);
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
) -> Vec<f64> {
    let mut best_action = vec![0.0_f64; challenge.num_batteries];
    let mut best_value = total_step_value(challenge, state, dps, &best_action);

    for seed in seeds {
        let (a, v) = projected_gradient_ascent(
            challenge, state, dps, sens, base_flows, seed, hp, delta_cong, cwv_lambda, gram,
        );
        if v > best_value && is_flow_feasible(challenge, state, &a) {
            best_value = v;
            best_action = a;
        }
    }
    best_action
}

/// Pairwise cross-battery coordinate-exchange polish (P39/P40/P41, ported from

/// equal-and-opposite shifts ±α·span that the independent gradient / per-battery
/// coordinate polish cannot see, keeping only feasible improving swaps. Sequential
/// (no threading); first-improvement restart; capped at `joint_pair_budget` pairs.
/// When `hp.use_pair_priority` is true, pairs are sorted by |price_i - price_j| ×

/// "most critical scenarios first"), so the top-K pairs by expected gain are visited
/// first. When false: sequential (i,j) order, byte-identical to v1816.
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

    // Build ordered pair list: priority-sorted or sequential.
    let ordered_pairs: Vec<(usize, usize)> = if hp.use_pair_priority {
        let mut scored: Vec<(f64, usize, usize)> = Vec::with_capacity(num_b * (num_b - 1) / 2);
        for i in 0..num_b {
            let price_i = state.rt_prices[challenge.batteries[i].node];
            let (lo_i, hi_i) = state.action_bounds[i];
            let span_i = hi_i - lo_i;
            for j in (i + 1)..num_b {
                let price_j = state.rt_prices[challenge.batteries[j].node];
                let (lo_j, hi_j) = state.action_bounds[j];
                let span_j = hi_j - lo_j;
                let score = (price_i - price_j).abs() * span_i.min(span_j);
                scored.push((score, i, j));
            }
        }
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(pair_budget);
        scored.into_iter().map(|(_, i, j)| (i, j)).collect()
    } else {
        let mut pairs = Vec::with_capacity(pair_budget);
        let mut count = 0usize;
        'fill: for i in 0..num_b {
            for j in (i + 1)..num_b {
                if count >= pair_budget {
                    break 'fill;
                }
                pairs.push((i, j));
                count += 1;
            }
        }
        pairs
    };

    let mut improved = true;
    while improved {
        improved = false;
        for &(i, j) in &ordered_pairs {
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
                break;
            }
        }
    }
}

/// P8 — 3-battery triplet exchange polish (ported from t51/i67 KEPT Q=2,811,442).
///
/// (i,j,k): tries ±α on i (gainer), ∓α on j (equal-and-opposite), +γ on k
/// (independent compensation). First-improvement restart; budget-capped per pass.
/// Uses linear flow update for feasibility (same as joint_pair_polish) — no
/// is_flow_feasible call in the inner loop (keeps cost proportional to num_lines).
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
    // Screening: sort batteries by |action| descending, form triplets only in top-K.
    // C(15,3)=455 >= budget=150 — concentrates budget on most active batteries (post pair-polish).
    let top_k = hp.joint_triplet_top_k.max(3).min(num_b);
    let mut batt_scores: Vec<(f64, usize)> = (0..num_b)
        .map(|b| (actions[b].abs(), b))
        .collect();
    batt_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let active: Vec<usize> = batt_scores.iter().take(top_k).map(|&(_, b)| b).collect();

    // Single-pass greedy: iterate at most `joint_triplet_budget` triplets sequentially,
    // applying improvements immediately without restarting.
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
    // i35 L7: compute battery iteration order once (sens is static per-nonce).
    // mode 0=natural (CTRL iso-i31), 1=high-PTDF-danger-first, 2=low-danger-first, 3=sum-PTDF-desc.
    let battery_order: Vec<usize> = if hp.disagg_order_mode == 0 {
        (0..challenge.num_batteries).collect()
    } else {
        let num_l = challenge.network.num_lines;
        let mut scores: Vec<(usize, f64)> = (0..challenge.num_batteries)
            .map(|b| {
                let score = match hp.disagg_order_mode {
                    1 => (0..num_l)
                        .map(|l| {
                            let limit = challenge.network.flow_limits[l].abs().max(1e-9);
                            sens[l][b].abs() / limit
                        })
                        .fold(0.0_f64, f64::max),
                    2 => -(0..num_l)
                        .map(|l| {
                            let limit = challenge.network.flow_limits[l].abs().max(1e-9);
                            sens[l][b].abs() / limit
                        })
                        .fold(0.0_f64, f64::max),
                    _ => (0..num_l).map(|l| sens[l][b].abs()).sum::<f64>(),
                };
                (b, score)
            })
            .collect();
        scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().map(|(b, _)| *b).collect()
    };
    for _ in 0..hp.coord_polish_passes {
        let mut improved = false;
        for &b in battery_order.iter() {
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

fn policy(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    hp: &Hyperparameters,
    delta_cong: &[Vec<f64>],
    gram: Option<&[Vec<f64>]>,
) -> Result<Vec<f64>> {
    PROF_POLICY_CALLS.fetch_add(1, Ordering::Relaxed);
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

    clamp_to_bounds(&mut target, &state.action_bounds);

    // Independent per-battery DP-preferred seed.
    // use_dp_seed=false (P29): skip construction entirely when the seed is dropped.
    let dp_seed: Vec<f64> = if hp.use_dp_seed {
        (0..challenge.num_batteries)
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
            .collect()
    } else {
        vec![]
    };

    // Baseline flows depend on this step's exogenous injections.
    let zero = vec![0.0_f64; challenge.num_batteries];
    let base_flows = compute_flows(challenge, state, &zero);

    // Joint projected gradient ascent over multiple seeds.
    // P27 (use_zero_seed) / P29 (use_dp_seed): drop seeds to cut PGA trajectories.
    // P36/P38 port (use_arb_seed): substitute dp_seed with greedy price-arbitrage seed
    // (congestion-agnostic). arb_pct=0 → mean threshold (P36). arb_pct>0 → percentile
    // threshold (P38, e.g. 75). TIME-NEUTRAL: keeps num_seeds=3 (dp_seed substituted).
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
                
                if price > arb_threshold { hi * hp.arb_scale } else { lo * hp.arb_scale }
            })
            .collect();
        vec![target, arb_seed, zero.clone()]
    } else {
        match (hp.use_zero_seed, hp.use_dp_seed) {
            (true, true) => vec![target, dp_seed, zero.clone()],
            (true, false) => vec![target, zero.clone()],
            (false, true) => vec![target, dp_seed],
            (false, false) => vec![target],
        }
    };

    // i87 (combine t51 P40/P42 → t52): append ONE maximally-distant arbitrage-diversity seed.
    // t52 champion runs single-start [target] (use_dp_seed=false, use_zero_seed=false) → no
    // basin diversity. An extreme binary arb seed (charge/discharge at action bounds by price
    // percentile) starts a PGA trajectory in a DISTINCT non-convex basin (Hindsight 813c100d
    // "arb_seed starts PGA in a distinct basin", be379b88 "diversity from direction/polarity,
    // extreme {lo,hi} > intermediate", 4906d1ee "std_arb_third distinct 3rd basin → higher Q").
    // arb_diversity_inverse=true = anti-correlated (charge when expensive), maximally distant
    // from the DP-like target (t51 P40 = biggest arb win). Off (default) = byte-exact sentinel.
    if hp.use_arb_diversity_seed {
        let prices = &state.rt_prices;
        let n = prices.len();
        let pct = if hp.arb_pct == 0 { 75 } else { hp.arb_pct as usize };
        let thr = if n == 0 {
            0.0
        } else {
            let mut sorted = prices.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            percentile(&sorted, pct, 100)
        };
        // Build one extreme binary arb seed at the given polarity. `inverse=true` =
        // anti-correlated (charge when expensive); `inverse=false` = standard economic arb
        // (discharge when expensive). Extreme {lo,hi} > intermediate (Hindsight be379b88).
        let build_arb_seed = |inverse: bool| -> Vec<f64> {
            (0..challenge.num_batteries)
                .map(|b| {
                    let battery = &challenge.batteries[b];
                    let price = prices[battery.node];
                    let (lo, hi) = state.action_bounds[b];
                    // hi = discharge (sell when expensive). standard: expensive→discharge.
                    // inverse: expensive→charge (anti-correlated, max distance from target).
                    let discharge = if inverse { price <= thr } else { price > thr };
                    if discharge { hi } else { lo }
                })
                .collect()
        };
        seeds.push(build_arb_seed(hp.arb_diversity_inverse));
        // i89 (P42 port): add the OPPOSITE-polarity std/inverse-arb seed as a distinct 3rd basin.
        // Champion i87 = [target, inverse-arb]; this yields [target, inverse-arb, std-arb] — 3
        // maximally-distant PTDF facets. t51 P42 proved +4,809 Q by this exact 3rd-basin add.
        if hp.use_arb_diversity_pair {
            seeds.push(build_arb_seed(!hp.arb_diversity_inverse));
        }
    }
    let cwv_lambda = if hp.use_composite_wv { hp.cwv_lambda } else { 0.0 };
    let mut result = joint_optimize_step(
        challenge, state, dps, sens, &base_flows, seeds, hp, delta_cong, cwv_lambda, gram,
    );
    result = coordinate_polish_step(challenge, state, dps, sens, result, hp);

    // P39/P40/P41: pairwise cross-battery exchange the gradient/coord polish miss.
    if hp.use_joint_pair_polish {
        let pre_polish = result.clone();
        joint_pair_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_polish;
        }
    }

    // P8: 3-battery triplet exchange polish (ported from t51/i67, budget-capped greedy).
    if hp.use_joint_triplet_polish {
        let pre_triplet = result.clone();
        joint_triplet_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_triplet;
        }
    }

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

    let sens = build_sensitivity(challenge);
    let gram_storage: Option<Vec<Vec<f64>>> = if hp.use_gram_incremental_proj {
        Some(build_gram(&sens))
    } else {
        None
    };
    let gram: Option<&[Vec<f64>]> = gram_storage.as_deref();

    
    // track_t52.rs anticipate_lmp mechanism, last piece of SOTA config v1686).
    // When off (default), expected_premiums are zeros → byte-equivalent no-op.
    let n_lines = challenge.network.flow_limits.len();

    // i76 congestion knees (per battery): u_sat = limit_{l*}/|sens_{l*,b}| where l* is the
    // line battery b most influences (max |sens|, among lines with a real limit). This is the
    // own-action magnitude that saturates that line; the DP action grid is densified around it.
    // congestion_grid_alpha==0 ⇒ all None ⇒ adaptive_action_grid stays byte-identical to i71.
    let cong_knees: Vec<Option<f64>> = if hp.congestion_grid_alpha > 0.0 && n_lines > 0 {
        (0..challenge.num_batteries)
            .map(|b| {
                let mut best_s = 0.0_f64;
                let mut best_l = usize::MAX;
                for l in 0..n_lines {
                    let lim = challenge.network.flow_limits[l];
                    if lim > 1e-6 {
                        let s = sens[l][b].abs();
                        if s > best_s {
                            best_s = s;
                            best_l = l;
                        }
                    }
                }
                if best_l != usize::MAX && best_s > 1e-9 {
                    Some(challenge.network.flow_limits[best_l] / best_s)
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![None; challenge.num_batteries]
    };

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
                    // i36 RÈGLE D'OR: proba^gamma concentrates premium on near-binding lines.
                    // gamma=1.0 (default) = CTRL iso-i31. gamma>1.0 = convex reshape.
                    let shaped = if hp.premium_shape_gamma == 1.0 { proba } else { proba.powf(hp.premium_shape_gamma) };
                    let premium = base_premium * shaped;
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
                cong_knees[b],
                &hp,
            )
        })
        .collect();

    // i64 OCO constraint tracking (port from t53/i65 lines 2241-2325, Huang 78025f66).
    // TIME-CLAWBACK: active-set candidate_lines + selective DP rebuild (touched only).
    // Discriminant vs static P8: μ tracks REAL dispatch flow violations (not exo-only),
    // capturing dispatch-induced congestion missed by the one-shot exo premium.
    // When use_ptdf_ct=false → this block is skipped, dps unchanged = iso-baseline (byte-exact i63).
    let dps = if hp.use_ptdf_ct && n_lines > 0 {
        let eta = hp.ct_step_eta;
        let ct_scale = 1.0 - hp.ct_ref_kappa;
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let limits = &challenge.network.flow_limits;

        // Active-set: only lines where batteries have meaningful sensitivity.
        const ACTIVE_SENS_THRESH: f64 = 1e-4;
        let candidate_lines: Vec<usize> = (0..n_lines)
            .filter(|&l| limits[l] > 1e-6 && sens[l].iter().any(|&s| s.abs() > ACTIVE_SENS_THRESH))
            .collect();

        // Step 1: fast greedy forward simulation restricted to candidate lines.
        let flows_all = ct_simulate_flows(challenge, &dps, &sens, &candidate_lines, &hp);

        // Step 2: OCO multipliers for candidate lines only.
        let beta = hp.ct_cap_sens_beta;
        let vq_v = hp.ct_vq_v;
        let vq_passes = hp.ct_vq_passes.max(1);
        let sens_agg: Vec<f64> = (0..n_lines)
            .map(|l| sens[l].iter().map(|&s| s.abs()).sum::<f64>())
            .collect();
        let mut mu_oco = vec![vec![0.0_f64; n_lines]; n_t];
        if vq_v > 0.0 {
            // i45 virtual-queue drift-plus-penalty (Zhou-Wang-Li 2024, recall 00cc7807/c55e83c8).
            // Q_l ← max(0, Q_l + viol_l[t]) accumulates violations forward across timesteps.
            // μ_l[t] = min(η·Q_l/V, limit·0.5) — integral multiplier bounded by V.
            // vq_passes>1: persistent q_virt across scans → stronger wind-up (smaller effective V).
            // CTRL path: vq_v=0.0 → else branch below → byte-EXACT i37.
            let mut q_virt = vec![0.0_f64; n_lines];
            for _pass in 0..vq_passes {
                for t in 0..n_t {
                    for &l in &candidate_lines {
                        let limit = limits[l];
                        let viol = flows_all[t][l].abs() - limit;
                        q_virt[l] = (q_virt[l] + viol).max(0.0);
                        mu_oco[t][l] = ((eta * q_virt[l]) / vq_v).min(limit * 0.5);
                    }
                }
            }
        } else {
            // Proportional one-shot (beta=0.0 → byte-exact CTRL=i37).
            // cap_l = (limit*0.5)/(1 + beta*sens_agg[l]): beta=0 → limit*0.5.
            for t in 0..n_t {
                for &l in &candidate_lines {
                    let limit = limits[l];
                    let viol = flows_all[t][l].abs() - limit;
                    if viol > 0.0 {
                        let cap_l = (limit * 0.5) / (1.0 + beta * sens_agg[l]);
                        mu_oco[t][l] = (eta * viol).min(cap_l);
                    }
                }
            }
        }

        // Step 3: adjust premiums + track which batteries are actually touched.
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

        
        // Untouched batteries → reuse existing DP (Q-EXACT). Only rebuild touched ones.
        // i71 OCO-CT full-res: keep hp_oco = hp (dp_soc_levels=65, dp_action_levels=9).
        // Half-res (32×5) produced inexact flux violations → biased ep_ct (cf memory
        // f32d19ff: low-res DP produces incorrect violations). Full-res → faithful
        // violations → more accurate congestion correction.
        let hp_oco = hp;
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
                        cong_knees[b],
                        &hp_oco,
                    )
                }
            })
            .collect()
    } else {
        dps
    };

    // Delta-congestion aggregate correction (L6b-bis + i53 per-cluster dé-dilution).
    // delta_cong[b][t] = per-battery congestion correction assigned from battery b's cluster.
    // cwv_clusters=1: fleet-average path (byte-identical to i50). cwv_clusters>1: K-cluster path.
    // When use_composite_wv=false → all zeros, byte-identical to prior iters.
    let delta_cong: Vec<Vec<f64>> = if hp.use_composite_wv {
        let n_b = challenge.num_batteries;
        let n_t = challenge.num_steps;
        let k = hp.cwv_clusters.max(1).min(n_b.max(1));

        if k == 1 {
            // Fleet-average path — byte-identical to i50 baseline.
            let total_cap: f64 = challenge.batteries.iter().map(|b| b.capacity_mwh).sum::<f64>().max(1.0);
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
            let da_with_cong: Vec<f64> = fleet_da.iter().zip(fleet_premium.iter())
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
            let e_mid: f64 = challenge.batteries.iter()
                .map(|b| (b.soc_min_mwh + b.soc_max_mwh) * 0.5)
                .sum();
            let fleet_delta: Vec<f64> = (0..n_t)
                .map(|t| aggregate_dv_dsoc(&agg_dp_cong, t, e_mid)
                        - aggregate_dv_dsoc(&agg_dp_nocong, t, e_mid))
                .collect();
            // All batteries share the same fleet delta (parity with i50 uniform application).
            vec![fleet_delta; n_b]
        } else {
            // Per-cluster path (i53 dé-dilution): K quantile clusters by avg congestion exposure.
            // Battery b's average expected_premium across all timesteps = its exposure score.
            let exposure: Vec<f64> = (0..n_b)
                .map(|b| expected_premiums.iter().map(|pt| pt[b]).sum::<f64>() / n_t.max(1) as f64)
                .collect();

            // Sort battery indices by exposure → assign cluster IDs via quantile split.
            let mut sorted_idx: Vec<usize> = (0..n_b).collect();
            sorted_idx.sort_by(|&a, &bi| exposure[a].partial_cmp(&exposure[bi])
                .unwrap_or(std::cmp::Ordering::Equal));
            let mut cluster_id = vec![0usize; n_b];
            for (rank, &b_idx) in sorted_idx.iter().enumerate() {
                cluster_id[b_idx] = (rank * k) / n_b;
            }

            // For each cluster: collect batteries, compute cluster DA + premium, build DP pair.
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
                let cluster_cap: f64 = cluster_bats.iter()
                    .map(|b| b.capacity_mwh).sum::<f64>().max(1.0);

                // Cluster capacity-weighted DA price.
                let cluster_da: Vec<f64> = (0..n_t)
                    .map(|t| {
                        cluster_bats.iter().map(|b| {
                            b.capacity_mwh * challenge.market.day_ahead_prices[t][b.node]
                        }).sum::<f64>() / cluster_cap
                    })
                    .collect();

                // Cluster capacity-weighted congestion premium.
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

                let da_with_cong: Vec<f64> = cluster_da.iter().zip(cluster_premium.iter())
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
                let e_mid_cluster: f64 = cluster_bats.iter()
                    .map(|b| (b.soc_min_mwh + b.soc_max_mwh) * 0.5)
                    .sum();
                let cluster_delta: Vec<f64> = (0..n_t)
                    .map(|t| aggregate_dv_dsoc(&agg_dp_cong, t, e_mid_cluster)
                            - aggregate_dv_dsoc(&agg_dp_nocong, t, e_mid_cluster))
                    .collect();
                cluster_deltas.push(cluster_delta);
            }

            // Build per-battery delta: battery b gets its cluster's delta vector.
            (0..n_b).map(|b| cluster_deltas[cluster_id[b]].clone()).collect()
        }
    } else {
        vec![vec![0.0_f64; challenge.num_steps]; challenge.num_batteries]
    };

    // i51 temporal-flow: precompute full-horizon schedule (use_temporal_flow=true path).
    // CTRL: use_temporal_flow=false → vec![] → grid_optimize uses existing PGA policy (iso-i46 byte-exact).
    // Active-set identical to OCO-CT candidate_lines (ACTIVE_SENS_THRESH=1e-4, same filter).
    let precomputed_schedule: Vec<Vec<f64>> = if hp.use_temporal_flow && n_lines > 0 {
        const ACTIVE_SENS_THRESH_TF: f64 = 1e-4;
        let limits_ref = &challenge.network.flow_limits;
        let cl_tf: Vec<usize> = (0..n_lines)
            .filter(|&l| limits_ref[l] > 1e-6 && sens[l].iter().any(|&s| s.abs() > ACTIVE_SENS_THRESH_TF))
            .collect();
        if cl_tf.is_empty() {
            vec![]
        } else {
            solve_temporal_flow(challenge, &dps, &sens, &cl_tf, &hp)
        }
    } else {
        vec![]
    };

    // Initial feasible solution: all zeros.
    let zero_solution = Solution {
        schedule: vec![vec![0.0; challenge.num_batteries]; challenge.num_steps],
    };
    save_solution(&zero_solution)?;

    // Reset profiling counters (called once per nonce).
    PROF_PGA_OUTER_TOTAL.store(0, Ordering::Relaxed);
    PROF_POLICY_CALLS.store(0, Ordering::Relaxed);

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
    // [PROF] log fuel budget at start of policy loop
    eprintln!("[t52-prof] nonce_start: fuel_available={} fuel_floor={} max_spend={}", available, fuel_floor, max_spend);
    let mut solution = challenge.grid_optimize(&|c, s| {
        if fuel_remaining() <= fuel_floor {
            return Ok(vec![0.0; c.num_batteries]);
        }
        
        // Re-project to actual state bounds to guard against FP SOC drift across nonces.
        if !precomputed_schedule.is_empty() {
            let t = s.time_step;
            if t < precomputed_schedule.len() {
                let mut action = precomputed_schedule[t].clone();
                let n_l_inner = c.network.flow_limits.len();
                let exo_inner = c.network.compute_flows(&c.exogenous_injections[t]);
                let base_inner: Vec<f64> = (0..n_l_inner)
                    .map(|l| exo_inner.get(l).copied().unwrap_or(0.0))
                    .collect();
                project_polytope(&mut action, &s.action_bounds, &sens, &base_inner,
                                 &c.network.flow_limits, hp.proj_max_iters, gram, hp.resync_period);
                return Ok(action);
            }
        }
        policy(c, s, &dps, &sens, &hp, &delta_cong, gram)
    })?;
    // [PROF] log fuel breakdown after policy loop
    let fuel_end = fuel_remaining();
    let pga_total = PROF_PGA_OUTER_TOTAL.load(Ordering::Relaxed);
    let policy_calls = PROF_POLICY_CALLS.load(Ordering::Relaxed);
    let avg_pga = if policy_calls > 0 { pga_total / policy_calls } else { 0 };
    eprintln!("[t52-prof] nonce_end: fuel_remaining={} policy_spent={} policy_calls={} pga_outer_total={} avg_pga_per_step={}",
        fuel_end, available.saturating_sub(fuel_end), policy_calls, pga_total, avg_pga);
    // i53 AUGMENT-form: post-pass temporal reschedule on champion schedule.
    // CTRL: use_temporal_flow_postpass=false → byte-exact (no-op, iso-i46).
    // VAR: apply step-2 shifts to solution.schedule (seed=champion, guard score>0).
    if hp.use_temporal_flow_postpass && n_lines > 0 {
        const ACTIVE_SENS_THRESH_PP: f64 = 1e-4;
        let cl_pp: Vec<usize> = (0..n_lines)
            .filter(|&l| challenge.network.flow_limits[l] > 1e-6
                && sens[l].iter().any(|&s| s.abs() > ACTIVE_SENS_THRESH_PP))
            .collect();
        if !cl_pp.is_empty() {
            apply_temporal_postpass(challenge, &mut solution.schedule, &sens, &cl_pp, &hp, gram);
        }
    }
    save_solution(&solution)?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
