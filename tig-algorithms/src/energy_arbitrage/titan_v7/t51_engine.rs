
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

/// Numerical tolerance for the per-battery tiny LP simplex pivots.
const LP_EPS: f64 = 1e-9;

/// Cut from 25→5: warm-start λ from prior step converges in ≤5 iters (Hindsight b28a07a4).
const RH_KKT_ITERS: usize = 5;
/// Initial step size for PTDF dual subgradient (diminishing: alpha0/sqrt(k+1)).
const RH_KKT_ALPHA0: f64 = 0.5;
/// Cap on dual step size to prevent overshoot when warm-start λ is near-optimal
/// and constraints shift slowly between time steps (Hindsight ba42ac1e).
const RH_KKT_ALPHA_MAX: f64 = 0.3;
/// Filter for PTDF lines included in the dual loop: only lines whose
/// |base_flow|/limit > this threshold are dualized (reduces dual dimension).
const RH_BINDING_RATIO: f64 = 0.7;
/// Tiny per-battery simplex dimensions (4 primal vars, 8 constraints).
const N_4VAR: usize = 4;
const M_4VAR: usize = 8;
const NV_4VAR: usize = N_4VAR + M_4VAR; // = 12 (primal + slacks)
const NC_4VAR: usize = NV_4VAR + 1;     // = 13 (+ RHS column)

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
    /// projected-gradient ascent. Default = false → hot-path identical to v1728
    /// (LLVM dead-code-eliminates the `false` branch; no extra arithmetic on the
    /// hot path). Ported from t52 i10 P5 (KEPT WIN +0.99 % Q / −7.4 % t).
    #[serde(default)]
    pub use_bb_clamps: bool,
    /// Enable Polyak heavy-ball momentum on the projected-gradient ascent step
    /// direction (`dir = MOMENTUM_BETA*velocity + grad`). Default `false` keeps
    /// the pure gradient path (iso-baseline). The velocity is reset per seed so
    /// the first step is identical to the no-momentum path. Ported from t53 i10
    /// P6 (KEPT +0.837 % Q on capstone; target: +Q margin on t51 multiday).
    #[serde(default)]
    pub use_momentum: bool,
    /// Anticipate network congestion by adding PTDF-weighted premiums to DP prices.
    
    /// Default false → no-op path byte-equivalent (expected_premiums are all zeros).
    #[serde(default)]
    pub anticipate_lmp: bool,
    /// Congestion ratio threshold: only lines with |f_exo|/limit > threshold receive
    
    pub lmp_threshold: f64,
    /// Scale factor applied to the base premium (20 $/MWh × scale). Default 1.0.
    pub lmp_premium_scale: f64,
    /// Enable joint pairwise battery-action exchange polish after coordinate polish
    /// (cross-poll P7 from t52/i13-i14). Alphas {±0.5,±0.25}×span, budget-reset-per-
    /// pass, rollback if infeasible. Default `false` = iso-baseline (no-op branch).
    #[serde(default)]
    pub use_joint_pair_polish: bool,
    /// Max distinct battery pairs probed per `joint_pair_polish` pass (reset each pass).
    /// Only read inside the `use_joint_pair_polish` branch.
    /// Default 780 = C(40,2): covers entire pair space for t51 N=40.
    #[serde(default = "default_joint_pair_budget")]
    pub joint_pair_budget: usize,
    /// Enable joint triplet battery-action exchange polish after pairwise polish (L2).
    /// Triplet move: ±α on i (gainer), ∓α on j (equal-and-opposite), +γ on k
    /// (independent compensation). Escapes the pairwise fixed point proven in i30
    /// (C(40,2)=780, 2nd pass Δ=0 EXACT). Default `false` = no-op (iso-baseline).
    #[serde(default)]
    pub use_joint_triplet_polish: bool,
    /// Max distinct triplets probed in a single greedy pass of `joint_triplet_polish`.
    /// Single-pass (no restart) keeps overhead proportional to budget. Default 150 ≈ +0.5-0.7s.
    #[serde(default = "default_joint_triplet_budget")]
    pub joint_triplet_budget: usize,
    
    /// C(top_k, 3) triplets explored; default 15 → C(15,3)=455 ≥ budget.
    #[serde(default = "default_joint_triplet_top_k")]
    pub joint_triplet_top_k: usize,
    /// L6: ADMM consensus polish post-jtp. Relaxes PTDF coupling in per-battery u-update,
    
    /// Hindsight 764cc45b). Default false = no-op (iso-baseline, LLVM dead-eliminates branch).
    #[serde(default)]
    pub use_admm_polish: bool,
    /// L6: variable-depth ejection-chain polish post-jtp (Lin-Kernighan style). Unlike the
    /// simultaneous/single-shot moves of jpp/jtp/quad/admm (all of which regress ~-1,100Q on
    /// t51 because any *immediately-improving* perturbation of the jtp fixed point is exhausted),
    /// the chain FORCES a worsening "ejection" move on a seed battery, then greedily absorbs the
    /// imbalance through a sequence of downstream batteries, committing only the best
    /// strictly-improving FEASIBLE prefix. The one post-jtp mechanism that can cross a local
    /// barrier (accepts intermediate loss). Default false = no-op (LLVM dead-eliminates branch).
    #[serde(default)]
    pub use_ejection_chain: bool,
    /// L7: SOC-contingent value correction (SCVC) — after the backward DP pass, applies a
    /// linear-in-SOC slope correction to dp.values[t][s]:
    ///   values[t][s] += scvc_alpha * (soc_s - soc_ref[t]) * marge_b
    /// where soc_ref[t] = greedy forward rollout trajectory (one-shot, O(T×levels)),
    /// marge_b = mean |DA price| (water-value proxy). Corrects slope miscalibration of the
    /// decoupled-per-battery surrogate (root cause of i37-i42 plateau: surrogat diverges
    /// from true multiday Q when SOC deviates from greedy ref). Init-time only, no re-solve.
    /// Distinct from i32 dead (quadratic penalty on total_step_value at runtime; this corrects
    
    /// Default false = iso-baseline no-op.
    #[serde(default)]
    pub use_scvc: bool,
    /// SCVC correction strength. Only read when use_scvc=true. A1=0.5, A2=1.0.
    /// Safe serde f64: runs once at init, not on PGA hot path (cf MOMENTUM_BETA note).
    #[serde(default = "default_scvc_alpha")]
    pub scvc_alpha: f64,
    /// L7b: K=2 rolling-horizon LP seed. At each step t, builds a 2-step joint LP
    /// (step t: RT prices; step t+1: DA prices) with PTDF coupling and SOC dynamics.
    /// Returns the LP-optimal actions at step t as an additional seed for joint_optimize_step.
    /// Sidesteps the per-battery decoupled surrogate root cause (i37-i42+i43 plateau).
    /// Port of LP simplex from t50::lp_solve_with_budget. RH_MAX_PIVOTS=60 hardcoded const.
    /// Default false = iso-baseline no-op (LLVM dead-eliminates branch).
    #[serde(default)]
    pub use_rolling_horizon: bool,
    /// Cadence for rolling-horizon activation: RH runs only on steps where
    /// time_step % rh_stride == 0. Default=1 (every step, backward-compatible).
    /// STRIDE=2 → 48/96 steps → overhead ~1.1s; STRIDE=3 → 32/96 → ~0.75s.
    /// On skipped steps lam_warm is NOT updated but stays warm for next active step.
    #[serde(default = "default_rh_stride")]
    pub rh_stride: usize,
    /// L8: SoC reference-tracking gradient penalty (Huang 2024, prediction-free two-stage).
    /// At each PGA gradient step, adds lambda*(next_soc - soc_ref_t1)*|dsoc_du| where
    
    /// (computed once at step 0 from DA prices P25/P75 thresholds, stored globally).
    /// lambda=0.0 (default) = iso-baseline no-op.
    #[serde(default)]
    pub soc_ref_lambda: f64,
    /// Cadence for SoC-reference recompute, DECOUPLED from rh_stride (LP reseed).
    /// soc_ref is recomputed at steps where t % soc_ref_dyn_stride == 0.
    
    /// stride=1: recompute every step (max adaptivity). stride=6: less frequent.
    #[serde(default = "default_soc_ref_dyn_stride")]
    pub soc_ref_dyn_stride: usize,
    /// L6+2: cosine annealing of heavy-ball β over the outer PGA iterations.
    /// β decays from MOMENTUM_BETA (0.99) at iter 0 to pga_beta_end at iter go-1:
    ///   β(t) = pga_beta_end + (0.99 - pga_beta_end) × (1 + cos(π·t/(T-1))) / 2
    /// High β early = wide exploration across flat zones; low β late = fine convergence.
    /// Directly addresses the momentum-overshoot hypothesis: β=0.99 constant forces the
    /// velocity accumulator to remain large at iter 70-79, preventing tight local polish.
    /// Default false = constant MOMENTUM_BETA=0.99 (iso-baseline, LLVM const-folds false branch).
    #[serde(default)]
    pub use_cosine_beta: bool,
    /// Cosine annealing endpoint β_end. Only read when use_cosine_beta=true.
    
    /// Lower = more gradient-pure steps in the final iters (finer polish).
    /// Safe to expose as HP: BETA_END is only used in the use_cosine_beta=true branch,
    /// no LLVM const-fold risk on the use_cosine_beta=false hot-path.
    #[serde(default = "default_pga_beta_end")]
    pub pga_beta_end: f64,
    /// L8-bis (i90): REPLACE the projected-gradient ascent inner solver by an
    /// augmented-Lagrangian ADMM operator-splitting QP solver (Srikanthan 2024
    /// `cc38eba3`/`5a1d4bcf`; reformulation `339fbf44`/`99718796`). Motivation: i89
    /// proved that a *non-projected* QP solve recovers the Q the saturated 6/6
    /// gradient-projected update rule cannot (root-cause i88 annulled), but PDHCG
    /// pays an inner-CG loop (cg_iters×grad/outer → ×3.2–5.5 the 11.5s budget,
    /// `64913a57`). ADMM gets the SAME split (smooth non-projected x-update ↔
    /// feasibility-by-projection z-update) at ~1 analytical solve/iter
    /// (`40f09769`/`a9520e21`: matrix-vector + elementwise, no inner solver) so it
    /// can hold the budget. CORRECTED form (`3c3b04c4`): we apply the *projected* z,
    /// never the unbounded x — the historical t51 ADMM failures (`8320cca3`/
    /// `c5545026`: applied unprojected x → out-of-bound actions → invalid nonces)
    /// came from applying x. Default false → projected_gradient_ascent hot path is
    /// byte-identical to i77 (LLVM dead-code-eliminates the `true` branch).
    #[serde(default)]
    pub use_admm_solver: bool,
    /// ADMM penalty parameter ρ (constant → cheap iters + value-function-decreasing
    /// convergence guarantee, `4317e23f`/`86f29e25`). Only read when
    /// use_admm_solver=true. Default 0.45 = best consensus without over-constraining
    /// batteries on t51 multiday (`6b513d2f`/`ee2d5282`).
    #[serde(default = "default_admm_rho")]
    pub admm_rho: f64,
    /// ADMM iteration count (each iter is cheap: 1 grad + 1 projection + 1 dual).
    /// Only read when use_admm_solver=true. Default 9 (`6b513d2f`).
    #[serde(default = "default_admm_iters")]
    pub admm_iters: usize,
    
    /// joint PGA multi-start. Replaces dp_seed with arb_seed: for each battery,
    /// discharge at max if rt_price > mean(rt_prices), else charge at max. Ignores
    /// PTDF coupling entirely. Motivation: gap-dual proven ≈0 (i19 bit-exact NULL);
    /// the only Q residual lives in the non-convex basin. A structurally distinct
    /// seed (≠ target/dp/zero) may reach a Q-superior basin. TIME-NEUTRAL: num_seeds
    /// stays 3 (dp_seed substituted, not added). Default false = iso-baseline, dp_seed
    /// unchanged, byte-exact CTRL.
    #[serde(default)]
    pub use_arb_seed: bool,
    
    /// 0 = use arithmetic mean (byte-exact CTRL vs i21). Non-zero = use this percentile
    /// of rt_prices as the discharge threshold (e.g. 75 → discharge when price > P75,
    /// charge otherwise). P75 is more selective than mean on right-skewed spot price
    /// distributions → seeds a distinct PGA starting basin.
    #[serde(default)]
    pub arb_pct: u64,
    
    /// (discharge hi when price > threshold). When true: charge hi when price > threshold
    /// (anti-correlated = maximally distant seed from arb_seed). TIME-NEUTRAL: same
    /// compute path, just flips the hi/lo branch. By proven mechanism (756420a7/8e4984d3):
    /// distance > precision for multi-start convergence → opposition seed may reach a
    /// Q-superior non-convex basin. Sentinel: false → 0-diff vs i23 byte-exact.
    #[serde(default)]
    pub arb_inverse: bool,
    
    /// (discharge when price > P75, charge otherwise) when use_arb_seed=true.
    /// Seeds become [target, anti-corr-arb(P75), std-arb(P75)] instead of
    /// [target, anti-corr-arb(P75), zero]. num_seeds=3 CONSTANT (replace, not add).
    /// TIME-NEUTRAL: O(num_batteries)/step. Sentinel: false → 0-diff vs i24 byte-exact.
    #[serde(default)]
    pub use_std_arb_third: bool,
    
    /// use_std_arb_third=true). 0 = fall back to arb_pct (byte-exact CTRL vs i25).
    /// Allows the 3rd seed (standard arb, discharge-high) to use a distinct threshold
    /// from the 2nd seed (anti-corr arb, arb_inverse=true). Hypothesis: optimal
    /// discharge threshold for std_arb may differ from anti-corr threshold (P75 was
    /// tuned for anti-corr polarity). TIME-NEUTRAL: single integer compare per step.
    #[serde(default)]
    pub arb_pct_third: u64,
    /// L9 i28 VarA: replace slot-3 (std-arb) with OBL opposition-of-target seed.
    /// opp_b = lo_b + hi_b − target_b (mirror of target through action midpoint).
    /// Structurally distinct from all price-level seeds: exploits TARGET topology
    /// rather than price levels. Sentinel false → byte-exact CTRL i26 (std-arb third).
    /// Overrides use_std_arb_third when true. TIME-NEUTRAL: O(num_batteries)/step.
    #[serde(default)]
    pub use_obl_target_seed: bool,
    /// L9 i28 VarB: replace slot-3 with a temporal ramp seed — discharge when
    /// DA price is rising (price[t] >= price[t-1]), charge when falling.
    /// Captures first-order price momentum (structurally distinct from level-based
    /// arb seeds). Sentinel false → byte-exact CTRL i26. Lower priority than
    /// use_obl_target_seed (only active when use_obl_target_seed=false).
    /// TIME-NEUTRAL: O(num_batteries)/step.
    #[serde(default)]
    pub use_ramp_seed: bool,
    /// L9 i29 VarC: replace slot-3 with a DA-price-level seed — discharge when
    /// day_ahead_prices[t][node] > percentile(da_prices_cross_node, arb_pct=75).
    /// Signal is orthogonal to RT (congestion DA≠RT by node). Time-neutral: O(batt)/step.
    /// Sentinel false → byte-exact CTRL i25 (std-arb-third). Priority above VarD
    /// and use_std_arb_third, below use_obl_target_seed and use_ramp_seed.
    /// Ref: mem b18dec80 (DA info orthogonal to RT), c6ffa673 (RT-DA bounded deviation).
    #[serde(default)]
    pub use_da_arb_third: bool,
    /// L9 i29 VarD: replace slot-3 with a DA-RT spread seed — discharge when
    /// (rt_prices[node] − da_prices[t][node]) > percentile(spread_cross_node, arb_pct=75).
    /// Captures real-time scarcity premium above DA forecast (structurally distinct from
    /// both RT-level and DA-level seeds). Spread already computed in history l.2280.
    /// Sentinel false → byte-exact CTRL i25. Lower priority than VarC.
    #[serde(default)]
    pub use_spread_arb_third: bool,
    
    /// decreases with battery fullness: trigger = arb_threshold − wv_kappa*(press−0.5)*|arb_threshold|.
    /// Full battery (press→1) → lower threshold → discharge more eagerly (cheap stored energy).
    /// Empty battery (press→0) → higher threshold → discharge only at high prices (scarce storage).
    /// Sentinel false → byte-exact CTRL i29 (DA-arb-third active). Highest priority among slot-3
    /// variants (overrides obl/ramp/da/spread). Congestion-agnostic, bang-bang, O(batteries)/step.
    /// Ref: mem c722c64b (opportunity cost raises trigger price), 4d155575 (dv/dSoC decreasing).
    #[serde(default)]
    pub use_water_value_third: bool,
    /// Modulation depth for water-value seed (only read when use_water_value_third=true).
    /// trigger = arb_threshold − wv_kappa*(press−0.5)*|arb_threshold|. kappa=0 → std-arb-threshold.
    #[serde(default = "default_wv_kappa")]
    pub wv_kappa: f64,
    
    /// temporel énergie-balancé. Charge (→lo) K cheapest DA steps, discharge (→hi) K costliest,
    /// idle between. K = min(round(cap/(charge_mw*DELTA_T)), round(block_frac*T)). Time-neutral:
    /// sort O(T·logT)/batt done ONCE before grid_optimize (not per-step). Sentinel false →
    /// byte-exact CTRL i29 (use_da_arb_third active). Lower priority than water_value.
    #[serde(default)]
    pub use_block_seed_third: bool,
    /// Fraction of horizon allocated to charge block = discharge block (symmetric K).
    #[serde(default = "default_block_frac")]
    pub block_frac: f64,
    
    /// bang-bang seed. For each battery b at step t: collect DA prices[t..t+W] on b's node;
    /// compute P75 of this local window; discharge (→hi) if price[t] > P75(window), else (→lo).
    /// Geometry #3: temporal per-battery window vs cross-node point-wise (#1 dead) and
    /// block-K SoC-aveugle (#2 dead). SoC-adaptive via action_bounds. Congestion-agnostic.
    /// Cheap O(W·batt)/step. Sentinel false → byte-exact CTRL i29 (da_arb_third active).
    /// Priority: above use_da_arb_third, below use_block_seed_third.
    /// Refs: mem 0f4fd9cb/9b872be0 (Powell-Ghadimi deterministic lookahead valid+cheap),
    ///       mem 8bdf733f (S3TS: short window W suffices).
    #[serde(default)]
    pub use_rollout_seed_third: bool,
    /// Window size W for lookahead-rollout seed (timesteps; only read when use_rollout_seed_third=true).
    /// P75 of da_prices[t..t+W] on battery node = local temporal threshold. Default 12.
    
    #[serde(default = "default_rollout_window")]
    pub rollout_window: u64,
    
    /// When true: num_seeds 3→4 (spends ~13k cyc time margin for extra diversity).
    /// Gate: elapsed ≤ 18,000 cyc on M11/M14. Only active when use_arb_seed=true.
    #[serde(default)]
    pub use_rollout_additive: bool,
    
    /// Only active when use_arb_seed=true AND use_rollout_additive=true (i33 4-seed stack active).
    /// Explores short-horizon basins orthogonal to W=12. Default false = byte-exact i33 baseline.
    #[serde(default)]
    pub use_rollout_additive_2: bool,
    /// Window size W2 for the 2nd additive rollout seed (timesteps). Default 4 (~2h at 30-min steps).
    
    #[serde(default = "default_rollout_window_2")]
    pub rollout_window_2: u64,
    
    /// Only active when use_arb_seed=true AND use_rollout_additive=true AND use_rollout_additive_2=true.
    /// Explores ultra-short-horizon basins (W=2: 1h cycles) orthogonal to W=4 and W=12. Default false.
    #[serde(default)]
    pub use_rollout_additive_3: bool,
    /// Window size W3 for the 3rd additive rollout seed (timesteps). Default 2 (~1h at 30-min steps).
    
    #[serde(default = "default_rollout_window_3")]
    pub rollout_window_3: u64,
    
    /// Mechanism: discharge if price[t] > price[t-1] (prices just rose — sell at momentum peak).
    /// Orthogonal to forward-looking W=2 (sells if price is about to fall); together they fire at
    /// local price maxima (both past rise and future fall). Default false.
    #[serde(default)]
    pub use_rollout_additive_4: bool,
    /// Conceptual window for 4th additive seed (always 1 = backward one step). Reserved for HP
    /// serialization consistency; the mechanism uses price[t-1] directly, not a P75 window.
    #[serde(default = "default_rollout_window_4")]
    pub rollout_window_4: u64,
    /// L7 basin-hopping ZDPG (i42): after pair+triplet polish, perturb K batteries (largest
    /// span_b) by N(0, basin_hop_scale·span_b) clamped [lo_b,hi_b], re-run pair polish only,
    /// accept-if-improved. Escapes the pairwise local optimum proven by i41 (ΔQ=0 BIT-EXACT
    /// at budget=1225). PRNG seeded deterministically: splitmix64(nonce_u64 ^ step_mix).
    /// Sentinel: false → 0-diff CTRL byte-exact (LLVM dead-eliminates branch). Default false.
    #[serde(default)]
    pub use_basin_hop: bool,
    /// Gaussian perturbation scale: N(0, basin_hop_scale·span_b). Only read when use_basin_hop=true.
    /// Sweep {0.05, 0.10}. Default 0.05.
    #[serde(default = "default_basin_hop_scale")]
    pub basin_hop_scale: f64,
    /// Number of batteries to perturb (K largest span_b). Only read when use_basin_hop=true.
    /// Sweep {4, 8}. Default 4.
    #[serde(default = "default_basin_hop_k")]
    pub basin_hop_k: usize,
    /// L8 b-LAHC late-acceptance for joint_pair_polish (i44). History-list of length Lh stores
    /// total solution quality at each pair-visit; accept a swap if it beats the current total
    /// (strict) OR if it beats the total from Lh pair-visits ago (late acceptance, traverses
    /// pairwise plateaus without random perturbation). Tracks best_seen over the full LAHC walk
    /// and returns argmax(Q_total) — anti-artefact guard analogous to i43 root-cause fix.
    /// Sentinel lh=0 → pure accept-if-improved = CTRL BIT-EXACT with i42. Sweep {10, 5, 20}.
    #[serde(default)]
    pub pair_lahc_lh: usize,
    /// i47 b-LAHC noise-init (paper 82c5d949): replace constant hist init with balanced
    /// perturbation. Each P[i] = current_total * factor_i, factor_i ~ Uniform[1-span, 1+span],
    /// drawn by PRNG nonce-seeded + time_step (fully deterministic → BIT-EXACT 2×).
    /// Avoids overly strict OR overly lenient acceptance thresholds before Lh steps.
    /// Span=0.0 (default sentinel) → constant init = BIT-EXACT i44.
    #[serde(default)]
    pub lahc_init_alpha_span: f64,
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
        hp.rh_stride = hp.rh_stride.max(1);
        hp.soc_ref_dyn_stride = hp.soc_ref_dyn_stride.max(1);
        // ADMM (i90): ρ strictly positive (divisor H_diag+ρ), at least 1 iteration.
        if !(hp.admm_rho > 0.0) {
            hp.admm_rho = 0.45;
        }
        hp.admm_iters = hp.admm_iters.max(1);
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


// soc_ref[b][t] = target SoC for battery b at step t.
static SOC_REF: OnceLock<Mutex<Vec<Vec<f64>>>> = OnceLock::new();
fn soc_ref_lock() -> &'static Mutex<Vec<Vec<f64>>> {
    SOC_REF.get_or_init(|| Mutex::new(Vec::new()))
}

// L8b: dynamic SoC reference — recomputes at each RH window from current_socs and
// DA+residual_shift prices. At start_t=0 shift=0 (pure DA); at later windows shift
// reflects realized RT deviation → reference adapts to actual prices (c1b96402/2184fa76).
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
        // P25/P75 from adjusted remaining-horizon DA prices.
        let mut da: Vec<f64> = (start_t..n_steps)
            .map(|t| challenge.market.day_ahead_prices[t][node] + shift)
            .collect();
        da.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p25 = da[da.len() / 4];
        let p75 = da[(da.len() * 3) / 4];
        // Forward greedy from start_t using adjusted prices.
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

// ---- L7: SOC-contingent value correction (SCVC) ----

/// Greedy forward rollout on the DP value function to obtain the reference SOC trajectory.
/// Returns soc_ref[t] for t=0..=num_steps (soc before step t's action).
/// Starting SOC = midpoint of [soc_min, soc_max] (API-safe; absolute start matters less than
/// relative slope shape since correction slope is uniform across all SOC levels).
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
        // Fixed small grid (9 levels) for rollout: cheap, deterministic, sufficient resolution.
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

/// Apply SCVC correction to the DP value table (in-place, one-shot post-processing).
/// For t=1..num_steps-1 (excluding terminal): values[t][s] += alpha * (soc_s - soc_ref[t]) * marge_b.
/// slope = alpha * marge_b shifts dv_dsoc uniformly, recalibrating the surrogate slope toward
/// the greedy trajectory's implied water value. Does NOT modify the terminal values[num_steps].
fn apply_scvc_to_dp(dp: &mut BatteryDP, battery: &Battery, da_at_node: &[f64], alpha: f64) {
    let num_steps = dp.values.len().saturating_sub(1);
    if num_steps < 2 || dp.levels < 2 {
        return;
    }
    let soc_lo = dp.soc_lo;
    let soc_step = 1.0 / dp.soc_step_inv;
    let levels = dp.levels;

    let traj = scvc_greedy_trajectory(dp, battery, da_at_node);

    // marge_b = mean |DA price|: proxy for SOC water value ($/MWh → $/MWh-SOC scale).
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



/// Per-battery 4-variable LP solver using a stack-allocated simplex tableau.
/// Maximizes f0*d0 + g0*c0 + f1*d1 + g1*c1 subject to:
///   d0 ≤ ub_d0, c0 ≤ ub_c0, d1 ≤ ub_d1, c1 ≤ ub_c1               (UBs)
///   d0*d_f - c0*c_f ≤ avail,  -d0*d_f + c0*c_f ≤ head              (SOC[t+1])
///   (d0+d1)*d_f - (c0+c1)*c_f ≤ avail,  -(d0+d1)*d_f + (c0+c1)*c_f ≤ head (SOC[t+2])
///   d0,c0,d1,c1 ≥ 0
/// All coefficients are stack-allocated (117 f64 = 936 bytes) — zero heap alloc.
/// Returns [d0, c0, d1, c1].
#[inline(always)]
fn solve_battery_kkt(
    f0: f64, g0: f64, f1: f64, g1: f64,
    ub_d0: f64, ub_c0: f64, ub_d1: f64, ub_c1: f64,
    avail: f64, head: f64, d_f: f64, c_f: f64,
) -> [f64; N_4VAR] {
    let mut tab = [[0.0_f64; NC_4VAR]; M_4VAR + 1];

    // Row 0: d0 ≤ ub_d0
    tab[0][0] = 1.0; tab[0][N_4VAR] = 1.0; tab[0][NV_4VAR] = ub_d0;
    // Row 1: c0 ≤ ub_c0
    tab[1][1] = 1.0; tab[1][N_4VAR + 1] = 1.0; tab[1][NV_4VAR] = ub_c0;
    // Row 2: d1 ≤ ub_d1
    tab[2][2] = 1.0; tab[2][N_4VAR + 2] = 1.0; tab[2][NV_4VAR] = ub_d1;
    // Row 3: c1 ≤ ub_c1
    tab[3][3] = 1.0; tab[3][N_4VAR + 3] = 1.0; tab[3][NV_4VAR] = ub_c1;
    // Row 4: d0*d_f - c0*c_f ≤ avail  (SOC[t+1] upper)
    tab[4][0] = d_f; tab[4][1] = -c_f; tab[4][N_4VAR + 4] = 1.0; tab[4][NV_4VAR] = avail;
    // Row 5: -d0*d_f + c0*c_f ≤ head  (SOC[t+1] lower)
    tab[5][0] = -d_f; tab[5][1] = c_f; tab[5][N_4VAR + 5] = 1.0; tab[5][NV_4VAR] = head;
    // Row 6: (d0+d1)*d_f - (c0+c1)*c_f ≤ avail  (SOC[t+2] upper)
    tab[6][0] = d_f; tab[6][1] = -c_f; tab[6][2] = d_f; tab[6][3] = -c_f;
    tab[6][N_4VAR + 6] = 1.0; tab[6][NV_4VAR] = avail;
    // Row 7: -(d0+d1)*d_f + (c0+c1)*c_f ≤ head  (SOC[t+2] lower)
    tab[7][0] = -d_f; tab[7][1] = c_f; tab[7][2] = -d_f; tab[7][3] = c_f;
    tab[7][N_4VAR + 7] = 1.0; tab[7][NV_4VAR] = head;
    // Objective row (negated for max→min)
    tab[M_4VAR][0] = -f0; tab[M_4VAR][1] = -g0;
    tab[M_4VAR][2] = -f1; tab[M_4VAR][3] = -g1;

    // Initial basis: all slacks (x=0 is always feasible since avail,head ≥ 0)
    let mut basis = [
        N_4VAR, N_4VAR + 1, N_4VAR + 2, N_4VAR + 3,
        N_4VAR + 4, N_4VAR + 5, N_4VAR + 6, N_4VAR + 7,
    ];

    for _ in 0..(3 * N_4VAR + 2) {
        // Find most-negative reduced cost (entering variable)
        let mut entering = NV_4VAR;
        let mut min_c = -LP_EPS;
        for j in 0..NV_4VAR {
            if tab[M_4VAR][j] < min_c {
                min_c = tab[M_4VAR][j];
                entering = j;
            }
        }
        if entering == NV_4VAR { break; } // Optimal

        // Min-ratio test (leaving row)
        let mut leaving = M_4VAR;
        let mut min_r = f64::MAX;
        for i in 0..M_4VAR {
            if tab[i][entering] > LP_EPS {
                let r = tab[i][NV_4VAR] / tab[i][entering];
                if r < min_r { min_r = r; leaving = i; }
            }
        }
        if leaving == M_4VAR { break; } // Unbounded (shouldn't occur in bounded problem)

        // Pivot
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


///
/// Replaces the dense O(m×n) simplex (i44/i45 floor ~2.0s) with:
///   1. Per-battery tiny 4-variable simplex (stack-allocated, zero heap alloc per step).
///   2. Subgradient on PTDF dual multipliers (RH_KKT_ITERS iterations).
///
/// The per-battery sub-problems are solved exactly. PTDF coupling is handled via
/// Lagrangian relaxation (dual decomposition): each iteration adjusts effective prices
/// for each battery based on constraint violations, then re-solves per-battery.
/// The seed is PGA-safe: PGA keeps it only if it improves quality.
/// lam_warm holds per-line (fwd, rev) multipliers from the previous time step.
/// Updated in place so the next step can warm-start from these values.
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

    // Per-battery base objectives and constraints
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

    // Quasi-binding PTDF lines (same threshold as before — only dualize congested lines)
    let binding_lines: Vec<usize> = limits.iter().enumerate()
        .filter(|&(l, &lim)| {
            lim > 1e-6
                && base_flows.get(l).copied().unwrap_or(0.0).abs() / lim > RH_BINDING_RATIO
        })
        .map(|(l, _)| l)
        .collect();
    let n_binding = binding_lines.len();

    // PTDF dual multipliers — warm-start from previous time step (i47).
    // lam_warm holds per-line values; project onto binding_lines for this step.
    let mut lam_fwd: Vec<f64> = binding_lines.iter()
        .map(|&l| lam_warm.0.get(l).copied().unwrap_or(0.0))
        .collect();
    let mut lam_rev: Vec<f64> = binding_lines.iter()
        .map(|&l| lam_warm.1.get(l).copied().unwrap_or(0.0))
        .collect();

    let mut d0_sol = vec![0.0_f64; num_b];
    let mut c0_sol = vec![0.0_f64; num_b];

    // Subgradient dual loop — capped step (RH_KKT_ALPHA_MAX) prevents overshoot
    // when warm-start λ is near-optimal and constraints vary slowly (Hindsight ba42ac1e).
    for iter in 0..RH_KKT_ITERS {
        let alpha = (RH_KKT_ALPHA0 / ((iter + 1) as f64).sqrt()).min(RH_KKT_ALPHA_MAX);

        // Per-battery solve with PTDF-adjusted prices
        for b in 0..num_b {
            // Effective price adjustment from PTDF dual: Σ_k (λ_fwd[k] - λ_rev[k]) * sens[k][b]
            let ptdf_adj: f64 = binding_lines.iter().enumerate()
                .map(|(k, &l)| {
                    let s = sens.get(l).and_then(|r| r.get(b)).copied().unwrap_or(0.0);
                    (lam_fwd[k] - lam_rev[k]) * s
                })
                .sum();
            // eff_f0 = f0 - adj (discharge penalized if forward line overloaded)
            // eff_g0 = g0 + adj (charge encouraged to relieve forward overload)
            let sol = solve_battery_kkt(
                f0[b] - ptdf_adj, g0[b] + ptdf_adj,
                f1[b], g1[b],
                ub_d0[b], ub_c0[b], ub_d1[b], ub_c1[b],
                available[b], headroom[b], d_f, c_f,
            );
            d0_sol[b] = sol[0];
            c0_sol[b] = sol[1];
        }

        // Dual update via subgradient (projected gradient, λ ≥ 0)
        for (k, &l) in binding_lines.iter().enumerate() {
            let lim = limits[l];
            let bf = base_flows.get(l).copied().unwrap_or(0.0);
            // Net power injected by batteries at step t on line l
            let net_flow: f64 = sens[l].iter().zip(d0_sol.iter().zip(c0_sol.iter()))
                .map(|(&s, (&d, &c))| s * (d - c))
                .sum();
            // Forward constraint: net_flow ≤ lim - bf; violation = net_flow - (lim - bf)
            let b_fwd = (lim - bf).max(0.0);
            // Reverse constraint: -net_flow ≤ lim + bf; violation = -net_flow - (lim + bf)
            let b_rev = (lim + bf).max(0.0);
            lam_fwd[k] = (lam_fwd[k] + alpha * (net_flow - b_fwd)).max(0.0);
            lam_rev[k] = (lam_rev[k] + alpha * (-net_flow - b_rev)).max(0.0);
        }
    }

    // Write back to warm state so next step can hot-start (per-line indexed).
    for (k, &l) in binding_lines.iter().enumerate() {
        if l < lam_warm.0.len() {
            lam_warm.0[l] = lam_fwd[k];
            lam_warm.1[l] = lam_rev[k];
        }
    }

    // Final per-battery solve with converged duals
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
/// When hp.soc_ref_lambda > 0, adds the L8 reference-tracking penalty
/// -lambda*(next_soc - soc_ref[b][t+1])*dsoc_du to steer SoC toward

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
            // Subgradient at u=0: pick direction with steeper slope average.
            -0.5 * (DELTA_T / ETA_DISCHARGE + ETA_CHARGE * DELTA_T)
        };
        let dv = dv_dsoc(&dps[b], state.time_step, next_soc);
        grad[b] = imm + dv * dsoc_du;

        // L8: SoC reference-tracking penalty (Huang 2024, prediction-free two-stage).
        // grad[b] -= lambda*(next_soc - soc_ref_t1)*dsoc_du
        // dsoc_du < 0 always, so:
        //   next_soc > soc_ref → penalty > 0 → increases discharge → SoC ↓ toward ref ✓
        //   next_soc < soc_ref → penalty < 0 → decreases discharge → SoC ↑ toward ref ✓
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

/// L8-bis (i90): augmented-Lagrangian ADMM operator-splitting solver of the
/// per-step joint QP (maximise total_step_value s.t. action ∈ box ∩ flow-polytope).
/// Replaces the gradient-projected ascent inner loop when `hp.use_admm_solver`.
///
/// Mechanism (Srikanthan 2024 `cc38eba3`/`5a1d4bcf`, reformulation `339fbf44`/
/// `99718796`), scaled-dual form (w = u/ρ), Gauss-Seidel sequential updates:
///   1. x-update (smooth, NON-projected, box-only): proximal-linearised argmax of the
///      local concave quadratic model of f minus the (ρ/2)‖x − z + w‖² penalty.
///      With diagonal curvature H_b = 2·κ_deg·Δt²/cap² (the only explicit quadratic
///      in the objective — the degradation penalty; concave ⇒ H_b ≥ 0, so the divisor
///      H_b+ρ > 0 is always well posed), the closed form per battery is
///        x_b ← (g_b + H_b·x_b + ρ(z_b − w_b)) / (H_b + ρ)
///      i.e. ONE analytical solve, NO inner CG (this is what annuls the PDHCG cost).
///      Box-clamped (not flow-projected): keeps x bounded without the projection.
///   2. z-update (feasibility-bearing): z ← Π_{box∩flow}(x + w) via
///      `safe_project_to_feasible` (the SAME projection PGA uses). z is the variable
///      we APPLY (corrected form `3c3b04c4`: never apply the unbounded x — the
///      historical t51 invalid-nonce failures `8320cca3`/`c5545026` applied x).
///   3. dual: w ← w + (x − z).
/// Per iter cost ≈ 1 gradient + 1 projection + O(n) elementwise (`40f09769`); ρ
/// constant ⇒ value-function-decreasing convergence (`4317e23f`/`86f29e25`).
/// Best tracked over z only (always feasible by construction).
/// Pure sequential arithmetic — no threading/IO/rayon (TIG sandbox).
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
    let rho = hp.admm_rho; // parse-clamped > 0

    // Per-battery diagonal curvature of the degradation penalty (concave ⇒ ≥ 0).
    let h_diag: Vec<f64> = (0..n)
        .map(|b| {
            let cap2 = challenge.batteries[b].capacity_mwh.powi(2).max(1e-9);
            2.0 * KAPPA_DEG * DELTA_T * DELTA_T / cap2
        })
        .collect();

    // z starts feasible (= projected seed); x mirrors it; scaled dual w = 0.
    let mut z = seed;
    safe_project_to_feasible(challenge, state, &mut z, sens, base_flows, hp);
    let mut x = z.clone();
    let mut w = vec![0.0_f64; n];

    let mut best_action = z.clone();
    let mut best_value = total_step_value(challenge, state, dps, &best_action);

    for _ in 0..hp.admm_iters {
        // (1) x-update — analytical, box-only, NON-projected. Re-linearise f at x.
        let g = analytic_gradient(challenge, state, dps, &x, hp);
        for b in 0..n {
            x[b] = (g[b] + h_diag[b] * x[b] + rho * (z[b] - w[b])) / (h_diag[b] + rho);
        }
        clamp_to_bounds(&mut x, &state.action_bounds);

        // (2) z-update — project (x + w) onto box ∩ flow; z carries feasibility.
        let mut z_new: Vec<f64> = x.iter().zip(w.iter()).map(|(xi, wi)| xi + wi).collect();
        safe_project_to_feasible(challenge, state, &mut z_new, sens, base_flows, hp);
        z = z_new;

        // (3) dual update (scaled): w ← w + (x − z).
        for b in 0..n {
            w[b] += x[b] - z[b];
        }

        // Track best over the feasible iterate z (the only one we may apply).
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
    // L8-bis (i90): swap the whole inner solver for ADMM when flagged. Default
    // false ⇒ this guard is const-folded away and the PGA path below is byte-exact.
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

    // Barzilai-Borwein step-size clamps (gated by `use_bb_clamps`, ported from
    
    // on improvement (anti-overshoot, cf Hindsight 1b78ed87); BB_DECAY_FACTOR floors
    // step decay on failure (anti clamp-saturation/oscillation, cf Hindsight d240f498).
    const LR_GROWTH_CAP: f64 = 1.05;
    const BB_DECAY_FACTOR: f64 = 0.85;
    // Heavy-ball momentum coefficient. Hardcoded const (NOT serde f64) — a serde
    // f64 field breaks LLVM const-fold and caused +51% regression at t52/i9.
    const MOMENTUM_BETA: f64 = 0.99;
    let t_max = hp.grad_outer_iters.saturating_sub(1).max(1) as f64;

    let mut lr = max_power * 0.5;
    // Velocity accumulator reset per call (= per seed). All-zero when
    // `use_momentum=false` → step direction = grad verbatim (iso-baseline).
    let mut velocity = vec![0.0_f64; action.len()];
    for outer_iter in 0..hp.grad_outer_iters {
        // β decays 0.99→pga_beta_end via cosine if use_cosine_beta, else constant 0.99.
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
        let mut cur_lr = lr;
        for _ in 0..hp.grad_ls_iters {
            // step_scale uses grad norm (unchanged backtracking); momentum only
            // re-orients direction, not the line-search budget.
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

/// Pairwise cross-battery coordinate-exchange polish (P7, ported from t52/i13).
/// splitmix64 PRNG — deterministic, no system entropy.
#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// L7 basin-hopping ZDPG (i42): perturb K batteries by N(0, scale·span_b), re-polish pairs,
/// accept-if-improved. Escapes pairwise local optimum (proven i41 ΔQ=0 at budget=1225).
/// PRNG seeded from challenge.seed (per-nonce) + time_step — fully deterministic.
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

    // Deterministic PRNG: splitmix64 seeded from nonce + timestep
    let nonce_u64 = u64::from_le_bytes([
        challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
        challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
    ]);
    let mut rng: u64 = nonce_u64
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_mul(0xbf58476d1ce4e5b9)
        ^ ((state.time_step as u64).wrapping_mul(0x94d049bb133111eb));

    // Select K batteries with largest action span (most degrees of freedom to perturb)
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

    // Perturb selected batteries via Box-Muller
    for &(span_b, b) in spans.iter().take(k) {
        if span_b < 1e-9 {
            continue;
        }
        let (lo, hi) = state.action_bounds[b];
        // Box-Muller: two uniform draws → N(0,1)
        let u1_raw = splitmix64(&mut rng);
        let u2_raw = splitmix64(&mut rng);
        // Map to (0,1) avoiding exact 0
        let u1 = (u1_raw as f64 + 0.5) / 18446744073709551616.0_f64;
        let u2 = (u2_raw as f64 + 0.5) / 18446744073709551616.0_f64;
        let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        actions[b] = (actions[b] + z * scale * span_b).clamp(lo, hi);
    }

    // Re-run pair polish only (skip triplet — saves ~+1s per nonce)
    joint_pair_polish(challenge, state, dps, sens, base_flows, actions, hp);

    // Accept only if feasible and strictly improved
    if !is_flow_feasible(challenge, state, actions) {
        *actions = pre_hop;
        return;
    }
    let new_val = total_step_value(challenge, state, dps, actions);
    if new_val <= current_val + 1e-9 {
        *actions = pre_hop;
    }
}

/// For each battery pair (i,j) probes equal-and-opposite shifts ±α·span that
/// the independent gradient / per-battery coordinate polish cannot see. Sequential
/// (no threading); first-improvement restart; capped at `joint_pair_budget` per pass.
///
/// When hp.pair_lahc_lh > 0: b-LAHC late-acceptance (i44). History list P[Lh] stores
/// the total solution quality at each pair-visit step. Acceptance criterion relaxed to:
/// "accept if new_total > current_total (strict) OR new_total > P[step-Lh] (late)".
/// This traverses pairwise plateaus where accept-if-improved stalls. Tracks best_seen
/// over the full LAHC walk and returns argmax(Q_total) — anti-artefact guard against
/// monotone-but-negative DP-VF micro-drift (cf i43 root-cause: DP-VF artefact 1e-9).
/// Sentinel lh=0: pure accept-if-improved, BIT-EXACT with i42/i43 code path.
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

    // b-LAHC state — only allocated/used when lh > 0 (sentinel lh=0 is BIT-EXACT with i42).
    // History P[Lh] tracks total solution quality at each pair-visit; late-accept if
    // new_total > P[step % lh] (quality from Lh pair-visits ago).
    let mut lahc_hist: Vec<f64> = Vec::new();
    let mut lahc_step: usize = 0;
    let mut current_total: f64 = 0.0;
    let mut best_total: f64 = f64::NEG_INFINITY;
    let mut best_actions: Vec<f64> = Vec::new();
    if lh > 0 {
        current_total = total_step_value(challenge, state, dps, actions);
        best_total = current_total;
        best_actions = actions.clone();
        // b-LAHC noise-init (i47): paper 82c5d949 init P[i] = current_total * factor_i,
        // factor_i ~ Uniform[1-span, 1+span], PRNG nonce-seeded (fully deterministic, BIT-EXACT 2×).
        // Balanced perturbation: avoids overly strict (factor<1) OR overly lenient (factor>1) thresholds.
        // Sentinel span=0.0 → constant init = BIT-EXACT i44.
        let span = hp.lahc_init_alpha_span;
        if span > 0.0 {
            let nonce_u64 = u64::from_le_bytes([
                challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
                challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
            ]);
            // Use different magic constant than basin_hop_restart to avoid correlation.
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

                // LAHC: read historical total from this history slot before updating it.
                // late_pair_thr = val threshold s.t. (current_total + val - base_val) > hist_total
                //               = base_val + (hist_total - current_total)
                // Only meaningful when lh > 0.
                let hist_total = if lh > 0 {
                    let slot = lahc_step % lh;
                    let hv = lahc_hist[slot];
                    lahc_hist[slot] = current_total;  // record current quality in this slot
                    lahc_step = lahc_step.wrapping_add(1);
                    hv
                } else {
                    0.0  // unused when lh == 0
                };
                let late_pair_thr = base_val + (hist_total - current_total);

                let mut best_val = base_val;
                let mut best_i = cur_i;
                let mut best_j = cur_j;
                // Late-acceptance candidates (separate tracking to avoid conflict with strict).
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
                        // Separate late-acceptance candidate (only when lh > 0).
                        if lh > 0 && val > late_val + 1e-9 {
                            late_val = val;
                            late_i = cand_i;
                            late_j = cand_j;
                        }
                    }
                }

                let is_strict = (best_i - cur_i).abs() > EPS || (best_j - cur_j).abs() > EPS;
                // is_late: lh>0, some candidate beats late threshold, AND it's not already strict.
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
                    // Update total and track best_seen (lh > 0 only).
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

    // LAHC: return best_seen over the full walk, not the current (wandering) state.
    // Prevents monotone-negative DP-VF micro-drift (cf i43 artefact root-cause).
    // Sentinel lh=0: best_actions is empty → no-op, BIT-EXACT.
    if lh > 0 && !best_actions.is_empty() {
        *actions = best_actions;
    }
}

/// Triplet battery-action exchange polish (L2). Escapes the pairwise fixed point
/// proven in i30 (C(40,2)=780, 2nd pass Δ=0 EXACT). For each sampled triplet
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
    
    // C(15,3)=455 ≥ budget=150 — concentrates budget on most active batteries (post pair-polish).
    // Lexicographic order across active[] preserves the first-improvement greedy structure.
    let top_k = hp.joint_triplet_top_k.max(3).min(num_b);
    let mut batt_scores: Vec<(f64, usize)> = (0..num_b)
        .map(|b| (actions[b].abs(), b))
        .collect();
    batt_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let active: Vec<usize> = batt_scores.iter().take(top_k).map(|&(_, b)| b).collect();

    // Single-pass greedy: iterate at most `joint_triplet_budget` triplets sequentially,
    // applying improvements immediately without restarting. Eliminates the while-improved
    // restart overhead that caused ×9.6 pair-cost in i32 (budget=1500 with restarts).
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
                // Read current actions (updated in-place as improvements are found).
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
                // ±α on i (gainer), ∓α on j (equal-and-opposite), +γ on k
                // alpha_k reduced to {-0.25, 0.0, 0.25} (3 values vs 5 in i32) for speed.
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
                    // Continue to next triplet (single-pass greedy, no restart).
                }
            }
        }
    }
}

/// L6 — variable-depth ejection-chain polish (Lin-Kernighan style), post-jtp.
///
/// Motivation: on t51 every *immediately-improving* perturbation of the jpp+jtp fixed point is
/// exhausted (i37 midpoint −183Q, i39 SOE −1,202Q, i40 quad −1,174Q, i41 ADMM −1,101Q — all
/// gated on local improvement / simultaneous moves). The residual barrier can only be crossed
/// by a move that *temporarily worsens* the objective. This routine does exactly that:

///   2. FORCE an ejection move on it (±α·span) regardless of sign — this usually lowers value.
///   3. Greedily extend a chain: at each depth pick the unused battery whose best PTDF-feasible
///      single move maximises the running total (it may itself be worsening — the chain absorbs).
///   4. After every depth step, if the full action vector is PTDF-feasible AND its value strictly
///      beats the chain start, snapshot it. Commit the best snapshot only (best-prefix rule).
/// Each chain runs on a private copy, so a chain that never recovers leaves `work` untouched —
/// the routine is monotone-safe (it can only ever raise total_step_value or no-op).
fn joint_ejection_chain_polish(
    challenge: &Challenge,
    state: &State,
    dps: &[BatteryDP],
    sens: &[Vec<f64>],
    base_flows: &[f64],
    actions: &mut Vec<f64>,
    hp: &Hyperparameters,
) {
    const EJECTION_MAX_DEPTH: usize = 3; // chain length: 1 eject + up to 2 absorbing moves
    const EJECTION_BUDGET: usize = 24; // max seed×direction chains explored (≈ jtp time cost)
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

    // Mutable working solution + its live flows (updated only when a chain commits).
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

            // Private chain copy: start from the current committed `work`/`flows`.
            let mut chain_act = work.clone();
            let mut chain_flows = flows.clone();
            let start_val = total_step_value(challenge, state, dps, &chain_act);
            let mut in_chain = vec![false; num_b];

            // Step 0 — forced ejection on the seed (sign-agnostic, may worsen).
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

            // Absorbing steps — extend the chain greedily.
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
                        // PTDF feasibility of this single move against current chain flows.
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
                    break; // no feasible absorbing move remains
                }
                // Apply the best absorbing move to the chain.
                let delta_m = best_cand_m - chain_act[best_m];
                for l in 0..num_l {
                    chain_flows[l] += sens[l][best_m] * delta_m;
                }
                chain_act[best_m] = best_cand_m;
                chain_val = best_gain_val;
                in_chain[best_m] = true;

                // Snapshot if the whole chain state is feasible AND strictly improves the start.
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

            // Commit the best strictly-improving feasible prefix into the working solution.
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

/// L6 ADMM consensus polish: relaxes PTDF coupling in per-battery u-update,
/// then coordinates via z-projection + dual y-update.

/// Only replaces actions if the final consensus strictly improves total_step_value.
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

    // z initialized to current feasible jtp-polished actions
    let mut z = actions.clone();
    // Dual variables (per battery, for u = z coupling)
    let mut y = vec![0.0_f64; num_b];

    for _iter in 0..ADMM_ITERS {
        // u-update: for each battery independently, maximize
        //   dp_val(b, u) - (RHO/2) * (u - (z_b - y_b/RHO))^2
        // This relaxes the PTDF coupling: u can be infeasible, dual y enforces consensus.
        let mut u = vec![0.0_f64; num_b];
        for b in 0..num_b {
            let battery = &challenge.batteries[b];
            let (lo, hi) = state.action_bounds[b];
            let soc = state.socs[b];
            let price = state.rt_prices[battery.node];
            // ADMM proximity center
            let center = (z[b] - y[b] / RHO).clamp(lo, hi);

            let eta_rt = ETA_CHARGE * ETA_DISCHARGE;
            let friction = 2.0 * KAPPA_TX;
            let charge_max = price * eta_rt - friction;
            let discharge_min = price / eta_rt + friction;

            let mut best_u = center;
            let mut best_v = dp_action_value(&dps[b], battery, t, soc, price, center)
                - (RHO / 2.0) * (center - z[b] + y[b] / RHO).powi(2);

            // Candidate actions: grid + boundary + center
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

        // z-update: project u onto PTDF feasible set (u may be infeasible by design)
        let mut new_z = u.clone();
        project_polytope(&mut new_z, &state.action_bounds, sens, base_flows, limits, ADMM_PROJ_ITERS);
        clamp_to_bounds(&mut new_z, &state.action_bounds);

        // y-update: dual ascent y_b += RHO * (u_b - z_b)
        for b in 0..num_b {
            y[b] += RHO * (u[b] - new_z[b]);
        }
        z = new_z;

        // Track best feasible consensus seen
        if is_flow_feasible(challenge, state, &z) {
            let v = total_step_value(challenge, state, dps, &z);
            if v > best_val {
                best_val = v;
                best_z = z.clone();
            }
        }
    }

    // Only replace if strictly improved (never degrade)
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

    // L8b: dynamic SOC reference — recompute at each RH stride using realized price shifts.
    // At t=0 shift is zero (no history yet, acts like static DA). At subsequent windows
    // shift encodes realized RT deviation, pulling the reference toward the actual price path.
    if hp.soc_ref_lambda > 0.0 && t % hp.soc_ref_dyn_stride == 0 {
        let refs = compute_soc_reference_dynamic(challenge, &state.socs, &residual_shift, t);
        let mut soc_ref = soc_ref_lock().lock().unwrap();
        *soc_ref = refs;
    }

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

    // Joint projected gradient ascent over multiple seeds.
    
    // congestion-agnostic seed. arb_pct=0 → mean threshold (i21 baseline, byte-exact CTRL).
    // arb_pct>0 → percentile(rt_prices, arb_pct, 100) threshold (i22 variant).
    // Keeps num_seeds=3 (time-neutral). Default false = byte-exact baseline dp_seed.
    
    // (discharge when price > threshold), giving [target, anti-corr-arb, std-arb] = 3
    // maximally distinct basins. num_seeds=3 CONSTANT (replace, not add).
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
                
                // discharge when price low) = maximally distant from standard arb_seed.
                let discharge_when_high = !hp.arb_inverse;
                if (price > arb_threshold) == discharge_when_high { hi } else { lo }
            })
            .collect();
        let third_seed: Vec<f64> = if hp.use_water_value_third {
            
            // − wv_kappa*(press−0.5)*|arb_threshold|. Full battery lowers discharge
            // threshold (cheap stored energy); empty battery raises it (scarce storage).
            // kappa=0 → std-arb-threshold (byte-exact). O(batteries)/step, time-neutral.
            // Ref: c722c64b (opportunity cost raises trigger), 4d155575 (dv/dSoC ↓ with SoC).
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
            
            // the action midpoint: opp_b = lo_b + hi_b − target_b.
            // Maximally distant from target in the action box (proven mechanism 813c100d).
            // Structurally distinct from all price-level seeds (no price comparison).
            (0..challenge.num_batteries)
                .map(|b| {
                    let (lo, hi) = state.action_bounds[b];
                    (lo + hi - target[b]).clamp(lo, hi)
                })
                .collect()
        } else if hp.use_ramp_seed {
            
            // (price[t] >= price[t-1]), charge when falling. Captures first-order
            // price momentum, structurally distinct from level-based arb seeds.
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
            
            // Charge K cheapest DA steps (→lo), discharge K costliest (→hi), idle between.
            // block_plan[b][t] ∈ {-1=charge, 0=idle, 1=discharge}.
            // Refs: 4b2c189d (charge 01:00-05:00/sell 05:00-10:00), 64d35212 (commitment > myopic).
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
            
            // For battery b at step t: collect DA prices on b's node over [t, t+W]; compute P75
            // of this local temporal window; discharge (→hi) if price[t] > P75, else charge (→lo).
            // Geometry #3: temporal per-battery window vs cross-node point-wise (#1, 7 dead) and
            // block-K SoC-aveugle (#2, i32 dead). SoC-adaptive: action_bounds[b] already encodes
            // current SoC feasibility (no separate SoC tracking needed). Congestion-agnostic.
            // Refs: mem 0f4fd9cb/9b872be0 (Powell-Ghadimi deterministic lookahead valid+cheap),
            //       mem 8bdf733f (S3TS: short window W suffices, full horizon not needed).
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
            
            // Signal orthogonal to RT (congestion DA≠RT by node). Same binary mechanism as
            // std-arb but on the DA price at timestep t (exogenous, not state-derived).
            // Ref: mem b18dec80 (DA info orthogonal to RT for opportunity value).
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
            
            // > P75(spread_cross_node). Captures real-time scarcity premium above DA forecast.
            // Structurally distinct from both RT-level and DA-level seeds.
            // Ref: mem c6ffa673 (RT-DA deviations = bounded structural uncertainty).
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
            // Standard arb: discharge when price > threshold (always, ignoring arb_inverse).
            
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

    
    // 3-seed arb family is intact). Spends ~13k cyc time margin for +1 diverse start.
    // da-arb (slot-3) stays active; rollout is an orthogonal geometry on top.
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

    
    // Short horizon captures intra-day micro-cycles missed by W=12 (medium horizon).
    // Same per-battery P75 mechanism but with rollout_window_2 (W2 ∈ {4,6}).
    // Only active when use_arb_seed=true AND use_rollout_additive=true (4-seed stack required).
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

    
    // W=2 (~1h) captures shortest intra-day micro-cycles missed by both W=4 and W=12.
    // Same per-battery P75 mechanism; orthogonal basins to W=4 (micro-cycle) and W=12 (medium).
    // Only active when use_arb_seed=true AND use_rollout_additive=true AND use_rollout_additive_2=true.
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

    
    // Discharge if price[t] > price[t-1]: prices just rose, sell at momentum peak.
    // Orthogonal to W=2 (forward: sell if price falling in next step); together they
    // fire at local price maxima (past rise AND future fall), exposing a new PGA basin.
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

    
    // RH runs only on steps where time_step % rh_stride == 0 (default stride=1 = every step).
    // stride=2 → 48/96 active steps → overhead halved (~1.1s vs ~2.25s); stride=3 → 32/96.
    // On skipped steps lam_warm retains last active step's λ (stays warm for next activation).
    
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
    result = coordinate_polish_step(challenge, state, dps, sens, result, hp);

    // P7: pairwise cross-battery exchange polish the gradient/coord polish miss.
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

    // L7 basin-hopping ZDPG (i42): perturb K batteries, re-polish pairs, accept-if-improved.
    if hp.use_basin_hop {
        basin_hop_restart(challenge, state, dps, sens, &base_flows, &mut result, hp);
    }

    // L6: ADMM consensus polish — per-battery Lagrangian relaxation of PTDF coupling
    // followed by feasibility projection and dual update (3 iters, RHO=0.1).
    if hp.use_admm_polish {
        let pre_admm = result.clone();
        admm_consensus_polish(challenge, state, dps, sens, &base_flows, &mut result);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_admm;
        }
    }

    // L6: ejection-chain polish — crosses the jtp local barrier via forced-worsening moves.
    if hp.use_ejection_chain {
        let pre_chain = result.clone();
        joint_ejection_chain_polish(challenge, state, dps, sens, &base_flows, &mut result, hp);
        if !is_flow_feasible(challenge, state, &result) {
            result = pre_chain;
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

    
    // t52 i15 P8). When off (default), expected_premiums are zeros → byte-equivalent no-op.
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

    // L7: SCVC — SOC-contingent value correction. One-shot post-processing of dp.values.
    // Corrects the per-battery surrogate slope miscalibration (root cause of i37-i42 plateau).
    // O(num_batteries × num_steps × dp_soc_levels): cheap init-time pass, no re-solve.
    if hp.use_scvc {
        for (b, battery) in challenge.batteries.iter().enumerate() {
            let node = battery.node;
            let da_at_node: Vec<f64> = (0..challenge.num_steps)
                .map(|t| challenge.market.day_ahead_prices[t][node] + expected_premiums[t][b])
                .collect();
            apply_scvc_to_dp(&mut dps[b], battery, &da_at_node, hp.scvc_alpha);
        }
    }

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
    // Warm-start λ state for rolling-horizon seed (i47). Per-rollout, not cross-nonce.
    let n_lines = challenge.network.flow_limits.len();
    let rh_lam: Mutex<(Vec<f64>, Vec<f64>)> = Mutex::new((
        vec![0.0_f64; n_lines],
        vec![0.0_f64; n_lines],
    ));
    
    // For each battery, sort timestep indices by DA price at battery node; assign cheapest K
    // steps to charge (-1), costliest K to discharge (+1), rest idle (0).
    // K = min(round(cap_mwh/(power_charge_mw*DELTA_T)), round(block_frac*num_steps)).
    // Energy-balanced: K charge steps ≈ K discharge steps (symmetric, avoids SoC drift).
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
                plan[step] = -1; // charge on cheapest steps
            }
            for &step in sorted_steps.iter().rev().take(k) {
                // Note: if k=0 this loop doesn't execute; if charge/discharge overlap (k>T/2,
                // guarded by .min(n_t/2)) discharge wins (last write). k≤T/2 prevents overlap.
                plan[step] = 1; // discharge on costliest steps
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

// Important! Do not include any tests in this file, it will result in your submission being rejected
