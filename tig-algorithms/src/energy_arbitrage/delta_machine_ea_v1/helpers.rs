//! Shared helpers used by every `track_*.rs` file.
//!
//! Two policy families live here:
//!
//! 1. `greedy_threshold_policy` — the forward-mean-vs-spot threshold rule
//! plus a flow-feasibility safety wrapper. Cheap and broadly OK; still
//! used by the harder tracks while their structural tactic lands.
//! 2. `vt_value_function_policy` — per-battery backward-induction DP
//! `V_t(soc)` over the deterministic DA prices, then a per-step greedy
//! over an action grid that maximises `rt_profit(u) + V_{t+1}(soc')`.
//! This is the litreview tactic #1 (Zheng-Jaworski-Xu 2021) faithfully
//! using a standard discretisation (soc_levels=201, action_grid=30,
//! linear interpolation in the next-state lookup).
//!
//! The DP is *per-battery, decoupled*. Network feasibility is enforced
//! after the per-battery proposal via the same flow-shrink wrapper used by
//! the threshold policy. This is the right starting structure: it matches
//! is correct on loosely-constrained tracks (baseline) and leaves room
//! for an ASCA-style coordinated dispatch to land on the network-tight
//! tracks (congested/dense/multiday/capstone).

use anyhow::{anyhow, Result};
use std::cell::RefCell;
use tig_challenges::energy_arbitrage::*;

/// Per-track knobs for the threshold policy. Each `track_*.rs` constructs
/// one of these with its own tuned values and then calls
/// [`greedy_threshold_policy`].
#[derive(Debug, Clone, Copy)]
pub struct TrackConfig {
 /// Number of forward DA steps used to compute the per-node mean
 /// price the spot RT price is compared against.
 pub lookahead_steps: usize,
 /// Charge if `rt_price < forward_mean - max(forward_mean*alpha, MIN_PROFIT_SPREAD)`.
 /// Larger alpha means stricter buying (bigger discount required).
 pub alpha: f64,
 /// Discharge if `rt_price > forward_mean + max(forward_mean*beta, MIN_PROFIT_SPREAD)`.
 /// Larger beta means stricter selling.
 pub beta: f64,
 /// After clamping to `state.action_bounds`, multiply the action by
 /// this factor before flow check. Use < 1.0 on the heavily-congested
 /// tracks to bias toward feasibility.
 pub action_scale: f64,
 /// When the proposed action fails the flow check, retry at this
 /// fraction of the original magnitude. We do up to 3 retries
 /// (action_scale, action_scale * shrink_factor, ...) before
 /// returning zeros.
 pub shrink_factor: f64,
}

/// Per-track knobs for the V_t value-function policy.
#[derive(Debug, Clone, Copy)]
pub struct VtConfig {
 /// Number of SOC discretisation levels per battery, uniform between
 /// `soc_min_mwh` and `soc_max_mwh`.
 pub soc_levels: usize,
 /// Number of candidate actions per SOC cell, uniform between
 /// `u_min(soc)` and `u_max(soc)`.
 pub action_grid: usize,
 /// Multiply the chosen action by this factor before the flow check.
 /// Use < 1.0 on the tighter tracks; 1.0 on baseline.
 pub action_scale: f64,
 /// On flow violation, shrink by this factor and retry. Up to 3
 /// retries before falling back to zeros (always feasible).
 pub shrink_factor: f64,
 /// If true, build `V_t` via K=5 Gauss-Hermite stochastic DP over the
 /// RT-price noise distribution (option value of state-contingent
 /// decisions), rather than a deterministic DP against the DA path.
 /// Captures the upside that the DA-only DP averages away. Matches
 /// `use_sdp=true` configuration. Costs ~5× the DP build
 /// time; runtime dispatch unchanged.
 pub use_sdp: bool,
 /// Multiplicative premium on the sell-side DA price used by the
 /// deterministic-DP branch. Ignored when `use_sdp = true` because the
 /// GH quadrature already integrates over the jump-augmented
 /// distribution. Set to 0.0 to disable.
 pub jump_premium: f64,
 /// Multiplicative derating on the battery's nameplate power
 /// (applied to both charge and discharge limits, at DP build AND at
 /// runtime). Used on tight-network tracks to make the DP plan for
 /// actions that are actually deliverable through the binding lines.
 /// 1.00 baseline, 0.22 congested, 0.10 dense/multiday,
 /// 1.00 capstone. Set to 1.0 to disable.
 pub network_derating: f64,
 /// Bake an anticipated-LMP congestion premium into the per-step
 /// price seen by the DP. When the exogenous-only flow on a line
 /// exceeds `lmp_threshold * limit`, batteries whose PTDF impact on
 /// that line is *opposite-sign* to the binding direction see a
 /// price *boost* (encouraging them to discharge there and relieve
 /// the constraint); same-sign batteries see a price *cut*. Off on
 /// baseline (lines don't bind), on for everything else in.
 pub anticipate_lmp: bool,
 /// Utilisation ratio above which a line is considered "congested"
 /// for the LMP-anticipation premium. 0.85 baseline,
 /// 0.65-0.80 for the tighter tracks (catch congestion earlier when
 /// the network has less slack).
 pub lmp_threshold: f64,
 /// Multiplier on the base premium (`GAMMA_PRICE = 20.0 $/MWh`) used
 /// when a line is congested. 0.45 baseline, 1.00 capstone.
 pub lmp_premium_scale: f64,
 /// Run ASCA (asynchronous coordinate-ascent) sweeps at runtime to
 /// refine the per-battery grid-computed action toward a joint
 /// optimum that accounts for currently-binding line constraints.
 /// Off on baseline (no binding lines); on for tighter tracks.
 pub use_asca: bool,
 /// Number of ASCA sweeps. 15 baseline, 25 congested,
 /// 35-45 multiday/dense, 60 capstone. Diminishing returns past
 /// ~20 on most tracks.
 pub asca_iters: usize,
 /// Stop ASCA early when the max absolute action change across one
 /// sweep is below this threshold. 1e-3 to 1e-4.
 pub convergence_tol: f64,
 /// Safety margin subtracted from each line's flow limit when
 /// computing the feasible-interval intersection inside ASCA. Helps
 /// the dispatched actions survive numerical-precision verification
 /// of the SDK's `verify_flows`. 1e-3 to 1e-4.
 pub flow_margin: f64,
 /// Number of deflator sweeps to rescue any post-ASCA flow violations
 /// before falling back to uniform scaling. 15-50. Deflator
 /// per-iteration: for each violated line, identify culprit batteries
 /// (those whose action contributes in the binding direction), sort
 /// by ROI ascending, trim them until safe. Smarter than uniform
 /// shrink because low-ROI batteries get penalised first.
 pub deflator_iters: usize,
 /// Number of Lagrangian dual-ascent iterations to run *after* the
 /// 2x(ASCA, deflator) pair, as a joint-dispatch refinement.
 /// Each iter: compute shadow prices for near-binding lines, augment
 /// per-battery rt_price by `-sum_l mu_l * ptdf[l][b]`, re-pick action
 /// via analytic argmax, update flows, repeat. attempt at
 /// closing the multiday LP/CG gap without a full QP solver.
 pub lp_iters: usize,
 /// Initial step size for the Lagrangian dual update (subgradient).
 /// Per-iter alpha = lp_step_size / sqrt(k+1). Per-track tunable
 /// because larger networks need smaller steps to avoid dual-price
 /// oscillation in the non-smooth dual.
 pub lp_step_size: f64,
 /// Heavy-ball momentum coefficient. mu[k+1] = mu[k] + alpha*grad
 /// + momentum*(mu[k]-mu[k-1]). 0.0 disables. 0.5-0.9 typical.
 /// Hill-climb safeguard catches any divergence.
 pub lp_momentum: f64,
}

// ── SDK constants we depend on. Mirror of
// `src/energy_arbitrage/constants.rs` in the tig-challenges crate. Kept
// inline so we don't have to import the (sometimes-non-pub) module path
// and so the policy code is self-contained.
const DELTA_T: f64 = 0.25;
const KAPPA_TX: f64 = 0.25;
const KAPPA_DEG: f64 = 1.0;
const BETA_DEG: f64 = 2.0;

/// Absolute minimum price spread ($/MWh) below which a trade does not
/// cover its own transaction + degradation cost. See header in the
/// threshold-policy section of this file for the derivation.
const MIN_PROFIT_SPREAD: f64 = 1.0;

// ────────────────────────────────────────────────────────────────────
// Threshold policy (legacy, broad fallback)
// ────────────────────────────────────────────────────────────────────

/// Run a greedy threshold policy with `cfg` and return the resulting
/// [`Solution`].
pub fn greedy_threshold_policy(
 challenge: &Challenge,
 cfg: TrackConfig,
) -> Result<Solution> {
 challenge.grid_optimize(&move |challenge: &Challenge, state: &State| {
 compute_action_threshold(challenge, state, &cfg)
 })
}

fn compute_action_threshold(
 challenge: &Challenge,
 state: &State,
 cfg: &TrackConfig,
) -> Result<Vec<f64>> {
 if state.time_step >= challenge.num_steps {
 return Err(anyhow!("time_step out of horizon"));
 }
 let n_bat = challenge.num_batteries;
 let mut action = vec![0.0_f64; n_bat];
 for (i, battery) in challenge.batteries.iter().enumerate() {
 let node = battery.node;
 let rt = state.rt_prices[node];
 let forward_mean = forward_da_mean(challenge, state.time_step, node, cfg.lookahead_steps);
 let buy_spread = (forward_mean * cfg.alpha).max(MIN_PROFIT_SPREAD);
 let sell_spread = (forward_mean * cfg.beta).max(MIN_PROFIT_SPREAD);
 let (u_min, u_max) = state.action_bounds[i];
 if rt < forward_mean - buy_spread {
 action[i] = u_min * cfg.action_scale;
 } else if rt > forward_mean + sell_spread {
 action[i] = u_max * cfg.action_scale;
 }
 }
 shrink_until_feasible(challenge, state, action, cfg.shrink_factor)
}

fn forward_da_mean(
 challenge: &Challenge,
 t: usize,
 node: usize,
 lookahead: usize,
) -> f64 {
 let h = challenge.num_steps;
 let start = (t + 1).min(h);
 let end = (start + lookahead).min(h);
 if start >= end {
 return challenge.market.day_ahead_prices[t.min(h - 1)][node];
 }
 let mut sum = 0.0;
 for s in start..end {
 sum += challenge.market.day_ahead_prices[s][node];
 }
 sum / (end - start) as f64
}

// ────────────────────────────────────────────────────────────────────
// V_t value-function policy (the structural tactic for)
// ────────────────────────────────────────────────────────────────────

/// Per-battery `V_t(soc)` table: backward-induction value over the
/// deterministic DA price path at the battery's own node.
///
/// Layout: `values[t][soc_idx]` for `t in 0..=num_steps` and
/// `soc_idx in 0..soc_levels`. `values[num_steps][..]` is identically 0
/// (terminal). Built backward.
#[derive(Debug, Clone)]
pub struct VtTable {
 pub soc_min: f64,
 pub soc_max: f64,
 pub soc_levels: usize,
 /// `(num_steps + 1) * soc_levels` flat row-major.
 pub values: Vec<f64>,
}

impl VtTable {
 #[inline]
 fn idx(&self, t: usize, soc_idx: usize) -> usize {
 t * self.soc_levels + soc_idx
 }

 /// Linear interpolation of `V_t(soc)` between adjacent SOC grid
 /// cells. SOC values below/above the grid clamp to the endpoint.
 #[inline]
 fn lookup(&self, t: usize, soc: f64) -> f64 {
 let n = self.soc_levels;
 debug_assert!(n >= 2);
 if soc <= self.soc_min {
 return self.values[self.idx(t, 0)];
 }
 if soc >= self.soc_max {
 return self.values[self.idx(t, n - 1)];
 }
 let frac = (soc - self.soc_min) / (self.soc_max - self.soc_min);
 let pos = frac * (n - 1) as f64;
 let lo = pos.floor() as usize;
 let hi = (lo + 1).min(n - 1);
 let w = pos - lo as f64;
 let v_lo = self.values[self.idx(t, lo)];
 let v_hi = self.values[self.idx(t, hi)];
 v_lo * (1.0 - w) + v_hi * w
 }
}

/// Pre-computed per-battery scalars used by `dp_analytic_max`. Hoisted
/// out of the inner loop so the per-cell work is reduced to the analytic
/// argmax + 1-3 evaluations.
#[derive(Debug, Clone, Copy)]
struct BatScratch {
 cap: f64,
 soc_min: f64,
 soc_max: f64,
 soc_span: f64,
 eff_c: f64,
 eff_d: f64,
 /// Quadratic coefficient on `u^2` in the per-step profit: `(dt / cap)^2`.
 /// Matches the SDK's `kappa_deg * (|u|*dt/cap)^beta` with
 /// `kappa_deg = 1`, `beta = 2`, so `deg_cost = deg_coeff * u^2`.
 deg_coeff: f64,
}

impl BatScratch {
 fn from(b: &Battery) -> Self {
 let cap = b.capacity_mwh.max(1e-9);
 let soc_span = (b.soc_max_mwh - b.soc_min_mwh).max(1e-9);
 Self {
 cap,
 soc_min: b.soc_min_mwh,
 soc_max: b.soc_max_mwh,
 soc_span,
 eff_c: b.efficiency_charge,
 eff_d: b.efficiency_discharge.max(1e-9),
 deg_coeff: (DELTA_T / cap).powi(2),
 }
 }
}

/// Linear-interp lookup of `V_t(soc)` from a flat slice, mirroring
/// `VtTable::lookup` but parameterised on the (already-extracted) slice
/// for the `t+1` row. Hot path.
#[inline]
fn lookup_slice(v_next: &[f64], soc: f64, sc: &BatScratch) -> f64 {
 let n = v_next.len();
 if n == 0 {
 return 0.0;
 }
 if n == 1 || soc <= sc.soc_min {
 return v_next[0];
 }
 if soc >= sc.soc_max {
 return v_next[n - 1];
 }
 let pos = (soc - sc.soc_min) / sc.soc_span * (n - 1) as f64;
 let lo = pos.floor() as usize;
 let hi = (lo + 1).min(n - 1);
 let w = pos - lo as f64;
 v_next[lo] * (1.0 - w) + v_next[hi] * w
}

/// Analytic per-branch quadratic argmax + 1-3 evaluations. Matches
/// `dp_analytic_max` (sub_t49). Replaces the grid inner-max
/// in `build_vt_table` and (the runtime sibling) `compute_action_vt`.
///
/// Returns `(value, argmax)`. The argmax is consumed by the runtime
/// dispatcher; the DP build only needs `value`.
///
/// Derivation: per-step profit on each branch is `u * dt * c_b - deg * u²`
/// (`c_b` is the linear coefficient for the branch: `p_sell - 0.25` on
/// discharge, `-(p_buy + 0.25)` on charge — signs of u handled via the
/// sign convention u<0 charge / u>0 discharge). Future value is
/// linearised: `V_{t+1}(soc') ≈ V_{t+1}(soc) - λ · Δsoc(u)`. The combined
/// objective on each branch is concave quadratic in u, so the argmax is
/// closed-form. Always evaluates u=0 and the boundary too — handles the
/// "interior optimum is infeasible" case.
#[inline]
fn dp_analytic_max(
 sc: &BatScratch,
 p_buy: f64,
 p_sell: f64,
 soc: f64,
 u_min: f64,
 u_max: f64,
 v_next: &[f64],
) -> (f64, f64) {
 // λ = dV_{t+1}/dSOC at current soc (one-sided slope on the SOC grid).
 let n = v_next.len();
 let lambda = if n > 1 {
 let pos = (soc - sc.soc_min) / sc.soc_span * (n - 1) as f64;
 let lo = (pos.floor() as isize).max(0) as usize;
 let lo = lo.min(n - 2);
 let delta_soc = sc.soc_span / (n - 1) as f64;
 (v_next[lo + 1] - v_next[lo]) / delta_soc.max(1e-12)
 } else {
 0.0
 };

 let eval = |u: f64| -> f64 {
 let price = if u > 0.0 { p_sell } else { p_buy };
 let abs_u = u.abs();
 let profit = u * price * DELTA_T - KAPPA_TX * abs_u * DELTA_T - sc.deg_coeff * u * u;
 let ns_raw = if u < 0.0 {
 soc + sc.eff_c * (-u) * DELTA_T
 } else {
 soc - u / sc.eff_d * DELTA_T
 };
 let ns = ns_raw.clamp(sc.soc_min, sc.soc_max);
 profit + lookup_slice(v_next, ns, sc)
 };

 let mut best_v = eval(0.0);
 let mut best_u = 0.0;

 // Charge branch: u in [u_min, 0).
 if u_min < 0.0 {
 let u_hi = 0.0_f64.min(u_max);
 if u_min < u_hi {
 // f(u) = u·dt·(p_buy + κ_tx − λ·η_c) − deg·u² (with u<0)
 // df/du = dt·(p_buy + κ_tx − λ·η_c) − 2·deg·u = 0
 // u* = dt·(p_buy + κ_tx − λ·η_c) / (2·deg); charging only if u* < 0
 // (equivalently `b_c > 0` in sign convention).
 let b_c = DELTA_T * (lambda * sc.eff_c - p_buy - KAPPA_TX);
 let x_star = if sc.deg_coeff > 1e-30 {
 b_c / (2.0 * sc.deg_coeff)
 } else {
 -u_min
 };
 // Clamp: x_star ∈ [0, -u_min], cand = -x_star ∈ [u_min, 0].
 let cand = (-x_star.clamp(0.0, -u_min)).clamp(u_min, u_hi);
 let v = eval(cand);
 if v > best_v {
 best_v = v;
 best_u = cand;
 }
 let v = eval(u_min);
 if v > best_v {
 best_v = v;
 best_u = u_min;
 }
 }
 }

 // Discharge branch: u in (0, u_max].
 if u_max > 0.0 {
 let u_lo = 0.0_f64.max(u_min);
 if u_lo < u_max {
 let b_d = DELTA_T * (p_sell - KAPPA_TX - lambda / sc.eff_d);
 let x_star = if sc.deg_coeff > 1e-30 {
 b_d / (2.0 * sc.deg_coeff)
 } else {
 u_max
 };
 let cand = x_star.clamp(u_lo, u_max);
 let v = eval(cand);
 if v > best_v {
 best_v = v;
 best_u = cand;
 }
 let v = eval(u_max);
 if v > best_v {
 best_v = v;
 best_u = u_max;
 }
 }
 }

 (best_v, best_u)
}

/// Per-step profit for one battery taking action `u` at price `price`,
/// matching `Challenge::compute_profit` per-battery term exactly.
#[inline]
fn battery_profit(u: f64, price: f64, capacity_mwh: f64) -> f64 {
 if u == 0.0 {
 return 0.0;
 }
 let revenue = u * price * DELTA_T;
 let abs_u = u.abs();
 let tx_cost = KAPPA_TX * abs_u * DELTA_T;
 let deg_base = (abs_u * DELTA_T) / capacity_mwh;
 let deg_cost = KAPPA_DEG * deg_base.powf(BETA_DEG);
 revenue - tx_cost - deg_cost
}

/// Apply action `u` to SOC, mirroring `Battery::apply_action_to_soc`
/// without needing the (private-ish) method path on the battery struct.
#[inline]
fn next_soc(u: f64, soc: f64, b: &Battery) -> f64 {
 let c = (-u).max(0.0);
 let d = u.max(0.0);
 let new_soc = soc + b.efficiency_charge * c * DELTA_T - d * DELTA_T / b.efficiency_discharge;
 new_soc.clamp(b.soc_min_mwh, b.soc_max_mwh)
}

/// Feasible per-battery action bounds at a given SOC, mirroring
/// `Battery::compute_action_bounds` (network coupling ignored here — the
/// V_t DP is per-battery; flow feasibility is enforced at execution).
///
/// `derating` scales the nameplate-power cap (not the SOC-derived cap),
/// matching `max_pwr_c = bat.power_charge_mw * hp.network_derating`.
#[inline]
fn battery_action_bounds(b: &Battery, soc: f64, derating: f64) -> (f64, f64) {
 let headroom = (b.soc_max_mwh - soc).max(0.0);
 let available = (soc - b.soc_min_mwh).max(0.0);
 let max_charge_from_soc = if b.efficiency_charge > 0.0 {
 headroom / (b.efficiency_charge * DELTA_T)
 } else {
 0.0
 };
 let max_discharge_from_soc = if b.efficiency_discharge > 0.0 {
 available * b.efficiency_discharge / DELTA_T
 } else {
 0.0
 };
 let max_pwr_c = b.power_charge_mw * derating;
 let max_pwr_d = b.power_discharge_mw * derating;
 let max_charge = max_charge_from_soc.min(max_pwr_c).max(0.0);
 let max_discharge = max_discharge_from_soc.min(max_pwr_d).max(0.0);
 (-max_charge, max_discharge)
}

// ── K=5 Gauss-Hermite quadrature over the standard normal (probabilist
// convention: nodes/weights normalised so `sum_k w_k * f(z_k)` integrates
// `E[f(Z)]` for `Z ~ N(0, 1)`). K=5 is the stable equilibrium with the
// downstream ASCA + deflator dispatch; finer quadrature destabilises
// the joint dispatch on tight networks.
const GH5_Z: [f64; 5] = [0.0, 0.9586, -0.9586, 2.0202, -2.0202];
const GH5_W: [f64; 5] = [0.5333, 0.2221, 0.2221, 0.0113, 0.0113];

// `GAMMA_PRICE` from the SDK constants — base size of the congestion
// premium ($/MWh). Inlined to keep this file self-contained; mirror of
// `src/energy_arbitrage/constants.rs:GAMMA_PRICE`.
const GAMMA_PRICE: f64 = 20.0;

/// Build a flow vector for "one battery at node b injecting 1 MW", with
/// slack-bus balancing applied (matching `Challenge::compute_total_injections`
/// without needing a `State` instance — only battery-node + slack-bus
/// indices are needed for the marginal probe).
fn unit_injection_for_battery(num_nodes: usize, b_node: usize, slack_bus: usize) -> Vec<f64> {
 let mut inj = vec![0.0_f64; num_nodes];
 if b_node != slack_bus {
 inj[b_node] = 1.0;
 inj[slack_bus] = -1.0;
 }
 inj
}

/// Sparse per-line list of `(battery_idx, impact)` where impact is the
/// flow change on the line when one MW is injected at the battery's
/// node. Linear network → impact is independent of the operating point,
/// so this is computed once per challenge. Sparse because most batteries
/// affect only a few lines (electrical distance).
fn compute_ptdf_sparse(challenge: &Challenge) -> Vec<Vec<(usize, f64)>> {
 let num_l = challenge.network.flow_limits.len();
 let num_b = challenge.num_batteries;
 let num_n = challenge.network.num_nodes;
 let slack = challenge.network.slack_bus;
 let mut ptdf: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_l];
 for b in 0..num_b {
 let inj = unit_injection_for_battery(num_n, challenge.batteries[b].node, slack);
 let flows = challenge.network.compute_flows(&inj);
 for l in 0..num_l {
 if flows[l].abs() > 1e-8 {
 ptdf[l].push((b, flows[l]));
 }
 }
 }
 ptdf
}

/// Per-`(timestep, battery)` LMP-anticipation price shift, $/MWh. Built
/// once per challenge after the PTDF. 
/// `nodal_shift = -impact · sign(f_exo) · base · saturation`, summed over
/// congested lines, with `base = GAMMA_PRICE · lmp_premium_scale` and
/// `saturation = clamp((ratio - threshold) / (1 - threshold), 0, 1)`.
///
/// Returns a zero matrix if `cfg.anticipate_lmp` is false (caller can
/// just skip the lookup, but having it zero-valued lets the inner loop
/// unconditionally add the premium without branching).
fn compute_expected_premiums(
 challenge: &Challenge,
 cfg: &VtConfig,
 ptdf_sparse: &[Vec<(usize, f64)>],
) -> Vec<Vec<f64>> {
 let num_t = challenge.num_steps;
 let num_b = challenge.num_batteries;
 let num_l = challenge.network.flow_limits.len();
 let num_n = challenge.network.num_nodes;
 let slack = challenge.network.slack_bus;
 let mut premiums = vec![vec![0.0_f64; num_b]; num_t];
 if !cfg.anticipate_lmp || num_l == 0 {
 return premiums;
 }
 let base_premium = GAMMA_PRICE * cfg.lmp_premium_scale;
 let thr = cfg.lmp_threshold;
 let denom = (1.0 - thr).max(1e-6);
 for t in 0..num_t {
 // Build the exogenous-only injection vector with slack balancing.
 let mut inj = challenge.exogenous_injections[t].clone();
 // The SDK zeros out the slack-bus exog before adding it, then
 // sets slack = -sum(non-slack). Replicate that so the flow we
 // compute matches what the simulator sees with action = 0.
 if slack < num_n {
 inj[slack] = 0.0;
 }
 let mut sum = 0.0;
 for i in 0..num_n {
 if i != slack {
 sum += inj[i];
 }
 }
 if slack < num_n {
 inj[slack] = -sum;
 }
 let f_exo = challenge.network.compute_flows(&inj);
 for l in 0..num_l {
 let limit = challenge.network.flow_limits[l];
 if limit <= 1e-6 {
 continue;
 }
 let ratio = f_exo[l].abs() / limit;
 if ratio > thr {
 let sat = ((ratio - thr) / denom).clamp(0.0, 1.0);
 let premium = base_premium * sat;
 let sign_f = f_exo[l].signum();
 for &(b, impact) in &ptdf_sparse[l] {
 if impact.abs() > 1e-6 {
 premiums[t][b] += -impact * sign_f * premium;
 }
 }
 }
 }
 }
 premiums
}

/// Effective RT-noise standard deviation (unitless multiplier on the DA
/// price), `σ_eff = √(σ² + jump_var)` per market-derived
/// derivation. `jump_var` uses the Pareto tail-index closed form when
/// `α > 2` (finite variance), with a fall-back constant for heavier tails.
fn sdp_sigma_eff(challenge: &Challenge) -> f64 {
 let sigma = challenge.market.params.volatility;
 let rho_j = challenge.market.params.jump_probability;
 let alpha_j = challenge.market.params.tail_index;
 let jump_var = if alpha_j > 2.0 {
 rho_j * alpha_j / (alpha_j - 2.0)
 } else {
 rho_j * 4.0
 };
 (sigma * sigma + jump_var).sqrt()
}

/// Build the per-battery `V_t(soc)` DP table by backward induction over
/// the DA price sequence at the battery's own node.
///
/// Two flavours, controlled by `cfg.use_sdp`:
/// - **Deterministic** (`use_sdp=false`): single per-step price equal to
/// the DA value (plus an optional sell-side `jump_premium` multiplier).
/// - **K=5 Gauss-Hermite SDP** (`use_sdp=true`): treats the per-step price
/// as `p_da * (1 + σ_eff * Z)` with `Z ~ N(0, 1)` and integrates via
/// a 5-point quadrature. Crucially, the inner `max_u` is done *per
/// scenario k* before averaging — this is what captures the option
/// value (be aggressive when realised price is extreme, hold when it
/// isn't) that the deterministic DP averages away.
/// `premium` is the per-timestep LMP premium ($/MWh) for *this* battery,
/// same length as `challenge.num_steps`. Added into the DP's per-step
/// price.
fn build_vt_table(
 challenge: &Challenge,
 battery: &Battery,
 cfg: &VtConfig,
 premium: &[f64],
) -> VtTable {
 let h = challenge.num_steps;
 let n = cfg.soc_levels;
 let mut table = VtTable {
 soc_min: battery.soc_min_mwh,
 soc_max: battery.soc_max_mwh,
 soc_levels: n,
 values: vec![0.0; (h + 1) * n],
 };
 // Precompute the SOC grid values once.
 let soc_grid: Vec<f64> = (0..n)
 .map(|i| {
 let frac = i as f64 / (n - 1) as f64;
 table.soc_min + (table.soc_max - table.soc_min) * frac
 })
 .collect();

 let sc = BatScratch::from(battery);
 let sigma_eff = if cfg.use_sdp { sdp_sigma_eff(challenge) } else { 0.0 };

 // Backward induction. Terminal value V_{H}(soc) = 0 (already zero).
 // Inner max-over-u is analytic per branch (`dp_analytic_max`): 1-5
 // evaluations per cell instead of `action_grid`. Quality equals a
 // very-fine grid (the per-branch objective is concave quadratic);
 // compute saving is the bandwidth that unlocks the LP joint dispatch.
 for t in (0..h).rev() {
 let p_da = challenge.market.day_ahead_prices[t][battery.node] + premium[t];
 for soc_idx in 0..n {
 let soc = soc_grid[soc_idx];
 let (u_min, u_max) = battery_action_bounds(battery, soc, cfg.network_derating);
 // Borrow the t+1 row as a flat slice once; lookup_slice and
 // dp_analytic_max both read it.
 let row_start = (t + 1) * n;
 let row_end = row_start + n;
 // Split-borrow: we read from row t+1 and write into row t,
 // which is a different chunk, so this is safe without aliasing.

 let best = if cfg.use_sdp {
 // SDP: dp_analytic_max PER SCENARIO, then weighted sum.
 // Option value lives in the per-scenario argmax happening
 // before the weighted average.
 let mut val_sum = 0.0_f64;
 for k in 0..GH5_Z.len() {
 let p_k = (p_da * (1.0 + sigma_eff * GH5_Z[k])).max(1e-6);
 let v_next = &table.values[row_start..row_end];
 let (v_k, _u_k) =
 dp_analytic_max(&sc, p_k, p_k, soc, u_min, u_max, v_next);
 val_sum += GH5_W[k] * v_k;
 }
 val_sum
 } else {
 // Deterministic DP. Sell-side bumped by jump_premium when
 // u > 0 (matching non-SDP branch).
 let p_sell = p_da * (1.0 + cfg.jump_premium);
 let p_buy = p_da;
 let v_next = &table.values[row_start..row_end];
 let (v, _u) =
 dp_analytic_max(&sc, p_buy, p_sell, soc, u_min, u_max, v_next);
 v
 };

 let idx = table.idx(t, soc_idx);
 table.values[idx] = best;
 }
 }
 table
}

/// Bundled per-challenge precomputation shared between the DP build and
/// the runtime dispatcher. PTDF + inverse map (b_to_lines) lets ASCA do
/// flow-update + per-battery feasible-interval computation in
/// `O(b_to_lines[battery])` rather than `O(num_lines)`.
struct VtCache {
 tables: Vec<VtTable>,
 ptdf_sparse: Vec<Vec<(usize, f64)>>,
 b_to_lines: Vec<Vec<(usize, f64)>>,
}

/// Run the V_t value-function policy: pre-build per-battery DP tables,
/// then dispatch per-step via greedy `argmax_u [rt_profit(u) + V_{t+1}(soc')]`,
/// optionally refined by ASCA sweeps.
pub fn vt_value_function_policy(
 challenge: &Challenge,
 cfg: VtConfig,
) -> Result<Solution> {
 let ptdf_sparse = compute_ptdf_sparse(challenge);
 // Invert ptdf to b_to_lines: per-battery list of (line_idx, impact).
 let num_b = challenge.num_batteries;
 let mut b_to_lines: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_b];
 for (l, entries) in ptdf_sparse.iter().enumerate() {
 for &(b, impact) in entries {
 b_to_lines[b].push((l, impact));
 }
 }
 let premiums = compute_expected_premiums(challenge, &cfg, &ptdf_sparse);
 let tables: Vec<VtTable> = challenge
 .batteries
 .iter()
 .enumerate()
 .map(|(b_idx, b)| {
 let per_battery_premium: Vec<f64> =
 (0..challenge.num_steps).map(|t| premiums[t][b_idx]).collect();
 build_vt_table(challenge, b, &cfg, &per_battery_premium)
 })
 .collect();
 let cache = VtCache { tables, ptdf_sparse, b_to_lines };
 // Warm-start dual prices: persists across timesteps. Decayed each
 // step (factor=0.7) so stale congestion patterns don't dominate.
 let num_l = challenge.network.flow_limits.len();
 let warm_mu: RefCell<Vec<f64>> = RefCell::new(vec![0.0_f64; num_l]);
 challenge.grid_optimize(&move |challenge: &Challenge, state: &State| {
 compute_action_vt(challenge, state, &cache, &cfg, &warm_mu)
 })
}

fn compute_action_vt(
 challenge: &Challenge,
 state: &State,
 cache: &VtCache,
 cfg: &VtConfig,
 warm_mu: &RefCell<Vec<f64>>,
) -> Result<Vec<f64>> {
 let tables = &cache.tables;
 if state.time_step >= challenge.num_steps {
 return Err(anyhow!("time_step out of horizon"));
 }
 let t = state.time_step;
 let t_next = t + 1;
 let n_bat = challenge.num_batteries;
 let mut action = vec![0.0_f64; n_bat];

 let action_denom = (cfg.action_grid - 1).max(1) as f64;

 for i in 0..n_bat {
 let battery = &challenge.batteries[i];
 let table = &tables[i];
 let soc = state.socs[i];
 let rt_price = state.rt_prices[battery.node];
 let (full_u_min, full_u_max) = state.action_bounds[i];
 let u_min = full_u_min * cfg.network_derating;
 let u_max = full_u_max * cfg.network_derating;

 // Grid scan seeds the per-battery action. The grid-based seed
 // outperforms an analytic-only seed on tight networks because
 // ASCA is a coordinate-ascent (path-dependent) and the grid
 // local optimum tends to a slightly better basin.
 let v_idle = table.lookup(t_next, soc);
 let mut best_score = v_idle;
 let mut chosen_u = 0.0;
 if cfg.action_grid >= 2 && u_max - u_min > 1e-12 {
 let span = u_max - u_min;
 for ag in 0..cfg.action_grid {
 let u = u_min + span * (ag as f64 / action_denom);
 let ns = next_soc(u, soc, battery);
 let candidate = battery_profit(u, rt_price, battery.capacity_mwh)
 + table.lookup(t_next, ns);
 if candidate > best_score {
 best_score = candidate;
 chosen_u = u;
 }
 }
 }

 // Clamp into the *full* `state.action_bounds` to defend against
 // float-precision overshoot. (The derated u_min/u_max already
 // sit strictly inside the full bounds when derating < 1, so the
 // clamp is a no-op there.)
 let scaled = chosen_u * cfg.action_scale;
 action[i] = scaled.clamp(full_u_min, full_u_max);
 }

 // ASCA + deflator + Lagrangian LP interleaved. Each deflator pass
 // trims ASCA's residual flow violations; the next ASCA pass
 // redistributes the freed flow budget toward higher-ROI batteries.
 // Three outer-loop iterations is the convergence sweet spot.
 if cfg.use_asca && cfg.asca_iters > 0 {
 for _ in 0..3 {
 run_asca(challenge, state, cache, cfg, &mut action, false);
 if cfg.deflator_iters > 0 {
 run_deflator(challenge, state, cache, cfg, &mut action);
 }
 if cfg.lp_iters > 0 {
 run_lp_lagrangian(challenge, state, cache, cfg, &mut action, warm_mu);
 if cfg.deflator_iters > 0 {
 run_deflator(challenge, state, cache, cfg, &mut action);
 }
 }
 }
 // Final ASCA polish after the converged loops catches anything
 // LP-driven that coordinate-ascent can still improve.
 run_asca(challenge, state, cache, cfg, &mut action, false);
 if cfg.deflator_iters > 0 {
 run_deflator(challenge, state, cache, cfg, &mut action);
 }
 } else if cfg.deflator_iters > 0 {
 run_deflator(challenge, state, cache, cfg, &mut action);
 }

 shrink_until_feasible(challenge, state, action, cfg.shrink_factor)
}

/// One-step ASCA refinement. Updates `actions` in place. Starts from
/// the per-battery dispatch passed in (already from the grid scan),
/// computes line flows under those actions, then iterates: for each
/// battery, intersect its action interval with the per-line feasibility
/// half-plane left over after the *other* batteries' flow
/// contributions, find the best interior point via analytic argmax on
/// each branch, commit it, update flows incrementally. Stops early on
/// convergence.
fn run_asca(
 challenge: &Challenge,
 state: &State,
 cache: &VtCache,
 cfg: &VtConfig,
 actions: &mut [f64],
 reverse: bool,
) {
 let num_b = challenge.num_batteries;
 let num_l = challenge.network.flow_limits.len();
 if num_b == 0 || num_l == 0 {
 return;
 }
 let t = state.time_step;
 let t_next = t + 1;

 // Initial flows = exo (from current state) + battery contributions
 // from the seed `actions`. Compute via the SDK so signs match exactly.
 let injections = challenge.compute_total_injections(state, actions);
 let mut flows: Vec<f64> = challenge.network.compute_flows(&injections);

 // Sort batteries: high potential (room to move profitably) and low
 // footprint (impact on congested lines) go first. Heuristic that
 // makes single-sweep ASCA converge close to multi-sweep optimum.
 let footprint = |b: usize| -> f64 {
 let mut fp = 1e-4;
 for &(l, p) in &cache.b_to_lines[b] {
 let limit = challenge.network.flow_limits[l];
 if limit > 1e-6 {
 let util = flows[l].abs() / limit;
 fp += p.abs() * util.powi(2) * 10.0;
 }
 }
 fp
 };
 let potential = |b: usize, table: &VtTable| -> f64 {
 let (u_lo, u_hi) = state.action_bounds[b];
 let battery = &challenge.batteries[b];
 let sc = BatScratch::from(battery);
 let rt_price = state.rt_prices[battery.node];
 let n = table.soc_levels;
 let row_start = t_next.min(challenge.num_steps) * n;
 let v_next = &table.values[row_start..row_start + n];
 let v0 = eval_runtime(state.socs[b], 0.0, rt_price, &sc, v_next);
 let v_lo = eval_runtime(state.socs[b], u_lo, rt_price, &sc, v_next);
 let v_hi = eval_runtime(state.socs[b], u_hi, rt_price, &sc, v_next);
 (v_lo.max(v_hi) - v0).max(0.0)
 };
 let mut order: Vec<usize> = (0..num_b).collect();
 order.sort_by(|&a, &b| {
 let sa = potential(a, &cache.tables[a]) / footprint(a);
 let sb = potential(b, &cache.tables[b]) / footprint(b);
 sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
 });
 if reverse {
 order.reverse();
 }

 // Main sweeps.
 for _sweep in 0..cfg.asca_iters {
 let mut max_change = 0.0_f64;
 for &b in &order {
 let battery = &challenge.batteries[b];
 let table = &cache.tables[b];
 // ASCA uses the *full* `state.action_bounds`, not the derated
 // planning bounds. Derating shapes what V_t plans for; ASCA's
 // job is to push beyond that envelope where the network
 // actually has room. The V_t value at the resulting soc' is
 // still queryable via lookup_slice (the table spans the full
 // SOC range, only the assumed *action* envelope was derated).
 let (u_min_full, u_max_full) = state.action_bounds[b];
 let mut u_min = u_min_full;
 let mut u_max = u_max_full;

 // Tighten by per-line feasibility, removing this battery's
 // own contribution from each line's residual flow.
 for &(l, p) in &cache.b_to_lines[b] {
 if p.abs() < 1e-9 {
 continue;
 }
 let limit = (challenge.network.flow_limits[l] - cfg.flow_margin).max(0.0);
 let f_other = flows[l] - p * actions[b];
 let b1 = (-limit - f_other) / p;
 let b2 = (limit - f_other) / p;
 let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
 if lo > u_min {
 u_min = lo;
 }
 if hi < u_max {
 u_max = hi;
 }
 }
 // Empty interval: hold the current action. Otherwise ensure
 // the current action is always inside (it must be — we got
 // here from a feasible-or-zero seed).
 if u_min > u_max {
 u_min = actions[b];
 u_max = actions[b];
 } else {
 u_min = u_min.min(actions[b]);
 u_max = u_max.max(actions[b]);
 }

 let sc = BatScratch::from(battery);
 let rt_price = state.rt_prices[battery.node];
 let n = table.soc_levels;
 let row_start = t_next.min(challenge.num_steps) * n;
 let v_next = &table.values[row_start..row_start + n];

 // Analytic argmax under the tightened bounds. Same `eval_runtime`
 // as the seed dispatch — V_t-grounded plus exact eval.
 let (_v, best_u) =
 dp_analytic_max(&sc, rt_price, rt_price, state.socs[b], u_min, u_max, v_next);

 // Compare against the current action explicitly (analytic
 // can sometimes pick a marginally-worse interior than the
 // current point when the linearisation overshoots).
 let v_cur = eval_runtime(state.socs[b], actions[b], rt_price, &sc, v_next);
 let v_new = eval_runtime(state.socs[b], best_u, rt_price, &sc, v_next);
 let chosen = if v_new > v_cur { best_u } else { actions[b] };
 let chosen = chosen.clamp(u_min, u_max);

 let delta = chosen - actions[b];
 if delta.abs() > 1e-9 {
 actions[b] = chosen;
 for &(l, p) in &cache.b_to_lines[b] {
 flows[l] += p * delta;
 }
 if delta.abs() > max_change {
 max_change = delta.abs();
 }
 }
 }
 if max_change < cfg.convergence_tol {
 break;
 }
 }
}

/// Deflator — selective flow-violation rescue. For each over-limit line,
/// identify culprit batteries (those contributing in the binding
/// direction), sort by ROI ascending, and trim them in order until safe.
/// Falls back to uniform β-scaling (the largest β that makes everything
/// feasible) if the iterative trim doesn't converge in `deflator_iters`.
/// Matches `run_deflator` structurally.
fn run_deflator(
 challenge: &Challenge,
 state: &State,
 cache: &VtCache,
 cfg: &VtConfig,
 actions: &mut [f64],
) {
 let num_l = challenge.network.flow_limits.len();
 let num_b = challenge.num_batteries;
 if num_l == 0 || num_b == 0 {
 return;
 }
 // Initial flows under the current `actions` (post-ASCA).
 let injections = challenge.compute_total_injections(state, actions);
 let mut flows: Vec<f64> = challenge.network.compute_flows(&injections);

 // Precompute the per-battery "value of current action vs zero", used
 // for ROI sorting. Stable across the inner loop because soc
 // and rt_price don't change.
 let mut value_curr = vec![0.0_f64; num_b];
 let mut value_zero = vec![0.0_f64; num_b];
 let t = state.time_step;
 let t_next = (t + 1).min(challenge.num_steps);
 for b in 0..num_b {
 let battery = &challenge.batteries[b];
 let table = &cache.tables[b];
 let sc = BatScratch::from(battery);
 let rt_price = state.rt_prices[battery.node];
 let n = table.soc_levels;
 let row_start = t_next * n;
 let v_next = &table.values[row_start..row_start + n];
 value_curr[b] = eval_runtime(state.socs[b], actions[b], rt_price, &sc, v_next);
 value_zero[b] = eval_runtime(state.socs[b], 0.0, rt_price, &sc, v_next);
 }

 let mut is_safe = true;
 for _ in 0..cfg.deflator_iters {
 is_safe = true;
 for l in 0..num_l {
 let limit = (challenge.network.flow_limits[l] - cfg.flow_margin).max(0.0);
 if flows[l].abs() <= limit {
 continue;
 }
 is_safe = false;
 let overflow = flows[l].abs() - limit;
 let sign = flows[l].signum();

 // Culprits: batteries contributing in the same direction as
 // the over-limit flow. Each entry is `(b, |contribution|, roi)`.
 let mut culprits: Vec<(usize, f64, f64)> = Vec::new();
 for &(b, impact) in &cache.ptdf_sparse[l] {
 let contrib = impact * actions[b];
 if contrib * sign > 1e-9 {
 let denom = actions[b].abs().max(1.0);
 let roi = ((value_curr[b] - value_zero[b]).max(0.0)) / denom;
 culprits.push((b, contrib.abs(), roi));
 }
 }
 culprits.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

 let mut remaining = overflow;
 for (b, contrib_abs, _roi) in culprits {
 if remaining <= 1e-9 {
 break;
 }
 if contrib_abs < 1e-12 {
 continue;
 }
 let reduction = contrib_abs.min(remaining);
 let ratio = 1.0 - reduction / contrib_abs;
 let new_action = actions[b] * ratio;
 let delta = new_action - actions[b];
 actions[b] = new_action;
 for &(ll, pp) in &cache.b_to_lines[b] {
 flows[ll] += pp * delta;
 }
 // Update this battery's value-at-current-action after the trim.
 let battery = &challenge.batteries[b];
 let table = &cache.tables[b];
 let sc = BatScratch::from(battery);
 let rt_price = state.rt_prices[battery.node];
 let n = table.soc_levels;
 let row_start = t_next * n;
 let v_next = &table.values[row_start..row_start + n];
 value_curr[b] = eval_runtime(state.socs[b], actions[b], rt_price, &sc, v_next);
 remaining -= reduction;
 }
 }
 if is_safe {
 break;
 }
 }
 if is_safe {
 return;
 }

 // Fallback: uniform β-scale by the largest β that makes everything
 // feasible. Then clamp into action_bounds.
 let injections = challenge.compute_total_injections(state, actions);
 let flows_total: Vec<f64> = challenge.network.compute_flows(&injections);
 // f_base = exo-only flow at this state. compute by zero-action probe.
 let zero = vec![0.0_f64; num_b];
 let inj_base = challenge.compute_total_injections(state, &zero);
 let flows_base: Vec<f64> = challenge.network.compute_flows(&inj_base);
 let f_act: Vec<f64> = (0..num_l).map(|l| flows_total[l] - flows_base[l]).collect();

 let mut beta = 1.0_f64;
 for l in 0..num_l {
 let limit = (challenge.network.flow_limits[l] - cfg.flow_margin).max(0.0);
 let total = flows_base[l] + f_act[l];
 if total.abs() <= limit {
 continue;
 }
 if f_act[l].abs() < 1e-9 {
 continue;
 }
 let target = if total > 0.0 { limit } else { -limit };
 let candidate = (target - flows_base[l]) / f_act[l];
 if candidate < beta {
 beta = candidate;
 }
 }
 let beta = beta.clamp(0.0, 1.0);
 for b in 0..num_b {
 actions[b] *= beta;
 let (lo, hi) = state.action_bounds[b];
 if actions[b] < lo {
 actions[b] = lo;
 }
 if actions[b] > hi {
 actions[b] = hi;
 }
 }
}

/// Lagrangian dual-ascent for joint dispatch. Runs after the
/// 2x(ASCA, deflator) pair. Each iteration:
/// 1. Compute current line flows
/// 2. For each *near-binding* line, increase the dual price `μ_l` toward
/// the value that makes the line exactly at-limit
/// 3. For each battery, recompute its analytic argmax with the rt_price
/// augmented by `-sum_l (μ_l^+ − μ_l^-) · ptdf[l][b]` (so a battery
/// that pushes flow in the binding direction sees a lower effective
/// price and pulls back)
/// 4. Update flows incrementally with the action deltas
///
/// This is the simplest joint-dispatch refinement; not a full LP solve,
/// but captures cross-battery coordination that ASCA's sequential
/// commits miss when binding lines couple distant batteries.
fn run_lp_lagrangian(
 challenge: &Challenge,
 state: &State,
 cache: &VtCache,
 cfg: &VtConfig,
 actions: &mut [f64],
 warm_mu: &RefCell<Vec<f64>>,
) {
 let num_b = challenge.num_batteries;
 let num_l = challenge.network.flow_limits.len();
 if num_b == 0 || num_l == 0 {
 return;
 }
 let t_next = (state.time_step + 1).min(challenge.num_steps);

 // Current flows under the starting actions.
 let injections = challenge.compute_total_injections(state, actions);
 let mut flows: Vec<f64> = challenge.network.compute_flows(&injections);

 // Dual prices: μ_l for upper-limit violation, μ_l_neg for lower.
 // Stored as signed `mu_signed_l = μ_l^+ − μ_l^-`. Sub-gradient step.
 // Warm-start from previous timestep's converged duals, decayed.
 let mut mu_signed: Vec<f64> = {
 let prev = warm_mu.borrow();
 prev.iter().map(|m| m * 0.9).collect()
 };
 let step_size = cfg.lp_step_size; // per-track tunable
 // heavy-ball momentum on the dual update. Hill-climb
 // safeguard catches any divergence (worst case: reverts to ASCA).
 let momentum = cfg.lp_momentum;
 let mut mu_prev = mu_signed.clone();

 // Compute total profit (per-battery sum) for a given actions vector.
 let compute_total = |acts: &[f64]| -> f64 {
 let mut s = 0.0;
 for b in 0..num_b {
 let battery = &challenge.batteries[b];
 let table = &cache.tables[b];
 let sc = BatScratch::from(battery);
 let rt_price = state.rt_prices[battery.node];
 let n = table.soc_levels;
 let row_start = t_next * n;
 let v_next = &table.values[row_start..row_start + n];
 s += eval_runtime(state.socs[b], acts[b], rt_price, &sc, v_next);
 }
 s
 };

 // Snapshot the starting actions and total — we only commit changes
 // that strictly improve the joint objective.
 let starting_actions: Vec<f64> = actions.to_vec();
 let starting_total = compute_total(actions);

 for k in 0..cfg.lp_iters {
 let alpha = step_size / ((k + 1) as f64).sqrt();
 // Update dual prices from current flow violations.
 let mut any_binding = false;
 for l in 0..num_l {
 let limit = (challenge.network.flow_limits[l] - cfg.flow_margin).max(0.0);
 let excess = flows[l].abs() - limit;
 let prev = mu_signed[l];
 if excess > 0.0 {
 any_binding = true;
 // Heavy-ball: gradient step + momentum from last delta.
 mu_signed[l] = mu_signed[l]
 + alpha * excess * flows[l].signum()
 + momentum * (mu_signed[l] - mu_prev[l]);
 } else {
 // Decay μ when the line is not binding (avoids drifting
 // dual prices on lines that aren't actually constraints).
 // Momentum still applies (preserves direction memory).
 mu_signed[l] = mu_signed[l] * 0.5
 + momentum * (mu_signed[l] - mu_prev[l]);
 }
 mu_prev[l] = prev;
 }
 if !any_binding && k > 0 {
 break;
 }

 // Re-solve each battery with augmented price.
 for b in 0..num_b {
 let battery = &challenge.batteries[b];
 let table = &cache.tables[b];
 let (full_u_min, full_u_max) = state.action_bounds[b];
 let u_min = full_u_min;
 let u_max = full_u_max;
 let sc = BatScratch::from(battery);
 let soc = state.socs[b];
 let rt_raw = state.rt_prices[battery.node];

 // Augment rt_price by the sum of dual-price contributions
 // across lines this battery affects. Sign convention:
 // if battery action u > 0 (discharge) and ptdf[l][b] > 0,
 // the battery pushes flow positive on line l. A positive
 // μ_signed_l (upper bound binding) means we want to pull
 // back from positive flow, so reduce the effective price.
 let mut price_shift = 0.0;
 for &(l, impact) in &cache.b_to_lines[b] {
 price_shift += mu_signed[l] * impact;
 }
 let rt_aug = rt_raw - price_shift;

 let n = table.soc_levels;
 let row_start = t_next * n;
 let v_next = &table.values[row_start..row_start + n];
 let (_v, best_u) =
 dp_analytic_max(&sc, rt_aug, rt_aug, soc, u_min, u_max, v_next);

 // Evaluate at AUGMENTED price plus actual current; pick best.
 let v_cur = eval_runtime(soc, actions[b], rt_aug, &sc, v_next);
 let v_new = eval_runtime(soc, best_u, rt_aug, &sc, v_next);
 let chosen = if v_new > v_cur { best_u } else { actions[b] };
 let chosen = chosen.clamp(u_min, u_max);

 let delta = chosen - actions[b];
 if delta.abs() > 1e-9 {
 actions[b] = chosen;
 for &(l, p) in &cache.b_to_lines[b] {
 flows[l] += p * delta;
 }
 }
 }
 }

 // Hill-climb safeguard: if the Lagrangian made the joint objective
 // worse (which it can do when it abandons a coordinated ASCA
 // dispatch for a per-battery one), revert to the starting actions.
 let ending_total = compute_total(actions);
 if ending_total < starting_total {
 actions.copy_from_slice(&starting_actions);
 } else {
 // Save converged mu for next timestep's warm-start (only when
 // the iteration actually improved — otherwise the mu was bad).
 let mut warm = warm_mu.borrow_mut();
 warm.clear();
 warm.extend_from_slice(&mu_signed);
 }
}

/// Runtime per-battery evaluation: `profit(u) + V_{t+1}(soc')`. Used by
/// ASCA's tie-breaker and footprint potential. Same formula as the
/// `eval` closure inside `dp_analytic_max`, exposed as a named function
/// because ASCA needs it from multiple call sites.
#[inline]
fn eval_runtime(soc: f64, u: f64, price: f64, sc: &BatScratch, v_next: &[f64]) -> f64 {
 let abs_u = u.abs();
 let profit = u * price * DELTA_T - KAPPA_TX * abs_u * DELTA_T - sc.deg_coeff * u * u;
 let ns_raw = if u < 0.0 {
 soc + sc.eff_c * (-u) * DELTA_T
 } else {
 soc - u / sc.eff_d * DELTA_T
 };
 let ns = ns_raw.clamp(sc.soc_min, sc.soc_max);
 profit + lookup_slice(v_next, ns, sc)
}

// ────────────────────────────────────────────────────────────────────
// Flow-feasibility wrapper, shared by both policy families.
// ────────────────────────────────────────────────────────────────────

/// If the proposed `action` violates any line flow, shrink by
/// `shrink_factor` repeatedly until feasible (max 4 attempts). On total
/// failure return zeros (always feasible).
fn shrink_until_feasible(
 challenge: &Challenge,
 state: &State,
 action: Vec<f64>,
 shrink_factor: f64,
) -> Result<Vec<f64>> {
 let mut scale = 1.0_f64;
 for _ in 0..4 {
 let scaled: Vec<f64> = action.iter().map(|a| a * scale).collect();
 if is_feasible(challenge, state, &scaled) {
 return Ok(scaled);
 }
 scale *= shrink_factor;
 }
 Ok(vec![0.0; action.len()])
}

fn is_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
 let injections = challenge.compute_total_injections(state, action);
 let flows = challenge.network.compute_flows(&injections);
 challenge.network.verify_flows(&flows).is_ok()
}
