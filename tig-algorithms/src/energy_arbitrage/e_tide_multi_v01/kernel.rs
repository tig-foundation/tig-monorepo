
//! Bounded deterministic online policy for a finite battery/network model.
//!
//! Positive actions discharge into the grid; negative actions charge. This is
//! a local decision model, not a claim about physical-grid safety or forecast
//! accuracy. Every returned action is checked against battery and linearized
//! network bounds, with monotone backoff to the always-feasible zero action.
//!
//! Lineage (structure stolen, not monofile paste):
//! - energy_v1: scarcity directional value, substitute/soften, post-feasibility
//!   re-expand, grouped pos/neg scale rescue, terminal liquidation.
//! - prometheus_eb1 / prior_art–v6: multi-seed portfolio, quantile charge/discharge
//!   thresholds, residual EMA shift, congestion soft duals, edge-sized intensity.

// ── Deterministic work meter ────────────────────────────────────────────────
//
// Wall time is not a usable proxy on a shared machine: measured under load 27,
// `lookahead_horizon` 8 timed SLOWER than 24, which is impossible for a cost
// dial and is pure contention noise.
//
// TIG meters instructions (`__fuel_remaining`, exit 87). Measured on the
// metered path: baseline burns ~10.0e9 of the 100e9 budget and congested
// ~21e9, while multiday/dense/capstone exhaust it on every nonce. To make
// those fit we need a work measure that is exact and load-independent.
//
// This counts the two operations that dominate: per-step policy evaluations
// and network-feasibility backoff iterations. It is a PROXY for instruction
// count, not the instruction count itself — the exchange rate must still be
// calibrated against a metered build, exactly as the knapsack rate
// (~23-81 instr/comparison, track-dependent) had to be.
use std::cell::{Cell, RefCell};

/// Submit package: process env knobs are compile-frozen (public surface defaults only).
#[inline(always)]
fn gnosis_env_var(_key: &str) -> Result<String, ()> {
    Err(())
}

thread_local! {
    static WORK_UNITS: Cell<u64> = const { Cell::new(0) };
    /// Per-line (fwd, rev) dual warm-start for rolling-horizon LP seed.
    /// Reset when `time == 0` so nonces don't leak state across solves.
    static RH_LAM_WARM: RefCell<Option<(Vec<f64>, Vec<f64>)>> = const { RefCell::new(None) };
}

/// Reset the work meter. Call before a solve.
pub fn reset_work_meter() {
    WORK_UNITS.with(|c| c.set(0));
}

/// Work units spent since the last reset.
#[must_use]
pub fn work_units() -> u64 {
    WORK_UNITS.with(Cell::get)
}

#[inline]
fn charge_work(units: u64) {
    WORK_UNITS.with(|c| c.set(c.get().saturating_add(units)));
}

/// Spec time step (hours). Official SOC/power conversion uses this.
pub const DELTA_T: f64 = 0.25;
/// Spec transaction friction ($/MWh).
const KAPPA_TX: f64 = 0.25;
/// Spec degradation scale ($).
const KAPPA_DEG: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Battery {
    pub node: usize,
    /// Usable upper SoC bound (`soc_max_mwh` = 0.90 x nominal).
    pub capacity: f64,
    /// Nameplate capacity (`capacity_mwh`). The evaluator's degradation term
    /// divides by THIS, not by `capacity`. Using `capacity` overstates
    /// degradation by (1/0.9)^2 = 1.2346x, which understates profit and breaks
    /// the upper-bound property of the clairvoyant DP.
    pub nominal_capacity: f64,
    pub max_charge: f64,
    pub max_discharge: f64,
    pub charge_efficiency: f64,
    pub discharge_efficiency: f64,
    pub reserve_fraction: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub limit: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnergyChallenge {
    /// Per-battery day-ahead value functions, built ONCE per solve. See
    /// `build_dp_cache`.
    pub dp_cache: Vec<BatteryDp>,
    /// `day_ahead_prices[time][node]`.
    pub day_ahead_prices: Vec<Vec<f64>>,
    pub batteries: Vec<Battery>,
    pub lines: Vec<Line>,
    /// Linearized line-flow sensitivity: `flow = base + PTDF * action_by_node`.
    pub ptdf: Vec<Vec<f64>>,
    pub residual_weight: f64,
    pub deadband: f64,
    pub max_backoffs: usize,
    /// Lookahead steps for quantile thresholds (prometheus/titan style).
    pub lookahead_horizon: usize,
    /// Soft congestion dual scale on tight base lines.
    pub congestion_weight: f64,
    /// Friction scale for tx costs in directional scoring.
    pub friction_weight: f64,
    /// titan `composite_wv` fleet series: `delta_cong[t]` added to ∂V/∂soc in
    /// the online PGA gradient (`cwv_lambda` already baked in). Empty = off.
    pub delta_cong: Vec<f64>,
    /// titan t51 L8 SoC reference trajectory `soc_ref[b][t]` (SoC *before*
    /// step `t`). Empty = off. Static fill via [`compute_soc_reference_static`];
    /// dynamic override is built per-step in [`policy`] when dyn is on.
    pub soc_ref: Vec<Vec<f64>>,
    /// Weight on the L8 reference-tracking term in the PGA gradient.
    /// `grad[b] -= λ·(next_soc − soc_ref[b][t+1])·∂soc/∂u`. Default 0 (off).
    pub soc_ref_lambda: f64,
    /// When true, recompute SoC-ref each step from current SoC + residual shift
    /// (titan t51 L8b). Static `soc_ref` is the fallback / t=0 seed only.
    pub soc_ref_dynamic: bool,
}

impl Default for EnergyChallenge {
    fn default() -> Self {
        Self {
            dp_cache: Vec::new(),
            day_ahead_prices: Vec::new(),
            batteries: Vec::new(),
            lines: Vec::new(),
            ptdf: Vec::new(),
            residual_weight: 0.15,
            deadband: 0.02,
            max_backoffs: 64,
            lookahead_horizon: 24,
            congestion_weight: 0.35,
            friction_weight: 1.0,
            delta_cong: Vec::new(),
            soc_ref: Vec::new(),
            soc_ref_lambda: 0.0,
            soc_ref_dynamic: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OnlineState {
    pub time: usize,
    pub state_of_charge: Vec<f64>,
    pub observed_prices: Vec<f64>,
    pub base_line_flows: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyOutcome {
    Action(Vec<f64>),
    Withheld(PolicyError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyError {
    InvalidChallenge(&'static str),
    InvalidState(&'static str),
}

/// Track-shaped defaults from (num_batteries, horizon) — prior_art dispatch idea.
/// **DIAGNOSTIC BISECT — not a promotion.**
///
/// Three tracks (multiday/dense/capstone, all >30 batteries) exit 87 on the
/// metered path. Cutting `lookahead_horizon` 24->6 removed 67-71% of the work
/// by our own meter and moved real fuel by ~nothing, so the meter is blind to
/// the dominant cost — and two recalibrations of it disagree by 4.6x.
///
/// Stop proxying. Backoffs are the only per-step term scaling with BOTH network
/// size and iteration count: each iteration runs a full PTDF `compute_flows`
/// pass. `GNOSIS_BISECT_BACKOFFS` cuts them hard on the large tracks so the
/// metered path answers directly.
///
/// * passes   -> the backoff loop IS the cost; remedy is a tighter feasibility search
/// * exits 87 -> cost is structural in grid_optimize/compute_flows; no parameter reaches it
///
/// Quality is expected to suffer (less feasibility repair). Acceptable: the only
/// output of a bisect is the exit code.
fn bisect_backoffs() -> Option<usize> {
    gnosis_env_var("GNOSIS_BISECT_BACKOFFS")
        .ok()
        .and_then(|v| v.parse().ok())
}

pub fn track_params(num_batteries: usize, horizon: usize) -> (f64, f64, usize, f64, f64) {
    // residual_weight, deadband, max_backoffs, congestion_weight, friction_weight
    let bisect = bisect_backoffs();
    let params = match (num_batteries, horizon) {
        // Softening residual REGRESSES baseline: paired n=24, rw 0.42/0.35/0.28
        // give -18.7/-27.5/-23.8% quality at t = -3.08/-3.31/-2.57. The old
        // "(resn1) -45% -- hold" call was right, but only the downhill side had
        // ever been probed. HARDENING is monotone better across 0.58..1.00, and
        // confirmed paired at n=48: rw=0.85 -> +12.4% (t=2.41, 31/48 nonces),
        // rw=1.00 -> +15.0% (t=2.62). 0.85 and 1.00 are indistinguishable; 0.85
        // is interior, wins on more nonces, and keeps a DA hedge, so take it.
        //
        // NOTE this track has quality CV ~98%, so an UNPAIRED test here cannot
        // resolve anything under ~56% even at n=24. Pairing, not sample size, is
        // what makes it measurable -- never tune this track unpaired.
        (n, h) if n <= 15 && h <= 96 => (0.85, 0.02, 80, 0.20, 1.0), // baseline
        // Hardened with baseline: rw 0.40 -> 0.85 gives +6.0% quality,
        // paired n=24, t=+2.10, CI [+0.0012,+0.0340], wins 14/24.
        (n, h) if n <= 30 && h <= 96 => (0.85, 0.03, 96, 0.50, 1.0), // congested
        // rw 0.45 -> 0.85 gives +1.4% quality, paired n=24, t=+2.01,
        // CI [+0.0008,+0.0609], wins 16/24. Smaller gain than baseline because
        // this track's margin over greedy is 68% -- low leverage.
        (n, h) if n <= 50 => (0.85, 0.02, 72, 0.30, 1.0),            // multiday
        // rw 0.40 -> 0.85: +3.3% quality, paired n=22, t=+5.64,
        // CI [+0.0311,+0.0642], wins 20/22 -- the most decisive of the five.
        (n, h) if n <= 80 => (0.85, 0.03, 64, 0.35, 1.05),           // dense
        // rw 0.35 -> 0.85: +8.2% quality, paired n=24, t=+7.13,
        // CI [+0.1107,+0.1947], wins 22/24. Completes the sweep: all five
        // tracks improve with the same parameter in the same direction.
        _ => (0.85, 0.04, 48, 0.40, 1.1),                            // capstone-ish
    };
    // POSITIVE CONTROL: threshold >10, not >30.
    //
    // The first bisect applied only to >30-battery tracks -- which are exactly
    // the tracks that exit 87 and emit NO output. "Still exit 87" was therefore
    // equally consistent with "backoffs are not the cost" and "the knob never
    // applied". That is the dead-knob trap: an arm that changes neither the
    // outcome nor the cost cannot be told from one that did not run.
    //
    // At >10, `congested` (30 batteries) is included. It COMPLETES and reports
    // fuel_consumed = 20.83e9 / 21.72e9. If that number moves, the mechanism
    // demonstrably works and the big-track result is a real negative. If it is
    // byte-identical, the bisect never applied and all of it was void.
    match bisect {
        Some(c) if num_batteries > 10 => (params.0, params.1, c, params.3, params.4),
        _ => params,
    }
}

/// Produce a finite, validated action for the current time step.
pub fn policy(challenge: &EnergyChallenge, state: &OnlineState) -> PolicyOutcome {
    if let Err(error) = validate(challenge, state) {
        if matches!(
            error,
            PolicyError::InvalidState("infeasible base network flow")
        ) {
            return PolicyOutcome::Action(vec![0.0; challenge.batteries.len()]);
        }
        return PolicyOutcome::Withheld(error);
    }

    let node_count = state.observed_prices.len();
    let shadows = line_shadows(challenge, state);
    let residual_shift = residual_shift_vector(challenge, state);
    // titan t51 L8b: dynamic SoC-ref from current SoC + residual-shifted DA.
    // Built each step when enabled (O(n·T) — free vs PGA). Passed as override
    // into PGA so we never need interior mutability on the challenge.
    let dyn_soc_ref = if challenge.soc_ref_lambda > 0.0 && challenge.soc_ref_dynamic {
        Some(compute_soc_reference_dynamic(
            &challenge.batteries,
            &challenge.day_ahead_prices,
            &state.state_of_charge,
            &residual_shift,
            state.time,
        ))
    } else {
        None
    };
    let soc_ref_override = dyn_soc_ref.as_deref();

    // Primary: per-battery DA value-function DP (prometheus/titan portable core).
    // Network coupling via project_feasible + post-improve. Multi-seed portfolio.
    // NOTE: extra residual-blend seeds / myopic polish were tried and regressed
    // multi-step total profit on baseline smoke — keep this portfolio tight.
    // DP BISECT. The portfolio cut (4 candidates -> 1) saved only 1.0-3.7%, but
    // k=1 KEEPS the DP seed, so `build_dp_actions` was never excluded. It is the
    // last large per-step component not yet measured: everything else is now
    // bounded (harness 0.25%, lookahead 0.29%, backoffs 1.7-5.3%, three
    // candidate generators + projections 1.0-3.7%, per-step rebuild 0.003%,
    // sparse product 0.07-0.17%).
    let n_bat = challenge.batteries.len();
    // Titan dense ships `use_dp_seed=false`. S1d lib n=8: default no-DP on dense
    // **−1.12%** — do not ship. Opt-in `GNOSIS_NO_DP_SEED=1` only.
    // `GNOSIS_SKIP_DP=1` zeroes DP build entirely (fuel bisect).
    let include_dp_seed = match gnosis_env_var("GNOSIS_NO_DP_SEED")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("1") | Some("true") | Some("on") => false,
        _ => true,
    };
    let skip_dp_build = gnosis_env_var("GNOSIS_SKIP_DP")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    let dp_seed = if skip_dp_build || !include_dp_seed {
        vec![0.0; challenge.batteries.len()]
    } else {
        build_dp_actions(challenge, state, &residual_shift)
    };
    let scarcity_da =
        build_scarcity_actions(challenge, state, &shadows, 0.0, ResidualWeightMode::DaOnly);
    let price_rank = build_price_rank_actions(challenge, state, &shadows);
    let zero = vec![0.0; challenge.batteries.len()];
    // Zero seed. titan dense ships `use_zero_seed=false`. Default ON (historical
    // portfolio) until measured under stage. Force off: `GNOSIS_ZERO_SEED=0`.
    let include_zero_seed = match gnosis_env_var("GNOSIS_ZERO_SEED")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        _ => true,
    };
    // Portfolio seeds. Dense/cap (n>50): also RT-blend scarcity — pure-DA
    // scarcity is 72% of multiday quality but dense residual is network/RT
    // heavy; an extra seed is free diversity under keep=4. Multiday stays
    // pure-DA portfolio (blend seed not added). Gate OFF: GNOSIS_SCARCITY_RT=0.
    // When DP seed is dropped (dense default), lead with scarcity_da so keep
    // bias still prefers a real action seed (idx 0 gets +1e-6).
    let mut candidates: Vec<Vec<f64>> = if include_dp_seed && !skip_dp_build {
        if include_zero_seed {
            vec![dp_seed, scarcity_da, price_rank, zero.clone()]
        } else {
            vec![dp_seed, scarcity_da, price_rank]
        }
    } else if include_zero_seed {
        vec![scarcity_da, price_rank, zero.clone()]
    } else {
        vec![scarcity_da, price_rank]
    };
    // Bule: no dense-only RT scarcity portfolio (crown seed diversity).
    let scarcity_rt_on = n_bat > 50
        && !surface_bule()
        && !matches!(
            gnosis_env_var("GNOSIS_SCARCITY_RT")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("0") | Some("false") | Some("off")
        );
    if scarcity_rt_on {
        let rw = challenge.residual_weight.max(0.0);
        candidates.push(build_scarcity_actions(
            challenge,
            state,
            &shadows,
            rw,
            ResidualWeightMode::Blend,
        ));
        // Full-RT scarcity seed (rw=1): denser residual signal than default
        // residual_weight (0.85). stage: dense n=16 **+1.14%** (10/16). Cap
        // lib n=8 **−0.36%** → dense-only default (`50 < n ≤ 80`). Force:
        // `GNOSIS_SCARCITY_FULL_RT=0|1`.
        let full_rt = match gnosis_env_var("GNOSIS_SCARCITY_FULL_RT")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("0") | Some("false") | Some("off") => false,
            Some("1") | Some("true") | Some("on") => true,
            // Bule: never full-RT dense seed.
            _ => !surface_bule() && n_bat > 50 && n_bat <= 80,
        };
        if full_rt && (rw - 1.0).abs() > 1e-9 {
            candidates.push(build_scarcity_actions(
                challenge,
                state,
                &shadows,
                1.0,
                ResidualWeightMode::Blend,
            ));
        }
    }

    // PORTFOLIO BISECT.
    //
    // The null-policy control established that 99.97% of fuel is ours and that
    // every track completes when the policy returns immediately -- so exit 87
    // is entirely our per-step cost. Five earlier probes accounted for ~6%.
    //
    // This is the coarsest remaining lever: the policy builds FOUR candidate
    // actions per step and runs `project_feasible` +
    // `post_feasibility_improvement` on each. `GNOSIS_BISECT_CANDIDATES=k`
    // keeps only the first k. If fuel falls roughly in proportion, the
    // portfolio structure is the cost; if not, it is inside the generators
    // (`build_dp_actions` in particular) and the bisect must go deeper.
    //
    // Applies to ALL tracks so it is measurable on ones that report fuel.
    //
    // TRACK-SCOPED CANDIDATE KEEP — retuned under legalMax 5e12 (G1 ecdb4647).
    //
    // Historical 100e9 EV forced capstone keep=1 (four candidates exited 87).
    // G1 @5e12: capstone peak ~227e9 (4.5% of legal). S1 ablation (lib n=8):
    //   keep=4 alone on capstone → **+16.8%** quality vs keep=1 control.
    // Dense already keep=4. Baseline/congested/multiday full portfolio.
    // Override: GNOSIS_BISECT_CANDIDATES=k.
    let default_keep = candidates.len();
    let keep = gnosis_env_var("GNOSIS_BISECT_CANDIDATES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_keep)
        .clamp(1, candidates.len());

    // One DA-value cache for ALL candidates. `post_feasibility_improvement`
    // called `directional_value` from inside a battery loop nested in its own
    // `for _ in 0..4` expansion passes, once per candidate -- so
    // O(candidates x 4 x batteries) calls per step, each allocating a fresh Vec
    // of every remaining price at the battery's node. The value keys only on
    // (battery, direction) and on `state`, which is identical across candidates
    // and across the four passes, so there are at most 2 x batteries distinct
    // results per step.
    let mut da_value_cache: Vec<Option<f64>> = vec![None; challenge.batteries.len() * 2];
    let mut projected: Vec<Vec<f64>> = candidates
        .iter_mut()
        .take(keep)
        .map(|raw| project_feasible(challenge, state, raw, node_count))
        .filter(|a| action_is_valid(challenge, state, a, node_count))
        .map(|a| {
            post_feasibility_improvement(challenge, state, a, node_count, &mut da_value_cache)
        })
        .collect();

    // titan t51 rolling-horizon LP seed (K=2 dual-decomp on binding PTDF lines).
    // Default OFF. When ON, projected and scored with the rest of the portfolio;
    // PGA (if on) then polishes the winner. Gate: GNOSIS_RH=1, optional
    // GNOSIS_RH_STRIDE (titan multiday default 3).
    if rh_enabled()
        && !challenge.dp_cache.is_empty()
        && state.time + 1 < challenge.day_ahead_prices.len()
        && rh_step_active(state.time)
    {
        if let Some(rh_raw) = rolling_horizon_lp_seed(challenge, state) {
            let mut rh = rh_raw;
            let rh = project_feasible(challenge, state, &mut rh, node_count);
            if action_is_valid(challenge, state, &rh, node_count) {
                let rh = post_feasibility_improvement(
                    challenge,
                    state,
                    rh,
                    node_count,
                    &mut da_value_cache,
                );
                projected.push(rh);
            }
        }
    }

    if projected.is_empty() {
        return PolicyOutcome::Action(zero);
    }

    // Adaptive DA weight: when RT residual is large, lean harder on DA inventory
    // so residual-heavy nonces do not drain SOC for noise spikes.
    let residual_mag = if residual_shift.is_empty() {
        0.0
    } else {
        residual_shift.iter().map(|x| x.abs()).sum::<f64>() / residual_shift.len() as f64
    };
    // resn1 raised floors REGRESS baseline −45% — hold original thresholds.
    let da_weight = if residual_mag > 8.0 {
        0.72
    } else if residual_mag > 4.0 {
        0.62
    } else {
        0.55
    };

    // Portfolio score. Default: RT step profit + DA inventory weight (pre-PGA).
    // `GNOSIS_PORTFOLIO_DP_SCORE=1`: total_step_dp_value — aligns seed pick with
    // the PGA / network-aware objective (titan joint_optimize scores this way).
    // Default OFF until paired measurement; free relative to DP cache build.
    let use_dp_score = env_flag_on("GNOSIS_PORTFOLIO_DP_SCORE")
        && !challenge.dp_cache.is_empty();
    let score_action = |a: &[f64], idx: usize| -> f64 {
        let bias = if idx == 0 { 1e-6 } else { 0.0 };
        if use_dp_score {
            total_step_dp_value(challenge, state, a) + bias
        } else {
            let da_score = estimated_action_value_da(challenge, state, a, &shadows);
            let rt_score = estimated_step_profit(challenge, state, a);
            rt_score + da_weight * da_score + bias
        }
    };

    // Online projected-gradient polish (titan PGA portable extract).
    // S2 EnergyParetoArchive (Lean `EnergyParetoArchive`): SF already applied
    // (invalid dropped). Non-dom on (quality, residual); Bule diversity face;
    // Buleyean weights for ranking (deterministic argmax — no RNG); BCS pick;
    // multi-PGA over archive only. `GNOSIS_PARETO_ARCHIVE=0` → legacy argmax.
    //
    // MEASURED win on library path (paired n=8, seed gnosis_replay_v1):
    //   multiday  +9.4%  t=+5.40  wins 8/8  work ×1.23
    //   dense     +9.0%  t=+3.95  wins 6/6  work ×1.63
    //   capstone  +13.1% t=+4.92  wins 8/8  work ×1.94
    //
    // Multi-seed PGA: default ON multiday (`pga_multi_enabled`); polish archive.
    let pga_on = pga_enabled(challenge.batteries.len())
        && !challenge.dp_cache.is_empty()
        && pga_step_active(state.time, challenge.batteries.len())
        && pga_congestion_active(challenge, state);
    let multi = pga_on && pga_multi_enabled(challenge.batteries.len());
    let use_archive = pareto_archive_enabled(challenge.batteries.len());

    // Score every projected seed (paths = food sources).
    let scored: Vec<(Vec<f64>, f64, f64, f64)> = projected
        .iter()
        .enumerate()
        .map(|(idx, a)| {
            let q = score_action(a, idx);
            let r = network_residual_score(challenge, state, a, &shadows);
            let d = bule_diversity_face(a);
            (a.clone(), q, r, d)
        })
        .collect();

    // Non-dominated filter on (quality, residual) for **BCS single-seed** path.
    // Multi-PGA keeps the full projected set (employed bees on all food sources);
    // pre-PGA non-dom can drop a seed that PGA would climb past the front
    // (S2 first measure: multiday archive −0.31% vs full multi).
    let archive: Vec<(Vec<f64>, f64, f64, f64)> = if use_archive && !multi {
        let mut nd = Vec::new();
        for (i, si) in scored.iter().enumerate() {
            let dominated = scored.iter().enumerate().any(|(j, sj)| {
                j != i
                    && sj.1 >= si.1 - 1e-15
                    && sj.2 >= si.2 - 1e-15
                    && (sj.1 > si.1 + 1e-15 || sj.2 > si.2 + 1e-15)
            });
            if !dominated {
                nd.push(si.clone());
            }
        }
        if nd.is_empty() {
            scored.clone()
        } else {
            // American Frontier: at-frontier keep all non-dom; post truncate by
            // Bule diversity if archive exceeds path count.
            let paths = scored.len().max(1);
            if nd.len() > paths {
                nd.sort_by(|a, b| {
                    b.3.partial_cmp(&a.3)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            b.1.partial_cmp(&a.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                });
                nd.truncate(paths);
            }
            nd
        }
    } else {
        scored.clone()
    };

    // Buleyean complement weights from void = inverted BCS (deterministic).
    // w = T - min(v,T) + 1; pick argmax w then BCS (onlooker rank, no RNG).
    let bcs = |q: f64, r: f64, d: f64| q + r + d;
    let pick_bcs = |pool: &[(Vec<f64>, f64, f64, f64)]| -> (Vec<f64>, f64) {
        if pool.is_empty() {
            return (zero.clone(), f64::NEG_INFINITY);
        }
        let max_b = pool
            .iter()
            .map(|s| bcs(s.1, s.2, s.3))
            .fold(f64::NEG_INFINITY, f64::max);
        let t_r = (max_b.abs() + 1.0).max(1.0);
        let w_of = |q: f64, r: f64, d: f64| {
            let void = (max_b - bcs(q, r, d)).max(0.0).min(t_r);
            t_r - void + 1.0
        };
        let mut best_i = 0usize;
        let mut best_w = f64::NEG_INFINITY;
        let mut best_b = f64::NEG_INFINITY;
        for (i, (_, q, r, d)) in pool.iter().enumerate() {
            let w = w_of(*q, *r, *d);
            let b = bcs(*q, *r, *d);
            if w > best_w + 1e-15 || ((w - best_w).abs() <= 1e-15 && b > best_b) {
                best_w = w;
                best_b = b;
                best_i = i;
            }
        }
        (pool[best_i].0.clone(), pool[best_i].1)
    };

    let non_dom_filter =
        |scored_in: &[(Vec<f64>, f64, f64, f64)]| -> Vec<(Vec<f64>, f64, f64, f64)> {
            let mut nd = Vec::new();
            for (i, si) in scored_in.iter().enumerate() {
                let dominated = scored_in.iter().enumerate().any(|(j, sj)| {
                    j != i
                        && sj.1 >= si.1 - 1e-15
                        && sj.2 >= si.2 - 1e-15
                        && (sj.1 > si.1 + 1e-15 || sj.2 > si.2 + 1e-15)
                });
                if !dominated {
                    nd.push(si.clone());
                }
            }
            if nd.is_empty() {
                scored_in.to_vec()
            } else {
                nd
            }
        };

    let mut best;
    let mut best_score;
    if multi {
        // Employed: polish every food source (full projected / archive input).
        let mut polished_scored: Vec<(Vec<f64>, f64, f64, f64)> = Vec::new();
        for (idx, (seed, _, _, _)) in archive.iter().enumerate() {
            let mut polished = projected_gradient_polish(
                challenge,
                state,
                seed.clone(),
                node_count,
                &shadows,
                soc_ref_override,
            );
            // Soft-CV (SF flowchart): if polish left the polytope, re-project
            // rather than drop the food source — empty front falls back to
            // pre-PGA BCS and wastes multi-PGA work. Opt-out: GNOSIS_SOFT_CV=0.
            if !action_is_valid(challenge, state, &polished, node_count) {
                if soft_cv_enabled() {
                    polished =
                        project_feasible(challenge, state, &mut polished, node_count);
                }
                if !action_is_valid(challenge, state, &polished, node_count) {
                    continue;
                }
            }
            let q = score_action(&polished, idx);
            let r = network_residual_score(challenge, state, &polished, &shadows);
            let d = bule_diversity_face(&polished);
            polished_scored.push((polished, q, r, d));
        }
        if polished_scored.is_empty() {
            // Fall back to pre-PGA BCS.
            let (a, s) = pick_bcs(&archive);
            best = a;
            best_score = s;
        } else {
            // Onlooker after employed PGA: pick max **quality** among polished
            // (S3 measure: residual BCS on post-PGA front −0.24% multiday).
            // Non-dom front keeps diversity for optional residual-aware mode
            // (`GNOSIS_POST_PGA_BCS=1`); default is pure score (titan-like).
            let front = if use_archive && env_flag_on("GNOSIS_POST_PGA_BCS") {
                non_dom_filter(&polished_scored)
            } else {
                polished_scored
            };
            if env_flag_on("GNOSIS_POST_PGA_BCS") {
                let (a, s) = pick_bcs(&front);
                best = a;
                best_score = s;
            } else {
                // Pure quality max first (SF primary objective).
                best_score = f64::NEG_INFINITY;
                best = front[0].0.clone();
                for (act, q, _, _) in &front {
                    let l1: f64 = act.iter().map(|a| a.abs()).sum();
                    let best_l1: f64 = best.iter().map(|a| a.abs()).sum();
                    if *q > best_score + 1e-12
                        || ((*q - best_score).abs() <= 1e-12 && l1 > best_l1)
                    {
                        best_score = *q;
                        best = act.clone();
                    }
                }
                // ε-constraint (paper feasible-vs-feasible secondary): among
                // polished seeds with quality ≥ (1−ε)·best_q, maximize residual
                // (network headroom). Full residual BCS regressed multiday
                // −0.24%; ε-band also regressed (S4: multi −1.69% @1%).
                // Default ε=0 (pure quality). Opt-in: GNOSIS_EPS_CONSTRAINT=0.01.
                let eps = post_pga_eps();
                if eps > 0.0 && front.len() > 1 {
                    let q_floor = if best_score >= 0.0 {
                        best_score * (1.0 - eps)
                    } else {
                        best_score * (1.0 + eps) // both negative: floor is more negative
                    };
                    let mut best_r = f64::NEG_INFINITY;
                    let mut best_act = best.clone();
                    let mut best_q = best_score;
                    for (act, q, r, _) in &front {
                        if *q + 1e-12 < q_floor {
                            continue;
                        }
                        // Prefer residual; tie-break quality then L1.
                        let l1: f64 = act.iter().map(|a| a.abs()).sum();
                        let bl1: f64 = best_act.iter().map(|a| a.abs()).sum();
                        if *r > best_r + 1e-12
                            || ((*r - best_r).abs() <= 1e-12 && *q > best_q + 1e-12)
                            || ((*r - best_r).abs() <= 1e-12
                                && (*q - best_q).abs() <= 1e-12
                                && l1 > bl1)
                        {
                            best_r = *r;
                            best_q = *q;
                            best_act = act.clone();
                        }
                    }
                    best = best_act;
                    best_score = best_q;
                }
            }
        }
    } else {
        // Single-seed path: optional pre-PGA non-dom, BCS, then one PGA polish.
        let pool = if use_archive {
            non_dom_filter(&archive)
        } else {
            archive.clone()
        };
        let (picked, score) = pick_bcs(&pool);
        best = picked;
        best_score = score;
        if pga_on {
            best = projected_gradient_polish(
                challenge,
                state,
                best,
                node_count,
                &shadows,
                soc_ref_override,
            );
        }
    }
    let _ = best_score;

    // titan t51 coordinate_polish: sequential 1D DP-value search per battery
    // with network residual bounds. Uses total_step_dp_value (PGA objective).
    // Default OFF (S1 stack with multiday-default regresses −7% when combined
    // with higher pair budgets). Opt-in: GNOSIS_COORD_POLISH=1 or PASSES=N.
    // Solo multiday was +1.85–2.6% (`receipt`) — not stacked.
    if let Some(passes) = coord_polish_passes(challenge.batteries.len()) {
        if !challenge.dp_cache.is_empty() {
            best = coordinate_polish(challenge, state, best, node_count, passes);
        }
    }

    // prior_art joint pair-exchange: equal-and-opposite ±α·span probes that
    // independent per-battery seeds cannot see. Budgeted; first-improvement.
    // Only n>30 (multiday/dense/capstone): baseline/congested REGRESS hard
    // (`dist/replay-energy-pair` baseline −114k, congested −88%).
    // Small-n RT+DA coord REGRESS (`dist/replay-energy-coord` base −141k);
    // DA-only coord worse (`dist/replay-energy-coordda` base −270k) — hold
    // portfolio-only for n≤30.
    if challenge.batteries.len() > 30 {
        best = joint_pair_exchange_polish(
            challenge,
            state,
            best,
            node_count,
            da_weight,
            &shadows,
        );
        // prior_art joint_triplet after pair: ±α on i, ∓α on j, +γ on k.
        // Default n>50 (dense/capstone). Multiday n≈40 was −0.33% pre-PGA
        // (`dist/replay-energy-triplet`). Re-measure under PGA via
        // GNOSIS_TRIPLET=1 (enables for all n>30 that already run pair).
        if triplet_enabled(challenge.batteries.len()) {
            best = joint_triplet_exchange_polish(
                challenge,
                state,
                best,
                node_count,
                da_weight,
                &shadows,
            );
        }
    }

    if action_is_valid(challenge, state, &best, node_count) {
        return PolicyOutcome::Action(best);
    }
    PolicyOutcome::Action(zero)
}

/// titan t51 RH dual loop constants (Hindsight-tuned).
const RH_KKT_ITERS: usize = 5;
const RH_KKT_ALPHA0: f64 = 0.5;
const RH_KKT_ALPHA_MAX: f64 = 0.3;
const RH_BINDING_RATIO: f64 = 0.7;
const RH_N_VAR: usize = 4;
const RH_M_CON: usize = 8;
const RH_NV: usize = RH_N_VAR + RH_M_CON; // 12
const RH_NC: usize = RH_NV + 1; // 13
const RH_LP_EPS: f64 = 1e-9;

fn rh_enabled() -> bool {
    env_flag_on("GNOSIS_RH")
}

fn rh_step_active(time: usize) -> bool {
    let stride = gnosis_env_var("GNOSIS_RH_STRIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    time % stride == 0
}

/// Per-battery 4-var K=2 horizon LP (stack tableau simplex). Portable extract
/// of prior_art t51 `solve_battery_kkt`. Returns `[d0, c0, d1, c1]` ≥ 0.
#[must_use]
pub fn solve_battery_kkt(
    f0: f64,
    g0: f64,
    f1: f64,
    g1: f64,
    ub_d0: f64,
    ub_c0: f64,
    ub_d1: f64,
    ub_c1: f64,
    avail: f64,
    head: f64,
    d_f: f64,
    c_f: f64,
) -> [f64; RH_N_VAR] {
    let mut tab = [[0.0_f64; RH_NC]; RH_M_CON + 1];
    // d0 ≤ ub_d0
    tab[0][0] = 1.0;
    tab[0][RH_N_VAR] = 1.0;
    tab[0][RH_NV] = ub_d0;
    // c0 ≤ ub_c0
    tab[1][1] = 1.0;
    tab[1][RH_N_VAR + 1] = 1.0;
    tab[1][RH_NV] = ub_c0;
    // d1 ≤ ub_d1
    tab[2][2] = 1.0;
    tab[2][RH_N_VAR + 2] = 1.0;
    tab[2][RH_NV] = ub_d1;
    // c1 ≤ ub_c1
    tab[3][3] = 1.0;
    tab[3][RH_N_VAR + 3] = 1.0;
    tab[3][RH_NV] = ub_c1;
    // d0*d_f - c0*c_f ≤ avail
    tab[4][0] = d_f;
    tab[4][1] = -c_f;
    tab[4][RH_N_VAR + 4] = 1.0;
    tab[4][RH_NV] = avail;
    // -d0*d_f + c0*c_f ≤ head
    tab[5][0] = -d_f;
    tab[5][1] = c_f;
    tab[5][RH_N_VAR + 5] = 1.0;
    tab[5][RH_NV] = head;
    // (d0+d1)*d_f - (c0+c1)*c_f ≤ avail
    tab[6][0] = d_f;
    tab[6][1] = -c_f;
    tab[6][2] = d_f;
    tab[6][3] = -c_f;
    tab[6][RH_N_VAR + 6] = 1.0;
    tab[6][RH_NV] = avail;
    // -(d0+d1)*d_f + (c0+c1)*c_f ≤ head
    tab[7][0] = -d_f;
    tab[7][1] = c_f;
    tab[7][2] = -d_f;
    tab[7][3] = c_f;
    tab[7][RH_N_VAR + 7] = 1.0;
    tab[7][RH_NV] = head;
    // objective (max → min)
    tab[RH_M_CON][0] = -f0;
    tab[RH_M_CON][1] = -g0;
    tab[RH_M_CON][2] = -f1;
    tab[RH_M_CON][3] = -g1;

    let mut basis = [
        RH_N_VAR,
        RH_N_VAR + 1,
        RH_N_VAR + 2,
        RH_N_VAR + 3,
        RH_N_VAR + 4,
        RH_N_VAR + 5,
        RH_N_VAR + 6,
        RH_N_VAR + 7,
    ];

    for _ in 0..(3 * RH_N_VAR + 2) {
        let mut entering = RH_NV;
        let mut min_c = -RH_LP_EPS;
        for j in 0..RH_NV {
            if tab[RH_M_CON][j] < min_c {
                min_c = tab[RH_M_CON][j];
                entering = j;
            }
        }
        if entering == RH_NV {
            break;
        }
        let mut leaving = RH_M_CON;
        let mut min_r = f64::MAX;
        for i in 0..RH_M_CON {
            if tab[i][entering] > RH_LP_EPS {
                let r = tab[i][RH_NV] / tab[i][entering];
                if r < min_r {
                    min_r = r;
                    leaving = i;
                }
            }
        }
        if leaving == RH_M_CON {
            break;
        }
        let pv = tab[leaving][entering];
        for j in 0..RH_NC {
            tab[leaving][j] /= pv;
        }
        for i in 0..=RH_M_CON {
            if i != leaving {
                let f = tab[i][entering];
                if f.abs() > 1e-15 {
                    for j in 0..RH_NC {
                        tab[i][j] -= f * tab[leaving][j];
                    }
                }
            }
        }
        basis[leaving] = entering;
    }

    let mut sol = [0.0_f64; RH_N_VAR];
    for (i, &bv) in basis.iter().enumerate() {
        if bv < RH_N_VAR {
            sol[bv] = tab[i][RH_NV].max(0.0);
        }
    }
    sol
}

/// Portable extract of prior_art t51 `rolling_horizon_lp_seed`:
/// K=2 rolling-horizon LP via PTDF dual decomposition.
///
/// Per-battery 4-var simplex on (d0,c0,d1,c1) with prices (RT now, DA next) and
/// ∂V/∂soc continuation; dual subgradient on quasi-binding lines
/// (`|base|/limit > 0.7`). Returns step-0 net action `d0 - c0` per battery, or
/// `None` near terminal horizon.
///
/// Dual warm-start is thread-local (reset at t=0). Cost O(RH_KKT_ITERS ·
/// (batteries · binding + binding · batteries)).
#[must_use]
pub fn rolling_horizon_lp_seed(
    challenge: &EnergyChallenge,
    state: &OnlineState,
) -> Option<Vec<f64>> {
    let t = state.time;
    let n_t = challenge.day_ahead_prices.len();
    if t + 1 >= n_t || challenge.batteries.is_empty() {
        return None;
    }
    let n_b = challenge.batteries.len();
    let n_lines = challenge.lines.len();
    let dt = DELTA_T;

    // sens[l][b] = ptdf[l][node_b] (our flow model; base already carries exo).
    let mut sens = vec![vec![0.0_f64; n_b]; n_lines];
    for l in 0..n_lines {
        let row = challenge.ptdf.get(l)?;
        for (b, bat) in challenge.batteries.iter().enumerate() {
            sens[l][b] = row.get(bat.node).copied().unwrap_or(0.0);
        }
    }

    let mut f0 = vec![0.0_f64; n_b];
    let mut g0 = vec![0.0_f64; n_b];
    let mut f1 = vec![0.0_f64; n_b];
    let mut g1 = vec![0.0_f64; n_b];
    let mut available = vec![0.0_f64; n_b];
    let mut headroom = vec![0.0_f64; n_b];
    let mut ub_d0 = vec![0.0_f64; n_b];
    let mut ub_c0 = vec![0.0_f64; n_b];
    let mut ub_d1 = vec![0.0_f64; n_b];
    let mut ub_c1 = vec![0.0_f64; n_b];
    let mut d_f_b = vec![0.0_f64; n_b];
    let mut c_f_b = vec![0.0_f64; n_b];

    for b in 0..n_b {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let p0 = state.observed_prices.get(node).copied().unwrap_or(0.0);
        let p1 = challenge
            .day_ahead_prices
            .get(t + 1)
            .and_then(|row| row.get(node))
            .copied()
            .unwrap_or(0.0);
        let soc0 = state.state_of_charge.get(b).copied().unwrap_or(0.0);
        let eta_d = battery.discharge_efficiency.max(1e-12);
        let eta_c = battery.charge_efficiency.max(1e-12);
        d_f_b[b] = dt / eta_d;
        c_f_b[b] = eta_c * dt;
        let dv2 = challenge
            .dp_cache
            .get(b)
            .map(|dp| dv_dsoc(dp, t + 1, soc0))
            .unwrap_or(0.0);
        let fri = KAPPA_TX * challenge.friction_weight;
        f0[b] = (p0 - fri) * dt;
        g0[b] = -(p0 + fri) * dt;
        f1[b] = (p1 - fri) * dt - dv2 * d_f_b[b];
        g1[b] = -(p1 + fri) * dt + dv2 * c_f_b[b];
        let soc_min = battery.capacity * battery.reserve_fraction;
        available[b] = (soc0 - soc_min).max(0.0);
        headroom[b] = (battery.capacity - soc0).max(0.0);
        let (lo, hi) = power_bounds(battery, soc0);
        ub_d0[b] = hi.max(0.0);
        ub_c0[b] = (-lo).max(0.0);
        ub_d1[b] = battery.max_discharge;
        ub_c1[b] = battery.max_charge;
    }

    let binding_lines: Vec<usize> = challenge
        .lines
        .iter()
        .enumerate()
        .filter(|&(l, line)| {
            line.limit > 1e-6
                && state
                    .base_line_flows
                    .get(l)
                    .copied()
                    .unwrap_or(0.0)
                    .abs()
                    / line.limit
                    > RH_BINDING_RATIO
        })
        .map(|(l, _)| l)
        .collect();

    // Warm duals: reset at t=0; otherwise reuse per-line vectors.
    let (mut lam_warm_fwd, mut lam_warm_rev) = RH_LAM_WARM.with(|cell| {
        let mut slot = cell.borrow_mut();
        if t == 0 || slot.is_none() || slot.as_ref().map(|(a, _)| a.len()) != Some(n_lines) {
            *slot = Some((vec![0.0; n_lines], vec![0.0; n_lines]));
        }
        slot.clone().unwrap()
    });

    let mut lam_fwd: Vec<f64> = binding_lines
        .iter()
        .map(|&l| lam_warm_fwd.get(l).copied().unwrap_or(0.0))
        .collect();
    let mut lam_rev: Vec<f64> = binding_lines
        .iter()
        .map(|&l| lam_warm_rev.get(l).copied().unwrap_or(0.0))
        .collect();

    let mut d0_sol = vec![0.0_f64; n_b];
    let mut c0_sol = vec![0.0_f64; n_b];

    charge_work((RH_KKT_ITERS * n_b * 16 + binding_lines.len() * 8) as u64);

    for iter in 0..RH_KKT_ITERS {
        let alpha = (RH_KKT_ALPHA0 / ((iter + 1) as f64).sqrt()).min(RH_KKT_ALPHA_MAX);
        for b in 0..n_b {
            let ptdf_adj: f64 = binding_lines
                .iter()
                .enumerate()
                .map(|(k, &l)| (lam_fwd[k] - lam_rev[k]) * sens[l][b])
                .sum();
            let sol = solve_battery_kkt(
                f0[b] - ptdf_adj,
                g0[b] + ptdf_adj,
                f1[b],
                g1[b],
                ub_d0[b],
                ub_c0[b],
                ub_d1[b],
                ub_c1[b],
                available[b],
                headroom[b],
                d_f_b[b],
                c_f_b[b],
            );
            d0_sol[b] = sol[0];
            c0_sol[b] = sol[1];
        }
        for (k, &l) in binding_lines.iter().enumerate() {
            let lim = challenge.lines[l].limit;
            let bf = state.base_line_flows.get(l).copied().unwrap_or(0.0);
            let net_flow: f64 = sens[l]
                .iter()
                .zip(d0_sol.iter().zip(c0_sol.iter()))
                .map(|(&s, (&d, &c))| s * (d - c))
                .sum();
            let b_fwd = (lim - bf).max(0.0);
            let b_rev = (lim + bf).max(0.0);
            lam_fwd[k] = (lam_fwd[k] + alpha * (net_flow - b_fwd)).max(0.0);
            lam_rev[k] = (lam_rev[k] + alpha * (-net_flow - b_rev)).max(0.0);
        }
    }

    for (k, &l) in binding_lines.iter().enumerate() {
        if l < lam_warm_fwd.len() {
            lam_warm_fwd[l] = lam_fwd[k];
            lam_warm_rev[l] = lam_rev[k];
        }
    }
    RH_LAM_WARM.with(|cell| {
        *cell.borrow_mut() = Some((lam_warm_fwd, lam_warm_rev));
    });

    // Final solve with converged duals.
    for b in 0..n_b {
        let ptdf_adj: f64 = binding_lines
            .iter()
            .enumerate()
            .map(|(k, &l)| (lam_fwd[k] - lam_rev[k]) * sens[l][b])
            .sum();
        let sol = solve_battery_kkt(
            f0[b] - ptdf_adj,
            g0[b] + ptdf_adj,
            f1[b],
            g1[b],
            ub_d0[b],
            ub_c0[b],
            ub_d1[b],
            ub_c1[b],
            available[b],
            headroom[b],
            d_f_b[b],
            c_f_b[b],
        );
        d0_sol[b] = sol[0];
        c0_sol[b] = sol[1];
    }

    Some((0..n_b).map(|b| d0_sol[b] - c0_sol[b]).collect())
}

/// Coord-polish pass count.
///
/// - `GNOSIS_COORD_PASSES=N` (N≥1) → N (cap 8); N=0 → off
/// - `GNOSIS_COORD_POLISH=0|false|off` → force off
/// - `GNOSIS_COORD_POLISH=1|true|on` → 2 passes any track
/// - unset → **2 passes on multiday + dense** (`30 < n ≤ 80`):
///   multiday S1c +2.08%; dense S5 +2.83% on S3 multi stack
///   (`receipt`). Capstone (n>80) stays off until measured.
fn coord_polish_passes(num_batteries: usize) -> Option<usize> {
    if let Some(n) = gnosis_env_var("GNOSIS_COORD_PASSES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return if n > 0 { Some(n.min(8)) } else { None };
    }
    match gnosis_env_var("GNOSIS_COORD_POLISH")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => return None,
        Some("1") | Some("true") | Some("on") => return Some(2),
        _ => {}
    }
    // Bule: coord only on multiday (paid multi free EV); dense coord is crown.
    if surface_bule() {
        if is_multiday_band(num_batteries) {
            Some(2)
        } else {
            None
        }
    } else if num_batteries > 30 && num_batteries <= 80 {
        Some(2)
    } else {
        None
    }
}

/// Portable extract of prior_art t51 `coordinate_polish_step`:
/// multipass sequential 1D search over each battery's action, scoring with
/// [`total_step_dp_value`] (same objective as PGA). Candidates = box corners /
/// quarters / local steps + network residual halfspace bounds from PTDF.
///
/// Only accepts feasible improving moves. Fuel: O(passes · batteries ·
/// candidates · lines) via incremental flow checks.
fn coordinate_polish(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    mut action: Vec<f64>,
    node_count: usize,
    passes: usize,
) -> Vec<f64> {
    const EPS: f64 = 1e-9;
    let n_b = challenge.batteries.len();
    if n_b == 0 || action.len() != n_b || passes == 0 {
        return action;
    }
    if !action_is_valid(challenge, state, &action, node_count) {
        return action;
    }

    let touched = touched_nodes(challenge, node_count);
    let mut best_value = total_step_dp_value(challenge, state, &action);
    let mut base_flows = sparse_line_flows(challenge, state, &action, node_count, &touched);

    for _ in 0..passes {
        let mut pass_improved = false;
        for b in 0..n_b {
            let battery = &challenge.batteries[b];
            let soc = state.state_of_charge[b];
            let (lo, hi) = power_bounds(battery, soc);
            let cur = action[b];
            let span = (hi - lo).max(0.0);
            if span <= EPS {
                continue;
            }
            let node = battery.node;
            // Network residual bounds: for each line, the a_b interval keeping
            // f_l = without_b + ptdf[l][node]·a_b inside ±limit.
            let mut net_lo = lo;
            let mut net_hi = hi;
            for (l, line) in challenge.lines.iter().enumerate() {
                let coeff = challenge.ptdf.get(l).and_then(|r| r.get(node)).copied().unwrap_or(0.0);
                if coeff.abs() <= 1e-12 {
                    continue;
                }
                let without_b = base_flows[l] - coeff * cur;
                let limit = line.limit;
                let low_at = (-limit - without_b) / coeff;
                let high_at = (limit - without_b) / coeff;
                let line_lo = low_at.min(high_at);
                let line_hi = low_at.max(high_at);
                net_lo = net_lo.max(line_lo);
                net_hi = net_hi.min(line_hi);
            }
            let net_span = net_hi - net_lo;

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

            let mut best_b = cur;
            let mut best_b_value = best_value;
            for &cand in &candidates {
                if (cand - cur).abs() <= EPS {
                    continue;
                }
                // Incremental flow check + power/SoC bounds.
                if !trial_flows_ok(challenge, &base_flows, &[(node, cand - cur)]) {
                    continue;
                }
                let mut trial = action.clone();
                trial[b] = cand;
                if !soc_bounds_ok(challenge, state, &trial) {
                    continue;
                }
                charge_work((n_b * 4) as u64);
                let v = total_step_dp_value(challenge, state, &trial);
                if v > best_b_value + 1e-9 {
                    best_b_value = v;
                    best_b = cand;
                }
            }

            if (best_b - cur).abs() > EPS {
                // Update flows by the accepted delta (affine).
                let delta = best_b - cur;
                for (l, flow) in base_flows.iter_mut().enumerate() {
                    let coeff = challenge
                        .ptdf
                        .get(l)
                        .and_then(|r| r.get(node))
                        .copied()
                        .unwrap_or(0.0);
                    *flow += coeff * delta;
                }
                action[b] = best_b;
                best_value = best_b_value;
                pass_improved = true;
            }
        }
        if !pass_improved {
            break;
        }
    }

    // Final safety: never return an action that only incremental checks accepted.
    if !action_is_valid(challenge, state, &action, node_count) {
        return project_feasible(challenge, state, &mut action.clone(), node_count);
    }
    action
}

/// Portable extract of prior_art `joint_pair_polish` (t52/t53): probe opposite
/// shifts on price-priority pairs; keep only feasible RT+DA score improvements.
fn joint_pair_exchange_polish(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    mut action: Vec<f64>,
    node_count: usize,
    da_weight: f64,
    shadows: &[f64],
) -> Vec<f64> {
    // Hoisted: the touched-node list is a sort over battery nodes and is fixed
    // for the solve, but `action_is_valid` rebuilt and re-sorted it on EVERY
    // trial. On capstone this loop issues ~1,605 validity checks per step
    // (308,084 over 192 steps), so the sort ran 308,084 times to produce one
    // unchanging vector. Exact: identical list, identical iteration order.
    // TRACK-SCOPED. The incremental trial check below is NOT bit-identical --
    // it sums a delta over 2-3 changed nodes instead of the full touched set,
    // and the rounding difference flips marginal feasibility verdicts, which a
    // chaotic local search then compounds over the horizon. Paired 8-nonce
    // replay, work versus quality:
    //
    //   baseline   0.0668 -> 0.0668    0.0%    work   0.0%   (search never runs)
    //   congested  0.3248 -> 0.3248    0.0%    work   0.0%   (search never runs)
    //   multiday   2.2023 -> 2.1409   -2.8%  t=-3.02  work -68.9%
    //   dense      1.4887 -> 1.4756   -0.9%  t=-0.77  work -88.5%
    //   capstone   1.5005 -> 1.4273   -4.9%  t=-5.41  work -90.0%
    //
    // multiday already fits in 30.8e9 of 100e9 -- 69% headroom -- so paying a
    // significant 2.8% for fuel it does not need is a straight loss. dense and
    // capstone are the opposite: capstone scores NOTHING today, and dense clears
    // by only 4% on n1, where an 88% work cut is margin against the nonces the
    // two-nonce probe never sampled.
    //
    // Scoped by battery count like the candidate count and track_params: > 40
    // selects dense (60) and capstone (100) and leaves multiday (40) and below
    // on the exact path, byte-identical.
    let incremental_trials = challenge.batteries.len() > 40;
    let ls_touched = touched_nodes(challenge, node_count);
    // Only the incremental path reads this; computing it on the exact path
    // costs multiday ~1.7% work for a vector it never touches.
    let mut base_flows = if incremental_trials {
        sparse_line_flows(challenge, state, &action, node_count, &ls_touched)
    } else {
        Vec::new()
    };
    let n = challenge.batteries.len();
    if n < 2 {
        return action;
    }
    // Score objective for joint pair/triplet local search.
    // Default n>30: total_step_dp_value — prior_art t52 style (imm + DP
    // continuation). stage: dense +7.0% / multi +10.0% / cap +11.6%
    // (`receipt`). Force: `GNOSIS_PAIR_DP_SCORE=0|1`.
    let use_pair_dp = pair_dp_score_enabled(n) && !challenge.dp_cache.is_empty();
    let score = |a: &[f64]| -> f64 {
        if use_pair_dp {
            total_step_dp_value(challenge, state, a)
        } else {
            estimated_step_profit(challenge, state, a)
                + da_weight * estimated_action_value_da(challenge, state, a, shadows)
        }
    };
    let mut best_score = score(&action);

    // Priority: |price_i − price_j| × min(span) — visit critical pairs first.
    // Opt-in shadow dual weight (`GNOSIS_PAIR_SHADOW_PRIO=1`): multiply by
    // (1 + |LMP_i − LMP_j|) where LMP_b = Σ_ℓ shadow_ℓ · PTDF[ℓ][node_b]
    // (`congestion_cost` with dir=1) to surface congestion-coupled pairs first.
    let shadow_prio = env_flag_on("GNOSIS_PAIR_SHADOW_PRIO");
    let mut scored: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        let (lo_i, hi_i) = power_bounds(&challenge.batteries[i], state.state_of_charge[i]);
        let span_i = (hi_i - lo_i).max(0.0);
        if span_i <= 1e-12 {
            continue;
        }
        let node_i = challenge.batteries[i].node;
        let price_i = state.observed_prices.get(node_i).copied().unwrap_or(0.0);
        let lmp_i = if shadow_prio {
            congestion_cost(challenge, i, 1.0, shadows)
        } else {
            0.0
        };
        for j in (i + 1)..n {
            let (lo_j, hi_j) = power_bounds(&challenge.batteries[j], state.state_of_charge[j]);
            let span_j = (hi_j - lo_j).max(0.0);
            if span_j <= 1e-12 {
                continue;
            }
            let node_j = challenge.batteries[j].node;
            let price_j = state.observed_prices.get(node_j).copied().unwrap_or(0.0);
            let mut prio = (price_i - price_j).abs() * span_i.min(span_j);
            if shadow_prio {
                let lmp_j = congestion_cost(challenge, j, 1.0, shadows);
                prio *= 1.0 + (lmp_i - lmp_j).abs();
            }
            // Opt-in: weight by incumbent |action| product — prefer pairs already
            // on the margin (`GNOSIS_PAIR_ACTION_PRIO=1`). Dense residual probe.
            if env_flag_on("GNOSIS_PAIR_ACTION_PRIO") {
                prio *= 1.0 + action[i].abs() * action[j].abs();
            }
            scored.push((prio, i, j));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    // Pair budget — track-shaped under stage pair-DP score:
    //   multiday stage: pb160 **+4.49%** t=8.50 8/8 under pair-DP+i8
    //     (stage pb80 was optimum under RT+DA score — dead under DP score).
    //   dense: pb1024 titan t52 align (monotone under pair-DP; S1 RT+DA
    //     pb512 regress is dead). Capstone pb768 + keep=4 S1 win.
    // Override: GNOSIS_PAIR_BUDGET=N.
    let pair_budget = if n <= 20 {
        scored.len()
    } else if n <= 50 {
        // Multi: pb768 under fi stack (stage). Bule keeps the same multi number —
        // dropping to 512 costs ~0.8% measured; not worth for "simpler".
        768
    } else if surface_bule() {
        // Bule dense/cap: small best-improve budget only (no dense pb1024 crown).
        256
    } else if n <= 80 {
        1024 // dense stage under pair-DP (titan t52 align)
    } else {
        1024 // capstone stage under pair-DP: n8 **+3.52%** t=3.32 7/8 (was 768)
    };
    let pair_budget = gnosis_env_var("GNOSIS_PAIR_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(pair_budget)
        .max(1);
    scored.truncate(pair_budget);

    // Search schedule.
    // stage dense: first-imp + fine alpha + 4-pass **+2.03%** t=3.88 10/14 under
    // stage stack (`receipt`). stage: cap too (**+1.96%**
    // t=3.02). stage: multiday (30<n≤50) same schedule **+1.80%** t=7.15 8/8
    // under pb512 (`receipt`). Baseline/congested (n≤30) stay
    // best-improve 2-pass. Force: `GNOSIS_PAIR_FIRST_IMP=0|1`,
    // `GNOSIS_PAIR_FINE_ALPHA=0|1`, `GNOSIS_LS_PASSES=N`.
    let first_imp = match gnosis_env_var("GNOSIS_PAIR_FIRST_IMP")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        // Research: multi+dense+cap first-imp. Bule: multi only (dense plain).
        _ => {
            if surface_bule() {
                is_multiday_band(n)
            } else {
                n > 30
            }
        }
    };
    let fine_alpha = match gnosis_env_var("GNOSIS_PAIR_FINE_ALPHA")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        _ => {
            if surface_bule() {
                is_multiday_band(n)
            } else {
                n > 30
            }
        }
    };
    let default_ls = if first_imp && n > 30 { 4 } else { 2 };
    let ls_passes = gnosis_env_var("GNOSIS_LS_PASSES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_ls)
        .max(1);
    // Independent (α,β) pair probes. stage multi: fine 8×8 + r48 **+0.95%**
    // t=3.94 — research only. Bule deliberately omits indep: same multi story
    // without the 6× wall / 8×8 grid IP. Force: `GNOSIS_PAIR_INDEP_ALPHA=0|1`.
    let indep_alpha = match gnosis_env_var("GNOSIS_PAIR_INDEP_ALPHA")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        _ => !surface_bule() && is_multiday_band(n),
    };
    // First-imp: each restart re-scans the budgeted pair list. Cap accept-restarts
    // hard — `ls_passes * budget` made dense SO hang (~230s/nonce) under
    // pb1024+fine-α. Track-shaped clamps:
    //   multi indep: **48** (stage knee; r16/r32 null/regress, r64 7.8× wall)
    //   multi opposite-only: **64**
    //   dense (≤80): **128** recovers full uncapped library quality
    //   capstone (>80): **64** (r128 non-monotone; r64 +1.96% t=3.02)
    // Override: `GNOSIS_PAIR_MAX_RESTARTS`.
    let max_restarts = if let Some(v) = gnosis_env_var("GNOSIS_PAIR_MAX_RESTARTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        v.clamp(1, 4096)
    } else if first_imp {
        if n > 80 {
            ls_passes.saturating_mul(16).clamp(1, 64)
        } else if n > 50 {
            ls_passes.saturating_mul(32).clamp(1, 128)
        } else if indep_alpha {
            ls_passes.saturating_mul(12).clamp(1, 48)
        } else {
            ls_passes.saturating_mul(16).clamp(1, 64)
        }
    } else {
        ls_passes
    };
    let alphas: &[f64] = if fine_alpha {
        &[-0.75, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 0.75]
    } else {
        &[-0.5, -0.25, 0.25, 0.5]
    };
    let mut restarts = 0usize;
    loop {
        let mut improved = false;
        for &(_, i, j) in &scored {
            let batt_i = &challenge.batteries[i];
            let batt_j = &challenge.batteries[j];
            let (lo_i, hi_i) = power_bounds(batt_i, state.state_of_charge[i]);
            let (lo_j, hi_j) = power_bounds(batt_j, state.state_of_charge[j]);
            let span_i = hi_i - lo_i;
            let span_j = hi_j - lo_j;
            if span_i <= 1e-12 || span_j <= 1e-12 {
                continue;
            }
            let cur_i = action[i];
            let cur_j = action[j];
            let mut local_best_i = cur_i;
            let mut local_best_j = cur_j;
            let mut local_best = best_score;
            // Opposite-signed α (flow-neutral) vs independent (α,β) grid.
            // Multi default indep uses full fine table (stage). Dense stays opposite.
            // `GNOSIS_PAIR_INDEP_FINE=0` forces coarse 4×4 (measured regress).
            let alpha_pairs: Vec<(f64, f64)> = if indep_alpha {
                let use_fine = match gnosis_env_var("GNOSIS_PAIR_INDEP_FINE")
                    .ok()
                    .as_deref()
                    .map(str::trim)
                {
                    Some("0") | Some("false") | Some("off") => false,
                    Some("1") | Some("true") | Some("on") => true,
                    _ => true, // multi stage default fine
                };
                let grid: &[f64] = if use_fine {
                    alphas
                } else if alphas.len() >= 8 {
                    &[-0.5, -0.25, 0.25, 0.5]
                } else {
                    alphas
                };
                let mut v = Vec::with_capacity(grid.len() * grid.len());
                for &a in grid {
                    for &b in grid {
                        v.push((a, b));
                    }
                }
                v
            } else {
                alphas.iter().map(|&a| (a, -a)).collect()
            };
            for &(ai, aj) in &alpha_pairs {
                let cand_i = (cur_i + ai * span_i).clamp(lo_i, hi_i);
                let cand_j = (cur_j + aj * span_j).clamp(lo_j, hi_j);
                if (cand_i - cur_i).abs() <= 1e-12 && (cand_j - cur_j).abs() <= 1e-12 {
                    continue;
                }
                let feasible = if incremental_trials {
                    battery_action_ok(challenge, state, i, cand_i)
                        && battery_action_ok(challenge, state, j, cand_j)
                        && trial_flows_ok(
                            challenge,
                            &base_flows,
                            &[
                                (challenge.batteries[i].node, cand_i - cur_i),
                                (challenge.batteries[j].node, cand_j - cur_j),
                            ],
                        )
                } else {
                    let mut probe = action.clone();
                    probe[i] = cand_i;
                    probe[j] = cand_j;
                    action_is_valid_with(challenge, state, &probe, node_count, &ls_touched)
                };
                if !feasible {
                    continue;
                }
                let mut trial = action.clone();
                trial[i] = cand_i;
                trial[j] = cand_j;
                let s = score(&trial);
                if s > local_best + 1e-9 {
                    local_best = s;
                    local_best_i = cand_i;
                    local_best_j = cand_j;
                }
            }
            if (local_best_i - cur_i).abs() > 1e-12 || (local_best_j - cur_j).abs() > 1e-12 {
                action[i] = local_best_i;
                action[j] = local_best_j;
                best_score = local_best;
                improved = true;
                // Incumbent moved: refresh the base the deltas are taken from.
                if incremental_trials {
                    base_flows =
                        sparse_line_flows(challenge, state, &action, node_count, &ls_touched);
                }
                if first_imp {
                    // Titan first-improvement: re-scan from top after accept.
                    break;
                }
            }
        }
        restarts += 1;
        if !improved || restarts >= max_restarts {
            break;
        }
    }

    // Optional finish pass: one best-improve sweep after first-imp restarts
    // exhaust. Opt-in `GNOSIS_PAIR_FINISH_BEST=1` — denser residual probe.
    if first_imp && env_flag_on("GNOSIS_PAIR_FINISH_BEST") {
        let mut improved = true;
        let mut finish_passes = 0usize;
        while improved && finish_passes < 2 {
            improved = false;
            finish_passes += 1;
            for &(_, i, j) in &scored {
                let batt_i = &challenge.batteries[i];
                let batt_j = &challenge.batteries[j];
                let (lo_i, hi_i) = power_bounds(batt_i, state.state_of_charge[i]);
                let (lo_j, hi_j) = power_bounds(batt_j, state.state_of_charge[j]);
                let span_i = hi_i - lo_i;
                let span_j = hi_j - lo_j;
                if span_i <= 1e-12 || span_j <= 1e-12 {
                    continue;
                }
                let cur_i = action[i];
                let cur_j = action[j];
                let mut local_best_i = cur_i;
                let mut local_best_j = cur_j;
                let mut local_best = best_score;
                for &alpha in alphas {
                    let cand_i = (cur_i + alpha * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - alpha * span_j).clamp(lo_j, hi_j);
                    if (cand_i - cur_i).abs() <= 1e-12 && (cand_j - cur_j).abs() <= 1e-12 {
                        continue;
                    }
                    let feasible = if incremental_trials {
                        battery_action_ok(challenge, state, i, cand_i)
                            && battery_action_ok(challenge, state, j, cand_j)
                            && trial_flows_ok(
                                challenge,
                                &base_flows,
                                &[
                                    (challenge.batteries[i].node, cand_i - cur_i),
                                    (challenge.batteries[j].node, cand_j - cur_j),
                                ],
                            )
                    } else {
                        let mut probe = action.clone();
                        probe[i] = cand_i;
                        probe[j] = cand_j;
                        action_is_valid_with(challenge, state, &probe, node_count, &ls_touched)
                    };
                    if !feasible {
                        continue;
                    }
                    let mut trial = action.clone();
                    trial[i] = cand_i;
                    trial[j] = cand_j;
                    let s = score(&trial);
                    if s > local_best + 1e-9 {
                        local_best = s;
                        local_best_i = cand_i;
                        local_best_j = cand_j;
                    }
                }
                if (local_best_i - cur_i).abs() > 1e-12 || (local_best_j - cur_j).abs() > 1e-12 {
                    action[i] = local_best_i;
                    action[j] = local_best_j;
                    best_score = local_best;
                    improved = true;
                    if incremental_trials {
                        base_flows =
                            sparse_line_flows(challenge, state, &action, node_count, &ls_touched);
                    }
                }
            }
        }
    }

    if action_is_valid(challenge, state, &action, node_count) {
        action
    } else {
        // Should not happen — each accept was validated.
        vec![0.0; n]
    }
}

/// Triplet gate: default `n > 50` (dense/capstone). `GNOSIS_TRIPLET=1` expands
/// to `n > 30` (includes multiday, where prior_art enables it). `GNOSIS_TRIPLET=0`
/// forces off even on dense/cap.
fn triplet_enabled(num_batteries: usize) -> bool {
    match gnosis_env_var("GNOSIS_TRIPLET")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => num_batteries > 30,
        // Bule: no triplet (dense/cap crown).
        _ => !surface_bule() && num_batteries > 50,
    }
}

/// Portable prior_art `joint_triplet_polish`: top-K by |action|, budgeted (i,j,k)
/// probes with ±α on i, ∓α on j, +γ on k. First-improvement sequential.
fn joint_triplet_exchange_polish(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    mut action: Vec<f64>,
    node_count: usize,
    da_weight: f64,
    shadows: &[f64],
) -> Vec<f64> {
    // Hoisted: the touched-node list is a sort over battery nodes and is fixed
    // for the solve, but `action_is_valid` rebuilt and re-sorted it on EVERY
    // trial. On capstone this loop issues ~1,605 validity checks per step
    // (308,084 over 192 steps), so the sort ran 308,084 times to produce one
    // unchanging vector. Exact: identical list, identical iteration order.
    // TRACK-SCOPED. The incremental trial check below is NOT bit-identical --
    // it sums a delta over 2-3 changed nodes instead of the full touched set,
    // and the rounding difference flips marginal feasibility verdicts, which a
    // chaotic local search then compounds over the horizon. Paired 8-nonce
    // replay, work versus quality:
    //
    //   baseline   0.0668 -> 0.0668    0.0%    work   0.0%   (search never runs)
    //   congested  0.3248 -> 0.3248    0.0%    work   0.0%   (search never runs)
    //   multiday   2.2023 -> 2.1409   -2.8%  t=-3.02  work -68.9%
    //   dense      1.4887 -> 1.4756   -0.9%  t=-0.77  work -88.5%
    //   capstone   1.5005 -> 1.4273   -4.9%  t=-5.41  work -90.0%
    //
    // multiday already fits in 30.8e9 of 100e9 -- 69% headroom -- so paying a
    // significant 2.8% for fuel it does not need is a straight loss. dense and
    // capstone are the opposite: capstone scores NOTHING today, and dense clears
    // by only 4% on n1, where an 88% work cut is margin against the nonces the
    // two-nonce probe never sampled.
    //
    // Scoped by battery count like the candidate count and track_params: > 40
    // selects dense (60) and capstone (100) and leaves multiday (40) and below
    // on the exact path, byte-identical.
    let incremental_trials = challenge.batteries.len() > 40;
    let ls_touched = touched_nodes(challenge, node_count);
    // Only the incremental path reads this; computing it on the exact path
    // costs multiday ~1.7% work for a vector it never touches.
    let mut base_flows = if incremental_trials {
        sparse_line_flows(challenge, state, &action, node_count, &ls_touched)
    } else {
        Vec::new()
    };
    let n = challenge.batteries.len();
    if n < 3 {
        return action;
    }
    // Same score gate as joint_pair (`pair_dp_score_enabled`).
    let use_pair_dp = pair_dp_score_enabled(n) && !challenge.dp_cache.is_empty();
    let score = |a: &[f64]| -> f64 {
        if use_pair_dp {
            total_step_dp_value(challenge, state, a)
        } else {
            estimated_step_profit(challenge, state, a)
                + da_weight * estimated_action_value_da(challenge, state, a, shadows)
        }
    };
    let mut best_score = score(&action);

    let top_k = 15.min(n).max(3);
    let mut batt_scores: Vec<(f64, usize)> = (0..n).map(|b| (action[b].abs(), b)).collect();
    batt_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let active: Vec<usize> = batt_scores.iter().take(top_k).map(|&(_, b)| b).collect();

    let triplet_budget = gnosis_env_var("GNOSIS_TRIPLET_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(if n <= 50 { 150 } else { 100 })
        .max(1);
    let mut tested = 0usize;
    'outer: for ii in 0..active.len() {
        let i = active[ii];
        let (lo_i, hi_i) = power_bounds(&challenge.batteries[i], state.state_of_charge[i]);
        let span_i = hi_i - lo_i;
        if span_i <= 1e-12 {
            continue;
        }
        for jj in (ii + 1)..active.len() {
            let j = active[jj];
            let (lo_j, hi_j) = power_bounds(&challenge.batteries[j], state.state_of_charge[j]);
            let span_j = hi_j - lo_j;
            if span_j <= 1e-12 {
                continue;
            }
            for kk in (jj + 1)..active.len() {
                if tested >= triplet_budget {
                    break 'outer;
                }
                tested += 1;
                let k = active[kk];
                let (lo_k, hi_k) = power_bounds(&challenge.batteries[k], state.state_of_charge[k]);
                let span_k = hi_k - lo_k;
                if span_k <= 1e-12 {
                    continue;
                }
                let cur_i = action[i];
                let cur_j = action[j];
                let cur_k = action[k];
                let mut local_best = best_score;
                let mut best_i = cur_i;
                let mut best_j = cur_j;
                let mut best_k = cur_k;
                for &alpha_ij in &[-0.5_f64, -0.25, 0.25, 0.5] {
                    let cand_i = (cur_i + alpha_ij * span_i).clamp(lo_i, hi_i);
                    let cand_j = (cur_j - alpha_ij * span_j).clamp(lo_j, hi_j);
                    for &alpha_k in &[-0.25_f64, 0.0, 0.25] {
                        let cand_k = (cur_k + alpha_k * span_k).clamp(lo_k, hi_k);
                        if (cand_i - cur_i).abs() <= 1e-12
                            && (cand_j - cur_j).abs() <= 1e-12
                            && (cand_k - cur_k).abs() <= 1e-12
                        {
                            continue;
                        }
                        let feasible = if incremental_trials {
                            battery_action_ok(challenge, state, i, cand_i)
                                && battery_action_ok(challenge, state, j, cand_j)
                                && battery_action_ok(challenge, state, k, cand_k)
                                && trial_flows_ok(
                                    challenge,
                                    &base_flows,
                                    &[
                                        (challenge.batteries[i].node, cand_i - cur_i),
                                        (challenge.batteries[j].node, cand_j - cur_j),
                                        (challenge.batteries[k].node, cand_k - cur_k),
                                    ],
                                )
                        } else {
                            let mut probe = action.clone();
                            probe[i] = cand_i;
                            probe[j] = cand_j;
                            probe[k] = cand_k;
                            action_is_valid_with(challenge, state, &probe, node_count, &ls_touched)
                        };
                        if !feasible {
                            continue;
                        }
                        let mut trial = action.clone();
                        trial[i] = cand_i;
                        trial[j] = cand_j;
                        trial[k] = cand_k;
                        let s = score(&trial);
                        if s > local_best + 1e-9 {
                            local_best = s;
                            best_i = cand_i;
                            best_j = cand_j;
                            best_k = cand_k;
                        }
                    }
                }
                if (best_i - cur_i).abs() > 1e-12
                    || (best_j - cur_j).abs() > 1e-12
                    || (best_k - cur_k).abs() > 1e-12
                {
                    action[i] = best_i;
                    action[j] = best_j;
                    action[k] = best_k;
                    base_flows =
                        sparse_line_flows(challenge, state, &action, node_count, &ls_touched);
                    best_score = local_best;
                }
            }
        }
    }

    if action_is_valid(challenge, state, &action, node_count) {
        action
    } else {
        vec![0.0; n]
    }
}

// ── Per-battery DA value DP (prometheus/titan portable extract) ──────────────

#[derive(Clone, Debug, PartialEq)]
pub struct BatteryDp {
    pub soc_lo: f64,
    pub soc_step: f64,
    pub levels: usize,
    /// values[t][soc_idx] = continuation from start of step t.
    pub values: Vec<Vec<f64>>,
}

fn immediate_step_profit(battery: &Battery, action: f64, price: f64) -> f64 {
    let dt = DELTA_T;
    let abs_u = action.abs();
    action * price * dt - KAPPA_TX * abs_u * dt
        - KAPPA_DEG * ((abs_u * dt) / battery.capacity.max(1e-9)).powi(2)
}

fn interp_value(values: &[f64], soc: f64, lo: f64, step: f64, last: usize) -> f64 {
    if step <= 1e-15 || values.is_empty() {
        return values.first().copied().unwrap_or(0.0);
    }
    let pos = ((soc - lo) / step).clamp(0.0, last as f64);
    let low = pos.floor() as usize;
    let high = (low + 1).min(last);
    let alpha = pos - low as f64;
    values[low] * (1.0 - alpha) + values[high] * alpha
}

fn build_battery_da_dp(battery: &Battery, da_at_node: &[f64], residual_shift: f64) -> BatteryDp {
    // prometheus-like resolution; 49×13 stays online-cheap on baseline (n=10,T=96).
    // denser grids (65/81) REGRESSED multiday/dense/capstone — hold 49×13 (tick 10:28Z).
    // Grid resolution. The DP is now hoisted (built once per solve), so its cost
    // is O(batteries x steps x levels x actions). After the hoist, multiday sits
    // at 65-70e9 of the 100e9 budget; dense (~80 batteries) and capstone (100+)
    // scale to ~104e9 and ~130e9 and still exit 87, needing ~1.5-2x more.
    //
    // The recorded history is one-sided: "denser grids (65/81) REGRESSED
    // multiday/dense/capstone — hold 49x13". DENSER was tested; SPARSER was
    // never tried, and halving `levels` halves the DP directly.
    //
    // Scoped to large instances so the small tracks -- which already fit with
    // room and would only lose quality -- are untouched, and gated so the
    // effect can be measured against a control.
    let (levels, action_levels) = match gnosis_env_var("GNOSIS_DP_LEVELS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(l) if l >= 5 => (l, 13usize),
        _ => (49usize, 13usize),
    };
    let soc_lo = battery.capacity * battery.reserve_fraction;
    let soc_hi = battery.capacity;
    let span = (soc_hi - soc_lo).max(1e-9);
    let soc_step = span / (levels - 1) as f64;
    let n_steps = da_at_node.len();
    let mut values = vec![vec![0.0; levels]; n_steps + 1];
    let last = levels - 1;

    // Terminal: salvage leftover energy against final DA.
    if n_steps > 0 {
        let salvage = da_at_node[n_steps - 1].max(0.0) * battery.discharge_efficiency * 0.25;
        for s in 0..levels {
            let soc = soc_lo + soc_step * s as f64;
            values[n_steps][s] = salvage * (soc - soc_lo).max(0.0);
        }
    }

    // HOIST EVERYTHING PRICE-INDEPENDENT OUT OF THE LAYER LOOP.
    //
    // The action grid, the resulting next-SoC, and the interpolation index and
    // weight depend only on (soc level, action index) -- never on `t`. They were
    // recomputed for all `n_steps` layers: O(steps x levels x actions) calls to
    // power_bounds / next_soc / clamp / floor / divide, for O(levels x actions)
    // distinct results. So are the two cost terms of `immediate_step_profit`;
    // only the revenue term `action * price * dt` moves with the layer.
    //
    // BIT-IDENTICAL, not merely equivalent. The inner expression stays
    // `a * price * dt - c1 - c2 + continuation` with the same operands in the
    // same association: precomputing c1 and c2 evaluates the same subexpressions
    // earlier, it does not re-associate them. The continuation is
    // `values[low] * (1 - alpha) + values[high] * alpha`, exactly `interp_value`'s
    // final line with exactly its indices.
    struct DpProbe {
        action: f64,
        tx_cost: f64,
        deg_cost: f64,
        low: usize,
        high: usize,
        alpha: f64,
    }
    let cap = battery.capacity.max(1e-9);
    let make_probe = |soc: f64, a: f64| -> DpProbe {
        let nxt = next_soc(battery, soc, a).clamp(soc_lo, soc_hi);
        let (low, high, alpha) = if soc_step <= 1e-15 {
            (0usize, 0usize, 0.0)
        } else {
            let pos = ((nxt - soc_lo) / soc_step).clamp(0.0, last as f64);
            let low = pos.floor() as usize;
            (low, (low + 1).min(last), pos - low as f64)
        };
        let abs_u = a.abs();
        DpProbe {
            action: a,
            tx_cost: KAPPA_TX * abs_u * DELTA_T,
            deg_cost: KAPPA_DEG * ((abs_u * DELTA_T) / cap).powi(2),
            low,
            high,
            alpha,
        }
    };

    // Same order the layer loop used: idle, then the coarse grid, then the
    // full-charge / full-discharge probes -- so `best` sees the same values in
    // the same sequence and the running max is unchanged.
    let mut probes: Vec<Vec<DpProbe>> = Vec::with_capacity(levels);
    for s_idx in 0..levels {
        let soc = soc_lo + soc_step * s_idx as f64;
        let (lo, hi) = power_bounds(battery, soc);
        let mut level_probes = Vec::with_capacity(action_levels + 3);
        level_probes.push(make_probe(soc, 0.0));
        for k in 0..action_levels {
            let frac = if action_levels <= 1 {
                0.5
            } else {
                k as f64 / (action_levels - 1) as f64
            };
            let action = lo + (hi - lo) * frac;
            if action.abs() <= 1e-15 {
                continue;
            }
            level_probes.push(make_probe(soc, action.clamp(lo, hi)));
        }
        for &a in &[lo, hi] {
            if a.abs() <= 1e-15 {
                continue;
            }
            level_probes.push(make_probe(soc, a));
        }
        probes.push(level_probes);
    }

    for t in (0..n_steps).rev() {
        // Residual shift only on the first remaining step (current); future is pure DA.
        let price = if t == 0 {
            da_at_node[t] + residual_shift
        } else {
            da_at_node[t]
        };
        let (head, tail) = values.split_at_mut(t + 1);
        let current = &mut head[t];
        let next_layer = &tail[0];
        for (s_idx, level_probes) in probes.iter().enumerate() {
            let mut best = f64::NEG_INFINITY;
            for probe in level_probes {
                let continuation = next_layer[probe.low] * (1.0 - probe.alpha)
                    + next_layer[probe.high] * probe.alpha;
                let value = probe.action * price * DELTA_T - probe.tx_cost - probe.deg_cost
                    + continuation;
                if value > best {
                    best = value;
                }
            }
            current[s_idx] = best;
        }
    }

    BatteryDp {
        soc_lo,
        soc_step,
        levels,
        values,
    }
}

fn pick_dp_action(
    dp: &BatteryDp,
    battery: &Battery,
    t: usize,
    soc: f64,
    price: f64,
) -> f64 {
    let (lo, hi) = power_bounds(battery, soc);
    let last = dp.levels.saturating_sub(1);
    let next_t = (t + 1).min(dp.values.len().saturating_sub(1));
    let next_vals = &dp.values[next_t];
    let soc_hi = battery.capacity;

    let mut best_a = 0.0;
    let mut best_v = f64::NEG_INFINITY;
    let probes = 17usize;
    for k in 0..probes {
        let frac = if probes <= 1 {
            0.5
        } else {
            k as f64 / (probes - 1) as f64
        };
        let a = (lo + (hi - lo) * frac).clamp(lo, hi);
        let nxt = next_soc(battery, soc, a).clamp(dp.soc_lo, soc_hi);
        let v = immediate_step_profit(battery, a, price)
            + interp_value(next_vals, nxt, dp.soc_lo, dp.soc_step, last);
        if v > best_v {
            best_v = v;
            best_a = a;
        }
    }
    best_a
}

fn pga_enabled(num_batteries: usize) -> bool {
    match gnosis_env_var("GNOSIS_PGA")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        // Default ON on tracks with a library quality win:
        //   baseline (≈10): +0.11 abs @i12, SO ~0.2e9 @100e9
        //   multiday (≈40): +10.7%, SO 41–49e9 @100e9
        //   dense (60) / capstone (100): +9–13% library — previously marked
        //     SO-red only because probes used _FUEL=100e9 (2% of mainnet
        //     max_fuel_budget=5e12; REPLAY_MATRIX / titan README 5T).
        //     GCB 31046a10/230b5675 exit 87 is a **wrong-ceiling** defect,
        //     not a structural SO ban. Wave-2: re-probe at 5e12; default ON.
        // Congested (≈20): library null (t≈0.1) — leave OFF.
        _ => !(num_batteries > 15 && num_batteries <= 30),
    }
}

/// Multi-seed PGA: polish every food source, then post-PGA BCS.
/// - force ON/OFF via `GNOSIS_PGA_MULTI=1|0`
/// - default ON for multiday + dense (`30 < n ≤ 80`): multiday S1d +1.45%;
///   dense S1d +2.36%. Capstone (n>80) stays single-seed (fuel/time).
fn pga_multi_enabled(num_batteries: usize) -> bool {
    match gnosis_env_var("GNOSIS_PGA_MULTI")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        // Bule: multi-seed PGA only on multiday (dense multi-PGA is crown).
        _ => {
            if surface_bule() {
                is_multiday_band(num_batteries)
            } else {
                num_batteries > 30 && num_batteries <= 80
            }
        }
    }
}

/// S2 Pareto archive (Lean `EnergyParetoArchive`).
/// Default ON for n>30 only (multiday/dense/cap): baseline BCS path was
/// −0.44% lib n=8. Multi-PGA tracks skip pre-PGA non-dom (full employed set).
/// `GNOSIS_PARETO_ARCHIVE=0|1` forces.
fn pareto_archive_enabled(num_batteries: usize) -> bool {
    match gnosis_env_var("GNOSIS_PARETO_ARCHIVE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        _ => num_batteries > 30,
    }
}

/// Joint pair/triplet score uses DP action value (imm + continuation) instead
/// of RT+DA inventory. Aligns local search with PGA/coord objective (titan t52).
/// Default ON wherever joint_pair runs (`n > 30`):
///   dense n16  **+7.03%** t=5.56 14/14
///   multi n8   **+10.0%** t=4.76 8/8
///   cap   n8   **+11.6%** t=4.40 8/8
/// (`receipt`). Force: `GNOSIS_PAIR_DP_SCORE=0|1`.
fn pair_dp_score_enabled(num_batteries: usize) -> bool {
    match gnosis_env_var("GNOSIS_PAIR_DP_SCORE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        Some("1") | Some("true") | Some("on") => true,
        _ => num_batteries > 30,
    }
}

/// Soft constraint repair after multi-PGA: re-`project_feasible` invalid
/// polished seeds instead of dropping them. Default **ON** (cheap; only hits
/// when polish exits the polytope). Off: `GNOSIS_SOFT_CV=0`.
fn soft_cv_enabled() -> bool {
    match gnosis_env_var("GNOSIS_SOFT_CV")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") | Some("false") | Some("off") => false,
        _ => true,
    }
}

/// Post-PGA ε-constraint band on quality for secondary residual pick.
/// Default **0** (pure quality). S4 library n=8: ε=0.01 regressed multi
/// −1.69% / dense −0.83%; ε=0.02 multi −0.76% / dense −0.25%. Opt-in only
/// via `GNOSIS_EPS_CONSTRAINT=0.01` (or 0.02 etc.).
fn post_pga_eps() -> f64 {
    match gnosis_env_var("GNOSIS_EPS_CONSTRAINT")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        None | Some("0") | Some("false") | Some("off") => 0.0,
        Some(s) => s.parse::<f64>().ok().unwrap_or(0.0).clamp(0.0, 0.2),
    }
}

/// Network residual score (higher = more headroom). Maps to archive residual
/// objective opposite of congestion waste (Bule waste face inverse).
fn network_residual_score(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &[f64],
    shadows: &[f64],
) -> f64 {
    if challenge.lines.is_empty() {
        return 1.0;
    }
    let mut headroom = 0.0;
    let mut cong = 0.0;
    for (b, &u) in action.iter().enumerate() {
        if u.abs() > 1e-15 {
            cong += congestion_cost(challenge, b, u.signum(), shadows).abs();
        }
    }
    // Prefer actions that leave line headroom under base flow.
    if !state.base_line_flows.is_empty()
        && state.base_line_flows.len() == challenge.lines.len()
    {
        for (line, &flow) in challenge.lines.iter().zip(state.base_line_flows.iter()) {
            let lim = line.limit.max(1e-9);
            headroom += (1.0 - (flow.abs() / lim).min(1.0)).max(0.0);
        }
        headroom /= challenge.lines.len() as f64;
    } else {
        headroom = 1.0;
    }
    // Scale residual into same order as typical step scores (~1e0–1e3).
    headroom * 100.0 - cong
}

/// Bule diversity face on an action: active batteries + mild L1 mass.
/// Maps to `BuleyUnit.diversity` / entropy face, not free NSGA crowding.
fn bule_diversity_face(action: &[f64]) -> f64 {
    let active = action.iter().filter(|a| a.abs() > 1e-9).count() as f64;
    let l1: f64 = action.iter().map(|a| a.abs()).sum();
    active + (1.0 + l1).ln()
}

/// Run PGA only on steps where `time % stride == 0` (plus always t=0).
///
/// `GNOSIS_PGA_STRIDE=k` (k≥1). Default 1 = every step. Sparse strides amortize
/// SO fuel on dense/capstone where every-step i1 still exit-87'd n1.
fn pga_step_active(time: usize, _num_batteries: usize) -> bool {
    let stride = gnosis_env_var("GNOSIS_PGA_STRIDE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    if stride <= 1 {
        return true;
    }
    time % stride == 0
}

/// Optional congestion gate: only run PGA when some line is tight on base flow.
///
/// `GNOSIS_PGA_CONG_THRESH=θ` (0–1]. If unset/0, always active (when enabled).
/// When set, require `max_l |base_flow_l| / limit_l ≥ θ`. Thesis: dense/cap
/// SO dies on every-step PGA; maybe only the congested steps need the polish.
fn pga_congestion_active(challenge: &EnergyChallenge, state: &OnlineState) -> bool {
    let thresh = match gnosis_env_var("GNOSIS_PGA_CONG_THRESH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(t) if t > 0.0 => t.clamp(0.0, 1.0),
        _ => return true,
    };
    if challenge.lines.is_empty() || state.base_line_flows.is_empty() {
        return true;
    }
    for (line, &flow) in challenge.lines.iter().zip(state.base_line_flows.iter()) {
        let lim = line.limit.max(1e-12);
        if flow.abs() / lim >= thresh {
            return true;
        }
    }
    false
}

/// ∂V/∂soc on the DP layer after step `t` (continuation from next SoC).
fn dv_dsoc(dp: &BatteryDp, t: usize, soc: f64) -> f64 {
    if dp.levels <= 1 || dp.values.is_empty() || dp.soc_step <= 1e-15 {
        return 0.0;
    }
    let next_t = (t + 1).min(dp.values.len().saturating_sub(1));
    let values = &dp.values[next_t];
    let last = dp.levels - 1;
    if last == 0 || values.len() <= last {
        return 0.0;
    }
    let pos = ((soc - dp.soc_lo) / dp.soc_step).clamp(0.0, last as f64);
    let mut low = pos.floor() as usize;
    if low >= last {
        low = last - 1;
    }
    (values[low + 1] - values[low]) / dp.soc_step
}

fn dp_action_value(dp: &BatteryDp, battery: &Battery, t: usize, soc: f64, price: f64, action: f64) -> f64 {
    let next_t = (t + 1).min(dp.values.len().saturating_sub(1));
    let nxt = next_soc(battery, soc, action).clamp(dp.soc_lo, battery.capacity);
    let last = dp.levels.saturating_sub(1);
    immediate_step_profit(battery, action, price)
        + interp_value(&dp.values[next_t], nxt, dp.soc_lo, dp.soc_step, last)
}

fn total_step_dp_value(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &[f64],
) -> f64 {
    let t = state.time;
    let mut total = 0.0;
    for (b, battery) in challenge.batteries.iter().enumerate() {
        let Some(dp) = challenge.dp_cache.get(b) else {
            continue;
        };
        let price = state.observed_prices.get(battery.node).copied().unwrap_or(0.0);
        let soc = state.state_of_charge.get(b).copied().unwrap_or(0.0);
        let u = action.get(b).copied().unwrap_or(0.0);
        total += dp_action_value(dp, battery, t, soc, price, u);
    }
    total
}

/// Analytic gradient of imm profit + DP continuation w.r.t. u_b (titan extract).
/// Optional soft network dual from current line shadows (online coupling term).
/// `soc_ref_override` (when Some) replaces `challenge.soc_ref` for the L8 term
/// — used by per-step dynamic recompute without mutating the challenge.
fn analytic_action_gradient(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &[f64],
    shadows: &[f64],
    network_lambda: f64,
    soc_ref_override: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    const EPS: f64 = 1e-12;
    let t = state.time;
    let n = challenge.batteries.len();
    let mut grad = vec![0.0_f64; n];
    for b in 0..n {
        let battery = &challenge.batteries[b];
        let Some(dp) = challenge.dp_cache.get(b) else {
            continue;
        };
        let price = state.observed_prices.get(battery.node).copied().unwrap_or(0.0);
        let soc = state.state_of_charge.get(b).copied().unwrap_or(0.0);
        let u = action.get(b).copied().unwrap_or(0.0);
        let s = if u > EPS {
            1.0
        } else if u < -EPS {
            -1.0
        } else {
            0.0
        };
        let cap2 = battery.nominal_capacity.max(1e-9).powi(2);
        // d/du [u·λ·Δt − κ|u|Δt − κ_deg ( |u|Δt / E )² ]
        let imm = price * DELTA_T
            - s * KAPPA_TX * challenge.friction_weight * DELTA_T
            - 2.0 * KAPPA_DEG * DELTA_T * DELTA_T * u / cap2;

        let nxt = next_soc(battery, soc, u);
        let dsoc_du = if u > EPS {
            if nxt <= battery.capacity * battery.reserve_fraction + EPS {
                0.0
            } else {
                -DELTA_T / battery.discharge_efficiency.max(1e-12)
            }
        } else if u < -EPS {
            if nxt >= battery.capacity - EPS {
                0.0
            } else {
                -battery.charge_efficiency * DELTA_T
            }
        } else {
            -0.5
                * (DELTA_T / battery.discharge_efficiency.max(1e-12)
                    + battery.charge_efficiency * DELTA_T)
        };
        let mut dv = dv_dsoc(dp, t, nxt.clamp(dp.soc_lo, battery.capacity));
        // titan composite_wv: re-inject fleet congestion shadow into ∂V/∂soc.
        if let Some(&dc) = challenge.delta_cong.get(t) {
            dv += dc;
        }
        // Soft congestion dual: linearised cost of +1 MW discharge (fixed shadows).
        let net = if network_lambda.abs() > EPS {
            network_lambda * congestion_cost(challenge, b, 1.0, shadows)
        } else {
            0.0
        };
        grad[b] = imm + dv * dsoc_du - net;
        // titan t51 L8: SoC reference-tracking (Huang 2024-style).
        // dsoc_du < 0 always on the interior, so next_soc > ref ⇒ more discharge.
        if challenge.soc_ref_lambda > EPS {
            let refs_row = soc_ref_override
                .and_then(|r| r.get(b))
                .or_else(|| challenge.soc_ref.get(b));
            if let Some(refs) = refs_row {
                if !refs.is_empty() {
                    let t1 = (t + 1).min(refs.len().saturating_sub(1));
                    let soc_ref_t1 = refs[t1];
                    grad[b] -=
                        challenge.soc_ref_lambda * (nxt - soc_ref_t1) * dsoc_du;
                }
            }
        }
    }
    grad
}

/// Static SoC-ref: DA P25/P75 greedy from `initial_soc` at t=0, no residual shift.
#[must_use]
pub fn compute_soc_reference_static(
    batteries: &[Battery],
    day_ahead_prices: &[Vec<f64>],
    initial_soc: &[f64],
) -> Vec<Vec<f64>> {
    compute_soc_reference_dynamic(batteries, day_ahead_prices, initial_soc, &[], 0)
}

/// Portable extract of prior_art t51 L8b `compute_soc_reference_dynamic`:
/// per-battery greedy **DA + residual_shift** P25/P75 trajectory from
/// `current_socs` starting at `start_t`.
///
/// `soc_ref[b][t]` = SoC **before** step `t` (length `n_steps + 1`). Entries
/// before `start_t` are left 0 (unused by the gradient which only reads t+1
/// from the current step). O(batteries · remaining steps).
#[must_use]
pub fn compute_soc_reference_dynamic(
    batteries: &[Battery],
    day_ahead_prices: &[Vec<f64>],
    current_socs: &[f64],
    residual_shift: &[f64],
    start_t: usize,
) -> Vec<Vec<f64>> {
    let n_b = batteries.len();
    let n_t = day_ahead_prices.len();
    let mut refs = vec![vec![0.0_f64; n_t.saturating_add(1)]; n_b];
    if n_t == 0 || n_b == 0 || start_t >= n_t {
        return refs;
    }
    for b in 0..n_b {
        let battery = &batteries[b];
        let node = battery.node;
        let soc_min = battery.capacity * battery.reserve_fraction;
        let soc_max = battery.capacity;
        let start = current_socs
            .get(b)
            .copied()
            .unwrap_or((soc_min + soc_max) * 0.5)
            .clamp(soc_min, soc_max);
        refs[b][start_t] = start;

        let shift = residual_shift.get(node).copied().unwrap_or(0.0);
        // P25/P75 of adjusted remaining-horizon DA (titan t51).
        let mut da_sorted: Vec<f64> = (start_t..n_t)
            .map(|t| {
                day_ahead_prices
                    .get(t)
                    .and_then(|row| row.get(node))
                    .copied()
                    .unwrap_or(0.0)
                    + shift
            })
            .collect();
        if da_sorted.is_empty() {
            continue;
        }
        da_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p25 = da_sorted[da_sorted.len() / 4];
        let p75 = da_sorted[(da_sorted.len() * 3) / 4];

        for t in start_t..n_t {
            let soc = refs[b][t];
            let price = day_ahead_prices
                .get(t)
                .and_then(|row| row.get(node))
                .copied()
                .unwrap_or(0.0)
                + shift;
            let delta_soc = if price > p75 {
                let max_disch = (soc - soc_min).max(0.0);
                let disch_mwh = (battery.max_discharge * DELTA_T
                    / battery.discharge_efficiency.max(1e-12))
                .min(max_disch);
                -disch_mwh
            } else if price < p25 {
                let max_chg = (soc_max - soc).max(0.0);
                let chg_mwh =
                    (battery.max_charge * DELTA_T * battery.charge_efficiency).min(max_chg);
                chg_mwh
            } else {
                0.0
            };
            refs[b][t + 1] = (soc + delta_soc).clamp(soc_min, soc_max);
        }
    }
    refs
}

/// Portable extract of prior_art `use_composite_wv` (fleet k=1 path):
/// dual aggregate DPs on capacity-weighted DA prices with/without congestion
/// premiums; `delta_cong[t] = λ · (∂V_cong/∂E − ∂V_nocong/∂E)` at fleet mid-SoC.
///
/// Returns a length-`n_t` series. Empty input premiums → pure-DA vs pure-DA
/// (all zeros). Built once per solve; consumed only by the PGA gradient.
#[must_use]
pub fn build_fleet_delta_cong(
    batteries: &[Battery],
    day_ahead_prices: &[Vec<f64>],
    // Optional per-battery premiums `[t][b]` (from `expected_lmp_premiums`).
    premiums: Option<&[Vec<f64>]>,
    cwv_lambda: f64,
    agg_levels: usize,
) -> Vec<f64> {
    let n_b = batteries.len();
    let n_t = day_ahead_prices.len();
    if n_b == 0 || n_t == 0 || cwv_lambda.abs() < 1e-15 {
        return vec![0.0; n_t];
    }
    let total_cap: f64 = batteries.iter().map(|b| b.capacity).sum::<f64>().max(1.0);
    let fleet_da: Vec<f64> = (0..n_t)
        .map(|t| {
            let row = day_ahead_prices.get(t);
            let mut p = 0.0_f64;
            for bat in batteries {
                let da = row
                    .and_then(|r| r.get(bat.node))
                    .copied()
                    .unwrap_or(0.0);
                p += bat.capacity * da;
            }
            p / total_cap
        })
        .collect();
    let fleet_premium: Vec<f64> = (0..n_t)
        .map(|t| {
            let Some(base) = premiums else {
                return 0.0;
            };
            let row = base.get(t);
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (b, bat) in batteries.iter().enumerate() {
                let prem = row.and_then(|r| r.get(b)).copied().unwrap_or(0.0);
                num += bat.capacity * prem;
                den += bat.capacity;
            }
            num / den.max(1.0)
        })
        .collect();
    let da_with_cong: Vec<f64> = fleet_da
        .iter()
        .zip(fleet_premium.iter())
        .map(|(da, prem)| da + prem)
        .collect();

    // Synthetic aggregate battery: sum of fleet power/energy bounds.
    let cap: f64 = batteries.iter().map(|b| b.capacity).sum();
    let reserve_energy: f64 = batteries
        .iter()
        .map(|b| b.capacity * b.reserve_fraction)
        .sum();
    let reserve_frac = if cap > 1e-12 {
        (reserve_energy / cap).clamp(0.0, 0.99)
    } else {
        0.0
    };
    let max_charge: f64 = batteries.iter().map(|b| b.max_charge).sum();
    let max_discharge: f64 = batteries.iter().map(|b| b.max_discharge).sum();
    let eta_c: f64 = {
        let w: f64 = batteries.iter().map(|b| b.capacity * b.charge_efficiency).sum();
        (w / total_cap).clamp(0.5, 1.0)
    };
    let eta_d: f64 = {
        let w: f64 = batteries.iter().map(|b| b.capacity * b.discharge_efficiency).sum();
        (w / total_cap).clamp(0.5, 1.0)
    };
    let fleet_bat = Battery {
        node: 0,
        capacity: cap.max(1.0),
        nominal_capacity: cap.max(1.0),
        max_charge: max_charge.max(1e-6),
        max_discharge: max_discharge.max(1e-6),
        charge_efficiency: eta_c,
        discharge_efficiency: eta_d,
        reserve_fraction: reserve_frac,
    };

    // Temporarily force aggregate DP resolution via env is invasive; build with
    // a direct call that uses default levels, then re-interp. For portability we
    // call build_battery_da_dp (respects GNOSIS_DP_LEVELS) on the synthetic unit.
    let _ = agg_levels; // reserved for a dedicated aggregate builder if needed
    let dp_cong = build_battery_da_dp(&fleet_bat, &da_with_cong, 0.0);
    let dp_nocong = build_battery_da_dp(&fleet_bat, &fleet_da, 0.0);
    let e_mid = 0.5
        * (fleet_bat.capacity * fleet_bat.reserve_fraction
            + fleet_bat.capacity);

    (0..n_t)
        .map(|t| {
            let d_cong = dv_dsoc(&dp_cong, t, e_mid);
            let d_nocong = dv_dsoc(&dp_nocong, t, e_mid);
            cwv_lambda * (d_cong - d_nocong)
        })
        .collect()
}

/// Project onto the feasible set cheaply for PGA line search.
/// Prefer halfspace (fast) then uniform scale; fall back to full project_feasible
/// only when both fail — fuel-sensitive on dense/capstone.
fn pga_project(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &mut [f64],
    node_count: usize,
) {
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let (lo, hi) = power_bounds(battery, state.state_of_charge[i]);
        action[i] = action[i].clamp(lo, hi);
    }
    if action_is_valid(challenge, state, action, node_count) {
        return;
    }
    let mut trial = action.to_vec();
    if project_halfspace(challenge, state, &mut trial, node_count)
        && action_is_valid(challenge, state, &trial, node_count)
    {
        action.copy_from_slice(&trial);
        return;
    }
    // Uniform scale bisection toward zero (always feasible if base network is).
    let original = action.to_vec();
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best = vec![0.0; original.len()];
    for _ in 0..16 {
        let mid = 0.5 * (lo + hi);
        let scaled: Vec<f64> = original.iter().map(|a| a * mid).collect();
        if action_is_valid(challenge, state, &scaled, node_count) {
            best = scaled;
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if action_is_valid(challenge, state, &best, node_count) {
        action.copy_from_slice(&best);
    } else {
        for a in action.iter_mut() {
            *a = 0.0;
        }
    }
}

/// Env flag true only for explicit `1`/`true`/`on` (default OFF).
fn env_flag_on(name: &str) -> bool {
    matches!(
        gnosis_env_var(name).ok().as_deref().map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

/// Dual surface: **research** (default) vs **bule** (public-minimal).
///
/// - `research`: full overnight cascade, all tracks, indep-α multi, dense crown.
/// - `bule`: multi-only competitive edge with the *smallest* measured stack that
///   still holds a strong multi beat; dense/cap on a plain path so a Code drip
///   does not gift dense residual weapons. See `BULE_SURFACE.md`.
///
/// Env: `GNOSIS_SURFACE=bule|public|research` (default `research` for climb).
/// Package submit: force `bule` as the published default.
#[inline]
fn surface_bule() -> bool {
    match gnosis_env_var("GNOSIS_SURFACE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("bule") | Some("public") | Some("minimal") => true,
        Some("research") | Some("full") => false,
        _ => true,
    }
}

/// Multiday battery band (historical ~40).
#[inline]
fn is_multiday_band(n: usize) -> bool {
    n > 30 && n <= 50
}

/// Dense battery band (historical ~60).
#[inline]
fn is_dense_band(n: usize) -> bool {
    n > 50 && n <= 80
}

/// Minimal titan-style PGA: analytic grad + backtracking + project.
/// Outer/line-search budgets env-gated; defaults stay fuel-cheap (4×4).
///
/// Optional prior_art multiday (t51) enhancements, **default OFF**:
/// - `GNOSIS_PGA_MOMENTUM=1` — heavy-ball direction `β·v + g` (β=0.99)
/// - `GNOSIS_PGA_COSINE=1` — cosine-anneal β from 0.99 → `GNOSIS_PGA_BETA_END` (0.6)
/// - `GNOSIS_PGA_BB=1` — Barzilai–Borwein lr growth cap (1.05) / decay floor (0.85)
///
/// With all three off the loop is byte-identical to the pure-gradient path.
fn projected_gradient_polish(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    seed: Vec<f64>,
    node_count: usize,
    shadows: &[f64],
    soc_ref_override: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    // Track-shaped outer-iter defaults (library, seed gnosis_replay_v1):
    //   baseline (n≤15): 12 iters +0.0135 abs over i4 (t=4.6), saturates by 12.
    //   multiday+dense (30 < n ≤ 80) under stage pair-DP:
    //     dense n16 i8 **+2.32%** t=3.53; multi n8 i8 **+2.47%** t=4.81 7/8
    //     (`receipt`). Capstone (n>80) stays 4 (fuel thinnest).
    let n_b = challenge.batteries.len();
    let default_outer = if n_b <= 15 {
        12
    } else if n_b > 30 && n_b <= 80 {
        8
    } else {
        4
    };
    let outer = gnosis_env_var("GNOSIS_PGA_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_outer)
        .clamp(1, 120);
    // titan dense grad_ls_iters=12; our historical default 4. Dense LS=12 is
    // opt-in via env until measured under stage stack (`GNOSIS_PGA_LS`).
    let default_ls = 4;
    let ls = gnosis_env_var("GNOSIS_PGA_LS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_ls)
        .clamp(1, 12);
    // stage: dense `GNOSIS_PGA_NET=2` → **+1.25%**. Bule: plain net=1 on dense
    // (do not publish the network-λ crown in the multi drip).
    let default_net = if !surface_bule() && is_dense_band(n_b) {
        2.0
    } else {
        1.0
    };
    let network_lambda = gnosis_env_var("GNOSIS_PGA_NET")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default_net)
        .max(0.0);

    // Titan t51 heavy-ball / BB / cosine-β.
    // stage: multiday momentum n=32 **+0.95%** (22/32) under PTDF stack →
    // default ON for `30 < n ≤ 50`. Dense default OFF until stage measure under
    // pair-DP+pb1024 (earlier momstack +0.96% null). Force: `GNOSIS_PGA_MOMENTUM=0|1`.
    // Cosine/BB still opt-in only (or via full titan stack probe).
    let use_momentum = match gnosis_env_var("GNOSIS_PGA_MOMENTUM")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("1") | Some("true") | Some("on") => true,
        Some("0") | Some("false") | Some("off") => false,
        _ => n_b > 30 && n_b <= 50,
    };
    let use_cosine = env_flag_on("GNOSIS_PGA_COSINE");
    let use_bb = env_flag_on("GNOSIS_PGA_BB");
    let beta_end = gnosis_env_var("GNOSIS_PGA_BETA_END")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.6)
        .clamp(0.0, 0.99);
    const MOMENTUM_BETA: f64 = 0.99;
    const LR_GROWTH_CAP: f64 = 1.05;
    const BB_DECAY_FACTOR: f64 = 0.85;
    let t_max = outer.saturating_sub(1).max(1) as f64;

    let mut action = seed;
    pga_project(challenge, state, &mut action, node_count);
    let mut best_action = action.clone();
    let mut best_value = total_step_dp_value(challenge, state, &best_action);

    let max_power = challenge
        .batteries
        .iter()
        .map(|b| b.max_charge.max(b.max_discharge))
        .fold(1.0_f64, f64::max);
    let mut lr = max_power * 0.5;
    // Velocity only read when use_momentum; zeros ⇒ dir = grad on first step.
    let mut velocity = vec![0.0_f64; action.len()];

    for outer_iter in 0..outer {
        let grad = analytic_action_gradient(
            challenge,
            state,
            &action,
            shadows,
            network_lambda,
            soc_ref_override,
        );
        let g_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if g_norm < 1e-9 {
            break;
        }
        charge_work((challenge.batteries.len() * 8) as u64);

        let beta = if use_momentum && use_cosine {
            let frac = outer_iter as f64 / t_max;
            beta_end
                + (MOMENTUM_BETA - beta_end)
                    * (1.0 + (std::f64::consts::PI * frac).cos())
                    * 0.5
        } else {
            MOMENTUM_BETA
        };
        // Direction: heavy-ball β·v + g carries across non-improving zones
        // where pure-gradient lr*=0.4 collapses (titan t51 / Hindsight d0a700d7).
        let dir: Vec<f64> = if use_momentum {
            grad.iter()
                .zip(velocity.iter())
                .map(|(g, v)| beta * v + g)
                .collect()
        } else {
            // Borrow-free clone so the no-momentum path stays pure-grad.
            grad.clone()
        };

        let prev_lr = lr;
        let mut improved = false;
        let mut cur_lr = lr;
        for _ in 0..ls {
            // step scale uses grad norm (titan); momentum only re-orients dir.
            let step = cur_lr / g_norm;
            let mut trial: Vec<f64> = action
                .iter()
                .zip(dir.iter())
                .map(|(a, d)| a + step * d)
                .collect();
            pga_project(challenge, state, &mut trial, node_count);
            let v = total_step_dp_value(challenge, state, &trial);
            if v > best_value + 1e-9 {
                action = trial.clone();
                best_action = trial;
                best_value = v;
                lr = if use_bb {
                    (cur_lr * 1.4).min(prev_lr * LR_GROWTH_CAP)
                } else {
                    cur_lr * 1.4
                };
                if use_momentum {
                    for (vel, g) in velocity.iter_mut().zip(grad.iter()) {
                        *vel = beta * *vel + g;
                    }
                }
                improved = true;
                break;
            }
            cur_lr *= 0.5;
        }
        if !improved {
            lr = if use_bb {
                (lr * 0.4).max(prev_lr * BB_DECAY_FACTOR)
            } else {
                lr * 0.4
            };
            if lr < max_power * 1e-4 {
                break;
            }
        }
    }

    // Final hard project through the quality path so we never return an action
    // that only halfspace accepted under looser numerics.
    if !action_is_valid(challenge, state, &best_action, node_count) {
        best_action = project_feasible(challenge, state, &mut best_action.clone(), node_count);
    }
    best_action
}

/// prior_art-style congestion anticipation premiums for the day-ahead DP.
///
/// For each step and line whose **exogenous** flow ratio exceeds `threshold`,
/// add a PTDF-weighted premium to every battery so the per-battery DP prices
/// the scarce line capacity *before* the joint projection has to repair it.
///
/// Formula (portable extract of prior_art t52/t53 `expected_premiums`):
///
/// ```text
/// proba   = clamp((|f_exo|/limit − θ) / (1 − θ), 0, 1)
/// premium = 20 · scale · proba
/// Δλ[t][b] += −sens[l][b] · sign(f_exo) · premium
/// sens[l][b] = ptdf[l][node_b] − ptdf[l][slack]
/// ```
///
/// `exo_flows[t][line]` must be the zero-battery network flow at step `t`
/// (known at t=0 from the challenge's exogenous injections). One-shot cost
/// O(steps · lines · batteries) — free relative to the rollout.
///
/// When `threshold ≥ 1` or `scale == 0` every premium is zero (byte-identical
/// DP to the no-LMP path, up to the same floating-point tree).
#[must_use]
pub fn expected_lmp_premiums(
    exo_flows: &[Vec<f64>],
    line_limits: &[f64],
    ptdf: &[Vec<f64>],
    battery_nodes: &[usize],
    slack_bus: usize,
    threshold: f64,
    premium_scale: f64,
) -> Vec<Vec<f64>> {
    let n_steps = exo_flows.len();
    let n_b = battery_nodes.len();
    let n_lines = line_limits.len();
    let mut prem = vec![vec![0.0_f64; n_b]; n_steps];
    if n_lines == 0 || n_b == 0 || premium_scale == 0.0 {
        return prem;
    }
    let base_premium = 20.0 * premium_scale;
    let denom = (1.0 - threshold).max(1e-6);
    for (t, flows) in exo_flows.iter().enumerate() {
        for l in 0..n_lines {
            let limit = line_limits.get(l).copied().unwrap_or(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let flow = flows.get(l).copied().unwrap_or(0.0);
            let ratio = flow.abs() / limit;
            if ratio <= threshold {
                continue;
            }
            let proba = ((ratio - threshold) / denom).clamp(0.0, 1.0);
            let premium = base_premium * proba;
            let sign_f = if flow >= 0.0 { 1.0_f64 } else { -1.0_f64 };
            let ptdf_row = match ptdf.get(l) {
                Some(row) => row,
                None => continue,
            };
            let ptdf_slack = ptdf_row.get(slack_bus).copied().unwrap_or(0.0);
            for (b, &node) in battery_nodes.iter().enumerate() {
                let impact = ptdf_row.get(node).copied().unwrap_or(0.0) - ptdf_slack;
                if impact.abs() > 1e-6 {
                    prem[t][b] += -impact * sign_f * premium;
                }
            }
        }
    }
    prem
}

/// **Build every battery's day-ahead value function ONCE, over the full horizon.**
///
/// `build_dp_actions` rebuilt a DP per battery PER STEP over the remaining
/// horizon, costing `batteries x levels x SUM_t (n_steps - t)` =
/// **O(batteries x steps^2 x levels)**. Measured: that is **94.7-98.3% of all
/// fuel** (baseline 9,992,483,557 -> 171,084,295 with the DP skipped), and it is
/// why multiday/dense/capstone exit 87.
///
/// The hoist is EXACT, for two reasons:
///
/// 1. The day-ahead prices are fixed and known in advance, so the backward
///    recursion over any suffix is a slice of the recursion over the whole
///    horizon. The terminal salvage uses `da[n-1]`, identical for every suffix.
/// 2. `residual_shift` enters only at local `t == 0`
///    (`build_battery_da_dp`), and `pick_dp_action` queries `values[t + 1]` —
///    so the shifted layer is **never read**. The shift therefore cannot make
///    the per-step rebuild differ from a slice of the global DP.
///
/// Querying at absolute `t` then reads exactly the layer the per-step build
/// would have produced. O(steps^2) -> O(steps).
///
/// `lmp_premiums` is optional per-battery additive congestion premium on the DA
/// series used only for this DP (`premiums[t][b]`). `None` is the original
/// pure-DA path. Premiums are **not** written back into
/// `challenge.day_ahead_prices`, so scarcity / price-rank seeds stay pure-DA
/// (titan only injects into the DP prices, not the whole portfolio).
pub fn build_dp_cache(challenge: &EnergyChallenge) -> Vec<BatteryDp> {
    build_dp_cache_with_lmp(challenge, None)
}

/// As [`build_dp_cache`], with optional titan-style LMP premiums on DP prices.
pub fn build_dp_cache_with_lmp(
    challenge: &EnergyChallenge,
    lmp_premiums: Option<&[Vec<f64>]>,
) -> Vec<BatteryDp> {
    let n_steps = challenge.day_ahead_prices.len();
    challenge
        .batteries
        .iter()
        .enumerate()
        .map(|(b_idx, battery)| {
            let da_at_node: Vec<f64> = (0..n_steps)
                .map(|t| {
                    let da = challenge
                        .day_ahead_prices
                        .get(t)
                        .and_then(|row| row.get(battery.node))
                        .copied()
                        .unwrap_or(0.0);
                    let prem = lmp_premiums
                        .and_then(|p| p.get(t))
                        .and_then(|row| row.get(b_idx))
                        .copied()
                        .unwrap_or(0.0);
                    da + prem
                })
                .collect();
            build_battery_da_dp(battery, &da_at_node, 0.0)
        })
        .collect()
}

/// Inputs for prior_art OCO constraint tracking (`use_ptdf_ct`).
///
/// Discriminant vs static exo-LMP: μ is computed from a **greedy DP rollout's
/// dispatch-induced** line violations, not from exo flows alone. Untouched
/// batteries keep their original DP (Q-exact for them).
pub struct PtdfCtInput<'a> {
    pub batteries: &'a [Battery],
    /// `day_ahead_prices[t][node]`.
    pub day_ahead_prices: &'a [Vec<f64>],
    pub ptdf: &'a [Vec<f64>],
    pub line_limits: &'a [f64],
    /// Zero-battery (exogenous) line flows `exo_flows[t][line]`.
    pub exo_flows: &'a [Vec<f64>],
    pub battery_nodes: &'a [usize],
    pub slack_bus: usize,
    pub initial_soc: &'a [f64],
    /// Optional static LMP premiums to seed ep_ct; same shape as
    /// `expected_lmp_premiums` (`[t][b]`).
    pub base_premiums: Option<&'a [Vec<f64>]>,
    /// OCO step size (titan dense `ct_step_eta` = 1.0, capstone 0.5).
    pub eta: f64,
    /// `1 - ct_ref_kappa` (titan dense/capstone both use kappa=0 → scale 1.0).
    pub ct_scale: f64,
}

/// Portable extract of prior_art t52/t53 `use_ptdf_ct`:
/// greedy unconstrained DP rollout → OCO μ on violated candidate lines →
/// adjust per-battery premiums → rebuild only touched DPs.
///
/// One-shot at solve start. Cost O(steps · batteries · action_probes) for the
/// rollout + O(touched · steps · levels · actions) for rebuilds.
#[must_use]
pub fn apply_ptdf_constraint_tracking(
    dps: Vec<BatteryDp>,
    input: &PtdfCtInput<'_>,
) -> Vec<BatteryDp> {
    let n_b = input.batteries.len();
    let n_t = input.day_ahead_prices.len();
    let n_lines = input.line_limits.len();
    if n_b == 0 || n_t == 0 || n_lines == 0 || dps.len() != n_b {
        return dps;
    }

    // sens[l][b] = ptdf[l][node_b] − ptdf[l][slack]
    let mut sens = vec![vec![0.0_f64; n_b]; n_lines];
    for l in 0..n_lines {
        let row = match input.ptdf.get(l) {
            Some(r) => r,
            None => continue,
        };
        let ptdf_slack = row.get(input.slack_bus).copied().unwrap_or(0.0);
        for b in 0..n_b {
            let node = input.battery_nodes.get(b).copied().unwrap_or(0);
            sens[l][b] = row.get(node).copied().unwrap_or(0.0) - ptdf_slack;
        }
    }

    const ACTIVE_SENS_THRESH: f64 = 1e-4;
    let candidate_lines: Vec<usize> = (0..n_lines)
        .filter(|&l| {
            input.line_limits.get(l).copied().unwrap_or(0.0) > 1e-6
                && sens[l].iter().any(|&s| s.abs() > ACTIVE_SENS_THRESH)
        })
        .collect();
    if candidate_lines.is_empty() {
        return dps;
    }

    // Step 1: greedy DP rollout (no network projection) — estimates which lines
    // the unconstrained DA policy would violate.
    let mut socs: Vec<f64> = (0..n_b)
        .map(|b| {
            input
                .initial_soc
                .get(b)
                .copied()
                .unwrap_or_else(|| {
                    let bat = &input.batteries[b];
                    0.5 * (bat.capacity * bat.reserve_fraction + bat.capacity)
                })
        })
        .collect();
    let mut flows_all: Vec<Vec<f64>> = Vec::with_capacity(n_t);
    for t in 0..n_t {
        let mut action = vec![0.0_f64; n_b];
        for b in 0..n_b {
            let battery = &input.batteries[b];
            let soc = socs[b];
            let (lo, hi) = power_bounds(battery, soc);
            if hi - lo <= 1e-15 {
                continue;
            }
            let node = battery.node;
            let price = input
                .day_ahead_prices
                .get(t)
                .and_then(|row| row.get(node))
                .copied()
                .unwrap_or(0.0);
            action[b] = pick_dp_action(&dps[b], battery, t, soc, price);
        }
        let exo = input.exo_flows.get(t);
        let mut flows_t = vec![0.0_f64; n_lines];
        for &l in &candidate_lines {
            let base = exo.and_then(|e| e.get(l)).copied().unwrap_or(0.0);
            let batt_flow: f64 = sens[l]
                .iter()
                .zip(action.iter())
                .map(|(s, a)| s * a)
                .sum();
            flows_t[l] = base + batt_flow;
        }
        flows_all.push(flows_t);
        for b in 0..n_b {
            let battery = &input.batteries[b];
            socs[b] = next_soc(battery, socs[b], action[b])
                .clamp(battery.capacity * battery.reserve_fraction, battery.capacity);
        }
    }

    // Step 2: OCO multipliers on candidate lines.
    let eta = input.eta.max(0.0);
    let ct_scale = input.ct_scale;
    let mut mu_oco = vec![vec![0.0_f64; n_lines]; n_t];
    for t in 0..n_t {
        for &l in &candidate_lines {
            let limit = input.line_limits[l];
            let viol = flows_all[t][l].abs() - limit;
            if viol > 0.0 {
                mu_oco[t][l] = (eta * viol).min(limit * 0.5);
            }
        }
    }

    // Step 3: seed ep_ct from base premiums, add dispatch-induced correction.
    let mut ep_ct = vec![vec![0.0_f64; n_b]; n_t];
    if let Some(base) = input.base_premiums {
        for t in 0..n_t {
            if let Some(row) = base.get(t) {
                for b in 0..n_b {
                    ep_ct[t][b] = row.get(b).copied().unwrap_or(0.0);
                }
            }
        }
    }
    let mut touched = vec![false; n_b];
    for t in 0..n_t {
        for &l in &candidate_lines {
            let mu_l = mu_oco[t][l];
            if mu_l <= 1e-12 {
                continue;
            }
            let sign = if flows_all[t][l] >= 0.0 {
                1.0_f64
            } else {
                -1.0_f64
            };
            for b in 0..n_b {
                let impact = sens[l][b];
                if impact.abs() > 1e-6 {
                    ep_ct[t][b] -= impact * sign * mu_l * ct_scale;
                    touched[b] = true;
                }
            }
        }
    }

    // Step 4: rebuild only touched batteries at full resolution.
    input
        .batteries
        .iter()
        .enumerate()
        .map(|(b, battery)| {
            if !touched[b] {
                return dps[b].clone();
            }
            let node = battery.node;
            let da_ct: Vec<f64> = (0..n_t)
                .map(|t| {
                    let da = input
                        .day_ahead_prices
                        .get(t)
                        .and_then(|row| row.get(node))
                        .copied()
                        .unwrap_or(0.0);
                    da + ep_ct[t][b]
                })
                .collect();
            build_battery_da_dp(battery, &da_ct, 0.0)
        })
        .collect()
}

fn build_dp_actions(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    residual_shift: &[f64],
) -> Vec<f64> {
    let t = state.time;
    let n_steps = challenge.day_ahead_prices.len();
    // Build DP only on remaining horizon (prometheus online query style).
    let mut raw = Vec::with_capacity(challenge.batteries.len());
    for (index, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let da_remaining: Vec<f64> = (t..n_steps)
            .map(|s| challenge.day_ahead_prices[s][node])
            .collect();
        if da_remaining.is_empty() {
            raw.push(0.0);
            continue;
        }
        let rt = state.observed_prices[node];
        let da = da_remaining[0];
        let w = challenge.residual_weight.clamp(0.0, 1.0);
        // Blend RT with DA for query stability (pure RT overfits noise).
        let price = rt * (0.70 + 0.30 * w) + da * (0.30 * (1.0 - w));

        // HOISTED. Query the cached full-horizon value function at ABSOLUTE t
        // instead of rebuilding the DP over the remaining horizon every step.
        // Exact: `pick_dp_action` reads `values[t + 1]`, and the only layer the
        // per-step build shifted was local `t == 0`, which is never read.
        // O(batteries x steps^2 x levels) -> O(batteries x steps x levels).
        let a = match challenge.dp_cache.get(index) {
            Some(dp) => pick_dp_action(dp, battery, t, state.state_of_charge[index], price),
            None => {
                // Fallback preserves the original behaviour if the cache is absent.
                let shift = residual_shift.get(node).copied().unwrap_or(0.0);
                let dp = build_battery_da_dp(battery, &da_remaining, shift);
                pick_dp_action(&dp, battery, 0, state.state_of_charge[index], price)
            }
        };
        raw.push(a);
    }
    raw
}

#[derive(Clone, Copy)]
enum ResidualWeightMode {
    DaOnly,
    Blend,
}

fn residual_shift_vector(challenge: &EnergyChallenge, state: &OnlineState) -> Vec<f64> {
    // One-step residual as a soft shift on future DA (prometheus residual history
    // collapsed to the current observation — no static mutex in the pure kernel).
    let t = state.time;
    if t >= challenge.day_ahead_prices.len() {
        return vec![0.0; state.observed_prices.len()];
    }
    state
        .observed_prices
        .iter()
        .zip(&challenge.day_ahead_prices[t])
        .map(|(&rt, &da)| {
            let r = rt - da;
            (0.55 * r * challenge.residual_weight.clamp(0.0, 1.0)).clamp(-25.0, 25.0)
        })
        .collect()
}

fn line_shadows(challenge: &EnergyChallenge, state: &OnlineState) -> Vec<f64> {
    // Soft dual ≈ tightness of base flow (titan LMP-anticipation lite).
    challenge
        .lines
        .iter()
        .zip(&state.base_line_flows)
        .map(|(line, &flow)| {
            let limit = line.limit.max(1e-9);
            let load = (flow.abs() / limit).clamp(0.0, 1.5);
            let excess = (flow.abs() - limit).max(0.0) / limit;
            challenge.congestion_weight * (load.powi(2) + 2.0 * excess) * flow.signum()
        })
        .collect()
}

fn congestion_cost(
    challenge: &EnergyChallenge,
    battery_idx: usize,
    direction: f64,
    shadows: &[f64],
) -> f64 {
    let node = challenge.batteries[battery_idx].node;
    let mut cost = 0.0;
    for (line_idx, shadow) in shadows.iter().enumerate() {
        let ptdf = challenge.ptdf[line_idx].get(node).copied().unwrap_or(0.0);
        cost += shadow * ptdf * direction;
    }
    cost
}

fn power_bounds(battery: &Battery, soc: f64) -> (f64, f64) {
    let reserve = battery.capacity * battery.reserve_fraction;
    let headroom = (battery.capacity - soc).max(0.0);
    let available = (soc - reserve).max(0.0);
    let dt = DELTA_T;
    let max_charge_from_soc = if battery.charge_efficiency > 0.0 {
        headroom / (battery.charge_efficiency * dt)
    } else {
        0.0
    };
    let max_discharge_from_soc = if battery.discharge_efficiency > 0.0 {
        available * battery.discharge_efficiency / dt
    } else {
        0.0
    };
    let max_charge = max_charge_from_soc.min(battery.max_charge).max(0.0);
    let max_discharge = max_discharge_from_soc.min(battery.max_discharge).max(0.0);
    (-max_charge, max_discharge)
}

fn next_soc(battery: &Battery, soc: f64, action: f64) -> f64 {
    let c = (-action).max(0.0);
    let d = action.max(0.0);
    let dt = DELTA_T;
    soc + battery.charge_efficiency * c * dt - d * dt / battery.discharge_efficiency.max(1e-12)
}

fn build_scarcity_actions(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    shadows: &[f64],
    residual_weight: f64,
    mode: ResidualWeightMode,
) -> Vec<f64> {
    let t = state.time;
    let n_steps = challenge.day_ahead_prices.len();
    let steps_remaining = n_steps.saturating_sub(t).max(1);
    let mut raw = Vec::with_capacity(challenge.batteries.len());

    for (index, battery) in challenge.batteries.iter().enumerate() {
        let soc = state.state_of_charge[index];
        let (lo, hi) = power_bounds(battery, soc);
        let discharge_value =
            directional_value(challenge, state, index, 1.0, residual_weight, mode, shadows);
        let charge_value =
            directional_value(challenge, state, index, -1.0, residual_weight, mode, shadows);

        // Terminal / near-terminal liquidation (energy_v1).
        let reserve = battery.capacity * battery.reserve_fraction;
        let energy_above = (soc - reserve).max(0.0);
        let discharge_step_energy =
            (battery.max_discharge * battery.discharge_efficiency * DELTA_T).max(1e-12);
        let steps_to_empty = (energy_above / discharge_step_energy).ceil() as usize;
        let liquidation = steps_remaining <= steps_to_empty.saturating_add(2) && hi > 0.0;

        let action = if steps_remaining <= 1 && hi > 0.0 {
            let fill = ((soc - reserve) / battery.capacity.max(1e-9)).clamp(0.0, 1.0);
            hi * fill
        } else if liquidation {
            let node = battery.node;
            let price = match mode {
                ResidualWeightMode::DaOnly => challenge.day_ahead_prices[t][node],
                ResidualWeightMode::Blend => {
                    let da = challenge.day_ahead_prices[t][node];
                    da + residual_weight * (state.observed_prices[node] - da)
                }
            };
            if price > KAPPA_TX {
                let urgency =
                    (1.0 - steps_remaining as f64 / (steps_to_empty as f64 + 4.0).max(1.0))
                        .clamp(0.35, 1.0);
                hi * urgency
            } else {
                0.0
            }
        } else if discharge_value > charge_value
            && discharge_value > challenge.deadband
            && hi > 0.0
        {
            let intensity = signal_strength(discharge_value, challenge.deadband.max(1e-6))
                .clamp(0.30, 1.0);
            hi * intensity
        } else if charge_value > discharge_value
            && charge_value > challenge.deadband
            && lo < 0.0
        {
            let intensity =
                signal_strength(charge_value, challenge.deadband.max(1e-6)).clamp(0.30, 1.0);
            lo * intensity
        } else {
            0.0
        };
        raw.push(action.clamp(lo, hi));
    }
    raw
}

/// energy_v1-style price-rank: act when current DA is in the extreme quantiles
/// of the remaining horizon (with residual boost only for large spikes).
fn build_price_rank_actions(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    shadows: &[f64],
) -> Vec<f64> {
    let t = state.time;
    let n_steps = challenge.day_ahead_prices.len();
    let n_remaining = n_steps.saturating_sub(t);
    let mut raw = Vec::with_capacity(challenge.batteries.len());
    let friction = KAPPA_TX * challenge.friction_weight;

    for (index, battery) in challenge.batteries.iter().enumerate() {
        let soc = state.state_of_charge[index];
        let (lo, hi) = power_bounds(battery, soc);
        let node = battery.node;
        let da = challenge.day_ahead_prices[t][node];
        let rt = state.observed_prices[node];
        let residual = rt - da;
        // Mild residual tilt only for strong RT innovation (prometheus residual idea).
        let price = if residual.abs() > 5.0 {
            da + challenge.residual_weight * residual
        } else {
            da
        };

        let horizon = challenge.lookahead_horizon.min(n_remaining).max(1);
        let end = (t + horizon).min(n_steps);
        let mut future: Vec<f64> = (t..end)
            .map(|tau| challenge.day_ahead_prices[tau][node])
            .collect();
        if future.is_empty() {
            raw.push(0.0);
            continue;
        }
        future.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = future.len();
        let q20 = future[((n - 1) * 20) / 100];
        let q80 = future[((n - 1) * 80) / 100];
        let q10 = future[((n - 1) * 10) / 100];
        let q90 = future[((n - 1) * 90) / 100];
        let band = (q80 - q20).abs().max(1.0);

        let cong_d = congestion_cost(challenge, index, 1.0, shadows);
        let cong_c = congestion_cost(challenge, index, -1.0, shadows);
        let eta_c = battery.charge_efficiency.max(1e-9);
        let eta_d = battery.discharge_efficiency.max(1e-9);

        // Discharge when price is in the upper tail vs future distribution.
        let discharge_edge = price * eta_d - q80 - friction - cong_d.max(0.0);
        // Charge when price is in the lower tail and cycle margin to q90 remains.
        let cycle_margin = q90 * eta_d - price / eta_c - friction;
        let charge_edge = (q20 - price).max(0.0) + 0.5 * cycle_margin.max(0.0) - cong_c.max(0.0);

        // Terminal liquidation urgency.
        let reserve = battery.capacity * battery.reserve_fraction;
        let energy_above = (soc - reserve).max(0.0);
        let step_e = (battery.max_discharge * eta_d * DELTA_T).max(1e-12);
        let steps_to_empty = (energy_above / step_e).ceil() as usize;
        let liquidate = n_remaining <= steps_to_empty.saturating_add(3) && hi > 0.0 && price > friction;

        let a = if liquidate {
            let urg = (1.0 - n_remaining as f64 / (steps_to_empty as f64 + 4.0).max(1.0))
                .clamp(0.4, 1.0);
            hi * urg
        } else if n_remaining <= 1 && hi > 0.0 {
            hi * ((soc - reserve) / battery.capacity.max(1e-9)).clamp(0.0, 1.0)
        } else if discharge_edge > challenge.deadband && hi > 0.0 {
            let peak_boost = ((price - q90).max(0.0) / band).clamp(0.0, 1.0);
            let frac = (0.40 + 0.60 * (discharge_edge / band).clamp(0.0, 1.0) + 0.15 * peak_boost)
                .clamp(0.35, 1.0);
            hi * frac
        } else if charge_edge > challenge.deadband && lo < 0.0 && n_remaining as f64 * DELTA_T >= 1.5
        {
            let dip_boost = ((q10 - price).max(0.0) / band).clamp(0.0, 1.0);
            let frac = (0.40 + 0.60 * (charge_edge / band).clamp(0.0, 1.0) + 0.15 * dip_boost)
                .clamp(0.35, 1.0);
            lo * frac
        } else {
            0.0
        };
        raw.push(a.clamp(lo, hi));
    }
    raw
}

fn project_feasible(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    raw: &mut [f64],
    node_count: usize,
) -> Vec<f64> {
    // Clamp to power/SOC bounds first.
    for (index, battery) in challenge.batteries.iter().enumerate() {
        let (lo, hi) = power_bounds(battery, state.state_of_charge[index]);
        raw[index] = raw[index].clamp(lo, hi);
    }

    if action_is_valid(challenge, state, raw, node_count) {
        return raw.to_vec();
    }

    if challenge.max_backoffs == 0 {
        return vec![0.0; challenge.batteries.len()];
    }

    // Halfspace projection is NOT used as a primary replacement for the
    // substitute/soften backoff: measured as a hard regression (dense −15.7%
    // t=−7.9, capstone −20.4% t=−7.0, 0 wins). Kept as an additive terminal
    // rescue under `GNOSIS_HALFSPACE_PROJ=1` (see project_feasible_terminal).

    let mut candidate = raw.to_vec();
    // Each backoff iteration recomputes network flows: a PTDF pass costing
    // O(lines x batteries). Charging a flat 1 per iteration was wrong -- it
    // made a 67-71% cut in meter units produce no change in real fuel, because
    // the meter was blind to the dominant term. Charge the actual shape.
    //
    // MEASURED: this loop IS the cost. GNOSIS_BISECT_BACKOFFS=0 removes 62-68%
    // of multiday's instruction fuel on its own and converts `dense` from exit
    // 87 to completing (63.3e9 / 83.9e9). The other three knobs together add
    // only 2.2e9 more. The decision rule written above resolved to "the backoff
    // loop is the cost; remedy is a tighter feasibility search".
    //
    // ONE FLOW PASS PER ITERATION, not two. Each iteration used to pay a DENSE
    // PTDF pass inside `most_violated_line_detail` -> `line_flows`
    // (O(lines x node_count)) AND a sparse one inside `action_is_valid`
    // (O(lines x touched)), computing the same vector twice.
    //
    // The two agree BIT-FOR-BIT, not merely mathematically: the terms the
    // sparse form skips are `row[node] * 0.0`, and `x + 0.0 == x` exactly in
    // IEEE-754, with `touched` sorted so the nonzero terms keep their relative
    // order. So one sparse pass can serve both the validity test and the
    // violated-line choice, and `action_is_valid` decomposes exactly into
    // `soc_bounds_ok` + `flows_within_limits`.
    //
    // REJECTED, and recorded so it is not retried: maintaining `flows`
    // incrementally across iterations via PTDF linearity
    // (`flows[l] += ptdf[l][node] * delta`, O(changed x lines), which is what
    // the scale search below does). It is exact in real arithmetic and WRONG in
    // floating point -- accumulating deltas over up to 64 iterations drifts
    // from the true flow, and a drifted feasibility verdict returns an
    // infeasible action. Measured on a paired replay: `congested`, the track
    // where lines actually bind, fell from quality 0.1637 to -0.0173, i.e.
    // BELOW the greedy baseline. `baseline` and `multiday` moved UP (+0.0014,
    // +0.0502) -- the same bug flattering itself on tracks with slack. The
    // scale search can accumulate safely because it computes one delta and
    // probes it; it never chains a delta onto a previous delta.
    let touched = touched_nodes(challenge, node_count);
    // MEMOISE `directional_value` on (battery, sign) ACROSS THE WHOLE LOOP.
    //
    // It reads `state` (time, SoC) and the price matrix and NEVER reads
    // `actions`, so its value is constant for the entire projection -- yet it
    // sat inside `substitute_on_line`'s O(batteries^2) pair search and was
    // recomputed from scratch on every one of up to `max_backoffs` iterations.
    // Each call allocates a fresh Vec of all remaining prices at the battery's
    // node, so the waste was O(backoffs x batteries^2 x steps) per step.
    //
    // Owned here and shared with `soften_violated_line`, which calls the same
    // function on the same (battery, sign) keys.
    //
    // EXACT: same inputs, same results. Lazy, because the harm/headroom/relief
    // guards reject most batteries before their value is ever needed.
    let mut value_cache: Vec<Option<f64>> = vec![None; challenge.batteries.len() * 2];
    // Per-line peak |PTDF| over battery-hosting nodes, for the pair-scan
    // pre-filter in `substitute_on_line`. Fixed for the whole projection.
    let line_node_peak: Vec<f64> = challenge
        .ptdf
        .iter()
        .map(|row| {
            challenge
                .batteries
                .iter()
                .map(|b| row.get(b.node).copied().unwrap_or(0.0).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    for _ in 0..challenge.max_backoffs {
        let flows = sparse_line_flows(challenge, state, &candidate, node_count, &touched);
        if soc_bounds_ok(challenge, state, &candidate) && flows_within_limits(challenge, &flows) {
            return candidate;
        }
        let Some((line, flow, excess)) = most_violated_from_flows(challenge, &flows) else {
            break;
        };
        let swapped = substitute_on_line(
            challenge,
            state,
            &mut candidate,
            line,
            flow,
            excess,
            &mut value_cache,
            &flows,
            &line_node_peak,
        );
        if !swapped
            && !soften_violated_line(
                challenge,
                state,
                &mut candidate,
                line,
                flow,
                excess,
                &mut value_cache,
            )
        {
            break;
        }
    }

    if action_is_valid(challenge, state, &candidate, node_count) {
        // Backoff succeeded. When halfspace is on, also project the *raw*
        // action and keep the higher step-profit feasible allocation.
        if halfspace_proj_enabled() {
            let mut hs = raw.to_vec();
            if project_halfspace(challenge, state, &mut hs, node_count)
                && action_is_valid(challenge, state, &hs, node_count)
            {
                let s_hs = estimated_step_profit(challenge, state, &hs);
                let s_bo = estimated_step_profit(challenge, state, &candidate);
                if s_hs > s_bo + 1e-12 {
                    return hs;
                }
            }
        }
        return candidate;
    }
    project_feasible_terminal(challenge, state, &mut candidate, node_count)
}

/// Grouped / cap-curtail / uniform-scale terminal rescue (shared by the
/// substitute-backoff path and the halfspace-projection path).
fn project_feasible_terminal(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    candidate: &mut [f64],
    node_count: usize,
) -> Vec<f64> {
    // Grouped pos/neg scale (energy_v1) and capacity-priority curtail both
    // produce feasible non-uniform allocations. When CAP_CURTAIL is on, score
    // both by `estimated_step_profit` and keep the winner so the instrument
    // actually competes on binding steps (not only the rare path where grouped
    // fails). Default OFF remains byte-identical to grouped-then-uniform.
    //
    // CAP_CURTAIL thesis: uniform scale taxes large efficient batteries and
    // small ones equally; profit-preserving order zeros *smallest* capacity
    // first (revenue linear in m — evidence/energy-generator-structure.md).
    let mut best_rescue: Option<(f64, Vec<f64>)> = None;
    let mut consider = |actions: Vec<f64>| {
        if !action_is_valid(challenge, state, &actions, node_count) {
            return;
        }
        let score = estimated_step_profit(challenge, state, &actions);
        let take = best_rescue
            .as_ref()
            .map_or(true, |(s, _)| score > *s + 1e-12);
        if take {
            best_rescue = Some((score, actions));
        }
    };
    if let Some(grouped) = grouped_scale_rescue(challenge, state, candidate, node_count) {
        consider(grouped);
    }
    if cap_curtail_enabled() {
        if let Some(curtailed) =
            capacity_priority_curtail(challenge, state, candidate, node_count)
        {
            consider(curtailed);
        }
    }
    // Additive halfspace: project the post-backoff candidate onto PTDF
    // halfspaces and compete by step profit. Does NOT replace substitute/soften
    // (that replacement was a measured −15–20% quality regression).
    if halfspace_proj_enabled() {
        let mut hs = candidate.to_vec();
        if project_halfspace(challenge, state, &mut hs, node_count) {
            consider(hs);
        }
    }
    if let Some((_, actions)) = best_rescue {
        return actions;
    }

    // Binary search largest scale in [0,1] — INCREMENTAL via PTDF linearity.
    //
    // Each probe previously re-ran `action_is_valid`, an O(lines x batteries)
    // PTDF pass, up to 32 times. But the probe scales the action UNIFORMLY, and
    // PTDF is linear:
    //
    //     flows(base + m*a) = base + m*(PTDF*a)
    //
    // So compute `delta = PTDF * a` ONCE and each probe becomes O(lines).
    // Since `lines` scales with `batteries` on these tracks (baseline 30/15,
    // capstone 300/100), the per-probe cost drops by a factor of `batteries`.
    //
    // The SOC/power feasibility half of `action_is_valid` is not linear in the
    // scale, so it is still checked per probe — but it is O(batteries), not
    // O(lines x batteries), and a uniform down-scale can only relax power
    // bounds that already held at scale 1.
    let mut nodal_delta = vec![0.0_f64; node_count];
    let mut touched: Vec<usize> = Vec::with_capacity(challenge.batteries.len());
    for (battery, value) in challenge.batteries.iter().zip(candidate.iter()) {
        if battery.node < node_count {
            nodal_delta[battery.node] += *value;
            touched.push(battery.node);
        }
    }
    touched.sort_unstable();
    touched.dedup();
    let delta_flows: Vec<f64> = (0..challenge.lines.len())
        .map(|line| {
            let row = &challenge.ptdf[line];
            touched
                .iter()
                .map(|&node| row[node] * nodal_delta[node])
                .sum::<f64>()
        })
        .collect();

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best_scaled = vec![0.0; challenge.batteries.len()];
    for _ in 0..32.min(challenge.max_backoffs.saturating_mul(2).max(8)) {
        let mid = 0.5 * (lo + hi);
        let flows_ok = challenge.lines.iter().enumerate().all(|(line, limit)| {
            (state.base_line_flows[line] + mid * delta_flows[line]).abs()
                <= limit.limit + f64::EPSILON
        });
        let scaled: Vec<f64> = candidate.iter().map(|a| a * mid).collect();
        if flows_ok && soc_bounds_ok(challenge, state, &scaled) {
            best_scaled = scaled;
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if action_is_valid(challenge, state, &best_scaled, node_count) {
        return best_scaled;
    }
    vec![0.0; challenge.batteries.len()]
}

fn cap_curtail_enabled() -> bool {
    matches!(
        gnosis_env_var("GNOSIS_CAP_CURTAIL")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

fn halfspace_proj_enabled() -> bool {
    matches!(
        gnosis_env_var("GNOSIS_HALFSPACE_PROJ")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

/// Portable extract of prior_art / prometheus `project_polytope`: alternating
/// projection onto box bounds and the most-violated PTDF halfspace.
///
/// Flow model matches ours: `f_l = base_l + Σ_b ptdf[l][node_b] · a_b` (no
/// separate slack subtraction — base already includes exo/slack balance).
/// Returns true when the final action is flow-feasible within `EPS_FLOW`.
fn project_halfspace(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &mut [f64],
    _node_count: usize,
) -> bool {
    const EPS_FLOW: f64 = 1e-6;
    let n_b = challenge.batteries.len();
    let n_lines = challenge.lines.len();
    if action.len() != n_b || n_lines == 0 {
        return false;
    }

    // sens[l][b] = ∂f_l / ∂a_b = ptdf[l][node_b]
    let mut sens = vec![vec![0.0_f64; n_b]; n_lines];
    for l in 0..n_lines {
        let row = &challenge.ptdf[l];
        for (b, battery) in challenge.batteries.iter().enumerate() {
            sens[l][b] = row.get(battery.node).copied().unwrap_or(0.0);
        }
    }

    let bounds: Vec<(f64, f64)> = challenge
        .batteries
        .iter()
        .enumerate()
        .map(|(i, battery)| power_bounds(battery, state.state_of_charge[i]))
        .collect();

    let max_iters = challenge.max_backoffs.max(8).min(64);
    for _ in 0..max_iters {
        for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
            *a = (*a).clamp(lo, hi);
        }
        let mut worst_l = usize::MAX;
        let mut worst_excess = 0.0_f64;
        let mut worst_sign = 0.0_f64;
        let mut worst_limit = 1.0_f64;
        for l in 0..n_lines {
            let mut f = state.base_line_flows.get(l).copied().unwrap_or(0.0);
            for b in 0..n_b {
                f += sens[l][b] * action[b];
            }
            let limit = challenge.lines[l].limit;
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
        for b in 0..n_b {
            action[b] -= worst_sign * mu * row[b];
        }
        charge_work(n_b as u64);
    }
    // Final clamp + verify.
    for (a, &(lo, hi)) in action.iter_mut().zip(bounds.iter()) {
        *a = (*a).clamp(lo, hi);
    }
    for l in 0..n_lines {
        let mut f = state.base_line_flows.get(l).copied().unwrap_or(0.0);
        for b in 0..n_b {
            f += sens[l][b] * action[b];
        }
        let limit = challenge.lines[l].limit;
        if f.abs() > limit * (1.0 + EPS_FLOW) + 1e-6 {
            return false;
        }
    }
    true
}

/// Zero the smallest-capacity batteries first until the remaining action is
/// network-feasible; then binary-scale the survivors if still needed.
///
/// Returns `None` only when even the all-zero action is rejected by the caller
/// path (caller already handles that). If progressive zeroing never finds a
/// non-zero feasible action, returns the uniform-scale result of the survivors
/// (may be zero).
fn capacity_priority_curtail(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    base: &[f64],
    node_count: usize,
) -> Option<Vec<f64>> {
    let n = challenge.batteries.len();
    if n == 0 || base.len() != n {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();
    // Smallest capacity first. Ties break by index for determinism.
    order.sort_by(|&i, &j| {
        challenge.batteries[i]
            .capacity
            .total_cmp(&challenge.batteries[j].capacity)
            .then(i.cmp(&j))
    });

    let mut action = base.to_vec();
    if action_is_valid(challenge, state, &action, node_count) {
        return Some(action);
    }

    // Progressive hard-zero of the smallest active batteries.
    for &i in &order {
        if action[i].abs() <= 1e-15 {
            continue;
        }
        action[i] = 0.0;
        if action_is_valid(challenge, state, &action, node_count) {
            return Some(action);
        }
    }

    // Survivors empty or still infeasible (shouldn't happen if zero is always
    // feasible when base network is). Fall through to caller's uniform scale
    // by returning None only if we made no progress — but we always zeroed all
    // nonzero, so return the zero action if valid.
    if action_is_valid(challenge, state, &action, node_count) {
        Some(action)
    } else {
        None
    }
}

fn grouped_scale_rescue(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    base: &[f64],
    node_count: usize,
) -> Option<Vec<f64>> {
    let mut pos_mag = 0.0;
    let mut neg_mag = 0.0;
    for &a in base {
        if a > 0.0 {
            pos_mag += a;
        } else {
            neg_mag += -a;
        }
    }
    if pos_mag <= 1e-12 && neg_mag <= 1e-12 {
        return None;
    }

    let mut best: Option<(f64, Vec<f64>)> = None;
    // Coarse 2D scale grid.
    for pi in 0..=8 {
        let px = pi as f64 / 8.0;
        for ni in 0..=8 {
            let nx = ni as f64 / 8.0;
            if px <= 1e-12 && nx <= 1e-12 {
                continue;
            }
            let scaled: Vec<f64> = base
                .iter()
                .map(|&a| {
                    if a > 0.0 {
                        a * px
                    } else if a < 0.0 {
                        a * nx
                    } else {
                        0.0
                    }
                })
                .collect();
            if !action_is_valid(challenge, state, &scaled, node_count) {
                continue;
            }
            let score = estimated_step_profit(challenge, state, &scaled);
            let better = best
                .as_ref()
                .map_or(true, |(s, _)| score > *s + 1e-12);
            if better {
                best = Some((score, scaled));
            }
        }
    }
    best.map(|(_, a)| a)
}

fn post_feasibility_improvement(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    mut action: Vec<f64>,
    node_count: usize,
    da_value_cache: &mut [Option<f64>],
) -> Vec<f64> {
    // energy_v1: expand high-value same-sign actions into remaining line headroom.
    // Rank by pure DA value so we do not expand RT noise into inventory drain.
    let mut flows = line_flows(challenge, state, &action, node_count);
    for _ in 0..4 {
        let mut order: Vec<(usize, f64)> = (0..challenge.batteries.len())
            .filter_map(|i| {
                let a = action[i];
                if a.abs() <= 1e-12 {
                    return None;
                }
                let direction = if a > 0.0 { 1.0 } else { -1.0 };
                // Memoised on (battery, direction). Distinct from the cache in
                // `project_feasible`: these calls use DaOnly / weight 0.0, a
                // different function of the same key, so the two must not share
                // storage. Same justification otherwise -- `directional_value`
                // reads only `state` and the price matrix, never `action`, so
                // its value is fixed for the step even as the ranking loop below
                // expands actions across its four passes.
                let key = i * 2 + usize::from(direction > 0.0);
                let v = match da_value_cache[key] {
                    Some(cached) => cached,
                    None => {
                        let computed = directional_value(
                            challenge,
                            state,
                            i,
                            direction,
                            0.0,
                            ResidualWeightMode::DaOnly,
                            &[],
                        );
                        da_value_cache[key] = Some(computed);
                        computed
                    }
                };
                if v > 1e-12 {
                    Some((i, v))
                } else {
                    None
                }
            })
            .collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut any = false;
        for (i, _) in order {
            let battery = &challenge.batteries[i];
            let node = battery.node;
            let (lo, hi) = power_bounds(battery, state.state_of_charge[i]);
            let current = action[i];
            if current.abs() <= 1e-12 {
                continue;
            }
            let (target, direction) = if current > 0.0 {
                (hi, 1.0)
            } else {
                (lo, -1.0)
            };
            let mut delta_max = direction * (target - current);
            if delta_max <= 1e-12 {
                continue;
            }
            for (l, line) in challenge.lines.iter().enumerate() {
                let ptdf = challenge.ptdf[l].get(node).copied().unwrap_or(0.0);
                let sens = ptdf * direction;
                if sens.abs() <= 1e-12 {
                    continue;
                }
                let headroom = if sens > 0.0 {
                    (line.limit - flows[l]) / sens
                } else {
                    (-line.limit - flows[l]) / sens
                };
                delta_max = delta_max.min(headroom);
            }
            if delta_max <= 1e-12 {
                continue;
            }
            let new_action = (current + direction * delta_max).clamp(lo, hi);
            let actual = new_action - current;
            if actual.abs() <= 1e-12 {
                continue;
            }
            for (l, _) in challenge.lines.iter().enumerate() {
                let ptdf = challenge.ptdf[l].get(node).copied().unwrap_or(0.0);
                flows[l] += ptdf * actual;
            }
            action[i] = new_action;
            any = true;
        }
        if !any {
            break;
        }
    }
    if action_is_valid(challenge, state, &action, node_count) {
        action
    } else {
        // Should not happen; fall back to zero rather than invalid.
        vec![0.0; challenge.batteries.len()]
    }
}

fn estimated_step_profit(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &[f64],
) -> f64 {
    let dt = DELTA_T;
    let mut total = 0.0;
    for (battery, &u) in challenge.batteries.iter().zip(action) {
        if u.abs() <= 1e-15 {
            continue;
        }
        let price = state.observed_prices.get(battery.node).copied().unwrap_or(0.0);
        let revenue = u * price * dt;
        let tx = KAPPA_TX * challenge.friction_weight * u.abs() * dt;
        let deg = KAPPA_DEG * ((u.abs() * dt) / battery.capacity.max(1e-9)).powi(2);
        total += revenue - tx - deg;
    }
    total
}

/// energy_v1 `estimated_action_value`: Σ |u| · directional_value(DA).
fn estimated_action_value_da(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    action: &[f64],
    shadows: &[f64],
) -> f64 {
    let mut total = 0.0;
    for (i, _) in challenge.batteries.iter().enumerate() {
        let u = action[i];
        if u.abs() <= 1e-12 {
            continue;
        }
        let direction = u.signum();
        let v = directional_value(
            challenge,
            state,
            i,
            direction,
            0.0,
            ResidualWeightMode::DaOnly,
            shadows,
        );
        total += u.abs() * v;
    }
    total
}

fn line_flows(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &[f64],
    node_count: usize,
) -> Vec<f64> {
    let mut nodal = vec![0.0; node_count];
    for (battery, &action) in challenge.batteries.iter().zip(actions) {
        if battery.node < node_count {
            nodal[battery.node] += action;
        }
    }
    challenge
        .lines
        .iter()
        .enumerate()
        .map(|(line_index, _)| {
            let added: f64 = challenge.ptdf[line_index]
                .iter()
                .zip(&nodal)
                .map(|(s, a)| s * a)
                .sum();
            state.base_line_flows[line_index] + added
        })
        .collect()
}

/// Distinct nodes hosting a battery. `nodal` is zero everywhere else, so every
/// PTDF product may skip the rest: the omitted terms are `sensitivity * 0.0`.
/// Hoisted out of the backoff loop -- it used to be rebuilt (and re-sorted) on
/// every call to `action_is_valid`, once per iteration per step.
/// Per-battery feasibility for ONE changed entry.
///
/// The bounds half of `action_is_valid`, restricted to a single battery. The
/// local searches below mutate two or three entries of an already-valid
/// incumbent, so every other battery's checks are known to pass and re-running
/// them is pure cost.
fn battery_action_ok(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    index: usize,
    action: f64,
) -> bool {
    if !action.is_finite() {
        return false;
    }
    let battery = &challenge.batteries[index];
    let soc = state.state_of_charge[index];
    let (lo, hi) = power_bounds(battery, soc);
    if action > hi + f64::EPSILON || action < lo - f64::EPSILON {
        return false;
    }
    let next = next_soc(battery, soc, action);
    let reserve = battery.capacity * battery.reserve_fraction;
    if next < -1e-6 || next > battery.capacity + 1e-6 {
        return false;
    }
    !(soc + 1e-9 >= reserve && next < reserve - 1e-6)
}

/// Line feasibility for a trial expressed as a DELTA on an incumbent's flows.
///
/// Flows are affine in the actions, so changing entries at `deltas` (node, amount)
/// moves line `l` by `sum ptdf[l][node] * amount`. A trial that touches two or
/// three batteries therefore costs O(lines x 3) instead of the O(lines x touched)
/// -- 300 x 63 on capstone -- that a full `action_is_valid` pass costs.
///
/// `base_flows` MUST be the flows of the current incumbent, recomputed whenever
/// the incumbent moves. Each trial's delta is taken FRESH from that fixed base
/// and never chained onto a previous delta: chaining accumulates floating-point
/// drift across iterations, which is exactly what drove `congested` below the
/// greedy baseline when it was tried in the backoff loop.
fn trial_flows_ok(challenge: &EnergyChallenge, base_flows: &[f64], deltas: &[(usize, f64)]) -> bool {
    charge_work((challenge.lines.len() * deltas.len()) as u64);
    challenge.lines.iter().enumerate().all(|(l, line)| {
        let row = &challenge.ptdf[l];
        let mut flow = base_flows[l];
        for &(node, delta) in deltas {
            flow += row.get(node).copied().unwrap_or(0.0) * delta;
        }
        flow.abs() <= line.limit + f64::EPSILON
    })
}

fn touched_nodes(challenge: &EnergyChallenge, node_count: usize) -> Vec<usize> {
    let mut touched: Vec<usize> = challenge
        .batteries
        .iter()
        .map(|battery| battery.node)
        .filter(|node| *node < node_count)
        .collect();
    touched.sort_unstable();
    touched.dedup();
    touched
}

/// `line_flows` restricted to battery-hosting nodes. Exactly equivalent to the
/// dense form; see `touched_nodes`.
fn sparse_line_flows(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &[f64],
    node_count: usize,
    touched: &[usize],
) -> Vec<f64> {
    let mut nodal = vec![0.0; node_count];
    for (battery, &action) in challenge.batteries.iter().zip(actions) {
        if battery.node < node_count {
            nodal[battery.node] += action;
        }
    }
    charge_work((challenge.lines.len() * touched.len()) as u64);
    challenge
        .lines
        .iter()
        .enumerate()
        .map(|(line_index, _)| {
            let row = &challenge.ptdf[line_index];
            let added: f64 = touched.iter().map(|&node| row[node] * nodal[node]).sum();
            state.base_line_flows[line_index] + added
        })
        .collect()
}

/// Flow half of `action_is_valid`, read off an already-maintained vector.
fn flows_within_limits(challenge: &EnergyChallenge, flows: &[f64]) -> bool {
    challenge
        .lines
        .iter()
        .zip(flows)
        .all(|(line, flow)| flow.abs() <= line.limit + f64::EPSILON)
}

/// `most_violated_line_detail` against a maintained flow vector.
fn most_violated_from_flows(
    challenge: &EnergyChallenge,
    flows: &[f64],
) -> Option<(usize, f64, f64)> {
    let mut worst: Option<(f64, usize, f64)> = None;
    for (line_index, line) in challenge.lines.iter().enumerate() {
        let flow = flows[line_index];
        let excess = flow.abs() - line.limit;
        if excess > f64::EPSILON && worst.is_none_or(|(known, _, _)| excess > known) {
            worst = Some((excess, line_index, flow));
        }
    }
    worst.map(|(excess, line, flow)| (line, flow, excess))
}

#[allow(dead_code)]
fn most_violated_line_detail(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &[f64],
    node_count: usize,
) -> Option<(usize, f64, f64)> {
    let flows = line_flows(challenge, state, actions, node_count);
    let mut worst: Option<(f64, usize, f64)> = None;
    for (line_index, line) in challenge.lines.iter().enumerate() {
        let flow = flows[line_index];
        let excess = flow.abs() - line.limit;
        if excess > f64::EPSILON {
            if worst.is_none_or(|(known, _, _)| excess > known) {
                worst = Some((excess, line_index, flow));
            }
        }
    }
    worst.map(|(excess, line, flow)| (line, flow, excess))
}

/// Move same-sign power from a high-harm battery to a lower-harm one (energy_v1 style).
fn substitute_on_line(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &mut [f64],
    line: usize,
    flow: f64,
    excess: f64,
    value_cache: &mut [Option<f64>],
    flows: &[f64],
    line_node_peak: &[f64],
) -> bool {
    let signed = flow.signum();
    if signed.abs() <= f64::EPSILON || challenge.batteries.len() < 2 {
        return false;
    }
    // `flows` is the vector `project_feasible` computed for THIS candidate at
    // the top of this iteration. Recomputing it here was a second full PTDF
    // pass over identical inputs; the sparse and dense forms agree bit-for-bit
    // (see the note in `project_feasible`), so this is the same vector.
    //
    // PRE-FILTER THE LINE SCAN. The inner loop below runs once per (i, j)
    // battery pair -- O(batteries^2 x lines) -- and almost all of it is spent
    // proving that slack lines are slack. A pair moves `amount` of power from
    // node `from` to node `to`, so line `l` shifts by
    // `delta = sign * (ptdf[l][to] - ptdf[l][from])`, and `amount` is bounded
    // above by `reducible = |actions[i]|` for every pair. Therefore
    //
    //     |delta| <= 2 * max_{n in battery nodes} |ptdf[l][n]| = bound_l
    //     amount  <= max_i |actions[i]|                        = amount_max
    //
    // and a line whose remaining headroom in BOTH directions exceeds
    // `bound_l * amount_max` cannot bind for ANY pair. Those lines are dropped
    // once, in O(lines x batteries), and the per-pair scan then walks only the
    // survivors. Exact: the dropped constraints are provably non-binding, not
    // approximated. Lines already over their limit have negative headroom and
    // are never dropped.
    // `line_node_peak[l]` is max_{n in battery nodes} |ptdf[l][n]|, a property of
    // the network and the fleet layout ONLY -- it moves with neither `flows` nor
    // `actions`. It is hoisted to `project_feasible` so it is built once per
    // projection instead of once per backoff iteration; this loop previously
    // rebuilt it O(lines x batteries) times per call, up to max_backoffs times
    // per step.
    let amount_max = actions.iter().fold(0.0_f64, |acc, a| acc.max(a.abs()));
    let scan_lines: Vec<usize> = (0..challenge.lines.len())
        .filter(|&l| {
            let reach = 2.0 * line_node_peak[l] * amount_max;
            let limit = challenge.lines[l].limit;
            !((limit - flows[l]) >= reach && (limit + flows[l]) >= reach)
        })
        .collect();
    let mut best: Option<(usize, usize, f64, f64, f64)> = None;

    // `directional_value` is memoised on (battery, sign) in a cache OWNED BY
    // `project_feasible` and shared with `soften_violated_line`, so it survives
    // every backoff iteration of the step -- see the note at its allocation.

    for i in 0..challenge.batteries.len() {
        let from_action = actions[i];
        if from_action.abs() <= f64::EPSILON {
            continue;
        }
        let sign = from_action.signum();
        let from_node = challenge.batteries[i].node;
        let harm_from = signed
            * challenge.ptdf[line].get(from_node).copied().unwrap_or(0.0)
            * sign;
        if harm_from <= f64::EPSILON {
            continue;
        }
        let reducible = from_action.abs();
        let from_key = i * 2 + usize::from(sign > 0.0);
        let value_from = match value_cache[from_key] {
            Some(cached) => cached,
            None => {
                let computed = directional_value(
                    challenge,
                    state,
                    i,
                    sign,
                    challenge.residual_weight,
                    ResidualWeightMode::Blend,
                    &[],
                );
                value_cache[from_key] = Some(computed);
                computed
            }
        };

        for j in 0..challenge.batteries.len() {
            if i == j {
                continue;
            }
            let to_action = actions[j];
            if to_action * sign < -f64::EPSILON {
                continue;
            }
            let to_node = challenge.batteries[j].node;
            let harm_to = signed
                * challenge.ptdf[line].get(to_node).copied().unwrap_or(0.0)
                * sign;
            let relief_per_unit = harm_from - harm_to;
            if relief_per_unit <= f64::EPSILON {
                continue;
            }
            let bat_j = &challenge.batteries[j];
            let (lo_j, hi_j) = power_bounds(bat_j, state.state_of_charge[j]);
            let headroom = if sign > 0.0 {
                (hi_j - to_action.max(0.0)).max(0.0)
            } else {
                (-lo_j + to_action.min(0.0)).max(0.0)
            };
            if headroom <= f64::EPSILON {
                continue;
            }
            let to_key = j * 2 + usize::from(sign > 0.0);
            let value_to = match value_cache[to_key] {
                Some(cached) => cached,
                None => {
                    let computed = directional_value(
                        challenge,
                        state,
                        j,
                        sign,
                        challenge.residual_weight,
                        ResidualWeightMode::Blend,
                        &[],
                    );
                    value_cache[to_key] = Some(computed);
                    computed
                }
            };
            let mut amount = reducible.min(headroom).min(excess / relief_per_unit);
            if amount <= f64::EPSILON {
                continue;
            }
            for &l in &scan_lines {
                let line_spec = &challenge.lines[l];
                let delta = sign
                    * (challenge.ptdf[l].get(to_node).copied().unwrap_or(0.0)
                        - challenge.ptdf[l].get(from_node).copied().unwrap_or(0.0));
                if delta > f64::EPSILON {
                    amount = amount.min((line_spec.limit - flows[l]) / delta);
                } else if delta < -f64::EPSILON {
                    amount = amount.min((-line_spec.limit - flows[l]) / delta);
                }
                if amount <= f64::EPSILON {
                    break;
                }
            }
            if amount <= f64::EPSILON {
                continue;
            }
            let loss_per_relief = (value_from - value_to) / relief_per_unit;
            let total_relief = amount * relief_per_unit;
            let better = best.as_ref().is_none_or(|b| {
                loss_per_relief < b.4 - 1e-12
                    || ((loss_per_relief - b.4).abs() <= 1e-12 && total_relief > b.3 + 1e-12)
            });
            if better {
                best = Some((i, j, sign, amount, loss_per_relief));
            }
        }
    }

    let Some((from, to, sign, amount, _)) = best else {
        return false;
    };
    let bat_from = &challenge.batteries[from];
    let bat_to = &challenge.batteries[to];
    let (lo_f, hi_f) = power_bounds(bat_from, state.state_of_charge[from]);
    let (lo_t, hi_t) = power_bounds(bat_to, state.state_of_charge[to]);
    let from_new = (actions[from] - sign * amount).clamp(lo_f, hi_f);
    let to_new = (actions[to] + sign * amount).clamp(lo_t, hi_t);
    let actual = ((actions[from] - from_new) * sign)
        .max(0.0)
        .min(((to_new - actions[to]) * sign).max(0.0));
    if actual <= f64::EPSILON {
        return false;
    }
    actions[from] = (actions[from] - sign * actual).clamp(lo_f, hi_f);
    actions[to] = (actions[to] + sign * actual).clamp(lo_t, hi_t);
    true
}

/// Reduce actions that worsen the violated line, preferring low-value batteries.
fn soften_violated_line(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &mut [f64],
    line: usize,
    flow: f64,
    excess: f64,
    value_cache: &mut [Option<f64>],
) -> bool {
    let signed = flow.signum();
    if signed.abs() <= f64::EPSILON {
        return false;
    }
    let mut ranked: Vec<(usize, f64, f64)> = Vec::new();
    for (index, battery) in challenge.batteries.iter().enumerate() {
        if actions[index].abs() <= f64::EPSILON {
            continue;
        }
        let ptdf = challenge.ptdf[line]
            .get(battery.node)
            .copied()
            .unwrap_or(0.0);
        let contribution = signed * ptdf * actions[index];
        if contribution <= f64::EPSILON {
            continue;
        }
        let direction = if actions[index] > 0.0 { 1.0 } else { -1.0 };
        let key = index * 2 + usize::from(direction > 0.0);
        let value = match value_cache[key] {
            Some(cached) => cached,
            None => {
                let computed = directional_value(
                    challenge,
                    state,
                    index,
                    direction,
                    challenge.residual_weight,
                    ResidualWeightMode::Blend,
                    &[],
                );
                value_cache[key] = Some(computed);
                computed
            }
        }
        .max(0.0);

        // NOT degradation-adjusted, and measured rather than assumed: see
        // evidence/energy-generator-structure.md. Degradation is ~1e-5 of
        // revenue at these constants, so it cannot reorder this ranking.
        let relief = (signed * ptdf).abs().max(1e-12);
        let loss_per_relief = value / relief;
        ranked.push((index, loss_per_relief, ptdf));
    }
    if ranked.is_empty() {
        return false;
    }
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut remaining = excess;
    let mut changed = false;
    for (index, _, ptdf) in ranked {
        if remaining <= f64::EPSILON {
            break;
        }
        let sensitivity = signed * ptdf;
        if sensitivity.abs() <= f64::EPSILON {
            continue;
        }
        let current_contrib = sensitivity * actions[index];
        if current_contrib <= f64::EPSILON {
            continue;
        }
        let target_contrib = (current_contrib - remaining).max(0.0);
        let new_action = target_contrib / sensitivity;
        let battery = &challenge.batteries[index];
        let (lo, hi) = power_bounds(battery, state.state_of_charge[index]);
        let clamped = new_action.clamp(lo, hi);
        let moved = if actions[index] > 0.0 {
            clamped.clamp(0.0, actions[index])
        } else {
            clamped.clamp(actions[index], 0.0)
        };
        let actual_reduction = sensitivity * (actions[index] - moved);
        if actual_reduction > f64::EPSILON {
            remaining -= actual_reduction;
            actions[index] = moved;
            changed = true;
        }
    }
    changed
}

/// The SOC / power-bound half of `action_is_valid`, without the PTDF pass.
///
/// Split out so the scale binary search can check bounds per probe while
/// evaluating line flows incrementally. Identical predicate to the first half of
/// `action_is_valid`.
fn soc_bounds_ok(challenge: &EnergyChallenge, state: &OnlineState, actions: &[f64]) -> bool {
    if actions.len() != challenge.batteries.len()
        || actions.iter().any(|action| !action.is_finite())
    {
        return false;
    }
    for ((battery, &soc), &action) in challenge
        .batteries
        .iter()
        .zip(&state.state_of_charge)
        .zip(actions)
    {
        let (lo, hi) = power_bounds(battery, soc);
        if action > hi + f64::EPSILON || action < lo - f64::EPSILON {
            return false;
        }
        let next = next_soc(battery, soc, action);
        let reserve = battery.capacity * battery.reserve_fraction;
        if next < -1e-6 || next > battery.capacity + 1e-6 {
            return false;
        }
        if soc + 1e-9 >= reserve && next < reserve - 1e-6 {
            return false;
        }
    }
    true
}

pub fn action_is_valid(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &[f64],
    node_count: usize,
) -> bool {
    let touched = touched_nodes(challenge, node_count);
    action_is_valid_with(challenge, state, actions, node_count, &touched)
}

/// `action_is_valid` against a PRECOMPUTED touched-node list.
///
/// The list is `sort_unstable` + `dedup` over every battery node -- a sort of
/// `batteries` elements. It depends only on the fleet layout, which is fixed for
/// the whole solve, yet the original rebuilt and re-sorted it on every call.
///
/// That is hot: on capstone the local search calls this 308,084 times over 192
/// steps (~1,605 per step, two 2-opt passes over a ~200-pair budget at four
/// alphas each), so the sort ran 308,084 times to produce one unchanging vector.
///
/// EXACT -- identical list, so identical iteration order and identical sums.
pub fn action_is_valid_with(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    actions: &[f64],
    node_count: usize,
    touched: &[usize],
) -> bool {
    if actions.len() != challenge.batteries.len()
        || actions.iter().any(|action| !action.is_finite())
    {
        return false;
    }

    let mut nodal_actions = vec![0.0; node_count];
    for ((battery, &soc), &action) in challenge
        .batteries
        .iter()
        .zip(&state.state_of_charge)
        .zip(actions)
    {
        let (lo, hi) = power_bounds(battery, soc);
        if action > hi + f64::EPSILON || action < lo - f64::EPSILON {
            return false;
        }
        let next = next_soc(battery, soc, action);
        let reserve = battery.capacity * battery.reserve_fraction;
        // Hard bounds: [0, capacity]. Soft reserve is enforced via power_bounds
        // (no discharge below reserve); allow charge from a below-reserve start.
        if next < -1e-6 || next > battery.capacity + 1e-6 {
            return false;
        }
        if soc + 1e-9 >= reserve && next < reserve - 1e-6 {
            return false;
        }
        if battery.node < node_count {
            nodal_actions[battery.node] += action;
        }
    }

    // SPARSE PTDF product.
    //
    // `nodal_actions` is zero everywhere except nodes that host a battery, so
    // the dense product multiplied through every node to add nothing. Profiling
    // (macOS `sample`, dense track) put `project_feasible` at ~40% of policy
    // time with this row product underneath it -- after lookahead, backoffs and
    // per-step reconstruction had each been measured NOT to be the cost
    // (~0%, ~5%, ~0.003% of real fuel respectively).
    //
    // Touched nodes are supplied by the caller. Battery nodes are a subset of
    // all nodes, so iterating only those is exactly equivalent: the skipped
    // terms are `sensitivity * 0.0`.

    // Charge the arithmetic actually performed: one multiply-add per (line,
    // touched node). This is the unit TIG meters -- instructions -- rather than
    // wall time, which the sparse rewrite showed can move 3.9x while fuel moves
    // 0.1%. Validated against fuel_consumed before use; see `work_units`.
    charge_work((challenge.lines.len() * touched.len()) as u64);

    challenge.lines.iter().enumerate().all(|(line_index, line)| {
        let row = &challenge.ptdf[line_index];
        let added_flow: f64 = touched
            .iter()
            .map(|&node| row[node] * nodal_actions[node])
            .sum();
        (state.base_line_flows[line_index] + added_flow).abs() <= line.limit + f64::EPSILON
    })
}

fn validate(
    challenge: &EnergyChallenge,
    state: &OnlineState,
) -> Result<(), PolicyError> {
    let Some(first_prices) = challenge.day_ahead_prices.first() else {
        return Err(PolicyError::InvalidChallenge("empty price horizon"));
    };
    let node_count = first_prices.len();
    if node_count == 0
        || state.time >= challenge.day_ahead_prices.len()
        || challenge
            .day_ahead_prices
            .iter()
            .any(|row| row.len() != node_count || row.iter().any(|value| !value.is_finite()))
    {
        return Err(PolicyError::InvalidChallenge("invalid price matrix"));
    }
    if !challenge.residual_weight.is_finite()
        || !challenge.deadband.is_finite()
        || challenge.deadband < 0.0
        || !challenge.congestion_weight.is_finite()
        || challenge.congestion_weight < 0.0
        || !challenge.friction_weight.is_finite()
        || challenge.friction_weight < 0.0
    {
        return Err(PolicyError::InvalidChallenge("invalid policy parameter"));
    }
    if challenge.ptdf.len() != challenge.lines.len()
        || challenge
            .ptdf
            .iter()
            .any(|row| row.len() != node_count || row.iter().any(|value| !value.is_finite()))
        || challenge
            .lines
            .iter()
            .any(|line| !line.limit.is_finite() || line.limit < 0.0)
    {
        return Err(PolicyError::InvalidChallenge("invalid network model"));
    }
    if challenge.batteries.iter().any(|battery| {
        battery.node >= node_count
            || !battery.capacity.is_finite()
            || battery.capacity < 0.0
            || !battery.max_charge.is_finite()
            || battery.max_charge < 0.0
            || !battery.max_discharge.is_finite()
            || battery.max_discharge < 0.0
            || !battery.charge_efficiency.is_finite()
            || battery.charge_efficiency <= 0.0
            || battery.charge_efficiency > 1.0
            || !battery.discharge_efficiency.is_finite()
            || battery.discharge_efficiency <= 0.0
            || battery.discharge_efficiency > 1.0
            || !battery.reserve_fraction.is_finite()
            || !(0.0..=1.0).contains(&battery.reserve_fraction)
    }) {
        return Err(PolicyError::InvalidChallenge("invalid battery"));
    }
    if state.state_of_charge.len() != challenge.batteries.len()
        || state
            .state_of_charge
            .iter()
            .zip(&challenge.batteries)
            .any(|(&soc, battery)| !soc.is_finite() || soc < 0.0 || soc > battery.capacity)
    {
        return Err(PolicyError::InvalidState("invalid state of charge"));
    }
    if state.observed_prices.len() != node_count
        || state.observed_prices.iter().any(|value| !value.is_finite())
        || state.base_line_flows.len() != challenge.lines.len()
        || state.base_line_flows.iter().any(|value| !value.is_finite())
    {
        return Err(PolicyError::InvalidState("invalid observation dimensions"));
    }
    if state
        .base_line_flows
        .iter()
        .zip(&challenge.lines)
        .any(|(&flow, line)| flow.abs() > line.limit + f64::EPSILON)
    {
        return Err(PolicyError::InvalidState("infeasible base network flow"));
    }
    Ok(())
}

fn reachable_slot_count(energy_mwh: f64, step_energy_mwh: f64, steps_remaining: usize) -> usize {
    if steps_remaining == 0 {
        return 0;
    }
    (((energy_mwh / step_energy_mwh.max(1e-12)).ceil() as usize).max(1)).min(steps_remaining)
}

fn scarcity_price_summary(prices: &[f64], low_slots: usize, high_slots: usize) -> (f64, f64, f64, f64) {
    if prices.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted = prices.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let low_slots = low_slots.max(1).min(n);
    let high_slots = high_slots.max(1).min(n);
    let avg_low = sorted.iter().take(low_slots).sum::<f64>() / low_slots as f64;
    let avg_high = sorted.iter().rev().take(high_slots).sum::<f64>() / high_slots as f64;
    let low_threshold = sorted[low_slots - 1];
    let high_threshold = sorted[n - high_slots];
    (avg_low, avg_high, low_threshold, high_threshold)
}

/// Expected economic value of moving one MW in `direction` (+ discharge / − charge).
/// energy_v1 scarcity-spread structure + RT residual blend + congestion dual + friction.
fn directional_value(
    challenge: &EnergyChallenge,
    state: &OnlineState,
    battery_idx: usize,
    direction: f64,
    residual_weight: f64,
    mode: ResidualWeightMode,
    shadows: &[f64],
) -> f64 {
    let battery = &challenge.batteries[battery_idx];
    let t = state.time;
    if t >= challenge.day_ahead_prices.len() {
        return 0.0;
    }
    let node = battery.node;
    let day_ahead = challenge.day_ahead_prices[t][node];
    let residual = state.observed_prices[node] - day_ahead;
    let current_price = match mode {
        ResidualWeightMode::DaOnly => day_ahead,
        ResidualWeightMode::Blend => day_ahead + residual_weight * residual,
    };
    let steps_remaining = (challenge.day_ahead_prices.len() - t).max(1);
    let soc = state.state_of_charge[battery_idx];
    let reserve = battery.capacity * battery.reserve_fraction;
    let soc_max = battery.capacity;
    let soc_min = reserve;
    let soc_range = (soc_max - soc_min).max(1e-12);
    let soc_level = ((soc - soc_min) / soc_range).clamp(0.0, 1.0);
    let energy_above_min = (soc - soc_min).max(0.0);
    let energy_below_max = (soc_max - soc).max(0.0);
    // Match energy_v1 step-energy convention (power × efficiency, no Δt).
    let discharge_step =
        (battery.max_discharge * battery.discharge_efficiency).max(1e-12);
    let charge_step = (battery.max_charge * battery.charge_efficiency).max(1e-12);
    let release_pressure =
        (energy_above_min / (steps_remaining as f64 * discharge_step)).clamp(0.0, 2.0);
    let refill_pressure =
        (energy_below_max / (steps_remaining as f64 * charge_step)).clamp(0.0, 2.0);
    let horizon_urgency = 1.0 / steps_remaining as f64;

    let future_prices: Vec<f64> = challenge.day_ahead_prices[t + 1..]
        .iter()
        .map(|row| row[node])
        .collect();

    // Soft congestion only (energy_v1 directional_value has no κ_tx; keep duals mild).
    let cong = if shadows.is_empty() {
        0.0
    } else {
        0.35 * congestion_cost(challenge, battery_idx, direction, shadows)
    };

    if future_prices.is_empty() {
        return if direction > 0.0 {
            if energy_above_min <= 1e-12 {
                0.0
            } else {
                current_price * battery.discharge_efficiency - cong
                    + 0.25 * current_price.abs() * release_pressure
            }
        } else if energy_below_max <= 1e-12 {
            0.0
        } else {
            -(current_price / battery.charge_efficiency.max(1e-12)) - cong
        };
    }

    let opportunity_steps = future_prices.len();
    if direction > 0.0 {
        if energy_above_min <= 1e-12 {
            return 0.0;
        }
        let sell_slots = reachable_slot_count(
            energy_above_min.max(discharge_step),
            discharge_step,
            opportunity_steps,
        );
        let refill_energy = (energy_below_max + charge_step).min(soc_range);
        let buy_slots =
            reachable_slot_count(refill_energy.max(charge_step), charge_step, opportunity_steps);
        let (avg_low, avg_high, low_threshold, high_threshold) =
            scarcity_price_summary(&future_prices, buy_slots, sell_slots);
        let realizable_spread = (avg_high - avg_low).max(0.0);
        // Match energy_v1 coefficients exactly for DA scarcity core.
        let immediate_value = current_price * battery.discharge_efficiency;
        let continuation_value = avg_high * battery.discharge_efficiency;
        let peak_bonus =
            (current_price - high_threshold).max(0.0) * battery.discharge_efficiency;
        let recycle_bonus =
            (immediate_value - low_threshold / battery.charge_efficiency.max(1e-12)).max(0.0);
        let slot_need = sell_slots as f64 / opportunity_steps as f64;
        immediate_value
            - continuation_value
            - cong
            + 0.28 * peak_bonus
            + 0.10 * recycle_bonus
            + realizable_spread * (0.40 * soc_level + 0.30 * release_pressure + 0.15 * slot_need)
            + 0.12 * immediate_value.abs() * release_pressure * horizon_urgency
    } else {
        if energy_below_max <= 1e-12 {
            return 0.0;
        }
        let sell_energy = (energy_above_min + discharge_step).min(soc_range);
        let sell_slots =
            reachable_slot_count(sell_energy.max(discharge_step), discharge_step, opportunity_steps);
        let buy_slots =
            reachable_slot_count(energy_below_max.max(charge_step), charge_step, opportunity_steps);
        let (avg_low, avg_high, low_threshold, _) =
            scarcity_price_summary(&future_prices, buy_slots, sell_slots);
        let realizable_spread = (avg_high - avg_low).max(0.0);
        let immediate_cost = current_price / battery.charge_efficiency.max(1e-12);
        let future_sale_value = avg_high * battery.discharge_efficiency;
        let cheap_now_bonus =
            (low_threshold - current_price).max(0.0) / battery.charge_efficiency.max(1e-12);
        let cycle_margin =
            (future_sale_value - avg_low / battery.charge_efficiency.max(1e-12)).max(0.0);
        let slot_need = sell_slots.max(buy_slots) as f64 / opportunity_steps as f64;
        future_sale_value
            - immediate_cost
            - cong
            + 0.28 * cheap_now_bonus
            + 0.10 * cycle_margin
            + realizable_spread
                * (0.40 * (1.0 - soc_level) + 0.30 * refill_pressure + 0.15 * slot_need)
            + 0.12 * refill_pressure * horizon_urgency * current_price.abs()
    }
}

fn signal_strength(magnitude: f64, deadband: f64) -> f64 {
    let excess = (magnitude - deadband).max(0.0);
    excess / (1.0 + excess)
}

// NOTE: this block must stay ABOVE the first `#[cfg(test)]`.
// `scripts/package-submissions.mjs` strips the overlay at the first
// occurrence of that marker, so anything below it never reaches the
// packaged kernel — it compiled locally and vanished in the overlay,
// surfacing only as "cannot find function `step_profit` in module
// `kernel`" from a build against the pin.
// ─── Clairvoyant bound (diagnostic) ─────────────────────────────────────────

/// Spec degradation exponent.
const BETA_DEG: f64 = 2.0;

// Evaluator constants already declared at the top of this module, mirroring
// `tig-challenges/src/energy_arbitrage/constants.rs`.

/// Per-step profit for one battery, matching the pinned evaluator exactly:
/// `u·λ·Δt − κ_tx·|u|·Δt − κ_deg·(|u|·Δt / capacity)^β`.
pub fn step_profit(u: f64, price: f64, nominal_capacity: f64) -> f64 {
    if u == 0.0 || nominal_capacity <= 0.0 {
        return 0.0;
    }
    let revenue = u * price * DELTA_T;
    let abs_u = u.abs();
    let tx = KAPPA_TX * abs_u * DELTA_T;
    let deg = KAPPA_DEG * ((abs_u * DELTA_T) / nominal_capacity).powf(BETA_DEG);
    revenue - tx - deg
}

/// **Clairvoyant upper bound** on total profit for a realised price path.
///
/// Energy is an *online* problem: prices come from a hidden seed and the policy
/// sees only `observed_prices` at each step. So "how far from optimal are we"
/// cannot be asked the way it is for knapsack, VRP or job — there is no known
/// instance to bound. What can be asked, after the fact, is: **of the profit
/// that was available on the path that actually occurred, what fraction did the
/// policy capture?** A quality of -25.9% vs live #1 cannot distinguish "our
/// policy is weak" from "this path was mostly unpredictable"; this can.
///
/// The bound relaxes the network. Profit is **separable** across batteries —
/// each depends only on its own action and its own node's price, with no
/// cross-battery term — and the only coupling is line limits, so dropping those
/// and optimising each battery independently can only over-state achievable
/// profit. That makes the sum a valid ceiling.
///
/// Each battery is then a clairvoyant single-battery arbitrage, solved exactly
/// by DP over discretised state of charge against the realised prices at its
/// node. `levels` sets the discretisation; finer is tighter but slower, and a
/// coarser grid still bounds because the DP maximises over a subset of actions
/// — under-counting achievable profit, never over-counting.
pub fn clairvoyant_profit_bound(
    challenge: &EnergyChallenge,
    realised_prices: &[Vec<f64>],
    levels: usize,
) -> f64 {
    clairvoyant_profit_bound_for(&challenge.batteries, realised_prices, levels)
}

/// Same bound, taking only the batteries — the sole input it needs.
pub fn clairvoyant_profit_bound_for(
    batteries: &[Battery],
    realised_prices: &[Vec<f64>],
    levels: usize,
) -> f64 {
    let levels = levels.max(2);
    if realised_prices.is_empty() {
        return 0.0;
    }
    batteries
        .iter()
        .map(|battery| single_battery_clairvoyant(battery, realised_prices, levels))
        .sum()
}

/// Plan on one price series, SCORE on another.
///
/// The clairvoyant bound plans and scores on the SAME (realised) series, so it
/// assumes foresight the challenge withholds and is unreachable by construction.
/// This variant builds the value function from `plan_prices` -- the day-ahead
/// matrix, known in full at t=0 -- then forward-simulates the resulting policy
/// and accumulates profit at `score_prices`, the realised path.
///
/// That is a real, executable strategy: plan upfront from information genuinely
/// available, then execute blindly. Its profit is therefore ACHIEVABLE, which
/// makes it a reference our policy can be measured against rather than another
/// unreachable ceiling.
///
/// HONEST LIMIT: like the clairvoyant bound, this ignores network coupling --
/// no line limits, no PTDF. So it is achievable only if the resulting schedule
/// happens to be flow-feasible, and it is properly read as an UPPER bound on
/// perfect day-ahead planning. That asymmetry is what makes it useful: if it
/// comes in BELOW our policy, then even an unconstrained perfect DA planner
/// loses to us, which is a strong statement. If it comes in far above, the
/// headroom is in the planning and is worth chasing.
/// The SCHEDULE the day-ahead planner would execute, per step per battery.
///
/// Same forward pass as `da_plan_realised_score`, emitting the actions instead
/// of the profit, so the caller can test them against the network the planner
/// does not model.
pub fn da_plan_schedule(
    batteries: &[Battery],
    plan_prices: &[Vec<f64>],
    levels: usize,
) -> Vec<Vec<f64>> {
    let levels = levels.max(2);
    let steps = plan_prices.len();
    let mut schedule = vec![vec![0.0_f64; batteries.len()]; steps];
    for (b, battery) in batteries.iter().enumerate() {
        let actions = single_battery_da_plan_actions(battery, plan_prices, levels);
        for (t, u) in actions.into_iter().enumerate() {
            if t < steps {
                schedule[t][b] = u;
            }
        }
    }
    schedule
}

fn single_battery_da_plan_actions(
    battery: &Battery,
    plan_prices: &[Vec<f64>],
    levels: usize,
) -> Vec<f64> {
    let steps = plan_prices.len();
    let capacity = battery.capacity;
    if capacity <= 0.0 || steps == 0 {
        return vec![0.0; steps];
    }
    let soc_min = capacity * battery.reserve_fraction.clamp(0.0, 1.0);
    let soc_max = capacity;
    if soc_max <= soc_min {
        return vec![0.0; steps];
    }
    let grid = |k: usize| soc_min + (soc_max - soc_min) * (k as f64) / ((levels - 1) as f64);
    let feasible = |soc: f64| -> (f64, f64) {
        let headroom = (soc_max - soc).max(0.0);
        let available = (soc - soc_min).max(0.0);
        let mc = if battery.charge_efficiency > 0.0 {
            (headroom / (battery.charge_efficiency * DELTA_T)).min(battery.max_charge)
        } else { 0.0 }.max(0.0);
        let md = if battery.discharge_efficiency > 0.0 {
            (available * battery.discharge_efficiency / DELTA_T).min(battery.max_discharge)
        } else { 0.0 }.max(0.0);
        (mc, md)
    };
    let step_soc = |soc: f64, u: f64| -> f64 {
        let delta = if u > 0.0 {
            -(u * DELTA_T) / battery.discharge_efficiency
        } else {
            -u * battery.charge_efficiency * DELTA_T
        };
        (soc + delta).clamp(soc_min, soc_max)
    };
    let mut layers: Vec<Vec<f64>> = vec![vec![0.0_f64; levels]; steps + 1];
    for t in (0..steps).rev() {
        let price = plan_prices[t].get(battery.node).copied().unwrap_or(0.0);
        for k in 0..levels {
            let soc = grid(k);
            let (mc, md) = feasible(soc);
            let mut best = layers[t + 1][k];
            for a in 0..levels {
                let frac = (a as f64) / ((levels - 1) as f64);
                for &u in &[-mc * frac, md * frac] {
                    if u == 0.0 { continue; }
                    let sn = step_soc(soc, u);
                    let kn = (((sn - soc_min) / (soc_max - soc_min)) * ((levels - 1) as f64))
                        .round().clamp(0.0, (levels - 1) as f64) as usize;
                    let cand = step_profit(u, price, battery.nominal_capacity) + layers[t + 1][kn];
                    if cand > best { best = cand; }
                }
            }
            layers[t][k] = best;
        }
    }
    let mut k = (((0.5) * ((levels - 1) as f64)).round() as usize).min(levels - 1);
    let mut out = Vec::with_capacity(steps);
    for t in 0..steps {
        let price = plan_prices[t].get(battery.node).copied().unwrap_or(0.0);
        let soc = grid(k);
        let (mc, md) = feasible(soc);
        let mut best_value = layers[t + 1][k];
        let mut best_u = 0.0_f64;
        let mut best_k = k;
        for a in 0..levels {
            let frac = (a as f64) / ((levels - 1) as f64);
            for &u in &[-mc * frac, md * frac] {
                if u == 0.0 { continue; }
                let sn = step_soc(soc, u);
                let kn = (((sn - soc_min) / (soc_max - soc_min)) * ((levels - 1) as f64))
                    .round().clamp(0.0, (levels - 1) as f64) as usize;
                let cand = step_profit(u, price, battery.nominal_capacity) + layers[t + 1][kn];
                if cand > best_value { best_value = cand; best_u = u; best_k = kn; }
            }
        }
        out.push(best_u);
        k = best_k;
    }
    out
}

pub fn da_plan_realised_score(
    batteries: &[Battery],
    plan_prices: &[Vec<f64>],
    score_prices: &[Vec<f64>],
    levels: usize,
) -> f64 {
    let levels = levels.max(2);
    if plan_prices.is_empty() || score_prices.is_empty() {
        return 0.0;
    }
    batteries
        .iter()
        .map(|battery| single_battery_da_plan(battery, plan_prices, score_prices, levels))
        .sum()
}

fn single_battery_da_plan(
    battery: &Battery,
    plan_prices: &[Vec<f64>],
    score_prices: &[Vec<f64>],
    levels: usize,
) -> f64 {
    let steps = plan_prices.len().min(score_prices.len());
    let capacity = battery.capacity;
    if capacity <= 0.0 || steps == 0 {
        return 0.0;
    }
    let soc_min = capacity * battery.reserve_fraction.clamp(0.0, 1.0);
    let soc_max = capacity;
    if soc_max <= soc_min {
        return 0.0;
    }
    let grid = |k: usize| soc_min + (soc_max - soc_min) * (k as f64) / ((levels - 1) as f64);

    let feasible = |soc: f64| -> (f64, f64) {
        let headroom = (soc_max - soc).max(0.0);
        let available = (soc - soc_min).max(0.0);
        let mc = if battery.charge_efficiency > 0.0 {
            (headroom / (battery.charge_efficiency * DELTA_T)).min(battery.max_charge)
        } else {
            0.0
        }
        .max(0.0);
        let md = if battery.discharge_efficiency > 0.0 {
            (available * battery.discharge_efficiency / DELTA_T).min(battery.max_discharge)
        } else {
            0.0
        }
        .max(0.0);
        (mc, md)
    };
    let step_soc = |soc: f64, u: f64| -> f64 {
        let delta = if u > 0.0 {
            -(u * DELTA_T) / battery.discharge_efficiency
        } else {
            -u * battery.charge_efficiency * DELTA_T
        };
        (soc + delta).clamp(soc_min, soc_max)
    };

    // Backward pass on the PLAN prices, keeping every layer so the forward pass
    // can replay the planner's own choices.
    let mut layers: Vec<Vec<f64>> = vec![vec![0.0_f64; levels]; steps + 1];
    for t in (0..steps).rev() {
        let price = plan_prices[t].get(battery.node).copied().unwrap_or(0.0);
        for k in 0..levels {
            let soc = grid(k);
            let (max_charge, max_discharge) = feasible(soc);
            let mut best = layers[t + 1][k];
            for a in 0..levels {
                let frac = (a as f64) / ((levels - 1) as f64);
                for &u in &[-max_charge * frac, max_discharge * frac] {
                    if u == 0.0 {
                        continue;
                    }
                    let soc_next = step_soc(soc, u);
                    let kn = (((soc_next - soc_min) / (soc_max - soc_min))
                        * ((levels - 1) as f64))
                        .round()
                        .clamp(0.0, (levels - 1) as f64) as usize;
                    let candidate =
                        step_profit(u, price, battery.nominal_capacity) + layers[t + 1][kn];
                    if candidate > best {
                        best = candidate;
                    }
                }
            }
            layers[t][k] = best;
        }
    }

    // Forward pass: choose by the PLAN, bank the profit at the REALISED price.
    let mut k = ((0.5) * ((levels - 1) as f64)).round() as usize;
    k = k.min(levels - 1);
    let mut earned = 0.0_f64;
    for t in 0..steps {
        let plan_price = plan_prices[t].get(battery.node).copied().unwrap_or(0.0);
        let real_price = score_prices[t].get(battery.node).copied().unwrap_or(0.0);
        let soc = grid(k);
        let (max_charge, max_discharge) = feasible(soc);
        let mut best_value = layers[t + 1][k];
        let mut best_u = 0.0_f64;
        let mut best_k = k;
        for a in 0..levels {
            let frac = (a as f64) / ((levels - 1) as f64);
            for &u in &[-max_charge * frac, max_discharge * frac] {
                if u == 0.0 {
                    continue;
                }
                let soc_next = step_soc(soc, u);
                let kn = (((soc_next - soc_min) / (soc_max - soc_min)) * ((levels - 1) as f64))
                    .round()
                    .clamp(0.0, (levels - 1) as f64) as usize;
                let candidate =
                    step_profit(u, plan_price, battery.nominal_capacity) + layers[t + 1][kn];
                if candidate > best_value {
                    best_value = candidate;
                    best_u = u;
                    best_k = kn;
                }
            }
        }
        earned += step_profit(best_u, real_price, battery.nominal_capacity);
        k = best_k;
    }
    earned
}

fn single_battery_clairvoyant(
    battery: &Battery,
    realised_prices: &[Vec<f64>],
    levels: usize,
) -> f64 {
    let steps = realised_prices.len();
    let capacity = battery.capacity;
    if capacity <= 0.0 || steps == 0 {
        return 0.0;
    }
    let soc_min = capacity * battery.reserve_fraction.clamp(0.0, 1.0);
    let soc_max = capacity;
    if soc_max <= soc_min {
        return 0.0;
    }
    let grid = |k: usize| soc_min + (soc_max - soc_min) * (k as f64) / ((levels - 1) as f64);

    // value[k] = best profit obtainable from `t` onward starting at SoC level k.
    let mut value = vec![0.0_f64; levels];
    let mut next = vec![0.0_f64; levels];

    for t in (0..steps).rev() {
        let price = realised_prices[t]
            .get(battery.node)
            .copied()
            .unwrap_or(0.0);
        for k in 0..levels {
            let soc = grid(k);
            let headroom = (soc_max - soc).max(0.0);
            let available = (soc - soc_min).max(0.0);
            let max_charge = if battery.charge_efficiency > 0.0 {
                (headroom / (battery.charge_efficiency * DELTA_T)).min(battery.max_charge)
            } else {
                0.0
            }
            .max(0.0);
            let max_discharge = if battery.discharge_efficiency > 0.0 {
                (available * battery.discharge_efficiency / DELTA_T).min(battery.max_discharge)
            } else {
                0.0
            }
            .max(0.0);

            // Search the action grid; idle is always admissible.
            let mut best = next[k];
            for a in 0..levels {
                let frac = (a as f64) / ((levels - 1) as f64);
                for &u in &[-max_charge * frac, max_discharge * frac] {
                    if u == 0.0 {
                        continue;
                    }
                    let delta = if u > 0.0 {
                        -(u * DELTA_T) / battery.discharge_efficiency
                    } else {
                        -u * battery.charge_efficiency * DELTA_T
                    };
                    let soc_next = (soc + delta).clamp(soc_min, soc_max);
                    let kn = (((soc_next - soc_min) / (soc_max - soc_min))
                        * ((levels - 1) as f64))
                        .round()
                        .clamp(0.0, (levels - 1) as f64) as usize;
                    let candidate = step_profit(u, price, battery.nominal_capacity) + next[kn];
                    if candidate > best {
                        best = candidate;
                    }
                }
            }
            value[k] = best;
        }
        // Manual swap — TIG submit bans `/*swapped*/` imports.
        let tmp = value;
        value = next;
        next = tmp;
    }

    // Start at mid-range. NOTE: the pinned challenge carries an explicit
    // `soc_initial_mwh` that our dependency-free `Battery` does not model, so
    // this is an assumption, not a reading. It matters: selling the initial
    // charge is real revenue on any path (a flat-price probe returned 182.9,
    // not 0, for exactly this reason). If the real initial SoC is higher this
    // under-states the bound and remains a bound; if lower, the bound is
    // optimistic by that margin. Carry `soc_initial` through before quoting
    // absolute capture fractions.
    let start = ((0.5) * ((levels - 1) as f64)).round() as usize;
    next[start.min(levels - 1)]
}

