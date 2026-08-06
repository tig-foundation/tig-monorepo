mod kernel;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::{Challenge, Solution, State};

/// Submit package: process env knobs are compile-frozen (public surface defaults only).
#[inline(always)]
fn gnosis_env_var(_key: &str) -> Result<String, ()> {
    Err(())
}
#[inline(always)]
fn gnosis_fs_read_to_string(_path: &str) -> Result<String, ()> {
    Err(())
}
#[inline(always)]
fn gnosis_fs_write(_path: &str, _body: impl AsRef<[u8]>) -> Result<(), ()> {
    Err(())
}


#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub residual_weight: f64,
    pub deadband: f64,
    pub max_backoffs: usize,
    pub lookahead_horizon: usize,
    pub congestion_weight: f64,
    pub friction_weight: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            // Higher residual: DP queries under RT (prometheus); DA still owns continuation.
            residual_weight: 0.45,
            deadband: 0.02,
            max_backoffs: 80,
            lookahead_horizon: 24,
            congestion_weight: 0.20,
            friction_weight: 1.0,
        }
    }
}

impl Hyperparameters {
    /// Track-shaped defaults (titan_v6 per-track idea) when user HP omit fields.
    fn for_challenge(challenge: &Challenge) -> Self {
        let (residual_weight, deadband, max_backoffs, congestion_weight, friction_weight) =
            kernel::track_params(challenge.num_batteries, challenge.num_steps);
        Self {
            residual_weight,
            deadband,
            max_backoffs,
            // TRACK-SCOPED. Lookahead is worth paying for when the fuel
            // budget allows it and is pure waste when it does not.
            //
            // Measured on the metered path (Cloud Build e55597d3, --fuel
            // 100e9): baseline burns ~10.0e9 and congested ~21e9, while
            // multiday, dense and capstone hit exit 87 on EVERY nonce and
            // return no solution at all.
            //
            // Measured against the deterministic work meter, dropping the
            // horizon 24 -> 6 removes 67-79% of the work on exactly those
            // three tracks while quality holds or improves:
            //
            //   multiday  189455 -> 62664 units,  q 1.1367 -> 1.1843
            //   dense     277904 -> 87560 units,  q 2.6073 -> 2.6843
            //   capstone  448100 -> 129605 units, q 2.8134 -> 2.8190
            //
            // But the SMALL tracks pay for it: baseline q 0.0472 -> 0.0451
            // (-4.4%) and congested 0.1637 -> 0.1088 (-33.5%). They have fuel
            // headroom and the lookahead earns its cost there, so they keep 24.
            //
            // Scoped by battery count, matching `kernel::track_params`.
            // GNOSIS_BISECT_LOOKAHEAD applies to ALL tracks, so the effect can
            // be measured on baseline/congested -- which complete and report
            // fuel. The earlier lookahead change touched only >30-battery
            // tracks, i.e. exactly the ones that exit 87 and report NO fuel, so
            // "lookahead moved ~0% fuel" was never actually measured. Same
            // positive-control error as the first backoff bisect.
            lookahead_horizon: gnosis_env_var("GNOSIS_BISECT_LOOKAHEAD")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(if challenge.num_batteries > 30 { 6 } else { 24 })
                .max(1)
                .min(challenge.num_steps.max(1)),
            congestion_weight,
            friction_weight,
        }
    }

    fn merge_user(mut self, raw: &Option<Map<String, Value>>) -> Result<Self> {
        let Some(map) = raw else {
            return Ok(self);
        };
        if let Some(v) = map.get("residual_weight").and_then(|v| v.as_f64()) {
            self.residual_weight = v;
        }
        if let Some(v) = map.get("deadband").and_then(|v| v.as_f64()) {
            self.deadband = v;
        }
        if let Some(v) = map.get("max_backoffs").and_then(|v| v.as_u64()) {
            self.max_backoffs = v as usize;
        }
        if let Some(v) = map.get("lookahead_horizon").and_then(|v| v.as_u64()) {
            self.lookahead_horizon = (v as usize).max(1);
        }
        if let Some(v) = map.get("congestion_weight").and_then(|v| v.as_f64()) {
            self.congestion_weight = v;
        }
        if let Some(v) = map.get("friction_weight").and_then(|v| v.as_f64()) {
            self.friction_weight = v;
        }
        Ok(self)
    }
}

pub fn help() {
    println!("Multi-policy portfolio: scarcity + quantile + residual seeds,");
    println!("congestion duals, post-feasibility expand, DELTA_T-correct SOC.");
}

/// Battery parameters only — all the clairvoyant bound needs.
fn local_batteries(challenge: &Challenge) -> Vec<kernel::Battery> {
    challenge
        .batteries
        .iter()
        .map(|battery| kernel::Battery {
            node: battery.node,
            capacity: battery.soc_max_mwh,
            nominal_capacity: battery.capacity_mwh,
            max_charge: battery.power_charge_mw,
            max_discharge: battery.power_discharge_mw,
            charge_efficiency: battery.efficiency_charge,
            discharge_efficiency: battery.efficiency_discharge,
            reserve_fraction: battery.soc_min_mwh / battery.soc_max_mwh.max(1e-12),
        })
        .collect()
}

/// Capture-fraction diagnostic for the online lane.
///
/// Energy gives the policy no future information — prices come from a hidden
/// seed and `policy()` sees only `rt_prices` for the current step. So "how far
/// from optimal" is not askable the way it is for the offline lanes. What IS
/// askable after the fact: **of the profit available on the path that actually
/// occurred, what fraction did the policy capture?** A quality of -25.9% vs
/// live #1 cannot separate "our policy is weak" from "these paths were largely
/// unpredictable"; a capture fraction can.
///
/// Both numbers must come from ONE pass: `grid_optimize` is callable only once
/// per process (it flips a global). So the policy closure records each step's
/// realised prices as it runs, and our profit is then recomputed from the
/// returned schedule against those same prices using `kernel::step_profit` —
/// the identical function the bound uses, so the comparison is exact rather
/// than approximate.
///
/// Also returns TIG's own `baseline_total_profit`, because the published
/// quality formula is an EXCESS ratio over it:
///
/// ```text
/// quality = (total_profit - baseline_total_profit) / baseline_total_profit
/// ```
///
/// so the scoreboard delta against live #1 is
/// `(x_ours - x_live) / (x_live - B)`. The baseline does NOT cancel: the metric
/// divides the profit difference by live's MARGIN over the baseline. A fixed
/// absolute improvement therefore pays far more score on a track (or an
/// instance) where that margin is small. Without B the scoreboard cannot be
/// interpreted at all, and ranking tracks by it is ranking by gap/margin rather
/// than by gap.
///
/// Returns `(our_profit, clairvoyant_bound, steps)`.
///
/// **Does NOT return TIG's baseline.** `compute_baseline` is wrapped in
/// `conditional_pub!` and is PRIVATE under the `hide_verification` feature —
/// which is exactly what `tig-binary` builds the submittable shared object
/// with (`tig-algorithms/Cargo.toml:22`). Calling it here compiled fine in the
/// replay harness (whose own `tig-challenges` has the feature off) and broke
/// the fuel-metered SO build with E0624, invisibly, for a whole session.
///
/// Verification-dependent quantities belong in the HARNESS, not in the
/// submittable algorithm. The replay tool calls `compute_baseline` itself and
/// combines the results.
fn score_prices_at(realised: &[Vec<f64>], t: usize, node: usize) -> f64 {
    realised.get(t).and_then(|r| r.get(node)).copied().unwrap_or(0.0)
}

pub fn capture_diagnostics(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<(f64, f64, usize)> {
    use std::cell::RefCell;
    let parameters = Hyperparameters::for_challenge(challenge).merge_user(hyperparameters)?;
    let realised: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::with_capacity(challenge.num_steps));

    let mut local = build_local(challenge, &parameters);

    // FORESIGHT DECOMPOSITION (diagnostic only -- never reached by
    // `solve_challenge`, which is the submitted entry point; both knobs are
    // env-gated and live here, in the harness-facing function, so no filesystem
    // access exists on the production path).
    //
    // The clairvoyant bound U is computed from the REALISED prices, so it
    // assumes foresight the challenge structurally withholds. That makes it a
    // valid upper bound and a loose one, and the amount it misses by has never
    // been split into its two causes:
    //
    //   B - A  = value of INFORMATION   -- unreachable by any online policy
    //   U - B  = our OPTIMISATION SLACK -- reachable, given the same information
    //
    // where A is this policy online and B is the SAME policy handed the realised
    // prices as its day-ahead forecast. If B lands near U the optimiser is sound
    // and the remaining gap is information, i.e. structural void; if B lands far
    // below U there is genuine reachable room.
    //
    // grid_optimize is callable once per process, so realised prices cannot be
    // known in the same pass that would consume them. Hence two passes:
    // GNOSIS_DUMP_REALIZED writes them, GNOSIS_FORESIGHT_PRICES injects them.
    if let Ok(path) = gnosis_env_var("GNOSIS_FORESIGHT_PRICES") {
        if let Ok(text) = gnosis_fs_read_to_string(&path) {
            let injected: Vec<Vec<f64>> = text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    line.split(',')
                        .filter_map(|value| value.trim().parse::<f64>().ok())
                        .collect()
                })
                .collect();
            // Shape must match exactly, or the run silently measures something
            // else -- the failure mode this programme keeps paying for.
            let rows_ok = injected.len() == local.day_ahead_prices.len();
            let cols_ok = injected
                .iter()
                .zip(&local.day_ahead_prices)
                .all(|(a, b)| a.len() == b.len());
            if rows_ok && cols_ok {
                // GRADED, not binary. A two-point contrast (pure forecast vs
                // pure realised) cannot separate "the policy cannot use
                // information" from "the policy cannot use a VOLATILE input",
                // because it moves both at once. Blending k from 0 to 1 varies
                // information content continuously; a monotone decay implicates
                // structural dependence on a smooth forecast, a cliff at k=1
                // implicates something else.
                let k = gnosis_env_var("GNOSIS_FORESIGHT_BLEND")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                for (row, injected_row) in
                    local.day_ahead_prices.iter_mut().zip(injected.iter())
                {
                    for (da, &real) in row.iter_mut().zip(injected_row.iter()) {
                        *da = (1.0 - k) * *da + k * real;
                    }
                }
                eprintln!("FORESIGHT active: blend k={k}");
            } else {
                eprintln!(
                    "FORESIGHT REJECTED: shape mismatch ({} rows vs {})",
                    injected.len(),
                    local.day_ahead_prices.len()
                );
            }
        } else {
            eprintln!("FORESIGHT REJECTED: cannot read {}", path);
        }
    }

    // Base line flows per step: the network loading BEFORE any battery acts.
    // The first overload probe omitted these and therefore measured only the
    // battery-induced component against the limits. Collected here so the
    // projected reference below uses the flow the network actually sees.
    let base_flows: RefCell<Vec<Vec<f64>>> = RefCell::new(Vec::with_capacity(challenge.num_steps));
    let solution = challenge.grid_optimize(&|challenge, state| {
        realised.borrow_mut().push(state.rt_prices.clone());
        let zero = vec![0.0; challenge.num_batteries];
        let inj = challenge.compute_total_injections(state, &zero);
        base_flows.borrow_mut().push(challenge.network.compute_flows(&inj));
        policy(challenge, state, &parameters, &local)
    })?;
    let base_flows = base_flows.into_inner();

    let realised = realised.into_inner();
    if let Ok(path) = gnosis_env_var("GNOSIS_DUMP_REALIZED") {
        let body: String = realised
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let _ = gnosis_fs_write(&path, body);
    }
    let batteries = local_batteries(challenge);

    // Our profit, scored with the same per-step function as the bound, so the
    // two are exactly comparable rather than approximately so.
    let mut ours = 0.0_f64;
    for (t, action) in solution.schedule.iter().enumerate() {
        let Some(prices) = realised.get(t) else { break };
        for (battery, &u) in batteries.iter().zip(action.iter()) {
            let price = prices.get(battery.node).copied().unwrap_or(0.0);
            ours += kernel::step_profit(u, price, battery.nominal_capacity);
        }
    }

    let bound = kernel::clairvoyant_profit_bound_for(&batteries, &realised, 33);
    // Achievable reference: plan on the day-ahead matrix (known at t=0), score
    // on the realised path. Diagnostic only; env-gated so the normal diag output
    // is unchanged.
    if gnosis_env_var("GNOSIS_DA_PLAN_REF").is_ok() {
        let da: Vec<Vec<f64>> = challenge.market.day_ahead_prices.clone();
        let ref_profit = kernel::da_plan_realised_score(&batteries, &da, &realised, 33);
        eprintln!("DAPLANREF {ref_profit:?}");

        // FEASIBILITY OF THE REFERENCE. It plans without line limits or PTDF, so
        // its schedule may be unrunnable. Measure by how much: for each step,
        // the largest |flow| / limit across all lines. A median well above 1.0
        // means the reference is an artefact of ignoring the network rather than
        // evidence of reachable headroom.
        let schedule = kernel::da_plan_schedule(&batteries, &da, 33);
        let mut overloads: Vec<f64> = Vec::with_capacity(schedule.len());
        for (t, actions) in schedule.iter().enumerate() {
            let mut nodal = vec![0.0_f64; challenge.network.num_nodes];
            for (battery, &u) in challenge.batteries.iter().zip(actions.iter()) {
                if battery.node < nodal.len() {
                    nodal[battery.node] += u;
                }
            }
            let flows = challenge.network.compute_flows(&nodal);
            let worst = flows
                .iter()
                .enumerate()
                .map(|(l, f)| {
                    let limit = challenge.network.flow_limits[l].max(1e-9);
                    f.abs() / limit
                })
                .fold(0.0_f64, f64::max);
            if t < schedule.len() {
                overloads.push(worst);
            }
        }
        // PROJECTED REFERENCE. Scale each step's planned actions by the largest
        // s in [0,1] that keeps base + battery flows inside every limit, then
        // score at the realised price. This is the reference the unprojected one
        // should have been: it respects the network, so its profit is genuinely
        // achievable rather than an artefact of ignoring congestion.
        let mut feasible_profit = 0.0_f64;
        for (t, actions) in schedule.iter().enumerate() {
            let Some(base) = base_flows.get(t) else { break };
            let mut nodal = vec![0.0_f64; challenge.network.num_nodes];
            for (battery, &u) in challenge.batteries.iter().zip(actions.iter()) {
                if battery.node < nodal.len() {
                    nodal[battery.node] += u;
                }
            }
            let delta = challenge.network.compute_flows(&nodal);
            let mut lo = 0.0_f64;
            let mut hi = 1.0_f64;
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                let ok = delta.iter().enumerate().all(|(l, d)| {
                    (base[l] + mid * d).abs() <= challenge.network.flow_limits[l] + 1e-9
                });
                if ok { lo = mid } else { hi = mid }
            }
            for (battery, &u) in batteries.iter().zip(actions.iter()) {
                feasible_profit += kernel::step_profit(
                    u * lo,
                    score_prices_at(&realised, t, battery.node),
                    battery.nominal_capacity,
                );
            }
        }
        eprintln!("DAPLANFEASIBLE {feasible_profit:?}");

        overloads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = overloads[overloads.len() / 2];
        let p90 = overloads[(overloads.len() * 9) / 10];
        let mx = overloads[overloads.len() - 1];
        let over = overloads.iter().filter(|x| **x > 1.0).count();
        eprintln!(
            "DAPLANFEAS median={med:.3} p90={p90:.3} max={mx:.3} steps_over_limit={over}/{}",
            overloads.len()
        );
    }
    Ok((ours, bound, realised.len()))
}

pub fn work_units() -> u64 {
    kernel::work_units()
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    kernel::reset_work_meter();
    let parameters = Hyperparameters::for_challenge(challenge).merge_user(hyperparameters)?;
    // Built ONCE, not once per step. See `build_local`.
    let local = build_local(challenge, &parameters);
    let solution = challenge.grid_optimize(&|challenge, state| {
        policy(challenge, state, &parameters, &local)
    })?;
    save_solution(&solution)
}

/// Build the immutable half of the local challenge ONCE.
///
/// This was previously rebuilt inside `policy`, i.e. once per time step: the
/// day-ahead price vector cloned, the battery and line vectors reallocated, and
/// `network.ptdf` — an L x N matrix — **fully cloned**, 192 times per solve.
/// None of it varies with the step; only `OnlineState` does.
///
/// That is why no parameter reached the fuel cost. Measured on the metered
/// path, cutting `max_backoffs` 96 -> 4 (a 24x reduction) moved fuel by
/// 1.7-5.3%, and `lookahead_horizon` 24 -> 6 moved it by ~nothing: both tune
/// terms that are not dominant. The dominant term was per-step reconstruction
/// of an immutable structure.
fn build_local(challenge: &Challenge, parameters: &Hyperparameters) -> kernel::EnergyChallenge {
    let mut local = kernel::EnergyChallenge {
        dp_cache: Vec::new(),
        day_ahead_prices: challenge.market.day_ahead_prices.clone(),
        batteries: challenge
            .batteries
            .iter()
            .map(|battery| kernel::Battery {
                node: battery.node,
                capacity: battery.soc_max_mwh,
                nominal_capacity: battery.capacity_mwh,
                max_charge: battery.power_charge_mw,
                max_discharge: battery.power_discharge_mw,
                charge_efficiency: battery.efficiency_charge,
                discharge_efficiency: battery.efficiency_discharge,
                reserve_fraction: battery.soc_min_mwh / battery.soc_max_mwh.max(1e-12),
            })
            .collect(),
        lines: challenge
            .network
            .flow_limits
            .iter()
            .map(|&limit| kernel::Line { limit })
            .collect(),
        ptdf: challenge.network.ptdf.clone(),
        residual_weight: parameters.residual_weight,
        deadband: parameters.deadband,
        max_backoffs: parameters.max_backoffs,
        lookahead_horizon: parameters.lookahead_horizon,
        congestion_weight: parameters.congestion_weight,
        friction_weight: parameters.friction_weight,
        delta_cong: Vec::new(),
        soc_ref: Vec::new(),
        soc_ref_lambda: 0.0,
        soc_ref_dynamic: false,
    };
    // Built ONCE per solve. The per-step rebuild was 94.7-98.3% of all fuel
    // BEFORE the hoist; that figure does not transfer to the hoisted code.
    //
    // Measured: with the DP cache in place, GNOSIS_SKIP_DP=1 removes only 6.9%
    // of baseline fuel while GNOSIS_DP_LEVELS=25 removes 27.4%. A knob cannot
    // save more than deleting what it configures, so the two were reaching
    // different code: SKIP_DP guarded only the per-step lookup in the policy
    // (src/energy_arbitrage.rs:241) and left THIS construction running. Post
    // hoist the DP cost is dominated by building the cache, not querying it.
    //
    // Guarded here so SKIP_DP means what it says and the two knobs decompose:
    // SKIP_DP = the whole DP, DP_LEVELS = its resolution.
    let skip_dp = gnosis_env_var("GNOSIS_SKIP_DP")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false);
    if !skip_dp {
        // LMP anticipation (titan_v6 t52/t53): PTDF-weighted congestion premiums
        // on DP prices only. DEFAULT OFF — paired n=8 (seed gnosis_replay_v1):
        //
        //   dense     mean Δq = −1.86%  t=−0.91  wins 2/6  (NaN nonces excl.)
        //   capstone  mean Δq = −1.43%  t=−0.75  wins 3/8
        //   multiday  mean Δq = +1.10%  t=+0.95  wins 3/8
        //
        // All |t|<1; sub-5% and inside the lane's chaotic noise floor
        // (HANDOFF_C008 §4: 1.25e-5 ranking perturbation → ±4%). Kept as an
        // opt-in instrument (`GNOSIS_ANTICIPATE_LMP=1`) because ptdf_ct /
        // composite_wv ports need the same premium builder, not because the
        // exo-only premium alone closes the gap.
        let premiums = lmp_premiums_for(challenge);
        let mut dps = match premiums.as_ref() {
            Some(p) => kernel::build_dp_cache_with_lmp(&local, Some(p.as_slice())),
            None => kernel::build_dp_cache(&local),
        };
        // OCO constraint tracking (titan use_ptdf_ct). S5c: default ON dense
        // only (`50 < n ≤ 80`); n=16 +1.74% under S5b stack.
        if ptdf_ct_enabled(challenge.batteries.len()) {
            let exo_flows = exo_line_flows(challenge);
            let battery_nodes: Vec<usize> = challenge.batteries.iter().map(|b| b.node).collect();
            let initial_soc: Vec<f64> = challenge
                .batteries
                .iter()
                .map(|b| b.soc_initial_mwh)
                .collect();
            let (eta, ct_scale) = ptdf_ct_params(challenge.num_batteries);
            dps = kernel::apply_ptdf_constraint_tracking(
                dps,
                &kernel::PtdfCtInput {
                    batteries: &local.batteries,
                    day_ahead_prices: &local.day_ahead_prices,
                    ptdf: &local.ptdf,
                    line_limits: &challenge.network.flow_limits,
                    exo_flows: &exo_flows,
                    battery_nodes: &battery_nodes,
                    slack_bus: challenge.network.slack_bus,
                    initial_soc: &initial_soc,
                    base_premiums: premiums.as_deref(),
                    eta,
                    ct_scale,
                },
            );
        }
        local.dp_cache = dps;
        // titan composite_wv (fleet k=1): dual aggregate DP → delta_cong series
        // for the PGA gradient. Only meaningful with GNOSIS_PGA=1. Default OFF.
        if composite_wv_enabled() {
            let lambda = gnosis_env_var("GNOSIS_CWV_LAMBDA")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.25);
            // Prefer LMP premiums already computed for the DP. If ANTICIPATE_LMP
            // is off, build exo premiums just for the dual DP (titan enables
            // LMP together with CWV; without a premium difference the series is 0).
            let prem_owned = if premiums.is_some() {
                None
            } else {
                let exo = exo_line_flows(challenge);
                let nodes: Vec<usize> = challenge.batteries.iter().map(|b| b.node).collect();
                Some(kernel::expected_lmp_premiums(
                    &exo,
                    &challenge.network.flow_limits,
                    &challenge.network.ptdf,
                    &nodes,
                    challenge.network.slack_bus,
                    0.5,
                    2.0,
                ))
            };
            let prem_ref = premiums.as_deref().or(prem_owned.as_deref());
            local.delta_cong = kernel::build_fleet_delta_cong(
                &local.batteries,
                &local.day_ahead_prices,
                prem_ref,
                lambda,
                33,
            );
        }
        // titan t51 L8 SoC-reference tracking into the PGA gradient.
        // Default OFF. Enable with GNOSIS_SOC_REF_LAMBDA=0.05 (titan multiday
        // baked default) or GNOSIS_SOC_REF=1 (uses λ=0.05).
        // Static: DA P25/P75 from initial SoC (once). Dynamic (L8b): also set
        // GNOSIS_SOC_REF_DYN=1 to recompute each step from residual shift.
        if let Some(lam) = soc_ref_lambda_from_env() {
            let initial_soc: Vec<f64> = challenge
                .batteries
                .iter()
                .map(|b| b.soc_initial_mwh)
                .collect();
            local.soc_ref = kernel::compute_soc_reference_static(
                &local.batteries,
                &local.day_ahead_prices,
                &initial_soc,
            );
            local.soc_ref_lambda = lam;
            local.soc_ref_dynamic = soc_ref_dynamic_from_env();
        }
    }
    local
}

/// `GNOSIS_SOC_REF_LAMBDA=<f64>` if >0, else `GNOSIS_SOC_REF=1` → 0.05, else None.
fn soc_ref_lambda_from_env() -> Option<f64> {
    if let Some(v) = gnosis_env_var("GNOSIS_SOC_REF_LAMBDA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        if v > 0.0 {
            return Some(v);
        }
        return None;
    }
    match gnosis_env_var("GNOSIS_SOC_REF")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("1") | Some("true") | Some("on") => Some(0.05),
        _ => None,
    }
}

/// `GNOSIS_SOC_REF_DYN=1` enables per-step residual-shift SoC-ref recompute.
fn soc_ref_dynamic_from_env() -> bool {
    matches!(
        gnosis_env_var("GNOSIS_SOC_REF_DYN")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

fn composite_wv_enabled() -> bool {
    matches!(
        gnosis_env_var("GNOSIS_COMPOSITE_WV")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

/// S5c/d: PTDF_CT under current stack.
/// - dense n=16 **+1.74%** (10/16)
/// - multiday n=8 **+3.72%** (6/8); n=32 **+2.22%** (23/32) → **−6.4% vs live**
/// Historical null was pre-S5b. Default ON for multiday+dense (`30 < n ≤ 80`);
/// capstone OFF. Force: `GNOSIS_PTDF_CT=0|1`.
fn ptdf_ct_enabled(num_batteries: usize) -> bool {
    match gnosis_env_var("GNOSIS_PTDF_CT")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("1") | Some("true") | Some("on") => true,
        Some("0") | Some("false") | Some("off") => false,
        _ => num_batteries > 30 && num_batteries <= 80,
    }
}

/// titan dense: ct_step_eta=1.0; capstone: 0.5; multiday not baked with CT.
fn ptdf_ct_params(num_batteries: usize) -> (f64, f64) {
    let eta = gnosis_env_var("GNOSIS_CT_ETA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(if num_batteries > 80 { 0.5 } else { 1.0 })
        .max(0.0);
    let kappa = gnosis_env_var("GNOSIS_CT_KAPPA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    (eta, 1.0 - kappa)
}

fn exo_line_flows(challenge: &Challenge) -> Vec<Vec<f64>> {
    (0..challenge.num_steps)
        .map(|t| {
            let inj = challenge
                .exogenous_injections
                .get(t)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if inj.len() == challenge.network.num_nodes {
                challenge.network.compute_flows(inj)
            } else {
                let mut full = vec![0.0_f64; challenge.network.num_nodes];
                for (i, &v) in inj.iter().enumerate().take(full.len()) {
                    full[i] = v;
                }
                challenge.network.compute_flows(&full)
            }
        })
        .collect()
}

/// Titan_v6 congestion anticipation for the DP. Returns `None` when disabled
/// so the pure-DA cache path is taken (and is byte-identical to pre-LMP).
///
/// Baked titan_v6 defaults for dense/capstone: threshold 0.5, scale 2.0.
/// Measured alone: null/regress on dense+capstone (see `build_local`). Opt-in
/// only via `GNOSIS_ANTICIPATE_LMP=1`.
fn lmp_premiums_for(challenge: &Challenge) -> Option<Vec<Vec<f64>>> {
    let force = gnosis_env_var("GNOSIS_ANTICIPATE_LMP").ok();
    let enabled = match force.as_deref().map(str::trim) {
        Some("1") | Some("true") | Some("on") => true,
        // Empty / unset / anything else: OFF (shipped default is pure-DA DP).
        _ => false,
    };
    if !enabled || challenge.network.num_lines == 0 || challenge.num_batteries == 0 {
        return None;
    }

    let threshold = gnosis_env_var("GNOSIS_LMP_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.5)
        .clamp(0.0, 0.99);
    let scale = gnosis_env_var("GNOSIS_LMP_SCALE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(2.0)
        .max(0.0);

    let exo_flows = exo_line_flows(challenge);
    let battery_nodes: Vec<usize> = challenge.batteries.iter().map(|b| b.node).collect();
    Some(kernel::expected_lmp_premiums(
        &exo_flows,
        &challenge.network.flow_limits,
        &challenge.network.ptdf,
        &battery_nodes,
        challenge.network.slack_bus,
        threshold,
        scale,
    ))
}

fn policy(
    challenge: &Challenge,
    state: &State,
    parameters: &Hyperparameters,
    local: &kernel::EnergyChallenge,
) -> Result<Vec<f64>> {
    // NULL-POLICY FLOOR CONTROL.
    //
    // Everything measured so far accounts for ~6% of fuel: lookahead 0.29%,
    // backoffs 1.7-5.3%, the per-step challenge hoist 0.003%, the sparse
    // action_is_valid 0.07-0.17%, incremental flows -0.004%. Baseline burns
    // ~104M instructions PER STEP on a 20-node, 30-line network, which is
    // absurd for the problem size.
    //
    // So the hypothesis this tests is that the fuel is not ours at all: TIG's
    // `simulate`/`grid_optimize` harness runs per step regardless of what the
    // policy does. Returning immediately measures that FLOOR.
    //
    //   fuel stays ~10e9  -> the floor is TIG's harness; NO policy optimisation
    //                        can help, and exit 87 on the large tracks is not
    //                        addressable by us at all.
    //   fuel collapses    -> the cost is ours, and it is in kernel::policy
    //                        somewhere none of the five probes has reached.
    //
    // This produces a zero action and therefore garbage quality. That is fine:
    // the only output is fuel_consumed.
    // NOTE `.is_ok()` was WRONG here: docker `-e VAR=""` SETS the variable to an
    // empty string, so `.is_ok()` was true on every build that forwarded the
    // substitution -- silently running the null policy during the portfolio
    // bisect and making it measure nothing. Require a non-empty value.
    // The other knobs use `.parse()`, which fails on empty, and were unaffected.
    if gnosis_env_var("GNOSIS_NULL_POLICY")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
    {
        return Ok(vec![0.0; challenge.num_batteries]);
    }

    let zero = vec![0.0; challenge.num_batteries];
    let base_injections = challenge.compute_total_injections(state, &zero);
    let base_line_flows = challenge.network.compute_flows(&base_injections);
    let local_state = kernel::OnlineState {
        time: state.time_step,
        state_of_charge: state.socs.clone(),
        observed_prices: state.rt_prices.clone(),
        base_line_flows,
    };
    let mut action = match kernel::policy(local, &local_state) {
        kernel::PolicyOutcome::Action(action) => action,
        // Never fail the official rollout on a withhold: zero is always safe.
        kernel::PolicyOutcome::Withheld(_) => zero.clone(),
    };
    for (value, &(lower, upper)) in action.iter_mut().zip(&state.action_bounds) {
        *value = value.clamp(lower, upper);
    }
    // INCREMENTAL FEASIBILITY via PTDF linearity.
    //
    // This loop previously called TIG's `compute_flows` on every iteration.
    // That function is DENSE — `num_lines x num_nodes` multiply-adds
    // (`network.rs:287`) — while the battery action is nonzero at only a
    // handful of nodes. Fuel is metered in instructions, and measurement showed
    // it is essentially FIXED per step regardless of adaptive work (baseline's
    // two nonces differ 1.4x in adaptive operations and 0.13% in fuel), which
    // is why no parameter ever moved it.
    //
    // PTDF is linear, so with `base` = flows at zero battery action:
    //
    //     flows(base + a) = base + PTDF * a_nodal
    //
    // and each retry halves `a`, hence halves the delta. So compute the delta
    // ONCE over battery nodes only — `lines x touched` instead of
    // `lines x nodes` — and scale it, instead of recomputing a dense product
    // per iteration.
    //
    // Exactly equivalent: the omitted terms are `ptdf * 0.0`, and the slack bus
    // contribution is carried in `base` since the action sums are rebalanced
    // identically at every scale.
    // PROBE: how dense are the exogenous injections, and the PTDF itself?
    // If injections are sparse the base product sparsifies too -- and the base
    // product is now the only remaining candidate for the fixed per-step cost.
    if gnosis_env_var("GNOSIS_DENSITY_PROBE").is_ok() && state.time_step == 0 {
        let n = challenge.network.num_nodes;
        let nz = state
            .exogenous_injections
            .iter()
            .filter(|v| v.abs() > 1e-12)
            .count();
        let lines = challenge.network.flow_limits.len();
        let ptdf_nz: usize = challenge
            .network
            .ptdf
            .iter()
            .map(|row| row.iter().filter(|v| v.abs() > 1e-12).count())
            .sum();
        eprintln!(
            "DENSITY nodes={n} lines={lines} inj_nonzero={nz} ({:.1}%) ptdf_nonzero={ptdf_nz} ({:.1}%) dense_ops={}",
            100.0 * nz as f64 / n.max(1) as f64,
            100.0 * ptdf_nz as f64 / (lines * n).max(1) as f64,
            lines * n
        );
    }
    let base_injections_now = challenge.compute_total_injections(state, &zero);
    let base_flows = challenge.network.compute_flows(&base_injections_now);

    // Nodal delta from the battery action, and the nodes it touches.
    let mut delta_nodal = vec![0.0_f64; challenge.network.num_nodes];
    let mut touched: Vec<usize> = Vec::with_capacity(challenge.batteries.len());
    for (battery, value) in challenge.batteries.iter().zip(action.iter()) {
        if battery.node < delta_nodal.len() {
            delta_nodal[battery.node] += *value;
            touched.push(battery.node);
        }
    }
    touched.sort_unstable();
    touched.dedup();

    let delta_flows: Vec<f64> = (0..challenge.network.flow_limits.len())
        .map(|line| {
            touched
                .iter()
                .map(|&node| challenge.network.ptdf[line][node] * delta_nodal[node])
                .sum::<f64>()
        })
        .collect();

    let mut scale = 1.0_f64;
    for _ in 0..=parameters.max_backoffs {
        let ok = challenge
            .network
            .flow_limits
            .iter()
            .enumerate()
            .all(|(line, limit)| (base_flows[line] + scale * delta_flows[line]).abs() <= *limit);
        if ok {
            for value in action.iter_mut() {
                *value *= scale;
            }
            return Ok(action);
        }
        scale *= 0.5;
    }
    Ok(zero)
}
