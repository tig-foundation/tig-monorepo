// closed_form_arb v2 — TIG c008 energy_arbitrage code submission draft.
//
// v2 hypothesis: v1 closed-form action almost always clamped to bounds for
// typical price gaps (Ē² / (2κ_deg Δt) ~ 20000 dominates any λ). So v1 was
// effectively "conservative.rs with 0% threshold," which lost to the
// official 5% threshold by ~45% on Track 1.
//
// v2 keeps threshold-based action but improves three things:
//   1. Compare current price to next-step DA forecast (not current grid avg).
//   2. Add free congestion premium γ_price · 1^cong_{i,t+1} to forecast.
//   3. Use multi-step horizon mean as the reference (like greedy.rs but
//      per-node instead of node 0 only).

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub charge_threshold: Option<f64>,
    pub discharge_threshold: Option<f64>,
    pub horizon: Option<usize>,
    pub use_congestion_premium: Option<bool>,
    /// Exponential-decay weight on horizon mean. 0 = uniform (v2.9), >0 = nearer
    /// steps weighted more.
    pub weight_alpha: Option<f64>,
}

impl Default for Hyperparameters {
    // Tuned 2026-05-05 from a 4×4×3 sweep over 50 BASELINE-track seeds.
    // 0.90/1.03/12 dominates 0.95/1.05/12 (the conservative.rs inheritance):
    // +9% T1, +24/+34/+35% T3/T4/T5. Wider charge band + narrower discharge
    // band = profit floor protects, free congestion premium catches upside.
    fn default() -> Self {
        Self {
            // v2.3 thresholds. Coarse sweep landed on 0.90/1.03; fine sweep
            // (0.01 increments around the optimum) shifted to 0.85/1.02. The
            // shift adds +2-7% across all tracks. The asymmetric tightening
            // (wider charge band, narrower discharge band) reflects that with
            // an accurate per-node target we want to capture more "high"
            // signals than tolerate noisy "low" signals.
            // v2.6: per-track threshold defaults are picked in policy() based on
            // (num_steps, num_batteries). None here means "use the per-track default."
            charge_threshold: None,
            discharge_threshold: None,
            // None → policy() picks track-adaptive (12 for ≤96 steps, 96 otherwise)
            horizon: None,
            use_congestion_premium: Some(true),
            weight_alpha: Some(0.0),  // 0 = uniform (v2.9 default)
        }
    }
}

pub fn help() {
    println!("closed_form_arb v2 — c008 energy_arbitrage");
    println!();
    println!("Threshold action with horizon-aware forecasting + free congestion signal.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp = parse_hp(hyperparameters)?;
    let solution = challenge.grid_optimize(&|c, s| policy(c, s, &hp))?;
    save_solution(&solution)?;
    Ok(())
}

fn parse_hp(hp: &Option<Map<String, Value>>) -> Result<Hyperparameters> {
    let default_hp = Hyperparameters::default();
    let Some(map) = hp.as_ref() else {
        return Ok(default_hp);
    };
    Ok(Hyperparameters {
        charge_threshold: map.get("charge_threshold").and_then(|v| v.as_f64())
            .or(default_hp.charge_threshold),
        discharge_threshold: map.get("discharge_threshold").and_then(|v| v.as_f64())
            .or(default_hp.discharge_threshold),
        horizon: map.get("horizon").and_then(|v| v.as_u64()).map(|v| v as usize)
            .or(default_hp.horizon),
        use_congestion_premium: map.get("use_congestion_premium").and_then(|v| v.as_bool())
            .or(default_hp.use_congestion_premium),
        weight_alpha: map.get("weight_alpha").and_then(|v| v.as_f64())
            .or(default_hp.weight_alpha),
    })
}

pub fn policy(challenge: &Challenge, state: &State, hp: &Hyperparameters) -> Result<Vec<f64>> {
    // v2.7: per-track thresholds re-tuned under new horizons.
    // T1: 0.87/1.00 → 0.85/0.99 (+0.4%)
    // T2: 0.80/1.01 → 0.75/0.99 (+0.5%)
    // T3-T5: kept at 0.80/1.01 (sweep gains were within noise).
    let (default_ct, default_dt) = match (challenge.num_steps, challenge.num_batteries) {
        (96, b) if b <= 10 => (0.85, 0.99),  // T1 BASELINE
        (96, _)            => (0.75, 0.99),  // T2 CONGESTED
        _                  => (0.80, 1.01),  // T3-T5
    };
    let charge_threshold = hp.charge_threshold.unwrap_or(default_ct);
    let discharge_threshold = hp.discharge_threshold.unwrap_or(default_dt);
    // Track-adaptive horizon: short horizons (12) win on 96-step instances
    // (Tracks 1-2), long horizons (~half remaining) win on 192-step instances
    // (Tracks 3-5). Discovered via 6-point sweep over 5 tracks: h=96 lifts
    // T3/T4/T5 by 31/28/42% but regresses T1 by 11%.
    // Per-track horizon. Re-swept under v2.7 settings (per-track CT/DT,
    // adaptive floor) which shifted T1/T2 optima upward — once the floor
    // and threshold logic stopped clipping, longer lookahead pays.
    //   T1 (96 steps, 10 batt): h=36   (was 18)
    //   T2 (96 steps, 20 batt): h=72   (was 36)
    //   T3 (192 steps, 40 batt): h=80  (unchanged)
    //   T4 (192 steps, 60 batt): h=192 (was 160)
    //   T5 (192 steps, 100 batt): h=192 (was 160)
    // Detection via (num_steps, num_batteries) — both are observable.
    let h = hp.horizon.unwrap_or_else(|| {
        match (challenge.num_steps, challenge.num_batteries) {
            (96, b) if b <= 10 => 36,    // T1 BASELINE
            (96, _)              => 72,   // T2 CONGESTED
            (192, b) if b <= 40  => 80,   // T3 MULTIDAY
            (192, _)             => 192,  // T4/T5 DENSE/CAPSTONE
            (n, _)               => n,    // unknown track: full remaining
        }
    });
    let use_cong = hp.use_congestion_premium.unwrap_or(true);

    let n = challenge.network.num_nodes;
    let m = challenge.num_batteries;
    let t = state.time_step;
    let da = &challenge.market.day_ahead_prices;

    let start = (t + 1).min(da.len());
    let end = (t + 1 + h).min(da.len());
    let horizon_len = end.saturating_sub(start);
    let mut horizon_mean = vec![0.0_f64; n];
    // v17: distance-weighted mean. weight(k) = exp(-alpha * k). alpha=0 → uniform (v2.9).
    let weight_alpha = hp.weight_alpha.unwrap_or(0.0);
    if horizon_len > 0 {
        let mut weight_sum = 0.0_f64;
        for (k, tt) in (start..end).enumerate() {
            let w = if weight_alpha > 0.0 {
                (-weight_alpha * k as f64).exp()
            } else {
                1.0
            };
            weight_sum += w;
            for i in 0..n {
                horizon_mean[i] += w * da[tt][i];
            }
        }
        for v in horizon_mean.iter_mut() {
            *v /= weight_sum;
        }
    } else if t < da.len() {
        horizon_mean.copy_from_slice(&da[t]);
    }

    let cong_next = if use_cong {
        compute_next_step_congestion_indicator(challenge, state)
    } else {
        vec![false; n]
    };

    let mut action = vec![0.0_f64; m];
    let cur_da = if t < da.len() { &da[t] } else { &da[da.len() - 1] };

    for (b, battery) in challenge.batteries.iter().enumerate() {
        let i = battery.node;
        let cur = cur_da[i];
        let cong_premium = if cong_next[i] { constants::GAMMA_PRICE } else { 0.0 };
        let target = horizon_mean[i] + cong_premium;
        if target.abs() < 1e-9 {
            continue;
        }

        let (lo, hi) = state.action_bounds[b];
        let can_charge = lo < -1e-6;
        let can_discharge = hi > 1e-6;

        // v16: end-of-instance forced discharge. Last K steps, dump SOC if
        // current price is non-negative — leftover SOC after the horizon is
        // not monetized, so any positive-priced step is +EV vs holding.
        let steps_remaining = challenge.num_steps.saturating_sub(t + 1);
        // Per-track end-window from v18 sweep. Differences are tiny (<$50 on
        // $100k+ means) but consistent across seeds.
        // T1: 8 | T2: 2 | T3: 1 | T4: 5 | T5: 1
        let end_window: usize = match (challenge.num_steps, challenge.num_batteries) {
            (96, b) if b <= 10 => 8,     // T1 BASELINE
            (96, _)            => 2,     // T2 CONGESTED
            (192, b) if b <= 40 => 1,    // T3 MULTIDAY
            (192, b) if b <= 60 => 5,    // T4 DENSE
            (192, _)           => 1,     // T5 CAPSTONE
            _ => 5,
        };
        let in_end_window = steps_remaining < end_window;

        if in_end_window && can_discharge && cur > constants::KAPPA_TX {
            // force discharge as long as price exceeds transaction cost
            action[b] = hi;
        } else if cur < charge_threshold * target && can_charge {
            action[b] = lo;
        } else if cur > discharge_threshold * target && can_discharge {
            action[b] = hi;
        } else {
            action[b] = 0.0;
        }
    }

    let action = enforce_flow_feasibility(challenge, state, action)?;
    // v2.5: profit floor enforcement is helpful in high-vol regimes (Tracks 3-5)
    // where rare losses compound, but it clips ~32% of upside on calm tracks (T1/T2).
    // Track-adaptive: disable on T1/T2 (96 steps), enable on T3+.
    let use_floor = match challenge.num_steps {
        96 => false,        // T1, T2 — disable (huge upside)
        _ => true,          // T3+ — enable (rare losses compound in high-vol)
    };
    if use_floor {
        Ok(enforce_profit_floor(challenge, state, action))
    } else {
        Ok(action)
    }
}

fn compute_next_step_congestion_indicator(challenge: &Challenge, state: &State) -> Vec<bool> {
    let n = challenge.network.num_nodes;
    let m = challenge.num_batteries;
    let zero_action = vec![0.0; m];
    let injections = challenge.compute_total_injections(state, &zero_action);

    let mut indicator = vec![false; n];
    for l in 0..challenge.network.num_lines {
        let flow: f64 = (0..n).map(|k| challenge.network.ptdf[l][k] * injections[k]).sum();
        let limit = challenge.network.flow_limits[l];
        if flow.abs() >= constants::TAU_CONG * limit {
            let (from, to) = challenge.network.lines[l];
            indicator[from] = true;
            indicator[to] = true;
        }
    }
    indicator
}

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    (0..challenge.network.num_lines)
        .map(|l| {
            (0..challenge.network.num_nodes)
                .map(|k| challenge.network.ptdf[l][k] * injections[k])
                .sum::<f64>()
        })
        .collect()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > constants::EPS_FLOW * limit {
            let candidate = Violation { line: l, flow, amount: violation };
            match best {
                Some(current) if candidate.amount <= current.amount => {}
                _ => best = Some(candidate),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, &flows).is_none()
}

fn soften_most_violated_line(
    challenge: &Challenge,
    violation: Violation,
    action: &mut [f64],
) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }
    let mut worsening_indices = Vec::new();
    let mut worsening_strength = 0.0;
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let contribution = challenge.network.ptdf[line][battery.node] * action[i];
        let signed_contribution = signed_direction * contribution;
        if signed_contribution > EPS {
            worsening_strength += signed_contribution;
            worsening_indices.push(i);
        }
    }
    if worsening_indices.is_empty() || worsening_strength <= EPS {
        return false;
    }
    let keep = (1.0 - violation.amount / worsening_strength).clamp(0.0, 1.0);
    if (1.0 - keep).abs() <= EPS {
        return false;
    }
    for i in worsening_indices {
        action[i] *= keep;
    }
    true
}

fn enforce_profit_floor(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Vec<f64> {
    let mut profit = challenge.compute_profit(state, &action);
    if state.total_profit + profit >= 0.0 {
        return action;
    }
    while state.total_profit + profit < 0.0 {
        action = action
            .into_iter()
            .map(|u| if u.abs() < EPS { 0.0 } else { u * 0.95 })
            .collect();
        profit = challenge.compute_profit(state, &action);
    }
    action
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_most_violated_line(challenge, violation, &mut action) {
            break;
        }
    }
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }
    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!(
            "Baseline fallback failed: grid is infeasible even with zero battery actions"
        ));
    }
    let base = action;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| mid * u).collect();
        if is_flow_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(base.into_iter().map(|u| low * u).collect())
}
