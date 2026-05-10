// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

const LOOKAHEAD_STEPS: usize = 16;
const COARSE_STEP_MW: f64 = 5.0;
const MIN_ACTION_MW: f64 = 0.01;
const FLOW_HEADROOM: f64 = 0.99;
const EPS: f64 = 1e-12;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

#[derive(Clone, Copy)]
struct Candidate {
    battery_idx: usize,
    action: f64,
    score: f64,
}

pub fn help() {
    println!(
        "RT-aware rolling arbitrage policy with DA lookahead and congestion-safe incremental dispatch."
    );
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    if state.time_step >= challenge.num_steps {
        return Err(anyhow!("time_step is outside the challenge horizon"));
    }
    if state.rt_prices.len() != challenge.network.num_nodes {
        return Err(anyhow!("state has malformed RT prices"));
    }

    let flow_base = base_flows(challenge, state);
    let mut candidates = build_candidates(challenge, state, &flow_base);
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut action = vec![0.0; challenge.num_batteries];
    let mut flows = flow_base;
    for candidate in candidates {
        if candidate.score <= 0.0 || candidate.action.abs() < MIN_ACTION_MW {
            continue;
        }
        let accepted = max_feasible_increment(
            challenge,
            candidate.battery_idx,
            candidate.action,
            &flows,
            FLOW_HEADROOM,
        );
        if accepted.abs() < MIN_ACTION_MW {
            continue;
        }
        action[candidate.battery_idx] += accepted;
        apply_flow_delta(challenge, candidate.battery_idx, accepted, &mut flows);
    }

    clamp_to_bounds(state, &mut action);
    if !is_feasible(challenge, state, &action) {
        action = scale_to_feasible(challenge, state, &action)?;
    }
    Ok(action)
}

fn build_candidates(challenge: &Challenge, state: &State, base_flows: &[f64]) -> Vec<Candidate> {
    let mut candidates = Vec::with_capacity(challenge.num_batteries * 2);
    let remaining = challenge.num_steps.saturating_sub(state.time_step + 1);
    let horizon = LOOKAHEAD_STEPS.min(remaining);
    let normalized_stress = line_stress(challenge, base_flows);

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let current_price = state.rt_prices[node];
        let current_da = challenge.market.day_ahead_prices[state.time_step][node];
        let da_weight = if challenge.num_batteries <= 10 {
            0.90
        } else {
            0.45
        };
        let (future_low, future_avg, future_high) =
            future_price_stats(challenge, state.time_step, node, horizon);
        let soc_mid = 0.5 * (battery.soc_min_mwh + battery.soc_max_mwh);
        let soc_span = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_balance = ((state.socs[i] - soc_mid) / soc_span).clamp(-0.5, 0.5);
        let (min_bound, max_bound) = state.action_bounds[i];
        if max_bound > MIN_ACTION_MW {
            let inventory_bias = if horizon <= 2 { 0.0 } else { 4.0 * soc_balance };
            let reference_sell = future_avg + 0.20 * (future_high - future_low);
            let da_spread = current_da - future_avg;
            let margin = current_price - reference_sell - constants::KAPPA_TX
                + inventory_bias
                + da_weight * da_spread;
            let score = margin * constants::DELTA_T
                - congestion_penalty_for_action(challenge, i, max_bound, base_flows)
                - 0.25 * normalized_stress;
            let size = choose_action_size(max_bound, score, battery.power_discharge_mw);
            candidates.push(Candidate {
                battery_idx: i,
                action: size,
                score,
            });
        }

        if min_bound < -MIN_ACTION_MW && horizon > 0 {
            let inventory_bias = 4.0 * soc_balance;
            let reference_sell = future_avg + 0.20 * (future_high - future_low);
            let da_spread = future_avg - current_da;
            let round_trip_gain =
                reference_sell * battery.efficiency_charge * battery.efficiency_discharge
                    - current_price;
            let margin =
                round_trip_gain - constants::KAPPA_TX - inventory_bias + da_weight * da_spread;
            let score = margin * constants::DELTA_T
                - congestion_penalty_for_action(challenge, i, min_bound, base_flows)
                - 0.25 * normalized_stress;
            let size = -choose_action_size(-min_bound, score, battery.power_charge_mw);
            candidates.push(Candidate {
                battery_idx: i,
                action: size,
                score,
            });
        }
    }

    candidates
}

fn future_price_stats(
    challenge: &Challenge,
    time_step: usize,
    node: usize,
    horizon: usize,
) -> (f64, f64, f64) {
    if horizon == 0 {
        let price = challenge.market.day_ahead_prices[time_step][node];
        return (price, price, price);
    }
    let end = (time_step + 1 + horizon).min(challenge.num_steps);
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut count = 0.0;
    for t in time_step + 1..end {
        let price = challenge.market.day_ahead_prices[t][node];
        low = low.min(price);
        high = high.max(price);
        sum += price;
        count += 1.0;
    }
    let avg = if count > 0.0 {
        sum / count
    } else {
        challenge.market.day_ahead_prices[time_step][node]
    };
    (low.min(avg), avg, high.max(avg))
}

fn choose_action_size(limit: f64, score: f64, nameplate: f64) -> f64 {
    if score <= 0.0 {
        0.0
    } else if score > 2.0 {
        limit
    } else if score > 0.5 {
        limit.min(0.75 * nameplate)
    } else {
        limit.min(COARSE_STEP_MW.max(0.35 * nameplate))
    }
}

fn base_flows(challenge: &Challenge, state: &State) -> Vec<f64> {
    let zero = vec![0.0; challenge.num_batteries];
    let injections = challenge.compute_total_injections(state, &zero);
    challenge.network.compute_flows(&injections)
}

fn line_stress(challenge: &Challenge, flows: &[f64]) -> f64 {
    flows
        .iter()
        .enumerate()
        .map(|(l, flow)| flow.abs() / challenge.network.flow_limits[l].max(EPS))
        .fold(0.0, f64::max)
}

fn congestion_penalty_for_action(
    challenge: &Challenge,
    battery_idx: usize,
    action: f64,
    flows: &[f64],
) -> f64 {
    let node = challenge.batteries[battery_idx].node;
    let mut penalty = 0.0;
    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].max(EPS);
        let before = flows[l].abs() / limit;
        let after = (flows[l] + challenge.network.ptdf[l][node] * action).abs() / limit;
        if after > before {
            penalty += (after - before) * (1.0 + 4.0 * after.max(0.0));
        }
    }
    penalty
}

fn max_feasible_increment(
    challenge: &Challenge,
    battery_idx: usize,
    requested: f64,
    flows: &[f64],
    headroom: f64,
) -> f64 {
    let node = challenge.batteries[battery_idx].node;
    let mut scale: f64 = 1.0;
    for l in 0..challenge.network.num_lines {
        let ptdf = challenge.network.ptdf[l][node];
        let delta = ptdf * requested;
        if delta.abs() <= EPS {
            continue;
        }
        let limit = challenge.network.flow_limits[l] * headroom;
        let current = flows[l];
        let upper = (limit - current) / delta;
        let lower = (-limit - current) / delta;
        let feasible_high = upper.max(lower);
        let feasible_low = upper.min(lower);
        if feasible_high < 0.0 || feasible_low > 1.0 {
            return 0.0;
        }
        scale = scale.min(feasible_high.clamp(0.0, 1.0));
    }
    requested * scale
}

fn apply_flow_delta(challenge: &Challenge, battery_idx: usize, delta: f64, flows: &mut [f64]) {
    let node = challenge.batteries[battery_idx].node;
    for (l, flow) in flows.iter_mut().enumerate() {
        *flow += challenge.network.ptdf[l][node] * delta;
    }
}

fn clamp_to_bounds(state: &State, action: &mut [f64]) {
    for (u, &(low, high)) in action.iter_mut().zip(state.action_bounds.iter()) {
        *u = u.clamp(low, high);
        if u.abs() < MIN_ACTION_MW {
            *u = 0.0;
        }
    }
}

fn is_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let injections = challenge.compute_total_injections(state, action);
    let flows = challenge.network.compute_flows(&injections);
    challenge.network.verify_flows(&flows).is_ok()
}

fn scale_to_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> Result<Vec<f64>> {
    let zero = vec![0.0; action.len()];
    if !is_feasible(challenge, state, &zero) {
        return Err(anyhow!("zero action is unexpectedly infeasible"));
    }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..40 {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = action.iter().map(|u| mid * u).collect();
        if is_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(action.iter().map(|u| low * u).collect())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
