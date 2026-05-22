use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

use tig_challenges::energy_arbitrage::*;
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

const EPS: f64 = 1e-9;
const LOOKAHEAD: usize = 32;
const FLOW_REPAIR_ITERS: usize = 72;
const FLOW_BISECT_ITERS: usize = 36;
const MIN_ACTION_MW: f64 = 1e-6;

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
    let mut action = vec![0.0; challenge.num_batteries];
    let remaining = challenge.num_steps.saturating_sub(state.time_step + 1);

    for b in 0..challenge.num_batteries {
        let battery = &challenge.batteries[b];
        let node = battery.node;
        let price = state.rt_prices[node];
        let (min_u, max_u) = state.action_bounds[b];
        let stats = forecast_stats(challenge, state.time_step, node);
        let soc = state.socs[b];
        let span = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_frac = ((soc - battery.soc_min_mwh) / span).clamp(0.0, 1.0);

        let endgame = remaining <= 8;
        let volatility = (stats.max_price - stats.min_price).max(4.0);
        let deadband = (0.06 * stats.mean_abs + 0.10 * volatility).clamp(3.0, 22.0);
        let round_trip = battery.efficiency_charge * battery.efficiency_discharge;

        let mut desired = 0.0;
        if endgame {
            if price > constants::KAPPA_TX + 0.25 {
                desired = max_u * endgame_discharge_fraction(price, stats.mean_price);
            } else if price < -2.0 {
                desired = min_u * 0.65;
            }
        } else {
            let high_trigger = stats.mean_price + deadband;
            let low_trigger = stats.mean_price * round_trip - deadband;
            let reserve_boost: f64 = if stats.max_price > stats.mean_price + 2.0 * deadband {
                0.10
            } else {
                0.0
            };
            let target_soc = (0.50 + reserve_boost).clamp(0.42, 0.68);

            if price > high_trigger && soc_frac > 0.12 {
                let strength = ((price - high_trigger) / (volatility + deadband)).clamp(0.20, 1.0);
                let soc_gate = ((soc_frac - 0.10) / 0.55).clamp(0.15, 1.0);
                desired = max_u * strength * soc_gate;
            } else if price < low_trigger && soc_frac < 0.92 {
                let strength = ((low_trigger - price) / (volatility + deadband)).clamp(0.20, 1.0);
                let soc_gate = ((0.95 - soc_frac) / 0.60).clamp(0.15, 1.0);
                desired = min_u * strength * soc_gate;
            } else if soc_frac > target_soc + 0.18 && price > stats.mean_price + 0.35 * deadband {
                desired = max_u * ((soc_frac - target_soc) / 0.35).clamp(0.10, 0.55);
            } else if soc_frac < target_soc - 0.18 && price < stats.mean_price - 0.35 * deadband {
                desired = min_u * ((target_soc - soc_frac) / 0.35).clamp(0.10, 0.55);
            }
        }

        if desired.abs() < MIN_ACTION_MW {
            desired = 0.0;
        }
        action[b] = desired.clamp(min_u, max_u);
    }

    enforce_flow_feasibility(challenge, state, action)
}

#[derive(Clone, Copy)]
struct PriceStats {
    mean_price: f64,
    mean_abs: f64,
    min_price: f64,
    max_price: f64,
}

fn forecast_stats(challenge: &Challenge, time_step: usize, node: usize) -> PriceStats {
    let end = (time_step + LOOKAHEAD + 1).min(challenge.num_steps);
    let mut count = 0.0;
    let mut sum = 0.0;
    let mut sum_abs = 0.0;
    let mut min_price = f64::INFINITY;
    let mut max_price = f64::NEG_INFINITY;

    for t in time_step + 1..end {
        let p = expected_rt_price(challenge, t, node);
        count += 1.0;
        sum += p;
        sum_abs += p.abs();
        min_price = min_price.min(p);
        max_price = max_price.max(p);
    }

    if count <= 0.0 {
        let p = expected_rt_price(challenge, time_step, node);
        return PriceStats {
            mean_price: p,
            mean_abs: p.abs(),
            min_price: p,
            max_price: p,
        };
    }

    PriceStats {
        mean_price: sum / count,
        mean_abs: sum_abs / count,
        min_price,
        max_price,
    }
}

fn expected_rt_price(challenge: &Challenge, time_step: usize, node: usize) -> f64 {
    let da = challenge.market.day_ahead_prices[time_step][node];
    let jump_multiplier = if challenge.market.params.tail_index > 1.0 {
        let pareto_mean =
            challenge.market.params.tail_index / (challenge.market.params.tail_index - 1.0);
        challenge.market.params.jump_probability * pareto_mean
    } else {
        0.0
    };
    da * (1.0 + jump_multiplier) + expected_congestion_premium(challenge, time_step, node)
}

fn expected_congestion_premium(challenge: &Challenge, time_step: usize, node: usize) -> f64 {
    if time_step == 0 || node >= challenge.network.node_incident_lines.len() {
        return 0.0;
    }

    let prior_injections = &challenge.exogenous_injections[time_step - 1];
    let flows = challenge.network.compute_flows(prior_injections);
    let mut no_congestion_prob = 1.0;

    for &line in &challenge.network.node_incident_lines[node] {
        let denom = challenge.network.congestion_threshold * challenge.network.flow_limits[line];
        if denom <= EPS {
            continue;
        }
        let p = (flows[line].abs() / denom).powf(10.0).clamp(0.0, 1.0);
        no_congestion_prob *= 1.0 - p;
    }

    let congestion_prob = 1.0 - no_congestion_prob;
    let expected_positive_normal = 0.398_942_280_401_432_7;
    constants::GAMMA_PRICE * expected_positive_normal * congestion_prob
}

fn endgame_discharge_fraction(price: f64, mean_price: f64) -> f64 {
    if price > mean_price + 20.0 {
        1.0
    } else if price > mean_price + 8.0 {
        0.75
    } else if price > 10.0 {
        0.55
    } else {
        0.30
    }
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
) -> Result<Vec<f64>> {
    clamp_to_bounds(state, &mut action);

    for _ in 0..FLOW_REPAIR_ITERS {
        let flows = flows_for_action(challenge, state, &action);
        let Some((line, flow, excess)) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !reduce_worsening_actions(challenge, state, line, flow, excess, &mut action) {
            break;
        }
        clamp_to_bounds(state, &mut action);
    }

    if is_feasible(challenge, state, &action) {
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    if !is_feasible(challenge, state, &zero) {
        return Err(anyhow!("zero battery action is unexpectedly flow-infeasible"));
    }

    let base = action;
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..FLOW_BISECT_ITERS {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| u * mid).collect();
        if is_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(base.into_iter().map(|u| u * low).collect())
}

fn clamp_to_bounds(state: &State, action: &mut [f64]) {
    for (u, &(lo, hi)) in action.iter_mut().zip(state.action_bounds.iter()) {
        *u = u.clamp(lo, hi);
        if u.abs() < MIN_ACTION_MW {
            *u = 0.0;
        }
    }
}

fn flows_for_action(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    challenge.network.compute_flows(&injections)
}

fn is_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = flows_for_action(challenge, state, action);
    challenge.network.verify_flows(&flows).is_ok()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<(usize, f64, f64)> {
    let mut best = None;
    for (line, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[line];
        let excess = flow.abs() - limit;
        if excess > constants::EPS_FLOW * limit {
            match best {
                Some((_, _, best_excess)) if excess <= best_excess => {}
                _ => best = Some((line, flow, excess)),
            }
        }
    }
    best
}

fn reduce_worsening_actions(
    challenge: &Challenge,
    state: &State,
    line: usize,
    flow: f64,
    excess: f64,
    action: &mut [f64],
) -> bool {
    let direction = flow.signum();
    if direction.abs() <= EPS {
        return false;
    }

    let mut worsening = Vec::new();
    let mut total_worsening = 0.0;
    for (b, battery) in challenge.batteries.iter().enumerate() {
        let contribution = challenge.network.ptdf[line][battery.node] * action[b];
        let signed = direction * contribution;
        if signed > EPS {
            let value_density = action_priority_value(challenge, state, b, action[b]) / signed;
            worsening.push((b, signed, value_density));
            total_worsening += signed;
        }
    }

    if worsening.is_empty() || total_worsening <= EPS {
        return false;
    }

    worsening.sort_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut remaining = (1.03 * excess).min(total_worsening);
    for (b, signed, _) in worsening {
        if remaining <= EPS {
            break;
        }
        let reduction = remaining.min(signed);
        let keep = (1.0 - reduction / signed).clamp(0.0, 1.0);
        action[b] *= keep;
        remaining -= reduction;
    }
    true
}

fn action_priority_value(challenge: &Challenge, state: &State, b: usize, u: f64) -> f64 {
    if u.abs() < MIN_ACTION_MW {
        return 0.0;
    }

    let battery = &challenge.batteries[b];
    let price = state.rt_prices[battery.node];
    let abs_u = u.abs();
    let dt = constants::DELTA_T;
    let tx_cost = constants::KAPPA_TX * abs_u * dt;
    let deg_base = (abs_u * dt) / battery.capacity_mwh;
    let deg_cost = constants::KAPPA_DEG * deg_base.powf(constants::BETA_DEG);

    if u > 0.0 {
        u * price * dt - tx_cost - deg_cost
    } else {
        let stats = forecast_stats(challenge, state.time_step, battery.node);
        let upside = (stats.max_price - stats.mean_price).max(0.0);
        let expected_exit = stats.mean_price + 0.25 * upside;
        let round_trip = battery.efficiency_charge * battery.efficiency_discharge;
        abs_u * dt * (expected_exit * round_trip - price) - tx_cost - deg_cost
    }
}

pub fn help() {
    println!("Prometheus solver");
}
