use anyhow::{anyhow, Result};
use tig_challenges::energy_arbitrage::{Challenge, State};

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;
const LOOKAHEAD_WINDOW: usize = 32;

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

fn signed_line_sensitivity(
    challenge: &Challenge,
    line: usize,
    signed_direction: f64,
    node: usize,
) -> f64 {
    signed_direction * challenge.network.ptdf[line][node]
}

fn estimate_corrective_capacity(
    challenge: &Challenge,
    state: &State,
    action: &[f64],
    line: usize,
    signed_direction: f64,
) -> f64 {
    const SIGN_FLIP_ROOM_WEIGHT: f64 = 0.5;

    let mut cap = 0.0f64;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let g = signed_line_sensitivity(challenge, line, signed_direction, battery.node);
        let g_abs = g.abs();
        if g_abs <= EPS {
            continue;
        }

        let (min_bound, max_bound) = state.action_bounds[i];
        let ai = action[i];

        let room = if g > 0.0 {
            if ai >= 0.0 {
                let direct_room = (ai - min_bound.max(0.0)).max(0.0);
                let sign_flip_room = (-min_bound).max(0.0);
                direct_room + SIGN_FLIP_ROOM_WEIGHT * sign_flip_room
            } else {
                (ai - min_bound).max(0.0)
            }
        } else if ai <= 0.0 {
            let direct_room = (max_bound.min(0.0) - ai).max(0.0);
            let sign_flip_room = max_bound.max(0.0);
            direct_room + SIGN_FLIP_ROOM_WEIGHT * sign_flip_room
        } else {
            (max_bound - ai).max(0.0)
        };

        cap += g_abs * room;
    }

    cap
}

fn most_violated_line(
    challenge: &Challenge,
    state: &State,
    action: &[f64],
    flows: &[f64],
) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    let mut best_score = f64::MIN;

    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation <= tig_challenges::energy_arbitrage::constants::EPS_FLOW * limit {
            continue;
        }

        let signed_direction = flow.signum();
        if signed_direction.abs() <= EPS {
            continue;
        }

        let cap = estimate_corrective_capacity(challenge, state, action, l, signed_direction);
        let score = violation / (cap + EPS);

        if score > best_score {
            best_score = score;
            best = Some(Violation {
                line: l,
                flow,
                amount: violation,
            });
        }
    }

    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, state, action, &flows).is_none()
}

fn allocate_weighted_corrections(weights: &[f64], caps: &[f64], amount: f64) -> Vec<f64> {
    let n = weights.len();
    let mut alloc = vec![0.0f64; n];
    if n == 0 {
        return alloc;
    }

    let mut target = amount.max(0.0);
    let total_cap: f64 = caps.iter().map(|&c| c.max(0.0)).sum();
    target = target.min(total_cap);

    if target <= EPS {
        return alloc;
    }
    if target >= total_cap - EPS {
        return caps.iter().map(|&c| c.max(0.0)).collect();
    }

    let sum_weights: f64 = (0..n)
        .map(|j| {
            if weights[j] > EPS && caps[j] > EPS {
                weights[j]
            } else {
                0.0
            }
        })
        .sum();

    if sum_weights <= EPS {
        return alloc;
    }

    let total_at = |lambda: f64| -> f64 {
        if lambda <= EPS {
            return 0.0;
        }

        let mut total = 0.0f64;
        for j in 0..n {
            let w = weights[j];
            let c = caps[j].max(0.0);
            if w <= EPS || c <= EPS {
                continue;
            }

            let scaled = lambda * w / c;
            total += if scaled >= 50.0 {
                c
            } else {
                c * (1.0 - (-scaled).exp())
            };
        }
        total.min(total_cap)
    };

    let alloc_at = |lambda: f64| -> Vec<f64> {
        let mut out = vec![0.0f64; n];
        if lambda <= EPS {
            return out;
        }

        for j in 0..n {
            let w = weights[j];
            let c = caps[j].max(0.0);
            if w <= EPS || c <= EPS {
                continue;
            }

            let scaled = lambda * w / c;
            out[j] = if scaled >= 50.0 {
                c
            } else {
                c * (1.0 - (-scaled).exp())
            };
        }

        out
    };

    let mut low = 0.0f64;
    let mut high = (target / sum_weights).max(EPS);
    while total_at(high) < target {
        low = high;
        high *= 2.0;
        if high >= 1e18 {
            break;
        }
    }

    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        if total_at(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }

    let low_total = total_at(low);
    let high_total = total_at(high);
    let low_alloc = alloc_at(low);

    if high_total - low_total <= EPS {
        return low_alloc;
    }

    let high_alloc = alloc_at(high);
    let mix = ((target - low_total) / (high_total - low_total)).clamp(0.0, 1.0);

    for j in 0..n {
        alloc[j] = low_alloc[j] + mix * (high_alloc[j] - low_alloc[j]);
    }

    alloc
}

fn signed_action_score(
    action: f64,
    charge_scores: &[f64],
    discharge_scores: &[f64],
    idx: usize,
) -> f64 {
    if action > EPS {
        discharge_scores[idx].max(0.0)
    } else if action < -EPS {
        charge_scores[idx].max(0.0)
    } else {
        0.0
    }
}

fn soften_most_violated_line(
    challenge: &Challenge,
    state: &State,
    violation: &Violation,
    charge_scores: &[f64],
    discharge_scores: &[f64],
    line_duals: &[f64],
    action: &mut [f64],
) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    struct HarmEntry {
        idx: usize,
        g2: f64,
        sc: f64,
        value: f64,
        dual_bias: f64,
    }

    struct HelpEntry {
        idx: usize,
        g_abs: f64,
        g2: f64,
        sign: f64,
        cap: f64,
        value: f64,
        dual_bias: f64,
    }

    let mut harmful: Vec<HarmEntry> = Vec::new();
    let mut helpful: Vec<HelpEntry> = Vec::new();

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let g = signed_direction * challenge.network.ptdf[line][battery.node];
        if g.abs() <= EPS {
            continue;
        }

        let ai = action[i];
        let sc = g * ai;
        let g2 = g * g;
        let value = signed_action_score(ai, charge_scores, discharge_scores, i);
        let dual_bias = if ai.abs() > EPS {
            line_dual_move_bias(challenge, line_duals, i, ai)
        } else {
            0.0
        };

        if sc > EPS {
            if g2 > EPS {
                harmful.push(HarmEntry {
                    idx: i,
                    g2,
                    sc,
                    value,
                    dual_bias,
                });
            }
            continue;
        }

        if sc < -EPS && g2 > EPS {
            let (min_bound, max_bound) = state.action_bounds[i];
            let room = if ai > EPS {
                (max_bound - ai).max(0.0)
            } else if ai < -EPS {
                (ai - min_bound).max(0.0)
            } else {
                0.0
            };

            if room > EPS {
                helpful.push(HelpEntry {
                    idx: i,
                    g_abs: g.abs(),
                    g2,
                    sign: ai.signum(),
                    cap: g.abs() * room,
                    value,
                    dual_bias,
                });
            }
        }
    }

    if harmful.is_empty() && helpful.is_empty() {
        return false;
    }

    let mut changed = false;
    let help_max = helpful.iter().map(|e| e.value).fold(0.0f64, |a, b| a.max(b));
    let help_min = helpful
        .iter()
        .map(|e| e.value)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let use_help_value = help_max > EPS && help_max - help_min > 0.10 * help_max;
    let help_weights: Vec<f64> = helpful
        .iter()
        .map(|e| {
            let eco = if use_help_value {
                0.80 + 0.60 * (e.value / help_max).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let dual_relief = (-e.dual_bias).max(0.0) / (1.0 + e.dual_bias.abs());
            let dual_harm = e.dual_bias.max(0.0) / (1.0 + e.dual_bias.abs());
            let dual = (1.0 + 0.45 * dual_relief - 0.20 * dual_harm).clamp(0.70, 1.55);
            e.g2 * eco * dual
        })
        .collect();
    let help_caps: Vec<f64> = helpful.iter().map(|e| e.cap).collect();
    let help_alloc = allocate_weighted_corrections(&help_weights, &help_caps, violation.amount);

    let mut corrected = 0.0f64;
    for (j, e) in helpful.iter().enumerate() {
        let r = help_alloc[j];
        if r <= EPS {
            continue;
        }

        let delta = r / e.g_abs;
        let (min_bound, max_bound) = state.action_bounds[e.idx];
        let new_ai = if e.sign > 0.0 {
            (action[e.idx] + delta).min(max_bound)
        } else {
            (action[e.idx] - delta).max(min_bound)
        };

        if (new_ai - action[e.idx]).abs() > EPS {
            action[e.idx] = new_ai;
            corrected += r;
            changed = true;
        }
    }

    let remaining = (violation.amount - corrected).max(0.0);
    if remaining <= EPS {
        return changed;
    }

    let harm_max = harmful.iter().map(|e| e.value).fold(0.0f64, |a, b| a.max(b));
    let harm_min = harmful
        .iter()
        .map(|e| e.value)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let use_harm_value = harm_max > EPS && harm_max - harm_min > 0.10 * harm_max;
    let harm_weights: Vec<f64> = harmful
        .iter()
        .map(|e| {
            let eco = if use_harm_value {
                1.25 - 0.60 * (e.value / harm_max).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let dual_relief = (-e.dual_bias).max(0.0) / (1.0 + e.dual_bias.abs());
            let dual_harm = e.dual_bias.max(0.0) / (1.0 + e.dual_bias.abs());
            let dual = (1.0 + 0.55 * dual_harm - 0.20 * dual_relief).clamp(0.70, 1.65);
            e.g2 * eco * dual
        })
        .collect();
    let harm_caps: Vec<f64> = harmful.iter().map(|e| e.sc).collect();
    let harm_alloc = allocate_weighted_corrections(&harm_weights, &harm_caps, remaining);

    for (j, e) in harmful.iter().enumerate() {
        let r = harm_alloc[j];
        if r <= EPS {
            continue;
        }

        let i = e.idx;
        let ai = action[i];
        if ai.abs() <= EPS {
            continue;
        }

        let keep = (1.0 - r / e.sc).clamp(0.0, 1.0);
        let new_ai = ai * keep;
        if (new_ai - ai).abs() > EPS {
            action[i] = new_ai;
            changed = true;
        }
    }

    changed
}

fn directional_project_to_feasible(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
    charge_scores: &[f64],
    discharge_scores: &[f64],
    line_duals: &mut [f64],
) -> Vec<f64> {
    for _ in 0..(MAX_FLOW_ADJUST_ITERS / 2) {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, state, &action, &flows) else {
            update_line_dual_memory(challenge, &flows, None, line_duals);
            return action;
        };

        update_line_dual_memory(challenge, &flows, Some(violation.line), line_duals);

        let line = violation.line;
        let signed_direction = violation.flow.signum();
        if signed_direction.abs() <= EPS {
            break;
        }

        struct HarmEntry {
            idx: usize,
            sc: f64,
            g2: f64,
            value: f64,
            dual_bias: f64,
        }

        let mut harmful: Vec<HarmEntry> = Vec::new();
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let g = signed_direction * challenge.network.ptdf[line][battery.node];
            if g.abs() <= EPS {
                continue;
            }

            let sc = g * action[i];
            if sc > EPS {
                harmful.push(HarmEntry {
                    idx: i,
                    sc,
                    g2: g * g,
                    value: signed_action_score(action[i], charge_scores, discharge_scores, i),
                    dual_bias: line_dual_move_bias(challenge, line_duals, i, action[i]),
                });
            }
        }

        if harmful.is_empty() {
            break;
        }

        let max_value = harmful.iter().map(|e| e.value).fold(0.0f64, |a, b| a.max(b));
        let weights: Vec<f64> = harmful
            .iter()
            .map(|e| {
                let eco = if max_value > EPS {
                    1.35 - 0.65 * (e.value / max_value).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let dual_relief = (-e.dual_bias).max(0.0) / (1.0 + e.dual_bias.abs());
                let dual_harm = e.dual_bias.max(0.0) / (1.0 + e.dual_bias.abs());
                let dual = (1.0 + 0.60 * dual_harm - 0.20 * dual_relief).clamp(0.70, 1.75);
                e.g2 * eco * dual
            })
            .collect();
        let caps: Vec<f64> = harmful.iter().map(|e| e.sc).collect();
        let alloc = allocate_weighted_corrections(&weights, &caps, violation.amount);

        let mut changed = false;
        for (j, e) in harmful.iter().enumerate() {
            let r = alloc[j];
            if r <= EPS {
                continue;
            }

            let keep = (1.0 - r / e.sc).clamp(0.0, 1.0);
            let new_ai = action[e.idx] * keep;
            if (new_ai - action[e.idx]).abs() > EPS {
                action[e.idx] = new_ai;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    action
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
    charge_scores: &[f64],
    discharge_scores: &[f64],
    line_duals: &mut [f64],
) -> Result<Vec<f64>> {
    for dual in line_duals.iter_mut() {
        *dual = 0.0;
    }

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, state, &action, &flows) else {
            update_line_dual_memory(challenge, &flows, None, line_duals);
            return Ok(action);
        };
        update_line_dual_memory(challenge, &flows, Some(violation.line), line_duals);
        if !soften_most_violated_line(
            challenge,
            state,
            &violation,
            charge_scores,
            discharge_scores,
            line_duals,
            &mut action,
        ) {
            break;
        }
    }
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!("Grid infeasible even with zero battery actions"));
    }

    let mut action = directional_project_to_feasible(
        challenge,
        state,
        action,
        charge_scores,
        discharge_scores,
        line_duals,
    );
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }

    for _ in 0..(MAX_FLOW_ADJUST_ITERS / 4) {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, state, &action, &flows) else {
            update_line_dual_memory(challenge, &flows, None, line_duals);
            return Ok(action);
        };
        update_line_dual_memory(challenge, &flows, Some(violation.line), line_duals);
        if !soften_most_violated_line(
            challenge,
            state,
            &violation,
            charge_scores,
            discharge_scores,
            line_duals,
            &mut action,
        ) {
            break;
        }
    }
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
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

fn opportunity_edges(
    current_price: f64,
    future_prices: &[Vec<f64>],
    node: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    if future_prices.is_empty() {
        return (current_price, current_price, 0.0, 0.0, 0.0, 0.0);
    }

    let mut min_future = f64::MAX;
    let mut max_future = f64::MIN;
    let mut weighted_charge_edge = 0.0f64;
    let mut weighted_discharge_edge = 0.0f64;
    let mut weight_sum = 0.0f64;
    let mut first_charge_timing = 0.0f64;
    let mut first_discharge_timing = 0.0f64;
    let mut best_timed_charge = 0.0f64;
    let mut best_timed_discharge = 0.0f64;
    let mut saw_charge = false;
    let mut saw_discharge = false;

    for (dt, prices) in future_prices.iter().enumerate() {
        let price = prices[node];
        min_future = min_future.min(price);
        max_future = max_future.max(price);

        let diff = price - current_price;
        let smooth_decay = 1.0 / (1.0 + 0.15 * dt as f64);
        let timing_decay = 1.0 / (1.0 + 0.35 * dt as f64);
        weight_sum += smooth_decay;

        if diff > 0.0 {
            weighted_charge_edge += smooth_decay * diff;
            let timed = diff * timing_decay;
            best_timed_charge = best_timed_charge.max(timed);
            if !saw_charge {
                first_charge_timing = timed;
                saw_charge = true;
            }
        } else if diff < 0.0 {
            let gain = -diff;
            weighted_discharge_edge += smooth_decay * gain;
            let timed = gain * timing_decay;
            best_timed_discharge = best_timed_discharge.max(timed);
            if !saw_discharge {
                first_discharge_timing = timed;
                saw_discharge = true;
            }
        }
    }

    let extreme_charge_edge = (max_future - current_price).max(0.0);
    let extreme_discharge_edge = (current_price - min_future).max(0.0);
    let timing_charge_edge = if saw_charge {
        0.70 * first_charge_timing + 0.30 * best_timed_charge
    } else {
        0.0
    };
    let timing_discharge_edge = if saw_discharge {
        0.70 * first_discharge_timing + 0.30 * best_timed_discharge
    } else {
        0.0
    };

    let (magnitude_charge_edge, magnitude_discharge_edge) = if future_prices.len() <= 4 {
        (extreme_charge_edge, extreme_discharge_edge)
    } else {
        let avg_charge_edge = weighted_charge_edge / weight_sum.max(EPS);
        let avg_discharge_edge = weighted_discharge_edge / weight_sum.max(EPS);
        (
            0.70 * extreme_charge_edge + 0.30 * avg_charge_edge,
            0.70 * extreme_discharge_edge + 0.30 * avg_discharge_edge,
        )
    };

    (
        min_future,
        max_future,
        magnitude_charge_edge,
        magnitude_discharge_edge,
        timing_charge_edge,
        timing_discharge_edge,
    )
}

fn cross_sectional_price_context(current_prices: &[f64]) -> (Vec<f64>, f64) {
    if current_prices.is_empty() {
        return (Vec::new(), 0.0);
    }

    let n = current_prices.len() as f64;
    let mut sum = 0.0f64;
    let mut abs_sum = 0.0f64;
    let mut min_price = f64::MAX;
    let mut max_price = f64::MIN;

    for &price in current_prices.iter() {
        sum += price;
        abs_sum += price.abs();
        min_price = min_price.min(price);
        max_price = max_price.max(price);
    }

    let mean = sum / n.max(1.0);
    let mut var = 0.0f64;
    for &price in current_prices.iter() {
        let d = price - mean;
        var += d * d;
    }

    let std = (var / n.max(1.0)).sqrt();
    let spread = (max_price - min_price).max(0.0);
    let reference_level = (abs_sum / n.max(1.0)).max(20.0);
    let dispersion_ratio = spread.max(2.0 * std) / reference_level;
    let activation = ((dispersion_ratio - 0.04) / 0.16).clamp(0.0, 1.0);
    let denom = std.max(0.25 * spread).max(1.0);

    let signals = current_prices
        .iter()
        .map(|&price| {
            let z_part = ((price - mean) / (2.5 * denom)).clamp(-1.0, 1.0);
            let range_part = if spread > EPS {
                (2.0 * (price - min_price) / spread - 1.0).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            activation * (0.65 * z_part + 0.35 * range_part)
        })
        .collect();

    let edge_scale = activation * (0.20 * std + 0.12 * spread);
    (signals, edge_scale)
}

fn terminal_inventory_schedule(
    current_soc: f64,
    soc_min: f64,
    power_charge_mw: f64,
    power_discharge_mw: f64,
    steps_after_now: usize,
) -> (f64, f64, f64) {
    let charge_power = power_charge_mw.max(0.0);
    let discharge_power = power_discharge_mw.max(0.0);
    let energy_above_min = (current_soc - soc_min).max(0.0);
    let future_discharge_capacity = steps_after_now as f64 * discharge_power;

    let required_discharge_now =
        (energy_above_min - future_discharge_capacity).clamp(0.0, discharge_power);

    let charge_room_factor = if charge_power > EPS {
        ((future_discharge_capacity - energy_above_min) / charge_power).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let discharge_urgency = if discharge_power > EPS {
        let soft_target = (future_discharge_capacity - discharge_power).max(0.0);
        ((energy_above_min - soft_target) / discharge_power).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (
        charge_room_factor,
        discharge_urgency,
        required_discharge_now,
    )
}

fn smooth_signed_drive(charge_utility: f64, discharge_utility: f64) -> f64 {
    let total = charge_utility.max(0.0) + discharge_utility.max(0.0);
    if total <= EPS {
        return 0.0;
    }

    let balance = ((discharge_utility - charge_utility) / (total + EPS)).clamp(-1.0, 1.0);
    let dead_zone = 0.06;
    let mag = balance.abs();
    if mag <= dead_zone {
        return 0.0;
    }

    let x = ((mag - dead_zone) / (1.0 - dead_zone)).clamp(0.0, 1.0);
    let smooth_mag = x * x * (3.0 - 2.0 * x);
    balance.signum() * smooth_mag
}

fn smoothstep01(x: f64) -> f64 {
    let y = x.clamp(0.0, 1.0);
    y * y * (3.0 - 2.0 * y)
}

fn marginal_push_fraction(
    directional_score: f64,
    opposing_score: f64,
    current_utilization: f64,
    score_scale: f64,
    sign_aligned: bool,
) -> f64 {
    if directional_score <= EPS {
        return 0.0;
    }

    let relative_strength = (directional_score / score_scale.max(EPS)).clamp(0.0, 1.0);
    let dominance =
        (directional_score / (directional_score + opposing_score + EPS)).clamp(0.0, 1.0);
    let signal = 0.55 * relative_strength.sqrt() + 0.45 * dominance;
    let shaped_signal = smoothstep01((signal - 0.12) / 0.88);

    let util = current_utilization.clamp(0.0, 1.0);
    let diminishing_returns = (1.0 - 0.50 * util - 0.50 * util * util).clamp(0.0, 1.0);
    let sign_flip_barrier = if sign_aligned {
        1.0
    } else {
        dominance * dominance * dominance
    };

    (shaped_signal * diminishing_returns * sign_flip_barrier).clamp(0.0, 1.0)
}

fn marginal_target_action(
    base: f64,
    min_bound: f64,
    max_bound: f64,
    directional_score: f64,
    opposing_score: f64,
    score_scale: f64,
    prefer_charge: bool,
) -> Option<f64> {
    let room = if prefer_charge {
        (base - min_bound).max(0.0)
    } else {
        (max_bound - base).max(0.0)
    };
    if room <= EPS || directional_score <= EPS {
        return None;
    }

    let capacity = if prefer_charge {
        (-min_bound).max(0.0)
    } else {
        max_bound.max(0.0)
    };
    if capacity <= EPS {
        return None;
    }

    let current_utilization = if prefer_charge {
        (-base).max(0.0) / capacity
    } else {
        base.max(0.0) / capacity
    };
    let sign_aligned = if prefer_charge { base <= EPS } else { base >= -EPS };
    let push_fraction = marginal_push_fraction(
        directional_score,
        opposing_score,
        current_utilization,
        score_scale,
        sign_aligned,
    );
    if push_fraction <= EPS {
        return None;
    }

    let bound = if prefer_charge { min_bound } else { max_bound };
    Some(base + push_fraction * (bound - base))
}

fn update_line_dual_memory(
    challenge: &Challenge,
    flows: &[f64],
    focus_line: Option<usize>,
    line_duals: &mut [f64],
) {
    for (l, dual) in line_duals.iter_mut().enumerate() {
        *dual *= 0.88;

        let limit = challenge.network.flow_limits[l].max(EPS);
        let util = flows[l].abs() / limit;
        let stress = ((util - 0.82) / 0.24).clamp(0.0, 1.0);
        if stress <= EPS {
            continue;
        }

        let signed_stress = flows[l].signum() * stress * stress;
        let focus_boost = if Some(l) == focus_line { 1.35 } else { 0.45 };
        *dual = (*dual + focus_boost * signed_stress).clamp(-4.0, 4.0);
    }
}

fn line_dual_move_bias(
    challenge: &Challenge,
    line_duals: &[f64],
    battery_idx: usize,
    signed_move: f64,
) -> f64 {
    if signed_move.abs() <= EPS {
        return 0.0;
    }

    let node = challenge.batteries[battery_idx].node;
    let direction = signed_move.signum();
    let mut bias = 0.0f64;

    for (l, &dual) in line_duals.iter().enumerate() {
        if dual.abs() <= EPS {
            continue;
        }
        bias += dual * challenge.network.ptdf[l][node] * direction;
    }

    bias
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let num_steps = challenge.num_steps;
    let da_prices = &challenge.market.day_ahead_prices;
    let current_prices = &da_prices[t];
    let (node_price_context, context_edge_scale) = cross_sectional_price_context(current_prices);

    let window_end = (t + LOOKAHEAD_WINDOW).min(num_steps);
    let steps_remaining_total = (num_steps - t) as f64 / num_steps as f64;

    let mut action = vec![0.0f64; challenge.num_batteries];
    let mut charge_scores = vec![0.0f64; challenge.num_batteries];
    let mut discharge_scores = vec![0.0f64; challenge.num_batteries];

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (min_bound, max_bound) = state.action_bounds[i];
        let current_price = current_prices[node];

        let future_prices = &da_prices[t..window_end];
        let (
            min_future,
            max_future,
            magnitude_charge_edge,
            magnitude_discharge_edge,
            charge_timing_edge,
            discharge_timing_edge,
        ) = opportunity_edges(current_price, future_prices, node);

        let price_range = (max_future - min_future).max(EPS);

        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = ((state.socs[i] - battery.soc_min_mwh) / soc_range).clamp(0.0, 1.0);
        let charge_headroom = 1.0 - soc_level;
        let discharge_available = soc_level;

        let charge_timing_weight = 0.20 + 0.35 * charge_headroom;
        let discharge_timing_weight = 0.20 + 0.35 * discharge_available;
        let base_charge_edge = (1.0 - charge_timing_weight) * magnitude_charge_edge
            + charge_timing_weight * charge_timing_edge.max(0.5 * magnitude_charge_edge);
        let base_discharge_edge = (1.0 - discharge_timing_weight) * magnitude_discharge_edge
            + discharge_timing_weight * discharge_timing_edge.max(0.5 * magnitude_discharge_edge);

        let node_context = if node < node_price_context.len() {
            node_price_context[node]
        } else {
            0.0
        };
        let charge_edge = base_charge_edge + context_edge_scale * (-node_context).max(0.0);
        let discharge_edge = base_discharge_edge + context_edge_scale * node_context.max(0.0);

        let steps_after_now = num_steps.saturating_sub(t + 1);
        let (terminal_charge_room, terminal_discharge_urgency, required_discharge_now) =
            terminal_inventory_schedule(
                state.socs[i],
                battery.soc_min_mwh,
                battery.power_charge_mw,
                battery.power_discharge_mw,
                steps_after_now,
            );
        let charge_terminal_factor = terminal_charge_room * terminal_charge_room;

        let charge_score = charge_edge
            * charge_headroom.sqrt()
            * charge_terminal_factor
            * (1.0 + 0.12 * (-node_context).max(0.0));
        let mut discharge_score = discharge_edge
            * discharge_available.sqrt()
            * (1.0 + 0.12 * node_context.max(0.0))
            * (1.0 + 0.85 * terminal_discharge_urgency);

        if steps_remaining_total < 0.10 && soc_level > 0.50 && terminal_discharge_urgency < 0.25 {
            discharge_score *= 1.0 + (soc_level - 0.50);
        }

        let opportunity_scale = (price_range + 1.5 * context_edge_scale).max(EPS);
        let charge_intensity = ((charge_edge / opportunity_scale).clamp(0.0, 1.0)
            * charge_terminal_factor.sqrt())
            .clamp(0.0, 1.0);
        let discharge_intensity = ((discharge_edge / opportunity_scale).clamp(0.0, 1.0)
            + 0.20 * terminal_discharge_urgency)
            .clamp(0.0, 1.0);

        let charge_utility = charge_score * (0.25 + 0.75 * charge_intensity);
        let discharge_utility = discharge_score * (0.25 + 0.75 * discharge_intensity);
        let signed_drive = smooth_signed_drive(charge_utility, discharge_utility);

        let mut desired = if signed_drive >= 0.0 {
            battery.power_discharge_mw
                * signed_drive
                * (discharge_intensity
                    * discharge_available.max(terminal_discharge_urgency * terminal_discharge_urgency))
                .sqrt()
        } else {
            -battery.power_charge_mw
                * (-signed_drive)
                * (charge_intensity * charge_headroom).sqrt()
        };

        if required_discharge_now > desired {
            desired = required_discharge_now;
        } else if steps_remaining_total < 0.10
            && soc_level > 0.50
            && terminal_discharge_urgency < 0.25
        {
            let liquidate = battery.power_discharge_mw * (soc_level - 0.50) * 0.8;
            if liquidate > desired {
                desired = liquidate;
            }
        }

        charge_scores[i] = charge_score;
        discharge_scores[i] = discharge_score;
        action[i] = desired.clamp(min_bound, max_bound);
    }

    let mut line_duals = vec![0.0f64; challenge.network.num_lines];
    let mut action = enforce_flow_feasibility(
        challenge,
        state,
        action,
        &charge_scores,
        &discharge_scores,
        &mut line_duals,
    )?;

    let charge_scale = charge_scores
        .iter()
        .copied()
        .fold(0.0f64, |a, b| a.max(b))
        .max(EPS);
    let discharge_scale = discharge_scores
        .iter()
        .copied()
        .fold(0.0f64, |a, b| a.max(b))
        .max(EPS);

    let mut best_charge_i: Option<usize> = None;
    let mut best_charge_target = 0.0f64;
    let mut best_charge_score = 0.0f64;
    let mut best_discharge_i: Option<usize> = None;
    let mut best_discharge_target = 0.0f64;
    let mut best_discharge_score = 0.0f64;

    for i in 0..challenge.num_batteries {
        let (min_bound, max_bound) = state.action_bounds[i];
        let a = action[i];
        let total_score = charge_scores[i] + discharge_scores[i];
        let balance = if total_score > EPS {
            (discharge_scores[i] - charge_scores[i]) / total_score
        } else {
            0.0
        };

        if let Some(target) = marginal_target_action(
            a,
            min_bound,
            max_bound,
            charge_scores[i],
            discharge_scores[i],
            charge_scale,
            true,
        ) {
            let delta = target - a;
            let dual_bias = line_dual_move_bias(challenge, &line_duals, i, delta);
            let dual_relief = (-dual_bias).max(0.0) / (1.0 + dual_bias.abs());
            let dual_harm = dual_bias.max(0.0) / (1.0 + dual_bias.abs());
            let score = charge_scores[i]
                * (-balance).max(0.0)
                * delta.abs()
                * (1.0 + 0.30 * dual_relief)
                / (1.0 + 0.45 * dual_harm);
            if score > best_charge_score {
                best_charge_score = score;
                best_charge_i = Some(i);
                best_charge_target = target;
            }
        }

        if let Some(target) = marginal_target_action(
            a,
            min_bound,
            max_bound,
            discharge_scores[i],
            charge_scores[i],
            discharge_scale,
            false,
        ) {
            let delta = target - a;
            let dual_bias = line_dual_move_bias(challenge, &line_duals, i, delta);
            let dual_relief = (-dual_bias).max(0.0) / (1.0 + dual_bias.abs());
            let dual_harm = dual_bias.max(0.0) / (1.0 + dual_bias.abs());
            let score = discharge_scores[i]
                * balance.max(0.0)
                * delta.abs()
                * (1.0 + 0.30 * dual_relief)
                / (1.0 + 0.45 * dual_harm);
            if score > best_discharge_score {
                best_discharge_score = score;
                best_discharge_i = Some(i);
                best_discharge_target = target;
            }
        }
    }

    let apply_move = |i: usize, target: f64, action: &mut Vec<f64>| {
        let base = action[i];
        let delta = target - base;
        if delta.abs() <= EPS {
            return;
        }

        let mut cand = action.clone();
        cand[i] = target;
        if is_flow_feasible(challenge, state, &cand) {
            action[i] = target;
            return;
        }

        let mut low = 0.0f64;
        let mut high = 1.0f64;

        for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
            let mid = 0.5 * (low + high);
            cand[i] = base + mid * delta;
            if is_flow_feasible(challenge, state, &cand) {
                low = mid;
            } else {
                high = mid;
            }
        }

        action[i] = base + low * delta;
    };

    match (best_charge_i, best_discharge_i) {
        (Some(ci), Some(di)) if ci == di => {
            if best_charge_score >= best_discharge_score {
                apply_move(ci, best_charge_target, &mut action);
            } else {
                apply_move(ci, best_discharge_target, &mut action);
            }
        }
        _ => {
            if let Some(ci) = best_charge_i {
                apply_move(ci, best_charge_target, &mut action);
            }
            if let Some(di) = best_discharge_i {
                apply_move(di, best_discharge_target, &mut action);
            }
        }
    }

    Ok(action)
}