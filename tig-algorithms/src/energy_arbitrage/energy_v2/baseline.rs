use anyhow::{anyhow, Result};
use tig_challenges::energy_arbitrage::{Challenge, State};

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;

struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn reachable_slot_count(energy_mwh: f64, step_energy_mwh: f64, steps_remaining: usize) -> usize {
    if steps_remaining == 0 {
        return 0;
    }
    (((energy_mwh / step_energy_mwh.max(EPS)).ceil() as usize).max(1)).min(steps_remaining)
}

fn scarcity_price_summary(prices: &[f64], low_slots: usize, high_slots: usize) -> (f64, f64, f64, f64) {
    if prices.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut sorted = prices.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let low_slots = low_slots.max(1).min(n);
    let high_slots = high_slots.max(1).min(n);

    let avg_low = sorted.iter().take(low_slots).sum::<f64>() / low_slots as f64;
    let avg_high = sorted.iter().rev().take(high_slots).sum::<f64>() / high_slots as f64;
    let low_threshold = sorted[low_slots - 1];
    let high_threshold = sorted[n - high_slots];

    (avg_low, avg_high, low_threshold, high_threshold)
}

fn directional_value(challenge: &Challenge, state: &State, battery_idx: usize, direction: f64) -> f64 {
    let battery = &challenge.batteries[battery_idx];
    let t = state.time_step;
    if t >= challenge.num_steps {
        return 0.0;
    }

    let node = battery.node;
    let current_price = challenge.market.day_ahead_prices[t][node];
    let steps_remaining = (challenge.num_steps - t).max(1);

    let soc = state.socs[battery_idx];
    let soc_min = battery.soc_min_mwh;
    let soc_max = battery.soc_max_mwh;
    let soc_range = (soc_max - soc_min).max(EPS);
    let soc_level = ((soc - soc_min) / soc_range).clamp(0.0, 1.0);

    let energy_above_min = (soc - soc_min).max(0.0);
    let energy_below_max = (soc_max - soc).max(0.0);
    let discharge_step_energy = (battery.power_discharge_mw * battery.efficiency_discharge).max(EPS);
    let charge_step_energy = (battery.power_charge_mw * battery.efficiency_charge).max(EPS);
    let release_pressure = (energy_above_min / (steps_remaining as f64 * discharge_step_energy)).clamp(0.0, 2.0);
    let refill_pressure = (energy_below_max / (steps_remaining as f64 * charge_step_energy)).clamp(0.0, 2.0);
    let horizon_urgency = 1.0 / steps_remaining as f64;

    let future_prices: Vec<f64> = ((t + 1)..challenge.num_steps)
        .map(|s| challenge.market.day_ahead_prices[s][node])
        .collect();

    if future_prices.is_empty() {
        return if direction > 0.0 {
            if energy_above_min <= EPS {
                0.0
            } else {
                current_price * battery.efficiency_discharge
                    + 0.25 * current_price.abs() * release_pressure
            }
        } else if energy_below_max <= EPS {
            0.0
        } else {
            -(current_price / battery.efficiency_charge.max(EPS))
        };
    }

    let opportunity_steps = future_prices.len();

    if direction > 0.0 {
        if energy_above_min <= EPS {
            return 0.0;
        }

        let sell_slots = reachable_slot_count(
            energy_above_min.max(discharge_step_energy),
            discharge_step_energy,
            opportunity_steps,
        );
        let refill_energy = (energy_below_max + charge_step_energy).min(soc_range);
        let buy_slots = reachable_slot_count(
            refill_energy.max(charge_step_energy),
            charge_step_energy,
            opportunity_steps,
        );
        let (avg_low, avg_high, low_threshold, high_threshold) =
            scarcity_price_summary(&future_prices, buy_slots, sell_slots);

        let realizable_spread = (avg_high - avg_low).max(0.0);
        let immediate_value = current_price * battery.efficiency_discharge;
        let continuation_value = avg_high * battery.efficiency_discharge;
        let peak_bonus = (current_price - high_threshold).max(0.0) * battery.efficiency_discharge;
        let recycle_bonus =
            (immediate_value - low_threshold / battery.efficiency_charge.max(EPS)).max(0.0);
        let slot_need = sell_slots as f64 / opportunity_steps as f64;

        immediate_value - continuation_value
            + 0.28 * peak_bonus
            + 0.10 * recycle_bonus
            + realizable_spread * (0.40 * soc_level + 0.30 * release_pressure + 0.15 * slot_need)
            + 0.12 * immediate_value.abs() * release_pressure * horizon_urgency
    } else {
        if energy_below_max <= EPS {
            return 0.0;
        }

        let sell_energy = (energy_above_min + charge_step_energy).min(soc_range);
        let sell_slots = reachable_slot_count(
            sell_energy.max(discharge_step_energy),
            discharge_step_energy,
            opportunity_steps,
        );
        let buy_slots = reachable_slot_count(
            energy_below_max.max(charge_step_energy),
            charge_step_energy,
            opportunity_steps,
        );
        let (avg_low, avg_high, low_threshold, _) =
            scarcity_price_summary(&future_prices, buy_slots, sell_slots);

        let realizable_spread = (avg_high - avg_low).max(0.0);
        let immediate_cost = current_price / battery.efficiency_charge.max(EPS);
        let future_sale_value = avg_high * battery.efficiency_discharge;
        let cheap_now_bonus = (low_threshold - current_price).max(0.0) / battery.efficiency_charge.max(EPS);
        let cycle_margin =
            (future_sale_value - avg_low / battery.efficiency_charge.max(EPS)).max(0.0);
        let slot_need = sell_slots.max(buy_slots) as f64 / opportunity_steps as f64;

        future_sale_value - immediate_cost
            + 0.28 * cheap_now_bonus
            + 0.10 * cycle_margin
            + realizable_spread * (0.40 * (1.0 - soc_level) + 0.30 * refill_pressure + 0.15 * slot_need)
    }
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
        if violation > tig_challenges::energy_arbitrage::constants::EPS_FLOW * limit {
            let candidate = Violation { line: l, flow, amount: violation };
            match best {
                Some(ref current) if candidate.amount <= current.amount => {}
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

fn soften_most_violated_line(challenge: &Challenge, state: &State, violation: &Violation, action: &mut [f64], action_bounds: &[(f64, f64)]) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }
    let mut worsening: Vec<(usize, f64, f64)> = Vec::new();
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let ptdf_val = challenge.network.ptdf[line][battery.node];
        let contribution = ptdf_val * action[i];
        let signed_contribution = signed_direction * contribution;
        if signed_contribution > EPS {
            let relief = (signed_direction * ptdf_val).abs();
            let direction = if action[i] > 0.0 { 1.0 } else { -1.0 };
            let economic_value = directional_value(challenge, state, i, direction);
            let loss_per_relief = if relief > EPS {
                economic_value.max(0.0) / relief
            } else {
                f64::INFINITY
            };
            worsening.push((i, ptdf_val, loss_per_relief));
        }
    }
    if worsening.is_empty() {
        return false;
    }
    worsening.sort_unstable_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut remaining_violation = violation.amount;
    let mut changed = false;
    for (i, ptdf_val, _) in &worsening {
        let i = *i;
        let ptdf_abs = ptdf_val.abs();
        if ptdf_abs <= EPS {
            continue;
        }
        let sensitivity = signed_direction * ptdf_val;
        let current_contribution = sensitivity * action[i];
        if current_contribution <= EPS {
            continue;
        }
        let target_contribution = (current_contribution - remaining_violation).max(0.0);
        let new_action = if sensitivity.abs() > EPS {
            target_contribution / sensitivity
        } else {
            action[i]
        };
        let (min_b, max_b) = action_bounds[i];
        let clamped_action = new_action.clamp(min_b, max_b);
        let actual_reduction = sensitivity * (action[i] - clamped_action);
        if actual_reduction > EPS {
            remaining_violation -= actual_reduction;
            action[i] = clamped_action;
            changed = true;
        }
        if remaining_violation <= EPS {
            break;
        }
    }
    changed
}

fn same_sign_reduction_capacity(action: f64, bounds: (f64, f64)) -> f64 {
    if action > EPS {
        (action - bounds.0.max(0.0)).max(0.0)
    } else if action < -EPS {
        (bounds.1.min(0.0) - action).max(0.0)
    } else {
        0.0
    }
}

fn same_sign_increase_capacity(action: f64, bounds: (f64, f64), sign: f64) -> f64 {
    if sign > 0.0 {
        if action < -EPS {
            0.0
        } else {
            (bounds.1 - action.max(0.0)).max(0.0)
        }
    } else if action > EPS {
        0.0
    } else {
        (action.min(0.0) - bounds.0).max(0.0)
    }
}

fn substitute_most_violated_line(
    challenge: &Challenge,
    state: &State,
    violation: &Violation,
    flows: &[f64],
    action: &mut [f64],
    action_bounds: &[(f64, f64)],
) -> bool {
    if challenge.num_batteries < 2 {
        return false;
    }

    struct BestSwap {
        from: usize,
        to: usize,
        sign: f64,
        amount: f64,
        loss_per_relief: f64,
        total_relief: f64,
    }

    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    let mut best: Option<BestSwap> = None;

    for i in 0..challenge.num_batteries {
        let from_action = action[i];
        if from_action.abs() <= EPS {
            continue;
        }

        let sign = from_action.signum();
        let reducible = same_sign_reduction_capacity(from_action, action_bounds[i]);
        if reducible <= EPS {
            continue;
        }

        let from_node = challenge.batteries[i].node;
        let harm_from = signed_direction * challenge.network.ptdf[line][from_node] * sign;
        if harm_from <= EPS {
            continue;
        }

        let value_from = directional_value(challenge, state, i, sign);

        for j in 0..challenge.num_batteries {
            if i == j {
                continue;
            }

            let to_action = action[j];
            if to_action * sign < -EPS {
                continue;
            }

            let increase_cap = same_sign_increase_capacity(to_action, action_bounds[j], sign);
            if increase_cap <= EPS {
                continue;
            }

            let value_to = directional_value(challenge, state, j, sign);
            if to_action.abs() <= EPS && value_to <= EPS {
                continue;
            }

            let to_node = challenge.batteries[j].node;
            let harm_to = signed_direction * challenge.network.ptdf[line][to_node] * sign;
            let relief_per_unit = harm_from - harm_to;
            if relief_per_unit <= EPS {
                continue;
            }

            let mut amount = reducible
                .min(increase_cap)
                .min(violation.amount / relief_per_unit);
            if amount <= EPS {
                continue;
            }

            for l in 0..challenge.network.num_lines {
                let delta_flow = sign
                    * (challenge.network.ptdf[l][to_node] - challenge.network.ptdf[l][from_node]);
                if delta_flow > EPS {
                    amount = amount.min((challenge.network.flow_limits[l] - flows[l]) / delta_flow);
                } else if delta_flow < -EPS {
                    amount = amount.min((-challenge.network.flow_limits[l] - flows[l]) / delta_flow);
                }
                if amount <= EPS {
                    break;
                }
            }

            if amount <= EPS {
                continue;
            }

            let loss_per_relief = (value_from - value_to) / relief_per_unit;
            let total_relief = amount * relief_per_unit;

            let is_better = match &best {
                None => true,
                Some(best_swap) => {
                    loss_per_relief < best_swap.loss_per_relief - EPS
                        || ((loss_per_relief - best_swap.loss_per_relief).abs() <= EPS
                            && total_relief > best_swap.total_relief + EPS)
                }
            };

            if is_better {
                best = Some(BestSwap {
                    from: i,
                    to: j,
                    sign,
                    amount,
                    loss_per_relief,
                    total_relief,
                });
            }
        }
    }

    let Some(best_swap) = best else {
        return false;
    };

    let from_old = action[best_swap.from];
    let to_old = action[best_swap.to];

    let from_trial = (from_old - best_swap.sign * best_swap.amount)
        .clamp(action_bounds[best_swap.from].0, action_bounds[best_swap.from].1);
    let to_trial = (to_old + best_swap.sign * best_swap.amount)
        .clamp(action_bounds[best_swap.to].0, action_bounds[best_swap.to].1);

    let actual_reduce = ((from_old - from_trial) * best_swap.sign).max(0.0);
    let actual_increase = ((to_trial - to_old) * best_swap.sign).max(0.0);
    let actual_amount = actual_reduce.min(actual_increase);

    if actual_amount <= EPS {
        return false;
    }

    action[best_swap.from] = (from_old - best_swap.sign * actual_amount)
        .clamp(action_bounds[best_swap.from].0, action_bounds[best_swap.from].1);
    action[best_swap.to] = (to_old + best_swap.sign * actual_amount)
        .clamp(action_bounds[best_swap.to].0, action_bounds[best_swap.to].1);

    true
}

struct FallbackMode {
    action: Vec<f64>,
    delta_flows: Vec<f64>,
    value: f64,
    volume: f64,
}

fn estimated_action_value(challenge: &Challenge, state: &State, action: &[f64]) -> f64 {
    let mut value = 0.0;
    for i in 0..action.len() {
        let a = action[i];
        if a.abs() <= EPS {
            continue;
        }
        let direction = if a > 0.0 { 1.0 } else { -1.0 };
        value += a.abs() * directional_value(challenge, state, i, direction);
    }
    value
}

fn append_rank_split_modes(items: &[(usize, f64, f64)], out: &mut Vec<Vec<usize>>) {
    if items.is_empty() {
        return;
    }
    if items.len() == 1 {
        out.push(vec![items[0].0]);
        return;
    }

    let total_volume: f64 = items.iter().map(|item| item.2.max(EPS)).sum();
    let mut split = (items.len() / 2).max(1).min(items.len() - 1);
    let mut accum = 0.0;

    for (k, item) in items.iter().enumerate() {
        accum += item.2.max(EPS);
        if accum >= 0.5 * total_volume {
            split = (k + 1).min(items.len() - 1);
            break;
        }
    }

    let left: Vec<usize> = items[..split].iter().map(|item| item.0).collect();
    let right: Vec<usize> = items[split..].iter().map(|item| item.0).collect();

    if !left.is_empty() {
        out.push(left);
    }
    if !right.is_empty() {
        out.push(right);
    }
}

fn mode_scale_axis(center: f64, step: f64) -> Vec<f64> {
    let mut points = Vec::new();
    for offset in [-2.0f64, -1.0, 0.0, 1.0, 2.0] {
        let point = (center + offset * step).clamp(0.0, 1.0);
        if points
            .last()
            .map(|prev: &f64| (point - *prev).abs() > 1e-9)
            .unwrap_or(true)
        {
            points.push(point);
        }
    }
    if points.is_empty() {
        points.push(center.clamp(0.0, 1.0));
    }
    points
}

fn search_mode_scale_grid(
    idx: usize,
    grids: &[Vec<f64>],
    current_scales: &mut [f64],
    current_flows: &mut [f64],
    current_score: f64,
    current_volume: f64,
    modes: &[FallbackMode],
    limits: &[f64],
    best_scales: &mut Vec<f64>,
    best_score: &mut f64,
    best_volume: &mut f64,
) {
    if idx == modes.len() {
        let feasible = current_flows.iter().enumerate().all(|(l, &flow)| {
            flow.abs() <= limits[l] + 1e-9 * (1.0 + limits[l].abs())
        });

        if feasible
            && (current_score > *best_score + 1e-9
                || ((current_score - *best_score).abs() <= 1e-9
                    && current_volume > *best_volume + 1e-9))
        {
            *best_score = current_score;
            *best_volume = current_volume;
            *best_scales = current_scales.to_vec();
        }
        return;
    }

    for &scale in &grids[idx] {
        current_scales[idx] = scale;
        for l in 0..current_flows.len() {
            current_flows[l] += scale * modes[idx].delta_flows[l];
        }

        search_mode_scale_grid(
            idx + 1,
            grids,
            current_scales,
            current_flows,
            current_score + scale * modes[idx].value,
            current_volume + scale * modes[idx].volume,
            modes,
            limits,
            best_scales,
            best_score,
            best_volume,
        );

        for l in 0..current_flows.len() {
            current_flows[l] -= scale * modes[idx].delta_flows[l];
        }
    }
}

fn low_rank_mode_scaling_fallback(challenge: &Challenge, state: &State, base_action: &[f64]) -> Option<Vec<f64>> {
    if base_action.is_empty() {
        return Some(Vec::new());
    }

    let zero_action = vec![0.0; base_action.len()];
    let zero_flows = compute_flows(challenge, state, &zero_action);
    let line_shadows = if challenge.network.num_lines > 0 {
        live_congestion_shadows(challenge, state, base_action)
    } else {
        Vec::new()
    };

    let mut pos_items: Vec<(usize, f64, f64)> = Vec::new();
    let mut neg_items: Vec<(usize, f64, f64)> = Vec::new();

    for i in 0..challenge.num_batteries {
        let a = base_action[i];
        if a.abs() <= EPS {
            continue;
        }

        let direction = if a > 0.0 { 1.0 } else { -1.0 };
        let value = directional_value(challenge, state, i, direction);
        let raw_shadow = if line_shadows.is_empty() {
            0.0
        } else {
            directional_congestion_cost(challenge, i, direction, &line_shadows)
        };
        let priority =
            value / (1.0 + raw_shadow.max(0.0)) + 0.10 * value.abs() * (-raw_shadow).max(0.0);
        let item = (i, priority, a.abs());

        if a > 0.0 {
            pos_items.push(item);
        } else {
            neg_items.push(item);
        }
    }

    pos_items.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    neg_items.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut assignments: Vec<Vec<usize>> = Vec::new();
    append_rank_split_modes(&pos_items, &mut assignments);
    append_rank_split_modes(&neg_items, &mut assignments);

    let has_pos = assignments
        .iter()
        .any(|members| members.first().map(|&i| base_action[i] > EPS).unwrap_or(false));
    let has_neg = assignments
        .iter()
        .any(|members| members.first().map(|&i| base_action[i] < -EPS).unwrap_or(false));

    if assignments.len() == 2 && has_pos && has_neg {
        return None;
    }

    let mut modes: Vec<FallbackMode> = Vec::new();
    for members in assignments {
        let mut mode_action = vec![0.0; base_action.len()];
        let mut value = 0.0;
        let mut volume = 0.0;

        for &i in &members {
            let a = base_action[i];
            mode_action[i] = a;
            volume += a.abs();
            let direction = if a > 0.0 { 1.0 } else { -1.0 };
            value += a.abs() * directional_value(challenge, state, i, direction);
        }

        if volume <= EPS {
            continue;
        }

        let flows = compute_flows(challenge, state, &mode_action);
        let delta_flows = flows
            .iter()
            .enumerate()
            .map(|(l, &flow)| flow - zero_flows[l])
            .collect();

        modes.push(FallbackMode {
            action: mode_action,
            delta_flows,
            value,
            volume,
        });
    }

    if modes.len() <= 1 {
        return None;
    }

    let mut best_scales = vec![0.0; modes.len()];
    let mut best_score = f64::NEG_INFINITY;
    let mut best_volume = f64::NEG_INFINITY;

    let full_grids = vec![vec![0.0, 0.25, 0.5, 0.75, 1.0]; modes.len()];
    let mut current_scales = vec![0.0; modes.len()];
    let mut current_flows = zero_flows.clone();
    search_mode_scale_grid(
        0,
        &full_grids,
        &mut current_scales,
        &mut current_flows,
        0.0,
        0.0,
        &modes,
        &challenge.network.flow_limits,
        &mut best_scales,
        &mut best_score,
        &mut best_volume,
    );

    let mut refine_step = 0.125;
    for _ in 0..2 {
        let local_grids: Vec<Vec<f64>> = best_scales
            .iter()
            .map(|&scale| mode_scale_axis(scale, refine_step))
            .collect();
        let mut current_scales = vec![0.0; modes.len()];
        let mut current_flows = zero_flows.clone();
        search_mode_scale_grid(
            0,
            &local_grids,
            &mut current_scales,
            &mut current_flows,
            0.0,
            0.0,
            &modes,
            &challenge.network.flow_limits,
            &mut best_scales,
            &mut best_score,
            &mut best_volume,
        );
        refine_step *= 0.5;
    }

    if !best_score.is_finite() {
        return None;
    }

    let mut result = vec![0.0; base_action.len()];
    for (m, &scale) in best_scales.iter().enumerate() {
        if scale <= EPS {
            continue;
        }
        for i in 0..result.len() {
            result[i] += scale * modes[m].action[i];
        }
    }

    Some(result)
}

fn grouped_scaled_action(base_action: &[f64], pos_scale: f64, neg_scale: f64) -> Vec<f64> {
    base_action
        .iter()
        .map(|&a| {
            if a > EPS {
                pos_scale * a
            } else if a < -EPS {
                neg_scale * a
            } else {
                0.0
            }
        })
        .collect()
}

fn grouped_scaling_fallback(challenge: &Challenge, state: &State, base_action: &[f64]) -> Option<Vec<f64>> {
    if base_action.is_empty() {
        return Some(Vec::new());
    }

    let zero_action = vec![0.0; base_action.len()];
    let zero_flows = compute_flows(challenge, state, &zero_action);

    let pos_action: Vec<f64> = base_action.iter().map(|&a| a.max(0.0)).collect();
    let neg_action: Vec<f64> = base_action.iter().map(|&a| a.min(0.0)).collect();

    let pos_mag: f64 = pos_action.iter().sum();
    let neg_mag: f64 = neg_action.iter().map(|a| -a).sum();

    let pos_flows = if pos_mag > EPS {
        compute_flows(challenge, state, &pos_action)
    } else {
        zero_flows.clone()
    };
    let neg_flows = if neg_mag > EPS {
        compute_flows(challenge, state, &neg_action)
    } else {
        zero_flows.clone()
    };

    let mut pos_value = 0.0f64;
    let mut neg_value = 0.0f64;
    for i in 0..challenge.num_batteries {
        let a = base_action[i];
        if a > EPS {
            pos_value += a * directional_value(challenge, state, i, 1.0);
        } else if a < -EPS {
            neg_value += (-a) * directional_value(challenge, state, i, -1.0);
        }
    }
    if pos_mag > EPS {
        pos_value = pos_value.max(1e-9 * pos_mag);
    }
    if neg_mag > EPS {
        neg_value = neg_value.max(1e-9 * neg_mag);
    }

    let mut constraints: Vec<(f64, f64, f64)> = Vec::with_capacity(2 * challenge.network.num_lines + 4);
    constraints.push((1.0, 0.0, 1.0));
    constraints.push((0.0, 1.0, 1.0));
    constraints.push((-1.0, 0.0, 0.0));
    constraints.push((0.0, -1.0, 0.0));

    for l in 0..challenge.network.num_lines {
        let delta_pos = pos_flows[l] - zero_flows[l];
        let delta_neg = neg_flows[l] - zero_flows[l];
        let limit = challenge.network.flow_limits[l];
        let base_flow = zero_flows[l];

        constraints.push((delta_pos, delta_neg, limit - base_flow));
        constraints.push((-delta_pos, -delta_neg, limit + base_flow));
    }

    let mut candidates: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
    for i in 0..constraints.len() {
        let (a1, b1, c1) = constraints[i];
        for j in (i + 1)..constraints.len() {
            let (a2, b2, c2) = constraints[j];
            let det = a1 * b2 - a2 * b1;
            if det.abs() <= EPS {
                continue;
            }
            let x = (c1 * b2 - c2 * b1) / det;
            let y = (a1 * c2 - a2 * c1) / det;
            candidates.push((x, y));
        }
    }

    let mut best_scales = (0.0f64, 0.0f64);
    let mut best_score = -f64::INFINITY;
    let mut best_volume = -f64::INFINITY;

    for (raw_x, raw_y) in candidates {
        if raw_x < -1e-9 || raw_x > 1.0 + 1e-9 || raw_y < -1e-9 || raw_y > 1.0 + 1e-9 {
            continue;
        }

        let x = raw_x.clamp(0.0, 1.0);
        let y = raw_y.clamp(0.0, 1.0);

        let mut feasible = true;
        for &(a, b, c) in &constraints {
            if a * x + b * y > c + 1e-9 * (1.0 + c.abs()) {
                feasible = false;
                break;
            }
        }
        if !feasible {
            continue;
        }

        let volume = pos_mag * x + neg_mag * y;
        let score = pos_value * x + neg_value * y;

        if score > best_score + 1e-9
            || ((score - best_score).abs() <= 1e-9 && volume > best_volume + 1e-9)
        {
            best_scales = (x, y);
            best_score = score;
            best_volume = volume;
        }
    }

    Some(grouped_scaled_action(base_action, best_scales.0, best_scales.1))
}

fn enforce_flow_feasibility(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, &flows) else {
            action = post_feasibility_improvement(challenge, state, action);
            return Ok(action);
        };

        let swapped = substitute_most_violated_line(
            challenge,
            state,
            &violation,
            &flows,
            &mut action,
            &state.action_bounds,
        );
        if swapped {
            continue;
        }

        if !soften_most_violated_line(challenge, state, &violation, &mut action, &state.action_bounds) {
            break;
        }
    }
    if is_flow_feasible(challenge, state, &action) {
        action = post_feasibility_improvement(challenge, state, action);
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!("Grid infeasible even with zero battery actions"));
    }

    let base = action;

    let mut best_rescue: Option<Vec<f64>> = None;
    let mut best_rescue_score = f64::NEG_INFINITY;
    let mut best_rescue_volume = f64::NEG_INFINITY;

    if let Some(mode_scaled) = low_rank_mode_scaling_fallback(challenge, state, &base) {
        if is_flow_feasible(challenge, state, &mode_scaled) {
            let score = estimated_action_value(challenge, state, &mode_scaled);
            let volume = mode_scaled.iter().map(|a| a.abs()).sum::<f64>();
            best_rescue_score = score;
            best_rescue_volume = volume;
            best_rescue = Some(mode_scaled);
        }
    }

    if let Some(grouped) = grouped_scaling_fallback(challenge, state, &base) {
        if is_flow_feasible(challenge, state, &grouped) {
            let score = estimated_action_value(challenge, state, &grouped);
            let volume = grouped.iter().map(|a| a.abs()).sum::<f64>();
            let better = best_rescue.is_none()
                || score > best_rescue_score + 1e-9
                || ((score - best_rescue_score).abs() <= 1e-9
                    && volume > best_rescue_volume + 1e-9);

            if better {
                best_rescue = Some(grouped);
            }
        }
    }

    if let Some(mut result) = best_rescue {
        result = post_feasibility_improvement(challenge, state, result);
        return Ok(result);
    }

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
    let mut result: Vec<f64> = base.into_iter().map(|u| low * u).collect();
    result = post_feasibility_improvement(challenge, state, result);
    Ok(result)
}

fn post_feasibility_improvement(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Vec<f64> {
    let num_batteries = challenge.num_batteries;
    let num_lines = challenge.network.num_lines;

    let mut flows = compute_flows(challenge, state, &action);

    const MAX_PASSES: usize = 4;

    for _pass in 0..MAX_PASSES {
        let mut battery_order: Vec<(usize, f64)> = (0..num_batteries)
            .filter_map(|i| {
                let a = action[i];
                if a.abs() <= EPS {
                    return None;
                }
                let direction = if a > 0.0 { 1.0 } else { -1.0 };
                let marginal = directional_value(challenge, state, i, direction);
                if marginal > EPS {
                    Some((i, marginal))
                } else {
                    None
                }
            })
            .collect();

        battery_order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut any_improvement = false;

        for (i, _) in &battery_order {
            let i = *i;
            let battery = &challenge.batteries[i];
            let node = battery.node;
            let (min_bound, max_bound) = state.action_bounds[i];
            let current = action[i];

            if current.abs() <= EPS {
                continue;
            }

            let (target_bound, direction) = if current > 0.0 {
                (max_bound, 1.0f64)
            } else {
                (min_bound, -1.0f64)
            };

            let max_increase = direction * (target_bound - current);
            if max_increase <= EPS {
                continue;
            }

            let mut delta_max = max_increase;
            for l in 0..num_lines {
                let ptdf_val = challenge.network.ptdf[l][node];
                let flow_sensitivity = ptdf_val * direction;
                if flow_sensitivity.abs() <= EPS {
                    continue;
                }
                let limit = challenge.network.flow_limits[l];
                let current_flow = flows[l];
                let headroom = if flow_sensitivity > 0.0 {
                    (limit - current_flow) / flow_sensitivity
                } else {
                    (-limit - current_flow) / flow_sensitivity
                };
                if headroom < delta_max {
                    delta_max = headroom;
                }
            }

            if delta_max <= EPS {
                continue;
            }

            let new_action = current + direction * delta_max;
            let clamped = new_action.clamp(min_bound, max_bound);
            let actual_delta = clamped - current;
            if actual_delta.abs() <= EPS {
                continue;
            }

            for l in 0..num_lines {
                let ptdf_val = challenge.network.ptdf[l][node];
                flows[l] += ptdf_val * actual_delta;
            }
            action[i] = clamped;
            any_improvement = true;
        }

        if !any_improvement {
            break;
        }
    }

    action
}

fn adaptive_price_summary(raw_prices: &[f64]) -> Vec<f64> {
    if raw_prices.is_empty() {
        return Vec::new();
    }
    if raw_prices.len() == 1 {
        return vec![raw_prices[0]];
    }

    let n = raw_prices.len();
    let mut min_price = raw_prices[0];
    let mut max_price = raw_prices[0];
    let mut sum_price = 0.0;
    let mut abs_sum_price = 0.0;
    let mut diff_abs_sum = 0.0;
    let mut sign_changes = 0usize;
    let mut prev_diff = 0.0;

    for k in 0..n {
        let price = raw_prices[k];
        min_price = min_price.min(price);
        max_price = max_price.max(price);
        sum_price += price;
        abs_sum_price += price.abs();

        if k > 0 {
            let diff = price - raw_prices[k - 1];
            diff_abs_sum += diff.abs();
            if k > 1 && diff * prev_diff < 0.0 {
                sign_changes += 1;
            }
            prev_diff = diff;
        }
    }

    let mean_price = sum_price / n as f64;
    let price_scale = abs_sum_price / n as f64 + 0.25 * mean_price.abs() + 1.0;
    let spread = (max_price - min_price).max(0.0);
    let avg_step_move = diff_abs_sum / (n - 1) as f64;
    let volatility =
        (avg_step_move / (0.35 * spread + 0.20 * price_scale + EPS)).clamp(0.0, 1.0);
    let mean_reversion = if n > 2 {
        sign_changes as f64 / (n - 2) as f64
    } else {
        0.0
    };
    let flatness =
        (1.0 - (spread / (0.30 * price_scale + spread + EPS)).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    let mut sorted = raw_prices.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q10 = sorted[((n - 1) * 10) / 100];
    let q20 = sorted[((n - 1) * 20) / 100];
    let q50 = sorted[(n - 1) / 2];
    let q80 = sorted[((n - 1) * 80) / 100];
    let q90 = sorted[((n - 1) * 90) / 100];

    let upper_tail_span = (max_price - q90).max(0.0);
    let lower_tail_span = (q10 - min_price).max(0.0);
    let tail_span = ((upper_tail_span + lower_tail_span) / (spread + EPS)).clamp(0.0, 1.0);

    let mut tail_runs = 0usize;
    let mut prev_tail = false;
    for &price in raw_prices {
        let is_tail = price >= q80 || price <= q20;
        if is_tail && !prev_tail {
            tail_runs += 1;
        }
        prev_tail = is_tail;
    }

    let spike_scarcity = (tail_span * (2.5 / (2.5 + tail_runs as f64))).clamp(0.0, 1.0);
    let base_alpha = (0.14
        + 0.26 * volatility
        + 0.14 * spike_scarcity
        - 0.12 * mean_reversion
        - 0.10 * flatness)
        .clamp(0.08, 0.72);

    let innovation_scale = 0.30 * spread + 0.25 * price_scale + EPS;
    let jump_scale = avg_step_move + 0.18 * spread + 0.06 * price_scale + EPS;

    let mut summarized = Vec::with_capacity(n);
    let mut ema = raw_prices[0];
    let mut spike_channel = raw_prices[0];
    summarized.push(raw_prices[0]);

    for k in 1..n {
        let price = raw_prices[k];

        let start = k.saturating_sub(2);
        let mut window = raw_prices[start..=k].to_vec();
        window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = window[window.len() / 2];

        let innovation = ((price - ema).abs() / innovation_scale).clamp(0.0, 1.0);
        let jump = ((price - raw_prices[k - 1]).abs() / jump_scale).clamp(0.0, 1.0);
        let upper_tail =
            ((price - q80).max(0.0) / (max_price - q80 + EPS)).clamp(0.0, 1.0);
        let lower_tail =
            ((q20 - price).max(0.0) / (q20 - min_price + EPS)).clamp(0.0, 1.0);
        let shoulder_tail = if price >= q90 {
            0.5 + 0.5 * ((price - q90).max(0.0) / (max_price - q90 + EPS)).clamp(0.0, 1.0)
        } else if price <= q10 {
            0.5 + 0.5 * ((q10 - price).max(0.0) / (q10 - min_price + EPS)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let tailness = upper_tail.max(lower_tail).max(shoulder_tail).clamp(0.0, 1.0);
        let deviation =
            ((price - median).abs() / (0.22 * spread + 0.08 * price_scale + EPS)).clamp(0.0, 1.0);

        let ema_alpha = (base_alpha
            + 0.16 * innovation
            + 0.10 * tailness * spike_scarcity)
            .clamp(0.08, 0.88);
        ema = ema_alpha * price + (1.0 - ema_alpha) * ema;

        let spike_gate = (spike_scarcity * tailness * (0.45 + 0.30 * innovation + 0.25 * jump)
            + 0.20 * spike_scarcity * deviation * jump)
            .clamp(0.0, 1.0);
        let spike_target = if tailness > 0.0 { price } else { median };
        let spike_alpha = (0.18 + 0.74 * spike_gate).clamp(0.12, 0.96);
        spike_channel = spike_alpha * spike_target + (1.0 - spike_alpha) * spike_channel;

        let median_weight =
            (0.20 + 0.28 * mean_reversion + 0.18 * flatness - 0.16 * spike_gate).clamp(0.08, 0.56);
        let smooth_channel = (1.0 - median_weight) * ema + median_weight * median;

        let spike_weight =
            (0.08 + 0.70 * spike_gate - 0.12 * flatness * (1.0 - tailness)).clamp(0.0, 0.78);
        let mut value = (1.0 - spike_weight) * smooth_channel + spike_weight * spike_channel;

        if flatness > 0.65 && tailness < 0.15 && innovation < 0.30 {
            let damp = (0.20 + 0.25 * flatness).clamp(0.0, 0.45);
            value = (1.0 - damp) * value + damp * (0.5 * median + 0.5 * ema);
        }

        if spread <= 0.08 * price_scale && jump > 0.40 && tailness < 0.10 {
            value = 0.65 * value + 0.35 * q50;
        }

        summarized.push(value);
    }

    if spread <= 0.12 * price_scale {
        let flatten = (1.0 - spread / (0.12 * price_scale + EPS)).clamp(0.0, 1.0);
        let blend = 0.40 * flatten;
        for price in &mut summarized {
            *price = (1.0 - blend) * *price + blend * mean_price;
        }
    }

    summarized
}

fn backward_precharge_action(
    challenge: &Challenge,
    state: &State,
    battery_idx: usize,
    smoothed_prices: &[f64],
    step_action: &[i8],
    soc: f64,
    soc_min: f64,
    soc_max: f64,
    max_charge_energy: f64,
    max_discharge_energy: f64,
) -> f64 {
    let battery = &challenge.batteries[battery_idx];
    let t = state.time_step;
    if t + 1 >= challenge.num_steps {
        return 0.0;
    }

    let current_price = smoothed_prices[0];
    let rt_eff = battery.efficiency_charge * battery.efficiency_discharge;
    if rt_eff <= EPS || max_charge_energy <= EPS || max_discharge_energy <= EPS {
        return 0.0;
    }

    let energy_above_min = (soc - soc_min).max(0.0);
    let storable_now = (soc_max - soc).min(max_charge_energy).max(0.0);
    if storable_now <= EPS {
        return 0.0;
    }

    let future_prices = &smoothed_prices[1..];
    if future_prices.is_empty() {
        return 0.0;
    }

    let mut sorted_future = future_prices.to_vec();
    sorted_future.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let low_ref = sorted_future[0];
    let high_ref = sorted_future[sorted_future.len() - 1];
    let spread = (high_ref - low_ref).max(0.0);
    let current_cycle_margin =
        high_ref * battery.efficiency_discharge - current_price / battery.efficiency_charge.max(EPS);
    let signal_scale = current_price.abs() + high_ref.abs() + low_ref.abs() + 1.0;
    if current_cycle_margin <= 0.02 * signal_scale {
        return 0.0;
    }

    let cheap_idle_cutoff = current_price + 0.10 * spread / rt_eff.max(EPS);

    let mut prefix_min = vec![0.0; smoothed_prices.len()];
    prefix_min[0] = smoothed_prices[0];
    for k in 1..smoothed_prices.len() {
        prefix_min[k] = prefix_min[k - 1].min(smoothed_prices[k]);
    }

    let mut required_energy_now = 0.0;
    let mut premium_demand_total = 0.0;
    let mut counted_discharge_slots = 0usize;
    let max_inventory = (soc_max - soc_min).max(EPS);

    for s in (t + 1..challenge.num_steps).rev() {
        let rel = s - t;
        let price = smoothed_prices[rel];

        if step_action[s] == 1 {
            let buy_ref = prefix_min[rel - 1];
            let premium_margin = (price * rt_eff - buy_ref).max(0.0);
            let margin_scale = price.abs() + buy_ref.abs() + 1.0;
            let normalized_margin = (premium_margin / margin_scale).clamp(0.0, 1.0);
            let highness = if spread > EPS {
                ((price - (low_ref + 0.55 * spread)) / (0.45 * spread + EPS)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let confidence = (0.60 * normalized_margin + 0.40 * highness).clamp(0.0, 1.0);

            if confidence > 0.05 {
                let required = max_discharge_energy * (0.25 + 0.75 * confidence);
                required_energy_now = (required_energy_now + required).min(max_inventory);
                premium_demand_total += required;
                counted_discharge_slots += 1;
            }
        }

        let scheduled_charge = step_action[s] == -1;
        let cheap_idle = step_action[s] == 0 && price <= cheap_idle_cutoff;
        if scheduled_charge || cheap_idle {
            let supply_weight = if scheduled_charge {
                1.0
            } else if spread > EPS {
                ((cheap_idle_cutoff - price) / (cheap_idle_cutoff - low_ref + EPS)).clamp(0.20, 1.0)
            } else if price <= current_price {
                0.5
            } else {
                0.0
            };

            required_energy_now = (required_energy_now - supply_weight * max_charge_energy).max(0.0);
        }
    }

    if counted_discharge_slots == 0 || premium_demand_total <= 0.2 * max_discharge_energy {
        return 0.0;
    }

    let energy_deficit = required_energy_now - energy_above_min;
    if energy_deficit <= 0.10 * max_charge_energy {
        return 0.0;
    }

    let stored_energy_to_add = energy_deficit.min(storable_now);
    if stored_energy_to_add <= EPS {
        return 0.0;
    }

    let charge_power = stored_energy_to_add / battery.efficiency_charge.max(EPS);
    -(charge_power.min(battery.power_charge_mw))
}

fn price_rank_action(
    challenge: &Challenge,
    state: &State,
    battery_idx: usize,
    _node: usize,
    smoothed_prices: &[f64],
) -> f64 {
    fn terminal_inventory_value(
        future_prices: &[f64],
        charge_step_energy: f64,
        discharge_step_energy: f64,
        energy_above_min: f64,
        empty_space_below_max: f64,
        eff_c: f64,
    ) -> f64 {
        if future_prices.is_empty() {
            return 0.0;
        }

        let n = future_prices.len().max(1);
        let discharge_step_energy = discharge_step_energy.max(EPS);
        let charge_step_energy = charge_step_energy.max(EPS);

        let mut sell_slots: Vec<f64> = future_prices
            .iter()
            .enumerate()
            .map(|(k, &p)| {
                let rel = (k + 1) as f64 / n as f64;
                let decay = 1.0 - 0.18 * rel;
                (p * decay).max(0.0)
            })
            .collect();
        sell_slots.sort_unstable_by(|a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut energy_value = 0.0;
        let mut remaining_energy = energy_above_min.max(0.0);
        let mut reserved_sell_slots = 0usize;
        for &price in &sell_slots {
            if price <= EPS || remaining_energy <= EPS {
                break;
            }
            let take = remaining_energy.min(discharge_step_energy);
            energy_value += price * take;
            remaining_energy -= take;
            reserved_sell_slots += 1;
        }

        if empty_space_below_max <= EPS
            || future_prices.len() <= 1
            || reserved_sell_slots >= sell_slots.len()
        {
            return energy_value;
        }

        let sell_reference = sell_slots[reserved_sell_slots];
        let mut suffix_best_sell: Vec<f64> = vec![0.0_f64; future_prices.len() + 1];
        for k in (0..future_prices.len()).rev() {
            let rel = (k + 1) as f64 / n as f64;
            let decayed_sell = (future_prices[k] * (1.0 - 0.12 * rel)).max(0.0_f64);
            suffix_best_sell[k] = suffix_best_sell[k + 1].max(decayed_sell);
        }

        let mut room_margins: Vec<f64> = Vec::new();
        for k in 0..future_prices.len() - 1 {
            let later_sell = suffix_best_sell[k + 1];
            if later_sell <= EPS {
                continue;
            }

            let rel = (k + 1) as f64 / n as f64;
            let effective_sell = 0.55 * later_sell + 0.45 * later_sell.min(sell_reference);
            let buy_cost = future_prices[k] * (1.0 + 0.06 * rel) / eff_c.max(EPS);
            let margin = effective_sell - buy_cost;
            if margin > EPS {
                room_margins.push(margin);
            }
        }

        room_margins.sort_unstable_by(|a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut room_value = 0.0;
        let mut remaining_room = empty_space_below_max.max(0.0);
        for margin in room_margins {
            if margin <= EPS || remaining_room <= EPS {
                break;
            }
            let take = remaining_room.min(charge_step_energy);
            room_value += margin * take;
            remaining_room -= take;
        }

        let remaining_sell_share =
            (sell_slots.len() - reserved_sell_slots) as f64 / sell_slots.len() as f64;
        let room_weight = (0.55 + 0.30 * remaining_sell_share).clamp(0.55, 0.85);

        energy_value + room_weight * room_value
    }

    let battery = &challenge.batteries[battery_idx];
    let t = state.time_step;
    let num_steps = challenge.num_steps;
    let steps_remaining = num_steps - t;

    if steps_remaining == 0 || smoothed_prices.is_empty() {
        return 0.0;
    }

    let soc = state.socs[battery_idx];
    let soc_min = battery.soc_min_mwh;
    let soc_max = battery.soc_max_mwh;
    let soc_range = (soc_max - soc_min).max(EPS);
    let soc_level = (soc - soc_min) / soc_range;

    let eff_c = battery.efficiency_charge.max(EPS);
    let eff_d = battery.efficiency_discharge.max(EPS);
    let rt_eff = eff_c * eff_d;
    let max_charge_energy = battery.power_charge_mw * eff_c;
    let max_discharge_energy = battery.power_discharge_mw * eff_d;

    let mut price_steps: Vec<(f64, usize)> = (t..num_steps)
        .map(|s| (smoothed_prices[s - t], s))
        .collect();
    let n = price_steps.len();

    if n == 0 {
        return 0.0;
    }

    let avg_price = smoothed_prices.iter().copied().sum::<f64>() / n as f64;
    let price_scale = smoothed_prices.iter().map(|p| p.abs()).sum::<f64>() / n as f64 + 1.0;

    price_steps.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let min_price = price_steps[0].0;
    let max_price = price_steps[n - 1].0;
    let spread = (max_price - min_price).max(0.0);
    let cycle_margin = max_price * eff_d - min_price / eff_c;
    if max_price <= min_price / rt_eff || cycle_margin <= 0.02 * price_scale {
        return 0.0;
    }

    let weak_signal = spread <= 0.10 * price_scale || cycle_margin <= 0.06 * price_scale;
    if weak_signal
        && soc_level > 0.12
        && soc_level < 0.88
        && (smoothed_prices[0] - avg_price).abs() <= 0.20 * spread + 0.02 * price_scale
    {
        return 0.0;
    }

    let mpc_window = steps_remaining.min(5);
    let local_slice = &smoothed_prices[..mpc_window];
    let mut local_min = local_slice[0];
    let mut local_max = local_slice[0];
    for &p in local_slice.iter().skip(1) {
        local_min = local_min.min(p);
        local_max = local_max.max(p);
    }
    let local_spread = (local_max - local_min).max(0.0);
    let extreme_soc = soc_level < 0.10 || soc_level > 0.90;
    let use_mpc = steps_remaining <= 9
        || local_spread > 0.50 * spread
        || (extreme_soc && steps_remaining <= 12);

    if use_mpc {
        let total_sequences = 3usize.pow(mpc_window as u32);
        let mut best_value = f64::NEG_INFINITY;
        let mut best_action: f64 = 0.0_f64;
        let mut best_zero_first = f64::NEG_INFINITY;

        for code in 0..total_sequences {
            let mut seq = code;
            let mut sim_soc = soc;
            let mut value = 0.0;
            let mut first_action: f64 = 0.0_f64;

            for k in 0..mpc_window {
                let choice = seq % 3;
                seq /= 3;
                let price = smoothed_prices[k];

                let discharge_cap = if k == 0 {
                    state.action_bounds[battery_idx].1.max(0.0)
                } else {
                    battery.power_discharge_mw
                };
                let charge_cap = if k == 0 {
                    (-state.action_bounds[battery_idx].0).max(0.0)
                } else {
                    battery.power_charge_mw
                };

                let action_k = match choice {
                    0 => {
                        let cap = charge_cap.min((soc_max - sim_soc).max(0.0) / eff_c);
                        if cap > EPS { -cap } else { 0.0 }
                    }
                    1 => 0.0,
                    _ => {
                        let cap = discharge_cap.min((sim_soc - soc_min).max(0.0) / eff_d);
                        if cap > EPS { cap } else { 0.0 }
                    }
                };

                if k == 0 {
                    first_action = action_k;
                }

                if action_k > 0.0 {
                    value += price * eff_d * action_k;
                    sim_soc = (sim_soc - action_k * eff_d).max(soc_min);
                } else if action_k < 0.0 {
                    value += price * action_k / eff_c;
                    sim_soc = (sim_soc - action_k * eff_c).min(soc_max);
                }
            }

            value += terminal_inventory_value(
                &smoothed_prices[mpc_window..],
                max_charge_energy,
                max_discharge_energy,
                (sim_soc - soc_min).max(0.0),
                (soc_max - sim_soc).max(0.0),
                eff_c,
            );

            if first_action.abs() <= EPS && value > best_zero_first {
                best_zero_first = value;
            }
            if value > best_value + 1e-9
                || ((value - best_value).abs() <= 1e-9
                    && first_action.abs() > best_action.abs() + 1e-9)
            {
                best_value = value;
                best_action = first_action;
            }
        }

        let gap = best_value - best_zero_first;
        let step_scale = max_charge_energy.max(max_discharge_energy).max(EPS);
        let confidence_scale = if best_action.abs() <= EPS || !best_zero_first.is_finite() {
            1.0
        } else {
            (gap / (0.05 * price_scale * step_scale + EPS)).clamp(0.25, 1.0)
        };

        let mpc_action = if gap <= 0.0 && !extreme_soc {
            0.0
        } else if weak_signal {
            best_action * (0.85 * confidence_scale)
        } else {
            best_action * confidence_scale
        };

        let action_scale_ref = battery.power_charge_mw.max(battery.power_discharge_mw).max(EPS);
        if steps_remaining <= 6
            || mpc_action.abs() > 0.35 * action_scale_ref
            || local_spread > 0.18 * price_scale
            || extreme_soc
        {
            return mpc_action;
        }
    }

    let mut step_action = vec![0i8; num_steps];

    let mut charge_ptr = 0usize;
    let mut discharge_ptr = n - 1;

    loop {
        while charge_ptr < discharge_ptr && step_action[price_steps[charge_ptr].1] != 0 {
            charge_ptr += 1;
        }
        while discharge_ptr > charge_ptr && step_action[price_steps[discharge_ptr].1] != 0 {
            if discharge_ptr == 0 { break; }
            discharge_ptr -= 1;
        }
        if charge_ptr >= discharge_ptr {
            break;
        }
        let charge_price = price_steps[charge_ptr].0;
        let discharge_price = price_steps[discharge_ptr].0;
        if discharge_price <= charge_price / rt_eff {
            break;
        }
        step_action[price_steps[charge_ptr].1] = -1;
        step_action[price_steps[discharge_ptr].1] = 1;
        charge_ptr += 1;
        if discharge_ptr == 0 { break; }
        discharge_ptr -= 1;
    }

    let mut fwd_soc = soc;
    for s in t..num_steps {
        match step_action[s] {
            -1 => {
                let added = max_charge_energy.min(soc_max - fwd_soc);
                if added < EPS {
                    step_action[s] = 0;
                } else {
                    fwd_soc = (fwd_soc + added).min(soc_max);
                }
            }
            1 => {
                let removed = max_discharge_energy.min(fwd_soc - soc_min);
                if removed < EPS {
                    step_action[s] = 0;
                } else {
                    fwd_soc = (fwd_soc - removed).max(soc_min);
                }
            }
            _ => {}
        }
    }

    match step_action[t] {
        -1 => {
            if soc_level < 0.99 {
                -battery.power_charge_mw
            } else {
                0.0
            }
        }
        1 => {
            if soc_level > 0.01 {
                battery.power_discharge_mw
            } else {
                0.0
            }
        }
        _ => backward_precharge_action(
            challenge,
            state,
            battery_idx,
            smoothed_prices,
            &step_action,
            soc,
            soc_min,
            soc_max,
            max_charge_energy,
            max_discharge_energy,
        ),
    }
}

fn live_congestion_shadows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let flows = compute_flows(challenge, state, action);
    if challenge.network.num_lines == 0 || challenge.num_batteries == 0 {
        return vec![0.0; challenge.network.num_lines];
    }

    #[derive(Clone, Copy)]
    struct DualCandidate {
        node: usize,
        direction: f64,
        cap: f64,
        value: f64,
        priority: f64,
    }

    let heuristic_shadows = || -> Vec<f64> {
        let mut shadows = vec![0.0; challenge.network.num_lines];

        struct ShadowCandidate {
            node: usize,
            direction: f64,
            weight: f64,
        }

        let mut candidates: Vec<ShadowCandidate> =
            Vec::with_capacity(challenge.num_batteries * 2);
        let mut weight_scale = 1.0f64;

        for i in 0..challenge.num_batteries {
            let battery = &challenge.batteries[i];

            for &direction in &[1.0f64, -1.0f64] {
                let bound_cap =
                    same_sign_increase_capacity(action[i], state.action_bounds[i], direction);
                if bound_cap <= EPS {
                    continue;
                }

                let step_cap = if direction > 0.0 {
                    battery.power_discharge_mw.max(EPS)
                } else {
                    battery.power_charge_mw.max(EPS)
                };
                let cap = bound_cap.min(step_cap);
                if cap <= EPS {
                    continue;
                }

                let marginal_value =
                    directional_value(challenge, state, i, direction).max(0.0);
                if marginal_value <= EPS {
                    continue;
                }

                let activity_bias = if action[i] * direction > EPS { 1.0 } else { 0.78 };
                let weight = marginal_value * cap * activity_bias;
                if weight > EPS {
                    candidates.push(ShadowCandidate {
                        node: battery.node,
                        direction,
                        weight,
                    });
                    weight_scale += weight;
                }
            }
        }

        if candidates.is_empty() {
            for l in 0..challenge.network.num_lines {
                let flow = flows[l];
                let limit = challenge.network.flow_limits[l].abs().max(EPS);
                let utilization = flow.abs() / limit;
                if utilization <= 0.72 {
                    continue;
                }

                let overload = (utilization - 1.0).max(0.0);
                let near_bind = ((utilization - 0.72) / 0.28).clamp(0.0, 1.0);
                let headroom_frac =
                    ((limit - flow.abs()).max(0.0) / limit).clamp(0.0, 1.0);
                let scarcity = if overload > EPS {
                    1.0 + overload / (0.12 + overload)
                } else {
                    ((0.18 / (headroom_frac + 0.18)) - 0.5).max(0.0)
                };

                let magnitude = if overload > EPS {
                    (0.75 + 1.90 * overload / (0.18 + overload)).min(2.75)
                } else {
                    0.12 * near_bind * near_bind + 0.24 * scarcity * near_bind
                };

                shadows[l] = flow.signum() * magnitude;
            }
            return shadows;
        }

        let max_candidates = (challenge.num_batteries.min(8).max(4)) * 2;
        if candidates.len() > max_candidates {
            candidates.sort_unstable_by(|a, b| {
                b.weight
                    .partial_cmp(&a.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(max_candidates);
            weight_scale = 1.0 + candidates.iter().map(|c| c.weight).sum::<f64>();
        }

        let top_per_line = candidates.len().min(6).max(1);

        for l in 0..challenge.network.num_lines {
            let flow = flows[l];
            let limit = challenge.network.flow_limits[l].abs().max(EPS);
            let utilization = flow.abs() / limit;
            let overload = (utilization - 1.0).max(0.0);
            let near_bind = ((utilization - 0.72) / 0.28).clamp(0.0, 1.0);
            let headroom_frac = ((limit - flow.abs()).max(0.0) / limit).clamp(0.0, 1.0);
            let scarcity = if overload > EPS {
                1.0 + overload / (0.12 + overload)
            } else {
                ((0.18 / (headroom_frac + 0.18)) - 0.5).max(0.0)
            };

            let stabilizer = if overload > EPS {
                (0.75 + 1.90 * overload / (0.18 + overload)).min(2.75)
            } else {
                0.12 * near_bind * near_bind + 0.24 * scarcity * near_bind
            };

            let pos_headroom = (limit - flow).max(0.0);
            let neg_headroom = (limit + flow).max(0.0);

            let mut pos_terms: Vec<f64> = Vec::new();
            let mut neg_terms: Vec<f64> = Vec::new();

            for candidate in &candidates {
                let sensitivity =
                    challenge.network.ptdf[l][candidate.node] * candidate.direction;
                let abs_sensitivity = sensitivity.abs();
                if abs_sensitivity <= EPS {
                    continue;
                }

                let contribution = candidate.weight * abs_sensitivity;
                if sensitivity > EPS {
                    pos_terms.push(contribution);
                } else if sensitivity < -EPS {
                    neg_terms.push(contribution);
                }
            }

            pos_terms.sort_unstable_by(|a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            neg_terms.sort_unstable_by(|a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });

            let pos_opp: f64 = pos_terms.iter().take(top_per_line).sum();
            let neg_opp: f64 = neg_terms.iter().take(top_per_line).sum();

            let shadow_sign = if flow.abs() > 0.08 * limit {
                flow.signum()
            } else if pos_opp > 1.20 * neg_opp && pos_opp > 0.03 * weight_scale {
                1.0
            } else if neg_opp > 1.20 * pos_opp && neg_opp > 0.03 * weight_scale {
                -1.0
            } else {
                0.0
            };

            if shadow_sign.abs() <= EPS {
                shadows[l] = 0.0;
                continue;
            }

            let directional_headroom = if shadow_sign > 0.0 {
                pos_headroom
            } else {
                neg_headroom
            };
            let same_opp = if shadow_sign > 0.0 { pos_opp } else { neg_opp };
            let counter_opp = if shadow_sign > 0.0 { neg_opp } else { pos_opp };

            let directional_headroom_frac = (directional_headroom / limit).clamp(0.0, 2.0);
            let directional_scarcity = if directional_headroom <= EPS {
                1.0 + overload / (0.10 + overload)
            } else {
                ((0.72 - directional_headroom_frac) / 0.72).clamp(0.0, 1.0)
            };

            let opportunity_signal = (same_opp
                / (0.30 * weight_scale + same_opp + 0.40 * counter_opp + EPS))
                .clamp(0.0, 1.0);
            let opportunity_balance = ((same_opp - 0.35 * counter_opp).max(0.0)
                / (0.15 * weight_scale + same_opp + counter_opp + EPS))
                .clamp(0.0, 1.0);

            let opportunity_mag = directional_scarcity
                * (0.18 + 1.25 * opportunity_signal + 0.45 * opportunity_balance);

            let magnitude = if shadow_sign == flow.signum() || flow.abs() <= 0.08 * limit {
                (0.55 * stabilizer + opportunity_mag).clamp(0.0, 3.25)
            } else {
                (0.25 * stabilizer + opportunity_mag).clamp(0.0, 3.25)
            };

            shadows[l] = shadow_sign * magnitude;
        }

        shadows
    };

    let mut candidates: Vec<DualCandidate> =
        Vec::with_capacity(challenge.num_batteries * 2);
    for i in 0..challenge.num_batteries {
        let battery = &challenge.batteries[i];
        for &direction in &[1.0f64, -1.0f64] {
            let bound_cap =
                same_sign_increase_capacity(action[i], state.action_bounds[i], direction);
            if bound_cap <= EPS {
                continue;
            }

            let step_cap = if direction > 0.0 {
                battery.power_discharge_mw.max(EPS)
            } else {
                battery.power_charge_mw.max(EPS)
            };
            let cap = bound_cap.min(step_cap);
            if cap <= EPS {
                continue;
            }

            let raw_value = directional_value(challenge, state, i, direction).max(0.0);
            if raw_value <= EPS {
                continue;
            }

            let activity_bias = if action[i] * direction > EPS { 1.0 } else { 0.82 };
            let value = raw_value * activity_bias;
            let priority = value * cap;
            if priority > EPS {
                candidates.push(DualCandidate {
                    node: battery.node,
                    direction,
                    cap,
                    value,
                    priority,
                });
            }
        }
    }

    if candidates.is_empty() {
        return heuristic_shadows();
    }

    let max_candidates = (challenge.num_batteries.min(10).max(5)) * 2;
    if candidates.len() > max_candidates {
        candidates.sort_unstable_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(max_candidates);
    }

    let value_scale = candidates.iter().map(|c| c.value).sum::<f64>()
        / candidates.len().max(1) as f64
        + 1.0;

    let mut lambda_pos = vec![0.0f64; challenge.network.num_lines];
    let mut lambda_neg = vec![0.0f64; challenge.network.num_lines];

    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        let pos_headroom = ((limit - flows[l]).max(0.0) / limit).clamp(0.0, 2.0);
        let neg_headroom = ((limit + flows[l]).max(0.0) / limit).clamp(0.0, 2.0);
        lambda_pos[l] = ((0.20 - pos_headroom).max(0.0) / 0.20).clamp(0.0, 0.25);
        lambda_neg[l] = ((0.20 - neg_headroom).max(0.0) / 0.20).clamp(0.0, 0.25);
    }

    const DUAL_ITERS: usize = 6;
    let active_take = candidates
        .len()
        .min((challenge.num_batteries.min(6).max(3)) * 2);
    let mut prev_change = f64::INFINITY;
    let mut stable_steps = 0usize;
    let mut saw_positive = false;

    for _ in 0..DUAL_ITERS {
        let mut ranked: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());

        for (idx, candidate) in candidates.iter().enumerate() {
            let mut congestion_cost = 0.0;
            for l in 0..challenge.network.num_lines {
                let limit = challenge.network.flow_limits[l].abs().max(EPS);
                let sensitivity =
                    challenge.network.ptdf[l][candidate.node] * candidate.direction / limit;
                if sensitivity > EPS {
                    congestion_cost += lambda_pos[l] * sensitivity;
                } else if sensitivity < -EPS {
                    congestion_cost += lambda_neg[l] * (-sensitivity);
                }
            }

            let net_value = candidate.value / value_scale - congestion_cost;
            ranked.push((idx, net_value));
        }

        ranked.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pressure_pos = vec![0.0f64; challenge.network.num_lines];
        let mut pressure_neg = vec![0.0f64; challenge.network.num_lines];

        for &(idx, net_value) in ranked.iter().take(active_take) {
            if net_value <= 0.0 {
                break;
            }

            saw_positive = true;
            let candidate = candidates[idx];
            let share = (0.18 + 0.82 * (net_value / (0.30 + net_value)).clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
            let dispatched_cap = candidate.cap * share;

            for l in 0..challenge.network.num_lines {
                let limit = challenge.network.flow_limits[l].abs().max(EPS);
                let delta =
                    dispatched_cap * challenge.network.ptdf[l][candidate.node] * candidate.direction
                        / limit;
                if delta > EPS {
                    pressure_pos[l] += delta;
                } else if delta < -EPS {
                    pressure_neg[l] += -delta;
                }
            }
        }

        let mut total_change = 0.0;
        for l in 0..challenge.network.num_lines {
            let limit = challenge.network.flow_limits[l].abs().max(EPS);
            let pos_headroom = ((limit - flows[l]).max(0.0) / limit).clamp(0.0, 2.0);
            let neg_headroom = ((limit + flows[l]).max(0.0) / limit).clamp(0.0, 2.0);

            let pos_mismatch = pressure_pos[l] - (0.84 * pos_headroom + 0.04);
            let neg_mismatch = pressure_neg[l] - (0.84 * neg_headroom + 0.04);

            let pos_step = if pos_mismatch > 0.0 { 0.80 } else { 0.30 };
            let neg_step = if neg_mismatch > 0.0 { 0.80 } else { 0.30 };

            let pos_trial = (lambda_pos[l] + pos_step * pos_mismatch).clamp(0.0, 3.5);
            let neg_trial = (lambda_neg[l] + neg_step * neg_mismatch).clamp(0.0, 3.5);

            let new_pos = 0.72 * lambda_pos[l] + 0.28 * pos_trial;
            let new_neg = 0.72 * lambda_neg[l] + 0.28 * neg_trial;

            total_change += (new_pos - lambda_pos[l]).abs() + (new_neg - lambda_neg[l]).abs();
            lambda_pos[l] = new_pos;
            lambda_neg[l] = new_neg;
        }

        if !total_change.is_finite() {
            return heuristic_shadows();
        }
        if total_change <= 0.88 * prev_change + 1e-9 {
            stable_steps += 1;
        }
        prev_change = total_change;
    }

    if !saw_positive || !prev_change.is_finite() {
        return heuristic_shadows();
    }

    let lambda_mass = lambda_pos.iter().sum::<f64>() + lambda_neg.iter().sum::<f64>();
    if lambda_mass <= EPS
        || (stable_steps == 0 && prev_change > 0.60)
        || (lambda_mass > 0.0 && prev_change > 0.55 * (1.0 + lambda_mass))
        || lambda_pos.iter().chain(lambda_neg.iter()).any(|x| !x.is_finite())
    {
        return heuristic_shadows();
    }

    let fallback = heuristic_shadows();
    let fallback_mass = fallback.iter().map(|x| x.abs()).sum::<f64>();
    let dual: Vec<f64> = (0..challenge.network.num_lines)
        .map(|l| (lambda_pos[l] - lambda_neg[l]).clamp(-3.25, 3.25))
        .collect();
    let dual_mass = dual.iter().map(|x| x.abs()).sum::<f64>();

    if dual_mass <= 0.05 * (1.0 + fallback_mass) {
        return fallback;
    }

    let stability = (stable_steps as f64 / DUAL_ITERS as f64).clamp(0.0, 1.0);
    let mut shadows = vec![0.0; challenge.network.num_lines];
    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        let utilization = (flows[l].abs() / limit).clamp(0.0, 1.5);
        let dual_val = dual[l];
        let fallback_val = fallback[l];

        let same_sign = dual_val * fallback_val >= 0.0
            || dual_val.abs() <= 0.20
            || fallback_val.abs() <= 0.20;

        let dual_weight = if same_sign {
            (0.58 + 0.22 * stability + 0.08 * utilization.min(1.0)).clamp(0.50, 0.88)
        } else {
            (0.32 + 0.18 * stability).clamp(0.32, 0.58)
        };

        shadows[l] =
            (dual_weight * dual_val + (1.0 - dual_weight) * fallback_val).clamp(-3.25, 3.25);
    }

    shadows
}

fn directional_congestion_cost(
    challenge: &Challenge,
    battery_idx: usize,
    direction: f64,
    line_shadows: &[f64],
) -> f64 {
    let node = challenge.batteries[battery_idx].node;
    let mut projection = 0.0;
    for l in 0..line_shadows.len() {
        projection += line_shadows[l] * challenge.network.ptdf[l][node];
    }
    (direction * projection).clamp(-2.5, 2.5)
}

#[derive(Clone, Copy)]
struct DirectionalCandidate {
    idx: usize,
    base: f64,
    cap: f64,
    score: f64,
    stress: f64,
}

struct DirectionalMode {
    weights: Vec<f64>,
    desirability: f64,
    stress: f64,
    is_base: bool,
}

fn normalize_weights(weights: &mut [f64]) -> f64 {
    let sum: f64 = weights.iter().copied().sum();
    if sum > EPS {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    sum
}

fn signed_sensitivity_vector(
    challenge: &Challenge,
    battery_idx: usize,
    sign: f64,
    line_shadows: &[f64],
) -> Vec<f64> {
    let node = challenge.batteries[battery_idx].node;
    let mut out = Vec::with_capacity(challenge.network.num_lines);
    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        let shadow_weight = if line_shadows.is_empty() {
            1.0
        } else {
            0.35 + line_shadows[l].abs()
        };
        out.push(sign * challenge.network.ptdf[l][node] * shadow_weight / limit);
    }
    out
}

fn mode_stress_from_weights(weights: &[f64], sensitivities: &[Vec<f64>]) -> f64 {
    if weights.is_empty() || sensitivities.is_empty() || sensitivities[0].is_empty() {
        return 0.0;
    }

    let num_lines = sensitivities[0].len();
    let mut combined = vec![0.0; num_lines];
    for (k, &weight) in weights.iter().enumerate() {
        if weight <= EPS {
            continue;
        }
        for l in 0..num_lines {
            combined[l] += weight * sensitivities[k][l];
        }
    }

    combined.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn mode_desirability(weights: &[f64], candidates: &[DirectionalCandidate]) -> f64 {
    weights
        .iter()
        .enumerate()
        .map(|(k, &w)| w * candidates[k].score)
        .sum()
}

fn score_seed_weights(candidates: &[DirectionalCandidate], confidence: f64) -> Vec<f64> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut weights = vec![0.0; n];
    let mut min_score = f64::INFINITY;
    let mut max_score = -f64::INFINITY;
    for c in candidates {
        min_score = min_score.min(c.score);
        max_score = max_score.max(c.score);
    }

    for (k, c) in candidates.iter().enumerate() {
        let normalized_score = if max_score > min_score + EPS {
            ((c.score - min_score) / (max_score - min_score)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let score_weight = 0.25 + 0.75 * normalized_score;
        weights[k] = ((1.0 - confidence) * c.base.max(EPS)
            + confidence * c.cap.max(EPS) * score_weight)
            .max(EPS);
    }

    weights
}

fn capped_weighted_allocation(capacities: &[f64], weights: &[f64], total_mag: f64) -> Vec<f64> {
    let n = capacities.len();
    let mut allocations = vec![0.0; n];
    if n == 0 || total_mag <= EPS {
        return allocations;
    }

    let mut remaining = total_mag;
    let mut active = vec![true; n];

    for _ in 0..=n {
        let mut weight_sum = 0.0;
        for k in 0..n {
            if !active[k] {
                continue;
            }
            let headroom = (capacities[k] - allocations[k]).max(0.0);
            if headroom > EPS {
                weight_sum += weights.get(k).copied().unwrap_or(0.0).max(0.0);
            } else {
                active[k] = false;
            }
        }
        if weight_sum <= EPS {
            break;
        }

        let target_remaining = remaining;
        let mut added_total = 0.0;
        for k in 0..n {
            if !active[k] {
                continue;
            }
            let headroom = (capacities[k] - allocations[k]).max(0.0);
            if headroom <= EPS {
                active[k] = false;
                continue;
            }
            let weight = weights.get(k).copied().unwrap_or(0.0).max(0.0);
            let desired_add = target_remaining * weight / weight_sum;
            let add = desired_add.min(headroom);
            if add > EPS {
                allocations[k] += add;
                added_total += add;
            }
            if capacities[k] - allocations[k] <= EPS {
                active[k] = false;
            }
        }

        if added_total <= EPS {
            break;
        }
        remaining -= added_total;
        if remaining <= EPS {
            break;
        }
    }

    if remaining > EPS {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    capacities[b]
                        .partial_cmp(&capacities[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        for k in order {
            if remaining <= EPS {
                break;
            }
            let headroom = (capacities[k] - allocations[k]).max(0.0);
            if headroom <= EPS {
                continue;
            }
            let add = headroom.min(remaining);
            allocations[k] += add;
            remaining -= add;
        }
    }

    allocations
}

fn apply_signed_allocations(
    action: &mut [f64],
    candidates: &[DirectionalCandidate],
    allocations: &[f64],
    sign: f64,
    bounds: &[(f64, f64)],
) {
    for (k, c) in candidates.iter().enumerate() {
        let (min_b, max_b) = bounds[c.idx];
        let signed_action = if sign > 0.0 {
            allocations[k].clamp(0.0, c.cap).clamp(min_b, max_b)
        } else {
            (-allocations[k].clamp(0.0, c.cap)).clamp(min_b, max_b)
        };
        action[c.idx] = signed_action;
    }
}

fn transport_projection_stats(
    challenge: &Challenge,
    from_idx: usize,
    from_sign: f64,
    to_idx: usize,
    to_sign: f64,
    line_shadows: &[f64],
) -> (f64, f64) {
    if challenge.network.num_lines == 0 {
        return (0.0, 0.0);
    }

    let from_node = challenge.batteries[from_idx].node;
    let to_node = challenge.batteries[to_idx].node;
    let mut relief = 0.0;
    let mut stress = 0.0;

    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        let delta =
            (-from_sign * challenge.network.ptdf[l][from_node]
                + to_sign * challenge.network.ptdf[l][to_node])
                / limit;

        if !line_shadows.is_empty() {
            relief -= line_shadows[l] * delta;
        }
        stress += delta * delta;
    }

    (relief, stress.sqrt())
}

fn bounded_transport_delta(
    challenge: &Challenge,
    flows: &[f64],
    from_idx: usize,
    from_sign: f64,
    to_idx: usize,
    to_sign: f64,
    delta_cap: f64,
) -> f64 {
    if delta_cap <= EPS || challenge.network.num_lines == 0 {
        return delta_cap.max(0.0);
    }

    let from_node = challenge.batteries[from_idx].node;
    let to_node = challenge.batteries[to_idx].node;
    let mut upper = delta_cap;

    for l in 0..challenge.network.num_lines {
        let delta_flow =
            -from_sign * challenge.network.ptdf[l][from_node]
                + to_sign * challenge.network.ptdf[l][to_node];

        if delta_flow > EPS {
            upper = upper.min((challenge.network.flow_limits[l] - flows[l]) / delta_flow);
        } else if delta_flow < -EPS {
            upper = upper.min((-challenge.network.flow_limits[l] - flows[l]) / delta_flow);
        }

        if upper <= EPS {
            return 0.0;
        }
    }

    upper.max(0.0)
}

fn apply_transport_move(
    action: &mut [f64],
    bounds: &[(f64, f64)],
    from_idx: usize,
    from_sign: f64,
    to_idx: usize,
    to_sign: f64,
    delta: f64,
) -> f64 {
    if delta <= EPS || from_idx == to_idx {
        return 0.0;
    }

    let from_old = action[from_idx];
    let to_old = action[to_idx];

    let provisional_from =
        (from_old - from_sign * delta).clamp(bounds[from_idx].0, bounds[from_idx].1);
    let source_realized = ((from_old - provisional_from) * from_sign).max(0.0);
    if source_realized <= EPS {
        return 0.0;
    }

    let provisional_to =
        (to_old + to_sign * source_realized).clamp(bounds[to_idx].0, bounds[to_idx].1);
    let target_realized = ((provisional_to - to_old) * to_sign).max(0.0);
    let actual = source_realized.min(target_realized);
    if actual <= EPS {
        return 0.0;
    }

    action[from_idx] =
        (from_old - from_sign * actual).clamp(bounds[from_idx].0, bounds[from_idx].1);
    action[to_idx] = (to_old + to_sign * actual).clamp(bounds[to_idx].0, bounds[to_idx].1);
    actual
}

fn joint_transport_redispatch(
    challenge: &Challenge,
    state: &State,
    action: &mut [f64],
    discharge_candidates: &[DirectionalCandidate],
    charge_candidates: &[DirectionalCandidate],
    confidence: f64,
) {
    const TOP_K: usize = 6;
    const MAX_TRANSPORT_MOVES: usize = 6;

    if challenge.num_batteries < 2 {
        return;
    }

    let mut discharge_sorted = discharge_candidates.to_vec();
    let mut charge_sorted = charge_candidates.to_vec();

    discharge_sorted.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    charge_sorted.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_discharge: Vec<DirectionalCandidate> =
        discharge_sorted.iter().take(TOP_K).copied().collect();
    let top_charge: Vec<DirectionalCandidate> =
        charge_sorted.iter().take(TOP_K).copied().collect();
    let low_discharge: Vec<DirectionalCandidate> =
        discharge_sorted.iter().rev().take(TOP_K).copied().collect();
    let low_charge: Vec<DirectionalCandidate> =
        charge_sorted.iter().rev().take(TOP_K).copied().collect();

    if top_discharge.is_empty() && top_charge.is_empty() {
        return;
    }

    let score_scale = discharge_candidates
        .iter()
        .chain(charge_candidates.iter())
        .map(|c| c.score.abs())
        .sum::<f64>()
        / (discharge_candidates.len() + charge_candidates.len()).max(1) as f64
        + 1.0;

    for _ in 0..MAX_TRANSPORT_MOVES {
        let flows = if challenge.network.num_lines > 0 {
            compute_flows(challenge, state, action)
        } else {
            Vec::new()
        };
        let shadow_view = if challenge.network.num_lines > 0 {
            live_congestion_shadows(challenge, state, action)
        } else {
            Vec::new()
        };

        let mut best_move: Option<(usize, usize, f64, f64, f64, f64)> = None;

        {
            let mut consider = |from: DirectionalCandidate,
                                to: DirectionalCandidate,
                                from_sign: f64,
                                to_sign: f64,
                                same_sign: bool| {
                if from.idx == to.idx {
                    return;
                }

                let current_from = action[from.idx];
                if current_from * from_sign <= EPS {
                    return;
                }

                let source_cap =
                    same_sign_reduction_capacity(current_from, state.action_bounds[from.idx]);
                if source_cap <= EPS {
                    return;
                }

                let target_cap = same_sign_increase_capacity(
                    action[to.idx],
                    state.action_bounds[to.idx],
                    to_sign,
                );
                if target_cap <= EPS {
                    return;
                }

                let delta_cap = bounded_transport_delta(
                    challenge,
                    &flows,
                    from.idx,
                    from_sign,
                    to.idx,
                    to_sign,
                    source_cap.min(target_cap),
                );
                if delta_cap <= EPS {
                    return;
                }

                let (relief, stress) = transport_projection_stats(
                    challenge,
                    from.idx,
                    from_sign,
                    to.idx,
                    to_sign,
                    &shadow_view,
                );
                let unit_gain = to.score - from.score;
                let unit_objective = unit_gain
                    + score_scale
                        * ((if same_sign { 0.18 } else { 0.24 }) * relief
                            - (if same_sign { 0.03 } else { 0.05 }) * stress);

                let gate = if same_sign {
                    unit_gain > 0.02 * score_scale
                        || (relief > 0.08 && unit_gain > -0.03 * score_scale)
                } else {
                    unit_gain > 0.05 * score_scale
                        || (relief > 0.16 && unit_gain > -0.015 * score_scale)
                };

                if !gate || unit_objective <= 0.0 {
                    return;
                }

                let trade_share = if same_sign {
                    (0.24
                        + 0.42 * confidence
                        + 0.18 * relief.clamp(0.0, 1.0)
                        + 0.16 * (unit_gain / score_scale).clamp(0.0, 1.0))
                        .clamp(0.10, 0.90)
                } else {
                    (0.14
                        + 0.34 * confidence
                        + 0.24 * relief.clamp(0.0, 1.2)
                        + 0.18 * (unit_gain / score_scale).clamp(0.0, 1.0))
                        .clamp(0.08, 0.75)
                };

                let delta = delta_cap * trade_share;
                if delta <= EPS {
                    return;
                }

                let total_score = delta * unit_objective;
                match best_move {
                    Some((_, _, _, _, _, best_score)) if total_score <= best_score + 1e-9 => {}
                    _ => {
                        best_move =
                            Some((from.idx, to.idx, from_sign, to_sign, delta, total_score))
                    }
                }
            };

            for from in &low_discharge {
                for to in &top_discharge {
                    consider(*from, *to, 1.0, 1.0, true);
                }
            }
            for from in &low_charge {
                for to in &top_charge {
                    consider(*from, *to, -1.0, -1.0, true);
                }
            }
            for from in &low_discharge {
                for to in &top_charge {
                    consider(*from, *to, 1.0, -1.0, false);
                }
            }
            for from in &low_charge {
                for to in &top_discharge {
                    consider(*from, *to, -1.0, 1.0, false);
                }
            }
        }

        let Some((from_idx, to_idx, from_sign, to_sign, delta, _)) = best_move else {
            break;
        };

        let actual = apply_transport_move(
            action,
            &state.action_bounds,
            from_idx,
            from_sign,
            to_idx,
            to_sign,
            delta,
        );
        if actual <= EPS {
            break;
        }
    }
}

fn allocation_value(allocations: &[f64], candidates: &[DirectionalCandidate]) -> f64 {
    allocations
        .iter()
        .enumerate()
        .map(|(k, &a)| a * candidates[k].score)
        .sum()
}

fn allocation_distribution_stress(
    allocations: &[f64],
    total_mag: f64,
    sensitivities: &[Vec<f64>],
) -> f64 {
    if total_mag <= EPS {
        return 0.0;
    }
    let weights: Vec<f64> = allocations.iter().map(|a| *a / total_mag.max(EPS)).collect();
    mode_stress_from_weights(&weights, sensitivities)
}

fn composite_sensitivity(weights: &[f64], sensitivities: &[Vec<f64>]) -> Vec<f64> {
    if weights.is_empty() || sensitivities.is_empty() || sensitivities[0].is_empty() {
        return Vec::new();
    }

    let num_lines = sensitivities[0].len();
    let mut combined = vec![0.0; num_lines];
    for (k, &weight) in weights.iter().enumerate() {
        if weight <= EPS {
            continue;
        }
        for l in 0..num_lines {
            combined[l] += weight * sensitivities[k][l];
        }
    }
    combined
}

fn bottleneck_focus_sets(seed_pattern: &[f64]) -> Vec<Vec<(usize, f64)>> {
    let mut active: Vec<(usize, f64)> = seed_pattern
        .iter()
        .enumerate()
        .filter_map(|(l, &value)| {
            let mag = value.abs();
            if mag > 1e-9 {
                Some((l, mag))
            } else {
                None
            }
        })
        .collect();

    if active.is_empty() {
        return Vec::new();
    }

    active.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    active.truncate(4);

    let top_mag = active[0].1.max(EPS);
    let mut filtered = Vec::new();
    for (rank, (line, mag)) in active.into_iter().enumerate() {
        if rank > 0 && mag < 0.20 * top_mag {
            break;
        }
        let weight = (0.45 + 0.55 * (mag / top_mag)).clamp(0.45, 1.0);
        filtered.push((line, weight));
    }

    let mut sets = Vec::new();
    for size in 1..=filtered.len().min(3) {
        let mut set = Vec::new();
        for (rank, &(line, weight)) in filtered.iter().take(size).enumerate() {
            let layered_weight = (weight * (1.0 - 0.18 * rank as f64)).clamp(0.20, 1.0);
            set.push((line, layered_weight));
        }
        sets.push(set);
    }

    sets
}

fn focused_line_stress(vec: &[f64], active_lines: &[(usize, f64)]) -> f64 {
    if vec.is_empty() || active_lines.is_empty() {
        return 0.0;
    }

    active_lines
        .iter()
        .map(|&(line, weight)| {
            let value = if line < vec.len() { vec[line] } else { 0.0 };
            let weighted = weight * value;
            weighted * weighted
        })
        .sum::<f64>()
        .sqrt()
}

fn focused_blend_alpha(
    current_vec: &[f64],
    candidate_vec: &[f64],
    active_lines: &[(usize, f64)],
) -> f64 {
    let mut numer = 0.0;
    let mut denom = 0.0;

    for &(line, weight) in active_lines {
        if line >= current_vec.len() || line >= candidate_vec.len() {
            continue;
        }
        let d = current_vec[line] - candidate_vec[line];
        let w2 = weight * weight;
        numer += w2 * candidate_vec[line] * d;
        denom += w2 * d * d;
    }

    if denom > EPS {
        (-numer / denom).clamp(0.15, 0.88)
    } else {
        0.5
    }
}

fn push_unique_mode(modes: &mut Vec<DirectionalMode>, mode: DirectionalMode) {
    let duplicate = modes.iter().any(|existing| {
        existing.weights.len() == mode.weights.len()
            && existing
                .weights
                .iter()
                .zip(mode.weights.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                <= 0.18
    });

    if !duplicate {
        modes.push(mode);
    }
}

fn targeted_nullspace_completion_mode(
    candidates: &[DirectionalCandidate],
    sensitivities: &[Vec<f64>],
    active_lines: &[(usize, f64)],
    anchor: usize,
    best_single_score: f64,
) -> Option<DirectionalMode> {
    if candidates.is_empty()
        || sensitivities.is_empty()
        || anchor >= candidates.len()
        || active_lines.is_empty()
    {
        return None;
    }

    let n = candidates.len();
    let score_scale =
        candidates.iter().map(|c| c.score.abs()).sum::<f64>() / n.max(1) as f64 + 1.0;
    let desirability_floor = if best_single_score > EPS {
        0.58 * best_single_score
    } else {
        best_single_score - 0.10 * score_scale
    };

    let mut weights = vec![0.0; n];
    weights[anchor] = 1.0;

    let mut current_vec = sensitivities[anchor].clone();
    let anchor_focus = focused_line_stress(&current_vec, active_lines);
    if anchor_focus <= EPS {
        return None;
    }

    let anchor_full = current_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut current_focus = anchor_focus;
    let mut current_full = anchor_full;
    let mut current_score = candidates[anchor].score;

    let mut used = vec![false; n];
    used[anchor] = true;

    for _ in 0..3 {
        let mut best_j: Option<usize> = None;
        let mut best_weights = Vec::new();
        let mut best_vec = Vec::new();
        let mut best_focus = current_focus;
        let mut best_full = current_full;
        let mut best_score = current_score;
        let mut best_objective = 0.0;

        for j in 0..n {
            if used[j] {
                continue;
            }

            let alpha = focused_blend_alpha(&current_vec, &sensitivities[j], active_lines);

            let mut trial_weights: Vec<f64> = weights.iter().map(|w| alpha * *w).collect();
            trial_weights[j] += 1.0 - alpha;

            let mut trial_vec = vec![0.0; current_vec.len()];
            for l in 0..current_vec.len() {
                trial_vec[l] =
                    alpha * current_vec[l] + (1.0 - alpha) * sensitivities[j][l];
            }

            let trial_focus = focused_line_stress(&trial_vec, active_lines);
            let trial_full = trial_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            let trial_score = mode_desirability(&trial_weights, candidates);

            if trial_score < desirability_floor {
                continue;
            }

            let focus_gain = current_focus - trial_focus;
            let full_gain = current_full - trial_full;
            let value_loss = (current_score - trial_score).max(0.0) / score_scale;

            let main_line = active_lines[0].0;
            let complement_bonus = if main_line < current_vec.len()
                && main_line < sensitivities[j].len()
                && current_vec[main_line] * sensitivities[j][main_line] < 0.0
            {
                (current_vec[main_line].abs() * sensitivities[j][main_line].abs()).sqrt()
            } else {
                0.0
            };

            let objective =
                1.35 * focus_gain + 0.50 * full_gain + 0.10 * complement_bonus
                    - 0.22 * value_loss;

            let acceptable = objective > 1e-9
                && (trial_focus + 1e-9 < current_focus * 0.995
                    || trial_full + 1e-9 < current_full * 0.98);

            if !acceptable {
                continue;
            }

            let is_better = best_j.is_none()
                || objective > best_objective + 1e-9
                || ((objective - best_objective).abs() <= 1e-9
                    && trial_score > best_score + 1e-9);

            if is_better {
                best_j = Some(j);
                best_weights = trial_weights;
                best_vec = trial_vec;
                best_focus = trial_focus;
                best_full = trial_full;
                best_score = trial_score;
                best_objective = objective;
            }
        }

        let Some(j) = best_j else {
            break;
        };

        let accept = best_focus + 1e-9 < current_focus * 0.96
            || best_full + 1e-9 < current_full * 0.93
            || (best_focus + 1e-9 < current_focus * 0.985
                && best_score >= current_score - 0.05 * score_scale);
        if !accept {
            break;
        }

        used[j] = true;
        weights = best_weights;
        current_vec = best_vec;
        current_focus = best_focus;
        current_full = best_full;
        current_score = best_score;

        if current_focus <= 0.25 * anchor_focus {
            break;
        }
    }

    normalize_weights(&mut weights);

    if current_score < desirability_floor {
        return None;
    }

    if current_focus + 1e-9 < anchor_focus * 0.82
        || current_full + 1e-9 < anchor_full * 0.90
    {
        Some(DirectionalMode {
            desirability: current_score,
            stress: current_full,
            weights,
            is_base: false,
        })
    } else {
        None
    }
}

fn build_signed_mode_basis(
    candidates: &[DirectionalCandidate],
    sensitivities: &[Vec<f64>],
) -> Vec<DirectionalMode> {
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut modes = Vec::new();

    let mut base_weights: Vec<f64> = candidates.iter().map(|c| c.base.max(0.0)).collect();
    let base_sum = normalize_weights(&mut base_weights);
    if base_sum > EPS {
        push_unique_mode(
            &mut modes,
            DirectionalMode {
                desirability: mode_desirability(&base_weights, candidates),
                stress: mode_stress_from_weights(&base_weights, sensitivities),
                weights: base_weights.clone(),
                is_base: true,
            },
        );
    }

    let mut score_weights = score_seed_weights(candidates, 1.0);
    let score_sum = normalize_weights(&mut score_weights);
    if score_sum > EPS {
        push_unique_mode(
            &mut modes,
            DirectionalMode {
                desirability: mode_desirability(&score_weights, candidates),
                stress: mode_stress_from_weights(&score_weights, sensitivities),
                weights: score_weights.clone(),
                is_base: false,
            },
        );
    }

    let mut min_score = f64::INFINITY;
    let mut max_score = -f64::INFINITY;
    for c in candidates {
        min_score = min_score.min(c.score);
        max_score = max_score.max(c.score);
    }

    let mut low_stress_mode = vec![0.0; n];
    for k in 0..n {
        let normalized_score = if max_score > min_score + EPS {
            ((candidates[k].score - min_score) / (max_score - min_score)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        low_stress_mode[k] = candidates[k].cap.max(EPS)
            * (0.25 + 0.75 * normalized_score)
            / (0.15 + candidates[k].stress);
    }
    if normalize_weights(&mut low_stress_mode) > EPS {
        push_unique_mode(
            &mut modes,
            DirectionalMode {
                desirability: mode_desirability(&low_stress_mode, candidates),
                stress: mode_stress_from_weights(&low_stress_mode, sensitivities),
                weights: low_stress_mode,
                is_base: false,
            },
        );
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        candidates[b]
            .score
            .partial_cmp(&candidates[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &k in order.iter().take(2) {
        let mut weights = vec![0.0; n];
        weights[k] = 1.0;
        push_unique_mode(
            &mut modes,
            DirectionalMode {
                desirability: candidates[k].score,
                stress: candidates[k].stress,
                weights,
                is_base: false,
            },
        );
    }

    let best_single_score = order
        .first()
        .map(|&k| candidates[k].score)
        .unwrap_or(0.0);
    let score_scale =
        candidates.iter().map(|c| c.score.abs()).sum::<f64>() / n as f64 + 1.0;

    let mut focus_sets: Vec<Vec<(usize, f64)>> = Vec::new();
    if base_sum > EPS {
        focus_sets.extend(bottleneck_focus_sets(&composite_sensitivity(
            &base_weights,
            sensitivities,
        )));
    }
    if score_sum > EPS {
        focus_sets.extend(bottleneck_focus_sets(&composite_sensitivity(
            &score_weights,
            sensitivities,
        )));
    }
    if let Some(&anchor) = order.first() {
        focus_sets.extend(bottleneck_focus_sets(&sensitivities[anchor]));
    }

    let mut targeted_generated = 0usize;
    for &anchor in order.iter().take(2) {
        for active_lines in focus_sets.iter().take(3) {
            if let Some(mode) = targeted_nullspace_completion_mode(
                candidates,
                sensitivities,
                active_lines,
                anchor,
                best_single_score,
            ) {
                let prev_len = modes.len();
                push_unique_mode(&mut modes, mode);
                if modes.len() > prev_len {
                    targeted_generated += 1;
                }
                if targeted_generated >= 4 {
                    break;
                }
            }
        }
        if targeted_generated >= 4 {
            break;
        }
    }

    if targeted_generated == 0 {
        for &anchor in order.iter().take(2) {
            let mut weights = vec![0.0; n];
            weights[anchor] = 1.0;

            let mut current_vec = sensitivities[anchor].clone();
            let mut current_stress = current_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            let anchor_stress = current_stress;
            let mut current_score = candidates[anchor].score;

            let mut used = vec![false; n];
            used[anchor] = true;

            for _ in 0..2 {
                let mut best_j: Option<usize> = None;
                let mut best_weights = Vec::new();
                let mut best_vec = Vec::new();
                let mut best_stress = current_stress;
                let mut best_score = current_score;

                for j in 0..n {
                    if used[j] {
                        continue;
                    }

                    let mut denom = 0.0;
                    let mut numer = 0.0;
                    for l in 0..current_vec.len() {
                        let d = current_vec[l] - sensitivities[j][l];
                        denom += d * d;
                        numer += sensitivities[j][l] * d;
                    }

                    let alpha = if denom > EPS {
                        (-numer / denom).clamp(0.20, 0.85)
                    } else {
                        0.5
                    };

                    let mut trial_weights: Vec<f64> =
                        weights.iter().map(|w| alpha * *w).collect();
                    trial_weights[j] += 1.0 - alpha;

                    let mut trial_vec = vec![0.0; current_vec.len()];
                    for l in 0..current_vec.len() {
                        trial_vec[l] =
                            alpha * current_vec[l] + (1.0 - alpha) * sensitivities[j][l];
                    }

                    let trial_stress = trial_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let trial_score = mode_desirability(&trial_weights, candidates);

                    let trial_objective = trial_stress - 0.06 * trial_score / score_scale;
                    let best_objective = best_stress - 0.06 * best_score / score_scale;

                    let is_better = trial_objective + 1e-9 < best_objective
                        || ((trial_objective - best_objective).abs() <= 1e-9
                            && trial_score > best_score + 1e-9);

                    if is_better {
                        best_j = Some(j);
                        best_weights = trial_weights;
                        best_vec = trial_vec;
                        best_stress = trial_stress;
                        best_score = trial_score;
                    }
                }

                let Some(j) = best_j else {
                    break;
                };

                let accept = best_stress + 1e-9 < current_stress * 0.97
                    || best_score > current_score + 0.04 * score_scale;
                if !accept {
                    break;
                }

                used[j] = true;
                weights = best_weights;
                current_vec = best_vec;
                current_stress = best_stress;
                current_score = best_score;
            }

            if current_stress + 1e-9 < anchor_stress * 0.94 {
                push_unique_mode(
                    &mut modes,
                    DirectionalMode {
                        desirability: current_score,
                        stress: current_stress,
                        weights,
                        is_base: false,
                    },
                );
            }
        }
    }

    modes
}

fn allocate_directional_modes(
    action: &mut [f64],
    candidates: &[DirectionalCandidate],
    sensitivities: &[Vec<f64>],
    total_mag: f64,
    sign: f64,
    bounds: &[(f64, f64)],
    confidence: f64,
) -> bool {
    if candidates.len() < 2 || sensitivities.is_empty() || total_mag <= EPS {
        return false;
    }

    let modes = build_signed_mode_basis(candidates, sensitivities);
    if modes.len() < 2 {
        return false;
    }

    let mut min_des = f64::INFINITY;
    let mut max_des = -f64::INFINITY;
    let mut min_stress = f64::INFINITY;
    let mut max_stress = -f64::INFINITY;
    for mode in &modes {
        min_des = min_des.min(mode.desirability);
        max_des = max_des.max(mode.desirability);
        min_stress = min_stress.min(mode.stress);
        max_stress = max_stress.max(mode.stress);
    }

    let mut mode_mix = vec![0.0; modes.len()];
    for (m, mode) in modes.iter().enumerate() {
        let desir_norm = if max_des > min_des + EPS {
            ((mode.desirability - min_des) / (max_des - min_des)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let stress_pref = if max_stress > min_stress + EPS {
            1.0 - ((mode.stress - min_stress) / (max_stress - min_stress)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        mode_mix[m] = if mode.is_base {
            (1.0 - confidence).clamp(0.15, 0.70) * (0.65 + 0.35 * desir_norm)
        } else {
            confidence.max(0.15) * (0.18 + 0.47 * desir_norm + 0.35 * stress_pref)
        };
    }

    if normalize_weights(&mut mode_mix) <= EPS {
        return false;
    }

    let mut target = vec![0.0; candidates.len()];
    for (m, mode) in modes.iter().enumerate() {
        for k in 0..candidates.len() {
            target[k] += mode_mix[m] * mode.weights[k];
        }
    }
    if normalize_weights(&mut target) <= EPS {
        return false;
    }

    let mut fallback_weights = score_seed_weights(candidates, confidence.max(0.2));
    if normalize_weights(&mut fallback_weights) <= EPS {
        return false;
    }

    for k in 0..target.len() {
        target[k] = (0.75 * target[k] + 0.25 * fallback_weights[k]).max(EPS);
    }
    normalize_weights(&mut target);

    let capacities: Vec<f64> = candidates.iter().map(|c| c.cap).collect();
    let mode_alloc = capped_weighted_allocation(&capacities, &target, total_mag);
    let fallback_alloc = capped_weighted_allocation(&capacities, &fallback_weights, total_mag);

    let score_scale = candidates
        .iter()
        .map(|c| c.score.abs())
        .sum::<f64>()
        / candidates.len() as f64
        + 1.0;

    let mode_value = allocation_value(&mode_alloc, candidates);
    let fallback_value = allocation_value(&fallback_alloc, candidates);
    let mode_stress = allocation_distribution_stress(&mode_alloc, total_mag, sensitivities);
    let fallback_stress =
        allocation_distribution_stress(&fallback_alloc, total_mag, sensitivities);

    let use_modes = mode_value + 0.02 * score_scale * total_mag >= fallback_value
        || (fallback_stress > EPS && mode_stress + 1e-9 < 0.90 * fallback_stress)
        || (fallback_stress <= EPS
            && mode_value > fallback_value + 0.01 * score_scale * total_mag);

    if !use_modes {
        return false;
    }

    apply_signed_allocations(action, candidates, &mode_alloc, sign, bounds);
    true
}

fn allocate_directional_total(
    action: &mut [f64],
    candidates: &[DirectionalCandidate],
    total_mag: f64,
    sign: f64,
    bounds: &[(f64, f64)],
    confidence: f64,
) {
    if candidates.is_empty() || total_mag <= EPS {
        return;
    }

    let mut weights = score_seed_weights(candidates, confidence);
    if normalize_weights(&mut weights) <= EPS {
        return;
    }

    let capacities: Vec<f64> = candidates.iter().map(|c| c.cap).collect();
    let allocations = capped_weighted_allocation(&capacities, &weights, total_mag);
    apply_signed_allocations(action, candidates, &allocations, sign, bounds);
}

fn coordinated_directional_redispatch(
    challenge: &Challenge,
    state: &State,
    action: &mut [f64],
    agg_soc: f64,
) {
    let t = state.time_step;
    let num_steps = challenge.num_steps;
    if t >= num_steps {
        return;
    }

    let steps_remaining = num_steps - t;
    let da_prices = &challenge.market.day_ahead_prices;
    let line_shadows = if challenge.network.num_lines > 0 {
        live_congestion_shadows(challenge, state, action)
    } else {
        Vec::new()
    };
    let max_shadow = line_shadows
        .iter()
        .fold(0.0f64, |acc, &shadow| acc.max(shadow.abs()));
    let mean_shadow = if line_shadows.is_empty() {
        0.0
    } else {
        line_shadows.iter().map(|shadow| shadow.abs()).sum::<f64>() / line_shadows.len() as f64
    };

    let mut discharge_candidates: Vec<DirectionalCandidate> = Vec::new();
    let mut charge_candidates: Vec<DirectionalCandidate> = Vec::new();
    let mut discharge_total = 0.0f64;
    let mut charge_total = 0.0f64;
    let mut signal_sum = 0.0f64;
    let mut price_scale = 0.0f64;
    let mut signal_count = 0usize;

    for i in 0..challenge.num_batteries {
        let base_action = action[i];
        if base_action.abs() <= EPS {
            continue;
        }

        let battery = &challenge.batteries[i];
        let node = battery.node;
        let current_price = da_prices[t][node];
        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = ((state.socs[i] - battery.soc_min_mwh) / soc_range).clamp(0.0, 1.0);
        let fleet_imbalance = soc_level - agg_soc;

        let mut future_min = current_price;
        let mut future_max = current_price;
        let mut future_sum = 0.0;
        for s in t..num_steps {
            let p = da_prices[s][node];
            future_min = future_min.min(p);
            future_max = future_max.max(p);
            future_sum += p;
        }
        let future_avg = future_sum / steps_remaining as f64;
        let spread = (future_max - future_min).max(0.0);
        let congestion_price_scale =
            0.10 * (current_price.abs() + future_avg.abs()) + 0.30 * spread + 1.0;

        if base_action > 0.0 {
            discharge_total += base_action;
            let energy_above_min = (state.socs[i] - battery.soc_min_mwh).max(0.0);
            let per_step_energy =
                (battery.power_discharge_mw * battery.efficiency_discharge).max(EPS);
            let liquidation_pressure =
                (energy_above_min / (steps_remaining as f64 * per_step_energy)).min(2.0);
            let raw_shadow = if line_shadows.is_empty() {
                0.0
            } else {
                directional_congestion_cost(challenge, i, 1.0, &line_shadows)
            };
            let congestion_adjustment = if raw_shadow > 0.0 {
                raw_shadow
            } else {
                0.35 * raw_shadow
            };
            let stress = if challenge.network.num_lines > 0 {
                let sensitivity = signed_sensitivity_vector(challenge, i, 1.0, &line_shadows);
                sensitivity.iter().map(|x| x * x).sum::<f64>().sqrt()
            } else {
                0.0
            };

            let score = current_price * battery.efficiency_discharge
                + 0.55 * (current_price - future_avg)
                + 0.35 * (current_price - future_min).max(0.0)
                + spread
                    * (0.35 * soc_level
                        + 0.25 * liquidation_pressure
                        + 0.15 * fleet_imbalance.max(0.0))
                - congestion_price_scale * congestion_adjustment;

            discharge_candidates.push(DirectionalCandidate {
                idx: i,
                base: base_action,
                cap: state.action_bounds[i].1.max(0.0),
                score,
                stress,
            });
            signal_sum += (current_price - future_avg).abs() + 0.5 * spread;
        } else {
            charge_total += -base_action;
            let energy_below_max = (battery.soc_max_mwh - state.socs[i]).max(0.0);
            let per_step_energy =
                (battery.power_charge_mw * battery.efficiency_charge).max(EPS);
            let refill_pressure =
                (energy_below_max / (steps_remaining as f64 * per_step_energy)).min(2.0);
            let future_value =
                future_max * battery.efficiency_charge * battery.efficiency_discharge;
            let raw_shadow = if line_shadows.is_empty() {
                0.0
            } else {
                directional_congestion_cost(challenge, i, -1.0, &line_shadows)
            };
            let congestion_adjustment = if raw_shadow > 0.0 {
                raw_shadow
            } else {
                0.35 * raw_shadow
            };
            let stress = if challenge.network.num_lines > 0 {
                let sensitivity = signed_sensitivity_vector(challenge, i, -1.0, &line_shadows);
                sensitivity.iter().map(|x| x * x).sum::<f64>().sqrt()
            } else {
                0.0
            };

            let score = (future_value - current_price)
                + 0.45 * (future_avg - current_price)
                + spread
                    * (0.35 * (1.0 - soc_level)
                        + 0.25 * refill_pressure
                        + 0.15 * (-fleet_imbalance).max(0.0))
                - congestion_price_scale * congestion_adjustment;

            charge_candidates.push(DirectionalCandidate {
                idx: i,
                base: -base_action,
                cap: (-state.action_bounds[i].0).max(0.0),
                score,
                stress,
            });
            signal_sum += (future_avg - current_price).abs() + 0.5 * spread;
        }

        price_scale += current_price.abs() + future_avg.abs() + spread;
        signal_count += 1;
    }

    let base_confidence = if signal_count > 0 {
        (signal_sum / (price_scale + signal_sum + EPS)).clamp(0.0, 0.85)
    } else {
        0.0
    };
    let congestion_confidence = if max_shadow > 0.08 {
        (0.05 + 0.10 * mean_shadow + 0.12 * max_shadow).clamp(0.0, 0.35)
    } else {
        0.0
    };
    let confidence = base_confidence.max(congestion_confidence).clamp(0.0, 0.85);
    if confidence <= 0.05 {
        return;
    }

    if discharge_total > EPS {
        let discharge_sensitivities = if challenge.network.num_lines > 0 {
            discharge_candidates
                .iter()
                .map(|c| signed_sensitivity_vector(challenge, c.idx, 1.0, &line_shadows))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let used_modes = allocate_directional_modes(
            action,
            &discharge_candidates,
            &discharge_sensitivities,
            discharge_total,
            1.0,
            &state.action_bounds,
            confidence,
        );
        if !used_modes {
            allocate_directional_total(
                action,
                &discharge_candidates,
                discharge_total,
                1.0,
                &state.action_bounds,
                confidence,
            );
        }
    }

    if charge_total > EPS {
        let charge_sensitivities = if challenge.network.num_lines > 0 {
            charge_candidates
                .iter()
                .map(|c| signed_sensitivity_vector(challenge, c.idx, -1.0, &line_shadows))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let used_modes = allocate_directional_modes(
            action,
            &charge_candidates,
            &charge_sensitivities,
            charge_total,
            -1.0,
            &state.action_bounds,
            confidence,
        );
        if !used_modes {
            allocate_directional_total(
                action,
                &charge_candidates,
                charge_total,
                -1.0,
                &state.action_bounds,
                confidence,
            );
        }
    }

    joint_transport_redispatch(
        challenge,
        state,
        action,
        &discharge_candidates,
        &charge_candidates,
        confidence,
    );
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let num_steps = challenge.num_steps;

    let (total_soc_above_min, total_capacity) = challenge.batteries.iter().enumerate().fold(
        (0.0f64, 0.0f64),
        |(acc_soc, acc_cap), (i, battery)| {
            let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
            let soc_above = (state.socs[i] - battery.soc_min_mwh).max(0.0);
            (acc_soc + soc_above, acc_cap + soc_range)
        },
    );
    let agg_soc = if total_capacity > EPS {
        total_soc_above_min / total_capacity
    } else {
        0.5
    };

    let mut action = vec![0.0f64; challenge.num_batteries];

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (min_bound, max_bound) = state.action_bounds[i];

        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = (state.socs[i] - battery.soc_min_mwh) / soc_range;
        let _imbalance = soc_level - agg_soc;

        let steps_remaining = num_steps - t;

        if steps_remaining == 0 {
            action[i] = 0.0_f64.clamp(min_bound, max_bound);
            continue;
        }

        if steps_remaining <= 1 {
            let desired = battery.power_discharge_mw * soc_level;
            action[i] = desired.clamp(min_bound, max_bound);
            continue;
        }

        let soc_above_min = (state.socs[i] - battery.soc_min_mwh).max(0.0);
        let max_discharge_energy = battery.power_discharge_mw * battery.efficiency_discharge;
        let steps_to_empty = if max_discharge_energy > EPS {
            (soc_above_min / max_discharge_energy).ceil() as usize
        } else {
            usize::MAX
        };
        let time_trigger = steps_remaining <= steps_to_empty + 4 && soc_above_min > EPS;
        let price_trigger = if steps_remaining < num_steps / 2 && soc_above_min > EPS && max_discharge_energy > EPS {
            let da_prices_node: Vec<f64> = (t..num_steps)
                .map(|s| challenge.market.day_ahead_prices[s][node])
                .collect();
            let mut sorted_prices = da_prices_node.clone();
            sorted_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p75_idx = (sorted_prices.len() * 3 / 4).min(sorted_prices.len().saturating_sub(1));
            let p75 = sorted_prices[p75_idx];
            let high_price_steps = da_prices_node.iter().filter(|&&p| p > p75).count();
            let discharge_steps_needed = (soc_above_min / max_discharge_energy).ceil() as usize;
            high_price_steps < discharge_steps_needed
        } else {
            false
        };
        let liquidation_urgent = time_trigger || price_trigger;

        let desired = if liquidation_urgent {
            let decay = 0.3_f64;
            let steps_f = steps_remaining as f64;
            let mut scored_steps: Vec<(f64, usize)> = (t..num_steps)
                .map(|s| {
                    let price = challenge.market.day_ahead_prices[s][node];
                    let marginal_revenue = price * battery.efficiency_discharge;
                    let time_urgency = (-decay * (s - t) as f64 / steps_f).exp();
                    (marginal_revenue * time_urgency, s)
                })
                .collect();
            scored_steps.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut soc_budget = soc_above_min;
            let mut current_step_is_discharge = false;
            for (_, step) in &scored_steps {
                if soc_budget <= EPS {
                    break;
                }
                let discharge_this_step = soc_budget.min(max_discharge_energy);
                if *step == t {
                    current_step_is_discharge = true;
                    break;
                }
                soc_budget -= discharge_this_step;
            }
            if current_step_is_discharge && max_discharge_energy > EPS {
                battery.power_discharge_mw.clamp(min_bound, max_bound)
            } else {
                0.0_f64.clamp(min_bound, max_bound)
            }
        } else {
            let raw_prices: Vec<f64> = (t..num_steps)
                .map(|s| challenge.market.day_ahead_prices[s][node])
                .collect();
            let smoothed_prices = adaptive_price_summary(&raw_prices);
            price_rank_action(challenge, state, i, node, &smoothed_prices)
        };

        action[i] = desired.clamp(min_bound, max_bound);
    }

    if challenge.num_batteries <= 4 {
        let mut discharge_batteries: Vec<(usize, f64, f64)> = Vec::new();
        let mut charge_batteries: Vec<(usize, f64, f64)> = Vec::new();

        for i in 0..challenge.num_batteries {
            let a = action[i];
            if a.abs() <= EPS {
                continue;
            }

            if a > 0.0 {
                discharge_batteries.push((i, directional_value(challenge, state, i, 1.0), a));
            } else {
                charge_batteries.push((i, directional_value(challenge, state, i, -1.0), a));
            }
        }

        discharge_batteries.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        charge_batteries.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let total_discharge_cap: f64 = discharge_batteries.iter().map(|&(_, _, a)| a).sum();
        let total_charge_cap: f64 = charge_batteries.iter().map(|&(_, _, a)| a.abs()).sum();

        if total_discharge_cap > EPS {
            let n = discharge_batteries.len();
            for (rank, &(idx, _, a)) in discharge_batteries.iter().enumerate() {
                let priority_scale = if n <= 1 {
                    1.0
                } else {
                    1.0 - 0.5 * (rank as f64 / (n - 1) as f64)
                };
                action[idx] = (a * priority_scale).clamp(state.action_bounds[idx].0, state.action_bounds[idx].1);
            }
        }

        if total_charge_cap > EPS {
            let n = charge_batteries.len();
            for (rank, &(idx, _, a)) in charge_batteries.iter().enumerate() {
                let priority_scale = if n <= 1 {
                    1.0
                } else {
                    1.0 - 0.5 * (rank as f64 / (n - 1) as f64)
                };
                action[idx] = (a * priority_scale).clamp(state.action_bounds[idx].0, state.action_bounds[idx].1);
            }
        }
    } else {
        coordinated_directional_redispatch(challenge, state, &mut action, agg_soc);
    }

    enforce_flow_feasibility(challenge, state, action)
}