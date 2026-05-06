use anyhow::{anyhow, Result};
use tig_challenges::energy_arbitrage::{Challenge, State};

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;
const LOOKAHEAD_WINDOW: usize = 20;

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

fn descend_total_violation(
    challenge: &Challenge,
    flows: &mut [f64],
    action: &mut [f64],
    values: Option<&[f64]>,
) -> bool {
    let mut violated: Vec<(usize, f64, f64)> = Vec::new();
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let amount = flow.abs() - limit;
        if amount > tig_challenges::energy_arbitrage::constants::EPS_FLOW * limit {
            violated.push((l, flow.signum(), amount));
        }
    }
    if violated.is_empty() {
        return false;
    }

    let values = values.filter(|v| v.len() == challenge.num_batteries);

    let mut best_i: Option<usize> = None;
    let mut best_score = 0.0;
    let mut best_relief = 0.0;
    let mut best_trim = 0.0;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let action_sign = action[i].signum();
        if action_sign.abs() <= EPS {
            continue;
        }

        let mut marginal_relief = 0.0;
        let mut trim_cap = action[i].abs();
        for &(line, signed_direction, amount) in &violated {
            let relief_per_mw =
                signed_direction * challenge.network.ptdf[line][battery.node] * action_sign;
            if relief_per_mw > EPS {
                marginal_relief += relief_per_mw;
                trim_cap = trim_cap.min(amount / relief_per_mw);
            }
        }
        if marginal_relief <= EPS || trim_cap <= EPS {
            continue;
        }

        let value_loss = values.map_or(1.0, |v| v[i].abs().max(EPS));
        let score = marginal_relief / value_loss;
        let better = match best_i {
            Some(_) => {
                score > best_score + EPS
                    || ((score - best_score).abs() <= EPS
                        && (marginal_relief > best_relief + EPS
                            || ((marginal_relief - best_relief).abs() <= EPS
                                && trim_cap > best_trim + EPS)))
            }
            None => true,
        };
        if better {
            best_i = Some(i);
            best_score = score;
            best_relief = marginal_relief;
            best_trim = trim_cap;
        }
    }

    let Some(i) = best_i else {
        return false;
    };
    let trim = best_trim.min(action[i].abs());
    if trim <= EPS {
        return false;
    }

    let delta = -action[i].signum() * trim;
    action[i] += delta;
    let node = challenge.batteries[i].node;
    for (l, flow) in flows.iter_mut().enumerate() {
        *flow += challenge.network.ptdf[l][node] * delta;
    }
    true
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
    values: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let mut flows = compute_flows(challenge, state, &action);

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let feasible = flows.iter().enumerate().all(|(l, &flow)| {
            let limit = challenge.network.flow_limits[l];
            flow.abs()
                <= limit * (1.0 + tig_challenges::energy_arbitrage::constants::EPS_FLOW)
        });
        if feasible {
            return Ok(action);
        }

        if !descend_total_violation(challenge, &mut flows, &mut action, values) {
            break;
        }
    }

    let feasible = flows.iter().enumerate().all(|(l, &flow)| {
        let limit = challenge.network.flow_limits[l];
        flow.abs() <= limit * (1.0 + tig_challenges::energy_arbitrage::constants::EPS_FLOW)
    });
    if feasible {
        return Ok(action);
    }

    let zero = vec![0.0; action.len()];
    let zero_flows = compute_flows(challenge, state, &zero);
    let zero_feasible = zero_flows.iter().enumerate().all(|(l, &flow)| {
        let limit = challenge.network.flow_limits[l];
        flow.abs() <= limit * (1.0 + tig_challenges::energy_arbitrage::constants::EPS_FLOW)
    });
    if !zero_feasible {
        return Err(anyhow!("Grid infeasible even with zero battery actions"));
    }

    let base = action;
    let delta_flows: Vec<f64> = flows
        .iter()
        .zip(zero_flows.iter())
        .map(|(&f, &z)| f - z)
        .collect();

    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let feasible = (0..challenge.network.num_lines).all(|l| {
            let flow = zero_flows[l] + mid * delta_flows[l];
            let limit = challenge.network.flow_limits[l];
            flow.abs()
                <= limit * (1.0 + tig_challenges::energy_arbitrage::constants::EPS_FLOW)
        });
        if feasible {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(base.into_iter().map(|u| low * u).collect())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let da_prices = &challenge.market.day_ahead_prices;
    let remaining_steps = num_steps - t;
    let horizon_scale = remaining_steps as f64 / num_steps as f64;

    let sample_budget = LOOKAHEAD_WINDOW.min(remaining_steps.max(1));
    let mut boundary_times = Vec::with_capacity(sample_budget);
    if sample_budget == 1 {
        boundary_times.push(t);
    } else {
        let span = remaining_steps - 1;
        let denom = (sample_budget - 1) * (sample_budget - 1);
        for s in 0..sample_budget {
            boundary_times.push(t + span * s * s / denom);
        }
        boundary_times.dedup();
    }

    let mut sampled_prices = vec![Vec::new(); num_nodes];
    for node in 0..num_nodes {
        let mut sample_times = Vec::with_capacity(boundary_times.len() + 1);
        sample_times.push(boundary_times[0]);

        for w in 0..boundary_times.len().saturating_sub(1) {
            let left = boundary_times[w];
            let right = boundary_times[w + 1];
            let chosen = if right <= left + 1 {
                right
            } else {
                let mid = (left + right) / 2;
                let pl = da_prices[left][node];
                let pm = da_prices[mid][node];
                let pr = da_prices[right][node];
                let mid_score =
                    (pm - 0.5 * (pl + pr)).abs() + (pm - pl).abs().max((pr - pm).abs());
                let end_score = (pr - pl).abs();
                if mid_score > end_score { mid } else { right }
            };
            if chosen > *sample_times.last().unwrap() {
                sample_times.push(chosen);
            }
        }

        if *sample_times.last().unwrap() != num_steps - 1 {
            sample_times.push(num_steps - 1);
        }

        let prices = &mut sampled_prices[node];
        prices.reserve(sample_times.len());
        for &tt in &sample_times {
            prices.push(da_prices[tt][node]);
        }
        prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    let mut node_batteries = vec![Vec::new(); num_nodes];
    for (i, battery) in challenge.batteries.iter().enumerate() {
        node_batteries[battery.node].push(i);
    }

    let mut node_preferred = vec![0.0f64; num_nodes];
    let mut node_discharge_room = vec![0.0f64; num_nodes];
    let mut node_charge_room = vec![0.0f64; num_nodes];
    let mut node_discharge_opportunity = vec![0.0f64; num_nodes];
    let mut node_charge_opportunity = vec![0.0f64; num_nodes];
    let mut node_discharge_score = vec![0.0f64; num_nodes];
    let mut node_charge_score = vec![0.0f64; num_nodes];

    let mut total_samples = 0.0;
    let mut discharge_slot_count = 0.0;
    let mut charge_slot_count = 0.0;
    let mut discharge_strength = 0.0;
    let mut charge_strength = 0.0;
    let mut discharge_strength_cap = 0.0;
    let mut charge_strength_cap = 0.0;
    let mut total_desired_discharge = 0.0;
    let mut total_desired_charge = 0.0;
    let mut total_discharge_room = 0.0;
    let mut total_charge_room = 0.0;

    for node in 0..num_nodes {
        let batteries = &node_batteries[node];
        if batteries.is_empty() {
            continue;
        }

        let current_price = da_prices[t][node];
        let prices = &sampled_prices[node];
        let price_floor = prices[0];
        let price_ceiling = prices[prices.len() - 1];
        let price_range = price_ceiling - price_floor;

        let mut soc_min_sum = 0.0;
        let mut soc_max_sum = 0.0;
        let mut soc_sum = 0.0;
        let mut charge_power_sum = 0.0;
        let mut discharge_power_sum = 0.0;
        let mut min_bound_sum = 0.0;
        let mut max_bound_sum = 0.0;
        let mut discharge_room = 0.0;
        let mut charge_room = 0.0;

        for &i in batteries {
            let battery = &challenge.batteries[i];
            let (min_bound, max_bound) = state.action_bounds[i];
            soc_min_sum += battery.soc_min_mwh;
            soc_max_sum += battery.soc_max_mwh;
            soc_sum += state.socs[i];
            charge_power_sum += battery.power_charge_mw;
            discharge_power_sum += battery.power_discharge_mw;
            min_bound_sum += min_bound;
            max_bound_sum += max_bound;
            discharge_room += max_bound.max(0.0);
            charge_room += (-min_bound).max(0.0);
        }

        let soc_range = (soc_max_sum - soc_min_sum).max(EPS);
        let soc_level = ((soc_sum - soc_min_sum) / soc_range).clamp(0.0, 1.0);

        let desired_raw = if price_range <= EPS {
            let urgency = (soc_level * (1.0 - horizon_scale)).sqrt();
            discharge_power_sum * urgency
        } else {
            let rank = (horizon_scale * (1.0 - soc_level) * (prices.len() - 1) as f64) as usize;
            let shadow_price = prices[rank.min(prices.len() - 1)];
            if current_price < shadow_price {
                let headroom = 1.0 - soc_level;
                let urgency = ((shadow_price - current_price) / price_range).clamp(0.0, 1.0);
                -charge_power_sum * (headroom * urgency).sqrt()
            } else if current_price > shadow_price {
                let available = soc_level;
                let urgency = ((current_price - shadow_price) / price_range).clamp(0.0, 1.0);
                discharge_power_sum * (available * urgency).sqrt()
            } else {
                0.0
            }
        };
        let desired_node = desired_raw.clamp(min_bound_sum, max_bound_sum);
        node_preferred[node] = desired_node;
        node_discharge_room[node] = discharge_room;
        node_charge_room[node] = charge_room;
        total_discharge_room += discharge_room;
        total_charge_room += charge_room;

        let discharge_opportunity: f64 = prices
            .iter()
            .map(|&p| (current_price - p).max(0.0))
            .sum();
        let charge_opportunity: f64 = prices
            .iter()
            .map(|&p| (p - current_price).max(0.0))
            .sum();
        let sample_count = prices.len() as f64;
        let discharge_count = prices.iter().filter(|&&p| current_price > p).count() as f64;
        let charge_count = prices.iter().filter(|&&p| current_price < p).count() as f64;

        total_samples += sample_count;
        discharge_slot_count += discharge_count;
        charge_slot_count += charge_count;
        if price_range > EPS {
            discharge_strength += discharge_opportunity;
            charge_strength += charge_opportunity;
            discharge_strength_cap += sample_count * price_range;
            charge_strength_cap += sample_count * price_range;
        }

        node_discharge_opportunity[node] = discharge_opportunity;
        node_charge_opportunity[node] = charge_opportunity;

        if desired_node > EPS {
            total_desired_discharge += desired_node;
            node_discharge_score[node] =
                discharge_opportunity * desired_node / discharge_room.max(EPS);
        } else if desired_node < -EPS {
            total_desired_charge += -desired_node;
            node_charge_score[node] =
                charge_opportunity * (-desired_node) / charge_room.max(EPS);
        }
    }

    let discharge_count_scale = if total_samples > EPS {
        discharge_slot_count / total_samples
    } else {
        0.0
    };
    let charge_count_scale = if total_samples > EPS {
        charge_slot_count / total_samples
    } else {
        0.0
    };
    let discharge_strength_scale = if discharge_strength_cap > EPS {
        discharge_strength / discharge_strength_cap
    } else {
        0.0
    };
    let charge_strength_scale = if charge_strength_cap > EPS {
        charge_strength / charge_strength_cap
    } else {
        0.0
    };

    let discharge_budget_scale = (discharge_count_scale.max(1.0 - horizon_scale)
        * discharge_strength_scale.max(1.0 - horizon_scale))
        .sqrt();
    let charge_budget_scale = (charge_count_scale * charge_strength_scale).sqrt();

    let discharge_budget =
        total_desired_discharge.min(total_discharge_room * discharge_budget_scale);
    let charge_budget = total_desired_charge.min(total_charge_room * charge_budget_scale);

    let mut node_target = vec![0.0f64; num_nodes];

    if discharge_budget > EPS {
        let mut order: Vec<usize> = (0..num_nodes)
            .filter(|&node| node_preferred[node] > EPS)
            .collect();
        order.sort_by(|&a, &b| {
            node_discharge_score[b]
                .partial_cmp(&node_discharge_score[a])
                .unwrap()
                .then_with(|| node_preferred[b].partial_cmp(&node_preferred[a]).unwrap())
        });

        let mut remaining = discharge_budget;
        for node in order {
            if remaining <= EPS {
                break;
            }
            let take = remaining.min(node_preferred[node]);
            node_target[node] = take;
            remaining -= take;
        }
    }

    if charge_budget > EPS {
        let mut order: Vec<usize> = (0..num_nodes)
            .filter(|&node| node_preferred[node] < -EPS)
            .collect();
        order.sort_by(|&a, &b| {
            node_charge_score[b]
                .partial_cmp(&node_charge_score[a])
                .unwrap()
                .then_with(|| (-node_preferred[b]).partial_cmp(&(-node_preferred[a])).unwrap())
        });

        let mut remaining = charge_budget;
        for node in order {
            if remaining <= EPS {
                break;
            }
            let take = remaining.min(-node_preferred[node]);
            node_target[node] = -take;
            remaining -= take;
        }
    }

    let mut action = vec![0.0f64; challenge.num_batteries];
    let mut battery_value = vec![0.0f64; challenge.num_batteries];

    for node in 0..num_nodes {
        let batteries = &node_batteries[node];
        let target = node_target[node];
        if batteries.is_empty() || target.abs() <= EPS {
            continue;
        }

        if target > EPS {
            let total_room = node_discharge_room[node];
            if total_room > EPS {
                let scale = (target / total_room).clamp(0.0, 1.0);
                for &i in batteries {
                    let room = state.action_bounds[i].1.max(0.0);
                    action[i] = scale * room;

                    let battery = &challenge.batteries[i];
                    let available_energy = (state.socs[i] - battery.soc_min_mwh).max(0.0);
                    let duration = if battery.power_discharge_mw > EPS {
                        available_energy / battery.power_discharge_mw
                    } else {
                        0.0
                    };
                    battery_value[i] = node_discharge_opportunity[node] * duration.max(EPS);
                }
            }
        } else {
            let total_room = node_charge_room[node];
            if total_room > EPS {
                let scale = (-target / total_room).clamp(0.0, 1.0);
                for &i in batteries {
                    let room = (-state.action_bounds[i].0).max(0.0);
                    action[i] = -scale * room;

                    let battery = &challenge.batteries[i];
                    let headroom_energy = (battery.soc_max_mwh - state.socs[i]).max(0.0);
                    let duration = if battery.power_charge_mw > EPS {
                        headroom_energy / battery.power_charge_mw
                    } else {
                        0.0
                    };
                    battery_value[i] = node_charge_opportunity[node] * duration.max(EPS);
                }
            }
        }
    }

    enforce_flow_feasibility(challenge, state, action, Some(&battery_value))
}