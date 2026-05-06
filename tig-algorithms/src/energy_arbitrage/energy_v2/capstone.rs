use anyhow::{anyhow, Result};
use std::cell::RefCell;
use tig_challenges::energy_arbitrage::{Challenge, State};

const MAX_FLOW_ADJUST_ITERS: usize = 16;
const EPS: f64 = 1e-12;
const CHARGE_QUANTILE: f64 = 0.30;
const DISCHARGE_QUANTILE: f64 = 0.70;
const NUM_TIME_SEGMENTS: usize = 3;
const CONTINUATION_TAIL_K: usize = 3;
const CONTINUATION_EXTREME_BLEND: f64 = 0.45;
const CONTINUATION_QUANTILE_BLEND: f64 = 0.20;

thread_local! {
    static CACHE: RefCell<Option<EpisodeCache>> = RefCell::new(None);
}

struct EpisodeCache {
    seed: [u8; 32],
    dp_schedule: Vec<Vec<f64>>,
    rt_da_bias: Vec<f64>,
    bias_caps: Vec<f64>,
    last_time_step: Option<usize>,
}

fn get_or_init_cache(challenge: &Challenge, state: &State) -> (Vec<Vec<f64>>, Vec<f64>) {
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if cache.as_ref().map_or(true, |e| {
            e.seed != challenge.seed
                || e.dp_schedule.len() != challenge.network.num_nodes
                || e.rt_da_bias.len() != challenge.network.num_nodes
        }) {
            let node_thresholds = compute_node_thresholds_inner(challenge);
            let (suffix_sell_env, suffix_buy_env) = compute_suffix_price_bounds_inner(challenge);

            let num_steps = challenge.num_steps;
            let num_nodes = challenge.network.num_nodes;
            let mut continuation_curves = vec![vec![0.0f64; 2 * num_steps]; num_nodes];

            for node in 0..num_nodes {
                let num_segments = node_thresholds[node].len().max(1);
                for t in 0..num_steps {
                    let future_t = (t + 1).min(num_steps.saturating_sub(1));
                    let seg = time_segment_index(future_t, num_steps, num_segments);
                    let (charge_q, discharge_q) = node_thresholds[node][seg];

                    let smooth_sell = suffix_sell_env[node][t + 1];
                    let smooth_buy = suffix_buy_env[node][t + 1];

                    let sell = if smooth_sell.is_finite() {
                        (smooth_sell * (1.0 - CONTINUATION_QUANTILE_BLEND)
                            + discharge_q * CONTINUATION_QUANTILE_BLEND)
                            .max(discharge_q)
                    } else {
                        discharge_q
                    };

                    let buy = if smooth_buy.is_finite() {
                        (smooth_buy * (1.0 - CONTINUATION_QUANTILE_BLEND)
                            + charge_q * CONTINUATION_QUANTILE_BLEND)
                            .min(charge_q)
                    } else {
                        charge_q
                    };

                    continuation_curves[node][2 * t] = sell;
                    continuation_curves[node][2 * t + 1] = buy;
                }
            }

            *cache = Some(EpisodeCache {
                seed: challenge.seed,
                dp_schedule: continuation_curves,
                rt_da_bias: vec![0.0; num_nodes],
                bias_caps: compute_bias_caps_inner(challenge),
                last_time_step: None,
            });
        }

        let entry = cache.as_mut().unwrap();

        if entry.last_time_step.map_or(false, |prev| state.time_step < prev) {
            for bias in entry.rt_da_bias.iter_mut() {
                *bias = 0.0;
            }
            entry.last_time_step = None;
        }

        if state.time_step < challenge.market.day_ahead_prices.len()
            && entry.last_time_step != Some(state.time_step)
        {
            let da_row = &challenge.market.day_ahead_prices[state.time_step];
            let obs_nodes = challenge.network.num_nodes
                .min(state.rt_prices.len())
                .min(da_row.len());

            let mut global_obs_sum = 0.0f64;
            let mut global_obs_count = 0usize;
            for node in 0..obs_nodes {
                let obs = state.rt_prices[node] - da_row[node];
                if obs.is_finite() {
                    global_obs_sum += obs;
                    global_obs_count += 1;
                }
            }
            let global_obs = if global_obs_count > 0 {
                global_obs_sum / global_obs_count as f64
            } else {
                0.0
            };

            let alpha = if state.time_step == 0 { 0.28 } else { 0.18 };
            let global_blend = 0.20;

            for node in 0..challenge.network.num_nodes {
                let da = da_row.get(node).copied().unwrap_or(0.0);
                let rt = state.rt_prices.get(node).copied().unwrap_or(da);
                let obs = if da.is_finite() && rt.is_finite() {
                    rt - da
                } else {
                    0.0
                };
                let blended_obs = (1.0 - global_blend) * obs + global_blend * global_obs;
                let next_bias = (1.0 - alpha) * entry.rt_da_bias[node] + alpha * blended_obs;
                let cap = entry.bias_caps.get(node).copied().unwrap_or(4.0).max(1.0);
                entry.rt_da_bias[node] = next_bias.clamp(-cap, cap);
            }

            entry.last_time_step = Some(state.time_step);
        }

        let global_bias = if entry.rt_da_bias.is_empty() {
            0.0
        } else {
            entry.rt_da_bias.iter().copied().sum::<f64>() / entry.rt_da_bias.len() as f64
        };

        let mut continuation_curves = entry.dp_schedule.clone();
        if challenge.num_steps > 0 {
            let current_t = state.time_step.min(challenge.num_steps.saturating_sub(1));
            let idx = 2 * current_t;
            for node in 0..continuation_curves.len() {
                if idx + 1 >= continuation_curves[node].len() {
                    continue;
                }
                let cap = entry.bias_caps.get(node).copied().unwrap_or(4.0).max(1.0);
                let blended_bias =
                    (0.75 * entry.rt_da_bias[node] + 0.25 * global_bias).clamp(-cap, cap);
                let shift = 0.20 * blended_bias;
                continuation_curves[node][idx] += shift;
                continuation_curves[node][idx + 1] += shift;
            }
        }

        let effective_bias = (0..challenge.network.num_nodes)
            .map(|node| {
                let cap = entry.bias_caps.get(node).copied().unwrap_or(4.0).max(1.0);
                (0.75 * entry.rt_da_bias.get(node).copied().unwrap_or(0.0) + 0.25 * global_bias)
                    .clamp(-cap, cap)
            })
            .collect();

        (continuation_curves, effective_bias)
    })
}

fn compute_bias_caps_inner(challenge: &Challenge) -> Vec<f64> {
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let mut caps = vec![4.0f64; num_nodes];

    for node in 0..num_nodes {
        let prices: Vec<f64> = (0..num_steps)
            .map(|t| challenge.market.day_ahead_prices[t][node])
            .collect();
        if prices.is_empty() {
            continue;
        }

        let (charge_q, discharge_q) = quantile_pair_from_prices(&prices);
        let mut min_price = f64::INFINITY;
        let mut max_price = f64::NEG_INFINITY;
        for &price in &prices {
            min_price = min_price.min(price);
            max_price = max_price.max(price);
        }

        let spread = (discharge_q - charge_q).abs();
        let range = (max_price - min_price).abs();
        caps[node] = (0.80 * spread + 0.20 * range).clamp(2.0, 18.0);
    }

    caps
}

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

fn most_violated_line(challenge: &Challenge, action: &[f64], flows: &[f64]) -> Option<Violation> {
    let mut best: Option<(Violation, f64, f64)> = None;

    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let amount = flow.abs() - limit;
        if amount <= tig_challenges::energy_arbitrage::constants::EPS_FLOW * limit {
            continue;
        }

        let signed_direction = flow.signum();
        if signed_direction.abs() <= EPS {
            continue;
        }

        let mut controllable_worsening = 0.0;
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let ptdf_val = challenge.network.ptdf[l][battery.node];
            let signed_contribution = signed_direction * (ptdf_val * action[i]);
            if signed_contribution > EPS {
                controllable_worsening += signed_contribution;
            }
        }

        let difficulty = amount / controllable_worsening.max(EPS);
        let candidate_v = Violation { line: l, flow, amount };

        match best {
            Some((_, best_difficulty, best_amount))
                if difficulty < best_difficulty
                    || (difficulty == best_difficulty && amount <= best_amount) => {}
            _ => best = Some((candidate_v, difficulty, amount)),
        }
    }

    best.map(|(v, _, _)| v)
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, action, &flows).is_none()
}

fn approx_repair_action_value(
    challenge: &Challenge,
    state: &State,
    continuation_curves: &[Vec<f64>],
    adaptive_bias: &[f64],
    i: usize,
    action: f64,
) -> f64 {
    let battery = &challenge.batteries[i];
    let node = battery.node;
    let t = state.time_step.min(challenge.num_steps.saturating_sub(1));

    let mut rt_price = state.rt_prices.get(node).copied().unwrap_or(0.0)
        + 0.15 * adaptive_bias.get(node).copied().unwrap_or(0.0);
    if !rt_price.is_finite() {
        rt_price = 0.0;
    }

    let (raw_future_sell, raw_future_buy) = if node < continuation_curves.len() {
        let curve = &continuation_curves[node];
        let j = 2 * t;
        if j + 1 < curve.len() {
            (curve[j], curve[j + 1])
        } else {
            (rt_price, rt_price)
        }
    } else {
        (rt_price, rt_price)
    };

    let future_sell = if raw_future_sell.is_finite() {
        raw_future_sell
    } else {
        rt_price
    };
    let future_buy = if raw_future_buy.is_finite() {
        raw_future_buy
    } else {
        rt_price
    };

    let eff_c = battery.efficiency_charge.max(EPS);
    let eff_d = battery.efficiency_discharge.max(EPS);
    let soc_now = state
        .socs
        .get(i)
        .copied()
        .unwrap_or(battery.soc_min_mwh)
        .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
    let soc_surplus_mwh = (soc_now - battery.soc_min_mwh).max(0.0);
    let steps_left = challenge.num_steps.saturating_sub(state.time_step).max(1) as f64;
    let required_discharge_mw = soc_surplus_mwh
        / (steps_left * tig_challenges::energy_arbitrage::constants::DELTA_T.max(EPS) * eff_d);
    let liquidation_urgency =
        (required_discharge_mw / battery.power_discharge_mw.max(EPS)).clamp(0.0, 1.0);

    let discharge_edge =
        rt_price * eff_d - future_buy / eff_c + liquidation_urgency * (future_buy.abs() / eff_c);
    let charge_edge = future_sell * eff_c * eff_d
        - rt_price
        - liquidation_urgency * (future_sell.abs() * eff_c * eff_d);

    if action >= 0.0 {
        if discharge_edge.is_finite() {
            discharge_edge * action
        } else {
            0.0
        }
    } else if charge_edge.is_finite() {
        charge_edge * (-action)
    } else {
        0.0
    }
}

fn try_bundle_exchange_repair(
    challenge: &Challenge,
    state: &State,
    violation: &Violation,
    action: &mut [f64],
    flows: &[f64],
    continuation_curves: &[Vec<f64>],
    adaptive_bias: &[f64],
    pos_shadows: &[f64],
    neg_shadows: &[f64],
) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS || challenge.batteries.len() <= 1 {
        return false;
    }

    #[derive(Clone, Copy)]
    struct LineMove {
        i: usize,
        dir: f64,
        cap: f64,
        relief_per_mw: f64,
        density: f64,
        contribution: f64,
    }

    let mut worseners: Vec<LineMove> = Vec::new();
    let mut helpers: Vec<LineMove> = Vec::new();

    for i in 0..challenge.batteries.len() {
        let battery = &challenge.batteries[i];
        let node = battery.node;
        let old_u = action[i];
        let (lo, hi) = state.action_bounds[i];
        let ptdf_val = challenge.network.ptdf[line][node];
        if ptdf_val.abs() <= EPS {
            continue;
        }

        if old_u.abs() > EPS && signed_direction * (ptdf_val * old_u) > EPS {
            let dir = -old_u.signum();
            let cap = if dir > 0.0 {
                (hi - old_u).max(0.0)
            } else {
                (old_u - lo).max(0.0)
            }
            .min(old_u.abs());

            if cap > EPS {
                let relief_per_mw = -signed_direction * ptdf_val * dir;
                if relief_per_mw > EPS {
                    let new_u = (old_u + dir * cap).clamp(lo, hi);
                    let applied = new_u - old_u;
                    let relief = -signed_direction * ptdf_val * applied;
                    if relief > EPS {
                        let value_loss = approx_repair_action_value(
                            challenge,
                            state,
                            continuation_curves,
                            adaptive_bias,
                            i,
                            old_u,
                        ) - approx_repair_action_value(
                            challenge,
                            state,
                            continuation_curves,
                            adaptive_bias,
                            i,
                            new_u,
                        );
                        let density = value_loss / relief.max(EPS);
                        if density.is_finite() {
                            worseners.push(LineMove {
                                i,
                                dir,
                                cap,
                                relief_per_mw,
                                density,
                                contribution: signed_direction * ptdf_val * old_u,
                            });
                        }
                    }
                }
            }
        }

        let dir = (-signed_direction * ptdf_val).signum();
        if dir.abs() <= EPS {
            continue;
        }

        let cap = if dir > 0.0 {
            (hi - old_u).max(0.0)
        } else {
            (old_u - lo).max(0.0)
        };
        if cap <= EPS {
            continue;
        }

        let relief_per_mw = -signed_direction * ptdf_val * dir;
        if relief_per_mw <= EPS {
            continue;
        }

        let sample_step = cap.min((violation.amount / relief_per_mw.max(EPS)).max(0.20 * cap));
        if sample_step <= EPS {
            continue;
        }

        let new_u = (old_u + dir * sample_step).clamp(lo, hi);
        let applied = new_u - old_u;
        let relief = -signed_direction * ptdf_val * applied;
        if relief <= EPS {
            continue;
        }

        let value_loss = approx_repair_action_value(
            challenge,
            state,
            continuation_curves,
            adaptive_bias,
            i,
            old_u,
        ) - approx_repair_action_value(
            challenge,
            state,
            continuation_curves,
            adaptive_bias,
            i,
            new_u,
        );
        let density = value_loss / relief.max(EPS);
        if density.is_finite() {
            helpers.push(LineMove {
                i,
                dir,
                cap,
                relief_per_mw,
                density,
                contribution: relief_per_mw * cap,
            });
        }
    }

    if worseners.is_empty() || helpers.is_empty() {
        return false;
    }

    worseners.sort_by(|a, b| {
        b.contribution
            .partial_cmp(&a.contribution)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.density
                    .partial_cmp(&b.density)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    helpers.sort_by(|a, b| {
        a.density
            .partial_cmp(&b.density)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.contribution
                    .partial_cmp(&a.contribution)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    worseners.truncate(if challenge.batteries.len() <= 10 { 3 } else { 4 });
    helpers.truncate(if challenge.batteries.len() <= 10 { 4 } else { 5 });

    let mut best_score = f64::INFINITY;
    let mut best_relief = 0.0f64;
    let mut best_moves: Vec<(usize, f64)> = Vec::new();

    {
        let mut evaluate_bundle =
            |w: LineMove, helper_a: LineMove, helper_b: Option<LineMove>| {
                let max_help_relief = helper_a.relief_per_mw * helper_a.cap
                    + helper_b.map_or(0.0, |h| h.relief_per_mw * h.cap);
                let desired_unwind = (violation.amount / w.relief_per_mw.max(EPS)).min(w.cap);
                let min_unwind =
                    ((violation.amount - max_help_relief).max(0.0) / w.relief_per_mw.max(EPS))
                        .min(w.cap);

                let mut step_candidates = vec![
                    0.20 * desired_unwind,
                    min_unwind.max(0.20 * desired_unwind),
                    ((min_unwind + desired_unwind) * 0.5).min(w.cap),
                    desired_unwind,
                ];
                step_candidates.retain(|x| x.is_finite() && *x > EPS);
                step_candidates.sort_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                step_candidates.dedup_by(|a, b| {
                    (*a - *b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
                });

                for step in step_candidates {
                    let old_w = action[w.i];
                    let (w_lo, w_hi) = state.action_bounds[w.i];
                    let new_w = (old_w + w.dir * step).clamp(w_lo, w_hi);
                    let applied_w = new_w - old_w;
                    let node_w = challenge.batteries[w.i].node;
                    let relief_w =
                        -signed_direction * challenge.network.ptdf[line][node_w] * applied_w;
                    if relief_w <= EPS {
                        continue;
                    }

                    let mut candidate_deltas: Vec<(usize, f64)> = vec![(w.i, applied_w)];
                    let mut remaining = (violation.amount - relief_w).max(0.0);
                    if remaining <= EPS {
                        continue;
                    }

                    let mut local_helpers = vec![helper_a];
                    if let Some(h) = helper_b {
                        local_helpers.push(h);
                    }
                    local_helpers.sort_by(|a, b| {
                        a.density
                            .partial_cmp(&b.density)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    for helper in local_helpers {
                        if remaining <= EPS {
                            break;
                        }

                        let old_u = action[helper.i];
                        let (lo, hi) = state.action_bounds[helper.i];
                        let desired_step =
                            (remaining / helper.relief_per_mw.max(EPS)).min(helper.cap);
                        if desired_step <= EPS {
                            continue;
                        }

                        let new_u = (old_u + helper.dir * desired_step).clamp(lo, hi);
                        let applied = new_u - old_u;
                        let node = challenge.batteries[helper.i].node;
                        let relief =
                            -signed_direction * challenge.network.ptdf[line][node] * applied;
                        if relief <= EPS {
                            continue;
                        }

                        candidate_deltas.push((helper.i, applied));
                        remaining = (remaining - relief).max(0.0);
                    }

                    if candidate_deltas.len() < 2 {
                        continue;
                    }

                    let mut delta_flows = vec![0.0f64; challenge.network.num_lines];
                    let mut value_loss = 0.0f64;

                    for &(idx, delta) in candidate_deltas.iter() {
                        let old_u = action[idx];
                        let new_u = old_u + delta;
                        value_loss += approx_repair_action_value(
                            challenge,
                            state,
                            continuation_curves,
                            adaptive_bias,
                            idx,
                            old_u,
                        ) - approx_repair_action_value(
                            challenge,
                            state,
                            continuation_curves,
                            adaptive_bias,
                            idx,
                            new_u,
                        );

                        let node = challenge.batteries[idx].node;
                        for l in 0..challenge.network.num_lines {
                            delta_flows[l] += challenge.network.ptdf[l][node] * delta;
                        }
                    }

                    let exact_relief = -signed_direction * delta_flows[line];
                    let effective_relief = exact_relief.min(violation.amount);
                    if effective_relief <= EPS {
                        continue;
                    }
                    if effective_relief + EPS < violation.amount
                        && effective_relief < 0.08 * violation.amount.max(EPS)
                    {
                        continue;
                    }

                    let mut collateral = 0.0f64;
                    for l in 0..challenge.network.num_lines {
                        if l == line {
                            continue;
                        }

                        let flow_dir = flows[l].signum();
                        if flow_dir.abs() <= EPS {
                            continue;
                        }

                        let delta_flow = delta_flows[l];
                        if delta_flow.abs() <= EPS {
                            continue;
                        }

                        let active_shadow = if flow_dir >= 0.0 {
                            pos_shadows[l]
                        } else {
                            neg_shadows[l]
                        };
                        let move_shadow =
                            directional_shadow_for_line_delta(pos_shadows, neg_shadows, l, delta_flow);
                        let signed_impact = flow_dir * delta_flow;
                        if signed_impact > EPS {
                            collateral += 1.40 * move_shadow.max(active_shadow) * signed_impact;
                        } else if signed_impact < -EPS {
                            collateral -= 0.85 * active_shadow * (-signed_impact);
                        }

                        let new_flow = flows[l] + delta_flow;
                        let spill =
                            (new_flow.abs() - challenge.network.flow_limits[l].abs()).max(0.0);
                        if spill > EPS {
                            collateral += 2.0 * spill;
                        }
                    }

                    let remaining_frac = ((violation.amount - effective_relief).max(0.0)
                        / violation.amount.max(EPS))
                    .clamp(0.0, 1.0);
                    let score =
                        (value_loss + collateral) / effective_relief.max(EPS) + 0.40 * remaining_frac;

                    if score < best_score - 1e-12
                        || ((score - best_score).abs() <= 1e-12
                            && exact_relief > best_relief + EPS)
                    {
                        best_score = score;
                        best_relief = exact_relief;
                        best_moves = candidate_deltas;
                    }
                }
            };

        for &w in worseners.iter() {
            let local_helpers: Vec<LineMove> = helpers
                .iter()
                .copied()
                .filter(|h| h.i != w.i)
                .collect();

            for a in 0..local_helpers.len() {
                evaluate_bundle(w, local_helpers[a], None);
                for b in (a + 1)..local_helpers.len() {
                    evaluate_bundle(w, local_helpers[a], Some(local_helpers[b]));
                }
            }
        }
    }

    if best_moves.is_empty() {
        return false;
    }

    for (i, delta) in best_moves.into_iter() {
        let old_u = action[i];
        let (lo, hi) = state.action_bounds[i];
        action[i] = (old_u + delta).clamp(lo, hi);
    }

    true
}

fn soften_most_violated_line(challenge: &Challenge, state: &State, violation: &Violation, action: &mut [f64]) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    let flows = compute_flows(challenge, state, action);
    let (continuation_curves, adaptive_bias) = get_or_init_cache(challenge, state);
    let (future_pos_shadows, future_neg_shadows) =
        compute_future_opportunity_shadows(challenge, state, &continuation_curves, &adaptive_bias);
    let (pos_shadows, neg_shadows) = directional_line_shadow_prices(
        challenge,
        &flows,
        &future_pos_shadows,
        &future_neg_shadows,
    );

    #[derive(Clone, Copy)]
    struct RepairOption {
        i: usize,
        dir: f64,
        relief_per_mw: f64,
        density: f64,
        is_compensating: bool,
    }

    let mut deletion_opts: Vec<RepairOption> = Vec::new();
    let mut compensating_opts: Vec<RepairOption> = Vec::new();

    {
        for i in 0..challenge.batteries.len() {
            let battery = &challenge.batteries[i];
            let node = battery.node;
            let old_u = action[i];
            let (lo, hi) = state.action_bounds[i];
            let ptdf_val = challenge.network.ptdf[line][node];
            if ptdf_val.abs() <= EPS {
                continue;
            }

            let worsening = signed_direction * (ptdf_val * old_u) > EPS;

            let maybe_push = |options: &mut Vec<RepairOption>, dir: f64, mut cap: f64, is_compensating: bool| {
                if dir.abs() <= EPS || cap <= EPS {
                    return;
                }

                if is_compensating {
                    let bound_cap = if dir > 0.0 {
                        (hi - old_u).max(0.0)
                    } else {
                        (old_u - lo).max(0.0)
                    };
                    cap = cap.min(bound_cap);

                    if old_u * dir < -EPS {
                        cap = cap.min(old_u.abs());
                    } else if old_u.abs() <= EPS {
                        let directional_power_cap = if dir > 0.0 {
                            battery.power_discharge_mw.max(EPS)
                        } else {
                            battery.power_charge_mw.max(EPS)
                        };
                        cap = cap.min(0.60 * directional_power_cap);
                    }
                } else {
                    cap = cap.min(old_u.abs());
                }

                if cap <= EPS {
                    return;
                }

                let relief_per_mw = -signed_direction * ptdf_val * dir;
                if relief_per_mw <= EPS {
                    return;
                }

                cap = cap.min((violation.amount / relief_per_mw.max(EPS)).max(EPS));
                if cap <= EPS {
                    return;
                }

                let new_u = (old_u + dir * cap).clamp(lo, hi);
                let applied = new_u - old_u;
                let relief = -signed_direction * ptdf_val * applied;
                if relief <= EPS {
                    return;
                }

                let value_impact = approx_repair_action_value(
                    challenge,
                    state,
                    &continuation_curves,
                    &adaptive_bias,
                    i,
                    old_u,
                ) - approx_repair_action_value(
                    challenge,
                    state,
                    &continuation_curves,
                    &adaptive_bias,
                    i,
                    new_u,
                );

                let mut collateral = 0.0f64;
                for l in 0..challenge.network.num_lines {
                    if l == line {
                        continue;
                    }

                    let flow_dir = flows[l].signum();
                    if flow_dir.abs() <= EPS {
                        continue;
                    }

                    let active_shadow = if flow_dir >= 0.0 {
                        pos_shadows[l]
                    } else {
                        neg_shadows[l]
                    };
                    if active_shadow <= EPS {
                        continue;
                    }

                    let delta_flow = challenge.network.ptdf[l][node] * applied;
                    let signed_impact = flow_dir * delta_flow;
                    if signed_impact > EPS {
                        let move_shadow =
                            directional_shadow_for_line_delta(&pos_shadows, &neg_shadows, l, delta_flow);
                        collateral += 1.30 * move_shadow.max(active_shadow) * signed_impact;
                    } else if signed_impact < -EPS {
                        collateral -= 0.80 * active_shadow * (-signed_impact);
                    }
                }

                let mut density = (value_impact + collateral) / relief;
                if !density.is_finite() {
                    return;
                }

                if is_compensating && old_u * dir > EPS {
                    density *= 0.95;
                } else if is_compensating && old_u.abs() <= EPS {
                    density *= 0.98;
                }

                options.push(RepairOption {
                    i,
                    dir,
                    relief_per_mw,
                    density,
                    is_compensating,
                });
            };

            if worsening && old_u.abs() > EPS {
                maybe_push(&mut deletion_opts, -old_u.signum(), old_u.abs(), false);
            }

            let compensate_dir = (-signed_direction * ptdf_val).signum();
            if compensate_dir.abs() > EPS && (!worsening || old_u.signum() == compensate_dir || old_u.abs() <= EPS) {
                let raw_cap = if compensate_dir > 0.0 {
                    (hi - old_u).max(0.0)
                } else {
                    (old_u - lo).max(0.0)
                };
                maybe_push(&mut compensating_opts, compensate_dir, raw_cap, true);
            }
        }
    }

    if deletion_opts.is_empty() && compensating_opts.is_empty() {
        return false;
    }

    compensating_opts.sort_by(|a, b| {
        a.density
            .partial_cmp(&b.density)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_compensating = if challenge.batteries.len() <= 8 {
        4
    } else if challenge.batteries.len() <= 24 {
        6
    } else {
        8
    };
    compensating_opts.truncate(max_compensating);

    let mut options = deletion_opts;
    options.extend(compensating_opts);
    options.sort_by(|a, b| {
        a.density
            .partial_cmp(&b.density)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut remaining_violation = violation.amount;
    let mut changed = false;

    for opt in options {
        if remaining_violation <= EPS {
            break;
        }

        let i = opt.i;
        let old_u = action[i];
        let (lo, hi) = state.action_bounds[i];
        let node = challenge.batteries[i].node;
        let ptdf_val = challenge.network.ptdf[line][node];

        let mut cap = if opt.is_compensating {
            if opt.dir > 0.0 {
                (hi - old_u).max(0.0)
            } else {
                (old_u - lo).max(0.0)
            }
        } else {
            old_u.abs()
        };

        if opt.is_compensating && old_u * opt.dir < -EPS {
            cap = cap.min(old_u.abs());
        }
        if cap <= EPS {
            continue;
        }

        let step = cap.min(remaining_violation / opt.relief_per_mw.max(EPS));
        if step <= EPS {
            continue;
        }

        let new_u = (old_u + opt.dir * step).clamp(lo, hi);
        let applied = new_u - old_u;
        let relief = -signed_direction * ptdf_val * applied;
        if relief <= EPS {
            continue;
        }

        action[i] = new_u;
        remaining_violation -= relief;
        changed = true;
    }

    let progress = 1.0 - remaining_violation / violation.amount.max(EPS);
    if remaining_violation > EPS && (!changed || progress < 0.70) {
        let bundle_flows = compute_flows(challenge, state, action);
        let bundle_amount = (bundle_flows[line].abs() - challenge.network.flow_limits[line]).max(0.0);

        if bundle_amount
            > tig_challenges::energy_arbitrage::constants::EPS_FLOW
                * challenge.network.flow_limits[line].abs().max(EPS)
        {
            let bundle_violation = Violation {
                line,
                flow: bundle_flows[line],
                amount: bundle_amount,
            };
            let (bundle_pos_shadows, bundle_neg_shadows) = directional_line_shadow_prices(
                challenge,
                &bundle_flows,
                &future_pos_shadows,
                &future_neg_shadows,
            );

            if try_bundle_exchange_repair(
                challenge,
                state,
                &bundle_violation,
                action,
                &bundle_flows,
                &continuation_curves,
                &adaptive_bias,
                &bundle_pos_shadows,
                &bundle_neg_shadows,
            ) {
                changed = true;
            }
        }
    }

    changed
}

fn enforce_flow_feasibility(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Result<Vec<f64>> {
    let min_valid: Vec<f64> = state.action_bounds.iter()
        .map(|&(lo, hi)| 0.0f64.clamp(lo, hi))
        .collect();

    let mut flows = compute_flows(challenge, state, &action);

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(violation) = most_violated_line(challenge, &action, &flows) else {
            return Ok(action);
        };
        let old_action = action.clone();
        if !soften_most_violated_line(challenge, state, &violation, &mut action) {
            break;
        }
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let delta = action[i] - old_action[i];
            if delta.abs() > EPS {
                let node = battery.node;
                for l in 0..challenge.network.num_lines {
                    flows[l] += challenge.network.ptdf[l][node] * delta;
                }
            }
            let (lo, hi) = state.action_bounds[i];
            action[i] = action[i].clamp(lo, hi);
        }
    }

    if most_violated_line(challenge, &action, &flows).is_none() {
        return Ok(action);
    }

    if !is_flow_feasible(challenge, state, &min_valid) {
        return Err(anyhow!("Grid infeasible even with minimum battery actions"));
    }

    let mut projected = action;
    let mut proj_flows = compute_flows(challenge, state, &projected);

    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(violation) = most_violated_line(challenge, &projected, &proj_flows) else {
            return Ok(projected);
        };

        let line = violation.line;
        let signed_direction = violation.flow.signum();
        if signed_direction.abs() <= EPS { break; }

        let mut total_worsening = 0.0;
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let ptdf_val = challenge.network.ptdf[line][battery.node];
            let signed_contribution = signed_direction * (ptdf_val * projected[i]);
            if signed_contribution > EPS {
                total_worsening += signed_contribution;
            }
        }
        if total_worsening <= EPS { break; }

        let frac = (violation.amount / total_worsening).clamp(0.0, 1.0);

        for (i, battery) in challenge.batteries.iter().enumerate() {
            let ptdf_val = challenge.network.ptdf[line][battery.node];
            let signed_contribution = signed_direction * (ptdf_val * projected[i]);
            if signed_contribution > EPS {
                let old = projected[i];
                projected[i] *= 1.0 - frac;
                let (lo, hi) = state.action_bounds[i];
                projected[i] = projected[i].clamp(lo, hi);
                let delta = projected[i] - old;
                if delta.abs() > EPS {
                    for l in 0..challenge.network.num_lines {
                        proj_flows[l] += challenge.network.ptdf[l][battery.node] * delta;
                    }
                }
            }
        }
    }

    if most_violated_line(challenge, &projected, &proj_flows).is_none() {
        return Ok(projected);
    }

    Ok(min_valid)
}

fn compute_node_thresholds_inner(challenge: &Challenge) -> Vec<Vec<(f64, f64)>> {
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let num_segments = if num_steps >= 9 { NUM_TIME_SEGMENTS } else { 1 };
    let mut thresholds = vec![vec![(0.0f64, 0.0f64); num_segments]; num_nodes];

    for node in 0..num_nodes {
        let prices: Vec<f64> = (0..num_steps)
            .map(|t| challenge.market.day_ahead_prices[t][node])
            .collect();
        let global_pair = quantile_pair_from_prices(&prices);

        for seg in 0..num_segments {
            let start = seg * num_steps / num_segments;
            let end = ((seg + 1) * num_steps / num_segments).min(num_steps);

            thresholds[node][seg] = if end > start {
                let local_pair = quantile_pair_from_prices(&prices[start..end]);
                let seg_len = end - start;
                let local_weight = if num_segments == 1 {
                    1.0
                } else if seg_len >= 6 {
                    0.75
                } else if seg_len >= 3 {
                    0.55
                } else {
                    0.35
                };

                (
                    global_pair.0 * (1.0 - local_weight) + local_pair.0 * local_weight,
                    global_pair.1 * (1.0 - local_weight) + local_pair.1 * local_weight,
                )
            } else {
                global_pair
            };
        }
    }

    thresholds
}

fn time_segment_index(t: usize, num_steps: usize, num_segments: usize) -> usize {
    if num_steps <= 1 || num_segments <= 1 {
        0
    } else {
        (t * num_segments / num_steps).min(num_segments - 1)
    }
}

fn quantile_pair_from_prices(prices: &[f64]) -> (f64, f64) {
    if prices.is_empty() {
        return (0.0, 0.0);
    }

    let mut sorted = prices.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let ci = ((n as f64 * CHARGE_QUANTILE) as usize).min(n.saturating_sub(1));
    let di = ((n as f64 * DISCHARGE_QUANTILE) as usize).min(n.saturating_sub(1));
    (sorted[ci], sorted[di])
}

fn insert_sorted_limited(values: &mut Vec<f64>, value: f64, limit: usize, descending: bool, sum: &mut f64) {
    if limit == 0 {
        return;
    }

    let mut pos = values.len();
    for idx in 0..values.len() {
        let better = if descending {
            value > values[idx]
        } else {
            value < values[idx]
        };
        if better {
            pos = idx;
            break;
        }
    }

    if pos < limit {
        values.insert(pos, value);
        *sum += value;
        if values.len() > limit {
            if let Some(removed) = values.pop() {
                *sum -= removed;
            }
        }
    } else if values.len() < limit {
        values.push(value);
        *sum += value;
    }
}

fn compute_suffix_price_bounds_inner(challenge: &Challenge) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_steps = challenge.num_steps;
    let num_nodes = challenge.network.num_nodes;
    let k = CONTINUATION_TAIL_K.min(num_steps.max(1));

    let mut suffix_sell = vec![vec![f64::NEG_INFINITY; num_steps + 1]; num_nodes];
    let mut suffix_buy = vec![vec![f64::INFINITY; num_steps + 1]; num_nodes];

    for node in 0..num_nodes {
        let mut suffix_max = f64::NEG_INFINITY;
        let mut suffix_min = f64::INFINITY;
        let mut top_vals: Vec<f64> = Vec::with_capacity(k);
        let mut bottom_vals: Vec<f64> = Vec::with_capacity(k);
        let mut top_sum = 0.0f64;
        let mut bottom_sum = 0.0f64;

        for t in (0..num_steps).rev() {
            let p = challenge.market.day_ahead_prices[t][node];
            suffix_max = suffix_max.max(p);
            suffix_min = suffix_min.min(p);

            insert_sorted_limited(&mut top_vals, p, k, true, &mut top_sum);
            insert_sorted_limited(&mut bottom_vals, p, k, false, &mut bottom_sum);

            let top_avg = top_sum / top_vals.len() as f64;
            let bottom_avg = bottom_sum / bottom_vals.len() as f64;

            suffix_sell[node][t] =
                top_avg + CONTINUATION_EXTREME_BLEND * (suffix_max - top_avg);
            suffix_buy[node][t] =
                bottom_avg + CONTINUATION_EXTREME_BLEND * (suffix_min - bottom_avg);
        }
    }

    (suffix_sell, suffix_buy)
}

fn compute_proactive_bounds(
    challenge: &Challenge,
    state: &State,
    baseline_flows: &[f64],
    preferred_dirs: &[f64],
) -> Vec<(f64, f64)> {
    let mut bounds: Vec<(f64, f64)> = state.action_bounds.clone();
    let eps_flow = tig_challenges::energy_arbitrage::constants::EPS_FLOW;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (mut lo, mut hi) = bounds[i];

        for l in 0..challenge.network.num_lines {
            let p = challenge.network.ptdf[l][node];
            if p.abs() <= EPS {
                continue;
            }

            let limit = challenge.network.flow_limits[l] * (1.0 + eps_flow);
            let f0 = baseline_flows[l];

            let mut lo_u = (-limit - f0) / p;
            let mut hi_u = (limit - f0) / p;
            if lo_u > hi_u {
                let tmp = lo_u;
                lo_u = hi_u;
                hi_u = tmp;
            }

            if lo_u > lo {
                lo = lo_u;
            }
            if hi_u < hi {
                hi = hi_u;
            }

            if hi < lo {
                let mid = 0.0f64.clamp(lo.min(hi), lo.max(hi));
                lo = mid;
                hi = mid;
                break;
            }
        }

        bounds[i] = (lo, hi);
    }

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let preferred_dir = preferred_dirs.get(i).copied().unwrap_or(0.0).signum();
        if preferred_dir.abs() <= EPS {
            continue;
        }

        let node = battery.node;
        let (mut lo, mut hi) = bounds[i];
        let mut soft_mag = if preferred_dir > 0.0 {
            hi.max(0.0)
        } else {
            (-lo).max(0.0)
        };
        if soft_mag <= EPS {
            continue;
        }

        for l in 0..challenge.network.num_lines {
            let limit_abs = challenge.network.flow_limits[l].abs().max(EPS);
            let util = baseline_flows[l].abs() / limit_abs;
            if util <= 0.72 {
                continue;
            }

            let signed_direction = baseline_flows[l].signum();
            if signed_direction.abs() <= EPS {
                continue;
            }

            let worsen_coeff = signed_direction * challenge.network.ptdf[l][node] * preferred_dir;
            if worsen_coeff <= EPS {
                continue;
            }

            let headroom = (limit_abs * (1.0 + eps_flow) - baseline_flows[l].abs()).max(0.0);
            let exact_cap = headroom / worsen_coeff.max(EPS);
            let stress = ((util - 0.72) / 0.28).clamp(0.0, 1.5);
            let soft_scale = (0.90 - 0.55 * stress).clamp(0.25, 0.90);
            soft_mag = soft_mag.min(exact_cap.max(0.0) * soft_scale);

            if soft_mag <= EPS {
                soft_mag = 0.0;
                break;
            }
        }

        if preferred_dir > 0.0 {
            hi = hi.min(soft_mag.max(0.0));
        } else {
            lo = lo.max(-soft_mag.max(0.0));
        }

        if hi < lo {
            let mid = 0.0f64.clamp(lo.min(hi), lo.max(hi));
            lo = mid;
            hi = mid;
        }

        bounds[i] = (lo, hi);
    }

    bounds
}

fn directional_shadow_from_signed_util(signed_util: f64) -> f64 {
    if signed_util <= 0.78 {
        0.0
    } else {
        let x = ((signed_util - 0.78) / 0.22).clamp(0.0, 2.0);
        x * x
    }
}

fn future_opportunity_shadow_from_util(util: f64) -> f64 {
    if util <= 0.30 {
        0.0
    } else {
        let x = ((util - 0.30) / 0.70).clamp(0.0, 2.0);
        0.90 * x * x
    }
}

fn compute_future_opportunity_shadows(
    challenge: &Challenge,
    state: &State,
    continuation_curves: &[Vec<f64>],
    adaptive_bias: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let num_lines = challenge.network.num_lines;
    let mut pos_demand = vec![0.0f64; num_lines];
    let mut neg_demand = vec![0.0f64; num_lines];

    if num_lines == 0 || challenge.num_steps == 0 {
        return (pos_demand, neg_demand);
    }

    let t = state.time_step.min(challenge.num_steps.saturating_sub(1));
    let steps_remaining = challenge.num_steps.saturating_sub(t);
    if steps_remaining <= 1 {
        return (pos_demand, neg_demand);
    }

    let da_row = challenge.market.day_ahead_prices.get(t);
    let delta_t = tig_challenges::energy_arbitrage::constants::DELTA_T.max(EPS);
    let horizon_weight =
        ((steps_remaining - 1) as f64 / challenge.num_steps.max(1) as f64).clamp(0.0, 1.0);

    for i in 0..challenge.batteries.len() {
        let battery = &challenge.batteries[i];
        let node = battery.node;
        if node >= continuation_curves.len() {
            continue;
        }

        let curve = &continuation_curves[node];
        let j = 2 * t;
        if j + 1 >= curve.len() {
            continue;
        }

        let future_sell = curve[j];
        let future_buy = curve[j + 1];
        if !future_sell.is_finite() || !future_buy.is_finite() {
            continue;
        }

        let spot = da_row
            .and_then(|row| row.get(node))
            .copied()
            .unwrap_or(0.0)
            + 0.35 * adaptive_bias.get(node).copied().unwrap_or(0.0);

        let eff_c = battery.efficiency_charge.max(EPS);
        let eff_d = battery.efficiency_discharge.max(EPS);
        let soc_now = state
            .socs
            .get(i)
            .copied()
            .unwrap_or(battery.soc_min_mwh)
            .clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = ((soc_now - battery.soc_min_mwh) / soc_range).clamp(0.0, 1.0);
        let soc_surplus_mwh = (soc_now - battery.soc_min_mwh).max(0.0);
        let soc_room_mwh = (battery.soc_max_mwh - soc_now).max(0.0);

        let discharge_cap_mw = battery
            .power_discharge_mw
            .min(soc_surplus_mwh / (delta_t * eff_d))
            .max(0.0);
        let charge_cap_mw = battery
            .power_charge_mw
            .min(soc_room_mwh / (delta_t * eff_c))
            .max(0.0);

        if discharge_cap_mw <= EPS && charge_cap_mw <= EPS {
            continue;
        }

        let steps_left = steps_remaining.max(1) as f64;
        let required_discharge_mw = soc_surplus_mwh / (steps_left * delta_t * eff_d);
        let liquidation_urgency =
            (required_discharge_mw / battery.power_discharge_mw.max(EPS)).clamp(0.0, 1.0);

        let cycle_edge = (future_sell * eff_c * eff_d - future_buy).max(0.0);
        let hold_discharge_edge = ((future_sell - spot).max(0.0)) * eff_d;
        let wait_charge_edge = (spot - future_buy).max(0.0);
        let price_scale = (future_sell.abs() + future_buy.abs() + spot.abs()).max(8.0);

        let discharge_strength = ((0.65 * cycle_edge + 0.35 * hold_discharge_edge) / price_scale)
            .clamp(0.0, 1.0)
            .max(0.35 * liquidation_urgency);
        let charge_strength = ((0.70 * cycle_edge + 0.30 * wait_charge_edge) / price_scale)
            .clamp(0.0, 1.0);

        let discharge_pressure =
            (0.55 * soc_level + 0.45 * liquidation_urgency).clamp(0.0, 1.0);
        let charge_pressure = ((1.0 - soc_level) * (1.0 - 0.55 * liquidation_urgency))
            .clamp(0.0, 1.0);

        let future_discharge_mw = 0.60
            * horizon_weight
            * discharge_cap_mw
            * discharge_pressure
            * (0.25 + 0.75 * discharge_strength);
        let future_charge_mw = if steps_remaining > 2 {
            0.55
                * horizon_weight
                * charge_cap_mw
                * charge_pressure
                * (0.20 + 0.80 * charge_strength)
        } else {
            0.0
        };

        for l in 0..num_lines {
            let p = challenge.network.ptdf[l][node];
            if p.abs() <= EPS {
                continue;
            }

            let discharge_flow = p * future_discharge_mw;
            if discharge_flow > EPS {
                pos_demand[l] += discharge_flow;
            } else if discharge_flow < -EPS {
                neg_demand[l] += -discharge_flow;
            }

            let charge_flow = -p * future_charge_mw;
            if charge_flow > EPS {
                pos_demand[l] += charge_flow;
            } else if charge_flow < -EPS {
                neg_demand[l] += -charge_flow;
            }
        }
    }

    let mut pos = vec![0.0f64; num_lines];
    let mut neg = vec![0.0f64; num_lines];
    for l in 0..num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        pos[l] = future_opportunity_shadow_from_util((pos_demand[l] / limit).clamp(0.0, 4.0));
        neg[l] = future_opportunity_shadow_from_util((neg_demand[l] / limit).clamp(0.0, 4.0));
    }

    (pos, neg)
}

fn directional_line_shadow_prices(
    challenge: &Challenge,
    flows: &[f64],
    future_pos_shadows: &[f64],
    future_neg_shadows: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut pos = vec![0.0f64; challenge.network.num_lines];
    let mut neg = vec![0.0f64; challenge.network.num_lines];

    for l in 0..challenge.network.num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        let util = (flows[l] / limit).clamp(-2.0, 2.0);
        let current_pos = directional_shadow_from_signed_util(util);
        let current_neg = directional_shadow_from_signed_util(-util);
        let future_pos = future_pos_shadows.get(l).copied().unwrap_or(0.0).max(0.0);
        let future_neg = future_neg_shadows.get(l).copied().unwrap_or(0.0).max(0.0);

        pos[l] = (current_pos + (0.45 + 0.25 * current_pos.min(1.0)) * future_pos).clamp(0.0, 4.0);
        neg[l] = (current_neg + (0.45 + 0.25 * current_neg.min(1.0)) * future_neg).clamp(0.0, 4.0);
    }

    (pos, neg)
}

fn directional_shadow_for_line_delta(
    pos_shadows: &[f64],
    neg_shadows: &[f64],
    line: usize,
    delta_flow: f64,
) -> f64 {
    if delta_flow > EPS {
        pos_shadows[line]
    } else if delta_flow < -EPS {
        neg_shadows[line]
    } else {
        0.0
    }
}

fn directional_shadow_cost_for_node(
    challenge: &Challenge,
    pos_shadows: &[f64],
    neg_shadows: &[f64],
    node: usize,
    dir: f64,
) -> f64 {
    if dir.abs() <= EPS {
        return 0.0;
    }

    let mut cost = 0.0f64;
    for l in 0..challenge.network.num_lines {
        let p = challenge.network.ptdf[l][node];
        let delta_flow = p * dir;
        let shadow = directional_shadow_for_line_delta(pos_shadows, neg_shadows, l, delta_flow);
        if shadow > EPS {
            cost += shadow * p.abs();
        }
    }

    cost
}

fn line_stress_weights(
    challenge: &Challenge,
    flows: &[f64],
    future_pos_shadows: &[f64],
    future_neg_shadows: &[f64],
) -> Vec<f64> {
    let (pos_shadows, neg_shadows) = directional_line_shadow_prices(
        challenge,
        flows,
        future_pos_shadows,
        future_neg_shadows,
    );
    (0..challenge.network.num_lines)
        .map(|l| {
            if flows[l] >= 0.0 {
                pos_shadows[l]
            } else {
                neg_shadows[l]
            }
        })
        .collect()
}

fn feasible_scale_for_delta(challenge: &Challenge, flows: &[f64], node: usize, delta: f64) -> f64 {
    if delta.abs() <= EPS {
        return 0.0;
    }

    let eps_flow = tig_challenges::energy_arbitrage::constants::EPS_FLOW;
    let mut s_low = 0.0f64;
    let mut s_high = 1.0f64;

    for l in 0..challenge.network.num_lines {
        let a = challenge.network.ptdf[l][node] * delta;
        if a.abs() <= EPS {
            continue;
        }

        let limit = challenge.network.flow_limits[l] * (1.0 + eps_flow);
        let f0 = flows[l];

        let mut lo_s = (-limit - f0) / a;
        let mut hi_s = (limit - f0) / a;
        if lo_s > hi_s {
            let tmp = lo_s;
            lo_s = hi_s;
            hi_s = tmp;
        }

        if lo_s > s_low {
            s_low = lo_s;
        }
        if hi_s < s_high {
            s_high = hi_s;
        }
        if s_high < s_low {
            return 0.0;
        }
    }

    s_high.clamp(0.0, 1.0)
}

fn feasible_pair_step(
    challenge: &Challenge,
    flows: &[f64],
    node_a: usize,
    delta_a: f64,
    node_b: usize,
    delta_b: f64,
    max_step: f64,
) -> f64 {
    if max_step <= EPS || (delta_a.abs() <= EPS && delta_b.abs() <= EPS) {
        return 0.0;
    }

    let mut s_low = 0.0f64;
    let mut s_high = max_step;
    let eps_flow = tig_challenges::energy_arbitrage::constants::EPS_FLOW;

    for l in 0..challenge.network.num_lines {
        let a = challenge.network.ptdf[l][node_a] * delta_a
            + challenge.network.ptdf[l][node_b] * delta_b;
        if a.abs() <= EPS {
            continue;
        }

        let limit = challenge.network.flow_limits[l] * (1.0 + eps_flow);
        let f0 = flows[l];

        let mut lo_s = (-limit - f0) / a;
        let mut hi_s = (limit - f0) / a;
        if lo_s > hi_s {
            let tmp = lo_s;
            lo_s = hi_s;
            hi_s = tmp;
        }

        if lo_s > s_low {
            s_low = lo_s;
        }
        if hi_s < s_high {
            s_high = hi_s;
        }
        if s_high < s_low {
            return 0.0;
        }
    }

    s_high.clamp(0.0, max_step)
}

fn apply_counterflow_preconditioning(
    challenge: &Challenge,
    flows: &mut [f64],
    action: &mut [f64],
    proactive_bounds: &[(f64, f64)],
    targets: &[f64],
    values: &[f64],
    discharge_edges: &[f64],
    charge_edges: &[f64],
    future_pos_shadows: &[f64],
    future_neg_shadows: &[f64],
) {
    let num_batteries = challenge.batteries.len();
    let num_lines = challenge.network.num_lines;
    if num_batteries <= 1 || num_lines == 0 {
        return;
    }

    let mut max_util = 0.0f64;
    for l in 0..num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        max_util = max_util.max(flows[l].abs() / limit);
    }
    if max_util < 0.80 {
        return;
    }

    let max_pre_moves = if max_util > 0.92 { 2 } else { 1 };

    #[derive(Clone, Copy)]
    struct Consumer {
        i: usize,
        need: f64,
        density: f64,
    }

    for _ in 0..max_pre_moves {
        let (pos_shadows, neg_shadows) = directional_line_shadow_prices(
            challenge,
            flows,
            future_pos_shadows,
            future_neg_shadows,
        );
        let active_stress: Vec<f64> = (0..num_lines)
            .map(|l| {
                if flows[l] >= 0.0 {
                    pos_shadows[l]
                } else {
                    neg_shadows[l]
                }
            })
            .collect();
        let mut stressed_lines: Vec<usize> = (0..num_lines).collect();
        stressed_lines.sort_by(|&a, &b| {
            let util_a = flows[a].abs() / challenge.network.flow_limits[a].abs().max(EPS);
            let util_b = flows[b].abs() / challenge.network.flow_limits[b].abs().max(EPS);
            util_b
                .partial_cmp(&util_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stressed_lines.retain(|&l| {
            let limit = challenge.network.flow_limits[l].abs().max(EPS);
            let util = flows[l].abs() / limit;
            util > 0.78 || active_stress[l] > EPS
        });
        stressed_lines.truncate(if max_util > 0.90 { 3 } else { 2 });

        if stressed_lines.is_empty() {
            break;
        }

        let mut best_move: Option<(usize, f64)> = None;
        let mut best_score = EPS;

        for &line in stressed_lines.iter() {
            let limit_abs = challenge.network.flow_limits[line].abs().max(EPS);
            let util = flows[line].abs() / limit_abs;
            let signed_direction = flows[line].signum();
            if signed_direction.abs() <= EPS {
                continue;
            }

            let headroom = (limit_abs
                * (1.0 + tig_challenges::energy_arbitrage::constants::EPS_FLOW)
                - flows[line].abs())
            .max(0.0);

            let mut consumers: Vec<Consumer> = Vec::new();
            let mut blocked_need = 0.0f64;

            for i in 0..num_batteries {
                let residual = targets[i] - action[i];
                if residual.abs() <= EPS || values[i] <= EPS {
                    continue;
                }

                let node = challenge.batteries[i].node;
                let worsen = signed_direction * challenge.network.ptdf[line][node] * residual.signum();
                if worsen <= EPS {
                    continue;
                }

                let need = worsen * residual.abs();
                blocked_need += need;
                consumers.push(Consumer {
                    i,
                    need,
                    density: values[i] / worsen.max(EPS),
                });
            }

            if consumers.is_empty() {
                continue;
            }
            if blocked_need <= headroom * 1.10 && util < 0.90 {
                continue;
            }

            consumers.sort_by(|a, b| {
                b.density
                    .partial_cmp(&a.density)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_density = consumers[0].density.max(EPS);
            let relief_goal = if blocked_need > headroom {
                (blocked_need - headroom).clamp(0.03 * limit_abs, 0.18 * limit_abs)
            } else {
                (0.08 * limit_abs).min(0.5 * blocked_need.max(0.0) + 0.02 * limit_abs)
            };
            if relief_goal <= EPS {
                continue;
            }

            for j in 0..num_batteries {
                let node = challenge.batteries[j].node;
                let residual_j = targets[j] - action[j];

                for &dir in &[-1.0f64, 1.0f64] {
                    let relief_per_mw = -signed_direction * challenge.network.ptdf[line][node] * dir;
                    if relief_per_mw <= EPS {
                        continue;
                    }

                    let (lo, hi) = proactive_bounds[j];
                    let raw_cap = if dir > 0.0 {
                        (hi - action[j]).max(0.0)
                    } else {
                        (action[j] - lo).max(0.0)
                    };
                    if raw_cap <= EPS {
                        continue;
                    }

                    let dir_edge = if dir > 0.0 {
                        discharge_edges.get(j).copied().unwrap_or(f64::NEG_INFINITY)
                    } else {
                        charge_edges.get(j).copied().unwrap_or(f64::NEG_INFINITY)
                    };

                    let along_target = residual_j.abs() > EPS && residual_j.signum() == dir;
                    let mut step_cap = raw_cap;
                    if along_target {
                        step_cap = step_cap.min(residual_j.abs());
                    } else {
                        step_cap = step_cap.min(0.35 * raw_cap);
                    }
                    step_cap = step_cap.min(relief_goal / relief_per_mw.max(EPS));
                    step_cap = step_cap.min(0.15 * limit_abs / relief_per_mw.max(EPS));
                    if step_cap <= EPS {
                        continue;
                    }

                    if !along_target && dir_edge < -0.90 * top_density {
                        continue;
                    }

                    let delta_try = dir * step_cap;
                    let scale = feasible_scale_for_delta(challenge, flows, node, delta_try);
                    if scale <= EPS {
                        continue;
                    }
                    let applied_delta = delta_try * scale;
                    if applied_delta.abs() <= EPS {
                        continue;
                    }

                    let relief_created =
                        -signed_direction * challenge.network.ptdf[line][node] * applied_delta;
                    if relief_created <= EPS {
                        continue;
                    }

                    let mut unlocked_gain = 0.0f64;
                    let mut remaining_relief = relief_created;
                    for consumer in consumers.iter() {
                        if consumer.i == j {
                            continue;
                        }
                        let take = consumer.need.min(remaining_relief);
                        unlocked_gain += take * consumer.density;
                        remaining_relief -= take;
                        if remaining_relief <= EPS {
                            break;
                        }
                    }

                    if unlocked_gain <= 0.55 * top_density * relief_created {
                        continue;
                    }

                    let direct_gain = dir_edge * applied_delta.abs();
                    if unlocked_gain + direct_gain <= EPS {
                        continue;
                    }
                    if unlocked_gain <= (-direct_gain).max(0.0) * 1.20 {
                        continue;
                    }

                    let mut side_risk = 0.0f64;
                    let mut side_relief = 0.0f64;
                    for l in 0..num_lines {
                        if l == line {
                            continue;
                        }

                        let flow_dir = flows[l].signum();
                        if flow_dir.abs() <= EPS {
                            continue;
                        }

                        let limit = challenge.network.flow_limits[l].abs().max(EPS);
                        let delta_flow = challenge.network.ptdf[l][node] * applied_delta;
                        let signed_push = flow_dir * delta_flow / limit;
                        let active_shadow = if flow_dir >= 0.0 {
                            pos_shadows[l]
                        } else {
                            neg_shadows[l]
                        };
                        let move_shadow =
                            directional_shadow_for_line_delta(&pos_shadows, &neg_shadows, l, delta_flow);

                        if signed_push > EPS {
                            side_risk += move_shadow * signed_push;
                        } else if signed_push < -EPS {
                            side_relief += active_shadow * (-signed_push);
                        }
                    }

                    let score = unlocked_gain
                        + direct_gain
                        + 0.15 * top_density * side_relief
                        - 0.35 * top_density * side_risk
                        - if along_target {
                            0.0
                        } else {
                            0.05 * top_density * relief_created
                        };

                    if score > best_score {
                        best_score = score;
                        best_move = Some((j, applied_delta));
                    }
                }
            }
        }

        let Some((j, delta)) = best_move else {
            break;
        };

        let old_u = action[j];
        let new_u = (old_u + delta).clamp(proactive_bounds[j].0, proactive_bounds[j].1);
        let applied = new_u - old_u;
        if applied.abs() <= EPS {
            break;
        }

        action[j] = new_u;
        let node = challenge.batteries[j].node;
        for l in 0..num_lines {
            flows[l] += challenge.network.ptdf[l][node] * applied;
        }
    }
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let da_prices = &challenge.market.day_ahead_prices;

    if t >= da_prices.len() {
        return Err(anyhow!("Missing DA prices for time_step {}", t));
    }

    let (continuation_curves, adaptive_bias) = get_or_init_cache(challenge, state);
    let (future_pos_shadows, future_neg_shadows) =
        compute_future_opportunity_shadows(challenge, state, &continuation_curves, &adaptive_bias);

    let steps_remaining = challenge.num_steps.saturating_sub(t);
    let horizon_fraction = steps_remaining as f64 / challenge.num_steps as f64;
    let num_batteries = challenge.batteries.len();
    let num_lines = challenge.network.num_lines;

    let baseline_action = state
        .action_bounds
        .iter()
        .map(|&(lo, hi)| 0.0f64.clamp(lo, hi))
        .collect::<Vec<f64>>();
    let baseline_flows = compute_flows(challenge, state, &baseline_action);

    let mut preferred_dirs = vec![0.0f64; num_batteries];
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (min_bound, max_bound) = state.action_bounds[i];
        if max_bound <= EPS && min_bound >= -EPS {
            continue;
        }

        let bias = adaptive_bias.get(node).copied().unwrap_or(0.0);
        let da_price = da_prices[t][node] + 0.50 * bias;

        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = (state.socs[i] - battery.soc_min_mwh) / soc_range;

        let (future_sell, future_buy) = if node < continuation_curves.len() {
            let curve = &continuation_curves[node];
            let j = 2 * t;
            if j + 1 < curve.len() {
                (curve[j], curve[j + 1])
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            }
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };

        let soc_surplus_mwh = (state.socs[i] - battery.soc_min_mwh).max(0.0);
        let soc_room_mwh = (battery.soc_max_mwh - state.socs[i]).max(0.0);
        let steps_left = steps_remaining.max(1) as f64;
        let required_discharge_mw = if battery.efficiency_discharge > EPS {
            soc_surplus_mwh
                / (steps_left
                    * tig_challenges::energy_arbitrage::constants::DELTA_T
                    * battery.efficiency_discharge)
        } else {
            0.0
        };
        let max_discharge_mw = battery.power_discharge_mw.max(EPS);
        let liquidation_urgency = (required_discharge_mw / max_discharge_mw).clamp(0.0, 1.0);

        let near_end = horizon_fraction < 0.30;
        let eff_c = battery.efficiency_charge.max(EPS);
        let eff_d = battery.efficiency_discharge;

        let discharge_edge = da_price * eff_d - future_buy / eff_c
            + liquidation_urgency * (future_buy.abs() / eff_c);
        let charge_edge = future_sell * battery.efficiency_charge * eff_d - da_price
            - liquidation_urgency * (future_sell.abs() * battery.efficiency_charge * eff_d);

        let want_discharge = max_bound > EPS
            && discharge_edge > EPS
            && (soc_surplus_mwh > EPS || liquidation_urgency > EPS || soc_level > EPS);
        let want_charge = min_bound < -EPS
            && !near_end
            && charge_edge > EPS
            && soc_room_mwh > EPS;

        preferred_dirs[i] = if want_discharge && (!want_charge || discharge_edge >= charge_edge) {
            1.0
        } else if want_charge {
            -1.0
        } else {
            0.0
        };
    }

    let proactive_bounds =
        compute_proactive_bounds(challenge, state, &baseline_flows, &preferred_dirs);

    let (base_pos_shadows, base_neg_shadows) = directional_line_shadow_prices(
        challenge,
        &baseline_flows,
        &future_pos_shadows,
        &future_neg_shadows,
    );

    let mut action = baseline_action;
    let mut flows = baseline_flows;

    #[derive(Clone, Copy)]
    struct Candidate {
        i: usize,
        target: f64,
        priority: f64,
    }

    let mut cands: Vec<Candidate> = Vec::with_capacity(num_batteries);
    let mut targets = vec![0.0f64; num_batteries];
    let mut values = vec![0.0f64; num_batteries];
    let mut discharge_edges = vec![f64::NEG_INFINITY; num_batteries];
    let mut charge_edges = vec![f64::NEG_INFINITY; num_batteries];

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (min_bound, max_bound) = proactive_bounds[i];

        let bias = adaptive_bias.get(node).copied().unwrap_or(0.0);
        let da_price = da_prices[t][node] + 0.50 * bias;

        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = (state.socs[i] - battery.soc_min_mwh) / soc_range;

        let (future_sell, future_buy) = if node < continuation_curves.len() {
            let curve = &continuation_curves[node];
            let j = 2 * t;
            if j + 1 < curve.len() {
                (curve[j], curve[j + 1])
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            }
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        };

        let soc_surplus_mwh = (state.socs[i] - battery.soc_min_mwh).max(0.0);
        let steps_left = steps_remaining.max(1) as f64;
        let required_discharge_mw = if battery.efficiency_discharge > EPS {
            soc_surplus_mwh
                / (steps_left
                    * tig_challenges::energy_arbitrage::constants::DELTA_T
                    * battery.efficiency_discharge)
        } else {
            0.0
        };
        let max_discharge_mw = battery.power_discharge_mw.max(EPS);
        let liquidation_urgency = (required_discharge_mw / max_discharge_mw).clamp(0.0, 1.0);

        let near_end = horizon_fraction < 0.30;

        let eff_c = battery.efficiency_charge.max(EPS);
        let eff_d = battery.efficiency_discharge;

        let discharge_edge_base = da_price * eff_d - future_buy / eff_c;
        let charge_edge_base = future_sell * battery.efficiency_charge * eff_d - da_price;

        let discharge_terminal = liquidation_urgency * (future_buy.abs() / eff_c);
        let charge_terminal =
            liquidation_urgency * (future_sell.abs() * battery.efficiency_charge * eff_d);

        let discharge_edge = discharge_edge_base + discharge_terminal;
        let charge_edge = charge_edge_base - charge_terminal;

        discharge_edges[i] = discharge_edge;
        charge_edges[i] = charge_edge;

        let discharge_frac =
            (soc_level + liquidation_urgency * (1.0 - soc_level)).clamp(0.0, 1.0);
        let charge_frac = ((1.0 - soc_level) * (1.0 - liquidation_urgency)).clamp(0.0, 1.0);

        let raw_u: f64 = if discharge_edge > EPS && max_bound > EPS {
            max_bound * discharge_frac
        } else if charge_edge > EPS && min_bound < -EPS && !near_end {
            min_bound * charge_frac
        } else {
            0.0f64
        }
        .clamp(min_bound, max_bound);

        let congestion_cost = directional_shadow_cost_for_node(
            challenge,
            &base_pos_shadows,
            &base_neg_shadows,
            node,
            raw_u.signum(),
        );
        let congestion_scale = if raw_u.abs() <= EPS {
            1.0
        } else {
            (1.0 / (1.0 + 0.60 * congestion_cost)).clamp(0.25, 1.0)
        };
        let u = (raw_u * (0.40 + 0.60 * congestion_scale)).clamp(min_bound, max_bound);

        let alpha = if u > EPS {
            discharge_edge
        } else if u < -EPS {
            charge_edge
        } else {
            0.0
        };

        let priority = alpha.abs() * (1.0 + liquidation_urgency) * congestion_scale;

        if priority > EPS && u.abs() > EPS {
            targets[i] = u;
            values[i] = priority;
            cands.push(Candidate {
                i,
                target: u,
                priority,
            });
        }
    }

    cands.sort_by(|a, b| {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    apply_counterflow_preconditioning(
        challenge,
        &mut flows,
        &mut action,
        &proactive_bounds,
        &targets,
        &values,
        &discharge_edges,
        &charge_edges,
        &future_pos_shadows,
        &future_neg_shadows,
    );

    let mut total_requested_mw = 0.0f64;
    let mut blocked_mw = 0.0f64;
    let mut blocked_moves = 0usize;

    for cand in cands {
        let i = cand.i;
        let (lo, hi) = proactive_bounds[i];

        let old_u = action[i];
        let desired = cand.target.clamp(lo, hi);
        let delta = desired - old_u;
        if delta.abs() <= EPS {
            continue;
        }

        total_requested_mw += delta.abs();

        let node = challenge.batteries[i].node;
        let scale = feasible_scale_for_delta(challenge, &flows, node, delta);
        if scale <= EPS {
            blocked_moves += 1;
            blocked_mw += delta.abs();
            continue;
        }
        if scale < 1.0 - 1e-9 {
            blocked_moves += 1;
            blocked_mw += delta.abs() * (1.0 - scale);
        }

        let new_u = (old_u + delta * scale).clamp(lo, hi);
        let applied = new_u - old_u;
        if applied.abs() <= EPS {
            continue;
        }

        action[i] = new_u;

        for l in 0..num_lines {
            flows[l] += challenge.network.ptdf[l][node] * applied;
        }
    }

    let mut max_util = 0.0f64;
    for l in 0..num_lines {
        let limit = challenge.network.flow_limits[l].abs().max(EPS);
        max_util = max_util.max(flows[l].abs() / limit);
    }

    let pair_trigger = max_util > 0.86
        || (blocked_moves > 0
            && blocked_mw > 0.03 * total_requested_mw.max(EPS)
            && max_util > 0.70);

    if pair_trigger && num_batteries > 1 && num_lines > 1 {
        let max_exchanges = if num_batteries <= 8 { 3 } else { 5 };

        for _ in 0..max_exchanges {
            let (pos_shadows, neg_shadows) = directional_line_shadow_prices(
                challenge,
                &flows,
                &future_pos_shadows,
                &future_neg_shadows,
            );
            let stress_weights = line_stress_weights(
                challenge,
                &flows,
                &future_pos_shadows,
                &future_neg_shadows,
            );
            let mut stressed_lines: Vec<usize> = (0..num_lines)
                .filter(|&l| stress_weights[l] > EPS)
                .collect();

            if stressed_lines.is_empty() {
                stressed_lines = (0..num_lines).collect();
                stressed_lines.sort_by(|&a, &b| {
                    let util_a =
                        flows[a].abs() / challenge.network.flow_limits[a].abs().max(EPS);
                    let util_b =
                        flows[b].abs() / challenge.network.flow_limits[b].abs().max(EPS);
                    util_b
                        .partial_cmp(&util_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                stressed_lines.truncate(3);
                stressed_lines.retain(|&l| {
                    flows[l].abs() / challenge.network.flow_limits[l].abs().max(EPS) > 0.72
                });
            } else {
                stressed_lines.sort_by(|&a, &b| {
                    stress_weights[b]
                        .partial_cmp(&stress_weights[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                stressed_lines.truncate(3);
            }

            if stressed_lines.is_empty() {
                break;
            }

            let mut best: Option<(usize, usize, f64, f64, f64)> = None;
            let mut best_score = EPS;

            for &line in stressed_lines.iter() {
                let signed_direction = flows[line].signum();
                if signed_direction.abs() <= EPS {
                    continue;
                }

                let mut improvers: Vec<usize> = (0..num_batteries)
                    .filter(|&i| {
                        let residual = targets[i] - action[i];
                        if residual.abs() <= EPS || values[i] <= EPS {
                            return false;
                        }
                        let node = challenge.batteries[i].node;
                        signed_direction * challenge.network.ptdf[line][node] * residual.signum()
                            > EPS
                    })
                    .collect();

                if improvers.is_empty() {
                    continue;
                }

                improvers.sort_by(|&a, &b| {
                    let node_a = challenge.batteries[a].node;
                    let node_b = challenge.batteries[b].node;
                    let resid_a = targets[a] - action[a];
                    let resid_b = targets[b] - action[b];
                    let worsen_a =
                        (signed_direction * challenge.network.ptdf[line][node_a] * resid_a.signum())
                            .max(EPS);
                    let worsen_b =
                        (signed_direction * challenge.network.ptdf[line][node_b] * resid_b.signum())
                            .max(EPS);
                    let score_a = values[a] * resid_a.abs() / worsen_a;
                    let score_b = values[b] * resid_b.abs() / worsen_b;
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                improvers.truncate(6);

                let mut target_partners: Vec<usize> = (0..num_batteries)
                    .filter(|&j| {
                        let residual = targets[j] - action[j];
                        if residual.abs() <= EPS || values[j] <= EPS {
                            return false;
                        }
                        let node = challenge.batteries[j].node;
                        -signed_direction * challenge.network.ptdf[line][node] * residual.signum()
                            > EPS
                    })
                    .collect();

                target_partners.sort_by(|&a, &b| {
                    let node_a = challenge.batteries[a].node;
                    let node_b = challenge.batteries[b].node;
                    let resid_a = targets[a] - action[a];
                    let resid_b = targets[b] - action[b];
                    let relief_a = (-signed_direction
                        * challenge.network.ptdf[line][node_a]
                        * resid_a.signum())
                    .max(EPS);
                    let relief_b = (-signed_direction
                        * challenge.network.ptdf[line][node_b]
                        * resid_b.signum())
                    .max(EPS);
                    let score_a = values[a] * relief_a * resid_a.abs();
                    let score_b = values[b] * relief_b * resid_b.abs();
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                target_partners.truncate(6);

                let mut release_partners: Vec<usize> = (0..num_batteries)
                    .filter(|&j| {
                        if action[j].abs() <= EPS || values[j] <= EPS {
                            return false;
                        }
                        let node = challenge.batteries[j].node;
                        -signed_direction * challenge.network.ptdf[line][node] * (-action[j].signum())
                            > EPS
                    })
                    .collect();

                release_partners.sort_by(|&a, &b| {
                    let node_a = challenge.batteries[a].node;
                    let node_b = challenge.batteries[b].node;
                    let relief_a = (-signed_direction
                        * challenge.network.ptdf[line][node_a]
                        * (-action[a].signum()))
                    .max(EPS);
                    let relief_b = (-signed_direction
                        * challenge.network.ptdf[line][node_b]
                        * (-action[b].signum()))
                    .max(EPS);
                    let score_a = values[a] / relief_a;
                    let score_b = values[b] / relief_b;
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                release_partners.truncate(6);

                if target_partners.is_empty() && release_partners.is_empty() {
                    continue;
                }

                for &i in improvers.iter() {
                    let residual = targets[i] - action[i];
                    let residual_cap = residual.abs();
                    let node_i = challenge.batteries[i].node;
                    let dir_i = residual.signum();
                    let worsen_i = signed_direction * challenge.network.ptdf[line][node_i] * dir_i;
                    if worsen_i <= EPS {
                        continue;
                    }

                    for &j in target_partners.iter() {
                        if i == j {
                            continue;
                        }

                        let residual_j = targets[j] - action[j];
                        let node_j = challenge.batteries[j].node;
                        let partner_dir = residual_j.signum();
                        let relief_j =
                            -signed_direction * challenge.network.ptdf[line][node_j] * partner_dir;
                        if relief_j <= EPS {
                            continue;
                        }

                        let ratio = worsen_i / relief_j;
                        if !ratio.is_finite() || ratio <= EPS {
                            continue;
                        }

                        let step_cap = residual_cap.min(residual_j.abs() / ratio);
                        if step_cap <= EPS {
                            continue;
                        }

                        let delta_i_unit = dir_i;
                        let delta_j_unit = partner_dir * ratio;
                        let step = feasible_pair_step(
                            challenge,
                            &flows,
                            node_i,
                            delta_i_unit,
                            node_j,
                            delta_j_unit,
                            step_cap,
                        );
                        if step <= EPS {
                            continue;
                        }

                        let mut side_risk = 0.0f64;
                        let mut side_relief = 0.0f64;

                        for l in 0..num_lines {
                            if l == line {
                                continue;
                            }

                            let flow_dir = flows[l].signum();
                            if flow_dir.abs() <= EPS {
                                continue;
                            }

                            let pair_push = challenge.network.ptdf[l][node_i] * delta_i_unit
                                + challenge.network.ptdf[l][node_j] * delta_j_unit;
                            let active_shadow = if flow_dir >= 0.0 {
                                pos_shadows[l]
                            } else {
                                neg_shadows[l]
                            };
                            let move_shadow =
                                directional_shadow_for_line_delta(&pos_shadows, &neg_shadows, l, pair_push);
                            let align = flow_dir * pair_push;
                            if align > EPS {
                                side_risk += move_shadow * align;
                            } else if align < -EPS {
                                side_relief += active_shadow * (-align);
                            }
                        }

                        let score_per_step =
                            (values[i] + values[j] * ratio) * (1.0 + 0.10 * side_relief)
                                / (1.0 + 0.20 * side_risk);
                        let score = score_per_step * step;

                        if score > best_score {
                            best_score = score;
                            best = Some((i, j, step, ratio, partner_dir));
                        }
                    }

                    for &j in release_partners.iter() {
                        if i == j {
                            continue;
                        }

                        let node_j = challenge.batteries[j].node;
                        let partner_dir = -action[j].signum();
                        let relief_j =
                            -signed_direction * challenge.network.ptdf[line][node_j] * partner_dir;
                        if relief_j <= EPS {
                            continue;
                        }

                        let ratio = worsen_i / relief_j;
                        if !ratio.is_finite() || ratio <= EPS {
                            continue;
                        }

                        let step_cap = residual_cap.min(action[j].abs() / ratio);
                        if step_cap <= EPS {
                            continue;
                        }

                        let delta_i_unit = dir_i;
                        let delta_j_unit = partner_dir * ratio;
                        let step = feasible_pair_step(
                            challenge,
                            &flows,
                            node_i,
                            delta_i_unit,
                            node_j,
                            delta_j_unit,
                            step_cap,
                        );
                        if step <= EPS {
                            continue;
                        }

                        let mut side_risk = 0.0f64;
                        let mut side_relief = 0.0f64;

                        for l in 0..num_lines {
                            if l == line {
                                continue;
                            }

                            let flow_dir = flows[l].signum();
                            if flow_dir.abs() <= EPS {
                                continue;
                            }

                            let pair_push = challenge.network.ptdf[l][node_i] * delta_i_unit
                                + challenge.network.ptdf[l][node_j] * delta_j_unit;
                            let active_shadow = if flow_dir >= 0.0 {
                                pos_shadows[l]
                            } else {
                                neg_shadows[l]
                            };
                            let move_shadow =
                                directional_shadow_for_line_delta(&pos_shadows, &neg_shadows, l, pair_push);
                            let align = flow_dir * pair_push;
                            if align > EPS {
                                side_risk += move_shadow * align;
                            } else if align < -EPS {
                                side_relief += active_shadow * (-align);
                            }
                        }

                        let score_per_step =
                            (values[i] - values[j] * ratio) * (1.0 + 0.08 * side_relief)
                                / (1.0 + 0.18 * side_risk);
                        let score = score_per_step * step;

                        if score > best_score {
                            best_score = score;
                            best = Some((i, j, step, ratio, partner_dir));
                        }
                    }
                }
            }

            let Some((i, j, step, ratio, partner_dir)) = best else {
                break;
            };

            let old_i = action[i];
            let old_j = action[j];
            let delta_i = (targets[i] - old_i).signum() * step;
            let delta_j = partner_dir * ratio * step;

            let new_i = (old_i + delta_i)
                .clamp(proactive_bounds[i].0, proactive_bounds[i].1);
            let new_j = (old_j + delta_j)
                .clamp(proactive_bounds[j].0, proactive_bounds[j].1);

            let applied_i = new_i - old_i;
            let applied_j = new_j - old_j;

            if applied_i.abs() <= EPS && applied_j.abs() <= EPS {
                break;
            }

            action[i] = new_i;
            action[j] = new_j;

            let node_i = challenge.batteries[i].node;
            let node_j = challenge.batteries[j].node;
            for l in 0..num_lines {
                flows[l] += challenge.network.ptdf[l][node_i] * applied_i
                    + challenge.network.ptdf[l][node_j] * applied_j;
            }
        }
    }

    enforce_flow_feasibility(challenge, state, action)
}