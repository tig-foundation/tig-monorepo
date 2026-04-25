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
    let mut cap = 0.0f64;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let g = signed_line_sensitivity(challenge, line, signed_direction, battery.node);
        if g.abs() <= EPS {
            continue;
        }

        let (min_bound, max_bound) = state.action_bounds[i];
        let ai = action[i];

        if g > 0.0 {
            let room = (ai - min_bound).max(0.0);
            cap += g * room;
        } else {
            let room = (max_bound - ai).max(0.0);
            cap += (-g) * room;
        }
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

fn soften_most_violated_line(challenge: &Challenge, violation: &Violation, action: &mut [f64]) -> bool {
    let line = violation.line;
    let signed_direction = violation.flow.signum();
    if signed_direction.abs() <= EPS {
        return false;
    }

    struct Entry {
        idx: usize,
        g2: f64,
        sc: f64,
    }

    let mut entries: Vec<Entry> = Vec::new();
    let mut total_removable = 0.0;

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let g = signed_direction * challenge.network.ptdf[line][battery.node];
        let sc = g * action[i];
        if sc > EPS {
            let g2 = g * g;
            if g2 > EPS {
                total_removable += sc;
                entries.push(Entry { idx: i, g2, sc });
            }
        }
    }

    if entries.is_empty() || total_removable <= EPS {
        return false;
    }

    if total_removable + EPS < violation.amount {
        return false;
    }

    let mut remaining = violation.amount;
    let mut active: Vec<usize> = (0..entries.len()).collect();
    let mut alloc = vec![0.0f64; entries.len()];

    while remaining > EPS && !active.is_empty() {
        let denom: f64 = active.iter().map(|&j| entries[j].g2).sum();
        if denom <= EPS {
            break;
        }
        let lambda = remaining / denom;

        let mut any_saturated = false;
        for &j in active.iter() {
            if lambda * entries[j].g2 >= entries[j].sc - EPS {
                any_saturated = true;
                break;
            }
        }

        if !any_saturated {
            for &j in active.iter() {
                alloc[j] = lambda * entries[j].g2;
            }
            remaining = 0.0;
            break;
        }

        let mut new_active: Vec<usize> = Vec::with_capacity(active.len());
        let mut saturated_sum = 0.0;
        for &j in active.iter() {
            if lambda * entries[j].g2 >= entries[j].sc - EPS {
                alloc[j] = entries[j].sc;
                saturated_sum += entries[j].sc;
            } else {
                new_active.push(j);
            }
        }

        if saturated_sum <= EPS {
            break;
        }
        remaining = (remaining - saturated_sum).max(0.0);
        active = new_active;
    }

    if remaining > EPS {
        return false;
    }

    let mut changed = false;
    for (j, e) in entries.iter().enumerate() {
        let r = alloc[j];
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

fn enforce_flow_feasibility(challenge: &Challenge, state: &State, mut action: Vec<f64>) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(violation) = most_violated_line(challenge, state, &action, &flows) else {
            return Ok(action);
        };
        if !soften_most_violated_line(challenge, &violation, &mut action) {
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

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let num_steps = challenge.num_steps;
    let da_prices = &challenge.market.day_ahead_prices;

    let window_end = (t + LOOKAHEAD_WINDOW).min(num_steps);
    let steps_remaining_total = (num_steps - t) as f64 / num_steps as f64;

    let mut action = vec![0.0f64; challenge.num_batteries];

    let mut min_futures = vec![0.0f64; challenge.num_batteries];
    let mut max_futures = vec![0.0f64; challenge.num_batteries];
    let mut current_prices = vec![0.0f64; challenge.num_batteries];

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        let (min_bound, max_bound) = state.action_bounds[i];
        let current_price = da_prices[t][node];
        current_prices[i] = current_price;

        let future_prices = &da_prices[t..window_end];
        let min_future = future_prices
            .iter()
            .map(|p| p[node])
            .fold(f64::MAX, f64::min);
        let max_future = future_prices
            .iter()
            .map(|p| p[node])
            .fold(f64::MIN, f64::max);
        min_futures[i] = min_future;
        max_futures[i] = max_future;

        let price_range = (max_future - min_future).max(EPS);

        let soc_range = (battery.soc_max_mwh - battery.soc_min_mwh).max(EPS);
        let soc_level = (state.socs[i] - battery.soc_min_mwh) / soc_range;

        let charge_edge = max_future - current_price;
        let discharge_edge = current_price - min_future;

        let charge_intensity = (charge_edge / price_range).clamp(0.0, 1.0);
        let discharge_intensity = (discharge_edge / price_range).clamp(0.0, 1.0);

        let mut desired = if charge_intensity > discharge_intensity {
            let headroom = 1.0 - soc_level;
            -battery.power_charge_mw * (charge_intensity * headroom).sqrt()
        } else if discharge_intensity > charge_intensity {
            let available = soc_level;
            battery.power_discharge_mw * (discharge_intensity * available).sqrt()
        } else {
            0.0
        };

        if steps_remaining_total < 0.10 && soc_level > 0.50 {
            let liquidate = battery.power_discharge_mw * (soc_level - 0.50) * 0.8;
            if liquidate > desired {
                desired = liquidate;
            }
        }

        action[i] = desired.clamp(min_bound, max_bound);
    }

    let mut action = enforce_flow_feasibility(challenge, state, action)?;

    let mut best_charge_i: Option<usize> = None;
    let mut best_charge_score = 0.0f64;
    let mut best_discharge_i: Option<usize> = None;
    let mut best_discharge_score = 0.0f64;

    for i in 0..challenge.num_batteries {
        let (min_bound, max_bound) = state.action_bounds[i];
        let a = action[i];
        let cp = current_prices[i];
        let min_f = min_futures[i];
        let max_f = max_futures[i];

        if a > min_bound + EPS {
            let score = max_f - cp;
            if score > best_charge_score {
                best_charge_score = score;
                best_charge_i = Some(i);
            }
        }

        if a < max_bound - EPS {
            let score = cp - min_f;
            if score > best_discharge_score {
                best_discharge_score = score;
                best_discharge_i = Some(i);
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
            let (min_bound, max_bound) = state.action_bounds[ci];
            if best_charge_score >= best_discharge_score {
                apply_move(ci, min_bound, &mut action);
            } else {
                apply_move(ci, max_bound, &mut action);
            }
        }
        _ => {
            if let Some(ci) = best_charge_i {
                let (min_bound, _) = state.action_bounds[ci];
                apply_move(ci, min_bound, &mut action);
            }
            if let Some(di) = best_discharge_i {
                let (_, max_bound) = state.action_bounds[di];
                apply_move(di, max_bound, &mut action);
            }
        }
    }

    Ok(action)
}