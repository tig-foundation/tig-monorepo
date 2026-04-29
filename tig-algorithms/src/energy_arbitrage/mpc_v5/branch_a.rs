use anyhow::Result;
use tig_challenges::energy_arbitrage::{Challenge, NextRTPrices, State};

use super::config::Config;

// warm_hints: previous time step's chosen actions (post-flow-scale), one per battery.
// When Some, the hint for battery b is appended as an extra candidate (clamped to [lo,hi]).
// This adds O(1) overhead — the uniform grid is always fully evaluated regardless.
pub fn policy(
    challenge: &Challenge,
    state: &State,
    config: &Config,
    warm_hints: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_bat = challenge.num_batteries;

    let mut bat_order: Vec<usize> = (0..n_bat).collect();
    bat_order.sort_by(|&a, &b| {
        challenge.batteries[b]
            .capacity_mwh
            .partial_cmp(&challenge.batteries[a].capacity_mwh)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut chosen = vec![0.0f64; n_bat];
    for &b in &bat_order {
        let hint = warm_hints.map(|h| h[b]);
        chosen[b] = optimize_battery(challenge, state, b, t, config, hint);
    }

    Ok(super::apply_flow_scale(challenge, state, chosen))
}

fn optimize_battery(
    challenge: &Challenge,
    state: &State,
    bat: usize,
    t: usize,
    config: &Config,
    warm_hint: Option<f64>,
) -> f64 {
    let (lo, hi) = state.action_bounds[bat];
    let mut candidates = super::make_candidates(lo, hi, config.n_cand_a);

    if let Some(prev) = warm_hint {
        let clamped = prev.clamp(lo, hi);
        if candidates.iter().all(|&p| (p - clamped).abs() > 1e-6) {
            candidates.push(clamped);
        }
    }

    let mut best_val = f64::NEG_INFINITY;
    let mut best_u = 0.0f64;
    for u in candidates {
        let u = u.clamp(lo, hi);
        let val = lookahead(challenge, state, bat, u, t, config);
        if val > best_val {
            best_val = val;
            best_u = u;
        }
    }
    best_u
}

fn lookahead(
    challenge: &Challenge,
    state: &State,
    bat: usize,
    u0: f64,
    t: usize,
    config: &Config,
) -> f64 {
    let n_bat = challenge.num_batteries;
    let da = &challenge.market.day_ahead_prices;
    let horizon = config.horizon.min(challenge.num_steps.saturating_sub(t));

    let mut sim = state.clone();
    let mut total = 0.0f64;

    for h in 0..horizon {
        let step = t + h;
        let mut action = vec![0.0f64; n_bat];
        action[bat] = if h == 0 {
            u0
        } else {
            da_greedy_action(challenge, &sim, bat, step)
        };
        action[bat] = action[bat].clamp(sim.action_bounds[bat].0, sim.action_bounds[bat].1);

        let step_profit = challenge.compute_profit(&sim, &action);
        let next_prices = da[(step + 1).min(da.len() - 1)].clone();

        match challenge.take_step(&sim, &action, NextRTPrices::Override(next_prices)) {
            Ok(next_sim) => {
                total += step_profit;
                sim = next_sim;
            }
            Err(_) => return f64::NEG_INFINITY,
        }
    }

    if !sim.socs.is_empty() {
        total += super::terminal_soc_value(challenge, &sim, bat, t + horizon);
    }
    total
}

fn da_greedy_action(challenge: &Challenge, state: &State, bat: usize, t: usize) -> f64 {
    let node = challenge.batteries[bat].node;
    let da = &challenge.market.day_ahead_prices;
    let current = da[t][node];
    let look_end = (t + 12).min(challenge.num_steps);
    let count = (look_end.saturating_sub(t + 1)) as f64;
    let avg = if count > 0.0 {
        ((t + 1)..look_end).map(|s| da[s][node]).sum::<f64>() / count
    } else {
        current
    };
    let (lo, hi) = state.action_bounds[bat];
    if current < avg * 0.9 {
        lo * 0.5
    } else if current > avg * 1.1 {
        hi * 0.5
    } else {
        0.0
    }
}
