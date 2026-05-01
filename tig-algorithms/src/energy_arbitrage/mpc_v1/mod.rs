// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::{constants, Challenge, NextRTPrices, Solution, State};

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

// MPC branch constants (BASELINE, CONGESTED)
const N_CAND: usize = 5;
const HORIZON: usize = 96;
const TERM_LOOK: usize = 12;

// Composite routing threshold: between CONGESTED (10×96+20×96=1920) and MULTIDAY (40×192=7680)
// Gap = 5760 → threshold at 3000 gives ≥60% clearance on both sides
const HONEST_SIZE_THRESHOLD: usize = 3000;

pub fn help() {
    println!(
        "mpc_v1: composite algorithm — MPC H={HORIZON} N={N_CAND} for small tracks, \
         DA-greedy fallback for large tracks (threshold size={})",
        HONEST_SIZE_THRESHOLD
    );
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let size = challenge.num_batteries * challenge.num_steps;
    let solution = if size <= HONEST_SIZE_THRESHOLD {
        challenge.grid_optimize(&mpc_policy)?
    } else {
        challenge.grid_optimize(&fallback_policy)?
    };
    save_solution(&solution)?;
    Ok(())
}

// Public entry point keeping original signature (for tig_challenges:: detection)
pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let size = challenge.num_batteries * challenge.num_steps;
    if size <= HONEST_SIZE_THRESHOLD {
        mpc_policy(challenge, state)
    } else {
        fallback_policy(challenge, state)
    }
}

// === Branch A: Full rolling-horizon MPC (BASELINE, CONGESTED) ===
// H=96, N=5, non-myopic (H/steps = 1.0 ≥ 0.25), sequential coordinate descent

fn mpc_policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
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
        chosen[b] = mpc_optimize_battery(challenge, state, b, t);
    }

    // Joint flow feasibility: binary-search uniform scale-down
    let inj = challenge.compute_total_injections(state, &chosen);
    let flows = challenge.network.compute_flows(&inj);
    if challenge.network.verify_flows(&flows).is_err() {
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        for _ in 0..32 {
            let mid = (lo + hi) * 0.5;
            let scaled: Vec<f64> = chosen.iter().map(|a| a * mid).collect();
            let inj2 = challenge.compute_total_injections(state, &scaled);
            let fl2 = challenge.network.compute_flows(&inj2);
            if challenge.network.verify_flows(&fl2).is_ok() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        for a in chosen.iter_mut() {
            *a *= lo;
        }
    }

    Ok(chosen)
}

fn mpc_optimize_battery(challenge: &Challenge, state: &State, bat: usize, t: usize) -> f64 {
    let (lo, hi) = state.action_bounds[bat];
    let candidates = make_candidates(lo, hi, N_CAND);

    let mut best_val = f64::NEG_INFINITY;
    let mut best_u = 0.0f64;
    for u in candidates {
        let u = u.clamp(lo, hi);
        let val = mpc_lookahead(challenge, state, bat, u, t);
        if val > best_val {
            best_val = val;
            best_u = u;
        }
    }
    best_u
}

fn mpc_lookahead(challenge: &Challenge, state: &State, bat: usize, u0: f64, t: usize) -> f64 {
    let n_bat = challenge.num_batteries;
    let da = &challenge.market.day_ahead_prices;
    let horizon = HORIZON.min(challenge.num_steps.saturating_sub(t));

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
        total += terminal_soc_value(challenge, &sim, bat, t + horizon);
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

fn terminal_soc_value(challenge: &Challenge, state: &State, bat: usize, horizon_end: usize) -> f64 {
    let b = &challenge.batteries[bat];
    let available = (state.socs[bat] - b.soc_min_mwh).max(0.0);
    if available < 1e-9 {
        return 0.0;
    }
    let da = &challenge.market.day_ahead_prices;
    let end = (horizon_end + TERM_LOOK).min(challenge.num_steps);
    if end <= horizon_end {
        return 0.0;
    }
    let node = b.node;
    let count = (end - horizon_end) as f64;
    let avg_price: f64 = (horizon_end..end).map(|s| da[s][node]).sum::<f64>() / count;
    let power = (available * b.efficiency_discharge / constants::DELTA_T).min(b.power_discharge_mw);
    let revenue = power * avg_price * constants::DELTA_T;
    let tx = constants::KAPPA_TX * power * constants::DELTA_T;
    let deg_base = (power * constants::DELTA_T) / b.capacity_mwh;
    let deg = constants::KAPPA_DEG * deg_base.powf(constants::BETA_DEG);
    (revenue - tx - deg).max(0.0)
}

fn make_candidates(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 || (hi - lo).abs() < 1e-9 {
        return vec![lo];
    }
    let mut pts: Vec<f64> = (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect();
    pts[0] = lo;
    pts[n - 1] = hi;
    pts
}

// === Branch B: Completion-focused fallback (MULTIDAY, DENSE, CAPSTONE) ===
// Single-step DA-greedy, 3-step window, 50% action scaling, binary flow shrink.
// No take_step simulation — O(batteries + lines×nodes) per step.

fn fallback_policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let t = state.time_step;
    let n_bat = challenge.num_batteries;
    let da = &challenge.market.day_ahead_prices;

    let mut actions: Vec<f64> = (0..n_bat)
        .map(|b| {
            let node = challenge.batteries[b].node;
            let current = state.rt_prices[node];

            // 3-step forward DA average
            let win_end = (t + 4).min(challenge.num_steps);
            let count = (win_end.saturating_sub(t + 1)) as f64;
            let da_avg = if count > 0.0 {
                ((t + 1)..win_end).map(|s| da[s][node]).sum::<f64>() / count
            } else {
                current
            };

            let (lo, hi) = state.action_bounds[b];
            if current < da_avg * 0.9 {
                lo * 0.5
            } else if current > da_avg * 1.1 {
                hi * 0.5
            } else {
                0.0
            }
        })
        .collect();

    // Flow feasibility: binary shrink at [1.0, 0.5, 0.25, 0.0] scales
    let inj = challenge.compute_total_injections(state, &actions);
    let flows = challenge.network.compute_flows(&inj);
    if challenge.network.verify_flows(&flows).is_ok() {
        return Ok(actions);
    }

    for &scale in &[0.5f64, 0.25, 0.0] {
        let scaled: Vec<f64> = actions.iter().map(|a| a * scale).collect();
        let inj2 = challenge.compute_total_injections(state, &scaled);
        let fl2 = challenge.network.compute_flows(&inj2);
        if challenge.network.verify_flows(&fl2).is_ok() {
            return Ok(scaled);
        }
    }

    // scale=0.0 is zeros which is always feasible (zero action = no flow)
    Ok(vec![0.0; n_bat])
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
