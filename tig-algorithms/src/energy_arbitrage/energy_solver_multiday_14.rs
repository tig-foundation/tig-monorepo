// energy_solver_multiday_14: STRUCTURAL — DA-volatility-aware threshold scaling. Independent of trend (which captures linear shift). Std of DA over lookahead raises thresholds when volatility high. vol_factor=0 reproduces multiday_12.
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub vol_factor: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self { vol_factor: 0.0 }
    }
}

pub fn help() {
    println!("HYPERPARAMETERS:");
    println!("  vol_factor: float range=[0.0, 1.0] default=0.0  # 0=multiday_12; positive raises thresholds in high-vol windows");
    println!();
    println!("Adds DA-volatility-aware threshold scaling to multiday_12. std_signal = std(DA[t+1..t+K]).");
    println!("discharge_th_eff = (discharge_th + trend_shift) * (1 + vol_factor * std_signal/std_ref)");
    println!("charge_th_eff    = (charge_th    - trend_shift) * (1 + vol_factor * std_signal/std_ref)");
    println!("std_ref = 5.0 (normalization). Higher std → higher thresholds → more selective.");
}

fn parse_hp(h: &Option<Map<String, Value>>) -> Hyperparameters {
    let mut out = Hyperparameters::default();
    if let Some(m) = h {
        if let Some(v) = m.get("vol_factor").and_then(|v| v.as_f64()) {
            out.vol_factor = v;
        }
    }
    out
}

const LOOKAHEAD_STEPS: usize = 66;
const END_WINDOW: usize = 66;
const END_FACTOR: f64 = 0.0;
const SLOPE_FACTOR: f64 = 1.273;
const CHARGE_THRESHOLD: f64 = 8.0;
const DISCHARGE_THRESHOLD: f64 = 1.0;
const TREND_FACTOR: f64 = 0.6364;
const PROFIT_WEIGHT_POWER: f64 = 2.545;
const STD_REF: f64 = 5.0;

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const EPS: f64 = 1e-12;
const FLOW_EPS: f64 = 1e-6;
const PROFIT_EPS: f64 = 1e-3;

#[derive(Clone, Copy)]
struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let injections = challenge.compute_total_injections(state, action);
    challenge.network.compute_flows(&injections)
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > FLOW_EPS * limit {
            let cand = Violation { line: l, flow, amount: violation };
            match best {
                Some(cur) if cand.amount <= cur.amount => {}
                _ => best = Some(cand),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let flows = compute_flows(challenge, state, action);
    most_violated_line(challenge, &flows).is_none()
}

fn soften_line_weighted(
    challenge: &Challenge,
    v: Violation,
    action: &mut [f64],
    profit: &[f64],
    p: f64,
) -> bool {
    let line = v.line;
    let dir = v.flow.signum();
    if dir.abs() <= EPS {
        return false;
    }

    let mut worsening: Vec<(usize, f64)> = Vec::new();
    let mut total_signed = 0.0;
    for (i, b) in challenge.batteries.iter().enumerate() {
        let contrib = challenge.network.ptdf[line][b.node] * action[i];
        let signed = dir * contrib;
        if signed > EPS {
            total_signed += signed;
            worsening.push((i, signed));
        }
    }
    if worsening.is_empty() || total_signed <= EPS {
        return false;
    }

    if p == 0.0 {
        let keep = (1.0 - v.amount / total_signed).clamp(0.0, 1.0);
        if (1.0 - keep).abs() <= EPS {
            return false;
        }
        for (i, _) in &worsening {
            action[*i] *= keep;
        }
        return true;
    }

    let mut weights: Vec<f64> = worsening
        .iter()
        .map(|(i, signed)| {
            let pi = profit[*i].max(0.0).powf(p) + PROFIT_EPS;
            signed / pi
        })
        .collect();

    let mut residual = v.amount;
    let mut sum_w: f64 = weights.iter().sum();
    let mut zeroed = vec![false; worsening.len()];

    loop {
        if sum_w <= EPS || residual <= EPS {
            break;
        }
        let mut any_zeroed = false;
        for k in 0..worsening.len() {
            if zeroed[k] {
                continue;
            }
            let (_, signed) = worsening[k];
            let reduce = (residual / sum_w) * weights[k];
            if reduce >= signed {
                zeroed[k] = true;
                residual -= signed;
                sum_w -= weights[k];
                weights[k] = 0.0;
                any_zeroed = true;
            }
        }
        if !any_zeroed {
            break;
        }
    }

    let mut changed = false;
    for k in 0..worsening.len() {
        let (i, signed) = worsening[k];
        if zeroed[k] {
            if action[i].abs() > EPS {
                action[i] = 0.0;
                changed = true;
            }
        } else if sum_w > EPS && residual > EPS {
            let reduce = (residual / sum_w) * weights[k];
            let keep = (1.0 - reduce / signed).clamp(0.0, 1.0);
            if (1.0 - keep).abs() > EPS {
                action[i] *= keep;
                changed = true;
            }
        }
    }
    changed
}

fn enforce_feasible(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
    profit: &[f64],
    p: f64,
) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(v) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_line_weighted(challenge, v, &mut action, profit, p) {
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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp = parse_hp(hyperparameters);
    let policy_fn = move |ch: &Challenge, st: &State| -> Result<Vec<f64>> {
        policy_with_hp(ch, st, &hp)
    };
    let solution = challenge.grid_optimize(&policy_fn)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    policy_with_hp(challenge, state, &Hyperparameters::default())
}

fn median_of(buf: &mut Vec<f64>) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = buf.len();
    if n % 2 == 1 {
        buf[n / 2]
    } else {
        0.5 * (buf[n / 2 - 1] + buf[n / 2])
    }
}

fn std_of(buf: &[f64]) -> f64 {
    if buf.len() < 2 {
        return 0.0;
    }
    let n = buf.len() as f64;
    let mean: f64 = buf.iter().sum::<f64>() / n;
    let var: f64 = buf.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

fn policy_with_hp(challenge: &Challenge, state: &State, hp: &Hyperparameters) -> Result<Vec<f64>> {
    let t = state.time_step;
    let end = (t + 1 + LOOKAHEAD_STEPS).min(challenge.num_steps);
    let da = &challenge.market.day_ahead_prices;

    let remaining = challenge.num_steps.saturating_sub(t);
    let in_end_window = END_WINDOW > 0 && remaining <= END_WINDOW;
    let (base_discharge_th, base_charge_th, suppress_charge) = if in_end_window {
        let progress = (remaining as f64 / END_WINDOW as f64).clamp(0.0, 1.0);
        let scale = END_FACTOR + (1.0 - END_FACTOR) * progress;
        (DISCHARGE_THRESHOLD * scale, CHARGE_THRESHOLD, true)
    } else {
        (DISCHARGE_THRESHOLD, CHARGE_THRESHOLD, false)
    };
    let slope = SLOPE_FACTOR.max(0.05);
    let vol_factor = hp.vol_factor.clamp(0.0, 4.0);

    let mut action = vec![0.0; challenge.num_batteries];
    let mut profit = vec![0.0; challenge.num_batteries];
    let mut buf: Vec<f64> = Vec::with_capacity(LOOKAHEAD_STEPS);
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let node = battery.node;
        buf.clear();
        for tt in (t + 1)..end {
            buf.push(da[tt][node]);
        }
        if buf.is_empty() {
            let (_, u_max) = state.action_bounds[i];
            action[i] = u_max;
            profit[i] = base_discharge_th;
            continue;
        }

        let trend_signal = if buf.len() >= 2 {
            buf[buf.len() - 1] - buf[0]
        } else {
            0.0
        };
        let std_signal = std_of(&buf);
        let vol_mult = 1.0 + vol_factor * (std_signal / STD_REF);

        let mut buf_for_median = buf.clone();
        let reference = median_of(&mut buf_for_median);
        let current = state.rt_prices[node];
        let diff = current - reference;
        profit[i] = diff.abs();

        let discharge_th_eff = ((base_discharge_th + TREND_FACTOR * trend_signal) * vol_mult).max(0.5);
        let charge_th_eff = ((base_charge_th - TREND_FACTOR * trend_signal) * vol_mult).max(0.5);

        let (u_min, u_max) = state.action_bounds[i];
        if diff > discharge_th_eff {
            let span = (slope * discharge_th_eff).max(1e-6);
            let intensity = ((diff - discharge_th_eff) / span).clamp(0.0, 1.0);
            action[i] = intensity * u_max;
        } else if !suppress_charge && diff < -charge_th_eff {
            let span = (slope * charge_th_eff).max(1e-6);
            let intensity = ((-diff - charge_th_eff) / span).clamp(0.0, 1.0);
            action[i] = intensity * u_min;
        }
    }

    enforce_feasible(challenge, state, action, &profit, PROFIT_WEIGHT_POWER)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
