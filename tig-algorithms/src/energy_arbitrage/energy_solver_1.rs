// energy_solver_1: v14 lineage (median-DA-lookahead threshold policy, end-window liquidation) with profit-weighted line softening (p=2.545 fixed). Cross-track starting point for per-track iteration runs.
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub charge_threshold: f64,
    pub discharge_threshold: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            charge_threshold: 8.0,
            discharge_threshold: 6.0,
        }
    }
}

pub fn help() {
    println!("HYPERPARAMETERS:");
    println!("  charge_threshold:    float range=[4.0, 12.0] default=8.0  # $/MWh below DA-median to charge");
    println!("  discharge_threshold: float range=[3.0, 10.0] default=6.0  # $/MWh above DA-median to discharge");
    println!();
    println!("Refines v18 by re-tuning charge/discharge thresholds for s=congested. v18's profit-weighted");
    println!("line softening (profit_weight_power=2.545) and slope_factor=2.0 are held fixed.");
}

fn parse_hp(h: &Option<Map<String, Value>>) -> Hyperparameters {
    let mut out = Hyperparameters::default();
    if let Some(m) = h {
        if let Some(v) = m.get("charge_threshold").and_then(|v| v.as_f64()) {
            out.charge_threshold = v;
        }
        if let Some(v) = m.get("discharge_threshold").and_then(|v| v.as_f64()) {
            out.discharge_threshold = v;
        }
    }
    out
}

const LOOKAHEAD_STEPS: usize = 48;
const END_WINDOW: usize = 24;
const END_FACTOR: f64 = 0.0;
const SLOPE_FACTOR: f64 = 2.0;
const PROFIT_WEIGHT_POWER: f64 = 2.545;

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

fn policy_with_hp(challenge: &Challenge, state: &State, hp: &Hyperparameters) -> Result<Vec<f64>> {
    let t = state.time_step;
    let end = (t + 1 + LOOKAHEAD_STEPS).min(challenge.num_steps);
    let da = &challenge.market.day_ahead_prices;

    let remaining = challenge.num_steps.saturating_sub(t);
    let in_end_window = END_WINDOW > 0 && remaining <= END_WINDOW;
    let (discharge_th, charge_th, suppress_charge) = if in_end_window {
        let progress = (remaining as f64 / END_WINDOW as f64).clamp(0.0, 1.0);
        let scale = END_FACTOR + (1.0 - END_FACTOR) * progress;
        (hp.discharge_threshold * scale, hp.charge_threshold, true)
    } else {
        (hp.discharge_threshold, hp.charge_threshold, false)
    };
    let slope = SLOPE_FACTOR.max(0.05);

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
            profit[i] = hp.discharge_threshold;
            continue;
        }
        let reference = median_of(&mut buf);
        let current = state.rt_prices[node];
        let diff = current - reference;
        profit[i] = diff.abs();

        let (u_min, u_max) = state.action_bounds[i];
        let dt = discharge_th.max(0.5);
        if diff > dt {
            let span = (slope * dt).max(1e-6);
            let intensity = ((diff - dt) / span).clamp(0.0, 1.0);
            action[i] = intensity * u_max;
        } else if !suppress_charge && diff < -charge_th {
            let span = (slope * charge_th).max(1e-6);
            let intensity = ((-diff - charge_th) / span).clamp(0.0, 1.0);
            action[i] = intensity * u_min;
        }
    }

    enforce_feasible(challenge, state, action, &profit, PROFIT_WEIGHT_POWER)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
