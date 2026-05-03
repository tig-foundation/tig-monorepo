use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::{constants, Battery, Challenge, State};

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Hyperparameters {
    pub horizon_steps: usize,
    pub min_window_std: f64,
    pub profit_floor_shrink: f64,
    pub jump_z_threshold: f64,
    pub dp_soc_levels: usize,
    pub dp_action_levels: usize,
    /// Legacy smooth-policy parameters retained for serde compat with prior submissions.
    pub urgency_gain: f64,
    pub rt_z_gain: f64,
    pub action_sharpness: f64,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            horizon_steps: 192,
            min_window_std: 0.0,
            profit_floor_shrink: 0.95,
            jump_z_threshold: 5.0,
            dp_soc_levels: 24,
            dp_action_levels: 21,
            urgency_gain: 1.0,
            rt_z_gain: 1.0,
            action_sharpness: 0.30,
        }
    }
}

pub fn help() {
    println!(
        "nodal_pair_arb: per-battery dynamic-programming policy with PTDF-coupled \
pair refinement. For each battery, computes a backward DP value table V[t][soc] \
using day-ahead prices at the battery's node. At each step, picks the action that \
maximises `reward(u, RT_now) + V[t+1][soc(u)]`. Tail RT jumps trigger a full-bound \
bang override; PTDF feasibility is enforced via greedy line softening + symmetric \
expansion, followed by coordinate refinement, an LP-flavoured null-space \
redistribution on tight lines, joint pair refinement on opposing-PTDF battery \
pairs, and stochastic local search. The pair refinement is the structural \
addition over single-coordinate methods — it discovers Pareto-improving joint \
moves that exist only in the constraint null-space of binding lines. \
Hyperparameters: horizon_steps, min_window_std, jump_z_threshold, dp_soc_levels, \
dp_action_levels. Legacy struct fields are retained for serde compatibility."
    );
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&tig_challenges::energy_arbitrage::Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp = parse_hyperparameters(hyperparameters);
    // Precompute per-battery DP value tables once for the whole rollout.
    let dp_tables: Vec<(Vec<Vec<f64>>, f64)> = challenge
        .batteries
        .iter()
        .map(|b| compute_battery_dp(challenge, b, hp.dp_soc_levels, hp.dp_action_levels))
        .collect();
    let solution = challenge.grid_optimize(&|c, s| policy(c, s, &hp, &dp_tables))?;
    save_solution(&solution)?;
    Ok(())
}

fn parse_hyperparameters(raw: &Option<Map<String, Value>>) -> Hyperparameters {
    let mut hp: Hyperparameters = raw
        .as_ref()
        .and_then(|m| serde_json::from_value(Value::Object(m.clone())).ok())
        .unwrap_or_default();
    hp.horizon_steps = hp.horizon_steps.max(1);
    hp.dp_soc_levels = hp.dp_soc_levels.max(2);
    hp.dp_action_levels = hp.dp_action_levels.max(3);
    if !hp.min_window_std.is_finite() || hp.min_window_std < 0.0 {
        hp.min_window_std = 0.0;
    }
    if !hp.jump_z_threshold.is_finite() || hp.jump_z_threshold < 0.0 {
        hp.jump_z_threshold = 0.0;
    }
    hp
}

const MAX_FLOW_ADJUST_ITERS: usize = 64;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;
const COORDINATE_REFINEMENT_PASSES: usize = 4;

// Pair refinement: joint search over opposing-PTDF battery pairs on tight lines.
// Constants tuned via coordinate descent on 50-instance benchmark (5 scenarios × 10 seeds).
const PAIR_REFINEMENT_PASSES: usize = 6;
const PAIR_GRID_LEVELS: usize = 17;
const PAIR_TIGHT_THRESHOLD: f64 = 0.1;
const PAIR_TIGHT_LINES_PER_PASS: usize = 150;
const PAIR_TOP_K_PER_LINE: usize = 10;

// Stochastic perturbation: random pair tries to catch what deterministic search misses.
const STOCHASTIC_TRIES: usize = 48;
const STOCHASTIC_SCALE: f64 = 0.3;

// LP-flavoured redistribution: targeted shrink-grow along tight-line null-space.
const REDISTRIBUTE_TIGHT_THRESHOLD: f64 = 0.7;
const REDISTRIBUTE_TIGHT_LINES: usize = 12;

const EPS: f64 = 1e-12;

#[derive(Clone, Copy)]
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

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let violation = flow.abs() - limit;
        if violation > constants::EPS_FLOW * limit {
            let candidate = Violation {
                line: l,
                flow,
                amount: violation,
            };
            match best {
                Some(current) if candidate.amount <= current.amount => {}
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

fn soften_most_violated_line(challenge: &Challenge, v: Violation, action: &mut [f64]) -> bool {
    let dir = v.flow.signum();
    if dir.abs() <= EPS {
        return false;
    }
    let mut idxs = Vec::new();
    let mut strength = 0.0;
    for (i, battery) in challenge.batteries.iter().enumerate() {
        let contrib = challenge.network.ptdf[v.line][battery.node] * action[i];
        let signed = dir * contrib;
        if signed > EPS {
            strength += signed;
            idxs.push(i);
        }
    }
    if idxs.is_empty() || strength <= EPS {
        return false;
    }
    let keep = (1.0 - v.amount / strength).clamp(0.0, 1.0);
    if (1.0 - keep).abs() <= EPS {
        return false;
    }
    for i in idxs {
        action[i] *= keep;
    }
    true
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
) -> Result<Vec<f64>> {
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let flows = compute_flows(challenge, state, &action);
        let Some(v) = most_violated_line(challenge, &flows) else {
            return Ok(action);
        };
        if !soften_most_violated_line(challenge, v, &mut action) {
            break;
        }
    }
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }
    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!("Grid infeasible even at zero action"));
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

/// After the projection, the action may be feasible but with slack on every line.
/// Find the largest scalar α ≥ 1 such that α * action is still feasible (PTDF + per-battery
/// bounds), and apply it. Captures any slack the greedy projection left on the table.
fn expand_to_feasibility_limit(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    if action.iter().all(|u| u.abs() <= EPS) {
        return action.to_vec();
    }
    let mut alpha_max = f64::INFINITY;
    for (i, &u) in action.iter().enumerate() {
        if u.abs() <= EPS {
            continue;
        }
        let (lo, hi) = state.action_bounds[i];
        let bound = if u > 0.0 { hi / u } else { lo / u };
        if bound.is_finite() && bound < alpha_max {
            alpha_max = bound;
        }
    }
    if !alpha_max.is_finite() || alpha_max <= 1.0 + EPS {
        return action.to_vec();
    }
    let mut lo_a = 1.0;
    let mut hi_a = alpha_max.max(1.0);
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (lo_a + hi_a);
        let scaled: Vec<f64> = action.iter().map(|u| mid * u).collect();
        if is_flow_feasible(challenge, state, &scaled) {
            lo_a = mid;
        } else {
            hi_a = mid;
        }
    }
    if lo_a <= 1.0 + EPS {
        return action.to_vec();
    }
    action
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let (lo, hi) = state.action_bounds[i];
            (lo_a * u).clamp(lo, hi)
        })
        .collect()
}

fn node_window_std(day_ahead_prices: &[Vec<f64>], node: usize, start: usize, end: usize) -> f64 {
    let n = end.saturating_sub(start) as f64;
    if n <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for prices in &day_ahead_prices[start..end] {
        let p = prices[node];
        sum += p;
        sum_sq += p * p;
    }
    let mean = sum / n;
    (sum_sq / n - mean * mean).max(0.0).sqrt()
}

fn action_value(
    battery: &Battery,
    state: &State,
    battery_idx: usize,
    action: f64,
    price: f64,
    v_next: &[f64],
    soc_step: f64,
) -> f64 {
    let dt = constants::DELTA_T;
    let abs_u = action.abs();
    let revenue = action * price * dt;
    let tx = constants::KAPPA_TX * abs_u * dt;
    let deg =
        constants::KAPPA_DEG * ((abs_u * dt) / battery.capacity_mwh).powf(constants::BETA_DEG);
    let new_soc = battery.apply_action_to_soc(action, state.socs[battery_idx]);
    let new_soc_idx = (((new_soc - battery.soc_min_mwh) / soc_step.max(EPS)).round() as i32)
        .max(0)
        .min((v_next.len().saturating_sub(1)) as i32) as usize;
    revenue - tx - deg + v_next[new_soc_idx]
}

fn delta_keeps_flows_feasible(
    challenge: &Challenge,
    flows: &[f64],
    node: usize,
    delta: f64,
) -> bool {
    for (line, &flow) in flows.iter().enumerate() {
        let next_flow = flow + challenge.network.ptdf[line][node] * delta;
        let limit = challenge.network.flow_limits[line];
        if next_flow.abs() - limit > constants::EPS_FLOW * limit {
            return false;
        }
    }
    true
}

fn apply_flow_delta(challenge: &Challenge, flows: &mut [f64], node: usize, delta: f64) {
    for (line, flow) in flows.iter_mut().enumerate() {
        *flow += challenge.network.ptdf[line][node] * delta;
    }
}

fn refine_coordinate_actions(
    challenge: &Challenge,
    state: &State,
    hp: &Hyperparameters,
    dp_tables: &[(Vec<Vec<f64>>, f64)],
    mut action: Vec<f64>,
) -> Vec<f64> {
    let t = state.time_step;
    let n_actions = hp.dp_action_levels.max(3);
    let mut flows = compute_flows(challenge, state, &action);

    for _ in 0..COORDINATE_REFINEMENT_PASSES {
        let mut changed = false;
        for (i, battery) in challenge.batteries.iter().enumerate() {
            let Some((v_tab, soc_step)) = dp_tables.get(i) else {
                continue;
            };
            if t + 1 >= v_tab.len() {
                continue;
            }
            let (lo, hi) = state.action_bounds[i];
            if hi - lo <= EPS {
                continue;
            }

            let node = battery.node;
            let price = state.rt_prices[node];
            let v_next = &v_tab[t + 1];
            let mut best_u = action[i];
            let mut best_value = action_value(battery, state, i, best_u, price, v_next, *soc_step);

            for j in 0..n_actions {
                let frac = (j as f64) / (n_actions - 1).max(1) as f64 * 2.0 - 1.0;
                let u_raw = if frac < 0.0 {
                    frac * battery.power_charge_mw
                } else {
                    frac * battery.power_discharge_mw
                };
                let u = u_raw.clamp(lo, hi);
                let delta = u - action[i];
                if delta.abs() <= EPS {
                    continue;
                }
                let value = action_value(battery, state, i, u, price, v_next, *soc_step);
                if value <= best_value + EPS {
                    continue;
                }
                if delta_keeps_flows_feasible(challenge, &flows, node, delta) {
                    best_u = u;
                    best_value = value;
                }
            }

            let delta = best_u - action[i];
            if delta.abs() > EPS {
                action[i] = best_u;
                apply_flow_delta(challenge, &mut flows, node, delta);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    action
}

/// Check whether changing battery `i` by Δu_i and battery `j` by Δu_j keeps every
/// line within its limit, given current flow vector.
fn pair_delta_feasible(
    challenge: &Challenge,
    flows: &[f64],
    n_i: usize,
    du_i: f64,
    n_j: usize,
    du_j: f64,
) -> bool {
    for (l, &flow) in flows.iter().enumerate() {
        let next = flow
            + challenge.network.ptdf[l][n_i] * du_i
            + challenge.network.ptdf[l][n_j] * du_j;
        let limit = challenge.network.flow_limits[l];
        if next.abs() - limit > constants::EPS_FLOW * limit {
            return false;
        }
    }
    true
}

fn apply_pair_flow_delta(
    challenge: &Challenge,
    flows: &mut [f64],
    n_i: usize,
    du_i: f64,
    n_j: usize,
    du_j: f64,
) {
    for (l, flow) in flows.iter_mut().enumerate() {
        *flow += challenge.network.ptdf[l][n_i] * du_i
            + challenge.network.ptdf[l][n_j] * du_j;
    }
}

/// Pair refinement on tight lines. For each line that's near its limit, find the
/// top-K batteries by |ptdf| to that line and try every (u_i, u_j) pair from a
/// small joint action grid. Accepts joint moves that preserve PTDF feasibility on
/// every line and improve the sum `value(i) + value(j)`. Catches LP-style
/// redistributions (one battery shrinks, another with opposite ptdf grows along
/// the constraint null-space) that single-coordinate refinement cannot find.
fn refine_pair_actions(
    challenge: &Challenge,
    state: &State,
    hp: &Hyperparameters,
    dp_tables: &[(Vec<Vec<f64>>, f64)],
    mut action: Vec<f64>,
) -> Vec<f64> {
    let n = action.len();
    let t = state.time_step;
    let l_count = challenge.network.num_lines;
    let n_grid = PAIR_GRID_LEVELS.max(3);
    let passes = PAIR_REFINEMENT_PASSES;
    let tight_thresh = PAIR_TIGHT_THRESHOLD;
    let lines_per_pass = PAIR_TIGHT_LINES_PER_PASS;
    let top_k = PAIR_TOP_K_PER_LINE;
    let mut flows = compute_flows(challenge, state, &action);

    // Per-battery raw action grid (before per-state bounds clamp).
    let raw_grid: Vec<Vec<f64>> = challenge
        .batteries
        .iter()
        .map(|b| {
            (0..n_grid)
                .map(|j| {
                    let frac = (j as f64) / (n_grid - 1).max(1) as f64 * 2.0 - 1.0;
                    if frac < 0.0 {
                        frac * b.power_charge_mw
                    } else {
                        frac * b.power_discharge_mw
                    }
                })
                .collect()
        })
        .collect();

    for _ in 0..passes {
        let mut changed = false;

        // Identify tight lines (sorted by tightness, descending).
        let mut tight: Vec<(usize, f64)> = (0..l_count)
            .map(|l| {
                let lim = challenge.network.flow_limits[l].max(EPS);
                (l, flows[l].abs() / lim)
            })
            .filter(|&(_, r)| r > tight_thresh)
            .collect();
        tight.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(l, _) in tight.iter().take(lines_per_pass) {
            // Top batteries by |ptdf| to this line.
            let mut by_ptdf: Vec<(usize, f64)> = (0..n)
                .map(|i| (i, challenge.network.ptdf[l][challenge.batteries[i].node]))
                .filter(|&(_, p)| p.abs() > EPS)
                .collect();
            by_ptdf.sort_unstable_by(|a, b| {
                b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top = top_k.min(by_ptdf.len());
            for ki in 0..top {
                let (i, _) = by_ptdf[ki];
                let bi = &challenge.batteries[i];
                let (lo_i, hi_i) = state.action_bounds[i];
                if hi_i - lo_i <= EPS {
                    continue;
                }
                let Some((v_tab_i, soc_step_i)) = dp_tables.get(i) else { continue };
                if t + 1 >= v_tab_i.len() {
                    continue;
                }
                let v_next_i = &v_tab_i[t + 1];
                let price_i = state.rt_prices[bi.node];

                for kj in (ki + 1)..top {
                    let (j, _) = by_ptdf[kj];
                    let bj = &challenge.batteries[j];
                    let (lo_j, hi_j) = state.action_bounds[j];
                    if hi_j - lo_j <= EPS {
                        continue;
                    }
                    let Some((v_tab_j, soc_step_j)) = dp_tables.get(j) else { continue };
                    if t + 1 >= v_tab_j.len() {
                        continue;
                    }
                    let v_next_j = &v_tab_j[t + 1];
                    let price_j = state.rt_prices[bj.node];

                    let n_i_node = bi.node;
                    let n_j_node = bj.node;

                    let base_val = action_value(bi, state, i, action[i], price_i, v_next_i, *soc_step_i)
                        + action_value(bj, state, j, action[j], price_j, v_next_j, *soc_step_j);
                    let mut best_u_i = action[i];
                    let mut best_u_j = action[j];
                    let mut best_val = base_val;

                    for ai in 0..n_grid {
                        let u_i_cand = raw_grid[i][ai].clamp(lo_i, hi_i);
                        let du_i = u_i_cand - action[i];
                        for aj in 0..n_grid {
                            let u_j_cand = raw_grid[j][aj].clamp(lo_j, hi_j);
                            let du_j = u_j_cand - action[j];
                            if du_i.abs() <= EPS && du_j.abs() <= EPS {
                                continue;
                            }
                            if !pair_delta_feasible(challenge, &flows, n_i_node, du_i, n_j_node, du_j) {
                                continue;
                            }
                            let new_val = action_value(bi, state, i, u_i_cand, price_i, v_next_i, *soc_step_i)
                                + action_value(bj, state, j, u_j_cand, price_j, v_next_j, *soc_step_j);
                            if new_val > best_val + EPS {
                                best_val = new_val;
                                best_u_i = u_i_cand;
                                best_u_j = u_j_cand;
                            }
                        }
                    }

                    let du_i = best_u_i - action[i];
                    let du_j = best_u_j - action[j];
                    if du_i.abs() > EPS || du_j.abs() > EPS {
                        action[i] = best_u_i;
                        action[j] = best_u_j;
                        apply_pair_flow_delta(challenge, &mut flows, n_i_node, du_i, n_j_node, du_j);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    action
}

/// Stochastic local search over pair perturbations. Picks a deterministic-seeded
/// pseudo-random sequence of (i, j, Δ_i, Δ_j) tuples and accepts the move if it's
/// PTDF-feasible and improves total value. Complements pair refinement by trying
/// pairs that don't naturally share a tight line.
fn refine_stochastic(
    challenge: &Challenge,
    state: &State,
    dp_tables: &[(Vec<Vec<f64>>, f64)],
    mut action: Vec<f64>,
) -> Vec<f64> {
    let n = action.len();
    if n < 2 {
        return action;
    }
    let t = state.time_step;
    let mut flows = compute_flows(challenge, state, &action);

    // Deterministic LCG seeded by time step (so repeat runs are reproducible).
    let mut rng: u64 = (t as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xBF58_476D_1CE4_E5B9);
    let mut next = || -> u64 {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        rng
    };
    let to_unit = |x: u64| -> f64 { (x >> 11) as f64 / (1u64 << 53) as f64 }; // [0, 1)
    let to_centred = |x: u64| -> f64 { to_unit(x) * 2.0 - 1.0 };

    let stoch_scale = STOCHASTIC_SCALE;
    for _ in 0..STOCHASTIC_TRIES {
        let i = (next() as usize) % n;
        let mut j = (next() as usize) % n;
        if j == i {
            j = (j + 1) % n;
        }
        let bi = &challenge.batteries[i];
        let bj = &challenge.batteries[j];
        let (lo_i, hi_i) = state.action_bounds[i];
        let (lo_j, hi_j) = state.action_bounds[j];
        if hi_i - lo_i <= EPS || hi_j - lo_j <= EPS {
            continue;
        }
        let Some((v_tab_i, soc_step_i)) = dp_tables.get(i) else { continue };
        let Some((v_tab_j, soc_step_j)) = dp_tables.get(j) else { continue };
        if t + 1 >= v_tab_i.len() || t + 1 >= v_tab_j.len() {
            continue;
        }
        let v_next_i = &v_tab_i[t + 1];
        let v_next_j = &v_tab_j[t + 1];

        let scale_i = (hi_i - lo_i) * stoch_scale;
        let scale_j = (hi_j - lo_j) * stoch_scale;
        let du_i_raw = to_centred(next()) * scale_i;
        let du_j_raw = to_centred(next()) * scale_j;
        let u_i_new = (action[i] + du_i_raw).clamp(lo_i, hi_i);
        let u_j_new = (action[j] + du_j_raw).clamp(lo_j, hi_j);
        let du_i = u_i_new - action[i];
        let du_j = u_j_new - action[j];
        if du_i.abs() <= EPS && du_j.abs() <= EPS {
            continue;
        }
        let n_i_node = bi.node;
        let n_j_node = bj.node;
        if !pair_delta_feasible(challenge, &flows, n_i_node, du_i, n_j_node, du_j) {
            continue;
        }
        let price_i = state.rt_prices[n_i_node];
        let price_j = state.rt_prices[n_j_node];
        let base_val = action_value(bi, state, i, action[i], price_i, v_next_i, *soc_step_i)
            + action_value(bj, state, j, action[j], price_j, v_next_j, *soc_step_j);
        let new_val = action_value(bi, state, i, u_i_new, price_i, v_next_i, *soc_step_i)
            + action_value(bj, state, j, u_j_new, price_j, v_next_j, *soc_step_j);
        if new_val > base_val + EPS {
            let final_du_i = u_i_new - action[i];
            let final_du_j = u_j_new - action[j];
            action[i] = u_i_new;
            action[j] = u_j_new;
            apply_pair_flow_delta(challenge, &mut flows, n_i_node, final_du_i, n_j_node, final_du_j);
        }
    }

    action
}

/// LP-flavoured redistribution. For each tight line, computes a one-step
/// continuous reallocation: shrink the largest positive contributor while growing
/// a battery with opposite-sign ptdf to take up the freed flow headroom. Accepts
/// the move only if the joint reward improves and all line constraints stay
/// satisfied. Replaces a full LP solve with a single targeted balancing step.
fn redistribute_balance(
    challenge: &Challenge,
    state: &State,
    dp_tables: &[(Vec<Vec<f64>>, f64)],
    mut action: Vec<f64>,
) -> Vec<f64> {
    let n = action.len();
    let t = state.time_step;
    let l_count = challenge.network.num_lines;
    let mut flows = compute_flows(challenge, state, &action);

    let mut tight: Vec<(usize, f64, f64)> = (0..l_count)
        .map(|l| {
            let lim = challenge.network.flow_limits[l].max(EPS);
            (l, flows[l] / lim, flows[l].abs() / lim)
        })
        .filter(|&(_, _, r)| r > REDISTRIBUTE_TIGHT_THRESHOLD)
        .collect();
    tight.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    for &(l, signed, _) in tight.iter().take(REDISTRIBUTE_TIGHT_LINES) {
        let dir = signed.signum();
        if dir.abs() <= EPS {
            continue;
        }
        // Battery i: shrink (largest signed contribution to line l).
        // Battery j: grow into the freed space (largest opposite-sign capacity).
        let mut best_i: Option<(usize, f64)> = None;
        let mut best_j: Option<(usize, f64)> = None;
        for k in 0..n {
            let p = challenge.network.ptdf[l][challenge.batteries[k].node];
            let contrib = dir * p * action[k]; // positive = pushing flow further toward limit
            if contrib > EPS {
                if best_i.map_or(true, |(_, c)| contrib > c) {
                    best_i = Some((k, contrib));
                }
            }
            // For j: looking for a battery whose ptdf has *opposite* sign — moving
            // it in its growth direction reduces flow toward limit_dir.
            let opp = -dir * p; // > 0 means moving u_k positive reduces flow
            let (lo, hi) = state.action_bounds[k];
            let headroom = if opp > EPS { hi - action[k] } else { action[k] - lo };
            let capacity = opp.abs() * headroom.max(0.0);
            if capacity > EPS {
                if best_j.map_or(true, |(_, c)| capacity > c) {
                    best_j = Some((k, capacity));
                }
            }
        }

        let (Some((i, _)), Some((j, _))) = (best_i, best_j) else { continue };
        if i == j {
            continue;
        }
        let bi = &challenge.batteries[i];
        let bj = &challenge.batteries[j];
        let (lo_i, hi_i) = state.action_bounds[i];
        let (lo_j, hi_j) = state.action_bounds[j];
        let n_i_node = bi.node;
        let n_j_node = bj.node;
        let p_li = challenge.network.ptdf[l][n_i_node];
        let p_lj = challenge.network.ptdf[l][n_j_node];
        if p_li.abs() <= EPS || p_lj.abs() <= EPS {
            continue;
        }

        let Some((v_tab_i, soc_step_i)) = dp_tables.get(i) else { continue };
        let Some((v_tab_j, soc_step_j)) = dp_tables.get(j) else { continue };
        if t + 1 >= v_tab_i.len() || t + 1 >= v_tab_j.len() {
            continue;
        }
        let v_next_i = &v_tab_i[t + 1];
        let v_next_j = &v_tab_j[t + 1];
        let price_i = state.rt_prices[n_i_node];
        let price_j = state.rt_prices[n_j_node];

        // Try several balanced (Δu_i, Δu_j) along the null-space of line l:
        //   p_li · Δu_i + p_lj · Δu_j = 0  →  Δu_j = -(p_li / p_lj) · Δu_i.
        // Sweep Δu_i over a small set of magnitudes.
        let ratio = -(p_li / p_lj);
        let scale_i = (hi_i - lo_i).max(EPS) * 0.5;
        let candidates: [f64; 7] = [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5];

        let base_val = action_value(bi, state, i, action[i], price_i, v_next_i, *soc_step_i)
            + action_value(bj, state, j, action[j], price_j, v_next_j, *soc_step_j);
        let mut best_u_i = action[i];
        let mut best_u_j = action[j];
        let mut best_val = base_val;

        for &c in &candidates {
            let u_i_new = (action[i] + c * scale_i).clamp(lo_i, hi_i);
            let du_i = u_i_new - action[i];
            if du_i.abs() <= EPS {
                continue;
            }
            let u_j_new = (action[j] + ratio * du_i).clamp(lo_j, hi_j);
            let du_j = u_j_new - action[j];
            if !pair_delta_feasible(challenge, &flows, n_i_node, du_i, n_j_node, du_j) {
                continue;
            }
            let new_val = action_value(bi, state, i, u_i_new, price_i, v_next_i, *soc_step_i)
                + action_value(bj, state, j, u_j_new, price_j, v_next_j, *soc_step_j);
            if new_val > best_val + EPS {
                best_val = new_val;
                best_u_i = u_i_new;
                best_u_j = u_j_new;
            }
        }

        let du_i = best_u_i - action[i];
        let du_j = best_u_j - action[j];
        if du_i.abs() > EPS || du_j.abs() > EPS {
            action[i] = best_u_i;
            action[j] = best_u_j;
            apply_pair_flow_delta(challenge, &mut flows, n_i_node, du_i, n_j_node, du_j);
        }
    }

    action
}

/// Compute per-battery value function V[t][soc_idx] via backward DP using the full
/// day-ahead price series at the battery's node. V[H] = 0; works back to t=0.
/// Returns (V_table, soc_step). The DP plans against DA prices only — the per-step
/// policy substitutes the realised RT price into the immediate-reward term.
pub fn compute_battery_dp(
    challenge: &Challenge,
    battery: &Battery,
    n_soc_levels: usize,
    n_actions: usize,
) -> (Vec<Vec<f64>>, f64) {
    let n_soc_levels = n_soc_levels.max(2);
    let n_actions = n_actions.max(3);
    let h = challenge.num_steps;
    let n = battery.node;
    let soc_min = battery.soc_min_mwh;
    let soc_max = battery.soc_max_mwh;
    let soc_step = (soc_max - soc_min) / (n_soc_levels - 1).max(1) as f64;
    let dt = constants::DELTA_T;
    let kappa_tx = constants::KAPPA_TX;
    let kappa_deg = constants::KAPPA_DEG;
    let beta_deg = constants::BETA_DEG;

    // Action grid spans [-power_charge, +power_discharge] linearly.
    let action_grid: Vec<f64> = (0..n_actions)
        .map(|i| {
            let frac = (i as f64) / (n_actions - 1).max(1) as f64 * 2.0 - 1.0;
            if frac < 0.0 {
                frac * battery.power_charge_mw
            } else {
                frac * battery.power_discharge_mw
            }
        })
        .collect();

    let mut v = vec![vec![0.0_f64; n_soc_levels]; h + 1];
    for t in (0..h).rev() {
        let p_da = challenge.market.day_ahead_prices[t][n];
        for soc_idx in 0..n_soc_levels {
            let soc = soc_min + soc_idx as f64 * soc_step;
            let headroom = (soc_max - soc).max(0.0);
            let available = (soc - soc_min).max(0.0);
            let max_charge = (headroom / (battery.efficiency_charge.max(EPS) * dt))
                .min(battery.power_charge_mw)
                .max(0.0);
            let max_discharge = (available * battery.efficiency_discharge / dt)
                .min(battery.power_discharge_mw)
                .max(0.0);
            let lo = -max_charge;
            let hi = max_discharge;
            let mut best = f64::NEG_INFINITY;
            for &u_raw in &action_grid {
                let u = u_raw.clamp(lo, hi);
                let revenue = u * p_da * dt;
                let abs_u = u.abs();
                let tx = kappa_tx * abs_u * dt;
                let deg = kappa_deg * ((abs_u * dt) / battery.capacity_mwh).powf(beta_deg);
                let reward = revenue - tx - deg;
                let new_soc = battery.apply_action_to_soc(u, soc);
                let new_soc_idx = (((new_soc - soc_min) / soc_step.max(EPS)).round() as i32)
                    .max(0)
                    .min((n_soc_levels - 1) as i32) as usize;
                let value = reward + v[t + 1][new_soc_idx];
                if value > best {
                    best = value;
                }
            }
            v[t][soc_idx] = if best.is_finite() { best } else { 0.0 };
        }
    }
    (v, soc_step)
}

pub fn policy(
    challenge: &Challenge,
    state: &State,
    hp: &Hyperparameters,
    dp_tables: &[(Vec<Vec<f64>>, f64)],
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let h = challenge.num_steps;
    let k = hp.horizon_steps.min(h.saturating_sub(t));
    let da = &challenge.market.day_ahead_prices;
    if t >= da.len() {
        return Err(anyhow!("DA prices missing for step {}", t));
    }
    let mut action = vec![0.0_f64; challenge.num_batteries];
    let dt = constants::DELTA_T;
    let kappa_tx = constants::KAPPA_TX;
    let kappa_deg = constants::KAPPA_DEG;
    let beta_deg = constants::BETA_DEG;
    let n_actions = hp.dp_action_levels.max(3);

    for (i, battery) in challenge.batteries.iter().enumerate() {
        let n = battery.node;
        let (lo, hi) = state.action_bounds[i];
        if (hi - lo).abs() <= EPS {
            continue;
        }
        let end = (t + k).min(h);
        if end <= t {
            continue;
        }
        let std = node_window_std(da, n, t, end);
        if !std.is_finite() || std < hp.min_window_std {
            continue;
        }
        let p_now = da[t][n];
        let z_rt = (state.rt_prices[n] - p_now) / std.max(EPS);

        // Tail-jump override
        if hp.jump_z_threshold > 0.0 && z_rt.abs() >= hp.jump_z_threshold {
            action[i] = if z_rt > 0.0 { hi } else { lo };
            continue;
        }

        // DP value-driven action selection: maximise reward(u, RT_now) + V[t+1][soc(u)]
        if i >= dp_tables.len() {
            continue;
        }
        let (ref v_tab, soc_step) = dp_tables[i];
        if t + 1 >= v_tab.len() {
            continue;
        }
        let v_next = &v_tab[t + 1];
        let soc_min_i = battery.soc_min_mwh;
        let p_rt = state.rt_prices[n];
        let mut best_value = f64::NEG_INFINITY;
        let mut best_u = 0.0_f64;
        for j in 0..n_actions {
            let frac = (j as f64) / (n_actions - 1).max(1) as f64 * 2.0 - 1.0;
            let u_raw = if frac < 0.0 {
                frac * battery.power_charge_mw
            } else {
                frac * battery.power_discharge_mw
            };
            let u = u_raw.clamp(lo, hi);
            let revenue = u * p_rt * dt;
            let abs_u = u.abs();
            let tx = kappa_tx * abs_u * dt;
            let deg = kappa_deg * ((abs_u * dt) / battery.capacity_mwh).powf(beta_deg);
            let reward = revenue - tx - deg;
            let new_soc = battery.apply_action_to_soc(u, state.socs[i]);
            let n_soc_levels = v_next.len();
            let new_soc_idx = (((new_soc - soc_min_i) / soc_step.max(EPS)).round() as i32)
                .max(0)
                .min((n_soc_levels - 1) as i32) as usize;
            let value = reward + v_next[new_soc_idx];
            if value > best_value {
                best_value = value;
                best_u = u;
            }
        }
        action[i] = best_u;
    }

    let action = enforce_flow_feasibility(challenge, state, action)?;
    let action = expand_to_feasibility_limit(challenge, state, &action);
    let action = refine_coordinate_actions(challenge, state, hp, dp_tables, action);
    let action = redistribute_balance(challenge, state, dp_tables, action);
    let action = refine_pair_actions(challenge, state, hp, dp_tables, action);
    let action = refine_stochastic(challenge, state, dp_tables, action);
    let mut action = expand_to_feasibility_limit(challenge, state, &action);
    // Defensive: take_step rejects actions that exceed bounds by even one ULP,
    // and the scaling/clamping arithmetic above can produce sub-ULP overshoots.
    for (i, u) in action.iter_mut().enumerate() {
        let (lo, hi) = state.action_bounds[i];
        if *u < lo { *u = lo; } else if *u > hi { *u = hi; }
    }
    Ok(action)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
