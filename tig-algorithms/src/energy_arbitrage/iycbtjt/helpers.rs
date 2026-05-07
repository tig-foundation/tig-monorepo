use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;
use tig_challenges::energy_arbitrage::*;

#[derive(Clone, Debug)]
pub struct TrackHp {
    pub soc_levels: usize,
    pub action_grid: usize,
    pub asca_iters: usize,
    pub ternary_iters: usize,
    pub convergence_tol: f64,
    pub anticipate_lmp: bool,
    pub lmp_threshold: f64,
    pub lmp_premium_scale: f64,
    pub jump_premium: f64,
    pub prune_ratio: f64,
    pub deflator_iters: usize,
    pub flow_margin: f64,
    pub network_derating: f64,
}

impl TrackHp {
    pub fn override_from_map(&mut self, h: &Option<Map<String, Value>>) {
        let Some(m) = h else { return };
        if let Some(v) = m.get("soc_levels").and_then(|v| v.as_u64()) { self.soc_levels = (v as usize).max(3); }
        if let Some(v) = m.get("action_grid").and_then(|v| v.as_u64()) { self.action_grid = (v as usize).max(4); }
        if let Some(v) = m.get("asca_iters").and_then(|v| v.as_u64()) { self.asca_iters = v as usize; }
        if let Some(v) = m.get("ternary_iters").and_then(|v| v.as_u64()) { self.ternary_iters = v as usize; }
        if let Some(v) = m.get("convergence_tol").and_then(|v| v.as_f64()) { self.convergence_tol = v; }
        if let Some(v) = m.get("anticipate_lmp").and_then(|v| v.as_bool()) { self.anticipate_lmp = v; }
        if let Some(v) = m.get("lmp_threshold").and_then(|v| v.as_f64()) { self.lmp_threshold = v; }
        if let Some(v) = m.get("lmp_premium_scale").and_then(|v| v.as_f64()) { self.lmp_premium_scale = v; }
        if let Some(v) = m.get("jump_premium").and_then(|v| v.as_f64()) { self.jump_premium = v; }
        if let Some(v) = m.get("prune_ratio").and_then(|v| v.as_f64()) { self.prune_ratio = v.clamp(0.0, 0.9); }
        if let Some(v) = m.get("deflator_iters").and_then(|v| v.as_u64()) { self.deflator_iters = v as usize; }
        if let Some(v) = m.get("flow_margin").and_then(|v| v.as_f64()) { self.flow_margin = v.max(0.0); }
        if let Some(v) = m.get("network_derating").and_then(|v| v.as_f64()) { self.network_derating = v.clamp(0.01, 1.0); }
    }
}

pub struct IycbtjtCache {
    pub dp: Vec<Vec<Vec<f64>>>,
    pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
    pub b_to_lines: Vec<Vec<(usize, f64)>>,
    pub batt_nodes: Vec<usize>,
}

struct Inner {
    hp: TrackHp,
    cache: Option<IycbtjtCache>,
}

thread_local! {
    static STATE: RefCell<Option<Inner>> = RefCell::new(None);
}

pub fn solve_baseline(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: TrackHp,
) -> Result<()> {
    STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
    let out = challenge.grid_optimize(&policy_baseline);
    STATE.with(|s| *s.borrow_mut() = None);
    let solution = out?;
    save_solution(&solution)?;
    Ok(())
}

fn policy_baseline(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    STATE.with(|s| -> Result<Vec<f64>> {
        let mut guard = s.borrow_mut();
        let inner = guard.as_mut().expect("Iycbtjt: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        Ok(run_small_multistart_policy(challenge, state, cache, hp, &flows_base))
    })
}

pub fn solve_congested(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: TrackHp,
) -> Result<()> {
    STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
    let out = challenge.grid_optimize(&policy_congested);
    STATE.with(|s| *s.borrow_mut() = None);
    let solution = out?;
    save_solution(&solution)?;
    Ok(())
}

fn policy_congested(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    STATE.with(|s| -> Result<Vec<f64>> {
        let mut guard = s.borrow_mut();
        let inner = guard.as_mut().expect("Iycbtjt: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        Ok(run_small_multistart_policy(challenge, state, cache, hp, &flows_base))
    })
}

pub fn solve_multiday(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: TrackHp,
) -> Result<()> {
    STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
    let out = challenge.grid_optimize(&policy_multiday);
    STATE.with(|s| *s.borrow_mut() = None);
    let solution = out?;
    save_solution(&solution)?;
    Ok(())
}

fn policy_multiday(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    STATE.with(|s| -> Result<Vec<f64>> {
        let mut guard = s.borrow_mut();
        let inner = guard.as_mut().expect("Iycbtjt: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        Ok(run_medium_multistart_policy(challenge, state, cache, hp, &flows_base))
    })
}

pub fn solve_dense(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: TrackHp,
) -> Result<()> {
    STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
    let out = challenge.grid_optimize(&policy_dense);
    STATE.with(|s| *s.borrow_mut() = None);
    let solution = out?;
    save_solution(&solution)?;
    Ok(())
}

fn policy_dense(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    STATE.with(|s| -> Result<Vec<f64>> {
        let mut guard = s.borrow_mut();
        let inner = guard.as_mut().expect("Iycbtjt: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        Ok(run_screened_multistart_policy(
            challenge,
            state,
            cache,
            hp,
            &flows_base,
            2,
        ))
    })
}

pub fn solve_capstone(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: TrackHp,
) -> Result<()> {
    STATE.with(|s| *s.borrow_mut() = Some(Inner { hp, cache: None }));
    let out = challenge.grid_optimize(&policy_capstone);
    STATE.with(|s| *s.borrow_mut() = None);
    let solution = out?;
    save_solution(&solution)?;
    Ok(())
}

fn policy_capstone(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    STATE.with(|s| -> Result<Vec<f64>> {
        let mut guard = s.borrow_mut();
        let inner = guard.as_mut().expect("Iycbtjt: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        let mut actions = vec![0.0; challenge.num_batteries];
        run_asca(challenge, state, cache, hp, &flows_base, &mut actions);
        run_deflator(challenge, state, cache, hp, &flows_base, &mut actions);
        Ok(actions)
    })
}

fn approx_actions_value(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &[f64],
) -> f64 {
    let flows = compute_action_flows(ca, flows_base, actions);
    for l in 0..challenge.network.flow_limits.len() {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if flows[l].abs() > limit + 1e-7 {
            return f64::NEG_INFINITY;
        }
    }

    let mut total = 0.0_f64;
    for b in 0..challenge.num_batteries {
        total += eval_profit(challenge, state, ca, b, actions[b]);
    }
    total
}

fn scale_actions_toward_zero_to_feasible(
    challenge: &Challenge,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_l = challenge.network.flow_limits.len();
    let flows = compute_action_flows(ca, flows_base, actions);

    let mut beta = 1.0_f64;
    let mut had_violation = false;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        let total = flows[l];
        if total.abs() <= limit + 1e-9 { continue; }
        let f_act = total - flows_base[l];
        if f_act.abs() < 1e-12 { continue; }
        had_violation = true;
        let target = if total > 0.0 { limit } else { -limit };
        let candidate = (target - flows_base[l]) / f_act;
        if candidate < beta { beta = candidate; }
    }

    if had_violation {
        let beta = beta.clamp(0.0, 1.0);
        if beta < 1.0 {
            for u in actions.iter_mut() {
                *u *= beta;
            }
        }
    }

    let flows = compute_action_flows(ca, flows_base, actions);
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if flows[l].abs() > limit + 1e-7 {
            for u in actions.iter_mut() {
                *u = 0.0;
            }
            break;
        }
    }
}

fn build_sequential_greedy_seed(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    keep: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    if num_b == 0 || keep == 0 {
        return vec![0.0; num_b];
    }

    let num_l = challenge.network.flow_limits.len();
    let mut ranked: Vec<(usize, f64)> = (0..num_b).map(|b| {
        let mut footprint = 1e-6_f64;
        for &(l, p) in &ca.b_to_lines[b] {
            let limit = challenge.network.flow_limits[l];
            if limit > 1e-6 {
                let util = (flows_base[l].abs() / limit).min(2.0);
                footprint += p.abs() * (0.25 + util * util);
            }
        }
        let score = potential(challenge, state, ca, b) / footprint
            + 1e-6 * challenge.batteries[b].capacity_mwh;
        (b, score)
    }).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut pool: Vec<usize> = ranked.into_iter().take(keep.min(num_b)).map(|(b, _)| b).collect();

    let mut actions = vec![0.0; num_b];
    let mut flows = flows_base.to_vec();

    while !pool.is_empty() {
        let mut best_pos: Option<usize> = None;
        let mut best_b = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_merit = 0.0_f64;

        for (pos, &b) in pool.iter().enumerate() {
            let (u, v) = best_action_in_window(challenge, state, ca, hp, &flows, &actions, b);
            let gain = v - eval_profit(challenge, state, ca, b, actions[b]);
            if gain <= 1e-9 || u.abs() <= 1e-9 {
                continue;
            }

            let mut stress_cost = 1.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit > 1e-6 {
                    let util = (flows[l].abs() / limit).min(2.0);
                    stress_cost += 0.02 * p.abs() * util;
                }
            }
            let merit = gain / stress_cost;
            if merit > best_merit + 1e-12 {
                best_merit = merit;
                best_pos = Some(pos);
                best_b = b;
                best_u = u;
            }
        }

        let Some(pos) = best_pos else { break; };
        let delta = best_u - actions[best_b];
        if delta.abs() > 1e-12 {
            actions[best_b] = best_u;
            for &(l, p) in &ca.b_to_lines[best_b] {
                if l < num_l {
                    flows[l] += p * delta;
                }
            }
        }
        pool.swap_remove(pos);
    }

    actions
}

fn build_scaled_greedy_seed(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    keep: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    let zero_actions = vec![0.0; num_b];
    let mut ranked: Vec<(usize, f64)> = (0..num_b).map(|b| {
        let score = potential(challenge, state, ca, b)
            + 1e-6 * challenge.batteries[b].capacity_mwh;
        (b, score)
    }).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected: Vec<usize> = ranked.into_iter().take(keep.min(num_b)).map(|(b, _)| b).collect();

    let mut scaled_seed = vec![0.0; num_b];
    for &b in &selected {
        let v0 = eval_profit(challenge, state, ca, b, 0.0);
        let (u, v) = best_action_in_window(challenge, state, ca, hp, flows_base, &zero_actions, b);
        if v > v0 + 1e-9 && u.abs() > 1e-9 {
            scaled_seed[b] = u;
        }
    }
    scale_actions_toward_zero_to_feasible(challenge, ca, hp, flows_base, &mut scaled_seed);

    let seq_seed = build_sequential_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    let scaled_score = approx_actions_value(challenge, state, ca, hp, flows_base, &scaled_seed);
    let seq_score = approx_actions_value(challenge, state, ca, hp, flows_base, &seq_seed);
    if seq_score > scaled_score + 1e-9 {
        seq_seed
    } else {
        scaled_seed
    }
}

fn move_relief_merit(
    challenge: &Challenge,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    focus_lines: &[bool],
    b: usize,
    delta: f64,
    gain: f64,
) -> f64 {
    if gain <= 1e-9 || delta.abs() <= 1e-9 {
        return 0.0;
    }

    let mut footprint = 1e-6_f64;
    let mut relief = 0.0_f64;
    let mut worsen = 0.0_f64;
    for &(l, p) in &ca.b_to_lines[b] {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        let move_flow = p * delta;
        footprint += move_flow.abs() * (0.15 + util * util);

        let focus = if focus_lines[l] { 1.0 } else { 0.0 };
        if util >= 0.55 || focus > 0.0 {
            let stress = 0.35 * util + focus;
            if flows[l] * move_flow < 0.0 {
                relief += stress * move_flow.abs();
            } else if flows[l] * move_flow > 0.0 {
                worsen += stress * move_flow.abs();
            }
        }
    }

    gain * (1.0 + 0.70 * relief) / (footprint * (1.0 + 0.85 * worsen))
}

fn build_line_relief_seed(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    keep: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b == 0 || keep == 0 {
        return vec![0.0; num_b];
    }
    if num_l == 0 {
        return build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    }

    let mut line_rank: Vec<(usize, f64)> = Vec::new();
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows_base[l].abs() / limit).min(2.0);
        if util > 0.30 {
            line_rank.push((l, util));
        }
    }
    line_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if line_rank.is_empty() {
        return build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    }

    let keep_lines = if num_b <= 40 { 2 } else { 3 };
    let mut focus_lines = vec![false; num_l];
    for (l, _) in line_rank.into_iter().take(keep_lines) {
        focus_lines[l] = true;
    }

    let zero_actions = vec![0.0; num_b];
    let mut ranked: Vec<(usize, f64)> = (0..num_b)
        .filter_map(|b| {
            let v0 = eval_profit(challenge, state, ca, b, 0.0);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, flows_base, &zero_actions, b);
            let gain = v - v0;
            if gain <= 1e-9 || u.abs() <= 1e-9 {
                return None;
            }

            let score = move_relief_merit(challenge, ca, hp, flows_base, &focus_lines, b, u, gain)
                + 0.03 * gain
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            Some((b, score))
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.is_empty() {
        return build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    }

    let mut pool: Vec<usize> = ranked.into_iter().take(keep.min(num_b)).map(|(b, _)| b).collect();
    let mut actions = vec![0.0; num_b];
    let mut flows = flows_base.to_vec();

    while !pool.is_empty() {
        let mut best_pos: Option<usize> = None;
        let mut best_b = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_merit = 0.0_f64;

        for (pos, &b) in pool.iter().enumerate() {
            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, &flows, &actions, b);
            let gain = v - v_cur;
            let delta = u - actions[b];
            if gain <= 1e-9 || delta.abs() <= 1e-9 {
                continue;
            }

            let merit = move_relief_merit(challenge, ca, hp, &flows, &focus_lines, b, delta, gain)
                + 0.02 * gain / (1.0 + actions[b].abs());
            if merit > best_merit + 1e-12 {
                best_merit = merit;
                best_pos = Some(pos);
                best_b = b;
                best_u = u;
            }
        }

        let Some(pos) = best_pos else { break; };
        let delta = best_u - actions[best_b];
        if delta.abs() > 1e-12 {
            actions[best_b] = best_u;
            for &(l, p) in &ca.b_to_lines[best_b] {
                if l < num_l {
                    flows[l] += p * delta;
                }
            }
        }
        pool.swap_remove(pos);
    }

    if actions.iter().any(|&u| u.abs() > 1e-8) {
        actions
    } else {
        build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep)
    }
}

fn run_screened_multistart_policy(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    refine_cap: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    let mut best_actions = vec![0.0; num_b];
    refine_basic_seed(challenge, state, ca, hp, flows_base, &mut best_actions);
    let mut best_score = approx_actions_value(challenge, state, ca, hp, flows_base, &best_actions);

    if num_b < 6 || refine_cap == 0 {
        return best_actions;
    }

    let mut max_util_base = 0.0_f64;
    for l in 0..challenge.network.flow_limits.len() {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        max_util_base = max_util_base.max((flows_base[l].abs() / limit).min(2.0));
    }
    if num_b > 110 && max_util_base < 0.20 {
        return best_actions;
    }

    let keep = if num_b <= 24 {
        num_b
    } else if num_b <= 48 {
        (5 * num_b + 5) / 6
    } else if num_b <= 100 {
        (3 * num_b + 3) / 4
    } else {
        (num_b + 1) / 2
    }.min(num_b);

    let mut seed_pool: Vec<(Vec<f64>, f64)> = Vec::new();

    let scaled_seed = build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    if scaled_seed.iter().any(|&u| u.abs() > 1e-8) {
        let score = approx_actions_value(challenge, state, ca, hp, flows_base, &scaled_seed);
        seed_pool.push((scaled_seed, score));
    }

    if max_util_base >= 0.30 || num_b <= 48 {
        let relief_keep = if num_b <= 100 {
            keep
        } else {
            ((2 * num_b + 2) / 5).min(keep).max(1)
        };
        let relief_seed = build_line_relief_seed(challenge, state, ca, hp, flows_base, relief_keep);
        if relief_seed.iter().any(|&u| u.abs() > 1e-8) {
            let score = approx_actions_value(challenge, state, ca, hp, flows_base, &relief_seed);
            seed_pool.push((relief_seed, score));
        }
    }

    seed_pool.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, (mut seed, seed_score)) in seed_pool.into_iter().enumerate() {
        if idx >= refine_cap || !seed_score.is_finite() {
            break;
        }
        refine_basic_seed(challenge, state, ca, hp, flows_base, &mut seed);
        let score = approx_actions_value(challenge, state, ca, hp, flows_base, &seed);
        if score > best_score + 1e-7 {
            best_score = score;
            best_actions = seed;
        }
    }

    best_actions
}

fn push_unique_probe(points: &mut Vec<f64>, u: f64) {
    if !u.is_finite() { return; }
    if points.iter().all(|&x| (x - u).abs() > 1e-7) {
        points.push(u);
    }
}

fn build_pair_probe_points(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
) -> Vec<f64> {
    let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);
    let cur = actions[b];
    let mut points = Vec::with_capacity(6);

    push_unique_probe(&mut points, u_min);
    push_unique_probe(&mut points, u_max);
    if u_min <= 0.0 && 0.0 <= u_max {
        push_unique_probe(&mut points, 0.0);
    }
    if u_min < cur - 1e-7 {
        push_unique_probe(&mut points, 0.5 * (u_min + cur));
    }
    if cur + 1e-7 < u_max {
        push_unique_probe(&mut points, 0.5 * (u_max + cur));
    }

    let mut lo_bias = 0.0_f64;
    let mut hi_bias = 0.0_f64;
    for &(l, p) in &ca.b_to_lines[b] {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        let util = (flows[l].abs() / limit).min(2.0);
        if util < 0.75 { continue; }
        if flows[l] * p > 0.0 {
            lo_bias += util;
        } else if flows[l] * p < 0.0 {
            hi_bias += util;
        }
    }
    if lo_bias > hi_bias + 0.05 && u_min < cur - 1e-7 {
        push_unique_probe(&mut points, (2.0 * cur + u_min) / 3.0);
    }
    if hi_bias > lo_bias + 0.05 && cur + 1e-7 < u_max {
        push_unique_probe(&mut points, (2.0 * cur + u_max) / 3.0);
    }

    points
}

fn pair_has_line_overlap(ca: &IycbtjtCache, a: usize, b: usize) -> bool {
    let lhs = &ca.b_to_lines[a];
    let rhs = &ca.b_to_lines[b];
    let mut i = 0usize;
    let mut j = 0usize;
    while i < lhs.len() && j < rhs.len() {
        let li = lhs[i].0;
        let lj = rhs[j].0;
        if li == lj {
            return true;
        }
        if li < lj {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

fn select_pair_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
) -> Vec<usize> {
    let num_b = challenge.num_batteries;
    if num_b <= 16 {
        return (0..num_b).collect();
    }

    let num_l = challenge.network.flow_limits.len();
    let mut interesting_line = vec![false; num_l];
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= 0.50 {
            interesting_line[l] = true;
        }
    }

    let mut chosen = vec![false; num_b];
    let mut candidates: Vec<usize> = Vec::new();
    for b in 0..num_b {
        let mut keep = actions[b].abs() > 1e-7;
        if !keep {
            for &(l, _) in &ca.b_to_lines[b] {
                if interesting_line[l] {
                    keep = true;
                    break;
                }
            }
        }
        if keep {
            chosen[b] = true;
            candidates.push(b);
        }
    }

    if candidates.len() < 10 {
        let mut ranked: Vec<(usize, f64)> = (0..num_b)
            .filter(|&b| !chosen[b])
            .map(|b| {
                let score = potential(challenge, state, ca, b)
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (b, _) in ranked {
            candidates.push(b);
            chosen[b] = true;
            if candidates.len() >= 10 { break; }
        }
    }

    if candidates.len() > 16 {
        let mut ranked: Vec<(usize, f64)> = candidates.iter().map(|&b| {
            let mut score = actions[b].abs() + 0.2 * potential(challenge, state, ca, b);
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit > 1e-6 {
                    let util = (flows[l].abs() / limit).min(2.0);
                    score += p.abs() * (0.15 + util * util);
                }
            }
            (b, score)
        }).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(16).map(|(b, _)| b).collect();
    }

    candidates
}

fn best_asymmetric_pair_move(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    anchor: usize,
    responder: usize,
    temp_actions: &mut Vec<f64>,
) -> Option<(f64, f64, f64)> {
    let cur_anchor = actions[anchor];
    let cur_responder = actions[responder];
    let base_val = eval_profit(challenge, state, ca, anchor, cur_anchor)
        + eval_profit(challenge, state, ca, responder, cur_responder);

    let mut best_delta = 0.0_f64;
    let mut best_move: Option<(f64, f64, f64)> = None;
    for u_anchor in build_pair_probe_points(challenge, state, ca, hp, flows, actions, anchor) {
        let delta_anchor = u_anchor - cur_anchor;
        if delta_anchor.abs() <= 1e-7 { continue; }

        let mut temp_flows = flows.to_vec();
        for &(l, p) in &ca.b_to_lines[anchor] {
            temp_flows[l] += p * delta_anchor;
        }

        temp_actions.as_mut_slice().copy_from_slice(actions);
        temp_actions[anchor] = u_anchor;

        let (u_responder, v_responder) =
            best_action_in_window(challenge, state, ca, hp, &temp_flows, temp_actions.as_slice(), responder);
        let delta = eval_profit(challenge, state, ca, anchor, u_anchor) + v_responder - base_val;
        if delta > best_delta + 1e-9 {
            best_delta = delta;
            best_move = Some((u_anchor, u_responder, delta));
        }
    }

    best_move
}

fn run_pair_exchange_polish(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) -> bool {
    let num_l = challenge.network.flow_limits.len();
    if challenge.num_batteries < 2 || num_l == 0 {
        return false;
    }

    let mut flows = compute_action_flows(ca, flows_base, actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return false;
    }

    let mut max_util = 0.0_f64;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        max_util = max_util.max((flows[l].abs() / limit).min(2.0));
    }
    if challenge.num_batteries > 16 && max_util < 0.55 {
        return false;
    }

    let candidates = select_pair_candidates(challenge, state, ca, hp, &flows, actions);
    if candidates.len() < 2 {
        return false;
    }

    let rounds = if challenge.num_batteries <= 12 { 2 } else { 1 };
    let mut temp_actions = actions.to_vec();
    let mut improved_any = false;

    for _ in 0..rounds {
        flows = compute_action_flows(ca, flows_base, actions);

        let mut best_delta = 0.0_f64;
        let mut best_move: Option<(usize, usize, f64, f64)> = None;

        for i_pos in 0..candidates.len() {
            let a = candidates[i_pos];
            for j_pos in (i_pos + 1)..candidates.len() {
                let b = candidates[j_pos];
                if !pair_has_line_overlap(ca, a, b) {
                    continue;
                }

                if let Some((u_a, u_b, delta)) =
                    best_asymmetric_pair_move(challenge, state, ca, hp, &flows, actions, a, b, &mut temp_actions)
                {
                    if delta > best_delta + 1e-9 {
                        best_delta = delta;
                        best_move = Some((a, b, u_a, u_b));
                    }
                }

                if let Some((u_b, u_a, delta)) =
                    best_asymmetric_pair_move(challenge, state, ca, hp, &flows, actions, b, a, &mut temp_actions)
                {
                    if delta > best_delta + 1e-9 {
                        best_delta = delta;
                        best_move = Some((b, a, u_b, u_a));
                    }
                }
            }
        }

        let Some((anchor, responder, u_anchor, u_responder)) = best_move else { break; };

        let delta_anchor = u_anchor - actions[anchor];
        if delta_anchor.abs() > 1e-12 {
            actions[anchor] = u_anchor;
            for &(l, p) in &ca.b_to_lines[anchor] {
                flows[l] += p * delta_anchor;
            }
        }

        let delta_responder = u_responder - actions[responder];
        if delta_responder.abs() > 1e-12 {
            actions[responder] = u_responder;
            for &(l, p) in &ca.b_to_lines[responder] {
                flows[l] += p * delta_responder;
            }
        }

        improved_any = true;
    }

    improved_any
}

fn refine_basic_seed(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    run_asca(challenge, state, ca, hp, flows_base, actions);
    run_deflator(challenge, state, ca, hp, flows_base, actions);
}

fn select_small_reset_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
) -> Vec<usize> {
    let num_b = challenge.num_batteries;
    if num_b == 0 {
        return Vec::new();
    }
    if num_b <= 12 {
        return (0..num_b).collect();
    }

    let num_l = challenge.network.flow_limits.len();
    let mut line_rank: Vec<(usize, f64)> = Vec::new();
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util > 0.20 {
            line_rank.push((l, util));
        }
    }
    line_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep_lines = if num_b <= 20 { 2 } else { 3 };
    let top_lines: Vec<usize> = line_rank.iter().take(keep_lines).map(|(l, _)| *l).collect();

    let mut mask = vec![false; num_b];
    for b in 0..num_b {
        if actions[b].abs() > 1e-7 {
            mask[b] = true;
        }
    }
    for &l in &top_lines {
        for &(b, _) in &ca.ptdf_sparse[l] {
            mask[b] = true;
        }
    }

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| mask[b]).collect();
    let target = 8.min(num_b);
    if candidates.len() < target {
        let mut ranked: Vec<(usize, f64)> = (0..num_b)
            .filter(|&b| !mask[b])
            .map(|b| {
                let mut touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    let focus = if top_lines.iter().any(|&ll| ll == l) { 1.0 } else { 0.0 };
                    touch += p.abs() * (util + focus);
                }
                let score = potential(challenge, state, ca, b) * (1.0 + 0.08 * touch)
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (b, _) in ranked.into_iter().take(target.saturating_sub(candidates.len())) {
            candidates.push(b);
            mask[b] = true;
        }
    }

    let cap = 12.min(num_b);
    if candidates.len() > cap {
        let mut ranked: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&b| {
                let mut touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    let focus = if top_lines.iter().any(|&ll| ll == l) { 1.0 } else { 0.0 };
                    touch += p.abs() * (0.2 * util + focus);
                }
                let score = actions[b].abs()
                    + 0.25 * potential(challenge, state, ca, b)
                    + 0.2 * touch;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(cap).map(|(b, _)| b).collect();
    }

    candidates
}

fn run_small_destroy_repair_polish(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) -> bool {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b < 3 || num_l == 0 {
        return false;
    }

    let flows = compute_action_flows(ca, flows_base, actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible || !actions.iter().any(|&u| u.abs() > 1e-7) {
        return false;
    }

    let mut max_util = 0.0_f64;
    let mut line_rank: Vec<(usize, f64)> = Vec::new();
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util > max_util {
            max_util = util;
        }
        if util > 0.20 {
            line_rank.push((l, util));
        }
    }
    if num_b > 12 && max_util < 0.35 {
        return false;
    }
    line_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let candidates = select_small_reset_candidates(challenge, state, ca, hp, &flows, actions);
    if candidates.len() < 3 {
        return false;
    }

    let mut line_weight = vec![0.0_f64; num_l];
    let keep_lines = if num_b <= 12 { 2 } else { 3 };
    for (l, util) in line_rank.into_iter().take(keep_lines) {
        line_weight[l] = util;
    }

    let mut ranked_active: Vec<(usize, f64)> = candidates
        .iter()
        .filter_map(|&b| {
            if actions[b].abs() <= 1e-7 {
                return None;
            }

            let keep_val = eval_profit(challenge, state, ca, b, actions[b]);
            let zero_val = eval_profit(challenge, state, ca, b, 0.0);
            let retained_value = (keep_val - zero_val).max(0.0);

            let mut stress = 0.0_f64;
            let mut relief = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let w = line_weight[l];
                if w <= 0.0 {
                    continue;
                }
                let signed = p * actions[b];
                if flows[l] * signed > 1e-9 {
                    stress += w * signed.abs();
                } else if flows[l] * signed < -1e-9 {
                    relief += w * signed.abs();
                }
            }

            let score = if stress > relief + 1e-9 {
                (stress - 0.5 * relief) / (retained_value + 1e-6)
            } else if max_util > 0.75 && stress > 1e-9 {
                0.25 * stress / (retained_value + 1e-6)
            } else {
                0.0
            };

            if score > 1e-8 {
                Some((b, score))
            } else {
                None
            }
        })
        .collect();
    ranked_active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked_active.is_empty() {
        return false;
    }

    let base_score = approx_actions_value(challenge, state, ca, hp, flows_base, actions);
    let mut best_score = base_score;
    let mut best_actions = actions.to_vec();

    let mut drop_counts = vec![1usize];
    if ranked_active.len() >= 2 {
        drop_counts.push(2usize);
    }
    if ranked_active.len() >= 3 && (num_b <= 12 || max_util > 0.75) {
        drop_counts.push(3usize);
    }
    if ranked_active.len() >= 4 && (num_b <= 10 || max_util > 0.90) {
        drop_counts.push(4usize);
    }

    let sweeps = if num_b <= 12 {
        hp.asca_iters.min(3).max(1)
    } else {
        hp.asca_iters.min(2).max(1)
    };

    for drop_count in drop_counts {
        let mut alt_actions = actions.to_vec();
        let mut changed = false;
        for &(b, _) in ranked_active.iter().take(drop_count) {
            if alt_actions[b].abs() > 1e-9 {
                alt_actions[b] = 0.0;
                changed = true;
            }
        }
        if !changed {
            continue;
        }

        run_asca_candidates(challenge, state, ca, hp, flows_base, &mut alt_actions, &candidates, sweeps);
        run_deflator(challenge, state, ca, hp, flows_base, &mut alt_actions);
        if challenge.num_batteries <= 15 && run_pair_exchange_polish(challenge, state, ca, hp, flows_base, &mut alt_actions) {
            run_deflator(challenge, state, ca, hp, flows_base, &mut alt_actions);
        }

        let alt_score = approx_actions_value(challenge, state, ca, hp, flows_base, &alt_actions);
        if alt_score > best_score + 1e-7 {
            best_score = alt_score;
            best_actions = alt_actions;
        }
    }

    if best_score > base_score + 1e-7 {
        actions.copy_from_slice(&best_actions);
        true
    } else {
        false
    }
}

fn refine_small_seed(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    refine_basic_seed(challenge, state, ca, hp, flows_base, actions);

    if run_pair_exchange_polish(challenge, state, ca, hp, flows_base, actions) {
        let all: Vec<usize> = (0..challenge.num_batteries).collect();
        let sweeps = if challenge.num_batteries <= 12 {
            hp.asca_iters.min(2).max(1)
        } else {
            1
        };
        run_asca_candidates(challenge, state, ca, hp, flows_base, actions, &all, sweeps);
        run_deflator(challenge, state, ca, hp, flows_base, actions);
    }

    run_small_destroy_repair_polish(challenge, state, ca, hp, flows_base, actions);
}

fn run_medium_multistart_policy(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
) -> Vec<f64> {
    run_screened_multistart_policy(challenge, state, ca, hp, flows_base, 2)
}

fn run_small_multistart_policy(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
) -> Vec<f64> {
    let num_b = challenge.num_batteries;

    let mut best_actions = vec![0.0; num_b];
    refine_small_seed(challenge, state, ca, hp, flows_base, &mut best_actions);
    let mut best_score = approx_actions_value(challenge, state, ca, hp, flows_base, &best_actions);

    if num_b < 4 {
        return best_actions;
    }

    let sparse_keep = if num_b <= 10 {
        0
    } else if num_b <= 20 {
        (3 * num_b + 3) / 4
    } else {
        (num_b + 1) / 2
    };

    let mut tried: Vec<usize> = Vec::new();
    for keep in [num_b, sparse_keep] {
        let keep = keep.min(num_b);
        if keep == 0 || tried.iter().any(|&k| k == keep) { continue; }
        tried.push(keep);

        let mut alt_actions = build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
        if !alt_actions.iter().any(|&u| u.abs() > 1e-8) { continue; }

        refine_small_seed(challenge, state, ca, hp, flows_base, &mut alt_actions);

        let alt_score = approx_actions_value(challenge, state, ca, hp, flows_base, &alt_actions);
        if alt_score > best_score + 1e-7 {
            best_score = alt_score;
            best_actions = alt_actions;
        }
    }

    best_actions
}

fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> IycbtjtCache {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    let num_t = challenge.num_steps;
    let num_n = challenge.network.num_nodes;

    let zero_action = vec![0.0_f64; num_b];
    let inj_base = challenge.compute_total_injections(state, &zero_action);
    let flows0 = challenge.network.compute_flows(&inj_base);

    let mut batt_nodes = vec![0usize; num_b];
    let mut ptdf_sparse: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_l];
    let mut b_to_lines: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_b];
    let mut dummy = zero_action.clone();
    for b in 0..num_b {
        dummy[b] = 1.0;
        let inj1 = challenge.compute_total_injections(state, &dummy);
        let flows1 = challenge.network.compute_flows(&inj1);
        for k in 0..num_n {
            if (inj1[k] - inj_base[k]).abs() > 0.5 && k != challenge.network.slack_bus {
                batt_nodes[b] = k;
                break;
            }
        }
        for l in 0..num_l {
            let impact = flows1[l] - flows0[l];
            if impact.abs() > 1e-8 {
                ptdf_sparse[l].push((b, impact));
                b_to_lines[b].push((l, impact));
            }
        }
        dummy[b] = 0.0;
    }

    let mut expected_premiums = vec![vec![0.0_f64; num_b]; num_t];
    if hp.anticipate_lmp && num_l > 0 {
        let base_premium = 20.0 * hp.lmp_premium_scale;
        for t in 0..num_t {
            let f_exo = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
            for l in 0..num_l {
                let limit = challenge.network.flow_limits[l];
                if limit <= 1e-6 { continue; }
                let ratio = f_exo[l].abs() / limit;
                if ratio > hp.lmp_threshold {
                    let proba = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6))
                        .clamp(0.0, 1.0);
                    let premium = base_premium * proba;
                    let sign_f = f_exo[l].signum();
                    for &(b, impact) in &ptdf_sparse[l] {
                        if impact.abs() > 1e-6 {
                            let nodal_shift = -impact * sign_f * premium;
                            expected_premiums[t][b] += nodal_shift;
                        }
                    }
                }
            }
        }
    }

    let soc_levels = hp.soc_levels;
    let action_grid = hp.action_grid;
    let dt = 0.25_f64;
    let mut dp = vec![vec![vec![0.0_f64; soc_levels]; num_t + 1]; num_b];

    for b in 0..num_b {
        let bat = &challenge.batteries[b];
        let node = batt_nodes[b];
        let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);

        for t in (0..num_t).rev() {
            let p_da = if node < challenge.market.day_ahead_prices[t].len() {
                challenge.market.day_ahead_prices[t][node]
            } else {
                challenge.market.day_ahead_prices[t][0]
            };
            let extra = expected_premiums[t][b];
            let p_sell = p_da * (1.0 + hp.jump_premium) + extra;
            let p_buy = p_da + extra;

            for i in 0..soc_levels {
                let soc = bat.soc_min_mwh + soc_span * (i as f64) / ((soc_levels - 1) as f64);

                let charge_soc_limit = if bat.efficiency_charge > 0.0 {
                    (bat.soc_max_mwh - soc) / (bat.efficiency_charge * dt)
                } else { 0.0 };
                let discharge_soc_limit = if bat.efficiency_discharge > 0.0 {
                    (soc - bat.soc_min_mwh) * bat.efficiency_discharge / dt
                } else { 0.0 };

                let max_pwr_c = bat.power_charge_mw * hp.network_derating;
                let max_pwr_d = bat.power_discharge_mw * hp.network_derating;

                let u_min = -(max_pwr_c.min(charge_soc_limit.max(0.0)));
                let u_max = max_pwr_d.min(discharge_soc_limit.max(0.0));
                let u_max = u_max.max(u_min);

                let mut max_val = f64::NEG_INFINITY;
                let span = u_max - u_min;
                for j in 0..=action_grid {
                    let u = if span > 0.0 {
                        u_min + span * (j as f64) / (action_grid as f64)
                    } else { u_min };
                    let price = if u > 0.0 { p_sell } else { p_buy };
                    let abs_u = u.abs();
                    let revenue = u * price * dt;
                    let tx = 0.25 * abs_u * dt;
                    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
                    let deg = deg_base * deg_base;
                    let profit = revenue - tx - deg;

                    let next_soc_raw = if u < 0.0 {
                        soc + bat.efficiency_charge * (-u) * dt
                    } else {
                        soc - u / bat.efficiency_discharge.max(1e-9) * dt
                    };
                    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

                    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
                    let idx0 = (idx_f.floor() as isize).max(0) as usize;
                    let idx0c = idx0.min(soc_levels - 1);
                    let idx1c = (idx0 + 1).min(soc_levels - 1);
                    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
                    let v_next = dp[b][t + 1][idx0c] * (1.0 - frac)
                        + dp[b][t + 1][idx1c] * frac;

                    let val = profit + v_next;
                    if val > max_val { max_val = val; }
                }
                dp[b][t][i] = max_val;
            }
        }
    }

    IycbtjtCache { dp, ptdf_sparse, b_to_lines, batt_nodes }
}

#[inline]
fn eval_profit(challenge: &Challenge, state: &State, ca: &IycbtjtCache, b: usize, u: f64) -> f64 {
    let bat = &challenge.batteries[b];
    let node = ca.batt_nodes[b];
    let rt_price = if node < state.rt_prices.len() { state.rt_prices[node] } else { 0.0 };
    let dt = 0.25_f64;
    let abs_u = u.abs();
    let revenue = u * rt_price * dt;
    let tx = 0.25 * abs_u * dt;
    let deg_base = (abs_u * dt) / bat.capacity_mwh.max(1e-9);
    let deg = deg_base * deg_base;
    let profit = revenue - tx - deg;

    let soc = state.socs[b];
    let next_soc_raw = if u < 0.0 {
        soc + bat.efficiency_charge * (-u) * dt
    } else {
        soc - u / bat.efficiency_discharge.max(1e-9) * dt
    };
    let next_soc = next_soc_raw.clamp(bat.soc_min_mwh, bat.soc_max_mwh);

    let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
    let soc_levels = ca.dp[b][0].len();
    let idx_f = (next_soc - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64);
    let idx0 = (idx_f.floor() as isize).max(0) as usize;
    let idx0c = idx0.min(soc_levels - 1);
    let idx1c = (idx0 + 1).min(soc_levels - 1);
    let frac = (idx_f - idx0 as f64).clamp(0.0, 1.0);
    let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
    profit + ca.dp[b][t_next][idx0c] * (1.0 - frac) + ca.dp[b][t_next][idx1c] * frac
}

#[inline]
fn potential(challenge: &Challenge, state: &State, ca: &IycbtjtCache, b: usize) -> f64 {
    let (u_lo, u_hi) = state.action_bounds[b];
    let v_lo = eval_profit(challenge, state, ca, b, u_lo);
    let v_hi = eval_profit(challenge, state, ca, b, u_hi);
    let v0 = eval_profit(challenge, state, ca, b, 0.0);
    (v_lo.max(v_hi) - v0).max(0.0)
}

fn ternary_search<F: Fn(f64) -> f64>(f: F, mut l: f64, mut r: f64, iters: usize) -> (f64, f64) {
    if l >= r { return (l, f(l)); }
    for _ in 0..iters {
        let m1 = l + (r - l) / 3.0;
        let m2 = r - (r - l) / 3.0;
        if f(m1) < f(m2) { l = m1; } else { r = m2; }
    }
    let u = 0.5 * (l + r);
    (u, f(u))
}

fn compute_action_flows(ca: &IycbtjtCache, flows_base: &[f64], actions: &[f64]) -> Vec<f64> {
    let mut flows = flows_base.to_vec();
    for (l, row) in ca.ptdf_sparse.iter().enumerate() {
        let mut delta = 0.0_f64;
        for &(b, imp) in row {
            delta += imp * actions[b];
        }
        if l < flows.len() { flows[l] += delta; }
    }
    flows
}

fn feasible_window(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
) -> (f64, f64) {
    let (mut u_min, mut u_max) = state.action_bounds[b];

    for &(l, p) in &ca.b_to_lines[b] {
        if p.abs() < 1e-9 { continue; }
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        let f_other = flows[l] - p * actions[b];
        let b1 = (-limit - f_other) / p;
        let b2 = (limit - f_other) / p;
        let (lo, hi) = if b1 < b2 { (b1, b2) } else { (b2, b1) };
        if lo > u_min { u_min = lo; }
        if hi < u_max { u_max = hi; }
    }

    if u_min > u_max {
        u_min = actions[b];
        u_max = actions[b];
    }
    u_min = u_min.min(actions[b]);
    u_max = u_max.max(actions[b]);
    (u_min, u_max)
}

fn best_action_in_window(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
) -> (f64, f64) {
    let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);

    let mut best_u = actions[b];
    let mut best_v = eval_profit(challenge, state, ca, b, best_u);

    let v_min = eval_profit(challenge, state, ca, b, u_min);
    if v_min > best_v {
        best_v = v_min;
        best_u = u_min;
    }

    if (u_max - u_min).abs() > 1e-12 {
        let v_max = eval_profit(challenge, state, ca, b, u_max);
        if v_max > best_v {
            best_v = v_max;
            best_u = u_max;
        }
    }

    if u_min <= 0.0 && 0.0 <= u_max {
        let v0 = eval_profit(challenge, state, ca, b, 0.0);
        if v0 > best_v {
            best_v = v0;
            best_u = 0.0;
        }
    }

    if u_min < 0.0 {
        let lo = u_min;
        let hi = 0.0_f64.min(u_max);
        if lo < hi {
            let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }
    }

    if u_max > 0.0 {
        let lo = 0.0_f64.max(u_min);
        let hi = u_max;
        if lo < hi {
            let (u, v) = ternary_search(|u| eval_profit(challenge, state, ca, b, u), lo, hi, hp.ternary_iters);
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }
    }

    (best_u, best_v)
}

fn dynamic_order(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    candidates: &[usize],
) -> Vec<usize> {
    let mut ranked: Vec<(usize, f64)> = candidates.iter().map(|&b| {
        let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);
        let v_cur = eval_profit(challenge, state, ca, b, actions[b]);

        let mut upside = 0.0_f64;
        let gain_min = eval_profit(challenge, state, ca, b, u_min) - v_cur;
        if gain_min > upside { upside = gain_min; }

        if (u_max - u_min).abs() > 1e-12 {
            let gain_max = eval_profit(challenge, state, ca, b, u_max) - v_cur;
            if gain_max > upside { upside = gain_max; }
        }

        if u_min <= 0.0 && 0.0 <= u_max {
            let gain_zero = eval_profit(challenge, state, ca, b, 0.0) - v_cur;
            if gain_zero > upside { upside = gain_zero; }
        }

        upside = upside.max(0.25 * potential(challenge, state, ca, b));

        let mut footprint = 1e-4_f64;
        let mut stressed_touch = 0.0_f64;
        for &(l, p) in &ca.b_to_lines[b] {
            let limit = challenge.network.flow_limits[l];
            if limit <= 1e-6 { continue; }
            let util = (flows[l].abs() / limit).min(2.0);
            footprint += p.abs() * (0.5 + 8.0 * util * util);
            if util > 0.85 {
                stressed_touch += p.abs() * (util - 0.85);
            }
        }

        let score = (upside + 1e-9) * (1.0 + 0.1 * stressed_touch) / footprint;
        (b, score)
    }).collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().map(|(b, _)| b).collect()
}

fn run_asca_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
    candidates: &[usize],
    max_sweeps: usize,
) {
    if candidates.is_empty() || max_sweeps == 0 { return; }

    let num_l = challenge.network.flow_limits.len();
    let mut flows = compute_action_flows(ca, flows_base, actions);
    let refresh_every_sweep = candidates.len() <= 40 || challenge.num_batteries <= 40;
    let mut order = dynamic_order(challenge, state, ca, hp, &flows, actions, candidates);
    let mut prev_max_change = f64::INFINITY;

    for sweep in 0..max_sweeps {
        if sweep > 0 && (refresh_every_sweep || sweep == 1 || prev_max_change > hp.convergence_tol * 5.0) {
            order = dynamic_order(challenge, state, ca, hp, &flows, actions, candidates);
        }

        let mut max_change = 0.0_f64;
        for &b in &order {
            let (best_u, _) = best_action_in_window(challenge, state, ca, hp, &flows, actions, b);
            let delta = best_u - actions[b];
            if delta.abs() > 1e-6 {
                actions[b] = best_u;
                for &(l, p) in &ca.b_to_lines[b] {
                    if l < num_l { flows[l] += p * delta; }
                }
                if delta.abs() > max_change { max_change = delta.abs(); }
            }
        }

        if max_change < hp.convergence_tol { break; }
        prev_max_change = max_change;
    }
}

fn run_asca(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_b = challenge.num_batteries;
    let mut candidates: Vec<usize> = (0..num_b).collect();

    if hp.prune_ratio > 0.0 && num_b >= 2 {
        let cutoff = ((num_b as f64) * hp.prune_ratio) as usize;
        if cutoff > 0 && cutoff < num_b {
            let mut ranked: Vec<(usize, f64)> = (0..num_b).map(|b| {
                let mut network_score = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = challenge.network.flow_limits[l];
                    if limit > 1e-6 {
                        let util = (flows_base[l].abs() / limit).min(2.0);
                        network_score += p.abs() * (1.0 + 4.0 * util * util);
                    }
                }
                let score = potential(challenge, state, ca, b) * (1.0 + network_score.sqrt())
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            }).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates = ranked.into_iter().take(num_b - cutoff).map(|(b, _)| b).collect();
        }
    }

    run_asca_candidates(challenge, state, ca, hp, flows_base, actions, &candidates, hp.asca_iters);
}

fn compute_line_relief_signal(
    challenge: &Challenge,
    hp: &TrackHp,
    flows_before: &[f64],
    flows_after: &[f64],
) -> Vec<f64> {
    let num_l = challenge.network.flow_limits.len();
    let mut ranked: Vec<(usize, f64)> = Vec::new();

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let before = (flows_before[l].abs() / limit).min(3.0);
        let after = (flows_after[l].abs() / limit).min(3.0);
        let improvement = (before - after).max(0.0);
        if improvement <= 1e-6 {
            continue;
        }

        let mut score = 0.0_f64;
        if before > 1.0 {
            score = improvement + 0.75 * (before - 1.0);
        } else if before > 0.90 {
            score = 0.5 * improvement;
        } else if improvement > 0.10 {
            score = 0.25 * improvement;
        }

        if score > 1e-6 {
            ranked.push((l, score));
        }
    }

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep = if num_l <= 8 { ranked.len() } else { ranked.len().min(4) };

    let mut signal = vec![0.0_f64; num_l];
    for (l, score) in ranked.into_iter().take(keep) {
        signal[l] = score;
    }
    signal
}

fn select_post_deflator_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    seed_mask: &[bool],
    relief_signal: &[f64],
) -> Vec<usize> {
    let num_b = challenge.num_batteries;
    if num_b == 0 {
        return Vec::new();
    }
    if num_b <= 30 {
        return (0..num_b).collect();
    }

    let num_l = challenge.network.flow_limits.len();
    let stress_cut = if num_b <= 50 { 0.70 } else if num_b <= 80 { 0.78 } else { 0.86 };
    let reserve_target = if num_b <= 50 { 10 } else if num_b <= 80 { 14 } else { 18 };
    let shortlist_cap = if num_b <= 50 { 20 } else if num_b <= 80 { 24 } else { 32 };
    let mut mask = seed_mask.to_vec();

    for b in 0..num_b {
        if actions[b].abs() > 1e-7 {
            mask[b] = true;
        }
    }

    for l in 0..num_l {
        if relief_signal[l] > 1e-9 {
            for &(b, _) in &ca.ptdf_sparse[l] {
                mask[b] = true;
            }
        }
    }

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= stress_cut {
            for &(b, _) in &ca.ptdf_sparse[l] {
                mask[b] = true;
            }
        }
    }

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| mask[b]).collect();

    if candidates.len() < reserve_target.min(num_b) {
        let mut coarse_ranked: Vec<(usize, f64)> = (0..num_b)
            .filter(|&b| !mask[b])
            .map(|b| {
                let mut footprint = 1e-6_f64;
                let mut stressed_touch = 0.0_f64;
                let mut relief_touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit > 1e-6 {
                        let util = (flows[l].abs() / limit).min(2.0);
                        footprint += p.abs() * (0.25 + util * util);
                        if util > 0.60 {
                            stressed_touch += p.abs() * util;
                        }
                    }
                    if relief_signal[l] > 0.0 {
                        relief_touch += p.abs() * relief_signal[l];
                    }
                }
                let score = potential(challenge, state, ca, b)
                    * (1.0 + 0.08 * stressed_touch + 0.25 * relief_touch) / footprint
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        coarse_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let shortlist: Vec<usize> = coarse_ranked
            .into_iter()
            .take(shortlist_cap)
            .map(|(b, _)| b)
            .collect();

        let mut refined: Vec<(usize, f64)> = Vec::new();
        for &b in &shortlist {
            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, flows, actions, b);
            let gain = v - v_cur;
            if gain <= 1e-9 || u.abs() <= 1e-9 {
                continue;
            }

            let mut footprint = 1e-6_f64;
            let mut stressed_touch = 0.0_f64;
            let mut relief_bonus = 0.0_f64;
            let mut relief_penalty = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit > 1e-6 {
                    let util = (flows[l].abs() / limit).min(2.0);
                    footprint += p.abs() * (0.25 + util * util);
                    if util > 0.60 {
                        stressed_touch += p.abs() * util;
                    }
                }

                let signal = relief_signal[l];
                if signal > 0.0 {
                    let signed_move = p * u;
                    if flows[l] * signed_move < 0.0 {
                        relief_bonus += signal * signed_move.abs();
                    } else if flows[l] * signed_move > 0.0 {
                        relief_penalty += signal * signed_move.abs();
                    }
                }
            }

            let score = gain * (1.0 + 0.10 * stressed_touch + 0.30 * relief_bonus)
                / (footprint * (1.0 + 0.40 * relief_penalty))
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            refined.push((b, score));
        }

        refined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let need = reserve_target.min(num_b).saturating_sub(candidates.len());
        if refined.is_empty() {
            for &b in shortlist.iter().take(need) {
                candidates.push(b);
                mask[b] = true;
            }
        } else {
            for (b, _) in refined.into_iter().take(need) {
                candidates.push(b);
                mask[b] = true;
            }
        }
    }

    let cap = if num_b <= 50 { 36 } else if num_b <= 80 { 48 } else { 64 };
    if candidates.len() > cap {
        let mut ranked: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&b| {
                let mut stressed_touch = 0.0_f64;
                let mut relief_touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit > 1e-6 {
                        let util = (flows[l].abs() / limit).min(2.0);
                        stressed_touch += p.abs() * util;
                    }
                    if relief_signal[l] > 0.0 {
                        relief_touch += p.abs() * relief_signal[l];
                    }
                }
                let mut score = actions[b].abs()
                    + 0.15 * potential(challenge, state, ca, b)
                    + 0.2 * stressed_touch
                    + 0.35 * relief_touch;
                if seed_mask[b] {
                    score += 1.0;
                }
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(cap).map(|(b, _)| b).collect();
    }

    candidates
}

fn run_post_deflator_polish(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
    seed_mask: &[bool],
    relief_signal: &[f64],
) {
    let num_b = challenge.num_batteries;
    if num_b == 0 {
        return;
    }

    let num_l = challenge.network.flow_limits.len();
    let flows = compute_action_flows(ca, flows_base, actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return;
    }

    let relief_mass: f64 = relief_signal.iter().sum();
    if !actions.iter().any(|&u| u.abs() > 1e-7) && relief_mass < 0.05 {
        return;
    }

    let mut max_util = 0.0_f64;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        max_util = max_util.max((flows[l].abs() / limit).min(2.0));
    }

    let seed_count = seed_mask.iter().filter(|&&v| v).count();
    if seed_count == 0 && relief_mass < 0.05 && max_util < 0.70 {
        return;
    }
    if num_b > 80 && seed_count <= 2 && relief_mass < 0.10 && max_util < 0.60 {
        return;
    }

    let candidates = select_post_deflator_candidates(
        challenge,
        state,
        ca,
        hp,
        &flows,
        actions,
        seed_mask,
        relief_signal,
    );
    if candidates.is_empty() {
        return;
    }

    let sweeps = if num_b <= 18 || (relief_mass > 0.35 && candidates.len() <= 32) { 2 } else { 1 };
    run_asca_candidates(challenge, state, ca, hp, flows_base, actions, &candidates, sweeps);
}

fn run_deflator(
    challenge: &Challenge,
    state: &State,
    ca: &IycbtjtCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_l = challenge.network.flow_limits.len();
    let num_b = challenge.num_batteries;
    let mut flows = compute_action_flows(ca, flows_base, actions);
    let flows_before_repair = flows.clone();
    let mut changed = vec![false; num_b];

    let mut is_safe = true;
    for _ in 0..hp.deflator_iters {
        is_safe = true;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if flows[l].abs() <= limit { continue; }
            is_safe = false;
            let overflow = flows[l].abs() - limit;
            let sign = flows[l].signum();

            let mut culprits: Vec<(usize, f64, f64)> = Vec::new();
            for &(b, impact) in &ca.ptdf_sparse[l] {
                let contrib = impact * actions[b];
                if contrib * sign > 1e-9 {
                    let val_curr = eval_profit(challenge, state, ca, b, actions[b]);
                    let val_zero = eval_profit(challenge, state, ca, b, 0.0);
                    let roi = ((val_curr - val_zero).max(0.0)) / contrib.abs().max(1e-6);
                    culprits.push((b, contrib, roi));
                }
            }
            culprits.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut remaining = overflow;
            for (b, contrib, _) in culprits {
                if remaining <= 1e-9 { break; }
                let contrib_abs = contrib.abs();
                if contrib_abs < 1e-12 { continue; }
                let reduction = contrib_abs.min(remaining);
                let ratio = 1.0 - (reduction / contrib_abs);
                let new_action = actions[b] * ratio;
                let delta = new_action - actions[b];
                if delta.abs() <= 1e-12 { continue; }
                actions[b] = new_action;
                changed[b] = true;
                for &(ll, pp) in &ca.b_to_lines[b] {
                    if ll < num_l { flows[ll] += pp * delta; }
                }
                remaining -= reduction;
            }
        }
        if is_safe { break; }
    }

    if !is_safe {
        let mut beta = 1.0_f64;
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            let total = flows[l];
            if total.abs() <= limit { continue; }
            let f_act = total - flows_base[l];
            if f_act.abs() < 1e-9 { continue; }
            let target = if total > 0.0 { limit } else { -limit };
            let candidate = (target - flows_base[l]) / f_act;
            if candidate < beta { beta = candidate; }
        }
        let beta = beta.clamp(0.0, 1.0);
        if beta < 1.0 {
            for b in 0..num_b {
                let new_action = actions[b] * beta;
                if (new_action - actions[b]).abs() > 1e-12 {
                    actions[b] = new_action;
                    changed[b] = true;
                }
            }
        }
    }

    for b in 0..num_b {
        let (lo, hi) = state.action_bounds[b];
        let clipped = actions[b].clamp(lo, hi);
        if (clipped - actions[b]).abs() > 1e-12 {
            actions[b] = clipped;
            changed[b] = true;
        }
    }

    flows = compute_action_flows(ca, flows_base, actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible { return; }

    let mut relief_signal = compute_line_relief_signal(challenge, hp, &flows_before_repair, &flows);

    let mut candidate_mask = changed;
    let stress_cut = if num_b <= 30 { 0.80 } else if num_b <= 80 { 0.88 } else { 0.93 };
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        if flows[l].abs() >= stress_cut * limit || relief_signal[l] > 1e-9 {
            for &(b, _) in &ca.ptdf_sparse[l] {
                candidate_mask[b] = true;
            }
        }
    }

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| candidate_mask[b]).collect();
    if candidates.is_empty() { return; }

    if num_b > 60 && candidates.len() > 48 {
        let mut ranked: Vec<(usize, f64)> = candidates.iter().map(|&b| {
            let mut score = actions[b].abs() + 0.1 * potential(challenge, state, ca, b);
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = challenge.network.flow_limits[l];
                if limit > 1e-6 {
                    let util = (flows[l].abs() / limit).min(2.0);
                    score += p.abs() * util;
                }
                if relief_signal[l] > 0.0 {
                    score += 0.35 * p.abs() * relief_signal[l];
                }
            }
            (b, score)
        }).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = if num_b <= 100 { 48 } else { 64 }.min(ranked.len());
        candidates = ranked.into_iter().take(keep).map(|(b, _)| b).collect();
    }

    let local_sweeps = if candidates.len() <= 16 {
        hp.asca_iters.min(3).max(1)
    } else if candidates.len() <= 40 {
        hp.asca_iters.min(2).max(1)
    } else {
        1
    };
    run_asca_candidates(challenge, state, ca, hp, flows_base, actions, &candidates, local_sweeps);

    flows = compute_action_flows(ca, flows_base, actions);
    relief_signal = compute_line_relief_signal(challenge, hp, &flows_before_repair, &flows);
    run_post_deflator_polish(challenge, state, ca, hp, flows_base, actions, &candidate_mask, &relief_signal);
}