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

pub struct AiioagaypCache {
    pub dp: Vec<Vec<Vec<f64>>>,
    pub ptdf_sparse: Vec<Vec<(usize, f64)>>,
    pub b_to_lines: Vec<Vec<(usize, f64)>>,
    pub batt_nodes: Vec<usize>,
}

use self::AiioagaypCache as aiioagaypCache;

struct Inner {
    hp: TrackHp,
    cache: Option<aiioagaypCache>,
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
        let inner = guard.as_mut().expect("aiioagayp: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        let mut actions = run_baseline_hybrid_policy(challenge, state, cache, hp, &flows_base);
        run_restricted_active_set_rebuild(challenge, state, cache, hp, &flows_base, &mut actions);
        Ok(actions)
    })
}

fn run_baseline_hybrid_policy(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    let mut best_actions = run_small_multistart_policy(challenge, state, ca, hp, flows_base);
    if num_b == 0 {
        return best_actions;
    }

    let best_score = approx_actions_value(challenge, state, ca, hp, flows_base, &best_actions);
    if !best_score.is_finite() || num_b < 6 {
        return best_actions;
    }

    let num_l = challenge.network.flow_limits.len();
    let flows = compute_action_flows(ca, flows_base, &best_actions);
    let mut feasible = true;
    let mut max_util = 0.0_f64;
    let mut stressed_lines = 0usize;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        let flow_abs = flows[l].abs();
        if flow_abs > limit + 1e-7 {
            feasible = false;
            break;
        }
        if limit <= 1e-6 {
            continue;
        }
        let util = (flow_abs / limit).min(2.0);
        if util > max_util {
            max_util = util;
        }
        if util >= 0.55 {
            stressed_lines += 1;
        }
    }
    if !feasible {
        return best_actions;
    }

    let active_count = best_actions.iter().filter(|&&u| u.abs() > 1e-7).count();

    if num_b > 72 && num_l > 0 && max_util > 0.45 && active_count > 28 && stressed_lines >= 2 {
        return best_actions;
    }

    let use_all = num_l == 0
        || num_b <= 28
        || max_util < 0.18
        || (num_b <= 48 && max_util < 0.42)
        || active_count <= 10;

    let cap = if use_all {
        num_b
    } else if num_b <= 56 {
        24
    } else if num_b <= 96 {
        36
    } else {
        48
    }
    .min(num_b);

    let mut mask = vec![use_all; num_b];
    if !use_all {
        for b in 0..num_b {
            if best_actions[b].abs() > 1e-7 {
                mask[b] = true;
            }
        }

        let stress_cut = if max_util > 0.82 {
            0.65
        } else if max_util > 0.55 {
            0.50
        } else {
            0.35
        };

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
    }

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| mask[b]).collect();

    if candidates.len() < cap {
        let mut ranked: Vec<(usize, f64)> = (0..num_b)
            .filter(|&b| !mask[b])
            .map(|b| {
                let mut footprint = 1e-6_f64;
                let mut stress_touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    footprint += p.abs() * (0.20 + util * util);
                    if util > 0.35 {
                        stress_touch += p.abs() * util;
                    }
                }

                let score = potential(challenge, state, ca, b)
                    * (1.0 + 0.06 * stress_touch)
                    / footprint
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (b, _) in ranked.into_iter().take(cap.saturating_sub(candidates.len())) {
            candidates.push(b);
            mask[b] = true;
        }
    }

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
                    touch += p.abs() * (0.12 + util);
                }

                let score = best_actions[b].abs()
                    + 0.20 * potential(challenge, state, ca, b)
                    + 0.12 * touch;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(cap).map(|(b, _)| b).collect();
    }

    if candidates.is_empty() {
        return best_actions;
    }

    let mut polished = best_actions.clone();
    let sweeps = if candidates.len() <= 18 {
        hp.asca_iters.min(2).max(1)
    } else {
        1
    };
    run_asca_candidates(
        challenge,
        state,
        ca,
        hp,
        flows_base,
        &mut polished,
        &candidates,
        sweeps,
    );
    run_deflator(challenge, state, ca, hp, flows_base, &mut polished);

    if (use_all && num_b <= 28) || candidates.len() <= 18 || max_util < 0.18 {
        run_asca_candidates(
            challenge,
            state,
            ca,
            hp,
            flows_base,
            &mut polished,
            &candidates,
            1,
        );
        if num_l > 0 && max_util > 0.35 {
            run_deflator(challenge, state, ca, hp, flows_base, &mut polished);
        }
    }

    let polish_score = approx_actions_value(challenge, state, ca, hp, flows_base, &polished);
    if polish_score.is_finite() && polish_score > best_score + 1e-7 {
        best_actions = polished;
    }

    best_actions
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
        let inner = guard.as_mut().expect("aiioagayp: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);

        let (max_util_base, focus_lines, focus_mask) =
            select_congested_focus_lines(challenge, hp, &flows_base);
        if challenge.num_batteries < 4
            || challenge.network.flow_limits.is_empty()
            || focus_lines.is_empty()
            || max_util_base < 0.58
        {
            return Ok(run_small_multistart_policy(challenge, state, cache, hp, &flows_base));
        }

        let incumbent_actions = run_small_multistart_policy(challenge, state, cache, hp, &flows_base);
        let incumbent_score =
            approx_actions_value(challenge, state, cache, hp, &flows_base, &incumbent_actions);

        let staged_actions = run_congested_relief_then_fill_policy(
            challenge,
            state,
            cache,
            hp,
            &flows_base,
            &focus_lines,
            &focus_mask,
        );
        let staged_score =
            approx_actions_value(challenge, state, cache, hp, &flows_base, &staged_actions);

        let mut best_actions = incumbent_actions.clone();
        let mut best_score = incumbent_score;
        if staged_score.is_finite() && staged_score > best_score + 1e-7 {
            best_score = staged_score;
            best_actions = staged_actions.clone();
        }

        let elites_distinct = incumbent_score.is_finite()
            && staged_score.is_finite()
            && elite_actions_are_distinct(challenge, state, &incumbent_actions, &staged_actions);

        if elites_distinct && (challenge.num_batteries <= 64 || max_util_base > 0.70) {
            if let Some((merged, score)) = run_elite_merge_rebuild(
                challenge,
                state,
                cache,
                hp,
                &flows_base,
                &incumbent_actions,
                &staged_actions,
            ) {
                if score > best_score + 1e-7 {
                    best_score = score;
                    best_actions = merged;
                }
            }
        }

        if (challenge.num_batteries <= 56 || max_util_base > 0.80) && elites_distinct {
            if let Some((relinked, score)) = run_elite_path_relink(
                challenge,
                state,
                cache,
                hp,
                &flows_base,
                &staged_actions,
                &incumbent_actions,
            ) {
                if score > best_score + 1e-7 {
                    best_score = score;
                    best_actions = relinked;
                }
            }

            if let Some((relinked, score)) = run_elite_path_relink(
                challenge,
                state,
                cache,
                hp,
                &flows_base,
                &incumbent_actions,
                &staged_actions,
            ) {
                if score > best_score + 1e-7 {
                    best_actions = relinked;
                }
            }
        }

        run_restricted_active_set_rebuild(challenge, state, cache, hp, &flows_base, &mut best_actions);
        Ok(best_actions)
    })
}

fn select_congested_focus_lines(
    challenge: &Challenge,
    hp: &TrackHp,
    flows_base: &[f64],
) -> (f64, Vec<usize>, Vec<bool>) {
    let num_l = challenge.network.flow_limits.len();
    let mut ranked: Vec<(usize, f64)> = Vec::new();
    let mut max_util = 0.0_f64;

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let util = (flows_base[l].abs() / limit).min(2.5);
        let slack_frac = ((limit - flows_base[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
        if util > max_util {
            max_util = util;
        }
        if util >= 0.48 {
            ranked.push((l, util + 0.20 * (1.0 - slack_frac)));
        }
    }

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep = if challenge.num_batteries <= 20 {
        2
    } else if challenge.num_batteries <= 60 {
        3
    } else {
        4
    };

    let lines: Vec<usize> = ranked.into_iter().take(keep).map(|(l, _)| l).collect();
    let mut mask = vec![false; num_l];
    for &l in &lines {
        mask[l] = true;
    }

    (max_util, lines, mask)
}

fn build_congested_relief_skeleton(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    focus_lines: &[usize],
    focus_mask: &[bool],
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    if num_b == 0 || focus_lines.is_empty() {
        return vec![0.0; num_b];
    }

    let mut ranked_candidates: Vec<(usize, f64)> = (0..num_b)
        .filter_map(|b| {
            let mut focus_touch = 0.0_f64;
            let mut relief_bias = 0.0_f64;
            let mut footprint = 1e-6_f64;

            for &(l, p) in &ca.b_to_lines[b] {
                if !focus_mask[l] {
                    continue;
                }
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }

                let util = (flows_base[l].abs() / limit).min(2.0);
                focus_touch += p.abs() * (0.25 + util);
                footprint += p.abs() * (0.10 + util * util);
                if flows_base[l] * p < 0.0 {
                    relief_bias += p.abs() * (0.35 + util);
                } else if flows_base[l].abs() <= 1e-9 {
                    relief_bias += 0.10 * p.abs();
                }
            }

            if focus_touch <= 1e-9 {
                return None;
            }

            let score = (0.25 * potential(challenge, state, ca, b) + relief_bias)
                * (1.0 + 0.10 * focus_touch)
                / footprint
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            Some((b, score))
        })
        .collect();
    ranked_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let shortlist_cap = if num_b <= 18 {
        ranked_candidates.len()
    } else if num_b <= 40 {
        ranked_candidates.len().min(18)
    } else {
        ranked_candidates.len().min(24)
    };
    let mut pool: Vec<usize> = ranked_candidates
        .into_iter()
        .take(shortlist_cap)
        .map(|(b, _)| b)
        .collect();

    let mut actions = vec![0.0_f64; num_b];
    let mut flows = flows_base.to_vec();
    let step_cap = if shortlist_cap <= 6 {
        shortlist_cap
    } else if num_b <= 24 {
        4
    } else if num_b <= 60 {
        5
    } else {
        6
    };
    let mut picked = 0usize;

    while !pool.is_empty() && picked < step_cap {
        let mut best_pos: Option<usize> = None;
        let mut best_b = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_score = 0.0_f64;

        for (pos, &b) in pool.iter().enumerate() {
            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, &flows, &actions, b);
            let gain = v - v_cur;
            let delta = u - actions[b];
            if delta.abs() <= 1e-9 {
                continue;
            }

            let mut focus_relief = 0.0_f64;
            let mut focus_worsen = 0.0_f64;
            let mut network_relief = 0.0_f64;
            let mut network_worsen = 0.0_f64;
            let mut footprint = 1e-6_f64;

            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }

                let util = (flows[l].abs() / limit).min(2.0);
                let move_frac = (p * delta).abs() / limit;
                if move_frac <= 1e-12 {
                    continue;
                }
                footprint += move_frac * (0.18 + util * util);

                let signed_move = flows[l] * p * delta;
                if focus_mask[l] {
                    if signed_move < 0.0 {
                        focus_relief += move_frac * (0.85 + 1.40 * util);
                    } else if signed_move > 0.0 {
                        focus_worsen += move_frac * (0.85 + 1.60 * util);
                    } else if flows[l].abs() <= 1e-9 {
                        focus_relief += 0.12 * move_frac;
                    }
                } else if util > 0.58 {
                    if signed_move < 0.0 {
                        network_relief += move_frac * (0.30 + util);
                    } else if signed_move > 0.0 {
                        network_worsen += move_frac * (0.30 + util);
                    }
                }
            }

            if focus_relief <= 1e-8 {
                continue;
            }
            if gain < -0.30 && focus_relief < 0.12 {
                continue;
            }

            let base = gain.max(0.0) + 0.08 * focus_relief + 0.03 * network_relief;
            let score = base
                * (1.0 + 1.25 * focus_relief + 0.20 * network_relief)
                / (footprint * (1.0 + 1.50 * focus_worsen + 0.45 * network_worsen));

            if score > best_score + 1e-12 {
                best_score = score;
                best_pos = Some(pos);
                best_b = b;
                best_u = u;
            }
        }

        let Some(pos) = best_pos else { break; };
        let delta = best_u - actions[best_b];
        if delta.abs() <= 1e-12 {
            break;
        }

        actions[best_b] = best_u;
        for &(l, p) in &ca.b_to_lines[best_b] {
            flows[l] += p * delta;
        }
        pool.swap_remove(pos);
        picked += 1;

        let mut max_focus_util = 0.0_f64;
        for &l in focus_lines {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            max_focus_util = max_focus_util.max((flows[l].abs() / limit).min(2.0));
        }
        if picked >= 2 && max_focus_util < 0.78 {
            break;
        }
    }

    if actions.iter().any(|&u| u.abs() > 1e-8) {
        actions
    } else {
        let keep = if num_b <= 18 {
            num_b
        } else if num_b <= 40 {
            (2 * num_b + 2) / 3
        } else {
            (num_b + 1) / 2
        };
        build_line_relief_seed(challenge, state, ca, hp, flows_base, keep.max(1))
    }
}

fn select_congested_fill_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    focus_lines: &[usize],
    focus_mask: &[bool],
) -> Vec<usize> {
    let num_b = challenge.num_batteries;
    if num_b <= 24 {
        return (0..num_b).collect();
    }

    let num_l = challenge.network.flow_limits.len();
    let target = if num_b <= 48 {
        18
    } else if num_b <= 80 {
        26
    } else {
        36
    }
    .min(num_b);
    let cap = if num_b <= 48 {
        24
    } else if num_b <= 80 {
        34
    } else {
        44
    }
    .min(num_b);

    let mut mask = vec![false; num_b];
    for b in 0..num_b {
        if actions[b].abs() > 1e-7 {
            mask[b] = true;
        }
    }
    for &l in focus_lines {
        for &(b, _) in &ca.ptdf_sparse[l] {
            mask[b] = true;
        }
    }
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= 0.68 {
            for &(b, _) in &ca.ptdf_sparse[l] {
                mask[b] = true;
            }
        }
    }

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| mask[b]).collect();

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
                    let focus = if focus_mask[l] { 1.0 } else { 0.0 };
                    touch += p.abs() * (0.25 * util + focus);
                }

                let score = potential(challenge, state, ca, b) * (1.0 + 0.10 * touch)
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
                    let focus = if focus_mask[l] { 1.0 } else { 0.0 };
                    touch += p.abs() * (0.20 * util + 1.20 * focus);
                }

                let score = actions[b].abs()
                    + 0.20 * potential(challenge, state, ca, b)
                    + 0.15 * touch;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(cap).map(|(b, _)| b).collect();
    }

    candidates
}

fn run_congested_relief_then_fill_policy(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    focus_lines: &[usize],
    focus_mask: &[bool],
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    if num_b == 0 {
        return Vec::new();
    }

    let mut actions = build_congested_relief_skeleton(
        challenge,
        state,
        ca,
        hp,
        flows_base,
        focus_lines,
        focus_mask,
    );
    let flows = compute_action_flows(ca, flows_base, &actions);
    let fill_candidates = select_congested_fill_candidates(
        challenge,
        state,
        ca,
        hp,
        &flows,
        &actions,
        focus_lines,
        focus_mask,
    );

    if !fill_candidates.is_empty() {
        let sweeps = if num_b <= 18 || fill_candidates.len() <= 18 {
            2
        } else {
            1
        };
        run_asca_candidates(
            challenge,
            state,
            ca,
            hp,
            flows_base,
            &mut actions,
            &fill_candidates,
            sweeps,
        );
        run_deflator(challenge, state, ca, hp, flows_base, &mut actions);

        let polish_sweeps = if num_b <= 18 || fill_candidates.len() <= 20 {
            2
        } else {
            1
        };
        run_asca_candidates(
            challenge,
            state,
            ca,
            hp,
            flows_base,
            &mut actions,
            &fill_candidates,
            polish_sweeps,
        );
        run_deflator(challenge, state, ca, hp, flows_base, &mut actions);
    }

    if num_b <= 36 {
        refine_small_seed(challenge, state, ca, hp, flows_base, &mut actions);
    } else {
        refine_basic_seed(challenge, state, ca, hp, flows_base, &mut actions);
    }

    actions
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
        let inner = guard.as_mut().expect("aiioagayp: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);

        let medium_actions = run_medium_multistart_policy(challenge, state, cache, hp, &flows_base);
        if challenge.num_batteries < 4 || state.time_step + 2 >= challenge.num_steps {
            return Ok(medium_actions);
        }

        let mut best_actions = medium_actions.clone();
        let mut best_score =
            approx_actions_value(challenge, state, cache, hp, &flows_base, &best_actions);

        let keep = if challenge.num_batteries <= 20 {
            challenge.num_batteries
        } else if challenge.num_batteries <= 48 {
            (4 * challenge.num_batteries + 4) / 5
        } else {
            (2 * challenge.num_batteries + 2) / 3
        }
        .min(challenge.num_batteries);

        let mut refined_gradient: Option<(Vec<f64>, f64)> = None;
        let mut gradient_actions =
            build_soc_gradient_multiday_seed(challenge, state, cache, hp, &flows_base, keep);
        if gradient_actions.iter().any(|&u| u.abs() > 1e-8) {
            let raw_score =
                approx_actions_value(challenge, state, cache, hp, &flows_base, &gradient_actions);
            if raw_score.is_finite() {
                let mut max_util_base = 0.0_f64;
                for l in 0..challenge.network.flow_limits.len() {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    max_util_base = max_util_base.max((flows_base[l].abs() / limit).min(2.0));
                }

                if challenge.num_batteries <= 24 || max_util_base > 0.55 {
                    refine_small_seed(challenge, state, cache, hp, &flows_base, &mut gradient_actions);
                } else {
                    refine_basic_seed(challenge, state, cache, hp, &flows_base, &mut gradient_actions);
                }

                let gradient_score = approx_actions_value(
                    challenge,
                    state,
                    cache,
                    hp,
                    &flows_base,
                    &gradient_actions,
                );
                if gradient_score.is_finite() {
                    refined_gradient = Some((gradient_actions.clone(), gradient_score));
                    if gradient_score > best_score + 1e-7 {
                        best_score = gradient_score;
                        best_actions = gradient_actions.clone();
                    }
                }
            }
        }

        if let Some((gradient_actions, gradient_score)) = refined_gradient {
            if gradient_score.is_finite()
                && elite_actions_are_distinct(challenge, state, &medium_actions, &gradient_actions)
            {
                if let Some((merged, score)) = run_multiday_shadow_merge_rebuild(
                    challenge,
                    state,
                    cache,
                    hp,
                    &flows_base,
                    &medium_actions,
                    &gradient_actions,
                ) {
                    if score > best_score + 1e-7 {
                        best_actions = merged;
                    }
                }
            }
        }

        run_restricted_active_set_rebuild(challenge, state, cache, hp, &flows_base, &mut best_actions);
        Ok(best_actions)
    })
}

#[inline]
fn multiday_shadow_direction_edge(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    b: usize,
) -> (i8, f64) {
    let bat = &challenge.batteries[b];
    let node = ca.batt_nodes[b];
    let rt_price = if node < state.rt_prices.len() {
        state.rt_prices[node]
    } else {
        0.0
    };
    let future_shadow = estimate_future_soc_shadow(challenge, state, ca, b);

    let discharge_edge = rt_price - future_shadow / bat.efficiency_discharge.max(1e-9) - 0.25;
    let charge_edge = future_shadow * bat.efficiency_charge - rt_price - 0.25;
    if discharge_edge >= charge_edge {
        (1_i8, discharge_edge.max(0.0))
    } else {
        (-1_i8, charge_edge.max(0.0))
    }
}

fn run_multiday_shadow_merge_rebuild(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    incumbent: &[f64],
    contrast: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b == 0 || incumbent.len() != num_b || contrast.len() != num_b {
        return None;
    }

    let incumbent_score = approx_actions_value(challenge, state, ca, hp, flows_base, incumbent);
    let contrast_score = approx_actions_value(challenge, state, ca, hp, flows_base, contrast);
    if !incumbent_score.is_finite() || !contrast_score.is_finite() {
        return None;
    }

    let (base, donor, base_score) = if incumbent_score >= contrast_score {
        (incumbent, contrast, incumbent_score)
    } else {
        (contrast, incumbent, contrast_score)
    };

    let base_flows = compute_action_flows(ca, flows_base, base);
    let feasible = (0..num_l).all(|l| {
        base_flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return None;
    }

    let mut ranked: Vec<(usize, f64)> = (0..num_b)
        .filter_map(|b| {
            let (dir, edge) = multiday_shadow_direction_edge(challenge, state, ca, b);
            let (lo, hi) = state.action_bounds[b];
            let scale = (hi - lo).abs().max(0.25);
            let diff_norm = ((base[b] - donor[b]).abs() / scale).min(4.0);

            let base_sign = if base[b] > 1e-7 {
                1_i8
            } else if base[b] < -1e-7 {
                -1_i8
            } else {
                0_i8
            };
            let donor_sign = if donor[b] > 1e-7 {
                1_i8
            } else if donor[b] < -1e-7 {
                -1_i8
            } else {
                0_i8
            };

            let mismatch = if edge <= 1e-9 {
                0.0
            } else if base_sign == dir {
                0.0
            } else if base_sign == 0 {
                0.45
            } else {
                1.0
            };
            let donor_align = if edge > 1e-9 && donor_sign == dir {
                1.0
            } else {
                0.0
            };
            let local_gain =
                eval_profit(challenge, state, ca, b, donor[b]) - eval_profit(challenge, state, ca, b, base[b]);

            if diff_norm <= 0.03 && mismatch <= 0.0 && donor_align <= 0.0 {
                return None;
            }

            let mut footprint = 1e-6_f64;
            let mut stressed_touch = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }
                let util = (base_flows[l].abs() / limit).min(2.0);
                footprint += p.abs() * (0.12 + util * util);
                if util > 0.45 {
                    stressed_touch += p.abs() * util;
                }
            }

            let score = (0.50 * diff_norm + 0.90 * mismatch + 0.30 * donor_align)
                * (1.0 + 0.03 * edge.min(20.0) + 0.08 * stressed_touch)
                / footprint
                + 0.04 * local_gain.max(0.0)
                + 0.02 * potential(challenge, state, ca, b)
                + 1e-6 * challenge.batteries[b].capacity_mwh;

            if score > 1e-8 {
                Some((b, score))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let block_cap = if num_b <= 16 {
        ranked.len().min(6)
    } else if num_b <= 40 {
        ranked.len().min(8)
    } else {
        ranked.len().min(10)
    };
    if block_cap == 0 {
        return None;
    }

    let block: Vec<usize> = ranked.into_iter().take(block_cap).map(|(b, _)| b).collect();
    let mut actions = base.to_vec();
    let mut touched = vec![false; num_b];

    for &b in &block {
        let (dir, edge) = multiday_shadow_direction_edge(challenge, state, ca, b);
        let donor_gain =
            eval_profit(challenge, state, ca, b, donor[b]) - eval_profit(challenge, state, ca, b, base[b]);

        let mut best_u = base[b];
        let mut best_v = f64::NEG_INFINITY;
        for &(u, from_donor) in &[(base[b], false), (donor[b], true), (0.0_f64, false)] {
            let sign = if u > 1e-7 {
                1_i8
            } else if u < -1e-7 {
                -1_i8
            } else {
                0_i8
            };
            let dir_bonus = if edge <= 1e-9 || sign == 0 {
                0.0
            } else if sign == dir {
                0.06 * edge.min(20.0) + 0.01 * u.abs()
            } else {
                -0.05 * edge.min(20.0)
            };
            let donor_bonus = if from_donor {
                0.05 * donor_gain.max(0.0)
            } else {
                0.0
            };
            let v = eval_profit(challenge, state, ca, b, u) + dir_bonus + donor_bonus;
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }

        if (best_u - actions[b]).abs() > 1e-9 {
            actions[b] = best_u;
            touched[b] = true;
        }
    }

    if !touched.iter().any(|&v| v) {
        return None;
    }

    scale_actions_toward_zero_to_feasible(challenge, state, ca, hp, flows_base, &mut actions);

    let flows = compute_action_flows(ca, flows_base, &actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return None;
    }

    let mut max_util = 0.0_f64;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        max_util = max_util.max((flows[l].abs() / limit).min(2.0));
    }

    let mut candidate_mask = touched.clone();
    for b in 0..num_b {
        if actions[b].abs() > 1e-7 || base[b].abs() > 1e-7 || donor[b].abs() > 1e-7 {
            candidate_mask[b] = true;
        }
    }
    for b in 0..num_b {
        if !touched[b] {
            continue;
        }
        for &(l, _) in &ca.b_to_lines[b] {
            if l < num_l {
                for &(bb, _) in &ca.ptdf_sparse[l] {
                    candidate_mask[bb] = true;
                }
            }
        }
    }

    let stress_cut = if max_util > 0.70 {
        0.60
    } else if max_util > 0.40 {
        0.45
    } else {
        0.30
    };
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= stress_cut {
            for &(b, _) in &ca.ptdf_sparse[l] {
                candidate_mask[b] = true;
            }
        }
    }

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| candidate_mask[b]).collect();
    if candidates.is_empty() {
        return None;
    }

    let cap = if num_b <= 24 {
        num_b
    } else if num_b <= 56 {
        22
    } else {
        30
    }
    .min(num_b);

    if candidates.len() > cap {
        let mut ranked: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&b| {
                let (lo, hi) = state.action_bounds[b];
                let scale = (hi - lo).abs().max(0.25);
                let diff_norm = ((base[b] - donor[b]).abs() / scale).min(4.0);

                let mut touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    touch += p.abs() * (0.10 + util);
                }

                let mut score = actions[b].abs()
                    + diff_norm
                    + 0.12 * potential(challenge, state, ca, b)
                    + 0.08 * touch;
                if touched[b] {
                    score += 1.0;
                }
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(cap).map(|(b, _)| b).collect();
    }

    let sweeps = if num_b <= 18 || candidates.len() <= 18 { 2 } else { 1 };
    run_asca_candidates(
        challenge,
        state,
        ca,
        hp,
        flows_base,
        &mut actions,
        &candidates,
        sweeps,
    );
    run_deflator(challenge, state, ca, hp, flows_base, &mut actions);

    if num_b <= 24 || candidates.len() <= 16 || max_util < 0.25 {
        run_asca_candidates(
            challenge,
            state,
            ca,
            hp,
            flows_base,
            &mut actions,
            &candidates,
            1,
        );
        if num_l > 0 && max_util > 0.35 {
            run_deflator(challenge, state, ca, hp, flows_base, &mut actions);
        }
    }

    let merged_score = approx_actions_value(challenge, state, ca, hp, flows_base, &actions);
    if merged_score > base_score + 1e-7 {
        Some((actions, merged_score))
    } else {
        None
    }
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
        let inner = guard.as_mut().expect("aiioagayp: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let zero_action = vec![0.0_f64; challenge.num_batteries];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        let mut actions = run_screened_multistart_policy(
            challenge,
            state,
            cache,
            hp,
            &flows_base,
            2,
        );
        run_restricted_active_set_rebuild(challenge, state, cache, hp, &flows_base, &mut actions);
        Ok(actions)
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
        let inner = guard.as_mut().expect("aiioagayp: STATE not initialised");
        if inner.cache.is_none() {
            inner.cache = Some(build_cache(challenge, state, &inner.hp));
        }
        let cache = inner.cache.as_ref().unwrap();
        let hp = &inner.hp;
        let num_b = challenge.num_batteries;
        if num_b == 0 {
            return Ok(Vec::new());
        }

        let zero_action = vec![0.0_f64; num_b];
        let inj_base_cur = challenge.compute_total_injections(state, &zero_action);
        let flows_base = challenge.network.compute_flows(&inj_base_cur);
        let (max_util_base, medium_tight_lines, scarce_lines) =
            capstone_stress_profile(challenge, hp, &flows_base);

        let mut seeds: Vec<Vec<f64>> = Vec::with_capacity(4);
        seeds.push(vec![0.0_f64; num_b]);

        let use_scaled_seed = challenge.network.flow_limits.is_empty()
            || max_util_base < 0.50
            || num_b <= 16;
        if use_scaled_seed {
            let keep = if num_b <= 20 {
                num_b
            } else if max_util_base < 0.25 {
                ((3 * num_b + 3) / 4).min(num_b)
            } else {
                ((num_b + 1) / 2).min(num_b)
            };
            let scaled_seed = build_scaled_greedy_seed(challenge, state, cache, hp, &flows_base, keep);
            if scaled_seed.iter().any(|&u| u.abs() > 1e-8) {
                seeds.push(scaled_seed);
            }
        }

        let use_relief_seed = !challenge.network.flow_limits.is_empty() && max_util_base >= 0.30;
        if use_relief_seed {
            let keep = if num_b <= 18 {
                num_b
            } else if scarce_lines >= 2 || max_util_base > 0.78 {
                ((3 * num_b + 3) / 4).min(num_b)
            } else {
                ((num_b + 1) / 2).min(num_b)
            };
            let relief_seed = build_line_relief_seed(challenge, state, cache, hp, &flows_base, keep);
            if relief_seed.iter().any(|&u| u.abs() > 1e-8) {
                seeds.push(relief_seed);
            }
        }

        let use_dual_seed = !challenge.network.flow_limits.is_empty()
            && (max_util_base >= 0.58 || scarce_lines >= 2 || medium_tight_lines >= 4);
        if use_dual_seed {
            let keep = if num_b <= 18 {
                num_b
            } else if scarce_lines >= 3 || max_util_base > 0.90 {
                ((2 * num_b + 2) / 3).min(num_b)
            } else {
                ((num_b + 1) / 2).min(num_b)
            };
            let dual_seed = build_dual_price_seed(challenge, state, cache, hp, &flows_base, keep);
            if dual_seed.iter().any(|&u| u.abs() > 1e-8) {
                seeds.push(dual_seed);
            }
        }

        let mut best_actions = vec![0.0_f64; num_b];
        let mut best_score = f64::NEG_INFINITY;
        for mut actions in seeds {
            let score = run_capstone_refinement(
                challenge,
                state,
                cache,
                hp,
                &flows_base,
                &mut actions,
            );
            if score.is_finite() && score > best_score + 1e-7 {
                best_score = score;
                best_actions = actions;
            }
        }

        run_restricted_active_set_rebuild(challenge, state, cache, hp, &flows_base, &mut best_actions);
        Ok(best_actions)
    })
}

fn capstone_stress_profile(
    challenge: &Challenge,
    hp: &TrackHp,
    flows_base: &[f64],
) -> (f64, usize, usize) {
    let mut max_util_base = 0.0_f64;
    let mut medium_tight_lines = 0usize;
    let mut scarce_lines = 0usize;

    for l in 0..challenge.network.flow_limits.len() {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let util = (flows_base[l].abs() / limit).min(2.0);
        if util > max_util_base {
            max_util_base = util;
        }
        if util >= 0.45 {
            medium_tight_lines += 1;
        }
        if util >= 0.70 {
            scarce_lines += 1;
        }
    }

    (max_util_base, medium_tight_lines, scarce_lines)
}

fn run_capstone_refinement(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) -> f64 {
    run_asca(challenge, state, ca, hp, flows_base, actions);
    run_deflator(challenge, state, ca, hp, flows_base, actions);
    approx_actions_value(challenge, state, ca, hp, flows_base, actions)
}

fn approx_actions_value(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &[f64],
) -> f64 {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();

    let mut total = 0.0_f64;
    if num_l == 0 {
        for b in 0..num_b {
            total += eval_profit(challenge, state, ca, b, actions[b]);
        }
        return total;
    }

    let mut flows = flows_base.to_vec();
    for b in 0..num_b {
        let u = actions[b];
        total += eval_profit(challenge, state, ca, b, u);
        if u == 0.0 {
            continue;
        }
        for &(l, p) in &ca.b_to_lines[b] {
            flows[l] += p * u;
        }
    }

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if flows[l].abs() > limit + 1e-7 {
            return f64::NEG_INFINITY;
        }
    }

    total
}

fn build_seed_value_densities(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    actions: &[f64],
) -> Vec<f64> {
    let mut densities = vec![0.0_f64; challenge.num_batteries];
    for b in 0..challenge.num_batteries {
        let u = actions[b];
        if u.abs() <= 1e-9 {
            continue;
        }
        let keep_val = eval_profit(challenge, state, ca, b, u);
        let zero_val = eval_profit(challenge, state, ca, b, 0.0);
        let gain = (keep_val - zero_val).max(0.0);
        densities[b] = gain / u.abs().max(1e-9);
    }
    densities
}

fn uniform_scale_seed_to_feasible(
    challenge: &Challenge,
    ca: &aiioagaypCache,
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

fn scale_actions_toward_zero_to_feasible(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();

    for b in 0..num_b {
        let (lo, hi) = state.action_bounds[b];
        actions[b] = actions[b].clamp(lo, hi);
    }
    if num_b == 0 || num_l == 0 {
        return;
    }

    let mut flows = compute_action_flows(ca, flows_base, actions);
    let mut feasible = true;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if flows[l].abs() > limit + 1e-7 {
            feasible = false;
            break;
        }
    }
    if feasible {
        return;
    }

    let value_density = build_seed_value_densities(challenge, state, ca, actions);
    let iter_cap = if num_b <= 24 {
        num_b + 8
    } else if num_b <= 80 {
        28
    } else {
        24
    };
    let mut line_overflow = vec![0.0_f64; num_l];
    let mut line_weight = vec![0.0_f64; num_l];
    let mut line_sign = vec![0.0_f64; num_l];

    for _ in 0..iter_cap {
        feasible = true;
        for l in 0..num_l {
            line_overflow[l] = 0.0;
            line_weight[l] = 0.0;
            line_sign[l] = 0.0;

            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if flows[l].abs() <= limit + 1e-7 {
                continue;
            }

            feasible = false;
            let overflow = flows[l].abs() - limit;
            let util = if limit > 1e-9 {
                (flows[l].abs() / limit).min(3.0)
            } else {
                3.0
            };
            line_overflow[l] = overflow;
            line_weight[l] = 1.0 + 2.5 * overflow / limit.max(1e-6) + 0.35 * util;
            line_sign[l] = flows[l].signum();
        }
        if feasible {
            break;
        }

        let mut best_b: Option<usize> = None;
        let mut best_score = 0.0_f64;
        let mut best_reduce = 0.0_f64;

        for b in 0..num_b {
            let u = actions[b];
            if u.abs() <= 1e-10 {
                continue;
            }

            let mut harm = 0.0_f64;
            let mut max_need = 0.0_f64;
            let mut stressed_touch = 0.0_f64;

            for &(l, p) in &ca.b_to_lines[b] {
                if line_overflow[l] <= 0.0 {
                    continue;
                }
                let contrib = p * u;
                if contrib * line_sign[l] <= 1e-10 {
                    continue;
                }

                let contrib_abs = contrib.abs();
                harm += line_weight[l] * contrib_abs;
                let need = line_overflow[l] / contrib_abs.max(1e-12);
                if need > max_need {
                    max_need = need;
                }
                stressed_touch += p.abs();
            }

            if harm <= 1e-12 || max_need <= 1e-12 {
                continue;
            }

            let kept_value = value_density[b] * u.abs()
                + 0.015 * u.abs()
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            let score = harm * (1.0 + 0.03 * stressed_touch) / (0.05 + kept_value);

            if score > best_score + 1e-12 {
                best_score = score;
                best_b = Some(b);
                best_reduce = max_need.clamp(0.0, 1.0);
            }
        }

        let Some(best_b) = best_b else { break; };
        if best_reduce <= 1e-12 {
            break;
        }

        let new_action = if best_reduce >= 1.0 - 1e-10 {
            0.0
        } else {
            actions[best_b] * (1.0 - best_reduce)
        };
        let delta = new_action - actions[best_b];
        if delta.abs() <= 1e-12 {
            break;
        }

        actions[best_b] = new_action;
        for &(l, p) in &ca.b_to_lines[best_b] {
            if l < num_l {
                flows[l] += p * delta;
            }
        }
    }

    feasible = true;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if flows[l].abs() > limit + 1e-7 {
            feasible = false;
            break;
        }
    }

    if !feasible {
        uniform_scale_seed_to_feasible(challenge, ca, hp, flows_base, actions);
    }

    for b in 0..num_b {
        let (lo, hi) = state.action_bounds[b];
        actions[b] = actions[b].clamp(lo, hi);
    }
}

fn build_sequential_revisit_pool(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    flows: &[f64],
    selected: &[usize],
    actions: &[f64],
) -> Vec<usize> {
    let num_l = challenge.network.flow_limits.len();
    if selected.len() <= 2 || num_l == 0 {
        return selected.to_vec();
    }

    let mut line_weight = vec![0.0_f64; num_l];
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let util = (flows[l].abs() / limit).min(2.0);
        let moved = ((flows[l] - flows_base[l]).abs() / limit).min(2.0);
        line_weight[l] = (util - 0.30).max(0.0) + 0.65 * moved;
    }

    let cap = if selected.len() <= 8 {
        selected.len()
    } else if selected.len() <= 18 {
        6
    } else if selected.len() <= 40 {
        8
    } else {
        10
    }.min(selected.len());

    let mut ranked: Vec<(usize, f64)> = selected
        .iter()
        .map(|&b| {
            let mut touch = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                touch += p.abs() * (0.15 + line_weight[l]);
            }

            let active = actions[b].abs();
            let score = touch * (1.0 + 0.35 * active)
                + 0.10 * potential(challenge, state, ca, b)
                + 0.05 * active
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            (b, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().take(cap).map(|(b, _)| b).collect()
}

fn build_sequential_greedy_seed(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    keep: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    if num_b == 0 || keep == 0 {
        return vec![0.0; num_b];
    }

    let num_l = challenge.network.flow_limits.len();
    let target_keep = keep.min(num_b);
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
    let selected: Vec<usize> = ranked.into_iter().take(target_keep).map(|(b, _)| b).collect();
    let mut pool = selected.clone();

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

    if !actions.iter().any(|&u| u.abs() > 1e-8) || num_l == 0 {
        return actions;
    }
    if target_keep <= 4 || (num_b > 80 && target_keep > (2 * num_b + 2) / 3) {
        return actions;
    }

    let revisit_pool = build_sequential_revisit_pool(
        challenge,
        state,
        ca,
        hp,
        flows_base,
        &flows,
        &selected,
        &actions,
    );
    if revisit_pool.len() < 2 {
        return actions;
    }

    let revisit_steps = if target_keep <= 10 {
        revisit_pool.len().min(4)
    } else if target_keep <= 24 {
        revisit_pool.len().min(5)
    } else {
        revisit_pool.len().min(6)
    };
    let mut revisited = vec![false; num_b];

    for _ in 0..revisit_steps {
        let mut best_move: Option<(usize, f64)> = None;
        let mut best_merit = 0.0_f64;

        for &b in &revisit_pool {
            if revisited[b] {
                continue;
            }

            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, &flows, &actions, b);
            let gain = v - v_cur;
            let delta = u - actions[b];
            if gain <= 1e-9 || delta.abs() <= 1e-9 {
                continue;
            }

            let mut move_cost = 1.0_f64;
            let mut relief_bonus = 0.0_f64;
            let mut worsen_penalty = 0.0_f64;

            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }

                let util = (flows[l].abs() / limit).min(2.0);
                let slack_frac = ((limit - flows[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
                let moved = ((flows[l] - flows_base[l]).abs() / limit).min(2.0);
                let move_frac = (p * delta).abs() / limit;
                move_cost += move_frac * (0.18 + util + 0.40 * moved);

                let signed_move = flows[l] * p * delta;
                if signed_move < 0.0 {
                    relief_bonus += move_frac * (0.35 + util + 0.20 * moved);
                } else if signed_move > 0.0 {
                    worsen_penalty += move_frac * (0.35 + util + 0.45 / (slack_frac + 0.12));
                }
            }

            let merit = gain * (1.0 + 0.30 * relief_bonus)
                / (move_cost * (1.0 + 0.35 * worsen_penalty));
            if merit > best_merit + 1e-12 {
                best_merit = merit;
                best_move = Some((b, u));
            }
        }

        let Some((b, best_u)) = best_move else { break; };
        let delta = best_u - actions[b];
        if delta.abs() <= 1e-12 {
            break;
        }

        actions[b] = best_u;
        revisited[b] = true;
        for &(l, p) in &ca.b_to_lines[b] {
            if l < num_l {
                flows[l] += p * delta;
            }
        }
    }

    actions
}

fn build_scaled_greedy_seed(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
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
    scale_actions_toward_zero_to_feasible(challenge, state, ca, hp, flows_base, &mut scaled_seed);

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
    ca: &aiioagaypCache,
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
    ca: &aiioagaypCache,
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

fn slack_seed_move_merit(
    challenge: &Challenge,
    ca: &aiioagaypCache,
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
    let mut scarce = 0.0_f64;
    let mut balance = 0.0_f64;

    for &(l, p) in &ca.b_to_lines[b] {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let util = (flows[l].abs() / limit).min(2.0);
        let slack_frac = ((limit - flows[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
        let move_flow = p * delta;
        footprint += move_flow.abs() * (0.12 + 0.55 * util * util);

        if focus_lines[l] {
            if flows[l].abs() <= 1e-9 {
                balance += move_flow.abs() * (0.25 + 0.2 * util);
            } else if flows[l] * move_flow < 0.0 {
                relief += move_flow.abs() * (0.8 + 1.2 * util);
            } else if flows[l] * move_flow > 0.0 {
                worsen += move_flow.abs() * (0.5 + 1.5 * util + 0.8 * (1.0 - slack_frac));
            }
        } else {
            scarce += move_flow.abs() * ((util - 0.30).max(0.0) + 0.20 / (slack_frac + 0.15));
        }
    }

    gain * (1.0 + 0.45 * relief + 0.10 * balance)
        / (footprint * (1.0 + 0.60 * worsen + 0.10 * scarce))
}

fn build_slack_balanced_seed(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
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
        if util > 0.18 {
            let slack_frac = ((limit - flows_base[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
            line_rank.push((l, util + 0.15 * (1.0 - slack_frac)));
        }
    }
    line_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if line_rank.is_empty() {
        return build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    }

    let keep_lines = if num_b <= 20 { 2 } else if num_b <= 60 { 3 } else { 4 };
    let selected_lines: Vec<usize> = line_rank.iter().take(keep_lines).map(|(l, _)| *l).collect();
    let mut focus_lines = vec![false; num_l];
    for &l in &selected_lines {
        focus_lines[l] = true;
    }

    let target_keep = keep.min(num_b);
    let shortlist_cap = if target_keep <= 8 {
        target_keep
    } else {
        ((3 * target_keep + 1) / 2).min(num_b)
    };
    let zero_actions = vec![0.0; num_b];

    let mut ranked: Vec<(usize, f64, usize)> = (0..num_b)
        .filter_map(|b| {
            let v0 = eval_profit(challenge, state, ca, b, 0.0);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, flows_base, &zero_actions, b);
            let gain = v - v0;
            if gain <= 1e-9 || u.abs() <= 1e-9 {
                return None;
            }

            let mut signature = 0usize;
            for &l in &selected_lines {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    signature *= 5;
                    signature += 2;
                    continue;
                }

                let mut move_flow = 0.0_f64;
                for &(ll, p) in &ca.b_to_lines[b] {
                    if ll == l {
                        move_flow = p * u;
                        break;
                    }
                }

                let move_abs = move_flow.abs();
                let mild_cut = 0.03 * limit;
                let strong_cut = 0.12 * limit;
                let state = if move_abs <= mild_cut {
                    2usize
                } else {
                    let signed = if flows_base[l].abs() > 1e-9 {
                        flows_base[l].signum() * move_flow
                    } else {
                        move_flow
                    };
                    if signed < 0.0 {
                        if move_abs > strong_cut { 0usize } else { 1usize }
                    } else if move_abs > strong_cut {
                        4usize
                    } else {
                        3usize
                    }
                };
                signature = signature * 5 + state;
            }

            let score = slack_seed_move_merit(challenge, ca, hp, flows_base, &focus_lines, b, u, gain)
                + 0.025 * gain
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            Some((b, score, signature))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.is_empty() {
        return build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep);
    }

    let rep_target = if target_keep <= 4 {
        target_keep
    } else if target_keep <= 10 {
        4
    } else if target_keep <= 20 {
        5
    } else {
        6
    }.min(target_keep);

    let mut bucket_count = 1usize;
    for _ in 0..selected_lines.len() {
        bucket_count *= 5;
    }
    let mut seen_bucket = vec![false; bucket_count.max(1)];
    let mut representative_mask = vec![false; num_b];
    let mut representatives: Vec<usize> = Vec::new();

    for &(b, _, signature) in &ranked {
        if !seen_bucket[signature] {
            seen_bucket[signature] = true;
            representative_mask[b] = true;
            representatives.push(b);
            if representatives.len() >= rep_target {
                break;
            }
        }
    }
    if representatives.len() < rep_target {
        for &(b, _, _) in &ranked {
            if !representative_mask[b] {
                representative_mask[b] = true;
                representatives.push(b);
                if representatives.len() >= rep_target {
                    break;
                }
            }
        }
    }

    let mut actions = vec![0.0; num_b];
    let mut flows = flows_base.to_vec();
    let mut picked = 0usize;
    let mut representative_pool = representatives.clone();

    while !representative_pool.is_empty() && picked < rep_target {
        let mut best_pos: Option<usize> = None;
        let mut best_b = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_merit = 0.0_f64;

        for (pos, &b) in representative_pool.iter().enumerate() {
            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, &flows, &actions, b);
            let gain = v - v_cur;
            let delta = u - actions[b];
            if gain <= 1e-9 || delta.abs() <= 1e-9 {
                continue;
            }

            let merit = slack_seed_move_merit(challenge, ca, hp, &flows, &focus_lines, b, delta, gain)
                + 0.020 * gain / (1.0 + actions[b].abs());
            if merit > best_merit + 1e-12 {
                best_merit = merit;
                best_pos = Some(pos);
                best_b = b;
                best_u = u;
            }
        }

        let Some(pos) = best_pos else { break; };
        let was_inactive = actions[best_b].abs() <= 1e-9;
        let delta = best_u - actions[best_b];
        if delta.abs() > 1e-12 {
            actions[best_b] = best_u;
            for &(l, p) in &ca.b_to_lines[best_b] {
                if l < num_l {
                    flows[l] += p * delta;
                }
            }
        }
        representative_pool.swap_remove(pos);
        if was_inactive && actions[best_b].abs() > 1e-9 {
            picked += 1;
        }
    }

    let mut pool: Vec<usize> = Vec::new();
    let mut in_pool = vec![false; num_b];
    for &b in &representatives {
        pool.push(b);
        in_pool[b] = true;
    }
    for &(b, _, _) in ranked.iter().take(shortlist_cap) {
        if !in_pool[b] {
            pool.push(b);
            in_pool[b] = true;
        }
    }

    while !pool.is_empty() && picked < target_keep {
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

            let merit = slack_seed_move_merit(challenge, ca, hp, &flows, &focus_lines, b, delta, gain)
                + 0.015 * gain / (1.0 + actions[b].abs());
            if merit > best_merit + 1e-12 {
                best_merit = merit;
                best_pos = Some(pos);
                best_b = b;
                best_u = u;
            }
        }

        let Some(pos) = best_pos else { break; };
        let was_inactive = actions[best_b].abs() <= 1e-9;
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
        if was_inactive && actions[best_b].abs() > 1e-9 {
            picked += 1;
        }
    }

    if actions.iter().any(|&u| u.abs() > 1e-8) {
        actions
    } else {
        build_scaled_greedy_seed(challenge, state, ca, hp, flows_base, keep)
    }
}

#[inline]
fn dual_price_penalized_value(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    b: usize,
    u: f64,
    priced_impacts: &[f64],
    line_dirs: &[f64],
    shadow_prices: &[f64],
) -> f64 {
    let mut value = eval_profit(challenge, state, ca, b, u);
    for k in 0..shadow_prices.len() {
        value -= shadow_prices[k] * line_dirs[k] * priced_impacts[k] * u;
    }
    value
}

fn best_dual_price_action(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    b: usize,
    cur: f64,
    priced_impacts: &[f64],
    line_dirs: &[f64],
    shadow_prices: &[f64],
) -> (f64, f64) {
    let (u_min, u_max) = state.action_bounds[b];
    let cur = cur.clamp(u_min, u_max);

    let mut best_u = cur;
    let mut best_v = dual_price_penalized_value(
        challenge,
        state,
        ca,
        b,
        best_u,
        priced_impacts,
        line_dirs,
        shadow_prices,
    );

    let mut probes = Vec::with_capacity(6);
    push_unique_probe(&mut probes, u_min);
    push_unique_probe(&mut probes, u_max);
    push_unique_probe(&mut probes, cur);
    if u_min <= 0.0 && 0.0 <= u_max {
        push_unique_probe(&mut probes, 0.0);
    }
    if u_min < cur - 1e-7 {
        push_unique_probe(&mut probes, 0.5 * (u_min + cur));
    }
    if cur + 1e-7 < u_max {
        push_unique_probe(&mut probes, 0.5 * (u_max + cur));
    }

    for u in probes {
        let v = dual_price_penalized_value(
            challenge,
            state,
            ca,
            b,
            u,
            priced_impacts,
            line_dirs,
            shadow_prices,
        );
        if v > best_v {
            best_v = v;
            best_u = u;
        }
    }

    if u_min < 0.0 {
        let lo = u_min;
        let hi = 0.0_f64.min(u_max);
        if lo < hi {
            let (u, v) = ternary_search(
                |u| dual_price_penalized_value(
                    challenge,
                    state,
                    ca,
                    b,
                    u,
                    priced_impacts,
                    line_dirs,
                    shadow_prices,
                ),
                lo,
                hi,
                hp.ternary_iters,
            );
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
            let (u, v) = ternary_search(
                |u| dual_price_penalized_value(
                    challenge,
                    state,
                    ca,
                    b,
                    u,
                    priced_impacts,
                    line_dirs,
                    shadow_prices,
                ),
                lo,
                hi,
                hp.ternary_iters,
            );
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }
    }

    (best_u, best_v)
}

fn best_post_deflator_shadow_action(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
    priced_impacts: &[f64],
    line_dirs: &[f64],
    shadow_prices: &[f64],
) -> (f64, f64) {
    let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);

    let mut best_u = actions[b].clamp(u_min, u_max);
    let mut best_v = dual_price_penalized_value(
        challenge,
        state,
        ca,
        b,
        best_u,
        priced_impacts,
        line_dirs,
        shadow_prices,
    );

    let v_min = dual_price_penalized_value(
        challenge,
        state,
        ca,
        b,
        u_min,
        priced_impacts,
        line_dirs,
        shadow_prices,
    );
    if v_min > best_v {
        best_v = v_min;
        best_u = u_min;
    }

    if (u_max - u_min).abs() > 1e-12 {
        let v_max = dual_price_penalized_value(
            challenge,
            state,
            ca,
            b,
            u_max,
            priced_impacts,
            line_dirs,
            shadow_prices,
        );
        if v_max > best_v {
            best_v = v_max;
            best_u = u_max;
        }
    }

    if u_min <= 0.0 && 0.0 <= u_max {
        let v0 = dual_price_penalized_value(
            challenge,
            state,
            ca,
            b,
            0.0,
            priced_impacts,
            line_dirs,
            shadow_prices,
        );
        if v0 > best_v {
            best_v = v0;
            best_u = 0.0;
        }
    }

    for u in build_single_extra_probes(challenge, ca, hp, flows, actions, b, u_min, u_max) {
        let v = dual_price_penalized_value(
            challenge,
            state,
            ca,
            b,
            u,
            priced_impacts,
            line_dirs,
            shadow_prices,
        );
        if v > best_v {
            best_v = v;
            best_u = u;
        }
    }

    if u_min < 0.0 {
        let lo = u_min;
        let hi = 0.0_f64.min(u_max);
        if lo < hi {
            let (u, v) = ternary_search(
                |u| dual_price_penalized_value(
                    challenge,
                    state,
                    ca,
                    b,
                    u,
                    priced_impacts,
                    line_dirs,
                    shadow_prices,
                ),
                lo,
                hi,
                hp.ternary_iters,
            );
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
            let (u, v) = ternary_search(
                |u| dual_price_penalized_value(
                    challenge,
                    state,
                    ca,
                    b,
                    u,
                    priced_impacts,
                    line_dirs,
                    shadow_prices,
                ),
                lo,
                hi,
                hp.ternary_iters,
            );
            if v > best_v {
                best_v = v;
                best_u = u;
            }
        }
    }

    (best_u, best_v)
}

fn build_post_deflator_shadow_profile(
    challenge: &Challenge,
    hp: &TrackHp,
    flows: &[f64],
    feedback: &LineFeedback,
) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let num_l = challenge.network.flow_limits.len();
    let mut ranked: Vec<(usize, f64, f64, f64)> = Vec::new();

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let util = (flows[l].abs() / limit).min(3.0);
        let relief = feedback.relieved[l].max(0.0);
        let tight = feedback.tight[l].max(0.0);
        if util < 0.62 && relief < 0.04 && tight < 0.04 {
            continue;
        }

        let util_tight = (util - 0.62).max(0.0);
        let very_tight = (util - 0.88).max(0.0);
        let price = (2.8 * util_tight + 16.0 * very_tight + 6.5 * tight + 3.0 * relief)
            .clamp(0.0, 32.0);
        let score = price + 1.20 * tight + 0.70 * relief;
        if score <= 1e-8 {
            continue;
        }

        let dir = if flows[l].abs() > 1e-9 {
            flows[l].signum()
        } else {
            0.0
        };
        ranked.push((l, score, dir, price));
    }

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep = if num_l <= 10 {
        ranked.len().min(4)
    } else if num_l <= 24 {
        ranked.len().min(5)
    } else {
        ranked.len().min(6)
    };

    let mut lines = Vec::with_capacity(keep);
    let mut dirs = Vec::with_capacity(keep);
    let mut prices = Vec::with_capacity(keep);
    for (l, _, dir, price) in ranked.into_iter().take(keep) {
        lines.push(l);
        dirs.push(dir);
        prices.push(price);
    }
    (lines, dirs, prices)
}

fn run_post_deflator_dual_reexpansion(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
    candidates: &[usize],
    feedback: &LineFeedback,
) -> bool {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b == 0 || num_l == 0 || candidates.is_empty() {
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
        if limit <= 1e-6 {
            continue;
        }
        max_util = max_util.max((flows[l].abs() / limit).min(2.0));
    }
    let relief_mass = feedback_relief_mass(feedback);
    let tight_mass = feedback_tight_mass(feedback);
    if max_util < 0.72 && relief_mass < 0.10 && tight_mass < 0.08 {
        return false;
    }

    let passes = if max_util > 0.90 || relief_mass + tight_mass > 0.30 { 2 } else { 1 };
    let shortlist_cap = if candidates.len() <= 16 {
        candidates.len()
    } else if candidates.len() <= 36 {
        16
    } else {
        24
    };
    let step_cap = if candidates.len() <= 16 {
        4
    } else if candidates.len() <= 36 {
        5
    } else {
        6
    };

    let mut improved_any = false;

    for _ in 0..passes {
        let mut moved_in_pass = false;

        for _ in 0..step_cap {
            let (shadow_lines, line_dirs, shadow_prices) =
                build_post_deflator_shadow_profile(challenge, hp, &flows, feedback);
            if shadow_lines.is_empty() {
                break;
            }

            let mut priced_impacts = vec![vec![0.0_f64; shadow_lines.len()]; num_b];
            for (k, &l) in shadow_lines.iter().enumerate() {
                for &(b, impact) in &ca.ptdf_sparse[l] {
                    priced_impacts[b][k] = impact;
                }
            }

            let mut ranked: Vec<(usize, f64)> = candidates
                .iter()
                .filter_map(|&b| {
                    let mut shadow_touch = 0.0_f64;
                    let mut relieved_touch = 0.0_f64;
                    let mut tight_touch = 0.0_f64;
                    for k in 0..shadow_lines.len() {
                        let abs_imp = priced_impacts[b][k].abs();
                        shadow_touch += shadow_prices[k] * abs_imp;
                        relieved_touch += feedback.relieved[shadow_lines[k]] * abs_imp;
                        tight_touch += feedback.tight[shadow_lines[k]] * abs_imp;
                    }

                    if shadow_touch <= 1e-9
                        && actions[b].abs() <= 1e-8
                        && relieved_touch <= 1e-9
                        && tight_touch <= 1e-9
                    {
                        return None;
                    }

                    let score = actions[b].abs()
                        + 0.12 * potential(challenge, state, ca, b)
                        + shadow_touch
                        + 0.20 * relieved_touch
                        + 0.45 * tight_touch;
                    Some((b, score))
                })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if ranked.is_empty() {
                break;
            }

            let mut best_move: Option<(usize, f64)> = None;
            let mut best_score = 0.0_f64;

            for &(b, _) in ranked.iter().take(shortlist_cap) {
                let cur_u = actions[b];
                let cur_pen = dual_price_penalized_value(
                    challenge,
                    state,
                    ca,
                    b,
                    cur_u,
                    &priced_impacts[b],
                    &line_dirs,
                    &shadow_prices,
                );
                let (best_u, best_pen) = best_post_deflator_shadow_action(
                    challenge,
                    state,
                    ca,
                    hp,
                    &flows,
                    actions,
                    b,
                    &priced_impacts[b],
                    &line_dirs,
                    &shadow_prices,
                );
                let delta_pen = best_pen - cur_pen;
                let delta = best_u - cur_u;
                if delta_pen <= 1e-8 || delta.abs() <= 1e-8 {
                    continue;
                }

                let actual_gain =
                    eval_profit(challenge, state, ca, b, best_u) - eval_profit(challenge, state, ca, b, cur_u);
                if actual_gain < -0.20 && delta_pen < 0.15 {
                    continue;
                }

                let score = delta_pen + 0.03 * actual_gain.max(-2.0);
                if score > best_score + 1e-12 {
                    best_score = score;
                    best_move = Some((b, best_u));
                }
            }

            let Some((b, best_u)) = best_move else { break; };
            let delta = best_u - actions[b];
            if delta.abs() <= 1e-12 {
                break;
            }

            actions[b] = best_u;
            for &(l, p) in &ca.b_to_lines[b] {
                if l < num_l {
                    flows[l] += p * delta;
                }
            }
            improved_any = true;
            moved_in_pass = true;
        }

        if !moved_in_pass {
            break;
        }
    }

    improved_any
}

fn build_dual_price_seed(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    keep: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b == 0 || num_l == 0 || keep == 0 {
        return vec![0.0; num_b];
    }

    let mut line_rank: Vec<(usize, f64)> = Vec::new();
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows_base[l].abs() / limit).min(2.0);
        if util < 0.20 {
            continue;
        }
        let slack_frac = ((limit - flows_base[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
        line_rank.push((l, util + 0.20 * (1.0 - slack_frac)));
    }
    line_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if line_rank.is_empty() {
        return vec![0.0; num_b];
    }

    let price_line_cap = if num_b <= 24 { 2 } else if num_b <= 64 { 3 } else { 4 };
    let selected_lines: Vec<usize> = line_rank
        .iter()
        .take(price_line_cap.min(line_rank.len()))
        .map(|(l, _)| *l)
        .collect();
    if selected_lines.is_empty() {
        return vec![0.0; num_b];
    }

    let mut focus_mask = vec![false; num_l];
    for &l in &selected_lines {
        focus_mask[l] = true;
    }

    let keep_cap = keep.min(num_b);
    let mut ranked_batteries: Vec<(usize, f64)> = (0..num_b)
        .map(|b| {
            let mut focus_touch = 0.0_f64;
            let mut stress_touch = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let abs_p = p.abs();
                if focus_mask[l] {
                    focus_touch += abs_p;
                }
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit > 1e-6 {
                    let util = (flows_base[l].abs() / limit).min(2.0);
                    stress_touch += abs_p * util;
                }
            }

            let score = potential(challenge, state, ca, b)
                * (1.0 + 0.55 * focus_touch + 0.08 * stress_touch)
                + 0.01 * focus_touch
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            (b, score)
        })
        .collect();
    ranked_batteries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let pool: Vec<usize> = ranked_batteries
        .into_iter()
        .take(keep_cap)
        .map(|(b, _)| b)
        .collect();
    if pool.is_empty() {
        return vec![0.0; num_b];
    }

    let mut priced_impacts = vec![vec![0.0_f64; selected_lines.len()]; num_b];
    for (k, &l) in selected_lines.iter().enumerate() {
        for &(b, impact) in &ca.ptdf_sparse[l] {
            priced_impacts[b][k] = impact;
        }
    }

    let dual_iters = if num_b <= 18 { 4 } else if num_b <= 48 { 3 } else { 2 };
    let mut actions = vec![0.0_f64; num_b];
    let mut shadow_prices = vec![0.0_f64; selected_lines.len()];
    let mut line_dirs: Vec<f64> = selected_lines
        .iter()
        .map(|&l| {
            if flows_base[l].abs() > 1e-9 {
                flows_base[l].signum()
            } else {
                1.0
            }
        })
        .collect();

    for _ in 0..dual_iters {
        for &b in &pool {
            let (u, _) = best_dual_price_action(
                challenge,
                state,
                ca,
                hp,
                b,
                actions[b],
                &priced_impacts[b],
                &line_dirs,
                &shadow_prices,
            );
            actions[b] = u;
        }

        let flows = compute_action_flows(ca, flows_base, &actions);
        for (k, &l) in selected_lines.iter().enumerate() {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }

            if flows[l].abs() > 1e-9 {
                line_dirs[k] = flows[l].signum();
            }
            let util = (flows[l].abs() / limit).min(2.5);
            let overflow = (util - 1.0).max(0.0);
            let near_tight = (util - 0.82).max(0.0);
            let target_shadow = 22.0 * overflow + 4.0 * near_tight;
            let decay = if util < 0.70 { 0.55 } else { 0.78 };
            shadow_prices[k] = (shadow_prices[k] * decay + target_shadow).clamp(0.0, 40.0);
        }
    }

    scale_actions_toward_zero_to_feasible(challenge, state, ca, hp, flows_base, &mut actions);
    if actions.iter().any(|&u| u.abs() > 1e-8) {
        actions
    } else {
        vec![0.0; num_b]
    }
}

fn estimate_future_soc_shadow(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    b: usize,
) -> f64 {
    let bat = &challenge.batteries[b];
    let soc_levels = ca.dp[b][0].len();
    if soc_levels <= 1 {
        return 0.0;
    }

    let soc_span = (bat.soc_max_mwh - bat.soc_min_mwh).max(1e-9);
    let t_next = (state.time_step + 1).min(ca.dp[b].len() - 1);
    let idx_f = ((state.socs[b] - bat.soc_min_mwh) / soc_span * ((soc_levels - 1) as f64))
        .clamp(0.0, (soc_levels - 1) as f64);

    let mid = idx_f.round() as usize;
    let i0 = mid.saturating_sub(1);
    let i1 = (mid + 1).min(soc_levels - 1);
    if i0 == i1 {
        return 0.0;
    }

    let dsoc = soc_span * ((i1 - i0) as f64) / ((soc_levels - 1) as f64);
    if dsoc <= 1e-9 {
        return 0.0;
    }

    (ca.dp[b][t_next][i1] - ca.dp[b][t_next][i0]) / dsoc
}

fn best_action_in_directional_window(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
    direction: i8,
) -> (f64, f64) {
    if direction == 0 {
        return best_action_in_window(challenge, state, ca, hp, flows, actions, b);
    }

    let (mut u_min, mut u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);
    if direction > 0 {
        u_min = u_min.max(0.0);
    } else {
        u_max = u_max.min(0.0);
    }

    if u_min > u_max + 1e-12 {
        let u = actions[b];
        return (u, eval_profit(challenge, state, ca, b, u));
    }

    let cur = actions[b].clamp(u_min, u_max);
    let mut best_u = cur;
    let mut best_v = eval_profit(challenge, state, ca, b, cur);

    let mut probes = Vec::with_capacity(6);
    push_unique_probe(&mut probes, u_min);
    push_unique_probe(&mut probes, u_max);
    push_unique_probe(&mut probes, cur);
    if u_min <= 0.0 && 0.0 <= u_max {
        push_unique_probe(&mut probes, 0.0);
    }
    if u_min < cur - 1e-7 {
        push_unique_probe(&mut probes, 0.5 * (u_min + cur));
    }
    if cur + 1e-7 < u_max {
        push_unique_probe(&mut probes, 0.5 * (u_max + cur));
    }

    for u in probes {
        let v = eval_profit(challenge, state, ca, b, u);
        if v > best_v {
            best_v = v;
            best_u = u;
        }
    }

    if u_min < u_max {
        let (u, v) =
            ternary_search(|u| eval_profit(challenge, state, ca, b, u), u_min, u_max, hp.ternary_iters);
        if v > best_v {
            best_v = v;
            best_u = u;
        }
    }

    (best_u, best_v)
}

fn build_soc_gradient_multiday_seed(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    keep: usize,
) -> Vec<f64> {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b == 0 || keep == 0 {
        return vec![0.0; num_b];
    }

    let target_keep = keep.min(num_b);
    let shortlist_cap = if target_keep <= 8 {
        target_keep
    } else {
        ((3 * target_keep + 1) / 2).min(num_b)
    };

    let mut ranked: Vec<(usize, f64, i8)> = (0..num_b)
        .filter_map(|b| {
            let bat = &challenge.batteries[b];
            let node = ca.batt_nodes[b];
            let rt_price = if node < state.rt_prices.len() {
                state.rt_prices[node]
            } else {
                0.0
            };
            let future_shadow = estimate_future_soc_shadow(challenge, state, ca, b);

            let discharge_edge =
                rt_price - future_shadow / bat.efficiency_discharge.max(1e-9) - 0.25;
            let charge_edge = future_shadow * bat.efficiency_charge - rt_price - 0.25;
            let (direction, edge) = if discharge_edge >= charge_edge {
                (1_i8, discharge_edge)
            } else {
                (-1_i8, charge_edge)
            };
            if edge <= 1e-6 {
                return None;
            }

            let mut footprint = 1e-6_f64;
            let mut stressed_touch = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }
                let util = (flows_base[l].abs() / limit).min(2.0);
                footprint += p.abs() * (0.18 + util * util);
                if util > 0.35 {
                    stressed_touch += p.abs() * util;
                }
            }

            let shadow_scale = rt_price.abs().max(25.0);
            let flatness = 1.0 / (1.0 + future_shadow.abs() / shadow_scale);
            let score = edge * (1.0 + 0.12 * flatness + 0.05 * stressed_touch) / footprint
                + 1e-6 * bat.capacity_mwh;
            Some((b, score, direction))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if ranked.is_empty() {
        return vec![0.0; num_b];
    }

    let mut pool: Vec<(usize, i8)> = ranked
        .into_iter()
        .take(shortlist_cap)
        .map(|(b, _, direction)| (b, direction))
        .collect();

    let mut actions = vec![0.0; num_b];
    let mut flows = flows_base.to_vec();
    let mut picked = 0usize;

    while !pool.is_empty() && picked < target_keep {
        let mut best_pos: Option<usize> = None;
        let mut best_b = 0usize;
        let mut best_u = 0.0_f64;
        let mut best_merit = 0.0_f64;

        for (pos, &(b, direction)) in pool.iter().enumerate() {
            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) =
                best_action_in_directional_window(challenge, state, ca, hp, &flows, &actions, b, direction);
            let gain = v - v_cur;
            let delta = u - actions[b];
            if gain <= 1e-9 || delta.abs() <= 1e-9 {
                continue;
            }

            let mut move_cost = 1.0_f64;
            let mut relief = 0.0_f64;
            let mut worsen = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }

                let util = (flows[l].abs() / limit).min(2.0);
                let slack_frac = ((limit - flows[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
                let move_frac = (p * delta).abs() / limit;
                move_cost += move_frac * (0.18 + util + 0.20 * util * util);

                let signed_move = flows[l] * p * delta;
                if signed_move < 0.0 {
                    relief += move_frac * (0.35 + util);
                } else if signed_move > 0.0 {
                    worsen += move_frac * (0.35 + util + 0.35 / (slack_frac + 0.15));
                }
            }

            let merit =
                gain * (1.0 + 0.25 * relief) / (move_cost * (1.0 + 0.35 * worsen));
            if merit > best_merit + 1e-12 {
                best_merit = merit;
                best_pos = Some(pos);
                best_b = b;
                best_u = u;
            }
        }

        let Some(pos) = best_pos else { break; };
        let was_inactive = actions[best_b].abs() <= 1e-9;
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
        if was_inactive && actions[best_b].abs() > 1e-9 {
            picked += 1;
        }
    }

    actions
}

fn elite_actions_distance(
    challenge: &Challenge,
    state: &State,
    lhs: &[f64],
    rhs: &[f64],
) -> (usize, f64) {
    let mut changed = 0usize;
    let mut total = 0.0_f64;

    for b in 0..challenge.num_batteries {
        let (lo, hi) = state.action_bounds[b];
        let scale = (hi - lo).abs().max(0.25);
        let delta = (lhs[b] - rhs[b]).abs() / scale;
        total += delta;
        if delta > 0.03 {
            changed += 1;
        }
    }

    (changed, total)
}

fn elite_actions_are_distinct(
    challenge: &Challenge,
    state: &State,
    lhs: &[f64],
    rhs: &[f64],
) -> bool {
    let (changed, total) = elite_actions_distance(challenge, state, lhs, rhs);
    (changed >= 2 && total > 0.35) || total > 0.85
}

fn relink_move_merit(
    challenge: &Challenge,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    b: usize,
    delta: f64,
    progress_norm: f64,
    profit_gain: f64,
) -> f64 {
    if delta.abs() <= 1e-9 || progress_norm <= 1e-9 {
        return f64::NEG_INFINITY;
    }

    let mut relief = 0.0_f64;
    let mut worsen = 0.0_f64;
    let mut footprint = 1e-6_f64;

    for &(l, p) in &ca.b_to_lines[b] {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let util = (flows[l].abs() / limit).min(2.0);
        let slack_frac = ((limit - flows[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
        let move_frac = (p * delta).abs() / limit;
        footprint += move_frac * (0.15 + util * util);

        let signed_move = flows[l] * p * delta;
        if signed_move < 0.0 {
            relief += move_frac * (0.45 + util + 0.15 * (1.0 - slack_frac));
        } else if signed_move > 0.0 {
            worsen += move_frac * (0.45 + util + 0.45 / (slack_frac + 0.12));
        }
    }

    let base = profit_gain + 0.10 * progress_norm;
    base * (1.0 + 0.35 * relief) / (1.0 + 0.55 * worsen) - 0.03 * footprint
}

fn run_elite_path_relink(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    source: &[f64],
    target: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let num_b = challenge.num_batteries;
    if num_b == 0 || source.len() != num_b || target.len() != num_b {
        return None;
    }

    let source_score = approx_actions_value(challenge, state, ca, hp, flows_base, source);
    if !source_score.is_finite() {
        return None;
    }

    let (diff_count, diff_total) = elite_actions_distance(challenge, state, source, target);
    if diff_count < 2 || diff_total < 0.35 {
        return None;
    }

    let mut ranked_diff: Vec<(usize, f64)> = (0..num_b)
        .filter_map(|b| {
            let (lo, hi) = state.action_bounds[b];
            let scale = (hi - lo).abs().max(0.25);
            let norm_delta = (target[b] - source[b]).abs() / scale;
            if norm_delta <= 0.03 {
                return None;
            }

            let mut stress_touch = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit > 1e-6 {
                    let util = (flows_base[l].abs() / limit).min(2.0);
                    stress_touch += p.abs() * (0.25 + util * util);
                }
            }

            let score = norm_delta * (1.0 + 0.05 * stress_touch)
                + 0.02 * potential(challenge, state, ca, b);
            Some((b, score))
        })
        .collect();
    ranked_diff.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let diff_cap = if num_b <= 18 {
        8
    } else if num_b <= 48 {
        10
    } else {
        12
    };
    let relink_batteries: Vec<usize> = ranked_diff
        .into_iter()
        .take(diff_cap)
        .map(|(b, _)| b)
        .collect();
    if relink_batteries.len() < 2 {
        return None;
    }

    let mut current_actions = source.to_vec();
    let mut current_flows = compute_action_flows(ca, flows_base, &current_actions);
    let mut current_score = source_score;
    let mut best_actions = current_actions.clone();
    let mut best_score = current_score;

    let mut touched = vec![false; num_b];
    let max_steps = relink_batteries.len().min(if num_b <= 24 {
        6
    } else if num_b <= 64 {
        8
    } else {
        10
    });
    let polish_every = if max_steps <= 4 { 2 } else { 3 };
    let mut weak_steps = 0usize;

    for step in 0..max_steps {
        let mut best_choice: Option<(usize, f64, f64, f64, f64)> = None;

        for &b in &relink_batteries {
            let desired = target[b];
            let cur = current_actions[b];
            let (lo, hi) = state.action_bounds[b];
            let scale = (hi - lo).abs().max(0.25);
            let dist0 = (desired - cur).abs() / scale;
            if dist0 <= 0.03 {
                continue;
            }

            let (u_min, u_max) = feasible_window(challenge, state, ca, hp, &current_flows, &current_actions, b);
            let proposed = desired.clamp(u_min, u_max);
            let dist1 = (desired - proposed).abs() / scale;
            let progress_norm = (dist0 - dist1).max(0.0);
            if progress_norm <= 1e-8 || (proposed - cur).abs() <= 1e-8 {
                continue;
            }

            let gain = eval_profit(challenge, state, ca, b, proposed)
                - eval_profit(challenge, state, ca, b, cur);
            let merit = relink_move_merit(
                challenge,
                ca,
                hp,
                &current_flows,
                b,
                proposed - cur,
                progress_norm,
                gain,
            );

            match best_choice {
                None => best_choice = Some((b, proposed, gain, merit, progress_norm)),
                Some((_, _, best_gain, best_merit, best_progress)) => {
                    if merit > best_merit + 1e-12
                        || ((merit - best_merit).abs() <= 1e-12
                            && (progress_norm > best_progress + 1e-12
                                || ((progress_norm - best_progress).abs() <= 1e-12
                                    && gain > best_gain + 1e-12)))
                    {
                        best_choice = Some((b, proposed, gain, merit, progress_norm));
                    }
                }
            }
        }

        let Some((best_b, best_u, gain, merit, progress_norm)) = best_choice else { break; };
        if merit <= -0.02 && (weak_steps > 0 || progress_norm < 0.12) {
            break;
        }

        let delta = best_u - current_actions[best_b];
        current_actions[best_b] = best_u;
        touched[best_b] = true;
        current_score += gain;
        for &(l, p) in &ca.b_to_lines[best_b] {
            if l < current_flows.len() {
                current_flows[l] += p * delta;
            }
        }

        if gain < -0.05 {
            weak_steps += 1;
        } else {
            weak_steps = 0;
        }

        if current_score > best_score + 1e-7 {
            best_score = current_score;
            best_actions = current_actions.clone();
        }

        let should_polish = step + 1 == max_steps || (step + 1) % polish_every == 0;
        if should_polish {
            let subset: Vec<usize> = relink_batteries
                .iter()
                .copied()
                .filter(|&bb| touched[bb] || (target[bb] - current_actions[bb]).abs() > 1e-8)
                .collect();
            if subset.len() >= 2 {
                let mut polished = current_actions.clone();
                let sweeps = if subset.len() <= 8 { 2 } else { 1 };
                run_asca_candidates(challenge, state, ca, hp, flows_base, &mut polished, &subset, sweeps);
                let polished_score = approx_actions_value(challenge, state, ca, hp, flows_base, &polished);
                if polished_score.is_finite() && polished_score > current_score + 1e-7 {
                    current_actions = polished;
                    current_score = polished_score;
                    current_flows = compute_action_flows(ca, flows_base, &current_actions);

                    if current_score > best_score + 1e-7 {
                        best_score = current_score;
                        best_actions = current_actions.clone();
                    }
                }
            }
        }
    }

    if best_score > source_score + 1e-7 {
        Some((best_actions, best_score))
    } else {
        None
    }
}

fn run_elite_merge_rebuild(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    elite_a: &[f64],
    elite_b: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b == 0 || elite_a.len() != num_b || elite_b.len() != num_b {
        return None;
    }
    if !elite_actions_are_distinct(challenge, state, elite_a, elite_b) {
        return None;
    }

    let score_a = approx_actions_value(challenge, state, ca, hp, flows_base, elite_a);
    let score_b = approx_actions_value(challenge, state, ca, hp, flows_base, elite_b);
    if !score_a.is_finite() || !score_b.is_finite() {
        return None;
    }

    let mut merged = vec![0.0_f64; num_b];
    let mut candidate_mask = vec![false; num_b];

    for b in 0..num_b {
        let ua = elite_a[b];
        let ub = elite_b[b];
        let v0 = eval_profit(challenge, state, ca, b, 0.0);
        let va = eval_profit(challenge, state, ca, b, ua);
        let vb = eval_profit(challenge, state, ca, b, ub);

        let mut footprint = 1e-6_f64;
        for &(l, p) in &ca.b_to_lines[b] {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let util = (flows_base[l].abs() / limit).min(2.0);
            footprint += p.abs() * (0.12 + util);
        }

        let merit_a = (va - v0).max(0.0) / footprint + 0.03 * ua.abs();
        let merit_b = (vb - v0).max(0.0) / footprint + 0.03 * ub.abs();
        let picked = if merit_a > merit_b + 1e-9 {
            ua
        } else if merit_b > merit_a + 1e-9 {
            ub
        } else if va > vb + 1e-9 {
            ua
        } else if vb > va + 1e-9 {
            ub
        } else if ua.abs() >= ub.abs() {
            ua
        } else {
            ub
        };

        if (ua - ub).abs() > 1e-7 {
            candidate_mask[b] = true;
        }
        if picked.abs() > 1e-9 || v0 < va.max(vb) - 1e-9 {
            merged[b] = picked;
        }
        if merged[b].abs() > 1e-7 {
            candidate_mask[b] = true;
        }
    }

    scale_actions_toward_zero_to_feasible(challenge, state, ca, hp, flows_base, &mut merged);

    let flows = compute_action_flows(ca, flows_base, &merged);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return None;
    }

    let mut max_util = 0.0_f64;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        max_util = max_util.max((flows[l].abs() / limit).min(2.0));
    }

    let stress_cut = if max_util > 0.80 {
        0.62
    } else if max_util > 0.55 {
        0.50
    } else {
        0.38
    };
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= stress_cut {
            for &(b, _) in &ca.ptdf_sparse[l] {
                candidate_mask[b] = true;
            }
        }
    }

    let target = if num_b <= 24 {
        num_b
    } else if num_b <= 56 {
        18
    } else {
        24
    }
    .min(num_b);
    let cap = if num_b <= 24 {
        num_b
    } else if num_b <= 56 {
        24
    } else {
        32
    }
    .min(num_b);

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| candidate_mask[b]).collect();

    if candidates.len() < target {
        let mut ranked: Vec<(usize, f64)> = (0..num_b)
            .filter(|&b| !candidate_mask[b])
            .map(|b| {
                let (lo, hi) = state.action_bounds[b];
                let scale = (hi - lo).abs().max(0.25);
                let diff_norm = (elite_a[b] - elite_b[b]).abs() / scale;

                let mut touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    touch += p.abs() * (0.15 + util);
                }

                let score = diff_norm
                    + 0.18 * potential(challenge, state, ca, b)
                    + 0.12 * touch
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (b, _) in ranked.into_iter().take(target.saturating_sub(candidates.len())) {
            candidates.push(b);
            candidate_mask[b] = true;
        }
    }

    if candidates.len() > cap {
        let mut ranked: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&b| {
                let (lo, hi) = state.action_bounds[b];
                let scale = (hi - lo).abs().max(0.25);
                let diff_norm = (elite_a[b] - elite_b[b]).abs() / scale;

                let mut touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    touch += p.abs() * (0.12 + util);
                }

                let score = merged[b].abs()
                    + diff_norm
                    + 0.12 * potential(challenge, state, ca, b)
                    + 0.10 * touch;
                (b, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().take(cap).map(|(b, _)| b).collect();
    }

    if !candidates.is_empty() {
        let sweeps = if candidates.len() <= 16 { 2 } else { 1 };
        run_asca_candidates(
            challenge,
            state,
            ca,
            hp,
            flows_base,
            &mut merged,
            &candidates,
            sweeps,
        );
        run_deflator(challenge, state, ca, hp, flows_base, &mut merged);

        if candidates.len() <= 18 || max_util < 0.35 {
            run_asca_candidates(
                challenge,
                state,
                ca,
                hp,
                flows_base,
                &mut merged,
                &candidates,
                1,
            );
            if num_l > 0 && max_util > 0.55 {
                run_deflator(challenge, state, ca, hp, flows_base, &mut merged);
            }
        }
    } else {
        run_deflator(challenge, state, ca, hp, flows_base, &mut merged);
    }

    let merged_score = approx_actions_value(challenge, state, ca, hp, flows_base, &merged);
    if merged_score.is_finite() && merged_score > score_a.max(score_b) + 1e-7 {
        Some((merged, merged_score))
    } else {
        None
    }
}

fn run_screened_multistart_policy(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
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

    if max_util_base >= 0.22 || num_b <= 32 {
        let slack_keep = if num_b <= 100 {
            keep
        } else {
            ((3 * keep + 3) / 5).min(keep).max(1)
        };
        let slack_seed = build_slack_balanced_seed(challenge, state, ca, hp, flows_base, slack_keep);
        if slack_seed.iter().any(|&u| u.abs() > 1e-8) {
            let score = approx_actions_value(challenge, state, ca, hp, flows_base, &slack_seed);
            seed_pool.push((slack_seed, score));
        }
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

    let mut refined_pool: Vec<(Vec<f64>, f64)> = vec![(best_actions.clone(), best_score)];

    for (idx, (mut seed, seed_score)) in seed_pool.into_iter().enumerate() {
        if idx >= refine_cap || !seed_score.is_finite() {
            break;
        }
        refine_basic_seed(challenge, state, ca, hp, flows_base, &mut seed);
        let score = approx_actions_value(challenge, state, ca, hp, flows_base, &seed);
        if score.is_finite() {
            refined_pool.push((seed.clone(), score));
            if score > best_score + 1e-7 {
                best_score = score;
                best_actions = seed;
            }
        }
    }

    refined_pool.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut elite_pool: Vec<(Vec<f64>, f64)> = Vec::new();
    for (actions, score) in refined_pool {
        if !score.is_finite() {
            continue;
        }
        let distinct = elite_pool
            .iter()
            .all(|(other, _)| elite_actions_are_distinct(challenge, state, &actions, other));
        if distinct {
            elite_pool.push((actions, score));
            if elite_pool.len() >= 3 {
                break;
            }
        }
    }

    if elite_pool.len() >= 2 && (num_b <= 64 || max_util_base > 0.55) {
        let mut merge_pairs = vec![(0usize, 1usize)];
        if elite_pool.len() >= 3 && (num_b <= 40 || max_util_base > 0.72) {
            merge_pairs.push((0usize, 2usize));
        }

        for (lhs_idx, rhs_idx) in merge_pairs {
            if lhs_idx >= elite_pool.len() || rhs_idx >= elite_pool.len() {
                continue;
            }

            if let Some((merged, score)) = run_elite_merge_rebuild(
                challenge,
                state,
                ca,
                hp,
                flows_base,
                &elite_pool[lhs_idx].0,
                &elite_pool[rhs_idx].0,
            ) {
                if score > best_score + 1e-7 {
                    best_score = score;
                    best_actions = merged;
                }
            }
        }
    }

    if elite_pool.len() >= 2 {
        let mut relink_pairs = vec![(1usize, 0usize)];
        if elite_pool.len() >= 3 {
            relink_pairs.push((2usize, 0usize));
        } else {
            relink_pairs.push((0usize, 1usize));
        }

        for (from_idx, to_idx) in relink_pairs {
            if from_idx >= elite_pool.len() || to_idx >= elite_pool.len() {
                continue;
            }

            if let Some((relinked, score)) = run_elite_path_relink(
                challenge,
                state,
                ca,
                hp,
                flows_base,
                &elite_pool[from_idx].0,
                &elite_pool[to_idx].0,
            ) {
                if score > best_score + 1e-7 {
                    best_score = score;
                    best_actions = relinked;
                }
            }
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
    ca: &aiioagaypCache,
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

fn pair_has_line_overlap(ca: &aiioagaypCache, a: usize, b: usize) -> bool {
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
    ca: &aiioagaypCache,
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
    ca: &aiioagaypCache,
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
    ca: &aiioagaypCache,
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
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    run_asca(challenge, state, ca, hp, flows_base, actions);
    run_deflator(challenge, state, ca, hp, flows_base, actions);
}

fn run_restricted_active_set_rebuild(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) -> bool {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b < 8 || num_l == 0 || num_b > 120 {
        return false;
    }

    let flows = compute_action_flows(ca, flows_base, actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return false;
    }

    let active_count = actions.iter().filter(|&&u| u.abs() > 1e-7).count();
    if active_count < 2 {
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
        if util > 0.32 {
            line_rank.push((l, util + 0.20 * (util - 0.75).max(0.0)));
        }
    }
    if line_rank.is_empty() || (max_util < 0.32 && active_count < 10) {
        return false;
    }

    let keep_lines = if num_b <= 24 {
        2
    } else if num_b <= 56 {
        3
    } else {
        4
    };
    let keep_lines = keep_lines.min(line_rank.len());
    if line_rank.len() > keep_lines {
        line_rank.select_nth_unstable_by(keep_lines - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        line_rank.truncate(keep_lines);
    }
    line_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let focus_lines: Vec<usize> = line_rank.iter().map(|(l, _)| *l).collect();
    if focus_lines.is_empty() {
        return false;
    }

    let mut focus_mask = vec![false; num_l];
    for &l in &focus_lines {
        focus_mask[l] = true;
    }

    let mut candidate_mask = vec![false; num_b];
    for b in 0..num_b {
        if actions[b].abs() > 1e-7 {
            candidate_mask[b] = true;
        }
    }
    for &l in &focus_lines {
        for &(b, _) in &ca.ptdf_sparse[l] {
            candidate_mask[b] = true;
        }
    }

    let stress_cut = if max_util > 0.78 {
        0.55
    } else if max_util > 0.50 {
        0.42
    } else {
        0.35
    };
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= stress_cut {
            for &(b, _) in &ca.ptdf_sparse[l] {
                candidate_mask[b] = true;
            }
        }
    }

    let target = if num_b <= 24 {
        num_b
    } else if num_b <= 56 {
        18
    } else {
        24
    }
    .min(num_b);
    let cap = if num_b <= 24 {
        num_b
    } else if num_b <= 56 {
        24
    } else {
        32
    }
    .min(num_b);

    let mut candidates: Vec<usize> = (0..num_b).filter(|&b| candidate_mask[b]).collect();

    if candidates.len() < target {
        let need = target - candidates.len();
        let mut ranked: Vec<(usize, f64)> = (0..num_b)
            .filter(|&b| !candidate_mask[b])
            .map(|b| {
                let mut footprint = 1e-6_f64;
                let mut touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit <= 1e-6 {
                        continue;
                    }
                    let util = (flows[l].abs() / limit).min(2.0);
                    footprint += p.abs() * (0.20 + util * util);
                    touch += p.abs() * (0.15 * util + if focus_mask[l] { 1.0 } else { 0.0 });
                }

                let score = potential(challenge, state, ca, b) * (1.0 + 0.08 * touch) / footprint
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        if ranked.len() > need {
            ranked.select_nth_unstable_by(need - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(need);
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (b, _) in ranked {
            candidates.push(b);
            candidate_mask[b] = true;
        }
    }

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
                    touch += p.abs() * (0.12 + util + if focus_mask[l] { 0.5 } else { 0.0 });
                }

                let mut score = 0.18 * potential(challenge, state, ca, b) + 0.10 * touch;
                if actions[b].abs() > 1e-7 {
                    score += 1.0 + actions[b].abs();
                }
                (b, score)
            })
            .collect();
        if ranked.len() > cap {
            ranked.select_nth_unstable_by(cap - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(cap);
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates = ranked.into_iter().map(|(b, _)| b).collect();
    }

    if candidates.len() < 2 {
        return false;
    }

    let mut ranked_drop: Vec<(usize, f64)> = candidates
        .iter()
        .filter_map(|&b| {
            if actions[b].abs() <= 1e-7 {
                return None;
            }

            let keep_val = eval_profit(challenge, state, ca, b, actions[b]);
            let zero_val = eval_profit(challenge, state, ca, b, 0.0);
            let retained_value = (keep_val - zero_val).max(0.0);

            let mut worsen = 0.0_f64;
            let mut relief = 0.0_f64;
            let mut touch = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }

                let util = (flows[l].abs() / limit).min(2.0);
                let weight = if focus_mask[l] {
                    0.55 + 1.20 * util
                } else if util > 0.45 {
                    0.10 + (util - 0.45)
                } else {
                    0.0
                };
                if weight <= 1e-9 {
                    continue;
                }

                let signed = p * actions[b];
                touch += p.abs() * (0.10 + weight);
                if flows[l] * signed > 1e-9 {
                    worsen += weight * signed.abs();
                } else if flows[l] * signed < -1e-9 {
                    relief += weight * signed.abs();
                }
            }

            if worsen <= 1e-9 && max_util < 0.72 {
                return None;
            }

            let relief_ratio = (relief / worsen.max(1e-9)).min(2.0);
            let score = (worsen + 0.12 * touch) / (0.05 + retained_value)
                * (1.0 - 0.35 * relief_ratio).max(0.35)
                + 0.015 * actions[b].abs();

            if score > 1e-8 {
                Some((b, score))
            } else {
                None
            }
        })
        .collect();
    if ranked_drop.is_empty() {
        return false;
    }

    let base_score = approx_actions_value(challenge, state, ca, hp, flows_base, actions);
    if !base_score.is_finite() {
        return false;
    }

    let sweeps = if num_b <= 20 || candidates.len() <= 16 { 2 } else { 1 };
    let mut drop_counts = vec![1usize];
    if ranked_drop.len() >= 2 {
        drop_counts.push(2usize);
    }
    if ranked_drop.len() >= 3 && (max_util > 0.78 || active_count > 12) {
        drop_counts.push(3usize);
    }

    let keep_drop = *drop_counts.last().unwrap();
    if ranked_drop.len() > keep_drop {
        ranked_drop.select_nth_unstable_by(keep_drop - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked_drop.truncate(keep_drop);
    }
    ranked_drop.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_score = base_score;
    let mut best_actions = actions.to_vec();

    for drop_count in drop_counts {
        let mut alt_actions = actions.to_vec();
        let mut changed = false;
        for &(b, _) in ranked_drop.iter().take(drop_count) {
            if alt_actions[b].abs() > 1e-9 {
                alt_actions[b] = 0.0;
                changed = true;
            }
        }
        if !changed {
            continue;
        }

        run_asca_candidates(
            challenge,
            state,
            ca,
            hp,
            flows_base,
            &mut alt_actions,
            &candidates,
            sweeps,
        );
        run_deflator(challenge, state, ca, hp, flows_base, &mut alt_actions);

        if num_b <= 24 || candidates.len() <= 18 || max_util < 0.40 {
            run_asca_candidates(
                challenge,
                state,
                ca,
                hp,
                flows_base,
                &mut alt_actions,
                &candidates,
                1,
            );
            if max_util > 0.45 {
                run_deflator(challenge, state, ca, hp, flows_base, &mut alt_actions);
            }
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

fn select_small_reset_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
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
    ca: &aiioagaypCache,
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

fn congestion_first_refine_trigger(
    challenge: &Challenge,
    hp: &TrackHp,
    flows_base: &[f64],
) -> bool {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    if num_b < 4 || num_l == 0 {
        return false;
    }

    let mut max_util = 0.0_f64;
    let mut stressed_lines = 0usize;
    let mut near_tight_lines = 0usize;

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows_base[l].abs() / limit).min(2.0);
        if util > max_util {
            max_util = util;
        }
        if util >= 0.65 {
            stressed_lines += 1;
        }
        if util >= 0.82 {
            near_tight_lines += 1;
        }
    }

    if num_b <= 12 {
        max_util > 0.90 || (max_util > 0.78 && near_tight_lines >= 1)
    } else if num_b <= 40 {
        max_util > 0.84 || near_tight_lines >= 2 || (max_util > 0.74 && stressed_lines >= 3)
    } else {
        max_util > 0.88 || near_tight_lines >= 3 || (max_util > 0.78 && stressed_lines >= 4)
    }
}

fn run_congestion_first_rebuild(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) -> bool {
    let num_l = challenge.network.flow_limits.len();
    if challenge.num_batteries == 0 || num_l == 0 {
        return false;
    }

    let flows = compute_action_flows(ca, flows_base, actions);
    let feasible = (0..num_l).all(|l| {
        flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
    });
    if !feasible {
        return false;
    }

    let candidates = select_small_reset_candidates(challenge, state, ca, hp, &flows, actions);
    if candidates.is_empty() {
        return false;
    }

    let base_score = approx_actions_value(challenge, state, ca, hp, flows_base, actions);
    if !base_score.is_finite() {
        return false;
    }

    let sweeps = if challenge.num_batteries <= 12 || candidates.len() <= 8 {
        hp.asca_iters.min(2).max(1)
    } else {
        1
    };

    let mut rebuild_actions = actions.to_vec();
    for &b in &candidates {
        rebuild_actions[b] = 0.0;
    }
    run_asca_candidates(
        challenge,
        state,
        ca,
        hp,
        flows_base,
        &mut rebuild_actions,
        &candidates,
        sweeps,
    );

    let rebuild_score = approx_actions_value(challenge, state, ca, hp, flows_base, &rebuild_actions);
    if rebuild_score > base_score + 1e-7 {
        actions.copy_from_slice(&rebuild_actions);
        true
    } else {
        false
    }
}

fn refine_small_seed(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    if congestion_first_refine_trigger(challenge, hp, flows_base) {
        run_deflator(challenge, state, ca, hp, flows_base, actions);
        run_congestion_first_rebuild(challenge, state, ca, hp, flows_base, actions);
        run_asca(challenge, state, ca, hp, flows_base, actions);
        run_deflator(challenge, state, ca, hp, flows_base, actions);

        if run_pair_exchange_polish(challenge, state, ca, hp, flows_base, actions) {
            run_deflator(challenge, state, ca, hp, flows_base, actions);
        }

        run_small_destroy_repair_polish(challenge, state, ca, hp, flows_base, actions);
        return;
    }

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
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
) -> Vec<f64> {
    run_screened_multistart_policy(challenge, state, ca, hp, flows_base, 2)
}

fn run_small_multistart_policy(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
) -> Vec<f64> {
    let num_b = challenge.num_batteries;

    let mut max_util_base = 0.0_f64;
    let mut medium_tight_lines = 0usize;
    for l in 0..challenge.network.flow_limits.len() {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows_base[l].abs() / limit).min(2.0);
        max_util_base = max_util_base.max(util);
        if util >= 0.40 {
            medium_tight_lines += 1;
        }
    }

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

    let use_dual_seed = if challenge.network.flow_limits.is_empty() {
        false
    } else if num_b <= 48 {
        max_util_base >= 0.22 || medium_tight_lines >= 2
    } else {
        max_util_base >= 0.38 || medium_tight_lines >= 4
    };

    if use_dual_seed {
        let dual_keep = if num_b <= 18 {
            num_b
        } else if max_util_base > 0.78 || medium_tight_lines >= 6 {
            ((4 * num_b + 4) / 5).min(num_b)
        } else if num_b <= 40 {
            ((3 * num_b + 3) / 4).min(num_b)
        } else {
            ((num_b + 1) / 2).min(num_b)
        };

        let mut dual_actions = build_dual_price_seed(challenge, state, ca, hp, flows_base, dual_keep);
        if dual_actions.iter().any(|&u| u.abs() > 1e-8) {
            if num_b <= 28 {
                refine_small_seed(challenge, state, ca, hp, flows_base, &mut dual_actions);
            } else {
                refine_basic_seed(challenge, state, ca, hp, flows_base, &mut dual_actions);
            }

            let dual_score = approx_actions_value(challenge, state, ca, hp, flows_base, &dual_actions);
            if dual_score > best_score + 1e-7 {
                best_score = dual_score;
                best_actions = dual_actions;
            }
        }
    }

    let use_slack_seed = if challenge.network.flow_limits.is_empty() {
        false
    } else if num_b <= 40 {
        max_util_base >= 0.18 || num_b <= 18
    } else {
        max_util_base >= 0.40
    };

    if use_slack_seed {
        let slack_keep = if num_b <= 18 {
            num_b
        } else if max_util_base > 0.72 {
            ((4 * num_b + 4) / 5).min(num_b)
        } else if num_b <= 40 {
            ((3 * num_b + 3) / 4).min(num_b)
        } else {
            ((num_b + 1) / 2).min(num_b)
        };

        let mut slack_actions = build_slack_balanced_seed(challenge, state, ca, hp, flows_base, slack_keep);
        if slack_actions.iter().any(|&u| u.abs() > 1e-8) {
            if num_b <= 28 {
                refine_small_seed(challenge, state, ca, hp, flows_base, &mut slack_actions);
            } else {
                refine_basic_seed(challenge, state, ca, hp, flows_base, &mut slack_actions);
            }

            let slack_score = approx_actions_value(challenge, state, ca, hp, flows_base, &slack_actions);
            if slack_score > best_score + 1e-7 {
                best_score = slack_score;
                best_actions = slack_actions;
            }
        }
    }

    let use_relief_seed = if challenge.network.flow_limits.is_empty() {
        false
    } else if num_b <= 48 {
        max_util_base >= 0.24 || num_b <= 14
    } else {
        max_util_base >= 0.55
    };

    if use_relief_seed {
        let relief_keep = if num_b <= 16 {
            num_b
        } else if max_util_base > 0.72 {
            ((4 * num_b + 4) / 5).min(num_b)
        } else if num_b <= 40 {
            num_b
        } else {
            ((2 * num_b + 2) / 3).min(num_b)
        };

        let mut relief_actions = build_line_relief_seed(challenge, state, ca, hp, flows_base, relief_keep);
        if relief_actions.iter().any(|&u| u.abs() > 1e-8) {
            if num_b <= 32 {
                refine_small_seed(challenge, state, ca, hp, flows_base, &mut relief_actions);
            } else {
                refine_basic_seed(challenge, state, ca, hp, flows_base, &mut relief_actions);
            }

            let relief_score = approx_actions_value(challenge, state, ca, hp, flows_base, &relief_actions);
            if relief_score > best_score + 1e-7 {
                best_actions = relief_actions;
            }
        }
    }

    best_actions
}

fn collect_future_exogenous_flows(
    challenge: &Challenge,
    hp: &TrackHp,
) -> (Vec<Vec<f64>>, f64, usize) {
    let num_t = challenge.num_steps;
    let num_l = challenge.network.flow_limits.len();
    let mut future_flows = Vec::with_capacity(num_t);
    let mut max_util = 0.0_f64;
    let mut scarce_events = 0usize;

    for t in 0..num_t {
        let flows = challenge.network.compute_flows(&challenge.exogenous_injections[t]);
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let util = (flows[l].abs() / limit).min(3.0);
            let slack_frac = ((limit - flows[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
            if util > max_util {
                max_util = util;
            }
            if util >= hp.lmp_threshold.max(0.45) || (util >= 0.55 && slack_frac < 0.20) {
                scarce_events += 1;
            }
        }
        future_flows.push(flows);
    }

    (future_flows, max_util, scarce_events)
}

fn build_legacy_future_premiums_from_flows(
    challenge: &Challenge,
    hp: &TrackHp,
    ptdf_sparse: &[Vec<(usize, f64)>],
    future_flows: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    let num_t = future_flows.len();
    let mut sell = vec![vec![0.0_f64; num_b]; num_t];
    let mut buy = vec![vec![0.0_f64; num_b]; num_t];

    let base_premium = 20.0 * hp.lmp_premium_scale;
    for t in 0..num_t {
        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let ratio = future_flows[t][l].abs() / limit;
            if ratio > hp.lmp_threshold {
                let proba = ((ratio - hp.lmp_threshold) / (1.0 - hp.lmp_threshold).max(1e-6))
                    .clamp(0.0, 1.0);
                let premium = base_premium * proba;
                let sign_f = future_flows[t][l].signum();
                for &(b, impact) in &ptdf_sparse[l] {
                    if impact.abs() > 1e-6 {
                        let nodal_shift = -impact * sign_f * premium;
                        sell[t][b] += nodal_shift;
                        buy[t][b] += nodal_shift;
                    }
                }
            }
        }
    }

    (sell, buy)
}

fn future_line_shadow_price(limit: f64, flow_abs: f64, threshold: f64, base_premium: f64) -> f64 {
    if limit <= 1e-6 || base_premium <= 1e-9 {
        return 0.0;
    }

    let util = (flow_abs / limit).min(3.0);
    let slack_frac = ((limit - flow_abs).max(0.0) / limit).clamp(0.0, 1.0);
    if util < (threshold * 0.75).max(0.20) && slack_frac > 0.35 {
        return 0.0;
    }

    let near = (util - threshold * 0.85).max(0.0);
    let tight = (util - 0.78).max(0.0);
    let overflow = (util - 1.0).max(0.0);
    let slack_curve = (1.0 / (slack_frac + 0.10) - 1.0 / 1.10).max(0.0);

    let scarcity = 0.10 * near * near
        + 0.55 * tight * tight
        + 1.80 * overflow * overflow
        + 0.018 * slack_curve * slack_curve;

    (base_premium * scarcity).clamp(0.0, 1.5 * base_premium)
}

fn build_shadow_future_premiums_from_flows(
    challenge: &Challenge,
    hp: &TrackHp,
    ptdf_sparse: &[Vec<(usize, f64)>],
    future_flows: &[Vec<f64>],
    max_future_util: f64,
    scarce_events: usize,
) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let num_b = challenge.num_batteries;
    let num_l = challenge.network.flow_limits.len();
    let num_t = future_flows.len();

    if num_l <= 3 {
        return None;
    }
    let trigger_events = num_t.min(6).max(2);
    if max_future_util < hp.lmp_threshold.max(0.22) * 0.92 && scarce_events < trigger_events {
        return None;
    }

    let base_premium = 20.0 * hp.lmp_premium_scale;
    let mut sell = vec![vec![0.0_f64; num_b]; num_t];
    let mut buy = vec![vec![0.0_f64; num_b]; num_t];

    for t in 0..num_t {
        let mut line_shadow = vec![0.0_f64; num_l];
        let mut active_lines = 0usize;
        let mut total_shadow = 0.0_f64;

        for l in 0..num_l {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let flow_abs = future_flows[t][l].abs();
            let shadow = future_line_shadow_price(limit, flow_abs, hp.lmp_threshold, base_premium);
            if shadow > 1e-8 && flow_abs > 1e-9 {
                line_shadow[l] = shadow;
                active_lines += 1;
                total_shadow += shadow;
            }
        }

        if active_lines == 0 {
            continue;
        }

        let active_line_scale = if active_lines <= 1 {
            1.0
        } else {
            (1.0 / (1.0 + 0.16 * ((active_lines - 1) as f64))).max(0.55)
        };
        let mass_scale = if total_shadow > 1e-9 {
            (base_premium / total_shadow.max(base_premium.max(1e-9)))
                .sqrt()
                .clamp(0.60, 1.0)
        } else {
            1.0
        };
        let time_scale = active_line_scale * mass_scale;

        for l in 0..num_l {
            let shadow = line_shadow[l];
            if shadow <= 1e-8 {
                continue;
            }

            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let flow = future_flows[t][l];
            if flow.abs() <= 1e-9 {
                continue;
            }

            let util = (flow.abs() / limit).min(3.0);
            let sign_f = flow.signum();
            let sell_harm_scale = 0.85 + 0.35 * (util - 0.70).max(0.0);
            let sell_relief_scale = 0.35 + 0.20 * util;
            let buy_harm_scale = 0.65 + 0.45 * (util - 0.55).max(0.0);
            let buy_relief_scale = 0.25 + 0.15 * util;

            for &(b, impact) in &ptdf_sparse[l] {
                if impact.abs() <= 1e-6 {
                    continue;
                }

                let signed = sign_f * impact;
                if signed > 0.0 {
                    sell[t][b] -= time_scale * shadow * sell_harm_scale * signed;
                    buy[t][b] -= time_scale * shadow * buy_relief_scale * signed;
                } else if signed < 0.0 {
                    let relief = -signed;
                    sell[t][b] += time_scale * shadow * sell_relief_scale * relief;
                    buy[t][b] += time_scale * shadow * buy_harm_scale * relief;
                }
            }
        }

        let cap = 1.35 * base_premium;
        for b in 0..num_b {
            sell[t][b] = sell[t][b].clamp(-cap, cap);
            buy[t][b] = buy[t][b].clamp(-cap, cap);
        }
    }

    Some((sell, buy))
}

fn build_cache(challenge: &Challenge, state: &State, hp: &TrackHp) -> aiioagaypCache {
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

    let (expected_sell_premiums, expected_buy_premiums) = if hp.anticipate_lmp && num_l > 0 {
        let (future_flows, max_future_util, scarce_events) =
            collect_future_exogenous_flows(challenge, hp);
        build_shadow_future_premiums_from_flows(
            challenge,
            hp,
            &ptdf_sparse,
            &future_flows,
            max_future_util,
            scarce_events,
        )
        .unwrap_or_else(|| {
            build_legacy_future_premiums_from_flows(
                challenge,
                hp,
                &ptdf_sparse,
                &future_flows,
            )
        })
    } else {
        (
            vec![vec![0.0_f64; num_b]; num_t],
            vec![vec![0.0_f64; num_b]; num_t],
        )
    };

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
            let sell_extra = expected_sell_premiums[t][b];
            let buy_extra = expected_buy_premiums[t][b];
            let p_sell = p_da * (1.0 + hp.jump_premium) + sell_extra;
            let p_buy = p_da + buy_extra;

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

    aiioagaypCache { dp, ptdf_sparse, b_to_lines, batt_nodes }
}

#[inline]
fn eval_profit(challenge: &Challenge, state: &State, ca: &aiioagaypCache, b: usize, u: f64) -> f64 {
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
fn potential(challenge: &Challenge, state: &State, ca: &aiioagaypCache, b: usize) -> f64 {
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

fn compute_action_flows(ca: &aiioagaypCache, flows_base: &[f64], actions: &[f64]) -> Vec<f64> {
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
    ca: &aiioagaypCache,
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

fn build_single_extra_probes(
    challenge: &Challenge,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
    u_min: f64,
    u_max: f64,
) -> Vec<f64> {
    let cur = actions[b];
    if (u_max - u_min).abs() <= 1e-8 {
        return Vec::new();
    }

    let mut points = Vec::with_capacity(4);
    let mut max_util = 0.0_f64;
    let mut lo_bias = 0.0_f64;
    let mut hi_bias = 0.0_f64;

    for &(l, p) in &ca.b_to_lines[b] {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        let util = (flows[l].abs() / limit).min(2.0);
        if util > max_util {
            max_util = util;
        }
        if util >= 0.65 {
            if flows[l] * p > 0.0 {
                lo_bias += util;
            } else if flows[l] * p < 0.0 {
                hi_bias += util;
            }
        }
    }

    if max_util > 0.40 {
        if u_min < cur - 1e-7 {
            push_unique_probe(&mut points, 0.5 * (u_min + cur));
        }
        if cur + 1e-7 < u_max {
            push_unique_probe(&mut points, 0.5 * (u_max + cur));
        }
    }

    if lo_bias > hi_bias + 0.05 && u_min < cur - 1e-7 {
        push_unique_probe(&mut points, (2.0 * cur + u_min) / 3.0);
    } else if hi_bias > lo_bias + 0.05 && cur + 1e-7 < u_max {
        push_unique_probe(&mut points, (2.0 * cur + u_max) / 3.0);
    } else if max_util > 0.90 {
        push_unique_probe(&mut points, 0.5 * (u_min + u_max));
    }

    points
}

fn best_action_in_window(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    b: usize,
) -> (f64, f64) {
    let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);

    let cur = actions[b];
    let mut best_u = cur;
    let mut best_v = eval_profit(challenge, state, ca, b, cur);

    let v_min = eval_profit(challenge, state, ca, b, u_min);
    if v_min > best_v {
        best_v = v_min;
        best_u = u_min;
    }

    let has_span = (u_max - u_min).abs() > 1e-12;
    if has_span {
        let v_max = eval_profit(challenge, state, ca, b, u_max);
        if v_max > best_v {
            best_v = v_max;
            best_u = u_max;
        }
    }

    let zero_in = u_min <= 0.0 && 0.0 <= u_max;
    if zero_in {
        let v0 = eval_profit(challenge, state, ca, b, 0.0);
        if v0 > best_v {
            best_v = v0;
            best_u = 0.0;
        }
    }

    if (u_max - u_min).abs() > 1e-8 {
        let mut max_util = 0.0_f64;
        let mut lo_bias = 0.0_f64;
        let mut hi_bias = 0.0_f64;

        for &(l, p) in &ca.b_to_lines[b] {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let util = (flows[l].abs() / limit).min(2.0);
            if util > max_util {
                max_util = util;
            }
            if util >= 0.65 {
                if flows[l] * p > 0.0 {
                    lo_bias += util;
                } else if flows[l] * p < 0.0 {
                    hi_bias += util;
                }
            }
        }

        let mut extras = [0.0_f64; 3];
        let mut extra_len = 0usize;

        if max_util > 0.40 {
            if u_min < cur - 1e-7 {
                let u = 0.5 * (u_min + cur);
                let mut unique = true;
                for i in 0..extra_len {
                    if (extras[i] - u).abs() <= 1e-7 {
                        unique = false;
                        break;
                    }
                }
                if unique {
                    extras[extra_len] = u;
                    extra_len += 1;
                }
            }
            if cur + 1e-7 < u_max {
                let u = 0.5 * (u_max + cur);
                let mut unique = true;
                for i in 0..extra_len {
                    if (extras[i] - u).abs() <= 1e-7 {
                        unique = false;
                        break;
                    }
                }
                if unique {
                    extras[extra_len] = u;
                    extra_len += 1;
                }
            }
        }

        if lo_bias > hi_bias + 0.05 && u_min < cur - 1e-7 {
            let u = (2.0 * cur + u_min) / 3.0;
            let mut unique = true;
            for i in 0..extra_len {
                if (extras[i] - u).abs() <= 1e-7 {
                    unique = false;
                    break;
                }
            }
            if unique {
                extras[extra_len] = u;
                extra_len += 1;
            }
        } else if hi_bias > lo_bias + 0.05 && cur + 1e-7 < u_max {
            let u = (2.0 * cur + u_max) / 3.0;
            let mut unique = true;
            for i in 0..extra_len {
                if (extras[i] - u).abs() <= 1e-7 {
                    unique = false;
                    break;
                }
            }
            if unique {
                extras[extra_len] = u;
                extra_len += 1;
            }
        } else if max_util > 0.90 {
            let u = 0.5 * (u_min + u_max);
            let mut unique = true;
            for i in 0..extra_len {
                if (extras[i] - u).abs() <= 1e-7 {
                    unique = false;
                    break;
                }
            }
            if unique {
                extras[extra_len] = u;
                extra_len += 1;
            }
        }

        for i in 0..extra_len {
            let u = extras[i];
            if (u - cur).abs() <= 1e-7
                || (u - u_min).abs() <= 1e-7
                || (u - u_max).abs() <= 1e-7
                || (zero_in && u.abs() <= 1e-7)
            {
                continue;
            }

            let v = eval_profit(challenge, state, ca, b, u);
            if v > best_v {
                best_v = v;
                best_u = u;
            }
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

fn base_dynamic_order_score(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    b: usize,
    u_min: f64,
    u_max: f64,
    v_cur: f64,
) -> f64 {
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
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        let util = (flows[l].abs() / limit).min(2.0);
        footprint += p.abs() * (0.5 + 8.0 * util * util);
        if util > 0.85 {
            stressed_touch += p.abs() * (util - 0.85);
        }
    }

    (upside + 1e-9) * (1.0 + 0.1 * stressed_touch) / footprint
}

fn dynamic_order(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    candidates: &[usize],
) -> Vec<usize> {
    let mut max_util_global = 0.0_f64;
    for l in 0..challenge.network.flow_limits.len() {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        max_util_global = max_util_global.max((flows[l].abs() / limit).min(2.0));
    }
    let use_shadow_order = max_util_global >= 0.42 && !challenge.network.flow_limits.is_empty();

    let mut ranked: Vec<(usize, f64)> = candidates.iter().map(|&b| {
        let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);
        let cur = actions[b].clamp(u_min, u_max);
        let v_cur = eval_profit(challenge, state, ca, b, cur);

        let score = if !use_shadow_order || (u_max - u_min).abs() <= 1e-6 {
            base_dynamic_order_score(challenge, state, ca, hp, flows, b, u_min, u_max, v_cur)
        } else {
            let span = (u_max - u_min).max(1e-9);
            let hi_room = (u_max - cur).max(0.0);
            let lo_room = (cur - u_min).max(0.0);
            let probe_frac = if max_util_global > 0.92 {
                0.22
            } else if max_util_global > 0.82 {
                0.28
            } else {
                0.35
            };

            let mut footprint = 1e-4_f64;
            let mut pos_relief = 0.0_f64;
            let mut pos_worsen = 0.0_f64;
            let mut neg_relief = 0.0_f64;
            let mut neg_worsen = 0.0_f64;
            let mut neutral_touch = 0.0_f64;

            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 { continue; }

                let util = (flows[l].abs() / limit).min(2.0);
                let abs_p = p.abs();
                footprint += abs_p * (0.30 + 3.5 * util * util);

                let shadow = if util > 0.97 {
                    2.0 + 18.0 * (util - 0.97)
                } else if util > 0.88 {
                    0.65 + 6.0 * (util - 0.88)
                } else if util > 0.72 {
                    0.12 + 1.8 * (util - 0.72)
                } else {
                    0.0
                };
                if shadow <= 0.0 {
                    continue;
                }

                if flows[l].abs() <= 1e-9 {
                    neutral_touch += shadow * abs_p;
                    continue;
                }

                let signed = flows[l] * p;
                if signed > 0.0 {
                    pos_worsen += shadow * abs_p;
                    neg_relief += shadow * abs_p;
                } else if signed < 0.0 {
                    pos_relief += shadow * abs_p;
                    neg_worsen += shadow * abs_p;
                }
            }

            let mut slope_hi = 0.0_f64;
            if hi_room > 1e-6 {
                let probe_hi = cur + probe_frac * hi_room;
                let delta = probe_hi - cur;
                if delta > 1e-9 {
                    let v_hi = eval_profit(challenge, state, ca, b, probe_hi);
                    slope_hi = ((v_hi - v_cur) / delta).max(0.0);
                }
            }

            let mut slope_lo = 0.0_f64;
            if lo_room > 1e-6 {
                let probe_lo = cur - probe_frac * lo_room;
                let delta = cur - probe_lo;
                if delta > 1e-9 {
                    let v_lo = eval_profit(challenge, state, ca, b, probe_lo);
                    slope_lo = ((v_lo - v_cur) / delta).max(0.0);
                }
            }

            let raw_hi = slope_hi * (0.35 + 0.65 * (hi_room / span));
            let raw_lo = slope_lo * (0.35 + 0.65 * (lo_room / span));

            let shadow_hi = raw_hi
                * (1.0 + 0.45 * pos_relief + 0.05 * neutral_touch)
                / (1.0 + 0.85 * pos_worsen + 0.08 * footprint);
            let shadow_lo = raw_lo
                * (1.0 + 0.45 * neg_relief + 0.05 * neutral_touch)
                / (1.0 + 0.85 * neg_worsen + 0.08 * footprint);

            let shadow_score = shadow_hi.max(shadow_lo);
            let raw_score = raw_hi.max(raw_lo) / (1.0 + 0.05 * footprint);

            if shadow_score > 1e-10 || raw_score > 1e-10 {
                0.72 * shadow_score + 0.28 * raw_score + 0.01 * actions[b].abs()
            } else {
                base_dynamic_order_score(challenge, state, ca, hp, flows, b, u_min, u_max, v_cur)
            }
        };

        (b, score)
    }).collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.into_iter().map(|(b, _)| b).collect()
}

fn run_asca_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
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
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
) {
    let num_b = challenge.num_batteries;
    let mut candidates: Vec<usize> = (0..num_b).collect();

    if hp.prune_ratio > 0.0 && num_b >= 2 {
        let cutoff = ((num_b as f64) * hp.prune_ratio) as usize;
        if cutoff > 0 && cutoff < num_b {
            let keep_target = num_b - cutoff;

            let mut max_util_base = 0.0_f64;
            for l in 0..challenge.network.flow_limits.len() {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit <= 1e-6 {
                    continue;
                }
                max_util_base = max_util_base.max((flows_base[l].abs() / limit).min(2.0));
            }

            let stress_cut = if max_util_base > 0.90 {
                0.60
            } else if max_util_base > 0.72 {
                0.72
            } else {
                0.85
            };

            let reserve = if max_util_base > 0.90 {
                keep_target.min(if num_b <= 40 { 6 } else if num_b <= 80 { 10 } else { 14 })
            } else if max_util_base > 0.72 {
                keep_target.min(if num_b <= 40 { 4 } else if num_b <= 80 { 8 } else { 12 })
            } else {
                0
            };

            let mut protected = vec![false; num_b];
            if reserve > 0 {
                let mut stressed_ranked: Vec<(usize, f64)> = (0..num_b)
                    .filter_map(|b| {
                        let mut stressed_touch = 0.0_f64;
                        for &(l, p) in &ca.b_to_lines[b] {
                            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                            if limit <= 1e-6 {
                                continue;
                            }
                            let util = (flows_base[l].abs() / limit).min(2.0);
                            if util >= stress_cut {
                                stressed_touch += p.abs() * (0.25 + util * util);
                            }
                        }

                        if stressed_touch <= 1e-9 {
                            return None;
                        }

                        let score = stressed_touch * (1.0 + 0.2 * potential(challenge, state, ca, b))
                            + 1e-6 * challenge.batteries[b].capacity_mwh;
                        Some((b, score))
                    })
                    .collect();
                stressed_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (b, _) in stressed_ranked.into_iter().take(reserve) {
                    protected[b] = true;
                }
            }

            let mut ranked: Vec<(usize, f64)> = (0..num_b).map(|b| {
                let mut network_score = 0.0_f64;
                let mut stressed_touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit > 1e-6 {
                        let util = (flows_base[l].abs() / limit).min(2.0);
                        network_score += p.abs() * (1.0 + 4.0 * util * util);
                        if util >= stress_cut {
                            stressed_touch += p.abs() * util;
                        }
                    }
                }
                let mut score = potential(challenge, state, ca, b) * (1.0 + network_score.sqrt())
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                if protected[b] {
                    score += 0.5 + 0.2 * stressed_touch;
                }
                (b, score)
            }).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if reserve == 0 {
                candidates = ranked.into_iter().take(keep_target).map(|(b, _)| b).collect();
            } else {
                candidates = Vec::with_capacity(keep_target);
                let mut chosen = vec![false; num_b];

                for (b, _) in &ranked {
                    if protected[*b] {
                        candidates.push(*b);
                        chosen[*b] = true;
                        if candidates.len() >= reserve {
                            break;
                        }
                    }
                }

                for (b, _) in ranked {
                    if !chosen[b] {
                        candidates.push(b);
                        chosen[b] = true;
                        if candidates.len() >= keep_target {
                            break;
                        }
                    }
                }
            }
        }
    }

    run_asca_candidates(challenge, state, ca, hp, flows_base, actions, &candidates, hp.asca_iters);
}

#[derive(Clone, Debug)]
struct LineFeedback {
    relieved: Vec<f64>,
    tight: Vec<f64>,
}

#[inline]
fn feedback_relief_mass(feedback: &LineFeedback) -> f64 {
    feedback.relieved.iter().sum()
}

#[inline]
fn feedback_tight_mass(feedback: &LineFeedback) -> f64 {
    feedback.tight.iter().sum()
}

#[inline]
fn feedback_activity(feedback: &LineFeedback, l: usize) -> f64 {
    feedback.relieved[l] + feedback.tight[l]
}

fn compute_line_relief_signal(
    challenge: &Challenge,
    hp: &TrackHp,
    flows_before: &[f64],
    flows_after: &[f64],
) -> LineFeedback {
    let num_l = challenge.network.flow_limits.len();
    let mut relieved_ranked: Vec<(usize, f64)> = Vec::new();
    let mut tight_ranked: Vec<(usize, f64)> = Vec::new();

    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }

        let before = (flows_before[l].abs() / limit).min(3.0);
        let after = (flows_after[l].abs() / limit).min(3.0);
        let improvement = (before - after).max(0.0);
        let deterioration = (after - before).max(0.0);

        let mut relieved_score = 0.0_f64;
        if improvement > 1e-6 {
            if before > 1.0 {
                relieved_score = improvement + 0.75 * (before - 1.0);
            } else if before > 0.90 {
                relieved_score = 0.5 * improvement;
            } else if improvement > 0.10 {
                relieved_score = 0.25 * improvement;
            }
        }
        relieved_score = relieved_score.clamp(0.0, 2.5);
        if relieved_score > 1e-6 {
            relieved_ranked.push((l, relieved_score));
        }

        let mut tight_score = 0.0_f64;
        if after > 1.0 {
            tight_score = (after - 1.0)
                + 0.50 * (after - 0.90).max(0.0)
                + 0.30 * deterioration;
        } else if after > 0.94 {
            tight_score = 0.40 * (after - 0.94) / 0.06
                + 0.25 * deterioration
                + 0.10 * before.max(after);
        } else if after > 0.86 && (before > 1.0 || deterioration > 0.04) {
            tight_score = 0.18 * (after - 0.86) / 0.14
                + 0.18 * deterioration;
        }
        tight_score = tight_score.clamp(0.0, 2.5);
        if tight_score > 1e-6 {
            tight_ranked.push((l, tight_score));
        }
    }

    relieved_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    tight_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let keep_relief = if num_l <= 8 { relieved_ranked.len() } else { relieved_ranked.len().min(4) };
    let keep_tight = if num_l <= 8 { tight_ranked.len() } else { tight_ranked.len().min(4) };

    let mut relieved = vec![0.0_f64; num_l];
    let mut tight = vec![0.0_f64; num_l];

    for (l, score) in relieved_ranked.into_iter().take(keep_relief) {
        relieved[l] = score;
    }
    for (l, score) in tight_ranked.into_iter().take(keep_tight) {
        tight[l] = score;
    }

    LineFeedback { relieved, tight }
}

fn expansion_compatibility_score(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    feedback: &LineFeedback,
    b: usize,
) -> f64 {
    let (u_min, u_max) = feasible_window(challenge, state, ca, hp, flows, actions, b);
    let cur = actions[b].clamp(u_min, u_max);
    let total_span = (u_max - u_min).abs().max(1e-9);
    if total_span <= 1e-8 {
        return 0.0;
    }

    let mut touches_stress = false;
    for &(l, _) in &ca.b_to_lines[b] {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[l].abs() / limit).min(2.0);
        if util >= 0.58 || feedback_activity(feedback, l) > 1e-9 {
            touches_stress = true;
            break;
        }
    }
    if !touches_stress {
        return 0.0;
    }

    let v_cur = eval_profit(challenge, state, ca, b, cur);
    let mut best = 0.0_f64;

    for (room_raw, dir_sign) in [(u_max - cur, 1.0_f64), (cur - u_min, -1.0_f64)] {
        let room = room_raw.max(0.0);
        if room <= 1e-7 {
            continue;
        }

        let room_frac = (room / total_span).clamp(0.0, 1.0);
        let probe_frac = if room_frac > 0.60 {
            0.35
        } else if room_frac > 0.25 {
            0.50
        } else {
            0.70
        };
        let probe = cur + dir_sign * probe_frac * room;
        let gain = (eval_profit(challenge, state, ca, b, probe) - v_cur).max(0.0);

        let mut footprint = 1e-6_f64;
        let mut relief_bonus = 0.0_f64;
        let mut neutral_bonus = 0.0_f64;
        let mut harm_penalty = 0.0_f64;
        let mut room_gate = 1.0_f64;

        for &(l, p) in &ca.b_to_lines[b] {
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }
            let abs_p = p.abs();
            if abs_p <= 1e-10 {
                continue;
            }

            let util = (flows[l].abs() / limit).min(2.0);
            let slack_frac = ((limit - flows[l].abs()).max(0.0) / limit).clamp(0.0, 1.0);
            let relieved = feedback.relieved[l];
            let tight = feedback.tight[l];
            let unit_move = p * dir_sign;
            footprint += abs_p * (0.18 + util * util);

            if flows[l].abs() <= 1e-9 {
                neutral_bonus += abs_p * (relieved + 0.25 * tight);
                continue;
            }

            if flows[l] * unit_move < 0.0 {
                relief_bonus += abs_p * (0.18 + relieved + 0.55 * tight + 0.25 * (util - 0.60).max(0.0));
            } else if flows[l] * unit_move > 0.0 {
                harm_penalty += abs_p
                    * (0.18 + 0.35 * relieved + 1.10 * tight + 0.20 * (util - 0.55).max(0.0))
                    / (slack_frac + 0.08);
                let local_room = ((limit - flows[l].abs()).max(0.0) / abs_p).clamp(0.0, room);
                room_gate = room_gate.min((local_room / room.max(1e-9)).clamp(0.0, 1.0));
            } else if relieved > 0.0 || tight > 0.0 {
                neutral_bonus += 0.20 * abs_p * (relieved + 0.5 * tight);
            }
        }

        let relief_bonus = relief_bonus.min(7.0);
        let neutral_bonus = neutral_bonus.min(4.5);
        let harm_penalty = harm_penalty.min(9.0);
        let base = gain + 0.01 * room * (1.0 + 0.25 * relief_bonus + 0.08 * neutral_bonus);
        let score = base
            * (0.35 + 0.65 * room_frac)
            * (1.0 + 0.24 * relief_bonus + 0.06 * neutral_bonus)
            * (0.45 + 0.55 * room_gate)
            / (1.0 + 0.45 * harm_penalty + 0.06 * footprint);

        if score > best {
            best = score;
        }
    }

    best
}

fn select_post_deflator_candidates(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    seed_mask: &[bool],
    feedback: &LineFeedback,
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
        if feedback_activity(feedback, l) > 1e-9 {
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
                let compat = expansion_compatibility_score(
                    challenge,
                    state,
                    ca,
                    hp,
                    flows,
                    actions,
                    feedback,
                    b,
                );

                let mut footprint = 1e-6_f64;
                let mut stressed_touch = 0.0_f64;
                let mut relieved_touch = 0.0_f64;
                let mut tight_touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit > 1e-6 {
                        let util = (flows[l].abs() / limit).min(2.0);
                        footprint += p.abs() * (0.25 + util * util);
                        if util > 0.60 {
                            stressed_touch += p.abs() * util;
                        }
                    }
                    relieved_touch += p.abs() * feedback.relieved[l];
                    tight_touch += p.abs() * feedback.tight[l];
                }

                let standalone = potential(challenge, state, ca, b);
                let score = (standalone + 0.70 * compat)
                    * (1.0 + 0.08 * stressed_touch + 0.18 * relieved_touch + 0.35 * tight_touch) / footprint
                    + 0.05 * compat
                    + 1e-6 * challenge.batteries[b].capacity_mwh;
                (b, score)
            })
            .collect();
        if coarse_ranked.len() > shortlist_cap {
            coarse_ranked.select_nth_unstable_by(shortlist_cap - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            coarse_ranked.truncate(shortlist_cap);
        }
        coarse_ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let shortlist: Vec<usize> = coarse_ranked.into_iter().map(|(b, _)| b).collect();

        let mut refined: Vec<(usize, f64)> = Vec::new();
        for &b in &shortlist {
            let compat = expansion_compatibility_score(
                challenge,
                state,
                ca,
                hp,
                flows,
                actions,
                feedback,
                b,
            );
            let v_cur = eval_profit(challenge, state, ca, b, actions[b]);
            let (u, v) = best_action_in_window(challenge, state, ca, hp, flows, actions, b);
            let gain = v - v_cur;
            if gain <= 1e-9 || u.abs() <= 1e-9 {
                if compat <= 1e-9 {
                    continue;
                }
            }

            let mut footprint = 1e-6_f64;
            let mut stressed_touch = 0.0_f64;
            let mut relief_bonus = 0.0_f64;
            let mut relief_penalty = 0.0_f64;
            let mut tight_bonus = 0.0_f64;
            let mut tight_penalty = 0.0_f64;
            for &(l, p) in &ca.b_to_lines[b] {
                let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                if limit > 1e-6 {
                    let util = (flows[l].abs() / limit).min(2.0);
                    footprint += p.abs() * (0.25 + util * util);
                    if util > 0.60 {
                        stressed_touch += p.abs() * util;
                    }
                }

                let relieved = feedback.relieved[l];
                let tight = feedback.tight[l];
                if relieved > 0.0 || tight > 0.0 {
                    let signed_move = p * u;
                    if flows[l] * signed_move < 0.0 {
                        relief_bonus += relieved * signed_move.abs();
                        tight_bonus += 0.65 * tight * signed_move.abs();
                    } else if flows[l] * signed_move > 0.0 {
                        relief_penalty += relieved * signed_move.abs();
                        tight_penalty += tight * signed_move.abs();
                    }
                }
            }

            let score = (gain + 0.15 * compat)
                * (1.0 + 0.10 * stressed_touch + 0.22 * relief_bonus + 0.35 * tight_bonus)
                / (footprint * (1.0 + 0.22 * relief_penalty + 0.55 * tight_penalty))
                + 0.25 * compat
                + 1e-6 * challenge.batteries[b].capacity_mwh;
            refined.push((b, score));
        }

        let need = reserve_target.min(num_b).saturating_sub(candidates.len());
        if refined.len() > need && need > 0 {
            refined.select_nth_unstable_by(need - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            refined.truncate(need);
        }
        refined.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
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
                let compat = expansion_compatibility_score(
                    challenge,
                    state,
                    ca,
                    hp,
                    flows,
                    actions,
                    feedback,
                    b,
                );
                let mut stressed_touch = 0.0_f64;
                let mut relieved_touch = 0.0_f64;
                let mut tight_touch = 0.0_f64;
                for &(l, p) in &ca.b_to_lines[b] {
                    let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
                    if limit > 1e-6 {
                        let util = (flows[l].abs() / limit).min(2.0);
                        stressed_touch += p.abs() * util;
                    }
                    relieved_touch += p.abs() * feedback.relieved[l];
                    tight_touch += p.abs() * feedback.tight[l];
                }
                let mut score = actions[b].abs()
                    + 0.15 * potential(challenge, state, ca, b)
                    + 0.15 * stressed_touch
                    + 0.25 * relieved_touch
                    + 0.50 * tight_touch
                    + 0.30 * compat;
                if seed_mask[b] {
                    score += 1.0;
                }
                (b, score)
            })
            .collect();
        if ranked.len() > cap {
            ranked.select_nth_unstable_by(cap - 1, |a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            ranked.truncate(cap);
        }
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        candidates = ranked.into_iter().map(|(b, _)| b).collect();
    }

    candidates
}

fn select_post_deflator_rebuild_block(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    seed_mask: &[bool],
    feedback: &LineFeedback,
    candidates: &[usize],
) -> Vec<usize> {
    let num_l = challenge.network.flow_limits.len();
    let mut ranked: Vec<(usize, f64)> = Vec::new();

    for &b in candidates {
        if actions[b].abs() <= 1e-7 {
            continue;
        }

        let keep_val = eval_profit(challenge, state, ca, b, actions[b]);
        let zero_val = eval_profit(challenge, state, ca, b, 0.0);
        let retained_value = (keep_val - zero_val).max(0.0);

        let mut worsen = 0.0_f64;
        let mut relief = 0.0_f64;
        let mut touch = 0.0_f64;
        for &(l, p) in &ca.b_to_lines[b] {
            if l >= num_l {
                continue;
            }
            let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
            if limit <= 1e-6 {
                continue;
            }

            let util = (flows[l].abs() / limit).min(2.0);
            let line_weight = 0.65 * feedback.relieved[l]
                + 1.35 * feedback.tight[l]
                + (util - 0.72).max(0.0);
            if line_weight <= 1e-9 {
                continue;
            }

            let signed = p * actions[b];
            touch += p.abs() * (0.1 + line_weight);
            if flows[l] * signed > 1e-9 {
                worsen += line_weight * signed.abs();
            } else if flows[l] * signed < -1e-9 {
                relief += line_weight * signed.abs();
            }
        }

        if worsen <= 1e-9 {
            continue;
        }

        let relief_discount = (relief / worsen.max(1e-9)).min(2.0);
        let mut score = (worsen + 0.15 * touch) / (retained_value + 1e-6)
            * (1.0 - 0.45 * relief_discount).max(0.25);
        score += 0.02 * actions[b].abs();
        if seed_mask[b] {
            score += 0.05;
        }

        if score > 1e-8 {
            ranked.push((b, score));
        }
    }

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let cap = if candidates.len() <= 16 {
        4
    } else if candidates.len() <= 36 {
        6
    } else {
        8
    };
    ranked.into_iter().take(cap).map(|(b, _)| b).collect()
}

fn run_post_deflator_block_rebuild(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
    candidates: &[usize],
    seed_mask: &[bool],
    feedback: &LineFeedback,
) -> bool {
    let num_l = challenge.network.flow_limits.len();
    if candidates.is_empty() || num_l == 0 {
        return false;
    }

    let flows = compute_action_flows(ca, flows_base, actions);
    let mut max_util = 0.0_f64;
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        max_util = max_util.max((flows[l].abs() / limit).min(2.0));
    }
    let relief_mass = feedback_relief_mass(feedback);
    let tight_mass = feedback_tight_mass(feedback);

    if max_util < 0.78 && relief_mass < 0.12 && tight_mass < 0.10 {
        return false;
    }
    if candidates.len() > 40 && max_util < 0.90 && relief_mass < 0.20 && tight_mass < 0.16 {
        return false;
    }

    let block = select_post_deflator_rebuild_block(
        challenge,
        state,
        ca,
        hp,
        &flows,
        actions,
        seed_mask,
        feedback,
        candidates,
    );
    if block.is_empty() || (block.len() < 2 && max_util < 0.92) {
        return false;
    }

    let base_score = approx_actions_value(challenge, state, ca, hp, flows_base, actions);
    if !base_score.is_finite() {
        return false;
    }

    let mut best_score = base_score;
    let mut best_actions = actions.to_vec();
    let sweeps = if candidates.len() <= 24 || block.len() <= 4 { 2 } else { 1 };

    let mut drop_counts = vec![1usize];
    if block.len() >= 2 {
        drop_counts.push(2usize);
    }
    if block.len() >= 3 && (max_util > 0.90 || relief_mass + tight_mass > 0.30) {
        drop_counts.push(3usize);
    }

    for drop_count in drop_counts {
        let mut alt_actions = actions.to_vec();
        let mut changed = false;
        for &b in block.iter().take(drop_count) {
            if alt_actions[b].abs() > 1e-9 {
                alt_actions[b] = 0.0;
                changed = true;
            }
        }
        if !changed {
            continue;
        }

        let alt_flows = compute_action_flows(ca, flows_base, &alt_actions);
        let feasible = (0..num_l).all(|l| {
            alt_flows[l].abs() <= (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0) + 1e-7
        });
        if !feasible {
            continue;
        }

        run_asca_candidates(challenge, state, ca, hp, flows_base, &mut alt_actions, candidates, sweeps);
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

fn run_post_deflator_polish(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows_base: &[f64],
    actions: &mut [f64],
    seed_mask: &[bool],
    feedback: &LineFeedback,
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

    let relief_mass = feedback_relief_mass(feedback);
    let tight_mass = feedback_tight_mass(feedback);
    if !actions.iter().any(|&u| u.abs() > 1e-7) && relief_mass < 0.05 && tight_mass < 0.05 {
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
    if seed_count == 0 && relief_mass < 0.05 && tight_mass < 0.05 && max_util < 0.70 {
        return;
    }
    if num_b > 80 && seed_count <= 2 && relief_mass < 0.10 && tight_mass < 0.08 && max_util < 0.60 {
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
        feedback,
    );
    if candidates.is_empty() {
        return;
    }

    let dual_trigger = max_util > 0.74 || relief_mass > 0.10 || tight_mass > 0.08;
    if dual_trigger {
        let base_score = approx_actions_value(challenge, state, ca, hp, flows_base, actions);
        if base_score.is_finite() {
            let mut dual_actions = actions.to_vec();
            if run_post_deflator_dual_reexpansion(
                challenge,
                state,
                ca,
                hp,
                flows_base,
                &mut dual_actions,
                &candidates,
                feedback,
            ) {
                let dual_sweeps = if num_b <= 18 || candidates.len() <= 20 { 2 } else { 1 };
                run_asca_candidates(
                    challenge,
                    state,
                    ca,
                    hp,
                    flows_base,
                    &mut dual_actions,
                    &candidates,
                    dual_sweeps,
                );
                let dual_score =
                    approx_actions_value(challenge, state, ca, hp, flows_base, &dual_actions);
                if dual_score > base_score + 1e-7 {
                    actions.copy_from_slice(&dual_actions);
                }
            }
        }
    }

    let sweeps = if num_b <= 18 || ((relief_mass + tight_mass) > 0.35 && candidates.len() <= 32) { 2 } else { 1 };
    run_asca_candidates(challenge, state, ca, hp, flows_base, actions, &candidates, sweeps);
    run_post_deflator_block_rebuild(
        challenge,
        state,
        ca,
        hp,
        flows_base,
        actions,
        &candidates,
        seed_mask,
        feedback,
    );
}

fn adjusted_deflator_roi(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
    hp: &TrackHp,
    flows: &[f64],
    actions: &[f64],
    line: usize,
    b: usize,
    contrib: f64,
) -> f64 {
    let val_curr = eval_profit(challenge, state, ca, b, actions[b]);
    let val_zero = eval_profit(challenge, state, ca, b, 0.0);
    let base_roi = ((val_curr - val_zero).max(0.0)) / contrib.abs().max(1e-6);

    let mut relieve_other = 0.0_f64;
    let mut worsen_other = 0.0_f64;
    for &(ll, p) in &ca.b_to_lines[b] {
        if ll == line {
            continue;
        }
        let limit = (challenge.network.flow_limits[ll] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 {
            continue;
        }
        let util = (flows[ll].abs() / limit).min(2.0);
        if util < 0.65 {
            continue;
        }

        let signed = p * actions[b];
        let weight = (util - 0.60).max(0.0);
        if flows[ll] * signed < 0.0 {
            relieve_other += weight * signed.abs();
        } else if flows[ll] * signed > 0.0 {
            worsen_other += weight * signed.abs();
        }
    }

    base_roi * (1.0 + 0.35 * relieve_other) / (1.0 + 0.30 * worsen_other)
}

fn run_deflator(
    challenge: &Challenge,
    state: &State,
    ca: &aiioagaypCache,
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
                    let roi = adjusted_deflator_roi(
                        challenge,
                        state,
                        ca,
                        hp,
                        &flows,
                        actions,
                        l,
                        b,
                        contrib,
                    );
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

    let mut feedback = compute_line_relief_signal(challenge, hp, &flows_before_repair, &flows);

    let mut candidate_mask = changed;
    let stress_cut = if num_b <= 30 { 0.80 } else if num_b <= 80 { 0.88 } else { 0.93 };
    for l in 0..num_l {
        let limit = (challenge.network.flow_limits[l] - hp.flow_margin).max(0.0);
        if limit <= 1e-6 { continue; }
        if flows[l].abs() >= stress_cut * limit || feedback_activity(&feedback, l) > 1e-9 {
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
                score += 0.20 * p.abs() * feedback.relieved[l];
                score += 0.40 * p.abs() * feedback.tight[l];
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
    feedback = compute_line_relief_signal(challenge, hp, &flows_before_repair, &flows);
    run_post_deflator_polish(challenge, state, ca, hp, flows_base, actions, &candidate_mask, &feedback);
}