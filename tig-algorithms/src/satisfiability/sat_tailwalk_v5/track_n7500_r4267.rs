use super::{target_track_high, Hyperparameters};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

const N7500_PHASE_MAX_FUEL: f64 = 190_000_000_000.0;
const N7500_ROUTE_MAX_FUEL: f64 = 170_000_000_000.0;
const N7500_MAX_PHASE_ATTEMPTS: usize = 6;

fn n7500_best_tail_extension_flips(
    hp: &Hyperparameters,
    best_unsat: usize,
    flip_fuel: f64,
) -> usize {
    let extension_fuel = hp.target_n7500_best_tail_fuel.unwrap_or(0.0);
    if !extension_fuel.is_finite()
        || extension_fuel <= 0.0
        || flip_fuel <= 0.0
        || best_unsat == 0
        || best_unsat > hp.target_n7500_best_tail_max_unsat.unwrap_or(1)
    {
        return 0;
    }

    (extension_fuel / flip_fuel) as usize
}

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[inline(always)]
fn lit_is_satisfied(lit: i32, vars: &[bool]) -> bool {
    let v = lit_var_index(lit);
    (lit > 0 && vars[v]) || (lit < 0 && !vars[v])
}

pub(crate) fn solve(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    seed_key: u64,
    nv: usize,
    nc: usize,
    density: f64,
    p_cnt: Vec<u32>,
    n_cnt: Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>,
    co: &[u32],
    all_three_clauses: bool,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    if nv != 7_500 || density < 4.24 {
        return target_track_high::solve(
            hp,
            rng,
            seed_key,
            nv,
            nc,
            density,
            p_cnt,
            n_cnt,
            all_off,
            p_bound,
            all_data,
            cl,
            co,
            all_three_clauses,
            save_solution,
        );
    }

    if n7500_use_high_route(hp) {
        let route_hp = n7500_v2_like_hp(hp);
        return target_track_high::solve(
            &route_hp,
            rng,
            seed_key,
            nv,
            nc,
            density,
            p_cnt,
            n_cnt,
            all_off,
            p_bound,
            all_data,
            cl,
            co,
            all_three_clauses,
            save_solution,
        );
    }

    solve_phase_attempts(
        hp,
        rng,
        nv,
        nc,
        density,
        &p_cnt,
        &n_cnt,
        all_off,
        p_bound,
        all_data,
        cl,
        co,
        all_three_clauses,
        save_solution,
    )
}

fn n7500_use_high_route(hp: &Hyperparameters) -> bool {
    hp.target_n7500_route.as_deref() == Some("high")
}

fn n7500_v2_like_hp(hp: &Hyperparameters) -> Hyperparameters {
    let mut route_hp = hp.clone();
    if route_hp.target_max_fuel.is_none() {
        route_hp.target_max_fuel = Some(N7500_ROUTE_MAX_FUEL);
    }
    route_hp
}

fn build_phase_appearances(nv: usize, p_cnt: &[u32], n_cnt: &[u32]) -> (Vec<u32>, u32) {
    let mut appearances = Vec::with_capacity(nv);
    let mut max_app = 1u32;
    for v in 0..nv {
        let app = p_cnt[v] + n_cnt[v];
        max_app = max_app.max(app);
        appearances.push(app);
    }
    (appearances, max_app)
}

fn build_phase_keep_threshold(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    var_appearances: &[u32],
    max_app: u32,
) -> Vec<u32> {
    let max_app = max_app.max(1) as f64;
    let mut thresholds = Vec::with_capacity(nv);
    for v in 0..nv {
        let app = var_appearances[v] as f64;
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        let skew = if np + nn > 0.0 {
            (np - nn).abs() / (np + nn)
        } else {
            0.0
        };
        let keep = (0.15 + 0.35 * (app / max_app) + 0.25 * skew).clamp(0.0, 0.90);
        thresholds.push((keep * u32::MAX as f64) as u32);
    }
    thresholds
}

fn build_phase_attempt_budgets(
    primary_flips: usize,
    primary_attempts: usize,
    max_flips: usize,
) -> ([usize; N7500_MAX_PHASE_ATTEMPTS], usize) {
    debug_assert!(primary_attempts > 0);
    debug_assert!(primary_attempts <= 4);

    let mut budgets = [0usize; N7500_MAX_PHASE_ATTEMPTS];
    let mut len = 0usize;
    for attempt in 0..primary_attempts {
        budgets[len] = primary_flips / primary_attempts
            + usize::from(attempt < primary_flips % primary_attempts);
        len += 1;
    }
    if max_flips > primary_flips {
        let extra_flips = max_flips - primary_flips;
        let extra_attempts = if extra_flips > primary_flips / 2 {
            2usize
        } else {
            1usize
        };
        for attempt in 0..extra_attempts {
            budgets[len] =
                extra_flips / extra_attempts + usize::from(attempt < extra_flips % extra_attempts);
            len += 1;
        }
    }
    (budgets, len)
}

fn solve_phase_attempts(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
    nv: usize,
    nc: usize,
    density: f64,
    p_cnt: &[u32],
    n_cnt: &[u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    cl: &mut [i32],
    co: &[u32],
    all_three_clauses: bool,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let max_fuel = hp.target_max_fuel.unwrap_or(N7500_PHASE_MAX_FUEL);
    let (var_appearances, max_phase_appearance) = build_phase_appearances(nv, p_cnt, n_cnt);
    let mut phase_keep_threshold = None::<Vec<u32>>;

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt();
    let flip_fuel = 200.0 + difficulty_factor;
    let flips_for_fuel = |fuel: f64| -> usize {
        if flip_fuel <= 0.0 {
            return 0;
        }
        ((fuel - base_fuel).max(0.0) / flip_fuel) as usize
    };
    let max_flips = flips_for_fuel(max_fuel);
    let primary_fuel = max_fuel.min(250_000_000_000.0);
    let primary_flips = flips_for_fuel(primary_fuel);

    let primary_attempts = if hp.target_max_fuel.is_some() && primary_fuel < 120_000_000_000.0 {
        2usize
    } else {
        4usize
    }
    .min(primary_flips.max(1));
    let (attempt_budgets, attempt_budget_len) =
        build_phase_attempt_budgets(primary_flips, primary_attempts, max_flips);

    let base_prob = hp.target_base_prob.unwrap_or(0.52);
    let max_random_prob = hp.max_prob.unwrap_or(0.9);
    let check_interval = hp.check_interval.unwrap_or(97).max(1);
    let variance_interval = 1_000usize;
    let compaction_factor = hp.target_residual_compaction_factor.unwrap_or(3);
    let compaction_min_gap = hp.target_residual_compaction_min_gap.unwrap_or(64);
    let trace_enabled = hp.target_trace_n7500.unwrap_or(false);
    let trace_interval = hp.target_trace_interval.unwrap_or(variance_interval).max(1);
    debug_assert_eq!(
        all_three_clauses,
        phase_clause_offsets_are_three(nc, co, cl.len())
    );
    let all_phase_clauses_are_three = all_three_clauses;

    let mut best_unsat = nc + 1;
    let mut best_vars = Vec::<bool>::new();
    let mut residual_seen = Vec::<u32>::new();
    let mut residual_seen_stamp = 1u32;
    let mut vars = Vec::with_capacity(nv);
    let mut num_good = vec![0u8; nc];
    let mut residual = Vec::with_capacity(super::initial_residual_capacity(nc));
    let mut var_age = Vec::<u16>::new();

    for attempt in 0..=attempt_budget_len {
        let is_best_tail = attempt == attempt_budget_len;
        let budget = if is_best_tail {
            n7500_best_tail_extension_flips(hp, best_unsat, flip_fuel)
        } else {
            attempt_budgets[attempt]
        };
        if budget == 0 {
            if is_best_tail {
                break;
            }
            continue;
        }

        if is_best_tail {
            if best_vars.len() != nv {
                break;
            }
            vars.clone_from(&best_vars);
        } else {
            let attempt_noise = hp
                .init_noise
                .unwrap_or((0.003 * (1.0 + 0.45 * attempt as f64)).min(0.08));
            let attempt_relax = (0.06 * attempt as f64).min(0.18);
            initial_assignment_phase_into(
                &mut vars,
                nv,
                p_cnt,
                n_cnt,
                rng,
                hp.target_nad.unwrap_or(1.0).max(0.01),
                attempt_noise,
                attempt_relax,
            );

            if attempt > 0 && best_unsat < nc {
                debug_assert_eq!(best_vars.len(), nv);
                let phase_keep_threshold = phase_keep_threshold.get_or_insert_with(|| {
                    build_phase_keep_threshold(
                        nv,
                        p_cnt,
                        n_cnt,
                        &var_appearances,
                        max_phase_appearance,
                    )
                });
                for v in 0..nv {
                    if p_cnt[v] > 0 && n_cnt[v] > 0 && rng.gen::<u32>() < phase_keep_threshold[v] {
                        vars[v] = best_vars[v];
                    }
                }
            }
        }

        rebuild_state(
            nc,
            co,
            cl,
            &vars,
            &mut num_good,
            &mut residual,
            all_phase_clauses_are_three,
        );
        let mut unsat_count = residual.len();
        let mut trace =
            trace_enabled.then(|| N7500TraceStats::new(attempt, is_best_tail, budget, unsat_count));

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            if unsat_count > 0 {
                best_vars.clone_from(&vars);
            }
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution {
                variables: vars,
            });
            return Ok(());
        }

        if var_age.len() == nv {
            var_age.fill(0);
        } else {
            var_age.resize(nv, 0);
        }
        let mut current_prob = base_prob;
        let mut rounds = 0usize;
        let mut stagnation = 0usize;
        let mut window_max = unsat_count;
        let mut window_min = unsat_count;
        let mut check_countdown = check_interval;
        let mut check_due = false;
        let mut variance_countdown = variance_interval;
        let mut variance_due = false;
        let mut trace_countdown = trace_interval;
        let mut trace_due = trace_enabled;

        unsafe {
            loop {
                if rounds >= budget || unsat_count == 0 {
                    break;
                }

                if unsat_count > window_max {
                    window_max = unsat_count;
                }
                if unsat_count < window_min {
                    window_min = unsat_count;
                }
                if trace_due {
                    if let Some(trace) = trace.as_mut() {
                        trace_due = false;
                        trace.observe_sample(rounds, residual.len(), unsat_count);
                    }
                }

                if check_due && unsat_count < best_unsat {
                    check_due = false;
                    best_unsat = unsat_count;
                    if unsat_count > 0 {
                        best_vars.clone_from(&vars);
                    }
                } else if check_due {
                    check_due = false;
                }

                if variance_due {
                    variance_due = false;
                    let variance = window_max.saturating_sub(window_min);
                    if let Some(trace) = trace.as_mut() {
                        trace.observe_variance(
                            residual.len(),
                            unsat_count,
                            variance,
                            compaction_factor,
                            compaction_min_gap,
                        );
                    }
                    if variance <= 2 {
                        stagnation += 1;
                        if let Some(trace) = trace.as_mut() {
                            trace.stagnation_ticks += 1;
                        }
                        current_prob = (current_prob + 0.15).min(max_random_prob);
                    } else if variance <= 6 {
                        stagnation += 1;
                        if let Some(trace) = trace.as_mut() {
                            trace.stagnation_ticks += 1;
                        }
                        current_prob = (current_prob + 0.05).min(max_random_prob);
                    } else if variance >= 20 {
                        stagnation = 0;
                        current_prob = base_prob;
                    } else {
                        stagnation = 0;
                        current_prob = current_prob * 0.8 + base_prob * 0.2;
                    }

                    if stagnation >= 3 {
                        for _ in 0..4 {
                            if residual.is_empty() || unsat_count == 0 {
                                break;
                            }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if *num_good.get_unchecked(pcid) > 0 {
                                remove_residual_unordered(&mut residual, rid);
                                if let Some(trace) = trace.as_mut() {
                                    trace.stale_kick_pops += 1;
                                }
                                continue;
                            }
                            let (pcs, pce) = phase_clause_bounds_unchecked(
                                pcid,
                                co,
                                all_phase_clauses_are_three,
                            );
                            if pcs == pce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = lit_var_index(lit);
                            let residual_pushes = flip_var(
                                v,
                                &mut vars,
                                &mut num_good,
                                &mut unsat_count,
                                &mut residual,
                                all_off,
                                p_bound,
                                all_data,
                            );
                            if let Some(trace) = trace.as_mut() {
                                trace.kick_flips += 1;
                                trace.residual_pushes += residual_pushes;
                            }
                            *var_age.get_unchecked_mut(v) = 0;
                        }
                        stagnation = 0;
                    }

                    if should_compact_residual(
                        residual.len(),
                        unsat_count,
                        compaction_factor,
                        compaction_min_gap,
                    ) {
                        let before_len = residual.len();
                        unsat_count = compact_residual(
                            &mut residual,
                            &num_good,
                            &mut residual_seen,
                            &mut residual_seen_stamp,
                        );
                        if let Some(trace) = trace.as_mut() {
                            trace.residual_compactions += 1;
                            trace.residual_compacted_clauses +=
                                before_len.saturating_sub(residual.len());
                        }
                    }

                    window_max = unsat_count;
                    window_min = unsat_count;
                }

                if unsat_count == 0 {
                    break;
                }

                let rand_val = rng.gen::<usize>();
                let mut cid = 0usize;
                let mut found = false;
                while !residual.is_empty() {
                    let rid = rand_val % residual.len();
                    cid = *residual.get_unchecked(rid) as usize;
                    if *num_good.get_unchecked(cid) > 0 {
                        remove_residual_unordered(&mut residual, rid);
                        if let Some(trace) = trace.as_mut() {
                            trace.stale_pops += 1;
                        }
                    } else {
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                let (cs, ce) = phase_clause_bounds_unchecked(cid, co, all_phase_clauses_are_three);
                let clen = ce - cs;

                if clen > 1 {
                    let ri = rand_val % clen;
                    if ri != 0 {
                        cl.swap(cs, cs + ri);
                    }
                }

                let v_idx = if all_phase_clauses_are_three {
                    debug_assert_eq!(clen, 3);
                    choose_var3(
                        rng,
                        current_prob,
                        cs,
                        cl,
                        &vars,
                        &num_good,
                        &var_age,
                        &var_appearances,
                        all_off,
                        p_bound,
                        all_data,
                    )
                } else {
                    choose_var(
                        rng,
                        current_prob,
                        cs,
                        ce,
                        cl,
                        &vars,
                        &num_good,
                        &var_age,
                        &var_appearances,
                        all_off,
                        p_bound,
                        all_data,
                    )
                };

                let residual_pushes = flip_var(
                    v_idx,
                    &mut vars,
                    &mut num_good,
                    &mut unsat_count,
                    &mut residual,
                    all_off,
                    p_bound,
                    all_data,
                );
                if let Some(trace) = trace.as_mut() {
                    trace.residual_pushes += residual_pushes;
                }
                *var_age.get_unchecked_mut(v_idx) = 0;
                if all_phase_clauses_are_three {
                    bump_clause_var_ages3(cs, cl, &mut var_age);
                } else {
                    bump_clause_var_ages(cs, ce, cl, &mut var_age);
                }

                rounds += 1;
                check_due = super::advance_interval_due(&mut check_countdown, check_interval);
                variance_due =
                    super::advance_interval_due(&mut variance_countdown, variance_interval);
                if trace.is_some() {
                    trace_due = super::advance_interval_due(&mut trace_countdown, trace_interval);
                }
                if unsat_count < best_unsat && unsat_count <= 12 {
                    best_unsat = unsat_count;
                    if unsat_count > 0 {
                        best_vars.clone_from(&vars);
                    }
                }
            }
        }

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            if unsat_count > 0 {
                best_vars.clone_from(&vars);
            }
        }
        if let Some(mut trace) = trace {
            trace.finish(rounds, residual.len(), unsat_count, best_unsat);
            trace.emit();
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution {
                variables: vars,
            });
            return Ok(());
        }
    }

    let _ = save_solution(&Solution {
        variables: finalize_phase_best_vars(best_vars, nv),
    });
    Ok(())
}

fn finalize_phase_best_vars(best_vars: Vec<bool>, nv: usize) -> Vec<bool> {
    if best_vars.is_empty() {
        vec![false; nv]
    } else {
        best_vars
    }
}

fn initial_assignment_phase(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    nad: f64,
    random_threshold: f64,
    relax: f64,
) -> Vec<bool> {
    let mut vars = Vec::with_capacity(nv);
    initial_assignment_phase_into(
        &mut vars,
        nv,
        p_cnt,
        n_cnt,
        rng,
        nad,
        random_threshold,
        relax,
    );
    vars
}

fn initial_assignment_phase_into(
    vars: &mut Vec<bool>,
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    nad: f64,
    random_threshold: f64,
    relax: f64,
) {
    let noise = random_threshold.clamp(0.0, 0.5);
    vars.clear();
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 {
            vars.push(true);
            continue;
        }
        if np == 0 && nn > 0 {
            vars.push(false);
            continue;
        }
        let vad = if nn > 0 {
            np as f64 / nn as f64
        } else {
            nad + 1.0
        };
        if vad <= nad {
            vars.push(rng.gen_bool(noise));
        } else {
            let bias = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            let prob = (bias * (1.0 - relax) + 0.5 * relax).clamp(0.001, 0.999);
            vars.push(rng.gen_bool(prob));
        }
    }
}

fn rebuild_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    all_three_literal: bool,
) {
    debug_assert!(num_good.len() >= nc);
    residual.clear();
    if all_three_literal {
        debug_assert_eq!(cl.len(), nc * 3);
        for i in 0..nc {
            let s = i * 3;
            let good = lit_is_satisfied(cl[s], vars) as u8
                + lit_is_satisfied(cl[s + 1], vars) as u8
                + lit_is_satisfied(cl[s + 2], vars) as u8;
            num_good[i] = good;
            if good == 0 {
                residual.push(i as u32);
            }
        }
        return;
    }
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let good = phase_clause_good_count(s, e, cl, vars);
        num_good[i] = good;
        if good == 0 {
            residual.push(i as u32);
        }
    }
}

#[inline(always)]
fn phase_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
    if co.len() <= nc || cl_len != nc.saturating_mul(3) || co[0] != 0 || co[nc] as usize != cl_len {
        return false;
    }
    for (i, &off) in co.iter().take(nc + 1).enumerate() {
        if off as usize != i * 3 {
            return false;
        }
    }
    true
}

#[inline(always)]
unsafe fn phase_clause_bounds_unchecked(
    cid: usize,
    co: &[u32],
    all_three_literal: bool,
) -> (usize, usize) {
    if all_three_literal {
        let s = cid * 3;
        (s, s + 3)
    } else {
        (
            *co.get_unchecked(cid) as usize,
            *co.get_unchecked(cid + 1) as usize,
        )
    }
}

#[inline(always)]
fn phase_clause_good_count(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> u8 {
    match e - s {
        1 => lit_is_satisfied(cl[s], vars) as u8,
        2 => lit_is_satisfied(cl[s], vars) as u8 + lit_is_satisfied(cl[s + 1], vars) as u8,
        3 => {
            lit_is_satisfied(cl[s], vars) as u8
                + lit_is_satisfied(cl[s + 1], vars) as u8
                + lit_is_satisfied(cl[s + 2], vars) as u8
        }
        _ => {
            let mut good = 0u8;
            for &lit in &cl[s..e] {
                if lit_is_satisfied(lit, vars) {
                    good += 1;
                }
            }
            good
        }
    }
}

#[inline(always)]
fn should_compact_residual(
    residual_len: usize,
    unsat_count: usize,
    factor: usize,
    min_gap: usize,
) -> bool {
    factor > 0 && residual_len > unsat_count.saturating_mul(factor).saturating_add(min_gap)
}

#[inline(always)]
fn remove_residual_unordered(residual: &mut Vec<u32>, rid: usize) {
    if rid + 1 == residual.len() {
        residual.pop();
    } else {
        residual.swap_remove(rid);
    }
}

#[inline(always)]
fn compact_residual(
    residual: &mut Vec<u32>,
    num_good: &[u8],
    seen: &mut Vec<u32>,
    stamp: &mut u32,
) -> usize {
    if residual.is_empty() {
        return 0;
    }
    if residual.len() == 1 {
        let cid = residual[0] as usize;
        if cid < num_good.len() && num_good[cid] == 0 {
            return 1;
        }
        residual.clear();
        return 0;
    }
    if seen.len() < num_good.len() {
        seen.clear();
        seen.resize(num_good.len(), 0);
        *stamp = 1;
    }
    if *stamp == u32::MAX {
        seen.fill(0);
        *stamp = 1;
    }
    let mark = *stamp;
    *stamp += 1;

    let mut write = 0usize;
    for read in 0..residual.len() {
        let cid_u32 = residual[read];
        let cid = cid_u32 as usize;
        if cid < num_good.len() && num_good[cid] == 0 && seen[cid] != mark {
            seen[cid] = mark;
            if write != read {
                residual[write] = cid_u32;
            }
            write += 1;
        }
    }
    residual.truncate(write);
    write
}

#[derive(Default)]
struct N7500TraceStats {
    attempt: usize,
    best_tail: bool,
    budget: usize,
    initial_unsat: usize,
    final_rounds: usize,
    final_unsat: usize,
    final_residual_len: usize,
    best_unsat: usize,
    max_residual_len: usize,
    max_residual_gap: usize,
    stale_pops: usize,
    stale_kick_pops: usize,
    residual_pushes: usize,
    variance_ticks: usize,
    stagnation_ticks: usize,
    kick_flips: usize,
    compaction_candidate_ticks: usize,
    residual_compactions: usize,
    residual_compacted_clauses: usize,
}

impl N7500TraceStats {
    fn new(attempt: usize, best_tail: bool, budget: usize, initial_unsat: usize) -> Self {
        Self {
            attempt,
            best_tail,
            budget,
            initial_unsat,
            final_unsat: initial_unsat,
            best_unsat: initial_unsat,
            max_residual_len: initial_unsat,
            ..Self::default()
        }
    }

    fn observe_sample(&mut self, _rounds: usize, residual_len: usize, unsat_count: usize) {
        self.max_residual_len = self.max_residual_len.max(residual_len);
        self.max_residual_gap = self
            .max_residual_gap
            .max(residual_len.saturating_sub(unsat_count));
        self.best_unsat = self.best_unsat.min(unsat_count);
    }

    fn observe_variance(
        &mut self,
        residual_len: usize,
        unsat_count: usize,
        _variance: usize,
        compaction_factor: usize,
        compaction_min_gap: usize,
    ) {
        self.variance_ticks += 1;
        if should_compact_residual(
            residual_len,
            unsat_count,
            compaction_factor,
            compaction_min_gap,
        ) {
            self.compaction_candidate_ticks += 1;
        }
    }

    fn finish(
        &mut self,
        rounds: usize,
        residual_len: usize,
        unsat_count: usize,
        best_unsat: usize,
    ) {
        self.final_rounds = rounds;
        self.final_residual_len = residual_len;
        self.final_unsat = unsat_count;
        self.best_unsat = self.best_unsat.min(best_unsat).min(unsat_count);
        self.observe_sample(rounds, residual_len, unsat_count);
    }

    fn emit(&self) {
        eprintln!(
            "c001_n7500_trace_diag attempt={} best_tail={} budget={} rounds={} initial_unsat={} final_unsat={} best_unsat={} final_residual_len={} max_residual_len={} max_residual_gap={} stale_pops={} stale_kick_pops={} residual_pushes={} variance_ticks={} stagnation_ticks={} kick_flips={} compaction_candidate_ticks={} residual_compactions={} residual_compacted_clauses={}",
            self.attempt,
            self.best_tail,
            self.budget,
            self.final_rounds,
            self.initial_unsat,
            self.final_unsat,
            self.best_unsat,
            self.final_residual_len,
            self.max_residual_len,
            self.max_residual_gap,
            self.stale_pops,
            self.stale_kick_pops,
            self.residual_pushes,
            self.variance_ticks,
            self.stagnation_ticks,
            self.kick_flips,
            self.compaction_candidate_ticks,
            self.residual_compactions,
            self.residual_compacted_clauses,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn n7500_lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn n7500_delayed_best_snapshot_preserves_empty_fallback() {
        assert_eq!(finalize_phase_best_vars(Vec::new(), 4), vec![false; 4]);
        assert_eq!(
            finalize_phase_best_vars(vec![true, false, true], 3),
            vec![true, false, true]
        );
    }

    #[test]
    fn initial_assignment_phase_into_matches_allocating_path() {
        let p_cnt = vec![3u32, 0, 1, 5, 2, 0, 10, 4];
        let n_cnt = vec![0u32, 4, 4, 1, 2, 0, 2, 8];
        let mut rng_expected = SmallRng::seed_from_u64(0x3141_5926_5358_9793);
        let expected = initial_assignment_phase(
            p_cnt.len(),
            &p_cnt,
            &n_cnt,
            &mut rng_expected,
            1.15,
            0.017,
            0.09,
        );

        let mut rng_actual = SmallRng::seed_from_u64(0x3141_5926_5358_9793);
        let mut actual = Vec::with_capacity(32);
        actual.extend([true, false, true, true, false]);
        let reused_capacity = actual.capacity();
        initial_assignment_phase_into(
            &mut actual,
            p_cnt.len(),
            &p_cnt,
            &n_cnt,
            &mut rng_actual,
            1.15,
            0.017,
            0.09,
        );

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), p_cnt.len());
        assert_eq!(actual.capacity(), reused_capacity);
    }

    #[test]
    fn n7500_phase_appearances_keep_u32_counts() {
        let p_cnt = vec![1_u32, 4, 0, 70_000];
        let n_cnt = vec![2_u32, 1, 300, 60_000];

        let (appearances, max_app) = build_phase_appearances(4, &p_cnt, &n_cnt);

        assert_eq!(appearances, vec![3_u32, 5, 300, 130_000]);
        assert_eq!(max_app, 130_000);
    }

    #[test]
    fn n7500_phase_appearances_max_matches_scan_reference() {
        let p_cnt = vec![0_u32, 4, 0, 70_000];
        let n_cnt = vec![0_u32, 1, 300, 60_000];
        let (appearances, max_app) = build_phase_appearances(4, &p_cnt, &n_cnt);
        assert_eq!(
            max_app,
            appearances.iter().copied().max().unwrap_or(1).max(1)
        );

        let empty: [u32; 0] = [];
        let (appearances, max_app) = build_phase_appearances(0, &empty, &empty);
        assert!(appearances.is_empty());
        assert_eq!(max_app, 1);

        let zeros = [0_u32, 0];
        let (appearances, max_app) = build_phase_appearances(2, &zeros, &zeros);
        assert_eq!(appearances, vec![0_u32, 0]);
        assert_eq!(max_app, 1);
    }

    #[test]
    fn n7500_phase_keep_threshold_matches_inline_formula() {
        let p_cnt = vec![1_u32, 4, 0, 70_000];
        let n_cnt = vec![2_u32, 1, 300, 60_000];
        let (appearances, max_app) = build_phase_appearances(4, &p_cnt, &n_cnt);
        let max_app_ref = appearances.iter().copied().max().unwrap_or(1).max(1) as f64;
        let thresholds = build_phase_keep_threshold(4, &p_cnt, &n_cnt, &appearances, max_app);
        let expected: Vec<u32> = (0..4)
            .map(|v| {
                let app = appearances[v] as f64;
                let np = p_cnt[v] as f64;
                let nn = n_cnt[v] as f64;
                let skew = if np + nn > 0.0 {
                    (np - nn).abs() / (np + nn)
                } else {
                    0.0
                };
                let keep = (0.15 + 0.35 * (app / max_app_ref) + 0.25 * skew).clamp(0.0, 0.90);
                (keep * u32::MAX as f64) as u32
            })
            .collect();

        assert_eq!(thresholds, expected);
    }

    #[test]
    fn n7500_phase_attempt_budgets_match_legacy_vector_order() {
        let cases = [
            (0usize, 1usize, 0usize),
            (1, 1, 3),
            (7, 4, 7),
            (10, 4, 14),
            (10, 4, 30),
            (101, 2, 153),
        ];

        for (primary_flips, primary_attempts, max_flips) in cases {
            let (budgets, len) =
                build_phase_attempt_budgets(primary_flips, primary_attempts, max_flips);
            let mut expected = Vec::with_capacity(primary_attempts + 2);
            for attempt in 0..primary_attempts {
                expected.push(
                    primary_flips / primary_attempts
                        + usize::from(attempt < primary_flips % primary_attempts),
                );
            }
            if max_flips > primary_flips {
                let extra_flips = max_flips - primary_flips;
                let extra_attempts = if extra_flips > primary_flips / 2 {
                    2usize
                } else {
                    1usize
                };
                for attempt in 0..extra_attempts {
                    expected.push(
                        extra_flips / extra_attempts
                            + usize::from(attempt < extra_flips % extra_attempts),
                    );
                }
            }

            assert_eq!(&budgets[..len], expected.as_slice());
            assert!(len <= N7500_MAX_PHASE_ATTEMPTS);
        }
    }

    #[test]
    fn n7500_best_tail_requires_explicit_fuel_and_near_solution() {
        let disabled = Hyperparameters::default();
        assert_eq!(n7500_best_tail_extension_flips(&disabled, 1, 250.0), 0);

        let enabled = Hyperparameters {
            target_n7500_best_tail_fuel: Some(25_000_000_000.0),
            target_n7500_best_tail_max_unsat: Some(1),
            ..Hyperparameters::default()
        };
        assert_eq!(
            n7500_best_tail_extension_flips(&enabled, 1, 250.0),
            100_000_000
        );
        assert_eq!(n7500_best_tail_extension_flips(&enabled, 2, 250.0), 0);
        assert_eq!(n7500_best_tail_extension_flips(&enabled, 0, 250.0), 0);
    }

    #[test]
    fn n7500_route_defaults_to_phase_and_allows_explicit_high_probe() {
        assert!(!n7500_use_high_route(&Hyperparameters::default()));
        assert!(n7500_use_high_route(&Hyperparameters {
            target_n7500_route: Some("high".to_string()),
            ..Hyperparameters::default()
        }));
    }

    #[test]
    fn n7500_v2_like_route_uses_capped_default_fuel() {
        let hp = n7500_v2_like_hp(&Hyperparameters::default());

        assert_eq!(hp.target_max_fuel, Some(N7500_ROUTE_MAX_FUEL));
    }

    #[test]
    fn residual_compaction_keeps_only_unique_unsat_clauses() {
        let mut residual = vec![0u32, 1, 0, 2, 3, 2, 4];
        let num_good = vec![0u8, 1, 0, 0, 2];
        let mut seen = Vec::new();
        let mut stamp = 1u32;

        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);

        assert_eq!(count, 3);
        assert_eq!(residual, vec![0, 2, 3]);
        assert_eq!(seen.len(), num_good.len());
        assert!(should_compact_residual(400, 100, 3, 64));
        assert!(!should_compact_residual(364, 100, 3, 64));
        assert!(!should_compact_residual(400, 100, 0, 64));
    }

    #[test]
    fn residual_compaction_preserves_already_compact_order() {
        let mut residual = vec![0u32, 2, 3];
        let num_good = vec![0u8, 1, 0, 0];
        let mut seen = Vec::new();
        let mut stamp = 1u32;

        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);

        assert_eq!(count, 3);
        assert_eq!(residual, vec![0, 2, 3]);
    }

    #[test]
    fn remove_residual_unordered_handles_moved_and_last_entries() {
        let mut residual = vec![2_u32, 5, 7];

        remove_residual_unordered(&mut residual, 1);
        assert_eq!(residual, vec![2_u32, 7]);

        remove_residual_unordered(&mut residual, 1);
        assert_eq!(residual, vec![2_u32]);
    }

    #[test]
    fn rebuild_state_overwrites_existing_counts_without_prefill() {
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];
        let mut num_good = vec![9_u8; 3];
        let mut residual = vec![99_u32];

        rebuild_state(3, &co, &cl, &vars, &mut num_good, &mut residual, false);

        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(residual, vec![2]);
    }

    #[test]
    fn rebuild_state_all_three_literal_fast_path_matches_generic() {
        let co = [0_u32, 3, 6, 9, 12];
        let cl = [1, -2, 3, -1, 2, -3, 1, 2, -3, -1, -2, -3];
        let vars = [true, true, false];

        let mut generic_good = vec![9_u8; 4];
        let mut generic_residual = Vec::new();
        rebuild_state(
            4,
            &co,
            &cl,
            &vars,
            &mut generic_good,
            &mut generic_residual,
            false,
        );

        let mut fast_good = vec![9_u8; 4];
        let mut fast_residual = Vec::new();
        rebuild_state(4, &co, &cl, &vars, &mut fast_good, &mut fast_residual, true);

        assert_eq!(fast_good, generic_good);
        assert_eq!(fast_residual, generic_residual);
    }

    #[test]
    fn phase_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9, 12];

        assert!(phase_clause_offsets_are_three(4, &co, 12));
        for cid in 0..4 {
            assert_eq!(
                unsafe { phase_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn phase_clause_offsets_reject_mixed_lengths_with_average_three() {
        let co = [0_u32, 2, 6, 9];

        assert!(!phase_clause_offsets_are_three(3, &co, 9));
        assert_eq!(
            unsafe { phase_clause_bounds_unchecked(1, &co, false) },
            (2, 6)
        );
    }

    #[test]
    fn phase_clause_good_count_matches_generic_reference() {
        let cl = [1, -2, 3, -1, 2, 4, -5, 5, -4, 3];
        let vars = [true, false, true, false, true];

        for (s, e, expected) in [(0, 0, 0_u8), (0, 1, 1), (0, 3, 3), (3, 5, 0), (5, 10, 3)] {
            let generic = cl[s..e]
                .iter()
                .filter(|&&lit| lit_is_satisfied(lit, &vars))
                .count() as u8;
            assert_eq!(generic, expected);
            assert_eq!(phase_clause_good_count(s, e, &cl, &vars), expected);
        }
    }

    #[test]
    fn choose_var_zero_break_returns_first_candidate_without_rng() {
        let mut rng = SmallRng::seed_from_u64(0x7500_4267);
        let mut expected_rng = SmallRng::seed_from_u64(0x7500_4267);
        let cl = [1, -2];
        let vars = [false, true];
        let num_good = [];
        let var_age = [0u16, 0];
        let var_appearances = [0u32, 0];
        let all_off = [0u32, 0, 0];
        let p_bound = [0u32, 0];
        let all_data = [];

        let chosen = unsafe {
            choose_var(
                &mut rng,
                1.0,
                0,
                cl.len(),
                &cl,
                &vars,
                &num_good,
                &var_age,
                &var_appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };

        assert_eq!(chosen, 0);
        assert_eq!(rng.gen::<u64>(), expected_rng.gen::<u64>());
    }

    #[test]
    fn choose_var3_matches_generic_and_rng_consumption() {
        let cl = [99, 1, 2, 3, 100];
        let vars = [true, true, true];
        let num_good = [1u8, 1, 1, 1];
        let var_age = [0u16, 0, 0];
        let var_appearances = [20u32, 5, 10];
        let all_off = [0u32, 2, 3, 4];
        let p_bound = [2u32, 3, 4];
        let all_data = [0u32, 1, 2, 3];
        let mut generic_rng = SmallRng::seed_from_u64(0x7500_0003);
        let mut fixed_rng = SmallRng::seed_from_u64(0x7500_0003);

        let generic = unsafe {
            choose_var(
                &mut generic_rng,
                0.0,
                1,
                4,
                &cl,
                &vars,
                &num_good,
                &var_age,
                &var_appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        let fixed = unsafe {
            choose_var3(
                &mut fixed_rng,
                0.0,
                1,
                &cl,
                &vars,
                &num_good,
                &var_age,
                &var_appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };

        assert_eq!(fixed, generic);
        assert_eq!(fixed, 1);
        assert_eq!(fixed_rng.gen::<u64>(), generic_rng.gen::<u64>());
    }

    #[test]
    fn choose_var3_zero_break_returns_first_candidate_without_rng() {
        let mut rng = SmallRng::seed_from_u64(0x7500_0004);
        let mut expected_rng = SmallRng::seed_from_u64(0x7500_0004);
        let cl = [1, 2, 3];
        let vars = [false, false, false];
        let num_good = [];
        let var_age = [0u16, 0, 0];
        let var_appearances = [0u32, 0, 0];
        let all_off = [0u32, 0, 0, 0];
        let p_bound = [0u32, 0, 0];
        let all_data = [];

        let chosen = unsafe {
            choose_var3(
                &mut rng,
                1.0,
                0,
                &cl,
                &vars,
                &num_good,
                &var_age,
                &var_appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };

        assert_eq!(chosen, 0);
        assert_eq!(rng.gen::<u64>(), expected_rng.gen::<u64>());
    }

    #[test]
    fn bump_clause_var_ages3_matches_generic_loop() {
        let cl = [99, 1, -2, 2, 100];
        let mut generic_age = [5_u16, u16::MAX - 1, 17];
        let mut fast_age = generic_age;

        unsafe {
            bump_clause_var_ages(1, 4, &cl, &mut generic_age);
            bump_clause_var_ages3(1, &cl, &mut fast_age);
        }

        assert_eq!(fast_age, generic_age);
        assert_eq!(fast_age, [6, u16::MAX, 17]);
    }

    #[test]
    fn residual_compaction_fast_paths_empty_and_singleton() {
        let num_good = vec![0u8, 1];
        let mut seen = Vec::new();
        let mut stamp = 7u32;

        let mut residual = Vec::new();
        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);
        assert_eq!(count, 0);
        assert!(residual.is_empty());
        assert!(seen.is_empty());
        assert_eq!(stamp, 7);

        residual.push(0);
        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);
        assert_eq!(count, 1);
        assert_eq!(residual, vec![0]);
        assert!(seen.is_empty());
        assert_eq!(stamp, 7);

        residual[0] = 1;
        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);
        assert_eq!(count, 0);
        assert!(residual.is_empty());
        assert!(seen.is_empty());
        assert_eq!(stamp, 7);
    }

    #[test]
    fn residual_compaction_reuses_seen_with_stamp_progression() {
        let num_good = vec![0u8, 1, 0, 0];
        let mut seen = Vec::new();
        let mut stamp = 1u32;

        let mut residual = vec![0u32, 1, 0, 2];
        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);
        assert_eq!(count, 2);
        assert_eq!(residual, vec![0, 2]);
        assert_eq!(seen.len(), num_good.len());
        assert_eq!(stamp, 2);

        let seen_len = seen.len();
        residual.clear();
        residual.extend_from_slice(&[2, 3, 2, 0]);
        let count = compact_residual(&mut residual, &num_good, &mut seen, &mut stamp);
        assert_eq!(count, 3);
        assert_eq!(residual, vec![2, 3, 0]);
        assert_eq!(seen.len(), seen_len);
        assert_eq!(stamp, 3);
    }

    #[test]
    fn n7500_trace_uses_open_compaction_thresholds() {
        let mut trace = N7500TraceStats::new(0, false, 1000, 10);

        trace.observe_variance(500, 100, 0, 4, 64);
        assert_eq!(trace.compaction_candidate_ticks, 1);

        trace.observe_variance(500, 100, 0, 5, 64);
        assert_eq!(trace.compaction_candidate_ticks, 1);
    }
}

#[inline(always)]
unsafe fn bump_clause_var_ages(cs: usize, ce: usize, cl: &[i32], var_age: &mut [u16]) {
    for j in cs..ce {
        let var = lit_var_index(*cl.get_unchecked(j));
        let age = var_age.get_unchecked_mut(var);
        *age = age.saturating_add(1);
    }
}

#[inline(always)]
unsafe fn bump_clause_var_ages3(cs: usize, cl: &[i32], var_age: &mut [u16]) {
    let v0 = lit_var_index(*cl.get_unchecked(cs));
    let age0 = var_age.get_unchecked_mut(v0);
    *age0 = age0.saturating_add(1);

    let v1 = lit_var_index(*cl.get_unchecked(cs + 1));
    let age1 = var_age.get_unchecked_mut(v1);
    *age1 = age1.saturating_add(1);

    let v2 = lit_var_index(*cl.get_unchecked(cs + 2));
    let age2 = var_age.get_unchecked_mut(v2);
    *age2 = age2.saturating_add(1);
}

#[inline(always)]
unsafe fn choose_var(
    rng: &mut SmallRng,
    current_prob: f64,
    cs: usize,
    ce: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    var_age: &[u16],
    var_appearances: &[u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    'outer_h: for j in cs..ce {
        let l = *cl.get_unchecked(j);
        let abs_l = lit_var_index(l);
        let (os, oe) = if *vars.get_unchecked(abs_l) {
            (
                *all_off.get_unchecked(abs_l) as usize,
                *p_bound.get_unchecked(abs_l) as usize,
            )
        } else {
            (
                *p_bound.get_unchecked(abs_l) as usize,
                *all_off.get_unchecked(abs_l + 1) as usize,
            )
        };

        for k in os..oe {
            let c = *all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                continue 'outer_h;
            }
        }
        return abs_l;
    }
    if rng.gen::<f64>() < current_prob {
        return lit_var_index(*cl.get_unchecked(cs));
    }

    let mut min_sad = usize::MAX;
    let mut v_min = lit_var_index(*cl.get_unchecked(cs));
    let mut min_weight = usize::MAX;

    for j in cs..ce {
        let l = *cl.get_unchecked(j);
        let abs_l = lit_var_index(l);
        let (os, oe) = if *vars.get_unchecked(abs_l) {
            (
                *all_off.get_unchecked(abs_l) as usize,
                *p_bound.get_unchecked(abs_l) as usize,
            )
        } else {
            (
                *p_bound.get_unchecked(abs_l) as usize,
                *all_off.get_unchecked(abs_l + 1) as usize,
            )
        };

        let mut sad = 0usize;
        for k in os..oe {
            let c = *all_data.get_unchecked(k) as usize;
            if *num_good.get_unchecked(c) == 1 {
                sad += 1;
            }
            if sad >= min_sad {
                break;
            }
        }

        if sad == 0 {
            let appearances = *var_appearances.get_unchecked(abs_l) as usize;
            let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
            let adjusted_weight = appearances.saturating_sub(age_bonus);
            if min_sad > 0 || adjusted_weight < min_weight {
                min_sad = 0;
                min_weight = adjusted_weight;
                v_min = abs_l;
            }
        } else if min_sad > 0 {
            let appearances = *var_appearances.get_unchecked(abs_l) as usize;
            let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
            let combined_weight = sad * 1000 + appearances - age_bonus.min(50);
            if combined_weight < min_weight {
                min_sad = sad;
                min_weight = combined_weight;
                v_min = abs_l;
            }
            if min_sad <= 1 {
                break;
            }
        }
    }

    v_min
}

#[inline(always)]
unsafe fn choose_var3(
    rng: &mut SmallRng,
    current_prob: f64,
    cs: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &[u8],
    var_age: &[u16],
    var_appearances: &[u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    debug_assert!(cs + 2 < cl.len());
    let v0 = lit_var_index(*cl.get_unchecked(cs));
    if var_sad_limited(v0, vars, num_good, all_off, p_bound, all_data, 1) == 0 {
        return v0;
    }
    let v1 = lit_var_index(*cl.get_unchecked(cs + 1));
    if var_sad_limited(v1, vars, num_good, all_off, p_bound, all_data, 1) == 0 {
        return v1;
    }
    let v2 = lit_var_index(*cl.get_unchecked(cs + 2));
    if var_sad_limited(v2, vars, num_good, all_off, p_bound, all_data, 1) == 0 {
        return v2;
    }

    if rng.gen::<f64>() < current_prob {
        return v0;
    }

    let mut min_sad = usize::MAX;
    let mut min_weight = usize::MAX;
    let mut v_min = v0;
    let sad0 = consider_var_choice(
        v0,
        vars,
        num_good,
        var_age,
        var_appearances,
        all_off,
        p_bound,
        all_data,
        &mut min_sad,
        &mut min_weight,
        &mut v_min,
    );
    if sad0 > 0 && min_sad <= 1 {
        return v_min;
    }
    let sad1 = consider_var_choice(
        v1,
        vars,
        num_good,
        var_age,
        var_appearances,
        all_off,
        p_bound,
        all_data,
        &mut min_sad,
        &mut min_weight,
        &mut v_min,
    );
    if sad1 > 0 && min_sad <= 1 {
        return v_min;
    }
    consider_var_choice(
        v2,
        vars,
        num_good,
        var_age,
        var_appearances,
        all_off,
        p_bound,
        all_data,
        &mut min_sad,
        &mut min_weight,
        &mut v_min,
    );
    v_min
}

#[inline(always)]
unsafe fn consider_var_choice(
    v: usize,
    vars: &[bool],
    num_good: &[u8],
    var_age: &[u16],
    var_appearances: &[u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    min_sad: &mut usize,
    min_weight: &mut usize,
    v_min: &mut usize,
) -> usize {
    let sad = var_sad_limited(v, vars, num_good, all_off, p_bound, all_data, *min_sad);
    if sad == 0 {
        let appearances = *var_appearances.get_unchecked(v) as usize;
        let age_bonus = (*var_age.get_unchecked(v) as usize) / 4;
        let adjusted_weight = appearances.saturating_sub(age_bonus);
        if *min_sad > 0 || adjusted_weight < *min_weight {
            *min_sad = 0;
            *min_weight = adjusted_weight;
            *v_min = v;
        }
    } else if *min_sad > 0 {
        let appearances = *var_appearances.get_unchecked(v) as usize;
        let age_bonus = (*var_age.get_unchecked(v) as usize) / 2;
        let combined_weight = sad * 1000 + appearances - age_bonus.min(50);
        if combined_weight < *min_weight {
            *min_sad = sad;
            *min_weight = combined_weight;
            *v_min = v;
        }
    }
    sad
}

#[inline(always)]
unsafe fn var_sad_limited(
    v: usize,
    vars: &[bool],
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    limit: usize,
) -> usize {
    let (os, oe) = if *vars.get_unchecked(v) {
        (
            *all_off.get_unchecked(v) as usize,
            *p_bound.get_unchecked(v) as usize,
        )
    } else {
        (
            *p_bound.get_unchecked(v) as usize,
            *all_off.get_unchecked(v + 1) as usize,
        )
    };
    let mut sad = 0usize;
    for k in os..oe {
        let c = *all_data.get_unchecked(k) as usize;
        if *num_good.get_unchecked(c) == 1 {
            sad += 1;
            if sad >= limit {
                break;
            }
        }
    }
    sad
}

#[inline(always)]
unsafe fn flip_var(
    v_idx: usize,
    vars: &mut [bool],
    num_good: &mut [u8],
    unsat_count: &mut usize,
    residual: &mut Vec<u32>,
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    let was_true = *vars.get_unchecked(v_idx);
    let (is, ie) = if was_true {
        (
            *p_bound.get_unchecked(v_idx) as usize,
            *all_off.get_unchecked(v_idx + 1) as usize,
        )
    } else {
        (
            *all_off.get_unchecked(v_idx) as usize,
            *p_bound.get_unchecked(v_idx) as usize,
        )
    };
    let (ds, de) = if was_true {
        (
            *all_off.get_unchecked(v_idx) as usize,
            *p_bound.get_unchecked(v_idx) as usize,
        )
    } else {
        (
            *p_bound.get_unchecked(v_idx) as usize,
            *all_off.get_unchecked(v_idx + 1) as usize,
        )
    };

    for k in is..ie {
        let c = *all_data.get_unchecked(k) as usize;
        let good = *num_good.get_unchecked(c);
        if good == 0 {
            debug_assert!(*unsat_count > 0);
            *unsat_count -= 1;
        }
        debug_assert!(good < 3);
        *num_good.get_unchecked_mut(c) = good + 1;
    }
    let mut residual_pushes = 0usize;
    for k in ds..de {
        let c = *all_data.get_unchecked(k) as usize;
        let ng = num_good.get_unchecked_mut(c);
        debug_assert!(*ng > 0);
        let new_val = *ng - 1;
        *ng = new_val;
        if new_val == 0 {
            *unsat_count += 1;
            residual.push(c as u32);
            residual_pushes += 1;
        }
    }

    *vars.get_unchecked_mut(v_idx) = !was_true;
    residual_pushes
}
