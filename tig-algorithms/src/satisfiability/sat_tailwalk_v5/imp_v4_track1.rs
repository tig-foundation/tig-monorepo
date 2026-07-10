use super::Hyperparameters;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

#[inline(always)]
fn lit_var_index(lit: i32) -> usize {
    if lit > 0 {
        lit as usize - 1
    } else {
        (-lit) as usize - 1
    }
}

#[inline(always)]
fn select_weighted3_u32(
    rand_val: u32,
    total_weight: u32,
    cnt: usize,
    w0: u32,
    v0: usize,
    w1: u32,
    v1: usize,
    w2: u32,
    v2: usize,
) -> usize {
    let mut r = rand_val % total_weight.max(1);
    if r < w0 {
        return v0;
    }
    if cnt > 1 {
        r -= w0;
        if r < w1 {
            return v1;
        }
        if cnt > 2 {
            r -= w1;
            if r < w2 {
                return v2;
            }
        }
    }
    v0
}

const N5000_DEFAULT_TAIL_EXTENSION_FUEL: f64 = 25_000_000_000.0;

fn track1_tail_extension_flips(
    hp: &Hyperparameters,
    nv: usize,
    best_unsat: usize,
    flip_fuel: f64,
    already_extended: bool,
) -> usize {
    if already_extended
        || nv != 5_000
        || best_unsat == 0
        || best_unsat > hp.target_tail_extend_max_unsat.unwrap_or(1)
        || flip_fuel <= 0.0
    {
        return 0;
    }

    let extension_fuel = hp
        .target_tail_extend_fuel
        .unwrap_or(N5000_DEFAULT_TAIL_EXTENSION_FUEL)
        .max(0.0);
    (extension_fuel / flip_fuel) as usize
}

#[inline(always)]
unsafe fn remove_unsat_clause_exact(unsat_list: &mut Vec<u32>, unsat_pos: &mut [u32], c: usize) {
    let pos = *unsat_pos.get_unchecked(c) as usize;
    let last_idx = unsat_list.len() - 1;
    if pos != last_idx {
        let last_c = *unsat_list.get_unchecked(last_idx) as usize;
        *unsat_list.get_unchecked_mut(pos) = last_c as u32;
        *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
    }
    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
    unsat_list.pop();
}

pub(crate) fn solve(
    hp: &Hyperparameters,
    rng: &mut SmallRng,
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
    let max_fuel = hp.target_max_fuel.unwrap_or(160_000_000_000.0);

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let mut max_flips = if flip_fuel > 0.0 {
        (remaining / flip_fuel) as usize
    } else {
        0
    };
    let base_max_flips = max_flips;
    let tail_cut_round = hp
        .target_tail_cut_fuel
        .filter(|fuel| *fuel > base_fuel && *fuel < max_fuel && flip_fuel > 0.0)
        .map(|fuel| ((fuel - base_fuel) / flip_fuel) as usize)
        .filter(|round| *round < max_flips);
    let tail_cut_unsat_threshold = hp.target_tail_cut_unsat_threshold.unwrap_or(usize::MAX);
    let tail_cut_best_unsat_threshold = hp
        .target_tail_cut_best_unsat_threshold
        .unwrap_or(usize::MAX);
    let trace_enabled =
        hp.target_trace_5000.unwrap_or(false) && nv == 5_000 && (4.25..4.28).contains(&density);

    let nad = 1.0;
    let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };
    let mut vars = Vec::with_capacity(nv);
    assignment_counts_into(&mut vars, &p_cnt, &n_cnt, rng, random_threshold, nad);

    let mut num_good = vec![0u8; nc];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(super::initial_residual_capacity(nc));
    let mut unsat_pos = vec![u32::MAX; nc];
    debug_assert_eq!(
        all_three_clauses,
        track1_clause_offsets_are_three(nc, co, cl.len())
    );
    if all_three_clauses {
        debug_assert!(co.len() > nc);
        debug_assert_eq!(co[0], 0);
        debug_assert_eq!(co[nc] as usize, cl.len());
        debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));
    }

    unsafe {
        rebuild_track1_state(
            nc,
            co,
            cl,
            &vars,
            &mut num_good,
            &mut unsat_list,
            &mut unsat_pos,
            all_three_clauses,
        );
    }

    let initial_unsat = unsat_list.len();
    if unsat_list.is_empty() {
        if trace_enabled {
            eprintln!(
                "c001_n5000_trace_diag max_fuel={:.0} base_fuel={:.3} flip_fuel={:.6} max_flips={} tail_cut_round={} tail_cut_triggered=false initial_unsat={} final_unsat=0 best_unsat=0 rounds=0 reinit_count=0 stagnation_ticks=0 kick_flips=0 solved=true",
                max_fuel,
                base_fuel,
                flip_fuel,
                max_flips,
                tail_cut_round
                    .map(|round| round.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                initial_unsat
            );
        }
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 {
        15.0
    } else {
        25.0
    };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.check_interval.unwrap_or(
        (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0)))
            .max(min_interval) as usize,
    );

    let mut last_check_residual = unsat_list.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit_t4 = hp.stagnation_limit.unwrap_or(3);

    let probs_break: [u32; 16] = [
        2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7,
    ];

    let mut current_reinit_stagnation: usize = ((nv * nc / 1000) as usize).clamp(1000, 500000);
    const REINIT_MIN_UNSAT: usize = 10;
    const MAX_REINITS: usize = 50;

    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;
    let mut trace_tail_cut_triggered = false;
    let mut trace_stagnation_ticks = 0usize;
    let mut trace_kick_flips = 0usize;
    let mut trace_tail_extension_flips = 0usize;
    let mut tail_extension_started = false;
    let mut check_countdown = check_interval;
    let mut check_due = false;

    unsafe {
        loop {
            if rounds >= max_flips {
                let extension_flips = track1_tail_extension_flips(
                    hp,
                    nv,
                    best_unsat,
                    flip_fuel,
                    tail_extension_started,
                );
                if extension_flips == 0 {
                    break;
                }

                tail_extension_started = true;
                trace_tail_extension_flips = extension_flips;
                max_flips = max_flips.saturating_add(extension_flips);
                vars.copy_from_slice(&best_vars);
                clear_current_unsat_positions(&unsat_list, &mut unsat_pos);
                unsat_list.clear();
                rebuild_track1_state(
                    nc,
                    co,
                    cl,
                    &vars,
                    &mut num_good,
                    &mut unsat_list,
                    &mut unsat_pos,
                    all_three_clauses,
                );
                debug_assert_eq!(unsat_list.len(), best_unsat);
                last_check_residual = unsat_list.len();
                stagnation = 0;
                stagnation_count = 0;
                continue;
            }
            if unsat_list.is_empty() {
                break;
            }
            if let Some(cut_round) = tail_cut_round {
                if rounds >= cut_round
                    && unsat_list.len() > tail_cut_unsat_threshold
                    && best_unsat > tail_cut_best_unsat_threshold
                {
                    trace_tail_cut_triggered = true;
                    break;
                }
            }

            if reinit_count >= MAX_REINITS
                && stagnation_count >= current_reinit_stagnation
                && best_unsat >= REINIT_MIN_UNSAT
            {
                break;
            }

            if stagnation_count >= current_reinit_stagnation
                && best_unsat >= REINIT_MIN_UNSAT
                && reinit_count < MAX_REINITS
            {
                reinit_count += 1;
                let reinit_factor = if density > 4.0 { 1.5 } else { 1.3 };
                current_reinit_stagnation = ((current_reinit_stagnation as f64 * reinit_factor)
                    as usize)
                    .clamp(1000, 500000);

                assignment_counts_into(&mut vars, &p_cnt, &n_cnt, rng, random_threshold, nad);

                clear_current_unsat_positions(&unsat_list, &mut unsat_pos);
                unsat_list.clear();
                rebuild_track1_state(
                    nc,
                    co,
                    cl,
                    &vars,
                    &mut num_good,
                    &mut unsat_list,
                    &mut unsat_pos,
                    all_three_clauses,
                );

                best_unsat = unsat_list.len();
                if best_unsat > 0 {
                    best_vars.copy_from_slice(&vars);
                }
                stagnation_count = 0;
            }

            if check_due {
                check_due = false;
                let progress = last_check_residual as i64 - unsat_list.len() as i64;

                if progress <= 0 {
                    stagnation += 1;
                    trace_stagnation_ticks += 1;

                    if stagnation >= stagnation_limit_t4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if unsat_list.is_empty() {
                                break;
                            }

                            let rid = rng.gen::<usize>() % unsat_list.len();
                            let pcid = *unsat_list.get_unchecked(rid) as usize;
                            let (pcs, pce) =
                                track1_clause_bounds_unchecked(pcid, co, all_three_clauses);
                            if pcs == pce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = lit_var_index(lit);

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (
                                    *p_bound.get_unchecked(v) as usize,
                                    *all_off.get_unchecked(v + 1) as usize,
                                )
                            } else {
                                (
                                    *all_off.get_unchecked(v) as usize,
                                    *p_bound.get_unchecked(v) as usize,
                                )
                            };
                            let (ds, de) = if was_true {
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

                            for k in is..ie {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = *num_good.get_unchecked(c);
                                if ng == 0 {
                                    remove_unsat_clause_exact(&mut unsat_list, &mut unsat_pos, c);
                                }
                                *num_good.get_unchecked_mut(c) = ng + 1;
                            }

                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = *num_good.get_unchecked(c);
                                *num_good.get_unchecked_mut(c) = ng - 1;
                                if ng == 1 {
                                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                                    unsat_list.push(c as u32);
                                }
                            }
                            *vars.get_unchecked_mut(v) = !was_true;
                            trace_kick_flips += 1;
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }

                last_check_residual = unsat_list.len();
            }

            if unsat_list.is_empty() {
                break;
            }

            let rand_val = rng.gen::<usize>();

            let cid = {
                let uc = unsat_list.len();
                let i1 = (rand_val as u32 as usize) % uc;
                *unsat_list.get_unchecked(i1) as usize
            };

            let v_idx = if all_three_clauses {
                choose_track1_var3(
                    rand_val,
                    cid,
                    cl,
                    &vars,
                    all_off,
                    p_bound,
                    all_data,
                    &num_good,
                    &probs_break,
                )
            } else {
                choose_track1_var_generic(
                    rand_val,
                    cid,
                    co,
                    cl,
                    &vars,
                    all_off,
                    p_bound,
                    all_data,
                    &num_good,
                    &probs_break,
                    all_three_clauses,
                )
            };

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
                let ng = *num_good.get_unchecked(c);
                if ng == 0 {
                    remove_unsat_clause_exact(&mut unsat_list, &mut unsat_pos, c);
                }
                *num_good.get_unchecked_mut(c) = ng + 1;
            }

            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = *num_good.get_unchecked(c);
                *num_good.get_unchecked_mut(c) = ng - 1;
                if ng == 1 {
                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                    unsat_list.push(c as u32);
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            rounds += 1;
            check_due = super::advance_interval_due(&mut check_countdown, check_interval);

            let cur = unsat_list.len();
            if cur < best_unsat {
                best_unsat = cur;
                if cur > 0 {
                    best_vars.copy_from_slice(&vars);
                }
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }
        }
    }

    let final_vars = if unsat_list.is_empty() {
        vars
    } else {
        best_vars
    };
    if trace_enabled {
        eprintln!(
            "c001_n5000_trace_diag max_fuel={:.0} base_fuel={:.3} flip_fuel={:.6} base_max_flips={} max_flips={} tail_extend={} tail_extend_flips={} tail_cut_round={} tail_cut_triggered={} initial_unsat={} final_unsat={} best_unsat={} rounds={} reinit_count={} stagnation_ticks={} kick_flips={} solved={}",
            max_fuel,
            base_fuel,
            flip_fuel,
            base_max_flips,
            max_flips,
            tail_extension_started,
            trace_tail_extension_flips,
            tail_cut_round
                .map(|round| round.to_string())
                .unwrap_or_else(|| "none".to_string()),
            trace_tail_cut_triggered,
            initial_unsat,
            unsat_list.len(),
            best_unsat,
            rounds,
            reinit_count,
            trace_stagnation_ticks,
            trace_kick_flips,
            unsat_list.is_empty()
        );
    }
    let _ = save_solution(&Solution {
        variables: final_vars,
    });

    Ok(())
}

#[inline(always)]
unsafe fn track1_break_sad(
    v: usize,
    vars: &[bool],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    num_good: &[u8],
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
        }
    }
    sad
}

#[inline(always)]
unsafe fn choose_track1_var_generic(
    rand_val: usize,
    cid: usize,
    co: &[u32],
    cl: &mut [i32],
    vars: &[bool],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    num_good: &[u8],
    probs_break: &[u32; 16],
    all_three_clauses: bool,
) -> usize {
    let (cs, ce) = track1_clause_bounds_unchecked(cid, co, all_three_clauses);
    let clen = ce - cs;

    if clen > 1 {
        let ri = rand_val % clen;
        if ri != 0 {
            cl.swap(cs, cs + ri);
        }
    }

    let mut zero0: usize = 0;
    let mut zero1: usize = 0;
    let mut zero2: usize = 0;
    let mut zero_cnt: usize = 0;
    let mut pw0 = 0u32;
    let mut pw1 = 0u32;
    let mut pw2 = 0u32;
    let mut pv0 = 0usize;
    let mut pv1 = 0usize;
    let mut pv2 = 0usize;
    let mut pw_cnt: usize = 0;
    let mut total_pw: u32 = 0;

    for j in cs..ce {
        let abs_l = lit_var_index(*cl.get_unchecked(j));
        let sad = track1_break_sad(abs_l, vars, all_off, p_bound, all_data, num_good);

        if sad == 0 {
            match zero_cnt {
                0 => zero0 = abs_l,
                1 => zero1 = abs_l,
                _ => zero2 = abs_l,
            }
            zero_cnt += 1;
            continue;
        }

        if zero_cnt == 0 {
            let pw = *probs_break.get_unchecked(sad.min(15));
            match pw_cnt {
                0 => {
                    pw0 = pw;
                    pv0 = abs_l;
                }
                1 => {
                    pw1 = pw;
                    pv1 = abs_l;
                }
                _ => {
                    pw2 = pw;
                    pv2 = abs_l;
                }
            }
            total_pw += pw;
            pw_cnt += 1;
        }
    }

    if zero_cnt > 0 {
        match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        }
    } else {
        select_weighted3_u32(
            rand_val as u32,
            total_pw,
            pw_cnt,
            pw0,
            pv0,
            pw1,
            pv1,
            pw2,
            pv2,
        )
    }
}

#[inline(always)]
unsafe fn choose_track1_var3(
    rand_val: usize,
    cid: usize,
    cl: &mut [i32],
    vars: &[bool],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    num_good: &[u8],
    probs_break: &[u32; 16],
) -> usize {
    let cs = cid * 3;
    let ri = rand_val % 3;
    if ri != 0 {
        cl.swap(cs, cs + ri);
    }

    let v0 = lit_var_index(*cl.get_unchecked(cs));
    let s0 = track1_break_sad(v0, vars, all_off, p_bound, all_data, num_good);
    let v1 = lit_var_index(*cl.get_unchecked(cs + 1));
    let s1 = track1_break_sad(v1, vars, all_off, p_bound, all_data, num_good);
    let v2 = lit_var_index(*cl.get_unchecked(cs + 2));
    let s2 = track1_break_sad(v2, vars, all_off, p_bound, all_data, num_good);

    let mut zero0: usize = 0;
    let mut zero1: usize = 0;
    let mut zero2: usize = 0;
    let mut zero_cnt = 0usize;
    if s0 == 0 {
        zero0 = v0;
        zero_cnt += 1;
    }
    if s1 == 0 {
        match zero_cnt {
            0 => zero0 = v1,
            1 => zero1 = v1,
            _ => zero2 = v1,
        }
        zero_cnt += 1;
    }
    if s2 == 0 {
        match zero_cnt {
            0 => zero0 = v2,
            1 => zero1 = v2,
            _ => zero2 = v2,
        }
        zero_cnt += 1;
    }

    if zero_cnt > 0 {
        return match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        };
    }

    let pw0 = *probs_break.get_unchecked(s0.min(15));
    let pw1 = *probs_break.get_unchecked(s1.min(15));
    let pw2 = *probs_break.get_unchecked(s2.min(15));
    select_weighted3_u32(
        rand_val as u32,
        pw0 + pw1 + pw2,
        3,
        pw0,
        v0,
        pw1,
        v1,
        pw2,
        v2,
    )
}

#[inline(always)]
fn is_lit_satisfied(vars: &[bool], lit: i32) -> bool {
    let v = lit_var_index(lit);
    (lit > 0 && vars[v]) || (lit < 0 && !vars[v])
}

#[inline(always)]
fn count_satisfied_clause(vars: &[bool], cl: &[i32], s: usize, e: usize) -> u8 {
    match e - s {
        1 => is_lit_satisfied(vars, cl[s]) as u8,
        2 => is_lit_satisfied(vars, cl[s]) as u8 + is_lit_satisfied(vars, cl[s + 1]) as u8,
        3 => {
            is_lit_satisfied(vars, cl[s]) as u8
                + is_lit_satisfied(vars, cl[s + 1]) as u8
                + is_lit_satisfied(vars, cl[s + 2]) as u8
        }
        _ => {
            let mut g = 0u8;
            for &lit in &cl[s..e] {
                if is_lit_satisfied(vars, lit) {
                    g += 1;
                }
            }
            g
        }
    }
}

#[inline(always)]
unsafe fn is_lit_satisfied_unchecked(vars: &[bool], lit: i32) -> bool {
    let v = lit_var_index(lit);
    (lit > 0 && *vars.get_unchecked(v)) || (lit < 0 && !*vars.get_unchecked(v))
}

#[inline(always)]
unsafe fn count_satisfied_clause_unchecked(vars: &[bool], cl: &[i32], s: usize, e: usize) -> u8 {
    match e - s {
        1 => is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8,
        2 => {
            is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8
                + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 1)) as u8
        }
        3 => {
            is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8
                + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 1)) as u8
                + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 2)) as u8
        }
        _ => {
            let mut g = 0u8;
            for j in s..e {
                if is_lit_satisfied_unchecked(vars, *cl.get_unchecked(j)) {
                    g += 1;
                }
            }
            g
        }
    }
}

#[inline(always)]
unsafe fn count_satisfied_clause3_unchecked(vars: &[bool], cl: &[i32], s: usize) -> u8 {
    is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s)) as u8
        + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 1)) as u8
        + is_lit_satisfied_unchecked(vars, *cl.get_unchecked(s + 2)) as u8
}

unsafe fn rebuild_track1_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    unsat_list: &mut Vec<u32>,
    unsat_pos: &mut [u32],
    all_three_clauses: bool,
) {
    debug_assert!(num_good.len() >= nc);
    debug_assert!(unsat_pos.len() >= nc);
    debug_assert!(unsat_list.is_empty());

    if all_three_clauses {
        debug_assert!(co.len() > nc);
        debug_assert_eq!(co[0], 0);
        debug_assert_eq!(co[nc] as usize, cl.len());
        debug_assert!((0..=nc).all(|i| co[i] as usize == i * 3));
        for c in 0..nc {
            let g = count_satisfied_clause3_unchecked(vars, cl, c * 3);
            num_good[c] = g;
            if g == 0 {
                unsat_pos[c] = unsat_list.len() as u32;
                unsat_list.push(c as u32);
            }
        }
        return;
    }

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let g = count_satisfied_clause_unchecked(vars, cl, s, e);
        num_good[c] = g;
        if g == 0 {
            unsat_pos[c] = unsat_list.len() as u32;
            unsat_list.push(c as u32);
        }
    }
}

#[inline(always)]
fn track1_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
    if co.len() <= nc || cl_len != nc.saturating_mul(3) {
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
unsafe fn track1_clause_bounds_unchecked(
    cid: usize,
    co: &[u32],
    all_three_clauses: bool,
) -> (usize, usize) {
    if all_three_clauses {
        let s = cid * 3;
        (s, s + 3)
    } else {
        (
            *co.get_unchecked(cid) as usize,
            *co.get_unchecked(cid + 1) as usize,
        )
    }
}

fn clear_current_unsat_positions(unsat_list: &[u32], unsat_pos: &mut [u32]) {
    for &cid in unsat_list {
        unsat_pos[cid as usize] = u32::MAX;
    }
}

fn assignment_counts_into(
    vars: &mut Vec<bool>,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    random_threshold: f64,
    nad: f64,
) {
    vars.clear();
    for v in 0..p_cnt.len() {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        let value;
        if nn == 0 && np > 0 {
            value = true;
            vars.push(value);
            continue;
        }
        if np == 0 && nn > 0 {
            value = false;
            vars.push(value);
            continue;
        }
        let vad = if nn > 0 {
            np as f64 / nn as f64
        } else {
            nad + 1.0
        };
        if vad <= nad {
            value = rng.gen_bool(random_threshold);
        } else {
            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            value = rng.gen_bool(prob);
        }
        vars.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn n5000_tail_extension_is_progress_gated_and_open_as_hp() {
        let hp = Hyperparameters::default();

        assert_eq!(
            track1_tail_extension_flips(&hp, 5_000, 1, 250.0, false),
            100_000_000
        );
        assert_eq!(track1_tail_extension_flips(&hp, 5_000, 2, 250.0, false), 0);
        assert_eq!(track1_tail_extension_flips(&hp, 7_500, 1, 250.0, false), 0);
        assert_eq!(track1_tail_extension_flips(&hp, 5_000, 1, 250.0, true), 0);

        let disabled = Hyperparameters {
            target_tail_extend_fuel: Some(0.0),
            ..Hyperparameters::default()
        };
        assert_eq!(
            track1_tail_extension_flips(&disabled, 5_000, 1, 250.0, false),
            0
        );
    }

    #[test]
    fn imp_v4_track1_lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn weighted3_selector_matches_array_reference() {
        fn reference(rand_val: u32, total: u32, cnt: usize, weights: [u32; 3]) -> usize {
            let vars = [3usize, 5, 8];
            let mut r = rand_val % total.max(1);
            let mut chosen = vars[0];
            for i in 0..cnt {
                if r < weights[i] {
                    chosen = vars[i];
                    break;
                }
                r -= weights[i];
            }
            chosen
        }

        for weights in [[0_u32, 0, 0], [7, 0, 5], [2, 3, 4], [2535, 551, 233]] {
            for cnt in 1usize..=3 {
                let total = weights[..cnt].iter().copied().sum::<u32>();
                for rand_val in 0u32..64 {
                    assert_eq!(
                        select_weighted3_u32(
                            rand_val, total, cnt, weights[0], 3, weights[1], 5, weights[2], 8,
                        ),
                        reference(rand_val, total, cnt, weights)
                    );
                }
            }
        }
    }

    #[test]
    fn choose_track1_var3_matches_generic_and_swap_side_effect() {
        let co = [0_u32, 3];
        let base_cl = vec![1, -2, 3];
        let vars = vec![true, false, true];
        let all_off = [0_u32, 2, 4, 6];
        let p_bound = [1_u32, 3, 5];
        let all_data = [0_u32, 3, 0, 2, 1, 3];
        let probs_break: [u32; 16] = [
            2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7,
        ];
        let num_good_cases = [
            vec![1_u8, 1, 1, 2],
            vec![2_u8, 1, 1, 2],
            vec![2_u8, 2, 1, 2],
            vec![2_u8, 2, 2, 2],
        ];

        for num_good in num_good_cases {
            for rand_val in 0usize..48 {
                let mut generic_cl = base_cl.clone();
                let mut fast_cl = base_cl.clone();
                let generic = unsafe {
                    choose_track1_var_generic(
                        rand_val,
                        0,
                        &co,
                        &mut generic_cl,
                        &vars,
                        &all_off,
                        &p_bound,
                        &all_data,
                        &num_good,
                        &probs_break,
                        true,
                    )
                };
                let fast = unsafe {
                    choose_track1_var3(
                        rand_val,
                        0,
                        &mut fast_cl,
                        &vars,
                        &all_off,
                        &p_bound,
                        &all_data,
                        &num_good,
                        &probs_break,
                    )
                };
                assert_eq!(fast, generic, "rand_val={rand_val} num_good={num_good:?}");
                assert_eq!(
                    fast_cl, generic_cl,
                    "rand_val={rand_val} num_good={num_good:?}"
                );
            }
        }
    }

    #[test]
    fn count_satisfied_clause_matches_generic_reference() {
        let vars = vec![true, false, true, false, true];
        let cl = vec![1, -2, 3, -1, 2, 4, -5, 5, -4, 3];

        for (s, e) in [(0, 0), (0, 1), (1, 3), (0, 3), (3, 7), (7, 10)] {
            let mut expected = 0u8;
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    expected += 1;
                }
            }

            assert_eq!(count_satisfied_clause(&vars, &cl, s, e), expected);
            assert_eq!(
                unsafe { count_satisfied_clause_unchecked(&vars, &cl, s, e) },
                expected
            );
        }
    }

    #[test]
    fn rebuild_track1_state_all_three_fast_path_matches_generic_reference() {
        let nc = 7;
        let co = [0_u32, 3, 6, 9, 12, 15, 18, 21];
        let cl = [
            1, 2, 3, -1, -2, -3, 1, -2, 4, -1, 2, -4, 3, 4, -2, -3, -4, 2, 1, -3, -4,
        ];
        let vars = [true, false, true, false];

        let mut expected_good = vec![0_u8; nc];
        let mut expected_unsat = Vec::new();
        let mut expected_pos = vec![u32::MAX; nc];
        for c in 0..nc {
            let good = count_satisfied_clause(&vars, &cl, co[c] as usize, co[c + 1] as usize);
            expected_good[c] = good;
            if good == 0 {
                expected_pos[c] = expected_unsat.len() as u32;
                expected_unsat.push(c as u32);
            }
        }

        let mut actual_good = vec![9_u8; nc];
        let mut actual_unsat = Vec::new();
        let mut actual_pos = vec![u32::MAX; nc];
        unsafe {
            rebuild_track1_state(
                nc,
                &co,
                &cl,
                &vars,
                &mut actual_good,
                &mut actual_unsat,
                &mut actual_pos,
                true,
            );
        }

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_unsat, expected_unsat);
        assert_eq!(actual_pos, expected_pos);
    }

    #[test]
    fn track1_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9, 12];
        assert!(track1_clause_offsets_are_three(4, &co, 12));
        for cid in 0..4 {
            assert_eq!(
                unsafe { track1_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn track1_clause_offsets_reject_mixed_lengths_with_average_three() {
        let co = [0_u32, 2, 5, 9];
        assert!(!track1_clause_offsets_are_three(3, &co, 9));
        assert_eq!(
            unsafe { track1_clause_bounds_unchecked(1, &co, false) },
            (2, 5)
        );
    }

    #[test]
    fn assignment_counts_into_matches_two_pass_reference() {
        let p_cnt = vec![0, 7, 0, 9, 2, 3, 11];
        let n_cnt = vec![0, 0, 8, 3, 5, 3, 1];
        let random_threshold = 0.003;
        let nad = 1.0;
        let mut old_vars = vec![true; p_cnt.len()];
        let mut new_vars = old_vars.clone();
        let mut old_rng = SmallRng::seed_from_u64(23);
        let mut new_rng = SmallRng::seed_from_u64(23);

        for v in 0..old_vars.len() {
            old_vars[v] = false;
        }
        for v in 0..old_vars.len() {
            let np = p_cnt[v] as usize;
            let nn = n_cnt[v] as usize;
            if nn == 0 && np > 0 {
                old_vars[v] = true;
                continue;
            }
            if np == 0 && nn > 0 {
                continue;
            }
            let vad = if nn > 0 {
                np as f64 / nn as f64
            } else {
                nad + 1.0
            };
            if vad <= nad {
                old_vars[v] = old_rng.gen_bool(random_threshold);
            } else {
                let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                old_vars[v] = old_rng.gen_bool(prob);
            }
        }

        let reused_capacity = new_vars.capacity();
        assignment_counts_into(
            &mut new_vars,
            &p_cnt,
            &n_cnt,
            &mut new_rng,
            random_threshold,
            nad,
        );

        assert_eq!(new_vars, old_vars);
        assert_eq!(new_vars.capacity(), reused_capacity);
    }

    #[test]
    fn clear_current_unsat_positions_only_resets_live_unsat_entries() {
        let unsat_list = vec![4_u32, 1, 7];
        let mut unsat_pos = vec![u32::MAX; 9];
        unsat_pos[1] = 1;
        unsat_pos[3] = 99;
        unsat_pos[4] = 0;
        unsat_pos[7] = 2;

        clear_current_unsat_positions(&unsat_list, &mut unsat_pos);

        assert_eq!(unsat_pos[1], u32::MAX);
        assert_eq!(unsat_pos[4], u32::MAX);
        assert_eq!(unsat_pos[7], u32::MAX);
        assert_eq!(unsat_pos[0], u32::MAX);
        assert_eq!(unsat_pos[3], 99);
        assert_eq!(unsat_pos[8], u32::MAX);
    }

    #[test]
    fn remove_unsat_clause_exact_handles_moved_and_last_entries() {
        let mut unsat_list = vec![2_u32, 5, 7];
        let mut unsat_pos = vec![u32::MAX; 8];
        unsat_pos[2] = 0;
        unsat_pos[5] = 1;
        unsat_pos[7] = 2;

        unsafe {
            remove_unsat_clause_exact(&mut unsat_list, &mut unsat_pos, 7);
        }

        assert_eq!(unsat_list, vec![2_u32, 5]);
        assert_eq!(unsat_pos[2], 0);
        assert_eq!(unsat_pos[5], 1);
        assert_eq!(unsat_pos[7], u32::MAX);

        unsafe {
            remove_unsat_clause_exact(&mut unsat_list, &mut unsat_pos, 2);
        }

        assert_eq!(unsat_list, vec![5_u32]);
        assert_eq!(unsat_pos[2], u32::MAX);
        assert_eq!(unsat_pos[5], 0);
        assert_eq!(unsat_pos[7], u32::MAX);
    }
}
