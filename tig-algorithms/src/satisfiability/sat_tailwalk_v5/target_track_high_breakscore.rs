use super::{target_state, Hyperparameters};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

#[inline(always)]
fn initial_residual_capacity(nc: usize) -> usize {
    (nc / 4).saturating_add(16)
}

fn default_max_fuel(nv: usize, density: f64) -> f64 {
    if nv == 10_000 && density >= 4.24 {
        125_000_000_000.0
    } else if nv == 7_500 && density >= 4.24 {
        215_000_000_000.0
    } else if nv >= 10000 {
        120_000_000_000.0
    } else {
        250_000_000_000.0
    }
}

fn build_var_age_if_needed(enabled: bool, nv: usize) -> Vec<u16> {
    if enabled {
        vec![0u16; nv]
    } else {
        Vec::new()
    }
}

fn build_var_appearances(nv: usize, p_cnt: &[u32], n_cnt: &[u32]) -> Vec<u32> {
    let mut appearances = Vec::with_capacity(nv);
    for v in 0..nv {
        appearances.push(p_cnt[v] + n_cnt[v]);
    }
    appearances
}

fn build_var_appearances_with_pair_bound(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
) -> (Vec<u32>, usize) {
    let mut appearances = Vec::with_capacity(nv);
    let mut max_occ = 0usize;
    for v in 0..nv {
        let p = p_cnt[v];
        let n = n_cnt[v];
        max_occ = max_occ.max(p as usize).max(n as usize);
        appearances.push(p + n);
    }
    (appearances, max_occ)
}

#[inline(always)]
fn select_weighted3_f64_sub_le(
    mut threshold: f64,
    cnt: usize,
    w0: f64,
    v0: usize,
    w1: f64,
    v1: usize,
    w2: f64,
    v2: usize,
) -> usize {
    threshold -= w0;
    if threshold <= 0.0 {
        return v0;
    }
    if cnt > 1 {
        threshold -= w1;
        if threshold <= 0.0 {
            return v1;
        }
        if cnt > 2 {
            threshold -= w2;
            if threshold <= 0.0 {
                return v2;
            }
        }
    }
    v0
}

fn build_probsat_break_weights(avg_clause_size: f64, p_cnt: &[u32], n_cnt: &[u32]) -> Vec<f64> {
    build_probsat_break_weights_from_bound(avg_clause_size, max_pair_occurrence_bound(p_cnt, n_cnt))
}

fn build_probsat_break_weights_from_bound(
    avg_clause_size: f64,
    occurrence_bound: usize,
) -> Vec<f64> {
    let limit = occurrence_bound.min(255);
    let cb: f64 = if avg_clause_size <= 3.2 {
        2.06
    } else if avg_clause_size <= 4.2 {
        2.85
    } else if avg_clause_size <= 5.2 {
        3.7
    } else if avg_clause_size <= 6.2 {
        5.1
    } else {
        5.4
    };

    let mut weights = Vec::with_capacity(limit + 1);
    for i in 0..=limit {
        weights.push((i as f64 + 1.0).powf(-cb));
    }
    weights
}

fn max_pair_occurrence_bound(p_cnt: &[u32], n_cnt: &[u32]) -> usize {
    let len = p_cnt.len().min(n_cnt.len());
    let mut max_occ = 0usize;
    for i in 0..len {
        max_occ = max_occ.max(p_cnt[i] as usize).max(n_cnt[i] as usize);
    }
    max_occ
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

#[inline(always)]
unsafe fn bump_high_breakscore_var_age(var: usize, var_age: &mut [u16]) {
    let age = var_age.get_unchecked_mut(var);
    *age = age.saturating_add(1);
}

#[inline(always)]
unsafe fn bump_high_breakscore_clause_var_ages(
    cs: usize,
    ce: usize,
    cl: &[i32],
    var_age: &mut [u16],
) {
    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        bump_high_breakscore_var_age(lit_var_index(lit), var_age);
    }
}

#[inline(always)]
unsafe fn bump_high_breakscore_clause_var_ages3(cs: usize, cl: &[i32], var_age: &mut [u16]) {
    debug_assert!(cs + 2 < cl.len());
    bump_high_breakscore_var_age(lit_var_index(*cl.get_unchecked(cs)), var_age);
    bump_high_breakscore_var_age(lit_var_index(*cl.get_unchecked(cs + 1)), var_age);
    bump_high_breakscore_var_age(lit_var_index(*cl.get_unchecked(cs + 2)), var_age);
}

pub fn solve(
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
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let max_fuel = hp.target_max_fuel.unwrap_or(default_max_fuel(nv, density));
    let use_probsat_pick = nv == 10_000 && density >= 4.24;
    let (var_appearances, probsat_occurrence_bound) = if use_probsat_pick {
        build_var_appearances_with_pair_bound(nv, &p_cnt, &n_cnt)
    } else {
        (build_var_appearances(nv, &p_cnt, &n_cnt), 0)
    };

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 {
        (remaining / flip_fuel) as usize
    } else {
        0
    };
    let all_three_clauses = high_breakscore_clause_offsets_are_three(nc, co, cl.len());
    let use_greedy_init = target_greedy_init_enabled(nv, density, seed_key, hp);
    let (greedy_clause_order, greedy_all_three_clauses) = if use_greedy_init {
        if all_three_clauses {
            (Vec::new(), true)
        } else {
            build_greedy_clause_order_with_all_three(nc, co, cl.len())
        }
    } else {
        (Vec::new(), false)
    };
    debug_assert!(!greedy_all_three_clauses || all_three_clauses);

    let mut vars = Vec::with_capacity(nv);
    if use_greedy_init {
        initial_assignment_greedy_shortest_with_order_into(
            nv,
            &p_cnt,
            &n_cnt,
            co,
            cl,
            &greedy_clause_order,
            greedy_all_three_clauses,
            rng,
            &mut vars,
        );
    } else {
        initial_assignment_counts_into(nv, &p_cnt, &n_cnt, rng, hp, seed_key, &mut vars);
    }

    let mut num_good = vec![0u8; nc];
    let mut break_score = vec![0u16; nv];
    let mut make_score = vec![0u16; nv];
    let mut sat_xor = vec![0u32; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(initial_residual_capacity(nc));
    let mut residual_pos = vec![u32::MAX; nc];
    target_state::rebuild_u8_exact_with_make_fresh(
        nc,
        co,
        cl,
        &vars,
        &mut num_good,
        &mut residual,
        &mut residual_pos,
        &mut break_score,
        &mut sat_xor,
        &mut make_score,
        all_three_clauses,
    );

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob: f64 = if use_probsat_pick {
        0.0
    } else {
        hp.target_base_prob.unwrap_or(0.52)
    };
    let mut current_prob = base_prob;
    let max_random_prob: f64 = if use_probsat_pick {
        0.0
    } else {
        hp.max_prob.unwrap_or(0.9)
    };
    let prob_adjustment_factor: f64 = 0.025;
    let smoothing_factor: f64 = 0.8;

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

    let mut last_check_residual = residual.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let mut check_countdown = check_interval;
    let mut check_due = false;
    let mut var_age = build_var_age_if_needed(!use_probsat_pick, nv);
    let probsat_break = if use_probsat_pick {
        build_probsat_break_weights_from_bound(avg_clause_size, probsat_occurrence_bound)
    } else {
        Vec::new()
    };
    let probsat_break_limit = probsat_break.len().saturating_sub(1);
    let restart_interval = if nv == 10_000 && density >= 4.24 {
        hp.restart_interval.unwrap_or(8_000_000).max(1)
    } else {
        usize::MAX
    };
    let restart_limit = if restart_interval == usize::MAX {
        0
    } else {
        12
    };
    let mut restart_count = 0usize;
    let mut restart_countdown = restart_interval;
    let mut restart_due = false;
    let mut best_unsat = residual.len();
    let mut best_vars = vars.clone();
    let mut rounds_since_best = 0usize;
    let disable_make_score = hp.disable_make_score.unwrap_or(false);

    unsafe {
        loop {
            if rounds >= max_flips {
                break;
            }
            if residual.is_empty() {
                break;
            }

            if restart_due {
                restart_due = false;
                if restart_count < restart_limit && !residual.is_empty() {
                    restart_count += 1;
                    let elite_rewind =
                        best_unsat <= 24 && rounds_since_best >= restart_interval / 4;
                    if elite_rewind {
                        vars.clone_from(&best_vars);
                        let flips = (nv / 192).clamp(24, 96);
                        for _ in 0..flips {
                            let v = rng.gen::<usize>() % nv;
                            vars[v] = !vars[v];
                        }
                    } else if use_greedy_init {
                        initial_assignment_greedy_shortest_with_order_into(
                            nv,
                            &p_cnt,
                            &n_cnt,
                            co,
                            cl,
                            &greedy_clause_order,
                            greedy_all_three_clauses,
                            rng,
                            &mut vars,
                        );
                    } else {
                        initial_assignment_counts_into(
                            nv, &p_cnt, &n_cnt, rng, hp, seed_key, &mut vars,
                        );
                    }
                    target_state::rebuild_u8_exact_with_make(
                        nc,
                        co,
                        cl,
                        &vars,
                        &mut num_good,
                        &mut residual,
                        &mut residual_pos,
                        &mut break_score,
                        &mut sat_xor,
                        &mut make_score,
                        all_three_clauses,
                    );
                    if !use_probsat_pick {
                        var_age.fill(0);
                    }
                    last_check_residual = residual.len();
                    if !use_probsat_pick {
                        current_prob = base_prob;
                    }
                    stagnation = 0;
                    rounds_since_best = 0;
                    if residual.len() < best_unsat {
                        best_unsat = residual.len();
                        if best_unsat > 0 {
                            best_vars.clone_from(&vars);
                        }
                    }
                    if residual.is_empty() {
                        break;
                    }
                }
            }

            if check_due {
                check_due = false;
                let progress = last_check_residual as i64 - residual.len() as i64;

                if progress <= 0 {
                    stagnation += 1;
                    if !use_probsat_pick {
                        let prob_adjustment = prob_adjustment_factor
                            * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                        current_prob = (current_prob + prob_adjustment).min(max_random_prob);
                    }

                    if stagnation >= 4 {
                        for _ in 0..3 {
                            if residual.is_empty() {
                                break;
                            }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if *num_good.get_unchecked(pcid) > 0 {
                                target_state::remove_unsat_exact(
                                    &mut residual,
                                    &mut residual_pos,
                                    pcid,
                                );
                                continue;
                            }
                            let (pcs, pce) = high_breakscore_clause_bounds_unchecked(
                                pcid,
                                co,
                                all_three_clauses,
                            );
                            if pcs == pce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = lit_var_index(lit);

                            target_state::flip_u8_exact_with_make(
                                v,
                                &mut vars,
                                &mut num_good,
                                &mut sat_xor,
                                &mut break_score,
                                &mut make_score,
                                &mut residual,
                                &mut residual_pos,
                                co,
                                cl,
                                all_off,
                                p_bound,
                                all_data,
                                all_three_clauses,
                            );
                            if !use_probsat_pick {
                                *var_age.get_unchecked_mut(v) = 0;
                            }
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                    if !use_probsat_pick {
                        let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                        let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);
                        if progress_ratio > progress_threshold {
                            current_prob = base_prob;
                        } else {
                            current_prob = current_prob * smoothing_factor
                                + base_prob * (1.0 - smoothing_factor);
                        }
                    }
                }

                last_check_residual = residual.len();
            }

            if residual.is_empty() {
                break;
            }

            let rand_val = rng.gen::<usize>();
            let cid = *residual.get_unchecked(rand_val % residual.len()) as usize;

            let (cs, ce) = high_breakscore_clause_bounds_unchecked(cid, co, all_three_clauses);
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                if ri != 0 {
                    cl.swap(cs, cs + ri);
                }
            }

            let v_idx = if use_probsat_pick {
                if all_three_clauses {
                    choose_high_breakscore_probsat_var3(
                        cs,
                        cl,
                        &break_score,
                        &make_score,
                        &var_appearances,
                        &probsat_break,
                        probsat_break_limit,
                        disable_make_score,
                        rng,
                    )
                } else {
                    choose_high_breakscore_probsat_var_generic(
                        cs,
                        ce,
                        cl,
                        &break_score,
                        &make_score,
                        &var_appearances,
                        &probsat_break,
                        probsat_break_limit,
                        disable_make_score,
                        rng,
                    )
                }
            } else {
                if all_three_clauses {
                    choose_high_breakscore_non_probsat_var3(
                        cs,
                        cl,
                        &break_score,
                        &make_score,
                        &var_appearances,
                        &var_age,
                        current_prob,
                        disable_make_score,
                        rng,
                    )
                } else {
                    choose_high_breakscore_non_probsat_var_generic(
                        cs,
                        ce,
                        cl,
                        &break_score,
                        &make_score,
                        &var_appearances,
                        &var_age,
                        current_prob,
                        disable_make_score,
                        rng,
                    )
                }
            };

            target_state::flip_u8_exact_with_make(
                v_idx,
                &mut vars,
                &mut num_good,
                &mut sat_xor,
                &mut break_score,
                &mut make_score,
                &mut residual,
                &mut residual_pos,
                co,
                cl,
                all_off,
                p_bound,
                all_data,
                all_three_clauses,
            );
            if !use_probsat_pick {
                *var_age.get_unchecked_mut(v_idx) = 0;
                if all_three_clauses {
                    bump_high_breakscore_clause_var_ages3(cs, cl, &mut var_age);
                } else {
                    bump_high_breakscore_clause_var_ages(cs, ce, cl, &mut var_age);
                }
            }

            rounds += 1;
            check_due = super::advance_interval_due(&mut check_countdown, check_interval);
            if restart_count < restart_limit {
                restart_due = super::advance_interval_due(&mut restart_countdown, restart_interval);
            }
            let cur_unsat = residual.len();
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                if cur_unsat > 0 {
                    best_vars.clone_from(&vars);
                }
                rounds_since_best = 0;
            } else {
                rounds_since_best = rounds_since_best.saturating_add(1);
            }
        }
    }

    let final_vars = if residual.is_empty() { vars } else { best_vars };
    let _ = save_solution(&Solution {
        variables: final_vars,
    });
    Ok(())
}

#[inline(always)]
unsafe fn high_breakscore_make_value(
    v: usize,
    make_score: &[u16],
    disable_make_score: bool,
) -> usize {
    if disable_make_score {
        1
    } else {
        (*make_score.get_unchecked(v) as usize).max(1)
    }
}

#[inline(always)]
fn high_breakscore_better_zero(
    current: usize,
    current_make: usize,
    current_app: u32,
    candidate: usize,
    candidate_make: usize,
    candidate_app: u32,
) -> bool {
    current == usize::MAX
        || candidate_make > current_make
        || (candidate_make == current_make && candidate_app < current_app)
}

#[inline(always)]
unsafe fn high_breakscore_non_probsat_make_value(
    v: usize,
    make_score: &[u16],
    disable_make_score: bool,
) -> i64 {
    if disable_make_score {
        1
    } else {
        (*make_score.get_unchecked(v) as i64).max(1)
    }
}

#[inline(always)]
unsafe fn high_breakscore_non_probsat_zero_score(
    v: usize,
    make_score: &[u16],
    var_appearances: &[u32],
    var_age: &[u16],
    disable_make_score: bool,
) -> i64 {
    let mk = high_breakscore_non_probsat_make_value(v, make_score, disable_make_score);
    let appearances = *var_appearances.get_unchecked(v) as i64;
    let age_bonus = (*var_age.get_unchecked(v) as i64) / 4;
    appearances * 8 - mk * 2048 - age_bonus.min(128)
}

#[inline(always)]
unsafe fn high_breakscore_non_probsat_weight(
    v: usize,
    break_score: &[u16],
    make_score: &[u16],
    var_appearances: &[u32],
    var_age: &[u16],
    disable_make_score: bool,
) -> i64 {
    let sad = *break_score.get_unchecked(v) as i64;
    let mk = high_breakscore_non_probsat_make_value(v, make_score, disable_make_score);
    let appearances = *var_appearances.get_unchecked(v) as i64;
    let age_bonus = (*var_age.get_unchecked(v) as i64) / 2;
    sad * 4096 + appearances * 8 - mk * 512 - age_bonus.min(128)
}

#[inline(always)]
unsafe fn choose_high_breakscore_non_probsat_var_generic(
    cs: usize,
    ce: usize,
    cl: &[i32],
    break_score: &[u16],
    make_score: &[u16],
    var_appearances: &[u32],
    var_age: &[u16],
    current_prob: f64,
    disable_make_score: bool,
    rng: &mut SmallRng,
) -> usize {
    let mut zero_best = usize::MAX;
    let mut zero_best_score = i64::MAX;
    for j in cs..ce {
        let abs_l = lit_var_index(*cl.get_unchecked(j));
        if *break_score.get_unchecked(abs_l) != 0 {
            continue;
        }
        let score = high_breakscore_non_probsat_zero_score(
            abs_l,
            make_score,
            var_appearances,
            var_age,
            disable_make_score,
        );
        if zero_best == usize::MAX || score < zero_best_score {
            zero_best = abs_l;
            zero_best_score = score;
        }
    }

    if zero_best != usize::MAX {
        return zero_best;
    }

    if rng.gen::<f64>() < current_prob {
        return lit_var_index(*cl.get_unchecked(cs));
    }

    let mut v_min = lit_var_index(*cl.get_unchecked(cs));
    let mut best_weight = i64::MAX;

    for j in cs..ce {
        let abs_l = lit_var_index(*cl.get_unchecked(j));
        let combined_weight = high_breakscore_non_probsat_weight(
            abs_l,
            break_score,
            make_score,
            var_appearances,
            var_age,
            disable_make_score,
        );
        if combined_weight < best_weight {
            best_weight = combined_weight;
            v_min = abs_l;
        }
    }
    v_min
}

#[inline(always)]
unsafe fn choose_high_breakscore_non_probsat_var3(
    cs: usize,
    cl: &[i32],
    break_score: &[u16],
    make_score: &[u16],
    var_appearances: &[u32],
    var_age: &[u16],
    current_prob: f64,
    disable_make_score: bool,
    rng: &mut SmallRng,
) -> usize {
    let v0 = lit_var_index(*cl.get_unchecked(cs));
    let v1 = lit_var_index(*cl.get_unchecked(cs + 1));
    let v2 = lit_var_index(*cl.get_unchecked(cs + 2));

    let mut zero_best = usize::MAX;
    let mut zero_best_score = i64::MAX;

    if *break_score.get_unchecked(v0) == 0 {
        zero_best = v0;
        zero_best_score = high_breakscore_non_probsat_zero_score(
            v0,
            make_score,
            var_appearances,
            var_age,
            disable_make_score,
        );
    }
    if *break_score.get_unchecked(v1) == 0 {
        let score = high_breakscore_non_probsat_zero_score(
            v1,
            make_score,
            var_appearances,
            var_age,
            disable_make_score,
        );
        if zero_best == usize::MAX || score < zero_best_score {
            zero_best = v1;
            zero_best_score = score;
        }
    }
    if *break_score.get_unchecked(v2) == 0 {
        let score = high_breakscore_non_probsat_zero_score(
            v2,
            make_score,
            var_appearances,
            var_age,
            disable_make_score,
        );
        if zero_best == usize::MAX || score < zero_best_score {
            zero_best = v2;
        }
    }

    if zero_best != usize::MAX {
        return zero_best;
    }

    if rng.gen::<f64>() < current_prob {
        return v0;
    }

    let w0 = high_breakscore_non_probsat_weight(
        v0,
        break_score,
        make_score,
        var_appearances,
        var_age,
        disable_make_score,
    );
    let w1 = high_breakscore_non_probsat_weight(
        v1,
        break_score,
        make_score,
        var_appearances,
        var_age,
        disable_make_score,
    );
    let w2 = high_breakscore_non_probsat_weight(
        v2,
        break_score,
        make_score,
        var_appearances,
        var_age,
        disable_make_score,
    );

    let mut v_min = v0;
    let mut best_weight = w0;
    if w1 < best_weight {
        v_min = v1;
        best_weight = w1;
    }
    if w2 < best_weight {
        v_min = v2;
    }
    v_min
}

#[inline(always)]
unsafe fn choose_high_breakscore_probsat_var_generic(
    cs: usize,
    ce: usize,
    cl: &[i32],
    break_score: &[u16],
    make_score: &[u16],
    var_appearances: &[u32],
    probsat_break: &[f64],
    probsat_break_limit: usize,
    disable_make_score: bool,
    rng: &mut SmallRng,
) -> usize {
    let mut total_weight = 0.0;
    let mut w0 = 0.0f64;
    let mut w1 = 0.0f64;
    let mut w2 = 0.0f64;
    let mut v0 = 0usize;
    let mut v1 = 0usize;
    let mut v2 = 0usize;
    let limit = (ce - cs).min(3);
    let mut zero_best = usize::MAX;
    let mut zero_best_make = 0usize;
    let mut zero_best_app = u32::MAX;

    for idx in 0..limit {
        let abs_l = lit_var_index(*cl.get_unchecked(cs + idx));
        let sad = *break_score.get_unchecked(abs_l) as usize;
        let mk = high_breakscore_make_value(abs_l, make_score, disable_make_score);

        if sad == 0 {
            let app = *var_appearances.get_unchecked(abs_l);
            if high_breakscore_better_zero(zero_best, zero_best_make, zero_best_app, abs_l, mk, app)
            {
                zero_best = abs_l;
                zero_best_make = mk;
                zero_best_app = app;
            }
        } else if zero_best == usize::MAX {
            let make_boost = 1.0 + 0.30 * (mk.saturating_sub(1).min(8) as f64);
            let weight = *probsat_break.get_unchecked(sad.min(probsat_break_limit)) * make_boost;
            match idx {
                0 => {
                    w0 = weight;
                    v0 = abs_l;
                }
                1 => {
                    w1 = weight;
                    v1 = abs_l;
                }
                _ => {
                    w2 = weight;
                    v2 = abs_l;
                }
            }
            total_weight += weight;
        }
    }

    if zero_best != usize::MAX {
        zero_best
    } else {
        select_weighted3_f64_sub_le(
            rng.gen::<f64>() * total_weight,
            limit,
            w0,
            v0,
            w1,
            v1,
            w2,
            v2,
        )
    }
}

#[inline(always)]
unsafe fn choose_high_breakscore_probsat_var3(
    cs: usize,
    cl: &[i32],
    break_score: &[u16],
    make_score: &[u16],
    var_appearances: &[u32],
    probsat_break: &[f64],
    probsat_break_limit: usize,
    disable_make_score: bool,
    rng: &mut SmallRng,
) -> usize {
    let v0 = lit_var_index(*cl.get_unchecked(cs));
    let sad0 = *break_score.get_unchecked(v0) as usize;
    let mk0 = high_breakscore_make_value(v0, make_score, disable_make_score);
    let mut zero_best = usize::MAX;
    let mut zero_best_make = 0usize;
    let mut zero_best_app = u32::MAX;
    let mut w0 = 0.0f64;
    let mut w1 = 0.0f64;
    let mut w2 = 0.0f64;

    if sad0 == 0 {
        zero_best = v0;
        zero_best_make = mk0;
        zero_best_app = *var_appearances.get_unchecked(v0);
    } else {
        let make_boost = 1.0 + 0.30 * (mk0.saturating_sub(1).min(8) as f64);
        w0 = *probsat_break.get_unchecked(sad0.min(probsat_break_limit)) * make_boost;
    }

    let v1 = lit_var_index(*cl.get_unchecked(cs + 1));
    let sad1 = *break_score.get_unchecked(v1) as usize;
    let mk1 = high_breakscore_make_value(v1, make_score, disable_make_score);
    if sad1 == 0 {
        let app = *var_appearances.get_unchecked(v1);
        if high_breakscore_better_zero(zero_best, zero_best_make, zero_best_app, v1, mk1, app) {
            zero_best = v1;
            zero_best_make = mk1;
            zero_best_app = app;
        }
    } else if zero_best == usize::MAX {
        let make_boost = 1.0 + 0.30 * (mk1.saturating_sub(1).min(8) as f64);
        w1 = *probsat_break.get_unchecked(sad1.min(probsat_break_limit)) * make_boost;
    }

    let v2 = lit_var_index(*cl.get_unchecked(cs + 2));
    let sad2 = *break_score.get_unchecked(v2) as usize;
    let mk2 = high_breakscore_make_value(v2, make_score, disable_make_score);
    if sad2 == 0 {
        let app = *var_appearances.get_unchecked(v2);
        if high_breakscore_better_zero(zero_best, zero_best_make, zero_best_app, v2, mk2, app) {
            zero_best = v2;
        }
    } else if zero_best == usize::MAX {
        let make_boost = 1.0 + 0.30 * (mk2.saturating_sub(1).min(8) as f64);
        w2 = *probsat_break.get_unchecked(sad2.min(probsat_break_limit)) * make_boost;
    }

    if zero_best != usize::MAX {
        zero_best
    } else {
        select_weighted3_f64_sub_le(rng.gen::<f64>() * (w0 + w1 + w2), 3, w0, v0, w1, v1, w2, v2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn high_breakscore_lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn high_breakscore_var_age_only_builds_when_needed() {
        assert!(build_var_age_if_needed(false, 10_000).is_empty());
        assert_eq!(build_var_age_if_needed(true, 4), vec![0_u16; 4]);
    }

    #[test]
    fn bump_high_breakscore_clause_var_ages3_matches_generic_loop() {
        let cl = [99, 1, -2, 2, 100];
        let mut generic_age = [5_u16, u16::MAX - 1, 17];
        let mut fast_age = generic_age;

        unsafe {
            bump_high_breakscore_clause_var_ages(1, 4, &cl, &mut generic_age);
            bump_high_breakscore_clause_var_ages3(1, &cl, &mut fast_age);
        }

        assert_eq!(fast_age, generic_age);
        assert_eq!(fast_age, [6, u16::MAX, 17]);
    }

    #[test]
    fn high_breakscore_var_appearances_keep_u32_counts() {
        let p_cnt = vec![1_u32, 4, 0, 70_000];
        let n_cnt = vec![2_u32, 1, 300, 60_000];

        let appearances: Vec<u32> = build_var_appearances(4, &p_cnt, &n_cnt);

        assert_eq!(appearances, vec![3_u32, 5, 300, 130_000]);
    }

    #[test]
    fn high_breakscore_appearances_pair_bound_matches_separate_scans() {
        let p_cnt = vec![1_u32, 4, 0, 70_000];
        let n_cnt = vec![2_u32, 1, 300, 60_000];

        let (appearances, bound) = build_var_appearances_with_pair_bound(4, &p_cnt, &n_cnt);

        assert_eq!(appearances, build_var_appearances(4, &p_cnt, &n_cnt));
        assert_eq!(bound, max_pair_occurrence_bound(&p_cnt, &n_cnt));
    }

    #[test]
    fn high_breakscore_probsat_break_weights_stop_at_occurrence_bound() {
        let p_cnt = vec![0_u32, 3, 9, 2];
        let n_cnt = vec![1_u32, 5, 4, 8];
        let cb: f64 = 3.7;

        let weights = build_probsat_break_weights(4.5, &p_cnt, &n_cnt);

        assert_eq!(weights.len(), 10);
        for (i, &weight) in weights.iter().enumerate() {
            assert_eq!(weight, (i as f64 + 1.0).powf(-cb));
        }
    }

    #[test]
    fn high_breakscore_pair_occurrence_bound_matches_zip_max_reference() {
        assert_eq!(
            max_pair_occurrence_bound(&[0_u32, 3, 9, 2], &[1_u32, 5, 4, 8]),
            9
        );
        assert_eq!(
            max_pair_occurrence_bound(&[0_u32, 300], &[1_u32, 5, 4]),
            300
        );
        assert_eq!(max_pair_occurrence_bound(&[0_u32], &[1_u32, 500]), 1);
        assert_eq!(max_pair_occurrence_bound(&[], &[9_u32]), 0);
    }

    #[test]
    fn high_breakscore_direct_assignment_fill_matches_legacy_rng_path() {
        fn legacy_initial_assignment(
            p_cnt: &[u32],
            n_cnt: &[u32],
            rng: &mut SmallRng,
            hp: &Hyperparameters,
        ) -> Vec<bool> {
            let nv = p_cnt.len();
            let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
            let random_threshold = hp.init_noise.unwrap_or(0.003).clamp(0.0, 0.5);
            let mut vars = vec![false; nv];
            for v in 0..nv {
                let np = p_cnt[v] as usize;
                let nn = n_cnt[v] as usize;
                if nn == 0 && np > 0 {
                    vars[v] = true;
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
                    vars[v] = rng.gen_bool(random_threshold);
                } else {
                    let prob = ((np as f64 + 0.25) / ((np + nn) as f64 + 1.2)).clamp(0.001, 0.999);
                    vars[v] = rng.gen_bool(prob);
                }
            }
            vars
        }

        let p_cnt = vec![1_u32, 0, 1, 10, 3];
        let n_cnt = vec![0_u32, 2, 2, 1, 3];
        let hp = Hyperparameters {
            init_noise: Some(0.25),
            target_nad: Some(1.0),
            ..Default::default()
        };
        let mut direct_rng = SmallRng::seed_from_u64(0x7755_3311_2468_ace0);
        let mut legacy_rng = SmallRng::seed_from_u64(0x7755_3311_2468_ace0);
        let mut direct = Vec::with_capacity(p_cnt.len());

        initial_assignment_counts_into(
            p_cnt.len(),
            &p_cnt,
            &n_cnt,
            &mut direct_rng,
            &hp,
            0,
            &mut direct,
        );
        let legacy = legacy_initial_assignment(&p_cnt, &n_cnt, &mut legacy_rng, &hp);

        assert_eq!(direct, legacy);
        assert_eq!(direct_rng.gen::<u64>(), legacy_rng.gen::<u64>());
    }

    #[test]
    fn n10000_dense_default_fuel_matches_r18_probe() {
        assert_eq!(default_max_fuel(10_000, 4.267), 125_000_000_000.0);
    }

    #[test]
    fn high_breakscore_greedy_init_is_only_enabled_by_explicit_hp() {
        let hp = Hyperparameters::default();

        assert!(!target_greedy_init_enabled(10_000, 4.267, 513, &hp));
        assert!(!target_greedy_init_enabled(7_500, 4.267, 514, &hp));

        let forced_off = Hyperparameters {
            target_greedy_init: Some(false),
            ..Default::default()
        };
        let forced_on = Hyperparameters {
            target_greedy_init: Some(true),
            ..Default::default()
        };
        assert!(!target_greedy_init_enabled(10_000, 4.267, 514, &forced_off));
        assert!(target_greedy_init_enabled(7_500, 4.267, 513, &forced_on));
    }

    #[test]
    fn greedy_shortest_init_prefers_high_count_literal() {
        let p_cnt = vec![1, 10, 20];
        let n_cnt = vec![0, 0, 0];
        let co = vec![0, 2, 4];
        let cl = vec![1, 2, -2, 3];
        let mut rng = SmallRng::seed_from_u64(7);

        let vars = initial_assignment_greedy_shortest(3, 2, &p_cnt, &n_cnt, &co, &cl, &mut rng);

        assert!(!vars[0]);
        assert!(vars[1]);
        assert!(vars[2]);
    }

    #[test]
    fn high_breakscore_greedy_shortest_all_three_fast_path_matches_generic_reference() {
        let nv = 6;
        let p_cnt = vec![0, 5, 5, 1, 3, 0];
        let n_cnt = vec![0, 5, 5, 7, 3, 0];
        let co = vec![0_u32, 3, 6, 9, 12];
        let cl = vec![1, -2, 3, -1, 2, -3, 4, -5, 6, -4, 5, -6];
        let clause_order = vec![0_u32, 1, 2, 3];

        for seed in [
            0x0123_4567_89ab_cdef,
            0x1111_2222_3333_4444,
            0xfeed_face_cafe_beef,
        ] {
            let mut generic_rng = SmallRng::seed_from_u64(seed);
            let mut fast_rng = SmallRng::seed_from_u64(seed);
            let mut generic_vars = Vec::new();
            let mut fast_vars = Vec::new();

            initial_assignment_greedy_shortest_with_order_into(
                nv,
                &p_cnt,
                &n_cnt,
                &co,
                &cl,
                &clause_order,
                false,
                &mut generic_rng,
                &mut generic_vars,
            );
            initial_assignment_greedy_shortest_with_order_into(
                nv,
                &p_cnt,
                &n_cnt,
                &co,
                &cl,
                &clause_order,
                true,
                &mut fast_rng,
                &mut fast_vars,
            );

            assert_eq!(fast_vars, generic_vars);
            assert_eq!(fast_rng.gen::<u64>(), generic_rng.gen::<u64>());
        }
    }

    #[test]
    fn high_breakscore_greedy_clause_order_keeps_u32_stable_order() {
        let co = [0_u32, 3, 3, 4, 6, 7, 7, 8];

        let order: Vec<u32> = build_greedy_clause_order(7, &co);

        assert_eq!(order, vec![2_u32, 4, 6, 3, 0]);
    }

    #[test]
    fn high_breakscore_greedy_clause_order_falls_back_for_long_clauses() {
        let co = [0_u32, 4, 5, 8, 13, 13, 15];

        let order: Vec<u32> = build_greedy_clause_order(6, &co);

        assert_eq!(order, vec![1_u32, 5, 2, 0, 3]);
    }

    #[test]
    fn high_breakscore_greedy_clause_order_all_three_literal_returns_natural_order() {
        let co = [0_u32, 3, 6, 9, 12];

        assert_eq!(build_greedy_clause_order(4, &co), vec![0_u32, 1, 2, 3]);
    }

    #[test]
    fn high_breakscore_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9, 12];
        assert!(high_breakscore_clause_offsets_are_three(4, &co, 12));
        for cid in 0..4 {
            assert_eq!(
                unsafe { high_breakscore_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn high_breakscore_clause_offsets_reject_mixed_lengths_with_average_three() {
        let co = [0_u32, 2, 5, 9];
        assert!(!high_breakscore_clause_offsets_are_three(3, &co, 9));
        assert_eq!(
            unsafe { high_breakscore_clause_bounds_unchecked(1, &co, false) },
            (2, 5)
        );
    }

    #[test]
    fn rebuild_state_overwrites_existing_counts_without_prefill() {
        let co = [0_u32, 2, 4, 5];
        let cl = [1, -2, -1, 2, 3];
        let vars = [true, true, false];
        let mut num_good = vec![9_u8; 3];
        let mut residual = vec![99_u32];
        let mut residual_pos = vec![7_u32; 3];

        rebuild_state(
            3,
            &co,
            &cl,
            &vars,
            &mut num_good,
            &mut residual,
            &mut residual_pos,
        );

        assert_eq!(num_good, vec![1, 1, 0]);
        assert_eq!(residual, vec![2]);
        assert_eq!(residual_pos, vec![u32::MAX, u32::MAX, 0]);
    }

    #[test]
    fn high_breakscore_weighted3_selector_matches_array_reference() {
        fn reference(threshold: f64, cnt: usize, weights: [f64; 3]) -> usize {
            let vars = [13usize, 21, 34];
            let mut r = threshold;
            let mut selected = vars[0];
            for i in 0..cnt {
                r -= weights[i];
                if r <= 0.0 {
                    selected = vars[i];
                    break;
                }
            }
            selected
        }

        for weights in [[1.0_f64, 0.5, 0.25], [0.125, 0.25, 0.5], [3.0, 7.0, 11.0]] {
            for cnt in 1usize..=3 {
                let total = weights[..cnt].iter().sum::<f64>();
                for threshold in [
                    0.0,
                    weights[0] * 0.5,
                    weights[0],
                    total * 0.9,
                    total,
                    total + 1.0,
                ] {
                    assert_eq!(
                        select_weighted3_f64_sub_le(
                            threshold, cnt, weights[0], 13, weights[1], 21, weights[2], 34,
                        ),
                        reference(threshold, cnt, weights)
                    );
                }
            }
        }
    }

    #[test]
    fn high_breakscore_probsat_var3_matches_generic_and_rng_consumption() {
        let cl = vec![1, -2, 3];
        let probsat_break = vec![1.0_f64, 0.5, 0.25, 0.125, 0.0625];
        let probsat_break_limit = probsat_break.len() - 1;
        let cases = [
            (vec![1_u16, 2, 3], vec![1_u16, 4, 2], vec![30_u32, 20, 10]),
            (vec![0_u16, 0, 0], vec![1_u16, 3, 2], vec![30_u32, 20, 10]),
            (vec![0_u16, 0, 1], vec![2_u16, 2, 9], vec![30_u32, 20, 10]),
            (vec![0_u16, 0, 1], vec![2_u16, 2, 9], vec![20_u32, 20, 10]),
            (vec![1_u16, 0, 0], vec![9_u16, 1, 5], vec![30_u32, 20, 10]),
        ];

        for disable_make_score in [false, true] {
            for (break_score, make_score, var_appearances) in cases.clone() {
                for seed in 0_u64..16 {
                    let mut generic_rng = SmallRng::seed_from_u64(seed);
                    let mut fast_rng = SmallRng::seed_from_u64(seed);
                    let generic = unsafe {
                        choose_high_breakscore_probsat_var_generic(
                            0,
                            3,
                            &cl,
                            &break_score,
                            &make_score,
                            &var_appearances,
                            &probsat_break,
                            probsat_break_limit,
                            disable_make_score,
                            &mut generic_rng,
                        )
                    };
                    let fast = unsafe {
                        choose_high_breakscore_probsat_var3(
                            0,
                            &cl,
                            &break_score,
                            &make_score,
                            &var_appearances,
                            &probsat_break,
                            probsat_break_limit,
                            disable_make_score,
                            &mut fast_rng,
                        )
                    };

                    assert_eq!(
                        fast, generic,
                        "seed={seed} disable_make_score={disable_make_score} break_score={break_score:?} make_score={make_score:?} appearances={var_appearances:?}"
                    );
                    assert_eq!(
                        fast_rng.gen::<u64>(),
                        generic_rng.gen::<u64>(),
                        "seed={seed} disable_make_score={disable_make_score} break_score={break_score:?} make_score={make_score:?} appearances={var_appearances:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn high_breakscore_non_probsat_var3_matches_generic_and_rng_consumption() {
        let cl = vec![1, -2, 3];
        let cases = [
            (
                vec![0_u16, 0, 0],
                vec![1_u16, 4, 2],
                vec![30_u32, 20, 10],
                vec![0_u16, 4, 8],
            ),
            (
                vec![0_u16, 0, 1],
                vec![2_u16, 2, 9],
                vec![30_u32, 20, 10],
                vec![0_u16, 0, 0],
            ),
            (
                vec![0_u16, 0, 1],
                vec![2_u16, 2, 9],
                vec![20_u32, 20, 10],
                vec![16_u16, 0, 0],
            ),
            (
                vec![1_u16, 2, 3],
                vec![1_u16, 4, 2],
                vec![30_u32, 20, 10],
                vec![0_u16, 10, 20],
            ),
            (
                vec![3_u16, 1, 2],
                vec![9_u16, 1, 5],
                vec![30_u32, 20, 10],
                vec![0_u16, 128, 32],
            ),
        ];

        for disable_make_score in [false, true] {
            for (break_score, make_score, var_appearances, var_age) in cases.clone() {
                for current_prob in [0.0_f64, 0.37, 1.0] {
                    for seed in 0_u64..24 {
                        let mut generic_rng = SmallRng::seed_from_u64(seed);
                        let mut fast_rng = SmallRng::seed_from_u64(seed);
                        let generic = unsafe {
                            choose_high_breakscore_non_probsat_var_generic(
                                0,
                                3,
                                &cl,
                                &break_score,
                                &make_score,
                                &var_appearances,
                                &var_age,
                                current_prob,
                                disable_make_score,
                                &mut generic_rng,
                            )
                        };
                        let fast = unsafe {
                            choose_high_breakscore_non_probsat_var3(
                                0,
                                &cl,
                                &break_score,
                                &make_score,
                                &var_appearances,
                                &var_age,
                                current_prob,
                                disable_make_score,
                                &mut fast_rng,
                            )
                        };

                        assert_eq!(
                            fast, generic,
                            "seed={seed} current_prob={current_prob} disable_make_score={disable_make_score} break_score={break_score:?} make_score={make_score:?} appearances={var_appearances:?} age={var_age:?}"
                        );
                        assert_eq!(
                            fast_rng.gen::<u64>(),
                            generic_rng.gen::<u64>(),
                            "seed={seed} current_prob={current_prob} disable_make_score={disable_make_score} break_score={break_score:?} make_score={make_score:?} appearances={var_appearances:?} age={var_age:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn high_breakscore_clause_has_satisfied_matches_generic_reference() {
        let cl = [1, -2, 3, -1, 2, 4, -5, 5, -4, 3, 3, -3, 3];
        let vars = [true, false, true, false, true];

        for (s, e, expected) in [
            (0, 0, false),
            (0, 1, true),
            (0, 3, true),
            (3, 5, false),
            (5, 10, true),
            (10, 13, true),
        ] {
            let generic = cl[s..e].iter().any(|&lit| lit_is_satisfied(lit, &vars));
            assert_eq!(generic, expected);
            assert_eq!(
                high_breakscore_clause_has_satisfied(s, e, &cl, &vars),
                expected
            );
        }
    }
}

fn initial_assignment_counts(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    seed_key: u64,
) -> Vec<bool> {
    let mut vars = Vec::with_capacity(nv);
    initial_assignment_counts_into(nv, p_cnt, n_cnt, rng, hp, seed_key, &mut vars);
    vars
}

fn initial_assignment_counts_into(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    _seed_key: u64,
    vars: &mut Vec<bool>,
) {
    let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
    let default_noise = if nv == 7_500 {
        0.012
    } else if nv >= 30_000 {
        0.01
    } else {
        0.003
    };
    let random_threshold = hp.init_noise.unwrap_or(default_noise).clamp(0.0, 0.5);
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
            vars.push(rng.gen_bool(random_threshold));
        } else {
            let prob = ((np as f64 + 0.25) / ((np + nn) as f64 + 1.2)).clamp(0.001, 0.999);
            vars.push(rng.gen_bool(prob));
        }
    }
    debug_assert_eq!(vars.len(), nv);
}

fn target_greedy_init_enabled(
    _nv: usize,
    _density: f64,
    _seed_key: u64,
    hp: &Hyperparameters,
) -> bool {
    hp.target_greedy_init.unwrap_or(false)
}

fn initial_assignment_greedy_shortest(
    nv: usize,
    nc: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    co: &[u32],
    cl: &[i32],
    rng: &mut SmallRng,
) -> Vec<bool> {
    let mut vars = Vec::with_capacity(nv);
    let (clause_order, all_three_clauses) =
        build_greedy_clause_order_with_all_three(nc, co, cl.len());
    initial_assignment_greedy_shortest_with_order_into(
        nv,
        p_cnt,
        n_cnt,
        co,
        cl,
        &clause_order,
        all_three_clauses,
        rng,
        &mut vars,
    );
    vars
}

fn build_greedy_clause_order(nc: usize, co: &[u32]) -> Vec<u32> {
    build_greedy_clause_order_with_all_three(nc, co, usize::MAX).0
}

fn build_greedy_clause_order_with_all_three(
    nc: usize,
    co: &[u32],
    cl_len: usize,
) -> (Vec<u32>, bool) {
    debug_assert!(nc <= u32::MAX as usize);
    let mut counts = [0usize; 4];
    let mut total = 0usize;
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len == 0 {
            continue;
        }
        if len > 3 {
            return (build_greedy_clause_order_generic(nc, co), false);
        }
        counts[len] += 1;
        total += 1;
    }
    if total == 0 {
        return (Vec::new(), false);
    }
    if counts[1] == 0 && counts[2] == 0 && total == nc {
        return (
            (0..nc as u32).collect(),
            high_breakscore_clause_offsets_are_three(nc, co, cl_len),
        );
    }

    let mut cursors = [0usize, 0, counts[1], counts[1] + counts[2]];
    let mut order = vec![0u32; total];
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len == 0 {
            continue;
        }
        let pos = cursors[len];
        order[pos] = cid as u32;
        cursors[len] = pos + 1;
    }
    (order, false)
}

fn build_greedy_clause_order_generic(nc: usize, co: &[u32]) -> Vec<u32> {
    debug_assert!(nc <= u32::MAX as usize);
    let mut max_len = 0usize;
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        max_len = max_len.max(len);
    }
    if max_len == 0 {
        return Vec::new();
    }

    let mut counts = vec![0usize; max_len + 1];
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len > 0 {
            counts[len] += 1;
        }
    }

    let mut starts = vec![0usize; max_len + 1];
    let mut total = 0usize;
    for len in 1..=max_len {
        starts[len] = total;
        total += counts[len];
    }

    let mut cursors = starts;
    let mut order = vec![0u32; total];
    for cid in 0..nc {
        let len = (co[cid + 1] - co[cid]) as usize;
        if len == 0 {
            continue;
        }
        let pos = cursors[len];
        order[pos] = cid as u32;
        cursors[len] = pos + 1;
    }
    order
}

#[inline(always)]
fn high_breakscore_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
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
unsafe fn high_breakscore_clause_bounds_unchecked(
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

fn initial_assignment_greedy_shortest_with_order_into(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    co: &[u32],
    cl: &[i32],
    clause_order: &[u32],
    all_three_clauses: bool,
    rng: &mut SmallRng,
    vars: &mut Vec<bool>,
) {
    // Experimental sentinel probe inspired by sat_imp_v4 track3; provenance is
    // recorded in the release handoff before promotion.
    vars.clear();
    vars.resize(nv, false);

    if all_three_clauses {
        debug_assert_eq!(cl.len() % 3, 0);
        debug_assert!(clause_order
            .iter()
            .enumerate()
            .all(|(idx, &cid)| cid as usize == idx));
        for cid in 0..(cl.len() / 3) {
            let s = cid * 3;
            let e = s + 3;
            if high_breakscore_clause_has_satisfied(s, e, cl, vars) {
                continue;
            }

            let mut best_score = 0u32;
            let mut best_var = 0usize;
            let mut best_value = false;
            let mut ties = 0usize;
            for &lit in &cl[s..e] {
                let v = lit_var_index(lit);
                let value = lit > 0;
                let score = if value { p_cnt[v] } else { n_cnt[v] };
                if score > best_score {
                    best_score = score;
                    best_var = v;
                    best_value = value;
                    ties = 1;
                } else if score == best_score {
                    ties += 1;
                    if rng.gen::<usize>() % ties == 0 {
                        best_var = v;
                        best_value = value;
                    }
                }
            }

            if best_score == 0 {
                let lit = cl[s + (rng.gen::<usize>() % 3)];
                best_var = lit_var_index(lit);
                best_value = lit > 0;
            }
            vars[best_var] = best_value;
        }
        return;
    }

    for &cid in clause_order {
        let cid = cid as usize;
        let (s, e) = (co[cid] as usize, co[cid + 1] as usize);
        if high_breakscore_clause_has_satisfied(s, e, cl, vars) {
            continue;
        }

        let mut best_score = 0u32;
        let mut best_var = 0usize;
        let mut best_value = false;
        let mut ties = 0usize;
        for &lit in &cl[s..e] {
            let v = lit_var_index(lit);
            let value = lit > 0;
            let score = if value { p_cnt[v] } else { n_cnt[v] };
            if score > best_score {
                best_score = score;
                best_var = v;
                best_value = value;
                ties = 1;
            } else if score == best_score {
                ties += 1;
                if rng.gen::<usize>() % ties == 0 {
                    best_var = v;
                    best_value = value;
                }
            }
        }

        if best_score == 0 {
            let lit = cl[s + (rng.gen::<usize>() % (e - s))];
            best_var = lit_var_index(lit);
            best_value = lit > 0;
        }
        vars[best_var] = best_value;
    }
}

#[inline(always)]
fn high_breakscore_clause_has_satisfied(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> bool {
    match e - s {
        1 => lit_is_satisfied(cl[s], vars),
        2 => lit_is_satisfied(cl[s], vars) || lit_is_satisfied(cl[s + 1], vars),
        3 => {
            lit_is_satisfied(cl[s], vars)
                || lit_is_satisfied(cl[s + 1], vars)
                || lit_is_satisfied(cl[s + 2], vars)
        }
        _ => cl[s..e].iter().any(|&lit| lit_is_satisfied(lit, vars)),
    }
}

#[allow(dead_code)]
fn rebuild_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    residual_pos: &mut [u32],
) {
    debug_assert!(num_good.len() >= nc);
    residual.clear();
    residual_pos.fill(u32::MAX);
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = lit_var_index(lit);
            if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                good += 1;
            }
        }
        num_good[i] = good;
        if good == 0 {
            residual_pos[i] = residual.len() as u32;
            residual.push(i as u32);
        }
    }
}
