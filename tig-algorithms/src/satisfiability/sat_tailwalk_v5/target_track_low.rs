use super::Hyperparameters;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

#[inline(always)]
fn initial_residual_capacity(nc: usize) -> usize {
    (nc / 4).saturating_add(16)
}

#[inline(always)]
fn prob_cutoff_u64(prob: f64) -> u64 {
    (prob.max(0.0).min(1.0) * (u64::MAX as f64)) as u64
}

pub fn solve(
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
    let nvf = nv as f64;
    let max_fuel = hp.target_max_fuel.unwrap_or(150_000_000_000.0);
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

    let (mut vars, appearances) = initialize_vars_and_appearances(nv, density, &p_cnt, &n_cnt, rng);
    drop(p_cnt);
    drop(n_cnt);

    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];
    let mut residual: Vec<u32> = Vec::with_capacity(initial_residual_capacity(nc));
    debug_assert_eq!(
        all_three_clauses,
        all_clause_offsets_are_three(nc, co, cl.len())
    );
    build_initial_packed_state(
        nc,
        co,
        cl,
        &vars,
        &mut num_good,
        &mut residual,
        all_three_clauses,
    );

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob = hp
        .target_base_prob
        .unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
    let mut current_prob = base_prob;
    let mut current_prob_cutoff = prob_cutoff_u64(current_prob);

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.check_interval.unwrap_or(
        (base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval)
            as usize,
    );
    let max_random_prob = hp.max_prob.unwrap_or(0.9);
    let prob_adjustment_factor = 0.03;
    let smoothing_factor = 0.8;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp
        .perturbation_flips
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let default_stagnation_limit = if nv >= 50_000 {
        match hp.hw_profile.as_deref() {
            Some("zen4") => 3,
            Some("zen5") => 3,
            Some("zen5c") => 5,
            _ if nv == 100_000 && (4.14..4.16).contains(&density) => 5,
            _ => 2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize,
        }
    } else {
        2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize
    };
    let stagnation_limit = hp.stagnation_limit.unwrap_or(default_stagnation_limit);

    let mut last_check_residual = residual.len();
    let mut stagnation = 0usize;
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;
    let mut rounds = 0usize;

    let _probs_break: [u32; 16] = [
        2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7,
    ];

    unsafe {
        loop {
            if residual.is_empty() || rounds >= max_flips {
                break;
            }

            countdown -= 1;
            if countdown == 0 {
                countdown = check_interval;
                let progress = last_check_residual as i64 - residual.len() as i64;

                if progress <= 0 {
                    stagnation += 1;
                    let prob_adjustment = prob_adjustment_factor
                        * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= stagnation_limit {
                        let kicks = if stagnation >= 5 {
                            (perturbation_flips * 12).min(100)
                        } else if stagnation >= 4 {
                            (perturbation_flips * 6).min(50)
                        } else if stagnation >= 3 {
                            (perturbation_flips * 3).min(20)
                        } else {
                            (perturbation_flips + 2).min(10)
                        };

                        for _ in 0..kicks {
                            if residual.is_empty() {
                                break;
                            }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            let ng_val =
                                (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3;
                            if ng_val > 0 {
                                remove_residual_unordered(&mut residual, rid);
                                continue;
                            }
                            let (pcs, pce) =
                                low_clause_bounds_unchecked(pcid, co, all_three_clauses);
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
                                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
                            }
                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let shift = (c & 3) << 1;
                                let byte_idx = c >> 2;
                                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                                if ng_before == 1 {
                                    residual.push(c as u32);
                                }
                            }
                            *vars.get_unchecked_mut(v) = !was_true;
                            *var_age.get_unchecked_mut(v) = 0;
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                    let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                    if progress_ratio > progress_threshold {
                        current_prob = base_prob;
                    } else {
                        current_prob =
                            current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                    }
                }

                current_prob_cutoff = prob_cutoff_u64(current_prob);
                last_check_residual = residual.len();
            }

            let rand_val = rng.gen::<usize>();
            let mut cid = 0usize;
            let mut found = false;
            while !residual.is_empty() {
                let id = rand_val % residual.len();
                let candidate = *residual.get_unchecked(id) as usize;
                let ng_val =
                    (*num_good.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3;
                if ng_val > 0 {
                    remove_residual_unordered(&mut residual, id);
                } else {
                    cid = candidate;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }

            let (cs, ce) = low_clause_bounds_unchecked(cid, co, all_three_clauses);
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                if ri != 0 {
                    cl.swap(cs, cs + ri);
                }
            }

            let v_idx = if all_three_clauses {
                debug_assert_eq!(clen, 3);
                choose_low_var3(
                    rng,
                    rand_val,
                    current_prob_cutoff,
                    cs,
                    cl,
                    &num_good,
                    &var_age,
                    &appearances,
                    all_off,
                    p_bound,
                    all_data,
                )
            } else {
                choose_low_var(
                    rng,
                    rand_val,
                    current_prob_cutoff,
                    cs,
                    ce,
                    cl,
                    &num_good,
                    &var_age,
                    &appearances,
                    all_off,
                    p_bound,
                    all_data,
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
                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                if ng_before == 1 {
                    residual.push(c as u32);
                }
            }
            *vars.get_unchecked_mut(v_idx) = !was_true;
            *var_age.get_unchecked_mut(v_idx) = 0;
            if all_three_clauses {
                bump_low_clause_var_ages3(cs, cl, &mut var_age);
            } else {
                bump_low_clause_var_ages(cs, ce, cl, &mut var_age);
            }
            rounds += 1;
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

fn initialize_vars_and_appearances(
    nv: usize,
    density: f64,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
) -> (Vec<bool>, Vec<u8>) {
    let nvf = nv as f64;
    let nad = 1.0;
    let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    let mut vars = Vec::with_capacity(nv);
    let mut appearances = Vec::with_capacity(nv);

    for v in 0..nv {
        let pc = p_cnt[v];
        let nc = n_cnt[v];
        appearances.push(((pc + nc) as usize).min(255) as u8);

        let np = pc as f64;
        let nn = nc as f64;
        if nn == 0.0 && np > 0.0 {
            vars.push(true);
            continue;
        }
        if np == 0.0 {
            vars.push(false);
            continue;
        }
        let vad = np / nn;
        let bias_prob = (np + 0.25) / (np + nn + 1.2);
        let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
        let prob = (random_threshold * (1.0 - s) + bias_prob * s)
            .max(0.0)
            .min(1.0);
        vars.push(rng.gen_bool(prob));
    }

    debug_assert_eq!(vars.len(), nv);
    debug_assert_eq!(appearances.len(), nv);
    (vars, appearances)
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
    let var = lit_var_index(lit);
    (lit > 0) == vars[var]
}

#[inline(always)]
unsafe fn bump_low_var_age(var: usize, var_age: &mut [u8]) {
    let age = var_age.get_unchecked_mut(var);
    *age = age.saturating_add(1);
}

#[inline(always)]
unsafe fn bump_low_clause_var_ages(cs: usize, ce: usize, cl: &[i32], var_age: &mut [u8]) {
    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        bump_low_var_age(lit_var_index(lit), var_age);
    }
}

#[inline(always)]
unsafe fn bump_low_clause_var_ages3(cs: usize, cl: &[i32], var_age: &mut [u8]) {
    debug_assert!(cs + 2 < cl.len());
    bump_low_var_age(lit_var_index(*cl.get_unchecked(cs)), var_age);
    bump_low_var_age(lit_var_index(*cl.get_unchecked(cs + 1)), var_age);
    bump_low_var_age(lit_var_index(*cl.get_unchecked(cs + 2)), var_age);
}

#[inline(always)]
fn initial_clause_good_count(s: usize, e: usize, cl: &[i32], vars: &[bool]) -> u8 {
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
unsafe fn choose_low_var(
    rng: &mut SmallRng,
    rand_val: usize,
    current_prob_cutoff: u64,
    cs: usize,
    ce: usize,
    cl: &[i32],
    num_good: &[u8],
    var_age: &[u8],
    appearances: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    let mut zero0 = 0usize;
    let mut zero1 = 0usize;
    let mut zero2 = 0usize;
    let mut zero_cnt = 0usize;
    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        let v = lit_var_index(lit);
        if low_sad_limited(lit, num_good, all_off, p_bound, all_data, 1) == 0 {
            match zero_cnt {
                0 => zero0 = v,
                1 => zero1 = v,
                _ => zero2 = v,
            }
            zero_cnt += 1;
        }
    }

    if zero_cnt > 0 {
        return match rand_val % zero_cnt {
            0 => zero0,
            1 => zero1,
            _ => zero2,
        };
    }
    if rng.gen::<u64>() <= current_prob_cutoff {
        return lit_var_index(*cl.get_unchecked(cs));
    }

    let mut min_sad = usize::MAX;
    let mut min_weight = usize::MAX;
    let mut v_min = lit_var_index(*cl.get_unchecked(cs));

    for j in cs..ce {
        let lit = *cl.get_unchecked(j);
        let sad = low_consider_weighted_var(
            lit,
            num_good,
            var_age,
            appearances,
            all_off,
            p_bound,
            all_data,
            &mut min_sad,
            &mut min_weight,
            &mut v_min,
        );
        if sad > 0 && min_sad <= 1 {
            break;
        }
    }

    v_min
}

#[inline(always)]
unsafe fn choose_low_var3(
    rng: &mut SmallRng,
    rand_val: usize,
    current_prob_cutoff: u64,
    cs: usize,
    cl: &[i32],
    num_good: &[u8],
    var_age: &[u8],
    appearances: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
) -> usize {
    debug_assert!(cs + 2 < cl.len());
    let lit0 = *cl.get_unchecked(cs);
    let lit1 = *cl.get_unchecked(cs + 1);
    let lit2 = *cl.get_unchecked(cs + 2);
    let v0 = lit_var_index(lit0);
    let v1 = lit_var_index(lit1);
    let v2 = lit_var_index(lit2);

    let z0 = low_sad_limited(lit0, num_good, all_off, p_bound, all_data, 1) == 0;
    let z1 = low_sad_limited(lit1, num_good, all_off, p_bound, all_data, 1) == 0;
    let z2 = low_sad_limited(lit2, num_good, all_off, p_bound, all_data, 1) == 0;
    let zero_cnt = z0 as usize + z1 as usize + z2 as usize;
    if zero_cnt > 0 {
        return match rand_val % zero_cnt {
            0 if z0 => v0,
            0 if z1 => v1,
            0 => v2,
            1 if z0 && z1 => v1,
            1 if z0 || z1 => v2,
            1 => v2,
            _ => v2,
        };
    }

    if rng.gen::<u64>() <= current_prob_cutoff {
        return v0;
    }

    let mut min_sad = usize::MAX;
    let mut min_weight = usize::MAX;
    let mut v_min = v0;
    let sad0 = low_consider_weighted_var(
        lit0,
        num_good,
        var_age,
        appearances,
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
    let sad1 = low_consider_weighted_var(
        lit1,
        num_good,
        var_age,
        appearances,
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
    low_consider_weighted_var(
        lit2,
        num_good,
        var_age,
        appearances,
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
unsafe fn low_consider_weighted_var(
    lit: i32,
    num_good: &[u8],
    var_age: &[u8],
    appearances: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    min_sad: &mut usize,
    min_weight: &mut usize,
    v_min: &mut usize,
) -> usize {
    let v = lit_var_index(lit);
    let sad = low_sad_limited(lit, num_good, all_off, p_bound, all_data, *min_sad);
    if sad == 0 {
        let app = *appearances.get_unchecked(v) as usize;
        let age_bonus = (*var_age.get_unchecked(v) as usize) / 4;
        let adjusted_weight = app.saturating_sub(age_bonus);
        if *min_sad > 0 || adjusted_weight < *min_weight {
            *min_sad = 0;
            *min_weight = adjusted_weight;
            *v_min = v;
        }
    } else if *min_sad > 0 {
        let app = *appearances.get_unchecked(v) as usize;
        let age_bonus = (*var_age.get_unchecked(v) as usize) / 2;
        let combined_weight = sad * sad * 1024 + app - age_bonus.min(50);
        if combined_weight < *min_weight {
            *min_sad = sad;
            *min_weight = combined_weight;
            *v_min = v;
        }
    }
    sad
}

#[inline(always)]
unsafe fn low_sad_limited(
    lit: i32,
    num_good: &[u8],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    limit: usize,
) -> usize {
    let v = lit_var_index(lit);
    let (os, oe) = if lit > 0 {
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
    let mut sad = 0usize;
    for k in os..oe {
        let c = *all_data.get_unchecked(k) as usize;
        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
            sad += 1;
            if sad >= limit {
                break;
            }
        }
    }
    sad
}

#[inline(always)]
fn remove_residual_unordered(residual: &mut Vec<u32>, rid: usize) {
    if rid + 1 == residual.len() {
        residual.pop();
    } else {
        residual.swap_remove(rid);
    }
}

fn build_initial_packed_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    all_three_clauses: bool,
) {
    debug_assert!(num_good.len() >= (nc + 3) >> 2);
    debug_assert!(residual.is_empty());

    if all_three_clauses {
        build_initial_packed_state_all_three(nc, cl, vars, num_good, residual);
        return;
    }

    let full_bytes = nc >> 2;
    for byte_idx in 0..full_bytes {
        let base = byte_idx << 2;
        let g0 = initial_clause_good_count(co[base] as usize, co[base + 1] as usize, cl, vars);
        let g1 = initial_clause_good_count(co[base + 1] as usize, co[base + 2] as usize, cl, vars);
        let g2 = initial_clause_good_count(co[base + 2] as usize, co[base + 3] as usize, cl, vars);
        let g3 = initial_clause_good_count(co[base + 3] as usize, co[base + 4] as usize, cl, vars);
        num_good[byte_idx] = g0 | (g1 << 2) | (g2 << 4) | (g3 << 6);

        if g0 == 0 {
            residual.push(base as u32);
        }
        if g1 == 0 {
            residual.push((base + 1) as u32);
        }
        if g2 == 0 {
            residual.push((base + 2) as u32);
        }
        if g3 == 0 {
            residual.push((base + 3) as u32);
        }
    }

    let mut i = full_bytes << 2;
    if i < nc {
        let byte_idx = i >> 2;
        let mut packed = 0u8;
        while i < nc {
            let good = initial_clause_good_count(co[i] as usize, co[i + 1] as usize, cl, vars);
            packed |= good << ((i & 3) << 1);
            if good == 0 {
                residual.push(i as u32);
            }
            i += 1;
        }
        num_good[byte_idx] = packed;
    }
}

#[inline(always)]
fn all_clause_offsets_are_three(nc: usize, co: &[u32], cl_len: usize) -> bool {
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
unsafe fn low_clause_bounds_unchecked(
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

fn build_initial_packed_state_all_three(
    nc: usize,
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
) {
    let full_bytes = nc >> 2;
    for byte_idx in 0..full_bytes {
        let base = byte_idx << 2;
        let s = base * 3;
        let g0 = initial_clause3_good_count(s, cl, vars);
        let g1 = initial_clause3_good_count(s + 3, cl, vars);
        let g2 = initial_clause3_good_count(s + 6, cl, vars);
        let g3 = initial_clause3_good_count(s + 9, cl, vars);
        num_good[byte_idx] = g0 | (g1 << 2) | (g2 << 4) | (g3 << 6);

        if g0 == 0 {
            residual.push(base as u32);
        }
        if g1 == 0 {
            residual.push((base + 1) as u32);
        }
        if g2 == 0 {
            residual.push((base + 2) as u32);
        }
        if g3 == 0 {
            residual.push((base + 3) as u32);
        }
    }

    let mut i = full_bytes << 2;
    if i < nc {
        let byte_idx = i >> 2;
        let mut packed = 0u8;
        while i < nc {
            let good = initial_clause3_good_count(i * 3, cl, vars);
            packed |= good << ((i & 3) << 1);
            if good == 0 {
                residual.push(i as u32);
            }
            i += 1;
        }
        num_good[byte_idx] = packed;
    }
}

#[inline(always)]
fn initial_clause3_good_count(s: usize, cl: &[i32], vars: &[bool]) -> u8 {
    lit_is_satisfied(cl[s], vars) as u8
        + lit_is_satisfied(cl[s + 1], vars) as u8
        + lit_is_satisfied(cl[s + 2], vars) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn lit_var_index_matches_abs_index_for_valid_literals() {
        for lit in [-128_i32, -7, -1, 1, 7, 128] {
            assert_eq!(lit_var_index(lit), (lit.abs() - 1) as usize);
        }
    }

    #[test]
    fn initial_residual_capacity_is_conservative_quarter_clause_estimate() {
        assert_eq!(initial_residual_capacity(0), 16);
        assert_eq!(initial_residual_capacity(3), 16);
        assert_eq!(initial_residual_capacity(100), 41);
        assert_eq!(initial_residual_capacity(400_000), 100_016);
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
    fn fused_initial_assignment_matches_legacy_rng_path() {
        let p_cnt = vec![10, 0, 4, 1, 260];
        let n_cnt = vec![0, 3, 2, 1, 10];
        let density = 4.15;
        let mut fused_rng = SmallRng::seed_from_u64(11);
        let mut legacy_rng = SmallRng::seed_from_u64(11);

        let (vars, appearances) =
            initialize_vars_and_appearances(5, density, &p_cnt, &n_cnt, &mut fused_rng);

        let mut legacy_vars = vec![false; 5];
        let nad = 1.0;
        let nvf = 5.0_f64;
        let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
        let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
        for v in 0..5 {
            let np = p_cnt[v] as f64;
            let nn = n_cnt[v] as f64;
            if nn == 0.0 && np > 0.0 {
                legacy_vars[v] = true;
                continue;
            }
            if np == 0.0 {
                continue;
            }
            let vad = np / nn;
            let bias_prob = (np + 0.25) / (np + nn + 1.2);
            let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
            let prob = (random_threshold * (1.0 - s) + bias_prob * s)
                .max(0.0)
                .min(1.0);
            legacy_vars[v] = legacy_rng.gen_bool(prob);
        }
        let legacy_appearances = (0..5)
            .map(|v| ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8)
            .collect::<Vec<_>>();

        assert_eq!(vars, legacy_vars);
        assert_eq!(appearances, legacy_appearances);
        assert_eq!(fused_rng.gen::<u64>(), legacy_rng.gen::<u64>());
    }

    #[test]
    fn initial_packed_state_collects_residual_in_clause_order() {
        let co = [0_u32, 2, 4, 5, 7];
        let cl = [1, -2, -1, 2, 3, -1, -3];
        let vars = [true, true, false];
        let mut num_good = vec![0_u8; (4 + 3) >> 2];
        let mut residual = Vec::new();

        build_initial_packed_state(4, &co, &cl, &vars, &mut num_good, &mut residual, false);

        assert_eq!(num_good, vec![1 | (1 << 2) | (0 << 4) | (1 << 6)]);
        assert_eq!(residual, vec![2]);
    }

    #[test]
    fn initial_packed_state_three_literal_fast_path_counts_duplicates() {
        let co = [0_u32, 3, 6, 9];
        let cl = [1, -2, -2, -1, 2, 3, -1, 2, -3];
        let vars = [true, false, true];
        let mut num_good = vec![0_u8; (3 + 3) >> 2];
        let mut residual = Vec::new();

        build_initial_packed_state(3, &co, &cl, &vars, &mut num_good, &mut residual, true);

        assert_eq!(num_good, vec![3 | (1 << 2) | (0 << 4)]);
        assert_eq!(residual, vec![2]);
    }

    #[test]
    fn initial_packed_state_all_three_fast_path_matches_generic_reference() {
        let co = [0_u32, 3, 6, 9, 12, 15];
        let cl = [1, -2, 3, -1, 2, -3, 1, 2, -3, -1, -2, -3, 2, 3, -1];
        let vars = [true, true, false];
        let mut expected_good = vec![0_u8; (5 + 3) >> 2];
        let mut expected_residual = Vec::new();
        for i in 0..5 {
            let good = initial_clause_good_count(co[i] as usize, co[i + 1] as usize, &cl, &vars);
            expected_good[i >> 2] |= good << ((i & 3) << 1);
            if good == 0 {
                expected_residual.push(i as u32);
            }
        }

        let mut actual_good = vec![0_u8; (5 + 3) >> 2];
        let mut actual_residual = Vec::new();

        assert!(all_clause_offsets_are_three(5, &co, cl.len()));
        build_initial_packed_state(
            5,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_residual,
            true,
        );

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_residual, expected_residual);
    }

    #[test]
    fn all_three_offset_guard_rejects_mixed_lengths_with_average_three() {
        let co = [0_u32, 2, 6, 9];

        assert!(!all_clause_offsets_are_three(3, &co, 9));
    }

    #[test]
    fn low_clause_bounds_all_three_fast_path_matches_offsets() {
        let co = [0_u32, 3, 6, 9];

        for cid in 0..3 {
            assert_eq!(
                unsafe { low_clause_bounds_unchecked(cid, &co, true) },
                (co[cid] as usize, co[cid + 1] as usize)
            );
        }
    }

    #[test]
    fn low_clause_bounds_mixed_lengths_use_offsets() {
        let co = [0_u32, 2, 6, 9];

        assert_eq!(
            unsafe { low_clause_bounds_unchecked(0, &co, false) },
            (0, 2)
        );
        assert_eq!(
            unsafe { low_clause_bounds_unchecked(1, &co, false) },
            (2, 6)
        );
        assert_eq!(
            unsafe { low_clause_bounds_unchecked(2, &co, false) },
            (6, 9)
        );
    }

    #[test]
    fn bump_low_clause_var_ages3_matches_generic_loop() {
        let cl = [99, 1, -2, 2, 100];
        let mut generic_age = [5_u8, u8::MAX - 1, 17];
        let mut fast_age = generic_age;

        unsafe {
            bump_low_clause_var_ages(1, 4, &cl, &mut generic_age);
            bump_low_clause_var_ages3(1, &cl, &mut fast_age);
        }

        assert_eq!(fast_age, generic_age);
        assert_eq!(fast_age, [6, u8::MAX, 17]);
    }

    #[test]
    fn choose_low_var3_zero_candidates_match_generic_without_rng() {
        let cl = [99, 1, 2, 3, 100];
        let num_good = [];
        let var_age = [0u8, 0, 0];
        let appearances = [10u8, 20, 30];
        let all_off = [0u32, 0, 0, 0];
        let p_bound = [0u32, 0, 0];
        let all_data = [];
        let rand_val = 5usize;
        let mut generic_rng = SmallRng::seed_from_u64(0x4150_0001);
        let mut fixed_rng = SmallRng::seed_from_u64(0x4150_0001);

        let generic = unsafe {
            choose_low_var(
                &mut generic_rng,
                rand_val,
                0,
                1,
                4,
                &cl,
                &num_good,
                &var_age,
                &appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        let fixed = unsafe {
            choose_low_var3(
                &mut fixed_rng,
                rand_val,
                0,
                1,
                &cl,
                &num_good,
                &var_age,
                &appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };

        assert_eq!(fixed, generic);
        assert_eq!(fixed, 2);
        assert_eq!(fixed_rng.gen::<u64>(), generic_rng.gen::<u64>());
    }

    #[test]
    fn choose_low_var3_weighted_path_matches_generic_and_rng() {
        let cl = [99, 1, 2, 3, 100];
        let num_good = [1 | (1 << 2) | (1 << 4)];
        let var_age = [0u8, 0, 0];
        let appearances = [20u8, 5, 10];
        let all_off = [0u32, 1, 2, 3];
        let p_bound = [0u32, 1, 2];
        let all_data = [0u32, 1, 2];
        let rand_val = 7usize;
        let mut generic_rng = SmallRng::seed_from_u64(0x4150_0002);
        let mut fixed_rng = SmallRng::seed_from_u64(0x4150_0002);

        let generic = unsafe {
            choose_low_var(
                &mut generic_rng,
                rand_val,
                0,
                1,
                4,
                &cl,
                &num_good,
                &var_age,
                &appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };
        let fixed = unsafe {
            choose_low_var3(
                &mut fixed_rng,
                rand_val,
                0,
                1,
                &cl,
                &num_good,
                &var_age,
                &appearances,
                &all_off,
                &p_bound,
                &all_data,
            )
        };

        assert_eq!(fixed, generic);
        assert_eq!(fixed, 0);
        assert_eq!(fixed_rng.gen::<u64>(), generic_rng.gen::<u64>());
    }

    #[test]
    fn initial_clause_good_count_fast_paths_match_generic_reference() {
        let cl = [1, -2, 3, -1, 2, -3, 1, -2, 3, -4];
        let vars = [true, true, false, true];

        for (s, e) in [(0, 0), (0, 1), (0, 2), (0, 3), (3, 6), (6, 10)] {
            let mut expected = 0u8;
            for &lit in &cl[s..e] {
                if lit_is_satisfied(lit, &vars) {
                    expected += 1;
                }
            }
            assert_eq!(initial_clause_good_count(s, e, &cl, &vars), expected);
        }
    }

    #[test]
    fn initial_packed_state_four_clause_packing_matches_legacy_order() {
        fn legacy_build_initial_packed_state(
            nc: usize,
            co: &[u32],
            cl: &[i32],
            vars: &[bool],
        ) -> (Vec<u8>, Vec<u32>) {
            let mut num_good = vec![0_u8; (nc + 3) >> 2];
            let mut residual = Vec::new();
            for i in 0..nc {
                let good = initial_clause_good_count(co[i] as usize, co[i + 1] as usize, cl, vars);
                num_good[i >> 2] += good << ((i & 3) << 1);
                if good == 0 {
                    residual.push(i as u32);
                }
            }
            (num_good, residual)
        }

        let co = [0_u32, 3, 6, 9, 12, 14, 17, 20];
        let cl = [
            1, -2, 3, -1, 2, -3, 1, 2, -3, -1, -2, -3, 1, -2, -1, 2, 3, 1, -2, -3,
        ];
        let vars = [true, true, false];
        let (expected_good, expected_residual) =
            legacy_build_initial_packed_state(7, &co, &cl, &vars);
        let mut actual_good = vec![0_u8; (7 + 3) >> 2];
        let mut actual_residual = Vec::new();

        build_initial_packed_state(
            7,
            &co,
            &cl,
            &vars,
            &mut actual_good,
            &mut actual_residual,
            false,
        );

        assert_eq!(actual_good, expected_good);
        assert_eq!(actual_residual, expected_residual);
    }
}
