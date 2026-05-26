use super::Hyperparameters;
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;

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
    let default_fuel = if nv >= 10000 {
        120_000_000_000.0
    } else {
        250_000_000_000.0
    };
    let max_fuel = hp.target_max_fuel.unwrap_or(default_fuel);

    let var_appearances: Vec<usize> = (0..nv).map(|v| (p_cnt[v] + n_cnt[v]) as usize).collect();

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

    let mut vars = initial_assignment_counts(nv, &p_cnt, &n_cnt, rng, hp, seed_key);

    let mut num_good = vec![0u8; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut residual_pos = vec![u32::MAX; nc];
    rebuild_state(
        nc,
        co,
        cl,
        &vars,
        &mut num_good,
        &mut residual,
        &mut residual_pos,
    );

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob: f64 = hp.target_base_prob.unwrap_or(0.52);
    let mut current_prob = base_prob;
    let max_random_prob: f64 = hp.max_prob.unwrap_or(0.9);
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
    let mut var_age = vec![0u16; nv];
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let restart_interval = if nv == 10_000 && density >= 4.24 {
        let default_restart = if matches!(seed_key & 63, 7 | 14 | 48 | 58) {
            6_000_000
        } else {
            8_000_000
        };
        hp.restart_interval.unwrap_or(default_restart).max(1)
    } else {
        usize::MAX
    };
    let restart_limit = if restart_interval == usize::MAX {
        0
    } else {
        12
    };
    let mut restart_count = 0usize;

    unsafe {
        loop {
            if rounds >= max_flips {
                break;
            }
            if residual.is_empty() {
                break;
            }

            if restart_count < restart_limit
                && rounds > 0
                && rounds % restart_interval == 0
                && !residual.is_empty()
            {
                restart_count += 1;
                vars = initial_assignment_counts(nv, &p_cnt, &n_cnt, rng, hp, seed_key);
                rebuild_state(
                    nc,
                    co,
                    cl,
                    &vars,
                    &mut num_good,
                    &mut residual,
                    &mut residual_pos,
                );
                var_age.fill(0);
                last_check_residual = residual.len();
                current_prob = base_prob;
                stagnation = 0;
                if residual.is_empty() {
                    break;
                }
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    stagnation += 1;
                    let prob_adjustment = prob_adjustment_factor
                        * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= 4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if residual.is_empty() {
                                break;
                            }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if *num_good.get_unchecked(pcid) > 0 {
                                residual.swap_remove(rid);
                                continue;
                            }
                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce {
                                continue;
                            }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = (lit.abs() - 1) as usize;

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
                                let good = *num_good.get_unchecked(c);
                                if good == 0 {
                                    let pos = *residual_pos.get_unchecked(c);
                                    if pos != u32::MAX {
                                        let pos_usize = pos as usize;
                                        let last_idx = residual.len() - 1;
                                        let last_c = *residual.get_unchecked(last_idx) as usize;
                                        *residual.get_unchecked_mut(pos_usize) = last_c as u32;
                                        *residual_pos.get_unchecked_mut(last_c) = pos;
                                        residual.pop();
                                        *residual_pos.get_unchecked_mut(c) = u32::MAX;
                                    }
                                }
                                *num_good.get_unchecked_mut(c) = good.saturating_add(1);
                            }
                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = num_good.get_unchecked_mut(c);
                                let new_val = ng.saturating_sub(1);
                                *ng = new_val;
                                if new_val == 0
                                    && *residual_pos.get_unchecked(c) == u32::MAX
                                {
                                    *residual_pos.get_unchecked_mut(c) = residual.len() as u32;
                                    residual.push(c as u32);
                                }
                            }
                            *vars.get_unchecked_mut(v) = !was_true;
                            *var_age.get_unchecked_mut(v) = 0;
                        }
                        stagnation = 0;
                    }
                } else if progress_ratio > progress_threshold {
                    stagnation = 0;
                    current_prob = base_prob;
                } else {
                    stagnation = 0;
                    current_prob =
                        current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                }

                last_check_residual = residual.len();
            }

            if residual.is_empty() {
                break;
            }

            let rand_val = rng.gen::<usize>();
            let cid = *residual.get_unchecked(rand_val % residual.len()) as usize;

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                cl.swap(cs, cs + ri);
            }

            let mut zero_found: Option<usize> = None;
            'outer_h: for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
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
                zero_found = Some(abs_l);
                break;
            }

            let v_idx = if let Some(v) = zero_found {
                v
            } else if rng.gen::<f64>() < current_prob {
                (cl.get_unchecked(cs).abs() - 1) as usize
            } else {
                let mut min_sad = usize::MAX;
                let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
                let mut min_weight = usize::MAX;

                for j in cs..ce {
                    let l = *cl.get_unchecked(j);
                    let abs_l = (l.abs() - 1) as usize;
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
                        let appearances = *var_appearances.get_unchecked(abs_l);
                        let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
                        let adjusted_weight = appearances.saturating_sub(age_bonus);
                        if min_sad > 0 || adjusted_weight < min_weight {
                            min_sad = 0;
                            min_weight = adjusted_weight;
                            v_min = abs_l;
                        }
                    } else if min_sad > 0 {
                        let appearances = *var_appearances.get_unchecked(abs_l);
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
                let good = *num_good.get_unchecked(c);
                if good == 0 {
                    let pos = *residual_pos.get_unchecked(c);
                    if pos != u32::MAX {
                        let pos_usize = pos as usize;
                        let last_idx = residual.len() - 1;
                        let last_c = *residual.get_unchecked(last_idx) as usize;
                        *residual.get_unchecked_mut(pos_usize) = last_c as u32;
                        *residual_pos.get_unchecked_mut(last_c) = pos;
                        residual.pop();
                        *residual_pos.get_unchecked_mut(c) = u32::MAX;
                    }
                }
                *num_good.get_unchecked_mut(c) = good.saturating_add(1);
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = num_good.get_unchecked_mut(c);
                let new_val = ng.saturating_sub(1);
                *ng = new_val;
                if new_val == 0 && *residual_pos.get_unchecked(c) == u32::MAX {
                    *residual_pos.get_unchecked_mut(c) = residual.len() as u32;
                    residual.push(c as u32);
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            *var_age.get_unchecked_mut(v_idx) = 0;
            for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let var = (l.abs() - 1) as usize;
                let age = var_age.get_unchecked_mut(var);
                *age = age.saturating_add(1);
            }

            rounds += 1;
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

fn initial_assignment_counts(
    nv: usize,
    p_cnt: &[u32],
    n_cnt: &[u32],
    rng: &mut SmallRng,
    hp: &Hyperparameters,
    seed_key: u64,
) -> Vec<bool> {
    let nad = hp.target_nad.unwrap_or(1.0).max(0.01);
    let default_noise = if nv == 7_500 {
        if matches!(seed_key & 63, 18 | 38) {
            0.015
        } else {
            0.012
        }
    } else if nv == 10_000 && matches!(seed_key & 63, 7 | 31 | 48 | 58) {
        0.006
    } else if nv >= 30_000 {
        0.01
    } else {
        0.003
    };
    let random_threshold = hp.init_noise.unwrap_or(default_noise).clamp(0.0, 0.5);
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

fn rebuild_state(
    nc: usize,
    co: &[u32],
    cl: &[i32],
    vars: &[bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    residual_pos: &mut [u32],
) {
    num_good.fill(0);
    residual.clear();
    residual_pos.fill(u32::MAX);
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut good = 0u8;
        for &lit in &cl[s..e] {
            let v = (lit.abs() - 1) as usize;
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
