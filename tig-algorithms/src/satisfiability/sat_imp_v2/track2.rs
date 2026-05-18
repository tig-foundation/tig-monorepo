use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;

pub fn solve(
    challenge: &Challenge,
    hp: &Option<Hyperparameters>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;

    let mut p_off = vec![0u32; nv + 1];
    let mut n_off = vec![0u32; nv + 1];
    for v in 0..nv {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    let mut p_data = vec![0u32; p_off[nv] as usize];
    let mut n_data = vec![0u32; n_off[nv] as usize];

    {
        let mut p_pos = p_off[..nv].to_vec();
        let mut n_pos = n_off[..nv].to_vec();
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { p_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { n_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { p_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { n_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { p_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { n_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            ci += 1;
        }
    }

    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);
    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        cl.push(a);
        if b != a { cl.push(b); }
        if c != a && c != b { cl.push(c); }
        co.push(cl.len() as u32);
    }

    let density = nc as f64 / nv as f64;
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(250_000_000_000.0);

    let var_appearances: Vec<usize> = (0..nv)
        .map(|v| (p_cnt[v] + n_cnt[v]) as usize)
        .collect();

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

    let nad = 1.0;
    let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };

    let base_prob: f64 = 0.52;
    let max_random_prob: f64 = 0.9;
    let prob_adjustment_factor: f64 = 0.025;
    let smoothing_factor: f64 = 0.8;

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let raw_restarts = if nv <= 12000 { 4usize } else if nv <= 40000 { 3usize } else { 2usize };
    let restart_attempts = raw_restarts
        .saturating_sub(if density > 5.0 { 1 } else { 0 })
        .max(1)
        .min(max_flips.max(1));

    let mut best_unsat = nc + 1;
    let mut best_vars = vec![false; nv];
    let mut vars = vec![false; nv];
    let mut num_good = vec![0u8; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut var_age = vec![0u16; nv];

    for attempt in 0..restart_attempts {
        let attempt_budget = if restart_attempts == 1 {
            max_flips
        } else {
            max_flips / restart_attempts
                + if attempt < (max_flips % restart_attempts) { 1 } else { 0 }
        };
        let attempt_random_threshold = if attempt == 0 {
            random_threshold
        } else {
            (random_threshold * (1.0 + 0.45 * attempt as f64)).min(0.08)
        };
        let attempt_relax = if attempt == 0 {
            0.0
        } else {
            (0.06 * attempt as f64).min(0.18)
        };

        vars.fill(false);
        for v in 0..nv {
            let np = p_cnt[v] as usize;
            let nn = n_cnt[v] as usize;
            if nn == 0 && np > 0 { vars[v] = true; continue; }
            if np == 0 && nn > 0 { continue; }
            let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
            if vad <= nad {
                vars[v] = rng.gen_bool(attempt_random_threshold);
            } else {
                let bias_prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                let prob = (bias_prob * (1.0 - attempt_relax) + 0.5 * attempt_relax).clamp(0.0, 1.0);
                vars[v] = rng.gen_bool(prob);
            }
        }

        if attempt > 0 && best_unsat < nc {
            let keep_mod = if nv <= 20000 { 7usize } else { 11usize };
            for v in 0..nv {
                if p_cnt[v] > 0 && n_cnt[v] > 0 && rng.gen::<usize>() % keep_mod == 0 {
                    vars[v] = best_vars[v];
                }
            }
        }

        num_good.fill(0);
        for i in 0..nc {
            let s = co[i] as usize;
            let e = co[i + 1] as usize;
            for j in s..e {
                let l = cl[j];
                let v = (l.abs() - 1) as usize;
                if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                    num_good[i] += 1;
                }
            }
        }

        residual.clear();
        for i in 0..nc {
            if num_good[i] == 0 {
                residual.push(i as u32);
            }
        }
        let mut unsat_count = residual.len();

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            best_vars.clone_from(&vars);
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }

        let mut current_prob = base_prob;
        let mut last_check_unsat = unsat_count;
        var_age.fill(0);
        let mut rounds = 0usize;
        let mut stagnation = 0usize;

        unsafe {
            loop {
                if rounds >= attempt_budget || unsat_count == 0 { break; }

                if rounds % check_interval == 0 && rounds > 0 {
                    if unsat_count < best_unsat {
                        best_unsat = unsat_count;
                        best_vars.clone_from(&vars);
                    }

                    let progress = last_check_unsat as i64 - unsat_count as i64;
                    let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;

                    if progress <= 0 {
                        stagnation += 1;
                        let prob_adjustment = prob_adjustment_factor
                            * (-progress as f64 / last_check_unsat.max(1) as f64).min(1.0);
                        current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                        if stagnation >= 4 {
                            let kicks = if stagnation >= 8 { 6 } else { 3 };
                            for _ in 0..kicks {
                                if residual.is_empty() || unsat_count == 0 { break; }
                                let rid = rng.gen::<usize>() % residual.len();
                                let pcid = *residual.get_unchecked(rid) as usize;
                                if *num_good.get_unchecked(pcid) > 0 {
                                    residual.swap_remove(rid);
                                    continue;
                                }
                                let pcs = *co.get_unchecked(pcid) as usize;
                                let pce = *co.get_unchecked(pcid + 1) as usize;
                                if pcs == pce { continue; }
                                let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                                let v = (lit.abs() - 1) as usize;

                                let was_true = *vars.get_unchecked(v);
                                let (is, ie, ia) = if was_true {
                                    (*n_off.get_unchecked(v), *n_off.get_unchecked(v + 1), &n_data)
                                } else {
                                    (*p_off.get_unchecked(v), *p_off.get_unchecked(v + 1), &p_data)
                                };
                                let (ds, de, da) = if was_true {
                                    (*p_off.get_unchecked(v), *p_off.get_unchecked(v + 1), &p_data)
                                } else {
                                    (*n_off.get_unchecked(v), *n_off.get_unchecked(v + 1), &n_data)
                                };

                                for k in is..ie {
                                    let c = *ia.get_unchecked(k as usize) as usize;
                                    let ng = *num_good.get_unchecked(c);
                                    if ng == 0 { unsat_count -= 1; }
                                    *num_good.get_unchecked_mut(c) = ng.saturating_add(1);
                                }
                                for k in ds..de {
                                    let c = *da.get_unchecked(k as usize) as usize;
                                    let ng = num_good.get_unchecked_mut(c);
                                    let new_val = ng.saturating_sub(1);
                                    *ng = new_val;
                                    if new_val == 0 {
                                        unsat_count += 1;
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
                        current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                    }

                    last_check_unsat = unsat_count;
                }

                let rand_val = rng.gen::<usize>();
                let mut cid = 0usize;
                let mut found = false;
                while !residual.is_empty() {
                    let id = rand_val % residual.len();
                    cid = *residual.get_unchecked(id) as usize;
                    if *num_good.get_unchecked(cid) > 0 {
                        residual.swap_remove(id);
                    } else {
                        found = true;
                        break;
                    }
                }
                if !found { break; }

                let cs = *co.get_unchecked(cid) as usize;
                let ce = *co.get_unchecked(cid + 1) as usize;
                let clen = ce - cs;

                match clen {
                    2 => {
                        let ri = rand_val & 1;
                        if ri != 0 {
                            cl.swap(cs, cs + 1);
                        }
                    }
                    3 => {
                        let ri = rand_val % 3;
                        if ri != 0 {
                            cl.swap(cs, cs + ri);
                        }
                    }
                    _ => {
                        if clen > 1 {
                            let ri = rand_val % clen;
                            cl.swap(cs, cs + ri);
                        }
                    }
                }

                let l0 = *cl.get_unchecked(cs);
                let mut zero_found: Option<usize> = None;
                match clen {
                    1 => {
                        let abs0 = (l0.abs() - 1) as usize;
                        let (os0, oe0, arr0) = if *vars.get_unchecked(abs0) {
                            (*p_off.get_unchecked(abs0), *p_off.get_unchecked(abs0 + 1), &p_data)
                        } else {
                            (*n_off.get_unchecked(abs0), *n_off.get_unchecked(abs0 + 1), &n_data)
                        };
                        let mut breaks0 = false;
                        for k in os0..oe0 {
                            let c = *arr0.get_unchecked(k as usize) as usize;
                            if *num_good.get_unchecked(c) == 1 {
                                breaks0 = true;
                                break;
                            }
                        }
                        if !breaks0 {
                            zero_found = Some(abs0);
                        }
                    }
                    2 => {
                        let l1 = *cl.get_unchecked(cs + 1);

                        let abs0 = (l0.abs() - 1) as usize;
                        let (os0, oe0, arr0) = if *vars.get_unchecked(abs0) {
                            (*p_off.get_unchecked(abs0), *p_off.get_unchecked(abs0 + 1), &p_data)
                        } else {
                            (*n_off.get_unchecked(abs0), *n_off.get_unchecked(abs0 + 1), &n_data)
                        };
                        let mut breaks0 = false;
                        for k in os0..oe0 {
                            let c = *arr0.get_unchecked(k as usize) as usize;
                            if *num_good.get_unchecked(c) == 1 {
                                breaks0 = true;
                                break;
                            }
                        }
                        if !breaks0 {
                            zero_found = Some(abs0);
                        } else {
                            let abs1 = (l1.abs() - 1) as usize;
                            let (os1, oe1, arr1) = if *vars.get_unchecked(abs1) {
                                (*p_off.get_unchecked(abs1), *p_off.get_unchecked(abs1 + 1), &p_data)
                            } else {
                                (*n_off.get_unchecked(abs1), *n_off.get_unchecked(abs1 + 1), &n_data)
                            };
                            let mut breaks1 = false;
                            for k in os1..oe1 {
                                let c = *arr1.get_unchecked(k as usize) as usize;
                                if *num_good.get_unchecked(c) == 1 {
                                    breaks1 = true;
                                    break;
                                }
                            }
                            if !breaks1 {
                                zero_found = Some(abs1);
                            }
                        }
                    }
                    3 => {
                        let l1 = *cl.get_unchecked(cs + 1);
                        let l2 = *cl.get_unchecked(cs + 2);

                        let abs0 = (l0.abs() - 1) as usize;
                        let (os0, oe0, arr0) = if *vars.get_unchecked(abs0) {
                            (*p_off.get_unchecked(abs0), *p_off.get_unchecked(abs0 + 1), &p_data)
                        } else {
                            (*n_off.get_unchecked(abs0), *n_off.get_unchecked(abs0 + 1), &n_data)
                        };
                        let mut breaks0 = false;
                        for k in os0..oe0 {
                            let c = *arr0.get_unchecked(k as usize) as usize;
                            if *num_good.get_unchecked(c) == 1 {
                                breaks0 = true;
                                break;
                            }
                        }
                        if !breaks0 {
                            zero_found = Some(abs0);
                        } else {
                            let abs1 = (l1.abs() - 1) as usize;
                            let (os1, oe1, arr1) = if *vars.get_unchecked(abs1) {
                                (*p_off.get_unchecked(abs1), *p_off.get_unchecked(abs1 + 1), &p_data)
                            } else {
                                (*n_off.get_unchecked(abs1), *n_off.get_unchecked(abs1 + 1), &n_data)
                            };
                            let mut breaks1 = false;
                            for k in os1..oe1 {
                                let c = *arr1.get_unchecked(k as usize) as usize;
                                if *num_good.get_unchecked(c) == 1 {
                                    breaks1 = true;
                                    break;
                                }
                            }
                            if !breaks1 {
                                zero_found = Some(abs1);
                            } else {
                                let abs2 = (l2.abs() - 1) as usize;
                                let (os2, oe2, arr2) = if *vars.get_unchecked(abs2) {
                                    (*p_off.get_unchecked(abs2), *p_off.get_unchecked(abs2 + 1), &p_data)
                                } else {
                                    (*n_off.get_unchecked(abs2), *n_off.get_unchecked(abs2 + 1), &n_data)
                                };
                                let mut breaks2 = false;
                                for k in os2..oe2 {
                                    let c = *arr2.get_unchecked(k as usize) as usize;
                                    if *num_good.get_unchecked(c) == 1 {
                                        breaks2 = true;
                                        break;
                                    }
                                }
                                if !breaks2 {
                                    zero_found = Some(abs2);
                                }
                            }
                        }
                    }
                    _ => {
                        'outer_h: for j in cs..ce {
                            let l = *cl.get_unchecked(j);
                            let abs_l = (l.abs() - 1) as usize;
                            let (os, oe, arr) = if *vars.get_unchecked(abs_l) {
                                (*p_off.get_unchecked(abs_l), *p_off.get_unchecked(abs_l + 1), &p_data)
                            } else {
                                (*n_off.get_unchecked(abs_l), *n_off.get_unchecked(abs_l + 1), &n_data)
                            };
                            for k in os..oe {
                                let c = *arr.get_unchecked(k as usize) as usize;
                                if *num_good.get_unchecked(c) == 1 {
                                    continue 'outer_h;
                                }
                            }
                            zero_found = Some(abs_l);
                            break;
                        }
                    }
                }

                let v_idx = if let Some(v) = zero_found {
                    v
                } else if rng.gen::<f64>() < current_prob {
                    (l0.abs() - 1) as usize
                } else {
                    let mut min_sad = usize::MAX;
                    let mut v_min = (l0.abs() - 1) as usize;
                    let mut min_weight = usize::MAX;

                    match clen {
                        1 => {
                            let abs0 = (l0.abs() - 1) as usize;
                            let (os0, oe0, arr0) = if *vars.get_unchecked(abs0) {
                                (*p_off.get_unchecked(abs0), *p_off.get_unchecked(abs0 + 1), &p_data)
                            } else {
                                (*n_off.get_unchecked(abs0), *n_off.get_unchecked(abs0 + 1), &n_data)
                            };

                            let mut sad0 = 0usize;
                            for k in os0..oe0 {
                                let c = *arr0.get_unchecked(k as usize) as usize;
                                if *num_good.get_unchecked(c) == 1 {
                                    sad0 += 1;
                                }
                                if sad0 >= min_sad { break; }
                            }

                            if sad0 == 0 {
                                let appearances = *var_appearances.get_unchecked(abs0);
                                let age_bonus = (*var_age.get_unchecked(abs0) as usize) / 4;
                                let adjusted_weight = appearances.saturating_sub(age_bonus);
                                if min_sad > 0 || adjusted_weight < min_weight {
                                    v_min = abs0;
                                }
                            } else if min_sad > 0 {
                                let appearances = *var_appearances.get_unchecked(abs0);
                                let age_bonus = (*var_age.get_unchecked(abs0) as usize) / 2;
                                let combined_weight = sad0 * 1000 + appearances - age_bonus.min(50);
                                if combined_weight < min_weight {
                                    v_min = abs0;
                                }
                            }
                        }
                        2 => {
                            let l1 = *cl.get_unchecked(cs + 1);

                            let abs0 = (l0.abs() - 1) as usize;
                            let (os0, oe0, arr0) = if *vars.get_unchecked(abs0) {
                                (*p_off.get_unchecked(abs0), *p_off.get_unchecked(abs0 + 1), &p_data)
                            } else {
                                (*n_off.get_unchecked(abs0), *n_off.get_unchecked(abs0 + 1), &n_data)
                            };

                            let mut sad0 = 0usize;
                            for k in os0..oe0 {
                                let c = *arr0.get_unchecked(k as usize) as usize;
                                if *num_good.get_unchecked(c) == 1 {
                                    sad0 += 1;
                                }
                                if sad0 >= min_sad { break; }
                            }

                            if sad0 == 0 {
                                let appearances = *var_appearances.get_unchecked(abs0);
                                let age_bonus = (*var_age.get_unchecked(abs0) as usize) / 4;
                                let adjusted_weight = appearances.saturating_sub(age_bonus);
                                if min_sad > 0 || adjusted_weight < min_weight {
                                    min_sad = 0;
                                    min_weight = adjusted_weight;
                                    v_min = abs0;
                                }
                            } else if min_sad > 0 {
                                let appearances = *var_appearances.get_unchecked(abs0);
                                let age_bonus = (*var_age.get_unchecked(abs0) as usize) / 2;
                                let combined_weight = sad0 * 1000 + appearances - age_bonus.min(50);
                                if combined_weight < min_weight {
                                    min_sad = sad0;
                                    min_weight = combined_weight;
                                    v_min = abs0;
                                }
                            }

                            if min_sad == 0 || min_sad > 1 {
                                let abs1 = (l1.abs() - 1) as usize;
                                let (os1, oe1, arr1) = if *vars.get_unchecked(abs1) {
                                    (*p_off.get_unchecked(abs1), *p_off.get_unchecked(abs1 + 1), &p_data)
                                } else {
                                    (*n_off.get_unchecked(abs1), *n_off.get_unchecked(abs1 + 1), &n_data)
                                };

                                let mut sad1 = 0usize;
                                for k in os1..oe1 {
                                    let c = *arr1.get_unchecked(k as usize) as usize;
                                    if *num_good.get_unchecked(c) == 1 {
                                        sad1 += 1;
                                    }
                                    if sad1 >= min_sad { break; }
                                }

                                if sad1 == 0 {
                                    let appearances = *var_appearances.get_unchecked(abs1);
                                    let age_bonus = (*var_age.get_unchecked(abs1) as usize) / 4;
                                    let adjusted_weight = appearances.saturating_sub(age_bonus);
                                    if min_sad > 0 || adjusted_weight < min_weight {
                                        v_min = abs1;
                                    }
                                } else if min_sad > 0 {
                                    let appearances = *var_appearances.get_unchecked(abs1);
                                    let age_bonus = (*var_age.get_unchecked(abs1) as usize) / 2;
                                    let combined_weight = sad1 * 1000 + appearances - age_bonus.min(50);
                                    if combined_weight < min_weight {
                                        v_min = abs1;
                                    }
                                }
                            }
                        }
                        3 => {
                            let l1 = *cl.get_unchecked(cs + 1);
                            let l2 = *cl.get_unchecked(cs + 2);

                            let abs0 = (l0.abs() - 1) as usize;
                            let (os0, oe0, arr0) = if *vars.get_unchecked(abs0) {
                                (*p_off.get_unchecked(abs0), *p_off.get_unchecked(abs0 + 1), &p_data)
                            } else {
                                (*n_off.get_unchecked(abs0), *n_off.get_unchecked(abs0 + 1), &n_data)
                            };

                            let mut sad0 = 0usize;
                            for k in os0..oe0 {
                                let c = *arr0.get_unchecked(k as usize) as usize;
                                if *num_good.get_unchecked(c) == 1 {
                                    sad0 += 1;
                                }
                                if sad0 >= min_sad { break; }
                            }

                            if sad0 == 0 {
                                let appearances = *var_appearances.get_unchecked(abs0);
                                let age_bonus = (*var_age.get_unchecked(abs0) as usize) / 4;
                                let adjusted_weight = appearances.saturating_sub(age_bonus);
                                if min_sad > 0 || adjusted_weight < min_weight {
                                    min_sad = 0;
                                    min_weight = adjusted_weight;
                                    v_min = abs0;
                                }
                            } else if min_sad > 0 {
                                let appearances = *var_appearances.get_unchecked(abs0);
                                let age_bonus = (*var_age.get_unchecked(abs0) as usize) / 2;
                                let combined_weight = sad0 * 1000 + appearances - age_bonus.min(50);
                                if combined_weight < min_weight {
                                    min_sad = sad0;
                                    min_weight = combined_weight;
                                    v_min = abs0;
                                }
                            }

                            if min_sad == 0 || min_sad > 1 {
                                let abs1 = (l1.abs() - 1) as usize;
                                let (os1, oe1, arr1) = if *vars.get_unchecked(abs1) {
                                    (*p_off.get_unchecked(abs1), *p_off.get_unchecked(abs1 + 1), &p_data)
                                } else {
                                    (*n_off.get_unchecked(abs1), *n_off.get_unchecked(abs1 + 1), &n_data)
                                };

                                let mut sad1 = 0usize;
                                for k in os1..oe1 {
                                    let c = *arr1.get_unchecked(k as usize) as usize;
                                    if *num_good.get_unchecked(c) == 1 {
                                        sad1 += 1;
                                    }
                                    if sad1 >= min_sad { break; }
                                }

                                if sad1 == 0 {
                                    let appearances = *var_appearances.get_unchecked(abs1);
                                    let age_bonus = (*var_age.get_unchecked(abs1) as usize) / 4;
                                    let adjusted_weight = appearances.saturating_sub(age_bonus);
                                    if min_sad > 0 || adjusted_weight < min_weight {
                                        min_sad = 0;
                                        min_weight = adjusted_weight;
                                        v_min = abs1;
                                    }
                                } else if min_sad > 0 {
                                    let appearances = *var_appearances.get_unchecked(abs1);
                                    let age_bonus = (*var_age.get_unchecked(abs1) as usize) / 2;
                                    let combined_weight = sad1 * 1000 + appearances - age_bonus.min(50);
                                    if combined_weight < min_weight {
                                        min_sad = sad1;
                                        min_weight = combined_weight;
                                        v_min = abs1;
                                    }
                                }
                            }

                            if min_sad == 0 || min_sad > 1 {
                                let abs2 = (l2.abs() - 1) as usize;
                                let (os2, oe2, arr2) = if *vars.get_unchecked(abs2) {
                                    (*p_off.get_unchecked(abs2), *p_off.get_unchecked(abs2 + 1), &p_data)
                                } else {
                                    (*n_off.get_unchecked(abs2), *n_off.get_unchecked(abs2 + 1), &n_data)
                                };

                                let mut sad2 = 0usize;
                                for k in os2..oe2 {
                                    let c = *arr2.get_unchecked(k as usize) as usize;
                                    if *num_good.get_unchecked(c) == 1 {
                                        sad2 += 1;
                                    }
                                    if sad2 >= min_sad { break; }
                                }

                                if sad2 == 0 {
                                    let appearances = *var_appearances.get_unchecked(abs2);
                                    let age_bonus = (*var_age.get_unchecked(abs2) as usize) / 4;
                                    let adjusted_weight = appearances.saturating_sub(age_bonus);
                                    if min_sad > 0 || adjusted_weight < min_weight {
                                        v_min = abs2;
                                    }
                                } else if min_sad > 0 {
                                    let appearances = *var_appearances.get_unchecked(abs2);
                                    let age_bonus = (*var_age.get_unchecked(abs2) as usize) / 2;
                                    let combined_weight = sad2 * 1000 + appearances - age_bonus.min(50);
                                    if combined_weight < min_weight {
                                        v_min = abs2;
                                    }
                                }
                            }
                        }
                        _ => {
                            for j in cs..ce {
                                let l = *cl.get_unchecked(j);
                                let abs_l = (l.abs() - 1) as usize;
                                let (os, oe, arr) = if *vars.get_unchecked(abs_l) {
                                    (*p_off.get_unchecked(abs_l), *p_off.get_unchecked(abs_l + 1), &p_data)
                                } else {
                                    (*n_off.get_unchecked(abs_l), *n_off.get_unchecked(abs_l + 1), &n_data)
                                };

                                let mut sad = 0usize;
                                for k in os..oe {
                                    let c = *arr.get_unchecked(k as usize) as usize;
                                    if *num_good.get_unchecked(c) == 1 {
                                        sad += 1;
                                    }
                                    if sad >= min_sad { break; }
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
                                    if min_sad <= 1 { break; }
                                }
                            }
                        }
                    }
                    v_min
                };

                let was_true = *vars.get_unchecked(v_idx);
                let (is, ie, ia) = if was_true {
                    (*n_off.get_unchecked(v_idx), *n_off.get_unchecked(v_idx + 1), &n_data)
                } else {
                    (*p_off.get_unchecked(v_idx), *p_off.get_unchecked(v_idx + 1), &p_data)
                };
                let (ds, de, da) = if was_true {
                    (*p_off.get_unchecked(v_idx), *p_off.get_unchecked(v_idx + 1), &p_data)
                } else {
                    (*n_off.get_unchecked(v_idx), *n_off.get_unchecked(v_idx + 1), &n_data)
                };

                for k in is..ie {
                    let c = *ia.get_unchecked(k as usize) as usize;
                    let ng = *num_good.get_unchecked(c);
                    if ng == 0 { unsat_count -= 1; }
                    *num_good.get_unchecked_mut(c) = ng.saturating_add(1);
                }
                for k in ds..de {
                    let c = *da.get_unchecked(k as usize) as usize;
                    let ng = num_good.get_unchecked_mut(c);
                    let new_val = ng.saturating_sub(1);
                    *ng = new_val;
                    if new_val == 0 {
                        unsat_count += 1;
                        residual.push(c as u32);
                    }
                }

                *vars.get_unchecked_mut(v_idx) = !was_true;
                *var_age.get_unchecked_mut(v_idx) = 0;
                match clen {
                    1 => {
                        let var0 = (cl.get_unchecked(cs).abs() - 1) as usize;
                        let age0 = var_age.get_unchecked_mut(var0);
                        *age0 = age0.saturating_add(1);
                    }
                    2 => {
                        let var0 = (cl.get_unchecked(cs).abs() - 1) as usize;
                        let var1 = (cl.get_unchecked(cs + 1).abs() - 1) as usize;
                        let age0 = var_age.get_unchecked_mut(var0);
                        *age0 = age0.saturating_add(1);
                        let age1 = var_age.get_unchecked_mut(var1);
                        *age1 = age1.saturating_add(1);
                    }
                    3 => {
                        let var0 = (cl.get_unchecked(cs).abs() - 1) as usize;
                        let var1 = (cl.get_unchecked(cs + 1).abs() - 1) as usize;
                        let var2 = (cl.get_unchecked(cs + 2).abs() - 1) as usize;
                        let age0 = var_age.get_unchecked_mut(var0);
                        *age0 = age0.saturating_add(1);
                        let age1 = var_age.get_unchecked_mut(var1);
                        *age1 = age1.saturating_add(1);
                        let age2 = var_age.get_unchecked_mut(var2);
                        *age2 = age2.saturating_add(1);
                    }
                    _ => {
                        for j in cs..ce {
                            let l = *cl.get_unchecked(j);
                            let var = (l.abs() - 1) as usize;
                            let age = var_age.get_unchecked_mut(var);
                            *age = age.saturating_add(1);
                        }
                    }
                }

                rounds += 1;
                if unsat_count < best_unsat && unsat_count <= 12 {
                    best_unsat = unsat_count;
                    best_vars.clone_from(&vars);
                }
            }
        }

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            best_vars.clone_from(&vars);
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }
    }

    let _ = save_solution(&Solution { variables: best_vars });
    Ok(())
}