use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;

pub fn solve(
    hp: &Option<Hyperparameters>,
    rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: Vec<u32>, n_cnt: Vec<u32>,
    all_off: &[u32], p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>, co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nvf = nv as f64;
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_low).unwrap_or(150_000_000_000.0);
    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

    let mut vars = vec![false; nv];
    let nad = 1.0;
    let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    for v in 0..nv {
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
        if np == 0.0 { continue; }
        let vad = np / nn;
        let bias_prob = (np + 0.25) / (np + nn + 1.2);
        let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
        let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
        vars[v] = rng.gen_bool(prob);
    }

    let appearances: Vec<u8> = (0..nv).map(|v| {
        ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
    }).collect();
    drop(p_cnt);
    drop(n_cnt);

    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];

    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let shift = (i & 3) << 1;
        let byte_idx = i >> 2;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                num_good[byte_idx] += 1u8 << shift;
            }
        }
    }

    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc];
    for i in 0..nc {
        if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
            unsat_pos[i] = unsat_list.len() as u32;
            unsat_list.push(i as u32);
        }
    }

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob = hp.as_ref().and_then(|h| h.base_prob)
        .unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
    let mut current_prob = base_prob;

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
    let max_random_prob = hp.as_ref().and_then(|h| h.max_prob).unwrap_or(0.9);
    let prob_adjustment_factor = 0.03;
    let smoothing_factor = 0.8;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp.as_ref().and_then(|h| h.perturbation_flips)
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp.as_ref().and_then(|h| h.stagnation_limit)
        .unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);

    let mut last_check_residual = unsat_list.len();
    let mut stagnation = 0usize;
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;
    let mut rounds = 0usize;

    const STAGNATION_RESTART_THRESHOLD: usize = 120_000_000;
    const RESTART_LIMIT: usize = 3;
    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut flips_since_improvement: usize = 0;
    let mut restart_count: usize = 0;

    let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    unsafe {
        loop {
            if unsat_list.is_empty() || rounds >= max_flips { break; }

            flips_since_improvement += 1;
            countdown -= 1;
            if countdown == 0 {
                countdown = check_interval;
                let progress = last_check_residual as i64 - unsat_list.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;

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
                            if unsat_list.is_empty() { break; }
                            let rid = rng.gen::<usize>() % unsat_list.len();
                            let pcid = *unsat_list.get_unchecked(rid) as usize;

                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce { continue; }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = (lit.abs() - 1) as usize;

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            } else {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            };
                            let (ds, de) = if was_true {
                                (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                            } else {
                                (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                            };

                            for k in is..ie {
                                let c = *all_data.get_unchecked(k) as usize;
                                let shift = (c & 3) << 1;
                                let byte_idx = c >> 2;
                                let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
                                if old == 0 {
                                    let pos = *unsat_pos.get_unchecked(c) as usize;
                                    let last_idx = unsat_list.len() - 1;
                                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                                    unsat_list.set_len(last_idx);
                                }
                            }

                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let shift = (c & 3) << 1;
                                let byte_idx = c >> 2;
                                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                                if ng_before == 1 {
                                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                                    unsat_list.push(c as u32);
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

                last_check_residual = unsat_list.len();
                if unsat_list.len() < best_unsat {
                    best_unsat = unsat_list.len();
                    best_vars.copy_from_slice(&vars);
                    flips_since_improvement = 0;
                }
            }

            let rand_val = rng.gen::<usize>();

            if unsat_list.is_empty() { break; }
            let cid = *unsat_list.get_unchecked(rand_val % unsat_list.len()) as usize;

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                cl.swap(cs, cs + ri);
            }

            let mut zero_buf: [usize; 3] = [0; 3];
            let mut zero_cnt: usize = 0;
            'outer: for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if l > 0 {
                    (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                } else {
                    (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                };
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                        continue 'outer;
                    }
                }
                *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                zero_cnt += 1;
            }

            let v_idx = if zero_cnt > 0 {
                if zero_cnt == 1 {
                    *zero_buf.get_unchecked(0)
                } else {
                    *zero_buf.get_unchecked(rand_val % zero_cnt)
                }
            } else if rng.gen::<f64>() < current_prob {
                (cl.get_unchecked(cs).abs() - 1) as usize
            } else {
                let mut min_sad = usize::MAX;
                let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
                let mut min_weight = usize::MAX;

                for j in cs..ce {
                    let l = *cl.get_unchecked(j);
                    let abs_l = (l.abs() - 1) as usize;
                    let (os, oe) = if l > 0 {
                        (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                    } else {
                        (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                    };
                    let mut sad = 0usize;
                    for k in os..oe {
                        let c = *all_data.get_unchecked(k) as usize;
                        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                            sad += 1;
                        }
                        if sad >= min_sad { break; }
                    }

                    if sad == 0 {
                        let app = *appearances.get_unchecked(abs_l) as usize;
                        let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
                        let adjusted_weight = app.saturating_sub(age_bonus);
                        if min_sad > 0 || adjusted_weight < min_weight {
                            min_sad = 0;
                            min_weight = adjusted_weight;
                            v_min = abs_l;
                        }
                    } else if min_sad > 0 {
                        let app = *appearances.get_unchecked(abs_l) as usize;
                        let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
                        let combined_weight = sad * sad * 1024 + app - age_bonus.min(50);
                        if combined_weight < min_weight {
                            min_sad = sad;
                            min_weight = combined_weight;
                            v_min = abs_l;
                        }
                        if min_sad <= 1 { break; }
                    }
                }
                v_min
            };

            let was_true = *vars.get_unchecked(v_idx);
            let (is, ie) = if was_true {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            } else {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            };
            let (ds, de) = if was_true {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            } else {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            };

            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let old = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                *num_good.get_unchecked_mut(byte_idx) += 1u8 << shift;
                if old == 0 {
                    let pos = *unsat_pos.get_unchecked(c) as usize;
                    let last_idx = unsat_list.len() - 1;
                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                    unsat_list.set_len(last_idx);
                }
            }

            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                if ng_before == 1 {
                    *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                    unsat_list.push(c as u32);
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

            if flips_since_improvement >= STAGNATION_RESTART_THRESHOLD {
                if restart_count >= RESTART_LIMIT {
                    let _ = save_solution(&Solution { variables: best_vars });
                    return Ok(());
                }
                vars.copy_from_slice(&best_vars);
                num_good.fill(0);
                unsat_list.clear();
                unsat_pos.fill(u32::MAX);
                for c in 0..nc {
                    let s = *co.get_unchecked(c) as usize;
                    let e = *co.get_unchecked(c + 1) as usize;
                    let shift = (c & 3) << 1;
                    let byte_idx = c >> 2;
                    let mut g = 0u8;
                    for j in s..e {
                        let lit = *cl.get_unchecked(j);
                        let v = (lit.abs() - 1) as usize;
                        if (lit > 0 && *vars.get_unchecked(v)) || (lit < 0 && !*vars.get_unchecked(v)) {
                            g += 1;
                        }
                    }
                    *num_good.get_unchecked_mut(byte_idx) |= g << shift;
                    if g == 0 {
                        *unsat_pos.get_unchecked_mut(c) = unsat_list.len() as u32;
                        unsat_list.push(c as u32);
                    }
                }
                restart_count += 1;
                flips_since_improvement = 0;
                stagnation = 0;
                current_prob = base_prob;
                last_check_residual = unsat_list.len();
            }
        }
    }

    let baseline_sat = nc - unsat_list.len();
    let fuel_budget: u64 = 5_000_000_000;
    let mut refined_vars = vars.clone();
    refine_solution(cl, co, all_off, p_bound, all_data, &mut refined_vars, nv, fuel_budget);
    let mut refined_sat = 0usize;
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && refined_vars[v]) || (l < 0 && !refined_vars[v]) {
                refined_sat += 1;
                break;
            }
        }
    }
    let final_vars = if refined_sat > baseline_sat { refined_vars } else { vars };
    let _ = save_solution(&Solution { variables: final_vars });
    Ok(())
}

// ============================================================
// V3 HEURISTIC SEARCH — refine_pass scaffolding
// ============================================================
// EVOLVE-BLOCK-START refine_pass
fn refine_solution(
    cl: &[i32],
    co: &[u32],
    all_off: &[u32],
    p_bound: &[u32],
    all_data: &[u32],
    vars: &mut [bool],
    nv: usize,
    fuel_budget: u64,
) {
    // PASS 1: rebuild ng/unsat + purity flips
    let nc = co.len() - 1;
    let ng_len = (nc + 3) >> 2;
    let mut ng = vec![0u8; ng_len];
    let mut unsat: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc];
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let shift = (i & 3) << 1;
        let bi = i >> 2;
        let mut g = 0u8;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1u8; }
        }
        ng[bi] |= g << shift;
        if g == 0 {
            unsat_pos[i] = unsat.len() as u32;
            unsat.push(i as u32);
        }
    }
    if !unsat.is_empty() {
        let mut purity = vec![0u8; nv];
        for &cid in &unsat {
            let s = co[cid as usize] as usize;
            let e = co[cid as usize + 1] as usize;
            for j in s..e {
                let l = cl[j];
                let v = (l.abs() - 1) as usize;
                let pol = if l > 0 { 1u8 } else { 2u8 };
                unsafe {
                    let p = purity.get_unchecked_mut(v);
                    *p = if *p == 0 { pol } else if *p != pol { 3 } else { *p };
                }
            }
        }
        let mut flips: Vec<(usize, bool)> = Vec::new();
        for v in 0..nv {
            if purity[v] == 1 && !vars[v] {
                let neg_start = p_bound[v] as usize;
                let neg_end = all_off[v + 1] as usize;
                let mut safe = true;
                for k in neg_start..neg_end {
                    let c = all_data[k] as usize;
                    let cs = co[c] as usize;
                    let ce = co[c + 1] as usize;
                    let mut other_true = false;
                    for j in cs..ce {
                        let lit = cl[j];
                        if (lit.abs() - 1) as usize == v { continue; }
                        let w = (lit.abs() - 1) as usize;
                        if (lit > 0 && vars[w]) || (lit < 0 && !vars[w]) {
                            other_true = true;
                            break;
                        }
                    }
                    if !other_true { safe = false; break; }
                }
                if safe { flips.push((v, true)); }
            } else if purity[v] == 2 && vars[v] {
                let pos_start = all_off[v] as usize;
                let pos_end = p_bound[v] as usize;
                let mut safe = true;
                for k in pos_start..pos_end {
                    let c = all_data[k] as usize;
                    let cs = co[c] as usize;
                    let ce = co[c + 1] as usize;
                    let mut other_true = false;
                    for j in cs..ce {
                        let lit = cl[j];
                        if (lit.abs() - 1) as usize == v { continue; }
                        let w = (lit.abs() - 1) as usize;
                        if (lit > 0 && vars[w]) || (lit < 0 && !vars[w]) {
                            other_true = true;
                            break;
                        }
                    }
                    if !other_true { safe = false; break; }
                }
                if safe { flips.push((v, false)); }
            }
        }
        for (v, target) in &flips {
            vars[*v] = *target;
            let (post_start, post_end) = if *target {
                (all_off[*v] as usize, p_bound[*v] as usize)
            } else {
                (p_bound[*v] as usize, all_off[*v + 1] as usize)
            };
            let (neg_start, neg_end) = if *target {
                (p_bound[*v] as usize, all_off[*v + 1] as usize)
            } else {
                (all_off[*v] as usize, p_bound[*v] as usize)
            };
            unsafe {
                for k in post_start..post_end {
                    let c = all_data[k] as usize;
                    let shift = (c & 3) << 1;
                    let bi = c >> 2;
                    let old = (ng.get_unchecked(bi) >> shift) & 3;
                    *ng.get_unchecked_mut(bi) += 1u8 << shift;
                    if old == 0 {
                        let pos = unsat_pos[c] as usize;
                        let last = unsat.len() - 1;
                        let lc = unsat[last] as usize;
                        unsat[pos] = lc as u32;
                        unsat_pos[lc] = pos as u32;
                        unsat_pos[c] = u32::MAX;
                        unsat.set_len(last);
                    }
                }
                for k in neg_start..neg_end {
                    let c = all_data[k] as usize;
                    let shift = (c & 3) << 1;
                    let bi = c >> 2;
                    let old = (ng.get_unchecked(bi) >> shift) & 3;
                    *ng.get_unchecked_mut(bi) -= 1u8 << shift;
                    if old == 1 {
                        unsat_pos[c] = unsat.len() as u32;
                        unsat.push(c as u32);
                    }
                }
            }
        }
    }
    // PASS 2: Greedy net-gain flips (appended after purity flips)
    if !unsat.is_empty() {
        let max_pass2_flips = {
            let from_fuel = (fuel_budget / (3 * 200)) as usize;
            from_fuel.min(20_000)
        };
        let mut seed: u64 = 0x123456789ABCDEF0;
        let mut prev_unsat_len: usize = unsat.len();
        let mut stall_count: u32 = 0;
        for _step in 0..max_pass2_flips {
            if unsat.is_empty() { break; }
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            let rnd = seed.wrapping_mul(0x2545F4914F6CDD1D);
            let cid = unsat[(rnd as usize) % unsat.len()] as usize;
            let cs = co[cid] as usize;
            let ce = co[cid + 1] as usize;
            let mut best_v = usize::MAX;
            let mut best_net: i32 = i32::MIN;
            for j in cs..ce {
                let lit = cl[j];
                let v_idx = (lit.abs() - 1) as usize;
                let val = vars[v_idx];
                let mut break_cnt: i32 = 0;
                let (br_s, br_e) = if val {
                    (all_off[v_idx] as usize, p_bound[v_idx] as usize)
                } else {
                    (p_bound[v_idx] as usize, all_off[v_idx + 1] as usize)
                };
                for k in br_s..br_e {
                    let c = all_data[k] as usize;
                    let shift = (c & 3) << 1;
                    let byte_idx = c >> 2;
                    if ((ng[byte_idx] >> shift) & 3) == 1 {
                        break_cnt += 1;
                    }
                }
                let mut make_cnt: i32 = 0;
                let (mk_s, mk_e) = if val {
                    (p_bound[v_idx] as usize, all_off[v_idx + 1] as usize)
                } else {
                    (all_off[v_idx] as usize, p_bound[v_idx] as usize)
                };
                for k in mk_s..mk_e {
                    let c = all_data[k] as usize;
                    if unsat_pos[c] != u32::MAX {
                        make_cnt += 1;
                    }
                }
                let net = make_cnt - break_cnt;
                if net > best_net {
                    best_net = net;
                    best_v = v_idx;
                }
            }
            if best_net < 0 { continue; }
            let was_true = vars[best_v];
            vars[best_v] = !was_true;
            let (is, ie) = if was_true {
                (p_bound[best_v] as usize, all_off[best_v + 1] as usize)
            } else {
                (all_off[best_v] as usize, p_bound[best_v] as usize)
            };
            let (ds, de) = if was_true {
                (all_off[best_v] as usize, p_bound[best_v] as usize)
            } else {
                (p_bound[best_v] as usize, all_off[best_v + 1] as usize)
            };
            for k in is..ie {
                let c = all_data[k] as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let old = (ng[byte_idx] >> shift) & 3;
                ng[byte_idx] += 1u8 << shift;
                if old == 0 {
                    let pos = unsat_pos[c] as usize;
                    let last_idx = unsat.len() - 1;
                    let last_c = unsat[last_idx] as usize;
                    unsat[pos] = last_c as u32;
                    unsat_pos[last_c] = pos as u32;
                    unsat_pos[c] = u32::MAX;
                    unsat.pop();
                }
            }
            for k in ds..de {
                let c = all_data[k] as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let ng_before = (ng[byte_idx] >> shift) & 3;
                ng[byte_idx] -= 1u8 << shift;
                if ng_before == 1 {
                    unsat_pos[c] = unsat.len() as u32;
                    unsat.push(c as u32);
                }
            }
            let cur_unsat_len = unsat.len();
            if cur_unsat_len >= prev_unsat_len {
                stall_count += 1;
                if stall_count >= 2 { break; }
            } else {
                stall_count = 0;
            }
            prev_unsat_len = cur_unsat_len;
        }
    }
}
// EVOLVE-BLOCK-END
