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

    let default_fuel = if nv >= 10000 { 125_000_000_000.0 } else { 250_000_000_000.0 };
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(default_fuel);

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let mut max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
    let base_max_flips = max_flips;

    let mut vars = vec![false; nv];
    // Compute clause lengths
    let mut max_len = 0usize;
    let mut lengths = vec![0usize; nc];
    for i in 0..nc {
        let len = (co[i + 1] - co[i]) as usize;
        lengths[i] = len;
        if len > max_len {
            max_len = len;
        }
    }
    // Bucket sort clauses by length
    let mut buckets: Vec<Vec<usize>> = (0..=max_len).map(|_| Vec::new()).collect();
    for i in 0..nc {
        buckets[lengths[i]].push(i);
    }

    // Greedy assignment: satisfy shortest clauses first
    for l in 1..=max_len {
        for &cid in buckets[l].iter() {
            let s = co[cid] as usize;
            let e = co[cid + 1] as usize;
            // Check if clause already satisfied
            let mut already = false;
            for j in s..e {
                let lit = cl[j];
                let v = (lit.abs() - 1) as usize;
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    already = true;
                    break;
                }
            }
            if already {
                continue;
            }

            // Choose literal with highest literal count (p_cnt for positive, n_cnt for negative)
            let mut best_score: u32 = 0;
            let mut best_v = 0usize;
            let mut best_target = false;
            let mut count = 0;
            for j in s..e {
                let lit = cl[j];
                let v = (lit.abs() - 1) as usize;
                let target_val = lit > 0;
                if vars[v] == target_val {
                    // already satisfied, can't happen due to check above
                    continue;
                }
                let score = if target_val { p_cnt[v] } else { n_cnt[v] };
                if score > best_score {
                    best_score = score;
                    best_v = v;
                    best_target = target_val;
                    count = 1;
                } else if score == best_score {
                    count += 1;
                    if rng.gen::<usize>() % count == 0 {
                        best_v = v;
                        best_target = target_val;
                    }
                }
            }
            if best_score == 0 {
                // All scores zero, pick any random literal
                let idx = rng.gen::<usize>() % (e - s);
                let lit = cl[s + idx];
                best_v = (lit.abs() - 1) as usize;
                best_target = lit > 0;
            }
            vars[best_v] = best_target;
        }
    }

    // Build num_good and residual from final assignment
    let mut num_good = vec![0u8; nc];
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut good = 0u8;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                good = good.saturating_add(1);
            }
        }
        num_good[i] = good;
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut true_unsat = 0usize;
    for i in 0..nc {
        if num_good[i] == 0 {
            residual.push(i as u32);
            true_unsat += 1;
        }
    }

    if true_unsat == 0 {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let mut best_unsat = true_unsat;
    let mut best_vars = vars.clone();

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let mut probsat_weights = vec![0.0f64; nc + 1];
    if avg_clause_size <= 3.2 {
        let cb: f64 = 2.06;
        for i in 0..=nc {
            probsat_weights[i] = cb.powf(-(i as f64));
        }
    } else {
        let cb: f64 = if avg_clause_size <= 4.2 {
            2.85
        } else if avg_clause_size <= 5.2 {
            3.7
        } else if avg_clause_size <= 6.2 {
            5.1
        } else {
            5.4
        };
        for i in 0..=nc {
            probsat_weights[i] = (i as f64 + 1.0).powf(-cb);
        }
    }

    let mut last_check_unsat = true_unsat;
    let mut rounds = 0usize;
    let mut stagnation = 0usize;

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if true_unsat == 0 { break; }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_unsat as i64 - true_unsat as i64;
                let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= 4 {
                        if stagnation >= 15 {
                            vars.copy_from_slice(&best_vars);
                            let perturb_cnt = (nv / 20).max(1);
                            for _ in 0..perturb_cnt {
                                let v = rng.gen::<usize>() % nv;
                                *vars.get_unchecked_mut(v) = !*vars.get_unchecked(v);
                            }
                            
                            residual.clear();
                            true_unsat = 0;
                            for i in 0..nc {
                                let s = *co.get_unchecked(i) as usize;
                                let e = *co.get_unchecked(i + 1) as usize;
                                let mut good = 0u8;
                                for j in s..e {
                                    let l = *cl.get_unchecked(j);
                                    let v = (l.abs() - 1) as usize;
                                    if (l > 0 && *vars.get_unchecked(v)) || (l < 0 && !*vars.get_unchecked(v)) {
                                        good = good.saturating_add(1);
                                    }
                                }
                                *num_good.get_unchecked_mut(i) = good;
                                if good == 0 {
                                    residual.push(i as u32);
                                    true_unsat += 1;
                                }
                            }
                            stagnation = 0;
                        } else {
                            let kicks = if stagnation >= 8 { 6 } else { 3 };
                            for _ in 0..kicks {
                                if true_unsat == 0 { break; }
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
                                    let ng = num_good.get_unchecked_mut(c);
                                    if *ng == 0 { true_unsat = true_unsat.saturating_sub(1); }
                                    *ng = ng.saturating_add(1);
                                }
                                for k in ds..de {
                                    let c = *all_data.get_unchecked(k) as usize;
                                    let ng = num_good.get_unchecked_mut(c);
                                    let new_val = ng.saturating_sub(1);
                                    *ng = new_val;
                                    if new_val == 0 {
                                        residual.push(c as u32);
                                        true_unsat += 1;
                                    }
                                }
                                *vars.get_unchecked_mut(v) = !was_true;
                            }
                            stagnation = 0;
                        }
                    }
                } else if progress_ratio > progress_threshold {
                    stagnation = 0;
                } else {
                    stagnation = 0;
                }

                last_check_unsat = true_unsat;

                // Dynamic budget adjustment
                if progress_ratio > 0.2 {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                }
                if stagnation >= 10 {
                    max_flips = rounds + (max_flips - rounds) / 2;
                    max_flips = max_flips.max(base_max_flips / 10);
                }
            }

            if true_unsat == 0 { break; }

            let mut cid = usize::MAX;
            let mut min_len = usize::MAX;
            for _ in 0..3 {
                while !residual.is_empty() {
                    let id = rng.gen::<usize>() % residual.len();
                    let cand = *residual.get_unchecked(id) as usize;
                    if *num_good.get_unchecked(cand) > 0 {
                        residual.swap_remove(id);
                    } else {
                        let c_s = *co.get_unchecked(cand) as usize;
                        let c_e = *co.get_unchecked(cand + 1) as usize;
                        let clen = c_e - c_s;
                        if clen < min_len {
                            min_len = clen;
                            cid = cand;
                        }
                        break;
                    }
                }
                if residual.is_empty() { break; }
            }
            if cid == usize::MAX { break; }

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rng.gen::<usize>() % clen;
                cl.swap(cs, cs + ri);
            }

            let clen_actual = clen.min(256);
            let mut total_weight = 0.0;
            let mut weights = [0.0; 256];
            let mut v_idx = (cl.get_unchecked(cs).abs() - 1) as usize;
            let mut found_zero = false;

            for idx in 0..clen_actual {
                let j = cs + idx;
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if *vars.get_unchecked(abs_l) {
                    (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                } else {
                    (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                };

                let mut sad = 0usize;
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        sad += 1;
                    }
                }

                if sad == 0 {
                    v_idx = abs_l;
                    found_zero = true;
                    break;
                }

                let w = *probsat_weights.get_unchecked(sad.min(nc));
                weights[idx] = w;
                total_weight += w;
            }

            if !found_zero {
                let mut r = rng.gen::<f64>() * total_weight;
                for idx in 0..clen_actual {
                    r -= weights[idx];
                    if r <= 0.0 {
                        v_idx = (cl.get_unchecked(cs + idx).abs() - 1) as usize;
                        break;
                    }
                }
            }

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
                let ng = num_good.get_unchecked_mut(c);
                if *ng == 0 { true_unsat = true_unsat.saturating_sub(1); }
                *ng = ng.saturating_add(1);
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = num_good.get_unchecked_mut(c);
                let new_val = ng.saturating_sub(1);
                *ng = new_val;
                if new_val == 0 {
                    residual.push(c as u32);
                    true_unsat += 1;
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            
            if true_unsat < best_unsat {
                best_unsat = true_unsat;
                best_vars.copy_from_slice(&vars);
            }

            rounds += 1;
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}