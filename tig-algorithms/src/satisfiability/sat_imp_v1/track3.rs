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

    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if nn == 0 && np > 0 { vars[v] = true; continue; }
        if np == 0 && nn > 0 { continue; }
        let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
        if vad <= nad {
            vars[v] = rng.gen_bool(random_threshold);
        } else {
            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
            vars[v] = rng.gen_bool(prob);
        }
    }

    let mut num_good = vec![0u8; nc];
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

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        if num_good[i] == 0 {
            residual.push(i as u32);
        }
    }

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob: f64 = 0.52;
    let mut current_prob = base_prob;
    let max_random_prob: f64 = 0.9;
    let prob_adjustment_factor: f64 = 0.025;
    let smoothing_factor: f64 = 0.8;

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;

    let mut last_check_residual = residual.len();
    let mut var_age = vec![0u16; nv];
    let mut rounds = 0usize;
    let mut stagnation = 0usize;

    unsafe {
        loop {
            if rounds >= max_flips { break; }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    stagnation += 1;
                    let prob_adjustment = prob_adjustment_factor * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= 4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        let sampled_clauses = if residual.len() <= 32 { 3 } else { 2 };
                        for _ in 0..kicks {
                            if residual.is_empty() { break; }

                            let mut best_v = usize::MAX;
                            let mut best_score = isize::MIN;
                            let mut best_make = 0usize;
                            let mut best_break = usize::MAX;
                            let mut best_age = 0usize;
                            let mut fallback_v = usize::MAX;

                            for _ in 0..sampled_clauses {
                                if residual.is_empty() { break; }
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
                                fallback_v = (lit.abs() - 1) as usize;

                                for j in pcs..pce {
                                    let lit = *cl.get_unchecked(j);
                                    let v = (lit.abs() - 1) as usize;
                                    let (ms, me, ma) = if lit > 0 {
                                        (*p_off.get_unchecked(v), *p_off.get_unchecked(v + 1), &p_data)
                                    } else {
                                        (*n_off.get_unchecked(v), *n_off.get_unchecked(v + 1), &n_data)
                                    };
                                    let (bs, be, ba) = if lit > 0 {
                                        (*n_off.get_unchecked(v), *n_off.get_unchecked(v + 1), &n_data)
                                    } else {
                                        (*p_off.get_unchecked(v), *p_off.get_unchecked(v + 1), &p_data)
                                    };

                                    let mut make = 0usize;
                                    for k in ms..me {
                                        let c = *ma.get_unchecked(k as usize) as usize;
                                        if *num_good.get_unchecked(c) == 0 {
                                            make += 1;
                                        }
                                    }

                                    let mut br = 0usize;
                                    for k in bs..be {
                                        let c = *ba.get_unchecked(k as usize) as usize;
                                        if *num_good.get_unchecked(c) == 1 {
                                            br += 1;
                                        }
                                    }

                                    let age = *var_age.get_unchecked(v) as usize;
                                    let score = make as isize - br as isize;
                                    if score > best_score
                                        || (score == best_score && (make > best_make
                                            || (make == best_make && (br < best_break
                                                || (br == best_break && age > best_age)))))
                                    {
                                        best_v = v;
                                        best_score = score;
                                        best_make = make;
                                        best_break = br;
                                        best_age = age;
                                    }
                                }
                            }

                            let v = if best_v != usize::MAX {
                                best_v
                            } else if fallback_v != usize::MAX {
                                fallback_v
                            } else {
                                continue;
                            };

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
                                let ng = num_good.get_unchecked_mut(c);
                                *ng += 1;
                            }
                            for k in ds..de {
                                let c = *da.get_unchecked(k as usize) as usize;
                                let ng = num_good.get_unchecked_mut(c);
                                *ng -= 1;
                                if *ng == 0 {
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

                last_check_residual = residual.len();
            }

            if residual.is_empty() { break; }

            let rand_val = rng.gen::<usize>();
            let mut cid = 0usize;
            while !residual.is_empty() {
                let id = rand_val % residual.len();
                cid = *residual.get_unchecked(id) as usize;
                if *num_good.get_unchecked(cid) > 0 {
                    residual.swap_remove(id);
                } else {
                    break;
                }
            }
            if residual.is_empty() { break; }

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
                let (os, oe, arr) = if l > 0 {
                    (*n_off.get_unchecked(abs_l), *n_off.get_unchecked(abs_l + 1), &n_data)
                } else {
                    (*p_off.get_unchecked(abs_l), *p_off.get_unchecked(abs_l + 1), &p_data)
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
                    let (os, oe, arr) = if l > 0 {
                        (*n_off.get_unchecked(abs_l), *n_off.get_unchecked(abs_l + 1), &n_data)
                    } else {
                        (*p_off.get_unchecked(abs_l), *p_off.get_unchecked(abs_l + 1), &p_data)
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
                let ng = num_good.get_unchecked_mut(c);
                *ng += 1;
            }
            for k in ds..de {
                let c = *da.get_unchecked(k as usize) as usize;
                let ng = num_good.get_unchecked_mut(c);
                *ng -= 1;
                if *ng == 0 {
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