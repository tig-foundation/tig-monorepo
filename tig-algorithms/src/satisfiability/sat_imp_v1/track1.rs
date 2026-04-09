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
    let mut kept = Vec::<[i32; 3]>::with_capacity(challenge.clauses.len());
    let mut kept_len = Vec::<u8>::with_capacity(challenge.clauses.len());

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }

        let mut lits = [0i32; 3];
        lits[0] = a;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        let mut len = 1usize;

        if b != a {
            lits[len] = b;
            len += 1;
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }

        if c != a && c != b {
            lits[len] = c;
            len += 1;
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }

        kept.push(lits);
        kept_len.push(len as u8);
    }

    let nc = kept.len();

    let mut p_off = vec![0u32; nv + 1];
    let mut n_off = vec![0u32; nv + 1];
    for v in 0..nv {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    let mut p_data = vec![0u32; p_off[nv] as usize];
    let mut n_data = vec![0u32; n_off[nv] as usize];

    p_cnt.copy_from_slice(&p_off[..nv]);
    n_cnt.copy_from_slice(&n_off[..nv]);

    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    for ci in 0..nc {
        let lits = &kept[ci];
        let len = kept_len[ci] as usize;
        for j in 0..len {
            let lit = lits[j];
            cl.push(lit);
            let v = (lit.abs() - 1) as usize;
            if lit > 0 {
                let pos = p_cnt[v] as usize;
                p_data[pos] = ci as u32;
                p_cnt[v] += 1;
            } else {
                let pos = n_cnt[v] as usize;
                n_data[pos] = ci as u32;
                n_cnt[v] += 1;
            }
        }
        co.push(cl.len() as u32);
    }

    for v in 0..nv {
        p_cnt[v] = p_off[v + 1] - p_off[v];
        n_cnt[v] = n_off[v + 1] - n_off[v];
    }

    let density = nc as f64 / nv as f64;

    solve_high_density(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, p_off, n_off, p_data, n_data, &mut cl, &co, save_solution)
}

#[inline(always)]
fn rebuild_clause_state(
    vars: &[bool],
    cl: &[i32],
    co: &[u32],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
) -> bool {
    residual.clear();
    let nc = num_good.len();
    unsafe {
        for i in 0..nc {
            let s = *co.get_unchecked(i) as usize;
            let e = *co.get_unchecked(i + 1) as usize;
            let mut good = 0u8;
            for j in s..e {
                let l = *cl.get_unchecked(j);
                let v = (l.abs() - 1) as usize;
                if (l > 0 && *vars.get_unchecked(v)) || (l < 0 && !*vars.get_unchecked(v)) {
                    good += 1;
                }
            }
            *num_good.get_unchecked_mut(i) = good;
            if good == 0 {
                residual.push(i as u32);
            }
        }
    }
    residual.is_empty()
}

fn solve_high_density(
    hp: &Option<Hyperparameters>,
    rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: Vec<u32>, n_cnt: Vec<u32>,
    p_off: Vec<u32>, n_off: Vec<u32>,
    p_data: Vec<u32>, n_data: Vec<u32>,
    cl: &mut Vec<i32>, co: &Vec<u32>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
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
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    if rebuild_clause_state(&vars, cl.as_slice(), co.as_slice(), num_good.as_mut_slice(), &mut residual) {
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
    let mut best_unsat = residual.len();
    let mut best_vars = vars.clone();
    let restart_enabled = max_flips >= (check_interval * 32).max(20_000) && (density >= 3.9 || nv >= 12000);
    let max_restarts = if restart_enabled {
        if max_flips >= 220_000 { 4 } else if max_flips >= 90_000 { 3 } else { 2 }
    } else {
        0
    };
    let mut restart_count = 0usize;
    let mut last_flip = vec![0u32; nv];
    let mut rounds = 0usize;
    let mut stagnation = 0usize;

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            let r32 = rounds as u32;

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                if residual.len() < best_unsat {
                    best_unsat = residual.len();
                    best_vars.clone_from(&vars);
                }
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    stagnation += 1;
                    let prob_adjustment = prob_adjustment_factor * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= 4 {
                        if restart_count < max_restarts {
                            if residual.len() < best_unsat {
                                best_unsat = residual.len();
                                best_vars.clone_from(&vars);
                            }
                            restart_count += 1;
                            let exploratory = (restart_count & 1) == 1;

                            for v in 0..nv {
                                let np = p_cnt[v] as usize;
                                let nn = n_cnt[v] as usize;
                                if nn == 0 && np > 0 {
                                    vars[v] = true;
                                    continue;
                                }
                                if np == 0 && nn > 0 {
                                    vars[v] = false;
                                    continue;
                                }
                                let bias_prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                                if exploratory {
                                    let total = (np + nn).max(1);
                                    let imbalance = ((np as i64 - nn as i64).abs() as f64) / total as f64;
                                    let prob = if imbalance < 0.10 {
                                        0.5
                                    } else if imbalance < 0.22 {
                                        0.5 + (bias_prob - 0.5) * 0.45
                                    } else if imbalance < 0.38 {
                                        0.5 + (bias_prob - 0.5) * 0.72
                                    } else {
                                        bias_prob
                                    };
                                    vars[v] = rng.gen_bool(prob.clamp(0.05, 0.95));
                                } else {
                                    let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
                                    if vad <= nad {
                                        vars[v] = rng.gen_bool(random_threshold);
                                    } else {
                                        vars[v] = rng.gen_bool(bias_prob);
                                    }
                                }
                            }

                            if rebuild_clause_state(&vars, cl.as_slice(), co.as_slice(), num_good.as_mut_slice(), &mut residual) {
                                let _ = save_solution(&Solution { variables: vars });
                                return Ok(());
                            }
                            current_prob = if exploratory {
                                (base_prob + 0.05).min(max_random_prob)
                            } else {
                                base_prob
                            };
                            last_check_residual = residual.len();
                            last_flip.fill(r32);
                            stagnation = 0;
                            rounds += 1;
                            continue;
                        } else {
                            if best_unsat > 10 && rounds > max_flips / 2 {
                                break;
                            }
                            let kicks = if stagnation >= 8 { 6 } else { 3 };
                            for _ in 0..kicks {
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
                                    *num_good.get_unchecked_mut(c) = num_good.get_unchecked(c).saturating_add(1);
                                }
                                for k in ds..de {
                                    let c = *da.get_unchecked(k as usize) as usize;
                                    let ng = num_good.get_unchecked_mut(c);
                                    let new_val = ng.saturating_sub(1);
                                    *ng = new_val;
                                    if new_val == 0 {
                                        residual.push(c as u32);
                                    }
                                }
                                *vars.get_unchecked_mut(v) = !was_true;
                                *last_flip.get_unchecked_mut(v) = r32;
                            }
                            stagnation = 0;
                        }
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
            let mut found = false;
            if !residual.is_empty() {
                let mut chosen_rid = 0usize;
                let mut best_clen = usize::MAX;

                if residual.len() <= 8 {
                    for rid in 0..residual.len() {
                        let candidate = *residual.get_unchecked(rid) as usize;
                        if *num_good.get_unchecked(candidate) == 0 {
                            let clen_c = *co.get_unchecked(candidate + 1) as usize
                                - *co.get_unchecked(candidate) as usize;
                            if clen_c < best_clen {
                                best_clen = clen_c;
                                cid = candidate;
                                chosen_rid = rid;
                                found = true;
                                if clen_c <= 1 {
                                    break;
                                }
                            }
                        }
                    }
                    if found {
                        residual.swap_remove(chosen_rid);
                    }
                } else {
                    let samples = if residual.len() >= 64 { 4 } else { 2 };
                    for t in 0..samples {
                        let rid = if t == 0 {
                            rand_val % residual.len()
                        } else {
                            rng.gen::<usize>() % residual.len()
                        };
                        let candidate = *residual.get_unchecked(rid) as usize;
                        if *num_good.get_unchecked(candidate) == 0 {
                            let clen_c = *co.get_unchecked(candidate + 1) as usize
                                - *co.get_unchecked(candidate) as usize;
                            if clen_c < best_clen {
                                best_clen = clen_c;
                                cid = candidate;
                                chosen_rid = rid;
                                found = true;
                                if clen_c <= 1 {
                                    break;
                                }
                            }
                        }
                    }
                    if found {
                        residual.swap_remove(chosen_rid);
                    }
                }

                if !found {
                    let len = residual.len();
                    if len > 0 {
                        let id = rand_val % len;
                        residual.swap(id, len - 1);
                    }
                    while let Some(candidate_u32) = residual.pop() {
                        cid = candidate_u32 as usize;
                        if *num_good.get_unchecked(cid) == 0 {
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found { break; }

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
                        let age_bonus = (r32.wrapping_sub(*last_flip.get_unchecked(abs_l)) as usize) / 4;
                        let adjusted_weight = appearances.saturating_sub(age_bonus);
                        if min_sad > 0 || adjusted_weight < min_weight {
                            min_sad = 0;
                            min_weight = adjusted_weight;
                            v_min = abs_l;
                        }
                    } else if min_sad > 0 {
                        let appearances = *var_appearances.get_unchecked(abs_l);
                        let age_bonus = (r32.wrapping_sub(*last_flip.get_unchecked(abs_l)) as usize) / 2;
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
                *num_good.get_unchecked_mut(c) = num_good.get_unchecked(c).saturating_add(1);
            }
            for k in ds..de {
                let c = *da.get_unchecked(k as usize) as usize;
                let ng = num_good.get_unchecked_mut(c);
                let new_val = ng.saturating_sub(1);
                *ng = new_val;
                if new_val == 0 {
                    residual.push(c as u32);
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            *last_flip.get_unchecked_mut(v_idx) = r32;

            rounds += 1;
        }
    }

    let final_vars = if best_unsat < residual.len() { best_vars } else { vars };
    let _ = save_solution(&Solution { variables: final_vars });
    Ok(())
}
