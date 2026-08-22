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
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(160_000_000_000.0);

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
    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    let mut unsat_pos = vec![u32::MAX; nc + 1];

    // ---- FLAT4. Stride-4 clause literals: clause i owns cl4[4i..4i+3],
    // zero-padded. One aligned block per clause; `co` leaves the hot loop.
    let mut cl4: Vec<i32> = vec![0i32; 4 * nc + 4];
    for i in 0..nc {
        let s4 = co[i] as usize;
        let e4 = co[i + 1] as usize;
        let m4 = (e4 - s4).min(4);
        for j in 0..m4 {
            cl4[4 * i + j] = cl[s4 + j];
        }
    }

    // ---- IL1 (ported from track4). One interleaved bounds array:
    // bnd[2v]=all_off[v], bnd[2v+1]=p_bound[v], bnd[2v+2]=all_off[v+1].
    // Both polarities of a variable then come from adjacent slots -- one cache
    // line instead of two arrays 20 KB apart. Same values, same reads.
    let mut bnd: Vec<u32> = vec![0u32; 2 * nv + 1];
    for v in 0..nv {
        unsafe {
            *bnd.get_unchecked_mut(2 * v) = *all_off.get_unchecked(v);
            *bnd.get_unchecked_mut(2 * v + 1) = *p_bound.get_unchecked(v);
        }
    }
    bnd[2 * nv] = all_off[nv];

    for c in 0..nc {
        let s = co[c] as usize;
        let e = co[c + 1] as usize;
        let mut g = 0u8;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
        }
        num_good[c] = g;
        if g == 0 {
            unsat_pos[c] = unsat_list.len() as u32;
            unsat_list.push(c as u32);
        }
    }

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let mut last_check_residual = unsat_list.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit_t4 = hp.as_ref().and_then(|h| h.stagnation_limit).unwrap_or(3);

    let probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    let mut current_reinit_stagnation: usize = ((nv * nc / 1000) as usize).clamp(1000, 500000);
    const REINIT_MIN_UNSAT: usize = 10;
    const MAX_REINITS: usize = 50;

    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if unsat_list.is_empty() { break; }

            if reinit_count >= MAX_REINITS && stagnation_count >= current_reinit_stagnation && best_unsat >= REINIT_MIN_UNSAT {
                break;
            }

            if stagnation_count >= current_reinit_stagnation && best_unsat >= REINIT_MIN_UNSAT && reinit_count < MAX_REINITS {
                reinit_count += 1;
                let reinit_factor = if density > 4.0 { 1.5 } else { 1.3 };
                current_reinit_stagnation = ((current_reinit_stagnation as f64 * reinit_factor) as usize).clamp(1000, 500000);

                for v in 0..nv { vars[v] = false; }
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

                unsat_list.clear();
                unsat_pos.fill(u32::MAX);
                for c in 0..nc {
                    let s = co[c] as usize;
                    let e = co[c + 1] as usize;
                    let mut g = 0u8;
                    for j in s..e {
                        let l = cl[j];
                        let v = (l.abs() - 1) as usize;
                        if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
                    }
                    num_good[c] = g;
                    if g == 0 {
                        unsat_pos[c] = unsat_list.len() as u32;
                        unsat_list.push(c as u32);
                    }
                }

                best_unsat = unsat_list.len();
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - unsat_list.len() as i64;

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= stagnation_limit_t4 {
                        let kicks = if stagnation >= 8 { 6 } else { 3 };
                        for _ in 0..kicks {
                            if unsat_list.is_empty() { break; }

                            let rid = rng.gen::<usize>() % unsat_list.len();
                            let pcid = *unsat_list.get_unchecked(rid) as usize;
                            let pcs = 4 * pcid;
                            let pl1 = *cl4.get_unchecked(pcs + 1);
                            let pl2 = *cl4.get_unchecked(pcs + 2);
                            let pclen = if pl2 != 0 { 3 } else if pl1 != 0 { 2 }
                                        else if *cl4.get_unchecked(pcs) != 0 { 1 } else { 0 };
                            if pclen == 0 { continue; }
                            let lit = *cl4.get_unchecked(pcs + rng.gen::<usize>() % pclen);
                            let v = (lit.abs() - 1) as usize;

                            let was_true = *vars.get_unchecked(v);
                            let (is, ie) = if was_true {
                                (*bnd.get_unchecked(2 * v + 1) as usize, *bnd.get_unchecked(2 * v + 2) as usize)
                            } else {
                                (*bnd.get_unchecked(2 * v) as usize, *bnd.get_unchecked(2 * v + 1) as usize)
                            };
                            let (ds, de) = if was_true {
                                (*bnd.get_unchecked(2 * v) as usize, *bnd.get_unchecked(2 * v + 1) as usize)
                            } else {
                                (*bnd.get_unchecked(2 * v + 1) as usize, *bnd.get_unchecked(2 * v + 2) as usize)
                            };

                            for k in is..ie {
                                let c = *all_data.get_unchecked(k) as usize;
                                let ng = *num_good.get_unchecked(c);
                                if ng == 0 {
                                    let pos = *unsat_pos.get_unchecked(c) as usize;
                                    let last_idx = unsat_list.len() - 1;
                                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                                    unsat_list.pop();
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
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }

                last_check_residual = unsat_list.len();
            }

            if unsat_list.is_empty() { break; }

            let rand_val = rng.gen::<usize>();

            let cid = {
                let uc = unsat_list.len();
                let i1 = (rand_val as u32 as usize) % uc;
                *unsat_list.get_unchecked(i1) as usize
            };

            let cs = 4 * cid;
            let h1 = *cl4.get_unchecked(cs + 1);
            let h2 = *cl4.get_unchecked(cs + 2);
            let clen = if h2 != 0 { 3 } else if h1 != 0 { 2 }
                       else if *cl4.get_unchecked(cs) != 0 { 1 } else { 0 };
            let ce = cs + clen;

            if clen > 1 {
                let ri = rand_val % clen;
                cl4.swap(cs, cs + ri);
            }

            let mut zero_buf: [usize; 3] = [0; 3];
            let mut zero_cnt: usize = 0;
            let mut pw_weights: [u32; 3] = [0; 3];
            let mut pw_vars: [usize; 3] = [0; 3];
            let mut pw_cnt: usize = 0;
            let mut total_pw: u32 = 0;

            for j in cs..ce {
                let l = *cl4.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if *vars.get_unchecked(abs_l) {
                    (*bnd.get_unchecked(2 * abs_l) as usize, *bnd.get_unchecked(2 * abs_l + 1) as usize)
                } else {
                    (*bnd.get_unchecked(2 * abs_l + 1) as usize, *bnd.get_unchecked(2 * abs_l + 2) as usize)
                };

                let mut sad = 0usize;
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        sad += 1;
                    }
                }

                if sad == 0 {
                    *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                    zero_cnt += 1;
                }

                let pw = *probs_break.get_unchecked(sad.min(15));
                *pw_weights.get_unchecked_mut(pw_cnt) = pw;
                *pw_vars.get_unchecked_mut(pw_cnt) = abs_l;
                total_pw += pw;
                pw_cnt += 1;
            }

            let v_idx = if zero_cnt > 0 {
                if zero_cnt == 1 {
                    *zero_buf.get_unchecked(0)
                } else {
                    *zero_buf.get_unchecked(rand_val % zero_cnt)
                }
            } else {
                let mut r = (rand_val as u32) % total_pw.max(1);
                let mut chosen = *pw_vars.get_unchecked(0);
                for i in 0..pw_cnt {
                    let pw = *pw_weights.get_unchecked(i);
                    if r < pw {
                        chosen = *pw_vars.get_unchecked(i);
                        break;
                    }
                    r -= pw;
                }
                chosen
            };

            let was_true = *vars.get_unchecked(v_idx);
            let (is, ie) = if was_true {
                (*bnd.get_unchecked(2 * v_idx + 1) as usize, *bnd.get_unchecked(2 * v_idx + 2) as usize)
            } else {
                (*bnd.get_unchecked(2 * v_idx) as usize, *bnd.get_unchecked(2 * v_idx + 1) as usize)
            };
            let (ds, de) = if was_true {
                (*bnd.get_unchecked(2 * v_idx) as usize, *bnd.get_unchecked(2 * v_idx + 1) as usize)
            } else {
                (*bnd.get_unchecked(2 * v_idx + 1) as usize, *bnd.get_unchecked(2 * v_idx + 2) as usize)
            };

            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                let ng = *num_good.get_unchecked(c);
                if ng == 0 {
                    let pos = *unsat_pos.get_unchecked(c) as usize;
                    let last_idx = unsat_list.len() - 1;
                    let last_c = *unsat_list.get_unchecked(last_idx) as usize;
                    *unsat_list.get_unchecked_mut(pos) = last_c as u32;
                    *unsat_pos.get_unchecked_mut(last_c) = pos as u32;
                    *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                    unsat_list.pop();
                }
                *num_good.get_unchecked_mut(c) = ng + 1;
            }

            {
                let ulp = unsat_list.as_mut_ptr();
                let upp = unsat_pos.as_mut_ptr();
                let mut l = unsat_list.len();
                for k in ds..de {
                    let c = *all_data.get_unchecked(k) as usize;
                    let ng = *num_good.get_unchecked(c);
                    *num_good.get_unchecked_mut(c) = ng - 1;
                    let hit = (ng == 1) as usize;
                    *ulp.add(l) = c as u32;
                    let ci = if hit == 1 { c } else { nc };
                    *upp.add(ci) = l as u32;
                    l += hit;
                }
                unsat_list.set_len(l);
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            rounds += 1;

            let cur = unsat_list.len();
            if cur < best_unsat {
                best_unsat = cur;
                best_vars.copy_from_slice(&vars);
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }
        }
    }

    let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
    let _ = save_solution(&Solution { variables: final_vars });

    Ok(())
}