use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;

use super::Hparams;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let super::Prepared {
        mut rng,
        nv,
        nc,
        density,
        p_cnt,
        n_cnt,
        all_off,
        p_bound,
        all_data,
        mut cl,
        co,
    } = super::preprocess(challenge, save_solution);

    let max_fuel = hp.max_fuel_high.unwrap_or(160_000_000_000.0);

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
    let mut unsat_pos = vec![u32::MAX; nc];

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

    let w = vec![1u8; nc];

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.check_interval
        .unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let mut last_check_residual = unsat_list.len();
    let mut rounds = 0usize;
    let mut stagnation = 0usize;
    let stagnation_limit_t4 = hp.stagnation_limit.unwrap_or(3);

    let probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    const REINIT_STAGNATION: usize = 2_000_000;
    const REINIT_MIN_UNSAT: usize = 10;
    let max_reinits = hp.max_reinits.unwrap_or(5);

    let mut best_unsat = unsat_list.len();
    let mut best_vars = vars.clone();
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;

    const N_BON_RESTARTS: usize = 5;
    let mut bon_candidate = vec![false; nv];
    let mut bon_num_good = vec![0u8; nc];

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if unsat_list.is_empty() { break; }

            if stagnation_count >= REINIT_STAGNATION && best_unsat >= REINIT_MIN_UNSAT && reinit_count < max_reinits {
                reinit_count += 1;

                let mut best_cand_unsat = usize::MAX;
                for _ in 0..N_BON_RESTARTS {
                    for v in 0..nv { bon_candidate[v] = false; }
                    for v in 0..nv {
                        let np = p_cnt[v] as usize;
                        let nn = n_cnt[v] as usize;
                        if nn == 0 && np > 0 { bon_candidate[v] = true; continue; }
                        if np == 0 && nn > 0 { continue; }
                        let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
                        if vad <= nad {
                            bon_candidate[v] = rng.gen_bool(random_threshold);
                        } else {
                            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                            bon_candidate[v] = rng.gen_bool(prob);
                        }
                    }
                    bon_num_good.fill(0);
                    for c in 0..nc {
                        let s = co[c] as usize;
                        let e = co[c + 1] as usize;
                        let mut g = 0u8;
                        for j in s..e {
                            let l = cl[j];
                            let v = (l.abs() - 1) as usize;
                            if (l > 0 && bon_candidate[v]) || (l < 0 && !bon_candidate[v]) { g += 1; }
                        }
                        bon_num_good[c] = g;
                    }
                    let cand_unsat = bon_num_good.iter().filter(|&&x| x == 0).count();
                    if cand_unsat < best_cand_unsat {
                        best_cand_unsat = cand_unsat;
                        vars.copy_from_slice(&bon_candidate);
                    }
                }

                num_good.fill(0);
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
                }

                unsat_list.clear();
                unsat_pos.fill(u32::MAX);
                for c in 0..nc {
                    if num_good[c] == 0 {
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
                let i2 = (rand_val >> 32) % uc;
                let c1 = *unsat_list.get_unchecked(i1) as usize;
                let c2 = *unsat_list.get_unchecked(i2) as usize;
                if *w.get_unchecked(c1) >= *w.get_unchecked(c2) { c1 } else { c2 }
            };

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rand_val % clen;
                cl.swap(cs, cs + ri);
            }

            let mut zero_buf: [usize; 3] = [0; 3];
            let mut zero_cnt: usize = 0;
            'outer_h: for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if *vars.get_unchecked(abs_l) {
                    (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                } else {
                    (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                };
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        continue 'outer_h;
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
            } else {

                let mut pw_weights: [u32; 3] = [0; 3];
                let mut pw_vars: [usize; 3] = [0; 3];
                let mut pw_cnt: usize = 0;
                let mut total_pw: u32 = 0;

                for j in cs..ce {
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

                    let b_idx = sad.min(15);
                    let pw = *probs_break.get_unchecked(b_idx);
                    *pw_weights.get_unchecked_mut(pw_cnt) = pw;
                    *pw_vars.get_unchecked_mut(pw_cnt) = abs_l;
                    total_pw += pw;
                    pw_cnt += 1;
                }

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
