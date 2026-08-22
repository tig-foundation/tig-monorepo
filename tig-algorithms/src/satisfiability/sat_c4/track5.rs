use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
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

    drop(p_cnt);
    drop(n_cnt);


    // ---- IL1. One interleaved bounds array: bnd[2v]=all_off[v],
    // bnd[2v+1]=p_bound[v], bnd[2v+2]=all_off[v+1]. Both polarities of a
    // literal then resolve from 8 contiguous bytes = one cache line instead
    // of two random lines in two 400 KB arrays. Pure layout change; every
    // value read downstream is identical, so the trajectory is unchanged.
    let mut bnd: Vec<u32> = vec![0u32; 2 * nv + 1];
    for v in 0..nv {
        unsafe {
            *bnd.get_unchecked_mut(2 * v) = *all_off.get_unchecked(v);
            *bnd.get_unchecked_mut(2 * v + 1) = *p_bound.get_unchecked(v);
        }
    }
    bnd[2 * nv] = all_off[nv];

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

    // ---- FLAT4: stride-4 clause literals, see mk_f3.py.
    let mut cl4: Vec<i32> = vec![0i32; 4 * nc + 4];
    for i in 0..nc {
        let s4 = co[i] as usize;
        let e4 = co[i + 1] as usize;
        let m4 = (e4 - s4).min(4);
        for j in 0..m4 {
            cl4[4 * i + j] = cl[s4 + j];
        }
    }

    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
            unsat_list.push(i as u32);
        }
    }

    if unsat_list.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp.as_ref().and_then(|h| h.perturbation_flips)
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp.as_ref().and_then(|h| h.stagnation_limit)
        .unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);

    let mut last_check_residual = unsat_list.len();
    let mut stagnation = 0usize;
    let mut countdown = check_interval;
    let mut rounds = 0usize;

    let cb = if avg_clause_size > 4.5 {
        3.5f64
    } else if avg_clause_size > 3.5 {
        2.85f64
    } else {
        2.06f64
    };
    let mut probs_break = [0.0f64; 256];
    for i in 0..256 {
        probs_break[i] = cb.powf(-(i as f64));
    }

    unsafe {
        loop {
            if unsat_list.is_empty() || rounds >= max_flips { break; }

            countdown -= 1;
            if countdown == 0 {
                countdown = check_interval;
                let progress = last_check_residual as i64 - unsat_list.len() as i64;

                if progress <= 0 {
                    stagnation += 1;

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
                            if (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3 > 0 {
                                unsat_list.swap_remove(rid);
                                continue;
                            }

                            let pcs = pcid << 2;
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
                                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
                            }

                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let shift = (c & 3) << 1;
                                let byte_idx = c >> 2;
                                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                                if ng_before == 1 {
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

            let rand_val = rng.gen::<usize>();

            let mut cid = 0usize;
            let mut found = false;
            while !unsat_list.is_empty() {
                let id = rand_val % unsat_list.len();
                let candidate = *unsat_list.get_unchecked(id) as usize;
                if (*num_good.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3 > 0 {
                    unsat_list.swap_remove(id);
                } else {
                    cid = candidate;
                    found = true;
                    break;
                }
            }
            if !found { break; }

            let cs = cid << 2;
            let h1 = *cl4.get_unchecked(cs + 1);
            let h2 = *cl4.get_unchecked(cs + 2);
            let clen = if h2 != 0 { 3 } else if h1 != 0 { 2 }
                       else if *cl4.get_unchecked(cs) != 0 { 1 } else { 0 };
            let ce = cs + clen;


            let mut zero_buf: [usize; 3] = [0; 3];
            let mut zero_cnt: usize = 0;
            // ---- PF1. Pass 1: resolve every candidate's occurrence run and put
            // its head into flight. Pass 2: scan, in the ORIGINAL j order. The
            // <=3 dependent DRAM chains overlap instead of serialising, and the
            // observable behaviour is unchanged -- same reads, same order, same
            // zero_buf. The [usize; 3] bound is the same one zero_buf already
            // relies on: mod.rs dedups literals, so clen <= 3 always.
            let mut pf_lo: [usize; 3] = [0; 3];
            let mut pf_hi: [usize; 3] = [0; 3];
            let mut pf_ab: [usize; 3] = [0; 3];
            let mut pf_n: usize = 0;
            let mut pf_st: [usize; 3] = [0; 3];
            for j in cs..ce {
                let l = *cl4.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if l > 0 {
                    (*bnd.get_unchecked(2 * abs_l + 1) as usize, *bnd.get_unchecked(2 * abs_l + 2) as usize)
                } else {
                    (*bnd.get_unchecked(2 * abs_l) as usize, *bnd.get_unchecked(2 * abs_l + 1) as usize)
                };
                *pf_ab.get_unchecked_mut(pf_n) = abs_l;
                *pf_lo.get_unchecked_mut(pf_n) = os;
                *pf_hi.get_unchecked_mut(pf_n) = oe;
                pf_n += 1;
                if oe > os {
                    _mm_prefetch::<_MM_HINT_T0>(all_data.as_ptr().add(os) as *const i8);
                    if oe - os > 16 {
                        _mm_prefetch::<_MM_HINT_T0>(all_data.as_ptr().add(oe - 1) as *const i8);
                    }
                }
            }
            'outer: for t in 0..pf_n {
                let abs_l = *pf_ab.get_unchecked(t);
                let os = *pf_lo.get_unchecked(t);
                let oe = *pf_hi.get_unchecked(t);
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                        *pf_st.get_unchecked_mut(t) = k;
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
            } else {
                let mut sum_scores = 0.0;
                let mut scores = [0.0; 3];
                let limit = (ce - cs).min(256);
                for idx in 0..limit {
                    let oe = *pf_hi.get_unchecked(idx);
                    let mut sad = 1usize;
                    for k in (*pf_st.get_unchecked(idx) + 1)..oe {
                        let c = *all_data.get_unchecked(k) as usize;
                        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                            sad += 1;
                        }
                    }
                    let score = *probs_break.get_unchecked(sad.min(255));
                    sum_scores += score;
                    *scores.get_unchecked_mut(idx) = score;
                }
                
                let threshold = rng.gen::<f64>() * sum_scores;
                let mut accum = 0.0;
                let mut v_sel = (cl4.get_unchecked(cs).abs() - 1) as usize;
                for idx in 0..limit {
                    accum += *scores.get_unchecked(idx);
                    if accum >= threshold {
                        v_sel = (cl4.get_unchecked(cs + idx).abs() - 1) as usize;
                        break;
                    }
                }
                v_sel
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
                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
            }

            unsat_list.reserve(de - ds);
            {
                let ulp = unsat_list.as_mut_ptr();
                let mut l = unsat_list.len();
                for k in ds..de {
                    let c = *all_data.get_unchecked(k) as usize;
                    let shift = (c & 3) << 1;
                    let byte_idx = c >> 2;
                    let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                    *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                    let hit = (ng_before == 1) as usize;
                    *ulp.add(l) = c as u32;
                    l += hit;
                }
                unsat_list.set_len(l);
            }
            *vars.get_unchecked_mut(v_idx) = !was_true;
            rounds += 1;
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}