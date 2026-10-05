use anyhow::Result;
use rand::Rng;
use tig_challenges::satisfiability::*;

use super::Hparams;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let super::common::Prepared {
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
    } = super::common::preprocess_enc(challenge, save_solution);

    let nvf = nv as f64;
    let max_fuel = hp.max_fuel_low.unwrap_or(150_000_000_000.0);
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

    let sp_cfg = SpCfg {
        f: hp.sp_f.unwrap_or(SP_F_DEF),
        eps: hp.sp_eps.unwrap_or(SP_EPS_DEF) as f32,
        triv: hp.sp_triv.unwrap_or(SP_TRIV_DEF) as f32,
        edge_budget: hp.sp_edge_budget.unwrap_or(SP_EDGE_BUDGET_DEF) as u64,
    };
    if hp.sp_on.unwrap_or(SP_ON_DEF) {
        let sp_seed = u64::from_le_bytes(challenge.seed[8..16].try_into().unwrap());
        if let Some(fixed) = sp_decimate(&cl, &co, nv, nc, sp_seed, &sp_cfg) {
            for v in 0..nv {
                let f = fixed[v];
                if f >= 0 { vars[v] = f == 1; }
            }
        }
    }

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
            let v = (l >> 1) as usize;
            if ((l & 1) == 1) == vars[v] {
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

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.check_interval
        .unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp.perturbation_flips
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp.stagnation_limit
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

                            let pcs = *co.get_unchecked(pcid) as usize;
                            let pce = *co.get_unchecked(pcid + 1) as usize;
                            if pcs == pce { continue; }
                            let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                            let v = (lit >> 1) as usize;

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
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }

                last_check_residual = unsat_list.len();
            }

            let r0 = rounds;
            let nblk = (max_flips - rounds).min(countdown);
            for _ in 0..nblk {

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
            if clen == 3 {
                macro_rules! probe3 { ($j:expr) => {{
                    let l = *cl.get_unchecked($j);
                    let abs_l = (l >> 1) as usize;
                    let (os, oe) = if (l & 1) == 1 {
                        (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                    } else {
                        (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                    };
                    let mut crit = false;
                    for k in os..oe {
                        let c = *all_data.get_unchecked(k) as usize;
                        if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 {
                            crit = true;
                            break;
                        }
                    }
                    if !crit {
                        *zero_buf.get_unchecked_mut(zero_cnt) = abs_l;
                        zero_cnt += 1;
                    }
                }}}
                probe3!(cs);
                probe3!(cs + 1);
                probe3!(cs + 2);
            } else {
            'outer: for j in cs..ce {
                let l = *cl.get_unchecked(j);
                let abs_l = (l >> 1) as usize;
                let (os, oe) = if (l & 1) == 1 {
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
            }

            let v_idx = if zero_cnt > 0 {
                if zero_cnt == 1 {
                    *zero_buf.get_unchecked(0)
                } else {
                    *zero_buf.get_unchecked(rand_val % zero_cnt)
                }
            } else {
                let mut sum_scores = 0.0;
                let mut scores = [0.0; 256];
                let limit = (ce - cs).min(256);
                for idx in 0..limit {
                    let j = cs + idx;
                    let l = *cl.get_unchecked(j);
                    let abs_l = (l >> 1) as usize;
                    let (os, oe) = if (l & 1) == 1 {
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
                    }
                    let score = *probs_break.get_unchecked(sad.min(255));
                    sum_scores += score;
                    *scores.get_unchecked_mut(idx) = score;
                }

                let threshold = rng.gen::<f64>() * sum_scores;
                let mut accum = 0.0;
                let mut v_sel = (*cl.get_unchecked(cs) >> 1) as usize;
                for idx in 0..limit {
                    accum += *scores.get_unchecked(idx);
                    if accum >= threshold {
                        v_sel = (*cl.get_unchecked(cs + idx) >> 1) as usize;
                        break;
                    }
                }
                v_sel
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
            rounds += 1;
            }
            countdown -= (rounds - r0).saturating_sub(1);
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

const SP_ON_DEF: bool = true;
const SP_F_DEF: f64 = 0.01;
const SP_EPS_DEF: f64 = 1e-3;
const SP_TRIV_DEF: f64 = 1e-2;
const SP_EDGE_BUDGET_DEF: f64 = 3e9;
const SP_TMAX: usize = 1000;
const SP_CLAMP: f32 = 1.0 - 1e-6;

struct SpCfg {
    f: f64,
    eps: f32,
    triv: f32,
    edge_budget: u64,
}

#[derive(Clone, Copy)]
struct SpEdge {
    vp: u32,
    eta: f32,
}

struct SpRes {
    m: usize,
    nc: usize,
    loc: Vec<u32>,
    co: Vec<u32>,
    ed: Vec<SpEdge>,
    voff: Vec<u32>,
    vedge: Vec<u32>,
}

#[inline]
fn sp_clause_of(co: &[u32], e: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = co.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) >> 1;
        if (co[mid] as usize) <= e { lo = mid; } else { hi = mid; }
    }
    lo
}

fn sp_index(r: &mut SpRes) {
    let ne = r.ed.len();
    let mut pc = vec![0u32; r.m];
    let mut ncnt = vec![0u32; r.m];
    for e in 0..ne {
        let v = (r.ed[e].vp >> 1) as usize;
        if r.ed[e].vp & 1 == 1 { pc[v] += 1; } else { ncnt[v] += 1; }
    }
    r.voff = vec![0u32; r.m + 1];
    for v in 0..r.m { r.voff[v + 1] = r.voff[v] + pc[v] + ncnt[v]; }
    let mut pp = vec![0u32; r.m];
    let mut np = vec![0u32; r.m];
    for v in 0..r.m { pp[v] = r.voff[v]; np[v] = r.voff[v] + pc[v]; }
    r.vedge = vec![0u32; ne];
    for e in 0..ne {
        let vp = r.ed[e].vp;
        let v = (vp >> 1) as usize;
        let k = if vp & 1 == 1 { let k = pp[v]; pp[v] += 1; k } else { let k = np[v]; np[v] += 1; k };
        r.vedge[k as usize] = e as u32;
    }
}

#[inline]
fn sp_scatter(r: &SpRes, pv: &mut [[f64; 2]]) {
    for v in 0..r.m { pv[v] = [1.0, 1.0]; }
    for e in 0..r.ed.len() {
        let ed = unsafe { *r.ed.get_unchecked(e) };
        let v = (ed.vp >> 1) as usize;
        let i = ((ed.vp & 1) ^ 1) as usize;
        unsafe { pv.get_unchecked_mut(v)[i] *= (1.0 - ed.eta) as f64; }
    }
}

fn sp_decimate(
    cl: &[i32],
    co: &[u32],
    nv: usize,
    nc: usize,
    seed: u64,
    cfg: &SpCfg,
) -> Option<Vec<i8>> {
    let ne = cl.len();
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0)
        | 1;
    let mut ed: Vec<SpEdge> = Vec::with_capacity(ne);
    for e in 0..ne {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let x = ((s >> 40) as f32) * (1.0 / 16_777_216.0);
        ed.push(SpEdge { vp: cl[e] as u32, eta: if x > SP_CLAMP { SP_CLAMP } else { x } });
    }
    let mut r = SpRes {
        m: nv,
        nc,
        loc: (0..nv as u32).collect(),
        co: co.to_vec(),
        ed,
        voff: Vec::new(), vedge: Vec::new(),
    };
    sp_index(&mut r);

    let mut fixed = vec![-1i8; nv];
    let per_step = ((nv as f64 * cfg.f).ceil() as usize).max(1);
    let mut budget = cfg.edge_budget;

    loop {
        let ne_r = r.ed.len();
        let mut pv = vec![[1.0f64, 1.0f64]; r.m];
        let mut converged = false;
        let mut maxeta = 0.0f32;
        let mut t = 0usize;
        while t < SP_TMAX {
            if budget < ne_r as u64 { return None; }
            budget -= ne_r as u64;
            t += 1;
            sp_scatter(&r, &mut pv);
            let mut delta = 0.0f32;
            maxeta = 0.0;
            for c in 0..r.nc {
                let cs = unsafe { *r.co.get_unchecked(c) } as usize;
                let ce = unsafe { *r.co.get_unchecked(c + 1) } as usize;
                let k = ce - cs;
                if k == 0 || k > 4 { continue; }
                let mut rb = [0.0f64; 4];
                for i in 0..k {
                    let e = unsafe { *r.ed.get_unchecked(cs + i) };
                    let v = (e.vp >> 1) as usize;
                    let f = 1.0 - e.eta as f64;
                    let p = unsafe { *pv.get_unchecked(v) };
                    let (pa, pd) = if e.vp & 1 == 1 { (p[0] / f, p[1]) } else { (p[1] / f, p[0]) };
                    let sum = pa + pd - pa * pd;
                    rb[i] = if sum > 1e-300 { ((1.0 - pd) * pa) / sum } else { 0.0 };
                }
                for i in 0..k {
                    let mut p = 1.0f64;
                    for j in 0..k { if j != i { p *= rb[j]; } }
                    let mut pf = p as f32;
                    if pf > SP_CLAMP { pf = SP_CLAMP; }
                    let e = cs + i;
                    unsafe {
                        let slot = r.ed.get_unchecked_mut(e);
                        let d = (pf - slot.eta).abs();
                        if d > delta { delta = d; }
                        slot.eta = pf;
                    }
                    if pf > maxeta { maxeta = pf; }
                }
            }
            if delta < cfg.eps { converged = true; break; }
        }
        if !converged { return None; }
        if maxeta < cfg.triv { return Some(fixed); }

        sp_scatter(&r, &mut pv);
        let mut bias = vec![0.0f64; r.m];
        let mut want = vec![false; r.m];
        for v in 0..r.m {
            let a = pv[v][0];
            let b = pv[v][1];
            let pip = (1.0 - a) * b;
            let pim = (1.0 - b) * a;
            let sum = pip + pim + a * b;
            if sum > 1e-300 {
                bias[v] = ((pip - pim) / sum).abs();
                want[v] = pip > pim;
            }
        }
        let mut order: Vec<u32> = (0..r.m as u32).collect();
        if order.is_empty() { return Some(fixed); }
        let kk = per_step.min(order.len());
        order.sort_unstable_by(|&x, &y| {
            bias[y as usize]
                .partial_cmp(&bias[x as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut lfix = vec![-1i8; r.m];
        let mut cact = vec![true; r.nc];
        let mut eact = vec![true; ne_r];
        let mut clen: Vec<u8> = (0..r.nc).map(|c| (r.co[c + 1] - r.co[c]) as u8).collect();
        let mut queue: Vec<(u32, bool)> =
            (0..kk).map(|i| (order[i], want[order[i] as usize])).collect();
        let mut qi = 0usize;
        let mut n_act = r.nc;
        while qi < queue.len() {
            let (vq, val) = queue[qi];
            qi += 1;
            let v = vq as usize;
            if lfix[v] >= 0 {
                if (lfix[v] == 1) != val { return None; }
                continue;
            }
            lfix[v] = val as i8;
            fixed[r.loc[v] as usize] = val as i8;
            for kx in r.voff[v] as usize..r.voff[v + 1] as usize {
                let e = r.vedge[kx] as usize;
                if !eact[e] { continue; }
                let c = sp_clause_of(&r.co, e);
                eact[e] = false;
                if !cact[c] { continue; }
                if (r.ed[e].vp & 1 == 1) == val {
                    cact[c] = false;
                    n_act -= 1;
                    for e2 in r.co[c] as usize..r.co[c + 1] as usize { eact[e2] = false; }
                } else {
                    clen[c] -= 1;
                    if clen[c] == 0 { return None; }
                    if clen[c] == 1 {
                        let mut found = usize::MAX;
                        for e2 in r.co[c] as usize..r.co[c + 1] as usize {
                            if eact[e2] { found = e2; break; }
                        }
                        if found == usize::MAX { return None; }
                        queue.push((r.ed[found].vp >> 1, r.ed[found].vp & 1 == 1));
                    }
                }
            }
        }
        if n_act == 0 { return Some(fixed); }

        let mut vmap = vec![u32::MAX; r.m];
        let mut loc: Vec<u32> = Vec::new();
        let mut nco: Vec<u32> = vec![0];
        let mut ned: Vec<SpEdge> = Vec::new();
        for c in 0..r.nc {
            if !cact[c] { continue; }
            for e in r.co[c] as usize..r.co[c + 1] as usize {
                if !eact[e] { continue; }
                let vp = r.ed[e].vp;
                let ov = (vp >> 1) as usize;
                let id = if vmap[ov] == u32::MAX {
                    let id = loc.len() as u32;
                    vmap[ov] = id;
                    loc.push(r.loc[ov]);
                    id
                } else { vmap[ov] };
                ned.push(SpEdge { vp: (id << 1) | (vp & 1), eta: r.ed[e].eta });
            }
            nco.push(ned.len() as u32);
        }
        if nco.len() <= 1 { return Some(fixed); }
        r = SpRes {
            m: loc.len(), nc: nco.len() - 1, loc, co: nco, ed: ned,
            voff: Vec::new(), vedge: Vec::new(),
        };
        sp_index(&mut r);
    }
}
