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
    } = super::common::preprocess(challenge, save_solution);

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

    let spr_on = hp.spr_on.unwrap_or(SPR_ON_DEF);
    let spr_flip_frac = hp.spr_flip_frac.unwrap_or(SPR_FLIP_FRAC_DEF);
    let vars_prod: Vec<bool> = if spr_on { vars.clone() } else { Vec::new() };
    if spr_on {
        let spr_cfg = SprCfg {
            maxit:       hp.spr_maxit.unwrap_or(SPR_MAXIT_DEF) as usize,
            rate:        hp.spr_rate.unwrap_or(SPR_RATE_DEF) as f32,
            gmax:        hp.spr_gmax.unwrap_or(SPR_GMAX_DEF) as f32,
            lam:         hp.spr_lam.unwrap_or(SPR_LAM_DEF) as f32,
            damp:        hp.spr_damp.unwrap_or(SPR_DAMP_DEF) as f32,
            check_every: (hp.spr_check.unwrap_or(SPR_CHECK_DEF) as usize).max(1),
            patience:    hp.spr_patience.unwrap_or(SPR_PATIENCE_DEF) as usize,
            min_it:      hp.spr_minit.unwrap_or(SPR_MINIT_DEF) as usize,
            target_u:    hp.spr_target.unwrap_or(SPR_TARGET_DEF) as usize,
            edge_budget: hp.spr_budget.unwrap_or(SPR_EDGE_BUDGET_DEF) as u64,
        };
        let cl_enc: Vec<i32> = cl.iter().map(|&l| super::common::enc(l)).collect();
        let s = super::common::splitmix64(
            super::common::content_seed(challenge) ^ 0x5350_5F52_4546_4E00);
        let (sx, _su, _sit) = spr_seed(&cl_enc, &co, nv, nc, s, &spr_cfg, SPR_FAST3_DEF);
        vars.copy_from_slice(&sx);
    }

    let appearances: Vec<u8> = (0..nv).map(|v| {
        ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
    }).collect();
    drop(p_cnt);
    drop(n_cnt);

    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut var_age = vec![0u8; nv];
    let mut rounds = 0usize;

    for phase in (if spr_on { 0usize } else { 1usize })..2usize {
    if phase == 1 {
        if spr_on { vars.copy_from_slice(&vars_prod); }
        for b in num_good.iter_mut() { *b = 0; }
        for a in var_age.iter_mut() { *a = 0; }
    }

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

    residual.clear();
    for i in 0..nc {
        if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
            residual.push(i as u32);
        }
    }

    if residual.is_empty() { break; }

    let base_prob = hp.base_prob.unwrap_or(
        if phase == 0 && spr_on { WALK_PROB_SEEDED_DEF }
        else { 0.45 + 0.1 * (density / 5.0).min(1.0) });
    let mut current_prob = base_prob;

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.check_interval
        .unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
    let max_random_prob = hp.max_prob.unwrap_or(0.9);
    let prob_adjustment_factor = 0.03;
    let smoothing_factor = 0.8;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp.perturbation_flips
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp.stagnation_limit
        .unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);

    let flip_cap = if phase == 0 {
        rounds.saturating_add(((max_flips as f64) * spr_flip_frac) as usize).min(max_flips)
    } else { max_flips };
    let mut last_check_residual = residual.len();
    let mut stagnation = 0usize;
    let mut countdown = check_interval;

    let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    unsafe {
        't3: loop {
            if residual.is_empty() || rounds >= flip_cap { break; }

            countdown -= 1;
            if countdown == 0 {
                countdown = check_interval;
                let progress = last_check_residual as i64 - residual.len() as i64;
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
                            if residual.is_empty() { break; }
                            let rid = rng.gen::<usize>() % residual.len();
                            let pcid = *residual.get_unchecked(rid) as usize;
                            let ng_val = (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3;
                            if ng_val > 0 {
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
                                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
                            }
                            for k in ds..de {
                                let c = *all_data.get_unchecked(k) as usize;
                                let shift = (c & 3) << 1;
                                let byte_idx = c >> 2;
                                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                                if ng_before == 1 {
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

            let r0 = rounds;
            let nblk = (flip_cap - rounds).min(countdown);
            for _ in 0..nblk {

            let rand_val = rng.gen::<usize>();
            let mut cid = 0usize;
            let mut found = false;
            while !residual.is_empty() {
                let id = rand_val % residual.len();
                let candidate = *residual.get_unchecked(id) as usize;
                let ng_val = (*num_good.get_unchecked(candidate >> 2) >> ((candidate & 3) << 1)) & 3;
                if ng_val > 0 {
                    residual.swap_remove(id);
                } else {
                    cid = candidate;
                    found = true;
                    break;
                }
            }
            if !found { break 't3; }

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
                    let abs_l = (l.abs() - 1) as usize;
                    let (os, oe) = if l > 0 {
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
                *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1);
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                let shift = (c & 3) << 1;
                let byte_idx = c >> 2;
                let ng_before = (*num_good.get_unchecked(byte_idx) >> shift) & 3;
                *num_good.get_unchecked_mut(byte_idx) -= 1u8 << shift;
                if ng_before == 1 {
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
            if residual.is_empty() { break; }
            }
            countdown -= (rounds - r0).saturating_sub(1);
        }
    }

    if residual.is_empty() { break; }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

const SPR_ON_DEF: bool = true;
const SPR_MAXIT_DEF: f64 = 1500.0;
const SPR_RATE_DEF: f64 = 6e-3;
const SPR_GMAX_DEF: f64 = 0.5;
const SPR_LAM_DEF: f64 = 0.1;
const SPR_DAMP_DEF: f64 = 0.3;
const SPR_CHECK_DEF: f64 = 5.0;
const SPR_PATIENCE_DEF: f64 = 150.0;
const SPR_MINIT_DEF: f64 = 0.0;
const SPR_TARGET_DEF: f64 = 1000.0;
const SPR_EDGE_BUDGET_DEF: f64 = 3e9;
const SPR_FLIP_FRAC_DEF: f64 = 0.15;
const SPR_FLOOR: f32 = 1e-6;

#[derive(Clone, Copy)]
struct SprCfg {
    maxit: usize,
    rate: f32,
    gmax: f32,
    lam: f32,
    damp: f32,
    check_every: usize,
    patience: usize,
    min_it: usize,
    target_u: usize,
    edge_budget: u64,
}

fn spr_seed(
    cl: &[i32],
    co: &[u32],
    nv: usize,
    nc: usize,
    seed: u64,
    cfg: &SprCfg,
    fast3: bool,
) -> (Vec<bool>, usize, usize) {
    let ne = cl.len();
    let mut s = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0)
        | 1;
    let mut eta: Vec<f32> = Vec::with_capacity(ne);
    for _ in 0..ne {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        eta.push(((s >> 40) as f32) * (1.0 / 16_777_216.0));
    }
    let mut p = vec![1f32; 2 * nv];
    let mut hf = vec![0f32; 2 * nv];
    let mut x = vec![false; nv];
    let mut best_x = vec![false; nv];
    let mut best_u = usize::MAX;
    let mut best_it = 0usize;
    let mut it = 0usize;
    let mut budget = cfg.edge_budget;

    while it < cfg.maxit {
        if budget < ne as u64 {
            break;
        }
        budget -= ne as u64;
        it += 1;
        let gamma = (cfg.rate * it as f32).min(cfg.gmax);

        for v in 0..nv {
            unsafe {
                *p.get_unchecked_mut(2 * v) = 1.0 - *hf.get_unchecked(2 * v);
                *p.get_unchecked_mut(2 * v + 1) = 1.0 - *hf.get_unchecked(2 * v + 1);
            }
        }
        for e in 0..ne {
            unsafe {
                let o = (1.0 - *eta.get_unchecked(e)).max(SPR_FLOOR);
                *p.get_unchecked_mut(*cl.get_unchecked(e) as usize) *= o;
            }
        }

        let mut rr = [0f32; 8];
        for a in 0..nc {
            let cs = unsafe { *co.get_unchecked(a) } as usize;
            let ce = unsafe { *co.get_unchecked(a + 1) } as usize;
            let len = ce - cs;
            if len == 0 || len > 8 {
                continue;
            }
            if fast3 && len == 3 {
                unsafe {
                    let mut r3 = [0f32; 3];
                    for q in 0..3 {
                        let e = cs + q;
                        let k = *cl.get_unchecked(e) as usize;
                        let o = (1.0 - *eta.get_unchecked(e)).max(SPR_FLOOR);
                        let av = *p.get_unchecked(k) / o;
                        let bv = *p.get_unchecked(k ^ 1);
                        let pu = (1.0 - bv) * av;
                        let ps = (1.0 - av) * bv;
                        let p0 = av * bv;
                        let den = pu + ps + p0;
                        r3[q] = if den > 0.0 { pu / den } else { 0.0 };
                    }
                    let d = cfg.damp;
                    let nd = 1.0 - d;
                    let mut pr = r3[1] * r3[2];
                    if pr > 1.0 { pr = 1.0; }
                    let s0 = eta.get_unchecked_mut(cs);     *s0 = d * *s0 + nd * pr;
                    let mut pr = r3[0] * r3[2];
                    if pr > 1.0 { pr = 1.0; }
                    let s1 = eta.get_unchecked_mut(cs + 1); *s1 = d * *s1 + nd * pr;
                    let mut pr = r3[0] * r3[1];
                    if pr > 1.0 { pr = 1.0; }
                    let s2 = eta.get_unchecked_mut(cs + 2); *s2 = d * *s2 + nd * pr;
                }
                continue;
            }
            for q in 0..len {
                unsafe {
                    let e = cs + q;
                    let k = *cl.get_unchecked(e) as usize;
                    let o = (1.0 - *eta.get_unchecked(e)).max(SPR_FLOOR);
                    let av = *p.get_unchecked(k) / o;
                    let bv = *p.get_unchecked(k ^ 1);
                    let pu = (1.0 - bv) * av;
                    let ps = (1.0 - av) * bv;
                    let p0 = av * bv;
                    let den = pu + ps + p0;
                    rr[q] = if den > 0.0 { pu / den } else { 0.0 };
                }
            }
            for q in 0..len {
                let mut prod = 1f32;
                for j in 0..len {
                    if j != q {
                        prod *= rr[j];
                    }
                }
                if prod > 1.0 {
                    prod = 1.0;
                }
                unsafe {
                    let slot = eta.get_unchecked_mut(cs + q);
                    *slot = cfg.damp * *slot + (1.0 - cfg.damp) * prod;
                }
            }
        }

        let lam = cfg.lam;
        for v in 0..nv {
            unsafe {
                let pm = *p.get_unchecked(2 * v);
                let pp = *p.get_unchecked(2 * v + 1);
                let pip = (1.0 - pp) * pm;
                let pim = (1.0 - pm) * pp;
                let den = pip + pim + pp * pm;
                let (wp, wm) = if den > 0.0 { (pip / den, pim / den) } else { (0.0, 0.0) };
                *x.get_unchecked_mut(v) = wp > wm;
                let hp = hf.get_unchecked_mut(2 * v + 1);
                *hp = (1.0 - lam) * *hp + lam * gamma * wp;
                let hm = hf.get_unchecked_mut(2 * v);
                *hm = (1.0 - lam) * *hm + lam * gamma * wm;
            }
        }

        if it % cfg.check_every == 0 {
            let mut u = 0usize;
            for a in 0..nc {
                let cs = unsafe { *co.get_unchecked(a) } as usize;
                let ce = unsafe { *co.get_unchecked(a + 1) } as usize;
                let mut ok = false;
                for j in cs..ce {
                    let l = unsafe { *cl.get_unchecked(j) };
                    if ((l & 1) == 1) == unsafe { *x.get_unchecked((l >> 1) as usize) } {
                        ok = true;
                        break;
                    }
                }
                if !ok {
                    u += 1;
                }
            }
            if u < best_u {
                best_u = u;
                best_it = it;
                best_x.copy_from_slice(&x);
                if u == 0 {
                    break;
                }
            } else if it - best_it >= cfg.patience {
                break;
            }
            if it >= cfg.min_it && best_u <= cfg.target_u {
                break;
            }
        }
    }
    (best_x, best_u, it)
}

const SPR_FAST3_DEF: bool = true;
const WALK_PROB_SEEDED_DEF: f64 = 0.45;

