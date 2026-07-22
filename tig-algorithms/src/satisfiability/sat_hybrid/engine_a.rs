// SAT solver engine A — self-contained per-track SAT engine.
mod track1 {
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
                *unsat_list.get_unchecked(i1) as usize
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
}
mod track2 {
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;

unsafe fn apply_flip(
    v_idx: usize,
    nv: usize,
    p_off: &[u32],
    n_off: &[u32],
    p_data: &[u32],
    n_data: &[u32],
    vars: &mut [bool],
    num_good: &mut [u8],
    residual: &mut Vec<u32>,
    unsat_count: &mut usize,
    var_age: &mut [u16],
    flip_journal: &mut Vec<u32>,
    attempt_base_vars: &mut [bool],
    best_vars: &mut [bool],
    best_journal_len: usize,
    attempt_best_pending: &mut bool,
) {
    let was_true = *vars.get_unchecked(v_idx);
    let (is, ie, ia): (usize, usize, &[u32]) = if was_true {
        (
            *n_off.get_unchecked(v_idx) as usize,
            *n_off.get_unchecked(v_idx + 1) as usize,
            n_data,
        )
    } else {
        (
            *p_off.get_unchecked(v_idx) as usize,
            *p_off.get_unchecked(v_idx + 1) as usize,
            p_data,
        )
    };
    let (ds, de, da): (usize, usize, &[u32]) = if was_true {
        (
            *p_off.get_unchecked(v_idx) as usize,
            *p_off.get_unchecked(v_idx + 1) as usize,
            p_data,
        )
    } else {
        (
            *n_off.get_unchecked(v_idx) as usize,
            *n_off.get_unchecked(v_idx + 1) as usize,
            n_data,
        )
    };

    for k in is..ie {
        let c = *ia.get_unchecked(k) as usize;
        let ng = *num_good.get_unchecked(c);
        if ng == 0 {
            *unsat_count -= 1;
        }
        *num_good.get_unchecked_mut(c) = ng.saturating_add(1);
    }
    for k in ds..de {
        let c = *da.get_unchecked(k) as usize;
        let ng = num_good.get_unchecked_mut(c);
        let new_val = ng.saturating_sub(1);
        *ng = new_val;
        if new_val == 0 {
            *unsat_count += 1;
            residual.push(c as u32);
        }
    }

    *vars.get_unchecked_mut(v_idx) = !was_true;
    *var_age.get_unchecked_mut(v_idx) = 0;
    flip_journal.push(v_idx as u32);
    if flip_journal.len() == nv {
        if *attempt_best_pending {
            best_vars.copy_from_slice(attempt_base_vars);
            for &fv in &flip_journal[..best_journal_len] {
                let idx = fv as usize;
                let bit = *best_vars.get_unchecked(idx);
                *best_vars.get_unchecked_mut(idx) = !bit;
            }
            *attempt_best_pending = false;
        }
        attempt_base_vars.copy_from_slice(vars);
        flip_journal.clear();
    }
}

/// Solves the SAT problem using SLS with Adaptive Phase Saving on Restarts.
/// Variables with higher appearance counts or polarity skew have a higher
/// probability of maintaining their phase (`best_vars`) during restarts.
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
    let mut compact_clauses = Vec::with_capacity(challenge.clauses.len());
    let mut compact_len = Vec::with_capacity(challenge.clauses.len());
    let mut total_lits = 0usize;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }

        let mut lits = [0i32; 3];
        let mut len = 1usize;
        lits[0] = a;

        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }

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

        total_lits += len;
        compact_clauses.push(lits);
        compact_len.push(len as u8);
    }

    let nc = compact_clauses.len();

    let mut p_off = vec![0u32; nv + 1];
    let mut n_off = vec![0u32; nv + 1];
    for v in 0..nv {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    let mut p_data = vec![0u32; p_off[nv] as usize];
    let mut n_data = vec![0u32; n_off[nv] as usize];

    let mut p_pos = p_off[..nv].to_vec();
    let mut n_pos = n_off[..nv].to_vec();
    let mut cl = Vec::with_capacity(total_lits);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);
    for (ci, lits) in compact_clauses.iter().enumerate() {
        let ci_u32 = ci as u32;
        let len = compact_len[ci] as usize;
        for j in 0..len {
            let lit = lits[j];
            let v = (lit.abs() - 1) as usize;
            if lit > 0 {
                p_data[p_pos[v] as usize] = ci_u32;
                p_pos[v] += 1;
            } else {
                n_data[n_pos[v] as usize] = ci_u32;
                n_pos[v] += 1;
            }
        }
        cl.extend_from_slice(&lits[..len]);
        co.push(cl.len() as u32);
    }

    let density = nc as f64 / nv as f64;
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(250_000_000_000.0);

    let var_appearances: Vec<usize> = (0..nv)
        .map(|v| (p_cnt[v] + n_cnt[v]) as usize)
        .collect();

    let max_app = *var_appearances.iter().max().unwrap_or(&1) as f64;
    let base_keep_prob = if nv <= 20000 { 0.15 } else { 0.09 };
    
    let phase_save_threshold: Vec<u32> = (0..nv).map(|v| {
        let app = var_appearances[v] as f64;
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        let skew = if np + nn > 0.0 { (np - nn).abs() / (np + nn) } else { 0.0 };
        let prob = (base_keep_prob + 0.35 * (app / max_app) + 0.25 * skew).clamp(0.0, 0.90);
        (prob * 4294967295.0) as u32
    }).collect();

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
    let smoothing_factor: f64 = 0.8;

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;
    let variance_interval = 1000usize;

    let raw_restarts = if nv <= 12000 { 4usize } else if nv <= 40000 { 3usize } else { 2usize };
    let restart_attempts = raw_restarts
        .saturating_sub(if density > 5.0 { 1 } else { 0 })
        .max(1)
        .min(max_flips.max(1));

    let mut best_unsat = nc + 1;
    let mut best_vars = vec![false; nv];
    let mut vars = vec![false; nv];
    let mut attempt_base_vars = vec![false; nv];
    let mut num_good = vec![0u8; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut flip_journal: Vec<u32> = Vec::with_capacity(nv);
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
            for v in 0..nv {
                if p_cnt[v] > 0 && n_cnt[v] > 0 && rng.gen::<u32>() < phase_save_threshold[v] {
                    vars[v] = best_vars[v];
                }
            }
        }

        attempt_base_vars.clone_from(&vars);
        flip_journal.clear();
        let mut best_journal_len = 0usize;
        let mut attempt_best_pending = false;

        num_good.fill(0);
        residual.clear();
        for i in 0..nc {
            let s = co[i] as usize;
            let e = co[i + 1] as usize;
            let mut ng = 0u8;
            for j in s..e {
                let l = cl[j];
                let v = (l.abs() - 1) as usize;
                if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                    ng += 1;
                }
            }
            num_good[i] = ng;
            if ng == 0 {
                residual.push(i as u32);
            }
        }
        let mut unsat_count = residual.len();

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            best_journal_len = flip_journal.len();
            attempt_best_pending = true;
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }

        let mut current_prob = base_prob;
        var_age.fill(0);
        let mut rounds = 0usize;
        let mut stagnation = 0usize;
        let mut max_unsat_window = unsat_count;
        let mut min_unsat_window = unsat_count;

        unsafe {
            loop {
                if rounds >= attempt_budget || unsat_count == 0 { break; }

                if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
                if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }

                if rounds % check_interval == 0 && rounds > 0 {
                    if unsat_count < best_unsat {
                        best_unsat = unsat_count;
                        best_journal_len = flip_journal.len();
                        attempt_best_pending = true;
                    }
                }

                if rounds % variance_interval == 0 && rounds > 0 {
                    let variance = max_unsat_window.saturating_sub(min_unsat_window);

                    if variance <= 2 {
                        stagnation += 1;
                        current_prob = (current_prob + 0.15).min(max_random_prob);
                    } else if variance <= 6 {
                        stagnation += 1;
                        current_prob = (current_prob + 0.05).min(max_random_prob);
                    } else if variance >= 20 {
                        stagnation = 0;
                        current_prob = base_prob;
                    } else {
                        stagnation = 0;
                        current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                    }

                    if stagnation >= 3 {
                        let kicks = if stagnation >= 6 { 8 } else { 4 };
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

                            apply_flip(
                                v,
                                nv,
                                &p_off,
                                &n_off,
                                &p_data,
                                &n_data,
                                &mut vars,
                                &mut num_good,
                                &mut residual,
                                &mut unsat_count,
                                &mut var_age,
                                &mut flip_journal,
                                &mut attempt_base_vars,
                                &mut best_vars,
                                best_journal_len,
                                &mut attempt_best_pending,
                            );
                        }
                        stagnation = 0;
                    }

                    max_unsat_window = unsat_count;
                    min_unsat_window = unsat_count;
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

                apply_flip(
                    v_idx,
                    nv,
                    &p_off,
                    &n_off,
                    &p_data,
                    &n_data,
                    &mut vars,
                    &mut num_good,
                    &mut residual,
                    &mut unsat_count,
                    &mut var_age,
                    &mut flip_journal,
                    &mut attempt_base_vars,
                    &mut best_vars,
                    best_journal_len,
                    &mut attempt_best_pending,
                );
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
                    best_journal_len = flip_journal.len();
                    attempt_best_pending = true;
                }
            }
        }

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
            best_journal_len = flip_journal.len();
            attempt_best_pending = true;
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }
        if attempt_best_pending {
            best_vars.clone_from(&attempt_base_vars);
            for &fv in &flip_journal[..best_journal_len] {
                best_vars[fv as usize] = !best_vars[fv as usize];
            }
        }
    }

    let _ = save_solution(&Solution { variables: best_vars });
    Ok(())
}
}
mod track3 {
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
}
mod track4 {
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

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 {
            residual.push(i as u32);
        }
    }

    if residual.is_empty() {
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

    let mut last_check_residual = residual.len();
    let mut stagnation = 0usize;
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;
    let mut rounds = 0usize;

    let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    unsafe {
        loop {
            if residual.is_empty() || rounds >= max_flips { break; }

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
            if !found { break; }

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
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

}
mod track5 {
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
                        }
                        stagnation = 0;
                    }
                } else {
                    stagnation = 0;
                }

                last_check_residual = unsat_list.len();
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
            } else {
                let mut sum_scores = 0.0;
                let mut scores = [0.0; 256];
                let limit = (ce - cs).min(256);
                for idx in 0..limit {
                    let j = cs + idx;
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
                    }
                    let score = *probs_break.get_unchecked(sad.min(255));
                    sum_scores += score;
                    *scores.get_unchecked_mut(idx) = score;
                }
                
                let threshold = rng.gen::<f64>() * sum_scores;
                let mut accum = 0.0;
                let mut v_sel = (cl.get_unchecked(cs).abs() - 1) as usize;
                for idx in 0..limit {
                    accum += *scores.get_unchecked(idx);
                    if accum >= threshold {
                        v_sel = (cl.get_unchecked(cs + idx).abs() - 1) as usize;
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
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}
}
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
}


pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    });

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

    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    let density = nc as f64 / nv as f64;

    if density >= 4.25 {
        if nv <= 5000 {
            return track1::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
        }
        if nv <= 7500 {
            return track2::solve(challenge, &hp, save_solution);
        }
        return track3::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
    }
    if density < 4.18 {
        return track4::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution);
    }
    track5::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution)
}
