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
    let mut unsat_pos = vec![u32::MAX; nc];
    let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);

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
        save_solution(&Solution { variables: vars })?;
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
    let mut phase_conf = vec![0u8; nv];
    let mut best_is_current = true;
    let mut stagnation_count: usize = 0;
    let mut reinit_count: usize = 0;

    #[derive(Clone, Copy)]
    struct OccB { ao0: u32, pb: u32, ao1: u32 }
    let mut occ_b = Vec::with_capacity(nv);
    for v in 0..nv {
        occ_b.push(OccB {
            ao0: all_off[v],
            pb: p_bound[v],
            ao1: all_off[v + 1],
        });
    }

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

                if best_is_current {
                    best_vars.copy_from_slice(&vars);
                    best_is_current = false;
                }
                vars.copy_from_slice(&best_vars);
                for confidence in phase_conf.iter_mut() {
                    *confidence = confidence.saturating_sub(2);
                }
                unsat_list.clear();
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
                    } else {
                        unsat_pos[c] = u32::MAX;
                    }
                }

                let mut polished = false;
                if best_unsat < 256 {
                    let repair_budget = (best_unsat * 16 + 64).min(2048);
                    let mut neutral_moves = 0usize;
                    for _ in 0..repair_budget {
                        if unsat_list.is_empty() {
                            polished = true;
                            break;
                        }

                        let clause_samples = unsat_list.len().min(6);
                        let clause_start = rng.gen::<usize>() % unsat_list.len();
                        let mut candidates: Vec<usize> = Vec::with_capacity(96);

                        for sample in 0..clause_samples {
                            let rid = (clause_start + sample) % unsat_list.len();
                            let Some(&raw_cid) = unsat_list.get(rid) else { continue; };
                            let cid = raw_cid as usize;
                            let Some(&raw_cs) = co.get(cid) else { continue; };
                            let Some(&raw_ce) = co.get(cid + 1) else { continue; };
                            let cs = raw_cs as usize;
                            let ce = raw_ce as usize;
                            if cs >= ce || ce > cl.len() {
                                continue;
                            }

                            let clen = ce - cs;
                            let inspected = clen.min(16);
                            let start = rng.gen::<usize>() % clen;
                            for step in 0..inspected {
                                let Some(&lit) = cl.get(cs + (start + step) % clen) else { continue; };
                                let v = (lit.abs() - 1) as usize;
                                if v < nv && !candidates.contains(&v) {
                                    candidates.push(v);
                                }
                            }
                        }

                        let mut chosen = None;
                        let mut best_score = isize::MIN;
                        let mut best_break = usize::MAX;
                        let mut ties = 0usize;

                        for v in candidates {
                            let Some(&was_true) = vars.get(v) else { continue; };
                            let Some(&ob) = occ_b.get(v) else { continue; };
                            let ao0 = ob.ao0 as usize;
                            let pb = ob.pb as usize;
                            let ao1 = ob.ao1 as usize;
                            if ao0 > pb || pb > ao1 || ao1 > all_data.len() {
                                continue;
                            }

                            let (is, ie, ds, de) = if was_true {
                                (pb, ao1, ao0, pb)
                            } else {
                                (ao0, pb, pb, ao1)
                            };
                            let mut make = 0usize;
                            for k in is..ie {
                                let Some(&raw_c) = all_data.get(k) else { continue; };
                                if num_good.get(raw_c as usize).copied() == Some(0) {
                                    make += 1;
                                }
                            }
                            let mut breaks = 0usize;
                            for k in ds..de {
                                let Some(&raw_c) = all_data.get(k) else { continue; };
                                if num_good.get(raw_c as usize).copied() == Some(1) {
                                    breaks += 1;
                                }
                            }

                            let score = make as isize - breaks as isize;
                            if score > best_score || (score == best_score && breaks < best_break) {
                                best_score = score;
                                best_break = breaks;
                                chosen = Some(v);
                                ties = 1;
                            } else if score == best_score && breaks == best_break {
                                ties += 1;
                                if rng.gen::<usize>() % ties == 0 {
                                    chosen = Some(v);
                                }
                            }
                        }

                        if best_score < 0 || (best_score == 0 && neutral_moves >= 8) {
                            continue;
                        }
                        if best_score == 0 {
                            neutral_moves += 1;
                        }

                        let Some(v) = chosen else { continue; };
                        let Some(&was_true) = vars.get(v) else { continue; };
                        let Some(&ob) = occ_b.get(v) else { continue; };
                        let ao0 = ob.ao0 as usize;
                        let pb = ob.pb as usize;
                        let ao1 = ob.ao1 as usize;
                        if ao0 > pb || pb > ao1 || ao1 > all_data.len() {
                            continue;
                        }
                        let (is, ie, ds, de) = if was_true {
                            (pb, ao1, ao0, pb)
                        } else {
                            (ao0, pb, pb, ao1)
                        };

                        for k in is..ie {
                            let Some(&raw_c) = all_data.get(k) else { continue; };
                            let c = raw_c as usize;
                            let Some(&ng) = num_good.get(c) else { continue; };
                            if ng == 0 {
                                let Some(&raw_pos) = unsat_pos.get(c) else { continue; };
                                let pos = raw_pos as usize;
                                if raw_pos != u32::MAX && pos < unsat_list.len() {
                                    let last_idx = unsat_list.len() - 1;
                                    let last_c = unsat_list[last_idx] as usize;
                                    unsat_list[pos] = last_c as u32;
                                    if let Some(last_pos) = unsat_pos.get_mut(last_c) {
                                        *last_pos = pos as u32;
                                    }
                                    if let Some(c_pos) = unsat_pos.get_mut(c) {
                                        *c_pos = u32::MAX;
                                    }
                                    unsat_list.pop();
                                }
                            }
                            if let Some(good) = num_good.get_mut(c) {
                                *good = good.saturating_add(1);
                            }
                        }

                        for k in ds..de {
                            let Some(&raw_c) = all_data.get(k) else { continue; };
                            let c = raw_c as usize;
                            let Some(&ng) = num_good.get(c) else { continue; };
                            if ng == 0 {
                                continue;
                            }
                            if let Some(good) = num_good.get_mut(c) {
                                *good = ng - 1;
                            }
                            if ng == 1 && unsat_pos.get(c).copied() == Some(u32::MAX) {
                                if let Some(c_pos) = unsat_pos.get_mut(c) {
                                    *c_pos = unsat_list.len() as u32;
                                    unsat_list.push(c as u32);
                                }
                            }
                        }
                        if let Some(value) = vars.get_mut(v) {
                            *value = !was_true;
                        }

                        if unsat_list.len() < best_unsat {
                            best_unsat = unsat_list.len();
                            best_vars.copy_from_slice(&vars);
                            best_is_current = true;
                            polished = true;
                            break;
                        }
                    }
                }

                if polished {
                    stagnation_count = 0;
                    last_check_residual = unsat_list.len();
                    continue;
                }

                let diversify_kicks = (best_unsat * 2 + 16).min(256).max(8);
                for _ in 0..diversify_kicks {
                    if unsat_list.is_empty() { break; }

                    let force_direct = rng.gen::<usize>() & 7 == 0;
                    let mut chosen = None;
                    let mut chosen_conf = u8::MAX;
                    let mut ties = 0usize;
                    let clause_samples = unsat_list.len().min(4);
                    let clause_start = rng.gen::<usize>() % unsat_list.len();

                    for sample in 0..clause_samples {
                        if force_direct && sample > 0 {
                            break;
                        }
                        let rid = (clause_start + sample) % unsat_list.len();
                        let Some(&raw_cid) = unsat_list.get(rid) else { continue; };
                        let pcid = raw_cid as usize;
                        let Some(&raw_pcs) = co.get(pcid) else { continue; };
                        let Some(&raw_pce) = co.get(pcid + 1) else { continue; };
                        let pcs = raw_pcs as usize;
                        let pce = raw_pce as usize;
                        if pcs >= pce || pce > cl.len() {
                            continue;
                        }

                        let plen = pce - pcs;
                        let inspected = plen.min(12);
                        let start = rng.gen::<usize>() % plen;
                        for step in 0..inspected {
                            let Some(&lit) = cl.get(pcs + (start + step) % plen) else { continue; };
                            let v = (lit.abs() - 1) as usize;
                            let Some(&confidence) = phase_conf.get(v) else { continue; };
                            if confidence < chosen_conf {
                                chosen_conf = confidence;
                                chosen = Some(v);
                                ties = 1;
                            } else if confidence == chosen_conf {
                                ties += 1;
                                if rng.gen::<usize>() % ties == 0 {
                                    chosen = Some(v);
                                }
                            }
                        }
                    }

                    let Some(v) = chosen else { continue; };
                    let was_true = *vars.get_unchecked(v);
                    let ob = *occ_b.get_unchecked(v);
                    let ao0 = ob.ao0 as usize;
                    let pb = ob.pb as usize;
                    let ao1 = ob.ao1 as usize;
                    let (is, ie, ds, de) = if was_true {
                        (pb, ao1, ao0, pb)
                    } else {
                        (ao0, pb, pb, ao1)
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

                let cur = unsat_list.len();
                if cur < best_unsat {
                    best_unsat = cur;
                    best_is_current = true;
                }
                stagnation_count = 0;
                last_check_residual = unsat_list.len();
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - unsat_list.len() as i64;

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= stagnation_limit_t4 {
                        if best_is_current {
                            best_vars.copy_from_slice(&vars);
                            best_is_current = false;
                        }
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
                            let ob = *occ_b.get_unchecked(v);
                            let ao0 = ob.ao0 as usize;
                            let pb = ob.pb as usize;
                            let ao1 = ob.ao1 as usize;
                            let (is, ie, ds, de) = if was_true {
                                (pb, ao1, ao0, pb)
                            } else {
                                (ao0, pb, pb, ao1)
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

            let v_idx = if clen == 3 {
                let l0 = *cl.get_unchecked(cs);
                let v0 = (l0.abs() - 1) as usize;
                let ob0 = *occ_b.get_unchecked(v0);
                let (os0, oe0) = if *vars.get_unchecked(v0) {
                    (ob0.ao0 as usize, ob0.pb as usize)
                } else {
                    (ob0.pb as usize, ob0.ao1 as usize)
                };
                let mut sad0 = 0usize;
                for k in os0..oe0 {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        sad0 += 1;
                    }
                }

                let l1 = *cl.get_unchecked(cs + 1);
                let v1 = (l1.abs() - 1) as usize;
                let ob1 = *occ_b.get_unchecked(v1);
                let (os1, oe1) = if *vars.get_unchecked(v1) {
                    (ob1.ao0 as usize, ob1.pb as usize)
                } else {
                    (ob1.pb as usize, ob1.ao1 as usize)
                };
                let mut sad1 = 0usize;
                for k in os1..oe1 {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        sad1 += 1;
                    }
                }

                let l2 = *cl.get_unchecked(cs + 2);
                let v2 = (l2.abs() - 1) as usize;
                let ob2 = *occ_b.get_unchecked(v2);
                let (os2, oe2) = if *vars.get_unchecked(v2) {
                    (ob2.ao0 as usize, ob2.pb as usize)
                } else {
                    (ob2.pb as usize, ob2.ao1 as usize)
                };
                let mut sad2 = 0usize;
                for k in os2..oe2 {
                    let c = *all_data.get_unchecked(k) as usize;
                    if *num_good.get_unchecked(c) == 1 {
                        sad2 += 1;
                    }
                }

                let z0 = sad0 == 0;
                let z1 = sad1 == 0;
                let z2 = sad2 == 0;
                if z0 | z1 | z2 {
                    let zcnt = (z0 as usize) + (z1 as usize) + (z2 as usize);
                    if zcnt == 1 {
                        if z0 { v0 } else if z1 { v1 } else { v2 }
                    } else {
                        let r = rand_val % zcnt;
                        if z0 {
                            if r == 0 {
                                v0
                            } else if z1 {
                                if r == 1 { v1 } else { v2 }
                            } else {
                                v2
                            }
                        } else if z1 {
                            if r == 0 { v1 } else { v2 }
                        } else {
                            v2
                        }
                    }
                } else {
                    let pw0 = *probs_break.get_unchecked(sad0.min(15));
                    let pw1 = *probs_break.get_unchecked(sad1.min(15));
                    let pw2 = *probs_break.get_unchecked(sad2.min(15));
                    let total_pw = pw0 + pw1 + pw2;
                    let mut r = (rand_val as u32) % total_pw.max(1);
                    if r < pw0 {
                        v0
                    } else {
                        r -= pw0;
                        if r < pw1 { v1 } else { v2 }
                    }
                }
            } else {
                let mut zero_buf: [usize; 3] = [0; 3];
                let mut zero_cnt: usize = 0;
                let mut pw_weights: [u32; 3] = [0; 3];
                let mut pw_vars: [usize; 3] = [0; 3];
                let mut pw_cnt: usize = 0;
                let mut total_pw: u32 = 0;

                for j in cs..ce {
                    let l = *cl.get_unchecked(j);
                    let abs_l = (l.abs() - 1) as usize;
                    let ob = *occ_b.get_unchecked(abs_l);
                    let (os, oe) = if *vars.get_unchecked(abs_l) {
                        (ob.ao0 as usize, ob.pb as usize)
                    } else {
                        (ob.pb as usize, ob.ao1 as usize)
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

                if zero_cnt > 0 {
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
                }
            };

            if best_is_current {
                best_vars.copy_from_slice(&vars);
                best_is_current = false;
            }

            let was_true = *vars.get_unchecked(v_idx);
            let ob = *occ_b.get_unchecked(v_idx);
            let ao0 = ob.ao0 as usize;
            let pb = ob.pb as usize;
            let ao1 = ob.ao1 as usize;
            let (is, ie, ds, de) = if was_true {
                (pb, ao1, ao0, pb)
            } else {
                (ao0, pb, pb, ao1)
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
                phase_conf[v_idx] = phase_conf[v_idx].saturating_add(3).min(15);
                best_is_current = true;
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }
        }
    }

    let final_vars = if unsat_list.is_empty() || best_is_current { vars } else { best_vars };
    save_solution(&Solution { variables: final_vars })?;

    Ok(())
}