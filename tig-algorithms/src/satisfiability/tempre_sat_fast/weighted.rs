use anyhow::Result;
use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, SeedableRng};
use tig_challenges::satisfiability::*;

pub struct Params {
    pub weighted_restarts: u32,
    pub flips_multiplier: u64,
    pub cb_exp: u32,
    pub cambium_interval_divisor: usize,
    pub smooth_every: u32,
    pub perturb_pct: u32,
    pub crossover_pct: u32,
    pub crossover_bias: u32,
    pub stagnation_factor: u64,
    pub fast_restarts: u32,
    pub fast_flips_multiplier: u64,
}

#[inline]
fn pow_approx(base: u64, cb_exp: u32) -> u64 {
    if base <= 1 { return 1; }
    let int_part = cb_exp / 10;
    let frac = cb_exp % 10;
    let mut r = 1u64;
    let mut b = base;
    let mut e = int_part;
    while e > 0 {
        if e & 1 == 1 { r = r.saturating_mul(b); }
        b = b.saturating_mul(b);
        e >>= 1;
    }
    if frac > 0 {
        let ln_b = if base <= 10 {
            [0, 0, 710, 1124, 1419, 1648, 1835, 1992, 2129, 2251, 2359][base as usize]
        } else {
            2359 + 1024u64.saturating_mul(base - 10) / base
        };
        let x = ln_b * (frac as u64) / 10;
        let x2h = x.saturating_mul(x) / (2 * 1024);
        r = r.saturating_mul(1024 + x + x2h) / 1024;
    }
    r.max(1)
}

#[inline]
fn probsat_score(brk: u32, cb_exp: u32) -> u64 {
    if brk == 0 { return 1_000_000; }
    let d = pow_approx((1 + brk) as u64, cb_exp);
    if d == 0 { 1 } else { 1_000_000u64 / d }
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    p: &Params,
) -> Result<()> {
    let n = challenge.num_variables;
    let m = challenge.clauses.len();
    if m == 0 || n == 0 {
        save_solution(&Solution { variables: vec![false; n] })?;
        return Ok(());
    }

    save_solution(&Solution { variables: vec![false; n] })?;
    let mut rng = SmallRng::from_seed(StdRng::from_seed(challenge.seed).gen());

    // ── Build clause data + CSR (NO preprocessing — preserves tempre_sat indexing) ──
    let mut clause_vars: Vec<[u32; 3]> = Vec::with_capacity(m);
    let mut clause_signs: Vec<[bool; 3]> = Vec::with_capacity(m);
    let mut polarity: Vec<i32> = vec![0; n];
    let mut p_cnt = vec![0u32; n];
    let mut n_cnt = vec![0u32; n];

    for clause in &challenge.clauses {
        let mut vars = [0u32; 3];
        let mut signs = [false; 3];
        for li in 0..3 {
            let lit = clause[li];
            let vi = (lit.unsigned_abs() as usize) - 1;
            let positive = lit > 0;
            vars[li] = vi as u32;
            signs[li] = positive;
            if positive { p_cnt[vi] += 1; polarity[vi] += 1; }
            else { n_cnt[vi] += 1; polarity[vi] -= 1; }
        }
        clause_vars.push(vars);
        clause_signs.push(signs);
    }

    let mut p_off = vec![0u32; n + 1];
    let mut n_off = vec![0u32; n + 1];
    for v in 0..n { p_off[v+1] = p_off[v] + p_cnt[v]; n_off[v+1] = n_off[v] + n_cnt[v]; }
    let mut p_data = vec![0u32; p_off[n] as usize];
    let mut n_data = vec![0u32; n_off[n] as usize];
    {
        let mut pp = p_off[..n].to_vec();
        let mut np = n_off[..n].to_vec();
        for ci in 0..m {
            let ci32 = ci as u32;
            for li in 0..3 {
                let vi = clause_vars[ci][li] as usize;
                if clause_signs[ci][li] { p_data[pp[vi] as usize] = ci32; pp[vi] += 1; }
                else { n_data[np[vi] as usize] = ci32; np[vi] += 1; }
            }
        }
    }
    let var_app: Vec<usize> = (0..n).map(|v| (p_cnt[v] + n_cnt[v]) as usize).collect();

    // ── Solver state ──
    let cambium_interval = (n / p.cambium_interval_divisor).max(50);
    let stagnation_threshold = (m as u64) * p.stagnation_factor;

    let mut global_best_asgn: Vec<bool> = (0..n).map(|vi| polarity[vi] >= 0).collect();
    let mut local_best_asgn: Vec<bool> = global_best_asgn.clone();
    let mut global_best_sat: usize = 0;
    let mut local_best_asgn_sat: usize = 0;
    let mut conf_changed: Vec<bool> = vec![true; n];
    let mut weights: Vec<u16> = vec![1; m];
    let mut last_flip: Vec<u32> = vec![0; n];

    let total_restarts = p.weighted_restarts + p.fast_restarts;

    for restart in 0..total_restarts {
        let is_fast = restart >= p.weighted_restarts;
        let flips_budget = if is_fast {
            (m as u64) * p.fast_flips_multiplier
        } else {
            (m as u64) * p.flips_multiplier
        };

        // ── Initialize assignment ──
        let mut asgn: Vec<bool> = if is_fast {
            // Fast restarts: diverse initialization for different search landscape
            let fast_idx = restart - p.weighted_restarts;
            match fast_idx % 4 {
                0 => {
                    // Polarity-biased random (sat_imp_v1 style)
                    (0..n).map(|vi| {
                        let np = p_cnt[vi] as f64;
                        let nn = n_cnt[vi] as f64;
                        if nn == 0.0 && np > 0.0 { return true; }
                        if np == 0.0 && nn > 0.0 { return false; }
                        let prob = (np + 0.25) / (np + nn + 1.2);
                        rng.gen_bool(prob.clamp(0.01, 0.99))
                    }).collect()
                }
                1 => {
                    // Perturbed global best (20% flip rate)
                    let mut a = global_best_asgn.clone();
                    for vi in 0..n { if rng.gen_range(0u32..100) < 20 { a[vi] = !a[vi]; } }
                    a
                }
                2 => {
                    // Pure random
                    (0..n).map(|_| rng.gen_bool(0.5)).collect()
                }
                _ => {
                    // Inverse polarity
                    (0..n).map(|vi| polarity[vi] < 0).collect()
                }
            }
        } else if restart == 0 {
            (0..n).map(|vi| polarity[vi] >= 0).collect()
        } else if restart % 7 == 0 {
            (0..n).map(|_| rng.gen_bool(0.5)).collect()
        } else if rng.gen_range(0u32..100) < p.crossover_pct {
            let mut a = Vec::with_capacity(n);
            for vi in 0..n {
                let from_global = rng.gen_range(0u32..100) < p.crossover_bias;
                let base_val = if from_global { global_best_asgn[vi] } else { local_best_asgn[vi] };
                a.push(if rng.gen_range(0u32..100) < (p.perturb_pct / 2) { !base_val } else { base_val });
            }
            a
        } else {
            let mut a = global_best_asgn.clone();
            for vi in 0..n { if rng.gen_range(0u32..100) < p.perturb_pct { a[vi] = !a[vi]; } }
            a
        };

        if !is_fast { for v in conf_changed.iter_mut() { *v = true; } }
        last_flip.fill(0);

        // ── Build tc and unsat list ──
        let mut tc: Vec<u8> = vec![0; m];
        let mut unsat: Vec<u32> = Vec::with_capacity(m / 4);
        let mut unsat_pos: Vec<u32> = vec![u32::MAX; m];
        for ci in 0..m {
            for li in 0..3 {
                let vi = clause_vars[ci][li] as usize;
                let pos = clause_signs[ci][li];
                if (pos && asgn[vi]) || (!pos && !asgn[vi]) { tc[ci] += 1; }
            }
            if tc[ci] == 0 { unsat_pos[ci] = unsat.len() as u32; unsat.push(ci as u32); }
        }

        let cur_sat = m - unsat.len();
        if cur_sat > global_best_sat {
            global_best_sat = cur_sat;
            global_best_asgn.copy_from_slice(&asgn);
            save_solution(&Solution { variables: asgn.clone() })?;
        }
        if unsat.is_empty() { return Ok(()); }

        let mut cambium_bumps: u32 = 0;
        let mut flips_since_improve: u64 = 0;
        let mut local_best_sat: usize = cur_sat;
        let mut restart_best_asgn: Vec<bool> = asgn.clone();

        // Fast-mode adaptive probability state
        let fast_base_prob: f64 = 0.52;
        let mut current_prob = fast_base_prob;
        let fast_check_interval: usize = 100;
        let mut fast_last_check_unsat = unsat.len();
        let mut fast_stagnation: u32 = 0;

        // ── Main flip loop ──
        unsafe {
        for flip_num in 1u32..=(flips_budget.min(u32::MAX as u64) as u32) {
            if unsat.is_empty() { break; }

            let pick = rng.gen_range(0..unsat.len());
            let ci = *unsat.get_unchecked(pick) as usize;
            let cvars = *clause_vars.get_unchecked(ci);

            // ── Compute break counts ──
            let mut brk = [0u32; 3];
            let mut has_freebie = false;
            for k in 0..3usize {
                let vi = cvars[k] as usize;
                let cur = *asgn.get_unchecked(vi);
                let (ls, le, la) = if cur {
                    (*p_off.get_unchecked(vi), *p_off.get_unchecked(vi+1), &p_data)
                } else {
                    (*n_off.get_unchecked(vi), *n_off.get_unchecked(vi+1), &n_data)
                };
                let mut b = 0u32;
                for j in ls..le {
                    if *tc.get_unchecked(*la.get_unchecked(j as usize) as usize) == 1 { b += 1; }
                }
                brk[k] = b;
                if b == 0 { has_freebie = true; }
            }

            // ── Select variable to flip ──
            let flip_var: usize;

            if is_fast {
                // ═══ FAST MODE: freebie → random walk → min-break with age ═══
                if has_freebie {
                    let mut best_k: usize = 0;
                    let mut best_age = 0u32;
                    let mut first = true;
                    for k in 0..3usize {
                        if brk[k] == 0 {
                            let vi = cvars[k] as usize;
                            let age = flip_num.wrapping_sub(*last_flip.get_unchecked(vi));
                            if first || age > best_age {
                                best_age = age; best_k = k; first = false;
                            }
                        }
                    }
                    flip_var = cvars[best_k] as usize;
                } else if rng.gen::<f64>() < current_prob {
                    flip_var = cvars[rng.gen_range(0..3usize)] as usize;
                } else {
                    // Min-break with age+appearance tie-breaking
                    let mut min_sad = usize::MAX;
                    let mut best_k: usize = 0;
                    let mut min_weight = usize::MAX;
                    for k in 0..3usize {
                        let vi = cvars[k] as usize;
                        let sad = brk[k] as usize;
                        let app = *var_app.get_unchecked(vi);
                        let age = flip_num.wrapping_sub(*last_flip.get_unchecked(vi)) as usize;
                        if sad == 0 {
                            let aw = app.saturating_sub(age / 4);
                            if min_sad > 0 || aw < min_weight { min_sad = 0; min_weight = aw; best_k = k; }
                        } else if min_sad > 0 {
                            let cw = sad * 1000 + app.saturating_sub((age / 2).min(50));
                            if cw < min_weight { min_sad = sad; min_weight = cw; best_k = k; }
                        }
                    }
                    flip_var = cvars[best_k] as usize;
                }
            } else {
                // ═══ WEIGHTED MODE: freebie(wmk) → ProbSAT+conf_changed+wmk ═══
                if has_freebie {
                    let mut best_k: usize = 0;
                    let mut best_wm: u64 = 0;
                    let mut first = true;
                    for k in 0..3usize {
                        if brk[k] != 0 { continue; }
                        let vi = cvars[k] as usize;
                        let cur = *asgn.get_unchecked(vi);
                        let (gs, ge, ga) = if cur {
                            (*n_off.get_unchecked(vi), *n_off.get_unchecked(vi+1), &n_data)
                        } else {
                            (*p_off.get_unchecked(vi), *p_off.get_unchecked(vi+1), &p_data)
                        };
                        let mut wm = 0u64;
                        for j in gs..ge {
                            let c = *ga.get_unchecked(j as usize) as usize;
                            if *tc.get_unchecked(c) == 0 { wm += *weights.get_unchecked(c) as u64; }
                        }
                        if first || wm > best_wm { best_wm = wm; best_k = k; first = false; }
                    }
                    flip_var = cvars[best_k] as usize;
                } else {
                    let mut scores = [0u64; 3];
                    let mut total = 0u64;
                    for k in 0..3usize {
                        let vi = cvars[k] as usize;
                        let cur = *asgn.get_unchecked(vi);
                        let mut s = probsat_score(brk[k], p.cb_exp);
                        if *conf_changed.get_unchecked(vi) { s = s.saturating_mul(2); }
                        let (gs, ge, ga) = if cur {
                            (*n_off.get_unchecked(vi), *n_off.get_unchecked(vi+1), &n_data)
                        } else {
                            (*p_off.get_unchecked(vi), *p_off.get_unchecked(vi+1), &p_data)
                        };
                        let mut wm = 0u64;
                        for j in gs..ge {
                            let c = *ga.get_unchecked(j as usize) as usize;
                            if *tc.get_unchecked(c) == 0 { wm += *weights.get_unchecked(c) as u64; }
                        }
                        s = s.saturating_add(wm);
                        scores[k] = s; total += s;
                    }
                    if total == 0 {
                        flip_var = cvars[rng.gen_range(0..3usize)] as usize;
                    } else {
                        let r = rng.gen_range(0..total);
                        if r < scores[0] { flip_var = cvars[0] as usize; }
                        else if r < scores[0] + scores[1] { flip_var = cvars[1] as usize; }
                        else { flip_var = cvars[2] as usize; }
                    }
                }
            }

            // ── Execute flip (same for both modes) ──
            *asgn.get_unchecked_mut(flip_var) = !*asgn.get_unchecked(flip_var);
            let new_val = *asgn.get_unchecked(flip_var);
            *last_flip.get_unchecked_mut(flip_var) = flip_num;
            if !is_fast { *conf_changed.get_unchecked_mut(flip_var) = false; }

            let (gs, ge, ga) = if new_val {
                (*p_off.get_unchecked(flip_var), *p_off.get_unchecked(flip_var+1), &p_data)
            } else {
                (*n_off.get_unchecked(flip_var), *n_off.get_unchecked(flip_var+1), &n_data)
            };
            for j in gs..ge {
                let oci = *ga.get_unchecked(j as usize) as usize;
                *tc.get_unchecked_mut(oci) += 1;
                if *tc.get_unchecked(oci) == 1 {
                    let pos = *unsat_pos.get_unchecked(oci) as usize;
                    let last = *unsat.last().unwrap() as usize;
                    *unsat.get_unchecked_mut(pos) = last as u32;
                    *unsat_pos.get_unchecked_mut(last) = pos as u32;
                    unsat.pop();
                    *unsat_pos.get_unchecked_mut(oci) = u32::MAX;
                    if !is_fast {
                        for lk in 0..3usize {
                            let vk = clause_vars.get_unchecked(oci)[lk] as usize;
                            if vk != flip_var { *conf_changed.get_unchecked_mut(vk) = true; }
                        }
                    }
                }
            }

            let (ls, le, la) = if new_val {
                (*n_off.get_unchecked(flip_var), *n_off.get_unchecked(flip_var+1), &n_data)
            } else {
                (*p_off.get_unchecked(flip_var), *p_off.get_unchecked(flip_var+1), &p_data)
            };
            for j in ls..le {
                let oci = *la.get_unchecked(j as usize) as usize;
                *tc.get_unchecked_mut(oci) -= 1;
                if *tc.get_unchecked(oci) == 0 {
                    *unsat_pos.get_unchecked_mut(oci) = unsat.len() as u32;
                    unsat.push(oci as u32);
                    if !is_fast {
                        for lk in 0..3usize {
                            let vk = clause_vars.get_unchecked(oci)[lk] as usize;
                            if vk != flip_var { *conf_changed.get_unchecked_mut(vk) = true; }
                        }
                    }
                }
            }

            // ── Track improvement ──
            let cur_sat2 = m - unsat.len();
            if cur_sat2 > local_best_sat {
                local_best_sat = cur_sat2;
                restart_best_asgn.copy_from_slice(&asgn);
                flips_since_improve = 0;
            } else {
                flips_since_improve += 1;
            }
            if cur_sat2 > global_best_sat {
                global_best_sat = cur_sat2;
                global_best_asgn.copy_from_slice(&asgn);
                save_solution(&Solution { variables: asgn.clone() })?;
                if unsat.is_empty() { return Ok(()); }
            }

            // ── Mode-specific evolution ──
            if is_fast {
                // Adaptive random walk probability + stagnation kicks
                if flip_num as usize % fast_check_interval == 0 {
                    let cur_u = unsat.len();
                    let progress = fast_last_check_unsat as i64 - cur_u as i64;
                    let pr = progress as f64 / fast_last_check_unsat.max(1) as f64;
                    if progress <= 0 {
                        fast_stagnation += 1;
                        current_prob = (current_prob + 0.025).min(0.9);
                        if fast_stagnation >= 4 {
                            let kicks = if fast_stagnation >= 8 { 6usize } else { 3 };
                            for _ in 0..kicks {
                                if unsat.is_empty() { break; }
                                let rid = rng.gen_range(0..unsat.len());
                                let pcid = *unsat.get_unchecked(rid) as usize;
                                let pcs = clause_vars.get_unchecked(pcid);
                                let lit_k = rng.gen_range(0..3usize);
                                let v = pcs[lit_k] as usize;
                                let was = *asgn.get_unchecked(v);
                                let (ks, ke, ka) = if was { (*n_off.get_unchecked(v), *n_off.get_unchecked(v+1), &n_data) } else { (*p_off.get_unchecked(v), *p_off.get_unchecked(v+1), &p_data) };
                                let (kds, kde, kda) = if was { (*p_off.get_unchecked(v), *p_off.get_unchecked(v+1), &p_data) } else { (*n_off.get_unchecked(v), *n_off.get_unchecked(v+1), &n_data) };
                                for j in ks..ke {
                                    let c = *ka.get_unchecked(j as usize) as usize;
                                    *tc.get_unchecked_mut(c) += 1;
                                    if *tc.get_unchecked(c) == 1 {
                                        let pos2 = *unsat_pos.get_unchecked(c) as usize;
                                        let last2 = *unsat.last().unwrap() as usize;
                                        *unsat.get_unchecked_mut(pos2) = last2 as u32;
                                        *unsat_pos.get_unchecked_mut(last2) = pos2 as u32;
                                        unsat.pop();
                                        *unsat_pos.get_unchecked_mut(c) = u32::MAX;
                                    }
                                }
                                for j in kds..kde {
                                    let c = *kda.get_unchecked(j as usize) as usize;
                                    *tc.get_unchecked_mut(c) -= 1;
                                    if *tc.get_unchecked(c) == 0 {
                                        *unsat_pos.get_unchecked_mut(c) = unsat.len() as u32;
                                        unsat.push(c as u32);
                                    }
                                }
                                *asgn.get_unchecked_mut(v) = !was;
                                *last_flip.get_unchecked_mut(v) = flip_num;
                            }
                            fast_stagnation = 0;
                        }
                    } else if pr > 0.15 {
                        fast_stagnation = 0;
                        current_prob = fast_base_prob;
                    } else {
                        fast_stagnation = 0;
                        current_prob = current_prob * 0.8 + fast_base_prob * 0.2;
                    }
                    fast_last_check_unsat = unsat.len();
                }
            } else {
                // Weighted: clause weight evolution
                if flips_since_improve > 0 && flips_since_improve as usize % cambium_interval == 0 {
                    for &uci in &unsat {
                        let w = weights.get_unchecked_mut(uci as usize);
                        *w = w.saturating_add(3);
                    }
                    cambium_bumps += 1;
                    if cambium_bumps % p.smooth_every == 0 {
                        for w in weights.iter_mut() { *w = (*w / 2).max(1); }
                    }
                }
                if flips_since_improve >= stagnation_threshold {
                    for w in weights.iter_mut() { *w = (*w / 4).max(1); }
                    flips_since_improve = 0;
                }
            }
        }
        } // end unsafe

        if local_best_sat > local_best_asgn_sat {
            local_best_asgn_sat = local_best_sat;
            local_best_asgn = restart_best_asgn;
        }
    }

    Ok(())
}