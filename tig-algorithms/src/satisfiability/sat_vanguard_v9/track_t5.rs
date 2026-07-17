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
        density: _density,
        p_cnt,
        n_cnt,
        all_off,
        p_bound,
        all_data,
        mut cl,
        co,
    } = super::preprocess(challenge, save_solution);

    let max_fuel = hp.max_fuel_high.unwrap_or(155_000_000_000.0);
    let base_fuel = 50_000_000.0;
    let flip_fuel = 200.0;
    let max_flips = if max_fuel > base_fuel {
        ((max_fuel - base_fuel) / flip_fuel) as u64
    } else {
        10_000_000
    };

    const WALK_P: f64 = 0.52;
    const RESTART_PERIOD: u64 = 80_000_000;
    const REINIT_MIN_UNSAT: usize = 30;

    let mut assignment = vec![false; nv];
    let mut best_assignment = vec![false; nv];
    let mut true_lit_count = vec![0u8; nc];
    let mut unsat_clauses: Vec<usize> = Vec::with_capacity(nc);
    let mut clause_pos_in_unsat = vec![usize::MAX; nc];

    for v in 0..nv {
        let pc = p_cnt[v] as f64;
        let nc_v = n_cnt[v] as f64;
        let total = pc + nc_v;
        if total == 0.0 {
            assignment[v] = rng.gen_bool(0.5);
        } else {
            assignment[v] = rng.gen_bool((pc / total).clamp(0.2, 0.8));
        }
    }

    for c in 0..nc {
        let off = co[c] as usize;
        let end = co[c + 1] as usize;
        let mut trues = 0u8;
        for i in off..end {
            let lit = cl[i];
            let v = (lit.abs() - 1) as usize;
            if assignment[v] == (lit > 0) { trues += 1; }
        }
        true_lit_count[c] = trues;
        if trues == 0 {
            clause_pos_in_unsat[c] = unsat_clauses.len();
            unsat_clauses.push(c);
        }
    }

    let mut best_unsat = unsat_clauses.len();
    best_assignment.copy_from_slice(&assignment);
    let _ = save_solution(&Solution { variables: best_assignment.clone() });

    let mut period_best_unsat = best_unsat;
    let mut var_age = vec![0u8; nv];
    let mut global_flips: u64 = 0;

    unsafe {
        loop {
            if global_flips >= max_flips { break; }
            if unsat_clauses.is_empty() { break; }

            if global_flips > 0 && global_flips % RESTART_PERIOD == 0 {
                if period_best_unsat >= REINIT_MIN_UNSAT {

                    for v in 0..nv {
                        *assignment.get_unchecked_mut(v) = rng.gen_bool(0.5);
                    }

                    true_lit_count.fill(0);
                    unsat_clauses.clear();
                    clause_pos_in_unsat.fill(usize::MAX);
                    for c in 0..nc {
                        let off = *co.get_unchecked(c) as usize;
                        let end = *co.get_unchecked(c + 1) as usize;
                        let mut trues = 0u8;
                        for i in off..end {
                            let lit = *cl.get_unchecked(i);
                            let v = (lit.abs() - 1) as usize;
                            if *assignment.get_unchecked(v) == (lit > 0) { trues += 1; }
                        }
                        *true_lit_count.get_unchecked_mut(c) = trues;
                        if trues == 0 {
                            *clause_pos_in_unsat.get_unchecked_mut(c) = unsat_clauses.len();
                            unsat_clauses.push(c);
                        }
                    }

                    var_age.fill(0);

                    let cur = unsat_clauses.len();
                    if cur < best_unsat {
                        best_unsat = cur;
                        best_assignment.copy_from_slice(&assignment);
                        let _ = save_solution(&Solution { variables: best_assignment.clone() });
                    }
                }
                period_best_unsat = unsat_clauses.len();
            }

            let cur_unsat = unsat_clauses.len();
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }

            global_flips += 1;

            let r_idx = rng.gen_range(0..unsat_clauses.len());
            let c = *unsat_clauses.get_unchecked(r_idx);
            let off = *co.get_unchecked(c) as usize;
            let end = *co.get_unchecked(c + 1) as usize;
            let len = end - off;

            if len > 1 {
                let ri = (global_flips as usize) % len;
                cl.swap(off, off + ri);
            }

            let mut picked_v = usize::MAX;
            let mut vars = [0usize; 3];
            let mut breaks = [0u8; 3];
            let mut nvars = 0;

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                *vars.get_unchecked_mut(nvars) = v;

                let val = *assignment.get_unchecked(v);
                let (start, stop) = if val {
                    (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                } else {
                    (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                };

                let mut b = 0u8;
                for k in start..stop {
                    if *true_lit_count.get_unchecked(*all_data.get_unchecked(k) as usize) == 1 { b += 1; }
                }
                *breaks.get_unchecked_mut(nvars) = b;
                if b == 0 {
                    picked_v = v;
                    break;
                }
                nvars += 1;
            }

            if picked_v == usize::MAX {
                if nvars == 0 {
                    if let Some(&lit) = cl.get(off) {
                        picked_v = (lit.abs() - 1) as usize;
                    }
                } else if rng.gen_bool(WALK_P) {
                    picked_v = *vars.get_unchecked(rng.gen_range(0..nvars));
                } else {
                    let mut min_b = u8::MAX;
                    let mut best_idx = 0;
                    for i in 0..nvars {
                        let b = *breaks.get_unchecked(i);
                        let vi = *vars.get_unchecked(i);
                        let vb = *vars.get_unchecked(best_idx);
                        if b < min_b || (b == min_b && *var_age.get_unchecked(vi) < *var_age.get_unchecked(vb)) {
                            min_b = b;
                            best_idx = i;
                        }
                    }
                    picked_v = *vars.get_unchecked(best_idx);
                }
            }

            if picked_v == usize::MAX { continue; }

            let new_val = !*assignment.get_unchecked(picked_v);
            *assignment.get_unchecked_mut(picked_v) = new_val;

            let p_start = *all_off.get_unchecked(picked_v) as usize;
            let p_end = *p_bound.get_unchecked(picked_v) as usize;
            for k in p_start..p_end {
                let c_idx = *all_data.get_unchecked(k) as usize;
                if new_val {
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
                    if trues == 0 {
                        let last_c = unsat_clauses.pop().unwrap_unchecked();
                        let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
                        if last_c != c_idx {
                            *unsat_clauses.get_unchecked_mut(pos) = last_c;
                            *clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
                        }
                    }
                } else {
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
                    if trues == 1 {
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
                        unsat_clauses.push(c_idx);
                    }
                }
            }

            let n_start = *p_bound.get_unchecked(picked_v) as usize;
            let n_end = *all_off.get_unchecked(picked_v + 1) as usize;
            for k in n_start..n_end {
                let c_idx = *all_data.get_unchecked(k) as usize;
                if !new_val {
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues + 1;
                    if trues == 0 {
                        let last_c = unsat_clauses.pop().unwrap_unchecked();
                        let pos = *clause_pos_in_unsat.get_unchecked(c_idx);
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = usize::MAX;
                        if last_c != c_idx {
                            *unsat_clauses.get_unchecked_mut(pos) = last_c;
                            *clause_pos_in_unsat.get_unchecked_mut(last_c) = pos;
                        }
                    }
                } else {
                    let trues = *true_lit_count.get_unchecked(c_idx);
                    *true_lit_count.get_unchecked_mut(c_idx) = trues - 1;
                    if trues == 1 {
                        *clause_pos_in_unsat.get_unchecked_mut(c_idx) = unsat_clauses.len();
                        unsat_clauses.push(c_idx);
                    }
                }
            }

            for i in off..end {
                let lit = *cl.get_unchecked(i);
                let v = (lit.abs() - 1) as usize;
                let a = *var_age.get_unchecked(v);
                *var_age.get_unchecked_mut(v) = a.saturating_add(1);
            }

            let cur_unsat = unsat_clauses.len();
            if cur_unsat < period_best_unsat { period_best_unsat = cur_unsat; }
            if cur_unsat < best_unsat {
                best_unsat = cur_unsat;
                best_assignment.copy_from_slice(&assignment);
                let _ = save_solution(&Solution { variables: best_assignment.clone() });
                if cur_unsat == 0 { break; }
            }
        }
    }

    let _ = save_solution(&Solution { variables: best_assignment });
    Ok(())
}
