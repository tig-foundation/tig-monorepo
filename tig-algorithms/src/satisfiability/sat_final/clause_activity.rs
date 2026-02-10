use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::convert::TryInto;
use tig_challenges::satisfiability::*;
use crate::{seeded_hasher, HashSet};
use super::helpers::*;

pub fn solve_track_2_clause_activity_impl(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    base_prob: f64,
    check_interval: usize,
    max_restarts: u8,
    max_fuel: f64,
    restart_after: usize,
    endgame_threshold: usize,
) -> anyhow::Result<()> {
    let _ = seeded_hasher;
    let _hs_smoke: HashSet<u8> = HashSet::default();
    drop(_hs_smoke);

    let n = challenge.num_variables as usize;

    let _ = save_solution(&Solution {
        variables: vec![false; n],
    });

    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(
        challenge.seed[..8].try_into().unwrap(),
    ));

    let mut clauses = challenge.clauses.clone();
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        let clause = &mut clauses[i];

        if clause.len() > 1 {
            let mut j = 1usize;
            while j < clause.len() {
                if clause[..j].contains(&clause[j]) {
                    clause.swap_remove(j);
                } else {
                    j += 1;
                }
            }
        }

        let mut is_tautology = false;
        for &lit in clause.iter() {
            if clause.contains(&-lit) {
                is_tautology = true;
                break;
            }
        }
        if is_tautology {
            clauses.swap_remove(i);
        }
    }

    let mut p_single = vec![false; n];
    let mut n_single = vec![false; n];
    let mut clauses_ = clauses;
    clauses = Vec::with_capacity(clauses_.len());
    let mut dead = false;

    while !dead {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len());
            let mut skip = false;
            for (ii, l) in c.iter().enumerate() {
                let v = (l.abs() as usize) - 1;
                if (p_single[v] && *l > 0) || (n_single[v] && *l < 0) || c[(ii + 1)..].contains(&-l)
                {
                    skip = true;
                    break;
                } else if p_single[v] || n_single[v] || c[(ii + 1)..].contains(l) {
                    done = false;
                    continue;
                } else {
                    c_.push(*l);
                }
            }
            if skip {
                done = false;
                continue;
            }
            match c_[..] {
                [l] => {
                    done = false;
                    let v = (l.abs() as usize) - 1;
                    if l > 0 {
                        if n_single[v] {
                            dead = true;
                            break;
                        } else {
                            p_single[v] = true;
                        }
                    } else if p_single[v] {
                        dead = true;
                        break;
                    } else {
                        n_single[v] = true;
                    }
                }
                [] => {
                    dead = true;
                    break;
                }
                _ => clauses.push(c_),
            }
        }
        if done {
            break;
        }
        clauses_ = clauses;
        clauses = Vec::with_capacity(clauses_.len());
    }

    if dead {
        return Ok(());
    }

    let m = clauses.len();
    if m == 0 {
        let mut variables = vec![false; n];
        for v in 0..n {
            if p_single[v] {
                variables[v] = true;
            } else if n_single[v] {
                variables[v] = false;
            } else {
                variables[v] = rng.gen();
            }
        }
        if verify_all(&challenge.clauses, &variables) {
            let _ = save_solution(&Solution { variables });
        }
        return Ok(());
    }
   
    let mut p_cnt = vec![0u32; n];
    let mut n_cnt = vec![0u32; n];
    
    for c in clauses.iter() {
        for &l in c.iter() {
            let v = (l.abs() as usize) - 1;
            if l > 0 {
                p_cnt[v] += 1;
            } else {
                n_cnt[v] += 1;
            }
        }
    }    
    
    let mut p_off = vec![0u32; n + 1];
    let mut n_off = vec![0u32; n + 1];
    for v in 0..n {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    
    let mut p_flat = vec![0u32; p_off[n] as usize];
    let mut n_flat = vec![0u32; n_off[n] as usize];
    
    let mut p_cur = p_off[..n].to_vec();
    let mut n_cur = n_off[..n].to_vec();
    
    for (cid, c) in clauses.iter().enumerate() {
        let cid_u32 = cid as u32;
        for &l in c.iter() {
            let v = (l.abs() as usize) - 1;
            if l > 0 {
                let idx = p_cur[v] as usize;
                p_flat[idx] = cid_u32;
                p_cur[v] += 1;
            } else {
                let idx = n_cur[v] as usize;
                n_flat[idx] = cid_u32;
                n_cur[v] += 1;
            }
        }
    }
    
    #[inline(always)]
    fn occ_slice<'a>(flat: &'a [u32], off: &[u32], v: usize) -> &'a [u32] {
        let s = off[v] as usize;
        let e = off[v + 1] as usize;
        &flat[s..e]
    }

    let mut p_clauses: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut n_clauses: Vec<Vec<usize>> = Vec::with_capacity(n);
    for v in 0..n {
        p_clauses.push(occ_slice(&p_flat, &p_off, v).iter().map(|&x| x as usize).collect());
        n_clauses.push(occ_slice(&n_flat, &n_off, v).iter().map(|&x| x as usize).collect());
    }

    let density = m as f64 / n as f64;
    let avg_clause_size = clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / m as f64;

    let pad = 1.8;
    let nad = 0.56;

    let mut pref: Vec<i8> = vec![0i8; n];
    for v in 0..n {
        if p_single[v] {
            pref[v] = 2;
            continue;
        }
        if n_single[v] {
            pref[v] = -2;
            continue;
        }
        let num_p = occ_slice(&p_flat, &p_off, v).len();
        let num_n = occ_slice(&n_flat, &n_off, v).len();
        if num_p == 0 && num_n == 0 {
            pref[v] = 0;
        } else {
            let vad = (num_p as f64 + 1.0) / (num_n as f64 + 1.0);
            if vad >= pad {
                pref[v] = 1;
            } else if vad <= nad {
                pref[v] = -1;
            } else {
                pref[v] = 0;
            }
        }
    }

    let mut variables = vec![false; n];
    init_from_pref(&mut rng, &pref, 0, &mut variables);

    let mut num_good: Vec<u8> = vec![0; m];
    let mut residual: Vec<usize> = Vec::with_capacity(m / 10 + 8);
    recompute_state(&clauses, &variables, &mut num_good, &mut residual);

    let mut current_prob = base_prob;

    let endgame_interval = (check_interval / 2).max(20);
    let mut last_check_residual = residual.len();

    let difficulty_factor = density * avg_clause_size.sqrt();
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (n as f64).sqrt();
    let flip_fuel = 200.0 + difficulty_factor;
    let max_num_rounds = ((max_fuel - base_fuel).max(0.0) / flip_fuel) as usize;

    let mut rounds = 0usize;
    let mut last_flip: usize = usize::MAX;

    let mut clause_marks: Vec<u32> = vec![0u32; m];
    let mut stamp: u32 = 1;

    let mut clause_activity_t2: Vec<u16> = vec![1u16; m];

    let mut var_marks: Vec<u32> = vec![0u32; n];
    let mut var_stamp: u32 = 1;
    let mut var_cnt: Vec<u8> = vec![0u8; n];

    let mut best_endgame = usize::MAX;
    let mut endgame_stuck = 0usize;

    let mut pos: Vec<i16> = vec![-1i16; n];
    let mut bf_tries: u8 = 0;

    let mut best_global = residual.len();
    let mut last_best_round = 0usize;
    let mut restarts_done: u8 = 0;
    
    let mut best_global_vars = variables.clone();

    unsafe {
        loop {
            if rounds >= max_num_rounds {
                return Ok(());
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    let prob_adjustment =
                        0.025 * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(0.9);

                    let mut bumped_t2 = 0usize;
                    for &cid in residual.iter() {
                        if *num_good.get_unchecked(cid) == 0 {
                            let w = clause_activity_t2.get_unchecked_mut(cid);
                            *w = w.saturating_add(1);
                            bumped_t2 += 1;
                            if bumped_t2 >= 128 {
                                break;
                            }
                        }
                    }
                } else if progress_ratio > progress_threshold {
                    current_prob = base_prob;
                } else {
                    current_prob = current_prob * 0.8 + base_prob * 0.2;
                }
                last_check_residual = residual.len();

                let decay_span_t2 = 1024usize.min(m);
                if decay_span_t2 > 0 {
                    let start_t2 = rng.gen::<usize>() % m;
                    let mut t2 = 0usize;
                    while t2 < decay_span_t2 {
                        let idx = (start_t2 + t2) % m;
                        let ww = *clause_activity_t2.get_unchecked(idx);
                        *clause_activity_t2.get_unchecked_mut(idx) = (ww >> 1).max(1);
                        t2 += 1;
                    }
                }

                if residual.len() < best_global {
                    best_global = residual.len();
                    last_best_round = rounds;
                    best_global_vars = variables.clone();
                } else if restarts_done < max_restarts
                    && rounds.wrapping_sub(last_best_round) >= restart_after
                {
                    restarts_done = restarts_done.wrapping_add(1);
                    let flip_one_in = if restarts_done == 1 {
                        23
                    } else if restarts_done == 2 {
                        15
                    } else {
                        12
                    };
                    init_from_pref(&mut rng, &pref, flip_one_in, &mut variables);
                    recompute_state(&clauses, &variables, &mut num_good, &mut residual);

                    current_prob = (base_prob + 0.03 * (restarts_done as f64)).min(0.86);
                    last_check_residual = residual.len();
                    best_global = residual.len();
                    last_best_round = rounds;

                    last_flip = usize::MAX;
                    best_endgame = usize::MAX;
                    endgame_stuck = 0;
                    bf_tries = 0;

                    stamp = 1;
                    var_stamp = 1;
                    clause_marks.fill(0);
                    var_marks.fill(0);
                    clause_activity_t2.fill(1);

                    if residual.is_empty() {
                        break;
                    }
                } else if best_global <= 3 && rounds.wrapping_sub(last_best_round) >= 10_000_000 {
                    variables = best_global_vars.clone();
                    let num_flips = (n / 50).max(15).min(150);
                    for _ in 0..num_flips {
                        let v = rng.gen::<usize>() % n;
                        variables[v] = !variables[v];
                    }
                    
                    recompute_state(&clauses, &variables, &mut num_good, &mut residual);
                    last_best_round = rounds;
                    current_prob = base_prob * 0.95;
                    
                    if residual.len() < best_global {
                        best_global = residual.len();
                        best_global_vars = variables.clone();
                    }
                    
                    if residual.is_empty() {
                        break;
                    }
                }
            }

            if rounds > 0 && rounds % 15_000_000 == 0 && best_global < 10 {
                let stagnant_for = rounds.wrapping_sub(last_best_round);
                if stagnant_for > 10_000_000 {
                    variables = best_global_vars.clone();
                    recompute_state(&clauses, &variables, &mut num_good, &mut residual);
                    
                    var_stamp = var_stamp.wrapping_add(1);
                    if var_stamp == 0 {
                        var_marks.fill(0);
                        var_stamp = 1;
                    }
                    
                    let mut core_vars: Vec<usize> = Vec::new();
                    
                    for &cid in residual.iter() {
                        for &lit in clauses.get_unchecked(cid).iter() {
                            let v = (lit.abs() as usize) - 1;
                            if *var_marks.get_unchecked(v) != var_stamp {
                                *var_marks.get_unchecked_mut(v) = var_stamp;
                                core_vars.push(v);
                            }
                        }
                    }
                    
                    for cid in 0..m.min(5000) {
                        if *num_good.get_unchecked(cid) == 1 {
                            for &lit in clauses.get_unchecked(cid).iter() {
                                let v = (lit.abs() as usize) - 1;
                                if *var_marks.get_unchecked(v) != var_stamp {
                                    *var_marks.get_unchecked_mut(v) = var_stamp;
                                    core_vars.push(v);
                                    if core_vars.len() >= 25 {
                                        break;
                                    }
                                }
                            }
                            if core_vars.len() >= 25 {
                                break;
                            }
                        }
                    }
                    
                    let saved_vars = variables.clone();
                    
                    if core_vars.len() <= 22 {
                        let lim = 1u64 << (core_vars.len() as u64);
                        let mut best_found = best_global;
                        let mut best_mask = 0u64;
                        
                        for mask in 0..lim {
                            for (idx, &v) in core_vars.iter().enumerate() {
                                if ((mask >> idx) & 1) != 0 {
                                    variables[v] = !saved_vars[v];
                                } else {
                                    variables[v] = saved_vars[v];
                                }
                            }
                            
                            let mut unsat_count = 0;
                            for c in clauses.iter() {
                                let mut sat = false;
                                for &l in c.iter() {
                                    let v = (l.abs() as usize) - 1;
                                    if (l > 0 && variables[v]) || (l < 0 && !variables[v]) {
                                        sat = true;
                                        break;
                                    }
                                }
                                if !sat {
                                    unsat_count += 1;
                                    if unsat_count >= best_found {
                                        break;
                                    }
                                }
                            }
                            
                            if unsat_count < best_found {
                                best_found = unsat_count;
                                best_mask = mask;
                                if best_found == 0 {
                                    break;
                                }
                            }
                        }
                        
                        if best_found < best_global {
                            for (idx, &v) in core_vars.iter().enumerate() {
                                if ((best_mask >> idx) & 1) != 0 {
                                    variables[v] = !saved_vars[v];
                                } else {
                                    variables[v] = saved_vars[v];
                                }
                            }
                            recompute_state(&clauses, &variables, &mut num_good, &mut residual);
                            best_global = best_found;
                            best_global_vars = variables.clone();
                            last_best_round = rounds;
                            
                            if residual.is_empty() {
                                break;
                            }
                        } else {
                            variables = saved_vars;
                            recompute_state(&clauses, &variables, &mut num_good, &mut residual);
                        }
                    } else if core_vars.len() <= 30 {
                        let mut best_found = best_global;
                        let mut best_vars = saved_vars.clone();
                        let max_probes = 10000;
                        
                        for _probe in 0..max_probes {
                            let k = 1 + (rng.gen::<usize>() % 10);
                            
                            variables = saved_vars.clone();
                            
                            for _ in 0..k {
                                let idx = rng.gen::<usize>() % core_vars.len();
                                let v = core_vars[idx];
                                variables[v] = !variables[v];
                            }
                            
                            let mut unsat_count = 0;
                            for c in clauses.iter() {
                                let mut sat = false;
                                for &l in c.iter() {
                                    let v = (l.abs() as usize) - 1;
                                    if (l > 0 && variables[v]) || (l < 0 && !variables[v]) {
                                        sat = true;
                                        break;
                                    }
                                }
                                if !sat {
                                    unsat_count += 1;
                                    if unsat_count >= best_found {
                                        break;
                                    }
                                }
                            }
                            
                            if unsat_count < best_found {
                                best_found = unsat_count;
                                best_vars = variables.clone();
                                if best_found == 0 {
                                    break;
                                }
                            }
                        }
                        
                        if best_found < best_global {
                            variables = best_vars;
                            recompute_state(&clauses, &variables, &mut num_good, &mut residual);
                            best_global = best_found;
                            best_global_vars = variables.clone();
                            last_best_round = rounds;
                            
                            if residual.is_empty() {
                                break;
                            }
                        } else {
                            variables = best_vars;
                            recompute_state(&clauses, &variables, &mut num_good, &mut residual);
                        }
                    } else {
                        variables = saved_vars;
                    }
                }
            }

            if rounds % endgame_interval == 0 && residual.len() <= endgame_threshold {
                residual.clear();
                for cid in 0..m {
                    if *num_good.get_unchecked(cid) == 0 {
                        residual.push(cid);
                    }
                }
                if residual.is_empty() {
                    break;
                }

                if residual.len() <= 18 {
                    let cur_unsat = residual.len();
                    if cur_unsat < best_endgame {
                        best_endgame = cur_unsat;
                        endgame_stuck = 0;
                    } else {
                        endgame_stuck = endgame_stuck.saturating_add(1);
                    }

                    if endgame_stuck >= 3 && cur_unsat > 0 {
                        var_stamp = var_stamp.wrapping_add(1);
                        if var_stamp == 0 {
                            var_marks.fill(0);
                            var_stamp = 1;
                        }

                        let mut cand: Vec<usize> = Vec::with_capacity(cur_unsat * 3);
                        for &cid in residual.iter() {
                            for &lit in clauses.get_unchecked(cid).iter() {
                                let v = (lit.abs() as usize) - 1;
                                if *var_marks.get_unchecked(v) != var_stamp {
                                    *var_marks.get_unchecked_mut(v) = var_stamp;
                                    *var_cnt.get_unchecked_mut(v) = 1;
                                    cand.push(v);
                                } else {
                                    let c = var_cnt.get_unchecked_mut(v);
                                    *c = c.wrapping_add(1);
                                }
                            }
                        }

                        if cand.len() > 48 {
                            for i in 0..cand.len() {
                                let j = i + (rng.gen::<usize>() % (cand.len() - i));
                                cand.swap(i, j);
                            }
                            cand.truncate(48);
                        }

                        let mut did = false;

                        if !cand.is_empty() {
                            let cur_unsat_i = cur_unsat as i32;
                            let mut best_new = cur_unsat_i;
                            let mut best_v = cand[0];

                            let mut affected: Vec<usize> = Vec::new();
                            for &v in cand.iter() {
                                stamp = stamp.wrapping_add(1);
                                if stamp == 0 {
                                    clause_marks.fill(0);
                                    stamp = 1;
                                }

                                affected.clear();
                                affected.reserve(
                                    p_clauses.get_unchecked(v).len() + n_clauses.get_unchecked(v).len(),
                                );

                                for &ccid in p_clauses.get_unchecked(v).iter() {
                                    if *clause_marks.get_unchecked(ccid) != stamp {
                                        *clause_marks.get_unchecked_mut(ccid) = stamp;
                                        affected.push(ccid);
                                    }
                                }
                                for &ccid in n_clauses.get_unchecked(v).iter() {
                                    if *clause_marks.get_unchecked(ccid) != stamp {
                                        *clause_marks.get_unchecked_mut(ccid) = stamp;
                                        affected.push(ccid);
                                    }
                                }

                                let mut new_unsat = cur_unsat_i;
                                for &ccid in affected.iter() {
                                    let old_sat = *num_good.get_unchecked(ccid) != 0;
                                    let new_sat = clause_sat_with_one_flip(
                                        clauses.get_unchecked(ccid),
                                        &variables,
                                        v,
                                    );
                                    if old_sat && !new_sat {
                                        new_unsat += 1;
                                    } else if !old_sat && new_sat {
                                        new_unsat -= 1;
                                    }
                                    if new_unsat > best_new {
                                        break;
                                    }
                                }

                                if new_unsat < best_new {
                                    best_new = new_unsat;
                                    best_v = v;
                                    if best_new == 0 {
                                        break;
                                    }
                                }
                            }

                            if best_new < cur_unsat_i {
                                apply_flip(
                                    best_v,
                                    &mut variables,
                                    &mut num_good,
                                    &mut residual,
                                    &p_clauses,
                                    &n_clauses,
                                );
                                did = true;
                            }
                        }

                        if !did && cand.len() >= 2 {
                            let cur_unsat_i = cur_unsat as i32;
                            let mut best_new = cur_unsat_i;
                            let mut best_a = cand[0];
                            let mut best_b = cand[1];

                            let mut affected: Vec<usize> = Vec::new();

                            for x in 0..cand.len() {
                                let a = cand[x];
                                for y in (x + 1)..cand.len() {
                                    let b = cand[y];

                                    stamp = stamp.wrapping_add(1);
                                    if stamp == 0 {
                                        clause_marks.fill(0);
                                        stamp = 1;
                                    }

                                    affected.clear();
                                    affected.reserve(
                                        p_clauses.get_unchecked(a).len()
                                            + n_clauses.get_unchecked(a).len()
                                            + p_clauses.get_unchecked(b).len()
                                            + n_clauses.get_unchecked(b).len(),
                                    );

                                    for &ccid in p_clauses.get_unchecked(a).iter() {
                                        if *clause_marks.get_unchecked(ccid) != stamp {
                                            *clause_marks.get_unchecked_mut(ccid) = stamp;
                                            affected.push(ccid);
                                        }
                                    }
                                    for &ccid in n_clauses.get_unchecked(a).iter() {
                                        if *clause_marks.get_unchecked(ccid) != stamp {
                                            *clause_marks.get_unchecked_mut(ccid) = stamp;
                                            affected.push(ccid);
                                        }
                                    }
                                    for &ccid in p_clauses.get_unchecked(b).iter() {
                                        if *clause_marks.get_unchecked(ccid) != stamp {
                                            *clause_marks.get_unchecked_mut(ccid) = stamp;
                                            affected.push(ccid);
                                        }
                                    }
                                    for &ccid in n_clauses.get_unchecked(b).iter() {
                                        if *clause_marks.get_unchecked(ccid) != stamp {
                                            *clause_marks.get_unchecked_mut(ccid) = stamp;
                                            affected.push(ccid);
                                        }
                                    }

                                    let mut new_unsat = cur_unsat_i;
                                    for &ccid in affected.iter() {
                                        let old_sat = *num_good.get_unchecked(ccid) != 0;
                                        let new_sat = clause_sat_with_two_flips(
                                            clauses.get_unchecked(ccid),
                                            &variables,
                                            a,
                                            b,
                                        );
                                        if old_sat && !new_sat {
                                            new_unsat += 1;
                                        } else if !old_sat && new_sat {
                                            new_unsat -= 1;
                                        }
                                    }

                                    if new_unsat < best_new
                                        || (new_unsat == best_new && (rng.gen::<u8>() & 1) == 0)
                                    {
                                        best_new = new_unsat;
                                        best_a = a;
                                        best_b = b;
                                        if best_new == 0 {
                                            break;
                                        }
                                    }
                                }
                                if best_new == 0 {
                                    break;
                                }
                            }

                            let accept_pair = best_new < cur_unsat_i
                                || (endgame_stuck >= 6 && best_new == cur_unsat_i)
                                || (endgame_stuck >= 10 && best_new == cur_unsat_i + 1);

                            if accept_pair {
                                apply_flip(
                                    best_a,
                                    &mut variables,
                                    &mut num_good,
                                    &mut residual,
                                    &p_clauses,
                                    &n_clauses,
                                );
                                apply_flip(
                                    best_b,
                                    &mut variables,
                                    &mut num_good,
                                    &mut residual,
                                    &p_clauses,
                                    &n_clauses,
                                );
                                did = true;
                            }
                        }

                        if !did
                            && cur_unsat <= 8
                            && endgame_stuck >= 6
                            && bf_tries < 2
                            && !cand.is_empty()
                        {
                            bf_tries = bf_tries.wrapping_add(1);

                            let mut cand_bf = cand.clone();
                            let k = cand_bf.len().min((12 + cur_unsat) as usize).min(16);
                            if k > 0 && cand_bf.len() > 1 {
                                let mut i = 0usize;
                                while i < k {
                                    let mut best_i = i;
                                    let mut best_c = *var_cnt.get_unchecked(cand_bf[i]);
                                    let mut j = i + 1;
                                    while j < cand_bf.len() {
                                        let c = *var_cnt.get_unchecked(cand_bf[j]);
                                        if c > best_c {
                                            best_c = c;
                                            best_i = j;
                                        }
                                        j += 1;
                                    }
                                    cand_bf.swap(i, best_i);
                                    i += 1;
                                }
                                cand_bf.truncate(k);
                            } else {
                                cand_bf.truncate(k);
                            }

                            let k = cand_bf.len();
                            if k > 0 && k <= 16 {
                                stamp = stamp.wrapping_add(1);
                                if stamp == 0 {
                                    clause_marks.fill(0);
                                    stamp = 1;
                                }

                                let mut affected_bf: Vec<usize> = Vec::new();
                                for &v in cand_bf.iter() {
                                    for &ccid in p_clauses.get_unchecked(v).iter() {
                                        if *clause_marks.get_unchecked(ccid) != stamp {
                                            *clause_marks.get_unchecked_mut(ccid) = stamp;
                                            affected_bf.push(ccid);
                                        }
                                    }
                                    for &ccid in n_clauses.get_unchecked(v).iter() {
                                        if *clause_marks.get_unchecked(ccid) != stamp {
                                            *clause_marks.get_unchecked_mut(ccid) = stamp;
                                            affected_bf.push(ccid);
                                        }
                                    }
                                }

                                for (idx, &v) in cand_bf.iter().enumerate() {
                                    *pos.get_unchecked_mut(v) = idx as i16;
                                }

                                let lim = 1u32 << (k as u32);
                                let mut found = u32::MAX;
                                let mut mask = 0u32;
                                while mask < lim {
                                    let mut ok = true;
                                    for &ccid in affected_bf.iter() {
                                        if !clause_sat_with_mask(
                                            clauses.get_unchecked(ccid),
                                            &variables,
                                            &pos,
                                            mask,
                                        ) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    if ok {
                                        found = mask;
                                        break;
                                    }
                                    mask = mask.wrapping_add(1);
                                }

                                for &v in cand_bf.iter() {
                                    *pos.get_unchecked_mut(v) = -1;
                                }

                                if found != u32::MAX && found != 0 {
                                    for (idx, &v) in cand_bf.iter().enumerate() {
                                        if ((found >> (idx as u32)) & 1) != 0 {
                                            apply_flip(
                                                v,
                                                &mut variables,
                                                &mut num_good,
                                                &mut residual,
                                                &p_clauses,
                                                &n_clauses,
                                            );
                                        }
                                    }
                                    did = true;
                                }
                            }
                        }

                        if !did && cur_unsat <= 10 && endgame_stuck >= 10 && !cand.is_empty() {
                            let k = 2 + (rng.gen::<usize>() & 1);
                            for _ in 0..k {
                                let v = cand[rng.gen::<usize>() % cand.len()];
                                apply_flip(
                                    v,
                                    &mut variables,
                                    &mut num_good,
                                    &mut residual,
                                    &p_clauses,
                                    &n_clauses,
                                );
                            }
                            did = true;
                        }

                        if did {
                            residual.clear();
                            for cid in 0..m {
                                if *num_good.get_unchecked(cid) == 0 {
                                    residual.push(cid);
                                }
                            }
                            if residual.is_empty() {
                                break;
                            }
                            best_endgame = best_endgame.min(residual.len());
                            endgame_stuck = 0;
                        }
                    }
                }
            }

            if residual.is_empty() {
                break;
            }

            let rand_val = rng.gen::<usize>();
            let mut ci = *residual.last().unwrap_unchecked();
            while !residual.is_empty() {
                let id = rand_val % residual.len();
                ci = *residual.get_unchecked(id);
                if *num_good.get_unchecked(ci) > 0 {
                    residual.swap_remove(id);
                } else {
                    break;
                }
            }
            if residual.is_empty() {
                break;
            }

            if residual.len() > 1 {
                let mut best_ci = ci;
                let mut best_w = *clause_activity_t2.get_unchecked(ci);
                let mut t = 0u8;
                while t < 2 {
                    let idx = rng.gen::<usize>() % residual.len();
                    let cand_ci = *residual.get_unchecked(idx);
                    if *num_good.get_unchecked(cand_ci) == 0 {
                        let w = *clause_activity_t2.get_unchecked(cand_ci);
                        if w > best_w {
                            best_w = w;
                            best_ci = cand_ci;
                        }
                    }
                    t = t.wrapping_add(1);
                }
                ci = best_ci;
            }

            *clause_activity_t2.get_unchecked_mut(ci) =
                clause_activity_t2.get_unchecked(ci).saturating_add(1);

            let c = clauses.get_unchecked_mut(ci);
            if c.len() > 1 {
                let random_index = rand_val % c.len();
                c.swap(0, random_index);
            }

            let mut zero_found = None;
            'outer: for &l in c.iter() {
                let v = (l.abs() as usize) - 1;
                let clauses_to_check = if *variables.get_unchecked(v) {
                    p_clauses.get_unchecked(v)
                } else {
                    n_clauses.get_unchecked(v)
                };

                for &ccid in clauses_to_check.iter() {
                    if *num_good.get_unchecked(ccid) == 1 {
                        continue 'outer;
                    }
                }
                zero_found = Some(v);
                break;
            }

            let thr = (current_prob * (usize::MAX as f64)) as usize;

            let v = if let Some(v) = zero_found {
                v
            } else if rand_val < thr {
                (c[0].abs() as usize) - 1
            } else {
                let mut best_weight = usize::MAX;
                let mut second_weight = usize::MAX;
                let mut best_var = (c[0].abs() as usize) - 1;
                let mut second_var = best_var;

                for &l in c.iter() {
                    let v = (l.abs() as usize) - 1;
                    let clauses_to_check = if *variables.get_unchecked(v) {
                        p_clauses.get_unchecked(v)
                    } else {
                        n_clauses.get_unchecked(v)
                    };

                    let mut sad = 0usize;
                    for &ccid in clauses_to_check.iter() {
                        if *num_good.get_unchecked(ccid) == 1 {
                            sad += 1;
                        }
                    }

                    if sad < best_weight {
                        second_weight = best_weight;
                        second_var = best_var;
                        best_weight = sad;
                        best_var = v;
                    } else if v != best_var && sad < second_weight {
                        second_weight = sad;
                        second_var = v;
                    }
                }

                if best_var == last_flip
                    && second_var != best_var
                    && second_weight != usize::MAX
                    && rng.gen::<usize>() < thr
                {
                    second_var
                } else {
                    best_var
                }
            };

            apply_flip(
                v,
                &mut variables,
                &mut num_good,
                &mut residual,
                &p_clauses,
                &n_clauses,
            );
            last_flip = v;
            rounds += 1;
        }
    }

    for v in 0..n {
        if p_single[v] {
            variables[v] = true;
        } else if n_single[v] {
            variables[v] = false;
        }
    }

    if verify_all(&challenge.clauses, &variables) {
        let _ = save_solution(&Solution { variables });
    }

    Ok(())
}
