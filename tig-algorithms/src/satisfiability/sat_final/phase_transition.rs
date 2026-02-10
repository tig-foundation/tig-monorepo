use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::convert::TryInto;
use tig_challenges::satisfiability::*;
use crate::{seeded_hasher, HashSet};
use super::helpers::*;

pub fn solve_phase_transition_impl(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    base_prob: f64,
    check_interval: usize,
    max_restarts: u8,
    max_fuel: f64,
    restart_after: usize,
    endgame_threshold: usize,
    use_portfolio_init: bool,
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

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); n];

    for c in &clauses {
        for &l in c.iter() {
            let var = (l.abs() as usize) - 1;
            if l > 0 {
                if p_clauses[var].capacity() == 0 {
                    p_clauses[var] = Vec::with_capacity(m / n.max(1) + 1);
                }
            } else if n_clauses[var].capacity() == 0 {
                n_clauses[var] = Vec::with_capacity(m / n.max(1) + 1);
            }
        }
    }

    for (ci, c) in clauses.iter().enumerate() {
        for &l in c.iter() {
            let var = (l.abs() as usize) - 1;
            if l > 0 {
                p_clauses[var].push(ci);
            } else {
                n_clauses[var].push(ci);
            }
        }
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
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();
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
    
    if use_portfolio_init {
        let mut best_vars = vec![false; n];
        let mut second_vars = vec![false; n];
        let mut best_u = usize::MAX;
        let mut second_u = usize::MAX;

        for &flip_one_in in [0u32, 23u32, 15u32, 12u32].iter() {
            init_from_pref(&mut rng, &pref, flip_one_in, &mut variables);
            let u = count_unsat(&clauses, &variables);
            if u < best_u {
                second_u = best_u;
                second_vars.clone_from(&best_vars);
                best_u = u;
                best_vars.clone_from(&variables);
            } else if u < second_u {
                second_u = u;
                second_vars.clone_from(&variables);
            }
        }

        let use_second = second_u != usize::MAX
            && second_u <= best_u.saturating_add(40)
            && (rng.gen::<u32>() % 6 == 0);

        if use_second {
            variables.clone_from(&second_vars);
        } else {
            variables.clone_from(&best_vars);
        }
    } else {
        init_from_pref(&mut rng, &pref, 0, &mut variables);
    }

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
                } else if progress_ratio > progress_threshold {
                    current_prob = base_prob;
                } else {
                    current_prob = current_prob * 0.8 + base_prob * 0.2;
                }
                last_check_residual = residual.len();

                if residual.len() < best_global {
                    best_global = residual.len();
                    last_best_round = rounds;
                } else if restarts_done < max_restarts
                    && best_global > 80
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

                    if residual.is_empty() {
                        break;
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
