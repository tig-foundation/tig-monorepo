use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::convert::TryInto;
use tig_challenges::satisfiability::*;
use crate::{seeded_hasher, HashSet};
use super::helpers::*;

pub fn solve_track_3_impl(
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
    let mut current_restart_interval = restart_after;
    let mut restart_best_at_start = residual.len();

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
                    if restarts_done > 0 && residual.len() < restart_best_at_start {
                        let rounds_since_restart = rounds.wrapping_sub(last_best_round.saturating_sub(check_interval));
                        if rounds_since_restart < current_restart_interval / 2 {
                            let extended = (current_restart_interval * 3 / 2).min(restart_after * 4);
                            if extended > current_restart_interval {
                                current_restart_interval = extended;
                            }
                        }
                    }
                } else if restarts_done < max_restarts
                    && best_global > 80
                    && rounds.wrapping_sub(last_best_round) >= current_restart_interval
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
                    restart_best_at_start = residual.len();

                    let next_interval = (current_restart_interval * 3 / 2).min(restart_after * 4);
                    current_restart_interval = next_interval;

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

                        if !did && cur_unsat > 0 {
                            let mut two_sat_vars: Vec<usize> = Vec::new();
                            let local_stamp = var_stamp.wrapping_add(1);
                            let mut local_var_map: Vec<i32> = vec![-1i32; n];
                            let mut binary_clauses_2sat: Vec<(i32, i32)> = Vec::new();
                            let mut has_non_binary = false;

                            for &cid in residual.iter() {
                                let cl = clauses.get_unchecked(cid);
                                if cl.len() == 2 {
                                    let l0 = cl[0];
                                    let l1 = cl[1];
                                    let v0 = (l0.abs() as usize) - 1;
                                    let v1 = (l1.abs() as usize) - 1;
                                    if local_var_map[v0] < 0 {
                                        let idx = two_sat_vars.len() as i32;
                                        local_var_map[v0] = idx;
                                        two_sat_vars.push(v0);
                                    }
                                    if local_var_map[v1] < 0 {
                                        let idx = two_sat_vars.len() as i32;
                                        local_var_map[v1] = idx;
                                        two_sat_vars.push(v1);
                                    }
                                    binary_clauses_2sat.push((l0, l1));
                                } else if cl.len() >= 3 {
                                    has_non_binary = true;
                                    for &lit in cl.iter() {
                                        let v = (lit.abs() as usize) - 1;
                                        if local_var_map[v] < 0 {
                                            let idx = two_sat_vars.len() as i32;
                                            local_var_map[v] = idx;
                                            two_sat_vars.push(v);
                                        }
                                    }
                                }
                            }

                            let nv2 = two_sat_vars.len();
                            let _ = has_non_binary;
                            let _ = local_stamp;

                            if nv2 > 0 && nv2 <= 60 && !binary_clauses_2sat.is_empty() {
                                let lit_to_node = |lit: i32| -> usize {
                                    let v = (lit.abs() as usize) - 1;
                                    let idx = local_var_map[v] as usize;
                                    if lit > 0 { idx } else { idx + nv2 }
                                };
                                let neg_node = |node: usize| -> usize {
                                    if node < nv2 { node + nv2 } else { node - nv2 }
                                };

                                let num_nodes = 2 * nv2;
                                let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

                                for &(l0, l1) in binary_clauses_2sat.iter() {
                                    let from0 = neg_node(lit_to_node(l0));
                                    let to0 = lit_to_node(l1);
                                    adj[from0].push(to0);
                                    let from1 = neg_node(lit_to_node(l1));
                                    let to1 = lit_to_node(l0);
                                    adj[from1].push(to1);
                                }

                                let mut index_counter = 0u32;
                                let mut stack: Vec<usize> = Vec::with_capacity(num_nodes);
                                let mut on_stack: Vec<bool> = vec![false; num_nodes];
                                let mut index: Vec<i32> = vec![-1i32; num_nodes];
                                let mut lowlink: Vec<u32> = vec![0u32; num_nodes];
                                let mut scc_id: Vec<i32> = vec![-1i32; num_nodes];
                                let mut num_sccs: i32 = 0;

                                struct Frame {
                                    v: usize,
                                    ei: usize,
                                }

                                for start in 0..num_nodes {
                                    if index[start] >= 0 {
                                        continue;
                                    }
                                    let mut call_stack: Vec<Frame> = Vec::new();
                                    call_stack.push(Frame { v: start, ei: 0 });
                                    index[start] = index_counter as i32;
                                    lowlink[start] = index_counter;
                                    index_counter += 1;
                                    stack.push(start);
                                    on_stack[start] = true;

                                    'dfs: loop {
                                        let frame = call_stack.last_mut().unwrap();
                                        let v = frame.v;
                                        if frame.ei < adj[v].len() {
                                            let w = adj[v][frame.ei];
                                            frame.ei += 1;
                                            if index[w] < 0 {
                                                index[w] = index_counter as i32;
                                                lowlink[w] = index_counter;
                                                index_counter += 1;
                                                stack.push(w);
                                                on_stack[w] = true;
                                                call_stack.push(Frame { v: w, ei: 0 });
                                            } else if on_stack[w] {
                                                let lw = lowlink[w];
                                                let frame2 = call_stack.last_mut().unwrap();
                                                if lw < lowlink[frame2.v] {
                                                    lowlink[frame2.v] = lw;
                                                }
                                            }
                                        } else {
                                            let v_ll = lowlink[v];
                                            call_stack.pop();
                                            if let Some(parent_frame) = call_stack.last_mut() {
                                                if v_ll < lowlink[parent_frame.v] {
                                                    lowlink[parent_frame.v] = v_ll;
                                                }
                                            }
                                            if lowlink[v] == index[v] as u32 {
                                                loop {
                                                    let w = *stack.last().unwrap();
                                                    stack.pop();
                                                    on_stack[w] = false;
                                                    scc_id[w] = num_sccs;
                                                    if w == v {
                                                        break;
                                                    }
                                                }
                                                num_sccs += 1;
                                            }
                                            if call_stack.is_empty() {
                                                break 'dfs;
                                            }
                                        }
                                    }
                                }

                                let mut sat_2sat = true;
                                for i in 0..nv2 {
                                    if scc_id[i] == scc_id[i + nv2] {
                                        sat_2sat = false;
                                        break;
                                    }
                                }

                                if sat_2sat {
                                    let mut flips_2sat: Vec<usize> = Vec::new();
                                    for i in 0..nv2 {
                                        let v = two_sat_vars[i];
                                        let pos_scc = scc_id[i];
                                        let neg_scc = scc_id[i + nv2];
                                        let assigned_true = pos_scc > neg_scc;
                                        let cur_val = *variables.get_unchecked(v);
                                        if assigned_true != cur_val {
                                            flips_2sat.push(v);
                                        }
                                    }

                                    if !flips_2sat.is_empty() {
                                        var_stamp = var_stamp.wrapping_add(1);
                                        if var_stamp == 0 {
                                            var_marks.fill(0);
                                            var_stamp = 1;
                                        }
                                        for &fv in flips_2sat.iter() {
                                            *var_marks.get_unchecked_mut(fv) = var_stamp;
                                        }

                                        stamp = stamp.wrapping_add(1);
                                        if stamp == 0 {
                                            clause_marks.fill(0);
                                            stamp = 1;
                                        }
                                        let mut aff2sat: Vec<usize> = Vec::new();
                                        for &fv in flips_2sat.iter() {
                                            for &ccid in p_clauses.get_unchecked(fv).iter() {
                                                if *clause_marks.get_unchecked(ccid) != stamp {
                                                    *clause_marks.get_unchecked_mut(ccid) = stamp;
                                                    aff2sat.push(ccid);
                                                }
                                            }
                                            for &ccid in n_clauses.get_unchecked(fv).iter() {
                                                if *clause_marks.get_unchecked(ccid) != stamp {
                                                    *clause_marks.get_unchecked_mut(ccid) = stamp;
                                                    aff2sat.push(ccid);
                                                }
                                            }
                                        }

                                        let cur_unsat_i = cur_unsat as i32;
                                        let mut new_unsat_2sat = cur_unsat_i;
                                        for &ccid in aff2sat.iter() {
                                            let old_sat = *num_good.get_unchecked(ccid) != 0;
                                            let cl2 = clauses.get_unchecked(ccid);
                                            let mut new_sat = false;
                                            for &lit in cl2.iter() {
                                                let lv = (lit.abs() as usize) - 1;
                                                let flipped = *var_marks.get_unchecked(lv) == var_stamp;
                                                let val = *variables.get_unchecked(lv) ^ flipped;
                                                let s = if lit > 0 { val } else { !val };
                                                if s { new_sat = true; break; }
                                            }
                                            if old_sat && !new_sat {
                                                new_unsat_2sat += 1;
                                            } else if !old_sat && new_sat {
                                                new_unsat_2sat -= 1;
                                            }
                                        }

                                        if new_unsat_2sat < cur_unsat_i {
                                            for fv in flips_2sat {
                                                apply_flip(
                                                    fv,
                                                    &mut variables,
                                                    &mut num_good,
                                                    &mut residual,
                                                    &p_clauses,
                                                    &n_clauses,
                                                );
                                            }
                                            did = true;
                                        }
                                    }
                                }
                            }

                            for &v in two_sat_vars.iter() {
                                local_var_map[v] = -1;
                            }
                        }

                        if !did && cur_unsat > 0 {
                            let mut two_sat_vars: Vec<usize> = Vec::new();
                            let mut local_var_map: Vec<i32> = vec![-1i32; n];
                            let mut binary_clauses_2sat: Vec<(i32, i32)> = Vec::new();

                            for &cid in residual.iter() {
                                let cl = clauses.get_unchecked(cid);
                                for &lit in cl.iter() {
                                    let v = (lit.abs() as usize) - 1;
                                    if local_var_map[v] < 0 {
                                        let idx = two_sat_vars.len() as i32;
                                        local_var_map[v] = idx;
                                        two_sat_vars.push(v);
                                    }
                                }
                                if cl.len() == 2 {
                                    binary_clauses_2sat.push((cl[0], cl[1]));
                                }
                            }

                            let nv2 = two_sat_vars.len();

                            if nv2 > 0 && nv2 <= 60 && !binary_clauses_2sat.is_empty() {
                                let lit_to_node = |lit: i32| -> usize {
                                    let v = (lit.abs() as usize) - 1;
                                    let idx = local_var_map[v] as usize;
                                    if lit > 0 { idx } else { idx + nv2 }
                                };
                                let neg_node = |node: usize| -> usize {
                                    if node < nv2 { node + nv2 } else { node - nv2 }
                                };

                                let num_nodes = 2 * nv2;
                                let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

                                for &(l0, l1) in binary_clauses_2sat.iter() {
                                    let from0 = neg_node(lit_to_node(l0));
                                    let to0 = lit_to_node(l1);
                                    adj[from0].push(to0);
                                    let from1 = neg_node(lit_to_node(l1));
                                    let to1 = lit_to_node(l0);
                                    adj[from1].push(to1);
                                }

                                let mut index_counter = 0u32;
                                let mut tarjan_stack: Vec<usize> = Vec::with_capacity(num_nodes);
                                let mut on_stack: Vec<bool> = vec![false; num_nodes];
                                let mut t_index: Vec<i32> = vec![-1i32; num_nodes];
                                let mut lowlink: Vec<u32> = vec![0u32; num_nodes];
                                let mut scc_id: Vec<i32> = vec![-1i32; num_nodes];
                                let mut num_sccs: i32 = 0;

                                struct TarjanFrame { v: usize, ei: usize }

                                for start in 0..num_nodes {
                                    if t_index[start] >= 0 { continue; }
                                    let mut call_stack: Vec<TarjanFrame> = Vec::new();
                                    call_stack.push(TarjanFrame { v: start, ei: 0 });
                                    t_index[start] = index_counter as i32;
                                    lowlink[start] = index_counter;
                                    index_counter += 1;
                                    tarjan_stack.push(start);
                                    on_stack[start] = true;

                                    loop {
                                        let frame = call_stack.last_mut().unwrap();
                                        let v = frame.v;
                                        if frame.ei < adj[v].len() {
                                            let w = adj[v][frame.ei];
                                            frame.ei += 1;
                                            if t_index[w] < 0 {
                                                t_index[w] = index_counter as i32;
                                                lowlink[w] = index_counter;
                                                index_counter += 1;
                                                tarjan_stack.push(w);
                                                on_stack[w] = true;
                                                call_stack.push(TarjanFrame { v: w, ei: 0 });
                                            } else if on_stack[w] {
                                                let lw = lowlink[w];
                                                let pf = call_stack.last_mut().unwrap();
                                                if lw < lowlink[pf.v] { lowlink[pf.v] = lw; }
                                            }
                                        } else {
                                            let v_ll = lowlink[v];
                                            call_stack.pop();
                                            if let Some(pf) = call_stack.last_mut() {
                                                if v_ll < lowlink[pf.v] { lowlink[pf.v] = v_ll; }
                                            }
                                            if lowlink[v] == t_index[v] as u32 {
                                                loop {
                                                    let w = *tarjan_stack.last().unwrap();
                                                    tarjan_stack.pop();
                                                    on_stack[w] = false;
                                                    scc_id[w] = num_sccs;
                                                    if w == v { break; }
                                                }
                                                num_sccs += 1;
                                            }
                                            if call_stack.is_empty() { break; }
                                        }
                                    }
                                }

                                let mut sat_2sat = true;
                                for i in 0..nv2 {
                                    if scc_id[i] == scc_id[i + nv2] {
                                        sat_2sat = false;
                                        break;
                                    }
                                }

                                if sat_2sat {
                                    let mut flips_2sat: Vec<usize> = Vec::new();
                                    for i in 0..nv2 {
                                        let v = two_sat_vars[i];
                                        let assigned_true = scc_id[i] > scc_id[i + nv2];
                                        let cur_val = *variables.get_unchecked(v);
                                        if assigned_true != cur_val {
                                            flips_2sat.push(v);
                                        }
                                    }

                                    if !flips_2sat.is_empty() {
                                        var_stamp = var_stamp.wrapping_add(1);
                                        if var_stamp == 0 {
                                            var_marks.fill(0);
                                            var_stamp = 1;
                                        }
                                        for &fv in flips_2sat.iter() {
                                            *var_marks.get_unchecked_mut(fv) = var_stamp;
                                        }

                                        stamp = stamp.wrapping_add(1);
                                        if stamp == 0 {
                                            clause_marks.fill(0);
                                            stamp = 1;
                                        }
                                        let mut aff2sat: Vec<usize> = Vec::new();
                                        for &fv in flips_2sat.iter() {
                                            for &ccid in p_clauses.get_unchecked(fv).iter() {
                                                if *clause_marks.get_unchecked(ccid) != stamp {
                                                    *clause_marks.get_unchecked_mut(ccid) = stamp;
                                                    aff2sat.push(ccid);
                                                }
                                            }
                                            for &ccid in n_clauses.get_unchecked(fv).iter() {
                                                if *clause_marks.get_unchecked(ccid) != stamp {
                                                    *clause_marks.get_unchecked_mut(ccid) = stamp;
                                                    aff2sat.push(ccid);
                                                }
                                            }
                                        }

                                        let cur_unsat_i = cur_unsat as i32;
                                        let mut new_unsat_2sat = cur_unsat_i;
                                        for &ccid in aff2sat.iter() {
                                            let old_sat = *num_good.get_unchecked(ccid) != 0;
                                            let cl2 = clauses.get_unchecked(ccid);
                                            let mut new_sat = false;
                                            for &lit in cl2.iter() {
                                                let lv = (lit.abs() as usize) - 1;
                                                let flipped = *var_marks.get_unchecked(lv) == var_stamp;
                                                let val = *variables.get_unchecked(lv) ^ flipped;
                                                let s = if lit > 0 { val } else { !val };
                                                if s { new_sat = true; break; }
                                            }
                                            if old_sat && !new_sat {
                                                new_unsat_2sat += 1;
                                            } else if !old_sat && new_sat {
                                                new_unsat_2sat -= 1;
                                            }
                                        }

                                        if new_unsat_2sat < cur_unsat_i {
                                            for fv in flips_2sat {
                                                apply_flip(
                                                    fv,
                                                    &mut variables,
                                                    &mut num_good,
                                                    &mut residual,
                                                    &p_clauses,
                                                    &n_clauses,
                                                );
                                            }
                                            did = true;
                                        }
                                    }
                                }
                            }

                            for &v in two_sat_vars.iter() {
                                local_var_map[v] = -1;
                            }
                        }

                        if !did
                            && cur_unsat <= 15
                            && endgame_stuck >= 2
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

                                let mut max_c: u16 = 0;
                                for &v in cand_bf.iter() {
                                    let c = *var_cnt.get_unchecked(v) as u16;
                                    if c > max_c {
                                        max_c = c;
                                    }
                                }
                                let denom = max_c.wrapping_add(1).max(1);

                                let mut tries = 0u32;
                                while tries < lim {
                                    let mut mask = 0u32;
                                    for (idx, &v) in cand_bf.iter().enumerate() {
                                        let c = *var_cnt.get_unchecked(v) as u16;
                                        let r = rng.gen::<u16>() % denom;
                                        if r < c {
                                            mask |= 1u32 << (idx as u32);
                                        }
                                    }

                                    let mut step = 0usize;
                                    while step < k {
                                        let mut bad_cid = usize::MAX;
                                        for &ccid in affected_bf.iter() {
                                            if !clause_sat_with_mask(
                                                clauses.get_unchecked(ccid),
                                                &variables,
                                                &pos,
                                                mask,
                                            ) {
                                                bad_cid = ccid;
                                                break;
                                            }
                                        }
                                        if bad_cid == usize::MAX {
                                            found = mask;
                                            break;
                                        }

                                        let cl = clauses.get_unchecked(bad_cid);
                                        let mut best_idx: i16 = -1;
                                        let mut best_req: u8 = 0;
                                        let mut best_score: u8 = 0;
                                        let mut seen = 0u8;

                                        for &lit in cl.iter() {
                                            let v = (lit.abs() as usize) - 1;
                                            let p = *pos.get_unchecked(v);
                                            if p < 0 {
                                                continue;
                                            }
                                            let idx = p as usize;
                                            let base = *variables.get_unchecked(v) as u8;
                                            let req = if lit > 0 { base ^ 1 } else { base };
                                            let cur = ((mask >> (idx as u32)) & 1) as u8;
                                            if cur == req {
                                                continue;
                                            }
                                            let score = *var_cnt.get_unchecked(v);
                                            if score > best_score {
                                                best_score = score;
                                                best_idx = p;
                                                best_req = req;
                                                seen = 1;
                                            } else if score == best_score {
                                                seen = seen.wrapping_add(1);
                                                if seen != 0 && (rng.gen::<u8>() % seen) == 0 {
                                                    best_idx = p;
                                                    best_req = req;
                                                }
                                            }
                                        }

                                        if best_idx < 0 {
                                            break;
                                        }
                                        let bi = best_idx as u32;
                                        mask = (mask & !(1u32 << bi)) | ((best_req as u32) << bi);

                                        step += 1;
                                    }

                                    if found != u32::MAX {
                                        break;
                                    }

                                    tries = tries.wrapping_add(1);
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