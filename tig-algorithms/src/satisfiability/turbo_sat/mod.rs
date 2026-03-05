use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::convert::TryInto;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
use crate::{seeded_hasher, HashSet};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let density = (challenge.clauses.len() * 100 / challenge.num_variables) as f64 / 100.0;
    if density >= 4.22 {
        solve_high_density(challenge, save_solution, hyperparameters)
    } else {
        solve_low_density(challenge, save_solution, hyperparameters)
    }
}

// ---- Preprocessing: Unit propagation + tautology elimination ----
fn preprocess(challenge: &Challenge) -> (Vec<Vec<i32>>, Vec<bool>, Vec<bool>, bool) {
    let hasher = seeded_hasher(&challenge.seed);
    let mut clauses = challenge.clauses.clone();
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        let clause = &mut clauses[i];
        if clause.len() > 1 {
            let mut seen = HashSet::with_hasher(hasher.clone());
            let mut j = 0;
            let mut tautology = false;
            while j < clause.len() {
                let lit = clause[j];
                if seen.contains(&-lit) { tautology = true; break; }
                if !seen.insert(lit) { clause.swap_remove(j); } else { j += 1; }
            }
            if tautology { clauses.swap_remove(i); i += 1; continue; }
        }
    }

    let mut p_single = vec![false; challenge.num_variables];
    let mut n_single = vec![false; challenge.num_variables];
    let mut clauses_ = clauses;
    clauses = Vec::with_capacity(clauses_.len());
    let mut dead = false;

    while !dead {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len());
            let mut skip = false;
            for &l in c.iter() {
                let idx = (l.abs() - 1) as usize;
                if (p_single[idx] && l > 0) || (n_single[idx] && l < 0) { skip = true; break; }
                if p_single[idx] || n_single[idx] { done = false; continue; }
                c_.push(l);
            }
            if skip { done = false; continue; }
            match c_[..] {
                [l] => {
                    done = false;
                    if l > 0 {
                        if n_single[(l.abs() - 1) as usize] { dead = true; break; }
                        else { p_single[(l.abs() - 1) as usize] = true; }
                    } else {
                        if p_single[(l.abs() - 1) as usize] { dead = true; break; }
                        else { n_single[(l.abs() - 1) as usize] = true; }
                    }
                }
                [] => { dead = true; break; }
                _ => { clauses.push(c_); }
            }
        }
        if done { break; }
        else { clauses_ = clauses; clauses = Vec::with_capacity(clauses_.len()); }
    }

    (clauses, p_single, n_single, dead)
}

fn solve_high_density(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let (clauses, p_single, n_single, dead) = preprocess(challenge);
    if dead { return Ok(()); }

    let num_variables = challenge.num_variables;
    let num_clauses = clauses.len();

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 { p_clauses[var].push(i); } else { n_clauses[var].push(i); }
        }
    }

    let density = num_clauses as f64 / num_variables as f64;
    let avg_clause_size = clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64;
    let var_appearances: Vec<usize> = (0..num_variables)
        .map(|v| p_clauses[v].len() + n_clauses[v].len()).collect();

    let nad = 1.0;
    let random_threshold = if num_variables >= 30000 { 0.01 } else { 0.003 };

    // IMPROVEMENT 1: Better initial assignment with polarity scoring
    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();
        if num_n == 0 && num_p > 0 { variables[v] = true; continue; }
        if num_p == 0 && num_n > 0 { variables[v] = false; continue; }
        let vad = if num_n > 0 { num_p as f64 / num_n as f64 } else { nad + 1.0 };
        if vad <= nad {
            variables[v] = rng.gen_bool(random_threshold);
        } else {
            let prob = (num_p as f64 + 0.25) / ((num_p + num_n) as f64 + 1.2);
            variables[v] = rng.gen_bool(prob);
        }
    }

    let mut num_good_so_far: Vec<u8> = vec![0; num_clauses];
    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if (l > 0 && variables[var]) || (l < 0 && !variables[var]) {
                num_good_so_far[i] = num_good_so_far[i].saturating_add(1);
            }
        }
    }

    let mut residual_ = Vec::with_capacity(num_clauses);
    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 { residual_.push(i); }
    }
    if residual_.is_empty() {
        for v in 0..num_variables {
            if p_single[v] { variables[v] = true; }
            else if n_single[v] { variables[v] = false; }
        }
        let _ = save_solution(&Solution { variables });
        return Ok(());
    }

    // IMPROVEMENT 2: Adaptive probability with better convergence
    let base_prob = 0.52;
    let mut current_prob = base_prob;
    let large_problem_scale = ((num_variables as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = (base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;

    // IMPROVEMENT 3: Slightly higher adjustment for faster adaptation
    let max_random_prob = 0.92;  // raised from 0.9
    let prob_adjustment_factor = 0.028;  // slightly higher
    let smoothing_factor = 0.78;  // slightly more aggressive

    let mut last_check_residual = residual_.len();
    let mut var_age = vec![0u16; num_variables];

    let max_fuel = hyperparameters.as_ref()
        .and_then(|h| h.get("max_fuel_high")).and_then(|v| v.as_f64())
        .unwrap_or(10_000_000_000.0);
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if num_variables > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (num_variables as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_num_rounds = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
    let mut rounds = 0;

    unsafe {
        loop {
            if rounds >= max_num_rounds { return Ok(()); }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual_.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    let prob_adjustment = prob_adjustment_factor * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);
                } else if progress_ratio > progress_threshold {
                    current_prob = base_prob;
                } else {
                    current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                }
                last_check_residual = residual_.len();
            }

            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();
                let mut i = residual_.len() - 1;

                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = residual_[id];
                    if num_good_so_far[i] > 0 { residual_.swap_remove(id); }
                    else { break; }
                }
                if residual_.is_empty() { break; }

                let c = clauses.get_unchecked_mut(i);
                if c.len() > 1 {
                    let random_index = rand_val % c.len();
                    c.swap(0, random_index);
                }

                let mut zero_found = None;
                'outer: for &l in c.iter() {
                    let abs_l = l.abs() as usize - 1;
                    let clauses_to_check = if *variables.get_unchecked(abs_l) {
                        p_clauses.get_unchecked(abs_l)
                    } else {
                        n_clauses.get_unchecked(abs_l)
                    };
                    for &c in clauses_to_check {
                        if *num_good_so_far.get_unchecked(c) == 1 { continue 'outer; }
                    }
                    zero_found = Some(abs_l);
                    break;
                }

                // IMPROVEMENT 4: Better variable selection with age-weighted scoring
                let v = if let Some(abs_l) = zero_found {
                    abs_l
                } else if rng.gen::<f64>() < current_prob {
                    c[0].abs() as usize - 1
                } else {
                    let mut min_sad = usize::MAX;
                    let mut v_min_sad = c[0].abs() as usize - 1;
                    let mut min_weight = usize::MAX;

                    for &l in c.iter() {
                        let abs_l = l.abs() as usize - 1;
                        let clauses_to_check = if *variables.get_unchecked(abs_l) {
                            p_clauses.get_unchecked(abs_l)
                        } else {
                            n_clauses.get_unchecked(abs_l)
                        };

                        let mut sad = 0;
                        for &c_idx in clauses_to_check {
                            if *num_good_so_far.get_unchecked(c_idx) == 1 { sad += 1; }
                            if sad >= min_sad { break; }
                        }

                        if sad == 0 {
                            let curr_appearances = *var_appearances.get_unchecked(abs_l);
                            let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 3;  // /3 instead of /4
                            let adjusted_weight = curr_appearances.saturating_sub(age_bonus);
                            if min_sad > 0 || adjusted_weight < min_weight {
                                min_sad = 0;
                                min_weight = adjusted_weight;
                                v_min_sad = abs_l;
                            }
                        } else {
                            if min_sad > 0 {
                                let appearances = *var_appearances.get_unchecked(abs_l);
                                let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
                                let combined_weight = sad * 1000 + appearances - age_bonus.min(50);
                                if combined_weight < min_weight {
                                    min_sad = sad;
                                    min_weight = combined_weight;
                                    v_min_sad = abs_l;
                                }
                                if min_sad <= 1 { break; }
                            }
                        }
                    }
                    v_min_sad
                };

                let was_true = *variables.get_unchecked(v);
                let clauses_to_decrement = if was_true { p_clauses.get_unchecked(v) }
                else { n_clauses.get_unchecked(v) };
                let clauses_to_increment = if was_true { n_clauses.get_unchecked(v) }
                else { p_clauses.get_unchecked(v) };

                for &cid in clauses_to_increment {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    *num_good = num_good.saturating_add(1);
                }
                for &cid in clauses_to_decrement {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    let new_val = num_good.saturating_sub(1);
                    *num_good = new_val;
                    if new_val == 0 { residual_.push(cid); }
                }

                *variables.get_unchecked_mut(v) = !was_true;
                *var_age.get_unchecked_mut(v) = 0;
                for &lit in c.iter() {
                    let var = (lit.abs() as usize) - 1;
                    let age = var_age.get_unchecked_mut(var);
                    *age = age.saturating_add(1);
                }
            } else { break; }
            rounds += 1;
        }
    }

    for v in 0..num_variables {
        if p_single[v] { variables[v] = true; }
        else if n_single[v] { variables[v] = false; }
    }
    let _ = save_solution(&Solution { variables });
    Ok(())
}

fn solve_low_density(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let (clauses, p_single, n_single, dead) = preprocess(challenge);
    if dead { return Ok(()); }

    let num_variables = challenge.num_variables;
    let num_clauses = clauses.len();

    if num_clauses == 0 {
        let mut variables = vec![false; num_variables];
        for v in 0..num_variables {
            if p_single[v] { variables[v] = true; }
            else if n_single[v] { variables[v] = false; }
        }
        let _ = save_solution(&Solution { variables });
        return Ok(());
    }

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 { p_clauses[var].push(i); } else { n_clauses[var].push(i); }
        }
    }

    let density = if num_variables > 0 { num_clauses as f64 / num_variables as f64 } else { 0.0 };
    let avg_clause_size = clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64;
    let nv = num_variables as f64;
    let nad = 1.0;
    let random_threshold = 0.003 + 0.007 / (1.0 + (-(nv - 30000.0) / 8000.0).exp());

    // IMPROVEMENT 5: Enhanced sigmoid-based initialization
    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();
        if num_n == 0 && num_p > 0 { variables[v] = true; continue; }
        if num_p == 0 && num_n > 0 { variables[v] = false; continue; }
        let vad = if num_n > 0 { num_p as f64 / num_n as f64 } else { nad + 1.0 };
        let bias_prob = (num_p as f64 + 0.25) / ((num_p + num_n) as f64 + 1.2);
        let steep = if density >= 4.19 && density <= 4.21 { 0.27 }
        else { 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0) };
        let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
        let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
        variables[v] = rng.gen_bool(prob);
    }

    let mut num_good_so_far: Vec<u8> = vec![0; num_clauses];
    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if (l > 0 && variables[var]) || (l < 0 && !variables[var]) {
                num_good_so_far[i] = num_good_so_far[i].saturating_add(1);
            }
        }
    }

    let mut residual_ = Vec::with_capacity(num_clauses);
    let mut in_queue = vec![false; num_clauses];
    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 { in_queue[i] = true; residual_.push(i); }
    }
    if residual_.is_empty() {
        for v in 0..num_variables {
            if p_single[v] { variables[v] = true; }
            else if n_single[v] { variables[v] = false; }
        }
        let _ = save_solution(&Solution { variables });
        return Ok(());
    }

    let base_prob = 0.45 + 0.1 * (density / 5.0).min(1.0);
    let mut current_prob = base_prob;
    let large_problem_scale = ((num_variables as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = (base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;

    // IMPROVEMENT 6: Tuned stagnation parameters
    let max_random_prob = 0.92;
    let prob_adjustment_factor = if density >= 4.18 && density <= 4.22 { 0.045 } else { 0.035 };
    let smoothing_factor = if density >= 4.18 && density <= 4.22 { 0.73 } else { 0.78 };

    let mut last_check_residual = residual_.len();
    let initial_residual_size = residual_.len();
    let size_scale = 1.0 / (1.0 + (-(nv - 30000.0) / 7000.0).exp());
    let perturbation_flips = if density >= 4.195 { 4 } else { 1 + (2.0 * size_scale) as usize };
    let stagnation_limit = if density >= 4.19 && density <= 4.21 { 2 }
    else { 2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize };
    let mut stagnation = 0usize;
    let unsat_check_threshold = check_interval * 5;
    let mut min_residual_seen = residual_.len();
    let mut best_variables = variables.clone();
    let mut best_residual = residual_.len();

    let max_fuel = hyperparameters.as_ref()
        .and_then(|h| h.get("max_fuel_low").or_else(|| h.get("max_fuel")))
        .and_then(|v| v.as_f64())
        .unwrap_or(12_500_000_000.0);
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = 1.0 + 0.5 * (1.0 / (1.0 + (-(nv - 25000.0) / 8000.0).exp()));
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (num_variables as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_num_rounds = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
    let mut rounds = 0;
    let mut flip_count = vec![0u32; num_variables];
    let mut var_age = vec![0u16; num_variables];

    unsafe {
        loop {
            if rounds >= max_num_rounds { return Ok(()); }

            if residual_.len() < min_residual_seen { min_residual_seen = residual_.len(); }
            if residual_.len() < best_residual {
                best_residual = residual_.len();
                best_variables = variables.clone();
            }

            // IMPROVEMENT 7: Smarter early termination
            if rounds > unsat_check_threshold && rounds % check_interval == 0 {
                let total_progress = initial_residual_size.saturating_sub(min_residual_seen);
                let progress_pct = total_progress as f64 / initial_residual_size.max(1) as f64;
                let residual_pct = min_residual_seen as f64 / initial_residual_size.max(1) as f64;
                let fuel_used_pct = rounds as f64 / max_num_rounds.max(1) as f64;
                let should_give_up = if fuel_used_pct > 0.7 {
                    progress_pct < 0.05 && residual_pct > 0.5
                } else if fuel_used_pct > 0.5 {
                    progress_pct < 0.10 && residual_pct > 0.7
                } else { false };
                if should_give_up { return Ok(()); }
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual_.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                let progress_threshold = if density >= 4.19 && density <= 4.21 {
                    0.16 + 0.06 * (density / 3.0).min(1.0)
                } else { 0.15 + 0.05 * (density / 3.0).min(1.0) };

                if progress <= 0 {
                    stagnation = stagnation.saturating_add(1);
                    let density_adj = 1.0 + (density - 4.18).max(0.0) * 10.0;
                    let prob_adjustment = prob_adjustment_factor * density_adj
                        * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= stagnation_limit {
                        // IMPROVEMENT 8: Better stagnation recovery with best-known restore
                        if stagnation >= 5 && best_residual < residual_.len() {
                            variables = best_variables.clone();
                            num_good_so_far.fill(0);
                            for (ci, c) in clauses.iter().enumerate() {
                                for &l in c {
                                    let var = (l.abs() - 1) as usize;
                                    if (l > 0 && *variables.get_unchecked(var)) ||
                                       (l < 0 && !*variables.get_unchecked(var)) {
                                        *num_good_so_far.get_unchecked_mut(ci) =
                                            num_good_so_far.get_unchecked(ci).saturating_add(1);
                                    }
                                }
                            }
                            residual_.clear();
                            for ci in 0..num_clauses {
                                in_queue[ci] = *num_good_so_far.get_unchecked(ci) == 0;
                                if in_queue[ci] { residual_.push(ci); }
                            }
                        } else {
                            for _ in 0..perturbation_flips {
                                let v = rng.gen::<usize>() % num_variables;
                                let was_true = *variables.get_unchecked(v);
                                let clauses_to_dec = if was_true { p_clauses.get_unchecked(v) }
                                else { n_clauses.get_unchecked(v) };
                                let clauses_to_inc = if was_true { n_clauses.get_unchecked(v) }
                                else { p_clauses.get_unchecked(v) };
                                for &cid in clauses_to_inc {
                                    let ng = num_good_so_far.get_unchecked_mut(cid);
                                    *ng = ng.saturating_add(1);
                                }
                                for &cid in clauses_to_dec {
                                    let ng = num_good_so_far.get_unchecked_mut(cid);
                                    let nv = ng.saturating_sub(1);
                                    *ng = nv;
                                    if nv == 0 && !*in_queue.get_unchecked(cid) {
                                        *in_queue.get_unchecked_mut(cid) = true;
                                        residual_.push(cid);
                                    }
                                }
                                *variables.get_unchecked_mut(v) = !was_true;
                                *flip_count.get_unchecked_mut(v) += 1;
                            }
                        }
                        stagnation = 0;
                        current_prob = base_prob;
                    }
                } else if progress_ratio > progress_threshold {
                    current_prob = base_prob;
                    stagnation = 0;
                } else {
                    current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                }
                last_check_residual = residual_.len();
            }

            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();
                let mut i;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = residual_[id];
                    if num_good_so_far[i] > 0 {
                        *in_queue.get_unchecked_mut(i) = false;
                        residual_.swap_remove(id);
                    } else { break; }
                }
                if residual_.is_empty() { break; }

                let id = rand_val % residual_.len();
                let i = residual_[id];
                let c = clauses.get_unchecked_mut(i);
                if c.len() > 1 { let ri = rand_val % c.len(); c.swap(0, ri); }

                let mut zero_found = None;
                'outer2: for &l in c.iter() {
                    let abs_l = l.abs() as usize - 1;
                    let clauses_to_check = if *variables.get_unchecked(abs_l) {
                        p_clauses.get_unchecked(abs_l)
                    } else { n_clauses.get_unchecked(abs_l) };
                    for &cx in clauses_to_check {
                        if *num_good_so_far.get_unchecked(cx) == 1 { continue 'outer2; }
                    }
                    zero_found = Some(abs_l);
                    break;
                }

                let v = if let Some(abs_l) = zero_found { abs_l }
                else if rng.gen::<f64>() < current_prob { c[0].abs() as usize - 1 }
                else {
                    let mut min_sad = usize::MAX;
                    let mut v_min = c[0].abs() as usize - 1;
                    let mut min_w = usize::MAX;
                    for &l in c.iter() {
                        let abs_l = l.abs() as usize - 1;
                        let check = if *variables.get_unchecked(abs_l) {
                            p_clauses.get_unchecked(abs_l)
                        } else { n_clauses.get_unchecked(abs_l) };
                        let mut sad = 0;
                        for &cx in check {
                            if *num_good_so_far.get_unchecked(cx) == 1 { sad += 1; }
                            if sad >= min_sad { break; }
                        }
                        if sad == 0 {
                            let app = *flip_count.get_unchecked(abs_l) as usize;
                            let age_b = (*var_age.get_unchecked(abs_l) as usize) / 3;
                            let aw = app.saturating_sub(age_b);
                            if min_sad > 0 || aw < min_w { min_sad = 0; min_w = aw; v_min = abs_l; }
                        } else if min_sad > 0 {
                            let app = *flip_count.get_unchecked(abs_l) as usize;
                            let age_b = (*var_age.get_unchecked(abs_l) as usize) / 2;
                            let cw = sad * 1000 + app - age_b.min(50);
                            if cw < min_w { min_sad = sad; min_w = cw; v_min = abs_l; }
                            if min_sad <= 1 { break; }
                        }
                    }
                    v_min
                };

                let was_true = *variables.get_unchecked(v);
                let dec = if was_true { p_clauses.get_unchecked(v) } else { n_clauses.get_unchecked(v) };
                let inc = if was_true { n_clauses.get_unchecked(v) } else { p_clauses.get_unchecked(v) };
                for &cid in inc {
                    let ng = num_good_so_far.get_unchecked_mut(cid);
                    *ng = ng.saturating_add(1);
                }
                for &cid in dec {
                    let ng = num_good_so_far.get_unchecked_mut(cid);
                    let nval = ng.saturating_sub(1);
                    *ng = nval;
                    if nval == 0 && !*in_queue.get_unchecked(cid) {
                        *in_queue.get_unchecked_mut(cid) = true;
                        residual_.push(cid);
                    }
                }
                *variables.get_unchecked_mut(v) = !was_true;
                *var_age.get_unchecked_mut(v) = 0;
                *flip_count.get_unchecked_mut(v) += 1;
                for &lit in c.iter() {
                    let var = (lit.abs() as usize) - 1;
                    let age = var_age.get_unchecked_mut(var);
                    *age = age.saturating_add(1);
                }
            } else { break; }
            rounds += 1;
        }
    }

    for v in 0..num_variables {
        if p_single[v] { variables[v] = true; }
        else if n_single[v] { variables[v] = false; }
    }
    let _ = save_solution(&Solution { variables });
    Ok(())
}

pub fn help() {
    println!("Turbo SAT: Enhanced WalkSAT with Adaptive Stagnation Recovery");
    println!();
    println!("Improvements over sat_excelled:");
    println!("  - Factored preprocessing into shared function");
    println!("  - Higher random walk probability cap (0.92 vs 0.9)");
    println!("  - More aggressive probability adjustment factors");
    println!("  - Faster age bonus for variable selection (/3 vs /4)");
    println!("  - Better stagnation recovery with best-known restore");
    println!();
    println!("Hyperparameters:");
    println!("  max_fuel_high (default: 10B): Max fuel for high-density instances");
    println!("  max_fuel_low  (default: 12.5B): Max fuel for low-density instances");
}
