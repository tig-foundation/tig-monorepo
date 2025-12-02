use rand::{rngs::SmallRng, SeedableRng, Rng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
use crate::{seeded_hasher, HashSet};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    // Save a random initial solution as fallback (in case solver times out)
    {
        let variables: Vec<bool> = (0..challenge.difficulty.num_variables)
            .map(|_| rng.gen_bool(0.5))
            .collect();
        let _ = save_solution(&Solution { variables });
    }

    let mut clauses = challenge.clauses.clone();
    let hasher = seeded_hasher(&challenge.seed);
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        let clause = &mut clauses[i];
        
        if clause.len() > 1 {
            let mut seen = HashSet::with_hasher(hasher.clone());
            let mut j = 0;
            let mut is_tautology = false;
            while j < clause.len() {
                let lit = clause[j];
                if seen.contains(&-lit) {
                    is_tautology = true;
                    break;
                }
                if !seen.insert(lit) {
                    clause.swap_remove(j);
                } else {
                    j += 1;
                }
            }
            if is_tautology {
                clauses.swap_remove(i);
            }
        }
    }

    let mut p_single = vec![false; challenge.difficulty.num_variables];
    let mut n_single = vec![false; challenge.difficulty.num_variables];
    let mut clauses_ = clauses;
    clauses = Vec::with_capacity(clauses_.len());
    let mut dead = false;

    while !dead {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len()); 
            let mut skip = false;
            for l in c.iter() {
                if (p_single[(l.abs() - 1) as usize] && *l > 0)
                    || (n_single[(l.abs() - 1) as usize] && *l < 0)
                {
                    skip = true;
                    break;
                } else if p_single[(l.abs() - 1) as usize]
                    || n_single[(l.abs() - 1) as usize]
                {
                    done = false;
                    continue;
                } else {
                    c_.push(*l);
                }
            }
            if skip {
                done = false;
                continue;
            };
            match c_[..] {
                [l] => {
                    done = false;
                    let idx = (l.abs() - 1) as usize;
                    let (check, set) = if l > 0 { (&n_single, &mut p_single) } else { (&p_single, &mut n_single) };
                    if check[idx] {
                        dead = true;
                        break;
                    }
                    set[idx] = true;
                }
                [] => {
                    dead = true;
                    break;
                }
                _ => {
                    clauses.push(c_);
                }
            }
        }
        if done {
            break;
        } else {
            clauses_ = clauses;
            clauses = Vec::with_capacity(clauses_.len());
        }
    }

    if dead {
        return Ok(());
    }

    // Early return if all clauses satisfied by unit propagation
    if clauses.is_empty() {
        let mut variables = vec![false; challenge.difficulty.num_variables];
        for v in 0..challenge.difficulty.num_variables {
            if p_single[v] {
                variables[v] = true;
            }
            // n_single means false, which is the default
        }
        let _ = save_solution(&Solution { variables });
        return Ok(());
    }

    let num_variables = challenge.difficulty.num_variables;
    let num_clauses = clauses.len();

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];

    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
            } else {
                n_clauses[var].push(i);
            }
        }
    }

    let var_appearances: Vec<usize> = (0..num_variables)
        .map(|v| p_clauses[v].len() + n_clauses[v].len())
        .collect();

    let clauses_ratio = challenge.difficulty.clauses_to_variables_percent as f64;

    if clauses_ratio < 417.0 {
        // Low ratio strategy from solve_low_density_excelled
        let density = num_clauses as f64 / num_variables as f64;
        let avg_clause_size = clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64;

        let nv = num_variables as f64;
        let nad = 1.0;
        let random_threshold = 0.003 + 0.007 / (1.0 + (-(nv - 30000.0) / 8000.0).exp());

        let mut variables = vec![false; num_variables];
        for v in 0..num_variables {
            // Apply forced values from unit propagation first
            if p_single[v] {
                variables[v] = true;
                continue;
            }
            if n_single[v] {
                variables[v] = false;
                continue;
            }

            let num_p = p_clauses[v].len();
            let num_n = n_clauses[v].len();

            if num_n == 0 && num_p > 0 {
                variables[v] = true;
                continue;
            } else if num_p == 0 && num_n > 0 {
                variables[v] = false;
                continue;
            }

            let vad = if num_n > 0 {
                num_p as f64 / num_n as f64
            } else {
                nad + 1.0
            };
            let bias_prob = (num_p as f64 + 0.25) / ((num_p + num_n) as f64 + 1.2);
            let steep = if density >= 4.19 && density <= 4.21 {
                0.27
            } else {
                0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0)
            };
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
            if num_good == 0 {
                in_queue[i] = true;
                residual_.push(i);
            }
        }

        if residual_.is_empty() {
            for v in 0..num_variables {
                if p_single[v] {
                    variables[v] = true;
                } else if n_single[v] {
                    variables[v] = false;
                }
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
        let max_random_prob = 0.9;
        let prob_adjustment_factor = if density >= 4.18 && density <= 4.22 { 0.04 } else { 0.03 };
        let smoothing_factor = if density >= 4.18 && density <= 4.22 { 0.75 } else { 0.8 };

        let mut last_check_residual = residual_.len();
        let initial_residual_size = residual_.len();
        let size_scale = 1.0 / (1.0 + (-(nv - 30000.0) / 7000.0).exp());
        let perturbation_flips = if density >= 4.195 { 4 } else { 1 + (2.0 * size_scale) as usize };

        let stagnation_limit = if density >= 4.19 && density <= 4.21 { 2 } else { 2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize };
        let mut stagnation = 0usize;

        let unsat_check_threshold = check_interval * 5;
        let mut min_residual_seen = residual_.len();

        let mut best_variables = variables.clone();
        let mut best_residual = residual_.len();

        let max_fuel = 12_500_000_000.0;
        let difficulty_factor = density * avg_clause_size.sqrt();
        let scale_factor = 1.0 + 0.5 * (1.0 / (1.0 + (-(nv - 25000.0) / 8000.0).exp()));
        let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (num_variables as f64).sqrt() * scale_factor;
        let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
        let remaining = (max_fuel - base_fuel).max(0.0);
        let max_num_rounds = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
        let mut rounds = 0;
        let mut flip_count = vec![0u32; num_variables];
        let mut var_age = vec![0u16; num_variables];

        let mut var_freq_buffer = vec![0u32; num_variables];
        let mut candidates_buffer: Vec<(usize, u64)> = Vec::with_capacity(num_variables);

        unsafe {
            loop {
                if rounds >= max_num_rounds {
                    return Ok(());
                }

                if residual_.len() < min_residual_seen {
                    min_residual_seen = residual_.len();
                }

                if residual_.len() < best_residual {
                    let improvement = best_residual - residual_.len();
                    if improvement * 20 > best_residual || residual_.len() < 10 {
                        best_residual = residual_.len();
                        best_variables = variables.clone();
                    }
                }

                if rounds > unsat_check_threshold && rounds % check_interval == 0 {
                    let total_progress = initial_residual_size.saturating_sub(min_residual_seen);
                    let progress_pct = total_progress as f64 / initial_residual_size.max(1) as f64;
                    let residual_pct = min_residual_seen as f64 / initial_residual_size.max(1) as f64;

                    let fuel_used_pct = rounds as f64 / max_num_rounds.max(1) as f64;

                    let should_give_up = if fuel_used_pct > 0.7 {
                        progress_pct < 0.05 && residual_pct > 0.5
                    } else if fuel_used_pct > 0.5 {
                        progress_pct < 0.10 && residual_pct > 0.7
                    } else {
                        false
                    };

                    if should_give_up {
                        return Ok(());
                    }
                }

                if rounds % check_interval == 0 && rounds > 0 {
                    let progress = last_check_residual as i64 - residual_.len() as i64;
                    let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;

                    let progress_threshold = if density >= 4.19 && density <= 4.21 {
                        0.16 + 0.06 * (density / 3.0).min(1.0)
                    } else {
                        0.15 + 0.05 * (density / 3.0).min(1.0)
                    };

                    if progress <= 0 {
                        stagnation = stagnation.saturating_add(1);
                        let density_adj = 1.0 + (density - 4.18).max(0.0) * 10.0;
                        let prob_adjustment = prob_adjustment_factor * density_adj * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                        current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                        if stagnation >= stagnation_limit {
                            if stagnation >= 5 && best_residual < residual_.len() {
                                variables = best_variables.clone();
                                num_good_so_far.fill(0);
                                for (i, c) in clauses.iter().enumerate() {
                                    for &l in c {
                                        let var = (l.abs() - 1) as usize;
                                        if (l > 0 && *variables.get_unchecked(var)) || (l < 0 && !*variables.get_unchecked(var)) {
                                            *num_good_so_far.get_unchecked_mut(i) = num_good_so_far.get_unchecked(i).saturating_add(1);
                                        }
                                    }
                                }
                                residual_.clear();
                                for (i, &num_good) in num_good_so_far.iter().enumerate() {
                                    if num_good == 0 {
                                        *in_queue.get_unchecked_mut(i) = true;
                                        residual_.push(i);
                                    } else {
                                        *in_queue.get_unchecked_mut(i) = false;
                                    }
                                }
                            }

                            let base_kicks = perturbation_flips;
                            let kicks = if stagnation >= 5 {
                                (base_kicks * 12).min(100)
                            } else if stagnation >= 4 {
                                (base_kicks * 6).min(50)
                            } else if stagnation >= 3 {
                                (base_kicks * 3).min(20)
                            } else {
                                (base_kicks + 2).min(10)
                            };

                            for _ in 0..kicks {
                                if residual_.is_empty() {
                                    break;
                                }

                                let v = if stagnation >= 3 {
                                    var_freq_buffer.fill(0);
                                    let residual_len = residual_.len();
                                    let base_sample = if stagnation >= 6 { 90 } else if stagnation >= 5 { 70 } else { 45 };
                                    let adaptive_sample = (residual_len / 12).max(base_sample);
                                    let sample_size = adaptive_sample.min(residual_len).min(180);

                                    for _ in 0..sample_size {
                                        let id = rng.gen::<usize>() % residual_len;
                                        let cid = *residual_.get_unchecked(id);
                                        let c = clauses.get_unchecked(cid);
                                        let clause_size = c.len() as u32;
                                        let weight = if clause_size <= 2 {
                                            800
                                        } else if clause_size == 3 {
                                            100
                                        } else if clause_size == 4 {
                                            12
                                        } else {
                                            1
                                        };

                                        for &lit in c {
                                            let var = (lit.abs() as usize) - 1;
                                            var_freq_buffer[var] = var_freq_buffer[var].saturating_add(weight);
                                        }
                                    }

                                    candidates_buffer.clear();
                                    for (var, &freq) in var_freq_buffer.iter().enumerate() {
                                        if freq >= 30 {
                                            let flips = *flip_count.get_unchecked(var);
                                            let freq_squared = (freq as u64).saturating_mul(freq as u64);
                                            let flip_divisor = 1000 + (flips.min(10000) as u64);
                                            let score = freq_squared.saturating_mul(100000) / flip_divisor;
                                            candidates_buffer.push((var, score));
                                        }
                                    }

                                    if candidates_buffer.is_empty() {
                                        let mut best_id = 0;
                                        let mut min_size = usize::MAX;
                                        let search_limit = 8.min(residual_len);
                                        for _ in 0..search_limit {
                                            let id = rng.gen::<usize>() % residual_len;
                                            let cid = *residual_.get_unchecked(id);
                                            let size = clauses.get_unchecked(cid).len();
                                            if size < min_size {
                                                min_size = size;
                                                best_id = id;
                                            }
                                        }
                                        let cid = residual_[best_id];
                                        let c = clauses.get_unchecked(cid);
                                        if c.is_empty() {
                                            continue;
                                        }
                                        let lit = c[rng.gen::<usize>() % c.len()];
                                        (lit.abs() as usize) - 1
                                    } else {
                                        let selection_size = if stagnation >= 6 {
                                            (candidates_buffer.len() / 7).max(1)
                                        } else if stagnation >= 5 {
                                            (candidates_buffer.len() / 6).max(1)
                                        } else {
                                            (candidates_buffer.len() / 5).max(1)
                                        };
                                        if selection_size < candidates_buffer.len() {
                                            candidates_buffer.select_nth_unstable_by_key(
                                                selection_size - 1,
                                                |&(_, score)| std::cmp::Reverse(score)
                                            );
                                        }
                                        candidates_buffer[rng.gen::<usize>() % selection_size].0
                                    }
                                } else {
                                    let id = rng.gen::<usize>() % residual_.len();
                                    let cid = residual_[id];
                                    let c = clauses.get_unchecked(cid);
                                    if c.is_empty() {
                                        continue;
                                    }
                                    let lit = c[rng.gen::<usize>() % c.len()];
                                    (lit.abs() as usize) - 1
                                };

                                let was_true = *variables.get_unchecked(v);
                                let inc = if was_true { n_clauses.get_unchecked(v) } else { p_clauses.get_unchecked(v) };
                                let dec = if was_true { p_clauses.get_unchecked(v) } else { n_clauses.get_unchecked(v) };

                                for &cid2 in inc {
                                    let num_good = num_good_so_far.get_unchecked_mut(cid2);
                                    *num_good = num_good.saturating_add(1);
                                }
                                for &cid2 in dec {
                                    let num_good = num_good_so_far.get_unchecked_mut(cid2);
                                    let new_val = num_good.saturating_sub(1);
                                    *num_good = new_val;
                                    if new_val == 0 && !*in_queue.get_unchecked(cid2) {
                                        *in_queue.get_unchecked_mut(cid2) = true;
                                        residual_.push(cid2);
                                    }
                                }
                                *variables.get_unchecked_mut(v) = !was_true;
                                *flip_count.get_unchecked_mut(v) = flip_count.get_unchecked(v).saturating_add(1);
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

                    last_check_residual = residual_.len();
                }

                if !residual_.is_empty() {
                    let rand_val = rng.gen::<usize>();

                    let mut i;
                    while !residual_.is_empty() {
                        let rand_pair = rng.gen::<u64>();
                        let residual_len = residual_.len();
                        let id1 = (rand_pair as usize) % residual_len;
                        let id2 = ((rand_pair >> 32) as usize) % residual_len;
                        let cid1 = residual_[id1];
                        let cid2 = residual_[id2];
                        let mut best_id = if clauses.get_unchecked(cid2).len() < clauses.get_unchecked(cid1).len() {
                            id2
                        } else {
                            id1
                        };
                        if density >= 4.195 {
                            let id3 = rng.gen::<usize>() % residual_len;
                            let cid3 = residual_[id3];
                            let best_cid = residual_[best_id];
                            if clauses.get_unchecked(cid3).len() < clauses.get_unchecked(best_cid).len() {
                                best_id = id3;
                            }
                        }
                        i = residual_[best_id];
                        if num_good_so_far[i] > 0 {
                            in_queue[i] = false;
                            residual_.swap_remove(best_id);
                        } else {
                            break;
                        }
                    }
                    if residual_.is_empty() {
                        for v in 0..num_variables {
                            if p_single[v] {
                                variables[v] = true;
                            } else if n_single[v] {
                                variables[v] = false;
                            }
                        }
                        save_solution(&Solution { variables })?;
                        return Ok(());
                    }

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
                            if *num_good_so_far.get_unchecked(c) == 1 {
                                continue 'outer;
                            }
                        }
                        zero_found = Some(abs_l);
                        break;
                    }

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
                                if *num_good_so_far.get_unchecked(c_idx) == 1 {
                                    sad += 1;
                                }
                                if sad >= min_sad {
                                    break;
                                }
                            }

                            if sad == 0 {
                                let curr_appearances = *var_appearances.get_unchecked(abs_l);
                                let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 4;
                                let adjusted_weight = curr_appearances.saturating_sub(age_bonus);
                                if min_sad > 0 || adjusted_weight < min_weight {
                                    min_sad = 0;
                                    min_weight = adjusted_weight;
                                    v_min_sad = abs_l;
                                }
                            } else {
                                if min_sad > 0 {
                                    let appearances = *var_appearances.get_unchecked(abs_l);

                                    let sad_weight = if density >= 4.19 && density <= 4.21 {
                                        1024
                                    } else if density >= 4.195 {
                                        512
                                    } else {
                                        256
                                    };
                                    let age_bonus = (*var_age.get_unchecked(abs_l) as usize) / 2;
                                    let combined_weight = sad * sad * sad_weight + appearances - age_bonus.min(50);

                                    if combined_weight < min_weight {
                                        min_sad = sad;
                                        min_weight = combined_weight;
                                        v_min_sad = abs_l;
                                    }

                                    if min_sad <= 1 {
                                        break;
                                    }
                                }
                            }
                        }
                        v_min_sad
                    };

                    let was_true = *variables.get_unchecked(v);
                    let clauses_to_decrement = if was_true {
                        p_clauses.get_unchecked(v)
                    } else {
                        n_clauses.get_unchecked(v)
                    };
                    let clauses_to_increment = if was_true {
                        n_clauses.get_unchecked(v)
                    } else {
                        p_clauses.get_unchecked(v)
                    };

                    for &cid in clauses_to_increment {
                        let num_good = num_good_so_far.get_unchecked_mut(cid);
                        *num_good = num_good.saturating_add(1);
                    }

                    for &cid in clauses_to_decrement {
                        let num_good = num_good_so_far.get_unchecked_mut(cid);
                        let new_val = num_good.saturating_sub(1);
                        *num_good = new_val;
                        if new_val == 0 && !*in_queue.get_unchecked(cid) {
                            *in_queue.get_unchecked_mut(cid) = true;
                            residual_.push(cid);
                        }
                    }

                    *variables.get_unchecked_mut(v) = !was_true;
                    *flip_count.get_unchecked_mut(v) = flip_count.get_unchecked(v).saturating_add(1);
                    *var_age.get_unchecked_mut(v) = 0;
                } else {
                    break;
                }
                rounds += 1;
                let selected_clause = clauses.get_unchecked(i);
                for &lit in selected_clause {
                    let var = (lit.abs() as usize) - 1;
                    let age = var_age.get_unchecked_mut(var);
                    *age = age.saturating_add(1);
                }
            }
        }

        for v in 0..num_variables {
            if p_single[v] {
                variables[v] = true;
            } else if n_single[v] {
                variables[v] = false;
            }
        }
        save_solution(&Solution { variables })?;
        return Ok(());
    }

    // High ratio strategy (clauses_ratio >= 417)
    let density = num_clauses as f64 / num_variables as f64;
    let avg_clause_size = clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64;
    
    let is_target_problem = num_variables >= 4500 && num_variables <= 5500 
                          && clauses_ratio >= 400.0 && clauses_ratio <= 450.0;    
    
    let use_lower_variable_strategy = num_variables <= 7000;    
    
    let nad = if use_lower_variable_strategy { 1.28 } else { 1.0 };  
    
    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        // Apply forced values from unit propagation
        if p_single[v] {
            variables[v] = true;
            continue;
        }
        if n_single[v] {
            variables[v] = false;
            continue;
        }
        
        // Only randomize unforced variables
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();

        let mut vad = nad + 1.0;
        if num_n > 0 {
            vad = num_p as f64 / num_n as f64;
        }

        if use_lower_variable_strategy {
            if vad <= nad {
                variables[v] = rng.gen::<f64>() < 0.001;
            } else {
                let prob = (num_p as f64 + 0.5) / ((num_p + num_n) as f64 + 1.0);
                variables[v] = rng.gen_bool(prob);
            }
        } else {
            if vad <= nad {
                variables[v] = rng.gen::<f64>() < 0.003;
            } else {
                let prob = num_p as f64 / (num_p + num_n).max(1) as f64;
                variables[v] = rng.gen_bool(prob);
            }
        }
    }

    let mut num_good_so_far: Vec<u8> = vec![0; num_clauses];
    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 && variables[var] {
                num_good_so_far[i] += 1
            } else if l < 0 && !variables[var] {
                num_good_so_far[i] += 1
            }
        }
    }

    let mut residual_ = Vec::with_capacity(num_clauses);

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
        }
    }
    
    let base_prob = 0.52;
    let mut current_prob = base_prob;
    let mut prob_threshold = (current_prob * (usize::MAX as f64)) as usize;
    
    let check_interval = (50.0 * (1.0 + (density / 3.0).ln().max(0.0))).max(20.0) as usize;
    let mut last_check_residual = residual_.len();
    
    let max_fuel = 10_000_000_000.0;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (num_variables as f64).sqrt();
    let flip_fuel = 200.0 + difficulty_factor;
    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;
    let mut rounds = 0;

    unsafe {
        loop {
            if rounds >= max_num_rounds {
                return Ok(());
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual_.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;
                
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    let prob_adjustment = 0.025 * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(0.9);
                } else if progress_ratio > progress_threshold { 
                    current_prob = base_prob;
                } else {
                    current_prob = current_prob * 0.8 + base_prob * 0.2;
                }
                prob_threshold = (current_prob * (usize::MAX as f64)) as usize;
                
                last_check_residual = residual_.len();
            }

            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();
                
                let mut i;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = residual_[id];
                    if num_good_so_far[i] > 0 { 
                        residual_.swap_remove(id); 
                    } else {
                        break
                    }
                }
                if residual_.is_empty() {
                    break;
                }
                
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
                       if *num_good_so_far.get_unchecked(c) == 1 {
                           continue 'outer;
                       }
                   }
                   zero_found = Some(abs_l);
                   break;
                }
                
                let v = if let Some(abs_l) = zero_found {
                   abs_l
                } else if rand_val < prob_threshold {
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
                           if *num_good_so_far.get_unchecked(c_idx) == 1 {
                               sad += 1;
                           }
                           if sad >= min_sad {
                               break;
                           }
                       }
                       
                       let appearances = p_clauses.get_unchecked(abs_l).len() + n_clauses.get_unchecked(abs_l).len();
                       
                       let combined_weight = if is_target_problem {
                           sad
                       } else if use_lower_variable_strategy {
                           sad * 100 + (appearances / 10)
                       } else {
                           sad * 1000 + appearances
                       };
                       
                       if combined_weight < min_weight {
                           min_sad = sad;
                           min_weight = combined_weight;
                           v_min_sad = abs_l;
                       }
                       
                       if min_sad <= 1 {
                            break;
                       }
                   }
                   v_min_sad
                };
                
                let was_true = *variables.get_unchecked(v);
                let clauses_to_decrement = if was_true {
                    p_clauses.get_unchecked(v)
                } else {
                    n_clauses.get_unchecked(v)
                };
                let clauses_to_increment = if was_true {
                    n_clauses.get_unchecked(v)
                } else {
                    p_clauses.get_unchecked(v)
                };
        
                for &cid in clauses_to_increment {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    *num_good += 1;
                }
        
                for &cid in clauses_to_decrement {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    *num_good -= 1;
                    if *num_good == 0 {
                        residual_.push(cid);
                    }
                }
        
                *variables.get_unchecked_mut(v) = !was_true;
            } else {
                break;
            }
            rounds += 1;
        }
     }
    let _ = save_solution(&Solution { variables });
    Ok(())
}