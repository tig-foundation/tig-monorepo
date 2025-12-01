use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let num_variables = challenge.num_variables;

    let solution = if num_variables >= 10000 {
        solve_large_instance(challenge)
    } else {
        solve_hard_instance(challenge)
    };
    if let Some(s) = solution? {
        let _ = save_solution(&s);
    }
    Ok(())
}

fn solve_large_instance(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng =
        SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let mut clauses = challenge.clauses.clone();
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        let clause = &mut clauses[i];

        if clause.len() > 1 {
            let mut j = 1;
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

    let mut p_single = vec![false; challenge.num_variables];
    let mut n_single = vec![false; challenge.num_variables];
    let mut clauses_ = clauses;
    clauses = Vec::with_capacity(clauses_.len());
    let mut dead = false;

    while !(dead) {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len());
            let mut skip = false;
            for (i, l) in c.iter().enumerate() {
                if (p_single[(l.abs() - 1) as usize] && *l > 0)
                    || (n_single[(l.abs() - 1) as usize] && *l < 0)
                    || c[(i + 1)..].contains(&-l)
                {
                    skip = true;
                    break;
                } else if p_single[(l.abs() - 1) as usize]
                    || n_single[(l.abs() - 1) as usize]
                    || c[(i + 1)..].contains(&l)
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
                    if l > 0 {
                        if n_single[(l.abs() - 1) as usize] {
                            dead = true;
                            break;
                        } else {
                            p_single[(l.abs() - 1) as usize] = true;
                        }
                    } else {
                        if p_single[(l.abs() - 1) as usize] {
                            dead = true;
                            break;
                        } else {
                            n_single[(l.abs() - 1) as usize] = true;
                        }
                    }
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
        return Ok(None);
    }

    let num_variables = challenge.num_variables;
    let num_clauses = clauses.len();

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];

    for c in &clauses {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                if p_clauses[var].capacity() == 0 {
                    p_clauses[var] = Vec::with_capacity(clauses.len() / num_variables + 1);
                }
            } else {
                if n_clauses[var].capacity() == 0 {
                    n_clauses[var] = Vec::with_capacity(clauses.len() / num_variables + 1);
                }
            }
        }
    }

    for (i, &ref c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
            } else {
                n_clauses[var].push(i);
            }
        }
    }

    let density = num_clauses as f64 / num_variables as f64;
    let avg_clause_size =
        clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64;
    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;

    let is_target_problem = num_variables >= 4500
        && num_variables <= 5500
        && clauses_ratio >= 400.0
        && clauses_ratio <= 450.0;

    let use_lower_variable_strategy = num_variables <= 7000;

    let nad = if use_lower_variable_strategy {
        1.28
    } else {
        1.0
    };

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
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
    for (i, &ref c) in clauses.iter().enumerate() {
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

    let is_large_problem = num_variables >= 80000;

    let (check_interval, max_random_prob, prob_adjustment_factor, smoothing_factor) =
        if is_large_problem {
            let check_int = (25.0 * (1.0 + (density / 3.0).ln().max(0.0))).max(12.0) as usize;
            let max_rand = 0.9;
            let adj_factor = 0.025;
            let smooth = 0.8;
            (check_int, max_rand, adj_factor, smooth)
        } else {
            let check_int = (50.0 * (1.0 + (density / 3.0).ln().max(0.0))).max(20.0) as usize;
            let max_rand = 0.9;
            let adj_factor = 0.025;
            let smooth = 0.8;
            (check_int, max_rand, adj_factor, smooth)
        };

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
                return Ok(None);
            }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_residual as i64 - residual_.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;

                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    let prob_adjustment = prob_adjustment_factor
                        * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);
                } else if progress_ratio > progress_threshold {
                    current_prob = base_prob;
                } else {
                    current_prob =
                        current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                }

                last_check_residual = residual_.len();
            }

            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();

                let mut i = residual_.len() - 1;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = residual_[id];
                    if num_good_so_far[i] > 0 {
                        residual_.swap_remove(id);
                    } else {
                        break;
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
                } else if rand_val < (current_prob * (usize::MAX as f64)) as usize {
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

                        let appearances = p_clauses[abs_l].len() + n_clauses[abs_l].len();

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
    return Ok(Some(Solution { variables }));
}

fn solve_hard_instance(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let seed = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap());

    if ((challenge.clauses.len() * 100 / challenge.num_variables) as f64) >= 425.0
        && challenge.num_variables <= 7500
    {
        match solve_with_seed(challenge, seed) {
            Ok(Some(solution)) => return Ok(Some(solution)),
            _ => {
                for i in 1..4 {
                    let alt_seed = seed.wrapping_add(i * 1000000007);
                    match solve_with_seed(challenge, alt_seed) {
                        Ok(Some(solution)) => return Ok(Some(solution)),
                        _ => continue,
                    }
                }
                return Ok(None);
            }
        }
    } else {
        return solve_with_seed(challenge, seed);
    }
}

fn solve_with_seed(challenge: &Challenge, seed: u64) -> anyhow::Result<Option<Solution>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut clauses = challenge.clauses.clone();
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        {
            let clause = &mut clauses[i];
            if clause.len() >= 3 && (clause[0] == clause[2] || clause[1] == clause[2]) {
                clause.pop();
            }
            if clause.len() >= 2 && clause[0] == clause[1] {
                clause.swap_remove(1);
            }
        }

        let should_remove = {
            let clause = &clauses[i];
            (clause.len() >= 2 && clause[0] == -clause[1])
                || (clause.len() >= 3 && (clause[0] == -clause[2] || clause[1] == -clause[2]))
        };
        if should_remove {
            clauses.swap_remove(i);
        }
    }

    let num_variables = challenge.num_variables;
    let num_clauses = clauses.len();

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];

    for (i, &ref c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
            } else {
                n_clauses[var].push(i);
            }
        }
    }

    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;
    let is_very_hard = clauses_ratio >= 428.0;

    let (pad, nad) = if is_very_hard {
        (1.85, 0.54)
    } else {
        (1.8, 0.56)
    };

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();

        let vad;
        if num_n == 0 {
            vad = pad + 1.0;
        } else {
            vad = num_p as f32 / num_n as f32;
        }

        if vad > pad {
            variables[v] = true;
        } else if vad < nad {
            variables[v] = false;
        } else {
            variables[v] = rng.gen_bool(0.5);
        }
    }

    let mut num_good_so_far: Vec<u8> = vec![0; num_clauses];
    for (i, &ref c) in clauses.iter().enumerate() {
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

    let base_prob = if is_very_hard { 0.54 } else { 0.52 };
    let current_prob = base_prob;

    let num_vars = challenge.num_variables as f64;
    let max_fuel = 2000000000.0;

    let base_fuel = if is_very_hard {
        (2000.0 + 45.0 * clauses_ratio) * num_vars
    } else {
        (2000.0 + 40.0 * clauses_ratio) * num_vars
    };

    let flip_fuel = if is_very_hard {
        320.0 + 0.95 * clauses_ratio
    } else {
        350.0 + 0.9 * clauses_ratio
    };

    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;
    let mut rounds = 0;

    let early_exit_threshold = if is_very_hard {
        max_num_rounds / 2
    } else {
        max_num_rounds
    };

    let stagnation_check_interval = if num_variables <= 5000 && clauses_ratio >= 428.0 {
        2000
    } else {
        2500
    };
    let mut last_check_residual_size = residual_.len();
    let mut stagnation_counter = 0;

    unsafe {
        loop {
            if is_very_hard && rounds > early_exit_threshold && residual_.len() > num_clauses / 20 {
                return Ok(None);
            }

            if rounds % stagnation_check_interval == 0 && rounds > 0 {
                if residual_.len() >= last_check_residual_size {
                    stagnation_counter += 1;

                    if stagnation_counter >= 3 && residual_.len() < num_clauses / 4 {
                        let restart_ratio = if num_variables < 3000 {
                            0.01
                        } else if num_variables <= 5000 {
                            0.004
                        } else {
                            0.002
                        };
                        let num_to_flip = (num_variables as f64 * restart_ratio) as usize;
                        for _ in 0..num_to_flip {
                            let v = rng.gen_range(0..num_variables);
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
                        }

                        stagnation_counter = 0;
                    }
                } else {
                    stagnation_counter = 0;
                }

                last_check_residual_size = residual_.len();
            }

            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();

                let mut i = residual_.len() - 1;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = *residual_.get_unchecked(id);
                    if *num_good_so_far.get_unchecked(i) > 0 {
                        residual_.swap_remove(id);
                    } else {
                        break;
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

                let v = if rand_val < (current_prob * (usize::MAX as f64)) as usize {
                    let mut zero_found = None;

                    'outer: for &l in c.iter() {
                        let abs_l = l.abs() as usize - 1;
                        let clauses_to_check = if l > 0 {
                            n_clauses.get_unchecked(abs_l)
                        } else {
                            p_clauses.get_unchecked(abs_l)
                        };

                        for &c in clauses_to_check.iter() {
                            if *num_good_so_far.get_unchecked(c) == 1 {
                                continue 'outer;
                            }
                        }
                        zero_found = Some(l);
                        break;
                    }

                    if let Some(l) = zero_found {
                        l
                    } else {
                        *c.get_unchecked(0)
                    }
                } else {
                    let mut min_sad = usize::MAX;
                    let mut v_min_sad = 0;

                    for &l in c.iter() {
                        let abs_l = l.abs() as usize - 1;
                        let clauses_to_check = if l > 0 {
                            n_clauses.get_unchecked(abs_l)
                        } else {
                            p_clauses.get_unchecked(abs_l)
                        };

                        let mut sad = 0;
                        for &c in clauses_to_check.iter() {
                            if *num_good_so_far.get_unchecked(c) == 1 {
                                sad += 1;
                            }
                            if sad >= min_sad {
                                break;
                            }
                        }

                        if sad < min_sad {
                            min_sad = sad;
                            v_min_sad = l;
                        }
                        if sad == 0 {
                            break;
                        }
                    }
                    v_min_sad
                };

                let v_idx = v.abs() as usize - 1;
                let was_true = v < 0;
                let clauses_to_decrement = if was_true {
                    p_clauses.get_unchecked(v_idx)
                } else {
                    n_clauses.get_unchecked(v_idx)
                };
                let clauses_to_increment = if was_true {
                    n_clauses.get_unchecked(v_idx)
                } else {
                    p_clauses.get_unchecked(v_idx)
                };

                for &cid in clauses_to_increment.iter() {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    *num_good += 1;
                }

                for &cid in clauses_to_decrement.iter() {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    *num_good -= 1;
                    if *num_good == 0 {
                        residual_.push(cid);
                    }
                }

                *variables.get_unchecked_mut(v_idx) = !was_true;
            } else {
                break;
            }
            rounds += 1;
        }
    }
    return Ok(Some(Solution { variables }));
}

pub fn help() {
    println!("No help information available.");
}
