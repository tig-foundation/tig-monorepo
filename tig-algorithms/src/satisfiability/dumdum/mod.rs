use rand::{rngs::SmallRng, SeedableRng, Rng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let vars = challenge.num_variables;
    let ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;

    if ratio >= 428.0 {
        return Ok(());
    }
    if ratio >= 425.0 && vars > 20000 {
        return Ok(());
    }
    if ratio >= 423.0 && vars > 30000 {
        return Ok(());
    }
    if vars > 70000 && ratio >= 421.0 {
        return Ok(());
    }

    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

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
        return Ok(());
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
    let avg_clause_size = clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64;
    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;

    let is_target_problem = num_variables >= 4500 && num_variables <= 5500
        && clauses_ratio >= 400.0 && clauses_ratio <= 450.0;

    let use_lower_variable_strategy = num_variables <= 7000;

    let nad = if use_lower_variable_strategy { 1.28 } else { 1.0 };

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
    let _ = save_solution(&Solution { variables });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
