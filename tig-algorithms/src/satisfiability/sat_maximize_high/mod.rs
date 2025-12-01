use rand::{rngs::SmallRng, SeedableRng, Rng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Clone)]
enum SolverConfig {
    Original,
    MoreRandom,
    MoreGreedy,
    FocusShortClauses,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    
    if let Some(solution) = attempt_solve(challenge, &mut rng, SolverConfig::Original)? {
        let _ = save_solution(&solution);
        return Ok(());
    }
    
    let configs = [
        SolverConfig::MoreRandom,
        SolverConfig::MoreGreedy,
        SolverConfig::FocusShortClauses,
    ];
    
    for config in &configs {
        let seed = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64;
        let new_seed = seed.wrapping_add(match config {
            SolverConfig::MoreRandom => 42,
            SolverConfig::MoreGreedy => 137,
            SolverConfig::FocusShortClauses => 271,
            _ => 0,
        });
        
        let mut config_rng = SmallRng::seed_from_u64(new_seed);
        
        if let Some(solution) = attempt_solve(challenge, &mut config_rng, config.clone())? {
            let _ = save_solution(&solution);
            return Ok(());
        }
    }
    
    let max_restarts = 3;
    
    for restart in 0..max_restarts {
        let seed = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64;
        let new_seed = seed.wrapping_add(restart as u64 + 1000);
        let mut restart_rng = SmallRng::seed_from_u64(new_seed);
        
        if let Some(solution) = attempt_solve(challenge, &mut restart_rng, SolverConfig::Original)? {
            let _ = save_solution(&solution);
            return Ok(());
        }
    }
    
    Ok(())
}

fn attempt_solve(challenge: &Challenge, rng: &mut SmallRng, config: SolverConfig) -> anyhow::Result<Option<Solution>> {
    let mut p_single = vec![false; challenge.num_variables];
    let mut n_single = vec![false; challenge.num_variables];

    let mut clauses_ = challenge.clauses.clone();
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses_.len());

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

    let mut variables = vec![false; num_variables];
    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;
    let nad = 1.28 + (clauses_ratio - 400.0) * 0.001 * (1.0 + (num_variables as f64 / 100000.0).ln().max(0.0));
    
    let randomize_factor = match config {
        SolverConfig::MoreRandom => 0.3,
        _ => 0.0,
    };
    
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();

        let p_size = p_clauses[v].iter().map(|&i| clauses[i].len()).sum::<usize>();
        let n_size = n_clauses[v].iter().map(|&i| clauses[i].len()).sum::<usize>();
        
        let mut vad = nad + 1.0;
        
        if num_n > 0 {
            let count_ratio = num_p as f64 / num_n as f64;
            let size_ratio = if n_size > 0 { p_size as f64 / n_size as f64 } else { count_ratio };
            vad = 0.7 * count_ratio + 0.3 * size_ratio;
        }

        if vad <= nad {
            variables[v] = if rng.gen_bool(randomize_factor) {
                rng.gen_bool(0.5)
            } else {
                false
            };
        } else {
            let prob = if num_p + num_n == 0 {
                0.5
            } else {
                let base_prob = num_p as f64 / (num_p + num_n) as f64;
                let size_prob = p_size as f64 / (p_size + n_size).max(1) as f64;
                0.7 * base_prob + 0.3 * size_prob
            };
            
            variables[v] = if rng.gen_bool(randomize_factor) {
                rng.gen_bool(0.5)
            } else {
                rng.gen_bool(prob)
            };
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
    let mut residual_indices = vec![usize::MAX; num_clauses];

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
            residual_indices[i] = residual_.len() - 1;
        }
    }
    
    let max_fuel = 10_000_000_000.0;
    let base_fuel = (1000.0 + 20.0 * clauses_ratio) * (num_variables as f64).sqrt();
    let flip_fuel = (100.0 + 0.5 * clauses_ratio) * 0.8;
    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;
    let mut rounds = 0;

    unsafe {
        loop {
            if rounds >= max_num_rounds {
                return Ok(None);
            }

            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();
                
                let random_threshold = match config {
                    SolverConfig::MoreRandom => 0.85f64,
                    SolverConfig::MoreGreedy => 0.5f64,
                    _ => 0.7f64,
                };
                
                let i = if rand_val < (random_threshold * (usize::MAX as f64)) as usize {
                    *residual_.get_unchecked(rand_val % residual_.len())
                } else {
                    let num_samples = match config {
                        SolverConfig::FocusShortClauses => (10.0 * (1.0 + (num_variables as f64 / 40000.0).ln())).min(20.0).max(5.0) as usize,
                        _ => (5.0 * (1.0 + (num_variables as f64 / 40000.0).ln())).min(10.0).max(2.0) as usize,
                    };
                    
                    if residual_.len() <= num_samples {
                        *residual_.get_unchecked(rand_val % residual_.len())
                    } else {
                        let step_size = residual_.len() / num_samples;
                        if step_size == 0 {
                            *residual_.get_unchecked(rand_val % residual_.len())
                        } else {
                            let start = rand_val % step_size;
                            let mut shortest_i = *residual_.get_unchecked(start);
                            let mut shortest_len = clauses.get_unchecked(shortest_i).len();
                            
                            for j in 1..num_samples {
                                let idx = (start + j * step_size) % residual_.len();
                                let curr_i = *residual_.get_unchecked(idx);
                                let curr_len = clauses.get_unchecked(curr_i).len();
                                if curr_len < shortest_len {
                                    shortest_i = curr_i;
                                    shortest_len = curr_len;
                                }
                            }
                            shortest_i
                        }
                    }
                };
                
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
                
                let random_var_threshold = match config {
                    SolverConfig::MoreRandom => 0.7f64,
                    SolverConfig::MoreGreedy => 0.3f64,
                    _ => 0.52f64,
                };
                
                let v = if let Some(abs_l) = zero_found {
                    abs_l
                } else if rand_val < ((random_var_threshold + 0.1 * (num_variables as f64 / 100000.0).min(0.2)) 
                                    * (usize::MAX as f64)) as usize {
                    c[0].abs() as usize - 1
                } else {
                    let mut min_sad = usize::MAX;
                    let mut v_min_sad = c[0].abs() as usize - 1;
                    
                    for &l in c.iter() {
                        let abs_l = l.abs() as usize - 1;
                        
                        let clauses_to_check = if *variables.get_unchecked(abs_l) {
                            p_clauses.get_unchecked(abs_l)
                        } else {
                            n_clauses.get_unchecked(abs_l)
                        };
                        
                        let mut sad = 0;
                        for &c in clauses_to_check {
                            if *num_good_so_far.get_unchecked(c) == 1 {
                                sad += 1;
                            }
                        }
                        
                        if sad < min_sad {
                            min_sad = sad;
                            v_min_sad = abs_l;
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
                    if *num_good == 0 {
                        let i = *residual_indices.get_unchecked(cid);
                        *residual_indices.get_unchecked_mut(cid) = usize::MAX;
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            *residual_.get_unchecked_mut(i) = last;
                            *residual_indices.get_unchecked_mut(last) = i;
                        }
                    }
                    *num_good += 1;
                }
        
                for &cid in clauses_to_decrement {
                    let num_good = num_good_so_far.get_unchecked_mut(cid);
                    *num_good -= 1;
                    if *num_good == 0 {
                        residual_.push(cid);
                        *residual_indices.get_unchecked_mut(cid) = residual_.len() - 1;
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

pub fn help() {
    println!("No help information available.");
}
