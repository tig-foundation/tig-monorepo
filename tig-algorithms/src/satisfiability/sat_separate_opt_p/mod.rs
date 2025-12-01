use rand::{rngs::{SmallRng, StdRng}, Rng, SeedableRng};
use std::collections::HashMap;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let mut p_single = vec![false; challenge.num_variables];
    let mut n_single = vec![false; challenge.num_variables];

    let mut clauses_ = challenge.clauses.clone();
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses_.len());

    let mut rounds = 0;

    let mut dead = false;

    while !(dead) {
        let mut done = true;
        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len()); // Preallocate with capacity
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

    // Preallocate capacity for p_clauses and n_clauses
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
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();

        let nad = 1.28;
        let mut vad = nad + 1.0;
        if num_n > 0 {
            vad = num_p as f32 / num_n as f32;
        }

        if vad <= nad {
            variables[v] = false;
        } else {
            let prob = num_p as f64 / (num_p + num_n).max(1) as f64;
            variables[v] = rng.gen_bool(prob)
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
    let check_interval = 50;
    let mut last_check_residual = residual_.len();

    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;
    let num_vars = challenge.num_variables as f64;
    let max_fuel = 2000000000.0;
    let base_fuel = (2000.0 + 40.0 * clauses_ratio) * num_vars;
    let flip_fuel = 350.0 + 0.9 * clauses_ratio;
    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;
    unsafe {
        loop {
            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();

                let mut i = residual_.len() - 1;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = *residual_.get_unchecked(id);
                    if num_good_so_far[i] > 0 { 
                        residual_.swap_remove(id); 
                    } else {
                        break
                    }
                }
                if residual_.is_empty() {
                    break;
                }
                 
                let mut min_sad = clauses.len();
                let mut v_min_sad = usize::MAX;
                let c = clauses.get_unchecked_mut(i);

                
                if (challenge.clauses.len() * 100 / challenge.num_variables) >= 410 {
                    if rounds % check_interval == 0 {
                        let progress = last_check_residual as i64 - residual_.len() as i64;
                        let progress_ratio = progress as f64 / last_check_residual as f64;
                        
                        let progress_threshold = 0.2 + 0.1 * f64::min(1.0, (clauses_ratio - 410.0) / 15.0);
        
                        if progress <= 0 {
                            let prob_adjustment = 0.025 * (-progress as f64 / last_check_residual as f64).min(1.0);
                            current_prob = (current_prob + prob_adjustment).min(0.9);
                        } else if progress_ratio > progress_threshold { 
                            current_prob = base_prob;
                        } else {
                            current_prob = current_prob * 0.8 + base_prob * 0.2;
                        }
                        
                        last_check_residual = residual_.len();
                    }
                }
                
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
                           if sad >= min_sad {
                               break;
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
            // if rounds >= (max_num_rounds as f32 * 1.0) as usize {
            //     return Ok(());
            // }
        }
     }
    let _ = save_solution(&Solution { variables });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
