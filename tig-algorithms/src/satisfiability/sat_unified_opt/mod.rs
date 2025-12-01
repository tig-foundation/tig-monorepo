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

    let mut clauses = challenge.clauses.clone();
    let mut i = clauses.len();
    while i > 0 {
        i -= 1;
        {
            let clause = &mut clauses[i];
            if clause[0] == clause[2] || clause[1] == clause[2] {
                clause.pop();
            }
            if clause[0] == clause[1] {
                clause.swap_remove(1);
            }
        }
        
        let should_remove = {
            let clause = &clauses[i];
            (clause.len() >= 2 && clause[0] == -clause[1]) || 
            (clause.len() >= 3 && (clause[0] == -clause[2] || clause[1] == -clause[2]))
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

    let mut variables = vec![false; num_variables];
    let (pad, nad) = (1.8, 0.56);
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();
        
        let mut vad;
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
    
    let base_prob = 0.52;
    let mut current_prob = base_prob;

    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;
    let num_vars = challenge.num_variables as f64;
    let max_fuel = 2000000000.0;
    let base_fuel = (2000.0 + 40.0 * clauses_ratio) * num_vars;
    let flip_fuel = 350.0 + 0.9 * clauses_ratio;
    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;
    let mut rounds = 0;

    unsafe {
        loop {
            if !residual_.is_empty() {
                let rand_val = rng.gen::<usize>();

                let mut i = residual_.len() - 1;
                while !residual_.is_empty() {
                    let id = rand_val % residual_.len();
                    i = *residual_.get_unchecked(id);
                    if *num_good_so_far.get_unchecked(i) > 0 { 
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
                
                let v_idx = v.abs() as usize  - 1;
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
            // if rounds >= (max_num_rounds as f32 * 0.2) as usize {
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
