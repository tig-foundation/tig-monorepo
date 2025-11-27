use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde_json::{Map, Value};
use std::collections::HashMap;
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution {
        variables: (0..challenge.num_variables).map(|_| false).collect(),
    });

    let mut rng =
        SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

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
    let mut residual_indices = vec![None; num_clauses];

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
            residual_indices[i] = Some(residual_.len() - 1);
        }
    }

    let clauses_ratio = 4.267;
    let num_vars = challenge.num_variables as f64;
    let max_fuel = 2000000000.0;
    let base_fuel = (2000.0 + 40.0 * clauses_ratio) * num_vars;
    let flip_fuel = 350.0 + 0.9 * clauses_ratio;
    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;
    loop {
        if !residual_.is_empty() {
            let rand_val = rng.gen::<usize>();

            let i = residual_[rand_val % residual_.len()];
            let mut min_sad = clauses.len();
            let mut v_min_sad = usize::MAX;
            let c = &mut clauses[i];

            if c.len() > 1 {
                let random_index = rand_val % c.len();
                c.swap(0, random_index);
            }
            for &l in c.iter() {
                let abs_l = l.abs() as usize - 1;
                let clauses_to_check = if variables[abs_l] {
                    &p_clauses[abs_l]
                } else {
                    &n_clauses[abs_l]
                };

                let mut sad = 0;
                for &c in clauses_to_check {
                    if num_good_so_far[c] == 1 {
                        sad += 1;
                    }
                }

                if sad < min_sad {
                    min_sad = sad;
                    v_min_sad = abs_l;
                }
            }

            let v = if min_sad == 0 {
                v_min_sad
            } else if rng.gen_bool(0.5) {
                c[0].abs() as usize - 1
            } else {
                v_min_sad
            };

            if variables[v] {
                for &c in &n_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices[c].take().unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices[last] = Some(i);
                        }
                    }
                }
                for &c in &p_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices[c] = Some(residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }
            } else {
                for &c in &n_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices[c] = Some(residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }

                for &c in &p_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices[c].take().unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices[last] = Some(i);
                        }
                    }
                }
            }

            variables[v] = !variables[v];
        } else {
            break;
        }
        rounds += 1;
        if rounds >= max_num_rounds {
            return Ok(());
        }
    }
    let _ = save_solution(&Solution { variables });
    return Ok(());
}


pub fn help() {
    println!("No help information provided.");
}

