use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

fn set_bit(arr: &mut [u64], index: usize, value: bool) {
    if value {
        arr[index / 64] |= 1 << (index % 64);
    } else {
        arr[index / 64] &= !(1 << (index % 64));
    }
}

fn get_bit(arr: &[u64], index: usize) -> bool {
    (arr[index / 64] & (1 << (index % 64))) != 0
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let num_bits = (challenge.num_variables + 63) / 64;
    let mut p_single = vec![0u64; num_bits];
    let mut n_single = vec![0u64; num_bits];

    let mut clauses_ = challenge.clauses.clone();
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses_.len());

    let mut rounds = 0;
    let mut dead = false;

    while !(dead) {
        let mut done = true;

        for c in &clauses_ {
            let mut c_: Vec<i32> = Vec::with_capacity(c.len());
            let mut skip = false;

            let mut checked_literals = HashMap::with_capacity(c.len());

            for &l in c {
                let var_idx = (l.abs() - 1) as usize;

                if let Some(&skip_result) = checked_literals.get(&l) {
                    if skip_result {
                        skip = true;
                        break;
                    } else {
                        continue;
                    }
                }

                if (get_bit(&p_single, var_idx) && l > 0)
                    || (get_bit(&n_single, var_idx) && l < 0)
                    || checked_literals.contains_key(&-l)
                {
                    skip = true;
                    checked_literals.insert(l, true);
                    break;
                } else if get_bit(&p_single, var_idx)
                    || get_bit(&n_single, var_idx)
                    || checked_literals.contains_key(&l)
                {
                    done = false;
                    continue;
                } else {
                    c_.push(l);
                    checked_literals.insert(l, false);
                }
            }

            if skip {
                done = false;
                continue;
            }

            match c_[..] {
                [l] => {
                    done = false;
                    let var_idx = (l.abs() - 1) as usize;

                    if l > 0 {
                        if get_bit(&n_single, var_idx) {
                            dead = true;
                            break;
                        } else {
                            set_bit(&mut p_single, var_idx, true);
                        }
                    } else {
                        if get_bit(&p_single, var_idx) {
                            dead = true;
                            break;
                        } else {
                            set_bit(&mut n_single, var_idx, true);
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

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        if get_bit(&p_single, v) {
            variables[v] = true
        } else if get_bit(&n_single, v) {
            variables[v] = false
        } else {
            variables[v] = rng.gen_bool(0.5)
        }
    }

    let mut num_good_so_far: Vec<usize> = vec![0; num_clauses];

    for c in &clauses {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 && p_clauses[var].is_empty() {
                p_clauses[var] = Vec::with_capacity(clauses.len() / num_variables + 1);
            } else if l < 0 && n_clauses[var].is_empty() {
                n_clauses[var] = Vec::with_capacity(clauses.len() / num_variables + 1);
            }
        }
    }

    for (i, &ref c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
                if variables[var] {
                    num_good_so_far[i] += 1
                }
            } else {
                n_clauses[var].push(i);
                if !variables[var] {
                    num_good_so_far[i] += 1
                }
            }
        }
    }

    let mut residual_ = Vec::with_capacity(num_clauses);
    let mut residual_indices = HashMap::with_capacity(num_clauses);

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
            residual_indices.insert(i, residual_.len() - 1);
        }
    }

    loop {
        if !residual_.is_empty() {
            let i = residual_[0];
            let mut min_sad = clauses.len();
            let mut v_min_sad = Vec::with_capacity(clauses[i].len());

            let c = &clauses[i];
            for &l in c {
                let mut sad = 0;
                let var_idx = (l.abs() - 1) as usize;

                if variables[var_idx] {
                    for &c in &p_clauses[var_idx] {
                        if num_good_so_far[c] == 1 {
                            sad += 1;
                            if sad > min_sad {
                                break;
                            }
                        }
                    }
                } else {
                    for &c in &n_clauses[var_idx] {
                        if num_good_so_far[c] == 1 {
                            sad += 1;
                            if sad > min_sad {
                                break;
                            }
                        }
                    }
                }

                if sad < min_sad {
                    min_sad = sad;
                    v_min_sad.clear();
                    v_min_sad.push(var_idx);
                } else if sad == min_sad {
                    v_min_sad.push(var_idx);
                }
            }

            let v = if min_sad == 0 {
                if v_min_sad.len() == 1 {
                    v_min_sad[0]
                } else {
                    v_min_sad[rng.gen_range(0..v_min_sad.len())]
                }
            } else {
                if rng.gen_bool(0.5) {
                    let l = c[rng.gen_range(0..c.len())];
                    (l.abs() - 1) as usize
                } else {
                    v_min_sad[rng.gen_range(0..v_min_sad.len())]
                }
            };

            if variables[v] {
                for &c in &n_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices.remove(&c).unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices.insert(last, i);
                        }
                    }
                }
                for &c in &p_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices.insert(c, residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }
            } else {
                for &c in &n_clauses[v] {
                    if num_good_so_far[c] == 1 {
                        residual_.push(c);
                        residual_indices.insert(c, residual_.len() - 1);
                    }
                    num_good_so_far[c] -= 1;
                }

                for &c in &p_clauses[v] {
                    num_good_so_far[c] += 1;
                    if num_good_so_far[c] == 1 {
                        let i = residual_indices.remove(&c).unwrap();
                        let last = residual_.pop().unwrap();
                        if i < residual_.len() {
                            residual_[i] = last;
                            residual_indices.insert(last, i);
                        }
                    }
                }
            }

            variables[v] = !variables[v];
        } else {
            break;
        }

        rounds += 1;
        if rounds >= num_variables * 35 {
            return Ok(());
        }
    }

    let _ = save_solution(&Solution { variables });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
