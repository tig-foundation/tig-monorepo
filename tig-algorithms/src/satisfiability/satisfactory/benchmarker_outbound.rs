/*!
Copyright 2024 KAJSHU

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use tig_challenges::satisfiability::*;

/// Preprocess clauses with simplification and unit propagation.
fn preprocess_clauses(clauses: &mut Vec<Vec<i32>>, num_variables: usize) -> (Vec<bool>, Vec<bool>) {
    let mut p_single = vec![false; num_variables];
    let mut n_single = vec![false; num_variables];
    
    let mut clauses_ = clauses.clone();
    let mut simplified_clauses = Vec::with_capacity(clauses.len());

    let mut dead = false;
    let mut done = false;

    while !dead && !done {
        done = true;
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
            }
            match c_.as_slice() {
                [l] => {
                    done = false;
                    if *l > 0 {
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
                    simplified_clauses.push(c_);
                }
            }
        }
        if done {
            break;
        } else {
            clauses_ = simplified_clauses;
            simplified_clauses = Vec::with_capacity(clauses_.len());
        }
    }

    *clauses = simplified_clauses;
    (p_single, n_single)
}

/// Conflict analysis to generate learned clauses.
fn analyze_conflict(residual_: &Vec<usize>, clauses: &Vec<Vec<i32>>) -> Vec<i32> {
    let mut conflict_clause = Vec::new();
    for &c in residual_ {
        conflict_clause.push(clauses[c][0]);
    }
    conflict_clause
}

/// Perform clause minimization.
fn minimize_clause(clause: Vec<i32>, num_variables: usize) -> Vec<i32> {
    let mut minimized_clause = clause.clone();
    minimized_clause.sort();
    minimized_clause.dedup();
    minimized_clause
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let num_variables = challenge.difficulty.num_variables;

    // Preprocessing the clauses
    let mut clauses = challenge.clauses.clone();
    let (mut p_single, mut n_single) = preprocess_clauses(&mut clauses, num_variables);

    if clauses.is_empty() {
        return Ok(Some(Solution { variables: vec![false; num_variables] }));
    }

    let mut decision_stack: Vec<(usize, bool)> = Vec::new();
    let mut learned_clauses: HashSet<Vec<i32>> = HashSet::new();
    let mut rounds = 0;
    let mut max_rounds = num_variables * 35;

    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        if p_single[v] {
            variables[v] = true;
        } else if n_single[v] {
            variables[v] = false;
        } else {
            variables[v] = rng.gen_bool(0.5);
        }
    }
    let mut num_good_so_far: Vec<usize> = vec![0; clauses.len()];

    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];

    for (i, c) in clauses.iter().enumerate() {
        for &l in c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
                if variables[var] {
                    num_good_so_far[i] += 1;
                }
            } else {
                n_clauses[var].push(i);
                if !variables[var] {
                    num_good_so_far[i] += 1;
                }
            }
        }
    }

    let mut residual_ = Vec::with_capacity(clauses.len());
    let mut residual_indices = HashMap::with_capacity(clauses.len());

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_.push(i);
            residual_indices.insert(i, residual_.len() - 1);
        }
    }

    while !residual_.is_empty() {
        let i = residual_[0];
        let mut min_sad = clauses.len();
        let mut v_min_sad = Vec::with_capacity(clauses[i].len());
        let c = &clauses[i];
        for &l in c {
            let mut sad = 0;
            if variables[(l.abs() - 1) as usize] {
                for &c in &p_clauses[(l.abs() - 1) as usize] {
                    if num_good_so_far[c] == 1 {
                        sad += 1;
                        if sad > min_sad {
                            break;
                        }
                    }
                }
            } else {
                for &c in &n_clauses[(l.abs() - 1) as usize] {
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
                v_min_sad.push((l.abs() - 1) as usize);
            } else if sad == min_sad {
                v_min_sad.push((l.abs() - 1) as usize);
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
                    let idx = residual_indices.remove(&c).unwrap();
                    let last = residual_.pop().unwrap();
                    if idx < residual_.len() {
                        residual_[idx] = last;
                        residual_indices.insert(last, idx);
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
                    let idx = residual_indices.remove(&c).unwrap();
                    let last = residual_.pop().unwrap();
                    if idx < residual_.len() {
                        residual_[idx] = last;
                        residual_indices.insert(last, idx);
                    }
                }
            }
        }

        variables[v] = !variables[v];
        rounds += 1;

        // Conflict-Driven Learning
        if rounds % 100 == 0 {
            let conflict_clause = analyze_conflict(&residual_, &clauses);
            let minimized_clause = minimize_clause(conflict_clause, num_variables);
            learned_clauses.insert(minimized_clause);
        }

        // Restart Strategy
        if rounds >= max_rounds {
            // Reset the state
            rounds = 0;
            max_rounds = (max_rounds as f64 * 1.05).ceil() as usize;
            p_single.fill(false);
            n_single.fill(false);
            variables.fill(false);
            clauses.extend(learned_clauses.iter().cloned());
            learned_clauses.clear();
            // Randomly initialize variables
            for i in 0..num_variables {
                if rng.gen_bool(0.5) {
                    variables[i] = true;
                }
            }
            // Reinitialize clauses
            let (new_p_single, new_n_single) = preprocess_clauses(&mut clauses, num_variables);
            p_single = new_p_single;
            n_single = new_n_single;
        }
    }

    Ok(Some(Solution { variables }))
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // Set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! Your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};