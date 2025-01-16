/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Commercial License v1.0 (the "License"); you 
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

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.difficulty.num_variables;
    
    // Pre-allocate vectors to avoid reallocations
    let mut variables = vec![false; num_variables];
    let mut p_single = vec![false; num_variables];
    let mut n_single = vec![false; num_variables];
    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::with_capacity(challenge.clauses.len() / num_variables); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::with_capacity(challenge.clauses.len() / num_variables); num_variables];

    // Use HashSet for faster lookups
    let mut clause_set: HashSet<Vec<i32>> = HashSet::with_capacity(challenge.clauses.len());
    let mut simplified_clauses = Vec::with_capacity(challenge.clauses.len());

    // Simplify clauses
    for clause in &challenge.clauses {
        let mut simplified = clause.clone();
        simplified.sort_unstable();
        simplified.dedup();
        
        if simplified.iter().any(|&l| simplified.contains(&-l)) {
            continue; // Tautology, skip this clause
        }
        
        if clause_set.insert(simplified.clone()) {
            simplified_clauses.push(simplified);
        }
    }

    // Unit propagation and simplification
    let mut dead = false;
    loop {
        let mut changed = false;
        let mut i = 0;
        while i < simplified_clauses.len() {
            match simplified_clauses[i].len() {
                0 => {
                    dead = true;
                    break;
                }
                1 => {
                    let l = simplified_clauses[i][0];
                    let var = (l.abs() - 1) as usize;
                    if (l > 0 && n_single[var]) || (l < 0 && p_single[var]) {
                        dead = true;
                        break;
                    }
                    if l > 0 {
                        p_single[var] = true;
                    } else {
                        n_single[var] = true;
                    }
                    simplified_clauses.swap_remove(i);
                    changed = true;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
        
        if dead || !changed {
            break;
        }
        
        // Apply unit propagation
        simplified_clauses.retain_mut(|clause| {
            clause.retain(|&l| {
                let var = (l.abs() - 1) as usize;
                !(p_single[var] && l > 0) && !(n_single[var] && l < 0)
            });
            !clause.is_empty()
        });
    }

    if dead {
        return Ok(None);
    }

    // Initialize variables
    for v in 0..num_variables {
        if p_single[v] {
            variables[v] = true;
        } else if n_single[v] {
            variables[v] = false;
        } else {
            variables[v] = rng.gen_bool(0.5);
        }
    }

    // Build clause index
    for (i, clause) in simplified_clauses.iter().enumerate() {
        for &l in clause {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
            } else {
                n_clauses[var].push(i);
            }
        }
    }

    let mut num_good_so_far = vec![0; simplified_clauses.len()];
    let mut residual = Vec::new();

    // Initialize num_good_so_far and residual
    for (i, clause) in simplified_clauses.iter().enumerate() {
        for &l in clause {
            let var = (l.abs() - 1) as usize;
            if (l > 0 && variables[var]) || (l < 0 && !variables[var]) {
                num_good_so_far[i] += 1;
            }
        }
        if num_good_so_far[i] == 0 {
            residual.push(i);
        }
    }

    // Main solving loop
    let max_attempts = num_variables * 25;
    for _ in 0..max_attempts {
        if residual.is_empty() {
            return Ok(Some(Solution { variables }));
        }

        let i = residual[0];
        let clause = &simplified_clauses[i];
        
        // Choose variable to flip
        let v = if rng.gen_bool(0.9) {
            // Greedy choice
            clause.iter()
                .map(|&l| (l.abs() - 1) as usize)
                .min_by_key(|&var| {
                    let clauses = if variables[var] { &p_clauses[var] } else { &n_clauses[var] };
                    clauses.iter().filter(|&&c| num_good_so_far[c] == 1).count()
                })
                .unwrap()
        } else {
            // Random choice
            let l = clause[rng.gen_range(0..clause.len())];
            (l.abs() - 1) as usize
        };

        // Flip variable
        variables[v] = !variables[v];
        
        // Update num_good_so_far and residual
        let (pos_clauses, neg_clauses) = if variables[v] {
            (&p_clauses[v], &n_clauses[v])
        } else {
            (&n_clauses[v], &p_clauses[v])
        };

        for &c in pos_clauses {
            num_good_so_far[c] += 1;
            if num_good_so_far[c] == 1 {
                residual.retain(|&x| x != c);
            }
        }

        for &c in neg_clauses {
            num_good_so_far[c] -= 1;
            if num_good_so_far[c] == 0 {
                residual.push(c);
            }
        }
    }

    Ok(None)
}
#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
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
