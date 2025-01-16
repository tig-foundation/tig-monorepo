/*!
Copyright 2024 Crypti (PTY) LTD

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;  
use tig_challenges::satisfiability::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.difficulty.num_variables;
    let mut variables = vec![false; num_variables];
    let mut clauses = challenge.clauses.clone();
    let num_clauses = clauses.len();
    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::with_capacity(num_clauses / num_variables); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::with_capacity(num_clauses / num_variables); num_variables];
    let mut num_good_so_far = vec![0; num_clauses];
    let mut residual = Vec::with_capacity(num_clauses);
    let mut residual_set = vec![false; num_clauses];

    // Initialize data structures
    for (i, clause) in clauses.iter().enumerate() {
        for &l in clause {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
            } else {
                n_clauses[var].push(i);
            }
        }
    }

    // Initial random assignment
    for v in 0..num_variables {
        variables[v] = rng.gen_bool(0.5);
        if variables[v] {
            for &c in &p_clauses[v] {
                num_good_so_far[c] += 1;
            }
        } else {
            for &c in &n_clauses[v] {
                num_good_so_far[c] += 1;
            }
        }
    }

    // Initialize residual
    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual.push(i);
            residual_set[i] = true;
        }
    }

    let max_attempts = num_variables * 20;
    for attempt in 0..max_attempts {
        if residual.is_empty() {
            return Ok(Some(Solution { variables }));
        }

        let i = residual[0];
        let c = &clauses[i];
        
        // Choose variable to flip
        let v = if rng.gen_bool(0.7) {
            // Choose randomly from the clause
            let l = c[rng.gen_range(0..c.len())];
            (l.abs() - 1) as usize
        } else {
            // Choose the variable that minimizes the number of newly unsatisfied clauses
            c.iter()
                .map(|&l| {
                    let var = (l.abs() - 1) as usize;
                    let sad = if variables[var] {
                        p_clauses[var].iter().filter(|&&c| num_good_so_far[c] == 1).count()
                    } else {
                        n_clauses[var].iter().filter(|&&c| num_good_so_far[c] == 1).count()
                    };
                    (var, sad)
                })
                .min_by_key(|&(_, sad)| sad)
                .unwrap().0
        };

        // Flip the chosen variable
        variables[v] = !variables[v];
        if variables[v] {
            for &c in &p_clauses[v] {
                num_good_so_far[c] += 1;
                if num_good_so_far[c] == 1 && residual_set[c] {
                    residual_set[c] = false;
                    residual.retain(|&x| x != c);
                }
            }
            for &c in &n_clauses[v] {
                num_good_so_far[c] -= 1;
                if num_good_so_far[c] == 0 && !residual_set[c] {
                    residual_set[c] = true;
                    residual.push(c);
                }
            }
        } else {
            for &c in &p_clauses[v] {
                num_good_so_far[c] -= 1;
                if num_good_so_far[c] == 0 && !residual_set[c] {
                    residual_set[c] = true;
                    residual.push(c);
                }
            }
            for &c in &n_clauses[v] {
                num_good_so_far[c] += 1;
                if num_good_so_far[c] == 1 && residual_set[c] {
                    residual_set[c] = false;
                    residual.retain(|&x| x != c);
                }
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