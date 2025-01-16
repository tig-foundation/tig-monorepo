/*!
Copyright 2024 Dominic Kennedy

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

*/

use tig_challenges::satisfiability::*;

use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.difficulty.num_variables;
    let mut variables = vec![false; num_variables];
    variables.iter_mut().for_each(|v| *v = rng.gen());

    let max_iterations = 50 * num_variables;
    let max_flips_without_improvement = num_variables / 2;
    let mut best_unsatisfied_clauses = challenge.clauses.len();
    let mut flips_without_improvement = 0;

    for _ in 0..max_iterations {
        let unsatisfied_clause_idx = challenge.clauses.iter().position(|clause| {
            !clause.iter().any(|&literal| {
                let var_idx = (literal.abs() - 1) as usize;
                let var_value = variables[var_idx];
                (literal > 0 && var_value) || (literal < 0 && !var_value)
            })
        });

        if let Some(idx) = unsatisfied_clause_idx {
            let mut best_var_idx = None;
            let mut min_unsatisfied_clauses = challenge.clauses.len();

            for &literal in &challenge.clauses[idx] {
                let var_idx = (literal.abs() - 1) as usize;
                variables[var_idx] = !variables[var_idx];

                let unsatisfied_clauses = challenge
                    .clauses
                    .iter()
                    .filter(|clause| {
                        !clause.iter().any(|&lit| {
                            let v_idx = (lit.abs() - 1) as usize;
                            let v_value = variables[v_idx];
                            (lit > 0 && v_value) || (lit < 0 && !v_value)
                        })
                    })
                    .count();

                if unsatisfied_clauses < min_unsatisfied_clauses {
                    min_unsatisfied_clauses = unsatisfied_clauses;
                    best_var_idx = Some(var_idx);
                }

                variables[var_idx] = !variables[var_idx];
            }

            if let Some(var_idx) = best_var_idx {
                variables[var_idx] = !variables[var_idx];
                if min_unsatisfied_clauses < best_unsatisfied_clauses {
                    best_unsatisfied_clauses = min_unsatisfied_clauses;
                    flips_without_improvement = 0;
                } else {
                    flips_without_improvement += 1;
                }
            } else {
                // If no variable flip improves the solution, randomly flip a variable
                let rand_var_idx = rng.gen_range(0..num_variables);
                variables[rand_var_idx] = !variables[rand_var_idx];
                flips_without_improvement = 0;
            }

            if flips_without_improvement >= max_flips_without_improvement {
                // Restart the search with a new random assignment
                variables.iter_mut().for_each(|v| *v = rng.gen());
                best_unsatisfied_clauses = challenge.clauses.len();
                flips_without_improvement = 0;
            }
        } else {
            return Ok(Some(Solution { variables }));
        }
    }

    if best_unsatisfied_clauses < challenge.clauses.len() / 10 {
        Ok(Some(Solution { variables }))
    } else {
        Ok(None)
    }
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
