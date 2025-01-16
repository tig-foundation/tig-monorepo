/*!
Copyright 2024 Dominic Kennedy

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

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

    let mut rand_ints = vec![0usize; 2 * num_variables];
    rand_ints
        .iter_mut()
        .for_each(|i| *i = rng.gen_range(0..1_000_000_000u32) as usize);

    let max_iterations = 100 * num_variables;
    let max_iterations_without_improvement = num_variables / 2;
    let mut best_unsatisfied_clauses = challenge.clauses.len();
    let mut iterations_without_improvement = 0;

    for i in 0..max_iterations {
        let unsatisfied_clause_idx = challenge.clauses.iter().position(|clause| {
            !clause.iter().any(|&literal| {
                let var_idx = (literal.abs() - 1) as usize;
                let var_value = variables[var_idx];
                (literal > 0 && var_value) || (literal < 0 && !var_value)
            })
        });

        if let Some(idx) = unsatisfied_clause_idx {
            let rand_variable_idx = rand_ints[i % (2 * num_variables)] % 3;
            let rand_variable = challenge.clauses[idx][rand_variable_idx].abs() as usize - 1;
            variables[rand_variable] = !variables[rand_variable];

            let unsatisfied_clauses = challenge
                .clauses
                .iter()
                .filter(|clause| {
                    !clause.iter().any(|&literal| {
                        let var_idx = (literal.abs() - 1) as usize;
                        let var_value = variables[var_idx];
                        (literal > 0 && var_value) || (literal < 0 && !var_value)
                    })
                })
                .count();

            if unsatisfied_clauses < best_unsatisfied_clauses {
                best_unsatisfied_clauses = unsatisfied_clauses;
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement += 1;
            }

            if iterations_without_improvement >= max_iterations_without_improvement {
                iterations_without_improvement = 0;
                variables[rand_variable] = !variables[rand_variable];
            }
        } else {
            return Ok(Some(Solution { variables }));
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
