/*!
Copyright 2024 Rootz

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use tig_challenges::knapsack::*;
use rand::prelude::*;
use std::cmp::{max, min};

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight as usize;
    let num_items = challenge.difficulty.num_items;
    let weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
    let values: Vec<i32> = challenge.values.iter().map(|&v| v as i32).collect();
    let interaction_values = &challenge.interaction_values;

    let mut rng = StdRng::seed_from_u64(challenge.seed[0] as u64);

    let mut best_solution = None;
    let mut best_value = 0;

    for _ in 0..5 {
        let mut solution = vec![false; num_items];
        let mut current_weight = 0;
        let mut current_value = 0;

        for i in 0..num_items {
            if current_weight + weights[i] <= max_weight && rng.gen_bool(0.5) {
                solution[i] = true;
                current_weight += weights[i];
                current_value += values[i];
            }
        }

        let num_iterations = 500 + num_items.saturating_sub(100);
        for _ in 0..num_iterations {
            let i = rng.gen_range(0..num_items);
            let j = rng.gen_range(0..num_items);
            if i != j && solution[i] != solution[j] {
                let new_weight = current_weight + weights[j] - weights[i];
                if new_weight <= max_weight {
                    let old_value = current_value;
                    let mut delta = values[j] - values[i];
                    
                    if solution[i] {
                        for k in 0..num_items {
                            if k != i && k != j && solution[k] {
                                delta -= interaction_values[min(i, k)][max(i, k)];
                                delta += interaction_values[min(j, k)][max(j, k)];
                            }
                        }
                    } else {
                        for k in 0..num_items {
                            if k != i && k != j && solution[k] {
                                delta += interaction_values[min(i, k)][max(i, k)];
                                delta -= interaction_values[min(j, k)][max(j, k)];
                            }
                        }
                    }

                    if delta > 0 {
                        solution.swap(i, j);
                        current_weight = new_weight;
                        current_value += delta;
                    } else {
                        current_value = old_value;
                    }
                }
            }
        }

        if current_value > best_value {
            best_value = current_value;
            best_solution = Some(solution);
        }

        if best_value >= challenge.min_value as i32 {
            break;
        }
    }

    if let Some(solution) = best_solution {
        if best_value >= challenge.min_value as i32 {
            Ok(Some(Solution {
                items: solution.iter().enumerate().filter_map(|(i, &included)| if included { Some(i) } else { None }).collect(),
            }))
        } else {
            Ok(None)
        }
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

    pub const KERNEL: Option<CudaKernel> = None;

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
