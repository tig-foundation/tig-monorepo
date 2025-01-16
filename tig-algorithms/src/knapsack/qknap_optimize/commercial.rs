/*!
Copyright 2024 ByteBandit

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use tig_challenges::knapsack::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let min_value = challenge.min_value as i32;
    let max_weight = challenge.max_weight;
    let num_items = challenge.difficulty.num_items;
    let weights = &challenge.weights;
    let values = &challenge.values;
    let interaction_values = &challenge.interaction_values;

    let mut current_solution = vec![false; num_items];
    let mut total_weight = 0;
    let mut total_value = 0;

    let mut sorted_items: Vec<(usize, f32)> = (0..num_items)
        .map(|i| {
            let interaction_value_sum: f32 = interaction_values[i].iter().map(|&v| v as f32).sum();
            let ratio = (values[i] as f32 + interaction_value_sum) / weights[i] as f32;
            (i, ratio)
        })
        .collect();
    sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (item_index, _) in sorted_items {
        if total_weight + weights[item_index] <= max_weight {
            current_solution[item_index] = true;
            total_weight += weights[item_index];
            total_value += values[item_index] as i32;

            for i in 0..num_items {
                if current_solution[i] {
                    total_value += interaction_values[item_index][i];
                }
            }
        }
    }

    if total_value < min_value {
        total_value = optimize_solution(
            &mut current_solution,
            num_items,
            weights,
            values,
            interaction_values,
            min_value,
            max_weight,
            total_weight,
            total_value,
        );
    }

    if total_value >= min_value {
        Ok(Some(Solution {
            items: current_solution
                .iter()
                .enumerate()
                .filter_map(|(i, &included)| if included { Some(i) } else { None })
                .collect(),
        }))
    } else {
        Ok(None)
    }
}

fn optimize_solution(
    solution: &mut Vec<bool>,
    num_items: usize,
    weights: &Vec<u32>,
    values: &Vec<u32>,
    interaction_values: &Vec<Vec<i32>>,
    min_value: i32,
    max_weight: u32,
    total_weight: u32,
    total_value: i32,
) -> i32 {
    let mut best_value = total_value;
    let mut current_value = best_value;
    let mut current_weight = total_weight;
    let mut improved = true;

    while improved {
        improved = false;

        for i in 0..num_items {
            for j in (i + 1)..num_items {
                if solution[i] != solution[j] {
                    let new_weight = if solution[i] {
                        current_weight - weights[i] + weights[j]
                    } else {
                        current_weight + weights[i] - weights[j]
                    };

                    if new_weight <= max_weight {
                        let delta_value = calculate_delta_value(
                            solution,
                            num_items,
                            i,
                            j,
                            values,
                            interaction_values,
                        );

                        let new_value = current_value + delta_value;
                        if new_value > best_value {
                            best_value = new_value;
                            current_value = new_value;
                            current_weight = new_weight;
                            solution.swap(i, j);
                            improved = true;
                            if best_value >= min_value {
                                return best_value;
                            }
                        }
                    }
                }
            }
        }
    }

    best_value
}

#[inline]
fn calculate_delta_value(
    solution: &Vec<bool>,
    num_items: usize,
    i: usize,
    j: usize,
    values: &Vec<u32>,
    interaction_values: &Vec<Vec<i32>>,
) -> i32 {
    let mut delta_value = 0;

    if solution[i] && !solution[j] {
        delta_value += (values[j] as i32) - (values[i] as i32);
        for k in 0..num_items {
            if k != i && k != j && solution[k] {
                delta_value += interaction_values[j][k] - interaction_values[i][k];
            }
        }
    } else if !solution[i] && solution[j] {
        delta_value -= (values[j] as i32) - (values[i] as i32);
        for k in 0..num_items {
            if k != i && k != j && solution[k] {
                delta_value += interaction_values[i][k] - interaction_values[j][k];
            }
        }
    }

    delta_value
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
