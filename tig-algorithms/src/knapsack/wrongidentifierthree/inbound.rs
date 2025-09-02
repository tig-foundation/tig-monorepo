/*!
Copyright 2024 VNX

Identity of Submitter VNX

UAI null

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use anyhow::Result;
use tig_challenges::knapsack::{Challenge, Solution};
pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    const WAIT_ITERATIONS: usize = 5;
    const MAX_STAGNANT_ITERATIONS: usize = 5;

    let num_items = challenge.weights.len();
    let mut selected_items = vec![false; num_items];
    let mut total_value: i32 = 0;
    let mut total_weight: u32 = 0;
    let mut wait_map = vec![None; num_items];
    let values: Vec<i32> = challenge.values.iter().map(|&v| v as i32).collect();
    let weights: Vec<f64> = challenge.weights.iter().map(|&w| w as f64).collect();

    let mut items_by_ratio: Vec<(usize, f64)> = (0..num_items)
        .map(|i| {
            let adjusted_value = values[i];
            let ratio = adjusted_value as f64 / weights[i];
            (i, ratio)
        })
        .collect();
    items_by_ratio.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut interaction_gains = vec![0; num_items];

    let mut iteration_count = 0;
    let mut stagnant_iterations = 0;
    let mut max_total_value = total_value;

    loop {
        iteration_count += 1;

        for entry in &mut wait_map {
            if let Some(iter) = entry {
                if *iter <= iteration_count {
                    *entry = None;
                }
            }
        }

        let mut available_items: Vec<_> = items_by_ratio
            .iter()
            .filter(|(i, _)| !selected_items[*i] && wait_map[*i].is_none())
            .collect();

        let mut improvement_found = false;
        let mut index = 0;

        while index < available_items.len() {
            let (i, _) = available_items[index];
            let individual_value = values[*i];
            let interaction_gain = interaction_gains[*i];
            let gain = individual_value + interaction_gain;
            if gain >= individual_value {
                selected_items[*i] = true;
                total_value += gain;
                total_weight += challenge.weights[*i];

                for j in 0..num_items {
                    interaction_gains[j] += challenge.interaction_values[*i][j];
                }

                improvement_found = true;
                available_items.remove(index);
            } else {
                index += 1;
            }
        }

        if !improvement_found {
            for &(i, _) in &available_items {
                let new_item_value = values[*i] + interaction_gains[*i];
                let new_item_weight = challenge.weights[*i];

                if new_item_value <= values[*i] {
                    continue;
                }

                for j in 0..num_items {
                    if selected_items[j] {
                        let removal_loss = values[j] + interaction_gains[j];
                        if total_value + new_item_value - removal_loss > total_value {
                            for k in 0..num_items {
                                interaction_gains[k] -= challenge.interaction_values[j][k];
                            }
                            selected_items[j] = false;
                            total_value -= removal_loss;
                            total_weight -= challenge.weights[j];

                            selected_items[*i] = true;
                            total_value += new_item_value;
                            total_weight += new_item_weight;

                            for k in 0..num_items {
                                interaction_gains[k] += challenge.interaction_values[*i][k];
                            }

                            wait_map[j] = Some(iteration_count + WAIT_ITERATIONS);
                            improvement_found = true;
                            break;
                        }
                    }
                }

                if improvement_found {
                    break;
                } else {
                    return Ok(None);
                }
            }
        }

        if total_weight > challenge.max_weight {
            let mut item_loss_ratios = Vec::new();
            for i in 0..num_items {
                if selected_items[i] {
                    let loss = values[i] + interaction_gains[i];
                    let ratio = weights[i] / (loss as f64).max(1.0);
                    item_loss_ratios.push((ratio, i));
                }
            }
            item_loss_ratios.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            while total_weight > challenge.max_weight {
                if let Some((_, item)) = item_loss_ratios.pop() {
                    for k in 0..num_items {
                        interaction_gains[k] -= challenge.interaction_values[item][k];
                    }
                    selected_items[item] = false;
                    total_weight -= challenge.weights[item];
                    total_value -= values[item] + interaction_gains[item];
                    wait_map[item] = Some(iteration_count + WAIT_ITERATIONS);
                } else {
                    break;
                }
            }
        }

        if total_value >= challenge.min_value as i32 && total_weight <= challenge.max_weight {
            let result_items: Vec<usize> = selected_items
                .iter()
                .enumerate()
                .filter(|&(_, &is_selected)| is_selected)
                .map(|(i, _)| i)
                .collect();

            return Ok(Some(Solution {
                items: result_items,
            }));
        }

        if total_value > max_total_value {
            max_total_value = total_value;
            stagnant_iterations = 0;
        } else {
            stagnant_iterations += 1;
        }

        if stagnant_iterations >= MAX_STAGNANT_ITERATIONS {
            return Ok(None);
        }
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