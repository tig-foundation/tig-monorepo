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
use anyhow::Result;
use rand::{SeedableRng, Rng, rngs::StdRng};
use tig_challenges::knapsack::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let vertex_count = challenge.weights.len();

    let mut item_scores: Vec<(usize, f32)> = (0..vertex_count)
        .map(|index| {
            let interaction_sum: i32 = challenge.interaction_values[index].iter().sum();
            let score = (challenge.values[index] as f32 * 0.7 + interaction_sum as f32 * 0.3) / challenge.weights[index] as f32;
            (index, score)
        })
        .collect();

    item_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut selected_items = Vec::with_capacity(vertex_count);
    let mut unselected_items = Vec::with_capacity(vertex_count);
    let mut current_weight = 0;
    let mut current_value = 0;

    for &(index, _) in &item_scores {
        if current_weight + challenge.weights[index] <= challenge.max_weight {
            current_weight += challenge.weights[index];
            current_value += challenge.values[index] as i32;

            for &selected in &selected_items {
                current_value += challenge.interaction_values[index][selected];
            }
            selected_items.push(index);
        } else {
            unselected_items.push(index);
        }
    }

    let mut mutation_rates = vec![0; vertex_count];
    for index in 0..vertex_count {
        mutation_rates[index] = challenge.values[index] as i32;
        for &selected in &selected_items {
            mutation_rates[index] += challenge.interaction_values[index][selected];
        }
    }

    let max_generations = 100;
    let mut cooling_schedule = vec![0; vertex_count];
    let mut rng = StdRng::seed_from_u64(challenge.seed[0] as u64);

    for iteration in 0..max_generations {
        let mut best_gain = 0;
        let mut best_swap = None;

        for (u_index, &mutant) in unselected_items.iter().enumerate() {
            if cooling_schedule[mutant] > 0 {
                continue;
            }

            unsafe {
                let mutant_fitness = *mutation_rates.get_unchecked(mutant);
                let extra_weight = *challenge.weights.get_unchecked(mutant) as i32 - (challenge.max_weight as i32 - current_weight as i32);

                if mutant_fitness < 0 {
                    continue;
                }

                for (c_index, &selected) in selected_items.iter().enumerate() {
                    if *cooling_schedule.get_unchecked(selected) > 0 {
                        continue;
                    }

                    if extra_weight > 0 {
                        let removed_weight = *challenge.weights.get_unchecked(selected) as i32;
                        if removed_weight < extra_weight {
                            continue;
                        }
                    }

                    let interaction_penalty = *challenge.interaction_values.get_unchecked(mutant).get_unchecked(selected);
                    let fitness_gain = mutant_fitness - *mutation_rates.get_unchecked(selected) - interaction_penalty;

                    if fitness_gain > best_gain {
                        best_gain = fitness_gain;
                        best_swap = Some((u_index, c_index));
                    }
                }
            }
        }

        if let Some((u_index, c_index)) = best_swap {
            let added_item = unselected_items[u_index];
            let removed_item = selected_items[c_index];

            selected_items.swap_remove(c_index);
            unselected_items.swap_remove(u_index);
            selected_items.push(added_item);
            unselected_items.push(removed_item);

            current_value += best_gain;
            current_weight = current_weight + challenge.weights[added_item] - challenge.weights[removed_item];

            if current_weight > challenge.max_weight {
                continue;
            }

            unsafe {
                for index in 0..vertex_count {
                    *mutation_rates.get_unchecked_mut(index) += challenge.interaction_values.get_unchecked(index).get_unchecked(added_item)
                        - challenge.interaction_values.get_unchecked(index).get_unchecked(removed_item);
                }
            }

            cooling_schedule[added_item] = 3;
            cooling_schedule[removed_item] = 3;
        } else if iteration % 10 == 0 { 
            let swap_index = rng.gen_range(0..selected_items.len());
            let random_unselected = rng.gen_range(0..unselected_items.len());

            let added_item = unselected_items[random_unselected];
            let removed_item = selected_items[swap_index];

            selected_items.swap_remove(swap_index);
            unselected_items.swap_remove(random_unselected);
            selected_items.push(added_item);
            unselected_items.push(removed_item);

            current_value = calculate_total_value(&selected_items, challenge);
            current_weight = calculate_total_weight(&selected_items, challenge);
        }

        if current_value as u32 >= challenge.min_value {
            return Ok(Some(Solution { items: selected_items }));
        }

        for cooling_rate in cooling_schedule.iter_mut() {
            *cooling_rate = if *cooling_rate > 0 { *cooling_rate - 1 } else { 0 };
        }
    }

    if current_value as u32 >= challenge.min_value && current_weight <= challenge.max_weight {
        Ok(Some(Solution { items: selected_items }))
    } else {
        Ok(None)
    }
}

fn calculate_total_value(selected_items: &[usize], challenge: &Challenge) -> i32 {
    let mut total_value = 0;
    for &item in selected_items {
        total_value += challenge.values[item] as i32;
        for &other in selected_items {
            if item != other {
                total_value += challenge.interaction_values[item][other];
            }
        }
    }
    total_value
}

fn calculate_total_weight(selected_items: &[usize], challenge: &Challenge) -> u32 {
    selected_items.iter().map(|&i| challenge.weights[i]).sum()
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