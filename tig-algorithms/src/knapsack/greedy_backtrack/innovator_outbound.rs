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

use tig_challenges::knapsack::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let capacity = challenge.max_weight as usize;
    let target_value = challenge.min_value as usize;
    let total_items = challenge.difficulty.num_items;

    let item_weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
    let item_values: Vec<usize> = challenge.values.iter().map(|&v| v as usize).collect();

    let mut item_efficiency: Vec<(usize, f64)> = (0..total_items)
        .map(|i| (i, item_values[i] as f64 / item_weights[i] as f64))
        .collect();
    item_efficiency.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut max_possible_value = 0;
    let mut available_weight = capacity;
    for &(idx, ratio) in &item_efficiency {
        let weight = item_weights[idx];
        let value = item_values[idx];

        if weight <= available_weight {
            max_possible_value += value;
            available_weight -= weight;
        } else {
            max_possible_value += (ratio * available_weight as f64).floor() as usize;
            break;
        }
    }

    if max_possible_value < target_value {
        return Ok(None);
    }

    let mut dp_table = vec![0; capacity + 1];
    let mut track_selection = vec![vec![false; capacity + 1]; total_items];

    for (i, &(item_index, _)) in item_efficiency.iter().enumerate() {
        let weight = item_weights[item_index];
        let value = item_values[item_index];

        for current_weight in (weight..=capacity).rev() {
            let potential_value = dp_table[current_weight - weight] + value;
            if potential_value > dp_table[current_weight] {
                dp_table[current_weight] = potential_value;
                track_selection[i][current_weight] = true;
            }
        }

        if dp_table[capacity] >= target_value {
            break;
        }
    }

    if dp_table[capacity] < target_value {
        return Ok(None);
    }

    let mut selected_items = Vec::new();
    let mut remaining_capacity = capacity;
    for i in (0..total_items).rev() {
        if track_selection[i][remaining_capacity] {
            let item_index = item_efficiency[i].0;
            selected_items.push(item_index);
            remaining_capacity -= item_weights[item_index];
        }
        if remaining_capacity == 0 {
            break;
        }
    }

    Ok(Some(Solution { items: selected_items }))
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