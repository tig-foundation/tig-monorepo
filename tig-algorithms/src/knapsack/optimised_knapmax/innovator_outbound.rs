/*!
Copyright 2024 dual mICe

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
    let max_weight = challenge.max_weight as usize;
    let min_value = challenge.min_value as usize;
    let num_items = challenge.difficulty.num_items;
    let weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
    let values: Vec<usize> = challenge.values.iter().map(|&v| v as usize).collect();

    // Early termination check
    if weights.iter().sum::<usize>() <= max_weight {
        let total_value: usize = values.iter().sum();
        if total_value >= min_value {
            return Ok(Some(Solution { items: (0..num_items).collect() }));
        }
    }

    // Sort items by value-to-weight ratio
    let mut sorted_items: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, values[i] as f64 / weights[i] as f64))
        .collect();
    sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Compute upper bound
    let mut upper_bound = 0;
    let mut remaining_weight = max_weight;
    for &(item_index, ratio) in &sorted_items {
        let item_weight = weights[item_index];
        let item_value = values[item_index];
        if item_weight <= remaining_weight {
            upper_bound += item_value;
            remaining_weight -= item_weight;
        } else {
            upper_bound += (ratio * remaining_weight as f64) as usize;
            break;
        }
    }

    if upper_bound < min_value {
        return Ok(None);
    }

    // Dynamic Programming
    let mut dp = vec![0; max_weight + 1];
    let mut prev_dp = vec![0; max_weight + 1];

    for &(item_index, _) in &sorted_items {
        std::mem::swap(&mut dp, &mut prev_dp);
        let item_weight = weights[item_index];
        let item_value = values[item_index];

        for w in item_weight..=max_weight {
            dp[w] = prev_dp[w].max(prev_dp[w - item_weight] + item_value);
        }

        // Early termination check
        if dp[max_weight] >= min_value {
            break;
        }
    }

    if dp[max_weight] < min_value {
        return Ok(None);
    }

    // Reconstruct solution
    let mut items = Vec::new();
    let mut w = max_weight;
    for &(item_index, _) in sorted_items.iter().rev() {
        if w == 0 || items.len() == num_items {
            break;
        }
        let item_weight = weights[item_index];
        if w >= item_weight && dp[w] != prev_dp[w] {
            items.push(item_index);
            w -= item_weight;
        }
    }

    Ok(Some(Solution { items }))
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
