/*!
Copyright 2024 Crypti (PTY) LTD

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

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

    // Pre-compute value-to-weight ratios and sort items
    let mut sorted_items: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, values[i] as f64 / weights[i] as f64))
        .collect();
    sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Quick upper bound check
    let mut upper_bound = 0;
    let mut remaining_weight = max_weight;
    for &(item_index, ratio) in &sorted_items {
        let item_weight = weights[item_index];
        let item_value = values[item_index];

        if item_weight <= remaining_weight {
            upper_bound += item_value;
            remaining_weight = remaining_weight.saturating_sub(item_weight);
        } else {
            upper_bound += (ratio * remaining_weight as f64) as usize;
            break;
        }
    }

    if upper_bound < min_value {
        return Ok(None);
    }

    // Use a single vector for dynamic programming
    let mut dp = vec![0; max_weight + 1];
    
    // Use a more compact representation for tracking included items
    let mut included = vec![0u64; (num_items * (max_weight + 1) + 63) / 64];

    for (i, &(item_index, _)) in sorted_items.iter().enumerate() {
        let item_weight = weights[item_index];
        let item_value = values[item_index];

        for w in (item_weight..=max_weight).rev() {
            let new_value = dp[w - item_weight] + item_value;
            if new_value > dp[w] {
                dp[w] = new_value;
                included[i * ((max_weight + 1) / 64) + w / 64] |= 1u64 << (w % 64);
            }
        }

        // Early termination check
        if dp[max_weight] >= min_value {
            let mut selected_items = Vec::with_capacity(i + 1);
            let mut w = max_weight;
            for j in (0..=i).rev() {
                if (included[j * ((max_weight + 1) / 64) + w / 64] & (1u64 << (w % 64))) != 0 {
                    selected_items.push(sorted_items[j].0);
                    w = w.saturating_sub(weights[sorted_items[j].0]);
                }
            }
            selected_items.reverse();
            return Ok(Some(Solution { items: selected_items }));
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