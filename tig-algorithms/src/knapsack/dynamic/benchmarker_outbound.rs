/*!
Copyright 2024 Uncharted Trading Limited

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
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
    let max_weight = challenge.max_weight;
    let min_value = challenge.min_value;
    let num_items = challenge.difficulty.num_items;

    // Sort items by value-to-weight ratio in descending order
    let mut sorted_items: Vec<usize> = (0..num_items).collect();
    sorted_items.sort_by(|&a, &b| {
        let ratio_a = challenge.values[a] as f64 / challenge.weights[a] as f64;
        let ratio_b = challenge.values[b] as f64 / challenge.weights[b] as f64;
        ratio_b.partial_cmp(&ratio_a).unwrap()
    });

    // Initialize combinations with a single empty combo
    let mut combinations: Vec<(Vec<bool>, u32, u32)> = vec![(vec![false; num_items], 0, 0)];

    let mut items = Vec::new();
    for &item in &sorted_items {
        // Create new combos with the current item
        let mut new_combinations: Vec<(Vec<bool>, u32, u32)> = combinations
            .iter()
            .map(|(combo, value, weight)| {
                let mut new_combo = combo.clone();
                new_combo[item] = true;
                (
                    new_combo,
                    value + challenge.values[item],
                    weight + challenge.weights[item],
                )
            })
            .filter(|&(_, _, weight)| weight <= max_weight) // Keep only combos within weight limit
            .collect();

        // Check if any new combination meets the minimum value requirement
        if let Some((combo, _, _)) = new_combinations
            .iter()
            .find(|&&(_, value, _)| value >= min_value)
        {
            items = combo
                .iter()
                .enumerate()
                .filter_map(|(i, &included)| if included { Some(i) } else { None })
                .collect();
            break;
        }

        // Merge new_combinations with existing combinations
        combinations.append(&mut new_combinations);

        // Deduplicate combinations by keeping the highest value for each weight
        combinations.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.cmp(&a.1))); // Sort by weight, then by value
        combinations.dedup_by(|a, b| a.2 == b.2 && a.1 <= b.1); // Deduplicate by weight, keeping highest value
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
