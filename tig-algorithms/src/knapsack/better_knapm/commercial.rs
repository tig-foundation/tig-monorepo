/*!
Copyright 2024 M H

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
use std::collections::HashMap;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight as usize;
    let min_value = challenge.min_value as usize;
    let num_items = challenge.difficulty.num_items;

    let weights: Vec<usize> = challenge.weights.iter().map(|weight| *weight as usize).collect();
    let values: Vec<usize> = challenge.values.iter().map(|value| *value as usize).collect();

    fn compute_combinations(weights: &[usize], values: &[usize]) -> Vec<(usize, usize)> {
        let n = weights.len();
        let mut combinations = Vec::with_capacity(1 << n);

        for i in 0..(1 << n) {
            let mut total_weight = 0;
            let mut total_value = 0;
            for j in 0..n {
                if (i & (1 << j)) != 0 {
                    total_weight += weights[j];
                    total_value += values[j];
                }
            }
            combinations.push((total_weight, total_value));
        }

        combinations
    }

    let (left_weights, right_weights) = weights.split_at(num_items / 2);
    let (left_values, right_values) = values.split_at(num_items / 2);

    let left_combinations = compute_combinations(left_weights, left_values);
    let right_combinations = compute_combinations(right_weights, right_values);

    let mut right_combinations_map = HashMap::new();
    for &(weight, value) in &right_combinations {
        right_combinations_map.entry(weight).or_insert(value);
    }

    let mut max_value = 0;
    let mut best_combination = Vec::new();

    for &(left_weight, left_value) in &left_combinations {
        if left_weight > max_weight {
            continue;
        }

        let remaining_weight = max_weight - left_weight;
        let mut best_right_value = 0;

        for (&weight, &value) in &right_combinations_map {
            if weight <= remaining_weight {
                best_right_value = best_right_value.max(value);
            }
        }

        let total_value = left_value + best_right_value;
        if total_value >= min_value && total_value > max_value {
            max_value = total_value;
            best_combination.clear();
            best_combination.extend(
                (0..left_weights.len())
                    .filter(|&i| (1 << i) & (1 << left_weights.len()) != 0)
                    .collect::<Vec<_>>(),
            );
            best_combination.extend(
                (0..right_weights.len())
                    .filter(|&i| (1 << i) & (1 << right_weights.len()) != 0)
                    .map(|i| i + left_weights.len())
                    .collect::<Vec<_>>(),
            );
        }
    }

    if max_value >= min_value {
        Ok(Some(Solution { items: best_combination }))
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
