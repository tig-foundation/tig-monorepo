/*!
Copyright 2024 M H

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use std::collections::HashMap;
use tig_challenges::knapsack::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight as usize;
    let min_value = challenge.min_value as usize;
    let num_items = challenge.difficulty.num_items;

    let weights: Vec<usize> = challenge
        .weights
        .iter()
        .map(|weight| *weight as usize)
        .collect();
    let values: Vec<usize> = challenge
        .values
        .iter()
        .map(|value| *value as usize)
        .collect();

    // Helper function to compute knapsack solution using memoization (Top-down DP)
    fn knapsack(
        weights: &[usize],
        values: &[usize],
        max_weight: usize,
        n: usize,
        memo: &mut HashMap<(usize, usize), usize>,
    ) -> usize {
        if n == 0 || max_weight == 0 {
            return 0;
        }

        if let Some(&result) = memo.get(&(n, max_weight)) {
            return result;
        }

        let result = if weights[n - 1] > max_weight {
            knapsack(weights, values, max_weight, n - 1, memo)
        } else {
            let included =
                values[n - 1] + knapsack(weights, values, max_weight - weights[n - 1], n - 1, memo);
            let excluded = knapsack(weights, values, max_weight, n - 1, memo);
            included.max(excluded)
        };

        memo.insert((n, max_weight), result);
        result
    }

    let mut memo = HashMap::new();
    let max_value = knapsack(&weights, &values, max_weight, num_items, &mut memo);

    if max_value < min_value {
        return Ok(None);
    }

    // Reconstructing the solution
    let mut items = Vec::with_capacity(num_items);
    let mut remaining_weight = max_weight;
    let mut total_value = max_value;

    for i in (1..=num_items).rev() {
        if remaining_weight == 0 {
            break;
        }

        if memo.get(&(i, remaining_weight)) == Some(&total_value) {
            continue;
        } else {
            items.push(i - 1);
            remaining_weight -= weights[i - 1];
            total_value -= values[i - 1];
        }
    }

    if total_value >= min_value {
        Ok(Some(Solution { items }))
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
