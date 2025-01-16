/*!
Copyright 2024 Tinhat Pete

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge


use tig_challenges::knapsack::*;
use anyhow::Result;

// use anyhow::{anyhow, Result};
// use tig_challenges::knapsack::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let max_weight = challenge.max_weight as usize;
    let num_items = challenge.difficulty.num_items;

    // Initialize a DP table where dp[w] holds the max value achievable with weight w
    let mut dp = vec![0; max_weight + 1];
    let mut keep = vec![vec![false; max_weight + 1]; num_items];

    for i in 0..num_items {
        let item_weight = challenge.weights[i] as usize;
        let item_value = challenge.values[i];

        for w in (item_weight..=max_weight).rev() {
            if dp[w - item_weight] + item_value as u32 > dp[w] {
                dp[w] = dp[w - item_weight] + item_value as u32;
                keep[i][w] = true;
            }
        }
    }

    // Find the optimal value achievable
    let optimal_value = dp[max_weight];

    // Check if the optimal value meets the minimum value requirement
    if optimal_value < challenge.min_value {
        return Ok(None);
    }

    // Reconstruct the list of items to pick
    let mut items = Vec::new();
    let mut w = max_weight;
    for i in (0..num_items).rev() {
        if keep[i][w] {
            items.push(i);
            w -= challenge.weights[i] as usize;
        }
    }
    items.sort(); // Ensure items are in the original order for the solution

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
