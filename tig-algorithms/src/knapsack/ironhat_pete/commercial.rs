/*!
Copyright 2024 Louis Silva

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
use anyhow::Result;

const DEBUG: bool = false;

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if DEBUG {
            println!($($arg)*);
        }
    };
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let max_weight = challenge.max_weight as usize;
    let num_items = challenge.difficulty.num_items;

    debug_log!("Challenge data: {:?}", challenge);
    debug_log!("Max weight: {}, Number of items: {}", max_weight, num_items);

    let mut dp = vec![0; max_weight + 1];
    let mut item_included = vec![vec![false; max_weight + 1]; num_items];

    let mut items: Vec<(usize, usize)> = (0..num_items)
        .map(|i| (challenge.values[i] as usize, challenge.weights[i] as usize))
        .collect();
    items.sort_by(|a, b| (b.0 as f64 / b.1 as f64).partial_cmp(&(a.0 as f64 / a.1 as f64)).unwrap());

    let upper_bound = {
        let mut bound = 0;
        let mut remaining_weight = max_weight;
        for (value, weight) in &items {
            if *weight <= remaining_weight {
                bound += value;
                remaining_weight -= weight;
            } else {
                bound += value * remaining_weight / weight;
                break;
            }
        }
        bound
    };

    debug_log!("Upper bound for optimization: {}", upper_bound);

    for i in 0..num_items {
        let item_weight = challenge.weights[i] as usize;
        let item_value = challenge.values[i] as usize;

        debug_log!("Processing item {}: weight = {}, value = {}", i, item_weight, item_value);

        for w in (item_weight..=max_weight).rev() {
            if dp[w - item_weight] + item_value > dp[w] {
                dp[w] = dp[w - item_weight] + item_value;
                item_included[i][w] = true;
                debug_log!("Updated dp[{}] to {} by including item {}", w, dp[w], i);
            }
        }
    }

    let optimal_value = dp[max_weight];
    debug_log!("Optimal value achievable: {}", optimal_value);

    if optimal_value < challenge.min_value as usize {
        debug_log!("Optimal value {} is less than minimum required value {}", optimal_value, challenge.min_value);
        return Ok(None);
    }

    let mut items = Vec::new();
    let mut w = max_weight;
    for i in (0..num_items).rev() {
        if item_included[i][w] {
            items.push(i);
            w -= challenge.weights[i] as usize;
            debug_log!("Item {} included, remaining weight {}", i, w);
        }
    }
    items.sort();
    debug_log!("Final items list: {:?}", items);

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
