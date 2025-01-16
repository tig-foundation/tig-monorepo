/*!
Copyright 2024 Chad Blanchard

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
use std::cmp;

struct Item {
    index: usize,
    weight: usize,
    value: usize,
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight as usize;
    let min_value = challenge.min_value as usize;
    let num_items = challenge.difficulty.num_items;

    let mut items: Vec<Item> = challenge.weights.iter().zip(challenge.values.iter()).enumerate()
        .map(|(i, (&w, &v))| Item {
            index: i,
            weight: w as usize,
            value: v as usize,
        })
        .collect();

    // Sort items by value-to-weight ratio
    items.sort_unstable_by(|a, b| (b.value * a.weight).cmp(&(a.value * b.weight)));

    // Quick check for trivial solution
    if items.iter().take_while(|item| item.weight <= max_weight).map(|item| item.value).sum::<usize>() < min_value {
        return Ok(None);
    }

    // Bitset DP
    let mut dp = vec![0u64; (max_weight + 63) / 64];
    dp[0] = 1;

    let mut best_value = 0;
    let mut best_weight = 0;

    for item in &items {
        let mut new_value = best_value;
        let mut new_weight = best_weight;

        for w in (item.weight..=max_weight).rev() {
            let idx = w / 64;
            let shift = w % 64;
            if (dp[idx - item.weight / 64] & (1 << (shift - item.weight % 64))) != 0 {
                dp[idx] |= 1 << shift;
                let value = best_value + item.value;
                if value > new_value {
                    new_value = value;
                    new_weight = w;
                }
            }
        }

        best_value = new_value;
        best_weight = new_weight;

        if best_value >= min_value {
            break;
        }
    }

    if best_value >= min_value {
        let mut solution = Vec::new();
        let mut remaining_value = best_value;
        let mut remaining_weight = best_weight;

        for item in items.iter().rev() {
            if remaining_weight >= item.weight && 
               (dp[(remaining_weight - item.weight) / 64] & (1 << ((remaining_weight - item.weight) % 64))) != 0 {
                solution.push(item.index);
                remaining_weight -= item.weight;
                remaining_value -= item.value;
                if remaining_value == 0 {
                    break;
                }
            }
        }

        Ok(Some(Solution { items: solution }))
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
