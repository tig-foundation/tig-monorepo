/*!
Copyright 2024 Goto Satoru

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
    let mut solution = Solution {
        sub_solutions: Vec::new(),
    };
    for sub_instance in &challenge.sub_instances {
        match solve_sub_instance(sub_instance)? {
            Some(sub_solution) => solution.sub_solutions.push(sub_solution),
            None => return Ok(None),
        }
    }
    Ok(Some(solution))
}

pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
    let max_weight = challenge.max_weight as usize;
    let baseline_value = challenge.baseline_value as usize;
    let num_items = challenge.difficulty.num_items;

    let weights: Vec<usize> = challenge
        .weights
        .iter()
        .map(|&weight| weight as usize)
        .collect();
    let values: Vec<usize> = challenge
        .values
        .iter()
        .map(|&value| value as usize)
        .collect();

    let mut sorted_items: Vec<(usize, f64)> = (0..num_items)
        .map(|i| (i, values[i] as f64 / weights[i] as f64))
        .collect();
    sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut dp = vec![0; max_weight + 1];

    for &(item_index, _) in &sorted_items {
        let weight = weights[item_index];
        let value = values[item_index];
        for w in (weight..=max_weight).rev() {
            dp[w] = dp[w].max(dp[w - weight] + value);
            if dp[w] >= baseline_value {
                return Ok(Some(SubSolution {
                    items: vec![item_index],
                }));
            }
        }
    }

    let mut items = Vec::new();
    let mut w = max_weight;
    while w > 0 {
        if let Some(&(item_index, _)) = sorted_items
            .iter()
            .find(|&&(i, _)| weights[i] <= w && dp[w] == dp[w - weights[i]] + values[i])
        {
            items.push(item_index);
            w -= weights[item_index];
        } else {
            break;
        }
    }

    if dp[max_weight] >= baseline_value {
        Ok(Some(SubSolution { items }))
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
        challenge: &SubInstance,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<SubSolution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
