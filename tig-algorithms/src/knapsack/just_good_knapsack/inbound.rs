/*!
Copyright 2024 JS

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::knapsack::{Challenge, Solution};
use std::cmp::Ordering;

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let num_items = challenge.weights.len();
    let mut remaining_capacity = challenge.max_weight as i32; 
    let mut densities = vec![0.0; num_items];
    let mut selected_items = Vec::with_capacity(num_items/2);

    for i in 0..num_items {
        densities[i] = challenge.values[i] as f64 / challenge.weights[i] as f64;
    }

    let mut sorted_indices: Vec<usize> = (0..num_items).collect();
    sorted_indices.sort_unstable_by(|&i, &j| densities[i].partial_cmp(&densities[j]).unwrap_or(Ordering::Equal));

    while let Some(i) = sorted_indices.pop() {
        let weight = challenge.weights[i] as i32;
        if weight > remaining_capacity {
            continue;
        }

        selected_items.push(i);
        remaining_capacity -= weight;

        for &j in &sorted_indices {
            let joint_profit = challenge.interaction_values[i][j] as f64;
            densities[j] += joint_profit / challenge.weights[j] as f64;
        }

        sorted_indices.sort_unstable_by(|&i, &j| densities[i].partial_cmp(&densities[j]).unwrap_or(Ordering::Equal));
    }

    Ok(Some(Solution { items: selected_items }))
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

// Important! Do not include any tests in this file, it will result in your submission being rejected
