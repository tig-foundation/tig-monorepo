/*!
Copyright 2024 Lump Picasso

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use std::collections::HashSet;
use tig_challenges::knapsack::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let n = challenge.difficulty.num_items;
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    let weights: Vec<u32> = pairs
        .iter()
        .map(|(i, j)| challenge.weights[*i] + challenge.weights[*j])
        .collect();
    let values: Vec<u32> = pairs
        .iter()
        .map(|(i, j)| challenge.values[*i] + challenge.values[*j])
        .collect();
    let ratios: Vec<f64> = weights
        .iter()
        .zip(values.iter())
        .map(|(w, v)| *v as f64 / *w as f64)
        .collect();
    let mut sorted_value_to_weight_ratio: Vec<usize> = (0..n).collect();
    sorted_value_to_weight_ratio.sort_by(|&a, &b| ratios[a].partial_cmp(&ratios[b]).unwrap());

    let items = HashSet::<usize>::new();
    let mut total_weight = 0;
    let max_weight = challenge.max_weight;
    for &idx in &sorted_value_to_weight_ratio {
        let mut additional_weight = 0;
        let p = pairs[idx];
        if !items.contains(&p.0) {
            additional_weight += challenge.weights[p.0];
        }
        if !items.contains(&p.1) {
            additional_weight += challenge.weights[p.1];
        }
        if total_weight + additional_weight > max_weight {
            continue;
        }
        total_weight += additional_weight;
    }
    Ok(Some(Solution {
        items: items.into_iter().collect(),
    }))
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
