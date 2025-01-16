/*!
Copyright 2024 AllFather

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use std::cmp::Ordering;
use tig_challenges::knapsack::*;

#[derive(Clone, Copy)]
struct Item {
    index: usize,
    weight: u32,
    value: u32,
    ratio: f32,
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight;
    let min_value = challenge.min_value;

    let mut items: Vec<Item> = challenge
        .weights
        .iter()
        .zip(challenge.values.iter())
        .enumerate()
        .map(|(i, (&w, &v))| Item {
            index: i,
            weight: w,
            value: v,
            ratio: v as f32 / w as f32,
        })
        .collect();

    items.sort_unstable_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap_or(Ordering::Equal));

    let mut best_value = 0;
    let mut best_solution = Vec::with_capacity(challenge.difficulty.num_items);
    let mut current_solution = Vec::with_capacity(challenge.difficulty.num_items);

    branch_and_bound(
        &items,
        0,
        0,
        0,
        max_weight,
        min_value,
        &mut best_value,
        &mut best_solution,
        &mut current_solution,
    );

    if best_value >= min_value {
        Ok(Some(Solution {
            items: best_solution,
        }))
    } else {
        Ok(None)
    }
}

#[inline(always)]
fn branch_and_bound(
    items: &[Item],
    index: usize,
    current_weight: u32,
    current_value: u32,
    max_weight: u32,
    min_value: u32,
    best_value: &mut u32,
    best_solution: &mut Vec<usize>,
    current_solution: &mut Vec<usize>,
) {
    if current_value > *best_value && current_value >= min_value {
        *best_value = current_value;
        best_solution.clear();
        best_solution.extend_from_slice(current_solution);
    }

    if index >= items.len() {
        return;
    }

    let mut upper_bound = current_value;
    let mut remaining_weight = max_weight - current_weight;

    for item in &items[index..] {
        if item.weight <= remaining_weight {
            upper_bound += item.value;
            remaining_weight -= item.weight;
        } else {
            upper_bound += (item.ratio * remaining_weight as f32) as u32;
            break;
        }
    }

    if upper_bound <= *best_value || upper_bound < min_value {
        return;
    }

    let item = &items[index];
    if current_weight + item.weight <= max_weight {
        current_solution.push(item.index);
        branch_and_bound(
            items,
            index + 1,
            current_weight + item.weight,
            current_value + item.value,
            max_weight,
            min_value,
            best_value,
            best_solution,
            current_solution,
        );
        current_solution.pop();
    }

    branch_and_bound(
        items,
        index + 1,
        current_weight,
        current_value,
        max_weight,
        min_value,
        best_value,
        best_solution,
        current_solution,
    );
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
