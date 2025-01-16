/*!
Copyright 2024 syebastian

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
    
    let threshold_eff = min_value as f32 / max_weight as f32;
    
    let mut upper_bound: u32 = 0;
    let mut remaining_weight = max_weight;
    let mut remaining_items = Vec::with_capacity(num_items);
    
    for (&weight, &value) in challenge.weights.iter().zip(&challenge.values) {
        let efficiency = value as f32 / weight as f32;
        if efficiency >= threshold_eff {
            upper_bound += value;
            remaining_weight -= weight;
        } else {
            remaining_items.push((efficiency, weight as u8, value as u8));
        }
    }

    fn insertion_sort(arr: &mut [(f32, u8, u8)]) {
        for i in 1..arr.len() {
            let mut j = i;
            let current = arr[i];
            while j > 0 && arr[j - 1].0 < current.0 {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = current;
        }
    }

    let index = remaining_items.len() / 2;
    remaining_items.select_nth_unstable_by(index, |a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let left_half = &remaining_items[..=index];
    let left_half_weights: u32 = left_half.iter().map(|&(_, w, _)| w as u32).sum();

    if left_half_weights > remaining_weight {
        insertion_sort(&mut remaining_items[..=index]);
    } else {
        let left_half_values: u32 = left_half.iter().map(|&(_, _, v)| v as u32).sum();

        remaining_weight -= left_half_weights;
        upper_bound += left_half_values;
    
        insertion_sort(&mut remaining_items[index+1..]);
        remaining_items.drain(..=index);
    }
    
    for &(ratio, item_weight, item_value) in &remaining_items {
        if item_weight as u32 <= remaining_weight {
            upper_bound += item_value as u32;
            remaining_weight -= item_weight  as u32;
        } else {
            upper_bound += (ratio * remaining_weight as f32).floor() as u32;
            break;
        }
    }
    
    if upper_bound < min_value {
        return Ok(None);
    }
    
    let max_capacity = max_weight as usize;
    let mut dp_inplace = vec![0; max_capacity + 1];
    let mut is_stored = vec![vec![false; max_capacity + 1]; num_items];
    
    for item_index in 0..num_items {
        let item_weight = challenge.weights[item_index] as usize;
        let item_value = challenge.values[item_index];
    
        for current_capacity in (item_weight..=max_capacity).rev() {
            let potential_value = dp_inplace[current_capacity - item_weight] + item_value;
            if potential_value > dp_inplace[current_capacity] {
                dp_inplace[current_capacity] = potential_value;
                is_stored[item_index][current_capacity] = true;
            }
        }
    
        if dp_inplace[max_capacity] >= min_value {
            break;
        }
    }
    
    if dp_inplace[max_capacity] < min_value {
        return Ok(None);
    }
    
    let mut items = Vec::new();
    let mut remaining_capacity = max_capacity;
    for item_index in (0..num_items).rev() {
        if is_stored[item_index][remaining_capacity] {
            items.push(item_index);
            remaining_capacity -= challenge.weights[item_index] as usize;
        }
        if remaining_capacity == 0 {
            break;
        }
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