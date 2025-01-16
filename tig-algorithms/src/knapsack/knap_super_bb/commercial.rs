/*!
Copyright 2024 ByteBandit

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

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight;
    let min_value = challenge.min_value;
    let num_items = challenge.difficulty.num_items;

    let weights: Vec<u32> = challenge.weights.iter().map(|&w| w).collect();
    let values: Vec<u32> = challenge.values.iter().map(|&v| v).collect();

    let mut sorted_items: Vec<(usize, u32, u32)> =
        (0..num_items).map(|i| (i, values[i], weights[i])).collect();
    sorted_items.sort_unstable_by(|a, b| (b.1 * a.2).cmp(&(a.1 * b.2)));

    let mut current_solution = vec![false; num_items];
    let mut best_solution = vec![false; num_items];
    let mut best_profit = 0;

    fn calculate_upper_bound(
        i: usize,
        w: u32,
        p: u32,
        max_weight: u32,
        sorted_items: &[(usize, u32, u32)],
    ) -> u32 {
        let mut ub = p;
        let mut remaining_weight = max_weight - w;

        for &(_, value, weight) in &sorted_items[i..] {
            if weight <= remaining_weight {
                remaining_weight -= weight;
                ub += value;
            } else {
                ub += value * remaining_weight / weight;
                break;
            }
        }
        ub
    }

    fn branch_and_bound(
        i: usize,
        w: u32,
        p: u32,
        max_weight: u32,
        num_items: usize,
        sorted_items: &[(usize, u32, u32)],
        current_solution: &mut [bool],
        best_solution: &mut [bool],
        best_profit: &mut u32,
        min_value: u32,
    ) -> u32 {
        let ub = calculate_upper_bound(i, w, p, max_weight, sorted_items);
        if ub <= *best_profit || ub < min_value {
            return *best_profit;
        }

        if p > *best_profit {
            *best_profit = p;
            best_solution.copy_from_slice(current_solution);
        }

        if i < num_items {
            let (item_index, value, weight) = sorted_items[i];
            if w + weight <= max_weight {
                current_solution[item_index] = true;
                branch_and_bound(
                    i + 1,
                    w + weight,
                    p + value,
                    max_weight,
                    num_items,
                    sorted_items,
                    current_solution,
                    best_solution,
                    best_profit,
                    min_value,
                );
                current_solution[item_index] = false;
            }

            branch_and_bound(
                i + 1,
                w,
                p,
                max_weight,
                num_items,
                sorted_items,
                current_solution,
                best_solution,
                best_profit,
                min_value,
            );
        }

        *best_profit
    }

    let _ = branch_and_bound(
        0,
        0,
        0,
        max_weight,
        num_items,
        &sorted_items,
        &mut current_solution,
        &mut best_solution,
        &mut best_profit,
        min_value,
    );

    if best_profit >= min_value {
        Ok(Some(Solution {
            items: best_solution
                .iter()
                .enumerate()
                .filter_map(|(i, &included)| if included { Some(i) } else { None })
                .collect(),
        }))
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

    pub const KERNEL: Option<CudaKernel> = None;

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
