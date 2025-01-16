/*!
Copyright 2014 stanlocht

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::knapsack::{Challenge, Solution};

#[derive(Clone, PartialEq)]
struct Node {
    level: usize,
    profit: usize,
    weight: usize,
    bound: f64,
    items: Vec<usize>,
}

impl Node {
    fn new(level: usize, profit: usize, weight: usize, bound: f64, items: Vec<usize>) -> Self {
        Self { level, profit, weight, bound, items }
    }
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let maximum_weight_capacity = challenge.max_weight as usize;
    let minimum_value_required = challenge.min_value as usize;
    let number_of_items = challenge.difficulty.num_items;

    let item_weights: Vec<usize> = challenge.weights.iter().map(|&weight| weight as usize).collect();
    let item_values: Vec<usize> = challenge.values.iter().map(|&value| value as usize).collect();

    let mut items_sorted_by_value_density: Vec<(usize, f64)> = (0..number_of_items)
        .map(|item_index| (item_index, item_values[item_index] as f64 / item_weights[item_index] as f64))
        .collect();
    items_sorted_by_value_density.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    fn calculate_bound(node: &Node, maximum_weight_capacity: usize, items_sorted_by_value_density: &[(usize, f64)], item_weights: &[usize], item_values: &[usize]) -> f64 {
        if node.weight >= maximum_weight_capacity {
            return 0.0;
        }

        let mut bound = node.profit as f64;
        let mut total_weight = node.weight;
        let mut j = node.level;

        while j < items_sorted_by_value_density.len() && total_weight + item_weights[items_sorted_by_value_density[j].0] <= maximum_weight_capacity {
            total_weight += item_weights[items_sorted_by_value_density[j].0];
            bound += item_values[items_sorted_by_value_density[j].0] as f64;
            j += 1;
        }

        if j < items_sorted_by_value_density.len() {
            bound += (maximum_weight_capacity - total_weight) as f64 * items_sorted_by_value_density[j].1;
        }

        bound
    }

    let mut max_profit = 0;
    let mut best_items = Vec::new();
    let mut nodes = vec![Node::new(0, 0, 0, 0.0, Vec::new())];
    nodes[0].bound = calculate_bound(&nodes[0], maximum_weight_capacity, &items_sorted_by_value_density, &item_weights, &item_values);

    while let Some(node) = nodes.pop() {
        if node.bound > max_profit as f64 && node.level < number_of_items {
            let next_level = node.level + 1;

            // Explore the node including the next item
            let next_item_index = items_sorted_by_value_density[node.level].0;
            let next_weight = node.weight + item_weights[next_item_index];
            let next_profit = node.profit + item_values[next_item_index];

            let mut include_items = node.items.clone();
            include_items.push(next_item_index);

            if next_weight <= maximum_weight_capacity && next_profit > max_profit {
                max_profit = next_profit;
                best_items = include_items.clone();
            }

            let mut include_node = Node::new(next_level, next_profit, next_weight, 0.0, include_items);
            include_node.bound = calculate_bound(&include_node, maximum_weight_capacity, &items_sorted_by_value_density, &item_weights, &item_values);

            if include_node.bound > max_profit as f64 {
                let pos = nodes.binary_search_by(|n| n.bound.partial_cmp(&include_node.bound).unwrap()).unwrap_or_else(|e| e);
                nodes.insert(pos, include_node);
            }

            // Explore the node excluding the next item
            let mut exclude_node = Node::new(next_level, node.profit, node.weight, 0.0, node.items.clone());
            exclude_node.bound = calculate_bound(&exclude_node, maximum_weight_capacity, &items_sorted_by_value_density, &item_weights, &item_values);

            if exclude_node.bound > max_profit as f64 {
                let pos = nodes.binary_search_by(|n| n.bound.partial_cmp(&exclude_node.bound).unwrap()).unwrap_or_else(|e| e);
                nodes.insert(pos, exclude_node);
            }
        }
    }

    if max_profit >= minimum_value_required {
        Ok(Some(Solution { items: best_items }))
    } else {
        Ok(None)
    }
}

// Important! Do not include any tests in this file, it will result in your submission being rejected

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
