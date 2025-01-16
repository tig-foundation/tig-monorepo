/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use tig_challenges::knapsack::{Challenge, Solution};

#[derive(Debug, Clone)]
struct Node {
    level: usize,
    profit: u32,
    weight: u32,
    bound: f64,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.bound.partial_cmp(&self.bound).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Node {}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.bound == other.bound
    }
}

fn bound(node: &Node, num_items: usize, max_weight: u32, weights: &[u32], values: &[u32]) -> f64 {
    if node.weight >= max_weight {
        return 0.0;
    }

    let mut profit_bound = node.profit as f64;
    let mut j = node.level;
    let mut total_weight = node.weight;

    while j < num_items && total_weight + weights[j] <= max_weight {
        total_weight += weights[j];
        profit_bound += values[j] as f64;
        j += 1;
    }

    if j < num_items {
        profit_bound += (max_weight - total_weight) as f64 * values[j] as f64 / weights[j] as f64;
    }

    profit_bound
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_weight = challenge.max_weight;
    let min_value = challenge.min_value;
    let num_items = challenge.difficulty.num_items;

    let mut sorted_items: Vec<usize> = (0..num_items).collect();
    sorted_items.sort_by(|&a, &b| {
        let ratio_a = challenge.values[a] as f64 / challenge.weights[a] as f64;
        let ratio_b = challenge.values[b] as f64 / challenge.weights[b] as f64;
        ratio_b.partial_cmp(&ratio_a).unwrap()
    });

    let sorted_weights: Vec<u32> = sorted_items.iter().map(|&i| challenge.weights[i]).collect();
    let sorted_values: Vec<u32> = sorted_items.iter().map(|&i| challenge.values[i]).collect();

    let mut max_profit = 0;
    let mut best_combination = vec![false; num_items];
    let mut queue = BinaryHeap::new();
    let root = Node {
        level: 0,
        profit: 0,
        weight: 0,
        bound: bound(&Node { level: 0, profit: 0, weight: 0, bound: 0.0 }, num_items, max_weight, &sorted_weights, &sorted_values),
    };

    queue.push(root);

    while let Some(node) = queue.pop() {
        if node.level == num_items || node.bound <= max_profit as f64 {
            continue;
        }

        let next_level = node.level + 1;
        let next_weight = node.weight + sorted_weights[node.level];
        let next_profit = node.profit + sorted_values[node.level];

        if next_weight <= max_weight && next_profit > max_profit {
            max_profit = next_profit;
            best_combination[node.level] = true;
        }

        if next_weight <= max_weight {
            let next_node = Node {
                level: next_level,
                profit: next_profit,
                weight: next_weight,
                bound: bound(&Node { level: next_level, profit: next_profit, weight: next_weight, bound: 0.0 }, num_items, max_weight, &sorted_weights, &sorted_values),
            };
            if next_node.bound > max_profit as f64 {
                queue.push(next_node);
            }
        }

        let next_node = Node {
            level: next_level,
            profit: node.profit,
            weight: node.weight,
            bound: bound(&Node { level: next_level, profit: node.profit, weight: node.weight, bound: 0.0 }, num_items, max_weight, &sorted_weights, &sorted_values),
        };
        if next_node.bound > max_profit as f64 {
            queue.push(next_node);
        }
    }

    let selected_items: Vec<usize> = sorted_items.into_iter().enumerate().filter_map(|(i, _)| if best_combination[i] { Some(i) } else { None }).collect();

    Ok(Some(Solution { items: selected_items }))
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
