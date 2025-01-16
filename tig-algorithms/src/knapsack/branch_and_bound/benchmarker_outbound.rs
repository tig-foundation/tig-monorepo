/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use std::collections::VecDeque;
use std::cmp::Ordering;
use tig_challenges::knapsack::*;

#[derive(Clone, Debug)]
struct Node {
    level: usize,
    profit: u32,
    weight: u32,
    bound: f64,
}

impl Node {
    fn bound(&self, num_items: usize, max_weight: u32, weights: &[u32], values: &[u32]) -> f64 {
        if self.weight >= max_weight {
            return 0.0;
        }
        let mut profit_bound = self.profit as f64;
        let mut j = self.level;
        let mut total_weight = self.weight;
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
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let num_items = challenge.difficulty.num_items as usize;
    let max_weight = challenge.max_weight;
    let values = &challenge.values;
    let weights = &challenge.weights;

    let mut queue: VecDeque<Node> = VecDeque::new();
    let mut u = Node { level: 0, profit: 0, weight: 0, bound: 0.0 };
    u.bound = u.bound(num_items, max_weight, weights, values);
    queue.push_back(u.clone());

    let mut max_profit = 0;
    let mut best_items = vec![];

    while let Some(node) = queue.pop_front() {
        if node.bound > max_profit as f64 {
            u.level = node.level + 1;
            u.weight = node.weight + weights[u.level];
            u.profit = node.profit + values[u.level];
            u.bound = u.bound(num_items, max_weight, weights, values);

            if u.weight <= max_weight && u.profit > max_profit {
                max_profit = u.profit;
                best_items.push(u.level);
            }
            if u.bound > max_profit as f64 {
                queue.push_back(u.clone());
            }

            u.weight = node.weight;
            u.profit = node.profit;
            u.bound = u.bound(num_items, max_weight, weights, values);
            if u.bound > max_profit as f64 {
                queue.push_back(u.clone());
            }
        }
    }

    Ok(Some(Solution { items: best_items }))
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
