/*!
Copyright 2024 Kouraf

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use tig_challenges::vehicle_routing::{Challenge, Solution};


pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let distance_matrix = &challenge.distance_matrix;
    let max_capacity = challenge.max_capacity;
    let num_nodes = challenge.difficulty.num_nodes;
    let demands = &challenge.demands;
    let average_demand = demands.iter().sum::<i32>() as f32 / num_nodes as f32;

    let mut best_solution: Option<Solution> = None;
    let mut best_cost = std::i32::MAX;

    // Iterate over parameter combinations
    for lambda in (1..=20).map(|x| x as f32 * 0.1) {
        for mu in (0..=20).map(|x| x as f32 * 0.1) {
            for nu in (0..=20).map(|x| x as f32 * 0.1) {
                
                let mut savings_scores: Vec<(f32, usize, usize)> = Vec::new();
                for i in 1..num_nodes {
                    for j in (i + 1)..num_nodes {
                        let savings = (distance_matrix[i][0] + distance_matrix[0][j]) as f32 
                                    - lambda * distance_matrix[i][j] as f32
                                    + mu * (distance_matrix[0][i] - distance_matrix[0][j]).abs() as f32
                                    + nu * (demands[i] + demands[j]) as f32 / average_demand;
                        savings_scores.push((savings, i, j));
                    }
                }
                savings_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                let mut routes: Vec<Option<Vec<usize>>> = (0..num_nodes).map(|i| Some(vec![i])).collect();
                routes[0] = None;
                let mut route_demands: Vec<i32> = demands.clone();

                for (savings, i, j) in savings_scores {
                    if savings < 0.0 {
                        break;
                    }

                    if routes[i].is_none() || routes[j].is_none() {
                        continue;
                    }

                    let left_route = routes[i].as_ref().unwrap();
                    let right_route = routes[j].as_ref().unwrap();
                    let mut left_start_node = left_route[0];
                    let right_start_node = right_route[0];
                    let left_end_node = left_route[left_route.len() - 1];
                    let mut right_end_node = right_route[right_route.len() - 1];
                    let combined_demand = route_demands[left_start_node] + route_demands[right_start_node];

                    if left_start_node == right_start_node || combined_demand > max_capacity {
                        continue;
                    }

                    let mut left_route = routes[i].take().unwrap();
                    let mut right_route = routes[j].take().unwrap();
                    routes[left_start_node] = None;
                    routes[right_start_node] = None;
                    routes[left_end_node] = None;
                    routes[right_end_node] = None;

                    if left_start_node == i {
                        left_route.reverse();
                        left_start_node = left_end_node;
                    }
                    if right_end_node == j {
                        right_route.reverse();
                        right_end_node = right_start_node;
                    }

                    let mut new_route = left_route;
                    new_route.extend(right_route);

                    routes[left_start_node] = Some(new_route.clone());
                    routes[right_end_node] = Some(new_route);
                    route_demands[left_start_node] = combined_demand;
                    route_demands[right_end_node] = combined_demand;
                }

                // Calculate the total cost of the solution
                let solution = Solution {
                    routes: routes
                        .into_iter()
                        .enumerate()
                        .filter(|(i, x)| x.as_ref().is_some_and(|x| x[0] == *i))
                        .map(|(_, mut x)| {
                            let mut route = vec![0];
                            route.append(x.as_mut().unwrap());
                            route.push(0);
                            route
                        })
                        .collect(),
                };

                let total_cost = solution.routes.iter().map(|route| {
                    route.windows(2).map(|w| distance_matrix[w[0]][w[1]]).sum::<i32>()
                }).sum::<i32>();

                // Check if this solution is the best one
                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_solution = Some(solution);
                }
            }
        }
    }

    Ok(best_solution)
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
