/*!
Copyright 2024 Crypti (PTY) LTD

Licensed under the TIG Innovator Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use tig_challenges::vehicle_routing::*;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.difficulty.num_nodes;

    // Pre-compute distances from depot to all nodes
    let depot_distances: Vec<i32> = (0..n).map(|i| d[0][i]).collect();

    // Calculate savings using a more efficient method
    let mut savings = Vec::with_capacity(n * n / 2);
    for i in 1..n {
        let d_i0 = depot_distances[i];
        for j in (i + 1)..n {
            let saving = d_i0 + depot_distances[j] - d[i][j];
            if saving > 0 {
                savings.push((saving, i, j));
            }
        }
    }
    savings.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = vec![None; n];
    for i in 1..n {
        routes[i] = Some(vec![i]);
    }
    let mut route_demands = challenge.demands.clone();

    // Iterate through node pairs, starting from greatest saving
    for (_, i, j) in savings {
        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        // Directly get the routes
        let (left_route, right_route) = (routes[i].as_ref().unwrap(), routes[j].as_ref().unwrap());

        // Cache indices and demands
        let (left_startnode, left_endnode) = (left_route[0], *left_route.last().unwrap());
        let (right_startnode, right_endnode) = (right_route[0], *right_route.last().unwrap());
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        // Check constraints
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        // Merge routes
        let mut new_route = routes[i].take().unwrap();
        let right_route = routes[j].take().unwrap();

        // Determine the correct order to merge routes
        if left_endnode != i {
            new_route.reverse();
        }
        if right_startnode == j {
            new_route.extend(right_route);
        } else {
            new_route.extend(right_route.into_iter().rev());
        }

        // Update routes and demands
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;
        let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
        routes[start] = Some(new_route);
        routes[end] = routes[start].clone();
        route_demands[start] = merged_demand;
        route_demands[end] = merged_demand;
    }

    let mut final_routes = Vec::new();
    let mut total_distance = 0;

    for (i, opt_route) in routes.into_iter().enumerate() {
        if let Some(route) = opt_route {
            if route[0] == i {
                let route_distance = calculate_route_distance(&route, d);
                total_distance += route_distance;

                if total_distance <= challenge.max_total_distance {
                    let mut full_route = Vec::with_capacity(route.len() + 2);
                    full_route.push(0);
                    full_route.extend(route);
                    full_route.push(0);
                    final_routes.push(full_route);
                } else {
                    return Ok(None);
                }
            }
        }
    }

    Ok(Some(Solution { routes: final_routes }))
}

#[inline(always)]
fn calculate_route_distance(route: &[usize], d: &[Vec<i32>]) -> i32 {
    d[0][route[0]] + 
    route.windows(2).map(|w| d[w[0]][w[1]]).sum::<i32>() +
    d[*route.last().unwrap()][0]
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