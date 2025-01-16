/*!
Copyright 2024 Crypti (PTY) LTD

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/


use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let max_total_distance = challenge.max_total_distance;
    let n = challenge.difficulty.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);
    for i in 1..n {
        let d_i0 = d[i][0]; // Cache this value to avoid repeated lookups
        for j in (i + 1)..n {
            let score = d_i0 + d[0][j] - d[i][j];
            scores.push((score, i, j));
        }
    }

    // Sort in descending order by score
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None; // Depot does not need a route
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    // A function to calculate the total distance of a route, including the return to the depot
    fn calculate_route_distance(route: &Vec<usize>, d: &Vec<Vec<i32>>) -> i32 {
        let mut total_distance = 0;
        let mut last_node = 0; // Start from the depot
        for &node in route {
            total_distance += d[last_node][node];
            last_node = node;
        }
        total_distance += d[last_node][0]; // Return to the depot
        total_distance
    }

    // Iterate through node pairs, starting from greatest score
    for (s, i, j) in scores {
        if s < 0 {
            break; // No further optimization possible with negative scores
        }

        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        let (left_route, right_route) = (routes[i].as_ref().unwrap(), routes[j].as_ref().unwrap());

        // Cache indices and demands
        let (left_startnode, left_endnode) = (left_route[0], *left_route.last().unwrap());
        let (right_startnode, right_endnode) = (right_route[0], *right_route.last().unwrap());
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        // Check capacity constraint
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        // Merge routes
        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();

        // Reverse if needed to make endpoints match
        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }

        let mut new_route = left_route;
        new_route.extend(right_route);

        // Calculate the total distance of the new route
        let new_route_distance = calculate_route_distance(&new_route, d);

        // Ensure the new route's distance does not exceed the maximum allowed total distance
        if new_route_distance > max_total_distance {
            continue; // Skip this merge if it exceeds the max distance constraint
        }

        // Update the routes and demands
        let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
        routes[start] = Some(new_route.clone());
        routes[end] = Some(new_route);
        route_demands[start] = merged_demand;
        route_demands[end] = merged_demand;
    }

    // Construct the final routes, ensuring that all routes start and end at the depot
    let final_routes: Vec<_> = routes.into_iter()
        .enumerate()
        .filter_map(|(i, opt_route)| {
            if let Some(mut route) = opt_route {
                if route[0] == i {
                    let mut full_route = vec![0];
                    full_route.append(&mut route);
                    full_route.push(0);
                    return Some(full_route);
                }
            }
            None
        })
        .collect();

    Ok(Some(Solution { routes: final_routes }))
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