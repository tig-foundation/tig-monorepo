/*!
Copyright 2024 OvErLoDe

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

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
    let n = challenge.difficulty.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    // vs distance between each other
    let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);
for i in 1..n {
    let d_i0 = d[i][0]; // Cache this value to avoid repeated lookups
    for j in (i + 1)..n {
        let score = d_i0 + d[0][j] - d[i][j];
        scores.push((score, i, j));
    }
}
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    // Iterate through node pairs, starting from greatest score
    for (s, i, j) in scores {
        // Stop if score is negative
        if s < 0 {
            break;
        }
    
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
        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;
    
        // Reverse if needed
        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }
    
        // Create new route
        let mut new_route = left_route;
        new_route.extend(right_route);
    
        // Update routes and demands
        let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
        routes[start] = Some(new_route.clone());
        routes[end] = Some(new_route);
        route_demands[start] = merged_demand;
        route_demands[end] = merged_demand;
    }

    let mut final_routes = Vec::new();

    for (i, opt_route) in routes.into_iter().enumerate() {
        if let Some(mut route) = opt_route {
            if route[0] == i {
                let mut full_route = Vec::with_capacity(route.len() + 2);
                full_route.push(0);
                full_route.append(&mut route);
                full_route.push(0);
                final_routes.push(full_route);
            }
        }
    }
    
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
