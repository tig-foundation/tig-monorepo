/*
Copyright 2024 Dominic Kennedy

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

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let mtd = challenge.max_total_distance;
    let n = challenge.difficulty.num_nodes;

    let worst_case: i32 = (0..n).map(|i| d[i][0]).sum();

    let required_saving = (worst_case - mtd) as f64;
    let required_per_node = required_saving / n as f64;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    // vs distance between each other
    let mut scores: Vec<(f64, usize, usize)> = Vec::new();
    let mut route_demands: Vec<i32> = challenge.demands.clone();
    for i in 1..n {
        for j in (i + 1)..n {
            let score = d[i][0] as f64 + d[0][j] as f64 - (1.05 * d[i][j] as f64);
            let ts = d[i][0] as f64 + d[0][j] as f64 - (1.0 * d[i][j] as f64);
            if ts > (0.2 * required_per_node) {
                scores.push((score, i, j));
            }
        }
    }
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;

    // Iterate through node pairs, starting from greatest score
    for (_s, i, j) in scores {
        // Stop if score is negative
        let s2 = d[i][0] + d[0][j] - d[i][j];
        if (s2 as f64) < 0.0 {
            break;
        }

        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        let left_route = routes[i].as_ref().unwrap();
        let right_route = routes[j].as_ref().unwrap();
        let mut left_startnode = left_route[0];
        let right_startnode = right_route[0];
        let left_endnode = left_route[left_route.len() - 1];
        let mut right_endnode = right_route[right_route.len() - 1];
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;

        // reverse it
        if left_startnode == i {
            left_route.reverse();
            left_startnode = left_endnode;
        }
        if right_endnode == j {
            right_route.reverse();
            right_endnode = right_startnode;
        }

        let mut new_route = left_route;
        new_route.extend(right_route);
        routes[left_startnode] = Some(new_route.clone());
        routes[right_endnode] = Some(new_route);
        route_demands[left_startnode] = merged_demand;
        route_demands[right_endnode] = merged_demand;
    }

    let final_routes: Vec<Vec<usize>> = routes
        .into_iter()
        .enumerate()
        .filter(|(i, x)| x.as_ref().is_some_and(|x| x[0] == *i))
        .map(|(_, mut x)| {
            let mut route = vec![0];
            route.append(x.as_mut().unwrap());
            route.push(0);
            route
        })
        .collect();

    Ok(Some(Solution {
        routes: final_routes,
    }))
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
