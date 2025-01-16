/*!
Copyright 2024 YourMama

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
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
    let distance_matrix = &challenge.distance_matrix;
    let max_capacity = challenge.max_capacity;
    let num_nodes = challenge.difficulty.num_nodes;

    let total_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
    let ratio = challenge.max_total_distance as f32 / total_dist;
    if ratio < 0.58 {
        return Ok(None);
    }

    // Clarke-Wright heuristic for node pairs
    let mut cw_scores: Vec<(i32, usize, usize)> = Vec::with_capacity((num_nodes - 1) * (num_nodes - 2) / 2);
    for node1 in 1..num_nodes {
        for node2 in (node1 + 1)..num_nodes {
            cw_scores.push((distance_matrix[node1][0] + distance_matrix[0][node2] - distance_matrix[node1][node2], node1, node2));
        }
    }

    cw_scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    // Initialize routes and demands
    let mut node_routes: Vec<Option<Vec<usize>>> = (0..num_nodes).map(|i| Some(vec![i])).collect();
    node_routes[0] = None;
    let mut node_demands: Vec<i32> = challenge.demands.clone();

    // Process the Clarke-Wright scores
    for (cw_score, node1, node2) in cw_scores {
        // Skip if score is negative
        if cw_score < 0 {
            break;
        }

        // Skip if either node cannot be merged
        if node_routes[node1].is_none() || node_routes[node2].is_none() {
            continue;
        }

        let route1 = node_routes[node1].as_ref().unwrap();
        let route2 = node_routes[node2].as_ref().unwrap();
        let mut start1 = route1[0];
        let start2 = route2[0];
        let end1 = route1[route1.len() - 1];
        let mut end2 = route2[route2.len() - 1];
        let combined_demand = node_demands[start1] + node_demands[start2];

        if start1 == start2 || combined_demand > max_capacity {
            continue;
        }

        let mut merged_route1 = node_routes[node1].take().unwrap();
        let mut merged_route2 = node_routes[node2].take().unwrap();
        node_routes[start1] = None;
        node_routes[start2] = None;
        node_routes[end1] = None;
        node_routes[end2] = None;

        // Reverse routes if necessary
        if start1 == node1 {
            merged_route1.reverse();
            start1 = end1;
        }
        if end2 == node2 {
            merged_route2.reverse();
            end2 = start2;
        }

        let mut merged_route = merged_route1;
        merged_route.extend(merged_route2);

        // Update the new route and demand
        node_routes[start1] = Some(merged_route.clone());
        node_routes[end2] = Some(merged_route);
        node_demands[start1] = combined_demand;
        node_demands[end2] = combined_demand;
    }

    let final_routes = node_routes
        .into_iter()
        .enumerate()
        .filter(|(idx, route)| route.as_ref().is_some_and(|r| r[0] == *idx))
        .map(|(_, mut route)| {
            let mut full_route = vec![0];
            full_route.append(route.as_mut().unwrap());
            full_route.push(0);
            full_route
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

    pub const KERNEL: Option<CudaKernel> = None;

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        device: &Arc<CudaDevice>,
        functions: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
