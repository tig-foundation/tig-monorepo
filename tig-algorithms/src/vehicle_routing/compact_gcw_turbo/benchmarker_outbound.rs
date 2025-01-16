/*!
Copyright 2024 Louis Silva

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
use std::collections::HashMap;
use ndarray::{Array2, ArrayBase, ArrayView2, Dim, OwnedRepr};

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let d: &Vec<Vec<i32>> = &challenge.distance_matrix;
    let c: i32 = challenge.max_capacity;
    let n: usize = challenge.difficulty.num_nodes;

    let distance_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
        Array2::from_shape_fn((n, n), |(i, j)| d[i][j] as f64);

    // Define nearest neighbors for each customer with adjustable size
    let neighbor_size: usize = 100.min(n);
    let mut nearest_neighbors: Vec<Vec<usize>> = vec![Vec::with_capacity(neighbor_size); n];
    for i in 1..n {
        // Combine filtering and mapping using flat_map
        let mut distances: Vec<(f64, usize)> = (1..n)
            .filter_map(|j| if j != i { Some((guided_distance(distance_matrix[[i, 0]], distance_matrix[[0, j]], distance_matrix[[i, j]]), j)) } else { None })
            .collect();
        distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nearest_neighbors[i].extend(distances.into_iter().take(neighbor_size).map(|(_, j)| j));
    }

    // Guided Clarke-Wright heuristic with guided distance
    let mut scores: Vec<(f64, usize, usize)> = Vec::new();
    for i in 1..n {
        for &j in &nearest_neighbors[i] {
            let distance_saving: f64 = distance_matrix[[i, 0]] + distance_matrix[[0, j]] - distance_matrix[[i, j]];
            let feature_guidance: f64 = calculate_combined_features(&[i, j], distance_matrix.view());  // Use the combined features as guidance
            scores.push((distance_saving - feature_guidance, i, j));
        }
    }
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();
    let mut route_features: HashMap<usize, f64> = HashMap::new(); // Track combined features per route

    // Helper function to calculate combined features
    fn calculate_combined_features(route: &[usize], d: ArrayView2<f64>) -> f64 {
        let mut total_feature_score: f64 = 0.0;
        total_feature_score += calculate_average_compactness(route, d);
        total_feature_score
    }

    // Helper function to calculate average compactness per route by width (S7)
    fn calculate_average_compactness(route: &[usize], d: ArrayView2<f64>) -> f64 {
        if route.len() <= 1 {
            return 0.0;
        }

        let total_compactness: f64 = route.iter()
            .enumerate()
            .flat_map(|(k, &node_k)| route.iter().skip(k + 1).map(move |&node_l| {
                (d[[node_k, node_l]] - d[[node_k, 0]]).abs() / d[[node_k, 0]].abs()
            }))
            .sum();
        total_compactness / route.len() as f64
    }

    // Helper function to calculate guided distance
    fn guided_distance(d_i0: f64, d_0j: f64, d_ij: f64) -> f64 {
        let l_c: f64 = (d_i0 + d_0j) * 0.5;  // midpoint between depot and customer
        let delta: f64 = (d_i0 - l_c).abs() + (d_0j - l_c).abs();
        d_ij + delta
    }

    // Iterate through node pairs, starting from the greatest score
    for (s, i, j) in scores {
        // Stop if score is negative
        if s < 0.0 {
            break;
        }

        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        let left_route: &Vec<usize> = routes[i].as_ref().unwrap();
        let right_route: &Vec<usize> = routes[j].as_ref().unwrap();
        let mut left_start_node: usize = left_route[0];
        let right_start_node: usize = right_route[0];
        let left_end_node: usize = left_route[left_route.len() - 1];
        let mut right_end_node: usize = right_route[right_route.len() - 1];
        let merged_demand: i32 = route_demands[left_start_node] + route_demands[right_start_node];

        if left_start_node == right_start_node || merged_demand > c {
            continue;
        }

        let mut left_route: Vec<usize> = routes[i].take().unwrap();
        let mut right_route: Vec<usize> = routes[j].take().unwrap();
        routes[left_start_node] = None;
        routes[right_start_node] = None;
        routes[left_end_node] = None;
        routes[right_end_node] = None;

        // Reverse if necessary
        if left_start_node == i {
            left_route.reverse();
            left_start_node = left_end_node;
        }
        if right_end_node == j {
            right_route.reverse();
            right_end_node = right_start_node;
        }

        let mut new_route: Vec<usize> = left_route.clone();
        new_route.extend(right_route.clone());

        // Calculate and check the new combined features per route
        let new_feature: f64 = calculate_combined_features(&new_route, distance_matrix.view());
        let valid_merge: bool = route_features.get(&left_start_node).unwrap_or(&0.0) <= &new_feature &&
            route_features.get(&right_end_node).unwrap_or(&0.0) <= &new_feature;

        if valid_merge {
            routes[left_start_node] = Some(new_route.clone());
            routes[right_end_node] = Some(new_route);
            route_demands[left_start_node] = merged_demand;
            route_demands[right_end_node] = merged_demand;
            route_features.insert(left_start_node, new_feature);
            route_features.insert(right_end_node, new_feature);
        } else {
            routes[i] = Some(left_route);
            routes[j] = Some(right_route);
        }
    }

    Ok(Some(Solution {
        routes: routes
            .into_iter()
            .enumerate()
            .filter_map(|(i, x)| x.filter(|x| x[0] == i))
            .map(|mut x| {
                let mut route = vec![0];
                route.append(&mut x);
                route.push(0);
                route
            })
            .collect(),
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
