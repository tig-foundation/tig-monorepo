/*!
Copyright 2024 Louis Silva

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
use std::collections::HashMap;

// Feature flags
const USE_S5: bool = false; // Average width per route
const USE_S3: bool = false; // Average distance from depot to directly connected customer (doesnt work)
const USE_S7: bool = true; // Average compactness per route by width
const USE_S9: bool = false; // Average depth per route

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let d: &Vec<Vec<i32>> = &challenge.distance_matrix;
    let c: i32 = challenge.max_capacity;
    let n: usize = challenge.difficulty.num_nodes;

    // Define nearest neighbors for each customer with adjustable size
    let neighbor_size: usize = 100;
    let mut nearest_neighbors: Vec<Vec<usize>> = vec![vec![]; n];
    for i in 1..n {
        let mut distances: Vec<(f64, usize)> = (1..n)
            .filter(|&j| j != i)
            .map(|j| (guided_distance(d[i][0], d[0][j], d[i][j]), j))
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nearest_neighbors[i] = distances.into_iter().take(neighbor_size).map(|(_, j)| j).collect();
    }

    // Guided Clarke-Wright heuristic with guided distance
    let mut scores: Vec<(f64, usize, usize)> = Vec::new();
    for i in 1..n {
        for &j in &nearest_neighbors[i] {
            let distance_saving = d[i][0] + d[0][j] - d[i][j];
            let feature_guidance = calculate_combined_features(&[i, j], d);  // Use the combined features as guidance
            scores.push((distance_saving as f64 - feature_guidance, i, j));
        }
    }
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();
    let mut route_features: HashMap<usize, f64> = HashMap::new(); // Track combined features per route

    // Helper function to calculate combined features
    fn calculate_combined_features(route: &[usize], d: &[Vec<i32>]) -> f64 {
        let mut total_feature_score = 0.0;

        if USE_S5 {
            total_feature_score += calculate_average_width(route, d);
        }
        if USE_S3 {
            total_feature_score += calculate_average_distance_from_depot(route, d);
        }
        if USE_S7 {
            total_feature_score += calculate_average_compactness(route, d);
        }
        if USE_S9 {
            total_feature_score += calculate_average_depth(route, d);
        }

        total_feature_score
    }

    // Helper function to calculate average width per route (S5)
    fn calculate_average_width(route: &[usize], d: &[Vec<i32>]) -> f64 {
        if route.len() <= 1 {
            return 0.0;
        }
        let x_coords: Vec<f64> = route.iter().map(|&node| d[node][0] as f64).collect();
        let min_x = x_coords.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_x = x_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max_x - min_x
    }

    // Helper function to calculate average distance from depot to directly connected customer (S3)
    fn calculate_average_distance_from_depot(route: &[usize], d: &[Vec<i32>]) -> f64 {
        if route.len() <= 1 {
            return 0.0;
        }
        route.iter().map(|&node| d[node][0] as f64).sum::<f64>() / route.len() as f64
    }

    // Helper function to calculate average compactness per route by width (S7)
    fn calculate_average_compactness(route: &[usize], d: &[Vec<i32>]) -> f64 {
        if route.len() <= 1 {
            return 0.0;
        }

        let total_compactness: f64 = route.iter()
            .enumerate()
            .flat_map(|(k, &node_k)| route.iter().skip(k + 1).map(move |&node_l| {
                let compactness = (d[node_k][node_l] - d[node_k][0]).abs() as f64 / d[node_k][0].abs() as f64;
                compactness
            }))
            .sum();
        total_compactness / route.len() as f64
    }

    // Helper function to calculate average depth per route (S9)
    fn calculate_average_depth(route: &[usize], d: &[Vec<i32>]) -> f64 {
        if route.len() <= 1 {
            return 0.0;
        }
        let y_coords: Vec<f64> = route.iter().map(|&node| d[node][1] as f64).collect();
        let min_y = y_coords.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y = y_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max_y - min_y
    }

    // Helper function to calculate guided distance
    fn guided_distance(d_i0: i32, d_0j: i32, d_ij: i32) -> f64 {
        let l_c = (d_i0 + d_0j) as f64 / 2.0;  // midpoint between depot and customer
        let delta = (d_i0 as f64 - l_c).abs() + (d_0j as f64 - l_c).abs();
        d_ij as f64 + delta
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

        let left_route = routes[i].as_ref().unwrap();
        let right_route = routes[j].as_ref().unwrap();
        let mut left_start_node = left_route[0];
        let right_start_node = right_route[0];
        let left_end_node = left_route[left_route.len() - 1];
        let mut right_end_node = right_route[right_route.len() - 1];
        let merged_demand = route_demands[left_start_node] + route_demands[right_start_node];

        if left_start_node == right_start_node || merged_demand > c {
            continue;
        }

        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();
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

        let mut new_route = left_route.clone();
        new_route.extend(right_route.clone());

        // Calculate and check the new combined features per route
        let new_feature = calculate_combined_features(&new_route, d);
        let valid_merge = route_features.get(&left_start_node).unwrap_or(&0.0) <= &new_feature &&
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

/*
References:
[1] B. Herdianto and Komarudin, "Guided Clarke and Wright Algorithm to Solve Large Scale of Capacitated Vehicle Routing Problem," 2021 IEEE 8th International Conference on Industrial Engineering and Applications (ICIEA), Chengdu, China, 2021, pp. 449-453, doi: 10.1109/ICIEA52957.2021.9436750. keywords: {Conferences;Vehicle routing;Transportation;Organizations;Industrial engineering;Routing;Complexity theory;Capacitated-Vehicle Routing Problem;Clarke and Wright Algorithm},
    Available at: https://doi.org/10.1109/ICIEA52957.2021.9436750
 */
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
