/*!
Copyright 2024 Louis Silva

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let distance_matrix = &challenge.distance_matrix;
    let vehicle_capacity = challenge.max_capacity;
    let num_nodes = challenge.difficulty.num_nodes;

    // Implement heuristic from [1]
    let w: Vec<i32> = distance_matrix.iter().map(|row| row[0]).collect();
    let mut l: Vec<Vec<i32>> = vec![vec![0; num_nodes]; num_nodes];
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                l[i][j] = distance_matrix[i][j] + (w[i] - w[j]).abs();
            }
        }
    }

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..num_nodes).map(|i| Some(vec![i])).collect();
    routes[0] = None;
    let mut route_demands: Vec<i32> = challenge.demands.clone();
    let mut savings = calculate_savings(num_nodes, &distance_matrix);

    let mut i = 0;
    while i < savings.len() {
        let (s, i_node, j_node) = savings[i];

        // Stop if score is negative
        if s < 0 {
            break;
        }

        // Skip if joining the nodes is not possible
        if routes[i_node].is_none() || routes[j_node].is_none() {
            i += 1;
            continue;
        }

        let left_route = routes[i_node].as_ref().unwrap();
        let right_route = routes[j_node].as_ref().unwrap();
        let mut left_start_node = left_route[0];
        let right_start_node = right_route[0];
        let left_end_node = left_route[left_route.len() - 1];
        let mut right_end_node = right_route[right_route.len() - 1];
        let merged_demand = route_demands[left_start_node] + route_demands[right_start_node];

        if left_start_node == right_start_node || merged_demand > vehicle_capacity {
            i += 1;
            continue;
        }

        let mut left_route = routes[i_node].take().unwrap();
        let mut right_route = routes[j_node].take().unwrap();
        routes[left_start_node] = None;
        routes[right_start_node] = None;
        routes[left_end_node] = None;
        routes[right_end_node] = None;

        // Reverse it
        if left_start_node == i_node {
            left_route.reverse();
            left_start_node = left_end_node;
        }
        if right_end_node == j_node {
            right_route.reverse();
            right_end_node = right_start_node;
        }

        let mut new_route = left_route;
        new_route.extend(right_route);

        // Only the start and end nodes of routes are kept
        routes[left_start_node] = Some(new_route.clone());
        routes[right_end_node] = Some(new_route);
        route_demands[left_start_node] = merged_demand;
        route_demands[right_end_node] = merged_demand;

        // Recalculate savings incrementally
        let mut new_savings = Vec::new();
        for k in 1..num_nodes {
            if routes[k].is_some() {
                for &node in &[left_start_node, right_end_node] {
                    if k != node {
                        let score = distance_matrix[node][0] + distance_matrix[0][k] - distance_matrix[node][k];
                        new_savings.push((score, node, k));
                    }
                }
            }
        }
        // new_savings.sort_by(|a, b| b.0.cmp(&a.0));

        // Remove outdated savings and insert updated ones
        savings.retain(|&(_, x, y)| x != left_start_node && y != right_end_node);
        savings.extend(new_savings);
        savings.sort_by(|a, b| b.0.cmp(&a.0));

        i = 0; // Restart from the beginning of the updated savings list
    }

    let mut solution: Vec<Vec<usize>> = routes
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

    // Apply 2-opt
    for route_option in solution.iter_mut() {
        *route_option = two_opt(route_option, distance_matrix);
    }

    three_opt(&mut solution, distance_matrix, &route_demands, vehicle_capacity);
    try_swap(&mut solution, distance_matrix, &route_demands, vehicle_capacity);

    Ok(Some(Solution { routes: solution }))
}

fn two_opt(route: &Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> Vec<usize> {
    let mut best_route = route.to_vec();
    let mut best_distance = calculate_route_distance(&best_route, distance_matrix);
    let mut improved = true;

    while improved {
        improved = false;

        for i in 1..best_route.len() - 2 {
            for j in i + 1..best_route.len() - 1 {
                // Compute the change in distance
                let delta = distance_matrix[best_route[i - 1]][best_route[j]] +
                    distance_matrix[best_route[i]][best_route[j + 1]] -
                    distance_matrix[best_route[i - 1]][best_route[i]] -
                    distance_matrix[best_route[j]][best_route[j + 1]];

                if delta < 0 {
                    // Perform the 2-opt swap
                    best_route[i..=j].reverse();
                    best_distance += delta;
                    improved = true;
                }
            }
        }
    }

    best_route
}

fn three_opt(
    solution: &mut Vec<Vec<usize>>,
    distance: &Vec<Vec<i32>>,
    demands: &Vec<i32>,
    max_capacity: i32,
) -> bool
{
    let mut improved = false;

    for r in 0..solution.len() {
        let n = solution[r].len();
        let original_route = solution[r].clone();

        for i in 0..n - 3 {
            for j in i + 1..n - 2 {
                for k in j + 1..n - 1 {
                    let segments = [
                        &original_route[0..=i],
                        &original_route[i + 1..=j],
                        &original_route[j + 1..=k],
                        &original_route[k + 1..n],
                    ];

                    let new_routes = vec![
                        segments[0].iter().chain(segments[1].iter().rev()).chain(segments[2].iter()).chain(segments[3].iter()).cloned().collect(),
                        segments[0].iter().chain(segments[1].iter()).chain(segments[2].iter().rev()).chain(segments[3].iter()).cloned().collect(),
                        segments[0].iter().chain(segments[1].iter().rev()).chain(segments[2].iter().rev()).chain(segments[3].iter()).cloned().collect(),
                        segments[0].iter().chain(segments[2].iter().chain(segments[1].iter().rev())).chain(segments[3].iter()).cloned().collect(),
                        segments[0].iter().chain(segments[2].iter().rev().chain(segments[1].iter().rev())).chain(segments[3].iter()).cloned().collect(),
                        segments[0].iter().chain(segments[2].iter()).chain(segments[1].iter().rev()).chain(segments[3].iter()).cloned().collect(),
                    ];

                    for new_route in new_routes {
                        let new_demand = compute_route_demand(&new_route, demands);

                        if new_demand <= max_capacity {
                            let current_cost = calculate_route_distance(&solution[r], distance);
                            let new_cost = calculate_route_distance(&new_route, distance);

                            if new_cost < current_cost {
                                solution[r] = new_route;
                                improved = true;
                                break; // Early exit once improvement is found
                            }
                        }
                    }

                    if improved {
                        break; // Early exit from the middle loop
                    }
                }

                if improved {
                    break; // Early exit from the outer loop
                }
            }
        }
    }

    improved
}

fn try_swap(
    solution: &mut Vec<Vec<usize>>,
    distance_matrix: &Vec<Vec<i32>>,
    demands: &Vec<i32>,
    vehicle_capacity: i32,
) -> bool
{
    let mut improved = true;

    while improved {
        improved = false;
        for i in 0..solution.len() {
            let demand_i = compute_route_demand(&solution[i], demands);
            for j in i + 1..solution.len() {
                let demand_j = compute_route_demand(&solution[j], demands);
                for k in 1..solution[i].len() - 1 {
                    for l in 1..solution[j].len() - 1 {
                        let customer_i = solution[i][k];
                        let customer_j = solution[j][l];
                        let new_demand_i = demand_i - demands[customer_i] + demands[customer_j];
                        let new_demand_j = demand_j - demands[customer_j] + demands[customer_i];

                        if new_demand_i <= vehicle_capacity && new_demand_j <= vehicle_capacity {
                            let current_cost = distance_matrix[solution[i][k - 1]][solution[i][k]]
                                + distance_matrix[solution[i][k]][solution[i][k + 1]]
                                + distance_matrix[solution[j][l - 1]][solution[j][l]]
                                + distance_matrix[solution[j][l]][solution[j][l + 1]];

                            let new_cost = distance_matrix[solution[i][k - 1]][solution[j][l]]
                                + distance_matrix[solution[j][l]][solution[i][k + 1]]
                                + distance_matrix[solution[j][l - 1]][solution[i][k]]
                                + distance_matrix[solution[i][k]][solution[j][l + 1]];

                            if new_cost < current_cost {
                                // Perform the swap
                                solution[i][k] = customer_j;
                                solution[j][l] = customer_i;

                                improved = true;
                            }
                        }
                    }
                }
            }
        }
    }

    improved
}

// Clarke-Wright heuristic: calculate savings for node pairs
fn calculate_savings(n: usize, d: &Vec<Vec<i32>>) -> Vec<(i32, usize, usize)> {
    let mut savings = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in (i + 1)..n {
            let score: i32 = d[i][0] + d[0][j] - d[i][j];
            savings.push((score, i, j));
        }
    }
    savings.sort_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score
    savings
}

fn calculate_route_distance(route: &Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> i32 {
    route.windows(2).map(|window| distance_matrix[window[0]][window[1]]).sum()
}

fn compute_route_demand(route: &Vec<usize>, demands: &Vec<i32>) -> i32 {
    route.iter().map(|&customer| demands[customer]).sum()
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
