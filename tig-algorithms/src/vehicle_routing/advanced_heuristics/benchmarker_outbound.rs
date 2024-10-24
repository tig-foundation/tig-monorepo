/*!
Copyright 2024 CodeAlchemist

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::{rngs::{SmallRng, StdRng}, Rng, SeedableRng};
use tig_challenges::vehicle_routing::{Challenge, Solution};

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
    let p = challenge.max_total_distance as f32 / max_dist;
    if p < 0.57 {
        return Ok(None)
    }

    let mut best_solution: Option<Solution> = None;
    let mut best_cost = std::i32::MAX;

    const INITIAL_TEMPERATURE: f32 = 2.0;
    const COOLING_RATE: f32 = 0.995;
    const ITERATIONS_PER_TEMPERATURE: usize = 2;

    let num_nodes = challenge.difficulty.num_nodes;

    let mut current_params = vec![1.0; num_nodes];
    let mut savings_list = create_initial_savings_list(challenge);
    recompute_and_sort_savings(&mut savings_list, &current_params, challenge);
    
    let mut current_solution = create_solution(challenge, &current_params, &savings_list);
    let mut current_cost = calculate_solution_cost(&current_solution, &challenge.distance_matrix);

    if current_cost <= challenge.max_total_distance {
        return Ok(Some(current_solution));
    }

    if (current_cost as f32 * 0.96) > challenge.max_total_distance as f32 {
        return Ok(None);
    }

    let mut temperature = INITIAL_TEMPERATURE;
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    while temperature > 1.0 {
        for _ in 0..ITERATIONS_PER_TEMPERATURE {
            let neighbor_params = generate_neighbor(&current_params, &mut rng);
            recompute_and_sort_savings(&mut savings_list, &neighbor_params, challenge);
            let mut neighbor_solution = create_solution(challenge, &neighbor_params, &savings_list);
            apply_local_search_until_no_improvement(&mut neighbor_solution, &challenge.distance_matrix);
            let neighbor_cost = calculate_solution_cost(&neighbor_solution, &challenge.distance_matrix);

            let delta = neighbor_cost as f32 - current_cost as f32;
            if delta < 0.0 || rng.gen::<f32>() < (-delta / temperature).exp() {
                current_params = neighbor_params;
                current_cost = neighbor_cost;
                current_solution = neighbor_solution;

                if current_cost < best_cost {
                    best_cost = current_cost;
                    best_solution = Some(Solution {
                        routes: current_solution.routes.clone(),
                    });
                }
            }
            if best_cost <= challenge.max_total_distance {
                return Ok(best_solution);
            }
        }

        temperature *= COOLING_RATE;
    }

    Ok(best_solution)
}

#[inline]
fn create_initial_savings_list(challenge: &Challenge) -> Vec<(f32, u8, u8)> {
    let num_nodes = challenge.difficulty.num_nodes;
    let capacity = ((num_nodes - 1) * (num_nodes - 2)) / 2;
    let mut savings = Vec::with_capacity(capacity);
    for i in 1..num_nodes {
        for j in (i + 1)..num_nodes {
            savings.push((0.0, i as u8, j as u8));
        }
    }
    savings
}

#[inline]
fn recompute_and_sort_savings(savings_list: &mut [(f32, u8, u8)], params: &[f32], challenge: &Challenge) {
    let distance_matrix = &challenge.distance_matrix;

    let mut zero_len = 0;
    for (score, i, j) in savings_list.iter_mut() {
        let i = *i as usize;
        let j = *j as usize;
        *score = params[i] * distance_matrix[0][i] as f32 + 
                 params[j] * distance_matrix[j][0] as f32 - 
                 params[i] * params[j] * distance_matrix[i][j] as f32;
    }

    savings_list.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
}

#[inline]
fn generate_neighbor<R: Rng + ?Sized>(current: &[f32], rng: &mut R) -> Vec<f32> {
    current.iter().map(|&param| {
        let delta = rng.gen_range(-0.1..=0.1);
        (param + delta).clamp(0.0, 2.0)
    }).collect()
}

#[inline]
fn apply_local_search_until_no_improvement(solution: &mut Solution, distance_matrix: &Vec<Vec<i32>>) {
    let mut improved = true;
    while improved {
        improved = false;
        for route in &mut solution.routes {
            if two_opt(route, distance_matrix) {
                improved = true;
            }
        }
    }
}
#[inline]
fn two_opt(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i32>>) -> bool {
    let n = route.len();
    let mut improved = false;
    
    for i in 1..n - 2 {
        for j in i + 1..n - 1 {
            let current_distance = distance_matrix[route[i - 1]][route[i]]
                + distance_matrix[route[j]][route[j + 1]];
            let new_distance = distance_matrix[route[i - 1]][route[j]]
                + distance_matrix[route[i]][route[j + 1]];

            if new_distance < current_distance {
                route[i..=j].reverse();
                improved = true;
            }
        }
    }

    improved
}

#[inline]
fn calculate_solution_cost(solution: &Solution, distance_matrix: &Vec<Vec<i32>>) -> i32 {
    solution.routes.iter().map(|route| {
        route.windows(2).map(|w| distance_matrix[w[0]][w[1]]).sum::<i32>()
    }).sum()
}

#[inline]
fn create_solution(challenge: &Challenge, params: &[f32], savings_list: &[(f32, u8, u8)]) -> Solution {
    let distance_matrix = &challenge.distance_matrix;
    let max_capacity = challenge.max_capacity;
    let num_nodes = challenge.difficulty.num_nodes;
    let demands = &challenge.demands;

    let mut routes = vec![None; num_nodes];
    for i in 1..num_nodes {
        routes[i] = Some(vec![i]);
    }
    let mut route_demands = demands.clone();

    for &(_, i, j) in savings_list {
        let (i, j) = (i as usize, j as usize);
        if let (Some(left_route), Some(right_route)) = (routes[i].as_ref(), routes[j].as_ref()) {
            let (left_start, left_end) = (*left_route.first().unwrap(), *left_route.last().unwrap());
            let (right_start, right_end) = (*right_route.first().unwrap(), *right_route.last().unwrap());
            
            if left_start == right_start || route_demands[left_start] + route_demands[right_start] > max_capacity {
                continue;
            }

            let mut new_route = routes[i].take().unwrap();
            let mut right_route = routes[j].take().unwrap();
            
            if left_start == i { new_route.reverse(); }
            if right_end == j { right_route.reverse(); }
            
            new_route.extend(right_route);
            
            let combined_demand = route_demands[left_start] + route_demands[right_start];
            let new_start = new_route[0];
            let new_end = *new_route.last().unwrap();
            
            route_demands[new_start] = combined_demand;
            route_demands[new_end] = combined_demand;
            
            routes[new_start] = Some(new_route.clone());
            routes[new_end] = Some(new_route);
        }
    }

    Solution {
        routes: routes
            .into_iter()
            .enumerate()
            .filter_map(|(i, route)| route.filter(|r| r[0] == i))
            .map(|mut route| {
                route.insert(0, 0);
                route.push(0);
                route
            })
            .collect(),
    }
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