/*
Copyright 2024 MasterMind

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
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.difficulty.num_nodes;

    let distances_to_depot: Vec<i32> = (0..n).map(|i| d[i][0]).collect();

    let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);
    for i in 1..n {
        for j in (i + 1)..n {
            let score = distances_to_depot[i] + distances_to_depot[j] - d[i][j];
            scores.push((score, i, j));
        }
    }
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut routes: Vec<Option<Vec<usize>>> = vec![None; n];
    for i in 1..n {
        routes[i] = Some(vec![i]);
    }
    let mut route_demands = challenge.demands.clone();

    for (s, i, j) in scores {
        if s < 0 {
            break;
        }

        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        let (left_route, right_route) = (routes[i].as_ref().unwrap(), routes[j].as_ref().unwrap());

        let (left_startnode, left_endnode) = (left_route[0], *left_route.last().unwrap());
        let (right_startnode, right_endnode) = (right_route[0], *right_route.last().unwrap());
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        let mut new_route = routes[i].take().unwrap();
        let right_route = routes[j].take().unwrap();
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;

        if left_startnode == i {
            new_route.reverse();
        }
        if right_endnode == j {
            new_route.extend(right_route.into_iter().rev());
        } else {
            new_route.extend(right_route);
        }

        let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
        routes[start] = Some(new_route.clone());
        routes[end] = Some(new_route);
        route_demands[start] = merged_demand;
        route_demands[end] = merged_demand;
    }

    let mut final_routes = Vec::new();
    let mut temp_route = Vec::new();

    for (i, opt_route) in routes.into_iter().enumerate() {
        if let Some(mut route) = opt_route {
            if route[0] == i {
                temp_route.clear();
                temp_route.reserve(route.len() + 2);
                temp_route.push(0);
                temp_route.extend_from_slice(&route);
                temp_route.push(0);

                two_opt_optimize(&mut temp_route, d);

                final_routes.push(temp_route.clone());
            }
        }
    }

    Ok(Some(Solution {
        routes: final_routes,
    }))
}

fn two_opt_optimize(route: &mut Vec<usize>, distances: &[Vec<i32>]) {
    let n = route.len();
    let mut total_distance = calc_route_distance(route, distances);
    let mut improved = true;
    while improved {
        improved = false;
        for i in 1..n - 2 {
            for j in i + 1..n - 1 {
                let change = two_opt_change(route, i, j, distances);
                if change < 0 {
                    route[i..=j].reverse();
                    total_distance += change;
                    improved = true;
                }
            }
        }
    }
}

fn two_opt_change(route: &[usize], i: usize, j: usize, distances: &[Vec<i32>]) -> i32 {
    let old_distance = distances[route[i - 1]][route[i]] + distances[route[j]][route[j + 1]];
    let new_distance = distances[route[i - 1]][route[j]] + distances[route[i]][route[j + 1]];
    new_distance - old_distance
}

fn calc_route_distance(route: &[usize], distances: &[Vec<i32>]) -> i32 {
    route.windows(2).map(|w| distances[w[0]][w[1]]).sum()
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
