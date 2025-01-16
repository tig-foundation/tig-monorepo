/*!
Copyright 2024 Chad Blanchard

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
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.difficulty.num_nodes;

    // Use a max heap for scores to avoid sorting
    let mut scores = BinaryHeap::new();
    for i in 1..n {
        for j in (i + 1)..n {
            let score = d[i][0] + d[0][j] - d[i][j];
            if score > 0 {
                scores.push((score, i, j));
            }
        }
    }

    // Use a HashMap for faster route lookup and modification
    let mut routes: HashMap<usize, Vec<usize>> = (1..n).map(|i| (i, vec![i])).collect();
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    while let Some((_, i, j)) = scores.pop() {
        if !routes.contains_key(&i) || !routes.contains_key(&j) {
            continue;
        }

        let left_route = routes.get(&i).unwrap();
        let right_route = routes.get(&j).unwrap();
        let left_startnode = *left_route.first().unwrap();
        let right_startnode = *right_route.first().unwrap();
        let left_endnode = *left_route.last().unwrap();
        let right_endnode = *right_route.last().unwrap();

        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        let mut left_route = routes.remove(&i).unwrap();
        let mut right_route = routes.remove(&j).unwrap();

        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }

        left_route.extend(right_route);

        routes.insert(left_startnode, left_route.clone());
        routes.insert(right_endnode, left_route);
        route_demands[left_startnode] = merged_demand;
        route_demands[right_endnode] = merged_demand;
    }

    let solution_routes: Vec<Vec<usize>> = routes
        .into_iter()
        .filter(|&(start, ref route)| start == route[0])
        .map(|(_, mut route)| {
            let mut complete_route = vec![0];
            complete_route.append(&mut route);
            complete_route.push(0);
            complete_route
        })
        .collect();

    Ok(Some(Solution { routes: solution_routes }))
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
