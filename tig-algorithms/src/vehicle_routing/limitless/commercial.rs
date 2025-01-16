/*!
Copyright 2024 Gojo Satoru

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use std::collections::HashSet;
use tig_challenges::vehicle_routing::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut solution = Solution { routes: vec![] };
    let mut remaining: HashSet<usize> = (1..challenge.difficulty.num_nodes).into_iter().collect();

    while !remaining.is_empty() {
        let mut best_route: HashSet<usize> = HashSet::new();
        let mut best_ratio: f64 = 0.0;
        for n in remaining.iter() {
            let mut closest: Vec<usize> = remaining
                .iter()
                .cloned()
                .filter(|n2| challenge.distance_matrix[*n][*n2] <= 30)
                .collect();
            closest.sort_by(|&a, &b| {
                challenge.demands[b]
                    .partial_cmp(&challenge.demands[a])
                    .unwrap()
            });
            let mut total_demand = challenge.demands[*n];
            let mut total_distance = 0;
            let mut route = HashSet::new();
            route.insert(*n);
            for n2 in closest.iter() {
                if total_demand + challenge.demands[*n2] <= challenge.max_capacity {
                    total_demand += challenge.demands[*n2];
                    total_distance += challenge.distance_matrix[*n][*n2];
                    route.insert(*n2);
                }
            }
            let ratio = total_demand as f64 / total_distance as f64;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_route = route;
            }
        }

        remaining = remaining.difference(&best_route).cloned().collect();

        let mut current_node = 0;
        let mut route = vec![0];
        while !best_route.is_empty() {
            let n = *best_route
                .iter()
                .min_by_key(|&n| challenge.distance_matrix[current_node][*n])
                .unwrap();
            route.push(n);
            best_route.remove(&n);
            current_node = n;
        }
        route.push(0);
        solution.routes.push(route);
    }

    Ok(Some(solution))
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
