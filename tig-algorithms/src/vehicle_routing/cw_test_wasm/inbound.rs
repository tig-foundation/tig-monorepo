/*!
Copyright 2024 mcmoid

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
    let dist: &Vec<Vec<i32>> = &challenge.distance_matrix;
    let capacity: i32 = challenge.max_capacity;
    let num_forks: usize = challenge.difficulty.num_nodes;

    let max_dist: f32 = challenge.distance_matrix[0].iter().sum::<i32>() as f32;
    
    if challenge.max_total_distance as f32 / max_dist < 0.57 { return Ok(None) }

    let mut values: Vec<(i32, usize, usize)> = Vec::with_capacity((num_forks-1)*(num_forks-2)/2);
    for v1 in 1..num_forks {
        for v2 in (v1 + 1)..num_forks {
            values.push((dist[v1][0] + dist[0][v2] - dist[v1][v2], v1, v2));
        }
    }

    values.sort_unstable_by(|a, b| b.0.cmp(&a.0));    
    
    let mut paths: Vec<Option<Vec<usize>>> = (0..num_forks).map(|i| Some(vec![i])).collect();
    paths[0] = None;
    let mut path_demands: Vec<i32> = challenge.demands.clone();
   
    for (idx, v1, v2) in values {
        if idx < 0 { break; }

        if paths[v1].is_none() || paths[v2].is_none() {
            continue;
        }

        let sx_path = paths[v1].as_ref().unwrap();
        let dx_path = paths[v2].as_ref().unwrap();
        let mut sx_startfork = sx_path[0];
        let dx_startfork = dx_path[0];
        let sx_endfork = sx_path[sx_path.len() - 1];
        let mut dx_endfork = dx_path[dx_path.len() - 1];
        let merged_demand = path_demands[sx_startfork] + path_demands[dx_startfork];

        if sx_startfork == dx_startfork || merged_demand > capacity { continue; }

        let mut sx_path = paths[v1].take().unwrap();
        let mut dx_path = paths[v2].take().unwrap();
        paths[sx_startfork] = None;
        paths[dx_startfork] = None;
        paths[sx_endfork] = None;
        paths[dx_endfork] = None;

        if sx_startfork == v1 {
            sx_path.reverse();
            sx_startfork = sx_endfork;
        }
        if dx_endfork == v2 {
            dx_path.reverse();
            dx_endfork = dx_startfork;
        }

        let mut new_path = sx_path;
        new_path.extend(dx_path);

        paths[sx_startfork] = Some(new_path.clone());
        paths[dx_endfork] = Some(new_path);
        path_demands[sx_startfork] = merged_demand;
        path_demands[dx_endfork] = merged_demand;
    }

    Ok(Some(Solution {
        routes: paths
        .into_iter()
        .enumerate()
        .filter(|(i, x)| x.as_ref().is_some_and(|x| x[0] == *i))
        .map(|(_, mut x)| {
            let mut path = vec![0];
            path.append(x.as_mut().unwrap());
            path.push(0);
            path
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