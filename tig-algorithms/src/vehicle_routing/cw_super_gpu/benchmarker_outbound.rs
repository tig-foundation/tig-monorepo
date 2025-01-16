/*!
Copyright 2024 OvErLoDe

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use tig_challenges::vehicle_routing::*;
use anyhow::Result;

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let d = &challenge.distance_matrix;
    let c = challenge.max_capacity;
    let n = challenge.difficulty.num_nodes;

    // Clarke-Wright heuristic for node pairs based on their distances to depot
    // vs distance between each other
    let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);

    for i in 1..n {
        let d_i0 = d[i][0]; // Cache this value to avoid repeated lookups
        for j in (i + 1)..n {
            let score = d_i0 + d[0][j] - d[i][j];
            scores.push((score, i, j));
        }
    }
    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order by score

    // Create a route for every node
    let mut routes: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    routes[0] = None; // The depot doesn't have a route
    let mut route_demands: Vec<i32> = challenge.demands.clone();

    // Iterate through node pairs, starting from greatest score
    for (s, i, j) in scores {
        // Stop if score is negative
        if s < 0 {
            break;
        }

        // Skip if joining the nodes is not possible
        if routes[i].is_none() || routes[j].is_none() {
            continue;
        }

        // Directly get the routes
        let (left_route, right_route) = (routes[i].as_ref().unwrap(), routes[j].as_ref().unwrap());

        // Cache indices and demands
        let (left_startnode, left_endnode) = (left_route[0], *left_route.last().unwrap());
        let (right_startnode, right_endnode) = (right_route[0], *right_route.last().unwrap());
        let merged_demand = route_demands[left_startnode] + route_demands[right_startnode];

        // Check constraints
        if left_startnode == right_startnode || merged_demand > c {
            continue;
        }

        // Merge routes
        let mut left_route = routes[i].take().unwrap();
        let mut right_route = routes[j].take().unwrap();
        routes[left_startnode] = None;
        routes[right_startnode] = None;
        routes[left_endnode] = None;
        routes[right_endnode] = None;

        // Reverse if needed
        if left_startnode == i {
            left_route.reverse();
        }
        if right_endnode == j {
            right_route.reverse();
        }

        // Create new route
        let mut new_route = left_route;
        new_route.extend(right_route);

        // Calculate the total distance of the new route
        let mut new_distance = 0;
        for k in 0..new_route.len() - 1 {
            new_distance += d[new_route[k]][new_route[k + 1]];
        }
        new_distance += d[*new_route.last().unwrap()][0]; // Add distance back to depot

        // Check if the new route exceeds the max total distance
        if new_distance > challenge.max_total_distance {
            continue;
        }

        // Update routes and demands
        let (start, end) = (*new_route.first().unwrap(), *new_route.last().unwrap());
        routes[start] = Some(new_route.clone());
        routes[end] = Some(new_route.clone()); // Clone here to avoid move
        route_demands[start] = merged_demand;
        route_demands[end] = merged_demand;
    }

    let mut final_routes = Vec::new();

    for (i, opt_route) in routes.into_iter().enumerate() {
        if let Some(mut route) = opt_route {
            if route[0] == i {
                let mut full_route = Vec::with_capacity(route.len() + 2);
                full_route.push(0);
                full_route.append(&mut route);
                full_route.push(0);
                final_routes.push(full_route);
            }
        }
    }

    Ok(Some(Solution { routes: final_routes }))
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        extern "C" __global__ void calculate_scores(
            const int* __restrict__ d,
            int n,
            int* scores
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

            if (i >= n || j >= n || i >= j) return;

            extern __shared__ int shared_d[];
            for (int k = threadIdx.x; k < n; k += blockDim.x) {
                shared_d[k] = d[k];
            }
            __syncthreads();

            int d_i0 = shared_d[i];
            int d_0j = shared_d[j];
            int d_ij = d[i * n + j];

            int score = d_i0 + d_0j - d_ij;

            int index = i * (n - 1) + (j - i - 1);
            if (index >= 0 && index < (n * (n - 1)) / 2) {
                scores[index * 3 + 0] = score;
                scores[index * 3 + 1] = i;
                scores[index * 3 + 2] = j;
            }
        }
        "#,
        funcs: &["calculate_scores"],
    });

    pub fn compute_grid_and_block_dims(num_elements: usize) -> (u32, u32) {
        let max_threads_per_block = 1024; // Typically 256, 512, or 1024 depending on the GPU
        let block_size = max_threads_per_block.min(256); // Example of a more flexible choice
        let num_blocks = (num_elements + block_size - 1) / block_size;
        (num_blocks as u32, block_size as u32)
    }

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Option<Solution>> {
        let d = &challenge.distance_matrix;
        let n = challenge.difficulty.num_nodes;

        let d_flat: Vec<i32> = d.iter().flat_map(|row| row.iter()).cloned().collect();

        let d_device: CudaSlice<i32> = dev.htod_copy(d_flat)?;
        let scores_device: CudaSlice<i32> = unsafe { dev.alloc((n * (n - 1)) / 2 * 3)? };

        let num_elements = (n * (n - 1)) / 2;
        let (num_blocks, block_size) = compute_grid_and_block_dims(num_elements);

        let func = funcs.get_mut("calculate_scores").unwrap().clone();

        unsafe {
            func.launch(LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: (n * std::mem::size_of::<i32>()) as u32,
            }, (*d_device.device_ptr() as u64, n as i32, *scores_device.device_ptr() as u64))?;
        }

        let mut scores_flat = vec![0; (n * (n - 1)) / 2 * 3];
        dev.dtoh_sync_copy_into(&scores_device, &mut scores_flat)?;

        let mut scores: Vec<(i32, usize, usize)> = Vec::with_capacity((n * (n - 1)) / 2);
        for k in 0..(n * (n - 1)) / 2 {
            let score = scores_flat[k * 3 + 0];
            let i = usize::try_from(scores_flat[k * 3 + 1]).unwrap();
            let j = usize::try_from(scores_flat[k * 3 + 2]).unwrap();
            scores.push((score, i, j));
        }

        super::solve_challenge(challenge)
    }

}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};