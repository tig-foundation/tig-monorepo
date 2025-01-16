/*!
Copyright 2024 OvErLoDe

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
use anyhow::Result;
use tig_challenges::vector_search::{Challenge, Solution};

#[derive(Debug)]
struct KDTreeNode {
    point: Vec<f32>,
    left: Option<Box<KDTreeNode>>,
    right: Option<Box<KDTreeNode>>,
}

#[derive(Debug)]
struct KDTree {
    root: Option<Box<KDTreeNode>>,
    k: usize,
}

impl KDTreeNode {
    pub fn nearest_neighbor(&self, target: &[f32], best: &mut Option<(Vec<f32>, f32)>, depth: usize, k: usize) -> Option<(Vec<f32>, f32)> {
        let axis = depth % k;
        let dist = euclidean_distance(&self.point, target);

        if best.is_none() || dist < best.clone().unwrap().1 {
            *best = Some((self.point.clone(), dist));
        }

        let next_branch = if target[axis] < self.point[axis] {
            &self.left
        } else {
            &self.right
        };

        if let Some(branch) = next_branch {
            branch.nearest_neighbor(target, best, depth + 1, k);
        }

        let other_branch = if target[axis] < self.point[axis] {
            &self.right
        } else {
            &self.left
        };

        if let Some(branch) = other_branch {
            if (target[axis] - self.point[axis]).abs() < best.clone().unwrap().1 {
                branch.nearest_neighbor(target, best, depth + 1, k);
            }
        }

        best.clone()
    }
}

impl KDTree {
    pub fn new(points: Vec<Vec<f32>>, depth: usize, k: usize) -> Option<Box<KDTreeNode>> {
        if points.is_empty() {
            return None;
        }

        let axis = depth % k;
        let mut sorted_points = points.clone();
        sorted_points.sort_by(|a, b| a[axis].partial_cmp(&b[axis]).unwrap());

        let median = sorted_points.len() / 2;

        Some(Box::new(KDTreeNode {
            point: sorted_points[median].clone(),
            left: KDTree::new(sorted_points[..median].to_vec(), depth + 1, k),
            right: KDTree::new(sorted_points[median + 1..].to_vec(), depth + 1, k),
        }))
    }

    pub fn nearest_neighbor(&self, target: &[f32]) -> Option<(Vec<f32>, f32)> {
        if let Some(root) = &self.root {
            let mut best = None;
            root.nearest_neighbor(target, &mut best, 0, self.k);
            best
        } else {
            None
        }
    }
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let k = challenge.vector_database[0].len();
    let tree = KDTree {
        root: KDTree::new(challenge.vector_database.clone(), 0, k),
        k,
    };

    let mut indexes = Vec::new();

    for query in &challenge.query_vectors {
        if let Some((best_point, _)) = tree.nearest_neighbor(query) {
            indexes.push(
                challenge.vector_database.iter().position(|x| x == &best_point).unwrap()
            );
        }
    }

    let solution = Solution { indexes };
    Ok(Some(solution))
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x1, &x2)| (x1 - x2) * (x1 - x2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        extern "C" __global__ void kd_tree_nearest_neighbor(
            const float* query_vectors,
            const float* vector_database,
            int num_vectors,
            int vector_size,
            int num_queries,
            int* closest_indices,
            float* min_distances
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int qid = blockIdx.y;

            if (idx < num_vectors && qid < num_queries) {
                float distance = 0.0f;
                for (int i = 0; i < vector_size; ++i) {
                    float diff = query_vectors[qid * vector_size + i] - vector_database[idx * vector_size + i];
                    distance += diff * diff;
                }
                distance = sqrtf(distance);

                if (distance < min_distances[qid]) {
                    min_distances[qid] = distance;
                    closest_indices[qid] = idx;
                }
            }
        }
        "#,
        funcs: &["kd_tree_nearest_neighbor"],
    });

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
        let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;

        let num_vectors = vector_database.len();
        let vector_size = vector_database[0].len();
        let num_queries = query_vectors.len();

        let vector_db_flat: Vec<f32> = vector_database.iter().flatten().cloned().collect();
        let query_vectors_flat: Vec<f32> = query_vectors.iter().flatten().cloned().collect();

        let vector_db_dev = dev.htod_sync_copy(&vector_db_flat)?;
        let query_dev = dev.htod_sync_copy(&query_vectors_flat)?;

        let closest_indices_dev: CudaSlice<i32> = unsafe { dev.alloc(num_queries)? };
        
        // Initialize min_distances_dev to std::f32::MAX
        let mut min_distances_host = vec![std::f32::MAX; num_queries];
        let min_distances_dev = dev.htod_sync_copy(&min_distances_host)?;

        let stream = dev.fork_default_stream()?;

        let block_dim = (256, 1, 1);
        let grid_dim = (
            ((num_vectors + block_dim.0 as usize - 1) / block_dim.0 as usize) as u32,
            num_queries as u32,
            1,
        );

        let func = funcs.get_mut("kd_tree_nearest_neighbor").unwrap().clone();

        unsafe {
            func.launch_on_stream(
                &stream,
                LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes: 0,
                },
                (
                    &query_dev,
                    &vector_db_dev,
                    num_vectors as i32,
                    vector_size as i32,
                    num_queries as i32,
                    &closest_indices_dev,
                    &min_distances_dev,
                ),
            )?;
        }

        dev.wait_for(&stream)?;

        let closest_indices = dev.dtoh_sync_copy(&closest_indices_dev)?;

        let indexes: Vec<usize> = closest_indices.iter().map(|&i| i as usize).collect();

        Ok(Some(Solution { indexes }))
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};






