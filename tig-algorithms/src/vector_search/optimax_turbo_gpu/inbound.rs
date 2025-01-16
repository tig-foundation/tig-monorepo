/*!
Copyright 2024 Lord Foulsbane

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
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use tig_challenges::vector_search::*; 

#[derive(Debug, PartialEq, PartialOrd)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

struct SpatialNode {
    coordinates: Vec<f32>,
    id: usize,
    left_subtree: Option<Box<SpatialNode>>,
    right_subtree: Option<Box<SpatialNode>>,
}

struct SpatialTree {
    root: Option<Box<SpatialNode>>,
}

impl SpatialTree {
    fn build_tree(points: &[Vec<f32>]) -> Self {
        let mut points_with_indices: Vec<(usize, &Vec<f32>)> = points.iter().enumerate().collect();
        let root_node = Self::split_into_nodes(&mut points_with_indices, 0);
        SpatialTree { root: root_node }
    }
    
    fn split_into_nodes(points: &mut [(usize, &Vec<f32>)], depth: usize) -> Option<Box<SpatialNode>> {
        if points.is_empty() {
            return None;
        }

        let dimension = points[0].1.len();
        let axis = depth % dimension;

        points.sort_by(|a, b| a.1[axis].partial_cmp(&b.1[axis]).unwrap_or(Ordering::Equal));

        let mid = points.len() / 2;
        let (id, coordinates) = points[mid];

        let mut node = Box::new(SpatialNode {
            coordinates: coordinates.clone(),
            id,
            left_subtree: None,
            right_subtree: None,
        });

        node.left_subtree = Self::split_into_nodes(&mut points[..mid], depth + 1);
        node.right_subtree = Self::split_into_nodes(&mut points[mid + 1..], depth + 1);

        Some(node)
    }
   
    fn find_closest_point(&self, target: &[f32], threshold: f32) -> Option<(usize, f32)> {
        let mut best_match = None;
        let mut smallest_dist = threshold * threshold * 10.0;
        self.closest_point_search(&self.root, target, 0, &mut best_match, &mut smallest_dist);
        best_match
    }
    
    fn closest_point_search(
        &self,
        node: &Option<Box<SpatialNode>>,
        target: &[f32],
        level: usize,
        current_best: &mut Option<(usize, f32)>,
        best_distance: &mut f32,
    ) {
        if let Some(n) = node {
            let dist = self.compute_squared_distance(target, &n.coordinates);
            if dist < *best_distance {
                *current_best = Some((n.id, dist));
                *best_distance = dist;
            }

            let axis_index = level % target.len();
            let diff_on_axis = target[axis_index] - n.coordinates[axis_index];

            let (near_branch, far_branch) = if diff_on_axis < 0.0 {
                (&n.left_subtree, &n.right_subtree)
            } else {
                (&n.right_subtree, &n.left_subtree)
            };

            self.closest_point_search(near_branch, target, level + 1, current_best, best_distance);

            if diff_on_axis.powi(2) < *best_distance {
                self.closest_point_search(far_branch, target, level + 1, current_best, best_distance);
            }
        }
    }
    
    fn compute_squared_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}

fn extract_relevant_vectors<'a>(
    database: &'a [Vec<f32>],
    queries: &[Vec<f32>],
    limit: usize,
) -> Vec<(&'a [f32], usize)> {
    let query_avg = compute_average_vector(&queries.iter().map(|v| &v[..]).collect::<Vec<&[f32]>>());

    let mut nearest_vectors: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::with_capacity(limit);

    for (index, vector) in database.iter().enumerate() {
        let distance = euclidean_distance(&query_avg, vector).powi(2);
        if nearest_vectors.len() < limit {
            nearest_vectors.push((FloatOrd(distance), index));
        } else if let Some(&(FloatOrd(farthest_distance), _)) = nearest_vectors.peek() {
            if distance < farthest_distance {
                nearest_vectors.pop();
                nearest_vectors.push((FloatOrd(distance), index));
            }
        }
    }

    nearest_vectors.into_iter().map(|(_, index)| (&database[index][..], index)).collect()
}

fn compute_average_vector(vectors: &[&[f32]]) -> Vec<f32> {
    let num_vectors = vectors.len();
    let vector_dim = vectors[0].len();

    let mut avg_vector = vec![0.0; vector_dim];

    for vector in vectors {
        for i in 0..vector_dim {
            avg_vector[i] += vector[i];
        }
    }

    avg_vector.iter_mut().for_each(|val| *val /= num_vectors as f32);
    avg_vector
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let query_count = challenge.query_vectors.len();

    let vector_limit = match query_count {
        10..=19 => 500,
        20..=28 => 1000,
        29..=50 => 500,
        51..=70 => 750,
        _ => 250,
    };

    let filtered_subset = extract_relevant_vectors(
        &challenge.vector_database,
        &challenge.query_vectors,
        vector_limit,
    );

    let kd_tree = SpatialTree::build_tree(
        &filtered_subset.iter().map(|&(vec, _)| vec.to_vec()).collect::<Vec<Vec<f32>>>(),
    );

    let mut result_indexes = Vec::with_capacity(challenge.query_vectors.len());

    for query in &challenge.query_vectors {
        if let Some((best_index, _)) = kd_tree.find_closest_point(query, challenge.max_distance) {
            result_indexes.push(best_index);
        } else {
            return Ok(None);
        }
    }

    Ok(Some(Solution { indexes: result_indexes }))
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        __device__ float euclidean_distance_squared(const float *a, const float *b, int dim)
        {
            float sum = 0.0f;
            for (int i = 0; i < dim; i += 4)
            {
                float d0 = a[i] - b[i];
                float d1 = a[i + 1] - b[i + 1];
                float d2 = a[i + 2] - b[i + 2];
                float d3 = a[i + 3] - b[i + 3];
                sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            }
            for (int i = (dim / 4) * 4; i < dim; i++)
            {
                float d = a[i] - b[i];
                sum += d * d;
            }
            return sum;
        }

        __device__ void atomicMinFloat(float *address, float val)
        {
            int *address_as_int = (int *)address;
            int old = *address_as_int, assumed;
            do
            {
                assumed = old;
                old = atomicCAS(address_as_int, assumed,
                                __float_as_int(min(val, __int_as_float(assumed))));
            } while (assumed != old);
        }

        extern "C" __global__ void vector_search_optimized(
            const float *__restrict__ database,
            const float *__restrict__ queries,
            int *__restrict__ results,
            float *__restrict__ distances,
            const int database_size,
            const int query_size,
            const int vector_dim,
            const float max_distance_sq)
        {
            const int query_idx = blockIdx.x;
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;

            __shared__ float shared_min_distance;
            __shared__ int shared_min_index;

            if (tid == 0)
            {
                shared_min_distance = 3.402823466e+38f; // FLT_MAX
                shared_min_index = -1;
            }
            __syncthreads();

            const float *query = &queries[query_idx * vector_dim];
            float local_min_distance = 3.402823466e+38f; // FLT_MAX
            int local_min_index = -1;

            for (int db_idx = tid; db_idx < database_size; db_idx += num_threads)
            {
                const float *vector = &database[db_idx * vector_dim];
                float distance_sq = euclidean_distance_squared(query, vector, vector_dim);

                if (distance_sq < local_min_distance)
                {
                    local_min_distance = distance_sq;
                    local_min_index = db_idx;
                }
            }

            atomicMinFloat(&shared_min_distance, local_min_distance);
            __syncthreads();

            if (local_min_distance == shared_min_distance)
            {
                shared_min_index = local_min_index;
            }
            __syncthreads();

            if (tid == 0)
            {
                results[query_idx] = shared_min_index;
                distances[query_idx] = shared_min_distance;
            }
        }
        "#,
        funcs: &["vector_search_optimized"],
    });

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let vector_database = &challenge.vector_database;
        let query_vectors = &challenge.query_vectors;
        let max_distance = challenge.max_distance;

        let database_size = vector_database.len();
        let query_size = query_vectors.len();
        let vector_dim = vector_database[0].len();

        let database_flat: Vec<f32> = vector_database.iter().flatten().cloned().collect();
        let queries_flat: Vec<f32> = query_vectors.iter().flatten().cloned().collect();

        let database_dev = dev.htod_sync_copy(&database_flat)?;
        let queries_dev = dev.htod_sync_copy(&queries_flat)?;
        let mut results_dev = dev.alloc_zeros::<i32>(query_size)?;
        let mut distances_dev = dev.alloc_zeros::<f32>(query_size)?;

        let block_size = 512;
        let grid_size = query_size as u32;
        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (grid_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            funcs.remove("vector_search_optimized").unwrap().launch(
                cfg,
                (
                    &database_dev,
                    &queries_dev,
                    &mut results_dev,
                    &mut distances_dev,
                    database_size as i32,
                    query_size as i32,
                    vector_dim as i32,
                    max_distance * max_distance,
                ),
            )?;
        }

        dev.synchronize()?;

        let mut results_host = vec![0i32; query_size];
        let mut distances_host = vec![0.0f32; query_size];
        dev.dtoh_sync_copy_into(&results_dev, &mut results_host)?;
        dev.dtoh_sync_copy_into(&distances_dev, &mut distances_host)?;

        let indexes: Vec<usize> = results_host.into_iter().map(|x| x as usize).collect();

        Ok(Some(Solution { indexes }))
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};

