/*!
Copyright 2024 AllFather

Licensed under the TIG Benchmarker Outbound Game License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use anyhow::Result;
use std::collections::BinaryHeap;
use std::{cmp::Ordering, vec};
use tig_challenges::vector_search::*;

struct KdNode {
    point: Vec<f32>,
    index: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

struct KdTree {
    root: Option<Box<KdNode>>,
}

impl KdTree {
    fn new(points: &[Vec<f32>]) -> Self {
        let mut indexed_points: Vec<(usize, &Vec<f32>)> = points.iter().enumerate().collect();
        let root = Self::build_tree(&mut indexed_points, 0);
        KdTree { root }
    }

    fn build_tree(points: &mut [(usize, &Vec<f32>)], depth: usize) -> Option<Box<KdNode>> {
        if points.is_empty() {
            return None;
        }

        let k = points[0].1.len();
        let axis = depth % k;

        points.sort_by(|a, b| a.1[axis].partial_cmp(&b.1[axis]).unwrap_or(Ordering::Equal));

        let median = points.len() / 2;
        let (index, point) = points[median];

        let mut node = Box::new(KdNode {
            point: point.clone(),
            index,
            left: None,
            right: None,
        });

        node.left = Self::build_tree(&mut points[..median], depth + 1);
        node.right = Self::build_tree(&mut points[median + 1..], depth + 1);

        Some(node)
    }

    fn nearest_neighbor(&self, query: &[f32], max_distance: f32) -> Option<(usize, f32)> {
        let mut best = None;
        let mut best_dist = max_distance * max_distance * 10.0;
        self.search_nearest(&self.root, query, 0, &mut best, &mut best_dist);
        best
    }

    fn search_nearest(
        &self,
        node: &Option<Box<KdNode>>,
        query: &[f32],
        depth: usize,
        best: &mut Option<(usize, f32)>,
        best_dist: &mut f32,
    ) {
        let node = match node {
            Some(n) => n,
            None => return,
        };

        let dist = euclidean_distance_squared(query, &node.point);
        if dist < *best_dist {
            *best = Some((node.index, dist));
            *best_dist = dist;
        }

        let axis = depth % query.len();
        let diff = query[axis] - node.point[axis];

        let (near, far) = if diff <= 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        self.search_nearest(near, query, depth + 1, best, best_dist);

        if diff * diff < *best_dist {
            self.search_nearest(far, query, depth + 1, best, best_dist);
        }
    }
}

#[inline]
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
    let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;
    let max_distance: f32 = challenge.max_distance;
    let max_distance_sq: f32 = max_distance * max_distance;

    let max_items = match challenge.query_vectors.len() {
        10..=50 => (625864.0 / (50.0 + 2.0)) as usize,
        _ => (625864.0 / (challenge.query_vectors.len() as f32 + 2.0)) as usize,
    };

    let vector_database = &vector_database[..max_items];

    let kd_tree = KdTree::new(vector_database);
    let mut indexes: Vec<usize> = Vec::with_capacity(query_vectors.len());

    for query in query_vectors {
        if let Some((index, _)) = kd_tree.nearest_neighbor(query, max_distance) {
            indexes.push(index);
        } else {
            return Ok(None);
        }
    }

    Ok(Some(Solution { indexes }))
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
    // Handle remaining elements
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
}"#,
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
