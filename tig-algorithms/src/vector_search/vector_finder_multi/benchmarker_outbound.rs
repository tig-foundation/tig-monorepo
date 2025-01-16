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

use anyhow::Ok;
use anyhow::Result;
use tig_challenges::vector_search::*;

#[inline]
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;

    while i + 3 < a.len() {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }

    sum
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
    let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;
    let max_distance: f32 = challenge.max_distance;
    let max_distance_sq: f32 = max_distance * max_distance;

    let mut indexes: Vec<usize> = Vec::with_capacity(query_vectors.len());

    for query in query_vectors {
        let mut closest_index: Option<usize> = None;
        let mut closest_distance_sq: f32 = f32::MAX;

        for (idx, vector) in vector_database.iter().enumerate() {
            let distance_sq = euclidean_distance_squared(query, vector);

            if distance_sq <= max_distance_sq {
                closest_index = Some(idx);
                break;
            } else if distance_sq < closest_distance_sq {
                closest_index = Some(idx);
                closest_distance_sq = distance_sq;
            }
        }

        if let Some(index) = closest_index {
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
