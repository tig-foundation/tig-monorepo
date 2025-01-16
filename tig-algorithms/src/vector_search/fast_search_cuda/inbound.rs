/*!
Copyright 2024 Cortex & Haz

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
 */

use anyhow::Result;
use tig_challenges::vector_search::{Challenge, Solution};

fn squared_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let max_distance_sq = challenge.max_distance * challenge.max_distance;

    let indexes: Vec<usize> = challenge
        .query_vectors
        .iter()
        .filter_map(|query| {
            challenge
                .vector_database
                .iter()
                .enumerate()
                .find_map(|(i, vector)| {
                    if squared_distance(query, vector) <= max_distance_sq {
                        Some(i)
                    } else {
                        None
                    }
                })
        })
        .collect();

    if indexes.len() == challenge.query_vectors.len() {
        Ok(Some(Solution { indexes }))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc, time::Instant, ptr};
    use tig_challenges::CudaKernel;

    const BLOCK_SIZE: u32 = 256;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
#define INT_MAX     (2147483647U)
#define FLT_MAX     (3.402823466e+38F)

__device__ float squared_distance(const float * v1, const float * v2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sum;
}

extern "C" __global__ void fast_search(
    const float * query_vectors,
    const float * vector_database,
    int query_count,
    int vector_count,
    int vector_dim,
    float max_distance_sq,
    int shared_mem_size,
    int * result_indexes
) {
    extern __shared__ int sh_mem[];
    int queryIdx = blockIdx.x;
    int thIdx = threadIdx.x;
    int * sh_indexes = sh_mem;
    assert(shared_mem_size >= sizeof(sh_indexes[0]) * blockDim.x);
    
    if (queryIdx >= query_count) {
        return;
    } 

    const float * query = &query_vectors[queryIdx * vector_dim];
    float minDistance = FLT_MAX;
    int minIndex = -1;
    for (int i = 0; i < vector_count; i += blockDim.x) {
        int vectorIdx = i + thIdx;
        const float * vector = &vector_database[vectorIdx * vector_dim];
        float distance = squared_distance(query, vector, vector_dim);
        if ((distance < minDistance) && (distance <= max_distance_sq)) {
            minDistance = distance;
            minIndex = vectorIdx;
            break;
        }
    }

    sh_indexes[thIdx] = minIndex;
    __syncthreads();

    if (thIdx == 0) {
        minIndex = INT_MAX;
        for (int i = 0; i < blockDim.x; i++) {
            if ((sh_indexes[i] > -1) && (sh_indexes[i] < minIndex)) {
                minIndex = sh_indexes[i];
            }
        }
        result_indexes[queryIdx] = minIndex;
    }
}
        "#,
        funcs: &["fast_search"],
    });

    fn flatten_vector_database(database: &[Vec<f32>]) -> Vec<f32> {
        let total_elements = database.len() * database[0].len();
        let mut database_flat: Vec<f32> = Vec::with_capacity(total_elements);
        let v_len = database[0].len();
        let mut offset = 0;
        unsafe {
            database_flat.set_len(total_elements);
            for v in database.iter() {
                let src = v.as_ptr();
                let dest = database_flat.as_mut_ptr().add(offset);
                ptr::copy_nonoverlapping(src, dest, v_len);
                offset += v_len;
            }
        }
        
        database_flat
    }

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let vector_count = challenge.vector_database.len();
        let query_count = challenge.query_vectors.len();
        let vector_dim = challenge.query_vectors[0].len();
        let max_distance_sq = challenge.max_distance * challenge.max_distance;
        
        /* Prepare databases */
        let vector_database_flat = flatten_vector_database(&challenge.vector_database);
        let query_vectors_flat = flatten_vector_database(&challenge.query_vectors);
    
        /* Copy data to GPU */
        let d_vector_database = dev.htod_copy(vector_database_flat)?;
        let d_query_vectors = dev.htod_copy(query_vectors_flat)?;
        let d_result_indexes = dev.alloc_zeros::<i32>(query_count)?;
        dev.synchronize()?;
    
        /* Fire Kernel */
        unsafe {
            const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * std::mem::size_of::<i32>() as u32 * 2;
            funcs.remove("fast_search").unwrap().launch(
                LaunchConfig {
                    grid_dim: (query_count as u32, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: SHARED_MEM_SIZE,
                },
                (
                    &d_query_vectors,
                    &d_vector_database,
                    query_count as i32,
                    vector_count as i32,
                    vector_dim as i32,
                    max_distance_sq,
                    SHARED_MEM_SIZE as i32,
                    &d_result_indexes,
                ),
            )?;
        }
    
        /* Get data from kernel */
        let result_indexes = dev.dtoh_sync_copy(&d_result_indexes)?;
        let indexes: Vec<usize> = result_indexes.into_iter()
            .filter_map(|i| if i != std::i32::MAX { Some(i as usize) } else { None })
            .collect();
    
        if indexes.len() == query_count {
            Ok(Some(Solution { indexes }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
