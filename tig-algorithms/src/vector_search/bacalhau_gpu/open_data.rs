/*!
Copyright 2024 Louis Silva

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
 */

use anyhow::Result;

use tig_challenges::vector_search::*;

#[inline]
fn squared_distance(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let max_distance_sq = challenge.max_distance * challenge.max_distance;

    let indexes: Vec<Option<usize>> = challenge
        .query_vectors
        .iter()
        .map(|query| {
            challenge
                .vector_database
                .iter()
                .enumerate()
                .filter_map(|(i, vector)| {
                    let dist_sq = squared_distance(query, vector);
                    if dist_sq <= max_distance_sq {
                        Some((i, dist_sq))
                    } else {
                        None
                    }
                })
                .min_by(|(_, dist_sq1), (_, dist_sq2)| dist_sq1.partial_cmp(dist_sq2).unwrap())
                .map(|(i, _)| i)
        })
        .collect();

    if indexes.iter().all(Option::is_some) {
        Ok(Some(Solution {
            indexes: indexes.into_iter().map(Option::unwrap).collect(),
        }))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
mod gpu_optimization {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use cudarc::driver::{CudaDevice, CudaFunction};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
extern "C" __global__ void find_nearest_neighbors(
    const float* __restrict__ vector_database,
    const float* __restrict__ query_vectors,
    const int num_queries,
    int* results
) {
    #define FLT_MAX 3.402823466e+38F

    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    float min_dist = FLT_MAX;
    int nearest_idx = 0;

    for (int i=0; i < 100000; i++) {
        float dist = 0.0f;
        for (int j=0; j < 250; j++) {
            float diff = query_vectors[query_idx * 250 + j] - vector_database[i * 250 + j];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }

    results[query_idx] = nearest_idx;
}
    "#,
        funcs: &["find_nearest_neighbors"]
    });

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Option<Solution>> {
        let num_query_vectors: usize = challenge.query_vectors.len();

        let flattened_vector_database: Vec<f32> = challenge.vector_database.iter().flatten().cloned().collect();
        let flattened_query_vectors: Vec<f32> = challenge.query_vectors.iter().flatten().cloned().collect();

        let d_vector_database = dev.htod_sync_copy(&flattened_vector_database)?;
        let d_query_vectors = dev.htod_sync_copy(&flattened_query_vectors)?;
        let mut d_results = dev.alloc_zeros::<i32>(num_query_vectors)?;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((num_query_vectors + 255) / 256) as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            funcs
                .remove("find_nearest_neighbors")
                .unwrap()
                .launch(cfg, (
                    &d_vector_database,
                    &d_query_vectors,
                    num_query_vectors as i32,
                    &mut d_results,
                ))?;
        }

        let mut h_indexes: Vec<i32> = vec![0; num_query_vectors];
        dev.dtoh_sync_copy_into(&d_results, &mut h_indexes)?;

        let indexes: Vec<usize> = h_indexes.iter().map(|&index| index as usize).collect();

        Ok(Some(Solution { indexes }))
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimization::{cuda_solve_challenge, KERNEL};