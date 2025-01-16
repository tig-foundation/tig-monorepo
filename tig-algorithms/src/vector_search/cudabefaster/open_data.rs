/*!
Copyright 2024 OvErLoDe

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use tig_challenges::vector_search::*;

#[inline]
fn euclidean_distance_with_precomputed_norm(
    a_norm_sq: f32,
    b_norm_sq: f32,
    ab_dot_product: f32,
) -> f32 {
    (a_norm_sq + b_norm_sq - 2.0 * ab_dot_product).sqrt()
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
    let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;
    let max_distance: f32 = challenge.max_distance;

    // Pre-compute vector norms
    let vector_norms_sq: Vec<f32> = vector_database
        .iter()
        .map(|vector| vector.iter().map(|&val| val * val).sum())
        .collect();

    let sum_norms_sq: f32 = vector_norms_sq.iter().sum();
    let std_dev: f32 = 2.0 * (sum_norms_sq / vector_norms_sq.len() as f32).sqrt();

    let mut indexes: Vec<usize> = Vec::with_capacity(query_vectors.len());

    for query in query_vectors {
        let query_norm_sq: f32 = query.iter().map(|&val| val * val).sum();

        let mut closest_index: Option<usize> = None;
        let mut closest_distance: f32 = f32::MAX;

        for (idx, vector) in vector_database.iter().enumerate() {
            let vector_norm_sq = vector_norms_sq[idx];
            if (vector_norm_sq.sqrt() - query_norm_sq.sqrt()).abs() > std_dev {
                continue;
            }

            let ab_dot_product: f32 = query.iter().zip(vector).map(|(&x1, &x2)| x1 * x2).sum();

            let distance = euclidean_distance_with_precomputed_norm(
                query_norm_sq,
                vector_norm_sq,
                ab_dot_product,
            );

            if distance <= max_distance {
                closest_index = Some(idx);
                break;
            } else if distance < closest_distance {
                closest_index = Some(idx);
                closest_distance = distance;
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
    use std::{collections::HashMap, sync::Arc};
    use cudarc::{
        driver::{CudaDevice, DriverError, LaunchConfig, CudaFunction, CudaSlice, LaunchAsync},
    };
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        extern "C" __global__ void calculate_distances_batch(
            const float* query_vectors,
            const float* vector_database,
            int num_vectors,
            int vector_size,
            int num_queries,
            float* distances,
            float* norms_query,
            float* norms_db
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int qid = blockIdx.y; 

            if (idx < num_vectors && qid < num_queries) {
                extern __shared__ float s_query[];

                // Compute L2 norm for query vector
                if (threadIdx.x < vector_size) {
                    s_query[threadIdx.x] = query_vectors[qid * vector_size + threadIdx.x];
                }
                __syncthreads();

                float query_norm_sq = 0.0f;
                for (int i = 0; i < vector_size; ++i) {
                    query_norm_sq += s_query[i] * s_query[i];
                }
                norms_query[qid] = sqrtf(query_norm_sq);

                // Compute L2 norm for database vector
                float db_norm_sq = 0.0f;
                for (int i = 0; i < vector_size; ++i) {
                    float db_val = vector_database[idx * vector_size + i];
                    db_norm_sq += db_val * db_val;
                }
                norms_db[idx] = sqrtf(db_norm_sq);

                // Compute distance using precomputed norms
                float distance = 0.0f;
                for (int i = 0; i < vector_size; ++i) {
                    float db_val = vector_database[idx * vector_size + i];
                    distance += (s_query[i] - db_val) * (s_query[i] - db_val);
                }
                distances[qid * num_vectors + idx] = sqrtf(distance);
            }
        }
        "#,
        funcs: &["calculate_distances_batch"],
    });

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Option<Solution>, DriverError> {
        let vector_database: &Vec<Vec<f32>> = &challenge.vector_database;
        let query_vectors: &Vec<Vec<f32>> = &challenge.query_vectors;
        let max_distance: f32 = challenge.max_distance;

        let num_vectors = vector_database.len();
        let vector_size = vector_database[0].len();
        let num_queries = query_vectors.len();

        // Flatten data
        let vector_db_flat: Vec<f32> = vector_database.iter().flatten().cloned().collect();
        let query_vectors_flat: Vec<f32> = query_vectors.iter().flatten().cloned().collect();

        // Allocate device memory
        let vector_db_dev = dev.htod_sync_copy(&vector_db_flat)?;
        let query_dev = dev.htod_sync_copy(&query_vectors_flat)?;
        let norms_query_dev: CudaSlice<f32> = unsafe { dev.alloc(num_queries)? };
        let norms_db_dev: CudaSlice<f32> = unsafe { dev.alloc(num_vectors)? };

        // Allocate distances buffer on the device
        let distances_dev: CudaSlice<f32> = unsafe { dev.alloc::<f32>(num_vectors * num_queries)? };

        // Create a CUDA stream
        let stream = dev.fork_default_stream()?;

        // Configure kernel launch parameters
        let block_dim = (512, 1, 1); // Maximize block size
        let grid_dim = (
            ((num_vectors + block_dim.0 as usize - 1) / block_dim.0 as usize) as u32,
            num_queries as u32,
            1,
        );
        let shared_mem_bytes = vector_size as u32 * std::mem::size_of::<f32>() as u32;

        let func = funcs.get_mut("calculate_distances_batch").unwrap().clone();

        // Launch the kernel
        unsafe {
            func.launch_on_stream(
                &stream,
                LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes,
                },
                (
                    &query_dev,
                    &vector_db_dev,
                    num_vectors as i32,
                    vector_size as i32,
                    num_queries as i32,
                    &distances_dev,
                    &norms_query_dev,
                    &norms_db_dev,
                ),
            )?;
        }

        // Wait for stream to complete
        dev.wait_for(&stream)?;

        // Allocate buffer for results on the host and copy data back
        let mut distances = vec![f32::MAX; num_vectors * num_queries];
        dev.dtoh_sync_copy_into(&distances_dev, &mut distances)?;

        // Process results
        let mut indexes = Vec::with_capacity(num_queries);
        for q in 0..num_queries {
            if let Some((index, _)) = distances[(q * num_vectors)..((q + 1) * num_vectors)]
                .iter()
                .enumerate()
                .filter(|&(_, &d)| d <= max_distance)
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                indexes.push(index);
            } else {
                return Ok(None);
            }
        }

        Ok(Some(Solution { indexes }))
    }
}

#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};













 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 

 
