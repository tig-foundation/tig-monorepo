/*!
Copyright 2024 Mateus Melo

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challengedetect your algorithm's challenge
use anyhow::Result;
use tig_challenges::vector_search::{Challenge, Solution};

fn squared_distance(v1: &[f32], v2: &[f32]) -> f32 {
    // Manual loop for performance
    let mut sum = 0.0;
    let len = v1.len();
    for i in 0..len {
        let diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    sum
}

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let max_distance_sq = challenge.max_distance * challenge.max_distance;

    // Pre-allocate capacity for indexes
    let mut indexes = Vec::with_capacity(challenge.query_vectors.len());

    for query in &challenge.query_vectors {
        let mut found = false;

        for (i, vector) in challenge.vector_database.iter().enumerate() {
            if squared_distance(query, vector) <= max_distance_sq {
                indexes.push(i);
                found = true;
                break; // Exit early if a match is found
            }
        }

        // If no matches found for this query, we can exit early
        if !found {
            continue;
        }
    }

    if indexes.len() == challenge.query_vectors.len() {
        Ok(Some(Solution { indexes }))
    } else {
        Ok(None)
    }
}
// Important! Do not include any tests in this file, it will result in your submission being rejected

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::sync::Arc;
    use std::collections::HashMap;
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        src: r#"
        extern "C" __global__ void computeSquaredDistance(
            const float* __restrict__ queryVectors,
            const float* __restrict__ databaseVectors,
            const int numQueries,
            const int vectorSize,
            const int databaseSize,
            float* __restrict__ distances
        ){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx<numQueries*databaseSize){
                int queryIdx = idx / databaseSize;
                int databaseIdx = idx % databaseSize;
                float sum = 0.0f;
                for(int i=0; i<vectorSize; i++){
                    float diff = queryVectors[queryIdx*vectorSize + i] - databaseVectors[databaseIdx*vectorSize + i];
                    sum += diff * diff;
                }
                distances[idx] = sum;
            }
        }
        "#,
        funcs: &["computeSquaredDistance"],
    });

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        let num_queries = challenge.query_vectors.len();   
        let vector_size = challenge.query_vectors[0].len();
        let database_size = challenge.vector_database.len();

       // Allocate device memory
        let mut query_vectors_device =  dev.alloc_zeros::<f32>(num_queries * vector_size)?;
        let mut database_vectors_device = dev.alloc_zeros::<f32>(database_size * vector_size)?;
        let mut distances_device = dev.alloc_zeros::<f32>(num_queries * database_size)?;

        // Copy query vectors and database vectors to device
        dev.htod_copy_into(challenge.query_vectors.concat(), &mut query_vectors_device)?;
        dev.htod_copy_into(challenge.vector_database.concat(), &mut database_vectors_device)?;

        // Execute the kernel
        let threads_per_block = 256;
        let blocks = (num_queries * database_size + threads_per_block - 1) / threads_per_block;
        unsafe {
            funcs
                .remove("computeSquaredDistance")
                .unwrap()
                .launch(
                    LaunchConfig {
                        block_dim: (threads_per_block.try_into().unwrap(),1,1),
                        grid_dim: (blocks.try_into().unwrap(),1,1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &query_vectors_device,
                        &database_vectors_device,
                        num_queries as i32,
                        vector_size as i32,
                        database_size as i32,
                        &mut distances_device,
                    ),
                )?;
        }

        // Copy the result back to host
        let mut distances_host = vec![0.0; num_queries * database_size];
        dev.dtoh_sync_copy_into(&distances_device,&mut distances_host)?;

        // Process results to determine matches
        let max_distance_sq = challenge.max_distance * challenge.max_distance;
        let mut indexes = Vec::new();
        for query_idx in 0..num_queries {
            for vector_idx in 0..database_size {
                let distance = distances_host[query_idx * database_size + vector_idx];
                if distance <= max_distance_sq {
                    indexes.push(vector_idx);
                    break; // Exit early if a match is found
                }
            }
        }

        if indexes.len() == num_queries {
            Ok(Some(Solution { indexes }))
        } else {
            Ok(None)
        }
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
