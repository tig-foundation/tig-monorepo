/*!
Copyright 2025 Uncharted Trading

Licensed under the TIG Commercial License v2.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::vector_search::*;

pub fn solve_challenge(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    let simple_search_kernel = module.load_function("simple_search")?;

    let block_size = prop.maxThreadsPerBlock as u32;
    let database_size = challenge.database_size;
    let cfg = LaunchConfig {
        grid_dim: ((database_size + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut indexes = vec![0; challenge.difficulty.num_queries as usize];
    for i in 0..challenge.difficulty.num_queries {
        let mut d_distances = stream.alloc_zeros::<f32>(database_size as usize)?;

        unsafe {
            stream
                .launch_builder(&simple_search_kernel)
                .arg(&challenge.vector_dims)
                .arg(&challenge.database_size)
                .arg(&challenge.difficulty.num_queries)
                .arg(&challenge.d_database_vectors)
                .arg(&challenge.d_query_vectors)
                .arg(&i)
                .arg(&mut d_distances)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let distances = stream.memcpy_dtov(&d_distances)?;
        indexes[i as usize] = distances
            .iter()
            .enumerate()
            .fold((0, f32::MAX), |(idx, min), (i, &dist)| {
                if dist < min {
                    (i as i32, dist)
                } else {
                    (idx, min)
                }
            })
            .0 as usize;
    }

    Ok(Some(Solution { indexes }))
}
