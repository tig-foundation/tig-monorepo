/*!
Copyright 2025 Uncharted Trading

Licensed under the TIG Open Data License v2.0 or (at your option) any later version
(the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

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
    let d_seed = stream.memcpy_stod(&challenge.seed.to_vec()).unwrap();
    let d_solution_indexes = stream
        .alloc_zeros::<u32>(challenge.difficulty.num_queries as usize)
        .unwrap();

    let simple_search_kernel = module.load_function("simple_search").unwrap();

    let threads_per_block = prop.maxThreadsPerBlock as u32;
    let blocks = challenge.difficulty.num_queries;

    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: (threads_per_block + 250) * 4,
    };

    let mut builder = stream.launch_builder(&simple_search_kernel);
    unsafe {
        builder
            .arg(&d_seed)
            .arg(&(challenge.vector_dims as u32))
            .arg(&(challenge.database_size as u32))
            .arg(&(challenge.difficulty.num_queries as u32))
            .arg(&d_solution_indexes)
            .launch(cfg)
            .unwrap();
    }

    stream.synchronize()?;

    let indexes = stream
        .memcpy_dtov(&d_solution_indexes)
        .unwrap()
        .into_iter()
        .map(|x| x as usize)
        .collect::<Vec<_>>();

    Ok(Some(Solution { indexes }))
}
