/*!
Copyright 2024 TIG Foundation

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use rand::{rngs::StdRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(challenge.seeds[0] as u64);
    let num_variables = challenge.difficulty.num_variables;
    let mut variables: Vec<bool> = (0..num_variables).map(|_| rng.gen::<bool>()).collect();

    // Pre-generate a bunch of random integers
    // IMPORTANT! When generating random numbers, never use usize! usize bytes varies from system to system
    let rand_ints: Vec<usize> = (0..2 * num_variables)
        .map(|_| rng.gen_range(0..1_000_000_000u32) as usize)
        .collect();

    for i in 0..num_variables {
        // Evaluate clauses and find any that are unsatisfied
        let substituted: Vec<bool> = challenge
            .clauses
            .iter()
            .map(|clause| {
                clause.iter().any(|&literal| {
                    let var_idx = literal.abs() as usize - 1;
                    let var_value = variables[var_idx];
                    (literal > 0 && var_value) || (literal < 0 && !var_value)
                })
            })
            .collect();

        let unsatisfied_clauses: Vec<usize> = substituted
            .iter()
            .enumerate()
            .filter_map(|(idx, &satisfied)| if !satisfied { Some(idx) } else { None })
            .collect();

        let num_unsatisfied_clauses = unsatisfied_clauses.len();
        if num_unsatisfied_clauses == 0 {
            break;
        }

        // Flip the value of a random variable from a random unsatisfied clause
        let rand_unsatisfied_clause_idx = rand_ints[2 * i] % num_unsatisfied_clauses;
        let rand_unsatisfied_clause = unsatisfied_clauses[rand_unsatisfied_clause_idx];
        let rand_variable_idx = rand_ints[2 * i + 1] % 3;
        let rand_variable =
            challenge.clauses[rand_unsatisfied_clause][rand_variable_idx].abs() as usize - 1;
        variables[rand_variable] = !variables[rand_variable];
    }

    Ok(Some(Solution { variables }))
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = Some(CudaKernel {
        // Example CUDA code from https://github.com/coreylowman/cudarc/blob/main/examples/matmul-kernel.rs
        src: r#"
extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}
"#,
        funcs: &["matmul"],
    });

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        // Example CUDA code from https://github.com/coreylowman/cudarc/blob/main/examples/matmul-kernel.rs
        let start = std::time::Instant::now();

        let a_host = [1.0f32, 2.0, 3.0, 4.0];
        let b_host = [1.0f32, 2.0, 3.0, 4.0];
        let mut c_host = [0.0f32; 4];

        let a_dev = dev.htod_sync_copy(&a_host)?;
        let b_dev = dev.htod_sync_copy(&b_host)?;
        let mut c_dev = dev.htod_sync_copy(&c_host)?;

        println!("Copied in {:?}", start.elapsed());

        let cfg = LaunchConfig {
            block_dim: (2, 2, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            funcs
                .remove("matmul")
                .unwrap()
                .launch(cfg, (&a_dev, &b_dev, &mut c_dev, 2i32))
        }?;

        dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
        println!("Found {:?} in {:?}", c_host, start.elapsed());

        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
