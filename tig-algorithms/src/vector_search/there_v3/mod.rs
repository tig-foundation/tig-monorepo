
use anyhow::Result;
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let _ = save_solution(&Solution {
        indexes: vec![0; challenge.num_queries as usize],
    });

    let num_queries = challenge.num_queries as u32;
    let database_size = challenge.database_size as u32;
    let vector_dims = challenge.vector_dims as u32;

    let num_queries_i = num_queries as i32;
    let database_size_i = database_size as i32;
    let vector_dims_i = vector_dims as i32;

    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    let mut d_best_dists = stream.alloc_zeros::<f32>(num_queries as usize)?;

    let num_db_blocks: u32 = if database_size == 0 {
        0
    } else {
        (database_size - 1) / 16 + 1
    };
    let blocked_elems: usize = num_db_blocks as usize * 16usize * vector_dims as usize;

    let mut d_database_blocked = unsafe { stream.alloc::<f32>(blocked_elems)? };

    let transform_kernel = module.load_function("transform_database_blocked")?;
    let init_kernel = module.load_function("init_best_dists")?;
    let search_kernel = module.load_function("batched_search")?;

    if num_db_blocks > 0 && vector_dims > 0 {
        let transform_config = LaunchConfig {
            grid_dim: (num_db_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&transform_kernel)
                .arg(&challenge.d_database_vectors)
                .arg(&mut d_database_blocked)
                .arg(&database_size_i)
                .arg(&vector_dims_i)
                .launch(transform_config)?;
        }
    }

    let init_config = LaunchConfig {
        grid_dim: ((num_queries + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&init_kernel)
            .arg(&mut d_best_dists)
            .arg(&num_queries_i)
            .launch(init_config)?;
    }

    let batch_size = if vector_dims >= 512 { 80000u32 } else { 200000u32 };
    let num_batches = (database_size + batch_size - 1) / batch_size;

    let smem_needed = 16usize * vector_dims as usize * 4usize;
    let shared_mem_bytes = if smem_needed <= 49152 {
        smem_needed as u32
    } else {
        0u32
    };

    let config = LaunchConfig {
        grid_dim: ((num_queries + 15) / 16, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes,
    };

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(database_size);
        let batch_count = batch_end - batch_start;

        let batch_start_i = batch_start as i32;
        let batch_count_i = batch_count as i32;

        unsafe {
            stream
                .launch_builder(&search_kernel)
                .arg(&challenge.d_query_vectors)
                .arg(&d_database_blocked)
                .arg(&mut d_results)
                .arg(&mut d_best_dists)
                .arg(&num_queries_i)
                .arg(&vector_dims_i)
                .arg(&batch_start_i)
                .arg(&batch_count_i)
                .launch(config)?;
        }
    }

    stream.synchronize()?;
    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;

    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= database_size as i32 {
                0
            } else {
                idx as usize
            }
        })
        .collect();

    save_solution(&Solution { indexes })?;

    Ok(())
}

pub fn help() {
    println!("GPU-Accelerated Vector Search - Exhaustive Brute Force");
    println!();
    println!("Algorithm: Batched exhaustive search with blocked-database layout");
    println!("Quality: At argmin ceiling (finds exact nearest neighbor for all queries)");
    println!();
    println!("Key optimizations:");
    println!("  - Half-warp per query (16 lanes), 2 queries per warp => higher query parallelism");
    println!("  - 256 threads per block for strong occupancy (16 queries per block)");
    println!("  - Database pre-transform into blocked AoSoA layout for coalesced per-dimension loads");
    println!("  - Adaptive batch sizing (80K for high-dim, 200K for low-dim)");
    println!();
    println!("Performance: Higher throughput via improved global memory transaction efficiency");
}
