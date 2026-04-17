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
    let mut d_best_dists = unsafe { stream.alloc::<f32>(num_queries as usize)? };

    let full_dim_chunks = vector_dims / 4;
    let tail_dims = vector_dims % 4;
    let blocked_block_elems: usize =
        full_dim_chunks as usize * 16usize * 4usize + tail_dims as usize * 16usize;

    let num_db_blocks: u32 = if database_size == 0 {
        0
    } else {
        (database_size - 1) / 16 + 1
    };
    let db_blocked_elems: usize = num_db_blocks as usize * blocked_block_elems;

    let mut d_database_blocked = unsafe { stream.alloc::<f32>(db_blocked_elems)? };

    let transform_database_kernel = module.load_function("transform_database_blocked")?;
    let search_kernel = module.load_function("batched_search")?;

    if num_db_blocks > 0 && vector_dims > 0 {
        let transform_config = LaunchConfig {
            grid_dim: (num_db_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&transform_database_kernel)
                .arg(&challenge.d_database_vectors)
                .arg(&mut d_database_blocked)
                .arg(&database_size_i)
                .arg(&vector_dims_i)
                .launch(transform_config)?;
        }
    }

    let shared_mem_bytes: u32 = if vector_dims > 0 && vector_dims <= 768 {
        16u32 * vector_dims * 4u32
    } else {
        0
    };

    let config = LaunchConfig {
        grid_dim: ((num_queries + 15) / 16, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes,
    };

    let batch_size = if vector_dims >= 512 { 80000u32 } else { 200000u32 };

    let num_batches = if database_size == 0 {
        0
    } else {
        (database_size + batch_size - 1) / batch_size
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
                0usize
            } else {
                idx as usize
            }
        })
        .collect();

    save_solution(&Solution { indexes })?;

    Ok(())
}

pub fn help() {
    println!("GPU Vector Search - Exact Search");
    println!();    
    println!("Main features:");
    println!("  - Exact solution solver");    
    println!("  - Uses batching to handle large databases efficiently");
    println!("  - Optimized memory layout for better performance");
    println!();
    println!("Notes:");
    println!("  - Batch size is chosen from the vector dimension");
    println!("  - 500b fuel needed on all tracks");
    println!();
    println!("Goal:");
    println!("  - Perfect quality with improved runtime");
}