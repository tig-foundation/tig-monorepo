use anyhow::Result;
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use super::*;
#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {}
    }
}

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
const CHUNK_DB: usize = 4096;

fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let num_queries   = challenge.num_queries   as usize;
    let database_size = challenge.database_size as usize;
    let nq_i32       = num_queries   as i32;
    let db_i32       = database_size as i32;
    let dims_i32     = DIMS     as i32;
    let dims_pad_i32 = DIMS_PAD as i32;

    let nq_pad = pad16(num_queries);
    let db_pad = pad16(database_size);

    let mut d_best_dist      = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_best_idx       = stream.alloc_zeros::<i32>(num_queries)?;
    let mut d_q_norms        = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_db_norms       = stream.alloc_zeros::<f32>(database_size)?;
    let mut d_q_fp16_padded  = stream.alloc_zeros::<u16>(nq_pad * DIMS_PAD)?;
    let mut d_db_fp16_padded = stream.alloc_zeros::<u16>(db_pad * DIMS_PAD)?;

    let norms_func  = module.load_function("compute_norms_v122")?;
    let conv_func   = module.load_function("convert_fp32_to_fp16_padded")?;
    let chunk_func  = module.load_function("fused_wmma_chunk_search_t20")?;
    let reduce_func = module.load_function("reduce_chunk_bests_t20")?;

    let q_blk  = (num_queries   as u32 + 255) / 256;
    let db_blk = (database_size as u32 + 255) / 256;

    unsafe {
        stream.launch_builder(&norms_func)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_norms)
            .arg(&nq_i32)
            .arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&norms_func)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_norms)
            .arg(&db_i32)
            .arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    unsafe {
        stream.launch_builder(&conv_func)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_fp16_padded)
            .arg(&nq_i32)
            .arg(&dims_i32)
            .arg(&dims_pad_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&conv_func)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_fp16_padded)
            .arg(&db_i32)
            .arg(&dims_i32)
            .arg(&dims_pad_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    let num_chunks = (database_size + CHUNK_DB - 1) / CHUNK_DB;
    let num_chunks_i32 = num_chunks as i32;
    let chunk_db_i32 = CHUNK_DB as i32;
    let nq_pad_i32 = nq_pad as i32;

    let mut d_chunk_dists = stream.alloc_zeros::<f32>(num_chunks * nq_pad)?;
    let mut d_chunk_idxs  = stream.alloc_zeros::<i32>(num_chunks * nq_pad)?;

    let grid_x = (nq_pad as u32) / 16;
    let grid_y = num_chunks as u32;
    unsafe {
        stream.launch_builder(&chunk_func)
            .arg(&d_q_fp16_padded)
            .arg(&d_db_fp16_padded)
            .arg(&d_q_norms)
            .arg(&d_db_norms)
            .arg(&mut d_chunk_dists)
            .arg(&mut d_chunk_idxs)
            .arg(&nq_i32)
            .arg(&db_i32)
            .arg(&nq_pad_i32)
            .arg(&chunk_db_i32)
            .launch(LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 16 * 16 * 4   +
                                  16 * 256 * 2,
            })?;
    }

    let reduce_grid = (num_queries as u32 + 255) / 256;
    unsafe {
        stream.launch_builder(&reduce_func)
            .arg(&d_chunk_dists)
            .arg(&d_chunk_idxs)
            .arg(&mut d_best_dist)
            .arg(&mut d_best_idx)
            .arg(&nq_i32)
            .arg(&num_chunks_i32)
            .arg(&nq_pad_i32)
            .launch(LaunchConfig {
                grid_dim: (reduce_grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    stream.synchronize()?;

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_best_idx)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| if idx < 0 || idx >= db_i32 { 0 } else { idx as usize })
        .collect();

    save_solution(&Solution { indexes })?;
    Ok(())
}

pub fn help() {
}
