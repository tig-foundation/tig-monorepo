use anyhow::Result;
use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
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
const TILE_DB: usize = 4096;
const WARPS_PER_BLOCK: u32 = 8;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    let nq_i32 = num_queries as i32;
    let db_i32 = database_size as i32;

    let mut d_best_dist = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_best_idx  = stream.alloc_zeros::<i32>(num_queries)?;
    let mut d_q_norms   = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_db_norms  = stream.alloc_zeros::<f32>(database_size)?;
    let mut d_dot       = stream.alloc_zeros::<f32>(num_queries * TILE_DB)?;

    let norms_func = module.load_function("compute_norms_v122")?;
    let find_func  = module.load_function("find_best_v122")?;

    let q_blk = (num_queries as u32 + 255) / 256;
    let dims_i32 = DIMS as i32;
    unsafe {
        stream.launch_builder(&norms_func)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_norms)
            .arg(&nq_i32)
            .arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    let db_blk = (database_size as u32 + 255) / 256;
    unsafe {
        stream.launch_builder(&norms_func)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_norms)
            .arg(&db_i32)
            .arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    let cublas = CudaBlas::new(stream.clone())?;

    let n_db_tiles = (database_size + TILE_DB - 1) / TILE_DB;

    for tile_idx in 0..n_db_tiles {
        let db_start = tile_idx * TILE_DB;
        let tile_len = TILE_DB.min(database_size - db_start);

        let db_tile = challenge.d_database_vectors.slice(db_start * DIMS..(db_start + tile_len) * DIMS);

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: tile_len as i32,
            n: nq_i32,
            k: DIMS as i32,
            alpha: -2.0f32,
            lda: DIMS as i32,
            ldb: DIMS as i32,
            beta: 0.0f32,
            ldc: tile_len as i32,
        };
        unsafe { cublas.gemm(cfg, &db_tile, &challenge.d_query_vectors, &mut d_dot)?; }

        let grid_y = (num_queries as u32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        let find_cfg = LaunchConfig {
            grid_dim: (1, grid_y, 1),
            block_dim: (32, WARPS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let db_start_i32 = db_start as i32;
        let tile_len_i32 = tile_len as i32;
        let first_tile: i32 = if tile_idx == 0 { 1 } else { 0 };
        unsafe {
            stream.launch_builder(&find_func)
                .arg(&d_q_norms)
                .arg(&d_db_norms)
                .arg(&d_dot)
                .arg(&mut d_best_idx)
                .arg(&mut d_best_dist)
                .arg(&db_start_i32)
                .arg(&tile_len_i32)
                .arg(&nq_i32)
                .arg(&first_tile)
                .launch(find_cfg)?;
        }
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
