
use anyhow::{anyhow, Result};
use cudarc::cublas::{result as cublas_result, sys as cublas_sys, CudaBlas};
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    DevicePtr, DevicePtrMut, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::ffi::c_void;
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {}
    }
}

const DIMS: usize = 250;
const TILE_DB: usize = 65536;
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
    let dims_i32 = DIMS as i32;

    let mut d_best_dist = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_best_idx  = stream.alloc_zeros::<i32>(num_queries)?;
    let mut d_q_norms   = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_db_norms  = stream.alloc_zeros::<f32>(database_size)?;

    let mut d_q_fp16  = stream.alloc_zeros::<u16>(num_queries * DIMS)?;
    let mut d_db_fp16 = stream.alloc_zeros::<u16>(database_size * DIMS)?;

    let mut d_dot = stream.alloc_zeros::<u16>(num_queries * TILE_DB)?;

    let norms_func   = module.load_function("compute_norms_v122")?;
    let convert_func = module.load_function("convert_fp32_to_fp16")?;
    let find_func    = module.load_function("find_best_fp16_t20")?;

    let q_blk  = (num_queries as u32 + 255) / 256;
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

    let q_elem  = (num_queries * DIMS) as i32;
    let db_elem = (database_size * DIMS) as i32;
    let conv_q_blk  = (q_elem  as u32 + 255) / 256;
    let conv_db_blk = (db_elem as u32 + 255) / 256;
    unsafe {
        stream.launch_builder(&convert_func)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_fp16)
            .arg(&q_elem)
            .launch(LaunchConfig { grid_dim: (conv_q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&convert_func)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_fp16)
            .arg(&db_elem)
            .launch(LaunchConfig { grid_dim: (conv_db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    let cublas = CudaBlas::new(stream.clone())?;

    let n_db_tiles = (database_size + TILE_DB - 1) / TILE_DB;

    let alpha_f32: f32 = -2.0f32;
    let beta_f32:  f32 =  0.0f32;

    for tile_idx in 0..n_db_tiles {
        let db_start = tile_idx * TILE_DB;
        let tile_len = TILE_DB.min(database_size - db_start);

        let db_fp16_tile = d_db_fp16.slice(db_start * DIMS..(db_start + tile_len) * DIMS);

        {
            let (db_ptr,  _db_sync)  = db_fp16_tile.device_ptr(&stream);
            let (q_ptr,   _q_sync)   = d_q_fp16.device_ptr(&stream);
            let (dot_ptr, _dot_sync) = d_dot.device_ptr_mut(&stream);
            unsafe {
                let status = cublas_result::gemm_ex(
                    *cublas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    tile_len as i32,
                    nq_i32,
                    dims_i32,
                    &alpha_f32 as *const f32 as *const c_void,
                    db_ptr as *const c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    dims_i32,
                    q_ptr as *const c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    dims_i32,
                    &beta_f32 as *const f32 as *const c_void,
                    dot_ptr as *mut c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    tile_len as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                );
                status.map_err(|e| anyhow!("cublasGemmEx failed: {:?}", e))?;
            }
        }

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
    println!("AutoVector t20_i7 — FP16 GEMM TILE_DB=65536 (COMPUTE_32F + CUDA_R_16F I/O)");
}
