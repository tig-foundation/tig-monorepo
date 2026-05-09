use anyhow::{anyhow, Result};
use cudarc::cublas::{
    result as cublas_result, sys as cublas_sys, CudaBlas, Gemm, GemmConfig,
};
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::ffi::c_void;
use std::sync::Arc;
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
const WARPS_PER_BLOCK: u32 = 8;
const FP16_TILE_DB_LARGE: usize = 65_536;
const FP16_TILE_DB_SMALL: usize = 32_768;
const FP32_TILE_DB: usize = 4_096;

fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

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

    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!(
            "there_v7_cublas expects {} dimensions, got {}",
            DIMS,
            challenge.vector_dims
        ));
    }

    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    let num_queries_i = num_queries as i32;
    let database_size_i = database_size as i32;

    let mut d_results = stream.alloc_zeros::<i32>(num_queries)?;
    let mut d_best_dists = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_query_norms = stream.alloc_zeros::<f32>(num_queries)?;
    let mut d_database_norms = stream.alloc_zeros::<f32>(database_size)?;

    let norm_kernel = module.load_function("there_v7_compute_norms_250")?;
    let q_blocks = (challenge.num_queries + 255) / 256;
    let db_blocks = (challenge.database_size + 255) / 256;

    unsafe {
        stream
            .launch_builder(&norm_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_query_norms)
            .arg(&num_queries_i)
            .launch(LaunchConfig {
                grid_dim: (q_blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&norm_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_database_norms)
            .arg(&database_size_i)
            .launch(LaunchConfig {
                grid_dim: (db_blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let cublas = CudaBlas::new(stream.clone())?;

    match num_queries {
        7_000 => solve_fp16_padded(
            challenge,
            &module,
            &stream,
            &cublas,
            &d_query_norms,
            &d_database_norms,
            &mut d_results,
            &mut d_best_dists,
            FP16_TILE_DB_LARGE,
        )?,
        9_000 => solve_fp16_padded(
            challenge,
            &module,
            &stream,
            &cublas,
            &d_query_norms,
            &d_database_norms,
            &mut d_results,
            &mut d_best_dists,
            FP16_TILE_DB_LARGE,
        )?,
        11_000 | 13_000 | 15_000 => solve_fp16_padded(
            challenge,
            &module,
            &stream,
            &cublas,
            &d_query_norms,
            &d_database_norms,
            &mut d_results,
            &mut d_best_dists,
            FP16_TILE_DB_SMALL,
        )?,
        _ => solve_fp32_gemm(
            challenge,
            &module,
            &stream,
            &cublas,
            &d_query_norms,
            &d_database_norms,
            &mut d_results,
            &mut d_best_dists,
        )?,
    }

    stream.synchronize()?;
    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes = result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= database_size_i {
                0usize
            } else {
                idx as usize
            }
        })
        .collect();

    save_solution(&Solution { indexes })?;
    Ok(())
}

fn solve_fp32_gemm(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
) -> Result<()> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    let num_queries_i = num_queries as i32;
    let mut d_dot = stream.alloc_zeros::<f32>(num_queries * FP32_TILE_DB)?;
    let reduce_kernel = module.load_function("there_v7_reduce_gemm_tile_f32")?;

    for tile_idx in 0..((database_size + FP32_TILE_DB - 1) / FP32_TILE_DB) {
        let db_start = tile_idx * FP32_TILE_DB;
        let tile_len = FP32_TILE_DB.min(database_size - db_start);
        let db_tile = challenge
            .d_database_vectors
            .slice(db_start * DIMS..(db_start + tile_len) * DIMS);

        let cfg = GemmConfig {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            m: tile_len as i32,
            n: num_queries_i,
            k: DIMS as i32,
            alpha: -2.0f32,
            lda: DIMS as i32,
            ldb: DIMS as i32,
            beta: 0.0f32,
            ldc: tile_len as i32,
        };
        unsafe {
            cublas.gemm(cfg, &db_tile, &challenge.d_query_vectors, &mut d_dot)?;
        }

        launch_reduce_f32(
            stream,
            &reduce_kernel,
            d_query_norms,
            d_database_norms,
            &d_dot,
            d_results,
            d_best_dists,
            db_start,
            tile_len,
            num_queries,
            tile_idx == 0,
        )?;
    }

    Ok(())
}

fn solve_fp16_padded(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
    tile_db: usize,
) -> Result<()> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    let nq_pad = pad16(num_queries);
    let count_q = num_queries as i32;
    let count_db = database_size as i32;
    let dims_i = DIMS as i32;
    let dims_pad_i = DIMS_PAD as i32;

    let mut d_q_half = stream.alloc_zeros::<u16>(nq_pad * DIMS_PAD)?;
    let mut d_db_half = stream.alloc_zeros::<u16>(database_size * DIMS_PAD)?;
    let mut d_dot = stream.alloc_zeros::<u16>(nq_pad * tile_db)?;

    let pack_kernel = module.load_function("there_v7_pack_fp16_250_to_256")?;
    let reduce_kernel = module.load_function("there_v7_reduce_gemm_tile_fp16")?;

    unsafe {
        stream
            .launch_builder(&pack_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_half)
            .arg(&count_q)
            .arg(&dims_i)
            .arg(&dims_pad_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.num_queries + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&pack_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_half)
            .arg(&count_db)
            .arg(&dims_i)
            .arg(&dims_pad_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.database_size + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    run_fp16_tiles(
        stream,
        cublas,
        &reduce_kernel,
        &d_q_half,
        &d_db_half,
        &mut d_dot,
        d_query_norms,
        d_database_norms,
        d_results,
        d_best_dists,
        database_size,
        num_queries,
        nq_pad,
        DIMS_PAD,
        tile_db,
    )
}

fn solve_fp16_unpadded(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
    _tile_db: usize,
) -> Result<()> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    let q_elems = (num_queries * DIMS) as i32;
    let db_elems = (database_size * DIMS) as i32;

    let mut d_q_half = stream.alloc_zeros::<u16>(num_queries * DIMS)?;
    let mut d_db_half = stream.alloc_zeros::<u16>(database_size * DIMS)?;
    let mut d_dot = stream.alloc_zeros::<u16>(num_queries * FP16_TILE_DB_LARGE)?;

    let pack_kernel = module.load_function("there_v7_pack_fp16_linear")?;
    let reduce_kernel = module.load_function("there_v7_reduce_gemm_tile_fp16")?;

    unsafe {
        stream
            .launch_builder(&pack_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_half)
            .arg(&q_elems)
            .launch(LaunchConfig {
                grid_dim: (((q_elems as u32) + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&pack_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_half)
            .arg(&db_elems)
            .launch(LaunchConfig {
                grid_dim: (((db_elems as u32) + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    run_fp16_tiles(
        stream,
        cublas,
        &reduce_kernel,
        &d_q_half,
        &d_db_half,
        &mut d_dot,
        d_query_norms,
        d_database_norms,
        d_results,
        d_best_dists,
        database_size,
        num_queries,
        num_queries,
        DIMS,
        FP16_TILE_DB_LARGE,
    )
}

fn run_fp16_tiles(
    stream: &Arc<CudaStream>,
    cublas: &CudaBlas,
    reduce_kernel: &cudarc::driver::safe::CudaFunction,
    d_q_half: &CudaSlice<u16>,
    d_db_half: &CudaSlice<u16>,
    d_dot: &mut CudaSlice<u16>,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
    database_size: usize,
    num_queries: usize,
    gemm_queries: usize,
    gemm_dims: usize,
    tile_db: usize,
) -> Result<()> {
    let alpha = -2.0f32;
    let beta = 0.0f32;

    for tile_idx in 0..((database_size + tile_db - 1) / tile_db) {
        let db_start = tile_idx * tile_db;
        let tile_len = tile_db.min(database_size - db_start);
        let db_tile = d_db_half.slice(db_start * gemm_dims..(db_start + tile_len) * gemm_dims);

        {
            let (db_ptr, _db_guard) = db_tile.device_ptr(stream);
            let (q_ptr, _q_guard) = d_q_half.device_ptr(stream);
            let (dot_ptr, _dot_guard) = d_dot.device_ptr_mut(stream);

            unsafe {
                let status = cublas_result::gemm_ex(
                    *cublas.handle(),
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    tile_len as i32,
                    gemm_queries as i32,
                    gemm_dims as i32,
                    &alpha as *const f32 as *const c_void,
                    db_ptr as *const c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    gemm_dims as i32,
                    q_ptr as *const c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    gemm_dims as i32,
                    &beta as *const f32 as *const c_void,
                    dot_ptr as *mut c_void,
                    cublas_sys::cudaDataType::CUDA_R_16F,
                    tile_len as i32,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                    cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                );
                status.map_err(|e| anyhow!("cublasGemmEx failed: {:?}", e))?;
            }
        }

        launch_reduce_fp16(
            stream,
            reduce_kernel,
            d_query_norms,
            d_database_norms,
            d_dot,
            d_results,
            d_best_dists,
            db_start,
            tile_len,
            num_queries,
            tile_idx == 0,
        )?;
    }

    Ok(())
}

fn launch_reduce_f32(
    stream: &Arc<CudaStream>,
    reduce_kernel: &cudarc::driver::safe::CudaFunction,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_dot: &CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
    db_start: usize,
    tile_len: usize,
    num_queries: usize,
    first_tile: bool,
) -> Result<()> {
    let db_start_i = db_start as i32;
    let tile_len_i = tile_len as i32;
    let num_queries_i = num_queries as i32;
    let first_tile_i = if first_tile { 1i32 } else { 0i32 };
    unsafe {
        stream
            .launch_builder(reduce_kernel)
            .arg(d_query_norms)
            .arg(d_database_norms)
            .arg(d_dot)
            .arg(d_results)
            .arg(d_best_dists)
            .arg(&db_start_i)
            .arg(&tile_len_i)
            .arg(&num_queries_i)
            .arg(&first_tile_i)
            .launch(reduce_config(num_queries))?;
    }
    Ok(())
}

fn launch_reduce_fp16(
    stream: &Arc<CudaStream>,
    reduce_kernel: &cudarc::driver::safe::CudaFunction,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_dot: &CudaSlice<u16>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
    db_start: usize,
    tile_len: usize,
    num_queries: usize,
    first_tile: bool,
) -> Result<()> {
    let db_start_i = db_start as i32;
    let tile_len_i = tile_len as i32;
    let num_queries_i = num_queries as i32;
    let first_tile_i = if first_tile { 1i32 } else { 0i32 };
    unsafe {
        stream
            .launch_builder(reduce_kernel)
            .arg(d_query_norms)
            .arg(d_database_norms)
            .arg(d_dot)
            .arg(d_results)
            .arg(d_best_dists)
            .arg(&db_start_i)
            .arg(&tile_len_i)
            .arg(&num_queries_i)
            .arg(&first_tile_i)
            .launch(reduce_config(num_queries))?;
    }
    Ok(())
}

fn reduce_config(num_queries: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, (num_queries as u32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1),
        block_dim: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    }
}

pub fn help() {
    println!("there_v7 - GEMM-based exhaustive vector search");
    println!("Uses cuBLAS dot-product tiles with CUDA norm and argmin reduction kernels.");
    println!("Standard tracks use padded FP16 GEMM for tensor-core throughput and fuel efficiency.");
}
