use anyhow::{anyhow, Result};
use cudarc::cublas::{sys as cublas_sys, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    CudaSlice, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
const WARPS_PER_BLOCK: u32 = 8;
const WMMA_WARPS_PER_BLOCK: u32 = 4;
const WMMA_QUERIES_PER_BLOCK: u32 = 16 * WMMA_WARPS_PER_BLOCK;
const FP32_TILE_DB: usize = 4_096;
const WMMA_CHUNK_DB_7K: usize = 16_384;
const WMMA_CHUNK_DB_9K: usize = 4_096;
const WMMA_CHUNK_DB_11K: usize = 8_192;
const WMMA_CHUNK_DB_13K: usize = 4_096;
const WMMA_CHUNK_DB_15K: usize = 4_096;

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
    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!(
            "there_v9 expects {} dimensions, got {}",
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

    match num_queries {
        7_000 | 9_000 | 11_000 | 13_000 | 15_000 => solve_wmma_chunked(
            challenge,
            &module,
            &stream,
            &mut d_query_norms,
            &mut d_database_norms,
            &mut d_results,
            &mut d_best_dists,
            match num_queries {
                7_000 => WMMA_CHUNK_DB_7K,
                9_000 => WMMA_CHUNK_DB_9K,
                11_000 => WMMA_CHUNK_DB_11K,
                13_000 => WMMA_CHUNK_DB_13K,
                15_000 => WMMA_CHUNK_DB_15K,
                _ => unreachable!(),
            },
        )?,
        _ => {
            compute_norms(
                challenge,
                &module,
                &stream,
                &mut d_query_norms,
                &mut d_database_norms,
                num_queries_i,
                database_size_i,
            )?;
            let cublas = CudaBlas::new(stream.clone())?;
            solve_fp32_gemm(
                challenge,
                &module,
                &stream,
                &cublas,
                &d_query_norms,
                &d_database_norms,
                &mut d_results,
                &mut d_best_dists,
            )?
        }
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

fn compute_norms(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    d_query_norms: &mut CudaSlice<f32>,
    d_database_norms: &mut CudaSlice<f32>,
    num_queries_i: i32,
    database_size_i: i32,
) -> Result<()> {
    let norm_kernel = module.load_function("there_v7_compute_norms_250")?;
    unsafe {
        stream
            .launch_builder(&norm_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut *d_query_norms)
            .arg(&num_queries_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.num_queries + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&norm_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut *d_database_norms)
            .arg(&database_size_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.database_size + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    Ok(())
}

fn solve_wmma_chunked(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    d_query_norms: &mut CudaSlice<f32>,
    d_database_norms: &mut CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
    chunk_db: usize,
) -> Result<()> {    
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    if num_queries == 0 || database_size == 0 {
        return Ok(());
    }

    let nq_pad = pad16(num_queries);
    let db_pad = pad16(database_size);
    let count_q = num_queries as i32;
    let count_db = database_size as i32;
    let dims_i = DIMS as i32;
    let dims_pad_i = DIMS_PAD as i32;

    let mut d_q_half = stream.alloc_zeros::<u16>(nq_pad * DIMS_PAD)?;
    let mut d_db_half = stream.alloc_zeros::<u16>(db_pad * DIMS_PAD)?;

    let pack_norm_kernel = module.load_function("there_v9_pack_norm_fp16_250_to_256")?;
    unsafe {
        stream
            .launch_builder(&pack_norm_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_half)
            .arg(&mut *d_query_norms)
            .arg(&count_q)
            .arg(&dims_i)
            .arg(&dims_pad_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.num_queries + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&pack_norm_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_half)
            .arg(&mut *d_database_norms)
            .arg(&count_db)
            .arg(&dims_i)
            .arg(&dims_pad_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.database_size + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let num_chunks = (database_size + chunk_db - 1) / chunk_db;
    let mut d_chunk_dists = stream.alloc_zeros::<f32>(num_chunks * nq_pad)?;
    let mut d_chunk_idxs = stream.alloc_zeros::<i32>(num_chunks * nq_pad)?;

    let chunk_kernel = module.load_function("there_v9_wmma_chunk_best")?;
    let reduce_kernel = module.load_function("there_v9_reduce_chunk_bests")?;

    let nq_i = num_queries as i32;
    let db_i = database_size as i32;
    let nq_pad_i = nq_pad as i32;
    let chunk_db_i = chunk_db as i32;
    unsafe {
        stream
            .launch_builder(&chunk_kernel)
            .arg(&d_q_half)
            .arg(&d_db_half)
            .arg(&*d_query_norms)
            .arg(&*d_database_norms)
            .arg(&mut d_chunk_dists)
            .arg(&mut d_chunk_idxs)
            .arg(&nq_i)
            .arg(&db_i)
            .arg(&nq_pad_i)
            .arg(&chunk_db_i)
            .launch(LaunchConfig {
                grid_dim: (
                    (nq_pad as u32 + WMMA_QUERIES_PER_BLOCK - 1) / WMMA_QUERIES_PER_BLOCK,
                    num_chunks as u32,
                    1,
                ),
                block_dim: (32, WMMA_WARPS_PER_BLOCK, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&reduce_kernel)
            .arg(&d_chunk_dists)
            .arg(&d_chunk_idxs)
            .arg(d_best_dists)
            .arg(d_results)
            .arg(&nq_i)
            .arg(&(num_chunks as i32))
            .arg(&nq_pad_i)
            .launch(LaunchConfig {
                grid_dim: ((num_queries as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    Ok(())
}

fn solve_fp32_gemm(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    _cublas: &CudaBlas,
    d_query_norms: &CudaSlice<f32>,
    d_database_norms: &CudaSlice<f32>,
    d_results: &mut CudaSlice<i32>,
    d_best_dists: &mut CudaSlice<f32>,
) -> Result<()> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    if num_queries == 0 || database_size == 0 {
        return Ok(());
    }

    let num_queries_i = num_queries as i32;
    let reduce_kernel = module.load_function("there_v7_reduce_gemm_tile_f32")?;
    let preprocess_done = stream.record_event(None)?;
    let gemm_stream = stream.fork()?;
    gemm_stream.wait(&preprocess_done)?;
    let gemm_cublas = CudaBlas::new(gemm_stream.clone())?;
    let num_tiles = (database_size + FP32_TILE_DB - 1) / FP32_TILE_DB;

    let mut d_dot = [
        stream.alloc_zeros::<f32>(num_queries * FP32_TILE_DB)?,
        stream.alloc_zeros::<f32>(num_queries * FP32_TILE_DB)?,
    ];
    let mut gemm_done = [None, None];
    let mut reduce_done = [None, None];

    for tile_idx in 0..num_tiles {
        let buf = tile_idx & 1;

        if let Some(evt) = &reduce_done[buf] {
            gemm_stream.wait(evt)?;
        }

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
            gemm_cublas.gemm(cfg, &db_tile, &challenge.d_query_vectors, &mut d_dot[buf])?;
        }

        gemm_done[buf] = Some(gemm_stream.record_event(None)?);

        if tile_idx > 0 {
            let prev_tile = tile_idx - 1;
            let prev_buf = prev_tile & 1;
            let prev_db_start = prev_tile * FP32_TILE_DB;
            let prev_tile_len = FP32_TILE_DB.min(database_size - prev_db_start);

            stream.wait(gemm_done[prev_buf].as_ref().unwrap())?;
            launch_reduce_f32(
                stream,
                &reduce_kernel,
                d_query_norms,
                d_database_norms,
                &d_dot[prev_buf],
                d_results,
                d_best_dists,
                prev_db_start,
                prev_tile_len,
                num_queries,
                prev_tile == 0,
            )?;
            reduce_done[prev_buf] = Some(stream.record_event(None)?);
        }
    }

    let last_tile = num_tiles - 1;
    let last_buf = last_tile & 1;
    let last_db_start = last_tile * FP32_TILE_DB;
    let last_tile_len = FP32_TILE_DB.min(database_size - last_db_start);

    stream.wait(gemm_done[last_buf].as_ref().unwrap())?;
    launch_reduce_f32(
        stream,
        &reduce_kernel,
        d_query_norms,
        d_database_norms,
        &d_dot[last_buf],
        d_results,
        d_best_dists,
        last_db_start,
        last_tile_len,
        num_queries,
        last_tile == 0,
    )?;
    reduce_done[last_buf] = Some(stream.record_event(None)?);

    if let Some(evt) = &reduce_done[0] {
        gemm_stream.wait(evt)?;
    }
    if let Some(evt) = &reduce_done[1] {
        gemm_stream.wait(evt)?;
    }
    gemm_stream.synchronize()?;
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

fn reduce_config(num_queries: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, (num_queries as u32 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1),
        block_dim: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    }
}

pub fn help() {
    println!("there_v9 - GPU exhaustive vector search");
    println!("Uses padded FP16 vectors and tensor-core WMMA search kernels for the standard tracks.");
    println!("Includes track-specific routing and a cuBLAS FP32 GEMM fallback for non-standard query counts.");
}
