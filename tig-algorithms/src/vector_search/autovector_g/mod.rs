use anyhow::{anyhow, Result};
use cudarc::cublas::{sys as cublas_sys, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    safe::{CudaFunction, CudaModule, CudaStream, DevicePtr, LaunchConfig},
    CudaSlice, PushKernelArg,
};
use cudarc::runtime::sys as rt_sys;
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock, Weak};
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

struct KernelTable {
    module: Weak<CudaModule>,
    compute_norms_250: CudaFunction,
    pack_norm_fp16_250_to_256: CudaFunction,
    wmma_chunk_best: CudaFunction,
    reduce_chunk_bests: CudaFunction,
    reduce_gemm_tile_f32_full_first: CudaFunction,
    reduce_gemm_tile_f32_full_next: CudaFunction,
    reduce_gemm_tile_f32_tail: CudaFunction,
}

struct Fp32ExecContext {
    owner_stream: Weak<CudaStream>,
    gemm_stream: Arc<CudaStream>,
    gemm_cublas: CudaBlas,
    dot_capacity: usize,
    d_dot0: Option<CudaSlice<f32>>,
    d_dot1: Option<CudaSlice<f32>>,
}

impl Fp32ExecContext {
    fn ensure_dot_capacity(&mut self, alloc_stream: &Arc<CudaStream>, min_len: usize) -> Result<()> {
        if self.dot_capacity >= min_len {
            return Ok(());
        }

        self.d_dot0 = Some(alloc_stream.alloc_zeros::<f32>(min_len)?);
        self.d_dot1 = Some(alloc_stream.alloc_zeros::<f32>(min_len)?);
        self.dot_capacity = min_len;
        Ok(())
    }
}

fn kernel_cache() -> &'static Mutex<HashMap<usize, Arc<KernelTable>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<KernelTable>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_kernel_table(module: &Arc<CudaModule>) -> Result<Arc<KernelTable>> {
    let key = Arc::as_ptr(module) as usize;
    let mut cache = kernel_cache().lock().unwrap();
    cache.retain(|_, entry| entry.module.upgrade().is_some());

    if let Some(entry) = cache.get(&key) {
        if let Some(cached_module) = entry.module.upgrade() {
            if Arc::ptr_eq(&cached_module, module) {
                return Ok(entry.clone());
            }
        }
    }

    let entry = Arc::new(KernelTable {
        module: Arc::downgrade(module),
        compute_norms_250: module.load_function("av_compute_norms_250")?,
        pack_norm_fp16_250_to_256: module.load_function("av_pack_norm_fp16_250_to_256")?,
        wmma_chunk_best: module.load_function("av_wmma_chunk_best")?,
        reduce_chunk_bests: module.load_function("av_reduce_chunk_bests")?,
        reduce_gemm_tile_f32_full_first: module
            .load_function("av_reduce_gemm_tile_f32_full_first")?,
        reduce_gemm_tile_f32_full_next: module
            .load_function("av_reduce_gemm_tile_f32_full_next")?,
        reduce_gemm_tile_f32_tail: module.load_function("av_reduce_gemm_tile_f32")?,
    });
    cache.insert(key, entry.clone());
    Ok(entry)
}

thread_local! {
    static FP32_EXEC_CACHE: RefCell<HashMap<usize, Rc<RefCell<Fp32ExecContext>>>> =
        RefCell::new(HashMap::new());
}

fn get_fp32_exec_context(stream: &Arc<CudaStream>) -> Result<Rc<RefCell<Fp32ExecContext>>> {
    let key = Arc::as_ptr(stream) as usize;
    FP32_EXEC_CACHE.with(|cache| -> Result<Rc<RefCell<Fp32ExecContext>>> {
        let mut cache = cache.borrow_mut();
        cache.retain(|_, entry| entry.borrow().owner_stream.upgrade().is_some());

        if let Some(entry) = cache.get(&key) {
            let reuse = entry
                .borrow()
                .owner_stream
                .upgrade()
                .map_or(false, |cached_stream| Arc::ptr_eq(&cached_stream, stream));
            if reuse {
                return Ok(entry.clone());
            }
        }

        let gemm_stream = stream.fork()?;
        let entry = Rc::new(RefCell::new(Fp32ExecContext {
            owner_stream: Arc::downgrade(stream),
            gemm_stream: gemm_stream.clone(),
            gemm_cublas: CudaBlas::new(gemm_stream)?,
            dot_capacity: 0,
            d_dot0: None,
            d_dot1: None,
        }));
        cache.insert(key, entry.clone());
        Ok(entry)
    })
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
            "autovector_g expects {} dimensions, got {}",
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
            solve_fp32_gemm(
                challenge,
                &module,
                &stream,
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
    let kernels = get_kernel_table(module)?;
    unsafe {
        stream
            .launch_builder(&kernels.compute_norms_250)
            .arg(&challenge.d_query_vectors)
            .arg(&mut *d_query_norms)
            .arg(&num_queries_i)
            .launch(LaunchConfig {
                grid_dim: ((challenge.num_queries + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&kernels.compute_norms_250)
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

/// hitRatio for the L2 persisting access-policy window on the fp16 DB (`d_db_half`).
/// 1.0 => all accesses in the window are persisting (ideal when the window <= L2 persist limit).
const L2_DB_HIT_RATIO: f32 = 1.0;

/// Pin the front region of the fp16 DB (`d_db_half`) into the L2 cache for the WMMA scan.
///
/// The `av_wmma_chunk_best` kernel re-reads each DB chunk-tile across many query-blocks
/// per chunk (and across waves on the SMs) => memory-bound on HBM re-reads of `d_db_half`.
/// Pinning the hot DB bytes in L2 (cudaAccessPropertyPersisting) frees HBM bandwidth for
/// streaming (L2 ~= 4x global BW). The window is bounded by the device
/// `MaxPersistingL2CacheSize` (DB typically >> L2 => pins a front slice).
/// Pure cache-policy hint => **bytes read are identical => Q strictly bit-invariant**.
///
/// Returns true if a persisting window was installed (so the caller resets it afterwards).
///
/// # Safety
/// Raw CUDA runtime FFI via `cudarc::runtime::sys`. `cu_stream()` (driver `CUstream`) and
/// runtime `cudaStream_t` are both `*mut CUstream_st` => cast is sound. All calls are best-effort:
/// on any error the policy is simply not applied and the kernel runs as the baseline.
#[inline(always)]
unsafe fn set_db_l2_persisting(stream: &Arc<CudaStream>, d_db_half: &CudaSlice<u16>) -> bool {
    let raw_stream = stream.cu_stream() as rt_sys::cudaStream_t;

    // Device-side max persisting L2 size (0 => feature unsupported => no-op, baseline path).
    let mut max_persist: i32 = 0;
    if rt_sys::cudaDeviceGetAttribute(
        &mut max_persist as *mut i32,
        rt_sys::cudaDeviceAttr::cudaDevAttrMaxPersistingL2CacheSize,
        0,
    ) != rt_sys::cudaError::cudaSuccess
        || max_persist <= 0
    {
        return false;
    }
    let max_persist = max_persist as usize;

    // Raise the carve-out so the persisting window is honoured.
    let _ = rt_sys::cudaDeviceSetLimit(rt_sys::cudaLimit::cudaLimitPersistingL2CacheSize, max_persist);

    let (dptr, _guard) = d_db_half.device_ptr(stream);
    let db_bytes = d_db_half.len() * std::mem::size_of::<u16>();
    let num_bytes = db_bytes.min(max_persist);
    if num_bytes == 0 {
        return false;
    }

    let window = rt_sys::cudaAccessPolicyWindow {
        base_ptr: dptr as *mut ::core::ffi::c_void,
        num_bytes,
        hitRatio: L2_DB_HIT_RATIO,
        hitProp: rt_sys::cudaAccessProperty::cudaAccessPropertyPersisting,
        missProp: rt_sys::cudaAccessProperty::cudaAccessPropertyStreaming,
    };
    let value = rt_sys::cudaLaunchAttributeValue { accessPolicyWindow: window };
    rt_sys::cudaStreamSetAttribute(
        raw_stream,
        rt_sys::cudaLaunchAttributeID::cudaLaunchAttributeAccessPolicyWindow,
        &value as *const rt_sys::cudaLaunchAttributeValue,
    ) == rt_sys::cudaError::cudaSuccess
}

/// Clear the stream's access-policy window and evict persisted lines so the next nonce
/// (the stream is reused across nonces) starts from a clean L2 state.
///
/// # Safety
/// Raw CUDA runtime FFI; best-effort (errors ignored).
#[inline(always)]
unsafe fn reset_l2_persisting(stream: &Arc<CudaStream>) {
    let raw_stream = stream.cu_stream() as rt_sys::cudaStream_t;
    let window = rt_sys::cudaAccessPolicyWindow {
        base_ptr: std::ptr::null_mut(),
        num_bytes: 0,
        hitRatio: 0.0,
        hitProp: rt_sys::cudaAccessProperty::cudaAccessPropertyNormal,
        missProp: rt_sys::cudaAccessProperty::cudaAccessPropertyNormal,
    };
    let value = rt_sys::cudaLaunchAttributeValue { accessPolicyWindow: window };
    let _ = rt_sys::cudaStreamSetAttribute(
        raw_stream,
        rt_sys::cudaLaunchAttributeID::cudaLaunchAttributeAccessPolicyWindow,
        &value as *const rt_sys::cudaLaunchAttributeValue,
    );
    let _ = rt_sys::cudaCtxResetPersistingL2Cache();
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
    let kernels = get_kernel_table(module)?;

    let mut d_q_half = stream.alloc_zeros::<u16>(nq_pad * DIMS_PAD)?;
    let mut d_db_half = stream.alloc_zeros::<u16>(db_pad * DIMS_PAD)?;

    unsafe {
        stream
            .launch_builder(&kernels.pack_norm_fp16_250_to_256)
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
            .launch_builder(&kernels.pack_norm_fp16_250_to_256)
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

    let nq_i = num_queries as i32;
    let db_i = database_size as i32;
    let nq_pad_i = nq_pad as i32;
    let chunk_db_i = chunk_db as i32;
    unsafe {
        // Pin the hot fp16 DB front-region in L2 for the chunked WMMA scan.
        // Pure cache-policy hint => bytes read are identical => Q strictly bit-invariant.
        let l2_pinned = set_db_l2_persisting(stream, &d_db_half);

        stream
            .launch_builder(&kernels.wmma_chunk_best)
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
            .launch_builder(&kernels.reduce_chunk_bests)
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

        // Clear the persisting window + evict persisted lines so the next nonce (reused
        // stream) starts from a clean L2 state.
        if l2_pinned {
            reset_l2_persisting(stream);
        }
    }

    Ok(())
}

fn solve_fp32_gemm(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
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
    let kernels = get_kernel_table(module)?;
    let fp32_ctx = get_fp32_exec_context(stream)?;
    let mut fp32_ctx = fp32_ctx.borrow_mut();
    fp32_ctx.ensure_dot_capacity(stream, num_queries * FP32_TILE_DB)?;

    let preprocess_done = stream.record_event(None)?;
    fp32_ctx.gemm_stream.wait(&preprocess_done)?;
    let num_tiles = (database_size + FP32_TILE_DB - 1) / FP32_TILE_DB;

    let mut gemm_done = [None, None];
    let mut reduce_done = [None, None];

    for tile_idx in 0..num_tiles {
        let buf = tile_idx & 1;

        if let Some(evt) = &reduce_done[buf] {
            fp32_ctx.gemm_stream.wait(evt)?;
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
            let mut d_dot = if buf == 0 {
                fp32_ctx.d_dot0.take().unwrap()
            } else {
                fp32_ctx.d_dot1.take().unwrap()
            };
            fp32_ctx
                .gemm_cublas
                .gemm(cfg, &db_tile, &challenge.d_query_vectors, &mut d_dot)?;
            if buf == 0 {
                fp32_ctx.d_dot0 = Some(d_dot);
            } else {
                fp32_ctx.d_dot1 = Some(d_dot);
            }
        }

        gemm_done[buf] = Some(fp32_ctx.gemm_stream.record_event(None)?);

        if tile_idx > 0 {
            let prev_tile = tile_idx - 1;
            let prev_buf = prev_tile & 1;
            let prev_db_start = prev_tile * FP32_TILE_DB;
            let prev_tile_len = FP32_TILE_DB.min(database_size - prev_db_start);

            stream.wait(gemm_done[prev_buf].as_ref().unwrap())?;
            let d_dot = if prev_buf == 0 {
                fp32_ctx.d_dot0.as_ref().unwrap()
            } else {
                fp32_ctx.d_dot1.as_ref().unwrap()
            };
            launch_reduce_f32(
                stream,
                &kernels.reduce_gemm_tile_f32_full_first,
                &kernels.reduce_gemm_tile_f32_full_next,
                &kernels.reduce_gemm_tile_f32_tail,
                d_query_norms,
                d_database_norms,
                d_dot,
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
    let d_dot = if last_buf == 0 {
        fp32_ctx.d_dot0.as_ref().unwrap()
    } else {
        fp32_ctx.d_dot1.as_ref().unwrap()
    };
    launch_reduce_f32(
        stream,
        &kernels.reduce_gemm_tile_f32_full_first,
        &kernels.reduce_gemm_tile_f32_full_next,
        &kernels.reduce_gemm_tile_f32_tail,
        d_query_norms,
        d_database_norms,
        d_dot,
        d_results,
        d_best_dists,
        last_db_start,
        last_tile_len,
        num_queries,
        last_tile == 0,
    )?;
    reduce_done[last_buf] = Some(stream.record_event(None)?);

    if let Some(evt) = &reduce_done[0] {
        fp32_ctx.gemm_stream.wait(evt)?;
    }
    if let Some(evt) = &reduce_done[1] {
        fp32_ctx.gemm_stream.wait(evt)?;
    }
    fp32_ctx.gemm_stream.synchronize()?;
    Ok(())
}

fn launch_reduce_f32(
    stream: &Arc<CudaStream>,
    reduce_full_first_kernel: &cudarc::driver::safe::CudaFunction,
    reduce_full_next_kernel: &cudarc::driver::safe::CudaFunction,
    reduce_tail_kernel: &cudarc::driver::safe::CudaFunction,
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
    let num_queries_i = num_queries as i32;
    unsafe {
        if tile_len == FP32_TILE_DB {
            let kernel = if first_tile {
                reduce_full_first_kernel
            } else {
                reduce_full_next_kernel
            };
            stream
                .launch_builder(kernel)
                .arg(d_query_norms)
                .arg(d_database_norms)
                .arg(d_dot)
                .arg(d_results)
                .arg(d_best_dists)
                .arg(&db_start_i)
                .arg(&num_queries_i)
                .launch(reduce_config(num_queries))?;
        } else {
            let tile_len_i = tile_len as i32;
            let first_tile_i = if first_tile { 1i32 } else { 0i32 };
            stream
                .launch_builder(reduce_tail_kernel)
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
    println!("autovector_g - GPU exhaustive vector search");
    println!("Standard tracks use padded FP16 vectors with track-specialized tensor-core WMMA search kernels.");
    println!("Includes fused/regular preprocessing paths, per-track routing, and a cuBLAS FP32 fallback for non-standard query counts.");
}
