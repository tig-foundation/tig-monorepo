use anyhow::{anyhow, Result};
use cudarc::driver::{
    safe::{CudaFunction, CudaModule, CudaStream, LaunchConfig},
    CudaSlice, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;

const Q_TILE_ROWS: usize = 32;
const DB_STEP: usize = 16;
const WARPS_PER_BLOCK: u32 = 8;
const QUERIES_PER_BLOCK: usize = Q_TILE_ROWS * WARPS_PER_BLOCK as usize;
const PACK_THREADS: u32 = 256;
const PACK_BLOCKS: u32 = 256;
const MERGE_THREADS: u32 = 256;

const CHUNK_DB_DEFAULT: usize = 8_192;

const TAU_HALF: f32 = 0.5;

struct KernelTable {
    module: Weak<CudaModule>,
    pack_fold: CudaFunction,
    chunk_top2: CudaFunction,
    merge_rerank: CudaFunction,
}

fn kernel_cache() -> &'static Mutex<HashMap<(usize, bool), Arc<KernelTable>>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, bool), Arc<KernelTable>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_kernel_table(module: &Arc<CudaModule>, use_fast_scan: bool) -> Result<Arc<KernelTable>> {
    let key = (Arc::as_ptr(module) as usize, use_fast_scan);
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
        pack_fold: module.load_function("there_v11_pack_fold")?,
        chunk_top2: module.load_function(if use_fast_scan {
            "there_v11_chunk_top2"
        } else {
            "there_v11_chunk_top2_baseline"
        })?,
        merge_rerank: module.load_function("there_v11_merge_rerank")?,
    });
    cache.insert(key, entry.clone());
    Ok(entry)
}

fn round_up(n: usize, m: usize) -> usize {
    (n + m - 1) / m * m
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!(
            "tuned_v11 expects {} dimensions, got {}",
            DIMS,
            challenge.vector_dims
        ));
    }

    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    if num_queries == 0 {
        save_solution(&Solution { indexes: Vec::new() })?;
        return Ok(());
    }
    if database_size == 0 {
        save_solution(&Solution {
            indexes: vec![0usize; num_queries],
        })?;
        return Ok(());
    }

    let chunk_db = hyperparameters.as_ref()
        .and_then(|m| m.get("chunk_db"))
        .and_then(Value::as_u64)
        .filter(|&v| (512..=32_768).contains(&v) && v % DB_STEP as u64 == 0)
        .map(|v| v as usize)
        .unwrap_or(CHUNK_DB_DEFAULT);
    let pack_blocks = hyperparameters.as_ref()
        .and_then(|m| m.get("pack_blocks"))
        .and_then(Value::as_u64)
        .filter(|&v| (32..=1_024).contains(&v))
        .map(|v| v as u32)
        .unwrap_or(PACK_BLOCKS);
    let tau_half = hyperparameters.as_ref()
        .and_then(|m| m.get("tau_half"))
        .and_then(Value::as_f64)
        .filter(|&v| v.is_finite() && (0.0..=0.5).contains(&v))
        .map(|v| v as f32)
        .unwrap_or(TAU_HALF);
    debug_assert!(chunk_db % DB_STEP == 0);

    // The 16-row scan is measured on Turing (sm_75). Elsewhere, run the
    // original 32-row scan until the smaller tile is benchmarked there.
    let use_fast_scan = prop.major == 7 && prop.minor == 5;
    let kernels = get_kernel_table(&module, use_fast_scan)?;

    let nq_pad = round_up(num_queries, Q_TILE_ROWS);
    let db_pad = round_up(database_size, DB_STEP);
    let num_chunks = (database_size + chunk_db - 1) / chunk_db;

    let mut d_q_half: CudaSlice<u16> = unsafe { stream.alloc::<u16>(nq_pad * DIMS_PAD)? };
    let mut d_db_half: CudaSlice<u16> = unsafe { stream.alloc::<u16>(db_pad * DIMS_PAD)? };
    let mut d_s1: CudaSlice<f32> = unsafe { stream.alloc::<f32>(num_chunks * nq_pad)? };
    let mut d_i1: CudaSlice<i32> = unsafe { stream.alloc::<i32>(num_chunks * nq_pad)? };
    let mut d_s2: CudaSlice<f32> = unsafe { stream.alloc::<f32>(num_chunks * nq_pad)? };
    let mut d_i2: Option<CudaSlice<i32>> = if use_fast_scan {
        None
    } else {
        Some(unsafe { stream.alloc::<i32>(num_chunks * nq_pad)? })
    };
    let mut d_results: CudaSlice<i32> = unsafe { stream.alloc::<i32>(num_queries)? };

    let nq_i = num_queries as i32;
    let nq_pad_i = nq_pad as i32;
    let db_i = database_size as i32;
    let db_pad_i = db_pad as i32;
    let chunk_db_i = chunk_db as i32;
    let num_chunks_i = num_chunks as i32;

    unsafe {
        let pack_rows = (nq_pad + db_pad) as u32;
        let pack_blocks = pack_blocks.min((pack_rows * 32 + PACK_THREADS - 1) / PACK_THREADS).max(1);
        stream
            .launch_builder(&kernels.pack_fold)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_q_half)
            .arg(&mut d_db_half)
            .arg(&nq_i)
            .arg(&nq_pad_i)
            .arg(&db_i)
            .arg(&db_pad_i)
            .launch(LaunchConfig {
                grid_dim: (pack_blocks, 1, 1),
                block_dim: (PACK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;

        let mut scan = stream.launch_builder(&kernels.chunk_top2);
        scan.arg(&d_q_half)
            .arg(&d_db_half)
            .arg(&mut d_s1)
            .arg(&mut d_i1)
            .arg(&mut d_s2);
        if let Some(ref mut second_indexes) = d_i2 {
            scan.arg(second_indexes);
        }
        scan.arg(&db_i)
            .arg(&nq_pad_i)
            .arg(&chunk_db_i)
            .launch(LaunchConfig {
                grid_dim: (
                    ((nq_pad + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK) as u32,
                    num_chunks as u32,
                    1,
                ),
                block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(&kernels.merge_rerank)
            .arg(&d_s1)
            .arg(&d_i1)
            .arg(&d_s2)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_results)
            .arg(&nq_i)
            .arg(&num_chunks_i)
            .arg(&nq_pad_i)
            .arg(&chunk_db_i)
            .arg(&db_i)
            .arg(&tau_half)
            .launch(LaunchConfig {
                grid_dim: ((num_queries as u32 * 32 + MERGE_THREADS - 1) / MERGE_THREADS, 1, 1),
                block_dim: (MERGE_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    stream.synchronize()?;
    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= db_i {
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
    println!("tuned_v11 - GPU 1-NN vector search, based on there_v11");
    println!("16-row scan on sm_75; original 32-row scan on other CUDA architectures.");
    println!("Hyperparameters: chunk_db (multiple of 16, 512..32768), pack_blocks (32..1024), tau_half (0..0.5).");
    println!("FP16 tensor-core screening with folded norms and top-2 scores per chunk,");
    println!("followed by FP32 reranking. Keep tau_half=0.5 for the measured quality parity.");
}
