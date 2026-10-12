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
const DB_TILE_ROWS: usize = 16;
const DB_STEP: usize = 32;
const WARPS_PER_BLOCK: u32 = 8;
const QUERIES_PER_BLOCK: usize = Q_TILE_ROWS * WARPS_PER_BLOCK as usize;
const PACK_THREADS: u32 = 256;
const PACK_BLOCKS: u32 = 256;
const MERGE_THREADS: u32 = 256;

const CHUNK_DB_7K: usize = 8_192;
const CHUNK_DB_9K: usize = 8_192;
const CHUNK_DB_11K: usize = 8_192;
const CHUNK_DB_13K: usize = 8_192;
const CHUNK_DB_15K: usize = 8_192;
const CHUNK_DB_DEFAULT: usize = 8_192;

const TAU_HALF: f32 = 0.5;

struct KernelTable {
    module: Weak<CudaModule>,
    pack_fold: CudaFunction,
    chunk_top2: CudaFunction,
    merge_rerank: CudaFunction,
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
        pack_fold: module.load_function("there_v11_pack_fold")?,
        chunk_top2: module.load_function("there_v11_chunk_top2")?,
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
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!(
            "there_v11 expects {} dimensions, got {}",
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

    let chunk_db = match num_queries {
        7_000 => CHUNK_DB_7K,
        9_000 => CHUNK_DB_9K,
        11_000 => CHUNK_DB_11K,
        13_000 => CHUNK_DB_13K,
        15_000 => CHUNK_DB_15K,
        _ => CHUNK_DB_DEFAULT,
    };
    debug_assert!(chunk_db % DB_STEP == 0);

    let kernels = get_kernel_table(&module)?;

    let nq_pad = round_up(num_queries, Q_TILE_ROWS);
    let db_pad = round_up(database_size, DB_STEP);
    let num_chunks = (database_size + chunk_db - 1) / chunk_db;

    let mut d_q_half: CudaSlice<u16> = unsafe { stream.alloc::<u16>(nq_pad * DIMS_PAD)? };
    let mut d_db_half: CudaSlice<u16> = unsafe { stream.alloc::<u16>(db_pad * DIMS_PAD)? };
    let mut d_s1: CudaSlice<f32> = unsafe { stream.alloc::<f32>(num_chunks * nq_pad)? };
    let mut d_i1: CudaSlice<i32> = unsafe { stream.alloc::<i32>(num_chunks * nq_pad)? };
    let mut d_s2: CudaSlice<f32> = unsafe { stream.alloc::<f32>(num_chunks * nq_pad)? };
    let mut d_i2: CudaSlice<i32> = unsafe { stream.alloc::<i32>(num_chunks * nq_pad)? };
    let mut d_results: CudaSlice<i32> = unsafe { stream.alloc::<i32>(num_queries)? };

    let nq_i = num_queries as i32;
    let nq_pad_i = nq_pad as i32;
    let db_i = database_size as i32;
    let db_pad_i = db_pad as i32;
    let chunk_db_i = chunk_db as i32;
    let num_chunks_i = num_chunks as i32;

    unsafe {
        let pack_rows = (nq_pad + db_pad) as u32;
        let pack_blocks = PACK_BLOCKS.min((pack_rows * 32 + PACK_THREADS - 1) / PACK_THREADS).max(1);
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

        stream
            .launch_builder(&kernels.chunk_top2)
            .arg(&d_q_half)
            .arg(&d_db_half)
            .arg(&mut d_s1)
            .arg(&mut d_i1)
            .arg(&mut d_s2)
            .arg(&mut d_i2)
            .arg(&db_i)
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

        let tau = TAU_HALF;
        stream
            .launch_builder(&kernels.merge_rerank)
            .arg(&d_s1)
            .arg(&d_i1)
            .arg(&d_s2)
            .arg(&d_i2)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_results)
            .arg(&nq_i)
            .arg(&num_chunks_i)
            .arg(&nq_pad_i)
            .arg(&chunk_db_i)
            .arg(&db_i)
            .arg(&tau)
            .launch(LaunchConfig {
                grid_dim: ((num_queries as u32 * 32 + MERGE_THREADS - 1) / MERGE_THREADS, 1, 1),
                block_dim: (MERGE_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    stream.synchronize()?;
    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes = result_indices
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
    println!("there_v11 - exact GPU 1-NN vector search");
    println!("fp16 tensor-core (m32n8k16, fp16 accumulate) screening with folded norms, top-2 per chunk,");
    println!("followed by an exact fp32 rerank of every candidate inside a rigorous error bound.");
}
