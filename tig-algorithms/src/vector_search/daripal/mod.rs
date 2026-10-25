use anyhow::{anyhow, Result};
use cudarc::driver::{
    result,
    safe::{CudaModule, CudaStream, LaunchConfig},
    sys, CudaSlice, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::sync::{Arc, Mutex, OnceLock};
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
const Q_TILE_ROWS: usize = 32;
const DB_PAD_ROWS: usize = 64;
const QUERIES_PER_BLOCK: usize = 256;
const CHUNK_DB: usize = 6_144;
const SCAN_SMEM: u32 = 4 * 32 * 16 * 24 * 2;
const TAU_HALF: f32 = 0.5;

fn keep_module(module: Arc<CudaModule>) {
    static KEPT: OnceLock<Mutex<Vec<Arc<CudaModule>>>> = OnceLock::new();
    KEPT.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap().push(module);
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
    prop: &cudaDeviceProp,
) -> Result<()> {
    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!("daripal expects {} dimensions, got {}", DIMS, challenge.vector_dims));
    }
    let nq = challenge.num_queries as usize;
    let db = challenge.database_size as usize;
    if nq == 0 {
        return save_solution(&Solution { indexes: Vec::new() });
    }
    if db == 0 {
        return save_solution(&Solution { indexes: vec![0usize; nq] });
    }

    let data = 4 * (nq + db) * DIMS;
    stream.context().bind_to_thread()?;
    let (free0, total) = result::mem_get_info()?;
    let others0 = (total - free0).saturating_sub(data);
    let n_sm = (prop.multiProcessorCount as u32).max(1);
    let alone_max = (96 + 4 * n_sm as usize) << 20;
    let nq_pad = round_up(nq, Q_TILE_ROWS);
    let db_pad = round_up(db, DB_PAD_ROWS);
    let nchunks = (db + CHUNK_DB - 1) / CHUNK_DB;
    let nqb = (nq_pad + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;
    let n_out = nchunks * nq_pad;
    let (nq_i, nq_pad_i, db_i, db_pad_i) = (nq as i32, nq_pad as i32, db as i32, db_pad as i32);
    let (chunk_i, nchunks_i, nqb_i) = (CHUNK_DB as i32, nchunks as i32, nqb as i32);
    let pack_blocks = (n_sm * 6).min((((nq_pad + db_pad) * 32 + 255) / 256) as u32).max(1);

    let pack = module.load_function("daripal_pack_fold")?;
    let mut d_q: CudaSlice<u16> = unsafe { stream.alloc::<u16>(nq_pad * DIMS_PAD)? };
    let mut d_db: CudaSlice<u16> = unsafe { stream.alloc::<u16>(db_pad * DIMS_PAD)? };
    let mut d_s1: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n_out)? };
    let mut d_i1: CudaSlice<i32> = unsafe { stream.alloc::<i32>(n_out)? };
    let mut d_s2: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n_out)? };
    unsafe {
        stream
            .launch_builder(&pack)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_q)
            .arg(&mut d_db)
            .arg(&nq_i)
            .arg(&nq_pad_i)
            .arg(&db_i)
            .arg(&db_pad_i)
            .launch(LaunchConfig {
                grid_dim: (pack_blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    let scan = module.load_function("daripal_scan")?;
    scan.set_attribute(
        sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SCAN_SMEM as i32,
    )?;
    unsafe {
        stream
            .launch_builder(&scan)
            .arg(&d_q)
            .arg(&d_db)
            .arg(&mut d_s1)
            .arg(&mut d_i1)
            .arg(&mut d_s2)
            .arg(&db_i)
            .arg(&nq_pad_i)
            .arg(&chunk_i)
            .arg(&nqb_i)
            .arg(&nchunks_i)
            .launch(LaunchConfig {
                grid_dim: (n_sm.min((nqb * nchunks) as u32), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: SCAN_SMEM,
            })?;
    }

    let merge = module.load_function("daripal_merge")?;
    let rescan = module.load_function("daripal_rescan")?;
    let mut d_best: CudaSlice<u64> = unsafe { stream.alloc::<u64>(nq)? };
    let mut d_rescan: CudaSlice<i32> = unsafe { stream.alloc::<i32>(2 * nq * nchunks)? };
    let mut d_count: CudaSlice<i32> = stream.alloc_zeros::<i32>(1)?;
    let tau = TAU_HALF;
    unsafe {
        stream
            .launch_builder(&merge)
            .arg(&d_s1)
            .arg(&d_i1)
            .arg(&d_s2)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_best)
            .arg(&mut d_rescan)
            .arg(&mut d_count)
            .arg(&nq_i)
            .arg(&nchunks_i)
            .arg(&nq_pad_i)
            .arg(&db_i)
            .arg(&tau)
            .launch(LaunchConfig {
                grid_dim: (((nq * 32 + 255) / 256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
        stream
            .launch_builder(&rescan)
            .arg(&d_rescan)
            .arg(&d_count)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_best)
            .arg(&chunk_i)
            .arg(&db_i)
            .launch(LaunchConfig {
                grid_dim: (n_sm * 4, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    stream.synchronize()?;
    let best: Vec<u64> = stream.memcpy_dtov(&d_best)?;
    let ours = data
        + 2 * (nq_pad + db_pad) * DIMS_PAD
        + 12 * n_out
        + 8 * nq
        + 8 * nq * nchunks;
    let (free, _) = result::mem_get_info()?;
    let others = (total - free).saturating_sub(ours);
    if others0 < alone_max && others < alone_max {
        keep_module(module.clone());
    }
    let indexes = best
        .iter()
        .map(|&k| {
            let idx = (k & 0xffff_ffff) as i64;
            if idx < 0 || idx >= db as i64 { 0usize } else { idx as usize }
        })
        .collect();
    save_solution(&Solution { indexes })
}

pub fn help() {
    println!("daripal - exact GPU 1-NN vector search, no hyperparameters.");
}
