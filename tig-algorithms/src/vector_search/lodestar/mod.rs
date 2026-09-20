// lodestar - GPU exact nearest-neighbour search.
// WMMA fp16 chunked distance scan; tuning read in Hparams::from_map; defaults ship.

use anyhow::{anyhow, Result};
use cudarc::driver::{
    safe::{CudaFunction, CudaModule, CudaStream, LaunchConfig},
    CudaSlice, PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;

#[derive(Clone, Copy, Debug)]
struct Hparams {
    chunk_db: usize,
    wmma_warps: u32,
    k1_delta: f32,
    geom: u32,
}

impl Hparams {
    fn defaults(_num_queries: usize) -> Self {
        Hparams {
            chunk_db: 8_192,
            wmma_warps: 4,
            k1_delta: 2.0,
            geom: 32,
        }
    }

    fn from_map(num_queries: usize, hp: &Option<Map<String, Value>>) -> Self {
        let mut p = Hparams::defaults(num_queries);
        let m = match hp {
            Some(m) => m,
            None => return p,
        };
        if let Some(v) = m.get("chunk_db").and_then(|v| v.as_u64()) {
            let v = (v as usize) & !15usize;
            if v >= 16 {
                p.chunk_db = v;
            }
        }
        if let Some(v) = m.get("wmma_warps").and_then(|v| v.as_u64()) {
            if v == 2 || v == 4 || v == 8 {
                p.wmma_warps = v as u32;
            }
        }
        if let Some(v) = m.get("k1_delta").and_then(|v| v.as_f64()) {
            if v.is_finite() && v >= 0.0 {
                p.k1_delta = v as f32;
            }
        }
        if let Some(v) = m.get("geom").and_then(|v| v.as_u64()) {
            if v == 16 || v == 32 {
                p.geom = v as u32;
            }
        }
        p
    }
}

fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

fn pad32(n: usize) -> usize {
    (n + 31) & !31
}

struct KernelTable {
    module: Weak<CudaModule>,
    pack_norm_fast: CudaFunction,
    pack_norm_fast_q32: CudaFunction,
    top2_f16acc_w4: CudaFunction,
    top2_g328: CudaFunction,
    reduce_filter_rerank: CudaFunction,
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
        pack_norm_fast: module.load_function("lds_pack_norm_fast")?,
        pack_norm_fast_q32: module.load_function("lds_pack_norm_fast_q32")?,
        top2_f16acc_w4: module.load_function("lds_top2_f16acc")?,
        top2_g328: module.load_function("lds_top2_g328")?,
        reduce_filter_rerank: module.load_function("lds_reduce_filter_rerank")?,
    });
    cache.insert(key, entry.clone());
    Ok(entry)
}

#[derive(Default)]
struct Bufs {
    q_half: Option<CudaSlice<u16>>,
    db_half: Option<CudaSlice<u16>>,
    q_norms: Option<CudaSlice<f32>>,
    db_norms: Option<CudaSlice<f32>>,
    chunk_dists: Option<CudaSlice<f32>>,
    chunk_idxs: Option<CudaSlice<i32>>,
    chunk_dists2: Option<CudaSlice<f32>>,
    chunk_idxs2: Option<CudaSlice<i32>>,
    results: Option<CudaSlice<i32>>,
}

thread_local! {
    static BUFS: RefCell<Bufs> = RefCell::new(Bufs::default());
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    if challenge.vector_dims as usize != DIMS {
        return Err(anyhow!(
            "the base solver expects {} dimensions, got {}",
            DIMS,
            challenge.vector_dims
        ));
    }

    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;
    if num_queries == 0 || database_size == 0 {
        save_solution(&Solution {
            indexes: vec![0usize; num_queries],
        })?;
        return Ok(());
    }

    let hp = Hparams::from_map(num_queries, hyperparameters);
    let indexes = solve_wmma_chunked(challenge, &module, &stream, hp)?;
    save_solution(&Solution { indexes })?;
    Ok(())
}

fn solve_wmma_chunked(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    hp: Hparams,
) -> Result<Vec<usize>> {
    let num_queries = challenge.num_queries as usize;
    let database_size = challenge.database_size as usize;

    let g328 = hp.geom == 32;
    let nq_pad = if g328 { pad32(num_queries) } else { pad16(num_queries) };
    let db_pad = pad16(database_size);
    let chunk_db = hp.chunk_db.min(pad16(database_size)).max(16);
    let num_chunks = (database_size + chunk_db - 1) / chunk_db;

    let count_q = num_queries as i32;
    let count_db = database_size as i32;
    let dims_i = DIMS as i32;
    let dims_pad_i = DIMS_PAD as i32;
    let nq_pad_i = nq_pad as i32;
    let chunk_db_i = chunk_db as i32;
    let tail_mode_q: i32 = 0;
    let tail_mode_db: i32 = 1;
    let num_chunks_i = num_chunks as i32;
    let kernels = get_kernel_table(module)?;

    let chunk_need = num_chunks * nq_pad;
    let (c_q_half, c_db_half, c_q_norms, c_db_norms, c_cd, c_ci, c_cd2, c_ci2, c_res) = BUFS.with(|c| {
        let mut b = c.borrow_mut();
        (
            b.q_half.take(),
            b.db_half.take(),
            b.q_norms.take(),
            b.db_norms.take(),
            b.chunk_dists.take(),
            b.chunk_idxs.take(),
            b.chunk_dists2.take(),
            b.chunk_idxs2.take(),
            b.results.take(),
        )
    });

    let mut d_q_half = match c_q_half {
        Some(b) if b.len() >= nq_pad * DIMS_PAD => b,
        _ => stream.alloc_zeros::<u16>(nq_pad * DIMS_PAD)?,
    };
    let mut d_db_half = match c_db_half {
        Some(b) if b.len() >= db_pad * DIMS_PAD => b,
        _ => stream.alloc_zeros::<u16>(db_pad * DIMS_PAD)?,
    };
    let mut d_query_norms = match c_q_norms {
        Some(b) if b.len() >= num_queries => b,
        _ => unsafe { stream.alloc::<f32>(num_queries)? },
    };
    let mut d_database_norms = match c_db_norms {
        Some(b) if b.len() >= database_size => b,
        _ => unsafe { stream.alloc::<f32>(database_size)? },
    };
    let mut d_chunk_dists = match c_cd {
        Some(b) if b.len() >= chunk_need => b,
        _ => unsafe { stream.alloc::<f32>(chunk_need)? },
    };
    let mut d_chunk_idxs = match c_ci {
        Some(b) if b.len() >= chunk_need => b,
        _ => unsafe { stream.alloc::<i32>(chunk_need)? },
    };
    let mut d_chunk_dists2 = match c_cd2 {
        Some(b) if b.len() >= chunk_need => b,
        _ => unsafe { stream.alloc::<f32>(chunk_need)? },
    };
    let mut d_chunk_idxs2 = match c_ci2 {
        Some(b) if b.len() >= chunk_need => b,
        _ => unsafe { stream.alloc::<i32>(chunk_need)? },
    };
    let mut d_results = match c_res {
        Some(b) if b.len() >= num_queries => b,
        _ => unsafe { stream.alloc::<i32>(num_queries)? },
    };

    const PACK_FAST_ROWS: u32 = 128;
    let pblock = PACK_FAST_ROWS;
    let q_grid = (challenge.num_queries + PACK_FAST_ROWS - 1) / PACK_FAST_ROWS;
    let db_grid = (challenge.database_size + PACK_FAST_ROWS - 1) / PACK_FAST_ROWS;
    let pack_kernel = &kernels.pack_norm_fast;
    let q_pack_kernel = if g328 {
        &kernels.pack_norm_fast_q32
    } else {
        pack_kernel
    };
    unsafe {
        stream
            .launch_builder(q_pack_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_half)
            .arg(&mut d_query_norms)
            .arg(&count_q)
            .arg(&dims_i)
            .arg(&dims_pad_i)
            .arg(&tail_mode_q)
            .launch(LaunchConfig {
                grid_dim: (q_grid, 1, 1),
                block_dim: (pblock, 1, 1),
                shared_mem_bytes: 0,
            })?;

        stream
            .launch_builder(pack_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_half)
            .arg(&mut d_database_norms)
            .arg(&count_db)
            .arg(&dims_i)
            .arg(&dims_pad_i)
            .arg(&tail_mode_db)
            .launch(LaunchConfig {
                grid_dim: (db_grid, 1, 1),
                block_dim: (pblock, 1, 1),
                shared_mem_bytes: 0,
            })?;

        let warps: u32 = if g328 { hp.wmma_warps } else { 4 };
        let rows_per_warp: u32 = if g328 { 32 } else { 16 };
        let queries_per_block: u32 = rows_per_warp * warps;
        let scan_cfg = LaunchConfig {
            grid_dim: (
                (nq_pad as u32 + queries_per_block - 1) / queries_per_block,
                num_chunks as u32,
                1,
            ),
            block_dim: (32, warps, 1),
            shared_mem_bytes: 0,
        };
        let reduce_cfg = LaunchConfig {
            grid_dim: ((challenge.num_queries + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        {
            let scan_kernel = if g328 {
                &kernels.top2_g328
            } else {
                &kernels.top2_f16acc_w4
            };
            stream
                .launch_builder(scan_kernel)
                .arg(&d_q_half)
                .arg(&d_db_half)
                .arg(&d_query_norms)
                .arg(&d_database_norms)
                .arg(&mut d_chunk_dists)
                .arg(&mut d_chunk_idxs)
                .arg(&mut d_chunk_dists2)
                .arg(&mut d_chunk_idxs2)
                .arg(&count_q)
                .arg(&count_db)
                .arg(&nq_pad_i)
                .arg(&chunk_db_i)
                .launch(scan_cfg)?;

            let dims_rr = DIMS as i32;
            let delta = hp.k1_delta * 0.5f32;
            stream
                .launch_builder(&kernels.reduce_filter_rerank)
                .arg(&d_chunk_dists)
                .arg(&d_chunk_idxs)
                .arg(&d_chunk_dists2)
                .arg(&d_chunk_idxs2)
                .arg(&challenge.d_query_vectors)
                .arg(&challenge.d_database_vectors)
                .arg(&count_q)
                .arg(&num_chunks_i)
                .arg(&nq_pad_i)
                .arg(&dims_rr)
                .arg(&count_db)
                .arg(&delta)
                .arg(&mut d_results)
                .launch(reduce_cfg)?;
        }
    }

    stream.synchronize()?;

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results.slice(0..num_queries))?;

    BUFS.with(|c| {
        let mut b = c.borrow_mut();
        b.q_half = Some(d_q_half);
        b.db_half = Some(d_db_half);
        b.q_norms = Some(d_query_norms);
        b.db_norms = Some(d_database_norms);
        b.chunk_dists = Some(d_chunk_dists);
        b.chunk_idxs = Some(d_chunk_idxs);
        b.chunk_dists2 = Some(d_chunk_dists2);
        b.chunk_idxs2 = Some(d_chunk_idxs2);
        b.results = Some(d_results);
    });

    Ok(result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= database_size as i32 {
                0usize
            } else {
                idx as usize
            }
        })
        .collect())
}

pub fn help() {
    println!("lodestar - GPU exact nearest-neighbour search (WMMA fp16 chunked scan).");
    println!("Hyperparameters (read in Hparams::from_map): chunk_db (multiple of 16, default 8192), \
wmma_warps (2|4|8, default 4), geom (16|32, default 32), k1_delta (>=0, default 2.0).");
}
