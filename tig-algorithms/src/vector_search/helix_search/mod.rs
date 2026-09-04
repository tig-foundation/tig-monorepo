// helix_search — GPU exact nearest-neighbour search.
// Tensor-Core (WMMA) fp16 distance scan over cache-resident database chunks,
// producing an fp16 candidate shortlist that is re-ranked exactly in fp32
// (coarse-to-fine). All tuning is exposed as hyperparameters read in from_map;
// the defaults are the shipped operating point.

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
    pack_block: u32,
    pack_impl: u32,
    k1_arm: u32,
    k1_delta: f32,
}

impl Hparams {
    fn defaults(_num_queries: usize) -> Self {
        Hparams {
            chunk_db: 8_192,
            wmma_warps: 4,
            pack_block: 256,
            pack_impl: 1,
            k1_arm: 1,
            k1_delta: 2.0,
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
        if let Some(v) = m.get("pack_block").and_then(|v| v.as_u64()) {
            if v >= 32 && v <= 1024 && v % 32 == 0 {
                p.pack_block = v as u32;
            }
        }
        if let Some(v) = m.get("pack_impl").and_then(|v| v.as_u64()) {
            if v <= 1 {
                p.pack_impl = v as u32;
            }
        }
        if let Some(v) = m.get("k1_arm").and_then(|v| v.as_u64()) {
            if v <= 1 {
                p.k1_arm = v as u32;
            }
        }
        if let Some(v) = m.get("k1_delta").and_then(|v| v.as_f64()) {
            if v.is_finite() && v >= 0.0 {
                p.k1_delta = v as f32;
            }
        }
        p
    }
}

fn pad16(n: usize) -> usize {
    (n + 15) & !15
}

struct KernelTable {
    module: Weak<CudaModule>,
    pack_norm_fp16_250_to_256: CudaFunction,
    pack_norm_fast: CudaFunction,
    wmma_chunk_best_w2: CudaFunction,
    wmma_chunk_best_w4: CudaFunction,
    wmma_chunk_best_w8: CudaFunction,
    reduce_chunk_bests: CudaFunction,
    top2_f16acc_w2: CudaFunction,
    top2_f16acc_w4: CudaFunction,
    top2_f16acc_w8: CudaFunction,
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
        pack_norm_fp16_250_to_256: module.load_function("hlx_pack_norm_fp16_250_to_256")?,
        pack_norm_fast: module.load_function("hlx_pack_norm_fast")?,
        wmma_chunk_best_w2: module.load_function("hlx_wmma_chunk_best_w2")?,
        wmma_chunk_best_w4: module.load_function("hlx_wmma_chunk_best")?,
        wmma_chunk_best_w8: module.load_function("hlx_wmma_chunk_best_w8")?,
        reduce_chunk_bests: module.load_function("hlx_reduce_chunk_bests")?,
        top2_f16acc_w2: module.load_function("hlx_top2_f16acc_w2")?,
        top2_f16acc_w4: module.load_function("hlx_top2_f16acc")?,
        top2_f16acc_w8: module.load_function("hlx_top2_f16acc_w8")?,
        reduce_filter_rerank: module.load_function("hlx_reduce_filter_rerank")?,
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
    best_dists: Option<CudaSlice<f32>>,
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

    let nq_pad = pad16(num_queries);
    let db_pad = pad16(database_size);
    let chunk_db = hp.chunk_db.min(pad16(database_size)).max(16);
    let num_chunks = (database_size + chunk_db - 1) / chunk_db;

    let count_q = num_queries as i32;
    let count_db = database_size as i32;
    let dims_i = DIMS as i32;
    let dims_pad_i = DIMS_PAD as i32;
    let nq_pad_i = nq_pad as i32;
    let chunk_db_i = chunk_db as i32;
    let num_chunks_i = num_chunks as i32;
    let kernels = get_kernel_table(module)?;

    let chunk_need = num_chunks * nq_pad;
    let (c_q_half, c_db_half, c_q_norms, c_db_norms, c_cd, c_ci, c_cd2, c_ci2, c_bd, c_res) = BUFS.with(|c| {
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
            b.best_dists.take(),
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
    let need2 = if hp.k1_arm == 1 { chunk_need } else { 1 };
    let mut d_chunk_dists2 = match c_cd2 {
        Some(b) if b.len() >= need2 => b,
        _ => unsafe { stream.alloc::<f32>(need2)? },
    };
    let mut d_chunk_idxs2 = match c_ci2 {
        Some(b) if b.len() >= need2 => b,
        _ => unsafe { stream.alloc::<i32>(need2)? },
    };
    let mut d_best_dists = match c_bd {
        Some(b) if b.len() >= num_queries => b,
        _ => unsafe { stream.alloc::<f32>(num_queries)? },
    };
    let mut d_results = match c_res {
        Some(b) if b.len() >= num_queries => b,
        _ => unsafe { stream.alloc::<i32>(num_queries)? },
    };

    const PACK_FAST_ROWS: u32 = 128;
    let pack_block = hp.pack_block;
    let (pack_kernel, q_grid, db_grid, pblock) = if hp.pack_impl == 1 {
        (
            &kernels.pack_norm_fast,
            (challenge.num_queries + PACK_FAST_ROWS - 1) / PACK_FAST_ROWS,
            (challenge.database_size + PACK_FAST_ROWS - 1) / PACK_FAST_ROWS,
            PACK_FAST_ROWS,
        )
    } else {
        (
            &kernels.pack_norm_fp16_250_to_256,
            (challenge.num_queries + pack_block - 1) / pack_block,
            (challenge.database_size + pack_block - 1) / pack_block,
            pack_block,
        )
    };
    unsafe {
        stream
            .launch_builder(pack_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_half)
            .arg(&mut d_query_norms)
            .arg(&count_q)
            .arg(&dims_i)
            .arg(&dims_pad_i)
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
            .launch(LaunchConfig {
                grid_dim: (db_grid, 1, 1),
                block_dim: (pblock, 1, 1),
                shared_mem_bytes: 0,
            })?;

        let queries_per_block: u32 = match hp.wmma_warps { 2 => 32, 8 => 128, _ => 64 };
        let scan_cfg = LaunchConfig {
            grid_dim: (
                (nq_pad as u32 + queries_per_block - 1) / queries_per_block,
                num_chunks as u32,
                1,
            ),
            block_dim: (32, hp.wmma_warps, 1),
            shared_mem_bytes: 0,
        };
        let reduce_cfg = LaunchConfig {
            grid_dim: ((challenge.num_queries + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        if hp.k1_arm == 1 {
            let scan_kernel = match hp.wmma_warps {
                2 => &kernels.top2_f16acc_w2,
                8 => &kernels.top2_f16acc_w8,
                _ => &kernels.top2_f16acc_w4,
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
            let delta = hp.k1_delta;
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
        } else {
            let scan_kernel = match hp.wmma_warps {
                2 => &kernels.wmma_chunk_best_w2,
                8 => &kernels.wmma_chunk_best_w8,
                _ => &kernels.wmma_chunk_best_w4,
            };
            stream
                .launch_builder(scan_kernel)
                .arg(&d_q_half)
                .arg(&d_db_half)
                .arg(&d_query_norms)
                .arg(&d_database_norms)
                .arg(&mut d_chunk_dists)
                .arg(&mut d_chunk_idxs)
                .arg(&count_q)
                .arg(&count_db)
                .arg(&nq_pad_i)
                .arg(&chunk_db_i)
                .launch(scan_cfg)?;

            stream
                .launch_builder(&kernels.reduce_chunk_bests)
                .arg(&d_chunk_dists)
                .arg(&d_chunk_idxs)
                .arg(&mut d_best_dists)
                .arg(&mut d_results)
                .arg(&count_q)
                .arg(&num_chunks_i)
                .arg(&nq_pad_i)
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
        b.best_dists = Some(d_best_dists);
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
    println!("helix_search - GPU nearest-neighbour search (WMMA fp16 chunked scan)");
    println!("Hyperparametres (--hyperparameters), tous LUS dans Hparams::from_map :");
    println!("  chunk_db   : entier, multiple de 16. Bloc de base par bloc CUDA du scan.");
    println!("               Defaut : 8192, valeur UNIQUE pour toutes les tracks.");
    println!("  wmma_warps : 2 | 4 | 8. Warps par bloc du scan (16 x wmma_warps requetes");
    println!("               par bloc). Defaut 4.");
    println!("  pack_block : multiple de 32 dans [32, 1024]. Threads par bloc du packing");
    println!("               HERITE (pack_impl=0). Defaut 256.");
    println!("  pack_impl  : 1 = packing coalesce par memoire partagee (defaut), 0 = kernel");
    println!("               herite (un thread par ligne). Les deux sont bit-identiques.");
    println!("  k1_arm     : 1 (defaut) = scan a accumulation fp16, deux candidats par");
    println!("               (chunk, requete), puis reduction + filtre + re-rang EXACT fp32.");
    println!("               0 = chemin de reference (accumulation fp32, top-1). Donne un");
    println!("               A/B APPARIE dans le MEME artefact.");
    println!("  k1_delta   : reel >= 0, defaut 2.0. Largeur du filtre de re-rang en");
    println!("               distance^2. Erreur fp16 mesuree sur la distance : max 0.136 ;");
    println!("               le defaut vaut 14,7x cette borne.");
}
