use anyhow::Result;
use cudarc::driver::{safe::{LaunchConfig, CudaModule, CudaStream}, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    pub block_size: u32,

    #[serde(default)]
    pub batch_size: Option<u32>,

    #[serde(default)]
    pub db_batch_size: Option<u32>,

    #[serde(default)]
    pub force_kernel: Option<String>,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            block_size: 256,
            batch_size: None,
            db_batch_size: None,
            force_kernel: None,
        }
    }
}

// =============================================================================
// Track 1-2: block-per-query (v7), unchanged
// =============================================================================
fn run_block_per_query(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    block_size: u32,
    batch_size: u32,
) -> Result<()> {
    let dim = challenge.vector_dims as u32;
    let num_vectors = challenge.database_size as u32;
    let num_queries = challenge.num_queries as u32;

    let d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    let search_func = module.load_function("search_block_per_query")?;

    let mut h_results: Vec<usize> = Vec::with_capacity(num_queries as usize);

    for batch_start in (0..num_queries).step_by(batch_size as usize) {
        let batch_end = (batch_start + batch_size).min(num_queries);
        let batch_count = batch_end - batch_start;

        unsafe {
            stream.launch_builder(&search_func)
                .arg(&challenge.d_query_vectors)
                .arg(&challenge.d_database_vectors)
                .arg(&d_results)
                .arg(&batch_start)
                .arg(&batch_count)
                .arg(&num_queries)
                .arg(&num_vectors)
                .arg(&dim)
                .launch(LaunchConfig {
                    grid_dim: (batch_count, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        stream.synchronize()?;

        let batch_slice = d_results.slice(batch_start as usize..batch_end as usize);
        let batch_indices: Vec<i32> = stream.memcpy_dtov(&batch_slice)?;

        for &idx in &batch_indices {
            if idx < 0 || idx as u64 >= challenge.database_size as u64 {
                return Err(anyhow::anyhow!(
                    "Invalid index {} at query {} (database_size={})",
                    idx, h_results.len(), challenge.database_size
                ));
            }
            h_results.push(idx as usize);
        }

        save_solution(&Solution { indexes: h_results.clone() })?;
    }

    Ok(())
}

// =============================================================================
// Track 3-5: warp-per-query cooperative search
//
// 32 threads (1 warp) per query. Each query = 1 block of 32 threads.
// DB processed in batches; state persists across batches.
// =============================================================================
fn run_warp_per_query(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    db_batch_size: u32,
) -> Result<()> {
    let num_queries = challenge.num_queries as u32;
    let database_size = challenge.database_size as u32;
    let vector_dims = challenge.vector_dims as u32;

    let num_queries_i = num_queries as i32;
    let vector_dims_i = vector_dims as i32;

    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    let mut d_best_dists = stream.alloc_zeros::<f32>(num_queries as usize)?;

    let search_func = module.load_function("search_warp_per_query")?;

    // 1 block per query, 32 threads (1 warp) per block
    let config = LaunchConfig {
        grid_dim: (num_queries, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let num_batches = (database_size + db_batch_size - 1) / db_batch_size;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * db_batch_size;
        let batch_end = (batch_start + db_batch_size).min(database_size);
        let batch_count = batch_end - batch_start;

        let batch_start_i = batch_start as i32;
        let batch_count_i = batch_count as i32;
        let is_first_batch_i: i32 = if batch_idx == 0 { 1 } else { 0 };

        unsafe {
            stream.launch_builder(&search_func)
                .arg(&challenge.d_query_vectors)
                .arg(&challenge.d_database_vectors)
                .arg(&mut d_results)
                .arg(&mut d_best_dists)
                .arg(&num_queries_i)
                .arg(&vector_dims_i)
                .arg(&batch_start_i)
                .arg(&batch_count_i)
                .arg(&is_first_batch_i)
                .launch(config)?;
        }

        stream.synchronize()?;

        // Intermediate save after each DB batch so we can track progress
        let partial_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
        let partial_indexes: Vec<usize> = partial_indices
            .iter()
            .map(|&idx| {
                if idx < 0 || idx >= database_size as i32 { 0 } else { idx as usize }
            })
            .collect();
        save_solution(&Solution { indexes: partial_indexes })?;
    }

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= database_size as i32 {
                0
            } else {
                idx as usize
            }
        })
        .collect();

    save_solution(&Solution { indexes })?;

    Ok(())
}

// =============================================================================
// Entry point
// =============================================================================
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let hps: Hyperparameters = hyperparameters.as_ref()
        .and_then(|m| serde_json::from_value(serde_json::Value::Object(m.clone())).ok())
        .unwrap_or_default();

    let num_queries = challenge.num_queries as u32;
    let vector_dims = challenge.vector_dims as u32;

    let use_block_kernel = match &hps.force_kernel {
        Some(k) if k == "block" => true,
        Some(k) if k == "warp" => false,
        _ => num_queries <= 9000,
    };

    if use_block_kernel {
        let batch_size = hps.batch_size.unwrap_or_else(|| {
            if num_queries <= 7000 { 256 } else { 128 }
        });
        run_block_per_query(
            challenge, save_solution, &module, &stream,
            hps.block_size, batch_size,
        )
    } else {
        let db_batch_size = hps.db_batch_size.unwrap_or_else(|| {
            if vector_dims >= 512 { 80_000 } else { 200_000 }
        });
        run_warp_per_query(
            challenge, save_solution, &module, &stream,
            db_batch_size,
        )
    }
}

pub fn help() {
    println!("AutoVector v8 Hybrid — Dual-Kernel (TIG v0.0.5)");
    println!();
    println!("Tracks 1-2 (<=9k): Block-per-query, 256 threads, warp shuffle");
    println!("Tracks 3-5 (>9k):  Warp-per-query, 32 threads, cooperative bound sharing");
    println!();
    println!("Hyperparameters:");
    println!("  block_size:     Threads per block [block kernel]. Default: 256");
    println!("  batch_size:     Query batch size [block kernel]. Default: auto");
    println!("  db_batch_size:  DB vectors per batch [warp kernel]. Default: auto");
    println!("  force_kernel:   'block' or 'warp' to override auto-selection");
}