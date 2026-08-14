// TIG's UI uses the pattern `tig_challenges::vector_search` to automatically detect your algorithm's challenge
use anyhow::Result;
use cudarc::driver::{safe::{LaunchConfig, CudaModule, CudaStream}, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    #[serde(default = "default_db_batch_size")]
    pub db_batch_size: u32,
}

fn default_db_batch_size() -> u32 { 1024 }

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            db_batch_size: 1024,
        }
    }
}

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
    let database_size = challenge.database_size as u32;
    let vector_dims = challenge.vector_dims as u32;

    let num_queries_i = num_queries as i32;
    let vector_dims_i = vector_dims as i32;

    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    let mut d_best_dists = stream.alloc_zeros::<f32>(num_queries as usize)?;

    let search_func = module.load_function("search_warp_per_query")?;

    let config = LaunchConfig {
        grid_dim: (num_queries, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Small batch sizes for frequent inter-batch bound sharing.
    // After each batch, warp_reduce_min writes the global best to best_dists.
    // On the next batch, all 32 threads load this shared bound, effectively
    // sharing bounds 32x faster than within a single batch.
    let db_batch_size = hps.db_batch_size;
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
    }

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= database_size as i32 { 0 } else { idx as usize }
        })
        .collect();

    save_solution(&Solution { indexes })?;

    Ok(())
}

pub fn help() {
    println!("AutoVector v9 — Warp-per-query with frequent inter-batch bound sharing");
    println!();
    println!("32 threads (1 warp) per query with strided DB scan.");
    println!("DB processed in small batches (default 1024 vectors).");
    println!("After each batch, all threads share the global best distance,");
    println!("tightening early-exit bounds 32x faster than single-thread search.");
    println!();
    println!("Hyperparameters:");
    println!("  db_batch_size: DB vectors per batch. Default: 1024");
}
