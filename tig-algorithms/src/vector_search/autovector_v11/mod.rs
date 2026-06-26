// TIG's UI uses the pattern `tig_challenges::vector_search` to automatically detect your algorithm's challenge
// Copyright (c) 2026 NVX
use anyhow::Result;
use cudarc::driver::{safe::{LaunchConfig, CudaModule, CudaStream}, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {}
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let num_queries = challenge.num_queries as i32;
    let database_size = challenge.database_size as i32;

    let mut d_results = stream.alloc_zeros::<i32>(challenge.num_queries as usize)?;
    let mut d_best_dists = stream.alloc_zeros::<f32>(challenge.num_queries as usize)?;

    let search_func = module.load_function("search_coalesced_v11")?;

    let queries_per_block: u32 = 8;
    let threads_per_block: u32 = 256;
    let num_blocks = (num_queries as u32 + queries_per_block - 1) / queries_per_block;

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&search_func)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_results)
            .arg(&mut d_best_dists)
            .arg(&num_queries)
            .arg(&database_size)
            .launch(config)?;
    }

    stream.synchronize()?;

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| {
            if idx < 0 || idx >= database_size { 0 } else { idx as usize }
        })
        .collect();

    save_solution(&Solution { indexes })?;

    Ok(())
}

pub fn help() {
    println!("AutoVector v11 — Coalesced Shared Memory Tiling");
}
