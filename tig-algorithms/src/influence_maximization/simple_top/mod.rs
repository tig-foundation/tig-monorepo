// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::influence_maximization::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("No help information provided.");
}

const MAX_THREADS_PER_BLOCK: u32 = 1024;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let count_degrees_kernel = module.load_function("count_degrees_kernel").unwrap();
    let mut d_degrees = stream
        .alloc_zeros::<i32>(challenge.num_nodes as usize)
        .unwrap();
    unsafe {
        stream
            .launch_builder(&count_degrees_kernel)
            .arg(&challenge.d_from_nodes)
            .arg(&mut d_degrees)
            .arg(&challenge.num_edges)
            .launch(LaunchConfig {
                grid_dim: (
                    (challenge.num_edges as u32 + MAX_THREADS_PER_BLOCK - 1)
                        / MAX_THREADS_PER_BLOCK,
                    1,
                    1,
                ),
                block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    stream.synchronize().unwrap();

    let degrees = stream.memcpy_dtov(&d_degrees).unwrap();
    let mut node_degree_pairs: Vec<(i32, i32)> = degrees
        .iter()
        .enumerate()
        .map(|(node, &deg)| (node as i32, deg))
        .collect();
    node_degree_pairs.sort_by(|a, b| b.1.cmp(&a.1));
    let highest_degree_nodes: Vec<i32> = node_degree_pairs
        .iter()
        .step_by(2)
        .take(challenge.max_starting_nodes as usize)
        .map(|&(node, _)| node)
        .collect();
    let _ = save_solution(&Solution {
        starting_nodes: highest_degree_nodes,
    });
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
