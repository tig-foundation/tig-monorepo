use anyhow::Result;
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::{Challenge, Solution};

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let num_nodes = challenge.num_nodes as usize;
    let num_hyperedges = challenge.num_hyperedges as usize;
    let num_parts = challenge.num_parts as usize;

    let rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100) as usize)
        .unwrap_or(15);

    // 1. Allocate device buffers
    let mut d_partition = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_part_sizes = stream.alloc_zeros::<i32>(num_parts)?;
    let mut d_hyperedge_masks = stream.alloc_zeros::<u64>(num_hyperedges)?;
    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(num_hyperedges)?;
    let mut d_best_targets = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_best_gains = stream.alloc_zeros::<i32>(num_nodes)?;

    // 2. Load kernels
    let cluster_k = module.load_function("ada_hyperedge_clustering_10k")?;
    let pref_k = module.load_function("ada_compute_initial_preferences_10k")?;
    let compute_masks_k = module.load_function("ada_compute_hyperedge_masks_10k")?;
    let find_moves_k = module.load_function("ada_find_best_moves_10k")?;
    let apply_moves_k = module.load_function("ada_apply_moves_10k")?;
    let balance_k = module.load_function("ada_enforce_balance_10k")?;

    let node_block = 256;
    let node_cfg = LaunchConfig {
        grid_dim: ((challenge.num_nodes + node_block - 1) / node_block, 1, 1),
        block_dim: (node_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let hedge_block = 256;
    let hedge_cfg = LaunchConfig {
        grid_dim: ((challenge.num_hyperedges + hedge_block - 1) / hedge_block, 1, 1),
        block_dim: (hedge_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // 3. Stage 1: Initial clustering and preferences
    let seed_u32 = u32::from_le_bytes([challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3]]);
    let num_clusters = 64i32;

    unsafe {
        stream.launch_builder(&cluster_k)
            .arg(&challenge.num_hyperedges)
            .arg(&challenge.d_hyperedge_sizes)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&num_clusters)
            .arg(&seed_u32)
            .arg(&mut d_hyperedge_clusters)
            .launch(hedge_cfg.clone())?;

        stream.launch_builder(&pref_k)
            .arg(&challenge.num_nodes)
            .arg(&challenge.num_parts)
            .arg(&num_clusters)
            .arg(&challenge.d_node_offsets)
            .arg(&challenge.d_node_hyperedges)
            .arg(&d_hyperedge_clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_partition)
            .launch(node_cfg.clone())?;
    }

    // Initialize partition sizes
    let init_part_host: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
    let mut part_sizes_host = vec![0i32; num_parts];
    for &p in &init_part_host {
        let p_idx = (p as usize).min(num_parts - 1);
        part_sizes_host[p_idx] += 1;
    }
    d_part_sizes = stream.memcpy_stod(&part_sizes_host)?;

    // Balance initial partition
    unsafe {
        stream.launch_builder(&balance_k)
            .arg(&challenge.num_nodes)
            .arg(&challenge.num_parts)
            .arg(&challenge.max_part_size)
            .arg(&mut d_partition)
            .arg(&mut d_part_sizes)
            .launch(node_cfg.clone())?;
    }

    // 4. Stage 2: Bitmask-accelerated GPU Refinement
    for _ in 0..rounds {
        unsafe {
            stream.launch_builder(&compute_masks_k)
                .arg(&challenge.num_hyperedges)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&d_partition)
                .arg(&mut d_hyperedge_masks)
                .launch(hedge_cfg.clone())?;

            stream.launch_builder(&find_moves_k)
                .arg(&challenge.num_nodes)
                .arg(&challenge.num_parts)
                .arg(&challenge.max_part_size)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&d_hyperedge_masks)
                .arg(&d_partition)
                .arg(&d_part_sizes)
                .arg(&mut d_best_targets)
                .arg(&mut d_best_gains)
                .launch(node_cfg.clone())?;

            stream.launch_builder(&apply_moves_k)
                .arg(&challenge.num_nodes)
                .arg(&challenge.max_part_size)
                .arg(&d_best_targets)
                .arg(&d_best_gains)
                .arg(&mut d_partition)
                .arg(&mut d_part_sizes)
                .launch(node_cfg.clone())?;

            stream.launch_builder(&balance_k)
                .arg(&challenge.num_nodes)
                .arg(&challenge.num_parts)
                .arg(&challenge.max_part_size)
                .arg(&mut d_partition)
                .arg(&mut d_part_sizes)
                .launch(node_cfg.clone())?;
        }
    }

    stream.synchronize()?;

    // 5. Save the valid partition
    let final_part: Vec<u32> = stream.memcpy_dtov(&d_partition)?
        .into_iter()
        .map(|x| (x.max(0) as u32).min((challenge.num_parts - 1) as u32))
        .collect();

    save_solution(&Solution { partition: final_part })?;
    Ok(())
}
