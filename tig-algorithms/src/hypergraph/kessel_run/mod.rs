use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde_json::{Map, Value};
use tig_challenges::hypergraph::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let block_size = std::cmp::min(256, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves_batched")?;
    let execute_moves_kernel = module.load_function("execute_refinement_moves")?;
    let balance_kernel = module.load_function("balance_final")?;

    let cfg = LaunchConfig {
        grid_dim: ((challenge.num_nodes as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let one_thread_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let density = challenge.difficulty.num_hyperedges as f32 / challenge.num_nodes as f32;
    let complexity_factor = if density > 4.0 { 8 } else if density > 2.0 { 6 } else { 4 };
    let base_clusters = challenge.num_parts as usize * complexity_factor;
    let num_hedge_clusters = if challenge.difficulty.num_hyperedges > 100000 {
        std::cmp::min(base_clusters, 256)
    } else {
        std::cmp::min(base_clusters, 128)
    };

    let node_offsets = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let max_degree = node_offsets.windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max().unwrap_or(0);
    let safe_max_degree = std::cmp::min(max_degree, 2000);

    let batch_size = if challenge.num_nodes < 30000 {
        std::cmp::min(32768, challenge.num_nodes as usize)
    } else if challenge.num_nodes < 100000 {
        65536
    } else {
        131072
    };
    let buffer_size = batch_size * safe_max_degree;

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(challenge.difficulty.num_hyperedges as usize)?;
    let mut d_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;

    let mut d_pref_nodes = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_gains = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let mut d_move_nodes = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_gains = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let mut d_edge_flags_buffer = stream.alloc_zeros::<u64>(buffer_size)?;

    unsafe {
        stream.launch_builder(&hyperedge_cluster_kernel)
            .arg(&(challenge.difficulty.num_hyperedges as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_hyperedge_clusters)
            .launch(LaunchConfig {
                grid_dim: ((challenge.difficulty.num_hyperedges as u32 + block_size - 1) / block_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    stream.synchronize()?;

    unsafe {
        stream.launch_builder(&compute_preferences_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_node_hyperedges)
            .arg(&challenge.d_node_offsets)
            .arg(&d_hyperedge_clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_pref_nodes)
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_gains)
            .arg(&mut d_pref_priorities)
            .launch(cfg.clone())?;
    }
    stream.synchronize()?;

    let pref_nodes = stream.memcpy_dtov(&d_pref_nodes)?;
    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..challenge.num_nodes as usize).collect();
    indices.sort_by(|&a, &b| pref_priorities[b].cmp(&pref_priorities[a]));

    let sorted_nodes: Vec<i32> = indices.iter().map(|&i| pref_nodes[i]).collect();
    let sorted_parts: Vec<i32> = indices.iter().map(|&i| pref_parts[i]).collect();

    let d_sorted_nodes = stream.memcpy_stod(&sorted_nodes)?;
    let d_sorted_parts = stream.memcpy_stod(&sorted_parts)?;

    unsafe {
        stream.launch_builder(&execute_assignments_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(challenge.max_part_size as i32))
            .arg(&d_sorted_nodes)
            .arg(&d_sorted_parts)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }
    stream.synchronize()?;

    let mut valid_moves: Vec<(i32, i32, i32)> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);

    let max_rounds = 50;  

    for round in 0..max_rounds {
        let mut d_num_valid_moves = stream.alloc_zeros::<i32>(1)?;

        let num_batches = (challenge.num_nodes as usize + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let current_batch_size = std::cmp::min(batch_size, challenge.num_nodes as usize - batch_start);

            unsafe {
                stream.launch_builder(&compute_moves_kernel)
                    .arg(&(batch_start as i32))
                    .arg(&(current_batch_size as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&(challenge.difficulty.num_hyperedges as i32))
                    .arg(&(safe_max_degree as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&mut d_move_nodes)
                    .arg(&mut d_move_parts)
                    .arg(&mut d_move_gains)
                    .arg(&mut d_move_priorities)
                    .arg(&mut d_num_valid_moves)
                    .arg(&round)
                    .arg(&mut d_edge_flags_buffer)
                    .launch(LaunchConfig {
                        grid_dim: ((current_batch_size as u32 + block_size - 1) / block_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        stream.synchronize()?;

        let num_valid_moves = stream.memcpy_dtov(&d_num_valid_moves)?[0];
        if num_valid_moves == 0 {
            break;
        }

        let move_gains = stream.memcpy_dtov(&d_move_gains)?;
        let valid_indices: Vec<usize> = move_gains.iter().enumerate()
            .filter(|(_, &gain)| gain > 0)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.is_empty() {
            break;
        }

        let move_nodes = stream.memcpy_dtov(&d_move_nodes)?;
        let move_parts = stream.memcpy_dtov(&d_move_parts)?;
        let move_priorities = stream.memcpy_dtov(&d_move_priorities)?;

        valid_moves.clear();
        for &i in &valid_indices {
            valid_moves.push((move_nodes[i], move_parts[i], move_priorities[i]));
        }

        valid_moves.sort_by(|a, b| b.2.cmp(&a.2));

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        sorted_move_nodes.extend(valid_moves.iter().map(|&(node, _, _)| node));
        sorted_move_parts.extend(valid_moves.iter().map(|&(_, part, _)| part));

        let d_sorted_move_nodes = stream.memcpy_stod(&sorted_move_nodes)?;
        let d_sorted_move_parts = stream.memcpy_stod(&sorted_move_parts)?;
        let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;

        unsafe {
            stream.launch_builder(&execute_moves_kernel)
                .arg(&(sorted_move_nodes.len() as i32))
                .arg(&d_sorted_move_nodes)
                .arg(&d_sorted_move_parts)
                .arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&mut d_moves_executed)
                .launch(one_thread_cfg.clone())?;
        }
        stream.synchronize()?;

        let moves_executed = stream.memcpy_dtov(&d_moves_executed)?[0];
        if moves_executed == 0 {
            break;
        }        
    }

    unsafe {
        stream.launch_builder(&balance_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&1)
            .arg(&(challenge.max_part_size as i32))
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }
    stream.synchronize()?;

    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    let _ = save_solution(&Solution { partition: partition_u32 });
    return Ok(());
}