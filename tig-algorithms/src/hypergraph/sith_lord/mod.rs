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
    let block_size = std::cmp::min(512, prop.maxThreadsPerBlock as u32);

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

    let density = challenge.num_hyperedges as f32 / challenge.num_nodes as f32;
    let complexity_factor = if density > 4.0 { 8 } else if density > 2.0 { 6 } else { 4 };
    let base_clusters = challenge.num_parts as usize * complexity_factor;
    let num_hedge_clusters = if challenge.num_hyperedges > 100000 {
        std::cmp::min(base_clusters, 256)
    } else {
        std::cmp::min(base_clusters, 128)
    };

    let sm_count = prop.multiProcessorCount as usize;
    let warp_size = prop.warpSize as usize;

    let batch_size = if challenge.num_nodes < 30000 {
        std::cmp::min(std::cmp::max(32768, sm_count * warp_size * 16), challenge.num_nodes as usize)
    } else if challenge.num_nodes < 100000 {
        std::cmp::max(65536, sm_count * warp_size * 32)
    } else {
        std::cmp::max(131072, sm_count * warp_size * 64)
    };    

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;

    let mut d_pref_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
   
    let mut d_move_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_gains = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    unsafe {
        stream.launch_builder(&hyperedge_cluster_kernel)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_hyperedge_clusters)
            .launch(LaunchConfig {
                grid_dim: ((challenge.num_hyperedges as u32 + block_size - 1) / block_size, 1, 1),
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
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_priorities)
            .launch(cfg.clone())?;
    }
    stream.synchronize()?;

    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..challenge.num_nodes as usize).collect();
    indices.sort_by(|&a, &b| pref_priorities[b].cmp(&pref_priorities[a]));

    let sorted_nodes: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
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

    let mut d_num_valid_moves = stream.alloc_zeros::<i32>(1)?;
    let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;

    let max_rounds = 50;

    for round in 0..max_rounds {
        stream.memset_zeros(&mut d_num_valid_moves)?;

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
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&mut d_move_parts)
                    .arg(&mut d_move_gains)
                    .arg(&mut d_move_priorities)
                    .arg(&mut d_num_valid_moves)
                    .arg(&round)
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

        let move_parts = stream.memcpy_dtov(&d_move_parts)?;
        let move_priorities = stream.memcpy_dtov(&d_move_priorities)?;

        valid_moves.clear();
        valid_moves.reserve(num_valid_moves as usize);
        valid_moves.extend(
            move_priorities.iter().enumerate()
                .filter(|(_, &prio)| prio > 0)
                .map(|(i, &prio)| (i as i32, move_parts[i], prio))
        );

        if valid_moves.is_empty() {
            break;
        }

        valid_moves.sort_unstable_by(|a, b| b.2.cmp(&a.2));

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        sorted_move_nodes.extend(valid_moves.iter().map(|&(node, _, _)| node));
        sorted_move_parts.extend(valid_moves.iter().map(|&(_, part, _)| part));

        let d_sorted_move_nodes_temp = stream.memcpy_stod(&sorted_move_nodes)?;
        let d_sorted_move_parts_temp = stream.memcpy_stod(&sorted_move_parts)?;
        stream.memset_zeros(&mut d_moves_executed)?;

        unsafe {
            stream.launch_builder(&execute_moves_kernel)
                .arg(&(sorted_move_nodes.len() as i32))
                .arg(&d_sorted_move_nodes_temp)
                .arg(&d_sorted_move_parts_temp)
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

        let termination_threshold = if challenge.num_nodes < 50000 {
            (challenge.num_nodes as i32 / 800).max(2)
        } else {
            (challenge.num_nodes as i32 / 1200).max(3)
        };

        if moves_executed < termination_threshold {
            if round > 8 {
                break;
            }
        } else if moves_executed < termination_threshold * 3 && round > 25 {
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

pub fn help() {
    println!("No help information available.");
}
