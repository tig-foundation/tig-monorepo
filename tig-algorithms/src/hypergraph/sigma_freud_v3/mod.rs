use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg}, //ALL imports MUST be present, DO NOT remove!!!
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {    
    let dummy_partition: Vec<u32> = (0..challenge.num_nodes as u32)
        .map(|i| i % challenge.num_parts as u32)
        .collect();
    save_solution(&Solution {
        partition: dummy_partition,
    })?;

    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves")?;
    let execute_moves_kernel = module.load_function("execute_refinement_moves")?;
    let balance_kernel = module.load_function("balance_final")?;
    let reset_counters_kernel = module.load_function("reset_counters")?;

    let cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_nodes as u32 + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let one_thread_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut num_hedge_clusters = if let Some(params) = hyperparameters {
        params
            .get("clusters")
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(4, 256) as i32)
            .unwrap_or(64)
    } else {
        64
    };

    if num_hedge_clusters % 4 != 0 {
        num_hedge_clusters += 4 - (num_hedge_clusters % 4);
    }

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;

    let mut d_pref_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
   
    let mut d_move_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let mut d_num_valid_moves = stream.alloc_zeros::<i32>(1)?;
    let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;

    let is_sparse = (challenge.num_nodes as usize) > 4 * (challenge.num_hyperedges as usize + 1);

    let refinement_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(50, 5000) as usize)
        .unwrap_or(500);

    let move_limit: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("move_limit").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(256, 1_000_000) as usize)
        .unwrap_or(if is_sparse {
            262_144
        } else if challenge.num_hyperedges as usize >= 150_000 || challenge.num_nodes as usize >= 250_000 {
            131_072
        } else {
            200_000
        });

    unsafe {
        stream
            .launch_builder(&hyperedge_cluster_kernel)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&mut d_hyperedge_clusters)
            .launch(LaunchConfig {
                grid_dim: (
                    (challenge.num_hyperedges as u32 + block_size - 1) / block_size,
                    1,
                    1,
                ),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    unsafe {
        stream
            .launch_builder(&compute_preferences_kernel)
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

    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..challenge.num_nodes as usize).collect();
    indices.sort_unstable_by(|&a, &b| {
        pref_priorities[b]
            .cmp(&pref_priorities[a])
            .then_with(|| a.cmp(&b))
    });

    let sorted_nodes: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let sorted_parts: Vec<i32> = indices.iter().map(|&i| pref_parts[i]).collect();

    let d_sorted_nodes = stream.memcpy_stod(&sorted_nodes)?;
    let d_sorted_parts = stream.memcpy_stod(&sorted_parts)?;

    unsafe {
        stream
            .launch_builder(&execute_assignments_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(challenge.max_part_size as i32))
            .arg(&d_sorted_nodes)
            .arg(&d_sorted_parts)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }

    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);

    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(challenge.num_nodes as usize);

    let mut stagnant_rounds = 0;
    let early_exit_round = if challenge.num_hyperedges < 20_000 { 90 } else { 70 };
    let max_stagnant_rounds = if challenge.num_hyperedges < 20_000 { 30 } else { 20 };

    for round in 0..refinement_rounds {
        unsafe {
            stream
                .launch_builder(&reset_counters_kernel)
                .arg(&mut d_num_valid_moves)
                .arg(&mut d_moves_executed)
                .launch(one_thread_cfg.clone())?;
        }

        unsafe {
            stream
                .launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .launch(cfg.clone())?;
        }
        let num_valid_moves = stream.memcpy_dtov(&d_num_valid_moves)?[0];
        if num_valid_moves == 0 {
            break;
        }

        let move_keys = stream.memcpy_dtov(&d_move_priorities)?;

        valid_moves.clear();
        for (node, &key) in move_keys.iter().enumerate() {
            if key > 0 {
                valid_moves.push((node, key));
            }
        }

        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

        let mut k = valid_moves.len();
        if k > move_limit {
            k = move_limit;
            valid_moves.select_nth_unstable_by(k - 1, cmp);
            valid_moves[..k].sort_unstable_by(cmp);
        } else {
            valid_moves.sort_unstable_by(cmp);
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        sorted_move_nodes.extend(valid_moves[..k].iter().map(|(node, _)| *node as i32));
        sorted_move_parts.extend(valid_moves[..k].iter().map(|(_, key)| (key & 63) as i32));

        let d_sorted_move_nodes = stream.memcpy_stod(&sorted_move_nodes)?;
        let d_sorted_move_parts = stream.memcpy_stod(&sorted_move_parts)?;

        unsafe {
            stream
                .launch_builder(&execute_moves_kernel)
                .arg(&(sorted_move_nodes.len() as i32))
                .arg(&d_sorted_move_nodes)
                .arg(&d_sorted_move_parts)
                .arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&mut d_moves_executed)
                .launch(one_thread_cfg.clone())?;
        }
        let mut moves_executed = stream.memcpy_dtov(&d_moves_executed)?[0];

        if moves_executed == 0 && k < valid_moves.len() {
            valid_moves.sort_unstable_by(cmp);

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            sorted_move_nodes.extend(valid_moves.iter().map(|(node, _)| *node as i32));
            sorted_move_parts.extend(valid_moves.iter().map(|(_, key)| (key & 63) as i32));

            let d_sorted_move_nodes = stream.memcpy_stod(&sorted_move_nodes)?;
            let d_sorted_move_parts = stream.memcpy_stod(&sorted_move_parts)?;

            unsafe {
                stream
                    .launch_builder(&execute_moves_kernel)
                    .arg(&(sorted_move_nodes.len() as i32))
                    .arg(&d_sorted_move_nodes)
                    .arg(&d_sorted_move_parts)
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&mut d_moves_executed)
                    .launch(one_thread_cfg.clone())?;
            }

            moves_executed = stream.memcpy_dtov(&d_moves_executed)?[0];
        }

        if moves_executed == 0 {
            break;
        }

        if moves_executed == 1 && round > early_exit_round {
            stagnant_rounds += 1;
            if stagnant_rounds > max_stagnant_rounds {
                break;
            }
        } else {
            stagnant_rounds = 0;
        }
    }

    unsafe {
        stream
            .launch_builder(&balance_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&1i32)
            .arg(&(challenge.max_part_size as i32))
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }

    let post_balance_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 128) as usize)
        .unwrap_or(64);

    for _ in 0..post_balance_rounds {
        unsafe {
            stream
                .launch_builder(&reset_counters_kernel)
                .arg(&mut d_num_valid_moves)
                .arg(&mut d_moves_executed)
                .launch(one_thread_cfg.clone())?;
        }

        unsafe {
            stream
                .launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .launch(cfg.clone())?;
        }
        let num_valid_moves = stream.memcpy_dtov(&d_num_valid_moves)?[0];
        if num_valid_moves == 0 {
            break;
        }

        let move_keys = stream.memcpy_dtov(&d_move_priorities)?;

        valid_moves.clear();
        for (node, &key) in move_keys.iter().enumerate() {
            if key > 0 {
                valid_moves.push((node, key));
            }
        }

        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

        let mut k = valid_moves.len();
        if k > move_limit {
            k = move_limit;
            valid_moves.select_nth_unstable_by(k - 1, cmp);
            valid_moves[..k].sort_unstable_by(cmp);
        } else {
            valid_moves.sort_unstable_by(cmp);
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        sorted_move_nodes.extend(valid_moves[..k].iter().map(|(node, _)| *node as i32));
        sorted_move_parts.extend(valid_moves[..k].iter().map(|(_, key)| (key & 63) as i32));

        let d_sorted_move_nodes = stream.memcpy_stod(&sorted_move_nodes)?;
        let d_sorted_move_parts = stream.memcpy_stod(&sorted_move_parts)?;

        unsafe {
            stream
                .launch_builder(&execute_moves_kernel)
                .arg(&(sorted_move_nodes.len() as i32))
                .arg(&d_sorted_move_nodes)
                .arg(&d_sorted_move_parts)
                .arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&mut d_moves_executed)
                .launch(one_thread_cfg.clone())?;
        }
        let mut moves_executed = stream.memcpy_dtov(&d_moves_executed)?[0];

        if moves_executed == 0 && k < valid_moves.len() {
            valid_moves.sort_unstable_by(cmp);

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            sorted_move_nodes.extend(valid_moves.iter().map(|(node, _)| *node as i32));
            sorted_move_parts.extend(valid_moves.iter().map(|(_, key)| (key & 63) as i32));

            let d_sorted_move_nodes = stream.memcpy_stod(&sorted_move_nodes)?;
            let d_sorted_move_parts = stream.memcpy_stod(&sorted_move_parts)?;

            unsafe {
                stream
                    .launch_builder(&execute_moves_kernel)
                    .arg(&(sorted_move_nodes.len() as i32))
                    .arg(&d_sorted_move_nodes)
                    .arg(&d_sorted_move_parts)
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&mut d_moves_executed)
                    .launch(one_thread_cfg.clone())?;
            }

            moves_executed = stream.memcpy_dtov(&d_moves_executed)?[0];
        }

        if moves_executed == 0 {
            break;
        }
    }

    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    save_solution(&Solution {
        partition: partition_u32,
    })?;
    Ok(())
}
//This help function MUST be present in it's entrirety, amend as necessary but DO NOT remove!!!
pub fn help() {
    println!("Hypergraph Partitioning - GPU-accelerated with adaptive clustering");
    println!();
    println!("Hyperparameters:");
    println!("  refinement      GPU refinement iterations (50-5000, default: 500)");
    println!("                  More iterations = better quality, longer runtime");
    println!("  post_refinement Extra balance passes after main solve (0-128, default: 64)");
    println!("                  Improves partition balance, slight runtime cost");
    println!("  You can override the default adaptive values by setting the hyperparameters explicitly.");
}