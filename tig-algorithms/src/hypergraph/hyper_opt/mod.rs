// TIG's UI uses the pattern `tig_challenges::hypergraph` to automatically detect your algorithm's challenge
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::hypergraph::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    #[serde(default = "default_refinement")]
    pub refinement: usize,
}

fn default_refinement() -> usize { 500 }

pub fn help() {
    println!("HyperOpt — Optimized hypergraph partitioner");
    println!("Uses conn[64] gain computation (no 72MB buffer)");
    println!();
    println!("Hyperparameters:");
    println!("  refinement - Number of refinement rounds (default: 500)");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let dummy: Vec<u32> = (0..challenge.num_nodes as u32)
        .map(|i| i % challenge.num_parts as u32)
        .collect();
    save_solution(&Solution { partition: dummy })?;

    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_fn = module.load_function("hyperedge_clustering")?;
    let compute_prefs_fn = module.load_function("compute_node_preferences")?;
    let execute_assign_fn = module.load_function("execute_node_assignments")?;
    let compute_moves_fn = module.load_function("compute_moves_v2")?;
    let execute_moves_fn = module.load_function("execute_refinement_moves")?;
    let balance_fn = module.load_function("balance_final")?;

    let n = challenge.num_nodes as usize;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let one_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let hedge_cfg = LaunchConfig {
        grid_dim: ((challenge.num_hyperedges as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_partition = stream.alloc_zeros::<i32>(n)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(n)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(n)?;
    let mut d_move_targets = stream.alloc_zeros::<i32>(n)?;
    let mut d_move_gains = stream.alloc_zeros::<i32>(n)?;
    let mut d_num_valid = stream.alloc_zeros::<i32>(1)?;
    let mut d_moves_exec = stream.alloc_zeros::<i32>(1)?;
    let mut d_sorted_nodes_buf = stream.alloc_zeros::<i32>(n)?;
    let mut d_sorted_targets_buf = stream.alloc_zeros::<i32>(n)?;

    let mut gains_buf: Vec<i32> = vec![0i32; n];
    let mut targets_buf: Vec<i32> = vec![0i32; n];
    let mut valid_idx: Vec<usize> = Vec::with_capacity(n);
    let mut sorted_nodes_host: Vec<i32> = Vec::with_capacity(n);
    let mut sorted_targets_host: Vec<i32> = Vec::with_capacity(n);
    let mut result_buf: Vec<i32> = vec![0i32; 1];

    let refinement_rounds = if let Some(params) = hyperparameters {
        params.get("refinement")
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(50, 10000) as usize)
            .unwrap_or(500)
    } else {
        500
    };

    let num_hedge_clusters: i32 = 64;

    unsafe {
        stream.launch_builder(&hyperedge_cluster_fn)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&num_hedge_clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_hyperedge_clusters)
            .launch(hedge_cfg.clone())?;
    }

    unsafe {
        stream.launch_builder(&compute_prefs_fn)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&num_hedge_clusters)
            .arg(&challenge.d_node_hyperedges)
            .arg(&challenge.d_node_offsets)
            .arg(&d_hyperedge_clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_priorities)
            .launch(cfg.clone())?;
    }
    stream.synchronize()?;

    let pref_parts: Vec<i32> = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities: Vec<i32> = stream.memcpy_dtov(&d_pref_priorities)?;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| pref_priorities[b].cmp(&pref_priorities[a]));

    sorted_nodes_host.clear();
    sorted_nodes_host.extend(indices.iter().map(|&i| i as i32));
    sorted_targets_host.clear();
    sorted_targets_host.extend(indices.iter().map(|&i| pref_parts[i]));

    stream.memcpy_htod(&sorted_nodes_host, &mut d_sorted_nodes_buf)?;
    stream.memcpy_htod(&sorted_targets_host, &mut d_sorted_targets_buf)?;

    unsafe {
        stream.launch_builder(&execute_assign_fn)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(challenge.max_part_size as i32))
            .arg(&d_sorted_nodes_buf)
            .arg(&d_sorted_targets_buf)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_cfg.clone())?;
    }
    stream.synchronize()?;

    let mut stagnant = 0;
    let balance_weight: i32 = 4;

    for round in 0..refinement_rounds {
        stream.memset_zeros(&mut d_num_valid)?;

        unsafe {
            stream.launch_builder(&compute_moves_fn)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&balance_weight)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_move_targets)
                .arg(&mut d_move_gains)
                .arg(&mut d_num_valid)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        stream.memcpy_dtoh(&d_move_gains, &mut gains_buf)?;
        stream.memcpy_dtoh(&d_move_targets, &mut targets_buf)?;

        valid_idx.clear();
        valid_idx.extend(
            gains_buf.iter()
                .enumerate()
                .filter(|(_, &g)| g > 0)
                .map(|(i, _)| i)
        );
        if valid_idx.is_empty() { break; }
        valid_idx.sort_unstable_by(|&a, &b| gains_buf[b].cmp(&gains_buf[a]));

        let num_to_process = valid_idx.len() as i32;

        sorted_nodes_host.clear();
        sorted_nodes_host.extend(valid_idx.iter().map(|&i| i as i32));
        sorted_targets_host.clear();
        sorted_targets_host.extend(valid_idx.iter().map(|&i| targets_buf[i]));

        stream.memcpy_htod(&sorted_nodes_host, &mut d_sorted_nodes_buf)?;
        stream.memcpy_htod(&sorted_targets_host, &mut d_sorted_targets_buf)?;
        stream.memset_zeros(&mut d_moves_exec)?;

        unsafe {
            stream.launch_builder(&execute_moves_fn)
                .arg(&num_to_process)
                .arg(&d_sorted_nodes_buf)
                .arg(&d_sorted_targets_buf)
                .arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&mut d_moves_exec)
                .launch(one_cfg.clone())?;
        }
        stream.synchronize()?;

        stream.memcpy_dtoh(&d_moves_exec, &mut result_buf)?;
        let moves_done = result_buf[0];
        if moves_done == 0 { break; }

        if moves_done <= 1 && round > 90 {
            stagnant += 1;
            if stagnant > 30 { break; }
        } else {
            stagnant = 0;
        }
    }

    unsafe {
        stream.launch_builder(&balance_fn)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&1i32)
            .arg(&(challenge.max_part_size as i32))
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_cfg.clone())?;
    }
    stream.synchronize()?;

    for _ in 0..24 {
        stream.memset_zeros(&mut d_num_valid)?;

        unsafe {
            stream.launch_builder(&compute_moves_fn)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&balance_weight)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_move_targets)
                .arg(&mut d_move_gains)
                .arg(&mut d_num_valid)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        stream.memcpy_dtoh(&d_move_gains, &mut gains_buf)?;
        stream.memcpy_dtoh(&d_move_targets, &mut targets_buf)?;

        valid_idx.clear();
        valid_idx.extend(
            gains_buf.iter()
                .enumerate()
                .filter(|(_, &g)| g > 0)
                .map(|(i, _)| i)
        );
        if valid_idx.is_empty() { break; }
        valid_idx.sort_unstable_by(|&a, &b| gains_buf[b].cmp(&gains_buf[a]));

        let num_to_process = valid_idx.len() as i32;
        sorted_nodes_host.clear();
        sorted_nodes_host.extend(valid_idx.iter().map(|&i| i as i32));
        sorted_targets_host.clear();
        sorted_targets_host.extend(valid_idx.iter().map(|&i| targets_buf[i]));

        stream.memcpy_htod(&sorted_nodes_host, &mut d_sorted_nodes_buf)?;
        stream.memcpy_htod(&sorted_targets_host, &mut d_sorted_targets_buf)?;
        stream.memset_zeros(&mut d_moves_exec)?;

        unsafe {
            stream.launch_builder(&execute_moves_fn)
                .arg(&num_to_process)
                .arg(&d_sorted_nodes_buf)
                .arg(&d_sorted_targets_buf)
                .arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&mut d_moves_exec)
                .launch(one_cfg.clone())?;
        }
        stream.synchronize()?;

        if result_buf[0] == 0 { break; }
    }

    let partition: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();
    save_solution(&Solution { partition: partition_u32 })?;

    Ok(())
}
