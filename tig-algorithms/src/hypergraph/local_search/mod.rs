use anyhow::Result;
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let sort_nodes_by_degrees_kernel = module.load_function("sort_nodes_by_degrees")?;
    let my_greedy_bipartition_kernel = module.load_function("my_greedy_bipartition")?;
    let my_finalize_bipartition_kernel = module.load_function("my_finalize_bipartition")?;
    let count_nodes_in_part_kernel = module.load_function("my_count_nodes_in_part")?;
    let local_search_kernel = module.load_function("local_search")?;
    let update_part_kernel = module.load_function("update_part")?;

    let block_size = prop.maxThreadsPerBlock as u32;
    let cfg = LaunchConfig {
        grid_dim: ((challenge.num_nodes + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let one_block_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let one_thread_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_sorted_nodes = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;
    let mut d_partition = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;

    unsafe {
        stream
            .launch_builder(&sort_nodes_by_degrees_kernel)
            .arg(&challenge.num_nodes)
            .arg(&challenge.d_node_degrees)
            .arg(&mut d_sorted_nodes)
            .launch(cfg.clone())?;
    }
    stream.synchronize()?;

    for level in 0..6 {
        let num_partitions_this_level = 1 << level;
        let num_flags = (challenge.num_hyperedges + 63) / 64 * num_partitions_this_level;
        let mut d_left_edge_flags = stream.alloc_zeros::<u64>(num_flags as usize)?;
        let mut d_right_edge_flags = stream.alloc_zeros::<u64>(num_flags as usize)?;
        let d_curr_partition = d_partition.clone();

        unsafe {
            stream
                .launch_builder(&my_greedy_bipartition_kernel)
                .arg(&level)
                .arg(&challenge.num_nodes)
                .arg(&challenge.num_hyperedges)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_sorted_nodes)
                .arg(&challenge.d_node_degrees)
                .arg(&d_curr_partition)
                .arg(&mut d_partition)
                .arg(&mut d_left_edge_flags)
                .arg(&mut d_right_edge_flags)
                .launch(LaunchConfig {
                    grid_dim: (num_partitions_this_level as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;
    }

    unsafe {
        stream
            .launch_builder(&my_finalize_bipartition_kernel)
            .arg(&challenge.num_nodes)
            .arg(&challenge.num_parts)
            .arg(&mut d_partition)
            .launch(one_block_cfg.clone())?;
    }
    stream.synchronize()?;

    let mut d_nodes_in_part = stream.alloc_zeros::<u32>(challenge.num_parts as usize)?;
    unsafe {
        stream
            .launch_builder(&count_nodes_in_part_kernel)
            .arg(&challenge.num_nodes)
            .arg(&challenge.num_parts)
            .arg(&d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_block_cfg.clone())?;
    }
    stream.synchronize()?;

    loop {
        let mut d_edge_flags = stream.alloc_zeros::<u64>(challenge.total_connections as usize)?;
        let mut d_best_part = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;
        let mut d_best_diff = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;

        unsafe {
            stream
                .launch_builder(&local_search_kernel)
                .arg(&challenge.num_nodes)
                .arg(&challenge.num_parts)
                .arg(&challenge.max_part_size)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags)
                .arg(&mut d_best_part)
                .arg(&mut d_best_diff)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let best_diff = stream.memcpy_dtov(&d_best_diff)?;
        let best_part = stream.memcpy_dtov(&d_best_part)?;
        let max_diff = *best_diff.iter().max().unwrap();
        if max_diff == 0 {
            break;
        }
        let node = best_diff.iter().position(|&x| x == max_diff).unwrap();

        unsafe {
            stream
                .launch_builder(&update_part_kernel)
                .arg(&(node as u32))
                .arg(&best_part[node])
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .launch(one_thread_cfg.clone())?;
        }
        stream.synchronize()?;
    }

    let partition = stream.memcpy_dtov(&d_partition)?;
    let _ = save_solution(&Solution { partition })?;
    Ok(())
}


pub fn help() {
    println!("No help information provided.");
}

