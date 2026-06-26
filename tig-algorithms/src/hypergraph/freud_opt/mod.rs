use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use std::time::Instant;
use serde_json::{Map, Value};
use tig_challenges::hypergraph::*;

pub fn help() {
    println!("Hypergraph Partitioning Algorithm");
    println!("Adaptive clustering with GPU-accelerated refinement");
    println!();
    println!("Hyperparameters:");
    println!("  refinement - Number of refinement rounds (default: 500, range: 50-5000)");
    println!();
    println!("Usage:");
    println!("  Set the 'refinement' parameter in your benchmarker config");
    println!("  to balance between solution quality and runtime.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    println!(">>> solve_challenge START");
    let total_start = Instant::now();

    let dummy_partition: Vec<u32> = (0..challenge.num_nodes as u32)
        .map(|i| i % challenge.num_parts as u32)
        .collect();
    save_solution(&Solution { partition: dummy_partition })?;

    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let t_load = Instant::now();
    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves")?;
    let execute_moves_kernel = module.load_function("execute_refinement_moves")?;
    let balance_kernel = module.load_function("balance_final")?;
    let radix_hist_kernel = module.load_function("radix_histogram_chunked")?;
    let radix_prefix_scatter_kernel = module.load_function("radix_prefix_and_scatter")?;
    let init_indices_kernel = module.load_function("init_indices")?;
    let invert_keys_kernel = module.load_function("invert_keys")?;
    let gather_sorted_kernel = module.load_function("gather_sorted")?;
    let t_load_elapsed = t_load.elapsed();

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

    let hedge_cfg = LaunchConfig {
        grid_dim: ((challenge.num_hyperedges as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };       

    let mut num_hedge_clusters = 64;

    let t_alloc = Instant::now();
    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;

    let mut d_pref_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let mut d_move_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let buffer_size = (challenge.num_nodes as usize) * 1024;
    let mut d_global_edge_flags = stream.alloc_zeros::<u64>(buffer_size)?;

    let mut d_edge_flags_all = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
    let mut d_edge_flags_double = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;

    let n = challenge.num_nodes as usize;
    let mut d_sort_keys_a = stream.alloc_zeros::<i32>(n)?;
    let mut d_sort_keys_b = stream.alloc_zeros::<i32>(n)?;
    let mut d_sort_vals_a = stream.alloc_zeros::<i32>(n)?;
    let mut d_sort_vals_b = stream.alloc_zeros::<i32>(n)?;
    let mut d_sorted_move_parts = stream.alloc_zeros::<i32>(n)?;

    let num_chunks: i32 = ((n + 255) / 256) as i32;
    let mut d_chunk_histograms = stream.alloc_zeros::<i32>((num_chunks as usize) * 256)?;
    let mut d_chunk_offsets = stream.alloc_zeros::<i32>((num_chunks as usize) * 256)?;
    let mut d_ready_flag = stream.alloc_zeros::<i32>(1)?;
    let t_alloc_elapsed = t_alloc.elapsed();

    let radix_cfg = LaunchConfig {
        grid_dim: (num_chunks as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(n);
    let mut sorted_move_parts_cpu: Vec<i32> = Vec::with_capacity(n);
    let mut valid_indices: Vec<usize> = Vec::with_capacity(n);    

    let default_refinement = if challenge.num_hyperedges < 20_000 {
        400usize
    } else {
        500usize
    };
    
    println!("refinement: {:?}", hyperparameters.as_ref().and_then(|p| p.get("refinement")));

    let refinement_rounds = if let Some(params) = hyperparameters {
        params.get("refinement")
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(50, 5000) as usize)
            .unwrap_or(default_refinement)
    } else {
        default_refinement
    };

    let t_init = Instant::now();
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
    indices.sort_unstable_by(|&a, &b| pref_priorities[b].cmp(&pref_priorities[a]));

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
    let t_init_elapsed = t_init.elapsed();

    let mut stagnant_rounds = 0;
    let early_exit_round = if challenge.num_hyperedges < 20_000 { 90 } else { 70 };
    let max_stagnant_rounds = if challenge.num_hyperedges < 20_000 { 30 } else { 20 };

    let t_refine1 = Instant::now();
    let mut t_gpu_kernels = 0u128;
    let mut t_gpu_sort = 0u128;
    let mut t_cpu_sort = 0u128;
    let mut t_execute = 0u128;
    let mut actual_rounds = 0usize;
    let mut gpu_sort_count = 0usize;
    let mut cpu_sort_count = 0usize;

    for round in 0..refinement_rounds {
        actual_rounds = round + 1;
        let zero = vec![0i32];
        let mut d_num_valid_moves = stream.memcpy_stod(&zero)?;

        let t0 = Instant::now();
        unsafe {
            stream.launch_builder(&precompute_edge_flags_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .launch(hedge_cfg.clone())?;
        }

        unsafe {
            stream.launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_move_parts)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .arg(&mut d_global_edge_flags)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;
        t_gpu_kernels += t0.elapsed().as_micros();

        let num_valid_moves = stream.memcpy_dtov(&d_num_valid_moves)?[0];
        if num_valid_moves == 0 {
            break;
        }

        let t2 = Instant::now();
        let move_priorities_vec = stream.memcpy_dtov(&d_move_priorities)?;
        let max_priority = move_priorities_vec.iter().copied().max().unwrap_or(0);

        let num_passes = if max_priority == 0 {
            0
        } else if max_priority < 256 {
            1
        } else if max_priority < 65536 {
            2
        } else if max_priority < 16777216 {
            3
        } else {
            4
        };

        let use_gpu_sort = num_passes > 0 && num_passes <= 3;

        let (d_sorted_nodes_ref, d_sorted_parts_ref): (&cudarc::driver::CudaSlice<i32>, &cudarc::driver::CudaSlice<i32>);
        let d_sorted_nodes_tmp: cudarc::driver::CudaSlice<i32>;
        let d_sorted_parts_tmp: cudarc::driver::CudaSlice<i32>;
        let num_to_process: i32;

        if use_gpu_sort {
            unsafe {
                stream.launch_builder(&invert_keys_kernel)
                    .arg(&(n as i32))
                    .arg(&max_priority)
                    .arg(&d_move_priorities)
                    .arg(&mut d_sort_keys_a)
                    .launch(cfg.clone())?;
                
                stream.launch_builder(&init_indices_kernel)
                    .arg(&(n as i32))
                    .arg(&mut d_sort_vals_a)
                    .launch(cfg.clone())?;
            }
            
            for pass in 0..num_passes {
                let shift = pass * 8;
                
                stream.memset_zeros(&mut d_ready_flag)?;
                
                if pass % 2 == 0 {
                    unsafe {
                        stream.launch_builder(&radix_hist_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_a)
                            .arg(&shift)
                            .arg(&mut d_chunk_histograms)
                            .launch(radix_cfg.clone())?;
                        
                        stream.launch_builder(&radix_prefix_scatter_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_a)
                            .arg(&d_sort_vals_a)
                            .arg(&shift)
                            .arg(&d_chunk_histograms)
                            .arg(&mut d_chunk_offsets)
                            .arg(&mut d_sort_keys_b)
                            .arg(&mut d_sort_vals_b)
                            .arg(&mut d_ready_flag)
                            .launch(radix_cfg.clone())?;
                    }
                } else {
                    unsafe {
                        stream.launch_builder(&radix_hist_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_b)
                            .arg(&shift)
                            .arg(&mut d_chunk_histograms)
                            .launch(radix_cfg.clone())?;
                        
                        stream.launch_builder(&radix_prefix_scatter_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_b)
                            .arg(&d_sort_vals_b)
                            .arg(&shift)
                            .arg(&d_chunk_histograms)
                            .arg(&mut d_chunk_offsets)
                            .arg(&mut d_sort_keys_a)
                            .arg(&mut d_sort_vals_a)
                            .arg(&mut d_ready_flag)
                            .launch(radix_cfg.clone())?;
                    }
                }
            }

            let sorted_vals = if num_passes % 2 == 0 { &d_sort_vals_a } else { &d_sort_vals_b };

            unsafe {
                stream.launch_builder(&gather_sorted_kernel)
                    .arg(&(n as i32))
                    .arg(sorted_vals)
                    .arg(&d_move_parts)
                    .arg(&mut d_sorted_move_parts)
                    .launch(cfg.clone())?;
            }
            stream.synchronize()?;

            d_sorted_nodes_ref = sorted_vals;
            d_sorted_parts_ref = &d_sorted_move_parts;
            num_to_process = n as i32;
            t_gpu_sort += t2.elapsed().as_micros();
            gpu_sort_count += 1;
        } else {
            let t_cpu = Instant::now();
            let move_parts = stream.memcpy_dtov(&d_move_parts)?;

            valid_indices.clear();
            valid_indices.extend(
                move_priorities_vec
                    .iter()
                    .enumerate()
                    .filter(|(_, &priority)| priority > 0)
                    .map(|(i, _)| i),
            );

            if valid_indices.is_empty() {
                break;
            }

            valid_indices.sort_unstable_by(|&a, &b| move_priorities_vec[b].cmp(&move_priorities_vec[a]));

            sorted_move_nodes.clear();
            sorted_move_parts_cpu.clear();
            sorted_move_nodes.extend(valid_indices.iter().map(|&i| i as i32));
            sorted_move_parts_cpu.extend(valid_indices.iter().map(|&i| move_parts[i]));

            d_sorted_nodes_tmp = stream.memcpy_stod(&sorted_move_nodes)?;
            d_sorted_parts_tmp = stream.memcpy_stod(&sorted_move_parts_cpu)?;
            d_sorted_nodes_ref = &d_sorted_nodes_tmp;
            d_sorted_parts_ref = &d_sorted_parts_tmp;
            num_to_process = sorted_move_nodes.len() as i32;
            t_cpu_sort += t_cpu.elapsed().as_micros();
            cpu_sort_count += 1;
        }

        let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;

        let t4 = Instant::now();
        unsafe {
            stream.launch_builder(&execute_moves_kernel)
                .arg(&num_to_process)
                .arg(d_sorted_nodes_ref)
                .arg(d_sorted_parts_ref)
                .arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&mut d_moves_executed)
                .launch(one_thread_cfg.clone())?;
        }
        stream.synchronize()?;
        t_execute += t4.elapsed().as_micros();

        let moves_executed = stream.memcpy_dtov(&d_moves_executed)?[0];
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

    let t_refine1_elapsed = t_refine1.elapsed();

    let t_balance = Instant::now();
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
    let t_balance_elapsed = t_balance.elapsed();

    let t_refine2 = Instant::now();
    for _ in 0..24 {
        let zero = vec![0i32];
        let mut d_num_valid_moves = stream.memcpy_stod(&zero)?;

        unsafe {
            stream.launch_builder(&precompute_edge_flags_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .launch(hedge_cfg.clone())?;
        }

        unsafe {
            stream.launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_move_parts)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .arg(&mut d_global_edge_flags)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let num_valid_moves = stream.memcpy_dtov(&d_num_valid_moves)?[0];
        if num_valid_moves == 0 {
            break;
        }

        let move_priorities_vec2 = stream.memcpy_dtov(&d_move_priorities)?;
        let max_priority2 = move_priorities_vec2.iter().copied().max().unwrap_or(0);

        let num_passes2 = if max_priority2 == 0 {
            0
        } else if max_priority2 < 256 {
            1
        } else if max_priority2 < 65536 {
            2
        } else if max_priority2 < 16777216 {
            3
        } else {
            4
        };

        let use_gpu_sort = num_passes2 > 0 && num_passes2 <= 3;

        let d_sorted_nodes_ref2: &cudarc::driver::CudaSlice<i32>;
        let d_sorted_parts_ref2: &cudarc::driver::CudaSlice<i32>;
        let d_sorted_nodes_tmp2: cudarc::driver::CudaSlice<i32>;
        let d_sorted_parts_tmp2: cudarc::driver::CudaSlice<i32>;
        let num_to_process2: i32;

        if use_gpu_sort {
            unsafe {
                stream.launch_builder(&invert_keys_kernel)
                    .arg(&(n as i32))
                    .arg(&max_priority2)
                    .arg(&d_move_priorities)
                    .arg(&mut d_sort_keys_a)
                    .launch(cfg.clone())?;
                
                stream.launch_builder(&init_indices_kernel)
                    .arg(&(n as i32))
                    .arg(&mut d_sort_vals_a)
                    .launch(cfg.clone())?;
            }
            
            for pass in 0..num_passes2 {
                let shift = pass * 8;
                
                stream.memset_zeros(&mut d_ready_flag)?;
                
                if pass % 2 == 0 {
                    unsafe {
                        stream.launch_builder(&radix_hist_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_a)
                            .arg(&shift)
                            .arg(&mut d_chunk_histograms)
                            .launch(radix_cfg.clone())?;
                        
                        stream.launch_builder(&radix_prefix_scatter_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_a)
                            .arg(&d_sort_vals_a)
                            .arg(&shift)
                            .arg(&d_chunk_histograms)
                            .arg(&mut d_chunk_offsets)
                            .arg(&mut d_sort_keys_b)
                            .arg(&mut d_sort_vals_b)
                            .arg(&mut d_ready_flag)
                            .launch(radix_cfg.clone())?;
                    }
                } else {
                    unsafe {
                        stream.launch_builder(&radix_hist_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_b)
                            .arg(&shift)
                            .arg(&mut d_chunk_histograms)
                            .launch(radix_cfg.clone())?;
                        
                        stream.launch_builder(&radix_prefix_scatter_kernel)
                            .arg(&(n as i32))
                            .arg(&num_chunks)
                            .arg(&d_sort_keys_b)
                            .arg(&d_sort_vals_b)
                            .arg(&shift)
                            .arg(&d_chunk_histograms)
                            .arg(&mut d_chunk_offsets)
                            .arg(&mut d_sort_keys_a)
                            .arg(&mut d_sort_vals_a)
                            .arg(&mut d_ready_flag)
                            .launch(radix_cfg.clone())?;
                    }
                }
            }
            
            let sorted_vals2 = if num_passes2 % 2 == 0 { &d_sort_vals_a } else { &d_sort_vals_b };
            
            unsafe {
                stream.launch_builder(&gather_sorted_kernel)
                    .arg(&(n as i32))
                    .arg(sorted_vals2)
                    .arg(&d_move_parts)
                    .arg(&mut d_sorted_move_parts)
                    .launch(cfg.clone())?;
            }
            stream.synchronize()?;
            
            d_sorted_nodes_ref2 = sorted_vals2;
            d_sorted_parts_ref2 = &d_sorted_move_parts;
            num_to_process2 = n as i32;
        } else {
            let move_parts = stream.memcpy_dtov(&d_move_parts)?;
            
            valid_indices.clear();
            valid_indices.extend(
                move_priorities_vec2
                    .iter()
                    .enumerate()
                    .filter(|(_, &priority)| priority > 0)
                    .map(|(i, _)| i),
            );
            
            if valid_indices.is_empty() {
                break;
            }

            valid_indices.sort_unstable_by(|&a, &b| move_priorities_vec2[b].cmp(&move_priorities_vec2[a]));
            
            sorted_move_nodes.clear();
            sorted_move_parts_cpu.clear();
            sorted_move_nodes.extend(valid_indices.iter().map(|&i| i as i32));
            sorted_move_parts_cpu.extend(valid_indices.iter().map(|&i| move_parts[i]));
            
            d_sorted_nodes_tmp2 = stream.memcpy_stod(&sorted_move_nodes)?;
            d_sorted_parts_tmp2 = stream.memcpy_stod(&sorted_move_parts_cpu)?;
            d_sorted_nodes_ref2 = &d_sorted_nodes_tmp2;
            d_sorted_parts_ref2 = &d_sorted_parts_tmp2;
            num_to_process2 = sorted_move_nodes.len() as i32;
        }
        
        let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;
        
        unsafe {
            stream.launch_builder(&execute_moves_kernel)
                .arg(&num_to_process2)
                .arg(d_sorted_nodes_ref2)
                .arg(d_sorted_parts_ref2)
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
    let t_refine2_elapsed = t_refine2.elapsed();
    
    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();
    
    save_solution(&Solution { partition: partition_u32 })?;
    
    let total_elapsed = total_start.elapsed();
    println!("=== FULL PROFILING ===");
    println!("load_function:     {:.2}ms", t_load_elapsed.as_micros() as f64 / 1000.0);
    println!("alloc_zeros:       {:.2}ms", t_alloc_elapsed.as_micros() as f64 / 1000.0);
    println!("init (cluster+assign): {:.2}ms", t_init_elapsed.as_micros() as f64 / 1000.0);
    println!("refine1 ({} rounds): {:.2}ms", actual_rounds, t_refine1_elapsed.as_micros() as f64 / 1000.0);
    println!("  - GPU kernels:   {:.2}ms", t_gpu_kernels as f64 / 1000.0);
    println!("  - GPU sort:      {:.2}ms ({} times)", t_gpu_sort as f64 / 1000.0, gpu_sort_count);
    println!("  - CPU sort:      {:.2}ms ({} times)", t_cpu_sort as f64 / 1000.0, cpu_sort_count);
    println!("  - execute_moves: {:.2}ms", t_execute as f64 / 1000.0);
    println!("balance:           {:.2}ms", t_balance_elapsed.as_micros() as f64 / 1000.0);
    println!("refine2 (24 rounds): {:.2}ms", t_refine2_elapsed.as_micros() as f64 / 1000.0);
    println!("TOTAL:             {:.2}ms", total_elapsed.as_micros() as f64 / 1000.0);
    println!(">>> solve_challenge END");
    
    Ok(())
}
