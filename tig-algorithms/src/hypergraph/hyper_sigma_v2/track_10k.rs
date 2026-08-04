use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg}, 
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {    
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_10k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_10k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_10k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_10k")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves_optimized_10k")?;
    let execute_moves_kernel = module.load_function("execute_refinement_moves_10k")?;
    let balance_kernel = module.load_function("balance_final_10k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_10k")?;
    let perturb_kernel = module.load_function("perturb_solution_10k")?;
    let perturb_guided_kernel = module.load_function("perturb_guided_10k")?;
    let perturb_hubs_kernel = module.load_function("perturb_hubs_10k")?;
    let perturb_ruin_recreate_kernel = module.load_function("perturb_ruin_recreate_10k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_10k")?;
    let perturb_path_relink_kernel = module.load_function("perturb_path_relink_10k")?;
    let compute_he_moves_kernel = module.load_function("compute_hyperedge_centric_moves_10k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_10k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_10k")?;

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

    let hedge_cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_hyperedges as u32 + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let init_restart_id = hyperparameters
        .as_ref()
        .and_then(|p| p.get("init_restart_id").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as i32)
        .unwrap_or(1);
    let init_random_seed = u32::from_le_bytes([
        challenge.seed[0],
        challenge.seed[1],
        challenge.seed[2],
        challenge.seed[3],
    ]);

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

    let grid_x_calc = (challenge.num_nodes as u32 + block_size - 1) / block_size;
    let mut d_num_valid_moves = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;

    let mut d_edge_flags_all = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
    let mut d_edge_flags_double = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;

    let num_parts_usize = challenge.num_parts as usize;
    let is_sparse = (challenge.num_nodes as usize) > 4 * (challenge.num_hyperedges as usize + 1);

    let effort = hyperparameters
        .as_ref()
        .and_then(|p| p.get("effort").and_then(|v| v.as_i64()))
        .unwrap_or(3);

    let (base_refine, base_ils, base_ils_quick, base_polish, base_post_balance) = match effort {
        5 => (12000, 6, 70, 300, 0),
        4 => (10000, 5, 60, 200, 0),
        3 => (8000, 5, 60, 150, 0),
        2 => (5000, 5, 60, 100, 0),
        1 => (4000, 3, 25, 40, 0),
        0 => (3000, 3, 20, 30, 0),
        _ => (6000, 5, 50, 150, 0),
    };

    let refinement_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(50, 50_000) as usize)
        .unwrap_or(base_refine);

    let ils_iterations = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ils_iterations").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 500) as usize)
        .unwrap_or(base_ils);

    let ils_quick_refine = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ils_quick_refine").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(10, 500) as usize)
        .unwrap_or(base_ils_quick);

    let post_ils_polish = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_ils_polish").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(20, 500) as usize)
        .unwrap_or(base_polish);

    let tabu_tenure: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("tabu_tenure").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 30) as usize)
        .unwrap_or(10);

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

    let neg_gain_thresh: i32 = 3;
    let scan_limit_swap = 32usize;
    let scan_limit_cycle = 8usize;

    let swap_buf_size = 4 * challenge.num_nodes as usize;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;

    let mut part_to_part: Vec<Vec<(usize, i32)>> = vec![vec![]; num_parts_usize * num_parts_usize];
    let mut swap_gains_host: Vec<i32> = vec![0i32; swap_buf_size];
    let mut partition_host_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut nodes_in_part_host: Vec<i32> = vec![0i32; num_parts_usize];
    let mut move_keys_host: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let zero_counter_1 = [0i32; 1];
    let zero_counter_grid = vec![0i32; ((challenge.num_nodes as u32 + block_size - 1) / block_size) as usize];
    let mut partition_host_mirror: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut nodes_in_part_mirror: Vec<i32> = vec![0i32; num_parts_usize];
    let mut accepted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut accepted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);    
    let mut d_accepted_move_nodes = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_accepted_move_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_hedge_moves = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize * 4)?;
    let mut d_hedge_choice = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut node_has_move = vec![false; challenge.num_nodes as usize];

    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut tgt_used: Vec<usize> = vec![0; num_parts_usize];
    let mut tgt_quota: Vec<usize> = vec![0; num_parts_usize];

    let mut stagnant_rounds = 0;
    let max_stagnant_rounds = 30;
    
    let mut node_tabu_until: Vec<i32> = vec![0; challenge.num_nodes as usize];
    let mut d_node_tabu_until = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut global_round = 0i32;

    macro_rules! reset_move_counters {
        () => {{
            stream.memcpy_htod(&zero_counter_grid, &mut d_num_valid_moves)?;
            stream.memcpy_htod(&zero_counter_1, &mut d_moves_executed)?;
        }};
    }

    macro_rules! refresh_host_mirrors_from_device {
        () => {{
            stream.memcpy_dtoh(&d_partition, &mut partition_host_mirror)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_mirror)?;
        }};
    }

    macro_rules! replay_initial_assignment_host {
        ($sorted_nodes:expr, $sorted_parts:expr) => {{
            partition_host_mirror.fill(0);
            nodes_in_part_mirror.fill(0);
            for i in 0..challenge.num_nodes as usize {
                let node_i32 = $sorted_nodes[i];
                let preferred_part = $sorted_parts[i];
                if node_i32 >= 0 && preferred_part >= 0 {
                    let node = node_i32 as usize;
                    if node < challenge.num_nodes as usize && (preferred_part as usize) < num_parts_usize {
                        let start_part = if i < num_parts_usize {
                            i as i32
                        } else {
                            preferred_part
                        };

                        let mut assigned = false;
                        for attempt in 0..num_parts_usize {
                            let try_part = ((start_part as usize) + attempt) % num_parts_usize;
                            if nodes_in_part_mirror[try_part] < challenge.max_part_size as i32 {
                                partition_host_mirror[node] = try_part as i32;
                                nodes_in_part_mirror[try_part] += 1;
                                assigned = true;
                                break;
                            }
                        }

                        if !assigned {
                            let fallback_part = node % num_parts_usize;
                            partition_host_mirror[node] = fallback_part as i32;
                            nodes_in_part_mirror[fallback_part] += 1;
                        }
                    }
                }
            }
        }};
    }

    macro_rules! replay_execute_moves_host {
        ($sorted_nodes:expr, $sorted_parts:expr) => {{
            accepted_move_nodes.clear();
            accepted_move_parts.clear();

            let mut host_moves_executed = 0i32;
            for i in 0..$sorted_nodes.len() {
                let node_i32 = $sorted_nodes[i];
                let target_part_i32 = $sorted_parts[i];

                if node_i32 >= 0 && target_part_i32 >= 0 {
                    let node = node_i32 as usize;
                    let target_part = target_part_i32 as usize;

                    if node < partition_host_mirror.len() && target_part < num_parts_usize {
                        let current_part = partition_host_mirror[node];
                        if current_part >= 0 {
                            let current_part_usize = current_part as usize;
                            if current_part_usize < num_parts_usize
                                && nodes_in_part_mirror[target_part] < challenge.max_part_size as i32
                                && nodes_in_part_mirror[current_part_usize] > 1
                                && partition_host_mirror[node] == current_part
                            {
                                partition_host_mirror[node] = target_part as i32;
                                nodes_in_part_mirror[current_part_usize] -= 1;
                                nodes_in_part_mirror[target_part] += 1;
                                accepted_move_nodes.push(node_i32);
                                accepted_move_parts.push(target_part_i32);
                                host_moves_executed += 1;
                            }
                        }
                    }
                }
            }

            if host_moves_executed > 0 {
                let accepted_len = accepted_move_nodes.len();
                stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
                stream.memcpy_htod(accepted_move_nodes.as_slice(), &mut d_accepted_move_nodes)?;
                stream.memcpy_htod(accepted_move_parts.as_slice(), &mut d_accepted_move_parts)?;
                unsafe {
                    stream
                        .launch_builder(&execute_moves_kernel)
                        .arg(&(accepted_len as i32))
                        .arg(&d_accepted_move_nodes)
                        .arg(&d_accepted_move_parts)
                        .arg(&(challenge.max_part_size as i32))
                        .arg(&mut d_partition)
                        .arg(&mut d_nodes_in_part)
                        .arg(&mut d_moves_executed)
                        .launch(LaunchConfig {
                            grid_dim: (
                                (accepted_len as u32 + block_size - 1) / block_size,
                                1,
                                1,
                            ),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
            }

            host_moves_executed
        }};
    }

    macro_rules! do_hyperedge_centric_phase {
        () => {{
            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
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
                stream
                    .launch_builder(&compute_he_moves_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&challenge.d_node_offsets)
                    .arg(&mut d_hedge_moves)
                    .launch(hedge_cfg.clone())?;
            }
            let hedge_moves_host = stream.memcpy_dtov(&d_hedge_moves)?;
            valid_moves.clear();
            for (h_idx, chunk) in hedge_moves_host.chunks_exact(4).enumerate() {
                let mut block_size = 0;
                for &m in chunk {
                    if m > 0 { block_size += 1; }
                }
                if block_size > 0 {
                    let prio = 32000 + block_size;
                    for &m in chunk {
                        if m > 0 {
                            let node = ((m >> 6) - 1) as usize;
                            let tgt = (m & 63) as i32;
                            if !node_has_move[node] {
                                node_has_move[node] = true;
                                valid_moves.push((node, (prio << 16) | ((h_idx as i32 & 0x3FF) << 6) | tgt));
                            }
                        }
                    }
                }
            }
            for &(node, _) in valid_moves.iter() {
                node_has_move[node] = false;
            }
            if !valid_moves.is_empty() {
                let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
                valid_moves.sort_unstable_by(cmp);
                nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
                tgt_used.fill(0);
                for p in 0..num_parts_usize {
                    let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
                    tgt_quota[p] = std::cmp::max(1, free + 4);
                }
                sorted_move_nodes.clear();
                sorted_move_parts.clear();
                for &(node, key) in valid_moves.iter() {
                    let tgt = (key & 63) as usize;
                    if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                        tgt_used[tgt] += 1;
                        sorted_move_nodes.push(node as i32);
                        sorted_move_parts.push(tgt as i32);
                    }
                }
                let me_he = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
                if me_he > 0 {
                    for &node in sorted_move_nodes.iter().take(me_he as usize) {
                        node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
                    }
                }
                me_he
            } else {
                0i32
            }
        }};
    }

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
            .arg(&init_restart_id)
            .arg(&init_random_seed)
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_priorities)
            .launch(cfg.clone())?;
    }

    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..challenge.num_nodes as usize).collect();
    indices.sort_unstable_by(|&a, &b| {
        pref_priorities[b].cmp(&pref_priorities[a]).then_with(|| a.cmp(&b))
    });

    let sorted_nodes: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let sorted_parts: Vec<i32> = indices.iter().map(|&i| pref_parts[i]).collect();

    let d_sorted_nodes = stream.memcpy_stod(&sorted_nodes)?;

    replay_initial_assignment_host!(sorted_nodes, sorted_parts);
    let assigned_parts: Vec<i32> = sorted_nodes
        .iter()
        .map(|&node| partition_host_mirror[node as usize])
        .collect();
    let d_assigned_parts = stream.memcpy_stod(&assigned_parts)?;
    stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;

    unsafe {
        stream
            .launch_builder(&execute_assignments_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(challenge.max_part_size as i32))
            .arg(&d_sorted_nodes)
            .arg(&d_assigned_parts)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(cfg.clone())?;
    }

    for round in 0..refinement_rounds {
        global_round += 1;
        reset_move_counters!();

        unsafe {
            stream
                .launch_builder(&precompute_edge_flags_kernel)
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
            stream
                .launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .launch(cfg.clone())?;
        }
        let nvm_vec = stream.memcpy_dtov(&d_num_valid_moves)?;
        let num_valid_moves: i32 = nvm_vec.iter().sum();
        if num_valid_moves == 0 {
            break;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;

        valid_moves.clear();
        let max_gain = move_keys_host.iter().filter(|&&k| k > 0).map(|&k| (k >> 16) - 1000).max().unwrap_or(0);
        let aspiration_threshold = std::cmp::max(1, (max_gain * 3) / 4);

        for (node, &key) in move_keys_host.iter().enumerate() {
            if key > 0 {
                let gain = (key >> 16) - 1000;
                if node_tabu_until[node] <= global_round || gain >= aspiration_threshold {
                    valid_moves.push((node, key));
                }
            }
        }

        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

        let mut k_base = valid_moves.len();
        
        let adaptive_limit = if round < 50 {
            move_limit / 2
        } else if round < 200 {
            move_limit
        } else {
            move_limit / 3
        };
        
        if k_base > adaptive_limit {
            k_base = adaptive_limit;
        }

        let extra_window = 16384usize;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
        let slack = if round < 64 { 8usize } else if round < 256 { 4usize } else { 2usize };

        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        for &(node, key) in valid_moves[..k_cand].iter() {
            if sorted_move_nodes.len() >= k_base {
                break;
            }
            let tgt = (key & 63) as usize;
            if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                tgt_used[tgt] += 1;
                sorted_move_nodes.push(node as i32);
                sorted_move_parts.push(tgt as i32);
            }
        }

        if sorted_move_nodes.is_empty() {
            let take = std::cmp::min(k_base, k_cand);
            sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
            sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
        }

        let mut moves_executed = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);

        if moves_executed == 0 && k_cand > k_base {
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            let tail = &valid_moves[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            sorted_move_nodes.extend(tail.iter().take(take).map(|(n, _)| *n as i32));
            sorted_move_parts.extend(tail.iter().take(take).map(|(_, key)| (key & 63) as i32));

            moves_executed = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
        }

        if moves_executed > 0 {
            for &node in sorted_move_nodes.iter().take(moves_executed as usize) {
                node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
            }
        }

        if moves_executed == 0 {
            let he_moves = do_hyperedge_centric_phase!();
            if he_moves > 0 {
                stagnant_rounds = 0;
                continue;
            }

            stagnant_rounds += 1;
            
            if stagnant_rounds >= 5 && round < refinement_rounds.saturating_sub(50) {
                let mini_seed = 987654321u64 + (round as u64) * 123456789u64;
                unsafe {
                    stream
                        .launch_builder(&perturb_kernel)
                        .arg(&(challenge.num_nodes as i32))
                        .arg(&(challenge.num_parts as i32))
                        .arg(&(challenge.max_part_size as i32))
                        .arg(&1i32)
                        .arg(&mut d_partition)
                        .arg(&mut d_nodes_in_part)
                        .arg(&mini_seed)
                        .launch(one_thread_cfg.clone())?;
                }
                refresh_host_mirrors_from_device!();
                stagnant_rounds = 0;
            } else if stagnant_rounds > max_stagnant_rounds {
                break;
            }
        } else {
            stagnant_rounds = 0;
        }
    }

    let crossover_kernel = module.load_function("crossover_partitions_10k")?;
    let mut d_connectivity = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;

    macro_rules! do_swap_phase {
        ($d_partition:expr, $d_nodes_in_part:expr,
         $d_edge_flags_all:expr, $d_edge_flags_double:expr,
         $d_swap_gains:expr, $swap_gains_host:expr,
         $partition_host_swap:expr, $partition_mut_swap:expr,
         $part_to_part:expr, $used_ba_buf:expr,
         $max_rounds:expr, $ngt:expr, $scan_lim:expr, $scan_lim_cyc:expr) => {{
            let num_nodes_i = challenge.num_nodes as i32;
            let num_parts_i = challenge.num_parts as i32;
            let np = num_parts_usize;
            let mut prev_swap_count = usize::MAX;
            let mut stagnant = 0usize;
            let mut total_swaps = 0usize;
            $partition_host_swap.copy_from_slice(&partition_host_mirror);
            for _swap_round in 0..$max_rounds {
                global_round += 1;
                stream.memcpy_htod(&node_tabu_until, &mut d_node_tabu_until)?;
                unsafe {
                    stream
                        .launch_builder(&precompute_edge_flags_kernel)
                        .arg(&(challenge.num_hyperedges as i32))
                        .arg(&num_nodes_i)
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&mut *$d_partition)
                        .arg(&mut *$d_edge_flags_all)
                        .arg(&mut *$d_edge_flags_double)
                        .launch(hedge_cfg.clone())?;
                }
                unsafe {
                    stream
                        .launch_builder(&compute_swap_gains_kernel)
                        .arg(&num_nodes_i)
                        .arg(&num_parts_i)
                        .arg(&$ngt)
                        .arg(&global_round)
                        .arg(&d_node_tabu_until)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&mut *$d_partition)
                        .arg(&mut *$d_edge_flags_all)
                        .arg(&mut *$d_edge_flags_double)
                        .arg(&mut *$d_swap_gains)
                        .launch(cfg.clone())?;
                }
                stream.memcpy_dtoh(&mut *$d_swap_gains, $swap_gains_host)?;
                let num_nodes = num_nodes_i as usize;

                for v in $part_to_part.iter_mut() { v.clear(); }
                for node in 0..num_nodes {
                    let src = $partition_host_swap[node] as usize;
                    if src >= np { continue; }
                    for k in 0..4usize {
                        let val = $swap_gains_host[node * 4 + k];
                        if val == 0 { continue; }
                        let tgt = (val & 0xFFFF) as usize;
                        let gain = ((val >> 16) as i16) as i32;
                        if tgt < np && tgt != src {
                            $part_to_part[src * np + tgt].push((node, gain));
                        }
                    }
                }

                $partition_mut_swap.copy_from_slice($partition_host_swap);
                let mut swap_count = 0usize;

                for a in 0..np {
                    for b in (a + 1)..np {
                        let idx_ab = a * np + b;
                        let idx_ba = b * np + a;
                        if $part_to_part[idx_ab].is_empty() || $part_to_part[idx_ba].is_empty() { continue; }
                        $part_to_part[idx_ab].sort_unstable_by(|x, y| y.1.cmp(&x.1));
                        $part_to_part[idx_ba].sort_unstable_by(|x, y| y.1.cmp(&x.1));
                        let lab_len = $part_to_part[idx_ab].len();
                        let lba_len = $part_to_part[idx_ba].len();
                        $used_ba_buf.clear();
                        $used_ba_buf.resize(lba_len, false);
                        for i in 0..lab_len {
                            let (node_a, gain_a) = $part_to_part[idx_ab][i];
                            if $partition_mut_swap[node_a] as usize != a { continue; }
                            let mut best_combined = 0i32;
                            let mut best_j = usize::MAX;
                            let sl = std::cmp::min(lba_len, $scan_lim);
                            for j in 0..sl {
                                if $used_ba_buf[j] { continue; }
                                let (node_b, gain_b) = $part_to_part[idx_ba][j];
                                if $partition_mut_swap[node_b] as usize != b { continue; }
                                let combined = gain_a + gain_b;
                                if combined > 0 {
                                    best_combined = combined;
                                    best_j = j;
                                }
                                break;
                            }
                            if best_j < lba_len && best_combined > 0 {
                                let (node_b, _) = $part_to_part[idx_ba][best_j];
                                $partition_mut_swap[node_a] = b as i32;
                                $partition_mut_swap[node_b] = a as i32;
                                $used_ba_buf[best_j] = true;
                                node_tabu_until[node_a] = global_round + tabu_tenure as i32;
                                node_tabu_until[node_b] = global_round + tabu_tenure as i32;
                                swap_count += 1;
                            }
                        }
                    }
                }

                let cyc_scan = $scan_lim_cyc;
                if cyc_scan > 0 {
                    for a in 0..np {
                        for b in 0..np {
                            if b == a { continue; }
                            let idx_ab = a * np + b;
                            if $part_to_part[idx_ab].is_empty() { continue; }
                            for c in 0..np {
                                if c == a || c == b { continue; }
                                let idx_bc = b * np + c;
                                let idx_ca = c * np + a;
                                if $part_to_part[idx_bc].is_empty() || $part_to_part[idx_ca].is_empty() { continue; }
                                let sl_ab = std::cmp::min($part_to_part[idx_ab].len(), cyc_scan);
                                let sl_bc = std::cmp::min($part_to_part[idx_bc].len(), cyc_scan);
                                let sl_ca = std::cmp::min($part_to_part[idx_ca].len(), cyc_scan);
                                'outer: for i in 0..sl_ab {
                                    let (node_ab, gain_ab) = $part_to_part[idx_ab][i];
                                    if $partition_mut_swap[node_ab] as usize != a { continue; }
                                    if gain_ab + $part_to_part[idx_bc][0].1 + $part_to_part[idx_ca][0].1 <= 0 { break; }
                                    for j in 0..sl_bc {
                                        let (node_bc, gain_bc) = $part_to_part[idx_bc][j];
                                        if $partition_mut_swap[node_bc] as usize != b { continue; }
                                        if node_bc == node_ab { continue; }
                                        if gain_ab + gain_bc + $part_to_part[idx_ca][0].1 <= 0 { break; }
                                        for k in 0..sl_ca {
                                            let (node_ca, gain_ca) = $part_to_part[idx_ca][k];
                                            if $partition_mut_swap[node_ca] as usize != c { continue; }
                                            if node_ca == node_ab || node_ca == node_bc { continue; }
                                            if gain_ab + gain_bc + gain_ca > 0 {
                                                $partition_mut_swap[node_ab] = b as i32;
                                                $partition_mut_swap[node_bc] = c as i32;
                                                $partition_mut_swap[node_ca] = a as i32;
                                                node_tabu_until[node_ab] = global_round + tabu_tenure as i32;
                                                node_tabu_until[node_bc] = global_round + tabu_tenure as i32;
                                                node_tabu_until[node_ca] = global_round + tabu_tenure as i32;
                                                swap_count += 1;
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let cyc4_scan = 2usize;
                if cyc4_scan > 0 {
                    for a in 0..np {
                        for b in 0..np {
                            if b == a { continue; }
                            let idx_ab = a * np + b;
                            if $part_to_part[idx_ab].is_empty() { continue; }
                            for c in 0..np {
                                if c == a || c == b { continue; }
                                let idx_bc = b * np + c;
                                if $part_to_part[idx_bc].is_empty() { continue; }
                                for d in 0..np {
                                    if d == a || d == b || d == c { continue; }
                                    let idx_cd = c * np + d;
                                    let idx_da = d * np + a;
                                    if $part_to_part[idx_cd].is_empty() || $part_to_part[idx_da].is_empty() { continue; }
                                    
                                    let sl_ab = std::cmp::min($part_to_part[idx_ab].len(), cyc4_scan);
                                    let sl_bc = std::cmp::min($part_to_part[idx_bc].len(), cyc4_scan);
                                    let sl_cd = std::cmp::min($part_to_part[idx_cd].len(), cyc4_scan);
                                    let sl_da = std::cmp::min($part_to_part[idx_da].len(), cyc4_scan);

                                    'outer4: for i in 0..sl_ab {
                                        let (node_ab, gain_ab) = $part_to_part[idx_ab][i];
                                        if $partition_mut_swap[node_ab] as usize != a { continue; }
                                        if gain_ab + $part_to_part[idx_bc][0].1 + $part_to_part[idx_cd][0].1 + $part_to_part[idx_da][0].1 <= 0 { break; }
                                        for j in 0..sl_bc {
                                            let (node_bc, gain_bc) = $part_to_part[idx_bc][j];
                                            if $partition_mut_swap[node_bc] as usize != b { continue; }
                                            if node_bc == node_ab { continue; }
                                            if gain_ab + gain_bc + $part_to_part[idx_cd][0].1 + $part_to_part[idx_da][0].1 <= 0 { break; }
                                            for k in 0..sl_cd {
                                                let (node_cd, gain_cd) = $part_to_part[idx_cd][k];
                                                if $partition_mut_swap[node_cd] as usize != c { continue; }
                                                if node_cd == node_ab || node_cd == node_bc { continue; }
                                                if gain_ab + gain_bc + gain_cd + $part_to_part[idx_da][0].1 <= 0 { break; }
                                                for m in 0..sl_da {
                                                    let (node_da, gain_da) = $part_to_part[idx_da][m];
                                                    if $partition_mut_swap[node_da] as usize != d { continue; }
                                                    if node_da == node_ab || node_da == node_bc || node_da == node_cd { continue; }
                                                    if gain_ab + gain_bc + gain_cd + gain_da > 0 {
                                                        $partition_mut_swap[node_ab] = b as i32;
                                                        $partition_mut_swap[node_bc] = c as i32;
                                                        $partition_mut_swap[node_cd] = d as i32;
                                                        $partition_mut_swap[node_da] = a as i32;
                                                        node_tabu_until[node_ab] = global_round + tabu_tenure as i32;
                                                        node_tabu_until[node_bc] = global_round + tabu_tenure as i32;
                                                        node_tabu_until[node_cd] = global_round + tabu_tenure as i32;
                                                        node_tabu_until[node_da] = global_round + tabu_tenure as i32;
                                                        swap_count += 1;
                                                        break 'outer4;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if swap_count == 0 { break; }
                total_swaps += swap_count;
                if swap_count >= prev_swap_count {
                    stagnant += 1;
                    if stagnant >= 3 { break; }
                } else {
                    stagnant = 0;
                }
                prev_swap_count = swap_count;
                $partition_host_swap.copy_from_slice($partition_mut_swap);
                stream.memcpy_htod($partition_host_swap, &mut *$d_partition)?;
            }
            partition_host_mirror.copy_from_slice($partition_host_swap);
            anyhow::Ok(total_swaps)
        }};
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        100, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    for _post_swap_round in 0..30 {
        global_round += 1;
        reset_move_counters!();
        unsafe {
            stream
                .launch_builder(&precompute_edge_flags_kernel)
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
            stream
                .launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .launch(cfg.clone())?;
        }
        let nvm_vec = stream.memcpy_dtov(&d_num_valid_moves)?;
        let nvm: i32 = nvm_vec.iter().sum();
        if nvm == 0 { break; }
        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key > 0 { valid_moves.push((node, key)); }
        }
        if valid_moves.is_empty() { break; }
        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let k_base = std::cmp::min(valid_moves.len(), move_limit / 2);
        let k_cand = std::cmp::min(valid_moves.len(), k_base + 8192);
        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }
        nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free + 4);
        }
        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        for &(node, key) in valid_moves[..k_cand].iter() {
            if sorted_move_nodes.len() >= k_base { break; }
            let tgt = (key & 63) as usize;
            if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                tgt_used[tgt] += 1;
                sorted_move_nodes.push(node as i32);
                sorted_move_parts.push(tgt as i32);
            }
        }
        if sorted_move_nodes.is_empty() {
            let take = std::cmp::min(k_base, k_cand);
            sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
            sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
        }
        let me = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
        if me > 0 {
            for &node in sorted_move_nodes.iter().take(me as usize) {
                node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
            }
        }
        if me == 0 {
            if do_hyperedge_centric_phase!() == 0 {
                break;
            }
        }
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        50, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;


    unsafe {
        stream
            .launch_builder(&compute_connectivity_kernel)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&d_partition)
            .arg(&mut d_connectivity)
            .launch(hedge_cfg.clone())?;
    }
    
    let connectivity_vec = stream.memcpy_dtov(&d_connectivity)?;
    let mut best_connectivity: i32 = connectivity_vec.iter().sum();

    let mut best_partition_host = stream.memcpy_dtov(&d_partition)?;
    let mut best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;

    let num_high_hedges = std::cmp::min(500usize, challenge.num_hyperedges as usize);
    let mut conn_with_idx: Vec<(i32, i32)> = connectivity_vec.iter().enumerate().map(|(i, &c)| (c, i as i32)).collect();
    conn_with_idx.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let high_hedge_ids: Vec<i32> = conn_with_idx.iter().take(num_high_hedges).map(|&(_, id)| id).collect();
    let mut d_high_hedge_ids = stream.memcpy_stod(&high_hedge_ids)?;

    let mut pop_partitions: Vec<Vec<i32>> = vec![best_partition_host.clone()];
    let mut pop_connectivities: Vec<i32> = vec![best_connectivity];

    let elite_pool_size = std::cmp::max(2usize, ils_iterations);
    let mut elite_scores: Vec<i32> = vec![i32::MAX; elite_pool_size];
    let mut elite_flat_host: Vec<i32> =
        vec![0i32; elite_pool_size * challenge.num_nodes as usize];
    elite_scores[0] = best_connectivity;
    elite_flat_host[..challenge.num_nodes as usize].copy_from_slice(&best_partition_host);
    let mut elite_count: usize = 1;
    let mut d_elite_flat =
        stream.alloc_zeros::<i32>(elite_pool_size * challenge.num_nodes as usize)?;
    let mut use_consensus_next = false;

    let sa_initial_temp = (best_connectivity as f64) * 0.02;
    let sa_cooling = if ils_iterations > 1 { 0.5f64.powf(1.0 / (ils_iterations as f64)) } else { 0.5 };
    let mut sa_temp = sa_initial_temp;
    let mut current_connectivity = best_connectivity;
    let mut current_partition_host = best_partition_host.clone();
    let mut current_nodes_in_part_host = best_nodes_in_part_host.clone();

    for ils_iter in 0..ils_iterations {
        let d_partition_restored = stream.memcpy_stod(&current_partition_host)?;
        let d_nodes_in_part_restored = stream.memcpy_stod(&current_nodes_in_part_host)?;
        d_partition = d_partition_restored;
        d_nodes_in_part = d_nodes_in_part_restored;
        partition_host_mirror.copy_from_slice(&current_partition_host);
        nodes_in_part_mirror.copy_from_slice(&current_nodes_in_part_host);

        let seed = 123456789u64 + (ils_iter as u64) * 987654321u64;

        if use_consensus_next && elite_count > 1 {
            stream.memcpy_htod(&elite_flat_host, &mut d_elite_flat)?;

            let mut elite_order_host: Vec<i32> = (0..elite_count as i32).collect();
            elite_order_host.sort_unstable_by(|&a, &b| {
                elite_scores[a as usize]
                    .cmp(&elite_scores[b as usize])
                    .then_with(|| a.cmp(&b))
            });
            let d_elite_order = stream.memcpy_stod(&elite_order_host)?;

            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
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
                stream
                    .launch_builder(&choose_elite_per_hyperedge_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(elite_count as i32))
                    .arg(&d_elite_flat)
                    .arg(&d_elite_order)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&mut d_hedge_choice)
                    .launch(hedge_cfg.clone())?;
            }

            unsafe {
                stream
                    .launch_builder(&assign_from_elite_votes_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(elite_count as i32))
                    .arg(&d_elite_flat)
                    .arg(&d_hedge_choice)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_edge_flags_all)
                    .arg(&mut d_partition)
                    .launch(cfg.clone())?;
            }

            stream.memcpy_dtoh(&d_partition, &mut partition_host_mirror)?;
            nodes_in_part_mirror.fill(0);
            for &part in partition_host_mirror.iter() {
                if part >= 0 && (part as usize) < num_parts_usize {
                    nodes_in_part_mirror[part as usize] += 1;
                }
            }
            stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;

            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
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
                stream
                    .launch_builder(&balance_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&1i32)
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .launch(one_thread_cfg.clone())?;
            }
            refresh_host_mirrors_from_device!();
        } else {
            let relink_fraction = if ils_iter % 2 == 0 { 30i32 } else { 15i32 };
            let d_best_partition_dev = stream.memcpy_stod(&best_partition_host)?;
            unsafe {
                stream
                    .launch_builder(&perturb_path_relink_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&relink_fraction)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_best_partition_dev)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&seed)
                    .launch(one_thread_cfg.clone())?;
            }
            let guided_seed = seed ^ 0xDEADBEEF_CAFEBABE_u64;
            unsafe {
                stream
                    .launch_builder(&perturb_guided_kernel)
                    .arg(&(num_high_hedges as i32))
                    .arg(&d_high_hedge_ids)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&guided_seed)
                    .launch(one_thread_cfg.clone())?;
            }
            let hubs_seed = seed ^ 0x123456789ABCDEF0_u64;
            unsafe {
                stream
                    .launch_builder(&perturb_hubs_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&hubs_seed)
                    .launch(one_thread_cfg.clone())?;
            }
            let rr_seed = seed ^ 0x5A5A5A5A5A5A5A5A_u64;
            unsafe {
                stream
                    .launch_builder(&perturb_ruin_recreate_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&rr_seed)
                    .launch(one_thread_cfg.clone())?;
            }
            refresh_host_mirrors_from_device!();
        }
        
        for _ in 0..ils_quick_refine {
            global_round += 1;
            reset_move_counters!();
            
            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
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
                stream
                    .launch_builder(&compute_moves_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_move_priorities)
                    .arg(&mut d_num_valid_moves)
                    .launch(cfg.clone())?;
            }
            
            let nvm_vec = stream.memcpy_dtov(&d_num_valid_moves)?;
            let num_valid_moves: i32 = nvm_vec.iter().sum();
            if num_valid_moves == 0 {
                break;
            }
            
            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;

            valid_moves.clear();
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key > 0 && (key >> 16) >= 1000 {
                    valid_moves.push((node, key));
                }
            }

            if valid_moves.is_empty() {
                break;
            }

            let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

            let mut k_base = valid_moves.len();
            if k_base > move_limit {
                k_base = move_limit;
            }
            let extra_window = 16_384usize;
            let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window));

            if k_cand > 1 {
                valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                valid_moves[..k_cand].sort_unstable_by(cmp);
            } else {
                valid_moves[..k_cand].sort_unstable_by(cmp);
            }

            nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
            let slack = 4usize;

            tgt_used.fill(0);
            for p in 0..num_parts_usize {
                let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
                tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
            }

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for &(node, key) in valid_moves[..k_cand].iter() {
                if sorted_move_nodes.len() >= k_base {
                    break;
                }
                let tgt = (key & 63) as usize;
                if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                    tgt_used[tgt] += 1;
                    sorted_move_nodes.push(node as i32);
                    sorted_move_parts.push(tgt as i32);
                }
            }
            if sorted_move_nodes.is_empty() {
                let take = std::cmp::min(k_base, k_cand);
                sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
                sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
            }
            
            let moves_executed = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
            if moves_executed > 0 {
                for &node in sorted_move_nodes.iter().take(moves_executed as usize) {
                    node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
                }
            }
            if moves_executed == 0 {
                break;
            }
        }

        do_swap_phase!(
            &mut d_partition, &mut d_nodes_in_part,
            &mut d_edge_flags_all, &mut d_edge_flags_double,
            &mut d_swap_gains, &mut swap_gains_host,
            &mut partition_host_swap, &mut partition_mut_swap,
            &mut part_to_part, &mut used_ba_buf,
            25, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
        )?;

        unsafe {
            stream
                .launch_builder(&compute_connectivity_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&mut d_connectivity)
                .launch(hedge_cfg.clone())?;
        }
        
        let connectivity_vec = stream.memcpy_dtov(&d_connectivity)?;
        let new_connectivity: i32 = connectivity_vec.iter().sum();

        let mut improved = false;
        if new_connectivity < best_connectivity {
            improved = true;
            best_connectivity = new_connectivity;
            best_partition_host = stream.memcpy_dtov(&d_partition)?;
            best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;

            let mut new_conn_with_idx: Vec<(i32, i32)> = connectivity_vec.iter().enumerate().map(|(i, &c)| (c, i as i32)).collect();
            new_conn_with_idx.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let new_high_hedge_ids: Vec<i32> = new_conn_with_idx.iter().take(num_high_hedges).map(|&(_, id)| id).collect();
            stream.memcpy_htod(&new_high_hedge_ids, &mut d_high_hedge_ids)?;
        }

        {
            let iter_partition = stream.memcpy_dtov(&d_partition)?;
            let mut worst_idx = 0usize;
            let mut worst_conn = pop_connectivities[0];
            for (pi, &pc) in pop_connectivities.iter().enumerate() {
                if pc > worst_conn { worst_conn = pc; worst_idx = pi; }
            }
            if pop_partitions.len() < 3 {
                pop_partitions.push(iter_partition);
                pop_connectivities.push(new_connectivity);
            } else if new_connectivity < worst_conn {
                pop_partitions[worst_idx] = iter_partition;
                pop_connectivities[worst_idx] = new_connectivity;
            }
        }

        let elite_slot: Option<usize> = if elite_count < elite_pool_size {
            let slot = elite_count;
            elite_count += 1;
            Some(slot)
        } else {
            let mut worst_idx = 0usize;
            let mut worst_score = elite_scores[0];
            for i in 1..elite_pool_size {
                if elite_scores[i] > worst_score {
                    worst_score = elite_scores[i];
                    worst_idx = i;
                }
            }
            if new_connectivity < worst_score {
                Some(worst_idx)
            } else {
                None
            }
        };

        if let Some(slot) = elite_slot {
            let src_part: Vec<i32>;
            let src_slice: &[i32] = if improved {
                &best_partition_host
            } else {
                src_part = stream.memcpy_dtov(&d_partition)?;
                &src_part
            };
            let n = challenge.num_nodes as usize;
            elite_flat_host[slot * n..(slot + 1) * n].copy_from_slice(src_slice);
            elite_scores[slot] = new_connectivity;
        }

        let delta = new_connectivity - current_connectivity;
        let accept = if delta <= 0 {
            true
        } else if sa_temp > 0.01 {
            let rng_seed = 0xDEADu64.wrapping_add(ils_iter as u64).wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
            let rng_val = ((rng_seed >> 33) as f64) / (u32::MAX as f64);
            let prob = (-(delta as f64) / sa_temp).exp();
            rng_val < prob
        } else {
            false
        };

        if accept {
            current_connectivity = new_connectivity;
            current_partition_host = stream.memcpy_dtov(&d_partition)?;
            current_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;
        }

        sa_temp *= sa_cooling;
        use_consensus_next = !improved;
    }
    
    if pop_partitions.len() >= 2 {
        let mut pop_order: Vec<usize> = (0..pop_partitions.len()).collect();
        pop_order.sort_unstable_by_key(|&i| pop_connectivities[i]);

        for cross_idx in 0..std::cmp::min(pop_order.len().saturating_sub(1), 2) {
            let parent_a_idx = pop_order[0];
            let parent_b_idx = pop_order[cross_idx + 1];

            let d_parent_a = stream.memcpy_stod(&pop_partitions[parent_a_idx])?;
            let d_parent_b = stream.memcpy_stod(&pop_partitions[parent_b_idx])?;
            let mut d_child_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
            let mut d_child_nip = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;

            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_parent_a)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .launch(hedge_cfg.clone())?;
            }
            let mut d_edge_flags_a = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
            stream.memcpy_dtod(&d_edge_flags_all, &mut d_edge_flags_a)?;

            let mut d_edge_flags_b_all = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_parent_b)
                    .arg(&mut d_edge_flags_b_all)
                    .arg(&mut d_edge_flags_double)
                    .launch(hedge_cfg.clone())?;
            }

            unsafe {
                stream
                    .launch_builder(&crossover_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_parent_a)
                    .arg(&d_parent_b)
                    .arg(&d_edge_flags_a)
                    .arg(&d_edge_flags_b_all)
                    .arg(&mut d_child_partition)
                    .launch(cfg.clone())?;
            }

            let child_partition_host = stream.memcpy_dtov(&d_child_partition)?;
            let mut child_nip_host = vec![0i32; challenge.num_parts as usize];
            for &p in child_partition_host.iter() {
                if (p as usize) < challenge.num_parts as usize {
                    child_nip_host[p as usize] += 1;
                }
            }
            stream.memcpy_htod(&child_nip_host, &mut d_child_nip)?;

            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_child_partition)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .launch(hedge_cfg.clone())?;
            }

            unsafe {
                stream
                    .launch_builder(&balance_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&1i32)
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_child_partition)
                    .arg(&mut d_child_nip)
                    .launch(one_thread_cfg.clone())?;
            }

            d_partition = d_child_partition;
            d_nodes_in_part = d_child_nip;
            refresh_host_mirrors_from_device!();

            for _ in 0..ils_quick_refine {
                global_round += 1;
                reset_move_counters!();
                unsafe {
                    stream
                        .launch_builder(&precompute_edge_flags_kernel)
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
                    stream
                        .launch_builder(&compute_moves_kernel)
                        .arg(&(challenge.num_nodes as i32))
                        .arg(&(challenge.num_parts as i32))
                        .arg(&(challenge.max_part_size as i32))
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&d_partition)
                        .arg(&d_nodes_in_part)
                        .arg(&d_edge_flags_all)
                        .arg(&d_edge_flags_double)
                        .arg(&mut d_move_priorities)
                        .arg(&mut d_num_valid_moves)
                        .launch(cfg.clone())?;
                }
                let nvm_vec = stream.memcpy_dtov(&d_num_valid_moves)?;
                let nvm: i32 = nvm_vec.iter().sum();
                if nvm == 0 { break; }
                stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
                valid_moves.clear();
                for (node, &key) in move_keys_host.iter().enumerate() {
                    if key > 0 && (key >> 16) >= 1000 { valid_moves.push((node, key)); }
                }
                if valid_moves.is_empty() { break; }
                let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
                let k_base = std::cmp::min(valid_moves.len(), move_limit / 2);
                let k_cand = std::cmp::min(valid_moves.len(), k_base + 8192);
                if k_cand > 1 {
                    valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                    valid_moves[..k_cand].sort_unstable_by(cmp);
                }
                nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
                tgt_used.fill(0);
                for p in 0..num_parts_usize {
                    let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
                    tgt_quota[p] = std::cmp::max(1, free + 4);
                }
                sorted_move_nodes.clear();
                sorted_move_parts.clear();
                for &(node, key) in valid_moves[..k_cand].iter() {
                    if sorted_move_nodes.len() >= k_base { break; }
                    let tgt = (key & 63) as usize;
                    if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                        tgt_used[tgt] += 1;
                        sorted_move_nodes.push(node as i32);
                        sorted_move_parts.push(tgt as i32);
                    }
                }
                if sorted_move_nodes.is_empty() {
                    let take = std::cmp::min(k_base, k_cand);
                    sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
                    sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
                }
                let me = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
                if me > 0 {
                    for &node in sorted_move_nodes.iter().take(me as usize) {
                        node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
                    }
                }
                if me == 0 {
                    if do_hyperedge_centric_phase!() == 0 {
                        break;
                    }
                }
            }

            do_swap_phase!(
                &mut d_partition, &mut d_nodes_in_part,
                &mut d_edge_flags_all, &mut d_edge_flags_double,
                &mut d_swap_gains, &mut swap_gains_host,
                &mut partition_host_swap, &mut partition_mut_swap,
                &mut part_to_part, &mut used_ba_buf,
                15, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
            )?;

            unsafe {
                stream
                    .launch_builder(&compute_connectivity_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&mut d_connectivity)
                    .launch(hedge_cfg.clone())?;
            }
            let cross_conn_vec = stream.memcpy_dtov(&d_connectivity)?;
            let cross_conn: i32 = cross_conn_vec.iter().sum();

            if cross_conn < best_connectivity {
                best_connectivity = cross_conn;
                best_partition_host = stream.memcpy_dtov(&d_partition)?;
                best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;

                let mut new_conn_with_idx: Vec<(i32, i32)> = cross_conn_vec.iter().enumerate().map(|(i, &c)| (c, i as i32)).collect();
                new_conn_with_idx.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                let new_high_hedge_ids: Vec<i32> = new_conn_with_idx.iter().take(num_high_hedges).map(|&(_, id)| id).collect();
                stream.memcpy_htod(&new_high_hedge_ids, &mut d_high_hedge_ids)?;
            }
        }
    }

    let d_partition_final = stream.memcpy_stod(&best_partition_host)?;
    let d_nodes_in_part_final = stream.memcpy_stod(&best_nodes_in_part_host)?;
    d_partition = d_partition_final;
    d_nodes_in_part = d_nodes_in_part_final;
    partition_host_mirror.copy_from_slice(&best_partition_host);
    nodes_in_part_mirror.copy_from_slice(&best_nodes_in_part_host);
    
    for _ in 0..post_ils_polish {
        global_round += 1;
        reset_move_counters!();
        
        unsafe {
            stream
                .launch_builder(&precompute_edge_flags_kernel)
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
            stream
                .launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .launch(cfg.clone())?;
        }
        
        let nvm_vec = stream.memcpy_dtov(&d_num_valid_moves)?;
        let num_valid_moves: i32 = nvm_vec.iter().sum();
        if num_valid_moves == 0 {
            break;
        }
        
        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;

        valid_moves.clear();
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key > 0 && (key >> 16) >= 1000 {
                valid_moves.push((node, key));
            }
        }

        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

        let polish_limit = 100_000usize;
        let k_base = std::cmp::min(valid_moves.len(), polish_limit);
        let extra_window = 16_384usize;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
        let slack = 3usize;

        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        for &(node, key) in valid_moves[..k_cand].iter() {
            if sorted_move_nodes.len() >= k_base {
                break;
            }
            let tgt = (key & 63) as usize;
            if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                tgt_used[tgt] += 1;
                sorted_move_nodes.push(node as i32);
                sorted_move_parts.push(tgt as i32);
            }
        }
        if sorted_move_nodes.is_empty() {
            let take = std::cmp::min(k_base, k_cand);
            sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
            sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
        }
        
        let moves_executed = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
        if moves_executed > 0 {
            for &node in sorted_move_nodes.iter().take(moves_executed as usize) {
                node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
            }
        }
        if moves_executed == 0 {
            if do_hyperedge_centric_phase!() == 0 {
                break;
            }
        }
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        10, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    unsafe {
        stream
            .launch_builder(&precompute_edge_flags_kernel)
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
        stream
            .launch_builder(&balance_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&1i32)
            .arg(&(challenge.max_part_size as i32))
            .arg(&challenge.d_node_offsets)
            .arg(&challenge.d_node_hyperedges)
            .arg(&d_edge_flags_all)
            .arg(&d_edge_flags_double)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }
    refresh_host_mirrors_from_device!();

    let post_balance_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 128) as usize)
        .unwrap_or(base_post_balance);

    for _ in 0..post_balance_rounds {
        global_round += 1;
        reset_move_counters!();

        unsafe {
            stream
                .launch_builder(&precompute_edge_flags_kernel)
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
            stream
                .launch_builder(&compute_moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_num_valid_moves)
                .launch(cfg.clone())?;
        }
        let nvm_vec = stream.memcpy_dtov(&d_num_valid_moves)?;
        let num_valid_moves: i32 = nvm_vec.iter().sum();
        if num_valid_moves == 0 {
            break;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;

        valid_moves.clear();
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key > 0 && (key >> 16) >= 1000 {
                valid_moves.push((node, key));
            }
        }

        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

        nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);

        let mut k = valid_moves.len();

        let adaptive_limit = move_limit / 2;
        
        if k > adaptive_limit {
            k = adaptive_limit;
            valid_moves.select_nth_unstable_by(k - 1, cmp);
            valid_moves[..k].sort_unstable_by(cmp);
        } else if k > 1000 {
            valid_moves.select_nth_unstable_by(k - 1, cmp);
            valid_moves[..k].sort_unstable_by(cmp);
        } else {
            valid_moves.sort_unstable_by(cmp);
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        sorted_move_nodes.extend(valid_moves[..k].iter().map(|(node, _)| *node as i32));
        sorted_move_parts.extend(valid_moves[..k].iter().map(|(_, key)| (key & 63) as i32));

        let mut moves_executed = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);

        if moves_executed == 0 && k < valid_moves.len() {
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for i in k..valid_moves.len() {
                sorted_move_nodes.push(valid_moves[i].0 as i32);
                sorted_move_parts.push((valid_moves[i].1 & 63) as i32);
            }

            moves_executed = replay_execute_moves_host!(sorted_move_nodes, sorted_move_parts);
        }

        if moves_executed > 0 {
            for &node in sorted_move_nodes.iter().take(moves_executed as usize) {
                node_tabu_until[node as usize] = global_round + tabu_tenure as i32;
            }
        }

        if moves_executed == 0 {
            if do_hyperedge_centric_phase!() == 0 {
                break;
            }
        }
    }

    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    save_solution(&Solution {
        partition: partition_u32,
    })?;
    Ok(())
}
