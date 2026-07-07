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

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_50k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_50k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_50k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_50k")?;

    let compute_moves_kernel = module.load_function("compute_refinement_moves_optimized_50k")?;
    let balance_kernel = module.load_function("balance_final_50k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_50k")?;
    let reduce_connectivity_sum_kernel = module.load_function("reduce_connectivity_sum_50k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_50k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_50k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_50k")?;
    let score_pr_g3_warp_kernel = module.load_function("score_pr_g3_warp_50k")?;

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

    let connectivity_reduce_cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_hyperedges as u32 + block_size * 2 - 1) / (block_size * 2),
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (block_size as usize * std::mem::size_of::<i32>()) as u32,
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
    let mut d_edge_flags_all = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
    let mut d_edge_flags_double = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
    let mut d_connectivity = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_total_connectivity = stream.alloc_zeros::<i32>(4096)?;

    let mut d_hedge_choice = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;

    let swap_buf_size = 3 * challenge.num_nodes as usize;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;

    let num_parts_usize = challenge.num_parts as usize;
    let is_sparse = (challenge.num_nodes as usize) > 4 * (challenge.num_hyperedges as usize + 1);

    let effort = hyperparameters
        .as_ref()
        .and_then(|p| p.get("effort").and_then(|v| v.as_i64()))
        .unwrap_or(3);

    let (base_refine, base_ils, base_ils_quick, base_polish, base_post_balance) = match effort {
        5 => (8000, 6, 70, 300, 0),
        4 => (6000, 5, 60, 200, 0),
        3 => (4000, 10, 50, 150, 64),
        2 => (2000, 5, 50, 100, 64),
        1 => (1000, 3, 25, 40, 32),
        0 => (500, 3, 20, 30, 32),
        _ => (2000, 5, 50, 150, 64),
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
        .unwrap_or(12);

    let tabu_fail_tenure = 3usize;
    let tabu_mark_base = 4096usize;
    let tabu_mark_mult = 16usize;
    let tabu_fail_mark_len = 4096usize;

    let extra_window = 57344usize;
    let slack_early = 8usize;
    let slack_mid = 4usize;
    let slack_late = 2usize;

    let move_limit: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("move_limit").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(256, 1_000_000) as usize)
        .unwrap_or(if is_sparse {
            262_144
        } else if challenge.num_hyperedges as usize >= 150_000
            || challenge.num_nodes as usize >= 250_000
        {
            131_072
        } else {
            200_000
        });

    let pr_rebal_hoist: bool = hyperparameters
        .as_ref()
        .and_then(|p| p.get("pr_rebal_hoist").and_then(|v| v.as_bool()))
        .unwrap_or(false);

    let neg_gain_thresh: i32 = 5;
    let scan_limit_swap = 32usize;
    let scan_limit_cycle = 8usize;

    let mut part_to_part: Vec<Vec<(usize, i32)>> = vec![vec![]; num_parts_usize * num_parts_usize];
    let mut swap_gains_host: Vec<i32> = vec![0i32; swap_buf_size];
    let mut partition_host_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_host_refine: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut nodes_in_part_host: Vec<i32> = vec![0i32; num_parts_usize];
    let mut move_keys_host: Vec<i32> = vec![0i32; challenge.num_nodes as usize];

    let hedge_offsets_host = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let hyperedge_nodes_host = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let mut hedge_sizes_host: Vec<i32> = Vec::with_capacity(challenge.num_hyperedges as usize);
    for h in 0..(challenge.num_hyperedges as usize) {
        let sz = hedge_offsets_host[h + 1] - hedge_offsets_host[h];
        hedge_sizes_host.push(sz);
    }

    let build_high_hedge_ids =
        |connectivity: &[i32], num_high_hedges: usize| -> Vec<i32> {
            let mut impact: Vec<(i32, i32, i32)> = connectivity
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, hedge_sizes_host[i], i as i32))
                .collect();
            impact.sort_unstable_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            impact
                .into_iter()
                .take(num_high_hedges)
                .map(|t| t.2)
                .collect()
        };

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
        pref_priorities[b].cmp(&pref_priorities[a]).then_with(|| a.cmp(&b))
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

    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
    stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

    let simulate_execute_moves =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         move_nodes: &[i32],
         move_parts: &[i32]|
         -> i32 {
            let mut moves_executed = 0i32;
            for i in 0..move_nodes.len() {
                let node = move_nodes[i];
                let target_part = move_parts[i];
                if node < 0 || target_part < 0 {
                    continue;
                }
                let node_usize = node as usize;
                if node_usize >= partition_mirror.len() {
                    continue;
                }
                let current_part = partition_mirror[node_usize];
                if current_part >= 0
                    && (current_part as usize) < nodes_in_part_mirror.len()
                    && (target_part as usize) < nodes_in_part_mirror.len()
                    && nodes_in_part_mirror[target_part as usize] < challenge.max_part_size as i32
                    && nodes_in_part_mirror[current_part as usize] > 1
                {
                    partition_mirror[node_usize] = target_part;
                    nodes_in_part_mirror[current_part as usize] -= 1;
                    nodes_in_part_mirror[target_part as usize] += 1;
                    moves_executed += 1;
                }
            }
            moves_executed
        };

    let perturb_on_host =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         perturb_strength: i32,
         seed: u64| {
            let mut state = seed;
            let mut moves_made = 0i32;
            let target_moves = (challenge.num_nodes as i32 * perturb_strength) / 100;
            for _attempt in 0..(challenge.num_nodes as usize) {
                if moves_made >= target_moves {
                    break;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let node = (state % challenge.num_nodes as u64) as usize;
                let current_part = partition_mirror[node];
                if current_part < 0 || current_part >= challenge.num_parts as i32 {
                    continue;
                }
                if nodes_in_part_mirror[current_part as usize] <= 1 {
                    continue;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let target_part = (state % challenge.num_parts as u64) as i32;
                if target_part != current_part
                    && nodes_in_part_mirror[target_part as usize] < challenge.max_part_size as i32
                {
                    partition_mirror[node] = target_part;
                    nodes_in_part_mirror[current_part as usize] -= 1;
                    nodes_in_part_mirror[target_part as usize] += 1;
                    moves_made += 1;
                }
            }
        };

    let perturb_guided_on_host =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         high_hedge_ids_host: &[i32]| {
            let np = std::cmp::min(num_parts_usize, 64usize);
            for &hedge_i32 in high_hedge_ids_host.iter() {
                if hedge_i32 < 0 {
                    continue;
                }
                let hedge = hedge_i32 as usize;
                let start = hedge_offsets_host[hedge] as usize;
                let end = hedge_offsets_host[hedge + 1] as usize;
                let hedge_size = end - start;
                if hedge_size <= 1 {
                    continue;
                }

                let mut part_count = [0i32; 64];
                for k in start..end {
                    let node = hyperedge_nodes_host[k] as usize;
                    let part = partition_mirror[node];
                    if part >= 0 && (part as usize) < np {
                        part_count[part as usize] += 1;
                    }
                }

                let mut majority_part = 0usize;
                for p in 1..np {
                    if part_count[p] > part_count[majority_part] {
                        majority_part = p;
                    }
                }

                let mut parts_present = 0i32;
                for p in 0..np {
                    if part_count[p] > 0 {
                        parts_present += 1;
                    }
                }
                if parts_present <= 1 {
                    continue;
                }

                for _iter in 0..np {
                    if parts_present <= 1 {
                        break;
                    }
                    if nodes_in_part_mirror[majority_part] >= challenge.max_part_size as i32 {
                        break;
                    }

                    let mut min_part = usize::MAX;
                    let mut min_cnt = 0i32;
                    for p in 0..np {
                        if p == majority_part {
                            continue;
                        }
                        let cnt = part_count[p];
                        if cnt <= 0 {
                            continue;
                        }
                        if min_part == usize::MAX || cnt < min_cnt || (cnt == min_cnt && p < min_part)
                        {
                            min_part = p;
                            min_cnt = cnt;
                        }
                    }
                    if min_part == usize::MAX {
                        break;
                    }

                    let mut moved_any = false;
                    for k in start..end {
                        if part_count[min_part] <= 0 {
                            break;
                        }
                        if nodes_in_part_mirror[majority_part] >= challenge.max_part_size as i32 {
                            break;
                        }

                        let node = hyperedge_nodes_host[k] as usize;
                        if partition_mirror[node] != min_part as i32 {
                            continue;
                        }
                        if nodes_in_part_mirror[min_part] <= 1 {
                            continue;
                        }

                        partition_mirror[node] = majority_part as i32;
                        nodes_in_part_mirror[min_part] -= 1;
                        nodes_in_part_mirror[majority_part] += 1;
                        part_count[min_part] -= 1;
                        part_count[majority_part] += 1;
                        moved_any = true;
                        if part_count[min_part] == 0 {
                            parts_present -= 1;
                            break;
                        }
                    }

                    if !moved_any {
                        break;
                    }
                }
            }
        };

    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(challenge.num_nodes as usize);

    let mut stagnant_rounds = 0usize;
    let max_stagnant_rounds = 30usize;

    let mut node_tabu_until: Vec<usize> = vec![0; challenge.num_nodes as usize];

    let mut tgt_used: Vec<usize> = vec![0; num_parts_usize];
    let mut tgt_quota: Vec<usize> = vec![0; num_parts_usize];

    let mut total_moves_executed = 0usize;
    let mut total_perturbations = 0usize;

    for round in 0..refinement_rounds {
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
                .launch(cfg.clone())?;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;

        valid_moves.clear();
        let max_gain = move_keys_host.iter().filter(|&&k| k as u32 != 0x80000000).map(|&k| k >> 16).max().unwrap_or(0);
        let aspiration_threshold = (max_gain * 3) / 4;
        
        let mut rng_state = 123456789u64.wrapping_add(round as u64);

        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                let is_tabu = node_tabu_until[node] > round && gain < aspiration_threshold;
                
                if !is_tabu {
                    if gain > 0 {
                        valid_moves.push((node, key));
                    } else if gain >= -3 {
                        let rounds_left = refinement_rounds.saturating_sub(round) as u64;
                        let penalty = if gain < 0 { (-gain) as u64 } else { 0 };
                        
                        let probability_num = rounds_left * rounds_left;
                        let probability_den = (refinement_rounds as u64 * refinement_rounds as u64)
                            .saturating_mul(1 + penalty * 5);
                        
                        rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
                        if probability_den > 0 && (rng_state % probability_den) < probability_num {
                            let new_key = (0 << 16) | (key & 0xFFFF);
                            valid_moves.push((node, new_key));
                        }
                    }
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
            (move_limit * 3) / 4
        } else {
            move_limit / 4
        };
        if k_base > adaptive_limit {
            k_base = adaptive_limit;
        }

        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        let slack = if round < 64 {
            slack_early
        } else if round < 256 {
            slack_mid
        } else {
            slack_late
        };

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

        let mut moves_executed = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }

        if moves_executed == 0 && k_cand > k_base {
            let fail_mark_len = std::cmp::min(sorted_move_nodes.len(), tabu_fail_mark_len);
            for &node in sorted_move_nodes.iter().take(fail_mark_len) {
                node_tabu_until[node as usize] = round + tabu_fail_tenure;
            }

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            let tail = &valid_moves[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            sorted_move_nodes.extend(tail.iter().take(take).map(|(n, _)| *n as i32));
            sorted_move_parts.extend(tail.iter().take(take).map(|(_, key)| (key & 63) as i32));

            if !sorted_move_nodes.is_empty() {
                moves_executed = simulate_execute_moves(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &sorted_move_nodes,
                    &sorted_move_parts,
                );
                if moves_executed > 0 {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
            }
        }

        total_moves_executed += moves_executed as usize;

        if moves_executed > 0 {
            let mark_len = std::cmp::min(
                sorted_move_nodes.len(),
                std::cmp::max(
                    tabu_mark_base,
                    (moves_executed as usize).saturating_mul(tabu_mark_mult),
                ),
            );
            let until = round + tabu_tenure;
            for &node in sorted_move_nodes.iter().take(mark_len) {
                node_tabu_until[node as usize] = until;
            }
        }

        if moves_executed == 0 {
            stagnant_rounds += 1;
            if stagnant_rounds >= 3 && round < refinement_rounds.saturating_sub(50) {
                let mini_seed = 987654321u64 + (round as u64) * 123456789u64;
                perturb_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    3i32,
                    mini_seed,
                );
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                total_perturbations += 1;
                stagnant_rounds = 0;
            } else if stagnant_rounds > max_stagnant_rounds {
                break;
            }
        } else {
            stagnant_rounds = 0;
        }
    }

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
            stream.memcpy_dtoh(&mut *$d_partition, $partition_host_swap)?;
            for _swap_round in 0..$max_rounds {
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
                    for k in 0..3usize {
                        let val = $swap_gains_host[node * 3 + k];
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
                                if combined > best_combined {
                                    best_combined = combined;
                                    best_j = j;
                                }
                            }
                            if best_j < lba_len && best_combined > 0 {
                                let (node_b, _) = $part_to_part[idx_ba][best_j];
                                $partition_mut_swap[node_a] = b as i32;
                                $partition_mut_swap[node_b] = a as i32;
                                $used_ba_buf[best_j] = true;
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
                                    for j in 0..sl_bc {
                                        let (node_bc, gain_bc) = $part_to_part[idx_bc][j];
                                        if $partition_mut_swap[node_bc] as usize != b { continue; }
                                        if node_bc == node_ab { continue; }
                                        if gain_ab + gain_bc <= 0 { break; }
                                        for k in 0..sl_ca {
                                            let (node_ca, gain_ca) = $part_to_part[idx_ca][k];
                                            if $partition_mut_swap[node_ca] as usize != c { continue; }
                                            if node_ca == node_ab || node_ca == node_bc { continue; }
                                            if gain_ab + gain_bc + gain_ca > 0 {
                                                $partition_mut_swap[node_ab] = b as i32;
                                                $partition_mut_swap[node_bc] = c as i32;
                                                $partition_mut_swap[node_ca] = a as i32;
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

                if swap_count == 0 { break; }
                total_swaps += swap_count;
                if swap_count >= prev_swap_count {
                    stagnant += 1;
                    if stagnant >= 3 { break; }
                } else {
                    stagnant = 0;
                }
                prev_swap_count = swap_count;
                stream.memcpy_htod($partition_mut_swap, &mut *$d_partition)?;
                $partition_host_swap.copy_from_slice($partition_mut_swap);
            }
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
    partition_host_refine.copy_from_slice(&partition_host_swap);

    for _post_swap_round in 0..30 {
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
                .launch(cfg.clone())?;
        }
        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                if gain > 0 { valid_moves.push((node, key)); }
            }
        }
        if valid_moves.is_empty() { break; }
        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let k_base = std::cmp::min(valid_moves.len(), move_limit / 2);
        let k_cand = std::cmp::min(valid_moves.len(), k_base + extra_window / 2);
        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }
        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free + slack_mid);
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
        let me = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );
        if me > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
        if me == 0 { break; }
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        50, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    let perturb_strength = hyperparameters
        .as_ref()
        .and_then(|p| p.get("perturb_strength").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 20) as i32)
        .unwrap_or(3);

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
    let conn_vec = stream.memcpy_dtov(&d_connectivity)?;
    let mut best_connectivity: i32 = conn_vec.iter().sum();
    let mut best_partition_host = stream.memcpy_dtov(&d_partition)?;
    let mut best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;

    let num_high_hedges = std::cmp::min(1000usize, challenge.num_hyperedges as usize);
    let mut high_hedge_ids_host: Vec<i32> = build_high_hedge_ids(&conn_vec, num_high_hedges);

    let pool_size = ils_iterations;
    let mut elite_scores: Vec<i32> = vec![i32::MAX; pool_size];
    let mut elite_flat_host: Vec<i32> = vec![0i32; pool_size * challenge.num_nodes as usize];
    elite_scores[0] = best_connectivity;
    elite_flat_host[..challenge.num_nodes as usize].copy_from_slice(&best_partition_host);
    let mut elite_count: usize = 1;
    let mut d_elite_flat = stream.alloc_zeros::<i32>(pool_size * challenge.num_nodes as usize)?;

    let mut use_consensus_next = false;

    for ils_iter in 0..ils_iterations {
        let d_partition_restored = stream.memcpy_stod(&best_partition_host)?;
        let d_nodes_in_part_restored = stream.memcpy_stod(&best_nodes_in_part_host)?;
        d_partition = d_partition_restored;
        d_nodes_in_part = d_nodes_in_part_restored;
        partition_host_refine.copy_from_slice(&best_partition_host);
        nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);

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

        let seed = 123456789u64 + (ils_iter as u64) * 987654321u64;

        let refreshed_host_from_device = if use_consensus_next && elite_count > 1 {
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

            stream.memset_zeros(&mut d_nodes_in_part)?;

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

            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            nodes_in_part_host.fill(0);
            for &p in partition_host_refine.iter() {
                if p >= 0 && (p as usize) < num_parts_usize {
                    nodes_in_part_host[p as usize] += 1;
                }
            }
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;

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
            true
        } else {
            if ils_iter % 2 == 0 {
                perturb_guided_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &high_hedge_ids_host,
                );
            } else {
                perturb_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    perturb_strength,
                    seed,
                );
            }
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            false
        };

        if refreshed_host_from_device {
            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
        }

        for _ in 0..ils_quick_refine {
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
                    .launch(cfg.clone())?;
            }

            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
            valid_moves.clear();
            let mut rng_state = 987654321u64;
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key as u32 != 0x80000000 {
                    let gain = key >> 16;
                    if gain > 0 {
                        valid_moves.push((node, key));
                    } else if gain == 0 {
                        rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                        if (rng_state % 10) < 2 {
                            let new_key = (0 << 16) | (key & 0xFFFF);
                            valid_moves.push((node, new_key));
                        }
                    }
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
            let ils_extra = extra_window / 2;
            let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(ils_extra));

            if k_cand > 1 {
                valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                valid_moves[..k_cand].sort_unstable_by(cmp);
            } else {
                valid_moves[..k_cand].sort_unstable_by(cmp);
            }

            let slack = slack_mid + 2;

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

            let moves_executed = simulate_execute_moves(
                &mut partition_host_refine,
                &mut nodes_in_part_host,
                &sorted_move_nodes,
                &sorted_move_parts,
            );

            if moves_executed > 0 {
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
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
        unsafe {
            stream
                .launch_builder(&reduce_connectivity_sum_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&d_connectivity)
                .arg(&mut d_total_connectivity)
                .launch(connectivity_reduce_cfg.clone())?;
        }

        let block_sums = stream.memcpy_dtov(&d_total_connectivity)?;
        let num_blocks = ((challenge.num_hyperedges as u32 + block_size * 2 - 1) / (block_size * 2)) as usize;
        let new_connectivity: i32 = block_sums[..num_blocks].iter().sum();

        let mut improved = false;
        if new_connectivity < best_connectivity {
            improved = true;
            best_connectivity = new_connectivity;
            best_partition_host = stream.memcpy_dtov(&d_partition)?;
            best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;

            let connectivity_vec = stream.memcpy_dtov(&d_connectivity)?;
            high_hedge_ids_host = build_high_hedge_ids(&connectivity_vec, num_high_hedges);
        }

        let slot_opt: Option<usize> = if elite_count < pool_size {
            let s = elite_count;
            elite_count += 1;
            Some(s)
        } else {
            let mut worst_idx = 0usize;
            let mut worst_score = elite_scores[0];
            for i in 1..pool_size {
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

        if let Some(slot) = slot_opt {
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

        use_consensus_next = !improved;
    }

    let d_partition_final = stream.memcpy_stod(&best_partition_host)?;
    let d_nodes_in_part_final = stream.memcpy_stod(&best_nodes_in_part_host)?;
    d_partition = d_partition_final;
    d_nodes_in_part = d_nodes_in_part_final;
    partition_host_refine.copy_from_slice(&best_partition_host);
    nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);

    // ==== RECOMBINE_MULTIPARENT i42 (extension i41 → donors k=2..9, rangs 5..9 plus divergents) ====
    
    // (élites plus éloignées du best) → plus de candidats pour greedy Δ-exact. Même garde monotone.
    {
        let n_rc = challenge.num_nodes as usize;
        let nh_rc = challenge.num_hyperedges as usize;
        let max_ps_rc = challenge.max_part_size as i32;
        let np_rc = num_parts_usize;
        // i118 B_UNCONSTRAINED_REBALANCE: slack 3% transitoire sur VETO RECOMBINE_MULTIPARENT
        // (RÈGLE D'OR L4 cross-poll P₁₁ t25/i100 PATH_RELINK +184 + t23/i101 RECOMBINE +141).
        // Le veto HARD (rc_nip[p_tgt] >= max_ps_rc) bloquait toute violation transitoire de balance ;
        // ici on autorise jusqu'à +3% PUIS on restaure la balance stricte via le rebalancer Jetrw WEAK
        // (block_verts O(n), ≤10 rounds greedy multi-move, rollback si imbalancé → 0 invalid garanti).
        let slack_max_rc = max_ps_rc + (max_ps_rc * 3 + 99) / 100;

        let num_donors = (elite_count.saturating_sub(1)).min(9);
        if elite_count >= 2 {
            // Sort elite slots by score asc (lower conn = better); donors = slots[1..=num_donors]
            let mut sorted_elites: Vec<usize> = (0..elite_count).collect();
            sorted_elites.sort_unstable_by(|&a, &b| elite_scores[a].cmp(&elite_scores[b]));

            // Build CSR once (node → incident hyperedges)
            let mut nd_deg_rc: Vec<usize> = vec![0usize; n_rc];
            for h in 0..nh_rc {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_rc { nd_deg_rc[nd] += 1; }
                }
            }
            let mut nd_off_rc: Vec<usize> = vec![0usize; n_rc + 1];
            for i in 0..n_rc { nd_off_rc[i + 1] = nd_off_rc[i] + nd_deg_rc[i]; }
            let total_pins_rc = nd_off_rc[n_rc];
            let mut nd_hedges_rc: Vec<usize> = vec![0usize; total_pins_rc];
            let mut nd_cur_rc: Vec<usize> = nd_off_rc[..n_rc].to_vec();
            for h in 0..nh_rc {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_rc {
                        nd_hedges_rc[nd_cur_rc[nd]] = h;
                        nd_cur_rc[nd] += 1;
                    }
                }
            }

            // Working partition = best_partition_host (already in partition_host_refine)
            let mut rc_part = partition_host_refine.clone();
            let mut rc_nip = nodes_in_part_host.clone();

            // Outer sweep loop: fixpoint, cap 3
            'sweeps: for _sweep in 0..3usize {
                let mut sweep_accepted = 0usize;

                for dk in 0..num_donors {
                    let donor_off = sorted_elites[dk + 1] * n_rc;

                    // Disagreement set vs rc_part COURANT, static Δconn sort
                    let mut disagree_rc: Vec<(i32, usize, i32)> = Vec::new();
                    for v in 0..n_rc {
                        let p_curr = rc_part[v];
                        let p_tgt = elite_flat_host[donor_off + v];
                        if p_curr == p_tgt || p_curr < 0 || p_tgt < 0 { continue; }
                        if (p_curr as usize) >= np_rc || (p_tgt as usize) >= np_rc { continue; }

                        let mut delta = 0i32;
                        let hs_v = nd_off_rc[v];
                        let he_v = nd_off_rc[v + 1];
                        for hi in hs_v..he_v {
                            let h = nd_hedges_rc[hi];
                            let es = hedge_offsets_host[h] as usize;
                            let ee = hedge_offsets_host[h + 1] as usize;
                            let mut cnt_c = 0i32;
                            let mut cnt_t = 0i32;
                            for kp in es..ee {
                                let nd = hyperedge_nodes_host[kp] as usize;
                                if nd == v { continue; }
                                let p = rc_part[nd];
                                if p == p_curr { cnt_c += 1; }
                                else if p == p_tgt { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }
                        disagree_rc.push((delta, v, p_tgt));
                    }
                    disagree_rc.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                    // Greedy pass: Δconn exact recomputed per accept
                    for &(_, v, p_tgt) in disagree_rc.iter() {
                        let p_curr = rc_part[v];
                        if p_curr == p_tgt { continue; }
                        if rc_nip[p_curr as usize] <= 1 { continue; }
                        if rc_nip[p_tgt as usize] >= slack_max_rc { continue; } 

                        let mut delta = 0i32;
                        let hs_v = nd_off_rc[v];
                        let he_v = nd_off_rc[v + 1];
                        for hi in hs_v..he_v {
                            let h = nd_hedges_rc[hi];
                            let es = hedge_offsets_host[h] as usize;
                            let ee = hedge_offsets_host[h + 1] as usize;
                            let mut cnt_c = 0i32;
                            let mut cnt_t = 0i32;
                            for kp in es..ee {
                                let nd = hyperedge_nodes_host[kp] as usize;
                                if nd == v { continue; }
                                let p = rc_part[nd];
                                if p == p_curr { cnt_c += 1; }
                                else if p == p_tgt { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }

                        if delta < 0 {
                            rc_part[v] = p_tgt;
                            rc_nip[p_curr as usize] -= 1;
                            rc_nip[p_tgt as usize] += 1;
                            sweep_accepted += 1;
                        }
                    }
                }

                if sweep_accepted == 0 { break 'sweeps; }
            }

            // ==== B_UNCONSTRAINED_REBALANCE WEAK (i118, Jetrw — cross-poll P₁₁ t25/i100 + t23/i101) ====
            // Restore strict balance after the slack-3% RECOMBINE moves above.
            // WEAK: pre-build block_verts O(n) once, commit greedily ≥1 move/round, ≤10 rounds.
            // Rollback to floor (no commit) if imbalanced → 0 invalid garanti. Réutilise nd_off_rc/nd_hedges_rc.
            let balance_ok_rc = if rc_nip.iter().any(|&c| c > max_ps_rc) {
                // Pre-build block vertex lists O(n) — avoids O(n×np) scan per round
                let mut block_verts_rc: Vec<Vec<usize>> = vec![Vec::new(); np_rc];
                for v in 0..n_rc {
                    let p = rc_part[v];
                    if p >= 0 && (p as usize) < np_rc { block_verts_rc[p as usize].push(v); }
                }
                let mut rebal_present_rc = vec![0i32; np_rc];
                let mut rebal_seen_rc = vec![false; np_rc];
                let ok = 'rebalance_rc: {
                    for _round in 0..10usize {
                        let mut all_balanced = true;
                        let mut did_move_round = false;
                        for src_p in 0..np_rc {
                            if rc_nip[src_p] <= max_ps_rc { continue; }
                            all_balanced = false;
                            // Score vertices in overfull block only (not all n)
                            let mut cands: Vec<(i32, usize, usize)> = Vec::new();
                            for &v in &block_verts_rc[src_p] {
                                let (best_delta, best_dst) = if pr_rebal_hoist {
                                    // Hoisted: scan each hedge ONCE per vertex.
                                    // base_delta = cnt_s-dependent terms (dst_p-invariant).
                                    // present[p] = #hedges where part p has >=1 pin (excl v).
                                    // delta(dst_p) = base_delta + (deg_v - present[dst_p]).
                                    for x in rebal_present_rc.iter_mut() { *x = 0; }
                                    let deg_v = (nd_off_rc[v + 1] - nd_off_rc[v]) as i32;
                                    let mut base_delta = 0i32;
                                    for hi in nd_off_rc[v]..nd_off_rc[v + 1] {
                                        let h = nd_hedges_rc[hi];
                                        let es = hedge_offsets_host[h] as usize;
                                        let ee = hedge_offsets_host[h + 1] as usize;
                                        let mut cnt_s = 0i32;
                                        for x in rebal_seen_rc.iter_mut() { *x = false; }
                                        for kp in es..ee {
                                            let nd = hyperedge_nodes_host[kp] as usize;
                                            if nd == v { continue; }
                                            let p = rc_part[nd];
                                            if p >= 0 && p == src_p as i32 { cnt_s += 1; }
                                            if p >= 0 && (p as usize) < np_rc { rebal_seen_rc[p as usize] = true; }
                                        }
                                        if cnt_s == 0 { base_delta -= 1; }
                                        for p in 0..np_rc { if rebal_seen_rc[p] { rebal_present_rc[p] += 1; } }
                                    }
                                    let mut bd = i32::MAX;
                                    let mut bdst = np_rc;
                                    for dst_p in 0..np_rc {
                                        if dst_p == src_p { continue; }
                                        if rc_nip[dst_p] >= max_ps_rc { continue; }
                                        let delta = base_delta + (deg_v - rebal_present_rc[dst_p]);
                                        if delta < bd || (delta == bd && dst_p < bdst) { bd = delta; bdst = dst_p; }
                                    }
                                    (bd, bdst)
                                } else {
                                    let mut bd = i32::MAX;
                                    let mut bdst = np_rc;
                                    for dst_p in 0..np_rc {
                                        if dst_p == src_p { continue; }
                                        if rc_nip[dst_p] >= max_ps_rc { continue; }
                                        let mut delta = 0i32;
                                        for hi in nd_off_rc[v]..nd_off_rc[v + 1] {
                                            let h = nd_hedges_rc[hi];
                                            let es = hedge_offsets_host[h] as usize;
                                            let ee = hedge_offsets_host[h + 1] as usize;
                                            let mut cnt_s = 0i32;
                                            let mut cnt_d = 0i32;
                                            for kp in es..ee {
                                                let nd = hyperedge_nodes_host[kp] as usize;
                                                if nd == v { continue; }
                                                let p = rc_part[nd];
                                                if p == src_p as i32 { cnt_s += 1; }
                                                else if p == dst_p as i32 { cnt_d += 1; }
                                            }
                                            if cnt_s == 0 { delta -= 1; }
                                            if cnt_d == 0 { delta += 1; }
                                        }
                                        if delta < bd || (delta == bd && dst_p < bdst) { bd = delta; bdst = dst_p; }
                                    }
                                    (bd, bdst)
                                };
                                if best_dst < np_rc {
                                    cands.push((best_delta, v, best_dst));
                                }
                            }
                            // Deterministic: sort (delta ASC, vertex_id ASC)
                            cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                            // Commit greedily until block balanced
                            for &(_, v, dst_p) in &cands {
                                if rc_nip[src_p] <= max_ps_rc { break; }
                                if rc_nip[dst_p] >= max_ps_rc { continue; }
                                rc_part[v] = dst_p as i32;
                                rc_nip[src_p] -= 1;
                                rc_nip[dst_p] += 1;
                                if let Some(pos) = block_verts_rc[src_p].iter().rposition(|&x| x == v) {
                                    block_verts_rc[src_p].swap_remove(pos);
                                }
                                block_verts_rc[dst_p].push(v);
                                did_move_round = true;
                            }
                        }
                        if all_balanced { break 'rebalance_rc true; }
                        if !did_move_round { break 'rebalance_rc false; }
                    }
                    rc_nip.iter().all(|&c| c <= max_ps_rc)
                };
                ok
            } else {
                true // no slack move taken → already strictly balanced
            };
            // ==== END B_UNCONSTRAINED_REBALANCE WEAK ====

            // Compute final connectivity (O(total_pins))
            let mut final_conn_rc = 0i32;
            let mut pseen_rc: Vec<bool> = vec![false; np_rc];
            for h in 0..nh_rc {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut distinct = 0usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    let p = rc_part[nd];
                    if p >= 0 && (p as usize) < np_rc {
                        let pu = p as usize;
                        if !pseen_rc[pu] { pseen_rc[pu] = true; distinct += 1; }
                    }
                }
                if distinct > 1 { final_conn_rc += (distinct - 1) as i32; }
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    let p = rc_part[nd];
                    if p >= 0 && (p as usize) < np_rc { pseen_rc[p as usize] = false; }
                }
            }

            // Monotone guard: accept only if strictly improved vs best_connectivity AND balance restored
            if balance_ok_rc && final_conn_rc < best_connectivity {
                partition_host_refine.copy_from_slice(&rc_part);
                nodes_in_part_host.copy_from_slice(&rc_nip);
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                best_connectivity = final_conn_rc;
                best_partition_host.copy_from_slice(&rc_part);
                best_nodes_in_part_host.copy_from_slice(&rc_nip);
            }
            // else no-op: partition_host_refine already holds best_partition_host

            let _ = total_pins_rc;
        }
    }
    // ==== END RECOMBINE_MULTIPARENT ====

    
    // i60 (KEPT WIN 268,630) varied only the node-ordering TIEBREAK -> correlated coarsenings (low marginal
    
    // different score groups different super-nodes -> FM explores genuinely distinct bassins:
    //   modes 0..6 = i60's 6 HEAVY-EDGE node-orderings (UNCHANGED, bit-exact -> floor = 268,630-basin)
    //   mode  6    = HEAVY-CLIQUE (clique-net weight SCALE/(|h|-1): favors small/tight nets, community-aware)
    //   mode  7    = FIRST-CHOICE (weight-agnostic, match lowest-id eligible neighbor: structurally distinct)
    //   mode  8    = HEAVY-CLIQUE + degree-DESC tiebreak
    // Every candidate is the SAME single-level V-cycle (CONSTANT disruption) -> the i55/i56/i59
    // non-monotonicity (ITERATION/DEPTH, not width/score) does NOT apply. Snapshot/rollback STRICT
    // (207b608f monotone): every candidate reads ONLY the frozen `parent_*`; SINGLE commit promotes the
    // best-connectivity candidate. Modes 0..6 == i60 EXACTLY, so committed conn <= i60's: if no orthogonal
    // matching mode wins conn_vc, the winner is byte-identical to i60 (worst case bit-exact 268,630).
    // Ranking by conn_vc (cheap proxy validated by i57/i60 wins); NO probe re-rank (i58 path-dep DEAD).
    {
        let n_vc = challenge.num_nodes as usize;
        let nh_vc = challenge.num_hyperedges as usize;
        let np_vc = num_parts_usize;
        let max_ps_vc = challenge.max_part_size as i32;

        // CSR: node -> incident hyperedges (instance-static, built ONCE)
        let mut nd_deg_vc = vec![0usize; n_vc];
        for h in 0..nh_vc {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            for k in hs..he { nd_deg_vc[hyperedge_nodes_host[k] as usize] += 1; }
        }
        let mut nd_off_vc = vec![0usize; n_vc + 1];
        for i in 0..n_vc { nd_off_vc[i + 1] = nd_off_vc[i] + nd_deg_vc[i]; }
        let total_pins_vc = nd_off_vc[n_vc];
        let mut nd_hedges_vc = vec![0usize; total_pins_vc];
        {
            let mut cur = nd_off_vc[..n_vc].to_vec();
            for h in 0..nh_vc {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    nd_hedges_vc[cur[v]] = h;
                    cur[v] += 1;
                }
            }
        }

        // --- FROZEN PARENT: the only partition any candidate is allowed to read ---
        let parent_part: Vec<i32> = best_partition_host.clone();
        let parent_nip: Vec<i32> = best_nodes_in_part_host.clone();
        let parent_conn = best_connectivity;

        // Best candidate across the portfolio (floor = parent: a candidate must STRICTLY improve to win)
        let mut cand_conn = parent_conn;
        let mut cand_part: Vec<i32> = parent_part.clone();
        let mut cand_nip: Vec<i32> = parent_nip.clone();
        let mut improved = false;

        // 9 candidates: modes 0..6 = i60's 6 node-orderings (heavy-edge count, UNCHANGED). modes 6,7,8 =
        // ORTHOGONAL matching scores (clique / first-choice / clique+deg). Each is a single-level V-cycle
        // on the SAME frozen parent -> CONSTANT disruption (i55/i56/i59 depth non-monotonicity N/A).
        // M = (n_vc+1) makes the composite keys (4,5) lexicographic. Clique-net fixed point SCALE/(|h|-1)
        // (integer, 0-RNG, deterministic).
        let m_vc = (n_vc as i64) + 1;
        const CLQ_SCALE_VC: i64 = 1 << 20;
        for mode in 0..9_usize { // 0=node_asc 1=deg_desc 2=deg_asc 3=node_desc 4=deg_desc+nd_desc 5=deg_asc+nd_desc 6=clique 7=first-choice 8=clique+deg_desc
            // Partition-preserving matching (intra-block only) on the FROZEN parent
            let mut rep_vc: Vec<usize> = (0..n_vc).collect();
            let mut matched_vc: Vec<bool> = vec![false; n_vc];
            let mut shared_vc: Vec<i64> = vec![-1i64; n_vc]; // heavy-edge count (sentinel -1)
            let mut clq_vc: Vec<i64> = vec![0i64; n_vc];     // heavy-clique fixed-point weight
            let mut cands_vc: Vec<usize> = Vec::new();
            let mut n_matched_vc = 0usize;
            let cap_vc = n_vc / 4;

            for u in 0..n_vc {
                if matched_vc[u] || n_matched_vc >= cap_vc { continue; }
                let part_u = parent_part[u];
                if part_u < 0 || (part_u as usize) >= np_vc { continue; }
                cands_vc.clear();
                for hi in nd_off_vc[u]..nd_off_vc[u + 1] {
                    let h = nd_hedges_vc[hi];
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    let sz = he - hs;
                    let cw = if sz > 1 { CLQ_SCALE_VC / ((sz - 1) as i64) } else { 0 };
                    for k in hs..he {
                        let v = hyperedge_nodes_host[k] as usize;
                        if v == u || matched_vc[v] || parent_part[v] != part_u { continue; }
                        if shared_vc[v] < 0 { shared_vc[v] = 1; clq_vc[v] = cw; cands_vc.push(v); }
                        else { shared_vc[v] += 1; clq_vc[v] += cw; }
                    }
                }
                let mut best_v = usize::MAX;
                let mut best_w = 0i64;
                let mut best_sec = 0i64;
                for &v in &cands_vc {
                    let deg_v = (nd_off_vc[v + 1] - nd_off_vc[v]) as i64;
                    // primary matching SCORE by mode (0..6 == i60 heavy-edge count, bit-exact floor)
                    let w = match mode {
                        6 | 8 => clq_vc[v],   // HEAVY-CLIQUE weight
                        7 => 1i64,            // FIRST-CHOICE (constant -> ordering decides)
                        _ => shared_vc[v],    // HEAVY-EDGE count (== i60 modes 0..6)
                    };
                    let sec = match mode {
                        0 => -(v as i64),               // node_id ASC (== i54 / i57)
                        1 => deg_v,                     // degree DESC (== i57)
                        2 => -deg_v,                    // degree ASC  (== i57)
                        3 => v as i64,                  // node_id DESC (== i60)
                        4 => deg_v * m_vc + (v as i64), // degree DESC, ties -> node_id DESC (== i60)
                        5 => -deg_v * m_vc + (v as i64),// degree ASC,  ties -> node_id DESC (== i60)
                        8 => deg_v,                     // clique + degree DESC (NEW)
                        _ => -(v as i64),               // clique / first-choice -> node_id ASC (NEW)
                    };
                    if w > best_w || (w == best_w && sec > best_sec) || (w == best_w && sec == best_sec && v < best_v) {
                        best_w = w; best_v = v; best_sec = sec;
                    }
                }
                for &v in &cands_vc { shared_vc[v] = -1; clq_vc[v] = 0; }
                if best_v != usize::MAX {
                    rep_vc[best_v] = u;
                    matched_vc[u] = true;
                    matched_vc[best_v] = true;
                    n_matched_vc += 1;
                }
            }

            // Coarse node sizes (1 = singleton, 2 = pair; 0 = absorbed non-rep)
            let mut csz_vc: Vec<i32> = vec![1i32; n_vc];
            for v in 0..n_vc {
                if rep_vc[v] != v { csz_vc[rep_vc[v]] += 1; csz_vc[v] = 0; }
            }

            // Build coarse hyperedge structure: for each hedge, list of distinct coarse reps
            let mut rep_seen_vc: Vec<u32> = vec![0u32; n_vc];
            let mut stamp_vc = 0u32;
            let mut hedge_reps_vc: Vec<Vec<usize>> = Vec::with_capacity(nh_vc);
            for h in 0..nh_vc {
                stamp_vc = stamp_vc.wrapping_add(1);
                let mut reps: Vec<usize> = Vec::new();
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let r = rep_vc[hyperedge_nodes_host[k] as usize];
                    if rep_seen_vc[r] != stamp_vc { rep_seen_vc[r] = stamp_vc; reps.push(r); }
                }
                hedge_reps_vc.push(reps);
            }

            // Build coarse node -> incident hedges
            let mut coarse_hedges_vc: Vec<Vec<usize>> = vec![Vec::new(); n_vc];
            for h in 0..nh_vc {
                for &r in &hedge_reps_vc[h] { coarse_hedges_vc[r].push(h); }
            }

            // FM pass on coarse graph: 1 pass, greedy, deterministic node_id ASC; init from FROZEN parent
            let mut cpart_vc: Vec<i32> = parent_part.clone();
            let mut cnip_vc: Vec<i32> = parent_nip.clone();
            let mut pcnt_vc = vec![0i32; np_vc];
            let mut gain_vc = vec![0i32; np_vc];

            for c in 0..n_vc {
                if csz_vc[c] == 0 { continue; }
                let pc = cpart_vc[c];
                if pc < 0 || (pc as usize) >= np_vc { continue; }
                let sz = csz_vc[c];

                gain_vc.fill(0);
                for &h in &coarse_hedges_vc[c] {
                    pcnt_vc.fill(0);
                    for &r in &hedge_reps_vc[h] {
                        let pq = cpart_vc[r];
                        if pq >= 0 && (pq as usize) < np_vc { pcnt_vc[pq as usize] += 1; }
                    }
                    let only_pc = pcnt_vc[pc as usize] == 1;
                    for q in 0..np_vc {
                        if q == pc as usize { continue; }
                        let mut d = if only_pc { -1i32 } else { 0i32 };
                        if pcnt_vc[q] == 0 { d += 1; }
                        gain_vc[q] += d;
                    }
                }

                let mut best_q = usize::MAX;
                let mut best_d = 0i32;
                for q in 0..np_vc {
                    if q == pc as usize { continue; }
                    let d = gain_vc[q];
                    if d < best_d {
                        let pc_new = cnip_vc[pc as usize] - sz;
                        let q_new = cnip_vc[q] + sz;
                        if pc_new >= 1 && q_new <= max_ps_vc {
                            best_d = d;
                            best_q = q;
                        }
                    }
                }
                if best_q != usize::MAX {
                    cnip_vc[pc as usize] -= sz;
                    cnip_vc[best_q] += sz;
                    cpart_vc[c] = best_q as i32;
                }
            }

            // Uncoarsen: project coarse partition back onto a fresh copy of the FROZEN parent
            let mut vp_vc = parent_part.clone();
            let mut vnip_vc = parent_nip.clone();
            for v in 0..n_vc {
                let r = rep_vc[v];
                let new_p = cpart_vc[r];
                let old_p = vp_vc[v];
                if new_p != old_p {
                    vp_vc[v] = new_p;
                    vnip_vc[old_p as usize] -= 1;
                    vnip_vc[new_p as usize] += 1;
                }
            }

            // Compute final connectivity (O(total_pins))
            let mut conn_vc = 0i32;
            let mut pseen_vc = vec![false; np_vc];
            for h in 0..nh_vc {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut distinct = 0usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    let p = vp_vc[nd];
                    if p >= 0 && (p as usize) < np_vc {
                        let pu = p as usize;
                        if !pseen_vc[pu] { pseen_vc[pu] = true; distinct += 1; }
                    }
                }
                if distinct > 1 { conn_vc += (distinct - 1) as i32; }
                for k in hs..he {
                    let p = vp_vc[hyperedge_nodes_host[k] as usize];
                    if p >= 0 && (p as usize) < np_vc { pseen_vc[p as usize] = false; }
                }
            }

            // Track best portfolio candidate (strictly-improving over the frozen parent). NO write to
            // the live best yet — pure candidate selection (rollback of all non-winning candidates).
            if conn_vc < cand_conn {
                cand_conn = conn_vc;
                cand_part.copy_from_slice(&vp_vc);
                cand_nip.copy_from_slice(&vnip_vc);
                improved = true;
            }
            let _ = n_matched_vc;
        } // end portfolio over distinct tiebreaks

        // SINGLE commit point — monotone guard (M1 non-decreasing by construction, 207b608f).
        // Promote the best candidate; if none strictly improved, the live best is left untouched.
        if improved && cand_conn < best_connectivity {
            partition_host_refine.copy_from_slice(&cand_part);
            nodes_in_part_host.copy_from_slice(&cand_nip);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            best_connectivity = cand_conn;
            best_partition_host.copy_from_slice(&cand_part);
            best_nodes_in_part_host.copy_from_slice(&cand_nip);
        }
        let _ = total_pins_vc;
        // Downstream phases (post_ils_polish, output) restart explicitly from the committed live best.
    }
    // ==== END B_VCYCLE_MUTATION ====

    
    // The single-level V-cycle above (modes 0..8) merges at most PAIRS (<=2 nodes/super-node) and was shown
    
    // the FROZEN elite over VC_LEVELS levels of INTRA-BLOCK matching (block-preserving, 6724a1e2), forming
    // super-nodes of up to 2^VC_LEVELS original nodes, then UNCOARSENS level-by-level running one greedy FM pass
    // at each scale. A single FM move at a coarse level relocates a WHOLE cluster (<=2^L nodes) across the cut ->
    // a basin unreachable by single-vertex FM (P18/refinement), the 2-node single-level V-cycle, or the
    // construction-time NLEVEL_FM (different caps/scales, never re-applied to the post-pipeline elite).
    // FLOOR GUARANTEED (024acb93): the coarsening is HIERARCHICAL (level lvl ids nest inside level lvl+1 ids) and
    // block-preserving, so the coarse connectivity at any level equals the fine connectivity of the projected
    // partition (a hyperedge spans the same set of blocks counted over nodes or super-nodes). Initialising the
    // coarsest partition from the parent gives conn == parent_conn EXACTLY; every FM move is applied ONLY if it
    // strictly reduces conn AND keeps balance valid (EXACT framework tolerance: nip[q]+sz <= max_part_size,
    // nip[pc]-sz >= 1 — copied verbatim from the proven block above) -> conn is monotone non-increasing across
    // all levels -> final conn <= parent_conn. Single STRICT-improve commit. INTRA-BLOCK only (NO cross-block:
    
    // the distinct basin).
    {
        const VC_LEVELS: usize = 3;
        let n_mv = challenge.num_nodes as usize;
        let nh_mv = challenge.num_hyperedges as usize;
        let np_mv = num_parts_usize;
        let max_ps_mv = challenge.max_part_size as i32;

        // FROZEN parent (the only partition any candidate reads)
        let parent_part_mv: Vec<i32> = best_partition_host.clone();
        let parent_conn_mv = best_connectivity;

        // ---- COARSENING: build up to VC_LEVELS levels of intra-block pair matching ----
        // level_id[lvl][v] = compact id (0..nc) of original node v at level lvl. level 0 = identity.
        let mut level_id: Vec<Vec<i32>> = Vec::with_capacity(VC_LEVELS + 1);
        let mut level_nc: Vec<usize> = Vec::with_capacity(VC_LEVELS + 1);
        level_id.push((0..n_mv as i32).collect());
        level_nc.push(n_mv);

        // scratch reused across levels (sized to max coarse id = n_mv)
        let mut seen_stamp = vec![0u32; n_mv];
        let mut share_cnt = vec![0i64; n_mv];
        let mut share_stamp = vec![0u32; n_mv];
        let mut stamp = 0u32;

        for lvl in 0..VC_LEVELS {
            let nc = level_nc[lvl];
            // block of each current node (intra-block matching reads the parent block; consistent because every
            // merge so far has been intra-block, so all originals under a coarse id share the parent block)
            let mut block_cur = vec![-1i32; nc];
            for v in 0..n_mv { let c = level_id[lvl][v] as usize; if block_cur[c] < 0 { block_cur[c] = parent_part_mv[v]; } }

            // hedge -> distinct current ids ; current node -> incident hedges
            let mut hedge_reps: Vec<Vec<usize>> = Vec::with_capacity(nh_mv);
            for h in 0..nh_mv {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut reps: Vec<usize> = Vec::new();
                for k in hs..he {
                    let c = level_id[lvl][hyperedge_nodes_host[k] as usize] as usize;
                    if seen_stamp[c] != stamp { seen_stamp[c] = stamp; reps.push(c); }
                }
                hedge_reps.push(reps);
            }
            let mut cnode_hedges: Vec<Vec<usize>> = vec![Vec::new(); nc];
            for h in 0..nh_mv { for &c in &hedge_reps[h] { cnode_hedges[c].push(h); } }

            // greedy intra-block heavy-edge matching (deterministic: node-id ASC scan, heavy-edge score, id tiebreak)
            let mut matched = vec![false; nc];
            let mut rep = vec![usize::MAX; nc]; // partner root; usize::MAX if unmatched (self)
            let mut n_matched = 0usize;
            let cap = nc / 2;
            let mut cand: Vec<usize> = Vec::new();
            for u in 0..nc {
                if matched[u] || n_matched >= cap { continue; }
                let bu = block_cur[u];
                if bu < 0 { continue; }
                cand.clear();
                stamp = stamp.wrapping_add(1);
                for &h in &cnode_hedges[u] {
                    for &w in &hedge_reps[h] {
                        if w == u || matched[w] || block_cur[w] != bu { continue; }
                        if share_stamp[w] != stamp { share_stamp[w] = stamp; share_cnt[w] = 1; cand.push(w); }
                        else { share_cnt[w] += 1; }
                    }
                }
                let mut best_w = usize::MAX;
                let mut best_s = 0i64;
                for &w in &cand {
                    let s = share_cnt[w];
                    if s > best_s || (s == best_s && (best_w == usize::MAX || w < best_w)) {
                        best_s = s; best_w = w;
                    }
                }
                if best_w != usize::MAX {
                    matched[u] = true; matched[best_w] = true;
                    rep[u] = u; rep[best_w] = u;
                    n_matched += 1;
                }
            }

            // compact next-level ids (root of each merge group -> fresh contiguous id)
            let mut newid = vec![-1i32; nc];
            let mut nxt_nc = 0usize;
            for c in 0..nc {
                let root = if rep[c] == usize::MAX { c } else { rep[c] };
                if newid[root] < 0 { newid[root] = nxt_nc as i32; nxt_nc += 1; }
            }
            let mut nxt: Vec<i32> = vec![0i32; n_mv];
            for v in 0..n_mv {
                let c = level_id[lvl][v] as usize;
                let root = if rep[c] == usize::MAX { c } else { rep[c] };
                nxt[v] = newid[root];
            }
            level_id.push(nxt);
            level_nc.push(nxt_nc);
            if nxt_nc == nc { break; } // nothing merged -> deeper levels identical, stop
        }

        // ---- UNCOARSENING with one FM pass at each level, coarsest -> finest ----
        let mut orig_block: Vec<i32> = parent_part_mv.clone();
        let n_levels_built = level_nc.len(); // 1 (orig) + actual coarsening levels
        let mut pcnt_mv = vec![0i32; np_mv];
        let mut gain_mv = vec![0i32; np_mv];
        for lvl in (0..n_levels_built).rev() {
            let nc = level_nc[lvl];
            // block + size per coarse node from current orig_block (all originals under a coarse id share a block
            // because the coarsening is hierarchical and block-preserving)
            let mut block_c = vec![-1i32; nc];
            let mut size_c = vec![0i32; nc];
            for v in 0..n_mv {
                let c = level_id[lvl][v] as usize;
                size_c[c] += 1;
                if block_c[c] < 0 { block_c[c] = orig_block[v]; }
            }
            let mut nip_c = vec![0i32; np_mv];
            for c in 0..nc {
                let b = block_c[c];
                if b >= 0 && (b as usize) < np_mv { nip_c[b as usize] += size_c[c]; }
            }
            // hedge -> distinct coarse ids ; coarse node -> hedges
            let mut hedge_reps: Vec<Vec<usize>> = Vec::with_capacity(nh_mv);
            for h in 0..nh_mv {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut reps: Vec<usize> = Vec::new();
                for k in hs..he {
                    let c = level_id[lvl][hyperedge_nodes_host[k] as usize] as usize;
                    if seen_stamp[c] != stamp { seen_stamp[c] = stamp; reps.push(c); }
                }
                hedge_reps.push(reps);
            }
            let mut cnode_hedges: Vec<Vec<usize>> = vec![Vec::new(); nc];
            for h in 0..nh_mv { for &c in &hedge_reps[h] { cnode_hedges[c].push(h); } }

            // one greedy FM pass (deterministic node order 0..nc) — same gain model as the single-level block
            for c in 0..nc {
                let pc = block_c[c];
                if pc < 0 || (pc as usize) >= np_mv { continue; }
                let sz = size_c[c];
                if sz == 0 { continue; }
                gain_mv.fill(0);
                for &h in &cnode_hedges[c] {
                    pcnt_mv.fill(0);
                    for &r in &hedge_reps[h] {
                        let pq = block_c[r];
                        if pq >= 0 && (pq as usize) < np_mv { pcnt_mv[pq as usize] += 1; }
                    }
                    let only_pc = pcnt_mv[pc as usize] == 1;
                    for q in 0..np_mv {
                        if q == pc as usize { continue; }
                        let mut d = if only_pc { -1i32 } else { 0i32 };
                        if pcnt_mv[q] == 0 { d += 1; }
                        gain_mv[q] += d;
                    }
                }
                let mut best_q = usize::MAX;
                let mut best_d = 0i32;
                for q in 0..np_mv {
                    if q == pc as usize { continue; }
                    let d = gain_mv[q];
                    if d < best_d {
                        let pc_new = nip_c[pc as usize] - sz;
                        let q_new = nip_c[q] + sz;
                        if pc_new >= 1 && q_new <= max_ps_mv {
                            best_d = d; best_q = q;
                        }
                    }
                }
                if best_q != usize::MAX {
                    nip_c[pc as usize] -= sz;
                    nip_c[best_q] += sz;
                    block_c[c] = best_q as i32;
                }
            }
            // propagate coarse blocks down to original nodes for the next (finer) level
            for v in 0..n_mv { orig_block[v] = block_c[level_id[lvl][v] as usize]; }
        }

        // ---- final connectivity of the deep-mutated partition (O(total_pins)) ----
        let mut conn_mv = 0i32;
        let mut pseen_mv = vec![false; np_mv];
        for h in 0..nh_mv {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            let mut distinct = 0usize;
            for k in hs..he {
                let p = orig_block[hyperedge_nodes_host[k] as usize];
                if p >= 0 && (p as usize) < np_mv {
                    let pu = p as usize;
                    if !pseen_mv[pu] { pseen_mv[pu] = true; distinct += 1; }
                }
            }
            if distinct > 1 { conn_mv += (distinct - 1) as i32; }
            for k in hs..he {
                let p = orig_block[hyperedge_nodes_host[k] as usize];
                if p >= 0 && (p as usize) < np_mv { pseen_mv[p as usize] = false; }
            }
        }

        // STRICT-improve single commit (floor guaranteed: conn_mv <= parent_conn_mv by construction)
        if conn_mv < parent_conn_mv && conn_mv < best_connectivity {
            let mut nip_final = vec![0i32; np_mv];
            for v in 0..n_mv {
                let b = orig_block[v];
                if b >= 0 && (b as usize) < np_mv { nip_final[b as usize] += 1; }
            }
            partition_host_refine.copy_from_slice(&orig_block);
            nodes_in_part_host.copy_from_slice(&nip_final);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            best_connectivity = conn_mv;
            best_partition_host.copy_from_slice(&orig_block);
            best_nodes_in_part_host.copy_from_slice(&nip_final);
        }
    }
    // ==== END B_MEMETIC_VCYCLE_BLOCK ====


    for _ in 0..post_ils_polish {
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
                .launch(cfg.clone())?;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        let mut rng_state = 11223344u64;
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                if gain > 0 {
                    valid_moves.push((node, key));
                } else if gain == 0 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                    if (rng_state % 20) == 0 {
                        let new_key = (0 << 16) | (key & 0xFFFF);
                        valid_moves.push((node, new_key));
                    }
                }
            }
        }
        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let polish_limit = 100000usize;
        let k_base = std::cmp::min(valid_moves.len(), polish_limit);
        let polish_extra = extra_window / 3;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(polish_extra));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        let slack = slack_mid;

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

        let moves_executed = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
        if moves_executed == 0 {
            break;
        }
    }

    // ==== KICK FM gain-negatif BORNE ====
    
    // FM gain-negatif borne deterministe sur noeuds frontiere, puis re-polish.
    // Garde stricte: revert-to-best si conn post-kick >= conn pre-kick. 0-RNG.
    
    
    {
        let kick_max_moves = 30usize;
        let kick_repolish = 15usize;

        let saved_part_kick: Vec<i32> = partition_host_refine.clone();
        let saved_nip_kick: Vec<i32> = nodes_in_part_host.clone();

        let nh_k = challenge.num_hyperedges as usize;
        let np_k = num_parts_usize;
        let nn_k = challenge.num_nodes as usize;

        let mut conn_pre_kick = 0i32;
        {
            let mut pseen_k = vec![false; np_k];
            for h in 0..nh_k {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut distinct = 0usize;
                for ki in hs..he {
                    let nd = hyperedge_nodes_host[ki] as usize;
                    let p = partition_host_refine[nd];
                    if p >= 0 && (p as usize) < np_k {
                        let pu = p as usize;
                        if !pseen_k[pu] { pseen_k[pu] = true; distinct += 1; }
                    }
                }
                if distinct > 1 { conn_pre_kick += (distinct - 1) as i32; }
                for ki in hs..he {
                    let p = partition_host_refine[hyperedge_nodes_host[ki] as usize];
                    if p >= 0 && (p as usize) < np_k { pseen_k[p as usize] = false; }
                }
            }
        }

        // Identify frontier nodes: in at least one cut hyperedge; sort by cut-degree DESC, id ASC
        let mut node_cut_deg_k: Vec<usize> = vec![0usize; nn_k];
        {
            let mut pseen_k = vec![false; np_k];
            for h in 0..nh_k {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut distinct = 0usize;
                for ki in hs..he {
                    let nd = hyperedge_nodes_host[ki] as usize;
                    let p = partition_host_refine[nd];
                    if p >= 0 && (p as usize) < np_k {
                        let pu = p as usize;
                        if !pseen_k[pu] { pseen_k[pu] = true; distinct += 1; }
                    }
                }
                let is_cut = distinct > 1;
                for ki in hs..he {
                    let p = partition_host_refine[hyperedge_nodes_host[ki] as usize];
                    if p >= 0 && (p as usize) < np_k { pseen_k[p as usize] = false; }
                }
                if is_cut {
                    for ki in hs..he {
                        let nd = hyperedge_nodes_host[ki] as usize;
                        node_cut_deg_k[nd] += 1;
                    }
                }
            }
        }

        let mut frontier_kick: Vec<(usize, usize)> = (0..nn_k)
            .filter(|&nd| node_cut_deg_k[nd] > 0)
            .map(|nd| (node_cut_deg_k[nd], nd))
            .collect();
        frontier_kick.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        // Apply bounded kick: force top frontier nodes to (cur_part+1)%num_parts (deterministic)
        let kick_count = std::cmp::min(frontier_kick.len(), kick_max_moves);
        for &(_, nd) in &frontier_kick[..kick_count] {
            let cur_p = partition_host_refine[nd] as usize;
            let tgt_p = (cur_p + 1) % np_k;
            if nodes_in_part_host[tgt_p] < challenge.max_part_size as i32 {
                nodes_in_part_host[cur_p] -= 1;
                nodes_in_part_host[tgt_p] += 1;
                partition_host_refine[nd] = tgt_p as i32;
            }
        }
        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;

        // Re-polish after kick (same logic as post_ils_polish)
        for _ in 0..kick_repolish {
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
                    .launch(cfg.clone())?;
            }

            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
            valid_moves.clear();
            let mut rng_state_kr = 11223344u64;
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key as u32 != 0x80000000 {
                    let gain = key >> 16;
                    if gain > 0 {
                        valid_moves.push((node, key));
                    } else if gain == 0 {
                        rng_state_kr = rng_state_kr.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                        if (rng_state_kr % 20) == 0 {
                            let new_key = (0 << 16) | (key & 0xFFFF);
                            valid_moves.push((node, new_key));
                        }
                    }
                }
            }
            if valid_moves.is_empty() {
                break;
            }

            let cmp_kr = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
            let k_base_kr = std::cmp::min(valid_moves.len(), 100_000usize);
            let kick_extra_kr = extra_window / 3;
            let k_cand_kr = std::cmp::min(valid_moves.len(), k_base_kr.saturating_add(kick_extra_kr));

            if k_cand_kr > 1 {
                valid_moves.select_nth_unstable_by(k_cand_kr - 1, cmp_kr);
                valid_moves[..k_cand_kr].sort_unstable_by(cmp_kr);
            } else {
                valid_moves[..k_cand_kr].sort_unstable_by(cmp_kr);
            }

            tgt_used.fill(0);
            for p in 0..num_parts_usize {
                let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
                tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack_mid));
            }

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for &(node, key) in valid_moves[..k_cand_kr].iter() {
                if sorted_move_nodes.len() >= k_base_kr {
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
                let take = std::cmp::min(k_base_kr, k_cand_kr);
                sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
                sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
            }

            let moves_ex = simulate_execute_moves(
                &mut partition_host_refine,
                &mut nodes_in_part_host,
                &sorted_move_nodes,
                &sorted_move_parts,
            );

            if moves_ex > 0 {
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            }
            if moves_ex == 0 {
                break;
            }
        }

        // Compute conn post kick+repolish
        let mut conn_post_kick = 0i32;
        {
            let mut pseen_k = vec![false; np_k];
            for h in 0..nh_k {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut distinct = 0usize;
                for ki in hs..he {
                    let nd = hyperedge_nodes_host[ki] as usize;
                    let p = partition_host_refine[nd];
                    if p >= 0 && (p as usize) < np_k {
                        let pu = p as usize;
                        if !pseen_k[pu] { pseen_k[pu] = true; distinct += 1; }
                    }
                }
                if distinct > 1 { conn_post_kick += (distinct - 1) as i32; }
                for ki in hs..he {
                    let p = partition_host_refine[hyperedge_nodes_host[ki] as usize];
                    if p >= 0 && (p as usize) < np_k { pseen_k[p as usize] = false; }
                }
            }
        }

        // Guard: revert-to-best if not strictly improved (ensures Q >= 268,642 floor)
        if conn_post_kick >= conn_pre_kick {
            partition_host_refine.copy_from_slice(&saved_part_kick);
            nodes_in_part_host.copy_from_slice(&saved_nip_kick);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        } else {
            if conn_post_kick < best_connectivity {
                best_connectivity = conn_post_kick;
                best_partition_host.copy_from_slice(&partition_host_refine);
                best_nodes_in_part_host.copy_from_slice(&nodes_in_part_host);
            }
        }
    }
    // ==== END KICK FM gain-negatif BORNE ====

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
    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
    stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

    let post_balance_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 128) as usize)
        .unwrap_or(base_post_balance);

    for _ in 0..post_balance_rounds {
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
                .launch(cfg.clone())?;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        let mut rng_state = 55667788u64;
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                if gain > 0 {
                    valid_moves.push((node, key));
                } else if gain == 0 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                    if (rng_state % 20) == 0 {
                        let new_key = (0 << 16) | (key & 0xFFFF);
                        valid_moves.push((node, new_key));
                    }
                }
            }
        }
        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let mut k_base = valid_moves.len();
        let adaptive_limit = move_limit / 2;
        if k_base > adaptive_limit {
            k_base = adaptive_limit;
        }

        let post_extra = extra_window / 3;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(post_extra));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        let slack = slack_mid;

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

        let mut moves_executed = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }

        if moves_executed == 0 && k_cand > k_base {
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            let tail = &valid_moves[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            sorted_move_nodes.extend(tail.iter().take(take).map(|(n, _)| *n as i32));
            sorted_move_parts.extend(tail.iter().take(take).map(|(_, key)| (key & 63) as i32));

            if !sorted_move_nodes.is_empty() {
                moves_executed = simulate_execute_moves(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &sorted_move_nodes,
                    &sorted_move_parts,
                );
                if moves_executed > 0 {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
            }
        }

        let _ = (moves_executed, total_moves_executed, total_perturbations);
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
        10, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    // ==== B_LAHC_ESCAPE i77 (TERMINAL position — after do_swap_phase!, before memcpy_dtov) ====
    
    // i76 regression (-55) was caused by LAHC at L1690 BEFORE post_ils_polish: polish re-converged to worse basin.
    // Terminal position: snapshot-best/restore guarantees plancher 268,643. DET (0-RNG, id-ASC boundary order).
    {
        // Rapatriate GPU final state (do_swap_phase! may have modified d_partition/d_nodes_in_part on GPU)
        stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
        stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

        let n_lahc = challenge.num_nodes as usize;
        let m_lahc = challenge.num_hyperedges as usize;
        let kc = num_parts_usize.min(64);
        let max_part_lahc = challenge.max_part_size as i32;

        // Build node->hedge incidence from hyperedge CSR (O(total_pins), one-time before LAHC loop)
        let mut node_hedge_list: Vec<Vec<u32>> = vec![Vec::new(); n_lahc];
        for h in 0..m_lahc {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            for k in hs..he {
                node_hedge_list[hyperedge_nodes_host[k] as usize].push(h as u32);
            }
        }

        // Build hedge_part_cnt from final GPU-rapatriated partition
        let mut hedge_part_cnt: Vec<[u8; 64]> = vec![[0u8; 64]; m_lahc];
        for v in 0..n_lahc {
            let p = partition_host_refine[v] as usize;
            if p < kc {
                for &h in &node_hedge_list[v] {
                    hedge_part_cnt[h as usize][p] =
                        hedge_part_cnt[h as usize][p].saturating_add(1);
                }
            }
        }

        // Connectivity from hedge_part_cnt (consistent with final partition post-do_swap_phase!)
        let mut lahc_conn: i32 = 0;
        for h in 0..m_lahc {
            let parts = hedge_part_cnt[h][..kc].iter().filter(|&&c| c > 0).count();
            if parts > 1 { lahc_conn += (parts - 1) as i32; }
        }

        // Boundary verts: adjacent to >=1 cut hedge; id-ASC order by construction
        let boundary_verts: Vec<usize> = (0..n_lahc)
            .filter(|&v| {
                node_hedge_list[v].iter().any(|&h| {
                    hedge_part_cnt[h as usize][..kc].iter().filter(|&&c| c > 0).count() > 1
                })
            })
            .collect();
        let n_boundary = boundary_verts.len();

        if n_boundary > 0 {
            let lahc_l: usize = hyperparameters
                .as_ref()
                .and_then(|p| p.get("lahc_history_len").and_then(|v| v.as_i64()))
                .map(|v| v.clamp(5, 200) as usize)
                .unwrap_or(20);
            let lahc_cap: usize = hyperparameters
                .as_ref()
                .and_then(|p| p.get("lahc_iters").and_then(|v| v.as_i64()))
                .map(|v| v.clamp(100, 2_000_000) as usize)
                .unwrap_or(n_boundary);

            let mut hist: Vec<i32> = vec![lahc_conn; lahc_l];
            let mut best_conn = lahc_conn;
            let mut best_part: Vec<i32> = partition_host_refine.clone();
            let mut best_nip: Vec<i32> = nodes_in_part_host.clone();

            for t in 0..lahc_cap {
                let v = boundary_verts[t % n_boundary];
                let cur = partition_host_refine[v] as usize;
                if cur >= kc || nodes_in_part_host[cur] <= 1 { continue; }

                // Best target: max gain (connectivity reduction), tiebreak tgt id ASC
                let mut best_gain = i32::MIN;
                let mut best_tgt = kc; // sentinel = no valid target
                for tgt in 0..kc {
                    if tgt == cur || nodes_in_part_host[tgt] >= max_part_lahc { continue; }
                    let mut g: i32 = 0;
                    for &h in &node_hedge_list[v] {
                        let cnt = &hedge_part_cnt[h as usize];
                        if cnt[cur] == 1 { g += 1; }
                        if cnt[tgt] == 0 { g -= 1; }
                    }
                    if g > best_gain || (g == best_gain && tgt < best_tgt) {
                        best_gain = g;
                        best_tgt = tgt;
                    }
                }
                if best_tgt >= kc { continue; }

                let new_conn = lahc_conn - best_gain;
                let slot = t % lahc_l;

                // Accept if new_conn <= hist[slot] (LAHC) OR new_conn <= lahc_conn (neutral/improve)
                if new_conn <= hist[slot] || new_conn <= lahc_conn {
                    for &h in &node_hedge_list[v] {
                        let cnt = &mut hedge_part_cnt[h as usize];
                        cnt[cur] = cnt[cur].saturating_sub(1);
                        cnt[best_tgt] = cnt[best_tgt].saturating_add(1);
                    }
                    nodes_in_part_host[cur] -= 1;
                    nodes_in_part_host[best_tgt] += 1;
                    partition_host_refine[v] = best_tgt as i32;

                    hist[slot] = lahc_conn; // record pre-move cost in history slot
                    lahc_conn = new_conn;

                    if lahc_conn < best_conn {
                        best_conn = lahc_conn;
                        best_part.copy_from_slice(&partition_host_refine);
                        best_nip.copy_from_slice(&nodes_in_part_host);
                    }
                } else {
                    hist[slot] = lahc_conn;
                }
            }

            // Restore best LAHC snapshot (initial if no improvement — plancher 268,643 garanti)
            partition_host_refine.copy_from_slice(&best_part);
            nodes_in_part_host.copy_from_slice(&best_nip);

            // Push best back to GPU so memcpy_dtov reads the LAHC-improved result (NOTHING after this)
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
    }
    // ==== END B_LAHC_ESCAPE i77 (TERMINAL) ====

    // ==== B_PATH_RELINK i86 (TERMINAL — ROUND-ROBIN CYCLÉ: {g1,g2} + 2 PASSES REVERSE g[k]→best) ====
    
    // {guide[1], guide[2]} tant qu'un CYCLE complet trouve ≥1 move accepté (Δconn≤0), cap dur max_cycles.
    // Motivation: gains non-additifs (i81 lesson `ed41ae61`): état évolué ré-expose moves résiduels
    // sur élites déjà visitées → cycles répétés captent du Q dans le slack ~156s SANS jamais toucher g[3].
    // Sweep: max_cycles ∈ {3, 6} via HP "pr_max_cycles". Floor = LAHC output GARANTI (monotone guard).
    {
        if elite_count >= 2 {
            let max_cycles: usize = hyperparameters
                .as_ref()
                .and_then(|p| p.get("pr_max_cycles").and_then(|v| v.as_i64()))
                .map(|v| v.clamp(1, 20) as usize)
                .unwrap_or(6);

            let mut elite_order_pr: Vec<usize> = (0..elite_count).collect();
            elite_order_pr.sort_unstable_by(|&a, &b| {
                elite_scores[a].cmp(&elite_scores[b]).then(a.cmp(&b))
            });

            let n_pr = challenge.num_nodes as usize;
            let nh_pr = challenge.num_hyperedges as usize;
            let np_pr = num_parts_usize;
            let max_ps_pr = challenge.max_part_size as i32;
            // i100 B_UNCONSTRAINED_REBALANCE: slack 3% transitoire sur VETO PR (cross-poll P₁₁ t23/i101 + P₁₂ t24/i95)
            let slack_max_pr = max_ps_pr + (max_ps_pr * 3 + 99) / 100;

            // Build CSR once (shared by all passes)
            let mut nd_deg_pr = vec![0usize; n_pr];
            for h in 0..nh_pr {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_pr { nd_deg_pr[nd] += 1; }
                }
            }
            let mut nd_off_pr = vec![0usize; n_pr + 1];
            for i in 0..n_pr { nd_off_pr[i + 1] = nd_off_pr[i] + nd_deg_pr[i]; }
            let mut nd_hedges_pr = vec![0usize; nd_off_pr[n_pr]];
            {
                let mut cur_off = nd_off_pr[..n_pr].to_vec();
                for h in 0..nh_pr {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        if nd < n_pr {
                            nd_hedges_pr[cur_off[nd]] = h;
                            cur_off[nd] += 1;
                        }
                    }
                }
            }

            // Initial connectivity (LAHC output = floor for all passes)
            let floor_conn_all = {
                let mut pseen = vec![false; np_pr];
                let mut c = 0i32;
                for h in 0..nh_pr {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    let mut d = 0usize;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        let p = partition_host_refine[nd];
                        if p >= 0 && (p as usize) < np_pr {
                            let pu = p as usize;
                            if !pseen[pu] { pseen[pu] = true; d += 1; }
                        }
                    }
                    if d > 1 { c += (d - 1) as i32; }
                    for k in hs..he {
                        let p = partition_host_refine[hyperedge_nodes_host[k] as usize];
                        if p >= 0 && (p as usize) < np_pr { pseen[p as usize] = false; }
                    }
                }
                c
            };

            // Global best across all cycles (floor = LAHC output)
            let mut global_best_conn = floor_conn_all;
            let mut global_best_part = partition_host_refine.clone();
            let mut global_best_nip = nodes_in_part_host.clone();

            // Current source partition (advances as cycles improve it)
            let mut cur_part = partition_host_refine.clone();
            let mut cur_nip = nodes_in_part_host.clone();

            // Round-robin cycles: each cycle = [g1..g7]. Stop when a full cycle finds 0 accepted moves.
            
            
            
            'cycles: for _cycle in 0..max_cycles {
                let mut cycle_accepted = 0usize;

                // One pass per guide in cycle order: g1..g9 (slot 1..9). g3+ GPU-scored. CEILING pool=10.
                for pass_idx in 0..9usize {
                    let guide_slot = pass_idx + 1;
                    if elite_order_pr.len() <= guide_slot { break; }
                    let guide_off = elite_order_pr[guide_slot] * n_pr;

                    // GPU-SCORED PASS for guide_slot >= 3: O(|D|×deg) CPU ≈ 300s on t25.
                    // GPU kernel score_pr_g3_warp_50k reduces to O(1) wall-clock. g4 D even larger → GPU mandatory.
                    if guide_slot >= 3 {
                        // Build D = {v : cur_part[v] ≠ g[k][v]} on CPU
                        let mut g3c_nodes: Vec<i32> = Vec::new();
                        let mut g3c_tgts:  Vec<i32> = Vec::new();
                        for v in 0..n_pr {
                            let p_src = cur_part[v];
                            let p_g3  = elite_flat_host[guide_off + v];
                            if p_src != p_g3 && p_src >= 0 && p_g3 >= 0
                                && (p_src as usize) < np_pr && (p_g3 as usize) < np_pr
                            {
                                g3c_nodes.push(v as i32);
                                g3c_tgts.push(p_g3);
                            }
                        }
                        let nd_g3c = g3c_nodes.len();
                        if nd_g3c > 0 {
                            // Upload cur_part to GPU, recompute edge flags
                            stream.memcpy_htod(&cur_part, &mut d_partition)?;
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
                            let d_g3c_nodes = stream.memcpy_stod(&g3c_nodes)?;
                            let d_g3c_tgts  = stream.memcpy_stod(&g3c_tgts)?;
                            let mut d_g3c_delta = stream.alloc_zeros::<i32>(nd_g3c)?;
                            let g3c_blocks = ((nd_g3c as u32) + 3) / 4;
                            let g3c_cfg = LaunchConfig { grid_dim: (g3c_blocks, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                            unsafe {
                                stream
                                    .launch_builder(&score_pr_g3_warp_kernel)
                                    .arg(&(nd_g3c as i32))
                                    .arg(&d_g3c_nodes)
                                    .arg(&d_g3c_tgts)
                                    .arg(&challenge.d_node_hyperedges)
                                    .arg(&challenge.d_node_offsets)
                                    .arg(&d_partition)
                                    .arg(&d_edge_flags_all)
                                    .arg(&d_edge_flags_double)
                                    .arg(&mut d_g3c_delta)
                                    .launch(g3c_cfg)?;
                            }
                            let delta_g3c: Vec<i32> = stream.memcpy_dtov(&d_g3c_delta)?;
                            let mut g3c_sorted: Vec<(i32, usize, i32)> = g3c_nodes.iter().enumerate()
                                .map(|(i, &v)| (delta_g3c[i], v as usize, g3c_tgts[i]))
                                .collect();
                            g3c_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                            // Greedy CPU commit: exact Δconn recomputed per accept (ε=0, DET, sequential)
                            let mut g3c_part = cur_part.clone();
                            let mut g3c_nip  = cur_nip.clone();
                            let mut g3c_conn = {
                                let mut pseen = vec![false; np_pr];
                                let mut c = 0i32;
                                for h in 0..nh_pr {
                                    let hs = hedge_offsets_host[h] as usize;
                                    let he = hedge_offsets_host[h + 1] as usize;
                                    let mut d = 0usize;
                                    for k in hs..he {
                                        let nd = hyperedge_nodes_host[k] as usize;
                                        let p = g3c_part[nd];
                                        if p >= 0 && (p as usize) < np_pr {
                                            let pu = p as usize;
                                            if !pseen[pu] { pseen[pu] = true; d += 1; }
                                        }
                                    }
                                    if d > 1 { c += (d - 1) as i32; }
                                    for k in hs..he {
                                        let p = g3c_part[hyperedge_nodes_host[k] as usize];
                                        if p >= 0 && (p as usize) < np_pr { pseen[p as usize] = false; }
                                    }
                                }
                                c
                            };
                            let floor_g3c = g3c_conn;
                            let mut g3c_best_conn = g3c_conn;
                            let mut g3c_best_part = g3c_part.clone();
                            let mut g3c_best_nip  = g3c_nip.clone();
                            let mut g3c_accepted  = 0usize;
                            for &(_, v, p_tgt_i32) in &g3c_sorted {
                                let p_curr_i32 = g3c_part[v];
                                if p_curr_i32 == p_tgt_i32 { continue; }
                                let p_curr = p_curr_i32 as usize;
                                let p_tgt  = p_tgt_i32 as usize;
                                if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                                if g3c_nip[p_curr] <= 1 { continue; }
                                if g3c_nip[p_tgt] >= max_ps_pr { continue; }
                                let mut delta = 0i32;
                                for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                                    let h  = nd_hedges_pr[hi];
                                    let es = hedge_offsets_host[h] as usize;
                                    let ee = hedge_offsets_host[h + 1] as usize;
                                    let mut cnt_c = 0i32;
                                    let mut cnt_t = 0i32;
                                    for kp in es..ee {
                                        let nd = hyperedge_nodes_host[kp] as usize;
                                        if nd == v { continue; }
                                        let p = g3c_part[nd];
                                        if p == p_curr_i32 { cnt_c += 1; }
                                        else if p == p_tgt_i32 { cnt_t += 1; }
                                    }
                                    if cnt_c == 0 { delta -= 1; }
                                    if cnt_t == 0 { delta += 1; }
                                }
                                if delta > 0 { continue; }
                                g3c_nip[p_curr] -= 1;
                                g3c_nip[p_tgt]  += 1;
                                g3c_part[v] = p_tgt_i32;
                                g3c_conn += delta;
                                g3c_accepted += 1;
                                if g3c_conn < g3c_best_conn {
                                    g3c_best_conn = g3c_conn;
                                    g3c_best_part.copy_from_slice(&g3c_part);
                                    g3c_best_nip.copy_from_slice(&g3c_nip);
                                }
                            }
                            cycle_accepted += g3c_accepted;
                            if g3c_best_conn < global_best_conn {
                                global_best_conn = g3c_best_conn;
                                global_best_part = g3c_best_part.clone();
                                global_best_nip  = g3c_best_nip.clone();
                            }
                            // Advance cur_part for next cycle
                            if g3c_best_conn < floor_g3c {
                                cur_part = g3c_best_part;
                                cur_nip  = g3c_best_nip;
                            }
                        }
                        continue; // skip CPU scoring path; proceed to next pass_idx
                    }

                    // Compute initial conn of cur_part for this pass
                    let mut pass_conn = {
                        let mut pseen = vec![false; np_pr];
                        let mut c = 0i32;
                        for h in 0..nh_pr {
                            let hs = hedge_offsets_host[h] as usize;
                            let he = hedge_offsets_host[h + 1] as usize;
                            let mut d = 0usize;
                            for k in hs..he {
                                let nd = hyperedge_nodes_host[k] as usize;
                                let p = cur_part[nd];
                                if p >= 0 && (p as usize) < np_pr {
                                    let pu = p as usize;
                                    if !pseen[pu] { pseen[pu] = true; d += 1; }
                                }
                            }
                            if d > 1 { c += (d - 1) as i32; }
                            for k in hs..he {
                                let p = cur_part[hyperedge_nodes_host[k] as usize];
                                if p >= 0 && (p as usize) < np_pr { pseen[p as usize] = false; }
                            }
                        }
                        c
                    };

                    let floor_pass = pass_conn;
                    let mut best_pass_conn = pass_conn;
                    let mut best_pass_part = cur_part.clone();
                    let mut best_pass_nip = cur_nip.clone();

                    let disagree_set: Vec<usize> = (0..n_pr)
                        .filter(|&v| {
                            let p_src = cur_part[v];
                            let p_guide = elite_flat_host[guide_off + v];
                            p_src != p_guide && p_src >= 0 && p_guide >= 0
                        })
                        .collect();

                    if disagree_set.is_empty() { continue; }

                    // Static Δconn estimate for sort
                    let mut disagree_sorted: Vec<(i32, usize, i32)> = Vec::with_capacity(disagree_set.len());
                    for &v in &disagree_set {
                        let p_curr = cur_part[v] as usize;
                        let p_tgt_i32 = elite_flat_host[guide_off + v];
                        let p_tgt = p_tgt_i32 as usize;
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                        let mut delta = 0i32;
                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                            let h = nd_hedges_pr[hi];
                            let es = hedge_offsets_host[h] as usize;
                            let ee = hedge_offsets_host[h + 1] as usize;
                            let mut cnt_c = 0i32;
                            let mut cnt_t = 0i32;
                            for kp in es..ee {
                                let nd = hyperedge_nodes_host[kp] as usize;
                                if nd == v { continue; }
                                let p = cur_part[nd];
                                if p == p_curr as i32 { cnt_c += 1; }
                                else if p == p_tgt as i32 { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }
                        disagree_sorted.push((delta, v, p_tgt_i32));
                    }
                    disagree_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                    // Greedy pass: exact Δconn recomputed, accept if ≤ 0 (ε=0)
                    let mut pass_part = cur_part.clone();
                    let mut pass_nip = cur_nip.clone();
                    let mut pass_accepted = 0usize;
                    for &(_, v, p_tgt_i32) in &disagree_sorted {
                        let p_curr = pass_part[v] as usize;
                        let p_tgt = p_tgt_i32 as usize;
                        if p_curr == p_tgt { continue; }
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                        if pass_nip[p_curr] <= 1 { continue; }
                        if pass_nip[p_tgt] >= slack_max_pr { continue; }

                        let mut delta = 0i32;
                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                            let h = nd_hedges_pr[hi];
                            let es = hedge_offsets_host[h] as usize;
                            let ee = hedge_offsets_host[h + 1] as usize;
                            let mut cnt_c = 0i32;
                            let mut cnt_t = 0i32;
                            for kp in es..ee {
                                let nd = hyperedge_nodes_host[kp] as usize;
                                if nd == v { continue; }
                                let p = pass_part[nd];
                                if p == p_curr as i32 { cnt_c += 1; }
                                else if p == p_tgt as i32 { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }

                        if delta > 0 { continue; }

                        pass_nip[p_curr] -= 1;
                        pass_nip[p_tgt] += 1;
                        pass_part[v] = p_tgt_i32;
                        pass_conn += delta;
                        pass_accepted += 1;

                        if pass_conn < best_pass_conn {
                            best_pass_conn = pass_conn;
                            best_pass_part.copy_from_slice(&pass_part);
                            best_pass_nip.copy_from_slice(&pass_nip);
                        }
                    }

                    cycle_accepted += pass_accepted;

                    // Update global best
                    if best_pass_conn < global_best_conn {
                        global_best_conn = best_pass_conn;
                        global_best_part = best_pass_part.clone();
                        global_best_nip = best_pass_nip.clone();
                    }

                    // Advance src for next pass: use best of this pass if improved
                    if best_pass_conn < floor_pass {
                        cur_part = best_pass_part;
                        cur_nip = best_pass_nip;
                    }
                }

                // Stop if this full cycle found no accepted moves (converged)
                if cycle_accepted == 0 { break 'cycles; }
            }

            // ==== REVERSE PATH-RELINK: g[k]→global_best, 2 passes ====
            // After forward round-robin converges, try reverse direction: src=g[k], guide=global_best_part.
            // Same D nodes as forward, different Δconn (computed from g[k]'s perspective).
            // Intermediate solutions on g[k]→best path may improve global_best (path-relink property).
            for rev_guide_slot in 1..=2usize {
                if elite_order_pr.len() <= rev_guide_slot { break; }
                let rev_guide_off = elite_order_pr[rev_guide_slot] * n_pr;

                // Load elite partition as reverse source
                let mut rev_part: Vec<i32> = elite_flat_host[rev_guide_off..rev_guide_off + n_pr].to_vec();

                // Compute rev_nip from rev_part
                let mut rev_nip = vec![0i32; np_pr];
                for &p in rev_part.iter() {
                    if p >= 0 && (p as usize) < np_pr { rev_nip[p as usize] += 1; }
                }

                // Initial connectivity of rev_part (g[k])
                let mut rev_conn = {
                    let mut pseen = vec![false; np_pr];
                    let mut c = 0i32;
                    for h in 0..nh_pr {
                        let hs = hedge_offsets_host[h] as usize;
                        let he = hedge_offsets_host[h + 1] as usize;
                        let mut d = 0usize;
                        for k in hs..he {
                            let nd = hyperedge_nodes_host[k] as usize;
                            let p = rev_part[nd];
                            if p >= 0 && (p as usize) < np_pr {
                                if !pseen[p as usize] { pseen[p as usize] = true; d += 1; }
                            }
                        }
                        if d > 1 { c += (d - 1) as i32; }
                        for k in hs..he {
                            let p = rev_part[hyperedge_nodes_host[k] as usize];
                            if p >= 0 && (p as usize) < np_pr { pseen[p as usize] = false; }
                        }
                    }
                    c
                };

                let mut best_rev_conn = rev_conn;
                let mut best_rev_part = rev_part.clone();
                let mut best_rev_nip = rev_nip.clone();

                // Disagreement set: nodes where g[k][v] ≠ global_best_part[v]
                // Static Δconn from g[k]'s perspective: p_curr=g[k][v], p_tgt=best[v]
                let mut disagree_rev: Vec<(i32, usize, i32)> = Vec::new();
                for v in 0..n_pr {
                    let p_curr = rev_part[v];
                    let p_tgt = global_best_part[v];
                    if p_curr == p_tgt || p_curr < 0 || p_tgt < 0 { continue; }
                    let p_curr_u = p_curr as usize;
                    let p_tgt_u = p_tgt as usize;
                    if p_curr_u >= np_pr || p_tgt_u >= np_pr { continue; }
                    let mut delta = 0i32;
                    for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                        let h = nd_hedges_pr[hi];
                        let es = hedge_offsets_host[h] as usize;
                        let ee = hedge_offsets_host[h + 1] as usize;
                        let mut cnt_c = 0i32;
                        let mut cnt_t = 0i32;
                        for kp in es..ee {
                            let nd = hyperedge_nodes_host[kp] as usize;
                            if nd == v { continue; }
                            let p = rev_part[nd];
                            if p == p_curr { cnt_c += 1; }
                            else if p == p_tgt { cnt_t += 1; }
                        }
                        if cnt_c == 0 { delta -= 1; }
                        if cnt_t == 0 { delta += 1; }
                    }
                    disagree_rev.push((delta, v, p_tgt));
                }
                disagree_rev.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                // Greedy pass: exact Δconn recomputed, accept if ≤ 0
                for &(_, v, p_tgt_i32) in &disagree_rev {
                    let p_curr_i32 = rev_part[v];
                    if p_curr_i32 == p_tgt_i32 { continue; }
                    let p_curr = p_curr_i32 as usize;
                    let p_tgt = p_tgt_i32 as usize;
                    if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                    if rev_nip[p_curr] <= 1 { continue; }
                    if rev_nip[p_tgt] >= max_ps_pr { continue; }

                    let mut delta = 0i32;
                    for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                        let h = nd_hedges_pr[hi];
                        let es = hedge_offsets_host[h] as usize;
                        let ee = hedge_offsets_host[h + 1] as usize;
                        let mut cnt_c = 0i32;
                        let mut cnt_t = 0i32;
                        for kp in es..ee {
                            let nd = hyperedge_nodes_host[kp] as usize;
                            if nd == v { continue; }
                            let p = rev_part[nd];
                            if p == p_curr_i32 { cnt_c += 1; }
                            else if p == p_tgt_i32 { cnt_t += 1; }
                        }
                        if cnt_c == 0 { delta -= 1; }
                        if cnt_t == 0 { delta += 1; }
                    }

                    if delta > 0 { continue; }

                    rev_nip[p_curr] -= 1;
                    rev_nip[p_tgt] += 1;
                    rev_part[v] = p_tgt_i32;
                    rev_conn += delta;

                    if rev_conn < best_rev_conn {
                        best_rev_conn = rev_conn;
                        best_rev_part.copy_from_slice(&rev_part);
                        best_rev_nip.copy_from_slice(&rev_nip);
                    }
                }

                // Update global best only if strictly better than all forward passes
                if best_rev_conn < global_best_conn {
                    global_best_conn = best_rev_conn;
                    global_best_part = best_rev_part;
                    global_best_nip = best_rev_nip;
                }
            }
            // ==== END REVERSE PATH-RELINK i86 ====

            // ==== g[3] FORWARD PASS — GPU SCORED ====
            // D = {v : global_best_part[v] ≠ g[3][v]}. GPU kernel (score_pr_g3_warp_50k)
            // scores all D-nodes in parallel (1 warp/node). CPU commits greedily (exact Δconn,
            // sequential, deterministic). NEVER atomic/device-commit (DET).
            // NEVER on-device loop: kernel bounded by used≤256 per warp.
            if elite_order_pr.len() > 3 {
                let g3_slot = elite_order_pr[3];
                let g3_off  = g3_slot * n_pr;

                // Build D = {v : global_best[v] ≠ g[3][v]} on CPU
                let mut g3_nodes: Vec<i32> = Vec::new();
                let mut g3_tgts:  Vec<i32> = Vec::new();
                for v in 0..n_pr {
                    let p_best = global_best_part[v];
                    let p_g3   = elite_flat_host[g3_off + v];
                    if p_best != p_g3
                        && p_best >= 0 && p_g3 >= 0
                        && (p_best as usize) < np_pr && (p_g3 as usize) < np_pr
                    {
                        g3_nodes.push(v as i32);
                        g3_tgts.push(p_g3);
                    }
                }

                let nd_g3 = g3_nodes.len();
                if nd_g3 > 0 {
                    // Upload global_best_part to GPU for precompute_edge_flags
                    stream.memcpy_htod(&global_best_part, &mut d_partition)?;
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

                    // Upload D to GPU (memcpy_stod allocates exact-size buffers)
                    let d_g3_nodes = stream.memcpy_stod(&g3_nodes)?;
                    let d_g3_tgts  = stream.memcpy_stod(&g3_tgts)?;
                    let mut d_g3_delta = stream.alloc_zeros::<i32>(nd_g3)?;

                    // Launch warp-per-node scoring: block_dim=128 → 4 D-nodes/block
                    let g3_blocks = ((nd_g3 as u32) + 3) / 4;
                    let g3_cfg = LaunchConfig {
                        grid_dim: (g3_blocks, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        stream
                            .launch_builder(&score_pr_g3_warp_kernel)
                            .arg(&(nd_g3 as i32))
                            .arg(&d_g3_nodes)
                            .arg(&d_g3_tgts)
                            .arg(&challenge.d_node_hyperedges)
                            .arg(&challenge.d_node_offsets)
                            .arg(&d_partition)
                            .arg(&d_edge_flags_all)
                            .arg(&d_edge_flags_double)
                            .arg(&mut d_g3_delta)
                            .launch(g3_cfg)?;
                    }

                    // Download GPU-computed deltas
                    let delta_g3: Vec<i32> = stream.memcpy_dtov(&d_g3_delta)?;

                    // Sort D by (delta ASC, node ASC) using GPU approximation as sort key
                    let mut g3_sorted: Vec<(i32, usize, i32)> = g3_nodes
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (delta_g3[i], v as usize, g3_tgts[i]))
                        .collect();
                    g3_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                    // Greedy CPU commit: exact Δconn recomputed per accept (ε=0, DET, sequential)
                    let mut g3_part = global_best_part.clone();
                    let mut g3_nip  = global_best_nip.clone();
                    let mut g3_conn = global_best_conn;
                    let mut g3_best_conn = g3_conn;
                    let mut g3_best_part = g3_part.clone();
                    let mut g3_best_nip  = g3_nip.clone();

                    for &(_, v, p_tgt_i32) in &g3_sorted {
                        let p_curr_i32 = g3_part[v];
                        if p_curr_i32 == p_tgt_i32 { continue; }
                        let p_curr = p_curr_i32 as usize;
                        let p_tgt  = p_tgt_i32 as usize;
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                        if g3_nip[p_curr] <= 1 { continue; }
                        if g3_nip[p_tgt] >= max_ps_pr { continue; }

                        // Exact Δconn recompute (same formula as all PR passes)
                        let mut delta = 0i32;
                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                            let h  = nd_hedges_pr[hi];
                            let es = hedge_offsets_host[h] as usize;
                            let ee = hedge_offsets_host[h + 1] as usize;
                            let mut cnt_c = 0i32;
                            let mut cnt_t = 0i32;
                            for kp in es..ee {
                                let nd = hyperedge_nodes_host[kp] as usize;
                                if nd == v { continue; }
                                let p = g3_part[nd];
                                if p == p_curr_i32 { cnt_c += 1; }
                                else if p == p_tgt_i32 { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }

                        if delta > 0 { continue; }  // ε=0 greedy (reject positive Δconn)

                        g3_nip[p_curr] -= 1;
                        g3_nip[p_tgt]  += 1;
                        g3_part[v] = p_tgt_i32;
                        g3_conn += delta;

                        if g3_conn < g3_best_conn {
                            g3_best_conn = g3_conn;
                            g3_best_part.copy_from_slice(&g3_part);
                            g3_best_nip.copy_from_slice(&g3_nip);
                        }
                    }

                    // Monotone guard: update global best only if strictly improved
                    if g3_best_conn < global_best_conn {
                        global_best_conn = g3_best_conn;
                        global_best_part = g3_best_part;
                        global_best_nip  = g3_best_nip;
                    }
                }
            }
            // ==== END g[3] FORWARD PASS GPU SCORED ====

            // ==== B_UNCONSTRAINED_REBALANCE WEAK (i100, Jetrw — cross-poll P₁₁ t23/i101 + P₁₂ t24/i95) ====
            // Restore strict balance after PATH_RELINK with slack 3% on VETO.
            // WEAK: pre-build block_verts O(n) once, commit greedily ≥1 move/round
            // ≤10 rounds. Rollback to floor (partition_host_refine unchanged) if imbalanced → 0 invalid.
            if global_best_conn < floor_conn_all {
                // Pre-build block vertex lists O(n) — avoids O(n×np) scan per round
                let mut block_verts: Vec<Vec<usize>> = vec![Vec::new(); np_pr];
                for v in 0..n_pr {
                    let p = global_best_part[v] as usize;
                    if p < np_pr { block_verts[p].push(v); }
                }

                let mut rebal_present = vec![0i32; np_pr];
                let mut rebal_seen = vec![false; np_pr];
                let balance_ok_pr = 'rebalance_pr: {
                    for _round in 0..10usize {
                        let mut all_balanced = true;
                        let mut did_move_round = false;
                        for src_p in 0..np_pr {
                            if global_best_nip[src_p] <= max_ps_pr { continue; }
                            all_balanced = false;
                            // Score vertices in overfull block only (not all n)
                            let mut cands: Vec<(i32, usize, usize)> = Vec::new();
                            for &v in &block_verts[src_p] {
                                let (best_delta, best_dst) = if pr_rebal_hoist {
                                    // Hoisted: scan each hedge ONCE per vertex.
                                    // base_delta = cnt_s-dependent terms (dst_p-invariant).
                                    // present[p] = #hedges where part p has >=1 pin (excl v).
                                    // delta(dst_p) = base_delta + (deg_v - present[dst_p]).
                                    for x in rebal_present.iter_mut() { *x = 0; }
                                    let deg_v = (nd_off_pr[v + 1] - nd_off_pr[v]) as i32;
                                    let mut base_delta = 0i32;
                                    for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                                        let h = nd_hedges_pr[hi];
                                        let es = hedge_offsets_host[h] as usize;
                                        let ee = hedge_offsets_host[h + 1] as usize;
                                        let mut cnt_s = 0i32;
                                        for x in rebal_seen.iter_mut() { *x = false; }
                                        for kp in es..ee {
                                            let nd = hyperedge_nodes_host[kp] as usize;
                                            if nd == v { continue; }
                                            let p = global_best_part[nd] as usize;
                                            if p == src_p { cnt_s += 1; }
                                            if p < np_pr { rebal_seen[p] = true; }
                                        }
                                        if cnt_s == 0 { base_delta -= 1; }
                                        for p in 0..np_pr { if rebal_seen[p] { rebal_present[p] += 1; } }
                                    }
                                    let mut bd = i32::MAX;
                                    let mut bdst = np_pr;
                                    for dst_p in 0..np_pr {
                                        if dst_p == src_p { continue; }
                                        if global_best_nip[dst_p] >= max_ps_pr { continue; }
                                        let delta = base_delta + (deg_v - rebal_present[dst_p]);
                                        if delta < bd || (delta == bd && dst_p < bdst) { bd = delta; bdst = dst_p; }
                                    }
                                    (bd, bdst)
                                } else {
                                    let mut bd = i32::MAX;
                                    let mut bdst = np_pr;
                                    for dst_p in 0..np_pr {
                                        if dst_p == src_p { continue; }
                                        if global_best_nip[dst_p] >= max_ps_pr { continue; }
                                        let mut delta = 0i32;
                                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                                            let h = nd_hedges_pr[hi];
                                            let es = hedge_offsets_host[h] as usize;
                                            let ee = hedge_offsets_host[h + 1] as usize;
                                            let mut cnt_s = 0i32;
                                            let mut cnt_d = 0i32;
                                            for kp in es..ee {
                                                let nd = hyperedge_nodes_host[kp] as usize;
                                                if nd == v { continue; }
                                                let p = global_best_part[nd] as usize;
                                                if p == src_p { cnt_s += 1; }
                                                else if p == dst_p { cnt_d += 1; }
                                            }
                                            if cnt_s == 0 { delta -= 1; }
                                            if cnt_d == 0 { delta += 1; }
                                        }
                                        if delta < bd || (delta == bd && dst_p < bdst) { bd = delta; bdst = dst_p; }
                                    }
                                    (bd, bdst)
                                };
                                if best_dst < np_pr {
                                    cands.push((best_delta, v, best_dst));
                                }
                            }
                            
                            cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                            // Commit greedily until block balanced (not just 1/round like i94)
                            for &(_, v, dst_p) in &cands {
                                if global_best_nip[src_p] <= max_ps_pr { break; }
                                if global_best_nip[dst_p] >= max_ps_pr { continue; }
                                global_best_part[v] = dst_p as i32;
                                global_best_nip[src_p] -= 1;
                                global_best_nip[dst_p] += 1;
                                // Update block_verts for subsequent rounds
                                if let Some(pos) = block_verts[src_p].iter().rposition(|&x| x == v) {
                                    block_verts[src_p].swap_remove(pos);
                                }
                                block_verts[dst_p].push(v);
                                did_move_round = true;
                            }
                        }
                        if all_balanced { break 'rebalance_pr true; }
                        if !did_move_round { break 'rebalance_pr false; }
                    }
                    global_best_nip.iter().all(|&c| c <= max_ps_pr)
                };
                if balance_ok_pr {
                    partition_host_refine.copy_from_slice(&global_best_part);
                    nodes_in_part_host.copy_from_slice(&global_best_nip);
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
                // else: balance not restored in 10 rounds → keep floor (partition_host_refine unchanged)
            }
            // ==== END B_UNCONSTRAINED_REBALANCE WEAK ====
        }
    }
    // ==== END B_PATH_RELINK i86 (TERMINAL ROUND-ROBIN + REVERSE PASSES) ====

    // ==== B_NLEVEL_FM i103 (4-level multi-scale FM — cross-poll t24/i99 +51pts DET×2, ext. 3-level i101 +131pts) ====
    // 4-level hierarchy: L1 pairs (~n/2=25k), L2 pair-of-pairs (~n/4=12.5k),
    // L3 2×2×2 (~n/8=6.25k), L4 2×2×2×2 (~n/16=3125 supernodes on 50K).
    // FM full at L4 (coarsest) → localized FM at L3 → localized FM at L2 → localized FM at L1 → fine.
    // Distinct basin from 3-level: groups of 16 (2^4) vs 8 (2^3) → different FM trajectories.
    // Gate t25: baseline ~99s << 249,540ms. Overhead ~+2-4s (FM on 3125 supernodes).
    // 
    // Balance HARD (max_ps) at all FM levels. 0 thread/rayon/spawn. Deterministic.
    {
        let n_nl   = challenge.num_nodes as usize;
        let nh_nl  = challenge.num_hyperedges as usize;
        let np_nl  = num_parts_usize.min(64);
        let max_ps_nl = challenge.max_part_size as i32;

        // Build node→hedges CSR (one-time, used at all levels).
        let mut nd_deg_nl = vec![0usize; n_nl];
        for h in 0..nh_nl {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            for k in hs..he {
                let nd = hyperedge_nodes_host[k] as usize;
                if nd < n_nl { nd_deg_nl[nd] += 1; }
            }
        }
        let mut nd_off_nl = vec![0usize; n_nl + 1];
        for i in 0..n_nl { nd_off_nl[i + 1] = nd_off_nl[i] + nd_deg_nl[i]; }
        let mut nd_hedges_nl = vec![0usize; nd_off_nl[n_nl]];
        {
            let mut cur = nd_off_nl[..n_nl].to_vec();
            for h in 0..nh_nl {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_nl { nd_hedges_nl[cur[nd]] = h; cur[nd] += 1; }
                }
            }
        }

        // Floor connectivity from current best partition.
        let floor_nl: i32 = {
            let mut pseen = [false; 64];
            let mut c = 0i32;
            for h in 0..nh_nl {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut d = 0usize;
                for k in hs..he {
                    let p = partition_host_refine[hyperedge_nodes_host[k] as usize];
                    if p >= 0 && (p as usize) < np_nl && !pseen[p as usize] {
                        pseen[p as usize] = true; d += 1;
                    }
                }
                if d > 1 { c += (d - 1) as i32; }
                for k in hs..he {
                    let p = partition_host_refine[hyperedge_nodes_host[k] as usize];
                    if p >= 0 && (p as usize) < np_nl { pseen[p as usize] = false; }
                }
            }
            c
        };

        // ---- LEVEL-1 COARSENING: intra-block pair matching (cap=2) ----
        let mut rep1 = (0..n_nl).collect::<Vec<usize>>();
        let mut sn_size1 = vec![1i32; n_nl];
        {
            let mut matched1 = vec![false; n_nl];
            let mut edge_cnt = vec![0i32; n_nl];
            let mut seen_flag = vec![false; n_nl];
            let mut cands = Vec::<usize>::new();
            for u in 0..n_nl {
                if matched1[u] { continue; }
                let pu = partition_host_refine[u];
                if pu < 0 || (pu as usize) >= np_nl { continue; }
                cands.clear();
                for hi in nd_off_nl[u]..nd_off_nl[u + 1] {
                    let h = nd_hedges_nl[hi];
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let v = hyperedge_nodes_host[k] as usize;
                        if v == u || matched1[v] || partition_host_refine[v] != pu { continue; }
                        if !seen_flag[v] { seen_flag[v] = true; cands.push(v); }
                        edge_cnt[v] += 1;
                    }
                }
                let mut best_v = usize::MAX;
                let mut best_w = 0i32;
                for &v in &cands {
                    let w = edge_cnt[v];
                    if w > best_w || (w == best_w && (best_v == usize::MAX || v < best_v)) {
                        best_w = w; best_v = v;
                    }
                    edge_cnt[v] = 0; seen_flag[v] = false;
                }
                if best_v != usize::MAX {
                    rep1[best_v] = u;
                    sn_size1[u] += sn_size1[best_v];
                    sn_size1[best_v] = 0;
                    matched1[u] = true; matched1[best_v] = true;
                }
            }
        }

        // Build level-1 supernode→hedges.
        let mut sn1_hdg: Vec<Vec<usize>> = vec![Vec::new(); n_nl];
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r1 = rep1[v];
                    if sn_size1[r1] == 0 { continue; }
                    if rep_seen[r1] != stamp { rep_seen[r1] = stamp; sn1_hdg[r1].push(h); }
                }
            }
        }

        // ---- LEVEL-2 COARSENING: pair matching on active L1 supernodes (cap=2) ----
        // Max L2 supernode = 2×L1 = up to 4 fine nodes (2×2).
        let mut rep2 = (0..n_nl).collect::<Vec<usize>>();
        let mut sn_size2 = sn_size1.clone();
        {
            let mut matched2 = vec![false; n_nl];
            let mut edge_cnt = vec![0i32; n_nl];
            let mut seen_flag = vec![false; n_nl];
            let mut cands = Vec::<usize>::new();
            for r1 in 0..n_nl {
                if sn_size1[r1] == 0 || matched2[r1] { continue; }
                let pr = partition_host_refine[r1];
                if pr < 0 || (pr as usize) >= np_nl { continue; }
                cands.clear();
                for &h in &sn1_hdg[r1] {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let v = hyperedge_nodes_host[k] as usize;
                        let s = rep1[v];
                        if s == r1 || sn_size1[s] == 0 || matched2[s] { continue; }
                        if partition_host_refine[s] != pr { continue; }
                        if sn_size2[r1] + sn_size2[s] > 4 { continue; }
                        if !seen_flag[s] { seen_flag[s] = true; cands.push(s); }
                        edge_cnt[s] += 1;
                    }
                }
                let mut best_s = usize::MAX;
                let mut best_w = 0i32;
                for &s in &cands {
                    let w = edge_cnt[s];
                    if w > best_w || (w == best_w && (best_s == usize::MAX || s < best_s)) {
                        best_w = w; best_s = s;
                    }
                    edge_cnt[s] = 0; seen_flag[s] = false;
                }
                if best_s != usize::MAX {
                    rep2[best_s] = r1;
                    sn_size2[r1] += sn_size2[best_s];
                    sn_size2[best_s] = 0;
                    matched2[r1] = true; matched2[best_s] = true;
                }
            }
        }

        // Build composite fine→L2 rep mapping and L2 supernode→hedges (structural, built once).
        let mut rep_c = vec![0usize; n_nl];
        for v in 0..n_nl { rep_c[v] = rep2[rep1[v]]; }

        let mut hpc2: Vec<[i32; 64]> = vec![[0i32; 64]; nh_nl]; // filled after L3 projection
        let mut sn2_hdg: Vec<Vec<usize>> = vec![Vec::new(); n_nl];
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r2 = rep_c[v];
                    if sn_size2[r2] == 0 { continue; }
                    if rep_seen[r2] != stamp {
                        rep_seen[r2] = stamp;
                        sn2_hdg[r2].push(h); // structural only; hpc2 filled after L3 projection
                    }
                }
            }
        }

        // ---- LEVEL-3 COARSENING: pair matching on active L2 supernodes (cap=2) ----
        // Max L3 supernode = 2×L2 = up to 8 fine nodes (2×2×2).
        let mut rep3 = (0..n_nl).collect::<Vec<usize>>();
        let mut sn_size3 = sn_size2.clone();
        {
            let mut matched3 = vec![false; n_nl];
            let mut edge_cnt = vec![0i32; n_nl];
            let mut seen_flag = vec![false; n_nl];
            let mut cands = Vec::<usize>::new();
            for r2 in 0..n_nl {
                if sn_size2[r2] == 0 || matched3[r2] { continue; }
                let pr = partition_host_refine[r2];
                if pr < 0 || (pr as usize) >= np_nl { continue; }
                cands.clear();
                for &h in &sn2_hdg[r2] {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let v = hyperedge_nodes_host[k] as usize;
                        let s = rep_c[v]; // L2 rep of fine node v
                        if s == r2 || sn_size2[s] == 0 || matched3[s] { continue; }
                        if partition_host_refine[s] != pr { continue; }
                        if sn_size3[r2] + sn_size3[s] > 8 { continue; } // cap=8 fine nodes
                        if !seen_flag[s] { seen_flag[s] = true; cands.push(s); }
                        edge_cnt[s] += 1;
                    }
                }
                let mut best_s = usize::MAX;
                let mut best_w = 0i32;
                for &s in &cands {
                    let w = edge_cnt[s];
                    if w > best_w || (w == best_w && (best_s == usize::MAX || s < best_s)) {
                        best_w = w; best_s = s;
                    }
                    edge_cnt[s] = 0; seen_flag[s] = false;
                }
                if best_s != usize::MAX {
                    rep3[best_s] = r2; // r2 is the L3 representative
                    sn_size3[r2] += sn_size3[best_s];
                    sn_size3[best_s] = 0;
                    matched3[r2] = true; matched3[best_s] = true;
                }
            }
        }

        // Build composite fine→L3 rep mapping: rep_c2[v] = rep3[rep2[rep1[v]]].
        let mut rep_c2 = vec![0usize; n_nl];
        for v in 0..n_nl { rep_c2[v] = rep3[rep2[rep1[v]]]; }

        // Build L3 supernode→hedges (structural only; hpc3 built after L4→L3 projection).
        let mut sn3_hdg: Vec<Vec<usize>> = vec![Vec::new(); n_nl];
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r3 = rep_c2[v];
                    if sn_size3[r3] == 0 { continue; }
                    if rep_seen[r3] != stamp { rep_seen[r3] = stamp; sn3_hdg[r3].push(h); }
                }
            }
        }

        // ---- LEVEL-4 COARSENING: pair matching on active L3 supernodes (cap=2) ----
        // Max L4 supernode = 2×L3 = up to 16 fine nodes (2×2×2×2).
        let mut rep4 = (0..n_nl).collect::<Vec<usize>>();
        let mut sn_size4 = sn_size3.clone();
        {
            let mut matched4 = vec![false; n_nl];
            let mut edge_cnt = vec![0i32; n_nl];
            let mut seen_flag = vec![false; n_nl];
            let mut cands = Vec::<usize>::new();
            for r3 in 0..n_nl {
                if sn_size3[r3] == 0 || matched4[r3] { continue; }
                let pr = partition_host_refine[r3];
                if pr < 0 || (pr as usize) >= np_nl { continue; }
                cands.clear();
                for &h in &sn3_hdg[r3] {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let v = hyperedge_nodes_host[k] as usize;
                        let s = rep_c2[v]; // L3 rep of fine node v
                        if s == r3 || sn_size3[s] == 0 || matched4[s] { continue; }
                        if partition_host_refine[s] != pr { continue; }
                        if sn_size4[r3] + sn_size4[s] > 16 { continue; } // cap=16 fine nodes
                        if !seen_flag[s] { seen_flag[s] = true; cands.push(s); }
                        edge_cnt[s] += 1;
                    }
                }
                let mut best_s = usize::MAX;
                let mut best_w = 0i32;
                for &s in &cands {
                    let w = edge_cnt[s];
                    if w > best_w || (w == best_w && (best_s == usize::MAX || s < best_s)) {
                        best_w = w; best_s = s;
                    }
                    edge_cnt[s] = 0; seen_flag[s] = false;
                }
                if best_s != usize::MAX {
                    rep4[best_s] = r3; // r3 is the L4 representative
                    sn_size4[r3] += sn_size4[best_s];
                    sn_size4[best_s] = 0;
                    matched4[r3] = true; matched4[best_s] = true;
                }
            }
        }

        // Build composite fine→L4 rep mapping: rep_c3[v] = rep4[rep_c2[v]].
        let mut rep_c3 = vec![0usize; n_nl];
        for v in 0..n_nl { rep_c3[v] = rep4[rep_c2[v]]; }

        // Build L4 hedge counts (hpc4) and supernode→hedges (sn4_hdg).
        let mut hpc4: Vec<[i32; 64]> = vec![[0i32; 64]; nh_nl];
        let mut sn4_hdg: Vec<Vec<usize>> = vec![Vec::new(); n_nl];
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r4 = rep_c3[v];
                    if sn_size4[r4] == 0 { continue; }
                    if rep_seen[r4] != stamp {
                        rep_seen[r4] = stamp;
                        let p = partition_host_refine[r4];
                        if p >= 0 && (p as usize) < np_nl { hpc4[h][p as usize] += 1; }
                        sn4_hdg[r4].push(h);
                    }
                }
            }
        }

        // ---- FM at Level 4 (coarsest, ~n/16=3125 supernodes on 50K nodes) ----
        let l4_init_conn: i32 = {
            let mut c = 0i32;
            for h in 0..nh_nl {
                let d = hpc4[h][..np_nl].iter().filter(|&&v| v > 0).count();
                if d > 1 { c += (d - 1) as i32; }
            }
            c
        };
        let mut cur_l4_conn  = l4_init_conn;
        let mut best_l4_conn = l4_init_conn;
        let mut l4_part = partition_host_refine.clone();
        let mut l4_nip  = nodes_in_part_host.clone();
        let mut best_l4_part = l4_part.clone();
        let mut best_l4_nip  = l4_nip.clone();

        {
            let mut order4: Vec<(i32, usize)> = Vec::new();
            for r4 in 0..n_nl {
                if sn_size4[r4] == 0 { continue; }
                let pc = l4_part[r4];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size4[r4];
                if l4_nip[pc_u] - sz < 1 { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l4_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn4_hdg[r4] {
                        if hpc4[h][pc_u] == 1 { g += 1; }
                        if hpc4[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt < np_nl { order4.push((bg, r4)); }
            }
            order4.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut locked4 = vec![false; n_nl];
            for (_, r4) in order4 {
                if locked4[r4] { continue; }
                let pc = l4_part[r4];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size4[r4];
                if l4_nip[pc_u] - sz < 1 { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l4_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn4_hdg[r4] {
                        if hpc4[h][pc_u] == 1 { g += 1; }
                        if hpc4[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt >= np_nl { continue; }
                l4_part[r4] = bt as i32;
                l4_nip[pc_u] -= sz;
                l4_nip[bt] += sz;
                locked4[r4] = true;
                cur_l4_conn -= bg;
                for &h in &sn4_hdg[r4] { hpc4[h][pc_u] -= 1; hpc4[h][bt] += 1; }
                if cur_l4_conn < best_l4_conn {
                    best_l4_conn = cur_l4_conn;
                    best_l4_part.copy_from_slice(&l4_part);
                    best_l4_nip.copy_from_slice(&l4_nip);
                }
            }
        }
        l4_part.copy_from_slice(&best_l4_part);
        l4_nip.copy_from_slice(&best_l4_nip);

        // ---- UNCOARSEN L4 → project best_l4_part to L3, rebuild hpc3, localized FM ----
        let mut l3_part = vec![0i32; n_nl];
        let mut l3_nip  = vec![0i32; num_parts_usize];
        for r3 in 0..n_nl {
            if sn_size3[r3] == 0 { continue; }
            let r4 = rep4[r3];
            let new_p = l4_part[r4];
            l3_part[r3] = new_p;
            if new_p >= 0 && (new_p as usize) < num_parts_usize { l3_nip[new_p as usize] += sn_size3[r3]; }
        }

        // Rebuild hpc3 from projected l3_part.
        let mut hpc3: Vec<[i32; 64]> = vec![[0i32; 64]; nh_nl];
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r3 = rep_c2[v];
                    if sn_size3[r3] == 0 { continue; }
                    if rep_seen[r3] != stamp {
                        rep_seen[r3] = stamp;
                        let p = l3_part[r3];
                        if p >= 0 && (p as usize) < np_nl { hpc3[h][p as usize] += 1; }
                    }
                }
            }
        }

        // Localized FM at boundary L3 supernodes only (uncoarsen L4→L3).
        let l3_init_conn: i32 = {
            let mut c = 0i32;
            for h in 0..nh_nl {
                let d = hpc3[h][..np_nl].iter().filter(|&&v| v > 0).count();
                if d > 1 { c += (d - 1) as i32; }
            }
            c
        };
        let mut cur_l3_conn  = l3_init_conn;
        let mut best_l3_conn = l3_init_conn;
        let mut best_l3_part = l3_part.clone();
        let mut best_l3_nip  = l3_nip.clone();

        {
            let mut order3: Vec<(i32, usize)> = Vec::new();
            for r3 in 0..n_nl {
                if sn_size3[r3] == 0 { continue; }
                let pc = l3_part[r3];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size3[r3];
                if l3_nip[pc_u] - sz < 1 { continue; }
                // Only boundary L3 supernodes (incident to a cut hedge).
                let on_boundary = sn3_hdg[r3].iter().any(|&h| {
                    hpc3[h][..np_nl].iter().filter(|&&c| c > 0).count() > 1
                });
                if !on_boundary { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l3_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn3_hdg[r3] {
                        if hpc3[h][pc_u] == 1 { g += 1; }
                        if hpc3[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt < np_nl { order3.push((bg, r3)); }
            }
            order3.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut locked3 = vec![false; n_nl];
            for (_, r3) in order3 {
                if locked3[r3] { continue; }
                let pc = l3_part[r3];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size3[r3];
                if l3_nip[pc_u] - sz < 1 { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l3_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn3_hdg[r3] {
                        if hpc3[h][pc_u] == 1 { g += 1; }
                        if hpc3[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt >= np_nl { continue; }
                l3_part[r3] = bt as i32;
                l3_nip[pc_u] -= sz;
                l3_nip[bt] += sz;
                locked3[r3] = true;
                cur_l3_conn -= bg;
                for &h in &sn3_hdg[r3] { hpc3[h][pc_u] -= 1; hpc3[h][bt] += 1; }
                if cur_l3_conn < best_l3_conn {
                    best_l3_conn = cur_l3_conn;
                    best_l3_part.copy_from_slice(&l3_part);
                    best_l3_nip.copy_from_slice(&l3_nip);
                }
            }
        }
        l3_part.copy_from_slice(&best_l3_part);
        l3_nip.copy_from_slice(&best_l3_nip);

        // ---- UNCOARSEN L3 → project best_l3_part to L2, rebuild hpc2, localized FM ----
        // 
        let mut l2_part = vec![0i32; n_nl];
        let mut l2_nip  = vec![0i32; num_parts_usize];
        for r2 in 0..n_nl {
            if sn_size2[r2] == 0 { continue; }
            let r3 = rep3[r2];
            let new_p = l3_part[r3];
            l2_part[r2] = new_p;
            if new_p >= 0 && (new_p as usize) < num_parts_usize { l2_nip[new_p as usize] += sn_size2[r2]; }
        }

        // Rebuild hpc2 from projected l2_part.
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r2 = rep_c[v];
                    if sn_size2[r2] == 0 { continue; }
                    if rep_seen[r2] != stamp {
                        rep_seen[r2] = stamp;
                        let p = l2_part[r2];
                        if p >= 0 && (p as usize) < np_nl { hpc2[h][p as usize] += 1; }
                    }
                }
            }
        }

        // Localized FM at boundary L2 supernodes only (uncoarsen L3→L2).
        let l2_init_conn: i32 = {
            let mut c = 0i32;
            for h in 0..nh_nl {
                let d = hpc2[h][..np_nl].iter().filter(|&&v| v > 0).count();
                if d > 1 { c += (d - 1) as i32; }
            }
            c
        };
        let mut cur_l2_conn  = l2_init_conn;
        let mut best_l2_conn = l2_init_conn;
        let mut best_l2_part = l2_part.clone();
        let mut best_l2_nip  = l2_nip.clone();

        {
            let mut order2: Vec<(i32, usize)> = Vec::new();
            for r2 in 0..n_nl {
                if sn_size2[r2] == 0 { continue; }
                let pc = l2_part[r2];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size2[r2];
                if l2_nip[pc_u] - sz < 1 { continue; }
                // Only boundary L2 supernodes (incident to a cut hedge).
                let on_boundary = sn2_hdg[r2].iter().any(|&h| {
                    hpc2[h][..np_nl].iter().filter(|&&c| c > 0).count() > 1
                });
                if !on_boundary { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l2_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn2_hdg[r2] {
                        if hpc2[h][pc_u] == 1 { g += 1; }
                        if hpc2[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt < np_nl { order2.push((bg, r2)); }
            }
            order2.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut locked2 = vec![false; n_nl];
            for (_, r2) in order2 {
                if locked2[r2] { continue; }
                let pc = l2_part[r2];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size2[r2];
                if l2_nip[pc_u] - sz < 1 { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l2_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn2_hdg[r2] {
                        if hpc2[h][pc_u] == 1 { g += 1; }
                        if hpc2[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt >= np_nl { continue; }
                l2_part[r2] = bt as i32;
                l2_nip[pc_u] -= sz;
                l2_nip[bt] += sz;
                locked2[r2] = true;
                cur_l2_conn -= bg;
                for &h in &sn2_hdg[r2] { hpc2[h][pc_u] -= 1; hpc2[h][bt] += 1; }
                if cur_l2_conn < best_l2_conn {
                    best_l2_conn = cur_l2_conn;
                    best_l2_part.copy_from_slice(&l2_part);
                    best_l2_nip.copy_from_slice(&l2_nip);
                }
            }
        }
        l2_part.copy_from_slice(&best_l2_part);
        l2_nip.copy_from_slice(&best_l2_nip);

        // ---- UNCOARSEN L2 → project + localized FM ----
        // 
        let mut l1_part = vec![0i32; n_nl];
        let mut l1_nip  = vec![0i32; num_parts_usize];
        for r1 in 0..n_nl {
            if sn_size1[r1] == 0 { continue; }
            let r2 = rep2[r1];
            let new_p = l2_part[r2];
            l1_part[r1] = new_p;
            if new_p >= 0 && (new_p as usize) < num_parts_usize { l1_nip[new_p as usize] += sn_size1[r1]; }
        }

        // Rebuild hpc1 from projected l1_part.
        let mut hpc1: Vec<[i32; 64]> = vec![[0i32; 64]; nh_nl];
        {
            let mut rep_seen = vec![0u32; n_nl];
            let mut stamp = 0u32;
            for h in 0..nh_nl {
                stamp = stamp.wrapping_add(1);
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let v = hyperedge_nodes_host[k] as usize;
                    let r1 = rep1[v];
                    if sn_size1[r1] == 0 { continue; }
                    if rep_seen[r1] != stamp {
                        rep_seen[r1] = stamp;
                        let p = l1_part[r1];
                        if p >= 0 && (p as usize) < np_nl { hpc1[h][p as usize] += 1; }
                    }
                }
            }
        }

        // Localized FM at boundary L1 supernodes only (uncoarsen L2→L1).
        let l1_init_conn: i32 = {
            let mut c = 0i32;
            for h in 0..nh_nl {
                let d = hpc1[h][..np_nl].iter().filter(|&&v| v > 0).count();
                if d > 1 { c += (d - 1) as i32; }
            }
            c
        };
        let mut cur_l1_conn  = l1_init_conn;
        let mut best_l1_conn = l1_init_conn;
        let mut best_l1_part = l1_part.clone();
        let mut best_l1_nip  = l1_nip.clone();
        {
            let mut order1: Vec<(i32, usize)> = Vec::new();
            for r1 in 0..n_nl {
                if sn_size1[r1] == 0 { continue; }
                let pc = l1_part[r1];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size1[r1];
                if l1_nip[pc_u] - sz < 1 { continue; }
                let on_boundary = sn1_hdg[r1].iter().any(|&h| {
                    hpc1[h][..np_nl].iter().filter(|&&c| c > 0).count() > 1
                });
                if !on_boundary { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l1_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn1_hdg[r1] {
                        if hpc1[h][pc_u] == 1 { g += 1; }
                        if hpc1[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt < np_nl { order1.push((bg, r1)); }
            }
            order1.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut locked1 = vec![false; n_nl];
            for (_, r1) in order1 {
                if locked1[r1] { continue; }
                let pc = l1_part[r1];
                if pc < 0 || (pc as usize) >= np_nl { continue; }
                let pc_u = pc as usize;
                let sz  = sn_size1[r1];
                if l1_nip[pc_u] - sz < 1 { continue; }
                let mut bg = i32::MIN; let mut bt = np_nl;
                for tgt in 0..np_nl {
                    if tgt == pc_u || l1_nip[tgt] + sz > max_ps_nl { continue; }
                    let mut g = 0i32;
                    for &h in &sn1_hdg[r1] {
                        if hpc1[h][pc_u] == 1 { g += 1; }
                        if hpc1[h][tgt]  == 0 { g -= 1; }
                    }
                    if g > bg || (g == bg && tgt < bt) { bg = g; bt = tgt; }
                }
                if bt >= np_nl { continue; }
                l1_part[r1] = bt as i32;
                l1_nip[pc_u] -= sz;
                l1_nip[bt] += sz;
                locked1[r1] = true;
                cur_l1_conn -= bg;
                for &h in &sn1_hdg[r1] { hpc1[h][pc_u] -= 1; hpc1[h][bt] += 1; }
                if cur_l1_conn < best_l1_conn {
                    best_l1_conn = cur_l1_conn;
                    best_l1_part.copy_from_slice(&l1_part);
                    best_l1_nip.copy_from_slice(&l1_nip);
                }
            }
        }

        // ---- UNCOARSEN L1 → Fine: project, compute fine connectivity ----
        // 
        let mut fine_part_nl = vec![0i32; n_nl];
        let mut fine_nip_nl  = vec![0i32; num_parts_usize];
        for v in 0..n_nl {
            let r1 = rep1[v];
            let new_p = best_l1_part[r1];
            fine_part_nl[v] = new_p;
            if new_p >= 0 && (new_p as usize) < num_parts_usize { fine_nip_nl[new_p as usize] += 1; }
        }

        // Balance HARD check (strict ≤ max_ps_nl, all parts non-empty).
        let balance_ok = fine_nip_nl.iter().all(|&c| c <= max_ps_nl)
            && fine_nip_nl.iter().all(|&c| c >= 1);

        // Compute fine connectivity (only parts 0..np_nl=64, consistent with floor_nl).
        let fine_conn_nl: i32 = if balance_ok {
            let mut pseen = [false; 64];
            let mut c = 0i32;
            for h in 0..nh_nl {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                let mut d = 0usize;
                for k in hs..he {
                    let p = fine_part_nl[hyperedge_nodes_host[k] as usize];
                    if p >= 0 && (p as usize) < np_nl && !pseen[p as usize] {
                        pseen[p as usize] = true; d += 1;
                    }
                }
                if d > 1 { c += (d - 1) as i32; }
                for k in hs..he {
                    let p = fine_part_nl[hyperedge_nodes_host[k] as usize];
                    if p >= 0 && (p as usize) < np_nl { pseen[p as usize] = false; }
                }
            }
            c
        } else { i32::MAX };

        // Non-decreasing guard: commit only if strictly improved vs floor AND balanced.
        if fine_conn_nl < floor_nl {
            partition_host_refine.copy_from_slice(&fine_part_nl);
            nodes_in_part_host.copy_from_slice(&fine_nip_nl);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }

        let _ = (nd_deg_nl, nd_off_nl, nd_hedges_nl, floor_nl, fine_conn_nl,
                 l4_init_conn, l3_init_conn, l2_init_conn, l1_init_conn,
                 best_l4_conn, best_l3_conn, best_l2_conn, best_l1_conn,
                 cur_l4_conn, cur_l3_conn, cur_l2_conn, cur_l1_conn,
                 hpc4, hpc3, hpc2, hpc1, rep_c, rep_c2, rep_c3,
                 l4_nip, l3_nip, l2_nip, l1_nip,
                 best_l4_nip, best_l3_nip, best_l2_nip, best_l1_nip,
                 best_l4_part, best_l3_part, best_l2_part, best_l1_part, fine_nip_nl,
                 rep4, sn_size4, sn4_hdg, rep3, sn_size3, sn3_hdg);
    }
    // ==== END B_NLEVEL_FM i103 ====

    // ==== B_BATCH_FM_GPU i104 — post-NLEVEL fine-level GPU swap ====
    // After NLEVEL coarse-to-fine projection, run one GPU swap phase to catch
    // pair-swap improvements at fine level that coarse FM misses.
    // Uses existing do_swap_phase! macro (compute_swap_gains_extended_50k GPU kernel).
    // d_partition/d_nodes_in_part already updated by NLEVEL if it improved.
    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        5, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;
    partition_host_refine.copy_from_slice(&partition_host_swap);
    // ==== END B_BATCH_FM_GPU i104 ====

    
    // Balance-NEUTRAL cross-block pair swaps post-B_BATCH_FM_GPU. A pair (v in A -> B, u in B -> A)
    // leaves EVERY block size unchanged (Δbalance=0 exact), so it can cross the balance barrier
    // that mono-vertex FM (NLEVEL_FM) and the single-target GPU swap (do_swap_phase!) cannot: a move
    // v->B that single-vertex/single-target moves reject because B is at capacity becomes feasible
    // because u simultaneously leaves B. Distinct operator class — non-subsumable by P18 batch-GPU
    
    // penalty is irrelevant+harmful for size-preserving swaps). Sequential EXACT gain recompute per
    // applied move (CPU sequential mandatory — no thread/rayon/spawn, TIG hard rule). Reads the
    // post-BATCH_FM_GPU state via partition_host_refine; commits to d_partition only if connectivity
    // strictly improves and balance holds (monotone guard mirrors RECOMBINE/NLEVEL).
    {
        let n_sp = challenge.num_nodes as usize;
        let m_sp = challenge.num_hyperedges as usize;
        let np_sp = num_parts_usize;
        let max_ps_sp = challenge.max_part_size as i32;

        if np_sp >= 2 {
            // node->hedge CSR (self-contained; rebuild is O(pins), << 1s)
            let mut nd_deg = vec![0usize; n_sp];
            for h in 0..m_sp {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_sp { nd_deg[nd] += 1; }
                }
            }
            let mut nd_off = vec![0usize; n_sp + 1];
            for i in 0..n_sp { nd_off[i + 1] = nd_off[i] + nd_deg[i]; }
            let total_pins = nd_off[n_sp];
            let mut nd_hedges = vec![0usize; total_pins];
            {
                let mut cur = nd_off[..n_sp].to_vec();
                for h in 0..m_sp {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        if nd < n_sp { nd_hedges[cur[nd]] = h; cur[nd] += 1; }
                    }
                }
            }

            // hpc[h*np + p] = #pins of hedge h in block p, from canonical partition_host_refine
            let mut hpc = vec![0u16; m_sp * np_sp];
            for h in 0..m_sp {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    let p = partition_host_refine[nd];
                    if p >= 0 && (p as usize) < np_sp {
                        hpc[h * np_sp + p as usize] = hpc[h * np_sp + p as usize].saturating_add(1);
                    }
                }
            }

            let initial_conn: i32 = {
                let mut c = 0i32;
                for h in 0..m_sp {
                    let mut d = 0i32;
                    for p in 0..np_sp { if hpc[h * np_sp + p] > 0 { d += 1; } }
                    if d > 1 { c += d - 1; }
                }
                c
            };

            // apply move v -> t on (sp_part, hpc); returns connectivity REDUCTION (gain).
            // Reversible: calling again with (v, old_block) restores hpc exactly.
            fn sp_apply(
                v: usize, t: usize,
                sp_part: &mut [i32], hpc: &mut [u16],
                nd_off: &[usize], nd_hedges: &[usize], np: usize,
            ) -> i32 {
                let p = sp_part[v] as usize;
                let hs = nd_off[v];
                let he = nd_off[v + 1];
                let mut g = 0i32;
                for hi in hs..he {
                    let h = nd_hedges[hi];
                    if hpc[h * np + p] == 1 { g += 1; }   // block p emptied -> λ-1
                    if hpc[h * np + t] == 0 { g -= 1; }   // block t newly occupied -> λ+1
                    if hpc[h * np + p] > 0 { hpc[h * np + p] -= 1; }
                    hpc[h * np + t] = hpc[h * np + t].saturating_add(1);
                }
                sp_part[v] = t as i32;
                g
            }

            let mut sp_part = partition_host_refine.clone();
            let mut current_conn = initial_conn;
            let mut tgt = vec![0i32; np_sp];
            // best_to[a*np + b] = (best gain, node) for moving a boundary node of block a to block b
            let mut best_to: Vec<(i32, usize)> = vec![(i32::MIN, usize::MAX); np_sp * np_sp];

            const MAX_SP_PASSES: usize = 2;
            for _sp_pass in 0..MAX_SP_PASSES {
                // ---- single scan: best single-move gain per (src_block, dst_block) ----
                for s in best_to.iter_mut() { *s = (i32::MIN, usize::MAX); }
                for v in 0..n_sp {
                    let p_v = sp_part[v];
                    if p_v < 0 || (p_v as usize) >= np_sp { continue; }
                    let p_v_u = p_v as usize;
                    let hs_v = nd_off[v];
                    let he_v = nd_off[v + 1];
                    let is_bnd = (hs_v..he_v).any(|hi| {
                        let h = nd_hedges[hi];
                        (0..np_sp).any(|q| q != p_v_u && hpc[h * np_sp + q] > 0)
                    });
                    if !is_bnd { continue; }
                    tgt.fill(0);
                    let mut base_gain = 0i32;
                    for hi in hs_v..he_v {
                        let h = nd_hedges[hi];
                        if hpc[h * np_sp + p_v_u] == 1 { base_gain += 1; }
                        for t in 0..np_sp {
                            if t != p_v_u && hpc[h * np_sp + t] == 0 { tgt[t] -= 1; }
                        }
                    }
                    for t in 0..np_sp {
                        if t == p_v_u { continue; }
                        let g = tgt[t] + base_gain;
                        let slot = p_v_u * np_sp + t;
                        if g > best_to[slot].0 || (g == best_to[slot].0 && v < best_to[slot].1) {
                            best_to[slot] = (g, v);
                        }
                    }
                }

                // ---- drain: one balance-neutral swap per block-pair (a<b) ----
                let mut improved = false;
                for a in 0..np_sp {
                    for b in (a + 1)..np_sp {
                        let (gv, v) = best_to[a * np_sp + b];
                        let (gu, u) = best_to[b * np_sp + a];
                        if v == usize::MAX || u == usize::MAX || v == u { continue; }
                        if gv == i32::MIN || gu == i32::MIN { continue; }
                        if gv + gu <= 0 { continue; }                 // approx prune
                        // node may have already moved earlier this pass -> stale candidate
                        if sp_part[v] != a as i32 || sp_part[u] != b as i32 { continue; }
                        // EXACT joint gain via sequential apply
                        let rgv = sp_apply(v, b, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                        let rgu = sp_apply(u, a, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                        let real = rgv + rgu;
                        if real > 0 {
                            current_conn -= real;
                            improved = true;
                        } else {
                            // revert exactly (counts restored; order-independent)
                            sp_apply(u, b, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                            sp_apply(v, a, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                        }
                    }
                }
                if !improved { break; }
            }

            // ---- commit: only if strictly improved; balance is neutral by construction ----
            if current_conn < initial_conn {
                let mut sp_nip = vec![0i32; np_sp];
                for &p in &sp_part {
                    if p >= 0 && (p as usize) < np_sp { sp_nip[p as usize] += 1; }
                }
                let balance_ok = sp_nip.iter().all(|&c| c <= max_ps_sp);
                // fresh exact connectivity from final partition (guards the incremental bookkeeping)
                let mut final_conn = 0i32;
                let mut pseen = vec![false; np_sp];
                for h in 0..m_sp {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    let mut d = 0i32;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        let p = sp_part[nd];
                        if p >= 0 && (p as usize) < np_sp && !pseen[p as usize] {
                            pseen[p as usize] = true; d += 1;
                        }
                    }
                    if d > 1 { final_conn += d - 1; }
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        let p = sp_part[nd];
                        if p >= 0 && (p as usize) < np_sp { pseen[p as usize] = false; }
                    }
                }
                if balance_ok && final_conn < initial_conn {
                    partition_host_refine.copy_from_slice(&sp_part);
                    nodes_in_part_host.copy_from_slice(&sp_nip);
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
            }

            let _ = (nd_deg, total_pins);
        }
    }
    // ==== END B_BALANCED_SWAP_PAIR i120 ====

    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    save_solution(&Solution {
        partition: partition_u32,
    })?;
    Ok(())
}
