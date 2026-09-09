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

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_200k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_200k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_200k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_200k")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves_lanes_200k")?;
    let compute_moves_orig_kernel = module.load_function("compute_refinement_moves_optimized_200k")?;
    let filter_fused_kernel = module.load_function("filter_fused_200k")?;
    let round_fused_kernel = module.load_function("round_fused_200k")?;
    let rounds_fused_kernel = module.load_function("rounds_fused_200k")?;
    let balance_kernel = module.load_function("balance_final_200k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_200k")?;
    let reduce_connectivity_sum_kernel = module.load_function("reduce_connectivity_sum_200k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_200k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_200k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_200k")?;

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


    let mlanes_raw: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("mlanes").and_then(|v| v.as_i64()))
        .unwrap_or(4);
    let use_lanes_kernel = mlanes_raw != 0;
    let moves_lanes: i32 = {
        let v = mlanes_raw.clamp(2, 32) as i32;
        1 << (31 - (v as u32).leading_zeros())
    };
    let cfg_moves_orig = cfg.clone();


    let tiebreak: i32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("tiebreak").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 2) as i32)
        .unwrap_or(1);


    let tabu_exec = hyperparameters
        .as_ref()
        .and_then(|p| p.get("tabu_exec").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        != 0;

    let swap_scale: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("swap_scale").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 400) as u64)
        .unwrap_or(100);
    let moves_groups = 256u32 / moves_lanes as u32;
    let moves_grid_max = (challenge.num_nodes as u32 + 7) / 8;


    let moves_grid_default = (challenge.num_nodes as u32 + 2 * moves_groups - 1) / (2 * moves_groups);
    let moves_grid_x = hyperparameters
        .as_ref()
        .and_then(|p| p.get("mblocks").and_then(|v| v.as_i64()))
        .map(|v| (v.max(1) as u32).min(moves_grid_max))
        .unwrap_or(moves_grid_default.max(1));
    let cfg_moves = LaunchConfig {
        grid_dim: (moves_grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };


    let use_host_flags = hyperparameters
        .as_ref()
        .and_then(|p| p.get("hostflags").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        != 0;

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


        5 => (20000, 1, 70, 20, 0),
        4 => (7000, 5, 60, 200, 0),
        3 => (5000, 5, 50, 150, 64),
        2 => (2000, 5, 50, 100, 64),
        1 => (1000, 5, 50, 50, 64),
        0 => (500, 5, 50, 25, 64),
        _ => (5000, 5, 50, 200, 64),
    };

    let refinement_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(50, 50_000) as usize)
        .unwrap_or(base_refine);

    let ils_iterations = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ils_iterations").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 500) as usize)
        .unwrap_or(base_ils);

    let ils_quick_refine = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ils_quick_refine").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(10, 500) as usize)
        .unwrap_or(base_ils_quick);

    let post_ils_polish = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_ils_polish").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 500) as usize)
        .unwrap_or(base_polish);

    let tabu_tenure: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("tabu_tenure").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 30) as usize)
        .unwrap_or(14);

    let tabu_fail_tenure = 4usize;
    let tabu_mark_base = 4096usize;
    let tabu_mark_mult = 16usize;
    let tabu_fail_mark_len = 4096usize;

    let extra_window = 65536usize;
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


    let n_he_host = challenge.num_hyperedges as usize;
    let node_offsets_host: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let node_hyperedges_host: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
    let mut edge_part_cnt: Vec<u16> = vec![0u16; if use_host_flags { n_he_host * 64 } else { 0 }];
    let mut flags_all_host: Vec<u64> = vec![0u64; n_he_host];
    let mut flags_double_host: Vec<u64> = vec![0u64; n_he_host];
    let mut host_flags_valid = false;


    macro_rules! rebuild_host_flags {
        () => {{
            edge_part_cnt.fill(0);
            let nn = challenge.num_nodes as u32;
            for e in 0..n_he_host {
                let s = hedge_offsets_host[e] as usize;
                let t = hedge_offsets_host[e + 1] as usize;
                let base = e * 64;
                let mut fa = 0u64;
                let mut fd = 0u64;
                for k in s..t {
                    let v = hyperedge_nodes_host[k];
                    if (v as u32) < nn {
                        let p = partition_host_refine[v as usize];
                        if (p as u32) < 64 {
                            let bit = 1u64 << p;
                            fd |= fa & bit;
                            fa |= bit;
                            edge_part_cnt[base + p as usize] += 1;
                        }
                    }
                }
                flags_all_host[e] = fa;
                flags_double_host[e] = fd;
            }
            host_flags_valid = true;
        }};
    }


    macro_rules! update_host_flags_for_move {
        ($node:expr, $from:expr, $to:expr) => {{
            let s = node_offsets_host[$node] as usize;
            let t = node_offsets_host[$node + 1] as usize;
            let ba = 1u64 << $from;
            let bb = 1u64 << $to;
            for k in s..t {
                let e = node_hyperedges_host[k] as usize;
                let base = e * 64;
                edge_part_cnt[base + $from] -= 1;
                edge_part_cnt[base + $to] += 1;
                let ca = edge_part_cnt[base + $from];
                let cb = edge_part_cnt[base + $to];
                let mut fa = flags_all_host[e];
                let mut fd = flags_double_host[e];
                if ca >= 1 { fa |= ba; } else { fa &= !ba; }
                if ca >= 2 { fd |= ba; } else { fd &= !ba; }
                fa |= bb;
                if cb >= 2 { fd |= bb; } else { fd &= !bb; }
                flags_all_host[e] = fa;
                flags_double_host[e] = fd;
            }
        }};
    }


    let mut exec_nodes: Vec<i32> = Vec::with_capacity(65536);


    let mut part_upd_host: Vec<i32> = Vec::with_capacity(2 * challenge.num_nodes as usize);
    let mut defer_part_upload = false;
    macro_rules! exec_moves_mirror {
        ($move_nodes:expr, $move_parts:expr) => {{
            let mut executed = 0i32;
            exec_nodes.clear();
            for i in 0..$move_nodes.len() {
                let node = $move_nodes[i];
                let target_part = $move_parts[i];
                if node < 0 || target_part < 0 {
                    continue;
                }
                let node_usize = node as usize;
                if node_usize >= partition_host_refine.len() {
                    continue;
                }
                let current_part = partition_host_refine[node_usize];
                if current_part >= 0
                    && (current_part as usize) < num_parts_usize
                    && (target_part as usize) < num_parts_usize
                    && nodes_in_part_host[target_part as usize] < challenge.max_part_size as i32
                    && nodes_in_part_host[current_part as usize] > 1
                {
                    partition_host_refine[node_usize] = target_part;
                    nodes_in_part_host[current_part as usize] -= 1;
                    nodes_in_part_host[target_part as usize] += 1;
                    if host_flags_valid {
                        update_host_flags_for_move!(node_usize, current_part as usize, target_part as usize);
                    }
                    exec_nodes.push(node);
                    if defer_part_upload {
                        part_upd_host.push(node);
                        part_upd_host.push(target_part);
                    }
                    executed += 1;
                }
            }
            if executed > 0 {
                if !defer_part_upload {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                }
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            }
            executed
        }};
    }

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


    let mut main_valid_moves: Vec<u64> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut tgt_buckets: Vec<Vec<u64>> = (0..num_parts_usize)
        .map(|_| Vec::with_capacity(1024))
        .collect();

    let mut gen_accepted: Vec<u64> = Vec::with_capacity(65536);
    let mv_node = |v: u64| (v & 0xFFFF_FFFF) as usize;
    let mv_key = |v: u64| !((v >> 32) as u32) as i32;


    let gfloor: i32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gfloor").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(-32768, 0) as i32)
        .unwrap_or(-3);
    let gfloor_off: i32 = -32768;


    let use_gfilter = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gfilter").and_then(|v| v.as_i64()))
        .map(|v| v != 0)
        .unwrap_or(true);


    let reheat_cycles: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("reheat").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 16) as usize)
        .unwrap_or(1);
    let reheat_temp: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("reheat_temp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100) as u64)
        .unwrap_or(50);
    let cycle_len = std::cmp::max(1, refinement_rounds / reheat_cycles);


    let sched_exp: u32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("sched_exp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 4) as u32)
        .unwrap_or(3);
    let pen_mul: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("pen_mul").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 64) as u64)
        .unwrap_or(3);
    let mut best_loop_conn: i32 = i32::MAX;
    let mut best_loop_partition: Vec<i32> = Vec::new();
    let mut best_loop_nodes_in_part: Vec<i32> = Vec::new();

    let filt_blocks = cfg.grid_dim.0 as usize;


    let mut d_tabu_until = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;
    let mut tabu_mirror: Vec<u32> = vec![0u32; challenge.num_nodes as usize];
    let mut tabu_dirty = true;
    let mut d_filt_stats = stream.alloc_zeros::<i32>(filt_blocks * 8)?;
    let mut d_filt_off = stream.alloc_zeros::<u32>(filt_blocks)?;


    let mut d_filt_hdr = stream.alloc_zeros::<i32>(40)?;
    let mut filt_hdr_host: Vec<i32> = vec![0i32; 40];
    let filt_hdr_zero: Vec<i32> = vec![0i32; 40];
    let mut d_cand_out = stream.alloc_zeros::<u64>(challenge.num_nodes as usize)?;


    let fused_max_per_sm = filter_fused_kernel
        .occupancy_max_active_blocks_per_multiprocessor(block_size, 0, None)
        .unwrap_or(1)
        .max(1);
    let fused_grid = std::cmp::min(
        filt_blocks as u32,
        (prop.multiProcessorCount as u32).max(1) * fused_max_per_sm,
    )
    .max(1);
    let fused_cfg = LaunchConfig {
        grid_dim: (fused_grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };


    let use_round_fused = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfuse").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        != 0
        && use_gfilter
        && use_lanes_kernel
        && !use_host_flags;


    let use_gsort = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gsort").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        != 0
        && use_round_fused;
    let gsort_arg: i32 = if use_gsort { 1 } else { 0 };


    let rmulti: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rmulti").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 256) as usize)
        .unwrap_or(8);
    let use_multi = use_round_fused && use_gsort && rmulti > 1;
    let mut rf_epoch: u32 = 0;


    let round_smem_words = std::cmp::max(moves_groups as usize * 64, 8 * 256);
    let round_smem_bytes = (round_smem_words * std::mem::size_of::<u32>()) as u32;


    let round_max_per_sm = if use_multi {
        rounds_fused_kernel
            .occupancy_max_active_blocks_per_multiprocessor(256, round_smem_bytes as usize, None)
            .unwrap_or(1)
            .max(1)
    } else {
        round_fused_kernel
            .occupancy_max_active_blocks_per_multiprocessor(256, round_smem_bytes as usize, None)
            .unwrap_or(1)
            .max(1)
    };
    let round_grid = std::cmp::min(
        moves_grid_x,
        (prop.multiProcessorCount as u32).max(1) * round_max_per_sm,
    )
    .max(1);
    let round_cfg = LaunchConfig {
        grid_dim: (round_grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: round_smem_bytes,
    };


    let mut d_edge_flags_pair = stream.alloc_zeros::<u64>(2 * challenge.num_hyperedges as usize)?;


    let rs_nt_max = (challenge.num_nodes as usize + 255) / 256;
    let mut d_tcnt = stream.alloc_zeros::<u32>(64 * rs_nt_max)?;
    let mut d_koff = stream.alloc_zeros::<u32>(rs_nt_max)?;
    let mut kept_host: Vec<u64> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut d_acc = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;
    let mut d_acc_off = stream.alloc_zeros::<u32>(filt_blocks)?;
    let mut d_cand_tmp = stream.alloc_zeros::<u64>(challenge.num_nodes as usize)?;
    let mut d_rs_counts = stream.alloc_zeros::<u32>(4 * 256 * rs_nt_max)?;


    let mut rf_soft = true;
    let mut rf_launch_idx: u32 = 0;


    let mut d_tabu_upd = stream.alloc_zeros::<u32>(2 * challenge.num_nodes as usize)?;
    let mut tabu_upd_host: Vec<u32> = Vec::with_capacity(2 * challenge.num_nodes as usize);
    let mut d_part_upd = stream.alloc_zeros::<i32>(2 * challenge.num_nodes as usize)?;

    macro_rules! flush_part_upd {
        () => {{
            if !part_upd_host.is_empty() {
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                part_upd_host.clear();
            }
        }};
    }


    macro_rules! refresh_host_mirrors_from_device {
        () => {{
            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
            host_flags_valid = false;
        }};
    }


    macro_rules! launch_compute_moves {
        ($floor:expr) => {{
            let floor_arg: i32 = $floor;
            if use_lanes_kernel {
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
                        .arg(&moves_lanes)
                        .arg(&tiebreak)
                        .arg(&floor_arg)
                        .launch(cfg_moves.clone())?;
                }
            } else {
                unsafe {
                    stream
                        .launch_builder(&compute_moves_orig_kernel)
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
                        .arg(&floor_arg)
                        .launch(cfg_moves_orig.clone())?;
                }
            }
        }};
    }


    macro_rules! eval_connectivity {
        () => {{
            flush_part_upd!();
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
            let v = stream.memcpy_dtov(&d_connectivity)?;
            v.iter().sum::<i32>()
        }};
    }

    macro_rules! restore_loop_best {
        () => {{
            partition_host_refine.copy_from_slice(&best_loop_partition);
            nodes_in_part_host.copy_from_slice(&best_loop_nodes_in_part);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            part_upd_host.clear();
            host_flags_valid = false;
        }};
    }


    if use_round_fused {
        stream.memcpy_htod(&filt_hdr_zero, &mut d_filt_hdr)?;
        rf_epoch = 0;
        tabu_upd_host.clear();
        part_upd_host.clear();


        defer_part_upload = !use_multi;
    }

    let cycle_bounds = |r: usize| -> (usize, usize, usize) {
        if reheat_cycles > 1 {
            let c = std::cmp::min(r / cycle_len, reheat_cycles - 1);
            let s = c * cycle_len;
            let e = if c + 1 == reheat_cycles { refinement_rounds } else { s + cycle_len };
            (c, s, e)
        } else {
            (0usize, 0usize, refinement_rounds)
        }
    };
    let mut round_idx = 0usize;
    'rounds: while round_idx < refinement_rounds {

        let round: usize;
        let mut moves_executed: i32;


        let mut rf_device_round = false;
        'body: {
        if use_multi {


            let (cycle_idx, cycle_start, cycle_end) = cycle_bounds(round_idx);
            let batch = std::cmp::min(rmulti, cycle_end - round_idx).max(1);
                let n_tabu_upd = (tabu_upd_host.len() / 2) as i32;
            if n_tabu_upd > 0 {
                let n = tabu_upd_host.len();
                stream.memcpy_htod(&tabu_upd_host[..n], &mut d_tabu_upd.slice_mut(0..n))?;
            }
            let n_part_upd = (part_upd_host.len() / 2) as i32;
            if n_part_upd > 0 {
                let n = part_upd_host.len();
                stream.memcpy_htod(&part_upd_host[..n], &mut d_part_upd.slice_mut(0..n))?;
            }
            let (num_mul, den_mul): (u64, u64) = if reheat_cycles > 1 {
                (if cycle_idx > 0 { reheat_temp } else { 100 }, 100)
            } else {
                (1, 1)
            };
            let cyc_len = (cycle_end - cycle_start) as u64;
            let rr = cyc_len.pow(sched_exp).saturating_mul(den_mul);
            let mut prob_den = [0u64; 4];
            for p in 0..4 {
                prob_den[p] = rr.saturating_mul(1 + p as u64 * pen_mul);
            }
            let arg_nh: i32 = challenge.num_hyperedges as i32;
            let arg_nn: i32 = challenge.num_nodes as i32;
            let arg_np: i32 = challenge.num_parts as i32;
            let arg_mps: i32 = challenge.max_part_size as i32;
            let arg_round0: i32 = round_idx as i32;
            let arg_batch: i32 = batch as i32;
            let arg_ml: i32 = move_limit.min(i32::MAX as usize) as i32;
            let arg_ten: i32 = tabu_tenure as i32;
            let arg_mark_base: i32 = tabu_mark_base as i32;
            let arg_mark_mult: i32 = tabu_mark_mult as i32;
            let arg_tabu_exec: i32 = if tabu_exec { 1 } else { 0 };
            let arg_se: i32 = slack_early as i32;
            let arg_sm: i32 = slack_mid as i32;
            let arg_sl: i32 = slack_late as i32;
            let arg_extw: i32 = extra_window as i32;
            let arg_cend: i32 = cycle_end as i32;
            let arg_sexp: i32 = sched_exp as i32;
            let soft: i32 = if rf_soft { 1 } else { 0 };
            let epoch_arg: u32 = rf_epoch;
            {
                let mut b = stream.launch_builder(&rounds_fused_kernel);
                b.arg(&arg_nh)
                    .arg(&arg_nn)
                    .arg(&arg_np)
                    .arg(&arg_mps)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&mut d_edge_flags_pair)
                    .arg(&mut d_move_priorities)
                    .arg(&moves_lanes)
                    .arg(&tiebreak)
                    .arg(&gfloor)
                    .arg(&arg_round0)
                    .arg(&arg_batch)
                    .arg(&arg_ml)
                    .arg(&arg_ten)
                    .arg(&arg_mark_base)
                    .arg(&arg_mark_mult)
                    .arg(&arg_tabu_exec)
                    .arg(&arg_se)
                    .arg(&arg_sm)
                    .arg(&arg_sl)
                    .arg(&arg_extw)
                    .arg(&arg_cend)
                    .arg(&arg_sexp)
                    .arg(&num_mul)
                    .arg(&mut d_tabu_until)
                    .arg(&mut d_filt_stats)
                    .arg(&mut d_filt_off)
                    .arg(&mut d_filt_hdr)
                    .arg(&prob_den[0])
                    .arg(&prob_den[1])
                    .arg(&prob_den[2])
                    .arg(&prob_den[3])
                    .arg(&mut d_cand_out)
                    .arg(&soft)
                    .arg(&epoch_arg)
                    .arg(&n_tabu_upd)
                    .arg(&d_tabu_upd)
                    .arg(&n_part_upd)
                    .arg(&d_part_upd)
                    .arg(&mut d_acc)
                    .arg(&mut d_acc_off)
                    .arg(&mut d_cand_tmp)
                    .arg(&mut d_rs_counts)
                    .arg(&mut d_tcnt)
                    .arg(&mut d_koff);
                unsafe {
                    if rf_soft {
                        b.launch(round_cfg.clone())?;
                    } else {
                        b.launch_cooperative(round_cfg.clone())?;
                    }
                }
            }
            stream.memcpy_dtoh(&d_filt_hdr, &mut filt_hdr_host)?;
            rf_launch_idx += 1;
            let attempted = (filt_hdr_host[31].max(0) as usize).min(batch);
            let status = filt_hdr_host[25];
            let aborted = filt_hdr_host[28] != 0;
            if rf_soft {
                rf_epoch = rf_epoch.wrapping_add(attempted as u32);
                if aborted {


                    rf_soft = false;
                }
            }
            tabu_upd_host.clear();
            part_upd_host.clear();
            round_idx += attempted;
            total_moves_executed += filt_hdr_host[32].max(0) as usize;
            if aborted {
                continue 'rounds;
            }
            if attempted == 0 {
                break 'rounds;
            }
            if status == 0 {


                round = round_idx - 1;
                moves_executed = 1;
                break 'body;
            }
            if status == 1 {
                break 'rounds;
            }


            if attempted > 1 {
                stagnant_rounds = 0;
            }
            round = round_idx - 1;
            rf_device_round = true;
            refresh_host_mirrors_from_device!();
        } else {
            round = round_idx;
            round_idx += 1;
        }
        if use_round_fused {

        } else if use_host_flags {


            if !host_flags_valid {
                rebuild_host_flags!();
            }
            stream.memcpy_htod(&flags_all_host, &mut d_edge_flags_all)?;
            stream.memcpy_htod(&flags_double_host, &mut d_edge_flags_double)?;
        } else {
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
        }
        if !use_round_fused {
            launch_compute_moves!(gfloor);
        }


        let (cycle_idx, cycle_start, cycle_end) = if reheat_cycles > 1 {
            let c = std::cmp::min(round / cycle_len, reheat_cycles - 1);
            let s = c * cycle_len;
            let e = if c + 1 == reheat_cycles { refinement_rounds } else { s + cycle_len };
            (c, s, e)
        } else {
            (0usize, 0usize, refinement_rounds)
        };
        let (num_mul, den_mul): (u64, u64) = if reheat_cycles > 1 {
            (if cycle_idx > 0 { reheat_temp } else { 100 }, 100)
        } else {
            (1, 1)
        };
        let rounds_left = cycle_end.saturating_sub(round) as u64;
        let prob_num = rounds_left.pow(sched_exp).saturating_mul(num_mul);
        let cyc_len = (cycle_end - cycle_start) as u64;
        let rr = cyc_len.pow(sched_exp).saturating_mul(den_mul);
        let mut prob_den = [0u64; 4];
        let mut prob_inv = [0u64; 4];
        for p in 0..4 {
            prob_den[p] = rr.saturating_mul(1 + p as u64 * pen_mul);
            prob_inv[p] = if prob_den[p] > 0 { u64::MAX / prob_den[p] } else { 0 };
        }
        let prob_accept = |r: u64, p: usize| -> bool {
            let den = prob_den[p];
            if den == 0 {
                return false;
            }
            let q = ((r as u128 * prob_inv[p] as u128) >> 64) as u64;
            let mut rem = r.wrapping_sub(q.wrapping_mul(den));
            while rem >= den {
                rem -= den;
            }
            rem < prob_num
        };

        let mut num_valid_keys = 0usize;
        let mut max_gain = i32::MIN;
        if use_gfilter {


            if tabu_dirty && !use_round_fused {
                stream.memcpy_htod(&tabu_mirror, &mut d_tabu_until)?;
                tabu_dirty = false;
            }
            let seed = 123456789u64.wrapping_add(round as u64);
            if rf_device_round {


            } else if use_round_fused {


                let n_tabu_upd = (tabu_upd_host.len() / 2) as i32;
                if n_tabu_upd > 0 {
                    let n = tabu_upd_host.len();
                    stream.memcpy_htod(&tabu_upd_host[..n], &mut d_tabu_upd.slice_mut(0..n))?;
                }
                let n_part_upd = (part_upd_host.len() / 2) as i32;
                if n_part_upd > 0 {
                    let n = part_upd_host.len();
                    stream.memcpy_htod(&part_upd_host[..n], &mut d_part_upd.slice_mut(0..n))?;
                }

                let arg_nh: i32 = challenge.num_hyperedges as i32;
                let arg_nn: i32 = challenge.num_nodes as i32;
                let arg_np: i32 = challenge.num_parts as i32;
                let arg_mps: i32 = challenge.max_part_size as i32;
                let arg_round: u32 = round as u32;

                let arg_slack: i32 = if round < 64 {
                    slack_early
                } else if round < 256 {
                    slack_mid
                } else {
                    slack_late
                } as i32;
                let arg_alim: i32 = if round < 50 {
                    move_limit / 2
                } else if round < 200 {
                    (move_limit * 3) / 4
                } else {
                    move_limit / 4
                } as i32;
                let arg_extw: i32 = extra_window as i32;
                let mut attempt = 0;
                loop {
                    let soft: i32 = if rf_soft { 1 } else { 0 };
                    let launch_idx_arg: u32 = rf_launch_idx;
                    {
                    let mut b = stream.launch_builder(&round_fused_kernel);
                    b.arg(&arg_nh)
                        .arg(&arg_nn)
                        .arg(&arg_np)
                        .arg(&arg_mps)
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&mut d_partition)
                        .arg(&d_nodes_in_part)
                        .arg(&mut d_edge_flags_pair)
                        .arg(&mut d_move_priorities)
                        .arg(&moves_lanes)
                        .arg(&tiebreak)
                        .arg(&gfloor)
                        .arg(&arg_round)
                        .arg(&mut d_tabu_until)
                        .arg(&mut d_filt_stats)
                        .arg(&mut d_filt_off)
                        .arg(&mut d_filt_hdr)
                        .arg(&seed)
                        .arg(&prob_num)
                        .arg(&prob_den[0])
                        .arg(&prob_den[1])
                        .arg(&prob_den[2])
                        .arg(&prob_den[3])
                        .arg(&mut d_cand_out)
                        .arg(&soft)
                        .arg(&launch_idx_arg)
                        .arg(&n_tabu_upd)
                        .arg(&d_tabu_upd)
                        .arg(&n_part_upd)
                        .arg(&d_part_upd)
                        .arg(&mut d_acc)
                        .arg(&mut d_acc_off)
                        .arg(&mut d_cand_tmp)
                        .arg(&mut d_rs_counts)
                        .arg(&gsort_arg)
                        .arg(&mut d_tcnt)
                        .arg(&mut d_koff)
                        .arg(&arg_slack)
                        .arg(&arg_alim)
                        .arg(&arg_extw);
                    unsafe {
                        if rf_soft {
                            b.launch(round_cfg.clone())?;
                        } else {
                            b.launch_cooperative(round_cfg.clone())?;
                        }
                    }
                    }
                    stream.memcpy_dtoh(&d_filt_hdr, &mut filt_hdr_host)?;
                    if rf_soft {
                        rf_launch_idx += 1;


                        if filt_hdr_host[28] != 0 {
                            rf_soft = false;
                            attempt += 1;
                            if attempt < 3 {
                                continue;
                            }
                        }
                    }
                    break;
                }
                tabu_upd_host.clear();
                part_upd_host.clear();
            } else {
                unsafe {
                    stream
                        .launch_builder(&filter_fused_kernel)
                        .arg(&(challenge.num_nodes as i32))
                        .arg(&(round as u32))
                        .arg(&d_move_priorities)
                        .arg(&d_tabu_until)
                        .arg(&mut d_filt_stats)
                        .arg(&mut d_filt_off)
                        .arg(&mut d_filt_hdr)
                        .arg(&seed)
                        .arg(&prob_num)
                        .arg(&prob_den[0])
                        .arg(&prob_den[1])
                        .arg(&prob_den[2])
                        .arg(&prob_den[3])
                        .arg(&mut d_cand_out)
                        .launch_cooperative(fused_cfg.clone())?;
                }
                stream.memcpy_dtoh(&d_filt_hdr, &mut filt_hdr_host)?;
            }
            num_valid_keys = filt_hdr_host[0].max(0) as usize;
            max_gain = filt_hdr_host[2];
        } else {

            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
            for &k in move_keys_host.iter() {
                if k as u32 != 0x80000000 {
                    num_valid_keys += 1;
                    let g = k >> 16;
                    if g > max_gain {
                        max_gain = g;
                    }
                }
            }
        }
        if num_valid_keys == 0 {
            break 'rounds;
        }
        let aspiration_threshold = (max_gain * 3) / 4;

        let mut rng_state = 123456789u64.wrapping_add(round as u64);

        let adaptive_limit = if round < 50 {
            move_limit / 2
        } else if round < 200 {
            (move_limit * 3) / 4
        } else {
            move_limit / 4
        };

        let slack = if round < 64 {
            slack_early
        } else if round < 256 {
            slack_mid
        } else {
            slack_late
        };

        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        main_valid_moves.clear();
        let mut k_base = 0usize;
        let mut k_cand = 0usize;
        let mut window_sorted = false;


        let fast_path = !use_gfilter && num_valid_keys <= adaptive_limit;
        if fast_path {
            for b in tgt_buckets.iter_mut() {
                b.clear();
            }
            let mut filtered = 0usize;
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key as u32 != 0x80000000 {
                    let gain = key >> 16;
                    let is_tabu = node_tabu_until[node] > round && gain < aspiration_threshold;
                    if !is_tabu {
                        let mut accept_key: Option<i32> = None;
                        if gain > 0 {
                            accept_key = Some(key);
                        } else if gain >= -3 {
                            let penalty = if gain < 0 { (-gain) as usize } else { 0 };
                            rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
                            if prob_accept(rng_state, penalty) {
                                accept_key = Some((0 << 16) | (key & 0xFFFF));
                            }
                        }
                        if let Some(k) = accept_key {
                            filtered += 1;
                            let tgt = (k & 63) as usize;
                            if tgt < num_parts_usize {
                                tgt_buckets[tgt].push(((!(k as u32) as u64) << 32) | node as u64);
                            }
                        }
                    }
                }
            }
            if filtered == 0 {
                break 'rounds;
            }
            k_base = filtered;
            k_cand = filtered;
            for t in 0..num_parts_usize {
                let q = tgt_quota[t];
                let b = &mut tgt_buckets[t];
                if b.len() > q {
                    b.select_nth_unstable(q - 1);
                    b.truncate(q);
                }
                main_valid_moves.extend_from_slice(b);
            }
            main_valid_moves.sort_unstable();
            for &v in main_valid_moves.iter() {
                sorted_move_nodes.push(mv_node(v) as i32);
                sorted_move_parts.push((mv_key(v) & 63) as i32);
            }
        }


        macro_rules! fetch_sorted_window {
            ($n:expr) => {{
                let n_: usize = $n;
                main_valid_moves.resize(n_, 0u64);
                if n_ > 0 {
                    if filt_hdr_host[29] == 1 {
                        stream.memcpy_dtoh(&d_cand_tmp.slice(0..n_), &mut main_valid_moves[..n_])?;
                    } else {
                        stream.memcpy_dtoh(&d_cand_out.slice(0..n_), &mut main_valid_moves[..n_])?;
                    }
                }
                window_sorted = true;
            }};
        }
        if use_gsort {
            let cnt = (filt_hdr_host[3].max(0) as usize).min(challenge.num_nodes as usize);
            if cnt == 0 {
                break 'rounds;
            }
            k_base = std::cmp::min(cnt, adaptive_limit);
            k_cand = std::cmp::min(cnt, k_base.saturating_add(extra_window));
            window_sorted = true;
            let kept = (filt_hdr_host[30].max(0) as usize).min(cnt);
            let take = std::cmp::min(kept, k_base);
            if take > 0 {
                kept_host.resize(take, 0u64);
                if filt_hdr_host[29] == 1 {
                    stream.memcpy_dtoh(&d_cand_out.slice(0..take), &mut kept_host[..take])?;
                } else {
                    stream.memcpy_dtoh(&d_cand_tmp.slice(0..take), &mut kept_host[..take])?;
                }
                for &v in kept_host.iter() {
                    sorted_move_nodes.push(mv_node(v) as i32);
                    sorted_move_parts.push((mv_key(v) & 63) as i32);
                }
            }
            if sorted_move_nodes.is_empty() {
                fetch_sorted_window!(k_cand);
                let take = std::cmp::min(k_base, k_cand);
                sorted_move_nodes.extend(main_valid_moves[..take].iter().map(|&v| mv_node(v) as i32));
                sorted_move_parts.extend(main_valid_moves[..take].iter().map(|&v| (mv_key(v) & 63) as i32));
            }
        }


        if sorted_move_nodes.is_empty() && !use_gsort {
            main_valid_moves.clear();
            if use_gfilter {

                let cnt = (filt_hdr_host[3].max(0) as usize).min(challenge.num_nodes as usize);
                if cnt > 0 {
                    main_valid_moves.resize(cnt, 0u64);
                    stream.memcpy_dtoh(&d_cand_out.slice(0..cnt), &mut main_valid_moves[..cnt])?;
                }
            } else {
                rng_state = 123456789u64.wrapping_add(round as u64);
                for (node, &key) in move_keys_host.iter().enumerate() {
                    if key as u32 != 0x80000000 {
                        let gain = key >> 16;
                        let is_tabu = node_tabu_until[node] > round && gain < aspiration_threshold;
                        if !is_tabu {
                            if gain > 0 {
                                main_valid_moves.push(((!(key as u32) as u64) << 32) | node as u64);
                            } else if gain >= -3 {
                                let penalty = if gain < 0 { (-gain) as usize } else { 0 };
                                rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
                                if prob_accept(rng_state, penalty) {
                                    let new_key = (0 << 16) | (key & 0xFFFF);
                                    main_valid_moves.push(((!(new_key as u32) as u64) << 32) | node as u64);
                                }
                            }
                        }
                    }
                }
            }

            if main_valid_moves.is_empty() {
                break 'rounds;
            }

            k_base = main_valid_moves.len();
            if k_base > adaptive_limit {
                k_base = adaptive_limit;
            }

            k_cand = std::cmp::min(main_valid_moves.len(), k_base.saturating_add(extra_window));


            if k_cand < main_valid_moves.len() {
                main_valid_moves.select_nth_unstable(k_cand - 1);
            }


            for b in tgt_buckets.iter_mut() {
                b.clear();
            }
            for &v in main_valid_moves[..k_cand].iter() {
                let tgt = (mv_key(v) & 63) as usize;
                if tgt < num_parts_usize {
                    tgt_buckets[tgt].push(v);
                }
            }
            gen_accepted.clear();
            for t in 0..num_parts_usize {
                let q = tgt_quota[t];
                let b = &mut tgt_buckets[t];
                if b.len() > q {
                    b.select_nth_unstable(q - 1);
                    b.truncate(q);
                }
                gen_accepted.extend_from_slice(b);
            }
            gen_accepted.sort_unstable();
            gen_accepted.truncate(k_base);
            for &v in gen_accepted.iter() {
                sorted_move_nodes.push(mv_node(v) as i32);
                sorted_move_parts.push((mv_key(v) & 63) as i32);
            }

            if sorted_move_nodes.is_empty() {
                main_valid_moves[..k_cand].sort_unstable();
                window_sorted = true;
                let take = std::cmp::min(k_base, k_cand);
                sorted_move_nodes.extend(main_valid_moves[..take].iter().map(|&v| mv_node(v) as i32));
                sorted_move_parts.extend(main_valid_moves[..take].iter().map(|&v| (mv_key(v) & 63) as i32));
            }
        }


        moves_executed = exec_moves_mirror!(sorted_move_nodes, sorted_move_parts);

        if moves_executed == 0 && k_cand > k_base {
            let fail_mark_len = std::cmp::min(sorted_move_nodes.len(), tabu_fail_mark_len);
            for &node in sorted_move_nodes.iter().take(fail_mark_len) {
                node_tabu_until[node as usize] = round + tabu_fail_tenure;
                tabu_mirror[node as usize] = (round + tabu_fail_tenure) as u32;
                tabu_upd_host.push(node as u32);
                tabu_upd_host.push((round + tabu_fail_tenure) as u32);
            }
            tabu_dirty = true;

            sorted_move_nodes.clear();
            sorted_move_parts.clear();


            if use_gsort && main_valid_moves.len() < k_cand {
                fetch_sorted_window!(k_cand);
            }
            if !window_sorted {
                main_valid_moves[..k_cand].sort_unstable();
            }
            let tail = &main_valid_moves[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            sorted_move_nodes.extend(tail.iter().take(take).map(|&v| mv_node(v) as i32));
            sorted_move_parts.extend(tail.iter().take(take).map(|&v| (mv_key(v) & 63) as i32));

            if !sorted_move_nodes.is_empty() {
                moves_executed = exec_moves_mirror!(sorted_move_nodes, sorted_move_parts);
            }
        }

        total_moves_executed += moves_executed as usize;

        if moves_executed > 0 {
            let until = round + tabu_tenure;


            if tabu_exec {
                for &node in exec_nodes.iter() {
                    node_tabu_until[node as usize] = until;
                    tabu_mirror[node as usize] = until as u32;
                    tabu_upd_host.push(node as u32);
                    tabu_upd_host.push(until as u32);
                }
            } else {
                let mark_len = std::cmp::min(
                    sorted_move_nodes.len(),
                    std::cmp::max(
                        tabu_mark_base,
                        (moves_executed as usize).saturating_mul(tabu_mark_mult),
                    ),
                );
                for &node in sorted_move_nodes.iter().take(mark_len) {
                    node_tabu_until[node as usize] = until;
                    tabu_mirror[node as usize] = until as u32;
                    tabu_upd_host.push(node as u32);
                    tabu_upd_host.push(until as u32);
                }
            }
            tabu_dirty = true;
        }
        }


        let (_, _, cycle_end_t) = cycle_bounds(round);
        if reheat_cycles > 1 && round + 1 == cycle_end_t {
            if use_multi {

                refresh_host_mirrors_from_device!();
            }
            let conn = eval_connectivity!();
            if conn < best_loop_conn {
                best_loop_conn = conn;
                best_loop_partition.clear();
                best_loop_partition.extend_from_slice(&partition_host_refine);
                best_loop_nodes_in_part.clear();
                best_loop_nodes_in_part.extend_from_slice(&nodes_in_part_host);
            } else if !best_loop_partition.is_empty() {
                restore_loop_best!();
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
                part_upd_host.clear();

                host_flags_valid = false;
                total_perturbations += 1;
                stagnant_rounds = 0;
            } else if stagnant_rounds > max_stagnant_rounds {
                break;
            }
        } else {
            stagnant_rounds = 0;
        }
    }


    flush_part_upd!();
    if use_multi {


        refresh_host_mirrors_from_device!();
    }
    tabu_upd_host.clear();


    if reheat_cycles > 1 && !best_loop_partition.is_empty() {
        let conn = eval_connectivity!();
        if conn > best_loop_conn {
            restore_loop_best!();
        }
    }


    let _ = host_flags_valid;

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
            let swap_rounds_eff = ((($max_rounds as u64) * swap_scale) / 100) as usize;
            let mut swap_rounds_done = 0usize;
            for _swap_round in 0..swap_rounds_eff {
                swap_rounds_done += 1;
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

            anyhow::Ok(swap_rounds_done * 1_000_000 + total_swaps)
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
        launch_compute_moves!(gfloor_off);
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

    let perturb_strength = 3;

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

    let num_high_hedges = std::cmp::min(500usize, challenge.num_hyperedges as usize);
    let mut high_hedge_ids_host: Vec<i32> = build_high_hedge_ids(&conn_vec, num_high_hedges);


    let pool_size = std::cmp::max(1, ils_iterations);
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
            launch_compute_moves!(gfloor_off);

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
        launch_compute_moves!(gfloor_off);

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
        launch_compute_moves!(gfloor_off);

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

    let partition = stream.memcpy_dtov(&d_partition)?;
    let mut partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    save_solution(&Solution {
        partition: partition_u32.clone(),
    })?;


    let hp_i64 = |name: &str, def: i64| -> i64 {
        hyperparameters
            .as_ref()
            .and_then(|p| p.get(name).and_then(|v| v.as_i64()))
            .unwrap_or(def)
    };
    let href = hp_i64("href", 0) != 0;
    let fuel_at_start = super::fuel_remaining();
    if href {


        let fuel_reserve: u64 = hp_i64("hfuel_reserve", 5_000_000_000).max(0) as u64;


        let fuel_budget: u64 = hp_i64("hfuel_budget", 12_000_000_000).max(0) as u64;
        let budget_ok = move || -> bool {
            let f = super::fuel_remaining();
            if fuel_at_start == 0 {
                return true;
            }
            if f <= fuel_reserve {
                return false;
            }
            if fuel_budget > 0 && fuel_at_start.saturating_sub(f) >= fuel_budget {
                return false;
            }
            true
        };
        let params = super::hrefine::Params {
            fm_rounds: hp_i64("hfm_rounds", 4).clamp(1, 500) as usize,
            fm_noimprove: hp_i64("hfm_noimprove", 12).clamp(1, 5000) as usize,
            fm_deficit: hp_i64("hfm_deficit", 3).clamp(0, 1_000_000) as i32,
            fm_max_steps: hp_i64("hfm_steps", 400).clamp(1, 100_000) as usize,
            fm_seeds: hp_i64("hfm_seeds", 20).clamp(1, 10_000) as usize,
            fm_touch_cap: hp_i64("hfm_touch", 48).clamp(2, 100_000) as usize,
            hcm_max_size: hp_i64("hcm_size", 24).clamp(2, 100_000) as usize,
            hcm_max_pins: hp_i64("hcm_pins", 3).clamp(1, 64) as u32,
            passes: hp_i64("hpasses", 1).clamp(1, 100) as usize,
            ils_iters: hp_i64("hils", 0).clamp(0, 100_000) as usize,
            ils_strength: hp_i64("hils_strength", 24).clamp(1, 1_000_000) as usize,
            flow: hp_i64("hflow", 0) != 0,
            flow_rounds: hp_i64("hflow_rounds", 3).clamp(1, 100) as usize,
            flow_params: super::hflow::FlowParams {
                alpha: hp_i64("hflow_alpha", 4).clamp(1, 64) as u32,
                max_region: hp_i64("hflow_region", 96).clamp(1, 100_000) as usize,
                edge_cap: hp_i64("hflow_ecap", 48).clamp(2, 100_000) as usize,
                min_shared: hp_i64("hflow_min", 3).clamp(1, 1_000_000) as usize,
            },
            flow_in_cycles: hp_i64("hflow_cycles", 0) != 0,
            vcycles: hp_i64("hvcycles", 1).clamp(0, 1000) as usize,
            ml_levels: hp_i64("hml_levels", 3).clamp(1, 20) as usize,
            ml_max_cluster: hp_i64("hml_cluster", 0).clamp(0, 100_000) as u32,
            ml_patience: hp_i64("hml_patience", 2).clamp(1, 1000) as usize,
            ml_fine_full: hp_i64("hml_fine_full", 0) != 0,
            ml_coarse_full: hp_i64("hml_coarse_full", 1) != 0,
            jet_rounds: hp_i64("hjet", 0).clamp(0, 10_000) as usize,
            jet_tolerance: hp_i64("hjet_tol", 6).clamp(1, 10_000) as usize,
            jet_neg_pct: hp_i64("hjet_c", 25).clamp(0, 100) as u32,
            jet_min_gain: hp_i64("hjet_min", 0).clamp(-1_000_000, 1_000_000) as i32,
            jet_stages: hp_i64("hjet_stage", 3).clamp(0, 3) as u32,
            jet_slack: hp_i64("hjet_slack", -1).clamp(-1, 1_000_000) as i32,
            seed: 0x5EED_5EED_1234_ABCDu64
                ^ u64::from_le_bytes([
                    challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
                    challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
                ]),
        };
        let he_off: Vec<u32> = hedge_offsets_host.iter().map(|&x| x as u32).collect();
        let he_nodes: Vec<u32> = hyperedge_nodes_host.iter().map(|&x| x as u32).collect();
        let rep = super::hrefine::improve(
            challenge.num_nodes as usize,
            challenge.num_hyperedges as usize,
            challenge.num_parts as usize,
            he_off,
            he_nodes,
            challenge.max_part_size,
            &mut partition_u32,
            &params,
            &budget_ok,
        );

        let mut sizes = vec![0u32; challenge.num_parts as usize];
        let mut ok = partition_u32.len() == challenge.num_nodes as usize;
        for &p in partition_u32.iter() {
            if (p as usize) < sizes.len() { sizes[p as usize] += 1; } else { ok = false; }
        }
        ok = ok && sizes.iter().all(|&s| s >= 1 && s <= challenge.max_part_size);
        if ok && rep.after < rep.before {
            save_solution(&Solution {
                partition: partition_u32,
            })?;
        }
    }

    let _ = (fuel_at_start, href);
    Ok(())
}
