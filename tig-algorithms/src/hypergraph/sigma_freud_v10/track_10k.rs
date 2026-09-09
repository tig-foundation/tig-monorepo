use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;


fn align_part_labels(src: &[i32], reference: &[i32], num_parts: usize) -> Vec<i32> {
    let np = num_parts;
    let mut overlap = vec![0i64; np * np];
    for (&a, &b) in src.iter().zip(reference.iter()) {
        if a >= 0 && (a as usize) < np && b >= 0 && (b as usize) < np {
            overlap[a as usize * np + b as usize] += 1;
        }
    }
    let max_ov = overlap.iter().copied().max().unwrap_or(0);

    let inf = i64::MAX / 4;
    let mut u = vec![0i64; np + 1];
    let mut v = vec![0i64; np + 1];
    let mut p = vec![0usize; np + 1];
    let mut way = vec![0usize; np + 1];
    for i in 1..=np {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; np + 1];
        let mut used = vec![false; np + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=np {
                if !used[j] {
                    let cur = (max_ov - overlap[(i0 - 1) * np + (j - 1)]) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=np {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut map = vec![0i32; np];
    for j in 1..=np {
        if p[j] > 0 {
            map[p[j] - 1] = (j - 1) as i32;
        }
    }
    src.iter()
        .map(|&a| if a >= 0 && (a as usize) < np { map[a as usize] } else { a })
        .collect()
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let block_size = std::cmp::min(256, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_10k")?;
    let hyperedge_cluster_sb_kernel = module.load_function("hyperedge_clustering_sb_10k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_10k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_10k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_10k")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves_lanes_10k")?;
    let execute_moves_kernel = module.load_function("execute_refinement_moves_10k")?;
    let execute_moves_packed_kernel = module.load_function("execute_refinement_moves_packed_10k")?;
    let balance_kernel = module.load_function("balance_final_10k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_10k")?;
    let perturb_kernel = module.load_function("perturb_solution_10k")?;
    let perturb_guided_kernel = module.load_function("perturb_guided_10k")?;
    let perturb_hubs_kernel = module.load_function("perturb_hubs_10k")?;
    let perturb_ruin_recreate_kernel = module.load_function("perturb_ruin_recreate_10k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_10k")?;
    let perturb_path_relink_kernel = module.load_function("perturb_path_relink_10k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_10k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_10k")?;
    let balance_find_best_under_kernel = module.load_function("balance_find_best_under_10k")?;
    let balance_find_best_over_kernel = module.load_function("balance_find_best_over_10k")?;
    let apply_balance_move_kernel = module.load_function("apply_balance_move_10k")?;
    let rebalance_parts_kernel = module.load_function("rebalance_parts_10k")?;
    let compute_swap_topk_kernel = module.load_function("compute_swap_topk_10k")?;
    let compute_best_swap_pairs_kernel = module.load_function("compute_best_swap_pairs_10k")?;
    let compute_hedge_consolidation_kernel = module.load_function("compute_hedge_consolidation_moves_10k")?;
    let polish_exploration_kernel = module.load_function("polish_exploration_10k")?;
    let compute_part_cut_cost_kernel = module.load_function("compute_part_cut_cost_10k")?;
    let repair_bottleneck_part_kernel = module.load_function("repair_bottleneck_part_10k")?;
    let round_fused_kernel = module.load_function("round_fused_10k")?;
    let rounds_fused_kernel = module.load_function("rounds_fused_10k")?;

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

    let balance_cfg = cfg.clone();


    let moves_lanes: i32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("mlanes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 32) as i32)
        .map(|v| 1 << (31 - (v as u32).leading_zeros()))
        .unwrap_or(4);
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

    let k_hashes: i32 = 4;
    let mut hash_a = vec![0i32; k_hashes as usize];
    let mut hash_b = vec![0i32; k_hashes as usize];
    {
        let mut rng = init_random_seed;
        for i in 0..k_hashes as usize {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            hash_a[i] = (rng & 0x7FFFFFFF) as i32;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            hash_b[i] = (rng & 0x7FFFFFFF) as i32;
        }
    }
    let d_hash_a = stream.memcpy_stod(&hash_a)?;
    let d_hash_b = stream.memcpy_stod(&hash_b)?;
    let mut d_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;

    let mut d_pref_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let grid_x_calc = (challenge.num_nodes as u32 + block_size - 1) / block_size;


    let counter_slots = std::cmp::max(grid_x_calc, moves_grid_max) as usize;
    let mut d_num_valid_moves = stream.alloc_zeros::<i32>(counter_slots)?;
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
        5 => (18000, 40, 300, 300, 0),
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
    let scan_limit_cycle = 8usize;

    let swap_buf_size = 4 * challenge.num_nodes as usize;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;
    let swap_topk_size = num_parts_usize * num_parts_usize * 32 * 2;
    let mut d_swap_topk = stream.alloc_zeros::<i32>(swap_topk_size)?;
    let mut swap_topk_host: Vec<i32> = vec![0i32; swap_topk_size];

    let mut partition_host_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut nodes_in_part_host: Vec<i32> = vec![0i32; num_parts_usize];
    let mut move_keys_host: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let zero_counter_1 = [0i32; 1];
    let zero_counter_grid = vec![0i32; counter_slots];
    let mut partition_host_mirror: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut nodes_in_part_mirror: Vec<i32> = vec![0i32; num_parts_usize];
    let mut accepted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut accepted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut d_accepted_move_nodes = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_accepted_move_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let mut exec_packed_host: Vec<i32> = Vec::with_capacity(num_parts_usize + 2 * challenge.num_nodes as usize);
    let mut d_exec_packed = stream.alloc_zeros::<i32>(num_parts_usize + 2 * challenge.num_nodes as usize)?;
    let mut main_valid_moves: Vec<u64> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut tgt_buckets: Vec<Vec<u64>> = (0..num_parts_usize).map(|_| Vec::with_capacity(256)).collect();
    let mut d_hedge_choice = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let max_moves_per_hedge = 16usize;
    let mut d_move_node_hedge = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize * max_moves_per_hedge)?;
    let mut d_move_part_hedge = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize * max_moves_per_hedge)?;
    let mut d_move_count_hedge = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_balance_best_node = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_balance_best_gain = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_balance_best_deg = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_balance_best_target = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;


    let use_rebalance_kernel = grid_x_calc <= 1024
        && block_size % 32 == 0
        && hyperparameters
            .as_ref()
            .and_then(|p| p.get("rebal").and_then(|v| v.as_i64()))
            .map(|v| v != 0)
            .unwrap_or(true);
    let mut d_rb_gains = stream.alloc_zeros::<i32>(challenge.num_nodes as usize * 64)?;
    let mut d_rb_nb = stream.alloc_zeros::<i32>(challenge.num_nodes as usize * 3)?;
    let mut d_rb_moves = stream.alloc_zeros::<i32>(2)?;
    let rb_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut d_part_cut_costs = stream.alloc_zeros::<i32>(num_parts_usize)?;
    let mut d_bottleneck_part = stream.alloc_zeros::<i32>(1)?;
    let mut d_repair_gains = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_repair_targets = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_polish_backup_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_polish_backup_nip = stream.alloc_zeros::<i32>(num_parts_usize)?;
    let mut d_polish_best_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_polish_best_nip = stream.alloc_zeros::<i32>(num_parts_usize)?;

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


    let n_he_host = challenge.num_hyperedges as usize;
    let he_offsets_host: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let he_nodes_host: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let node_offsets_host: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let node_hyperedges_host: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
    let mut edge_part_cnt: Vec<u16> = vec![0u16; n_he_host * 64];
    let mut flags_all_host: Vec<u64> = vec![0u64; n_he_host];
    let mut flags_double_host: Vec<u64> = vec![0u64; n_he_host];
    let mut host_flags_valid = false;


    let use_round_fused = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfuse").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        != 0;


    let rmulti: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rmulti").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 256) as usize)
        .unwrap_or(8);
    let use_multi = use_round_fused && rmulti > 1;


    let use_ils_fused = use_multi
        && hyperparameters
            .as_ref()
            .and_then(|p| p.get("ils_fused").and_then(|v| v.as_i64()))
            .map(|v| v != 0)
            .unwrap_or(true);
    let mut rf_epoch: u32 = 0;
    let rf_tiles = (challenge.num_nodes as usize + 127) / 128;
    let rf_nt_max = (challenge.num_nodes as usize + 255) / 256;


    let mut d_edge_flags_pair = stream.alloc_zeros::<u64>(2 * n_he_host)?;

    let mut d_rf_tabu_until = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;
    let mut d_rf_stats = stream.alloc_zeros::<i32>(rf_tiles * 8)?;


    let mut d_rf_hdr = stream.alloc_zeros::<i32>(32)?;
    let mut rf_hdr_host: Vec<i32> = vec![0i32; 32];
    let rf_hdr_zero: Vec<i32> = vec![0i32; 32];
    let mut d_rf_out = stream.alloc_zeros::<u64>(challenge.num_nodes as usize)?;
    let mut d_rf_out2 = stream.alloc_zeros::<u64>(challenge.num_nodes as usize)?;
    let mut d_rf_acc = stream.alloc_zeros::<u32>(challenge.num_nodes as usize)?;
    let mut d_rf_acc_off = stream.alloc_zeros::<u32>(rf_tiles)?;
    let mut d_rf_rs_counts = stream.alloc_zeros::<u32>(4 * 256 * rf_nt_max)?;
    let mut d_rf_tcnt = stream.alloc_zeros::<u32>(64 * rf_nt_max)?;
    let mut d_rf_koff = stream.alloc_zeros::<u32>(rf_nt_max)?;
    let mut d_rf_tabu_upd = stream.alloc_zeros::<u32>(2 * challenge.num_nodes as usize)?;
    let mut d_rf_part_upd = stream.alloc_zeros::<i32>(2 * challenge.num_nodes as usize)?;
    let mut tabu_upd_host: Vec<u32> = Vec::with_capacity(2 * challenge.num_nodes as usize);
    let mut part_upd_host: Vec<i32> = Vec::with_capacity(2 * challenge.num_nodes as usize);
    let mut kept_host: Vec<u64> = Vec::with_capacity(challenge.num_nodes as usize);


    let mut defer_part_upload = false;


    let mut rf_tabu_pairs_active = false;

    let rf_smem_words = std::cmp::max(moves_groups as usize * 64, 8 * 256);
    let rf_smem_bytes = (rf_smem_words * std::mem::size_of::<u32>()) as u32;


    let rf_max_per_sm = if use_multi {
        rounds_fused_kernel
            .occupancy_max_active_blocks_per_multiprocessor(256, rf_smem_bytes as usize, None)
            .unwrap_or(1)
            .max(1)
    } else {
        round_fused_kernel
            .occupancy_max_active_blocks_per_multiprocessor(256, rf_smem_bytes as usize, None)
            .unwrap_or(1)
            .max(1)
    };


    let rf_grid_req = if hyperparameters
        .as_ref()
        .and_then(|p| p.get("mblocks"))
        .is_some()
    {
        moves_grid_x
    } else {
        (challenge.num_nodes as u32 + moves_groups - 1) / moves_groups
    };
    let rf_grid = std::cmp::min(
        rf_grid_req,
        (prop.multiProcessorCount as u32).max(1) * rf_max_per_sm,
    )
    .max(1);
    let rf_cfg = LaunchConfig {
        grid_dim: (rf_grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: rf_smem_bytes,
    };
    let mut rf_soft = true;
    let mut rf_launch_idx: u32 = 0;
    macro_rules! flush_part_upd {
        () => {{
            if !part_upd_host.is_empty() {
                stream.memcpy_htod(&partition_host_mirror, &mut d_partition)?;
                part_upd_host.clear();
            }
        }};
    }

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
            host_flags_valid = false;
        }};
    }


    macro_rules! rebalance_parts {
        () => {{
            let mut rb_under = 0i32;
            let mut rb_over = 0i32;
            if use_rebalance_kernel {
                let arg_nn: i32 = challenge.num_nodes as i32;
                let arg_np: i32 = challenge.num_parts as i32;
                let arg_min: i32 = 1;
                let arg_max: i32 = challenge.max_part_size as i32;
                let arg_vb: i32 = block_size as i32;
                unsafe {
                    stream
                        .launch_builder(&rebalance_parts_kernel)
                        .arg(&arg_nn)
                        .arg(&arg_np)
                        .arg(&arg_min)
                        .arg(&arg_max)
                        .arg(&arg_vb)
                        .arg(&challenge.d_node_offsets)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&mut d_partition)
                        .arg(&mut d_nodes_in_part)
                        .arg(&d_edge_flags_all)
                        .arg(&d_edge_flags_double)
                        .arg(&mut d_rb_gains)
                        .arg(&mut d_rb_nb)
                        .arg(&mut d_rb_moves)
                        .launch(rb_cfg.clone())?;
                }
                let mv = stream.memcpy_dtov(&d_rb_moves)?;
                rb_under = mv[0];
                rb_over = mv[1];
            } else {
                let min_part_size = 1i32;
                stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
                for part in 0..num_parts_usize {
                    while nodes_in_part_host[part] < min_part_size {
                        unsafe {
                            stream
                                .launch_builder(&balance_find_best_under_kernel)
                                .arg(&(challenge.num_nodes as i32))
                                .arg(&(challenge.num_parts as i32))
                                .arg(&(part as i32))
                                .arg(&min_part_size)
                                .arg(&challenge.d_node_offsets)
                                .arg(&challenge.d_node_hyperedges)
                                .arg(&d_partition)
                                .arg(&d_nodes_in_part)
                                .arg(&d_edge_flags_all)
                                .arg(&d_edge_flags_double)
                                .arg(&mut d_balance_best_node)
                                .arg(&mut d_balance_best_gain)
                                .arg(&mut d_balance_best_deg)
                                .launch(balance_cfg.clone())?;
                        }
                        let best_node_vec = stream.memcpy_dtov(&d_balance_best_node)?;
                        let best_gain_vec = stream.memcpy_dtov(&d_balance_best_gain)?;
                        let best_deg_vec = stream.memcpy_dtov(&d_balance_best_deg)?;
                        let mut best_node = -1i32;
                        let mut best_gain = -999999i32;
                        let mut best_deg = i32::MAX;
                        for i in 0..best_node_vec.len() {
                            if best_gain_vec[i] > best_gain
                                || (best_gain_vec[i] == best_gain && best_deg_vec[i] < best_deg)
                            {
                                best_gain = best_gain_vec[i];
                                best_node = best_node_vec[i];
                                best_deg = best_deg_vec[i];
                            }
                        }
                        if best_node >= 0 {
                            let src_part = partition_host_mirror[best_node as usize];
                            partition_host_mirror[best_node as usize] = part as i32;
                            nodes_in_part_host[src_part as usize] -= 1;
                            nodes_in_part_host[part] += 1;
                            rb_under += 1;
                            unsafe {
                                stream
                                    .launch_builder(&apply_balance_move_kernel)
                                    .arg(&best_node)
                                    .arg(&(part as i32))
                                    .arg(&mut d_partition)
                                    .arg(&mut d_nodes_in_part)
                                    .launch(one_thread_cfg.clone())?;
                            }
                        } else {
                            break;
                        }
                    }
                }
                for part in 0..num_parts_usize {
                    while nodes_in_part_host[part] > challenge.max_part_size as i32 {
                        unsafe {
                            stream
                                .launch_builder(&balance_find_best_over_kernel)
                                .arg(&(challenge.num_nodes as i32))
                                .arg(&(challenge.num_parts as i32))
                                .arg(&(part as i32))
                                .arg(&(challenge.max_part_size as i32))
                                .arg(&challenge.d_node_offsets)
                                .arg(&challenge.d_node_hyperedges)
                                .arg(&d_partition)
                                .arg(&d_nodes_in_part)
                                .arg(&d_edge_flags_all)
                                .arg(&d_edge_flags_double)
                                .arg(&mut d_balance_best_node)
                                .arg(&mut d_balance_best_target)
                                .arg(&mut d_balance_best_gain)
                                .arg(&mut d_balance_best_deg)
                                .launch(balance_cfg.clone())?;
                        }
                        let best_node_vec = stream.memcpy_dtov(&d_balance_best_node)?;
                        let best_target_vec = stream.memcpy_dtov(&d_balance_best_target)?;
                        let best_gain_vec = stream.memcpy_dtov(&d_balance_best_gain)?;
                        let best_deg_vec = stream.memcpy_dtov(&d_balance_best_deg)?;
                        let mut best_node = -1i32;
                        let mut best_target = -1i32;
                        let mut best_gain = -999999i32;
                        let mut best_deg = i32::MAX;
                        for i in 0..best_node_vec.len() {
                            if best_gain_vec[i] > best_gain
                                || (best_gain_vec[i] == best_gain && best_deg_vec[i] < best_deg)
                            {
                                best_gain = best_gain_vec[i];
                                best_node = best_node_vec[i];
                                best_target = best_target_vec[i];
                                best_deg = best_deg_vec[i];
                            }
                        }
                        if best_node >= 0 && best_target >= 0 {
                            let old_part = partition_host_mirror[best_node as usize];
                            partition_host_mirror[best_node as usize] = best_target;
                            nodes_in_part_host[old_part as usize] -= 1;
                            nodes_in_part_host[best_target as usize] += 1;
                            rb_over += 1;
                            unsafe {
                                stream
                                    .launch_builder(&apply_balance_move_kernel)
                                    .arg(&best_node)
                                    .arg(&best_target)
                                    .arg(&mut d_partition)
                                    .arg(&mut d_nodes_in_part)
                                    .launch(one_thread_cfg.clone())?;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
            (rb_under, rb_over)
        }};
    }


    macro_rules! rebuild_host_flags {
        () => {{
            edge_part_cnt.fill(0);
            let nn = challenge.num_nodes as u32;
            for e in 0..n_he_host {
                let s = he_offsets_host[e] as usize;
                let t = he_offsets_host[e + 1] as usize;
                let base = e * 64;
                let mut fa = 0u64;
                let mut fd = 0u64;
                for k in s..t {
                    let v = he_nodes_host[k];
                    if (v as u32) < nn {
                        let p = partition_host_mirror[v as usize];
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


    macro_rules! replay_moves_mirror {
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
                                let ns = node_offsets_host[node] as usize;
                                let nt = node_offsets_host[node + 1] as usize;
                                let mut exact_safe = current_part_usize < 64 && target_part < 64;
                                let mut exact_delta = 0i32;
                                let mut examined = std::collections::BTreeSet::<usize>::new();

                                for k in ns..nt {
                                    let hedge_i32 = node_hyperedges_host[k];
                                    if hedge_i32 < 0 {
                                        exact_safe = false;
                                        break;
                                    }
                                    let hedge = hedge_i32 as usize;
                                    if hedge >= n_he_host || !examined.insert(hedge) {
                                        continue;
                                    }
                                    let hs_i32 = he_offsets_host[hedge];
                                    let ht_i32 = he_offsets_host[hedge + 1];
                                    if hs_i32 < 0 || ht_i32 < hs_i32
                                        || ht_i32 as usize > he_nodes_host.len()
                                    {
                                        exact_safe = false;
                                        break;
                                    }

                                    let mut before = 0u64;
                                    let mut source_pins = 0usize;
                                    let mut moving_pins = 0usize;
                                    for q in hs_i32 as usize..ht_i32 as usize {
                                        let v_i32 = he_nodes_host[q];
                                        if v_i32 < 0 || v_i32 as usize >= partition_host_mirror.len() {
                                            exact_safe = false;
                                            break;
                                        }
                                        let v = v_i32 as usize;
                                        let p = partition_host_mirror[v];
                                        if p < 0 || p as usize >= 64 {
                                            exact_safe = false;
                                            break;
                                        }
                                        before |= 1u64 << p;
                                        if p as usize == current_part_usize {
                                            source_pins += 1;
                                        }
                                        if v == node {
                                            moving_pins += 1;
                                        }
                                    }
                                    if !exact_safe || moving_pins == 0 {
                                        exact_safe = false;
                                        break;
                                    }

                                    let mut after = before;
                                    if source_pins == moving_pins {
                                        after &= !(1u64 << current_part_usize);
                                    }
                                    after |= 1u64 << target_part;
                                    exact_delta += after.count_ones() as i32 - before.count_ones() as i32;
                                }

                                if !exact_safe || exact_delta > 0 {
                                    continue;
                                }

                                partition_host_mirror[node] = target_part as i32;
                                nodes_in_part_mirror[current_part_usize] -= 1;
                                nodes_in_part_mirror[target_part] += 1;
                                if host_flags_valid {
                                    update_host_flags_for_move!(node, current_part_usize, target_part);
                                }
                                accepted_move_nodes.push(node_i32);
                                accepted_move_parts.push(target_part_i32);
                                host_moves_executed += 1;
                            }
                        }
                    }
                }
            }
            host_moves_executed
        }};
    }


    macro_rules! replay_execute_moves_upload {
        ($sorted_nodes:expr, $sorted_parts:expr) => {{
            let host_moves_executed = replay_moves_mirror!($sorted_nodes, $sorted_parts);
            if host_moves_executed > 0 {
                if defer_part_upload {

                    for i in 0..accepted_move_nodes.len() {
                        part_upd_host.push(accepted_move_nodes[i]);
                        part_upd_host.push(accepted_move_parts[i]);
                    }
                } else {
                    stream.memcpy_htod(&partition_host_mirror, &mut d_partition)?;
                }
                stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
            }
            host_moves_executed
        }};
    }

    macro_rules! replay_execute_moves_host {
        ($sorted_nodes:expr, $sorted_parts:expr) => {{
            let host_moves_executed = replay_moves_mirror!($sorted_nodes, $sorted_parts);

            if host_moves_executed > 0 {
                let accepted_len = accepted_move_nodes.len();


                exec_packed_host.clear();
                exec_packed_host.extend_from_slice(&nodes_in_part_mirror);
                exec_packed_host.extend_from_slice(accepted_move_nodes.as_slice());
                exec_packed_host.extend_from_slice(accepted_move_parts.as_slice());
                stream.memcpy_htod(exec_packed_host.as_slice(), &mut d_exec_packed)?;
                let threads_needed = std::cmp::max(accepted_len, num_parts_usize) as u32;
                unsafe {
                    stream
                        .launch_builder(&execute_moves_packed_kernel)
                        .arg(&(accepted_len as i32))
                        .arg(&(num_parts_usize as i32))
                        .arg(&d_exec_packed)
                        .arg(&mut d_partition)
                        .arg(&mut d_nodes_in_part)
                        .launch(LaunchConfig {
                            grid_dim: ((threads_needed + block_size - 1) / block_size, 1, 1),
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
                    .launch_builder(&compute_hedge_consolidation_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&mut d_move_node_hedge)
                    .arg(&mut d_move_part_hedge)
                    .arg(&mut d_move_count_hedge)
                    .launch(hedge_cfg.clone())?;
            }
            let move_counts = stream.memcpy_dtov(&d_move_count_hedge)?;
            let move_node_hedge_host = stream.memcpy_dtov(&d_move_node_hedge)?;
            let move_part_hedge_host = stream.memcpy_dtov(&d_move_part_hedge)?;
            valid_moves.clear();
            let mut node_included = vec![false; challenge.num_nodes as usize];
            for hedge_idx in 0..challenge.num_hyperedges as usize {
                let cnt = move_counts[hedge_idx] as usize;
                if cnt == 0 { continue; }
                let base = hedge_idx * max_moves_per_hedge;
                for m in 0..cnt {
                    let node = move_node_hedge_host[base + m] as usize;
                    let packed = move_part_hedge_host[base + m] as i32;
                    let target = packed & 63;
                    let gain_est = (packed >> 6) & 0x3FF;
                    if node < challenge.num_nodes as usize && !node_included[node] {
                        node_included[node] = true;
                        let prio = (gain_est << 10) | ((hedge_idx as i32) & 0x3FF);
                        let key = (prio << 16) | ((node as i32 & 0xFF) << 8) | (target & 63);
                        valid_moves.push((node, key));
                    }
                }
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
                        if rf_tabu_pairs_active {
                            tabu_upd_host.push(node as u32);
                            tabu_upd_host.push((global_round + tabu_tenure as i32) as u32);
                        }
                    }
                }
                me_he
            } else {
                0i32
            }
        }};
    }


    let cinit_hp = hyperparameters
        .as_ref()
        .and_then(|p| p.get("cinit").and_then(|v| v.as_i64()))
        .unwrap_or(0);


    let ninit: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ninit").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 8) as usize)
        .unwrap_or(if effort == 5 { 8 } else { 1 });
    let probe_rounds: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("probe_rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(10, 50_000) as usize)
        .unwrap_or(400);
    let init_candidates: Vec<(i64, i32)> = {
        let mut v: Vec<(i64, i32)> = vec![(cinit_hp, init_restart_id)];
        for &c in &[(0i64, 0i32), (1, 1), (1, 0), (0, 7), (1, 7), (0, 2), (1, 2), (0, 3), (1, 3)] {
            if v.len() >= ninit { break; }
            if !v.contains(&c) { v.push(c); }
        }
        v
    };


    let nfull: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("nfull").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 8) as usize)
        .unwrap_or(1)
        .min(ninit);
    let n_passes = if ninit > 1 { ninit + nfull } else { 1 };
    let mut probe_results: Vec<(usize, i64)> = Vec::new();
    let mut best_probe: Option<(i64, usize)> = None;
    let mut ranked_candidates: Vec<usize> = Vec::new();
    let mut full_results: Vec<(usize, i64)> = Vec::new();


    let mut full_parts: Vec<Vec<i32>> = Vec::new();
    let mut full_winner_pos: usize = 0;

    let mut best_full: Option<(i64, usize, Vec<i32>, Vec<i32>, Vec<i32>, i32)> = None;
    let full_refinement_rounds = refinement_rounds;

    let mut d_connectivity = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;

    for pass in 0..n_passes {
    let probing = ninit > 1 && pass < ninit;
    let full_idx = pass.saturating_sub(ninit);
    let last_full = pass + 1 == n_passes;
    if ninit > 1 && pass == ninit {

        let mut r: Vec<(i64, usize)> = probe_results.iter().map(|&(i, c)| (c, i)).collect();
        r.sort();
        ranked_candidates = r.into_iter().map(|(_, i)| i).collect();
    }
    let cand_idx = if probing {
        pass
    } else if ninit > 1 {
        ranked_candidates[full_idx.min(ranked_candidates.len().saturating_sub(1))]
    } else {
        0
    };
    let (cinit, init_restart_id) = init_candidates[cand_idx];
    let refinement_rounds = if probing {
        probe_rounds.min(full_refinement_rounds)
    } else {
        full_refinement_rounds
    };
    if pass > 0 {


        node_tabu_until.fill(0);
        stream.memcpy_htod(&node_tabu_until, &mut d_node_tabu_until)?;
        global_round = 0;
        stagnant_rounds = 0;
        host_flags_valid = false;
        tabu_upd_host.clear();
        part_upd_host.clear();
        defer_part_upload = false;
        rf_tabu_pairs_active = false;
    }

    let cluster_cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_hyperedges as u32 + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    if cinit == 1 {
        unsafe {
            stream
                .launch_builder(&hyperedge_cluster_sb_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(num_hedge_clusters as i32))
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&mut d_hyperedge_clusters)
                .launch(cluster_cfg)?;
        }
    } else {
        unsafe {
            stream
                .launch_builder(&hyperedge_cluster_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(num_hedge_clusters as i32))
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&k_hashes)
                .arg(&d_hash_a)
                .arg(&d_hash_b)
                .arg(&mut d_hyperedge_clusters)
                .launch(cluster_cfg)?;
        }
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

    if use_round_fused {


        stream.memcpy_htod(&rf_hdr_zero, &mut d_rf_hdr)?;
        rf_launch_idx = 0;
        tabu_upd_host.clear();
        part_upd_host.clear();
        let tabu_u32: Vec<u32> = node_tabu_until.iter().map(|&v| v.max(0) as u32).collect();
        stream.memcpy_htod(&tabu_u32, &mut d_rf_tabu_until)?;
        defer_part_upload = !use_multi;
        rf_tabu_pairs_active = true;
        rf_epoch = 0;
    }


    let mv_node = |v: u64| (v & 0xFFFF_FFFF) as usize;
    let mv_key = |v: u64| !((v >> 32) as u32) as i32;
    let mut round_idx = 0usize;
    while round_idx < refinement_rounds {

        let round: usize;
        let mut moves_executed: i32;
        let mut k_base = 0usize;
        let mut k_cand = 0usize;
        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        main_valid_moves.clear();

        let mut window_fetched = false;
        macro_rules! fetch_sorted_window {
            ($n:expr) => {{
                let n_: usize = $n;
                main_valid_moves.resize(n_, 0u64);
                if n_ > 0 {
                    stream.memcpy_dtoh(&d_rf_out.slice(0..n_), &mut main_valid_moves[..n_])?;
                }
                window_fetched = true;
            }};
        }


        if use_multi {


            let batch = std::cmp::min(rmulti, refinement_rounds - round_idx);
            let n_tabu_upd = (tabu_upd_host.len() / 2) as i32;
            if n_tabu_upd > 0 {
                let n = tabu_upd_host.len();
                stream.memcpy_htod(&tabu_upd_host[..n], &mut d_rf_tabu_upd.slice_mut(0..n))?;
            }
            let n_part_upd = (part_upd_host.len() / 2) as i32;
            if n_part_upd > 0 {
                let n = part_upd_host.len();
                stream.memcpy_htod(&part_upd_host[..n], &mut d_rf_part_upd.slice_mut(0..n))?;
            }
            let arg_nh: i32 = challenge.num_hyperedges as i32;
            let arg_nn: i32 = challenge.num_nodes as i32;
            let arg_np: i32 = challenge.num_parts as i32;
            let arg_mps: i32 = challenge.max_part_size as i32;
            let arg_round0: i32 = round_idx as i32;
            let arg_ground0: u32 = (global_round + 1) as u32;
            let arg_batch: i32 = batch as i32;
            let arg_ml: i32 = move_limit.min(i32::MAX as usize) as i32;
            let arg_ten: i32 = tabu_tenure as i32;
            let arg_extw: i32 = 16384;
            let arg_gain_only: i32 = 0;
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
                    .arg(&arg_round0)
                    .arg(&arg_ground0)
                    .arg(&arg_batch)
                    .arg(&arg_ml)
                    .arg(&arg_ten)
                    .arg(&arg_extw)
                    .arg(&mut d_rf_tabu_until)
                    .arg(&mut d_rf_stats)
                    .arg(&mut d_rf_hdr)
                    .arg(&mut d_rf_out)
                    .arg(&soft)
                    .arg(&epoch_arg)
                    .arg(&n_tabu_upd)
                    .arg(&d_rf_tabu_upd)
                    .arg(&n_part_upd)
                    .arg(&d_rf_part_upd)
                    .arg(&mut d_rf_acc)
                    .arg(&mut d_rf_acc_off)
                    .arg(&mut d_rf_out2)
                    .arg(&mut d_rf_rs_counts)
                    .arg(&mut d_rf_tcnt)
                    .arg(&mut d_rf_koff)
                    .arg(&arg_gain_only);
                unsafe {
                    if rf_soft {
                        b.launch(rf_cfg.clone())?;
                    } else {
                        b.launch_cooperative(rf_cfg.clone())?;
                    }
                }
            }
            stream.memcpy_dtoh(&d_rf_hdr, &mut rf_hdr_host)?;
            rf_launch_idx += 1;
            let attempted = (rf_hdr_host[31].max(0) as usize).min(batch);
            let status = rf_hdr_host[25];
            let aborted = rf_hdr_host[28] != 0;
            if rf_soft {
                rf_epoch = rf_epoch.wrapping_add(attempted as u32);
                if aborted {


                    rf_soft = false;
                }
            }
            tabu_upd_host.clear();
            part_upd_host.clear();
            global_round += attempted as i32;
            round_idx += attempted;
            if aborted {
                continue;
            }
            if attempted == 0 {
                break;
            }
            if status == 0 {
                stagnant_rounds = 0;
                continue;
            }
            if status == 1 {
                break;
            }


            if attempted > 1 {
                stagnant_rounds = 0;
            }
            round = round_idx - 1;

            refresh_host_mirrors_from_device!();
            let cnt = (rf_hdr_host[3].max(0) as usize).min(challenge.num_nodes as usize);
            let kept = (rf_hdr_host[30].max(0) as usize).min(cnt);
            let adaptive_limit = if round < 50 {
                move_limit / 2
            } else if round < 200 {
                move_limit
            } else {
                move_limit / 3
            };
            k_base = std::cmp::min(cnt, adaptive_limit);
            k_cand = std::cmp::min(cnt, k_base.saturating_add(16384));
            moves_executed = 0;
            if kept == 0 && cnt > 0 {

                fetch_sorted_window!(k_cand);
                let take = std::cmp::min(k_base, k_cand);
                sorted_move_nodes.extend(main_valid_moves[..take].iter().map(|&v| mv_node(v) as i32));
                sorted_move_parts.extend(main_valid_moves[..take].iter().map(|&v| (mv_key(v) & 63) as i32));
                moves_executed = replay_execute_moves_upload!(sorted_move_nodes, sorted_move_parts);
            }
        } else {
        round = round_idx;
        round_idx += 1;
        global_round += 1;
        let mut num_valid_moves = 0i32;
        let mut max_gain = i32::MIN;


        let adaptive_limit = if round < 50 {
            move_limit / 2
        } else if round < 200 {
            move_limit
        } else {
            move_limit / 3
        };
        let slack = if round < 64 { 8usize } else if round < 256 { 4usize } else { 2usize };
        let extra_window = 16384usize;

        if use_round_fused {


            let n_tabu_upd = (tabu_upd_host.len() / 2) as i32;
            if n_tabu_upd > 0 {
                let n = tabu_upd_host.len();
                stream.memcpy_htod(&tabu_upd_host[..n], &mut d_rf_tabu_upd.slice_mut(0..n))?;
            }
            let n_part_upd = (part_upd_host.len() / 2) as i32;
            if n_part_upd > 0 {
                let n = part_upd_host.len();
                stream.memcpy_htod(&part_upd_host[..n], &mut d_rf_part_upd.slice_mut(0..n))?;
            }

            let arg_nh: i32 = challenge.num_hyperedges as i32;
            let arg_nn: i32 = challenge.num_nodes as i32;
            let arg_np: i32 = challenge.num_parts as i32;
            let arg_mps: i32 = challenge.max_part_size as i32;
            let arg_round: u32 = global_round as u32;
            let arg_slack: i32 = slack as i32;
            let arg_alim: i32 = adaptive_limit.min(i32::MAX as usize) as i32;
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
                        .arg(&arg_round)
                        .arg(&mut d_rf_tabu_until)
                        .arg(&mut d_rf_stats)
                        .arg(&mut d_rf_hdr)
                        .arg(&mut d_rf_out)
                        .arg(&soft)
                        .arg(&launch_idx_arg)
                        .arg(&n_tabu_upd)
                        .arg(&d_rf_tabu_upd)
                        .arg(&n_part_upd)
                        .arg(&d_rf_part_upd)
                        .arg(&mut d_rf_acc)
                        .arg(&mut d_rf_acc_off)
                        .arg(&mut d_rf_out2)
                        .arg(&mut d_rf_rs_counts)
                        .arg(&mut d_rf_tcnt)
                        .arg(&mut d_rf_koff)
                        .arg(&arg_slack)
                        .arg(&arg_alim)
                        .arg(&arg_extw);
                    unsafe {
                        if rf_soft {
                            b.launch(rf_cfg.clone())?;
                        } else {
                            b.launch_cooperative(rf_cfg.clone())?;
                        }
                    }
                }
                stream.memcpy_dtoh(&d_rf_hdr, &mut rf_hdr_host)?;
                if rf_soft {
                    rf_launch_idx += 1;


                    if rf_hdr_host[28] != 0 {
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
            num_valid_moves = rf_hdr_host[0].max(0);
            max_gain = rf_hdr_host[2];
        } else {


            if !host_flags_valid {
                rebuild_host_flags!();
            }
            stream.memcpy_htod(&flags_all_host, &mut d_edge_flags_all)?;
            stream.memcpy_htod(&flags_double_host, &mut d_edge_flags_double)?;

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
                    .arg(&moves_lanes)
                    .launch(cfg_moves.clone())?;
            }
            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;


            for &k in move_keys_host.iter() {
                if k > 0 {
                    num_valid_moves += 1;
                    let g = (k >> 16) - 1000;
                    if g > max_gain {
                        max_gain = g;
                    }
                }
            }
        }
        if num_valid_moves == 0 {
            break;
        }

        let aspiration_threshold = std::cmp::max(1, (max_gain * 3) / 4);

        nodes_in_part_host.copy_from_slice(&nodes_in_part_mirror);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }


        if use_round_fused {
            let cnt = (rf_hdr_host[3].max(0) as usize).min(challenge.num_nodes as usize);
            if cnt == 0 {
                break;
            }
            k_base = std::cmp::min(cnt, adaptive_limit);
            k_cand = std::cmp::min(cnt, k_base.saturating_add(extra_window));
            let kept = (rf_hdr_host[30].max(0) as usize).min(cnt);
            let take = std::cmp::min(kept, k_base);
            if take > 0 {
                kept_host.resize(take, 0u64);
                stream.memcpy_dtoh(&d_rf_out2.slice(0..take), &mut kept_host[..take])?;
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


        let fast_path = !use_round_fused && (num_valid_moves as usize) <= adaptive_limit;
        if fast_path {
            for b in tgt_buckets.iter_mut() {
                b.clear();
            }
            let mut filtered = 0usize;
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key > 0 {
                    let gain = (key >> 16) - 1000;
                    if node_tabu_until[node] <= global_round || gain >= aspiration_threshold {
                        filtered += 1;
                        let tgt = (key & 63) as usize;
                        if tgt < num_parts_usize {
                            tgt_buckets[tgt].push(((!(key as u32) as u64) << 32) | node as u64);
                        }
                    }
                }
            }
            if filtered == 0 {
                break;
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


        if sorted_move_nodes.is_empty() && !use_round_fused {
            main_valid_moves.clear();
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key > 0 {
                    let gain = (key >> 16) - 1000;
                    if node_tabu_until[node] <= global_round || gain >= aspiration_threshold {
                        main_valid_moves.push(((!(key as u32) as u64) << 32) | node as u64);
                    }
                }
            }

            if main_valid_moves.is_empty() {
                break;
            }

            k_base = main_valid_moves.len();
            if k_base > adaptive_limit {
                k_base = adaptive_limit;
            }

            k_cand = std::cmp::min(main_valid_moves.len(), k_base.saturating_add(extra_window));

            if k_cand < main_valid_moves.len() {
                main_valid_moves.select_nth_unstable(k_cand - 1);
            }
            main_valid_moves[..k_cand].sort_unstable();
            window_fetched = true;

            tgt_used.fill(0);
            for &v in main_valid_moves[..k_cand].iter() {
                if sorted_move_nodes.len() >= k_base {
                    break;
                }
                let tgt = (mv_key(v) & 63) as usize;
                if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                    tgt_used[tgt] += 1;
                    sorted_move_nodes.push(mv_node(v) as i32);
                    sorted_move_parts.push(tgt as i32);
                }
            }

            if sorted_move_nodes.is_empty() {
                let take = std::cmp::min(k_base, k_cand);
                sorted_move_nodes.extend(main_valid_moves[..take].iter().map(|&v| mv_node(v) as i32));
                sorted_move_parts.extend(main_valid_moves[..take].iter().map(|&v| (mv_key(v) & 63) as i32));
            }
        }

        moves_executed = replay_execute_moves_upload!(sorted_move_nodes, sorted_move_parts);
        }

        if moves_executed == 0 && k_cand > k_base {
            sorted_move_nodes.clear();
            sorted_move_parts.clear();


            if !window_fetched {
                fetch_sorted_window!(k_cand);
            }

            let _ = window_fetched;
            let tail = &main_valid_moves[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            sorted_move_nodes.extend(tail.iter().take(take).map(|&v| mv_node(v) as i32));
            sorted_move_parts.extend(tail.iter().take(take).map(|&v| (mv_key(v) & 63) as i32));

            moves_executed = replay_execute_moves_upload!(sorted_move_nodes, sorted_move_parts);
        }

        if moves_executed > 0 {
            let until = global_round + tabu_tenure as i32;
            for &node in sorted_move_nodes.iter().take(moves_executed as usize) {
                node_tabu_until[node as usize] = until;
                if use_round_fused {
                    tabu_upd_host.push(node as u32);
                    tabu_upd_host.push(until as u32);
                }
            }
        }

        if moves_executed == 0 {

            flush_part_upd!();
            let he_moves = do_hyperedge_centric_phase!();
            if he_moves > 0 {
                stagnant_rounds = 0;
                continue;
            }

            stagnant_rounds += 1;
            if stagnant_rounds >= 5 && round < refinement_rounds.saturating_sub(50) {
                let mini_seed = 987654321u64 + (round as u64) * 123456789u64;
                flush_part_upd!();
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


    flush_part_upd!();
    defer_part_upload = false;
    rf_tabu_pairs_active = false;


    host_flags_valid = false;
    if use_multi {


        refresh_host_mirrors_from_device!();
        let tabu_dev: Vec<u32> = stream.memcpy_dtov(&d_rf_tabu_until)?;
        for (i, &v) in tabu_dev.iter().enumerate() {
            node_tabu_until[i] = v.min(i32::MAX as u32) as i32;
        }
        for pair in tabu_upd_host.chunks_exact(2) {
            node_tabu_until[pair[0] as usize] = pair[1].min(i32::MAX as u32) as i32;
        }
    }
    if probing {


        if use_round_fused && !use_multi {
            flush_part_upd!();
        }
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
        let conn: i64 = v.iter().map(|&x| x as i64).sum();
        probe_results.push((pass, conn));

        if best_probe.map(|(c, _)| conn < c).unwrap_or(true) {
            best_probe = Some((conn, pass));
        }
        continue;
    }
    if ninit > 1 && nfull > 1 {

        if use_round_fused && !use_multi {
            flush_part_upd!();
        }
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
        let conn: i64 = v.iter().map(|&x| x as i64).sum();
        let part = stream.memcpy_dtov(&d_partition)?;
        full_results.push((cand_idx, conn));
        full_parts.push(part.clone());
        if best_full.as_ref().map(|b| conn < b.0).unwrap_or(true) {
            let nip = stream.memcpy_dtov(&d_nodes_in_part)?;
            best_full = Some((conn, cand_idx, part, nip, node_tabu_until.clone(), global_round));
            full_winner_pos = full_results.len() - 1;
        }
        if !last_full {
            continue;
        }

        if let Some((bconn, bidx, part, nip, tabu, ground)) = best_full.take() {
            if bidx != cand_idx || bconn < conn {
                stream.memcpy_htod(&part, &mut d_partition)?;
                stream.memcpy_htod(&nip, &mut d_nodes_in_part)?;
                partition_host_mirror.copy_from_slice(&part);
                nodes_in_part_mirror.copy_from_slice(&nip);
                node_tabu_until.copy_from_slice(&tabu);
                global_round = ground;
                host_flags_valid = false;
            }
        }
    }
    }

    let crossover_kernel = module.load_function("crossover_partitions_10k")?;

    macro_rules! do_swap_phase {
        ($d_partition:expr, $d_nodes_in_part:expr,
         $d_edge_flags_all:expr, $d_edge_flags_double:expr,
         $d_swap_gains:expr,
         $partition_host_swap:expr, $partition_mut_swap:expr,
         $d_swap_topk:expr, $swap_topk_host:expr,
         $max_rounds:expr, $ngt:expr, $scan_lim_cyc:expr) => {{
            let num_nodes_i = challenge.num_nodes as i32;
            let num_parts_i = challenge.num_parts as i32;
            let np = num_parts_usize;
            let mut prev_swap_count = usize::MAX;
            let mut stagnant = 0usize;
            let mut total_swaps = 0usize;
            let mut top_nodes: [[(usize, i32); 32]; 64*64] = [[(0,0);32]; 64*64];
            let mut top_counts: [usize; 64*64] = [0; 64*64];
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
                unsafe {
                    stream
                        .launch_builder(&compute_swap_topk_kernel)
                        .arg(&num_nodes_i)
                        .arg(&num_parts_i)
                        .arg(&mut *$d_partition)
                        .arg(&mut *$d_swap_gains)
                        .arg(&mut *$d_swap_topk)
                        .launch(LaunchConfig {
                            grid_dim: ((np * np) as u32, 1, 1),
                            block_dim: (block_size.min(128), 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }
                stream.memcpy_dtoh(&*$d_swap_topk, $swap_topk_host)?;

                for i in 0..np*np {
                    top_counts[i] = 0;
                }
                for a in 0..np {
                    for b in 0..np {
                        if a == b { continue; }
                        let idx = a * np + b;
                        let base = idx * 32 * 2;
                        let mut cnt = 0usize;
                        for k in 0..32usize {
                            let node = $swap_topk_host[base + k * 2];
                            let gain = $swap_topk_host[base + k * 2 + 1];
                            if node >= 0 && gain > 0 {
                                top_nodes[idx][cnt] = (node as usize, gain);
                                cnt += 1;
                            } else {
                                break;
                            }
                        }
                        top_counts[idx] = cnt;
                    }
                }

                $partition_mut_swap.copy_from_slice($partition_host_swap);
                let mut swap_count = 0usize;

                {
                    let nn = challenge.num_nodes as usize;
                    let bp_size = np * np * 3;
                    let mut d_bp = stream.alloc_zeros::<i32>(bp_size)?;
                    unsafe {
                        stream
                            .launch_builder(&compute_best_swap_pairs_kernel)
                            .arg(&(challenge.num_nodes as i32))
                            .arg(&(challenge.num_parts as i32))
                            .arg(&*$d_partition)
                            .arg(&*$d_swap_topk)
                            .arg(&mut d_bp)
                            .launch(LaunchConfig {
                                grid_dim: (np as u32, 1, 1),
                                block_dim: (np as u32, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                    let bp_host = stream.memcpy_dtov(&d_bp)?;
                    for a in 0..np {
                        for b in 0..np {
                            if a == b { continue; }
                            let base = (a * np + b) * 3;
                            let node_a = bp_host[base] as usize;
                            let node_b = bp_host[base + 1] as usize;
                            let gain = bp_host[base + 2];
                            if node_a < nn && node_b < nn && gain > 0
                                && $partition_mut_swap[node_a] as usize == a
                                && $partition_mut_swap[node_b] as usize == b
                                && node_a != node_b
                            {
                                $partition_mut_swap[node_a] = b as i32;
                                $partition_mut_swap[node_b] = a as i32;
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
                            let cnt_ab = top_counts[idx_ab];
                            if cnt_ab == 0 { continue; }
                            for c in 0..np {
                                if c == a || c == b { continue; }
                                let idx_bc = b * np + c;
                                let idx_ca = c * np + a;
                                let cnt_bc = top_counts[idx_bc];
                                let cnt_ca = top_counts[idx_ca];
                                if cnt_bc == 0 || cnt_ca == 0 { continue; }
                                let sl_ab = std::cmp::min(cnt_ab, cyc_scan);
                                let sl_bc = std::cmp::min(cnt_bc, cyc_scan);
                                let sl_ca = std::cmp::min(cnt_ca, cyc_scan);
                                'outer: for i in 0..sl_ab {
                                    let (node_ab, gain_ab) = top_nodes[idx_ab][i];
                                    if $partition_mut_swap[node_ab] as usize != a { continue; }
                                    if gain_ab + top_nodes[idx_bc][0].1 + top_nodes[idx_ca][0].1 <= 0 { break; }
                                    for j in 0..sl_bc {
                                        let (node_bc, gain_bc) = top_nodes[idx_bc][j];
                                        if $partition_mut_swap[node_bc] as usize != b { continue; }
                                        if node_bc == node_ab { continue; }
                                        if gain_ab + gain_bc + top_nodes[idx_ca][0].1 <= 0 { break; }
                                        for k in 0..sl_ca {
                                            let (node_ca, gain_ca) = top_nodes[idx_ca][k];
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
                            let cnt_ab = top_counts[idx_ab];
                            if cnt_ab == 0 { continue; }
                            for c in 0..np {
                                if c == a || c == b { continue; }
                                let idx_bc = b * np + c;
                                let cnt_bc = top_counts[idx_bc];
                                if cnt_bc == 0 { continue; }
                                for d in 0..np {
                                    if d == a || d == b || d == c { continue; }
                                    let idx_cd = c * np + d;
                                    let idx_da = d * np + a;
                                    let cnt_cd = top_counts[idx_cd];
                                    let cnt_da = top_counts[idx_da];
                                    if cnt_cd == 0 || cnt_da == 0 { continue; }
                                    let sl_ab = std::cmp::min(cnt_ab, cyc4_scan);
                                    let sl_bc = std::cmp::min(cnt_bc, cyc4_scan);
                                    let sl_cd = std::cmp::min(cnt_cd, cyc4_scan);
                                    let sl_da = std::cmp::min(cnt_da, cyc4_scan);

                                    'outer4: for i in 0..sl_ab {
                                        let (node_ab, gain_ab) = top_nodes[idx_ab][i];
                                        if $partition_mut_swap[node_ab] as usize != a { continue; }
                                        if gain_ab + top_nodes[idx_bc][0].1 + top_nodes[idx_cd][0].1 + top_nodes[idx_da][0].1 <= 0 { break; }
                                        for j in 0..sl_bc {
                                            let (node_bc, gain_bc) = top_nodes[idx_bc][j];
                                            if $partition_mut_swap[node_bc] as usize != b { continue; }
                                            if node_bc == node_ab { continue; }
                                            if gain_ab + gain_bc + top_nodes[idx_cd][0].1 + top_nodes[idx_da][0].1 <= 0 { break; }
                                            for k in 0..sl_cd {
                                                let (node_cd, gain_cd) = top_nodes[idx_cd][k];
                                                if $partition_mut_swap[node_cd] as usize != c { continue; }
                                                if node_cd == node_ab || node_cd == node_bc { continue; }
                                                if gain_ab + gain_bc + gain_cd + top_nodes[idx_da][0].1 <= 0 { break; }
                                                for m in 0..sl_da {
                                                    let (node_da, gain_da) = top_nodes[idx_da][m];
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
        &mut d_swap_gains,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut d_swap_topk, &mut swap_topk_host,
        100, neg_gain_thresh, scan_limit_cycle
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
                .arg(&moves_lanes)
                .launch(cfg_moves.clone())?;
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
        &mut d_swap_gains,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut d_swap_topk, &mut swap_topk_host,
        50, neg_gain_thresh, scan_limit_cycle
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

    let mut pop_protected: Vec<bool> = vec![false];

    let elite_pool_size = std::cmp::max(2usize, ils_iterations);
    let mut elite_scores: Vec<i32> = vec![i32::MAX; elite_pool_size];
    let mut elite_flat_host: Vec<i32> =
        vec![0i32; elite_pool_size * challenge.num_nodes as usize];
    elite_scores[0] = best_connectivity;
    elite_flat_host[..challenge.num_nodes as usize].copy_from_slice(&best_partition_host);
    let mut elite_count: usize = 1;


    let xseed = hyperparameters
        .as_ref()
        .and_then(|p| p.get("xseed").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        != 0;
    let mut n_xseeds = 0usize;
    if xseed && full_parts.len() > 1 {
        let n = challenge.num_nodes as usize;
        for (k, part) in full_parts.iter().enumerate() {
            if k == full_winner_pos {
                continue;
            }
            let aligned = align_part_labels(part, &best_partition_host, num_parts_usize);
            let conn = full_results[k].1.min(i32::MAX as i64) as i32;
            pop_partitions.push(aligned.clone());
            pop_connectivities.push(conn);
            pop_protected.push(true);
            if elite_count < elite_pool_size {
                elite_flat_host[elite_count * n..(elite_count + 1) * n].copy_from_slice(&aligned);
                elite_scores[elite_count] = conn;
                elite_count += 1;
            }
            n_xseeds += 1;
        }
    }
    let mut d_elite_flat =
        stream.alloc_zeros::<i32>(elite_pool_size * challenge.num_nodes as usize)?;
    let mut use_consensus_next = false;
    let mut ils_stagnation = 0usize;
    let ils_stagnation_limit = std::cmp::max(3, ils_iterations / 4);

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

            let _ = rebalance_parts!();
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


        macro_rules! fused_quick_refine {
            ($n_rounds:expr) => {{
                let n_rounds: usize = $n_rounds;
                stream.memcpy_htod(&rf_hdr_zero, &mut d_rf_hdr)?;
                let tabu_u32: Vec<u32> = node_tabu_until.iter().map(|&v| v.max(0) as u32).collect();
                stream.memcpy_htod(&tabu_u32, &mut d_rf_tabu_until)?;
                let mut qr_epoch: u32 = 0;
                let mut done = 0usize;
                let arg_nh: i32 = challenge.num_hyperedges as i32;
                let arg_nn: i32 = challenge.num_nodes as i32;
                let arg_np: i32 = challenge.num_parts as i32;
                let arg_mps: i32 = challenge.max_part_size as i32;
                let arg_round0: i32 = 64;
                let arg_ml: i32 = move_limit.min(i32::MAX as usize) as i32;
                let arg_ten: i32 = tabu_tenure as i32;
                let arg_extw: i32 = 16384;
                let arg_gain_only: i32 = 1;
                let zero_upd: i32 = 0;
                while done < n_rounds {
                    let batch = std::cmp::min(rmulti.min(136), n_rounds - done);
                    let arg_ground0: u32 = (global_round + 1) as u32;
                    let arg_batch: i32 = batch as i32;
                    let soft: i32 = if rf_soft { 1 } else { 0 };
                    let epoch_arg: u32 = qr_epoch;
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
                            .arg(&arg_round0)
                            .arg(&arg_ground0)
                            .arg(&arg_batch)
                            .arg(&arg_ml)
                            .arg(&arg_ten)
                            .arg(&arg_extw)
                            .arg(&mut d_rf_tabu_until)
                            .arg(&mut d_rf_stats)
                            .arg(&mut d_rf_hdr)
                            .arg(&mut d_rf_out)
                            .arg(&soft)
                            .arg(&epoch_arg)
                            .arg(&zero_upd)
                            .arg(&d_rf_tabu_upd)
                            .arg(&zero_upd)
                            .arg(&d_rf_part_upd)
                            .arg(&mut d_rf_acc)
                            .arg(&mut d_rf_acc_off)
                            .arg(&mut d_rf_out2)
                            .arg(&mut d_rf_rs_counts)
                            .arg(&mut d_rf_tcnt)
                            .arg(&mut d_rf_koff)
                            .arg(&arg_gain_only);
                        unsafe {
                            if rf_soft {
                                b.launch(rf_cfg.clone())?;
                            } else {
                                b.launch_cooperative(rf_cfg.clone())?;
                            }
                        }
                    }
                    stream.memcpy_dtoh(&d_rf_hdr, &mut rf_hdr_host)?;
                    let attempted = (rf_hdr_host[31].max(0) as usize).min(batch);
                    let status = rf_hdr_host[25];
                    let aborted = rf_hdr_host[28] != 0;
                    if rf_soft {
                        qr_epoch = qr_epoch.wrapping_add(attempted as u32);
                        if aborted {


                            rf_soft = false;
                        }
                    }
                    global_round += attempted as i32;
                    done += attempted;
                    if aborted {
                        continue;
                    }


                    if attempted == 0 || status != 0 {
                        break;
                    }
                }
                refresh_host_mirrors_from_device!();
                let tabu_dev: Vec<u32> = stream.memcpy_dtov(&d_rf_tabu_until)?;
                for (h, &d) in node_tabu_until.iter_mut().zip(tabu_dev.iter()) {
                    *h = d as i32;
                }
                done
            }};
        }

        if use_ils_fused {
            let _ = fused_quick_refine!(ils_quick_refine);
        } else {
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
                    .arg(&moves_lanes)
                    .launch(cfg_moves.clone())?;
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
        }

        do_swap_phase!(
            &mut d_partition, &mut d_nodes_in_part,
            &mut d_edge_flags_all, &mut d_edge_flags_double,
            &mut d_swap_gains,
            &mut partition_host_swap, &mut partition_mut_swap,
            &mut d_swap_topk, &mut swap_topk_host,
            25, neg_gain_thresh, scan_limit_cycle
        )?;

        {
            unsafe {
                stream
                    .launch_builder(&compute_part_cut_cost_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&d_edge_flags_all)
                    .arg(&mut d_part_cut_costs)
                    .arg(&mut d_bottleneck_part)
                    .launch(one_thread_cfg.clone())?;
            }
            let bp_val = stream.memcpy_dtov(&d_bottleneck_part)?[0];
            if bp_val >= 0 && (bp_val as usize) < num_parts_usize {
                let bp = bp_val as usize;
                unsafe {
                    stream
                        .launch_builder(&repair_bottleneck_part_kernel)
                        .arg(&(challenge.num_nodes as i32))
                        .arg(&(challenge.num_parts as i32))
                        .arg(&(challenge.max_part_size as i32))
                        .arg(&(bp as i32))
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&d_partition)
                        .arg(&d_nodes_in_part)
                        .arg(&d_edge_flags_all)
                        .arg(&d_edge_flags_double)
                        .arg(&mut d_repair_gains)
                        .arg(&mut d_repair_targets)
                        .launch(cfg.clone())?;
                }
                let gain_host = stream.memcpy_dtov(&d_repair_gains)?;
                let tgt_host = stream.memcpy_dtov(&d_repair_targets)?;
                refresh_host_mirrors_from_device!();
                let limit = 50usize;
                valid_moves.clear();
                for node in 0..challenge.num_nodes as usize {
                    if partition_host_mirror[node] == bp as i32 && gain_host[node] > 0 {
                        let tgt = tgt_host[node] as usize;
                        if tgt < num_parts_usize {
                            valid_moves.push((node, (gain_host[node] << 16) | (tgt as i32)));
                        }
                    }
                }
                if !valid_moves.is_empty() {
                    valid_moves.sort_unstable_by(|a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    sorted_move_nodes.clear();
                    sorted_move_parts.clear();
                    for &(node, key) in valid_moves.iter().take(limit) {
                        let tgt = (key & 63) as usize;
                        if partition_host_mirror[node] as usize == bp && nodes_in_part_mirror[tgt] < challenge.max_part_size as i32 {
                            partition_host_mirror[node] = tgt as i32;
                            nodes_in_part_mirror[bp] -= 1;
                            nodes_in_part_mirror[tgt] += 1;
                            sorted_move_nodes.push(node as i32);
                            sorted_move_parts.push(tgt as i32);
                        }
                    }
                    if !sorted_move_nodes.is_empty() {
                        let accepted_len = sorted_move_nodes.len();
                        stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
                        stream.memcpy_htod(sorted_move_nodes.as_slice(), &mut d_accepted_move_nodes)?;
                        stream.memcpy_htod(sorted_move_parts.as_slice(), &mut d_accepted_move_parts)?;
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
                                    grid_dim: ((accepted_len as u32 + block_size - 1) / block_size, 1, 1),
                                    block_dim: (block_size, 1, 1),
                                    shared_mem_bytes: 0,
                                })?;
                        }
                    }
                }
            }
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

            let mut worst_idx: Option<usize> = None;
            let mut worst_conn = i32::MIN;
            for (pi, &pc) in pop_connectivities.iter().enumerate() {
                if !pop_protected[pi] && (worst_idx.is_none() || pc > worst_conn) {
                    worst_conn = pc;
                    worst_idx = Some(pi);
                }
            }
            if pop_partitions.len() < 3 + n_xseeds {
                pop_partitions.push(iter_partition);
                pop_connectivities.push(new_connectivity);
                pop_protected.push(false);
            } else if let Some(wi) = worst_idx {
                if new_connectivity < worst_conn {
                    pop_partitions[wi] = iter_partition;
                    pop_connectivities[wi] = new_connectivity;
                }
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

        if improved {
            ils_stagnation = 0;
        } else {
            ils_stagnation += 1;
            if ils_stagnation >= ils_stagnation_limit {
                break;
            }
        }
    }
    if pop_partitions.len() >= 2 {
        let mut pop_order: Vec<usize> = (0..pop_partitions.len()).collect();
        pop_order.sort_unstable_by_key(|&i| pop_connectivities[i]);


        for cross_idx in 0..std::cmp::min(pop_order.len().saturating_sub(1), 2 + n_xseeds) {
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
                        .arg(&moves_lanes)
                        .launch(cfg_moves.clone())?;
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
                &mut d_swap_gains,
                &mut partition_host_swap, &mut partition_mut_swap,
                &mut d_swap_topk, &mut swap_topk_host,
                15, neg_gain_thresh, scan_limit_cycle
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
                .arg(&moves_lanes)
                .launch(cfg_moves.clone())?;
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
        &mut d_swap_gains,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut d_swap_topk, &mut swap_topk_host,
        10, neg_gain_thresh, scan_limit_cycle
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

    let _ = rebalance_parts!();
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
                .arg(&moves_lanes)
                .launch(cfg_moves.clone())?;
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

    let use_pexp = hyperparameters
        .as_ref()
        .and_then(|p| p.get("pexp").and_then(|v| v.as_i64()))
        .map(|v| v != 0)
        .unwrap_or(true);
    if use_pexp {
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
        let polish_seed = 0x123456789ABCDEF0u64 ^ (global_round as u64);
        unsafe {
            stream
                .launch_builder(&polish_exploration_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .arg(&d_edge_flags_all)
                .arg(&d_edge_flags_double)
                .arg(&mut d_polish_backup_partition)
                .arg(&mut d_polish_backup_nip)
                .arg(&mut d_polish_best_partition)
                .arg(&mut d_polish_best_nip)
                .arg(&polish_seed)
                .launch(one_thread_cfg.clone())?;
        }
    }

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
    let href = hp_i64("href", 1) != 0;

    let he_off_i32 = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let he_nodes_i32 = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;

    if href {


        let fuel_reserve: u64 = hp_i64("hfuel_reserve", 5_000_000_000).max(0) as u64;
        let fuel_at_start = super::fuel_remaining();
        let budget_ok = move || -> bool {
            let f = super::fuel_remaining();
            fuel_at_start == 0 || f > fuel_reserve
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
            flow: hp_i64("hflow", 1) != 0,
            flow_rounds: hp_i64("hflow_rounds", 3).clamp(1, 100) as usize,
            flow_params: super::hflow::FlowParams {
                alpha: hp_i64("hflow_alpha", 4).clamp(1, 64) as u32,
                max_region: hp_i64("hflow_region", 96).clamp(1, 100_000) as usize,
                edge_cap: hp_i64("hflow_ecap", 48).clamp(2, 100_000) as usize,
                min_shared: hp_i64("hflow_min", 3).clamp(1, 1_000_000) as usize,
            },
            flow_in_cycles: hp_i64("hflow_cycles", 0) != 0,
            vcycles: hp_i64("hvcycles", 5).clamp(0, 1000) as usize,
            ml_levels: hp_i64("hml_levels", 3).clamp(1, 20) as usize,
            ml_max_cluster: hp_i64("hml_cluster", 0).clamp(0, 100_000) as u32,
            ml_patience: hp_i64("hml_patience", 2).clamp(1, 1000) as usize,
            ml_fine_full: hp_i64("hml_fine_full", 0) != 0,
            ml_coarse_full: hp_i64("hml_coarse_full", 1) != 0,
            jet_rounds: hp_i64("hjet", if effort == 5 { 24 } else { 0 }).clamp(0, 10_000) as usize,
            jet_tolerance: hp_i64("hjet_tol", 6).clamp(1, 10_000) as usize,
            jet_neg_pct: hp_i64("hjet_c", if effort == 5 { 50 } else { 25 }).clamp(0, 100) as u32,
            jet_min_gain: hp_i64("hjet_min", 0).clamp(-1_000_000, 1_000_000) as i32,
            jet_stages: hp_i64("hjet_stage", 3).clamp(0, 3) as u32,
            jet_slack: hp_i64("hjet_slack", if effort == 5 { 0 } else { -1 }).clamp(-1, 1_000_000) as i32,
            seed: 0x5EED_5EED_1234_ABCDu64
                ^ u64::from_le_bytes([
                    challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3],
                    challenge.seed[4], challenge.seed[5], challenge.seed[6], challenge.seed[7],
                ]),
        };
        let he_off: Vec<u32> = he_off_i32.iter().map(|&x| x as u32).collect();
        let he_nodes: Vec<u32> = he_nodes_i32.iter().map(|&x| x as u32).collect();
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
    Ok(())
}
