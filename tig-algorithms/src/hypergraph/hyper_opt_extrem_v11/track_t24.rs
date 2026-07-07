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

    let lb_min = hyperparameters
        .as_ref()
        .and_then(|p| p.get("launch_bounds_min_blocks").and_then(|v| v.as_i64()))
        .unwrap_or(0);
    let compute_moves_name = match lb_min {
        2 => "compute_refinement_moves_optimized_200k_lb2",
        3 => "compute_refinement_moves_optimized_200k_lb3",
        4 => "compute_refinement_moves_optimized_200k_lb4",
        _ => "compute_refinement_moves_optimized_200k",
    };
    let compute_moves_kernel = module.load_function(compute_moves_name)?;
    let balance_kernel = module.load_function("balance_final_200k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_200k")?;
    let reduce_connectivity_sum_kernel = module.load_function("reduce_connectivity_sum_200k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_200k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_200k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_200k")?;
    let score_pr_g3_warp_kernel = module.load_function("score_pr_g3_warp_200k")?;

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
        5 => (9000, 6, 70, 300, 0),
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
        .unwrap_or(14);

    let aspiration_frac: f64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("aspiration_frac").and_then(|v| v.as_f64()))
        .unwrap_or(0.75)
        .clamp(0.1, 1.0);

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

    
    // hyperedges processed by the hyperedge-centric atomic GROUP MOVE inserted after the pre-loop
    // swap phase. 0 ≡ baseline verbatim (the whole block is gated off ⇒ byte-identical to i1).
    // Sweep {64, 128}.
    let collapse_top_k: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("collapse_top_k").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 4096) as usize)
        .unwrap_or(0);

    
    // group move commits the best PREFIX of minority pins (sorted by marginal Δλ, accumulated
    
    // i18's all-or-nothing (which harvested as a no-op: moving ALL pins inflates λ of the OTHER
    // nets of those pins ⇒ Δtotal≥0). 0 ⇒ i18 all-or-nothing path (collapse_top_k=0 ⇒ baseline).
    let collapse_subseq: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("collapse_subseq").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 1) as usize)
        .unwrap_or(0);

    
    // = number of single-level global FM passes inserted after the pre-loop swap phase. Each pass
    // collects EVERY boundary vertex's best single-pin move (target minimising isolated Δλ, NOT
    // forced to one net's plurality block as in i18/i19), sorts them by gain, recomputes Δλ
    // IN-SEQUENCE, and commits the PREFIX with minimum cumulative Δλ (revert-to-best-prefix). 0 ≡
    // baseline verbatim (block gated off ⇒ byte-identical to i1). Sweep {1, 2}.
    let global_prefix: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("global_prefix").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 8) as usize)
        .unwrap_or(0);

    
    // single global best-balanced-prefix pass. i20 admits only moves with isolated Δλ ≤ 0 (L1332),
    // which exhausts the pool too fast (the zero-gain trap). ab_thresh relaxes admission to
    // dl_iso ≤ ab_thresh, letting moves that are WORSE-in-isolation but promoted to Δλ<0 by the
    
    // COMMIT stays STRICTLY best_cum < 0 with full queue revert (L1372) ⇒ validity by construction
    // (only a Δtotal<0 prefix is ever applied; capacity guards unchanged). ab_thresh=0 ≡ i20 EXACT.
    // Sweep {0,1,2,3}. NB: NEVER commits a ≥0 move (≠ dead t22 vrai-Jet i25 −793k invalid).
    let ab_thresh: i32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ab_thresh").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as i32)
        .unwrap_or(0);

    // i22 (gp_topk top-k targets + gain-density ordering, roadmap 232130Z): i20/i21 collect a SINGLE
    // best target per node (the part maximising pres[t]) and sort the global sequence by ABSOLUTE Δλ.
    
    // lever on the length of the Δcum<0 prefix (insight 27364d9c authorises more destinations/node).
    // gp_topk ≥ 2 collects the k lowest-Δλ targets per node (more destinations) AND orders the walk by
    // gain-density (gain = -Δλ_iso, weight = node degree) descending instead of absolute Δλ. The
    // in-sequence walk applies each node AT MOST ONCE (first feasible target reached wins); if a node's
    // densest target is capacity-blocked, a lower-density alternative can still fire later. COMMIT stays
    // STRICTLY best_cum < 0 + full queue revert, capacity by running per-state nip_cur ⇒ validity by
    
    
    let gp_topk: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gp_topk").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 8) as usize)
        .unwrap_or(1);

    
    // 0 (default) = skip entirely = bit-identical to i22 (OFF flag, exact Q=262470 control).
    
    // gain-cache), dd13c18f (flow beats FM especially on LARGE hyperedges = t24 regime).
    let fl_n_pairs: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fl_n_pairs").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as usize)
        .unwrap_or(0);

    
    // gain-density AND same absolute gain, prefer moving to the LESS LOADED target part
    // (lower nip_cur[t] = more slack for subsequent moves within the same pass).
    // Hypothesis: more balanced load after the prefix creates more feasible moves later
    // in the walk (capacity guard nip_cur[t]+1 ≤ max_ps is less likely to block good moves).
    // gp_target_load_tb=false (default) ≡ i31 bit-identical (OFF gate).
    
    let gp_target_load_tb: bool = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gp_target_load_tb").and_then(|v| v.as_bool()))
        .unwrap_or(false);

    
    // After the final swap phase, run up to seq_sweeps CPU-side greedy FM passes using exact
    // sequential gain recomputation via per-(hedge,part) pin-count cache. GPU kernels
    // (precompute_edge_flags_200k + compute_refinement_moves_optimized_200k) recompute
    // priorities each sweep; host applies moves in gain-desc order (sees earlier moves).
    // Monotone guard: revert if total km1 connectivity does not improve.
    // seq_sweeps=0 (default) ≡ i22/i30 bit-identical OFF.
    
    
    // best-subsequence method accumulates gains — academic ref), 8dacefe9 (same).
    let seq_sweeps: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("seq_sweeps").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 32) as usize)
        .unwrap_or(0);

    
    
    
    let term_refine_mode: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("term_refine_mode").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 2) as usize)
        .unwrap_or(0);
    
    
    // 1=LP-diversity: partition_B via label-propagation-lite (CPU, 1 pass, node-degree DESC order,
    //    c03b15f6 high-affinity first) + cheap FM 2 rounds before merge-consensus.
    
    
    //    merge: agree=keep, disagree=pref_parts greedy re-assign DESC priority order.
    //    ⛔ i90 re-seed NULL: same algo + different seed = too correlated (merge no-op, predicted).
    let init_ensemble_mode: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("init_ensemble_mode").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 1) as usize)
        .unwrap_or(0);
    
    
    let max_sp_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("max_sp_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 8) as usize)
        .unwrap_or(2);

    
    // Terminal compound-move chains: trigger → best part → best node in dest → …
    // In-sequence gain recompute (hpc updated after each move). Best-prefix subseq applied.
    
    //      7db1d2ef (t24=200k: 1-node FM moves often gain-zero → chains compose gains).
    let ejchain_max_len: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ejchain_max_len").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 8) as usize)
        .unwrap_or(0);

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
    // i18 (net-collapse): host-side node→hyperedge incidence (exact epc Δconnectivity machinery,
    // ported from t24/i16 epc plumbing 5d21b2c3; only read when collapse_top_k>0 below).
    let node_offsets_host = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let node_hedges_host = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
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

    let cluster_seed_a: u32 = 0u32;
    unsafe {
        stream
            .launch_builder(&hyperedge_cluster_kernel)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&mut d_hyperedge_clusters)
            .arg(&cluster_seed_a)
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

    
    // ⛔ i90 re-seed NULL: same algo + different seed = too correlated (merge no-op).
    
    // + cheap FM on partition_B BEFORE merge (40990c33: improve candidate before consensus).
    if init_ensemble_mode == 1 {
        let partition_a = partition_host_refine.clone();
        let n_nodes = challenge.num_nodes as usize;
        let max_per_lp = challenge.max_part_size as i32;

        // Step 1: Label-propagation-lite — 1 CPU pass, node-degree DESC order (c03b15f6 high-affinity first).
        // For each node: count part-votes from already-assigned hyperedge-neighbors (hedge-size weights),
        // pick max-vote balance-constrained part. Fallback to pref_parts[node] when no neighbors assigned.
        let node_degrees: Vec<i32> = (0..n_nodes)
            .map(|nd| node_offsets_host[nd + 1] - node_offsets_host[nd])
            .collect();
        let mut node_order_lp: Vec<usize> = (0..n_nodes).collect();
        node_order_lp.sort_unstable_by(|&a, &b| {
            node_degrees[b].cmp(&node_degrees[a]).then_with(|| a.cmp(&b))
        });

        let mut partition_b: Vec<i32> = vec![-1i32; n_nodes];
        let mut counts_b: Vec<i32> = vec![0i32; num_parts_usize];
        let mut part_votes: Vec<i32> = vec![0i32; num_parts_usize];

        for &nd in node_order_lp.iter() {
            for pv in part_votes.iter_mut() { *pv = 0; }
            let nhs = node_offsets_host[nd] as usize;
            let nhe = node_offsets_host[nd + 1] as usize;
            let mut has_votes = false;
            for j in nhs..nhe {
                let hedge = node_hedges_host[j] as usize;
                if hedge + 1 >= hedge_offsets_host.len() { continue; }
                let hs = hedge_offsets_host[hedge] as usize;
                let he = hedge_offsets_host[hedge + 1] as usize;
                let hsz = he.saturating_sub(hs) as i32;
                let wt: i32 = if hsz <= 2 { 6 } else if hsz <= 4 { 4 } else if hsz <= 8 { 2 } else { 1 };
                for k in hs..he {
                    if k >= hyperedge_nodes_host.len() { continue; }
                    let nb = hyperedge_nodes_host[k] as usize;
                    if nb < n_nodes && nb != nd {
                        let nb_part = partition_b[nb];
                        if nb_part >= 0 && (nb_part as usize) < num_parts_usize {
                            part_votes[nb_part as usize] += wt;
                            has_votes = true;
                        }
                    }
                }
            }

            let start_pref: i32 = if has_votes {
                let mut best_v = i32::MIN;
                let mut best_p = 0i32;
                for p in 0..num_parts_usize {
                    if counts_b[p] < max_per_lp {
                        let v = part_votes[p];
                        if v > best_v || (v == best_v && (p as i32) < best_p) {
                            best_v = v;
                            best_p = p as i32;
                        }
                    }
                }
                best_p
            } else {
                pref_parts[nd]
            };

            let mut placed = false;
            for d in 0..num_parts_usize {
                let p = ((start_pref as usize) + d) % num_parts_usize;
                if counts_b[p] < max_per_lp {
                    partition_b[nd] = p as i32;
                    counts_b[p] += 1;
                    placed = true;
                    break;
                }
            }
            if !placed {
                let fb = counts_b.iter().enumerate().min_by_key(|&(_, &c)| c).map(|(p, _)| p).unwrap_or(0);
                partition_b[nd] = fb as i32;
                counts_b[fb] += 1;
            }
        }

        // Step 2: Cheap FM on partition_B — 2 GPU rounds, greedy gain>0 (40990c33: improve before consensus).
        stream.memcpy_htod(&partition_b, &mut d_partition)?;
        stream.memcpy_htod(&counts_b, &mut d_nodes_in_part)?;

        let lp_fm_rounds = 2usize;
        let lp_move_limit = (move_limit / 10).max(1000);
        for _fm_r in 0..lp_fm_rounds {
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

            let mut lp_top: Vec<(usize, i32)> = move_keys_host
                .iter()
                .enumerate()
                .filter(|(_, &k)| k as u32 != 0x80000000 && (k >> 16) > 0)
                .map(|(n, &k)| (n, k))
                .collect();
            lp_top.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            lp_top.truncate(lp_move_limit);
            if lp_top.is_empty() { break; }

            stream.memcpy_dtoh(&d_partition, &mut partition_b)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut counts_b)?;
            for &(nd, key) in lp_top.iter() {
                let tgt = (key & 63) as i32;
                if nd >= partition_b.len() || tgt < 0 { continue; }
                let cur = partition_b[nd];
                if cur < 0 { continue; }
                let cu = cur as usize;
                let tu = tgt as usize;
                if tu < counts_b.len() && cu < counts_b.len()
                    && counts_b[tu] < max_per_lp && counts_b[cu] > 1 {
                    partition_b[nd] = tgt;
                    counts_b[cu] -= 1;
                    counts_b[tu] += 1;
                }
            }
            stream.memcpy_htod(&partition_b, &mut d_partition)?;
            stream.memcpy_htod(&counts_b, &mut d_nodes_in_part)?;
        }

        // Step 3: Merge partition_A + partition_B_FM — agree=keep, disagree=re-assign pref_parts DESC.
        let mut merged_counts = vec![0i32; num_parts_usize];
        for i in 0..n_nodes {
            if partition_a[i] == partition_b[i] {
                partition_host_refine[i] = partition_a[i];
                merged_counts[partition_a[i] as usize] += 1;
            } else {
                partition_host_refine[i] = -1;
            }
        }
        let max_per_merge = challenge.max_part_size as i32;
        for &i in indices.iter() {
            if partition_host_refine[i] >= 0 { continue; }
            let preferred = pref_parts[i];
            let mut placed = false;
            for d in 0..(num_parts_usize as i32) {
                let p = (preferred + d).rem_euclid(num_parts_usize as i32);
                if merged_counts[p as usize] < max_per_merge {
                    partition_host_refine[i] = p;
                    merged_counts[p as usize] += 1;
                    placed = true;
                    break;
                }
            }
            if !placed {
                let p = merged_counts.iter().position(|&c| c < max_per_merge).unwrap_or(0) as i32;
                partition_host_refine[i] = p;
                merged_counts[p as usize] += 1;
            }
        }
        nodes_in_part_host = merged_counts;
        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
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
        let aspiration_threshold = ((max_gain as f64) * aspiration_frac) as i32;
        
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

    // ============================================================================================
    
    // --------------------------------------------------------------------------------------------
    // The vertex-centric operator (do_swap_phase! pair/3-cycle, gain>0) is saturated on t24's 200k
    // hyperedges: a net scattered over k≥3 blocks has NO single-pin move with gain>0 ⇒ the zero-gain
    
    // one that consolidates MANY pins of the same net in ONE atomic move. For the `collapse_top_k`
    
    // ALL minority pins to the plurality block B. Δconnectivity is EXACT via per-hedge pin counts
    // (epc, t24/i16 5d21b2c3; Algorithm-6.2 counting c15c6a7a). Two hard guards make it monotone +
    
    //   (1) CAPACITY: gated to the EXACT verifier constraint nodes_in_part[p] ∈ [1, max_part_size]
    //       (c005 count-based balance) — NEVER widen the slack beyond the verifier.
    //   (2) MONOTONE: commit a group move ssi exact Δλ < 0 ; empty/infeasible/non-improving = NO-OP.
    // IN-SEQUENCE (upstream of best_connectivity below). collapse_top_k=0 ⇒ skipped ⇒ baseline.
    if collapse_top_k > 0 {
        let np = num_parts_usize;
        let nh_count = challenge.num_hyperedges as usize;
        let max_ps = challenge.max_part_size as i32;

        let mut part: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
        // exact per-part node counts (rebuilt from the partition — count-based, matches verifier)
        let mut nip: Vec<i32> = vec![0i32; np];
        for &p in part.iter() {
            let pu = p as usize;
            if pu < np { nip[pu] += 1; }
        }
        // exact per-hedge pin counts + per-hedge connectivity λ for the current partition
        let mut epc: Vec<i32> = vec![0i32; nh_count * np];
        let mut lambda: Vec<i32> = vec![0i32; nh_count];
        for h in 0..nh_count {
            let s = hedge_offsets_host[h] as usize;
            let e = hedge_offsets_host[h + 1] as usize;
            let base = h * np;
            for k in s..e {
                let p = part[hyperedge_nodes_host[k] as usize] as usize;
                if p < np { epc[base + p] += 1; }
            }
            let mut lam = 0i32;
            for p in 0..np { if epc[base + p] > 0 { lam += 1; } }
            lambda[h] = lam;
        }
        // rank scattered hyperedges (λ≥3 = the zero-gain trap) by λ desc, tie-break id asc (deterministic)
        let mut order: Vec<usize> = (0..nh_count).filter(|&h| lambda[h] >= 3).collect();
        order.sort_unstable_by(|&a, &b| lambda[b].cmp(&lambda[a]).then(a.cmp(&b)));
        let topk = std::cmp::min(collapse_top_k, order.len());

        let mut s_nodes: Vec<usize> = Vec::new();
        let mut aff: Vec<usize> = Vec::new();
        let mut stamp: Vec<u32> = vec![0u32; nh_count];
        let mut stamp_tok: u32 = 0;
        let mut delta_count: Vec<i32> = vec![0i32; np];
        let mut collapsed = 0usize;
        // i19 best-prefix scratch: marginal-ordered minority pins + the walked (tentatively applied) prefix
        let mut s_order: Vec<(i32, usize)> = Vec::new();
        let mut walked: Vec<usize> = Vec::new();

        for &h in order.iter().take(topk) {
            let base = h * np;
            // plurality block B (tie-break lowest index)
            let mut b = 0usize;
            let mut bc = epc[base];
            for p in 1..np { if epc[base + p] > bc { bc = epc[base + p]; b = p; } }
            // minority pins of h (nodes not in B)
            s_nodes.clear();
            let s = hedge_offsets_host[h] as usize;
            let e = hedge_offsets_host[h + 1] as usize;
            for k in s..e {
                let n = hyperedge_nodes_host[k] as usize;
                if part[n] as usize != b { s_nodes.push(n); }
            }
            if s_nodes.is_empty() { continue; }

            if collapse_subseq > 0 {
                // ====================================================================================
                
                // Order minority pins by marginal Δλ asc (best benefit first), then accumulate Δλ
                // IN-SEQUENCE (recompute each move's gain assuming the prior committed moves applied —
                
                // all-or-nothing failure: moving ALL pins inflates λ of the OTHER nets of those pins so
                // Δtotal≥0 (no-op); the optimal prefix stops before those harmful tail moves. Strict
                // Δλ<0 + EXACT capacity ⇒ monotone & balance-valid (c005; i25 catastrophe fa630dc2).
                // ====================================================================================
                // (a) order pins by IN-ISOLATION marginal Δλ (current epc), tie-break node id asc
                s_order.clear();
                for &n in s_nodes.iter() {
                    let from = part[n] as usize;
                    let ns = node_offsets_host[n] as usize;
                    let ne = node_offsets_host[n + 1] as usize;
                    let mut dl = 0i32;
                    for kk in ns..ne {
                        let hb = node_hedges_host[kk] as usize * np;
                        if epc[hb + from] == 1 { dl -= 1; } // from-part empties ⇒ λ−1
                        if epc[hb + b] == 0 { dl += 1; }     // B newly occupied ⇒ λ+1
                    }
                    s_order.push((dl, n));
                }
                s_order.sort_unstable_by(|a, c| a.0.cmp(&c.0).then(a.1.cmp(&c.1)));

                // (b) in-sequence walk: tentatively apply each move to epc, accumulate true marginal
                //     Δλ, track the prefix with minimum cumulative Δλ. Capacity tracked incrementally:
                //     all moves go INTO B (grows), out-parts shrink. delta_count = per-part out-count.
                for d in delta_count.iter_mut() { *d = 0; }
                walked.clear();
                let mut cum = 0i32;
                let mut best_cum = 0i32; // empty prefix
                let mut best_len = 0usize;
                let mut nb_added = 0i32;
                for &(_, n) in s_order.iter() {
                    let from = part[n] as usize;
                    // guard (1): EXACT capacity — B must not overflow, from must not empty
                    if nip[b] + nb_added + 1 > max_ps { break; }
                    if nip[from] - (delta_count[from] + 1) < 1 { break; }
                    // true in-sequence marginal Δλ over n's incident hedges, then apply to epc
                    let ns = node_offsets_host[n] as usize;
                    let ne = node_offsets_host[n + 1] as usize;
                    let mut dl = 0i32;
                    for kk in ns..ne {
                        let hb = node_hedges_host[kk] as usize * np;
                        if epc[hb + from] == 1 { dl -= 1; }
                        if epc[hb + b] == 0 { dl += 1; }
                        epc[hb + from] -= 1;
                        epc[hb + b] += 1;
                    }
                    cum += dl;
                    delta_count[from] += 1;
                    nb_added += 1;
                    walked.push(n);
                    if cum < best_cum { best_cum = cum; best_len = walked.len(); }
                }

                // (c) commit the best prefix iff strictly improving; undo every non-committed move
                if best_len > 0 && best_cum < 0 {
                    // roll back walked[best_len..] (tentatively applied above)
                    for &n in walked[best_len..].iter() {
                        let from = part[n] as usize;
                        let ns = node_offsets_host[n] as usize;
                        let ne = node_offsets_host[n + 1] as usize;
                        for kk in ns..ne {
                            let hb = node_hedges_host[kk] as usize * np;
                            epc[hb + from] += 1;
                            epc[hb + b] -= 1;
                        }
                    }
                    // commit walked[..best_len]: part/nip + collect affected hedges for λ refresh
                    stamp_tok += 1;
                    aff.clear();
                    for &n in walked[..best_len].iter() {
                        let from = part[n] as usize;
                        nip[from] -= 1;
                        nip[b] += 1;
                        part[n] = b as i32;
                        let ns = node_offsets_host[n] as usize;
                        let ne = node_offsets_host[n + 1] as usize;
                        for kk in ns..ne {
                            let hh = node_hedges_host[kk] as usize;
                            if stamp[hh] != stamp_tok { stamp[hh] = stamp_tok; aff.push(hh); }
                        }
                    }
                    for &hh in aff.iter() {
                        let hb = hh * np;
                        let mut lam = 0i32;
                        for p in 0..np { if epc[hb + p] > 0 { lam += 1; } }
                        lambda[hh] = lam;
                    }
                    collapsed += 1;
                } else {
                    // no improving prefix: undo ALL walked epc changes (full no-op for this hedge)
                    for &n in walked.iter() {
                        let from = part[n] as usize;
                        let ns = node_offsets_host[n] as usize;
                        let ne = node_offsets_host[n + 1] as usize;
                        for kk in ns..ne {
                            let hb = node_hedges_host[kk] as usize * np;
                            epc[hb + from] += 1;
                            epc[hb + b] -= 1;
                        }
                    }
                }
                continue;
            }

            // ---- guard (1): EXACT capacity (verifier c005: count ∈ [1, max_part_size]) ----
            for d in delta_count.iter_mut() { *d = 0; }
            for &n in s_nodes.iter() { delta_count[part[n] as usize] += 1; }
            let mut feasible = nip[b] + (s_nodes.len() as i32) <= max_ps;
            if feasible {
                for p in 0..np {
                    if p == b { continue; }
                    if delta_count[p] > 0 && nip[p] - delta_count[p] < 1 { feasible = false; break; }
                }
            }
            if !feasible { continue; }

            // ---- exact Δconnectivity over the affected hedges (incident to any moved node) ----
            stamp_tok += 1;
            aff.clear();
            for &n in s_nodes.iter() {
                let ns = node_offsets_host[n] as usize;
                let ne = node_offsets_host[n + 1] as usize;
                for kk in ns..ne {
                    let hh = node_hedges_host[kk] as usize;
                    if stamp[hh] != stamp_tok { stamp[hh] = stamp_tok; aff.push(hh); }
                }
            }
            let mut lam_before: i32 = 0;
            for &hh in aff.iter() { lam_before += lambda[hh]; }
            // tentatively apply moves to epc (each moved node: from → B)
            for &n in s_nodes.iter() {
                let from = part[n] as usize;
                let ns = node_offsets_host[n] as usize;
                let ne = node_offsets_host[n + 1] as usize;
                for kk in ns..ne {
                    let hb = node_hedges_host[kk] as usize * np;
                    epc[hb + from] -= 1;
                    epc[hb + b] += 1;
                }
            }
            let mut lam_after: i32 = 0;
            for &hh in aff.iter() {
                let hb = hh * np;
                let mut lam = 0i32;
                for p in 0..np { if epc[hb + p] > 0 { lam += 1; } }
                lam_after += lam;
            }

            // ---- guard (2): MONOTONE — commit ssi exact Δλ < 0, else undo (cannot regress) ----
            if lam_after - lam_before < 0 {
                for &n in s_nodes.iter() {
                    let from = part[n] as usize;
                    nip[from] -= 1;
                    nip[b] += 1;
                    part[n] = b as i32;
                }
                for &hh in aff.iter() {
                    let hb = hh * np;
                    let mut lam = 0i32;
                    for p in 0..np { if epc[hb + p] > 0 { lam += 1; } }
                    lambda[hh] = lam;
                }
                collapsed += 1;
            } else {
                for &n in s_nodes.iter() {
                    let from = part[n] as usize;
                    let ns = node_offsets_host[n] as usize;
                    let ne = node_offsets_host[n + 1] as usize;
                    for kk in ns..ne {
                        let hb = node_hedges_host[kk] as usize * np;
                        epc[hb + from] += 1;
                        epc[hb + b] -= 1;
                    }
                }
            }
        }

        if collapsed > 0 {
            stream.memcpy_htod(&part, &mut d_partition)?;
            stream.memcpy_htod(&nip, &mut d_nodes_in_part)?;
        }
    }
    // ============================================================================================

    // ============================================================================================
    
    // --------------------------------------------------------------------------------------------
    // i18 (net-collapse all-or-nothing) and i19 (per-net best-prefix) both harvested EXACT NULL on
    // t24: a per-net group move forces every minority pin INTO that net's plurality block B, which
    // inflates λ of the OTHER nets those pins belong to ⇒ Δtotal≥0 ⇒ 0 move committed. The PUBLISHED
    // cure (Mt-KaHyPar 2nd FM phase: combined prefix-sum + reduce to find the best balanced prefix,
    
    
    
    
    //   (a) collect EVERY boundary vertex's BEST single-pin move (target t minimising the isolated
    //       Δλ — t = argmax_{p≠from} pres[p] where pres[p]=#incident hedges already containing p;
    //       NOT forced to one net's plurality block as in i18/i19), keep moves with isolated Δλ ≤ 0;
    //   (b) sort the global candidate sequence by gain (Δλ asc, node id asc, deterministic);
    //   (c) walk IN-SEQUENCE recomputing each move's true Δλ assuming all prior committed moves
    //       applied (tentatively mutate epc), accumulate cumulative Δλ, track the prefix of minimum
    //       cumulative Δλ (= best balanced prefix);
    //   (d) commit that prefix iff strictly improving (best_cum < 0), revert the tail.
    // Unlike per-net, the OTHER nets' pins are ALSO candidates in the SAME global sequence, so the
    // λ-inflation they cause can be UNDONE within the pass ⇒ a scattered net can fully collapse where
    // the zero-gain pair-swap operator sees Δ=0 (the 749 s/+406 s slack pays for this heavier pass).
    // Guards (identical invariants to i18/i19 — monotone & balance-valid for the c005 verifier):
    //   (1) CAPACITY: per running state nip_cur[t]+1 ≤ max_part_size AND nip_cur[from]-1 ≥ 1 (EXACT
    //       count-based balance ∈ [1, max_part_size], never widened). A capacity-blocked move is
    //       SKIPPED (continue), not break (moves target VARIED parts, ≠ i19 where all grew B). Each
    //       committed move's check saw only prior committed (prefix) moves ⇒ every intermediate
    //       committed state is valid ⇒ final state valid.
    //   (2) MONOTONE: commit the best prefix ssi best_cum < 0 STRICT; else full no-op for the pass.
    //       Each node is moved at most once per pass (precondition for in-sequence gain recalc,
    
    //       stop early when a pass commits nothing). 0 ⇒ skipped ⇒ baseline exact (abort control).
    if global_prefix > 0 {
        let np = num_parts_usize;
        let nh_count = challenge.num_hyperedges as usize;
        let nn = challenge.num_nodes as usize;
        let max_ps = challenge.max_part_size as i32;

        let mut part: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
        let mut nip: Vec<i32> = vec![0i32; np];
        for &p in part.iter() { let pu = p as usize; if pu < np { nip[pu] += 1; } }
        // exact per-hedge pin counts epc[h*np + p] for the current partition
        let mut epc: Vec<i32> = vec![0i32; nh_count * np];
        for h in 0..nh_count {
            let s = hedge_offsets_host[h] as usize;
            let e = hedge_offsets_host[h + 1] as usize;
            let base = h * np;
            for k in s..e {
                let p = part[hyperedge_nodes_host[k] as usize] as usize;
                if p < np { epc[base + p] += 1; }
            }
        }
        // per-hedge present-part bitmask (np ≤ 64 on c005); recomputed at each pass start
        let mut hmask: Vec<u64> = vec![0u64; nh_count];

        let mut cand: Vec<(i32, u32, u32)> = Vec::new(); // (isolated Δλ, node, target)
        let mut pres: Vec<i32> = vec![0i32; np];
        let mut touched: Vec<usize> = Vec::with_capacity(np);
        let mut nip_cur: Vec<i32> = vec![0i32; np];
        let mut walked: Vec<(u32, u32)> = Vec::new(); // (node, target)
        let mut total_committed = 0usize;
        
        // appearing with k targets is applied at most once during the in-sequence walk. Unused for
        // gp_topk=1 (one candidate/node ⇒ guard never fires) ⇒ exact i20/i21 behaviour preserved.
        let mut topk_buf: Vec<(i32, u32)> = Vec::new(); // (isolated Δλ, target)
        let mut moved: Vec<bool> = vec![false; nn];

        
        // fl_n_pairs=0 → these are never called → zero overhead.
        #[inline(always)]
        fn fl30_add_edge(g: &mut Vec<Vec<usize>>, e_to: &mut Vec<usize>,
                         e_cap: &mut Vec<i64>, u: usize, v: usize, c: i64) {
            let i = e_to.len();
            g[u].push(i);     e_to.push(v); e_cap.push(c);
            g[v].push(i + 1); e_to.push(u); e_cap.push(0);
        }
        fn fl30_bfs(s: usize, t: usize, g: &Vec<Vec<usize>>, e_to: &[usize],
                    e_cap: &[i64], level: &mut [i32]) -> bool {
            for x in level.iter_mut() { *x = -1; }
            level[s] = 0;
            let mut q = vec![s]; let mut head = 0usize;
            while head < q.len() {
                let u = q[head]; head += 1;
                for &idx in &g[u] {
                    let v = e_to[idx];
                    if e_cap[idx] > 0 && level[v] < 0 { level[v] = level[u] + 1; q.push(v); }
                }
            }
            level[t] >= 0
        }
        fn fl30_dfs(u: usize, t: usize, f: i64, g: &Vec<Vec<usize>>, e_to: &[usize],
                    e_cap: &mut [i64], level: &[i32], it: &mut [usize]) -> i64 {
            if u == t { return f; }
            while it[u] < g[u].len() {
                let idx = g[u][it[u]];
                let v = e_to[idx];
                if e_cap[idx] > 0 && level[v] == level[u] + 1 {
                    let d = fl30_dfs(v, t, f.min(e_cap[idx]), g, e_to, e_cap, level, it);
                    if d > 0 { e_cap[idx] -= d; e_cap[idx ^ 1] += d; return d; }
                }
                it[u] += 1;
            }
            0
        }
        // flow candidate vector (node, target) — populated per-pass, pre-walked before cand
        let mut flow_cand: Vec<(u32, u32)> = Vec::new();
        // scratch (allocated once, reused across passes/pairs to avoid per-pair alloc)
        let mut fl_in_region: Vec<bool> = vec![false; nn];
        let mut fl_rid: Vec<i32> = vec![-1i32; nn];
        let mut fl_hmark: Vec<u32> = vec![0u32; nh_count];
        let mut fl_cur_tag: u32 = 0u32;
        let mut fl_reg_pins: Vec<usize> = Vec::new();

        for _pass in 0..global_prefix {
            // refresh present-part bitmask from epc
            for h in 0..nh_count {
                let base = h * np;
                let mut m = 0u64;
                for p in 0..np { if epc[base + p] > 0 { m |= 1u64 << p; } }
                hmask[h] = m;
            }
            
            flow_cand.clear();
            if fl_n_pairs > 0 {
                const FL30_NODE_CAP: usize = 256;
                const FL30_HEDGE_CAP: usize = 80_000;
                const FL30_INF: i64 = 1i64 << 40;

                // cut weight per pair = #hedges spanning both parts (from epc)
                let mut pair_cut: Vec<(u32, usize, usize)> = Vec::new();
                for a in 0..np {
                    for b in (a + 1)..np {
                        let mut w = 0u32;
                        for h in 0..nh_count {
                            let base = h * np;
                            if epc[base + a] > 0 && epc[base + b] > 0 { w += 1; }
                        }
                        if w > 0 { pair_cut.push((w, a, b)); }
                    }
                }
                pair_cut.sort_by(|x, y| y.0.cmp(&x.0));

                for pi in 0..fl_n_pairs.min(pair_cut.len()) {
                    let (_, a, b) = pair_cut[pi];

                    // boundary region: immediate a<->b boundary nodes, BFS-grown within a∪b
                    let mut region: Vec<usize> = Vec::new();
                    for v in 0..nn {
                        let pv = part[v] as usize;
                        if pv != a && pv != b { continue; }
                        let other = if pv == a { b } else { a };
                        let ns_v = node_offsets_host[v] as usize;
                        let ne_v = node_offsets_host[v + 1] as usize;
                        let touches = (ns_v..ne_v).any(|hi| {
                            let h = node_hedges_host[hi] as usize;
                            epc[h * np + other] > 0
                        });
                        if touches { fl_in_region[v] = true; region.push(v); }
                    }
                    if region.len() < 2 {
                        for &v in &region { fl_in_region[v] = false; }
                        continue;
                    }
                    // BFS growth within a∪b, capped at FL30_NODE_CAP
                    let mut qi = 0usize;
                    while qi < region.len() && region.len() < FL30_NODE_CAP {
                        let v = region[qi]; qi += 1;
                        let ns_v = node_offsets_host[v] as usize;
                        let ne_v = node_offsets_host[v + 1] as usize;
                        let mut full = false;
                        for hi in ns_v..ne_v {
                            let h = node_hedges_host[hi] as usize;
                            let hs = hedge_offsets_host[h] as usize;
                            let he = hedge_offsets_host[h + 1] as usize;
                            for k in hs..he {
                                let u = hyperedge_nodes_host[k] as usize;
                                if !fl_in_region[u] {
                                    let pu = part[u] as usize;
                                    if pu == a || pu == b {
                                        fl_in_region[u] = true; region.push(u);
                                        if region.len() >= FL30_NODE_CAP { full = true; break; }
                                    }
                                }
                            }
                            if full { break; }
                        }
                    }

                    // collect incident hedges (dedup via tag)
                    fl_cur_tag += 1;
                    let mut h_list: Vec<usize> = Vec::new();
                    for &v in &region {
                        let ns_v = node_offsets_host[v] as usize;
                        let ne_v = node_offsets_host[v + 1] as usize;
                        for hi in ns_v..ne_v {
                            let h = node_hedges_host[hi] as usize;
                            if fl_hmark[h] != fl_cur_tag { fl_hmark[h] = fl_cur_tag; h_list.push(h); }
                        }
                    }
                    if h_list.len() > FL30_HEDGE_CAP {
                        for &v in &region { fl_in_region[v] = false; }
                        continue;
                    }

                    // Lawler/Liu-Wong flow network: node 0=source(a-side), 1=sink(b-side)
                    let nb = region.len();
                    for (i, &v) in region.iter().enumerate() { fl_rid[v] = (2 + i) as i32; }
                    let mut g_fl: Vec<Vec<usize>> = (0..2 + nb).map(|_| Vec::new()).collect();
                    let mut e_to_fl: Vec<usize> = Vec::new();
                    let mut e_cap_fl: Vec<i64> = Vec::new();
                    for &h in &h_list {
                        let hs = hedge_offsets_host[h] as usize;
                        let he = hedge_offsets_host[h + 1] as usize;
                        let mut has_fa = false; let mut has_fb = false;
                        fl_reg_pins.clear();
                        for k in hs..he {
                            let nd = hyperedge_nodes_host[k] as usize;
                            let r = fl_rid[nd];
                            if r >= 2 { fl_reg_pins.push(r as usize); }
                            else {
                                let pb = part[nd] as usize;
                                if pb == a { has_fa = true; }
                                else if pb == b { has_fb = true; }
                            }
                        }
                        if fl_reg_pins.is_empty() { continue; }
                        let e_in = g_fl.len(); g_fl.push(Vec::new());
                        let e_out = g_fl.len(); g_fl.push(Vec::new());
                        fl30_add_edge(&mut g_fl, &mut e_to_fl, &mut e_cap_fl, e_in, e_out, 1);
                        if has_fa { fl30_add_edge(&mut g_fl, &mut e_to_fl, &mut e_cap_fl, 0, e_in, FL30_INF); }
                        if has_fb { fl30_add_edge(&mut g_fl, &mut e_to_fl, &mut e_cap_fl, e_out, 1, FL30_INF); }
                        for &rp in &fl_reg_pins {
                            fl30_add_edge(&mut g_fl, &mut e_to_fl, &mut e_cap_fl, rp, e_in, FL30_INF);
                            fl30_add_edge(&mut g_fl, &mut e_to_fl, &mut e_cap_fl, e_out, rp, FL30_INF);
                        }
                    }
                    let nvtot = g_fl.len();

                    // Dinic max-flow s=0 → t=1
                    let mut level = vec![-1i32; nvtot];
                    while fl30_bfs(0, 1, &g_fl, &e_to_fl, &e_cap_fl, &mut level) {
                        let mut it = vec![0usize; nvtot];
                        loop {
                            let f = fl30_dfs(0, 1, FL30_INF, &g_fl, &e_to_fl, &mut e_cap_fl, &level, &mut it);
                            if f == 0 { break; }
                        }
                    }

                    // source-side min-cut: reachable from source via residual
                    let mut vis_s = vec![false; nvtot]; vis_s[0] = true;
                    let mut st_fl = vec![0usize];
                    while let Some(u) = st_fl.pop() {
                        for &idx in &g_fl[u] {
                            let v = e_to_fl[idx];
                            if e_cap_fl[idx] > 0 && !vis_s[v] { vis_s[v] = true; st_fl.push(v); }
                        }
                    }

                    // extract flow-induced moves: region node on wrong side of min-cut
                    // node in a but SINK-side (!vis_s) → flow suggests move to b
                    // node in b but SOURCE-side (vis_s) → flow suggests move to a
                    for &v in &region {
                        let pv = part[v] as usize;
                        let r = fl_rid[v] as usize;
                        let (target, on_wrong_side) = if pv == a { (b, !vis_s[r]) } else { (a, vis_s[r]) };
                        if on_wrong_side { flow_cand.push((v as u32, target as u32)); }
                    }

                    // cleanup: reset rid and in_region for next pair
                    for &v in &region { fl_rid[v] = -1; fl_in_region[v] = false; }
                }
            }

            // (a) per-node best single-pin move (target maximising pres[t]); keep if isolated Δλ ≤ 0
            cand.clear();
            for n in 0..nn {
                let from = part[n] as usize;
                if from >= np { continue; }
                if nip[from] <= 1 { continue; } // a singleton part cannot be emptied
                let ns = node_offsets_host[n] as usize;
                let ne = node_offsets_host[n + 1] as usize;
                let d = ne - ns;
                if d == 0 { continue; }
                // pres[p] = #incident hedges already containing part p ; from_loss = #incident hedges
                // where `from` is present exactly once (emptied if n leaves)
                let mut from_loss = 0i32;
                touched.clear();
                for kk in ns..ne {
                    let h = node_hedges_host[kk] as usize;
                    let base = h * np;
                    if epc[base + from] == 1 { from_loss += 1; }
                    let mut m = hmask[h];
                    while m != 0 {
                        let p = m.trailing_zeros() as usize;
                        m &= m - 1;
                        if pres[p] == 0 { touched.push(p); }
                        pres[p] += 1;
                    }
                }
                if gp_topk <= 1 {
                    // i20/i21 EXACT: single best target t≠from maximising pres[t] (minimising
                    // Δλ = -from_loss + d - pres[t]); tie-break lowest part index for determinism.
                    let mut best_t = usize::MAX;
                    let mut best_pres = -1i32;
                    for &p in touched.iter() {
                        if p == from { continue; }
                        if pres[p] > best_pres || (pres[p] == best_pres && (best_t == usize::MAX || p < best_t)) {
                            best_pres = pres[p];
                            best_t = p;
                        }
                    }
                    for &p in touched.iter() { pres[p] = 0; }
                    if best_t == usize::MAX { continue; }
                    let dl_iso = -from_loss + (d as i32) - best_pres;
                    
                    // Candidates with 0 < dl_iso ≤ ab_thresh are worse-in-isolation but may be promoted
                    // to Δλ<0 by the in-sequence recompute below (their from-part may already be emptied
                    // / target already present once an earlier move ran). Sorted last by gain so their
                    // enabling moves precede them. Commit gate (best_cum<0) keeps validity intact.
                    if dl_iso <= ab_thresh {
                        cand.push((dl_iso, n as u32, best_t as u32));
                    }
                } else {
                    
                    // authorise more destinations/node). The walk picks the first feasible one reached
                    // in gain-density order; alternatives let a node still move if its densest target
                    // is capacity-blocked. Deterministic: keep k smallest dl_iso, tie lowest part index.
                    topk_buf.clear();
                    for &p in touched.iter() {
                        if p == from { continue; }
                        let dl_p = -from_loss + (d as i32) - pres[p];
                        if dl_p <= ab_thresh {
                            topk_buf.push((dl_p, p as u32));
                        }
                    }
                    for &p in touched.iter() { pres[p] = 0; }
                    if topk_buf.is_empty() { continue; }
                    topk_buf.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                    let keep = gp_topk.min(topk_buf.len());
                    for &(dl_p, p) in topk_buf[..keep].iter() {
                        cand.push((dl_p, n as u32, p));
                    }
                }
            }
            if cand.is_empty() { break; }
            
            
            // -Δλ_iso, weight = node degree; compare via cross-multiply (a before b iff ga/wa > gb/wb
            // ⇔ ga*wb > gb*wa). Ties: higher absolute gain first, then (i32) target-load asc if
            // gp_target_load_tb=true, then node id asc, then target asc.
            if gp_topk <= 1 {
                cand.sort_unstable_by(|a, c| a.0.cmp(&c.0).then(a.1.cmp(&c.1)));
            } else if gp_target_load_tb {
                
                // subsequent moves within the same pass; capacity guard fires less often later).
                // nip_cur == nip at sort time (walk hasn't started yet). OFF flag ≡ i31 exact.
                cand.sort_unstable_by(|a, c| {
                    let ga = (-a.0) as i64;
                    let wa = (node_offsets_host[a.1 as usize + 1] - node_offsets_host[a.1 as usize]) as i64;
                    let gb = (-c.0) as i64;
                    let wb = (node_offsets_host[c.1 as usize + 1] - node_offsets_host[c.1 as usize]) as i64;
                    (gb * wa).cmp(&(ga * wb))                          // density desc
                        .then(gb.cmp(&ga))                             // absolute gain desc
                        .then(nip_cur[a.2 as usize].cmp(&nip_cur[c.2 as usize])) // target load asc (i32)
                        .then(a.1.cmp(&c.1))                           // node id asc
                        .then(a.2.cmp(&c.2))                           // target asc
                });
            } else {
                cand.sort_unstable_by(|a, c| {
                    let ga = (-a.0) as i64;
                    let wa = (node_offsets_host[a.1 as usize + 1] - node_offsets_host[a.1 as usize]) as i64;
                    let gb = (-c.0) as i64;
                    let wb = (node_offsets_host[c.1 as usize + 1] - node_offsets_host[c.1 as usize]) as i64;
                    (gb * wa).cmp(&(ga * wb))      // density desc
                        .then(gb.cmp(&ga))         // higher absolute gain first
                        .then(a.1.cmp(&c.1))       // node id asc
                        .then(a.2.cmp(&c.2))       // target asc
                });
            }

            // (c) in-sequence walk: tentatively apply each move to epc, recompute true marginal Δλ,
            //     track the prefix with minimum cumulative Δλ. Capacity per running nip_cur.
            nip_cur.copy_from_slice(&nip);
            walked.clear();
            
            // may be applied at most once). No-op for gp_topk=1 (one candidate/node).
            for m in moved.iter_mut() { *m = false; }
            let mut cum = 0i32;
            let mut best_cum = 0i32; // empty prefix
            let mut best_len = 0usize;

            
            // In-sequence epc update = correct chain-effect gain capture.
            // Monotone guard (best_cum<0) and capacity guards unchanged → validity preserved.
            // fl_n_pairs=0 → flow_cand empty → this loop is a no-op → bit-identical to i22.
            for &(n_u, t_u) in flow_cand.iter() {
                let n = n_u as usize;
                let t = t_u as usize;
                if moved[n] { continue; }
                let from = part[n] as usize;
                if from == t { continue; } // already moved to target by prior flow candidate
                if nip_cur[from] - 1 < 1 { continue; }
                if nip_cur[t] + 1 > max_ps { continue; }
                let ns = node_offsets_host[n] as usize;
                let ne = node_offsets_host[n + 1] as usize;
                let mut dl = 0i32;
                for kk in ns..ne {
                    let hb = node_hedges_host[kk] as usize * np;
                    if epc[hb + from] == 1 { dl -= 1; }
                    if epc[hb + t] == 0 { dl += 1; }
                    epc[hb + from] -= 1;
                    epc[hb + t] += 1;
                }
                cum += dl;
                nip_cur[from] -= 1;
                nip_cur[t] += 1;
                moved[n] = true;
                walked.push((n_u, t_u));
                if cum < best_cum { best_cum = cum; best_len = walked.len(); }
            }

            for &(_, n_u, t_u) in cand.iter() {
                let n = n_u as usize;
                let t = t_u as usize;
                
                if moved[n] { continue; }
                let from = part[n] as usize;
                // guard (1): EXACT capacity (verifier c005: count ∈ [1, max_part_size])
                if nip_cur[from] - 1 < 1 { continue; }
                if nip_cur[t] + 1 > max_ps { continue; }
                let ns = node_offsets_host[n] as usize;
                let ne = node_offsets_host[n + 1] as usize;
                let mut dl = 0i32;
                for kk in ns..ne {
                    let hb = node_hedges_host[kk] as usize * np;
                    if epc[hb + from] == 1 { dl -= 1; }
                    if epc[hb + t] == 0 { dl += 1; }
                    epc[hb + from] -= 1;
                    epc[hb + t] += 1;
                }
                cum += dl;
                nip_cur[from] -= 1;
                nip_cur[t] += 1;
                moved[n] = true; 
                walked.push((n_u, t_u));
                if cum < best_cum { best_cum = cum; best_len = walked.len(); }
            }

            // (d) commit best prefix iff strictly improving; revert everything otherwise
            if best_len > 0 && best_cum < 0 {
                // roll back the tail walked[best_len..] (tentatively applied above)
                for &(n_u, t_u) in walked[best_len..].iter() {
                    let n = n_u as usize;
                    let t = t_u as usize;
                    let from = part[n] as usize;
                    let ns = node_offsets_host[n] as usize;
                    let ne = node_offsets_host[n + 1] as usize;
                    for kk in ns..ne {
                        let hb = node_hedges_host[kk] as usize * np;
                        epc[hb + from] += 1;
                        epc[hb + t] -= 1;
                    }
                }
                // commit walked[..best_len]: part/nip (epc already reflects committed prefix)
                for &(n_u, t_u) in walked[..best_len].iter() {
                    let n = n_u as usize;
                    let t = t_u as usize;
                    let from = part[n] as usize;
                    nip[from] -= 1;
                    nip[t] += 1;
                    part[n] = t as i32;
                }
                total_committed += best_len;
            } else {
                // no improving prefix: undo ALL tentatively-applied epc changes, stop passing
                for &(n_u, t_u) in walked.iter() {
                    let n = n_u as usize;
                    let t = t_u as usize;
                    let from = part[n] as usize;
                    let ns = node_offsets_host[n] as usize;
                    let ne = node_offsets_host[n + 1] as usize;
                    for kk in ns..ne {
                        let hb = node_hedges_host[kk] as usize * np;
                        epc[hb + from] += 1;
                        epc[hb + t] -= 1;
                    }
                }
                break;
            }
        }

        if total_committed > 0 {
            stream.memcpy_htod(&part, &mut d_partition)?;
            stream.memcpy_htod(&nip, &mut d_nodes_in_part)?;
        }
    }
    // ============================================================================================

    let perturb_strength = 3;

    
    // (build_high_hedge_ids, computed once from initial partition at L1807) makes every even-ILS-iter
    // guided perturbation consolidate the SAME 500 hyperedges => re-enters the same basin every round
    
    // IDs from the CURRENT best partition each even round, subsample stochastically by ils_iter seed
    // => each round targets a distinct interaction structure => distinct basin => pool spreads out.
    // vigw_perturb_mode=1 (default) => i69 KEPT (even-round VIGw-lite).
    // 0=off(i63 bit-id), 1=even-only K=vigw_k, 2=all-rounds K=vigw_k,
    
    // 4=all-rounds adaptive box-plot β K. Default=1 (WIN i69).
    let vigw_perturb_mode: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("vigw_perturb_mode").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 4) as usize)
        .unwrap_or(1);
    // vigw_k: K target-hedges for VIGw (modes 1/2); fallback for modes 3/4 if boxplot empty.
    let vigw_k: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("vigw_k").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 4096) as usize)
        .unwrap_or(500);
    // vigw_iqr_mult: IQR multiplier for box-plot β threshold (modes 3/4). Default=1.5 (Q3+1.5*IQR).
    // Sweep {1.0, 2.0, 2.5}: smaller = tighter precision targets, larger = wider coverage.
    let vigw_iqr_mult: f64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("vigw_iqr_mult").and_then(|v| v.as_f64()))
        .map(|v| v.clamp(0.1, 10.0))
        .unwrap_or(1.5);
    
    
    // per-hedge from last do_swap_phase! swap_gains_host (true ω_{g,h} per 19d92cad, host-side free).
    
    // mode1 (FM-gain replaces structural, i85 -54Q); product keeps hedges BOTH cut AND high-gain.
    let vigw_score_mode: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("vigw_score_mode").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 2) as usize)
        .unwrap_or(0);
    
    
    // for each vigw_ids hedge, pick anchor vertex, extend to its adjacent vigw_ids hedges; guardrail
    // b0377bd1 adds 1 random adjacent cut-hedge if 0 coupled neighbors found).
    let vigw_couple_mode: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("vigw_couple_mode").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 1) as usize)
        .unwrap_or(0);
    
    
    // 1=edge-L1 diversity: sort cut hedges by L1(λ_best(h), λ_cur(h)) ASC, prioritizing hedges
    //   most similar to elite partition (low L1 = ambiguous basin, novel exploration direction).
    
    //   which used Hamming node-based distance). λ(e)[i]=pins of h in block i; L1=Σ|best[i]-cur[i]|.
    //   Lower fence box-plot: keep L1 ≤ Q1 − vigw_iqr_mult×IQR (most ambiguous cluster).
    let vigw_div_mode: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("vigw_div_mode").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 1) as usize)
        .unwrap_or(0);

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
            // VIGw coverage: modes 1/3=even rounds, modes 2/4=all rounds, mode 0=off.
            
            let apply_vigw = match vigw_perturb_mode {
                1 | 3 => ils_iter % 2 == 0,
                2 | 4 => true,
                _ => false,
            };
            let use_boxplot = vigw_perturb_mode >= 3;
            if apply_vigw {
                // FM-gain-weight (vigw_score_mode=1): per-hedge sum of max |gain| from last
                // do_swap_phase! swap_gains_host. True ω_{g,h}=|δgain FM| per 19d92cad, host-side.
                let hedge_fm_gain: Vec<i32> = if vigw_score_mode == 1 || vigw_score_mode == 2 {
                    let num_hedges = challenge.num_hyperedges as usize;
                    let mut gains = vec![0i32; num_hedges];
                    for node in 0..challenge.num_nodes as usize {
                        let max_ag: i32 = (0..3usize).filter_map(|k| {
                            let v = swap_gains_host[node * 3 + k];
                            if v == 0 { None } else { Some((((v >> 16) as i16) as i32).abs()) }
                        }).max().unwrap_or(0);
                        if max_ag > 0 {
                            let s = node_offsets_host[node] as usize;
                            let e = node_offsets_host[node + 1] as usize;
                            for idx in s..e {
                                let h = node_hedges_host[idx] as usize;
                                if h < num_hedges {
                                    gains[h] = gains[h].saturating_add(max_ag);
                                }
                            }
                        }
                    }
                    gains
                } else {
                    Vec::new()
                };
                let vigw_ids: Vec<i32> = {
                    let num_hedges = challenge.num_hyperedges as usize;
                    let np = std::cmp::min(num_parts_usize, 64usize);
                    let mut part_cnt = [0i32; 64];
                    let mut cut_hedges: Vec<(i32, i32, i32)> =
                        Vec::with_capacity(num_hedges / 4);
                    for h in 0..num_hedges {
                        let start = hedge_offsets_host[h] as usize;
                        let end = hedge_offsets_host[h + 1] as usize;
                        if end - start <= 1 {
                            continue;
                        }
                        let ps = &mut part_cnt[..np];
                        ps.fill(0);
                        for k in start..end {
                            let node = hyperedge_nodes_host[k] as usize;
                            let p = best_partition_host[node];
                            if p >= 0 && (p as usize) < np {
                                ps[p as usize] += 1;
                            }
                        }
                        let pp = ps.iter().filter(|&&c| c > 0).count();
                        if pp <= 1 {
                            continue;
                        }
                        cut_hedges.push(((pp - 1) as i32, hedge_sizes_host[h], h as i32));
                    }
                    if vigw_div_mode == 1 && !cut_hedges.is_empty() {
                        
                        // L1(λ_best(h), λ_cur(h)): per-block pin counts in elite vs current partition.
                        // LOW L1 = hedge in same state in both = ambiguous basin = novel exploration target.
                        // HIGH L1 = hedge already diverged from elite = skip (exploration already happened).
                        // Label-invariant: immune to isomorphism problem that killed i66 Hamming -3Q.
                        let mut l1_hedges: Vec<(i32, i32)> = cut_hedges.iter().map(|&(_, _, hid)| {
                            let h = hid as usize;
                            let hs2 = hedge_offsets_host[h] as usize;
                            let he2 = hedge_offsets_host[h + 1] as usize;
                            let mut cur_ps = [0i32; 64];
                            let mut best_ps2 = [0i32; 64];
                            for k in hs2..he2 {
                                let node = hyperedge_nodes_host[k] as usize;
                                let cp = partition_host_refine[node];
                                let bp = best_partition_host[node];
                                if cp >= 0 && (cp as usize) < np { cur_ps[cp as usize] += 1; }
                                if bp >= 0 && (bp as usize) < np { best_ps2[bp as usize] += 1; }
                            }
                            let l1: i32 = cur_ps[..np].iter().zip(best_ps2[..np].iter())
                                .map(|(a, b)| (a - b).abs()).sum();
                            (l1, hid)
                        }).collect();
                        // ASC: lowest L1 first (most similar to elite = most ambiguous = novel basin).
                        l1_hedges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                        // Lower-fence box-plot: keep L1 ≤ Q1 − vigw_iqr_mult×IQR (most ambiguous cluster).
                        if use_boxplot && l1_hedges.len() >= 4 {
                            let scores: Vec<i32> = l1_hedges.iter().map(|&(l1, _)| l1).collect();
                            let n = scores.len();
                            // ASC order: scores[n/4]=Q1 (25th pct), scores[3*n/4]=Q3 (75th pct).
                            let q1_v = scores[n / 4];
                            let q3_v = scores[3 * n / 4];
                            let iqr_v = q3_v.saturating_sub(q1_v);
                            let threshold = (q1_v as f64 - iqr_v as f64 * vigw_iqr_mult).round() as i32;
                            let adaptive: Vec<i32> = l1_hedges.iter()
                                .filter(|&&(l1, _)| l1 <= threshold)
                                .map(|&(_, hid)| hid)
                                .collect();
                            if adaptive.is_empty() {
                                l1_hedges.into_iter().take(vigw_k).map(|(_, hid)| hid).collect()
                            } else {
                                adaptive
                            }
                        } else {
                            l1_hedges.into_iter().take(vigw_k).map(|(_, hid)| hid).collect()
                        }
                    } else {
                    cut_hedges.sort_unstable_by(|a, b| {
                        b.0.cmp(&a.0)
                            .then_with(|| b.1.cmp(&a.1))
                            .then_with(|| a.2.cmp(&b.2))
                    });
                    if use_boxplot && cut_hedges.len() >= 4 {
                        if vigw_score_mode == 1 {
                            // FM-gain-weight: re-sort cut_hedges by FM-gain DESC, box-plot on
                            // FM-gain scores. Faithfully implements ω=|δgain FM| (19d92cad).
                            cut_hedges.sort_unstable_by(|a, b| {
                                hedge_fm_gain[b.2 as usize].cmp(&hedge_fm_gain[a.2 as usize])
                                    .then_with(|| a.2.cmp(&b.2))
                            });
                            let n = cut_hedges.len();
                            let scores: Vec<i32> = cut_hedges.iter()
                                .map(|&(_, _, hid)| hedge_fm_gain[hid as usize])
                                .collect();
                            // DESC order: scores[n/4]=Q3 (75th pct), scores[3*n/4]=Q1 (25th pct).
                            let q3 = scores[n / 4];
                            let q1 = scores[3 * n / 4];
                            let iqr = q3.saturating_sub(q1);
                            let threshold = (q3 as f64 + iqr as f64 * vigw_iqr_mult).round() as i32;
                            let adaptive: Vec<i32> = cut_hedges.iter()
                                .zip(scores.iter())
                                .filter(|(_, &s)| s > 0 && s >= threshold)
                                .map(|(t, _)| t.2)
                                .collect();
                            if adaptive.is_empty() {
                                cut_hedges.into_iter().take(vigw_k).map(|t| t.2).collect()
                            } else {
                                adaptive
                            }
                        } else if vigw_score_mode == 2 {
                        
                        // Selects hedges BOTH structurally cut AND high FM-interaction (true VIGw 19d92cad).
                        cut_hedges.sort_unstable_by(|a, b| {
                            let sa = (a.0 as i64) * (a.1 as i64) * (hedge_fm_gain[a.2 as usize].max(0) as i64);
                            let sb = (b.0 as i64) * (b.1 as i64) * (hedge_fm_gain[b.2 as usize].max(0) as i64);
                            sb.cmp(&sa).then_with(|| a.2.cmp(&b.2))
                        });
                        let n = cut_hedges.len();
                        let scores: Vec<i32> = cut_hedges.iter()
                            .map(|&(pp1, sz, hid)| {
                                let p = (pp1 as i64) * (sz as i64) * (hedge_fm_gain[hid as usize].max(0) as i64);
                                p.min(i32::MAX as i64) as i32
                            })
                            .collect();
                        let q3 = scores[n / 4];
                        let q1 = scores[3 * n / 4];
                        let iqr = q3.saturating_sub(q1);
                        let threshold = (q3 as f64 + iqr as f64 * vigw_iqr_mult).round() as i32;
                        let adaptive: Vec<i32> = cut_hedges.iter()
                            .zip(scores.iter())
                            .filter(|(_, &s)| s > 0 && s >= threshold)
                            .map(|(t, _)| t.2)
                            .collect();
                        if adaptive.is_empty() {
                            cut_hedges.into_iter().take(vigw_k).map(|t| t.2).collect()
                        } else {
                            adaptive
                        }
                        } else {
                        // Adaptive box-plot β: interaction score = (pp-1)*size, DESC sorted.
                        // Q3+1.5*IQR threshold isolates top 1-5% strongest interactions (2dd2736e).
                        // Parameterless: no fixed K (2ed72a15).
                        let scores: Vec<i32> = cut_hedges.iter()
                            .map(|&(pp1, sz, _)| pp1.saturating_mul(sz))
                            .collect();
                        let n = scores.len();
                        // DESC order: scores[n/4]=Q3 (75th pct), scores[3*n/4]=Q1 (25th pct).
                        let q3 = scores[n / 4];
                        let q1 = scores[3 * n / 4];
                        let iqr = q3.saturating_sub(q1);
                        // Q3 + vigw_iqr_mult*IQR (default 1.5 = Q3+1.5*IQR, sweep {1.0,2.0,2.5})
                        let threshold = (q3 as f64 + iqr as f64 * vigw_iqr_mult).round() as i32;
                        let adaptive: Vec<i32> = cut_hedges.iter()
                            .zip(scores.iter())
                            .filter(|(_, &s)| s >= threshold)
                            .map(|(t, _)| t.2)
                            .collect();
                        if adaptive.is_empty() {
                            // Fallback: top vigw_k (same as mode 1)
                            cut_hedges.into_iter().take(vigw_k).map(|t| t.2).collect()
                        } else {
                            adaptive
                        }
                        }
                    } else {
                        // Fixed K with stochastic subsampling (modes 1/2, i69 parité).
                        let pool_n = std::cmp::min(
                            vigw_k.saturating_mul(2),
                            cut_hedges.len(),
                        );
                        let take_k = vigw_k.min(cut_hedges.len());
                        if pool_n <= take_k {
                            cut_hedges.into_iter().take(take_k).map(|t| t.2).collect()
                        } else {
                            let mut pool: Vec<i32> =
                                cut_hedges[..pool_n].iter().map(|t| t.2).collect();
                            let mut rng = seed
                                .wrapping_mul(0x9e3779b97f4a7c15u64)
                                .wrapping_add(0x6c62272e07bb0142u64);
                            for i in 0..take_k {
                                rng = rng
                                    .wrapping_mul(6364136223846793005u64)
                                    .wrapping_add(1442695040888963407u64);
                                let j = i + (rng % (pool_n - i) as u64) as usize;
                                pool.swap(i, j);
                            }
                            pool.truncate(take_k);
                            pool
                        }
                    }
                    } // close vigw_div_mode else
                };
                // vigw_couple_mode=1: Algorithm 3 VIGwbP — extend vigw_ids with anchor's coupled hedges.
                let perturb_ids: Vec<i32> = if vigw_couple_mode == 1 && vigw_ids.len() >= 2 {
                    let in_vigw: std::collections::HashSet<i32> = vigw_ids.iter().cloned().collect();
                    let npc = num_parts_usize.min(64);
                    let mut ext: Vec<i32> = vigw_ids.clone();
                    let mut added: std::collections::HashSet<i32> = vigw_ids.iter().cloned().collect();
                    let mut rng_c = seed.wrapping_mul(0x517cc1b727220a95u64).wrapping_add(ils_iter as u64 * 0x3c6ef35fu64);
                    for &hid in &vigw_ids {
                        let h = hid as usize;
                        let hs = hedge_offsets_host[h] as usize;
                        let he = hedge_offsets_host[h + 1] as usize;
                        let mut pcc = [0i32; 64];
                        for k in hs..he {
                            let nd = hyperedge_nodes_host[k] as usize;
                            let p = partition_host_refine[nd] as usize;
                            if p < npc { pcc[p] += 1; }
                        }
                        let maj = (0..npc).max_by_key(|&p| pcc[p]).unwrap_or(0);
                        let mut anchor = usize::MAX;
                        for k in hs..he {
                            let nd = hyperedge_nodes_host[k] as usize;
                            if (partition_host_refine[nd] as usize) != maj { anchor = nd; break; }
                        }
                        if anchor == usize::MAX { continue; }
                        let as_ = node_offsets_host[anchor] as usize;
                        let ae = node_offsets_host[anchor + 1] as usize;
                        let mut found = false;
                        for ni in as_..ae {
                            let nh = node_hedges_host[ni] as i32;
                            if in_vigw.contains(&nh) && !added.contains(&nh) {
                                ext.push(nh);
                                added.insert(nh);
                                found = true;
                            }
                        }
                        // Guardrail b0377bd1: no coupled neighbor → add 1 random adjacent cut-hedge (≥2 bits).
                        if !found && ae > as_ {
                            rng_c = rng_c.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
                            let pick = as_ + (rng_c % (ae - as_) as u64) as usize;
                            let nh = node_hedges_host[pick] as i32;
                            let nhh = nh as usize;
                            if !added.contains(&nh) && nhh < challenge.num_hyperedges as usize {
                                ext.push(nh);
                                added.insert(nh);
                            }
                        }
                    }
                    ext
                } else {
                    vigw_ids
                };
                perturb_guided_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &perturb_ids,
                );
            } else if ils_iter % 2 == 0 {
                // mode 0 even rounds: static high-connectivity target set (i63 baseline)
                perturb_guided_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &high_hedge_ids_host,
                );
            } else {
                // mode 0 odd rounds: random perturbation
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

    
    // floor_conn_gpu = min(pre_conn, post_conn) after monotone guard; reused in PR block (no new kernel launch).
    let pr_gpu_passconn_early: bool = hyperparameters
        .as_ref()
        .and_then(|p| p.get("pr_gpu_passconn").and_then(|v| v.as_bool()))
        .unwrap_or(false);
    let mut floor_conn_gpu: Option<i32> = None;

    
    if seq_sweeps > 0 {
        let p = num_parts_usize;
        let n_hedges = challenge.num_hyperedges as usize;
        let nn = challenge.num_nodes as usize;
        let maxps = challenge.max_part_size as i32;

        // snapshot current partition and measure pre-connectivity
        let saved_part = stream.memcpy_dtov(&d_partition)?;
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
        let pre_conn: i32 = stream.memcpy_dtov(&d_connectivity)?.iter().sum();

        // build pin-count cache and per-part node counts from current partition
        let mut seq_part = saved_part.clone();
        let mut pc = vec![0i32; n_hedges * p];
        let mut nip = vec![0i32; p];
        for node in 0..nn {
            let pt = seq_part[node];
            if pt >= 0 && (pt as usize) < p {
                nip[pt as usize] += 1;
                let s = node_offsets_host[node] as usize;
                let e = node_offsets_host[node + 1] as usize;
                for k in s..e {
                    let h = node_hedges_host[k] as usize;
                    pc[h * p + pt as usize] += 1;
                }
            }
        }

        for _sweep in 0..seq_sweeps {
            // upload current state, recompute GPU move priorities
            stream.memcpy_htod(&seq_part, &mut d_partition)?;
            stream.memcpy_htod(&nip, &mut d_nodes_in_part)?;
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

            // collect positive-gain candidates, sort by gain desc (node asc tie-break)
            let mut cands: Vec<(i32, usize, usize)> = Vec::new();
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key as u32 != 0x80000000 {
                    let gain = key >> 16;
                    if gain > 0 {
                        cands.push((gain, node, (key & 63) as usize));
                    }
                }
            }
            if cands.is_empty() { break; }
            cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut applied = 0usize;
            for &(_g, node, b) in cands.iter() {
                let a = seq_part[node];
                if a < 0 { continue; }
                let a = a as usize;
                if a == b || b >= p { continue; }
                if nip[a] <= 1 || nip[b] >= maxps { continue; }
                // exact sequential gain via pin-count cache (sees earlier-applied moves)
                let s = node_offsets_host[node] as usize;
                let e = node_offsets_host[node + 1] as usize;
                let mut exact_gain = 0i32;
                for k in s..e {
                    let h = node_hedges_host[k] as usize;
                    if pc[h * p + a] == 1 { exact_gain += 1; }  // removing sole a-pin: conn(h) drops
                    if pc[h * p + b] == 0 { exact_gain -= 1; }  // adding first b-pin: conn(h) rises
                }
                if exact_gain > 0 {
                    for k in s..e {
                        let h = node_hedges_host[k] as usize;
                        pc[h * p + a] -= 1;
                        pc[h * p + b] += 1;
                    }
                    seq_part[node] = b as i32;
                    nip[a] -= 1;
                    nip[b] += 1;
                    applied += 1;
                }
            }
            if applied == 0 { break; }
        }

        // upload final seq state and measure post-connectivity
        stream.memcpy_htod(&seq_part, &mut d_partition)?;
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
        let post_conn: i32 = stream.memcpy_dtov(&d_connectivity)?.iter().sum();
        // monotone guard: revert to pre-seq-sweeps state if no improvement
        if post_conn >= pre_conn {
            stream.memcpy_htod(&saved_part, &mut d_partition)?;
        }
        
        // floor = post_conn if improved, else pre_conn (connectivity of the reverted saved_part).
        if pr_gpu_passconn_early {
            floor_conn_gpu = Some(if post_conn < pre_conn { post_conn } else { pre_conn });
        }
        let _ = nn; // suppress unused warning
    }

    
    
    
    
    
    //   pr_trunc_frac=1.0 (default) ≡ pmc=1 bit-id. sweep {0.5} → target ~824s ≤ 825130ms.
    
    //   also run a backward sub-pass (elite→global_best direction, trunc_frac fraction), exchanging
    //   S/T roles each cycle. pr_mixed=false → bit-identical to i48 (block skipped).
    {
        let pr_max_cycles: usize = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_max_cycles").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(0, 12) as usize)
            .unwrap_or(0);

        let pr_trunc_frac: f64 = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_trunc_frac").and_then(|v| v.as_f64()))
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(1.0);

        let pr_mixed: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_mixed").and_then(|v| v.as_bool()))
            .unwrap_or(false);

        
        // (INCLUDING the moving node v). The per-move delta re-scan of pins
        // (`for kp in es..ee`) is O(disagree x deg x pins) and is the +150s fixed
        // cost of PR-plein (i54 proved walk-truncation does NOT touch it). Replace
        // each inner pin loop by the algebraic identity (phi INCLUDES v, so cnt_c
        // excludes v => cnt_c = phi[p_curr]-1):
        //   cnt_c==0 (pins of p_curr excluding v)  <=>  phi[h][p_curr]==1
        //   cnt_t==0 (pins of p_tgt, v not in it)   <=>  phi[h][p_tgt]==0
        // -> delta = sum_{h in nd(v)} ((phi[p_curr]==1 ? -1:0) + (phi[p_tgt]==0 ? +1:0)),
        // O(deg(v)) with no pin loop. phi is rebuilt from the scanned partition at the
        // start of each pass / reverse guide, then maintained incrementally on each
        // accepted move (phi[p_curr]-=1; phi[p_tgt]+=1 for h in nd(v)). Q is
        
        let pr_gain_cache: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_gain_cache").and_then(|v| v.as_bool()))
            .unwrap_or(false);
        // i56 (Pivot-2a): hoist the O(total_pins) `refill_phi` rebuild out of the
        // per-pass (L2728) and per-reverse-guide (L3009) loops. i55 proved Q is
        // bit-identical (262684) but elapsed unchanged (904s) => the per-pass
        // full rebuild may be eating the O(1)-lookup gain. Instead of rebuilding
        // phi from scratch at every pass, we keep a shadow `phi_part` = the
        // partition phi currently reflects, and reconcile phi to any target
        // partition by touching ONLY the differing nodes (`sync_phi`, O(diff x deg)).
        // This is self-healing (no undo-list / rollback needed): phi_part always
        // tracks phi's true state, so a single diff-sync brings phi to the needed
        // partition regardless of the walk history (accepted-then-reverted moves,
        // best-prefix adoption, or a fresh reverse guide). Q stays bit-identical
        // because sync_phi produces the EXACT same phi as refill_phi would (proven
        // by induction on the tracked partition). pr_refill_hoist=false (default)
        
        // gain table between passes, avoid recompute-from-scratch), 8250371d
        // (incremental delta-cache: recompute only for neighbors of moved nodes).
        let pr_refill_hoist: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_refill_hoist").and_then(|v| v.as_bool()))
            .unwrap_or(false);

        
        // LOOP-DETECT fired: i54 (walk-trunc), i55 (pin-recompute), i56 (refill-rebuild)
        // = 3 guesses, all Q-bit-id 262684 but elapsed unchanged (~906s). No 4th guess.
        // `pr_abl` is a bitmask that SKIPS one PR cost-center at a time so that the
        // ONLY signal read is the delta elapsed_s (Q goes junk — the partition falls
        // back to the valid post-seq_sweeps floor via the balance guard, so nonces
        // stay valid). Delta(full - skip_k) = the wall-cost of block k.
        //   bit0 (1)  ABL_PASSCONN : the O(nh*deg) full-connectivity recomputes never
        //                            ablated by i54/i55/i56 — floor_conn_all (L2687),
        //                            pass_conn (L2736), rev_conn (L3037). PRIME suspect.
        //   bit1 (2)  ABL_EVAL     : disagree_sorted/disagree_rev delta-eval inner loop.
        //   bit2 (4)  ABL_SORT     : the two disagree sort_unstable calls.
        //   bit3 (8)  ABL_WALK     : the fwd + rev walk-apply loops.
        //   bit4 (16) ABL_REVERSE  : the entire reverse-PR guide loop.
        let pr_abl: i64 = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_abl").and_then(|v| v.as_i64()))
            .unwrap_or(0);
        let abl_passconn = pr_abl & 1 != 0;
        let abl_eval = pr_abl & 2 != 0;
        let abl_sort = pr_abl & 4 != 0;
        let abl_walk = pr_abl & 8 != 0;
        let abl_reverse = pr_abl & 16 != 0;
        
        // Diagnostic: isolate rebalance cost vs walk cost (BRANCHE D attribution).
        let abl_rebalance = pr_abl & 32 != 0;
        
        // base_delta = Σ_h[cnt_s(h)==0 ? -1:0] (invariant across all dst_p for fixed v and src_p).
        // present[dst_p] = # hedges incident to v where dst_p has >=1 pin (excl v).
        // delta(dst_p) = base_delta + (deg_v - present[dst_p]). Arithmetic-equivalent => Q BIT-EXACT.
        // Complexity: O(deg×edge_size + deg×np_pr) vs O(np_pr×deg×edge_size) => ~np_pr× speedup on rebalance.
        // pr_rebal_hoist=false (default) => original code, bit-identical to i62 baseline.
        let pr_rebal_hoist: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_rebal_hoist").and_then(|v| v.as_bool()))
            .unwrap_or(false);
        
        // When ON, cycling iterates up to 9 guides (all available elites) with GPU scoring g3+.
        // pr_gpu_interleave=false (default) → exact i95 behavior (bit-identical control).
        let pr_gpu_interleave: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_gpu_interleave").and_then(|v| v.as_bool()))
            .unwrap_or(false);

        
        // Valid because each forward pass restarts from partition_host_refine (monotone guard).
        let pr_incr_passconn: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_incr_passconn").and_then(|v| v.as_bool()))
            .unwrap_or(false);
        
        // (1) compute floor_conn_all via compute_connectivity_200k on d_partition (~1ms vs ~55s CPU)
        // (2) per-pass carry pass_conn = global_best_conn O(1) vs O(nh*deg) rescan per pass.
        // Invariant: connectivity(cur_part) == global_best_conn at start of each pass (exact).
        
        let pr_gpu_passconn: bool = hyperparameters
            .as_ref()
            .and_then(|p| p.get("pr_gpu_passconn").and_then(|v| v.as_bool()))
            .unwrap_or(false);

        if pr_max_cycles > 0 && elite_count >= 2 {
            // Sync device → host so PR sees post-seq_sweeps best partition.
            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

            let n_pr = challenge.num_nodes as usize;
            let nh_pr = challenge.num_hyperedges as usize;
            let np_pr = num_parts_usize;
            let max_ps_pr = challenge.max_part_size as i32;
            // Slack 3% on forward passes (restored to strict by weak rebalancer).
            let slack_max_pr = max_ps_pr + (max_ps_pr * 3 + 99) / 100;

            let mut elite_order_pr: Vec<usize> = (0..elite_count).collect();
            elite_order_pr.sort_unstable_by(|&a, &b| {
                elite_scores[a].cmp(&elite_scores[b]).then(a.cmp(&b))
            });
            
            // pr_n_guides=2 (default) ≡ i37 bit-identique (n_guides_pr=2 hardcoded).
            
            let n_guides_pr: usize = {
                let req = hyperparameters
                    .as_ref()
                    .and_then(|p| p.get("pr_n_guides").and_then(|v| v.as_i64()))
                    .map(|v| v.clamp(2, 13) as usize)
                    .unwrap_or(2);
                req.min(elite_count.saturating_sub(1)).max(2)
            };

            // Build node→hyperedge CSR (one-time, reused across all passes).
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
                        if nd < n_pr { nd_hedges_pr[cur_off[nd]] = h; cur_off[nd] += 1; }
                    }
                }
            }

            
            // rebuilt in place (fill 0 then accumulate) from a partition slice at the
            // start of every pass / reverse guide before it is read, then maintained
            // incrementally through the walk. Empty when pr_gain_cache=false.
            let mut phi: Vec<i32> = if pr_gain_cache { vec![0i32; nh_pr * np_pr] } else { Vec::new() };
            let refill_phi = |phi: &mut Vec<i32>, part: &[i32]| {
                for x in phi.iter_mut() { *x = 0; }
                for h in 0..nh_pr {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    let base = h * np_pr;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        let p = part[nd];
                        if p >= 0 && (p as usize) < np_pr { phi[base + p as usize] += 1; }
                    }
                }
            };
            
            // is tracked in `phi_part`) to `target` by touching ONLY nodes whose
            // part changed: remove the node's pins from its old part, add to the
            // new part, across each hyperedge it belongs to (node->hedge CSR).
            // Produces the EXACT same phi as `refill_phi(phi, target)` would, but
            // in O(#changed x deg) instead of O(total_pins). phi_part is left equal
            // to target on return. Only used when pr_refill_hoist=true.
            let sync_phi = |phi: &mut Vec<i32>, phi_part: &mut Vec<i32>, target: &[i32]| {
                for v in 0..n_pr {
                    let old = phi_part[v];
                    let new = target[v];
                    if old == new { continue; }
                    for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                        let base = nd_hedges_pr[hi] * np_pr;
                        if old >= 0 && (old as usize) < np_pr { phi[base + old as usize] -= 1; }
                        if new >= 0 && (new as usize) < np_pr { phi[base + new as usize] += 1; }
                    }
                    phi_part[v] = new;
                }
            };

            // Floor connectivity = post-seq_sweeps partition (monotone guard).
            
            // i59 launched a NEW kernel here after ~50s GPU-idle (phi/nd_hedges setup) → wakeup +70s.
            // floor_conn_gpu = min(pre_conn, post_conn) set at end of seq_sweeps, exact same value.
            let floor_conn_all = if abl_passconn { 0 }
                else if pr_gpu_passconn && floor_conn_gpu.is_some() {
                    floor_conn_gpu.unwrap()
                } else {
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

            let mut global_best_conn = floor_conn_all;
            let mut global_best_part = partition_host_refine.clone();
            let mut global_best_nip = nodes_in_part_host.clone();
            let mut cur_part = partition_host_refine.clone();
            let mut cur_nip = nodes_in_part_host.clone();

            
            // `sync_phi` maintains it incrementally (no per-pass full rebuild).
            // `phi_part` shadows the partition phi currently reflects. Empty unless
            // pr_gain_cache && pr_refill_hoist.
            let mut phi_part: Vec<i32> = Vec::new();
            if pr_gain_cache && pr_refill_hoist {
                refill_phi(&mut phi, &cur_part);
                phi_part = cur_part.clone();
            }

            if !pr_gpu_interleave {
            // Forward round-robin cycles: 2 guides CPU pure.
            'cycles: for _cycle in 0..pr_max_cycles {
                let mut cycle_accepted = 0usize;

                for pass_idx in 0..n_guides_pr {
                    let guide_slot = pass_idx + 1;
                    if elite_order_pr.len() <= guide_slot { break; }
                    let guide_off = elite_order_pr[guide_slot] * n_pr;

                    let mut pass_conn = if abl_passconn { 0 }
                        else if pr_gpu_passconn { global_best_conn }  // O(1) exact carry
                        else if pr_incr_passconn { floor_conn_all }   // i58 (not bit-exact)
                        else {
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

                    
                    // cur_part == pass_part at the walk's start, so phi is valid for
                    // Site A (build below) and Site B (walk) until maintained. hoist=0
                    // rebuilds from scratch (i55); hoist=1 diff-syncs from phi_part's
                    // current state (self-heals the previous pass's leftover moves).
                    if pr_gain_cache {
                        if pr_refill_hoist { sync_phi(&mut phi, &mut phi_part, &cur_part); }
                        else { refill_phi(&mut phi, &cur_part); }
                    }

                    let mut disagree_sorted: Vec<(i32, usize, i32)> = Vec::with_capacity(disagree_set.len());
                    for &v in &disagree_set {
                        let p_curr = cur_part[v] as usize;
                        let p_tgt_i32 = elite_flat_host[guide_off + v];
                        let p_tgt = p_tgt_i32 as usize;
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                        let mut delta = 0i32;
                        if !abl_eval {
                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                            let h = nd_hedges_pr[hi];
                            if pr_gain_cache {
                                let base = h * np_pr;
                                if phi[base + p_curr] == 1 { delta -= 1; }
                                if phi[base + p_tgt] == 0 { delta += 1; }
                            } else {
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
                        }
                        }
                        disagree_sorted.push((delta, v, p_tgt_i32));
                    }
                    if !abl_sort { disagree_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))); }
                    
                    let trunc_fwd = ((pr_trunc_frac * disagree_sorted.len() as f64).ceil() as usize)
                        .min(disagree_sorted.len());
                    let trunc_fwd = if abl_walk { 0 } else { trunc_fwd };

                    let mut pass_part = cur_part.clone();
                    let mut pass_nip = cur_nip.clone();
                    let mut pass_accepted = 0usize;
                    for &(_, v, p_tgt_i32) in &disagree_sorted[..trunc_fwd] {
                        let p_curr = pass_part[v] as usize;
                        let p_tgt = p_tgt_i32 as usize;
                        if p_curr == p_tgt { continue; }
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                        if pass_nip[p_curr] <= 1 { continue; }
                        if pass_nip[p_tgt] >= slack_max_pr { continue; }

                        let mut delta = 0i32;
                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                            let h = nd_hedges_pr[hi];
                            if pr_gain_cache {
                                let base = h * np_pr;
                                if phi[base + p_curr] == 1 { delta -= 1; }
                                if phi[base + p_tgt] == 0 { delta += 1; }
                            } else {
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
                        }

                        if delta > 0 { continue; }

                        pass_nip[p_curr] -= 1;
                        pass_nip[p_tgt] += 1;
                        pass_part[v] = p_tgt_i32;
                        pass_conn += delta;
                        pass_accepted += 1;

                        
                        if pr_gain_cache {
                            for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                                let base = nd_hedges_pr[hi] * np_pr;
                                phi[base + p_curr] -= 1;
                                phi[base + p_tgt] += 1;
                            }
                            
                            // sync_phi can reconcile from here in O(diff).
                            if pr_refill_hoist { phi_part[v] = p_tgt_i32; }
                        }

                        if pass_conn < best_pass_conn {
                            best_pass_conn = pass_conn;
                            best_pass_part.copy_from_slice(&pass_part);
                            best_pass_nip.copy_from_slice(&pass_nip);
                        }
                    }

                    cycle_accepted += pass_accepted;

                    if best_pass_conn < global_best_conn {
                        global_best_conn = best_pass_conn;
                        global_best_part = best_pass_part.clone();
                        global_best_nip = best_pass_nip.clone();
                    }

                    if best_pass_conn < floor_pass {
                        cur_part = best_pass_part;
                        cur_nip = best_pass_nip;
                    }
                }

                
                // Each cycle: after forward round-robin, explore T-end (elite→global_best)
                
                // pr_mixed=false → this block is skipped, bit-identical to i48.
                if pr_mixed {
                    for rev_guide_slot_m in 1..=n_guides_pr {
                        if elite_order_pr.len() <= rev_guide_slot_m { break; }
                        let rev_guide_off_m = elite_order_pr[rev_guide_slot_m] * n_pr;

                        let mut rev_part_m: Vec<i32> = elite_flat_host[rev_guide_off_m..rev_guide_off_m + n_pr].to_vec();
                        let mut rev_nip_m = vec![0i32; np_pr];
                        for &p in rev_part_m.iter() {
                            if p >= 0 && (p as usize) < np_pr { rev_nip_m[p as usize] += 1; }
                        }

                        let mut rev_conn_m = {
                            let mut pseen = vec![false; np_pr];
                            let mut c = 0i32;
                            for h in 0..nh_pr {
                                let hs = hedge_offsets_host[h] as usize;
                                let he = hedge_offsets_host[h + 1] as usize;
                                let mut d = 0usize;
                                for k in hs..he {
                                    let nd = hyperedge_nodes_host[k] as usize;
                                    let p = rev_part_m[nd];
                                    if p >= 0 && (p as usize) < np_pr {
                                        if !pseen[p as usize] { pseen[p as usize] = true; d += 1; }
                                    }
                                }
                                if d > 1 { c += (d - 1) as i32; }
                                for k in hs..he {
                                    let p = rev_part_m[hyperedge_nodes_host[k] as usize];
                                    if p >= 0 && (p as usize) < np_pr { pseen[p as usize] = false; }
                                }
                            }
                            c
                        };

                        let mut best_rev_conn_m = rev_conn_m;
                        let mut best_rev_part_m = rev_part_m.clone();
                        let mut best_rev_nip_m = rev_nip_m.clone();

                        let mut disagree_rev_m: Vec<(i32, usize, i32)> = Vec::new();
                        for v in 0..n_pr {
                            let p_curr = rev_part_m[v];
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
                                    let p = rev_part_m[nd];
                                    if p == p_curr { cnt_c += 1; }
                                    else if p == p_tgt { cnt_t += 1; }
                                }
                                if cnt_c == 0 { delta -= 1; }
                                if cnt_t == 0 { delta += 1; }
                            }
                            disagree_rev_m.push((delta, v, p_tgt));
                        }
                        disagree_rev_m.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                        let trunc_rev_m = ((pr_trunc_frac * disagree_rev_m.len() as f64).ceil() as usize)
                            .min(disagree_rev_m.len());

                        let mut mixed_accepted = 0usize;
                        for &(_, v, p_tgt_i32) in &disagree_rev_m[..trunc_rev_m] {
                            let p_curr_i32 = rev_part_m[v];
                            if p_curr_i32 == p_tgt_i32 { continue; }
                            let p_curr = p_curr_i32 as usize;
                            let p_tgt = p_tgt_i32 as usize;
                            if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                            if rev_nip_m[p_curr] <= 1 { continue; }
                            if rev_nip_m[p_tgt] >= max_ps_pr { continue; }

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
                                    let p = rev_part_m[nd];
                                    if p == p_curr_i32 { cnt_c += 1; }
                                    else if p == p_tgt_i32 { cnt_t += 1; }
                                }
                                if cnt_c == 0 { delta -= 1; }
                                if cnt_t == 0 { delta += 1; }
                            }

                            if delta > 0 { continue; }

                            rev_nip_m[p_curr] -= 1;
                            rev_nip_m[p_tgt] += 1;
                            rev_part_m[v] = p_tgt_i32;
                            rev_conn_m += delta;
                            mixed_accepted += 1;

                            if rev_conn_m < best_rev_conn_m {
                                best_rev_conn_m = rev_conn_m;
                                best_rev_part_m.copy_from_slice(&rev_part_m);
                                best_rev_nip_m.copy_from_slice(&rev_nip_m);
                            }
                        }

                        cycle_accepted += mixed_accepted;

                        if best_rev_conn_m < global_best_conn {
                            global_best_conn = best_rev_conn_m;
                            global_best_part = best_rev_part_m;
                            global_best_nip = best_rev_nip_m;
                        }
                    }
                }

                if cycle_accepted == 0 { break 'cycles; }
            }

            // ==== REVERSE PATH-RELINK: g[k]→global_best, 2 guides, strict balance ====
            for rev_guide_slot in 1..=n_guides_pr {
                if abl_reverse { break; }
                if elite_order_pr.len() <= rev_guide_slot { break; }
                let rev_guide_off = elite_order_pr[rev_guide_slot] * n_pr;

                let mut rev_part: Vec<i32> = elite_flat_host[rev_guide_off..rev_guide_off + n_pr].to_vec();
                let mut rev_nip = vec![0i32; np_pr];
                for &p in rev_part.iter() {
                    if p >= 0 && (p as usize) < np_pr { rev_nip[p as usize] += 1; }
                }

                
                // Covers the 2 remaining rescans not skipped by pr_gpu_passconn in i60.
                let mut rev_conn = if abl_passconn { 0 }
                    else if pr_gpu_passconn { elite_scores[elite_order_pr[rev_guide_slot]] }
                    else {
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

                
                // is a fresh elite each guide (no incremental relation to the prior
                // partition), but hoist=1 still reconciles via diff-sync from phi_part
                // (touches only the nodes that differ) instead of a full O(pins)
                // rebuild — identical result, cheaper when the elites overlap.
                if pr_gain_cache {
                    if pr_refill_hoist { sync_phi(&mut phi, &mut phi_part, &rev_part); }
                    else { refill_phi(&mut phi, &rev_part); }
                }

                let mut disagree_rev: Vec<(i32, usize, i32)> = Vec::new();
                for v in 0..n_pr {
                    let p_curr = rev_part[v];
                    let p_tgt = global_best_part[v];
                    if p_curr == p_tgt || p_curr < 0 || p_tgt < 0 { continue; }
                    let p_curr_u = p_curr as usize;
                    let p_tgt_u = p_tgt as usize;
                    if p_curr_u >= np_pr || p_tgt_u >= np_pr { continue; }
                    let mut delta = 0i32;
                    if !abl_eval {
                    for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                        let h = nd_hedges_pr[hi];
                        if pr_gain_cache {
                            let base = h * np_pr;
                            if phi[base + p_curr_u] == 1 { delta -= 1; }
                            if phi[base + p_tgt_u] == 0 { delta += 1; }
                        } else {
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
                    }
                    }
                    disagree_rev.push((delta, v, p_tgt));
                }
                if !abl_sort { disagree_rev.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1))); }
                
                let trunc_rev = ((pr_trunc_frac * disagree_rev.len() as f64).ceil() as usize)
                    .min(disagree_rev.len());
                let trunc_rev = if abl_walk { 0 } else { trunc_rev };

                for &(_, v, p_tgt_i32) in &disagree_rev[..trunc_rev] {
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
                        if pr_gain_cache {
                            let base = h * np_pr;
                            if phi[base + p_curr] == 1 { delta -= 1; }
                            if phi[base + p_tgt] == 0 { delta += 1; }
                        } else {
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
                    }

                    if delta > 0 { continue; }

                    rev_nip[p_curr] -= 1;
                    rev_nip[p_tgt] += 1;
                    rev_part[v] = p_tgt_i32;
                    rev_conn += delta;

                    
                    if pr_gain_cache {
                        for hi in nd_off_pr[v]..nd_off_pr[v + 1] {
                            let base = nd_hedges_pr[hi] * np_pr;
                            phi[base + p_curr] -= 1;
                            phi[base + p_tgt] += 1;
                        }
                        
                        if pr_refill_hoist { phi_part[v] = p_tgt_i32; }
                    }

                    if rev_conn < best_rev_conn {
                        best_rev_conn = rev_conn;
                        best_rev_part.copy_from_slice(&rev_part);
                        best_rev_nip.copy_from_slice(&rev_nip);
                    }
                }

                if best_rev_conn < global_best_conn {
                    global_best_conn = best_rev_conn;
                    global_best_part = best_rev_part;
                    global_best_nip = best_rev_nip;
                }
            }
            // ==== END REVERSE PATH-RELINK ====
            } else {
            // ==== GPU-INTERLEAVED PR CYCLING (i96 PORT P_GPU_PR_INTERLEAVED from t25/i53) ====
            // guide_slot 1,2: CPU-pure walk. guide_slot >= 3: GPU-scored via score_pr_g3_warp_200k.
            // D recalculated from cur_part before each guide → D fresh per guide (not exhausted).
            // Enables affordable multi-guide cycling for 200k hedges within 40s time margin.
            'cycles_gpu: for _cycle in 0..pr_max_cycles {
                let mut cycle_accepted = 0usize;

                for pass_idx in 0..9usize {
                    let guide_slot = pass_idx + 1;
                    if elite_order_pr.len() <= guide_slot { break; }
                    let guide_off = elite_order_pr[guide_slot] * n_pr;

                    if guide_slot >= 3 {
                        // GPU-scored path: build D, upload cur_part, launch kernel, greedy commit.
                        let mut gn_nodes: Vec<i32> = Vec::new();
                        let mut gn_tgts:  Vec<i32> = Vec::new();
                        for v in 0..n_pr {
                            let ps = cur_part[v];
                            let pg = elite_flat_host[guide_off + v];
                            if ps != pg && ps >= 0 && pg >= 0
                                && (ps as usize) < np_pr && (pg as usize) < np_pr
                            {
                                gn_nodes.push(v as i32);
                                gn_tgts.push(pg);
                            }
                        }
                        let nd_gn = gn_nodes.len();
                        if nd_gn == 0 { continue; }

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
                        let d_gn_nodes = stream.memcpy_stod(&gn_nodes)?;
                        let d_gn_tgts  = stream.memcpy_stod(&gn_tgts)?;
                        let mut d_gn_delta = stream.alloc_zeros::<i32>(nd_gn)?;
                        let gn_blocks = ((nd_gn as u32) + 3) / 4;
                        let gn_cfg = LaunchConfig { grid_dim: (gn_blocks, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                        unsafe {
                            stream
                                .launch_builder(&score_pr_g3_warp_kernel)
                                .arg(&(nd_gn as i32))
                                .arg(&d_gn_nodes)
                                .arg(&d_gn_tgts)
                                .arg(&challenge.d_node_hyperedges)
                                .arg(&challenge.d_node_offsets)
                                .arg(&d_partition)
                                .arg(&d_edge_flags_all)
                                .arg(&d_edge_flags_double)
                                .arg(&mut d_gn_delta)
                                .launch(gn_cfg)?;
                        }
                        let delta_gn: Vec<i32> = stream.memcpy_dtov(&d_gn_delta)?;
                        let mut gn_sorted: Vec<(i32, usize, i32)> = gn_nodes.iter().enumerate()
                            .map(|(i, &v)| (delta_gn[i], v as usize, gn_tgts[i]))
                            .collect();
                        gn_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                        // Greedy CPU commit: exact Δconn recomputed per accept, sequential.
                        let mut gc_part = cur_part.clone();
                        let mut gc_nip  = cur_nip.clone();
                        let mut gc_conn = {
                            let mut pseen = vec![false; np_pr];
                            let mut c = 0i32;
                            for h in 0..nh_pr {
                                let hs = hedge_offsets_host[h] as usize;
                                let he = hedge_offsets_host[h + 1] as usize;
                                let mut d = 0usize;
                                for k in hs..he {
                                    let nd = hyperedge_nodes_host[k] as usize;
                                    let p = gc_part[nd];
                                    if p >= 0 && (p as usize) < np_pr {
                                        let pu = p as usize;
                                        if !pseen[pu] { pseen[pu] = true; d += 1; }
                                    }
                                }
                                if d > 1 { c += (d - 1) as i32; }
                                for k in hs..he {
                                    let p = gc_part[hyperedge_nodes_host[k] as usize];
                                    if p >= 0 && (p as usize) < np_pr { pseen[p as usize] = false; }
                                }
                            }
                            c
                        };
                        let floor_gc = gc_conn;
                        let mut gc_best_conn = gc_conn;
                        let mut gc_best_part = gc_part.clone();
                        let mut gc_best_nip  = gc_nip.clone();
                        let mut gc_accepted  = 0usize;
                        for &(_, v, p_tgt_i32) in &gn_sorted {
                            let p_curr_i32 = gc_part[v];
                            if p_curr_i32 == p_tgt_i32 { continue; }
                            let p_curr = p_curr_i32 as usize;
                            let p_tgt  = p_tgt_i32 as usize;
                            if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                            if gc_nip[p_curr] <= 1 { continue; }
                            if gc_nip[p_tgt] >= slack_max_pr { continue; }
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
                                    let p = gc_part[nd];
                                    if p == p_curr_i32 { cnt_c += 1; }
                                    else if p == p_tgt_i32 { cnt_t += 1; }
                                }
                                if cnt_c == 0 { delta -= 1; }
                                if cnt_t == 0 { delta += 1; }
                            }
                            if delta > 0 { continue; }
                            gc_nip[p_curr] -= 1;
                            gc_nip[p_tgt]  += 1;
                            gc_part[v] = p_tgt_i32;
                            gc_conn += delta;
                            gc_accepted += 1;
                            if gc_conn < gc_best_conn {
                                gc_best_conn = gc_conn;
                                gc_best_part.copy_from_slice(&gc_part);
                                gc_best_nip.copy_from_slice(&gc_nip);
                            }
                        }
                        cycle_accepted += gc_accepted;
                        if gc_best_conn < global_best_conn {
                            global_best_conn = gc_best_conn;
                            global_best_part = gc_best_part.clone();
                            global_best_nip  = gc_best_nip.clone();
                        }
                        if gc_best_conn < floor_gc {
                            cur_part = gc_best_part;
                            cur_nip  = gc_best_nip;
                        }
                        continue;
                    }

                    // CPU path for guide_slot 1,2 (t25/i53 style, simple delta-conn loop).
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
                    let mut best_pass_nip  = cur_nip.clone();

                    let disagree_set: Vec<usize> = (0..n_pr)
                        .filter(|&v| {
                            let ps = cur_part[v];
                            let pg = elite_flat_host[guide_off + v];
                            ps != pg && ps >= 0 && pg >= 0
                        })
                        .collect();
                    if disagree_set.is_empty() { continue; }

                    let mut disagree_sorted: Vec<(i32, usize, i32)> = Vec::with_capacity(disagree_set.len());
                    for &v in &disagree_set {
                        let p_curr_i32 = cur_part[v];
                        let p_tgt_i32  = elite_flat_host[guide_off + v];
                        let p_curr = p_curr_i32 as usize;
                        let p_tgt  = p_tgt_i32 as usize;
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
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
                                let p = cur_part[nd];
                                if p == p_curr_i32 { cnt_c += 1; }
                                else if p == p_tgt_i32 { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }
                        disagree_sorted.push((delta, v, p_tgt_i32));
                    }
                    disagree_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                    let mut pass_part = cur_part.clone();
                    let mut pass_nip  = cur_nip.clone();
                    let mut pass_accepted = 0usize;
                    for &(_, v, p_tgt_i32) in &disagree_sorted {
                        let p_curr_i32 = pass_part[v];
                        if p_curr_i32 == p_tgt_i32 { continue; }
                        let p_curr = p_curr_i32 as usize;
                        let p_tgt  = p_tgt_i32 as usize;
                        if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                        if pass_nip[p_curr] <= 1 { continue; }
                        if pass_nip[p_tgt] >= slack_max_pr { continue; }
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
                                let p = pass_part[nd];
                                if p == p_curr_i32 { cnt_c += 1; }
                                else if p == p_tgt_i32 { cnt_t += 1; }
                            }
                            if cnt_c == 0 { delta -= 1; }
                            if cnt_t == 0 { delta += 1; }
                        }
                        if delta > 0 { continue; }
                        pass_nip[p_curr] -= 1;
                        pass_nip[p_tgt]  += 1;
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
                    if best_pass_conn < global_best_conn {
                        global_best_conn = best_pass_conn;
                        global_best_part = best_pass_part.clone();
                        global_best_nip  = best_pass_nip.clone();
                    }
                    if best_pass_conn < floor_pass {
                        cur_part = best_pass_part;
                        cur_nip  = best_pass_nip;
                    }
                }

                if cycle_accepted == 0 { break 'cycles_gpu; }
            }

            // ==== REVERSE PATH-RELINK (GPU branch): g[k]→global_best, n_guides_pr passes, strict balance ====
            for rev_guide_slot in 1..=n_guides_pr {
                if elite_order_pr.len() <= rev_guide_slot { break; }
                let rev_guide_off = elite_order_pr[rev_guide_slot] * n_pr;

                let mut rev_part: Vec<i32> = elite_flat_host[rev_guide_off..rev_guide_off + n_pr].to_vec();
                let mut rev_nip = vec![0i32; np_pr];
                for &p in rev_part.iter() {
                    if p >= 0 && (p as usize) < np_pr { rev_nip[p as usize] += 1; }
                }
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
                let mut best_rev_nip  = rev_nip.clone();

                let mut disagree_rev: Vec<(i32, usize, i32)> = Vec::new();
                for v in 0..n_pr {
                    let p_curr = rev_part[v];
                    let p_tgt  = global_best_part[v];
                    if p_curr == p_tgt || p_curr < 0 || p_tgt < 0 { continue; }
                    let p_curr_u = p_curr as usize;
                    let p_tgt_u  = p_tgt as usize;
                    if p_curr_u >= np_pr || p_tgt_u >= np_pr { continue; }
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

                for &(_, v, p_tgt_i32) in &disagree_rev {
                    let p_curr_i32 = rev_part[v];
                    if p_curr_i32 == p_tgt_i32 { continue; }
                    let p_curr = p_curr_i32 as usize;
                    let p_tgt  = p_tgt_i32 as usize;
                    if p_curr >= np_pr || p_tgt >= np_pr { continue; }
                    if rev_nip[p_curr] <= 1 { continue; }
                    if rev_nip[p_tgt] >= max_ps_pr { continue; }
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
                            let p = rev_part[nd];
                            if p == p_curr_i32 { cnt_c += 1; }
                            else if p == p_tgt_i32 { cnt_t += 1; }
                        }
                        if cnt_c == 0 { delta -= 1; }
                        if cnt_t == 0 { delta += 1; }
                    }
                    if delta > 0 { continue; }
                    rev_nip[p_curr] -= 1;
                    rev_nip[p_tgt]  += 1;
                    rev_part[v] = p_tgt_i32;
                    rev_conn += delta;
                    if rev_conn < best_rev_conn {
                        best_rev_conn = rev_conn;
                        best_rev_part.copy_from_slice(&rev_part);
                        best_rev_nip.copy_from_slice(&rev_nip);
                    }
                }
                if best_rev_conn < global_best_conn {
                    global_best_conn = best_rev_conn;
                    global_best_part = best_rev_part;
                    global_best_nip  = best_rev_nip;
                }
            }
            // ==== END GPU-INTERLEAVED PR CYCLING + REVERSE ====
            } // end if/else pr_gpu_interleave

            // ==== B_UNCONSTRAINED_REBALANCE WEAK — restore strict balance ====
            
            if !abl_rebalance && global_best_conn < floor_conn_all {
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
                            let mut cands: Vec<(i32, usize, usize)> = Vec::new();
                            for &v in &block_verts[src_p] {
                                let (best_delta, best_dst) = if pr_rebal_hoist {
                                    // Hoisted: scan each incident hedge ONCE.
                                    // base_delta accumulates cnt_s-dependent terms (dst_p-invariant).
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
                                    // Original: rescan pins for every dst_p (bit-identical to i62 baseline).
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
                                if best_dst < np_pr { cands.push((best_delta, v, best_dst)); }
                            }
                            cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                            for &(_, v, dst_p) in &cands {
                                if global_best_nip[src_p] <= max_ps_pr { break; }
                                if global_best_nip[dst_p] >= max_ps_pr { continue; }
                                global_best_part[v] = dst_p as i32;
                                global_best_nip[src_p] -= 1;
                                global_best_nip[dst_p] += 1;
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
                // else: balance not restored → keep floor (partition_host_refine unchanged, 0 invalid).
            }
            // ==== END B_PATH_RELINK + REBALANCE ====
        }
    }
    // ==== END i35 P_PATH_RELINK_CPU2G ====

    // ==== B_TERMINAL_REFINE_CHAIN i81 (PORT P_TERMINAL_REFINE_CHAIN t25/i34, stacked with vigw_k+lb3) ====
    
    // mode>=2: + B_BALANCED_SWAP_PAIR (balance-neutral cross-block pair swaps).
    // In t24, this runs AFTER PATH_RELINK; d_partition is already the post-PR best state.
    
    if term_refine_mode >= 1 {
        // ==== B_BATCH_FM_GPU ====
        do_swap_phase!(
            &mut d_partition, &mut d_nodes_in_part,
            &mut d_edge_flags_all, &mut d_edge_flags_double,
            &mut d_swap_gains, &mut swap_gains_host,
            &mut partition_host_swap, &mut partition_mut_swap,
            &mut part_to_part, &mut used_ba_buf,
            5, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
        )?;
        partition_host_refine.copy_from_slice(&partition_host_swap);
        // ==== END B_BATCH_FM_GPU ====

        if term_refine_mode >= 2 {
            // ==== B_BALANCED_SWAP_PAIR (verbatim t25/i34 L4429-4612) ====
            {
                let n_sp = challenge.num_nodes as usize;
                let m_sp = challenge.num_hyperedges as usize;
                let np_sp = num_parts_usize;
                let max_ps_sp = challenge.max_part_size as i32;

                if np_sp >= 2 {
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
                            if hpc[h * np + p] == 1 { g += 1; }
                            if hpc[h * np + t] == 0 { g -= 1; }
                            if hpc[h * np + p] > 0 { hpc[h * np + p] -= 1; }
                            hpc[h * np + t] = hpc[h * np + t].saturating_add(1);
                        }
                        sp_part[v] = t as i32;
                        g
                    }

                    let mut sp_part = partition_host_refine.clone();
                    let mut current_conn = initial_conn;
                    let mut tgt = vec![0i32; np_sp];
                    let mut best_to: Vec<(i32, usize)> = vec![(i32::MIN, usize::MAX); np_sp * np_sp];

                    for _sp_pass in 0..max_sp_passes {
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

                        let mut improved = false;
                        for a in 0..np_sp {
                            for b in (a + 1)..np_sp {
                                let (gv, v) = best_to[a * np_sp + b];
                                let (gu, u) = best_to[b * np_sp + a];
                                if v == usize::MAX || u == usize::MAX || v == u { continue; }
                                if gv == i32::MIN || gu == i32::MIN { continue; }
                                if gv + gu <= 0 { continue; }
                                if sp_part[v] != a as i32 || sp_part[u] != b as i32 { continue; }
                                let rgv = sp_apply(v, b, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                                let rgu = sp_apply(u, a, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                                let real = rgv + rgu;
                                if real > 0 {
                                    current_conn -= real;
                                    improved = true;
                                } else {
                                    sp_apply(u, b, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                                    sp_apply(v, a, &mut sp_part, &mut hpc, &nd_off, &nd_hedges, np_sp);
                                }
                            }
                        }
                        if !improved { break; }
                    }

                    if current_conn < initial_conn {
                        let mut sp_nip = vec![0i32; np_sp];
                        for &p in &sp_part {
                            if p >= 0 && (p as usize) < np_sp { sp_nip[p as usize] += 1; }
                        }
                        let balance_ok = sp_nip.iter().all(|&c| c <= max_ps_sp);
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
            // ==== END B_BALANCED_SWAP_PAIR ====
        }
    }
    // ==== END B_TERMINAL_REFINE_CHAIN i81 ====

    // ==== B_EJECTION_CHAIN i95 (PIVOT L2/L3 — TERMINAL sequential host-side) ====
    // Compound-move chains: trigger → best part → best node in dest → ...
    // In-sequence gain recompute via hpc (updated after each applied move).
    // Best-prefix subsequence applied (57b695f2: approx-future-state max-cumsum prefix).
    
    if ejchain_max_len > 0 {
        let n_ej = challenge.num_nodes as usize;
        let m_ej = challenge.num_hyperedges as usize;
        let np_ej = num_parts_usize;
        let max_ps_ej = challenge.max_part_size as i32;

        // Build hpc[h * np_ej + p] from partition_host_refine
        let mut ej_hpc = vec![0u16; m_ej * np_ej];
        for h in 0..m_ej {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            for k in hs..he {
                let nd = hyperedge_nodes_host[k] as usize;
                if nd < n_ej {
                    let p = partition_host_refine[nd];
                    if p >= 0 && (p as usize) < np_ej {
                        ej_hpc[h * np_ej + p as usize] =
                            ej_hpc[h * np_ej + p as usize].saturating_add(1);
                    }
                }
            }
        }

        let mut ej_part = partition_host_refine.clone();
        let mut ej_nip = nodes_in_part_host.clone();

        let initial_ej_conn: i32 = {
            let mut c = 0i32;
            for h in 0..m_ej {
                let mut d = 0i32;
                for p in 0..np_ej { if ej_hpc[h * np_ej + p] > 0 { d += 1; } }
                if d > 1 { c += d - 1; }
            }
            c
        };

        // Collect boundary nodes with best single-move gain (in-sequence, initial hpc).
        // gain(v→t) = Σ_h∈nd(v): (hpc[h][src]==1 ? +1:0) + (hpc[h][t]==0 ? -1:0)
        let mut seed_gains: Vec<(i32, usize)> = Vec::new();
        for v in 0..n_ej {
            let p = ej_part[v];
            if p < 0 || (p as usize) >= np_ej { continue; }
            let p_u = p as usize;
            let hs_v = node_offsets_host[v] as usize;
            let he_v = node_offsets_host[v + 1] as usize;
            if hs_v >= he_v { continue; }
            let is_bnd = (hs_v..he_v).any(|hi| {
                let h = node_hedges_host[hi] as usize;
                h < m_ej && (0..np_ej).any(|q| q != p_u && ej_hpc[h * np_ej + q] > 0)
            });
            if !is_bnd { continue; }
            let mut best_g = i32::MIN;
            for t in 0..np_ej {
                if t == p_u || ej_nip[t] >= max_ps_ej { continue; }
                let mut g = 0i32;
                for hi in hs_v..he_v {
                    let h = node_hedges_host[hi] as usize;
                    if h >= m_ej { continue; }
                    if ej_hpc[h * np_ej + p_u] == 1 { g += 1; }
                    if ej_hpc[h * np_ej + t] == 0 { g -= 1; }
                }
                if g > best_g { best_g = g; }
            }
            if best_g > i32::MIN { seed_gains.push((best_g, v)); }
        }
        seed_gains.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        // Process ejection chains: trigger = top seeds by initial gain.
        // For each step: apply move (in-sequence hpc update), track best-prefix cumsum.
        // Next trigger (source-to-dest rule): best node sharing a hedge with cur_v in dest part.
        // Rollback to best_prefix; only commit net improvements.
        let max_triggers = seed_gains.len().min(500);
        for &(_sg, v0) in &seed_gains[..max_triggers] {
            let mut chain: Vec<(usize, usize, usize)> = Vec::with_capacity(ejchain_max_len);
            let mut cumgain = 0i32;
            let mut best_cum = 0i32;
            let mut best_len = 0usize;
            let mut cur_v = v0;
            let mut prev_dest: usize = np_ej; // invalid sentinel

            for step in 0..ejchain_max_len {
                let p_v = ej_part[cur_v];
                if p_v < 0 || (p_v as usize) >= np_ej { break; }
                let p_v_u = p_v as usize;
                // Source-to-dest rule: for step>0, cur_v must be in prev_dest
                if step > 0 && p_v_u != prev_dest { break; }

                let hs_v = node_offsets_host[cur_v] as usize;
                let he_v = node_offsets_host[cur_v + 1] as usize;

                // In-sequence gain: hpc already reflects prior accepted chain moves
                let mut best_g = i32::MIN;
                let mut best_t = np_ej;
                for t in 0..np_ej {
                    if t == p_v_u || ej_nip[t] >= max_ps_ej { continue; }
                    let mut g = 0i32;
                    for hi in hs_v..he_v {
                        let h = node_hedges_host[hi] as usize;
                        if h >= m_ej { continue; }
                        if ej_hpc[h * np_ej + p_v_u] == 1 { g += 1; }
                        if ej_hpc[h * np_ej + t] == 0 { g -= 1; }
                    }
                    if g > best_g { best_g = g; best_t = t; }
                }
                if best_t >= np_ej { break; }

                // Apply move: update hpc, nip, part
                for hi in hs_v..he_v {
                    let h = node_hedges_host[hi] as usize;
                    if h < m_ej {
                        if ej_hpc[h * np_ej + p_v_u] > 0 { ej_hpc[h * np_ej + p_v_u] -= 1; }
                        ej_hpc[h * np_ej + best_t] =
                            ej_hpc[h * np_ej + best_t].saturating_add(1);
                    }
                }
                ej_nip[p_v_u] -= 1;
                ej_nip[best_t] += 1;
                ej_part[cur_v] = best_t as i32;
                chain.push((cur_v, p_v_u, best_t));
                cumgain += best_g;
                if cumgain > best_cum { best_cum = cumgain; best_len = chain.len(); }

                // Find next trigger (source-to-dest): best boundary node in best_t
                // that shares a hedge with cur_v (neighbor-scoped, O(deg × hedge_size × np)).
                let mut nx_best_g = i32::MIN;
                let mut nx_v = n_ej;
                for hi in hs_v..he_v {
                    let h = node_hedges_host[hi] as usize;
                    if h >= m_ej { continue; }
                    let hsh = hedge_offsets_host[h] as usize;
                    let heh = hedge_offsets_host[h + 1] as usize;
                    for k in hsh..heh {
                        let u = hyperedge_nodes_host[k] as usize;
                        if u >= n_ej || u == cur_v { continue; }
                        if ej_part[u] as usize != best_t { continue; }
                        let hsu = node_offsets_host[u] as usize;
                        let heu = node_offsets_host[u + 1] as usize;
                        for t2 in 0..np_ej {
                            if t2 == best_t || ej_nip[t2] >= max_ps_ej { continue; }
                            let mut g2 = 0i32;
                            for hi2 in hsu..heu {
                                let hh = node_hedges_host[hi2] as usize;
                                if hh >= m_ej { continue; }
                                if ej_hpc[hh * np_ej + best_t] == 1 { g2 += 1; }
                                if ej_hpc[hh * np_ej + t2] == 0 { g2 -= 1; }
                            }
                            if g2 > nx_best_g { nx_best_g = g2; nx_v = u; }
                        }
                    }
                }
                prev_dest = best_t;
                if nx_v >= n_ej { break; }
                cur_v = nx_v;
            }

            // Rollback to best_len prefix
            for i in (best_len..chain.len()).rev() {
                let (nd, from_p, to_t) = chain[i];
                let hsn = node_offsets_host[nd] as usize;
                let hen = node_offsets_host[nd + 1] as usize;
                for hi in hsn..hen {
                    let h = node_hedges_host[hi] as usize;
                    if h < m_ej {
                        if ej_hpc[h * np_ej + to_t] > 0 { ej_hpc[h * np_ej + to_t] -= 1; }
                        ej_hpc[h * np_ej + from_p] =
                            ej_hpc[h * np_ej + from_p].saturating_add(1);
                    }
                }
                ej_nip[to_t] -= 1;
                ej_nip[from_p] += 1;
                ej_part[nd] = from_p as i32;
            }
        }

        // Commit if connectivity improved and balance maintained
        let final_ej_conn: i32 = {
            let mut c = 0i32;
            for h in 0..m_ej {
                let mut d = 0i32;
                for p in 0..np_ej { if ej_hpc[h * np_ej + p] > 0 { d += 1; } }
                if d > 1 { c += d - 1; }
            }
            c
        };
        if final_ej_conn < initial_ej_conn && ej_nip.iter().all(|&c| c <= max_ps_ej) {
            partition_host_refine.copy_from_slice(&ej_part);
            nodes_in_part_host.copy_from_slice(&ej_nip);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
    }
    // ==== END B_EJECTION_CHAIN i95 ====

    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    save_solution(&Solution {
        partition: partition_u32,
    })?;
    Ok(())
}
