use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg}, 
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

// ============================================================================
// BEST-OF-K (`runs`) WRAPPER  --  the ONLY change vs the base 10k solver.
// ----------------------------------------------------------------------------
// The construction phase (seeded MinHash hyperedge clustering -> node
// preferences -> greedy priority assignment) is a pure function of the
// challenge and the hyperparameters, so it runs ONCE.
// Everything after it is re-run `runs` times from that same construction
// result, with a per-run offset
//     run_off = bok_run * 0x9E3779B97F4A7C15   (wrapping)
// folded with `wrapping_add` into EVERY seed that can influence the trajectory
// after construction.  Run 0 has run_off == 0, so it reproduces
// the base solver bit for bit.  The run with the smallest TRUE km1 is saved.
//
// On THIS track the four post-construction seed sites are:
//   1. the stagnation mini-perturbation seed  (perturb_solution_10k)
//   2. the ILS perturbation seed `seed`, from which perturb_path_relink /
//      perturb_guided / perturb_hubs / perturb_ruin_recreate derive theirs
//      by XOR with a fixed constant
//   3. the simulated-annealing acceptance RNG in the ILS loop
//   4. the final polish_exploration_10k seed
// The MinHash hyperedge-clustering seeds (hash_a/hash_b, from challenge.seed)
// and `init_restart_id` belong to the CONSTRUCTION and are therefore NOT
// offset: the construction runs once and is shared by every run.
// ============================================================================

/// Upper clamp of the `runs` hyperparameter.  One run costs ~1e10 fuel against
/// a 5e12 budget, so this is a sanity bound, not a fuel bound.
///
/// wave12: raised 64 -> 256.  K_MAX is read in EXACTLY ONE place, the
/// `.clamp(1, K_MAX)` on the `runs` hyperparameter; no buffer, stride, array
/// length or loop bound is derived from it (`sanity.py` [4] asserts the
/// declaration + that one reader are the only two occurrences in the file).
/// Raising it only widens the range of `runs` a hyperparameter set may ask
/// for; it costs nothing at any smaller `runs`, and the real ceiling remains
/// the 5e12 fuel budget, which binds LONG before 256 runs on this track.
const K_MAX: i64 = 256;

/// Odd golden-ratio constant used to mix the run index into every seed.
const RUN_OFF_MUL: u64 = 0x9E3779B97F4A7C15;

/// Odd constant used to derive the per-run RUIN seed (kept distinct from
/// `RUN_OFF_MUL` so the ruin shuffle and the trajectory seeds do not move in
/// lockstep).
const RUIN_SEED_MUL: u64 = 0xD1B54A32D192ED03;
/// Additive salt for the ruin seed; makes run 1's seed differ from the value
/// `improvement007_ruin` used, purely so the two are not accidentally coupled.
const RUIN_SEED_ADD: u64 = 0xCA5A826395121157;

/// Exact connectivity (km1) of a partition, computed on the host:
///
///     sum over hyperedges of (number of DISTINCT parts the hyperedge touches - 1)
///
/// This is exactly what the GPU `compute_connectivity_10k` kernel accumulates
/// and what the challenge scores, so best-of-K selects on the real objective
/// rather than on any internal proxy.  `seen` is caller-owned scratch of length
/// `np`; it is all-false on entry and is left all-false on exit (each hyperedge
/// clears precisely the flags it set), so the routine allocates nothing.
fn eval_km1(
    part: &[i32],
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    nh: usize,
    np: usize,
    seen: &mut [bool],
) -> i64 {
    // `nh` is passed in rather than derived from `hedge_offsets.len() - 1` so
    // the count never depends on how the offsets buffer happens to be sized.
    let mut total = 0i64;
    for h in 0..nh {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut d = 0i64;
        for k in s..e {
            let p = part[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < np && !seen[p as usize] {
                seen[p as usize] = true;
                d += 1;
            }
        }
        if d > 1 {
            total += d - 1;
        }
        for k in s..e {
            let p = part[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < np {
                seen[p as usize] = false;
            }
        }
    }
    total
}

/// ------------------------------------------------------------------------
/// wave8: RUIN-AND-RECREATE of a partition (host side, fully deterministic).
/// ------------------------------------------------------------------------
/// This is the diversification operator that `improvement007_ruin` /
/// `improvement008_best3ruin` / `improvement009_best5` measured as the single
/// largest quality lever on this track (+2,081 as a straight replacement of the
/// greedy construction, +3,033 as best-of-5 diversified starts, both at 90
/// nonces).  It is reproduced here VERBATIM in behaviour; the only changes are
/// that `free_cap` is now a parameter instead of `max(64, num_nodes/4)`, and
/// that the caller supplies `k_ruin`.
///
/// RUIN   : rank hyperedges by lambda = #distinct parts they touch (descending,
///          ties by ascending hyperedge id), walk the top `k_ruin` of them and
///          unassign (`part = -1`) every node they contain until `free_cap`
///          nodes have been freed.  A node is never freed out of a part of size
///          1, so no part can be emptied.
/// RECREATE: shuffle the freed list with a 64-bit LCG seeded by `seed`, then
///          insert the nodes back ONE AT A TIME, each into the part that
///          minimises the ADDED connectivity, i.e. maximises the number of
///          incident hyperedges that already touch that part (ties broken by
///          smaller part, then by lower part index), skipping parts that are
///          already at `max_part_size`.
///
/// !! The recreate ORDER is the fragile part.  The July campaign measured an
/// !! "improved" recreate (exact min-lambda scoring with a high-degree-first
/// !! order) at -20,620 quality.  Random-shuffle order is the proven recipe.
/// !! Do not reorder, do not sort, do not "improve" the scoring.
///
/// Returns `false` and leaves `partition` / `part_sizes` COMPLETELY UNTOUCHED
/// if anything about the result would be invalid (a part outside
/// `[1, max_part_size]`, an unassigned node, nothing to ruin).  All of the work
/// happens on private clones and is only committed on the last line, so a
/// `false` return is a guaranteed no-op.
fn ruin_recreate(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    k_ruin: usize,
    free_cap: usize,
    seed: u64,
) -> bool {
    let num_nodes = partition.len();
    if num_nodes == 0 || num_parts == 0 || k_ruin == 0 || free_cap == 0 {
        return false;
    }
    let free_cap = std::cmp::min(free_cap, num_nodes);

    let mut part_new: Vec<i32> = partition.clone();
    let mut sizes_new: Vec<i32> = part_sizes.clone();

    // Stamp array: `stamp_arr[p] == stamp` means part p was already counted for
    // the hyperedge currently being scanned.  Cheaper than clearing 64 bools.
    let mut stamp_arr: Vec<u32> = vec![0u32; num_parts];
    let mut stamp: u32 = 0;

    // ---- rank hyperedges by lambda ----------------------------------------
    let mut ranked: Vec<(i32, i32)> = Vec::with_capacity(num_hyperedges);
    for h in 0..num_hyperedges {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        if e <= s + 1 {
            continue;
        }
        stamp = stamp.wrapping_add(1);
        let mut lambda = 0i32;
        for k in s..e {
            let p = part_new[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < num_parts && stamp_arr[p as usize] != stamp {
                stamp_arr[p as usize] = stamp;
                lambda += 1;
            }
        }
        if lambda >= 2 {
            ranked.push((lambda, h as i32));
        }
    }
    if ranked.is_empty() {
        return false;
    }
    // Total order (lambda descending, then id ascending) => `sort_unstable_by`
    // has a unique answer and is deterministic.
    ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    // ---- RUIN --------------------------------------------------------------
    let mut freed_flag: Vec<bool> = vec![false; num_nodes];
    let mut freed: Vec<usize> = Vec::new();
    'ruin: for &(_, h32) in ranked.iter().take(k_ruin) {
        let h = h32 as usize;
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        for k in s..e {
            if freed.len() >= free_cap {
                break 'ruin;
            }
            let n = hedge_nodes[k] as usize;
            if freed_flag[n] {
                continue;
            }
            let cur = part_new[n];
            if cur < 0 || (cur as usize) >= num_parts {
                continue;
            }
            // Never empty a part: the balance validation below requires every
            // part to keep at least one node.
            if sizes_new[cur as usize] <= 1 {
                continue;
            }
            sizes_new[cur as usize] -= 1;
            part_new[n] = -1;
            freed_flag[n] = true;
            freed.push(n);
        }
    }
    if freed.is_empty() {
        return false;
    }

    // ---- deterministic Fisher-Yates over the freed list ---------------------
    let mut state: u64 = seed | 1;
    let mut i = freed.len();
    while i > 1 {
        state = state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let j = (state % i as u64) as usize;
        i -= 1;
        freed.swap(i, j);
    }

    // ---- RECREATE: greedy min-added-connectivity, in shuffled order ---------
    let mut score: Vec<i32> = vec![0i32; num_parts];
    for &n in freed.iter() {
        for v in score.iter_mut() {
            *v = 0;
        }
        let ns = node_offsets[n] as usize;
        let ne = node_offsets[n + 1] as usize;
        for idx in ns..ne {
            let h = node_hedges[idx] as usize;
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                for v in stamp_arr.iter_mut() {
                    *v = 0;
                }
                stamp = 1;
            }
            for k in s..e {
                let p = part_new[hedge_nodes[k] as usize];
                if p >= 0 && (p as usize) < num_parts && stamp_arr[p as usize] != stamp {
                    stamp_arr[p as usize] = stamp;
                    score[p as usize] += 1;
                }
            }
        }
        let mut best_p: i32 = -1;
        let mut best_score = -1i32;
        let mut best_size = i32::MAX;
        for p in 0..num_parts {
            if sizes_new[p] >= max_part_size {
                continue;
            }
            if score[p] > best_score || (score[p] == best_score && sizes_new[p] < best_size) {
                best_score = score[p];
                best_size = sizes_new[p];
                best_p = p as i32;
            }
        }
        if best_p < 0 {
            // Every part is full: fall back to the smallest part.  The final
            // validation below rejects the whole operation if that overflows.
            let mut min_sz = i32::MAX;
            for p in 0..num_parts {
                if sizes_new[p] < min_sz {
                    min_sz = sizes_new[p];
                    best_p = p as i32;
                }
            }
        }
        if best_p < 0 {
            return false;
        }
        part_new[n] = best_p;
        sizes_new[best_p as usize] += 1;
    }

    // ---- validate, then commit (all-or-nothing) ----------------------------
    for p in 0..num_parts {
        if sizes_new[p] < 1 || sizes_new[p] > max_part_size {
            return false;
        }
    }
    for &pv in part_new.iter() {
        if pv < 0 || (pv as usize) >= num_parts {
            return false;
        }
    }
    partition.copy_from_slice(&part_new);
    part_sizes.copy_from_slice(&sizes_new);
    true
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
    let compute_preferences_kernel = module.load_function("compute_node_preferences_10k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_10k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_10k")?;
    let compute_moves_kernel = module.load_function("compute_refinement_moves_optimized_10k")?;
    // STAGE 2: the two kernels above, fused behind one software grid barrier.
    // Both originals are STILL loaded and STILL launched: precompute_edge_flags_10k
    // on its own at the five non-refine sites (the elite-consensus block, the
    // bottleneck-repair block, the crossover, the pre-balance flush and
    // do_hyperedge_centric_phase!), and compute_refinement_moves_optimized_10k
    // nowhere -- it is kept in the .cu as the readable reference for phase 2.
    // ================================================================
    // wave14 knob (see NOTES.md).  engine_legacy is the ONLY difference
    // between this file and wave13_10k_round/fix/track_10k.rs.
    //
    //   engine_legacy 0 (default) = `fused_flags_moves_bs_10k`, the bit-sliced
    //                               phase-2 kernel (no per-thread part array,
    //                               4-deep pin walk, 20 ns barrier spin)
    //                 1           = `fused_flags_moves_10k`, the p10kf kernel
    //
    // Both kernels take the SAME 15 arguments in the SAME order and use the
    // same grid barrier accounting, so this decides a name and nothing else:
    // every launch site, every buffer and every host line below is untouched.
    // The bit-sliced counters wrap at 2^16, which reproduces the champion's
    // `part_info[p] & 0xFFFF` / `>> 16` exactly IFF a node's degree is below
    // 65536; a node's degree is the number of hyperedges that list it, so the
    // guard below makes that unconditional.
    // ================================================================
    let engine_legacy: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("engine_legacy").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    let engine_bitsliced: bool =
        engine_legacy == 0 && (challenge.num_hyperedges as i64) < 65536;
    let fused_moves_kernel = module.load_function(if engine_bitsliced {
        "fused_flags_moves_bs_10k"
    } else {
        "fused_flags_moves_10k"
    })?;
    let _ = &compute_moves_kernel;
    let execute_moves_kernel = module.load_function("execute_refinement_moves_10k")?;
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
    let compute_swap_topk_kernel = module.load_function("compute_swap_topk_10k")?;
    let compute_best_swap_pairs_kernel = module.load_function("compute_best_swap_pairs_10k")?;
    let compute_hedge_consolidation_kernel = module.load_function("compute_hedge_consolidation_moves_10k")?;
    let polish_exploration_kernel = module.load_function("polish_exploration_10k")?;
    let compute_part_cut_cost_kernel = module.load_function("compute_part_cut_cost_10k")?;
    let repair_bottleneck_part_kernel = module.load_function("repair_bottleneck_part_10k")?;

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

    // ---- STAGE 2: launch geometry of the fused kernel ------------------
    // The fused kernel spins on a software grid barrier, so EVERY block must be
    // resident at once.  Two things buy that: `__launch_bounds__(128, 2)` in the
    // kernel (the compiler budgets registers for 2 blocks of 128 threads per SM)
    // and the cap below.  This track's `block_size` is 256, which would make
    // `__launch_bounds__(128, ..)` an illegal launch, so the fused kernel gets its
    // OWN block size; every ORIGINAL kernel keeps `block_size` and its original
    // launch geometry, untouched.
    //
    // The fused kernel's OUTPUT does not depend on either number: phase 1 writes
    // edge_flags_*[h] as a pure function of h, phase 2 writes
    // move_priorities[node] as a pure function of node, and nothing is compacted
    // or counted per block.  The grid only has to be big enough to be worth a
    // grid-stride loop and small enough to be co-resident.
    let fused_block: u32 = std::cmp::min(128, prop.maxThreadsPerBlock as u32);
    let fused_blocks_needed = std::cmp::max(
        (challenge.num_hyperedges as u32 + fused_block - 1) / fused_block,
        (challenge.num_nodes as u32 + fused_block - 1) / fused_block,
    );
    let fused_grid: u32 = std::cmp::max(
        1,
        std::cmp::min(
            fused_blocks_needed,
            2 * std::cmp::max(1, prop.multiProcessorCount as u32),
        ),
    );
    let fused_cfg = LaunchConfig {
        grid_dim: (fused_grid, 1, 1),
        block_dim: (fused_block, 1, 1),
        shared_mem_bytes: 0,
    };

    let balance_cfg = cfg.clone();

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
    let scan_limit_cycle = 8usize;

    // ================================================================
    // wave11 (10k track).  Nothing here changes a move the solver makes
    // at the default hyperparameters.
    //
    //   swap_legacy   0 (default) = pruned swap-phase cycle scans
    //                 1           = the original unpruned scans
    //   reset_legacy  0 (default) = skip the two per-round counter H2Ds
    //                               that no kernel ever reads
    //                 1           = the original H2Ds
    //   probe_phases  0 (default) = OFF and completely inert.  No clock is
    //                               read, no km1 is computed, no file is
    //                               opened and nothing is allocated for it.
    //                 1           = one line per pipeline phase (km1 + us)
    //                               on stderr and appended to
    //                               /tmp/probe_phases.txt
    //                 2           = 1 + a main-loop km1 trace every
    //                               `probe_every` rounds (0 = auto, 100
    //                               samples).  Level 2 is NOT cost-neutral;
    //                               never quote solve_ms from it.
    //
    // Why the two `*_legacy` knobs are pure no-ops in the default
    // direction is argued in NOTES.md; they exist so the pod can A/B the
    // claim instead of believing it.
    // ================================================================
    let swap_legacy: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("swap_legacy").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    let reset_legacy: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("reset_legacy").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);

    // ================================================================
    // wave13 fix knobs (see fix/NOTES.md)
    //   sel_legacy  0 (default) = counting-sort ("bucket") move selection
    //               1           = the original select_nth_unstable_by +
    //                             sort_unstable_by over the whole candidate
    //                             list.  Same SET, same ORDER, so the two
    //                             must agree per-nonce bit for bit.
    //   exec_fast   0 (default) = the original 3 H2D + execute-moves launch
    //               1           = replace them with ONE wholesale H2D of
    //                             partition_host_mirror.  Saves 1 launch +
    //                             1 H2D per refine round; relies on the
    //                             mirror invariant, hence OFF by default.
    // ================================================================
    let sel_legacy: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("sel_legacy").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    let exec_fast: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("exec_fast").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    let probe_phases: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("probe_phases").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 2);
    let probe_every: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("probe_every").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1_000_000);
    let probe_on: bool = probe_phases >= 1;
    let probe_trace: bool = probe_phases >= 2;
    // Read-only fingerprint so the 4 concurrent workers can be demultiplexed
    // offline.  `init_random_seed` is the same four seed bytes the solver
    // already reads, so no new assumption is made about `challenge.seed`.
    let probe_nonce: u32 = if probe_on { init_random_seed } else { 0 };
    let mut probe_seen: Vec<bool> = if probe_on {
        vec![false; num_parts_usize]
    } else {
        Vec::new()
    };
    // wave12 MERGE: the hyperedge CSR host copies live HERE, before the probe
    // macros, and are UNCONDITIONAL.
    //   * wave11 fetched them only `if probe_on` (they were measurement-only).
    //   * p10kf fetches them on every solve: `eval_km1` is the best-of-K
    //     selector, not a probe.
    //   * `probe_km1!` is a `macro_rules!`, and macro_rules bodies resolve
    //     LOCAL VARIABLES at the macro's DEFINITION site (mixed-site hygiene),
    //     so the binding must precede the macro definition below -- it cannot
    //     be left in p10kf's best-of-K prologue.
    // Cost is identical to p10kf's (two D2Hs, once per solve, never in a loop).
    let hedge_offsets_host = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let hyperedge_nodes_host = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    // `Instant::now()` is only ever reached from inside `if probe_on`.
    let mut probe_mark: Option<std::time::Instant> = None;

    macro_rules! probe_out {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            eprint!("{}", line);
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/probe_phases.txt")
            {
                let _ = f.write_all(line.as_bytes());
            }
        }};
    }
    macro_rules! probe_lap {
        () => {{
            let d: u64 = match probe_mark {
                Some(t) => t.elapsed().as_micros() as u64,
                None => 0,
            };
            probe_mark = Some(std::time::Instant::now());
            d
        }};
    }
    macro_rules! probe_km1 {
        ($part:expr) => {
            eval_km1(
                $part,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                challenge.num_hyperedges as usize,
                num_parts_usize,
                &mut probe_seen,
            )
        };
    }

    let mut probe_rounds_exec = 0usize;
    let mut probe_moves = 0i64;
    let mut probe_he_calls = 0usize;
    let mut probe_perturbs = 0usize;
    let mut probe_quick_exec = 0usize;
    let mut probe_polish_exec = 0usize;
    let mut probe_ils_iters = 0usize;
    let probe_step: usize = if probe_every > 0 {
        probe_every as usize
    } else {
        std::cmp::max(1, refinement_rounds / 100)
    };

    // ---- host buffers that used to be allocated inside the hot loops ----
    // `memcpy_dtov` allocates a fresh Vec on every call; `memcpy_dtoh` fills a
    // buffer we own.  Same bytes, same order, one allocation for the solve.
    let mut nvm_host: Vec<i32> = vec![0i32; grid_x_calc as usize];
    // wave12 MERGE: wave11 hoisted this out of six `memcpy_dtov(&d_num_valid_moves)`
    // call sites; p10kf's fusion DELETES all six D2Hs outright (the fused kernel
    // never writes `d_num_valid_moves`, and the `num_valid_moves == 0` break is
    // subsumed by the `valid_moves.is_empty()` break immediately below it at
    // every site).  A's change strictly subsumes B's, so the buffer is now
    // unread.  It is kept, not deleted, so that the wave11 hunk is still
    // literally present and so that restoring an unfused site is a one-line
    // edit.  It is also re-zeroed per best-of-K run with the rest of the
    // hoisted state.
    let _ = &nvm_host;
    // 16 == `max_moves_per_hedge`, declared a few lines below.
    let mut move_counts: Vec<i32> = vec![0i32; challenge.num_hyperedges as usize];
    let mut move_node_hedge_host: Vec<i32> =
        vec![0i32; challenge.num_hyperedges as usize * 16];
    let mut move_part_hedge_host: Vec<i32> =
        vec![0i32; challenge.num_hyperedges as usize * 16];
    let mut node_included: Vec<bool> = vec![false; challenge.num_nodes as usize];

    // Bits >= k of a 64-bit mask (k >= 64 -> none).  Lets a mask-driven loop
    // resume exactly where a plain `for i in 0..np` would have, so the pruned
    // cycle scans visit the same indices in the same ascending order.
    let sp_hi = |k: usize| -> u64 {
        if k >= 64 { 0u64 } else { u64::MAX << k }
    };

    // ---- swap-phase scratch: replaces a 2 MB zero-initialised stack array ----
    const SP_SLOTS: usize = 8;
    let sp_np2 = num_parts_usize * num_parts_usize;
    let mut sp_cnt: Vec<u8> = vec![0u8; sp_np2];
    let mut sp_nodes: Vec<i32> = vec![0i32; sp_np2 * SP_SLOTS];
    let mut sp_gains: Vec<i32> = vec![0i32; sp_np2 * SP_SLOTS];
    let mut sp_vm: Vec<u8> = vec![0u8; sp_np2];
    let mut sp_m2: Vec<u8> = vec![0u8; sp_np2];
    let mut sp_mc: Vec<u8> = vec![0u8; sp_np2];
    let mut sp_row2: Vec<u64> = vec![0u64; num_parts_usize];
    let mut sp_col2: Vec<u64> = vec![0u64; num_parts_usize];
    let mut sp_row8: Vec<u64> = vec![0u64; num_parts_usize];
    let mut sp_col8: Vec<u64> = vec![0u64; num_parts_usize];
    let mut sp_rev_head: Vec<i32> = vec![-1i32; challenge.num_nodes as usize];
    let mut sp_rev_next: Vec<i32> = vec![-1i32; sp_np2 * SP_SLOTS];
    let mut sp_rev_pair: Vec<u32> = vec![0u32; sp_np2 * SP_SLOTS];
    let mut sp_rev_slot: Vec<u8> = vec![0u8; sp_np2 * SP_SLOTS];
    let mut sp_rev_touched: Vec<i32> = Vec::with_capacity(sp_np2 * SP_SLOTS);
    // `compute_best_swap_pairs_10k` writes every off-diagonal entry on every
    // launch and the host only ever reads off-diagonal entries, so this buffer
    // no longer has to be re-allocated (cudaMalloc + memset) once per swap round.
    let mut d_bp = stream.alloc_zeros::<i32>(sp_np2 * 3)?;
    let mut bp_host: Vec<i32> = vec![0i32; sp_np2 * 3];

    let swap_buf_size = 4 * challenge.num_nodes as usize;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;
    let swap_topk_size = num_parts_usize * num_parts_usize * 32 * 2;
    let mut d_swap_topk = stream.alloc_zeros::<i32>(swap_topk_size)?;
    let mut swap_topk_host: Vec<i32> = vec![0i32; swap_topk_size];

    let mut partition_host_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut nodes_in_part_host: Vec<i32> = vec![0i32; num_parts_usize];
    let mut move_keys_host: Vec<i32> = vec![0i32; challenge.num_nodes as usize];

    // ---- wave13 fix F1: packed-key scratch, one allocation per solve -----
    // `by_buf` holds the candidates bucketed by target part, `sel_keys` the
    // per-part survivors.  Both are indexed only below `valid_moves.len()`,
    // which is bounded by num_nodes.
    let mut by_buf: Vec<u64> = vec![0u64; challenge.num_nodes as usize];
    let mut sel_keys: Vec<u64> = vec![0u64; challenge.num_nodes as usize];
    let zero_counter_1 = [0i32; 1];
    let zero_counter_grid = vec![0i32; ((challenge.num_nodes as u32 + block_size - 1) / block_size) as usize];
    // STAGE 2: `zero_counter_1` / `zero_counter_grid` are now referenced only
    // from `reset_move_counters!`, which is no longer expanded anywhere (the
    // two counters it zeroed are never read: `d_num_valid_moves`'s only reader
    // was the per-round D2H this stage removes, and `d_moves_executed` is an
    // output argument of execute_refinement_moves_10k that the kernel body
    // never writes).  Kept, rather than deleted, so the rest of the file stays
    // byte-for-byte the stage-1 text.
    let _ = (&zero_counter_1, &zero_counter_grid);
    let mut partition_host_mirror: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut nodes_in_part_mirror: Vec<i32> = vec![0i32; num_parts_usize];
    let mut accepted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut accepted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);    
    let mut d_accepted_move_nodes = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_accepted_move_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_hedge_choice = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let max_moves_per_hedge = 16usize;
    let mut d_move_node_hedge = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize * max_moves_per_hedge)?;
    let mut d_move_part_hedge = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize * max_moves_per_hedge)?;
    let mut d_move_count_hedge = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_balance_best_node = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_balance_best_gain = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_balance_best_deg = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
    let mut d_balance_best_target = stream.alloc_zeros::<i32>(grid_x_calc as usize)?;
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

    // ---- STAGE 2: software grid-barrier counter -------------------------
    // Monotone on the device; the host passes an ABSOLUTE target
    // `fused_calls * fused_grid`, so a block that arrives early spins until all
    // `fused_grid` blocks of this call have arrived.  Host mirror and device
    // counter are reset TOGETHER at the top of every best-of-K run (resetting
    // only one of the two either deadlocks or lets the u32 wrap).
    let mut d_grid_barrier = stream.alloc_zeros::<u32>(1)?;
    let mut fused_calls: u32 = 0;

    // wave11: both of these H2Ds are dead.
    //   * `compute_refinement_moves_optimized_10k` ends with
    //     `if (threadIdx.x == 0) num_valid_moves[blockIdx.x] = valid_move;`
    //     -- a plain STORE, executed by every block of a grid that is exactly
    //     `grid_x_calc` blocks wide, i.e. the whole buffer is overwritten on
    //     every launch, and it is only ever read after that launch.
    //     VERIFIED against p10kf's kernels_10k.cu (line ~431), unchanged there.
    //   * `execute_refinement_moves_10k` never dereferences `moves_executed`
    //     at all (its body is `partition[node] = target_part;`), and the host
    //     never copies `d_moves_executed` back.
    //     VERIFIED against p10kf's kernels_10k.cu (line ~279), unchanged there.
    // So skipping them cannot change one bit of any partition.  `reset_legacy`
    // = 1 puts them back for an A/B.
    //
    // wave12 MERGE: p10kf's fusion makes them deader still -- the fused kernel
    // `fused_flags_moves_10k` takes NEITHER buffer as an argument, and
    // `compute_refinement_moves_optimized_10k` is no longer launched on any
    // live path (its only remaining mention is the never-expanded
    // `do_combined_phase!`).  The macro CALL is nevertheless kept at all six
    // fused sites so that `reset_legacy = 1` still restores exactly wave11's
    // pair of H2Ds and the A/B stays runnable; at the default it expands to a
    // single predictable branch on a loop-invariant `i64`.
    macro_rules! reset_move_counters {
        () => {{
            if reset_legacy != 0 {
                stream.memcpy_htod(&zero_counter_grid, &mut d_num_valid_moves)?;
                stream.memcpy_htod(&zero_counter_1, &mut d_moves_executed)?;
            }
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
                if exec_fast != 0 {
                    // ---- wave13 fix F2 (OFF by default) -------------------
                    // The loop above already applied exactly these moves to
                    // `partition_host_mirror`, and execute_refinement_moves_10k
                    // does nothing but `partition[node] = target_part` for the
                    // same list (node ids in the list are unique), so one
                    // wholesale H2D of the mirror leaves d_partition in the
                    // same state -- PROVIDED the mirror agreed with d_partition
                    // on entry.  That invariant is the whole risk; hence the
                    // knob defaults to 0.  Two runtime calls instead of four.
                    stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
                    stream.memcpy_htod(&partition_host_mirror, &mut d_partition)?;
                } else {
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
            stream.memcpy_dtoh(&d_move_count_hedge, &mut move_counts)?;
            stream.memcpy_dtoh(&d_move_node_hedge, &mut move_node_hedge_host)?;
            stream.memcpy_dtoh(&d_move_part_hedge, &mut move_part_hedge_host)?;
            valid_moves.clear();
            for v in node_included.iter_mut() {
                *v = false;
            }
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
                    }
                }
                me_he
            } else {
                0i32
            }
        }};
    }

    macro_rules! do_combined_phase {
        () => {{
            let mut total_combined = 0i32;

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

            unsafe {
                stream
                    .launch_builder(&compute_swap_gains_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&neg_gain_thresh)
                    .arg(&global_round)
                    .arg(&d_node_tabu_until)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_swap_gains)
                    .launch(cfg.clone())?;
            }

            unsafe {
                stream
                    .launch_builder(&compute_swap_topk_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&d_partition)
                    .arg(&d_swap_gains)
                    .arg(&mut d_swap_topk)
                    .launch(LaunchConfig {
                        grid_dim: ((num_parts_usize * num_parts_usize) as u32, 1, 1),
                        block_dim: (block_size.min(128), 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
            stream.memcpy_dtoh(&d_swap_topk, &mut swap_topk_host)?;

            let np = num_parts_usize;
            let mut candidates: Vec<(i32, u8, usize, usize)> = Vec::new();

            for (node, &key) in move_keys_host.iter().enumerate() {
                if key <= 0 { continue; }
                let gain = (key >> 16) - 1000;
                let target = (key & 63) as usize;
                if gain > 0 && target < np && partition_host_mirror[node] as usize != target {
                    candidates.push((gain, 0, node, target));
                }
            }

            for a in 0..np {
                for b in 0..np {
                    if a == b { continue; }
                    let idx_ab = a * np + b;
                    let base_ab = idx_ab * 32 * 2;
                    let idx_ba = b * np + a;
                    let base_ba = idx_ba * 32 * 2;
                    let node_a = swap_topk_host[base_ab] as usize;
                    let gain_a = swap_topk_host[base_ab + 1];
                    if node_a >= num_nodes_i as usize || gain_a <= 0 { continue; }
                    if partition_host_mirror[node_a] as usize != a { continue; }
                    let node_b = swap_topk_host[base_ba] as usize;
                    let gain_b = swap_topk_host[base_ba + 1];
                    if node_b >= num_nodes_i as usize || gain_b <= 0 { continue; }
                    if partition_host_mirror[node_b] as usize != b { continue; }
                    if node_a == node_b { continue; }
                    let total = gain_a + gain_b;
                    if total > 0 {
                        candidates.push((total, 1, node_a, node_b));
                    }
                }
            }

            candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));

            let mut used = vec![false; challenge.num_nodes as usize];
            let mut move_nodes = Vec::with_capacity(candidates.len());
            let mut move_targets = Vec::with_capacity(candidates.len());

            for (gain, op, n1, n2) in candidates.iter() {
                if used[*n1] { continue; }
                if *op == 0 {
                    let tgt = *n2;
                    if tgt >= np { continue; }
                    let cur = partition_host_mirror[*n1];
                    if cur as usize == tgt { continue; }
                    if nodes_in_part_mirror[tgt] >= challenge.max_part_size as i32 { continue; }
                    if nodes_in_part_mirror[cur as usize] <= 1 { continue; }
                    partition_host_mirror[*n1] = tgt as i32;
                    nodes_in_part_mirror[cur as usize] -= 1;
                    nodes_in_part_mirror[tgt] += 1;
                    move_nodes.push(*n1 as i32);
                    move_targets.push(tgt as i32);
                    used[*n1] = true;
                    total_combined += 1;
                } else {
                    let n2 = *n2;
                    if n2 >= challenge.num_nodes as usize || used[n2] { continue; }
                    let cur_a = partition_host_mirror[*n1];
                    let cur_b = partition_host_mirror[n2];
                    if cur_a as usize != a || cur_b as usize != b { continue; }
                    if nodes_in_part_mirror[cur_a as usize] <= 1 || nodes_in_part_mirror[cur_b as usize] <= 1 { continue; }
                    partition_host_mirror[*n1] = cur_b;
                    partition_host_mirror[n2] = cur_a;
                    nodes_in_part_mirror[cur_a as usize] -= 1;
                    nodes_in_part_mirror[cur_b as usize] -= 1;
                    nodes_in_part_mirror[cur_a as usize] += 1;
                    nodes_in_part_mirror[cur_b as usize] += 1;
                    move_nodes.push(*n1 as i32);
                    move_targets.push(cur_b);
                    move_nodes.push(n2 as i32);
                    move_targets.push(cur_a);
                    used[*n1] = true;
                    used[n2] = true;
                    total_combined += 1;
                }
            }

            if total_combined > 0 {
                stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
                let accepted_len = move_nodes.len();
                stream.memcpy_htod(move_nodes.as_slice(), &mut d_accepted_move_nodes)?;
                stream.memcpy_htod(move_targets.as_slice(), &mut d_accepted_move_parts)?;
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
                for i in (0..move_nodes.len()).step_by(if i < move_nodes.len()-1 && move_nodes[i+1] == move_nodes[i] { 2 } else { 1 }) {
                    let node = move_nodes[i] as usize;
                    node_tabu_until[node] = global_round + tabu_tenure as i32;
                }
            }
            total_combined
        }};
    }

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


    // =====================================================================
    // ==== best-of-K (`runs`): restarts from the ONE construction result ===
    // =====================================================================
    // Everything ABOVE this line is CONSTRUCTION: the seeded MinHash
    // hyperedge clustering (`hyperedge_clustering_10k` with hash_a/hash_b
    // derived from `challenge.seed`), `compute_node_preferences_10k` (which
    // consumes `init_restart_id` and `init_random_seed`), the host priority
    // sort, `replay_initial_assignment_host!` and
    // `execute_node_assignments_10k`.  It is a pure function of the challenge
    // and of the hyperparameters, so it runs ONCE and every run starts from
    // its result.  Everything BELOW is the per-run body.
    let bok_runs = hyperparameters
        .as_ref()
        .and_then(|p| p.get("runs").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, K_MAX) as usize)
        .unwrap_or(1);

    // ---- START DIVERSIFICATION knobs (runs k >= 1 only) -------------------
    // Every one of these is read ONLY inside `if bok_run >= 1 { .. }`, so run 0
    // -- and therefore `runs == 1` -- is byte-identical to the base solver no
    // matter how they are set.
    //
    // `run_ruin_frac` : fraction of hyperedges considered for the ruin.  The
    //                   walk stops at `run_ruin_cap` freed nodes, so above
    //                   ~0.02 this knob is nearly inert (the cap binds first).
    //                   0 disables the ruin entirely.
    // `run_ruin_cap`  : fraction of NODES freed = the real strength knob.
    // `run_ruin_growth_pct`: cap multiplier per extra run, in percent, so a
    //                   large K sweeps a ladder of ruin strengths.  100 = flat.
    let run_ruin_frac: f64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_frac").and_then(|v| v.as_f64()))
        .unwrap_or(0.02)
        .max(0.0)
        .min(1.0);
    let run_ruin_cap: f64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_cap").and_then(|v| v.as_f64()))
        .unwrap_or(0.25)
        .max(0.0)
        .min(0.90);
    let run_ruin_growth_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_growth_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(25, 400);
    // OPT-IN, default 0.  Set to 1 to ruin run 0's start as well.  This
    // DELIBERATELY breaks the run-0 / K=1 bit-identity, so it is never on by
    // default -- it exists so that ONE build can also measure the ruin's pure
    // REPLACEMENT effect (`{"runs":1,"run_ruin_run0":1}`) without a rebuild.
    // It must never be baked into `mod.rs`.
    let run_ruin_run0: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_run0").and_then(|v| v.as_i64()))
        .unwrap_or(0);

    // Integer derivations, done ONCE.  The two f64 -> usize conversions are a
    // single IEEE multiply plus a truncation of a value far below 2^53, so they
    // are exact and identical on every host; everything downstream is
    // integer-only.  `+ 0.5` then truncate = round-half-up, so a fraction whose
    // binary double lands a hair below the intended product can never silently
    // drop an item.
    let ruin_k_base: usize =
        ((challenge.num_hyperedges as f64) * run_ruin_frac + 0.5) as usize;
    let ruin_cap_base: usize = ((challenge.num_nodes as f64) * run_ruin_cap + 0.5) as usize;
    let ruin_enabled: bool = ruin_k_base >= 1
        && ruin_cap_base >= 1
        && (bok_runs > 1 || run_ruin_run0 != 0);

    // ---- host copies of the CSRs ------------------------------------------
    // `eval_km1` (the best-of-K selector) needs the hyperedge CSR; the RECREATE
    // step of the ruin additionally needs the node->hyperedge CSR.  Both are
    // read-only for the whole solve, so they are fetched once.
    // wave12 MERGE: `hedge_offsets_host` / `hyperedge_nodes_host` moved UP, to
    // just before the probe macros (see the note there).  Only the ruin's two
    // node-CSR copies are still fetched here, still lazily.
    let node_offsets_host: Vec<i32> = if ruin_enabled {
        stream.memcpy_dtov(&challenge.d_node_offsets)?
    } else {
        Vec::new()
    };
    let node_hedges_host: Vec<i32> = if ruin_enabled {
        stream.memcpy_dtov(&challenge.d_node_hyperedges)?
    } else {
        Vec::new()
    };

    // ---- the construction result, kept verbatim as every run's start ------
    // Both the HOST mirrors and the DEVICE buffers are snapshotted.  The device
    // snapshot exists so that a non-diversified run (run 0, always, by default)
    // is restored byte for byte WITHOUT relying on the claim that
    // `d_partition == partition_host_mirror` here -- that claim is true
    // (`execute_node_assignments_10k` writes `partition[node] = assigned_part`
    // for every node exactly once, and `assigned_parts` was built from
    // `partition_host_mirror`), but it is load-bearing enough to not want to
    // depend on it.
    let bok_base_part = partition_host_mirror.clone();
    let bok_base_nip = nodes_in_part_mirror.clone();
    let bok_base_dev_part = stream.memcpy_dtov(&d_partition)?;
    let bok_base_dev_nip = stream.memcpy_dtov(&d_nodes_in_part)?;
    let mut bok_seen: Vec<bool> = vec![false; num_parts_usize];
    let mut bok_best_part: Vec<u32> = Vec::new();
    let mut bok_best_km1: i64 = i64::MAX;

    for bok_run in 0..bok_runs {
        // run 0 => run_off == 0 => bit-identical to the base solver.
        let run_off: u64 = (bok_run as u64).wrapping_mul(RUN_OFF_MUL);

        // ---- per-run HOST state reset -------------------------------------
        // The first three carry the partition itself; the rest are scratch that
        // is fully overwritten before it is read, and are cleared anyway so a
        // run's state is provably independent of the previous run.
        partition_host_mirror.copy_from_slice(&bok_base_part);
        nodes_in_part_mirror.copy_from_slice(&bok_base_nip);
        nodes_in_part_host.copy_from_slice(&bok_base_nip);
        partition_host_swap.copy_from_slice(&bok_base_part);
        partition_mut_swap.copy_from_slice(&bok_base_part);
        for v in node_tabu_until.iter_mut() {
            *v = 0;
        }
        for v in move_keys_host.iter_mut() {
            *v = 0;
        }
        for v in swap_topk_host.iter_mut() {
            *v = 0;
        }
        for v in tgt_used.iter_mut() {
            *v = 0;
        }
        for v in tgt_quota.iter_mut() {
            *v = 0;
        }
        valid_moves.clear();
        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        accepted_move_nodes.clear();
        accepted_move_parts.clear();
        // `global_round` is the tabu clock.  It MUST restart at 0 together with
        // `node_tabu_until`: leaving it monotone while zeroing the stamps is
        // harmless, but leaving the stamps while restarting the clock would
        // make every node tabu for the whole of run k.
        global_round = 0;
        stagnant_rounds = 0;

        // ---- START DIVERSIFICATION, runs k >= 1 ONLY ----------------------
        // Run 0 never enters this block (unless `run_ruin_run0` was explicitly
        // set), so its starting partition is exactly the construction result
        // the base solver saw.  All of this runs BEFORE the H2Ds below, so the
        // diversified start is what gets uploaded.
        let mut diversified = false;
        if (bok_run >= 1 || run_ruin_run0 != 0) && ruin_enabled {
            // Strength ladder: run 1 gets `ruin_cap_base`, run k gets it
            // multiplied by (growth_pct/100)^(k-1), integer arithmetic.
            let mut cap: usize = ruin_cap_base;
            for _ in 1..bok_run {
                cap = cap.saturating_mul(run_ruin_growth_pct as usize) / 100;
                if cap >= challenge.num_nodes as usize {
                    cap = challenge.num_nodes as usize;
                    break;
                }
            }
            let cap = cap.clamp(1, challenge.num_nodes as usize);
            let ruin_seed: u64 = (bok_run as u64)
                .wrapping_mul(RUIN_SEED_MUL)
                .wrapping_add(RUIN_SEED_ADD)
                | 1;
            // `false` means the operator declined and left BOTH host mirrors
            // untouched, i.e. run k simply starts from the plain construction
            // (still diversified by `run_off`).
            diversified = ruin_recreate(
                &mut partition_host_mirror,
                &mut nodes_in_part_mirror,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                &node_offsets_host,
                &node_hedges_host,
                challenge.num_hyperedges as usize,
                num_parts_usize,
                challenge.max_part_size as i32,
                ruin_k_base,
                cap,
                ruin_seed,
            );
        }

        // ---- per-run DEVICE state reset -----------------------------------
        // `d_partition` / `d_nodes_in_part` are RE-BOUND inside the body (the
        // ILS restore, the crossover child and the final best-restore all
        // assign a fresh allocation), so at the top of run k these names refer
        // to the LAST buffer run k-1 created.  Same length and element type, so
        // an H2D is the correct reset -- a rebind here would be wrong because
        // the macros below capture the names, not the buffers.
        if diversified {
            stream.memcpy_htod(&partition_host_mirror, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
        } else {
            stream.memcpy_htod(&bok_base_dev_part, &mut d_partition)?;
            stream.memcpy_htod(&bok_base_dev_nip, &mut d_nodes_in_part)?;
        }
        // Every buffer below was `alloc_zeros`'d and is untouched by the
        // construction (which writes only d_hyperedge_clusters, d_pref_parts,
        // d_pref_priorities, d_partition and d_nodes_in_part), so on run 0
        // these memsets are a provable no-op.  For runs k >= 1 they make the
        // device state independent of the previous run rather than merely
        // argued to be: each of these IS fully written by its producing kernel
        // before it is read within a round.
        // STAGE 2: barrier counter + its host mirror, reset TOGETHER.
        stream.memset_zeros(&mut d_grid_barrier)?;
        fused_calls = 0;
        stream.memset_zeros(&mut d_node_tabu_until)?;
        stream.memset_zeros(&mut d_num_valid_moves)?;
        stream.memset_zeros(&mut d_moves_executed)?;
        stream.memset_zeros(&mut d_edge_flags_all)?;
        stream.memset_zeros(&mut d_edge_flags_double)?;
        stream.memset_zeros(&mut d_move_priorities)?;
        stream.memset_zeros(&mut d_swap_gains)?;
        stream.memset_zeros(&mut d_swap_topk)?;
        stream.memset_zeros(&mut d_hedge_choice)?;
        stream.memset_zeros(&mut d_move_node_hedge)?;
        stream.memset_zeros(&mut d_move_part_hedge)?;
        stream.memset_zeros(&mut d_move_count_hedge)?;
        stream.memset_zeros(&mut d_accepted_move_nodes)?;
        stream.memset_zeros(&mut d_accepted_move_parts)?;
        stream.memset_zeros(&mut d_balance_best_node)?;
        stream.memset_zeros(&mut d_balance_best_gain)?;
        stream.memset_zeros(&mut d_balance_best_deg)?;
        stream.memset_zeros(&mut d_balance_best_target)?;
        stream.memset_zeros(&mut d_part_cut_costs)?;
        stream.memset_zeros(&mut d_bottleneck_part)?;
        stream.memset_zeros(&mut d_repair_gains)?;
        stream.memset_zeros(&mut d_repair_targets)?;
        stream.memset_zeros(&mut d_polish_backup_partition)?;
        stream.memset_zeros(&mut d_polish_backup_nip)?;
        stream.memset_zeros(&mut d_polish_best_partition)?;
        stream.memset_zeros(&mut d_polish_best_nip)?;
        // NOT reset, by design: d_hyperedge_clusters, d_hash_a, d_hash_b,
        // d_pref_parts, d_pref_priorities, d_sorted_nodes, d_assigned_parts --
        // these are the construction's output / inputs, read-only for the rest
        // of the solve and shared by every run.

        // ---- wave12 MERGE: per-run reset of wave11's HOISTED host state ----
        // wave11 moved these allocations OUT of `do_swap_phase!` / out of the
        // refine loops and out of `do_hyperedge_centric_phase!`, so they now
        // live for the whole solve instead of for one call.  Its exactness
        // argument is per CALL: every one of them is either (a) fully
        // overwritten before it is read within a call, or (b) explicitly
        // re-initialised at the top of the call.  The best-of-K wrapper turns
        // one solve into `runs` solves, so the same argument has to hold per
        // RUN -- and it does, but only because of the resets below, which
        // restore EXACTLY the values a fresh allocation would have had:
        //
        //   sp_cnt/sp_nodes/sp_gains  vec![0]   -- rebuilt every swap round for
        //       all np*np pairs, but slots >= cnt are stale-by-design and are
        //       never read (all reads are bounded by min(cnt, limit)).
        //   sp_vm/sp_m2/sp_mc         vec![0]   -- rewritten for every pair in
        //       the mask build at the top of each `sw_fast` round.
        //   sp_row2/col2/row8/col8    vec![0]   -- sp_col* are zeroed and
        //       sp_row* fully assigned in that same build.
        //   sp_rev_head               vec![-1]  -- the ONLY genuinely
        //       cross-call state.  wave11 keeps the invariant
        //       "sp_rev_head[n] >= 0  =>  n is in sp_rev_touched" and clears
        //       the heads through that list at the top of every round; the
        //       pair (head array, touched list) must therefore be reset
        //       TOGETHER, exactly as `global_round` and `node_tabu_until` are.
        //   sp_rev_next/pair/slot     vec![-1]/vec![0] -- `rev_len` restarts at
        //       0 each round and every entry is written before it is read.
        //   bp_host / d_bp            vec![0] / alloc_zeros -- `bp_host` is
        //       fully overwritten by its D2H; `d_bp`'s off-diagonal is written
        //       by every `compute_best_swap_pairs_10k` launch and its diagonal
        //       is never read, so the memset is belt-and-braces.
        //   nvm_host / move_counts / move_{node,part}_hedge_host -- fully
        //       overwritten by their D2Hs (nvm_host is now unread entirely).
        //   node_included             vec![false] -- cleared in place at the
        //       top of every `do_hyperedge_centric_phase!` call anyway.
        //
        // None of this is on a hot path: it is ~2 MB of memset once per run.
        for v in sp_cnt.iter_mut() { *v = 0; }
        for v in sp_nodes.iter_mut() { *v = 0; }
        for v in sp_gains.iter_mut() { *v = 0; }
        for v in sp_vm.iter_mut() { *v = 0; }
        for v in sp_m2.iter_mut() { *v = 0; }
        for v in sp_mc.iter_mut() { *v = 0; }
        for v in sp_row2.iter_mut() { *v = 0; }
        for v in sp_col2.iter_mut() { *v = 0; }
        for v in sp_row8.iter_mut() { *v = 0; }
        for v in sp_col8.iter_mut() { *v = 0; }
        // head array and touch list, together -- see above.
        for v in sp_rev_head.iter_mut() { *v = -1; }
        sp_rev_touched.clear();
        for v in sp_rev_next.iter_mut() { *v = -1; }
        for v in sp_rev_pair.iter_mut() { *v = 0; }
        for v in sp_rev_slot.iter_mut() { *v = 0; }
        for v in bp_host.iter_mut() { *v = 0; }
        stream.memset_zeros(&mut d_bp)?;
        for v in nvm_host.iter_mut() { *v = 0; }
        for v in move_counts.iter_mut() { *v = 0; }
        for v in move_node_hedge_host.iter_mut() { *v = 0; }
        for v in move_part_hedge_host.iter_mut() { *v = 0; }
        for v in node_included.iter_mut() { *v = false; }

        // ---- wave12 MERGE: per-run reset of the wave11 PROBE accumulators --
        // Measurement only (all of these are inert at `probe_phases = 0`).
        // Resetting them makes `probe_phases >= 1` emit one complete phase
        // ladder PER RUN, i.e. the probe measures a run, which is the unit
        // wave11's model is written in.  `probe_mark = None` makes the run's
        // first `probe_lap!()` report us=0 exactly as a fresh solve does.
        // NOTE for phase_report.py: the `n=` tag is the instance fingerprint,
        // not a run index, so a probe run at `runs > 1` emits `runs` ladders
        // under one tag.  Probe with `{"runs":1}`.
        probe_mark = None;
        probe_rounds_exec = 0;
        probe_moves = 0;
        probe_he_calls = 0;
        probe_perturbs = 0;
        probe_quick_exec = 0;
        probe_polish_exec = 0;
        probe_ils_iters = 0;

        // ---- wave11 probe: phase=start, now INSIDE the per-run loop --------
        if probe_on {
            let _ = probe_lap!();
            let k = probe_km1!(&partition_host_mirror);
            probe_out!(
                "n={:08x} phase=start km1={} refine={} ils={} quick={} polish={} tenure={} clusters={} move_limit={} us=0\n",
                probe_nonce, k, refinement_rounds, ils_iterations, ils_quick_refine,
                post_ils_polish, tabu_tenure, num_hedge_clusters, move_limit
            );
        }

    // ==== BEGIN per-run body: the base 10k solver, verbatim except that the
    // ==== four trajectory seeds below carry `.wrapping_add(run_off)`.
    // ==== (deliberately NOT re-indented, so a diff against the original shows
    // ==== only the inserted prologue/epilogue and the four seed lines.)
    for round in 0..refinement_rounds {
        global_round += 1;
        if probe_on {
            probe_rounds_exec += 1;
            if probe_trace && (round % probe_step == 0 || round + 1 == refinement_rounds) {
                let k = probe_km1!(&partition_host_mirror);
                probe_out!(
                    "n={:08x} trace round={} km1={} moves={} he={} perturbs={}\n",
                    probe_nonce, round, k, probe_moves, probe_he_calls, probe_perturbs
                );
            }
        }
        reset_move_counters!();
        // STAGE 2: precompute_edge_flags_10k + compute_refinement_moves_10k,
        // one launch, one software grid barrier.  The `d_num_valid_moves` D2H is
        // gone: nothing reads that buffer any more (the fused kernel does not even
        // take it), and the `num_valid_moves == 0` break is subsumed by the
        // `valid_moves.is_empty()` break a few lines below (see NOTES.md S2.3).
        // wave12 MERGE: the `reset_move_counters!()` above is wave11's, and is a
        // NO-OP at the default `reset_legacy = 0`.  It is kept at all six fused
        // sites so `reset_legacy = 1` still restores wave11's pair of H2Ds and the
        // A/B remains runnable; the fusion has made both buffers deader, not less
        // dead, so the knob cannot change a bit in either direction.
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_cfg.clone())?;
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

        // ---- wave13 fix F1: the sort is dead on the bucket path -----
        let bucket_path = sel_legacy == 0
            && k_base >= valid_moves.len()
            && k_cand >= valid_moves.len();
        if !bucket_path {
        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }
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
        if bucket_path {
            // ---- wave13 fix F1: counting-sort selection ------------------
            // Identical SET and identical ORDER to the scan in the `else`
            // branch; see fix/NOTES.md S2 for the proof.  The O(V log V)
            // tuple sort above is skipped entirely on this path.
            let mut pc = [0usize; 64];
            for &(_, key) in valid_moves.iter() {
                pc[(key & 63) as usize] += 1;
            }
            let mut po = [0usize; 64];
            let mut acc = 0usize;
            for p in 0..64 {
                po[p] = acc;
                acc += pc[p];
            }
            let mut cur = po;
            for &(node, key) in valid_moves.iter() {
                let t = (key & 63) as usize;
                let i = cur[t];
                cur[t] = i + 1;
                by_buf[i] = ((((key as u32) ^ 0x7FFF_FFFFu32) as u64) << 32) | (node as u64);
            }
            let np64 = std::cmp::min(num_parts_usize, 64usize);
            let mut m = 0usize;
            for p in 0..np64 {
                let c = pc[p];
                if c == 0 {
                    continue;
                }
                let s = po[p];
                let q = tgt_quota[p];
                let keep = if c > q {
                    by_buf[s..s + c].select_nth_unstable(q - 1);
                    q
                } else {
                    c
                };
                sel_keys[m..m + keep].copy_from_slice(&by_buf[s..s + keep]);
                m += keep;
            }
            sel_keys[..m].sort_unstable();
            for &v in sel_keys[..m].iter() {
                sorted_move_nodes.push(v as u32 as i32);
                sorted_move_parts.push(((((v >> 32) as usize) & 63) ^ 63) as i32);
            }
        } else {
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
        }

        if bucket_path && sorted_move_nodes.is_empty() {
            valid_moves.sort_unstable_by(cmp);
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

        if probe_on {
            probe_moves += moves_executed as i64;
        }

        if moves_executed == 0 {
            if probe_on {
                probe_he_calls += 1;
            }
            let he_moves = do_hyperedge_centric_phase!();
            if he_moves > 0 {
                stagnant_rounds = 0;
                continue;
            }

            stagnant_rounds += 1;
            
            if stagnant_rounds >= 5 && round < refinement_rounds.saturating_sub(50) {
                let mini_seed = (987654321u64 + (round as u64) * 123456789u64)
                    .wrapping_add(run_off);
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
                if probe_on {
                    probe_perturbs += 1;
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

    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_mirror);
        probe_out!(
            "n={:08x} phase=main_end km1={} rounds_exec={} budget={} moves={} he_calls={} perturbs={} us={}\n",
            probe_nonce, k, probe_rounds_exec, refinement_rounds, probe_moves,
            probe_he_calls, probe_perturbs, us
        );
    }

    let crossover_kernel = module.load_function("crossover_partitions_10k")?;
    let mut d_connectivity = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;

    // ================================================================
    // wave11: do_swap_phase, with the SAME decisions and the SAME order.
    // ----------------------------------------------------------------
    // Two things changed, neither of which can move a single swap:
    //
    // 1. `top_nodes: [[(usize,i32);32]; 64*64]` (a 2 MB zero-initialised
    //    STACK array, rebuilt on every call) is gone.  It was a verbatim
    //    copy of the leading valid entries of `swap_topk_host`; the cycle
    //    scans never look past slot 7 (`scan_limit_cycle` = 8), so the
    //    same values now live in `sp_nodes`/`sp_gains` (2 x 128 KB, one
    //    allocation for the whole solve).
    //
    // 2. The 3-cycle scan is O(np^3) and the 4-cycle scan is O(np^4) =
    //    64^4 = 16.7 M iterations of a body that chases four scattered
    //    16-byte loads out of that 2 MB array -- per SWAP ROUND, of which
    //    a nonce runs up to 1,290 (100 + 50 + 25*ils_iterations + 2*15 +
    //    10).  Every entry counted into `top_counts` has `gain > 0`, so
    //    the loops' own `... <= 0 { break }` guards can never fire and
    //    the ONLY thing that can stop a quadruple from firing is a node
    //    that has already been moved this round.  Define
    //
    //        ok(x,y) = "pair (x,y) still has an unmoved node in the
    //                   first min(count, scan_limit) slots"
    //
    //    Every node in pair (x,y)'s top-k list was in part x when
    //    `compute_swap_topk_10k` ran (the kernel only records nodes with
    //    `partition[n] == src`), so `partition_mut_swap[n] == x` is
    //    exactly "n has not been moved yet this round" -- and a node can
    //    never be moved twice, because after its first move every list it
    //    appears in fails that test.  If ok(a,b), ok(b,c), ok(c,d) or
    //    ok(d,a) is false the original code walks the whole quadruple and
    //    applies nothing, so SKIPPING it leaves the state bit-identical.
    //    `sp_row*[x]` / `sp_col*[y]` hold ok() as 64-bit masks.  Each loop
    //    level RE-READS the live mask and takes the lowest set bit that is
    //    >= the next index it would have visited (`sp_hi`), so it is exactly
    //    `for b in 0..np { if ok(a,b) { .. } }` with ok() evaluated at the
    //    moment the original would have evaluated it -- no assumption is
    //    made about ok() being monotone, and every applied cycle updates
    //    ok() for all four of its nodes in both directions.
    //
    // `swap_legacy = 1` restores the original unpruned scans (reading the
    // same `sp_*` tables) so the two can be A/B'd for bit-identity.
    // ================================================================
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
            let mut sw_rounds = 0usize;
            let mut sw_us_gpu = 0u64;
            let mut sw_us_bp = 0u64;
            let mut sw_us_c3 = 0u64;
            let mut sw_us_c4 = 0u64;
            $partition_host_swap.copy_from_slice(&partition_host_mirror);
            for _swap_round in 0..$max_rounds {
                sw_rounds += 1;
                global_round += 1;
                let sw_t0 = if probe_on { Some(std::time::Instant::now()) } else { None };
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
                if let Some(t) = sw_t0 {
                    sw_us_gpu += t.elapsed().as_micros() as u64;
                }

                // Compact pair table, byte-for-byte the leading entries the
                // original copied into `top_nodes` / `top_counts`.
                for a in 0..np {
                    for b in 0..np {
                        let idx = a * np + b;
                        if a == b {
                            sp_cnt[idx] = 0;
                            continue;
                        }
                        let base = idx * 32 * 2;
                        let mut cnt = 0usize;
                        while cnt < SP_SLOTS {
                            let node = $swap_topk_host[base + cnt * 2];
                            let gain = $swap_topk_host[base + cnt * 2 + 1];
                            if node >= 0 && gain > 0 {
                                sp_nodes[idx * SP_SLOTS + cnt] = node;
                                sp_gains[idx * SP_SLOTS + cnt] = gain;
                                cnt += 1;
                            } else {
                                break;
                            }
                        }
                        sp_cnt[idx] = cnt as u8;
                    }
                }

                $partition_mut_swap.copy_from_slice($partition_host_swap);
                let mut swap_count = 0usize;

                let sw_t1 = if probe_on { Some(std::time::Instant::now()) } else { None };
                {
                    let nn = challenge.num_nodes as usize;
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
                    stream.memcpy_dtoh(&d_bp, &mut bp_host)?;
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
                if let Some(t) = sw_t1 {
                    sw_us_bp += t.elapsed().as_micros() as u64;
                }

                let cyc_scan = $scan_lim_cyc;
                let cyc4_scan = 2usize;
                // The masks are u64 rows, and `sp_*` holds SP_SLOTS slots.
                let sw_fast = swap_legacy == 0 && cyc_scan <= SP_SLOTS && np <= 64;

                // ---- ok() masks, built AFTER the pair-swap block above so
                // ---- they see exactly the partition the cycle scans see.
                if sw_fast {
                    for &n in sp_rev_touched.iter() {
                        sp_rev_head[n as usize] = -1;
                    }
                    sp_rev_touched.clear();
                    for y in 0..np {
                        sp_col2[y] = 0;
                        sp_col8[y] = 0;
                    }
                    let mut rev_len = 0usize;
                    for x in 0..np {
                        let mut r2 = 0u64;
                        let mut r8 = 0u64;
                        for y in 0..np {
                            let p = x * np + y;
                            let cnt = sp_cnt[p] as usize;
                            sp_vm[p] = 0;
                            sp_m2[p] = 0;
                            sp_mc[p] = 0;
                            if cnt == 0 { continue; }
                            let lim2 = if cnt < cyc4_scan { cnt } else { cyc4_scan };
                            let limc = if cnt < cyc_scan { cnt } else { cyc_scan };
                            sp_m2[p] = ((1u16 << lim2) - 1) as u8;
                            sp_mc[p] = ((1u16 << limc) - 1) as u8;
                            let mut vm = 0u8;
                            for i in 0..cnt {
                                let node = sp_nodes[p * SP_SLOTS + i];
                                if $partition_mut_swap[node as usize] as usize == x {
                                    vm |= 1u8 << i;
                                }
                                if sp_rev_head[node as usize] < 0 {
                                    sp_rev_touched.push(node);
                                }
                                sp_rev_next[rev_len] = sp_rev_head[node as usize];
                                sp_rev_pair[rev_len] = p as u32;
                                sp_rev_slot[rev_len] = i as u8;
                                sp_rev_head[node as usize] = rev_len as i32;
                                rev_len += 1;
                            }
                            sp_vm[p] = vm;
                            if vm & sp_m2[p] != 0 {
                                r2 |= 1u64 << y;
                                sp_col2[y] |= 1u64 << x;
                            }
                            if vm & sp_mc[p] != 0 {
                                r8 |= 1u64 << y;
                                sp_col8[y] |= 1u64 << x;
                            }
                        }
                        sp_row2[x] = r2;
                        sp_row8[x] = r8;
                    }
                }

                let sw_t2 = if probe_on { Some(std::time::Instant::now()) } else { None };
                if cyc_scan > 0 {
                    if sw_fast {
                        for a in 0..np {
                            let bita = 1u64 << a;
                            let mut bfrom = 0usize;
                            loop {
                                let mb = sp_row8[a] & !bita & sp_hi(bfrom);
                                if mb == 0 { break; }
                                let b = mb.trailing_zeros() as usize;
                                bfrom = b + 1;
                                let bitb = 1u64 << b;
                                let idx_ab = a * np + b;
                                let cnt_ab = sp_cnt[idx_ab] as usize;
                                let mut cfrom = 0usize;
                                loop {
                                    let mc = sp_row8[b] & sp_col8[a] & !bita & !bitb & sp_hi(cfrom);
                                    if mc == 0 { break; }
                                    let c = mc.trailing_zeros() as usize;
                                    cfrom = c + 1;
                                    let idx_bc = b * np + c;
                                    let idx_ca = c * np + a;
                                    let cnt_bc = sp_cnt[idx_bc] as usize;
                                    let cnt_ca = sp_cnt[idx_ca] as usize;
                                        let sl_ab = std::cmp::min(cnt_ab, cyc_scan);
                                        let sl_bc = std::cmp::min(cnt_bc, cyc_scan);
                                        let sl_ca = std::cmp::min(cnt_ca, cyc_scan);
                                        'outer: for i in 0..sl_ab {
                                            let node_ab = sp_nodes[idx_ab * SP_SLOTS + i] as usize;
                                            let gain_ab = sp_gains[idx_ab * SP_SLOTS + i];
                                            if $partition_mut_swap[node_ab] as usize != a { continue; }
                                            if gain_ab + sp_gains[idx_bc * SP_SLOTS] + sp_gains[idx_ca * SP_SLOTS] <= 0 { break; }
                                            for j in 0..sl_bc {
                                                let node_bc = sp_nodes[idx_bc * SP_SLOTS + j] as usize;
                                                let gain_bc = sp_gains[idx_bc * SP_SLOTS + j];
                                                if $partition_mut_swap[node_bc] as usize != b { continue; }
                                                if node_bc == node_ab { continue; }
                                                if gain_ab + gain_bc + sp_gains[idx_ca * SP_SLOTS] <= 0 { break; }
                                                for k in 0..sl_ca {
                                                    let node_ca = sp_nodes[idx_ca * SP_SLOTS + k] as usize;
                                                    let gain_ca = sp_gains[idx_ca * SP_SLOTS + k];
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
                                                        if sw_fast {
                                                            let sp_upd = [node_ab, node_bc, node_ca];
                                                            for &un in sp_upd.iter() {
                                                                let mut e = sp_rev_head[un];
                                                                while e >= 0 {
                                                                    let ei = e as usize;
                                                                    let pp = sp_rev_pair[ei] as usize;
                                                                    let bit = 1u8 << sp_rev_slot[ei];
                                                                    let xx = pp / np;
                                                                    let yy = pp - xx * np;
                                                                    if $partition_mut_swap[un] as usize == xx {
                                                                        sp_vm[pp] |= bit;
                                                                    } else {
                                                                        sp_vm[pp] &= !bit;
                                                                    }
                                                                    if sp_vm[pp] & sp_m2[pp] != 0 {
                                                                        sp_row2[xx] |= 1u64 << yy;
                                                                        sp_col2[yy] |= 1u64 << xx;
                                                                    } else {
                                                                        sp_row2[xx] &= !(1u64 << yy);
                                                                        sp_col2[yy] &= !(1u64 << xx);
                                                                    }
                                                                    if sp_vm[pp] & sp_mc[pp] != 0 {
                                                                        sp_row8[xx] |= 1u64 << yy;
                                                                        sp_col8[yy] |= 1u64 << xx;
                                                                    } else {
                                                                        sp_row8[xx] &= !(1u64 << yy);
                                                                        sp_col8[yy] &= !(1u64 << xx);
                                                                    }
                                                                    e = sp_rev_next[ei];
                                                                }
                                                            }
                                                        }
                                                        break 'outer;
                                                    }
                                                }
                                            }
                                        }
                                }
                            }
                        }
                    } else {
                        for a in 0..np {
                            for b in 0..np {
                                if b == a { continue; }
                                let idx_ab = a * np + b;
                                let cnt_ab = sp_cnt[idx_ab] as usize;
                                if cnt_ab == 0 { continue; }
                                for c in 0..np {
                                    if c == a || c == b { continue; }
                                    let idx_bc = b * np + c;
                                    let idx_ca = c * np + a;
                                    let cnt_bc = sp_cnt[idx_bc] as usize;
                                    let cnt_ca = sp_cnt[idx_ca] as usize;
                                    if cnt_bc == 0 || cnt_ca == 0 { continue; }
                                        let sl_ab = std::cmp::min(cnt_ab, cyc_scan);
                                        let sl_bc = std::cmp::min(cnt_bc, cyc_scan);
                                        let sl_ca = std::cmp::min(cnt_ca, cyc_scan);
                                        'outer: for i in 0..sl_ab {
                                            let node_ab = sp_nodes[idx_ab * SP_SLOTS + i] as usize;
                                            let gain_ab = sp_gains[idx_ab * SP_SLOTS + i];
                                            if $partition_mut_swap[node_ab] as usize != a { continue; }
                                            if gain_ab + sp_gains[idx_bc * SP_SLOTS] + sp_gains[idx_ca * SP_SLOTS] <= 0 { break; }
                                            for j in 0..sl_bc {
                                                let node_bc = sp_nodes[idx_bc * SP_SLOTS + j] as usize;
                                                let gain_bc = sp_gains[idx_bc * SP_SLOTS + j];
                                                if $partition_mut_swap[node_bc] as usize != b { continue; }
                                                if node_bc == node_ab { continue; }
                                                if gain_ab + gain_bc + sp_gains[idx_ca * SP_SLOTS] <= 0 { break; }
                                                for k in 0..sl_ca {
                                                    let node_ca = sp_nodes[idx_ca * SP_SLOTS + k] as usize;
                                                    let gain_ca = sp_gains[idx_ca * SP_SLOTS + k];
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
                                                        if sw_fast {
                                                            let sp_upd = [node_ab, node_bc, node_ca];
                                                            for &un in sp_upd.iter() {
                                                                let mut e = sp_rev_head[un];
                                                                while e >= 0 {
                                                                    let ei = e as usize;
                                                                    let pp = sp_rev_pair[ei] as usize;
                                                                    let bit = 1u8 << sp_rev_slot[ei];
                                                                    let xx = pp / np;
                                                                    let yy = pp - xx * np;
                                                                    if $partition_mut_swap[un] as usize == xx {
                                                                        sp_vm[pp] |= bit;
                                                                    } else {
                                                                        sp_vm[pp] &= !bit;
                                                                    }
                                                                    if sp_vm[pp] & sp_m2[pp] != 0 {
                                                                        sp_row2[xx] |= 1u64 << yy;
                                                                        sp_col2[yy] |= 1u64 << xx;
                                                                    } else {
                                                                        sp_row2[xx] &= !(1u64 << yy);
                                                                        sp_col2[yy] &= !(1u64 << xx);
                                                                    }
                                                                    if sp_vm[pp] & sp_mc[pp] != 0 {
                                                                        sp_row8[xx] |= 1u64 << yy;
                                                                        sp_col8[yy] |= 1u64 << xx;
                                                                    } else {
                                                                        sp_row8[xx] &= !(1u64 << yy);
                                                                        sp_col8[yy] &= !(1u64 << xx);
                                                                    }
                                                                    e = sp_rev_next[ei];
                                                                }
                                                            }
                                                        }
                                                        break 'outer;
                                                    }
                                                }
                                            }
                                        }
                                }
                            }
                        }
                    }
                }
                if let Some(t) = sw_t2 {
                    sw_us_c3 += t.elapsed().as_micros() as u64;
                }

                let sw_t3 = if probe_on { Some(std::time::Instant::now()) } else { None };
                if cyc4_scan > 0 {
                    if sw_fast {
                        for a in 0..np {
                            let bita = 1u64 << a;
                            let mut bfrom = 0usize;
                            loop {
                                let mb = sp_row2[a] & !bita & sp_hi(bfrom);
                                if mb == 0 { break; }
                                let b = mb.trailing_zeros() as usize;
                                bfrom = b + 1;
                                let bitb = 1u64 << b;
                                let idx_ab = a * np + b;
                                let cnt_ab = sp_cnt[idx_ab] as usize;
                                let mut cfrom = 0usize;
                                loop {
                                    let mc = sp_row2[b] & !bita & !bitb & sp_hi(cfrom);
                                    if mc == 0 { break; }
                                    let c = mc.trailing_zeros() as usize;
                                    cfrom = c + 1;
                                    let bitc = 1u64 << c;
                                    let idx_bc = b * np + c;
                                    let cnt_bc = sp_cnt[idx_bc] as usize;
                                    let mut dfrom = 0usize;
                                    loop {
                                        let md = sp_row2[c] & sp_col2[a] & !bita & !bitb & !bitc & sp_hi(dfrom);
                                        if md == 0 { break; }
                                        let d = md.trailing_zeros() as usize;
                                        dfrom = d + 1;
                                        let idx_cd = c * np + d;
                                        let idx_da = d * np + a;
                                        let cnt_cd = sp_cnt[idx_cd] as usize;
                                        let cnt_da = sp_cnt[idx_da] as usize;
                                            let sl_ab = std::cmp::min(cnt_ab, cyc4_scan);
                                            let sl_bc = std::cmp::min(cnt_bc, cyc4_scan);
                                            let sl_cd = std::cmp::min(cnt_cd, cyc4_scan);
                                            let sl_da = std::cmp::min(cnt_da, cyc4_scan);
                                            'outer4: for i in 0..sl_ab {
                                                let node_ab = sp_nodes[idx_ab * SP_SLOTS + i] as usize;
                                                let gain_ab = sp_gains[idx_ab * SP_SLOTS + i];
                                                if $partition_mut_swap[node_ab] as usize != a { continue; }
                                                if gain_ab + sp_gains[idx_bc * SP_SLOTS] + sp_gains[idx_cd * SP_SLOTS] + sp_gains[idx_da * SP_SLOTS] <= 0 { break; }
                                                for j in 0..sl_bc {
                                                    let node_bc = sp_nodes[idx_bc * SP_SLOTS + j] as usize;
                                                    let gain_bc = sp_gains[idx_bc * SP_SLOTS + j];
                                                    if $partition_mut_swap[node_bc] as usize != b { continue; }
                                                    if node_bc == node_ab { continue; }
                                                    if gain_ab + gain_bc + sp_gains[idx_cd * SP_SLOTS] + sp_gains[idx_da * SP_SLOTS] <= 0 { break; }
                                                    for k in 0..sl_cd {
                                                        let node_cd = sp_nodes[idx_cd * SP_SLOTS + k] as usize;
                                                        let gain_cd = sp_gains[idx_cd * SP_SLOTS + k];
                                                        if $partition_mut_swap[node_cd] as usize != c { continue; }
                                                        if node_cd == node_ab || node_cd == node_bc { continue; }
                                                        if gain_ab + gain_bc + gain_cd + sp_gains[idx_da * SP_SLOTS] <= 0 { break; }
                                                        for m in 0..sl_da {
                                                            let node_da = sp_nodes[idx_da * SP_SLOTS + m] as usize;
                                                            let gain_da = sp_gains[idx_da * SP_SLOTS + m];
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
                                                                if sw_fast {
                                                                    let sp_upd = [node_ab, node_bc, node_cd, node_da];
                                                                    for &un in sp_upd.iter() {
                                                                        let mut e = sp_rev_head[un];
                                                                        while e >= 0 {
                                                                            let ei = e as usize;
                                                                            let pp = sp_rev_pair[ei] as usize;
                                                                            let bit = 1u8 << sp_rev_slot[ei];
                                                                            let xx = pp / np;
                                                                            let yy = pp - xx * np;
                                                                            if $partition_mut_swap[un] as usize == xx {
                                                                                sp_vm[pp] |= bit;
                                                                            } else {
                                                                                sp_vm[pp] &= !bit;
                                                                            }
                                                                            if sp_vm[pp] & sp_m2[pp] != 0 {
                                                                                sp_row2[xx] |= 1u64 << yy;
                                                                                sp_col2[yy] |= 1u64 << xx;
                                                                            } else {
                                                                                sp_row2[xx] &= !(1u64 << yy);
                                                                                sp_col2[yy] &= !(1u64 << xx);
                                                                            }
                                                                            if sp_vm[pp] & sp_mc[pp] != 0 {
                                                                                sp_row8[xx] |= 1u64 << yy;
                                                                                sp_col8[yy] |= 1u64 << xx;
                                                                            } else {
                                                                                sp_row8[xx] &= !(1u64 << yy);
                                                                                sp_col8[yy] &= !(1u64 << xx);
                                                                            }
                                                                            e = sp_rev_next[ei];
                                                                        }
                                                                    }
                                                                }
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
                    } else {
                        for a in 0..np {
                            for b in 0..np {
                                if b == a { continue; }
                                let idx_ab = a * np + b;
                                let cnt_ab = sp_cnt[idx_ab] as usize;
                                if cnt_ab == 0 { continue; }
                                for c in 0..np {
                                    if c == a || c == b { continue; }
                                    let idx_bc = b * np + c;
                                    let cnt_bc = sp_cnt[idx_bc] as usize;
                                    if cnt_bc == 0 { continue; }
                                    for d in 0..np {
                                        if d == a || d == b || d == c { continue; }
                                        let idx_cd = c * np + d;
                                        let idx_da = d * np + a;
                                        let cnt_cd = sp_cnt[idx_cd] as usize;
                                        let cnt_da = sp_cnt[idx_da] as usize;
                                        if cnt_cd == 0 || cnt_da == 0 { continue; }
                                            let sl_ab = std::cmp::min(cnt_ab, cyc4_scan);
                                            let sl_bc = std::cmp::min(cnt_bc, cyc4_scan);
                                            let sl_cd = std::cmp::min(cnt_cd, cyc4_scan);
                                            let sl_da = std::cmp::min(cnt_da, cyc4_scan);
                                            'outer4: for i in 0..sl_ab {
                                                let node_ab = sp_nodes[idx_ab * SP_SLOTS + i] as usize;
                                                let gain_ab = sp_gains[idx_ab * SP_SLOTS + i];
                                                if $partition_mut_swap[node_ab] as usize != a { continue; }
                                                if gain_ab + sp_gains[idx_bc * SP_SLOTS] + sp_gains[idx_cd * SP_SLOTS] + sp_gains[idx_da * SP_SLOTS] <= 0 { break; }
                                                for j in 0..sl_bc {
                                                    let node_bc = sp_nodes[idx_bc * SP_SLOTS + j] as usize;
                                                    let gain_bc = sp_gains[idx_bc * SP_SLOTS + j];
                                                    if $partition_mut_swap[node_bc] as usize != b { continue; }
                                                    if node_bc == node_ab { continue; }
                                                    if gain_ab + gain_bc + sp_gains[idx_cd * SP_SLOTS] + sp_gains[idx_da * SP_SLOTS] <= 0 { break; }
                                                    for k in 0..sl_cd {
                                                        let node_cd = sp_nodes[idx_cd * SP_SLOTS + k] as usize;
                                                        let gain_cd = sp_gains[idx_cd * SP_SLOTS + k];
                                                        if $partition_mut_swap[node_cd] as usize != c { continue; }
                                                        if node_cd == node_ab || node_cd == node_bc { continue; }
                                                        if gain_ab + gain_bc + gain_cd + sp_gains[idx_da * SP_SLOTS] <= 0 { break; }
                                                        for m in 0..sl_da {
                                                            let node_da = sp_nodes[idx_da * SP_SLOTS + m] as usize;
                                                            let gain_da = sp_gains[idx_da * SP_SLOTS + m];
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
                                                                if sw_fast {
                                                                    let sp_upd = [node_ab, node_bc, node_cd, node_da];
                                                                    for &un in sp_upd.iter() {
                                                                        let mut e = sp_rev_head[un];
                                                                        while e >= 0 {
                                                                            let ei = e as usize;
                                                                            let pp = sp_rev_pair[ei] as usize;
                                                                            let bit = 1u8 << sp_rev_slot[ei];
                                                                            let xx = pp / np;
                                                                            let yy = pp - xx * np;
                                                                            if $partition_mut_swap[un] as usize == xx {
                                                                                sp_vm[pp] |= bit;
                                                                            } else {
                                                                                sp_vm[pp] &= !bit;
                                                                            }
                                                                            if sp_vm[pp] & sp_m2[pp] != 0 {
                                                                                sp_row2[xx] |= 1u64 << yy;
                                                                                sp_col2[yy] |= 1u64 << xx;
                                                                            } else {
                                                                                sp_row2[xx] &= !(1u64 << yy);
                                                                                sp_col2[yy] &= !(1u64 << xx);
                                                                            }
                                                                            if sp_vm[pp] & sp_mc[pp] != 0 {
                                                                                sp_row8[xx] |= 1u64 << yy;
                                                                                sp_col8[yy] |= 1u64 << xx;
                                                                            } else {
                                                                                sp_row8[xx] &= !(1u64 << yy);
                                                                                sp_col8[yy] &= !(1u64 << xx);
                                                                            }
                                                                            e = sp_rev_next[ei];
                                                                        }
                                                                    }
                                                                }
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
                }
                if let Some(t) = sw_t3 {
                    sw_us_c4 += t.elapsed().as_micros() as u64;
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
            if probe_on {
                let us = probe_lap!();
                probe_out!(
                    "n={:08x} phase=swap{} rounds={} swaps={} us_gpu={} us_bp={} us_c3={} us_c4={} us={}\n",
                    probe_nonce, $max_rounds, sw_rounds, total_swaps,
                    sw_us_gpu, sw_us_bp, sw_us_c3, sw_us_c4, us
                );
            }
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
        // STAGE 2: precompute_edge_flags_10k + compute_refinement_moves_10k,
        // one launch, one software grid barrier.  The `d_num_valid_moves` D2H is
        // gone: nothing reads that buffer any more (the fused kernel does not even
        // take it), and the `num_valid_moves == 0` break is subsumed by the
        // `valid_moves.is_empty()` break a few lines below (see NOTES.md S2.3).
        // wave12 MERGE: the `reset_move_counters!()` above is wave11's, and is a
        // NO-OP at the default `reset_legacy = 0`.  It is kept at all six fused
        // sites so `reset_legacy = 1` still restores wave11's pair of H2Ds and the
        // A/B remains runnable; the fusion has made both buffers deader, not less
        // dead, so the knob cannot change a bit in either direction.
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_cfg.clone())?;
        }
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

    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_mirror);
        probe_out!(
            "n={:08x} phase=postswap30 km1={} us={}\n",
            probe_nonce, k, us
        );
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

    if probe_on {
        let us = probe_lap!();
        probe_out!(
            "n={:08x} phase=ils_in km1={} us={}\n",
            probe_nonce, best_connectivity, us
        );
    }

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

        let seed = (123456789u64 + (ils_iter as u64) * 987654321u64)
            .wrapping_add(run_off);

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

            {
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
                            if best_gain_vec[i] > best_gain || 
                               (best_gain_vec[i] == best_gain && best_deg_vec[i] < best_deg) {
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
                            if best_gain_vec[i] > best_gain || 
                               (best_gain_vec[i] == best_gain && best_deg_vec[i] < best_deg) {
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
            if probe_on {
                probe_quick_exec += 1;
            }
            reset_move_counters!();
            // STAGE 2: precompute_edge_flags_10k + compute_refinement_moves_10k,
            // one launch, one software grid barrier.  The `d_num_valid_moves` D2H is
            // gone: nothing reads that buffer any more (the fused kernel does not even
            // take it), and the `num_valid_moves == 0` break is subsumed by the
            // `valid_moves.is_empty()` break a few lines below (see NOTES.md S2.3).
            // wave12 MERGE: the `reset_move_counters!()` above is wave11's, and is a
            // NO-OP at the default `reset_legacy = 0`.  It is kept at all six fused
            // sites so `reset_legacy = 1` still restores wave11's pair of H2Ds and the
            // A/B remains runnable; the fusion has made both buffers deader, not less
            // dead, so the knob cannot change a bit in either direction.
            fused_calls = fused_calls.wrapping_add(1);
            let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
            unsafe {
                stream
                    .launch_builder(&fused_moves_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .arg(&mut d_move_priorities)
                    .arg(&mut d_grid_barrier)
                    .arg(&fused_target)
                    .launch(fused_cfg.clone())?;
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

            // ---- wave13 fix F1: the sort is dead on the bucket path -----
            let bucket_path = sel_legacy == 0
                && k_base >= valid_moves.len()
                && k_cand >= valid_moves.len();
            if !bucket_path {
            if k_cand > 1 {
                valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                valid_moves[..k_cand].sort_unstable_by(cmp);
            } else {
                valid_moves[..k_cand].sort_unstable_by(cmp);
            }
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
            if bucket_path {
                // ---- wave13 fix F1: counting-sort selection ------------------
                // Identical SET and identical ORDER to the scan in the `else`
                // branch; see fix/NOTES.md S2 for the proof.  The O(V log V)
                // tuple sort above is skipped entirely on this path.
                let mut pc = [0usize; 64];
                for &(_, key) in valid_moves.iter() {
                    pc[(key & 63) as usize] += 1;
                }
                let mut po = [0usize; 64];
                let mut acc = 0usize;
                for p in 0..64 {
                    po[p] = acc;
                    acc += pc[p];
                }
                let mut cur = po;
                for &(node, key) in valid_moves.iter() {
                    let t = (key & 63) as usize;
                    let i = cur[t];
                    cur[t] = i + 1;
                    by_buf[i] = ((((key as u32) ^ 0x7FFF_FFFFu32) as u64) << 32) | (node as u64);
                }
                let np64 = std::cmp::min(num_parts_usize, 64usize);
                let mut m = 0usize;
                for p in 0..np64 {
                    let c = pc[p];
                    if c == 0 {
                        continue;
                    }
                    let s = po[p];
                    let q = tgt_quota[p];
                    let keep = if c > q {
                        by_buf[s..s + c].select_nth_unstable(q - 1);
                        q
                    } else {
                        c
                    };
                    sel_keys[m..m + keep].copy_from_slice(&by_buf[s..s + keep]);
                    m += keep;
                }
                sel_keys[..m].sort_unstable();
                for &v in sel_keys[..m].iter() {
                    sorted_move_nodes.push(v as u32 as i32);
                    sorted_move_parts.push(((((v >> 32) as usize) & 63) ^ 63) as i32);
                }
            } else {
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
            }
            if bucket_path && sorted_move_nodes.is_empty() {
                valid_moves.sort_unstable_by(cmp);
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
            let rng_seed = 0xDEADu64.wrapping_add(ils_iter as u64)
                .wrapping_add(run_off)
                .wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
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

        if probe_on {
            probe_ils_iters += 1;
            let us = probe_lap!();
            probe_out!(
                "n={:08x} phase=ils iter={} km1={} best={} improved={} consensus={} accept={} quick_exec={} us={}\n",
                probe_nonce, ils_iter, new_connectivity, best_connectivity,
                improved as i32, use_consensus_next as i32, accept as i32,
                probe_quick_exec, us
            );
            probe_quick_exec = 0;
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
    
    if probe_on {
        let us = probe_lap!();
        probe_out!(
            "n={:08x} phase=ils_end km1={} iters={} budget={} us={}\n",
            probe_nonce, best_connectivity, probe_ils_iters, ils_iterations, us
        );
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
                // STAGE 2: precompute_edge_flags_10k + compute_refinement_moves_10k,
                // one launch, one software grid barrier.  The `d_num_valid_moves` D2H is
                // gone: nothing reads that buffer any more (the fused kernel does not even
                // take it), and the `num_valid_moves == 0` break is subsumed by the
                // `valid_moves.is_empty()` break a few lines below (see NOTES.md S2.3).
                // wave12 MERGE: the `reset_move_counters!()` above is wave11's, and is a
                // NO-OP at the default `reset_legacy = 0`.  It is kept at all six fused
                // sites so `reset_legacy = 1` still restores wave11's pair of H2Ds and the
                // A/B remains runnable; the fusion has made both buffers deader, not less
                // dead, so the knob cannot change a bit in either direction.
                fused_calls = fused_calls.wrapping_add(1);
                let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
                unsafe {
                    stream
                        .launch_builder(&fused_moves_kernel)
                        .arg(&(challenge.num_hyperedges as i32))
                        .arg(&(challenge.num_nodes as i32))
                        .arg(&(challenge.num_parts as i32))
                        .arg(&(challenge.max_part_size as i32))
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&d_partition)
                        .arg(&d_nodes_in_part)
                        .arg(&mut d_edge_flags_all)
                        .arg(&mut d_edge_flags_double)
                        .arg(&mut d_move_priorities)
                        .arg(&mut d_grid_barrier)
                        .arg(&fused_target)
                        .launch(fused_cfg.clone())?;
                }
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

    if probe_on {
        let us = probe_lap!();
        probe_out!(
            "n={:08x} phase=cross km1={} us={}\n",
            probe_nonce, best_connectivity, us
        );
    }

    let d_partition_final = stream.memcpy_stod(&best_partition_host)?;
    let d_nodes_in_part_final = stream.memcpy_stod(&best_nodes_in_part_host)?;
    d_partition = d_partition_final;
    d_nodes_in_part = d_nodes_in_part_final;
    partition_host_mirror.copy_from_slice(&best_partition_host);
    nodes_in_part_mirror.copy_from_slice(&best_nodes_in_part_host);
    
    for _ in 0..post_ils_polish {
        global_round += 1;
        if probe_on {
            probe_polish_exec += 1;
        }
        reset_move_counters!();
        // STAGE 2: precompute_edge_flags_10k + compute_refinement_moves_10k,
        // one launch, one software grid barrier.  The `d_num_valid_moves` D2H is
        // gone: nothing reads that buffer any more (the fused kernel does not even
        // take it), and the `num_valid_moves == 0` break is subsumed by the
        // `valid_moves.is_empty()` break a few lines below (see NOTES.md S2.3).
        // wave12 MERGE: the `reset_move_counters!()` above is wave11's, and is a
        // NO-OP at the default `reset_legacy = 0`.  It is kept at all six fused
        // sites so `reset_legacy = 1` still restores wave11's pair of H2Ds and the
        // A/B remains runnable; the fusion has made both buffers deader, not less
        // dead, so the knob cannot change a bit in either direction.
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_cfg.clone())?;
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

    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_mirror);
        probe_out!(
            "n={:08x} phase=polish km1={} rounds={} budget={} us={}\n",
            probe_nonce, k, probe_polish_exec, post_ils_polish, us
        );
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

    {
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
                    if best_gain_vec[i] > best_gain || 
                       (best_gain_vec[i] == best_gain && best_deg_vec[i] < best_deg) {
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
                    if best_gain_vec[i] > best_gain || 
                       (best_gain_vec[i] == best_gain && best_deg_vec[i] < best_deg) {
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
    refresh_host_mirrors_from_device!();

    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_mirror);
        probe_out!(
            "n={:08x} phase=postbal km1={} us={}\n",
            probe_nonce, k, us
        );
    }

    let post_balance_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 128) as usize)
        .unwrap_or(base_post_balance);

    for _ in 0..post_balance_rounds {
        global_round += 1;
        reset_move_counters!();
        // STAGE 2: precompute_edge_flags_10k + compute_refinement_moves_10k,
        // one launch, one software grid barrier.  The `d_num_valid_moves` D2H is
        // gone: nothing reads that buffer any more (the fused kernel does not even
        // take it), and the `num_valid_moves == 0` break is subsumed by the
        // `valid_moves.is_empty()` break a few lines below (see NOTES.md S2.3).
        // wave12 MERGE: the `reset_move_counters!()` above is wave11's, and is a
        // NO-OP at the default `reset_legacy = 0`.  It is kept at all six fused
        // sites so `reset_legacy = 1` still restores wave11's pair of H2Ds and the
        // A/B remains runnable; the fusion has made both buffers deader, not less
        // dead, so the knob cannot change a bit in either direction.
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_cfg.clone())?;
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

    {
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
        let polish_seed = (0x123456789ABCDEF0u64 ^ (global_round as u64))
            .wrapping_add(run_off);
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

    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition);
        probe_out!(
            "n={:08x} phase=final km1={} global_round={} us={}\n",
            probe_nonce, k, global_round, us
        );
    }
    // ==== END per-run body ====

        // ---- best-of-K selection on the TRUE km1 objective ----------------
        // `eval_km1` is exactly what `compute_connectivity_10k` accumulates and
        // what the challenge scores, recomputed on the FINAL partition -- the
        // ILS's own `best_connectivity` is a per-run incumbent measured before
        // the post-ILS polish / balance / swap / polish_exploration phases, so
        // it is not comparable across runs.
        let run_km1 = eval_km1(
            &partition,
            &hedge_offsets_host,
            &hyperedge_nodes_host,
            challenge.num_hyperedges as usize,
            num_parts_usize,
            &mut bok_seen,
        );
        // Strict `<` keeps the EARLIEST best run, so at runs == 1 (and whenever
        // no later run strictly wins) the saved solution is run 0's, i.e. the
        // base solver's, bit for bit.
        if run_km1 < bok_best_km1 {
            bok_best_km1 = run_km1;
            bok_best_part = partition.iter().map(|&x| x as u32).collect();
        }
    }

    // bok_runs >= 1, so bok_best_part is always populated here.
    save_solution(&Solution {
        partition: bok_best_part,
    })?;
    Ok(())
}
