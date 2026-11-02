use cudarc::{
    driver::{
        safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PinnedHostSlice, PushKernelArg,
    },
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

use super::params::{Knobs, Track, GROUP_SIZES};
use super::{
    decode_pick, decode_sorted, to_u32, Hp, Saver, FILTER_THREADS, LCG_INC, LCG_MUL,
    SELECT_THREADS, TRI100_DEFAULT_AT, TRI100_DEFAULT_BAR, TRI100_DEFAULT_ENABLED,
    TRI100_DEFAULT_HI, TRI100_DEFAULT_K, TRI100_DEFAULT_LAG, TRI100_DEFAULT_MARGIN,
    TRI100_DEFAULT_RHO, TRI100_DEFAULT_SURE,
};

const SCAN_LIMIT_SWAP: usize = 32;
const SCAN_LIMIT_CYCLE: usize = 8;
const MOVE_BLOCK: u32 = 64;

pub(super) fn execute_moves(
    partition: &mut [i32],
    nodes_in_part: &mut [i32],
    max_part_size: i32,
    move_nodes: &[i32],
    move_parts: &[i32],
    accepted: &mut Vec<i32>,
) -> i32 {
    accepted.clear();
    let mut executed = 0i32;
    for i in 0..move_nodes.len() {
        let node = move_nodes[i];
        let target = move_parts[i];
        if node < 0 || target < 0 || node as usize >= partition.len() {
            continue;
        }
        let current = partition[node as usize];
        if current >= 0
            && (current as usize) < nodes_in_part.len()
            && (target as usize) < nodes_in_part.len()
            && nodes_in_part[target as usize] < max_part_size
            && nodes_in_part[current as usize] > 1
        {
            partition[node as usize] = target;
            nodes_in_part[current as usize] -= 1;
            nodes_in_part[target as usize] += 1;
            executed += 1;
            accepted.push(node);
        }
    }
    executed
}

pub(super) fn perturb_random(
    partition: &mut [i32],
    nodes_in_part: &mut [i32],
    max_part_size: i32,
    strength: i32,
    seed: u64,
) {
    let num_nodes = partition.len();
    let num_parts = nodes_in_part.len();
    let mut state = seed;
    let mut moves_made = 0i32;
    let target_moves = (num_nodes as i32 * strength) / 100;
    for _ in 0..num_nodes {
        if moves_made >= target_moves {
            break;
        }
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        let node = (state % num_nodes as u64) as usize;
        let current = partition[node];
        if current < 0 || current >= num_parts as i32 || nodes_in_part[current as usize] <= 1 {
            continue;
        }
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        let target = (state % num_parts as u64) as i32;
        if target != current && nodes_in_part[target as usize] < max_part_size {
            partition[node] = target;
            nodes_in_part[current as usize] -= 1;
            nodes_in_part[target as usize] += 1;
            moves_made += 1;
        }
    }
}

fn perturb_guided(
    partition: &mut [i32],
    nodes_in_part: &mut [i32],
    max_part_size: i32,
    he_off: &[i32],
    he_nodes: &[i32],
    hedge_ids: &[i32],
) {
    let np = nodes_in_part.len().min(64);
    for &hedge in hedge_ids.iter() {
        if hedge < 0 {
            continue;
        }
        let start = he_off[hedge as usize] as usize;
        let end = he_off[hedge as usize + 1] as usize;
        if end - start <= 1 {
            continue;
        }
        let mut part_count = [0i32; 64];
        for k in start..end {
            let part = partition[he_nodes[k] as usize];
            if part >= 0 && (part as usize) < np {
                part_count[part as usize] += 1;
            }
        }
        let mut majority = 0usize;
        for p in 1..np {
            if part_count[p] > part_count[majority] {
                majority = p;
            }
        }
        let mut parts_present = part_count[..np].iter().filter(|&&c| c > 0).count() as i32;
        if parts_present <= 1 {
            continue;
        }
        for _ in 0..np {
            if parts_present <= 1 || nodes_in_part[majority] >= max_part_size {
                break;
            }
            let mut min_part = usize::MAX;
            let mut min_cnt = 0i32;
            for p in 0..np {
                let cnt = part_count[p];
                if p == majority || cnt <= 0 {
                    continue;
                }
                if min_part == usize::MAX || cnt < min_cnt || (cnt == min_cnt && p < min_part) {
                    min_part = p;
                    min_cnt = cnt;
                }
            }
            if min_part == usize::MAX {
                break;
            }
            let mut moved_any = false;
            for k in start..end {
                if part_count[min_part] <= 0 || nodes_in_part[majority] >= max_part_size {
                    break;
                }
                let node = he_nodes[k] as usize;
                if partition[node] != min_part as i32 || nodes_in_part[min_part] <= 1 {
                    continue;
                }
                partition[node] = majority as i32;
                nodes_in_part[min_part] -= 1;
                nodes_in_part[majority] += 1;
                part_count[min_part] -= 1;
                part_count[majority] += 1;
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
}

pub(super) fn degree_order(node_offsets: &[i32], num_nodes: usize) -> Vec<i32> {
    let degree = |v: usize| (node_offsets[v + 1] - node_offsets[v]).min(256) as usize;
    let mut cursor = [0usize; 257];
    for v in 0..num_nodes {
        cursor[degree(v)] += 1;
    }
    let mut prefix = 0usize;
    for c in cursor.iter_mut().rev() {
        let count = *c;
        *c = prefix;
        prefix += count;
    }
    let mut order = vec![0i32; num_nodes];
    for v in 0..num_nodes {
        let d = degree(v);
        order[cursor[d]] = v as i32;
        cursor[d] += 1;
    }
    order
}

fn high_hedge_ids(connectivity: &[i32], hedge_sizes: &[i32], count: usize) -> Vec<i32> {
    let mut impact: Vec<(i32, i32, i32)> = connectivity
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, hedge_sizes[i], i as i32))
        .collect();
    let cmp = |a: &(i32, i32, i32), b: &(i32, i32, i32)| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.2.cmp(&b.2))
    };
    if count < impact.len() {
        impact.select_nth_unstable_by(count, cmp);
        impact.truncate(count);
    }
    impact.sort_unstable_by(cmp);
    impact.into_iter().map(|t| t.2).collect()
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
    track: &'static Track,
) -> anyhow::Result<()> {
    let hp = Hp(hyperparameters);
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);
    let num_nodes = challenge.num_nodes as usize;
    let num_hedges = challenge.num_hyperedges as usize;
    let num_parts = challenge.num_parts as usize;
    let max_part_size = challenge.max_part_size as i32;

    let hyperedge_cluster_kernel = module.load_function(track.k_cluster)?;
    let compute_preferences_kernel = module.load_function(track.k_preferences)?;
    let execute_assignments_kernel = module.load_function(track.k_assignments)?;
    let edge_flags_kernel = module.load_function(track.k_edge_flags)?;
    let moves_name = track.k_moves;
    let moves_kernel = module.load_function(moves_name)?;
    let balance_kernel = module.load_function(track.k_balance)?;
    let compute_connectivity_kernel = module.load_function(track.k_connectivity)?;
    let reduce_connectivity_sum_kernel = module.load_function(track.k_reduce_conn)?;
    let compute_swap_gains_kernel = module.load_function(track.k_swap_gains)?;
    let choose_elite_per_hyperedge_kernel = module.load_function(track.k_choose_elite)?;
    let assign_from_elite_votes_kernel = module.load_function(track.k_assign_elite)?;
    let edge_flags_incr_kernel = match track.k_edge_flags_incr {
        Some(name) => Some(module.load_function(name)?),
        None => None,
    };
    let expand_kernel = match track.k_expand {
        Some(name) => Some(module.load_function(name)?),
        None => None,
    };
    let filt_stats_kernel = module.load_function("mica_filt_stats")?;
    let filt_pick_kernel = module.load_function("mica_filt_pick4")?;
    let sortsel_kernel = module.load_function("mica_sortsel")?;

    let stream = stream.fork()?;

    let grid = |threads: u32| LaunchConfig {
        grid_dim: ((threads + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg = grid(num_nodes as u32);
    let hedge_cfg = grid(num_hedges as u32);
    let one_thread_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let num_reduce_blocks = ((num_hedges as u32 + block_size * 2 - 1) / (block_size * 2)) as usize;
    let connectivity_reduce_cfg = LaunchConfig {
        grid_dim: (num_reduce_blocks as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4,
    };
    let filt_grid: u32 = hp.int("filt_g", 1, 256, track.filt_blocks as i64) as u32;
    let filt_cfg = LaunchConfig {
        grid_dim: (filt_grid, 1, 1),
        block_dim: (FILTER_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let sortsel_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (SELECT_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(num_hedges)?;
    let mut d_partition = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(num_parts)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_tabu = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_block_max = stream.alloc_zeros::<i32>(filt_grid as usize)?;
    let mut d_stats = stream.alloc_zeros::<i32>(filt_grid as usize * 5)?;
    let mut d_block_sel = stream.alloc_zeros::<i32>(filt_grid as usize)?;
    let mut d_filt_out = stream.alloc_zeros::<u64>(num_nodes + filt_grid as usize)?;
    let mut d_sort_buf = stream.alloc_zeros::<u64>(num_nodes)?;
    let mut d_sorted = stream.alloc_zeros::<u64>(num_nodes)?;
    let mut d_picks = stream.alloc_zeros::<u64>(2 + num_nodes)?;
    let mut picks_host: Vec<u64> = vec![0u64; 2 + num_nodes];
    let mut d_edge_flags = stream.alloc_zeros::<u64>(2 * num_hedges)?;
    let mut d_connectivity = stream.alloc_zeros::<i32>(num_hedges)?;
    let mut d_total_connectivity = stream.alloc_zeros::<i32>(num_reduce_blocks)?;
    let mut d_hedge_choice = stream.alloc_zeros::<i32>(num_hedges)?;
    let swap_buf_size = 3 * num_nodes;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;

    let mut knobs = Knobs::read(&hp, &track.d, num_hedges);
    let gl_kernel = "gl_k1";
    let (sort_mb_default, move_npt_default) = if knobs.pz {
        (track.pz_sort_mb, track.pz_move_npt)
    } else {
        (track.sort_mb, track.move_npt)
    };
    let sort_mb = hp.int("sort_mb", 0, 64, sort_mb_default) as u32;
    let configured_hp = super::params::configured_hp(&hp, &track.d, &mut knobs, move_npt_default)?;
    let hp = if configured_hp.is_some() {
        Hp(&configured_hp)
    } else {
        hp
    };
    let hl_blocks = hp.int("hl_g", 0, 4096, 32) as u32;
    let flg_blocks = hp.int("flg_g", 0, 4096, 32) as u32;
    let flags_cfg = if flg_blocks == 0 {
        grid(num_hedges as u32)
    } else {
        LaunchConfig {
            grid_dim: (
                flg_blocks
                    .min((num_hedges as u32 + block_size - 1) / block_size)
                    .max(1),
                1,
                1,
            ),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    let hlgrid = |threads: u32| -> LaunchConfig {
        if hl_blocks == 0 {
            grid(threads)
        } else {
            LaunchConfig {
                grid_dim: (
                    hl_blocks
                        .min((threads + block_size - 1) / block_size)
                        .max(1),
                    1,
                    1,
                ),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            }
        }
    };
    let mut cool_rounds: usize =
        hp.int("cool_rounds", 50, 400_000, knobs.refinement_rounds as i64) as usize;
    let mut pert_rounds: usize =
        hp.int("pert_rounds", 50, 400_000, knobs.refinement_rounds as i64) as usize;
    let cool_pow: u32 = hp.int("cool_pow", 1, 3, track.cool_pow as i64) as u32;
    let cool_pct: u64 = hp.int("cool_pct", 100, 1600, 100) as u64;
    let slack_scale: usize = hp.int("slack_scale", 1, 256, track.slack_scale as i64) as usize;
    let slack_early = 8usize * slack_scale;
    let slack_mid = 4usize * slack_scale;
    let slack_late = 2usize * slack_scale;
    let perturb_strength: i32 = hp.int("perturb_strength", 1, 40, 3) as i32;
    let post_balance_rounds =
        hp.int("post_refinement", 0, 128, knobs.effort.post_balance as i64) as usize;
    let extra_window = track.extra_window;
    let move_limit = knobs.move_limit;
    let neg_gain_thresh = knobs.neg_gain_thresh;
    let tabu_tenure = knobs.tabu_tenure;

    let mut part_to_part: Vec<Vec<(usize, i32)>> = vec![vec![]; num_parts * num_parts];
    let mut top_gain: Vec<i32> = vec![0i32; num_parts * num_parts];
    let mut swap_gains_host: Vec<i32> = vec![0i32; swap_buf_size];
    let mut partition_host_swap: Vec<i32> = vec![0i32; num_nodes];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; num_nodes];
    let mut partition_host_refine: Vec<i32> = vec![0i32; num_nodes];
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut nodes_in_part_host: Vec<i32> = vec![0i32; num_parts];

    let hedge_offsets_host = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let hyperedge_nodes_host = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let hedge_sizes_host: Vec<i32> = (0..num_hedges)
        .map(|h| hedge_offsets_host[h + 1] - hedge_offsets_host[h])
        .collect();

    let (d_large_hedge_ids, large_hedge_count) = if track.edge_flags_large_list {
        let mut ids: Vec<i32> = (0..num_hedges)
            .filter(|&h| hedge_sizes_host[h] > 32)
            .map(|h| h as i32)
            .collect();
        let count = ids.len() as i32;
        if ids.is_empty() {
            ids.push(0);
        }
        (Some(stream.memcpy_stod(&ids)?), count)
    } else {
        (None, 0)
    };

    macro_rules! launch_edge_flags {
        () => {{
            unsafe {
                if let Some(ids) = d_large_hedge_ids.as_ref() {
                    stream
                        .launch_builder(&edge_flags_kernel)
                        .arg(&(num_hedges as i32))
                        .arg(&(num_nodes as i32))
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&d_partition)
                        .arg(&mut d_edge_flags)
                        .arg(ids)
                        .arg(&large_hedge_count)
                        .arg(&0u64)
                        .launch(flags_cfg.clone())?;
                } else {
                    stream
                        .launch_builder(&edge_flags_kernel)
                        .arg(&(num_hedges as i32))
                        .arg(&(num_nodes as i32))
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&d_partition)
                        .arg(&mut d_edge_flags)
                        .arg(&0u64)
                        .launch(flags_cfg.clone())?;
                }
            }
        }};
    }

    unsafe {
        stream
            .launch_builder(&hyperedge_cluster_kernel)
            .arg(&(num_hedges as i32))
            .arg(&knobs.clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&mut d_hyperedge_clusters)
            .launch(hedge_cfg.clone())?;
    }

    unsafe {
        stream
            .launch_builder(&compute_preferences_kernel)
            .arg(&(num_nodes as i32))
            .arg(&(num_parts as i32))
            .arg(&knobs.clusters)
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

    let mut indices: Vec<usize> = (0..num_nodes).collect();
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
            .arg(&(num_nodes as i32))
            .arg(&(num_parts as i32))
            .arg(&max_part_size)
            .arg(&d_sorted_nodes)
            .arg(&d_sorted_parts)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }

    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
    stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

    let node_offsets = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let node_hedges = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;

    let gl_enabled = num_hedges == 100_000 && knobs.pz && hp.int("gf", 0, 1, 1) == 1;
    let gl_band = gl_enabled && hp.int("gf_band", 0, 1, super::GF_BAND) == 1;
    let mut saver = Saver::new(
        challenge,
        &hedge_offsets_host,
        &hyperedge_nodes_host,
        save_solution,
    );
    if gl_band {
        saver.best_partition = Some(super::round_robin(challenge));
    }
    saver.save(&to_u32(&partition_host_refine))?;

    let inferred = super::infer::starting_partition(
        &super::infer::Params::read(&hp),
        challenge,
        &module,
        &stream,
        &partition_host_refine,
    )?;
    if let Some(p) = inferred {
        partition_host_refine.copy_from_slice(&p);
        nodes_in_part_host.fill(0);
        for &b in partition_host_refine.iter() {
            nodes_in_part_host[b as usize] += 1;
        }
        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
    }

    let node_order = degree_order(&node_offsets, num_nodes);
    let d_node_order = stream.memcpy_stod(&node_order)?;
    let above = |t: i32| {
        (0..num_nodes)
            .filter(|&v| node_offsets[v + 1] - node_offsets[v] > t)
            .count()
    };
    let hub_count = above(track.hub_degree);
    let lanes4_count = above(track.lanes4_degree).max(hub_count);
    let lanes2_count = above(track.lanes2_degree).max(lanes4_count);
    let hub_count_i = hub_count as i32;
    let lanes4_i = lanes4_count as i32;
    let lanes2_i = lanes2_count as i32;
    let move_npt = hp.int("mv_npt", 1, 16, move_npt_default) as u32;
    let move_seq = if num_hedges == 200_000 {
        hp.int("mv_seq", 0, 1, track.move_seq) == 1
    } else {
        track.move_seq == 1
    };
    let move_ilp = track.move_ilp;
    debug_assert!(move_seq || (num_hedges == 200_000 && move_ilp == 2));
    let moves_kernel = if move_seq {
        moves_kernel
    } else {
        module.load_function(&format!("{}_ilp{}", moves_name, move_ilp))?
    };
    let ordinary_threads = ((num_nodes - lanes2_count).max(1) as u32 + move_npt - 1) / move_npt;
    let lane_blocks = (hub_count as u32 + MOVE_BLOCK / 8 - 1) / (MOVE_BLOCK / 8)
        + ((lanes4_count - hub_count) as u32 + MOVE_BLOCK / 4 - 1) / (MOVE_BLOCK / 4)
        + ((lanes2_count - lanes4_count) as u32 + MOVE_BLOCK / 2 - 1) / (MOVE_BLOCK / 2);
    let scalar_blocks = (ordinary_threads + MOVE_BLOCK - 1) / MOVE_BLOCK;
    let moves_cfg = LaunchConfig {
        grid_dim: (lane_blocks + scalar_blocks, 1, 1),
        block_dim: (MOVE_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    macro_rules! launch_moves {
        () => {{
            unsafe {
                stream
                    .launch_builder(&moves_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&d_edge_flags)
                    .arg(&mut d_move_priorities)
                    .arg(&d_node_order)
                    .arg(&hub_count_i)
                    .arg(&lanes4_i)
                    .arg(&lanes2_i)
                    .arg(&0u64)
                    .launch(moves_cfg.clone())?;
            }
        }};
    }

    let mut accepted: Vec<i32> = Vec::new();
    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(num_nodes);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(num_nodes);

    macro_rules! apply_moves {
        () => {{
            let executed = execute_moves(
                &mut partition_host_refine,
                &mut nodes_in_part_host,
                max_part_size,
                &sorted_move_nodes,
                &sorted_move_parts,
                &mut accepted,
            );
            if executed > 0 {
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            }
            executed
        }};
    }

    macro_rules! load_fallback_moves {
        ($k_base:expr, $k_cand:expr) => {{
            let take = std::cmp::min($k_cand - $k_base, $k_base);
            stream.memcpy_dtoh(
                &d_sorted.slice($k_base..$k_base + take),
                &mut picks_host[0..take],
            )?;
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for &w in picks_host[..take].iter() {
                let (node, part) = decode_sorted(w);
                sorted_move_nodes.push(node);
                sorted_move_parts.push(part);
            }
        }};
    }

    let mut node_tabu_until: Vec<i32> = vec![0; num_nodes];
    let mut tabu_dirty = true;
    let chunk_i: i32 = ((num_nodes as u32 + filt_grid - 1) / filt_grid) as i32;
    let mut d_h4 = stream.alloc_zeros::<i32>(1024)?;
    macro_rules! device_pick {
        ($tabu_round:expr, $lot_min:expr, $rng0:expr, $lcg_c:expr, $num:expr, $dens:expr,
         $limit:expr, $extra:expr, $slack:expr) => {{
            if tabu_dirty {
                stream.memcpy_htod(&node_tabu_until, &mut d_tabu)?;
                tabu_dirty = false;
            }
            let n_i = num_nodes as i32;
            let tabu_round_i: i32 = $tabu_round;
            let lot_min_i: i32 = $lot_min;
            let rng0_u: u64 = $rng0;
            let lcg_c_u: u64 = $lcg_c;
            let num_u: u64 = $num;
            let dens: [u64; 4] = $dens;
            let g_i = filt_grid as i32;
            let np_i = num_parts as i32;
            let slack_i = ($slack as usize).min(i32::MAX as usize) as i32;
            let limit_i = ($limit as usize).min(i32::MAX as usize) as i32;
            let extra_i = ($extra as usize).min(i32::MAX as usize) as i32;
            unsafe {
                stream
                    .launch_builder(&filt_stats_kernel)
                    .arg(&d_move_priorities)
                    .arg(&d_tabu)
                    .arg(&n_i)
                    .arg(&tabu_round_i)
                    .arg(&lot_min_i)
                    .arg(&mut d_block_max)
                    .arg(&mut d_stats)
                    .arg(&0u64)
                    .launch(filt_cfg.clone())?;
                stream
                    .launch_builder(&filt_pick_kernel)
                    .arg(&d_move_priorities)
                    .arg(&d_tabu)
                    .arg(&n_i)
                    .arg(&tabu_round_i)
                    .arg(&lot_min_i)
                    .arg(&d_block_max)
                    .arg(&d_stats)
                    .arg(&rng0_u)
                    .arg(&lcg_c_u)
                    .arg(&num_u)
                    .arg(&dens[0])
                    .arg(&dens[1])
                    .arg(&dens[2])
                    .arg(&dens[3])
                    .arg(&mut d_filt_out)
                    .arg(&mut d_block_sel)
                    .arg(&mut d_h4)
                    .arg(&0u64)
                    .launch(filt_cfg.clone())?;
                stream
                    .launch_builder(&sortsel_kernel)
                    .arg(&d_filt_out)
                    .arg(&d_block_sel)
                    .arg(&g_i)
                    .arg(&chunk_i)
                    .arg(&mut d_sort_buf)
                    .arg(&mut d_sorted)
                    .arg(&d_nodes_in_part)
                    .arg(&np_i)
                    .arg(&max_part_size)
                    .arg(&slack_i)
                    .arg(&limit_i)
                    .arg(&extra_i)
                    .arg(&mut d_picks)
                    .arg(&mut d_h4)
                    .arg(&0u64)
                    .launch(sortsel_cfg.clone())?;
            }
            let head = 2 + track.picks_prefix.min(num_nodes);
            stream.memcpy_dtoh(&d_picks.slice(0..head), &mut picks_host[0..head])?;
            let m = (picks_host[1] as usize).min(num_nodes);
            let k = ((picks_host[0] & 0xFFFF_FFFF) as usize).min(m);
            if 2 + k > head {
                stream.memcpy_dtoh(&d_picks.slice(head..2 + k), &mut picks_host[head..2 + k])?;
            }
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for &w in picks_host[2..2 + k].iter() {
                let (node, part) = decode_pick(w);
                sorted_move_nodes.push(node);
                sorted_move_parts.push(part);
            }
            m
        }};
    }

    let mut moved_nodes: Vec<i32> = Vec::new();
    let mut incr_pairs: Vec<i32> = Vec::new();
    let mut d_incr_pairs = match edge_flags_incr_kernel {
        Some(_) => Some(stream.alloc_zeros::<i32>(2 * num_nodes.max(1))?),
        None => None,
    };
    let mut d_incr_list = match edge_flags_incr_kernel {
        Some(_) => Some(stream.alloc_zeros::<i32>(node_hedges.len().max(1))?),
        None => None,
    };
    let mut d_edge_stamp = stream.alloc_zeros::<i32>(num_hedges)?;
    let mut flags_need_full = true;
    let mut stagnant_rounds = 0usize;

    let grp_n = hp.choice("grp_n", &GROUP_SIZES, track.grp_n) as usize;
    let gc = hp.int("gc", 0, 1, 0) != 0;
    let mut gl_run = None;
    let mut gl_hp = None;
    let mut gl_batch = super::GF_BATCH as usize;
    let tri100 = 'gl_main: {
        let mut tri100 = if matches!(num_hedges, 20_000 | 50_000) {
            Tri100::small(&hp, knobs.refinement_rounds, num_hedges)
        } else if matches!(num_hedges, 100_000 | 200_000) && knobs.pz {
            if gl_enabled {
                let gl_cfg = if grp_n == 0 {
                    Err("g")
                } else {
                    super::grouped::gl_a1(
                        challenge,
                        &module,
                        prop,
                        &partition_host_refine,
                        &nodes_in_part_host,
                        gl_kernel,
                    )
                };
                match gl_cfg {
                    Ok(gl_cfg) => {
                        gl_batch = (hp.int("gf_batch", 1, 2_048, super::GF_BATCH) as usize)
                            .min(i32::MAX as usize / num_nodes)
                            .min((u32::MAX as usize / gl_cfg.grid_dim.0 as usize - 2) / 11);
                        debug_assert_eq!(gl_a5(100_000, 0, &track.d), 100_000);
                        debug_assert_eq!(gl_a5(100_000, 900_000_000_000, &track.d), 60_000);
                        debug_assert_eq!(gl_a5(500_000, 1, &track.d), 50);
                        let gl_permille =
                            hp.int("gf_permille", 500, 3_000, super::GF_ROUNDS_PERMILLE) as u64;
                        let gl_requested = (knobs.refinement_requested as u64 * gl_permille / 1_000)
                            .max(50)
                            .min(super::grouped::GL_MAX_ROUNDS as u64) as usize;
                        knobs.refinement_rounds =
                            gl_a5(gl_requested, knobs.refinement_fuel, &track.d);
                        let gl_raw = Hp(hyperparameters);
                        knobs.ils_iterations = 0;
                        knobs.post_ils_polish =
                            gl_raw.int("post_ils_polish", 0, 500, knobs.effort.polish as i64)
                                as usize;
                        knobs.swap_scale =
                            gl_raw.int("swap_scale", 0, 400, track.d.swap_scale as i64) as u64;
                        gl_hp = super::params::configured_hp(
                            &gl_raw,
                            &track.d,
                            &mut knobs,
                            move_npt_default,
                        )?;
                        knobs.ils_iterations = 0;
                        let gl_h = Hp(&gl_hp);
                        let mut gl_tri = Gl1::new(&gl_h, knobs.refinement_rounds);
                        super::grouped::gl_a2(
                            challenge,
                            &module,
                            &stream,
                            gl_cfg.clone(),
                            gl_kernel,
                            0,
                            0,
                            gl_batch,
                            knobs.move_limit,
                            &mut d_partition,
                            &mut d_nodes_in_part,
                            &mut d_tabu,
                            &mut d_edge_flags,
                            &mut d_move_priorities,
                            &mut partition_host_refine,
                            &mut nodes_in_part_host,
                            &mut node_tabu_until,
                            &node_offsets,
                            &node_hedges,
                            &hedge_sizes_host,
                            &mut gl_tri,
                            &mut saver,
                        )?;
                        tabu_dirty = false;
                        if gl_tri.cut {
                            return Ok(());
                        }
                        gl_run = Some((gl_cfg, gl_tri));
                        break 'gl_main None;
                    }
                    Err(_) => {
                        saver.best_partition = None;
                    }
                }
            }
            let enabled = if num_hedges == 200_000 {
                super::TRI200_DEFAULT_ENABLED
            } else {
                TRI100_DEFAULT_ENABLED
            };
            if hp.int("tri", 0, 1, enabled) == 1 {
                Some(Tri100::new(&hp, knobs.refinement_rounds, num_hedges))
            } else {
                None
            }
        } else {
            None
        };
        if grp_n != 0 {
            if super::grouped::run(
                challenge,
                &module,
                &stream,
                track,
                &mut knobs,
                &Hp(hyperparameters),
                grp_n,
                gc,
                cool_rounds,
                cool_pow,
                cool_pct,
                pert_rounds,
                slack_scale,
                filt_grid,
                sort_mb,
                hlgrid(num_nodes as u32),
                moves_cfg.clone(),
                scalar_blocks,
                move_seq,
                move_ilp,
                &d_node_order,
                hub_count_i,
                lanes4_i,
                lanes2_i,
                &mut d_partition,
                &mut d_nodes_in_part,
                &mut d_tabu,
                &mut d_edge_flags,
                &mut d_move_priorities,
                &mut d_block_max,
                &mut d_stats,
                &mut d_block_sel,
                &mut d_h4,
                &mut d_filt_out,
                &mut d_sort_buf,
                &mut d_sorted,
                &mut d_picks,
                &mut partition_host_refine,
                &mut nodes_in_part_host,
                &mut node_tabu_until,
                &node_offsets,
                &node_hedges,
                &mut tri100,
                &mut saver,
            )? {
                return Ok(());
            }
            tabu_dirty = false;
        }

        if grp_n == 0 {
            for round in 0..knobs.refinement_rounds {
                if round >= knobs.refinement_rounds {
                    break;
                }
                if let Some(tri) = tri100.as_mut() {
                    tri.round = round + 1;
                    tri.launched();
                }
                let mut did_incr = false;
                if let (Some(kernel), Some(expand), Some(d_list), Some(d_pairs), false) = (
                    edge_flags_incr_kernel.as_ref(),
                    expand_kernel.as_ref(),
                    d_incr_list.as_mut(),
                    d_incr_pairs.as_mut(),
                    flags_need_full,
                ) {
                    incr_pairs.clear();
                    let mut listed = 0usize;
                    for &v in moved_nodes.iter() {
                        incr_pairs.push(v);
                        incr_pairs.push(listed as i32);
                        listed +=
                            (node_offsets[v as usize + 1] - node_offsets[v as usize]) as usize;
                    }
                    if listed < num_hedges {
                        if listed > 0 {
                            let n_pairs = moved_nodes.len();
                            stream
                                .memcpy_htod(&incr_pairs, &mut d_pairs.slice_mut(0..2 * n_pairs))?;
                            let stamp = round as i32 + 1;
                            unsafe {
                                stream
                                    .launch_builder(expand)
                                    .arg(&(n_pairs as i32))
                                    .arg(&*d_pairs)
                                    .arg(&challenge.d_node_hyperedges)
                                    .arg(&challenge.d_node_offsets)
                                    .arg(&mut *d_list)
                                    .arg(&0u64)
                                    .launch(hlgrid(n_pairs as u32))?;
                                stream
                                    .launch_builder(kernel)
                                    .arg(&(listed as i32))
                                    .arg(&*d_list)
                                    .arg(&stamp)
                                    .arg(&mut d_edge_stamp)
                                    .arg(&(num_hedges as i32))
                                    .arg(&(num_nodes as i32))
                                    .arg(&challenge.d_hyperedge_nodes)
                                    .arg(&challenge.d_hyperedge_offsets)
                                    .arg(&d_partition)
                                    .arg(&mut d_edge_flags)
                                    .arg(&0u64)
                                    .launch(hlgrid(listed as u32))?;
                            }
                        }
                        did_incr = true;
                    }
                }
                if !did_incr {
                    launch_edge_flags!();
                    flags_need_full = false;
                }
                launch_moves!();

                let rounds_left = cool_rounds.saturating_sub(round) as u64;
                let cool_span = cool_rounds as u64;
                let (probability_num, probability_base) = match cool_pow {
                    1 => (rounds_left, cool_span),
                    3 => (
                        rounds_left
                            .saturating_mul(rounds_left)
                            .saturating_mul(rounds_left),
                        cool_span
                            .saturating_mul(cool_span)
                            .saturating_mul(cool_span),
                    ),
                    _ => (
                        rounds_left.saturating_mul(rounds_left),
                        cool_span.saturating_mul(cool_span),
                    ),
                };
                let scaled_base = probability_base.saturating_mul(100) / cool_pct;
                let probability_den_by_penalty: [u64; 4] = [
                    scaled_base,
                    scaled_base.saturating_mul(6),
                    scaled_base.saturating_mul(11),
                    scaled_base.saturating_mul(16),
                ];

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

                let m = device_pick!(
                    round as i32,
                    -3,
                    123456789u64.wrapping_add(round as u64),
                    LCG_INC,
                    probability_num,
                    probability_den_by_penalty,
                    adaptive_limit,
                    extra_window,
                    slack
                );
                if m == 0 {
                    break;
                }
                let k_base = m.min(adaptive_limit);
                let k_cand = m.min(k_base.saturating_add(extra_window));

                let mut moves_executed = apply_moves!();

                if moves_executed == 0 && k_cand > k_base {
                    let fail_mark_len =
                        std::cmp::min(sorted_move_nodes.len(), track.tabu_fail_mark_len);
                    for &node in sorted_move_nodes.iter().take(fail_mark_len) {
                        node_tabu_until[node as usize] = (round + track.tabu_fail_tenure) as i32;
                        tabu_dirty = true;
                    }
                    load_fallback_moves!(k_base, k_cand);
                    if !sorted_move_nodes.is_empty() {
                        moves_executed = apply_moves!();
                    }
                }

                moved_nodes.clear();
                moved_nodes.extend_from_slice(&accepted);

                if moves_executed > 0 {
                    let until = (round + tabu_tenure) as i32;
                    for &node in accepted.iter() {
                        node_tabu_until[node as usize] = until;
                    }
                    tabu_dirty = true;
                }

                if moves_executed == 0 {
                    stagnant_rounds += 1;
                    if stagnant_rounds >= 3 && round < pert_rounds.saturating_sub(50) {
                        let mini_seed = 987654321u64 + (round as u64) * 123456789u64;
                        perturb_random(
                            &mut partition_host_refine,
                            &mut nodes_in_part_host,
                            max_part_size,
                            3,
                            mini_seed,
                        );
                        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                        flags_need_full = true;
                        stagnant_rounds = 0;
                    } else if stagnant_rounds > knobs.max_stagnant_rounds {
                        break;
                    }
                } else {
                    stagnant_rounds = 0;
                }
                if let Some(tri) = tri100.as_mut() {
                    if tri.finish_pending(
                        false,
                        &d_partition,
                        challenge,
                        &stream,
                        &node_offsets,
                        &node_hedges,
                        &mut saver,
                    )? {
                        return Ok(());
                    }
                    if tri.check(
                        round + 1,
                        &d_partition,
                        challenge,
                        &module,
                        &stream,
                        &mut saver,
                    )? {
                        return Ok(());
                    }
                    if let Some((cool, pert)) =
                        tri.reduce_budget(&Hp(hyperparameters), track, &mut knobs)?
                    {
                        cool_rounds = cool;
                        pert_rounds = pert;
                    }
                }
            }
        }
        if let Some(tri) = tri100.as_mut() {
            if tri.check(
                tri.round,
                &d_partition,
                challenge,
                &module,
                &stream,
                &mut saver,
            )? {
                return Ok(());
            }
            if tri.finish_pending(
                true,
                &d_partition,
                challenge,
                &stream,
                &node_offsets,
                &node_hedges,
                &mut saver,
            )? {
                return Ok(());
            }
            tri.reduce_budget(&Hp(hyperparameters), track, &mut knobs)?;
        }
        tri100
    };
    let hp = if gl_hp.is_some() {
        Hp(&gl_hp)
    } else {
        match tri100.as_ref() {
            Some(tri) if tri.budget_hp.is_some() => Hp(&tri.budget_hp),
            _ => hp,
        }
    };
    macro_rules! do_swap_phase {
        ($max_rounds:expr) => {{
            let np = num_parts;
            let mut prev_swap_count = usize::MAX;
            let mut stagnant = 0usize;
            stream.memcpy_dtoh(&d_partition, &mut partition_host_swap)?;
            let swap_rounds = ((($max_rounds as u64) * knobs.swap_scale) / 100) as usize;
            for _ in 0..swap_rounds {
                launch_edge_flags!();
                unsafe {
                    stream
                        .launch_builder(&compute_swap_gains_kernel)
                        .arg(&(num_nodes as i32))
                        .arg(&(np as i32))
                        .arg(&neg_gain_thresh)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&d_partition)
                        .arg(&d_edge_flags)
                        .arg(&mut d_swap_gains)
                        .launch(cfg.clone())?;
                }
                stream.memcpy_dtoh(&d_swap_gains, &mut swap_gains_host)?;

                for v in part_to_part.iter_mut() {
                    v.clear();
                }
                for node in 0..num_nodes {
                    let src = partition_host_swap[node] as usize;
                    if src >= np {
                        continue;
                    }
                    for slot in 0..3usize {
                        let val = swap_gains_host[node * 3 + slot];
                        if val == 0 {
                            continue;
                        }
                        let tgt = (val & 0xFFFF) as usize;
                        let gain = ((val >> 16) as i16) as i32;
                        if tgt < np && tgt != src {
                            part_to_part[src * np + tgt].push((node, gain));
                        }
                    }
                }

                partition_mut_swap.copy_from_slice(&partition_host_swap);
                let mut swap_count = 0usize;

                for a in 0..np {
                    for b in (a + 1)..np {
                        let idx_ab = a * np + b;
                        let idx_ba = b * np + a;
                        part_to_part[idx_ab]
                            .sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
                        part_to_part[idx_ba]
                            .sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
                        if part_to_part[idx_ab].is_empty() || part_to_part[idx_ba].is_empty() {
                            continue;
                        }
                        let lab_len = part_to_part[idx_ab].len();
                        let lba_len = part_to_part[idx_ba].len();
                        used_ba_buf.clear();
                        used_ba_buf.resize(lba_len, false);
                        for i in 0..lab_len {
                            let (node_a, gain_a) = part_to_part[idx_ab][i];
                            if partition_mut_swap[node_a] as usize != a {
                                continue;
                            }
                            let mut best_combined = 0i32;
                            let mut best_j = usize::MAX;
                            for j in 0..std::cmp::min(lba_len, SCAN_LIMIT_SWAP) {
                                if used_ba_buf[j] {
                                    continue;
                                }
                                let (node_b, gain_b) = part_to_part[idx_ba][j];
                                if partition_mut_swap[node_b] as usize != b {
                                    continue;
                                }
                                let combined = gain_a + gain_b;
                                if combined > best_combined {
                                    best_combined = combined;
                                    best_j = j;
                                }
                                break;
                            }
                            if best_j < lba_len && best_combined > 0 {
                                let (node_b, _) = part_to_part[idx_ba][best_j];
                                partition_mut_swap[node_a] = b as i32;
                                partition_mut_swap[node_b] = a as i32;
                                used_ba_buf[best_j] = true;
                                swap_count += 1;
                            }
                        }
                    }
                }

                for i in 0..np * np {
                    top_gain[i] = match part_to_part[i].first() {
                        Some(&(_, g)) => g,
                        None => i32::MIN / 4,
                    };
                }
                for a in 0..np {
                    for b in 0..np {
                        if b == a {
                            continue;
                        }
                        let idx_ab = a * np + b;
                        if part_to_part[idx_ab].is_empty() {
                            continue;
                        }
                        let need = -top_gain[idx_ab];
                        for c in 0..np {
                            if c == a || c == b {
                                continue;
                            }
                            let idx_bc = b * np + c;
                            let idx_ca = c * np + a;
                            if top_gain[idx_bc] + top_gain[idx_ca] <= need {
                                continue;
                            }
                            let sl_ab = std::cmp::min(part_to_part[idx_ab].len(), SCAN_LIMIT_CYCLE);
                            let sl_bc = std::cmp::min(part_to_part[idx_bc].len(), SCAN_LIMIT_CYCLE);
                            let sl_ca = std::cmp::min(part_to_part[idx_ca].len(), SCAN_LIMIT_CYCLE);
                            'outer: for i in 0..sl_ab {
                                let (node_ab, gain_ab) = part_to_part[idx_ab][i];
                                if partition_mut_swap[node_ab] as usize != a {
                                    continue;
                                }
                                if gain_ab + top_gain[idx_bc] + top_gain[idx_ca] <= 0 {
                                    break;
                                }
                                for j in 0..sl_bc {
                                    let (node_bc, gain_bc) = part_to_part[idx_bc][j];
                                    if partition_mut_swap[node_bc] as usize != b {
                                        continue;
                                    }
                                    if node_bc == node_ab {
                                        continue;
                                    }
                                    if gain_ab + gain_bc + top_gain[idx_ca] <= 0 {
                                        break;
                                    }
                                    for t in 0..sl_ca {
                                        let (node_ca, gain_ca) = part_to_part[idx_ca][t];
                                        if partition_mut_swap[node_ca] as usize != c {
                                            continue;
                                        }
                                        if node_ca == node_ab || node_ca == node_bc {
                                            continue;
                                        }
                                        if gain_ab + gain_bc + gain_ca > 0 {
                                            partition_mut_swap[node_ab] = b as i32;
                                            partition_mut_swap[node_bc] = c as i32;
                                            partition_mut_swap[node_ca] = a as i32;
                                            swap_count += 1;
                                            break 'outer;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                if swap_count == 0 {
                    break;
                }
                stream.memcpy_htod(&partition_mut_swap, &mut d_partition)?;
                partition_host_swap.copy_from_slice(&partition_mut_swap);
                if swap_count >= prev_swap_count {
                    stagnant += 1;
                    if stagnant >= 3 {
                        break;
                    }
                } else {
                    stagnant = 0;
                }
                prev_swap_count = swap_count;
            }
        }};
    }

    let skip = hp.int("skip", 0, 127, 0) as u32;
    let mut gl_passes = 0usize;
    let mut gl_ext = 0u64;
    for gl_pass in 0..=4usize {
        let gl_before = saver.best_km1;
        if gl_pass > 0 {
            let (gl_cfg, gl_tri) = gl_run.as_mut().unwrap();
            let requested = (gl_tri.total as u64 * gl_ext / 1_000) as usize;
            let remaining = super::fuel_remaining();
            let Some(e) = gl_a6(requested, remaining, &track.d) else {
                if gl_pass >= gl_passes {
                    break;
                }
                continue;
            };
            let gl_e = e;
            let best = saver
                .best_partition
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing saved partition"))?;
            anyhow::ensure!(best.len() == num_nodes, "invalid saved partition length");
            nodes_in_part_host.fill(0);
            for (v, &b) in best.iter().enumerate() {
                partition_host_refine[v] = b as i32;
                nodes_in_part_host[b as usize] += 1;
            }
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            node_tabu_until.fill(0);
            tabu_dirty = true;
            if gl_e > 0 {
                super::grouped::gl_a2(
                    challenge,
                    &module,
                    &stream,
                    gl_cfg.clone(),
                    gl_kernel,
                    gl_tri.total - gl_e,
                    gl_pass as u64,
                    gl_batch,
                    knobs.move_limit,
                    &mut d_partition,
                    &mut d_nodes_in_part,
                    &mut d_tabu,
                    &mut d_edge_flags,
                    &mut d_move_priorities,
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &mut node_tabu_until,
                    &node_offsets,
                    &node_hedges,
                    &hedge_sizes_host,
                    gl_tri,
                    &mut saver,
                )?;
                tabu_dirty = false;
            }
            knobs.ils_iterations = 0;
        }
        do_swap_phase!(100);
        partition_host_refine.copy_from_slice(&partition_host_swap);

        for _ in 0..30 {
            launch_edge_flags!();
            launch_moves!();
            let m = device_pick!(
                i32::MAX,
                1,
                0u64,
                1u64,
                0u64,
                [0u64; 4],
                move_limit / 2,
                extra_window / 2,
                slack_mid
            );
            if m == 0 {
                break;
            }
            if apply_moves!() == 0 {
                break;
            }
        }

        do_swap_phase!(50);

        if matches!(num_hedges, 20_000 | 50_000)
            && hp.int("sp_ils0", 0, 1, 1) == 1
            && knobs.ils_iterations == 0
        {
            debug_assert_eq!(knobs.ils_iterations, 0);
            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
        } else {
            unsafe {
                stream
                    .launch_builder(&compute_connectivity_kernel)
                    .arg(&(num_hedges as i32))
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
            let mut high_hedge_ids_host =
                high_hedge_ids(&conn_vec, &hedge_sizes_host, knobs.num_high_hedges);

            let pool_size = std::cmp::max(2usize, knobs.ils_iterations);
            let mut elite_scores: Vec<i32> = vec![i32::MAX; pool_size];
            let mut elite_flat_host: Vec<i32> = vec![0i32; pool_size * num_nodes];
            elite_scores[0] = best_connectivity;
            elite_flat_host[..num_nodes].copy_from_slice(&best_partition_host);
            let mut elite_count: usize = 1;
            let mut d_elite_flat = stream.alloc_zeros::<i32>(pool_size * num_nodes)?;
            let mut use_consensus_next = false;

            for ils_iter in 0..knobs.ils_iterations {
                stream.memcpy_htod(&best_partition_host, &mut d_partition)?;
                stream.memcpy_htod(&best_nodes_in_part_host, &mut d_nodes_in_part)?;
                partition_host_refine.copy_from_slice(&best_partition_host);
                nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);

                launch_edge_flags!();

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
                            .launch_builder(&choose_elite_per_hyperedge_kernel)
                            .arg(&(num_hedges as i32))
                            .arg(&(num_nodes as i32))
                            .arg(&(num_parts as i32))
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
                            .arg(&(num_nodes as i32))
                            .arg(&(num_parts as i32))
                            .arg(&d_elite_flat)
                            .arg(&d_hedge_choice)
                            .arg(&challenge.d_node_hyperedges)
                            .arg(&challenge.d_node_offsets)
                            .arg(&d_edge_flags)
                            .arg(&mut d_partition)
                            .launch(cfg.clone())?;
                    }

                    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
                    nodes_in_part_host.fill(0);
                    for &p in partition_host_refine.iter() {
                        if p >= 0 && (p as usize) < num_parts {
                            nodes_in_part_host[p as usize] += 1;
                        }
                    }
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;

                    unsafe {
                        stream
                            .launch_builder(&balance_kernel)
                            .arg(&(num_nodes as i32))
                            .arg(&(num_parts as i32))
                            .arg(&1i32)
                            .arg(&max_part_size)
                            .arg(&mut d_partition)
                            .arg(&mut d_nodes_in_part)
                            .launch(one_thread_cfg.clone())?;
                    }
                    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
                    stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
                } else {
                    if ils_iter % 2 == 0 {
                        perturb_guided(
                            &mut partition_host_refine,
                            &mut nodes_in_part_host,
                            max_part_size,
                            &hedge_offsets_host,
                            &hyperedge_nodes_host,
                            &high_hedge_ids_host,
                        );
                    } else {
                        let seed = 123456789u64 + (ils_iter as u64) * 987654321u64;
                        perturb_random(
                            &mut partition_host_refine,
                            &mut nodes_in_part_host,
                            max_part_size,
                            perturb_strength,
                            seed,
                        );
                    }
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }

                for _ in 0..knobs.ils_quick_refine {
                    launch_edge_flags!();
                    launch_moves!();
                    let m = device_pick!(
                        i32::MAX,
                        0,
                        987654321u64,
                        1u64,
                        2u64,
                        [10u64, 0, 0, 0],
                        move_limit,
                        extra_window / 2,
                        slack_mid + 2
                    );
                    if m == 0 {
                        break;
                    }
                    if apply_moves!() == 0 {
                        break;
                    }
                }

                do_swap_phase!(25);

                unsafe {
                    stream
                        .launch_builder(&compute_connectivity_kernel)
                        .arg(&(num_hedges as i32))
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&d_partition)
                        .arg(&mut d_connectivity)
                        .launch(hedge_cfg.clone())?;
                    stream
                        .launch_builder(&reduce_connectivity_sum_kernel)
                        .arg(&(num_hedges as i32))
                        .arg(&d_connectivity)
                        .arg(&mut d_total_connectivity)
                        .launch(connectivity_reduce_cfg.clone())?;
                }
                let new_connectivity: i32 = stream.memcpy_dtov(&d_total_connectivity)?.iter().sum();

                let improved = new_connectivity < best_connectivity;
                if improved {
                    best_connectivity = new_connectivity;
                    best_partition_host = stream.memcpy_dtov(&d_partition)?;
                    best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;
                    let connectivity_vec = stream.memcpy_dtov(&d_connectivity)?;
                    high_hedge_ids_host =
                        high_hedge_ids(&connectivity_vec, &hedge_sizes_host, knobs.num_high_hedges);
                }

                let slot: Option<usize> = if elite_count < pool_size {
                    elite_count += 1;
                    Some(elite_count - 1)
                } else {
                    let mut worst_idx = 0usize;
                    for i in 1..pool_size {
                        if elite_scores[i] > elite_scores[worst_idx] {
                            worst_idx = i;
                        }
                    }
                    if new_connectivity < elite_scores[worst_idx] {
                        Some(worst_idx)
                    } else {
                        None
                    }
                };
                if let Some(slot) = slot {
                    let current: Vec<i32>;
                    let src: &[i32] = if improved {
                        &best_partition_host
                    } else {
                        current = stream.memcpy_dtov(&d_partition)?;
                        &current
                    };
                    elite_flat_host[slot * num_nodes..(slot + 1) * num_nodes].copy_from_slice(src);
                    elite_scores[slot] = new_connectivity;
                }

                use_consensus_next = !improved;
            }

            stream.memcpy_htod(&best_partition_host, &mut d_partition)?;
            stream.memcpy_htod(&best_nodes_in_part_host, &mut d_nodes_in_part)?;
            partition_host_refine.copy_from_slice(&best_partition_host);
            nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);
        }

        for _ in 0..knobs.post_ils_polish {
            launch_edge_flags!();
            launch_moves!();
            let m = device_pick!(
                i32::MAX,
                0,
                11223344u64,
                1u64,
                1u64,
                [20u64, 0, 0, 0],
                100000usize,
                extra_window / 3,
                slack_mid
            );
            if m == 0 {
                break;
            }
            if apply_moves!() == 0 {
                break;
            }
        }

        let run_final_balance = if skip & 64 != 0 {
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
            nodes_in_part_host
                .iter()
                .any(|&s| s < 1 || s > max_part_size)
        } else {
            true
        };
        if run_final_balance {
            unsafe {
                stream
                    .launch_builder(&balance_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&1i32)
                    .arg(&max_part_size)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .launch(one_thread_cfg.clone())?;
            }
        }
        stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
        stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

        saver.save(&to_u32(&partition_host_refine))?;

        for _ in 0..(if skip & 16 == 0 {
            post_balance_rounds
        } else {
            0
        }) {
            launch_edge_flags!();
            launch_moves!();
            let m = device_pick!(
                i32::MAX,
                0,
                55667788u64,
                1u64,
                1u64,
                [20u64, 0, 0, 0],
                move_limit / 2,
                extra_window / 3,
                slack_mid
            );
            if m == 0 {
                break;
            }
            let k_base = m.min(move_limit / 2);
            let k_cand = m.min(k_base.saturating_add(extra_window / 3));
            let mut moves_executed = apply_moves!();
            if moves_executed == 0 && k_cand > k_base {
                load_fallback_moves!(k_base, k_cand);
                if !sorted_move_nodes.is_empty() {
                    moves_executed = apply_moves!();
                }
            }
            if moves_executed == 0 {
                break;
            }
        }

        do_swap_phase!(10);

        let mut partition_u32 = to_u32(&stream.memcpy_dtov(&d_partition)?);
        let c_post = saver.save_km1(&partition_u32)?;
        let ml2 = post_ml2(num_hedges, &hp, track, tri100.as_ref(), c_post);

        let cg = hp.int("cg", 0, 2, 0);
        if cg != 0 {
            partition_u32 = super::grouped::run_cg(
                challenge, &module, &stream, track, &knobs, &hp,
                cool_pow, cool_pct, slack_scale, &hedge_offsets_host,
                &hyperedge_nodes_host, &partition_u32, &mut saver,
            )?;
        }
        macro_rules! post {
            ($fast:literal) => {
                super::refine::improve::<$fast>(
                    num_nodes,
                    num_hedges,
                    hedge_offsets_host.iter().map(|&x| x as u32).collect(),
                    hyperedge_nodes_host.iter().map(|&x| x as u32).collect(),
                    num_parts,
                    challenge.max_part_size,
                    &partition_u32,
                    knobs.passes,
                    knobs.fm_rounds,
                    knobs.fm_steps,
                    knobs.fm_seeds,
                    knobs.fm_stop,
                    knobs.jet_rounds,
                    knobs.cyc_rounds,
                    super::ml::Config::read(&hp, track.ml_tail),
                    ml2,
                    cg == 2,
                    hp.int("jkeep", 0, 1, if num_hedges == 20_000 { 1 } else { 0 }) == 1,
                )
            };
        }
        let terminal = num_hedges != 100_000
            || gl_run.is_none()
            || !gl_band
            || (gl_pass > 0 && gl_pass >= gl_passes);
        let refined = if terminal { post!(true) } else { post!(false) };
        if refined.len() == num_nodes {
            saver.save(&refined)?;
        }
        if gl_pass == 0 {
            if let Some((_, gl_tri)) = gl_run.as_ref() {
                let gl_km1_0 = saver.best_km1;
                let mut gl_selected = false;
                if gl_band {
                    let gl_end_k = hp.int("gf_end_k", 5_000, 15_000, super::GF_END_K);
                    let lo = hp.int("gf_band_lo", -20_000, 20_000, super::GF_BAND_LO);
                    let hi = hp.int("gf_band_hi", -20_000, 20_000, super::GF_BAND_HI);
                    gl_selected = !gl_tri.cut
                        && Gl1::in_band(
                            gl_km1_0,
                            gl_tri.bids,
                            gl_end_k,
                            gl_tri.tri.bar + lo,
                            gl_tri.tri.bar + hi,
                        );
                }
                if gl_selected {
                    gl_ext = hp.int("gf_ext", 0, 1_000, super::GF_EXT) as u64;
                    gl_passes = hp.int("gf_ext_passes", 1, 4, super::GF_EXT_PASSES) as usize;
                }
            }
        } else {
            debug_assert!(saver.best_km1 <= gl_before);
        }
        if gl_pass >= gl_passes {
            break;
        }
    }
    Ok(())
}

fn post_ml2(
    num_hedges: usize,
    hp: &Hp<'_>,
    track: &Track,
    tri: Option<&Tri100>,
    c_post: Option<u64>,
) -> Option<super::ml2::Config> {
    let defaults = if (num_hedges == 20_000 && hp.int("ml2_20k", 0, 1, 1) == 0)
        || (num_hedges == 50_000 && hp.int("ml2_50k", 0, 1, 1) == 0)
    {
        None
    } else {
        track.ml2.as_ref()
    };
    let mut config = super::ml2::Config::read(hp, defaults);
    if num_hedges != 200_000 {
        return config;
    }
    let enabled = hp.int("post_band", 0, 1, 1) == 1;
    if !enabled {
        return config;
    }
    let (bids, k) = tri.map_or((0, 0), |tri| {
        let (_, bids) = tri.estimate.unwrap_or((0, 0));
        (bids, tri.k)
    });
    let deep = if bids == 0 {
        true
    } else if let Some(c) = c_post {
        let bar = hp.int("post_bar", 200_000, 400_000, 357_750);
        let lo = hp.int("post_lo", -20_000, 0, -3_976);
        let hi = hp.int("post_hi", 0, 20_000, 2_000);
        Gl1::in_band(c, bids, k, bar + lo, bar + hi)
    } else {
        true
    };
    if !deep {
        if let Some(config) = config.as_mut() {
            config.passes = 3;
            config.levels = 1;
            config.fine_rounds = 0;
            config.cycles = 1;
        }
    }
    config
}

pub(super) struct Tri100 {
    enabled: bool,
    at: u64,
    total: u64,
    pub(super) round: usize,
    done: bool,
    ext: u64,
    bar: i64,
    margin: i64,
    k: i64,
    sure: bool,
    hi: i64,
    rho: u64,
    pub(super) reduce: bool,
    budget_hp: Option<Map<String, Value>>,
    lag: usize,
    pending: Option<Tri100Pending>,
    estimate: Option<(u64, u32)>,
}

struct Tri100Pending {
    launches: usize,
    partition: PinnedHostSlice<i32>,
    degrees: PinnedHostSlice<i32>,
}

impl Tri100 {
    pub(super) fn small(hp: &Hp, total: usize, track: usize) -> Option<Self> {
        let (suffix, default_bar) = match track {
            10_000 => ("10k", 273_661),
            20_000 => ("20k", 225_200),
            50_000 => ("50k", 236_339),
            _ => return None,
        };
        let enabled = hp.int("tri", 0, 1, 1) == 1;
        if !enabled {
            return None;
        }
        let value = |name: &str, lo, hi, default| {
            hp.get(&format!("{}_{}", name, suffix), lo, hi)
                .unwrap_or_else(|| hp.int(name, lo, hi, default))
        };
        let tri = Self {
            enabled,
            at: value("tri_at", 50, 900, 100) as u64,
            total: total as u64,
            round: 0,
            done: false,
            ext: if matches!(track, 10_000 | 20_000) {
                value("tri_ext", 1_000, 2_000, if track == 10_000 { 1_400 } else { 1_200 },
                ) as u64
            } else {
                1_000
            },
            bar: value("tri_bar", 200_000, 400_000, default_bar),
            margin: value("tri_margin", 0, 50_000, 0),
            k: value("tri_k", 5_000, 15_000, 10_000),
            sure: false,
            hi: 0,
            rho: 1_000,
            reduce: false,
            budget_hp: None,
            lag: 0,
            pending: None,
            estimate: None,
        };
        debug_assert!(!Self::below(75, 100, 10_000, 250_000));
        debug_assert!(Self::below(76, 100, 10_000, 250_000));
        debug_assert!(!tri.due(0));
        debug_assert!(tri.due(total as u64));
        debug_assert_eq!(tri.cut(u32::MAX as u64, 1), enabled);
        Some(tri)
    }

    fn new(hp: &Hp, total: usize, track: usize) -> Self {
        debug_assert!(!Self::below(75, 100, 10_000, 250_000));
        debug_assert!(Self::below(76, 100, 10_000, 250_000));
        debug_assert!(!Self::below(74, 100, 10_000, 250_000));
        debug_assert!(!Self::below(75, 200, 5_000, 250_000));
        debug_assert!(Self::below(101, 100, 10_000, 0));
        debug_assert!(!Self::below(71, 100, 10_000, 290_000));
        debug_assert!(Self::below(72, 100, 10_000, 290_000));
        debug_assert!(!Self::below(70, 100, 10_000, 290_000));
        debug_assert_eq!(Self::scaled_refinement(100_000, 650), 65_000);
        debug_assert_eq!(Self::scaled_refinement(50, 300), 50);
        debug_assert_eq!(Self::scaled_refinement(8_077, 650), 5_250);
        debug_assert_eq!(
            Knobs::resolve_rounds(65_000, true, true, &super::params::P_100K.d, 0),
            40_235,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                100_000,
                true,
                true,
                &super::params::P_100K.d,
                900_000_000_000
            ),
            60_000,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                65_000,
                true,
                true,
                &super::params::P_100K.d,
                900_000_000_000
            ),
            40_235,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                65_000,
                true,
                true,
                &super::params::P_100K.d,
                450_000_000_000
            ),
            25_000,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(65_000, true, true, &super::params::P_100K.d, 1),
            50,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(8_078, true, true, &super::params::P_100K.d, 0),
            5_000,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(8_077, true, true, &super::params::P_100K.d, 0),
            4_999,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                Self::scaled_refinement(500, 650),
                true,
                true,
                &super::params::P_100K.d,
                0
            ),
            201,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                Self::scaled_refinement(503, 650),
                true,
                true,
                &super::params::P_100K.d,
                0
            ),
            201,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(60_000, true, true, &super::params::P_200K.d, 0),
            32_490,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                Self::scaled_refinement(60_000, 650),
                true,
                true,
                &super::params::P_200K.d,
                0
            ),
            21_118,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                39_000,
                true,
                true,
                &super::params::P_200K.d,
                900_000_000_000
            ),
            17_200,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(
                39_000,
                true,
                true,
                &super::params::P_200K.d,
                450_000_000_000
            ),
            4_600,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(39_000, true, true, &super::params::P_200K.d, 1),
            50,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(9_234, true, true, &super::params::P_200K.d, 0),
            5_000,
        );
        debug_assert_eq!(
            Knobs::resolve_rounds(9_233, true, true, &super::params::P_200K.d, 0),
            4_999,
        );
        let t200 = track == 200_000;
        let lag = hp.int(
            "tri_lag",
            0,
            4,
            if t200 {
                super::TRI200_DEFAULT_LAG
            } else {
                TRI100_DEFAULT_LAG
            },
        ) as usize;
        if lag != 0 {
            debug_assert_eq!(
                tri100_replay_host(
                    &[0, 0, 32, 32],
                    &[1, 2, 1, 2, 999],
                    &[0, 1, 3, 4, 6],
                    &[0, 0, 0, 0, 0, 0],
                    1,
                )
                .unwrap(),
                vec![15, 31, 47, 63],
            );
            debug_assert_eq!(
                tri100_replay_host(
                    &[0, 0, 0, 0],
                    &[2, 2, 1, 1],
                    &[0, 2, 4, 5, 6],
                    &[0, 0, 0, 0, 1, 1],
                    2,
                )
                .unwrap(),
                vec![31, 23, 15, 7],
            );
        }
        let tri = Self {
            enabled: true,
            at: hp.int(
                "tri_at",
                50,
                900,
                if t200 {
                    super::TRI200_DEFAULT_AT
                } else {
                    TRI100_DEFAULT_AT
                },
            ) as u64,
            total: total as u64,
            round: 0,
            done: false,
            ext: 1_000,
            bar: hp.int(
                "tri_bar",
                200_000,
                400_000,
                if t200 {
                    super::TRI200_DEFAULT_BAR
                } else {
                    TRI100_DEFAULT_BAR
                },
            ),
            margin: hp.int(
                "tri_margin",
                0,
                50_000,
                if t200 {
                    super::TRI200_DEFAULT_MARGIN
                } else {
                    TRI100_DEFAULT_MARGIN
                },
            ),
            k: hp.int(
                "tri_k",
                5_000,
                15_000,
                if t200 {
                    super::TRI200_DEFAULT_K
                } else {
                    TRI100_DEFAULT_K
                },
            ),
            sure: hp.int(
                "tri_sure",
                0,
                1,
                if t200 {
                    super::TRI200_DEFAULT_SURE
                } else {
                    TRI100_DEFAULT_SURE
                },
            ) == 1,
            hi: hp.int(
                "tri_hi",
                0,
                50_000,
                if t200 {
                    super::TRI200_DEFAULT_HI
                } else {
                    TRI100_DEFAULT_HI
                },
            ),
            rho: hp.int(
                "tri_rho",
                300,
                1_000,
                if t200 {
                    super::TRI200_DEFAULT_RHO
                } else {
                    TRI100_DEFAULT_RHO
                },
            ) as u64,
            reduce: false,
            budget_hp: None,
            lag,
            pending: None,
            estimate: None,
        };
        debug_assert!(tri.cut(u32::MAX as u64, 1));
        tri
    }

    pub(super) fn due(&self, round: u64) -> bool {
        !self.done && round * 1_000 >= self.total * self.at
    }

    pub(super) fn take_extension(&mut self) -> Option<u64> {
        if self.ext <= 1_000 || !self.enabled || !self.done || self.estimate.is_none() {
            return None;
        }
        let ext = self.ext;
        self.ext = 1_000;
        Some(ext)
    }

    fn below(c: u64, bids: u32, k: i64, threshold: i64) -> bool {
        let b = i128::from(bids) * i128::from(k);
        (b - i128::from(c) * 10_000) * 1_000_000 < i128::from(threshold) * b
    }

    fn cut(&self, c: u64, bids: u32) -> bool {
        self.enabled && Self::below(c, bids, self.k, self.bar - self.margin)
    }

    fn scaled_refinement(refinement: usize, rho: u64) -> usize {
        ((refinement as u64 * rho / 1_000) as usize).max(50)
    }

    pub(super) fn reduce_budget(
        &mut self,
        hp: &Hp,
        track: &Track,
        knobs: &mut Knobs,
    ) -> anyhow::Result<Option<(usize, usize)>> {
        if !self.reduce {
            return Ok(None);
        }
        self.reduce = false;
        let requested = Self::scaled_refinement(knobs.refinement_requested, self.rho);
        let total = Knobs::resolve_rounds(requested, true, true, &track.d, knobs.refinement_fuel);
        debug_assert!(total <= knobs.refinement_rounds);
        let changed = total < knobs.refinement_rounds;
        let mut schedule = None;
        if changed {
            knobs.refinement_rounds = total;
            knobs.ils_iterations =
                hp.int("ils_iterations", 0, 500, knobs.effort.ils as i64) as usize;
            knobs.post_ils_polish =
                hp.int("post_ils_polish", 0, 500, knobs.effort.polish as i64) as usize;
            knobs.swap_scale = hp.int("swap_scale", 0, 400, track.d.swap_scale as i64) as u64;
            self.budget_hp = super::params::configured_hp(hp, &track.d, knobs, track.pz_move_npt)?;
            let hp = Hp(&self.budget_hp);
            schedule = Some((
                hp.int("cool_rounds", 50, 400_000, total as i64) as usize,
                hp.int("pert_rounds", 50, 400_000, total as i64) as usize,
            ));
        }
        Ok(schedule)
    }

    pub(super) fn launched(&mut self) {
        if let Some(pending) = self.pending.as_mut() {
            pending.launches += 1;
        }
    }

    pub(super) fn finish_pending(
        &mut self,
        force: bool,
        part: &CudaSlice<i32>,
        challenge: &Challenge,
        stream: &Arc<CudaStream>,
        offsets: &[i32],
        edges: &[i32],
        saver: &mut Saver<'_>,
    ) -> anyhow::Result<bool> {
        if self
            .pending
            .as_ref()
            .map_or(true, |p| !force && p.launches < self.lag)
        {
            return Ok(false);
        }
        let pending = self.pending.take().unwrap();
        debug_assert!(pending.launches <= self.lag);
        let result = (|| -> anyhow::Result<_> {
            let partition = to_u32(pending.partition.as_slice()?);
            let degrees = pending.degrees.as_slice()?.to_vec();
            let c = super::feasible_km1(
                saver.he_off,
                saver.he_nodes,
                &partition,
                challenge.num_parts,
                challenge.max_part_size,
            )
            .ok_or_else(|| anyhow::anyhow!("16"))?;
            anyhow::ensure!(challenge.num_parts == 64, "17");
            let replay = tri100_replay_host(
                &partition,
                &degrees,
                offsets,
                edges,
                challenge.num_hyperedges as usize,
            )?;
            let bids = super::feasible_km1(
                saver.he_off,
                saver.he_nodes,
                &replay,
                challenge.num_parts,
                challenge.max_part_size,
            )
            .ok_or_else(|| anyhow::anyhow!("18"))?;
            let bids = u32::try_from(bids)?;
            anyhow::ensure!(bids > 0, "19");
            Ok((c, bids, self.cut(c, bids)))
        })();
        stream.synchronize()?;
        match result {
            Ok((c, bids, cut)) => {
                if self.enabled && !cut {
                    self.estimate = Some((c, bids));
                }
                let sure = self.sure && !cut && !Self::below(c, bids, self.k, self.bar + self.hi);
                self.reduce = sure;
                if cut {
                    let current = stream.memcpy_dtov(part)?;
                    stream.synchronize()?;
                    saver.save(&to_u32(&current))?;
                }
                Ok(cut)
            }
            Err(_) => Ok(false),
        }
    }

    pub(super) fn check(
        &mut self,
        round: usize,
        part: &CudaSlice<i32>,
        challenge: &Challenge,
        module: &Arc<CudaModule>,
        stream: &Arc<CudaStream>,
        saver: &mut Saver<'_>,
    ) -> anyhow::Result<bool> {
        if !self.due(round as u64) {
            return Ok(false);
        }
        if self.lag != 0 {
            let pending = (|| -> anyhow::Result<_> {
                let n = challenge.num_nodes as usize;
                anyhow::ensure!(part.len() == n, "20");
                anyhow::ensure!(challenge.d_node_degrees.len() >= n, "invalid degree count");
                let mut partition = unsafe { stream.context().alloc_pinned(n)? };
                let mut degrees = unsafe { stream.context().alloc_pinned(n)? };
                stream.memcpy_dtoh(part, &mut partition)?;
                stream.memcpy_dtoh(&challenge.d_node_degrees.slice(..n), &mut degrees)?;
                Ok(Tri100Pending {
                    launches: 0,
                    partition,
                    degrees,
                })
            })();
            match pending {
                Ok(pending) => {
                    self.pending = Some(pending);
                    self.done = true;
                    return Ok(false);
                }
                Err(_) => {}
            }
        }
        self.done = true;
        let result = (|| -> anyhow::Result<_> {
            let snapshot = stream.memcpy_dtov(part)?;
            stream.synchronize()?;
            let partition = to_u32(&snapshot);
            let c = super::feasible_km1(
                saver.he_off,
                saver.he_nodes,
                &partition,
                challenge.num_parts,
                challenge.max_part_size,
            )
            .ok_or_else(|| anyhow::anyhow!("16"))?;
            let bids = tri100_bids(challenge, module, stream, &partition)?;
            anyhow::ensure!(bids > 0, "19");
            Ok((partition, c, bids))
        })();
        match result {
            Ok((partition, c, bids)) => {
                let cut = self.cut(c, bids);
                if self.enabled && !cut {
                    self.estimate = Some((c, bids));
                }
                let sure = self.sure && !cut && !Self::below(c, bids, self.k, self.bar + self.hi);
                self.reduce = sure;
                if cut {
                    saver.save(&partition)?;
                }
                Ok(cut)
            }
            Err(_) => Ok(false),
        }
    }
}

fn tri100_replay_host(
    last: &[u32],
    degrees: &[i32],
    offsets: &[i32],
    edges: &[i32],
    num_edges: usize,
) -> anyhow::Result<Vec<u32>> {
    let n = last.len();
    anyhow::ensure!(
        last.iter().all(|&p| p < 64),
        "17"
    );
    anyhow::ensure!(degrees.len() >= n, "invalid degree count");
    anyhow::ensure!(
        offsets.len() == n + 1
            && offsets[0] == 0
            && offsets.windows(2).all(|w| w[0] <= w[1])
            && offsets[n] as usize <= edges.len(),
        "invalid node offsets"
    );
    anyhow::ensure!(
        edges[..offsets[n] as usize]
            .iter()
            .all(|&e| e >= 0 && (e as usize) < num_edges),
        "invalid node hyperedges"
    );
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_unstable_by(|&a, &b| degrees[b].cmp(&degrees[a]).then_with(|| a.cmp(&b)));
    let mut partition: Vec<u32> = last.iter().map(|&p| if p < 32 { 1 } else { 2 }).collect();
    let mut flags = vec![0u64; num_edges];
    for level in 1..6 {
        let first = (1usize << level) - 1;
        let mut sizes = [0usize; 32];
        let mut left = [0usize; 32];
        let mut right = [0usize; 32];
        for &p in &partition {
            sizes[p as usize - first] += 1;
        }
        flags.fill(0);
        for &v in &sorted {
            let p = partition[v];
            let i = p as usize - first;
            let left_bit = 1u64 << (2 * i);
            let right_bit = left_bit << 1;
            let incident = &edges[offsets[v] as usize..offsets[v + 1] as usize];
            let assign_left = if left[i] >= sizes[i] / 2 {
                false
            } else if right[i] >= sizes[i] - sizes[i] / 2 {
                true
            } else {
                let mut connections_left = 0;
                let mut connections_right = 0;
                for &e in incident {
                    let f = flags[e as usize];
                    connections_left += i32::from(f & left_bit != 0);
                    connections_right += i32::from(f & right_bit != 0);
                }
                if connections_left == connections_right {
                    left[i] < right[i]
                } else {
                    connections_left > connections_right
                }
            };
            let bit = if assign_left {
                left[i] += 1;
                partition[v] = p * 2 + 1;
                left_bit
            } else {
                right[i] += 1;
                partition[v] = p * 2 + 2;
                right_bit
            };
            for &e in incident {
                flags[e as usize] |= bit;
            }
        }
        debug_assert!((0..(1usize << level))
            .all(|i| { left[i] == sizes[i] / 2 && right[i] == sizes[i] - sizes[i] / 2 }));
    }
    for p in &mut partition {
        *p -= 63;
    }
    Ok(partition)
}

pub(super) fn tri100_bids(
    challenge: &Challenge,
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    last: &[u32],
) -> anyhow::Result<u32> {
    anyhow::ensure!(
        challenge.num_parts == 64
            && last.len() == challenge.num_nodes as usize
            && last.iter().all(|&p| p < 64),
        "17"
    );
    let degrees = stream.memcpy_dtov(&challenge.d_node_degrees)?;
    stream.synchronize()?;
    anyhow::ensure!(
        degrees.len() >= challenge.num_nodes as usize,
        "invalid degree count"
    );
    let degrees = &degrees[..challenge.num_nodes as usize];
    let mut sorted: Vec<i32> = (0..challenge.num_nodes as i32).collect();
    sorted.sort_unstable_by(|&a, &b| {
        degrees[b as usize]
            .cmp(&degrees[a as usize])
            .then_with(|| a.cmp(&b))
    });
    debug_assert!(sorted.windows(2).all(|w| {
        degrees[w[0] as usize] > degrees[w[1] as usize]
            || (degrees[w[0] as usize] == degrees[w[1] as usize]
                && w[0] < w[1])
    }));
    let sorted = stream.memcpy_stod(&sorted)?;
    let root: Vec<i32> = last.iter().map(|&p| if p < 32 { 1 } else { 2 }).collect();
    let mut partition = stream.memcpy_stod(&root)?;
    let greedy = module.load_function("greedy_bipartition")?;
    let finalize = module.load_function("finalize_bipartition")?;
    let connectivity = module.load_function("calc_connectivity_metric")?;
    for level in 1..6u32 {
        let parts = 1u32 << level;
        let num_flags = ((challenge.num_hyperedges + 63) / 64) as usize * parts as usize;
        let mut left = stream.alloc_zeros::<u64>(num_flags)?;
        let mut right = stream.alloc_zeros::<u64>(num_flags)?;
        let current = partition.try_clone()?;
        unsafe {
            stream
                .launch_builder(&greedy)
                .arg(&level)
                .arg(&challenge.num_nodes)
                .arg(&challenge.num_hyperedges)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&sorted)
                .arg(&challenge.d_node_degrees)
                .arg(&current)
                .arg(&mut partition)
                .arg(&mut left)
                .arg(&mut right)
                .launch(LaunchConfig {
                    grid_dim: (parts, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 400,
                })?;
        }
        stream.synchronize()?;
    }
    unsafe {
        stream
            .launch_builder(&finalize)
            .arg(&challenge.num_nodes)
            .arg(&challenge.num_parts)
            .arg(&mut partition)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let mut metric = stream.alloc_zeros::<u32>(1)?;
    unsafe {
        stream
            .launch_builder(&connectivity)
            .arg(&challenge.num_hyperedges)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&partition)
            .arg(&mut metric)
            .launch(LaunchConfig {
                grid_dim: ((challenge.num_hyperedges + 1023) / 1024, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let mut host_metric = [0u32];
    stream.memcpy_dtoh(&metric, &mut host_metric)?;
    stream.synchronize()?;
    Ok(host_metric[0])
}

fn gl_a5(requested: usize, fuel: u64, d: &super::params::Defaults) -> usize {
    if fuel == 0 {
        requested
    } else {
        let usable = (fuel / 10) * 7;
        let affordable = usable.saturating_sub(d.fuel_fixed) / d.fuel_per_round;
        requested.min(affordable.max(50) as usize)
    }
}

fn gl_a6(requested: usize, fuel: u64, d: &super::params::Defaults) -> Option<usize> {
    if fuel != 0 && (fuel / 10) * 7 < d.fuel_fixed + requested.min(50) as u64 * d.fuel_per_round {
        None
    } else {
        Some(gl_a5(requested, fuel, d))
    }
}

pub(super) struct Gl1 {
    tri: Tri100,
    enabled: bool,
    bids: u32,
    cut: bool,
    sure: bool,
    pub(super) total: usize,
    pub(super) cool: usize,
    pub(super) pert: usize,
    fixed_cool: bool,
    fixed_pert: bool,
}

impl Gl1 {
    fn new(hp: &Hp, total: usize) -> Self {
        debug_assert_eq!(Tri100::scaled_refinement(125_000, 650), 81_250);
        debug_assert_eq!(Tri100::scaled_refinement(50, 300), 50);
        debug_assert!(Self::in_band(75, 100, 10_000, 250_000, 260_000));
        debug_assert!(!Self::in_band(74, 100, 10_000, 250_000, 260_000));
        debug_assert!(!Self::in_band(76, 100, 10_000, 250_000, 260_000));
        debug_assert!(!Self::in_band(75, 100, 10_000, 260_000, 250_000));
        debug_assert!(!Self::in_band(0, 0, 10_000, 250_000, 260_000));
        debug_assert_eq!(gl_a6(25_000, 0, &super::params::P_100K.d), Some(25_000));
        debug_assert_eq!(
            gl_a6(25_000, 130_000_000_000, &super::params::P_100K.d),
            Some(111)
        );
        debug_assert_eq!(
            gl_a6(25_000, 129_000_000_000, &super::params::P_100K.d),
            None
        );
        debug_assert_eq!(gl_a6(0, 1, &super::params::P_100K.d), None);
        Self {
            tri: Tri100 {
                enabled: true,
                at: hp.int("tri_at", 50, 900, TRI100_DEFAULT_AT) as u64,
                total: total as u64,
                round: 0,
                done: false,
                ext: 1_000,
                bar: hp.int("tri_bar", 200_000, 400_000, TRI100_DEFAULT_BAR),
                margin: hp.int("tri_margin", 0, 50_000, super::GF_TRI_MARGIN),
                k: hp.int("tri_k", 5_000, 15_000, super::GF_TRI_K),
                sure: hp.int("tri_sure", 0, 1, super::GF_TRI_SURE) == 1,
                hi: hp.int("tri_hi", 0, 50_000, super::GF_TRI_HI),
                rho: hp.int("tri_rho", 300, 1_000, super::GF_TRI_RHO) as u64,
                reduce: false,
                budget_hp: None,
                lag: hp.int("tri_lag", 0, 4, TRI100_DEFAULT_LAG) as usize,
                pending: None,
                estimate: None,
            },
            enabled: hp.int("tri", 0, 1, TRI100_DEFAULT_ENABLED) == 1,
            bids: 0,
            cut: false,
            sure: false,
            total,
            cool: hp.int("cool_rounds", 50, 400_000, total as i64) as usize,
            pert: hp.int("pert_rounds", 50, 400_000, total as i64) as usize,
            fixed_cool: hp.get("cool_rounds", 50, 400_000).is_some(),
            fixed_pert: hp.get("pert_rounds", 50, 400_000).is_some(),
        }
    }

    pub(super) fn batch(&self, start: usize, round: usize, batch_size: usize) -> (usize, bool) {
        let remaining = self.total - round;
        if self.enabled && self.tri.pending.is_some() {
            return (remaining.min(128), true);
        }
        if self.enabled && !self.tri.done {
            let due = ((self.tri.total * self.tri.at + 999) / 1_000) as usize;
            let checkpoint =
                (start + ((due.saturating_sub(start) + 127) / 128).max(1) * 128).min(self.total);
            debug_assert!(self.tri.due(checkpoint as u64));
            debug_assert!(checkpoint - start <= 128 || !self.tri.due((checkpoint - 128) as u64));
            let batch = remaining.min(batch_size).min(checkpoint - round);
            return (batch, round + batch == checkpoint);
        }
        (remaining.min(batch_size), true)
    }

    pub(super) fn reduce_budget(&mut self) -> bool {
        if !self.tri.reduce {
            return false;
        }
        self.tri.reduce = false;
        let total = Tri100::scaled_refinement(self.tri.total as usize, self.tri.rho);
        debug_assert!(total <= self.total);
        if total == self.total {
            return false;
        }
        self.total = total;
        if !self.fixed_cool {
            self.cool = total;
        }
        if !self.fixed_pert {
            self.pert = total;
        }
        true
    }

    fn in_band(km1: u64, bids: u32, k: i64, lo: i64, hi: i64) -> bool {
        bids > 0 && !Tri100::below(km1, bids, k, lo) && Tri100::below(km1, bids, k, hi)
    }

    pub(super) fn launched(&mut self, round: usize) {
        self.tri.round = round;
        self.tri.launched();
    }

    pub(super) fn check(
        &mut self,
        round: usize,
        part: &CudaSlice<i32>,
        challenge: &Challenge,
        stream: &Arc<CudaStream>,
        offsets: &[i32],
        edges: &[i32],
        saver: &Saver<'_>,
    ) -> anyhow::Result<bool> {
        if !self.enabled || !self.tri.due(round as u64) {
            return Ok(false);
        }
        let n = challenge.num_nodes as usize;
        anyhow::ensure!(
            part.len() == n && challenge.d_node_degrees.len() >= n,
            "21",
        );
        let mut partition = unsafe { stream.context().alloc_pinned(n)? };
        let mut degrees = unsafe { stream.context().alloc_pinned(n)? };
        stream.memcpy_dtoh(part, &mut partition)?;
        stream.memcpy_dtoh(&challenge.d_node_degrees.slice(..n), &mut degrees)?;
        self.tri.pending = Some(Tri100Pending {
            launches: 0,
            partition,
            degrees,
        });
        self.tri.done = true;
        self.finish_pending(false, challenge, stream, offsets, edges, saver)
    }

    pub(super) fn finish_pending(
        &mut self,
        force: bool,
        challenge: &Challenge,
        stream: &Arc<CudaStream>,
        offsets: &[i32],
        edges: &[i32],
        saver: &Saver<'_>,
    ) -> anyhow::Result<bool> {
        if self
            .tri
            .pending
            .as_ref()
            .map_or(true, |p| !force && p.launches < self.tri.lag)
        {
            return Ok(self.cut);
        }
        let pending = self.tri.pending.take().unwrap();
        debug_assert!(pending.launches <= self.tri.lag);
        stream.synchronize()?;
        let partition = to_u32(pending.partition.as_slice()?);
        let degrees = pending.degrees.as_slice()?;
        let result = (|| -> anyhow::Result<(u64, u32)> {
            let c = super::feasible_km1(
                saver.he_off,
                saver.he_nodes,
                &partition,
                challenge.num_parts,
                challenge.max_part_size,
            )
            .ok_or_else(|| anyhow::anyhow!("16"))?;
            anyhow::ensure!(challenge.num_parts == 64, "17");
            let replay = tri100_replay_host(
                &partition,
                degrees,
                offsets,
                edges,
                challenge.num_hyperedges as usize,
            )?;
            let bids = super::feasible_km1(
                saver.he_off,
                saver.he_nodes,
                &replay,
                challenge.num_parts,
                challenge.max_part_size,
            )
            .ok_or_else(|| anyhow::anyhow!("18"))?;
            let bids = u32::try_from(bids)?;
            anyhow::ensure!(bids > 0, "19");
            Ok((c, bids))
        })();
        if let Ok((c, bids)) = result {
            self.bids = bids;
            self.cut = Tri100::below(c, bids, self.tri.k, self.tri.bar - self.tri.margin);
            self.sure = self.tri.sure
                && !self.cut
                && !Tri100::below(c, bids, self.tri.k, self.tri.bar + self.tri.hi);
            self.tri.reduce = self.sure;
        }
        Ok(self.cut)
    }
}
