// Solver for the 20k, 50k, 100k and 200k tracks.

use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

use super::params::{Knobs, Track, GROUP_SIZES};
use super::{
    decode_pick, decode_sorted, to_u32, Hp, Saver, FILTER_THREADS, LCG_INC, LCG_MUL, SELECT_THREADS,
};

const SCAN_LIMIT_SWAP: usize = 32;
const SCAN_LIMIT_CYCLE: usize = 8;
const MOVE_NODES_PER_THREAD: u32 = 4;
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
    let moves_kernel = module.load_function(track.k_moves)?;
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
    let configured_hp = super::params::configured_hp(&hp, &track.d, &mut knobs)?;
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
    let cool_rounds: usize =
        hp.int("cool_rounds", 50, 400_000, knobs.refinement_rounds as i64) as usize;
    let pert_rounds: usize =
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

    let mut saver = Saver::new(
        challenge,
        &hedge_offsets_host,
        &hyperedge_nodes_host,
        save_solution,
    );
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
    let ordinary_threads = ((num_nodes - lanes2_count).max(1) as u32 + MOVE_NODES_PER_THREAD - 1)
        / MOVE_NODES_PER_THREAD;
    let lane_blocks = (hub_count as u32 + MOVE_BLOCK / 8 - 1) / (MOVE_BLOCK / 8)
        + ((lanes4_count - hub_count) as u32 + MOVE_BLOCK / 4 - 1) / (MOVE_BLOCK / 4)
        + ((lanes2_count - lanes4_count) as u32 + MOVE_BLOCK / 2 - 1) / (MOVE_BLOCK / 2);
    let moves_cfg = LaunchConfig {
        grid_dim: (
            lane_blocks + (ordinary_threads + MOVE_BLOCK - 1) / MOVE_BLOCK,
            1,
            1,
        ),
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
    if grp_n != 0 {
        super::grouped::run(
            challenge,
            &module,
            &stream,
            track,
            &knobs,
            grp_n,
            gc,
            cool_rounds,
            cool_pow,
            cool_pct,
            pert_rounds,
            slack_scale,
            filt_grid,
            flags_cfg.clone(),
            hlgrid(num_nodes as u32),
            moves_cfg.clone(),
            d_large_hedge_ids.as_ref(),
            large_hedge_count,
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
            &mut d_filt_out,
            &mut d_sort_buf,
            &mut d_sorted,
            &mut d_picks,
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &mut node_tabu_until,
            &node_offsets,
        )?;
        tabu_dirty = false;
    }

    if grp_n == 0 {
        for round in 0..knobs.refinement_rounds {
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
                    listed += (node_offsets[v as usize + 1] - node_offsets[v as usize]) as usize;
                }
                if listed < num_hedges {
                    if listed > 0 {
                        let n_pairs = moved_nodes.len();
                        stream.memcpy_htod(&incr_pairs, &mut d_pairs.slice_mut(0..2 * n_pairs))?;
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
        }
    }
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

    let partition_u32 = to_u32(&stream.memcpy_dtov(&d_partition)?);
    saver.save(&partition_u32)?;

    let refined = super::refine::improve(
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
        super::ml2::Config::read(&hp, track.ml2.as_ref()),
    );
    if refined.len() == num_nodes {
        saver.save(&refined)?;
    }
    Ok(())
}
