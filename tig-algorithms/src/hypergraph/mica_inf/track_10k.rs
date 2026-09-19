// Solver of the 10k track: device refinement rounds with a hyperedge-centric fallback, swap
// phases, an iterated local search with an elite pool and a small population, then the host
// refinement chain.

use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

use super::params::{Knobs, D_10K};
use super::{decode_pick, decode_sorted, to_u32, Hp, Saver};

// Picks buffer: two header words then the moves; the host reads a fixed prefix in one copy.
const PICKS_PREFIX: usize = 2048;
const SCAN_LIMIT_SWAP: usize = 32;
const SCAN_LIMIT_CYCLE: usize = 8;
// Nodes above this degree get a warp each in the move kernel.
const HUB_DEGREE: i32 = 64;
const INIT_RESTART_ID: i32 = 1;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let hp = Hp(hyperparameters);
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);
    let num_nodes = challenge.num_nodes as usize;
    let num_hedges = challenge.num_hyperedges as usize;
    let num_parts = challenge.num_parts as usize;
    let max_part_size = challenge.max_part_size as i32;

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_10k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_10k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_10k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_10k")?;
    let moves_fn = module.load_function("compute_refinement_moves_10k")?;
    let balance_fn = module.load_function("balance_final_par_10k")?;
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
    let crossover_kernel = module.load_function("crossover_partitions_10k")?;
    let pick_kernel = module.load_function("mica_pick1")?;

    let stream = stream.fork()?;

    let grid = |threads: u32| LaunchConfig {
        grid_dim: ((threads + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg = grid(num_nodes as u32);
    let hedge_cfg = grid(num_hedges as u32);
    let one_thread_cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    let sortsel_cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
    let init_random_seed = u32::from_le_bytes([challenge.seed[0], challenge.seed[1], challenge.seed[2], challenge.seed[3]]);

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(num_hedges)?;
    let mut d_partition = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(num_parts)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_move_priorities = stream.alloc_zeros::<i32>(num_nodes)?;
    let mut d_sort_buf = stream.alloc_zeros::<u64>(num_nodes)?;
    let mut d_sorted = stream.alloc_zeros::<u64>(num_nodes)?;
    let mut d_picks = stream.alloc_zeros::<u64>(2 + num_nodes)?;
    let mut picks_host: Vec<u64> = vec![0u64; 2 + num_nodes];

    // Eight threads per node in the move kernel; hub nodes get a warp each in extra blocks.
    let grid_x8 = ((num_nodes as u32) * 8 + block_size - 1) / block_size;
    let hub_nodes: Vec<i32> = {
        let off = stream.memcpy_dtov(&challenge.d_node_offsets)?;
        (0..num_nodes).filter(|&v| off[v + 1] - off[v] > HUB_DEGREE).map(|v| v as i32).collect()
    };
    let num_hubs = hub_nodes.len() as i32;
    let d_hub_nodes = stream.memcpy_stod(if hub_nodes.is_empty() { &[0i32][..] } else { &hub_nodes[..] })?;
    let hub_blocks = (num_hubs as u32 * 32 + block_size - 1) / block_size;
    let moves_cfg = LaunchConfig { grid_dim: (grid_x8 + hub_blocks, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
    let main_blocks_i = grid_x8 as i32;
    let bal_threads: u32 = std::cmp::min(1024, prop.maxThreadsPerBlock as u32).max(1);
    let balance_cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (bal_threads, 1, 1), shared_mem_bytes: 0 };

    // Crossover pairs tried after the iterated local search; the population holds three members.
    let crossover: usize = hp.int("crossover", 0, 2, 2) as usize;

    let mut d_edge_flags_all = stream.alloc_zeros::<u64>(num_hedges)?;
    let mut d_edge_flags_double = stream.alloc_zeros::<u64>(num_hedges)?;

    let knobs = Knobs::read(&hp, &D_10K, num_hedges);
    let move_limit = knobs.move_limit;
    let neg_gain_thresh = knobs.neg_gain_thresh;
    let tabu_tenure = knobs.tabu_tenure as i32;

    let swap_buf_size = 4 * num_nodes;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;

    let mut part_to_part: Vec<Vec<(usize, i32)>> = vec![vec![]; num_parts * num_parts];
    let mut swap_gains_host: Vec<i32> = vec![0i32; swap_buf_size];
    let mut partition_host_swap: Vec<i32> = vec![0i32; num_nodes];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; num_nodes];
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut partition_host_mirror: Vec<i32> = vec![0i32; num_nodes];
    let mut nodes_in_part_mirror: Vec<i32> = vec![0i32; num_parts];
    let mut d_hedge_moves = stream.alloc_zeros::<i32>(num_hedges * 4)?;
    let mut d_hedge_choice = stream.alloc_zeros::<i32>(num_hedges)?;
    let mut node_has_move = vec![false; num_nodes];

    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(num_nodes);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(num_nodes);
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(num_nodes);
    let mut tgt_used: Vec<usize> = vec![0; num_parts];
    let mut tgt_quota: Vec<usize> = vec![0; num_parts];

    let mut stagnant_rounds = 0;

    let mut node_tabu_until: Vec<i32> = vec![0; num_nodes];
    let mut d_node_tabu_until = stream.alloc_zeros::<i32>(num_nodes)?;

    macro_rules! launch_edge_flags {
        ($part:expr) => {{
            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
                    .arg(&(num_hedges as i32))
                    .arg(&(num_nodes as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg($part)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .launch(hedge_cfg.clone())?;
            }
        }};
    }

    // One refinement step on the device: tabu upload, edge flags, then the move kernel with its
    // extra hub blocks.
    macro_rules! launch_moves {
        () => {{
            stream.memcpy_htod(&node_tabu_until, &mut d_node_tabu_until)?;
            launch_edge_flags!(&d_partition);
            unsafe {
                stream
                    .launch_builder(&moves_fn)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_move_priorities)
                    .arg(&HUB_DEGREE)
                    .arg(&main_blocks_i)
                    .arg(&num_hubs)
                    .arg(&d_hub_nodes)
                    .launch(moves_cfg.clone())?;
            }
        }};
    }

    // One round of move selection on the device; fills the move list, yields the candidate count.
    macro_rules! device_pick {
        ($tabu_round:expr, $gmin:expr, $limit:expr, $extra:expr, $slack:expr) => {{
            let n_i = num_nodes as i32;
            let tabu_round_i: i32 = $tabu_round;
            let gmin_i: i32 = $gmin;
            let np_i = num_parts as i32;
            let slack_i = ($slack as usize).min(i32::MAX as usize) as i32;
            let limit_i = ($limit as usize).min(i32::MAX as usize) as i32;
            let extra_i = ($extra as usize).min(i32::MAX as usize) as i32;
            unsafe {
                stream.launch_builder(&pick_kernel)
                    .arg(&d_move_priorities).arg(&d_node_tabu_until).arg(&n_i).arg(&tabu_round_i).arg(&1i32).arg(&1i32).arg(&gmin_i)
                    .arg(&0u64).arg(&1u64).arg(&0u64)
                    .arg(&0u64).arg(&0u64).arg(&0u64).arg(&0u64)
                    .arg(&mut d_sort_buf).arg(&mut d_sorted)
                    .arg(&d_nodes_in_part).arg(&np_i).arg(&max_part_size).arg(&slack_i)
                    .arg(&limit_i).arg(&extra_i)
                    .arg(&mut d_picks)
                    .launch(sortsel_cfg.clone())?;
            }
            let head = 2 + PICKS_PREFIX.min(num_nodes);
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
    let mut global_round = 0i32;

    macro_rules! refresh_host_mirrors_from_device {
        () => {{
            stream.memcpy_dtoh(&d_partition, &mut partition_host_mirror)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_mirror)?;
        }};
    }

    // Applies the move list on the host mirrors and uploads them when something moved; the
    // first `executed` nodes of the list become tabu.
    macro_rules! apply_moves {
        () => {{
            let mut executed = 0i32;
            for i in 0..sorted_move_nodes.len() {
                let node_i32 = sorted_move_nodes[i];
                let target_i32 = sorted_move_parts[i];
                if node_i32 < 0 || target_i32 < 0 {
                    continue;
                }
                let node = node_i32 as usize;
                let target = target_i32 as usize;
                if node >= num_nodes || target >= num_parts {
                    continue;
                }
                let current = partition_host_mirror[node];
                if current >= 0
                    && (current as usize) < num_parts
                    && nodes_in_part_mirror[target] < max_part_size
                    && nodes_in_part_mirror[current as usize] > 1
                {
                    partition_host_mirror[node] = target as i32;
                    nodes_in_part_mirror[current as usize] -= 1;
                    nodes_in_part_mirror[target] += 1;
                    executed += 1;
                }
            }
            if executed > 0 {
                stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
                stream.memcpy_htod(&partition_host_mirror, &mut d_partition)?;
                for &node in sorted_move_nodes.iter().take(executed as usize) {
                    node_tabu_until[node as usize] = global_round + tabu_tenure;
                }
            }
            executed
        }};
    }

    // Moves proposed per hyperedge when the node-centric round finds nothing.
    macro_rules! do_hyperedge_centric_phase {
        () => {{
            launch_edge_flags!(&d_partition);
            unsafe {
                stream
                    .launch_builder(&compute_he_moves_kernel)
                    .arg(&(num_hedges as i32))
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
                let group = chunk.iter().filter(|&&m| m > 0).count() as i32;
                if group > 0 {
                    let prio = 32000 + group;
                    for &m in chunk {
                        if m > 0 {
                            let node = ((m >> 6) - 1) as usize;
                            let tgt = m & 63;
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
            if valid_moves.is_empty() {
                0i32
            } else {
                valid_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                tgt_used.fill(0);
                for p in 0..num_parts {
                    let free = (max_part_size - nodes_in_part_mirror[p]).max(0) as usize;
                    tgt_quota[p] = std::cmp::max(1, free + 4);
                }
                sorted_move_nodes.clear();
                sorted_move_parts.clear();
                for &(node, key) in valid_moves.iter() {
                    let tgt = (key & 63) as usize;
                    if tgt < num_parts && tgt_used[tgt] < tgt_quota[tgt] {
                        tgt_used[tgt] += 1;
                        sorted_move_nodes.push(node as i32);
                        sorted_move_parts.push(tgt as i32);
                    }
                }
                apply_moves!()
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
            .arg(&INIT_RESTART_ID)
            .arg(&init_random_seed)
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_priorities)
            .launch(cfg.clone())?;
    }

    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..num_nodes).collect();
    indices.sort_unstable_by(|&a, &b| pref_priorities[b].cmp(&pref_priorities[a]).then_with(|| a.cmp(&b)));
    let sorted_nodes: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let sorted_parts: Vec<i32> = indices.iter().map(|&i| pref_parts[i]).collect();
    let d_sorted_nodes = stream.memcpy_stod(&sorted_nodes)?;

    // Initial assignment on the host: the first `num_parts` nodes seed one block each, the
    // others take their preferred block or the next one with room.
    partition_host_mirror.fill(0);
    nodes_in_part_mirror.fill(0);
    for i in 0..num_nodes {
        let node_i32 = sorted_nodes[i];
        let preferred_part = sorted_parts[i];
        if node_i32 >= 0 && preferred_part >= 0 {
            let node = node_i32 as usize;
            if node < num_nodes && (preferred_part as usize) < num_parts {
                let start_part = if i < num_parts { i as i32 } else { preferred_part };
                let mut assigned = false;
                for attempt in 0..num_parts {
                    let try_part = ((start_part as usize) + attempt) % num_parts;
                    if nodes_in_part_mirror[try_part] < max_part_size {
                        partition_host_mirror[node] = try_part as i32;
                        nodes_in_part_mirror[try_part] += 1;
                        assigned = true;
                        break;
                    }
                }
                if !assigned {
                    let fallback_part = node % num_parts;
                    partition_host_mirror[node] = fallback_part as i32;
                    nodes_in_part_mirror[fallback_part] += 1;
                }
            }
        }
    }
    let assigned_parts: Vec<i32> = sorted_nodes.iter().map(|&node| partition_host_mirror[node as usize]).collect();
    let d_assigned_parts = stream.memcpy_stod(&assigned_parts)?;
    stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;

    unsafe {
        stream
            .launch_builder(&execute_assignments_kernel)
            .arg(&(num_nodes as i32))
            .arg(&(num_parts as i32))
            .arg(&max_part_size)
            .arg(&d_sorted_nodes)
            .arg(&d_assigned_parts)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(cfg.clone())?;
    }

    // Host copy of the hyperedge CSR; it never changes.
    let he_off_i32 = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let he_nodes_i32 = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;

    let mut saver = Saver::new(challenge, &he_off_i32, &he_nodes_i32, save_solution);
    // The constructed partition is feasible: it is saved before the stages that follow.
    saver.save(&to_u32(&partition_host_mirror))?;

    let inferred = super::infer::starting_partition(
        &super::infer::Params::read(&hp),
        challenge,
        &module,
        &stream,
        &partition_host_mirror,
    )?;
    if let Some(p) = inferred {
        partition_host_mirror.copy_from_slice(&p);
        nodes_in_part_mirror.fill(0);
        for &b in partition_host_mirror.iter() {
            nodes_in_part_mirror[b as usize] += 1;
        }
        stream.memcpy_htod(&partition_host_mirror, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;
    }

    for round in 0..knobs.refinement_rounds {
        global_round += 1;

        launch_moves!();
        let adaptive_limit = if round < 50 {
            move_limit / 2
        } else if round < 200 {
            move_limit
        } else {
            move_limit / 3
        };
        let extra_window = 16384usize;
        let slack = if round < 64 { 8usize } else if round < 256 { 4usize } else { 2usize };
        let m = device_pick!(global_round, i32::MIN, adaptive_limit, extra_window, slack);
        if m == 0 {
            break;
        }
        let k_base = m.min(adaptive_limit);
        let k_cand = m.min(k_base.saturating_add(extra_window));

        let mut moves_executed = apply_moves!();

        if moves_executed == 0 && k_cand > k_base {
            let take = std::cmp::min(k_cand - k_base, k_base);
            stream.memcpy_dtoh(&d_sorted.slice(k_base..k_base + take), &mut picks_host[0..take])?;
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for &w in picks_host[..take].iter() {
                let (node, part) = decode_sorted(w);
                sorted_move_nodes.push(node);
                sorted_move_parts.push(part);
            }
            moves_executed = apply_moves!();
        }

        if moves_executed == 0 {
            if do_hyperedge_centric_phase!() > 0 {
                stagnant_rounds = 0;
                continue;
            }
            stagnant_rounds += 1;
            if stagnant_rounds >= 5 && round < knobs.refinement_rounds.saturating_sub(50) {
                let mini_seed = 987654321u64 + (round as u64) * 123456789u64;
                unsafe {
                    stream
                        .launch_builder(&perturb_kernel)
                        .arg(&(num_nodes as i32))
                        .arg(&(num_parts as i32))
                        .arg(&max_part_size)
                        .arg(&1i32)
                        .arg(&mut d_partition)
                        .arg(&mut d_nodes_in_part)
                        .arg(&mini_seed)
                        .launch(one_thread_cfg.clone())?;
                }
                refresh_host_mirrors_from_device!();
                stagnant_rounds = 0;
            } else if stagnant_rounds > knobs.max_stagnant_rounds {
                break;
            }
        } else {
            stagnant_rounds = 0;
        }
    }

    let mut d_connectivity = stream.alloc_zeros::<i32>(num_hedges)?;

    // Pairwise exchanges between blocks, then three-block cycles, on the swap-gain lists.
    macro_rules! do_swap_phase {
        ($max_rounds:expr) => {{
            let np = num_parts;
            let mut prev_swap_count = usize::MAX;
            let mut stagnant = 0usize;
            partition_host_swap.copy_from_slice(&partition_host_mirror);
            let swap_rounds = ((($max_rounds as u64) * knobs.swap_scale) / 100) as usize;
            for _ in 0..swap_rounds {
                global_round += 1;
                stream.memcpy_htod(&node_tabu_until, &mut d_node_tabu_until)?;
                launch_edge_flags!(&d_partition);
                unsafe {
                    stream
                        .launch_builder(&compute_swap_gains_kernel)
                        .arg(&(num_nodes as i32))
                        .arg(&(np as i32))
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
                stream.memcpy_dtoh(&d_swap_gains, &mut swap_gains_host)?;

                for v in part_to_part.iter_mut() { v.clear(); }
                for node in 0..num_nodes {
                    let src = partition_host_swap[node] as usize;
                    if src >= np { continue; }
                    for slot in 0..4usize {
                        let val = swap_gains_host[node * 4 + slot];
                        if val == 0 { continue; }
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
                        part_to_part[idx_ab].sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
                        part_to_part[idx_ba].sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
                        if part_to_part[idx_ab].is_empty() || part_to_part[idx_ba].is_empty() { continue; }
                        let lab_len = part_to_part[idx_ab].len();
                        let lba_len = part_to_part[idx_ba].len();
                        used_ba_buf.clear();
                        used_ba_buf.resize(lba_len, false);
                        for i in 0..lab_len {
                            let (node_a, gain_a) = part_to_part[idx_ab][i];
                            if partition_mut_swap[node_a] as usize != a { continue; }
                            // The first free partner still in `b` is taken when the pair gains.
                            let mut best_combined = 0i32;
                            let mut best_j = usize::MAX;
                            for j in 0..std::cmp::min(lba_len, SCAN_LIMIT_SWAP) {
                                if used_ba_buf[j] { continue; }
                                let (node_b, gain_b) = part_to_part[idx_ba][j];
                                if partition_mut_swap[node_b] as usize != b { continue; }
                                let combined = gain_a + gain_b;
                                if combined > 0 {
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
                                node_tabu_until[node_a] = global_round + tabu_tenure;
                                node_tabu_until[node_b] = global_round + tabu_tenure;
                                swap_count += 1;
                            }
                        }
                    }
                }

                for a in 0..np {
                    for b in 0..np {
                        if b == a { continue; }
                        let idx_ab = a * np + b;
                        if part_to_part[idx_ab].is_empty() { continue; }
                        for c in 0..np {
                            if c == a || c == b { continue; }
                            let idx_bc = b * np + c;
                            let idx_ca = c * np + a;
                            if part_to_part[idx_bc].is_empty() || part_to_part[idx_ca].is_empty() { continue; }
                            let sl_ab = std::cmp::min(part_to_part[idx_ab].len(), SCAN_LIMIT_CYCLE);
                            let sl_bc = std::cmp::min(part_to_part[idx_bc].len(), SCAN_LIMIT_CYCLE);
                            let sl_ca = std::cmp::min(part_to_part[idx_ca].len(), SCAN_LIMIT_CYCLE);
                            'outer: for i in 0..sl_ab {
                                let (node_ab, gain_ab) = part_to_part[idx_ab][i];
                                if partition_mut_swap[node_ab] as usize != a { continue; }
                                if gain_ab + part_to_part[idx_bc][0].1 + part_to_part[idx_ca][0].1 <= 0 { break; }
                                for j in 0..sl_bc {
                                    let (node_bc, gain_bc) = part_to_part[idx_bc][j];
                                    if partition_mut_swap[node_bc] as usize != b { continue; }
                                    if node_bc == node_ab { continue; }
                                    if gain_ab + gain_bc + part_to_part[idx_ca][0].1 <= 0 { break; }
                                    for t in 0..sl_ca {
                                        let (node_ca, gain_ca) = part_to_part[idx_ca][t];
                                        if partition_mut_swap[node_ca] as usize != c { continue; }
                                        if node_ca == node_ab || node_ca == node_bc { continue; }
                                        if gain_ab + gain_bc + gain_ca > 0 {
                                            partition_mut_swap[node_ab] = b as i32;
                                            partition_mut_swap[node_bc] = c as i32;
                                            partition_mut_swap[node_ca] = a as i32;
                                            node_tabu_until[node_ab] = global_round + tabu_tenure;
                                            node_tabu_until[node_bc] = global_round + tabu_tenure;
                                            node_tabu_until[node_ca] = global_round + tabu_tenure;
                                            swap_count += 1;
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if swap_count == 0 { break; }
                if swap_count >= prev_swap_count {
                    stagnant += 1;
                    if stagnant >= 3 { break; }
                } else {
                    stagnant = 0;
                }
                prev_swap_count = swap_count;
                partition_host_swap.copy_from_slice(&partition_mut_swap);
                stream.memcpy_htod(&partition_host_swap, &mut d_partition)?;
            }
            partition_host_mirror.copy_from_slice(&partition_host_swap);
        }};
    }

    do_swap_phase!(100);

    for _ in 0..30 {
        global_round += 1;
        launch_moves!();
        if device_pick!(i32::MAX, i32::MIN, move_limit / 2, 8192usize, 4usize) == 0 { break; }
        if apply_moves!() == 0 && do_hyperedge_centric_phase!() == 0 {
            break;
        }
    }

    do_swap_phase!(50);

    macro_rules! launch_connectivity {
        () => {{
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
            stream.memcpy_dtov(&d_connectivity)?
        }};
    }
    // Hyperedges of highest connectivity, lower index first.
    let high_hedge_ids = |connectivity: &[i32]| -> Vec<i32> {
        let mut conn_with_idx: Vec<(i32, i32)> = connectivity.iter().enumerate().map(|(i, &c)| (c, i as i32)).collect();
        conn_with_idx.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        conn_with_idx.iter().take(knobs.num_high_hedges).map(|&(_, id)| id).collect()
    };

    let connectivity_vec = launch_connectivity!();
    let mut best_connectivity: i32 = connectivity_vec.iter().sum();
    let mut best_partition_host = stream.memcpy_dtov(&d_partition)?;
    let mut best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;
    let mut d_high_hedge_ids = stream.memcpy_stod(&high_hedge_ids(&connectivity_vec))?;

    // Population for the crossover stage, and elite pool of the iterated local search.
    let mut pop_partitions: Vec<Vec<i32>> = vec![best_partition_host.clone()];
    let mut pop_connectivities: Vec<i32> = vec![best_connectivity];

    let elite_pool_size = std::cmp::max(2usize, knobs.ils_iterations);
    let mut elite_scores: Vec<i32> = vec![i32::MAX; elite_pool_size];
    let mut elite_flat_host: Vec<i32> = vec![0i32; elite_pool_size * num_nodes];
    elite_scores[0] = best_connectivity;
    elite_flat_host[..num_nodes].copy_from_slice(&best_partition_host);
    let mut elite_count: usize = 1;
    let mut d_elite_flat = stream.alloc_zeros::<i32>(elite_pool_size * num_nodes)?;
    let mut use_consensus_next = false;

    // Annealed acceptance of the iterated local search.
    let sa_cooling = if knobs.ils_iterations > 1 { 0.5f64.powf(1.0 / (knobs.ils_iterations as f64)) } else { 0.5 };
    let mut sa_temp = (best_connectivity as f64) * 0.02;
    let mut current_connectivity = best_connectivity;
    let mut current_partition_host = best_partition_host.clone();
    let mut current_nodes_in_part_host = best_nodes_in_part_host.clone();

    for ils_iter in 0..knobs.ils_iterations {
        stream.memcpy_htod(&current_partition_host, &mut d_partition)?;
        stream.memcpy_htod(&current_nodes_in_part_host, &mut d_nodes_in_part)?;
        partition_host_mirror.copy_from_slice(&current_partition_host);
        nodes_in_part_mirror.copy_from_slice(&current_nodes_in_part_host);

        let seed = 123456789u64 + (ils_iter as u64) * 987654321u64;

        if use_consensus_next && elite_count > 1 {
            stream.memcpy_htod(&elite_flat_host, &mut d_elite_flat)?;
            let mut elite_order_host: Vec<i32> = (0..elite_count as i32).collect();
            elite_order_host.sort_unstable_by(|&a, &b| {
                elite_scores[a as usize].cmp(&elite_scores[b as usize]).then_with(|| a.cmp(&b))
            });
            let d_elite_order = stream.memcpy_stod(&elite_order_host)?;

            launch_edge_flags!(&d_partition);
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
                stream
                    .launch_builder(&assign_from_elite_votes_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
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
                if part >= 0 && (part as usize) < num_parts {
                    nodes_in_part_mirror[part as usize] += 1;
                }
            }
            stream.memcpy_htod(&nodes_in_part_mirror, &mut d_nodes_in_part)?;

            launch_edge_flags!(&d_partition);
            unsafe {
                stream
                    .launch_builder(&balance_fn)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&1i32)
                    .arg(&max_part_size)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .launch(balance_cfg.clone())?;
            }
            refresh_host_mirrors_from_device!();
        } else {
            let relink_fraction = if ils_iter % 2 == 0 { 30i32 } else { 15i32 };
            let d_best_partition_dev = stream.memcpy_stod(&best_partition_host)?;
            unsafe {
                stream
                    .launch_builder(&perturb_path_relink_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
                    .arg(&relink_fraction)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_best_partition_dev)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&seed)
                    .launch(one_thread_cfg.clone())?;
                stream
                    .launch_builder(&perturb_guided_kernel)
                    .arg(&(knobs.num_high_hedges as i32))
                    .arg(&d_high_hedge_ids)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&(seed ^ 0xDEADBEEF_CAFEBABE_u64))
                    .launch(one_thread_cfg.clone())?;
                stream
                    .launch_builder(&perturb_hubs_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&(seed ^ 0x123456789ABCDEF0_u64))
                    .launch(one_thread_cfg.clone())?;
                stream
                    .launch_builder(&perturb_ruin_recreate_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .arg(&(seed ^ 0x5A5A5A5A5A5A5A5A_u64))
                    .launch(one_thread_cfg.clone())?;
            }
            refresh_host_mirrors_from_device!();
        }

        for _ in 0..knobs.ils_quick_refine {
            global_round += 1;
            launch_moves!();
            if device_pick!(i32::MAX, 0, move_limit, 16_384usize, 4usize) == 0 {
                break;
            }
            if apply_moves!() == 0 {
                break;
            }
        }

        do_swap_phase!(25);

        let connectivity_vec = launch_connectivity!();
        let new_connectivity: i32 = connectivity_vec.iter().sum();

        let improved = new_connectivity < best_connectivity;
        if improved {
            best_connectivity = new_connectivity;
            best_partition_host = stream.memcpy_dtov(&d_partition)?;
            best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;
            stream.memcpy_htod(&high_hedge_ids(&connectivity_vec), &mut d_high_hedge_ids)?;
        }

        {
            let iter_partition = stream.memcpy_dtov(&d_partition)?;
            let mut worst_idx = 0usize;
            let mut worst_conn = pop_connectivities[0];
            for (idx, &pc) in pop_connectivities.iter().enumerate() {
                if pc > worst_conn { worst_conn = pc; worst_idx = idx; }
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
            elite_count += 1;
            Some(elite_count - 1)
        } else {
            let mut worst_idx = 0usize;
            for i in 1..elite_pool_size {
                if elite_scores[i] > elite_scores[worst_idx] {
                    worst_idx = i;
                }
            }
            if new_connectivity < elite_scores[worst_idx] { Some(worst_idx) } else { None }
        };
        if let Some(slot) = elite_slot {
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

        let delta = new_connectivity - current_connectivity;
        let accept = if delta <= 0 {
            true
        } else if sa_temp > 0.01 {
            let rng_seed = 0xDEADu64.wrapping_add(ils_iter as u64).wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
            let rng_val = ((rng_seed >> 33) as f64) / (u32::MAX as f64);
            rng_val < (-(delta as f64) / sa_temp).exp()
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

        for cross_idx in 0..std::cmp::min(pop_order.len().saturating_sub(1), crossover) {
            let d_parent_a = stream.memcpy_stod(&pop_partitions[pop_order[0]])?;
            let d_parent_b = stream.memcpy_stod(&pop_partitions[pop_order[cross_idx + 1]])?;
            let mut d_child_partition = stream.alloc_zeros::<i32>(num_nodes)?;
            let mut d_child_nip = stream.alloc_zeros::<i32>(num_parts)?;

            launch_edge_flags!(&d_parent_a);
            let mut d_edge_flags_a = stream.alloc_zeros::<u64>(num_hedges)?;
            stream.memcpy_dtod(&d_edge_flags_all, &mut d_edge_flags_a)?;
            let mut d_edge_flags_b_all = stream.alloc_zeros::<u64>(num_hedges)?;
            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
                    .arg(&(num_hedges as i32))
                    .arg(&(num_nodes as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_parent_b)
                    .arg(&mut d_edge_flags_b_all)
                    .arg(&mut d_edge_flags_double)
                    .launch(hedge_cfg.clone())?;
                stream
                    .launch_builder(&crossover_kernel)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&max_part_size)
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
            let mut child_nip_host = vec![0i32; num_parts];
            for &p in child_partition_host.iter() {
                if (p as usize) < num_parts {
                    child_nip_host[p as usize] += 1;
                }
            }
            stream.memcpy_htod(&child_nip_host, &mut d_child_nip)?;

            launch_edge_flags!(&d_child_partition);
            unsafe {
                stream
                    .launch_builder(&balance_fn)
                    .arg(&(num_nodes as i32))
                    .arg(&(num_parts as i32))
                    .arg(&1i32)
                    .arg(&max_part_size)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&d_edge_flags_all)
                    .arg(&d_edge_flags_double)
                    .arg(&mut d_child_partition)
                    .arg(&mut d_child_nip)
                    .launch(balance_cfg.clone())?;
            }

            d_partition = d_child_partition;
            d_nodes_in_part = d_child_nip;
            refresh_host_mirrors_from_device!();

            for _ in 0..knobs.ils_quick_refine {
                global_round += 1;
                launch_moves!();
                if device_pick!(i32::MAX, 0, move_limit / 2, 8192usize, 4usize) == 0 { break; }
                if apply_moves!() == 0 && do_hyperedge_centric_phase!() == 0 {
                    break;
                }
            }

            do_swap_phase!(15);

            let cross_conn: i32 = launch_connectivity!().iter().sum();
            if cross_conn < best_connectivity {
                best_connectivity = cross_conn;
                best_partition_host = stream.memcpy_dtov(&d_partition)?;
                best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;
            }
        }
    }

    stream.memcpy_htod(&best_partition_host, &mut d_partition)?;
    stream.memcpy_htod(&best_nodes_in_part_host, &mut d_nodes_in_part)?;
    partition_host_mirror.copy_from_slice(&best_partition_host);
    nodes_in_part_mirror.copy_from_slice(&best_nodes_in_part_host);

    for _ in 0..knobs.post_ils_polish {
        global_round += 1;
        launch_moves!();
        if device_pick!(i32::MAX, 0, 100_000usize, 16_384usize, 3usize) == 0 {
            break;
        }
        if apply_moves!() == 0 && do_hyperedge_centric_phase!() == 0 {
            break;
        }
    }

    do_swap_phase!(10);

    launch_edge_flags!(&d_partition);
    unsafe {
        stream
            .launch_builder(&balance_fn)
            .arg(&(num_nodes as i32))
            .arg(&(num_parts as i32))
            .arg(&1i32)
            .arg(&max_part_size)
            .arg(&challenge.d_node_offsets)
            .arg(&challenge.d_node_hyperedges)
            .arg(&d_edge_flags_all)
            .arg(&d_edge_flags_double)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(balance_cfg.clone())?;
    }

    let partition_u32 = to_u32(&stream.memcpy_dtov(&d_partition)?);
    saver.save(&partition_u32)?;

    let refined = super::refine::improve(
        num_nodes,
        num_hedges,
        he_off_i32.iter().map(|&x| x as u32).collect(),
        he_nodes_i32.iter().map(|&x| x as u32).collect(),
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
    );
    if refined.len() == num_nodes {
        saver.save(&refined)?;
    }
    Ok(())
}
