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
    pi: &super::params::Piste,
) -> anyhow::Result<()> {
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function(pi.k_cluster)?;
    let compute_preferences_kernel = module.load_function(pi.k_preferences)?;
    let execute_assignments_kernel = module.load_function(pi.k_assignments)?;
    let planted_kernel = module.load_function(pi.k_planted)?;
    let precompute_edge_flags_kernel = module.load_function(pi.k_edge_flags)?;

    let compute_moves_kernel = module.load_function(pi.k_moves)?;
    let compute_moves_w8 = module.load_function(pi.k_moves_w8)?;
    let balance_kernel = module.load_function(pi.k_balance)?;
    let compute_connectivity_kernel = module.load_function(pi.k_connectivity)?;
    let reduce_connectivity_sum_kernel = module.load_function(pi.k_reduce_conn)?;
    let compute_swap_gains_kernel = module.load_function(pi.k_swap_gains)?;
    let choose_elite_per_hyperedge_kernel = module.load_function(pi.k_choose_elite)?;
    let assign_from_elite_votes_kernel = module.load_function(pi.k_assign_elite)?;

    let stream = stream.fork()?;

    if hyperparameters
        .as_ref()
        .and_then(|p| p.get("i6").and_then(|v| v.as_u64()))
        .map(|v| v != 0)
        .unwrap_or(false)
    {
        unsafe { stream.context().disable_event_tracking() };
    }

    let cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_nodes as u32 + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_w8 = LaunchConfig {
        grid_dim: (
            (challenge.num_nodes as u32 * 8 + block_size - 1) / block_size,
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
        shared_mem_bytes: block_size * 4,
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
    let mut d_edge_flags = stream.alloc_zeros::<u64>(2 * challenge.num_hyperedges as usize)?;
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
        .unwrap_or(5);

    // Effort table: see `params.rs`. Indices 0..=5, default outside that range.
    let (base_refine, base_ils, base_ils_quick, base_polish, base_post_balance) =
        if (0..=5).contains(&effort) {
            pi.effort[effort as usize]
        } else {
            pi.effort_def
        };

    let refinement_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(50, 400_000) as usize)
        .unwrap_or(base_refine);

    // Caps the round count to what the granted fuel affords.
    let refinement_rounds = {
        let budget = super::fuel_remaining();
        if budget == 0 {
            refinement_rounds
        } else {
            let usable = (budget / 10) * 7;
            let fixed: u64 = pi.fuel_fixed;
            let per_round: u64 = pi.fuel_per_round;
            let affordable = usable.saturating_sub(fixed) / per_round;
            refinement_rounds.min(affordable.max(50) as usize)
        }
    };

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

    // swap_scale: swap-phase round count, in percent.
    let swap_scale: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("swap_scale").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 400) as u64)
        .unwrap_or(pi.swap_scale_def);

    // hchain: 0 = no host chain, 1 = FM only, 2 = full chain (shipped).
    let hchain: u32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("hchain").and_then(|v| v.as_u64()))
        .map(|v| v.clamp(0, 2) as u32)
        .unwrap_or(2);
    let fm_rounds: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fm_rounds").and_then(|v| v.as_u64()))
        .map(|v| v.clamp(1, 200) as usize)
        .unwrap_or(pi.fm_rounds_def);

    let iso_host: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("iso_host").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(false);
    let iso_gpu: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("iso_gpu").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(false);
    let radix_skip: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("radix_skip").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(true);
    // radix_skip, warp8: alternate code paths, off unless requested.
    let warp8: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("warp8").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(false);
    let host_balance: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("host_balance").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(false);
    let planted: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("planted").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(true);
    // planted_lvl: depth of the starting-partition constraint.
    let planted_lvl: i32 = hyperparameters.as_ref()
        .and_then(|p| p.get("planted_lvl").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 8) as i32).unwrap_or(2);
    let fm_steps: usize = hyperparameters.as_ref()
        .and_then(|p| p.get("fm_steps").and_then(|v| v.as_u64()))
        .map(|v| v.clamp(10, 4000) as usize).unwrap_or(400);
    let fm_seeds: usize = hyperparameters.as_ref()
        .and_then(|p| p.get("fm_seeds").and_then(|v| v.as_u64()))
        .map(|v| v.clamp(1, 4000) as usize).unwrap_or(25);
    let fm_stop: f64 = hyperparameters.as_ref()
        .and_then(|p| p.get("fm_stop").and_then(|v| v.as_u64()))
        .map(|v| (v.clamp(1, 400) as f64) / 100.0).unwrap_or(0.25);
    let jet_rounds: usize = hyperparameters.as_ref()
        .and_then(|p| p.get("jet_rounds").and_then(|v| v.as_u64()))
        .map(|v| v.clamp(1, 64) as usize).unwrap_or(8);
    let cyc_rounds: usize = hyperparameters.as_ref()
        .and_then(|p| p.get("cyc_rounds").and_then(|v| v.as_u64()))
        .map(|v| v.clamp(0, 64) as usize).unwrap_or(4);
    // zgm: selects the kernel's equal-gain tie-break.
    let zgm: i32 = hyperparameters.as_ref()
        .and_then(|p| p.get("zgm").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, pi.zgm_clamp) as i32).unwrap_or(pi.zgm_def);

    let balance_final_on_host = |part_h: &mut [i32], nip: &mut [i32], np: usize, mn: i32, mx: i32| {
        let n = part_h.len();
        for p in 0..np {
            loop {
                if nip[p] >= mn { break; }
                let mut moved = false;
                for o in 0..np {
                    if moved { break; }
                    if o != p && nip[o] > mn {
                        for node in 0..n {
                            if part_h[node] == o as i32 {
                                part_h[node] = p as i32;
                                nip[o] -= 1; nip[p] += 1;
                                moved = true; break;
                            }
                        }
                    }
                }
                if !moved { break; }
            }
        }
        for p in 0..np {
            loop {
                if nip[p] <= mx { break; }
                let mut moved = false;
                for o in 0..np {
                    if moved { break; }
                    if o != p && nip[o] < mx {
                        for node in 0..n {
                            if part_h[node] == p as i32 {
                                part_h[node] = o as i32;
                                nip[p] -= 1; nip[o] += 1;
                                moved = true; break;
                            }
                        }
                    }
                }
                if !moved { break; }
            }
        }
    };
    let moves_kernel = if warp8 { &compute_moves_w8 } else { &compute_moves_kernel };
    let moves_cfg = if warp8 { &cfg_w8 } else { &cfg };
    let tie_host: bool = hyperparameters.as_ref()
        .and_then(|p| p.get("tie_host").and_then(|v| v.as_u64())).map(|v| v != 0).unwrap_or(false);

    let tabu_tenure: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("tabu_tenure").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 30) as usize)
        .unwrap_or(pi.tabu_tenure_def);

    let tabu_fail_tenure = pi.tabu_fail_tenure;
    let tabu_fail_mark_len = pi.tabu_fail_mark_len;

    let extra_window = pi.extra_window;
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
            pi.move_limit_dense
        });

    let neg_gain_thresh: i32 = hyperparameters.as_ref()
        .and_then(|p| p.get("neg_gain_thresh").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 200) as i32).unwrap_or(5);
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
            let cmp = |a: &(i32, i32, i32), b: &(i32, i32, i32)| {
                b.0.cmp(&a.0)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| a.2.cmp(&b.2))
            };
            if iso_gpu && impact.len() > num_high_hedges {
                impact.select_nth_unstable_by(num_high_hedges, cmp);
                impact.truncate(num_high_hedges);
            }
            impact.sort_unstable_by(cmp);
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

    let cible_plausible = challenge.num_nodes <= challenge.num_hyperedges
        && (challenge.num_nodes as u64) * 2 >= challenge.num_hyperedges as u64;

    if planted && cible_plausible {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_seed(challenge.seed);
        let lw: Vec<f32> = vec![
            20.0,
            20.0 + rng.gen::<f32>() * 4.0,
            19.0 + rng.gen::<f32>() * 14.0,
            20.0 + rng.gen::<f32>() * 6.0,
            1.0 + rng.gen::<f32>() * 15.0,
            rng.gen::<f32>() * 0.3,
            rng.gen::<f32>() * 0.1,
            rng.gen::<f32>() * 0.01,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cible = challenge.num_hyperedges as i32;
        let d_seed = stream.memcpy_stod(&challenge.seed)?;
        let d_lw = stream.memcpy_stod(&lw)?;
        let mut d_rs = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
        let mut d_re = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
        let mut d_rl = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
        unsafe {
            stream
                .launch_builder(&planted_kernel)
                .arg(&d_seed)
                .arg(&cible)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&challenge.d_hyperedge_sizes)
                .arg(&d_lw)
                .arg(&(lw.len() as i32))
                .arg(&mut d_rs)
                .arg(&mut d_re)
                .arg(&mut d_rl)
                .launch(hedge_cfg.clone())?;
        }
        let rs = stream.memcpy_dtov(&d_rs)?;
        let re = stream.memcpy_dtov(&d_re)?;
        let rl = stream.memcpy_dtov(&d_rl)?;
        let nd_off = stream.memcpy_dtov(&challenge.d_node_offsets)?;
        let nd_he = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;

        let np = challenge.num_parts as i64;
        let n = challenge.num_nodes as usize;
        let mut incoherents = 0usize;
        for v in 0..n {
            let lo = nd_off[v] as usize;
            let hi = nd_off[v + 1] as usize;
            let (mut best_lvl, mut best_s, mut best_e) = (-1i32, 0i32, cible);
            for k in lo..hi {
                let e = nd_he[k] as usize;
                if e >= rl.len() || rl[e] > planted_lvl { continue; }
                if best_lvl >= 0 && (re[e] <= best_s || rs[e] >= best_e) { incoherents += 1; }
                if rl[e] > best_lvl { best_lvl = rl[e]; best_s = rs[e]; best_e = re[e]; }
            }
            if best_lvl < 0 { continue; }
            let base = best_s as i64 * np / cible as i64;
            let fin = (best_e as i64 * np + cible as i64 - 1) / cible as i64;
            let etendue = (fin - base).max(1);
            partition_host_refine[v] = ((base + (v as i64 % etendue)) % np) as i32;
        }
        if incoherents * 1000 > n {
            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
        } else {
        for x in nodes_in_part_host.iter_mut() { *x = 0; }
        for v in 0..n { nodes_in_part_host[partition_host_refine[v] as usize] += 1; }
        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
    }

    let simulate_execute_moves =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         move_nodes: &[i32],
         move_parts: &[i32],
         acceptes: &mut Vec<i32>|
         -> i32 {
            acceptes.clear();
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
                    acceptes.push(node);
                }
            }
            moves_executed
        };
    let mut acceptes_buf: Vec<i32> = Vec::new();

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
    let mut radix_buf: Vec<(usize, i32)> = Vec::with_capacity(challenge.num_nodes as usize);

    let mut stagnant_rounds = 0usize;
    let max_stagnant_rounds: usize = hyperparameters.as_ref()
        .and_then(|p| p.get("max_stagnant_rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 500) as usize).unwrap_or(30);

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
                .arg(&mut d_edge_flags)
                .launch(hedge_cfg.clone())?;
        }
        unsafe {
            stream
                .launch_builder(moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags)
                .arg(&mut d_move_priorities)
                .arg(&zgm)
                .launch(moves_cfg.clone())?;
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

        if valid_moves.len() > 1 {
            let n_mv = valid_moves.len();
            if radix_buf.len() < n_mv { radix_buf.resize(n_mv, (0usize, 0i32)); }
            else { radix_buf.truncate(n_mv); }
            let mut impair = false;
            {
                let mut src = &mut valid_moves;
                let mut dst = &mut radix_buf;
                for shift in [0u32, 8, 16, 24] {
                    let mut count = [0usize; 256];
                    for &(_, k) in src.iter() {
                        count[(((!k) as u32 >> shift) & 255) as usize] += 1;
                    }
                    if radix_skip && count.iter().any(|&c| c == n_mv) { continue; }
                    let mut acc = 0usize;
                    for c in count.iter_mut() {
                        let t = *c;
                        *c = acc;
                        acc += t;
                    }
                    for &item in src.iter() {
                        let b = (((!item.1) as u32 >> shift) & 255) as usize;
                        dst[count[b]] = item;
                        count[b] += 1;
                    }
                    let tmp = src; src = dst; dst = tmp;
                    impair = !impair;
                }
            }
            // An ODD number of executed passes leaves the result in radix_buf.
            // Echange des deux tampons par transfert de propriete : trois deplacements,
            // aucune copie de donnees, aucun appel de bibliotheque.
            if impair {
                let tmp = valid_moves;
                valid_moves = radix_buf;
                radix_buf = tmp;
            }
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
            &mut acceptes_buf,
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
                    &mut acceptes_buf,
                );
                if moves_executed > 0 {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
            }
        }

        total_moves_executed += moves_executed as usize;

        if moves_executed > 0 {
            let until = round + tabu_tenure;
            for &node in acceptes_buf.iter() {
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
         $d_edge_flags:expr,
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
            let __sr_eff = ((($max_rounds as u64) * swap_scale) / 100) as usize;
            for _swap_round in 0..__sr_eff {
                unsafe {
                    stream
                        .launch_builder(&precompute_edge_flags_kernel)
                        .arg(&(challenge.num_hyperedges as i32))
                        .arg(&num_nodes_i)
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&mut *$d_partition)
                        .arg(&mut *$d_edge_flags)
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
                        .arg(&mut *$d_edge_flags)
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
                        $part_to_part[idx_ab].sort_unstable_by(|x, y| y.1.cmp(&x.1));
                        $part_to_part[idx_ba].sort_unstable_by(|x, y| y.1.cmp(&x.1));
                        if $part_to_part[idx_ab].is_empty() || $part_to_part[idx_ba].is_empty() { continue; }
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
                                break;
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
                                    if gain_ab + $part_to_part[idx_bc][0].1
                                               + $part_to_part[idx_ca][0].1 <= 0 { break; }
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
                stream.memcpy_htod($partition_mut_swap, &mut *$d_partition)?;
                $partition_host_swap.copy_from_slice($partition_mut_swap);
                if swap_count >= prev_swap_count {
                    stagnant += 1;
                    if stagnant >= 3 { break; }
                } else {
                    stagnant = 0;
                }
                prev_swap_count = swap_count;
            }
            anyhow::Ok(total_swaps)
        }};
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags,
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
                .arg(&mut d_edge_flags)
                .launch(hedge_cfg.clone())?;
        }
        unsafe {
            stream
                .launch_builder(moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags)
                .arg(&mut d_move_priorities)
                .arg(&zgm)
                .launch(moves_cfg.clone())?;
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
            &mut acceptes_buf,
        );
        if me > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
        if me == 0 { break; }
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        50, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    let perturb_strength: i32 = hyperparameters.as_ref()
        .and_then(|p| p.get("perturb_strength").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 40) as i32).unwrap_or(3);

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

    let num_high_hedges = std::cmp::min(
        hyperparameters
            .as_ref()
            .and_then(|p| p.get("num_high_hedges").and_then(|v| v.as_u64()))
            .map(|v| v.clamp(50, 50_000) as usize)
            .unwrap_or(500usize),
        challenge.num_hyperedges as usize);
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
                .arg(&mut d_edge_flags)
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
                    .arg(&d_edge_flags)
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

            if host_balance {
                let mut pv = stream.memcpy_dtov(&d_partition)?;
                let mut nv = stream.memcpy_dtov(&d_nodes_in_part)?;
                balance_final_on_host(&mut pv, &mut nv, challenge.num_parts as usize,
                                      1i32, challenge.max_part_size as i32);
                stream.memcpy_htod(&pv, &mut d_partition)?;
                stream.memcpy_htod(&nv, &mut d_nodes_in_part)?;
            } else {
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
                    .arg(&mut d_edge_flags)
                    .launch(hedge_cfg.clone())?;
            }
            unsafe {
                stream
                    .launch_builder(moves_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&d_edge_flags)
                    .arg(&mut d_move_priorities)
                .arg(&zgm)
                    .launch(moves_cfg.clone())?;
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
                &mut acceptes_buf,
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
            &mut d_edge_flags,
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
                .arg(&mut d_edge_flags)
                .launch(hedge_cfg.clone())?;
        }
        unsafe {
            stream
                .launch_builder(moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags)
                .arg(&mut d_move_priorities)
                .arg(&zgm)
                .launch(moves_cfg.clone())?;
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

        let polish_limit = 100000usize;
        let k_base = std::cmp::min(valid_moves.len(), polish_limit);
        let polish_extra = extra_window / 3;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(polish_extra));

        if valid_moves.len() > 1 {
            let n_mv = valid_moves.len();
            if radix_buf.len() < n_mv { radix_buf.resize(n_mv, (0usize, 0i32)); }
            else { radix_buf.truncate(n_mv); }
            let mut impair = false;
            {
                let mut src = &mut valid_moves;
                let mut dst = &mut radix_buf;
                for shift in [0u32, 8, 16, 24] {
                    let mut count = [0usize; 256];
                    for &(_, k) in src.iter() {
                        count[(((!k) as u32 >> shift) & 255) as usize] += 1;
                    }
                    if radix_skip && count.iter().any(|&c| c == n_mv) { continue; }
                    let mut acc = 0usize;
                    for c in count.iter_mut() {
                        let t = *c;
                        *c = acc;
                        acc += t;
                    }
                    for &item in src.iter() {
                        let b = (((!item.1) as u32 >> shift) & 255) as usize;
                        dst[count[b]] = item;
                        count[b] += 1;
                    }
                    let tmp = src; src = dst; dst = tmp;
                    impair = !impair;
                }
            }
            // An ODD number of executed passes leaves the result in radix_buf.
            // Echange des deux tampons par transfert de propriete : trois deplacements,
            // aucune copie de donnees, aucun appel de bibliotheque.
            if impair {
                let tmp = valid_moves;
                valid_moves = radix_buf;
                radix_buf = tmp;
            }
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
            &mut acceptes_buf,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
        if moves_executed == 0 {
            break;
        }
    }

    if host_balance {
        let mut pv = stream.memcpy_dtov(&d_partition)?;
        let mut nv = stream.memcpy_dtov(&d_nodes_in_part)?;
        balance_final_on_host(&mut pv, &mut nv, challenge.num_parts as usize,
                              1i32, challenge.max_part_size as i32);
        stream.memcpy_htod(&pv, &mut d_partition)?;
        stream.memcpy_htod(&nv, &mut d_nodes_in_part)?;
    } else {
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
                .arg(&mut d_edge_flags)
                .launch(hedge_cfg.clone())?;
        }
        unsafe {
            stream
                .launch_builder(moves_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_edge_flags)
                .arg(&mut d_move_priorities)
                .arg(&zgm)
                .launch(moves_cfg.clone())?;
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
            &mut acceptes_buf,
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
                    &mut acceptes_buf,
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
        &mut d_edge_flags,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        10, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    let partition = stream.memcpy_dtov(&d_partition)?;
    let partition_u32: Vec<u32> = partition.iter().map(|&x| x as u32).collect();

    save_solution(&Solution {
        partition: partition_u32.clone(),
    })?;

    let (he_off, he_nodes): (Vec<u32>, Vec<u32>) = if iso_gpu {
        (hedge_offsets_host.iter().map(|&x| x as u32).collect(),
         hyperedge_nodes_host.iter().map(|&x| x as u32).collect())
    } else {
        let he_off_i32 = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
        let he_nodes_i32 = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
        (he_off_i32.iter().map(|&x| x as u32).collect(),
         he_nodes_i32.iter().map(|&x| x as u32).collect())
    };

    let refined = super::refine::improve(
        challenge.num_nodes as usize,
        challenge.num_hyperedges as usize,
        he_off,
        he_nodes,
        challenge.num_parts as usize,
        challenge.max_part_size,
        &partition_u32,
        pi.passes,
        hyperparameters
            .as_ref()
            .and_then(|p| p.get("tie_fm").and_then(|v| v.as_u64()))
            .map(|v| v as u32)
            .unwrap_or(3),
        hchain,
        fm_rounds,
        iso_host,
        tie_host,
        fm_steps,
        fm_seeds,
        fm_stop,
        jet_rounds,
        cyc_rounds,
    );

    let faisable = {
        let mut tailles = vec![0u32; challenge.num_parts as usize];
        let mut hors_bornes = false;
        for &b in refined.iter() {
            if (b as usize) < tailles.len() { tailles[b as usize] += 1; } else { hors_bornes = true; }
        }
        !hors_bornes
            && refined.len() == challenge.num_nodes as usize
            && tailles.iter().all(|&s| s >= 1 && s <= challenge.max_part_size)
    };
    if faisable {
        save_solution(&Solution {
            partition: refined,
        })?;
    }
    Ok(())
}
