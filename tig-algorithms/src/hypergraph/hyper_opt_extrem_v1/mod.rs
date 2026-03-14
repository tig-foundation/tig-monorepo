// TIG's UI uses the pattern `tig_challenges::hypergraph` to automatically detect your algorithm's challenge
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::hypergraph::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub refinement: Option<usize>,
    pub ils: Option<usize>,
    pub ils_refine: Option<usize>,
    pub polish: Option<usize>,
    pub post_balance: Option<usize>,
    pub move_limit: Option<usize>,
}

pub fn help() {
    println!("HyperOpt Extrem");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let dummy: Vec<u32> = (0..challenge.num_nodes as u32).map(|i| i % challenge.num_parts as u32).collect();
    save_solution(&Solution { partition: dummy })?;

    let get_hp = |name: &str, default: i64, min: i64, max: i64| -> usize {
        hyperparameters.as_ref()
            .and_then(|p| p.get(name).and_then(|v| v.as_i64()))
            .map(|v| v.clamp(min, max) as usize)
            .unwrap_or(default as usize)
    };

    let n = challenge.num_nodes as usize;
    let num_parts = challenge.num_parts as usize;
    let n_hedges = challenge.num_hyperedges as usize;
    let max_ps = challenge.max_part_size as usize;

    let (def_refine, def_ils, def_ils_refine, def_polish, def_post_bal, def_ml) = if n_hedges >= 150_000 {
        (2000, 7, 35, 80, 35, 131072)
    } else if n_hedges >= 75_000 {
        (2000, 8, 35, 80, 35, 131072)
    } else if n_hedges >= 30_000 {
        (5000, 10, 50, 150, 60, 200000)
    } else if n_hedges >= 15_000 {
        (5000, 10, 50, 150, 60, 200000)
    } else {
        (5000, 15, 50, 150, 80, 200000)
    };
    let refinement = get_hp("refinement", def_refine, 50, 5000);
    let ils_iters = get_hp("ils", def_ils, 0, 40);
    let ils_refine = get_hp("ils_refine", def_ils_refine, 10, 500);
    let polish_rounds = get_hp("polish", def_polish, 0, 500);
    let post_balance = get_hp("post_balance", def_post_bal, 0, 200);
    let move_limit = get_hp("move_limit", def_ml, 1000, 1000000);
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let extra_window: usize = if n_hedges >= 75_000 { 61440 } else { 16384 };
    let slack_early: usize = 8;
    let slack_mid: usize = 4;
    let slack_late: usize = 2;
    let tabu_tenure: usize = 8;

    let use_gpu_swaps = n_hedges >= 75_000;
    let neg_gain_thresh: i32 = 5;
    let swap_scan_limit: usize = 32;
    let cycle_scan_limit: usize = 8;

    let cluster_fn = module.load_function("hyperedge_clustering")?;
    let prefs_fn = module.load_function("compute_node_preferences")?;
    let assign_fn = module.load_function("execute_node_assignments")?;
    let precompute_fn = module.load_function("precompute_edge_flags")?;
    let compute_moves_fn = module.load_function("compute_moves_flags")?;
    let perturb_fn = module.load_function("perturb_solution")?;
    let reset_fn = module.load_function("reset_counters")?;
    let balance_fn = module.load_function("balance_final")?;
    let conn_fn = module.load_function("my_calc_connectivity")?;
    let swap_gains_fn = if use_gpu_swaps { Some(module.load_function("compute_swap_gains_top3")?) } else { None };

    let one_cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
    let node_cfg = LaunchConfig {
        grid_dim: ((n as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1), shared_mem_bytes: 0,
    };
    let hedge_cfg = LaunchConfig {
        grid_dim: ((n_hedges as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1), shared_mem_bytes: 0,
    };

    let mut d_hclusters = stream.alloc_zeros::<i32>(n_hedges)?;
    let mut d_partition = stream.alloc_zeros::<i32>(n)?;
    let mut d_nip = stream.alloc_zeros::<i32>(num_parts)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(n)?;
    let mut d_pref_prio = stream.alloc_zeros::<i32>(n)?;
    let mut d_move_prio = stream.alloc_zeros::<i32>(n)?;
    let mut d_num_valid = stream.alloc_zeros::<i32>(1)?;
    let mut d_flags_all = stream.alloc_zeros::<u64>(n_hedges)?;
    let mut d_flags_double = stream.alloc_zeros::<u64>(n_hedges)?;
    let mut d_swap_gains = if use_gpu_swaps { Some(stream.alloc_zeros::<i32>(n * 3)?) } else { None };
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(n);
    let mut sorted_nodes_buf: Vec<i32> = Vec::with_capacity(n);
    let mut sorted_targets_buf: Vec<i32> = Vec::with_capacity(n);
    let mut best_partition: Vec<i32> = vec![0i32; n];
    let mut best_nip: Vec<i32> = vec![0i32; num_parts];
    let mut best_conn: i32 = i32::MAX;
    let mut cur_partition: Vec<i32> = vec![0i32; n];
    let mut cur_nip: Vec<i32> = vec![0i32; num_parts];
    let mut cur_conn: i32 = i32::MAX;
    let mut tabu: Vec<usize> = vec![0; n];
    let mut tgt_quota: Vec<usize> = vec![0; num_parts];
    let mut tgt_used: Vec<usize> = vec![0; num_parts];
    let mut swap_gains_host: Vec<i32> = if use_gpu_swaps { vec![0i32; n * 3] } else { vec![] };
    let mut part_to_part: Vec<Vec<(usize, i32)>> = if use_gpu_swaps { vec![Vec::new(); num_parts * num_parts] } else { vec![] };
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut keys_buf: Vec<i32> = vec![0i32; n];
    let mut exec_part: Vec<i32> = vec![0i32; n];
    let mut exec_nip: Vec<i32> = vec![0i32; num_parts];

    let mut rng = SmallRng::from_seed({
        let mut s = [0u8; 32];
        s[..challenge.seed.len().min(32)].copy_from_slice(&challenge.seed[..challenge.seed.len().min(32)]);
        s
    });

    macro_rules! refine {
        ($max_rounds:expr, $bw:expr, $mlimit:expr, $use_tabu:expr, $use_adaptive:expr, $quick:expr, $cap_aware:expr) => {{
            let bw: i32 = $bw;
            let mlimit: usize = $mlimit;
            let use_tabu: bool = $use_tabu;
            let use_adaptive: bool = $use_adaptive;
            let quick: bool = $quick;
            let cap_aware: bool = $cap_aware;
            let mut stagnant = 0usize;
            let mut low_move_streak = 0usize;
            stream.memcpy_dtoh(&d_partition, &mut exec_part)?;
            stream.memcpy_dtoh(&d_nip, &mut exec_nip)?;
            for round in 0..$max_rounds {
                unsafe {
                    stream.launch_builder(&precompute_fn)
                        .arg(&(n_hedges as i32)).arg(&(n as i32))
                        .arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets)
                        .arg(&d_partition)
                        .arg(&mut d_flags_all).arg(&mut d_flags_double)
                        .launch(hedge_cfg.clone())?;
                }
                unsafe {
                    stream.launch_builder(&reset_fn)
                        .arg(&mut d_num_valid)
                        .launch(one_cfg.clone())?;
                }
                unsafe {
                    stream.launch_builder(&compute_moves_fn)
                        .arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32))
                        .arg(&bw)
                        .arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets)
                        .arg(&d_partition).arg(&d_nip)
                        .arg(&d_flags_all).arg(&d_flags_double)
                        .arg(&mut d_move_prio).arg(&mut d_num_valid)
                        .launch(node_cfg.clone())?;
                }
                stream.synchronize()?;

                let nv = stream.memcpy_dtov(&d_num_valid)?[0];
                if nv == 0 { if quick { break; } stagnant += 1; if stagnant > 30 { break; } continue; }

                {
                    stream.memcpy_dtoh(&d_move_prio, &mut keys_buf)?;
                    valid_moves.clear();
                    let max_gain = keys_buf.iter().map(|&k| k >> 16).max().unwrap_or(0);
                    let aspiration = (max_gain * 3) / 4;

                    for (node, &key) in keys_buf.iter().enumerate() {
                        if key > 0 {
                            if !use_tabu || tabu[node] <= round || (key >> 16) >= aspiration {
                                valid_moves.push((node, key));
                            }
                        }
                    }
                }
                if valid_moves.is_empty() { if quick { break; } stagnant += 1; if stagnant > 30 { break; } continue; }

                let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));

                let k_base_limit = if !use_adaptive { mlimit }
                    else if round < 50 { mlimit / 2 }
                    else if round < 200 { mlimit }
                    else { mlimit / 3 };
                let k_base = std::cmp::min(valid_moves.len(), k_base_limit);

                let k_cand = if cap_aware {
                    std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window))
                } else {
                    k_base
                };

                if k_cand > 0 && k_cand < valid_moves.len() {
                    valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                }
                valid_moves[..k_cand].sort_unstable_by(cmp);

                sorted_nodes_buf.clear();
                sorted_targets_buf.clear();

                if cap_aware && k_cand > k_base {
                    let nip_host: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
                    let slack = if round < 64 { slack_early } else if round < 256 { slack_mid } else { slack_late };
                    for p in 0..num_parts {
                        let free = max_ps.saturating_sub(nip_host[p] as usize);
                        tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
                        tgt_used[p] = 0;
                    }

                    let mut selected = 0usize;
                    for i in 0..k_cand {
                        if selected >= k_base { break; }
                        let (nd, key) = valid_moves[i];
                        let tgt = (key & 63) as usize;
                        if tgt < num_parts && tgt_used[tgt] < tgt_quota[tgt] {
                            sorted_nodes_buf.push(nd as i32);
                            sorted_targets_buf.push(tgt as i32);
                            tgt_used[tgt] += 1;
                            selected += 1;
                        }
                    }

                    if sorted_nodes_buf.is_empty() {
                        for i in 0..k_base {
                            sorted_nodes_buf.push(valid_moves[i].0 as i32);
                            sorted_targets_buf.push((valid_moves[i].1 & 63) as i32);
                        }
                    }
                } else {
                    for i in 0..k_cand {
                        sorted_nodes_buf.push(valid_moves[i].0 as i32);
                        sorted_targets_buf.push((valid_moves[i].1 & 63) as i32);
                    }
                }

                let mut me = 0i32;
                let mps_i32 = challenge.max_part_size as i32;
                for i in 0..sorted_nodes_buf.len() {
                    let node = sorted_nodes_buf[i] as usize;
                    let target = sorted_targets_buf[i];
                    let src = exec_part[node];
                    if src == target { continue; }
                    if exec_nip[target as usize] >= mps_i32 { continue; }
                    exec_part[node] = target;
                    exec_nip[src as usize] -= 1;
                    exec_nip[target as usize] += 1;
                    me += 1;
                }

                if me == 0 && k_cand > sorted_nodes_buf.len() {
                    let tail_start = sorted_nodes_buf.len();
                    sorted_nodes_buf.clear();
                    sorted_targets_buf.clear();
                    let tail_end = std::cmp::min(valid_moves.len(), tail_start + k_base);
                    for i in tail_start..tail_end {
                        sorted_nodes_buf.push(valid_moves[i].0 as i32);
                        sorted_targets_buf.push((valid_moves[i].1 & 63) as i32);
                    }
                    for i in 0..sorted_nodes_buf.len() {
                        let node = sorted_nodes_buf[i] as usize;
                        let target = sorted_targets_buf[i];
                        let src = exec_part[node];
                        if src == target { continue; }
                        if exec_nip[target as usize] >= mps_i32 { continue; }
                        exec_part[node] = target;
                        exec_nip[src as usize] -= 1;
                        exec_nip[target as usize] += 1;
                        me += 1;
                    }
                }

                if me > 0 {
                    stream.memcpy_htod(&exec_part, &mut d_partition)?;
                    stream.memcpy_htod(&exec_nip, &mut d_nip)?;
                }

                if use_tabu && me > 0 {
                    for &nd in sorted_nodes_buf.iter().take(me as usize) {
                        tabu[nd as usize] = round + tabu_tenure;
                    }
                }

                if me == 0 {
                    if quick { break; }
                    stagnant += 1;
                    low_move_streak = 0;
                    let perturb_str: i32 = if n_hedges >= 75_000 { 3 } else { 1 };
                    let stag_thresh: usize = if n_hedges >= 75_000 { 3 } else { 5 };
                    if stagnant >= stag_thresh && round < ($max_rounds as usize).saturating_sub(50) {
                        let seed_val: u64 = rng.gen();
                        unsafe {
                            stream.launch_builder(&perturb_fn)
                                .arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32))
                                .arg(&perturb_str)
                                .arg(&mut d_partition).arg(&mut d_nip).arg(&seed_val)
                                .launch(one_cfg.clone())?;
                        }
                        stream.synchronize()?;
                        stream.memcpy_dtoh(&d_partition, &mut exec_part)?;
                        stream.memcpy_dtoh(&d_nip, &mut exec_nip)?;
                        stagnant = 0;
                    } else if stagnant > 30 {
                        break;
                    }
                } else {
                    stagnant = 0;
                    if (me as usize) < n / 200 { low_move_streak += 1; } else { low_move_streak = 0; }
                    if low_move_streak >= 10 && !quick { break; }
                }
            }
        }};
    }

    macro_rules! eval_conn {
        () => {{
            let mut d_metric = stream.alloc_zeros::<u32>(1)?;
            unsafe {
                stream.launch_builder(&conn_fn)
                    .arg(&(n_hedges as i32)).arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes).arg(&d_partition).arg(&mut d_metric)
                    .launch(hedge_cfg.clone())?;
            }
            stream.synchronize()?;
            stream.memcpy_dtov(&d_metric)?[0] as i32
        }};
    }

    macro_rules! balance {
        () => {{
            unsafe {
                stream.launch_builder(&balance_fn)
                    .arg(&(n as i32)).arg(&(num_parts as i32))
                    .arg(&1i32).arg(&(challenge.max_part_size as i32))
                    .arg(&mut d_partition).arg(&mut d_nip)
                    .launch(one_cfg.clone())?;
            }
            stream.synchronize()?;
        }};
    }

    macro_rules! cpu_fm {
        ($rounds:expr, $max_deg:expr, $ph:expr, $nh:expr, $fa:expr, $fd:expr, $noff:expr, $nhedges:expr, $hoff:expr, $hnodes:expr, $epcnt:expr) => {{
            let max_deg: usize = $max_deg;
            let ph: &mut Vec<i32> = $ph;
            let nh: &mut Vec<i32> = $nh;
            let fa: &mut Vec<u64> = $fa;
            let fd: &mut Vec<u64> = $fd;
            let noff: &[i32] = $noff;
            let nhedges_arr: &[i32] = $nhedges;
            let hoff: &[i32] = $hoff;
            let hnodes: &[i32] = $hnodes;
            let epcnt: &mut Vec<u8> = $epcnt;
            let npp = std::cmp::min(num_parts, 64);
            let mps = max_ps as i32;
            for _cfm in 0..$rounds {
                let mut moves_made = 0usize;
                for node in 0..n {
                    let src = ph[node] as usize;
                    if src >= npp || nh[src] <= 1 { continue; }
                    let start = noff[node] as usize;
                    let end = noff[node + 1] as usize;
                    let degree = end - start;
                    if degree == 0 || degree > max_deg { continue; }
                    let cur_bit: u64 = 1u64 << src;
                    let mut part_counts = [0i16; 64];
                    let mut count_current = 0i32;
                    for j in 0..degree {
                        let hedge = nhedges_arr[start + j] as usize;
                        let ffa = fa[hedge];
                        let ffd = fd[hedge];
                        let mask = (ffa & !cur_bit) | (ffd & cur_bit);
                        if mask & cur_bit != 0 { count_current += 1; }
                        let mut fl = mask & !cur_bit;
                        while fl != 0 {
                            let bit = fl.trailing_zeros() as usize;
                            fl &= fl - 1;
                            if bit < npp { part_counts[bit] += 1; }
                        }
                    }
                    let mut best_gain = 0i32;
                    let mut best_tgt = src;
                    for p in 0..npp {
                        if p == src { continue; }
                        if nh[p] >= mps { continue; }
                        let gain = part_counts[p] as i32 - count_current;
                        if gain > best_gain || (gain == best_gain && gain > 0 && p < best_tgt) {
                            best_gain = gain;
                            best_tgt = p;
                        }
                    }
                    if best_gain <= 0 || best_tgt == src { continue; }
                    ph[node] = best_tgt as i32;
                    nh[src] -= 1;
                    nh[best_tgt] += 1;
                    moves_made += 1;
                    for j in start..end {
                        let h = nhedges_arr[j] as usize;
                        if h >= n_hedges { continue; }
                        let idx_src = h * npp + src;
                        let idx_tgt = h * npp + best_tgt;
                        let old_src = epcnt[idx_src];
                        let old_tgt = epcnt[idx_tgt];
                        if old_src > 0 { epcnt[idx_src] -= 1; }
                        if epcnt[idx_tgt] < 255 { epcnt[idx_tgt] += 1; }
                        let new_src = epcnt[idx_src];
                        let new_tgt = epcnt[idx_tgt];
                        let src_bit = 1u64 << src;
                        let tgt_bit = 1u64 << best_tgt;
                        if new_src == 0 { fa[h] &= !src_bit; }
                        if old_tgt == 0 { fa[h] |= tgt_bit; }
                        if new_src < 2 { fd[h] &= !src_bit; }
                        if new_tgt >= 2 { fd[h] |= tgt_bit; }
                    }
                }
                if moves_made == 0 { break; }
            }
        }};
    }

    macro_rules! gpu_swap_phase {
        ($max_rounds:expr, $do_cycles:expr) => {{
            let sgfn = swap_gains_fn.as_ref().unwrap();
            let dsg = d_swap_gains.as_mut().unwrap();
            {
                let mut prev_swap_count = usize::MAX;
                let mut stagnant_sw = 0usize;
                for _sw_round in 0..$max_rounds {
                    unsafe {
                        stream.launch_builder(&precompute_fn)
                            .arg(&(n_hedges as i32)).arg(&(n as i32))
                            .arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets)
                            .arg(&d_partition)
                            .arg(&mut d_flags_all).arg(&mut d_flags_double)
                            .launch(hedge_cfg.clone())?;
                    }
                    unsafe {
                        stream.launch_builder(sgfn)
                            .arg(&(n as i32)).arg(&(num_parts as i32))
                            .arg(&neg_gain_thresh)
                            .arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets)
                            .arg(&d_partition)
                            .arg(&d_flags_all).arg(&d_flags_double)
                            .arg(&mut *dsg)
                            .launch(node_cfg.clone())?;
                    }
                    stream.synchronize()?;
                    stream.memcpy_dtoh(&*dsg, &mut swap_gains_host)?;
                    let part_snap: Vec<i32> = stream.memcpy_dtov(&d_partition)?;

                    for v in part_to_part.iter_mut() { v.clear(); }
                    for node in 0..n {
                        let src = part_snap[node] as usize;
                        if src >= num_parts { continue; }
                        for k in 0..3usize {
                            let val = swap_gains_host[node * 3 + k];
                            if val == 0 { continue; }
                            let tgt = (val & 0xFFFF) as usize;
                            let gain = ((val >> 16) as i16) as i32;
                            if tgt < num_parts && tgt != src {
                                part_to_part[src * num_parts + tgt].push((node, gain));
                            }
                        }
                    }
                    for list in part_to_part.iter_mut() {
                        if !list.is_empty() { list.sort_unstable_by(|x, y| y.1.cmp(&x.1)); }
                    }

                    let mut part_mut = part_snap.clone();
                    let mut swap_count = 0usize;

                    for a in 0..num_parts {
                        for b in (a+1)..num_parts {
                            let idx_ab = a * num_parts + b;
                            let idx_ba = b * num_parts + a;
                            if part_to_part[idx_ab].is_empty() || part_to_part[idx_ba].is_empty() { continue; }
                            let lba_len = part_to_part[idx_ba].len();
                            let scan_lba = std::cmp::min(lba_len, swap_scan_limit);
                            used_ba_buf.clear();
                            used_ba_buf.resize(lba_len, false);
                            for i in 0..part_to_part[idx_ab].len() {
                                let (node_a, gain_a) = part_to_part[idx_ab][i];
                                if part_mut[node_a] as usize != a { continue; }
                                let mut best_combined = 0i32;
                                let mut best_j = usize::MAX;
                                for j in 0..scan_lba {
                                    if used_ba_buf[j] { continue; }
                                    let (node_b, gain_b) = part_to_part[idx_ba][j];
                                    if part_mut[node_b] as usize != b { continue; }
                                    let combined = gain_a + gain_b;
                                    if combined > best_combined {
                                        best_combined = combined;
                                        best_j = j;
                                    }
                                }
                                if best_j < lba_len && best_combined > 0 {
                                    let (node_b, _) = part_to_part[idx_ba][best_j];
                                    part_mut[node_a] = b as i32;
                                    part_mut[node_b] = a as i32;
                                    used_ba_buf[best_j] = true;
                                    swap_count += 1;
                                }
                            }
                        }
                    }

                    if $do_cycles {
                        for a in 0..num_parts {
                            for b in 0..num_parts {
                                if b == a { continue; }
                                let idx_ab = a * num_parts + b;
                                if part_to_part[idx_ab].is_empty() { continue; }
                                let ab_top = std::cmp::min(part_to_part[idx_ab].len(), cycle_scan_limit);
                                for c in 0..num_parts {
                                    if c == a || c == b { continue; }
                                    let idx_bc = b * num_parts + c;
                                    let idx_ca = c * num_parts + a;
                                    if part_to_part[idx_bc].is_empty() || part_to_part[idx_ca].is_empty() { continue; }
                                    let bc_top = std::cmp::min(part_to_part[idx_bc].len(), cycle_scan_limit);
                                    let ca_top = std::cmp::min(part_to_part[idx_ca].len(), cycle_scan_limit);

                                    let mut best_gain = 0i32;
                                    let mut best_ia = usize::MAX;
                                    let mut best_ib = usize::MAX;
                                    let mut best_ic = usize::MAX;

                                    for ia in 0..ab_top {
                                        let (node_a, gain_a) = part_to_part[idx_ab][ia];
                                        if part_mut[node_a] as usize != a { continue; }
                                        if gain_a + part_to_part[idx_bc][0].1 + part_to_part[idx_ca][0].1 <= best_gain { break; }
                                        for ib in 0..bc_top {
                                            let (node_b, gain_b) = part_to_part[idx_bc][ib];
                                            if part_mut[node_b] as usize != b { continue; }
                                            let ab_bc = gain_a + gain_b;
                                            if ab_bc + part_to_part[idx_ca][0].1 <= best_gain { break; }
                                            for ic in 0..ca_top {
                                                let (node_c, gain_c) = part_to_part[idx_ca][ic];
                                                if part_mut[node_c] as usize != c { continue; }
                                                let total = ab_bc + gain_c;
                                                if total > best_gain {
                                                    best_gain = total;
                                                    best_ia = ia; best_ib = ib; best_ic = ic;
                                                }
                                                break;
                                            }
                                        }
                                    }

                                    if best_gain > 0 && best_ia < ab_top && best_ib < bc_top && best_ic < ca_top {
                                        let (na, _) = part_to_part[idx_ab][best_ia];
                                        let (nb, _) = part_to_part[idx_bc][best_ib];
                                        let (nc, _) = part_to_part[idx_ca][best_ic];
                                        if part_mut[na] as usize == a && part_mut[nb] as usize == b && part_mut[nc] as usize == c {
                                            part_mut[na] = b as i32;
                                            part_mut[nb] = c as i32;
                                            part_mut[nc] = a as i32;
                                            swap_count += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if swap_count == 0 { break; }
                    if swap_count >= prev_swap_count { stagnant_sw += 1; if stagnant_sw >= 3 { break; } } else { stagnant_sw = 0; }
                    prev_swap_count = swap_count;
                    stream.memcpy_htod(&part_mut, &mut d_partition)?;
                    let mut nip_recount: Vec<i32> = vec![0i32; num_parts];
                    for &p in part_mut.iter() { nip_recount[p as usize] += 1; }
                    stream.memcpy_htod(&nip_recount, &mut d_nip)?;
                }
            }
        }};
    }

    unsafe {
        stream.launch_builder(&cluster_fn)
            .arg(&(n_hedges as i32)).arg(&64i32)
            .arg(&challenge.d_hyperedge_offsets).arg(&challenge.d_hyperedge_nodes)
            .arg(&mut d_hclusters)
            .launch(hedge_cfg.clone())?;
    }
    stream.synchronize()?;

    let num_init_restarts = if n_hedges >= 150_000 { 3 } else if n_hedges >= 75_000 { 5 } else if n_hedges >= 30_000 { 5 } else { 8 };
    let scout_rounds: usize = if n_hedges >= 75_000 { 30 } else if n_hedges < 30_000 { 80 } else { 0 };
    let init_seed: u64 = rng.gen();
    let mut init_rng = SmallRng::seed_from_u64(init_seed);
    for restart in 0..num_init_restarts {
        let restart_seed: u32 = init_rng.gen();
        unsafe {
            stream.launch_builder(&prefs_fn)
                .arg(&(n as i32)).arg(&(num_parts as i32)).arg(&64i32)
                .arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets)
                .arg(&d_hclusters).arg(&challenge.d_hyperedge_offsets)
                .arg(&(restart as i32)).arg(&restart_seed)
                .arg(&mut d_pref_parts).arg(&mut d_pref_prio)
                .launch(node_cfg.clone())?;
        }
        stream.synchronize()?;

        let pp: Vec<i32> = stream.memcpy_dtov(&d_pref_parts)?;
        let pprio: Vec<i32> = stream.memcpy_dtov(&d_pref_prio)?;
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&a, &b| pprio[b].cmp(&pprio[a]));

        let sn: Vec<i32> = idx.iter().map(|&i| i as i32).collect();
        let st: Vec<i32> = idx.iter().map(|&i| pp[i]).collect();
        let d_sn = stream.memcpy_stod(&sn)?;
        let d_st = stream.memcpy_stod(&st)?;
        stream.memset_zeros(&mut d_nip)?;
        unsafe {
            stream.launch_builder(&assign_fn)
                .arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32))
                .arg(&d_sn).arg(&d_st)
                .arg(&mut d_partition).arg(&mut d_nip)
                .launch(one_cfg.clone())?;
        }
        stream.synchronize()?;

        if scout_rounds > 0 {
            tabu.iter_mut().for_each(|x| *x = 0);
            refine!(scout_rounds, 4, move_limit, false, false, true, false);
        }

        let conn = eval_conn!();
        if conn < best_conn {
            best_conn = conn;
            stream.memcpy_dtoh(&d_partition, &mut best_partition)?;
            stream.memcpy_dtoh(&d_nip, &mut best_nip)?;
        }
    }
    stream.memcpy_htod(&best_partition, &mut d_partition)?;
    stream.memcpy_htod(&best_nip, &mut d_nip)?;
    best_conn = i32::MAX;

    let hedge_nodes_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let hedge_offsets_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let node_hedges_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
    let node_offsets_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let np = std::cmp::min(num_parts, 64);
    let cpu_fm_rounds = if n_hedges >= 150_000 { 1 } else if n_hedges >= 75_000 { 2 } else if n_hedges >= 30_000 { 2 } else { 8 };
    let cpu_fm_max_degree: usize = if n_hedges >= 150_000 { 128 } else if n_hedges >= 75_000 { 128 } else { 512 };
    let ils_cpu_fm: usize = if n_hedges < 30_000 { 2 } else { 0 };

    refine!(refinement, 4, move_limit, true, true, false, false);

    if use_gpu_swaps {
        gpu_swap_phase!(35, true);
        tabu.iter_mut().for_each(|x| *x = 0);
        refine!(25, 4, move_limit, false, false, true, false);
        gpu_swap_phase!(15, true);
    }

    best_conn = eval_conn!();
    stream.memcpy_dtoh(&d_partition, &mut best_partition)?;
    stream.memcpy_dtoh(&d_nip, &mut best_nip)?;

    cur_partition.copy_from_slice(&best_partition);
    cur_nip.copy_from_slice(&best_nip);
    cur_conn = best_conn;

    for ils_iter in 0..ils_iters {
        stream.memcpy_htod(&cur_partition, &mut d_partition)?;
        stream.memcpy_htod(&cur_nip, &mut d_nip)?;

        let strong: i32 = if n_hedges >= 75_000 { 3 } else { 2 };
        let ils_perturb: i32 = if ils_iter < ils_iters / 3 { strong }
            else if ils_iter < 2 * ils_iters / 3 { std::cmp::max(1, strong - 1) }
            else { 1 };
        let seed_val: u64 = rng.gen();
        unsafe {
            stream.launch_builder(&perturb_fn)
                .arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32))
                .arg(&ils_perturb)
                .arg(&mut d_partition).arg(&mut d_nip).arg(&seed_val)
                .launch(one_cfg.clone())?;
        }
        stream.synchronize()?;

        tabu.iter_mut().for_each(|x| *x = 0);
        refine!(ils_refine, 4, move_limit, false, false, true, false);

        if use_gpu_swaps {
            gpu_swap_phase!(10, true);
        }

        if ils_cpu_fm > 0 {
            unsafe {
                stream.launch_builder(&precompute_fn)
                    .arg(&(n_hedges as i32)).arg(&(n as i32))
                    .arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&mut d_flags_all).arg(&mut d_flags_double)
                    .launch(hedge_cfg.clone())?;
            }
            stream.synchronize()?;
            let mut ils_fa: Vec<u64> = stream.memcpy_dtov(&d_flags_all)?;
            let mut ils_fd: Vec<u64> = stream.memcpy_dtov(&d_flags_double)?;
            let mut ils_ph: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
            let mut ils_nh: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
            let mut ils_epcnt: Vec<u8> = vec![0u8; n_hedges * np];
            for h in 0..n_hedges {
                let hs = hedge_offsets_data[h] as usize;
                let he = hedge_offsets_data[h + 1] as usize;
                for k in hs..he {
                    let nd = hedge_nodes_data[k] as usize;
                    if nd < n {
                        let p = ils_ph[nd] as usize;
                        if p < np { let idx = h * np + p; if ils_epcnt[idx] < 255 { ils_epcnt[idx] += 1; } }
                    }
                }
            }
            cpu_fm!(ils_cpu_fm, cpu_fm_max_degree, &mut ils_ph, &mut ils_nh, &mut ils_fa, &mut ils_fd, &node_offsets_data, &node_hedges_data, &hedge_offsets_data, &hedge_nodes_data, &mut ils_epcnt);
            stream.memcpy_htod(&ils_ph, &mut d_partition)?;
            for p in ils_nh.iter_mut().take(num_parts) { *p = 0; }
            for &p in ils_ph.iter() { ils_nh[p as usize] += 1; }
            stream.memcpy_htod(&ils_nh, &mut d_nip)?;
        }

        let new_conn = eval_conn!();

        if new_conn < best_conn {
            best_conn = new_conn;
            stream.memcpy_dtoh(&d_partition, &mut best_partition)?;
            stream.memcpy_dtoh(&d_nip, &mut best_nip)?;
        }

        let delta = new_conn as f64 - cur_conn as f64;
        let temp = 1000.0 * (1.0 - (ils_iter as f64 / ils_iters.max(1) as f64)).powi(2);
        let accept = if delta < 0.0 { true }
            else { ((-delta) / temp).exp() > rng.gen::<f64>() };

        if accept {
            cur_conn = new_conn;
            stream.memcpy_dtoh(&d_partition, &mut cur_partition)?;
            stream.memcpy_dtoh(&d_nip, &mut cur_nip)?;
        }
    }

    stream.memcpy_htod(&best_partition, &mut d_partition)?;
    stream.memcpy_htod(&best_nip, &mut d_nip)?;

    tabu.iter_mut().for_each(|x| *x = 0);
    refine!(polish_rounds, 4, move_limit, false, false, false, false);

    balance!();

    tabu.iter_mut().for_each(|x| *x = 0);
    refine!(post_balance, 4, move_limit / 2, false, false, false, false);

    let mut part_host: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
    let mut nip_host: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
    unsafe {
        stream.launch_builder(&precompute_fn)
            .arg(&(n_hedges as i32)).arg(&(n as i32))
            .arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets)
            .arg(&d_partition)
            .arg(&mut d_flags_all).arg(&mut d_flags_double)
            .launch(hedge_cfg.clone())?;
    }
    stream.synchronize()?;

    let mut flags_all_host: Vec<u64> = stream.memcpy_dtov(&d_flags_all)?;
    let mut flags_double_host: Vec<u64> = stream.memcpy_dtov(&d_flags_double)?;

    let swap_rounds = if n_hedges >= 150_000 { 4 } else if n_hedges >= 75_000 { 5 } else { 10 };
    for _sr in 0..swap_rounds {
        let mut swap_lists: Vec<Vec<(usize, i32)>> = vec![Vec::new(); num_parts * num_parts];

        for node in 0..n {
            let src = part_host[node] as usize;
            if src >= num_parts { continue; }
            if nip_host[src] <= 1 { continue; }

            let start = node_offsets_data[node] as usize;
            let end = node_offsets_data[node + 1] as usize;
            let degree = end - start;
            let used = std::cmp::min(degree, 256);
            if used == 0 { continue; }

            let cur_bit: u64 = 1u64 << src;
            let mut part_counts = [0i16; 64];
            let mut count_current = 0i32;
            let np = std::cmp::min(num_parts, 64);

            for j in 0..used {
                let rel = (j * degree) / used;
                let hedge = node_hedges_data[start + rel] as usize;
                let fa = flags_all_host[hedge];
                let fd = flags_double_host[hedge];
                let mask = (fa & !cur_bit) | (fd & cur_bit);
                if mask & cur_bit != 0 { count_current += 1; }
                let mut flags = mask & !cur_bit;
                while flags != 0 {
                    let bit = flags.trailing_zeros() as usize;
                    flags &= flags - 1;
                    if bit < np { part_counts[bit] += 1; }
                }
            }

            let mut best: [(i32, usize); 3] = [(0, src); 3];
            for p in 0..np {
                if p == src { continue; }
                let gain = part_counts[p] as i32 - count_current;
                if gain > 0 {
                    if gain > best[0].0 || (gain == best[0].0 && p < best[0].1) {
                        best[2] = best[1]; best[1] = best[0]; best[0] = (gain, p);
                    } else if gain > best[1].0 || (gain == best[1].0 && p < best[1].1) {
                        best[2] = best[1]; best[1] = (gain, p);
                    } else if gain > best[2].0 || (gain == best[2].0 && p < best[2].1) {
                        best[2] = (gain, p);
                    }
                }
            }

            for &(gain, tgt) in &best {
                if gain > 0 && tgt != src {
                    swap_lists[src * num_parts + tgt].push((node, gain));
                }
            }
        }

        for list in swap_lists.iter_mut() {
            list.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        }

        let mut swap_count = 0usize;
        let mut used_nodes: Vec<bool> = vec![false; n];

        for a in 0..num_parts {
            for b in (a+1)..num_parts {
                let idx_ab = a * num_parts + b;
                let idx_ba = b * num_parts + a;
                if swap_lists[idx_ab].is_empty() || swap_lists[idx_ba].is_empty() { continue; }

                let scan_ba = std::cmp::min(swap_lists[idx_ba].len(), 256);

                for i in 0..swap_lists[idx_ab].len() {
                    let (node_a, gain_a) = swap_lists[idx_ab][i];
                    if used_nodes[node_a] || part_host[node_a] as usize != a { continue; }

                    let mut best_combined = 0i32;
                    let mut best_j = usize::MAX;
                    for j in 0..scan_ba {
                        let (node_b, gain_b) = swap_lists[idx_ba][j];
                        if used_nodes[node_b] || part_host[node_b] as usize != b { continue; }
                        let combined = gain_a + gain_b;
                        if combined > best_combined {
                            best_combined = combined;
                            best_j = j;
                        }
                    }

                    if best_j < swap_lists[idx_ba].len() && best_combined > 0 {
                        let (node_b, _) = swap_lists[idx_ba][best_j];
                        part_host[node_a] = b as i32;
                        part_host[node_b] = a as i32;
                        used_nodes[node_a] = true;
                        used_nodes[node_b] = true;
                        swap_count += 1;
                    }
                }
            }
        }

        let cycle_limit: usize = if n_hedges >= 75_000 { 8 } else { 16 };
        for a in 0..num_parts {
            for b in 0..num_parts {
                if b == a { continue; }
                let idx_ab = a * num_parts + b;
                if swap_lists[idx_ab].is_empty() { continue; }
                let ab_top = std::cmp::min(swap_lists[idx_ab].len(), cycle_limit);
                for c in 0..num_parts {
                    if c == a || c == b { continue; }
                    let idx_bc = b * num_parts + c;
                    let idx_ca = c * num_parts + a;
                    if swap_lists[idx_bc].is_empty() || swap_lists[idx_ca].is_empty() { continue; }
                    let bc_top = std::cmp::min(swap_lists[idx_bc].len(), cycle_limit);
                    let ca_top = std::cmp::min(swap_lists[idx_ca].len(), cycle_limit);

                    let mut best_gain = 0i32;
                    let mut best_ia = usize::MAX;
                    let mut best_ib = usize::MAX;
                    let mut best_ic = usize::MAX;

                    for ia in 0..ab_top {
                        let (node_a, gain_a) = swap_lists[idx_ab][ia];
                        if used_nodes[node_a] || part_host[node_a] as usize != a { continue; }
                        if gain_a + swap_lists[idx_bc][0].1 + swap_lists[idx_ca][0].1 <= best_gain { break; }
                        for ib in 0..bc_top {
                            let (node_b, gain_b) = swap_lists[idx_bc][ib];
                            if used_nodes[node_b] || part_host[node_b] as usize != b { continue; }
                            let ab_bc = gain_a + gain_b;
                            if ab_bc + swap_lists[idx_ca][0].1 <= best_gain { break; }
                            for ic in 0..ca_top {
                                let (node_c, gain_c) = swap_lists[idx_ca][ic];
                                if used_nodes[node_c] || part_host[node_c] as usize != c { continue; }
                                let total = ab_bc + gain_c;
                                if total > best_gain {
                                    best_gain = total;
                                    best_ia = ia; best_ib = ib; best_ic = ic;
                                }
                                break;
                            }
                        }
                    }

                    if best_gain > 0 && best_ia < ab_top && best_ib < bc_top && best_ic < ca_top {
                        let (na, _) = swap_lists[idx_ab][best_ia];
                        let (nb, _) = swap_lists[idx_bc][best_ib];
                        let (nc, _) = swap_lists[idx_ca][best_ic];
                        if !used_nodes[na] && !used_nodes[nb] && !used_nodes[nc]
                            && part_host[na] as usize == a && part_host[nb] as usize == b && part_host[nc] as usize == c
                        {
                            part_host[na] = b as i32;
                            part_host[nb] = c as i32;
                            part_host[nc] = a as i32;
                            used_nodes[na] = true;
                            used_nodes[nb] = true;
                            used_nodes[nc] = true;
                            swap_count += 1;
                        }
                    }
                }
            }
        }

        if swap_count == 0 { break; }

        stream.memcpy_htod(&part_host, &mut d_partition)?;
        for p in nip_host.iter_mut() { *p = 0; }
        for &p in part_host.iter() { nip_host[p as usize] += 1; }
        stream.memcpy_htod(&nip_host, &mut d_nip)?;

        unsafe {
            stream.launch_builder(&precompute_fn)
                .arg(&(n_hedges as i32)).arg(&(n as i32))
                .arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&mut d_flags_all).arg(&mut d_flags_double)
                .launch(hedge_cfg.clone())?;
        }
        stream.synchronize()?;

        flags_all_host = stream.memcpy_dtov(&d_flags_all)?;
        flags_double_host = stream.memcpy_dtov(&d_flags_double)?;
    }

    if cpu_fm_rounds > 0 {
        let mut edge_pcnt: Vec<u8> = vec![0u8; n_hedges * np];
        for h in 0..n_hedges {
            let hs = hedge_offsets_data[h] as usize;
            let he = hedge_offsets_data[h + 1] as usize;
            for k in hs..he {
                let nd = hedge_nodes_data[k] as usize;
                if nd < n {
                    let p = part_host[nd] as usize;
                    if p < np { let idx = h * np + p; if edge_pcnt[idx] < 255 { edge_pcnt[idx] += 1; } }
                }
            }
        }
        cpu_fm!(cpu_fm_rounds, cpu_fm_max_degree, &mut part_host, &mut nip_host, &mut flags_all_host, &mut flags_double_host, &node_offsets_data, &node_hedges_data, &hedge_offsets_data, &hedge_nodes_data, &mut edge_pcnt);
    }

    stream.memcpy_htod(&part_host, &mut d_partition)?;
    for p in nip_host.iter_mut().take(num_parts) { *p = 0; }
    for &p in part_host.iter() { nip_host[p as usize] += 1; }
    stream.memcpy_htod(&nip_host, &mut d_nip)?;

    let final_partition: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
    save_solution(&Solution { partition: final_partition.iter().map(|&x| x as u32).collect() })?;

    Ok(())
}
