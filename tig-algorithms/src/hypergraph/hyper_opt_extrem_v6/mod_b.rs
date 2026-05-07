use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::hypergraph::*;

mod track_10k {
    use super::TrackConfig;
    pub fn config() -> TrackConfig {
        TrackConfig {

            refinement: 1000,
            ils: 75,
            ils_refine: 90,
            polish: 100,
            post_balance: 60,
            move_limit: 50_000,
            extra_window: 16_384,
            tabu_tenure: 10,
            use_gpu_swaps: true,
            num_init_restarts: 1,
            scout_rounds: 0,
            vcycle_depth: 0,
            vcycle_ils_freq: 0,
            vcycle_post_refine: 0,
            neg_gain_thresh: 5,
            swap_scan_limit: 96,
            cycle_scan_limit: 32,
            cpu_fm_rounds: 12,
            cpu_fm_max_degree: 512,
            ils_cpu_fm: 3,
            final_swap_rounds: 5,
            final_cycle_limit: 32,
            max_coarsen_levels: 6,
            min_coarse_nodes: 500,
            perturb_strong: 2,
            perturb_stagnant_thresh: 5,
            perturb_stagnant_strength: 1,
            balance_weight: 4,
            stagnation_limit: 20,
            use_tiebreaker: true,
            init_swap_rounds1: 20,
            init_swap_rounds2: 0,
            ils_swap_rounds: 5,
        }
    }
}
mod track_20k {
    use super::TrackConfig;
    pub fn config() -> TrackConfig {
        TrackConfig {

            refinement: 600,
            ils: 7,
            ils_refine: 100,
            polish: 150,
            post_balance: 150,
            move_limit: 200_000,
            extra_window: 200_000,
            tabu_tenure: 8,
            use_gpu_swaps: true,
            num_init_restarts: 1,
            scout_rounds: 0,
            vcycle_depth: 0,
            vcycle_ils_freq: 0,
            vcycle_post_refine: 0,
            neg_gain_thresh: 5,
            swap_scan_limit: 64,
            cycle_scan_limit: 20,
            cpu_fm_rounds: 0,
            cpu_fm_max_degree: 512,
            ils_cpu_fm: 0,
            final_swap_rounds: 5,
            final_cycle_limit: 5,
            max_coarsen_levels: 0,
            min_coarse_nodes: 400,
            perturb_strong: 3,
            perturb_stagnant_thresh: 9999,
            perturb_stagnant_strength: 1,
            balance_weight: 2,
            stagnation_limit: 30,
            use_tiebreaker: true,
            init_swap_rounds1: 15,
            init_swap_rounds2: 10,
            ils_swap_rounds: 5,
        }
    }
}
mod track_50k {
    use super::TrackConfig;
    pub fn config() -> TrackConfig {
        TrackConfig {

            refinement: 1500,
            ils: 15,
            ils_refine: 60,
            polish: 150,
            post_balance: 60,
            move_limit: 200_000,
            extra_window: 57_344,
            tabu_tenure: 12,
            use_gpu_swaps: true,
            num_init_restarts: 1,
            scout_rounds: 0,
            vcycle_depth: 0,
            vcycle_ils_freq: 0,
            vcycle_post_refine: 0,
            neg_gain_thresh: 5,
            swap_scan_limit: 80,
            cycle_scan_limit: 16,
            cpu_fm_rounds: 3,
            cpu_fm_max_degree: 256,
            ils_cpu_fm: 0,
            final_swap_rounds: 15,
            final_cycle_limit: 16,
            max_coarsen_levels: 5,
            min_coarse_nodes: 800,
            perturb_strong: 2,
            perturb_stagnant_thresh: 4,
            perturb_stagnant_strength: 2,
            balance_weight: 4,
            stagnation_limit: 20,
            use_tiebreaker: false,
            init_swap_rounds1: 30,
            init_swap_rounds2: 0,
            ils_swap_rounds: 6,
        }
    }
}
mod track_100k {
    use super::TrackConfig;
    pub fn config() -> TrackConfig {
        TrackConfig {

            refinement: 500,
            ils: 6,
            ils_refine: 40,
            polish: 80,
            post_balance: 30,
            move_limit: 200_000,
            extra_window: 61_440,
            tabu_tenure: 12,
            use_gpu_swaps: true,
            num_init_restarts: 1,
            scout_rounds: 0,
            vcycle_depth: 0,
            vcycle_ils_freq: 0,
            vcycle_post_refine: 0,
            neg_gain_thresh: 5,
            swap_scan_limit: 32,
            cycle_scan_limit: 8,
            cpu_fm_rounds: 0,
            cpu_fm_max_degree: 64,
            ils_cpu_fm: 0,
            final_swap_rounds: 3,
            final_cycle_limit: 8,
            max_coarsen_levels: 4,
            min_coarse_nodes: 2000,
            perturb_strong: 3,
            perturb_stagnant_thresh: 3,
            perturb_stagnant_strength: 3,
            balance_weight: 4,
            stagnation_limit: 24,
            use_tiebreaker: true,
            init_swap_rounds1: 10,
            init_swap_rounds2: 0,
            ils_swap_rounds: 2,
        }
    }
}
mod track_200k {
    use super::TrackConfig;

    pub fn config() -> TrackConfig {
        TrackConfig {

            refinement: 350,
            ils: 4,
            ils_refine: 30,
            polish: 50,
            post_balance: 30,
            move_limit: 200_000,
            extra_window: 65_536,
            tabu_tenure: 14,
            use_gpu_swaps: true,
            swap_scan_limit: 32,
            cycle_scan_limit: 8,
            num_init_restarts: 1,
            scout_rounds: 0,
            vcycle_depth: 0,
            vcycle_ils_freq: 0,
            vcycle_post_refine: 0,
            cpu_fm_rounds: 0,
            cpu_fm_max_degree: 128,
            ils_cpu_fm: 0,
            final_swap_rounds: 2,
            final_cycle_limit: 5,
            max_coarsen_levels: 4,
            min_coarse_nodes: 2000,
            neg_gain_thresh: 5,
            perturb_strong: 3,
            perturb_stagnant_thresh: 3,
            perturb_stagnant_strength: 3,
            balance_weight: 4,
            stagnation_limit: 20,
            use_tiebreaker: true,
            init_swap_rounds1: 5,
            init_swap_rounds2: 0,
            ils_swap_rounds: 1,
        }
    }
}

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
    println!("HyperOpt Extrem V5b");
    println!("Hypergraph k-way partitioning with guided perturbation,");
    println!("boundary-only refinement, incremental edge flags, and heavy swap phases.");
    println!();
    println!("Tracks: T21(10K) T23(20K) T25(50K) T22(100K) T24(200K)");
    println!("HP: refinement, ils, ils_refine, polish, post_balance, move_limit");
}

pub(crate) struct TrackConfig {
    pub refinement: usize,
    pub ils: usize,
    pub ils_refine: usize,
    pub polish: usize,
    pub post_balance: usize,
    pub move_limit: usize,
    pub extra_window: usize,
    pub tabu_tenure: usize,
    pub use_gpu_swaps: bool,
    pub num_init_restarts: usize,
    pub scout_rounds: usize,
    pub vcycle_depth: usize,
    pub vcycle_ils_freq: usize,
    pub vcycle_post_refine: usize,
    pub neg_gain_thresh: i32,
    pub swap_scan_limit: usize,
    pub cycle_scan_limit: usize,
    pub cpu_fm_rounds: usize,
    pub cpu_fm_max_degree: usize,
    pub ils_cpu_fm: usize,
    pub final_swap_rounds: usize,
    pub final_cycle_limit: usize,
    pub max_coarsen_levels: usize,
    pub min_coarse_nodes: usize,
    pub perturb_strong: i32,
    pub perturb_stagnant_thresh: usize,
    pub perturb_stagnant_strength: i32,
    pub balance_weight: i32,
    pub stagnation_limit: usize,
    pub use_tiebreaker: bool,
    pub init_swap_rounds1: usize,
    pub init_swap_rounds2: usize,
    pub ils_swap_rounds: usize,
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let cfg = match challenge.num_hyperedges {
        10000 => track_10k::config(),
        20000 => track_20k::config(),
        50000 => track_50k::config(),
        100000 => track_100k::config(),
        200000 => track_200k::config(),
        _ => track_10k::config(),
    };
    solve_core(cfg, challenge, save_solution, hyperparameters, module, stream, prop)
}

struct CoarseLevel { n: usize, nh: usize, ho: Vec<i32>, hn: Vec<i32>, no: Vec<i32>, ne: Vec<i32>, nw: Vec<i32>, cm: Vec<usize> }

fn heavy_edge_matching(n: usize, _nh: usize, ho: &[i32], hn: &[i32], no: &[i32], ne: &[i32], nw: &[i32], rng: &mut SmallRng) -> (Vec<usize>, usize) {
    let mut wbuf = vec![0i32; n]; let mut touched: Vec<usize> = Vec::with_capacity(512);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..order.len()).rev() { let j = rng.gen_range(0..=i); order.swap(i, j); }
    let mut matched = vec![false; n]; let mut leader = vec![usize::MAX; n];
    for &u in &order {
        if matched[u] { continue; }
        touched.clear();
        let us = no[u] as usize; let ue = no[u + 1] as usize;
        for j in us..ue {
            let h = ne[j] as usize; let hs = ho[h] as usize; let he = ho[h + 1] as usize; let hsz = he - hs;
            if hsz > 50 || hsz < 2 { continue; }
            let w = 1000 / hsz as i32;
            for k in hs..he { let v = hn[k] as usize; if v != u && !matched[v] { if wbuf[v] == 0 { touched.push(v); } wbuf[v] += w; } }
        }
        let mut best = usize::MAX; let mut bw = 0i32;
        for &v in &touched { let w = wbuf[v]; if w > bw || (w == bw && w > 0 && nw[v] < nw.get(best).copied().unwrap_or(i32::MAX)) { bw = w; best = v; } wbuf[v] = 0; }
        if best < n && bw > 0 { matched[u] = true; matched[best] = true; leader[u] = u; leader[best] = u; } else { leader[u] = u; }
    }
    let mut cmap = vec![0usize; n]; let mut lid_map = vec![usize::MAX; n]; let mut next_id = 0usize;
    for u in 0..n {
        let l = leader[u];
        if l == usize::MAX { cmap[u] = next_id; next_id += 1; }
        else if l == u { lid_map[u] = next_id; cmap[u] = next_id; next_id += 1; }
        else { if lid_map[l] == usize::MAX { lid_map[l] = next_id; cmap[l] = next_id; next_id += 1; } cmap[u] = lid_map[l]; }
    }
    (cmap, next_id)
}

fn contract_level(n: usize, nh: usize, ho: &[i32], hn: &[i32], nw: &[i32], cmap: &[usize], nc: usize) -> CoarseLevel {
    let mut cnw = vec![0i32; nc]; for u in 0..n { cnw[cmap[u]] += nw[u]; }
    let mut cho = vec![0i32; 1]; let mut chn: Vec<i32> = Vec::new(); let mut temp: Vec<usize> = Vec::new();
    for h in 0..nh { let hs = ho[h] as usize; let he = ho[h + 1] as usize; temp.clear(); for k in hs..he { temp.push(cmap[hn[k] as usize]); } temp.sort_unstable(); temp.dedup(); if temp.len() >= 2 { for &cn in &temp { chn.push(cn as i32); } cho.push(chn.len() as i32); } }
    let cnh = cho.len() - 1;
    let mut cnt = vec![0usize; nc]; for h in 0..cnh { for k in cho[h] as usize..cho[h+1] as usize { cnt[chn[k] as usize] += 1; } }
    let mut cno = vec![0i32; nc + 1]; for u in 0..nc { cno[u + 1] = cno[u] + cnt[u] as i32; }
    let mut cne = vec![0i32; cno[nc] as usize]; let mut ins = vec![0usize; nc];
    for h in 0..cnh { for k in cho[h] as usize..cho[h+1] as usize { let u = chn[k] as usize; cne[cno[u] as usize + ins[u]] = h as i32; ins[u] += 1; } }
    CoarseLevel { n: nc, nh: cnh, ho: cho, hn: chn, no: cno, ne: cne, nw: cnw, cm: cmap.to_vec() }
}

fn build_coarse_levels(n: usize, nh: usize, ho: &[i32], hn: &[i32], no: &[i32], ne: &[i32], max_levels: usize, min_nodes: usize, rng: &mut SmallRng) -> Vec<CoarseLevel> {
    let mut levels: Vec<CoarseLevel> = Vec::new();
    levels.push(CoarseLevel { n, nh, ho: ho.to_vec(), hn: hn.to_vec(), no: no.to_vec(), ne: ne.to_vec(), nw: vec![1i32; n], cm: Vec::new() });
    for _ in 0..max_levels {
        let prev = levels.last().unwrap(); if prev.n <= min_nodes { break; }
        let (cmap, nc) = heavy_edge_matching(prev.n, prev.nh, &prev.ho, &prev.hn, &prev.no, &prev.ne, &prev.nw, rng);
        if nc as f64 > prev.n as f64 * 0.85 { break; }
        let coarse = contract_level(prev.n, prev.nh, &prev.ho, &prev.hn, &prev.nw, &cmap, nc);
        if coarse.nh == 0 { break; }
        levels.push(coarse);
    }
    levels
}

fn cpu_fm_weighted(n: usize, nh: usize, num_parts: usize, max_pw: i32, ho: &[i32], hn: &[i32], no: &[i32], ne: &[i32], nw: &[i32], partition: &mut [i32], nip: &mut [i32], rounds: usize, max_deg: usize) {
    let np = std::cmp::min(num_parts, 64);
    let mut fa = vec![0u64; nh]; let mut fd = vec![0u64; nh];
    for h in 0..nh { let hs = ho[h] as usize; let he = ho[h+1] as usize; for k in hs..he { let p = partition[hn[k] as usize] as usize; if p < 64 { let bit = 1u64 << p; fd[h] |= fa[h] & bit; fa[h] |= bit; } } }
    let mut epcnt = vec![0u8; nh * np];
    for h in 0..nh { let hs = ho[h] as usize; let he = ho[h+1] as usize; for k in hs..he { let p = partition[hn[k] as usize] as usize; if p < np { let idx = h * np + p; if epcnt[idx] < 255 { epcnt[idx] += 1; } } } }
    for _ in 0..rounds {
        let mut moves = 0usize;
        for node in 0..n {
            let src = partition[node] as usize; let w = nw[node];
            if src >= np || nip[src] <= w { continue; }
            let start = no[node] as usize; let end = no[node + 1] as usize; let degree = end - start;
            if degree == 0 || degree > max_deg { continue; }
            let cur_bit = 1u64 << src; let mut part_counts = [0i16; 64]; let mut count_cur = 0i32;
            for j in start..end { let h = ne[j] as usize; let mask = (fa[h] & !cur_bit) | (fd[h] & cur_bit); if mask & cur_bit != 0 { count_cur += 1; } let mut fl = mask & !cur_bit; while fl != 0 { let bit = fl.trailing_zeros() as usize; fl &= fl - 1; if bit < np { part_counts[bit] += 1; } } }
            let mut best_gain = 0i32; let mut best_tgt = src;
            for p in 0..np { if p == src { continue; } if nip[p] + w > max_pw { continue; } let gain = part_counts[p] as i32 - count_cur; if gain > best_gain || (gain == best_gain && gain > 0 && p < best_tgt) { best_gain = gain; best_tgt = p; } }
            if best_gain <= 0 || best_tgt == src { continue; }
            partition[node] = best_tgt as i32; nip[src] -= w; nip[best_tgt] += w; moves += 1;
            for j in start..end { let h = ne[j] as usize; let is = h*np+src; let it = h*np+best_tgt; let os = epcnt[is]; let ot = epcnt[it]; if os > 0 { epcnt[is] -= 1; } if epcnt[it] < 255 { epcnt[it] += 1; } let ns = epcnt[is]; let nt = epcnt[it]; let sb = 1u64 << src; let tb = 1u64 << best_tgt; if ns == 0 { fa[h] &= !sb; } if ot == 0 { fa[h] |= tb; } if ns < 2 { fd[h] &= !sb; } if nt >= 2 { fd[h] |= tb; } }
        }
        if moves == 0 { break; }
    }
}

fn vcycle_refine(levels: &[CoarseLevel], num_parts: usize, max_ps: i32, partition: &mut Vec<i32>, nip: &mut Vec<i32>, depth: usize) {
    if levels.len() < 2 || depth == 0 { return; }
    let target_level = std::cmp::min(depth, levels.len() - 1);
    let mut cur_part = partition.clone();
    for lvl in 1..=target_level {
        let coarser = &levels[lvl]; let finer = &levels[lvl - 1]; let cn = coarser.n; let np = std::cmp::min(num_parts, 64);
        let mut votes = vec![vec![0i32; np]; cn];
        for u in 0..finer.n { let cu = coarser.cm[u]; let p = cur_part[u] as usize; if p < np { votes[cu][p] += finer.nw[u]; } }
        let mut new_part = vec![0i32; cn]; let mut new_nip = vec![0i32; num_parts];
        for cu in 0..cn { let mut bp = 0; let mut bv = votes[cu][0]; for p in 1..np { if votes[cu][p] > bv { bv = votes[cu][p]; bp = p; } } new_part[cu] = bp as i32; new_nip[bp] += coarser.nw[cu]; }
        cur_part = new_part;
    }
    let clvl = &levels[target_level];
    let mut c_nip = vec![0i32; num_parts]; for u in 0..clvl.n { c_nip[cur_part[u] as usize] += clvl.nw[u]; }
    let fm_r = if clvl.n > 20000 { 3 } else if clvl.n > 5000 { 8 } else { 30 };
    let fm_d = if clvl.n > 20000 { 128 } else { 512 };
    cpu_fm_weighted(clvl.n, clvl.nh, num_parts, max_ps, &clvl.ho, &clvl.hn, &clvl.no, &clvl.ne, &clvl.nw, &mut cur_part, &mut c_nip, fm_r, fm_d);
    for lvl in (1..=target_level).rev() {
        let finer = &levels[lvl - 1]; let coarser = &levels[lvl];
        let mut f_part = vec![0i32; finer.n]; for u in 0..finer.n { f_part[u] = cur_part[coarser.cm[u]]; }
        let mut f_nip = vec![0i32; num_parts]; for u in 0..finer.n { f_nip[f_part[u] as usize] += finer.nw[u]; }
        let fm_r = if finer.n > 50000 { 2 } else if finer.n > 10000 { 4 } else { 10 };
        let md = if finer.n > 50000 { 128 } else { 256 };
        cpu_fm_weighted(finer.n, finer.nh, num_parts, max_ps, &finer.ho, &finer.hn, &finer.no, &finer.ne, &finer.nw, &mut f_part, &mut f_nip, fm_r, md);
        cur_part = f_part; c_nip = f_nip;
    }
    partition.copy_from_slice(&cur_part); nip.copy_from_slice(&c_nip);
}

fn recursive_tree_init(
    n: usize, n_hedges: usize,
    ho: &[i32], hn: &[i32],
    no: &[i32], ne: &[i32],
    num_parts: usize,
) -> Vec<i32> {
    if num_parts == 0 || !num_parts.is_power_of_two() { return Vec::new(); }
    let depth = num_parts.trailing_zeros() as usize;
    let mut part = vec![0i32; n];
    let nw = vec![1i32; n];
    for level in 0..depth {
        let num_parents = 1usize << level;
        for parent_id in 0..num_parents {
            let nodes_in_parent: Vec<usize> = (0..n)
                .filter(|&u| part[u] == parent_id as i32)
                .collect();
            if nodes_in_parent.is_empty() { continue; }
            let subtree_size = nodes_in_parent.len();
            let half = ((subtree_size + 1) / 2) as i32;

            let limit = (subtree_size * 55 / 100) as i32;
            let mut temp_part = vec![2i32; n];
            let mut cnt0 = 0i32;
            for &u in nodes_in_parent.iter() {
                if cnt0 < half { temp_part[u] = 0; cnt0 += 1; }
                else { temp_part[u] = 1; }
            }
            let mut nip2 = vec![cnt0, subtree_size as i32 - cnt0];

            cpu_fm_weighted(n, n_hedges, 2, limit, ho, hn, no, ne, &nw,
                            &mut temp_part, &mut nip2, 20, 512);
            for &u in nodes_in_parent.iter() {
                part[u] = if temp_part[u] == 0 { (parent_id * 2) as i32 } else { (parent_id * 2 + 1) as i32 };
            }
        }
    }
    part
}

pub(crate) fn solve_core(
    tc: TrackConfig, challenge: &Challenge, save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>, module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let dummy: Vec<u32> = (0..challenge.num_nodes as u32).map(|i| i % challenge.num_parts as u32).collect();
    save_solution(&Solution { partition: dummy })?;

    let get_hp = |name: &str, default: usize, min: i64, max: i64| -> usize {
        hyperparameters.as_ref().and_then(|p| p.get(name).and_then(|v| v.as_i64())).map(|v| v.clamp(min, max) as usize).unwrap_or(default)
    };
    let n = challenge.num_nodes as usize;
    let num_parts = challenge.num_parts as usize;
    let n_hedges = challenge.num_hyperedges as usize;
    let max_ps = challenge.max_part_size as usize;

    let refinement = get_hp("refinement", tc.refinement, 50, 50000);
    let ils_iters = get_hp("ils", tc.ils, 0, 2000);
    let ils_refine = get_hp("ils_refine", tc.ils_refine, 10, 2000);
    let polish_rounds = get_hp("polish", tc.polish, 0, 5000);
    let post_balance = get_hp("post_balance", tc.post_balance, 0, 2000);
    let move_limit = get_hp("move_limit", tc.move_limit, 1000, 1000000);
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);
    let extra_window = tc.extra_window;
    let tabu_tenure = tc.tabu_tenure;
    let neg_gain_thresh = tc.neg_gain_thresh;
    let swap_scan_limit = tc.swap_scan_limit;
    let cycle_scan_limit = tc.cycle_scan_limit;

    let hedge_nodes_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let hedge_offsets_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let node_hedges_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
    let node_offsets_data: Vec<i32> = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let mut rng = SmallRng::from_seed({ let mut s = [0u8; 32]; s[..challenge.seed.len().min(32)].copy_from_slice(&challenge.seed[..challenge.seed.len().min(32)]); s });

    let levels = build_coarse_levels(n, n_hedges, &hedge_offsets_data, &hedge_nodes_data, &node_offsets_data, &node_hedges_data, tc.max_coarsen_levels, tc.min_coarse_nodes, &mut rng);

    let cluster_fn = module.load_function("hyperedge_clustering")?;
    let prefs_fn = module.load_function("compute_node_preferences")?;
    let assign_fn = module.load_function("execute_node_assignments")?;
    let precompute_fn = module.load_function("precompute_edge_flags")?;
    let compute_moves_fn = module.load_function("compute_moves_flags_weighted")?;
    let perturb_fn = module.load_function("perturb_solution")?;
    let perturb_guided_fn = module.load_function("perturb_guided")?;
    let perturb_targeted_fn = module.load_function("perturb_targeted")?;
    let conn_per_hedge_fn = module.load_function("compute_connectivity_per_hedge")?;
    let exec_moves_fn = module.load_function("execute_selected_moves")?;
    let lp_step_fn = module.load_function("label_propagation_step")?;
    let rebalance_fn = module.load_function("rebalance_greedy")?;
    let mark_boundary_fn = module.load_function("mark_boundary_nodes")?;
    let compute_moves_boundary_fn = module.load_function("compute_moves_boundary_weighted")?;
    let compute_moves_boundary_thermal_fn = module.load_function("compute_moves_boundary_thermal")?;
    let mark_dirty_fn = module.load_function("mark_dirty_edges")?;
    let update_dirty_fn = module.load_function("update_dirty_edge_flags")?;
    let reset_fn = module.load_function("reset_counters")?;
    let balance_fn = module.load_function("balance_final")?;
    let conn_fn = module.load_function("my_calc_connectivity")?;
    let swap_gains_fn = if tc.use_gpu_swaps { Some(module.load_function("compute_swap_gains_top3")?) } else { None };
    let init_weights_fn = module.load_function("init_hyperedge_weights")?;
    let update_weights_fn = module.load_function("update_hyperedge_weights_dcw")?;
    let decay_weights_fn = module.load_function("decay_hyperedge_weights")?;

    let one_cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
    let node_cfg = LaunchConfig { grid_dim: ((n as u32 + block_size - 1) / block_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
    let hedge_cfg = LaunchConfig { grid_dim: ((n_hedges as u32 + block_size - 1) / block_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };

    let mut d_hclusters = stream.alloc_zeros::<i32>(n_hedges)?;
    let mut d_partition = stream.alloc_zeros::<i32>(n)?;
    let mut d_nip = stream.alloc_zeros::<i32>(num_parts)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(n)?;
    let mut d_pref_prio = stream.alloc_zeros::<i32>(n)?;
    let mut d_move_prio = stream.alloc_zeros::<i32>(n)?;
    let mut d_num_valid = stream.alloc_zeros::<i32>(1)?;
    let mut d_flags_all = stream.alloc_zeros::<u64>(n_hedges)?;
    let mut d_flags_double = stream.alloc_zeros::<u64>(n_hedges)?;
    let mut d_swap_gains = if tc.use_gpu_swaps { Some(stream.alloc_zeros::<i32>(n * 3)?) } else { None };
    let mut d_is_boundary = stream.alloc_zeros::<i32>(n)?;
    let mut d_boundary_count = stream.alloc_zeros::<i32>(1)?;
    let mut d_edge_dirty = stream.alloc_zeros::<i32>(n_hedges)?;
    let mut d_moved_nodes = stream.alloc_zeros::<i32>(n)?;
    let mut d_moves_executed = stream.alloc_zeros::<i32>(1)?;
    let mut d_hedge_weights = stream.alloc_zeros::<i32>(n_hedges)?;
    let mut d_sorted_move_nodes = stream.alloc_zeros::<i32>(n)?;
    let mut d_sorted_move_parts = stream.alloc_zeros::<i32>(n)?;
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(n);
    let mut sorted_nodes_buf: Vec<i32> = Vec::with_capacity(n);
    let mut sorted_targets_buf: Vec<i32> = Vec::with_capacity(n);
    let mut best_partition: Vec<i32> = vec![0i32; n];
    let mut best_nip: Vec<i32> = vec![0i32; num_parts];
    let mut best_conn: i32 = i32::MAX;
    let mut cur_partition: Vec<i32> = vec![0i32; n];
    let mut cur_nip: Vec<i32> = vec![0i32; num_parts];
    let mut cur_conn: i32;
    let mut tabu: Vec<usize> = vec![0; n];
    let mut tgt_quota: Vec<usize> = vec![0; num_parts];
    let mut tgt_used: Vec<usize> = vec![0; num_parts];
    let mut swap_gains_host: Vec<i32> = if tc.use_gpu_swaps { vec![0i32; n * 3] } else { vec![] };
    let mut part_to_part: Vec<Vec<(usize, i32)>> = if tc.use_gpu_swaps { vec![Vec::new(); num_parts * num_parts] } else { vec![] };
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut keys_buf: Vec<i32> = vec![0i32; n];
    let mut exec_part: Vec<i32> = vec![0i32; n];
    let mut exec_nip: Vec<i32> = vec![0i32; num_parts];

    let mut moved_nodes_buf: Vec<i32> = Vec::with_capacity(n);
    let (slack_early, slack_mid, slack_late) = (8usize, 4usize, 2usize);

    macro_rules! refine {
        ($max_rounds:expr, $bw:expr, $mlimit:expr, $use_tabu:expr, $use_adaptive:expr, $quick:expr, $cap_aware:expr) => {{
            refine!($max_rounds, $bw, $mlimit, $use_tabu, $use_adaptive, $quick, $cap_aware, 0.0f64)
        }};
        ($max_rounds:expr, $bw:expr, $mlimit:expr, $use_tabu:expr, $use_adaptive:expr, $quick:expr, $cap_aware:expr, $init_temp:expr) => {{
            let bw: i32 = $bw; let mlimit: usize = $mlimit; let use_tabu: bool = $use_tabu; let use_adaptive: bool = $use_adaptive; let quick: bool = $quick; let cap_aware: bool = $cap_aware;
            let init_temperature: f64 = $init_temp;
            let mut stagnant = 0usize; let mut low_move_streak = 0usize;
            stream.memcpy_dtoh(&d_partition, &mut exec_part)?; stream.memcpy_dtoh(&d_nip, &mut exec_nip)?;
            let use_boundary_base = n_hedges < 100000;
            let mut use_incremental = false;
            for round in 0..$max_rounds {

                let use_boundary = use_boundary_base && round < ($max_rounds * 7 / 10);

                if !use_incremental || (use_boundary && round % 20 == 0) || (!use_boundary) {
                    unsafe { stream.launch_builder(&precompute_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
                }
                unsafe { stream.launch_builder(&reset_fn).arg(&mut d_num_valid).launch(one_cfg.clone())?; }
                let mud = 256i32;
                let neg_thresh = if n_hedges >= 100000 { 1i32 } else { 0i32 };

                let t_min = 0.05f32;
                let tpls_active = init_temperature > 0.0 && use_boundary;
                let current_temp = if tpls_active && $max_rounds > 1 {
                    let t_init = init_temperature as f32;
                    t_init * (t_min / t_init).powf(round as f32 / ($max_rounds as f32 - 1.0))
                } else { 0.0f32 };
                let tpls_seed: u32 = if tpls_active { rng.gen::<u32>() } else { 0 };
                if use_boundary {

                    stream.memset_zeros(&mut d_boundary_count)?;
                    unsafe { stream.launch_builder(&mark_boundary_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_partition).arg(&d_flags_all).arg(&mut d_is_boundary).arg(&mut d_boundary_count).launch(node_cfg.clone())?; }
                    if tpls_active {

                        unsafe { stream.launch_builder(&compute_moves_boundary_thermal_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32)).arg(&bw).arg(&mud).arg(&current_temp).arg(&tpls_seed).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_partition).arg(&d_nip).arg(&d_flags_all).arg(&d_flags_double).arg(&d_hedge_weights).arg(&d_is_boundary).arg(&mut d_move_prio).arg(&mut d_num_valid).launch(node_cfg.clone())?; }
                    } else {
                        unsafe { stream.launch_builder(&compute_moves_boundary_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32)).arg(&bw).arg(&mud).arg(&neg_thresh).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_partition).arg(&d_nip).arg(&d_flags_all).arg(&d_flags_double).arg(&d_hedge_weights).arg(&d_is_boundary).arg(&mut d_move_prio).arg(&mut d_num_valid).launch(node_cfg.clone())?; }
                    }
                } else {

                    let tb = if tc.use_tiebreaker { 1i32 } else { 0i32 };
                    unsafe { stream.launch_builder(&compute_moves_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32)).arg(&bw).arg(&mud).arg(&neg_thresh).arg(&tb).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_partition).arg(&d_nip).arg(&d_flags_all).arg(&d_flags_double).arg(&d_hedge_weights).arg(&mut d_move_prio).arg(&mut d_num_valid).launch(node_cfg.clone())?; }
                }
                stream.synchronize()?;
                let nv = stream.memcpy_dtov(&d_num_valid)?[0];
                if nv == 0 { if quick { break; } stagnant += 1; if stagnant > tc.stagnation_limit { break; } continue; }
                { stream.memcpy_dtoh(&d_move_prio, &mut keys_buf)?; valid_moves.clear();
                  let max_gain = keys_buf.iter().map(|&k| k >> 16).max().unwrap_or(0); let aspiration = (max_gain * 3) / 4;
                  for (node, &key) in keys_buf.iter().enumerate() { if key > 0 { if !use_tabu || tabu[node] <= round || (key >> 16) >= aspiration { valid_moves.push((node, key)); } } } }
                if valid_moves.is_empty() { if quick { break; } stagnant += 1; if stagnant > tc.stagnation_limit { break; } continue; }
                let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
                let k_base_limit = if !use_adaptive { mlimit } else if round < 50 { mlimit / 2 } else if round < 200 { mlimit } else { mlimit / 3 };
                let k_base = std::cmp::min(valid_moves.len(), k_base_limit);
                let k_cand = if cap_aware { std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window)) } else { k_base };
                if k_cand > 0 && k_cand < valid_moves.len() { valid_moves.select_nth_unstable_by(k_cand - 1, cmp); }
                valid_moves[..k_cand].sort_unstable_by(cmp);
                sorted_nodes_buf.clear(); sorted_targets_buf.clear();
                if cap_aware && k_cand > k_base {
                    let nip_host: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
                    let slack = if round < 64 { slack_early } else if round < 256 { slack_mid } else { slack_late };
                    for p in 0..num_parts { let free = max_ps.saturating_sub(nip_host[p] as usize); tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack)); tgt_used[p] = 0; }
                    let mut selected = 0usize;
                    for i in 0..k_cand { if selected >= k_base { break; } let (nd, key) = valid_moves[i]; let tgt = (key & 63) as usize; if tgt < num_parts && tgt_used[tgt] < tgt_quota[tgt] { sorted_nodes_buf.push(nd as i32); sorted_targets_buf.push(tgt as i32); tgt_used[tgt] += 1; selected += 1; } }
                    if sorted_nodes_buf.is_empty() { for i in 0..k_base { sorted_nodes_buf.push(valid_moves[i].0 as i32); sorted_targets_buf.push((valid_moves[i].1 & 63) as i32); } }
                } else { for i in 0..k_cand { sorted_nodes_buf.push(valid_moves[i].0 as i32); sorted_targets_buf.push((valid_moves[i].1 & 63) as i32); } }

                moved_nodes_buf.clear();
                let mut me = 0i32; let mps_i32 = challenge.max_part_size as i32;
                for i in 0..sorted_nodes_buf.len() { let node = sorted_nodes_buf[i] as usize; let target = sorted_targets_buf[i]; let src = exec_part[node]; if src == target { continue; } if exec_nip[target as usize] >= mps_i32 { continue; } if exec_nip[src as usize] <= 1 { continue; } exec_part[node] = target; exec_nip[src as usize] -= 1; exec_nip[target as usize] += 1; moved_nodes_buf.push(node as i32); me += 1; }
                if me > 0 {
                    stream.memcpy_htod(&exec_part, &mut d_partition)?;
                    stream.memcpy_htod(&exec_nip, &mut d_nip)?;
                }
                if use_tabu && me > 0 { for &nd in moved_nodes_buf.iter() { tabu[nd as usize] = round + tabu_tenure; } }

                if me > 0 && use_boundary {
                    stream.memcpy_htod(&moved_nodes_buf[..], &mut d_moved_nodes)?;
                    stream.memset_zeros(&mut d_edge_dirty)?;
                    let moved_cfg = LaunchConfig { grid_dim: (((me as u32) + block_size - 1) / block_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
                    unsafe { stream.launch_builder(&mark_dirty_fn).arg(&me).arg(&d_moved_nodes).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&mut d_edge_dirty).launch(moved_cfg)?; }
                    unsafe { stream.launch_builder(&update_dirty_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&d_edge_dirty).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
                    use_incremental = true;
                }

                if me == 0 && cap_aware && k_cand > k_base && sorted_nodes_buf.len() < k_cand {

                    let fail_mark = std::cmp::min(sorted_nodes_buf.len(), std::cmp::max(n / 8, 64));
                    for &nd in sorted_nodes_buf.iter().take(fail_mark) { tabu[nd as usize] = round + std::cmp::max(3, tabu_tenure / 2); }

                    sorted_nodes_buf.clear(); sorted_targets_buf.clear();
                    let tail_start = k_base;
                    let tail_take = std::cmp::min(k_cand - tail_start, k_base);
                    for &(nd, key) in valid_moves[tail_start..k_cand].iter().take(tail_take) {
                        sorted_nodes_buf.push(nd as i32); sorted_targets_buf.push((key & 63) as i32);
                    }

                    if !sorted_nodes_buf.is_empty() {
                        me = 0;
                        for i in 0..sorted_nodes_buf.len() { let node = sorted_nodes_buf[i] as usize; let target = sorted_targets_buf[i]; let src = exec_part[node]; if src == target { continue; } if exec_nip[target as usize] >= mps_i32 { continue; } if exec_nip[src as usize] <= 1 { continue; } exec_part[node] = target; exec_nip[src as usize] -= 1; exec_nip[target as usize] += 1; me += 1; }
                        if me > 0 { stream.memcpy_htod(&exec_part, &mut d_partition)?; stream.memcpy_htod(&exec_nip, &mut d_nip)?; }
                    }
                }
                if me == 0 {
                    if quick { break; }

                    stagnant += 1; low_move_streak = 0;
                    if stagnant >= tc.perturb_stagnant_thresh && round < ($max_rounds as usize).saturating_sub(50) {
                        let seed_val: u64 = rng.gen();
                        unsafe { stream.launch_builder(&perturb_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32)).arg(&tc.perturb_stagnant_strength).arg(&mut d_partition).arg(&mut d_nip).arg(&seed_val).launch(one_cfg.clone())?; }
                        stream.synchronize()?; stream.memcpy_dtoh(&d_partition, &mut exec_part)?; stream.memcpy_dtoh(&d_nip, &mut exec_nip)?; stagnant = 0; use_incremental = false;
                    } else if stagnant > tc.stagnation_limit { break; }
                } else {

                    stagnant = 0;
                    let move_thresh = if quick { n / 400 } else { n / 1000 };
                    if (me as usize) < move_thresh { low_move_streak += 1; } else { low_move_streak = 0; }
                    let max_streak = if quick { 4 } else { 8 };
                    if low_move_streak >= max_streak { break; }
                }
            }
        }};
    }

    macro_rules! lp_phase {
        ($max_iters:expr) => {{
            let mut d_partition_out = stream.alloc_zeros::<i32>(n)?;
            let mut d_move_count = stream.alloc_zeros::<i32>(1)?;

            let pre_lp_conn = eval_conn!();
            let pre_lp_part: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
            let pre_lp_nip: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
            for _lp_iter in 0..$max_iters {
                unsafe { stream.launch_builder(&precompute_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
                stream.memset_zeros(&mut d_move_count)?;
                unsafe { stream.launch_builder(&lp_step_fn)
                    .arg(&(n as i32)).arg(&(num_parts as i32))
                    .arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets)
                    .arg(&d_partition).arg(&d_flags_all).arg(&d_flags_double)
                    .arg(&mut d_partition_out).arg(&mut d_move_count)
                    .launch(node_cfg.clone())?; }
                stream.synchronize()?;
                let moves = stream.memcpy_dtov(&d_move_count)?[0];
                if moves == 0 { break; }
                stream.memcpy_dtod(&d_partition_out, &mut d_partition)?;

                let part_snap: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
                let mut nip_recount = vec![0i32; num_parts];
                for &p in part_snap.iter() { nip_recount[p as usize] += 1; }
                stream.memcpy_htod(&nip_recount, &mut d_nip)?;

                balance!();
            }

            let post_lp_conn = eval_conn!();
            if post_lp_conn >= pre_lp_conn {
                stream.memcpy_htod(&pre_lp_part, &mut d_partition)?;
                stream.memcpy_htod(&pre_lp_nip, &mut d_nip)?;
            }
        }};
    }

    macro_rules! eval_conn { () => {{ let mut d_metric = stream.alloc_zeros::<u32>(1)?; unsafe { stream.launch_builder(&conn_fn).arg(&(n_hedges as i32)).arg(&challenge.d_hyperedge_offsets).arg(&challenge.d_hyperedge_nodes).arg(&d_partition).arg(&mut d_metric).launch(hedge_cfg.clone())?; } stream.synchronize()?; stream.memcpy_dtov(&d_metric)?[0] as i32 }}; }
    macro_rules! balance { () => {{ unsafe { stream.launch_builder(&balance_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&1i32).arg(&(challenge.max_part_size as i32)).arg(&mut d_partition).arg(&mut d_nip).launch(one_cfg.clone())?; } stream.synchronize()?; }}; }
    macro_rules! gpu_swap_phase {
        ($max_rounds:expr, $do_cycles:expr) => {{
            let sgfn = swap_gains_fn.as_ref().unwrap(); let dsg = d_swap_gains.as_mut().unwrap();
            let mut prev_swap_count = usize::MAX; let mut stagnant_sw = 0usize;
            for _sw_round in 0..$max_rounds {
                unsafe { stream.launch_builder(&precompute_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
                unsafe { stream.launch_builder(sgfn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&neg_gain_thresh).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_partition).arg(&d_flags_all).arg(&d_flags_double).arg(&mut *dsg).launch(node_cfg.clone())?; }
                stream.synchronize()?; stream.memcpy_dtoh(&*dsg, &mut swap_gains_host)?;
                let part_snap: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
                for v in part_to_part.iter_mut() { v.clear(); }

                for node in 0..n { let src = part_snap[node] as usize; if src >= num_parts { continue; } for k in 0..3usize { let val = swap_gains_host[node * 3 + k]; if val == 0x7FFFFFFF_i32 { continue; } let tgt = (val & 0xFFFF) as usize; let gain = ((val >> 16) as i16) as i32; if tgt < num_parts && tgt != src { part_to_part[src * num_parts + tgt].push((node, gain)); } } }
                for list in part_to_part.iter_mut() { if !list.is_empty() { list.sort_unstable_by(|x, y| y.1.cmp(&x.1)); } }
                let mut part_mut = part_snap.clone(); let mut swap_count = 0usize;
                for a in 0..num_parts { for b in (a+1)..num_parts { let idx_ab = a*num_parts+b; let idx_ba = b*num_parts+a; if part_to_part[idx_ab].is_empty() || part_to_part[idx_ba].is_empty() { continue; } let lba_len = part_to_part[idx_ba].len(); let scan_lba = std::cmp::min(lba_len, swap_scan_limit); used_ba_buf.clear(); used_ba_buf.resize(lba_len, false);
                    for i in 0..part_to_part[idx_ab].len() { let (node_a, gain_a) = part_to_part[idx_ab][i]; if part_mut[node_a] as usize != a { continue; } let mut best_combined = 0i32; let mut best_j = usize::MAX; for j in 0..scan_lba { if used_ba_buf[j] { continue; } let (node_b, gain_b) = part_to_part[idx_ba][j]; if part_mut[node_b] as usize != b { continue; } let combined = gain_a + gain_b; if combined > best_combined { best_combined = combined; best_j = j; } } if best_j < lba_len && best_combined > 0 { let (node_b, _) = part_to_part[idx_ba][best_j]; part_mut[node_a] = b as i32; part_mut[node_b] = a as i32; used_ba_buf[best_j] = true; swap_count += 1; } } } }
                if $do_cycles { for a in 0..num_parts { for b in 0..num_parts { if b == a { continue; } let idx_ab = a*num_parts+b; if part_to_part[idx_ab].is_empty() { continue; } let ab_top = std::cmp::min(part_to_part[idx_ab].len(), cycle_scan_limit); for c in 0..num_parts { if c == a || c == b { continue; } let idx_bc = b*num_parts+c; let idx_ca = c*num_parts+a; if part_to_part[idx_bc].is_empty() || part_to_part[idx_ca].is_empty() { continue; } let bc_top = std::cmp::min(part_to_part[idx_bc].len(), cycle_scan_limit); let ca_top = std::cmp::min(part_to_part[idx_ca].len(), cycle_scan_limit);
                    let mut bg = 0i32; let mut bia = usize::MAX; let mut bib = usize::MAX; let mut bic = usize::MAX;
                    for ia in 0..ab_top { let (na, ga) = part_to_part[idx_ab][ia]; if part_mut[na] as usize != a { continue; } if ga + part_to_part[idx_bc][0].1 + part_to_part[idx_ca][0].1 <= bg { break; } for ib in 0..bc_top { let (nb, gb) = part_to_part[idx_bc][ib]; if part_mut[nb] as usize != b { continue; } let ab = ga + gb; if ab + part_to_part[idx_ca][0].1 <= bg { break; } for ic in 0..ca_top { let (nc, gc) = part_to_part[idx_ca][ic]; if part_mut[nc] as usize != c { continue; } let total = ab + gc; if total > bg { bg = total; bia = ia; bib = ib; bic = ic; } break; } } }
                    if bg > 0 && bia < ab_top && bib < bc_top && bic < ca_top { let (na, _) = part_to_part[idx_ab][bia]; let (nb, _) = part_to_part[idx_bc][bib]; let (nc, _) = part_to_part[idx_ca][bic]; if part_mut[na] as usize == a && part_mut[nb] as usize == b && part_mut[nc] as usize == c { part_mut[na] = b as i32; part_mut[nb] = c as i32; part_mut[nc] = a as i32; swap_count += 1; } }
                } } } }
                if swap_count == 0 { break; } if swap_count >= prev_swap_count { stagnant_sw += 1; if stagnant_sw >= 3 { break; } } else { stagnant_sw = 0; } prev_swap_count = swap_count;
                stream.memcpy_htod(&part_mut, &mut d_partition)?; let mut nip_recount = vec![0i32; num_parts]; for &p in part_mut.iter() { nip_recount[p as usize] += 1; } stream.memcpy_htod(&nip_recount, &mut d_nip)?;
            }
        }};
    }

    macro_rules! lce_swap_phase {
        ($max_rounds:expr, $do_4cycles:expr) => {
            if tc.use_gpu_swaps && $max_rounds > 0 {
                let sgfn = swap_gains_fn.as_ref().unwrap();
                let dsg = d_swap_gains.as_mut().unwrap();
                let mut lce_part: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
                let mut lce_nip: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
                used_ba_buf.clear(); used_ba_buf.resize(n, false);

                for _sr in 0..$max_rounds {
                    unsafe { stream.launch_builder(&precompute_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
                    unsafe { stream.launch_builder(sgfn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&neg_gain_thresh).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_partition).arg(&d_flags_all).arg(&d_flags_double).arg(&mut *dsg).launch(node_cfg.clone())?; }
                    stream.synchronize()?;
                    stream.memcpy_dtoh(&*dsg, &mut swap_gains_host)?;

                    for list in part_to_part.iter_mut() { list.clear(); }
                    for node in 0..n {
                        let src = lce_part[node] as usize;
                        if src >= num_parts { continue; }
                        for k in 0..3 {
                            let encoded = swap_gains_host[node * 3 + k];
                            if encoded == 0x7FFFFFFF_i32 { continue; }
                            let gain = (encoded >> 16) as i16 as i32;
                            let tgt = (encoded & 0xFFFF) as usize;

                            if tgt < num_parts && tgt != src { part_to_part[src * num_parts + tgt].push((node, gain)); }
                        }
                    }
                    for list in part_to_part.iter_mut() { list.sort_unstable_by(|a, b| b.1.cmp(&a.1)); }

                    let mut swap_count = 0usize;
                    used_ba_buf.iter_mut().for_each(|x| *x = false);

                    for a in 0..num_parts {
                        for b in (a+1)..num_parts {
                            let idx_ab = a * num_parts + b; let idx_ba = b * num_parts + a;
                            if part_to_part[idx_ab].is_empty() || part_to_part[idx_ba].is_empty() { continue; }
                            let mut p_a = 0usize; let mut p_b = 0usize;
                            let len_ab = part_to_part[idx_ab].len(); let len_ba = part_to_part[idx_ba].len();
                            while p_a < len_ab && p_b < len_ba {
                                let (na, ga) = part_to_part[idx_ab][p_a];
                                if used_ba_buf[na] || lce_part[na] as usize != a { p_a += 1; continue; }
                                let (nb, gb) = part_to_part[idx_ba][p_b];
                                if used_ba_buf[nb] || lce_part[nb] as usize != b { p_b += 1; continue; }
                                if ga + gb > 0 {
                                    lce_part[na] = b as i32; lce_part[nb] = a as i32;
                                    used_ba_buf[na] = true; used_ba_buf[nb] = true;
                                    swap_count += 1; p_a += 1; p_b += 1;
                                } else { break; }
                            }
                        }
                    }

                    for a in 0..num_parts { for b in 0..num_parts { if b == a { continue; }
                        let idx_ab = a * num_parts + b;
                        if part_to_part[idx_ab].is_empty() { continue; }
                        for c in 0..num_parts { if c == a || c == b { continue; }
                            let idx_bc = b * num_parts + c; let idx_ca = c * num_parts + a;
                            if part_to_part[idx_bc].is_empty() || part_to_part[idx_ca].is_empty() { continue; }
                            let mut p_a = 0usize; let mut p_b = 0usize; let mut p_c = 0usize;
                            let len_ab = part_to_part[idx_ab].len(); let len_bc = part_to_part[idx_bc].len(); let len_ca = part_to_part[idx_ca].len();
                            while p_a < len_ab && p_b < len_bc && p_c < len_ca {
                                let (na, ga) = part_to_part[idx_ab][p_a]; if used_ba_buf[na] || lce_part[na] as usize != a { p_a += 1; continue; }
                                let (nb, gb) = part_to_part[idx_bc][p_b]; if used_ba_buf[nb] || lce_part[nb] as usize != b { p_b += 1; continue; }
                                let (nc, gc) = part_to_part[idx_ca][p_c]; if used_ba_buf[nc] || lce_part[nc] as usize != c { p_c += 1; continue; }
                                if ga + gb + gc > 0 {
                                    lce_part[na] = b as i32; lce_part[nb] = c as i32; lce_part[nc] = a as i32;
                                    used_ba_buf[na] = true; used_ba_buf[nb] = true; used_ba_buf[nc] = true;
                                    swap_count += 1; p_a += 1; p_b += 1; p_c += 1;
                                } else { break; }
                            }
                        }
                    }}

                    if $do_4cycles {
                        for a in 0..num_parts { for b in 0..num_parts { if b == a { continue; }
                            let idx_ab = a * num_parts + b; if part_to_part[idx_ab].is_empty() { continue; }
                            for c in 0..num_parts { if c == a || c == b { continue; }
                                let idx_bc = b * num_parts + c; if part_to_part[idx_bc].is_empty() { continue; }
                                for d in 0..num_parts { if d == a || d == b || d == c { continue; }
                                    let idx_cd = c * num_parts + d; let idx_da = d * num_parts + a;
                                    if part_to_part[idx_cd].is_empty() || part_to_part[idx_da].is_empty() { continue; }
                                    let mut p_a = 0usize; let mut p_b = 0usize; let mut p_c = 0usize; let mut p_d = 0usize;
                                    let len_ab = part_to_part[idx_ab].len(); let len_bc = part_to_part[idx_bc].len();
                                    let len_cd = part_to_part[idx_cd].len(); let len_da = part_to_part[idx_da].len();
                                    while p_a < len_ab && p_b < len_bc && p_c < len_cd && p_d < len_da {
                                        let (na, ga) = part_to_part[idx_ab][p_a]; if used_ba_buf[na] || lce_part[na] as usize != a { p_a += 1; continue; }
                                        let (nb, gb) = part_to_part[idx_bc][p_b]; if used_ba_buf[nb] || lce_part[nb] as usize != b { p_b += 1; continue; }
                                        let (nc, gc) = part_to_part[idx_cd][p_c]; if used_ba_buf[nc] || lce_part[nc] as usize != c { p_c += 1; continue; }
                                        let (nd, gd) = part_to_part[idx_da][p_d]; if used_ba_buf[nd] || lce_part[nd] as usize != d { p_d += 1; continue; }
                                        if ga + gb + gc + gd > 0 {
                                            lce_part[na] = b as i32; lce_part[nb] = c as i32; lce_part[nc] = d as i32; lce_part[nd] = a as i32;
                                            used_ba_buf[na] = true; used_ba_buf[nb] = true; used_ba_buf[nc] = true; used_ba_buf[nd] = true;
                                            swap_count += 1; p_a += 1; p_b += 1; p_c += 1; p_d += 1;
                                        } else { break; }
                                    }
                                }
                            }
                        }}
                    }

                    if swap_count == 0 { break; }
                    stream.memcpy_htod(&lce_part, &mut d_partition)?;
                    for nip in lce_nip.iter_mut() { *nip = 0; }
                    for &p in lce_part.iter() { if (p as usize) < num_parts { lce_nip[p as usize] += 1; } }
                    stream.memcpy_htod(&lce_nip, &mut d_nip)?;
                }
            }
        };
    }

    unsafe { stream.launch_builder(&init_weights_fn).arg(&(n_hedges as i32)).arg(&mut d_hedge_weights).launch(hedge_cfg.clone())?; }
    unsafe { stream.launch_builder(&cluster_fn).arg(&(n_hedges as i32)).arg(&64i32).arg(&challenge.d_hyperedge_offsets).arg(&challenge.d_hyperedge_nodes).arg(&mut d_hclusters).launch(hedge_cfg.clone())?; }
    stream.synchronize()?;
    let init_seed: u64 = rng.gen(); let mut init_rng = SmallRng::seed_from_u64(init_seed);
    for restart in 0..tc.num_init_restarts {
        let restart_seed: u32 = init_rng.gen();
        unsafe { stream.launch_builder(&prefs_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&64i32).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets).arg(&d_hclusters).arg(&challenge.d_hyperedge_offsets).arg(&(restart as i32)).arg(&restart_seed).arg(&mut d_pref_parts).arg(&mut d_pref_prio).launch(node_cfg.clone())?; }
        stream.synchronize()?;
        let pp: Vec<i32> = stream.memcpy_dtov(&d_pref_parts)?; let pprio: Vec<i32> = stream.memcpy_dtov(&d_pref_prio)?;
        let mut idx: Vec<usize> = (0..n).collect(); idx.sort_unstable_by(|&a, &b| pprio[b].cmp(&pprio[a]));
        let sn: Vec<i32> = idx.iter().map(|&i| i as i32).collect(); let st: Vec<i32> = idx.iter().map(|&i| pp[i]).collect();
        let d_sn = stream.memcpy_stod(&sn)?; let d_st = stream.memcpy_stod(&st)?;
        stream.memset_zeros(&mut d_nip)?;
        unsafe { stream.launch_builder(&assign_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32)).arg(&d_sn).arg(&d_st).arg(&mut d_partition).arg(&mut d_nip).launch(one_cfg.clone())?; }
        stream.synchronize()?;
        if tc.scout_rounds > 0 { tabu.iter_mut().for_each(|x| *x = 0); refine!(tc.scout_rounds, tc.balance_weight, move_limit, false, false, true, false); }
        let conn = eval_conn!();
        if conn < best_conn { best_conn = conn; stream.memcpy_dtoh(&d_partition, &mut best_partition)?; stream.memcpy_dtoh(&d_nip, &mut best_nip)?; }
    }

    {
        let tree_part = recursive_tree_init(n, n_hedges, &hedge_offsets_data, &hedge_nodes_data, &node_offsets_data, &node_hedges_data, num_parts);
        if !tree_part.is_empty() {
            let mut tree_nip = vec![0i32; num_parts];
            for &p in tree_part.iter() { if (p as usize) < num_parts { tree_nip[p as usize] += 1; } }
            let valid = tree_nip.iter().all(|&c| c >= 1);
            if valid {
                stream.memcpy_htod(&tree_part, &mut d_partition)?;
                stream.memcpy_htod(&tree_nip, &mut d_nip)?;
                let tree_conn = eval_conn!();
                if tree_conn < best_conn {
                    best_conn = tree_conn;
                    best_partition.copy_from_slice(&tree_part);
                    best_nip.copy_from_slice(&tree_nip);
                }
            }
        }
    }
    stream.memcpy_htod(&best_partition, &mut d_partition)?; stream.memcpy_htod(&best_nip, &mut d_nip)?;
    best_conn = i32::MAX;

    refine!(refinement, tc.balance_weight, move_limit, true, true, false, true, 0.0_f64);

    if tc.use_gpu_swaps {
        gpu_swap_phase!(tc.init_swap_rounds1, true);
        tabu.iter_mut().for_each(|x| *x = 0);
        refine!(40, tc.balance_weight, move_limit, false, false, true, false);
        gpu_swap_phase!(tc.init_swap_rounds2, true);
    }

    if tc.vcycle_depth > 0 {
        let mut vc_part: Vec<i32> = stream.memcpy_dtov(&d_partition)?; let mut vc_nip: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
        vcycle_refine(&levels, num_parts, max_ps as i32, &mut vc_part, &mut vc_nip, tc.vcycle_depth);
        stream.memcpy_htod(&vc_part, &mut d_partition)?; stream.memcpy_htod(&vc_nip, &mut d_nip)?;
        tabu.iter_mut().for_each(|x| *x = 0); refine!(std::cmp::min(tc.vcycle_post_refine, refinement), tc.balance_weight, move_limit, true, false, false, false);
        if tc.use_gpu_swaps { gpu_swap_phase!(15, true); }
    }

    best_conn = eval_conn!(); stream.memcpy_dtoh(&d_partition, &mut best_partition)?; stream.memcpy_dtoh(&d_nip, &mut best_nip)?;
    cur_partition.copy_from_slice(&best_partition); cur_nip.copy_from_slice(&best_nip); cur_conn = best_conn;

    let np = std::cmp::min(num_parts, 64);

    let mut d_hedge_conn = stream.alloc_zeros::<i32>(n_hedges)?;
    unsafe { stream.launch_builder(&conn_per_hedge_fn).arg(&(n_hedges as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_hedge_conn).launch(hedge_cfg.clone())?; }
    stream.synchronize()?;
    let conn_vec: Vec<i32> = stream.memcpy_dtov(&d_hedge_conn)?;
    let num_high_hedges = std::cmp::min(2000usize, n_hedges);
    let mut conn_with_idx: Vec<(i32, i32)> = conn_vec.iter().enumerate().map(|(i, &c)| (c, i as i32)).collect();
    conn_with_idx.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let high_hedge_ids: Vec<i32> = conn_with_idx.iter().take(num_high_hedges).map(|&(_, id)| id).collect();
    let mut d_high_hedge_ids = stream.memcpy_stod(&high_hedge_ids)?;

    for ils_iter in 0..ils_iters {

        let pre_ils_part = cur_partition.clone();
        let pre_ils_nip = cur_nip.clone();
        stream.memcpy_htod(&cur_partition, &mut d_partition)?; stream.memcpy_htod(&cur_nip, &mut d_nip)?;
        let seed_val: u64 = rng.gen();

        let mode = ils_iter % 3;
        if mode == 0 {
            let guided_seed = seed_val ^ 0xDEADBEEF_CAFEBABE_u64;
            unsafe { stream.launch_builder(&perturb_guided_fn)
                .arg(&(num_high_hedges as i32)).arg(&d_high_hedge_ids)
                .arg(&challenge.d_hyperedge_offsets).arg(&challenge.d_hyperedge_nodes)
                .arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32))
                .arg(&mut d_partition).arg(&mut d_nip).arg(&guided_seed)
                .launch(one_cfg.clone())?; }
        } else if mode == 1 {
            unsafe { stream.launch_builder(&precompute_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
            let targeted_seed = seed_val ^ 0xBADC0FFEE_u64;
            let perturb_str = if ils_iter < ils_iters / 3 { tc.perturb_strong } else { std::cmp::max(1, tc.perturb_strong - 1) };
            unsafe { stream.launch_builder(&perturb_targeted_fn)
                .arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32))
                .arg(&perturb_str).arg(&challenge.d_node_hyperedges).arg(&challenge.d_node_offsets)
                .arg(&d_flags_all).arg(&mut d_partition).arg(&mut d_nip).arg(&targeted_seed)
                .launch(one_cfg.clone())?; }
        } else {
            let ils_perturb: i32 = if ils_iter < ils_iters / 3 { tc.perturb_strong } else { std::cmp::max(1, tc.perturb_strong - 1) };
            unsafe { stream.launch_builder(&perturb_fn).arg(&(n as i32)).arg(&(num_parts as i32)).arg(&(challenge.max_part_size as i32)).arg(&ils_perturb).arg(&mut d_partition).arg(&mut d_nip).arg(&seed_val).launch(one_cfg.clone())?; }
        }
        stream.synchronize()?;

        tabu.iter_mut().for_each(|x| *x = 0); refine!(ils_refine, tc.balance_weight, move_limit, false, true, false, true);

        if tc.ils_swap_rounds > 0 { gpu_swap_phase!(tc.ils_swap_rounds, false); }

        if tc.vcycle_depth > 0 && tc.vcycle_ils_freq > 0 && ils_iter > 0 && ils_iter % tc.vcycle_ils_freq == 0 {
            let mut vc_part: Vec<i32> = stream.memcpy_dtov(&d_partition)?; let mut vc_nip: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
            vcycle_refine(&levels, num_parts, challenge.max_part_size as i32, &mut vc_part, &mut vc_nip, tc.vcycle_depth);
            stream.memcpy_htod(&vc_part, &mut d_partition)?; stream.memcpy_htod(&vc_nip, &mut d_nip)?;
            tabu.iter_mut().for_each(|x| *x = 0); refine!(ils_refine, tc.balance_weight, move_limit, false, false, false, true);
            if tc.use_gpu_swaps { gpu_swap_phase!(15, true); }
        }
        if tc.ils_cpu_fm > 0 {
            unsafe { stream.launch_builder(&precompute_fn).arg(&(n_hedges as i32)).arg(&(n as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_flags_all).arg(&mut d_flags_double).launch(hedge_cfg.clone())?; }
            stream.synchronize()?;
            let mut ils_fa: Vec<u64> = stream.memcpy_dtov(&d_flags_all)?; let mut ils_fd: Vec<u64> = stream.memcpy_dtov(&d_flags_double)?;
            let mut ils_ph: Vec<i32> = stream.memcpy_dtov(&d_partition)?; let mut ils_nh: Vec<i32> = stream.memcpy_dtov(&d_nip)?;
            let mut ils_epcnt: Vec<u8> = vec![0u8; n_hedges * np];
            for h in 0..n_hedges { let hs = hedge_offsets_data[h] as usize; let he = hedge_offsets_data[h+1] as usize; for k in hs..he { let nd = hedge_nodes_data[k] as usize; if nd < n { let p = ils_ph[nd] as usize; if p < np { let idx = h * np + p; if ils_epcnt[idx] < 255 { ils_epcnt[idx] += 1; } } } } }
            cpu_fm_weighted(n, n_hedges, num_parts, max_ps as i32, &hedge_offsets_data, &hedge_nodes_data, &node_offsets_data, &node_hedges_data, &vec![1i32; n], &mut ils_ph, &mut ils_nh, tc.ils_cpu_fm, tc.cpu_fm_max_degree);
            stream.memcpy_htod(&ils_ph, &mut d_partition)?; for p in ils_nh.iter_mut().take(num_parts) { *p = 0; } for &p in ils_ph.iter() { ils_nh[p as usize] += 1; } stream.memcpy_htod(&ils_nh, &mut d_nip)?;
        }
        let new_conn = eval_conn!();
        if new_conn < best_conn {
            best_conn = new_conn;
            stream.memcpy_dtoh(&d_partition, &mut best_partition)?;
            stream.memcpy_dtoh(&d_nip, &mut best_nip)?;

            unsafe { stream.launch_builder(&conn_per_hedge_fn).arg(&(n_hedges as i32)).arg(&challenge.d_hyperedge_nodes).arg(&challenge.d_hyperedge_offsets).arg(&d_partition).arg(&mut d_hedge_conn).launch(hedge_cfg.clone())?; }
            stream.synchronize()?;
            let new_conn_vec: Vec<i32> = stream.memcpy_dtov(&d_hedge_conn)?;
            let mut new_conn_with_idx: Vec<(i32, i32)> = new_conn_vec.iter().enumerate().map(|(i, &c)| (c, i as i32)).collect();
            new_conn_with_idx.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let new_high_ids: Vec<i32> = new_conn_with_idx.iter().take(num_high_hedges).map(|&(_, id)| id).collect();
            d_high_hedge_ids = stream.memcpy_stod(&new_high_ids)?;
        }

        if new_conn < best_conn {
            cur_conn = best_conn;
            cur_partition.copy_from_slice(&best_partition);
            cur_nip.copy_from_slice(&best_nip);
        } else {

            cur_conn = best_conn;
            cur_partition.copy_from_slice(&pre_ils_part);
            cur_nip.copy_from_slice(&pre_ils_nip);
            stream.memcpy_htod(&cur_partition, &mut d_partition)?;
            stream.memcpy_htod(&cur_nip, &mut d_nip)?;
        }
    }

    stream.memcpy_htod(&best_partition, &mut d_partition)?; stream.memcpy_htod(&best_nip, &mut d_nip)?;
    tabu.iter_mut().for_each(|x| *x = 0); refine!(polish_rounds, tc.balance_weight, move_limit, false, false, false, false);
    balance!();
    tabu.iter_mut().for_each(|x| *x = 0); refine!(post_balance, tc.balance_weight, move_limit / 2, false, false, false, false);

    let mut part_host: Vec<i32> = stream.memcpy_dtov(&d_partition)?; let mut nip_host: Vec<i32> = stream.memcpy_dtov(&d_nip)?;

    gpu_swap_phase!(tc.final_swap_rounds, true);
    stream.memcpy_dtoh(&d_partition, &mut part_host)?;
    stream.memcpy_dtoh(&d_nip, &mut nip_host)?;

    if tc.cpu_fm_rounds > 0 {
        let mut edge_pcnt: Vec<u8> = vec![0u8; n_hedges * np];
        for h in 0..n_hedges { let hs = hedge_offsets_data[h] as usize; let he = hedge_offsets_data[h+1] as usize; for k in hs..he { let nd = hedge_nodes_data[k] as usize; if nd < n { let p = part_host[nd] as usize; if p < np { let idx = h*np+p; if edge_pcnt[idx] < 255 { edge_pcnt[idx] += 1; } } } } }
        cpu_fm_weighted(n, n_hedges, num_parts, max_ps as i32, &hedge_offsets_data, &hedge_nodes_data, &node_offsets_data, &node_hedges_data, &vec![1i32; n], &mut part_host, &mut nip_host, tc.cpu_fm_rounds, tc.cpu_fm_max_degree);
    }

    save_solution(&Solution { partition: part_host.iter().map(|&x| x as u32).collect() })?;
    Ok(())
}
