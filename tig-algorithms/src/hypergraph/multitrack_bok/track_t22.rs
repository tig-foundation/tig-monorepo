use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;


// Gain-IN-SEQUENCE closed-cycle chain refinement helpers.
//
// `chain_seq_gain` returns the exact km1 connectivity REDUCTION of moving `node`
// from partition `from` to `to`, evaluated against the LIVE per-(hyperedge,part)
// pin-count table `epc` (which already reflects every earlier tentative move in
// the chain) — i.e. "gain updated assuming smaller-index neighbours already moved"

//   reduction = #(hedges where `from` had exactly 1 pin) - #(hedges where `to` had 0).
// Algebra (each incident hedge has epc[from]>=1): with pt=#(epc[to]>=1) and
//   cc=#(epc[from]>=2), pt-cc = #(from==1) - #(to==0) = reduction. High-degree
// nodes are sampled (<=256 hedges) to bound cost; the whole pass is wrapped by the
// trusted GPU `measure_total_conn!` guard so any sampling slack cannot regress Q.
#[inline(always)]
fn chain_seq_gain(
    node: usize,
    from: usize,
    to: usize,
    num_parts: usize,
    no: &[i32],
    nh: &[i32],
    epc: &[i32],
) -> i32 {
    let start = no[node] as usize;
    let end = no[node + 1] as usize;
    let deg = end - start;
    if deg == 0 {
        return 0;
    }
    let used = if deg > 256 { 256usize } else { deg };
    let mut cc = 0i32;
    let mut pt = 0i32;
    for j in 0..used {
        let rel = ((j as i64 * deg as i64) / used as i64) as usize;
        let h = nh[start + rel] as usize;
        let base = h * num_parts;
        if epc[base + from] >= 2 {
            cc += 1;
        }
        if to < num_parts && epc[base + to] >= 1 {
            pt += 1;
        }
    }
    pt - cc
}

// Apply (delta=+1) or undo (delta=-1) a single move on the EXACT pin-count table
// (full incidence, never sampled): node leaves `from`, enters `to`.
#[inline(always)]
fn chain_apply(
    node: usize,
    from: usize,
    to: usize,
    num_parts: usize,
    no: &[i32],
    nh: &[i32],
    epc: &mut [i32],
    delta: i32,
) {
    let start = no[node] as usize;
    let end = no[node + 1] as usize;
    for k in start..end {
        let h = nh[k] as usize;
        let base = h * num_parts;
        epc[base + from] -= delta;
        epc[base + to] += delta;
    }
}

// ============================================================================
// Host-side helpers ported from track_t23 for best-of-K + cut-overlay.
// Size-agnostic: operate on hyperedge CSR arrays + a partition. No kernels.
// ============================================================================

// Exact lambda-1 connectivity metric (matches GPU compute_connectivity).
fn mem_eval_lambda(
    part: &[i32],
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    np: usize,
    seen: &mut [bool],
) -> i64 {
    let nh = hedge_offsets.len() - 1;
    let mut c = 0i64;
    for h in 0..nh {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut d = 0i64;
        for k in s..e {
            let p = part[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < np {
                let pu = p as usize;
                if !seen[pu] {
                    seen[pu] = true;
                    d += 1;
                }
            }
        }
        if d > 1 {
            c += d - 1;
        }
        for k in s..e {
            let p = part[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < np {
                seen[p as usize] = false;
            }
        }
    }
    c
}

// ============================================================================
// CUT-OVERLAY ENSEMBLING (K-SpecPart-style)
// Overlay the cuts of all K elite partitions: a hyperedge is "clean" iff it is
// uncut in EVERY elite. Union-Find over clean hyperedges yields supernodes
// (components never separated by any elite cut). Contract, re-solve the small
// hypergraph with exact-km1 supernode FM, project back, and adopt only on a
// strict true-km1 improvement (monotone). All host-side and RNG-free.
// ============================================================================

#[inline]
fn co_uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let g = parent[parent[x as usize] as usize];
        parent[x as usize] = g;
        x = g;
    }
    x
}

#[inline]
fn co_uf_union(parent: &mut [u32], a: u32, b: u32) {
    let ra = co_uf_find(parent, a);
    let rb = co_uf_find(parent, b);
    if ra != rb {
        // Deterministic: larger root id attaches under smaller root id.
        if ra < rb {
            parent[rb as usize] = ra;
        } else {
            parent[ra as usize] = rb;
        }
    }
}

// Full cut-overlay pipeline: contraction, contracted-hypergraph build,
// multi-restart supernode FM, projection. Returns the projected node partition
// (or None if there is nothing to re-solve). The caller re-scores the result
// with mem_eval_lambda and adopts only on strict improvement, so any internal
// miscount could at worst be a no-op, never a regression.
fn co_overlay_resolve(
    elites: &[Vec<i32>],
    elite_km1: &[i64],
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    num_nodes: usize,
    np: usize,
    maxp: i32,
    rounds: usize,
    restarts: usize,
) -> Option<Vec<i32>> {
    if elites.is_empty() || num_nodes == 0 || np == 0 {
        return None;
    }
    let nh = hedge_offsets.len() - 1;

    // ---- 1) Clean-edge Union-Find => supernodes ----------------------------
    let mut parent: Vec<u32> = (0..num_nodes as u32).collect();
    for h in 0..nh {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        if e <= s + 1 {
            continue;
        }
        let mut clean = true;
        'elite_scan: for pt in elites.iter() {
            let p0 = pt[hedge_nodes[s] as usize];
            for k in (s + 1)..e {
                if pt[hedge_nodes[k] as usize] != p0 {
                    clean = false;
                    break 'elite_scan;
                }
            }
        }
        if clean {
            let first = hedge_nodes[s] as u32;
            for k in (s + 1)..e {
                co_uf_union(&mut parent, first, hedge_nodes[k] as u32);
            }
        }
    }

    // Supernode ids in first-seen node order (deterministic labeling).
    let mut node2super: Vec<i32> = vec![-1; num_nodes];
    let mut root2id: Vec<i32> = vec![-1; num_nodes];
    let mut super_weight: Vec<i32> = Vec::new();
    let mut first_member: Vec<usize> = Vec::new();
    for n in 0..num_nodes {
        let r = co_uf_find(&mut parent, n as u32) as usize;
        if root2id[r] < 0 {
            root2id[r] = super_weight.len() as i32;
            super_weight.push(0);
            first_member.push(n);
        }
        let sid = root2id[r];
        node2super[n] = sid;
        super_weight[sid as usize] += 1;
    }
    let ns = super_weight.len();
    if ns <= 1 {
        return None;
    }

    // ---- 2) Contracted hypergraph (dedup pins, drop internal edges) --------
    let mut ce_offsets: Vec<i32> = Vec::with_capacity(nh + 1);
    let mut ce_pins: Vec<i32> = Vec::new();
    ce_offsets.push(0);
    {
        let mut stamp_arr: Vec<u32> = vec![0u32; ns];
        let mut stamp: u32 = 0;
        for h in 0..nh {
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                for v in stamp_arr.iter_mut() {
                    *v = 0;
                }
                stamp = 1;
            }
            let base = ce_pins.len();
            for k in s..e {
                let sid = node2super[hedge_nodes[k] as usize];
                if sid >= 0 && stamp_arr[sid as usize] != stamp {
                    stamp_arr[sid as usize] = stamp;
                    ce_pins.push(sid);
                }
            }
            if ce_pins.len() - base <= 1 {
                // Internal (or empty) after contraction: km1 contribution is 0
                // for ANY supernode-respecting partition; drop it.
                ce_pins.truncate(base);
            } else {
                ce_offsets.push(ce_pins.len() as i32);
            }
        }
    }
    let nce = ce_offsets.len() - 1;
    if nce == 0 {
        return None;
    }

    // Supernode -> incident contracted edges CSR (each edge once per pin;
    // pins are deduped, so each incident edge appears exactly once).
    let mut sn_offsets: Vec<i32> = vec![0; ns + 1];
    for &sid in ce_pins.iter() {
        sn_offsets[sid as usize + 1] += 1;
    }
    for i in 0..ns {
        sn_offsets[i + 1] += sn_offsets[i];
    }
    let mut cursor: Vec<i32> = sn_offsets[..ns].to_vec();
    let mut sn_hedges: Vec<i32> = vec![0; ce_pins.len()];
    for hc in 0..nce {
        let s = ce_offsets[hc] as usize;
        let e = ce_offsets[hc + 1] as usize;
        for k in s..e {
            let sid = ce_pins[k] as usize;
            sn_hedges[cursor[sid] as usize] = hc as i32;
            cursor[sid] += 1;
        }
    }

    // ---- 3) Restart inits: up to `restarts` distinct elites, best-km1 first.
    // Supernode part is well-defined per elite: members are never separated by
    // any elite cut, so they share one part in every elite.
    let mut order: Vec<usize> = (0..elites.len()).collect();
    order.sort_by(|&a, &b| elite_km1[a].cmp(&elite_km1[b]).then(a.cmp(&b)));
    let mut inits: Vec<Vec<i32>> = Vec::new();
    for &ei in order.iter() {
        if inits.len() >= restarts {
            break;
        }
        let mut sp: Vec<i32> = vec![0; ns];
        let mut ok = true;
        for sidx in 0..ns {
            let p = elites[ei][first_member[sidx]];
            if p < 0 || (p as usize) >= np {
                ok = false;
                break;
            }
            sp[sidx] = p;
        }
        if !ok || inits.iter().any(|x| x == &sp) {
            continue;
        }
        inits.push(sp);
    }
    if inits.is_empty() {
        return None;
    }

    // ---- 4) Exact-km1 supernode FM on the contracted hypergraph -----------
    let mut best_sp: Vec<i32> = Vec::new();
    let mut best_ckm1 = i64::MAX;
    let mut counts: Vec<i32> = vec![0; nce * np];
    let mut present: Vec<i32> = vec![0; np];
    for sp in inits.iter() {
        // counts[hc*np + p] = # contracted pins of hc in part p.
        for v in counts.iter_mut() {
            *v = 0;
        }
        let mut pw: Vec<i32> = vec![0; np];
        for sidx in 0..ns {
            pw[sp[sidx] as usize] += super_weight[sidx];
        }
        let mut cur: Vec<i32> = sp.clone();
        let mut ckm1: i64 = 0;
        for hc in 0..nce {
            let base = hc * np;
            let s = ce_offsets[hc] as usize;
            let e = ce_offsets[hc + 1] as usize;
            for k in s..e {
                counts[base + cur[ce_pins[k] as usize] as usize] += 1;
            }
            let mut lam = 0i64;
            for p in 0..np {
                if counts[base + p] > 0 {
                    lam += 1;
                }
            }
            ckm1 += lam - 1;
        }

        for _pass in 0..rounds {
            let mut improved = false;
            for sidx in 0..ns {
                let a = cur[sidx] as usize;
                let w = super_weight[sidx];
                if pw[a] - w < 1 {
                    continue; // keep every part non-empty (node-weight >= 1)
                }
                let ds = sn_offsets[sidx] as usize;
                let de = sn_offsets[sidx + 1] as usize;
                let deg = (de - ds) as i32;
                if deg == 0 {
                    continue;
                }
                // present[p] = # incident edges already touching part p;
                // crit_a = # incident edges where this supernode is part a's
                // only pin (moving it removes a from those edges' lambda).
                for v in present.iter_mut() {
                    *v = 0;
                }
                let mut crit_a = 0i32;
                for &hc32 in sn_hedges[ds..de].iter() {
                    let base = hc32 as usize * np;
                    for p in 0..np {
                        if counts[base + p] > 0 {
                            present[p] += 1;
                        }
                    }
                    if counts[base + a] == 1 {
                        crit_a += 1;
                    }
                }
                // delta(move a -> p) = (deg - present[p]) - crit_a : the
                // exact km1 change of moving ALL member nodes together.
                let mut best_p: i32 = -1;
                let mut best_delta = 0i64; // strictly-improving only
                let mut best_pw = i32::MAX;
                for p in 0..np {
                    if p == a || pw[p] + w > maxp {
                        continue;
                    }
                    let delta = (deg - present[p]) as i64 - crit_a as i64;
                    if delta < best_delta
                        || (delta == best_delta
                            && best_p >= 0
                            && (pw[p] < best_pw || (pw[p] == best_pw && (p as i32) < best_p)))
                    {
                        best_delta = delta;
                        best_p = p as i32;
                        best_pw = pw[p];
                    }
                }
                if best_p >= 0 && best_delta < 0 {
                    let bp = best_p as usize;
                    for &hc32 in sn_hedges[ds..de].iter() {
                        let base = hc32 as usize * np;
                        counts[base + a] -= 1;
                        counts[base + bp] += 1;
                    }
                    pw[a] -= w;
                    pw[bp] += w;
                    cur[sidx] = best_p;
                    ckm1 += best_delta;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }

        if ckm1 < best_ckm1 {
            best_ckm1 = ckm1;
            best_sp = cur;
        }
    }
    if best_sp.is_empty() {
        return None;
    }

    // ---- 5) Project back: node part = its supernode's final part ----------
    let mut out: Vec<i32> = vec![0; num_nodes];
    for n in 0..num_nodes {
        out[n] = best_sp[node2super[n] as usize];
    }
    Some(out)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let block_size = std::cmp::min(128, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_100k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_100k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_100k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_100k")?;

    let lb_min = hyperparameters
        .as_ref()
        .and_then(|p| p.get("launch_bounds_min_blocks").and_then(|v| v.as_i64()))
        .unwrap_or(0);
    let compute_moves_name = match lb_min {
        2 => "compute_refinement_moves_optimized_100k_lb2",
        3 => "compute_refinement_moves_optimized_100k_lb3",
        4 => "compute_refinement_moves_optimized_100k_lb4",
        _ => "compute_refinement_moves_optimized_100k",
    };
    let compute_moves_kernel = module.load_function(compute_moves_name)?;
    let balance_kernel = module.load_function("balance_final_100k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_100k")?;
    let reduce_connectivity_sum_kernel = module.load_function("reduce_connectivity_sum_100k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_100k")?;
    let compute_bsp_kernel = module.load_function("compute_balanced_swap_scores_100k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_100k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_100k")?;

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
        1 => (1000, 3, 25, 40, 32),
        0 => (500, 3, 20, 30, 32),
        _ => (2000, 5, 50, 150, 64),
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
        .unwrap_or(12);

    let tabu_fail_tenure = 4usize;
    let tabu_mark_base = 4096usize;
    let tabu_mark_mult = 16usize;
    let tabu_fail_mark_len = 4096usize;

    let extra_window = 61440usize;
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

    
    // coordinated positive-gain sweeps. 0 => phase disabled (byte-identical to i5).
    let seq_sweeps: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("seq_sweeps").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 64) as usize)
        .unwrap_or(6);

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

    // ==== best-of-K wrap: re-run the whole solve core `runs` times with
    // per-run seed offsets; keep the globally best partition by true km1. ====
    let bok_runs = hyperparameters
        .as_ref()
        .and_then(|p| p.get("runs").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 21) as usize) // K_max=14: measured fuel-safe max
        .unwrap_or(1); // default 1 => byte-identical to current t22
    // ==== cut-overlay ensembling hyperparameters (overlay=0 => exactly bok) ====
    let overlay_enabled = hyperparameters
        .as_ref()
        .and_then(|p| p.get("overlay").and_then(|v| v.as_i64()))
        .unwrap_or(0) // overlay OFF by default => byte-identical to current t22
        != 0;
    let overlay_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("overlay_rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 200) as usize)
        .unwrap_or(20);
    let overlay_restarts = hyperparameters
        .as_ref()
        .and_then(|p| p.get("overlay_restarts").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 16) as usize)
        .unwrap_or(4);
    // overlay_select: keep only this many maximally cut-diverse elites for the
    // overlay (0 = use all; farthest-point selection anchored on best-km1).
    let overlay_select = hyperparameters
        .as_ref()
        .and_then(|p| p.get("overlay_select").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 64) as usize)
        .unwrap_or(0);
    // Elite pool collected across the best-of-K runs (only when overlay is on).
    let mut co_elites: Vec<Vec<i32>> = Vec::new();
    let mut co_elite_km1: Vec<i64> = Vec::new();
    let bok_base_part = partition_host_refine.clone();
    let bok_base_nip = nodes_in_part_host.clone();
    let mut bok_best_part: Vec<u32> = Vec::new();
    let mut bok_best_km1: i64 = i64::MAX;
    for bok_run in 0..bok_runs {
        // Independent restart from the deterministic construction seed.
        partition_host_refine.copy_from_slice(&bok_base_part);
        nodes_in_part_host.copy_from_slice(&bok_base_nip);
        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        // run 0 => offset 0 => byte-identical to champion (monotone at K=1).
        let run_off: u64 = (bok_run as u64).wrapping_mul(0x9E3779B97F4A7C15);

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
        let aspiration_threshold = (max_gain * 3) / 4;
        
        let mut rng_state = 123456789u64.wrapping_add(round as u64).wrapping_add(run_off);

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
                let mini_seed = (987654321u64 + (round as u64) * 123456789u64).wrapping_add(run_off);
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

        let seed = (123456789u64 + (ils_iter as u64) * 987654321u64).wrapping_add(run_off);

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
            let mut rng_state = 987654321u64.wrapping_add(run_off);
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
        let mut rng_state = 11223344u64.wrapping_add(run_off);
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
        let mut rng_state = 55667788u64.wrapping_add(run_off);
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

    
    // Terminal coordinated positive-gain sweep with EXACT incremental km1 gain recomputed
    // against the live partition (= "gains updated assuming smaller-index neighbours already
    // applied"). The GPU compute_moves kernel only proposes candidate targets (approx gain);
    // each move is committed ONLY if its exact km1 gain (full incident-hyperedge scan against
    // the running pin-count table) is strictly positive => each commit strictly lowers total
    // km1 connectivity => the sweep is monotone by construction. As a backstop the whole phase
    // is gated on the TRUSTED GPU-recomputed connectivity (same metric used to track best
    // throughout): the refined partition is kept only if it strictly improves, else the i5
    // floor partition is restored. => cannot regress below the i5 floor (Q=267728).
    macro_rules! measure_total_conn {
        ($dpart:expr) => {{
            unsafe {
                stream
                    .launch_builder(&compute_connectivity_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg($dpart)
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
            let bs = stream.memcpy_dtov(&d_total_connectivity)?;
            let nb = ((challenge.num_hyperedges as u32 + block_size * 2 - 1) / (block_size * 2))
                as usize;
            let s: i64 = bs[..nb].iter().map(|&x| x as i64).sum();
            s
        }};
    }

    if seq_sweeps > 0 {
        let p = num_parts_usize;
        let n_hedges = challenge.num_hyperedges as usize;
        let maxps = challenge.max_part_size as i32;

        // node -> incident hyperedges (CSR), pulled to host once
        let node_off = stream.memcpy_dtov(&challenge.d_node_offsets)?;
        let node_hes = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;

        // trusted floor partition (post-swap i5 result) + its GPU-measured connectivity
        let saved_part = stream.memcpy_dtov(&d_partition)?;
        let pre_conn = measure_total_conn!(&d_partition);

        let mut seq_part = saved_part.clone();
        // per-(hyperedge, part) pin counts, maintained incrementally across sweeps
        let mut pc = vec![0i32; n_hedges * p];
        for h in 0..n_hedges {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            for k in hs..he {
                let node = hyperedge_nodes_host[k] as usize;
                let part = seq_part[node];
                if part >= 0 && (part as usize) < p {
                    pc[h * p + part as usize] += 1;
                }
            }
        }
        let mut nip = vec![0i32; p];
        for &pt in seq_part.iter() {
            if pt >= 0 && (pt as usize) < p {
                nip[pt as usize] += 1;
            }
        }

        for _sweep in 0..seq_sweeps {
            // recompute per-node best target on the CURRENT partition (fresh snapshot)
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

            // candidate sequence: positive-gain moves, larger gains first (node asc tie-break)
            let mut cands: Vec<(i32, usize, usize)> = Vec::new();
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key as u32 != 0x80000000 {
                    let gain = key >> 16;
                    if gain > 0 {
                        cands.push((gain, node, (key & 63) as usize));
                    }
                }
            }
            if cands.is_empty() {
                break;
            }
            cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

            let mut applied = 0usize;
            for &(_g, node, b) in cands.iter() {
                let a = seq_part[node];
                if a < 0 {
                    continue;
                }
                let a = a as usize;
                if a == b || b >= p {
                    continue;
                }
                if nip[a] <= 1 || nip[b] >= maxps {
                    continue;
                }
                let s = node_off[node] as usize;
                let e = node_off[node + 1] as usize;
                // exact km1 gain against the LIVE partition (sees earlier-applied moves)
                let mut a_loses = 0i32;
                let mut b_gains = 0i32;
                for k in s..e {
                    let h = node_hes[k] as usize;
                    if pc[h * p + a] == 1 {
                        a_loses += 1;
                    }
                    if pc[h * p + b] == 0 {
                        b_gains += 1;
                    }
                }
                if a_loses - b_gains > 0 {
                    for k in s..e {
                        let h = node_hes[k] as usize;
                        pc[h * p + a] -= 1;
                        pc[h * p + b] += 1;
                    }
                    seq_part[node] = b as i32;
                    nip[a] -= 1;
                    nip[b] += 1;
                    applied += 1;
                }
            }
            if applied == 0 {
                break;
            }
        }

        // monotone guard on the TRUSTED GPU connectivity
        stream.memcpy_htod(&seq_part, &mut d_partition)?;
        let post_conn = measure_total_conn!(&d_partition);
        if post_conn >= pre_conn {
            stream.memcpy_htod(&saved_part, &mut d_partition)?;
        }
    }

    // ===== L4: gain-in-sequence CLOSED-CYCLE chain refinement (port t21 i16) =====
    // The seq sweep above only commits single positive-gain moves; the chain is the
    // next cran up: it builds variable-length balance-neutral cycles s0->p1->..->pk->s0
    // and commits the best one whose total IN-SEQUENCE km1 reduction is strictly > 0
    // (each move's gain recomputed against the live pin-count table `epc`, which already
    // reflects prior tentative moves). This is the exact lever that broke t21's FM
    // zero-gain single-vertex wall (i14 +67). Closed cycles are balance-neutral by
    // construction (each partition in the cycle loses & gains exactly one node), so no
    // max_part_size check is needed. The whole pass is wrapped in the SAME trusted GPU
    // `measure_total_conn!` monotone guard as the seq block: kept only if it strictly
    // improves, else the pre-chain partition is restored => cannot regress below i8.
    //
    // chain_sweeps == 0 (default) => this block is skipped => byte-identical to i8.
    let chain_sweeps: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("chain_sweeps").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 32) as usize)
        .unwrap_or(0);
    let chain_max_len: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("chain_max_len").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(4, 12) as usize)
        .unwrap_or(4);
    let chain_scan: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("chain_scan").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 256) as usize)
        .unwrap_or(32);

    if chain_sweeps > 0 {
        let p = num_parts_usize;
        let n = challenge.num_nodes as usize;
        let n_hedges = challenge.num_hyperedges as usize;
        let num_nodes_i = challenge.num_nodes as i32;
        let num_parts_i = challenge.num_parts as i32;

        // node -> incident hyperedges (CSR), pulled to host once
        let node_off = stream.memcpy_dtov(&challenge.d_node_offsets)?;
        let node_hes = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;

        // trusted floor (post-seq i8 result) + its GPU-measured connectivity
        let saved_part = stream.memcpy_dtov(&d_partition)?;
        let pre_conn = measure_total_conn!(&d_partition);

        let mut cpart = saved_part.clone();

        // exact per-(hyperedge, part) pin counts, maintained incrementally
        let mut epc = vec![0i32; n_hedges * p];
        for h in 0..n_hedges {
            let hs = hedge_offsets_host[h] as usize;
            let he = hedge_offsets_host[h + 1] as usize;
            for k in hs..he {
                let node = hyperedge_nodes_host[k] as usize;
                let pt = cpart[node];
                if pt >= 0 && (pt as usize) < p {
                    epc[h * p + pt as usize] += 1;
                }
            }
        }

        // scratch reused across s0
        let mut chain_used: Vec<bool> = vec![false; n];
        let mut in_parts: Vec<bool> = vec![false; p];
        let mut tent: Vec<usize> = Vec::with_capacity(chain_max_len + 1);
        let mut parts_seq: Vec<usize> = Vec::with_capacity(chain_max_len + 1);
        let mut edge_nodes: Vec<usize> = Vec::with_capacity(chain_max_len + 1);
        let mut applied: Vec<(usize, usize, usize)> = Vec::with_capacity(chain_max_len + 1);

        for _sweep in 0..chain_sweeps {
            // (1) recompute per-node top-3 swap candidates on the CURRENT partition
            stream.memcpy_htod(&cpart, &mut d_partition)?;
            unsafe {
                stream
                    .launch_builder(&precompute_edge_flags_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&num_nodes_i)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&mut d_partition)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .launch(hedge_cfg.clone())?;
            }
            unsafe {
                stream
                    .launch_builder(&compute_swap_gains_kernel)
                    .arg(&num_nodes_i)
                    .arg(&num_parts_i)
                    .arg(&neg_gain_thresh)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&mut d_partition)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .arg(&mut d_swap_gains)
                    .launch(cfg.clone())?;
            }
            stream.memcpy_dtoh(&d_swap_gains, &mut swap_gains_host)?;

            for v in part_to_part.iter_mut() {
                v.clear();
            }
            for node in 0..n {
                let src = cpart[node];
                if src < 0 || (src as usize) >= p {
                    continue;
                }
                let src = src as usize;
                for kk in 0..3usize {
                    let val = swap_gains_host[node * 3 + kk];
                    if val == 0 {
                        continue;
                    }
                    let tgt = (val & 0xFFFF) as usize;
                    let gain = ((val >> 16) as i16) as i32;
                    if tgt < p && tgt != src {
                        part_to_part[src * p + tgt].push((node, gain));
                    }
                }
            }
            for lst in part_to_part.iter_mut() {
                lst.sort_unstable_by(|x, y| y.1.cmp(&x.1));
            }

            // (2) build & commit best strictly-positive closed cycle per starting part
            for b in chain_used.iter_mut() {
                *b = false;
            }
            let mut committed = 0usize;
            for s0 in 0..p {
                for b in in_parts.iter_mut() {
                    *b = false;
                }
                tent.clear();
                parts_seq.clear();
                edge_nodes.clear();
                applied.clear();
                parts_seq.push(s0);
                in_parts[s0] = true;
                let mut cur = s0;
                let mut real_prefix: i32 = 0;
                let mut best_total: i32 = 0;
                let mut best_pcount: usize = 0;
                let mut best_close: usize = usize::MAX;

                while parts_seq.len() < chain_max_len {
                    // pick best-isolation-gain available successor (heuristic only)
                    let mut best_t = usize::MAX;
                    let mut best_node = usize::MAX;
                    let mut best_gain = i32::MIN;
                    for t in 0..p {
                        if in_parts[t] {
                            continue;
                        }
                        let lst = &part_to_part[cur * p + t];
                        let scan = std::cmp::min(lst.len(), chain_scan);
                        for j in 0..scan {
                            let (node, g) = lst[j];
                            if chain_used[node] || cpart[node] as usize != cur {
                                continue;
                            }
                            if tent.iter().any(|&x| x == node) {
                                continue;
                            }
                            if g > best_gain {
                                best_gain = g;
                                best_t = t;
                                best_node = node;
                            }
                            break; // list sorted desc => first available is best for t
                        }
                    }
                    if best_t == usize::MAX {
                        break;
                    }

                    // exact IN-SEQUENCE reduction against the live epc
                    let rg = chain_seq_gain(best_node, cur, best_t, p, &node_off, &node_hes, &epc);
                    real_prefix += rg;
                    edge_nodes.push(best_node);
                    tent.push(best_node);
                    chain_apply(best_node, cur, best_t, p, &node_off, &node_hes, &mut epc, 1);
                    applied.push((best_node, cur, best_t));
                    parts_seq.push(best_t);
                    in_parts[best_t] = true;
                    cur = best_t;

                    // evaluate closing the cycle once path has >= 4 partitions
                    if parts_seq.len() >= 4 {
                        let lst = &part_to_part[cur * p + s0];
                        let scan = std::cmp::min(lst.len(), chain_scan);
                        for j in 0..scan {
                            let (node, _g) = lst[j];
                            if chain_used[node] || cpart[node] as usize != cur {
                                continue;
                            }
                            if tent.iter().any(|&x| x == node) {
                                continue;
                            }
                            let cg =
                                chain_seq_gain(node, cur, s0, p, &node_off, &node_hes, &epc);
                            let total = real_prefix + cg;
                            if total > best_total {
                                best_total = total;
                                best_pcount = edge_nodes.len();
                                best_close = node;
                            }
                            break;
                        }
                    }
                }

                // undo every tentative move to restore epc for the next s0
                for &(nd, fr, to) in applied.iter().rev() {
                    chain_apply(nd, fr, to, p, &node_off, &node_hes, &mut epc, -1);
                }

                // commit the best strictly-positive (balance-neutral) closed cycle
                if best_total > 0 && best_close != usize::MAX {
                    for i in 0..best_pcount {
                        let nd = edge_nodes[i];
                        let fr = parts_seq[i];
                        let to = parts_seq[i + 1];
                        cpart[nd] = to as i32;
                        chain_used[nd] = true;
                        chain_apply(nd, fr, to, p, &node_off, &node_hes, &mut epc, 1);
                    }
                    let fr = parts_seq[best_pcount];
                    cpart[best_close] = s0 as i32;
                    chain_used[best_close] = true;
                    chain_apply(best_close, fr, s0, p, &node_off, &node_hes, &mut epc, 1);
                    committed += 1;
                }
            }

            if committed == 0 {
                break;
            }
        }

        // monotone guard on the TRUSTED GPU connectivity (cannot regress below i8)
        stream.memcpy_htod(&cpart, &mut d_partition)?;
        let post_conn = measure_total_conn!(&d_partition);
        if post_conn >= pre_conn {
            stream.memcpy_htod(&saved_part, &mut d_partition)?;
        }
    }

    // ===== L5: flow_based_terminal_refinement (KaHyPar-MF max-flow/min-cut post-chain) =====
    // Pivot après saturation veine gain-séquence (i22 +2 NOISE). Lawler/Liu-Wong gadget
    
    // + garde locale λ−1 + garde globale + garde GPU measure_total_conn!.
    // flow_pairs=0 (défaut) => bit-identical à i22 KEPT (267929). Port de v8/t23/i126 validé.
    let flow_pairs: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("flow_pairs").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 8) as usize)
        .unwrap_or(0);

    if flow_pairs > 0 {
        let n_fl = challenge.num_nodes as usize;
        let m_fl = challenge.num_hyperedges as usize;
        let np = num_parts_usize;
        let max_ps = challenge.max_part_size as i32;
        let fl_node_cap: usize = hyperparameters
            .as_ref()
            .and_then(|p| p.get("fl_node_cap").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(32, 1024) as usize)
            .unwrap_or(256);

        let mut fl_part: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
        let fl_pre_conn = measure_total_conn!(&d_partition);
        let fl_saved = fl_part.clone();

        // node->hedge CSR built from host hedge CSR
        let mut nd_deg_fl = vec![0usize; n_fl];
        for h in 0..m_fl {
            for k in (hedge_offsets_host[h] as usize)..(hedge_offsets_host[h + 1] as usize) {
                let nd = hyperedge_nodes_host[k] as usize;
                if nd < n_fl { nd_deg_fl[nd] += 1; }
            }
        }
        let mut nd_off_fl = vec![0usize; n_fl + 1];
        for i in 0..n_fl { nd_off_fl[i + 1] = nd_off_fl[i] + nd_deg_fl[i]; }
        let mut nd_hedges_fl = vec![0usize; nd_off_fl[n_fl]];
        {
            let mut cur = nd_off_fl[..n_fl].to_vec();
            for h in 0..m_fl {
                for k in (hedge_offsets_host[h] as usize)..(hedge_offsets_host[h + 1] as usize) {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_fl { nd_hedges_fl[cur[nd]] = h; cur[nd] += 1; }
                }
            }
        }

        // hpc_fl[h*np + p] = #pins of hedge h in block p
        let mut hpc_fl = vec![0u16; m_fl * np];
        for h in 0..m_fl {
            for k in (hedge_offsets_host[h] as usize)..(hedge_offsets_host[h + 1] as usize) {
                let nd = hyperedge_nodes_host[k] as usize;
                let p = fl_part[nd];
                if p >= 0 && (p as usize) < np {
                    hpc_fl[h * np + p as usize] = hpc_fl[h * np + p as usize].saturating_add(1);
                }
            }
        }

        let mut fl_size = vec![0i32; np];
        for &p in &fl_part {
            if p >= 0 && (p as usize) < np { fl_size[p as usize] += 1; }
        }

        let conn_full_fl = |hpc: &[u16]| -> i64 {
            let mut c = 0i64;
            for h in 0..m_fl {
                let mut lam = 0i64;
                for p in 0..np { if hpc[h * np + p] > 0 { lam += 1; } }
                if lam > 1 { c += lam - 1; }
            }
            c
        };
        let initial_conn_fl = conn_full_fl(&hpc_fl);

        // active pair scheduling: shared-cut weight per pair
        let mut pair_w = vec![0u32; np * np];
        {
            let mut present: Vec<usize> = Vec::with_capacity(np);
            for h in 0..m_fl {
                present.clear();
                for p in 0..np { if hpc_fl[h * np + p] > 0 { present.push(p); } }
                if present.len() < 2 { continue; }
                for ai in 0..present.len() {
                    for bi in (ai + 1)..present.len() {
                        let a2 = present[ai]; let b2 = present[bi];
                        pair_w[a2 * np + b2] += 1;
                    }
                }
            }
        }
        let mut pairs_fl: Vec<(u32, usize, usize)> = Vec::new();
        for a in 0..np {
            for b in (a + 1)..np {
                let w = pair_w[a * np + b];
                if w > 0 { pairs_fl.push((w, a, b)); }
            }
        }
        pairs_fl.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));

        // Lawler/Liu-Wong helpers (paired fwd/bwd edges: idx^1 = reverse)
        #[inline(always)]
        fn fl_add_edge(g: &mut Vec<Vec<usize>>, e_to: &mut Vec<usize>, e_cap: &mut Vec<i64>,
                       e_from: &mut Vec<usize>, u: usize, v: usize, c: i64) {
            let i = e_to.len();
            g[u].push(i);     e_to.push(v); e_cap.push(c); e_from.push(u);
            g[v].push(i + 1); e_to.push(u); e_cap.push(0); e_from.push(v);
        }
        fn fl_bfs(s: usize, t: usize, g: &[Vec<usize>], e_to: &[usize],
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
        fn fl_dfs(u: usize, t: usize, f: i64, g: &[Vec<usize>], e_to: &[usize],
                  e_cap: &mut [i64], level: &[i32], it: &mut [usize]) -> i64 {
            if u == t { return f; }
            while it[u] < g[u].len() {
                let idx = g[u][it[u]];
                let v = e_to[idx];
                if e_cap[idx] > 0 && level[v] == level[u] + 1 {
                    let d = fl_dfs(v, t, f.min(e_cap[idx]), g, e_to, e_cap, level, it);
                    if d > 0 { e_cap[idx] -= d; e_cap[idx ^ 1] += d; return d; }
                }
                it[u] += 1;
            }
            0
        }
        fn fl_move_node(v: usize, nb: usize, fl_part: &mut [i32], hpc: &mut [u16],
                        fl_size: &mut [i32], nd_off: &[usize], nd_hedges: &[usize], np: usize) {
            let cur = fl_part[v] as usize;
            if cur == nb { return; }
            for hi in nd_off[v]..nd_off[v + 1] {
                let h = nd_hedges[hi];
                if hpc[h * np + cur] > 0 { hpc[h * np + cur] -= 1; }
                hpc[h * np + nb] = hpc[h * np + nb].saturating_add(1);
            }
            fl_size[cur] -= 1; fl_size[nb] += 1; fl_part[v] = nb as i32;
        }

        const FL_INF: i64 = 1i64 << 40;
        const FL_HEDGE_CAP: usize = 60_000;

        let mut in_region = vec![false; n_fl];
        let mut rid = vec![-1i32; n_fl];
        let mut h_mark = vec![0u32; m_fl];
        let mut cur_tag = 0u32;
        let mut reg_pins: Vec<usize> = Vec::new();

        let n_pairs_fl = pairs_fl.len().min(flow_pairs);
        for pi in 0..n_pairs_fl {
            let (_w, a, b) = pairs_fl[pi];

            // BFS corridor: boundary first, expand within a∪b up to fl_node_cap
            let mut region: Vec<usize> = Vec::new();
            for v in 0..n_fl {
                let pv = fl_part[v];
                let other = if pv == a as i32 { b } else if pv == b as i32 { a } else { continue };
                let touches = (nd_off_fl[v]..nd_off_fl[v + 1])
                    .any(|hi| hpc_fl[nd_hedges_fl[hi] * np + other] > 0);
                if touches { in_region[v] = true; region.push(v); }
            }
            if region.len() < 2 {
                for &v in &region { in_region[v] = false; }
                continue;
            }
            let mut qi = 0usize;
            while qi < region.len() && region.len() < fl_node_cap {
                let v = region[qi]; qi += 1;
                let mut full = false;
                for hi in nd_off_fl[v]..nd_off_fl[v + 1] {
                    let h = nd_hedges_fl[hi];
                    for k in (hedge_offsets_host[h] as usize)..(hedge_offsets_host[h + 1] as usize) {
                        let u = hyperedge_nodes_host[k] as usize;
                        if u < n_fl && !in_region[u] {
                            let pu = fl_part[u];
                            if pu == a as i32 || pu == b as i32 {
                                in_region[u] = true; region.push(u);
                                if region.len() >= fl_node_cap { full = true; break; }
                            }
                        }
                    }
                    if full { break; }
                }
            }

            // incident hedges — dedup via stamp
            cur_tag += 1;
            let mut h_list: Vec<usize> = Vec::new();
            for &v in &region {
                for hi in nd_off_fl[v]..nd_off_fl[v + 1] {
                    let h = nd_hedges_fl[hi];
                    if h_mark[h] != cur_tag { h_mark[h] = cur_tag; h_list.push(h); }
                }
            }
            if h_list.len() > FL_HEDGE_CAP {
                for &v in &region { in_region[v] = false; }
                continue;
            }

            // Lawler flow network: 0=source(a), 1=sink(b), region nodes 2..2+nb
            let nb_fl = region.len();
            for (i, &v) in region.iter().enumerate() { rid[v] = (2 + i) as i32; }
            let mut g_fl: Vec<Vec<usize>> = (0..2 + nb_fl).map(|_| Vec::new()).collect();
            let mut e_to:  Vec<usize> = Vec::new();
            let mut e_cap: Vec<i64>   = Vec::new();
            let mut e_from: Vec<usize> = Vec::new();

            for &h in &h_list {
                let mut has_fa = false; let mut has_fb = false;
                reg_pins.clear();
                for k in (hedge_offsets_host[h] as usize)..(hedge_offsets_host[h + 1] as usize) {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd >= n_fl { continue; }
                    let r = rid[nd];
                    if r >= 0 {
                        reg_pins.push(r as usize);
                    } else {
                        let pb = fl_part[nd];
                        if pb == a as i32 { has_fa = true; }
                        else if pb == b as i32 { has_fb = true; }
                    }
                }
                if reg_pins.is_empty() { continue; }
                let e_in  = g_fl.len(); g_fl.push(Vec::new());
                let e_out = g_fl.len(); g_fl.push(Vec::new());
                fl_add_edge(&mut g_fl, &mut e_to, &mut e_cap, &mut e_from, e_in, e_out, 1);
                if has_fa { fl_add_edge(&mut g_fl, &mut e_to, &mut e_cap, &mut e_from, 0, e_in, FL_INF); }
                if has_fb { fl_add_edge(&mut g_fl, &mut e_to, &mut e_cap, &mut e_from, e_out, 1, FL_INF); }
                for &rp in &reg_pins {
                    fl_add_edge(&mut g_fl, &mut e_to, &mut e_cap, &mut e_from, rp, e_in, FL_INF);
                    fl_add_edge(&mut g_fl, &mut e_to, &mut e_cap, &mut e_from, e_out, rp, FL_INF);
                }
            }
            let nvtot = g_fl.len();

            // Dinic max-flow s=0 -> t=1
            let mut level_fl = vec![-1i32; nvtot];
            while fl_bfs(0, 1, &g_fl, &e_to, &e_cap, &mut level_fl) {
                let mut it = vec![0usize; nvtot];
                loop {
                    let f = fl_dfs(0, 1, FL_INF, &g_fl, &e_to, &mut e_cap, &level_fl, &mut it);
                    if f == 0 { break; }
                }
            }

            // two canonical min-cuts (HFC most-balanced selection)
            let mut vis_s_fl = vec![false; nvtot]; vis_s_fl[0] = true;
            {
                let mut stk = vec![0usize];
                while let Some(u) = stk.pop() {
                    for &idx in &g_fl[u] {
                        let v = e_to[idx];
                        if e_cap[idx] > 0 && !vis_s_fl[v] { vis_s_fl[v] = true; stk.push(v); }
                    }
                }
            }
            let mut rg_fl: Vec<Vec<usize>> = (0..nvtot).map(|_| Vec::new()).collect();
            for idx in 0..e_to.len() { rg_fl[e_to[idx]].push(idx); }
            let mut vis_t_fl = vec![false; nvtot]; vis_t_fl[1] = true;
            {
                let mut stk2 = vec![1usize];
                while let Some(v) = stk2.pop() {
                    for &idx in &rg_fl[v] {
                        if e_cap[idx] > 0 && !vis_t_fl[e_from[idx]] {
                            vis_t_fl[e_from[idx]] = true; stk2.push(e_from[idx]);
                        }
                    }
                }
            }

            // local λ−1 connectivity over affected hedges
            let conn_local_fl = |hpc: &[u16]| -> i64 {
                let mut c = 0i64;
                for &h in &h_list {
                    let mut lam = 0i64;
                    for p in 0..np { if hpc[h * np + p] > 0 { lam += 1; } }
                    if lam > 1 { c += lam - 1; }
                }
                c
            };
            let before_fl = conn_local_fl(&hpc_fl);

            // evaluate move-set: apply, measure, revert
            let eval_fl = |moves: &[(usize, usize, usize)],
                           hpc: &mut [u16], fl_sz: &mut [i32], fp: &mut [i32]| -> (i64, bool) {
                for &(v, t, _) in moves.iter() {
                    fl_move_node(v, t, fp, hpc, fl_sz, &nd_off_fl, &nd_hedges_fl, np);
                }
                let after = conn_local_fl(hpc);
                let bal = fl_sz.iter().all(|&c| c >= 1 && c <= max_ps);
                for &(v, _, o) in moves.iter().rev() {
                    fl_move_node(v, o, fp, hpc, fl_sz, &nd_off_fl, &nd_hedges_fl, np);
                }
                (after, bal)
            };

            let build_fl_moves = |to_a: &dyn Fn(usize) -> bool,
                                   fp: &[i32]| -> Vec<(usize, usize, usize)> {
                let mut mv = Vec::new();
                for (i, &v) in region.iter().enumerate() {
                    let tb = if to_a(i) { a } else { b };
                    let o = fp[v] as usize;
                    if o != tb { mv.push((v, tb, o)); }
                }
                mv
            };

            let moves_min_fl = build_fl_moves(&|i| vis_s_fl[2 + i], &fl_part);
            let moves_max_fl = build_fl_moves(&|i| !vis_t_fl[2 + i], &fl_part);

            let (after_min, bal_min) = eval_fl(&moves_min_fl, &mut hpc_fl, &mut fl_size, &mut fl_part);
            let (after_max, bal_max) = eval_fl(&moves_max_fl, &mut hpc_fl, &mut fl_size, &mut fl_part);
            let red_min = if bal_min && !moves_min_fl.is_empty() { before_fl - after_min } else { i64::MIN };
            let red_max = if bal_max && !moves_max_fl.is_empty() { before_fl - after_max } else { i64::MIN };

            let (winner_moves, winner_red) = if red_min >= red_max {
                (&moves_min_fl, red_min)
            } else {
                (&moves_max_fl, red_max)
            };
            if winner_red > 0 {
                for &(v, t, _) in winner_moves.iter() {
                    fl_move_node(v, t, &mut fl_part, &mut hpc_fl, &mut fl_size,
                                 &nd_off_fl, &nd_hedges_fl, np);
                }
            }

            for &v in &region { in_region[v] = false; rid[v] = -1; }
        }

        // global guard: exact full connectivity + strict framework balance
        let final_conn_fl = conn_full_fl(&hpc_fl);
        let balance_ok_fl = fl_size.iter().all(|&c| c >= 1 && c <= max_ps);
        if balance_ok_fl && final_conn_fl < initial_conn_fl {
            stream.memcpy_htod(&fl_part, &mut d_partition)?;
        }

        // GPU monotone guard: source of truth, cannot regress below i22 KEPT (267929)
        let post_conn_fl = measure_total_conn!(&d_partition);
        if post_conn_fl >= fl_pre_conn {
            stream.memcpy_htod(&fl_saved, &mut d_partition)?;
        }
    }

    // B_BALANCED_SWAP_PAIR terminal: GPU-scored, CPU-committed.
    // sp_gpu_score=false (default) => no-op, Q=269250 bit-exact (baseline i30).
    // sp_gpu_score=true => GPU scores swap-pair gains, CPU commits greedily.
    let sp_gpu_score: bool = hyperparameters
        .as_ref()
        .and_then(|p| p.get("sp_gpu_score").and_then(|v| v.as_bool()))
        .unwrap_or(false);

    if sp_gpu_score {
        let n_sp = challenge.num_nodes as usize;
        let m_sp = challenge.num_hyperedges as usize;
        let np_sp = num_parts_usize;
        let max_ps_sp = challenge.max_part_size as i32;

        if np_sp >= 2 {
            // Read partition; build static node->hedge CSR (ONCE — partition-independent).
            let mut sp_part_host: Vec<i32> = stream.memcpy_dtov(&d_partition)?;
            let mut nd_deg = vec![0usize; n_sp];
            for h in 0..m_sp {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    if nd < n_sp { nd_deg[nd] += 1; }
                }
            }
            let mut nd_off = vec![0u32; n_sp + 1];
            for i in 0..n_sp { nd_off[i + 1] = nd_off[i] + nd_deg[i] as u32; }
            let total_pins = nd_off[n_sp] as usize;
            let mut nd_hedges = vec![0u32; total_pins];
            {
                let mut cur: Vec<u32> = nd_off[..n_sp].to_vec();
                for h in 0..m_sp {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        if nd < n_sp {
                            nd_hedges[cur[nd] as usize] = h as u32;
                            cur[nd] += 1;
                        }
                    }
                }
            }

            // Allocate GPU buffers (ONCE — reused across passes).
            let mut d_nd_off = stream.alloc_zeros::<u32>(nd_off.len())?;
            stream.memcpy_htod(&nd_off, &mut d_nd_off)?;
            let mut d_nd_hedges = stream.alloc_zeros::<u32>(nd_hedges.len())?;
            stream.memcpy_htod(&nd_hedges, &mut d_nd_hedges)?;
            let mut d_gain_scores = stream.alloc_zeros::<i32>(n_sp * np_sp)?;
            let mut d_hpc = stream.alloc_zeros::<u16>(m_sp * np_sp)?;

            let nd_off_usize: Vec<usize> = nd_off.iter().map(|&x| x as usize).collect();
            let nd_hedges_usize: Vec<usize> = nd_hedges.iter().map(|&x| x as usize).collect();

            // Build initial hpc and compute pre-swap connectivity baseline.
            let mut hpc: Vec<u16> = vec![0u16; m_sp * np_sp];
            for h in 0..m_sp {
                let hs = hedge_offsets_host[h] as usize;
                let he = hedge_offsets_host[h + 1] as usize;
                for k in hs..he {
                    let nd = hyperedge_nodes_host[k] as usize;
                    let p = sp_part_host[nd];
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
            let mut current_conn = initial_conn;
            let mut any_improved = false;

            fn sp_apply_bsp(
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

            // 2-pass loop: pass 2 re-scores on the partition updated by pass 1.
            for _sp_pass in 0..2 {
                // Recompute hpc from current sp_part_host (accurate after pass-1 commits).
                hpc.iter_mut().for_each(|x| *x = 0);
                for h in 0..m_sp {
                    let hs = hedge_offsets_host[h] as usize;
                    let he = hedge_offsets_host[h + 1] as usize;
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        let p = sp_part_host[nd];
                        if p >= 0 && (p as usize) < np_sp {
                            hpc[h * np_sp + p as usize] = hpc[h * np_sp + p as usize].saturating_add(1);
                        }
                    }
                }

                // Upload updated partition + hpc; launch scoring kernel.
                stream.memcpy_htod(&sp_part_host, &mut d_partition)?;
                stream.memcpy_htod(&hpc, &mut d_hpc)?;
                let bsp_cfg = LaunchConfig {
                    grid_dim: ((n_sp as u32 + block_size - 1) / block_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&compute_bsp_kernel)
                        .arg(&(n_sp as i32))
                        .arg(&(np_sp as i32))
                        .arg(&d_partition)
                        .arg(&d_hpc)
                        .arg(&d_nd_off)
                        .arg(&d_nd_hedges)
                        .arg(&mut d_gain_scores)
                        .launch(bsp_cfg)?;
                }
                let gain_scores: Vec<i32> = stream.memcpy_dtov(&d_gain_scores)?;

                // Build best_to: best candidate node per (src_part, dst_part) pair.
                let mut best_to: Vec<(i32, usize)> = vec![(i32::MIN, usize::MAX); np_sp * np_sp];
                for v in 0..n_sp {
                    let p_v = sp_part_host[v];
                    if p_v < 0 || (p_v as usize) >= np_sp { continue; }
                    let p_v_u = p_v as usize;
                    for t in 0..np_sp {
                        if t == p_v_u { continue; }
                        let g = gain_scores[v * np_sp + t];
                        if g == i32::MIN { continue; }
                        let slot = p_v_u * np_sp + t;
                        if g > best_to[slot].0 || (g == best_to[slot].0 && v < best_to[slot].1) {
                            best_to[slot] = (g, v);
                        }
                    }
                }

                // Greedy commit: swap (a→b, b→a) pairs with positive joint gain.
                let mut pass_improved = false;
                for a in 0..np_sp {
                    for b in (a + 1)..np_sp {
                        let (gv, v) = best_to[a * np_sp + b];
                        let (gu, u) = best_to[b * np_sp + a];
                        if v == usize::MAX || u == usize::MAX || v == u { continue; }
                        if gv == i32::MIN || gu == i32::MIN { continue; }
                        if gv + gu <= 0 { continue; }
                        if sp_part_host[v] != a as i32 || sp_part_host[u] != b as i32 { continue; }
                        let rgv = sp_apply_bsp(v, b, &mut sp_part_host, &mut hpc, &nd_off_usize, &nd_hedges_usize, np_sp);
                        let rgu = sp_apply_bsp(u, a, &mut sp_part_host, &mut hpc, &nd_off_usize, &nd_hedges_usize, np_sp);
                        let real = rgv + rgu;
                        if real > 0 {
                            current_conn -= real;
                            pass_improved = true;
                            any_improved = true;
                        } else {
                            sp_apply_bsp(u, b, &mut sp_part_host, &mut hpc, &nd_off_usize, &nd_hedges_usize, np_sp);
                            sp_apply_bsp(v, a, &mut sp_part_host, &mut hpc, &nd_off_usize, &nd_hedges_usize, np_sp);
                        }
                    }
                }

                if !pass_improved { break; }
            }

            if any_improved && current_conn < initial_conn {
                let mut sp_nip = vec![0i32; np_sp];
                for &p in &sp_part_host {
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
                        let p = sp_part_host[nd];
                        if p >= 0 && (p as usize) < np_sp && !pseen[p as usize] {
                            pseen[p as usize] = true; d += 1;
                        }
                    }
                    if d > 1 { final_conn += d - 1; }
                    for k in hs..he {
                        let nd = hyperedge_nodes_host[k] as usize;
                        let p = sp_part_host[nd];
                        if p >= 0 && (p as usize) < np_sp { pseen[p as usize] = false; }
                    }
                }
                if balance_ok && final_conn < initial_conn {
                    let mut nodes_in_part_bsp = vec![0i32; np_sp];
                    nodes_in_part_bsp.copy_from_slice(&sp_nip);
                    stream.memcpy_htod(&sp_part_host, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_bsp, &mut d_nodes_in_part)?;
                }
            }

            let _ = (d_hpc, d_nd_off, d_nd_hedges, d_gain_scores, total_pins);
        }
    }

    let partition = stream.memcpy_dtov(&d_partition)?;

    // ---- best-of-K: score this run's final partition, keep global best ----
    {
        let mut bok_seen = vec![false; num_parts_usize];
        let run_km1 = mem_eval_lambda(
            &partition,
            &hedge_offsets_host,
            &hyperedge_nodes_host,
            num_parts_usize,
            &mut bok_seen,
        );
        if run_km1 < bok_best_km1 {
            bok_best_km1 = run_km1;
            bok_best_part = partition.iter().map(|&x| x as u32).collect();
        }
        if overlay_enabled {
            co_elites.push(partition.clone());
            co_elite_km1.push(run_km1);
        }
    }
    } // ==== end best-of-K run loop ====

    // ==== CUT-OVERLAY ENSEMBLING (K-SpecPart-style) ====
    // Contract components never separated by any elite cut into supernodes,
    // re-solve the contracted hypergraph with exact-km1 supernode FM, project
    // back, and adopt only on a strict true-km1 win (monotone vs best elite).
    // ==== DIVERSE-ELITE SELECTION (overlay_select>0) ====
    // The overlay is diversity-starved: independent-seed elites cut nearly the
    // same hyperedges, so the "clean in every elite" agreement backbone leaves
    // the re-solve almost no freedom. From the K raw elites, greedily pick an
    // `overlay_select`-sized subset with maximally DIFFERENT cut-sets (label-
    // invariant Hamming distance over the per-hyperedge cut boolean), anchored
    // on the best-km1 elite. Diverse restarts => richer overlay re-solve.
    // overlay_select==0 (default) or >= pool size => use all elites (unchanged).
    if overlay_enabled && overlay_select > 0 && overlay_select < co_elites.len() {
        let nh = hedge_offsets_host.len().saturating_sub(1);
        // Per-elite cut bitmask (u64 words over hyperedges), label-invariant.
        let words = (nh + 63) / 64;
        let cut_bits: Vec<Vec<u64>> = co_elites
            .iter()
            .map(|part| {
                let mut bits = vec![0u64; words];
                for e in 0..nh {
                    let s = hedge_offsets_host[e] as usize;
                    let t = hedge_offsets_host[e + 1] as usize;
                    if t <= s {
                        continue;
                    }
                    let first = part[hyperedge_nodes_host[s] as usize];
                    let mut cut = false;
                    for &pin in &hyperedge_nodes_host[s + 1..t] {
                        if part[pin as usize] != first {
                            cut = true;
                            break;
                        }
                    }
                    if cut {
                        bits[e >> 6] |= 1u64 << (e & 63);
                    }
                }
                bits
            })
            .collect();
        let ham = |a: &[u64], b: &[u64]| -> u64 {
            a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones() as u64).sum()
        };
        // Anchor on best-km1 elite.
        let mut anchor = 0usize;
        for i in 1..co_elite_km1.len() {
            if co_elite_km1[i] < co_elite_km1[anchor] {
                anchor = i;
            }
        }
        let mut chosen = vec![anchor];
        let mut mindist: Vec<u64> = (0..co_elites.len())
            .map(|i| ham(&cut_bits[i], &cut_bits[anchor]))
            .collect();
        while chosen.len() < overlay_select {
            // farthest-point: pick the elite maximizing min-distance to chosen set.
            let mut best = usize::MAX;
            let mut best_d = 0u64;
            for i in 0..co_elites.len() {
                if chosen.contains(&i) {
                    continue;
                }
                if mindist[i] > best_d || (mindist[i] == best_d && best == usize::MAX) {
                    best_d = mindist[i];
                    best = i;
                }
            }
            if best == usize::MAX {
                break;
            }
            chosen.push(best);
            for i in 0..co_elites.len() {
                let d = ham(&cut_bits[i], &cut_bits[best]);
                if d < mindist[i] {
                    mindist[i] = d;
                }
            }
        }
        let sel_elites: Vec<Vec<i32>> = chosen.iter().map(|&i| co_elites[i].clone()).collect();
        let sel_km1: Vec<i64> = chosen.iter().map(|&i| co_elite_km1[i]).collect();
        co_elites = sel_elites;
        co_elite_km1 = sel_km1;
    }
    if overlay_enabled && !co_elites.is_empty() {
        if let Some(proj) = co_overlay_resolve(
            &co_elites,
            &co_elite_km1,
            &hedge_offsets_host,
            &hyperedge_nodes_host,
            challenge.num_nodes as usize,
            num_parts_usize,
            challenge.max_part_size as i32,
            overlay_rounds,
            overlay_restarts,
        ) {
            // Feasibility guard: every node assigned, sizes within [1, max_ps].
            let mut co_sizes = vec![0i32; num_parts_usize];
            let mut feasible = true;
            for &p in proj.iter() {
                if p < 0 || (p as usize) >= num_parts_usize {
                    feasible = false;
                    break;
                }
                co_sizes[p as usize] += 1;
            }
            if feasible {
                for p in 0..num_parts_usize {
                    if co_sizes[p] < 1 || co_sizes[p] > challenge.max_part_size as i32 {
                        feasible = false;
                        break;
                    }
                }
            }
            if feasible {
                // mem_eval_lambda cross-check: the contracted km1 equals the
                // projected true km1 by construction, but only a recomputed
                // strict win over the incumbent is ever adopted.
                let mut co_seen = vec![false; num_parts_usize];
                let true_km1 = mem_eval_lambda(
                    &proj,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    num_parts_usize,
                    &mut co_seen,
                );
                if true_km1 < bok_best_km1 {
                    bok_best_km1 = true_km1;
                    bok_best_part = proj.iter().map(|&x| x as u32).collect();
                }
            }
        }
    }

    save_solution(&Solution {
        partition: bok_best_part,
    })?;
    Ok(())
}
