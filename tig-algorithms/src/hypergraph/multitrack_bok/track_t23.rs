use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

// ============================================================================
// MEMETIC RECOMBINATION HELPERS (deep_research_3)
// Block-aligned partition crossover: population + agreement-backbone recombine.
// All integer / index-ordered => deterministic. No RNG floats, no HashMap.
// ============================================================================

#[inline]
fn mem_compute_sizes(part: &[i32], np: usize, sizes: &mut [i32]) {
    for s in sizes.iter_mut() {
        *s = 0;
    }
    for &p in part.iter() {
        if p >= 0 && (p as usize) < np {
            sizes[p as usize] += 1;
        }
    }
}

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

// Greedy assignment of the "freed" nodes (part[n] == -1) by neighbor-block
// frequency, respecting max_part_size. Identical logic to the proven
// ruin-recreate recreate step. Returns false if it cannot place a node.
fn mem_greedy_recreate(
    part: &mut [i32],
    sizes: &mut [i32],
    free_nodes: &[usize],
    node_offsets: &[i32],
    node_hedges: &[i32],
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    np: usize,
    maxp: i32,
    score: &mut [i32],
    stamp_arr: &mut [u32],
) -> bool {
    let mut stamp: u32 = 0;
    for &n in free_nodes.iter() {
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
                let p = part[hedge_nodes[k] as usize];
                if p >= 0 && (p as usize) < np && stamp_arr[p as usize] != stamp {
                    stamp_arr[p as usize] = stamp;
                    score[p as usize] += 1;
                }
            }
        }
        let mut best_p = -1i32;
        let mut best_score = -1i32;
        let mut best_size = i32::MAX;
        for p in 0..np {
            if sizes[p] >= maxp {
                continue;
            }
            if score[p] > best_score || (score[p] == best_score && sizes[p] < best_size) {
                best_score = score[p];
                best_size = sizes[p];
                best_p = p as i32;
            }
        }
        if best_p < 0 {
            let mut mn = i32::MAX;
            for p in 0..np {
                if sizes[p] < mn {
                    mn = sizes[p];
                    best_p = p as i32;
                }
            }
        }
        if best_p < 0 {
            return false;
        }
        part[n] = best_p;
        sizes[best_p as usize] += 1;
    }
    true
}

// Solve the block-label symmetry: relabel `other`'s blocks to maximally overlap
// `reference` via greedy max-overlap bipartite matching. Writes the relabeled
// partition into `out`. All scratch buffers are caller-preallocated.
fn mem_align(
    reference: &[i32],
    other: &[i32],
    np: usize,
    overlap: &mut [i64],
    perm: &mut [i32],
    out: &mut [i32],
    scratch: &mut Vec<(i64, i32, i32)>,
    ref_used: &mut [bool],
    oth_used: &mut [bool],
) {
    for v in overlap.iter_mut() {
        *v = 0;
    }
    for i in 0..reference.len() {
        let r = reference[i];
        let o = other[i];
        if r >= 0 && (r as usize) < np && o >= 0 && (o as usize) < np {
            overlap[o as usize * np + r as usize] += 1;
        }
    }
    scratch.clear();
    for o in 0..np {
        for r in 0..np {
            let c = overlap[o * np + r];
            if c > 0 {
                scratch.push((c, o as i32, r as i32));
            }
        }
    }
    // Descending overlap; deterministic tie-break by (other, ref) index.
    scratch.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
    });
    for u in ref_used.iter_mut() {
        *u = false;
    }
    for u in oth_used.iter_mut() {
        *u = false;
    }
    for p in perm.iter_mut() {
        *p = -1;
    }
    for &(_, o, r) in scratch.iter() {
        let ou = o as usize;
        let ru = r as usize;
        if !oth_used[ou] && !ref_used[ru] {
            perm[ou] = r;
            oth_used[ou] = true;
            ref_used[ru] = true;
        }
    }
    // Leftover other-blocks (no positive overlap) -> leftover ref-blocks.
    let mut next_ref = 0usize;
    for o in 0..np {
        if perm[o] < 0 {
            while next_ref < np && ref_used[next_ref] {
                next_ref += 1;
            }
            if next_ref < np {
                perm[o] = next_ref as i32;
                ref_used[next_ref] = true;
                next_ref += 1;
            } else {
                perm[o] = o as i32;
            }
        }
    }
    for i in 0..other.len() {
        let o = other[i];
        out[i] = if o >= 0 && (o as usize) < np {
            perm[o as usize]
        } else {
            o
        };
    }
}

// Free a pseudo-random subset of nodes (deterministic LCG from `seed`) to build
// a diverse population member. Leaves part[n] == -1 for freed nodes; caller then
// runs mem_greedy_recreate. freed_flag is left all-false on exit (reusable).
fn mem_random_ruin(
    part: &mut [i32],
    sizes: &mut [i32],
    free_nodes: &mut Vec<usize>,
    freed_flag: &mut [bool],
    num_free: usize,
    seed: u64,
    np: usize,
) {
    free_nodes.clear();
    let n = part.len();
    let mut state = seed | 1;
    let cap = num_free.min(n);
    let mut attempts = 0usize;
    let max_attempts = n.saturating_mul(4);
    while free_nodes.len() < cap && attempts < max_attempts {
        attempts += 1;
        state = state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let node = (state % n as u64) as usize;
        if freed_flag[node] {
            continue;
        }
        let cur = part[node];
        if cur < 0 || (cur as usize) >= np {
            continue;
        }
        if sizes[cur as usize] <= 1 {
            continue;
        }
        sizes[cur as usize] -= 1;
        part[node] = -1;
        freed_flag[node] = true;
        free_nodes.push(node);
    }
    for &nd in free_nodes.iter() {
        freed_flag[nd] = false;
    }
}

// ============================================================================
// CUT-OVERLAY ENSEMBLING (K-SpecPart-style)
// Overlay the cuts of all K elite partitions: a hyperedge is "clean" iff it is
// uncut in EVERY elite. Union-Find over clean hyperedges yields supernodes
// (components never separated by any elite cut). Contract, re-solve the small
// hypergraph with exact-km1 supernode FM (moves whole supernodes = correlated
// multi-node consolidation single-node FM cannot do), project back, and adopt
// only on a strict true-km1 improvement (monotone). All host-side, all
// index-ordered and RNG-free => deterministic.
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

    let hyperedge_cluster_kernel = module.load_function("hyperedge_clustering_20k")?;
    let compute_preferences_kernel = module.load_function("compute_node_preferences_20k")?;
    let execute_assignments_kernel = module.load_function("execute_node_assignments_20k")?;
    let precompute_edge_flags_kernel = module.load_function("precompute_edge_flags_20k")?;

    let compute_moves_kernel = module.load_function("compute_refinement_moves_optimized_20k")?;
    let balance_kernel = module.load_function("balance_final_20k")?;
    let compute_connectivity_kernel = module.load_function("compute_connectivity_20k")?;
    let reduce_connectivity_sum_kernel = module.load_function("reduce_connectivity_sum_20k")?;
    let compute_swap_gains_kernel = module.load_function("compute_swap_gains_extended_20k")?;
    let choose_elite_per_hyperedge_kernel = module.load_function("choose_elite_per_hyperedge_20k")?;
    let assign_from_elite_votes_kernel = module.load_function("assign_from_elite_votes_20k")?;

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
        5 => (7000, 6, 70, 300, 0),
        4 => (5000, 5, 60, 200, 0),
        3 => (3000, 5, 50, 150, 64),
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

    let tabu_fail_tenure = 3usize;
    let tabu_mark_base = 2048usize;
    let tabu_mark_mult = 16usize;
    let tabu_fail_mark_len = 2048usize;

    let extra_window = 49152usize;
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
        .map(|v| v.clamp(1, 15) as usize) // K_max=14: measured fuel-safe max
        .unwrap_or(2);
    // ==== cut-overlay ensembling hyperparameters (overlay=0 => exactly bok) ====
    let overlay_enabled = hyperparameters
        .as_ref()
        .and_then(|p| p.get("overlay").and_then(|v| v.as_i64()))
        .unwrap_or(1) // overlay ON by default: monotone (strict-win adopt only)
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

    // ==== ported block ====
{
    let node_offsets_host = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let node_hedges_host = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;

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
        seed: u64,
    ) -> bool {
        let num_nodes = partition.len();
        let mut part_new: Vec<i32> = partition.clone();
        let mut sizes_new: Vec<i32> = part_sizes.clone();

        let mut stamp_arr: Vec<u32> = vec![0u32; num_parts];
        let mut stamp: u32 = 0;

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
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let free_cap = std::cmp::max(64usize, num_nodes / 4);
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

    let np010 = num_parts_usize;
    let nh010 = challenge.num_hyperedges as usize;
    let eval010 = |pp: &Vec<i32>| -> i64 { let mut c=0i64; let mut ps=vec![false;np010]; for h in 0..nh010 { let hs=hedge_offsets_host[h] as usize; let he=hedge_offsets_host[h+1] as usize; let mut d=0i64; for k in hs..he { let p=pp[hyperedge_nodes_host[k] as usize]; if p>=0 && (p as usize)<np010 { let pu=p as usize; if !ps[pu]{ps[pu]=true; d+=1;} } } if d>1 { c+=d-1; } for k in hs..he { let p=pp[hyperedge_nodes_host[k] as usize]; if p>=0 && (p as usize)<np010 { ps[p as usize]=false; } } } c };
    let k010 = ((challenge.num_hyperedges as usize)/50).max(1);
    let mut best_pp = partition_host_refine.clone(); let mut best_nn = nodes_in_part_host.clone(); let mut best_cc = eval010(&partition_host_refine);
    for sd_base in [0x9E3779B97F4A7C15u64, 0xD1B54A32D192ED03u64, 0xCA5A826395121157u64, 0xBF58476D1CE4E5B9u64, 0x94D049BB133111EBu64] {
        let sd = sd_base.wrapping_add(run_off);
        let mut pp = partition_host_refine.clone(); let mut nn = nodes_in_part_host.clone();
        if ruin_recreate(&mut pp, &mut nn, &hedge_offsets_host, &hyperedge_nodes_host, &node_offsets_host, &node_hedges_host, nh010, np010, challenge.max_part_size as i32, k010, sd) {
            let c = eval010(&pp); if c < best_cc { best_cc=c; best_pp=pp; best_nn=nn; }
        }
    }
    partition_host_refine.copy_from_slice(&best_pp);
    nodes_in_part_host.copy_from_slice(&best_nn);
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

    // ========================================================================
    // MEMETIC RECOMBINATION PHASE (deep_research_3)
    // The single-solution local optimum reached above is empirically stable
    // (FM / flow / multilevel / swaps all null). Escape it GLOBALLY: build a
    // small population of diverse-but-good partitions, then recombine PAIRS via
    // block-aligned agreement-backbone crossover. The child freezes the region
    // where two independent good solutions AGREE and re-optimizes only where
    // they DISAGREE -- a coordinated structural jump no single-move / swap /
    // flow operator can express. Uses the ~16k ms of otherwise-idle budget.
    // Safe-by-construction: the champion is only replaced on strict improvement.
    // ========================================================================
    let mem_enabled = hyperparameters
        .as_ref()
        .and_then(|p| p.get("memetic").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        != 0;
    if mem_enabled {
        let node_offsets_host = stream.memcpy_dtov(&challenge.d_node_offsets)?;
        let node_hedges_host = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
        let np = num_parts_usize;
        let n = challenge.num_nodes as usize;
        let maxp = challenge.max_part_size as i32;

        let mem_pop = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_pop").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(2, 16) as usize)
            .unwrap_or(5);
        let mem_recomb = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_recomb").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(0, 200) as usize)
            .unwrap_or(16);
        let mem_polish_rounds = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_polish").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(20, 2000) as usize)
            .unwrap_or(250);
        let mem_free_pct = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_free_pct").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(2, 60) as usize)
            .unwrap_or(20);

        // --- deep_research_6: diversity-preserving generational GA knobs ---
        // The recombination loop below IS steady-state generational (children
        // re-enter the pool, best-of-pool is re-selected each iteration), but it
        // converges prematurely: once every pool member descends from the same
        // champion basin, aligned agreement -> ~100%, free_nodes -> empty, and
        // crossover degenerates to a no-op. `mem_diversity` re-injects fresh
        // strong-ruin immigrants once the pool stagnates, restoring the
        // structural diversity crossover needs to keep compounding. Defaults OFF
        // => byte-identical behavior to the shipped +183 config.
        let mem_diversity = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_diversity").and_then(|v| v.as_i64()))
            .unwrap_or(0)
            != 0;
        // Stagnation streak (degenerate/non-improving crossovers) that triggers
        // a diversity re-injection.
        let mem_immig_gap = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_immig_gap").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(1, 200) as usize)
            .unwrap_or(8);
        // Ruin strength for an immigrant (stronger than mem_free_pct => distinct
        // basin, not a near-clone of the champion).
        let mem_immig_free_pct = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_immig_free_pct").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(5, 80) as usize)
            .unwrap_or(35);
        // Cheap FM polish for an immigrant (short: it only needs to reach a
        // basin, crossover does the heavy lifting). Keeps the budget bounded.
        let mem_immig_polish = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_immig_polish").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(0, 2000) as usize)
            .unwrap_or((mem_polish_rounds / 3).max(30));
        // Hard cap on total immigrants (budget guard: each costs one short
        // polish and revives full-cost productive crossovers afterwards).
        let mem_immig_max = hyperparameters
            .as_ref()
            .and_then(|p| p.get("mem_immig_max").and_then(|v| v.as_i64()))
            .map(|v| v.clamp(0, 500) as usize)
            .unwrap_or(mem_recomb / 6 + 2);

        // Preallocated scratch (no per-iteration heap churn).
        let mut seen_buf: Vec<bool> = vec![false; np];
        let mut score_buf: Vec<i32> = vec![0i32; np];
        let mut stamp_arr: Vec<u32> = vec![0u32; np];
        let mut overlap: Vec<i64> = vec![0i64; np * np];
        let mut perm: Vec<i32> = vec![0i32; np];
        let mut align_scratch: Vec<(i64, i32, i32)> = Vec::with_capacity(np * np);
        let mut ref_used: Vec<bool> = vec![false; np];
        let mut oth_used: Vec<bool> = vec![false; np];
        let mut aligned: Vec<i32> = vec![0i32; n];
        let mut child: Vec<i32> = vec![0i32; n];
        let mut child_sizes: Vec<i32> = vec![0i32; np];
        let mut free_nodes: Vec<usize> = Vec::with_capacity(n);
        let mut freed_flag: Vec<bool> = vec![false; n];

        // Population buffers (preallocated).
        let mut pop_part: Vec<Vec<i32>> = vec![vec![0i32; n]; mem_pop];
        let mut pop_sizes: Vec<Vec<i32>> = vec![vec![0i32; np]; mem_pop];
        let mut pop_score: Vec<i64> = vec![i64::MAX; mem_pop];

        // GPU FM descent (pure positive-gain greedy) + swap phase, in place on
        // partition_host_refine / nodes_in_part_host / d_partition.
        macro_rules! mem_fm_polish {
            ($rounds:expr) => {{
                for _ in 0..$rounds {
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
                            if gain > 0 {
                                valid_moves.push((node, key));
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
                    let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(extra_window / 2));
                    if k_cand > 1 {
                        valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                        valid_moves[..k_cand].sort_unstable_by(cmp);
                    } else {
                        valid_moves[..k_cand].sort_unstable_by(cmp);
                    }
                    tgt_used.fill(0);
                    for p in 0..num_parts_usize {
                        let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
                        tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack_mid + 2));
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
                        sorted_move_nodes.extend(valid_moves[..take].iter().map(|(nn, _)| *nn as i32));
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
                    if me == 0 {
                        break;
                    }
                }
                do_swap_phase!(
                    &mut d_partition, &mut d_nodes_in_part,
                    &mut d_edge_flags_all, &mut d_edge_flags_double,
                    &mut d_swap_gains, &mut swap_gains_host,
                    &mut partition_host_swap, &mut partition_mut_swap,
                    &mut part_to_part, &mut used_ba_buf,
                    15, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
                )?;
                partition_host_refine.copy_from_slice(&partition_host_swap);
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                mem_compute_sizes(&partition_host_refine, num_parts_usize, &mut nodes_in_part_host);
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            }};
        }

        // Member 0 = the champion.
        pop_part[0].copy_from_slice(&best_partition_host);
        pop_sizes[0].copy_from_slice(&best_nodes_in_part_host);
        pop_score[0] = best_connectivity as i64;

        let mut mem_best_part = best_partition_host.clone();
        let mut mem_best_sizes = best_nodes_in_part_host.clone();
        let mut mem_best_score = best_connectivity as i64;

        // Build diverse population members 1..mem_pop via strong random ruin of
        // the champion + greedy recreate + short FM descent -> distinct basins.
        for m in 1..mem_pop {
            pop_part[m].copy_from_slice(&best_partition_host);
            pop_sizes[m].copy_from_slice(&best_nodes_in_part_host);
            let num_free = (n * mem_free_pct) / 100;
            let seed = 0x1234_5678_9ABC_DEF0u64
                .wrapping_mul(m as u64 + 1)
                .wrapping_add(0x9E3779B97F4A7C15u64)
                .wrapping_add(run_off);
            mem_random_ruin(
                &mut pop_part[m],
                &mut pop_sizes[m],
                &mut free_nodes,
                &mut freed_flag,
                num_free,
                seed,
                np,
            );
            let ok = mem_greedy_recreate(
                &mut pop_part[m],
                &mut pop_sizes[m],
                &free_nodes,
                &node_offsets_host,
                &node_hedges_host,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                np,
                maxp,
                &mut score_buf,
                &mut stamp_arr,
            );
            if !ok {
                pop_part[m].copy_from_slice(&best_partition_host);
                pop_sizes[m].copy_from_slice(&best_nodes_in_part_host);
            }
            partition_host_refine.copy_from_slice(&pop_part[m]);
            nodes_in_part_host.copy_from_slice(&pop_sizes[m]);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            mem_fm_polish!(mem_polish_rounds);
            pop_part[m].copy_from_slice(&partition_host_refine);
            pop_sizes[m].copy_from_slice(&nodes_in_part_host);
            pop_score[m] = mem_eval_lambda(
                &pop_part[m],
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                np,
                &mut seen_buf,
            );
            if pop_score[m] < mem_best_score {
                mem_best_score = pop_score[m];
                mem_best_part.copy_from_slice(&pop_part[m]);
                mem_best_sizes.copy_from_slice(&pop_sizes[m]);
            }
        }

        // Recombination rounds: cross the current best member with a rotating
        // partner, freeze the aligned agreement backbone, recreate + polish.
        // deep_research_6: mem_stagnation counts degenerate/non-improving
        // crossovers; when it hits mem_immig_gap we re-inject a diverse
        // immigrant (bounded by mem_immig_max) to break premature convergence.
        let mut mem_stagnation = 0usize;
        let mut mem_immig_count = 0usize;
        let mut mem_immig_ctr = 0u64;
        for it in 0..mem_recomb {
            // Diversity re-injection: replace the worst non-champion member with
            // a fresh strong-ruin immigrant so subsequent crossovers again have
            // a genuinely different (yet good) basin to recombine against.
            if mem_diversity
                && mem_stagnation >= mem_immig_gap
                && mem_immig_count < mem_immig_max
                && mem_pop > 1
            {
                let mut w = 1usize;
                let mut w_sc = pop_score[1];
                for i in 2..mem_pop {
                    if pop_score[i] > w_sc {
                        w_sc = pop_score[i];
                        w = i;
                    }
                }
                pop_part[w].copy_from_slice(&best_partition_host);
                pop_sizes[w].copy_from_slice(&best_nodes_in_part_host);
                mem_immig_ctr = mem_immig_ctr.wrapping_add(1);
                let num_free = (n * mem_immig_free_pct) / 100;
                let seed = 0xC2B2AE3D27D4EB4Fu64
                    .wrapping_mul(mem_immig_ctr.wrapping_add(1))
                    .wrapping_add(0x165667B19E3779F9u64.wrapping_mul(it as u64 + 1))
                    .wrapping_add(run_off);
                mem_random_ruin(
                    &mut pop_part[w],
                    &mut pop_sizes[w],
                    &mut free_nodes,
                    &mut freed_flag,
                    num_free,
                    seed,
                    np,
                );
                let ok_im = mem_greedy_recreate(
                    &mut pop_part[w],
                    &mut pop_sizes[w],
                    &free_nodes,
                    &node_offsets_host,
                    &node_hedges_host,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    np,
                    maxp,
                    &mut score_buf,
                    &mut stamp_arr,
                );
                if ok_im {
                    partition_host_refine.copy_from_slice(&pop_part[w]);
                    nodes_in_part_host.copy_from_slice(&pop_sizes[w]);
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                    mem_fm_polish!(mem_immig_polish);
                    pop_part[w].copy_from_slice(&partition_host_refine);
                    pop_sizes[w].copy_from_slice(&nodes_in_part_host);
                    pop_score[w] = mem_eval_lambda(
                        &pop_part[w],
                        &hedge_offsets_host,
                        &hyperedge_nodes_host,
                        np,
                        &mut seen_buf,
                    );
                    if pop_score[w] < mem_best_score {
                        mem_best_score = pop_score[w];
                        mem_best_part.copy_from_slice(&pop_part[w]);
                        mem_best_sizes.copy_from_slice(&pop_sizes[w]);
                    }
                } else {
                    pop_part[w].copy_from_slice(&best_partition_host);
                    pop_sizes[w].copy_from_slice(&best_nodes_in_part_host);
                    pop_score[w] = best_connectivity as i64;
                }
                mem_immig_count += 1;
                mem_stagnation = 0;
            }

            let mut a = 0usize;
            let mut a_sc = pop_score[0];
            for i in 1..mem_pop {
                if pop_score[i] < a_sc {
                    a_sc = pop_score[i];
                    a = i;
                }
            }
            let mut b = it % mem_pop;
            if b == a {
                b = (a + 1) % mem_pop;
            }
            if b == a {
                mem_stagnation += 1;
                continue;
            }

            mem_align(
                &pop_part[a],
                &pop_part[b],
                np,
                &mut overlap,
                &mut perm,
                &mut aligned,
                &mut align_scratch,
                &mut ref_used,
                &mut oth_used,
            );

            // Agreement backbone from A; disagreement nodes freed.
            free_nodes.clear();
            for i in 0..n {
                if pop_part[a][i] == aligned[i] {
                    child[i] = pop_part[a][i];
                } else {
                    child[i] = -1;
                    free_nodes.push(i);
                }
            }
            if free_nodes.is_empty() {
                // Degenerate crossover: A and B agree everywhere after alignment
                // (pool has converged). Count it toward re-injection.
                mem_stagnation += 1;
                continue;
            }
            mem_compute_sizes(&child, np, &mut child_sizes);
            let ok = mem_greedy_recreate(
                &mut child,
                &mut child_sizes,
                &free_nodes,
                &node_offsets_host,
                &node_hedges_host,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                np,
                maxp,
                &mut score_buf,
                &mut stamp_arr,
            );
            if !ok {
                mem_stagnation += 1;
                continue;
            }
            let mut valid = true;
            for p in 0..np {
                if child_sizes[p] < 1 || child_sizes[p] > maxp {
                    valid = false;
                    break;
                }
            }
            if !valid {
                mem_stagnation += 1;
                continue;
            }

            partition_host_refine.copy_from_slice(&child);
            nodes_in_part_host.copy_from_slice(&child_sizes);
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            mem_fm_polish!(mem_polish_rounds);
            let child_score = mem_eval_lambda(
                &partition_host_refine,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                np,
                &mut seen_buf,
            );

            let improved_best = child_score < mem_best_score;
            if improved_best {
                mem_best_score = child_score;
                mem_best_part.copy_from_slice(&partition_host_refine);
                mem_best_sizes.copy_from_slice(&nodes_in_part_host);
            }

            // Insert into population by replacing the worst member if better.
            let mut w = 0usize;
            let mut w_sc = pop_score[0];
            for i in 1..mem_pop {
                if pop_score[i] > w_sc {
                    w_sc = pop_score[i];
                    w = i;
                }
            }
            let inserted = child_score < w_sc;
            if inserted {
                pop_part[w].copy_from_slice(&partition_host_refine);
                pop_sizes[w].copy_from_slice(&nodes_in_part_host);
                pop_score[w] = child_score;
            }

            // deep_research_6: a crossover that neither improved the champion nor
            // entered the pool is "wasted" -> advance the stagnation streak toward
            // the next diversity re-injection. Any real progress resets it.
            if improved_best || inserted {
                mem_stagnation = 0;
            } else {
                mem_stagnation += 1;
            }
        }

        // Commit the memetic champion (strict improvement only).
        if mem_best_score < best_connectivity as i64 {
            best_connectivity = mem_best_score as i32;
            best_partition_host.copy_from_slice(&mem_best_part);
            best_nodes_in_part_host.copy_from_slice(&mem_best_sizes);
        }
        stream.memcpy_htod(&best_partition_host, &mut d_partition)?;
        stream.memcpy_htod(&best_nodes_in_part_host, &mut d_nodes_in_part)?;
        partition_host_refine.copy_from_slice(&best_partition_host);
        nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);
    }

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
