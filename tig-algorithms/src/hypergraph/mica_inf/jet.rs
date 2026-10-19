// Batch refinement: every node proposes its best move, the proposals are chained by source
// block, filtered for balance and for conflicts on shared hyperedges, then applied in one
// pass; a rebalance follows each round. Also three-block cyclic exchanges.
// Precondition: unit node weights (the uncontracted hypergraph) and a feasible input partition.

use super::gain::{Block, Node, State};

// Blocks may overshoot max_block_size by this much during a batch; rebalance repairs it.
const OVER: u32 = 1;
// Longest chain of proposals followed from one start node.
const CHAIN_MAX: usize = 64;

fn round(st: &mut State, lock: &[bool], moved: &mut [bool]) {
    let n = st.n;
    let mut target: Vec<Block> = vec![u16::MAX; n];
    let mut gain0: Vec<i32> = vec![0; n];
    let mut cands: Vec<Node> = Vec::new();
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    for v in 0..n as Node {
        if lock[v as usize] {
            continue;
        }
        let from = st.part[v as usize];
        if st.block_size[from as usize] <= st.weight[v as usize] {
            continue;
        }
        st.adjacent_blocks(v, &mut adj);
        let mut bg = i32::MIN;
        let mut bb = u16::MAX;
        for &b in adj.iter() {
            if b == from {
                continue;
            }
            let g = st.gain_cached(v, b);
            if g > bg {
                bg = g;
                bb = b;
            }
        }
        if bb == u16::MAX {
            continue;
        }
        if bg >= 0 {
            target[v as usize] = bb;
            gain0[v as usize] = bg;
            cands.push(v);
        }
    }
    if cands.is_empty() {
        return;
    }

    // Priority: chains of proposals (each node followed by the best untaken proposal leaving
    // its target block), ordered by total gain.
    let mut prio: Vec<u32> = vec![u32::MAX; n];
    {
        let mut by_source: Vec<Vec<Node>> = vec![Vec::new(); st.k];
        for &v in cands.iter() {
            by_source[st.part[v as usize] as usize].push(v);
        }
        for l in by_source.iter_mut() {
            l.sort_unstable_by(|&a, &b| {
                gain0[b as usize].cmp(&gain0[a as usize]).then(a.cmp(&b))
            });
        }
        let mut head: Vec<usize> = vec![0; st.k];
        let mut taken = vec![false; n];
        macro_rules! next_available {
            ($b:expr) => {{
                let bb = $b;
                while head[bb] < by_source[bb].len() && taken[by_source[bb][head[bb]] as usize] {
                    head[bb] += 1;
                }
                if head[bb] < by_source[bb].len() { Some(by_source[bb][head[bb]]) } else { None }
            }};
        }
        let mut start_nodes: Vec<Node> = cands.clone();
        start_nodes.sort_unstable_by(|&a, &b| {
            gain0[b as usize].cmp(&gain0[a as usize]).then(a.cmp(&b))
        });
        let mut chains: Vec<(i64, Vec<Node>)> = Vec::new();
        for &d in start_nodes.iter() {
            if taken[d as usize] { continue; }
            let mut c: Vec<Node> = Vec::new();
            let mut g = 0i64;
            let mut x = d;
            loop {
                taken[x as usize] = true;
                g += gain0[x as usize] as i64;
                c.push(x);
                let t = target[x as usize] as usize;
                match next_available!(t) {
                    Some(m) if !taken[m as usize] => x = m,
                    _ => break,
                }
                if c.len() >= CHAIN_MAX { break; }
            }
            chains.push((g, c));
        }
        chains.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1[0].cmp(&b.1[0])));
        let mut rank = 0u32;
        for (_, c) in chains.iter() {
            for &v in c.iter() {
                prio[v as usize] = rank;
                rank += 1;
            }
        }
    }

    // Balance filter in priority order.
    {
        let mut pre_order: Vec<Node> = cands.clone();
        pre_order.sort_unstable_by_key(|&v| prio[v as usize]);
        let mut sizes: Vec<i64> = st.block_size.iter().map(|&x| x as i64).collect();
        let cap = (st.max_block_size + OVER) as i64;
        let mut kept: Vec<Node> = Vec::with_capacity(pre_order.len());
        for &v in pre_order.iter() {
            let from = st.part[v as usize] as usize;
            let to = target[v as usize] as usize;
            let w = st.weight[v as usize] as i64;
            if sizes[to] + w > cap || sizes[from] <= w {
                target[v as usize] = u16::MAX;
                continue;
            }
            sizes[to] += w;
            sizes[from] -= w;
            kept.push(v);
        }
        cands = kept;
        if cands.is_empty() { return; }
    }

    // Gain of each proposal once the higher-priority proposals on the same hyperedges are applied.
    let mut rg: Vec<i32> = vec![0; n];
    let mut moving: Vec<Node> = Vec::with_capacity(64);
    let mut phi: Vec<i32> = vec![0; st.k];
    for e in 0..st.m {
        let lo = st.he_off[e] as usize;
        let hi = st.he_off[e + 1] as usize;
        moving.clear();
        for i in lo..hi {
            let v = st.he_nodes[i];
            if target[v as usize] != u16::MAX {
                moving.push(v);
            }
        }
        if moving.is_empty() {
            continue;
        }
        phi.fill(0);
        for i in lo..hi {
            phi[st.part[st.he_nodes[i] as usize] as usize] += 1;
        }
        moving.sort_unstable_by_key(|&v| prio[v as usize]);
        for &v in moving.iter() {
            let from = st.part[v as usize] as usize;
            let to = target[v as usize] as usize;
            phi[from] -= 1;
            if phi[from] == 0 {
                rg[v as usize] += 1;
            }
            phi[to] += 1;
            if phi[to] == 1 {
                rg[v as usize] -= 1;
            }
        }
    }

    let mut order: Vec<Node> = cands
        .iter()
        .copied()
        .filter(|&v| rg[v as usize] >= 0)
        .collect();
    order.sort_unstable_by_key(|&v| prio[v as usize]);

    for &v in order.iter() {
        let to = target[v as usize];
        if st.part[v as usize] == to {
            continue;
        }
        if st.block_size[to as usize] + st.weight[v as usize] > st.max_block_size + OVER {
            continue;
        }
        if st.block_size[st.part[v as usize] as usize] <= st.weight[v as usize] {
            continue;
        }
        st.apply(v, to);
        moved[v as usize] = true;
    }
}

// Nodes moved in one round are locked for the next; stops at the first round that leaves km1 unchanged.
pub fn refine(st: &mut State, rounds: usize) {
    let mut lock = vec![false; st.n];
    for _ in 0..rounds {
        let mut moved = vec![false; st.n];
        let before = st.km1();
        round(st, &lock, &mut moved);
        lock.copy_from_slice(&moved);
        rebalance(st);
        if st.km1() == before {
            break;
        }
    }
}

// Tries to drain blocks above max_block_size into blocks below max_block_size - 1, best gain first
// (at most k + 1 rounds).
fn rebalance(st: &mut State) {
    let k = st.k;
    let l_max = st.max_block_size as i64;
    let dead_zone: i64 = 1;
    let z = l_max - dead_zone;
    debug_assert!(k as i64 * z >= st.n as i64, "dead zone too wide, deadlock possible");

    let mut adj: Vec<Block> = Vec::with_capacity(64);
    for _round in 0..(k + 1) {
        let mut surplus: Vec<i64> = (0..k)
            .map(|p| (st.block_size[p] as i64 - l_max).max(0))
            .collect();
        if surplus.iter().all(|&s| s == 0) { break; }

        let underfull: Vec<Block> = (0..k)
            .filter(|&p| (st.block_size[p] as i64) < z)
            .map(|p| p as Block)
            .collect();
        if underfull.is_empty() { break; }

        let mut cand: Vec<(i32, Node, Block)> = Vec::new();
        for v in 0..st.n as Node {
            let p = st.part[v as usize] as usize;
            if surplus[p] == 0 { continue; }
            st.adjacent_blocks(v, &mut adj);
            let mut best: Option<(i32, Block)> = None;
            for &b in adj.iter() {
                if b as usize == p { continue; }
                if (st.block_size[b as usize] as i64) >= z { continue; }
                let g = st.gain_cached(v, b);
                if best.map_or(true, |(bg, bb)| g > bg || (g == bg && b < bb)) {
                    best = Some((g, b));
                }
            }
            let (g, to) = match best {
                Some(x) => x,
                None => {
                    let to = underfull[(v as usize) % underfull.len()];
                    if to as usize == p { continue; }
                    (st.gain_cached(v, to), to)
                }
            };
            cand.push((g, v, to));
        }
        if cand.is_empty() { break; }

        cand.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

        let mut remaining = surplus.clone();
        for &(_, v, to0) in cand.iter() {
            let p = st.part[v as usize] as usize;
            if remaining[p] == 0 { continue; }
            let mut to = to0;
            if (st.block_size[to as usize] as i64) >= z {
                let mut best: Option<(i32, Block)> = None;
                st.adjacent_blocks(v, &mut adj);
                for &b in adj.iter() {
                    if b as usize == p { continue; }
                    if (st.block_size[b as usize] as i64) >= z { continue; }
                    let g = st.gain_cached(v, b);
                    if best.map_or(true, |(bg, bb)| g > bg || (g == bg && b < bb)) {
                        best = Some((g, b));
                    }
                }
                match best {
                    Some((_, b)) => to = b,
                    None => match underfull.iter().find(|&&b| (st.block_size[b as usize] as i64) < z
                                                        && b as usize != p) {
                        Some(&b) => to = b,
                        None => continue,
                    },
                }
            }
            if st.block_size[p] <= st.weight[v as usize] { continue; }
            st.apply(v, to);
            remaining[p] -= 1;
            surplus[p] -= 1;
        }
    }
}

// Cyclic exchanges x -> y -> z -> x of the best node of each ordered block pair (block sizes are
// preserved); an exchange is kept only when its measured gain is positive.
pub fn cycles(st: &mut State, rounds: usize) {
    let k = st.k;
    let mut adj: Vec<Block> = Vec::with_capacity(64);

    for _ in 0..rounds {
        let mut best_v: Vec<u32> = vec![u32::MAX; k * k];
        let mut best_g: Vec<i32> = vec![i32::MIN; k * k];
        for v in 0..st.n as Node {
            let x = st.part[v as usize] as usize;
            if st.block_size[x] <= st.weight[v as usize] { continue; }
            st.adjacent_blocks(v, &mut adj);
            for &y in adj.iter() {
                if y as usize == x { continue; }
                let g = st.gain_cached(v, y);
                let i = x * k + y as usize;
                if best_v[i] == u32::MAX || g > best_g[i] || (g == best_g[i] && v < best_v[i]) {
                    best_g[i] = g;
                    best_v[i] = v;
                }
            }
        }

        let mut found = 0usize;
        let mut used = vec![false; st.n];
        let mut cands: Vec<(i32, usize, usize, usize)> = Vec::new();
        for x in 0..k {
            for y in 0..k {
                if y == x || best_v[x * k + y] == u32::MAX { continue; }
                for z in 0..k {
                    if z == x || z == y { continue; }
                    if best_v[y * k + z] == u32::MAX || best_v[z * k + x] == u32::MAX { continue; }
                    let s = best_g[x * k + y] + best_g[y * k + z] + best_g[z * k + x];
                    if s > 0 { cands.push((s, x, y, z)); }
                }
            }
        }
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)));

        for &(_, x, y, z) in cands.iter() {
            let (v1, v2, v3) = (best_v[x * k + y], best_v[y * k + z], best_v[z * k + x]);
            if used[v1 as usize] || used[v2 as usize] || used[v3 as usize] { continue; }
            if v1 == v2 || v2 == v3 || v1 == v3 { continue; }
            let mut total = 0i64;
            total += st.apply(v1, y as Block) as i64;
            total += st.apply(v2, z as Block) as i64;
            total += st.apply(v3, x as Block) as i64;
            if total <= 0 {
                st.apply(v1, x as Block);
                st.apply(v2, y as Block);
                st.apply(v3, z as Block);
            } else {
                used[v1 as usize] = true;
                used[v2 as usize] = true;
                used[v3 as usize] = true;
                found += 1;
            }
        }
        if found == 0 { break; }
    }
}
