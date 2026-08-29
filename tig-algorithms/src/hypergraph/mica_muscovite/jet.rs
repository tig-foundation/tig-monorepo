// Batch refinement with an afterburner, plus exchange cycles and paths.

use super::gain::{Block, Node, State};

pub struct Config {
    // iso_host: skips host work that cannot change the result.
    pub iso_host: bool,
    // tie_host: unifies the equal-gain tie-break across host stages.
    pub tie_host: bool,
    pub rounds: usize,
    pub accept_zero: bool,
    pub unconstrained: bool,
    pub over: u32,
    pub chain: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config { iso_host: false, tie_host: false, rounds: 8, accept_zero: true, unconstrained: true, over: 0, chain: false }
    }
}

pub fn round(st: &mut State, cfg: &Config, lock: &[bool], bouges: &mut [bool]) -> i64 {
    let chain_mode = cfg.chain;
    let n = st.n;
    let mut target: Vec<Block> = vec![u16::MAX; n];
    let mut gain0: Vec<i32> = vec![0; n];
    let mut cands: Vec<Node> = Vec::new();
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    for v in 0..n as Node {
        if !st.is_free(v) || lock[v as usize] {
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
            let plus_leger = cfg.tie_host && g == bg && bb != u16::MAX
                && st.block_size[b as usize] < st.block_size[bb as usize];
            if g > bg || plus_leger {
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
        return 0;
    }

    let mut prio: Vec<u32> = vec![u32::MAX; n];
    if !chain_mode {
        let mut ord: Vec<Node> = cands.clone();
        ord.sort_unstable_by(|&a, &b| {
            gain0[b as usize]
                .cmp(&gain0[a as usize])
                .then(a.cmp(&b))
        });
        for (rank, &v) in ord.iter().enumerate() {
            prio[v as usize] = rank as u32;
        }
    } else {
        let mut par_source: Vec<Vec<Node>> = vec![Vec::new(); st.k];
        for &v in cands.iter() {
            par_source[st.part[v as usize] as usize].push(v);
        }
        for l in par_source.iter_mut() {
            l.sort_unstable_by(|&a, &b| {
                gain0[b as usize].cmp(&gain0[a as usize]).then(a.cmp(&b))
            });
        }
        let mut tete: Vec<usize> = vec![0; st.k];
        let mut pris = vec![false; n];
        macro_rules! dispo {
            ($b:expr) => {{
                let bb = $b;
                while tete[bb] < par_source[bb].len() && pris[par_source[bb][tete[bb]] as usize] {
                    tete[bb] += 1;
                }
                if tete[bb] < par_source[bb].len() { Some(par_source[bb][tete[bb]]) } else { None }
            }};
        }
        let mut depart: Vec<Node> = cands.clone();
        depart.sort_unstable_by(|&a, &b| {
            gain0[b as usize].cmp(&gain0[a as usize]).then(a.cmp(&b))
        });
        let mut chaines: Vec<(i64, Vec<Node>)> = Vec::new();
        for &d in depart.iter() {
            if pris[d as usize] { continue; }
            let mut c: Vec<Node> = Vec::new();
            let mut g = 0i64;
            let mut x = d;
            loop {
                pris[x as usize] = true;
                g += gain0[x as usize] as i64;
                c.push(x);
                let t = target[x as usize] as usize;
                match dispo!(t) {
                    Some(m) if !pris[m as usize] => x = m,
                    _ => break,
                }
                if c.len() >= 64 { break; }
            }
            chaines.push((g, c));
        }
        chaines.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1[0].cmp(&b.1[0])));
        let mut rank = 0u32;
        for (_, c) in chaines.iter() {
            for &v in c.iter() {
                prio[v as usize] = rank;
                rank += 1;
            }
        }
    }

    {
        let mut ordre_pre: Vec<Node> = cands.clone();
        ordre_pre.sort_unstable_by_key(|&v| prio[v as usize]);
        let mut taille: Vec<i64> = st.block_size.iter().map(|&x| x as i64).collect();
        let cap = (st.max_block_size + cfg.over) as i64;
        let mut gardes: Vec<Node> = Vec::with_capacity(ordre_pre.len());
        for &v in ordre_pre.iter() {
            let from = st.part[v as usize] as usize;
            let to = target[v as usize] as usize;
            let w = st.weight[v as usize] as i64;
            if taille[to] + w > cap || taille[from] <= w {
                target[v as usize] = u16::MAX;   // discarded
                continue;
            }
            taille[to] += w;
            taille[from] -= w;
            gardes.push(v);
        }
        cands = gardes;
        if cands.is_empty() { return 0; }
    }

    let mut rg: Vec<i32> = vec![0; n];
    let mut moving: Vec<Node> = Vec::with_capacity(64);
    let iso_host = cfg.iso_host;
    let mut phi: Vec<i32> = vec![0; st.k];
    let mut phi_ep: Vec<u32> = vec![0; st.k];
    let mut epoch: u32 = 0;
    macro_rules! touch {
        ($p:expr) => {{
            let __p = $p as usize;
            if iso_host && phi_ep[__p] != epoch { phi_ep[__p] = epoch; phi[__p] = 0; }
            __p
        }};
    }
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
        if iso_host {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 { for s in phi_ep.iter_mut() { *s = 0; } epoch = 1; }
        } else {
            phi.clear();
            phi.resize(st.k, 0);
        }
        for i in lo..hi {
            let p = touch!(st.part[st.he_nodes[i] as usize]);
            phi[p] += 1;
        }
        moving.sort_unstable_by_key(|&v| prio[v as usize]);
        for &v in moving.iter() {
            let from = st.part[v as usize] as usize;
            let to = target[v as usize] as usize;
            let from = touch!(from);
            phi[from] -= 1;
            if phi[from] == 0 {
                rg[v as usize] += 1;
            }
            let to = touch!(to);
            phi[to] += 1;
            if phi[to] == 1 {
                rg[v as usize] -= 1;
            }
        }
    }

    let mut ordre: Vec<Node> = cands
        .iter()
        .copied()
        .filter(|&v| if cfg.accept_zero { rg[v as usize] >= 0 } else { rg[v as usize] > 0 })
        .collect();
    ordre.sort_unstable_by_key(|&v| prio[v as usize]);

    let before = st.km1();
    for &v in ordre.iter() {
        let to = target[v as usize];
        if st.part[v as usize] == to {
            continue;
        }
        let cap = st.max_block_size + cfg.over;
        if st.block_size[to as usize] + st.weight[v as usize] > cap {
            continue;
        }
        if st.block_size[st.part[v as usize] as usize] <= st.weight[v as usize] {
            continue;
        }
        st.apply(v, to);
        bouges[v as usize] = true;
    }
    before as i64 - st.km1() as i64
}

pub fn refine(st: &mut State, cfg: &Config) -> i64 {
    let mut total = 0i64;
    let mut lock = vec![false; st.n];
    for r in 0..cfg.rounds {
        let mut bouges = vec![false; st.n];
        let before = st.km1();
        round(st, cfg, &mut lock, &mut bouges);
        lock.copy_from_slice(&bouges);
        if cfg.unconstrained {
            rebalance(st);
        }
        let g = before as i64 - st.km1() as i64;
        total += g;
        if g == 0 {
            break;
        }
    }
    total
}

pub fn rebalance(st: &mut State) -> i64 {
    let k = st.k;
    let l_max = st.max_block_size as i64;
    let dz: i64 = 1;   // dead zone of one node
    let z = l_max - dz;
    debug_assert!(k as i64 * z >= st.n as i64, "dead zone too wide, deadlock possible");

    let before = st.km1();
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    for _round in 0..(k + 1) {
        let mut surplus: Vec<i64> = (0..k)
            .map(|p| (st.block_size[p] as i64 - l_max).max(0))
            .collect();
        if surplus.iter().all(|&s| s == 0) { break; }

        let sous: Vec<Block> = (0..k)
            .filter(|&p| (st.block_size[p] as i64) < z)
            .map(|p| p as Block)
            .collect();
        if sous.is_empty() { break; }

        let mut cand: Vec<(i32, Node, Block)> = Vec::new();
        for v in 0..st.n as Node {
            let p = st.part[v as usize] as usize;
            if surplus[p] == 0 || !st.is_free(v) { continue; }
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
                    let to = sous[(v as usize) % sous.len()];
                    if to as usize == p { continue; }
                    (st.gain_cached(v, to), to)
                }
            };
            cand.push((g, v, to));
        }
        if cand.is_empty() { break; }

        cand.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

        let mut reste = surplus.clone();
        for &(_, v, to0) in cand.iter() {
            let p = st.part[v as usize] as usize;
            if reste[p] == 0 { continue; }
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
                    None => match sous.iter().find(|&&b| (st.block_size[b as usize] as i64) < z
                                                        && b as usize != p) {
                        Some(&b) => to = b,
                        None => continue,
                    },
                }
            }
            if st.block_size[p] <= st.weight[v as usize] { continue; }
            st.apply(v, to);
            reste[p] -= 1;
            surplus[p] -= 1;
        }
    }
    before as i64 - st.km1() as i64
}

pub fn cycles(st: &mut State, tours: usize, iso_host: bool) -> i64 {
    let k = st.k;
    let total_before = st.km1();
    let mut adj: Vec<Block> = Vec::with_capacity(64);

    for _ in 0..tours {
        let m_par_paire: usize = 1;  // one candidate per block pair
        let (best_v, best_g): (Vec<u32>, Vec<i32>) = if iso_host {
            let mut bv: Vec<u32> = vec![u32::MAX; k * k];
            let mut bg: Vec<i32> = vec![i32::MIN; k * k];
            for v in 0..st.n as Node {
                if !st.is_free(v) { continue; }
                let x = st.part[v as usize] as usize;
                if st.block_size[x] <= st.weight[v as usize] { continue; }
                st.adjacent_blocks(v, &mut adj);
                for &y in adj.iter() {
                    if y as usize == x { continue; }
                    let g = st.gain_cached(v, y);
                    let i = x * k + y as usize;
                    if bv[i] == u32::MAX || g > bg[i] || (g == bg[i] && v < bv[i]) {
                        bg[i] = g;
                        bv[i] = v;
                    }
                }
            }
            (bv, bg)
        } else {
            let mut pool: Vec<Vec<(i32, Node)>> = vec![Vec::new(); k * k];
            for v in 0..st.n as Node {
                if !st.is_free(v) { continue; }
                let x = st.part[v as usize] as usize;
                if st.block_size[x] <= st.weight[v as usize] { continue; }
                st.adjacent_blocks(v, &mut adj);
                for &y in adj.iter() {
                    if y as usize == x { continue; }
                    let g = st.gain_cached(v, y);
                    let p = &mut pool[x * k + y as usize];
                    p.push((g, v));
                    if p.len() > 8 * m_par_paire {
                        p.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                        p.truncate(m_par_paire);
                    }
                }
            }
            for p in pool.iter_mut() {
                p.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                p.truncate(m_par_paire);
            }
            (pool.iter().map(|p| p.first().map_or(u32::MAX, |x| x.1)).collect(),
             pool.iter().map(|p| p.first().map_or(i32::MIN, |x| x.0)).collect())
        };

        let mut trouve = 0usize;
        let mut utilise = vec![false; st.n];
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
            if utilise[v1 as usize] || utilise[v2 as usize] || utilise[v3 as usize] { continue; }
            if v1 == v2 || v2 == v3 || v1 == v3 { continue; }
            let mut somme = 0i64;
            somme += st.apply(v1, y as Block) as i64;
            somme += st.apply(v2, z as Block) as i64;
            somme += st.apply(v3, x as Block) as i64;
            if somme <= 0 {
                st.apply(v1, x as Block);
                st.apply(v2, y as Block);
                st.apply(v3, z as Block);
            } else {
                utilise[v1 as usize] = true;
                utilise[v2 as usize] = true;
                utilise[v3 as usize] = true;
                trouve += 1;
            }
        }
        if trouve == 0 { break; }
    }
    total_before as i64 - st.km1() as i64
}
