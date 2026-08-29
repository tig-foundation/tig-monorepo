// Label propagation.

use super::gain::{Block, Node, State};

pub struct LpConfig {
    pub tie_host: bool,
    pub rounds: usize,
    pub unconstrained: bool,
    pub penalty: f64,
}

impl Default for LpConfig {
    fn default() -> Self {
        LpConfig { tie_host: false, rounds: 5, unconstrained: true, penalty: 0.5 }
    }
}

pub fn rebalance(st: &mut State) -> i64 {
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    let mut total = 0i64;
    let mut guard = 0;
    while st.block_size.iter().any(|&s| s > st.max_block_size) && guard < 100000 {
        guard += 1;
        let src = (0..st.k).max_by_key(|&b| st.block_size[b]).unwrap() as Block;
        let mut best: Option<(i32, Node, Block)> = None;
        for v in 0..st.n as Node {
            if !st.is_free(v) { continue; }
            if st.part[v as usize] != src { continue; }
            st.adjacent_blocks(v, &mut adj);
            for &b in adj.iter() {
                if b == src || !st.fits_node(v, b) { continue; }
                let g = st.gain_cached(v, b);
                if best.map_or(true, |(bg, bv, _)| g > bg || (g == bg && v < bv)) {
                    best = Some((g, v, b));
                }
            }
        }
        match best {
            Some((_, v, b)) => { total += st.apply(v, b) as i64; }
            None => break,
        }
    }
    total
}

pub fn refine(st: &mut State, cfg: &LpConfig) -> i64 {
    let mut total = 0i64;
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    let mut order: Vec<Node> = (0..st.n as Node).collect();

    for round in 0..cfg.rounds {
        let strict = !cfg.unconstrained || round + 1 == cfg.rounds;
        let mut moved = 0usize;
        let mut round_gain = 0i64;

        for &v in &order {
            if !st.is_free(v) { continue; }
            let from = st.part[v as usize];
            if st.block_size[from as usize] <= st.weight[v as usize] {
                continue;
            }
            st.adjacent_blocks(v, &mut adj);
            let mut bg = 0i32;
            let mut bb: Option<Block> = None;
            for &b in adj.iter() {
                if b == from {
                    continue;
                }
                let fits = st.fits_node(v, b);
                if !fits && strict {
                    continue;
                }
                let mut g = st.gain_cached(v, b);
                if !fits {
                    let over = st.block_size[b as usize] + st.weight[v as usize]
                        - st.max_block_size;
                    g -= (cfg.penalty * over as f64).ceil() as i32;
                }
                let plus_leger = cfg.tie_host && g == bg
                    && bb.map_or(false, |x| st.block_size[b as usize] < st.block_size[x as usize]);
                let mieux = g > bg || plus_leger;
                if mieux {
                    bg = g;
                    bb = Some(b);
                }
            }
            if let Some(to) = bb {
                round_gain += st.apply(v, to) as i64;
                moved += 1;
            }
        }
        total += round_gain;
        if moved == 0 {
            break;
        }
    }

    total += rebalance(st);
    total
}
