// Label propagation: each node joins the adjacent block of highest gain, an overshoot of the
// block-size cap being penalised until the last round, which is strict; then a greedy rebalance.

use super::gain::{Block, Node, State};

const ROUNDS: usize = 5;
// Gain penalty per unit of block-size overshoot in the non-strict rounds.
const PENALTY: f64 = 0.5;

// Moves the best-gain node of the largest oversized block to an adjacent block with room, while possible.
fn rebalance(st: &mut State) {
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    let mut guard = 0;
    while st.block_size.iter().any(|&s| s > st.max_block_size) && guard < 100000 {
        guard += 1;
        let src = (0..st.k).max_by_key(|&b| st.block_size[b]).unwrap() as Block;
        let mut best: Option<(i32, Node, Block)> = None;
        for v in 0..st.n as Node {
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
            Some((_, v, b)) => { st.apply(v, b); }
            None => break,
        }
    }
}

pub fn refine(st: &mut State) {
    let mut adj: Vec<Block> = Vec::with_capacity(64);

    for round in 0..ROUNDS {
        let strict = round + 1 == ROUNDS;
        let mut moved = 0usize;

        for v in 0..st.n as Node {
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
                    g -= (PENALTY * over as f64).ceil() as i32;
                }
                if g > bg {
                    bg = g;
                    bb = Some(b);
                }
            }
            if let Some(to) = bb {
                st.apply(v, to);
                moved += 1;
            }
        }
        if moved == 0 {
            break;
        }
    }

    rebalance(st);
}
