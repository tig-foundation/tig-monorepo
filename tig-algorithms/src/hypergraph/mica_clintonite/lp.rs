// Label propagation.

use super::gain::{Block, Node, State};

const ROUNDS: usize = 5;
const MAX_REBALANCE_MOVES: usize = 100_000;

fn rebalance(st: &mut State) {
    let mut adj: Vec<Block> = Vec::with_capacity(64);
    let mut moves = 0;
    while st.block_size.iter().any(|&s| s > st.max_block_size) && moves < MAX_REBALANCE_MOVES {
        moves += 1;
        let src = (0..st.k).max_by_key(|&b| st.block_size[b]).unwrap() as Block;
        let mut best: Option<(i32, Node, Block)> = None;
        for v in 0..st.n as Node {
            if st.part[v as usize] != src {
                continue;
            }
            st.adjacent_blocks(v, &mut adj);
            for &b in adj.iter() {
                if b == src || !st.fits_node(v, b) {
                    continue;
                }
                let g = st.gain_cached(v, b);
                if best.map_or(true, |(bg, bv, _)| g > bg || (g == bg && v < bv)) {
                    best = Some((g, v, b));
                }
            }
        }
        match best {
            Some((_, v, b)) => {
                st.apply(v, b);
            }
            None => break,
        }
    }
}

pub fn refine(st: &mut State) {
    let cap = st.max_block_size;
    for round in 0..ROUNDS {
        let strict = round + 1 == ROUNDS;
        let mut moved = 0usize;

        for v in 0..st.n as Node {
            let vi = v as usize;
            let (from, w) = unsafe {
                (
                    *st.part.get_unchecked(vi) as usize,
                    *st.weight.get_unchecked(vi),
                )
            };
            if unsafe { *st.block_size.get_unchecked(from) } <= w {
                continue;
            }
            let mut m = st.adj_mask(v) & !(1u64 << from);
            let mut best = 0u64;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                m &= m - 1;
                let sz = unsafe { *st.block_size.get_unchecked(b) };
                let fits = sz + w <= cap;
                let over = (sz + w).wrapping_sub(cap);
                let g =
                    st.gain_off(v, b) - ((over.wrapping_add(1) >> 1) as i32 & -((!fits) as i32));
                let keep = (fits | !strict) & (g > 0);
                let key = ((g as u64) << 6) | (63 - b as u64);
                best = best.max(key & (keep as u64).wrapping_neg());
            }
            if best != 0 {
                st.apply(v, (63 - (best & 63)) as Block);
                moved += 1;
            }
        }
        if moved == 0 {
            break;
        }
    }

    rebalance(st);
}
