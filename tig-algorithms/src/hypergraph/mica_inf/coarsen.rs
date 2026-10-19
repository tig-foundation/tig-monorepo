// One level of contraction (pairs within a block), then Fiduccia-Mattheyses on the contracted
// hypergraph; the result is projected back only when it improves km1.

use super::gain::{Block, Node, State};

// Pairs vertices that share a hyperedge and a block, at most one partner each.
fn match_within_blocks(st: &State) -> (Vec<u32>, usize) {
    let n = st.n;
    let mut super_of = vec![u32::MAX; n];
    let mut count = 0u32;
    for u in 0..n {
        if super_of[u] != u32::MAX {
            continue;
        }
        let bu = st.part[u];
        let mut found = usize::MAX;
        'outer: for ei in st.nd_off[u] as usize..st.nd_off[u + 1] as usize {
            let e = st.nd_edges[ei] as usize;
            for pi in st.he_off[e] as usize..st.he_off[e + 1] as usize {
                let v = st.he_nodes[pi] as usize;
                if v != u && super_of[v] == u32::MAX && st.part[v] == bu {
                    found = v;
                    break 'outer;
                }
            }
        }
        super_of[u] = count;
        if found != usize::MAX {
            super_of[found] = count;
        }
        count += 1;
    }
    (super_of, count as usize)
}

// Contracts the hyperedges onto the super-nodes, dropping those left with fewer than two pins.
fn contract_edges(st: &State, super_of: &[u32], sn: usize) -> (Vec<u32>, Vec<Node>) {
    let mut off = Vec::with_capacity(st.m + 1);
    let mut pins: Vec<Node> = Vec::with_capacity(st.he_nodes.len());
    let mut seen = vec![u32::MAX; sn];
    off.push(0u32);
    for e in 0..st.m {
        let start = pins.len();
        for pi in st.he_off[e] as usize..st.he_off[e + 1] as usize {
            let s = super_of[st.he_nodes[pi] as usize];
            if seen[s as usize] != e as u32 {
                seen[s as usize] = e as u32;
                pins.push(s as Node);
            }
        }
        if pins.len() - start < 2 {
            pins.truncate(start);
        } else {
            off.push(pins.len() as u32);
        }
    }
    (off, pins)
}

// Returns the projected partition and its km1 when it beats `st` and stays feasible.
pub fn supernode_fm(st: &State, fcfg: &super::fm::Config) -> Option<(Vec<Block>, u64)> {
    if st.n < 2 {
        return None;
    }
    let (super_of, sn) = match_within_blocks(st);
    if sn == st.n {
        return None;
    }
    let (he_off, he_nodes) = contract_edges(st, &super_of, sn);
    let m = he_off.len() - 1;
    if m == 0 {
        return None;
    }

    let mut part = vec![0 as Block; sn];
    let mut weight = vec![0u32; sn];
    for v in 0..st.n {
        let s = super_of[v] as usize;
        part[s] = st.part[v];
        weight[s] += st.weight[v];
    }

    let mut cst = State::new(
        sn,
        m,
        st.k,
        he_off,
        he_nodes,
        part,
        st.max_block_size,
        Some(weight),
    );
    let before = cst.km1();
    super::fm::refine(&mut cst, fcfg);
    let after = cst.km1();
    if after >= before {
        return None;
    }
    if cst
        .block_size
        .iter()
        .any(|&s| s < 1 || s > cst.max_block_size)
    {
        return None;
    }
    let projected: Vec<Block> = (0..st.n).map(|v| cst.part[super_of[v] as usize]).collect();
    Some((projected, after))
}
