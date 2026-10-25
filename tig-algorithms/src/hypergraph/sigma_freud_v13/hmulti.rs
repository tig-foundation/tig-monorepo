use super::hrefine::{Block, State};

pub struct Coarsen {
    pub map: Vec<u32>,
    pub state: State,
}

pub struct Scratch {
    score: Vec<f32>,
    touched: Vec<u32>,
    mate: Vec<u32>,
    order: Vec<u32>,
    pins: Vec<u32>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            score: vec![0.0; n],
            touched: Vec::with_capacity(256),
            mate: vec![u32::MAX; n],
            order: Vec::with_capacity(n),
            pins: Vec::with_capacity(64),
        }
    }
}

#[inline]
fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub fn coarsen(st: &State, max_cluster: u32, edge_cap: usize, rng: &mut u64, sc: &mut Scratch) -> Option<Coarsen> {
    let n = st.n;
    sc.mate.clear();
    sc.mate.resize(n, u32::MAX);
    sc.order.clear();
    sc.order.extend(0..n as u32);
    for i in (1..n).rev() {
        let j = (splitmix(rng) % (i as u64 + 1)) as usize;
        sc.order.swap(i, j);
    }
    if sc.score.len() < n {
        sc.score.resize(n, 0.0);
    }

    for oi in 0..n {
        let v = sc.order[oi] as usize;
        if sc.mate[v] != u32::MAX {
            continue;
        }
        let bv = st.part[v];
        let wv = st.wt[v];
        sc.touched.clear();
        for &e in st.edges_of(v) {
            let pins = st.pins_of(e as usize);
            if pins.len() > edge_cap || pins.len() < 2 {
                continue;
            }
            let r = 1.0f32 / (pins.len() - 1) as f32;
            for &u in pins {
                let uu = u as usize;
                if uu == v || st.part[uu] != bv || sc.mate[uu] != u32::MAX || wv + st.wt[uu] > max_cluster {
                    continue;
                }
                if sc.score[uu] == 0.0 {
                    sc.touched.push(u);
                }
                sc.score[uu] += r;
            }
        }
        let mut best: Option<(f32, u32, u32)> = None;
        for &u in &sc.touched {
            let uu = u as usize;
            let s = sc.score[uu];
            let w = st.wt[uu];
            let better = match best {
                None => true,
                Some((bs, bw, bid)) => s > bs || (s == bs && (w < bw || (w == bw && u < bid))),
            };
            if better {
                best = Some((s, w, u));
            }
        }
        for &u in &sc.touched {
            sc.score[u as usize] = 0.0;
        }
        if let Some((_, _, u)) = best {
            sc.mate[v] = u;
            sc.mate[u as usize] = v as u32;
        }
    }

    let mut map = vec![u32::MAX; n];
    let mut wt_c: Vec<u32> = Vec::with_capacity(n);
    let mut part_c: Vec<Block> = Vec::with_capacity(n);
    for v in 0..n {
        if map[v] != u32::MAX {
            continue;
        }
        let id = wt_c.len() as u32;
        map[v] = id;
        let m = sc.mate[v];
        let mut w = st.wt[v];
        if m != u32::MAX {
            map[m as usize] = id;
            w += st.wt[m as usize];
        }
        wt_c.push(w);
        part_c.push(st.part[v]);
    }
    let nc = wt_c.len();
    if nc as f64 > 0.95 * n as f64 {
        return None;
    }

    let mut he_off: Vec<u32> = Vec::with_capacity(st.m + 1);
    let mut he_nodes: Vec<u32> = Vec::with_capacity(st.he_nodes.len());
    he_off.push(0);
    for e in 0..st.m {
        sc.pins.clear();
        for &v in st.pins_of(e) {
            sc.pins.push(map[v as usize]);
        }
        sc.pins.sort_unstable();
        sc.pins.dedup();
        if sc.pins.len() >= 2 {
            he_nodes.extend_from_slice(&sc.pins);
            he_off.push(he_nodes.len() as u32);
        }
    }
    let mc = he_off.len() - 1;
    let state = State::new(nc, mc, st.k, he_off, he_nodes, part_c, wt_c, st.max_block_size);
    Some(Coarsen { map, state })
}
