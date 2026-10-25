use super::gain::{Block, Node, State};
use super::{fm, shuffle, Hp, SPLITMIX_GAMMA};

pub struct Config {
    pub levels: usize,
    pub fine: usize,
}

impl Config {
    pub fn read(hp: &Hp<'_>, default_tail: i64) -> Option<Self> {
        if hp.int("ml_tail", 0, 1, default_tail) == 0 {
            return None;
        }
        Some(Self {
            levels: hp.int("ml_levels", 1, 4, 2) as usize,
            fine: hp.int("ml_fine", 0, 5, 1) as usize,
        })
    }
}

fn contract(st: &State, salt: u64) -> Option<(State, Vec<Node>)> {
    if st.n < 2 {
        return None;
    }
    let mut order: Vec<Node> = (0..st.n as Node).collect();
    shuffle(&mut order, salt);
    let mut mate = vec![Node::MAX; st.n];
    let mut score = vec![0.0f64; st.n];
    let mut touched = Vec::new();
    let max_cluster = (st.max_block_size / 8).max(1);
    for &v in &order {
        let vi = v as usize;
        if mate[vi] != Node::MAX {
            continue;
        }
        touched.clear();
        for ei in st.nd_off[vi]..st.nd_off[vi + 1] {
            let e = st.nd_edges[ei as usize] as usize;
            let lo = st.he_off[e] as usize;
            let hi = st.he_off[e + 1] as usize;
            if hi - lo < 2 || hi - lo > 64 {
                continue;
            }
            let r = 1.0 / (hi - lo - 1) as f64;
            for &u in &st.he_nodes[lo..hi] {
                let ui = u as usize;
                if u == v
                    || mate[ui] != Node::MAX
                    || st.part[ui] != st.part[vi]
                    || st.weight[vi] as u64 + st.weight[ui] as u64 > max_cluster as u64
                {
                    continue;
                }
                if score[ui] == 0.0 {
                    touched.push(u);
                }
                score[ui] += r;
            }
        }
        let mut best: Option<(f64, u32, Node)> = None;
        for &u in &touched {
            let s = score[u as usize];
            let w = st.weight[u as usize];
            if best.map_or(true, |(bs, bw, bu)| {
                s > bs || (s == bs && (w < bw || (w == bw && u < bu)))
            }) {
                best = Some((s, w, u));
            }
            score[u as usize] = 0.0;
        }
        if let Some((_, _, u)) = best {
            mate[vi] = u;
            mate[u as usize] = v;
        }
    }
    let mut map = vec![Node::MAX; st.n];
    let mut weights = Vec::new();
    let mut parts = Vec::new();
    for v in 0..st.n {
        if map[v] != Node::MAX {
            continue;
        }
        let id = weights.len() as Node;
        map[v] = id;
        let mut w = st.weight[v];
        if mate[v] != Node::MAX {
            let u = mate[v] as usize;
            map[u] = id;
            w += st.weight[u];
        }
        weights.push(w);
        parts.push(st.part[v]);
    }
    let n = weights.len();
    if n as u64 * 100 > st.n as u64 * 95 {
        return None;
    }
    let mut offsets = vec![0u32];
    let mut nodes = Vec::with_capacity(st.he_nodes.len());
    let mut pins = Vec::new();
    for e in 0..st.m {
        pins.clear();
        for i in st.he_off[e]..st.he_off[e + 1] {
            pins.push(map[st.he_nodes[i as usize] as usize]);
        }
        pins.sort_unstable();
        pins.dedup();
        if pins.len() >= 2 {
            nodes.extend_from_slice(&pins);
            offsets.push(nodes.len() as u32);
        }
    }
    let coarse = State::new(
        n,
        offsets.len() - 1,
        st.k,
        offsets,
        nodes,
        parts,
        st.max_block_size,
        Some(weights),
    )?;
    Some((coarse, map))
}

fn descend(st: &mut State, cfg: &Config, fcfg: &fm::Config, depth: usize, pass: usize) {
    let rounds = if depth == 0 { cfg.fine } else { fcfg.rounds };
    let mut seeds = None;
    if depth < cfg.levels {
        let salt = (pass as u64)
            .wrapping_mul(SPLITMIX_GAMMA)
            .wrapping_add(depth as u64)
            .wrapping_add(1);
        if let Some((mut coarse, map)) = contract(st, salt) {
            descend(&mut coarse, cfg, fcfg, depth + 1, pass);
            let part: Vec<Block> = map.iter().map(|&v| coarse.part[v as usize]).collect();
            let mut active = vec![false; st.n];
            for v in 0..st.n {
                if part[v] == st.part[v] {
                    continue;
                }
                active[v] = true;
                for ei in st.nd_off[v]..st.nd_off[v + 1] {
                    let e = st.nd_edges[ei as usize] as usize;
                    let lo = st.he_off[e] as usize;
                    let hi = st.he_off[e + 1] as usize;
                    if hi - lo <= 64 {
                        for &u in &st.he_nodes[lo..hi] {
                            active[u as usize] = true;
                        }
                    }
                }
            }
            let local: Vec<Node> = active
                .iter()
                .enumerate()
                .filter_map(|(v, &yes)| if yes { Some(v as Node) } else { None })
                .collect();
            if part != st.part {
                *st = st.with_partition(part);
            }
            seeds = Some(local);
        }
    }
    let local_cfg = fm::Config {
        rounds,
        max_steps: fcfg.max_steps,
        seeds_per_search: fcfg.seeds_per_search,
        stop_factor: fcfg.stop_factor,
    };
    if local_cfg.rounds != 0 {
        match seeds {
            Some(ref local) => fm::refine_seeded(st, &local_cfg, local),
            None => fm::refine(st, &local_cfg),
        }
    }
}

pub fn refine(st: &mut State, cfg: &Config, fcfg: &fm::Config, pass: usize) {
    let before = st.km1() as i64;
    let original = st.part.clone();
    descend(st, cfg, fcfg, 0, pass);
    let rejected = st
        .block_size
        .iter()
        .any(|&s| s == 0 || s > st.max_block_size)
        || st.km1() as i64 > before;
    if rejected {
        *st = st.with_partition(original);
    }
}
