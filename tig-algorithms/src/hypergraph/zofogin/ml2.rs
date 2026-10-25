// Multilevel refinement.

use super::fm;
use super::gain::{Block, Node, State};
use super::params::Ml2Defaults;
use super::{shuffle, Hp, SPLITMIX_GAMMA};

const MAX_RATED_EDGE: usize = 64;

pub struct Config {
    pub passes: usize,
    pub levels: usize,
    pub cmax: u32,
    pub coarse_rounds: usize,
    pub fine_rounds: usize,
    pub cycles: usize,
    pub cycles_last: bool,
    pub min_gain: i32,
    pub fm_steps: usize,
    pub fm_seeds: usize,
    pub fm_stop: f64,
}

impl Config {
    pub fn read(hp: &Hp<'_>, d: Option<&Ml2Defaults>) -> Option<Self> {
        let d = d?;
        if hp.int("ml2", 0, 1, 1) == 0 {
            return None;
        }
        Some(Config {
            passes: hp.int("ml2_passes", 1, 16, d.passes) as usize,
            levels: hp.int("ml2_levels", 1, 8, d.levels) as usize,
            cmax: hp.int("ml2_cmax", 2, 4096, d.cmax) as u32,
            coarse_rounds: hp.int("ml2_rounds", 1, 20, d.rounds) as usize,
            fine_rounds: hp.int("ml2_fine", 0, 20, 0) as usize,
            cycles: hp.int("ml2_cyc", 0, 64, d.cycles) as usize,
            cycles_last: hp.int("ml2_cyc_last", 0, 1, d.cycles_last) == 1,
            min_gain: hp.int("ml2_gain", -100_000, 64, 0) as i32,
            fm_steps: hp.int("ml2_steps", 10, 100_000, d.steps) as usize,
            fm_seeds: hp.int("ml2_seeds", 1, 100_000, d.seeds) as usize,
            fm_stop: hp.int("ml2_stop", 1, 400, 25) as f64 / 100.0,
        })
    }
}

fn contract(st: &State, salt: u64, cmax: u32) -> Option<(State, Vec<Node>)> {
    let n = st.n;
    if n < 2 {
        return None;
    }
    let mut order: Vec<Node> = (0..n as Node).collect();
    shuffle(&mut order, salt);
    let mut cid: Vec<u32> = (0..n as u32).collect();
    let mut csize = vec![1u32; n];
    let mut cw: Vec<u32> = st.weight.clone();
    let mut score = vec![0.0f64; n];
    let mut touched: Vec<u32> = vec![0; n + 1];
    for &v in &order {
        let vi = v as usize;
        if cid[vi] != v || csize[vi] > 1 {
            continue;
        }
        let pv = st.part[vi];
        let wv = st.weight[vi];
        let mut tl = 0usize;
        unsafe {
            let tp = touched.as_mut_ptr();
            let sp = score.as_mut_ptr();
            let cp = cid.as_ptr();
            let pp = st.part.as_ptr();
            let e0 = *st.nd_off.get_unchecked(vi) as usize;
            let e1 = *st.nd_off.get_unchecked(vi + 1) as usize;
            for ei in e0..e1 {
                let e = *st.nd_edges.get_unchecked(ei) as usize;
                let lo = *st.he_off.get_unchecked(e) as usize;
                let hi = *st.he_off.get_unchecked(e + 1) as usize;
                if hi - lo < 2 || hi - lo > MAX_RATED_EDGE {
                    continue;
                }
                let rb = (1.0 / (hi - lo - 1) as f64).to_bits();
                for i in lo..hi {
                    let u = *st.he_nodes.get_unchecked(i);
                    let ui = u as usize;
                    let ok = (u != v) & (*pp.add(ui) == pv);
                    let c = *cp.add(ui) as usize;
                    let sc = *sp.add(c);
                    *tp.add(tl) = c as u32;
                    tl += (ok & (sc == 0.0)) as usize;
                    *sp.add(c) = sc + f64::from_bits(rb & (ok as u64).wrapping_neg());
                }
            }
        }
        let mut best = 0u128;
        for t in 0..tl {
            let c = unsafe { *touched.get_unchecked(t) };
            let ci = c as usize;
            let (s, w) = unsafe { (*score.get_unchecked(ci), *cw.get_unchecked(ci)) };
            unsafe { *score.get_unchecked_mut(ci) = 0.0 };
            let key = ((s.to_bits() as u128) << 64) | (((!w) as u128) << 32) | (!c) as u128;
            best = best.max(key & ((w + wv <= cmax) as u128).wrapping_neg());
        }
        let best: Option<Node> = if best == 0 {
            None
        } else {
            Some(!(best as u32))
        };
        if let Some(c) = best {
            let ci = c as usize;
            cid[vi] = c;
            csize[ci] += 1;
            csize[vi] -= 1;
            cw[ci] += wv;
            cw[vi] -= wv;
        }
    }
    let mut rep = vec![Node::MAX; n];
    let mut map = vec![Node::MAX; n];
    let mut weights: Vec<u32> = Vec::new();
    let mut parts: Vec<Block> = Vec::new();
    for v in 0..n {
        let c = cid[v] as usize;
        if rep[c] == Node::MAX {
            rep[c] = weights.len() as Node;
            weights.push(0);
            parts.push(st.part[v]);
        }
        map[v] = rep[c];
        weights[rep[c] as usize] += st.weight[v];
    }
    let nc = weights.len();
    if nc as u64 * 100 > n as u64 * 95 {
        return None;
    }
    let mut offsets = vec![0u32];
    let mut nodes: Vec<Node> = vec![0; st.he_nodes.len()];
    let mut seen = vec![u32::MAX; nc];
    let mut nl = 0usize;
    for e in 0..st.m {
        let start = nl;
        unsafe {
            let lo = *st.he_off.get_unchecked(e) as usize;
            let hi = *st.he_off.get_unchecked(e + 1) as usize;
            for i in lo..hi {
                let c = *map.get_unchecked(*st.he_nodes.get_unchecked(i) as usize);
                let fresh = *seen.get_unchecked(c as usize) != e as u32;
                *seen.get_unchecked_mut(c as usize) = e as u32;
                *nodes.get_unchecked_mut(nl) = c;
                nl += fresh as usize;
            }
        }
        if nl - start < 2 {
            nl = start;
        } else {
            offsets.push(nl as u32);
        }
    }
    nodes.truncate(nl);
    let coarse = State::new(
        nc,
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

fn descend(st: &mut State, cfg: &Config, depth: usize, pass: usize) {
    let mut seeds: Option<Vec<Node>> = None;
    let rounds = if depth == 0 {
        cfg.fine_rounds
    } else {
        cfg.coarse_rounds
    };
    if depth < cfg.levels {
        let salt = (pass as u64)
            .wrapping_mul(SPLITMIX_GAMMA)
            .wrapping_add(depth as u64)
            .wrapping_add(1);
        if let Some((mut coarse, map)) = contract(st, salt, cfg.cmax) {
            descend(&mut coarse, cfg, depth + 1, pass);
            let mut active = vec![false; st.n];
            for v in 0..st.n {
                let to = coarse.part[map[v] as usize];
                if to == st.part[v] {
                    continue;
                }
                st.apply(v as Node, to);
                if rounds == 0 {
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
            if rounds != 0 {
                seeds = Some(
                    active
                        .iter()
                        .enumerate()
                        .filter_map(|(v, &yes)| if yes { Some(v as Node) } else { None })
                        .collect(),
                );
            }
        }
    }
    if rounds == 0 {
        return;
    }
    let local_cfg = fm::Config {
        rounds,
        max_steps: cfg.fm_steps,
        seeds_per_search: cfg.fm_seeds,
        stop_factor: cfg.fm_stop,
    };
    fm::refine_filtered(st, &local_cfg, seeds.as_deref(), cfg.min_gain);
}

pub fn refine(st: &mut State, cfg: &Config, pass: usize) {
    let before = st.km1();
    let original = st.part.clone();
    descend(st, cfg, 0, pass);
    let rejected = st
        .block_size
        .iter()
        .any(|&s| s == 0 || s > st.max_block_size)
        || st.km1() > before;
    if rejected {
        *st = st.with_partition(original);
    }
}
