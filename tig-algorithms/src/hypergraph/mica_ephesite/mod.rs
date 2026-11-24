use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

pub(crate) fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

pub(crate) struct Hp<'a>(pub &'a Option<Map<String, Value>>);

impl Hp<'_> {
    fn raw(&self, name: &str) -> Option<i64> {
        self.0
            .as_ref()
            .and_then(|p| p.get(name))
            .and_then(Value::as_i64)
    }

    pub fn get(&self, name: &str, lo: i64, hi: i64) -> Option<i64> {
        self.raw(name).filter(|v| (lo..=hi).contains(v))
    }

    pub fn int(&self, name: &str, lo: i64, hi: i64, default: i64) -> i64 {
        self.get(name, lo, hi).unwrap_or(default)
    }

    pub fn choice(&self, name: &str, allowed: &[i64], default: i64) -> i64 {
        self.raw(name)
            .filter(|v| allowed.contains(v))
            .unwrap_or(default)
    }
}

const LZ: i64 = 1;

pub(crate) const FILTER_THREADS: u32 = 256;
pub(crate) const SELECT_THREADS: u32 = 1024;
pub(crate) const FINISH_THREADS: u32 = 256;

const TRI100_DEFAULT_ENABLED: i64 = 1;
const TRI100_DEFAULT_AT: i64 = 100;
const TRI100_DEFAULT_BAR: i64 = 280_425;
const TRI100_DEFAULT_MARGIN: i64 = 3_270;
const TRI100_DEFAULT_K: i64 = 10_632;
const TRI100_DEFAULT_LAG: i64 = 2;
const TRI100_DEFAULT_SURE: i64 = 1;
const TRI100_DEFAULT_HI: i64 = 9_000;
const TRI100_DEFAULT_RHO: i64 = 300;

const TRI200_DEFAULT_ENABLED: i64 = 1;
const TRI200_DEFAULT_AT: i64 = 100;
const TRI200_DEFAULT_BAR: i64 = 282_738;
const TRI200_DEFAULT_MARGIN: i64 = 8_042;
const TRI200_DEFAULT_K: i64 = 11_106;
const TRI200_DEFAULT_LAG: i64 = 2;
const TRI200_DEFAULT_SURE: i64 = 0;
const TRI200_DEFAULT_HI: i64 = 9_000;
const TRI200_DEFAULT_RHO: i64 = 650;

const GF_TRI_K: i64 = 9_997;
const GF_TRI_MARGIN: i64 = 3_270;
const GF_ROUNDS_PERMILLE: i64 = 1_250;
const GF_BATCH: i64 = 2_048;
const GF_TRI_SURE: i64 = 1;
const GF_TRI_HI: i64 = 8_000;
const GF_TRI_RHO: i64 = 300;
const GF_BAND: i64 = 1;
const GF_BAND_LO: i64 = -2_000;
const GF_BAND_HI: i64 = 3_000;
const GF_END_K: i64 = 9_428;
const GF_EXT: i64 = 200;
const GF_EXT_PASSES: i64 = 2;

const GM_ILS_ITERATIONS: usize = 5;
const GM_PASSES: usize = 1;
const GM_FM_ROUNDS: usize = 30;
const GM_PERMILLE: i64 = 900;
const GM_BATCH: i64 = 2_048;

pub(crate) const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
pub(crate) const LCG_MUL: u64 = 6_364_136_223_846_793_005;
pub(crate) const LCG_INC: u64 = 1_442_695_040_888_963_407;

pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub(crate) fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    for i in (1..items.len()).rev() {
        items.swap(i, (splitmix64(&mut state) % (i as u64 + 1)) as usize);
    }
}

pub(crate) fn feasible_km1(
    he_off: &[i32],
    he_nodes: &[i32],
    part: &[u32],
    num_parts: u32,
    max_part_size: u32,
) -> Option<u64> {
    let np = num_parts as usize;
    if np == 0 || part.is_empty() {
        return None;
    }
    let mut sizes = vec![0u32; np];
    for &b in part.iter() {
        if b as usize >= np {
            return None;
        }
        sizes[b as usize] += 1;
    }
    if sizes.iter().any(|&s| s < 1 || s > max_part_size) {
        return None;
    }
    let m = he_off.len().saturating_sub(1);
    let mut stamp = vec![0u32; np];
    let mut km1 = 0u64;
    for e in 0..m {
        let lo = he_off[e].max(0) as usize;
        let hi = (he_off[e + 1].max(0) as usize).min(he_nodes.len());
        let tag = e as u32 + 1;
        let mut touched = 0u64;
        for k in lo..hi {
            let v = he_nodes[k];
            if v < 0 || v as usize >= part.len() {
                return None;
            }
            let b = part[v as usize] as usize;
            if stamp[b] != tag {
                stamp[b] = tag;
                touched += 1;
            }
        }
        km1 += touched.saturating_sub(1);
    }
    Some(km1)
}

pub(crate) struct Saver<'a> {
    he_off: &'a [i32],
    he_nodes: &'a [i32],
    num_parts: u32,
    max_part_size: u32,
    save_solution: &'a dyn Fn(&Solution) -> anyhow::Result<()>,
    best_km1: u64,
    best_partition: Option<Vec<u32>>,
}

impl<'a> Saver<'a> {
    pub fn new(
        challenge: &Challenge,
        he_off: &'a [i32],
        he_nodes: &'a [i32],
        save_solution: &'a dyn Fn(&Solution) -> anyhow::Result<()>,
    ) -> Self {
        let round_robin = round_robin(challenge);
        let best_km1 = feasible_km1(
            he_off,
            he_nodes,
            &round_robin,
            challenge.num_parts,
            challenge.max_part_size,
        )
        .unwrap_or(u64::MAX);
        Saver {
            he_off,
            he_nodes,
            num_parts: challenge.num_parts,
            max_part_size: challenge.max_part_size,
            save_solution,
            best_km1,
            best_partition: None,
        }
    }

    pub fn save(&mut self, partition: &[u32]) -> anyhow::Result<()> {
        self.save_km1(partition).map(|_| ())
    }

    pub fn save_km1(&mut self, partition: &[u32]) -> anyhow::Result<Option<u64>> {
        let km1 = feasible_km1(
            self.he_off,
            self.he_nodes,
            partition,
            self.num_parts,
            self.max_part_size,
        );
        if let Some(k) = km1 {
            if k < self.best_km1 {
                self.best_km1 = k;
                let solution = Solution {
                    partition: partition.to_vec(),
                };
                (self.save_solution)(&solution)?;
                if self.best_partition.is_some() {
                    self.best_partition = Some(solution.partition);
                }
            }
        }
        Ok(km1)
    }
}

pub(crate) fn to_u32(partition: &[i32]) -> Vec<u32> {
    partition.iter().map(|&x| x as u32).collect()
}

fn round_robin(challenge: &Challenge) -> Vec<u32> {
    (0..challenge.num_nodes)
        .map(|i| i % challenge.num_parts)
        .collect()
}

#[inline]
pub(crate) fn decode_pick(w: u64) -> (i32, i32) {
    ((w >> 32) as i32, (w as u32 as i32) & 63)
}

#[inline]
pub(crate) fn decode_sorted(w: u64) -> (i32, i32) {
    (w as u32 as i32, (!((w >> 32) as u32) as i32) & 63)
}

mod coarsen {
use super::gain::{Block, Node, State};

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

pub fn supernode_fm<const FAST: bool>(st: &State, fcfg: &super::fm::Config) -> Option<(Vec<Block>, u64)> {
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
    )?;
    let before = cst.km1();
    super::fm::refine::<FAST>(&mut cst, fcfg);
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
}
mod fm;
mod gain;
mod grouped;
mod grouped_small;
mod infer;
mod jet;
mod lp {
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
}
mod ml {
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

fn descend<const FAST: bool>(st: &mut State, cfg: &Config, fcfg: &fm::Config, depth: usize, pass: usize) {
    let rounds = if depth == 0 { cfg.fine } else { fcfg.rounds };
    let mut seeds = None;
    if depth < cfg.levels {
        let salt = (pass as u64)
            .wrapping_mul(SPLITMIX_GAMMA)
            .wrapping_add(depth as u64)
            .wrapping_add(1);
        if let Some((mut coarse, map)) = contract(st, salt) {
            descend::<FAST>(&mut coarse, cfg, fcfg, depth + 1, pass);
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
            Some(ref local) => fm::refine_seeded::<FAST>(st, &local_cfg, local),
            None => fm::refine::<FAST>(st, &local_cfg),
        }
    }
}

pub fn refine<const FAST: bool>(st: &mut State, cfg: &Config, fcfg: &fm::Config, pass: usize) {
    let before = st.km1() as i64;
    let original = st.part.clone();
    descend::<FAST>(st, cfg, fcfg, 0, pass);
    let rejected = st
        .block_size
        .iter()
        .any(|&s| s == 0 || s > st.max_block_size)
        || st.km1() as i64 > before;
    if rejected {
        *st = st.with_partition(original);
    }
}
}
mod ml2 {
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
            fine_rounds: hp.int("ml2_fine", 0, 20, d.fine) as usize,
            cycles: hp.int("ml2_cyc", 0, 64, d.cycles) as usize,
            min_gain: hp.int("ml2_gain", -100_000, 64, 0) as i32,
            fm_steps: hp.int("ml2_steps", 10, 100_000, d.steps) as usize,
            fm_seeds: hp.int("ml2_seeds", 1, 100_000, d.seeds) as usize,
            fm_stop: hp.int("ml2_stop", 1, 400, 25) as f64 / 100.0,
        })
    }
}

pub(super) fn contract<const FAST: bool>(st: &State, salt: u64, cmax: u32,
) -> Option<(State, Vec<Node>)> {
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
                if FAST {
                    let mut rate = |i: usize| {
                        let u = *st.he_nodes.get_unchecked(i);
                        let ui = u as usize;
                        let ok = (u != v) & (*pp.add(ui) == pv);
                        let c = *cp.add(ui) as usize;
                        let sc = *sp.add(c);
                        *tp.add(tl) = c as u32;
                        tl += (ok & (sc == 0.0)) as usize;
                        *sp.add(c) = sc + f64::from_bits(rb & (ok as u64).wrapping_neg());
                    };
                    let mut i = lo;
                    while hi - i >= 4 {
                        rate(i);
                        rate(i + 1);
                        rate(i + 2);
                        rate(i + 3);
                        i += 4;
                    }
                    while i < hi {
                        rate(i);
                        i += 1;
                    }
                } else {
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
            if FAST {
                let mut append = |i: usize| {
                    let c = *map.get_unchecked(*st.he_nodes.get_unchecked(i) as usize);
                    let fresh = *seen.get_unchecked(c as usize) != e as u32;
                    *seen.get_unchecked_mut(c as usize) = e as u32;
                    *nodes.get_unchecked_mut(nl) = c;
                    nl += fresh as usize;
                };
                let mut i = lo;
                while hi - i >= 4 {
                    append(i);
                    append(i + 1);
                    append(i + 2);
                    append(i + 3);
                    i += 4;
                }
                while i < hi {
                    append(i);
                    i += 1;
                }
            } else {
                for i in lo..hi {
                    let c = *map.get_unchecked(*st.he_nodes.get_unchecked(i) as usize);
                    let fresh = *seen.get_unchecked(c as usize) != e as u32;
                    *seen.get_unchecked_mut(c as usize) = e as u32;
                    *nodes.get_unchecked_mut(nl) = c;
                    nl += fresh as usize;
                }
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
    debug_assert_eq!(
        coarse.weight.iter().map(|&w| u64::from(w)).sum::<u64>(),
        st.weight.iter().map(|&w| u64::from(w)).sum::<u64>(),
    );
    debug_assert!(map.iter().enumerate().all(|(v, &c)| coarse.part[c as usize] == st.part[v]));
    debug_assert_eq!(coarse.km1(), st.km1());
    Some((coarse, map))
}

fn descend<const FAST: bool>(st: &mut State, cfg: &Config, depth: usize, pass: usize) {
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
        if let Some((mut coarse, map)) = contract::<FAST>(st, salt, cfg.cmax) {
            descend::<FAST>(&mut coarse, cfg, depth + 1, pass);
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
    fm::refine_filtered::<FAST>(st, &local_cfg, seeds.as_deref(), cfg.min_gain);
}

pub fn refine<const FAST: bool>(st: &mut State, cfg: &Config, pass: usize) {
    let before = st.km1();
    let original = st.part.clone();
    descend::<FAST>(st, cfg, 0, pass);
    let rejected = st
        .block_size
        .iter()
        .any(|&s| s == 0 || s > st.max_block_size)
        || st.km1() > before;
    if rejected {
        *st = st.with_partition(original);
    }
}
}
mod params;
mod refine {
use super::gain::{Block, State};

pub fn improve<const FAST: bool>(
    n: usize,
    m: usize,
    he_off: Vec<u32>,
    he_nodes: Vec<u32>,
    k: usize,
    max_part_size: u32,
    partition: &[u32],
    passes: usize,
    fm_rounds: usize,
    fm_steps: usize,
    fm_seeds: usize,
    fm_stop: f64,
    jet_rounds: usize,
    cyc_rounds: usize,
    ml: Option<super::ml::Config>,
    ml2: Option<super::ml2::Config>,
    local_only: bool,
    jkeep: bool,
) -> Vec<u32> {
    if n == 0 || m == 0 || k == 0 || partition.len() != n {
        return partition.to_vec();
    }
    if he_off.len() != m + 1 || he_off[m] as usize != he_nodes.len() {
        return partition.to_vec();
    }
    if partition.iter().any(|&b| b as usize >= k) {
        return partition.to_vec();
    }
    if he_nodes.iter().any(|&v| v as usize >= n) {
        return partition.to_vec();
    }
    if (k as u64) * (max_part_size as u64) < n as u64 {
        return partition.to_vec();
    }

    let part: Vec<Block> = partition.iter().map(|&x| x as Block).collect();
    let Some(mut st) = State::new(n, m, k, he_off, he_nodes, part, max_part_size, None) else {
        return partition.to_vec();
    };

    let mut best_km1 = st.km1();
    let mut best_part = st.part.clone();
    macro_rules! keep {
        () => {{
            let cur = st.km1();
            let feasible = st
                .block_size
                .iter()
                .all(|&s| s >= 1 && s <= st.max_block_size);
            if feasible && cur < best_km1 {
                best_km1 = cur;
                best_part.copy_from_slice(&st.part);
            }
        }};
    }

    let fcfg = super::fm::Config {
        rounds: fm_rounds,
        max_steps: fm_steps,
        seeds_per_search: fm_seeds,
        stop_factor: fm_stop,
    };

    let npass = passes.max(1);
    for pass in 0..npass {
        let before = st.km1();
        super::lp::refine(&mut st);
        keep!();
        super::jet::refine(&mut st, jet_rounds, jkeep);
        keep!();
        super::jet::cycles(&mut st, cyc_rounds);
        keep!();
        if local_only {
            if st.km1() == before {
                break;
            }
            continue;
        }
        if let Some(cfg2) = ml2.as_ref() {
            for sub in 0..cfg2.passes {
                super::ml2::refine::<FAST>(&mut st, cfg2, sub);
                if cfg2.cycles > 0 {
                    super::jet::cycles(&mut st, cfg2.cycles);
                }
                keep!();
            }
            if st.km1() == before {
                break;
            }
            continue;
        }
        if let Some(cfg) = ml.as_ref() {
            super::ml::refine::<FAST>(&mut st, cfg, &fcfg, pass);
            keep!();
            if st.km1() == before {
                break;
            }
            continue;
        }
        super::fm::refine::<FAST>(&mut st, &fcfg);
        keep!();
        let mut cur = st.km1();
        if let Some((p, after)) = super::coarsen::supernode_fm::<FAST>(&st, &fcfg) {
            if after < best_km1 {
                best_km1 = after;
                best_part.copy_from_slice(&p);
            }
            cur = after;
            if pass + 1 < npass {
                for (v, &to) in p.iter().enumerate() {
                    if st.part[v] != to {
                        st.apply(v as u32, to);
                    }
                }
            }
        }
        if cur == before {
            break;
        }
    }

    best_part.iter().map(|&x| x as u32).collect()
}
}
mod track;
mod track_10k;

const DEFAULTS_200K: [(&str, i64, i64, i64); 13] = [
    ("ml2_fine", 2, 0, 20),
    ("ml2_levels", 2, 1, 8),
    ("ml2_passes", 6, 1, 16),
    ("ml2_cyc", 2, 0, 64),
    ("cg", 1, 0, 2),
    ("cg_t", 656, 2, 1000),
    ("cool_pow", 3, 1, 3),
    ("tri_bar", 286_762, 200_000, 400_000),
    ("tri_margin", 0, 0, 50_000),
    ("post_band", 1, 0, 1),
    ("post_bar", 357_750, 200_000, 400_000),
    ("post_lo", -3_976, -20_000, 0),
    ("post_hi", 2_000, 0, 20_000),
];

fn with_200k_defaults(hp: &Option<Map<String, Value>>) -> Option<Map<String, Value>> {
    let mut m = hp.clone().unwrap_or_default();
    for &(name, value, lo, hi) in DEFAULTS_200K.iter() {
        if Hp(hp).get(name, lo, hi).is_none() {
            m.insert(name.to_owned(), Value::from(value));
        }
    }
    Some(m)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    if challenge.num_parts == 0
        || challenge.num_parts > 64
        || challenge.num_nodes < challenge.num_parts
    {
        anyhow::bail!(
            "unsupported instance: {} nodes, {} blocks",
            challenge.num_nodes,
            challenge.num_parts
        );
    }
    save_solution(&Solution {
        partition: round_robin(challenge),
    })?;

    let result = match challenge.num_hyperedges {
        20000 => track::solve(
            challenge,
            save_solution,
            hyperparameters,
            module,
            stream,
            prop,
            &params::P_20K,
        ),
        50000 => track::solve(
            challenge,
            save_solution,
            hyperparameters,
            module,
            stream,
            prop,
            &params::P_50K,
        ),
        100000 => track::solve(
            challenge,
            save_solution,
            hyperparameters,
            module,
            stream,
            prop,
            &params::P_100K,
        ),
        200000 => track::solve(
            challenge,
            save_solution,
            &with_200k_defaults(hyperparameters),
            module,
            stream,
            prop,
            &params::P_200K,
        ),
        _ => track_10k::solve(
            challenge,
            save_solution,
            hyperparameters,
            module,
            stream,
            prop,
        ),
    };
    result
}

pub fn help() {
    println!("mica_celadonite hyperparameters:");
    println!("  lz: read at 200k with pz only, 0..1 (default 1); invalid values use 1.");
    println!("  jkeep: 0..1 (1 at 20k, 0 elsewhere).");
    println!("  post_band: 0..1 (default 1, 200k), post_bar: 200000..400000 (357750, 200k),");
    println!("  post_lo: -20000..0 (-3976, 200k), post_hi: 0..20000 (2000, 200k).");
    println!(
        "  cg: 0..2 (default 1 at 200k, 0 elsewhere), cg_t: 2..1000 (656 at 200k, 160 elsewhere),"
    );
    println!("  cg_lv: 1..12 (8), cg_r: 1..500 (32).");
    println!("  clusters, cool_pct, cool_pow, cool_rounds, crossover, cyc_rounds,");
    println!("  effort, filt_g, flg_g, fm_rounds, fm_seeds, fm_steps, fm_stop, gc, gf, gf_band,");
    println!("  gf_band_hi, gf_band_lo, gf_batch, gf_end_k, gf_ext, gf_ext_passes,");
    println!("  gf_permille, gm, gm_batch, gm_fm, gm_permille, grp_ils, grp_n, hl_g,");
    println!("  ils_iterations, ils_quick_refine, inf_iters, inf_slack, inf_tau, jet_rounds,");
    println!("  max_stagnant_rounds, ml2, ml2_10k: 0..1 (default 1), ml2_20k: 0..1 (default 1),");
    println!("  ml2_50k: 0..1 (default 1), ml2_cmax, ml2_cyc, ml2_fine, ml2_gain, ml2_levels,");
    println!("  ml2_passes, ml2_rounds, ml2_seeds, ml2_steps, ml2_stop, ml_fine, ml_levels,");
    println!(
        "  ml_tail, move_limit, mv_npt, neg_gain_thresh, num_high_hedges, passes, pert_rounds,"
    );
    println!("  perturb_strength, post_ils_polish, post_refinement, pz, pz_rounds, refinement,");
    println!("  skip, slack_scale, sort_mb, sp_ils0, swap_scale, tabu_tenure, tri, tri_at,");
    println!(
        "  tri_at_10k, tri_at_20k, tri_at_50k, tri_bar, tri_bar_10k, tri_bar_20k, tri_bar_50k,"
    );
    println!("  tri_ext, tri_ext_10k, tri_ext_20k: 1000..2000 permille (default 1400 at 10k, 1200 at 20k).");
    println!(
        "  tri_hi, tri_k, tri_k_10k, tri_k_20k, tri_k_50k, tri_lag, tri_margin, tri_margin_10k,"
    );
    println!("  tri_margin_20k, tri_margin_50k, tri_rho, tri_sure.");
    println!(
        "  mv_seq: read at 200k only, 0..1 (default 0); fixed 1 at 20k/50k/100k; absent at 10k."
    );
    println!("  sp_sel: 0..1 (default 1, 20k/50k).");
}
