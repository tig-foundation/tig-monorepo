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

pub(crate) const FILTER_THREADS: u32 = 256;
pub(crate) const SELECT_THREADS: u32 = 1024;
pub(crate) const FINISH_THREADS: u32 = 256;

const TRI100_DEFAULT_ENABLED: i64 = 1;
const TRI100_DEFAULT_AT: i64 = 100;
const TRI100_DEFAULT_BAR: i64 = 280_425;
const TRI100_DEFAULT_MARGIN: i64 = 9_000;
const TRI100_DEFAULT_K: i64 = 10_632;
const TRI100_DEFAULT_LAG: i64 = 2;
const TRI100_DEFAULT_SURE: i64 = 1;
const TRI100_DEFAULT_HI: i64 = 9_000;
const TRI100_DEFAULT_RHO: i64 = 650;

const AIM_TRACKS: [usize; 4] = [20_000, 50_000, 100_000, 200_000];
const AIM_EXT: [usize; 4] = [0, 6, 0, 6];
const AIM_DEFAULT_TARGET: [i64; 4] = [285_567, 284_779, 280_911, 282_733];
const AIM_BIAS_PPM: [i64; 4] = [-1_000, 450, 850, 515];
const AIM_WINDOW: i64 = 2_000;
const AIM_ABOVE: i64 = 1_500;
const AIM_SAMPLES: usize = 4;
const CUT_TRACKS: [usize; 13] = [
    10_000,
    20_000,
    100_000,
    10_000,
    10_000,
    20_000,
    50_000,
    200_000,
    50_000,
    20_000,
    200_000,
    20_000,
    100_000,
];
const CUT_DEFAULT_TARGET: [i64; 13] = [
    288_542,
    285_567,
    280_911,
    288_542,
    288_542,
    285_567,
    284_779,
    282_733,
    284_779,
    285_567,
    282_733,
    285_567,
    280_911,
];
const CUT_BIAS_PPM: [i64; 13] = [200, -1_000, 850, 200, 200, -1_000, 450, 515, 450, -1_000, 515, -1_000, 850];
const CUT_MARGIN: [i64; 13] = [8_300, 7_250, 1_000, 8_300, 8_300, 7_250, 3_000, 1_500, 3_000, 7_250, 1_500, 7_250, 1_000];
const CUT_ABOVE: [i64; 13] = [
    8_300,
    7_250,
    1_000,
    8_300,
    1_000_000,
    7_250,
    1_000_000,
    1_000_000,
    1_000_000,
    1_000_000,
    1_000_000,
    1_000_000,
    1_000_000,
];
const CUT_GAIN: [i64; 13] = [5_000, 2_000, 0, 6_500, 6_500, 2_700, 0, 0, 0, 3_000, 0, 3_600, 0];
const CUT_SETTINGS: [&[(&str, i64)]; 13] = [
    &[
        ("clusters", 4),
        ("effort", 5),
        ("fm_rounds", 48),
        ("ils_iterations", 28),
        ("ils_quick_refine", 240),
        ("post_ils_polish", 300),
        ("refinement", 80_000),
        ("swap_scale", 120),
    ],
    &[
        ("clusters", 4),
        ("effort", 5),
        ("fm_rounds", 41),
        ("ils_iterations", 8),
        ("ils_quick_refine", 260),
        ("post_ils_polish", 90),
        ("refinement", 100_000),
        ("swap_scale", 65),
    ],
    &[],
    &[],
    &[("refinement", 60_000)],
    &[
        ("clusters", 4),
        ("effort", 5),
        ("fm_rounds", 41),
        ("ils_iterations", 8),
        ("ils_quick_refine", 260),
        ("post_ils_polish", 90),
        ("pz", 0),
        ("refinement", 100_000),
        ("swap_scale", 65),
    ],
    &[("clusters", 4), ("refinement", 100_000)],
    &[("clusters", 4), ("refinement", 100_000)],
    &[("clusters", 4), ("refinement", 80_000)],
    &[
        ("clusters", 4),
        ("effort", 5),
        ("fm_rounds", 41),
        ("ils_iterations", 16),
        ("ils_quick_refine", 280),
        ("post_ils_polish", 96),
        ("refinement", 100_000),
        ("swap_scale", 65),
    ],
    &[("clusters", 4), ("refinement", 80_000)],
    &[
        ("clusters", 4),
        ("effort", 5),
        ("fm_rounds", 41),
        ("ils_iterations", 16),
        ("ils_quick_refine", 280),
        ("post_ils_polish", 96),
        ("refinement", 80_000),
        ("swap_scale", 65),
    ],
    &[],
];
const CUT_OPTIONAL: [&[(&str, i64)]; 13] = [
    &[],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[("effort", 5)],
    &[("effort", 5)],
    &[("effort", 5)],
    &[],
    &[("effort", 5)],
    &[],
    &[("clusters", 4), ("effort", 5), ("refinement", 100_000)],
];
const CUT100_STEPS: [(usize, i64); 5] = [(600, 30_400), (700, 20_900), (800, 12_600), (900, 4_600), (1_000, 2_900)];
const CUT20_STEPS: [(usize, i64); 7] =
    [(300, 53_600), (400, 48_600), (500, 44_800), (600, 35_400), (700, 26_100), (800, 19_700), (900, 9_900)];
const CUT50_STEPS: [(usize, i64); 7] =
    [(300, 45_100), (400, 40_800), (500, 37_900), (600, 29_700), (700, 22_300), (800, 13_400), (900, 7_100)];
const CUT200_STEPS: [(usize, i64); 7] =
    [(300, 81_200), (400, 74_900), (500, 69_100), (600, 59_900), (700, 46_400), (800, 37_300), (900, 23_000)];

const CUT_STEPS: [&[(usize, i64)]; 13] = [
    &[],
    &CUT20_STEPS,
    &[],
    &[],
    &[],
    &[],
    &CUT50_STEPS,
    &CUT200_STEPS,
    &CUT50_STEPS,
    &CUT20_STEPS,
    &CUT200_STEPS,
    &CUT20_STEPS,
    &[],
];

const TRI200_DEFAULT_ENABLED: i64 = 1;
const TRI200_DEFAULT_AT: i64 = 100;
const TRI200_DEFAULT_BAR: i64 = 282_738;
const TRI200_DEFAULT_MARGIN: i64 = 12_000;
const TRI200_DEFAULT_K: i64 = 11_106;
const TRI200_DEFAULT_LAG: i64 = 2;
const TRI200_DEFAULT_SURE: i64 = 0;
const TRI200_DEFAULT_HI: i64 = 9_000;
const TRI200_DEFAULT_RHO: i64 = 650;

const GF_TRI_K: i64 = 9_997;
const GF_TRI_MARGIN: i64 = 9_000;
const GF_ROUNDS_PERMILLE: i64 = 1_250;
const GF_BATCH: i64 = 2_048;
const GF_TRI_SURE: i64 = 1;
const GF_TRI_HI: i64 = 8_000;
const GF_TRI_RHO: i64 = 650;
const GF_BAND: i64 = 1;
const GF_BAND_LO: i64 = -2_000;
const GF_BAND_HI: i64 = 3_000;
const GF_END_K: i64 = 9_428;
const GF_EXT: i64 = 200;
const GF_EXT_ILS: i64 = 0;
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
    est: Option<baseline::Est>,
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
            est: None,
        }
    }

    pub fn pump(&mut self) {
        if let Some(e) = self.est.as_mut() {
            e.step();
        }
    }

    pub fn est_value(&mut self) -> f64 {
        self.est.as_mut().map_or(f64::NAN, |e| e.value())
    }

    pub fn save(&mut self, partition: &[u32]) -> anyhow::Result<()> {
        if let Some(k) = feasible_km1(
            self.he_off,
            self.he_nodes,
            partition,
            self.num_parts,
            self.max_part_size,
        ) {
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
        Ok(())
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

mod coarsen;
mod fm;
mod gain;
mod grouped;
mod grouped_small;
mod infer;
mod jet;
mod baseline;
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
mod ml;
mod ml2;
mod params;
mod refine;
mod track;
mod track_10k;

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

    match challenge.num_hyperedges {
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
            hyperparameters,
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
    }
}

pub fn help() {
    println!("zofogin_v3: mica_bityite with extra refinement for nonces that end just below the qualifying quality T, and an earlier end on every track.");
    println!(" - Send {{\"aim_target\": T}} on every track, T = the track's qualifying quality (282733 on chain = 0.282733).");
    println!(" - Default T: 10000 0.288542, 20000 0.285567, 50000 0.284779, 100000 0.280911, 200000 0.282733 (block 1414645). Stop margin 0.0083 / 0.00725 / 0.0015 / 0.001 / 0.0015: update T when the line moves by more.");
    println!(" - Nonces end early only at the recommended settings (10000 also with null or {{\"refinement\": 60000}}, 20000 also with {{\"pz\": 0}} added; \"effort\": 5 may be added on 50000 and 200000, and 100000 also ends early with {{\"clusters\": 4, \"refinement\": 100000}}); on 50000 and 200000, and on 100000 with any of these keys, only nonces far below T end early. The same checks also run on 50000 and 200000 with {{\"clusters\": 4, \"refinement\": 80000}} (\"effort\": 5 may be added), and on 20000 with {{\"clusters\": 4, \"effort\": 5, \"fm_rounds\": 41, \"ils_iterations\": 16, \"ils_quick_refine\": 280, \"post_ils_polish\": 96, \"refinement\": 100000, \"swap_scale\": 65}} or the same with \"refinement\": 80000, again only far below T. Every mica_bityite hyperparameter applies with its default.");
}
