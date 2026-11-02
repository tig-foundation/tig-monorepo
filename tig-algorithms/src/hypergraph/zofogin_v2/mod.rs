// mica_clintonite: hypergraph partitioning.

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

    /// The integer value of `name` when it lies in `lo..=hi`.
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
        }
    }

    pub fn best(&self) -> u64 {
        self.best_km1
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
                (self.save_solution)(&Solution {
                    partition: partition.to_vec(),
                })?;
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

mod aim {
    use serde_json::{Map, Value};

    const TRACKS: [u32; 4] = [20000, 50000, 100000, 200000];
    const DEFAULT_TARGET: [f64; 4] = [0.28523, 0.284982, 0.280511, 0.282474];
    const SAMPLES: [usize; 4] = [4, 4, 4, 4];
    const BIAS: [f64; 4] = [1.0, 1.00088, 1.00057, 1.00061];
    const ABOVE: [f64; 4] = [0.0076, 0.00375, 0.0021, 0.00135];
    const DEFAULT_ABOVE: [f64; 4] = [0.009, 0.005, 0.005, 0.005];
    const ROUNDS: [usize; 4] = [100000, 100000, 61900, 72200];
    const WINDOW: [f64; 4] = [0.002, 0.003, 0.003, 0.002];
    const MARGIN: f64 = 0.001;
    const PASSES: [usize; 4] = [6, 16, 16, 16];
    const REHEATS: [usize; 4] = [0, 4, 4, 4];
    const REHEAT_ROUNDS: [usize; 4] = [0, 3000, 1000, 1000];
    const REHEAT_PASSES: usize = 3;
    const ILS_20K: [(&str, f64); 6] = [
        ("effort", 5.0),
        ("fm_rounds", 41.0),
        ("ils_iterations", 8.0),
        ("ils_quick_refine", 260.0),
        ("post_ils_polish", 90.0),
        ("swap_scale", 65.0),
    ];
    const BINS: usize = 100;
    const GAIN: [[f64; BINS + 1]; 4] = [
        [
            0.0774, 0.0766, 0.0761, 0.0736, 0.0724, 0.0723, 0.0716, 0.0711, 0.0714, 0.0711, 0.0703, 0.0701,
            0.0717, 0.0710, 0.0702, 0.0687, 0.0677, 0.0663, 0.0657, 0.0658, 0.0657, 0.0659, 0.0662, 0.0653,
            0.0662, 0.0669, 0.0652, 0.0645, 0.0645, 0.0638, 0.0632, 0.0637, 0.0628, 0.0634, 0.0617, 0.0616,
            0.0606, 0.0591, 0.0590, 0.0587, 0.0585, 0.0586, 0.0580, 0.0583, 0.0579, 0.0579, 0.0558, 0.0556,
            0.0559, 0.0542, 0.0533, 0.0523, 0.0521, 0.0518, 0.0503, 0.0513, 0.0500, 0.0513, 0.0489, 0.0485,
            0.0467, 0.0450, 0.0431, 0.0431, 0.0431, 0.0433, 0.0418, 0.0403, 0.0372, 0.0353, 0.0347, 0.0328,
            0.0322, 0.0324, 0.0303, 0.0281, 0.0275, 0.0260, 0.0256, 0.0243, 0.0237, 0.0223, 0.0218, 0.0211,
            0.0198, 0.0185, 0.0173, 0.0163, 0.0139, 0.0125, 0.0116, 0.0108, 0.0101, 0.0099, 0.0097, 0.0096,
            0.0095, 0.0095, 0.0094, 0.0093, 0.0093,
        ],
        [
            0.1037, 0.0612, 0.0580, 0.0568, 0.0564, 0.0564, 0.0546, 0.0541, 0.0536, 0.0537, 0.0537, 0.0529,
            0.0518, 0.0517, 0.0521, 0.0519, 0.0514, 0.0496, 0.0501, 0.0499, 0.0489, 0.0488, 0.0488, 0.0493,
            0.0480, 0.0481, 0.0470, 0.0471, 0.0465, 0.0460, 0.0459, 0.0452, 0.0456, 0.0451, 0.0444, 0.0440,
            0.0437, 0.0428, 0.0431, 0.0426, 0.0421, 0.0407, 0.0412, 0.0422, 0.0412, 0.0404, 0.0406, 0.0397,
            0.0387, 0.0381, 0.0379, 0.0372, 0.0374, 0.0358, 0.0355, 0.0356, 0.0346, 0.0339, 0.0334, 0.0331,
            0.0320, 0.0312, 0.0309, 0.0292, 0.0286, 0.0282, 0.0275, 0.0263, 0.0249, 0.0242, 0.0238, 0.0222,
            0.0216, 0.0213, 0.0191, 0.0190, 0.0171, 0.0166, 0.0155, 0.0148, 0.0138, 0.0127, 0.0119, 0.0113,
            0.0105, 0.0102, 0.0093, 0.0083, 0.0078, 0.0070, 0.0062, 0.0058, 0.0052, 0.0048, 0.0046, 0.0044,
            0.0042, 0.0041, 0.0040, 0.0039, 0.0039,
        ],
        [
            0.1421, 0.1212, 0.1178, 0.1171, 0.1170, 0.1158, 0.1119, 0.1154, 0.1145, 0.1115, 0.1102, 0.1124,
            0.1075, 0.1065, 0.1050, 0.1041, 0.1046, 0.1043, 0.1022, 0.1016, 0.0980, 0.1009, 0.0981, 0.0976,
            0.0948, 0.0955, 0.0940, 0.0934, 0.0915, 0.0913, 0.0882, 0.0884, 0.0857, 0.0850, 0.0842, 0.0836,
            0.0812, 0.0812, 0.0786, 0.0764, 0.0751, 0.0747, 0.0724, 0.0718, 0.0701, 0.0691, 0.0673, 0.0659,
            0.0641, 0.0622, 0.0605, 0.0588, 0.0574, 0.0558, 0.0545, 0.0525, 0.0510, 0.0489, 0.0479, 0.0466,
            0.0446, 0.0426, 0.0436, 0.0405, 0.0398, 0.0381, 0.0376, 0.0364, 0.0333, 0.0333, 0.0307, 0.0295,
            0.0279, 0.0257, 0.0244, 0.0231, 0.0213, 0.0192, 0.0171, 0.0156, 0.0135, 0.0121, 0.0110, 0.0094,
            0.0088, 0.0080, 0.0071, 0.0064, 0.0057, 0.0054, 0.0048, 0.0048, 0.0045, 0.0043, 0.0040, 0.0040,
            0.0040, 0.0038, 0.0038, 0.0037, 0.0037,
        ],
        [
            0.2525, 0.1037, 0.1011, 0.1000, 0.0985, 0.0989, 0.0957, 0.0960, 0.0941, 0.0929, 0.0939, 0.0956,
            0.0923, 0.0939, 0.0912, 0.0921, 0.0912, 0.0900, 0.0893, 0.0896, 0.0883, 0.0885, 0.0873, 0.0873,
            0.0874, 0.0864, 0.0861, 0.0848, 0.0839, 0.0840, 0.0830, 0.0819, 0.0834, 0.0822, 0.0807, 0.0804,
            0.0793, 0.0786, 0.0772, 0.0770, 0.0786, 0.0754, 0.0755, 0.0747, 0.0745, 0.0731, 0.0714, 0.0711,
            0.0713, 0.0704, 0.0695, 0.0668, 0.0673, 0.0677, 0.0647, 0.0637, 0.0631, 0.0624, 0.0613, 0.0597,
            0.0592, 0.0575, 0.0567, 0.0562, 0.0548, 0.0534, 0.0520, 0.0506, 0.0501, 0.0485, 0.0469, 0.0457,
            0.0437, 0.0431, 0.0418, 0.0402, 0.0390, 0.0377, 0.0375, 0.0362, 0.0352, 0.0345, 0.0328, 0.0318,
            0.0310, 0.0291, 0.0277, 0.0265, 0.0241, 0.0231, 0.0200, 0.0179, 0.0159, 0.0112, 0.0088, 0.0066,
            0.0044, 0.0034, 0.0030, 0.0026, 0.0026,
        ],
    ];

    pub struct Aim {
        idx: usize,
        target: f64,
        base: f64,
        rounds: usize,
        give_up: bool,
        stop: bool,
        recipe: Option<&'static super::recipes::Recipe>,
    }

    impl Aim {
        pub fn new(num_hyperedges: u32, hp: &Option<Map<String, Value>>, rounds: usize, clusters: i32) -> Option<Aim> {
            let idx = TRACKS.iter().position(|&t| t == num_hyperedges)?;
            let given = hp
                .as_ref()
                .and_then(|m| m.get("aim_target"))
                .and_then(Value::as_f64)
                .map(|t| if t > 1.0 { t / 1e6 } else { t });
            let target = given.unwrap_or(DEFAULT_TARGET[idx]);
            let plain = hp.as_ref().map_or(idx != 0, |m| {
                let listed = |k: &str, v: &Value| idx == 0 && ILS_20K.iter().any(|&(n, x)| n == k && v.as_f64() == Some(x));
                let effort = |k: &str, v: &Value| k == "effort" && v.as_f64() == Some(5.0);
                m.iter().all(|(k, v)| k == "aim_target" || k == "clusters" || k == "refinement" || effort(k, v) || listed(k, v))
                    && (idx != 0 || ILS_20K.iter().all(|&(n, _)| m.contains_key(n)))
            });
            let near = rounds * 5 >= ROUNDS[idx] * 4 && rounds * 5 <= ROUNDS[idx] * 6;
            let give_up = plain && clusters == 4 && near;
            let recipe = super::recipes::select(num_hyperedges, hp, rounds, clusters);
            Some(Aim {
                idx,
                target,
                base: 0.0,
                rounds: rounds.max(1),
                give_up,
                stop: given.is_some(),
                recipe,
            })
        }

        pub fn estimate(&mut self, he_off: &[i32], he_nodes: &[i32], part: &[i32], seed: u64) {
            let b = super::baseline::estimate(he_off, he_nodes, part.len(), part, SAMPLES[self.idx], seed);
            self.base = b / BIAS[self.idx];
        }

        pub fn step(&self) -> usize {
            (self.rounds / BINS).max(1)
        }

        pub fn bin(&self, round: u64) -> usize {
            ((round as usize).saturating_mul(BINS) / self.rounds).min(BINS - 1)
        }

        pub fn settled(&self, bin: usize, km1: u64) -> bool {
            if self.base <= 0.0 {
                return false;
            }
            let q = 1.0 - km1 as f64 / self.base;
            if let Some(recipe) = self.recipe {
                return (self.stop && q >= self.target + recipe.above) || q + recipe.gain[bin.min(BINS)] < self.target - WINDOW[self.idx];
            }
            (self.stop && q >= self.target + ABOVE[self.idx])
                || (self.give_up && !self.stop && q >= self.target + DEFAULT_ABOVE[self.idx])
                || (self.give_up && q + GAIN[self.idx][bin.min(BINS)] < self.target - WINDOW[self.idx])
        }

        pub fn done(&self, km1: u64) -> bool {
            self.settled(BINS, km1)
        }

        fn quality(&self, km1: u64) -> f64 {
            1.0 - km1 as f64 / self.base
        }

        pub fn extend(&self, km1: u64) -> bool {
            let q = self.quality(km1);
            let top = if self.stop { MARGIN } else { WINDOW[self.idx] };
            self.base > 0.0 && q >= self.target - WINDOW[self.idx] && q < self.target + top
        }

        pub fn passes(&self) -> usize {
            if self.reheats() > 0 { REHEAT_PASSES } else { PASSES[self.idx] }
        }

        pub fn reheats(&self) -> usize {
            if self.recipe.is_some() { 0 } else { REHEATS[self.idx] }
        }

        pub fn reheat_rounds(&self) -> usize {
            REHEAT_ROUNDS[self.idx]
        }

        pub fn extended(&self, km1: u64) -> bool {
            self.stop && self.quality(km1) >= self.target + MARGIN
        }
    }
}
mod baseline;
mod coarsen;
mod fm;
mod gain;
mod grouped;
mod grouped_small;
mod infer;
mod jet;
mod lp {
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
}
mod ml;
mod ml2;
mod params;
mod recipes {
    use serde_json::{Map, Value};

    pub struct Recipe {
        pub above: f64,
        pub gain: [f64; 101],
    }

    static C100: Recipe = Recipe {
        above: 0.0039,
        gain: [
            0.8609, 0.8609, 0.8609, 0.1225, 0.1202, 0.1184, 0.1155, 0.1149, 0.1136, 0.1135,
            0.1124, 0.1110, 0.1095, 0.1095, 0.1069, 0.1080, 0.1069, 0.1064, 0.1042, 0.1040,
            0.1029, 0.1018, 0.0997, 0.0997, 0.0992, 0.0981, 0.0962, 0.0947, 0.0942, 0.0926,
            0.0925, 0.0930, 0.0914, 0.0890, 0.0878, 0.0859, 0.0837, 0.0828, 0.0821, 0.0801,
            0.0791, 0.0788, 0.0773, 0.0762, 0.0747, 0.0725, 0.0713, 0.0718, 0.0678, 0.0672,
            0.0650, 0.0639, 0.0624, 0.0606, 0.0592, 0.0581, 0.0563, 0.0546, 0.0532, 0.0523,
            0.0504, 0.0487, 0.0470, 0.0453, 0.0448, 0.0426, 0.0422, 0.0411, 0.0395, 0.0377,
            0.0367, 0.0336, 0.0338, 0.0322, 0.0295, 0.0282, 0.0260, 0.0252, 0.0232, 0.0195,
            0.0185, 0.0180, 0.0169, 0.0149, 0.0142, 0.0133, 0.0116, 0.0109, 0.0105, 0.0095,
            0.0092, 0.0087, 0.0083, 0.0081, 0.0077, 0.0077, 0.0077, 0.0075, 0.0073, 0.0074,
            0.0075,
        ],
    };

    static C200: Recipe = Recipe {
        above: 0.0033,
        gain: [
            0.8142, 0.8142, 0.8142, 0.1022, 0.1009, 0.0997, 0.0995, 0.0975, 0.0977, 0.0982,
            0.0982, 0.0982, 0.0983, 0.0961, 0.0944, 0.0941, 0.0943, 0.0928, 0.0929, 0.0921,
            0.0909, 0.0913, 0.0913, 0.0906, 0.0889, 0.0878, 0.0879, 0.0877, 0.0879, 0.0861,
            0.0869, 0.0857, 0.0855, 0.0849, 0.0842, 0.0835, 0.0837, 0.0825, 0.0818, 0.0819,
            0.0806, 0.0798, 0.0782, 0.0775, 0.0769, 0.0764, 0.0763, 0.0756, 0.0746, 0.0737,
            0.0725, 0.0719, 0.0710, 0.0710, 0.0702, 0.0686, 0.0674, 0.0669, 0.0653, 0.0648,
            0.0636, 0.0624, 0.0612, 0.0608, 0.0591, 0.0581, 0.0573, 0.0562, 0.0550, 0.0530,
            0.0524, 0.0520, 0.0494, 0.0484, 0.0476, 0.0464, 0.0451, 0.0441, 0.0427, 0.0412,
            0.0406, 0.0395, 0.0380, 0.0372, 0.0358, 0.0356, 0.0338, 0.0318, 0.0305, 0.0294,
            0.0269, 0.0254, 0.0235, 0.0230, 0.0196, 0.0158, 0.0131, 0.0100, 0.0083, 0.0069,
            0.0065,
        ],
    };

    static EXTRA50: Recipe = Recipe {
        above: 0.0048,
        gain: [
            0.9167, 0.9167, 0.9167, 0.0666, 0.0611, 0.0605, 0.0602, 0.0592, 0.0579, 0.0581,
            0.0565, 0.0565, 0.0562, 0.0558, 0.0557, 0.0562, 0.0562, 0.0556, 0.0553, 0.0538,
            0.0538, 0.0532, 0.0529, 0.0532, 0.0533, 0.0524, 0.0514, 0.0510, 0.0504, 0.0503,
            0.0502, 0.0497, 0.0495, 0.0484, 0.0486, 0.0477, 0.0470, 0.0471, 0.0459, 0.0458,
            0.0460, 0.0452, 0.0446, 0.0434, 0.0433, 0.0428, 0.0428, 0.0425, 0.0424, 0.0407,
            0.0398, 0.0399, 0.0397, 0.0396, 0.0395, 0.0380, 0.0376, 0.0371, 0.0367, 0.0353,
            0.0349, 0.0344, 0.0340, 0.0331, 0.0326, 0.0318, 0.0318, 0.0316, 0.0300, 0.0286,
            0.0272, 0.0262, 0.0256, 0.0256, 0.0243, 0.0244, 0.0220, 0.0216, 0.0195, 0.0193,
            0.0185, 0.0179, 0.0170, 0.0162, 0.0157, 0.0142, 0.0134, 0.0129, 0.0118, 0.0111,
            0.0105, 0.0097, 0.0093, 0.0086, 0.0081, 0.0078, 0.0076, 0.0073, 0.0079, 0.0079,
            0.0079,
        ],
    };

    pub fn select(num_hyperedges: u32, hp: &Option<Map<String, Value>>, rounds: usize, clusters: i32) -> Option<&'static Recipe> {
        let (recipe, expected): (&Recipe, &[(&str, i64)]) = match (num_hyperedges, rounds) {
            (100000, 30950) => (&C100, &[("clusters", 4), ("refinement", 50000)]),
            (200000, 43320) => (&C200, &[("clusters", 4), ("refinement", 60000)]),
            (50000, 100000) => (&EXTRA50, &[("clusters", 4), ("effort", 5), ("fm_rounds", 10), ("ils_iterations", 16), ("ils_quick_refine", 150), ("post_ils_polish", 260), ("refinement", 100000), ("swap_scale", 120)]),
            _ => return None,
        };
        let values = hp.as_ref()?;
        if clusters != 4 || values.len() != expected.len() + usize::from(values.contains_key("aim_target")) {
            return None;
        }
        if expected.iter().all(|&(key, value)| values.get(key).and_then(Value::as_i64) == Some(value)) {
            Some(recipe)
        } else {
            None
        }
    }
}
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
    println!("n_h_edges=50000, 100000, 200000: send {{\"clusters\": 4, \"refinement\": 100000, \"aim_target\": T}}.");
    println!("n_h_edges=20000: send {{\"clusters\": 4, \"effort\": 5, \"fm_rounds\": 41, \"ils_iterations\": 8, \"ils_quick_refine\": 260, \"post_ils_polish\": 90, \"refinement\": 100000, \"swap_scale\": 65, \"aim_target\": T}}.");
    println!("T = the track's qualifying quality (280037 on chain = 0.280037); update T when the line moves by more than the stop margin in the README.");
    println!("n_h_edges=10000: mica_clintonite settings, same output.");
}
