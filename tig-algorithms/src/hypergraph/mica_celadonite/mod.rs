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

mod coarsen;
mod fm;
mod gain;
mod grouped;
mod grouped_small;
mod infer;
mod jet;
mod lp;
mod ml;
mod ml2;
mod params;
mod refine;
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
