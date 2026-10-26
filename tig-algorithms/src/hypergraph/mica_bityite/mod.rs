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
mod lp;
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
    println!("mica_bityite: device partitioning with a host refinement chain.");
    println!(" - Dispatches on the number of hyperedges; each track has its own defaults.");
    println!(" - Deterministic decisions: no host thread, no clock, no hash map.");
    println!(" - Saves a feasible partition as soon as one exists and only replaces it by a");
    println!("   better one.");
    println!(" - The defaults are the intended operating point. A missing, non-integer or");
    println!("   out-of-range hyperparameter takes its default.");
    println!(" - HP: clusters, cool_pct, cool_pow, cool_rounds, crossover, cyc_rounds, effort,");
    println!("   filt_g, flg_g, fm_rounds, fm_seeds, fm_steps, fm_stop, gc, grp_ils, grp_n,");
    println!("   hl_g, ils_iterations, ils_quick_refine, inf_iters, inf_slack, inf_tau,");
    println!("   jet_rounds, max_stagnant_rounds, ml_fine, ml_levels, ml_tail, ml2, ml2_cmax,");
    println!("   ml2_cyc, ml2_cyc_last, ml2_fine, ml2_gain, ml2_levels, ml2_passes, ml2_rounds,");
    println!(
        "   ml2_seeds, ml2_steps, ml2_stop, move_limit, mv_npt, neg_gain_thresh, num_high_hedges,"
    );
    println!("   passes, pert_rounds, perturb_strength, post_ils_polish, post_refinement,");
    println!("   refinement, skip, slack_scale, sort_mb, swap_scale, tabu_tenure,");
    println!("   gf, gf_band, gf_band_hi, gf_band_lo, gf_batch, gf_end_k, gf_ext,");
    println!("   gf_ext_ils, gf_ext_passes, gf_permille, gf_trace, gm, gm_batch, gm_fm,");
    println!("   gm_permille, gm_trace, mv_ilp, mv_seq, pz, pz_rounds, tri, tri_at, tri_bar,");
    println!("   tri_hi, tri_k, tri_lag, tri_margin, tri_rho, tri_sure, tri_trace.");
}
