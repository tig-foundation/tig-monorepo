// mica_inf: hypergraph partitioning on the device, then a host refinement chain.

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

// Hyperparameter access: a missing value takes the default, an out-of-range value is clamped.
pub(crate) struct Hp<'a>(pub &'a Option<Map<String, Value>>);

impl Hp<'_> {
    pub fn int(&self, name: &str, lo: i64, hi: i64, default: i64) -> i64 {
        self.0
            .as_ref()
            .and_then(|p| p.get(name))
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(lo, hi))
            .unwrap_or(default)
    }
}

// Connectivity (km1) of a partition, or None when it violates the block-size constraint.
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

// Saves a partition only when it is feasible and strictly better than every partition saved so
// far, the round-robin one of `solve_challenge` included.
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
        let best_km1 = feasible_km1(he_off, he_nodes, &round_robin, challenge.num_parts, challenge.max_part_size)
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

    pub fn save(&mut self, partition: &[u32]) -> anyhow::Result<()> {
        if let Some(k) = feasible_km1(self.he_off, self.he_nodes, partition, self.num_parts, self.max_part_size) {
            if k < self.best_km1 {
                self.best_km1 = k;
                (self.save_solution)(&Solution { partition: partition.to_vec() })?;
            }
        }
        Ok(())
    }
}

pub(crate) fn to_u32(partition: &[i32]) -> Vec<u32> {
    partition.iter().map(|&x| x as u32).collect()
}

fn round_robin(challenge: &Challenge) -> Vec<u32> {
    (0..challenge.num_nodes).map(|i| i % challenge.num_parts).collect()
}

// Device move lists. A pick word is (node << 32) | key; a sorted word is (!key << 32) | node.
// The target block is the low six bits of the key.
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
mod infer;
mod jet;
mod lp;
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
    if challenge.num_parts == 0 || challenge.num_parts > 64 || challenge.num_nodes < challenge.num_parts {
        anyhow::bail!("unsupported instance: {} nodes, {} blocks", challenge.num_nodes, challenge.num_parts);
    }
    save_solution(&Solution { partition: round_robin(challenge) })?;

    match challenge.num_hyperedges {
        20000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_20K),
        50000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_50K),
        100000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_100K),
        200000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_200K),
        _ => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("mica_inf: device partitioning with a host refinement chain.");
    println!(" - Dispatches on the number of hyperedges; each track has its own defaults.");
    println!(" - Deterministic: no thread, no clock, no hash map; grid-wide synchronisation only at launch boundaries.");
    println!(" - Saves a feasible partition as soon as one exists and only replaces it by a better one.");
    println!(" - The defaults are the intended operating point. A missing or non-integer hyperparameter");
    println!("   takes its default; an integer (representable as i64) outside its range is clamped.");
    println!(" - Read on every track: clusters, cyc_rounds, effort (0..5), fm_rounds, fm_seeds,");
    println!("   fm_steps, fm_stop, ils_iterations, ils_quick_refine, inf_iters, inf_slack,");
    println!("   inf_tau (tenths), jet_rounds, max_stagnant_rounds, move_limit, neg_gain_thresh,");
    println!("   num_high_hedges, passes, post_ils_polish, refinement, swap_scale (percent), tabu_tenure");
    println!(" - Read on 20k/50k/100k/200k only: cool_pct, cool_pow, cool_rounds, pert_rounds,");
    println!("   perturb_strength, post_refinement, slack_scale");
    println!(" - Read on 10k only: crossover");
}
