// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
//
// mica_muscovite: GPU refinement followed by a host refinement chain, dispatched by
// instance size.

use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

// Set by the runtime to the fuel cap and decremented by the instrumentation.
extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

pub(crate) fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

mod gain;
mod lp;
mod jet;
mod fm;
mod refine;
mod params;
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
    let dummy_partition: Vec<u32> = (0..challenge.num_nodes as u32)
        .map(|i| i % challenge.num_parts as u32)
        .collect();
    save_solution(&Solution {
        partition: dummy_partition,
    })?;

    match challenge.num_hyperedges {
        10000 => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
        20000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_20K),
        50000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_50K),
        100000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_100K),
        200000 => track::solve(challenge, save_solution, hyperparameters, module, stream, prop, &params::P_200K),
        _ => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("mica_muscovite: GPU partitioning with a host-side refinement chain.");
    println!(" - Dispatches on the number of hyperedges; each track has its own settings.");
    println!(" - Deterministic: no hash map on the host, explicit tie-breaks, seeded orders.");
    println!(" - Saves a valid partition as soon as one exists and never returns a worse one.");
    println!(" - Defaults are the intended operating point; any HP overrides them.");
    println!(" - HP: effort, refinement, ils_iterations, ils_quick_refine, post_ils_polish,");
    println!("       post_refinement, tabu_tenure, move_limit, stall_window, init_restart_id,");
    println!("       tie_fm, planted, planted_lvl, fm_rounds, swap_scale, zgm");
}
