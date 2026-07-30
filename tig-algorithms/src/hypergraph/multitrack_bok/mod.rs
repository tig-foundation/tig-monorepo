use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Number, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

pub mod track_t21;
pub mod track_t22;
pub mod track_t23;
pub mod track_t24;
pub mod track_t25;

fn merge_hp(user_hp: &Option<Map<String, Value>>, defaults: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user_hp.clone().unwrap_or_default();
    for (k, v) in defaults {
        m.entry(k.to_string()).or_insert(v);
    }
    Some(m)
}

fn n(v: u64) -> Value { Value::Number(Number::from(v)) }
fn f(v: f64) -> Value { Value::Number(Number::from_f64(v).unwrap()) }
fn b(v: bool) -> Value { Value::Bool(v) }

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
    save_solution(&Solution { partition: dummy_partition })?;

    // Per-track baked HP = winning iter config (reproduces both Q AND cycle at hp={}).
    match challenge.num_hyperedges {
        10000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("max_chain_len", self::n(4)),
                ("harvest_stop_eps", self::n(4)),
                ("harvest_incremental", self::n(1)),
                ("harvest_boundary_only", self::n(1)),
                ("overlay", self::n(1)),
                ("runs", self::n(36)),
            ]);
            track_t21::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        20000 => {
            // CHAMPION config (imp022): refinement 3x + memetic (agreement-backbone
            // crossover + immigrants + polish) + cut-overlay. Was stale (23800, no
            // memetic) => gave the light 269k; champion default gives ~274k @90n K=2.
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("clusters", self::n(64)),
                ("move_limit", self::n(800000)),
                ("refinement", self::n(71400)),
                ("tabu_tenure", self::n(8)),
                ("ils_iterations", self::n(10)),
                ("post_ils_polish", self::n(200)),
                ("post_refinement", self::n(128)),
                ("ils_quick_refine", self::n(100)),
                ("memetic", self::n(1)),
                ("mem_pop", self::n(10)),
                ("mem_recomb", self::n(84)),
                ("mem_polish", self::n(140)),
                ("mem_diversity", self::n(1)),
                ("mem_immig_gap", self::n(3)),
                ("mem_immig_free_pct", self::n(50)),
                ("mem_immig_polish", self::n(55)),
                ("mem_immig_max", self::n(24)),
                ("runs", self::n(15)),
            ]);
            track_t23::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        50000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("clusters", self::n(64)),
                ("move_limit", self::n(880000)),
                ("refinement", self::n(10800)),
                ("tabu_tenure", self::n(8)),
                ("ils_iterations", self::n(10)),
                ("post_ils_polish", self::n(100)),
                ("post_refinement", self::n(64)),
                ("ils_quick_refine", self::n(50)),
                ("perturb_strength", self::n(5)),
                ("pr_rebal_hoist", b(true)),
                ("overlay", self::n(1)),
                ("runs", self::n(31)),
            ]);
            track_t25::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        100000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("post_ils_polish", self::n(200)),
                ("sp_gpu_score", b(true)),
                ("overlay", self::n(1)),
                ("runs", self::n(21)),
            ]);
            track_t22::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        200000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("clusters", self::n(80)),
                ("move_limit", self::n(300000)),
                ("refinement", self::n(9000)),
                ("tabu_tenure", self::n(10)),
                ("post_ils_polish", self::n(80)),
                ("post_refinement", self::n(2)),
                ("ils_quick_refine", self::n(0)),
                ("ils_iterations", self::n(5)),
                ("gp_topk", self::n(2)),
                ("seq_sweeps", self::n(8)),
                ("pr_n_guides", self::n(2)),
                ("global_prefix", self::n(1)),
                ("pr_gain_cache", b(true)),
                ("pr_max_cycles", self::n(2)),
                ("pr_trunc_frac", f(1.0)),
                ("vigw_iqr_mult", f(2.0)),
                ("pr_rebal_hoist", b(true)),
                ("aspiration_frac", f(0.75)),
                ("ejchain_max_len", self::n(5)),
                ("term_refine_mode", self::n(2)),
                ("fm_localized_mode", self::n(0)),
                ("pr_gpu_interleave", b(true)),
                ("vigw_perturb_mode", self::n(3)),
                ("fm_crossblock_mode", self::n(0)),
                ("launch_bounds_min_blocks", self::n(3)),
                ("overlay", self::n(1)),
                ("runs", self::n(23)),
            ]);
            track_t24::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        _ => track_t21::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("algo");
}
