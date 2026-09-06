use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value, Number};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

mod track_10k;
mod track_20k;
mod track_50k;
mod track_100k;
mod track_200k;

fn merge_hp(user: &Option<Map<String, Value>>, defs: Vec<(&str, Value)>) -> Option<Map<String, Value>> {
    let mut m = user.clone().unwrap_or_default();
    for (k, v) in defs { m.entry(k.to_string()).or_insert(v); }
    Some(m)
}
fn n(v: i64) -> Value { Value::Number(Number::from(v)) }

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

    match challenge.num_hyperedges {
        10000 => {
            let hp = merge_hp(hyperparameters, vec![("effort", n(5)), ("clusters", n(64)), ("move_limit", n(800000)), ("refinement", n(14000)), ("tabu_tenure", n(7)), ("ils_iterations", n(15)), ("init_restart_id", n(0)), ("post_ils_polish", n(220)), ("post_refinement", n(0)), ("ils_quick_refine", n(180))]);
            track_10k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        20000 => {
            let hp = merge_hp(hyperparameters, vec![("effort", n(5)), ("clusters", n(64)), ("move_limit", n(800000)), ("refinement", n(36800)), ("tabu_tenure", n(8)), ("ils_iterations", n(10)), ("post_ils_polish", n(200)), ("post_refinement", n(128)), ("ils_quick_refine", n(100))]);
            track_20k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        50000 => {
            let hp = merge_hp(hyperparameters, vec![("refinement", n(24000)), ("boundary_only", n(1))]);
            track_50k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        100000 => {
            let hp = merge_hp(hyperparameters, vec![("effort", n(5)), ("clusters", n(64)), ("refinement", n(13000)), ("accept_mode", n(0)), ("tabu_tenure", n(12)), ("zgm_tiebreak", n(1)), ("ils_iterations", n(15)), ("num_high_hedges", n(500)), ("post_ils_polish", n(32)), ("ils_quick_refine", n(160)), ("perturb_escalate", n(0))]);
            track_100k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        200000 => {
            let hp = merge_hp(hyperparameters, vec![("clusters", n(80)), ("move_limit", n(500000)), ("refinement", n(20000)), ("tabu_tenure", n(8)), ("post_ils_polish", n(200)), ("post_refinement", n(2)), ("ils_quick_refine", n(96))]);
            track_200k::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        _ => track_10k::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}
pub fn help() { println!("hyper_sigma_v2 - GPU hypergraph partitioning, per-track"); }
