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
            track_t21::solve(challenge, save_solution, hyperparameters, module, stream, prop)
        }
        20000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("clusters", self::n(64)),
                ("move_limit", self::n(800000)),
                ("refinement", self::n(7300)),
                ("tabu_tenure", self::n(8)),
                ("ils_iterations", self::n(10)),
                ("post_ils_polish", self::n(200)),
                ("post_refinement", self::n(128)),
                ("ils_quick_refine", self::n(100)),
            ]);
            track_t23::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        50000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("clusters", self::n(64)),
                ("move_limit", self::n(960000)),
                ("refinement", self::n(9800)),
                ("tabu_tenure", self::n(8)),
                ("ils_iterations", self::n(10)),
                ("post_ils_polish", self::n(100)),
                ("post_refinement", self::n(64)),
                ("ils_quick_refine", self::n(50)),
                ("perturb_strength", self::n(5)),
            ]);
            track_t25::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        100000 => {
            track_t22::solve(challenge, save_solution, hyperparameters, module, stream, prop)
        }
        200000 => {
            let hp = merge_hp(hyperparameters, vec![
                ("effort", self::n(5)),
                ("clusters", self::n(80)),
                ("move_limit", self::n(500000)),
                ("refinement", self::n(9000)),
                ("tabu_tenure", self::n(8)),
                ("post_ils_polish", self::n(0)),
                ("post_refinement", self::n(2)),
                ("ils_quick_refine", self::n(0)),
                ("ils_iterations", self::n(5)),
            ]);
            track_t24::solve(challenge, save_solution, &hp, module, stream, prop)
        }
        _ => track_t21::solve(challenge, save_solution, hyperparameters, module, stream, prop),
    }
}

pub fn help() {
    println!("algo");
}
