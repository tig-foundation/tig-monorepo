// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
pub mod types;
pub mod preprocess;
pub mod infra;
pub mod detect;
pub mod flow_shop;
pub mod job_shop;
pub mod fjsp_high;
pub mod our_search;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use types::EffortConfig;
use preprocess::build_pre;
use infra::run_simple_greedy_baseline;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub num_restarts: Option<u64>,
    #[serde(default)]
    pub job_shop_iters: Option<u64>,
    #[serde(default)]
    pub flow_shop_iters: Option<u64>,
    #[serde(default)]
    pub fjsp_high_iters: Option<u64>,
}

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    let mut cfg = EffortConfig::default_effort();
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("num_restarts") {
            if let Some(v) = n.as_u64() {
                cfg = EffortConfig::from_value(v as usize);
            }
        }
        if let Some(Value::String(s)) = map.get("effort") {
            cfg = EffortConfig::from_str(s);
        }
        if let Some(Value::Number(n)) = map.get("job_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_job_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_flow_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_high_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_high_iters(v as usize);
            }
        }
    }
    cfg
}

#[derive(Debug)]
enum Track { FlowShop, HybridFlowShop, JobShop, FjspMedium, FjspHigh }

fn detect_track_simple(challenge: &Challenge) -> Track {
    let mut total_flex = 0usize;
    let mut total_ops = 0usize;
    for p in 0..challenge.product_processing_times.len() {
        for op in &challenge.product_processing_times[p] {
            total_flex += op.len();
            total_ops += 1;
        }
    }
    let flex_avg = if total_ops > 0 { total_flex as f64 / total_ops as f64 } else { 1.0 };

    let mut max_ops = 0usize;
    let mut min_ops = usize::MAX;
    for p in 0..challenge.product_processing_times.len() {
        let nops = challenge.product_processing_times[p].len();
        if nops > max_ops { max_ops = nops; }
        if nops < min_ops { min_ops = nops; }
    }
    let uniform_routing = max_ops == min_ops;

    if flex_avg > 5.0 {
        Track::FjspHigh
    } else if flex_avg > 1.5 && !uniform_routing {
        Track::FjspMedium
    } else if flex_avg > 1.5 {
        Track::HybridFlowShop
    } else if uniform_routing {
        Track::FlowShop
    } else {
        Track::JobShop
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let track = detect_track_simple(challenge);

    match track {
        Track::FjspMedium | Track::HybridFlowShop => {
            our_search::solve_our(challenge, save_solution)
        }
        _ => {
            let effort = parse_effort(hyperparameters);
            let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
            save_solution(&greedy_sol)?;
            let pre = build_pre(challenge)?;
            let j2_track = detect::detect_track(&pre);

            match j2_track {
                detect::DetectedTrack::FlowShop => flow_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort),
                detect::DetectedTrack::JobShop => job_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort),
                detect::DetectedTrack::FjspHigh => fjsp_high::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort),
                detect::DetectedTrack::FjspMedium => job_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort),
                detect::DetectedTrack::HybridFlowShop => job_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort),
            }
        }
    }
}

pub fn help() {
    println!("job_tree - Hybrid FJSP solver");
    println!("  FlowShop/JobShop/FjspHigh: specialized per-track solvers");
    println!("  FjspMedium/HybridFlowShop: N7 tabu search with k-insertion");
}
