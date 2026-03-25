// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::infra::run_simple_greedy_baseline;
use super::detect;
use super::flow_shop;
use super::hybrid_flow_shop;
use super::job_shop;
use super::fjsp_medium;
use super::fjsp_high;
use super::our_search;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    #[serde(default)]
    pub track: Option<String>,
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub num_restarts: Option<u64>,
    #[serde(default)]
    pub job_shop_iters: Option<u64>,
    #[serde(default)]
    pub flow_shop_iters: Option<u64>,
    #[serde(default)]
    pub hybrid_flow_shop_iters: Option<u64>,
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
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_iters(v as usize);
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

#[derive(Debug, Clone, Copy)]
enum Track { FlowShop, HybridFlowShop, JobShop, FjspMedium, FjspHigh }

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Option<Track> {
    if let Some(map) = hyperparameters {
        if let Some(Value::String(s)) = map.get("track") {
            return Some(match s.to_lowercase().as_str() {
                "flow_shop" | "flow" => Track::FlowShop,
                "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                "job_shop" | "job" => Track::JobShop,
                "fjsp_medium" | "medium" => Track::FjspMedium,
                "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                _ => return None,
            });
        }
    }
    None
}

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
    // HP "track" overrides auto-detection — guarantees correct routing per track
    let track = parse_track(hyperparameters).unwrap_or_else(|| detect_track_simple(challenge));

    match track {
        Track::FjspMedium => {
            fjsp_medium::solve(challenge, save_solution)
        }
        Track::HybridFlowShop => {
            let effort = parse_effort(hyperparameters);
            let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
            save_solution(&greedy_sol)?;
            let pre = build_pre(challenge)?;
            hybrid_flow_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        Track::FlowShop => {
            let effort = parse_effort(hyperparameters);
            let pre = build_pre(challenge)?;
            flow_shop::solve(challenge, save_solution, &pre, &effort)
        }
        Track::JobShop => {
            let effort = parse_effort(hyperparameters);
            let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
            save_solution(&greedy_sol)?;
            let pre = build_pre(challenge)?;
            job_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        Track::FjspHigh => {
            let effort = parse_effort(hyperparameters);
            let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
            save_solution(&greedy_sol)?;
            let pre = build_pre(challenge)?;
            fjsp_high::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
    }
}

pub fn help() {
    println!("job_four - Per-track FJSP solver");
    println!();
    println!("HYPERPARAMETERS:");
    println!("  track: \"flow_shop\" | \"hybrid_flow_shop\" | \"job_shop\" | \"fjsp_medium\" | \"fjsp_high\"");
    println!("  effort: \"default\" | \"medium\" | \"high\" | \"extreme\"");
    println!("  job_shop_iters: integer (default 12000)");
    println!("  hybrid_flow_shop_iters: integer (default 12000)");
    println!("  fjsp_high_iters: integer (default 5000)");
}
