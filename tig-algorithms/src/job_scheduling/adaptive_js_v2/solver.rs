use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::infra::run_simple_greedy_baseline;
use super::detect::{detect_track, DetectedTrack};
use super::flow_shop;
use super::hybrid_flow_shop;
use super::job_shop;
use super::fjsp_medium;
use super::fjsp_high;

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
        if let Some(Value::Number(n)) = map.get("fjsp_medium_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_iters(v as usize);
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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;

    let pre = build_pre(challenge)?;
    let track = detect_track(&pre);
    let effort = parse_effort(hyperparameters);

    match track {
        DetectedTrack::FlowShop => {
            flow_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::HybridFlowShop => {
            hybrid_flow_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::JobShop => {
            job_shop::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::FjspMedium => {
            fjsp_medium::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::FjspHigh => {
            fjsp_high::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
    }
}

pub fn help() {
    println!("Job Scheduling Solver - Track-Specialized Architecture v2");
    println!();
    println!("DESCRIPTION:");
    println!("  Multi-phase hybrid algorithm with track detection and isolated solvers.");
    println!("  Automatically detects instance type and dispatches to a track-specific");
    println!("  solver for maximum quality.");
    println!();
    println!("TRACK DETECTION:");
    println!("  FlowShop       -> Strict flow routing, single machine per stage");
    println!("  HybridFlowShop -> Flow-like with parallel machines at stages");
    println!("  JobShop        -> Random routing, low flexibility");
    println!("  FjspMedium     -> Medium flexibility FJSP");
    println!("  FjspHigh       -> High flexibility / chaotic FJSP");
    println!();
    println!("HYPERPARAMETERS:");
    println!("  effort: \"default\" | \"medium\" | \"high\" | \"extreme\"");
    println!("    default:    500 restarts, 3000/2600/2000/2000/2000 iters (js/fs/hfs/fjspM/fjspH)");
    println!("    medium:   1,000 restarts, 4000/4000/3000/3000/3000 iters");
    println!("    high:     1,500 restarts, 5000/6000/4000/4000/4000 iters");
    println!("    extreme:  2,000 restarts, 6000/10000/5000/5000/5000 iters");
    println!("  num_restarts: <integer>  (1-20000, constructor restarts for all tracks)");
    println!("  job_shop_iters: <integer>  (100-50000, tabu search iters for job_shop)");
    println!("  flow_shop_iters: <integer>  (100-50000, iterated greedy iters for flow_shop)");
    println!("  hybrid_flow_shop_iters: <integer>  (100-50000, local search iters for hybrid_flow_shop)");
    println!("  fjsp_medium_iters: <integer>  (100-50000, local search iters for fjsp_medium)");
    println!("  fjsp_high_iters: <integer>  (100-50000, local search iters for fjsp_high)");
    println!();
    println!("USAGE:");
    println!("  null                        -> default effort");
    println!("  '{{\"effort\":\"high\"}}'   -> high effort");
}
