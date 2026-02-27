// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
use anyhow::Result;
use serde::{Deserialize, Serialize};
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
    pub hybrid_flow_shop_iters: Option<u64>,
    #[serde(default)]
    pub fjsp_medium_iters: Option<u64>,
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
    println!("job_two - Track-Specialized FJSP Solver");
    println!();
    println!("DESCRIPTION:");
    println!("  Multi-phase hybrid algorithm with automatic track detection.");
    println!("  Dispatches to dedicated per-track solvers with tailored");
    println!("  construction heuristics, tabu search, and local search.");
    println!();
    println!("TRACK DETECTION:");
    println!("  FlowShop       -> NEH + Taillard acceleration + Iterated Greedy + VND");
    println!("  HybridFlowShop -> Bandit construction + plateau LS + critical block moves");
    println!("  JobShop        -> Bandit construction + tabu search (12K iters)");
    println!("  FjspMedium     -> Bandit construction + hybrid tabu (swap + reassign)");
    println!("  FjspHigh       -> Bandit construction + critical block LS + greedy reassign");
    println!();
    println!("HYPERPARAMETERS:");
    println!("  effort: \"default\" | \"medium\" | \"high\" | \"extreme\"");
    println!("    default:  2,000 restarts, 12K/20K/5K/5K/5K iters (js/fs/hfs/fjspM/fjspH)");
    println!("    medium:   3,000 restarts, 10K/12K/8K/8K/8K iters");
    println!("    high:     4,000 restarts, 15K/18K/12K/12K/12K iters");
    println!("    extreme:  6,000 restarts, 20K/25K/15K/15K/15K iters");
    println!("  num_restarts: <integer>  (1-20000)");
    println!("  job_shop_iters: <integer>  (100-50000)");
    println!("  flow_shop_iters: <integer>  (100-50000)");
    println!("  hybrid_flow_shop_iters: <integer>  (100-50000)");
    println!("  fjsp_medium_iters: <integer>  (100-50000)");
    println!("  fjsp_high_iters: <integer>  (100-50000)");
}
