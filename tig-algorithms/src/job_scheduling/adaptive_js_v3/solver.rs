use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::flow_shop;
use super::hybrid_flow_shop;
use super::job_shop;
use super::fjsp_medium;
use super::fjsp_high;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Track {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Track {
    if let Some(map) = hyperparameters {
        if let Some(Value::String(s)) = map.get("track") {
            return match s.to_lowercase().as_str() {
                "flow_shop" | "flow" => Track::FlowShop,
                "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                "job_shop" | "job" => Track::JobShop,
                "fjsp_medium" | "medium" => Track::FjspMedium,
                "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                _ => Track::FjspHigh,
            };
        }
    }
    Track::FjspHigh
}

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    let mut cfg = EffortConfig::default_effort();
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("job_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_job_shop_iters(v as usize);
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
    }
    cfg
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    let track = parse_track(hyperparameters);
    let effort = parse_effort(hyperparameters);

    match track {
        Track::FlowShop => {
            flow_shop::solve(challenge, save_solution, &pre, &effort)
        }
        Track::HybridFlowShop => {
            hybrid_flow_shop::solve(challenge, save_solution, &pre, &effort)
        }
        Track::JobShop => {
            job_shop::solve(challenge, save_solution, &pre, &effort)
        }
        Track::FjspMedium => {
            fjsp_medium::solve(challenge, save_solution, &pre, &effort)
        }
        Track::FjspHigh => {
            fjsp_high::solve(challenge, save_solution, &pre, &effort)
        }
    }
}

pub fn help() {
    println!("Job Scheduling Solver - Modular Independent Track Architecture v1");
    println!();
    println!("DESCRIPTION:");
    println!("  Each track is fully self-contained. Changing one track file cannot affect any other.");
    println!();
    println!("HYPERPARAMETERS:");
    println!("  track (required): \"flow_shop\" | \"hybrid_flow_shop\" | \"job_shop\" | \"fjsp_medium\" | \"fjsp_high\"");
    println!();
    println!("  job_shop_iters:          integer, default 10000, max 100000  (tabu search depth)");
    println!("  hybrid_flow_shop_iters:  integer, default 2000,  max 50000");
    println!("  fjsp_medium_iters:       integer, default 2000,  max 50000");
    println!();
    println!("NOTES:");
    println!("  job_shop scales well with more iterations - higher values give meaningfully better quality.");
    println!("  hybrid_flow_shop and fjsp_medium see diminishing returns beyond their defaults.");
    println!("  flow_shop and fjsp_high are not affected by their respective iters parameters.");
}
