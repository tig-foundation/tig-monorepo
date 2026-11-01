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

fn classic_job_shop(challenge: &Challenge) -> bool {
    if challenge.jobs_per_product.len() < 27 {
        return false;
    }
    let max_flex = challenge
        .product_processing_times
        .iter()
        .flat_map(|ops| ops.iter())
        .map(|op| op.len())
        .max()
        .unwrap_or(1);
    max_flex == 1
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
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_seeds") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_seeds(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_stagnation") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_stagnation(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_tabu_reassign_every") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_tabu_reassign_every(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_tabu_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_tabu_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_tabu_follow_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_tabu_follow_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_far_tabu_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_far_tabu_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hfs_tabu_fuel") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hfs_tabu_fuel(v);
            }
        }
    }
    cfg
}

const HYBRID_RESTARTS: usize = 100;

fn solve_hybrid(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &super::types::Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        if let Ok(mk) = challenge.evaluate_makespan(s) {
            if mk <= best_mk.get() {
                best_mk.set(mk);
                return save_solution(s);
            }
        }
        Ok(())
    };

    let effort = effort.with_hybrid_flow_shop_iters(HYBRID_RESTARTS);
    hybrid_flow_shop::solve_seeded(challenge, &guarded, pre, &effort, None)
}

fn common_route(challenge: &Challenge) -> bool {
    let Some(first) = challenge.product_processing_times.first() else { return true };
    let key = |op: &std::collections::HashMap<usize, u32>| {
        let mut ms: Vec<usize> = op.keys().copied().collect();
        ms.sort_unstable();
        ms
    };
    challenge.product_processing_times.iter().skip(1).all(|route| {
        route.len() == first.len() && route.iter().zip(first).all(|(a, b)| key(a) == key(b))
    })
}

fn flexible_common_route(challenge: &Challenge, pre: &super::types::Pre) -> bool {
    let max_flex = challenge
        .product_processing_times
        .iter()
        .flat_map(|ops| ops.iter())
        .map(|op| op.len())
        .max()
        .unwrap_or(1);
    max_flex > 1 && pre.high_flex <= 0.85 && common_route(challenge)
}

fn solve_engine_only(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let best_mk = std::cell::Cell::new(u32::MAX);
    let guarded = |s: &Solution| -> Result<()> {
        if let Ok(mk) = challenge.evaluate_makespan(s) {
            if mk <= best_mk.get() {
                best_mk.set(mk);
                return save_solution(s);
            }
        }
        Ok(())
    };
    super::fuel::run_bounded(u64::MAX, &guarded, |bound| {
        super::ember_hybrid_route::solve(challenge, bound)
    });
    Ok(())
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    let track = if classic_job_shop(challenge) {
        Track::JobShop
    } else {
        parse_track(hyperparameters)
    };
    let effort = parse_effort(hyperparameters);

    match track {
        Track::FlowShop => flow_shop::solve(challenge, save_solution, &pre, &effort),
        Track::HybridFlowShop => solve_hybrid(challenge, save_solution, &pre, &effort),
        Track::JobShop => job_shop::solve(challenge, save_solution, &pre, &effort),
        Track::FjspMedium | Track::FjspHigh if flexible_common_route(challenge, &pre) => {
            solve_engine_only(challenge, save_solution)
        }
        Track::FjspMedium => fjsp_medium::solve(challenge, save_solution, &pre, &effort),
        Track::FjspHigh => fjsp_high::solve(challenge, save_solution, &pre, &effort),
    }
}

pub fn help() {
    println!("ember_opt_v4 (c007 job_scheduling)");
    println!();
    println!("track (string):");
    println!("  selects which solver runs; each track is independent");
    println!("  accepted values:");
    println!("    \"flow_shop\" | \"flow\"");
    println!("    \"hybrid_flow_shop\" | \"hybrid\"");
    println!("    \"job_shop\" | \"job\"");
    println!("    \"fjsp_medium\" | \"medium\"");
    println!("    \"fjsp_high\" | \"high\" | \"fjsp\"");
    println!("  default if omitted or invalid: \"fjsp_high\"");
    println!("  classic JSSP (max_flex==1, n_products>=27) always routes to job_shop");
    println!();
    println!("job_shop_iters (integer):");
    println!("  affects track: job_shop (tabu search iteration budget)");
    println!("  range after clamp: 100..300000");
    println!("  default: 35000");
    println!();
    println!("hybrid_flow_shop_tabu_iters (integer):");
    println!("  affects track: hybrid_flow_shop (final CPDT tabu iteration budget)");
    println!("  range after clamp: 100..100000");
    println!("  default: 1800");
    println!();
    println!("hybrid_flow_shop_tabu_seeds (integer):");
    println!("  affects track: hybrid_flow_shop (number of final CPDT tabu seed solutions)");
    println!("  range after clamp: 1..16");
    println!("  default: 5");
    println!();
    println!("hybrid_flow_shop_tabu_stagnation (integer):");
    println!("  affects track: hybrid_flow_shop (final CPDT tabu stagnation/reset limit)");
    println!("  range after clamp: 20..10000");
    println!("  default: 400");
    println!();
    println!("hybrid_flow_shop_tabu_reassign_every (integer):");
    println!("  affects track: hybrid_flow_shop (frequency of final CPDT machine reassignment moves)");
    println!("  range after clamp: 1..32");
    println!("  default: 1");
    println!();
    println!("hfs_tabu_fuel (integer):");
    println!("  affects track: hybrid_flow_shop (fuel cap of the final graph tabu)");
    println!("  0 spends all remaining fuel; any other value is that fuel cap");
    println!("  default: 120B");
    println!();
    println!("notes:");
    println!("  hybrid: adaptive restarts, refine, bridge and tabu phases, then a graph tabu");
    println!("          (restarts fixed at 100; stops after 40B fuel without improvement)");
    println!("  flow_shop: heavy search skipped when fuel is already low");
    println!("  fjsp_medium / fjsp_high: fixed internal budgets; not tunable via hyperparameters");
    println!("  all other hyperparameter keys are ignored");
}
