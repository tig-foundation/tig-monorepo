use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

use tig_challenges::job_scheduling::*;
use anyhow::Result;
use serde_json::{Map, Value};
use std::cell::RefCell;

pub mod types;
pub mod preprocess;
mod infra_shared;
pub mod solver;
pub mod flow_shop;
pub mod hybrid_flow_shop;
pub mod job_shop;
pub mod fjsp_medium;
pub mod fjsp_high;

pub use solver::help;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Track {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Option<Track> {
    let map = hyperparameters.as_ref()?;
    let Value::String(s) = map.get("track")? else { return None };
    Some(match s.to_lowercase().as_str() {
        "flow_shop" | "flow" => Track::FlowShop,
        "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
        "job_shop" | "job" => Track::JobShop,
        "fjsp_medium" | "medium" => Track::FjspMedium,
        "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
        "auto" => return None,
        _ => Track::FjspHigh,
    })
}

fn choose_track(pre: &types::Pre) -> Track {
    if pre.strict_route.is_some() || (pre.flow_route.is_some() && pre.jobshopness < 0.45) {
        return Track::FlowShop;
    }
    if pre.flow_like > 0.78 && pre.high_flex < 0.20 {
        return Track::HybridFlowShop;
    }
    if pre.jobshopness > 0.72 && pre.flex_avg <= 2.4 {
        return Track::JobShop;
    }
    if pre.high_flex >= 0.58 || pre.flex_avg >= 4.0 {
        return Track::FjspHigh;
    }
    Track::FjspMedium
}

fn solve_selected_track(
    track: Track,
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &types::Pre,
    effort: &types::EffortConfig,
) -> Result<()> {
    match track {
        Track::FlowShop => flow_shop::solve(challenge, save_solution, pre, effort),
        Track::HybridFlowShop => hybrid_flow_shop::solve(challenge, save_solution, pre, effort),
        Track::JobShop => job_shop::solve(challenge, save_solution, pre, effort),
        Track::FjspMedium => fjsp_medium::solve(challenge, save_solution, pre, effort),
        Track::FjspHigh => fjsp_high::solve(challenge, save_solution, pre, effort),
    }
}

fn portfolio_backup_tracks(pre: &types::Pre, primary: Track) -> Vec<Track> {
    match primary {
        Track::JobShop => {
            let mut backups = Vec::with_capacity(2);
            if pre.flex_avg > 1.85 || pre.high_flex > 0.14 {
                backups.push(Track::FjspMedium);
            }
            if pre.high_flex > 0.24 || pre.flex_avg > 2.55 {
                backups.push(Track::FjspHigh);
            }
            backups
        }
        Track::HybridFlowShop if pre.jobshopness > 0.60 && pre.flex_avg <= 3.2 => vec![Track::JobShop],
        Track::FjspMedium | Track::FjspHigh if pre.jobshopness > 0.62 && pre.flex_avg <= 3.2 => vec![Track::JobShop],
        Track::FlowShop if pre.jobshopness > 0.55 || pre.high_flex > 0.12 => vec![Track::JobShop],
        _ => Vec::new(),
    }
}

fn scaled_backup_effort(effort: &types::EffortConfig, primary: Track, backup: Track) -> types::EffortConfig {
    let mut scaled = *effort;
    let (num, den) = match (primary, backup) {
        (Track::JobShop, Track::FjspMedium) => (25usize, 100usize),
        (Track::JobShop, Track::FjspHigh) => (20usize, 100usize),
        (Track::FjspHigh, Track::JobShop) | (Track::FjspMedium, Track::JobShop) => (40usize, 100usize),
        (Track::FlowShop, Track::JobShop) | (Track::HybridFlowShop, Track::JobShop) => (45usize, 100usize),
        _ => (30usize, 100usize),
    };
    let scale = |v: usize| -> usize { ((v.saturating_mul(num)).max(den - 1)) / den };
    scaled.job_shop_iters = scale(scaled.job_shop_iters).max(100);
    scaled.hybrid_flow_shop_iters = scale(scaled.hybrid_flow_shop_iters).max(100);
    scaled.fjsp_medium_iters = scale(scaled.fjsp_medium_iters).max(100);
    scaled.fjsp_high_iters = scale(scaled.fjsp_high_iters).max(100);
    scaled
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = preprocess::build_pre(challenge)?;
    let track = parse_track(hyperparameters).unwrap_or_else(|| choose_track(&pre));
    let effort = {
        let mut cfg = types::EffortConfig::default_effort();
        if let Some(map) = hyperparameters {
            if let Some(Value::Number(n)) = map.get("job_shop_iters") {
                if let Some(v) = n.as_u64() {
                    cfg = cfg.with_job_shop_iters(v as usize);
                }
            }
        }
        cfg
    };
    let best_makespan = RefCell::new(None::<u32>);
    let best_solution = RefCell::new(None::<Solution>);
    let save_best = |sol: &Solution| -> Result<()> {
        let mk = challenge.evaluate_makespan(sol)?;
        let should_save = {
            let mut best = best_makespan.borrow_mut();
            if best.map_or(true, |cur| mk < cur) {
                *best = Some(mk);
                true
            } else {
                false
            }
        };
        if should_save {
            *best_solution.borrow_mut() = Some(Solution { job_schedule: sol.job_schedule.clone() });
            save_solution(sol)?;
        }
        Ok(())
    };

    if let Some(track) = parse_track(hyperparameters) {
        solve_selected_track(track, challenge, &save_best, &pre, &effort)?;
    } else {
        let primary = choose_track(&pre);
        solve_selected_track(primary, challenge, &save_best, &pre, &effort)?;
        for backup in portfolio_backup_tracks(&pre, primary) {
            let backup_effort = scaled_backup_effort(&effort, primary, backup);
            solve_selected_track(backup, challenge, &save_best, &pre, &backup_effort)?;
        }
    }

    if let Some(sol) = best_solution.borrow().as_ref() {
        save_solution(sol)?;
    }
    Ok(())
}

pub fn help() {
    println!("Prometheus solver");
}
