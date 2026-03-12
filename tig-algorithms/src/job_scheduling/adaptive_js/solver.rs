use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocessing::build_pre;
use super::greedy::run_simple_greedy_baseline;
use super::detect::{detect_track, DetectedTrack};
use super::track_strict;
use super::track_parallel;
use super::track_random;
use super::track_complex;
use super::track_chaotic;

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("num_restarts") {
            if let Some(v) = n.as_u64() {
                return EffortConfig::from_value(v as usize);
            }
        }
        if let Some(Value::String(s)) = map.get("effort") {
            return EffortConfig::from_str(s);
        }
    }
    EffortConfig::default_effort()
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;

    let pre = build_pre(challenge)?;
    let track = detect_track(challenge);
    let effort = parse_effort(hyperparameters);

    match track {
        DetectedTrack::Strict => {
            track_strict::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::Parallel => {
            track_parallel::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::Random => {
            track_random::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::Complex => {
            track_complex::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
        DetectedTrack::Chaotic => {
            track_chaotic::solve(challenge, save_solution, &pre, greedy_sol, greedy_mk, &effort)
        }
    }
}

pub fn help() {
    println!("Job Scheduling Solver - Track-Specialized Architecture");
    println!();
    println!("DESCRIPTION:");
    println!("  Multi-phase hybrid algorithm with track detection and isolated solvers.");
    println!("  Automatically detects instance type and dispatches to a track-specific");
    println!("  solver for maximum quality.");
    println!();
    println!("TRACK DETECTION:");
    println!("  Strict    -> flow_shop (reentrant flow shop)");
    println!("  Parallel  -> hybrid_flow_shop (parallel machines)");
    println!("  Random    -> job_shop (flexible job shop, low flexibility)");
    println!("  Complex   -> fjsp_medium (medium flexibility FJSP)");
    println!("  Chaotic   -> fjsp_high (high flexibility FJSP)");
    println!();
    println!("HYPERPARAMETERS:");
    println!("  effort: \"default\" | \"medium\" | \"high\" | \"extreme\"");
    println!("    default:    500 restarts");
    println!("    medium:   1,000 restarts");
    println!("    high:     1,500 restarts");
    println!("    extreme:  2,000 restarts");
    println!("  num_restarts: <integer>  (overrides effort, any value)");
    println!();
    println!("USAGE:");
    println!("  null                        -> default effort");
    println!("  '{{\"effort\":\"high\"}}'   -> high effort");
}
