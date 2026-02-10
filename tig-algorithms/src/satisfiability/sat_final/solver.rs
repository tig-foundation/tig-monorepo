use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use super::problem::detect_track;
use super::phase_transition::solve_phase_transition_impl;
use super::clause_activity::solve_track_2_clause_activity_impl;
use super::track3::solve_track_3_impl;
use super::low_density::solve_low_density_impl;
use super::critical::solve_track_5_impl;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let track_id = detect_track(challenge);

    match track_id {
        1 => solve_track_1(challenge, save_solution, hyperparameters),
        2 => solve_track_2(challenge, save_solution, hyperparameters),
        3 => solve_track_3(challenge, save_solution, hyperparameters),
        4 => solve_track_4(challenge, save_solution, hyperparameters),
        5 => solve_track_5(challenge, save_solution, hyperparameters),
        _ => {
            eprintln!("[SAT TEST] WARNING: Unknown track detected, using Track 1 defaults");
            solve_track_1(challenge, save_solution, hyperparameters)
        }
    }
}

fn solve_track_1(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let max_fuel = hyperparameters
        .as_ref()
        .and_then(|h| h.get("max_fuel"))
        .and_then(|v| v.as_f64())
        .unwrap_or(15_000_000_000.0)
        .min(500_000_000_000.0);

    let n = challenge.num_variables;
    let density = challenge.clauses.len() as f64 / n as f64;

    let base_prob = 0.52;
    let check_interval = (50.0 * (1.0 + (density / 3.0).ln().max(0.0))).max(20.0) as usize;
    let max_restarts: u8 = 3;
    let restart_after = (1_500_000usize).max((n as usize).saturating_mul(150));
    let endgame_threshold = 24usize;

    solve_phase_transition_impl(
        challenge,
        save_solution,
        base_prob,
        check_interval,
        max_restarts,
        max_fuel,
        restart_after,
        endgame_threshold,
        true,
    )
}

fn solve_track_2(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let max_fuel = hyperparameters
        .as_ref()
        .and_then(|h| h.get("max_fuel"))
        .and_then(|v| v.as_f64())
        .unwrap_or(20_000_000_000.0)
        .min(500_000_000_000.0);

    let n = challenge.num_variables;
    let density = challenge.clauses.len() as f64 / n as f64;

    let base_prob = 0.51;
    let check_interval = (50.0 * (1.0 + (density / 3.0).ln().max(0.0))).max(20.0) as usize;
    let max_restarts: u8 = 2;
    let restart_after = (1_500_000usize).max((n as usize).saturating_mul(150));
    let endgame_threshold = 6usize;

    solve_track_2_clause_activity_impl(
        challenge,
        save_solution,
        base_prob,
        check_interval,
        max_restarts,
        max_fuel,
        restart_after,
        endgame_threshold,
    )
}

fn solve_track_3(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let max_fuel = hyperparameters
        .as_ref()
        .and_then(|h| h.get("max_fuel"))
        .and_then(|v| v.as_f64())
        .unwrap_or(20_000_000_000.0)
        .min(500_000_000_000.0);

    let n = challenge.num_variables;
    let density = challenge.clauses.len() as f64 / n as f64;

    let base_prob = 0.52;
    let check_interval = (50.0 * (1.0 + (density / 3.0).ln().max(0.0))).max(20.0) as usize;
    let max_restarts: u8 = 3;
    let restart_after = (1_500_000usize).max((n as usize).saturating_mul(150));
    let endgame_threshold = 24usize;

    solve_track_3_impl(
        challenge,
        save_solution,
        base_prob,
        check_interval,
        max_restarts,
        max_fuel,
        restart_after,
        endgame_threshold,
    )
}

fn solve_track_4(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pad = 1.8f32;
    let nad = 0.56f32;
    solve_low_density_impl(challenge, save_solution, pad, nad)
}

fn solve_track_5(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    solve_track_5_impl(challenge, save_solution, hyperparameters)
}

#[allow(dead_code)]
pub fn help() {
    println!("Track-Specific SAT Solver (5 Tracks)");   
}
