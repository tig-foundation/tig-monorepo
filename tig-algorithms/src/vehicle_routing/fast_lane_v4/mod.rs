mod instance;
mod config;
mod route_eval;
mod solution;
mod builder;
mod operators;
mod gene_pool;
mod evolution;
mod runner;

pub use runner::Solver;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match Solver::solve_challenge_instance(challenge, hyperparameters, Some(save_solution))? {
        Some(solution) => {
            let _ = save_solution(&solution);
            Ok(())
        }
        None => Ok(()),
    }
}

pub fn help() {
    println!("Fast Lane v4: Hybrid Genetic Algorithm with Route-Based Crossover");
    println!("");
    println!("RECOMMENDED SETTINGS:");
    println!("");
    println!("For best quality:      {{\"exploration_level\": 4}}");
    println!("For balanced quality:  {{\"exploration_level\": 3}}");
    println!("For fastest runtime:   {{\"exploration_level\": 0}} or null");
    println!("");
    println!("EXPLORATION LEVELS (0-4):");
    println!("  0: Single LS pass, fastest");
    println!("  1: Multi-start LS, more initial diversity");
    println!("  2: Very short HGS (50 iterations)");
    println!("  3: Balanced HGS (500 iterations, recommended)");
    println!("  4: Deep HGS (5,000 iterations, maximum quality)");
    println!("  Note: levels above 4 fall back to level 4");
    println!("");
    println!("KEY ALGORITHMIC IMPROVEMENTS OVER v3:");
    println!("  • Greedy tour mutation: deterministic best-improvement relocate");
    println!("  • Pareto-dominance split: multi-dimensional DP pruning");
    println!("  • Flat distance matrix: faster cache-friendly lookups");
    println!("  • ALNS: adaptive large neighbourhood search perturbation");
}
