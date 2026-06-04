use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

use tig_challenges::satisfiability::*;
use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng, Rng};
use serde_json::{Map, Value};
use std::time::Instant;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let clauses = &challenge.clauses;
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
    let deadline = Instant::now() + std::time::Duration::from_secs(295);
    let p: f64 = 0.4;

    let mut assignment: Vec<bool> = (0..nv).map(|_| rng.gen()).collect();

    // WalkSAT loop
    while Instant::now() < deadline {
        // Find unsatisfied clauses
        let mut unsat: Vec<usize> = Vec::new();
        for (i, clause) in clauses.iter().enumerate() {
            let mut sat = false;
            for &lit in clause {
                let v = (lit.abs() - 1) as usize;
                let val = assignment[v];
                if (lit > 0 && val) || (lit < 0 && !val) {
                    sat = true;
                    break;
                }
            }
            if !sat {
                unsat.push(i);
            }
        }
        if unsat.is_empty() {
            let _ = save_solution(&Solution { variables: assignment });
            return Ok(());
        }
        // Pick random unsat clause
        let cidx = unsat[rng.gen_range(0..unsat.len())];
        let clause = &clauses[cidx];
        // Decide flip: random or min-break
        let flip_idx = if rng.gen::<f64>() < p {
            rng.gen_range(0..3)
        } else {
            // min break count
            let mut best = 0usize;
            let mut best_break = usize::MAX;
            for (j, &lit) in clause.iter().enumerate() {
                let v = (lit.abs() - 1) as usize;
                // tentative flip
                assignment[v] = !assignment[v];
                let mut brk = 0usize;
                for (k, cl) in clauses.iter().enumerate() {
                    if k == cidx { continue; }
                    let mut sat = false;
                    for &l2 in cl {
                        let vv = (l2.abs() - 1) as usize;
                        let val = assignment[vv];
                        if (l2 > 0 && val) || (l2 < 0 && !val) {
                            sat = true;
                            break;
                        }
                    }
                    if !sat { brk += 1; }
                }
                assignment[v] = !assignment[v];
                if brk < best_break {
                    best_break = brk;
                    best = j;
                }
            }
            best
        };
        let lit = clause[flip_idx];
        let v = (lit.abs() - 1) as usize;
        assignment[v] = !assignment[v];
    }
    // timeout, save current (may be partial)
    let _ = save_solution(&Solution { variables: assignment });
    Ok(())
}

pub fn help() {
    println!("Prometheus solver");
}
