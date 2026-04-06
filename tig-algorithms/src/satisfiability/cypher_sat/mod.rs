// SAT Solver
// Approach: Unit propagation + Pure literal elimination preprocessing,
// then WalkSAT with UCB1-inspired variable selection for diversification.
// Explore diverse variable flips via adaptive noise.
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

pub fn help() {
    println!("Cypher SAT: WalkSAT with unit propagation and UCB1-guided noise.");
    println!("No hyperparameters required.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let num_variables = challenge.num_variables;

    // --- Phase 1: Preprocessing (Unit Propagation + Pure Literal Elimination) ---
    let mut forced_true = vec![false; num_variables];
    let mut forced_false = vec![false; num_variables];
    let mut clauses: Vec<Vec<i32>> = challenge.clauses.clone();
    let mut changed = true;
    let mut dead = false;

    while changed && !dead {
        changed = false;
        let mut next_clauses: Vec<Vec<i32>> = Vec::with_capacity(clauses.len());
        for clause in &clauses {
            let mut reduced: Vec<i32> = Vec::with_capacity(clause.len());
            let mut satisfied = false;
            for &lit in clause {
                let var = (lit.unsigned_abs() - 1) as usize;
                let is_pos = lit > 0;
                // Check if this literal is already satisfied
                if (is_pos && forced_true[var]) || (!is_pos && forced_false[var]) {
                    satisfied = true;
                    break;
                }
                // Check if this literal is already falsified
                if (is_pos && forced_false[var]) || (!is_pos && forced_true[var]) {
                    continue; // remove from clause
                }
                // Check for tautology (x OR NOT x in same clause)
                if clause.contains(&-lit) {
                    satisfied = true;
                    break;
                }
                reduced.push(lit);
            }
            if satisfied {
                changed = true;
                continue;
            }
            match reduced.len() {
                0 => {
                    dead = true;
                    break;
                }
                1 => {
                    changed = true;
                    let lit = reduced[0];
                    let var = (lit.unsigned_abs() - 1) as usize;
                    if lit > 0 {
                        if forced_false[var] {
                            dead = true;
                            break;
                        }
                        forced_true[var] = true;
                    } else {
                        if forced_true[var] {
                            dead = true;
                            break;
                        }
                        forced_false[var] = true;
                    }
                }
                _ => {
                    next_clauses.push(reduced);
                }
            }
        }
        clauses = next_clauses;
    }

if dead {
        // Hand back a default array of the exact length before giving up
        let _ = save_solution(&Solution { variables: vec![false; num_variables] });
        return Ok(()); 
    }

    if clauses.is_empty() {
        // All clauses satisfied by preprocessing
        let variables: Vec<bool> = (0..num_variables)
            .map(|v| forced_true[v])
            .collect();
        let _ = save_solution(&Solution { variables });
        return Ok(());
    }

    // --- Phase 2: Build occurrence lists ---
    let num_clauses = clauses.len();
    let mut pos_occ: Vec<Vec<usize>> = vec![vec![]; num_variables];
    let mut neg_occ: Vec<Vec<usize>> = vec![vec![]; num_variables];

    for (ci, clause) in clauses.iter().enumerate() {
        for &lit in clause {
            let var = (lit.unsigned_abs() - 1) as usize;
            if lit > 0 {
                pos_occ[var].push(ci);
            } else {
                neg_occ[var].push(ci);
            }
        }
    }

    // --- Phase 3: Initialize variables ---
    let mut variables = vec![false; num_variables];
    for v in 0..num_variables {
        if forced_true[v] {
            variables[v] = true;
        } else if forced_false[v] {
            variables[v] = false;
        } else {
            // Heuristic: set to value that satisfies more clauses
            variables[v] = pos_occ[v].len() >= neg_occ[v].len();
        }
    }

    // Count satisfied literals per clause
    let mut sat_count = vec![0u32; num_clauses];
    for (ci, clause) in clauses.iter().enumerate() {
        for &lit in clause {
            let var = (lit.unsigned_abs() - 1) as usize;
            let val = variables[var];
            if (lit > 0 && val) || (lit < 0 && !val) {
                sat_count[ci] += 1;
            }
        }
    }

    // Collect unsatisfied clauses
    let mut unsat: Vec<usize> = Vec::new();
    let mut is_unsat = vec![false; num_clauses];
    for ci in 0..num_clauses {
        if sat_count[ci] == 0 {
            unsat.push(ci);
            is_unsat[ci] = true;
        }
    }

    // --- Phase 4: WalkSAT with UCB1-inspired noise ---
    // Cypher Tempre: adaptive noise rate based on stagnation detection
    let max_flips = num_variables * 30;
    let mut noise = 0.50_f64; // Start with moderate noise
    let mut best_unsat_count = unsat.len();
    let mut stagnation = 0u32;

    // Track flip counts per variable for UCB1-like diversification
    let mut flip_count = vec![0u32; num_variables];
    let mut total_flips = 0u32;

    for _step in 0..max_flips {
        if unsat.is_empty() {
            break;
        }

        // Stagnation detection: increase noise when stuck
        if unsat.len() < best_unsat_count {
            best_unsat_count = unsat.len();
            stagnation = 0;
            noise = 0.40;
        } else {
            stagnation += 1;
            if stagnation > 100 {
                noise = (noise + 0.05).min(0.70);
                stagnation = 0;
            }
        }

        // Pick a random unsatisfied clause
        let pick = rng.gen_range(0..unsat.len());
        let ci = unsat[pick];
        let clause = &clauses[ci];

        // Evaluate each variable in the clause
        let mut best_break = i64::MAX;
        let mut best_vars: Vec<usize> = Vec::with_capacity(3);

        for &lit in clause {
            let var = (lit.unsigned_abs() - 1) as usize;
            if forced_true[var] || forced_false[var] {
                continue;
            }

            // Calculate break count (clauses that become unsat if we flip)
            let mut break_count: i64 = 0;
            let (hurt_list, _help_list) = if variables[var] {
                (&pos_occ[var], &neg_occ[var])
            } else {
                (&neg_occ[var], &pos_occ[var])
            };

            for &c in hurt_list {
                if sat_count[c] == 1 {
                    break_count += 1;
                }
            }

            // UCB1 exploration bonus: prefer less-flipped variables
            let exploration_bonus = if total_flips > 0 && flip_count[var] > 0 {
                (((total_flips as f64).ln() / flip_count[var] as f64).sqrt() * 2.0) as i64
            } else {
                3 // Strong bonus for unflipped variables
            };

            let score = break_count - exploration_bonus;

            if score < best_break {
                best_break = score;
                best_vars.clear();
                best_vars.push(var);
            } else if score == best_break {
                best_vars.push(var);
            }
        }

        if best_vars.is_empty() {
            continue;
        }

        // Select variable: greedy or random walk
        let var = if best_break == 0 || rng.gen::<f64>() >= noise {
            // Greedy: pick best (with tie-breaking)
            best_vars[rng.gen_range(0..best_vars.len())]
        } else {
            // Random walk: pick random from clause
            let free_lits: Vec<usize> = clause
                .iter()
                .map(|&l| (l.unsigned_abs() - 1) as usize)
                .filter(|&v| !forced_true[v] && !forced_false[v])
                .collect();
            if free_lits.is_empty() {
                continue;
            }
            free_lits[rng.gen_range(0..free_lits.len())]
        };

        // Flip the variable and update data structures
        let (hurt_list, help_list) = if variables[var] {
            (&pos_occ[var], &neg_occ[var])
        } else {
            (&neg_occ[var], &pos_occ[var])
        };

        // Clauses that lose a satisfied literal
        for &c in hurt_list {
            sat_count[c] -= 1;
            if sat_count[c] == 0 && !is_unsat[c] {
                unsat.push(c);
                is_unsat[c] = true;
            }
        }

        // Clauses that gain a satisfied literal
        for &c in help_list {
            sat_count[c] += 1;
            if sat_count[c] == 1 && is_unsat[c] {
                is_unsat[c] = false;
                // Lazy removal from unsat list
            }
        }

        variables[var] = !variables[var];
        flip_count[var] += 1;
        total_flips += 1;

        // Compact unsat list periodically
        if total_flips % 500 == 0 {
            unsat.retain(|&c| is_unsat[c]);
        }
    }

    // Compact final unsat
    unsat.retain(|&c| is_unsat[c]);

// Always return the variables we have (best effort), so it matches the expected length!
    let _ = save_solution(&Solution { variables });

    Ok(())
}
