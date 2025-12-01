// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};
use std::collections::{HashSet, VecDeque};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut rng =
        StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.num_variables;
    let max_flips = 1000;
    let initial_noise: f64 = 0.3;
    let mut noise: f64 = initial_noise;
    let mut variables: Vec<bool> = (0..num_variables).map(|_| rng.gen()).collect();
    let mut best_solution: Option<Solution> = None;
    let mut best_unsatisfied = usize::MAX;
    let mut clause_weights: Vec<usize> = vec![1; challenge.clauses.len()];
    let mut unsatisfied_clauses: HashSet<usize> = challenge
        .clauses
        .iter()
        .enumerate()
        .filter_map(|(i, clause)| {
            if !clause_satisfied(clause, &variables) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let mut tabu_list: VecDeque<usize> = VecDeque::with_capacity(10);
    let mut tabu_tenure = 10;

    for flip in 0..max_flips {
        let num_unsatisfied = unsatisfied_clauses.len();

        // Update the best solution found so far
        if num_unsatisfied < best_unsatisfied {
            best_unsatisfied = num_unsatisfied;
            best_solution = Some(Solution {
                variables: variables.clone(),
            });

            if num_unsatisfied == 0 {
                let _ = save_solution(best_solution.as_ref().unwrap());
            }
        }

        // Adaptive noise adjustment based on progress
        if num_unsatisfied == best_unsatisfied {
            noise = (noise + 0.005).min(1.0);
        } else {
            noise = (noise - 0.01).max(0.1);
        }

        // Dynamic adjustment of tabu tenure
        tabu_tenure = (num_unsatisfied as f64 / best_unsatisfied as f64 * 10.0)
            .max(5.0)
            .min(20.0) as usize;

        // Choose an unsatisfied clause
        if let Some(&clause_idx) = unsatisfied_clauses.iter().choose(&mut rng) {
            let clause = &challenge.clauses[clause_idx];

            // Flip a variable with a heuristic
            if rng.gen::<f64>() < noise {
                // Random flip
                if let Some(&literal) = clause.iter().choose(&mut rng) {
                    let var_idx = literal.abs() as usize - 1;
                    if !tabu_list.contains(&var_idx) {
                        variables[var_idx] = !variables[var_idx];
                        update_unsatisfied_clauses(
                            &mut unsatisfied_clauses,
                            &challenge.clauses,
                            var_idx,
                            &variables,
                        );
                        tabu_list.push_back(var_idx);
                        if tabu_list.len() > tabu_tenure {
                            tabu_list.pop_front();
                        }
                    }
                }
            } else {
                // Greedy flip
                let mut best_var = None;
                let mut best_reduction = usize::MAX;
                for &literal in clause {
                    let var_idx = literal.abs() as usize - 1;
                    if tabu_list.contains(&var_idx) {
                        continue;
                    }
                    variables[var_idx] = !variables[var_idx]; // Tentative flip
                    let reduction = challenge
                        .clauses
                        .iter()
                        .enumerate()
                        .filter(|(i, clause)| {
                            !clause_satisfied(clause, &variables) && clause_weights[*i] > 0
                        })
                        .count();
                    variables[var_idx] = !variables[var_idx]; // Revert flip

                    if reduction < best_reduction {
                        best_reduction = reduction;
                        best_var = Some(var_idx);
                    }
                }

                if let Some(var_idx) = best_var {
                    variables[var_idx] = !variables[var_idx];
                    update_unsatisfied_clauses(
                        &mut unsatisfied_clauses,
                        &challenge.clauses,
                        var_idx,
                        &variables,
                    );
                    tabu_list.push_back(var_idx);
                    if tabu_list.len() > tabu_tenure {
                        tabu_list.pop_front();
                    }
                }
            }

            // Adaptive weight update for unsatisfied clauses
            for &clause_idx in &unsatisfied_clauses {
                clause_weights[clause_idx] +=
                    1 + (best_unsatisfied.saturating_sub(num_unsatisfied)) / best_unsatisfied;
            }
        }
    }

    let _ = save_solution(best_solution.as_ref().unwrap());
    Ok(())
}

fn clause_satisfied(clause: &[i32], variables: &[bool]) -> bool {
    clause.iter().any(|&literal| {
        let var_idx = literal.abs() as usize - 1;
        (literal > 0 && variables[var_idx]) || (literal < 0 && !variables[var_idx])
    })
}

fn update_unsatisfied_clauses(
    unsatisfied_clauses: &mut HashSet<usize>,
    clauses: &[Vec<i32>],
    flipped_var: usize,
    variables: &[bool],
) {
    for (i, clause) in clauses.iter().enumerate() {
        if clause
            .iter()
            .any(|&literal| literal.abs() as usize - 1 == flipped_var)
        {
            if clause_satisfied(clause, variables) {
                unsatisfied_clauses.remove(&i);
            } else {
                unsatisfied_clauses.insert(i);
            }
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
