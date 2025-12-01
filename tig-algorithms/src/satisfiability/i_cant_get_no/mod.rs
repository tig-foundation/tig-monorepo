use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.num_variables;
    let mut variables = vec![false; num_variables];
    variables.iter_mut().for_each(|v| *v = rng.gen());

    let max_iterations = 50 * num_variables;
    let max_flips_without_improvement = num_variables / 2;
    let mut best_unsatisfied_clauses = challenge.clauses.len();
    let mut flips_without_improvement = 0;

    for _ in 0..max_iterations {
        let unsatisfied_clause_idx = challenge.clauses.iter().position(|clause| {
            !clause.iter().any(|&literal| {
                let var_idx = (literal.abs() - 1) as usize;
                let var_value = variables[var_idx];
                (literal > 0 && var_value) || (literal < 0 && !var_value)
            })
        });

        if let Some(idx) = unsatisfied_clause_idx {
            let mut best_var_idx = None;
            let mut min_unsatisfied_clauses = challenge.clauses.len();

            for &literal in &challenge.clauses[idx] {
                let var_idx = (literal.abs() - 1) as usize;
                variables[var_idx] = !variables[var_idx];

                let unsatisfied_clauses = challenge
                    .clauses
                    .iter()
                    .filter(|clause| {
                        !clause.iter().any(|&lit| {
                            let v_idx = (lit.abs() - 1) as usize;
                            let v_value = variables[v_idx];
                            (lit > 0 && v_value) || (lit < 0 && !v_value)
                        })
                    })
                    .count();

                if unsatisfied_clauses < min_unsatisfied_clauses {
                    min_unsatisfied_clauses = unsatisfied_clauses;
                    best_var_idx = Some(var_idx);
                }

                variables[var_idx] = !variables[var_idx];
            }

            if let Some(var_idx) = best_var_idx {
                variables[var_idx] = !variables[var_idx];
                if min_unsatisfied_clauses < best_unsatisfied_clauses {
                    best_unsatisfied_clauses = min_unsatisfied_clauses;
                    flips_without_improvement = 0;
                } else {
                    flips_without_improvement += 1;
                }
            } else {
                // If no variable flip improves the solution, randomly flip a variable
                let rand_var_idx = rng.gen_range(0..num_variables);
                variables[rand_var_idx] = !variables[rand_var_idx];
                flips_without_improvement = 0;
            }

            if flips_without_improvement >= max_flips_without_improvement {
                // Restart the search with a new random assignment
                variables.iter_mut().for_each(|v| *v = rng.gen());
                best_unsatisfied_clauses = challenge.clauses.len();
                flips_without_improvement = 0;
            }
        } else {
            let _ = save_solution(&Solution { variables });
            return Ok(());
        }
    }

    if best_unsatisfied_clauses < challenge.clauses.len() / 10 {
        let _ = save_solution(&Solution { variables });
        return Ok(());
    } else {
        Ok(())
    }
}

pub fn help() {
    println!("No help information available.");
}
