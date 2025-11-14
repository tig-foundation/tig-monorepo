use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut rng = StdRng::from_seed(challenge.seed);
    let num_variables = challenge.num_variables;
    let mut variables: Vec<bool> = (0..num_variables).map(|_| rng.gen::<bool>()).collect();

    // Pre-generate a bunch of random integers
    // IMPORTANT! When generating random numbers, never use usize! usize bytes varies from system to system
    let rand_ints: Vec<usize> = (0..2 * num_variables)
        .map(|_| rng.gen_range(0..1_000_000_000u32) as usize)
        .collect();

    for i in 0..num_variables {
        // Evaluate clauses and find any that are unsatisfied
        let substituted: Vec<bool> = challenge
            .clauses
            .iter()
            .map(|clause| {
                clause.iter().any(|&literal| {
                    let var_idx = literal.abs() as usize - 1;
                    let var_value = variables[var_idx];
                    (literal > 0 && var_value) || (literal < 0 && !var_value)
                })
            })
            .collect();

        let unsatisfied_clauses: Vec<usize> = substituted
            .iter()
            .enumerate()
            .filter_map(|(idx, &satisfied)| if !satisfied { Some(idx) } else { None })
            .collect();

        let num_unsatisfied_clauses = unsatisfied_clauses.len();
        if num_unsatisfied_clauses == 0 {
            break;
        }

        // Flip the value of a random variable from a random unsatisfied clause
        let rand_unsatisfied_clause_idx = rand_ints[2 * i] % num_unsatisfied_clauses;
        let rand_unsatisfied_clause = unsatisfied_clauses[rand_unsatisfied_clause_idx];
        let rand_variable_idx = rand_ints[2 * i + 1] % 3;
        let rand_variable =
            challenge.clauses[rand_unsatisfied_clause][rand_variable_idx].abs() as usize - 1;
        variables[rand_variable] = !variables[rand_variable];
    }

    let _ = save_solution(&Solution { variables });
    return Ok(());
}