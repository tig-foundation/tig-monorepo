// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.num_variables;
    let max_flips = 1000;

    let mut variables: Vec<bool> = (0..num_variables).map(|_| rng.gen::<bool>()).collect();

    for _ in 0..max_flips {
        let mut unsatisfied_clauses: Vec<&Vec<i32>> = challenge
            .clauses
            .iter()
            .filter(|clause| !clause_satisfied(clause, &variables))
            .collect();

        if unsatisfied_clauses.is_empty() {
            let _ = save_solution(&Solution { variables });
            return Ok(());
        }

        let clause = unsatisfied_clauses.choose(&mut rng).unwrap();
        let literal = clause.choose(&mut rng).unwrap();
        let var_idx = literal.abs() as usize - 1;
        variables[var_idx] = !variables[var_idx];
    }

    Ok(())
}

fn clause_satisfied(clause: &Vec<i32>, variables: &[bool]) -> bool {
    clause.iter().any(|&literal| {
        let var_idx = literal.abs() as usize - 1;
        (literal > 0 && variables[var_idx]) || (literal < 0 && !variables[var_idx])
    })
}

pub fn help() {
    println!("No help information available.");
}
