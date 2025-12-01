// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;
use std::collections::HashMap;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let num_variables = challenge.num_variables;
    let mut assignment = vec![None; num_variables];

    if dpll(&challenge.clauses, &mut assignment) {
        let variables = assignment.into_iter().map(|x| x.unwrap_or(false)).collect();
        let _ = save_solution(&Solution { variables });
        return Ok(());
    } else {
        Ok(())
    }
}

fn dpll(clauses: &[Vec<i32>], assignment: &mut [Option<bool>]) -> bool {
    if clauses.is_empty() {
        return true;
    }

    if clauses.iter().any(|clause| clause.is_empty()) {
        return false;
    }

    let unit_clauses: Vec<&Vec<i32>> = clauses.iter().filter(|clause| clause.len() == 1).collect();
    for unit in unit_clauses {
        let literal = unit[0];
        let var = literal.abs() as usize - 1;
        let value = literal > 0;
        if assignment[var].is_some() && assignment[var] != Some(value) {
            return false;
        }
        assignment[var] = Some(value);
    }

    let mut pure_literals: HashMap<i32, bool> = HashMap::new();
    for clause in clauses {
        for &literal in clause {
            let var = literal.abs();
            if !pure_literals.contains_key(&var) {
                pure_literals.insert(var, literal > 0);
            }
        }
    }
    for (&var, &value) in pure_literals.iter() {
        let index = var as usize - 1;
        if assignment[index].is_none() {
            assignment[index] = Some(value);
        }
    }

    let first_unassigned = assignment.iter().position(|&x| x.is_none()).unwrap();
    assignment[first_unassigned] = Some(true);
    if dpll(clauses, assignment) {
        return true;
    }
    assignment[first_unassigned] = Some(false);
    if dpll(clauses, assignment) {
        return true;
    }
    assignment[first_unassigned] = None;
    false
}

pub fn help() {
    println!("No help information available.");
}
