use std::collections::{HashMap, HashSet};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let mut solution = Solution {
        variables: vec![false; challenge.num_variables],
    };
    let mut vars_map = HashMap::<i32, HashSet<usize>>::new();
    for (idx, clause) in challenge.clauses.iter().enumerate() {
        for v in clause.iter() {
            vars_map.entry(*v).or_insert(HashSet::new()).insert(idx);
        }
    }
    while !vars_map.is_empty() {
        let mut lens = vars_map
            .iter()
            .map(|v| (v.0.clone(), v.1.len()))
            .collect::<Vec<_>>();
        lens.sort_by(|a, b| b.1.cmp(&a.1));
        let s = &lens[0];
        if s.1 == 0 {
            break;
        }
        solution.variables[(s.0.abs() - 1) as usize] = s.0 > 0;
        let c = vars_map.remove(&s.0).unwrap();
        vars_map.remove(&-s.0);
        vars_map.retain(|_, v| {
            *v = v.difference(&c).cloned().collect();
            !v.is_empty()
        });
    }

    let _ = save_solution(&solution);
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
