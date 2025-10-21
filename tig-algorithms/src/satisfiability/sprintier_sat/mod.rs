use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);
    let num_variables = challenge.difficulty.num_variables;
    let mut variables = vec![false; num_variables];

    // Preprocessing
    let mut p_single = vec![false; num_variables];
    let mut n_single = vec![false; num_variables];
    let mut clauses: Vec<Vec<i32>> = Vec::with_capacity(challenge.clauses.len());

    for clause in &challenge.clauses {
        let mut new_clause: Vec<i32> = Vec::with_capacity(clause.len());
        let mut skip = false;
        let mut literals = std::collections::HashSet::new();

        for &l in clause {
            let var = (l.abs() - 1) as usize;
            if (p_single[var] && l > 0) || (n_single[var] && l < 0) {
                skip = true;
                break;
            } 
            if !p_single[var] && !n_single[var] && !literals.contains(&-l) {
                if !literals.insert(l) {
                    continue;
                }
                new_clause.push(l);
            }
        }

        if skip {
            continue;
        }

        match new_clause.len() {
            0 => return Ok(()),
            1 => {
                let l = new_clause[0];
                let var = (l.abs() - 1) as usize;
                if l > 0 {
                    if n_single[var] {
                        return Ok(());
                    }
                    p_single[var] = true;
                } else {
                    if p_single[var] {
                        return Ok(());
                    }
                    n_single[var] = true;
                }
            }
            _ => clauses.push(new_clause),
        }
    }

    // Initialize variables
    for (i, &p) in p_single.iter().enumerate() {
        variables[i] = if p {
            true
        } else if n_single[i] {
            false
        } else {
            rng.gen_bool(0.5)
        };
    }

    let num_clauses = clauses.len();
    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::new(); num_variables];
    let mut num_good_so_far = vec![0; num_clauses];

    // Precompute clause satisfaction and variable occurrences
    for (i, clause) in clauses.iter().enumerate() {
        for &l in clause {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
                if variables[var] {
                    num_good_so_far[i] += 1;
                }
            } else {
                n_clauses[var].push(i);
                if !variables[var] {
                    num_good_so_far[i] += 1;
                }
            }
        }
    }

    let mut residual: Vec<usize> = (0..num_clauses).filter(|&i| num_good_so_far[i] == 0).collect();
    let mut residual_indices = HashMap::new();
    for (i, &clause_index) in residual.iter().enumerate() {
        residual_indices.insert(clause_index, i);
    }

    for _ in 0..num_variables * 25 {
        if residual.is_empty() {
            let _ = save_solution(&Solution { variables });
            return Ok(());
        }

        let i = residual[0];
        let c = &clauses[i];
        
        let v = if rng.gen_bool(0.5) {
            let l = c[rng.gen_range(0..c.len())];
            (l.abs() - 1) as usize
        } else {
            c.iter()
                .map(|&l| {
                    let var = (l.abs() - 1) as usize;
                    let sad = if variables[var] {
                        p_clauses[var].iter().filter(|&&c| num_good_so_far[c] == 1).count()
                    } else {
                        n_clauses[var].iter().filter(|&&c| num_good_so_far[c] == 1).count()
                    };
                    (var, sad)
                })
                .min_by_key(|&(_, sad)| sad)
                .map(|(var, _)| var)
                .unwrap()
        };

        variables[v] = !variables[v];

        let (satisfied, unsatisfied) = if variables[v] {
            (&n_clauses[v], &p_clauses[v])
        } else {
            (&p_clauses[v], &n_clauses[v])
        };

        for &c in satisfied {
            num_good_so_far[c] += 1;
            if num_good_so_far[c] == 1 {
                if let Some(&index) = residual_indices.get(&c) {
                    residual.swap_remove(index);
                    if index < residual.len() {
                        residual_indices.insert(residual[index], index);
                    }
                    residual_indices.remove(&c);
                }
            }
        }

        for &c in unsatisfied {
            if num_good_so_far[c] == 1 {
                residual.push(c);
                residual_indices.insert(c, residual.len() - 1);
            }
            num_good_so_far[c] -= 1;
        }
    }

    Ok(())
}