/*!
Copyright 2024 YourMama

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use tig_challenges::satisfiability::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut random_gen = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let mut pos_single = vec![false; challenge.difficulty.num_variables];
    let mut neg_single = vec![false; challenge.difficulty.num_variables];

    let mut current_clauses = challenge.clauses.clone();
    let mut clause_vector: Vec<Vec<i32>> = Vec::with_capacity(current_clauses.len());

    let mut iteration_count = 0;

    let mut is_unsolvable = false;

    while !is_unsolvable {
        let mut all_done = true;
        for clause in &current_clauses {
            let mut filtered_clause: Vec<i32> = Vec::with_capacity(clause.len()); 
            let mut should_skip = false;
            for (idx, literal) in clause.iter().enumerate() {
                if (pos_single[(literal.abs() - 1) as usize] && *literal > 0)
                    || (neg_single[(literal.abs() - 1) as usize] && *literal < 0)
                    || clause[(idx + 1)..].contains(&-literal)
                {
                    should_skip = true;
                    break;
                } else if pos_single[(literal.abs() - 1) as usize]
                    || neg_single[(literal.abs() - 1) as usize]
                    || clause[(idx + 1)..].contains(&literal)
                {
                    all_done = false;
                    continue;
                } else {
                    filtered_clause.push(*literal);
                }
            }
            if should_skip {
                all_done = false;
                continue;
            }
            match filtered_clause[..] {
                [l] => {
                    all_done = false;
                    if l > 0 {
                        if neg_single[(l.abs() - 1) as usize] {
                            is_unsolvable = true;
                            break;
                        } else {
                            pos_single[(l.abs() - 1) as usize] = true;
                        }
                    } else {
                        if pos_single[(l.abs() - 1) as usize] {
                            is_unsolvable = true;
                            break;
                        } else {
                            neg_single[(l.abs() - 1) as usize] = true;
                        }
                    }
                }
                [] => {
                    is_unsolvable = true;
                    break;
                }
                _ => {
                    clause_vector.push(filtered_clause);
                }
            }
        }
        if all_done {
            break;
        } else {
            current_clauses = clause_vector;
            clause_vector = Vec::with_capacity(current_clauses.len());
        }
    }

    if is_unsolvable {
        return Ok(None);
    }

    let total_variables = challenge.difficulty.num_variables;
    let remaining_clauses = clause_vector.len();

    let mut pos_clause_refs: Vec<Vec<usize>> = vec![Vec::new(); total_variables];
    let mut neg_clause_refs: Vec<Vec<usize>> = vec![Vec::new(); total_variables];

    for clause in &clause_vector {
        for &literal in clause {
            let var_index = (literal.abs() - 1) as usize;
            if literal > 0 {
                if pos_clause_refs[var_index].capacity() == 0 {
                    pos_clause_refs[var_index] = Vec::with_capacity(clause_vector.len() / total_variables + 1);
                }
            } else {
                if neg_clause_refs[var_index].capacity() == 0 {
                    neg_clause_refs[var_index] = Vec::with_capacity(clause_vector.len() / total_variables + 1);
                }
            }
        }
    }

    for (clause_index, clause) in clause_vector.iter().enumerate() {
        for &literal in clause {
            let var_index = (literal.abs() - 1) as usize;
            if literal > 0 {
                pos_clause_refs[var_index].push(clause_index);
            } else {
                neg_clause_refs[var_index].push(clause_index);
            }
        }
    }

    let mut solution_variables = vec![false; total_variables];
    for variable in 0..total_variables {
        let pos_len = pos_clause_refs[variable].len();
        let neg_len = neg_clause_refs[variable].len();

        let var_ratio = pos_len as f32 / (pos_len + neg_len).max(1) as f32;

        if var_ratio >= 1.8 {
            solution_variables[variable] = true;
        } else if var_ratio <= 0.55 {
            solution_variables[variable] = false;
        } else {
            if pos_single[variable] {
                solution_variables[variable] = true
            } else if neg_single[variable] {
                solution_variables[variable] = false
            } else {
                solution_variables[variable] = random_gen.gen_bool(0.5)
            }
        }
    }

    let mut good_clause_count: Vec<u8> = vec![0; remaining_clauses];
    for (clause_idx, clause) in clause_vector.iter().enumerate() {
        for &literal in clause {
            let var_index = (literal.abs() - 1) as usize;
            if literal > 0 && solution_variables[var_index] {
                good_clause_count[clause_idx] += 1;
            } else if literal < 0 && !solution_variables[var_index] {
                good_clause_count[clause_idx] += 1;
            }
        }
    }

    let mut unsatisfied_clauses = Vec::with_capacity(remaining_clauses);
    let mut clause_map = HashMap::with_capacity(remaining_clauses);

    for (clause_idx, &count_good) in good_clause_count.iter().enumerate() {
        if count_good == 0 {
            unsatisfied_clauses.push(clause_idx);
            clause_map.insert(clause_idx, unsatisfied_clauses.len() - 1);
        }
    }

    let clauses_to_vars_ratio = challenge.difficulty.clauses_to_variables_percent as f64;
    let total_vars = challenge.difficulty.num_variables as f64;
    let max_fuel_limit = 2000000000.0;
    let initial_fuel = (2000.0 + 40.0 * clauses_to_vars_ratio) * total_vars;
    let adjustment_fuel = 900.0 + 1.8 * clauses_to_vars_ratio;
    let max_iterations = ((max_fuel_limit - initial_fuel) / adjustment_fuel) as usize;
    loop {
        if !unsatisfied_clauses.is_empty() {
            let random_value = random_gen.gen::<usize>();

            let clause_idx = unsatisfied_clauses[random_value % unsatisfied_clauses.len()];
            let mut least_conflicts = clause_vector.len();
            let mut best_variable = usize::MAX;
            let clause = &mut clause_vector[clause_idx];

            if clause.len() > 1 {
                let rand_idx = random_value % clause.len();
                clause.swap(0, rand_idx);
            }
            for &literal in clause.iter() {
                let abs_literal = literal.abs() as usize - 1;
                let related_clauses = if solution_variables[abs_literal] { &pos_clause_refs[abs_literal] } else { &neg_clause_refs[abs_literal] };

                let mut conflict_count = 0;
                for &c in related_clauses {
                    if good_clause_count[c] == 1 {
                        conflict_count += 1;
                    }
                }

                if conflict_count < least_conflicts {
                    least_conflicts = conflict_count;
                    best_variable = abs_literal;
                }
            }

            let chosen_variable = if least_conflicts == 0 {
                best_variable
            } else if random_gen.gen_bool(0.5) {
                clause[0].abs() as usize - 1
            } else {
                best_variable
            };

            if solution_variables[chosen_variable] {
                for &clause_ref in &neg_clause_refs[chosen_variable] {
                    good_clause_count[clause_ref] += 1;
                    if good_clause_count[clause_ref] == 1 {
                        let i = clause_map.remove(&clause_ref).unwrap();
                        let last = unsatisfied_clauses.pop().unwrap();
                        if i < unsatisfied_clauses.len() {
                            unsatisfied_clauses[i] = last;
                            clause_map.insert(last, i);
                        }
                    }
                }
                for &clause_ref in &pos_clause_refs[chosen_variable] {
                    if good_clause_count[clause_ref] == 1 {
                        unsatisfied_clauses.push(clause_ref);
                        clause_map.insert(clause_ref, unsatisfied_clauses.len() - 1);
                    }
                    good_clause_count[clause_ref] -= 1;
                }
            } else {
                for &clause_ref in &neg_clause_refs[chosen_variable] {
                    if good_clause_count[clause_ref] == 1 {
                        unsatisfied_clauses.push(clause_ref);
                        clause_map.insert(clause_ref, unsatisfied_clauses.len() - 1);
                    }
                    good_clause_count[clause_ref] -= 1;
                }

                for &clause_ref in &pos_clause_refs[chosen_variable] {
                    good_clause_count[clause_ref] += 1;
                    if good_clause_count[clause_ref] == 1 {
                        let i = clause_map.remove(&clause_ref).unwrap();
                        let last = unsatisfied_clauses.pop().unwrap();
                        if i < unsatisfied_clauses.len() {
                            unsatisfied_clauses[i] = last;
                            clause_map.insert(last, i);
                        }
                    }
                }
            }

            solution_variables[chosen_variable] = !solution_variables[chosen_variable];
        } else {
            break;
        }
        iteration_count += 1;
        if iteration_count >= max_iterations {
            return Ok(None);
        }
    }
    return Ok(Some(Solution { variables: solution_variables }));
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = None;

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};
