/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Open Data License v1.0 or (at your option) any later version 
(the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use tig_challenges::satisfiability::*;
use std::collections::HashMap;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let num_variables = challenge.difficulty.num_variables;
    let mut assignment = vec![None; num_variables];

    if dpll(&challenge.clauses, &mut assignment) {
        let variables = assignment.into_iter().map(|x| x.unwrap_or(false)).collect();
        Ok(Some(Solution { variables }))
    } else {
        Ok(None)
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

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    // set KERNEL to None if algorithm only has a CPU implementation
    pub const KERNEL: Option<CudaKernel> = None;

    // Important! your GPU and CPU version of the algorithm should return the same result
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
