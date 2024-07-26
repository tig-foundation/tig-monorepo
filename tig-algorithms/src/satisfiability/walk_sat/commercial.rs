/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Commercial License v1.0 (the "License"); you 
may not use this file except in compliance with the License. You may obtain a copy 
of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tig_challenges::satisfiability::*;

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let mut rng = StdRng::seed_from_u64(challenge.seed as u64);
    let num_variables = challenge.difficulty.num_variables;
    let max_flips = 1000;

    let mut variables: Vec<bool> = (0..num_variables).map(|_| rng.gen::<bool>()).collect();

    for _ in 0..max_flips {
        let mut unsatisfied_clauses: Vec<&Vec<i32>> = challenge
            .clauses
            .iter()
            .filter(|clause| !clause_satisfied(clause, &variables))
            .collect();

        if unsatisfied_clauses.is_empty() {
            return Ok(Some(Solution { variables }));
        }

        let clause = unsatisfied_clauses.choose(&mut rng).unwrap();
        let literal = clause.choose(&mut rng).unwrap();
        let var_idx = literal.abs() as usize - 1;
        variables[var_idx] = !variables[var_idx];
    }

    Ok(None)
}

fn clause_satisfied(clause: &Vec<i32>, variables: &[bool]) -> bool {
    clause.iter().any(|&literal| {
        let var_idx = literal.abs() as usize - 1;
        (literal > 0 && variables[var_idx]) || (literal < 0 && !variables[var_idx])
    })
}
