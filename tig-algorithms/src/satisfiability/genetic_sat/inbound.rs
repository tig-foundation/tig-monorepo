/*!
Copyright 2024 Chad Blanchard

Licensed under the TIG Inbound Game License v1.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

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

#[derive(Clone, Debug)]
struct Individual {
    variables: Vec<bool>,
    fitness: usize,
}

impl Individual {
    fn new(variables: Vec<bool>, fitness: usize) -> Self {
        Self { variables, fitness }
    }

    fn calculate_fitness(&mut self, clauses: &[Vec<i32>]) {
        self.fitness = clauses.iter().filter(|&&ref clause| clause_satisfied(clause, &self.variables)).count();
    }
}

pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
    let num_variables = challenge.difficulty.num_variables;
    let clauses = &challenge.clauses;
    let population_size = 50;
    let generations = 100;
    let mutation_rate = 0.01;
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

    let mut population: Vec<Individual> = (0..population_size).map(|_| {
        let variables: Vec<bool> = (0..num_variables).map(|_| rng.gen::<bool>()).collect();
        let mut individual = Individual::new(variables, 0);
        individual.calculate_fitness(clauses);
        individual
    }).collect();

    for _ in 0..generations {
        population.sort_by(|a, b| b.fitness.cmp(&a.fitness));

        let mut new_population = population[..population_size / 2].to_vec();

        while new_population.len() < population_size {
            let parent1 = &population[rng.gen_range(0..population_size / 2)];
            let parent2 = &population[rng.gen_range(0..population_size / 2)];
            let crossover_point = rng.gen_range(0..num_variables);

            let mut child_variables = parent1.variables[..crossover_point].to_vec();
            child_variables.extend_from_slice(&parent2.variables[crossover_point..]);

            if rng.gen::<f64>() < mutation_rate {
                let mutation_point = rng.gen_range(0..num_variables);
                child_variables[mutation_point] = !child_variables[mutation_point];
            }

            let mut child = Individual::new(child_variables, 0);
            child.calculate_fitness(clauses);
            new_population.push(child);
        }

        population = new_population;
    }

    let best_individual = &population[0];
    if best_individual.fitness == clauses.len() {
        return Ok(Some(Solution { variables: best_individual.variables.clone() }));
    }

    Ok(None)
}

fn clause_satisfied(clause: &Vec<i32>, variables: &[bool]) -> bool {
    clause.iter().any(|&literal| {
        let var_idx = literal.abs() as usize - 1;
        (literal > 0 && variables[var_idx]) || (literal < 0 && !variables[var_idx])
    })
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
