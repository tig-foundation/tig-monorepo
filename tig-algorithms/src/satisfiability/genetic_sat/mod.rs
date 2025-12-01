// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{Map, Value};
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

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    let num_variables = challenge.num_variables;
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
        let _ = save_solution(&Solution { variables: best_individual.variables.clone() });
        return Ok(());
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
