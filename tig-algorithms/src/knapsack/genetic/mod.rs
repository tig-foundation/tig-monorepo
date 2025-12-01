use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    Err(anyhow!("This algorithm is no longer compatible."))
}

// Old code that is no longer compatible
#[cfg(none)]
mod dead_code {
    // TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tig_challenges::knapsack::*;

    #[derive(Clone, Debug)]
    struct Individual {
        items: Vec<bool>,
        fitness: u32,
    }

    impl Individual {
        fn new(items: Vec<bool>, fitness: u32) -> Self {
            Self { items, fitness }
        }

        fn calculate_fitness(&mut self, values: &[u32], weights: &[u32], max_weight: u32) {
            let mut total_value = 0;
            let mut total_weight = 0;
            for i in 0..self.items.len() {
                if self.items[i] {
                    total_value += values[i];
                    total_weight += weights[i];
                }
            }
            if total_weight <= max_weight {
                self.fitness = total_value;
            } else {
                self.fitness = 0;
            }
        }
    }


    pub fn solve_challenge(challenge: &Challenge) -> anyhow::Result<Option<Solution>> {
        let mut solution = Solution {
            sub_solutions: Vec::new(),
        };
        for sub_instance in &challenge.sub_instances {
            match solve_sub_instance(sub_instance)? {
                Some(sub_solution) => solution.sub_solutions.push(sub_solution),
                None => return Ok(None),
            }
        }
        Ok(Some(solution))
    }

    pub fn solve_sub_instance(challenge: &SubInstance) -> anyhow::Result<Option<SubSolution>> {
        let max_weight = challenge.max_weight;
        let baseline_value = challenge.baseline_value;
        let num_items = challenge.num_items;
        let values = &challenge.values;
        let weights = &challenge.weights;

        let population_size = 50;
        let generations = 100;
        let mutation_rate = 0.01;
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

        let mut population: Vec<Individual> = (0..population_size).map(|_| {
            let items: Vec<bool> = (0..num_items).map(|_| rng.gen_bool(0.5)).collect();
            let mut individual = Individual::new(items, 0);
            individual.calculate_fitness(values, weights, max_weight);
            individual
        }).collect();

        for _ in 0..generations {
            population.sort_by(|a, b| b.fitness.cmp(&a.fitness));

            let mut new_population = population[..population_size / 2].to_vec();

            while new_population.len() < population_size {
                let parent1 = &population[rng.gen_range(0..population_size / 2)];
                let parent2 = &population[rng.gen_range(0..population_size / 2)];
                let crossover_point = rng.gen_range(0..num_items);

                let mut child_items = parent1.items[..crossover_point].to_vec();
                child_items.extend_from_slice(&parent2.items[crossover_point..]);

                if rng.gen::<f64>() < mutation_rate {
                    let mutation_point = rng.gen_range(0..num_items);
                    child_items[mutation_point] = !child_items[mutation_point];
                }

                let mut child = Individual::new(child_items, 0);
                child.calculate_fitness(values, weights, max_weight);
                new_population.push(child);
            }

            population = new_population;
        }

        let best_individual = &population[0];
        if best_individual.fitness >= baseline_value {
            let items = best_individual.items.iter().enumerate()
                .filter_map(|(i, &included)| if included { Some(i) } else { None })
                .collect();
            Ok(Some(SubSolution { items }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
