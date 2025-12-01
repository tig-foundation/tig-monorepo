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
    use std::collections::VecDeque;
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

        let tabu_tenure = 10;
        let max_iterations = 100;
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()) as u64);

        let initial_items: Vec<bool> = (0..num_items).map(|_| rng.gen_bool(0.5)).collect();
        let mut current_solution = Individual::new(initial_items.clone(), 0);
        current_solution.calculate_fitness(values, weights, max_weight);

        let mut best_solution = current_solution.clone();
        let mut tabu_list: VecDeque<Vec<bool>> = VecDeque::new();

        for _ in 0..max_iterations {
            let mut neighborhood = Vec::new();

            for i in 0..num_items {
                let mut neighbor_items = current_solution.items.clone();
                neighbor_items[i] = !neighbor_items[i];
                let mut neighbor = Individual::new(neighbor_items, 0);
                neighbor.calculate_fitness(values, weights, max_weight);
                if !tabu_list.contains(&neighbor.items) {
                    neighborhood.push(neighbor);
                }
            }

            if let Some(best_neighbor) = neighborhood.iter().max_by_key(|ind| ind.fitness) {
                current_solution = best_neighbor.clone();
                if current_solution.fitness > best_solution.fitness {
                    best_solution = current_solution.clone();
                }
            }

            tabu_list.push_back(current_solution.items.clone());
            if tabu_list.len() > tabu_tenure {
                tabu_list.pop_front();
            }
        }

        if best_solution.fitness >= baseline_value {
            let items = best_solution.items.iter().enumerate()
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
