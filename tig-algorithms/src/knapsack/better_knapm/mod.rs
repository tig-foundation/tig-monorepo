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
    use tig_challenges::knapsack::*;
    use std::collections::HashMap;


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
        let max_weight = challenge.max_weight as usize;
        let baseline_value = challenge.baseline_value as usize;
        let num_items = challenge.num_items;

        let weights: Vec<usize> = challenge.weights.iter().map(|weight| *weight as usize).collect();
        let values: Vec<usize> = challenge.values.iter().map(|value| *value as usize).collect();

        fn compute_combinations(weights: &[usize], values: &[usize]) -> Vec<(usize, usize)> {
            let n = weights.len();
            let mut combinations = Vec::with_capacity(1 << n);

            for i in 0..(1 << n) {
                let mut total_weight = 0;
                let mut total_value = 0;
                for j in 0..n {
                    if (i & (1 << j)) != 0 {
                        total_weight += weights[j];
                        total_value += values[j];
                    }
                }
                combinations.push((total_weight, total_value));
            }

            combinations
        }

        let (left_weights, right_weights) = weights.split_at(num_items / 2);
        let (left_values, right_values) = values.split_at(num_items / 2);

        let left_combinations = compute_combinations(left_weights, left_values);
        let right_combinations = compute_combinations(right_weights, right_values);

        let mut right_combinations_map = HashMap::new();
        for &(weight, value) in &right_combinations {
            right_combinations_map.entry(weight).or_insert(value);
        }

        let mut max_value = 0;
        let mut best_combination = Vec::new();

        for &(left_weight, left_value) in &left_combinations {
            if left_weight > max_weight {
                continue;
            }

            let remaining_weight = max_weight - left_weight;
            let mut best_right_value = 0;

            for (&weight, &value) in &right_combinations_map {
                if weight <= remaining_weight {
                    best_right_value = best_right_value.max(value);
                }
            }

            let total_value = left_value + best_right_value;
            if total_value >= baseline_value && total_value > max_value {
                max_value = total_value;
                best_combination.clear();
                best_combination.extend(
                    (0..left_weights.len())
                        .filter(|&i| (1 << i) & (1 << left_weights.len()) != 0)
                        .collect::<Vec<_>>(),
                );
                best_combination.extend(
                    (0..right_weights.len())
                        .filter(|&i| (1 << i) & (1 << right_weights.len()) != 0)
                        .map(|i| i + left_weights.len())
                        .collect::<Vec<_>>(),
                );
            }
        }

        if max_value >= baseline_value {
            Ok(Some(SubSolution { items: best_combination }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
