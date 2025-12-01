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
    use std::collections::HashMap;
    use tig_challenges::knapsack::*;


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

        let weights: Vec<usize> = challenge
            .weights
            .iter()
            .map(|weight| *weight as usize)
            .collect();
        let values: Vec<usize> = challenge
            .values
            .iter()
            .map(|value| *value as usize)
            .collect();

        // Helper function to compute knapsack solution using memoization (Top-down DP)
        fn knapsack(
            weights: &[usize],
            values: &[usize],
            max_weight: usize,
            n: usize,
            memo: &mut HashMap<(usize, usize), usize>,
        ) -> usize {
            if n == 0 || max_weight == 0 {
                return 0;
            }

            if let Some(&result) = memo.get(&(n, max_weight)) {
                return result;
            }

            let result = if weights[n - 1] > max_weight {
                knapsack(weights, values, max_weight, n - 1, memo)
            } else {
                let included =
                    values[n - 1] + knapsack(weights, values, max_weight - weights[n - 1], n - 1, memo);
                let excluded = knapsack(weights, values, max_weight, n - 1, memo);
                included.max(excluded)
            };

            memo.insert((n, max_weight), result);
            result
        }

        let mut memo = HashMap::new();
        let max_value = knapsack(&weights, &values, max_weight, num_items, &mut memo);

        if max_value < baseline_value {
            return Ok(None);
        }

        // Reconstructing the solution
        let mut items = Vec::with_capacity(num_items);
        let mut remaining_weight = max_weight;
        let mut total_value = max_value;

        for i in (1..=num_items).rev() {
            if remaining_weight == 0 {
                break;
            }

            if memo.get(&(i, remaining_weight)) == Some(&total_value) {
                continue;
            } else {
                items.push(i - 1);
                remaining_weight -= weights[i - 1];
                total_value -= values[i - 1];
            }
        }

        if total_value >= baseline_value {
            Ok(Some(SubSolution { items }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
