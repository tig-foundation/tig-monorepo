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
    use std::cmp::Reverse;


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
        let max_weight_plus_one = max_weight + 1;

        let items: Vec<_> = challenge.weights.iter().zip(challenge.values.iter())
            .enumerate()
            .map(|(i, (&weight, &value))| (i, weight as usize, value as usize))
            .collect();

        let mut sorted_items = items.clone();
        sorted_items.sort_unstable_by_key(|&(_, weight, value)| Reverse((value as f64 / weight as f64 * 1000.0) as usize));

        if !is_solution_possible(&sorted_items, max_weight, baseline_value) {
            return Ok(None);
        }

        let mut dp = vec![0; max_weight_plus_one];
        let mut selected = vec![vec![]; max_weight_plus_one];

        for &(index, weight, value) in &sorted_items {
            for w in (weight..=max_weight).rev() {
                let new_value = dp[w - weight] + value;
                if new_value > dp[w] {
                    dp[w] = new_value;
                    selected[w] = selected[w - weight].clone();
                    selected[w].push(index);
                }
            }
        }

        if dp[max_weight] >= baseline_value {
            Ok(Some(SubSolution { items: selected[max_weight].clone() }))
        } else {
            Ok(None)
        }
    }

    fn is_solution_possible(sorted_items: &[(usize, usize, usize)], max_weight: usize, baseline_value: usize) -> bool {
        let mut remaining_weight = max_weight;
        let mut total_value = 0;

        for &(_, weight, value) in sorted_items {
            if weight <= remaining_weight {
                total_value += value;
                remaining_weight -= weight;
            } else {
                let fractional_value = (value as f64 * remaining_weight as f64 / weight as f64) as usize;
                return total_value + fractional_value >= baseline_value;
            }

            if total_value >= baseline_value {
                return true;
            }
        }

        total_value >= baseline_value
    }
}

pub fn help() {
    println!("No help information available.");
}
