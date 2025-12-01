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
        let num_items = challenge.num_items;
        let weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
        let values: Vec<usize> = challenge.values.iter().map(|&v| v as usize).collect();

        let mut sorted_items: Vec<usize> = (0..num_items).collect();
        sorted_items.sort_unstable_by_key(|&i| Reverse((values[i] as f64 / weights[i] as f64 * 1000.0) as usize));

        let mut dp = vec![0; max_weight + 1];
        let mut selected = vec![0u64; (num_items * (max_weight + 1) + 63) / 64];

        let mut upper_bound = 0;
        let mut remaining_weight = max_weight;

        for (i, &item_index) in sorted_items.iter().enumerate() {
            let weight = weights[item_index];
            let value = values[item_index];

            if weight <= remaining_weight {
                upper_bound += value;
                remaining_weight -= weight;
            } else {
                upper_bound += ((value as f64 * remaining_weight as f64 / weight as f64) as usize).min(value);
                if upper_bound < baseline_value {
                    return Ok(None);
                }
                remaining_weight = 0;
            }

            for w in (weight..=max_weight).rev() {
                let new_value = dp[w - weight] + value;
                if new_value > dp[w] {
                    dp[w] = new_value;
                    let idx = i * (max_weight + 1) + w;
                    selected[idx / 64] |= 1 << (idx % 64);
                }
            }

            if dp[max_weight] >= baseline_value {
                break;
            }
        }

        if dp[max_weight] < baseline_value {
            return Ok(None);
        }

        let mut items = Vec::with_capacity(num_items);
        let mut w = max_weight;
        for (i, &item_index) in sorted_items.iter().enumerate().rev() {
            let idx = i * (max_weight + 1) + w;
            if (selected[idx / 64] & (1 << (idx % 64))) != 0 {
                items.push(item_index);
                w -= weights[item_index];
            }
            if w == 0 {
                break;
            }
        }

        Ok(Some(SubSolution { items }))
    }
}

pub fn help() {
    println!("No help information available.");
}
