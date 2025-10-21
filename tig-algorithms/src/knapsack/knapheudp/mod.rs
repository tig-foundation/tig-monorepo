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
        let num_items = challenge.difficulty.num_items;

        let weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
        let values: Vec<usize> = challenge.values.iter().map(|&v| v as usize).collect();

        let mut sorted_items: Vec<(usize, f64)> = (0..num_items)
            .map(|i| (i, values[i] as f64 / weights[i] as f64))
            .collect();
        sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut upper_bound = 0;
        let mut remaining_weight = max_weight;
        for &(item_index, ratio) in &sorted_items {
            let item_weight = weights[item_index];
            let item_value = values[item_index];

            if item_weight <= remaining_weight {
                upper_bound += item_value;
                remaining_weight -= item_weight;
            } else {
                upper_bound += (ratio * remaining_weight as f64).floor() as usize;
                break;
            }
        }

        if upper_bound < baseline_value {
            return Ok(None);
        }

        let mut dp = vec![0; max_weight + 1];
        let mut selected = vec![vec![false; max_weight + 1]; num_items];

        for (i, &(item_index, _)) in sorted_items.iter().enumerate() {
            let weight = weights[item_index];
            let value = values[item_index];

            for w in (weight..=max_weight).rev() {
                let new_value = dp[w - weight] + value;
                if new_value > dp[w] {
                    dp[w] = new_value;
                    selected[i][w] = true;
                }
            }

            if dp[max_weight] >= baseline_value {
                break;
            }
        }

        if dp[max_weight] < baseline_value {
            return Ok(None);
        }

        let mut items = Vec::new();
        let mut w = max_weight;
        for i in (0..num_items).rev() {
            if selected[i][w] {
                let item_index = sorted_items[i].0;
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