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
        let num_items = challenge.num_items;

        let max_weight_plus_one = max_weight + 1;

        let weights: Vec<usize> = challenge.weights.iter().map(|weight| *weight as usize).collect();
        let values: Vec<usize> = challenge.values.iter().map(|value| *value as usize).collect();

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

        let num_states = (num_items + 1) * (max_weight_plus_one);
        let mut dp = vec![0; num_states];

        for i in 1..=num_items {
            let (item_index, _) = sorted_items[i - 1];
            let item_weight = weights[item_index];
            let item_value = values[item_index];

            let i_minus_one_times_max_weight_plus_one = (i - 1) * max_weight_plus_one;
            let i_times_max_weight_plus_one = i * max_weight_plus_one;
            for w in (item_weight..=max_weight).rev() {
                let prev_state = i_minus_one_times_max_weight_plus_one + w;
                let curr_state = i_times_max_weight_plus_one + w;
                dp[curr_state] = dp[prev_state].max(dp[prev_state - item_weight] + item_value);
            }
        }

        let mut items = Vec::with_capacity(num_items);
        let mut i = num_items;
        let mut w = max_weight;
        let mut total_value = 0;
        while i > 0 && total_value < baseline_value {
            let (item_index, _) = sorted_items[i - 1];
            let item_weight = weights[item_index];
            let item_value = values[item_index];

            let prev_state = (i - 1) * (max_weight_plus_one) + w;
            let curr_state = i * (max_weight_plus_one) + w;
            if dp[curr_state] != dp[prev_state] {
                items.push(item_index);
                w -= item_weight;
                total_value += item_value;
            }
            i -= 1;
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
