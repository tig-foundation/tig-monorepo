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
        let capacity = challenge.max_weight as usize;
        let target_value = challenge.baseline_value as usize;
        let total_items = challenge.num_items;

        let item_weights: Vec<usize> = challenge.weights.iter().map(|&w| w as usize).collect();
        let item_values: Vec<usize> = challenge.values.iter().map(|&v| v as usize).collect();

        let mut item_efficiency: Vec<(usize, f64)> = (0..total_items)
            .map(|i| (i, item_values[i] as f64 / item_weights[i] as f64))
            .collect();
        item_efficiency.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut max_possible_value = 0;
        let mut available_weight = capacity;
        for &(idx, ratio) in &item_efficiency {
            let weight = item_weights[idx];
            let value = item_values[idx];

            if weight <= available_weight {
                max_possible_value += value;
                available_weight -= weight;
            } else {
                max_possible_value += (ratio * available_weight as f64).floor() as usize;
                break;
            }
        }

        if max_possible_value < target_value {
            return Ok(None);
        }

        let mut dp_table = vec![0; capacity + 1];
        let mut track_selection = vec![vec![false; capacity + 1]; total_items];

        for (i, &(item_index, _)) in item_efficiency.iter().enumerate() {
            let weight = item_weights[item_index];
            let value = item_values[item_index];

            for current_weight in (weight..=capacity).rev() {
                let potential_value = dp_table[current_weight - weight] + value;
                if potential_value > dp_table[current_weight] {
                    dp_table[current_weight] = potential_value;
                    track_selection[i][current_weight] = true;
                }
            }

            if dp_table[capacity] >= target_value {
                break;
            }
        }

        if dp_table[capacity] < target_value {
            return Ok(None);
        }

        let mut selected_items = Vec::new();
        let mut remaining_capacity = capacity;
        for i in (0..total_items).rev() {
            if track_selection[i][remaining_capacity] {
                let item_index = item_efficiency[i].0;
                selected_items.push(item_index);
                remaining_capacity -= item_weights[item_index];
            }
            if remaining_capacity == 0 {
                break;
            }
        }

        Ok(Some(SubSolution { items: selected_items }))
    }
}

pub fn help() {
    println!("No help information available.");
}
