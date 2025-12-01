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
    use anyhow::Result;
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

    pub fn solve_sub_instance(challenge: &SubInstance) -> Result<Option<SubSolution>> {
        const WAIT_ITERATIONS: usize = 5;
        const MAX_STAGNANT_ITERATIONS: usize = 15;

        let num_items = challenge.weights.len();
        let mut selected_items = vec![false; num_items];
        let mut total_value: i32 = 0;
        let mut total_weight: u32 = 0;
        let mut wait_map = vec![None; num_items];
        let values: Vec<i32> = challenge.values.iter().map(|&v| v as i32).collect();
    
        let mut items_by_ratio: Vec<(usize, f64)> = (0..num_items)
            .map(|i| {
                let ratio = values[i] as f64 / challenge.weights[i] as f64;
                (i, ratio)
            })
            .collect();
        items_by_ratio.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
        let mut interaction_gains = vec![0; num_items];
        let mut weight_reduction_candidates = Vec::with_capacity(num_items);
        let mut available_items = Vec::with_capacity(num_items);

        let mut iteration_count = 0;
        let mut stagnant_iterations = 0;
        let mut max_total_value = total_value;
        let weight_threshold = challenge.max_weight * 85 / 100;
        let baseline_value = challenge.baseline_value as i32;
        let max_weight = challenge.max_weight;

        let interaction_rows: Vec<&[i32]> = challenge
            .interaction_values
            .iter()
            .map(|row| row.as_slice())
            .collect();

        loop {
            iteration_count += 1;

            for entry in &mut wait_map {
                if let Some(iter) = entry {
                    if *iter <= iteration_count {
                        *entry = None;
                    }
                }
            }

            available_items.clear();
            available_items.extend(
                items_by_ratio.iter()
                    .filter(|&&(i, _)| !selected_items[i] && wait_map[i].is_none())
                    .copied()
            );

            let mut improvement_found = false;
            let mut index = 0;

            while index < available_items.len() {
                let (i, _) = available_items[index];
                let individual_value = values[i];
                let interaction_gain = interaction_gains[i];
                let gain = individual_value + interaction_gain;
                let potential_weight = total_weight + challenge.weights[i];
            
                if gain >= individual_value ||
                   (gain >= individual_value - 2 && potential_weight <= weight_threshold) {
                    selected_items[i] = true;
                    total_value += gain;
                    total_weight = potential_weight;

                    let interaction_row = interaction_rows[i];
                    for (j, gain) in interaction_gains.iter_mut().enumerate() {
                        *gain += interaction_row[j];
                    }

                    improvement_found = true;
                    available_items.remove(index);
                } else {
                    index += 1;
                }
            }

            if !improvement_found {
                for &(i, _) in &available_items {
                    let new_item_value = values[i] + interaction_gains[i];
                    let new_item_weight = challenge.weights[i];

                    if new_item_value <= values[i] {
                        continue;
                    }

                    for j in 0..num_items {
                        if selected_items[j] {
                            let removal_loss = values[j] + interaction_gains[j];
                            if total_value + new_item_value - removal_loss > total_value {
                                let remove_row = interaction_rows[j];
                                let add_row = interaction_rows[i];
                            
                                for k in 0..num_items {
                                    interaction_gains[k] = interaction_gains[k] - remove_row[k] + add_row[k];
                                }
                            
                                selected_items[j] = false;
                                total_value -= removal_loss;
                                total_weight -= challenge.weights[j];

                                selected_items[i] = true;
                                total_value += new_item_value;
                                total_weight += new_item_weight;

                                wait_map[j] = Some(iteration_count + WAIT_ITERATIONS);
                                improvement_found = true;
                                break;
                            }
                        }
                    }

                    if improvement_found {
                        break;
                    } else {
                        return Ok(None);
                    }
                }
            }

            if total_weight > max_weight {
                weight_reduction_candidates.clear();
                for i in 0..num_items {
                    if selected_items[i] {
                        let loss = values[i] + interaction_gains[i];
                        let ratio = challenge.weights[i] as f64 / (loss as f64).max(1.0);
                        weight_reduction_candidates.push((ratio, i));
                    }
                }
                weight_reduction_candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                while total_weight > max_weight {
                    if let Some((_, item)) = weight_reduction_candidates.pop() {
                        let remove_row = interaction_rows[item];
                        for k in 0..num_items {
                            interaction_gains[k] -= remove_row[k];
                        }
                        selected_items[item] = false;
                        total_weight -= challenge.weights[item];
                        total_value -= values[item] + interaction_gains[item];
                        wait_map[item] = Some(iteration_count + WAIT_ITERATIONS);
                    } else {
                        break;
                    }
                }
            }

            if total_value >= baseline_value && total_weight <= max_weight {
                let result_items: Vec<usize> = selected_items
                    .iter()
                    .enumerate()
                    .filter(|&(_, &is_selected)| is_selected)
                    .map(|(i, _)| i)
                    .collect();

                return Ok(Some(SubSolution {
                    items: result_items,
                }));
            }

            if total_value > max_total_value {
                max_total_value = total_value;
                stagnant_iterations = 0;
            } else {
                stagnant_iterations += 1;
            }

            if stagnant_iterations >= MAX_STAGNANT_ITERATIONS {
                return Ok(None);
            }
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
