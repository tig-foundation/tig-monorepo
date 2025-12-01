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
        let baseline_value = challenge.baseline_value as i32;
        let max_weight = challenge.max_weight;
        let num_items = challenge.num_items;
        let weights = &challenge.weights;
        let values = &challenge.values;
        let interaction_values = &challenge.interaction_values;

        let mut current_solution = vec![false; num_items];
        let mut total_weight = 0;
        let mut total_value = 0;

        let mut sorted_items: Vec<(usize, f32)> = (0..num_items)
            .map(|i| {
                let interaction_value_sum: f32 = interaction_values[i].iter().map(|&v| v as f32).sum();
                let ratio = (values[i] as f32 + interaction_value_sum) / weights[i] as f32;
                (i, ratio)
            })
            .collect();
        sorted_items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (item_index, _) in sorted_items {
            if total_weight + weights[item_index] <= max_weight {
                current_solution[item_index] = true;
                total_weight += weights[item_index];
                total_value += values[item_index] as i32;

                for i in 0..num_items {
                    if current_solution[i] {
                        total_value += interaction_values[item_index][i];
                    }
                }
            }
        }

        if total_value < baseline_value {
            total_value = optimize_solution(
                &mut current_solution,
                num_items,
                weights,
                values,
                interaction_values,
                baseline_value,
                max_weight,
                total_weight,
                total_value,
            );
        }

        if total_value >= baseline_value {
            Ok(Some(SubSolution {
                items: current_solution
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &included)| if included { Some(i) } else { None })
                    .collect(),
            }))
        } else {
            Ok(None)
        }
    }

    fn optimize_solution(
        solution: &mut Vec<bool>,
        num_items: usize,
        weights: &Vec<u32>,
        values: &Vec<u32>,
        interaction_values: &Vec<Vec<i32>>,
        baseline_value: i32,
        max_weight: u32,
        total_weight: u32,
        total_value: i32,
    ) -> i32 {
        let mut best_value = total_value;
        let mut current_value = best_value;
        let mut current_weight = total_weight;
        let mut improved = true;

        while improved {
            improved = false;

            for i in 0..num_items {
                for j in (i + 1)..num_items {
                    if solution[i] != solution[j] {
                        let new_weight = if solution[i] {
                            current_weight - weights[i] + weights[j]
                        } else {
                            current_weight + weights[i] - weights[j]
                        };

                        if new_weight <= max_weight {
                            let delta_value = calculate_delta_value(
                                solution,
                                num_items,
                                i,
                                j,
                                values,
                                interaction_values,
                            );

                            let new_value = current_value + delta_value;
                            if new_value > best_value {
                                best_value = new_value;
                                current_value = new_value;
                                current_weight = new_weight;
                                solution.swap(i, j);
                                improved = true;
                                if best_value >= baseline_value {
                                    return best_value;
                                }
                            }
                        }
                    }
                }
            }
        }

        best_value
    }

    #[inline]
    fn calculate_delta_value(
        solution: &Vec<bool>,
        num_items: usize,
        i: usize,
        j: usize,
        values: &Vec<u32>,
        interaction_values: &Vec<Vec<i32>>,
    ) -> i32 {
        let mut delta_value = 0;

        if solution[i] && !solution[j] {
            delta_value += (values[j] as i32) - (values[i] as i32);
            for k in 0..num_items {
                if k != i && k != j && solution[k] {
                    delta_value += interaction_values[j][k] - interaction_values[i][k];
                }
            }
        } else if !solution[i] && solution[j] {
            delta_value -= (values[j] as i32) - (values[i] as i32);
            for k in 0..num_items {
                if k != i && k != j && solution[k] {
                    delta_value += interaction_values[i][k] - interaction_values[j][k];
                }
            }
        }

        delta_value
    }
}

pub fn help() {
    println!("No help information available.");
}
