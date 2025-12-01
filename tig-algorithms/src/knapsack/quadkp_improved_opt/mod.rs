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
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use tig_challenges::knapsack::*;


    fn add_item(
        i: usize,
        selected_bits: &mut [bool],
        total_value: &mut i32,
        total_weight: &mut u32,
        interaction_gains: &mut [i32],
        challenge: &SubInstance,
        values: &[i32],
        weights: &[u32],
    ) {
        let gain = values[i] + interaction_gains[i];
        selected_bits[i] = true;
        *total_value += gain;
        *total_weight += weights[i];
        // Mise à jour des interactions
        for j in 0..interaction_gains.len() {
            interaction_gains[j] += challenge.interaction_values[i][j];
        }
    }


    fn remove_item(
        i: usize,
        selected_bits: &mut [bool],
        total_value: &mut i32,
        total_weight: &mut u32,
        interaction_gains: &mut [i32],
        challenge: &SubInstance,
        values: &[i32],
        weights: &[u32],
    ) {
        let removal_loss = values[i] + interaction_gains[i];
        selected_bits[i] = false;
        *total_value -= removal_loss;
        *total_weight -= weights[i];
        for j in 0..interaction_gains.len() {
            interaction_gains[j] -= challenge.interaction_values[i][j];
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

    pub fn solve_sub_instance(challenge: &SubInstance) -> Result<Option<SubSolution>> {
        let vertex_count = challenge.weights.len();

        let values: Vec<i32> = challenge.values.iter().map(|&v| v as i32).collect();
        let weights: Vec<u32> = challenge.weights.clone(); // ou .to_vec()

        let mut item_scores: Vec<(usize, f32)> = (0..vertex_count)
            .map(|i| {
                let interaction_sum: i32 = challenge.interaction_values[i].iter().sum();
                let secondary_score = values[i] as f32 / (weights[i].max(1)) as f32;
                let combined_score = (values[i] as f32 * 0.75
                    + interaction_sum as f32 * 0.15
                    + secondary_score * 0.1)
                    / (weights[i] as f32).max(1.0);
                (i, combined_score)
            })
            .collect();

        item_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut selected_bits = vec![false; vertex_count];
        let mut interaction_gains = vec![0; vertex_count];

        let mut total_value: i32 = 0;
        let mut total_weight: u32 = 0;

        let mut selected_list = Vec::with_capacity(vertex_count);
        let mut unselected_list = Vec::with_capacity(vertex_count);

        for &(i, _) in &item_scores {
            if total_weight + weights[i] <= challenge.max_weight {
                add_item(
                    i,
                    &mut selected_bits,
                    &mut total_value,
                    &mut total_weight,
                    &mut interaction_gains,
                    challenge,
                    &values,
                    &weights,
                );
                selected_list.push(i);
            } else {
                unselected_list.push(i);
            }
        }

        let mut mutation_rates = vec![0; vertex_count];
        for i in 0..vertex_count {
            mutation_rates[i] = values[i] + interaction_gains[i];
        }

        let max_generations = (vertex_count ) / 2;    
        let mut cooling_schedule = vec![0; vertex_count];
        let rng = StdRng::seed_from_u64(challenge.seed[0] as u64);

        for _generation in 0..max_generations {
            let mut best_gain = 0;
            let mut best_swap = None;

            for (u_index, &mutant) in unselected_list.iter().enumerate() {
                if cooling_schedule[mutant] > 0 {
                    continue;
                }
                let mutant_fitness = mutation_rates[mutant];

                let extra_weight = (weights[mutant] as i32)
                    - (challenge.max_weight as i32 - total_weight as i32);

                if mutant_fitness < 0 {
                    continue;
                }

                for (c_index, &sel) in selected_list.iter().enumerate() {
                    if cooling_schedule[sel] > 0 {
                        continue;
                    }
                    if extra_weight > 0 && (weights[sel] as i32) < extra_weight {
                        continue;
                    }

                    let interaction_penalty =
                        (challenge.interaction_values[mutant][sel] as f32 * 0.3) as i32;
                    let fitness_gain = mutant_fitness - mutation_rates[sel] - interaction_penalty;

                    if fitness_gain > best_gain {
                        best_gain = fitness_gain;
                        best_swap = Some((u_index, c_index));
                    }
                }
            }

            if let Some((u_index, c_index)) = best_swap {
                let added_item = unselected_list[u_index];
                let removed_item = selected_list[c_index];

                remove_item(
                    removed_item,
                    &mut selected_bits,
                    &mut total_value,
                    &mut total_weight,
                    &mut interaction_gains,
                    challenge,
                    &values,
                    &weights,
                );
                add_item(
                    added_item,
                    &mut selected_bits,
                    &mut total_value,
                    &mut total_weight,
                    &mut interaction_gains,
                    challenge,
                    &values,
                    &weights,
                );

                selected_list.swap_remove(c_index);
                unselected_list.swap_remove(u_index);
                selected_list.push(added_item);
                unselected_list.push(removed_item);

                for i in 0..vertex_count {
                    mutation_rates[i] = values[i] + interaction_gains[i];
                }

                if total_weight > challenge.max_weight {
                    continue;
                }

                cooling_schedule[added_item] = 3;
                cooling_schedule[removed_item] = 3;
            }

            if total_value >= challenge.baseline_value as i32 {
                let final_items: Vec<usize> = selected_list.clone();
                return Ok(Some(SubSolution { items: final_items }));
            }

            for c in cooling_schedule.iter_mut() {
                *c = if *c > 0 { *c - 1 } else { 0 };
            }

            if total_value as u32 > (challenge.baseline_value * 9 / 10) {
                let high_potential_items: Vec<usize> = unselected_list
                    .iter()
                    .copied()
                    .filter(|&i| values[i] > (challenge.baseline_value as i32 / 4))
                    .collect();

                for &item in high_potential_items.iter().take(2) {
                    if total_weight + weights[item] <= challenge.max_weight {
                        unselected_list.retain(|&x| x != item);

                        add_item(
                            item,
                            &mut selected_bits,
                            &mut total_value,
                            &mut total_weight,
                            &mut interaction_gains,
                            challenge,
                            &values,
                            &weights,
                        );
                        selected_list.push(item);

                        for i in 0..vertex_count {
                            mutation_rates[i] = values[i] + interaction_gains[i];
                        }

                        if total_value >= challenge.baseline_value as i32 {
                            let final_items: Vec<usize> = selected_list.clone();
                            return Ok(Some(SubSolution { items: final_items }));
                        }
                    }
                }
            }
        }

        if total_value as u32 >= challenge.baseline_value && total_weight <= challenge.max_weight {
            let final_items: Vec<usize> = selected_list.clone();
            Ok(Some(SubSolution { items: final_items }))
        } else {
            Ok(None)
        }
    }
}

pub fn help() {
    println!("No help information available.");
}
