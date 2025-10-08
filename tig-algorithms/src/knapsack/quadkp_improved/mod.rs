// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;


pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let vertex_count = challenge.weights.len();

    let mut item_scores: Vec<(usize, f32)> = (0..vertex_count)
        .map(|index| {
            let interaction_sum: i32 = challenge.interaction_values[index].iter().sum();
            let secondary_score = challenge.values[index] as f32 / challenge.weights[index] as f32;
            let combined_score = (challenge.values[index] as f32 * 0.75
                + interaction_sum as f32 * 0.15
                + secondary_score * 0.1)
                / challenge.weights[index] as f32;
            (index, combined_score)
        })
        .collect();

    item_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut selected_items = Vec::with_capacity(vertex_count);
    let mut unselected_items = Vec::with_capacity(vertex_count);
    let mut current_weight = 0;
    let mut current_value = 0;

    for &(index, _) in &item_scores {
        if current_weight + challenge.weights[index] <= challenge.max_weight {
            current_weight += challenge.weights[index];
            current_value += challenge.values[index] as i32;

            for &selected in &selected_items {
                current_value += challenge.interaction_values[index][selected];
            }
            selected_items.push(index);
        } else {
            unselected_items.push(index);
        }
    }

    let mut mutation_rates = vec![0; vertex_count];
    for index in 0..vertex_count {
        mutation_rates[index] = challenge.values[index] as i32;
        for &selected in &selected_items {
            mutation_rates[index] += challenge.interaction_values[index][selected];
        }
    }

    let max_generations = 150;
    let mut cooling_schedule = vec![0; vertex_count];
    let mut rng = StdRng::seed_from_u64(challenge.seed[0] as u64);

    for generation in 0..max_generations {
        let mut best_gain = 0;
        let mut best_swap = None;

        for (u_index, &mutant) in unselected_items.iter().enumerate() {
            if cooling_schedule[mutant] > 0 {
                continue;
            }

            let mutant_fitness = mutation_rates[mutant];
            let extra_weight = challenge.weights[mutant] as i32
                - (challenge.max_weight as i32 - current_weight as i32);

            if mutant_fitness < 0 {
                continue;
            }

            for (c_index, &selected) in selected_items.iter().enumerate() {
                if cooling_schedule[selected] > 0 {
                    continue;
                }

                if extra_weight > 0 && (challenge.weights[selected] as i32) < extra_weight {
                    continue;
                }

                let interaction_penalty =
                    (challenge.interaction_values[mutant][selected] as f32 * 0.3) as i32;
                let fitness_gain = mutant_fitness - mutation_rates[selected] - interaction_penalty;

                if fitness_gain > best_gain {
                    best_gain = fitness_gain;
                    best_swap = Some((u_index, c_index));
                }
            }
        }

        if let Some((u_index, c_index)) = best_swap {
            let added_item = unselected_items[u_index];
            let removed_item = selected_items[c_index];

            selected_items.swap_remove(c_index);
            unselected_items.swap_remove(u_index);
            selected_items.push(added_item);
            unselected_items.push(removed_item);

            current_value += best_gain;
            current_weight =
                current_weight + challenge.weights[added_item] - challenge.weights[removed_item];

            if current_weight > challenge.max_weight {
                continue;
            }

            for index in 0..vertex_count {
                mutation_rates[index] += challenge.interaction_values[index][added_item]
                    - challenge.interaction_values[index][removed_item];
            }

            cooling_schedule[added_item] = 3;
            cooling_schedule[removed_item] = 3;
        }

        if current_value as u32 >= challenge.baseline_value {
            let _ = save_solution(&Solution {
                items: selected_items,
            });
            return Ok(());
        }

        for cooling_rate in cooling_schedule.iter_mut() {
            *cooling_rate = if *cooling_rate > 0 {
                *cooling_rate - 1
            } else {
                0
            };
        }

        if current_value as u32 > (challenge.baseline_value * 9 / 10) {
            let high_potential_items: Vec<usize> = unselected_items
                .iter()
                .filter(|&&i| challenge.values[i] as i32 > (challenge.baseline_value as i32 / 4))
                .copied()
                .collect();

            for &item in high_potential_items.iter().take(2) {
                if current_weight + challenge.weights[item] <= challenge.max_weight {
                    selected_items.push(item);
                    unselected_items.retain(|&x| x != item);
                    current_weight += challenge.weights[item];
                    current_value += challenge.values[item] as i32;

                    for &selected in &selected_items {
                        if selected != item {
                            current_value += challenge.interaction_values[item][selected];
                        }
                    }

                    if current_value as u32 >= challenge.baseline_value {
                        let _ = save_solution(&Solution {
                            items: selected_items,
                        });
                        return Ok(());
                    }
                }
            }
        }
    }

    if current_value as u32 >= challenge.baseline_value && current_weight <= challenge.max_weight {
        let _ = save_solution(&Solution {
            items: selected_items,
        });
        return Ok(());
    } else {
        Ok(())
    }
}