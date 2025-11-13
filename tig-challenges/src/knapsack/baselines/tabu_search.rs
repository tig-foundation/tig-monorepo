use crate::knapsack::{Challenge, Solution};
use anyhow::Result;
use serde_json::{Map, Value};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Precompute the ratio between the total value (value + sum of interactive values) and
    // weight for each item. Pair the ratio with the item's weight and index
    let mut item_values: Vec<(usize, f32)> = (0..challenge.num_items)
        .map(|i| {
            let total_value =
                challenge.values[i] as i32 + challenge.interaction_values[i].iter().sum::<i32>();
            let ratio = total_value as f32 / challenge.weights[i] as f32;
            (i, ratio)
        })
        .collect();

    // Sort the list of ratios in descending order
    item_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Step 1: Initial solution obtained by greedily selecting items based on value-weight ratio
    let mut selected_items = Vec::with_capacity(challenge.num_items);
    let mut unselected_items = Vec::with_capacity(challenge.num_items);
    let mut total_weight = 0;
    let mut is_selected = vec![false; challenge.num_items];
    for &(item, _) in &item_values {
        if total_weight + challenge.weights[item] <= challenge.max_weight {
            total_weight += challenge.weights[item];
            selected_items.push(item);
            is_selected[item] = true;
        } else {
            unselected_items.push(item);
        }
    }

    // Step 2: Improvement of solution with Local Search and Tabu-List
    // Precompute sum of interaction values with each selected item for all items
    let mut interaction_sum_list = vec![0; challenge.num_items];
    for x in 0..challenge.num_items {
        interaction_sum_list[x] = challenge.values[x] as i32;
        for &item in &selected_items {
            interaction_sum_list[x] += challenge.interaction_values[x][item];
        }
    }

    let mut min_selected_item_values = i32::MAX;
    for x in 0..challenge.num_items {
        if is_selected[x] {
            min_selected_item_values = min_selected_item_values.min(interaction_sum_list[x]);
        }
    }

    // Optimized local search with tabu list
    let max_iterations = 100;
    let mut tabu_list = vec![0; challenge.num_items];

    for _ in 0..max_iterations {
        let mut best_improvement = 0;
        let mut best_swap = None;

        for i in 0..unselected_items.len() {
            let new_item = unselected_items[i];
            if tabu_list[new_item] > 0 {
                continue;
            }

            let new_item_values_sum = interaction_sum_list[new_item];
            if new_item_values_sum < best_improvement + min_selected_item_values {
                continue;
            }

            // Compute minimal weight of remove_item required to put new_item
            let min_weight = challenge.weights[new_item] as i32
                - (challenge.max_weight as i32 - total_weight as i32);
            for j in 0..selected_items.len() {
                let remove_item = selected_items[j];
                if tabu_list[remove_item] > 0 {
                    continue;
                }

                // Don't check the weight if there is enough remaining capacity
                if min_weight > 0 {
                    // Skip a remove_item if the remaining capacity after removal is insufficient to push a new_item
                    let removed_item_weight = challenge.weights[remove_item] as i32;
                    if removed_item_weight < min_weight {
                        continue;
                    }
                }

                let remove_item_values_sum = interaction_sum_list[remove_item];
                let value_diff = new_item_values_sum
                    - remove_item_values_sum
                    - challenge.interaction_values[new_item][remove_item];

                if value_diff > best_improvement {
                    best_improvement = value_diff;
                    best_swap = Some((i, j));
                }
            }
        }

        if let Some((unselected_index, selected_index)) = best_swap {
            let new_item = unselected_items[unselected_index];
            let remove_item = selected_items[selected_index];

            selected_items.swap_remove(selected_index);
            unselected_items.swap_remove(unselected_index);
            selected_items.push(new_item);
            unselected_items.push(remove_item);

            is_selected[new_item] = true;
            is_selected[remove_item] = false;

            total_weight =
                total_weight + challenge.weights[new_item] - challenge.weights[remove_item];

            // Update sum of interaction values after swapping items
            min_selected_item_values = i32::MAX;
            for x in 0..challenge.num_items {
                interaction_sum_list[x] += challenge.interaction_values[x][new_item]
                    - challenge.interaction_values[x][remove_item];
                if is_selected[x] {
                    min_selected_item_values =
                        min_selected_item_values.min(interaction_sum_list[x]);
                }
            }

            // Update tabu list
            tabu_list[new_item] = 3;
            tabu_list[remove_item] = 3;
        } else {
            break; // No improvement found, terminate local search
        }

        // Decrease tabu counters
        for t in tabu_list.iter_mut() {
            *t = if *t > 0 { *t - 1 } else { 0 };
        }
    }

    let _ = save_solution(&Solution {
        items: selected_items,
    });
    Ok(())
}
