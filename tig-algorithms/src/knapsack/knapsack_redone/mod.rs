use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

fn calculate_density_variance(item_densities: &[(usize, f32)]) -> f32 {
    if item_densities.len() < 2 {
        return 0.5;
    }

    let mut sum = 0.0;
    for (_, density) in item_densities {
        sum += *density;
    }
    let mean = sum / item_densities.len() as f32;

    let mut variance_sum = 0.0;
    for (_, density) in item_densities {
        let diff = *density - mean;
        variance_sum += diff * diff;
    }
    let variance = variance_sum / item_densities.len() as f32;

    (variance.sqrt() / mean.abs()).clamp(0.1, 1.0)
}

fn compute_solution(
    challenge: &Challenge,
    contribution_list: &mut [i32],
    unselected_items: &mut Vec<usize>,
    rng: &mut StdRng,
) -> Result<Option<(Solution, i32)>> {
    let mut selected_items = Vec::new();
    let mut total_weight = 0;
    let mut total_value = 0;

    let mut inv_weights: Vec<f32> = Vec::with_capacity(challenge.weights.len());
    for &w in &challenge.weights {
        inv_weights.push(1.0 / w as f32);
    }

    let rcl_max = if challenge.num_items <= 165 {
        9
    } else {
        10
    };

    let mut item_densities: Vec<(usize, f32)> = Vec::with_capacity(unselected_items.len());
    for &idx in unselected_items.iter() {
        let ratio = contribution_list[idx] as f32 * inv_weights[idx];
        item_densities.push((idx, ratio));
    }

    let density_variance = calculate_density_variance(&item_densities);
    let adaptive_exponent = 1.4 + (density_variance - 0.6).clamp(-0.4, 0.2);

    let mut probs: Vec<f32> = Vec::with_capacity(rcl_max);
    for rank in 0..rcl_max {
        probs.push(1.0 / ((rank + 1) as f32).powf(adaptive_exponent));
    }

    let mut acc_probs: Vec<f32> = Vec::with_capacity(rcl_max);
    let mut sum = 0.0;
    for &prob in &probs {
        sum += prob;
        acc_probs.push(sum);
    }
    let total_prob_max = sum;

    let mut max_item_weight = 0;
    for &w in &challenge.weights {
        if w > max_item_weight {
            max_item_weight = w;
        }
    }

    let list_size = 2;
    let mut top_ranks = vec![0; list_size];

    while !item_densities.is_empty() {
        let num_candidates = item_densities.len();
        if num_candidates < 2 {
            break;
        }

        let actual_rcl_size = num_candidates.min(rcl_max);
        let total_prob = if actual_rcl_size == rcl_max {
            total_prob_max
        } else {
            acc_probs[actual_rcl_size - 1]
        };

        let random_threshold = rng.gen_range(0.0..total_prob);
        let mut selected_rank = match acc_probs[..actual_rcl_size]
            .binary_search_by(|prob| prob.partial_cmp(&random_threshold).unwrap())
        {
            Ok(i) | Err(i) => i,
        };
        if selected_rank >= actual_rcl_size {
            selected_rank = actual_rcl_size - 1;
        }

        let selected_item;
        if selected_rank < list_size
            && !selected_items.is_empty()
            && top_ranks[selected_rank] < item_densities.len()
        {
            selected_rank = top_ranks[selected_rank];
            selected_item = item_densities[selected_rank].0;
        } else {
            item_densities
                .select_nth_unstable_by(selected_rank, |a, b| b.1.partial_cmp(&a.1).unwrap());
            selected_item = item_densities[selected_rank].0;
        }

        selected_items.push(selected_item);
        total_weight += challenge.weights[selected_item];
        total_value += contribution_list[selected_item];

        if total_weight + max_item_weight > challenge.max_weight {
            item_densities.retain(|(idx, _)| {
                total_weight + challenge.weights[*idx] <= challenge.max_weight
                    && *idx != selected_item
            });
        } else {
            item_densities.swap_remove(selected_rank);
        }

        unsafe {
            for x in 0..challenge.num_items {
                *contribution_list.get_unchecked_mut(x) += *challenge
                    .interaction_values
                    .get_unchecked(selected_item)
                    .get_unchecked(x);
            }

            let mut first_density = f32::MIN;
            let mut first_rank = 0;
            let mut second_density = f32::MIN;
            let mut second_rank = 0;

            for (i, density) in item_densities.iter_mut().enumerate() {
                let interaction = *challenge
                    .interaction_values
                    .get_unchecked(selected_item)
                    .get_unchecked(density.0);
                density.1 += interaction as f32 * inv_weights[density.0];
                let current_density = density.1;

                if current_density > first_density {
                    second_density = first_density;
                    second_rank = first_rank;
                    first_density = current_density;
                    first_rank = i;
                } else if current_density > second_density {
                    second_density = current_density;
                    second_rank = i;
                }
            }

            top_ranks[0] = first_rank;
            top_ranks[1] = second_rank;
        }
    }
    unselected_items.clear();
    for i in 0..challenge.num_items {
        unselected_items.push(i);
    }

    let mut sorted_selected = selected_items.clone();
    sorted_selected.sort_unstable_by(|a, b| b.cmp(a));

    for &selected in &sorted_selected {
        unselected_items.swap_remove(selected);
    }

    let mut weight_item_pairs: Vec<(u32, usize)> = Vec::with_capacity(unselected_items.len());
    for &idx in unselected_items.iter() {
        weight_item_pairs.push((challenge.weights[idx], idx));
    }
    weight_item_pairs.sort_unstable_by_key(|&(weight, _)| weight);

    unselected_items.clear();
    for (_, idx) in weight_item_pairs {
        unselected_items.push(idx);
    }

    let local_search_iterations = if challenge.num_items <= 165 {
        60
    } else {
        100
    };
    let mut feasible_adds = Vec::with_capacity(50);
    let mut feasible_swaps = Vec::with_capacity(100);

    for _ in 0..local_search_iterations {
        let mut improved = false;

        if total_weight < challenge.max_weight {
            for (i, &cand) in unselected_items.iter().enumerate() {
                let new_w = total_weight + challenge.weights[cand];
                if new_w > challenge.max_weight {
                    break;
                }
                let new_val = total_value + contribution_list[cand];
                if new_val > total_value {
                    feasible_adds.push(i);
                }
            }
            if !feasible_adds.is_empty() {
                let pick = rng.gen_range(0..feasible_adds.len());
                let add_idx = feasible_adds[pick];
                let new_item = unselected_items[add_idx];

                unselected_items.remove(add_idx);
                selected_items.push(new_item);

                total_weight += challenge.weights[new_item];
                total_value += contribution_list[new_item];
                improved = true;

                unsafe {
                    for x in 0..challenge.num_items {
                        *contribution_list.get_unchecked_mut(x) += *challenge
                            .interaction_values
                            .get_unchecked(x)
                            .get_unchecked(new_item);
                    }
                }
            }
            feasible_adds.clear();
        }

        let free_capacity = challenge.max_weight as i32 - total_weight as i32;
        for (j, &rem_item) in selected_items.iter().enumerate() {
            let rem_w = challenge.weights[rem_item] as i32;

            for (i, &cand_item) in unselected_items.iter().enumerate() {
                let cand_w = challenge.weights[cand_item] as i32;
                if rem_w + free_capacity < cand_w {
                    break;
                }

                let val_diff = contribution_list[cand_item]
                    - contribution_list[rem_item]
                    - challenge.interaction_values[cand_item][rem_item];
                if val_diff > 0 {
                    feasible_swaps.push((i, j));
                }
            }
        }

        if !feasible_swaps.is_empty() {
            let pick = rng.gen_range(0..feasible_swaps.len());
            let (unsel_idx, sel_idx) = feasible_swaps[pick];
            let new_item = unselected_items[unsel_idx];
            let remove_item = selected_items[sel_idx];

            selected_items.swap_remove(sel_idx);
            selected_items.push(new_item);

            let new_item_weight = challenge.weights[new_item];
            let remove_item_weight = challenge.weights[remove_item];

            let current_pos = unsel_idx;
            let mut target_pos = current_pos;
            if new_item_weight != remove_item_weight {
                target_pos = unselected_items
                    .binary_search_by(|&probe| challenge.weights[probe].cmp(&remove_item_weight))
                    .unwrap_or_else(|e| e);
            }
            if current_pos != target_pos {
                unsafe {
                    let ptr = unselected_items.as_mut_ptr();
                    if target_pos < current_pos {
                        std::ptr::copy(
                            ptr.add(target_pos),
                            ptr.add(target_pos + 1),
                            current_pos - target_pos,
                        );
                    } else {
                        target_pos = target_pos - 1;
                        std::ptr::copy(
                            ptr.add(current_pos + 1),
                            ptr.add(current_pos),
                            target_pos - current_pos,
                        );
                    }
                }
            }
            unselected_items[target_pos] = remove_item;

            total_value += contribution_list[new_item]
                - contribution_list[remove_item]
                - challenge.interaction_values[new_item][remove_item];
            total_weight =
                total_weight + challenge.weights[new_item] - challenge.weights[remove_item];
            improved = true;

            unsafe {
                for x in 0..challenge.num_items {
                    *contribution_list.get_unchecked_mut(x) += *challenge
                        .interaction_values
                        .get_unchecked(x)
                        .get_unchecked(new_item)
                        - *challenge
                            .interaction_values
                            .get_unchecked(x)
                            .get_unchecked(remove_item);
                }
            }
        }
        feasible_swaps.clear();

        if !improved {
            break;
        }
    }

    if selected_items.is_empty() {
        Ok(None)
    } else {
        Ok(Some((
            Solution {
                items: selected_items,
            },
            total_value,
        )))
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let num_iterations: i32 = 5;
    let mut rng =
        StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut best_solution: Option<Solution> = None;
    let mut best_value = 0;

    for _outer_iter in 0..num_iterations {
        let mut best_local_solution: Option<Solution> = None;
        let mut best_local_value = 0;

        let k = 5;
        for _ in 0..k {
            let mut unselected_items: Vec<usize> =
                Vec::with_capacity(challenge.num_items);
            for i in 0..challenge.num_items {
                unselected_items.push(i);
            }

            let mut contribution_list: Vec<i32> = Vec::with_capacity(challenge.values.len());
            for &v in &challenge.values {
                contribution_list.push(v as i32);
            }

            let sol_result = compute_solution(
                challenge,
                &mut contribution_list,
                &mut unselected_items,
                &mut rng,
            )?;

            let (solution, value) = match sol_result {
                Some(x) => x,
                None => continue,
            };

            if value > best_local_value {
                best_local_value = value;
                best_local_solution = Some(Solution {
                    items: solution.items.clone(),
                });
            }
        }

        if let Some(local_solution) = best_local_solution {
            if best_local_value > best_value {
                best_value = best_local_value;
                best_solution = Some(local_solution);
            }
        }
    }

    if let Some(solution) = best_solution {
        let _ = save_solution(&solution);
    }
    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
