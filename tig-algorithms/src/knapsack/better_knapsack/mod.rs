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

    fn try_ejection(
        selected_items: &mut Vec<usize>,
        unselected_items: &mut Vec<usize>,
        total_weight: &mut u32,
        total_value: &mut i32,
        best_solution: &mut Vec<usize>,
        best_solution_value: &mut i32,
        ejection_candidates: &[(usize, i32, u32, f32)],
        target: usize,
        target_value: i32,
        _target_weight: u32,
        needed_space: i32,
        contribution_list: &mut [i32],
        challenge: &SubInstance,
        improved: &mut bool,
        no_improve_count: &mut i32,
        rng: &mut StdRng
    ) -> bool {
        let mut smart_candidates: Vec<(usize, i32, u32, f32, i32)> = Vec::with_capacity(ejection_candidates.len());
    
        for &(idx, value, weight, ratio) in ejection_candidates {
            let item = selected_items[idx];
            let mut item_interactions = 0;
            for (other_idx, &other_item) in selected_items.iter().enumerate() {
                if other_idx != idx {
                    item_interactions += unsafe { *challenge.interaction_values.get_unchecked(item).get_unchecked(other_item) };
                }
            }
        
            let target_interaction = unsafe { *challenge.interaction_values.get_unchecked(item).get_unchecked(target) };
        
            smart_candidates.push((idx, value, weight, ratio, item_interactions - target_interaction));
        }
    
        smart_candidates.sort_unstable_by(|a, b| {
            a.4.cmp(&b.4).then_with(|| {
                a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal)
            })
        });
    
        let mut weight_freed = 0;
        let mut value_lost = 0;
        let mut items_to_eject = Vec::with_capacity(ejection_candidates.len());
    
        for &(idx, value, weight, _, _) in &smart_candidates {
            items_to_eject.push(idx);
            weight_freed += weight as i32;
            value_lost += value;
        
            if weight_freed >= needed_space {
                break;
            }
        }
    
        if weight_freed < needed_space {
            return false;
        }
    
        let mut items_after_ejection = selected_items.clone();
        items_to_eject.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &items_to_eject {
            items_after_ejection.swap_remove(idx);
        }
    
        let mut interaction_gain = 0;
        for item in &items_after_ejection {
            interaction_gain += unsafe { *challenge.interaction_values.get_unchecked(target).get_unchecked(*item) };
        }
    
        let current_ratio = *total_value as f32 / challenge.baseline_value as f32;
        let acceptance_threshold = if current_ratio < 1.0 {
            0.60 - (*no_improve_count as f32 * 0.05).min(0.2)
        } else if current_ratio < 1.004 {
            0.65 - (*no_improve_count as f32 * 0.03).min(0.15)
        } else if current_ratio < 1.008 {
            0.70
        } else {
            0.75
        };
    
        if target_value + interaction_gain > value_lost || 
           (target_value + interaction_gain >= value_lost * (acceptance_threshold * 100.0) as i32 / 100 && 
           rng.gen::<f32>() < 0.98) {
            let mut ejected_items = Vec::with_capacity(items_to_eject.len());
            for &idx in &items_to_eject {
                let ejected_item = selected_items.swap_remove(idx);
                ejected_items.push(ejected_item);
            
                *total_weight -= unsafe { *challenge.weights.get_unchecked(ejected_item) };
            
                unsafe {
                    for x in 0..challenge.num_items {
                        *contribution_list.get_unchecked_mut(x) -= *challenge.interaction_values.get_unchecked(ejected_item).get_unchecked(x);
                    }
                }
            }
        
            selected_items.push(target);
        
            unselected_items.retain(|&item| item != target);
            for ejected_item in ejected_items {
                unselected_items.push(ejected_item);
            }
        
            *total_weight += unsafe { *challenge.weights.get_unchecked(target) };
        
            unsafe {
                for x in 0..challenge.num_items {
                    *contribution_list.get_unchecked_mut(x) += *challenge.interaction_values.get_unchecked(target).get_unchecked(x);
                }
            }
        
            let _old_value = *total_value;
            *total_value = 0;
            for item in selected_items.iter() {
                *total_value += unsafe { *challenge.values.get_unchecked(*item) as i32 };
            }
        
            for i in 0..selected_items.len() {
                for j in (i+1)..selected_items.len() {
                    let item1 = selected_items[i];
                    let item2 = selected_items[j];
                    *total_value += unsafe { *challenge.interaction_values.get_unchecked(item1).get_unchecked(item2) };
                }
            }
        
            *improved = true;
            *no_improve_count = 0;
        
            if *total_value > *best_solution_value {
                *best_solution = selected_items.clone();
                *best_solution_value = *total_value;
            }
        
            unselected_items.sort_unstable_by_key(|&idx| unsafe { *challenge.weights.get_unchecked(idx) });
        
            return true;
        }
    
        return false;
    }

    fn try_swap_pair(
        selected_items: &mut Vec<usize>,
        unselected_items: &mut Vec<usize>,
        total_weight: &mut u32,
        total_value: &mut i32,
        best_solution: &mut Vec<usize>,
        best_solution_value: &mut i32,
        contribution_list: &mut [i32],
        challenge: &SubInstance,
        rng: &mut StdRng
    ) -> bool {
        if selected_items.len() < 2 || unselected_items.len() < 2 {
            return false;
        }

        let items_to_check = 4;
    
        let sel_count = std::cmp::min(selected_items.len(), items_to_check);
        let unsel_count = std::cmp::min(unselected_items.len(), items_to_check);
    
        let mut selected_indices = Vec::with_capacity(sel_count);
        let mut unselected_indices = Vec::with_capacity(unsel_count);
    
        for _ in 0..sel_count {
            let idx = rng.gen_range(0..selected_items.len());
            if !selected_indices.contains(&idx) {
                selected_indices.push(idx);
            }
        }
    
        for _ in 0..unsel_count {
            let idx = rng.gen_range(0..unselected_items.len());
            if !unselected_indices.contains(&idx) {
                unselected_indices.push(idx);
            }
        }
    
        for i in 0..selected_indices.len() {
            for j in (i+1)..selected_indices.len() {
                let sel_idx1 = selected_indices[i];
                let sel_idx2 = selected_indices[j];
            
                let item1 = selected_items[sel_idx1];
                let item2 = selected_items[sel_idx2];
            
                let combined_weight = unsafe { 
                    *challenge.weights.get_unchecked(item1) + *challenge.weights.get_unchecked(item2) 
                };
                let combined_value = unsafe { 
                    *contribution_list.get_unchecked(item1) + *contribution_list.get_unchecked(item2) 
                };
                let internal_interaction = unsafe { 
                    *challenge.interaction_values.get_unchecked(item1).get_unchecked(item2) 
                };
            
                for u in 0..unselected_indices.len() {
                    for v in (u+1)..unselected_indices.len() {
                        let unsel_idx1 = unselected_indices[u];
                        let unsel_idx2 = unselected_indices[v];
                    
                        let cand1 = unselected_items[unsel_idx1];
                        let cand2 = unselected_items[unsel_idx2];
                    
                        let new_combined_weight = unsafe { 
                            *challenge.weights.get_unchecked(cand1) + *challenge.weights.get_unchecked(cand2) 
                        };
                    
                        if *total_weight - combined_weight + new_combined_weight > challenge.max_weight {
                            continue;
                        }
                    
                        let new_combined_value = unsafe { 
                            *contribution_list.get_unchecked(cand1) + *contribution_list.get_unchecked(cand2) 
                        };
                        let new_internal_interaction = unsafe { 
                            *challenge.interaction_values.get_unchecked(cand1).get_unchecked(cand2) 
                        };
                    
                        let mut interaction_diff = new_internal_interaction - internal_interaction;
                    
                        for (idx, &sel_item) in selected_items.iter().enumerate() {
                            if idx != sel_idx1 && idx != sel_idx2 {
                                unsafe {
                                    interaction_diff -= *challenge.interaction_values.get_unchecked(item1).get_unchecked(sel_item);
                                    interaction_diff -= *challenge.interaction_values.get_unchecked(item2).get_unchecked(sel_item);
                                
                                    interaction_diff += *challenge.interaction_values.get_unchecked(cand1).get_unchecked(sel_item);
                                    interaction_diff += *challenge.interaction_values.get_unchecked(cand2).get_unchecked(sel_item);
                                }
                            }
                        }
                    
                        let total_diff = (new_combined_value - combined_value) + interaction_diff;
                    
                        if total_diff > 0 || (total_diff >= -5 && rng.gen::<f32>() < 0.1) {
                            let sel_item1 = selected_items.swap_remove(sel_idx2);
                            let sel_item2 = selected_items.swap_remove(sel_idx1);
                        
                            let unsel_item1 = unselected_items.swap_remove(unsel_idx2);
                            let unsel_item2 = unselected_items.swap_remove(unsel_idx1);
                        
                            selected_items.push(unsel_item1);
                            selected_items.push(unsel_item2);
                        
                            unselected_items.push(sel_item1);
                            unselected_items.push(sel_item2);
                        
                            unselected_items.sort_unstable_by_key(|&idx| unsafe { *challenge.weights.get_unchecked(idx) });
                        
                            *total_weight = *total_weight - combined_weight + new_combined_weight;
                        
                            unsafe {
                                for x in 0..challenge.num_items {
                                    *contribution_list.get_unchecked_mut(x) -= *challenge.interaction_values.get_unchecked(sel_item1).get_unchecked(x);
                                    *contribution_list.get_unchecked_mut(x) -= *challenge.interaction_values.get_unchecked(sel_item2).get_unchecked(x);
                                    *contribution_list.get_unchecked_mut(x) += *challenge.interaction_values.get_unchecked(unsel_item1).get_unchecked(x);
                                    *contribution_list.get_unchecked_mut(x) += *challenge.interaction_values.get_unchecked(unsel_item2).get_unchecked(x);
                                }
                            }
                        
                            *total_value += total_diff;
                        
                            if *total_value > *best_solution_value {
                                *best_solution = selected_items.clone();
                                *best_solution_value = *total_value;
                            }
                        
                            return true;
                        }
                    }
                }
            }
        }
    
        return false;
    }

    fn compute_solution(
        challenge: &SubInstance,
        contribution_list: &mut [i32],
        unselected_items: &mut Vec<usize>,
        rng: &mut StdRng,
    ) -> Result<Option<(SubSolution, i32)>> {
        let mut selected_items = Vec::with_capacity(challenge.num_items / 4);
        let mut total_weight = 0;
        let mut total_value = 0;

        let inv_weights: Vec<f32> = challenge.weights.iter().map(|&w| 1.0 / w as f32).collect();

        const RCL_MAX: usize = 12;

        let probs: Vec<f32> = (0..RCL_MAX)
            .map(|rank| 1.0 / ((rank + 1) as f32).exp())
            .collect();

        let mut acc_probs: Vec<f32> = Vec::with_capacity(RCL_MAX);
        let mut sum = 0.0;
        for &prob in &probs {
            sum += prob;
            acc_probs.push(sum);
        }
        let total_prob_max = sum;
        let max_item_weight = challenge.weights.iter().max().unwrap();

        let mut item_densities: Vec<(usize, f32)> = Vec::with_capacity(unselected_items.len());
        for &idx in unselected_items.iter() {
            let ratio = unsafe { *contribution_list.get_unchecked(idx) as f32 * inv_weights[idx] };
            item_densities.push((idx, ratio));
        }

        let list_size = 3;
        let mut top_ranks = vec![0; list_size];
        let mut top_densities = vec![f32::MIN; list_size];

        while !item_densities.is_empty() {
            let num_candidates = item_densities.len();
            if num_candidates < 2 {
                break;
            }

            let actual_rcl_size = num_candidates.min(RCL_MAX);
            let total_prob = if actual_rcl_size == RCL_MAX {
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
            if selected_rank < list_size && !selected_items.is_empty() {
                selected_rank = top_ranks[selected_rank];
                selected_item = item_densities[selected_rank].0;
            } else {
                item_densities
                    .select_nth_unstable_by(selected_rank, |a, b| b.1.partial_cmp(&a.1).unwrap());
                selected_item = item_densities[selected_rank].0;
            }

            selected_items.push(selected_item);
            total_weight += unsafe { *challenge.weights.get_unchecked(selected_item) };
            total_value += unsafe { *contribution_list.get_unchecked(selected_item) };

            if total_weight + max_item_weight > challenge.max_weight {
                item_densities.retain(|(idx, _)| {
                    total_weight + unsafe { *challenge.weights.get_unchecked(*idx) } <= challenge.max_weight
                        && *idx != selected_item
                });
            } else {
                item_densities.swap_remove(selected_rank);
            }

            unsafe {
                for x in 0..challenge.num_items {
                    *contribution_list.get_unchecked_mut(x) += *challenge.interaction_values.get_unchecked(selected_item).get_unchecked(x);
                }
            }

            let mut first_density = f32::MIN;
            let mut first_rank = 0;
            let mut second_density = f32::MIN;
            let mut second_rank = 0;
            let mut third_density = f32::MIN;
            let mut third_rank = 0;

            for (i, density) in item_densities.iter_mut().enumerate() {
                let interaction = unsafe { *challenge.interaction_values.get_unchecked(selected_item).get_unchecked(density.0) };
            
                density.1 += interaction as f32 * inv_weights[density.0];
                let current_density = density.1;

                if current_density > first_density {
                    third_density = second_density;
                    third_rank = second_rank;
                    second_density = first_density;
                    second_rank = first_rank;
                    first_density = current_density;
                    first_rank = i;
                } else if current_density > second_density {
                    third_density = second_density;
                    third_rank = second_rank;
                    second_density = current_density;
                    second_rank = i;
                } else if current_density > third_density {
                    third_density = current_density;
                    third_rank = i;
                }
            }

            top_ranks[0] = first_rank;
            top_ranks[1] = second_rank;
            top_ranks[2] = third_rank;
            top_densities[0] = first_density;
            top_densities[1] = second_density;
            top_densities[2] = third_density;
        }
    
        unselected_items.clear();
        unselected_items.extend(0..challenge.num_items);

        let mut sorted_selected = selected_items.clone();
        sorted_selected.sort_unstable_by(|a, b| b.cmp(a));

        for &selected in &sorted_selected {
            unselected_items.swap_remove(selected);
        }

        unselected_items.sort_unstable_by_key(|&idx| unsafe { *challenge.weights.get_unchecked(idx) });

        let mut best_solution = selected_items.clone();
        let mut best_solution_value = total_value;
    
        let local_search_iterations = 100;
        let mut feasible_adds = Vec::with_capacity(16);
        let mut feasible_swaps = Vec::with_capacity(32);
    
        let mut no_improve_count = 0;
        let max_no_improve = 3;
    
        let mut total_improvements = 0;
    
        for iter_count in 0..local_search_iterations {
            let mut improved = false;

            if total_weight < challenge.max_weight {
                for (i, &cand) in unselected_items.iter().enumerate() {
                    let new_w = total_weight + unsafe { *challenge.weights.get_unchecked(cand) };
                    let new_val = total_value + unsafe { *contribution_list.get_unchecked(cand) };
                    if new_w > challenge.max_weight {
                        break;
                    }

                    if new_val >= total_value {
                        feasible_adds.push(i);
                    }
                }
                if !feasible_adds.is_empty() {
                    let pick = if rng.gen::<f32>() < 0.1 + (iter_count as f32 / local_search_iterations as f32) * 0.5 {
                        let mut best_idx = 0;
                        let mut best_gain = 0;
                        for &add_idx in &feasible_adds {
                            let item = unselected_items[add_idx];
                            let gain = unsafe { *contribution_list.get_unchecked(item) };
                            if gain > best_gain {
                                best_gain = gain;
                                best_idx = add_idx;
                            }
                        }
                        best_idx
                    } else {
                        rng.gen_range(0..feasible_adds.len())
                    };
                
                    let add_idx = feasible_adds[pick];
                    let new_item = unselected_items[add_idx];

                    unselected_items.remove(add_idx);
                    selected_items.push(new_item);

                    total_weight += unsafe { *challenge.weights.get_unchecked(new_item) };
                    total_value += unsafe { *contribution_list.get_unchecked(new_item) };
                    improved = true;
                    no_improve_count = 0;

                    unsafe {
                        for x in 0..challenge.num_items {
                            *contribution_list.get_unchecked_mut(x) += *challenge.interaction_values.get_unchecked(x).get_unchecked(new_item);
                        }
                    }
                
                    if total_value > best_solution_value {
                        best_solution = selected_items.clone();
                        best_solution_value = total_value;
                    }
                }
                feasible_adds.clear();
            }

            if !improved || rng.gen::<f32>() < 0.6 {
                let free_capacity = challenge.max_weight as i32 - total_weight as i32;
                for (j, &rem_item) in selected_items.iter().enumerate() {
                    let rem_w = unsafe { *challenge.weights.get_unchecked(rem_item) } as i32;
                    let rem_val = unsafe { *contribution_list.get_unchecked(rem_item) };

                    for (i, &cand_item) in unselected_items.iter().enumerate() {
                        let cand_w = unsafe { *challenge.weights.get_unchecked(cand_item) } as i32;
                        if rem_w + free_capacity < cand_w {
                            break;
                        }

                        let cand_val = unsafe { *contribution_list.get_unchecked(cand_item) };
                    
                        if cand_val < rem_val * 98 / 100 && cand_w >= rem_w {
                            continue;
                        }

                        let val_diff = cand_val - rem_val - unsafe { *challenge.interaction_values.get_unchecked(cand_item).get_unchecked(rem_item) };
                    
                        if val_diff >= 0 || (val_diff >= -5 && rng.gen::<f32>() < 0.2) {
                            feasible_swaps.push((i, j, val_diff));
                        }
                    }
                }

                if !feasible_swaps.is_empty() {
                    feasible_swaps.sort_unstable_by(|a, b| b.2.cmp(&a.2));
                
                    let top_count = feasible_swaps.len().min(5);
                    let pick_idx = if rng.gen::<f32>() < 0.7 {
                        0
                    } else {
                        rng.gen_range(0..top_count)
                    };
                
                    let (unsel_idx, sel_idx, _) = feasible_swaps[pick_idx];
                    let new_item = unselected_items[unsel_idx];
                    let remove_item = selected_items[sel_idx];

                    selected_items.swap_remove(sel_idx);
                    selected_items.push(new_item);

                    let new_item_weight = unsafe { *challenge.weights.get_unchecked(new_item) };
                    let remove_item_weight = unsafe { *challenge.weights.get_unchecked(remove_item) };

                    let current_pos = unsel_idx;
                    let mut target_pos = current_pos;
                    if new_item_weight != remove_item_weight {
                        target_pos = unselected_items
                            .binary_search_by(|&probe| unsafe { challenge.weights.get_unchecked(probe).cmp(challenge.weights.get_unchecked(remove_item)) })
                            .unwrap_or_else(|e| e);
                    }
                    if current_pos != target_pos {
                        let ptr = unselected_items.as_mut_ptr();
                        if target_pos < current_pos {
                            unsafe {
                                std::ptr::copy(
                                    ptr.add(target_pos),
                                    ptr.add(target_pos + 1),
                                    current_pos - target_pos,
                                );
                            }
                        } else {
                            target_pos = target_pos - 1;
                            unsafe {
                                std::ptr::copy(
                                    ptr.add(current_pos + 1),
                                    ptr.add(current_pos),
                                    target_pos - current_pos,
                                );
                            }
                        }
                    }
                    unselected_items[target_pos] = remove_item;

                    total_value += unsafe { 
                        *contribution_list.get_unchecked(new_item)
                            - *contribution_list.get_unchecked(remove_item)
                            - *challenge.interaction_values.get_unchecked(new_item).get_unchecked(remove_item)
                    };
                    total_weight =
                        total_weight + new_item_weight - remove_item_weight;
                    improved = true;
                    no_improve_count = 0;

                    unsafe {
                        for x in 0..challenge.num_items {
                            *contribution_list.get_unchecked_mut(x) += *challenge.interaction_values.get_unchecked(x).get_unchecked(new_item)
                                - *challenge.interaction_values.get_unchecked(x).get_unchecked(remove_item);
                        }
                    }
                
                    if total_value > best_solution_value {
                        best_solution = selected_items.clone();
                        best_solution_value = total_value;
                    }
                }
                feasible_swaps.clear();
            }

            if !improved {
                no_improve_count += 1;
            
                if no_improve_count >= max_no_improve || iter_count % std::cmp::max(15 - no_improve_count * 2, 5) == 0 {
                    let mut high_value_items = Vec::with_capacity(challenge.num_items / 2);
                
                    for &item in unselected_items.iter() {
                        let item_weight = unsafe { *challenge.weights.get_unchecked(item) };
                    
                        if total_weight + item_weight <= challenge.max_weight {
                            continue;
                        }
                    
                        let item_value = unsafe { *contribution_list.get_unchecked(item) };
                        let value_per_weight = item_value as f32 / item_weight as f32;
                    
                        if value_per_weight > 0.0 {
                            high_value_items.push((item, item_value, item_weight, value_per_weight));
                        }
                    }
                
                    if no_improve_count > 1 && rng.gen::<f32>() < 0.3 {
                        high_value_items.sort_by(|a, b| {
                            let noise_a = a.3 * (0.9 + rng.gen::<f32>() * 0.2);
                            let noise_b = b.3 * (0.9 + rng.gen::<f32>() * 0.2);
                            noise_b.partial_cmp(&noise_a).unwrap()
                        });
                    } else {
                        high_value_items.sort_unstable_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
                    }
                
                    let candidates_to_try = high_value_items.len().min(12);
                
                    for candidate_idx in 0..candidates_to_try {
                        let (target, target_value, target_weight, _) = high_value_items[candidate_idx];
                        let needed_space = (total_weight + target_weight) as i32 - challenge.max_weight as i32;
                    
                        if needed_space <= 0 {
                            continue;
                        }
                    
                        let mut ejection_candidates: Vec<(usize, i32, u32, f32)> = Vec::with_capacity(selected_items.len());
                    
                        for (idx, &item) in selected_items.iter().enumerate() {
                            let item_weight = unsafe { *challenge.weights.get_unchecked(item) };
                            let effective_value = unsafe { *contribution_list.get_unchecked(item) };
                            let value_per_weight = effective_value as f32 / item_weight as f32;
                        
                            ejection_candidates.push((idx, effective_value, item_weight, value_per_weight));
                        }
                    
                                            ejection_candidates.sort_unstable_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
                    
                        for max_items_to_remove in 1..=6.min(ejection_candidates.len()) {
                            if try_ejection(&mut selected_items, unselected_items, &mut total_weight,
                                          &mut total_value, &mut best_solution, &mut best_solution_value,
                                          &ejection_candidates[0..max_items_to_remove], target, target_value, 
                                          target_weight, needed_space, contribution_list, challenge, &mut improved,
                                          &mut no_improve_count, rng) {
                                break;
                            }
                        
                            if max_items_to_remove > 1 {
                                let mut weight_sorted = ejection_candidates.clone();
                                weight_sorted.sort_unstable_by(|a, b| a.2.cmp(&b.2));
                            
                                if try_ejection(&mut selected_items, unselected_items, &mut total_weight,
                                              &mut total_value, &mut best_solution, &mut best_solution_value,
                                              &weight_sorted[0..max_items_to_remove], target, target_value, 
                                              target_weight, needed_space, contribution_list, challenge, &mut improved,
                                              &mut no_improve_count, rng) {
                                    break;
                                }
                            
                                if max_items_to_remove > 2 {
                                    let half_size = max_items_to_remove / 2;
                                    let mut mixed = Vec::with_capacity(max_items_to_remove);
                                    for i in 0..half_size {
                                        mixed.push(ejection_candidates[i]);
                                    }
                                    for i in 0..(max_items_to_remove - half_size) {
                                        mixed.push(weight_sorted[i]);
                                    }
                                
                                    if try_ejection(&mut selected_items, unselected_items, &mut total_weight,
                                                  &mut total_value, &mut best_solution, &mut best_solution_value,
                                                  &mixed, target, target_value, target_weight, needed_space,
                                                  contribution_list, challenge, &mut improved, &mut no_improve_count, rng) {
                                        break;
                                    }
                                }
                            }
                        }
                    
                        if improved {
                            break;
                        }
                    }
                }
            }
        
            if !improved && rng.gen::<f32>() < 0.15 + (no_improve_count as f32 * 0.05).min(0.2) {
                if try_swap_pair(&mut selected_items, unselected_items, &mut total_weight,
                              &mut total_value, &mut best_solution, &mut best_solution_value,
                              contribution_list, challenge, rng) {
                    improved = true;
                    no_improve_count = 0;
                }
            }
        
            if improved {
                total_improvements += 1;
            }
        
            if !improved && total_improvements > 0 {
                break;
            }
        }
    
        if best_solution_value > 0 {
            return Ok(Some((
                SubSolution {
                    items: best_solution,
                },
                best_solution_value,
            )));
        }

        if selected_items.is_empty() {
            Ok(None)
        } else {
            Ok(Some((
                SubSolution {
                    items: selected_items,
                },
                total_value,
            )))
        }
    }

    pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
        let difficulty_level = challenge.difficulty.better_than_baseline;
        let min_ratio = 1.0 + difficulty_level as f64 / 1000.0;
    
        let mut solution = Solution {
            sub_solutions: Vec::with_capacity(challenge.sub_instances.len()),
        };
    
        for _ in 0..challenge.sub_instances.len() {
            solution.sub_solutions.push(SubSolution { items: Vec::new() });
        }

        let mut ratio_indices: Vec<(f64, usize)> = Vec::with_capacity(challenge.sub_instances.len());

        for (index, sub_instance) in challenge.sub_instances.iter().enumerate() {
            match solve_sub_instance(sub_instance, 1)? {
                Some((_, best_value)) => {
                    let threshold_multiplier = 0.011;
                
                    let upper_ratio = best_value as f64 * 
                        (1.0 + threshold_multiplier * lookup_threshold(challenge.num_items) as f64) / 
                        sub_instance.baseline_value as f64;
                
                    ratio_indices.push((upper_ratio, index));
                },
                None => {
                    return Ok(None);
                },
            }
        }

        ratio_indices.sort_unstable_by(|a, b| 
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        );

        let mut ratio_threshold = min_ratio;
        let average_ratio_sqr = 16.0 * ratio_threshold * ratio_threshold;
        let mut sum_of_ratios_sqr = 0.0;
        let mut instance_i = 0;
    
        for &(upper_ratio, index) in &ratio_indices {
            let ratio_threshold_sqr = (average_ratio_sqr - sum_of_ratios_sqr) / (16.0 - instance_i as f64);
            ratio_threshold = ratio_threshold_sqr.sqrt();
        
            if upper_ratio < ratio_threshold {
                return Ok(None);
            }

            let sub_instance = &challenge.sub_instances[index];
        
            let iterations = 20;
        
            match solve_sub_instance(sub_instance, iterations)? {
                Some((sub_solution, best_value)) => {
                    let ratio = best_value as f64 / sub_instance.baseline_value as f64;
                    let ratio_sqr = ratio * ratio;
                    sum_of_ratios_sqr += ratio_sqr;
                
                    solution.sub_solutions[index] = sub_solution;
                },
                None => {
                    return Ok(None);
                },
            }
        
            instance_i += 1;
        }
    
        let avg_ratio = (sum_of_ratios_sqr / 16.0).sqrt();

        if avg_ratio < min_ratio {
            return Ok(None);
        }
    
        Ok(Some(solution))
    }

    fn solve_sub_instance(challenge: &SubInstance, num_iterations: i32) -> Result<Option<(SubSolution, i32)>> {
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));
        let mut best_solution: Option<SubSolution> = None;
        let mut best_value = 0;

        for _ in 0..num_iterations {
            let mut unselected_items: Vec<usize> = Vec::with_capacity(challenge.num_items);
            unselected_items.extend(0..challenge.num_items);

            let mut contribution_list = Vec::with_capacity(challenge.num_items);
            contribution_list.extend(challenge.values.iter().map(|&v| v as i32));

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

            if value > best_value {
                best_value = value;
                best_solution = Some(SubSolution {
                    items: solution.items.clone(),
                });
            }
        }

        match best_solution {
            Some(solution) => Ok(Some((solution, best_value))),
            None => Ok(None),
        }
    }

    fn lookup_threshold(num_items: usize) -> f32 {
        let points = vec![
            (100, 1.071),
            (105, 1.015),
            (110, 0.973),
            (120, 0.882),
            (125, 0.791),
            (130, 0.770),
            (135, 0.760),
            (140, 0.749),
            (145, 0.700),
            (150, 0.616),
            (155, 0.574),
            (160, 0.532),
            (165, 0.511),
            (170, 0.494),
            (175, 0.485),
            (180, 0.476),
            (190, 0.448),
            (195, 0.434),
            (200, 0.427),
            (205, 0.420),
            (210, 0.420),
            (215, 0.385),
            (220, 0.350),
            (225, 0.347),
            (230, 0.343),
            (235, 0.343),
            (240, 0.338),
            (245, 0.334),
            (250, 0.329),
        ];

        points
            .iter()
            .filter(|&&(x, _)| x <= num_items)
            .max_by_key(|&&(x, _)| x)
            .unwrap()
            .1
    }
}

pub fn help() {
    println!("No help information available.");
}
