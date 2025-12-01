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

    pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    
        let mut solution = Solution {
            sub_solutions: Vec::with_capacity(challenge.sub_instances.len()),
        };    
    
        for sub_instance in &challenge.sub_instances {
            match solve_sub_instance(sub_instance)? {
                Some(sub_solution) => solution.sub_solutions.push(sub_solution),
                None => {
                
                    return Ok(None);
                }
            }
        }
    
        Ok(Some(solution))
    }

    fn solve_sub_instance(sub_instance: &SubInstance) -> Result<Option<SubSolution>> {
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(
            sub_instance.seed[..8].try_into().unwrap(),
        ));
    
        let vertex_count = sub_instance.weights.len();
        let mut best_solution: Option<SubSolution> = None;
        let mut best_value = 0;     
    
        let target_ratio = 1.0 + (sub_instance.difficulty.better_than_baseline as f32 / 100.0);
    
        let early_exit_ratio = target_ratio + 0.1;
    
        let mut no_improvement_count = 0;
        let max_no_improvement = 30;
        let mut best_items = Vec::new(); 
    
        for _outer_iter in 0..200 {
            let mut unselected_items: Vec<usize> = (0..vertex_count).collect();
            let mut contribution_list = sub_instance
                .values
                .iter()
                .map(|&v| v as i32)
                .collect::<Vec<i32>>();
        
            if no_improvement_count > 0 && !best_items.is_empty() {
                let mut selected = best_items.clone();
            
                let remove_count = (selected.len() as f32 * rng.gen_range(0.1..0.3)) as usize;
                for _ in 0..remove_count {
                    if selected.is_empty() { break; }
                    let idx = rng.gen_range(0..selected.len());
                    selected.swap_remove(idx);
                }
            
                let sol_result = compute_solution_with_seed(
                    sub_instance, 
                    &mut contribution_list, 
                    &mut unselected_items, 
                    &mut rng,
                    &selected
                )?;
            
                match sol_result {
                    Some((solution, value)) => {
                        if value > best_value {
                            best_value = value;
                            best_solution = Some(SubSolution { items: solution.items.clone() });
                            best_items = solution.items.clone();
                            no_improvement_count = 0; 
                        
                            let current_ratio = best_value as f32 / sub_instance.baseline_value as f32;
                            if current_ratio > early_exit_ratio {
                                return Ok(best_solution);
                            }
                        } else {
                            no_improvement_count += 1;
                        }
                    },
                    None => {
                        no_improvement_count += 1;
                        continue;
                    }
                }
            } else {
                let sol_result = compute_solution(
                    sub_instance, 
                    &mut contribution_list, 
                    &mut unselected_items, 
                    &mut rng
                )?;
            
                match sol_result {
                    Some((solution, value)) => {
                        if value > best_value {
                            best_value = value;
                            best_solution = Some(SubSolution { items: solution.items.clone() });
                            best_items = solution.items.clone();
                            no_improvement_count = 0; 
                        
                            let current_ratio = best_value as f32 / sub_instance.baseline_value as f32;
                            if current_ratio > early_exit_ratio {
                                return Ok(best_solution);
                            }
                        } else {
                            no_improvement_count += 1;

                            if no_improvement_count >= max_no_improvement && best_items.is_empty() {
                                best_items = solution.items.clone();
                            }
                        }
                    },
                    None => {
                        no_improvement_count += 1;
                        continue;
                    }
                }
            }
        }
    
        Ok(best_solution)
    }

    fn compute_solution_with_seed(
        sub_instance: &SubInstance,
        contribution_list: &mut [i32],
        unselected_items: &mut Vec<usize>,
        rng: &mut StdRng,
        seed_items: &[usize]
    ) -> Result<Option<(ItemSelection, i32)>> {
        let mut selected_items = seed_items.to_vec();
        let mut total_weight = 0;
        let mut total_value = 0;
    
        for &item in &selected_items {
            total_weight += sub_instance.weights[item];
            total_value += sub_instance.values[item] as i32;
        }
    
        for i in 0..selected_items.len() {
            for j in i+1..selected_items.len() {
                let a = selected_items[i].min(selected_items[j]);
                let b = selected_items[i].max(selected_items[j]);
                total_value += sub_instance.interaction_values[a][b];
            }
        }
    
        for &selected_item in &selected_items {
            for x in 0..contribution_list.len() {
                contribution_list[x] += sub_instance.interaction_values[selected_item.min(x)][selected_item.max(x)];
            }
        }
    
        unselected_items.clear();
        unselected_items.extend(0..sub_instance.weights.len());
    
        for &selected in &selected_items {
            let idx = unselected_items.iter().position(|&x| x == selected);
            if let Some(pos) = idx {
                unselected_items.swap_remove(pos);
            }
        }
    
        let vertex_count = sub_instance.weights.len();
        let inv_weights: Vec<f32> = sub_instance.weights.iter().map(|&w| 1.0 / w as f32).collect();
    
        let mut item_densities: Vec<(usize, f32)> = unselected_items
            .iter()
            .map(|&idx| {
                let ratio = contribution_list[idx] as f32 * inv_weights[idx];
                (idx, ratio)
            })
            .collect();
    
        const RCL_MAX: usize = 10;
    
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
        let max_item_weight = sub_instance.weights.iter().max().unwrap();
    
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
            let selected_rank = match acc_probs[..actual_rcl_size].binary_search_by(|prob| {
                prob.partial_cmp(&random_threshold).unwrap()
            }) {
                Ok(i) | Err(i) => i.min(actual_rcl_size - 1)
            };
        
            item_densities.select_nth_unstable_by(selected_rank, |a, b| {
                b.1.partial_cmp(&a.1).unwrap()
            });
            let selected_item = item_densities[selected_rank].0;
        
            selected_items.push(selected_item);
            total_weight += sub_instance.weights[selected_item];
            total_value += contribution_list[selected_item];
        
            if total_weight + max_item_weight > sub_instance.max_weight {
                item_densities.retain(|(idx, _)| {
                    total_weight + sub_instance.weights[*idx] <= sub_instance.max_weight && *idx != selected_item
                });
            } else {
                item_densities.swap_remove(selected_rank);
            }
        
            for x in 0..vertex_count {
                contribution_list[x] += sub_instance.interaction_values[selected_item.min(x)][selected_item.max(x)];
            }
        
            for density in item_densities.iter_mut() {
                let interaction = sub_instance.interaction_values[selected_item.min(density.0)][selected_item.max(density.0)];
                density.1 += interaction as f32 * inv_weights[density.0];
            }
        }
    
        unselected_items.clear();
        unselected_items.extend(0..vertex_count);
    
        for &selected in &selected_items {
            let idx = unselected_items.iter().position(|&x| x == selected);
            if let Some(pos) = idx {
                unselected_items.swap_remove(pos);
            }
        }
    
        unselected_items.sort_unstable_by_key(|&idx| sub_instance.weights[idx]);
    
        let local_search_iterations = 150;
        let mut feasible_adds = Vec::new();
        let mut feasible_swaps = Vec::new();
    
        for _ in 0..local_search_iterations {
            let mut improved = false;
        
            if total_weight < sub_instance.max_weight {
                for (i, &cand) in unselected_items.iter().enumerate() {
                    let new_w = total_weight + sub_instance.weights[cand];
                    let new_val = total_value + contribution_list[cand];
                
                    if new_w > sub_instance.max_weight {
                        break;
                    }
                
                    if new_val >= total_value {
                        feasible_adds.push(i);
                    }
                }
            
                if !feasible_adds.is_empty() {
                    let pick = rng.gen_range(0..feasible_adds.len());
                    let add_idx = feasible_adds[pick];
                    let new_item = unselected_items[add_idx];
                
                    unselected_items.remove(add_idx);
                    selected_items.push(new_item);
                
                    total_weight += sub_instance.weights[new_item];
                    total_value += contribution_list[new_item];
                    improved = true;
                
                    for x in 0..vertex_count {
                        let min_idx = x.min(new_item);
                        let max_idx = x.max(new_item);
                        contribution_list[x] += sub_instance.interaction_values[min_idx][max_idx];
                    }
                }
                feasible_adds.clear();
            }
        
            let free_capacity = sub_instance.max_weight as i32 - total_weight as i32;
        
            for (j, &rem_item) in selected_items.iter().enumerate() {
                let rem_w = sub_instance.weights[rem_item] as i32;
            
                for (i, &cand_item) in unselected_items.iter().enumerate() {
                    let cand_w = sub_instance.weights[cand_item] as i32;
                
                    if rem_w + free_capacity < cand_w {
                        break;
                    }
                
                    let val_diff = contribution_list[cand_item]
                        - contribution_list[rem_item]
                        - sub_instance.interaction_values[cand_item.min(rem_item)][cand_item.max(rem_item)];
                
                    if val_diff >= 0 {
                        feasible_swaps.push((i, j));
                    }
                }
            }
        
            if !feasible_swaps.is_empty() {
                feasible_swaps.sort_by(|&(i1, j1), &(i2, j2)| {
                    let item1 = unselected_items[i1];
                    let rem1 = selected_items[j1];
                    let val_diff1 = contribution_list[item1]
                        - contribution_list[rem1]
                        - sub_instance.interaction_values[item1.min(rem1)][item1.max(rem1)];
                      
                    let item2 = unselected_items[i2];
                    let rem2 = selected_items[j2];
                    let val_diff2 = contribution_list[item2]
                        - contribution_list[rem2]
                        - sub_instance.interaction_values[item2.min(rem2)][item2.max(rem2)];
                      
                    val_diff2.cmp(&val_diff1)
                });
            
                let top_count = (feasible_swaps.len() / 4).max(1);
                let pick = if rng.gen_bool(0.7) {
                    rng.gen_range(0..top_count)
                } else {
                    rng.gen_range(0..feasible_swaps.len())
                };
            
                let (unsel_idx, sel_idx) = feasible_swaps[pick];
                let new_item = unselected_items[unsel_idx];
                let remove_item = selected_items[sel_idx];
            
                selected_items.swap_remove(sel_idx);
                selected_items.push(new_item);
                unselected_items.swap_remove(unsel_idx);
                unselected_items.push(remove_item);
            
                unselected_items.sort_unstable_by_key(|&idx| sub_instance.weights[idx]);
            
                total_value += contribution_list[new_item]
                    - contribution_list[remove_item]
                    - sub_instance.interaction_values[new_item.min(remove_item)][new_item.max(remove_item)];
                
                total_weight = total_weight + sub_instance.weights[new_item] - sub_instance.weights[remove_item];
                improved = true;
            
                for x in 0..vertex_count {
                    let added_min = x.min(new_item);
                    let added_max = x.max(new_item);
                    let removed_min = x.min(remove_item);
                    let removed_max = x.max(remove_item);
                
                    contribution_list[x] += 
                        sub_instance.interaction_values[added_min][added_max] -
                        sub_instance.interaction_values[removed_min][removed_max];
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
            Ok(Some((ItemSelection { items: selected_items }, total_value)))
        }
    }

    fn compute_solution(
        sub_instance: &SubInstance,
        contribution_list: &mut [i32],
        unselected_items: &mut Vec<usize>,
        rng: &mut StdRng,
    ) -> Result<Option<(ItemSelection, i32)>> {
        let mut selected_items = Vec::new();
        let mut total_weight = 0;
        let mut total_value = 0;
    
        let vertex_count = sub_instance.weights.len();
        let inv_weights: Vec<f32> = sub_instance.weights.iter().map(|&w| 1.0 / w as f32).collect();
    
        const RCL_MAX: usize = 10;
    
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
        let max_item_weight = sub_instance.weights.iter().max().unwrap();
    
        let mut item_densities: Vec<(usize, f32)> = unselected_items
            .iter()
            .map(|&idx| {
                let ratio = contribution_list[idx] as f32 * inv_weights[idx];
                (idx, ratio)
            })
            .collect();
    
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
            let selected_rank = match acc_probs[..actual_rcl_size].binary_search_by(|prob| {
                prob.partial_cmp(&random_threshold).unwrap()
            }) {
                Ok(i) | Err(i) => i.min(actual_rcl_size - 1)
            };
        
            item_densities.select_nth_unstable_by(selected_rank, |a, b| {
                b.1.partial_cmp(&a.1).unwrap()
            });
            let selected_item = item_densities[selected_rank].0;
        
            selected_items.push(selected_item);
            total_weight += sub_instance.weights[selected_item];
            total_value += contribution_list[selected_item];
        
            if total_weight + max_item_weight > sub_instance.max_weight {
                item_densities.retain(|(idx, _)| {
                    total_weight + sub_instance.weights[*idx] <= sub_instance.max_weight && *idx != selected_item
                });
            } else {
                item_densities.swap_remove(selected_rank);
            }
        
            for x in 0..vertex_count {
                contribution_list[x] += sub_instance.interaction_values[selected_item.min(x)][selected_item.max(x)];
            }
        
            for density in item_densities.iter_mut() {
                let interaction = sub_instance.interaction_values[selected_item.min(density.0)][selected_item.max(density.0)];
                density.1 += interaction as f32 * inv_weights[density.0];
            }
        }
    
        unselected_items.clear();
        unselected_items.extend(0..vertex_count);
    
        for &selected in &selected_items {
            let idx = unselected_items.iter().position(|&x| x == selected);
            if let Some(pos) = idx {
                unselected_items.swap_remove(pos);
            }
        }
    
        unselected_items.sort_unstable_by_key(|&idx| sub_instance.weights[idx]);
    
        let local_search_iterations = 150;
        let mut feasible_adds = Vec::new();
        let mut feasible_swaps = Vec::new();
    
        for _ in 0..local_search_iterations {
            let mut improved = false;
        
            if total_weight < sub_instance.max_weight {
                for (i, &cand) in unselected_items.iter().enumerate() {
                    let new_w = total_weight + sub_instance.weights[cand];
                    let new_val = total_value + contribution_list[cand];
                
                    if new_w > sub_instance.max_weight {
                        break;
                    }
                
                    if new_val >= total_value {
                        feasible_adds.push(i);
                    }
                }
            
                if !feasible_adds.is_empty() {
                    let pick = rng.gen_range(0..feasible_adds.len());
                    let add_idx = feasible_adds[pick];
                    let new_item = unselected_items[add_idx];
                
                    unselected_items.remove(add_idx);
                    selected_items.push(new_item);
                
                    total_weight += sub_instance.weights[new_item];
                    total_value += contribution_list[new_item];
                    improved = true;
                
                    for x in 0..vertex_count {
                        let min_idx = x.min(new_item);
                        let max_idx = x.max(new_item);
                        contribution_list[x] += sub_instance.interaction_values[min_idx][max_idx];
                    }
                }
                feasible_adds.clear();
            }
        
            let free_capacity = sub_instance.max_weight as i32 - total_weight as i32;
        
            for (j, &rem_item) in selected_items.iter().enumerate() {
                let rem_w = sub_instance.weights[rem_item] as i32;
            
                for (i, &cand_item) in unselected_items.iter().enumerate() {
                    let cand_w = sub_instance.weights[cand_item] as i32;
                
                    if rem_w + free_capacity < cand_w {
                        break;
                    }
                
                    let val_diff = contribution_list[cand_item]
                        - contribution_list[rem_item]
                        - sub_instance.interaction_values[cand_item.min(rem_item)][cand_item.max(rem_item)];
                
                    if val_diff >= 0 {
                        feasible_swaps.push((i, j));
                    }
                }
            }
        
            if !feasible_swaps.is_empty() {
                feasible_swaps.sort_by(|&(i1, j1), &(i2, j2)| {
                    let item1 = unselected_items[i1];
                    let rem1 = selected_items[j1];
                    let val_diff1 = contribution_list[item1]
                        - contribution_list[rem1]
                        - sub_instance.interaction_values[item1.min(rem1)][item1.max(rem1)];
                      
                    let item2 = unselected_items[i2];
                    let rem2 = selected_items[j2];
                    let val_diff2 = contribution_list[item2]
                        - contribution_list[rem2]
                        - sub_instance.interaction_values[item2.min(rem2)][item2.max(rem2)];
                      
                    val_diff2.cmp(&val_diff1)
                });
            
                let top_count = (feasible_swaps.len() / 4).max(1);
                let pick = if rng.gen_bool(0.7) {
                    rng.gen_range(0..top_count)
                } else {
                    rng.gen_range(0..feasible_swaps.len())
                };
            
                let (unsel_idx, sel_idx) = feasible_swaps[pick];
                let new_item = unselected_items[unsel_idx];
                let remove_item = selected_items[sel_idx];
            
                selected_items.swap_remove(sel_idx);
                selected_items.push(new_item);
                unselected_items.swap_remove(unsel_idx);
                unselected_items.push(remove_item);
            
                unselected_items.sort_unstable_by_key(|&idx| sub_instance.weights[idx]);
            
                total_value += contribution_list[new_item]
                    - contribution_list[remove_item]
                    - sub_instance.interaction_values[new_item.min(remove_item)][new_item.max(remove_item)];
                
                total_weight = total_weight + sub_instance.weights[new_item] - sub_instance.weights[remove_item];
                improved = true;
            
                for x in 0..vertex_count {
                    let added_min = x.min(new_item);
                    let added_max = x.max(new_item);
                    let removed_min = x.min(remove_item);
                    let removed_max = x.max(remove_item);
                
                    contribution_list[x] += 
                        sub_instance.interaction_values[added_min][added_max] -
                        sub_instance.interaction_values[removed_min][removed_max];
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
            Ok(Some((ItemSelection { items: selected_items }, total_value)))
        }
    }

    struct ItemSelection {
        items: Vec<usize>,
    }
}

pub fn help() {
    println!("No help information available.");
}
