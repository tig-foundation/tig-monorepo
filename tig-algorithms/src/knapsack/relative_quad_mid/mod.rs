use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;
 
 fn compute_solution(
    challenge: &Challenge,
    contribution_list: &mut [i32],
    unselected_items: &mut Vec<usize>,
    rng: &mut StdRng,
) -> Result<Option<(Solution, i32)>> {
    let mut selected_items = Vec::new();
    let mut total_weight = 0;
    let mut total_value = 0;

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
    let max_item_weight = challenge.weights.iter().max().unwrap();

    let mut item_densities: Vec<(usize, f32)> = unselected_items
        .iter()
        .map(|&idx| {
            let ratio = contribution_list[idx] as f32 / challenge.weights[idx] as f32;
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
        let mut selected_rank = match acc_probs[..actual_rcl_size].binary_search_by(|prob| {
            prob.partial_cmp(&random_threshold).unwrap()
        }) { Ok(i) | Err(i) => i };
        if selected_rank >= actual_rcl_size {
            selected_rank = actual_rcl_size - 1;
        }

        item_densities.select_nth_unstable_by(selected_rank, |a, b| {
            b.1.partial_cmp(&a.1).unwrap()
        });
        let selected_item = item_densities[selected_rank].0;

        selected_items.push(selected_item);
        total_weight += challenge.weights[selected_item];
        total_value += contribution_list[selected_item];

        unsafe {
            for x in 0..challenge.num_items {
                *contribution_list.get_unchecked_mut(x) += 
                    *challenge.interaction_values.get_unchecked(x).get_unchecked(selected_item);
            }
        }

        if total_weight + max_item_weight > challenge.max_weight {
            item_densities.retain(|(idx, _)| {
                total_weight + challenge.weights[*idx] <= challenge.max_weight && *idx != selected_item
            });
        } else {
            item_densities.swap_remove(selected_rank);
        }

        unsafe {
            for density in item_densities.iter_mut() {
                let interaction = *challenge.interaction_values.get_unchecked(selected_item).get_unchecked(density.0);
                let w = *challenge.weights.get_unchecked(density.0) as f32;
                density.1 += interaction as f32 / w;
            }
        }
    }
    unselected_items.clear();
    unselected_items.extend(0..challenge.num_items);

    let mut sorted_selected = selected_items.clone();
    sorted_selected.sort_unstable_by(|a, b| b.cmp(a));

    for &selected in &sorted_selected {
        unselected_items.swap_remove(selected);
    }
    
    let local_search_iterations = 150;
    for _ in 0..local_search_iterations {
        let mut improved = false;
 
        let mut feasible_adds = Vec::new();
        for (i, &cand) in unselected_items.iter().enumerate() {
            let new_w = total_weight + challenge.weights[cand];
            let new_val = total_value + contribution_list[cand];
            if new_w <= challenge.max_weight && new_val >= total_value {
                feasible_adds.push(i);
            }
        }
        if !feasible_adds.is_empty() {
            let pick = rng.gen_range(0..feasible_adds.len());
            let add_idx = feasible_adds[pick];
            let new_item = unselected_items[add_idx];
 
            unselected_items.swap_remove(add_idx);
            selected_items.push(new_item);
 
            total_weight += challenge.weights[new_item];
            total_value += contribution_list[new_item];
            improved = true;
 
            unsafe {
                for x in 0..challenge.num_items {
                    *contribution_list.get_unchecked_mut(x) += 
                        *challenge.interaction_values.get_unchecked(x).get_unchecked(new_item);
                }
            }
        }

        let mut feasible_swaps = Vec::new();
        for (i, &cand_item) in unselected_items.iter().enumerate() {
            let min_needed =
                challenge.weights[cand_item] as i32 - (challenge.max_weight as i32 - total_weight as i32);
            for (j, &rem_item) in selected_items.iter().enumerate() {
                let rem_w = challenge.weights[rem_item] as i32;
                if rem_w < min_needed {
                    continue;
                }
                let val_diff = contribution_list[cand_item]
                    - contribution_list[rem_item]
                    - challenge.interaction_values[cand_item][rem_item];
                if val_diff >= 0 {
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
            unselected_items.swap_remove(unsel_idx);
            selected_items.push(new_item);
            unselected_items.push(remove_item);
 
            total_value += contribution_list[new_item]
                - contribution_list[remove_item]
                - challenge.interaction_values[new_item][remove_item];
            total_weight = total_weight + challenge.weights[new_item] - challenge.weights[remove_item];
            improved = true;
 
            unsafe {
                for x in 0..challenge.num_items {
                    *contribution_list.get_unchecked_mut(x) += 
                        *challenge.interaction_values.get_unchecked(x).get_unchecked(new_item) -
                        *challenge.interaction_values.get_unchecked(x).get_unchecked(remove_item);
                }
            }
        }
 
        if !improved {
            break;
        }
    }
 
    if selected_items.is_empty() {
        Ok(None)
    } else {
        Ok(Some((Solution { items: selected_items }, total_value)))
    }
 }

 
 

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(
        challenge.seed[..8].try_into().unwrap(),
    ));
 
    let mut best_solution: Option<Solution> = None;
    let mut best_value = 0;

    for _outer_iter in 0..5 {
        let mut unselected_items: Vec<usize> = (0..challenge.num_items).collect();
        let mut contribution_list = challenge
            .values
            .iter()
            .map(|&v| v as i32)
            .collect::<Vec<i32>>();
 
        let sol_result =
            compute_solution(challenge, &mut contribution_list, &mut unselected_items, &mut rng)?;
 
        let (solution, value) = match sol_result {
            Some(x) => x,
            None => continue,
        };
 
        if value > best_value {
            best_value = value;
            best_solution = Some(Solution { items: solution.items.clone() });
        }
    }
 
    if let Some(s) = best_solution {
       let _ = save_solution(&s);
    }
    return Ok(());
 }

pub fn help() {
    println!("No help information available.");
}
