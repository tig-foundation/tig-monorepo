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
 
     fn compute_solution(
        challenge: &SubInstance,
        contribution_list: &mut [i32],
        unselected_items: &mut Vec<usize>,
        rng: &mut StdRng,
    ) -> Result<Option<(SubSolution, i32)>> {
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
            Ok(Some((SubSolution { items: selected_items }, total_value)))
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
        let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(
            challenge.seed[..8].try_into().unwrap(),
        ));
 
        let mut best_solution: Option<SubSolution> = None;
        let mut best_value = 0;
 
        for _outer_iter in 0..100 {
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
                best_solution = Some(SubSolution { items: solution.items.clone() });
            }
 
            let threshold = lookup_threshold(challenge.num_items);
            if (challenge.baseline_value as f32) * (1.0 - threshold * 0.01) >= best_value as f32 {
                return Ok(None);
            }
            else if challenge.baseline_value <= best_value as u32 {
                return Ok(best_solution);
            }
        }
 
        Ok(best_solution)
     }
 
     fn lookup_threshold(num_items: usize) -> f32 {
        let points = vec![
            (100, 1.071), (105, 1.015), (110, 0.973), (120, 0.882),
            (125, 0.791), (130, 0.770), (135, 0.760), (140, 0.749),
            (145, 0.700), (150, 0.616), (155, 0.574), (160, 0.532),
            (165, 0.511), (170, 0.494), (175, 0.485), (180, 0.476),
            (190, 0.448), (195, 0.434), (200, 0.427), (205, 0.420),
            (210, 0.420), (215, 0.385), (220, 0.350), (225, 0.347),
            (230, 0.343), (235, 0.343), (240, 0.338), (245, 0.334),
            (250, 0.329)
        ];
 
        points.iter()
            .filter(|&&(x, _)| x <= num_items)
            .max_by_key(|&&(x, _)| x)
            .unwrap()
            .1
     }
}

pub fn help() {
    println!("No help information available.");
}
