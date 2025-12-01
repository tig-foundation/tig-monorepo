use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

#[inline]
fn calculate_density_variance(item_densities: &[(usize, f32)]) -> f32 {
    if item_densities.len() < 2 {
        return 0.5;
    }

    let mut sum: f32 = 0.0;
    let mut resonance_sum: f32 = 0.0;
    let len = item_densities.len() as f32;

    for (i, (_, density_i)) in item_densities.iter().enumerate() {
        sum += *density_i;

        for (_, density_j) in item_densities.iter().skip(i + 1).take(3) {
            let ratio = (density_i / density_j).abs();
            let resonance = if (ratio - 1.0).abs() < 0.1 {
                1.5
            } else if (ratio - 1.0).abs() < 0.2 {
                1.2
            } else if (ratio - 1.0).abs() < 0.3 {
                1.1
            } else {
                1.0
            };
            resonance_sum += resonance;
        }
    }
    let mean = sum / len;

    let mut variance_sum: f32 = 0.0;
    for (_, density) in item_densities {
        let diff = *density - mean;
        variance_sum += diff * diff;
    }
    let variance = variance_sum / len;

    let resonance_factor = (1.0 + resonance_sum / (len * 3.0)).clamp(1.0, 1.5);
    ((variance.sqrt() / mean.abs()) * resonance_factor).clamp(0.1, 1.0)
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

    let inv_weights: Vec<f32> = challenge.weights.iter().map(|&w| 1.0 / w as f32).collect();
    let rcl_max = 10;
    let max_item_weight = *challenge.weights.iter().max().unwrap();

    let mut item_densities: Vec<(usize, f32)> = Vec::with_capacity(unselected_items.len());
    for &idx in unselected_items.iter() {
        let ratio = contribution_list[idx] as f32 * inv_weights[idx];
        item_densities.push((idx, ratio));
    }

    let density_variance = calculate_density_variance(&item_densities);

    let num_items = challenge.num_items as f32;

    let mut densities: Vec<f32> = item_densities.iter().map(|(_, d)| *d).collect();
    densities.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let adaptive_range = if num_items >= 500.0 {
        if densities.len() >= 10 {
            let p10_idx = densities.len() / 10;
            let p90_idx = densities.len() * 9 / 10;
            let p10 = densities[p10_idx];
            let p90 = densities[p90_idx];
            let density_range = (p90 - p10).abs();
            (density_range * 0.3).clamp(0.8, 2.2)
        } else {
            1.5
        }
    } else {
        1.5
    };

    let base_exponent = if num_items >= 500.0 {
        1.4 - ((num_items - 500.0) / 200.0).sqrt() * 0.2
    } else {
        1.4
    };
    let adaptive_exponent =
        base_exponent + (density_variance - 0.6).clamp(-adaptive_range, adaptive_range);

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
    unselected_items.extend(0..challenge.num_items);

    selected_items.sort_unstable_by(|a, b| b.cmp(a));
    for &selected in &selected_items {
        unselected_items.swap_remove(selected);
    }

    unselected_items.sort_unstable_by_key(|&idx| challenge.weights[idx]);

    let mut feasible_adds = Vec::with_capacity(100);
    let mut feasible_swaps = Vec::with_capacity(200);

    let base_iterations = if challenge.num_items > 1000 {
        35
    } else if challenge.num_items > 500 {
        40
    } else if challenge.num_items > 200 {
        60
    } else {
        80
    };

    let max_local_iterations = if total_value > 0 {
        (base_iterations as f32 * (1.0 + total_value as f32 / 10000.0).min(1.5)) as i32
    } else {
        base_iterations
    };

    for iter in 0..max_local_iterations {
        let mut improved = false;
        let intensive_search = iter >= max_local_iterations - 5;

        if total_weight < challenge.max_weight {
            feasible_adds.clear();
            for (i, &cand) in unselected_items.iter().enumerate() {
                if total_weight + challenge.weights[cand] > challenge.max_weight {
                    break;
                }
                let new_val = total_value + contribution_list[cand];
                if new_val > total_value {
                    feasible_adds.push(i);
                }
            }

            if !feasible_adds.is_empty() {
                let temperature = 1.0 - (iter as f32 / max_local_iterations as f32);
                let add_idx = if rng.gen::<f32>() > temperature {
                    feasible_adds[0]
                } else {
                    feasible_adds[rng.gen_range(0..feasible_adds.len())]
                };
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
        }

        if !improved {
            feasible_swaps.clear();
            let free_capacity = challenge.max_weight - total_weight;

            'outer: for (j, &rem_item) in selected_items.iter().enumerate() {
                let rem_w = challenge.weights[rem_item];
                let available_weight = free_capacity + rem_w;

                for (i, &cand_item) in unselected_items.iter().enumerate() {
                    if challenge.weights[cand_item] > available_weight {
                        break;
                    }

                    let val_diff = contribution_list[cand_item]
                        - contribution_list[rem_item]
                        - challenge.interaction_values[cand_item][rem_item];
                    if val_diff > 0 {
                        feasible_swaps.push((i, j));

                        if val_diff > total_value / 100 {
                            break 'outer;
                        }
                        if feasible_swaps.len() >= 50 {
                            break 'outer;
                        }
                    }
                }
            }

            if !feasible_swaps.is_empty() {
                let (unsel_idx, sel_idx) = feasible_swaps[rng.gen_range(0..feasible_swaps.len())];
                let new_item = unselected_items[unsel_idx];
                let remove_item = selected_items[sel_idx];

                selected_items.swap_remove(sel_idx);
                selected_items.push(new_item);
                unselected_items.remove(unsel_idx);

                let insert_pos = unselected_items
                    .binary_search_by_key(&challenge.weights[remove_item], |&idx| {
                        challenge.weights[idx]
                    })
                    .unwrap_or_else(|e| e);
                unselected_items.insert(insert_pos, remove_item);

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
        }

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
    let base_iterations = 11;
    let num_iterations = if challenge.num_items > 1000 {
        (base_iterations as f32 * 0.7) as i32
    } else if challenge.num_items > 500 {
        (base_iterations as f32 * 0.85) as i32
    } else {
        base_iterations
    };
    let mut rng =
        StdRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut best_solution: Option<Solution> = None;
    let mut best_value = 0;

    let mut unselected_items: Vec<usize> = Vec::with_capacity(challenge.num_items);
    let mut contribution_list: Vec<i32> = Vec::with_capacity(challenge.values.len());

    for _outer_iter in 0..num_iterations {
        let mut best_local_solution: Option<Solution> = None;
        let mut best_local_value = 0;

        let k = 2;
        for _ in 0..k {
            unselected_items.clear();
            unselected_items.reserve(challenge.num_items);
            for i in 0..challenge.num_items {
                unselected_items.push(i);
            }

            contribution_list.clear();
            contribution_list.reserve(challenge.values.len());
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

    if let Some(s) = best_solution {
        let _ = save_solution(&s);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod gpu_optimisation {
    use super::*;
    use cudarc::driver::*;
    use std::{collections::HashMap, sync::Arc};
    use tig_challenges::CudaKernel;

    pub const KERNEL: Option<CudaKernel> = None;

    pub fn cuda_solve_challenge(
        challenge: &Challenge,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> anyhow::Result<Option<Solution>> {
        solve_challenge(challenge)
    }
}
#[cfg(feature = "cuda")]
pub use gpu_optimisation::{cuda_solve_challenge, KERNEL};

pub fn help() {
    println!("No help information available.");
}
