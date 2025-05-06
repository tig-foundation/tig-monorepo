use anyhow::{anyhow, Result};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::collections::HashSet;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub num_items: usize,
    pub better_than_baseline: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            num_items: arr[0] as usize,
            better_than_baseline: arr[1] as u32,
        }
    }
}

impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.num_items as i32, self.better_than_baseline as i32]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Solution {
    pub sub_solutions: Vec<SubSolution>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubSolution {
    pub items: Vec<usize>,
}

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;

    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub sub_instances: Vec<SubInstance>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubInstance {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub weights: Vec<u32>,
    pub values: Vec<u32>,
    pub interaction_values: Vec<Vec<i32>>,
    pub max_weight: u32,
    pub baseline_value: u32,
}

pub const NUM_SUB_INSTANCES: usize = 16;

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Challenge> {
        let mut rng = StdRng::from_seed(seed.clone());
        let mut sub_instances = Vec::new();
        for _ in 0..NUM_SUB_INSTANCES {
            sub_instances.push(SubInstance::generate_instance(&rng.gen(), difficulty)?);
        }

        Ok(Challenge {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            sub_instances,
        })
    }

    pub fn verify_solution(&self, solution: &Solution) -> Result<()> {
        let mut better_than_baselines = Vec::new();
        for (i, (sub_instance, sub_solution)) in self
            .sub_instances
            .iter()
            .zip(&solution.sub_solutions)
            .enumerate()
        {
            match sub_instance.verify_solution(&sub_solution) {
                Ok(total_value) => better_than_baselines
                    .push(total_value as f64 / sub_instance.baseline_value as f64),
                Err(e) => return Err(anyhow!("Instance {}: {}", i, e.to_string())),
            }
        }
        let average = (better_than_baselines.iter().map(|x| x * x).sum::<f64>()
            / better_than_baselines.len() as f64)
            .sqrt()
            - 1.0;
        let threshold = self.difficulty.better_than_baseline as f64 / 10000.0;
        if average >= threshold {
            Ok(())
        } else {
            Err(anyhow!(
                "Average better_than_baseline ({}) is less than ({})",
                average,
                threshold
            ))
        }
    }
}

impl SubInstance {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<SubInstance> {
        let mut rng = SmallRng::from_seed(seed.clone());
        // Set constant density for value generation
        let density = 0.25;

        // Generate weights w_i in the range [1, 50]
        let weights: Vec<u32> = (0..difficulty.num_items)
            .map(|_| rng.gen_range(1..=50))
            .collect();

        // Generate values v_i in the range [1, 100] with density probability, 0 otherwise
        let values: Vec<u32> = (0..difficulty.num_items)
            .map(|_| {
                if rng.gen_bool(density) {
                    rng.gen_range(1..=100)
                } else {
                    0
                }
            })
            .collect();

        // Generate interaction values V_ij with the following properties:
        // - V_ij == V_ji (symmetric matrix)
        // - V_ii == 0 (diagonal is zero)
        // - Values are in range [1, 100] with density probability, 0 otherwise
        let mut interaction_values: Vec<Vec<i32>> =
            vec![vec![0; difficulty.num_items]; difficulty.num_items];

        for i in 0..difficulty.num_items {
            for j in (i + 1)..difficulty.num_items {
                let value = if rng.gen_bool(density) {
                    rng.gen_range(1..=100)
                } else {
                    0
                };

                // Set both V_ij and V_ji due to symmetry
                interaction_values[i][j] = value;
                interaction_values[j][i] = value;
            }
        }

        let max_weight: u32 = weights.iter().sum::<u32>() / 2;

        // Precompute the ratio between the total value (value + sum of interactive values) and
        // weight for each item. Pair the ratio with the item's weight and index
        let mut item_values: Vec<(usize, f32)> = (0..difficulty.num_items)
            .map(|i| {
                let total_value = values[i] as i32 + interaction_values[i].iter().sum::<i32>();
                let ratio = total_value as f32 / weights[i] as f32;
                (i, ratio)
            })
            .collect();

        // Sort the list of ratios in descending order
        item_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Step 1: Initial solution obtained by greedily selecting items based on value-weight ratio
        let mut selected_items = Vec::with_capacity(difficulty.num_items);
        let mut unselected_items = Vec::with_capacity(difficulty.num_items);
        let mut total_weight = 0;
        let mut is_selected = vec![false; difficulty.num_items];

        for &(item, _) in &item_values {
            if total_weight + weights[item] <= max_weight {
                total_weight += weights[item];
                selected_items.push(item);
                is_selected[item] = true;
            } else {
                unselected_items.push(item);
            }
        }

        // Step 2: Improvement of solution with Local Search and Tabu-List
        // Precompute sum of interaction values with each selected item for all items
        let mut interaction_sum_list = vec![0; difficulty.num_items];
        for x in 0..difficulty.num_items {
            interaction_sum_list[x] = values[x] as i32;
            for &item in &selected_items {
                interaction_sum_list[x] += interaction_values[x][item];
            }
        }

        let mut min_selected_item_values = i32::MAX;
        for x in 0..difficulty.num_items {
            if is_selected[x] {
                min_selected_item_values = min_selected_item_values.min(interaction_sum_list[x]);
            }
        }

        // Optimized local search with tabu list
        let max_iterations = 100;
        let mut tabu_list = vec![0; difficulty.num_items];

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
                let min_weight =
                    weights[new_item] as i32 - (max_weight as i32 - total_weight as i32);
                for j in 0..selected_items.len() {
                    let remove_item = selected_items[j];
                    if tabu_list[remove_item] > 0 {
                        continue;
                    }

                    // Don't check the weight if there is enough remaining capacity
                    if min_weight > 0 {
                        // Skip a remove_item if the remaining capacity after removal is insufficient to push a new_item
                        let removed_item_weight = weights[remove_item] as i32;
                        if removed_item_weight < min_weight {
                            continue;
                        }
                    }

                    let remove_item_values_sum = interaction_sum_list[remove_item];
                    let value_diff = new_item_values_sum
                        - remove_item_values_sum
                        - interaction_values[new_item][remove_item];

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

                total_weight = total_weight + weights[new_item] - weights[remove_item];

                // Update sum of interaction values after swapping items
                min_selected_item_values = i32::MAX;
                for x in 0..difficulty.num_items {
                    interaction_sum_list[x] +=
                        interaction_values[x][new_item] - interaction_values[x][remove_item];
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

        let baseline_value = calculate_total_value(&selected_items, &values, &interaction_values);

        Ok(SubInstance {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            weights,
            values,
            interaction_values,
            max_weight,
            baseline_value,
        })
    }

    pub fn verify_solution(&self, solution: &SubSolution) -> Result<u32> {
        let selected_items: HashSet<usize> = solution.items.iter().cloned().collect();
        if selected_items.len() != solution.items.len() {
            return Err(anyhow!("Duplicate items selected."));
        }

        let total_weight = selected_items
            .iter()
            .map(|&item| {
                if item >= self.weights.len() {
                    return Err(anyhow!("Item ({}) is out of bounds", item));
                }
                Ok(self.weights[item])
            })
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum::<u32>();

        if total_weight > self.max_weight {
            return Err(anyhow!(
                "Total weight ({}) exceeded max weight ({})",
                total_weight,
                self.max_weight
            ));
        }
        let selected_items_vec: Vec<usize> = selected_items.into_iter().collect();
        let total_value =
            calculate_total_value(&selected_items_vec, &self.values, &self.interaction_values);
        Ok(total_value)
    }
}

pub fn calculate_total_value(
    indices: &Vec<usize>,
    values: &Vec<u32>,
    interaction_values: &Vec<Vec<i32>>,
) -> u32 {
    let mut indices = indices.clone();
    indices.sort();

    let mut total_value = 0i32;

    // Sum the individual values
    for &i in &indices {
        total_value += values[i] as i32;
    }

    // Sum the interactive values for pairs in indices
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let idx_i = indices[i];
            let idx_j = indices[j];
            total_value += interaction_values[idx_i][idx_j];
        }
    }

    match total_value {
        v if v < 0 => 0u32,
        v => v as u32,
    }
}
