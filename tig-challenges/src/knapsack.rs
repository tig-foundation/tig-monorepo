use anyhow::{anyhow, Result};
use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::collections::HashSet;

#[cfg(feature = "cuda")]
use crate::CudaKernel;
#[cfg(feature = "cuda")]
use cudarc::driver::*;
#[cfg(feature = "cuda")]
use std::{collections::HashMap, sync::Arc};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub num_items: usize,
    pub better_than_baseline: u32,
}

impl crate::DifficultyTrait<2> for Difficulty {
    fn from_arr(arr: &[i32; 2]) -> Self {
        Self {
            num_items: arr[0] as usize,
            better_than_baseline: arr[1] as u32,
        }
    }

    fn to_arr(&self) -> [i32; 2] {
        [self.num_items as i32, self.better_than_baseline as i32]
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Solution {
    pub sub_solutions: Vec<SubSolution>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubSolution {
    pub items: Vec<usize>,
}

impl crate::SolutionTrait for Solution {}

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;

    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub sub_instances: Vec<SubInstance>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubInstance {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub weights: Vec<u32>,
    pub values: Vec<u32>,
    pub interaction_values: Vec<Vec<i32>>,
    pub max_weight: u32,
    pub baseline_value: u32,
}

// TIG dev bounty available for a GPU optimisation for instance generation!
#[cfg(feature = "cuda")]
pub const KERNEL: Option<CudaKernel> = None;

pub const NUM_SUB_INSTANCES: usize = 16;

impl crate::ChallengeTrait<Solution, Difficulty, 2> for Challenge {
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seed: [u8; 32],
        difficulty: &Difficulty,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        // TIG dev bounty available for a GPU optimisation for instance generation!
        Self::generate_instance(seed, difficulty)
    }

    fn generate_instance(seed: [u8; 32], difficulty: &Difficulty) -> Result<Challenge> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed).gen());
        let mut sub_instances = Vec::new();
        for _ in 0..NUM_SUB_INSTANCES {
            sub_instances.push(SubInstance::generate_instance(&mut rng, seed, difficulty)?);
        }

        Ok(Challenge {
            seed,
            difficulty: difficulty.clone(),
            sub_instances,
        })
    }

    fn verify_solution(&self, solution: &Solution) -> Result<()> {
        let mut better_than_baselines = Vec::new();
        for (i, (sub_instance, sub_solution)) in self
            .sub_instances
            .iter()
            .zip(&solution.sub_solutions)
            .enumerate()
        {
            match sub_instance.verify_solution(&sub_solution) {
                Ok(total_value) => better_than_baselines
                    .push(total_value as f64 / sub_instance.baseline_value as f64 - 1.0),
                Err(e) => return Err(anyhow!("Instance {}: {}", i, e.to_string())),
            }
        }
        let average =
            better_than_baselines.iter().sum::<f64>() / better_than_baselines.len() as f64;
        let threshold = self.difficulty.better_than_baseline as f64 / 1000.0;
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
    fn generate_instance(
        rng: &mut SmallRng,
        seed: [u8; 32],
        difficulty: &Difficulty,
    ) -> Result<SubInstance> {
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
        let mut total_weight = 0;

        for &(item, _) in &item_values {
            if total_weight + weights[item] <= max_weight {
                total_weight += weights[item];
                selected_items.push(item);
            }
        }

        let baseline_value = calculate_total_value(&selected_items, &values, &interaction_values);

        Ok(SubInstance {
            seed,
            difficulty: difficulty.clone(),
            weights,
            values,
            interaction_values,
            max_weight,
            baseline_value,
        })
    }

    fn verify_solution(&self, solution: &SubSolution) -> Result<u32> {
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
        if total_value < self.baseline_value {
            Err(anyhow!(
                "Total value ({}) does not reach minimum value ({})",
                total_value,
                self.baseline_value
            ))
        } else {
            Ok(total_value)
        }
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
