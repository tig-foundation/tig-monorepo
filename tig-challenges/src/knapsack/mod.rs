use crate::QUALITY_PRECISION;
mod baselines;
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashSet;

impl_base64_serde! {
    Solution {
        items: Vec<usize>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_items: usize,
    pub weights: Vec<u32>,
    pub values: Vec<u32>,
    pub interaction_values: Vec<Vec<i32>>,
    pub max_weight: u32,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], num_items: usize) -> Result<Self> {
        let mut rng = SmallRng::from_seed(seed.clone());
        // Set constant density for value generation
        let density = 0.25;

        // Generate weights w_i in the range [1, 50]
        let weights: Vec<u32> = (0..num_items).map(|_| rng.gen_range(1..=50)).collect();

        // Generate values v_i in the range [1, 100] with density probability, 0 otherwise
        let values: Vec<u32> = (0..num_items)
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
        let mut interaction_values: Vec<Vec<i32>> = vec![vec![0; num_items]; num_items];

        for i in 0..num_items {
            for j in (i + 1)..num_items {
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

        Ok(Challenge {
            seed: seed.clone(),
            num_items,
            weights,
            values,
            interaction_values,
            max_weight,
        })
    }

    pub fn evaluate_total_value(&self, solution: &Solution) -> Result<u32> {
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
        let mut indices: Vec<usize> = selected_items.into_iter().collect();
        indices.sort();

        let mut total_value = 0i32;

        // Sum the individual values
        for &i in &indices {
            total_value += self.values[i] as i32;
        }

        // Sum the interactive values for pairs in indices
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let idx_i = indices[i];
                let idx_j = indices[j];
                total_value += self.interaction_values[idx_i][idx_j];
            }
        }

        Ok(match total_value {
            v if v < 0 => 0u32,
            v => v as u32,
        })
    }

    conditional_pub!(
        fn compute_greedy_baseline(&self) -> Result<Solution> {
            let solution = RefCell::new(Solution::new());
            let save_solution_fn = |s: &Solution| -> Result<()> {
                *solution.borrow_mut() = s.clone();
                Ok(())
            };
            baselines::tabu_search::solve_challenge(self, &save_solution_fn, &None)?;
            Ok(solution.into_inner())
        }
    );

    conditional_pub!(
        fn compute_sota_baseline(&self) -> Result<Solution> {
            Err(anyhow!("Not implemented yet"))
        }
    );

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let total_value = self.evaluate_total_value(solution)?;
            let greedy_solution = self.compute_greedy_baseline()?;
            let greedy_total_value = self.evaluate_total_value(&greedy_solution)?;
            // TODO: implement SOTA baseline
            let sota_total_value = greedy_total_value;
            // if total_value < greedy_total_value {
            //     return Err(anyhow!(
            //         "Total value {} is less than greedy baseline value {}",
            //         total_value,
            //         greedy_total_value
            //     ));
            // }
            // let sota_solution = self.compute_sota_baseline()?;
            // let sota_total_value = self.evaluate_total_value(&sota_solution)?;
            let quality = (total_value as f64 - sota_total_value as f64) / sota_total_value as f64;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}
