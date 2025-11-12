use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub num_variables: usize,
    pub clauses_to_variables_percent: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            num_variables: arr[0] as usize,
            clauses_to_variables_percent: arr[1] as u32,
        }
    }
}

impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![
            self.num_variables as i32,
            self.clauses_to_variables_percent as i32,
        ]
    }
}

impl_base64_serde! {
    Solution {
        variables: Vec<bool>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub clauses: Vec<Vec<i32>>,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed.clone()).gen());
        let num_clauses = (difficulty.num_variables as f64
            * difficulty.clauses_to_variables_percent as f64
            / 100.0)
            .floor() as usize;

        let var_distr = Uniform::new(1, difficulty.num_variables as i32 + 1);
        // Create a uniform distribution for negations.
        let neg_distr = Uniform::new(0, 2);

        // Generate the clauses array.
        let clauses_array = Array2::from_shape_fn((num_clauses, 3), |_| var_distr.sample(&mut rng));

        // Generate the negations array.
        let negations = Array2::from_shape_fn((num_clauses, 3), |_| {
            if neg_distr.sample(&mut rng) == 0 {
                -1
            } else {
                1
            }
        });

        // Combine clauses array with negations.
        let clauses_array = clauses_array * negations;

        // Convert Array2<i32> to Vec<Vec<i32>>
        let clauses = clauses_array
            .axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect();

        Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            clauses,
        })
    }

    conditional_pub!(
        fn verify_solution(&self, solution: &Solution) -> Result<()> {
            if solution.variables.len() != self.difficulty.num_variables {
                return Err(anyhow!(
                    "Invalid number of variables. Expected: {}, Actual: {}",
                    self.difficulty.num_variables,
                    solution.variables.len()
                ));
            }

            if let Some((idx, _)) = self.clauses.iter().enumerate().find(|(_, clause)| {
                !clause.iter().any(|&literal| {
                    let var_idx = literal.abs() as usize - 1;
                    let var_value = solution.variables[var_idx];
                    (literal > 0 && var_value) || (literal < 0 && !var_value)
                })
            }) {
                Err(anyhow!("Clause '{}' not satisfied", idx))
            } else {
                Ok(())
            }
        }
    );
}
