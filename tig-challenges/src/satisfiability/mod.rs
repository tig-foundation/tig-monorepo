use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use serde::{Deserialize, Serialize};

impl_kv_string_serde! {
    Track {
        n_vars: usize,
        ratio: u32
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
    pub num_variables: usize,
    pub clauses: Vec<Vec<i32>>,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed.clone()).r#gen());
        let num_clauses = (track.n_vars as f64 * track.ratio as f64 / 1000.0).floor() as usize;

        let var_distr = Uniform::new(1, track.n_vars as i32 + 1);
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
            num_variables: track.n_vars.clone(),
            clauses,
        })
    }

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            if solution.variables.len() != self.num_variables {
                return Err(anyhow!(
                    "Invalid number of variables. Expected: {}, Actual: {}",
                    self.num_variables,
                    solution.variables.len()
                ));
            }

            if self.clauses.iter().all(|clause| {
                clause.iter().any(|&literal| {
                    let var_idx = literal.abs() as usize - 1;
                    let var_value = solution.variables[var_idx];
                    (literal > 0 && var_value) || (literal < 0 && !var_value)
                })
            }) {
                Ok(QUALITY_PRECISION)
            } else {
                Ok(0)
            }
        }
    );
}
