use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub num_variables: usize,
    pub clauses_to_variables_percent: u32,
}

impl crate::DifficultyTrait<2> for Difficulty {
    fn from_arr(arr: &[i32; 2]) -> Self {
        Self {
            num_variables: arr[0] as usize,
            clauses_to_variables_percent: arr[1] as u32,
        }
    }

    fn to_arr(&self) -> [i32; 2] {
        [
            self.num_variables as i32,
            self.clauses_to_variables_percent as i32,
        ]
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Solution {
    pub variables: Vec<bool>,
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
    pub seed: u32,
    pub difficulty: Difficulty,
    pub clauses: Vec<Vec<i32>>,
}

impl crate::ChallengeTrait<Solution, Difficulty, 2> for Challenge {
    fn generate_instance(seed: u32, difficulty: &Difficulty) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
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
            seed,
            difficulty: difficulty.clone(),
            clauses,
        })
    }

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
}
