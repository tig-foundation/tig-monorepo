use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
pub trait DifficultyTrait<const N: usize>: Serialize + DeserializeOwned {
    fn from_arr(arr: &[i32; N]) -> Self;
    fn to_arr(&self) -> [i32; N];
}
pub trait SolutionTrait: Serialize + DeserializeOwned {}

pub trait ChallengeTrait<T, U, const N: usize>: Serialize + DeserializeOwned
where
    T: SolutionTrait,
    U: DifficultyTrait<N>,
{
    fn generate_instance(seed: u32, difficulty: &U) -> Result<Self>;
    fn generate_instance_from_str(seed: u32, difficulty: &str) -> Result<Self> {
        Self::generate_instance(seed, &serde_json::from_str(difficulty)?)
    }
    fn generate_instance_from_vec(seed: u32, difficulty: &Vec<i32>) -> Result<Self> {
        match difficulty.as_slice().try_into() {
            Ok(difficulty) => Self::generate_instance_from_arr(seed, &difficulty),
            Err(_) => Err(anyhow!("Invalid difficulty length")),
        }
    }
    fn generate_instance_from_arr(seed: u32, difficulty: &[i32; N]) -> Result<Self> {
        Self::generate_instance(seed, &U::from_arr(difficulty))
    }

    fn verify_solution(&self, solution: &T) -> Result<()>;
    fn verify_solution_from_json(&self, solution: &str) -> Result<Result<()>> {
        Ok(self.verify_solution(&serde_json::from_str(solution)?))
    }
}

pub mod knapsack;
pub mod satisfiability;
pub mod vehicle_routing;
