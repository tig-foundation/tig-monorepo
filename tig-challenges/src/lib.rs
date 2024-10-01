use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

#[cfg(feature = "cuda")]
use cudarc::driver::*;
#[cfg(feature = "cuda")]
use std::{collections::HashMap, sync::Arc};

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
    fn generate_instance(seed: [u8; 32], difficulty: &U) -> Result<Self>;
    fn generate_instance_from_str(seed: [u8; 32], difficulty: &str) -> Result<Self> {
        Self::generate_instance(seed, &serde_json::from_str(difficulty)?)
    }
    fn generate_instance_from_vec(seed: [u8; 32], difficulty: &Vec<i32>) -> Result<Self> {
        match difficulty.as_slice().try_into() {
            Ok(difficulty) => Self::generate_instance_from_arr(seed, &difficulty),
            Err(_) => Err(anyhow!("Invalid difficulty length")),
        }
    }
    fn generate_instance_from_arr(seed: [u8; 32], difficulty: &[i32; N]) -> Result<Self> {
        Self::generate_instance(seed, &U::from_arr(difficulty))
    }

    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seed: [u8; 32],
        difficulty: &U,
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self>;
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance_from_str(
        seed: [u8; 32],
        difficulty: &str,
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        Self::cuda_generate_instance(seed, &serde_json::from_str(difficulty)?, dev, funcs)
    }
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance_from_vec(
        seed: [u8; 32],
        difficulty: &Vec<i32>,
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        match difficulty.as_slice().try_into() {
            Ok(difficulty) => Self::cuda_generate_instance_from_arr(seed, &difficulty, dev, funcs),
            Err(_) => Err(anyhow!("Invalid difficulty length")),
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance_from_arr(
        seed: [u8; 32],
        difficulty: &[i32; N],
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        Self::cuda_generate_instance(seed, &U::from_arr(difficulty), dev, funcs)
    }

    fn verify_solution(&self, solution: &T) -> Result<()>;
    fn verify_solution_from_json(&self, solution: &str) -> Result<()> {
        let solution = serde_json::from_str(solution)
            .map_err(|e| anyhow!("Failed to parse solution: {}", e))?;
        self.verify_solution(&solution)
    }
}

pub mod knapsack;
pub use knapsack as c003;
pub mod satisfiability;
pub use satisfiability as c001;
pub mod vector_search;
pub use vector_search as c004;
pub mod vehicle_routing;
pub use vehicle_routing as c002;

// #[cfg(feature = "cuda")]
pub struct CudaKernel {
    pub src: &'static str,
    pub funcs: &'static [&'static str],
}
