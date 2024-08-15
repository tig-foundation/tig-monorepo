use anyhow::{anyhow, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
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
    fn generate_instance(seeds: [u64; 8], difficulty: &U) -> Result<Self>;
    fn generate_instance_from_str(seeds: [u64; 8], difficulty: &str) -> Result<Self> {
        Self::generate_instance(seeds, &serde_json::from_str(difficulty)?)
    }
    fn generate_instance_from_vec(seeds: [u64; 8], difficulty: &Vec<i32>) -> Result<Self> {
        match difficulty.as_slice().try_into() {
            Ok(difficulty) => Self::generate_instance_from_arr(seeds, &difficulty),
            Err(_) => Err(anyhow!("Invalid difficulty length")),
        }
    }
    fn generate_instance_from_arr(seeds: [u64; 8], difficulty: &[i32; N]) -> Result<Self> {
        Self::generate_instance(seeds, &U::from_arr(difficulty))
    }

    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seeds: [u64; 8],
        difficulty: &U,
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self>;
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance_from_str(
        seeds: [u64; 8],
        difficulty: &str,
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        Self::cuda_generate_instance(seeds, &serde_json::from_str(difficulty)?, dev, funcs)
    }
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance_from_vec(
        seeds: [u64; 8],
        difficulty: &Vec<i32>,
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        match difficulty.as_slice().try_into() {
            Ok(difficulty) => Self::cuda_generate_instance_from_arr(seeds, &difficulty, dev, funcs),
            Err(_) => Err(anyhow!("Invalid difficulty length")),
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance_from_arr(
        seeds: [u64; 8],
        difficulty: &[i32; N],
        dev: &Arc<CudaDevice>,
        funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> {
        Self::cuda_generate_instance(seeds, &U::from_arr(difficulty), dev, funcs)
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

pub struct RngArray {
    rngs: [StdRng; 8],
    index: u32,
}

impl RngArray {
    pub fn new(seeds: [u64; 8]) -> Self {
        let rngs = seeds.map(StdRng::seed_from_u64);
        RngArray { rngs, index: 0 }
    }

    pub fn get_mut(&mut self) -> &mut StdRng {
        self.index = (&mut self.rngs[self.index as usize]).gen_range(0..8);
        &mut self.rngs[self.index as usize]
    }
}
