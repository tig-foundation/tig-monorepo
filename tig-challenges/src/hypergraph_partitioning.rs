use anyhow::{anyhow, Result};
use rand::
{
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
pub struct Difficulty 
{
    pub num_items:                                  usize,
    pub better_than_baseline:                       u32,
}

impl crate::DifficultyTrait<2> for Difficulty {
    fn from_arr(arr: &[i32; 2])                     -> Self 
    {
        return Self 
        {
            num_items:                              arr[0] as usize,
            better_than_baseline:                   arr[1] as u32,
        };
    }

    fn to_arr(&self)                                -> [i32; 2] 
    {
        return [ self.num_items as i32, self.better_than_baseline as i32 ];
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Solution 
{
    pub items:                                      Vec<usize>,
}

impl crate::SolutionTrait for Solution 
{
}

impl TryFrom<Map<String, Value>> for Solution 
{
    type Error                                      = serde_json::Error;

    fn try_from(v: Map<String, Value>)              -> Result<Self, Self::Error> 
    {
        from_value(Value::Object(v))
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Challenge 
{
    pub seed:                                       [u8; 32],
    pub difficulty:                                 Difficulty,
    pub weights:                                    Vec<u32>,
    pub values:                                     Vec<u32>,
    pub interaction_values:                         Vec<Vec<i32>>,
    pub max_weight:                                 u32,
    pub min_value:                                  u32,
}

// TIG dev bounty available for a GPU optimisation for instance generation!
#[cfg(feature = "cuda")]
pub const KERNEL: Option<CudaKernel> = None;

impl crate::ChallengeTrait<Solution, Difficulty, 2> for Challenge 
{
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seed: [u8; 32],
        difficulty: &Difficulty,
        dev: &Arc<CudaDevice>,
        mut funcs: HashMap<&'static str, CudaFunction>,
    ) -> Result<Self> 
    {
        // TIG dev bounty available for a GPU optimisation for instance generation!
        Self::generate_instance(seed, difficulty)
    }

    fn generate_instance(seed: [u8; 32], difficulty: &Difficulty) -> Result<Challenge> 
    {
        return Ok(Challenge
        {
            seed:                                   seed,
            difficulty:                             difficulty.clone(),
            weights:                                vec![],
            values:                                 vec![],
            interaction_values:                     vec![],
            max_weight:                             0,
            min_value:                              0,
        });
    }

    fn verify_solution(&self, solution: &Solution) -> Result<()> 
    {
        return Ok(());
    }
}
