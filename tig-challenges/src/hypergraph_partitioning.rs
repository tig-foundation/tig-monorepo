use anyhow::{anyhow, Result};
use rand::
{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng, RngCore
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
    pub num_vertices:                               usize,
    pub num_nodes:                                  usize,
    pub num_edges:                                  usize,
}

impl crate::DifficultyTrait<3> for Difficulty 
{
    fn from_arr(arr: &[i32; 3])                     -> Self 
    {
        return Self 
        {
            num_vertices:                           arr[0] as usize,
            num_nodes:                              arr[1] as usize,
            num_edges:                              arr[2] as usize,
        };
    }

    fn to_arr(&self)                                -> [i32; 3] 
    {
        return [ self.num_vertices as i32, self.num_nodes as i32, self.num_edges as i32 ];
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
        return from_value(Value::Object(v));
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Challenge 
{
    pub seed:                                       [u8; 32],
    pub difficulty:                                 Difficulty,
    pub vertices:                                   Vec<u64>,
    pub hyperedge_indices:                          Vec<u64>,
    pub hyperedges:                                 Vec<Vec<u64>>,
    pub node_weights:                               Vec<f32>,
    pub edge_weights:                               Vec<f32>,
}

// TIG dev bounty available for a GPU optimisation for instance generation!
#[cfg(feature = "cuda")]
pub const KERNEL:                                   Option<CudaKernel> = None;

impl crate::ChallengeTrait<Solution, Difficulty, 3> for Challenge 
{
    #[cfg(feature = "cuda")]
    fn cuda_generate_instance(
        seed:                           [u8; 32],
        difficulty:                     &Difficulty,
        dev:                            &Arc<CudaDevice>,
        mut funcs:                      HashMap<&'static str, CudaFunction>,
    )                                               -> Result<Self> 
    {
        // TIG dev bounty available for a GPU optimisation for instance generation!
        return Self::generate_instance(seed, difficulty);
    }

    fn generate_instance(
        seed:                           [u8; 32], 
        difficulty:                     &Difficulty
    )                                               -> Result<Challenge> 
    {
        let mut rng                                 = SmallRng::from_seed(StdRng::from_seed(seed).gen());

        let mut hyperedge_indices                   = Vec::new();
        for i in (0..(difficulty.num_nodes * difficulty.num_edges)+1).step_by(difficulty.num_edges)
        {
            hyperedge_indices.push(i as u64 * difficulty.num_nodes as u64);
        }

        let vertices                                : Vec<u64> = (0..difficulty.num_vertices as u64).collect();

        let mut hyperedges                          = Vec::new();
        for i in 0..difficulty.num_nodes
        {
            let mut vec                             = Vec::new();
            for j in 0..difficulty.num_edges
            {
                vec.push(
                    vertices[(rng.next_u32()%difficulty.num_vertices as u32) as usize]
                );
            }

            hyperedges.push(vec);
        }

        return Ok(Challenge
        {
            seed:                                   seed,
            difficulty:                             difficulty.clone(),
            vertices:                               vertices,
            hyperedge_indices:                      hyperedge_indices,
            hyperedges:                             hyperedges,
            node_weights:                           vec![1.0f32; difficulty.num_vertices as usize],
            edge_weights:                           vec![1.0f32; difficulty.num_edges as usize],
        });
    }

    fn verify_solution(&self, solution: &Solution) -> Result<()> 
    {
        return Ok(());
    }
}
