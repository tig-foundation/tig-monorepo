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
    pub num_blocks:                                 usize,
}

impl crate::DifficultyTrait<4> for Difficulty 
{
    fn from_arr(arr: &[i32; 4])                     -> Self 
    {
        return Self 
        {
            num_vertices:                           arr[0] as usize,
            num_nodes:                              arr[1] as usize,
            num_edges:                              arr[2] as usize,
            num_blocks:                             arr[3] as usize
        };
    }

    fn to_arr(&self)                                -> [i32; 4] 
    {
        return [ self.num_vertices as i32, self.num_nodes as i32, self.num_edges as i32, self.num_blocks as i32 ];
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

fn get_init_values(
    difficulty:                         &Difficulty, 
    seed:                               [u8; 32]
)                                                   -> (Vec<u64>, Vec<u64>, Vec<Vec<u64>>)
{
    let mut rng                                     = SmallRng::from_seed(StdRng::from_seed(seed).gen());

    let mut hyperedge_indices                       = Vec::with_capacity(((difficulty.num_nodes * difficulty.num_edges)+1) / difficulty.num_edges + 1);
    for i in (0..(difficulty.num_nodes * difficulty.num_edges)+1).step_by(difficulty.num_edges)
    {
        hyperedge_indices.push(i as u64);
    }

    let vertices                                    : Vec<u64> = (0..difficulty.num_vertices as u64).collect();

    let mut hyperedges                              = Vec::with_capacity(difficulty.num_nodes);
    for i in 0..difficulty.num_nodes
    {
        let mut vec                                 = Vec::with_capacity(difficulty.num_edges);
        for j in 0..difficulty.num_edges
        {
            vec.push(
                vertices[(rng.next_u32()%difficulty.num_vertices as u32) as usize]
            );
        }

        hyperedges.push(vec);
    }

    return (vertices, hyperedge_indices, hyperedges);
}

impl crate::ChallengeTrait<Solution, Difficulty, 4> for Challenge 
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
        let (vertices, hyperedge_indices, hyperedges) = get_init_values(difficulty, seed);

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

    fn verify_solution(&self, solution: &Solution)  -> Result<()> 
    {
        return Ok(());
    }
}

fn recursive_bipartition(
    vertices_subset:                    &Vec<u64>, 
    partitions:                         &mut Vec<i32>,
    current_id:                         i32, 
    current_depth:                      u32, 
    partitions_per_subset:              usize
)
{
    if current_depth == 0
    {
        for i in 0..vertices_subset.len()
        {
            partitions[i]               = current_id;
        }

        return;
    }

    let half_partitions                 = partitions_per_subset / 2;
    let left_partitions                 = half_partitions;
    let right_partitions                = partitions_per_subset - half_partitions;

    let target_left                     = vertices_subset.len() * left_partitions / partitions_per_subset;
    let target_right                    = vertices_subset.len() - target_left;

    let (left, right)                   = bipartition(vertices_subset, target_left, target_right);

    recursive_bipartition(&left, partitions, current_id * 2, current_depth - 1, left_partitions);
    recursive_bipartition(&right, partitions, current_id * 2 + 1, current_depth - 1, right_partitions);
}

fn bipartition(
    vertices_subset:                    &Vec<u64>, 
    target_left:                        usize, 
    target_right:                       usize
)                                                   -> (Vec<u64>, Vec<u64>)
{
    assert!(target_left + target_right == vertices_subset.len());

    return (Vec::new(), Vec::new());
}

fn solve_shape(
    hyperedges:                         &Vec<Vec<u64>>
)                                                   -> (usize, usize)
{
    return (hyperedges.len(), hyperedges[0].len());
}

fn solve_greedy_bipartition(
    vertices:                           &Vec<u64>, 
    hyperedges:                         &Vec<Vec<u64>>,
    num_partitions:                     Option<usize>
)                                                   -> Vec<i32>
{
    let depth                           = num_partitions.unwrap_or(16).ilog2();
    let M                               = solve_shape(hyperedges).0;

    // Preprocessing: Build mappings
    // vertex_to_hyperedges[v] will contain the hyperedge indices that include vertex v
    let mut vertex_to_hyperedges:       Vec<Vec<usize>> = Vec::with_capacity(vertices.len());
    for i in 0..hyperedges.len()
    {
        for j in hyperedges[i].iter()
        {
            vertex_to_hyperedges[*j as usize].push(i);
        }
    }

    let mut partitions                  = vec![-1 as i32; vertices.len()];
    recursive_bipartition(vertices, &mut partitions, 0, depth, num_partitions.unwrap_or(16));

    return partitions;
}