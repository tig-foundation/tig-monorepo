use anyhow::{anyhow, Result};
use rand::
{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng, RngCore
};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::collections::HashSet;
use mt19937::{MT19937};

#[cfg(feature = "cuda")]
use crate::CudaKernel;
#[cfg(feature = "cuda")]
use cudarc::driver::*;
#[cfg(feature = "cuda")]
use std::{collections::HashMap, sync::Arc};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty 
{
    pub num_vertices:                                   usize,
    pub num_nodes:                                      usize,
    pub num_edges:                                      usize,
    pub num_blocks:                                     usize,
}

impl crate::DifficultyTrait<4> for Difficulty 
{
    fn from_arr(arr: &[i32; 4])                     -> Self 
    {
        return Self 
        {
            num_vertices:                               arr[0] as usize,
            num_nodes:                                  arr[1] as usize,
            num_edges:                                  arr[2] as usize,
            num_blocks:                                 arr[3] as usize
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
    pub partitions                                      : Vec<i32>
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
    pub seed:                                           [u8; 32],
    pub difficulty:                                     Difficulty,
    pub vertices:                                       Vec<u64>,
    pub hyperedge_indices:                              Vec<u64>,
    pub hyperedges:                                     Vec<Vec<u64>>,
}

// TIG dev bounty available for a GPU optimisation for instance generation!
#[cfg(feature = "cuda")]
pub const KERNEL:                                   Option<CudaKernel> = None;

fn get_hyperedges(
    difficulty:                         &Difficulty,
    seed:                               [u8; 32],
    vertices:                           &Vec<u64>
)                                                   -> Vec<Vec<u64>>
{
    /*
    let mut rng                                         = SmallRng::from_seed(StdRng::from_seed(seed).gen());

    let mut hyperedges                                  = Vec::with_capacity(difficulty.num_nodes);
    for i in 0..difficulty.num_nodes
    {
        let mut vec                                     = Vec::with_capacity(difficulty.num_edges);
        for j in 0..difficulty.num_edges
        {
            let mut r_vtx;
            while // do while
            {
                r_vtx                                   = rng.next_u32() as usize % vertices.len();

                vec.contains(&vertices[r_vtx as usize])
            } {}

            vec.push(vertices[r_vtx as usize]);
        }

        hyperedges.push(vec);
    }*/
    
    let mut rng                                         = mt19937::MT19937::new_with_slice_seed(&[5489]);
    
    let mut hyperedges                                  = Vec::with_capacity(difficulty.num_nodes);
    for i in 0..difficulty.num_nodes
    {
        let mut vec                                     = Vec::with_capacity(difficulty.num_edges);
        for j in 0..difficulty.num_edges
        {
            let mut r_vtx;
            while
            {
                r_vtx                                   = ((vertices.len() - 1) as f64 * mt19937::gen_res53(&mut rng)).round() as u64;

                vec.contains(&vertices[r_vtx as usize])
            } {}

            vec.push(vertices[r_vtx as usize]);
        }

        hyperedges.push(vec);
    }

    return hyperedges;
}

fn get_init_values(
    difficulty:                         &Difficulty, 
    seed:                               [u8; 32]
)                                                   -> (Vec<u64>, Vec<u64>, Vec<Vec<u64>>)
{
    let mut hyperedge_indices                           = Vec::with_capacity(((difficulty.num_nodes * difficulty.num_edges)+1) / difficulty.num_edges + 1);
    for i in (0..(difficulty.num_nodes * difficulty.num_edges)+1).step_by(difficulty.num_edges)
    {
        hyperedge_indices.push(i as u64);
    }

    let vertices                                        : Vec<u64> = (0..difficulty.num_vertices as u64).collect();
    let hyperedges                                      = get_hyperedges(difficulty, seed, &vertices);

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
        let (vertices, hyperedge_indices, hyperedges)   = get_init_values(difficulty, seed);

        return Ok(Challenge
        {
            seed:                                       seed,
            difficulty:                                 difficulty.clone(),
            vertices:                                   vertices,
            hyperedge_indices:                          hyperedge_indices,
            hyperedges:                                 hyperedges
        });
    }

    fn verify_solution(&self, solution: &Solution) -> Result<()> 
    {
        if self.vertices.len() != solution.partitions.len()
        {
            return Err(anyhow!("self.vertices.len() != solution.partitions.len()"));
        }

        for part in &solution.partitions
        {
            if *part < 0 || *part as usize >= self.difficulty.num_blocks      
            {
                return Err(anyhow!("part < 0 || part >= self.difficulty.num_blocks"));
            }
        }

        let partition                                   = solve_greedy_bipartition(&self.vertices, &self.hyperedges, Some(self.difficulty.num_blocks));
        let greedy_connectivity                         = calculate_connectivity(&partition, &self.hyperedges);

        let connectivity                                = calculate_connectivity(&solution.partitions, &self.hyperedges);

        if connectivity < greedy_connectivity
        {
            return Err(anyhow!("connectivity < greedy_connectivity"));
        }
        
        return Ok(());
    }
}

fn calculate_connectivity(
    partition_mapping:                      &Vec<i32>, 
    hyperedges:                             &Vec<Vec<u64>>
)                                                   -> usize
{
    let mut connectivity                                = 0;

    for i in 0..hyperedges.len()
    {
        let mut parts_in_h                              = Vec::with_capacity(hyperedges[i].len());
        for j in 0..hyperedges[i].len()
        {
            if !parts_in_h.contains(&partition_mapping[hyperedges[i][j] as usize])
            {
                parts_in_h.push(partition_mapping[hyperedges[i][j] as usize]);
            }
        }

        connectivity                                    += parts_in_h.len() - 1;
    }
     
    return connectivity;
}

fn recursive_bipartition(
    vertices_subset:                    &Vec<u64>, 
    partitions:                         &mut Vec<i32>,
    vertex_to_hyperedges:               &Vec<Vec<usize>>,
    shape_hyperedge:                    usize,
    current_id:                         i32, 
    current_depth:                      u32, 
    partitions_per_subset:              usize
)
{
    if current_depth == 0
    {
        for i in vertices_subset
        {
            partitions[*i as usize]                     = current_id;
        }

        return;
    }

    let half_partitions                                 = partitions_per_subset / 2;
    let left_partitions                                 = half_partitions;
    let right_partitions                                = partitions_per_subset - half_partitions;

    let target_left                                     = vertices_subset.len() * left_partitions / partitions_per_subset;
    let target_right                                    = vertices_subset.len() - target_left;

    let (left, right)                                   = bipartition(vertices_subset, vertex_to_hyperedges, shape_hyperedge, target_left, target_right);

    recursive_bipartition(&left, partitions, vertex_to_hyperedges, shape_hyperedge, current_id * 2, current_depth - 1, left_partitions);
    recursive_bipartition(&right, partitions, vertex_to_hyperedges, shape_hyperedge, current_id * 2 + 1, current_depth - 1, right_partitions);
}

use std::cmp::Reverse;
fn bipartition(
    vertices_subset:                    &Vec<u64>,
    vertex_to_hyperedges:               &Vec<Vec<usize>>,
    shape_hyperedge:                    usize,
    target_left:                        usize, 
    target_right:                       usize
)                                                   -> (Vec<u64>, Vec<u64>)
{
    assert!(target_left + target_right == vertices_subset.len());

    let mut degrees                                     = Vec::<i64>::with_capacity(vertices_subset.len());
    for v in vertices_subset
    {
        degrees.push(vertex_to_hyperedges[*v as usize].len() as i64 * -1);
    }

    let mut indices                                     : Vec<usize> = (0..vertices_subset.len()).collect();
    indices.sort_by_key(|&i| degrees[i as usize]);
    
    let mut sorted_vertices                             : Vec<u64> = Vec::with_capacity(vertices_subset.len());
    for i in indices
    {
        sorted_vertices.push(vertices_subset[i]);
    }

    let mut left                                        : Vec<u64> = Vec::with_capacity(target_left);
    let mut right                                       : Vec<u64> = Vec::with_capacity(target_right);
    let (mut current_left, mut current_right)           = (0, 0);

    let mut hyperedges_left                             : Vec<bool> = vec![false; shape_hyperedge];
    let mut hyperedges_right                            : Vec<bool> = vec![false; shape_hyperedge];

    for v in &sorted_vertices
    {
        let (mut connections_left, mut connections_right) = (0, 0);
        for h in &vertex_to_hyperedges[*v as usize]
        {
            if hyperedges_left[*h]
            {
                connections_left                        += 1;
            }

            if hyperedges_right[*h]
            {
                connections_right                       += 1;
            }
        }

        if connections_left > connections_right
        {
            if current_left < target_left
            {
                left.push(*v);

                current_left                            += 1;
                for h in &vertex_to_hyperedges[*v as usize]
                {
                    hyperedges_left[*h]                 = true;
                }
            }
            else
            {
                right.push(*v);

                current_right                           += 1;
                for h in &vertex_to_hyperedges[*v as usize]
                {
                    hyperedges_right[*h]                = true;
                }
            }
        }
        else if connections_left < connections_right
        {
            if current_right < target_right
            {
                right.push(*v);

                current_right                           += 1;
                for h in &vertex_to_hyperedges[*v as usize]
                {
                    hyperedges_left[*h]                 = true;
                }
            }
            else
            {
                left.push(*v);

                current_left                            += 1;
                for h in &vertex_to_hyperedges[*v as usize]
                {
                    hyperedges_right[*h]                = true;
                }
            }
        }
        else
        {
            if current_left < target_left
            {
                left.push(*v);

                current_left                            += 1;
                for h in &vertex_to_hyperedges[*v as usize]
                {
                    hyperedges_left[*h]                 = true;
                }
            }
            else
            {
                right.push(*v);

                current_right                           += 1;
                for h in &vertex_to_hyperedges[*v as usize]
                {
                    hyperedges_right[*h]                = true;
                }
            }
        }
    }

    while current_left < target_left && sorted_vertices.len() > (left.len() + right.len())
    {
        left.push(sorted_vertices[left.len() + right.len()]);

        current_left                                += 1;
    }

    while current_right < target_right && sorted_vertices.len() > (left.len() + right.len())
    {
        right.push(sorted_vertices[left.len() + right.len()]);

        current_right                               += 1;
    }

    assert!(left.len() == target_left);
    assert!(right.len() == target_right);

    return (left, right);
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
    let depth                                           = num_partitions.unwrap_or(16).ilog2();
    let M                                               = solve_shape(hyperedges).0;

    // Preprocessing: Build mappings
    // vertex_to_hyperedges[v] will contain the hyperedge indices that include vertex v
    let mut vertex_to_hyperedges                        : Vec<Vec<usize>> = vec![Vec::with_capacity(hyperedges.len()); vertices.len()];
    for i in 0..hyperedges.len()
    {
        for j in hyperedges[i].iter()
        {
            vertex_to_hyperedges[*j as usize].push(i);
        }
    }

    let mut partitions                                  = vec![-1 as i32; vertices.len()];
    recursive_bipartition(vertices, &mut partitions, &vertex_to_hyperedges, M, 0, depth, num_partitions.unwrap_or(16));

    return partitions;
}