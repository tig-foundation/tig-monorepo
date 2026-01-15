use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use cudarc::driver::*;
use cudarc::runtime::sys::cudaDeviceProp;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{collections::HashSet, sync::Arc};

impl_kv_string_serde! {
    Track {
        scale: i32,
        n_start: i32,
    }
}

impl_base64_serde! {
    Solution {
        starting_nodes: Vec<i32>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            starting_nodes: Vec::new(),
        }
    }
}

pub struct Challenge {
    pub seed: [u8; 32],
    pub num_nodes: i32,
    pub num_edges: i32,
    pub max_starting_nodes: i32,
    pub d_from_nodes: CudaSlice<i32>,
    pub d_to_nodes: CudaSlice<i32>,
}

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let mut rng = StdRng::from_seed(seed.clone());
        let num_nodes: i32 = 1 << track.scale;
        let num_edges: i32 = 16 * num_nodes;

        let rmat_kernel = module.load_function("rmat_kernel")?;

        let mut d_from_nodes = stream.alloc_zeros::<i32>(num_edges as usize)?;
        let mut d_to_nodes = stream.alloc_zeros::<i32>(num_edges as usize)?;

        unsafe {
            stream
                .launch_builder(&rmat_kernel)
                .arg(&mut d_from_nodes)
                .arg(&mut d_to_nodes)
                .arg(&track.scale)
                .arg(&(0.57f32)) // quadrant a prob
                .arg(&(0.19f32)) // quadrant b prob
                .arg(&(0.19f32)) // quadrant c prob
                .arg(&(0.05f32)) // quadrant d prob
                .arg(&rng.r#gen::<u64>())
                .arg(&num_edges)
                .launch(LaunchConfig {
                    grid_dim: (
                        (num_edges as u32 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
                        1,
                        1,
                    ),
                    block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        Ok(Self {
            seed: *seed,
            num_nodes,
            num_edges,
            max_starting_nodes: track.n_start,
            d_from_nodes,
            d_to_nodes,
        })
    }

    pub fn evaluate_activations(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<i32> {
        if solution.starting_nodes.len() != self.max_starting_nodes as usize {
            return Err(anyhow!(
                "Invalid number of starting nodes. Expected: {}, Actual: {}",
                self.max_starting_nodes,
                solution.starting_nodes.len()
            ));
        }
        if solution
            .starting_nodes
            .iter()
            .any(|&n| n < 0 || n >= self.num_nodes)
        {
            return Err(anyhow!("Starting nodes contain invalid node indices."));
        }
        if solution.starting_nodes.len()
            != solution.starting_nodes.iter().collect::<HashSet<_>>().len()
        {
            return Err(anyhow!("Starting nodes contain duplicates."));
        }

        let cascade_kernel = module.load_function("cascade")?;

        let mut activated_nodes: Vec<bool> = vec![false; self.num_nodes as usize];
        for &node in &solution.starting_nodes {
            activated_nodes[node as usize] = true;
        }

        let mut d_activated_nodes = stream.memcpy_stod(&activated_nodes)?;
        let mut d_cascade_nodes = d_activated_nodes.clone();
        let mut d_next_cascade_nodes = stream.alloc_zeros::<bool>(self.num_nodes as usize)?;
        let mut d_num_new_activations = stream.alloc_zeros::<i32>(1)?;

        let mut total_activations = self.max_starting_nodes;

        let mut rng = StdRng::from_seed(self.seed.clone());
        loop {
            let seed = rng.r#gen::<u64>();
            unsafe {
                stream
                    .launch_builder(&cascade_kernel)
                    .arg(&self.d_from_nodes)
                    .arg(&self.d_to_nodes)
                    .arg(&mut d_activated_nodes)
                    .arg(&mut d_cascade_nodes)
                    .arg(&mut d_next_cascade_nodes)
                    .arg(&mut d_num_new_activations)
                    .arg(&0.01f32) // activation probability
                    .arg(&seed)
                    .arg(&self.num_edges)
                    .launch(LaunchConfig {
                        grid_dim: (
                            (self.num_edges as u32 + MAX_THREADS_PER_BLOCK - 1)
                                / MAX_THREADS_PER_BLOCK,
                            1,
                            1,
                        ),
                        block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            stream.synchronize()?;

            let new_activations = stream.memcpy_dtov(&d_num_new_activations)?[0];
            if new_activations == 0 {
                break;
            }
            total_activations += new_activations;

            std::mem::swap(&mut d_cascade_nodes, &mut d_next_cascade_nodes);
            stream.memset_zeros(&mut d_next_cascade_nodes)?;
            stream.memset_zeros(&mut d_num_new_activations)?;
        }
        Ok(total_activations)
    }

    conditional_pub!(
        fn compute_greedy_baseline(
            &self,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            _prop: &cudaDeviceProp,
        ) -> Result<Solution> {
            let count_degrees_kernel = module.load_function("count_degrees_kernel").unwrap();
            let mut d_degrees = stream.alloc_zeros::<i32>(self.num_nodes as usize).unwrap();
            unsafe {
                stream
                    .launch_builder(&count_degrees_kernel)
                    .arg(&self.d_from_nodes)
                    .arg(&mut d_degrees)
                    .arg(&self.num_edges)
                    .launch(LaunchConfig {
                        grid_dim: (
                            (self.num_edges as u32 + MAX_THREADS_PER_BLOCK - 1)
                                / MAX_THREADS_PER_BLOCK,
                            1,
                            1,
                        ),
                        block_dim: (MAX_THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            stream.synchronize().unwrap();

            let degrees = stream.memcpy_dtov(&d_degrees).unwrap();
            let mut node_degree_pairs: Vec<(i32, i32)> = degrees
                .iter()
                .enumerate()
                .map(|(node, &deg)| (node as i32, deg))
                .collect();
            node_degree_pairs.sort_by(|a, b| b.1.cmp(&a.1));
            let highest_degree_nodes: Vec<i32> = node_degree_pairs
                [0..self.max_starting_nodes as usize]
                .iter()
                .map(|&(node, _)| node)
                .collect();
            Ok(Solution {
                starting_nodes: highest_degree_nodes,
            })
        }
    );

    conditional_pub!(
        fn compute_sota_baseline(&self) -> Result<Solution> {
            Err(anyhow!("Not implemented yet"))
        }
    );

    conditional_pub!(
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            _prop: &cudaDeviceProp,
        ) -> Result<i32> {
            let activations =
                self.evaluate_activations(solution, module.clone(), stream.clone(), _prop)?;
            let baseline_solution =
                self.compute_greedy_baseline(module.clone(), stream.clone(), _prop)?;
            let baseline_activations = self.evaluate_activations(
                &baseline_solution,
                module.clone(),
                stream.clone(),
                _prop,
            )?;
            let quality =
                (baseline_activations as f64 - activations as f64) / baseline_activations as f64;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}
