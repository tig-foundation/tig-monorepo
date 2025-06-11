use anyhow::{anyhow, Result};
use cudarc::driver::*;
use cudarc::runtime::sys::cudaDeviceProp;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub num_hyperedges: u32,
    pub better_than_baseline: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            num_hyperedges: arr[0] as u32,
            better_than_baseline: arr[1] as u32,
        }
    }
}
impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.num_hyperedges as i32, self.better_than_baseline as i32]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Solution {
    pub sub_solutions: Vec<SubSolution>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubSolution {
    pub partition: Vec<u32>,
}

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;

    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub sub_instances: Vec<SubInstance>,
}

pub struct SubInstance {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub num_nodes: u32,
    pub num_parts: u32,
    pub max_part_size: u32,
    pub total_connections: u32,
    pub d_hyperedge_sizes: CudaSlice<i32>,
    // start = hyperedge_offsets[i], end = hyperedge_offsets[i + 1]
    // nodes_in_hyperedge_i = hyperedge_nodes[start..end]
    pub d_hyperedge_offsets: CudaSlice<i32>,
    pub d_hyperedge_nodes: CudaSlice<i32>,
    // start = node_offsets[j], end = node_offsets[j + 1]
    // hyperedge_with_node_j = node_hyperedges[start..end]
    pub d_node_degrees: CudaSlice<i32>,
    pub d_node_offsets: CudaSlice<i32>,
    pub d_node_hyperedges: CudaSlice<i32>,
    pub baseline_connectivity_metric: u32,
}

pub const NUM_SUB_INSTANCES: usize = 4;
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        difficulty: &Difficulty,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        prop: &cudaDeviceProp,
    ) -> Result<Challenge> {
        let mut rng = StdRng::from_seed(seed.clone());
        let mut sub_instances = Vec::new();
        for _ in 0..NUM_SUB_INSTANCES {
            sub_instances.push(SubInstance::generate_instance(
                &rng.gen(),
                difficulty,
                module.clone(),
                stream.clone(),
                prop,
            )?);
        }

        Ok(Challenge {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            sub_instances,
        })
    }

    pub fn verify_solution(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        prop: &cudaDeviceProp,
    ) -> Result<()> {
        let mut better_than_baselines = Vec::new();
        for (i, (sub_instance, sub_solution)) in self
            .sub_instances
            .iter()
            .zip(&solution.sub_solutions)
            .enumerate()
        {
            match sub_instance.verify_solution(sub_solution, module.clone(), stream.clone(), prop) {
                Ok(connectivity_metric) => better_than_baselines.push(
                    connectivity_metric as f64 / sub_instance.baseline_connectivity_metric as f64,
                ),
                Err(e) => return Err(anyhow!("Instance {}: {}", i, e.to_string())),
            }
        }
        let average = 1.0
            - (better_than_baselines.iter().map(|x| x * x).sum::<f64>()
                / better_than_baselines.len() as f64)
                .sqrt();
        let threshold = self.difficulty.better_than_baseline as f64 / 1000.0;
        if average >= threshold {
            Ok(())
        } else {
            Err(anyhow!(
                "Average better_than_baseline ({}) is less than ({})",
                average,
                threshold
            ))
        }
    }
}

impl SubInstance {
    pub fn generate_instance(
        seed: &[u8; 32],
        difficulty: &Difficulty,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let mut rng = StdRng::from_seed(seed.clone());
        let num_hyperedges = difficulty.num_hyperedges;
        let target_num_nodes = difficulty.num_hyperedges; // actual number may be around 8% less
        let depth = 6;
        let num_parts = 1 << depth; // 2^6 = 64 partitions
        let level_weights: Vec<f32> = vec![
            20.0,
            20.0 + rng.gen::<f32>() * 4.0,
            19.0 + rng.gen::<f32>() * 14.0,
            20.0 + rng.gen::<f32>() * 6.0,
            1.0 + rng.gen::<f32>() * 15.0,
            rng.gen::<f32>() * 0.3,
            rng.gen::<f32>() * 0.1,
            rng.gen::<f32>() * 0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        // Get kernels
        let generate_hyperedge_sizes_kernel = module.load_function("generate_hyperedge_sizes")?;
        let finalize_hyperedge_sizes_kernel = module.load_function("finalize_hyperedge_sizes")?;
        let generate_node_weights_kernel = module.load_function("generate_node_weights")?;
        let generate_hyperedges_kernel = module.load_function("generate_hyperedges")?;
        let finalize_hyperedges_kernel = module.load_function("finalize_hyperedges")?;
        let initialize_partitioning_kernel = module.load_function("initialize_partitioning")?;
        let greedy_bipartition_kernel = module.load_function("greedy_bipartition")?;
        let finalize_bipartition_kernel = module.load_function("finalize_bipartition")?;
        let shuffle_nodes_kernel = module.load_function("shuffle_nodes")?;
        let finalize_shuffle_kernel = module.load_function("finalize_shuffle")?;
        let calc_connectivity_metric_kernel = module.load_function("calc_connectivity_metric")?;

        let block_size = MAX_THREADS_PER_BLOCK;
        let cfg = LaunchConfig {
            grid_dim: ((num_hyperedges + block_size - 1) / block_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let one_block_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let one_thread_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_seed = stream.memcpy_stod(seed)?;
        let mut d_hyperedge_sizes = stream.alloc_zeros::<i32>(num_hyperedges as usize)?;
        let mut d_hyperedge_offsets = stream.alloc_zeros::<i32>((num_hyperedges + 1) as usize)?;
        let mut d_node_weights = stream.alloc_zeros::<f32>(target_num_nodes as usize)?;
        let mut d_total_connections = stream.alloc_zeros::<u32>(1)?;

        // 1.1 Generate node weights
        let min_node_weight: f32 = 1.0;
        let max_node_weight: f32 = 4966.0;
        let alpha: f32 = 1.0 - 2.2864;
        unsafe {
            stream
                .launch_builder(&generate_node_weights_kernel)
                .arg(&d_seed)
                .arg(&target_num_nodes)
                .arg(&min_node_weight)
                .arg(&max_node_weight)
                .arg(&alpha)
                .arg(&mut d_node_weights)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 1.2 Generate hyperedge sizes
        let min_hyperedge_size: u32 = 2;
        let max_hyperedge_size: u32 = 1954.min(target_num_nodes);
        let alpha: f32 = 1.0 - 2.5608;
        unsafe {
            stream
                .launch_builder(&generate_hyperedge_sizes_kernel)
                .arg(&d_seed)
                .arg(&num_hyperedges)
                .arg(&min_hyperedge_size)
                .arg(&max_hyperedge_size)
                .arg(&alpha)
                .arg(&mut d_hyperedge_sizes)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 1.3 Finalize hyperedge sizes
        unsafe {
            stream
                .launch_builder(&finalize_hyperedge_sizes_kernel)
                .arg(&num_hyperedges)
                .arg(&d_hyperedge_sizes)
                .arg(&mut d_hyperedge_offsets)
                .arg(&mut d_total_connections)
                .launch(one_thread_cfg.clone())?;
        }
        stream.synchronize()?;

        // Get hyperedge offsets for hyperedge_nodes allocation
        let total_connections = stream.memcpy_dtov(&d_total_connections)?[0];

        // 1.4 Generate hyperedges
        let d_level_weights = stream.memcpy_stod(&level_weights)?;
        let mut d_hyperedge_nodes = stream.alloc_zeros::<i32>(total_connections as usize)?;
        let mut d_node_degrees = stream.alloc_zeros::<i32>(target_num_nodes as usize)?;
        let mut d_node_hyperedges = stream.alloc_zeros::<i32>(total_connections as usize)?;
        let mut d_node_offsets = stream.alloc_zeros::<i32>(target_num_nodes as usize + 1)?;
        unsafe {
            stream
                .launch_builder(&generate_hyperedges_kernel)
                .arg(&d_seed)
                .arg(&target_num_nodes)
                .arg(&num_hyperedges)
                .arg(&total_connections)
                .arg(&d_hyperedge_sizes)
                .arg(&d_hyperedge_offsets)
                .arg(&d_node_weights)
                .arg(&d_level_weights)
                .arg(&mut d_hyperedge_nodes)
                .arg(&mut d_node_degrees)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 1.5 Finalize hyperedges
        unsafe {
            stream
                .launch_builder(&finalize_hyperedges_kernel)
                .arg(&target_num_nodes)
                .arg(&num_hyperedges)
                .arg(&d_hyperedge_sizes)
                .arg(&d_hyperedge_offsets)
                .arg(&d_hyperedge_nodes)
                .arg(&d_node_degrees)
                .arg(&mut d_node_hyperedges)
                .arg(&mut d_node_offsets)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 2.1 Initialize partitioning
        let mut d_partition = stream.alloc_zeros::<i32>(target_num_nodes as usize)?;
        let mut d_sorted_nodes = stream.alloc_zeros::<i32>(target_num_nodes as usize)?;

        unsafe {
            stream
                .launch_builder(&initialize_partitioning_kernel)
                .arg(&target_num_nodes)
                .arg(&d_node_degrees)
                .arg(&mut d_partition)
                .arg(&mut d_sorted_nodes)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 2.2 Greedy bipartitioning
        for level in 1..depth {
            let num_parts_this_level = 1 << level;
            let num_flags = (num_hyperedges + 63) / 64 * num_parts_this_level;
            let mut d_left_hyperedge_flags = stream.alloc_zeros::<u64>(num_flags as usize)?;
            let mut d_right_hyperedge_flags = stream.alloc_zeros::<u64>(num_flags as usize)?;
            let d_curr_partition = d_partition.clone();

            unsafe {
                stream
                    .launch_builder(&greedy_bipartition_kernel)
                    .arg(&level)
                    .arg(&target_num_nodes)
                    .arg(&num_hyperedges)
                    .arg(&d_node_hyperedges)
                    .arg(&d_node_offsets)
                    .arg(&d_sorted_nodes)
                    .arg(&d_node_degrees)
                    .arg(&d_curr_partition)
                    .arg(&mut d_partition)
                    .arg(&mut d_left_hyperedge_flags)
                    .arg(&mut d_right_hyperedge_flags)
                    .launch(LaunchConfig {
                        grid_dim: (num_parts_this_level as u32, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 400,
                    })?;
            }
            stream.synchronize()?;
        }

        // 2.3 Finalize bipartitioning
        unsafe {
            stream
                .launch_builder(&finalize_bipartition_kernel)
                .arg(&target_num_nodes)
                .arg(&num_parts)
                .arg(&mut d_partition)
                .launch(one_block_cfg.clone())?;
        }
        stream.synchronize()?;

        // 3.1 Shuffle nodes
        let mut d_rand_weights = stream.alloc_zeros::<f32>(target_num_nodes as usize)?;
        let mut d_shuffled_partition = stream.alloc_zeros::<i32>(target_num_nodes as usize)?;
        let mut d_shuffled_hyperedge_nodes =
            stream.alloc_zeros::<i32>(total_connections as usize)?;
        let mut d_shuffled_node_weights = stream.alloc_zeros::<f32>(target_num_nodes as usize)?;
        let mut d_shuffled_node_degrees = stream.alloc_zeros::<i32>(target_num_nodes as usize)?;
        let mut d_num_prune = stream.alloc_zeros::<u32>(1)?;
        unsafe {
            stream
                .launch_builder(&shuffle_nodes_kernel)
                .arg(&d_seed)
                .arg(&target_num_nodes)
                .arg(&d_partition)
                .arg(&d_hyperedge_sizes)
                .arg(&d_hyperedge_offsets)
                .arg(&d_hyperedge_nodes)
                .arg(&d_node_degrees)
                .arg(&d_node_hyperedges)
                .arg(&d_node_offsets)
                .arg(&d_node_weights)
                .arg(&d_sorted_nodes)
                .arg(&mut d_rand_weights)
                .arg(&mut d_shuffled_partition)
                .arg(&mut d_shuffled_hyperedge_nodes)
                .arg(&mut d_shuffled_node_weights)
                .arg(&mut d_shuffled_node_degrees)
                .arg(&mut d_num_prune)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let num_prune = stream.memcpy_dtov(&d_num_prune)?[0];
        let num_nodes = target_num_nodes - num_prune;

        // 3.2 Finalize shuffle
        unsafe {
            stream
                .launch_builder(&finalize_shuffle_kernel)
                .arg(&d_seed)
                .arg(&num_hyperedges)
                .arg(&d_hyperedge_sizes)
                .arg(&d_hyperedge_offsets)
                .arg(&mut d_shuffled_hyperedge_nodes)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 3.3 Reconstruct node hyperedges and offsets
        let mut d_shuffled_node_hyperedges =
            stream.alloc_zeros::<i32>(total_connections as usize)?;
        let mut d_shuffled_node_offsets = stream.alloc_zeros::<i32>(num_nodes as usize + 1)?;
        unsafe {
            stream
                .launch_builder(&finalize_hyperedges_kernel)
                .arg(&num_nodes)
                .arg(&num_hyperedges)
                .arg(&d_hyperedge_sizes)
                .arg(&d_hyperedge_offsets)
                .arg(&d_shuffled_hyperedge_nodes)
                .arg(&d_shuffled_node_degrees)
                .arg(&mut d_shuffled_node_hyperedges)
                .arg(&mut d_shuffled_node_offsets)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        // 4.1 Calculate connectivity
        let mut d_connectivity_metric = stream.alloc_zeros::<u32>(1)?;

        unsafe {
            stream
                .launch_builder(&calc_connectivity_metric_kernel)
                .arg(&num_hyperedges)
                .arg(&d_hyperedge_offsets)
                .arg(&d_shuffled_hyperedge_nodes)
                .arg(&d_shuffled_partition)
                .arg(&mut d_connectivity_metric)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let connectivity_metric = stream.memcpy_dtov(&d_connectivity_metric)?[0];
        let max_part_size = ((num_nodes as f32 / num_parts as f32) * 1.03).ceil() as u32;

        Ok(Self {
            seed: *seed,
            difficulty: difficulty.clone(),
            num_nodes: target_num_nodes - num_prune,
            num_parts,
            max_part_size,
            total_connections,
            d_hyperedge_sizes,
            d_hyperedge_offsets,
            d_hyperedge_nodes: d_shuffled_hyperedge_nodes,
            d_node_degrees: d_shuffled_node_degrees,
            d_node_offsets: d_shuffled_node_offsets,
            d_node_hyperedges: d_shuffled_node_hyperedges,
            baseline_connectivity_metric: connectivity_metric,
        })
    }

    pub fn verify_solution(
        &self,
        solution: &SubSolution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<u32> {
        if solution.partition.len() != self.num_nodes as usize {
            return Err(anyhow!(
                "Invalid number of partitions. Expected: {}, Actual: {}",
                self.num_nodes,
                solution.partition.len()
            ));
        }

        // Get the kernels
        let validate_partition_kernel = module.load_function("validate_partition")?;
        let calc_connectivity_metric_kernel = module.load_function("calc_connectivity_metric")?;
        let count_nodes_in_part_kernel = module.load_function("count_nodes_in_part")?;

        let block_size = MAX_THREADS_PER_BLOCK;
        let grid_size = (self.difficulty.num_hyperedges + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // 1.1 Check if all nodes are assigned to a part
        let d_partition = stream.memcpy_stod(&solution.partition)?;
        let mut d_error_flag = stream.alloc_zeros::<u32>(1)?;

        unsafe {
            stream
                .launch_builder(&validate_partition_kernel)
                .arg(&self.num_nodes)
                .arg(&self.num_parts)
                .arg(&d_partition)
                .arg(&mut d_error_flag)
                .launch(cfg)?;
        }
        stream.synchronize()?;

        if stream.memcpy_dtov(&d_error_flag)?[0] != 0 {
            return Err(anyhow!(
                "Invalid partition. All nodes must be assigned to one of {} parts",
                self.num_parts
            ));
        };

        // 1.2 Check if any partition exceeds the maximum size
        let mut d_nodes_in_part = stream.alloc_zeros::<u32>(self.num_parts as usize)?;
        unsafe {
            stream
                .launch_builder(&count_nodes_in_part_kernel)
                .arg(&self.num_nodes)
                .arg(&self.num_parts)
                .arg(&d_partition)
                .arg(&mut d_nodes_in_part)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let nodes_in_partition = stream.memcpy_dtov(&d_nodes_in_part)?;
        if nodes_in_partition
            .iter()
            .any(|&x| x < 1 || x > self.max_part_size)
        {
            return Err(anyhow!(
                "Each part must have at least 1 and at most {} nodes",
                self.max_part_size
            ));
        }

        // 1.3 Calculate connectivity
        let mut d_connectivity_metric = stream.alloc_zeros::<u32>(1)?;
        unsafe {
            stream
                .launch_builder(&calc_connectivity_metric_kernel)
                .arg(&self.difficulty.num_hyperedges)
                .arg(&self.d_hyperedge_offsets)
                .arg(&self.d_hyperedge_nodes)
                .arg(&d_partition)
                .arg(&mut d_connectivity_metric)
                .launch(cfg.clone())?;
        }
        stream.synchronize()?;

        let connectivity_metric = stream.memcpy_dtov(&d_connectivity_metric)?[0];
        Ok(connectivity_metric)
    }
}
