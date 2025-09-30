use anyhow::{anyhow, Ok, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{from_value, Map, Value};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Difficulty {
    pub num_queries: u32,
    pub better_than_baseline: u32,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            num_queries: arr[0] as u32,
            better_than_baseline: arr[1] as u32,
        }
    }
}
impl Into<Vec<i32>> for Difficulty {
    fn into(self) -> Vec<i32> {
        vec![self.num_queries as i32, self.better_than_baseline as i32]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Solution {
    pub indexes: Vec<usize>,
}

impl Solution {
    pub fn new() -> Self {
        Self {
            indexes: Vec::new(),
        }
    }
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
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,
    pub d_query_vectors: CudaSlice<f32>,
    pub max_distance: f32,
}

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        difficulty: &Difficulty,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let mut rng = StdRng::from_seed(seed.clone());
        let better_than_baseline = difficulty.better_than_baseline;
        let max_distance = 11.0 - (better_than_baseline as f32) / 1000.0;
        let vector_dims = 250;
        let database_size = 100 * difficulty.num_queries;
        let avg_cluster_size: f32 = 700.0;
        let num_clusters: u32 = ((1.0 + rng.gen::<f32>() * 0.05)
            + database_size as f32 / avg_cluster_size)
            .round() as u32;
        let var: f32 = 0.2;
        let alpha: f32 = 0.05;
        let avg_cluster_weight = avg_cluster_size.ln() - var / 2.0;

        let generate_clusters_kernel = module.load_function("generate_clusters")?;
        let generate_vectors_kernel = module.load_function("generate_vectors")?;

        let block_size = MAX_THREADS_PER_BLOCK;

        let d_seed = stream.memcpy_stod(seed)?;
        let mut d_cluster_means =
            stream.alloc_zeros::<f32>((num_clusters * vector_dims) as usize)?;
        let mut d_cluster_weights = stream.alloc_zeros::<f32>(num_clusters as usize)?;
        let mut d_cluster_stds =
            stream.alloc_zeros::<f32>((num_clusters * vector_dims) as usize)?;

        unsafe {
            stream
                .launch_builder(&generate_clusters_kernel)
                .arg(&d_seed)
                .arg(&avg_cluster_weight)
                .arg(&vector_dims)
                .arg(&var)
                .arg(&alpha)
                .arg(&num_clusters)
                .arg(&mut d_cluster_means)
                .arg(&mut d_cluster_stds)
                .arg(&mut d_cluster_weights)
                .launch(LaunchConfig {
                    grid_dim: ((num_clusters + block_size - 1) / block_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        let cluster_weights = stream.memcpy_dtov(&d_cluster_weights)?;
        let total_weight: f32 = cluster_weights.iter().sum();
        let mut cluster_cum_prob = cluster_weights
            .iter()
            .scan(0.0, |state, &weight| {
                let ret = *state;
                *state += weight / total_weight;
                Some(ret)
            })
            .collect::<Vec<_>>();
        cluster_cum_prob.push(1.0);

        let d_cluster_cum_prob = stream.memcpy_stod(&cluster_cum_prob)?;
        let mut d_database_vectors =
            stream.alloc_zeros::<f32>((database_size * vector_dims) as usize)?;
        let mut d_query_vectors =
            stream.alloc_zeros::<f32>((difficulty.num_queries * vector_dims) as usize)?;

        unsafe {
            stream
                .launch_builder(&generate_vectors_kernel)
                .arg(&d_seed)
                .arg(&database_size)
                .arg(&difficulty.num_queries)
                .arg(&vector_dims)
                .arg(&num_clusters)
                .arg(&d_cluster_cum_prob)
                .arg(&d_cluster_means)
                .arg(&d_cluster_stds)
                .arg(&mut d_database_vectors)
                .arg(&mut d_query_vectors)
                .launch(LaunchConfig {
                    grid_dim: ((database_size + block_size - 1) / block_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        return Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            vector_dims,
            database_size,
            d_database_vectors,
            d_query_vectors,
            max_distance,
        });
    }

    pub fn verify_solution(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        prop: &cudaDeviceProp,
    ) -> Result<f32> {
        let avg_dist = calc_average_distance(
            self.difficulty.num_queries,
            self.vector_dims,
            self.database_size,
            &self.d_query_vectors,
            &self.d_database_vectors,
            &solution.indexes,
            module.clone(),
            stream.clone(),
            prop,
        )?;
        if avg_dist > self.max_distance {
            return Err(anyhow!(
                "Average query vector distance is '{}'. Max dist: '{}'",
                avg_dist,
                self.max_distance
            ));
        }
        Ok(avg_dist)
    }
}

pub fn calc_average_distance(
    num_queries: u32,
    vector_dims: u32,
    database_size: u32,
    d_query_vectors: &CudaSlice<f32>,
    d_database_vectors: &CudaSlice<f32>,
    indexes: &Vec<usize>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<f32> {
    if indexes.len() != num_queries as usize {
        return Err(anyhow!(
            "Invalid number of indexes. Expected: {}, Actual: {}",
            num_queries,
            indexes.len()
        ));
    }

    let calc_total_distance_kernel = module.load_function("calc_total_distance")?;

    let d_solution_indexes = stream.memcpy_stod(indexes)?;
    let mut d_total_distance = stream.alloc_zeros::<f32>(1)?;
    let mut errorflag = stream.alloc_zeros::<u32>(1)?;

    let threads_per_block = MAX_THREADS_PER_BLOCK;
    let blocks = (num_queries + threads_per_block - 1) / threads_per_block;

    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&calc_total_distance_kernel)
            .arg(&vector_dims)
            .arg(&database_size)
            .arg(&num_queries)
            .arg(d_query_vectors)
            .arg(d_database_vectors)
            .arg(&d_solution_indexes)
            .arg(&mut d_total_distance)
            .arg(&mut errorflag)
            .launch(cfg)?;
    }

    stream.synchronize()?;

    let total_distance = stream.memcpy_dtov(&d_total_distance)?[0];
    let error_flag = stream.memcpy_dtov(&errorflag)?[0];

    match error_flag {
        0 => {}
        1 => {
            return Err(anyhow!("Invalid index in solution"));
        }
        _ => {
            return Err(anyhow!("Unknown error code: {}", error_flag));
        }
    }

    let avg_dist = total_distance / num_queries as f32;
    Ok(avg_dist)
}
