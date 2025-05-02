use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
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

impl TryFrom<Map<String, Value>> for Solution {
    type Error = serde_json::Error;

    fn try_from(v: Map<String, Value>) -> Result<Self, Self::Error> {
        from_value(Value::Object(v))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub vector_dims: usize,
    pub database_size: usize,
    pub max_distance: f32,
}

impl Challenge {
    pub fn generate_instance(
        seed: [u8; 32],
        difficulty: &Difficulty,
        _module: Arc<CudaModule>,
        _stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let better_than_baseline = difficulty.better_than_baseline;
        let max_distance = 6.0 - (better_than_baseline as f32) / 1000.0;
        return Ok(Self {
            seed: seed.clone(),
            difficulty: difficulty.clone(),
            vector_dims: 250,
            database_size: 1_000_000,
            max_distance,
        });
    }

    pub fn verify_solution(
        &self,
        solution: &Solution,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        prop: &cudaDeviceProp,
    ) -> Result<()> {
        if solution.indexes.len() != self.difficulty.num_queries as usize {
            return Err(anyhow!(
                "Invalid number of indexes. Expected: {}, Actual: {}",
                self.difficulty.num_queries,
                solution.indexes.len()
            ));
        }

        let calc_total_distance_kernel = module.load_function("calc_total_distance")?;

        let num_queries = self.difficulty.num_queries as usize;
        let d_seed = stream.memcpy_stod(&self.seed.to_vec())?;
        let d_solution_indexes = stream.memcpy_stod(&solution.indexes)?;
        let mut d_query_vectors = stream.alloc_zeros::<f32>(num_queries * self.vector_dims)?;
        let mut d_database_vectors = stream.alloc_zeros::<f32>(num_queries * self.vector_dims)?;
        let mut d_total_distance = stream.alloc_zeros::<f32>(1)?;
        let mut errorflag = stream.alloc_zeros::<u32>(1)?;

        let threads_per_block = prop.maxThreadsPerBlock as u32;
        let blocks =
            (self.difficulty.num_queries as u32 + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&calc_total_distance_kernel);
        unsafe {
            builder
                .arg(&d_seed)
                .arg(&(self.vector_dims as u32))
                .arg(&(self.database_size as u32))
                .arg(&(self.difficulty.num_queries as u32))
                .arg(&mut d_query_vectors)
                .arg(&mut d_database_vectors)
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

        let avg_dist = total_distance / self.difficulty.num_queries as f32;
        if avg_dist > self.max_distance {
            return Err(anyhow!(
                "Average query vector distance is '{}'. Max dist: '{}'",
                avg_dist,
                self.max_distance
            ));
        }

        Ok(())
    }
}
