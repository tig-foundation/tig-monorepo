use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use serde_json::{Map, Value};
use tig_challenges::vector_search::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let vector_dims = challenge.vector_dims as i32;
    let database_size = challenge.database_size as i32;
    let num_queries = challenge.num_queries as i32;
    
    let block_size = 128;
    let num_clusters = if num_queries <= 6000 {
        2
    } else if num_queries < 9000 {
        4
    } else if num_queries < 10000 {
        6    
    } else if num_queries < 11000  {
        10
    }  else if num_queries < 12000  {
        12
    }  else if num_queries < 14000  {
        14
    } else {
        14
    };
    
    let deterministic_clustering = module.load_function("deterministic_clustering")?;
    let cluster_search = module.load_function("cluster_search")?;
    
    let mut d_cluster_centers = stream.alloc_zeros::<f32>((num_clusters * vector_dims) as usize)?;
    let mut d_cluster_assignments = stream.alloc_zeros::<i32>(database_size as usize)?;
    let mut d_cluster_sizes = stream.alloc_zeros::<i32>(num_clusters as usize)?;
    
    let cluster_config = LaunchConfig {
        grid_dim: (num_clusters as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (vector_dims * 4) as u32,
    };
    
    unsafe {
        stream.launch_builder(&deterministic_clustering)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_cluster_centers)
            .arg(&mut d_cluster_assignments)
            .arg(&mut d_cluster_sizes)
            .arg(&database_size)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .arg(&num_queries)
            .launch(cluster_config)?;
    }
    stream.synchronize()?;
    
    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    
    let search_config = if num_queries <= 4000 {
        LaunchConfig {
            grid_dim: (num_queries as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (num_queries as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (num_clusters * 8) as u32,
        }
    };
    
    unsafe {
        stream.launch_builder(&cluster_search)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&d_cluster_centers)
            .arg(&d_cluster_assignments)
            .arg(&d_cluster_sizes)
            .arg(&mut d_results)
            .arg(&num_queries)
            .arg(&database_size)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .launch(search_config)?;
    }
    stream.synchronize()?;

    let indices = stream.memcpy_dtov(&d_results)?;
    let indexes = indices.iter().map(|&idx| idx as usize).collect();
    
    let _ = save_solution(&Solution { indexes });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
