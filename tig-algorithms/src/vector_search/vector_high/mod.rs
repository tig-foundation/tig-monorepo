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

    let block_size: u32 = 128;
    fn calculate_optimal_clusters(num_queries: i32, database_size: i32, vector_dims: i32) -> i32 {
        let base_clusters = if num_queries <= 3000 {
            3
        } else if num_queries <= 5000 {
            4
        } else if num_queries <= 7000 {
            5
        } else if num_queries <= 9000 {
            6
        } else if num_queries <= 11000 {
            7
        } else if num_queries <= 15000 {
            8
        } else if num_queries <= 20000 {
            9
        } else {            
            ((database_size as f32).sqrt() / 1000.0).max(8.0).min(12.0) as i32
        };
        
        let memory_factor = if vector_dims > 1000 { 0.8 } else { 1.0 };
        ((base_clusters as f32 * memory_factor) as i32).max(3).min(12)
    }
    
    let num_clusters = calculate_optimal_clusters(num_queries, database_size, vector_dims);

    let deterministic_clustering = module.load_function("deterministic_clustering")?;
    let assign_clusters = module.load_function("assign_clusters")?;
    let build_cluster_index = module.load_function("build_cluster_index")?;
    let cluster_search = module.load_function("cluster_search")?;
    let exclusive_scan_sizes = module.load_function("exclusive_scan_sizes")?;

    let mut d_cluster_centers = stream.alloc_zeros::<f32>((num_clusters * vector_dims) as usize)?;
    let mut d_cluster_assignments = stream.alloc_zeros::<i32>(database_size as usize)?;
    let mut d_cluster_sizes = stream.alloc_zeros::<i32>(num_clusters as usize)?;

    let cluster_config = LaunchConfig {
        grid_dim: (num_clusters as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: ((vector_dims as u32) * 4) as u32,
    };

    unsafe {
        stream
            .launch_builder(&deterministic_clustering)
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

    let assign_threads: u32 = if database_size > 80000 { 
        512 
    } else if database_size > 40000 { 
        384 
    } else { 
        256 
    };
    let assign_blocks: u32 = ((database_size as u32) + assign_threads - 1) / assign_threads;
    let assign_config = LaunchConfig {
        grid_dim: (assign_blocks, 1, 1),
        block_dim: (assign_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&assign_clusters)
            .arg(&challenge.d_database_vectors)
            .arg(&d_cluster_centers)
            .arg(&mut d_cluster_assignments)
            .arg(&mut d_cluster_sizes)
            .arg(&database_size)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .arg(&num_queries)
            .launch(assign_config)?;
    }
    stream.synchronize()?;

    let mut d_cluster_offsets = stream.alloc_zeros::<i32>(num_clusters as usize)?;
    let mut d_write_offsets = stream.alloc_zeros::<i32>(num_clusters as usize)?;
    let scan_config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&exclusive_scan_sizes)
            .arg(&d_cluster_sizes)
            .arg(&mut d_cluster_offsets)
            .arg(&mut d_write_offsets)
            .arg(&num_clusters)
            .launch(scan_config)?;
    }
    stream.synchronize()?;

    let mut d_cluster_indices = stream.alloc_zeros::<i32>(database_size as usize)?;
    let db_u32 = database_size as u32;
    let fill_blocks: u32 = (db_u32 + block_size - 1) / block_size;
    let fill_config = LaunchConfig {
        grid_dim: (fill_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&build_cluster_index)
            .arg(&d_cluster_assignments)
            .arg(&mut d_write_offsets)
            .arg(&mut d_cluster_indices)
            .arg(&database_size)
            .launch(fill_config)?;
    }
    stream.synchronize()?;

    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    
    let search_config = if num_queries <= 3000 {
        LaunchConfig {
            grid_dim: (num_queries as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        let threads_per_block = if num_queries > 10000 { 
            256 
        } else if num_queries > 6000 { 
            192 
        } else { 
            128 
        };
        let blocks = ((num_queries as u32) + threads_per_block - 1) / threads_per_block;
        LaunchConfig {
            grid_dim: (blocks.min(2048), 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    unsafe {
        stream
            .launch_builder(&cluster_search)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&d_cluster_centers)
            .arg(&d_cluster_assignments)
            .arg(&d_cluster_sizes)
            .arg(&d_cluster_indices)
            .arg(&d_cluster_offsets)
            .arg(&mut d_results)
            .arg(&num_queries)
            .arg(&database_size)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .launch(search_config)?;
    }
    stream.synchronize()?;

    let indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes = indices.iter().map(|&idx| idx as usize).collect();

    let _ = save_solution(&Solution { indexes });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
