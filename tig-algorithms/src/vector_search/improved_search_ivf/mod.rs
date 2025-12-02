use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, DeviceSlice, PushKernelArg},
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

    let block_size = 256u32;
    let num_clusters = (challenge.database_size / 25000).max(2).min(16) as i32;

    let initialize_centers_kernel = module.load_function("initialize_centers")?;
    let assign_clusters_kernel = module.load_function("assign_clusters_and_count_sizes")?;
    let build_index_kernel = module.load_function("build_inverted_index")?;
    let search_kernel = module.load_function("search_ivf")?;

    let mut d_cluster_centers = stream.alloc_zeros::<f32>((num_clusters * vector_dims) as usize)?;
    let mut d_cluster_assignments = stream.alloc_zeros::<i32>(database_size as usize)?;
    let mut d_cluster_sizes = stream.alloc_zeros::<i32>(num_clusters as usize)?;
    let mut d_cluster_offsets = stream.alloc_zeros::<i32>(num_clusters as usize)?;
    let mut d_temp_cluster_heads = stream.alloc_zeros::<i32>(num_clusters as usize)?;
    let mut d_inverted_indices = stream.alloc_zeros::<i32>(database_size as usize)?;
    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;

    let init_cfg = LaunchConfig {
        grid_dim: ((num_clusters as u32 + block_size - 1) / block_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&initialize_centers_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_cluster_centers)
            .arg(&database_size)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .launch(init_cfg)?;
    }

    let grid_dim_main = (database_size as u32 + block_size - 1) / block_size;
    let assign_cfg = LaunchConfig {
        grid_dim: (grid_dim_main, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (num_clusters * vector_dims * 4) as u32,
    };
    unsafe {
        stream
            .launch_builder(&assign_clusters_kernel)
            .arg(&challenge.d_database_vectors)
            .arg(&d_cluster_centers)
            .arg(&mut d_cluster_assignments)
            .arg(&mut d_cluster_sizes)
            .arg(&database_size)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .launch(assign_cfg)?;
    }

    let h_cluster_sizes = stream.memcpy_dtov(&d_cluster_sizes)?;
    let mut h_cluster_offsets = vec![0i32; num_clusters as usize];
    let mut current_offset = 0;
    for i in 0..num_clusters as usize {
        h_cluster_offsets[i] = current_offset;
        current_offset += h_cluster_sizes[i];
    }
    stream.memcpy_htod(&h_cluster_offsets, &mut d_cluster_offsets)?;
    stream.memcpy_htod(&h_cluster_offsets, &mut d_temp_cluster_heads)?;

    let build_index_cfg = LaunchConfig {
        grid_dim: (grid_dim_main, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&build_index_kernel)
            .arg(&d_cluster_assignments)
            .arg(&d_cluster_offsets)
            .arg(&mut d_inverted_indices)
            .arg(&mut d_temp_cluster_heads)
            .arg(&database_size)
            .launch(build_index_cfg)?;
    }

    let grid_dim_search = (num_queries as u32 + block_size - 1) / block_size;
    let search_cfg = LaunchConfig {
        grid_dim: (grid_dim_search, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (num_clusters * vector_dims * 4) as u32,
    };
    unsafe {
        stream
            .launch_builder(&search_kernel)
            .arg(&challenge.d_query_vectors)
            .arg(&challenge.d_database_vectors)
            .arg(&d_cluster_centers)
            .arg(&d_inverted_indices)
            .arg(&d_cluster_offsets)
            .arg(&d_cluster_sizes)
            .arg(&mut d_results)
            .arg(&num_queries)
            .arg(&vector_dims)
            .arg(&num_clusters)
            .arg(&database_size)
            .launch(search_cfg)?;
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
