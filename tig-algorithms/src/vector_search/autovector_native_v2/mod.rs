use anyhow::Result;
use cudarc::driver::{safe::{LaunchConfig, CudaModule, CudaStream}, CudaSlice, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    pub num_centroids: u32,
    pub search_clusters: u32,
    pub multi_probe_boost: u32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self { 
            num_centroids: 4,
            search_clusters: 2,
            multi_probe_boost: 2,
        }
    }
}

fn calculate_adaptive_clusters(database_size: u32, vector_dims: u32, num_queries: u32) -> u32 {
    let base_clusters = (database_size as f32).sqrt() as u32;
    let dim_factor = 1.0 + (vector_dims as f32 / 100.0).ln();
    let query_factor = (num_queries as f32 / 10.0).sqrt().max(1.0);
    
    let adaptive_count = (base_clusters as f32 * dim_factor / query_factor) as u32;
    adaptive_count.clamp(8, 64)
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn build_probe_list_adaptive(
    query: &[f32],
    centroids: &[f32],
    num_centroids: u32,
    dim: u32,
    total_probes: u32,
) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = (0..num_centroids)
        .map(|i| {
            let start = (i * dim) as usize;
            let end = start + dim as usize;
            let centroid = &centroids[start..end];
            let dist = l2_squared(query, centroid);
            (i, dist)
        })
        .collect();
    
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    distances.into_iter()
        .take(total_probes.min(num_centroids) as usize)
        .map(|(i, _)| i)
        .collect()
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    let hps: Hyperparameters = hyperparameters.as_ref()
        .and_then(|m| serde_json::from_value(serde_json::Value::Object(m.clone())).ok())
        .unwrap_or_default();
    
    let dim = challenge.vector_dims as u32;
    let num_vectors = challenge.database_size as u32;
    let num_queries = challenge.num_queries as u32;

    let adaptive_centroids = calculate_adaptive_clusters(num_vectors, dim, num_queries);
    let num_centroids = hps.num_centroids.min(adaptive_centroids);
    let total_search_clusters = (hps.search_clusters + hps.multi_probe_boost).min(num_centroids);
    
    let use_adaptive_probes = total_search_clusters < num_centroids;

    // Select centroids on GPU
    let d_centroids = stream.alloc_zeros::<f32>((num_centroids * dim) as usize)?;
    let select_func = module.load_function("select_centroids_strided")?;
    
    let total_threads = num_centroids * dim;
    unsafe {
        stream.launch_builder(&select_func)
            .arg(&challenge.d_database_vectors)
            .arg(&d_centroids)
            .arg(&num_vectors)
            .arg(&num_centroids)
            .arg(&dim)
            .launch(LaunchConfig {
                grid_dim: ((total_threads + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0
            })?;
    }
    
    stream.synchronize()?;
    
    let centroids = if use_adaptive_probes {
        stream.memcpy_dtov(&d_centroids)?
    } else {
        Vec::new()
    };
    
    // Assign vectors to nearest centroids
    let d_assignments = stream.alloc_zeros::<i32>(num_vectors as usize)?;
    let assign_func = module.load_function("assign_to_nearest_centroid")?;
    
    unsafe {
        stream.launch_builder(&assign_func)
            .arg(&challenge.d_database_vectors)
            .arg(&d_centroids)
            .arg(&d_assignments)
            .arg(&num_vectors)
            .arg(&num_centroids)
            .arg(&dim)
            .launch(LaunchConfig { 
                grid_dim: ((num_vectors + 255) / 256, 1, 1), 
                block_dim: (256, 1, 1), 
                shared_mem_bytes: 0 
            })?;
    }
    
    stream.synchronize()?;
    
    // GPU-based cluster building
    let d_cluster_sizes = stream.alloc_zeros::<i32>(num_centroids as usize)?;
    let count_func = module.load_function("count_cluster_sizes")?;
    
    unsafe {
        stream.launch_builder(&count_func)
            .arg(&d_assignments)
            .arg(&d_cluster_sizes)
            .arg(&num_vectors)
            .launch(LaunchConfig {
                grid_dim: ((num_vectors + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0
            })?;
    }
    
    stream.synchronize()?;
    
    let cluster_sizes: Vec<i32> = stream.memcpy_dtov(&d_cluster_sizes)?;
    
    let mut cluster_offsets: Vec<i32> = vec![0];
    let mut total = 0i32;
    for &size in &cluster_sizes {
        total += size;
        cluster_offsets.push(total);
    }
    
    let d_cluster_indices = stream.alloc_zeros::<i32>(num_vectors as usize)?;
    let d_cluster_offsets = stream.memcpy_stod(&cluster_offsets)?;
    let d_cluster_positions = stream.alloc_zeros::<i32>(num_centroids as usize)?;
    
    let build_func = module.load_function("build_cluster_indices")?;
    unsafe {
        stream.launch_builder(&build_func)
            .arg(&d_assignments)
            .arg(&d_cluster_offsets)
            .arg(&d_cluster_indices)
            .arg(&d_cluster_positions)
            .arg(&num_vectors)
            .launch(LaunchConfig {
                grid_dim: ((num_vectors + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0
            })?;
    }
    
    stream.synchronize()?;
    
    let d_cluster_sizes = stream.memcpy_stod(&cluster_sizes)?;
    
    let h_queries = if use_adaptive_probes {
        stream.memcpy_dtov(&challenge.d_query_vectors)?
    } else {
        Vec::new()
    };
    
    let cluster_search = module.load_function("search_coalesced_multiquery")?;
    let mut d_results = stream.alloc_zeros::<i32>(num_queries as usize)?;
    
    let shared_mem_centroids = (num_centroids * dim * 4) as u32;
    let max_shared = prop.sharedMemPerBlock as u32;
    let use_shared_mem = shared_mem_centroids < (max_shared / 2);
    let use_shared_flag = if use_shared_mem { 1i32 } else { 0i32 };
    
    let batch_size = 256;
    
    for query_batch_start in (0..num_queries).step_by(batch_size) {
        let query_batch_end = (query_batch_start + batch_size as u32).min(num_queries);
        let batch_count = query_batch_end - query_batch_start;
        
        let all_probe_lists = if use_adaptive_probes {
            let mut lists = Vec::new();
            for q in query_batch_start..query_batch_end {
                let query_start = (q * dim) as usize;
                let query_end = query_start + dim as usize;
                let query = &h_queries[query_start..query_end];
                
                let probe_list = build_probe_list_adaptive(
                    query,
                    &centroids,
                    num_centroids,
                    dim,
                    total_search_clusters
                );
                lists.extend(probe_list);
            }
            lists
        } else {
            let fixed_list: Vec<u32> = (0..num_centroids).collect();
            fixed_list.repeat(batch_count as usize)
        };
        
        let d_probe_lists = stream.memcpy_stod(&all_probe_lists)?;
        
        let search_config = LaunchConfig {
            grid_dim: (batch_count, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: if use_shared_mem { shared_mem_centroids } else { 0 },
        };

        unsafe {
            stream.launch_builder(&cluster_search)
                .arg(&challenge.d_query_vectors)
                .arg(&challenge.d_database_vectors)
                .arg(&d_centroids)
                .arg(&d_cluster_indices)
                .arg(&d_cluster_sizes)
                .arg(&d_cluster_offsets)
                .arg(&d_probe_lists)
                .arg(&mut d_results)
                .arg(&query_batch_start)
                .arg(&batch_count)
                .arg(&num_vectors)
                .arg(&dim)
                .arg(&num_centroids)
                .arg(&total_search_clusters)
                .arg(&use_shared_flag)
                .launch(search_config)?;
        }
        
        stream.synchronize()?;
        
        // SIGMA UPDATE: Save partial solution after each batch
        // If we run out of fuel, this ensures we have results for completed queries
        let partial_indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
        let partial_indexes = partial_indices.iter().map(|&idx| idx as usize).collect();
        save_solution(&Solution { indexes: partial_indexes })?;
    }
    
    // Final synchronization and solution save
    stream.synchronize()?;
    
    let indices: Vec<i32> = stream.memcpy_dtov(&d_results)?;
    let indexes = indices.iter().map(|&idx| idx as usize).collect();
    
    save_solution(&Solution { indexes })?;
    
    Ok(())
}

pub fn help() {
    println!("No help information provided.");
}
