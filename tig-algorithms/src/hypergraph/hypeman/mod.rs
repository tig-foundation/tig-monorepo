use anyhow::Result;
use cudarc::{
    driver::{
        safe::{LaunchConfig, CudaModule, CudaStream}, 
        PushKernelArg 
    },
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::{Challenge, Solution};

fn seed_to_u64(seed: &[u8; 32]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&seed[0..8]);
    u64::from_le_bytes(bytes)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    
    let num_nodes = challenge.num_nodes as u32;
    let num_parts = challenge.num_parts as u32;
    let max_part_size = challenge.max_part_size as u32;
    let seed = seed_to_u64(&challenge.seed);

    // 1. GRAPH DATA (Direct GPU Access)
    let d_edge_nodes = &challenge.d_hyperedge_nodes;
    let d_edge_ptrs = &challenge.d_hyperedge_offsets;
    let d_node_edges = &challenge.d_node_hyperedges;
    let d_node_ptrs = &challenge.d_node_offsets;

    // 2. ALLOCATE BUFFERS
    let mut d_partition = stream.alloc_zeros::<i32>(num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(num_parts as usize)?;
    
    let mut d_move_nodes = stream.alloc_zeros::<i32>(num_nodes as usize)?;
    let mut d_move_parts = stream.alloc_zeros::<i32>(num_nodes as usize)?;
    let mut d_move_gains = stream.alloc_zeros::<i32>(num_nodes as usize)?;
    let mut d_num_valid = stream.alloc_zeros::<i32>(1)?;
    
    let d_fallback_buffer = stream.alloc_zeros::<u64>((num_nodes * 3000) as usize)?;

    // 3. LOAD KERNELS
    let k_init = module.load_function("init_solution_kernel")?;
    let k_compute = module.load_function("compute_moves_annealing")?;
    let k_apply = module.load_function("apply_moves_kernel")?;

    let block_size = 256;
    let grid_size = (num_nodes + block_size - 1) / block_size;
    let launch_conf = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // 4. INITIALIZATION
    let mut launch_init = stream.launch_builder(&k_init);
    launch_init.arg(&num_nodes).arg(&num_parts).arg(&seed).arg(&mut d_partition).arg(&mut d_nodes_in_part);
    unsafe { launch_init.launch(launch_conf.clone()) }?;

    // 5. ANNEALING LOOP
    let rounds = 200; 
    for r in 0..rounds {
        // Calculate Temperature (Cooling Schedule)
        let progress = r as f32 / rounds as f32;
        let mut temperature = 5.0f32 * (1.0f32 - progress);
        if progress > 0.8 { temperature = 0.0; } // Final Greedy Phase

        stream.memcpy_htod(&[0], &mut d_num_valid)?;

        // Step A: Compute Moves with Temperature
        let round_idx = r as i32; 
        let mut l_comp = stream.launch_builder(&k_compute);
        l_comp.arg(&num_nodes).arg(&num_parts).arg(&max_part_size)
              .arg(d_node_edges).arg(d_node_ptrs)
              .arg(d_edge_nodes).arg(d_edge_ptrs)
              .arg(&d_partition).arg(&d_nodes_in_part)
              .arg(&mut d_move_nodes).arg(&mut d_move_parts).arg(&mut d_move_gains)
              .arg(&mut d_num_valid).arg(&round_idx).arg(&temperature)
              .arg(&d_fallback_buffer);
        unsafe { l_comp.launch(launch_conf.clone()) }?;

        // Step B: Apply Moves
        let mut l_apply = stream.launch_builder(&k_apply);
        l_apply.arg(&num_nodes).arg(&max_part_size)
               .arg(&mut d_move_nodes).arg(&mut d_move_parts)
               .arg(&mut d_partition).arg(&mut d_nodes_in_part);
        unsafe { l_apply.launch(launch_conf.clone()) }?;
    }

    // 6. SAVE RESULT
    stream.synchronize()?;
    
    let mut raw_part_host = vec![0i32; num_nodes as usize];
    stream.memcpy_dtoh(&d_partition, &mut raw_part_host)?;
    
    let solution_vec: Vec<u32> = raw_part_host.iter().map(|&x| x as u32).collect();

    let solution = Solution {
        partition: solution_vec,
    };
    save_solution(&solution)?;

    Ok(())
}

pub fn help() {
    println!("No help information available.");
}
