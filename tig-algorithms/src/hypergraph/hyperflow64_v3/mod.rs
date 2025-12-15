use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use std::sync::Arc;
use std::time::Instant;
use serde_json::{Map, Value};
use tig_challenges::hypergraph::*;

pub struct HyperFlow64LabelProp {
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: cudaDeviceProp,
    hyperparameters: Option<Map<String, Value>>,
}

// Structure to hold a potential FM move
#[derive(Clone, Copy, Debug)]
struct FMMove {
    node: i32,
    from_part: i32,
    to_part: i32,
    gain: i32,
}

impl HyperFlow64LabelProp {
    pub fn new(
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        prop: cudaDeviceProp,
        hyperparameters: Option<Map<String, Value>>,
    ) -> Self {
        Self {
            module,
            stream,
            prop,
            hyperparameters,
        }
    }

    pub fn solve(&self, challenge: &Challenge) -> anyhow::Result<Solution> {
        // Copy hyperedge structure for connectivity calculation
        let h_hyperedge_offsets: Vec<i32> = self.stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
        let h_hyperedge_nodes: Vec<i32> = self.stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
        
        let calc_connectivity = |partition: &[i32]| -> u32 {
            let mut total = 0u32;
            for hedge_id in 0..challenge.num_hyperedges as usize {
                let mut parts_bitmap = 0u64;
                let start = h_hyperedge_offsets[hedge_id] as usize;
                let end = h_hyperedge_offsets[hedge_id + 1] as usize;
                
                for &node in &h_hyperedge_nodes[start..end] {
                    let part = partition[node as usize];
                    parts_bitmap |= 1u64 << part;
                }
                
                total += parts_bitmap.count_ones() - 1;
            }
            total
        };
        
        let solve_start = Instant::now();
        println!("\n{}", "=".repeat(80));
        println!("HyperFlow64 - Label Propagation Based Partitioning");
        println!("{}", "=".repeat(80));
        println!("Challenge: {} nodes, {} hyperedges, {} parts",
                  challenge.num_nodes, challenge.num_hyperedges, challenge.num_parts);
        println!("Max part size: {}", challenge.max_part_size);
        
        let block_size = 256u32;
        let num_blocks = (challenge.num_nodes as u32 + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Load kernels
        let kernel_start = Instant::now();
        let init_labels_kernel = self.module.load_function("initialize_labels")?;
        let label_prop_kernel = self.module.load_function("label_propagation_iteration")?;
        let compute_prefs_kernel = self.module.load_function("compute_label_preferences")?;
        let execute_assign_kernel = self.module.load_function("execute_label_assignments")?;
        let compute_moves_kernel = self.module.load_function("compute_fm_moves")?;
        let apply_single_move_kernel = self.module.load_function("apply_single_move")?;
        println!("[Setup] Kernels loaded: {:.2}ms", kernel_start.elapsed().as_secs_f64() * 1000.0);

        // Allocate memory
        let mem_start = Instant::now();
        let mut d_labels = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        let mut d_labels_temp = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        let mut d_changed = self.stream.alloc_zeros::<i32>(1)?;
        let mut d_partition = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        let mut d_nodes_in_part = self.stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;
        let mut d_pref_nodes = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        let mut d_pref_parts = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        let mut d_pref_priorities = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        
        // FM move arrays
        let mut d_move_gains = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        let mut d_move_targets = self.stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
        
        println!("[Setup] Memory allocated: {:.2}ms", mem_start.elapsed().as_secs_f64() * 1000.0);

        let lp_iterations = self.get_hyperparam("lp_iterations", 5);
        let fm_rounds = self.get_hyperparam("fm_rounds", 100);
        let epsilon = self.get_hyperparam("epsilon", 0.03f32);
        println!("[Setup] Hyperparameters: lp_iterations={}, fm_rounds={}, epsilon={:.3}",
                  lp_iterations, fm_rounds, epsilon);
        println!("{}", "-".repeat(80));

        let single_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            // Phase 1: Initialize labels (each node = unique label)
            let phase_start = Instant::now();
            self.stream.launch_builder(&init_labels_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&mut d_labels)
                .launch(cfg)?;
            self.stream.synchronize()?;
            println!("[Phase 1] Initialize labels:       {:.2}ms", phase_start.elapsed().as_secs_f64() * 1000.0);

            // Phase 2: Label Propagation
            let lp_start = Instant::now();
            
            for iter in 0..lp_iterations {
                let iter_start = Instant::now();
                
                // Reset changed counter
                let zero = vec![0i32];
                self.stream.memcpy_htod(&zero, &mut d_changed)?;
                
                // Run label propagation iteration
                self.stream.launch_builder(&label_prop_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_labels)
                    .arg(&mut d_labels_temp)
                    .arg(&mut d_changed)
                    .launch(cfg)?;
                self.stream.synchronize()?;
                
                // Swap buffers using tuple swap (TIG compliant - no std::mem::swap)
                (d_labels, d_labels_temp) = (d_labels_temp, d_labels);
                
                // Check convergence
                let changed: Vec<i32> = self.stream.memcpy_dtov(&d_changed)?;
                println!("           LP Iteration {}/{}: {:.2}ms, changed: {}",
                          iter + 1, lp_iterations, iter_start.elapsed().as_secs_f64() * 1000.0, changed[0]);
                
                // Early stopping if converged
                if changed[0] == 0 {
                    println!("           Converged early!");
                    break;
                }
            }
            println!("[Phase 2] Label Propagation Total: {:.2}ms", lp_start.elapsed().as_secs_f64() * 1000.0);

            // Phase 3: Compute preferences based on labels
            let phase_start = Instant::now();
            self.stream.launch_builder(&compute_prefs_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&d_labels)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&mut d_pref_nodes)
                .arg(&mut d_pref_parts)
                .arg(&mut d_pref_priorities)
                .launch(cfg)?;
            self.stream.synchronize()?;
            println!("[Phase 3] Compute Preferences:     {:.2}ms", phase_start.elapsed().as_secs_f64() * 1000.0);
            
            // Phase 4: Sort and execute assignments
            let phase_start = Instant::now();
            let pref_nodes: Vec<i32> = self.stream.memcpy_dtov(&d_pref_nodes)?;
            let pref_parts: Vec<i32> = self.stream.memcpy_dtov(&d_pref_parts)?;
            let pref_priorities: Vec<i32> = self.stream.memcpy_dtov(&d_pref_priorities)?;
            
            let mut indices: Vec<usize> = (0..challenge.num_nodes as usize).collect();
            indices.sort_by(|&a, &b| pref_priorities[b].cmp(&pref_priorities[a]));
            
            let sorted_nodes: Vec<i32> = indices.iter().map(|&i| pref_nodes[i]).collect();
            let sorted_parts: Vec<i32> = indices.iter().map(|&i| pref_parts[i]).collect();
            
            let d_sorted_nodes = self.stream.memcpy_stod(&sorted_nodes)?;
            let d_sorted_parts = self.stream.memcpy_stod(&sorted_parts)?;
            
            self.stream.launch_builder(&execute_assign_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&d_sorted_nodes)
                .arg(&d_sorted_parts)
                .arg(&mut d_partition)
                .arg(&mut d_nodes_in_part)
                .launch(single_cfg)?;
            self.stream.synchronize()?;
            
            let partition_check: Vec<i32> = self.stream.memcpy_dtov(&d_partition)?;
            let nodes_in_part_check: Vec<i32> = self.stream.memcpy_dtov(&d_nodes_in_part)?;
            
            let connectivity = calc_connectivity(&partition_check);
            println!("[Phase 4] Initial Assignment:      {:.2}ms", phase_start.elapsed().as_secs_f64() * 1000.0);
            println!("          Initial Connectivity: {}", connectivity);

            // Phase 5: Serial FM Refinement
            let fm_start = Instant::now();
            let mut best_connectivity = connectivity;
            let mut best_partition = partition_check.clone();
            let mut stale_rounds = 0;
            let max_stale_rounds = 10;
            
            println!("[Phase 5] Starting Serial FM Refinement ({} rounds)", fm_rounds);
            
            for round in 0..fm_rounds {
                let round_start = Instant::now();
                
                // GPU: Compute all potential moves in parallel
                self.stream.launch_builder(&compute_moves_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&mut d_move_gains)
                    .arg(&mut d_move_targets)
                    .launch(cfg)?;
                self.stream.synchronize()?;
                
                // CPU: Copy moves to host
                let move_gains: Vec<i32> = self.stream.memcpy_dtov(&d_move_gains)?;
                let move_targets: Vec<i32> = self.stream.memcpy_dtov(&d_move_targets)?;
                let mut partition_host: Vec<i32> = self.stream.memcpy_dtov(&d_partition)?;
                let mut nodes_in_part_host: Vec<i32> = self.stream.memcpy_dtov(&d_nodes_in_part)?;
                
                // CPU: Build list of valid moves
                let mut valid_moves: Vec<FMMove> = Vec::new();
                for node in 0..challenge.num_nodes as usize {
                    let gain = move_gains[node];
                    let target = move_targets[node];
                    
                    if target == 255 || gain <= 0 {
                        continue;
                    }
                    
                    let from_part = partition_host[node];
                    let to_part = target as i32;
                    
                    // Validate move
                    if from_part == to_part || to_part < 0 || to_part >= challenge.num_parts as i32 {
                        continue;
                    }
                    
                    // Check balance constraints
                    if nodes_in_part_host[from_part as usize] <= 1 {
                        continue;  // Don't empty a partition
                    }
                    if nodes_in_part_host[to_part as usize] >= challenge.max_part_size as i32 {
                        continue;  // Don't overfill a partition
                    }
                    
                    valid_moves.push(FMMove {
                        node: node as i32,
                        from_part,
                        to_part,
                        gain,
                    });
                }
                
                if valid_moves.is_empty() {
                    println!("           Round {}/{}: {:.2}ms, no valid moves - converged",
                              round + 1, fm_rounds, round_start.elapsed().as_secs_f64() * 1000.0);
                    break;
                }
                
                // CPU: Sort by gain (descending)
                valid_moves.sort_by(|a, b| b.gain.cmp(&a.gain));
                
                // CPU: Execute moves serially (greedy selection)
                let mut moves_executed = 0;
                let mut total_gain = 0;
                let mut moved_nodes = vec![false; challenge.num_nodes as usize];
                
                for fm_move in &valid_moves {
                    let node = fm_move.node as usize;
                    
                    // Skip if node already moved this round
                    if moved_nodes[node] {
                        continue;
                    }
                    
                    let from = fm_move.from_part as usize;
                    let to = fm_move.to_part as usize;
                    
                    // Re-check constraints (they may have changed)
                    if nodes_in_part_host[from] <= 1 || nodes_in_part_host[to] >= challenge.max_part_size as i32 {
                        continue;
                    }
                    
                    // Execute move on CPU
                    partition_host[node] = fm_move.to_part;
                    nodes_in_part_host[from] -= 1;
                    nodes_in_part_host[to] += 1;
                    moved_nodes[node] = true;
                    
                    moves_executed += 1;
                    total_gain += fm_move.gain;
                    
                    // Limit moves per round to prevent instability
                    if moves_executed >= 100 {
                        break;
                    }
                }
                
                // Copy updated partition back to GPU
                let d_partition_new = self.stream.memcpy_stod(&partition_host)?;
                let d_nodes_in_part_new = self.stream.memcpy_stod(&nodes_in_part_host)?;
                d_partition = d_partition_new;
                d_nodes_in_part = d_nodes_in_part_new;
                
                // Calculate new connectivity
                let new_connectivity = calc_connectivity(&partition_host);
                let improvement = connectivity as i32 - new_connectivity as i32;
                
                println!("           Round {}/{}: {:.2}ms, moves: {}, gain: {}, conn: {} → {} (Δ: {})",
                          round + 1, fm_rounds, round_start.elapsed().as_secs_f64() * 1000.0,
                         moves_executed, total_gain, best_connectivity, new_connectivity, improvement);
                
                // Track best solution
                if new_connectivity < best_connectivity {
                    best_connectivity = new_connectivity;
                    best_partition = partition_host.clone();
                    stale_rounds = 0;
                } else {
                    stale_rounds += 1;
                }
                
                // Early termination if no improvement
                if stale_rounds >= max_stale_rounds {
                    println!("           No improvement for {} rounds - stopping FM", max_stale_rounds);
                    break;
                }
            }
            
            println!("[Phase 5] FM Refinement Total:     {:.2}ms", fm_start.elapsed().as_secs_f64() * 1000.0);
            println!("          Final Connectivity: {} (improved by {})",
                      best_connectivity, connectivity as i32 - best_connectivity as i32);
            
            // Use best partition found
            let d_partition_final = self.stream.memcpy_stod(&best_partition)?;
            d_partition = d_partition_final;
        }

        self.stream.synchronize()?;

        let copy_start = Instant::now();
        let partition: Vec<i32> = self.stream.memcpy_dtov(&d_partition)?;
        let partition_u32: Vec<u32> = partition.into_iter().map(|x| x as u32).collect();
        println!("[Finish]  Copy to host:            {:.2}ms", copy_start.elapsed().as_secs_f64() * 1000.0);
        
        let total_time = solve_start.elapsed().as_secs_f64() * 1000.0;
        println!("{}", "=".repeat(80));
        println!("TOTAL EXECUTION TIME: {:.2}ms", total_time);
        println!("{}", "=".repeat(80));

        Ok(Solution { partition: partition_u32 })
    }

    fn get_hyperparam<T: std::str::FromStr>(&self, key: &str, default: T) -> T
    where
        T: Copy,
    {
        if let Some(ref hp) = self.hyperparameters {
            if let Some(val) = hp.get(key) {
                if let Some(s) = val.as_str() {
                    if let Ok(parsed) = s.parse() {
                        return parsed;
                    }
                }
            }
        }
        default
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let solver = HyperFlow64LabelProp::new(module, stream, *prop, hyperparameters.clone());
    let solution = solver.solve(challenge)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn help() {
    println!("HyperFlow64: Label Propagation based hypergraph partitioning");
    println!("Uses iterative label propagation to cluster nodes with high co-occurrence");
    println!("Hyperparameters:");
    println!("  - lp_iterations: Number of label propagation iterations (default: 5)");
    println!("  - fm_rounds: Number of FM refinement rounds (default: 100)");
    println!("  - epsilon: Balance tolerance (default: 0.03)");
}
