#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// GPU Label Propagation for Initial Partition
// ============================================================================

extern "C" __global__ void initialize_labels(
    int num_nodes,
    int* labels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    // Each node starts with its own ID as label
    labels[tid] = tid;
}

extern "C" __global__ void label_propagation_iteration(
    int num_nodes,
    const int* node_hyperedges,
    const int* node_offsets,
    const int* hyperedge_nodes,
    const int* hyperedge_offsets,
    const int* labels_in,
    int* labels_out,
    int* changed
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    
    int current_label = labels_in[node];
    
    // Count label frequencies from neighbors in hyperedges
    int label_count[256];  // Assume max 256 unique labels in neighborhood
    int unique_labels[256];
    int num_unique = 0;
    
    for (int i = 0; i < 256; i++) {
        label_count[i] = 0;
    }
    
    // Get all hyperedges containing this node
    int hedge_start = node_offsets[node];
    int hedge_end = node_offsets[node + 1];
    
    // For each hyperedge, collect labels of other nodes
    for (int he_idx = hedge_start; he_idx < hedge_end; he_idx++) {
        int hedge = node_hyperedges[he_idx];
        int nodes_start = hyperedge_offsets[hedge];
        int nodes_end = hyperedge_offsets[hedge + 1];
        
        // Get labels of all nodes in this hyperedge
        for (int n_idx = nodes_start; n_idx < nodes_end; n_idx++) {
            int neighbor = hyperedge_nodes[n_idx];
            if (neighbor == node) continue;  // Skip self
            
            int neighbor_label = labels_in[neighbor];
            
            // Find or add this label
            bool found = false;
            for (int i = 0; i < num_unique; i++) {
                if (unique_labels[i] == neighbor_label) {
                    label_count[i]++;
                    found = true;
                    break;
                }
            }
            
            if (!found && num_unique < 256) {
                unique_labels[num_unique] = neighbor_label;
                label_count[num_unique] = 1;
                num_unique++;
            }
        }
    }
    
    // Find most frequent label (with tie-breaking by smallest label)
    int best_label = current_label;
    int best_count = 0;
    
    for (int i = 0; i < num_unique; i++) {
        if (label_count[i] > best_count || 
            (label_count[i] == best_count && unique_labels[i] < best_label)) {
            best_count = label_count[i];
            best_label = unique_labels[i];
        }
    }
    
    // If no neighbors found, keep current label
    if (best_count == 0) {
        best_label = current_label;
    }
    
    labels_out[node] = best_label;
    
    // Track if any changes occurred
    if (best_label != current_label) {
        atomicAdd(changed, 1);
    }
}

// ============================================================================
// Label-Based Preference Assignment (Hybrid Approach)
// ============================================================================

// Compute node preferences based on label propagation results
extern "C" __global__ void compute_label_preferences(
    int num_nodes,
    int num_parts,
    const int* labels,
    const int* node_hyperedges,
    const int* node_offsets,
    const int* hyperedge_offsets,
    int* pref_nodes,
    int* pref_parts,
    int* pref_priorities
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    
    int node_label = labels[node];
    int node_degree = node_offsets[node + 1] - node_offsets[node];
    
    // Hybrid approach: some clustering + spreading to ensure all 64 partitions used
    // Balance between preserving label clusters and distributing across partitions
    int target_partition;
    
    if (node_degree <= 3) {
        // Low degree: mostly by label with some spreading
        target_partition = (node_label * 3 + node) % num_parts;
    } else if (node_degree <= 8) {
        // Medium degree: balanced mixing
        target_partition = (node_label * 2 + node_degree + node) % num_parts;
    } else {
        // High degree: more spreading
        target_partition = (node_label + node_degree * 2 + node) % num_parts;
    }
    
    pref_nodes[node] = node;
    pref_parts[node] = target_partition;
    
    // Priority: high-degree nodes first, then by label
    pref_priorities[node] = (10000 - node_degree) * 1000 + node_label;
}

// Execute assignments with fallback (like hyper_cluster)
extern "C" __global__ void execute_label_assignments(
    int num_nodes,
    int num_parts,
    int max_part_size,
    const int* sorted_nodes,
    const int* sorted_parts,
    int* partition,
    int* nodes_in_part
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    // Initialize
    for (int p = 0; p < num_parts; p++) {
        nodes_in_part[p] = 0;
    }
    
    // First pass: assign nodes to preferred partitions
    for (int i = 0; i < num_nodes; i++) {
        int node = sorted_nodes[i];
        int preferred_part = sorted_parts[i];
        
        if (node < 0 || node >= num_nodes) continue;
        if (preferred_part < 0 || preferred_part >= num_parts) continue;
        
        bool assigned = false;
        
        // Try preferred partition first
        if (nodes_in_part[preferred_part] < max_part_size) {
            partition[node] = preferred_part;
            nodes_in_part[preferred_part]++;
            assigned = true;
        } else {
            // Try nearby partitions (within 8 slots)
            for (int offset = 1; offset <= 8 && !assigned; offset++) {
                for (int sign = -1; sign <= 1; sign += 2) {
                    int try_part = (preferred_part + sign * offset + num_parts) % num_parts;
                    if (nodes_in_part[try_part] < max_part_size) {
                        partition[node] = try_part;
                        nodes_in_part[try_part]++;
                        assigned = true;
                        break;
                    }
                }
            }
        }
        
        // Round-robin fallback to ensure all partitions used
        if (!assigned) {
            int fallback = i % num_parts;
            for (int attempt = 0; attempt < num_parts; attempt++) {
                int try_part = (fallback + attempt) % num_parts;
                if (nodes_in_part[try_part] < max_part_size) {
                    partition[node] = try_part;
                    nodes_in_part[try_part]++;
                    assigned = true;
                    break;
                }
            }
        }
        
        // Last resort: force assign to least full partition
        if (!assigned) {
            int min_part = 0;
            int min_size = nodes_in_part[0];
            for (int p = 1; p < num_parts; p++) {
                if (nodes_in_part[p] < min_size) {
                    min_size = nodes_in_part[p];
                    min_part = p;
                }
            }
            partition[node] = min_part;
            nodes_in_part[min_part]++;
        }
    }
    
    // Second pass: ensure every partition has at least 1 node
    for (int p = 0; p < num_parts; p++) {
        if (nodes_in_part[p] == 0) {
            // Find a partition with >1 node and steal one
            for (int donor = 0; donor < num_parts; donor++) {
                if (nodes_in_part[donor] > 1) {
                    // Find a node in donor partition
                    for (int n = 0; n < num_nodes; n++) {
                        if (partition[n] == donor) {
                            partition[n] = p;
                            nodes_in_part[donor]--;
                            nodes_in_part[p]++;
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Serial FM Refinement - Compute Moves Only (No Execution)
// ============================================================================

__device__ int calculate_connectivity_gain(
    int node,
    int from_part,
    int to_part,
    const int* node_hedges,
    const int* node_offsets,
    const int* hedge_nodes,
    const int* hedge_offsets,
    const int* partition,
    int num_parts
) {
    if (from_part == to_part) return 0;
    
    int total_gain = 0;
    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    
    for (int i = start; i < end; i++) {
        int hedge = node_hedges[i];
        int hedge_start = hedge_offsets[hedge];
        int hedge_end = hedge_offsets[hedge + 1];
        
        bool parts_present[64] = {false};
        int parts_count = 0;
        
        for (int j = hedge_start; j < hedge_end; j++) {
            int other_node = hedge_nodes[j];
            int part = (other_node == node) ? from_part : partition[other_node];
            
            if (part >= 0 && part < num_parts && !parts_present[part]) {
                parts_present[part] = true;
                parts_count++;
            }
        }
        
        int connectivity_before = parts_count;
        
        bool from_part_has_others = false;
        for (int j = hedge_start; j < hedge_end; j++) {
            int other_node = hedge_nodes[j];
            if (other_node != node && partition[other_node] == from_part) {
                from_part_has_others = true;
                break;
            }
        }
        
        int connectivity_after = connectivity_before;
        
        if (!from_part_has_others) {
            connectivity_after--;
        }
        
        if (!parts_present[to_part]) {
            connectivity_after++;
        }
        
        total_gain += (connectivity_before - connectivity_after);
    }
    
    return total_gain;
}

// Compute potential FM moves (parallel, read-only)
extern "C" __global__ void compute_fm_moves(
    int num_nodes, 
    int num_parts, 
    int max_part_size,
    const int* node_hedges, 
    const int* node_offsets,
    const int* hedge_nodes, 
    const int* hedge_offsets,
    const int* partition,
    const int* nodes_in_part,
    int* move_gains,      // Output: gain for each node's best move
    int* move_targets     // Output: target partition (255 = no move)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    int current_part = partition[tid];
    int current_size = nodes_in_part[current_part];
    
    // Don't move from partitions with only 1 node
    if (current_size <= 1) {
        move_gains[tid] = 0;
        move_targets[tid] = 255;
        return;
    }
    
    int best_gain = 0;
    int best_target = 255;
    
    // Try all possible target partitions
    for (int target = 0; target < num_parts; target++) {
        if (target == current_part) continue;
        
        int target_size = nodes_in_part[target];
        
        // Don't move to full partitions
        if (target_size >= max_part_size) continue;
        
        // Calculate gain for this move
        int gain = calculate_connectivity_gain(
            tid, current_part, target, 
            node_hedges, node_offsets, 
            hedge_nodes, hedge_offsets, 
            partition, num_parts
        );
        
        // Keep track of best move
        if (gain > best_gain) {
            best_gain = gain;
            best_target = target;
        }
    }
    
    // Store result
    move_gains[tid] = best_gain;
    move_targets[tid] = best_target;
}

// Apply a single move (called serially from CPU)
extern "C" __global__ void apply_single_move(
    int node,
    int to_part,
    int* partition,
    int* nodes_in_part
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    
    int from_part = partition[node];
    
    if (from_part == to_part) return;
    if (to_part < 0 || to_part >= 64) return;
    
    // Update partition assignment
    partition[node] = to_part;
    
    // Update counters
    nodes_in_part[from_part]--;
    nodes_in_part[to_part]++;
}
