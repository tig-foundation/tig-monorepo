#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void hyperedge_clustering(
    const int num_hyperedges,
    const int num_clusters,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        int hedge_size = end - start;
        
        int quarter_clusters = num_clusters >> 2;
        int cluster_mask = quarter_clusters - 1;
        
        int cluster;
        if (hedge_size <= 2) {
            cluster = hedge & cluster_mask;
        } else if (hedge_size <= 4) {
            cluster = quarter_clusters + (hedge & cluster_mask);
        } else if (hedge_size <= 8) {
            cluster = (quarter_clusters << 1) + (hedge & cluster_mask);
        } else {
            cluster = (quarter_clusters * 3) + (hedge & cluster_mask);
        }
        
        hyperedge_clusters[hedge] = cluster;
    }
}

extern "C" __global__ void compute_node_preferences(
    const int num_nodes,
    const int num_parts,
    const int num_hedge_clusters,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_clusters,
    const int *hyperedge_offsets,
    int *pref_nodes,
    int *pref_parts,
    int *pref_gains,
    int *pref_priorities
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;
        
        int cluster_votes[256];
        int max_clusters = min(num_hedge_clusters, 256);
        for (int i = 0; i < max_clusters; i++) {
            cluster_votes[i] = 0;
        }
        
        int max_votes = 0;
        int best_cluster = 0;
        
        for (int j = start; j < end; j++) {
            int hyperedge = node_hyperedges[j];
            int cluster = hyperedge_clusters[hyperedge];
            
            if (cluster >= 0 && cluster < max_clusters) {
                int hedge_start = hyperedge_offsets[hyperedge];
                int hedge_end = hyperedge_offsets[hyperedge + 1];
                int hedge_size = hedge_end - hedge_start;
                int weight = (hedge_size <= 3) ? 4 : (hedge_size <= 6) ? 2 : 1;
                
                cluster_votes[cluster] += weight;
                
                if (cluster_votes[cluster] > max_votes || 
                    (cluster_votes[cluster] == max_votes && cluster < best_cluster)) {
                    max_votes = cluster_votes[cluster];
                    best_cluster = cluster;
                }
            }
        }
        
        int target_partition;
        if (node_degree <= 3) {
            target_partition = (best_cluster + node) % num_parts;
        } else if (node_degree <= 8) {
            target_partition = (best_cluster + node_degree + node) % num_parts;
        } else {
            target_partition = (best_cluster * 2 + node_degree + node) % num_parts;
        }
        
        pref_nodes[node] = node;
        pref_parts[node] = target_partition;
        pref_gains[node] = max_votes;            
        pref_priorities[node] = (max_votes << 16) + (node_degree << 8) + (num_parts - (node % num_parts));
    }
}

extern "C" __global__ void execute_node_assignments(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *sorted_nodes,
    const int *sorted_parts,
    int *partition,
    int *nodes_in_part
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < num_nodes; i++) {
            int node = sorted_nodes[i];
            int preferred_part = sorted_parts[i];
            
            if (node >= 0 && node < num_nodes && preferred_part >= 0 && preferred_part < num_parts) {
                bool assigned = false;
                for (int attempt = 0; attempt < num_parts; attempt++) {
                    int try_part = (preferred_part + attempt) % num_parts;
                    if (nodes_in_part[try_part] < max_part_size) {
                        partition[node] = try_part;
                        nodes_in_part[try_part]++;
                        assigned = true;
                        break;
                    }
                }
                
                if (!assigned) {
                    int fallback_part = node % num_parts;
                    partition[node] = fallback_part;
                    nodes_in_part[fallback_part]++;
                }
            }
        }
    }
}

extern "C" __global__ void compute_refinement_moves(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int num_hyperedges,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    const int *nodes_in_part,
    int *move_nodes,
    int *move_parts,
    int *move_gains,
    int *move_priorities,
    int *num_valid_moves,
    const int round,
    unsigned long long *global_edge_flags_low,
    unsigned long long *global_edge_flags_high
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        move_nodes[node] = node;
        move_parts[node] = partition[node];
        move_gains[node] = 0;
        move_priorities[node] = 0;
        
        int current_part = partition[node];
        if (current_part < 0 || current_part >= num_parts || nodes_in_part[current_part] <= 1) return;
        
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;
        
        if (node_degree > 2900) return;
        
        bool use_dual_buffer = (num_parts > 64);
        
        unsigned long long *edge_flags_low = &global_edge_flags_low[node * 2900];
        unsigned long long *edge_flags_high = &global_edge_flags_high[node * 2900];
        
        for (int j = 0; j < node_degree; j++) {
            edge_flags_low[j] = 0;
            if (use_dual_buffer) {
                edge_flags_high[j] = 0;
            }
            
            int hyperedge = node_hyperedges[start + j];
            int hedge_start = hyperedge_offsets[hyperedge];
            int hedge_end = hyperedge_offsets[hyperedge + 1];
            
            for (int k = hedge_start; k < hedge_end; k++) {
                int other_node = hyperedge_nodes[k];
                if (other_node != node && other_node >= 0 && other_node < num_nodes) {
                    int part = partition[other_node];
                    if (part >= 0 && part < num_parts) {
                        if (use_dual_buffer) {
                            if (part < 64) {
                                edge_flags_low[j] |= 1ULL << part;
                            } else {
                                edge_flags_high[j] |= 1ULL << (part - 64);
                            }
                        } else {
                            if (part < min(num_parts, 64)) {
                                edge_flags_low[j] |= 1ULL << part;
                            }
                        }
                    }
                }
            }
        }
        
        int original_cost = 0;
        for (int j = 0; j < node_degree; j++) {
            unsigned long long current_low = edge_flags_low[j];
            unsigned long long current_high = use_dual_buffer ? edge_flags_high[j] : 0;
            
            int lambda;
            if (use_dual_buffer) {
                if (current_part < 64) {
                    current_low |= 1ULL << current_part;
                } else {
                    current_high |= 1ULL << (current_part - 64);
                }
                lambda = __popcll(current_low) + __popcll(current_high);
            } else {
                lambda = __popcll(current_low | (1ULL << current_part));
            }
            
            if (lambda > 1) {
                original_cost += (lambda - 1);
            }
        }
        
        int best_gain = 0;
        int best_target = current_part;
        
        for (int offset = 0; offset < num_parts; offset++) {
            int target_part = (node + round + offset) % num_parts;
            if (target_part == current_part) continue;
            if (target_part < 0 || target_part >= num_parts) continue;
            if (nodes_in_part[target_part] >= max_part_size) continue;
            
            int new_cost = 0;
            for (int j = 0; j < node_degree; j++) {
                unsigned long long target_low = edge_flags_low[j];
                unsigned long long target_high = use_dual_buffer ? edge_flags_high[j] : 0;
                
                int lambda;
                if (use_dual_buffer) {
                    if (target_part < 64) {
                        target_low |= 1ULL << target_part;
                    } else {
                        target_high |= 1ULL << (target_part - 64);
                    }
                    lambda = __popcll(target_low) + __popcll(target_high);
                } else {
                    lambda = __popcll(target_low | (1ULL << target_part));
                }
                
                if (lambda > 1) {
                    new_cost += (lambda - 1);
                }
            }
            
            int basic_gain = original_cost - new_cost;
            
            int current_size = nodes_in_part[current_part];
            int target_size = nodes_in_part[target_part];
            int balance_bonus = 0;
            
            if (current_size > target_size + 1) {
                int size_diff = current_size - target_size;
                if (num_parts >= 120) {
                    balance_bonus = min(size_diff / 2 + 1, 5);
                } else if (num_parts >= 100) {
                    balance_bonus = min(size_diff / 2 + 2, 6);
                } else {
                    balance_bonus = min(size_diff / 2 + 3, 7);
                }
            }
            
            int total_gain = basic_gain + balance_bonus;
            
            if (total_gain > best_gain || 
                (total_gain == best_gain && target_part < best_target)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        }
        
        if (best_gain > 0 && best_target != current_part) {
            move_parts[node] = best_target;
            move_gains[node] = best_gain;
            move_priorities[node] = (best_gain << 16) + (node_degree << 8) + (num_parts - (node % num_parts));
            atomicAdd(num_valid_moves, 1);
        }
    }
}

extern "C" __global__ void execute_refinement_moves(
    const int num_valid_moves,
    const int *sorted_nodes,
    const int *sorted_parts,
    const int max_part_size,
    int *partition,
    int *nodes_in_part,
    int *moves_executed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < num_valid_moves; i++) {
            int node = sorted_nodes[i];
            int target_part = sorted_parts[i];
            
            if (node >= 0 && target_part >= 0) {
                int current_part = partition[node];
                
                if (current_part >= 0 &&
                    nodes_in_part[target_part] < max_part_size && 
                    nodes_in_part[current_part] > 1 &&
                    partition[node] == current_part) {
                    
                    partition[node] = target_part;
                    nodes_in_part[current_part]--;
                    nodes_in_part[target_part]++;
                    (*moves_executed)++;
                }
            }
        }
    }
}

extern "C" __global__ void balance_final(
    const int num_nodes,
    const int num_parts,
    const int min_part_size,
    const int max_part_size,
    int *partition,
    int *nodes_in_part
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int part = 0; part < num_parts; part++) {
            while (nodes_in_part[part] < min_part_size) {
                bool moved = false;
                for (int other_part = 0; other_part < num_parts && !moved; other_part++) {
                    if (other_part != part && nodes_in_part[other_part] > min_part_size) {
                        for (int node = 0; node < num_nodes; node++) {
                            if (partition[node] == other_part) {
                                partition[node] = part;
                                nodes_in_part[other_part]--;
                                nodes_in_part[part]++;
                                moved = true;
                                break;
                            }
                        }
                    }
                }
                if (!moved) break;
            }
        }
        
        for (int part = 0; part < num_parts; part++) {
            while (nodes_in_part[part] > max_part_size) {
                bool moved = false;
                for (int other_part = 0; other_part < num_parts && !moved; other_part++) {
                    if (other_part != part && nodes_in_part[other_part] < max_part_size) {
                        for (int node = 0; node < num_nodes; node++) {
                            if (partition[node] == part) {
                                partition[node] = other_part;
                                nodes_in_part[part]--;
                                nodes_in_part[other_part]++;
                                moved = true;
                                break;
                            }
                        }
                    }
                }
                if (!moved) break;
            }
        }
    }
}
