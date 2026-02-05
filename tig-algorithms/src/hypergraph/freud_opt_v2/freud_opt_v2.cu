#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void hyperedge_clustering(
    const int num_hyperedges,
    const int num_clusters,
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
        
        int bucket = (hedge_size > 8) ? 3 :
                     (hedge_size > 4) ? 2 :
                     (hedge_size > 2) ? 1 : 0;
        int cluster = bucket * quarter_clusters + (hedge & cluster_mask);
        
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
    int *pref_parts,
    int *pref_priorities
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;
        
        int cluster_votes[64];
        int max_clusters = min(num_hedge_clusters, 64);
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
                int weight = (hedge_size <= 2) ? 6 :
                             (hedge_size <= 4) ? 4 :
                             (hedge_size <= 8) ? 2 : 1;
                
                cluster_votes[cluster] += weight;
                
                if (cluster_votes[cluster] > max_votes || 
                    (cluster_votes[cluster] == max_votes && cluster < best_cluster)) {
                    max_votes = cluster_votes[cluster];
                    best_cluster = cluster;
                }
            }
        }

        int base_part = (num_parts > 0) ? (best_cluster % num_parts) : 0;
        int target_partition = base_part;
        
        pref_parts[node] = target_partition;
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        pref_priorities[node] = (max_votes << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
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

extern "C" __global__ void precompute_edge_flags(
    const int num_hyperedges,
    const int num_nodes,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        
        unsigned long long flags_all = 0;
        unsigned long long flags_double = 0;
        
        for (int k = start; k < end; k++) {
            int node = hyperedge_nodes[k];
            if (node >= 0 && node < num_nodes) {
                int part = partition[node];
                if (part >= 0 && part < 64) {
                    unsigned long long bit = 1ULL << part;
                    flags_double |= (flags_all & bit);
                    flags_all |= bit;
                }
            }
        }
        
        edge_flags_all[hedge] = flags_all;
        edge_flags_double[hedge] = flags_double;
    }
}

extern "C" __global__ void compute_refinement_moves(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *move_parts,
    int *move_priorities,
    int *num_valid_moves,
    const int edge_flag_cap,
    unsigned long long *global_edge_flags
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        move_parts[node] = partition[node];
        move_priorities[node] = 0;
        
        int current_part = partition[node];
        if (current_part < 0 || current_part >= num_parts || nodes_in_part[current_part] <= 1) return;
        
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        int used_degree = node_degree > edge_flag_cap ? edge_flag_cap : node_degree;
        
        unsigned long long *edge_flags = &global_edge_flags[(size_t)node * edge_flag_cap];
        unsigned long long cur_node_bit = 1ULL << current_part;

        for (int j = 0; j < used_degree; j++) {
            int rel = (int)(((long long)j * node_degree) / used_degree);
            int hyperedge = node_hyperedges[start + rel];

            unsigned long long flags_all = edge_flags_all[hyperedge];
            unsigned long long flags_double = edge_flags_double[hyperedge];

            edge_flags[j] = (flags_all & ~cur_node_bit) | (flags_double & cur_node_bit);
        }

        int original_cost = 0;
        for (int j = 0; j < used_degree; j++) {
            int lambda = __popcll(edge_flags[j] | cur_node_bit);
            if (lambda > 1) {
                original_cost += (lambda - 1);
            }
        }

        int candidates[64];
        int num_candidates = 0;
        bool seen[64] = {false};

        for (int j = 0; j < used_degree; j++) {
            unsigned long long flags = edge_flags[j];
            
            while (flags) {
                int bit = __ffsll(flags) - 1;
                flags &= ~(1ULL << bit);
                if (bit != current_part && !seen[bit] && num_candidates < 64) {
                    candidates[num_candidates++] = bit;
                    seen[bit] = true;
                }
            }
        }
        
        int best_gain = 0;
        int best_target = current_part;
        
        for (int i = 0; i < num_candidates; i++) {
            int target_part = candidates[i];
            if (target_part < 0 || target_part >= num_parts) continue;
            if (nodes_in_part[target_part] >= max_part_size) continue;
            
            int new_cost = 0;
            for (int j = 0; j < used_degree; j++) {
                int lambda = __popcll(edge_flags[j] | (1ULL << target_part));
                if (lambda > 1) {
                    new_cost += (lambda - 1);
                }
            }
            
            int basic_gain = original_cost - new_cost;
            
            int current_size = nodes_in_part[current_part];
            int target_size = nodes_in_part[target_part];
            int balance_bonus = 0;
            
            if (current_size > target_size + 1) {
                balance_bonus = 4;
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
            move_priorities[node] = (best_gain << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
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

extern "C" __global__ void radix_histogram_chunked(
    const int n,
    const int num_chunks,
    const int *keys,
    const int shift,
    int *chunk_histograms
) {
    int chunk = blockIdx.x;
    if (chunk >= num_chunks) return;
    
    __shared__ int local_hist[256];
    
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    int chunk_start = chunk * 256;
    int chunk_end = min(chunk_start + 256, n);
    
    for (int i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        int digit = (keys[i] >> shift) & 0xFF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();
    
    for (int d = threadIdx.x; d < 256; d += blockDim.x) {
        chunk_histograms[chunk * 256 + d] = local_hist[d];
    }
}

extern "C" __global__ void radix_prefix_and_scatter(
    const int n,
    const int num_chunks,
    const int *keys_in,
    const int *vals_in,
    const int shift,
    const int *chunk_histograms,
    int *chunk_offsets,
    int *keys_out,
    int *vals_out,
    int *ready_flag
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int digit_totals[256];
        for (int d = 0; d < 256; d++) {
            digit_totals[d] = 0;
            for (int c = 0; c < num_chunks; c++) {
                digit_totals[d] += chunk_histograms[c * 256 + d];
            }
        }
        
        int digit_starts[256];
        int sum = 0;
        for (int d = 0; d < 256; d++) {
            digit_starts[d] = sum;
            sum += digit_totals[d];
        }
        
        int running[256];
        for (int d = 0; d < 256; d++) running[d] = digit_starts[d];
        
        for (int c = 0; c < num_chunks; c++) {
            for (int d = 0; d < 256; d++) {
                chunk_offsets[c * 256 + d] = running[d];
                running[d] += chunk_histograms[c * 256 + d];
            }
        }
        
        __threadfence();
        atomicExch(ready_flag, 1);
    }
    
    if (threadIdx.x == 0) {
        while (atomicAdd(ready_flag, 0) == 0) {}
    }
    __syncthreads();
    
    int chunk = blockIdx.x;
    if (chunk >= num_chunks) return;
    
    __shared__ int offsets[256];
    
    for (int d = threadIdx.x; d < 256; d += blockDim.x) {
        offsets[d] = chunk_offsets[chunk * 256 + d];
    }
    __syncthreads();
    
    int chunk_start = chunk * 256;
    int chunk_end = min(chunk_start + 256, n);
    
    if (threadIdx.x == 0) {
        for (int i = chunk_start; i < chunk_end; i++) {
            int key = keys_in[i];
            int digit = (key >> shift) & 0xFF;
            int pos = offsets[digit]++;
            keys_out[pos] = key;
            vals_out[pos] = vals_in[i];
        }
    }
}

extern "C" __global__ void init_indices(
    const int n,
    int *indices
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        indices[i] = i;
    }
}

extern "C" __global__ void invert_keys(
    const int n,
    const int max_key,
    const int *keys_in,
    int *keys_out
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        keys_out[i] = max_key - keys_in[i];
    }
}

extern "C" __global__ void gather_sorted(
    const int n,
    const int *sorted_indices,
    const int *src,
    int *dst
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = src[sorted_indices[i]];
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
