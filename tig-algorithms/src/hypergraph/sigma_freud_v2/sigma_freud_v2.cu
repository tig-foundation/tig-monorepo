#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void reset_counters(int *num_valid_moves, int *moves_executed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *num_valid_moves = 0;
        *moves_executed = 0;
    }
}

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

        int bucket = (hedge_size > 8) ? 3 :
                     (hedge_size > 4) ? 2 :
                     (hedge_size > 2) ? 1 : 0;
        int cluster = bucket * quarter_clusters + (hedge % quarter_clusters);

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
        for (int i = 0; i < max_clusters; i++) cluster_votes[i] = 0;

        int max_votes = 0;
        int best_cluster = 0;

        for (int j = start; j < end; j++) {
            int hyperedge = node_hyperedges[j];
            int cluster = hyperedge_clusters[hyperedge] % max_clusters;

            if (cluster >= 0) {
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

extern "C" __global__ void compute_refinement_moves(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    const int *nodes_in_part,
    int *move_parts,
    int *move_priorities,
    int *num_valid_moves,
    unsigned long long *global_edge_flags
) {
    __shared__ int shared_nodes_in_part[64];    
    
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();
    
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < num_nodes) {
        move_parts[node] = partition[node];
        move_priorities[node] = 0;

        int current_part = partition[node];
        if (current_part < 0 || current_part >= num_parts || shared_nodes_in_part[current_part] <= 1) return;

        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;        
        
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        int used_degree = node_degree > 256 ? 256 : node_degree;

        unsigned long long *edge_flags = &global_edge_flags[(size_t)node * 256];

        for (int j = 0; j < used_degree; j++) {
            edge_flags[j] = 0;

            int rel = (int)(((long long)j * node_degree) / used_degree);
            int hyperedge = node_hyperedges[start + rel];
            int hedge_start = hyperedge_offsets[hyperedge];
            int hedge_end = hyperedge_offsets[hyperedge + 1];

            for (int k = hedge_start; k < hedge_end; k++) {
                int other_node = hyperedge_nodes[k];
                if (other_node != node && other_node >= 0 && other_node < num_nodes) {
                    int part = partition[other_node];
                    if (part >= 0 && part < 64) {
                        edge_flags[j] |= 1ULL << part;
                    }
                }
            }
        }

        int original_cost = 0;
        for (int j = 0; j < used_degree; j++) {
            int lambda = __popcll(edge_flags[j] | (1ULL << current_part));
            if (lambda > 1) original_cost += (lambda - 1);
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
            if (shared_nodes_in_part[target_part] >= max_part_size) continue;

            int new_cost = 0;
            for (int j = 0; j < used_degree; j++) {
                int lambda = __popcll(edge_flags[j] | (1ULL << target_part));
                if (lambda > 1) new_cost += (lambda - 1);
            }

            int basic_gain = original_cost - new_cost;

            int current_size = shared_nodes_in_part[current_part];
            int target_size = shared_nodes_in_part[target_part];
            int balance_bonus = 0;

            if (current_size > target_size + 1) balance_bonus = 4;

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
