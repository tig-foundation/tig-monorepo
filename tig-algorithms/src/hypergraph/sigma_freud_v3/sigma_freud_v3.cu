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
    const int *hyperedge_nodes,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;

    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        int hedge_size = end - start;

        int quarter_clusters = num_clusters >> 2;
        if (quarter_clusters <= 0) quarter_clusters = 1;

        int bucket = (hedge_size > 8) ? 3 :
                     (hedge_size > 4) ? 2 :
                     (hedge_size > 2) ? 1 : 0;

        int first = (hedge_size > 0) ? hyperedge_nodes[start] : hedge;
        int last  = (hedge_size > 0) ? hyperedge_nodes[end - 1] : hedge;

        unsigned int h = (unsigned int)first * 1103515245u
                       + (unsigned int)last * 12345u
                       + (unsigned int)hedge_size * 2654435761u
                       + (unsigned int)hedge;

        int cluster = bucket * quarter_clusters + (int)(h % (unsigned int)quarter_clusters);
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

        int base_part = 0;
        if (num_parts > 0 && max_clusters > 0) {
            base_part = (best_cluster * num_parts) / max_clusters;
            if (base_part >= num_parts) base_part = num_parts - 1;
        }

        pref_parts[node] = base_part;
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        int mv = max_votes > 32767 ? 32767 : max_votes;
        pref_priorities[node] = (mv << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
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
                int start_part = (i < num_parts) ? i : preferred_part;

                bool assigned = false;
                for (int attempt = 0; attempt < num_parts; attempt++) {
                    int try_part = (start_part + attempt) % num_parts;
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
    int *move_priorities,
    int *num_valid_moves
) {
    __shared__ int shared_nodes_in_part[64];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    move_priorities[node] = 0;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;
    if (shared_nodes_in_part[current_part] <= 1) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;

    int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) return;

    int degree_weight = node_degree > 255 ? 255 : node_degree;

    unsigned long long full_mask = (num_parts >= 64) ? ~0ULL : ((1ULL << num_parts) - 1ULL);
    unsigned long long current_bit = 1ULL << current_part;

    unsigned short part_counts[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_counts[p] = 0;

    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;

    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];
        int hedge_start = hyperedge_offsets[hyperedge];
        int hedge_end = hyperedge_offsets[hyperedge + 1];

        unsigned long long mask = 0ULL;

        for (int k = hedge_start; k < hedge_end; k++) {
            int other_node = hyperedge_nodes[k];
            if (other_node != node && (unsigned)other_node < (unsigned)num_nodes) {
                int part = partition[other_node];
                if ((unsigned)part < (unsigned)num_parts) {
                    mask |= 1ULL << part;
                    if ((mask & full_mask) == full_mask) break;
                }
            }
        }

        if (mask & current_bit) count_current_present++;

        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            part_counts[bit]++;
            cand_mask |= 1ULL << bit;
        }
    }

    int best_gain = 0;
    int best_target = current_part;

    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int basic_gain = (int)part_counts[target_part] - count_current_present;

        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;

        int total_gain = basic_gain + balance_bonus;

        if (total_gain > best_gain ||
            (total_gain == best_gain && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }

    if (best_gain > 0 && best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : best_gain;
        move_priorities[node] =
            (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
        atomicAdd(num_valid_moves, 1);
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