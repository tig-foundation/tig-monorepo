#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void hyperedge_clustering(
    const int num_hyperedges, const int num_clusters,
    const int *hyperedge_offsets, const int *hyperedge_nodes,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        int hedge_size = end - start;
        int quarter = num_clusters >> 2;
        if (quarter <= 0) quarter = 1;
        int bucket = (hedge_size > 8) ? 3 : (hedge_size > 4) ? 2 : (hedge_size > 2) ? 1 : 0;
        int first = (hedge_size > 0) ? hyperedge_nodes[start] : hedge;
        int last  = (hedge_size > 0) ? hyperedge_nodes[end - 1] : hedge;
        unsigned int h = (unsigned int)first * 1103515245u
                       + (unsigned int)last * 12345u
                       + (unsigned int)hedge_size * 2654435761u
                       + (unsigned int)hedge;
        hyperedge_clusters[hedge] = bucket * quarter + (int)(h % (unsigned int)quarter);
    }
}

extern "C" __global__ void compute_node_preferences(
    const int num_nodes, const int num_parts, const int num_hedge_clusters,
    const int *node_hyperedges, const int *node_offsets,
    const int *hyperedge_clusters, const int *hyperedge_offsets,
    const int restart_id, const unsigned int random_seed,
    int *pref_parts, int *pref_priorities
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;

    int cluster_votes[64];
    for (int i = 0; i < 64; i++) cluster_votes[i] = 0;

    for (int j = start; j < end; j++) {
        int hyperedge = node_hyperedges[j];
        int cluster = hyperedge_clusters[hyperedge];
        if (cluster >= 0 && cluster < 64) {
            int hedge_size = hyperedge_offsets[hyperedge + 1] - hyperedge_offsets[hyperedge];
            int weight = (hedge_size <= 2) ? 6 : (hedge_size <= 4) ? 4 : (hedge_size <= 8) ? 2 : 1;
            cluster_votes[cluster] += weight;
        }
    }

    int max_votes = -1;
    int best_cluster = 0;
    for (int i = 0; i < 64; i++) {
        if (cluster_votes[i] > 0 && restart_id > 0) {
            unsigned int hash = node * 1664525 + i * 1013904223 + restart_id * 12345 + random_seed;
            hash ^= (hash >> 16); hash *= 0x85ebca6b;
            hash ^= (hash >> 13); hash *= 0xc2b2ae35;
            hash ^= (hash >> 16);
            cluster_votes[i] += (hash % 4);
        }
        if (cluster_votes[i] > max_votes || (cluster_votes[i] == max_votes && i < best_cluster)) {
            max_votes = cluster_votes[i];
            best_cluster = i;
        }
    }
    int base_part = (num_parts > 0 && num_hedge_clusters > 0) ? (best_cluster * num_parts) / num_hedge_clusters : 0;
    if (base_part >= num_parts) base_part = num_parts - 1;
    pref_parts[node] = base_part;
    if (max_votes > 32700) max_votes = 32700;
    int degree_weight = node_degree > 255 ? 255 : node_degree;
    pref_priorities[node] = (max_votes << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
}

extern "C" __global__ void execute_node_assignments(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int *sorted_nodes, const int *sorted_parts,
    int *partition, int *nodes_in_part
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
                    partition[node] = node % num_parts;
                    nodes_in_part[node % num_parts]++;
                }
            }
        }
    }
}

extern "C" __global__ void precompute_edge_flags(
    const int num_hyperedges, const int num_nodes,
    const int *hyperedge_nodes, const int *hyperedge_offsets,
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

extern "C" __global__ void compute_moves_flags(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int balance_weight, const int max_used_degree,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition, const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *move_priorities, int *num_valid_moves
) {
    __shared__ int shared_nip[64];
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nip[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    move_priorities[node] = 0;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;
    if (shared_nip[current_part] <= 1) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    if (node_degree <= 0) return;

    int cap = (max_used_degree > 0) ? max_used_degree : 256;
    int used_degree = node_degree > cap ? cap : node_degree;
    int degree_weight = node_degree > 255 ? 255 : node_degree;
    unsigned long long current_bit = 1ULL << current_part;

    unsigned short part_counts[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_counts[p] = 0;

    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;

    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];

        unsigned long long fa = edge_flags_all[hyperedge];
        unsigned long long fd = edge_flags_double[hyperedge];
        unsigned long long mask = (fa & ~current_bit) | (fd & current_bit);

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
    int current_size = shared_nip[current_part];

    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nip[target_part] >= max_part_size) continue;

        int basic_gain = (int)part_counts[target_part] - count_current_present;
        int balance_bonus = (balance_weight > 0 && current_size > shared_nip[target_part] + 1) ? balance_weight : 0;
        int total_gain = basic_gain + balance_bonus;

        if (total_gain > best_gain || (total_gain == best_gain && total_gain > 0 && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }

    if (best_gain > 0 && best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : best_gain;
        move_priorities[node] = (bg << 16) | (degree_weight << 8) | (best_target & 63);
        atomicAdd(num_valid_moves, 1);
    }
}

extern "C" __global__ void perturb_solution(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int perturb_strength,
    int *partition, int *nodes_in_part, unsigned long long seed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long state = seed;
        int moves_made = 0;
        int target_moves = (num_nodes * perturb_strength) / 100;
        for (int attempt = 0; attempt < num_nodes && moves_made < target_moves; attempt++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            int node = (int)(state % (unsigned long long)num_nodes);
            int current_part = partition[node];
            if (current_part < 0 || current_part >= num_parts) continue;
            if (nodes_in_part[current_part] <= 1) continue;
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            int target_part = (int)(state % (unsigned long long)num_parts);
            if (target_part != current_part && nodes_in_part[target_part] < max_part_size) {
                partition[node] = target_part;
                nodes_in_part[current_part]--;
                nodes_in_part[target_part]++;
                moves_made++;
            }
        }
    }
}

extern "C" __global__ void reset_counters(int *num_valid_moves) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *num_valid_moves = 0;
    }
}

extern "C" __global__ void balance_final(
    const int num_nodes, const int num_parts,
    const int min_part_size, const int max_part_size,
    int *partition, int *nodes_in_part
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int part = 0; part < num_parts; part++) {
            while (nodes_in_part[part] < min_part_size) {
                bool moved = false;
                for (int other = 0; other < num_parts && !moved; other++) {
                    if (other != part && nodes_in_part[other] > min_part_size) {
                        for (int node = 0; node < num_nodes; node++) {
                            if (partition[node] == other) {
                                partition[node] = part;
                                nodes_in_part[other]--;
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
                for (int other = 0; other < num_parts && !moved; other++) {
                    if (other != part && nodes_in_part[other] < max_part_size) {
                        for (int node = 0; node < num_nodes; node++) {
                            if (partition[node] == part) {
                                partition[node] = other;
                                nodes_in_part[part]--;
                                nodes_in_part[other]++;
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

extern "C" __global__ void my_calc_connectivity(
    const int num_hyperedges, const int *hyperedge_offsets,
    const int *hyperedge_nodes, const int *partition, unsigned int *metric
) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_hyperedges; idx += blockDim.x * gridDim.x) {
        unsigned long long flags = 0;
        for (int pos = hyperedge_offsets[idx]; pos < hyperedge_offsets[idx + 1]; pos++) {
            int part = partition[hyperedge_nodes[pos]];
            if (part >= 0 && part < 64) flags |= (1ULL << part);
        }
        int conn = __popcll(flags);
        if (conn > 0) atomicAdd(metric, conn - 1);
    }
}

extern "C" __global__ void compute_swap_gains_top3(
    const int num_nodes,
    const int num_parts,
    const int neg_gain_thresh,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *swap_gains
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    swap_gains[node * 3 + 0] = 0;
    swap_gains[node * 3 + 1] = 0;
    swap_gains[node * 3 + 2] = 0;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) return;

    unsigned long long current_bit = 1ULL << current_part;

    unsigned short part_counts[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_counts[p] = 0;

    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;

    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];

        unsigned long long fa = edge_flags_all[hyperedge];
        unsigned long long fd = edge_flags_double[hyperedge];
        unsigned long long mask = (fa & ~current_bit) | (fd & current_bit);

        if (mask & current_bit) count_current_present++;

        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            part_counts[bit]++;
            cand_mask |= 1ULL << bit;
        }
    }

    int degree_scaled_thresh = neg_gain_thresh + node_degree / 20;
    int min_gain = -degree_scaled_thresh;

    int top_gain[3];
    int top_part[3];
    top_gain[0] = min_gain - 1; top_gain[1] = min_gain - 1; top_gain[2] = min_gain - 1;
    top_part[0] = -1; top_part[1] = -1; top_part[2] = -1;

    unsigned long long tmp = cand_mask;
    while (tmp) {
        int target_part = __ffsll(tmp) - 1;
        tmp &= (tmp - 1);
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        int basic_gain = (int)part_counts[target_part] - count_current_present;
        if (basic_gain < min_gain) continue;

        if (basic_gain > top_gain[0] || (basic_gain == top_gain[0] && target_part < top_part[0])) {
            top_gain[2] = top_gain[1]; top_part[2] = top_part[1];
            top_gain[1] = top_gain[0]; top_part[1] = top_part[0];
            top_gain[0] = basic_gain; top_part[0] = target_part;
        } else if (basic_gain > top_gain[1] || (basic_gain == top_gain[1] && target_part < top_part[1])) {
            top_gain[2] = top_gain[1]; top_part[2] = top_part[1];
            top_gain[1] = basic_gain; top_part[1] = target_part;
        } else if (basic_gain > top_gain[2] || (basic_gain == top_gain[2] && target_part < top_part[2])) {
            top_gain[2] = basic_gain; top_part[2] = target_part;
        }
    }

    for (int k = 0; k < 3; k++) {
        if (top_part[k] >= 0 && top_gain[k] >= min_gain) {
            int g = top_gain[k];
            if (g > 32767) g = 32767;
            if (g < -32768) g = -32768;
            short g16 = (short)g;
            unsigned short t16 = (unsigned short)(top_part[k] & 0xFFFF);
            swap_gains[node * 3 + k] = ((int)(unsigned short)g16 << 16) | (int)t16;
        }
    }
}
