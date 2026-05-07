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
                // Round-robin seeding: first K nodes anchor different partitions (sigma's key trick)
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
                    partition[node] = node % num_parts;
                    nodes_in_part[node % num_parts]++;
                }
            }
        }
    }
}

// Mark boundary nodes: nodes with at least one cut hyperedge
extern "C" __global__ void mark_boundary_nodes(
    const int num_nodes, const int num_parts,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition,
    const unsigned long long *edge_flags_all,
    int *is_boundary,
    int *boundary_count
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) { is_boundary[node] = 0; return; }

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    unsigned long long current_bit = 1ULL << current_part;

    for (int j = start; j < end; j++) {
        int he = node_hyperedges[j];
        unsigned long long fa = edge_flags_all[he];
        if (fa & ~current_bit) {
            is_boundary[node] = 1;
            atomicAdd(boundary_count, 1);
            return;
        }
    }
    is_boundary[node] = 0;
}

// Only recompute flags for dirty hyperedges (containing moved nodes)
extern "C" __global__ void update_dirty_edge_flags(
    const int num_hyperedges, const int num_nodes,
    const int *hyperedge_nodes, const int *hyperedge_offsets,
    const int *partition,
    const int *edge_dirty,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;
    if (!edge_dirty[hedge]) return;

    int start = hyperedge_offsets[hedge];
    int end = hyperedge_offsets[hedge + 1];
    unsigned long long flags_all = 0, flags_double = 0;
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

// Mark hyperedges containing moved nodes as dirty
extern "C" __global__ void mark_dirty_edges(
    const int num_moved, const int *moved_nodes,
    const int *node_hyperedges, const int *node_offsets,
    int *edge_dirty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_moved) return;

    int node = moved_nodes[idx];
    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    for (int j = start; j < end; j++) {
        edge_dirty[node_hyperedges[j]] = 1;
    }
}

// Compute moves ONLY for boundary nodes
extern "C" __global__ void compute_moves_boundary(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int balance_weight, const int max_used_degree,
    const int neg_gain_threshold,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition, const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    const int *is_boundary,
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
    if (!is_boundary[node]) return;  // SKIP interior nodes

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

    if (best_gain >= neg_gain_threshold && best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : (best_gain < -32768 ? -32768 : best_gain);
        move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
        atomicAdd(num_valid_moves, 1);
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
    const int neg_gain_threshold,
    const int use_tiebreaker,
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

    if (best_gain >= neg_gain_threshold && best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : (best_gain < -32768 ? -32768 : best_gain);
        int encoded_target = use_tiebreaker ? ((best_target & 63) | ((node & 3) << 6)) : (best_target & 63);
        move_priorities[node] = (bg << 16) | (degree_weight << 8) | encoded_target;
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

// Execute selected moves on GPU (avoids uploading full partition each round)
extern "C" __global__ void execute_selected_moves(
    const int num_moves,
    const int *sorted_nodes,
    const int *sorted_parts,
    const int max_part_size,
    int *partition,
    int *nodes_in_part,
    int *moves_executed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int me = 0;
        for (int i = 0; i < num_moves; i++) {
            int node = sorted_nodes[i];
            int target_part = sorted_parts[i];
            if (node >= 0 && target_part >= 0) {
                int current_part = partition[node];
                if (current_part >= 0 &&
                    current_part != target_part &&
                    nodes_in_part[target_part] < max_part_size &&
                    nodes_in_part[current_part] > 1) {
                    partition[node] = target_part;
                    nodes_in_part[current_part]--;
                    nodes_in_part[target_part]++;
                    me++;
                }
            }
        }
        *moves_executed = me;
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

extern "C" __global__ void compute_connectivity_per_hedge(
    const int num_hyperedges,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    int *connectivity
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        unsigned long long parts_mask = 0;
        for (int k = start; k < end; k++) {
            int node = hyperedge_nodes[k];
            int part = partition[node];
            if (part >= 0 && part < 64) {
                parts_mask |= (1ULL << part);
            }
        }
        int count = __popcll(parts_mask);
        connectivity[hedge] = (count > 1) ? (count - 1) : 0;
    }
}

// Jet-style: every node computes best move and applies it simultaneously
// No balance check - rebalancing is done separately after
extern "C" __global__ void label_propagation_step(
    const int num_nodes, const int num_parts,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition_in,    // Read-only: current partition
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *partition_out,         // Write: new partition (can be same buffer if careful)
    int *move_count             // Atomic: count of moves made
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int current_part = partition_in[node];
    if ((unsigned)current_part >= (unsigned)num_parts) {
        partition_out[node] = current_part;
        return;
    }

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    if (node_degree <= 0) {
        partition_out[node] = current_part;
        return;
    }

    int cap = 256;
    int used_degree = node_degree > cap ? cap : node_degree;
    unsigned long long current_bit = 1ULL << current_part;
    int np = (num_parts < 64) ? num_parts : 64;

    unsigned short part_counts[64];
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

    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);
        if ((unsigned)target_part >= (unsigned)num_parts) continue;

        int gain = (int)part_counts[target_part] - count_current_present;
        if (gain > best_gain || (gain == best_gain && gain > 0 && target_part < best_target)) {
            best_gain = gain;
            best_target = target_part;
        }
    }

    partition_out[node] = best_target;
    if (best_target != current_part && best_gain > 0) {
        atomicAdd(move_count, 1);
    }
}

// Rebalance: move nodes from oversized to undersized parts
extern "C" __global__ void rebalance_greedy(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int *node_hyperedges, const int *node_offsets,
    const unsigned long long *edge_flags_all,
    int *partition, int *nodes_in_part
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int np = (num_parts < 64) ? num_parts : 64;

        // Fix oversized partitions
        for (int part = 0; part < np; part++) {
            while (nodes_in_part[part] > max_part_size) {
                // Find the node in this part with best connectivity to another undersized part
                int best_node = -1;
                int best_target = -1;
                int best_gain = -999999;

                for (int node = 0; node < num_nodes; node++) {
                    if (partition[node] != part) continue;

                    int start = node_offsets[node];
                    int end = node_offsets[node + 1];
                    int degree = end - start;
                    int used = (degree > 64) ? 64 : degree;
                    if (used == 0) {
                        // Zero-degree node: move to smallest part
                        if (best_gain < 0) {
                            int smallest = -1; int smallest_count = max_part_size + 1;
                            for (int p = 0; p < np; p++) {
                                if (p != part && nodes_in_part[p] < max_part_size && nodes_in_part[p] < smallest_count) {
                                    smallest_count = nodes_in_part[p]; smallest = p;
                                }
                            }
                            if (smallest >= 0) { best_node = node; best_target = smallest; best_gain = 0; }
                        }
                        continue;
                    }

                    unsigned long long cur_bit = 1ULL << part;
                    int part_counts[64];
                    for (int p = 0; p < np; p++) part_counts[p] = 0;

                    for (int j = 0; j < used; j++) {
                        int rel = (j * degree) / used;
                        int he = node_hyperedges[start + rel];
                        unsigned long long fa = edge_flags_all[he];
                        unsigned long long others = fa & ~cur_bit;
                        while (others) {
                            int bit = __ffsll(others) - 1;
                            others &= (others - 1);
                            part_counts[bit]++;
                        }
                    }

                    for (int p = 0; p < np; p++) {
                        if (p == part || nodes_in_part[p] >= max_part_size) continue;
                        if (part_counts[p] > best_gain) {
                            best_gain = part_counts[p]; best_node = node; best_target = p;
                        }
                    }
                }

                if (best_node >= 0 && best_target >= 0) {
                    partition[best_node] = best_target;
                    nodes_in_part[part]--;
                    nodes_in_part[best_target]++;
                } else {
                    break;
                }
            }
        }
    }
}

extern "C" __global__ void perturb_guided(
    const int num_high_hedges,
    const int *high_hedge_ids,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int num_parts,
    const int max_part_size,
    int *partition,
    int *nodes_in_part,
    unsigned long long seed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long state = seed;
        int np = (num_parts < 64) ? num_parts : 64;

        for (int h = 0; h < num_high_hedges; h++) {
            int hedge = high_hedge_ids[h];
            int start = hyperedge_offsets[hedge];
            int end = hyperedge_offsets[hedge + 1];
            int hedge_size = end - start;
            if (hedge_size <= 1) continue;

            int part_count[64];
            for (int p = 0; p < np; p++) part_count[p] = 0;

            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                int part = partition[node];
                if (part >= 0 && part < np) part_count[part]++;
            }

            // Find majority partition
            int majority_part = 0;
            for (int p = 1; p < np; p++) {
                if (part_count[p] > part_count[majority_part]) majority_part = p;
            }

            // Skip if already concentrated
            int num_parts_present = 0;
            for (int p = 0; p < np; p++) {
                if (part_count[p] > 0) num_parts_present++;
            }
            if (num_parts_present <= 1) continue;

            // Move minority nodes to majority (25% probability)
            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                int cur_part = partition[node];
                if (cur_part == majority_part) continue;
                if (nodes_in_part[cur_part] <= 1) continue;
                if (nodes_in_part[majority_part] >= max_part_size) continue;

                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                if ((state & 3) != 0) continue;  // 25% move probability

                partition[node] = majority_part;
                nodes_in_part[cur_part]--;
                nodes_in_part[majority_part]++;
            }
        }
    }
}

extern "C" __global__ void perturb_targeted(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int perturb_strength,
    const int *node_hyperedges,
    const int *node_offsets,
    const unsigned long long *edge_flags_all,
    int *partition,
    int *nodes_in_part,
    unsigned long long seed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long state = seed;
        int moves_made = 0;
        int target_moves = (num_nodes * perturb_strength) / 100;
        int np = (num_parts < 64) ? num_parts : 64;

        for (int attempt = 0; attempt < num_nodes * 2 && moves_made < target_moves; attempt++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            int node = (int)(state % (unsigned long long)num_nodes);
            int current_part = partition[node];
            if (current_part < 0 || current_part >= np) continue;
            if (nodes_in_part[current_part] <= 1) continue;

            // Find the partition with most shared hyperedge presence
            int start = node_offsets[node];
            int end = node_offsets[node + 1];
            int degree = end - start;
            if (degree == 0) continue;

            int used = (degree > 32) ? 32 : degree;
            unsigned long long cur_bit = 1ULL << current_part;
            int part_counts[64];
            for (int p = 0; p < np; p++) part_counts[p] = 0;

            for (int j = 0; j < used; j++) {
                int rel = (j * degree) / used;
                int he = node_hyperedges[start + rel];
                unsigned long long fa = edge_flags_all[he];
                unsigned long long others = fa & ~cur_bit;
                while (others) {
                    int bit = __ffsll(others) - 1;
                    others &= (others - 1);
                    part_counts[bit]++;
                }
            }

            // Move to most connected alternative partition
            int best_part = -1;
            int best_count = 0;
            for (int p = 0; p < np; p++) {
                if (p == current_part) continue;
                if (nodes_in_part[p] >= max_part_size) continue;
                if (part_counts[p] > best_count) {
                    best_count = part_counts[p];
                    best_part = p;
                }
            }

            if (best_part >= 0 && best_count > 0) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                if ((state & 1) == 0) {  // 50% probability
                    partition[node] = best_part;
                    nodes_in_part[current_part]--;
                    nodes_in_part[best_part]++;
                    moves_made++;
                }
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

    swap_gains[node * 3 + 0] = 0x7FFFFFFF;
    swap_gains[node * 3 + 1] = 0x7FFFFFFF;
    swap_gains[node * 3 + 2] = 0x7FFFFFFF;

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

// GSA kernels: GPU Swap Architecture
extern "C" __global__ void clear_swap_matrix(unsigned long long *swap_matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 4096) swap_matrix[idx] = 0;
}

extern "C" __global__ void build_swap_matrix(
    const int num_nodes, const int *partition, const int *swap_gains, unsigned long long *swap_matrix
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    int src_part = partition[node];
    if ((unsigned)src_part >= 64) return;

    for (int k = 0; k < 3; k++) {
        int val = swap_gains[node * 3 + k];
        if (val == 0) continue;

        int target_part = val & 0xFFFF;
        short basic_gain = (short)(val >> 16);

        if (target_part >= 0 && target_part < 64 && target_part != src_part && basic_gain > 0) {
            unsigned long long offset_gain = (unsigned long long)((int)basic_gain + 32768);
            unsigned long long packed = (offset_gain << 32) | (unsigned int)node;
            atomicMax(&swap_matrix[src_part * 64 + target_part], packed);
        }
    }
}

extern "C" __global__ void resolve_swap_cycles(
    unsigned long long *swap_matrix, int *partition, int *moves_executed
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int best_node[64][64];
    int best_gain[64][64];

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            unsigned long long packed = swap_matrix[i * 64 + j];
            if (packed == 0) {
                best_node[i][j] = -1;
                best_gain[i][j] = -999999;
            } else {
                best_node[i][j] = (int)(packed & 0xFFFFFFFFULL);
                best_gain[i][j] = (int)(packed >> 32) - 32768;
            }
        }
    }

    bool part_locked[64];
    for (int i = 0; i < 64; i++) part_locked[i] = false;
    int swaps = 0;

    for (int a = 0; a < 64; a++) {
        if (part_locked[a]) continue;
        for (int b = a + 1; b < 64; b++) {
            if (part_locked[b]) continue;
            int na = best_node[a][b];
            int nb = best_node[b][a];
            if (na >= 0 && nb >= 0 && (best_gain[a][b] + best_gain[b][a] > 0)) {
                partition[na] = b; partition[nb] = a;
                part_locked[a] = true; part_locked[b] = true;
                swaps += 2; break;
            }
        }
    }

    for (int a = 0; a < 64; a++) {
        if (part_locked[a]) continue;
        for (int b = 0; b < 64; b++) {
            if (a == b || part_locked[b]) continue;
            int na = best_node[a][b]; if (na < 0) continue;
            int gab = best_gain[a][b];
            for (int c = 0; c < 64; c++) {
                if (c == a || c == b || part_locked[c]) continue;
                int nb = best_node[b][c]; if (nb < 0) continue;
                int nc = best_node[c][a]; if (nc < 0) continue;

                if (gab + best_gain[b][c] + best_gain[c][a] > 0) {
                    partition[na] = b; partition[nb] = c; partition[nc] = a;
                    part_locked[a] = true; part_locked[b] = true; part_locked[c] = true;
                    swaps += 3; break;
                }
            }
            if (part_locked[a]) break;
        }
    }
    *moves_executed = swaps;
}

// ===== DCW: Dynamic Clause Weighting kernels =====

extern "C" __global__ void init_hyperedge_weights(const int num_hyperedges, int *weights) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) weights[hedge] = 1;
}

extern "C" __global__ void update_hyperedge_weights_dcw(
    const int num_hyperedges,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *weights
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;
    unsigned long long fa = edge_flags_all[hedge];
    unsigned long long fd = edge_flags_double[hedge];
    int k = __popcll(fa);
    if (k > 1) {
        int kd = __popcll(fd);
        int inc = 1;
        if (k == 2 && kd == 1) inc = 4;
        else if (k == 2) inc = 2;
        int w = weights[hedge] + inc;
        if (w > 128) w = 128;
        weights[hedge] = w;
    }
}

extern "C" __global__ void decay_hyperedge_weights(const int num_hyperedges, int *weights) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) {
        int w = weights[hedge];
        if (w > 1) {
            int new_w = w - (w >> 2);
            weights[hedge] = (new_w < 1) ? 1 : new_w;
        }
    }
}

extern "C" __global__ void compute_moves_boundary_weighted(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int balance_weight, const int max_used_degree,
    const int neg_gain_threshold,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition, const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    const int *edge_weights,
    const int *is_boundary,
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
    if (!is_boundary[node]) return;
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
    int part_counts[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_counts[p] = 0;
    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;
    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];
        int w = edge_weights[hyperedge];
        unsigned long long fa = edge_flags_all[hyperedge];
        unsigned long long fd = edge_flags_double[hyperedge];
        unsigned long long mask = (fa & ~current_bit) | (fd & current_bit);
        if (mask & current_bit) count_current_present += w;
        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            part_counts[bit] += w;
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
        int basic_gain = part_counts[target_part] - count_current_present;
        int balance_bonus = (balance_weight > 0 && current_size > shared_nip[target_part] + 1) ? balance_weight : 0;
        int total_gain = basic_gain + balance_bonus;
        if (total_gain > best_gain || (total_gain == best_gain && total_gain > 0 && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }
    if (best_gain >= neg_gain_threshold && best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : (best_gain < -32768 ? -32768 : best_gain);
        move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
        atomicAdd(num_valid_moves, 1);
    }
}

extern "C" __global__ void compute_moves_flags_weighted(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int balance_weight, const int max_used_degree,
    const int neg_gain_threshold, const int use_tiebreaker,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition, const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    const int *edge_weights,
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
    int part_counts[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_counts[p] = 0;
    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;
    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];
        int w = edge_weights[hyperedge];
        unsigned long long fa = edge_flags_all[hyperedge];
        unsigned long long fd = edge_flags_double[hyperedge];
        unsigned long long mask = (fa & ~current_bit) | (fd & current_bit);
        if (mask & current_bit) count_current_present += w;
        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            part_counts[bit] += w;
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
        int basic_gain = part_counts[target_part] - count_current_present;
        int balance_bonus = (balance_weight > 0 && current_size > shared_nip[target_part] + 1) ? balance_weight : 0;
        int total_gain = basic_gain + balance_bonus;
        if (total_gain > best_gain || (total_gain == best_gain && total_gain > 0 && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }
    if (best_gain >= neg_gain_threshold && best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : (best_gain < -32768 ? -32768 : best_gain);
        int encoded_target = use_tiebreaker ? ((best_target & 63) | ((node & 3) << 6)) : (best_target & 63);
        move_priorities[node] = (bg << 16) | (degree_weight << 8) | encoded_target;
        atomicAdd(num_valid_moves, 1);
    }
}

// TPLS: Thermal Parallel Local Search — SA acceptance inside GPU kernel
// Stochastic acceptance for negative-gain moves avoids parallel thrashing
// while enabling barrier crossing during the high-temperature phase.
extern "C" __global__ void compute_moves_boundary_thermal(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int balance_weight, const int max_used_degree,
    const float current_temp, const unsigned int random_seed,
    const int *node_hyperedges, const int *node_offsets,
    const int *partition, const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    const int *edge_weights,
    const int *is_boundary,
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
    if (!is_boundary[node]) return;
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
    int part_counts[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_counts[p] = 0;
    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;
    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];
        int w = edge_weights[hyperedge];
        unsigned long long fa = edge_flags_all[hyperedge];
        unsigned long long fd = edge_flags_double[hyperedge];
        unsigned long long mask = (fa & ~current_bit) | (fd & current_bit);
        if (mask & current_bit) count_current_present += w;
        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            part_counts[bit] += w;
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
        int basic_gain = part_counts[target_part] - count_current_present;
        int balance_bonus = (balance_weight > 0 && current_size > shared_nip[target_part] + 1) ? balance_weight : 0;
        int total_gain = basic_gain + balance_bonus;
        if (total_gain > best_gain || (total_gain == best_gain && total_gain > 0 && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }
    if (best_target == current_part) return;
    bool accept = false;
    if (best_gain > 0) {
        accept = true;
    } else if (best_gain >= -3 && current_temp > 0.001f) {
        // Wang hash fast RNG — no global memory, O(1) per thread
        unsigned int h = (unsigned int)node ^ random_seed;
        h ^= h >> 16; h *= 0x85ebca6bU;
        h ^= h >> 13; h *= 0xc2b2ae35U;
        h ^= h >> 16;
        float rand_val = (float)(h & 0x00FFFFFFu) / (float)0x01000000;
        float prob = expf((float)best_gain / current_temp);
        if (rand_val < prob) accept = true;
    }
    if (accept) {
        int bg = best_gain > 32767 ? 32767 : (best_gain < -32768 ? -32768 : best_gain);
        move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
        atomicAdd(num_valid_moves, 1);
    }
}
