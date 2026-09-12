#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void choose_elite_per_hyperedge_10k(
    const int num_hyperedges,
    const int num_nodes,
    const int num_parts,
    const int num_elites,
    const int *elite_partitions,
    const int *elite_order,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    int *hedge_choice_elite
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;

    int start = hyperedge_offsets[hedge];
    int end = hyperedge_offsets[hedge + 1];
    int np = (num_parts < 64) ? num_parts : 64;

    int best_elite = elite_order[0];
    int best_parts = 2147483647;

    for (int r = 0; r < num_elites; r++) {
        int elite_id = elite_order[r];
        unsigned long long mask = 0ULL;

        for (int k = start; k < end; k++) {
            int node = hyperedge_nodes[k];
            if ((unsigned)node >= (unsigned)num_nodes) continue;
            long long idx = (long long)elite_id * (long long)num_nodes + (long long)node;
            int part = elite_partitions[idx];
            if ((unsigned)part < (unsigned)np) {
                mask |= (1ULL << part);
            }
        }

        int parts = __popcll(mask);
        if (parts < best_parts) {
            best_parts = parts;
            best_elite = elite_id;
            if (best_parts <= 1) break;
        }
    }

    hedge_choice_elite[hedge] = best_elite;
}

extern "C" __global__ void assign_from_elite_votes_10k(
    const int num_nodes,
    const int num_parts,
    const int num_elites,
    const int *elite_partitions,
    const int *hedge_choice_elite,
    const int *node_hyperedges,
    const int *node_offsets,
    const unsigned long long *edge_flags_all_best,
    int *partition_out
) {
    (void)num_elites;

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int np = (num_parts < 64) ? num_parts : 64;
    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int deg = end - start;
    int used_degree = (deg > 768) ? 768 : deg;

    unsigned short votes[64];
    for (int p = 0; p < np; p++) votes[p] = 0;

    if (used_degree > 0) {
        for (int j = 0; j < used_degree; j++) {
            int rel = (int)(((long long)j * deg) / used_degree);
            int hedge = node_hyperedges[start + rel];
            int elite_id = hedge_choice_elite[hedge];
            long long idx = (long long)elite_id * (long long)num_nodes + (long long)node;
            int part = elite_partitions[idx];
            if ((unsigned)part < (unsigned)np) {
                votes[part]++;
            }
        }
    }

    int maxv = 0;
    unsigned long long cand_mask = 0ULL;
    for (int p = 0; p < np; p++) {
        int v = (int)votes[p];
        if (v > maxv) {
            maxv = v;
            cand_mask = (1ULL << p);
        } else if (v == maxv && v > 0) {
            cand_mask |= (1ULL << p);
        }
    }

    int best_part = 0;
    if (cand_mask == 0ULL) {
        best_part = 0;
    } else if ((cand_mask & (cand_mask - 1ULL)) == 0ULL) {
        best_part = __ffsll(cand_mask) - 1;
    } else {
        int best_score = -2147483647;
        unsigned long long tmp = cand_mask;
        while (tmp) {
            int p = __ffsll(tmp) - 1;
            tmp &= (tmp - 1ULL);

            unsigned long long bit = 1ULL << p;
            int score = 0;
            if (used_degree > 0) {
                for (int j = 0; j < used_degree; j++) {
                    int rel = (int)(((long long)j * deg) / used_degree);
                    int hedge = node_hyperedges[start + rel];
                    score += ((edge_flags_all_best[hedge] & bit) != 0ULL);
                }
            }

            if (score > best_score || (score == best_score && p < best_part)) {
                best_score = score;
                best_part = p;
            }
        }
    }

    partition_out[node] = best_part;
}



extern "C" __global__ void hyperedge_clustering_10k(
    const int num_hyperedges,
    const int num_clusters,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int k_hashes,
    const int *hash_a,
    const int *hash_b,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;

    int start = hyperedge_offsets[hedge];
    int end   = hyperedge_offsets[hedge + 1];
    
    unsigned int min_hashes[4];
    for (int j = 0; j < k_hashes; j++) {
        min_hashes[j] = 0xFFFFFFFFu;
    }

    for (int k = start; k < end; k++) {
        int node = hyperedge_nodes[k];
        for (int j = 0; j < k_hashes; j++) {
            long long a = hash_a[j];
            long long b = hash_b[j];
            long long h = a * (long long)node + b;
            unsigned int hv = (unsigned int)(h & 0x7FFFFFFF);
            if (hv < min_hashes[j]) {
                min_hashes[j] = hv;
            }
        }
    }

    unsigned int combined = 0;
    for (int j = 0; j < k_hashes; j++) {
        combined ^= min_hashes[j];
        combined = combined * 1103515245u + 12345u;
    }
    int cluster = (int)(combined % (unsigned int)num_clusters);
    hyperedge_clusters[hedge] = cluster;
}

extern "C" __global__ void compute_node_preferences_10k(
    const int num_nodes,
    const int num_parts,
    const int num_hedge_clusters,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_clusters,
    const int *hyperedge_offsets,
    const int restart_id,
    const unsigned int random_seed,
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

                if (restart_id > 0) {
                    unsigned int hash = (unsigned int)node * 1664525u
                                      + (unsigned int)cluster * 1013904223u
                                      + (unsigned int)restart_id * 12345u
                                      + random_seed;
                    hash ^= (hash >> 16);
                    hash *= 0x85ebca6bu;
                    hash ^= (hash >> 13);
                    hash *= 0xc2b2ae35u;
                    hash ^= (hash >> 16);
                    cluster_votes[cluster] += (int)(hash & 3u);
                }

                cluster_votes[cluster] += weight;

                if (cluster_votes[cluster] > max_votes ||
                    (cluster_votes[cluster] == max_votes && ((cluster * 17 + node) & 255) < ((best_cluster * 17 + node) & 255))) {
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

extern "C" __global__ void execute_node_assignments_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *sorted_nodes,
    const int *sorted_parts,
    int *partition,
    int *nodes_in_part
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int node = sorted_nodes[i];
    int assigned_part = sorted_parts[i];
    if ((unsigned)node < (unsigned)num_nodes && assigned_part >= 0) {
        partition[node] = assigned_part;
    }
}

extern "C" __global__ void execute_refinement_moves_10k(
    const int num_valid_moves,
    const int *sorted_nodes,
    const int *sorted_parts,
    const int max_part_size,
    int *partition,
    int *nodes_in_part,
    int *moves_executed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_valid_moves) return;

    int node = sorted_nodes[i];
    int target_part = sorted_parts[i];
    if (node >= 0 && target_part >= 0) {
        partition[node] = target_part;
    }
}

extern "C" __global__ void compute_refinement_moves_optimized_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *move_priorities,
    int *num_valid_moves
) {
    __shared__ int shared_nodes_in_part[64];
    __shared__ int block_moves[16];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_move = 0;

    if (node < num_nodes) {
        move_priorities[node] = 0;
        int current_part = __ldg(&partition[node]);
        if ((unsigned)current_part < (unsigned)num_parts && shared_nodes_in_part[current_part] > 1) {
            int start = __ldg(&node_offsets[node]);
            int end   = __ldg(&node_offsets[node + 1]);
            int node_degree = end - start;

            if (node_degree > 0) {
                int degree_weight = node_degree > 255 ? 255 : node_degree;
                unsigned long long current_bit = 1ULL << current_part;

                unsigned int part_info[64];
                int np = (num_parts < 64) ? num_parts : 64;
                for (int p = 0; p < np; p++) part_info[p] = 0;

                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                for (int j = start; j < end; j++) {
                    int hyperedge = __ldg(&node_hyperedges[j]);
                    unsigned long long flags_all = __ldg(&edge_flags_all[hyperedge]);
                    unsigned long long flags_double = __ldg(&edge_flags_double[hyperedge]);
                    unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

                    if (mask & current_bit) count_current_present++;

                    unsigned long long f_all = mask & ~current_bit;
                    unsigned long long f_dbl = flags_double & ~current_bit;

                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        part_info[bit] += 1;
                        cand_mask |= 1ULL << bit;
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        part_info[bit] += 65536;
                    }
                }

                int best_gain = -999999;
                int best_target = current_part;

                while (cand_mask) {
                    int target_part = __ffsll(cand_mask) - 1;
                    cand_mask &= (cand_mask - 1);

                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count = part_info[target_part] & 0xFFFF;
                    int p_double = part_info[target_part] >> 16;

                    int basic_gain = p_count - count_current_present;
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    bool better = (total_gain > best_gain);
                    if (!better && total_gain == best_gain) {
                        int best_double = part_info[best_target] >> 16;
                        if (p_double > best_double) {
                            better = true;
                        } else if (p_double == best_double) {
                            int best_target_size = shared_nodes_in_part[best_target];
                            if (target_size < best_target_size) {
                                better = true;
                            } else if (target_size == best_target_size) {
                                int hash_tgt = (target_part * 17 + node) & 63;
                                int hash_best = (best_target * 17 + node) & 63;
                                if (hash_tgt < hash_best) better = true;
                            }
                        }
                    }

                    if (better) {
                        best_gain = total_gain;
                        best_target = target_part;
                    }
                }

                if (best_gain >= -1 && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
                    valid_move = 1;
                }
            }
        }
    }

    unsigned active = __activemask();
    for (int offset = 16; offset > 0; offset /= 2) {
        valid_move += __shfl_down_sync(active, valid_move, offset);
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        block_moves[warp_id] = valid_move;
    }
    __syncthreads();

    if (warp_id == 0 && lane < (blockDim.x + 31) / 32) {
        valid_move = block_moves[lane];
    } else {
        valid_move = 0;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        valid_move += __shfl_down_sync(active, valid_move, offset);
    }

    if (threadIdx.x == 0) {
        num_valid_moves[blockIdx.x] = valid_move;
    }
}

static __device__ __forceinline__ unsigned long long shfl_xor_u64_10k(
    unsigned mask,
    unsigned long long v,
    int lane_mask
) {
    unsigned lo = __shfl_xor_sync(mask, (unsigned)(v & 0xFFFFFFFFu), lane_mask);
    unsigned hi = __shfl_xor_sync(mask, (unsigned)(v >> 32), lane_mask);
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}

extern "C" __global__ void precompute_edge_flags_10k(
    const int num_hyperedges,
    const int num_nodes,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_hyperedges) {
        int start = __ldg(&hyperedge_offsets[tid]);
        int end   = __ldg(&hyperedge_offsets[tid + 1]);
        int hedge_size = end - start;

        if (hedge_size <= warpSize) {
            unsigned long long flags_all = 0ULL;
            unsigned long long flags_double = 0ULL;

            for (int k = start; k < end; k++) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldg(&partition[node]);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        flags_double |= (flags_all & bit);
                        flags_all |= bit;
                    }
                }
            }

            edge_flags_all[tid] = flags_all;
            edge_flags_double[tid] = flags_double;
        }
    }

    int lane = threadIdx.x & 31;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
    int total_warps = gridDim.x * warps_per_block;
    unsigned active = __activemask();

    for (int hedge = global_warp; hedge < num_hyperedges; hedge += total_warps) {
        int start = 0;
        int end = 0;
        int hedge_size = 0;

        if (lane == 0) {
            start = __ldg(&hyperedge_offsets[hedge]);
            end   = __ldg(&hyperedge_offsets[hedge + 1]);
            hedge_size = end - start;
        }

        start = __shfl_sync(active, start, 0);
        end   = __shfl_sync(active, end, 0);
        hedge_size = __shfl_sync(active, hedge_size, 0);

        if (hedge_size <= warpSize) {
            continue;
        }

        unsigned long long local_all = 0ULL;
        unsigned long long local_double = 0ULL;

        for (int k = start + lane; k < end; k += warpSize) {
            int node = __ldg(&hyperedge_nodes[k]);
            if ((unsigned)node < (unsigned)num_nodes) {
                int part = __ldg(&partition[node]);
                if ((unsigned)part < 64u) {
                    unsigned long long bit = 1ULL << part;
                    local_double |= (local_all & bit);
                    local_all |= bit;
                }
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            unsigned long long other_all = shfl_xor_u64_10k(active, local_all, offset);
            unsigned long long other_double = shfl_xor_u64_10k(active, local_double, offset);
            local_double |= other_double | (local_all & other_all);
            local_all |= other_all;
        }

        if (lane == 0) {
            edge_flags_all[hedge] = local_all;
            edge_flags_double[hedge] = local_double;
        }
    }
}

extern "C" __global__ void compute_connectivity_10k(
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

extern "C" __global__ void perturb_solution_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int perturb_strength,
    int *partition,
    int *nodes_in_part,
    unsigned long long seed
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

            if (target_part != current_part &&
                nodes_in_part[target_part] < max_part_size) {
                partition[node] = target_part;
                nodes_in_part[current_part]--;
                nodes_in_part[target_part]++;
                moves_made++;
            }
        }
    }
}

extern "C" __global__ void crossover_partitions_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition_a,
    const int *partition_b,
    const unsigned long long *edge_flags_a,
    const unsigned long long *edge_flags_b,
    int *child_partition
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int part_a = __ldg(&partition_a[node]);
    int part_b = __ldg(&partition_b[node]);

    int chosen_part;
    if (part_a == part_b) {
        chosen_part = part_a;
    } else {
        int start = __ldg(&node_offsets[node]);
        int end   = __ldg(&node_offsets[node + 1]);
        int degree = end - start;
        int used_deg = degree > 64 ? 64 : degree;

        int cost_a = 0, cost_b = 0;
        int agreed_a = 0, agreed_b = 0;

        for (int j = 0; j < used_deg; j++) {
            int rel = (int)(((long long)j * degree) / used_deg);
            int hedge = __ldg(&node_hyperedges[start + rel]);
            
            int h_start = __ldg(&hyperedge_offsets[hedge]);
            int h_end   = __ldg(&hyperedge_offsets[hedge + 1]);
            int hedge_size = h_end - h_start;
            int weight = 100 / (hedge_size + 1);

            if (hedge_size <= 64) {
                for (int k = h_start; k < h_end; k++) {
                    int nbr = __ldg(&hyperedge_nodes[k]);
                    if (nbr != node) {
                        int pa = __ldg(&partition_a[nbr]);
                        int pb = __ldg(&partition_b[nbr]);
                        if (pa == pb) {
                            if (pa == part_a) agreed_a += weight;
                            else if (pa == part_b) agreed_b += weight;
                        }
                    }
                }
            }

            unsigned long long fa = __ldg(&edge_flags_a[hedge]);
            unsigned long long fb = __ldg(&edge_flags_b[hedge]);
            
            int parts_a = __popcll(fa);
            int parts_b = __popcll(fb);
            
            int penalty_a = (parts_a <= 1) ? 0 : (parts_a * parts_a);
            int penalty_b = (parts_b <= 1) ? 0 : (parts_b * parts_b);
            
            cost_a += penalty_a * weight;
            cost_b += penalty_b * weight;
        }

        if (agreed_a > agreed_b) {
            chosen_part = part_a;
        } else if (agreed_b > agreed_a) {
            chosen_part = part_b;
        } else {
            chosen_part = (cost_a <= cost_b) ? part_a : part_b;
        }
    }

    if (chosen_part < 0 || chosen_part >= num_parts) chosen_part = node % num_parts;
    child_partition[node] = chosen_part;
}



extern "C" __global__ void compute_hyperedge_centric_moves_10k(
    const int num_hyperedges,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    const int *node_offsets,
    int *hedge_moves
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;

    int base_idx = hedge * 4;
    hedge_moves[base_idx + 0] = 0;
    hedge_moves[base_idx + 1] = 0;
    hedge_moves[base_idx + 2] = 0;
    hedge_moves[base_idx + 3] = 0;

    unsigned long long flags_all = __ldg(&edge_flags_all[hedge]);
    int span = __popcll(flags_all);

    if (span >= 2 && span <= 8) {
        int start = __ldg(&hyperedge_offsets[hedge]);
        int end   = __ldg(&hyperedge_offsets[hedge + 1]);

        int part_counts[64];
        
        unsigned long long temp_flags = flags_all;
        while (temp_flags) {
            int p = __ffsll(temp_flags) - 1;
            temp_flags &= temp_flags - 1;
            part_counts[p] = 0;
        }

        for (int k = start; k < end; k++) {
            int node = __ldg(&hyperedge_nodes[k]);
            int p = __ldg(&partition[node]);
            if (p >= 0 && p < 64 && ((flags_all >> p) & 1)) {
                part_counts[p]++;
            }
        }

        int max_p = -1;
        int max_count = -1;
        int min_p = -1;
        int min_count = 999999;

        temp_flags = flags_all;
        while (temp_flags) {
            int p = __ffsll(temp_flags) - 1;
            temp_flags &= temp_flags - 1;
            
            int c = part_counts[p];
            if (c > max_count) {
                max_count = c;
                max_p = p;
            }
            if (c > 0 && c <= min_count) {
                min_count = c;
                min_p = p;
            }
        }

        if (min_p != -1 && max_p != -1 && min_p != max_p && min_count <= 4) {
            int move_idx = 0;
            for (int k = start; k < end; k++) {
                int node = __ldg(&hyperedge_nodes[k]);
                if (__ldg(&partition[node]) == min_p) {
                    hedge_moves[base_idx + move_idx] = ((node + 1) << 6) | max_p;
                    move_idx++;
                    if (move_idx >= 4) break;
                }
            }
        }
    }
}



extern "C" __global__ void compute_swap_gains_extended_10k(
    const int num_nodes,
    const int num_parts,
    const int neg_gain_thresh,
    const int global_round,
    const int *node_tabu_until,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *swap_gains
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    swap_gains[node * 4 + 0] = 0;
    swap_gains[node * 4 + 1] = 0;
    swap_gains[node * 4 + 2] = 0;
    swap_gains[node * 4 + 3] = 0;

    int current_part = __ldg(&partition[node]);
    if ((unsigned)current_part >= (unsigned)num_parts) return;

    bool is_tabu = (__ldg(&node_tabu_until[node]) > global_round);

    int start = __ldg(&node_offsets[node]);
    int end   = __ldg(&node_offsets[node + 1]);
    int node_degree = end - start;
    int used_degree = node_degree > 768 ? 768 : node_degree;
    if (used_degree <= 0) return;

    const int sample_step = node_degree / used_degree;
    const int sample_rem = node_degree - sample_step * used_degree;
    int sample_rel = 0;
    int sample_acc = 0;

    unsigned long long current_bit = 1ULL << current_part;

    const int np=min(num_parts,64);
    unsigned long long pc0=0ULL, dc0=0ULL;
    unsigned long long pc1=0ULL, dc1=0ULL;
    unsigned long long pc2=0ULL, dc2=0ULL;
    unsigned long long pc3=0ULL, dc3=0ULL;
    unsigned long long pc4=0ULL, dc4=0ULL;
    unsigned long long pc5=0ULL, dc5=0ULL;
    unsigned long long pc6=0ULL, dc6=0ULL;
    unsigned long long pc7=0ULL, dc7=0ULL;
    unsigned long long pc8=0ULL, dc8=0ULL;
    unsigned long long pc9=0ULL, dc9=0ULL;
    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;

    for (int j = 0; j < used_degree; j++) {
        int hyperedge = __ldg(&node_hyperedges[start + sample_rel]);

        unsigned long long flags_all   = __ldg(&edge_flags_all[hyperedge]);
        unsigned long long flags_double= __ldg(&edge_flags_double[hyperedge]);
        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

        if (mask & current_bit) count_current_present++;

        unsigned long long f_all = mask & ~current_bit;
        unsigned long long f_dbl = flags_double & ~current_bit;

        cand_mask |= f_all;
        { unsigned long long carry=f_all,next;
          next=pc0&carry;pc0^=carry;carry=next;
          next=pc1&carry;pc1^=carry;carry=next;
          next=pc2&carry;pc2^=carry;carry=next;
          next=pc3&carry;pc3^=carry;carry=next;
          next=pc4&carry;pc4^=carry;carry=next;
          next=pc5&carry;pc5^=carry;carry=next;
          next=pc6&carry;pc6^=carry;carry=next;
          next=pc7&carry;pc7^=carry;carry=next;
          next=pc8&carry;pc8^=carry;carry=next;
          next=pc9&carry;pc9^=carry;carry=next;
        }
        { unsigned long long carry=f_dbl,next;
          next=dc0&carry;dc0^=carry;carry=next;
          next=dc1&carry;dc1^=carry;carry=next;
          next=dc2&carry;dc2^=carry;carry=next;
          next=dc3&carry;dc3^=carry;carry=next;
          next=dc4&carry;dc4^=carry;carry=next;
          next=dc5&carry;dc5^=carry;carry=next;
          next=dc6&carry;dc6^=carry;carry=next;
          next=dc7&carry;dc7^=carry;carry=next;
          next=dc8&carry;dc8^=carry;carry=next;
          next=dc9&carry;dc9^=carry;carry=next;
        }
        sample_rel += sample_step;
        sample_acc += sample_rem;
        if (sample_acc >= used_degree) {
            sample_acc -= used_degree;
            sample_rel++;
        }
    }

    int degree_scaled_thresh = neg_gain_thresh + node_degree / 20;
    int min_gain = -degree_scaled_thresh;
    if (is_tabu) {
        min_gain = 1;  
    }

    unsigned long long active=cand_mask & (np==64?~0ULL:((1ULL<<np)-1ULL));
    for(int slot=0;slot<4 && active;slot++){
        unsigned long long pick=active;int count=0;
        { unsigned long long subset=pick&pc9;if(subset){pick=subset;count|=512;} }
        { unsigned long long subset=pick&pc8;if(subset){pick=subset;count|=256;} }
        { unsigned long long subset=pick&pc7;if(subset){pick=subset;count|=128;} }
        { unsigned long long subset=pick&pc6;if(subset){pick=subset;count|=64;} }
        { unsigned long long subset=pick&pc5;if(subset){pick=subset;count|=32;} }
        { unsigned long long subset=pick&pc4;if(subset){pick=subset;count|=16;} }
        { unsigned long long subset=pick&pc3;if(subset){pick=subset;count|=8;} }
        { unsigned long long subset=pick&pc2;if(subset){pick=subset;count|=4;} }
        { unsigned long long subset=pick&pc1;if(subset){pick=subset;count|=2;} }
        { unsigned long long subset=pick&pc0;if(subset){pick=subset;count|=1;} }
        { unsigned long long subset=pick&dc9;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc8;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc7;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc6;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc5;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc4;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc3;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc2;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc1;if(subset)pick=subset; }
        { unsigned long long subset=pick&dc0;if(subset)pick=subset; }
        int target=-1,hash_min=999;
        while(pick){int p=__ffsll(pick)-1;pick&=pick-1;int hash=(p*17+node)&63;
            if(hash<hash_min){hash_min=hash;target=p;}}
        int gain=count-count_current_present;
        if(used_degree<node_degree)gain=(gain*node_degree)/used_degree;
        if(gain<min_gain)break;
        int g=max(-32768,min(32767,gain));
        swap_gains[node*4+slot]=((int)(unsigned short)(short)g<<16)|target;
        active&=~(1ULL<<target);
    }

}

extern "C" __global__ void perturb_path_relink_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int fraction_percent,
    const int *node_offsets,
    const int *best_partition,
    int *partition,
    int *nodes_in_part,
    unsigned long long seed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long state = seed;
        int target_moves = (num_nodes * fraction_percent) / 100;
        if (target_moves < 1) target_moves = 1;

        int moves_made = 0;
        int diff_count = 0;

        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int start_node = (int)(state % (unsigned long long)num_nodes);

        for (int i = 0; i < num_nodes && moves_made < target_moves; i++) {
            int node = start_node + i;
            if (node >= num_nodes) node -= num_nodes;

            int cur_part = partition[node];
            int best_part = best_partition[node];

            if (cur_part == best_part) continue;
            diff_count++;

            int deg = node_offsets[node + 1] - node_offsets[node];
            int scaled_fraction = fraction_percent + deg * 2;
            if (scaled_fraction > 95) scaled_fraction = 95;

            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            unsigned long long threshold = ((unsigned long long)scaled_fraction * 0x100000000ULL) / 100ULL;
            if ((state >> 32) >= threshold) continue;

            if (cur_part < 0 || cur_part >= num_parts) continue;
            if (best_part < 0 || best_part >= num_parts) continue;
            if (nodes_in_part[cur_part] <= 1) continue;
            if (nodes_in_part[best_part] >= max_part_size) continue;

            partition[node] = best_part;
            nodes_in_part[cur_part]--;
            nodes_in_part[best_part]++;
            moves_made++;
        }
    }
}

extern "C" __global__ void perturb_guided_10k(
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

            int majority_part = 0;
            for (int p = 1; p < np; p++) {
                if (part_count[p] > part_count[majority_part]) majority_part = p;
            }

            int num_parts_present = 0;
            for (int p = 0; p < np; p++) {
                if (part_count[p] > 0) num_parts_present++;
            }
            if (num_parts_present <= 1) continue;

            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                int cur_part = partition[node];
                if (cur_part == majority_part) continue;
                if (nodes_in_part[cur_part] <= 1) continue;
                if (nodes_in_part[majority_part] >= max_part_size) continue;

                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                if ((state & 3) != 0) continue;

                partition[node] = majority_part;
                nodes_in_part[cur_part]--;
                nodes_in_part[majority_part]++;
            }
        }
    }
}

extern "C" __global__ void perturb_ruin_recreate_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    int *partition,
    int *nodes_in_part,
    unsigned long long seed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long state = seed;
        int np = (num_parts < 64) ? num_parts : 64;
        
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int target_part = (int)(state % (unsigned long long)num_parts);
        
        for (int node = 0; node < num_nodes; node++) {
            if (partition[node] == target_part && nodes_in_part[target_part] > 1) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                if ((state & 1) == 0) continue; 
                
                int start = node_offsets[node];
                int end = node_offsets[node + 1];
                
                int part_counts[64];
                for (int p = 0; p < np; p++) part_counts[p] = 0;
                
                int used_deg = end - start;
                if (used_deg > 30) used_deg = 30;
                
                for (int j = 0; j < used_deg; j++) {
                    int hedge = node_hyperedges[start + j];
                    int h_start = hyperedge_offsets[hedge];
                    int h_end = hyperedge_offsets[hedge + 1];
                    int h_size = h_end - h_start;
                    if (h_size > 20) continue; 
                    
                    for (int k = h_start; k < h_end; k++) {
                        int nbr = hyperedge_nodes[k];
                        if (nbr != node) {
                            int n_part = partition[nbr];
                            if (n_part >= 0 && n_part < np) {
                                part_counts[n_part]++;
                            }
                        }
                    }
                }
                
                int best_p = -1;
                int max_c = -1;
                for (int p = 0; p < np; p++) {
                    if (p != target_part && nodes_in_part[p] < max_part_size) {
                        if (part_counts[p] > max_c) {
                            max_c = part_counts[p];
                            best_p = p;
                        }
                    }
                }
                
                if (best_p == -1) {
                    for(int attempt = 0; attempt < 10; attempt++) {
                        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                        int rp = (int)(state % (unsigned long long)num_parts);
                        if (rp != target_part && nodes_in_part[rp] < max_part_size) {
                            best_p = rp;
                            break;
                        }
                    }
                }
                
                if (best_p != -1 && best_p != target_part) {
                    partition[node] = best_p;
                    nodes_in_part[target_part]--;
                    nodes_in_part[best_p]++;
                }
            }
        }
    }
}

extern "C" __global__ void perturb_hubs_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    int *partition,
    int *nodes_in_part,
    unsigned long long seed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long state = seed;
        
        int num_hubs = num_nodes / 100;
        if (num_hubs < 1) num_hubs = 1;
        
        int moves_made = 0;
        int max_moves = num_nodes / 20; 
        
        for (int attempt = 0; attempt < num_hubs; attempt++) {            
            int best_hub = -1;
            int max_deg = -1;
            for (int s = 0; s < 20; s++) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                int node = (int)(state % (unsigned long long)num_nodes);
                int deg = node_offsets[node + 1] - node_offsets[node];
                if (deg > max_deg) {
                    max_deg = deg;
                    best_hub = node;
                }
            }
            
            if (best_hub != -1) {                
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                int new_part = (int)(state % (unsigned long long)num_parts);
                int cur_part = partition[best_hub];
                if (cur_part != new_part && cur_part >= 0 && cur_part < num_parts && nodes_in_part[cur_part] > 1 && nodes_in_part[new_part] < max_part_size) {
                    partition[best_hub] = new_part;
                    nodes_in_part[cur_part]--;
                    nodes_in_part[new_part]++;
                    moves_made++;
                }
                
                int start = node_offsets[best_hub];
                int end = node_offsets[best_hub + 1];
                int deg = end - start;
                int used_deg = deg > 20 ? 20 : deg;
                
                for (int j = 0; j < used_deg; j++) {
                    int hedge = node_hyperedges[start + j];
                    int h_start = hyperedge_offsets[hedge];
                    int h_end = hyperedge_offsets[hedge + 1];
                    int h_size = h_end - h_start;
                    if (h_size > 10) continue;
                    
                    for (int k = h_start; k < h_end; k++) {
                        int neighbor = hyperedge_nodes[k];
                        if (neighbor == best_hub) continue;
                        
                        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                        if ((state & 3) != 0) continue; 
                        
                        int n_part = partition[neighbor];
                        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                        int n_new_part = (int)(state % (unsigned long long)num_parts);
                        
                        if (n_part != n_new_part && n_part >= 0 && n_part < num_parts && nodes_in_part[n_part] > 1 && nodes_in_part[n_new_part] < max_part_size) {
                            partition[neighbor] = n_new_part;
                            nodes_in_part[n_part]--;
                            nodes_in_part[n_new_part]++;
                            moves_made++;
                        }
                    }
                }
            }
            if (moves_made >= max_moves) break;
        }
    }
}

extern "C" __global__ void balance_final_10k(
    const int num_nodes,
    const int num_parts,
    const int min_part_size,
    const int max_part_size,
    const int *node_offsets,
    const int *node_hyperedges,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *partition,
    int *nodes_in_part
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int part = 0; part < num_parts; part++) {
            while (nodes_in_part[part] < min_part_size) {
                int best_node = -1;
                int best_gain = -999999;
                
                for (int node = 0; node < num_nodes; node++) {
                    int p = partition[node];
                    if (p != part && nodes_in_part[p] > min_part_size) {
                        int start = node_offsets[node];
                        int end = node_offsets[node + 1];
                        int gain = 0;
                        unsigned long long p_bit = 1ULL << p;
                        unsigned long long tgt_bit = 1ULL << part;
                        
                        for (int j = start; j < end; j++) {
                            int hedge = node_hyperedges[j];
                            unsigned long long flags = edge_flags_all[hedge];
                            unsigned long long dbl = edge_flags_double[hedge];
                            
                            int p_alone = ((dbl & p_bit) == 0) ? 1 : 0;
                            int tgt_present = ((flags & tgt_bit) != 0) ? 1 : 0;
                            
                            gain += p_alone - (1 - tgt_present);
                        }
                        
                        int deg = end - start;
                        if (gain > best_gain) {
                            best_gain = gain;
                            best_node = node;
                        } else if (gain == best_gain && best_node != -1) {
                            int best_deg = node_offsets[best_node + 1] - node_offsets[best_node];
                            if (deg < best_deg) {
                                best_node = node;
                            }
                        }
                    }
                }
                
                if (best_node != -1) {
                    int p = partition[best_node];
                    partition[best_node] = part;
                    nodes_in_part[p]--;
                    nodes_in_part[part]++;
                } else {
                    break;
                }
            }
        }

        for (int part = 0; part < num_parts; part++) {
            while (nodes_in_part[part] > max_part_size) {
                int best_node = -1;
                int best_target = -1;
                int best_gain = -999999;
                
                for (int node = 0; node < num_nodes; node++) {
                    if (partition[node] == part) {
                        int start = node_offsets[node];
                        int end = node_offsets[node + 1];
                        unsigned long long p_bit = 1ULL << part;
                        
                        for (int tgt = 0; tgt < num_parts; tgt++) {
                            if (tgt != part && nodes_in_part[tgt] < max_part_size) {
                                int gain = 0;
                                unsigned long long tgt_bit = 1ULL << tgt;
                                
                                for (int j = start; j < end; j++) {
                                    int hedge = node_hyperedges[j];
                                    unsigned long long flags = edge_flags_all[hedge];
                                    unsigned long long dbl = edge_flags_double[hedge];
                                    
                                    int p_alone = ((dbl & p_bit) == 0) ? 1 : 0;
                                    int tgt_present = ((flags & tgt_bit) != 0) ? 1 : 0;
                                    
                                    gain += p_alone - (1 - tgt_present);
                                }
                                
                                int deg = end - start;
                                if (gain > best_gain) {
                                    best_gain = gain;
                                    best_node = node;
                                    best_target = tgt;
                                } else if (gain == best_gain && best_node != -1) {
                                    int best_deg = node_offsets[best_node + 1] - node_offsets[best_node];
                                    if (deg < best_deg) {
                                        best_node = node;
                                        best_target = tgt;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (best_node != -1 && best_target != -1) {
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

extern "C" __global__ void balance_find_best_under_10k(
    const int num_nodes,
    const int num_parts,
    const int target_part,
    const int min_part_size,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *partition,
    const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *best_node_out,
    int *best_gain_out,
    int *best_deg_out
) {
    __shared__ int warp_gains[32];
    __shared__ int warp_degs[32];
    __shared__ int warp_nodes[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_best_node = -1;
    int local_best_gain = -2147483647;
    int local_best_deg = 2147483647;

    unsigned long long tgt_bit = 1ULL << target_part;

    for (int node = idx; node < num_nodes; node += stride) {
        int src = __ldg(&partition[node]);
        if (src == target_part) continue;
        if (__ldg(&nodes_in_part[src]) <= min_part_size) continue;
        int start = __ldg(&node_offsets[node]);
        int end   = __ldg(&node_offsets[node + 1]);
        unsigned long long p_bit = 1ULL << src;
        int gain = 0;
        for (int j = start; j < end; j++) {
            int hedge = __ldg(&node_hyperedges[j]);
            unsigned long long flags = __ldg(&edge_flags_all[hedge]);
            unsigned long long dbl   = __ldg(&edge_flags_double[hedge]);
            int p_alone = ((dbl & p_bit) == 0) ? 1 : 0;
            int tgt_present = ((flags & tgt_bit) != 0) ? 1 : 0;
            gain += p_alone - (1 - tgt_present);
        }
        int deg = end - start;
        if (gain > local_best_gain || (gain == local_best_gain && deg < local_best_deg)) {
            local_best_gain = gain;
            local_best_node = node;
            local_best_deg = deg;
        }
    }

    unsigned mask = __activemask();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_gain = __shfl_down_sync(mask, local_best_gain, offset);
        int other_deg  = __shfl_down_sync(mask, local_best_deg, offset);
        int other_node = __shfl_down_sync(mask, local_best_node, offset);
        if (other_gain > local_best_gain || (other_gain == local_best_gain && other_deg < local_best_deg)) {
            local_best_gain = other_gain;
            local_best_deg  = other_deg;
            local_best_node = other_node;
        }
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        warp_gains[warp_id] = local_best_gain;
        warp_degs[warp_id]  = local_best_deg;
        warp_nodes[warp_id] = local_best_node;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int best_n = -1;
        int best_g = -2147483647;
        int best_d = 2147483647;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            int g = warp_gains[w];
            int d = warp_degs[w];
            if (g > best_g || (g == best_g && d < best_d)) {
                best_g = g;
                best_n = warp_nodes[w];
                best_d = d;
            }
        }
        best_node_out[blockIdx.x] = best_n;
        best_gain_out[blockIdx.x] = best_g;
        best_deg_out[blockIdx.x]  = best_d;
    }
}

template<int Bits>
static __device__ __forceinline__ int balance_bits_pick10(
 int start,int end,unsigned long long source,unsigned long long available,
 const int*incident,const unsigned long long*all,const unsigned long long*dbl,int*gain){
 unsigned long long planes[Bits];
 #pragma unroll
 for(int b=0;b<Bits;++b)planes[b]=0;
 int repeated=0;
 for(int j=start;j<end;++j){
  int edge=__ldg(incident+j);unsigned long long carry=__ldg(all+edge);
  repeated+=(__ldg(dbl+edge)&source)!=0ULL;
  #pragma unroll
  for(int b=0;b<Bits;++b){unsigned long long next=planes[b]&carry;planes[b]^=carry;carry=next;}
 }
 int count=0;
 #pragma unroll
 for(int b=Bits-1;b>=0;--b){unsigned long long subset=available&planes[b];if(subset){available=subset;count|=1<<b;}}
 *gain=count-repeated;return __ffsll(available)-1;
}
extern "C" __global__ void balance_find_best_over_10k(
    const int num_nodes,
    const int num_parts,
    const int source_part,
    const int max_part_size,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *partition,
    const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *best_node_out,
    int *best_target_out,
    int *best_gain_out,
    int *best_deg_out
) {
    __shared__ int warp_gains[32];
    __shared__ int warp_degs[32];
    __shared__ int warp_nodes[32];
    __shared__ int warp_targets[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_best_node = -1;
    int local_best_target = -1;
    int local_best_gain = -2147483647;
    int local_best_deg = 2147483647;

    unsigned long long src_bit = 1ULL << source_part;
    __shared__ unsigned long long exact_available;
    if(threadIdx.x<32){
        int p=threadIdx.x;
        bool lo=p<num_parts && p!=source_part && __ldg(nodes_in_part+p)<max_part_size;
        bool hi=p+32<num_parts && p+32!=source_part && __ldg(nodes_in_part+p+32)<max_part_size;
        unsigned l=__ballot_sync(0xffffffffu,lo),h=__ballot_sync(0xffffffffu,hi);
        if(p==0)exact_available=(unsigned long long)l|((unsigned long long)h<<32);
    }
    __syncthreads();


    for (int n = idx; n < num_nodes; n += stride) {
        int part = __ldg(&partition[n]);
        if (part != source_part) continue;
        int start = __ldg(&node_offsets[n]);
        int end   = __ldg(&node_offsets[n + 1]);
        int deg = end - start;


        if(num_parts<=64 && deg>=0 && deg<=65535){
            if(exact_available){
                int gain,target;
                if(deg<=255)target=balance_bits_pick10<8>(start,end,src_bit,exact_available,node_hyperedges,edge_flags_all,edge_flags_double,&gain);
                else target=balance_bits_pick10<16>(start,end,src_bit,exact_available,node_hyperedges,edge_flags_all,edge_flags_double,&gain);
                if(gain>local_best_gain || (gain==local_best_gain && deg<local_best_deg)){
                    local_best_gain=gain;local_best_node=n;local_best_target=target;local_best_deg=deg;
                }
            }
        }else{
        for (int tgt = 0; tgt < num_parts; tgt++) {
            if (tgt == source_part) continue;
            if (__ldg(&nodes_in_part[tgt]) >= max_part_size) continue;
            unsigned long long tgt_bit = 1ULL << tgt;
            int gain = 0;
            for (int j = start; j < end; j++) {
                int hedge = __ldg(&node_hyperedges[j]);
                unsigned long long flags = __ldg(&edge_flags_all[hedge]);
                unsigned long long dbl   = __ldg(&edge_flags_double[hedge]);
                int p_alone = ((dbl & src_bit) == 0) ? 1 : 0;
                int tgt_present = ((flags & tgt_bit) != 0) ? 1 : 0;
                gain += p_alone - (1 - tgt_present);
            }
            if (gain > local_best_gain || (gain == local_best_gain && deg < local_best_deg)) {
                local_best_gain   = gain;
                local_best_node   = n;
                local_best_target = tgt;
                local_best_deg    = deg;
            }
        }
        }
    }

    unsigned mask = __activemask();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_gain   = __shfl_down_sync(mask, local_best_gain, offset);
        int other_deg    = __shfl_down_sync(mask, local_best_deg, offset);
        int other_node   = __shfl_down_sync(mask, local_best_node, offset);
        int other_target = __shfl_down_sync(mask, local_best_target, offset);
        if (other_gain > local_best_gain || (other_gain == local_best_gain && other_deg < local_best_deg)) {
            local_best_gain   = other_gain;
            local_best_deg    = other_deg;
            local_best_node   = other_node;
            local_best_target = other_target;
        }
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        warp_gains[warp_id]   = local_best_gain;
        warp_degs[warp_id]    = local_best_deg;
        warp_nodes[warp_id]   = local_best_node;
        warp_targets[warp_id] = local_best_target;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int best_n = -1, best_t = -1;
        int best_g = -2147483647;
        int best_d = 2147483647;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            int g = warp_gains[w];
            int d = warp_degs[w];
            if (g > best_g || (g == best_g && d < best_d)) {
                best_g = g;
                best_n = warp_nodes[w];
                best_t = warp_targets[w];
                best_d = d;
            }
        }
        best_node_out[blockIdx.x]   = best_n;
        best_target_out[blockIdx.x] = best_t;
        best_gain_out[blockIdx.x]   = best_g;
        best_deg_out[blockIdx.x]    = best_d;
    }
}

extern "C" __global__ void apply_balance_move_10k(
    int node,
    int new_part,
    int *partition,
    int *nodes_in_part
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int old_part = partition[node];
        partition[node] = new_part;
        nodes_in_part[old_part]--;
        nodes_in_part[new_part]++;
    }
}

constexpr int SWAP_TOPK_THREADS = 128;

static __device__ __forceinline__ void i25_topk_barrier(unsigned *counter,unsigned target){
    __syncthreads();
    if(threadIdx.x==0){
        unsigned value;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;":"=r"(value):"l"(counter):"memory");
        do{
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(counter):"memory");
            if(value<target)__nanosleep(20);
        }while(value<target);
    }
    __syncthreads();
}

// Rare oversized list: deterministic node-order insertion, never arrival order.
static __device__ __noinline__ void i25_topk_large_pair(
    int pair,int nn,int np,const int *parts,const int *gains,int *output){
    const int src=pair/np,tgt=pair%np;
    unsigned long long best[32];
    for(int i=0;i<32;++i)best[i]=~0ULL;
    for(int node=0;node<nn;++node){
        if(__ldg(parts+node)!=src)continue;
        for(int k=0;k<4;++k){
            int v=__ldg(gains+4*node+k);
            if(v==0 || (v&65535)!=tgt)continue;
            int gain=(int)(short)((unsigned)v>>16);
            unsigned long long key=((unsigned long long)(32767-gain)<<32)|(unsigned)node;
            if(key>=best[31])continue;
            int j=31;
            while(j>0 && key<best[j-1]){best[j]=best[j-1];--j;}
            best[j]=key;
        }
    }
    for(int i=0;i<32;++i){
        unsigned long long key=best[i];
        output[pair*64+i*2]=key==~0ULL?-1:(int)(unsigned)key;
        output[pair*64+i*2+1]=key==~0ULL?0:32767-(int)(key>>32);
    }
}
// Output-independent work schedule for top-32: count, prefix, scatter, then
// fixed comparator networks. Arrival order changes only data positions; all
// network loop bounds and compare/store schedules depend on fixed list lengths.
extern "C" __global__ __launch_bounds__(128,2) void compute_swap_topk_prepare_10k(
    const int num_nodes,const int num_parts,const int *partition,
    const int *swap_gains,int *swap_topk,unsigned *grid_barrier,unsigned barrier_base,
    int *metadata,unsigned long long *entries){
    const int np=min(num_parts,64),tid=threadIdx.x;
    if(np<=0)return;
    const int pairs=np*np,gtid=blockIdx.x*blockDim.x+tid,stride=gridDim.x*blockDim.x;
    int *counts=metadata,*offsets=metadata+pairs,*cursor=metadata+2*pairs+1;
    __shared__ union {int scan[4096];unsigned long long keys[4096];} scratch;
    for(int p=gtid;p<pairs;p+=stride){counts[p]=0;cursor[p]=0;}
    __syncthreads();
    for(int node=gtid;node<num_nodes;node+=stride){
        int src=__ldg(partition+node);if((unsigned)src>=(unsigned)np)continue;
        for(int k=0;k<4;++k){
            int value=__ldg(swap_gains+4*node+k);if(value==0)continue;
            int target=value&65535;
            if((unsigned)target<(unsigned)np && target!=src)atomicAdd(counts+src*np+target,1);
        }
    }
    __syncthreads();
    if(blockIdx.x==0){
        int power=1;while(power<pairs)power<<=1;
        for(int i=tid;i<power;i+=blockDim.x)scratch.scan[i]=i<pairs?__ldcg(counts+i):0;
        __syncthreads();
        for(int d=1;d<power;d<<=1){
            for(int j=(tid+1)*2*d-1;j<power;j+=blockDim.x*2*d)scratch.scan[j]+=scratch.scan[j-d];
            __syncthreads();
        }
        if(tid==0){offsets[pairs]=scratch.scan[power-1];scratch.scan[power-1]=0;}
        __syncthreads();
        for(int d=power>>1;d>0;d>>=1){
            for(int j=(tid+1)*2*d-1;j<power;j+=blockDim.x*2*d){int v=scratch.scan[j-d];scratch.scan[j-d]=scratch.scan[j];scratch.scan[j]+=v;}
            __syncthreads();
        }
        for(int i=tid;i<pairs;i+=blockDim.x)offsets[i]=scratch.scan[i];
    }
    __syncthreads();
    for(int node=gtid;node<num_nodes;node+=stride){
        int src=__ldg(partition+node);if((unsigned)src>=(unsigned)np)continue;
        for(int k=0;k<4;++k){
            int value=__ldg(swap_gains+4*node+k);if(value==0)continue;
            int target=value&65535;if((unsigned)target>=(unsigned)np || target==src)continue;
            int gain=(int)(short)((unsigned)value>>16),pair=src*np+target;
            int slot=atomicAdd(cursor+pair,1);
            entries[__ldcg(offsets+pair)+slot]=((unsigned long long)(32767-gain)<<32)|(unsigned)node;
        }
    }
    __syncthreads();

}

extern "C" __global__ __launch_bounds__(128,2) void compute_swap_topk_finish_10k(
    const int num_nodes,const int num_parts,const int *partition,
    const int *swap_gains,int *swap_topk,unsigned *grid_barrier,unsigned barrier_base,
    int *metadata,unsigned long long *entries){
    const int np=min(num_parts,64),tid=threadIdx.x;
    if(np<=0)return;
    const int pairs=np*np,gtid=blockIdx.x*blockDim.x+tid,stride=gridDim.x*blockDim.x;
    int *counts=metadata,*offsets=metadata+pairs,*cursor=metadata+2*pairs+1;
    __shared__ union {int scan[4096];unsigned long long keys[4096];} scratch;
    for(int pair=blockIdx.x;pair<pairs;pair+=gridDim.x){
        if(pair/np==pair%np)continue; // original diagonal bytes stay untouched
        int n=__ldcg(counts+pair),base=__ldcg(offsets+pair);
        if(n>4096){
            if(tid==0)i25_topk_large_pair(pair,num_nodes,np,partition,swap_gains,swap_topk);
            __syncthreads();continue;
        }
        int power=1;while(power<n)power<<=1;
        unsigned long long out=~0ULL;
        if(n<=(int)blockDim.x){
            unsigned long long value=tid<n?__ldcg(entries+base+tid):~0ULL;
            for(int size=2;size<=power;size<<=1){
                for(int distance=size>>1;distance>0;distance>>=1){
                    unsigned long long other;
                    if(distance<32)other=__shfl_xor_sync(0xffffffffu,value,distance);
                    else{scratch.keys[tid]=value;__syncthreads();other=scratch.keys[tid^distance];__syncthreads();}
                    bool take_min=((tid&distance)==0)==((tid&size)==0);
                    value=take_min?min(value,other):max(value,other);
                }
            }
            out=value;
        }else{
            for(int i=tid;i<power;i+=blockDim.x)scratch.keys[i]=i<n?__ldcg(entries+base+i):~0ULL;
            __syncthreads();
            for(int size=2;size<=power;size<<=1){
                for(int distance=size>>1;distance>0;distance>>=1){
                    for(int i=tid;i<power;i+=blockDim.x){
                        int peer=i^distance;
                        if(peer>i){
                            unsigned long long a=scratch.keys[i],b=scratch.keys[peer];
                            unsigned long long lo=min(a,b),hi=max(a,b);
                            scratch.keys[i]=(i&size)==0?lo:hi;
                            scratch.keys[peer]=(i&size)==0?hi:lo;
                        }
                    }
                    __syncthreads();
                }
            }
            if(tid<32)out=scratch.keys[tid];
        }
        if(tid<32){
            swap_topk[pair*64+tid*2]=out==~0ULL?-1:(int)(unsigned)out;
            swap_topk[pair*64+tid*2+1]=out==~0ULL?0:32767-(int)(out>>32);
        }
        __syncthreads();
    }
}



extern "C" __global__ void compute_hedge_consolidation_moves_10k(
    const int num_hyperedges,
    const int num_parts,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    int *move_node_out,
    int *move_part_out,
    int *move_counts
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;

    int start = __ldg(&hyperedge_offsets[hedge]);
    int end = __ldg(&hyperedge_offsets[hedge + 1]);
    int hedge_size = end - start;
    if (hedge_size <= 1) {
        move_counts[hedge] = 0;
        return;
    }

    int np = (num_parts < 64) ? num_parts : 64;
    int part_counts[64] = {0};

    for (int k = start; k < end; k++) {
        int node = __ldg(&hyperedge_nodes[k]);
        int p = __ldg(&partition[node]);
        if (p >= 0 && p < np) {
            part_counts[p]++;
        }
    }

    int majority_part = 0;
    for (int p = 1; p < np; p++) {
        if (part_counts[p] > part_counts[majority_part]) {
            majority_part = p;
        }
    }

    if (part_counts[majority_part] == hedge_size) {
        move_counts[hedge] = 0;
        return;
    }

    const int MAX_MOVES = 16;
    int base = hedge * MAX_MOVES;
    int idx = 0;
    for (int k = start; k < end && idx < MAX_MOVES; k++) {
        int node = __ldg(&hyperedge_nodes[k]);
        int p = __ldg(&partition[node]);
        if (p != majority_part) {
            int gain = (part_counts[p] == 1) ? 1 : 0;
            int packed = (gain << 6) | (majority_part & 63);
            move_node_out[base + idx] = node;
            move_part_out[base + idx] = packed;
            idx++;
        }
    }
    move_counts[hedge] = idx;
}

extern "C" __global__ void compute_best_swap_pairs_10k(
    const int num_nodes,
    const int num_parts,
    const int *partition,
    const int *swap_topk,
    int *best_swap_pair_out
) {
    int np = (num_parts < 64) ? num_parts : 64;
    int src = blockIdx.x;
    if (src >= np) return;
    int tgt = threadIdx.x;
    if (tgt >= np || src == tgt) return;

    int base_out = (src * np + tgt) * 3;
    int base_src_tgt = (src * np + tgt) * 32 * 2;
    int base_tgt_src = (tgt * np + src) * 32 * 2;

    int nodes_ab[32];
    int gains_ab[32];
    int cnt_ab = 0;
    for (int k = 0; k < 32; k++) {
        int node = swap_topk[base_src_tgt + k * 2];
        int gain = swap_topk[base_src_tgt + k * 2 + 1];
        if (node >= 0 && gain > 0) {
            nodes_ab[cnt_ab] = node;
            gains_ab[cnt_ab] = gain;
            cnt_ab++;
        }
    }
    int nodes_ba[32];
    int gains_ba[32];
    int cnt_ba = 0;
    for (int k = 0; k < 32; k++) {
        int node = swap_topk[base_tgt_src + k * 2];
        int gain = swap_topk[base_tgt_src + k * 2 + 1];
        if (node >= 0 && gain > 0) {
            nodes_ba[cnt_ba] = node;
            gains_ba[cnt_ba] = gain;
            cnt_ba++;
        }
    }

    int best_node_a = -1, best_node_b = -1;
    int best_total_gain = 0;
    if (cnt_ab > 0 && cnt_ba > 0) {
        for (int i = 0; i < cnt_ab; i++) {
            int node_a = nodes_ab[i];
            if (__ldg(&partition[node_a]) != src) continue;
            int gain_a = gains_ab[i];
            for (int j = 0; j < cnt_ba; j++) {
                int node_b = nodes_ba[j];
                if (node_a == node_b) continue;
                if (__ldg(&partition[node_b]) != tgt) continue;
                int gain_b = gains_ba[j];
                int total = gain_a + gain_b;
                if (total > best_total_gain) {
                    best_total_gain = total;
                    best_node_a = node_a;
                    best_node_b = node_b;
                }
            }
        }
    }
    best_swap_pair_out[base_out + 0] = best_node_a;
    best_swap_pair_out[base_out + 1] = best_node_b;
    best_swap_pair_out[base_out + 2] = best_total_gain;
}

extern "C" __global__ void polish_exploration_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int num_hyperedges,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    int *partition,
    int *nodes_in_part,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double,
    int *backup_partition,
    int *backup_nip,
    int *best_partition,
    int *best_nip,
    unsigned long long seed
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int np = (num_parts < 64) ? num_parts : 64;

    for (int i = 0; i < num_nodes; i++) backup_partition[i] = partition[i];
    for (int i = 0; i < num_parts; i++) backup_nip[i] = nodes_in_part[i];

    auto compute_connectivity = [&]() {
        int total = 0;
        for (int h = 0; h < num_hyperedges; h++) {
            int start = hyperedge_offsets[h];
            int end   = hyperedge_offsets[h + 1];
            unsigned long long mask = 0ULL;
            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                int part = partition[node];
                if (part >= 0 && part < np) mask |= 1ULL << part;
            }
            int span = __popcll(mask);
            total += (span > 1) ? (span - 1) : 0;
        }
        return total;
    };

    int best_conn = compute_connectivity();
    for (int i = 0; i < num_nodes; i++) best_partition[i] = partition[i];
    for (int i = 0; i < num_parts; i++) best_nip[i] = nodes_in_part[i];

    unsigned long long state = seed;
    const int num_shakes = 5;
    int shake_nodes = (num_nodes * 1) / 100;
    if (shake_nodes < 1) shake_nodes = 1;
    const int refine_passes = 3;

    auto recompute_flags = [&]() {
        for (int h = 0; h < num_hyperedges; h++) {
            int start = hyperedge_offsets[h];
            int end   = hyperedge_offsets[h + 1];
            unsigned long long flags_all = 0ULL;
            unsigned long long flags_double = 0ULL;
            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                int part = partition[node];
                if (part >= 0 && part < np) {
                    unsigned long long bit = 1ULL << part;
                    flags_double |= (flags_all & bit);
                    flags_all |= bit;
                }
            }
            edge_flags_all[h] = flags_all;
            edge_flags_double[h] = flags_double;
        }
    };

    for (int shake = 0; shake < num_shakes; shake++) {
        for (int i = 0; i < num_nodes; i++) partition[i] = backup_partition[i];
        for (int i = 0; i < num_parts; i++) nodes_in_part[i] = backup_nip[i];

        for (int k = 0; k < shake_nodes; k++) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            int node = (int)(state % (unsigned long long)num_nodes);
            int cur = partition[node];
            if (cur < 0 || cur >= np) continue;
            if (nodes_in_part[cur] <= 1) continue;
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            int tgt = (int)(state % (unsigned long long)num_parts);
            if (tgt == cur) continue;
            if (nodes_in_part[tgt] >= max_part_size) continue;
            partition[node] = tgt;
            nodes_in_part[cur]--;
            nodes_in_part[tgt]++;
        }

        recompute_flags();

        for (int pass = 0; pass < refine_passes; pass++) {
            bool any_move = false;
            for (int node = 0; node < num_nodes; node++) {
                int cur = partition[node];
                if (cur < 0 || cur >= np) continue;
                int start = node_offsets[node];
                int end   = node_offsets[node + 1];
                if (start >= end) continue;

                unsigned long long cur_bit = 1ULL << cur;
                unsigned int part_info[64] = {0};
                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                for (int j = start; j < end; j++) {
                    int hedge = node_hyperedges[j];
                    unsigned long long fa = edge_flags_all[hedge];
                    unsigned long long fd = edge_flags_double[hedge];
                    unsigned long long mask = (fa & ~cur_bit) | (fd & cur_bit);
                    if (mask & cur_bit) count_current_present++;
                    unsigned long long f_all = mask & ~cur_bit;
                    unsigned long long f_dbl = fd & ~cur_bit;
                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        part_info[bit] += 1;
                        cand_mask |= 1ULL << bit;
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        part_info[bit] += 65536;
                    }
                }

                int best_gain = 0;
                int best_tgt = cur;
                unsigned long long tmp = cand_mask;
                while (tmp) {
                    int tgt = __ffsll(tmp) - 1;
                    tmp &= (tmp - 1);
                    if ((unsigned)tgt >= (unsigned)num_parts) continue;
                    if (nodes_in_part[tgt] >= max_part_size) continue;
                    int p_count = part_info[tgt] & 0xFFFF;
                    int basic_gain = p_count - count_current_present;
                    if (basic_gain > best_gain) {
                        best_gain = basic_gain;
                        best_tgt = tgt;
                    } else if (basic_gain == best_gain && basic_gain > 0) {
                        if (nodes_in_part[tgt] < nodes_in_part[best_tgt]) {
                            best_tgt = tgt;
                        }
                    }
                }

                if (best_gain > 0) {
                    partition[node] = best_tgt;
                    nodes_in_part[cur]--;
                    nodes_in_part[best_tgt]++;
                    any_move = true;
                }
            }
            if (any_move) {
                recompute_flags();
            } else {
                break;
            }
        }

        int conn = compute_connectivity();
        if (conn < best_conn) {
            best_conn = conn;
            for (int i = 0; i < num_nodes; i++) best_partition[i] = partition[i];
            for (int i = 0; i < num_parts; i++) best_nip[i] = nodes_in_part[i];
        }
    }

    for (int i = 0; i < num_nodes; i++) partition[i] = best_partition[i];
    for (int i = 0; i < num_parts; i++) nodes_in_part[i] = best_nip[i];
}

extern "C" __global__ void compute_part_cut_cost_10k(
    const int num_hyperedges,
    const int num_parts,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    const unsigned long long *edge_flags_all,
    int *out_per_part_cost,
    int *out_bottleneck_part
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int np = (num_parts < 64) ? num_parts : 64;
    int cost[64] = {0};
    for (int h = 0; h < num_hyperedges; h++) {
        unsigned long long fa = edge_flags_all[h];
        int span = __popcll(fa);
        if (span <= 1) continue;
        int contrib = span - 1;
        unsigned long long m = fa;
        while (m) {
            int p = __ffsll(m) - 1;
            m &= (m - 1);
            if (p < np) cost[p] += contrib;
        }
    }
    for (int p = 0; p < num_parts; p++) out_per_part_cost[p] = cost[p];
    int max_cost = -1;
    int max_part = 0;
    for (int p = 0; p < num_parts; p++) {
        if (cost[p] > max_cost) {
            max_cost = cost[p];
            max_part = p;
        }
    }
    out_bottleneck_part[0] = max_part;
}

extern "C" __global__ void repair_bottleneck_part_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int bottleneck_part,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *out_gain,
    int *out_target
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    out_gain[node] = -1;
    out_target[node] = -1;

    int cur = __ldg(&partition[node]);
    if (cur != bottleneck_part) return;

    int start = __ldg(&node_offsets[node]);
    int end   = __ldg(&node_offsets[node + 1]);
    int node_degree = end - start;
    if (node_degree <= 0) return;

    unsigned long long cur_bit = 1ULL << cur;
    unsigned int part_info[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_info[p] = 0;

    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;

    for (int j = start; j < end; j++) {
        int hedge = __ldg(&node_hyperedges[j]);
        unsigned long long fa = __ldg(&edge_flags_all[hedge]);
        unsigned long long fd = __ldg(&edge_flags_double[hedge]);
        unsigned long long mask = (fa & ~cur_bit) | (fd & cur_bit);
        if (mask & cur_bit) count_current_present++;

        unsigned long long f_all = mask & ~cur_bit;
        unsigned long long f_dbl = fd & ~cur_bit;
        while (f_all) {
            int bit = __ffsll(f_all) - 1;
            f_all &= (f_all - 1);
            part_info[bit] += 1;
            cand_mask |= 1ULL << bit;
        }
        while (f_dbl) {
            int bit = __ffsll(f_dbl) - 1;
            f_dbl &= (f_dbl - 1);
            part_info[bit] += 65536;
        }
    }

    int best_gain = -999999;
    int best_target = -1;

    unsigned long long tmp = cand_mask;
    while (tmp) {
        int target = __ffsll(tmp) - 1;
        tmp &= (tmp - 1);
        if ((unsigned)target >= (unsigned)num_parts) continue;
        if (__ldg(&nodes_in_part[target]) >= max_part_size) continue;

        int p_count = part_info[target] & 0xFFFF;
        int basic_gain = p_count - count_current_present;
        if (basic_gain > best_gain) {
            best_gain = basic_gain;
            best_target = target;
        }
    }

    if (best_gain > 0 && best_target != cur) {
        out_gain[node] = best_gain;
        out_target[node] = best_target;
    }
}


// ===========================================================================
// port_10k_fused -- `precompute_edge_flags_10k` + `compute_refinement_moves_
// optimized_10k` in ONE launch, ordered by a software grid barrier.
//
// This is the exp6 `grid_fused` transplant (the first champion of the 20k
// speed campaign, +9.6% there) onto the 10k track.  It is a LAUNCH-COUNT
// change only: both phases are the two original kernels' bodies, character for
// character, wrapped in grid-stride loops.  Nothing about the move arithmetic,
// the pin walk, the candidate mask, the tie-break or the emitted key changes.
//
//   phase 1 : per-hyperedge (all, double) part flags
//             1a thread-per-hyperedge for hedge_size <= warpSize
//             1b warp-per-hyperedge  for hedge_size >  warpSize
//   BARRIER   (monotonic counter, absolute target = calls * gridDim.x, passed
//             by the host; requires every block to be CO-RESIDENT, which the
//             host guarantees by capping the grid at 2 blocks/SM and by
//             __launch_bounds__(128, 2))
//   phase 2 : per-node best (gain, target) -> move_priorities
//
// WHY THE OUTPUT CANNOT DEPEND ON THE LAUNCH GEOMETRY
//   * phase 1 writes edge_flags_*[h] as a pure function of hyperedge h; each h
//     is handled by exactly one thread (1a) or exactly one warp (1b), and the
//     warp's (all, double) butterfly uses only OR / AND, which are associative
//     and commutative.  Which thread or warp gets h is irrelevant.
//   * phase 2 writes move_priorities[node] as a pure function of node and of
//     the (now final) flags array.  Again, which thread gets `node` is
//     irrelevant.
//   * NOTHING is compacted, counted per block, or accumulated across threads,
//     so unlike the 20k stack's `fused_flags_moves_filter_20k` there is not
//     even a block-size independence argument to make.
//   The original kernel's per-block `num_valid_moves[blockIdx.x]` output is
//   therefore dropped: it is the ONE output that is launch-geometry dependent,
//   and the host only ever used its SUM, to take a `break` that the very next
//   host filter takes anyway (see track_10k.rs / NOTES.md S2.3).
//
// TWO MECHANICAL DIFFERENCES FROM THE TWO ORIGINAL KERNELS
//   (1) phase 2 reads edge_flags_all / edge_flags_double with __ldcg, not
//       __ldg.  __ldg is `ld.global.nc`, which goes through the non-coherent
//       read-only cache; the flags are written by phase 1 of the SAME launch
//       from OTHER SMs, so only a .cg load (L2, bypassing L1) is guaranteed to
//       see them.  This is exactly what the 20k fused kernel does.  Same
//       values, different cache path.
//   (2) `__syncwarp()` before the `__activemask()` of phase 1b.  In the
//       original, the code reaches that point out of a single
//       `if (tid < num_hyperedges)`, so the warp is converged and the mask is
//       full; a grid-stride loop can give lanes different trip counts, so the
//       reconvergence is made explicit.  The mask is full either way.
// ===========================================================================

__device__ __forceinline__ unsigned int ff10k_ld_vol_u32(const unsigned int *p)
{
    unsigned int r;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
}

// Software grid barrier (monotonic counter, host-supplied absolute target).
// Lifted verbatim from the 20k stack (wave6_kmicro4 W6: volatile-load spin
// rather than atomicAdd(gb, 0), so pollers do not queue ahead of arrivals).
#define FF10K_GRID_BARRIER(gb, target)                     \
    do {                                                   \
        __threadfence();                                   \
        __syncthreads();                                   \
        if (threadIdx.x == 0) {                            \
            atomicAdd((gb), 1u);                           \
            while (ff10k_ld_vol_u32(gb) < (target)) {      \
                __nanosleep(200);                          \
            }                                              \
        }                                                  \
        __syncthreads();                                   \
        __threadfence();                                   \
    } while (0)

extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_10k(
    const int num_hyperedges,
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const int *nodes_in_part,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int barrier_target
) {
    __shared__ int shared_nodes_in_part[64];
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    // `nodes_in_part` is an INPUT of this kernel and is never written by it, so
    // the barrier's __syncthreads() below is all the publication this needs
    // (same structure as fused_flags_moves_20k).
    const int stride = gridDim.x * blockDim.x;

    // ================= phase 1a: precompute_edge_flags_10k, thread pass =====
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_hyperedges;
         tid += stride) {
        int start = __ldg(&hyperedge_offsets[tid]);
        int end   = __ldg(&hyperedge_offsets[tid + 1]);
        int hedge_size = end - start;

        if (hedge_size <= warpSize) {
            unsigned long long flags_all = 0ULL;
            unsigned long long flags_double = 0ULL;

            for (int k = start; k < end; k++) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldg(&partition[node]);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        flags_double |= (flags_all & bit);
                        flags_all |= bit;
                    }
                }
            }

            edge_flags_all[tid] = flags_all;
            edge_flags_double[tid] = flags_double;
        }
    }

    __syncwarp();

    // ================= phase 1b: precompute_edge_flags_10k, warp pass =======
    {
        int lane = threadIdx.x & 31;
        int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;
        int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
        int total_warps = gridDim.x * warps_per_block;
        unsigned active = __activemask();

        for (int hedge = global_warp; hedge < num_hyperedges; hedge += total_warps) {
            int start = 0;
            int end = 0;
            int hedge_size = 0;

            if (lane == 0) {
                start = __ldg(&hyperedge_offsets[hedge]);
                end   = __ldg(&hyperedge_offsets[hedge + 1]);
                hedge_size = end - start;
            }

            start = __shfl_sync(active, start, 0);
            end   = __shfl_sync(active, end, 0);
            hedge_size = __shfl_sync(active, hedge_size, 0);

            if (hedge_size <= warpSize) {
                continue;
            }

            unsigned long long local_all = 0ULL;
            unsigned long long local_double = 0ULL;

            for (int k = start + lane; k < end; k += warpSize) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldg(&partition[node]);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        local_double |= (local_all & bit);
                        local_all |= bit;
                    }
                }
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                unsigned long long other_all = shfl_xor_u64_10k(active, local_all, offset);
                unsigned long long other_double = shfl_xor_u64_10k(active, local_double, offset);
                local_double |= other_double | (local_all & other_all);
                local_all |= other_all;
            }

            if (lane == 0) {
                edge_flags_all[hedge] = local_all;
                edge_flags_double[hedge] = local_double;
            }
        }
    }

    FF10K_GRID_BARRIER(grid_barrier, barrier_target);

    // ============ phase 2: compute_refinement_moves_optimized_10k ===========
    for (int node = blockIdx.x * blockDim.x + threadIdx.x;
         node < num_nodes;
         node += stride) {
        move_priorities[node] = 0;
        int current_part = __ldg(&partition[node]);
        if ((unsigned)current_part < (unsigned)num_parts && shared_nodes_in_part[current_part] > 1) {
            int start = __ldg(&node_offsets[node]);
            int end   = __ldg(&node_offsets[node + 1]);
            int node_degree = end - start;

            if (node_degree > 0) {
                int degree_weight = node_degree > 255 ? 255 : node_degree;
                unsigned long long current_bit = 1ULL << current_part;

                unsigned int part_info[64];
                int np = (num_parts < 64) ? num_parts : 64;
                for (int p = 0; p < np; p++) part_info[p] = 0;

                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                for (int j = start; j < end; j++) {
                    int hyperedge = __ldg(&node_hyperedges[j]);
                    unsigned long long flags_all = __ldcg(&edge_flags_all[hyperedge]);
                    unsigned long long flags_double = __ldcg(&edge_flags_double[hyperedge]);
                    unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

                    if (mask & current_bit) count_current_present++;

                    unsigned long long f_all = mask & ~current_bit;
                    unsigned long long f_dbl = flags_double & ~current_bit;

                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        part_info[bit] += 1;
                        cand_mask |= 1ULL << bit;
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        part_info[bit] += 65536;
                    }
                }

                int best_gain = -999999;
                int best_target = current_part;

                while (cand_mask) {
                    int target_part = __ffsll(cand_mask) - 1;
                    cand_mask &= (cand_mask - 1);

                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count = part_info[target_part] & 0xFFFF;
                    int p_double = part_info[target_part] >> 16;

                    int basic_gain = p_count - count_current_present;
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    bool better = (total_gain > best_gain);
                    if (!better && total_gain == best_gain) {
                        int best_double = part_info[best_target] >> 16;
                        if (p_double > best_double) {
                            better = true;
                        } else if (p_double == best_double) {
                            int best_target_size = shared_nodes_in_part[best_target];
                            if (target_size < best_target_size) {
                                better = true;
                            } else if (target_size == best_target_size) {
                                int hash_tgt = (target_part * 17 + node) & 63;
                                int hash_best = (best_target * 17 + node) & 63;
                                if (hash_tgt < hash_best) better = true;
                            }
                        }
                    }

                    if (better) {
                        best_gain = total_gain;
                        best_target = target_part;
                    }
                }

                if (best_gain >= -1 && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
                }
            }
        }
    }
}

// ===========================================================================
// wave14 -- 10k on the 20k round machinery, stage A.
//
// `fused_flags_moves_bs_10k` below is `fused_flags_moves_10k` with phases 1a
// and 1b copied BYTE FOR BYTE and phase 2 rewritten so that the champion's
//
//     unsigned int part_info[64];            // 256 B per thread
//     for (int p = 0; p < np; p++) part_info[p] = 0;
//     while (f_all) { bit = __ffsll(f_all)-1; ...; part_info[bit] += 1; }
//     while (f_dbl) { bit = __ffsll(f_dbl)-1; ...; part_info[bit] += 65536; }
//
// -- a dynamically indexed 64-entry array that CANNOT live in registers, i.e.
// 9,216 threads x 256 B = 2.4 MB of local memory, 590 k zeroing stores per
// launch and one dependent local load+store per (pin, part) pair -- becomes
// two 16-plane BIT-SLICED counter sets (I5, the 20k stack's FF_PC_*):
//
//     pc0..pc15   plane k of the 64 low  counters ("count",  part_info & 0xFFFF)
//     dc0..dc15   plane k of the 64 high counters ("double", part_info >> 16)
//
// pcK's bit p is bit K of counter p.  `+= 1` on every counter selected by a
// 64-bit mask is one ripple-carry sweep over the planes (FF10K_PC_INC), so the
// per-pin `while (__ffsll(...))` loops -- popcount(mask) dependent local
// accesses each -- collapse to a handful of register AND/XORs that touch ALL
// 64 counters at once.  Reading counter p back is FF10K_PC_GET(p).
//
// BIT-IDENTITY (see DESIGN.md S3):
//   * value: plane k of counter p holds bit k of that counter, so
//     FF10K_PC_GET(p) == (sum of the += 1's applied to p) mod 2^16 and
//     FF10K_DC_GET(p) == (sum of the += 1's applied to p) mod 2^16.  The
//     champion's part_info[p] is (count + 65536*double) in a u32, so
//     part_info[p] & 0xFFFF == count mod 2^16 == FF10K_PC_GET(p) and
//     part_info[p] >> 16 == (double + (count >> 16)) mod 2^16, which equals
//     FF10K_DC_GET(p) IFF count < 65536.  count <= node_degree, and the HOST
//     only selects this kernel when num_hyperedges < 65536 (a node's degree is
//     the number of hyperedges that list it), so the two agree for every input
//     this kernel can be handed.
//   * order: `|=` on cand_mask, `++` on count_current_present and the counter
//     increments are commutative and associative, so the 4-deep unroll (I10)
//     of the pin walk -- same set of pins, same guards -- cannot change a bit.
//   * geometry: unchanged.  Phase 2 still writes move_priorities[node] as a
//     pure function of node and of the final flags array, one thread per node.
//
// The grid barrier is the same monotonic-counter barrier with the poll's
// __nanosleep lowered 200 -> 20 ns (I11); a spin interval cannot change a
// value, only how long a block waits after the counter has already reached
// its target.
// ===========================================================================

#define FF10K_GRID_BARRIER20(gb, target)                   \
    do {                                                   \
        __threadfence();                                   \
        __syncthreads();                                   \
        if (threadIdx.x == 0) {                            \
            atomicAdd((gb), 1u);                           \
            while (ff10k_ld_vol_u32(gb) < (target)) {      \
                __nanosleep(20);                           \
            }                                              \
        }                                                  \
        __syncthreads();                                   \
        __threadfence();                                   \
    } while (0)

// ---- generated by make_engine.py (PLANES = 16) ---------------------
#define FF10K_BS_DECL \
    unsigned long long pc0 = 0ULL, pc1 = 0ULL, pc2 = 0ULL, pc3 = 0ULL; \
    unsigned long long pc4 = 0ULL, pc5 = 0ULL, pc6 = 0ULL, pc7 = 0ULL; \
    unsigned long long pc8 = 0ULL, pc9 = 0ULL, pc10 = 0ULL, pc11 = 0ULL; \
    unsigned long long pc12 = 0ULL, pc13 = 0ULL, pc14 = 0ULL, pc15 = 0ULL; \
    unsigned long long dc0 = 0ULL, dc1 = 0ULL, dc2 = 0ULL, dc3 = 0ULL; \
    unsigned long long dc4 = 0ULL, dc5 = 0ULL, dc6 = 0ULL, dc7 = 0ULL; \
    unsigned long long dc8 = 0ULL, dc9 = 0ULL, dc10 = 0ULL, dc11 = 0ULL; \
    unsigned long long dc12 = 0ULL, dc13 = 0ULL, dc14 = 0ULL, dc15 = 0ULL;

#define FF10K_PC_INC(flags) do { \
    unsigned long long c_ = (flags), t_; \
    t_ = pc0 & c_; pc0 ^= c_; c_ = t_; \
    t_ = pc1 & c_; pc1 ^= c_; c_ = t_; \
    t_ = pc2 & c_; pc2 ^= c_; c_ = t_; \
    t_ = pc3 & c_; pc3 ^= c_; c_ = t_; \
    if (c_) { \
      t_ = pc4 & c_; pc4 ^= c_; c_ = t_; \
      t_ = pc5 & c_; pc5 ^= c_; c_ = t_; \
      t_ = pc6 & c_; pc6 ^= c_; c_ = t_; \
      t_ = pc7 & c_; pc7 ^= c_; c_ = t_; \
    if (c_) { \
        t_ = pc8 & c_; pc8 ^= c_; c_ = t_; \
        t_ = pc9 & c_; pc9 ^= c_; c_ = t_; \
        t_ = pc10 & c_; pc10 ^= c_; c_ = t_; \
        t_ = pc11 & c_; pc11 ^= c_; c_ = t_; \
    if (c_) { \
          t_ = pc12 & c_; pc12 ^= c_; c_ = t_; \
          t_ = pc13 & c_; pc13 ^= c_; c_ = t_; \
          t_ = pc14 & c_; pc14 ^= c_; c_ = t_; \
          t_ = pc15 & c_; pc15 ^= c_; c_ = t_; \
    } } } \
} while (0)

#define FF10K_DC_INC(flags) do { \
    unsigned long long c_ = (flags), t_; \
    t_ = dc0 & c_; dc0 ^= c_; c_ = t_; \
    t_ = dc1 & c_; dc1 ^= c_; c_ = t_; \
    t_ = dc2 & c_; dc2 ^= c_; c_ = t_; \
    t_ = dc3 & c_; dc3 ^= c_; c_ = t_; \
    if (c_) { \
      t_ = dc4 & c_; dc4 ^= c_; c_ = t_; \
      t_ = dc5 & c_; dc5 ^= c_; c_ = t_; \
      t_ = dc6 & c_; dc6 ^= c_; c_ = t_; \
      t_ = dc7 & c_; dc7 ^= c_; c_ = t_; \
    if (c_) { \
        t_ = dc8 & c_; dc8 ^= c_; c_ = t_; \
        t_ = dc9 & c_; dc9 ^= c_; c_ = t_; \
        t_ = dc10 & c_; dc10 ^= c_; c_ = t_; \
        t_ = dc11 & c_; dc11 ^= c_; c_ = t_; \
    if (c_) { \
          t_ = dc12 & c_; dc12 ^= c_; c_ = t_; \
          t_ = dc13 & c_; dc13 ^= c_; c_ = t_; \
          t_ = dc14 & c_; dc14 ^= c_; c_ = t_; \
          t_ = dc15 & c_; dc15 ^= c_; c_ = t_; \
    } } } \
} while (0)

#define FF10K_PC_GET(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((pc6 >> (tp)) & 1ULL) << 6) | \
    ((int)((pc7 >> (tp)) & 1ULL) << 7) | \
    ((int)((pc8 >> (tp)) & 1ULL) << 8) | \
    ((int)((pc9 >> (tp)) & 1ULL) << 9) | \
    ((int)((pc10 >> (tp)) & 1ULL) << 10) | \
    ((int)((pc11 >> (tp)) & 1ULL) << 11) | \
    ((int)((pc12 >> (tp)) & 1ULL) << 12) | \
    ((int)((pc13 >> (tp)) & 1ULL) << 13) | \
    ((int)((pc14 >> (tp)) & 1ULL) << 14) | \
    ((int)((pc15 >> (tp)) & 1ULL) << 15) )

#define FF10K_DC_GET(tp) ( \
    ((int)((dc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((dc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((dc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((dc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((dc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((dc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((dc6 >> (tp)) & 1ULL) << 6) | \
    ((int)((dc7 >> (tp)) & 1ULL) << 7) | \
    ((int)((dc8 >> (tp)) & 1ULL) << 8) | \
    ((int)((dc9 >> (tp)) & 1ULL) << 9) | \
    ((int)((dc10 >> (tp)) & 1ULL) << 10) | \
    ((int)((dc11 >> (tp)) & 1ULL) << 11) | \
    ((int)((dc12 >> (tp)) & 1ULL) << 12) | \
    ((int)((dc13 >> (tp)) & 1ULL) << 13) | \
    ((int)((dc14 >> (tp)) & 1ULL) << 14) | \
    ((int)((dc15 >> (tp)) & 1ULL) << 15) )

// One pin, token for token the champion's loop body between
// `unsigned long long flags_all = ...` and the end of the two while-loops,
// with the two while-loops replaced by the bit-sliced increments.
#define FF10K_PIN(FA, FD) do {                                                 \
    const unsigned long long fa_ = (FA);                                       \
    const unsigned long long fd_ = (FD);                                       \
    const unsigned long long mask_ = (fa_ & ~current_bit) | (fd_ & current_bit); \
    if (mask_ & current_bit) count_current_present++;                          \
    const unsigned long long fl_ = mask_ & ~current_bit;                       \
    const unsigned long long fb_ = fd_ & ~current_bit;                         \
    cand_mask |= fl_;                                                          \
    if (fl_) { FF10K_PC_INC(fl_); }                                            \
    if (fb_) { FF10K_DC_INC(fb_); }                                            \
} while (0)


// Precompute bit masks for the EXACT 10k lexicographic target ordering.
static __device__ __forceinline__ void exact_target_masks_10k(
    int nn, int np, int cap, int *sizes, unsigned long long *free_mask,
    unsigned long long *bonus, unsigned long long *size_planes)
{
    __syncthreads();
    int lane=threadIdx.x&31, warp=threadIdx.x>>5, nw=blockDim.x>>5;
    if(warp==0){
        unsigned lo=__ballot_sync(0xffffffffu,lane<np&&sizes[lane]<cap);
        unsigned hi=__ballot_sync(0xffffffffu,lane+32<np&&sizes[lane+32]<cap);
        if(lane==0)*free_mask=(unsigned long long)lo|((unsigned long long)hi<<32);
    }
    for(int c=warp;c<64;c+=nw){
        int cs=c<np?sizes[c]:0;
        unsigned lo=__ballot_sync(0xffffffffu,lane<np&&cs>sizes[lane]+1);
        unsigned hi=__ballot_sync(0xffffffffu,lane+32<np&&cs>sizes[lane+32]+1);
        if(lane==0)bonus[c]=(unsigned long long)lo|((unsigned long long)hi<<32);
    }
    int bits=32-__clz((unsigned)nn);
    for(int b=warp;b<bits;b+=nw){
        unsigned lo=__ballot_sync(0xffffffffu,lane<np&&((sizes[lane]>>b)&1));
        unsigned hi=__ballot_sync(0xffffffffu,lane+32<np&&((sizes[lane+32]>>b)&1));
        if(lane==0)size_planes[b]=(unsigned long long)lo|((unsigned long long)hi<<32);
    }
    __syncthreads();
}

#define C10K_SMALL_PIN(FA,FD) \
do { \
unsigned long long fa=(FA),fd=(FD),fl=fa&~current_bit,fb=fd&~current_bit; \
count_current_present+=(int)((fd&current_bit)!=0);cand_mask|=fl; \
if(fl){unsigned long long carry=fl,next; \
next=pc0&carry;pc0^=carry;carry=next; \
next=pc1&carry;pc1^=carry;carry=next; \
next=pc2&carry;pc2^=carry;carry=next; \
next=pc3&carry;pc3^=carry;carry=next; \
pc4^=carry; \
} \
if(fb){unsigned long long carry=fb,next; \
next=dc0&carry;dc0^=carry;carry=next; \
next=dc1&carry;dc1^=carry;carry=next; \
next=dc2&carry;dc2^=carry;carry=next; \
next=dc3&carry;dc3^=carry;carry=next; \
dc4^=carry; \
} \
}while(0)
#define C10K_MID_PIN(FA,FD) \
do { \
unsigned long long fa=(FA),fd=(FD),fl=fa&~current_bit,fb=fd&~current_bit; \
count_current_present+=(int)((fd&current_bit)!=0);cand_mask|=fl; \
if(fl){unsigned long long carry=fl,next; \
next=pc0&carry;pc0^=carry;carry=next; \
next=pc1&carry;pc1^=carry;carry=next; \
next=pc2&carry;pc2^=carry;carry=next; \
next=pc3&carry;pc3^=carry;carry=next; \
next=pc4&carry;pc4^=carry;carry=next; \
pc5^=carry; \
} \
if(fb){unsigned long long carry=fb,next; \
next=dc0&carry;dc0^=carry;carry=next; \
next=dc1&carry;dc1^=carry;carry=next; \
next=dc2&carry;dc2^=carry;carry=next; \
next=dc3&carry;dc3^=carry;carry=next; \
next=dc4&carry;dc4^=carry;carry=next; \
dc5^=carry; \
} \
}while(0)
extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_bs_10k(
    const int num_hyperedges,
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const int *nodes_in_part,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int barrier_target
) {
    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64], exact_size_planes[32];
    if (threadIdx.x < 64) {
        shared_nodes_in_part[threadIdx.x] = threadIdx.x < num_parts ? nodes_in_part[threadIdx.x] : 0;
    }
    exact_target_masks_10k(num_nodes,num_parts,max_part_size,shared_nodes_in_part,
                           &exact_free,exact_bonus,exact_size_planes);
    // `nodes_in_part` is an INPUT of this kernel and is never written by it, so
    // the barrier's __syncthreads() below is all the publication this needs
    // (same structure as fused_flags_moves_20k).
    const int stride = gridDim.x * blockDim.x;

    // ================= phase 1a: precompute_edge_flags_10k, thread pass =====
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_hyperedges;
         tid += stride) {
        int start = __ldg(&hyperedge_offsets[tid]);
        int end   = __ldg(&hyperedge_offsets[tid + 1]);
        int hedge_size = end - start;

        if (hedge_size <= warpSize) {
            unsigned long long flags_all = 0ULL;
            unsigned long long flags_double = 0ULL;

            for (int k = start; k < end; k++) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldg(&partition[node]);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        flags_double |= (flags_all & bit);
                        flags_all |= bit;
                    }
                }
            }

            edge_flags_all[tid] = flags_all;
            edge_flags_double[tid] = flags_double;
        }
    }

    __syncwarp();

    // ================= phase 1b: precompute_edge_flags_10k, warp pass =======
    {
        int lane = threadIdx.x & 31;
        int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;
        int global_warp = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
        int total_warps = gridDim.x * warps_per_block;
        unsigned active = __activemask();

        for (int hedge = global_warp; hedge < num_hyperedges; hedge += total_warps) {
            int start = 0;
            int end = 0;
            int hedge_size = 0;

            if (lane == 0) {
                start = __ldg(&hyperedge_offsets[hedge]);
                end   = __ldg(&hyperedge_offsets[hedge + 1]);
                hedge_size = end - start;
            }

            start = __shfl_sync(active, start, 0);
            end   = __shfl_sync(active, end, 0);
            hedge_size = __shfl_sync(active, hedge_size, 0);

            if (hedge_size <= warpSize) {
                continue;
            }

            unsigned long long local_all = 0ULL;
            unsigned long long local_double = 0ULL;

            for (int k = start + lane; k < end; k += warpSize) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldg(&partition[node]);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        local_double |= (local_all & bit);
                        local_all |= bit;
                    }
                }
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                unsigned long long other_all = shfl_xor_u64_10k(active, local_all, offset);
                unsigned long long other_double = shfl_xor_u64_10k(active, local_double, offset);
                local_double |= other_double | (local_all & other_all);
                local_all |= other_all;
            }

            if (lane == 0) {
                edge_flags_all[hedge] = local_all;
                edge_flags_double[hedge] = local_double;
            }
        }
    }

    FF10K_GRID_BARRIER20(grid_barrier, barrier_target);

    // ============ phase 2: compute_refinement_moves_optimized_10k ===========
    // wave14 (I5 + I10): identical arithmetic, no local memory, 4-deep pin walk.
    for (int node = blockIdx.x * blockDim.x + threadIdx.x;
         node < num_nodes;
         node += stride) {
        int start = __ldg(&node_offsets[node]);
        int end = __ldg(&node_offsets[node+1]);
        int node_degree=end-start;
        // Disjoint ownership: this pass never writes a high-degree node.
        if(node_degree>16)continue;
        move_priorities[node] = 0;
        int current_part = __ldg(&partition[node]);
        if ((unsigned)current_part < (unsigned)num_parts && shared_nodes_in_part[current_part] > 1) {


            if (node_degree > 0) {
                int degree_weight = node_degree > 255 ? 255 : node_degree;
                unsigned long long current_bit = 1ULL << current_part;

                FF10K_BS_DECL

                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                int j = start;
                for (; j + 3 < end; j += 4) {
                    const int h0_ = __ldg(&node_hyperedges[j]);
                    const int h1_ = __ldg(&node_hyperedges[j + 1]);
                    const int h2_ = __ldg(&node_hyperedges[j + 2]);
                    const int h3_ = __ldg(&node_hyperedges[j + 3]);
                    const unsigned long long a0_ = __ldcg(&edge_flags_all[h0_]);
                    const unsigned long long d0_ = __ldcg(&edge_flags_double[h0_]);
                    const unsigned long long a1_ = __ldcg(&edge_flags_all[h1_]);
                    const unsigned long long d1_ = __ldcg(&edge_flags_double[h1_]);
                    const unsigned long long a2_ = __ldcg(&edge_flags_all[h2_]);
                    const unsigned long long d2_ = __ldcg(&edge_flags_double[h2_]);
                    const unsigned long long a3_ = __ldcg(&edge_flags_all[h3_]);
                    const unsigned long long d3_ = __ldcg(&edge_flags_double[h3_]);
                    C10K_SMALL_PIN(a0_, d0_);
                    C10K_SMALL_PIN(a1_, d1_);
                    C10K_SMALL_PIN(a2_, d2_);
                    C10K_SMALL_PIN(a3_, d3_);
                }
                for (; j < end; j++) {
                    const int hy_ = __ldg(&node_hyperedges[j]);
                    const unsigned long long at_ = __ldcg(&edge_flags_all[hy_]);
                    const unsigned long long dt_ = __ldcg(&edge_flags_double[hy_]);
                    C10K_SMALL_PIN(at_, dt_);
                }

                
                // This degree class cannot populate higher counter planes.
                pc5=0ULL; pc6=0ULL; pc7=0ULL; pc8=0ULL; pc9=0ULL; pc10=0ULL; pc11=0ULL; pc12=0ULL; pc13=0ULL; pc14=0ULL; pc15=0ULL; dc5=0ULL; dc6=0ULL; dc7=0ULL; dc8=0ULL; dc9=0ULL; dc10=0ULL; dc11=0ULL; dc12=0ULL; dc13=0ULL; dc14=0ULL; dc15=0ULL;
                int best_gain = -999999;
                int best_target = current_part;

                if (node_degree <= 65531 && num_nodes > 0) {
                    unsigned long long active = cand_mask & exact_free;
                    if (active) {
                        unsigned long long carry = exact_bonus[current_part], tmp;
                        tmp=pc2&carry; pc2^=carry; carry=tmp;
                        tmp=pc3&carry; pc3^=carry; carry=tmp;
                        tmp=pc4&carry; pc4^=carry; carry=tmp;
                        best_gain=0;
                        { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
                        { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
                        { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
                        { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
                        { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
                        best_gain-=count_current_present;
                        { unsigned long long m=active&dc4; if(m)active=m; }
                        { unsigned long long m=active&dc3; if(m)active=m; }
                        { unsigned long long m=active&dc2; if(m)active=m; }
                        { unsigned long long m=active&dc1; if(m)active=m; }
                        { unsigned long long m=active&dc0; if(m)active=m; }
                        for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                            unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
                        }
                        unsigned long long x1=active&0x2222222222222222ULL;
                        unsigned long long x2=active&0x4444444444444444ULL;
                        unsigned long long x3=active&0x8888888888888888ULL;
                        unsigned long long perm=(active&0x1111111111111111ULL)|
                            ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
                        unsigned shift=(unsigned)node&63;
                        unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
                        unsigned h=(unsigned)(__ffsll(hashed)-1);
                        best_target=(int)(((h-(unsigned)node)*49u)&63u);
                    }
                } else {
                while (cand_mask) {
                    int target_part = __ffsll(cand_mask) - 1;
                    cand_mask &= (cand_mask - 1);

                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count = FF10K_PC_GET(target_part);
                    int p_double = FF10K_DC_GET(target_part);

                    int basic_gain = p_count - count_current_present;
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    bool better = (total_gain > best_gain);
                    if (!better && total_gain == best_gain) {
                        int best_double = FF10K_DC_GET(best_target);
                        if (p_double > best_double) {
                            better = true;
                        } else if (p_double == best_double) {
                            int best_target_size = shared_nodes_in_part[best_target];
                            if (target_size < best_target_size) {
                                better = true;
                            } else if (target_size == best_target_size) {
                                int hash_tgt = (target_part * 17 + node) & 63;
                                int hash_best = (best_target * 17 + node) & 63;
                                if (hash_tgt < hash_best) better = true;
                            }
                        }
                    }

                    if (better) {
                        best_gain = total_gain;
                        best_target = target_part;
                    }
                }


                }

                if (best_gain >= -1 && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
                }
            }
        }
    }
    // Degree 17..128: four lanes per node, five/eight-bit exact partial sums.
    // All lanes execute the reductions, including inactive/padding groups.
    {
        int lane=threadIdx.x&31,group=lane>>2,within=lane&3;
        int warp=(int)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
        int total_warps=(int)gridDim.x*(blockDim.x>>5);
        for(int base_node=warp*8;base_node<num_nodes;base_node+=total_warps*8){
            int node=base_node+group;
            int start=node<num_nodes?__ldg(&node_offsets[node]):0;
            int end=node<num_nodes?__ldg(&node_offsets[node+1]):0;
            int node_degree=end-start;
            bool in_class=node<num_nodes && node_degree>16 && node_degree<=128;
            if(!__any_sync(0xffffffffu,in_class))continue;
            int current_part=node<num_nodes?__ldg(&partition[node]):-1;
            bool go=in_class && (unsigned)current_part<(unsigned)num_parts && shared_nodes_in_part[current_part]>1;
            unsigned long long current_bit=go?(1ULL<<current_part):0ULL;
            FF10K_BS_DECL
            unsigned long long cand_mask=0ULL;int count_current_present=0;
            if(go)for(int j=start+within;j<end;j+=16){
                int j0=j+0;bool v0=j0<end;int h0=v0?__ldg(&node_hyperedges[j0]):0;
                int j1=j+4;bool v1=j1<end;int h1=v1?__ldg(&node_hyperedges[j1]):0;
                int j2=j+8;bool v2=j2<end;int h2=v2?__ldg(&node_hyperedges[j2]):0;
                int j3=j+12;bool v3=j3<end;int h3=v3?__ldg(&node_hyperedges[j3]):0;
                unsigned long long a0=v0?__ldcg(&edge_flags_all[h0]):0ULL,d0=v0?__ldcg(&edge_flags_double[h0]):0ULL;
                unsigned long long a1=v1?__ldcg(&edge_flags_all[h1]):0ULL,d1=v1?__ldcg(&edge_flags_double[h1]):0ULL;
                unsigned long long a2=v2?__ldcg(&edge_flags_all[h2]):0ULL,d2=v2?__ldcg(&edge_flags_double[h2]):0ULL;
                unsigned long long a3=v3?__ldcg(&edge_flags_all[h3]):0ULL,d3=v3?__ldcg(&edge_flags_double[h3]):0ULL;
                if(v0){C10K_MID_PIN(a0,d0);}
                if(v1){C10K_MID_PIN(a1,d1);}
                if(v2){C10K_MID_PIN(a2,d2);}
                if(v3){C10K_MID_PIN(a3,d3);}
            }
            for(int off=2;off;off>>=1){
                cand_mask|=__shfl_xor_sync(0xffffffffu,cand_mask,off);
                count_current_present+=__shfl_xor_sync(0xffffffffu,count_current_present,off);
                {unsigned long long other,sum,carry,next;
                  other=__shfl_xor_sync(0xffffffffu,pc0,off);
                  sum=pc0^other;carry=pc0&other;pc0=sum;
                  other=__shfl_xor_sync(0xffffffffu,pc1,off);
                  sum=pc1^other;next=(pc1&other)|(sum&carry);pc1=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc2,off);
                  sum=pc2^other;next=(pc2&other)|(sum&carry);pc2=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc3,off);
                  sum=pc3^other;next=(pc3&other)|(sum&carry);pc3=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc4,off);
                  sum=pc4^other;next=(pc4&other)|(sum&carry);pc4=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc5,off);
                  sum=pc5^other;next=(pc5&other)|(sum&carry);pc5=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc6,off);
                  sum=pc6^other;next=(pc6&other)|(sum&carry);pc6=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc7,off);
                  sum=pc7^other;next=(pc7&other)|(sum&carry);pc7=sum^carry;carry=next;
                }
                {unsigned long long other,sum,carry,next;
                  other=__shfl_xor_sync(0xffffffffu,dc0,off);
                  sum=dc0^other;carry=dc0&other;dc0=sum;
                  other=__shfl_xor_sync(0xffffffffu,dc1,off);
                  sum=dc1^other;next=(dc1&other)|(sum&carry);dc1=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc2,off);
                  sum=dc2^other;next=(dc2&other)|(sum&carry);dc2=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc3,off);
                  sum=dc3^other;next=(dc3&other)|(sum&carry);dc3=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc4,off);
                  sum=dc4^other;next=(dc4&other)|(sum&carry);dc4=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc5,off);
                  sum=dc5^other;next=(dc5&other)|(sum&carry);dc5=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc6,off);
                  sum=dc6^other;next=(dc6&other)|(sum&carry);dc6=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc7,off);
                  sum=dc7^other;next=(dc7&other)|(sum&carry);dc7=sum^carry;carry=next;
                }
            }
            int outkey=0;
            if(go){int degree_weight=min(255,node_degree);
    
                // This degree class cannot populate higher counter planes.
                pc8=0ULL; pc9=0ULL; pc10=0ULL; pc11=0ULL; pc12=0ULL; pc13=0ULL; pc14=0ULL; pc15=0ULL; dc8=0ULL; dc9=0ULL; dc10=0ULL; dc11=0ULL; dc12=0ULL; dc13=0ULL; dc14=0ULL; dc15=0ULL;
                int best_gain = -999999;
    int best_target = current_part;

    if (node_degree <= 65531 && num_nodes > 0) {
        unsigned long long active = cand_mask & exact_free;
        if (active) {
            unsigned long long carry = exact_bonus[current_part], tmp;
            tmp=pc2&carry; pc2^=carry; carry=tmp;
            tmp=pc3&carry; pc3^=carry; carry=tmp;
            tmp=pc4&carry; pc4^=carry; carry=tmp;
            tmp=pc5&carry; pc5^=carry; carry=tmp;
            tmp=pc6&carry; pc6^=carry; carry=tmp;
            tmp=pc7&carry; pc7^=carry; carry=tmp;
            best_gain=0;
            { unsigned long long m=active&pc7; if(m){active=m;best_gain|=128;} }
            { unsigned long long m=active&pc6; if(m){active=m;best_gain|=64;} }
            { unsigned long long m=active&pc5; if(m){active=m;best_gain|=32;} }
            { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
            { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
            { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
            { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
            { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
            best_gain-=count_current_present;
            { unsigned long long m=active&dc7; if(m)active=m; }
            { unsigned long long m=active&dc6; if(m)active=m; }
            { unsigned long long m=active&dc5; if(m)active=m; }
            { unsigned long long m=active&dc4; if(m)active=m; }
            { unsigned long long m=active&dc3; if(m)active=m; }
            { unsigned long long m=active&dc2; if(m)active=m; }
            { unsigned long long m=active&dc1; if(m)active=m; }
            { unsigned long long m=active&dc0; if(m)active=m; }
            for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
            }
            unsigned long long x1=active&0x2222222222222222ULL;
            unsigned long long x2=active&0x4444444444444444ULL;
            unsigned long long x3=active&0x8888888888888888ULL;
            unsigned long long perm=(active&0x1111111111111111ULL)|
                ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
            unsigned shift=(unsigned)node&63;
            unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
            unsigned h=(unsigned)(__ffsll(hashed)-1);
            best_target=(int)(((h-(unsigned)node)*49u)&63u);
        }
    } else {
    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = FF10K_PC_GET(target_part);
        int p_double = FF10K_DC_GET(target_part);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = FF10K_DC_GET(best_target);
            if (p_double > best_double) {
                better = true;
            } else if (p_double == best_double) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better = true;
                }
            }
        }

        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }


    }

    if (best_gain >= -1 && best_target != current_part) {
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        outkey = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
            }
            if(within==0 && in_class){move_priorities[node]=outkey;}
        }
    }

    // High-degree nodes get a whole warp; small-node threads never touch
    // their output slots. This removes long divergent pin walks without
    // adding a grid barrier or changing any sampled/full-degree semantics.
    {
        int lane=threadIdx.x&31;
        int warp=(int)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
        int warp_stride=(int)gridDim.x*(blockDim.x>>5);
        for(int node=warp;node<num_nodes;node+=warp_stride){
            int start=__ldg(&node_offsets[node]),end=__ldg(&node_offsets[node+1]);
            int node_degree=end-start;
            if(node_degree<=128)continue;
            int outkey=0;
            int current_part=__ldg(&partition[node]);
            if((unsigned)current_part<(unsigned)num_parts && shared_nodes_in_part[current_part]>1){
                int degree_weight=min(255,node_degree);
                unsigned long long current_bit=1ULL<<current_part;
                FF10K_BS_DECL
                unsigned long long cand_mask=0ULL;int count_current_present=0;
                for(int base=start;base<end;base+=128){
                    int j0=base+lane+0;bool v0=j0<end;int h0=v0?__ldg(&node_hyperedges[j0]):0;
                    int j1=base+lane+32;bool v1=j1<end;int h1=v1?__ldg(&node_hyperedges[j1]):0;
                    int j2=base+lane+64;bool v2=j2<end;int h2=v2?__ldg(&node_hyperedges[j2]):0;
                    int j3=base+lane+96;bool v3=j3<end;int h3=v3?__ldg(&node_hyperedges[j3]):0;
                    unsigned long long a0=v0?__ldcg(&edge_flags_all[h0]):0ULL,d0=v0?__ldcg(&edge_flags_double[h0]):0ULL;
                    unsigned long long a1=v1?__ldcg(&edge_flags_all[h1]):0ULL,d1=v1?__ldcg(&edge_flags_double[h1]):0ULL;
                    unsigned long long a2=v2?__ldcg(&edge_flags_all[h2]):0ULL,d2=v2?__ldcg(&edge_flags_double[h2]):0ULL;
                    unsigned long long a3=v3?__ldcg(&edge_flags_all[h3]):0ULL,d3=v3?__ldcg(&edge_flags_double[h3]):0ULL;
                    if(v0){FF10K_PIN(a0,d0);}
                    if(v1){FF10K_PIN(a1,d1);}
                    if(v2){FF10K_PIN(a2,d2);}
                    if(v3){FF10K_PIN(a3,d3);}
                }
                for(int off=16;off;off>>=1){
                    cand_mask|=__shfl_xor_sync(0xffffffffu,cand_mask,off);
                    count_current_present+=__shfl_xor_sync(0xffffffffu,count_current_present,off);
                    { unsigned long long other,carry,sum,next;
                      other=__shfl_xor_sync(0xffffffffu,pc0,off);
                      sum=pc0^other;carry=pc0&other;pc0=sum;
                      other=__shfl_xor_sync(0xffffffffu,pc1,off);
                      sum=pc1^other;next=(pc1&other)|(sum&carry);pc1=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc2,off);
                      sum=pc2^other;next=(pc2&other)|(sum&carry);pc2=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc3,off);
                      sum=pc3^other;next=(pc3&other)|(sum&carry);pc3=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc4,off);
                      sum=pc4^other;next=(pc4&other)|(sum&carry);pc4=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc5,off);
                      sum=pc5^other;next=(pc5&other)|(sum&carry);pc5=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc6,off);
                      sum=pc6^other;next=(pc6&other)|(sum&carry);pc6=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc7,off);
                      sum=pc7^other;next=(pc7&other)|(sum&carry);pc7=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc8,off);
                      sum=pc8^other;next=(pc8&other)|(sum&carry);pc8=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc9,off);
                      sum=pc9^other;next=(pc9&other)|(sum&carry);pc9=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc10,off);
                      sum=pc10^other;next=(pc10&other)|(sum&carry);pc10=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc11,off);
                      sum=pc11^other;next=(pc11&other)|(sum&carry);pc11=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc12,off);
                      sum=pc12^other;next=(pc12&other)|(sum&carry);pc12=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc13,off);
                      sum=pc13^other;next=(pc13&other)|(sum&carry);pc13=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc14,off);
                      sum=pc14^other;next=(pc14&other)|(sum&carry);pc14=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc15,off);
                      sum=pc15^other;next=(pc15&other)|(sum&carry);pc15=sum^carry;carry=next;
                    }
                    { unsigned long long other,carry,sum,next;
                      other=__shfl_xor_sync(0xffffffffu,dc0,off);
                      sum=dc0^other;carry=dc0&other;dc0=sum;
                      other=__shfl_xor_sync(0xffffffffu,dc1,off);
                      sum=dc1^other;next=(dc1&other)|(sum&carry);dc1=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc2,off);
                      sum=dc2^other;next=(dc2&other)|(sum&carry);dc2=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc3,off);
                      sum=dc3^other;next=(dc3&other)|(sum&carry);dc3=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc4,off);
                      sum=dc4^other;next=(dc4&other)|(sum&carry);dc4=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc5,off);
                      sum=dc5^other;next=(dc5&other)|(sum&carry);dc5=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc6,off);
                      sum=dc6^other;next=(dc6&other)|(sum&carry);dc6=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc7,off);
                      sum=dc7^other;next=(dc7&other)|(sum&carry);dc7=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc8,off);
                      sum=dc8^other;next=(dc8&other)|(sum&carry);dc8=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc9,off);
                      sum=dc9^other;next=(dc9&other)|(sum&carry);dc9=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc10,off);
                      sum=dc10^other;next=(dc10&other)|(sum&carry);dc10=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc11,off);
                      sum=dc11^other;next=(dc11&other)|(sum&carry);dc11=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc12,off);
                      sum=dc12^other;next=(dc12&other)|(sum&carry);dc12=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc13,off);
                      sum=dc13^other;next=(dc13&other)|(sum&carry);dc13=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc14,off);
                      sum=dc14^other;next=(dc14&other)|(sum&carry);dc14=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc15,off);
                      sum=dc15^other;next=(dc15&other)|(sum&carry);dc15=sum^carry;carry=next;
                    }
                }
    int best_gain = -999999;
    int best_target = current_part;

    if (node_degree <= 65531 && num_nodes > 0) {
        unsigned long long active = cand_mask & exact_free;
        if (active) {
            unsigned long long carry = exact_bonus[current_part], tmp;
            tmp=pc2&carry; pc2^=carry; carry=tmp;
            tmp=pc3&carry; pc3^=carry; carry=tmp;
            tmp=pc4&carry; pc4^=carry; carry=tmp;
            tmp=pc5&carry; pc5^=carry; carry=tmp;
            tmp=pc6&carry; pc6^=carry; carry=tmp;
            tmp=pc7&carry; pc7^=carry; carry=tmp;
            tmp=pc8&carry; pc8^=carry; carry=tmp;
            tmp=pc9&carry; pc9^=carry; carry=tmp;
            tmp=pc10&carry; pc10^=carry; carry=tmp;
            tmp=pc11&carry; pc11^=carry; carry=tmp;
            tmp=pc12&carry; pc12^=carry; carry=tmp;
            tmp=pc13&carry; pc13^=carry; carry=tmp;
            tmp=pc14&carry; pc14^=carry; carry=tmp;
            tmp=pc15&carry; pc15^=carry; carry=tmp;
            best_gain=0;
            { unsigned long long m=active&pc15; if(m){active=m;best_gain|=32768;} }
            { unsigned long long m=active&pc14; if(m){active=m;best_gain|=16384;} }
            { unsigned long long m=active&pc13; if(m){active=m;best_gain|=8192;} }
            { unsigned long long m=active&pc12; if(m){active=m;best_gain|=4096;} }
            { unsigned long long m=active&pc11; if(m){active=m;best_gain|=2048;} }
            { unsigned long long m=active&pc10; if(m){active=m;best_gain|=1024;} }
            { unsigned long long m=active&pc9; if(m){active=m;best_gain|=512;} }
            { unsigned long long m=active&pc8; if(m){active=m;best_gain|=256;} }
            { unsigned long long m=active&pc7; if(m){active=m;best_gain|=128;} }
            { unsigned long long m=active&pc6; if(m){active=m;best_gain|=64;} }
            { unsigned long long m=active&pc5; if(m){active=m;best_gain|=32;} }
            { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
            { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
            { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
            { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
            { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
            best_gain-=count_current_present;
            { unsigned long long m=active&dc15; if(m)active=m; }
            { unsigned long long m=active&dc14; if(m)active=m; }
            { unsigned long long m=active&dc13; if(m)active=m; }
            { unsigned long long m=active&dc12; if(m)active=m; }
            { unsigned long long m=active&dc11; if(m)active=m; }
            { unsigned long long m=active&dc10; if(m)active=m; }
            { unsigned long long m=active&dc9; if(m)active=m; }
            { unsigned long long m=active&dc8; if(m)active=m; }
            { unsigned long long m=active&dc7; if(m)active=m; }
            { unsigned long long m=active&dc6; if(m)active=m; }
            { unsigned long long m=active&dc5; if(m)active=m; }
            { unsigned long long m=active&dc4; if(m)active=m; }
            { unsigned long long m=active&dc3; if(m)active=m; }
            { unsigned long long m=active&dc2; if(m)active=m; }
            { unsigned long long m=active&dc1; if(m)active=m; }
            { unsigned long long m=active&dc0; if(m)active=m; }
            for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
            }
            unsigned long long x1=active&0x2222222222222222ULL;
            unsigned long long x2=active&0x4444444444444444ULL;
            unsigned long long x3=active&0x8888888888888888ULL;
            unsigned long long perm=(active&0x1111111111111111ULL)|
                ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
            unsigned shift=(unsigned)node&63;
            unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
            unsigned h=(unsigned)(__ffsll(hashed)-1);
            best_target=(int)(((h-(unsigned)node)*49u)&63u);
        }
    } else {
    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = FF10K_PC_GET(target_part);
        int p_double = FF10K_DC_GET(target_part);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = FF10K_DC_GET(best_target);
            if (p_double > best_double) {
                better = true;
            } else if (p_double == best_double) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better = true;
                }
            }
        }

        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }


    }

    if (best_gain >= -1 && best_target != current_part) {
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        outkey = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
            }
            if(lane==0){move_priorities[node]=outkey;}
        }
    }

}

static __device__ __forceinline__ int ff10k_block_sum(int v, int *s_warp)
{
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0u) s_warp[warp] = v;
    __syncthreads();
    const int nw = (int)(blockDim.x >> 5);
    int s = 0;
    for (int w = 0; w < nw; w++) s += s_warp[w];
    __syncthreads();
    return s;
}

static __device__ __forceinline__ int ff10k_block_max(int v, int *s_warp)
{
    for (int off = 16; off > 0; off >>= 1) {
        int o = __shfl_xor_sync(0xffffffffu, v, off);
        if (o > v) v = o;
    }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0u) s_warp[warp] = v;
    __syncthreads();
    const int nw = (int)(blockDim.x >> 5);
    int m = s_warp[0];
    for (int w = 1; w < nw; w++) { int c = s_warp[w]; if (c > m) m = c; }
    __syncthreads();
    return m;
}

// Exact bounded workspace. Sorting and application use disjoint phases.
// Exact bounded workspace. Sorting and application use disjoint phases.
// Exact bounded workspace. Sorting and application use disjoint phases.
// Exact bounded workspace. Sorting and application use disjoint phases.
union ExactScratch_10k {
    struct {
        int initial[64],outgoing[64],risk[64],offsets[65],final_sizes[64];
        int before[512],rejected[512],packed[512];
        unsigned long long ckeys[512],skeys[512],mark_cutoff;
        int live[512],possible[64],active_count;
        int accepted;
    } keyed;

    struct {
        int initial[64],outgoing[64],risk[64],offsets[65];
        int rank[512],before[512],packed[512];
        unsigned keys[512];unsigned char rejected[512];int accepted;
    } critical;

    unsigned long long sort[4096];
    unsigned int bins[264];
    struct {
        int packed[1024];
        unsigned char accept_a[1024],accept_b[1024];
        int counts[64],outgoing[64],result[64];
        unsigned int events[32*64];
        unsigned short word_prefix[32*64],event_ids[2048];
        int starts[65];
    } fixed;
    struct {
        int packed[1024];
        short prev_source[1024];
        short prev_target[1024];
        int done_epoch[1024];
        int counts[64];
        int last[64];
    } dag;
};

// Replay the ordered move sequence as a dependency DAG. A move depends on
// the previous move touching each of its two parts. Ready moves therefore
// touch disjoint counters and commute, including the capacity rejection test.
static __device__ __forceinline__ int exact_apply_dag_10k(
    int M, int nn, int np, int cap, int *partition, int *part_sizes,
    const int *ordered, int *tabu, int round, int tenure,
    int mark_base, int mark_mult, ExactScratch_10k *work, int *reduce)
{
    int tid = threadIdx.x;
    int local_accepted = 0;
    if (tid < 64) work->dag.counts[tid] = tid < np ? __ldcg(&part_sizes[tid]) : 0;
    __syncthreads();
    for (int base = 0; base < M; base += 1024) {
        int len = min(1024, M - base);
        if (tid < 64) work->dag.last[tid] = -1;
        for (int i = tid; i < len; i += blockDim.x) {
            int nt = __ldcg(&ordered[base+i]);
            int node = nt >> 6, target = nt & 63;
            int current = (unsigned)node < (unsigned)nn ? __ldcg(&partition[node]) : -1;
            work->dag.packed[i] = ((unsigned)current < (unsigned)np && target < np)
                ? ((node << 12) | (current << 6) | target) : -1;
            work->dag.prev_source[i] = -1;
            work->dag.prev_target[i] = -1;
            work->dag.done_epoch[i] = -1;
        }
        __syncthreads();
        // One warp constructs BOTH predecessor chains in O(M), in ordered
        // batches of 16 moves. Its 32 lanes represent the two endpoints.
        if (tid < 32) {
            for (int first = 0; first < len; first += 16) {
                int i = first + (tid >> 1);
                int packed = i < len ? work->dag.packed[i] : -1;
                int part = packed < 0 ? (64 + tid) : ((tid & 1) ? (packed & 63) : ((packed >> 6) & 63));
                unsigned matches = __match_any_sync(0xffffffffu, part);
                unsigned earlier = matches & ((1u << (tid & ~1)) - 1u);
                int pred = -1;
                if (packed >= 0) {
                    pred = earlier ? first + ((31 - __clz(earlier)) >> 1) : work->dag.last[part];
                    if (tid & 1) work->dag.prev_target[i] = (short)pred;
                    else work->dag.prev_source[i] = (short)pred;
                }
                __syncwarp(0xffffffffu);
                if (packed >= 0 && tid == 31 - __clz(matches)) work->dag.last[part] = i;
                __syncwarp(0xffffffffu);
            }
        }
        __syncthreads();
        int i = tid;
        for (int epoch = 0; ; ++epoch) {
            if (i < len) {
                int ps = work->dag.prev_source[i], pt = work->dag.prev_target[i];
                // Atomic flags avoid a read/write data race inside an epoch.
                // A flag from THIS epoch is deliberately not consumable until
                // the next block barrier, so counter accesses cannot overlap.
                int ds = ps < 0 ? -2 : atomicAdd(&work->dag.done_epoch[ps], 0);
                int dt = pt < 0 ? -2 : atomicAdd(&work->dag.done_epoch[pt], 0);
                bool ready = (ps < 0 || (ds >= 0 && ds < epoch)) &&
                             (pt < 0 || (dt >= 0 && dt < epoch));
                if (ready) {
                    int packed = work->dag.packed[i];
                    if (packed >= 0) {
                        int source = (packed >> 6) & 63, target = packed & 63;
                        if (work->dag.counts[target] < cap && work->dag.counts[source] > 1) {
                            --work->dag.counts[source];
                            ++work->dag.counts[target];
                            partition[packed >> 12] = target;
                            ++local_accepted;
                        }
                    }
                    atomicExch(&work->dag.done_epoch[i], epoch);
                    i += blockDim.x;
                }
            }
            if (__syncthreads_count(i < len) == 0) break;
        }
    }
    if (tid < np && tid < 64) part_sizes[tid] = work->dag.counts[tid];
    int accepted = ff10k_block_sum(local_accepted, reduce);
    if (accepted > 0) {
        int marks = min(M, max(mark_base, accepted * mark_mult));
        for (int j = tid; j < marks; j += blockDim.x) {
            int node = __ldcg(&ordered[j]) >> 6;
            if ((unsigned)node < (unsigned)nn) tabu[node] = round + tenure;
        }
    }
    return accepted;
}

// Exact fast replay when even rejecting every possible incoming move cannot
// empty a source part. Capacity decisions then form a monotone, causal system.
// Iterate from all-accepted; each part independently replays its token balance.
// Stop after four passes and use the generic exact DAG for the untouched suffix
// if convergence is slow. No approximation or changed acceptance is allowed.
static __device__ __forceinline__ int exact_apply_10k(
    int M, int nn, int np, int cap, int *partition, int *part_sizes,
    const int *ordered, int *tabu, int round, int tenure,
    int mark_base, int mark_mult, ExactScratch_10k *work, int *reduce)
{
    int tid=threadIdx.x;
    if(tid<64){
        work->fixed.counts[tid]=tid<np?__ldcg(&part_sizes[tid]):0;
        work->fixed.outgoing[tid]=0;
    }
    __syncthreads();
    for(int i=tid;i<M;i+=blockDim.x){
        int nt=__ldcg(&ordered[i]), node=nt>>6, target=nt&63;
        if((unsigned)node<(unsigned)nn && target<np){
            int source=__ldcg(&partition[node]);
            if((unsigned)source<(unsigned)np)atomicAdd(&work->fixed.outgoing[source],1);
        }
    }
    __syncthreads();
    int unsafe_source=__syncthreads_or(tid<np && tid<64 &&
        (work->fixed.counts[tid]-work->fixed.outgoing[tid]<1 || work->fixed.counts[tid]>cap));
    if(unsafe_source)
        return exact_apply_dag_10k(M,nn,np,cap,partition,part_sizes,ordered,
                                    tabu,round,tenure,mark_base,mark_mult,work,reduce);
    int local_accepted=0;
    for(int base=0;base<M;base+=1024){
        int len=min(1024,M-base);
        for(int i=tid;i<64*32;i+=blockDim.x)work->fixed.events[i]=0;
        __syncthreads();
        for(int i=tid;i<len;i+=blockDim.x){
            int nt=__ldcg(&ordered[base+i]), node=nt>>6, target=nt&63;
            int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
            int packed=((unsigned)source<(unsigned)np && target<np)
                ?((node<<12)|(source<<6)|target):-1;
            work->fixed.packed[i]=packed;
            work->fixed.accept_a[i]=(packed>=0);
            work->fixed.accept_b[i]=0;
            if(packed>=0){
                atomicOr(&work->fixed.events[(i>>5)*64+source],1u<<(i&31));
                if(source!=target)atomicOr(&work->fixed.events[(i>>5)*64+target],1u<<(i&31));
            }
        }
        __syncthreads();
        // Transposed bitmaps provide conflict-free per-part scans. Build a
        // sorted CSR of both endpoint events without sorting the move list.
        if(tid<64){
            int prefix=0;
            for(int w=0;w<((len+31)>>5);++w){
                work->fixed.word_prefix[w*64+tid]=(unsigned short)prefix;
                prefix+=__popc(work->fixed.events[w*64+tid]);
            }
            int inclusive=prefix,lane=tid&31;
            for(int off=1;off<32;off<<=1){
                int v=__shfl_up_sync(0xffffffffu,inclusive,off);
                if(lane>=off)inclusive+=v;
            }
            work->fixed.starts[tid+1]=inclusive;
        }
        if(tid==0)work->fixed.starts[0]=0;
        __syncthreads();
        if(tid>=32 && tid<64)work->fixed.starts[tid+1]+=work->fixed.starts[32];
        __syncthreads();
        for(int i=tid;i<len;i+=blockDim.x){
            int packed=work->fixed.packed[i];
            if(packed>=0){
                int source=(packed>>6)&63,target=packed&63;
                unsigned before=(1u<<(i&31))-1u;
                int word=i>>5;
                int at=work->fixed.starts[source]+work->fixed.word_prefix[word*64+source]
                       +__popc(work->fixed.events[word*64+source]&before);
                work->fixed.event_ids[at]=(unsigned short)i;
                if(source!=target){
                    at=work->fixed.starts[target]+work->fixed.word_prefix[word*64+target]
                       +__popc(work->fixed.events[word*64+target]&before);
                    work->fixed.event_ids[at]=(unsigned short)i;
                }
            }
        }
        __syncthreads();
        int converged=0;
        unsigned char *accepted=work->fixed.accept_a;
        for(int iteration=0;iteration<8;++iteration){
            const unsigned char *prev=(iteration&1)?work->fixed.accept_b:work->fixed.accept_a;
            unsigned char *next=(iteration&1)?work->fixed.accept_a:work->fixed.accept_b;
            int changed=0;
            // A part's greedy token queue has a closed-form reflected prefix
            // sum. Departures use the preceding fixed-point iterate; incoming
            // attempts are resolved exactly by an inclusive prefix minimum.
            // One warp handles 32 ordered endpoint events at a time.
            int lane=tid&31,warp=tid>>5,nwarps=blockDim.x>>5;
            for(int part=warp;part<np && part<64;part+=nwarps){
                int free=cap-work->fixed.counts[part];
                int start=work->fixed.starts[part],end=work->fixed.starts[part+1];
                for(int first=start;first<end;first+=32){
                    int index=first+lane;
                    int i=index<end?(int)work->fixed.event_ids[index]:-1;
                    int packed=i>=0?work->fixed.packed[i]:-1;
                    int incoming=packed>=0 && (packed&63)==part;
                    int release=packed>=0 && ((packed>>6)&63)==part?(int)prev[i]:0;
                    int delta=release-incoming,inclusive=delta;
                    for(int off=1;off<32;off<<=1){
                        int other=__shfl_up_sync(0xffffffffu,inclusive,off);
                        if(lane>=off)inclusive+=other;
                    }
                    int tentative=free+inclusive-delta-incoming;
                    int minimum=tentative;
                    for(int off=1;off<32;off<<=1){
                        int other=__shfl_up_sync(0xffffffffu,minimum,off);
                        if(lane>=off)minimum=min(minimum,other);
                    }
                    int before_min=__shfl_up_sync(0xffffffffu,minimum,1);
                    if(lane==0)before_min=0;
                    if(incoming){
                        int ok=tentative-min(0,before_min)>=0;
                        next[i]=(unsigned char)ok;
                        changed|=ok!=(int)prev[i];
                    }
                    free+=__shfl_sync(0xffffffffu,inclusive,31)
                          -min(0,__shfl_sync(0xffffffffu,minimum,31));
                }
                if(lane==0)work->fixed.result[part]=cap-free;
            }
            int any_change=__syncthreads_or(changed);
            if(!any_change){converged=1;accepted=next;break;}
        }
        if(!converged){
            if(tid<np && tid<64)part_sizes[tid]=work->fixed.counts[tid];
            __syncthreads();
            int prefix=ff10k_block_sum(local_accepted,reduce);
            int suffix=exact_apply_dag_10k(M-base,nn,np,cap,partition,part_sizes,ordered+base,
                                           tabu,round,tenure,0,0,work,reduce);
            int total=prefix+suffix;
            if(total>0){
                int marks=min(M,max(mark_base,total*mark_mult));
                for(int j=tid;j<marks;j+=blockDim.x){
                    int node=__ldcg(&ordered[j])>>6;
                    if((unsigned)node<(unsigned)nn)tabu[node]=round+tenure;
                }
            }
            return total;
        }
        if(tid<np && tid<64)work->fixed.counts[tid]=work->fixed.result[tid];
        for(int i=tid;i<len;i+=blockDim.x){
            if(accepted[i]){
                int packed=work->fixed.packed[i];
                if(packed>=0){partition[packed>>12]=packed&63;++local_accepted;}
            }
        }
        __syncthreads();
    }
    if(tid<np && tid<64)part_sizes[tid]=work->fixed.counts[tid];
    int total=ff10k_block_sum(local_accepted,reduce);
    if(total>0){
        int marks=min(M,max(mark_base,total*mark_mult));
        for(int j=tid;j<marks;j+=blockDim.x){
            int node=__ldcg(&ordered[j])>>6;
            if((unsigned)node<(unsigned)nn)tabu[node]=round+tenure;
        }
    }
    return total;
}

// Only incoming moves after a target's initial free-slot quota can fail a
// capacity test. All earlier incoming moves are unconditionally capacity-safe.
// Once the source-size lower bound is certified, replay only these critical
// moves (at most 8 per part); all other moves are guaranteed accepted.
// Exact replay directly from the 64 sorted target prefixes. Safe moves do
// not need a global merge; only critical incoming events need a total order.
static __device__ __forceinline__ int keyed_apply_10k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_10k *work)
{
    const int tid=threadIdx.x,lane=tid&31;
    int bad=0;
    if(tid<64){
        int c=tid<np?__ldcg(&part_sizes[tid]):0;
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-c)):0;
        work->keyed.initial[tid]=c;work->keyed.outgoing[tid]=0;
        work->keyed.risk[tid]=risk;
        bad|=risk>8 || (tid<np && c>cap);
        int sum=risk;
        for(int off=1;off<32;off<<=1){int v=__shfl_up_sync(0xffffffffu,sum,off);if(lane>=off)sum+=v;}
        work->keyed.offsets[tid+1]=sum;
    }
    if(tid==0)work->keyed.offsets[0]=0;
    if(__syncthreads_or(bad))return -1;
    if(tid>=32 && tid<64)work->keyed.offsets[tid+1]+=work->keyed.offsets[32];
    __syncthreads();
    const int S=work->keyed.offsets[64];
    for(int flat=tid;flat<512;flat+=blockDim.x){
        int p=flat>>3,j=flat&7;
        if(j<work->keyed.risk[p]){
            int index=work->keyed.offsets[p]+j;
            int position=offsets[p]+max(0,cap-work->keyed.initial[p])+j;
            unsigned long long key=__ldcg(&selected[position]);
            int node=(int)(unsigned)key,target=(((unsigned)(key>>32))&63)^63;
            int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
            work->keyed.ckeys[index]=key;
            work->keyed.before[index]=0;work->keyed.rejected[index]=0;work->keyed.live[index]=1;
            if((unsigned)source>=(unsigned)np || target!=p){bad=1;continue;}
            work->keyed.packed[index]=(node<<12)|(source<<6)|target;
        }
    }
    if(__syncthreads_or(bad))return -1;
    for(int i=tid;i<M;i+=blockDim.x){
        unsigned long long key=__ldcg(&selected[i]);
        int node=(int)(unsigned)key,target=(((unsigned)(key>>32))&63)^63;
        int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
        if((unsigned)source>=(unsigned)np || target>=np){bad=1;continue;}
        atomicAdd(&work->keyed.outgoing[source],1);
        int start=work->keyed.offsets[source],end=work->keyed.offsets[source+1];
        for(int j=start;j<end;++j)if(key<work->keyed.ckeys[j])atomicAdd(&work->keyed.before[j],1);
    }
    __syncthreads();
    if(tid<np && tid<64)
        bad|=work->keyed.outgoing[tid]>0 && work->keyed.initial[tid]-work->keyed.outgoing[tid]<1;
    if(__syncthreads_or(bad))return -1;

    // Rejected departures are the only possible positive correction to a
    // target's all-accepted reference size. If its free margin exceeds the
    // number of still-possibly-rejected departures, this event is certain to
    // succeed. Repeating this conservative elimination never removes a reject.
    for(int iteration=0;iteration<3;++iteration){
        if(tid<64)work->keyed.possible[tid]=0;
        __syncthreads();
        for(int j=tid;j<S;j+=blockDim.x)if(work->keyed.live[j]){
            int source=(work->keyed.packed[j]>>6)&63;
            atomicAdd(&work->keyed.possible[source],1);
        }
        __syncthreads();
        int changed=0;
        for(int j=tid;j<S;j+=blockDim.x)if(work->keyed.live[j]){
            int target=work->keyed.packed[j]&63;
            int margin=work->keyed.before[j]-(j-work->keyed.offsets[target]);
            if(margin>work->keyed.possible[target]){work->keyed.live[j]=0;changed=1;}
        }
        if(!__syncthreads_or(changed))break;
    }
    if(tid==0)work->keyed.active_count=0;
    __syncthreads();
    for(int j=tid;j<S;j+=blockDim.x)if(work->keyed.live[j]){
        unsigned long long key=work->keyed.ckeys[j];
        // Node ids use fewer than 19 bits on the guarded fast path. Append
        // the original 9-bit event index without changing the total key order.
        unsigned long long encoded=(key&0xffffffff00000000ULL)|((unsigned long long)(unsigned)key<<9)|(unsigned)j;
        int pos=atomicAdd(&work->keyed.active_count,1);work->keyed.skeys[pos]=encoded;
    }
    __syncthreads();
    const int K=work->keyed.active_count;
    int power=1;while(power<K)power<<=1;
    if(power<=(int)blockDim.x){
        unsigned long long value=tid<K?work->keyed.skeys[tid]:~0ULL;
        __syncthreads();
        for(int size=2;size<=power;size<<=1)for(int dist=size>>1;dist;dist>>=1){
            unsigned long long other;
            if(dist<32)other=__shfl_xor_sync(0xffffffffu,value,dist);
            else{work->keyed.skeys[tid]=value;__syncthreads();other=work->keyed.skeys[tid^dist];__syncthreads();}
            bool take_min=((tid&dist)==0)==((tid&size)==0);
            value=take_min?min(value,other):max(value,other);
        }
        if(tid<K)work->keyed.skeys[tid]=value;
        __syncthreads();
    }else{
        for(int i=tid;i<power;i+=blockDim.x)if(i>=K)work->keyed.skeys[i]=~0ULL;
        __syncthreads();
        for(int size=2;size<=power;size<<=1)for(int dist=size>>1;dist;dist>>=1){
            for(int i=tid;i<power;i+=blockDim.x){int peer=i^dist;if(peer>i){
                unsigned long long a=work->keyed.skeys[i],b=work->keyed.skeys[peer];
                if((i&size)==0?a>b:a<b){work->keyed.skeys[i]=b;work->keyed.skeys[peer]=a;}
            }}__syncthreads();
        }
    }
    if(tid<32){
        int d0=0,d1=0,rejected=0;
        for(int j=0;j<K;++j){
            int index=(int)(work->keyed.skeys[j]&511ULL);
            int packed=work->keyed.packed[index],source=(packed>>6)&63,target=packed&63;
            int ri=index-work->keyed.offsets[target];
            int base_free=work->keyed.before[index]-ri;
            int delta=__shfl_sync(0xffffffffu,target<32?d0:d1,target&31);
            if(base_free<=delta){
                d0+=(source==tid)-(target==tid);
                d1+=(source==tid+32)-(target==tid+32);
                ++rejected;if(tid==0)work->keyed.rejected[index]=1;
            }
        }
        work->keyed.final_sizes[tid]=work->keyed.initial[tid]-work->keyed.outgoing[tid]+keeps[tid]+d0;
        work->keyed.final_sizes[tid+32]=work->keyed.initial[tid+32]-work->keyed.outgoing[tid+32]+keeps[tid+32]+d1;
        if(tid==0){work->keyed.accepted=M-rejected;work->keyed.mark_cutoff=~0ULL;}
    }
    __syncthreads();
    int accepted=work->keyed.accepted;
    int marks=accepted>0?min(M,max(mark_base,accepted*mark_mult)):0;
    int omitted=M-marks;
    // 10k marks only the first accepted-count candidates, not every accepted
    // node. Obtain this exact prefix boundary by merging just the omitted tail.
    if(marks>0 && omitted>64)return -1;
    if(marks>0 && omitted>0 && tid<32){
        int q0=keeps[tid],q1=keeps[tid+32];
        for(int step=0;step<=omitted;++step){
            unsigned long long a=q0>0?__ldcg(&selected[offsets[tid]+q0-1]):0;
            unsigned long long b=q1>0?__ldcg(&selected[offsets[tid+32]+q1-1]):0;
            unsigned long long mx=max(a,b);
            for(int off=16;off>0;off>>=1)mx=max(mx,__shfl_xor_sync(0xffffffffu,mx,off));
            if(step==omitted){if(tid==0)work->keyed.mark_cutoff=mx;}
            else{if(a==mx)--q0;if(b==mx)--q1;}
        }
    }
    __syncthreads();
    if(tid<np && tid<64)part_sizes[tid]=work->keyed.final_sizes[tid];
    const unsigned long long mark_cutoff=work->keyed.mark_cutoff;
    for(int i=tid;i<M;i+=blockDim.x){
        unsigned long long key=__ldcg(&selected[i]);
        int node=(int)(unsigned)key,target=(((unsigned)(key>>32))&63)^63;
        partition[node]=target;
        if(marks>0 && key<=mark_cutoff)tabu[node]=round+tenure;
    }
    __syncthreads();
    for(int j=tid;j<S;j+=blockDim.x)if(work->keyed.rejected[j]){
        int packed=work->keyed.packed[j];partition[packed>>12]=(packed>>6)&63;
    }
    return accepted;
}

static __device__ __forceinline__ int exact_apply_critical_10k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const int *ordered,int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_10k *work,int *reduce,const int *critical_input,const int *keeps)
{
    int tid=threadIdx.x,lane=tid&31;
    int bad=0;
    if(tid<64){
        int c=tid<np?__ldcg(&part_sizes[tid]):0;
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-c)):0;
        work->critical.initial[tid]=c;work->critical.outgoing[tid]=0;
        work->critical.risk[tid]=risk;
        bad|=risk>8 || (tid<np && c>cap);
        int sum=risk;
        for(int off=1;off<32;off<<=1){int v=__shfl_up_sync(0xffffffffu,sum,off);if(lane>=off)sum+=v;}
        work->critical.offsets[tid+1]=sum;
    }
    if(tid==0)work->critical.offsets[0]=0;
    if(__syncthreads_or(bad))
        return exact_apply_10k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
                                 round,tenure,mark_base,mark_mult,work,reduce);
    if(tid>=32 && tid<64)work->critical.offsets[tid+1]+=work->critical.offsets[32];
    __syncthreads();
    int S=work->critical.offsets[64];
    for(int flat=tid;flat<512;flat+=blockDim.x){
        int p=flat>>3,j=flat&7;
        if(j<work->critical.risk[p]){
            int index=work->critical.offsets[p]+j;
            int rank=__ldcg(&critical_input[flat]);
            work->critical.rank[index]=rank;
            work->critical.before[index]=0;work->critical.rejected[index]=0;
            work->critical.keys[index]=((unsigned)rank<<9)|(unsigned)index;
            if((unsigned)rank>=(unsigned)M){bad=1;continue;}
            int nt=__ldcg(&ordered[rank]),node=nt>>6,target=nt&63;
            int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
            if((unsigned)source>=(unsigned)np || target!=p){bad=1;continue;}
            work->critical.packed[index]=(node<<12)|(source<<6)|target;
        }
    }
    if(__syncthreads_or(bad))
        return exact_apply_10k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
                                 round,tenure,mark_base,mark_mult,work,reduce);
    // Independent integer tallies: count all departures before each critical
    // incoming move. Their order of accumulation cannot affect the result.
    for(int i=tid;i<M;i+=blockDim.x){
        int nt=__ldcg(&ordered[i]),node=nt>>6,target=nt&63;
        int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
        if((unsigned)source>=(unsigned)np || target>=np){bad=1;continue;}
        atomicAdd(&work->critical.outgoing[source],1);
        int start=work->critical.offsets[source],end=work->critical.offsets[source+1];
        for(int j=start;j<end;++j)if(i<work->critical.rank[j])atomicAdd(&work->critical.before[j],1);
    }
    __syncthreads();
    if(tid<np && tid<64)
        bad|=work->critical.outgoing[tid]>0 && work->critical.initial[tid]-work->critical.outgoing[tid]<1;
    if(__syncthreads_or(bad))
        return exact_apply_10k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
                                 round,tenure,mark_base,mark_mult,work,reduce);
    // Sort only the critical-event ranks, never the full selected move list.
    int power=1;while(power<S)power<<=1;
    if(power<=(int)blockDim.x){
        unsigned value=tid<S?work->critical.keys[tid]:~0u;
        __syncthreads();
        for(int size=2;size<=power;size<<=1)for(int dist=size>>1;dist;dist>>=1){
            unsigned other;
            if(dist<32)other=__shfl_xor_sync(0xffffffffu,value,dist);
            else{
                work->critical.keys[tid]=value;__syncthreads();
                other=work->critical.keys[tid^dist];__syncthreads();
            }
            bool take_min=((tid&dist)==0)==((tid&size)==0);
            value=take_min?min(value,other):max(value,other);
        }
        if(tid<S)work->critical.keys[tid]=value;
        __syncthreads();
    }else{
        for(int i=tid;i<power;i+=blockDim.x)if(i>=S)work->critical.keys[i]=~0u;
        __syncthreads();
        for(int size=2;size<=power;size<<=1)for(int dist=size>>1;dist;dist>>=1){
            for(int i=tid;i<power;i+=blockDim.x){int peer=i^dist;if(peer>i){
                unsigned a=work->critical.keys[i],b=work->critical.keys[peer];
                if((i&size)==0?a>b:a<b){work->critical.keys[i]=b;work->critical.keys[peer]=a;}
            }}
            __syncthreads();
        }
    }
    if(tid<32){
        // delta[p] = actual size minus the all-accepted reference size.
        // Rejecting a move adds one at its source and subtracts one at target.
        int d0=0,d1=0,rejected=0;
        for(int j=0;j<S;++j){
            int index=(int)(work->critical.keys[j]&511u);
            int packed=work->critical.packed[index];
            int target=packed&63,source=(packed>>6)&63;
            int risk_index=index-work->critical.offsets[target];
            int base_free=work->critical.before[index]-risk_index;
            int delta=__shfl_sync(0xffffffffu,target<32?d0:d1,target&31);
            if(base_free<=delta){
                d0+=(int)(source==tid)-(int)(target==tid);
                d1+=(int)(source==tid+32)-(int)(target==tid+32);
                ++rejected;if(tid==0)work->critical.rejected[index]=1;
            }
        }
        if(tid<np)part_sizes[tid]=work->critical.initial[tid]-work->critical.outgoing[tid]+keeps[tid]+d0;
        if(tid+32<np)part_sizes[tid+32]=work->critical.initial[tid+32]-work->critical.outgoing[tid+32]+keeps[tid+32]+d1;
        if(tid==0)work->critical.accepted=M-rejected;
    }
    __syncthreads();
    for(int i=tid;i<M;i+=blockDim.x){int nt=__ldcg(&ordered[i]);partition[nt>>6]=nt&63;}
    // No other block may read this partition until the caller's grid barrier.
    // Restore rejected critical moves after every initial scatter has finished.
    __syncthreads();
    for(int j=tid;j<S;j+=blockDim.x)if(work->critical.rejected[j]){
        int packed=work->critical.packed[j];partition[packed>>12]=(packed>>6)&63;
    }
    int accepted=work->critical.accepted;
    if(accepted>0){
        int marks=min(M,max(mark_base,accepted*mark_mult));
        for(int i=tid;i<marks;i+=blockDim.x){int node=__ldcg(&ordered[i])>>6;tabu[node]=round+tenure;}
    }
    return accepted;
}

static __device__ __forceinline__ void exact_sort_prefix_10k(
    unsigned long long *keys,int n,int keep,ExactScratch_10k *work,int *ctl)
{
    int tid=threadIdx.x, nb=blockDim.x, lane=tid&31;
    int limit=32;
    while(limit<keep && limit<4096)limit<<=1;
    limit=min(4096,limit*2);
    int count=n;
    if(n>limit){
        unsigned long long low=~0ULL,high=0ULL;
        for(int j=tid;j<n;j+=nb){
            unsigned long long key=__ldcg(&keys[j]);
            low=min(low,key);high=max(high,key);
        }
        for(int off=16;off>0;off>>=1){
            low=min(low,__shfl_xor_sync(0xffffffffu,low,off));
            high=max(high,__shfl_xor_sync(0xffffffffu,high,off));
        }
        if(lane==0){work->sort[tid>>5]=low;work->sort[16+(tid>>5)]=high;}
        __syncthreads();
        low=~0ULL;high=0ULL;
        for(int w=0;w<(nb>>5);++w){low=min(low,work->sort[w]);high=max(high,work->sort[16+w]);}
        // All warps must finish reading min/max before the same union
        // storage is reused as histogram bins (shared-memory WAR hazard).
        __syncthreads();
        int first_shift=max(0,63-__clzll(low^high)-7);
        unsigned long long mask=first_shift==56?0ULL:(~0ULL<<(first_shift+8));
        unsigned long long prefix=low&mask,upper=~0ULL;
        int rank=keep,below=0;
        for(int shift=first_shift;shift>=0;shift=(shift>8?shift-8:(shift>0?0:-1))){
            for(int j=tid;j<256;j+=nb)work->bins[j]=0;
            __syncthreads();
            for(int base=0;base<n;base+=nb){
                int j=base+tid;
                unsigned long long key=j<n?__ldcg(&keys[j]):0ULL;
                int bin=(j<n && (key&mask)==prefix)?(int)((key>>shift)&255):256;
                unsigned group=__match_any_sync(0xffffffffu,bin);
                if(bin<256 && lane==__ffs(group)-1)atomicAdd(&work->bins[bin],__popc(group));
            }
            __syncthreads();
            if(tid<32){
                for(int c=0;c<8;++c){
                    unsigned value=work->bins[c*32+lane];
                    for(int off=16;off>0;off>>=1)value+=__shfl_xor_sync(0xffffffffu,value,off);
                    if(lane==0)work->bins[256+c]=value;
                }
                __syncwarp(0xffffffffu);
                if(lane==0){
                    int sum=0,chunk=0;
                    while(chunk<7 && sum+(int)work->bins[256+chunk]<rank){sum+=(int)work->bins[256+chunk];++chunk;}
                    ctl[0]=chunk;ctl[1]=rank-sum;ctl[2]=below+sum;
                }
                __syncwarp(0xffffffffu);
                int chunk=ctl[0],r=ctl[1];
                unsigned value=work->bins[chunk*32+lane],inclusive=value;
                for(int off=1;off<32;off<<=1){
                    unsigned other=__shfl_up_sync(0xffffffffu,inclusive,off);
                    if(lane>=off)inclusive+=other;
                }
                unsigned pick=__ballot_sync(0xffffffffu,(int)inclusive>=r && (int)(inclusive-value)<r);
                int chosen_lane=__ffs(pick)-1;
                int before=(int)__shfl_sync(0xffffffffu,inclusive-value,chosen_lane);
                int amount=(int)__shfl_sync(0xffffffffu,value,chosen_lane);
                __syncwarp(0xffffffffu); // finish reading the broadcast control record
                if(lane==0){ctl[0]=chunk*32+chosen_lane;ctl[1]=r-before;ctl[2]+=before;ctl[3]=amount;}
            }
            __syncthreads();
            int chosen=ctl[0];rank=ctl[1];below=ctl[2];
            prefix|=(unsigned long long)(unsigned)chosen<<shift;
            mask|=255ULL<<shift;
            upper=prefix|(shift?((1ULL<<shift)-1ULL):0ULL);
            if(below+ctl[3]<=limit)break;
        }
        __syncthreads(); // all readers finish the final radix decision before reset
        if(tid==0)ctl[3]=0;
        __syncthreads();
        for(int j=tid;j<n;j+=nb){
            unsigned long long key=__ldcg(&keys[j]);
            if(key<=upper){int pos=atomicAdd(&ctl[3],1);work->sort[pos]=key;}
        }
        __syncthreads();
        count=ctl[3];
    }else{
        for(int j=tid;j<n;j+=nb)work->sort[j]=__ldcg(&keys[j]);
        __syncthreads();
    }
    if(count<=nb){
        // One key per lane: all within-warp compare exchanges stay in registers.
        // Only the cross-warp stages need shared memory and block barriers.
        unsigned long long value=tid<count?work->sort[tid]:~0ULL;
        __syncthreads();
        int network=1;while(network<count)network<<=1;
        for(int size=2;size<=network;size<<=1){
            for(int distance=size>>1;distance>0;distance>>=1){
                unsigned long long other;
                if(distance<32){
                    other=__shfl_xor_sync(0xffffffffu,value,distance);
                }else{
                    work->sort[tid]=value;
                    __syncthreads();
                    other=work->sort[tid^distance];
                    __syncthreads();
                }
                bool take_min=((tid&distance)==0)==((tid&size)==0);
                value=take_min?min(value,other):max(value,other);
            }
        }
        if(tid<keep)keys[tid]=value;
        __syncthreads();
    }else{
        int power=1;while(power<count)power<<=1;
        for(int j=tid;j<power;j+=nb)if(j>=count)work->sort[j]=~0ULL;
        __syncthreads();
        for(int size=2;size<=power;size<<=1){
            for(int distance=size>>1;distance>0;distance>>=1){
                for(int j=tid;j<power;j+=nb){
                    int peer=j^distance;
                    if(peer>j){
                        unsigned long long a=work->sort[j],b=work->sort[peer];
                        if((j&size)==0?a>b:a<b){work->sort[j]=b;work->sort[peer]=a;}
                    }
                }
                __syncthreads();
            }
        }
        for(int j=tid;j<keep;j+=nb)keys[j]=work->sort[j];
        __syncthreads();
    }
}

#define FF10K_GRID_BARRIER(g,t) FF10K_GRID_BARRIER20(g,t)

static __device__ __forceinline__ int grouped_bucket_10k(int *counts,int target) {
    unsigned active=__activemask();
    unsigned same=__match_any_sync(active,target);
    int lane=threadIdx.x&31,leader=__ffs(same)-1,base=0;
    if(lane==leader)base=atomicAdd(counts+target,__popc(same));
    base=__shfl_sync(active,base,leader);
    return base+__popc(same&((1u<<lane)-1u));
}
template<int G>
static __device__ __forceinline__ void flags10_task(int4 item,int lane,int nn,
    const int *pins,const int *partition,unsigned long long *all,unsigned long long *dbl){
    int member=lane&(G-1),start=item.y,size=item.z;
    unsigned long long a=0,d=0;
    for(int j=member;j<size;j+=4*G){
        int nodes[4],parts[4];
#pragma unroll
        for(int k=0;k<4;++k)nodes[k]=j+k*G<size?__ldg(pins+start+j+k*G):-1;
#pragma unroll
        for(int k=0;k<4;++k)parts[k]=(unsigned)nodes[k]<(unsigned)nn?__ldcg(partition+nodes[k]):-1;
#pragma unroll
        for(int k=0;k<4;++k)if((unsigned)parts[k]<64u){unsigned long long bit=1ULL<<parts[k];d|=a&bit;a|=bit;}
    }
#pragma unroll
    for(int off=G/2;off;off>>=1){
        unsigned long long b=__shfl_xor_sync(0xffffffffu,a,off),e=__shfl_xor_sync(0xffffffffu,d,off);
        d|=e|(a&b);a|=b;
    }
    if(member==0 && item.x>=0){all[item.x]=a;dbl[item.x]=d;}
}
static __device__ __forceinline__ void flags10_classed(int nn,const int *pins,const int *partition,
    const int4 *slots,int tasks,const int4 *giants,int ngiants,
    unsigned long long *all,unsigned long long *dbl){
    __shared__ unsigned long long partial[16];
    int lane=threadIdx.x&31,warps=blockDim.x>>5;
    int warp=blockIdx.x*warps+(threadIdx.x>>5),stride=gridDim.x*warps;
    for(int g=blockIdx.x;g<ngiants;g+=gridDim.x){
        int4 item=__ldg(giants+g);unsigned long long a=0,d=0;
        for(int j=threadIdx.x;j<item.z;j+=blockDim.x*8){
            int nodes[8],parts[8];
#pragma unroll
            for(int k=0;k<8;++k)nodes[k]=j+k*blockDim.x<item.z?__ldg(pins+item.y+j+k*blockDim.x):-1;
#pragma unroll
            for(int k=0;k<8;++k)parts[k]=(unsigned)nodes[k]<(unsigned)nn?__ldcg(partition+nodes[k]):-1;
#pragma unroll
            for(int k=0;k<8;++k)if((unsigned)parts[k]<64u){unsigned long long bit=1ULL<<parts[k];d|=a&bit;a|=bit;}
        }
        for(int off=16;off;off>>=1){unsigned long long b=__shfl_xor_sync(0xffffffffu,a,off),e=__shfl_xor_sync(0xffffffffu,d,off);d|=e|(a&b);a|=b;}
        if(lane==0){partial[2*(threadIdx.x>>5)]=a;partial[2*(threadIdx.x>>5)+1]=d;}
        __syncthreads();
        if(threadIdx.x==0){
            unsigned long long aa=0,dd=0;
            for(int w=0;w<warps;++w){unsigned long long b=partial[2*w],e=partial[2*w+1];dd|=e|(aa&b);aa|=b;}
            all[item.x]=aa;dbl[item.x]=dd;
        }
        __syncthreads();
    }
    for(int t=warp;t<tasks;t+=stride){
        int4 item=__ldg(slots+t*32+lane);
        switch(item.w){
            case 0:flags10_task<1>(item,lane,nn,pins,partition,all,dbl);break;
            case 1:flags10_task<2>(item,lane,nn,pins,partition,all,dbl);break;
            case 2:flags10_task<4>(item,lane,nn,pins,partition,all,dbl);break;
            case 3:flags10_task<8>(item,lane,nn,pins,partition,all,dbl);break;
            case 4:flags10_task<16>(item,lane,nn,pins,partition,all,dbl);break;
            default:flags10_task<32>(item,lane,nn,pins,partition,all,dbl);break;
        }
    }
}

// CTA barriers publish each block's writes to its leader and distribute the
// leader's acquire to its peers. Device-scope acq_rel RMWs form a release chain
// across block leaders; polling the completed epoch acquires that chain.
static __device__ __forceinline__ unsigned acqbar_load_10k(const unsigned *p){
    unsigned value;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(p):"memory");return value;
}
static __device__ __forceinline__ void acqbar_10k(unsigned *gb,unsigned target){
    __syncthreads();
    if(threadIdx.x==0){
        unsigned previous;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;":"=r"(previous):"l"(gb):"memory");
        while(acqbar_load_10k(gb)<target)__nanosleep(20);
    }
    __syncthreads();
}

#undef FF10K_GRID_BARRIER
#define FF10K_GRID_BARRIER(g,t) acqbar_10k((g),(t))

#undef FF10K_GRID_BARRIER20
#define FF10K_GRID_BARRIER20(g,t) acqbar_10k((g),(t))
extern "C" __global__ __launch_bounds__(512, 1) void exact_round_loop_10k(
    const int num_hyperedges,
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *node_hyperedges,
    const int *node_offsets,
    int *partition,
    int *nodes_in_part,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int initial_barrier_base,
    const int round_start, const int round_end, const int global_start,
    const int batch_rounds, const int move_limit, const int tabu_tenure,
    const int initial_stagnant, int *tabu_until, int *max_gain_buf, int *cand,
    unsigned long long *bucket_keys, int *bucket_counts,
    unsigned long long *selected_keys, int *ordered_moves, int *control
, const int4 *fe_slots10, const int n_fe_tasks10,
    const int4 *fe_giants10, const int n_fe_giants10
, const int4 *mv_units10,const int n_low10,const int n_mid10,const int n_high10
) {
    // Keep the original node iteration; the optional class-list buffers are unused.

    __shared__ ExactScratch_10k exact_work;
    __shared__ int exact_bucket_starts[65],exact_radix_ctl[4];
    __shared__ int exact_keeps[64],exact_offsets[65],exact_meta[4],s_warp[32];
    unsigned int nbar=0;
    int rounds_done=0,reason=0,total_moves=0,stagnant=initial_stagnant;
    const int slack_early=8,slack_mid=4,slack_late=2;
    const int tabu_mark_base=0,tabu_mark_mult=1;
    for(int step=0;step<batch_rounds && round_start+step<round_end;++step){
        const int round_idx=round_start+step;
        const unsigned int barrier_target=initial_barrier_base+(nbar+1)*gridDim.x;

    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64], exact_size_planes[32];
    if (threadIdx.x < 64) {
        shared_nodes_in_part[threadIdx.x] = threadIdx.x < num_parts ? __ldcg(&nodes_in_part[threadIdx.x]) : 0;
    }
    exact_target_masks_10k(num_nodes,num_parts,max_part_size,shared_nodes_in_part,
                           &exact_free,exact_bonus,exact_size_planes);
    // `nodes_in_part` is an INPUT of this kernel and is never written by it, so
    // the barrier's __syncthreads() below is all the publication this needs
    // (same structure as fused_flags_moves_20k).
    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_global = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int warp_stride = gridDim.x * (blockDim.x >> 5);
    if (gtid < 64) bucket_counts[gtid] = 0;
    if (gtid == 0) { max_gain_buf[0]=0; cand[0]=0; }
    int blk_max=0;


    flags10_classed(num_nodes,hyperedge_nodes,partition,fe_slots10,n_fe_tasks10,
                     fe_giants10,n_fe_giants10,edge_flags_all,edge_flags_double);

    FF10K_GRID_BARRIER20(grid_barrier, barrier_target);

    // ============ phase 2: compute_refinement_moves_optimized_10k ===========
    // wave14 (I5 + I10): identical arithmetic, no local memory, 4-deep pin walk.
    for (int node = blockIdx.x * blockDim.x + threadIdx.x;
         node < num_nodes;
         node += stride) {
        int start = __ldg(&node_offsets[node]);
        int end = __ldg(&node_offsets[node+1]);
        int node_degree=end-start;
        // Disjoint ownership: this pass never writes a high-degree node.
        if(node_degree>16)continue;
        move_priorities[node] = 0;
        int current_part = __ldcg(&partition[node]);
        if ((unsigned)current_part < (unsigned)num_parts && shared_nodes_in_part[current_part] > 1) {


            if (node_degree > 0) {
                int degree_weight = node_degree > 255 ? 255 : node_degree;
                unsigned long long current_bit = 1ULL << current_part;

                FF10K_BS_DECL

                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                int j = start;
                for (; j + 3 < end; j += 4) {
                    const int h0_ = __ldg(&node_hyperedges[j]);
                    const int h1_ = __ldg(&node_hyperedges[j + 1]);
                    const int h2_ = __ldg(&node_hyperedges[j + 2]);
                    const int h3_ = __ldg(&node_hyperedges[j + 3]);
                    const unsigned long long a0_ = __ldcg(&edge_flags_all[h0_]);
                    const unsigned long long d0_ = __ldcg(&edge_flags_double[h0_]);
                    const unsigned long long a1_ = __ldcg(&edge_flags_all[h1_]);
                    const unsigned long long d1_ = __ldcg(&edge_flags_double[h1_]);
                    const unsigned long long a2_ = __ldcg(&edge_flags_all[h2_]);
                    const unsigned long long d2_ = __ldcg(&edge_flags_double[h2_]);
                    const unsigned long long a3_ = __ldcg(&edge_flags_all[h3_]);
                    const unsigned long long d3_ = __ldcg(&edge_flags_double[h3_]);
                    C10K_SMALL_PIN(a0_, d0_);
                    C10K_SMALL_PIN(a1_, d1_);
                    C10K_SMALL_PIN(a2_, d2_);
                    C10K_SMALL_PIN(a3_, d3_);
                }
                for (; j < end; j++) {
                    const int hy_ = __ldg(&node_hyperedges[j]);
                    const unsigned long long at_ = __ldcg(&edge_flags_all[hy_]);
                    const unsigned long long dt_ = __ldcg(&edge_flags_double[hy_]);
                    C10K_SMALL_PIN(at_, dt_);
                }

                
                // This degree class cannot populate higher counter planes.
                pc5=0ULL; pc6=0ULL; pc7=0ULL; pc8=0ULL; pc9=0ULL; pc10=0ULL; pc11=0ULL; pc12=0ULL; pc13=0ULL; pc14=0ULL; pc15=0ULL; dc5=0ULL; dc6=0ULL; dc7=0ULL; dc8=0ULL; dc9=0ULL; dc10=0ULL; dc11=0ULL; dc12=0ULL; dc13=0ULL; dc14=0ULL; dc15=0ULL;
                int best_gain = -999999;
                int best_target = current_part;

                if (node_degree <= 65531 && num_nodes > 0) {
                    unsigned long long active = cand_mask & exact_free;
                    if (active) {
                        unsigned long long carry = exact_bonus[current_part], tmp;
                        tmp=pc2&carry; pc2^=carry; carry=tmp;
                        tmp=pc3&carry; pc3^=carry; carry=tmp;
                        tmp=pc4&carry; pc4^=carry; carry=tmp;
                        best_gain=0;
                        { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
                        { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
                        { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
                        { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
                        { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
                        best_gain-=count_current_present;
                        { unsigned long long m=active&dc4; if(m)active=m; }
                        { unsigned long long m=active&dc3; if(m)active=m; }
                        { unsigned long long m=active&dc2; if(m)active=m; }
                        { unsigned long long m=active&dc1; if(m)active=m; }
                        { unsigned long long m=active&dc0; if(m)active=m; }
                        for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                            unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
                        }
                        unsigned long long x1=active&0x2222222222222222ULL;
                        unsigned long long x2=active&0x4444444444444444ULL;
                        unsigned long long x3=active&0x8888888888888888ULL;
                        unsigned long long perm=(active&0x1111111111111111ULL)|
                            ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
                        unsigned shift=(unsigned)node&63;
                        unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
                        unsigned h=(unsigned)(__ffsll(hashed)-1);
                        best_target=(int)(((h-(unsigned)node)*49u)&63u);
                    }
                } else {
                while (cand_mask) {
                    int target_part = __ffsll(cand_mask) - 1;
                    cand_mask &= (cand_mask - 1);

                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count = FF10K_PC_GET(target_part);
                    int p_double = FF10K_DC_GET(target_part);

                    int basic_gain = p_count - count_current_present;
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    bool better = (total_gain > best_gain);
                    if (!better && total_gain == best_gain) {
                        int best_double = FF10K_DC_GET(best_target);
                        if (p_double > best_double) {
                            better = true;
                        } else if (p_double == best_double) {
                            int best_target_size = shared_nodes_in_part[best_target];
                            if (target_size < best_target_size) {
                                better = true;
                            } else if (target_size == best_target_size) {
                                int hash_tgt = (target_part * 17 + node) & 63;
                                int hash_best = (best_target * 17 + node) & 63;
                                if (hash_tgt < hash_best) better = true;
                            }
                        }
                    }

                    if (better) {
                        best_gain = total_gain;
                        best_target = target_part;
                    }
                }


                }

                if (best_gain >= -1 && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
                }
            }
        }
        int key = move_priorities[node];
        if (key > 0) blk_max = max(blk_max, (key >> 16) - 1000);
    }
    // Degree 17..128: four lanes per node, five/eight-bit exact partial sums.
    // All lanes execute the reductions, including inactive/padding groups.
    {
        int lane=threadIdx.x&31,group=lane>>2,within=lane&3;
        int warp=(int)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
        int total_warps=(int)gridDim.x*(blockDim.x>>5);
        for(int base_node=warp*8;base_node<num_nodes;base_node+=total_warps*8){
            int node=base_node+group;
            int start=node<num_nodes?__ldg(&node_offsets[node]):0;
            int end=node<num_nodes?__ldg(&node_offsets[node+1]):0;
            int node_degree=end-start;
            bool in_class=node<num_nodes && node_degree>16 && node_degree<=128;
            if(!__any_sync(0xffffffffu,in_class))continue;
            int current_part=node<num_nodes?__ldcg(&partition[node]):-1;
            bool go=in_class && (unsigned)current_part<(unsigned)num_parts && shared_nodes_in_part[current_part]>1;
            unsigned long long current_bit=go?(1ULL<<current_part):0ULL;
            FF10K_BS_DECL
            unsigned long long cand_mask=0ULL;int count_current_present=0;
            if(go)for(int j=start+within;j<end;j+=16){
                int j0=j+0;bool v0=j0<end;int h0=v0?__ldg(&node_hyperedges[j0]):0;
                int j1=j+4;bool v1=j1<end;int h1=v1?__ldg(&node_hyperedges[j1]):0;
                int j2=j+8;bool v2=j2<end;int h2=v2?__ldg(&node_hyperedges[j2]):0;
                int j3=j+12;bool v3=j3<end;int h3=v3?__ldg(&node_hyperedges[j3]):0;
                unsigned long long a0=v0?__ldcg(&edge_flags_all[h0]):0ULL,d0=v0?__ldcg(&edge_flags_double[h0]):0ULL;
                unsigned long long a1=v1?__ldcg(&edge_flags_all[h1]):0ULL,d1=v1?__ldcg(&edge_flags_double[h1]):0ULL;
                unsigned long long a2=v2?__ldcg(&edge_flags_all[h2]):0ULL,d2=v2?__ldcg(&edge_flags_double[h2]):0ULL;
                unsigned long long a3=v3?__ldcg(&edge_flags_all[h3]):0ULL,d3=v3?__ldcg(&edge_flags_double[h3]):0ULL;
                if(v0){C10K_MID_PIN(a0,d0);}
                if(v1){C10K_MID_PIN(a1,d1);}
                if(v2){C10K_MID_PIN(a2,d2);}
                if(v3){C10K_MID_PIN(a3,d3);}
            }
            for(int off=2;off;off>>=1){
                cand_mask|=__shfl_xor_sync(0xffffffffu,cand_mask,off);
                count_current_present+=__shfl_xor_sync(0xffffffffu,count_current_present,off);
                {unsigned long long other,sum,carry,next;
                  other=__shfl_xor_sync(0xffffffffu,pc0,off);
                  sum=pc0^other;carry=pc0&other;pc0=sum;
                  other=__shfl_xor_sync(0xffffffffu,pc1,off);
                  sum=pc1^other;next=(pc1&other)|(sum&carry);pc1=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc2,off);
                  sum=pc2^other;next=(pc2&other)|(sum&carry);pc2=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc3,off);
                  sum=pc3^other;next=(pc3&other)|(sum&carry);pc3=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc4,off);
                  sum=pc4^other;next=(pc4&other)|(sum&carry);pc4=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc5,off);
                  sum=pc5^other;next=(pc5&other)|(sum&carry);pc5=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc6,off);
                  sum=pc6^other;next=(pc6&other)|(sum&carry);pc6=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc7,off);
                  sum=pc7^other;next=(pc7&other)|(sum&carry);pc7=sum^carry;carry=next;
                }
                {unsigned long long other,sum,carry,next;
                  other=__shfl_xor_sync(0xffffffffu,dc0,off);
                  sum=dc0^other;carry=dc0&other;dc0=sum;
                  other=__shfl_xor_sync(0xffffffffu,dc1,off);
                  sum=dc1^other;next=(dc1&other)|(sum&carry);dc1=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc2,off);
                  sum=dc2^other;next=(dc2&other)|(sum&carry);dc2=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc3,off);
                  sum=dc3^other;next=(dc3&other)|(sum&carry);dc3=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc4,off);
                  sum=dc4^other;next=(dc4&other)|(sum&carry);dc4=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc5,off);
                  sum=dc5^other;next=(dc5&other)|(sum&carry);dc5=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc6,off);
                  sum=dc6^other;next=(dc6&other)|(sum&carry);dc6=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc7,off);
                  sum=dc7^other;next=(dc7&other)|(sum&carry);dc7=sum^carry;carry=next;
                }
            }
            int outkey=0;
            if(go){int degree_weight=min(255,node_degree);
    
                // This degree class cannot populate higher counter planes.
                pc8=0ULL; pc9=0ULL; pc10=0ULL; pc11=0ULL; pc12=0ULL; pc13=0ULL; pc14=0ULL; pc15=0ULL; dc8=0ULL; dc9=0ULL; dc10=0ULL; dc11=0ULL; dc12=0ULL; dc13=0ULL; dc14=0ULL; dc15=0ULL;
                int best_gain = -999999;
    int best_target = current_part;

    if (node_degree <= 65531 && num_nodes > 0) {
        unsigned long long active = cand_mask & exact_free;
        if (active) {
            unsigned long long carry = exact_bonus[current_part], tmp;
            tmp=pc2&carry; pc2^=carry; carry=tmp;
            tmp=pc3&carry; pc3^=carry; carry=tmp;
            tmp=pc4&carry; pc4^=carry; carry=tmp;
            tmp=pc5&carry; pc5^=carry; carry=tmp;
            tmp=pc6&carry; pc6^=carry; carry=tmp;
            tmp=pc7&carry; pc7^=carry; carry=tmp;
            best_gain=0;
            { unsigned long long m=active&pc7; if(m){active=m;best_gain|=128;} }
            { unsigned long long m=active&pc6; if(m){active=m;best_gain|=64;} }
            { unsigned long long m=active&pc5; if(m){active=m;best_gain|=32;} }
            { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
            { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
            { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
            { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
            { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
            best_gain-=count_current_present;
            { unsigned long long m=active&dc7; if(m)active=m; }
            { unsigned long long m=active&dc6; if(m)active=m; }
            { unsigned long long m=active&dc5; if(m)active=m; }
            { unsigned long long m=active&dc4; if(m)active=m; }
            { unsigned long long m=active&dc3; if(m)active=m; }
            { unsigned long long m=active&dc2; if(m)active=m; }
            { unsigned long long m=active&dc1; if(m)active=m; }
            { unsigned long long m=active&dc0; if(m)active=m; }
            for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
            }
            unsigned long long x1=active&0x2222222222222222ULL;
            unsigned long long x2=active&0x4444444444444444ULL;
            unsigned long long x3=active&0x8888888888888888ULL;
            unsigned long long perm=(active&0x1111111111111111ULL)|
                ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
            unsigned shift=(unsigned)node&63;
            unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
            unsigned h=(unsigned)(__ffsll(hashed)-1);
            best_target=(int)(((h-(unsigned)node)*49u)&63u);
        }
    } else {
    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = FF10K_PC_GET(target_part);
        int p_double = FF10K_DC_GET(target_part);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = FF10K_DC_GET(best_target);
            if (p_double > best_double) {
                better = true;
            } else if (p_double == best_double) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better = true;
                }
            }
        }

        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }


    }

    if (best_gain >= -1 && best_target != current_part) {
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        outkey = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
            }
            if(within==0 && in_class){move_priorities[node]=outkey;if(outkey>0)blk_max=max(blk_max,(outkey>>16)-1000);}
        }
    }

    // High-degree nodes get a whole warp; small-node threads never touch
    // their output slots. This removes long divergent pin walks without
    // adding a grid barrier or changing any sampled/full-degree semantics.
    {
        int lane=threadIdx.x&31;
        int warp=(int)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
        int warp_stride=(int)gridDim.x*(blockDim.x>>5);
        for(int node=warp;node<num_nodes;node+=warp_stride){
            int start=__ldg(&node_offsets[node]),end=__ldg(&node_offsets[node+1]);
            int node_degree=end-start;
            if(node_degree<=128)continue;
            int outkey=0;
            int current_part=__ldcg(&partition[node]);
            if((unsigned)current_part<(unsigned)num_parts && shared_nodes_in_part[current_part]>1){
                int degree_weight=min(255,node_degree);
                unsigned long long current_bit=1ULL<<current_part;
                FF10K_BS_DECL
                unsigned long long cand_mask=0ULL;int count_current_present=0;
                for(int base=start;base<end;base+=128){
                    int j0=base+lane+0;bool v0=j0<end;int h0=v0?__ldg(&node_hyperedges[j0]):0;
                    int j1=base+lane+32;bool v1=j1<end;int h1=v1?__ldg(&node_hyperedges[j1]):0;
                    int j2=base+lane+64;bool v2=j2<end;int h2=v2?__ldg(&node_hyperedges[j2]):0;
                    int j3=base+lane+96;bool v3=j3<end;int h3=v3?__ldg(&node_hyperedges[j3]):0;
                    unsigned long long a0=v0?__ldcg(&edge_flags_all[h0]):0ULL,d0=v0?__ldcg(&edge_flags_double[h0]):0ULL;
                    unsigned long long a1=v1?__ldcg(&edge_flags_all[h1]):0ULL,d1=v1?__ldcg(&edge_flags_double[h1]):0ULL;
                    unsigned long long a2=v2?__ldcg(&edge_flags_all[h2]):0ULL,d2=v2?__ldcg(&edge_flags_double[h2]):0ULL;
                    unsigned long long a3=v3?__ldcg(&edge_flags_all[h3]):0ULL,d3=v3?__ldcg(&edge_flags_double[h3]):0ULL;
                    if(v0){FF10K_PIN(a0,d0);}
                    if(v1){FF10K_PIN(a1,d1);}
                    if(v2){FF10K_PIN(a2,d2);}
                    if(v3){FF10K_PIN(a3,d3);}
                }
                for(int off=16;off;off>>=1){
                    cand_mask|=__shfl_xor_sync(0xffffffffu,cand_mask,off);
                    count_current_present+=__shfl_xor_sync(0xffffffffu,count_current_present,off);
                    { unsigned long long other,carry,sum,next;
                      other=__shfl_xor_sync(0xffffffffu,pc0,off);
                      sum=pc0^other;carry=pc0&other;pc0=sum;
                      other=__shfl_xor_sync(0xffffffffu,pc1,off);
                      sum=pc1^other;next=(pc1&other)|(sum&carry);pc1=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc2,off);
                      sum=pc2^other;next=(pc2&other)|(sum&carry);pc2=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc3,off);
                      sum=pc3^other;next=(pc3&other)|(sum&carry);pc3=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc4,off);
                      sum=pc4^other;next=(pc4&other)|(sum&carry);pc4=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc5,off);
                      sum=pc5^other;next=(pc5&other)|(sum&carry);pc5=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc6,off);
                      sum=pc6^other;next=(pc6&other)|(sum&carry);pc6=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc7,off);
                      sum=pc7^other;next=(pc7&other)|(sum&carry);pc7=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc8,off);
                      sum=pc8^other;next=(pc8&other)|(sum&carry);pc8=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc9,off);
                      sum=pc9^other;next=(pc9&other)|(sum&carry);pc9=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc10,off);
                      sum=pc10^other;next=(pc10&other)|(sum&carry);pc10=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc11,off);
                      sum=pc11^other;next=(pc11&other)|(sum&carry);pc11=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc12,off);
                      sum=pc12^other;next=(pc12&other)|(sum&carry);pc12=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc13,off);
                      sum=pc13^other;next=(pc13&other)|(sum&carry);pc13=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc14,off);
                      sum=pc14^other;next=(pc14&other)|(sum&carry);pc14=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc15,off);
                      sum=pc15^other;next=(pc15&other)|(sum&carry);pc15=sum^carry;carry=next;
                    }
                    { unsigned long long other,carry,sum,next;
                      other=__shfl_xor_sync(0xffffffffu,dc0,off);
                      sum=dc0^other;carry=dc0&other;dc0=sum;
                      other=__shfl_xor_sync(0xffffffffu,dc1,off);
                      sum=dc1^other;next=(dc1&other)|(sum&carry);dc1=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc2,off);
                      sum=dc2^other;next=(dc2&other)|(sum&carry);dc2=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc3,off);
                      sum=dc3^other;next=(dc3&other)|(sum&carry);dc3=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc4,off);
                      sum=dc4^other;next=(dc4&other)|(sum&carry);dc4=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc5,off);
                      sum=dc5^other;next=(dc5&other)|(sum&carry);dc5=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc6,off);
                      sum=dc6^other;next=(dc6&other)|(sum&carry);dc6=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc7,off);
                      sum=dc7^other;next=(dc7&other)|(sum&carry);dc7=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc8,off);
                      sum=dc8^other;next=(dc8&other)|(sum&carry);dc8=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc9,off);
                      sum=dc9^other;next=(dc9&other)|(sum&carry);dc9=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc10,off);
                      sum=dc10^other;next=(dc10&other)|(sum&carry);dc10=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc11,off);
                      sum=dc11^other;next=(dc11&other)|(sum&carry);dc11=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc12,off);
                      sum=dc12^other;next=(dc12&other)|(sum&carry);dc12=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc13,off);
                      sum=dc13^other;next=(dc13&other)|(sum&carry);dc13=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc14,off);
                      sum=dc14^other;next=(dc14&other)|(sum&carry);dc14=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc15,off);
                      sum=dc15^other;next=(dc15&other)|(sum&carry);dc15=sum^carry;carry=next;
                    }
                }
    int best_gain = -999999;
    int best_target = current_part;

    if (node_degree <= 65531 && num_nodes > 0) {
        unsigned long long active = cand_mask & exact_free;
        if (active) {
            unsigned long long carry = exact_bonus[current_part], tmp;
            tmp=pc2&carry; pc2^=carry; carry=tmp;
            tmp=pc3&carry; pc3^=carry; carry=tmp;
            tmp=pc4&carry; pc4^=carry; carry=tmp;
            tmp=pc5&carry; pc5^=carry; carry=tmp;
            tmp=pc6&carry; pc6^=carry; carry=tmp;
            tmp=pc7&carry; pc7^=carry; carry=tmp;
            tmp=pc8&carry; pc8^=carry; carry=tmp;
            tmp=pc9&carry; pc9^=carry; carry=tmp;
            tmp=pc10&carry; pc10^=carry; carry=tmp;
            tmp=pc11&carry; pc11^=carry; carry=tmp;
            tmp=pc12&carry; pc12^=carry; carry=tmp;
            tmp=pc13&carry; pc13^=carry; carry=tmp;
            tmp=pc14&carry; pc14^=carry; carry=tmp;
            tmp=pc15&carry; pc15^=carry; carry=tmp;
            best_gain=0;
            { unsigned long long m=active&pc15; if(m){active=m;best_gain|=32768;} }
            { unsigned long long m=active&pc14; if(m){active=m;best_gain|=16384;} }
            { unsigned long long m=active&pc13; if(m){active=m;best_gain|=8192;} }
            { unsigned long long m=active&pc12; if(m){active=m;best_gain|=4096;} }
            { unsigned long long m=active&pc11; if(m){active=m;best_gain|=2048;} }
            { unsigned long long m=active&pc10; if(m){active=m;best_gain|=1024;} }
            { unsigned long long m=active&pc9; if(m){active=m;best_gain|=512;} }
            { unsigned long long m=active&pc8; if(m){active=m;best_gain|=256;} }
            { unsigned long long m=active&pc7; if(m){active=m;best_gain|=128;} }
            { unsigned long long m=active&pc6; if(m){active=m;best_gain|=64;} }
            { unsigned long long m=active&pc5; if(m){active=m;best_gain|=32;} }
            { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
            { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
            { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
            { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
            { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
            best_gain-=count_current_present;
            { unsigned long long m=active&dc15; if(m)active=m; }
            { unsigned long long m=active&dc14; if(m)active=m; }
            { unsigned long long m=active&dc13; if(m)active=m; }
            { unsigned long long m=active&dc12; if(m)active=m; }
            { unsigned long long m=active&dc11; if(m)active=m; }
            { unsigned long long m=active&dc10; if(m)active=m; }
            { unsigned long long m=active&dc9; if(m)active=m; }
            { unsigned long long m=active&dc8; if(m)active=m; }
            { unsigned long long m=active&dc7; if(m)active=m; }
            { unsigned long long m=active&dc6; if(m)active=m; }
            { unsigned long long m=active&dc5; if(m)active=m; }
            { unsigned long long m=active&dc4; if(m)active=m; }
            { unsigned long long m=active&dc3; if(m)active=m; }
            { unsigned long long m=active&dc2; if(m)active=m; }
            { unsigned long long m=active&dc1; if(m)active=m; }
            { unsigned long long m=active&dc0; if(m)active=m; }
            for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
            }
            unsigned long long x1=active&0x2222222222222222ULL;
            unsigned long long x2=active&0x4444444444444444ULL;
            unsigned long long x3=active&0x8888888888888888ULL;
            unsigned long long perm=(active&0x1111111111111111ULL)|
                ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
            unsigned shift=(unsigned)node&63;
            unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
            unsigned h=(unsigned)(__ffsll(hashed)-1);
            best_target=(int)(((h-(unsigned)node)*49u)&63u);
        }
    } else {
    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = FF10K_PC_GET(target_part);
        int p_double = FF10K_DC_GET(target_part);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = FF10K_DC_GET(best_target);
            if (p_double > best_double) {
                better = true;
            } else if (p_double == best_double) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better = true;
                }
            }
        }

        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }


    }

    if (best_gain >= -1 && best_target != current_part) {
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        outkey = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
            }
            if(lane==0){move_priorities[node]=outkey;if(outkey>0)blk_max=max(blk_max,(outkey>>16)-1000);}
        }
    }


        ++nbar;
        int maximum = ff10k_block_max(blk_max,s_warp);
        if(threadIdx.x==0)atomicMax(max_gain_buf,maximum);
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        int aspiration=max(1,(__ldcg(max_gain_buf)*3)/4);
        int local_count=0;
        for(int node=gtid;node<num_nodes;node+=stride){
            int key=__ldcg(&move_priorities[node]);
            cand[1+2*node]=node; cand[2+2*node]=-1;
            if(key>0){
                int gain=(key>>16)-1000;
                if(__ldcg(&tabu_until[node]) <= global_start+rounds_done+1 || gain>=aspiration){
                    ++local_count;
                    cand[2+2*node]=key;
                    int target=key&63;
                    int pos=grouped_bucket_10k(bucket_counts,target);
                    if(pos<4096)bucket_keys[target*4096+pos]=
                        ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
                }
            }
        }
        int count=ff10k_block_sum(local_count,s_warp);
        if(threadIdx.x==0)atomicAdd(cand,count);
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        // Parallel exclusive prefixes replace 64 dependent scalar loads
        // in every resident block. All sums are exact bounded integers.
        int meta_n=threadIdx.x<64?__ldcg(&bucket_counts[threadIdx.x]):0;
        int meta_keep=0;
        if(threadIdx.x<64){
            int p=threadIdx.x;
            int slack=round_idx<64?slack_early:(round_idx<256?slack_mid:slack_late);
            int q=max(1,max(0,max_part_size-shared_nodes_in_part[p])+slack);
            meta_keep=p<num_parts?min(meta_n,q):0;
            exact_keeps[p]=meta_keep;
            int ns=meta_n,ks=meta_keep;
            for(int off=1;off<32;off<<=1){
                int a=__shfl_up_sync(0xffffffffu,ns,off),b=__shfl_up_sync(0xffffffffu,ks,off);
                if((p&31)>=off){ns+=a;ks+=b;}
            }
            exact_bucket_starts[p+1]=ns;exact_offsets[p+1]=ks;
        }
        if(threadIdx.x==0){exact_bucket_starts[0]=0;exact_offsets[0]=0;}
        __syncthreads();
        if(threadIdx.x>=32 && threadIdx.x<64){
            exact_bucket_starts[threadIdx.x+1]+=exact_bucket_starts[32];
            exact_offsets[threadIdx.x+1]+=exact_offsets[32];
        }
        int meta_compact=__syncthreads_or(threadIdx.x<64 && meta_n>4096);
        int meta_overflow=__syncthreads_or(threadIdx.x<64 && meta_keep>4096);
        if(threadIdx.x==0){
            exact_meta[0]=__ldcg(&cand[0]);exact_meta[1]=exact_offsets[64];
            exact_meta[2]=meta_overflow;exact_meta[3]=meta_compact;
        }
        if(!meta_compact && threadIdx.x<65)exact_bucket_starts[threadIdx.x]=threadIdx.x*4096;
        __syncthreads();
        int N = exact_meta[0], M = exact_meta[1];
        if (N == 0) { reason = 1; break; }
        int adaptive_limit = round_idx < 50 ? move_limit / 2 :
                            (round_idx < 200 ? move_limit : move_limit / 3);
        // Even when the original global k_base clips N, quota selection is
        // identical if its candidate window still contains ALL N entries and
        // the union of per-target prefixes does not exceed k_base.
        bool full_quota_equivalent = N <= adaptive_limit ||
            (N <= adaptive_limit + 16384 && M <= adaptive_limit);
        if (!full_quota_equivalent || exact_meta[2] || M <= 0 || num_nodes<256) { reason = 4; break; }


        if(exact_meta[3]){
            if(gtid<64)bucket_counts[64+gtid]=0;
            ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
            for(int i=gtid;i<num_nodes;i+=stride){
                int key=__ldcg(&cand[2+2*i]);
                if(key<0)continue;
                int node=__ldcg(&cand[1+2*i]),target=key&63;
                int pos=grouped_bucket_10k(bucket_counts + 64,target);
                bucket_keys[exact_bucket_starts[target]+pos]=
                    ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
            }
            ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        }
        for(int p=blockIdx.x;p<num_parts && p<64;p+=gridDim.x){
            int n=__ldcg(&bucket_counts[p]),keep=exact_keeps[p],off=exact_offsets[p];
            int start=exact_bucket_starts[p];
            if(keep==0)continue;
            exact_sort_prefix_10k(&bucket_keys[start],n,keep,&exact_work,exact_radix_ctl);
            for(int j=threadIdx.x;j<keep;j+=blockDim.x)selected_keys[off+j]=__ldcg(&bucket_keys[start+j]);
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

        // Try a key-ordered sparse replay without materializing the
        // globally merged list. Failure leaves all solver state untouched.
        if(blockIdx.x==0) {
            int result=keyed_apply_10k(M,num_nodes,num_parts,max_part_size,
                partition,nodes_in_part,selected_keys,exact_keeps,exact_offsets,
                tabu_until,global_start + rounds_done + 1,tabu_tenure,tabu_mark_base,tabu_mark_mult,&exact_work);
            if(threadIdx.x==0)control[15]=result;
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        if(__ldcg(&control[15])<0) {
        // Replicate the small kept-prefix table into shared memory. It is
        // immutable until the merge barrier; the next phase reuses the union.
        const bool shared_merge = M <= 4096;
        if(shared_merge){
            for(int j=threadIdx.x;j<M;j+=blockDim.x)exact_work.sort[j]=__ldcg(&selected_keys[j]);
            __syncthreads();
        }
        // Merge-rank the 64 sorted prefixes. Each warp owns one output item;
        // each lane binary-searches two lists. No all-pairs candidate ranking.
        for (int i = warp_global; i < M; i += warp_stride) {
            unsigned long long key = shared_merge ? exact_work.sort[i] : __ldcg(&selected_keys[i]);
            int rank = 0;
            for (int p = lane; p < num_parts && p < 64; p += 32) {
                int lo = 0, hi = exact_keeps[p];
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    unsigned long long other = shared_merge ? exact_work.sort[exact_offsets[p]+mid] : __ldcg(&bucket_keys[exact_bucket_starts[p] + mid]);
                    if (other < key) lo = mid + 1; else hi = mid;
                }
                rank += lo;
            }
            for (int offset = 16; offset > 0; offset >>= 1) rank += __shfl_xor_sync(0xffffffffu, rank, offset);
            if (lane == 0) {
                int node = (int)(unsigned int)key;
                int target = (((unsigned int)(key >> 32)) & 63) ^ 63;
                ordered_moves[rank] = (node << 6) | target;
                int free=max(0,max_part_size-shared_nodes_in_part[target]);
                int risk_index=i-exact_offsets[target]-free;
                if(risk_index>=0 && risk_index<8) cand[1+target*8+risk_index]=rank;
            }
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        if (blockIdx.x == 0) {
            int accepted = exact_apply_critical_10k(M, num_nodes, num_parts, max_part_size,
                partition, nodes_in_part, ordered_moves, tabu_until, global_start + rounds_done + 1,
                tabu_tenure, tabu_mark_base, tabu_mark_mult, &exact_work, s_warp, cand+1, exact_keeps);
            if (threadIdx.x == 0) control[15] = accepted;
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        }
        int accepted = __ldcg(&control[15]);
        // A zero-execution clipped round must run the original secondary
        // tail fallback (and failed tabu marks). No partition or tabu entry
        // changed when accepted==0, so replay it unchanged on the host.
        if(accepted==0 && N>adaptive_limit){reason=4;break;}
        total_moves += accepted;
        ++rounds_done;
        if (accepted == 0) { reason = 2; break; }
        stagnant = 0;

    }
    if(blockIdx.x==0 && threadIdx.x==0){
        control[0]=reason;control[1]=rounds_done;control[2]=(int)nbar;
        control[3]=total_moves;control[4]=stagnant;
    }
}

// Exact 10k tail policy: no aspiration filter, unchanged quota and marking.
extern "C" __global__ __launch_bounds__(512, 1) void exact_tail_loop_10k(
    const int num_hyperedges,
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *node_hyperedges,
    const int *node_offsets,
    int *partition,
    int *nodes_in_part,
    unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int initial_barrier_base,
    const int round_start, const int round_end, const int global_start,
    const int batch_rounds, const int move_limit, const int tabu_tenure,
    const int initial_stagnant, int *tabu_until, int *max_gain_buf, int *cand,
    unsigned long long *bucket_keys, int *bucket_counts,
    unsigned long long *selected_keys, int *ordered_moves, int *control, const int min_gain, const int tail_slack
, const int4 *fe_slots10, const int n_fe_tasks10,
    const int4 *fe_giants10, const int n_fe_giants10
, const int4 *mv_units10,const int n_low10,const int n_mid10,const int n_high10
) {
    // Keep the original node iteration; the optional class-list buffers are unused.

    __shared__ ExactScratch_10k exact_work;
    __shared__ int exact_bucket_starts[65],exact_radix_ctl[4];
    __shared__ int exact_keeps[64],exact_offsets[65],exact_meta[4],s_warp[32];
    unsigned int nbar=0;
    int rounds_done=0,reason=0,total_moves=0,stagnant=initial_stagnant;
    const int slack_early=tail_slack,slack_mid=tail_slack,slack_late=tail_slack;
    const int tabu_mark_base=0,tabu_mark_mult=1;
    for(int step=0;step<batch_rounds && round_start+step<round_end;++step){
        const int round_idx=round_start+step;
        const unsigned int barrier_target=initial_barrier_base+(nbar+1)*gridDim.x;

    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64], exact_size_planes[32];
    if (threadIdx.x < 64) {
        shared_nodes_in_part[threadIdx.x] = threadIdx.x < num_parts ? __ldcg(&nodes_in_part[threadIdx.x]) : 0;
    }
    exact_target_masks_10k(num_nodes,num_parts,max_part_size,shared_nodes_in_part,
                           &exact_free,exact_bonus,exact_size_planes);
    // `nodes_in_part` is an INPUT of this kernel and is never written by it, so
    // the barrier's __syncthreads() below is all the publication this needs
    // (same structure as fused_flags_moves_20k).
    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_global = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int warp_stride = gridDim.x * (blockDim.x >> 5);
    if (gtid < 64) bucket_counts[gtid] = 0;
    if (gtid == 0) { max_gain_buf[0]=0; cand[0]=0; }
    int blk_max=0;


    flags10_classed(num_nodes,hyperedge_nodes,partition,fe_slots10,n_fe_tasks10,
                     fe_giants10,n_fe_giants10,edge_flags_all,edge_flags_double);

    FF10K_GRID_BARRIER20(grid_barrier, barrier_target);

    // ============ phase 2: compute_refinement_moves_optimized_10k ===========
    // wave14 (I5 + I10): identical arithmetic, no local memory, 4-deep pin walk.
    for (int node = blockIdx.x * blockDim.x + threadIdx.x;
         node < num_nodes;
         node += stride) {
        int start = __ldg(&node_offsets[node]);
        int end = __ldg(&node_offsets[node+1]);
        int node_degree=end-start;
        // Disjoint ownership: this pass never writes a high-degree node.
        if(node_degree>16)continue;
        move_priorities[node] = 0;
        int current_part = __ldcg(&partition[node]);
        if ((unsigned)current_part < (unsigned)num_parts && shared_nodes_in_part[current_part] > 1) {


            if (node_degree > 0) {
                int degree_weight = node_degree > 255 ? 255 : node_degree;
                unsigned long long current_bit = 1ULL << current_part;

                FF10K_BS_DECL

                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                int j = start;
                for (; j + 3 < end; j += 4) {
                    const int h0_ = __ldg(&node_hyperedges[j]);
                    const int h1_ = __ldg(&node_hyperedges[j + 1]);
                    const int h2_ = __ldg(&node_hyperedges[j + 2]);
                    const int h3_ = __ldg(&node_hyperedges[j + 3]);
                    const unsigned long long a0_ = __ldcg(&edge_flags_all[h0_]);
                    const unsigned long long d0_ = __ldcg(&edge_flags_double[h0_]);
                    const unsigned long long a1_ = __ldcg(&edge_flags_all[h1_]);
                    const unsigned long long d1_ = __ldcg(&edge_flags_double[h1_]);
                    const unsigned long long a2_ = __ldcg(&edge_flags_all[h2_]);
                    const unsigned long long d2_ = __ldcg(&edge_flags_double[h2_]);
                    const unsigned long long a3_ = __ldcg(&edge_flags_all[h3_]);
                    const unsigned long long d3_ = __ldcg(&edge_flags_double[h3_]);
                    C10K_SMALL_PIN(a0_, d0_);
                    C10K_SMALL_PIN(a1_, d1_);
                    C10K_SMALL_PIN(a2_, d2_);
                    C10K_SMALL_PIN(a3_, d3_);
                }
                for (; j < end; j++) {
                    const int hy_ = __ldg(&node_hyperedges[j]);
                    const unsigned long long at_ = __ldcg(&edge_flags_all[hy_]);
                    const unsigned long long dt_ = __ldcg(&edge_flags_double[hy_]);
                    C10K_SMALL_PIN(at_, dt_);
                }

                
                // This degree class cannot populate higher counter planes.
                pc5=0ULL; pc6=0ULL; pc7=0ULL; pc8=0ULL; pc9=0ULL; pc10=0ULL; pc11=0ULL; pc12=0ULL; pc13=0ULL; pc14=0ULL; pc15=0ULL; dc5=0ULL; dc6=0ULL; dc7=0ULL; dc8=0ULL; dc9=0ULL; dc10=0ULL; dc11=0ULL; dc12=0ULL; dc13=0ULL; dc14=0ULL; dc15=0ULL;
                int best_gain = -999999;
                int best_target = current_part;

                if (node_degree <= 65531 && num_nodes > 0) {
                    unsigned long long active = cand_mask & exact_free;
                    if (active) {
                        unsigned long long carry = exact_bonus[current_part], tmp;
                        tmp=pc2&carry; pc2^=carry; carry=tmp;
                        tmp=pc3&carry; pc3^=carry; carry=tmp;
                        tmp=pc4&carry; pc4^=carry; carry=tmp;
                        best_gain=0;
                        { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
                        { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
                        { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
                        { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
                        { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
                        best_gain-=count_current_present;
                        { unsigned long long m=active&dc4; if(m)active=m; }
                        { unsigned long long m=active&dc3; if(m)active=m; }
                        { unsigned long long m=active&dc2; if(m)active=m; }
                        { unsigned long long m=active&dc1; if(m)active=m; }
                        { unsigned long long m=active&dc0; if(m)active=m; }
                        for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                            unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
                        }
                        unsigned long long x1=active&0x2222222222222222ULL;
                        unsigned long long x2=active&0x4444444444444444ULL;
                        unsigned long long x3=active&0x8888888888888888ULL;
                        unsigned long long perm=(active&0x1111111111111111ULL)|
                            ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
                        unsigned shift=(unsigned)node&63;
                        unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
                        unsigned h=(unsigned)(__ffsll(hashed)-1);
                        best_target=(int)(((h-(unsigned)node)*49u)&63u);
                    }
                } else {
                while (cand_mask) {
                    int target_part = __ffsll(cand_mask) - 1;
                    cand_mask &= (cand_mask - 1);

                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count = FF10K_PC_GET(target_part);
                    int p_double = FF10K_DC_GET(target_part);

                    int basic_gain = p_count - count_current_present;
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    bool better = (total_gain > best_gain);
                    if (!better && total_gain == best_gain) {
                        int best_double = FF10K_DC_GET(best_target);
                        if (p_double > best_double) {
                            better = true;
                        } else if (p_double == best_double) {
                            int best_target_size = shared_nodes_in_part[best_target];
                            if (target_size < best_target_size) {
                                better = true;
                            } else if (target_size == best_target_size) {
                                int hash_tgt = (target_part * 17 + node) & 63;
                                int hash_best = (best_target * 17 + node) & 63;
                                if (hash_tgt < hash_best) better = true;
                            }
                        }
                    }

                    if (better) {
                        best_gain = total_gain;
                        best_target = target_part;
                    }
                }


                }

                if (best_gain >= -1 && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
                }
            }
        }
        int key = move_priorities[node];
        if (key > 0) blk_max = max(blk_max, (key >> 16) - 1000);
    }
    // Degree 17..128: four lanes per node, five/eight-bit exact partial sums.
    // All lanes execute the reductions, including inactive/padding groups.
    {
        int lane=threadIdx.x&31,group=lane>>2,within=lane&3;
        int warp=(int)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
        int total_warps=(int)gridDim.x*(blockDim.x>>5);
        for(int base_node=warp*8;base_node<num_nodes;base_node+=total_warps*8){
            int node=base_node+group;
            int start=node<num_nodes?__ldg(&node_offsets[node]):0;
            int end=node<num_nodes?__ldg(&node_offsets[node+1]):0;
            int node_degree=end-start;
            bool in_class=node<num_nodes && node_degree>16 && node_degree<=128;
            if(!__any_sync(0xffffffffu,in_class))continue;
            int current_part=node<num_nodes?__ldcg(&partition[node]):-1;
            bool go=in_class && (unsigned)current_part<(unsigned)num_parts && shared_nodes_in_part[current_part]>1;
            unsigned long long current_bit=go?(1ULL<<current_part):0ULL;
            FF10K_BS_DECL
            unsigned long long cand_mask=0ULL;int count_current_present=0;
            if(go)for(int j=start+within;j<end;j+=16){
                int j0=j+0;bool v0=j0<end;int h0=v0?__ldg(&node_hyperedges[j0]):0;
                int j1=j+4;bool v1=j1<end;int h1=v1?__ldg(&node_hyperedges[j1]):0;
                int j2=j+8;bool v2=j2<end;int h2=v2?__ldg(&node_hyperedges[j2]):0;
                int j3=j+12;bool v3=j3<end;int h3=v3?__ldg(&node_hyperedges[j3]):0;
                unsigned long long a0=v0?__ldcg(&edge_flags_all[h0]):0ULL,d0=v0?__ldcg(&edge_flags_double[h0]):0ULL;
                unsigned long long a1=v1?__ldcg(&edge_flags_all[h1]):0ULL,d1=v1?__ldcg(&edge_flags_double[h1]):0ULL;
                unsigned long long a2=v2?__ldcg(&edge_flags_all[h2]):0ULL,d2=v2?__ldcg(&edge_flags_double[h2]):0ULL;
                unsigned long long a3=v3?__ldcg(&edge_flags_all[h3]):0ULL,d3=v3?__ldcg(&edge_flags_double[h3]):0ULL;
                if(v0){C10K_MID_PIN(a0,d0);}
                if(v1){C10K_MID_PIN(a1,d1);}
                if(v2){C10K_MID_PIN(a2,d2);}
                if(v3){C10K_MID_PIN(a3,d3);}
            }
            for(int off=2;off;off>>=1){
                cand_mask|=__shfl_xor_sync(0xffffffffu,cand_mask,off);
                count_current_present+=__shfl_xor_sync(0xffffffffu,count_current_present,off);
                {unsigned long long other,sum,carry,next;
                  other=__shfl_xor_sync(0xffffffffu,pc0,off);
                  sum=pc0^other;carry=pc0&other;pc0=sum;
                  other=__shfl_xor_sync(0xffffffffu,pc1,off);
                  sum=pc1^other;next=(pc1&other)|(sum&carry);pc1=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc2,off);
                  sum=pc2^other;next=(pc2&other)|(sum&carry);pc2=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc3,off);
                  sum=pc3^other;next=(pc3&other)|(sum&carry);pc3=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc4,off);
                  sum=pc4^other;next=(pc4&other)|(sum&carry);pc4=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc5,off);
                  sum=pc5^other;next=(pc5&other)|(sum&carry);pc5=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc6,off);
                  sum=pc6^other;next=(pc6&other)|(sum&carry);pc6=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,pc7,off);
                  sum=pc7^other;next=(pc7&other)|(sum&carry);pc7=sum^carry;carry=next;
                }
                {unsigned long long other,sum,carry,next;
                  other=__shfl_xor_sync(0xffffffffu,dc0,off);
                  sum=dc0^other;carry=dc0&other;dc0=sum;
                  other=__shfl_xor_sync(0xffffffffu,dc1,off);
                  sum=dc1^other;next=(dc1&other)|(sum&carry);dc1=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc2,off);
                  sum=dc2^other;next=(dc2&other)|(sum&carry);dc2=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc3,off);
                  sum=dc3^other;next=(dc3&other)|(sum&carry);dc3=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc4,off);
                  sum=dc4^other;next=(dc4&other)|(sum&carry);dc4=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc5,off);
                  sum=dc5^other;next=(dc5&other)|(sum&carry);dc5=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc6,off);
                  sum=dc6^other;next=(dc6&other)|(sum&carry);dc6=sum^carry;carry=next;
                  other=__shfl_xor_sync(0xffffffffu,dc7,off);
                  sum=dc7^other;next=(dc7&other)|(sum&carry);dc7=sum^carry;carry=next;
                }
            }
            int outkey=0;
            if(go){int degree_weight=min(255,node_degree);
    
                // This degree class cannot populate higher counter planes.
                pc8=0ULL; pc9=0ULL; pc10=0ULL; pc11=0ULL; pc12=0ULL; pc13=0ULL; pc14=0ULL; pc15=0ULL; dc8=0ULL; dc9=0ULL; dc10=0ULL; dc11=0ULL; dc12=0ULL; dc13=0ULL; dc14=0ULL; dc15=0ULL;
                int best_gain = -999999;
    int best_target = current_part;

    if (node_degree <= 65531 && num_nodes > 0) {
        unsigned long long active = cand_mask & exact_free;
        if (active) {
            unsigned long long carry = exact_bonus[current_part], tmp;
            tmp=pc2&carry; pc2^=carry; carry=tmp;
            tmp=pc3&carry; pc3^=carry; carry=tmp;
            tmp=pc4&carry; pc4^=carry; carry=tmp;
            tmp=pc5&carry; pc5^=carry; carry=tmp;
            tmp=pc6&carry; pc6^=carry; carry=tmp;
            tmp=pc7&carry; pc7^=carry; carry=tmp;
            best_gain=0;
            { unsigned long long m=active&pc7; if(m){active=m;best_gain|=128;} }
            { unsigned long long m=active&pc6; if(m){active=m;best_gain|=64;} }
            { unsigned long long m=active&pc5; if(m){active=m;best_gain|=32;} }
            { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
            { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
            { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
            { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
            { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
            best_gain-=count_current_present;
            { unsigned long long m=active&dc7; if(m)active=m; }
            { unsigned long long m=active&dc6; if(m)active=m; }
            { unsigned long long m=active&dc5; if(m)active=m; }
            { unsigned long long m=active&dc4; if(m)active=m; }
            { unsigned long long m=active&dc3; if(m)active=m; }
            { unsigned long long m=active&dc2; if(m)active=m; }
            { unsigned long long m=active&dc1; if(m)active=m; }
            { unsigned long long m=active&dc0; if(m)active=m; }
            for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
            }
            unsigned long long x1=active&0x2222222222222222ULL;
            unsigned long long x2=active&0x4444444444444444ULL;
            unsigned long long x3=active&0x8888888888888888ULL;
            unsigned long long perm=(active&0x1111111111111111ULL)|
                ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
            unsigned shift=(unsigned)node&63;
            unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
            unsigned h=(unsigned)(__ffsll(hashed)-1);
            best_target=(int)(((h-(unsigned)node)*49u)&63u);
        }
    } else {
    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = FF10K_PC_GET(target_part);
        int p_double = FF10K_DC_GET(target_part);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = FF10K_DC_GET(best_target);
            if (p_double > best_double) {
                better = true;
            } else if (p_double == best_double) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better = true;
                }
            }
        }

        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }


    }

    if (best_gain >= -1 && best_target != current_part) {
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        outkey = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
            }
            if(within==0 && in_class){move_priorities[node]=outkey;if(outkey>0)blk_max=max(blk_max,(outkey>>16)-1000);}
        }
    }

    // High-degree nodes get a whole warp; small-node threads never touch
    // their output slots. This removes long divergent pin walks without
    // adding a grid barrier or changing any sampled/full-degree semantics.
    {
        int lane=threadIdx.x&31;
        int warp=(int)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
        int warp_stride=(int)gridDim.x*(blockDim.x>>5);
        for(int node=warp;node<num_nodes;node+=warp_stride){
            int start=__ldg(&node_offsets[node]),end=__ldg(&node_offsets[node+1]);
            int node_degree=end-start;
            if(node_degree<=128)continue;
            int outkey=0;
            int current_part=__ldcg(&partition[node]);
            if((unsigned)current_part<(unsigned)num_parts && shared_nodes_in_part[current_part]>1){
                int degree_weight=min(255,node_degree);
                unsigned long long current_bit=1ULL<<current_part;
                FF10K_BS_DECL
                unsigned long long cand_mask=0ULL;int count_current_present=0;
                for(int base=start;base<end;base+=128){
                    int j0=base+lane+0;bool v0=j0<end;int h0=v0?__ldg(&node_hyperedges[j0]):0;
                    int j1=base+lane+32;bool v1=j1<end;int h1=v1?__ldg(&node_hyperedges[j1]):0;
                    int j2=base+lane+64;bool v2=j2<end;int h2=v2?__ldg(&node_hyperedges[j2]):0;
                    int j3=base+lane+96;bool v3=j3<end;int h3=v3?__ldg(&node_hyperedges[j3]):0;
                    unsigned long long a0=v0?__ldcg(&edge_flags_all[h0]):0ULL,d0=v0?__ldcg(&edge_flags_double[h0]):0ULL;
                    unsigned long long a1=v1?__ldcg(&edge_flags_all[h1]):0ULL,d1=v1?__ldcg(&edge_flags_double[h1]):0ULL;
                    unsigned long long a2=v2?__ldcg(&edge_flags_all[h2]):0ULL,d2=v2?__ldcg(&edge_flags_double[h2]):0ULL;
                    unsigned long long a3=v3?__ldcg(&edge_flags_all[h3]):0ULL,d3=v3?__ldcg(&edge_flags_double[h3]):0ULL;
                    if(v0){FF10K_PIN(a0,d0);}
                    if(v1){FF10K_PIN(a1,d1);}
                    if(v2){FF10K_PIN(a2,d2);}
                    if(v3){FF10K_PIN(a3,d3);}
                }
                for(int off=16;off;off>>=1){
                    cand_mask|=__shfl_xor_sync(0xffffffffu,cand_mask,off);
                    count_current_present+=__shfl_xor_sync(0xffffffffu,count_current_present,off);
                    { unsigned long long other,carry,sum,next;
                      other=__shfl_xor_sync(0xffffffffu,pc0,off);
                      sum=pc0^other;carry=pc0&other;pc0=sum;
                      other=__shfl_xor_sync(0xffffffffu,pc1,off);
                      sum=pc1^other;next=(pc1&other)|(sum&carry);pc1=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc2,off);
                      sum=pc2^other;next=(pc2&other)|(sum&carry);pc2=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc3,off);
                      sum=pc3^other;next=(pc3&other)|(sum&carry);pc3=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc4,off);
                      sum=pc4^other;next=(pc4&other)|(sum&carry);pc4=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc5,off);
                      sum=pc5^other;next=(pc5&other)|(sum&carry);pc5=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc6,off);
                      sum=pc6^other;next=(pc6&other)|(sum&carry);pc6=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc7,off);
                      sum=pc7^other;next=(pc7&other)|(sum&carry);pc7=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc8,off);
                      sum=pc8^other;next=(pc8&other)|(sum&carry);pc8=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc9,off);
                      sum=pc9^other;next=(pc9&other)|(sum&carry);pc9=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc10,off);
                      sum=pc10^other;next=(pc10&other)|(sum&carry);pc10=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc11,off);
                      sum=pc11^other;next=(pc11&other)|(sum&carry);pc11=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc12,off);
                      sum=pc12^other;next=(pc12&other)|(sum&carry);pc12=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc13,off);
                      sum=pc13^other;next=(pc13&other)|(sum&carry);pc13=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc14,off);
                      sum=pc14^other;next=(pc14&other)|(sum&carry);pc14=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,pc15,off);
                      sum=pc15^other;next=(pc15&other)|(sum&carry);pc15=sum^carry;carry=next;
                    }
                    { unsigned long long other,carry,sum,next;
                      other=__shfl_xor_sync(0xffffffffu,dc0,off);
                      sum=dc0^other;carry=dc0&other;dc0=sum;
                      other=__shfl_xor_sync(0xffffffffu,dc1,off);
                      sum=dc1^other;next=(dc1&other)|(sum&carry);dc1=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc2,off);
                      sum=dc2^other;next=(dc2&other)|(sum&carry);dc2=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc3,off);
                      sum=dc3^other;next=(dc3&other)|(sum&carry);dc3=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc4,off);
                      sum=dc4^other;next=(dc4&other)|(sum&carry);dc4=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc5,off);
                      sum=dc5^other;next=(dc5&other)|(sum&carry);dc5=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc6,off);
                      sum=dc6^other;next=(dc6&other)|(sum&carry);dc6=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc7,off);
                      sum=dc7^other;next=(dc7&other)|(sum&carry);dc7=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc8,off);
                      sum=dc8^other;next=(dc8&other)|(sum&carry);dc8=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc9,off);
                      sum=dc9^other;next=(dc9&other)|(sum&carry);dc9=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc10,off);
                      sum=dc10^other;next=(dc10&other)|(sum&carry);dc10=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc11,off);
                      sum=dc11^other;next=(dc11&other)|(sum&carry);dc11=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc12,off);
                      sum=dc12^other;next=(dc12&other)|(sum&carry);dc12=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc13,off);
                      sum=dc13^other;next=(dc13&other)|(sum&carry);dc13=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc14,off);
                      sum=dc14^other;next=(dc14&other)|(sum&carry);dc14=sum^carry;carry=next;
                      other=__shfl_xor_sync(0xffffffffu,dc15,off);
                      sum=dc15^other;next=(dc15&other)|(sum&carry);dc15=sum^carry;carry=next;
                    }
                }
    int best_gain = -999999;
    int best_target = current_part;

    if (node_degree <= 65531 && num_nodes > 0) {
        unsigned long long active = cand_mask & exact_free;
        if (active) {
            unsigned long long carry = exact_bonus[current_part], tmp;
            tmp=pc2&carry; pc2^=carry; carry=tmp;
            tmp=pc3&carry; pc3^=carry; carry=tmp;
            tmp=pc4&carry; pc4^=carry; carry=tmp;
            tmp=pc5&carry; pc5^=carry; carry=tmp;
            tmp=pc6&carry; pc6^=carry; carry=tmp;
            tmp=pc7&carry; pc7^=carry; carry=tmp;
            tmp=pc8&carry; pc8^=carry; carry=tmp;
            tmp=pc9&carry; pc9^=carry; carry=tmp;
            tmp=pc10&carry; pc10^=carry; carry=tmp;
            tmp=pc11&carry; pc11^=carry; carry=tmp;
            tmp=pc12&carry; pc12^=carry; carry=tmp;
            tmp=pc13&carry; pc13^=carry; carry=tmp;
            tmp=pc14&carry; pc14^=carry; carry=tmp;
            tmp=pc15&carry; pc15^=carry; carry=tmp;
            best_gain=0;
            { unsigned long long m=active&pc15; if(m){active=m;best_gain|=32768;} }
            { unsigned long long m=active&pc14; if(m){active=m;best_gain|=16384;} }
            { unsigned long long m=active&pc13; if(m){active=m;best_gain|=8192;} }
            { unsigned long long m=active&pc12; if(m){active=m;best_gain|=4096;} }
            { unsigned long long m=active&pc11; if(m){active=m;best_gain|=2048;} }
            { unsigned long long m=active&pc10; if(m){active=m;best_gain|=1024;} }
            { unsigned long long m=active&pc9; if(m){active=m;best_gain|=512;} }
            { unsigned long long m=active&pc8; if(m){active=m;best_gain|=256;} }
            { unsigned long long m=active&pc7; if(m){active=m;best_gain|=128;} }
            { unsigned long long m=active&pc6; if(m){active=m;best_gain|=64;} }
            { unsigned long long m=active&pc5; if(m){active=m;best_gain|=32;} }
            { unsigned long long m=active&pc4; if(m){active=m;best_gain|=16;} }
            { unsigned long long m=active&pc3; if(m){active=m;best_gain|=8;} }
            { unsigned long long m=active&pc2; if(m){active=m;best_gain|=4;} }
            { unsigned long long m=active&pc1; if(m){active=m;best_gain|=2;} }
            { unsigned long long m=active&pc0; if(m){active=m;best_gain|=1;} }
            best_gain-=count_current_present;
            { unsigned long long m=active&dc15; if(m)active=m; }
            { unsigned long long m=active&dc14; if(m)active=m; }
            { unsigned long long m=active&dc13; if(m)active=m; }
            { unsigned long long m=active&dc12; if(m)active=m; }
            { unsigned long long m=active&dc11; if(m)active=m; }
            { unsigned long long m=active&dc10; if(m)active=m; }
            { unsigned long long m=active&dc9; if(m)active=m; }
            { unsigned long long m=active&dc8; if(m)active=m; }
            { unsigned long long m=active&dc7; if(m)active=m; }
            { unsigned long long m=active&dc6; if(m)active=m; }
            { unsigned long long m=active&dc5; if(m)active=m; }
            { unsigned long long m=active&dc4; if(m)active=m; }
            { unsigned long long m=active&dc3; if(m)active=m; }
            { unsigned long long m=active&dc2; if(m)active=m; }
            { unsigned long long m=active&dc1; if(m)active=m; }
            { unsigned long long m=active&dc0; if(m)active=m; }
            for(int bit=31-__clz(min((unsigned)num_nodes,(unsigned)max(1,max_part_size)));bit>=0 && (active & (active-1ULL));--bit){
                unsigned long long m=active&~exact_size_planes[bit]; if(m)active=m;
            }
            unsigned long long x1=active&0x2222222222222222ULL;
            unsigned long long x2=active&0x4444444444444444ULL;
            unsigned long long x3=active&0x8888888888888888ULL;
            unsigned long long perm=(active&0x1111111111111111ULL)|
                ((x1<<16)|(x1>>48))|((x2<<32)|(x2>>32))|((x3<<48)|(x3>>16));
            unsigned shift=(unsigned)node&63;
            unsigned long long hashed=shift?((perm<<shift)|(perm>>(64-shift))):perm;
            unsigned h=(unsigned)(__ffsll(hashed)-1);
            best_target=(int)(((h-(unsigned)node)*49u)&63u);
        }
    } else {
    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);

        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = FF10K_PC_GET(target_part);
        int p_double = FF10K_DC_GET(target_part);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = FF10K_DC_GET(best_target);
            if (p_double > best_double) {
                better = true;
            } else if (p_double == best_double) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better = true;
                }
            }
        }

        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }


    }

    if (best_gain >= -1 && best_target != current_part) {
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        outkey = (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
            }
            if(lane==0){move_priorities[node]=outkey;if(outkey>0)blk_max=max(blk_max,(outkey>>16)-1000);}
        }
    }


        ++nbar;
        int maximum = ff10k_block_max(blk_max,s_warp);
        if(threadIdx.x==0)atomicMax(max_gain_buf,maximum);
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        int aspiration=max(1,(__ldcg(max_gain_buf)*3)/4);
        int local_count=0;
        for(int node=gtid;node<num_nodes;node+=stride){
            int key=__ldcg(&move_priorities[node]);
            cand[1+2*node]=node; cand[2+2*node]=-1;
            if(key>0){
                int gain=(key>>16)-1000;
                if(gain>=min_gain){
                    ++local_count;
                    cand[2+2*node]=key;
                    int target=key&63;
                    int pos=grouped_bucket_10k(bucket_counts,target);
                    if(pos<4096)bucket_keys[target*4096+pos]=
                        ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
                }
            }
        }
        int count=ff10k_block_sum(local_count,s_warp);
        if(threadIdx.x==0)atomicAdd(cand,count);
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        // Parallel exclusive prefixes replace 64 dependent scalar loads
        // in every resident block. All sums are exact bounded integers.
        int meta_n=threadIdx.x<64?__ldcg(&bucket_counts[threadIdx.x]):0;
        int meta_keep=0;
        if(threadIdx.x<64){
            int p=threadIdx.x;
            int slack=round_idx<64?slack_early:(round_idx<256?slack_mid:slack_late);
            int q=max(1,max(0,max_part_size-shared_nodes_in_part[p])+slack);
            meta_keep=p<num_parts?min(meta_n,q):0;
            exact_keeps[p]=meta_keep;
            int ns=meta_n,ks=meta_keep;
            for(int off=1;off<32;off<<=1){
                int a=__shfl_up_sync(0xffffffffu,ns,off),b=__shfl_up_sync(0xffffffffu,ks,off);
                if((p&31)>=off){ns+=a;ks+=b;}
            }
            exact_bucket_starts[p+1]=ns;exact_offsets[p+1]=ks;
        }
        if(threadIdx.x==0){exact_bucket_starts[0]=0;exact_offsets[0]=0;}
        __syncthreads();
        if(threadIdx.x>=32 && threadIdx.x<64){
            exact_bucket_starts[threadIdx.x+1]+=exact_bucket_starts[32];
            exact_offsets[threadIdx.x+1]+=exact_offsets[32];
        }
        int meta_compact=__syncthreads_or(threadIdx.x<64 && meta_n>4096);
        int meta_overflow=__syncthreads_or(threadIdx.x<64 && meta_keep>4096);
        if(threadIdx.x==0){
            exact_meta[0]=__ldcg(&cand[0]);exact_meta[1]=exact_offsets[64];
            exact_meta[2]=meta_overflow;exact_meta[3]=meta_compact;
        }
        if(!meta_compact && threadIdx.x<65)exact_bucket_starts[threadIdx.x]=threadIdx.x*4096;
        __syncthreads();
        int N = exact_meta[0], M = exact_meta[1];
        if (N == 0) { reason = 1; break; }
        int adaptive_limit = move_limit;
        // Even when the original global k_base clips N, quota selection is
        // identical if its candidate window still contains ALL N entries and
        // the union of per-target prefixes does not exceed k_base.
        bool full_quota_equivalent = N <= adaptive_limit ||
            (N <= adaptive_limit + 16384 && M <= adaptive_limit);
        if (!full_quota_equivalent || exact_meta[2] || M <= 0 || num_nodes<256) { reason = 4; break; }


        if(exact_meta[3]){
            if(gtid<64)bucket_counts[64+gtid]=0;
            ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
            for(int i=gtid;i<num_nodes;i+=stride){
                int key=__ldcg(&cand[2+2*i]);
                if(key<0)continue;
                int node=__ldcg(&cand[1+2*i]),target=key&63;
                int pos=grouped_bucket_10k(bucket_counts + 64,target);
                bucket_keys[exact_bucket_starts[target]+pos]=
                    ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
            }
            ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        }
        for(int p=blockIdx.x;p<num_parts && p<64;p+=gridDim.x){
            int n=__ldcg(&bucket_counts[p]),keep=exact_keeps[p],off=exact_offsets[p];
            int start=exact_bucket_starts[p];
            if(keep==0)continue;
            exact_sort_prefix_10k(&bucket_keys[start],n,keep,&exact_work,exact_radix_ctl);
            for(int j=threadIdx.x;j<keep;j+=blockDim.x)selected_keys[off+j]=__ldcg(&bucket_keys[start+j]);
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

        // Try a key-ordered sparse replay without materializing the
        // globally merged list. Failure leaves all solver state untouched.
        if(blockIdx.x==0) {
            int result=keyed_apply_10k(M,num_nodes,num_parts,max_part_size,
                partition,nodes_in_part,selected_keys,exact_keeps,exact_offsets,
                tabu_until,global_start + rounds_done + 1,tabu_tenure,tabu_mark_base,tabu_mark_mult,&exact_work);
            if(threadIdx.x==0)control[15]=result;
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        if(__ldcg(&control[15])<0) {
        // Replicate the small kept-prefix table into shared memory. It is
        // immutable until the merge barrier; the next phase reuses the union.
        const bool shared_merge = M <= 4096;
        if(shared_merge){
            for(int j=threadIdx.x;j<M;j+=blockDim.x)exact_work.sort[j]=__ldcg(&selected_keys[j]);
            __syncthreads();
        }
        // Merge-rank the 64 sorted prefixes. Each warp owns one output item;
        // each lane binary-searches two lists. No all-pairs candidate ranking.
        for (int i = warp_global; i < M; i += warp_stride) {
            unsigned long long key = shared_merge ? exact_work.sort[i] : __ldcg(&selected_keys[i]);
            int rank = 0;
            for (int p = lane; p < num_parts && p < 64; p += 32) {
                int lo = 0, hi = exact_keeps[p];
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    unsigned long long other = shared_merge ? exact_work.sort[exact_offsets[p]+mid] : __ldcg(&bucket_keys[exact_bucket_starts[p] + mid]);
                    if (other < key) lo = mid + 1; else hi = mid;
                }
                rank += lo;
            }
            for (int offset = 16; offset > 0; offset >>= 1) rank += __shfl_xor_sync(0xffffffffu, rank, offset);
            if (lane == 0) {
                int node = (int)(unsigned int)key;
                int target = (((unsigned int)(key >> 32)) & 63) ^ 63;
                ordered_moves[rank] = (node << 6) | target;
                int free=max(0,max_part_size-shared_nodes_in_part[target]);
                int risk_index=i-exact_offsets[target]-free;
                if(risk_index>=0 && risk_index<8) cand[1+target*8+risk_index]=rank;
            }
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        if (blockIdx.x == 0) {
            int accepted = exact_apply_critical_10k(M, num_nodes, num_parts, max_part_size,
                partition, nodes_in_part, ordered_moves, tabu_until, global_start + rounds_done + 1,
                tabu_tenure, tabu_mark_base, tabu_mark_mult, &exact_work, s_warp, cand+1, exact_keeps);
            if (threadIdx.x == 0) control[15] = accepted;
        }
        ++nbar; FF10K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        }
        int accepted = __ldcg(&control[15]);
        // A zero-execution clipped round must run the original secondary
        // tail fallback (and failed tabu marks). No partition or tabu entry
        // changed when accepted==0, so replay it unchanged on the host.
        if(accepted==0 && N>adaptive_limit){reason=4;break;}
        total_moves += accepted;
        ++rounds_done;
        if (accepted == 0) { reason = 2; break; }
        stagnant = 0;

    }
    if(blockIdx.x==0 && threadIdx.x==0){
        control[0]=reason;control[1]=rounds_done;control[2]=(int)nbar;
        control[3]=total_moves;control[4]=stagnant;
    }
}
