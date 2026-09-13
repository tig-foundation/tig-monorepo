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

    unsigned int part_info[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) part_info[p] = 0;

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

    int top_gain[4];
    int top_double[4];
    int top_part[4];
    for (int k = 0; k < 4; k++) {
        top_gain[k] = min_gain - 1;
        top_double[k] = -1;
        top_part[k] = -1;
    }

    unsigned long long tmp = cand_mask;
    while (tmp) {
        int target_part = __ffsll(tmp) - 1;
        tmp &= (tmp - 1);
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        
        int p_count = part_info[target_part] & 0xFFFF;
        int p_double = part_info[target_part] >> 16;
        
        int basic_gain = p_count - count_current_present;
        if (used_degree < node_degree) {
            basic_gain = (basic_gain * node_degree) / used_degree;
        }
        if (basic_gain < min_gain) continue;

        int hash_tgt = (target_part * 17 + node) & 63;
        int hash_top0 = (top_part[0] >= 0) ? ((top_part[0] * 17 + node) & 63) : 999;
        int hash_top1 = (top_part[1] >= 0) ? ((top_part[1] * 17 + node) & 63) : 999;
        int hash_top2 = (top_part[2] >= 0) ? ((top_part[2] * 17 + node) & 63) : 999;
        int hash_top3 = (top_part[3] >= 0) ? ((top_part[3] * 17 + node) & 63) : 999;

        bool better0 = (basic_gain > top_gain[0]) || 
                       (basic_gain == top_gain[0] && p_double > top_double[0]) ||
                       (basic_gain == top_gain[0] && p_double == top_double[0] && hash_tgt < hash_top0);
                       
        bool better1 = (basic_gain > top_gain[1]) || 
                       (basic_gain == top_gain[1] && p_double > top_double[1]) ||
                       (basic_gain == top_gain[1] && p_double == top_double[1] && hash_tgt < hash_top1);

        bool better2 = (basic_gain > top_gain[2]) || 
                       (basic_gain == top_gain[2] && p_double > top_double[2]) ||
                       (basic_gain == top_gain[2] && p_double == top_double[2] && hash_tgt < hash_top2);

        bool better3 = (basic_gain > top_gain[3]) || 
                       (basic_gain == top_gain[3] && p_double > top_double[3]) ||
                       (basic_gain == top_gain[3] && p_double == top_double[3] && hash_tgt < hash_top3);

        if (better0) {
            top_gain[3] = top_gain[2]; top_double[3] = top_double[2]; top_part[3] = top_part[2];
            top_gain[2] = top_gain[1]; top_double[2] = top_double[1]; top_part[2] = top_part[1];
            top_gain[1] = top_gain[0]; top_double[1] = top_double[0]; top_part[1] = top_part[0];
            top_gain[0] = basic_gain; top_double[0] = p_double; top_part[0] = target_part;
        } else if (better1) {
            top_gain[3] = top_gain[2]; top_double[3] = top_double[2]; top_part[3] = top_part[2];
            top_gain[2] = top_gain[1]; top_double[2] = top_double[1]; top_part[2] = top_part[1];
            top_gain[1] = basic_gain; top_double[1] = p_double; top_part[1] = target_part;
        } else if (better2) {
            top_gain[3] = top_gain[2]; top_double[3] = top_double[2]; top_part[3] = top_part[2];
            top_gain[2] = basic_gain; top_double[2] = p_double; top_part[2] = target_part;
        } else if (better3) {
            top_gain[3] = basic_gain; top_double[3] = p_double; top_part[3] = target_part;
        }
    }

    for (int k = 0; k < 4; k++) {
        if (top_part[k] >= 0 && top_gain[k] >= min_gain) {
            int g = top_gain[k];
            if (g > 32767) g = 32767;
            if (g < -32768) g = -32768;
            short g16 = (short)g;
            unsigned short t16 = (unsigned short)(top_part[k] & 0xFFFF);
            swap_gains[node * 4 + k] = ((int)(unsigned short)g16 << 16) | (int)t16;
        }
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

    for (int n = idx; n < num_nodes; n += stride) {
        int part = __ldg(&partition[n]);
        if (part != source_part) continue;
        int start = __ldg(&node_offsets[n]);
        int end   = __ldg(&node_offsets[n + 1]);
        int deg = end - start;

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
extern "C" __global__ void compute_swap_topk_10k(
    const int num_nodes,
    const int num_parts,
    const int *partition,
    const int *swap_gains,
    int *swap_topk
) {
    int np = (num_parts < 64) ? num_parts : 64;
    int src = blockIdx.x / np;
    int tgt = blockIdx.x % np;
    if (src == tgt || src >= np || tgt >= np) return;

    __shared__ int sh_nodes[SWAP_TOPK_THREADS][32];
    __shared__ int sh_gains[SWAP_TOPK_THREADS][32];
    __shared__ int sh_counts[SWAP_TOPK_THREADS];

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    int local_nodes[32];
    int local_gains[32];
    int local_count = 0;

    for (int n = tid; n < num_nodes; n += nthreads) {
        int part = __ldg(&partition[n]);
        if (part == src) {
            int base = n * 4;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                int val = __ldg(&swap_gains[base + k]);
                if (val == 0) continue;
                int t = val & 0xFFFF;
                int gain = ((val >> 16) & 0xFFFF);
                gain = (gain >= 0x8000) ? (int)(gain - 0x10000) : (int)gain;
                if (t != tgt) continue;
                int pos = local_count;
                while (pos > 0 &&
                       (local_gains[pos - 1] < gain ||
                       (local_gains[pos - 1] == gain && local_nodes[pos - 1] > n))) {
                    if (pos < 32) {
                        local_nodes[pos] = local_nodes[pos - 1];
                        local_gains[pos] = local_gains[pos - 1];
                    }
                    pos--;
                }
                if (pos < 32) {
                    local_nodes[pos] = n;
                    local_gains[pos] = gain;
                    if (local_count < 32) local_count++;
                }
            }
        }
    }

    sh_counts[tid] = local_count;
    for (int i = 0; i < local_count; i++) {
        sh_nodes[tid][i] = local_nodes[i];
        sh_gains[tid][i] = local_gains[i];
    }
    __syncthreads();

    if (tid == 0) {
        int top_nodes[32];
        int top_gains[32];
        int top_count = 0;
        for (int t = 0; t < nthreads; t++) {
            int cnt = sh_counts[t];
            for (int i = 0; i < cnt; i++) {
                int node = sh_nodes[t][i];
                int gain = sh_gains[t][i];
                int pos = top_count;
                while (pos > 0 &&
                       (top_gains[pos - 1] < gain ||
                       (top_gains[pos - 1] == gain && top_nodes[pos - 1] > node))) {
                    if (pos < 32) {
                        top_nodes[pos] = top_nodes[pos - 1];
                        top_gains[pos] = top_gains[pos - 1];
                    }
                    pos--;
                }
                if (pos < 32) {
                    top_nodes[pos] = node;
                    top_gains[pos] = gain;
                    if (top_count < 32) top_count++;
                }
            }
        }
        int base_out = blockIdx.x * 32 * 2;
        for (int i = 0; i < 32; i++) {
            if (i < top_count) {
                swap_topk[base_out + i * 2] = top_nodes[i];
                swap_topk[base_out + i * 2 + 1] = top_gains[i];
            } else {
                swap_topk[base_out + i * 2] = -1;
                swap_topk[base_out + i * 2 + 1] = 0;
            }
        }
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
