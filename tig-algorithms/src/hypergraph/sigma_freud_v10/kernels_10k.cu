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

extern "C" __global__ void hyperedge_clustering_sb_10k(
    const int num_hyperedges,
    const int num_clusters,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;

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

extern "C" __global__ void execute_refinement_moves_packed_10k(
    const int num_moves,
    const int num_parts,
    const int *packed,
    int *partition,
    int *nodes_in_part
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_parts) {
        nodes_in_part[i] = packed[i];
    }
    if (i < num_moves) {
        int node = packed[num_parts + i];
        int target_part = packed[num_parts + num_moves + i];
        if (node >= 0 && target_part >= 0) {
            partition[node] = target_part;
        }
    }
}

#ifndef CM_GROUPS_MAX
#define CM_GROUPS_MAX 128
#endif
#ifndef CM_HEAVY_DEG
#define CM_HEAVY_DEG 48
#endif

static __device__ __forceinline__ int select_move_target_10k(
    const unsigned int *pinfo,
    const int *shared_nodes_in_part,
    int np,
    int num_parts,
    int max_part_size,
    int current_part,
    int count_current_present,
    int node,
    int node_degree
) {
    int best_gain = -999999;
    int best_target = current_part;

    for (int target_part = 0; target_part < np; target_part++) {
        if ((pinfo[target_part] & 0xFFFFu) == 0u) continue;
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int p_count = (int)(pinfo[target_part] & 0xFFFFu);
        int p_double = (int)(pinfo[target_part] >> 16);

        int basic_gain = p_count - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
        int total_gain = basic_gain + balance_bonus;

        bool better = (total_gain > best_gain);
        if (!better && total_gain == best_gain) {
            int best_double = (int)(pinfo[best_target] >> 16);
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
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        return (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
    return 0;
}

extern "C" __global__ void compute_refinement_moves_lanes_10k(
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
    int *num_valid_moves,
    const int lanes
) {
    __shared__ int shared_nodes_in_part[64];
    __shared__ int block_moves[8];
    __shared__ unsigned int grp_info[CM_GROUPS_MAX][64];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int groups = blockDim.x / lanes;
    const int grp = threadIdx.x / lanes;
    const int sub = threadIdx.x & (lanes - 1);
    const int np = (num_parts < 64) ? num_parts : 64;
    int valid_move = 0;

    unsigned int *pinfo = grp_info[grp];
    const int grp_stride = gridDim.x * groups;
    for (int node_base = blockIdx.x * groups; node_base < num_nodes; node_base += grp_stride) {
        const int node = node_base + grp;
        int start = 0, end = 0, node_degree = 0, current_part = -1;
        bool regular = false;
        if (node < num_nodes) {
            start = __ldg(&node_offsets[node]);
            end   = __ldg(&node_offsets[node + 1]);
            node_degree = end - start;
            regular = (node_degree <= CM_HEAVY_DEG);
            if (regular) current_part = __ldg(&partition[node]);
        }
        bool eligible = regular
                     && node_degree > 0
                     && ((unsigned)current_part < (unsigned)num_parts)
                     && shared_nodes_in_part[current_part] > 1;

        for (int p = sub; p < 64; p += lanes) pinfo[p] = 0u;
        __syncwarp();

        int my_current_present = 0;
        if (eligible) {
            unsigned long long current_bit = 1ULL << current_part;
            for (int j = start + sub; j < end; j += lanes) {
                int hyperedge = __ldg(&node_hyperedges[j]);
                unsigned long long flags_all = __ldg(&edge_flags_all[hyperedge]);
                unsigned long long flags_double = __ldg(&edge_flags_double[hyperedge]);
                if (flags_double & current_bit) my_current_present++;
                unsigned long long f_all = flags_all & ~current_bit;
                unsigned long long f_dbl = flags_double & ~current_bit;
                while (f_all) {
                    int bit = __ffsll(f_all) - 1;
                    f_all &= (f_all - 1);
                    atomicAdd(&pinfo[bit], 1u);
                }
                while (f_dbl) {
                    int bit = __ffsll(f_dbl) - 1;
                    f_dbl &= (f_dbl - 1);
                    atomicAdd(&pinfo[bit], 65536u);
                }
            }
        }
        for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
            my_current_present += __shfl_xor_sync(0xffffffffu, my_current_present, offset);
        }
        __syncwarp();

        if (sub == 0 && node < num_nodes) {
            int key = 0;
            if (eligible) {
                key = select_move_target_10k(pinfo, shared_nodes_in_part, np, num_parts, max_part_size,
                                             current_part, my_current_present, node, node_degree);
                if (key != 0) valid_move += 1;
            }
            if (regular) move_priorities[node] = key;
        }
        __syncwarp();
    }
    __syncthreads();

    {
        int warps_per_block = (blockDim.x + 31) >> 5;
        int global_warp = blockIdx.x * warps_per_block + warp_id;
        int total_warps = gridDim.x * warps_per_block;
        unsigned int *pinfo = grp_info[warp_id];

        for (int hn = global_warp; hn < num_nodes; hn += total_warps) {
            int start = __ldg(&node_offsets[hn]);
            int end   = __ldg(&node_offsets[hn + 1]);
            int node_degree = end - start;
            if (node_degree <= CM_HEAVY_DEG) continue;

            int current_part = __ldg(&partition[hn]);
            bool eligible = ((unsigned)current_part < (unsigned)num_parts)
                         && shared_nodes_in_part[current_part] > 1;

            pinfo[lane] = 0u;
            pinfo[lane + 32] = 0u;
            __syncwarp();

            int my_current_present = 0;
            if (eligible) {
                unsigned long long current_bit = 1ULL << current_part;
                for (int j = start + lane; j < end; j += 32) {
                    int hyperedge = __ldg(&node_hyperedges[j]);
                    unsigned long long flags_all = __ldg(&edge_flags_all[hyperedge]);
                    unsigned long long flags_double = __ldg(&edge_flags_double[hyperedge]);
                    if (flags_double & current_bit) my_current_present++;
                    unsigned long long f_all = flags_all & ~current_bit;
                    unsigned long long f_dbl = flags_double & ~current_bit;
                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        atomicAdd(&pinfo[bit], 1u);
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        atomicAdd(&pinfo[bit], 65536u);
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_current_present += __shfl_xor_sync(0xffffffffu, my_current_present, offset);
            }
            __syncwarp();

            if (lane == 0) {
                int key = 0;
                if (eligible) {
                    key = select_move_target_10k(pinfo, shared_nodes_in_part, np, num_parts, max_part_size,
                                                 current_part, my_current_present, hn, node_degree);
                    if (key != 0) valid_move += 1;
                }
                move_priorities[hn] = key;
            }
            __syncwarp();
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        valid_move += __shfl_xor_sync(0xffffffffu, valid_move, offset);
    }
    if (lane == 0) block_moves[warp_id] = valid_move;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw && w < 8; w++) total += block_moves[w];
        num_valid_moves[blockIdx.x] = total;
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

#define RB_SENT_10k  (-2147483647)
#define RB_MAXVB_10k 1024

static __device__ __forceinline__ void rb_warp_tournament_10k(int &g, int &d, int &n, int &t) {
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int og = __shfl_down_sync(mask, g, offset);
        int od = __shfl_down_sync(mask, d, offset);
        int on = __shfl_down_sync(mask, n, offset);
        int ot = __shfl_down_sync(mask, t, offset);
        if (og > g || (og == g && od < d)) { g = og; d = od; n = on; t = ot; }
    }
}

static __device__ __forceinline__ void rb_select_10k(
    const int num_nodes, const int grid_v, const int vblock,
    const int *nb, int *s_bg, int *s_bd, int *s_bn, int *s_bt, int *s_sel
) {
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nwarps = blockDim.x >> 5;
    const int vwarps = vblock >> 5;
    for (int vb = wid; vb < grid_v; vb += nwarps) {
        int bg = RB_SENT_10k, bd = 2147483647, bn = -1, bt = -1;
        for (int vw = 0; vw < vwarps; vw++) {
            const int node = vb * vblock + vw * 32 + lane;
            int g = RB_SENT_10k, d = 2147483647, n = -1, t = -1;
            if (node < num_nodes) {
                const int tt = nb[2 * num_nodes + node];
                if (tt >= 0) { g = nb[node]; d = nb[num_nodes + node]; n = node; t = tt; }
            }
            rb_warp_tournament_10k(g, d, n, t);
            if (lane == 0 && (g > bg || (g == bg && d < bd))) { bg = g; bd = d; bn = n; bt = t; }
        }
        if (lane == 0) { s_bg[vb] = bg; s_bd[vb] = bd; s_bn[vb] = bn; s_bt[vb] = bt; }
    }
    __syncthreads();
    if (tid == 0) {
        int bg = -999999, bd = 2147483647, bn = -1, bt = -1;
        for (int vb = 0; vb < grid_v; vb++) {
            if (s_bg[vb] > bg || (s_bg[vb] == bg && s_bd[vb] < bd)) {
                bg = s_bg[vb]; bd = s_bd[vb]; bn = s_bn[vb]; bt = s_bt[vb];
            }
        }
        s_sel[0] = bn; s_sel[1] = bt;
    }
    __syncthreads();
}

extern "C" __global__ void rebalance_parts_10k(
    const int num_nodes,
    const int num_parts,
    const int min_part_size,
    const int max_part_size,
    const int vblock,
    const int *node_offsets,
    const int *node_hyperedges,
    int *partition,
    int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *gains_tab,
    int *nb,
    int *moves_out
) {
    __shared__ int s_nip[64];
    __shared__ int s_bg[RB_MAXVB_10k], s_bd[RB_MAXVB_10k], s_bn[RB_MAXVB_10k], s_bt[RB_MAXVB_10k];
    __shared__ int s_sel[2];
    const int tid = threadIdx.x;
    const int grid_v = (num_nodes + vblock - 1) / vblock;
    if (grid_v > RB_MAXVB_10k || (vblock & 31) != 0) {
        if (tid == 0) { moves_out[0] = -1; moves_out[1] = -1; }
        return;
    }
    for (int i = tid; i < 64; i += blockDim.x) s_nip[i] = (i < num_parts) ? nodes_in_part[i] : 0;
    __syncthreads();
    int moves_under = 0, moves_over = 0;

    for (int part = 0; part < num_parts; part++) {
        while (s_nip[part] < min_part_size) {
            const unsigned long long tgt_bit = 1ULL << part;
            for (int node = tid; node < num_nodes; node += blockDim.x) {
                int g = RB_SENT_10k, d = 2147483647, t = -1;
                const int src = partition[node];
                if (src != part && s_nip[src] > min_part_size) {
                    const int start = node_offsets[node];
                    const int end = node_offsets[node + 1];
                    const unsigned long long p_bit = 1ULL << src;
                    int gain = 0;
                    for (int j = start; j < end; j++) {
                        const int hedge = node_hyperedges[j];
                        const unsigned long long flags = edge_flags_all[hedge];
                        const unsigned long long dbl = edge_flags_double[hedge];
                        const int p_alone = ((dbl & p_bit) == 0) ? 1 : 0;
                        const int tgt_present = ((flags & tgt_bit) != 0) ? 1 : 0;
                        gain += p_alone - (1 - tgt_present);
                    }
                    g = gain; d = end - start; t = part;
                }
                nb[node] = g; nb[num_nodes + node] = d; nb[2 * num_nodes + node] = t;
            }
            __syncthreads();
            rb_select_10k(num_nodes, grid_v, vblock, nb, s_bg, s_bd, s_bn, s_bt, s_sel);
            if (s_sel[0] < 0) break;
            if (tid == 0) {
                const int n = s_sel[0];
                const int src = partition[n];
                partition[n] = part;
                s_nip[src]--;
                s_nip[part]++;
            }
            moves_under++;
            __syncthreads();
        }
    }

    for (int part = 0; part < num_parts; part++) {
        if (s_nip[part] <= max_part_size) continue;
        const unsigned long long p_bit = 1ULL << part;
        for (int node = tid; node < num_nodes; node += blockDim.x) {
            if (partition[node] != part) continue;
            const int start = node_offsets[node];
            const int end = node_offsets[node + 1];
            int cnt[64];
#pragma unroll
            for (int t = 0; t < 64; t++) cnt[t] = 0;
            int a = 0;
            for (int j = start; j < end; j++) {
                const int hedge = node_hyperedges[j];
                const unsigned long long flags = edge_flags_all[hedge];
                const unsigned long long dbl = edge_flags_double[hedge];
                a += ((dbl & p_bit) == 0) ? 1 : 0;
#pragma unroll
                for (int t = 0; t < 64; t++) cnt[t] += (int)((flags >> t) & 1ULL);
            }
            const int base = a - (end - start);
            int *row = gains_tab + (size_t)node * 64;
#pragma unroll
            for (int t = 0; t < 64; t++) row[t] = base + cnt[t];
        }
        __syncthreads();
        while (s_nip[part] > max_part_size) {
            for (int node = tid; node < num_nodes; node += blockDim.x) {
                int g = RB_SENT_10k, d = 2147483647, t = -1;
                if (partition[node] == part) {
                    const int *row = gains_tab + (size_t)node * 64;
                    for (int tgt = 0; tgt < num_parts; tgt++) {
                        if (tgt == part) continue;
                        if (s_nip[tgt] >= max_part_size) continue;
                        const int gain = row[tgt];
                        if (gain > g) { g = gain; t = tgt; }
                    }
                    if (t >= 0) d = node_offsets[node + 1] - node_offsets[node];
                }
                nb[node] = g; nb[num_nodes + node] = d; nb[2 * num_nodes + node] = t;
            }
            __syncthreads();
            rb_select_10k(num_nodes, grid_v, vblock, nb, s_bg, s_bd, s_bn, s_bt, s_sel);
            if (s_sel[0] < 0 || s_sel[1] < 0) break;
            if (tid == 0) {
                partition[s_sel[0]] = s_sel[1];
                s_nip[part]--;
                s_nip[s_sel[1]]++;
            }
            moves_over++;
            __syncthreads();
        }
    }

    __syncthreads();
    for (int i = tid; i < num_parts; i += blockDim.x) nodes_in_part[i] = s_nip[i];
    if (tid == 0) { moves_out[0] = moves_under; moves_out[1] = moves_over; }
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

#include <cooperative_groups.h>

static __device__ __forceinline__ bool rf_barrier_10k(
    cooperative_groups::grid_group &grid, const int soft,
    unsigned int *ctr, const unsigned int target, int *abort_flag, int *s_abort
) {
    if (!soft) { grid.sync(); return true; }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(ctr, 1u);
        unsigned int spins = 0;
        int ab = 0;
        while (true) {
            if (*((volatile unsigned int *)ctr) >= target) break;
            ab = *((volatile int *)abort_flag);
            if (ab != 0) break;
            if (++spins > (1u << 21)) { atomicExch(abort_flag, 1); ab = 1; break; }
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        __threadfence();
        *s_abort = ab;
    }
    __syncthreads();
    return *s_abort == 0;
}

struct BS6_10k { unsigned long long d0, d1, d2, d3, d4, d5; };

static __device__ __forceinline__ void bs6_add_mask_10k(BS6_10k &s, unsigned long long m) {
    unsigned long long c = m, t;
    t = s.d0 & c; s.d0 ^= c; c = t;
    t = s.d1 & c; s.d1 ^= c; c = t;
    t = s.d2 & c; s.d2 ^= c; c = t;
    t = s.d3 & c; s.d3 ^= c; c = t;
    t = s.d4 & c; s.d4 ^= c; c = t;
    s.d5 ^= c;
}

#define BS6_FA_10k(a, b, c) { unsigned long long x_ = (a) ^ (b); unsigned long long cn_ = ((a) & (b)) | ((c) & x_); (a) = x_ ^ (c); (c) = cn_; }
static __device__ __forceinline__ void bs6_add_10k(BS6_10k &a, const BS6_10k &b) {
    unsigned long long c = 0ULL;
    BS6_FA_10k(a.d0, b.d0, c)
    BS6_FA_10k(a.d1, b.d1, c)
    BS6_FA_10k(a.d2, b.d2, c)
    BS6_FA_10k(a.d3, b.d3, c)
    BS6_FA_10k(a.d4, b.d4, c)
    a.d5 = a.d5 ^ b.d5 ^ c;
}

static __device__ __forceinline__ BS6_10k bs6_shfl_xor_10k(const BS6_10k &s, int offset) {
    BS6_10k o;
    o.d0 = shfl_xor_u64_10k(0xffffffffu, s.d0, offset);
    o.d1 = shfl_xor_u64_10k(0xffffffffu, s.d1, offset);
    o.d2 = shfl_xor_u64_10k(0xffffffffu, s.d2, offset);
    o.d3 = shfl_xor_u64_10k(0xffffffffu, s.d3, offset);
    o.d4 = shfl_xor_u64_10k(0xffffffffu, s.d4, offset);
    o.d5 = shfl_xor_u64_10k(0xffffffffu, s.d5, offset);
    return o;
}

static __device__ __forceinline__ unsigned int bs6_get_10k(const BS6_10k &s, int t) {
    return (unsigned int)((s.d0 >> t) & 1ULL)
         | ((unsigned int)((s.d1 >> t) & 1ULL) << 1)
         | ((unsigned int)((s.d2 >> t) & 1ULL) << 2)
         | ((unsigned int)((s.d3 >> t) & 1ULL) << 3)
         | ((unsigned int)((s.d4 >> t) & 1ULL) << 4)
         | ((unsigned int)((s.d5 >> t) & 1ULL) << 5);
}

static __device__ __forceinline__ bool rf_sel_better_10k(
    int gain_a, int t_a, unsigned int dbl_a,
    int gain_b, int t_b, unsigned int dbl_b,
    const int *sizes, int current_part, int node
) {
    if (t_b == current_part) return t_a != current_part;
    if (t_a == current_part) return false;
    if (gain_a != gain_b) return gain_a > gain_b;
    if (dbl_a != dbl_b) return dbl_a > dbl_b;
    int sa = sizes[t_a], sb = sizes[t_b];
    if (sa != sb) return sa < sb;
    int ha = (t_a * 17 + node) & 63, hb = (t_b * 17 + node) & 63;
    if (ha != hb) return ha < hb;
    return t_a < t_b;
}

static __device__ __forceinline__ int rf_key_10k(int best_gain, int best_target, int current_part, int node, int node_degree) {
    if (best_gain >= -1 && best_target != current_part) {
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        int bg = best_gain + 1000;
        if (bg > 32767) bg = 32767;
        if (bg < 0) bg = 0;
        return (bg << 16) | (degree_weight << 8) | ((best_target & 63) | ((node & 3) << 6));
    }
    return 0;
}

static __device__ __forceinline__ int rf_select_bs_10k(
    const BS6_10k &cnt, const BS6_10k &dbl, const int *sizes,
    int num_parts, int max_part_size, int current_part, int count_current_present,
    int node, int node_degree, bool eligible, int sub, int lanes
) {
    int best_gain = -999999;
    int best_target = current_part;
    unsigned int best_dbl = 0u;
    if (eligible) {
        const int current_size = sizes[current_part];
        const unsigned long long any = cnt.d0 | cnt.d1 | cnt.d2 | cnt.d3 | cnt.d4 | cnt.d5;
        for (int t = sub; t < 64; t += lanes) {
            if (((any >> t) & 1ULL) == 0ULL) continue;
            if ((unsigned)t >= (unsigned)num_parts) continue;
            int ts = sizes[t];
            if (ts >= max_part_size) continue;
            int gain = (int)bs6_get_10k(cnt, t) - count_current_present + ((current_size > ts + 1) ? 4 : 0);
            unsigned int d = bs6_get_10k(dbl, t);
            if (rf_sel_better_10k(gain, t, d, best_gain, best_target, best_dbl, sizes, current_part, node)) {
                best_gain = gain; best_target = t; best_dbl = d;
            }
        }
    }
    for (int off = lanes >> 1; off > 0; off >>= 1) {
        int og = __shfl_xor_sync(0xffffffffu, best_gain, off);
        int ot = __shfl_xor_sync(0xffffffffu, best_target, off);
        unsigned int od = __shfl_xor_sync(0xffffffffu, best_dbl, off);
        if (rf_sel_better_10k(og, ot, od, best_gain, best_target, best_dbl, sizes, current_part, node)) {
            best_gain = og; best_target = ot; best_dbl = od;
        }
    }
    return rf_key_10k(best_gain, best_target, current_part, node, node_degree);
}

static __device__ __forceinline__ int rf_select_par_10k(
    const unsigned int *pinfo, const int *sizes,
    int num_parts, int max_part_size, int current_part, int count_current_present,
    int node, int node_degree, bool eligible, int sub, int lanes
) {
    int best_gain = -999999;
    int best_target = current_part;
    unsigned int best_dbl = 0u;
    if (eligible) {
        const int current_size = sizes[current_part];
        for (int t = sub; t < 64; t += lanes) {
            unsigned int pc = pinfo[t] & 0xFFFFu;
            if (pc == 0u) continue;
            if ((unsigned)t >= (unsigned)num_parts) continue;
            int ts = sizes[t];
            if (ts >= max_part_size) continue;
            int gain = (int)pc - count_current_present + ((current_size > ts + 1) ? 4 : 0);
            unsigned int d = pinfo[t] >> 16;
            if (rf_sel_better_10k(gain, t, d, best_gain, best_target, best_dbl, sizes, current_part, node)) {
                best_gain = gain; best_target = t; best_dbl = d;
            }
        }
    }
    for (int off = lanes >> 1; off > 0; off >>= 1) {
        int og = __shfl_xor_sync(0xffffffffu, best_gain, off);
        int ot = __shfl_xor_sync(0xffffffffu, best_target, off);
        unsigned int od = __shfl_xor_sync(0xffffffffu, best_dbl, off);
        if (rf_sel_better_10k(og, ot, od, best_gain, best_target, best_dbl, sizes, current_part, node)) {
            best_gain = og; best_target = ot; best_dbl = od;
        }
    }
    return rf_key_10k(best_gain, best_target, current_part, node, node_degree);
}

#define RF_TILE_10k 128
#define RF_WPT_10k 4

#ifndef HG_RF_SMEM_DECLARED
#define HG_RF_SMEM_DECLARED
extern __shared__ unsigned int rf_smem[];
#endif
static __device__ __forceinline__ bool rf_round_10k(
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
    unsigned long long *edge_flags_pair,
    int *move_priorities,
    const int lanes,
    const unsigned int round,
    unsigned int *tabu_until,
    int *stats,
    int *hdr,
    unsigned long long *out,
    const int soft,
    const unsigned int epoch,
    const int n_tabu_upd,
    const unsigned int *tabu_pairs,
    const int n_part_upd,
    const int *part_pairs,
    unsigned int *acc,
    unsigned int *acc_off,
    unsigned long long *out2,
    unsigned int *rs_counts,
    unsigned int *tcnt,
    unsigned int *koff,
    const int slack,
    const int adaptive_limit,
    const int extra_window,
    const int gain_only
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ int s_abort;
    unsigned int *bar = (unsigned int *)(hdr + 4);
    int *abort_flag = hdr + 28;
    const unsigned int bar_target = (epoch + 1u) * gridDim.x;
#define RF_BAR_10k(k) if (!rf_barrier_10k(grid, soft, bar + (k), bar_target, abort_flag, &s_abort)) return false;
    __shared__ int shared_nodes_in_part[64];
    unsigned int *grp_info = rf_smem;
    int *red = (int *)rf_smem;
    int *red_max = (int *)(rf_smem + 7 * 32);
    int *red_valid = (int *)(rf_smem + 7 * 32 + 256);
    unsigned int *red_cnt = rf_smem + 7 * 32 + 512;
    unsigned int *warp_low = rf_smem + 7 * 32 + 768;
    int *s_asp_p = (int *)(rf_smem + 7 * 32 + 800);

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int nthreads = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = gtid; i < n_tabu_upd; i += nthreads) {
        tabu_until[tabu_pairs[2 * i]] = tabu_pairs[2 * i + 1];
    }
    for (int i = gtid; i < n_part_upd; i += nthreads) {
        partition[part_pairs[2 * i]] = part_pairs[2 * i + 1];
    }
    RF_BAR_10k(0)

    for (int h = gtid; h < num_hyperedges; h += nthreads) {
        int start = __ldg(&hyperedge_offsets[h]);
        int end   = __ldg(&hyperedge_offsets[h + 1]);
        if (end - start <= 32) {
            unsigned long long flags_all = 0ULL;
            unsigned long long flags_double = 0ULL;
            for (int k = start; k < end; k++) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldcg(partition + node);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        flags_double |= (flags_all & bit);
                        flags_all |= bit;
                    }
                }
            }
            ((ulonglong2 *)edge_flags_pair)[h] = make_ulonglong2(flags_all, flags_double);
        }
    }
    {
        int warps_per_block = (blockDim.x + 31) >> 5;
        int global_warp = blockIdx.x * warps_per_block + warp_id;
        int total_warps = gridDim.x * warps_per_block;
        for (int hedge = global_warp; hedge < num_hyperedges; hedge += total_warps) {
            int start = 0, end = 0;
            if (lane == 0) {
                start = __ldg(&hyperedge_offsets[hedge]);
                end   = __ldg(&hyperedge_offsets[hedge + 1]);
            }
            start = __shfl_sync(0xffffffffu, start, 0);
            end   = __shfl_sync(0xffffffffu, end, 0);
            if (end - start <= 32) continue;
            unsigned long long local_all = 0ULL;
            unsigned long long local_double = 0ULL;
            for (int k = start + lane; k < end; k += 32) {
                int node = __ldg(&hyperedge_nodes[k]);
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = __ldcg(partition + node);
                    if ((unsigned)part < 64u) {
                        unsigned long long bit = 1ULL << part;
                        local_double |= (local_all & bit);
                        local_all |= bit;
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                unsigned long long other_all = shfl_xor_u64_10k(0xffffffffu, local_all, offset);
                unsigned long long other_double = shfl_xor_u64_10k(0xffffffffu, local_double, offset);
                local_double |= other_double | (local_all & other_all);
                local_all |= other_all;
            }
            if (lane == 0) {
                ((ulonglong2 *)edge_flags_pair)[hedge] = make_ulonglong2(local_all, local_double);
            }
        }
    }
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = __ldcg(nodes_in_part + threadIdx.x);
    }
    RF_BAR_10k(1)
    {
        const int groups = blockDim.x / lanes;
        const int grp = threadIdx.x / lanes;
        const int sub = threadIdx.x & (lanes - 1);
        const int grp_stride = gridDim.x * groups;
        for (int node_base = blockIdx.x * groups; node_base < num_nodes; node_base += grp_stride) {
            const int node = node_base + grp;
            int start = 0, end = 0, node_degree = 0, current_part = -1;
            bool regular = false;
            if (node < num_nodes) {
                start = __ldg(&node_offsets[node]);
                end   = __ldg(&node_offsets[node + 1]);
                node_degree = end - start;
                regular = (node_degree <= CM_HEAVY_DEG);
                if (regular) current_part = __ldcg(partition + node);
            }
            bool eligible = regular
                         && node_degree > 0
                         && ((unsigned)current_part < (unsigned)num_parts)
                         && shared_nodes_in_part[current_part] > 1;

            BS6_10k cnt = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            BS6_10k dbl = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            int my_present = 0;
            if (eligible) {
                unsigned long long current_bit = 1ULL << current_part;
                for (int j = start + sub; j < end; j += lanes) {
                    int hyperedge = __ldg(&node_hyperedges[j]);
                    ulonglong2 fp = __ldcg((const ulonglong2 *)edge_flags_pair + hyperedge);
                    if (fp.y & current_bit) my_present++;
                    bs6_add_mask_10k(cnt, fp.x & ~current_bit);
                    bs6_add_mask_10k(dbl, fp.y & ~current_bit);
                }
            }
            for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
                my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
                BS6_10k oc = bs6_shfl_xor_10k(cnt, offset);
                bs6_add_10k(cnt, oc);
                BS6_10k od = bs6_shfl_xor_10k(dbl, offset);
                bs6_add_10k(dbl, od);
            }
            {
                int key = rf_select_bs_10k(cnt, dbl, shared_nodes_in_part, num_parts, max_part_size,
                                           current_part, my_present, node, node_degree, eligible, sub, lanes);
                if (sub == 0 && node < num_nodes && regular) move_priorities[node] = key;
            }
        }
        __syncthreads();

        int warps_per_block = (blockDim.x + 31) >> 5;
        int global_warp = blockIdx.x * warps_per_block + warp_id;
        int total_warps = gridDim.x * warps_per_block;
        unsigned int *wpinfo = grp_info + warp_id * 64;
        for (int hn = global_warp; hn < num_nodes; hn += total_warps) {
            int start = __ldg(&node_offsets[hn]);
            int end   = __ldg(&node_offsets[hn + 1]);
            int node_degree = end - start;
            if (node_degree <= CM_HEAVY_DEG) continue;

            int current_part = __ldcg(partition + hn);
            bool eligible = ((unsigned)current_part < (unsigned)num_parts)
                         && shared_nodes_in_part[current_part] > 1;

            wpinfo[lane] = 0u;
            wpinfo[lane + 32] = 0u;
            __syncwarp();

            int my_present = 0;
            if (eligible) {
                unsigned long long current_bit = 1ULL << current_part;
                for (int j = start + lane; j < end; j += 32) {
                    int hyperedge = __ldg(&node_hyperedges[j]);
                    ulonglong2 fp = __ldcg((const ulonglong2 *)edge_flags_pair + hyperedge);
                    if (fp.y & current_bit) my_present++;
                    unsigned long long f_all = fp.x & ~current_bit;
                    unsigned long long f_dbl = fp.y & ~current_bit;
                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        atomicAdd(&wpinfo[bit], 1u);
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        atomicAdd(&wpinfo[bit], 65536u);
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
            }
            __syncwarp();
            {
                int key = rf_select_par_10k(wpinfo, shared_nodes_in_part, num_parts, max_part_size,
                                            current_part, my_present, hn, node_degree, eligible, lane, 32);
                if (lane == 0) move_priorities[hn] = key;
            }
            __syncwarp();
        }
    }
    RF_BAR_10k(2)
    {
        const int nt_max = (num_nodes + 255) >> 8;
        const int rs_total = 4 * 256 * nt_max;
        for (int i = gtid; i < rs_total; i += nthreads) rs_counts[i] = 0u;
        for (int i = gtid; i < 64 * nt_max; i += nthreads) tcnt[i] = 0u;
    }
    const int num_tiles = (num_nodes + RF_TILE_10k - 1) / RF_TILE_10k;
    const int subs = blockDim.x / RF_TILE_10k;
    const int tsub = threadIdx.x / RF_TILE_10k;
    const int tth = threadIdx.x & (RF_TILE_10k - 1);
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const int num_groups = (num_tiles + subs - 1) / subs;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_10k + tth;
        int mx = (-2147483647 - 1);
        int v_valid = 0;
        if (t < num_tiles && node < num_nodes) {
            int key = __ldcg(move_priorities + node);
            if (key != 0) {
                mx = (key >> 16) - 1000;
                v_valid = 1;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            v_valid += __shfl_xor_sync(0xffffffffu, v_valid, off);
        }
        if (lane == 0) {
            red[0 * 32 + warp_id] = mx; red[1 * 32 + warp_id] = v_valid;
        }
        __syncthreads();
        if (tth == 0 && t < num_tiles) {
            int bmx = (-2147483647 - 1), s1 = 0;
            for (int w = tsub * RF_WPT_10k; w < (tsub + 1) * RF_WPT_10k; w++) {
                bmx = max(bmx, red[0 * 32 + w]); s1 += red[1 * 32 + w];
            }
            int *o = stats + t * 8;
            o[0] = bmx; o[1] = s1; o[2] = 0; o[3] = 0; o[4] = 0; o[5] = 0; o[6] = 0; o[7] = 0;
        }
        __syncthreads();
    }
    RF_BAR_10k(3)
    if (blockIdx.x == 0) {
        const int tid = threadIdx.x;
        const int nt = blockDim.x;
        int mx = (-2147483647 - 1);
        int valid = 0;
        for (int b = tid; b < num_tiles; b += nt) {
            mx = max(mx, __ldcg(stats + b * 8));
            valid += __ldcg(stats + b * 8 + 1);
        }
        red_max[tid] = mx;
        red_valid[tid] = valid;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                red_max[tid] = max(red_max[tid], red_max[tid + s]);
                red_valid[tid] += red_valid[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int m = red_max[0];
            int asp = 0;
            if (red_valid[0] > 0) {
                asp = (m * 3) / 4;
                if (asp < 1) asp = 1;
            }
            *s_asp_p = asp;
            hdr[0] = red_valid[0];
            hdr[1] = asp;
            hdr[2] = m;
            hdr[3] = 0;
            hdr[29] = 0;
        }
    }
    RF_BAR_10k(4)
    const int aspiration_h = __ldcg(hdr + 1);
    unsigned int *warp_acc = warp_low + 16;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_10k + tth;
        int key = 0;
        bool accept = false;
        if (t < num_tiles && node < num_nodes) {
            key = __ldcg(move_priorities + node);
            if (key != 0) {
                int gain = (key >> 16) - 1000;
                if (gain_only) {
                    accept = (gain >= 0);
                } else {
                    bool is_tabu = (__ldcg(tabu_until + node) > round) && (gain < aspiration_h);
                    accept = !is_tabu;
                }
            }
        }
        unsigned acc_mask = __ballot_sync(0xffffffffu, accept);
        if (t < num_tiles && node < num_nodes) acc[node] = accept ? ~((unsigned int)key) : 0u;
        if (lane == 0) warp_acc[warp_id] = __popc(acc_mask);
        __syncthreads();
        if (tth == 0 && t < num_tiles) {
            unsigned c = 0;
            for (int w = tsub * RF_WPT_10k; w < (tsub + 1) * RF_WPT_10k; w++) c += warp_acc[w];
            stats[t * 8 + 7] = (int)c;
        }
        __syncthreads();
    }
    RF_BAR_10k(5)
    if (blockIdx.x == 0) {
        const int tid = threadIdx.x;
        const int nt_ = blockDim.x;
        const int per = (num_tiles + nt_ - 1) / nt_;
        const int b0 = tid * per;
        const int b1 = min(num_tiles, b0 + per);
        unsigned int local = 0;
        for (int b = b0; b < b1; b++) local += (unsigned int)__ldcg(stats + b * 8 + 7);
        red_cnt[tid] = local;
        __syncthreads();
        if (tid == 0) {
            unsigned int a = 0;
            for (int i = 0; i < nt_; i++) {
                unsigned int v = red_cnt[i];
                red_cnt[i] = a;
                a += v;
            }
            hdr[3] = (int)a;
        }
        __syncthreads();
        unsigned int a = red_cnt[tid];
        for (int b = b0; b < b1; b++) {
            acc_off[b] = a;
            a += (unsigned int)__ldcg(stats + b * 8 + 7);
        }
    }
    RF_BAR_10k(6)

    const unsigned int cnt_c = (unsigned int)__ldcg(hdr + 3);
    const int nt_c = (int)((cnt_c + 255u) >> 8);
    const int npass = 4;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_10k + tth;
        unsigned int v = 0u;
        if (t < num_tiles && node < num_nodes) v = __ldcg(acc + node);
        unsigned m = __ballot_sync(0xffffffffu, v != 0u);
        unsigned r = __popc(m & lanemask_lt);
        if (lane == 0) warp_low[warp_id] = __popc(m);
        __syncthreads();
        unsigned pre = 0;
        for (int w = tsub * RF_WPT_10k; w < warp_id; w++) pre += warp_low[w];
        if (v != 0u) {
            unsigned pos = __ldcg(acc_off + t) + pre + r;
            out[pos] = ((unsigned long long)v << 32) | (unsigned long long)(unsigned int)node;
            atomicAdd(&rs_counts[(pos >> 8) * 256u + (v & 0xFFu)], 1u);
        }
        __syncthreads();
    }

    unsigned int *wc = rf_smem;
    for (int k = 0; k < 4; k++) {
        RF_BAR_10k(7 + 2 * k)
        unsigned int *ck = rs_counts + k * 256 * nt_c;
        if (blockIdx.x == 0) {
            const int b = threadIdx.x;
            unsigned int tot = 0;
            #pragma unroll 8
            for (int tile = 0; tile < nt_c; tile++) tot += __ldcg(ck + tile * 256 + b);
            red_cnt[b] = tot;
            __syncthreads();
            if (b == 0) {
                unsigned int a = 0;
                for (int i = 0; i < 256; i++) {
                    unsigned int v = red_cnt[i];
                    red_cnt[i] = a;
                    a += v;
                }
            }
            __syncthreads();
            unsigned int run = red_cnt[b];
            #pragma unroll 8
            for (int tile = 0; tile < nt_c; tile++) {
                unsigned int c = __ldcg(ck + tile * 256 + b);
                ck[tile * 256 + b] = run;
                run += c;
            }
        }
        RF_BAR_10k(8 + 2 * k)
        const unsigned long long *src = (k & 1) ? out2 : out;
        unsigned long long *dst = (k & 1) ? out : out2;
        unsigned int *cn = rs_counts + (k + 1) * 256 * nt_c;
        const int sh = 32 + 8 * k;
        for (int tile = blockIdx.x; tile < nt_c; tile += gridDim.x) {
            const unsigned int i = (unsigned int)tile * 256u + threadIdx.x;
            const bool has = i < cnt_c;
            unsigned long long v = 0ULL;
            unsigned int b = 0xFFFFFFFFu;
            if (has) {
                v = __ldcg(src + i);
                b = (unsigned int)((v >> sh) & 0xFFULL);
            }
            for (int q = threadIdx.x; q < 8 * 256; q += blockDim.x) wc[q] = 0u;
            __syncthreads();
            unsigned peers = __match_any_sync(0xffffffffu, b);
            unsigned lr = __popc(peers & lanemask_lt);
            if (has && lr == 0) wc[warp_id * 256 + b] = __popc(peers);
            __syncthreads();
            {
                unsigned int run = 0;
                for (int w = 0; w < 8; w++) {
                    unsigned int c = wc[w * 256 + threadIdx.x];
                    wc[w * 256 + threadIdx.x] = run;
                    run += c;
                }
            }
            __syncthreads();
            if (has) {
                unsigned int dest = __ldcg(ck + tile * 256 + b) + wc[warp_id * 256 + b] + lr;
                dst[dest] = v;
                if (k + 1 < npass) {
                    unsigned int bn = (unsigned int)((v >> (sh + 8)) & 0xFFULL);
                    atomicAdd(&cn[(dest >> 8) * 256u + bn], 1u);
                } else {
                    unsigned int tg = (~(unsigned int)(v >> 32)) & 63u;
                    atomicAdd(&tcnt[(dest >> 8) * 64u + tg], 1u);
                }
            }
            __syncthreads();
        }
    }

    const unsigned long long *sorted_l = out;
    unsigned long long *kept_l = out2;
    const unsigned int k_base_d = min(cnt_c, (unsigned int)adaptive_limit);
    const unsigned int k_cand_d = min(cnt_c, k_base_d + (unsigned int)extra_window);
    RF_BAR_10k(15)
    if (blockIdx.x == 0 && threadIdx.x < 64) {
        const int t = threadIdx.x;
        unsigned int run = 0;
        for (int tile = 0; tile < nt_c; tile++) {
            unsigned int c = __ldcg(tcnt + tile * 64 + t);
            tcnt[tile * 64 + t] = run;
            run += c;
        }
    }
    RF_BAR_10k(16)
    for (int tile = blockIdx.x; tile < nt_c; tile += gridDim.x) {
        const unsigned int p = (unsigned int)tile * 256u + threadIdx.x;
        const bool has = p < cnt_c;
        unsigned long long v = 0ULL;
        unsigned int tg = 0xFFFFFFFFu;
        if (has) {
            v = __ldcg(sorted_l + p);
            tg = (~(unsigned int)(v >> 32)) & 63u;
        }
        for (int q = threadIdx.x; q < 8 * 64; q += blockDim.x) wc[q] = 0u;
        __syncthreads();
        unsigned peers = __match_any_sync(0xffffffffu, tg);
        unsigned lr = __popc(peers & lanemask_lt);
        if (has && lr == 0) wc[warp_id * 64 + tg] = __popc(peers);
        __syncthreads();
        if (threadIdx.x < 64) {
            unsigned int run = 0;
            for (int w = 0; w < 8; w++) {
                unsigned int c = wc[w * 64 + threadIdx.x];
                wc[w * 64 + threadIdx.x] = run;
                run += c;
            }
        }
        __syncthreads();
        bool keep = false;
        if (has && p < k_cand_d) {
            unsigned int rank = __ldcg(tcnt + tile * 64 + tg) + wc[warp_id * 64 + tg] + lr;
            int freep = max_part_size - __ldcg(nodes_in_part + tg);
            if (freep < 0) freep = 0;
            unsigned int quota = (unsigned int)max(1, freep + slack);
            keep = rank < quota;
        }
        if (has) acc[p] = keep ? 1u : 0u;
        unsigned km = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) warp_low[warp_id] = __popc(km);
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int c = 0;
            for (int w = 0; w < 8; w++) c += warp_low[w];
            koff[tile] = c;
        }
        __syncthreads();
    }
    RF_BAR_10k(17)
    if (blockIdx.x == 0) {
        const int tid = threadIdx.x;
        const int nt_ = blockDim.x;
        const int per = (nt_c + nt_ - 1) / nt_;
        const int b0 = tid * per;
        const int b1 = min(nt_c, b0 + per);
        unsigned int local = 0;
        for (int b = b0; b < b1; b++) local += __ldcg(koff + b);
        red_cnt[tid] = local;
        __syncthreads();
        if (tid == 0) {
            unsigned int a = 0;
            for (int i = 0; i < nt_; i++) {
                unsigned int v = red_cnt[i];
                red_cnt[i] = a;
                a += v;
            }
            hdr[30] = (int)a;
        }
        __syncthreads();
        unsigned int a = red_cnt[tid];
        for (int b = b0; b < b1; b++) {
            unsigned int c = __ldcg(koff + b);
            koff[b] = a;
            a += c;
        }
    }
    RF_BAR_10k(18)
    for (int tile = blockIdx.x; tile < nt_c; tile += gridDim.x) {
        const unsigned int p = (unsigned int)tile * 256u + threadIdx.x;
        const bool has = p < cnt_c;
        bool keep = has && (__ldcg(acc + p) != 0u);
        unsigned km = __ballot_sync(0xffffffffu, keep);
        unsigned r = __popc(km & lanemask_lt);
        if (lane == 0) warp_low[warp_id] = __popc(km);
        __syncthreads();
        unsigned pre = 0;
        for (int w = 0; w < warp_id; w++) pre += warp_low[w];
        if (keep) kept_l[__ldcg(koff + tile) + pre + r] = __ldcg(sorted_l + p);
        __syncthreads();
    }
    return true;
}

extern "C" __global__ void __launch_bounds__(256, 3) round_fused_10k(
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
    unsigned long long *edge_flags_pair,
    int *move_priorities,
    const int lanes,
    const unsigned int round,
    unsigned int *tabu_until,
    int *stats,
    int *hdr,
    unsigned long long *out,
    const int soft,
    const unsigned int launch_idx,
    const int n_tabu_upd,
    const unsigned int *tabu_pairs,
    const int n_part_upd,
    const int *part_pairs,
    unsigned int *acc,
    unsigned int *acc_off,
    unsigned long long *out2,
    unsigned int *rs_counts,
    unsigned int *tcnt,
    unsigned int *koff,
    const int slack,
    const int adaptive_limit,
    const int extra_window
) {
    rf_round_10k(num_hyperedges, num_nodes, num_parts, max_part_size, hyperedge_nodes, hyperedge_offsets,
                 node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_pair, move_priorities,
                 lanes, round, tabu_until, stats, hdr, out, soft, launch_idx, n_tabu_upd, tabu_pairs,
                 n_part_upd, part_pairs, acc, acc_off, out2, rs_counts, tcnt, koff, slack, adaptive_limit,
                 extra_window, 0);
}

extern "C" __global__ void __launch_bounds__(256, 3) rounds_fused_10k(
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
    unsigned long long *edge_flags_pair,
    int *move_priorities,
    const int lanes,
    const int round0,
    const unsigned int global_round0,
    const int max_rounds,
    const int move_limit,
    const int tabu_tenure,
    const int extra_window,
    unsigned int *tabu_until,
    int *stats,
    int *hdr,
    unsigned long long *out,
    const int soft,
    const unsigned int epoch0,
    const int n_tabu_upd,
    const unsigned int *tabu_pairs,
    const int n_part_upd,
    const int *part_pairs,
    unsigned int *acc,
    unsigned int *acc_off,
    unsigned long long *out2,
    unsigned int *rs_counts,
    unsigned int *tcnt,
    unsigned int *koff,
    const int gain_only
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ int s_abort2;
    __shared__ int s_nip[64];
    __shared__ int s_node[32];
    __shared__ int s_tgt[32];
    __shared__ int s_cur[32];
    __shared__ unsigned int s_mask;
    unsigned int *bar = (unsigned int *)(hdr + 4);
    int *abort_flag = hdr + 28;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        hdr[25] = 0; hdr[26] = 0; hdr[27] = 0; hdr[31] = 0;
    }
    for (int r = 0; r < max_rounds; r++) {
        const int round = round0 + r;
        const unsigned int ground = global_round0 + (unsigned int)r;
        const unsigned int epoch = epoch0 + (unsigned int)r;
        const int adaptive_limit = (round < 50) ? (move_limit / 2) : (round < 200) ? move_limit : (move_limit / 3);
        const int slack = (round < 64) ? 8 : (round < 256) ? 4 : 2;
        const int ntu = (r == 0) ? n_tabu_upd : 0;
        const int npu = (r == 0) ? n_part_upd : 0;
        if (!rf_round_10k(num_hyperedges, num_nodes, num_parts, max_part_size, hyperedge_nodes, hyperedge_offsets,
                          node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_pair, move_priorities,
                          lanes, ground, tabu_until, stats, hdr, out, soft, epoch, ntu, tabu_pairs,
                          npu, part_pairs, acc, acc_off, out2, rs_counts, tcnt, koff, slack, adaptive_limit,
                          extra_window, gain_only)) {
            return;
        }
        if (!rf_barrier_10k(grid, soft, bar + 19, (epoch + 1u) * gridDim.x, abort_flag, &s_abort2)) return;

        if (blockIdx.x == 0 && warp_id == 0) {
            const unsigned int cnt = (unsigned int)__ldcg(hdr + 3);
            const unsigned int kept = min((unsigned int)__ldcg(hdr + 30), cnt);
            const unsigned int k_base = min(cnt, (unsigned int)adaptive_limit);
            const unsigned int take = min(kept, k_base);
            for (int t = lane; t < 64; t += 32) {
                s_nip[t] = (t < num_parts) ? __ldcg(nodes_in_part + t) : 0;
            }
            __syncwarp();
            unsigned int executed = 0;
            for (unsigned int base = 0; base < take; base += 32) {
                const unsigned int i = base + lane;
                const bool has = i < take;
                int node = -1, tgt = -1, cur = -1;
                if (has) {
                    unsigned long long v = __ldcg(out2 + i);
                    node = (int)(unsigned int)(v & 0xFFFFFFFFULL);
                    tgt = (int)((~(unsigned int)(v >> 32)) & 63u);
                    if ((unsigned)node < (unsigned)num_nodes) cur = __ldcg(partition + node);
                }
                s_node[lane] = node; s_tgt[lane] = tgt; s_cur[lane] = cur;
                __syncwarp();
                if (lane == 0) {
                    unsigned int m = 0u;
                    const unsigned int n = min(32u, take - base);
                    for (unsigned int j = 0; j < n; j++) {
                        const int nj = s_node[j], tj = s_tgt[j], cj = s_cur[j];
                        if ((unsigned)nj < (unsigned)num_nodes && (unsigned)tj < (unsigned)num_parts
                            && (unsigned)cj < (unsigned)num_parts
                            && s_nip[tj] < max_part_size && s_nip[cj] > 1) {
                            s_nip[cj] -= 1;
                            s_nip[tj] += 1;
                            m |= (1u << j);
                            executed++;
                        }
                    }
                    s_mask = m;
                }
                __syncwarp();
                if (has && ((s_mask >> lane) & 1u)) partition[node] = tgt;
                __syncwarp();
            }
            executed = __shfl_sync(0xffffffffu, executed, 0);
            for (unsigned int i = lane; i < executed; i += 32) {
                unsigned long long v = __ldcg(out2 + i);
                int node = (int)(unsigned int)(v & 0xFFFFFFFFULL);
                if ((unsigned)node < (unsigned)num_nodes) tabu_until[node] = ground + (unsigned int)tabu_tenure;
            }
            for (int t = lane; t < num_parts && t < 64; t += 32) nodes_in_part[t] = s_nip[t];
            __threadfence();
            if (lane == 0) {
                hdr[26] += (int)cnt;
                hdr[27] += (int)take;
                hdr[31] = r + 1;
                hdr[25] = (cnt == 0u) ? 1 : (executed == 0u) ? 2 : 0;
                __threadfence();
            }
        }
        if (!rf_barrier_10k(grid, soft, bar + 20, (epoch + 1u) * gridDim.x, abort_flag, &s_abort2)) return;
        if (__ldcg(hdr + 25) != 0) return;
    }
}
