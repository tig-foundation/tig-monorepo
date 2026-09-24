#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void choose_elite_per_hyperedge_200k(
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

extern "C" __global__ void assign_from_elite_votes_200k(
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
    int used_degree = (deg > 256) ? 256 : deg;

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

extern "C" __global__ void hyperedge_clustering_200k(
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
        int bucket = (hedge_size > 8) ? 3 : (hedge_size > 4) ? 2 : (hedge_size > 2) ? 1 : 0;
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

extern "C" __global__ void compute_node_preferences_200k(
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
                int weight = (hedge_size <= 2) ? 6 : (hedge_size <= 4) ? 4 : (hedge_size <= 8) ? 2 : 1;
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

extern "C" __global__ void execute_node_assignments_200k(
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


static __device__ __forceinline__ unsigned long long shfl_xor_u64_200k(
    unsigned mask,
    unsigned long long v,
    int lane_mask
) {
    unsigned lo = __shfl_xor_sync(mask, (unsigned)(v & 0xFFFFFFFFu), lane_mask);
    unsigned hi = __shfl_xor_sync(mask, (unsigned)(v >> 32), lane_mask);
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}


extern "C" __global__ void precompute_edge_flags_200k(
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
            unsigned long long other_all = shfl_xor_u64_200k(active, local_all, offset);
            unsigned long long other_double = shfl_xor_u64_200k(active, local_double, offset);
            local_double |= other_double | (local_all & other_all);
            local_all |= other_all;
        }

        if (lane == 0) {
            edge_flags_all[hedge] = local_all;
            edge_flags_double[hedge] = local_double;
        }
    }
}

extern "C" __global__ void compute_refinement_moves_optimized_200k(
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
    const int gain_floor
) {
    __shared__ int shared_nodes_in_part[64];
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    move_priorities[node] = 0x80000000;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;
    if (shared_nodes_in_part[current_part] <= 1) return;

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
    int crit = 0;

    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = node_hyperedges[start + rel];

        unsigned long long flags_all = edge_flags_all[hyperedge];
        unsigned long long flags_double = edge_flags_double[hyperedge];

        if ((flags_all & current_bit) != 0ULL && (flags_double & current_bit) == 0ULL) {
            int parts = __popcll(flags_all);
            if (parts > 1) {
                crit += (parts - 1);
                if (crit > 255) crit = 255;
            }
        }

        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

        if (mask & current_bit) count_current_present++;

        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            part_counts[bit]++;
            cand_mask |= 1ULL << bit;
        }
    }

    int degree_weight = node_degree > 255 ? 255 : node_degree;
    int rank_byte = degree_weight + crit;
    if (rank_byte > 255) rank_byte = 255;

    int best_gain = -999999;
    int best_target = current_part;

    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int basic_gain = (int)part_counts[target_part] - count_current_present;

        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 2) ? 1 : 0;

        int total_gain = basic_gain + balance_bonus;

        if (total_gain > best_gain || (total_gain == best_gain && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }

    if (best_target != current_part && best_gain >= gain_floor) {
        int bg = best_gain > 32767 ? 32767 : best_gain;
        if (bg < -32768) bg = -32768;
        unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
        move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
    }
}


#ifndef CM_GROUPS_MAX
#define CM_GROUPS_MAX 128
#endif
#ifndef CM_HEAVY_DEG
#define CM_HEAVY_DEG 48
#endif
#ifndef CM_MAX_USED
#define CM_MAX_USED 256
#endif

static __device__ __forceinline__ int select_move_target_200k(
    const unsigned int *pcount,
    const int *shared_nodes_in_part,
    int num_parts,
    int max_part_size,
    int current_part,
    int count_current_present,
    int node,
    int node_degree,
    int crit,
    int tiebreak,
    int gain_floor
) {
    int best_gain = -999999;
    int best_target = current_part;


    for (int target_part = 0; target_part < 64; target_part++) {
        if (pcount[target_part] == 0u) continue;
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int basic_gain = (int)pcount[target_part] - count_current_present;
        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 2) ? 1 : 0;
        int total_gain = basic_gain + balance_bonus;


        bool better = total_gain > best_gain;
        if (!better && total_gain == best_gain && best_target != current_part) {
            if (tiebreak == 1) {
                int ts = shared_nodes_in_part[target_part];
                int bs = shared_nodes_in_part[best_target];
                if (ts != bs) better = ts < bs;
                else better = ((target_part * 17 + node) & 63) < ((best_target * 17 + node) & 63);
            } else if (tiebreak == 2) {
                unsigned int ct = pcount[target_part];
                unsigned int cb = pcount[best_target];
                if (ct != cb) better = ct > cb;
            }
        }
        if (better) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }

    if (best_target != current_part && best_gain >= gain_floor) {
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        if (crit > 255) crit = 255;
        int rank_byte = degree_weight + crit;
        if (rank_byte > 255) rank_byte = 255;
        int bg = best_gain > 32767 ? 32767 : best_gain;
        if (bg < -32768) bg = -32768;
        unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
        return ((int)(short)bg << 16) | (rank_byte << 8) | t16;
    }
    return (int)0x80000000;
}


static __device__ __forceinline__ bool sel_better_200k(
    int gain_a, int t_a, int gain_b, int t_b,
    const unsigned int *pcount, const int *sizes, int current_part, int node, int tiebreak
) {
    if (t_b == current_part) return t_a != current_part;
    if (t_a == current_part) return false;
    if (gain_a != gain_b) return gain_a > gain_b;
    if (tiebreak == 1) {
        int sa = sizes[t_a], sb = sizes[t_b];
        if (sa != sb) return sa < sb;
        int ha = (t_a * 17 + node) & 63, hb = (t_b * 17 + node) & 63;
        if (ha != hb) return ha < hb;
    } else if (tiebreak == 2) {
        unsigned int ca = pcount[t_a], cb = pcount[t_b];
        if (ca != cb) return ca > cb;
    }
    return t_a < t_b;
}


struct BS6_200k { unsigned long long d0, d1, d2, d3, d4, d5; };

static __device__ __forceinline__ void bs6_add_mask_200k(BS6_200k &s, unsigned long long m) {
    unsigned long long c = m, t;
    t = s.d0 & c; s.d0 ^= c; c = t;
    t = s.d1 & c; s.d1 ^= c; c = t;
    t = s.d2 & c; s.d2 ^= c; c = t;
    t = s.d3 & c; s.d3 ^= c; c = t;
    t = s.d4 & c; s.d4 ^= c; c = t;
    s.d5 ^= c;
}

#define BS6_FA_200k(a, b, c) { unsigned long long x_ = (a) ^ (b); unsigned long long cn_ = ((a) & (b)) | ((c) & x_); (a) = x_ ^ (c); (c) = cn_; }
static __device__ __forceinline__ void bs6_add_200k(BS6_200k &a, const BS6_200k &b) {
    unsigned long long c = 0ULL;
    BS6_FA_200k(a.d0, b.d0, c)
    BS6_FA_200k(a.d1, b.d1, c)
    BS6_FA_200k(a.d2, b.d2, c)
    BS6_FA_200k(a.d3, b.d3, c)
    BS6_FA_200k(a.d4, b.d4, c)
    a.d5 = a.d5 ^ b.d5 ^ c;
}

static __device__ __forceinline__ BS6_200k bs6_shfl_xor_200k(const BS6_200k &s, int offset) {
    BS6_200k o;
    o.d0 = shfl_xor_u64_200k(0xffffffffu, s.d0, offset);
    o.d1 = shfl_xor_u64_200k(0xffffffffu, s.d1, offset);
    o.d2 = shfl_xor_u64_200k(0xffffffffu, s.d2, offset);
    o.d3 = shfl_xor_u64_200k(0xffffffffu, s.d3, offset);
    o.d4 = shfl_xor_u64_200k(0xffffffffu, s.d4, offset);
    o.d5 = shfl_xor_u64_200k(0xffffffffu, s.d5, offset);
    return o;
}

static __device__ __forceinline__ unsigned int bs6_get_200k(const BS6_200k &s, int t) {
    return (unsigned int)((s.d0 >> t) & 1ULL)
         | ((unsigned int)((s.d1 >> t) & 1ULL) << 1)
         | ((unsigned int)((s.d2 >> t) & 1ULL) << 2)
         | ((unsigned int)((s.d3 >> t) & 1ULL) << 3)
         | ((unsigned int)((s.d4 >> t) & 1ULL) << 4)
         | ((unsigned int)((s.d5 >> t) & 1ULL) << 5);
}

static __device__ __forceinline__ bool sel_better_bs_200k(
    int gain_a, int t_a, int gain_b, int t_b,
    const BS6_200k &s, const int *sizes, int current_part, int node, int tiebreak
) {
    if (t_b == current_part) return t_a != current_part;
    if (t_a == current_part) return false;
    if (gain_a != gain_b) return gain_a > gain_b;
    if (tiebreak == 1) {
        int sa = sizes[t_a], sb = sizes[t_b];
        if (sa != sb) return sa < sb;
        int ha = (t_a * 17 + node) & 63, hb = (t_b * 17 + node) & 63;
        if (ha != hb) return ha < hb;
    } else if (tiebreak == 2) {
        unsigned int ca = bs6_get_200k(s, t_a), cb = bs6_get_200k(s, t_b);
        if (ca != cb) return ca > cb;
    }
    return t_a < t_b;
}


static __device__ __forceinline__ int select_move_target_bs_200k(
    const BS6_200k &s,
    const int *shared_nodes_in_part,
    int num_parts,
    int max_part_size,
    int current_part,
    int count_current_present,
    int node,
    int node_degree,
    int crit,
    int tiebreak,
    int gain_floor,
    bool eligible,
    int sub,
    int lanes
) {
    int best_gain = -999999;
    int best_target = current_part;
    if (eligible) {
        const int current_size = shared_nodes_in_part[current_part];
        const unsigned long long any = s.d0 | s.d1 | s.d2 | s.d3 | s.d4 | s.d5;
        for (int target_part = sub; target_part < 64; target_part += lanes) {
            if (((any >> target_part) & 1ULL) == 0ULL) continue;
            if ((unsigned)target_part >= (unsigned)num_parts) continue;
            int target_size = shared_nodes_in_part[target_part];
            if (target_size >= max_part_size) continue;
            int total_gain = (int)bs6_get_200k(s, target_part) - count_current_present
                           + ((current_size > target_size + 2) ? 1 : 0);
            if (sel_better_bs_200k(total_gain, target_part, best_gain, best_target,
                                   s, shared_nodes_in_part, current_part, node, tiebreak)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        }
    }
    for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
        int og = __shfl_xor_sync(0xffffffffu, best_gain, offset);
        int ot = __shfl_xor_sync(0xffffffffu, best_target, offset);
        if (eligible && sel_better_bs_200k(og, ot, best_gain, best_target,
                                           s, shared_nodes_in_part, current_part, node, tiebreak)) {
            best_gain = og;
            best_target = ot;
        }
    }
    if (eligible && best_target != current_part && best_gain >= gain_floor) {
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        if (crit > 255) crit = 255;
        int rank_byte = degree_weight + crit;
        if (rank_byte > 255) rank_byte = 255;
        int bg = best_gain > 32767 ? 32767 : best_gain;
        if (bg < -32768) bg = -32768;
        unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
        return ((int)(short)bg << 16) | (rank_byte << 8) | t16;
    }
    return (int)0x80000000;
}


static __device__ __forceinline__ int select_move_target_par_200k(
    const unsigned int *pcount,
    const int *shared_nodes_in_part,
    int num_parts,
    int max_part_size,
    int current_part,
    int count_current_present,
    int node,
    int node_degree,
    int crit,
    int tiebreak,
    int gain_floor,
    bool eligible,
    int sub,
    int lanes
) {
    int best_gain = -999999;
    int best_target = current_part;
    if (eligible) {
        const int current_size = shared_nodes_in_part[current_part];
        for (int target_part = sub; target_part < 64; target_part += lanes) {
            if (pcount[target_part] == 0u) continue;
            if ((unsigned)target_part >= (unsigned)num_parts) continue;
            int target_size = shared_nodes_in_part[target_part];
            if (target_size >= max_part_size) continue;
            int total_gain = (int)pcount[target_part] - count_current_present
                           + ((current_size > target_size + 2) ? 1 : 0);
            if (sel_better_200k(total_gain, target_part, best_gain, best_target,
                                pcount, shared_nodes_in_part, current_part, node, tiebreak)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        }
    }
    for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
        int og = __shfl_xor_sync(0xffffffffu, best_gain, offset);
        int ot = __shfl_xor_sync(0xffffffffu, best_target, offset);
        if (eligible && sel_better_200k(og, ot, best_gain, best_target,
                                        pcount, shared_nodes_in_part, current_part, node, tiebreak)) {
            best_gain = og;
            best_target = ot;
        }
    }
    if (eligible && best_target != current_part && best_gain >= gain_floor) {
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        if (crit > 255) crit = 255;
        int rank_byte = degree_weight + crit;
        if (rank_byte > 255) rank_byte = 255;
        int bg = best_gain > 32767 ? 32767 : best_gain;
        if (bg < -32768) bg = -32768;
        unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
        return ((int)(short)bg << 16) | (rank_byte << 8) | t16;
    }
    return (int)0x80000000;
}

extern "C" __global__ void compute_refinement_moves_lanes_200k(
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
    const int lanes,
    const int tiebreak,
    const int gain_floor
) {
    __shared__ int shared_nodes_in_part[64];
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

        int my_present = 0;
        int my_crit = 0;
        if (eligible) {
            unsigned long long current_bit = 1ULL << current_part;

            for (int j = start + sub; j < end; j += lanes) {
                int hyperedge = __ldg(&node_hyperedges[j]);
                unsigned long long fa = __ldg(&edge_flags_all[hyperedge]);
                unsigned long long fd = __ldg(&edge_flags_double[hyperedge]);
                if ((fa & current_bit) != 0ULL && (fd & current_bit) == 0ULL) {
                    int parts = __popcll(fa);
                    if (parts > 1) my_crit += (parts - 1);
                }
                if (fd & current_bit) my_present++;
                unsigned long long f = fa & ~current_bit;
                while (f) {
                    int bit = __ffsll(f) - 1;
                    f &= (f - 1);
                    atomicAdd(&pinfo[bit], 1u);
                }
            }
        }
        for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
            my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
            my_crit    += __shfl_xor_sync(0xffffffffu, my_crit, offset);
        }
        __syncwarp();

        if (sub == 0 && node < num_nodes && regular) {
            int key = (int)0x80000000;
            if (eligible) {
                key = select_move_target_200k(pinfo, shared_nodes_in_part, num_parts, max_part_size,
                                             current_part, my_present, node, node_degree, my_crit, tiebreak, gain_floor);
            }
            move_priorities[node] = key;
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

            int my_present = 0;
            int my_crit = 0;
            if (eligible) {
                unsigned long long current_bit = 1ULL << current_part;
                int used_degree = node_degree > CM_MAX_USED ? CM_MAX_USED : node_degree;
                for (int j = lane; j < used_degree; j += 32) {
                    int rel = (int)(((long long)j * node_degree) / used_degree);
                    int hyperedge = __ldg(&node_hyperedges[start + rel]);
                    unsigned long long fa = __ldg(&edge_flags_all[hyperedge]);
                    unsigned long long fd = __ldg(&edge_flags_double[hyperedge]);
                    if ((fa & current_bit) != 0ULL && (fd & current_bit) == 0ULL) {
                        int parts = __popcll(fa);
                        if (parts > 1) my_crit += (parts - 1);
                    }
                    if (fd & current_bit) my_present++;
                    unsigned long long f = fa & ~current_bit;
                    while (f) {
                        int bit = __ffsll(f) - 1;
                        f &= (f - 1);
                        atomicAdd(&pinfo[bit], 1u);
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
                my_crit    += __shfl_xor_sync(0xffffffffu, my_crit, offset);
            }
            __syncwarp();

            if (lane == 0) {
                int key = (int)0x80000000;
                if (eligible) {
                    key = select_move_target_200k(pinfo, shared_nodes_in_part, num_parts, max_part_size,
                                                 current_part, my_present, hn, node_degree, my_crit, tiebreak, gain_floor);
                }
                move_priorities[hn] = key;
            }
            __syncwarp();
        }
    }
}

extern "C" __global__ void compute_swap_gains_extended_200k(
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

        unsigned long long flags_all = edge_flags_all[hyperedge];
        unsigned long long flags_double = edge_flags_double[hyperedge];
        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

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
    top_gain[0] = min_gain - 1;
    top_gain[1] = min_gain - 1;
    top_gain[2] = min_gain - 1;
    top_part[0] = -1;
    top_part[1] = -1;
    top_part[2] = -1;

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

extern "C" __global__ void compute_connectivity_200k(
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

extern "C" __global__ void reduce_connectivity_sum_200k(
    const int num_hyperedges,
    const int *connectivity,
    int *total_connectivity_blocks
) {
    extern __shared__ int shared_sum[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int stride = blockDim.x * 2 * gridDim.x;

    int local_sum = 0;
    while (idx < num_hyperedges) {
        local_sum += connectivity[idx];
        int idx2 = idx + blockDim.x;
        if (idx2 < num_hyperedges) {
            local_sum += connectivity[idx2];
        }
        idx += stride;
    }

    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    if ((tid & 31) == 0) {
        shared_sum[tid >> 5] = local_sum;
    }
    __syncthreads();

    if (tid < 32) {
        int num_warps = (blockDim.x + 31) >> 5;
        local_sum = (tid < num_warps) ? shared_sum[tid] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        if (tid == 0) {
            total_connectivity_blocks[blockIdx.x] = local_sum;
        }
    }
}


extern "C" __global__ void balance_final_200k(
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


__device__ __forceinline__ unsigned long long lcg_jump_200k(unsigned long long x0, unsigned int k) {
    const unsigned long long a = 6364136223846793005ULL;
    const unsigned long long c = 1442695040888963407ULL;
    unsigned long long A = 1ULL, C = 0ULL;
    unsigned long long Ab = a, Cb = c;
    while (k) {
        if (k & 1u) { C = A * Cb + C; A = A * Ab; }
        Cb = Ab * Cb + Cb; Ab = Ab * Ab;
        k >>= 1;
    }
    return A * x0 + C;
}


#include <cooperative_groups.h>

extern "C" __global__ void filter_fused_200k(
    const int num_nodes,
    const unsigned int round,
    const int *keys,
    const unsigned int *tabu_until,
    int *stats,
    unsigned int *tile_off,
    int *hdr,
    const unsigned long long seed,
    const unsigned long long prob_num,
    const unsigned long long den0,
    const unsigned long long den1,
    const unsigned long long den2,
    const unsigned long long den3,
    unsigned long long *out
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ int red[7][32];
    __shared__ int red_max[1024];
    __shared__ int red_valid[1024];
    __shared__ unsigned int red_cnt[1024];
    __shared__ unsigned int warp_low[32];
    __shared__ int s_asp;

    const int T = blockDim.x;
    const int num_tiles = (num_nodes + T - 1) / T;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nwarps = (blockDim.x + 31u) >> 5;
    const unsigned lanemask_lt = (1u << lane) - 1u;


    for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        const int node = t * T + threadIdx.x;
        int mx = (-2147483647 - 1);
        int v_valid = 0, v_free = 0, v_m0 = 0, v_m1 = 0, v_m2 = 0, v_m3 = 0;
        if (node < num_nodes) {
            int key = keys[node];
            if ((unsigned)key != 0x80000000u) {
                int gain = key >> 16;
                mx = gain;
                v_valid = 1;
                if (gain <= 0 && gain >= -3) {
                    bool marked = tabu_until[node] > round;
                    if (!marked) v_free = 1;
                    else if (gain == 0) v_m0 = 1;
                    else if (gain == -1) v_m1 = 1;
                    else if (gain == -2) v_m2 = 1;
                    else v_m3 = 1;
                }
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            v_valid += __shfl_xor_sync(0xffffffffu, v_valid, off);
            v_free  += __shfl_xor_sync(0xffffffffu, v_free, off);
            v_m0    += __shfl_xor_sync(0xffffffffu, v_m0, off);
            v_m1    += __shfl_xor_sync(0xffffffffu, v_m1, off);
            v_m2    += __shfl_xor_sync(0xffffffffu, v_m2, off);
            v_m3    += __shfl_xor_sync(0xffffffffu, v_m3, off);
        }
        if (lane == 0) {
            red[0][warp] = mx; red[1][warp] = v_valid; red[2][warp] = v_free;
            red[3][warp] = v_m0; red[4][warp] = v_m1; red[5][warp] = v_m2; red[6][warp] = v_m3;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int bmx = (-2147483647 - 1), s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (unsigned w = 0; w < nwarps; w++) {
                bmx = max(bmx, red[0][w]); s1 += red[1][w]; s2 += red[2][w];
                s3 += red[3][w]; s4 += red[4][w]; s5 += red[5][w]; s6 += red[6][w];
            }
            int *o = stats + t * 8;
            o[0] = bmx; o[1] = s1; o[2] = s2; o[3] = s3; o[4] = s4; o[5] = s5; o[6] = s6; o[7] = 0;
        }
        __syncthreads();
    }
    grid.sync();


    if (blockIdx.x == 0) {
        const int tid = threadIdx.x;
        const int nt = blockDim.x;
        int mx = (-2147483647 - 1);
        int valid = 0;
        for (int b = tid; b < num_tiles; b += nt) {
            mx = max(mx, stats[b * 8]);
            valid += stats[b * 8 + 1];
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
            int asp = (red_valid[0] > 0) ? ((m * 3) / 4) : 0;
            s_asp = asp;
            hdr[0] = red_valid[0];
            hdr[1] = asp;
            hdr[2] = m;
            hdr[3] = 0;
        }
        __syncthreads();
        const int asp = s_asp;
        const int per = (num_tiles + nt - 1) / nt;
        const int b0 = tid * per;
        const int b1 = min(num_tiles, b0 + per);
        unsigned int local = 0;
        for (int b = b0; b < b1; b++) {
            const int *s = stats + b * 8;
            unsigned int c = (unsigned int)s[2];
            for (int g = 0; g < 4; g++) {
                if (-g >= asp) c += (unsigned int)s[3 + g];
            }
            local += c;
        }
        red_cnt[tid] = local;
        __syncthreads();
        if (tid == 0) {
            unsigned int acc = 0;
            for (int i = 0; i < nt; i++) {
                unsigned int v = red_cnt[i];
                red_cnt[i] = acc;
                acc += v;
            }
        }
        __syncthreads();
        unsigned int acc = red_cnt[tid];
        for (int b = b0; b < b1; b++) {
            tile_off[b] = acc;
            const int *s = stats + b * 8;
            unsigned int c = (unsigned int)s[2];
            for (int g = 0; g < 4; g++) {
                if (-g >= asp) c += (unsigned int)s[3 + g];
            }
            acc += c;
        }
    }
    grid.sync();


    const int aspiration_h = hdr[1];
    unsigned int *count = (unsigned int *)(hdr + 3);
    for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        const int node = t * T + threadIdx.x;
        int key = 0;
        int gain = 0;
        bool valid = false;
        bool pos = false;
        bool low = false;
        if (node < num_nodes) {
            key = keys[node];
            valid = ((unsigned)key != 0x80000000u);
            if (valid) {
                gain = key >> 16;
                bool is_tabu = (tabu_until[node] > round) && (gain < aspiration_h);
                if (!is_tabu) {
                    if (gain > 0) pos = true;
                    else if (gain >= -3) low = true;
                }
            }
        }
        unsigned low_mask = __ballot_sync(0xffffffffu, low);
        unsigned low_rank = __popc(low_mask & lanemask_lt);
        if (lane == 0) warp_low[warp] = __popc(low_mask);
        __syncthreads();
        unsigned warp_prefix = 0;
        for (unsigned w = 0; w < warp; w++) warp_prefix += warp_low[w];

        bool accept = pos;
        unsigned int out_key32 = (unsigned int)key;
        if (low) {
            unsigned int j = tile_off[t] + warp_prefix + low_rank;
            unsigned long long x = lcg_jump_200k(seed, j + 1u);
            int penalty = -gain;
            unsigned long long den = (penalty == 0) ? den0 : (penalty == 1) ? den1 : (penalty == 2) ? den2 : den3;
            if (den > 0ULL && (x % den) < prob_num) {
                accept = true;
                out_key32 = (unsigned int)(key & 0xFFFF);
            }
        }
        unsigned acc_mask = __ballot_sync(0xffffffffu, accept);
        unsigned base = 0;
        if (lane == 0 && acc_mask) base = atomicAdd(count, __popc(acc_mask));
        base = __shfl_sync(0xffffffffu, base, 0);
        if (accept) {
            unsigned idx = base + __popc(acc_mask & lanemask_lt);
            out[idx] = (((unsigned long long)(~out_key32)) << 32) | (unsigned long long)(unsigned int)node;
        }
        __syncthreads();
    }
}


#define RF_TILE_200k 128
#define RF_WPT_200k 4


static __device__ __forceinline__ bool rf_barrier_200k(
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

#define RF_FILT_WORDS_200k (8 * 256)


#ifndef HG_RF_SMEM_DECLARED
#define HG_RF_SMEM_DECLARED
extern __shared__ unsigned int rf_smem[];
#endif
static __device__ __forceinline__ bool rf_round_200k(
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
    const int tiebreak,
    const int gain_floor,
    const unsigned int round,
    unsigned int *tabu_until,
    int *stats,
    unsigned int *tile_off,
    int *hdr,
    const unsigned long long seed,
    const unsigned long long prob_num,
    const unsigned long long den0,
    const unsigned long long den1,
    const unsigned long long den2,
    const unsigned long long den3,
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
    const int gsort,
    unsigned int *tcnt,
    unsigned int *koff,
    const int slack,
    const int adaptive_limit,
    const int extra_window
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ int s_abort;
    unsigned int *bar = (unsigned int *)(hdr + 4);
    int *abort_flag = hdr + 28;
    const unsigned int bar_target = (epoch + 1u) * gridDim.x;
#define RF_BAR_200k(k) if (!rf_barrier_200k(grid, soft, bar + (k), bar_target, abort_flag, &s_abort)) return false;
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
    RF_BAR_200k(0)


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
                unsigned long long other_all = shfl_xor_u64_200k(0xffffffffu, local_all, offset);
                unsigned long long other_double = shfl_xor_u64_200k(0xffffffffu, local_double, offset);
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
    RF_BAR_200k(1)


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


            BS6_200k bs = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
            int my_present = 0;
            int my_crit = 0;
            if (eligible) {
                unsigned long long current_bit = 1ULL << current_part;
                for (int j = start + sub; j < end; j += lanes) {
                    int hyperedge = __ldg(&node_hyperedges[j]);
                    ulonglong2 fp = __ldcg((const ulonglong2 *)edge_flags_pair + hyperedge);
                    unsigned long long fa = fp.x;
                    unsigned long long fd = fp.y;
                    if ((fa & current_bit) != 0ULL && (fd & current_bit) == 0ULL) {
                        int parts = __popcll(fa);
                        if (parts > 1) my_crit += (parts - 1);
                    }
                    if (fd & current_bit) my_present++;
                    bs6_add_mask_200k(bs, fa & ~current_bit);
                }
            }
            for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
                my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
                my_crit    += __shfl_xor_sync(0xffffffffu, my_crit, offset);
                BS6_200k o = bs6_shfl_xor_200k(bs, offset);
                bs6_add_200k(bs, o);
            }

            {
                int key = select_move_target_bs_200k(bs, shared_nodes_in_part, num_parts, max_part_size,
                                                    current_part, my_present, node, node_degree, my_crit,
                                                    tiebreak, gain_floor, eligible, sub, lanes);
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
            int my_crit = 0;
            if (eligible) {
                unsigned long long current_bit = 1ULL << current_part;
                int used_degree = node_degree > CM_MAX_USED ? CM_MAX_USED : node_degree;
                for (int j = lane; j < used_degree; j += 32) {
                    int rel = (int)(((long long)j * node_degree) / used_degree);
                    int hyperedge = __ldg(&node_hyperedges[start + rel]);
                    ulonglong2 fp = __ldcg((const ulonglong2 *)edge_flags_pair + hyperedge);
                    unsigned long long fa = fp.x;
                    unsigned long long fd = fp.y;
                    if ((fa & current_bit) != 0ULL && (fd & current_bit) == 0ULL) {
                        int parts = __popcll(fa);
                        if (parts > 1) my_crit += (parts - 1);
                    }
                    if (fd & current_bit) my_present++;
                    unsigned long long f = fa & ~current_bit;
                    while (f) {
                        int bit = __ffsll(f) - 1;
                        f &= (f - 1);
                        atomicAdd(&wpinfo[bit], 1u);
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
                my_crit    += __shfl_xor_sync(0xffffffffu, my_crit, offset);
            }
            __syncwarp();

            {
                int key = select_move_target_par_200k(wpinfo, shared_nodes_in_part, num_parts, max_part_size,
                                                     current_part, my_present, hn, node_degree, my_crit,
                                                     tiebreak, gain_floor, eligible, lane, 32);
                if (lane == 0) move_priorities[hn] = key;
            }
            __syncwarp();
        }
    }
    RF_BAR_200k(2)


    if (gsort) {
        const int nt_max = (num_nodes + 255) >> 8;
        const int rs_total = 4 * 256 * nt_max;
        for (int i = gtid; i < rs_total; i += nthreads) rs_counts[i] = 0u;
        for (int i = gtid; i < 64 * nt_max; i += nthreads) tcnt[i] = 0u;
    }
    const int num_tiles = (num_nodes + RF_TILE_200k - 1) / RF_TILE_200k;
    const int subs = blockDim.x / RF_TILE_200k;
    const int tsub = threadIdx.x / RF_TILE_200k;
    const int tth = threadIdx.x & (RF_TILE_200k - 1);
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const int num_groups = (num_tiles + subs - 1) / subs;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_200k + tth;
        int mx = (-2147483647 - 1);
        int v_valid = 0, v_free = 0, v_m0 = 0, v_m1 = 0, v_m2 = 0, v_m3 = 0;
        if (t < num_tiles && node < num_nodes) {
            int key = __ldcg(move_priorities + node);
            if ((unsigned)key != 0x80000000u) {
                int gain = key >> 16;
                mx = gain;
                v_valid = 1;
                if (gain <= 0 && gain >= -3) {
                    bool marked = __ldcg(tabu_until + node) > round;
                    if (!marked) v_free = 1;
                    else if (gain == 0) v_m0 = 1;
                    else if (gain == -1) v_m1 = 1;
                    else if (gain == -2) v_m2 = 1;
                    else v_m3 = 1;
                }
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            v_valid += __shfl_xor_sync(0xffffffffu, v_valid, off);
            v_free  += __shfl_xor_sync(0xffffffffu, v_free, off);
            v_m0    += __shfl_xor_sync(0xffffffffu, v_m0, off);
            v_m1    += __shfl_xor_sync(0xffffffffu, v_m1, off);
            v_m2    += __shfl_xor_sync(0xffffffffu, v_m2, off);
            v_m3    += __shfl_xor_sync(0xffffffffu, v_m3, off);
        }
        if (lane == 0) {
            red[0 * 32 + warp_id] = mx; red[1 * 32 + warp_id] = v_valid; red[2 * 32 + warp_id] = v_free;
            red[3 * 32 + warp_id] = v_m0; red[4 * 32 + warp_id] = v_m1; red[5 * 32 + warp_id] = v_m2; red[6 * 32 + warp_id] = v_m3;
        }
        __syncthreads();
        if (tth == 0 && t < num_tiles) {
            int bmx = (-2147483647 - 1), s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int w = tsub * RF_WPT_200k; w < (tsub + 1) * RF_WPT_200k; w++) {
                bmx = max(bmx, red[0 * 32 + w]); s1 += red[1 * 32 + w]; s2 += red[2 * 32 + w];
                s3 += red[3 * 32 + w]; s4 += red[4 * 32 + w]; s5 += red[5 * 32 + w]; s6 += red[6 * 32 + w];
            }
            int *o = stats + t * 8;
            o[0] = bmx; o[1] = s1; o[2] = s2; o[3] = s3; o[4] = s4; o[5] = s5; o[6] = s6; o[7] = 0;
        }
        __syncthreads();
    }
    RF_BAR_200k(3)


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
            int asp = (red_valid[0] > 0) ? ((m * 3) / 4) : 0;
            *s_asp_p = asp;
            hdr[0] = red_valid[0];
            hdr[1] = asp;
            hdr[2] = m;
            hdr[3] = 0;
        }
        __syncthreads();
        const int asp = *s_asp_p;
        const int per = (num_tiles + nt - 1) / nt;
        const int b0 = tid * per;
        const int b1 = min(num_tiles, b0 + per);
        unsigned int local = 0;
        for (int b = b0; b < b1; b++) {
            const int *s = stats + b * 8;
            unsigned int c = (unsigned int)__ldcg(s + 2);
            for (int g = 0; g < 4; g++) {
                if (-g >= asp) c += (unsigned int)__ldcg(s + 3 + g);
            }
            local += c;
        }
        red_cnt[tid] = local;
        __syncthreads();
        if (tid == 0) {
            unsigned int acc = 0;
            for (int i = 0; i < nt; i++) {
                unsigned int v = red_cnt[i];
                red_cnt[i] = acc;
                acc += v;
            }
        }
        __syncthreads();
        unsigned int acc = red_cnt[tid];
        for (int b = b0; b < b1; b++) {
            tile_off[b] = acc;
            const int *s = stats + b * 8;
            unsigned int c = (unsigned int)__ldcg(s + 2);
            for (int g = 0; g < 4; g++) {
                if (-g >= asp) c += (unsigned int)__ldcg(s + 3 + g);
            }
            acc += c;
        }
    }
    RF_BAR_200k(4)


    const int aspiration_h = __ldcg(hdr + 1);
    unsigned int *warp_acc = warp_low + 16;
    unsigned int *count = (unsigned int *)(hdr + 3);
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_200k + tth;
        int key = 0;
        int gain = 0;
        bool pos = false;
        bool low = false;
        if (t < num_tiles && node < num_nodes) {
            key = __ldcg(move_priorities + node);
            if ((unsigned)key != 0x80000000u) {
                gain = key >> 16;
                bool is_tabu = (__ldcg(tabu_until + node) > round) && (gain < aspiration_h);
                if (!is_tabu) {
                    if (gain > 0) pos = true;
                    else if (gain >= -3) low = true;
                }
            }
        }
        unsigned low_mask = __ballot_sync(0xffffffffu, low);
        unsigned low_rank = __popc(low_mask & lanemask_lt);
        if (lane == 0) warp_low[warp_id] = __popc(low_mask);
        __syncthreads();
        unsigned warp_prefix = 0;
        for (int w = tsub * RF_WPT_200k; w < warp_id; w++) warp_prefix += warp_low[w];

        bool accept = pos;
        unsigned int out_key32 = (unsigned int)key;
        if (low) {
            unsigned int j = __ldcg(tile_off + t) + warp_prefix + low_rank;
            unsigned long long x = lcg_jump_200k(seed, j + 1u);
            int penalty = -gain;
            unsigned long long den = (penalty == 0) ? den0 : (penalty == 1) ? den1 : (penalty == 2) ? den2 : den3;
            if (den > 0ULL && (x % den) < prob_num) {
                accept = true;
                out_key32 = (unsigned int)(key & 0xFFFF);
            }
        }
        unsigned acc_mask = __ballot_sync(0xffffffffu, accept);
        if (!gsort) {
            unsigned base = 0;
            if (lane == 0 && acc_mask) base = atomicAdd(count, __popc(acc_mask));
            base = __shfl_sync(0xffffffffu, base, 0);
            if (accept) {
                unsigned idx = base + __popc(acc_mask & lanemask_lt);
                out[idx] = (((unsigned long long)(~out_key32)) << 32) | (unsigned long long)(unsigned int)node;
            }
            __syncthreads();
            continue;
        }
        if (t < num_tiles && node < num_nodes) acc[node] = accept ? ~out_key32 : 0u;
        if (lane == 0) warp_acc[warp_id] = __popc(acc_mask);
        __syncthreads();
        if (tth == 0 && t < num_tiles) {
            unsigned c = 0;
            for (int w = tsub * RF_WPT_200k; w < (tsub + 1) * RF_WPT_200k; w++) c += warp_acc[w];
            stats[t * 8 + 7] = (int)c;
        }
        __syncthreads();
    }
    if (!gsort) {
        return true;
    }
    RF_BAR_200k(5)


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
            const int mg = __ldcg(hdr + 2);
            hdr[29] = (mg >= 256) ? 0 : 1;
        }
        __syncthreads();
        unsigned int a = red_cnt[tid];
        for (int b = b0; b < b1; b++) {
            acc_off[b] = a;
            a += (unsigned int)__ldcg(stats + b * 8 + 7);
        }
    }
    RF_BAR_200k(6)


    const unsigned int cnt_c = (unsigned int)__ldcg(hdr + 3);
    const int nt_c = (int)((cnt_c + 255u) >> 8);
    const int npass = (__ldcg(hdr + 2) >= 256) ? 4 : 3;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_200k + tth;
        unsigned int v = 0u;
        if (t < num_tiles && node < num_nodes) v = __ldcg(acc + node);
        unsigned m = __ballot_sync(0xffffffffu, v != 0u);
        unsigned r = __popc(m & lanemask_lt);
        if (lane == 0) warp_low[warp_id] = __popc(m);
        __syncthreads();
        unsigned pre = 0;
        for (int w = tsub * RF_WPT_200k; w < warp_id; w++) pre += warp_low[w];
        if (v != 0u) {
            unsigned pos = __ldcg(acc_off + t) + pre + r;
            out[pos] = ((unsigned long long)v << 32) | (unsigned long long)(unsigned int)node;
            atomicAdd(&rs_counts[(pos >> 8) * 256u + (v & 0xFFu)], 1u);
        }
        __syncthreads();
    }

    unsigned int *wc = rf_smem;


    for (int k = 0; k < 4; k++) {
        RF_BAR_200k(7 + 2 * k)
        unsigned int *ck = rs_counts + k * 256 * nt_c;
        if (k < npass && blockIdx.x == 0) {


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
        RF_BAR_200k(8 + 2 * k)
        const unsigned long long *src = (k & 1) ? out2 : out;
        unsigned long long *dst = (k & 1) ? out : out2;
        unsigned int *cn = rs_counts + (k + 1) * 256 * nt_c;
        const int sh = 32 + 8 * k;
        if (k >= npass) continue;
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


    const unsigned long long *sorted_l = (npass & 1) ? out2 : out;
    unsigned long long *kept_l = (npass & 1) ? out : out2;
    const unsigned int k_base_d = min(cnt_c, (unsigned int)adaptive_limit);
    const unsigned int k_cand_d = min(cnt_c, k_base_d + (unsigned int)extra_window);
    RF_BAR_200k(15)
    if (blockIdx.x == 0 && threadIdx.x < 64) {

        const int t = threadIdx.x;
        unsigned int run = 0;
        for (int tile = 0; tile < nt_c; tile++) {
            unsigned int c = __ldcg(tcnt + tile * 64 + t);
            tcnt[tile * 64 + t] = run;
            run += c;
        }
    }
    RF_BAR_200k(16)
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
    RF_BAR_200k(17)
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
    RF_BAR_200k(18)
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


extern "C" __global__ void __launch_bounds__(256, 4) round_fused_200k(
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
    const int tiebreak,
    const int gain_floor,
    const unsigned int round,
    unsigned int *tabu_until,
    int *stats,
    unsigned int *tile_off,
    int *hdr,
    const unsigned long long seed,
    const unsigned long long prob_num,
    const unsigned long long den0,
    const unsigned long long den1,
    const unsigned long long den2,
    const unsigned long long den3,
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
    const int gsort,
    unsigned int *tcnt,
    unsigned int *koff,
    const int slack,
    const int adaptive_limit,
    const int extra_window
) {
    rf_round_200k(num_hyperedges, num_nodes, num_parts, max_part_size, hyperedge_nodes, hyperedge_offsets,
                  node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_pair, move_priorities,
                  lanes, tiebreak, gain_floor, round, tabu_until, stats, tile_off, hdr, seed, prob_num,
                  den0, den1, den2, den3, out, soft, launch_idx, n_tabu_upd, tabu_pairs,
                  n_part_upd, part_pairs, acc, acc_off, out2, rs_counts, gsort, tcnt, koff, slack,
                  adaptive_limit, extra_window);
}


static __device__ __forceinline__ unsigned long long rf_sat_mul_200k(unsigned long long a, unsigned long long b) {
    return (__umul64hi(a, b) != 0ULL) ? 0xFFFFFFFFFFFFFFFFULL : a * b;
}


extern "C" __global__ void __launch_bounds__(256, 4) rounds_fused_200k(
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
    const int tiebreak,
    const int gain_floor,
    const int round0,
    const int max_rounds,
    const int move_limit,
    const int tabu_tenure,
    const int tabu_mark_base,
    const int tabu_mark_mult,
    const int tabu_exec,
    const int slack_early,
    const int slack_mid,
    const int slack_late,
    const int extra_window,
    const int cycle_end,
    const int sched_exp,
    const unsigned long long num_mul,
    unsigned int *tabu_until,
    int *stats,
    unsigned int *tile_off,
    int *hdr,
    const unsigned long long den0,
    const unsigned long long den1,
    const unsigned long long den2,
    const unsigned long long den3,
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
    unsigned int *koff
) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ int s_abort2;
    __shared__ int s_nip[64];
    unsigned int *bar = (unsigned int *)(hdr + 4);
    int *abort_flag = hdr + 28;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        hdr[25] = 0; hdr[26] = 0; hdr[27] = 0; hdr[31] = 0; hdr[32] = 0; hdr[33] = 0;
    }
    for (int r = 0; r < max_rounds; r++) {
        const int round = round0 + r;
        const unsigned int epoch = epoch0 + (unsigned int)r;
        const int adaptive_limit = (round < 50) ? (move_limit / 2) : (round < 200) ? ((move_limit * 3) / 4) : (move_limit / 4);
        const int slack = (round < 64) ? slack_early : (round < 256) ? slack_mid : slack_late;
        const unsigned long long seed = 123456789ULL + (unsigned long long)round;
        unsigned long long left = (unsigned long long)(cycle_end > round ? cycle_end - round : 0);
        unsigned long long pw = 1ULL;
        for (int i = 0; i < sched_exp; i++) pw *= left;
        const unsigned long long prob_num = rf_sat_mul_200k(pw, num_mul);

        const int ntu = (r == 0) ? n_tabu_upd : 0;
        const int npu = (r == 0) ? n_part_upd : 0;
        if (!rf_round_200k(num_hyperedges, num_nodes, num_parts, max_part_size, hyperedge_nodes, hyperedge_offsets,
                           node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_pair, move_priorities,
                           lanes, tiebreak, gain_floor, (unsigned int)round, tabu_until, stats, tile_off, hdr, seed,
                           prob_num, den0, den1, den2, den3, out, soft, epoch, ntu, tabu_pairs, npu,
                           part_pairs, acc, acc_off, out2, rs_counts, 1, tcnt, koff, slack, adaptive_limit,
                           extra_window)) {
            return;
        }


        if (!rf_barrier_200k(grid, soft, bar + 19, (epoch + 1u) * gridDim.x, abort_flag, &s_abort2)) return;


        if (blockIdx.x == 0) {

            int *s_node = (int *)rf_smem;
            int *s_tgt = (int *)rf_smem + 256;
            int *s_cur = (int *)rf_smem + 512;
            int *s_ct = (int *)rf_smem + 768;
            int *s_cc = (int *)rf_smem + 832;
            unsigned int *s_mask = rf_smem + 896;
            const int tid = threadIdx.x;
            const unsigned int cnt = (unsigned int)__ldcg(hdr + 3);
            const unsigned int kept = min((unsigned int)__ldcg(hdr + 30), cnt);
            const unsigned int k_base = min(cnt, (unsigned int)adaptive_limit);
            const unsigned int take = min(kept, k_base);

            const unsigned long long *kept_l = (__ldcg(hdr + 29) == 1) ? out : out2;
            const unsigned int until = (unsigned int)round + (unsigned int)tabu_tenure;
            if (tid < 64) s_nip[tid] = (tid < num_parts) ? __ldcg(nodes_in_part + tid) : 0;
            __syncthreads();
            unsigned int executed = 0;
            for (unsigned int base = 0; base < take; base += 256) {
                const unsigned int i = base + (unsigned int)tid;
                const bool has = i < take;
                int node = -1, tgt = -1, cur = -1;
                if (has) {
                    unsigned long long v = __ldcg(kept_l + i);
                    node = (int)(unsigned int)(v & 0xFFFFFFFFULL);
                    tgt = (int)((~(unsigned int)(v >> 32)) & 63u);
                    if ((unsigned)node < (unsigned)num_nodes) cur = __ldcg(partition + node);
                }
                const bool valid = has && (unsigned)node < (unsigned)num_nodes
                                   && (unsigned)tgt < (unsigned)num_parts && (unsigned)cur < (unsigned)num_parts;
                s_node[tid] = node; s_tgt[tid] = tgt; s_cur[tid] = cur;
                if (tid < 64) { s_ct[tid] = 0; s_cc[tid] = 0; }
                __syncthreads();
                if (valid) { atomicAdd(&s_ct[tgt], 1); atomicAdd(&s_cc[cur], 1); }
                __syncthreads();
                bool ok = true;
                if (tid < 64) {
                    const int n0 = s_nip[tid];
                    if (s_ct[tid] > 0 && n0 + s_ct[tid] > max_part_size) ok = false;
                    if (s_cc[tid] > 0 && n0 - s_cc[tid] < 1) ok = false;
                }
                const int nvalid = __syncthreads_count(valid);
                const bool all_ok = __syncthreads_and(ok) != 0;
                bool apply;
                if (all_ok) {
                    apply = valid;
                    if (tid < 64) s_nip[tid] += s_ct[tid] - s_cc[tid];
                    executed += (unsigned int)nvalid;
                } else {
                    if (tid == 0) {
                        unsigned int m[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
                        unsigned int ex = 0u;
                        const unsigned int n = min(256u, take - base);
                        for (unsigned int j = 0; j < n; j++) {
                            const int nj = s_node[j], tj = s_tgt[j], cj = s_cur[j];
                            if ((unsigned)nj < (unsigned)num_nodes && (unsigned)tj < (unsigned)num_parts
                                && (unsigned)cj < (unsigned)num_parts
                                && s_nip[tj] < max_part_size && s_nip[cj] > 1) {
                                s_nip[cj] -= 1;
                                s_nip[tj] += 1;
                                m[j >> 5] |= (1u << (j & 31));
                                ex++;
                            }
                        }
                        for (int k = 0; k < 8; k++) s_mask[k] = m[k];
                        s_mask[8] = ex;
                    }
                    __syncthreads();
                    apply = has && (((s_mask[tid >> 5] >> (tid & 31)) & 1u) != 0u);
                    executed += s_mask[8];
                }
                if (apply) {
                    partition[node] = tgt;
                    if (tabu_exec) tabu_until[node] = until;
                }
                __syncthreads();
            }
            if (executed > 0u) {
                if (!tabu_exec) {


                    unsigned int mark_len = max((unsigned int)tabu_mark_base, executed * (unsigned int)tabu_mark_mult);
                    mark_len = min(mark_len, take);
                    for (unsigned int i = (unsigned int)tid; i < mark_len; i += 256u) {
                        unsigned long long v = __ldcg(kept_l + i);
                        int node = (int)(unsigned int)(v & 0xFFFFFFFFULL);
                        if ((unsigned)node < (unsigned)num_nodes) tabu_until[node] = until;
                    }
                }
                if (tid < num_parts && tid < 64) nodes_in_part[tid] = s_nip[tid];
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                if (executed > 0u) {


                    hdr[26] += (int)cnt;
                    hdr[27] += (int)take;
                    hdr[32] += (int)executed;
                    hdr[33] += __ldcg(hdr + 0);
                }
                hdr[31] = r + 1;
                hdr[25] = (cnt == 0u) ? 1 : (executed == 0u) ? 2 : 0;
                __threadfence();
            }
        }

        if (!rf_barrier_200k(grid, soft, bar + 20, (epoch + 1u) * gridDim.x, abort_flag, &s_abort2)) return;
        if (__ldcg(hdr + 25) != 0) return;
    }
}

#define FF200K_SENT    ((int)0x80000000)
#define FF200K_MG_INIT (-1048576)

#if 0
static __device__ __forceinline__ unsigned long long ff200k_lcg_advance(
    unsigned long long x, unsigned long long n)
{
    unsigned long long A = 1ULL, C = 0ULL;
    unsigned long long Ab = 6364136223846793005ULL, Cb = 1442695040888963407ULL;
    while (n != 0ULL) {
        if (n & 1ULL) {
            C = Ab * C + Cb;
            A = Ab * A;
        }
        Cb = Ab * Cb + Cb;
        Ab = Ab * Ab;
        n >>= 1;
    }
    return A * x + C;
}
#endif

#if 0
static __device__ __forceinline__ unsigned int ff200k_ld_vol_u32(const unsigned int *p)
{
    unsigned int r;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
}
#endif

#define FF200K_GRID_BARRIER(gb, target)                        \
    do {                                                   \
        __threadfence();                                   \
        __syncthreads();                                   \
        if (threadIdx.x == 0) {                            \
            atomicAdd((gb), 1u);                           \
            while (ff200k_ld_vol_u32(gb) < (target)) {         \
                __nanosleep(20);                          \
            }                                              \
        }                                                  \
        __syncthreads();                                   \
        __threadfence();                                   \
    } while (0)

static __device__ __forceinline__ int ff200k_block_sum(int v, int *s_warp)
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

static __device__ __forceinline__ int ff200k_block_max(int v, int *s_warp)
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

static __device__ __forceinline__ int ff200k_block_scan(int pred, int *s_warp, int *total_out)
{
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned m = __ballot_sync(0xffffffffu, pred != 0);
    const int lrank = __popc(m & ((1u << lane) - 1u));
    if (lane == 0u) s_warp[warp] = __popc(m);
    __syncthreads();
    const int nw = (int)(blockDim.x >> 5);
    int off = 0, tot = 0;
    for (int w = 0; w < nw; w++) {
        const int c = s_warp[w];
        if (w < (int)warp) off += c;
        tot += c;
    }
    __syncthreads();
    *total_out = tot;
    return off + lrank;
}

static __device__ __forceinline__ ulonglong2 ff200k_ldcg_u2(const ulonglong2 *p)
{
    ulonglong2 r;
    asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%2];"
                 : "=l"(r.x), "=l"(r.y)
                 : "l"(p)
                 : "memory");
    return r;
}

#define FF200K_PC_DECL \
    unsigned long long pc0 = 0ULL, pc1 = 0ULL, pc2 = 0ULL, pc3 = 0ULL, pc4 = 0ULL, \
                       pc5 = 0ULL, pc6 = 0ULL, pc7 = 0ULL, pc8 = 0ULL;

#define FF200K_PC_INC3(flags) do { \
    unsigned long long c = (flags), t; \
    t = pc0 & c; pc0 ^= c; c = t; \
    t = pc1 & c; pc1 ^= c; c = t; \
    pc2 ^= c; \
} while (0)

#define FF200K_PC_INC4(flags) do { \
    unsigned long long c = (flags), t; \
    t = pc0 & c; pc0 ^= c; c = t; \
    t = pc1 & c; pc1 ^= c; c = t; \
    t = pc2 & c; pc2 ^= c; c = t; \
    pc3 ^= c; \
} while (0)

#define FF200K_PC_ADD_SHFL4(off) do { \
    unsigned long long b_, cy_, t1_, t2_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc0, (off)); \
    t1_ = pc0 ^ b_; cy_ = pc0 & b_; pc0 = t1_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc1, (off)); \
    t1_ = pc1 ^ b_; t2_ = pc1 & b_; pc1 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc2, (off)); \
    t1_ = pc2 ^ b_; t2_ = pc2 & b_; pc2 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc3, (off)); \
    t1_ = pc3 ^ b_; pc3 = t1_ ^ cy_; \
} while (0)

#define FF200K_PC_ADD_SHFL5(off) do { \
    unsigned long long b_, cy_, t1_, t2_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc0, (off)); \
    t1_ = pc0 ^ b_; cy_ = pc0 & b_; pc0 = t1_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc1, (off)); \
    t1_ = pc1 ^ b_; t2_ = pc1 & b_; pc1 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc2, (off)); \
    t1_ = pc2 ^ b_; t2_ = pc2 & b_; pc2 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc3, (off)); \
    t1_ = pc3 ^ b_; t2_ = pc3 & b_; pc3 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc4, (off)); \
    t1_ = pc4 ^ b_; pc4 = t1_ ^ cy_; \
} while (0)

#define FF200K_PC_ADD_SHFL6(off) do { \
    unsigned long long b_, cy_, t1_, t2_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc0, (off)); \
    t1_ = pc0 ^ b_; cy_ = pc0 & b_; pc0 = t1_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc1, (off)); \
    t1_ = pc1 ^ b_; t2_ = pc1 & b_; pc1 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc2, (off)); \
    t1_ = pc2 ^ b_; t2_ = pc2 & b_; pc2 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc3, (off)); \
    t1_ = pc3 ^ b_; t2_ = pc3 & b_; pc3 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc4, (off)); \
    t1_ = pc4 ^ b_; t2_ = pc4 & b_; pc4 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc5, (off)); \
    t1_ = pc5 ^ b_; pc5 = t1_ ^ cy_; \
} while (0)

#define FF200K_PC_ADD_SHFL7(off) do { \
    unsigned long long b_, cy_, t1_, t2_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc0, (off)); \
    t1_ = pc0 ^ b_; cy_ = pc0 & b_; pc0 = t1_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc1, (off)); \
    t1_ = pc1 ^ b_; t2_ = pc1 & b_; pc1 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc2, (off)); \
    t1_ = pc2 ^ b_; t2_ = pc2 & b_; pc2 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc3, (off)); \
    t1_ = pc3 ^ b_; t2_ = pc3 & b_; pc3 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc4, (off)); \
    t1_ = pc4 ^ b_; t2_ = pc4 & b_; pc4 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc5, (off)); \
    t1_ = pc5 ^ b_; t2_ = pc5 & b_; pc5 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc6, (off)); \
    t1_ = pc6 ^ b_; pc6 = t1_ ^ cy_; \
} while (0)

#define FF200K_PC_ADD_SHFL9(off) do { \
    unsigned long long b_, cy_, t1_, t2_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc0, (off)); \
    t1_ = pc0 ^ b_; cy_ = pc0 & b_; pc0 = t1_; \
    b_ = __shfl_xor_sync(0xffffffffu, pc1, (off)); \
    t1_ = pc1 ^ b_; t2_ = pc1 & b_; pc1 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc2, (off)); \
    t1_ = pc2 ^ b_; t2_ = pc2 & b_; pc2 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc3, (off)); \
    t1_ = pc3 ^ b_; t2_ = pc3 & b_; pc3 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc4, (off)); \
    t1_ = pc4 ^ b_; t2_ = pc4 & b_; pc4 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc5, (off)); \
    t1_ = pc5 ^ b_; t2_ = pc5 & b_; pc5 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc6, (off)); \
    t1_ = pc6 ^ b_; t2_ = pc6 & b_; pc6 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc7, (off)); \
    t1_ = pc7 ^ b_; t2_ = pc7 & b_; pc7 = t1_ ^ cy_; cy_ = t2_ | (t1_ & cy_); \
    b_ = __shfl_xor_sync(0xffffffffu, pc8, (off)); \
    t1_ = pc8 ^ b_; pc8 = t1_ ^ cy_; \
} while (0)

#define FF200K_PC_GET3(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) )

#define FF200K_PC_GET4(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) )

#define FF200K_PC_GET5(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) )

#define FF200K_PC_GET6(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) )

#define FF200K_PC_GET7(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((pc6 >> (tp)) & 1ULL) << 6) )

#define FF200K_PC_GET9(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((pc6 >> (tp)) & 1ULL) << 6) | \
    ((int)((pc7 >> (tp)) & 1ULL) << 7) | \
    ((int)((pc8 >> (tp)) & 1ULL) << 8) )

#define FF200K_FE_STEP(off) do {                                                   \
    const unsigned long long oa_ = __shfl_xor_sync(0xffffffffu, a, (off));     \
    const unsigned long long od_ = __shfl_xor_sync(0xffffffffu, d, (off));     \
    d = d | od_ | (a & oa_);                                                   \
    a = a | oa_;                                                               \
} while (0)

#define FF200K_FE_RED_1
#define FF200K_FE_RED_2   FF200K_FE_STEP(1);
#define FF200K_FE_RED_4   FF200K_FE_STEP(2); FF200K_FE_STEP(1);
#define FF200K_FE_RED_8   FF200K_FE_STEP(4); FF200K_FE_STEP(2); FF200K_FE_STEP(1);
#define FF200K_FE_RED_16  FF200K_FE_STEP(8); FF200K_FE_STEP(4); FF200K_FE_STEP(2); FF200K_FE_STEP(1);
#define FF200K_FE_RED_32  FF200K_FE_STEP(16); FF200K_FE_STEP(8); FF200K_FE_STEP(4); FF200K_FE_STEP(2); FF200K_FE_STEP(1);

#define FF200K_MV_STEP(ADDM, off) do {                                             \
    cand_mask |= __shfl_xor_sync(0xffffffffu, cand_mask, (off));               \
    count_current_present +=                                                   \
        __shfl_xor_sync(0xffffffffu, count_current_present, (off));            \
    crit += __shfl_xor_sync(0xffffffffu, crit, (off));                         \
    ADDM(off);                                                                 \
} while (0)

#define FF200K_MV_RED_1
#define FF200K_MV_RED_2   FF200K_MV_STEP(FF200K_PC_ADD_SHFL4, 1);
#define FF200K_MV_RED_4   FF200K_MV_STEP(FF200K_PC_ADD_SHFL5, 2); FF200K_MV_STEP(FF200K_PC_ADD_SHFL5, 1);
#define FF200K_MV_RED_8   FF200K_MV_STEP(FF200K_PC_ADD_SHFL6, 4); FF200K_MV_STEP(FF200K_PC_ADD_SHFL6, 2); \
                      FF200K_MV_STEP(FF200K_PC_ADD_SHFL6, 1);
#define FF200K_MV_RED_16  FF200K_MV_STEP(FF200K_PC_ADD_SHFL7, 8); FF200K_MV_STEP(FF200K_PC_ADD_SHFL7, 4); \
                      FF200K_MV_STEP(FF200K_PC_ADD_SHFL7, 2); FF200K_MV_STEP(FF200K_PC_ADD_SHFL7, 1);
#define FF200K_MV_RED_32  FF200K_MV_STEP(FF200K_PC_ADD_SHFL9, 16); FF200K_MV_STEP(FF200K_PC_ADD_SHFL9, 8); \
                      FF200K_MV_STEP(FF200K_PC_ADD_SHFL9, 4); FF200K_MV_STEP(FF200K_PC_ADD_SHFL9, 2); \
                      FF200K_MV_STEP(FF200K_PC_ADD_SHFL9, 1);
#define FF200K_GIANT_BATCH 16

#define FF200K_FE_ACC(P) do {                                                      \
    const int part_ = (P);                                                     \
    if ((unsigned)part_ < 64u) {                                               \
        const unsigned long long bit_ = 1ULL << part_;                         \
        d |= (a & bit_);                                                       \
        a |= bit_;                                                             \
    }                                                                          \
} while (0)

#define FF200K_MV_PIN(FP, INCM) do {                                               \
    const unsigned long long fa_ = (FP).x;                                     \
    const unsigned long long fd_ = (FP).y;                                     \
    const unsigned long long complete_ = num_parts >= 64 ? ~0ULL : ((1ULL << num_parts) - 1ULL); \
    if (num_parts <= 64 && fa_ == complete_ && (fd_ & current_bit) != 0ULL) { \
        cand_mask |= fa_ & ~current_bit; \
    } else { \
    if ((fa_ & current_bit) != 0ULL &&                                         \
        (fd_ & current_bit) == 0ULL) {                                         \
        const int parts_ = __popcll(fa_);                                      \
        if (parts_ > 1) crit += (parts_ - 1);                                  \
    }                                                                          \
    const unsigned long long mask_ =                                           \
        (fa_ & ~current_bit) | (fd_ & current_bit);                            \
    if (mask_ & current_bit) count_current_present++;                          \
    const unsigned long long fl_ = mask_ & ~current_bit;                       \
    cand_mask |= fl_;                                                          \
    INCM(fl_);                                                                 \
    } \
} while (0)

#define FF200K_PART_LOAD(n) __ldg(&partition[(n)])
#define FF200K_FE_BODY(G, SH, RED)                                                 \
{                                                                              \
    const int lg_ = (int)(lane & (unsigned)((G) - 1));                         \
    const int hedge = im_.x;                                                   \
    const int hstart = im_.y;                                                  \
    const int hsz = im_.z;                                                     \
    unsigned long long a = 0ULL, d = 0ULL;                                     \
    for (int k = lg_; k < hsz; k += 4 * (G)) {                                 \
        const int ka_ = k + (G), kb_ = k + 2 * (G), kc_ = k + 3 * (G);         \
        const bool va_ = (ka_ < hsz);                                          \
        const bool vb_ = (kb_ < hsz);                                          \
        const bool vc_ = (kc_ < hsz);                                          \
        const int n0_ = __ldg(&hyperedge_nodes[hstart + k]);                   \
        const int na_ = va_ ? __ldg(&hyperedge_nodes[hstart + ka_]) : -1;      \
        const int nb_ = vb_ ? __ldg(&hyperedge_nodes[hstart + kb_]) : -1;      \
        const int nc_ = vc_ ? __ldg(&hyperedge_nodes[hstart + kc_]) : -1;      \
        const int p0_ = ((unsigned)n0_ < (unsigned)num_nodes)                  \
                            ? FF200K_PART_LOAD(n0_) : -1;                     \
        const int pa_ = ((unsigned)na_ < (unsigned)num_nodes)                  \
                            ? FF200K_PART_LOAD(na_) : -1;                     \
        const int pb_ = ((unsigned)nb_ < (unsigned)num_nodes)                  \
                            ? FF200K_PART_LOAD(nb_) : -1;                     \
        const int pc_ = ((unsigned)nc_ < (unsigned)num_nodes)                  \
                            ? FF200K_PART_LOAD(nc_) : -1;                     \
        FF200K_FE_ACC(p0_);                                                        \
        FF200K_FE_ACC(pa_);                                                        \
        FF200K_FE_ACC(pb_);                                                        \
        FF200K_FE_ACC(pc_);                                                        \
    }                                                                          \
    RED                                                                        \
    if (lg_ == 0) {                                                            \
        ulonglong2 v_; v_.x = a; v_.y = d;                                     \
        edge_flags_pair[hedge] = v_;                                           \
    }                                                                          \
}

#define FF200K_MV_CSR(G) do {                                                      \
    const int j1_ = j_ + (G), j2_ = j_ + 2 * (G), j3_ = j_ + 3 * (G);          \
    v1_ = (j1_ < used_degree);                                                 \
    v2_ = (j2_ < used_degree);                                                 \
    v3_ = (j3_ < used_degree);                                                 \
    const int r0_ = capped_ ? (int)(((long long)j_  * node_degree) >> 8) : j_;  \
    const int r1_ = capped_ ? (int)(((long long)j1_ * node_degree) >> 8) : j1_; \
    const int r2_ = capped_ ? (int)(((long long)j2_ * node_degree) >> 8) : j2_; \
    const int r3_ = capped_ ? (int)(((long long)j3_ * node_degree) >> 8) : j3_; \
    h0_ = __ldg(&node_hyperedges[nstart + r0_]);                               \
    h1_ = v1_ ? __ldg(&node_hyperedges[nstart + r1_]) : 0;                     \
    h2_ = v2_ ? __ldg(&node_hyperedges[nstart + r2_]) : 0;                     \
    h3_ = v3_ ? __ldg(&node_hyperedges[nstart + r3_]) : 0;                     \
} while (0)


static __device__ __forceinline__ void exact_masks_200k(
    const int np, const int cap, const int *sizes,
    unsigned long long *free_mask, unsigned long long *bonus_masks)
{
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    if (warp == 0) {
        unsigned lo = __ballot_sync(0xffffffffu, lane < np && sizes[lane] < cap);
        unsigned hi = __ballot_sync(0xffffffffu, lane + 32 < np && sizes[lane + 32] < cap);
        if (lane == 0) *free_mask = (unsigned long long)lo | ((unsigned long long)hi << 32);
    }
    for (int c = warp; c < 64; c += nw) {
        int cs = c < np ? sizes[c] : 0;
        unsigned lo = __ballot_sync(0xffffffffu, lane < np && cs > sizes[lane] + 2);
        unsigned hi = __ballot_sync(0xffffffffu, lane + 32 < np && cs > sizes[lane + 32] + 2);
        if (lane == 0) bonus_masks[c] = (unsigned long long)lo | ((unsigned long long)hi << 32);
    }
    __syncthreads();
}

#define FF200K_MV_BODY(G, SH, INCM, GETM, RED)                                     \
{                                                                              \
    const int lg_ = (int)(lane & (unsigned)((G) - 1));                         \
    const int node = im_.x;                                                    \
    const int nstart = im_.y;                                                  \
    const int node_degree = im_.z;                                             \
    int outkey = FF200K_SENT;                                                      \
    const int used_degree = node_degree > 256 ? 256 : node_degree;             \
    const bool capped_ = (node_degree > 256);                                  \
    const bool go_ = (node_degree > 0) && (lg_ < used_degree);                 \
    int j_ = lg_;                                                              \
    int h0_ = 0, h1_ = 0, h2_ = 0, h3_ = 0;                                    \
    bool v1_ = false, v2_ = false, v3_ = false;                                \
    if (go_) { FF200K_MV_CSR(G); }                                                 \
    const int current_part = FF200K_PART_LOAD(node);                          \
    const bool warp_node = (unsigned)current_part < (unsigned)num_parts &&     \
        shared_nodes_in_part[current_part] > 1 &&                              \
        node_degree > 0;                                                       \
    const unsigned long long current_bit = warp_node ? (1ULL << current_part) : 0ULL; \
    FF200K_PC_DECL                                                                \
    unsigned long long cand_mask = 0ULL;                                       \
    int count_current_present = 0;                                             \
    int crit = 0;                                                              \
    if (warp_node && go_) {                                                    \
            for (;;) {                                                         \
                ulonglong2 f1_; f1_.x = 0ULL; f1_.y = 0ULL;                    \
                ulonglong2 f2_; f2_.x = 0ULL; f2_.y = 0ULL;                    \
                ulonglong2 f3_; f3_.x = 0ULL; f3_.y = 0ULL;                    \
                const ulonglong2 f0_ = ff200k_ldcg_u2(&edge_flags_pair[h0_]);      \
                if (v1_) f1_ = ff200k_ldcg_u2(&edge_flags_pair[h1_]);              \
                if (v2_) f2_ = ff200k_ldcg_u2(&edge_flags_pair[h2_]);              \
                if (v3_) f3_ = ff200k_ldcg_u2(&edge_flags_pair[h3_]);              \
                FF200K_MV_PIN(f0_, INCM);                                          \
                if (v1_) { FF200K_MV_PIN(f1_, INCM); }                             \
                if (v2_) { FF200K_MV_PIN(f2_, INCM); }                             \
                if (v3_) { FF200K_MV_PIN(f3_, INCM); }                             \
                j_ += 4 * (G);                                                 \
                if (j_ >= used_degree) break;                                  \
                FF200K_MV_CSR(G);                                                  \
            }                                                                  \
        }                                                                      \
        RED                                                                    \
        if (crit > 255) crit = 255;                                            \
        int degree_weight = node_degree > 255 ? 255 : node_degree;             \
        int rank_byte = degree_weight + crit;                                  \
        if (rank_byte > 255) rank_byte = 255;                                  \
        int best_gain = -999999;                                               \
        int best_target = current_part;                                        \
        unsigned long long active_ = cand_mask & exact_free; \
        if (active_) { \
            unsigned long long cy_ = exact_bonus[current_part], tmp_; \
            tmp_ = pc0 & cy_; pc0 ^= cy_; cy_ = tmp_; \
            tmp_ = pc1 & cy_; pc1 ^= cy_; cy_ = tmp_; \
            tmp_ = pc2 & cy_; pc2 ^= cy_; cy_ = tmp_; \
            tmp_ = pc3 & cy_; pc3 ^= cy_; cy_ = tmp_; \
            tmp_ = pc4 & cy_; pc4 ^= cy_; cy_ = tmp_; \
            tmp_ = pc5 & cy_; pc5 ^= cy_; cy_ = tmp_; \
            tmp_ = pc6 & cy_; pc6 ^= cy_; cy_ = tmp_; \
            tmp_ = pc7 & cy_; pc7 ^= cy_; cy_ = tmp_; \
            tmp_ = pc8 & cy_; pc8 ^= cy_; cy_ = tmp_; \
            best_gain = 0; \
            { unsigned long long m_ = active_ & pc8; \
              if (m_) { active_ = m_; best_gain |= 256; } } \
            { unsigned long long m_ = active_ & pc7; \
              if (m_) { active_ = m_; best_gain |= 128; } } \
            { unsigned long long m_ = active_ & pc6; \
              if (m_) { active_ = m_; best_gain |= 64; } } \
            { unsigned long long m_ = active_ & pc5; \
              if (m_) { active_ = m_; best_gain |= 32; } } \
            { unsigned long long m_ = active_ & pc4; \
              if (m_) { active_ = m_; best_gain |= 16; } } \
            { unsigned long long m_ = active_ & pc3; \
              if (m_) { active_ = m_; best_gain |= 8; } } \
            { unsigned long long m_ = active_ & pc2; \
              if (m_) { active_ = m_; best_gain |= 4; } } \
            { unsigned long long m_ = active_ & pc1; \
              if (m_) { active_ = m_; best_gain |= 2; } } \
            { unsigned long long m_ = active_ & pc0; \
              if (m_) { active_ = m_; best_gain |= 1; } } \
            best_target = __ffsll(active_) - 1; \
            best_gain -= count_current_present; \
            if (capped_) { \
                const int bonus_ = (int)((exact_bonus[current_part] >> best_target) & 1ULL); \
                const int conn_ = best_gain - bonus_; \
                best_gain = (int)(((long long)conn_ * node_degree) / 256) + bonus_; \
            } \
        } \
        if (best_target != current_part) {                                     \
            int bg = best_gain > 32767 ? 32767 : best_gain;                    \
            if (bg < -32768) bg = -32768;                                      \
            unsigned short t16 =                                               \
                (unsigned short)(best_target & 63) | ((node & 3) << 6);        \
            outkey = ((int)(short)bg << 16) | (rank_byte << 8) | t16;          \
        }                                                                      \
    if (lg_ == 0) {                                                            \
        move_priorities[node] = outkey;                                        \
        if (outkey != FF200K_SENT) {                                               \
            const int g_ = outkey >> 16;                                       \
            if (g_ > blk_max) blk_max = g_;                                    \
        }                                                                      \
    }                                                                          \
}

#if 0
extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_filter_200k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *__restrict__ hyperedge_nodes,
    const int *__restrict__ node_hyperedges,
    const int *__restrict__ partition,
    const int *__restrict__ nodes_in_part,
    const int4 *__restrict__ fe_slots,
    const int n_fe_tasks,
    const int4 *__restrict__ fe_giant,
    const int n_fe_giant,
    const int4 *__restrict__ mv_slots,
    const int n_mv_tasks,
    ulonglong2 *edge_flags_pair,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int barrier_base,
    int *tabu_until,
    const int *__restrict__ mark_nodes,
    const int n_mark_a,
    const int until_a,
    const int n_mark_b,
    const int until_b,
    const int round_idx,
    const unsigned long long lcg_x0,
    const unsigned long long prob_num,
    const unsigned long long prob_den_base,
    int *max_gain_buf,
    int *blk_elig,
    int *tile_valid,
    const int ntiles,
    int *cand
) {
    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64];
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[32];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_200k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

    int4 fim_; fim_.x = 0; fim_.y = 0; fim_.z = 0; fim_.w = 0;
    if (warp_global < n_fe_tasks) {
        fim_ = __ldg(&fe_slots[warp_global * 32 + (int)lane]);
    }
    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }
    const int mark0_ = (gtid < n_mark_a) ? mark_nodes[gtid] : -1;

    if (gtid == 0) {
        max_gain_buf[0] = FF200K_MG_INIT;
    }

    for (int i = (int)blockIdx.x; i < n_fe_giant; i += (int)gridDim.x) {
        const int4 im = __ldg(&fe_giant[i]);
        const int hedge = im.x;
        const int hstart = im.y;
        const int hsz = im.z;
        unsigned long long a = 0ULL, d = 0ULL;
        for (int k0 = (int)threadIdx.x; k0 < hsz;
             k0 += (int)blockDim.x * FF200K_GIANT_BATCH) {
            int gn_[FF200K_GIANT_BATCH];
            int gp_[FF200K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                FF200K_FE_ACC(gp_[m]);
            }
        }
        FF200K_FE_RED_32
        if (lane == 0u) {
            s_ab[(threadIdx.x >> 5) * 2 + 0] = a;
            s_ab[(threadIdx.x >> 5) * 2 + 1] = d;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long A = s_ab[0], D = s_ab[1];
            for (int w = 1; w < warps_per_block; w++) {
                const unsigned long long oa = s_ab[w * 2 + 0];
                const unsigned long long od = s_ab[w * 2 + 1];
                D = D | od | (A & oa);
                A = A | oa;
            }
            ulonglong2 v; v.x = A; v.y = D;
            edge_flags_pair[hedge] = v;
        }
        __syncthreads();
    }

    if ((unsigned)mark0_ < (unsigned)num_nodes) tabu_until[mark0_] = until_a;
    for (int i = gtid + stride; i < n_mark_a; i += stride) {
        const int n = mark_nodes[i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_a;
    }

    for (int t = warp_global; t < n_fe_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_fe_tasks) {
            nx_ = __ldg(&fe_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = fim_;
        switch (im_.w) {
            case 0:  FF200K_FE_BODY(1,  0, FF200K_FE_RED_1)  break;
            case 1:  FF200K_FE_BODY(2,  1, FF200K_FE_RED_2)  break;
            case 2:  FF200K_FE_BODY(4,  2, FF200K_FE_RED_4)  break;
            case 3:  FF200K_FE_BODY(8,  3, FF200K_FE_RED_8)  break;
            case 4:  FF200K_FE_BODY(16, 4, FF200K_FE_RED_16) break;
            default: FF200K_FE_BODY(32, 5, FF200K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF200K_MG_INIT;

    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF200K_MV_BODY(1,  0, FF200K_PC_INC3, FF200K_PC_GET3, FF200K_MV_RED_1)  break;
            case 1:  FF200K_MV_BODY(2,  1, FF200K_PC_INC3, FF200K_PC_GET4, FF200K_MV_RED_2)  break;
            case 2:  FF200K_MV_BODY(4,  2, FF200K_PC_INC3, FF200K_PC_GET5, FF200K_MV_RED_4)  break;
            case 3:  FF200K_MV_BODY(8,  3, FF200K_PC_INC3, FF200K_PC_GET6, FF200K_MV_RED_8)  break;
            case 4:  FF200K_MV_BODY(16, 4, FF200K_PC_INC3, FF200K_PC_GET7, FF200K_MV_RED_16) break;
            default: FF200K_MV_BODY(32, 5, FF200K_PC_INC4, FF200K_PC_GET9, FF200K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff200k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    const int mg = __ldcg(max_gain_buf);
    const int aspiration = (mg * 3) / 4;
    const int TILE = (int)blockDim.x;

    const int nrounds = (ntiles + (int)gridDim.x - 1) / (int)gridDim.x;
    unsigned int btgt_ = barrier_base + 2u * gridDim.x;

    for (int it = 0; it < nrounds; it++) {
        const int tile = (int)blockIdx.x + it * (int)gridDim.x;
        const bool act = (tile < ntiles);
        const int node = act ? (tile * TILE + (int)threadIdx.x) : num_nodes;
        const int plim = act ? tile : 0;

        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;

        int key = 0, gain = 0, elig = 0, posval = 0;
        if (node < num_nodes) {
            key = key_ld;
            if (key != FF200K_SENT) {
                gain = key >> 16;
                const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
                if (!is_tabu) {
                    if (gain > 0) posval = 1;
                    else if (gain >= -3) elig = 1;
                }
            }
        }

        int etot = 0;
        const int erank = ff200k_block_scan(elig, s_warp, &etot);
        if (act && threadIdx.x == 0) blk_elig[tile] = etot;

        btgt_ += (unsigned int)gridDim.x;
        FF200K_GRID_BARRIER(grid_barrier, btgt_);

        int partsum = 0;
        for (int i = (int)threadIdx.x; i < plim; i += (int)blockDim.x) {
            partsum += __ldcg(&blk_elig[i]);
        }
        const int baseE = ff200k_block_sum(partsum, s_warp);

        int outkey = key;
        int accept = 0;
        if (elig) {
            const unsigned long long n = (unsigned long long)(baseE + erank) + 1ULL;
            const unsigned long long x = ff200k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key & 0xFFFF;
            }
        }

        const int valid = (posval | accept);

        int vtot = 0;
        const int vrank = ff200k_block_scan(valid, s_warp, &vtot);
        if (act && threadIdx.x == 0) tile_valid[tile] = vtot;

        btgt_ += (unsigned int)gridDim.x;
        FF200K_GRID_BARRIER(grid_barrier, btgt_);

        int partsum2 = 0;
        for (int i = (int)threadIdx.x; i < plim; i += (int)blockDim.x) {
            partsum2 += __ldcg(&tile_valid[i]);
        }
        const int base = ff200k_block_sum(partsum2, s_warp);

        if (valid) {
            const int slot = base + vrank;
            cand[1 + 2 * slot] = node;
            cand[2 + 2 * slot] = outkey;
        }
    }

    if (blockIdx.x == 0) {
        int s = 0;
        for (int i = (int)threadIdx.x; i < ntiles; i += (int)blockDim.x) {
            s += __ldcg(&tile_valid[i]);
        }
        const int t = ff200k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = t;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_filter_flat_200k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *__restrict__ hyperedge_nodes,
    const int *__restrict__ node_hyperedges,
    const int *__restrict__ partition,
    const int *__restrict__ nodes_in_part,
    const int4 *__restrict__ fe_slots,
    const int n_fe_tasks,
    const int4 *__restrict__ fe_giant,
    const int n_fe_giant,
    const int4 *__restrict__ mv_slots,
    const int n_mv_tasks,
    ulonglong2 *edge_flags_pair,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int barrier_base,
    int *tabu_until,
    const int *__restrict__ mark_nodes,
    const int n_mark_a,
    const int until_a,
    const int n_mark_b,
    const int until_b,
    const int round_idx,
    const unsigned long long lcg_x0,
    const unsigned long long prob_num,
    const unsigned long long prob_den_base,
    int *max_gain_buf,
    int *blk_elig,
    int *tile_valid,
    const int ntiles,
    int *cand
) {
    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64];
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[32];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_200k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

    int4 fim_; fim_.x = 0; fim_.y = 0; fim_.z = 0; fim_.w = 0;
    if (warp_global < n_fe_tasks) {
        fim_ = __ldg(&fe_slots[warp_global * 32 + (int)lane]);
    }
    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }
    const int mark0_ = (gtid < n_mark_a) ? mark_nodes[gtid] : -1;

    if (gtid == 0) {
        max_gain_buf[0] = FF200K_MG_INIT;
    }

    for (int i = (int)blockIdx.x; i < n_fe_giant; i += (int)gridDim.x) {
        const int4 im = __ldg(&fe_giant[i]);
        const int hedge = im.x;
        const int hstart = im.y;
        const int hsz = im.z;
        unsigned long long a = 0ULL, d = 0ULL;
        for (int k0 = (int)threadIdx.x; k0 < hsz;
             k0 += (int)blockDim.x * FF200K_GIANT_BATCH) {
            int gn_[FF200K_GIANT_BATCH];
            int gp_[FF200K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                FF200K_FE_ACC(gp_[m]);
            }
        }
        FF200K_FE_RED_32
        if (lane == 0u) {
            s_ab[(threadIdx.x >> 5) * 2 + 0] = a;
            s_ab[(threadIdx.x >> 5) * 2 + 1] = d;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long A = s_ab[0], D = s_ab[1];
            for (int w = 1; w < warps_per_block; w++) {
                const unsigned long long oa = s_ab[w * 2 + 0];
                const unsigned long long od = s_ab[w * 2 + 1];
                D = D | od | (A & oa);
                A = A | oa;
            }
            ulonglong2 v; v.x = A; v.y = D;
            edge_flags_pair[hedge] = v;
        }
        __syncthreads();
    }

    if ((unsigned)mark0_ < (unsigned)num_nodes) tabu_until[mark0_] = until_a;
    for (int i = gtid + stride; i < n_mark_a; i += stride) {
        const int n = mark_nodes[i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_a;
    }

    for (int t = warp_global; t < n_fe_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_fe_tasks) {
            nx_ = __ldg(&fe_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = fim_;
        switch (im_.w) {
            case 0:  FF200K_FE_BODY(1,  0, FF200K_FE_RED_1)  break;
            case 1:  FF200K_FE_BODY(2,  1, FF200K_FE_RED_2)  break;
            case 2:  FF200K_FE_BODY(4,  2, FF200K_FE_RED_4)  break;
            case 3:  FF200K_FE_BODY(8,  3, FF200K_FE_RED_8)  break;
            case 4:  FF200K_FE_BODY(16, 4, FF200K_FE_RED_16) break;
            default: FF200K_FE_BODY(32, 5, FF200K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF200K_MG_INIT;

    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF200K_MV_BODY(1,  0, FF200K_PC_INC3, FF200K_PC_GET3, FF200K_MV_RED_1)  break;
            case 1:  FF200K_MV_BODY(2,  1, FF200K_PC_INC3, FF200K_PC_GET4, FF200K_MV_RED_2)  break;
            case 2:  FF200K_MV_BODY(4,  2, FF200K_PC_INC3, FF200K_PC_GET5, FF200K_MV_RED_4)  break;
            case 3:  FF200K_MV_BODY(8,  3, FF200K_PC_INC3, FF200K_PC_GET6, FF200K_MV_RED_8)  break;
            case 4:  FF200K_MV_BODY(16, 4, FF200K_PC_INC3, FF200K_PC_GET7, FF200K_MV_RED_16) break;
            default: FF200K_MV_BODY(32, 5, FF200K_PC_INC4, FF200K_PC_GET9, FF200K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff200k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    const int mg = __ldcg(max_gain_buf);
    const int aspiration = (mg * 3) / 4;
    const int TILE = (int)blockDim.x;    
    const int tpb = (ntiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int t0 = (int)blockIdx.x * tpb;
    int t1_ = t0 + tpb;
    if (t1_ > ntiles) t1_ = ntiles;
    if (t1_ < t0) t1_ = t0;
    const int t1 = t1_;

    int blkE = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int elig = 0;
        if (node < num_nodes && key_ld != FF200K_SENT) {
            const int gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu && gain <= 0 && gain >= -3) elig = 1;
        }
        blkE += ff200k_block_sum(elig, s_warp);
    }
    if (threadIdx.x == 0) blk_elig[blockIdx.x] = blkE;

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 3u * gridDim.x);

    int psE_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psE_ += __ldcg(&blk_elig[i]);
    }
    const int baseE_block = ff200k_block_sum(psE_, s_warp);

    int runE = 0;
    int blkV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF200K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff200k_block_scan(elig, s_warp, &etot);
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE + erank) + 1ULL;
            const unsigned long long x = ff200k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) accept = 1;
        }
        const int valid = (posval | accept);
        int vtot = 0;
        (void)ff200k_block_scan(valid, s_warp, &vtot);
        runE += etot;
        blkV += vtot;
    }
    if (threadIdx.x == 0) tile_valid[blockIdx.x] = blkV;

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 4u * gridDim.x);

    int psV_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psV_ += __ldcg(&tile_valid[i]);
    }
    const int baseV_block = ff200k_block_sum(psV_, s_warp);

    int runE2 = 0;
    int runV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF200K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff200k_block_scan(elig, s_warp, &etot);
        int outkey = key_ld;
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE2 + erank) + 1ULL;
            const unsigned long long x = ff200k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key_ld & 0xFFFF;
            }
        }
        const int valid = (posval | accept);
        int vtot = 0;
        const int vrank = ff200k_block_scan(valid, s_warp, &vtot);
        if (valid) {
            const int slot = baseV_block + runV + vrank;
            cand[1 + 2 * slot] = node;
            cand[2 + 2 * slot] = outkey;
        }
        runE2 += etot;
        runV += vtot;
    }

    if (blockIdx.x == 0) {
        int s = 0;
        for (int i = (int)threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
            s += __ldcg(&tile_valid[i]);
        }
        const int tot = ff200k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = tot;
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void fused_flags_moves_filter_flatw_200k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *__restrict__ hyperedge_nodes,
    const int *__restrict__ node_hyperedges,
    const int *__restrict__ partition,
    const int *__restrict__ nodes_in_part,
    const int4 *__restrict__ fe_slots,
    const int n_fe_tasks,
    const int4 *__restrict__ fe_giant,
    const int n_fe_giant,
    const int4 *__restrict__ mv_slots,
    const int n_mv_tasks,
    ulonglong2 *edge_flags_pair,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int barrier_base,
    int *tabu_until,
    const int *__restrict__ mark_nodes,
    const int n_mark_a,
    const int until_a,
    const int n_mark_b,
    const int until_b,
    const int round_idx,
    const unsigned long long lcg_x0,
    const unsigned long long prob_num,
    const unsigned long long prob_den_base,
    int *max_gain_buf,
    int *blk_elig,
    int *tile_valid,
    const int ntiles,
    int *cand
) {
    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64];
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[32];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_200k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

    int4 fim_; fim_.x = 0; fim_.y = 0; fim_.z = 0; fim_.w = 0;
    if (warp_global < n_fe_tasks) {
        fim_ = __ldg(&fe_slots[warp_global * 32 + (int)lane]);
    }
    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }
    const int mark0_ = (gtid < n_mark_a) ? mark_nodes[gtid] : -1;

    if (gtid == 0) {
        max_gain_buf[0] = FF200K_MG_INIT;
    }

    for (int i = (int)blockIdx.x; i < n_fe_giant; i += (int)gridDim.x) {
        const int4 im = __ldg(&fe_giant[i]);
        const int hedge = im.x;
        const int hstart = im.y;
        const int hsz = im.z;
        unsigned long long a = 0ULL, d = 0ULL;
        for (int k0 = (int)threadIdx.x; k0 < hsz;
             k0 += (int)blockDim.x * FF200K_GIANT_BATCH) {
            int gn_[FF200K_GIANT_BATCH];
            int gp_[FF200K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF200K_GIANT_BATCH; m++) {
                FF200K_FE_ACC(gp_[m]);
            }
        }
        FF200K_FE_RED_32
        if (lane == 0u) {
            s_ab[(threadIdx.x >> 5) * 2 + 0] = a;
            s_ab[(threadIdx.x >> 5) * 2 + 1] = d;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned long long A = s_ab[0], D = s_ab[1];
            for (int w = 1; w < warps_per_block; w++) {
                const unsigned long long oa = s_ab[w * 2 + 0];
                const unsigned long long od = s_ab[w * 2 + 1];
                D = D | od | (A & oa);
                A = A | oa;
            }
            ulonglong2 v; v.x = A; v.y = D;
            edge_flags_pair[hedge] = v;
        }
        __syncthreads();
    }

    if ((unsigned)mark0_ < (unsigned)num_nodes) tabu_until[mark0_] = until_a;
    for (int i = gtid + stride; i < n_mark_a; i += stride) {
        const int n = mark_nodes[i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_a;
    }

    for (int t = warp_global; t < n_fe_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_fe_tasks) {
            nx_ = __ldg(&fe_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = fim_;
        switch (im_.w) {
            case 0:  FF200K_FE_BODY(1,  0, FF200K_FE_RED_1)  break;
            case 1:  FF200K_FE_BODY(2,  1, FF200K_FE_RED_2)  break;
            case 2:  FF200K_FE_BODY(4,  2, FF200K_FE_RED_4)  break;
            case 3:  FF200K_FE_BODY(8,  3, FF200K_FE_RED_8)  break;
            case 4:  FF200K_FE_BODY(16, 4, FF200K_FE_RED_16) break;
            default: FF200K_FE_BODY(32, 5, FF200K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF200K_MG_INIT;

    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF200K_MV_BODY(1,  0, FF200K_PC_INC3, FF200K_PC_GET3, FF200K_MV_RED_1)  break;
            case 1:  FF200K_MV_BODY(2,  1, FF200K_PC_INC3, FF200K_PC_GET4, FF200K_MV_RED_2)  break;
            case 2:  FF200K_MV_BODY(4,  2, FF200K_PC_INC3, FF200K_PC_GET5, FF200K_MV_RED_4)  break;
            case 3:  FF200K_MV_BODY(8,  3, FF200K_PC_INC3, FF200K_PC_GET6, FF200K_MV_RED_8)  break;
            case 4:  FF200K_MV_BODY(16, 4, FF200K_PC_INC3, FF200K_PC_GET7, FF200K_MV_RED_16) break;
            default: FF200K_MV_BODY(32, 5, FF200K_PC_INC4, FF200K_PC_GET9, FF200K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff200k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    const int mg = __ldcg(max_gain_buf);
    const int aspiration = (mg * 3) / 4;
    const int TILE = (int)blockDim.x;
    const int tpb = (ntiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int t0 = (int)blockIdx.x * tpb;
    int t1_ = t0 + tpb;
    if (t1_ > ntiles) t1_ = ntiles;
    if (t1_ < t0) t1_ = t0;
    const int t1 = t1_;

    int blkE = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int elig = 0;
        if (node < num_nodes && key_ld != FF200K_SENT) {
            const int gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu && gain <= 0 && gain >= -3) elig = 1;
        }
        blkE += ff200k_block_sum(elig, s_warp);
    }
    if (threadIdx.x == 0) blk_elig[blockIdx.x] = blkE;

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 3u * gridDim.x);

    int psE_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psE_ += __ldcg(&blk_elig[i]);
    }
    const int baseE_block = ff200k_block_sum(psE_, s_warp);

    int runE = 0;
    int blkV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF200K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff200k_block_scan(elig, s_warp, &etot);
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE + erank) + 1ULL;
            const unsigned long long x = ff200k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) accept = 1;
        }
        const int valid = (posval | accept);
        int vtot = 0;
        (void)ff200k_block_scan(valid, s_warp, &vtot);
        runE += etot;
        blkV += vtot;
    }
    if (threadIdx.x == 0) tile_valid[blockIdx.x] = blkV;

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 4u * gridDim.x);

    int psV_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psV_ += __ldcg(&tile_valid[i]);
    }
    const int baseV_block = ff200k_block_sum(psV_, s_warp);

    int runE2 = 0;
    int runV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF200K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff200k_block_scan(elig, s_warp, &etot);
        int outkey = key_ld;
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE2 + erank) + 1ULL;
            const unsigned long long x = ff200k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key_ld & 0xFFFF;
            }
        }
        const int valid = (posval | accept);
        int vtot = 0;
        const int vrank = ff200k_block_scan(valid, s_warp, &vtot);
        if (valid) {
            const int slot = baseV_block + runV + vrank;
            cand[1 + 2 * slot] = node;
            cand[2 + 2 * slot] = outkey;
        }
        runE2 += etot;
        runV += vtot;
    }

    if (blockIdx.x == 0) {
        int s = 0;
        for (int i = (int)threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
            s += __ldcg(&tile_valid[i]);
        }
        const int tot = ff200k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = tot;
    }
}

#endif
union ExactScratch_200k {
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

static __device__ __forceinline__ int exact_apply_dag_200k(
    int M, int nn, int np, int cap, int *partition, int *part_sizes,
    const int *ordered, int *tabu, int round, int tenure,
    int mark_base, int mark_mult, ExactScratch_200k *work, int *reduce)
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
    int accepted = ff200k_block_sum(local_accepted, reduce);
    if (accepted > 0) {
        int marks = min(M, max(mark_base, accepted * mark_mult));
        for (int j = tid; j < marks; j += blockDim.x) {
            int node = __ldcg(&ordered[j]) >> 6;
            if ((unsigned)node < (unsigned)nn) tabu[node] = round + tenure;
        }
    }
    return accepted;
}

static __device__ __forceinline__ int exact_apply_200k(
    int M, int nn, int np, int cap, int *partition, int *part_sizes,
    const int *ordered, int *tabu, int round, int tenure,
    int mark_base, int mark_mult, ExactScratch_200k *work, int *reduce)
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
        return exact_apply_dag_200k(M,nn,np,cap,partition,part_sizes,ordered,
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
            int prefix=ff200k_block_sum(local_accepted,reduce);
            int suffix=exact_apply_dag_200k(M-base,nn,np,cap,partition,part_sizes,ordered+base,
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
    int total=ff200k_block_sum(local_accepted,reduce);
    if(total>0){
        int marks=min(M,max(mark_base,total*mark_mult));
        for(int j=tid;j<marks;j+=blockDim.x){
            int node=__ldcg(&ordered[j])>>6;
            if((unsigned)node<(unsigned)nn)tabu[node]=round+tenure;
        }
    }
    return total;
}

#if 0
static __device__ __forceinline__ int keyed_apply_200k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work)
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
static __device__ __forceinline__ int sparse_reject_200k(
    int M,int nn,int np,int cap,int *partition,int *sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work,const int *data,const ulonglong2 *events){
    const int tid=threadIdx.x,lane=tid&31;
    if(blockDim.x<512 || __ldcg(data+576))return -1;
    int bad=0;
    if(tid<64){
        int initial=tid<np?__ldcg(sizes+tid):0,out=__ldcg(data+tid);
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-initial)):0;
        work->keyed.initial[tid]=initial;work->keyed.outgoing[tid]=out;work->keyed.risk[tid]=risk;
        work->keyed.final_sizes[tid]=initial-out+keeps[tid];
        bad|=risk>8 || initial>cap || initial<0 || (out>0 && initial-out<1);
    }
    if(__syncthreads_or(bad))return -1;
    const int target=tid>>3,local=tid&7;
    const bool valid=tid<512 && target<np && local<work->keyed.risk[target];
    unsigned long long key=0;int packed=-1,margin=0;
    if(valid){
        ulonglong2 event=ff200k_ldcg_u2(events+tid);
        key=event.x;packed=(int)(unsigned)event.y;
        margin=__ldcg(data+64+tid)-local;
        bad|=(packed&63)!=target || ((packed>>6)&63)>=np || (unsigned)(packed>>12)>=(unsigned)nn;
    }
    if(tid<512){work->keyed.ckeys[tid]=key;work->keyed.packed[tid]=packed;}
    if(__syncthreads_or(bad))return -1;
    bool reject=valid && margin<=0,converged=false;int rejected_count=0;
    for(int iteration=0;iteration<6;++iteration){
        unsigned bits=__ballot_sync(0xffffffffu,reject);
        if(lane==0)work->keyed.possible[tid>>5]=(int)bits;
        rejected_count=__syncthreads_count(reject);
        if(rejected_count>64)return -1;
        bool next=false;
        if(valid){
            int delta=0;
            for(int word=0;word<16;++word){
                unsigned pending=(unsigned)work->keyed.possible[word];
                while(pending){
                    int previous=word*32+__ffs(pending)-1;pending&=pending-1;
                    if(work->keyed.ckeys[previous]<key){
                        int event=work->keyed.packed[previous];
                        delta+=(((event>>6)&63)==target)-((event&63)==target);
                    }
                }
            }
            next=margin<=delta;
        }
        bool changed=__syncthreads_or(next!=reject);
        reject=next;
        if(!changed){converged=true;break;}
    }
    if(!converged)return -1;
    const int accepted=M-rejected_count;
    int marks=accepted>0?min(M,max(mark_base,accepted*mark_mult)):0;
    if(marks!=0 && marks!=M)return -1;
    if(reject){
        int source=(packed>>6)&63;
        atomicAdd(work->keyed.final_sizes+source,1);
        atomicSub(work->keyed.final_sizes+target,1);
    }
    __syncthreads();
    if(tid<np && tid<64)sizes[tid]=work->keyed.final_sizes[tid];
    for(int i=tid;i<M;i+=blockDim.x){
        unsigned long long move=__ldcg(selected+i);int node=(int)(unsigned)move,part=(((unsigned)(move>>32))&63)^63;
        partition[node]=part;if(marks)tabu[node]=round+tenure;
    }
    __syncthreads();
    if(reject)partition[packed>>12]=(packed>>6)&63;
    return accepted;
}

static __device__ __forceinline__ int parallel_keyed_apply_200k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work,const int *parallel_data,const ulonglong2 *parallel_events)
{
    const int tid=threadIdx.x,lane=tid&31;
    if(__ldcg(parallel_data+576))return -1;
    int bad=0;
    if(tid<64){
        int c=tid<np?__ldcg(&part_sizes[tid]):0;
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-c)):0;
        work->keyed.initial[tid]=c;work->keyed.outgoing[tid]=__ldcg(parallel_data+tid);
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
            unsigned long long key=ff200k_ldcg_u2(parallel_events+p*8+j).x;
            int node=(int)(unsigned)key,target=(((unsigned)(key>>32))&63)^63;
            int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
            work->keyed.ckeys[index]=key;
            work->keyed.before[index]=__ldcg(parallel_data+64+p*8+j);work->keyed.rejected[index]=0;work->keyed.live[index]=1;
            if((unsigned)source>=(unsigned)np || target!=p){bad=1;continue;}
            work->keyed.packed[index]=(node<<12)|(source<<6)|target;
        }
    }
    if(__syncthreads_or(bad))return -1;
    if(tid<np && tid<64)
        bad|=work->keyed.outgoing[tid]>0 && work->keyed.initial[tid]-work->keyed.outgoing[tid]<1;
    if(__syncthreads_or(bad))return -1;

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
#endif


static __device__ __forceinline__ int exact_apply_critical_200k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const int *ordered,int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work,int *reduce,const int *critical_input,const int *keeps)
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
        return exact_apply_200k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
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
        return exact_apply_200k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
                                 round,tenure,mark_base,mark_mult,work,reduce);
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
        return exact_apply_200k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
                                 round,tenure,mark_base,mark_mult,work,reduce);
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

static __device__ __forceinline__ void exact_sort_prefix_200k(
    unsigned long long *keys,int n,int keep,ExactScratch_200k *work,int *ctl)
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
                __syncwarp(0xffffffffu);
                if(lane==0){ctl[0]=chunk*32+chosen_lane;ctl[1]=r-before;ctl[2]+=before;ctl[3]=amount;}
            }
            __syncthreads();
            int chosen=ctl[0];rank=ctl[1];below=ctl[2];
            prefix|=(unsigned long long)(unsigned)chosen<<shift;
            mask|=255ULL<<shift;
            upper=prefix|(shift?((1ULL<<shift)-1ULL):0ULL);
            if(below+ctl[3]<=limit)break;
        }
        __syncthreads();
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

#undef FF200K_PART_LOAD
#define FF200K_PART_LOAD(n) __ldcg(&partition[(n)])
static __device__ __forceinline__ unsigned long long exact_lcg_cached_200k(
    unsigned long long x,unsigned long long n,const ulonglong2 *table) {
    ulonglong2 ac=__ldg(&table[n]);return ac.x*x+ac.y;
}

static __device__ __forceinline__ int grouped_bucket_200k(int *counts,int target) {
    unsigned active=__activemask();
    unsigned same=__match_any_sync(active,target);
    int lane=threadIdx.x&31,leader=__ffs(same)-1,base=0;
    if(lane==leader)base=atomicAdd(counts+target,__popc(same));
    base=__shfl_sync(active,base,leader);
    return base+__popc(same&((1u<<lane)-1u));
}
static __device__ __forceinline__ unsigned acqbar_load_200k(const unsigned *p){
    unsigned value;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(p):"memory");return value;
}
static __device__ __forceinline__ void acqbar_200k(unsigned *gb,unsigned target){
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile(
            "{\n\t"
            ".reg .u32 _prev;\n\t"
            "atom.acq_rel.gpu.global.add.u32 _prev, [%0], 1;\n\t"
            "}\n"
            :: "l"(gb) : "memory");
        while(acqbar_load_200k(gb)<target)__nanosleep(20);
    }
    __syncthreads();
}

#undef FF200K_GRID_BARRIER
#define FF200K_GRID_BARRIER(g,t) acqbar_200k((g),(t))

static __device__ __forceinline__ int sparse_slot_200k(int *counts,int target){
    unsigned active=__activemask(),group=__match_any_sync(active,target);
    int lane=threadIdx.x&31,leader=__ffs(group)-1,base=0;
    if(lane==leader)base=atomicAdd(counts+target,__popc(group));
    base=__shfl_sync(active,base,leader);
    return base+__popc(group&((1u<<lane)-1u));
}

static __device__ __forceinline__ int packed_count_200k(
    int *counts,const unsigned *metadata,int edge,int part,bool subtract){
    const unsigned record=__ldg(metadata+edge),format=record&3u,base=record>>2;
    const unsigned width=8u<<format,log_fields=2u-format;
    const unsigned shift=((unsigned)part&((1u<<log_fields)-1u))*width;
    unsigned *word=(unsigned*)(counts+base+((unsigned)part>>log_fields));
    const unsigned increment=1u<<shift;
    const unsigned old=atomicAdd(word,subtract?0u-increment:increment);
    const unsigned mask=format==2u?~0u:((1u<<width)-1u);
    return (int)((old>>shift)&mask);
}

static __device__ __forceinline__ int dense_deferred128_200k(
    int M,int nn,int np,int cap,int *partition,int *sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work,const int *data,const ulonglong2 *events, int *reject_flags){
    const int tid=threadIdx.x,lane=tid&31;
    if(blockDim.x<128 || __ldcg(data+576))return -1;
    int bad=0;
    if(tid<64){
        int initial=tid<np?__ldcg(sizes+tid):0,out=__ldcg(data+tid);
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-initial)):0;
        work->keyed.initial[tid]=initial;work->keyed.outgoing[tid]=out;work->keyed.risk[tid]=risk;
        work->keyed.final_sizes[tid]=initial-out+keeps[tid];
        bad|=risk>2 || initial>cap || initial<0 || (out>0 && initial-out<1);
    }
    if(__syncthreads_or(bad))return -1;
    const int target=tid>>1,local=tid&1;
    const bool valid=tid<128 && target<np && local<work->keyed.risk[target];
    unsigned long long key=0;int packed=-1,margin=0;
    if(valid){
        ulonglong2 event=ff200k_ldcg_u2(events+target*8+local);
        key=event.x;packed=(int)(unsigned)event.y;
        margin=__ldcg(data+64+target*8+local)-local;
        bad|=(packed&63)!=target || ((packed>>6)&63)>=np || (unsigned)(packed>>12)>=(unsigned)nn;
    }
    if(tid<128){work->keyed.ckeys[tid]=key;work->keyed.packed[tid]=packed;}
    if(__syncthreads_or(bad))return -1;
    bool reject=valid && margin<=0,converged=false;int rejected_count=0;
    for(int iteration=0;iteration<6;++iteration){
        unsigned bits=__ballot_sync(0xffffffffu,reject);
        if(lane==0 && tid<128)work->keyed.possible[tid>>5]=(int)bits;
        rejected_count=__syncthreads_count(reject);
        if(rejected_count>64)return -1;
        bool next=false;
        if(valid){
            int delta=0;
            for(int word=0;word<4;++word){
                unsigned pending=(unsigned)work->keyed.possible[word];
                while(pending){
                    int previous=word*32+__ffs(pending)-1;pending&=pending-1;
                    if(work->keyed.ckeys[previous]<key){
                        int event=work->keyed.packed[previous];
                        delta+=(((event>>6)&63)==target)-((event&63)==target);
                    }
                }
            }
            next=margin<=delta;
        }
        bool changed=__syncthreads_or(next!=reject);
        reject=next;
        if(!changed){converged=true;break;}
    }
    if(!converged)return -1;
    const int accepted=M-rejected_count;
    int marks=accepted>0?min(M,max(mark_base,accepted*mark_mult)):0;
    if(marks!=0 && marks!=M)return -1;
    if(reject){
        int source=(packed>>6)&63;
        atomicAdd(work->keyed.final_sizes+source,1);
        atomicSub(work->keyed.final_sizes+target,1);
    }
    __syncthreads();
    if(tid<np && tid<64)sizes[tid]=work->keyed.final_sizes[tid];
    if(reject){int index=offsets[target]+max(0,cap-work->keyed.initial[target])+local;reject_flags[index]=1;}
    return accepted;
}
static __device__ __forceinline__ int deferred_sparse_200k(
    int M,int nn,int np,int cap,int *partition,int *sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work,const int *data,const ulonglong2 *events, int *reject_flags){
    int fast=dense_deferred128_200k(M,nn,np,cap,partition,sizes,selected,keeps,offsets,
        tabu,round,tenure,mark_base,mark_mult,work,data,events,reject_flags);
    if(fast>=0)return fast;

    const int tid=threadIdx.x,lane=tid&31;
    if(blockDim.x<512 || __ldcg(data+576))return -1;
    int bad=0;
    if(tid<64){
        int initial=tid<np?__ldcg(sizes+tid):0,out=__ldcg(data+tid);
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-initial)):0;
        work->keyed.initial[tid]=initial;work->keyed.outgoing[tid]=out;work->keyed.risk[tid]=risk;
        work->keyed.final_sizes[tid]=initial-out+keeps[tid];
        bad|=risk>8 || initial>cap || initial<0 || (out>0 && initial-out<1);
    }
    if(__syncthreads_or(bad))return -1;
    const int target=tid>>3,local=tid&7;
    const bool valid=tid<512 && target<np && local<work->keyed.risk[target];
    unsigned long long key=0;int packed=-1,margin=0;
    if(valid){
        ulonglong2 event=ff200k_ldcg_u2(events+tid);
        key=event.x;packed=(int)(unsigned)event.y;
        margin=__ldcg(data+64+tid)-local;
        bad|=(packed&63)!=target || ((packed>>6)&63)>=np || (unsigned)(packed>>12)>=(unsigned)nn;
    }
    if(tid<512){work->keyed.ckeys[tid]=key;work->keyed.packed[tid]=packed;}
    if(__syncthreads_or(bad))return -1;
    bool reject=valid && margin<=0,converged=false;int rejected_count=0;
    for(int iteration=0;iteration<6;++iteration){
        unsigned bits=__ballot_sync(0xffffffffu,reject);
        if(lane==0)work->keyed.possible[tid>>5]=(int)bits;
        rejected_count=__syncthreads_count(reject);
        if(rejected_count>64)return -1;
        bool next=false;
        if(valid){
            int delta=0;
            for(int word=0;word<16;++word){
                unsigned pending=(unsigned)work->keyed.possible[word];
                while(pending){
                    int previous=word*32+__ffs(pending)-1;pending&=pending-1;
                    if(work->keyed.ckeys[previous]<key){
                        int event=work->keyed.packed[previous];
                        delta+=(((event>>6)&63)==target)-((event&63)==target);
                    }
                }
            }
            next=margin<=delta;
        }
        bool changed=__syncthreads_or(next!=reject);
        reject=next;
        if(!changed){converged=true;break;}
    }
    if(!converged)return -1;
    const int accepted=M-rejected_count;
    int marks=accepted>0?min(M,max(mark_base,accepted*mark_mult)):0;
    if(marks!=0 && marks!=M)return -1;
    if(reject){
        int source=(packed>>6)&63;
        atomicAdd(work->keyed.final_sizes+source,1);
        atomicSub(work->keyed.final_sizes+target,1);
    }
    __syncthreads();
    if(tid<np && tid<64)sizes[tid]=work->keyed.final_sizes[tid];
    if(reject){int index=offsets[target]+max(0,cap-work->keyed.initial[target])+local;reject_flags[index]=1;}
    return accepted;
}
static __device__ __forceinline__ int deferred_parallel_200k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_200k *work,const int *parallel_data,const ulonglong2 *parallel_events, int *reject_flags)
{
    const int tid=threadIdx.x,lane=tid&31;
    if(__ldcg(parallel_data+576))return -1;
    int bad=0;
    if(tid<64){
        int c=tid<np?__ldcg(&part_sizes[tid]):0;
        int risk=tid<np?max(0,keeps[tid]-max(0,cap-c)):0;
        work->keyed.initial[tid]=c;work->keyed.outgoing[tid]=__ldcg(parallel_data+tid);
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
            unsigned long long key=ff200k_ldcg_u2(parallel_events+p*8+j).x;
            int node=(int)(unsigned)key,target=(((unsigned)(key>>32))&63)^63;
            int source=(unsigned)node<(unsigned)nn?__ldcg(&partition[node]):-1;
            work->keyed.ckeys[index]=key;
            work->keyed.before[index]=__ldcg(parallel_data+64+p*8+j);work->keyed.rejected[index]=0;work->keyed.live[index]=1;
            if((unsigned)source>=(unsigned)np || target!=p){bad=1;continue;}
            work->keyed.packed[index]=(node<<12)|(source<<6)|target;
        }
    }
    if(__syncthreads_or(bad))return -1;
    if(tid<np && tid<64)
        bad|=work->keyed.outgoing[tid]>0 && work->keyed.initial[tid]-work->keyed.outgoing[tid]<1;
    if(__syncthreads_or(bad))return -1;

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
    if(marks!=0 && marks!=M)return -1;
    int omitted=M-marks;
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
    for(int j=tid;j<S;j+=blockDim.x)if(work->keyed.rejected[j]){
        int target=work->keyed.packed[j]&63;
        int index=offsets[target]+max(0,cap-work->keyed.initial[target])+j-work->keyed.offsets[target];
        reject_flags[index]=1;
    }
    return accepted;
}
extern "C" __global__ __launch_bounds__(512, 1) void exact_round_loop_200k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *__restrict__ hyperedge_nodes,
    const int *__restrict__ node_hyperedges,
    int *partition,
    int *nodes_in_part,
    const int4 *__restrict__ fe_slots,
    const int n_fe_tasks,
    const int4 *__restrict__ fe_giant,
    const int n_fe_giant,
    const int4 *__restrict__ mv_slots,
    const int n_mv_tasks,
    ulonglong2 *edge_flags_pair,
    int *move_priorities,
    unsigned int *grid_barrier,
    const unsigned int initial_barrier_base,
    int *tabu_until,
    const int *__restrict__ mark_nodes,
    const int initial_mark_a,
    const int until_a,
    const int initial_mark_b,
    const int until_b,
    const int round_start,
    const unsigned long long run_off,
    const unsigned long long batch_rounds,
    const unsigned long long prob_den_base,
    int *max_gain_buf,
    int *blk_elig,
    int *tile_valid,
    const int ntiles,
    int *cand,
    const int round_end, const int schedule_end, const int schedule_exp,
    const int move_limit, const int slack_early, const int slack_mid, const int slack_late,
    const int tabu_tenure, const int tabu_mark_base, const int tabu_mark_mult,
    const int perturb_end, const int run_stop, const int initial_stagnant, const int initial_zero,
    unsigned long long *bucket_keys, int *bucket_counts,
    unsigned long long *selected_keys, int *ordered_moves, int *control, const ulonglong2 *lcg_table, const int *count_node_offsets, const int *count_edge_offsets,
    const int count_nh, int *part_counts, int *dirty_epoch, int *dirty_list,
    int *selected_old_parts, int *parallel_data,ulonglong2 *parallel_events, const unsigned *count_meta, const int count_words
)  {
    __shared__ ExactScratch_200k exact_work;
    __shared__ int exact_bucket_starts[65],exact_radix_ctl[4];
    __shared__ int exact_keeps[64], exact_offsets[65], exact_meta[4];
    unsigned int nbar = 0;
    int rounds_done = 0, reason = 0, total_moves = 0;
    int stagnant = initial_stagnant, zero_streak = initial_zero;
    int zero_count = 0, max_zero = initial_zero;
    bool counts_removed=false;
    bool deferred_pending=false,deferred_mark_pending=false;int deferred_until=0;
    for (int step = 0; step < (int)batch_rounds && round_start + step < round_end; ++step) {
        const int round_idx = round_start + step;
        const int n_mark_a = step == 0 ? initial_mark_a : 0;
        const int n_mark_b = step == 0 ? initial_mark_b : 0;
        unsigned int barrier_base = initial_barrier_base + nbar * gridDim.x;
        const unsigned long long lcg_x0 = 123456789ULL + (unsigned long long)round_idx + run_off;
        const unsigned long long left = (unsigned long long)max(0, schedule_end - round_idx);
        unsigned long long prob_num = 1ULL;
        for (int e = 0; e < schedule_exp; ++e) prob_num *= left;

    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64];
    __shared__ int s_warp[32];

    if (threadIdx.x < 64) {
        shared_nodes_in_part[threadIdx.x] = threadIdx.x < num_parts ? __ldcg(&nodes_in_part[threadIdx.x]) : 0;
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_200k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }

    if (gtid < 64) bucket_counts[gtid] = 0;
    for(int j=gtid;j<577;j+=stride)parallel_data[j]=0;
    if (gtid == 0) {
        max_gain_buf[0] = FF200K_MG_INIT; cand[0]=0;
    }

    if(step==0){
        for(int j=gtid;j<count_words;j+=stride)part_counts[j]=0;
        for(int h=gtid;h<count_nh;h+=stride)edge_flags_pair[h]=make_ulonglong2(0,0);
        ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        for(int h=warp_global;h<count_nh;h+=warp_stride){
            int begin=__ldg(&count_edge_offsets[h]),end=__ldg(&count_edge_offsets[h+1]);
            for(int j=begin+(int)lane;j<end;j+=32){
                int n=__ldg(&hyperedge_nodes[j]);
                int part=(unsigned)n<(unsigned)num_nodes?__ldcg(&partition[n]):-1;
                if((unsigned)part<64u){
                    int old=packed_count_200k(part_counts,count_meta,h,part,false);
                    if(old==0)atomicXor(&edge_flags_pair[h].x,1ULL<<part);
                    if(old==1)atomicXor(&edge_flags_pair[h].y,1ULL<<part);
                }
            }
        }
    }else{
        const int previous=__ldcg(&control[13]);
        const int first=(blockIdx.x*blockDim.x+threadIdx.x)/8;
        const int stride8=gridDim.x*blockDim.x/8,lg=(int)threadIdx.x&7;
        for(int i=first;i<previous;i+=stride8){
            unsigned long long key=__ldcg(&selected_keys[i]);
            int node=(int)(unsigned)key,now;
            if(deferred_pending){
                now=__ldcg(ordered_moves+i)?__ldcg(selected_old_parts+i):(int)((((unsigned)(key>>32))&63)^63);
                if(lg==0){partition[node]=now;if(deferred_mark_pending)tabu_until[node]=deferred_until;}
            }else now=__ldcg(&partition[node]);
            int begin=__ldg(&count_node_offsets[node]),end=__ldg(&count_node_offsets[node+1]);
            if((unsigned)now<64u)for(int j=begin+lg;j<end;j+=8){
                int h=__ldg(&node_hyperedges[j]);
                int old=packed_count_200k(part_counts,count_meta,h,now,false);
                if(old==0)atomicXor(&edge_flags_pair[h].x,1ULL<<now);
                if(old==1)atomicXor(&edge_flags_pair[h].y,1ULL<<now);
            }
        }
    }
    counts_removed=false;
    deferred_pending=false;deferred_mark_pending=false;
    for(int i=gtid;i<n_mark_a;i+=stride){
        int node=mark_nodes[i];if((unsigned)node<(unsigned)num_nodes)tabu_until[node]=until_a;
    }
    barrier_base=initial_barrier_base+nbar*gridDim.x;

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF200K_MG_INIT;

    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF200K_MV_BODY(1,  0, FF200K_PC_INC3, FF200K_PC_GET3, FF200K_MV_RED_1)  break;
            case 1:  FF200K_MV_BODY(2,  1, FF200K_PC_INC3, FF200K_PC_GET4, FF200K_MV_RED_2)  break;
            case 2:  FF200K_MV_BODY(4,  2, FF200K_PC_INC3, FF200K_PC_GET5, FF200K_MV_RED_4)  break;
            case 3:  FF200K_MV_BODY(8,  3, FF200K_PC_INC3, FF200K_PC_GET6, FF200K_MV_RED_8)  break;
            case 4:  FF200K_MV_BODY(16, 4, FF200K_PC_INC3, FF200K_PC_GET7, FF200K_MV_RED_16) break;
            default: FF200K_MV_BODY(32, 5, FF200K_PC_INC4, FF200K_PC_GET9, FF200K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff200k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    FF200K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    const int mg=__ldcg(max_gain_buf),aspiration=(mg*3)/4;
    const int tiles_per_block=(ntiles+(int)gridDim.x-1)/(int)gridDim.x;
    const int first_tile=(int)blockIdx.x*tiles_per_block;
    const int last_tile=min(ntiles,first_tile+tiles_per_block);
    int local_eligible=0;
    for(int tile=first_tile;tile<last_tile;++tile){
        int node=tile*(int)blockDim.x+(int)threadIdx.x;
        if(node<num_nodes){
            int key=__ldcg(&move_priorities[node]),tu=__ldcg(&tabu_until[node]);
            if(key!=FF200K_SENT){int gain=key>>16;bool is_tabu=(tu > round_idx) && (gain < aspiration);
                if(!is_tabu && gain <= 0 && gain >= -3)++local_eligible;
            }
        }
    }
    int block_eligible=ff200k_block_sum(local_eligible,s_warp);
    if(threadIdx.x==0)blk_elig[blockIdx.x]=block_eligible;
    FF200K_GRID_BARRIER(grid_barrier,barrier_base+3u*gridDim.x);
    int prefix=0;
    for(int block=(int)threadIdx.x;block<(int)blockIdx.x;block+=(int)blockDim.x)
        prefix+=__ldcg(&blk_elig[block]);
    const int base_eligible=ff200k_block_sum(prefix,s_warp);
    int run_eligible=0,local_valid=0;
    for(int tile=first_tile;tile<last_tile;++tile){
        const int node=tile*(int)blockDim.x+(int)threadIdx.x;
        int key=node<num_nodes?__ldcg(&move_priorities[node]):FF200K_SENT;
        int tu=node<num_nodes?__ldcg(&tabu_until[node]):0;
        int gain=key>>16,eligible=0,positive=0;
        if(node<num_nodes && key!=FF200K_SENT){
            bool is_tabu=(tu > round_idx) && (gain < aspiration);
            positive=!is_tabu && gain>0;
            eligible=(!is_tabu && gain <= 0 && gain >= -3);
        }
        int eligible_count=0;
        int rank=ff200k_block_scan(eligible,s_warp,&eligible_count);
        int outkey=positive?key:-1;
        if(eligible){
            unsigned long long index=(unsigned long long)(base_eligible+run_eligible+rank)+1ULL;
            unsigned long long x=exact_lcg_cached_200k(lcg_x0,index,lcg_table);
            unsigned long long den=prob_den_base*(1ULL+5ULL*(unsigned long long)(-gain));
            if(den>0ULL && x%den<prob_num)outkey=key&0xffff;
        }
        if(node<num_nodes)cand[1+node]=outkey;
        if(outkey>=0){
            ++local_valid;
            int target=outkey&63,slot=sparse_slot_200k(bucket_counts,target);
            if(slot<4096)bucket_keys[target*4096+slot]=
                ((unsigned long long)((unsigned int)outkey^0x7fffffffu)<<32)|(unsigned int)node;
        }
        run_eligible+=eligible_count;
    }
    int block_valid=ff200k_block_sum(local_valid,s_warp);
    if(threadIdx.x==0)atomicAdd(cand,block_valid);
    nbar+=3;
    ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

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
        if(gtid==0)control[13]=M;
        if (N == 0) { reason = 1; break; }
        int adaptive_limit = round_idx < 50 ? move_limit / 2 :
                            (round_idx < 200 ? (move_limit * 3) / 4 : move_limit / 4);
        bool full_quota_equivalent = N <= adaptive_limit ||
            (N <= adaptive_limit + 65536 && M <= adaptive_limit);
        if (!full_quota_equivalent || exact_meta[2] || M <= 0 || num_nodes<256) { reason = 4; break; }


        if(exact_meta[3]){
            if(gtid<64)bucket_counts[64+gtid]=0;
            ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
            for(int i=gtid;i<num_nodes;i+=stride){
                int key=__ldcg(&cand[1+i]);
                if(key<0)continue;
                int node=i,target=key&63;
                int pos=grouped_bucket_200k(bucket_counts + 64,target);
                bucket_keys[exact_bucket_starts[target]+pos]=
                    ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
            }
            ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        }
        for(int p=blockIdx.x;p<num_parts && p<64;p+=gridDim.x){
            int n=__ldcg(&bucket_counts[p]),keep=exact_keeps[p],off=exact_offsets[p];
            int start=exact_bucket_starts[p];
            if(keep==0)continue;
            exact_sort_prefix_200k(&bucket_keys[start],n,keep,&exact_work,exact_radix_ctl);
            for(int j=threadIdx.x;j<keep;j+=blockDim.x){
                unsigned long long key=__ldcg(&bucket_keys[start+j]);
                int node=(int)(unsigned)key;
                int source=__ldcg(&partition[node]);
                selected_keys[off+j]=key;selected_old_parts[off+j]=source;
                if((unsigned)source<(unsigned)num_parts)atomicAdd(parallel_data+source,1);
                else atomicExch(parallel_data+576,1);
                int free=max(0,max_part_size-shared_nodes_in_part[p]),risk=j-free;
                if(risk>=0 && risk<8)parallel_events[p*8+risk]=make_ulonglong2(key,(unsigned)((node<<12)|(source<<6)|p));

            }
        }
        ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

        for(int i=gtid;i<M;i+=stride){
            ordered_moves[i]=0;
            unsigned long long key=__ldcg(selected_keys+i);
            int source=__ldcg(selected_old_parts+i);
            if((unsigned)source>=(unsigned)num_parts)continue;
            int risk=min(8,max(0,exact_keeps[source]-max(0,max_part_size-shared_nodes_in_part[source])));
            for(int j=0;j<risk;++j){
                unsigned long long event=ff200k_ldcg_u2(parallel_events+source*8+j).x;
                if(key<event)atomicAdd(parallel_data+64+source*8+j,1);
            }
        }
        ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        if(gridDim.x==1){
        const int remove_stride=(gridDim.x>1?((int)gridDim.x-1):(int)gridDim.x)*(int)blockDim.x/8;
        const int remove_first=((gridDim.x>1?((int)blockIdx.x-1):(int)blockIdx.x)*(int)blockDim.x+(int)threadIdx.x)/8;
        const int lg=(int)threadIdx.x&7;
        for(int i=remove_first;i<M;i+=remove_stride){
            unsigned long long key=__ldcg(&selected_keys[i]);
            int node=(int)(unsigned)key,old_part=__ldcg(&selected_old_parts[i]);
            int begin=__ldg(&count_node_offsets[node]),end=__ldg(&count_node_offsets[node+1]);
            if((unsigned)old_part<64u)for(int j=begin+lg;j<end;j+=8){
                int h=__ldg(&node_hyperedges[j]);
                int old=packed_count_200k(part_counts,count_meta,h,old_part,true);
                if(old==1)atomicXor(&edge_flags_pair[h].x,1ULL<<old_part);
                if(old==2)atomicXor(&edge_flags_pair[h].y,1ULL<<old_part);
            }
        }
        }
        if(blockIdx.x==0) {
            int result=deferred_sparse_200k(M,num_nodes,num_parts,max_part_size,
                partition,nodes_in_part,selected_keys,exact_keeps,exact_offsets,
                tabu_until,round_idx,tabu_tenure,tabu_mark_base,tabu_mark_mult,&exact_work,parallel_data,parallel_events,ordered_moves);
            if(result<0){
            result=deferred_parallel_200k(M,num_nodes,num_parts,max_part_size,
                partition,nodes_in_part,selected_keys,exact_keeps,exact_offsets,
                tabu_until,round_idx,tabu_tenure,tabu_mark_base,tabu_mark_mult,&exact_work,parallel_data,parallel_events,ordered_moves);
            }
            if(threadIdx.x==0){control[15]=result;control[14]=(result>=0);}
        } else {
        const int remove_stride=(gridDim.x>1?((int)gridDim.x-1):(int)gridDim.x)*(int)blockDim.x/8;
        const int remove_first=((gridDim.x>1?((int)blockIdx.x-1):(int)blockIdx.x)*(int)blockDim.x+(int)threadIdx.x)/8;
        const int lg=(int)threadIdx.x&7;
        for(int i=remove_first;i<M;i+=remove_stride){
            unsigned long long key=__ldcg(&selected_keys[i]);
            int node=(int)(unsigned)key,old_part=__ldcg(&selected_old_parts[i]);
            int begin=__ldg(&count_node_offsets[node]),end=__ldg(&count_node_offsets[node+1]);
            if((unsigned)old_part<64u)for(int j=begin+lg;j<end;j+=8){
                int h=__ldg(&node_hyperedges[j]);
                int old=packed_count_200k(part_counts,count_meta,h,old_part,true);
                if(old==1)atomicXor(&edge_flags_pair[h].x,1ULL<<old_part);
                if(old==2)atomicXor(&edge_flags_pair[h].y,1ULL<<old_part);
            }
        }
        }
        counts_removed=true;
        ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        if(__ldcg(&control[15])<0) {
        const bool shared_merge = M <= 4096;
        if(shared_merge){
            for(int j=threadIdx.x;j<M;j+=blockDim.x)exact_work.sort[j]=__ldcg(&selected_keys[j]);
            __syncthreads();
        }
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
        ++nbar; FF200K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        if (blockIdx.x == 0) {
            int accepted = exact_apply_critical_200k(M, num_nodes, num_parts, max_part_size,
                partition, nodes_in_part, ordered_moves, tabu_until, round_idx,
                tabu_tenure, tabu_mark_base, tabu_mark_mult, &exact_work, s_warp, cand+1, exact_keeps);
            if (threadIdx.x == 0) control[15] = accepted;
        }
        ++nbar; FF200K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        }
        int accepted = __ldcg(&control[15]);
        deferred_pending=__ldcg(control+14)!=0;
        deferred_mark_pending=accepted>0 && (tabu_mark_base>0 || tabu_mark_mult>0);
        deferred_until=round_idx+tabu_tenure;

        if(accepted==0 && N>adaptive_limit){reason=4;break;}
        total_moves += accepted;
        ++rounds_done;
        if (accepted == 0) {
            ++stagnant;
            ++zero_streak;
            ++zero_count;
            max_zero = max(max_zero, zero_streak);
            if (stagnant >= 3 && round_idx < max(0, perturb_end - 50)) { reason = 2; break; }
            if (stagnant > 30) { reason = 3; break; }
        } else { stagnant = 0; zero_streak = 0; }
        if (run_stop > 0 && zero_streak >= run_stop) { reason = 3; break; }

    }
    if(counts_removed){
        const int previous=__ldcg(&control[13]);
        const int first=(blockIdx.x*blockDim.x+threadIdx.x)/8;
        const int stride8=gridDim.x*blockDim.x/8,lg=(int)threadIdx.x&7;
        for(int i=first;i<previous;i+=stride8){
            unsigned long long key=__ldcg(&selected_keys[i]);
            int node=(int)(unsigned)key,now;
            if(deferred_pending){
                now=__ldcg(ordered_moves+i)?__ldcg(selected_old_parts+i):(int)((((unsigned)(key>>32))&63)^63);
                if(lg==0){partition[node]=now;if(deferred_mark_pending)tabu_until[node]=deferred_until;}
            }else now=__ldcg(&partition[node]);
            int begin=__ldg(&count_node_offsets[node]),end=__ldg(&count_node_offsets[node+1]);
            if((unsigned)now<64u)for(int j=begin+lg;j<end;j+=8){
                int h=__ldg(&node_hyperedges[j]);
                int old=packed_count_200k(part_counts,count_meta,h,now,false);
                if(old==0)atomicXor(&edge_flags_pair[h].x,1ULL<<now);
                if(old==1)atomicXor(&edge_flags_pair[h].y,1ULL<<now);
            }
        }
        ++nbar; FF200K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        control[0] = reason; control[1] = rounds_done; control[2] = (int)nbar;
        control[3] = total_moves; control[4] = stagnant; control[5] = zero_streak;
        control[7] = zero_count; control[8] = max_zero;
    }
}

#undef FF200K_PART_LOAD
#define FF200K_PART_LOAD(n) __ldg(&partition[(n)])
