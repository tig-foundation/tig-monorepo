#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void choose_elite_per_hyperedge_50k(
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

extern "C" __global__ void assign_from_elite_votes_50k(
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

extern "C" __global__ void hyperedge_clustering_50k(
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

extern "C" __global__ void compute_node_preferences_50k(
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

extern "C" __global__ void execute_node_assignments_50k(
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


static __device__ __forceinline__ unsigned long long shfl_xor_u64_50k(
    unsigned mask,
    unsigned long long v,
    int lane_mask
) {
    unsigned lo = __shfl_xor_sync(mask, (unsigned)(v & 0xFFFFFFFFu), lane_mask);
    unsigned hi = __shfl_xor_sync(mask, (unsigned)(v >> 32), lane_mask);
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}

extern "C" __global__ void precompute_edge_flags_50k(
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
            unsigned long long other_all = shfl_xor_u64_50k(active, local_all, offset);
            unsigned long long other_double = shfl_xor_u64_50k(active, local_double, offset);
            local_double |= other_double | (local_all & other_all);
            local_all |= other_all;
        }

        if (lane == 0) {
            edge_flags_all[hedge] = local_all;
            edge_flags_double[hedge] = local_double;
        }
    }
}

extern "C" __global__ void compute_refinement_moves_optimized_50k(
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

    int warp_id = threadIdx.x / 32;
    int node = blockIdx.x * (blockDim.x / 32) + warp_id;
    int lane = threadIdx.x & 31;

    if (node >= num_nodes) return;

    int current_part = __ldg(partition + node);
    if ((unsigned)current_part >= (unsigned)num_parts) {
        if (lane == 0) move_priorities[node] = 0x80000000;
        return;
    }
    if (shared_nodes_in_part[current_part] <= 1) {
        if (lane == 0) move_priorities[node] = 0x80000000;
        return;
    }

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) {
        if (lane == 0) move_priorities[node] = 0x80000000;
        return;
    }

    unsigned long long current_bit = 1ULL << current_part;
    int np = (num_parts < 64) ? num_parts : 64;

    int p0 = lane;
    int p1 = lane + 32;

    unsigned short local_count0 = 0;
    unsigned short local_count1 = 0;
    int local_crit = 0;
    int local_current_present = 0;

    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = __ldg(node_hyperedges + (start + rel));

        unsigned long long flags_all = __ldg(edge_flags_all + hyperedge);
        unsigned long long flags_double = __ldg(edge_flags_double + hyperedge);

        if ((flags_all & current_bit) != 0ULL && (flags_double & current_bit) == 0ULL) {
            int parts = __popcll(flags_all);
            if (parts > 1) {
                local_crit += (parts - 1);
                if (local_crit > 255) local_crit = 255;
            }
        }

        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

        if (mask & current_bit) local_current_present++;

        if ((unsigned)p0 < (unsigned)np && (mask & (1ULL << p0))) {
            local_count0++;
        }
        if ((unsigned)p1 < (unsigned)np && (mask & (1ULL << p1))) {
            local_count1++;
        }
    }

    unsigned int full_mask = 0xffffffff;

    int total_crit = __shfl_sync(full_mask, local_crit, 0);
    if (total_crit > 255) total_crit = 255;
    int total_current_present = __shfl_sync(full_mask, local_current_present, 0);

    int degree_weight = node_degree > 255 ? 255 : node_degree;
    int rank_byte = degree_weight + total_crit;
    if (rank_byte > 255) rank_byte = 255;

    int best_gain = -999999;
    int best_target = current_part;
    int current_size = shared_nodes_in_part[current_part];

    if ((unsigned)p0 < (unsigned)np && p0 != current_part) {
        if (shared_nodes_in_part[p0] < max_part_size) {
            int basic_gain = (int)local_count0 - total_current_present;
            int target_size = shared_nodes_in_part[p0];
            int balance_bonus = (current_size > target_size + 2) ? 1 : 0;
            int total_gain = basic_gain + balance_bonus;
            if (total_gain > best_gain || (total_gain == best_gain && p0 < best_target)) {
                best_gain = total_gain;
                best_target = p0;
            }
        }
    }
    if ((unsigned)p1 < (unsigned)np && p1 != current_part) {
        if (shared_nodes_in_part[p1] < max_part_size) {
            int basic_gain = (int)local_count1 - total_current_present;
            int target_size = shared_nodes_in_part[p1];
            int balance_bonus = (current_size > target_size + 2) ? 1 : 0;
            int total_gain = basic_gain + balance_bonus;
            if (total_gain > best_gain || (total_gain == best_gain && p1 < best_target)) {
                best_gain = total_gain;
                best_target = p1;
            }
        }
    }

    int other_gain, other_target;
    for (int offset = 1; offset < 32; offset <<= 1) {
        other_gain = __shfl_xor_sync(full_mask, best_gain, offset);
        other_target = __shfl_xor_sync(full_mask, best_target, offset);
        if (other_gain > best_gain || (other_gain == best_gain && other_target < best_target)) {
            best_gain = other_gain;
            best_target = other_target;
        }
    }

    if (lane == 0) {
        if (best_target != current_part && best_gain >= gain_floor) {
            int bg = best_gain > 32767 ? 32767 : best_gain;
            if (bg < -32768) bg = -32768;
            unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
            move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
        } else {
            move_priorities[node] = 0x80000000;
        }
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

static __device__ __forceinline__ int select_move_target_50k(
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
        if (target_part == current_part) continue;
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

static __device__ __forceinline__ bool sel_better_50k(
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

struct BS6_50k { unsigned long long d0, d1, d2, d3, d4, d5; };

static __device__ __forceinline__ void bs6_add_mask_50k(BS6_50k &s, unsigned long long m) {
    unsigned long long c = m, t;
    t = s.d0 & c; s.d0 ^= c; c = t;
    t = s.d1 & c; s.d1 ^= c; c = t;
    t = s.d2 & c; s.d2 ^= c; c = t;
    t = s.d3 & c; s.d3 ^= c; c = t;
    t = s.d4 & c; s.d4 ^= c; c = t;
    s.d5 ^= c;
}

#define BS6_FA_50k(a, b, c) { unsigned long long x_ = (a) ^ (b); unsigned long long cn_ = ((a) & (b)) | ((c) & x_); (a) = x_ ^ (c); (c) = cn_; }
static __device__ __forceinline__ void bs6_add_50k(BS6_50k &a, const BS6_50k &b) {
    unsigned long long c = 0ULL;
    BS6_FA_50k(a.d0, b.d0, c)
    BS6_FA_50k(a.d1, b.d1, c)
    BS6_FA_50k(a.d2, b.d2, c)
    BS6_FA_50k(a.d3, b.d3, c)
    BS6_FA_50k(a.d4, b.d4, c)
    a.d5 = a.d5 ^ b.d5 ^ c;
}

static __device__ __forceinline__ BS6_50k bs6_shfl_xor_50k(const BS6_50k &s, int offset) {
    BS6_50k o;
    o.d0 = shfl_xor_u64_50k(0xffffffffu, s.d0, offset);
    o.d1 = shfl_xor_u64_50k(0xffffffffu, s.d1, offset);
    o.d2 = shfl_xor_u64_50k(0xffffffffu, s.d2, offset);
    o.d3 = shfl_xor_u64_50k(0xffffffffu, s.d3, offset);
    o.d4 = shfl_xor_u64_50k(0xffffffffu, s.d4, offset);
    o.d5 = shfl_xor_u64_50k(0xffffffffu, s.d5, offset);
    return o;
}

static __device__ __forceinline__ unsigned int bs6_get_50k(const BS6_50k &s, int t) {
    return (unsigned int)((s.d0 >> t) & 1ULL)
         | ((unsigned int)((s.d1 >> t) & 1ULL) << 1)
         | ((unsigned int)((s.d2 >> t) & 1ULL) << 2)
         | ((unsigned int)((s.d3 >> t) & 1ULL) << 3)
         | ((unsigned int)((s.d4 >> t) & 1ULL) << 4)
         | ((unsigned int)((s.d5 >> t) & 1ULL) << 5);
}

static __device__ __forceinline__ bool sel_better_bs_50k(
    int gain_a, int t_a, int gain_b, int t_b,
    const BS6_50k &s, const int *sizes, int current_part, int node, int tiebreak
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
        unsigned int ca = bs6_get_50k(s, t_a), cb = bs6_get_50k(s, t_b);
        if (ca != cb) return ca > cb;
    }
    return t_a < t_b;
}

static __device__ __forceinline__ int select_move_target_bs_50k(
    const BS6_50k &s,
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
            if (target_part == current_part) continue;
            if ((unsigned)target_part >= (unsigned)num_parts) continue;
            int target_size = shared_nodes_in_part[target_part];
            if (target_size >= max_part_size) continue;
            int total_gain = (int)bs6_get_50k(s, target_part) - count_current_present
                           + ((current_size > target_size + 2) ? 1 : 0);
            if (sel_better_bs_50k(total_gain, target_part, best_gain, best_target,
                                   s, shared_nodes_in_part, current_part, node, tiebreak)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        }
    }
    for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
        int og = __shfl_xor_sync(0xffffffffu, best_gain, offset);
        int ot = __shfl_xor_sync(0xffffffffu, best_target, offset);
        if (eligible && sel_better_bs_50k(og, ot, best_gain, best_target,
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

static __device__ __forceinline__ int select_move_target_par_50k(
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
            if (target_part == current_part) continue;
            if ((unsigned)target_part >= (unsigned)num_parts) continue;
            int target_size = shared_nodes_in_part[target_part];
            if (target_size >= max_part_size) continue;
            int total_gain = (int)pcount[target_part] - count_current_present
                           + ((current_size > target_size + 2) ? 1 : 0);
            if (sel_better_50k(total_gain, target_part, best_gain, best_target,
                                pcount, shared_nodes_in_part, current_part, node, tiebreak)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        }
    }
    for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
        int og = __shfl_xor_sync(0xffffffffu, best_gain, offset);
        int ot = __shfl_xor_sync(0xffffffffu, best_target, offset);
        if (eligible && sel_better_50k(og, ot, best_gain, best_target,
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

extern "C" __global__ void compute_refinement_moves_lanes_50k(
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
                key = select_move_target_50k(pinfo, shared_nodes_in_part, num_parts, max_part_size,
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
                    key = select_move_target_50k(pinfo, shared_nodes_in_part, num_parts, max_part_size,
                                                 current_part, my_present, hn, node_degree, my_crit, tiebreak, gain_floor);
                }
                move_priorities[hn] = key;
            }
            __syncwarp();
        }
    }
}

extern "C" __global__ void compute_swap_gains_extended_50k(
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

extern "C" __global__ void compute_connectivity_50k(
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

extern "C" __global__ void reduce_connectivity_sum_50k(
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


extern "C" __global__ void balance_final_50k(
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


__device__ __forceinline__ unsigned long long lcg_jump_50k(unsigned long long x0, unsigned int k) {
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

extern "C" __global__ void filter_fused_50k(
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
            unsigned long long x = lcg_jump_50k(seed, j + 1u);
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

#define RF_TILE_50k 128
#define RF_WPT_50k 4

static __device__ __forceinline__ bool rf_barrier_50k(
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

#define RF_FILT_WORDS_50k (8 * 256)
#ifndef HG_RF_SMEM_DECLARED
#define HG_RF_SMEM_DECLARED
extern __shared__ unsigned int rf_smem[];
#endif
static __device__ __forceinline__ bool rf_round_50k(
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
#define RF_BAR_50k(k) if (!rf_barrier_50k(grid, soft, bar + (k), bar_target, abort_flag, &s_abort)) return false;
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
    RF_BAR_50k(0)

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
                unsigned long long other_all = shfl_xor_u64_50k(0xffffffffu, local_all, offset);
                unsigned long long other_double = shfl_xor_u64_50k(0xffffffffu, local_double, offset);
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
    RF_BAR_50k(1)

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

            BS6_50k bs = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
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
                    bs6_add_mask_50k(bs, fa & ~current_bit);
                }
            }
            for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
                my_present += __shfl_xor_sync(0xffffffffu, my_present, offset);
                my_crit    += __shfl_xor_sync(0xffffffffu, my_crit, offset);
                BS6_50k o = bs6_shfl_xor_50k(bs, offset);
                bs6_add_50k(bs, o);
            }

            {
                int key = select_move_target_bs_50k(bs, shared_nodes_in_part, num_parts, max_part_size,
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
                int key = select_move_target_par_50k(wpinfo, shared_nodes_in_part, num_parts, max_part_size,
                                                     current_part, my_present, hn, node_degree, my_crit,
                                                     tiebreak, gain_floor, eligible, lane, 32);
                if (lane == 0) move_priorities[hn] = key;
            }
            __syncwarp();
        }
    }
    RF_BAR_50k(2)

    if (gsort) {
        const int nt_max = (num_nodes + 255) >> 8;
        const int rs_total = 4 * 256 * nt_max;
        for (int i = gtid; i < rs_total; i += nthreads) rs_counts[i] = 0u;
        for (int i = gtid; i < 64 * nt_max; i += nthreads) tcnt[i] = 0u;
    }
    const int num_tiles = (num_nodes + RF_TILE_50k - 1) / RF_TILE_50k;
    const int subs = blockDim.x / RF_TILE_50k;
    const int tsub = threadIdx.x / RF_TILE_50k;
    const int tth = threadIdx.x & (RF_TILE_50k - 1);
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const int num_groups = (num_tiles + subs - 1) / subs;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_50k + tth;
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
            for (int w = tsub * RF_WPT_50k; w < (tsub + 1) * RF_WPT_50k; w++) {
                bmx = max(bmx, red[0 * 32 + w]); s1 += red[1 * 32 + w]; s2 += red[2 * 32 + w];
                s3 += red[3 * 32 + w]; s4 += red[4 * 32 + w]; s5 += red[5 * 32 + w]; s6 += red[6 * 32 + w];
            }
            int *o = stats + t * 8;
            o[0] = bmx; o[1] = s1; o[2] = s2; o[3] = s3; o[4] = s4; o[5] = s5; o[6] = s6; o[7] = 0;
        }
        __syncthreads();
    }
    RF_BAR_50k(3)

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
    RF_BAR_50k(4)

    const int aspiration_h = __ldcg(hdr + 1);
    unsigned int *warp_acc = warp_low + 16;
    unsigned int *count = (unsigned int *)(hdr + 3);
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_50k + tth;
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
        for (int w = tsub * RF_WPT_50k; w < warp_id; w++) warp_prefix += warp_low[w];

        bool accept = pos;
        unsigned int out_key32 = (unsigned int)key;
        if (low) {
            unsigned int j = __ldcg(tile_off + t) + warp_prefix + low_rank;
            unsigned long long x = lcg_jump_50k(seed, j + 1u);
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
            for (int w = tsub * RF_WPT_50k; w < (tsub + 1) * RF_WPT_50k; w++) c += warp_acc[w];
            stats[t * 8 + 7] = (int)c;
        }
        __syncthreads();
    }
    if (!gsort) {
        return true;
    }
    RF_BAR_50k(5)

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
    RF_BAR_50k(6)

    const unsigned int cnt_c = (unsigned int)__ldcg(hdr + 3);
    const int nt_c = (int)((cnt_c + 255u) >> 8);
    const int npass = (__ldcg(hdr + 2) >= 256) ? 4 : 3;
    for (int g = blockIdx.x; g < num_groups; g += gridDim.x) {
        const int t = g * subs + tsub;
        const int node = t * RF_TILE_50k + tth;
        unsigned int v = 0u;
        if (t < num_tiles && node < num_nodes) v = __ldcg(acc + node);
        unsigned m = __ballot_sync(0xffffffffu, v != 0u);
        unsigned r = __popc(m & lanemask_lt);
        if (lane == 0) warp_low[warp_id] = __popc(m);
        __syncthreads();
        unsigned pre = 0;
        for (int w = tsub * RF_WPT_50k; w < warp_id; w++) pre += warp_low[w];
        if (v != 0u) {
            unsigned pos = __ldcg(acc_off + t) + pre + r;
            out[pos] = ((unsigned long long)v << 32) | (unsigned long long)(unsigned int)node;
            atomicAdd(&rs_counts[(pos >> 8) * 256u + (v & 0xFFu)], 1u);
        }
        __syncthreads();
    }

    unsigned int *wc = rf_smem;
    for (int k = 0; k < 4; k++) {
        RF_BAR_50k(7 + 2 * k)
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
        RF_BAR_50k(8 + 2 * k)
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
    RF_BAR_50k(15)
    if (blockIdx.x == 0 && threadIdx.x < 64) {
        const int t = threadIdx.x;
        unsigned int run = 0;
        for (int tile = 0; tile < nt_c; tile++) {
            unsigned int c = __ldcg(tcnt + tile * 64 + t);
            tcnt[tile * 64 + t] = run;
            run += c;
        }
    }
    RF_BAR_50k(16)
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
    RF_BAR_50k(17)
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
    RF_BAR_50k(18)
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

extern "C" __global__ void __launch_bounds__(256, 4) round_fused_50k(
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
    rf_round_50k(num_hyperedges, num_nodes, num_parts, max_part_size, hyperedge_nodes, hyperedge_offsets,
                  node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_pair, move_priorities,
                  lanes, tiebreak, gain_floor, round, tabu_until, stats, tile_off, hdr, seed, prob_num,
                  den0, den1, den2, den3, out, soft, launch_idx, n_tabu_upd, tabu_pairs,
                  n_part_upd, part_pairs, acc, acc_off, out2, rs_counts, gsort, tcnt, koff, slack,
                  adaptive_limit, extra_window);
}

static __device__ __forceinline__ unsigned long long rf_sat_mul_50k(unsigned long long a, unsigned long long b) {
    return (__umul64hi(a, b) != 0ULL) ? 0xFFFFFFFFFFFFFFFFULL : a * b;
}

extern "C" __global__ void __launch_bounds__(256, 4) rounds_fused_50k(
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
        const unsigned long long prob_num = rf_sat_mul_50k(pw, num_mul);
        const int ntu = (r == 0) ? n_tabu_upd : 0;
        const int npu = (r == 0) ? n_part_upd : 0;
        if (!rf_round_50k(num_hyperedges, num_nodes, num_parts, max_part_size, hyperedge_nodes, hyperedge_offsets,
                           node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_pair, move_priorities,
                           lanes, tiebreak, gain_floor, (unsigned int)round, tabu_until, stats, tile_off, hdr, seed,
                           prob_num, den0, den1, den2, den3, out, soft, epoch, ntu, tabu_pairs, npu,
                           part_pairs, acc, acc_off, out2, rs_counts, 1, tcnt, koff, slack, adaptive_limit,
                           extra_window)) {
            return;
        }
        if (!rf_barrier_50k(grid, soft, bar + 19, (epoch + 1u) * gridDim.x, abort_flag, &s_abort2)) return;
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
                    if (tid < 32) {
                        const unsigned int n = min(256u, take - base);
                        unsigned int ex = 0u;
                        for (unsigned int w = 0; w < 8u; w++) {
                            const unsigned int j0 = w * 32u;
                            if (j0 >= n) { if (tid == 0) s_mask[w] = 0u; continue; }
                            const unsigned int j = j0 + (unsigned int)tid;
                            const bool in_w = j < n;
                            const int nj = in_w ? s_node[j] : -1;
                            const int tj = in_w ? s_tgt[j] : -1;
                            const int cj = in_w ? s_cur[j] : -1;
                            const bool v = in_w && (unsigned)nj < (unsigned)num_nodes
                                           && (unsigned)tj < (unsigned)num_parts && (unsigned)cj < (unsigned)num_parts;
                            const unsigned int vmask = __ballot_sync(0xFFFFFFFFu, v);
                            unsigned int m = 0u;
                            if (vmask != 0u) {
                                s_ct[tid] = 0; s_ct[tid + 32] = 0;
                                s_cc[tid] = 0; s_cc[tid + 32] = 0;
                                __syncwarp();
                                if (v) { atomicAdd(&s_ct[tj], 1); atomicAdd(&s_cc[cj], 1); }
                                __syncwarp();
                                bool okw = true;
                                if (v) {
                                    if (s_nip[tj] + s_ct[tj] > max_part_size) okw = false;
                                    if (s_nip[cj] - s_cc[cj] < 1) okw = false;
                                }
                                const bool win_ok = __all_sync(0xFFFFFFFFu, okw);
                                if (win_ok) {
                                    m = vmask;
                                    s_nip[tid] += s_ct[tid] - s_cc[tid];
                                    s_nip[tid + 32] += s_ct[tid + 32] - s_cc[tid + 32];
                                } else {
                                    bool tight = false;
                                    if (v) {
                                        const bool tp = (s_nip[tj] + s_ct[tj] > max_part_size) || (s_nip[tj] - s_cc[tj] < 1);
                                        const bool cp = (s_nip[cj] + s_ct[cj] > max_part_size) || (s_nip[cj] - s_cc[cj] < 1);
                                        tight = tp || cp;
                                    }
                                    const unsigned int scan_mask = __ballot_sync(0xFFFFFFFFu, tight);
                                    __syncwarp();
                                    s_ct[tid] = 0; s_ct[tid + 32] = 0;
                                    s_cc[tid] = 0; s_cc[tid + 32] = 0;
                                    __syncwarp();
                                    if (v && !tight) { atomicAdd(&s_ct[tj], 1); atomicAdd(&s_cc[cj], 1); }
                                    __syncwarp();
                                    s_nip[tid] += s_ct[tid] - s_cc[tid];
                                    s_nip[tid + 32] += s_ct[tid + 32] - s_cc[tid + 32];
                                    __syncwarp();
                                    m = vmask & ~scan_mask;
                                    if (tid == 0) {
                                        unsigned int rem = scan_mask;
                                        while (rem != 0u) {
                                            const unsigned int q = (unsigned int)__ffs(rem) - 1u;
                                            rem &= rem - 1u;
                                            const unsigned int jj = j0 + q;
                                            const int tq = s_tgt[jj], cq = s_cur[jj];
                                            if (s_nip[tq] < max_part_size && s_nip[cq] > 1) {
                                                s_nip[cq] -= 1;
                                                s_nip[tq] += 1;
                                                m |= (1u << q);
                                            }
                                        }
                                    }
                                    m = __shfl_sync(0xFFFFFFFFu, m, 0);
                                }
                                __syncwarp();
                            }
                            if (tid == 0) s_mask[w] = m;
                            ex += (unsigned int)__popc(m);
                        }
                        if (tid == 0) s_mask[8] = ex;
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
        if (!rf_barrier_50k(grid, soft, bar + 20, (epoch + 1u) * gridDim.x, abort_flag, &s_abort2)) return;
        if (__ldcg(hdr + 25) != 0) return;
    }
}
