#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void choose_elite_per_hyperedge_20k(
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

extern "C" __global__ void assign_from_elite_votes_20k(
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

extern "C" __global__ void hyperedge_clustering_20k(
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

extern "C" __global__ void compute_node_preferences_20k(
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

extern "C" __global__ void execute_node_assignments_20k(
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



extern "C" __global__ void precompute_edge_flags_20k(
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

extern "C" __global__ void compute_refinement_moves_optimized_20k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *partition,
    const int *nodes_in_part,
    const unsigned long long *edge_flags_all,
    const unsigned long long *edge_flags_double,
    int *move_priorities
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

    if (best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : best_gain;
        if (bg < -32768) bg = -32768;
        unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
        move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
    }
}

extern "C" __global__ void compute_swap_gains_extended_20k(
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

extern "C" __global__ void compute_connectivity_20k(
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

extern "C" __global__ void reduce_connectivity_sum_20k(
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



extern "C" __global__ void balance_final_20k(
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


// ---- exp6: flags + moves in ONE launch. Phase 1 computes the per-hyperedge flags exactly as
// precompute_edge_flags_20k (once per hyperedge); a software grid barrier (monotonic counter,
// target = calls * gridDim.x, passed by the host) orders it before phase 2, which is
// compute_refinement_moves_optimized_20k verbatim in a grid-stride loop. Requires all blocks to be
// co-resident: host launches at most 2 blocks per SM (128 threads each).
extern "C" __global__ void fused_flags_moves_20k(
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
    const int stride = gridDim.x * blockDim.x;

    // ---- phase 1a: small hyperedges, one thread each (identical to precompute_edge_flags_20k) ----
    const int WARP_PIN_THRESHOLD = 32;
    for (int hedge = blockIdx.x * blockDim.x + threadIdx.x; hedge < num_hyperedges; hedge += stride) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        if (end - start > WARP_PIN_THRESHOLD) continue;   // handled by phase 1b
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
    // ---- phase 1b: large hyperedges, one WARP each. Exact: (all, double) combine is associative and
    // commutative: all = a|b, double = da|db|(a&b) (a part is seen >=2 times overall iff seen >=2 in one
    // subset or >=1 in both). Bitwise ops => reduction order irrelevant => deterministic.
    {
        const int lane = threadIdx.x & 31;
        const int warps_per_block = blockDim.x >> 5;
        const int warp_global = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
        const int warp_stride = gridDim.x * warps_per_block;
        for (int hedge = warp_global; hedge < num_hyperedges; hedge += warp_stride) {
            int start = hyperedge_offsets[hedge];
            int end = hyperedge_offsets[hedge + 1];
            if (end - start <= WARP_PIN_THRESHOLD) continue;
            unsigned long long a = 0, d = 0;
            for (int k = start + lane; k < end; k += 32) {
                int node = hyperedge_nodes[k];
                if (node >= 0 && node < num_nodes) {
                    int part = partition[node];
                    if (part >= 0 && part < 64) {
                        unsigned long long bit = 1ULL << part;
                        d |= (a & bit);
                        a |= bit;
                    }
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                unsigned long long oa = __shfl_xor_sync(0xffffffffu, a, off);
                unsigned long long od = __shfl_xor_sync(0xffffffffu, d, off);
                d = d | od | (a & oa);
                a = a | oa;
            }
            if (lane == 0) {
                edge_flags_all[hedge] = a;
                edge_flags_double[hedge] = d;
            }
        }
    }

    // ---- grid barrier ----
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(grid_barrier, 1u);
        while (atomicAdd(grid_barrier, 0u) < barrier_target) {
            __nanosleep(20);
        }
    }
    __syncthreads();
    __threadfence();

    // ---- phase 2: moves (identical to compute_refinement_moves_optimized_20k; flags read via L2) ----
    for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < num_nodes; node += stride) {
        move_priorities[node] = 0x80000000;

        int current_part = partition[node];
        if ((unsigned)current_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[current_part] <= 1) continue;

        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;
        int used_degree = node_degree > 256 ? 256 : node_degree;
        if (used_degree <= 0) continue;

        unsigned long long current_bit = 1ULL << current_part;

        unsigned long long pc0 = 0ULL, pc1 = 0ULL, pc2 = 0ULL, pc3 = 0ULL, pc4 = 0ULL, pc5 = 0ULL, pc6 = 0ULL, pc7 = 0ULL, pc8 = 0ULL;

        unsigned long long cand_mask = 0ULL;
        int count_current_present = 0;
        int crit = 0;

        for (int j = 0; j < used_degree; j++) {
            int rel = (int)(((long long)j * node_degree) / used_degree);
            int hyperedge = node_hyperedges[start + rel];

            unsigned long long flags_all = __ldcg(&edge_flags_all[hyperedge]);
            unsigned long long flags_double = __ldcg(&edge_flags_double[hyperedge]);

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
            cand_mask |= flags;
            {
                unsigned long long c = flags, t;
                t = pc0 & c; pc0 ^= c; c = t;
                t = pc1 & c; pc1 ^= c; c = t;
                t = pc2 & c; pc2 ^= c; c = t;
                t = pc3 & c; pc3 ^= c; c = t;
                t = pc4 & c; pc4 ^= c; c = t;
                t = pc5 & c; pc5 ^= c; c = t;
                t = pc6 & c; pc6 ^= c; c = t;
                t = pc7 & c; pc7 ^= c; c = t;
                t = pc8 & c; pc8 ^= c; c = t;
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

            int pcount = ((int)((pc0 >> target_part) & 1ULL) << 0) | ((int)((pc1 >> target_part) & 1ULL) << 1) | ((int)((pc2 >> target_part) & 1ULL) << 2) | ((int)((pc3 >> target_part) & 1ULL) << 3) | ((int)((pc4 >> target_part) & 1ULL) << 4) | ((int)((pc5 >> target_part) & 1ULL) << 5) | ((int)((pc6 >> target_part) & 1ULL) << 6) | ((int)((pc7 >> target_part) & 1ULL) << 7) | ((int)((pc8 >> target_part) & 1ULL) << 8);
            int basic_gain = pcount - count_current_present;

            int current_size = shared_nodes_in_part[current_part];
            int target_size = shared_nodes_in_part[target_part];
            int balance_bonus = (current_size > target_size + 2) ? 1 : 0;

            int total_gain = basic_gain + balance_bonus;

            if (total_gain > best_gain || (total_gain == best_gain && target_part < best_target)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        }

        if (best_target != current_part) {
            int bg = best_gain > 32767 ? 32767 : best_gain;
            if (bg < -32768) bg = -32768;
            unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
            move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
        }
    }
}


// ===========================================================================
// exp12 / wave2_gpufilter: fused flags + moves + the MAIN-LOOP HOST FILTER,
// in ONE launch. Adds three phases behind two extra software grid barriers:
//   phase 0   : reset max-gain accumulator, scatter tabu mark list A
//   phase 1   : edge flags (thread-per-small-hedge + warp-per-large-hedge)
//   BARRIER 1
//   phase 2   : scatter tabu mark list B (ordered after A by barrier 1),
//               then the moves kernel, accumulating max(gain) over
//               non-sentinel keys via one atomicMax per block
//   BARRIER 2
//   phase 3 (pass A) : per-block count of lottery-eligible nodes
//   BARRIER 3
//   phase 4 (pass B) : global eligible prefix -> exact LCG index per eligible
//               node (jump-ahead), lottery, then a node-index-ordered
//               compaction of the valid candidates into this block's own
//               output segment.
// (wave3_kfilter has SUPERSEDED this layout: see the block just above
// fused_flags_moves_filter_20k. The description above is kept because the
// helpers below are shared; the shipped kernel uses 4 barriers and a globally
// contiguous, node-ordered candidate array.)
//
// Barrier targets: the host passes `barrier_base` = (barriers issued so far)
// * gridDim.x; barrier k waits for barrier_base + k*gridDim.x.
// The counter is monotone (never reset) and one nonce issues < 2^25 ticks.
//
// EVERY cross-phase global read uses __ldcg (L2, bypass the non-coherent L1),
// exactly as the champion already does for edge_flags_all/double.
// ===========================================================================

#define FF20K_SENT    ((int)0x80000000)
#define FF20K_MG_INIT (-1048576)

// f(x) = x*6364136223846793005 + 1442695040888963407  (mod 2^64)
// Returns f^n(x0). Affine composition by squaring:
//   f^m = (A,C): x -> A*x + C ; (Ab,Cb) o (A,C) = (Ab*A, Ab*C + Cb)
//   square: f^(2m) = f^m o f^m = (Ab*Ab, Ab*Cb + Cb)
// n <= num_nodes+1 (~18401) => at most 15 iterations.
static __device__ __forceinline__ unsigned long long ff20k_lcg_advance(
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

// wave6_kmicro4 (W6): the SPIN reads the counter with a volatile load instead
// of atomicAdd(gb, 0).  The protocol is untouched -- one atomicAdd(gb,1) per
// block on arrival, an absolute monotone target, the same two __syncthreads
// and the same two __threadfence -- only the poll instruction changes.  The
// champion had 157 block leaders issuing a read-modify-write to ONE L2 address
// every 200 ns (~0.8 atomics/ns, at the single-address atomic throughput of
// the L2), so the arriving atomicAdd(gb,1) increments had to queue behind the
// pollers; a plain volatile load is served by the L2 like any other read and
// several can be in flight, so arrivals are seen promptly.  ld.volatile.global
// bypasses L1 and is re-issued on every trip (asm volatile + "memory"), so the
// loop cannot spin on a cached value.  The counter is monotone and one nonce
// issues < 2^25 ticks, so the unsigned comparison is safe.
static __device__ __forceinline__ unsigned int ff20k_ld_vol_u32(const unsigned int *p)
{
    unsigned int r;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
}

// Software grid barrier (monotonic counter, host-supplied absolute target).
#define FF20K_GRID_BARRIER(gb, target)                        \
    do {                                                   \
        __threadfence();                                   \
        __syncthreads();                                   \
        if (threadIdx.x == 0) {                            \
            atomicAdd((gb), 1u);                           \
            while (ff20k_ld_vol_u32(gb) < (target)) {         \
                __nanosleep(20);                          \
            }                                              \
        }                                                  \
        __syncthreads();                                   \
        __threadfence();                                   \
    } while (0)

// All three helpers are called by every thread of the block (uniform control
// flow) and use only integer +/max => order-independent => deterministic.
// Each leaves s_warp reusable (trailing __syncthreads).
static __device__ __forceinline__ int ff20k_block_sum(int v, int *s_warp)
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

static __device__ __forceinline__ int ff20k_block_max(int v, int *s_warp)
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

// Exclusive prefix count of `pred` over the block in threadIdx.x order,
// plus the block total in *total_out.
static __device__ __forceinline__ int ff20k_block_scan(int pred, int *s_warp, int *total_out)
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


// ===========================================================================
// wave4_kmicro2: fused flags + moves + host-filter, ONE launch, 4 grid
// barriers, SAME host-visible semantics as wave3_kfilter (identical d_cand,
// identical d_move_priorities, identical tabu behaviour).  Four changes, all
// bit-identical and all order-independent:
//
//  (M1) WARP TASKS WITH LANE GROUPS (both the flags phase and the moves
//       phase).  Both phases are serial walks over a work item's pin list, so
//       a warp used to cost max(size) over the 32 items its lanes happened to
//       land on, while the mean size is ~5 -- 32 lanes waiting for one.  The
//       host now sorts the work items into DEGREE CLASSES c=0..5 with
//       G(c) = 1,2,4,8,16,32 lanes per item, chosen so that every class runs
//       at most 4 loop iterations per lane (class 5 at most 8, because
//       used_degree is capped at 256).  A "warp task" is 32/G(c) consecutive
//       items of one class; the task list is a flat int array, one int per
//       task, packed as (first_item_index << 3) | c, and a warp grid-strides
//       over it.  Every lane of a warp works on the same class, so the
//       control flow inside a task is warp-uniform and every
//       __shfl_xor_sync(0xffffffff, ...) sees all 32 lanes.
//       Exactness: within a lane group the per-item accumulators are combined
//       with a butterfly over offsets G/2..1 (an xor-shuffle with offset < G
//       never leaves the G-aligned group), using only associative and
//       commutative operators -- OR (cand_mask, edge flags_all), plain int +
//       (count_current_present, crit), and the bit-sliced ripple-carry ADD
//       (per-part counts) -- so the lane split is invisible.  After the
//       butterfly every lane of the group holds the identical reduced state
//       and each lane then runs the CHAMPION'S VERBATIM SERIAL best-(gain,
//       target) scan; only lane 0 of the group stores.  Redundant identical
//       work, no parallel-max re-derivation, hence trivially the same answer.
//
//  (M2) PACKED FLAGS.  edge_flags_all/edge_flags_double were two u64 arrays,
//       so every pin of the moves phase issued TWO 8-byte gathers landing in
//       two different 32-byte sectors (64 B fetched per pin).  They are now
//       one ulonglong2 array written once in phase 1 with a single 16-byte
//       store and read in phase 2 with a single 16-byte ld.global.cg.v2.u64
//       (32 B per pin).  Halves both the request count and the L2 traffic of
//       the moves phase, which is ~96k pin gathers per round.
//
//  (M3) NO 64-BIT DIVIDE IN THE MOVES LOOP.  The champion computes
//       rel = (long long)j * node_degree / used_degree with
//       used_degree = min(node_degree, 256).  When node_degree <= 256 the two
//       are equal and rel == j exactly (integer identity j*d/d = j, d >= 1);
//       when node_degree > 256 the divisor is the literal 256 and the
//       division is an exact right shift by 8 (both operands non-negative,
//       j*node_degree <= 255*num_hyperedges < 2^31).  Classes 0..4 have
//       node_degree <= 64 so they never even test it.
//
//  (M4) SIZE-CLASSED FLAGS PHASE + A BLOCK CLASS FOR THE GIANTS.
//       Hyperedges up to 256 pins go through the same warp-task machinery;
//       hyperedges above 256 pins (up to 1,954 here) get a whole 128-thread
//       BLOCK each, so the phase-1 tail drops from ceil(1954/32) = 61 warp
//       iterations to ceil(1954/128) = 16 block iterations plus a 4-way
//       shared-memory combine.  The combine is the champion's associative
//       (a,d) merge, applied in a fixed warp order, so it is deterministic.
//
// Every cross-phase global read still uses a .cg load (L2, bypassing the
// non-coherent L1), exactly as the champion does.
//
// Barrier targets: host passes barrier_base = (barriers issued so far) *
// gridDim.x; barrier k (k=1..4) waits for barrier_base + k*gridDim.x.
// ===========================================================================

// 16-byte .cg load of a (flags_all, flags_double) pair.  Written exactly the
// way CUDA's own __ldcg is written in sm_32_intrinsics.hpp (asm volatile + a
// "memory" clobber) so it cannot be hoisted across the grid barrier that
// separates the producing phase 1 from the consuming phase 2; the champion
// already pays this twice per pin (two 8-byte __ldcg), this pays it once.
// Written as PTX rather than as __ldcg(const ulonglong2*) only so the build
// cannot depend on which vector overloads the toolkit header happens to
// declare; the emitted instruction is identical.
static __device__ __forceinline__ ulonglong2 ff20k_ldcg_u2(const ulonglong2 *p)
{
    ulonglong2 r;
    asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%2];"
                 : "=l"(r.x), "=l"(r.y)
                 : "l"(p)
                 : "memory");
    return r;
}

// ---- generated: bit-sliced per-part counter primitives -------------------
// pc0..pc8 are 64 parallel counters (one bit-plane each). The plane count is
// chosen per degree class as the smallest P with (max representable value)
// 2^P - 1 >= (max count that class can reach), so the dropped carry-out of
// plane P-1 is provably never set.  Generated, not hand-written.

#define FF20K_PC_DECL \
    unsigned long long pc0 = 0ULL, pc1 = 0ULL, pc2 = 0ULL, pc3 = 0ULL, pc4 = 0ULL, \
                       pc5 = 0ULL, pc6 = 0ULL, pc7 = 0ULL, pc8 = 0ULL;

#define FF20K_PC_INC3(flags) do { \
    unsigned long long c = (flags), t; \
    t = pc0 & c; pc0 ^= c; c = t; \
    t = pc1 & c; pc1 ^= c; c = t; \
    pc2 ^= c; \
} while (0)

#define FF20K_PC_INC4(flags) do { \
    unsigned long long c = (flags), t; \
    t = pc0 & c; pc0 ^= c; c = t; \
    t = pc1 & c; pc1 ^= c; c = t; \
    t = pc2 & c; pc2 ^= c; c = t; \
    pc3 ^= c; \
} while (0)

#define FF20K_PC_ADD_SHFL4(off) do { \
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

#define FF20K_PC_ADD_SHFL5(off) do { \
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

#define FF20K_PC_ADD_SHFL6(off) do { \
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

#define FF20K_PC_ADD_SHFL7(off) do { \
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

#define FF20K_PC_ADD_SHFL9(off) do { \
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

#define FF20K_PC_GET3(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) )

#define FF20K_PC_GET4(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) )

#define FF20K_PC_GET5(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) )

#define FF20K_PC_GET6(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) )

#define FF20K_PC_GET7(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((pc6 >> (tp)) & 1ULL) << 6) )

#define FF20K_PC_GET9(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((pc6 >> (tp)) & 1ULL) << 6) | \
    ((int)((pc7 >> (tp)) & 1ULL) << 7) | \
    ((int)((pc8 >> (tp)) & 1ULL) << 8) )

// ---- lane-group butterflies -------------------------------------------------
// An xor-shuffle with offset < G stays inside the G-aligned lane group, so a
// full-warp mask is correct and every lane of the group ends up holding the
// group's reduction.  All 32 lanes of the warp execute these (the task's class
// is warp-uniform), which is what __shfl_xor_sync(0xffffffff, ...) requires.

// flags phase: the champion's associative (a,d) merge.
#define FF20K_FE_STEP(off) do {                                                   \
    const unsigned long long oa_ = __shfl_xor_sync(0xffffffffu, a, (off));     \
    const unsigned long long od_ = __shfl_xor_sync(0xffffffffu, d, (off));     \
    d = d | od_ | (a & oa_);                                                   \
    a = a | oa_;                                                               \
} while (0)

#define FF20K_FE_RED_1
#define FF20K_FE_RED_2   FF20K_FE_STEP(1);
#define FF20K_FE_RED_4   FF20K_FE_STEP(2); FF20K_FE_STEP(1);
#define FF20K_FE_RED_8   FF20K_FE_STEP(4); FF20K_FE_STEP(2); FF20K_FE_STEP(1);
#define FF20K_FE_RED_16  FF20K_FE_STEP(8); FF20K_FE_STEP(4); FF20K_FE_STEP(2); FF20K_FE_STEP(1);
#define FF20K_FE_RED_32  FF20K_FE_STEP(16); FF20K_FE_STEP(8); FF20K_FE_STEP(4); FF20K_FE_STEP(2); FF20K_FE_STEP(1);

// moves phase: OR (cand_mask), + (count_current_present, crit), bit-sliced +.
#define FF20K_MV_STEP(ADDM, off) do {                                             \
    cand_mask |= __shfl_xor_sync(0xffffffffu, cand_mask, (off));               \
    count_current_present +=                                                   \
        __shfl_xor_sync(0xffffffffu, count_current_present, (off));            \
    crit += __shfl_xor_sync(0xffffffffu, crit, (off));                         \
    ADDM(off);                                                                 \
} while (0)

#define FF20K_MV_RED_1
#define FF20K_MV_RED_2   FF20K_MV_STEP(FF20K_PC_ADD_SHFL4, 1);
#define FF20K_MV_RED_4   FF20K_MV_STEP(FF20K_PC_ADD_SHFL5, 2); FF20K_MV_STEP(FF20K_PC_ADD_SHFL5, 1);
#define FF20K_MV_RED_8   FF20K_MV_STEP(FF20K_PC_ADD_SHFL6, 4); FF20K_MV_STEP(FF20K_PC_ADD_SHFL6, 2); \
                      FF20K_MV_STEP(FF20K_PC_ADD_SHFL6, 1);
#define FF20K_MV_RED_16  FF20K_MV_STEP(FF20K_PC_ADD_SHFL7, 8); FF20K_MV_STEP(FF20K_PC_ADD_SHFL7, 4); \
                      FF20K_MV_STEP(FF20K_PC_ADD_SHFL7, 2); FF20K_MV_STEP(FF20K_PC_ADD_SHFL7, 1);
#define FF20K_MV_RED_32  FF20K_MV_STEP(FF20K_PC_ADD_SHFL9, 16); FF20K_MV_STEP(FF20K_PC_ADD_SHFL9, 8); \
                      FF20K_MV_STEP(FF20K_PC_ADD_SHFL9, 4); FF20K_MV_STEP(FF20K_PC_ADD_SHFL9, 2); \
                      FF20K_MV_STEP(FF20K_PC_ADD_SHFL9, 1);

// ---- per-pin bodies, factored out so the batched gather can hoist the loads --
// FF20K_FE_ACC is the champion's flags accumulation for ONE pin, taking the pin's
// already-loaded partition value (or -1 when the pin/node was out of range, in
// which case the unsigned test rejects it exactly as the champion's two nested
// range tests did).
#define FF20K_GIANT_BATCH 16

#define FF20K_FE_ACC(P) do {                                                      \
    const int part_ = (P);                                                     \
    if ((unsigned)part_ < 64u) {                                               \
        const unsigned long long bit_ = 1ULL << part_;                         \
        d |= (a & bit_);                                                       \
        a |= bit_;                                                             \
    }                                                                          \
} while (0)

// FF20K_MV_PIN is the champion's moves accumulation for ONE pin, taking the pin's
// already-loaded (flags_all, flags_double) pair.  Token-for-token the champion's
// loop body from `const unsigned long long flags_all` down to `INCM(flags)`.
#define FF20K_MV_PIN(FP, INCM) do {                                               \
    const unsigned long long fa_ = (FP).x;                                     \
    const unsigned long long fd_ = (FP).y;                                     \
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
} while (0)


// ===========================================================================
// wave6_kmicro4 (W1/W2/W3): SLOT-MAJOR warp tasks + descriptor prefetch +
// a CSR prefetch that is issued in parallel with partition[node].
//
// kmicro3 measured (probe8, per fused call, 4 workers): flags+phase0 5.3 us,
// b1 1.5, moves 20.0, b2 1.3, passA 1.0, b3 1.0, passB1 3.0, b4 1.0,
// passB2 1.5  =>  ~36 us.  At 8 warps/SM (the grid-barrier residency cap)
// there is no TLP to hide latency with, so a phase costs
//   (number of SERIALIZED dependent global loads on the critical path)
//   x (effective latency).
// kmicro3 removed the serialization WITHIN a work item's pin walk (quad
// batching).  What is left is the serialization BEFORE it:
//
//   champion moves, per warp task:
//     1. mv_tasks[t]                    (task descriptor, broadcast)
//     2. mv_items[base_item + lane>>SH] (item record; address needs #1)
//     3. partition[node]                (address needs #2; GATES the loop)
//     4. node_hyperedges[nstart + r]    (inside the gate => after #3)
//     5. edge_flags_pair[h]             (address needs #4)
//   = 5 serialized latencies per task, and with ~716 tasks over 628 resident
//   warps the critical warp runs TWO tasks => ~10 latencies for the phase.
//
// wave6 removes three of the five and prefetches the fourth:
//   (W1) SLOT-MAJOR LAYOUT.  The host emits, for every warp task t and every
//        lane l, the int4 record {id, csr_start, size, class} that lane l of
//        task t works on (duplicated across the G lanes of a lane group).  The
//        kernel reads slots[t*32 + lane]: a fully coalesced 512-B load whose
//        address depends on NOTHING, so steps 1 and 2 collapse into one.
//        The class comes out of the same load (.w), so the switch is still
//        warp-uniform (a task is class-homogeneous by construction).
//   (W2) DESCRIPTOR PREFETCH.  The first task's slot record is loaded at the
//        very top of the kernel (both phases), so it overlaps phase 0, the
//        giant loop and grid barrier 1; inside the task loop the NEXT task's
//        record is issued before the current task's body.  Cost of step 1-2
//        becomes zero for every iteration.
//   (W3) CSR PREFETCH ACROSS THE GATE.  node_hyperedges[nstart + r] depends
//        only on the item record, never on partition[node]; the champion put
//        it behind the `current_part` gate purely as a side effect of the
//        source order.  The first quad's CSR indices are now loaded BEFORE
//        the gate, in parallel with partition[node], so steps 3 and 4 issue
//        together.  In bounds whenever node_degree > 0 (see the argument in
//        NOTES.md section 2).
//
//     moves, per warp task:  5 latencies -> 2   (slot prefetched, {partition,
//                            CSR} together, then the flag gather)
//     phase-1b flags:        4 -> 2
//     critical warp (2 tasks):  10 -> 4  (moves),  8 -> 4  (flags)
//
// Nothing about the per-pin bodies, the class table, the plane budgets, the
// butterflies, the lottery or the compaction changes.
// ===========================================================================

// ---- one hyperedge per lane group (flags phase) -----------------------------
// G lanes, SH = log2(G), RED = the matching butterfly.  `im_` (the lane's item
// record) and `lane` come from the enclosing task loop; the champion loaded
// `im_` here from fe_items[base_item + (lane >> SH)], wave6 hands it in
// already-loaded (W1/W2).  Everything else is kmicro3's body verbatim: the
// per-lane walk is QUAD-BATCHED (K1), so a lane group's whole pin list costs
// TWO dependent memory latencies instead of 2 x (trips).
#define FF20K_FE_BODY(G, SH, RED)                                                 \
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
                            ? __ldg(&partition[n0_]) : -1;                     \
        const int pa_ = ((unsigned)na_ < (unsigned)num_nodes)                  \
                            ? __ldg(&partition[na_]) : -1;                     \
        const int pb_ = ((unsigned)nb_ < (unsigned)num_nodes)                  \
                            ? __ldg(&partition[nb_]) : -1;                     \
        const int pc_ = ((unsigned)nc_ < (unsigned)num_nodes)                  \
                            ? __ldg(&partition[nc_]) : -1;                     \
        FF20K_FE_ACC(p0_);                                                        \
        FF20K_FE_ACC(pa_);                                                        \
        FF20K_FE_ACC(pb_);                                                        \
        FF20K_FE_ACC(pc_);                                                        \
    }                                                                          \
    RED                                                                        \
    if (lg_ == 0) {                                                            \
        ulonglong2 v_; v_.x = a; v_.y = d;                                     \
        edge_flags_pair[hedge] = v_;                                           \
    }                                                                          \
}

// ---- moves phase: one quad of CSR indices ----------------------------------
// Loads the four `node_hyperedges` entries for pin slots j_, j_+G, j_+2G,
// j_+3G into h0_..h3_ and their validity into v1_..v3_ (slot 0 is valid by the
// caller's contract).  `j_`, `used_degree`, `capped_`, `node_degree`, `nstart`
// come from the enclosing FF20K_MV_BODY scope.
//
// `rel` is kmicro3's, unchanged: node_degree <= 256 => used_degree ==
// node_degree and (j*d)/d == j exactly; node_degree > 256 => the divisor is
// the literal 256 and the division is an exact >> 8 (both operands are
// non-negative and j <= 255, node_degree <= num_hyperedges, so the product is
// < 2^23 and it is computed in long long anyway).
#define FF20K_MV_CSR(G) do {                                                      \
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

// ---- one node per lane group (moves phase) ----------------------------------
// G lanes, SH = log2(G), INCM/GETM the plane-truncated bit-sliced primitives
// for this class, RED the matching butterfly.  `im_` is handed in by the task
// loop (W1/W2).  Everything from `if (crit>255)` down is the champion's
// phase-2a body verbatim, executed redundantly (and therefore identically) by
// all G lanes of the group.
//
// wave6 (W3): the loop is ROTATED so the first quad's CSR loads sit ABOVE the
// `current_part` gate and issue in parallel with partition[node].  Coverage is
// unchanged: the champion visits j in {lg_ + m*G : m >= 0, j < used_degree} in
// increasing m; here j_ starts at lg_, each pass processes j_, j_+G, j_+2G,
// j_+3G individually guarded by `< used_degree`, then j_ += 4G and the pass
// repeats iff j_ < used_degree -- the same set in the same order.
#define FF20K_MV_BODY(G, SH, INCM, GETM, RED)                                     \
{                                                                              \
    const int lg_ = (int)(lane & (unsigned)((G) - 1));                         \
    const int node = im_.x;                                                    \
    const int nstart = im_.y;                                                  \
    const int node_degree = im_.z;                                             \
    int outkey = FF20K_SENT;                                                      \
    const int used_degree = node_degree > 256 ? 256 : node_degree;             \
    const bool capped_ = (node_degree > 256);                                  \
    const bool go_ = (node_degree > 0) && (lg_ < used_degree);                 \
    int j_ = lg_;                                                              \
    int h0_ = 0, h1_ = 0, h2_ = 0, h3_ = 0;                                    \
    bool v1_ = false, v2_ = false, v3_ = false;                                \
    if (go_) { FF20K_MV_CSR(G); }                                                 \
    const int current_part = __ldg(&partition[node]);                          \
    if ((unsigned)current_part < (unsigned)num_parts &&                        \
        shared_nodes_in_part[current_part] > 1 &&                              \
        node_degree > 0) {                                                     \
        const unsigned long long current_bit = 1ULL << current_part;           \
        FF20K_PC_DECL                                                             \
        unsigned long long cand_mask = 0ULL;                                   \
        int count_current_present = 0;                                         \
        int crit = 0;                                                          \
        if (go_) {                                                             \
            for (;;) {                                                         \
                ulonglong2 f1_; f1_.x = 0ULL; f1_.y = 0ULL;                    \
                ulonglong2 f2_; f2_.x = 0ULL; f2_.y = 0ULL;                    \
                ulonglong2 f3_; f3_.x = 0ULL; f3_.y = 0ULL;                    \
                const ulonglong2 f0_ = ff20k_ldcg_u2(&edge_flags_pair[h0_]);      \
                if (v1_) f1_ = ff20k_ldcg_u2(&edge_flags_pair[h1_]);              \
                if (v2_) f2_ = ff20k_ldcg_u2(&edge_flags_pair[h2_]);              \
                if (v3_) f3_ = ff20k_ldcg_u2(&edge_flags_pair[h3_]);              \
                FF20K_MV_PIN(f0_, INCM);                                          \
                if (v1_) { FF20K_MV_PIN(f1_, INCM); }                             \
                if (v2_) { FF20K_MV_PIN(f2_, INCM); }                             \
                if (v3_) { FF20K_MV_PIN(f3_, INCM); }                             \
                j_ += 4 * (G);                                                 \
                if (j_ >= used_degree) break;                                  \
                FF20K_MV_CSR(G);                                                  \
            }                                                                  \
        }                                                                      \
        RED                                                                    \
        if (crit > 255) crit = 255;                                            \
        int degree_weight = node_degree > 255 ? 255 : node_degree;             \
        int rank_byte = degree_weight + crit;                                  \
        if (rank_byte > 255) rank_byte = 255;                                  \
        int best_gain = -999999;                                               \
        int best_target = current_part;                                        \
        while (cand_mask) {                                                    \
            int target_part = __ffsll(cand_mask) - 1;                          \
            cand_mask &= (cand_mask - 1);                                      \
            if ((unsigned)target_part >= (unsigned)num_parts) continue;        \
            if (shared_nodes_in_part[target_part] >= max_part_size) continue;  \
            int pcount = GETM(target_part);                                    \
            int basic_gain = pcount - count_current_present;                   \
            int current_size = shared_nodes_in_part[current_part];             \
            int target_size = shared_nodes_in_part[target_part];               \
            int balance_bonus = (current_size > target_size + 2) ? 1 : 0;      \
            int total_gain = basic_gain + balance_bonus;                       \
            if (total_gain > best_gain ||                                      \
                (total_gain == best_gain && target_part < best_target)) {      \
                best_gain = total_gain;                                        \
                best_target = target_part;                                     \
            }                                                                  \
        }                                                                      \
        if (best_target != current_part) {                                     \
            int bg = best_gain > 32767 ? 32767 : best_gain;                    \
            if (bg < -32768) bg = -32768;                                      \
            unsigned short t16 =                                               \
                (unsigned short)(best_target & 63) | ((node & 3) << 6);        \
            outkey = ((int)(short)bg << 16) | (rank_byte << 8) | t16;          \
        }                                                                      \
    }                                                                          \
    if (lg_ == 0) {                                                            \
        move_priorities[node] = outkey;                                        \
        if (outkey != FF20K_SENT) {                                               \
            const int g_ = outkey >> 16;                                       \
            if (g_ > blk_max) blk_max = g_;                                    \
        }                                                                      \
    }                                                                          \
}

extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_filter_20k(
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
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[16];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;

    // ---- W2: the two task-list descriptor loads and the phase-0 mark load are
    // issued at the very top.  All three are read-only for the whole kernel
    // (fe_slots / mv_slots / mark_nodes are never written here), their
    // addresses depend on nothing, and they are consumed much later -- the
    // moves descriptor not until after grid barrier 1 -- so these three
    // latencies are completely hidden behind phase 0, the giant loop, phase 1b
    // and barrier 1.
    int4 fim_; fim_.x = 0; fim_.y = 0; fim_.z = 0; fim_.w = 0;
    if (warp_global < n_fe_tasks) {
        fim_ = __ldg(&fe_slots[warp_global * 32 + (int)lane]);
    }
    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }
    const int mark0_ = (gtid < n_mark_a) ? mark_nodes[gtid] : -1;

    // ---- phase 0: reset max-gain -------------------------------------------
    if (gtid == 0) {
        max_gain_buf[0] = FF20K_MG_INIT;
    }

    // ---- phase 1a: GIANT hyperedges (> 256 pins), one BLOCK each -----------
    // n_fe_giant / gridDim.x / blockIdx.x are block-uniform, so every thread of
    // the block reaches the two __syncthreads() the same number of times.
    for (int i = (int)blockIdx.x; i < n_fe_giant; i += (int)gridDim.x) {
        const int4 im = __ldg(&fe_giant[i]);
        const int hedge = im.x;
        const int hstart = im.y;
        const int hsz = im.z;
        unsigned long long a = 0ULL, d = 0ULL;
        // kmicro3 (K2): the giant walk is batched 16 deep -- 16 coalesced
        // hyperedge_nodes loads, then 16 scattered partition gathers, then 16
        // register updates, i.e. TWO dependent latencies per 16 pins instead of
        // 32 for a 1,954-pin giant.  Bounds are compile-time constants so the
        // arrays stay in registers.
        for (int k0 = (int)threadIdx.x; k0 < hsz;
             k0 += (int)blockDim.x * FF20K_GIANT_BATCH) {
            int gn_[FF20K_GIANT_BATCH];
            int gp_[FF20K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                FF20K_FE_ACC(gp_[m]);
            }
        }
        FF20K_FE_RED_32
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

    // ---- phase 0 (cont): scatter tabu mark list A --------------------------
    // The champion did this before the giant loop; the value it needs
    // (mark_nodes[gtid]) is loaded at the top of the kernel instead, so the
    // load overlaps the giant loop and only the store is left here.  Nothing
    // in between reads or writes tabu_until, and list A must merely be visible
    // before grid barrier 1 (list B, which overwrites shared nodes, is applied
    // after it).  `(unsigned)n < (unsigned)num_nodes` is `n >= 0 && n <
    // num_nodes` for num_nodes >= 0, and the -1 sentinel is rejected by it.
    if ((unsigned)mark0_ < (unsigned)num_nodes) tabu_until[mark0_] = until_a;
    for (int i = gtid + stride; i < n_mark_a; i += stride) {
        const int n = mark_nodes[i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_a;
    }

    // ---- phase 1b: all other hyperedges, lane groups over warp tasks -------
    // W1/W2: the lane's item record comes straight from fe_slots[t*32+lane]
    // (one coalesced 512-B load per warp, no dependency on a task descriptor),
    // and the NEXT task's record is issued before the current task's body.
    for (int t = warp_global; t < n_fe_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_fe_tasks) {
            nx_ = __ldg(&fe_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = fim_;
        switch (im_.w) {
            case 0:  FF20K_FE_BODY(1,  0, FF20K_FE_RED_1)  break;
            case 1:  FF20K_FE_BODY(2,  1, FF20K_FE_RED_2)  break;
            case 2:  FF20K_FE_BODY(4,  2, FF20K_FE_RED_4)  break;
            case 3:  FF20K_FE_BODY(8,  3, FF20K_FE_RED_8)  break;
            case 4:  FF20K_FE_BODY(16, 4, FF20K_FE_RED_16) break;
            default: FF20K_FE_BODY(32, 5, FF20K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF20K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF20K_MV_BODY(1,  0, FF20K_PC_INC3, FF20K_PC_GET3, FF20K_MV_RED_1)  break;
            case 1:  FF20K_MV_BODY(2,  1, FF20K_PC_INC3, FF20K_PC_GET4, FF20K_MV_RED_2)  break;
            case 2:  FF20K_MV_BODY(4,  2, FF20K_PC_INC3, FF20K_PC_GET5, FF20K_MV_RED_4)  break;
            case 3:  FF20K_MV_BODY(8,  3, FF20K_PC_INC3, FF20K_PC_GET6, FF20K_MV_RED_8)  break;
            case 4:  FF20K_MV_BODY(16, 4, FF20K_PC_INC3, FF20K_PC_GET7, FF20K_MV_RED_16) break;
            default: FF20K_MV_BODY(32, 5, FF20K_PC_INC4, FF20K_PC_GET9, FF20K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff20k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF20K_MG_INIT if there are none
    // (then no node passes the filter and the host breaks out, so the
    // aspiration value is never observed).
    const int mg = __ldcg(max_gain_buf);
    const int aspiration = (mg * 3) / 4;      // C and Rust both truncate toward 0
    const int TILE = (int)blockDim.x;

    // ===================================================================
    // wave6 (W5): passes A, B1 and B2 FUSED into one tile loop.
    //
    // The champion ran three grid-stride loops over the same tiles: pass A
    // recomputed, for every node, exactly the predicate pass B1 recomputes
    // (`!is_tabu && gain in [-3,0]`) only to publish a per-tile count; and
    // pass B2 re-read from `filt` what pass B1 had just computed.  Because a
    // block owns the SAME tile in all three passes, all of that can live in
    // registers across the barriers: `key`, `gain`, `elig`, `posval` are
    // computed once, and `outkey`/`valid` never leave the register file.
    // That deletes one full pass over move_priorities+tabu_until (pass A) and
    // the `filt` store/load round trip, and the `filt` buffer entirely.
    //
    // The grid barriers are now INSIDE the loop, so the loop trip count must
    // be block-uniform or a block that ran out of tiles would never reach the
    // barrier (deadlock).  It is: `nrounds = ceil(ntiles / gridDim.x)` depends
    // only on kernel arguments and gridDim, and every block runs exactly
    // nrounds iterations, with `act` marking the (at most one) trailing
    // iteration in which a block has no tile.  The host adds 2 + 2*nrounds to
    // its barrier counter, computed from the same two numbers.
    //
    // Ordering is unchanged: for a tile T, the eligible prefix reads
    // blk_elig[i] for i < T and the valid prefix reads tile_valid[i] for
    // i < T.  Tiles with i < T are either owned by another block in the SAME
    // iteration (published before the same barrier) or by some block in an
    // EARLIER iteration (published before an earlier barrier).  With
    // ntiles <= gridDim.x (the case on the target GPU) there is exactly one
    // iteration and this is literally the champion's schedule.
    // ===================================================================
    const int nrounds = (ntiles + (int)gridDim.x - 1) / (int)gridDim.x;
    unsigned int btgt_ = barrier_base + 2u * gridDim.x;

    for (int it = 0; it < nrounds; it++) {
        const int tile = (int)blockIdx.x + it * (int)gridDim.x;
        const bool act = (tile < ntiles);
        // a block with no tile this iteration uses an out-of-range node, so
        // every guarded read/write below is skipped exactly as it would be for
        // a tail node of the last tile.
        const int node = act ? (tile * TILE + (int)threadIdx.x) : num_nodes;
        const int plim = act ? tile : 0;

        // kmicro3 (K3): key and tabu stamp are issued back to back; the tabu
        // value is never used when key == FF20K_SENT and tabu_until[node] is in
        // range for every node < num_nodes.
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;

        int key = 0, gain = 0, elig = 0, posval = 0;
        if (node < num_nodes) {
            key = key_ld;
            if (key != FF20K_SENT) {
                gain = key >> 16;
                const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
                if (!is_tabu) {
                    if (gain > 0) posval = 1;
                    else if (gain >= -3) elig = 1;
                }
            }
        }

        // pass A's per-tile count and pass B1's intra-tile eligible rank come
        // from ONE ballot scan (the champion computed the count with a separate
        // ff20k_block_sum in pass A and the rank with this scan in pass B1; the
        // scan already returns both).
        int etot = 0;
        const int erank = ff20k_block_scan(elig, s_warp, &etot);
        if (act && threadIdx.x == 0) blk_elig[tile] = etot;

        // ---- barrier (was barrier 3) --------------------------------------
        btgt_ += (unsigned int)gridDim.x;
        FF20K_GRID_BARRIER(grid_barrier, btgt_);

        int partsum = 0;
        for (int i = (int)threadIdx.x; i < plim; i += (int)blockDim.x) {
            partsum += __ldcg(&blk_elig[i]);
        }
        const int baseE = ff20k_block_sum(partsum, s_warp);

        int outkey = key;
        int accept = 0;
        if (elig) {
            // the host's rng_state is advanced ONCE per eligible node in node
            // order and tested AFTER the advance => the r-th (0-based) eligible
            // node tests f^(r+1)(x0). Tiles are consecutive node ranges and the
            // intra-tile rank is a threadIdx-ordered ballot prefix, so
            // baseE + erank IS the global 0-based eligible rank.
            const unsigned long long n = (unsigned long long)(baseE + erank) + 1ULL;
            const unsigned long long x = ff20k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain); // gain in [-3,0]
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key & 0xFFFF;      // host: (0 << 16) | (key & 0xFFFF)
            }
        }

        const int valid = (posval | accept);

        int vtot = 0;
        const int vrank = ff20k_block_scan(valid, s_warp, &vtot);
        if (act && threadIdx.x == 0) tile_valid[tile] = vtot;

        // ---- barrier (was barrier 4) --------------------------------------
        btgt_ += (unsigned int)gridDim.x;
        FF20K_GRID_BARRIER(grid_barrier, btgt_);

        int partsum2 = 0;
        for (int i = (int)threadIdx.x; i < plim; i += (int)blockDim.x) {
            partsum2 += __ldcg(&tile_valid[i]);
        }
        const int base = ff20k_block_sum(partsum2, s_warp);

        if (valid) {
            const int slot = base + vrank;
            cand[1 + 2 * slot] = node;
            cand[2 + 2 * slot] = outkey;
        }
    }

    // total candidate count for the host's single prefix D2H
    if (blockIdx.x == 0) {
        int s = 0;
        for (int i = (int)threadIdx.x; i < ntiles; i += (int)blockDim.x) {
            s += __ldcg(&tile_valid[i]);
        }
        const int t = ff20k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = t;
    }
}


// ===========================================================================
// wave13_parity: the FLAT filter kernels.
//
// Body: `fused_flags_moves_filter_20k`'s prologue VERBATIM (phase 0, the
// giant hyperedges, mark list A, phase 1b, barrier 1, mark list B, phase 2b,
// the per-block max-gain atomicMax, barrier 2 and the aspiration) followed by
// the FLAT filter instead of wave6's fused tile loop.  Everything above the
// filter -- including every per-track move semantic -- is character for
// character the shipped kernel, because this script copies it out of it.
//
//   ..._flat_20k   __launch_bounds__(128, 2)   host `fused_mode = 1`
//   ..._flatw_20k  __launch_bounds__(256, 2)   host `fused_mode = 2`
//
// s_ab[16] is 2 entries per warp and 256 threads is 8 warps, so it is exactly
// full at 256 and its "blockDim.x <= 256" bound still holds.  s_warp[32]
// covers up to 1024 threads.
// ===========================================================================

extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_filter_flat_20k(
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
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[16];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;

    // ---- W2: the two task-list descriptor loads and the phase-0 mark load are
    // issued at the very top.  All three are read-only for the whole kernel
    // (fe_slots / mv_slots / mark_nodes are never written here), their
    // addresses depend on nothing, and they are consumed much later -- the
    // moves descriptor not until after grid barrier 1 -- so these three
    // latencies are completely hidden behind phase 0, the giant loop, phase 1b
    // and barrier 1.
    int4 fim_; fim_.x = 0; fim_.y = 0; fim_.z = 0; fim_.w = 0;
    if (warp_global < n_fe_tasks) {
        fim_ = __ldg(&fe_slots[warp_global * 32 + (int)lane]);
    }
    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }
    const int mark0_ = (gtid < n_mark_a) ? mark_nodes[gtid] : -1;

    // ---- phase 0: reset max-gain -------------------------------------------
    if (gtid == 0) {
        max_gain_buf[0] = FF20K_MG_INIT;
    }

    // ---- phase 1a: GIANT hyperedges (> 256 pins), one BLOCK each -----------
    // n_fe_giant / gridDim.x / blockIdx.x are block-uniform, so every thread of
    // the block reaches the two __syncthreads() the same number of times.
    for (int i = (int)blockIdx.x; i < n_fe_giant; i += (int)gridDim.x) {
        const int4 im = __ldg(&fe_giant[i]);
        const int hedge = im.x;
        const int hstart = im.y;
        const int hsz = im.z;
        unsigned long long a = 0ULL, d = 0ULL;
        // kmicro3 (K2): the giant walk is batched 16 deep -- 16 coalesced
        // hyperedge_nodes loads, then 16 scattered partition gathers, then 16
        // register updates, i.e. TWO dependent latencies per 16 pins instead of
        // 32 for a 1,954-pin giant.  Bounds are compile-time constants so the
        // arrays stay in registers.
        for (int k0 = (int)threadIdx.x; k0 < hsz;
             k0 += (int)blockDim.x * FF20K_GIANT_BATCH) {
            int gn_[FF20K_GIANT_BATCH];
            int gp_[FF20K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                FF20K_FE_ACC(gp_[m]);
            }
        }
        FF20K_FE_RED_32
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

    // ---- phase 0 (cont): scatter tabu mark list A --------------------------
    // The champion did this before the giant loop; the value it needs
    // (mark_nodes[gtid]) is loaded at the top of the kernel instead, so the
    // load overlaps the giant loop and only the store is left here.  Nothing
    // in between reads or writes tabu_until, and list A must merely be visible
    // before grid barrier 1 (list B, which overwrites shared nodes, is applied
    // after it).  `(unsigned)n < (unsigned)num_nodes` is `n >= 0 && n <
    // num_nodes` for num_nodes >= 0, and the -1 sentinel is rejected by it.
    if ((unsigned)mark0_ < (unsigned)num_nodes) tabu_until[mark0_] = until_a;
    for (int i = gtid + stride; i < n_mark_a; i += stride) {
        const int n = mark_nodes[i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_a;
    }

    // ---- phase 1b: all other hyperedges, lane groups over warp tasks -------
    // W1/W2: the lane's item record comes straight from fe_slots[t*32+lane]
    // (one coalesced 512-B load per warp, no dependency on a task descriptor),
    // and the NEXT task's record is issued before the current task's body.
    for (int t = warp_global; t < n_fe_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_fe_tasks) {
            nx_ = __ldg(&fe_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = fim_;
        switch (im_.w) {
            case 0:  FF20K_FE_BODY(1,  0, FF20K_FE_RED_1)  break;
            case 1:  FF20K_FE_BODY(2,  1, FF20K_FE_RED_2)  break;
            case 2:  FF20K_FE_BODY(4,  2, FF20K_FE_RED_4)  break;
            case 3:  FF20K_FE_BODY(8,  3, FF20K_FE_RED_8)  break;
            case 4:  FF20K_FE_BODY(16, 4, FF20K_FE_RED_16) break;
            default: FF20K_FE_BODY(32, 5, FF20K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF20K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF20K_MV_BODY(1,  0, FF20K_PC_INC3, FF20K_PC_GET3, FF20K_MV_RED_1)  break;
            case 1:  FF20K_MV_BODY(2,  1, FF20K_PC_INC3, FF20K_PC_GET4, FF20K_MV_RED_2)  break;
            case 2:  FF20K_MV_BODY(4,  2, FF20K_PC_INC3, FF20K_PC_GET5, FF20K_MV_RED_4)  break;
            case 3:  FF20K_MV_BODY(8,  3, FF20K_PC_INC3, FF20K_PC_GET6, FF20K_MV_RED_8)  break;
            case 4:  FF20K_MV_BODY(16, 4, FF20K_PC_INC3, FF20K_PC_GET7, FF20K_MV_RED_16) break;
            default: FF20K_MV_BODY(32, 5, FF20K_PC_INC4, FF20K_PC_GET9, FF20K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff20k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF20K_MG_INIT if there are none
    // (then no node passes the filter and the host breaks out, so the
    // aspiration value is never observed).
    const int mg = __ldcg(max_gain_buf);
    const int aspiration = (mg * 3) / 4;      // C and Rust both truncate toward 0
    const int TILE = (int)blockDim.x;


    // ===================================================================
    // wave13_parity (F1): the FLAT filter -- three passes, EXACTLY two
    // barriers, block-CONTIGUOUS tile ranges.
    //
    // wave6 (W5) fused passes A / B1 / B2 into ONE tile loop and moved the two
    // grid barriers INSIDE it, so a launch issues 2 + 2*nrounds barriers with
    // nrounds = ceil(ntiles / gridDim.x), and the per-tile prefix loops read
    // blk_elig[0..tile) and tile_valid[0..tile) -- O(ntiles^2) loads per
    // launch.  On the 20k that is free: ntiles = ceil(18400/128) = 144 <=
    // gridDim = 157, so nrounds = 1, four barriers, ~20 k prefix loads.  On the
    // larger tracks ntiles is 360 / 719 / 1438 against a grid that saturates at
    // 2*SM = 164, so wave6 costs 8 / 12 / 20 barriers and 0.13 / 0.52 / 2.07 M
    // prefix loads per round.
    //
    // Here block b owns the CONTIGUOUS tile range
    //     [t0, t1) = [b*tpb, min((b+1)*tpb, ntiles)),  tpb = ceil(ntiles/grid)
    // walks it three times and publishes ONE per-BLOCK total per pass, so the
    // two prefixes are over gridDim.x entries and the barrier count is 2 (plus
    // the two before the filter) for every track, every GPU and every ntiles.
    //
    // EXACTNESS.  The filter emits exactly two derived quantities:
    //   (a) the LOTTERY INDEX of an eligible node = the number of eligible
    //       nodes with a strictly smaller node id, and
    //   (b) the CANDIDATE SLOT of a valid node   = the number of valid nodes
    //       with a strictly smaller node id.
    // Tile t owns nodes [t*TILE, (t+1)*TILE) and inside a tile the ballot scan
    // ranks by threadIdx.x, i.e. by node id.  Blocks own disjoint, INCREASING
    // tile ranges, so for a node in tile t of block b
    //     count(< node) = SUM over blocks b' < b of that block's total
    //                   + SUM over this block's own tiles t' in [t0, t)
    //                   + the intra-tile ballot rank,
    // which is exactly (per-block prefix) + (running accumulator) + (rank).
    // Both totals are integer sums, so the reduction order is irrelevant.
    //
    // Passes B and C RECOMPUTE the per-node predicate instead of carrying it in
    // registers across the barriers (wave6's saving, which is only available
    // when the whole tile range fits in one iteration).  That is exact:
    // `move_priorities` and `tabu_until` are not written by anything after
    // barrier 2, so all three passes read the same bytes.  It costs two extra
    // L2-resident i32 loads per node per pass and one extra ff20k_lcg_advance per
    // eligible node -- a few microseconds against the 4 / 8 / 16 barriers and
    // the O(ntiles^2) traffic it removes.
    //
    // The loops contain NO grid barrier, so a block with an empty tile range
    // (possible only when gridDim > ntiles) cannot deadlock; t1 is clamped to
    // t0 for exactly that case.
    // ===================================================================
    const int tpb = (ntiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int t0 = (int)blockIdx.x * tpb;
    int t1_ = t0 + tpb;
    if (t1_ > ntiles) t1_ = ntiles;
    if (t1_ < t0) t1_ = t0;
    const int t1 = t1_;

    // ---- pass A: this block's total number of lottery-eligible nodes -------
    int blkE = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int elig = 0;
        if (node < num_nodes && key_ld != FF20K_SENT) {
            const int gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu && gain <= 0 && gain >= -3) elig = 1;
        }
        blkE += ff20k_block_sum(elig, s_warp);
    }
    if (threadIdx.x == 0) blk_elig[blockIdx.x] = blkE;

    // ---- barrier 3 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 3u * gridDim.x);

    int psE_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psE_ += __ldcg(&blk_elig[i]);
    }
    const int baseE_block = ff20k_block_sum(psE_, s_warp);

    // ---- pass B: the lottery, and this block's total number of valid nodes --
    int runE = 0;
    int blkV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF20K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff20k_block_scan(elig, s_warp, &etot);
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE + erank) + 1ULL;
            const unsigned long long x = ff20k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) accept = 1;
        }
        const int valid = (posval | accept);
        int vtot = 0;
        (void)ff20k_block_scan(valid, s_warp, &vtot);
        runE += etot;
        blkV += vtot;
    }
    if (threadIdx.x == 0) tile_valid[blockIdx.x] = blkV;

    // ---- barrier 4 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 4u * gridDim.x);

    int psV_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psV_ += __ldcg(&tile_valid[i]);
    }
    const int baseV_block = ff20k_block_sum(psV_, s_warp);

    // ---- pass C: emit the candidates ---------------------------------------
    int runE2 = 0;
    int runV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF20K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff20k_block_scan(elig, s_warp, &etot);
        int outkey = key_ld;
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE2 + erank) + 1ULL;
            const unsigned long long x = ff20k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key_ld & 0xFFFF;   // host: (0 << 16) | (key & 0xFFFF)
            }
        }
        const int valid = (posval | accept);
        int vtot = 0;
        const int vrank = ff20k_block_scan(valid, s_warp, &vtot);
        if (valid) {
            const int slot = baseV_block + runV + vrank;
            cand[1 + 2 * slot] = node;
            cand[2 + 2 * slot] = outkey;
        }
        runE2 += etot;
        runV += vtot;
    }

    // total candidate count for the host's single prefix D2H
    if (blockIdx.x == 0) {
        int s = 0;
        for (int i = (int)threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
            s += __ldcg(&tile_valid[i]);
        }
        const int tot = ff20k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = tot;
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void fused_flags_moves_filter_flatw_20k(
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
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[16];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;

    // ---- W2: the two task-list descriptor loads and the phase-0 mark load are
    // issued at the very top.  All three are read-only for the whole kernel
    // (fe_slots / mv_slots / mark_nodes are never written here), their
    // addresses depend on nothing, and they are consumed much later -- the
    // moves descriptor not until after grid barrier 1 -- so these three
    // latencies are completely hidden behind phase 0, the giant loop, phase 1b
    // and barrier 1.
    int4 fim_; fim_.x = 0; fim_.y = 0; fim_.z = 0; fim_.w = 0;
    if (warp_global < n_fe_tasks) {
        fim_ = __ldg(&fe_slots[warp_global * 32 + (int)lane]);
    }
    int4 mim_; mim_.x = 0; mim_.y = 0; mim_.z = 0; mim_.w = 0;
    if (warp_global < n_mv_tasks) {
        mim_ = __ldg(&mv_slots[warp_global * 32 + (int)lane]);
    }
    const int mark0_ = (gtid < n_mark_a) ? mark_nodes[gtid] : -1;

    // ---- phase 0: reset max-gain -------------------------------------------
    if (gtid == 0) {
        max_gain_buf[0] = FF20K_MG_INIT;
    }

    // ---- phase 1a: GIANT hyperedges (> 256 pins), one BLOCK each -----------
    // n_fe_giant / gridDim.x / blockIdx.x are block-uniform, so every thread of
    // the block reaches the two __syncthreads() the same number of times.
    for (int i = (int)blockIdx.x; i < n_fe_giant; i += (int)gridDim.x) {
        const int4 im = __ldg(&fe_giant[i]);
        const int hedge = im.x;
        const int hstart = im.y;
        const int hsz = im.z;
        unsigned long long a = 0ULL, d = 0ULL;
        // kmicro3 (K2): the giant walk is batched 16 deep -- 16 coalesced
        // hyperedge_nodes loads, then 16 scattered partition gathers, then 16
        // register updates, i.e. TWO dependent latencies per 16 pins instead of
        // 32 for a 1,954-pin giant.  Bounds are compile-time constants so the
        // arrays stay in registers.
        for (int k0 = (int)threadIdx.x; k0 < hsz;
             k0 += (int)blockDim.x * FF20K_GIANT_BATCH) {
            int gn_[FF20K_GIANT_BATCH];
            int gp_[FF20K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF20K_GIANT_BATCH; m++) {
                FF20K_FE_ACC(gp_[m]);
            }
        }
        FF20K_FE_RED_32
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

    // ---- phase 0 (cont): scatter tabu mark list A --------------------------
    // The champion did this before the giant loop; the value it needs
    // (mark_nodes[gtid]) is loaded at the top of the kernel instead, so the
    // load overlaps the giant loop and only the store is left here.  Nothing
    // in between reads or writes tabu_until, and list A must merely be visible
    // before grid barrier 1 (list B, which overwrites shared nodes, is applied
    // after it).  `(unsigned)n < (unsigned)num_nodes` is `n >= 0 && n <
    // num_nodes` for num_nodes >= 0, and the -1 sentinel is rejected by it.
    if ((unsigned)mark0_ < (unsigned)num_nodes) tabu_until[mark0_] = until_a;
    for (int i = gtid + stride; i < n_mark_a; i += stride) {
        const int n = mark_nodes[i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_a;
    }

    // ---- phase 1b: all other hyperedges, lane groups over warp tasks -------
    // W1/W2: the lane's item record comes straight from fe_slots[t*32+lane]
    // (one coalesced 512-B load per warp, no dependency on a task descriptor),
    // and the NEXT task's record is issued before the current task's body.
    for (int t = warp_global; t < n_fe_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_fe_tasks) {
            nx_ = __ldg(&fe_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = fim_;
        switch (im_.w) {
            case 0:  FF20K_FE_BODY(1,  0, FF20K_FE_RED_1)  break;
            case 1:  FF20K_FE_BODY(2,  1, FF20K_FE_RED_2)  break;
            case 2:  FF20K_FE_BODY(4,  2, FF20K_FE_RED_4)  break;
            case 3:  FF20K_FE_BODY(8,  3, FF20K_FE_RED_8)  break;
            case 4:  FF20K_FE_BODY(16, 4, FF20K_FE_RED_16) break;
            default: FF20K_FE_BODY(32, 5, FF20K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF20K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF20K_MV_BODY(1,  0, FF20K_PC_INC3, FF20K_PC_GET3, FF20K_MV_RED_1)  break;
            case 1:  FF20K_MV_BODY(2,  1, FF20K_PC_INC3, FF20K_PC_GET4, FF20K_MV_RED_2)  break;
            case 2:  FF20K_MV_BODY(4,  2, FF20K_PC_INC3, FF20K_PC_GET5, FF20K_MV_RED_4)  break;
            case 3:  FF20K_MV_BODY(8,  3, FF20K_PC_INC3, FF20K_PC_GET6, FF20K_MV_RED_8)  break;
            case 4:  FF20K_MV_BODY(16, 4, FF20K_PC_INC3, FF20K_PC_GET7, FF20K_MV_RED_16) break;
            default: FF20K_MV_BODY(32, 5, FF20K_PC_INC4, FF20K_PC_GET9, FF20K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff20k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF20K_MG_INIT if there are none
    // (then no node passes the filter and the host breaks out, so the
    // aspiration value is never observed).
    const int mg = __ldcg(max_gain_buf);
    const int aspiration = (mg * 3) / 4;      // C and Rust both truncate toward 0
    const int TILE = (int)blockDim.x;


    // ===================================================================
    // wave13_parity (F1): the FLAT filter -- three passes, EXACTLY two
    // barriers, block-CONTIGUOUS tile ranges.
    //
    // wave6 (W5) fused passes A / B1 / B2 into ONE tile loop and moved the two
    // grid barriers INSIDE it, so a launch issues 2 + 2*nrounds barriers with
    // nrounds = ceil(ntiles / gridDim.x), and the per-tile prefix loops read
    // blk_elig[0..tile) and tile_valid[0..tile) -- O(ntiles^2) loads per
    // launch.  On the 20k that is free: ntiles = ceil(18400/128) = 144 <=
    // gridDim = 157, so nrounds = 1, four barriers, ~20 k prefix loads.  On the
    // larger tracks ntiles is 360 / 719 / 1438 against a grid that saturates at
    // 2*SM = 164, so wave6 costs 8 / 12 / 20 barriers and 0.13 / 0.52 / 2.07 M
    // prefix loads per round.
    //
    // Here block b owns the CONTIGUOUS tile range
    //     [t0, t1) = [b*tpb, min((b+1)*tpb, ntiles)),  tpb = ceil(ntiles/grid)
    // walks it three times and publishes ONE per-BLOCK total per pass, so the
    // two prefixes are over gridDim.x entries and the barrier count is 2 (plus
    // the two before the filter) for every track, every GPU and every ntiles.
    //
    // EXACTNESS.  The filter emits exactly two derived quantities:
    //   (a) the LOTTERY INDEX of an eligible node = the number of eligible
    //       nodes with a strictly smaller node id, and
    //   (b) the CANDIDATE SLOT of a valid node   = the number of valid nodes
    //       with a strictly smaller node id.
    // Tile t owns nodes [t*TILE, (t+1)*TILE) and inside a tile the ballot scan
    // ranks by threadIdx.x, i.e. by node id.  Blocks own disjoint, INCREASING
    // tile ranges, so for a node in tile t of block b
    //     count(< node) = SUM over blocks b' < b of that block's total
    //                   + SUM over this block's own tiles t' in [t0, t)
    //                   + the intra-tile ballot rank,
    // which is exactly (per-block prefix) + (running accumulator) + (rank).
    // Both totals are integer sums, so the reduction order is irrelevant.
    //
    // Passes B and C RECOMPUTE the per-node predicate instead of carrying it in
    // registers across the barriers (wave6's saving, which is only available
    // when the whole tile range fits in one iteration).  That is exact:
    // `move_priorities` and `tabu_until` are not written by anything after
    // barrier 2, so all three passes read the same bytes.  It costs two extra
    // L2-resident i32 loads per node per pass and one extra ff20k_lcg_advance per
    // eligible node -- a few microseconds against the 4 / 8 / 16 barriers and
    // the O(ntiles^2) traffic it removes.
    //
    // The loops contain NO grid barrier, so a block with an empty tile range
    // (possible only when gridDim > ntiles) cannot deadlock; t1 is clamped to
    // t0 for exactly that case.
    // ===================================================================
    const int tpb = (ntiles + (int)gridDim.x - 1) / (int)gridDim.x;
    const int t0 = (int)blockIdx.x * tpb;
    int t1_ = t0 + tpb;
    if (t1_ > ntiles) t1_ = ntiles;
    if (t1_ < t0) t1_ = t0;
    const int t1 = t1_;

    // ---- pass A: this block's total number of lottery-eligible nodes -------
    int blkE = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int elig = 0;
        if (node < num_nodes && key_ld != FF20K_SENT) {
            const int gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu && gain <= 0 && gain >= -3) elig = 1;
        }
        blkE += ff20k_block_sum(elig, s_warp);
    }
    if (threadIdx.x == 0) blk_elig[blockIdx.x] = blkE;

    // ---- barrier 3 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 3u * gridDim.x);

    int psE_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psE_ += __ldcg(&blk_elig[i]);
    }
    const int baseE_block = ff20k_block_sum(psE_, s_warp);

    // ---- pass B: the lottery, and this block's total number of valid nodes --
    int runE = 0;
    int blkV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF20K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff20k_block_scan(elig, s_warp, &etot);
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE + erank) + 1ULL;
            const unsigned long long x = ff20k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) accept = 1;
        }
        const int valid = (posval | accept);
        int vtot = 0;
        (void)ff20k_block_scan(valid, s_warp, &vtot);
        runE += etot;
        blkV += vtot;
    }
    if (threadIdx.x == 0) tile_valid[blockIdx.x] = blkV;

    // ---- barrier 4 ---------------------------------------------------------
    FF20K_GRID_BARRIER(grid_barrier, barrier_base + 4u * gridDim.x);

    int psV_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psV_ += __ldcg(&tile_valid[i]);
    }
    const int baseV_block = ff20k_block_sum(psV_, s_warp);

    // ---- pass C: emit the candidates ---------------------------------------
    int runE2 = 0;
    int runV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF20K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff20k_block_scan(elig, s_warp, &etot);
        int outkey = key_ld;
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE2 + erank) + 1ULL;
            const unsigned long long x = ff20k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key_ld & 0xFFFF;   // host: (0 << 16) | (key & 0xFFFF)
            }
        }
        const int valid = (posval | accept);
        int vtot = 0;
        const int vrank = ff20k_block_scan(valid, s_warp, &vtot);
        if (valid) {
            const int slot = baseV_block + runV + vrank;
            cand[1 + 2 * slot] = node;
            cand[2 + 2 * slot] = outkey;
        }
        runE2 += etot;
        runV += vtot;
    }

    // total candidate count for the host's single prefix D2H
    if (blockIdx.x == 0) {
        int s = 0;
        for (int i = (int)threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
            s += __ldcg(&tile_valid[i]);
        }
        const int tot = ff20k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = tot;
    }
}
