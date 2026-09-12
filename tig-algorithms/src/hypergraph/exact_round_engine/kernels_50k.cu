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

    unsigned long long votes[9]={0,0,0,0,0,0,0,0,0};

    if (used_degree > 0) {
        for (int j = 0; j < used_degree; j++) {
            int rel = (int)(((long long)j * deg) / used_degree);
            int hedge = node_hyperedges[start + rel];

            int elite_id = hedge_choice_elite[hedge];
            long long idx = (long long)elite_id * (long long)num_nodes + (long long)node;
            int part = elite_partitions[idx];
            if ((unsigned)part < (unsigned)np) {
                unsigned long long carry=1ULL<<part;
                #pragma unroll
                for(int bit=0;bit<9;++bit){unsigned long long old=votes[bit];votes[bit]=old^carry;carry&=old;}

            }
        }
    }

    unsigned long long any_votes=0;
    #pragma unroll
    for(int bit=0;bit<9;++bit)any_votes|=votes[bit];
    unsigned long long cand_mask=any_votes;
    #pragma unroll
    for(int bit=8;bit>=0;--bit){unsigned long long better=cand_mask&votes[bit];if(better)cand_mask=better;}

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



extern "C" __global__ void precompute_edge_flags_50k(
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
            int node = __ldg(hyperedge_nodes + k);
            if (node >= 0 && node < num_nodes) {
                int part = __ldg(partition + node);
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
    int *move_priorities
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
        if (best_target != current_part) {
            int bg = best_gain > 32767 ? 32767 : best_gain;
            if (bg < -32768) bg = -32768;
            unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
            move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
        } else {
            move_priorities[node] = 0x80000000;
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

    int current_part = __ldg(partition + node);
    if ((unsigned)current_part >= (unsigned)num_parts) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) return;

    unsigned long long current_bit = 1ULL << current_part;

    const int np=min(num_parts,64);
    unsigned long long pc0=0ULL;
    unsigned long long pc1=0ULL;
    unsigned long long pc2=0ULL;
    unsigned long long pc3=0ULL;
    unsigned long long pc4=0ULL;
    unsigned long long pc5=0ULL;
    unsigned long long pc6=0ULL;
    unsigned long long pc7=0ULL;
    unsigned long long pc8=0ULL;
    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;

    for (int j = 0; j < used_degree; j++) {
        int rel = (int)(((long long)j * node_degree) / used_degree);
        int hyperedge = __ldg(node_hyperedges + (start + rel));

        unsigned long long flags_all = __ldg(edge_flags_all + hyperedge);
        unsigned long long flags_double = __ldg(edge_flags_double + hyperedge);
        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

        if (mask & current_bit) count_current_present++;

        unsigned long long flags = mask & ~current_bit;
        cand_mask |= flags;
        { unsigned long long carry=flags,next;
          next=pc0&carry;pc0^=carry;carry=next;
          next=pc1&carry;pc1^=carry;carry=next;
          next=pc2&carry;pc2^=carry;carry=next;
          next=pc3&carry;pc3^=carry;carry=next;
          next=pc4&carry;pc4^=carry;carry=next;
          next=pc5&carry;pc5^=carry;carry=next;
          next=pc6&carry;pc6^=carry;carry=next;
          next=pc7&carry;pc7^=carry;carry=next;
          next=pc8&carry;pc8^=carry;carry=next;
        }

    }

    int degree_scaled_thresh = neg_gain_thresh + node_degree / 20;
    int min_gain = -degree_scaled_thresh;

    unsigned long long active=cand_mask & (np==64?~0ULL:((1ULL<<np)-1ULL));
    for(int slot=0;slot<3 && active;slot++){
        unsigned long long pick=active;int count=0;
        { unsigned long long subset=pick&pc8;if(subset){pick=subset;count|=256;} }
        { unsigned long long subset=pick&pc7;if(subset){pick=subset;count|=128;} }
        { unsigned long long subset=pick&pc6;if(subset){pick=subset;count|=64;} }
        { unsigned long long subset=pick&pc5;if(subset){pick=subset;count|=32;} }
        { unsigned long long subset=pick&pc4;if(subset){pick=subset;count|=16;} }
        { unsigned long long subset=pick&pc3;if(subset){pick=subset;count|=8;} }
        { unsigned long long subset=pick&pc2;if(subset){pick=subset;count|=4;} }
        { unsigned long long subset=pick&pc1;if(subset){pick=subset;count|=2;} }
        { unsigned long long subset=pick&pc0;if(subset){pick=subset;count|=1;} }
        int target=__ffsll(pick)-1;
        int gain=count-count_current_present;
        if(gain<min_gain)break;
        int g=max(-32768,min(32767,gain));
        swap_gains[node*3+slot]=((int)(unsigned short)(short)g<<16)|target;
        active&=~(1ULL<<target);
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
            int node = __ldg(hyperedge_nodes + k);
            int part = __ldg(partition + node);
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


// ===========================================================================
// PORT (waves/port_tracks): the 20k `diverse` stack's fused kernels, applied
// to the 50k track.  Everything ABOVE this line is the ORIGINAL
// kernels_50k.cu, byte for byte.  Every kernel below carries the _50k suffix
// so one PTX can hold all five tracks.
// ===========================================================================


// ---- exp6: flags + moves in ONE launch. Phase 1 computes the per-hyperedge flags exactly as
// precompute_edge_flags_50k (once per hyperedge); a software grid barrier (monotonic counter,
// target = calls * gridDim.x, passed by the host) orders it before phase 2, which is
// compute_refinement_moves_optimized_50k verbatim in a grid-stride loop. Requires all blocks to be
// co-resident: host launches at most 2 blocks per SM (128 threads each).
extern "C" __global__ void fused_flags_moves_50k(
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

    // ---- phase 1a: small hyperedges, one thread each (identical to precompute_edge_flags_50k) ----
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

    // ---- phase 2: moves (identical to compute_refinement_moves_optimized_50k; flags read via L2) ----
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

        // --- 50k semantics: scan EVERY part, not only the parts that appear in
        // some incident hyperedge.  `pc*` holds count[p] for all 64 parts (0
        // where the part was never incremented), so a zero-count part gets
        // basic_gain = -count_current_present exactly as the original
        // warp-per-node kernel computes it.  `cand_mask` is still accumulated
        // (and reduced) but is no longer the iteration set.
        (void)cand_mask;
        const int np_ = (num_parts < 64) ? num_parts : 64;
        for (int target_part = 0; target_part < np_; target_part++) {
            if (target_part == current_part) continue;
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
// fused_flags_moves_filter_50k. The description above is kept because the
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

#define FF50K_SENT    ((int)0x80000000)
#define FF50K_MG_INIT (-1048576)

// f(x) = x*6364136223846793005 + 1442695040888963407  (mod 2^64)
// Returns f^n(x0). Affine composition by squaring:
//   f^m = (A,C): x -> A*x + C ; (Ab,Cb) o (A,C) = (Ab*A, Ab*C + Cb)
//   square: f^(2m) = f^m o f^m = (Ab*Ab, Ab*Cb + Cb)
// n <= num_nodes+1 (~18401) => at most 15 iterations.
static __device__ __forceinline__ unsigned long long ff50k_lcg_advance(
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
static __device__ __forceinline__ unsigned int ff50k_ld_vol_u32(const unsigned int *p)
{
    unsigned int r;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
    return r;
}

// Software grid barrier (monotonic counter, host-supplied absolute target).
#define FF50K_GRID_BARRIER(gb, target)                        \
    do {                                                   \
        __threadfence();                                   \
        __syncthreads();                                   \
        if (threadIdx.x == 0) {                            \
            atomicAdd((gb), 1u);                           \
            while (ff50k_ld_vol_u32(gb) < (target)) {         \
                __nanosleep(20);                          \
            }                                              \
        }                                                  \
        __syncthreads();                                   \
        __threadfence();                                   \
    } while (0)

// All three helpers are called by every thread of the block (uniform control
// flow) and use only integer +/max => order-independent => deterministic.
// Each leaves s_warp reusable (trailing __syncthreads).
static __device__ __forceinline__ int ff50k_block_sum(int v, int *s_warp)
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

static __device__ __forceinline__ int ff50k_block_max(int v, int *s_warp)
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
static __device__ __forceinline__ int ff50k_block_scan(int pred, int *s_warp, int *total_out)
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
static __device__ __forceinline__ ulonglong2 ff50k_ldcg_u2(const ulonglong2 *p)
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

#define FF50K_PC_DECL \
    unsigned long long pc0 = 0ULL, pc1 = 0ULL, pc2 = 0ULL, pc3 = 0ULL, pc4 = 0ULL, \
                       pc5 = 0ULL, pc6 = 0ULL, pc7 = 0ULL, pc8 = 0ULL;

#define FF50K_PC_INC3(flags) do { \
    unsigned long long c = (flags), t; \
    t = pc0 & c; pc0 ^= c; c = t; \
    t = pc1 & c; pc1 ^= c; c = t; \
    pc2 ^= c; \
} while (0)

#define FF50K_PC_INC4(flags) do { \
    unsigned long long c = (flags), t; \
    t = pc0 & c; pc0 ^= c; c = t; \
    t = pc1 & c; pc1 ^= c; c = t; \
    t = pc2 & c; pc2 ^= c; c = t; \
    pc3 ^= c; \
} while (0)

#define FF50K_PC_ADD_SHFL4(off) do { \
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

#define FF50K_PC_ADD_SHFL5(off) do { \
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

#define FF50K_PC_ADD_SHFL6(off) do { \
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

#define FF50K_PC_ADD_SHFL7(off) do { \
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

#define FF50K_PC_ADD_SHFL9(off) do { \
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

#define FF50K_PC_GET3(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) )

#define FF50K_PC_GET4(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) )

#define FF50K_PC_GET5(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) )

#define FF50K_PC_GET6(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) )

#define FF50K_PC_GET7(tp) ( \
    ((int)((pc0 >> (tp)) & 1ULL) << 0) | \
    ((int)((pc1 >> (tp)) & 1ULL) << 1) | \
    ((int)((pc2 >> (tp)) & 1ULL) << 2) | \
    ((int)((pc3 >> (tp)) & 1ULL) << 3) | \
    ((int)((pc4 >> (tp)) & 1ULL) << 4) | \
    ((int)((pc5 >> (tp)) & 1ULL) << 5) | \
    ((int)((pc6 >> (tp)) & 1ULL) << 6) )

#define FF50K_PC_GET9(tp) ( \
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
#define FF50K_FE_STEP(off) do {                                                   \
    const unsigned long long oa_ = __shfl_xor_sync(0xffffffffu, a, (off));     \
    const unsigned long long od_ = __shfl_xor_sync(0xffffffffu, d, (off));     \
    d = d | od_ | (a & oa_);                                                   \
    a = a | oa_;                                                               \
} while (0)

#define FF50K_FE_RED_1
#define FF50K_FE_RED_2   FF50K_FE_STEP(1);
#define FF50K_FE_RED_4   FF50K_FE_STEP(2); FF50K_FE_STEP(1);
#define FF50K_FE_RED_8   FF50K_FE_STEP(4); FF50K_FE_STEP(2); FF50K_FE_STEP(1);
#define FF50K_FE_RED_16  FF50K_FE_STEP(8); FF50K_FE_STEP(4); FF50K_FE_STEP(2); FF50K_FE_STEP(1);
#define FF50K_FE_RED_32  FF50K_FE_STEP(16); FF50K_FE_STEP(8); FF50K_FE_STEP(4); FF50K_FE_STEP(2); FF50K_FE_STEP(1);

// moves phase: OR (cand_mask), + (count_current_present, crit), bit-sliced +.
#define FF50K_MV_STEP(ADDM, off) do {                                             \
    cand_mask |= __shfl_xor_sync(0xffffffffu, cand_mask, (off));               \
    count_current_present +=                                                   \
        __shfl_xor_sync(0xffffffffu, count_current_present, (off));            \
    crit += __shfl_xor_sync(0xffffffffu, crit, (off));                         \
    ADDM(off);                                                                 \
} while (0)

#define FF50K_MV_RED_1
#define FF50K_MV_RED_2   FF50K_MV_STEP(FF50K_PC_ADD_SHFL4, 1);
#define FF50K_MV_RED_4   FF50K_MV_STEP(FF50K_PC_ADD_SHFL5, 2); FF50K_MV_STEP(FF50K_PC_ADD_SHFL5, 1);
#define FF50K_MV_RED_8   FF50K_MV_STEP(FF50K_PC_ADD_SHFL6, 4); FF50K_MV_STEP(FF50K_PC_ADD_SHFL6, 2); \
                      FF50K_MV_STEP(FF50K_PC_ADD_SHFL6, 1);
#define FF50K_MV_RED_16  FF50K_MV_STEP(FF50K_PC_ADD_SHFL7, 8); FF50K_MV_STEP(FF50K_PC_ADD_SHFL7, 4); \
                      FF50K_MV_STEP(FF50K_PC_ADD_SHFL7, 2); FF50K_MV_STEP(FF50K_PC_ADD_SHFL7, 1);
#define FF50K_MV_RED_32  FF50K_MV_STEP(FF50K_PC_ADD_SHFL9, 16); FF50K_MV_STEP(FF50K_PC_ADD_SHFL9, 8); \
                      FF50K_MV_STEP(FF50K_PC_ADD_SHFL9, 4); FF50K_MV_STEP(FF50K_PC_ADD_SHFL9, 2); \
                      FF50K_MV_STEP(FF50K_PC_ADD_SHFL9, 1);

// ---- per-pin bodies, factored out so the batched gather can hoist the loads --
// FF50K_FE_ACC is the champion's flags accumulation for ONE pin, taking the pin's
// already-loaded partition value (or -1 when the pin/node was out of range, in
// which case the unsigned test rejects it exactly as the champion's two nested
// range tests did).
#define FF50K_GIANT_BATCH 16

#define FF50K_FE_ACC(P) do {                                                      \
    const int part_ = (P);                                                     \
    if ((unsigned)part_ < 64u) {                                               \
        const unsigned long long bit_ = 1ULL << part_;                         \
        d |= (a & bit_);                                                       \
        a |= bit_;                                                             \
    }                                                                          \
} while (0)

// FF50K_MV_PIN is the champion's moves accumulation for ONE pin, taking the pin's
// already-loaded (flags_all, flags_double) pair.  Token-for-token the champion's
// loop body from `const unsigned long long flags_all` down to `INCM(flags)`.
#define FF50K_MV_PIN(FP, INCM) do {                                               \
    const unsigned long long fa_ = (FP).x;                                     \
    const unsigned long long fd_ = (FP).y;                                     \
    /* A fully spanning edge with repeated source contributes zero gain \
       to every candidate. Retain its candidate mask, omit equal counts. */ \
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
#define FF50K_PART_LOAD(n) __ldg(&partition[(n)])
#define FF50K_FE_BODY(G, SH, RED)                                                 \
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
                            ? FF50K_PART_LOAD(n0_) : -1;                     \
        const int pa_ = ((unsigned)na_ < (unsigned)num_nodes)                  \
                            ? FF50K_PART_LOAD(na_) : -1;                     \
        const int pb_ = ((unsigned)nb_ < (unsigned)num_nodes)                  \
                            ? FF50K_PART_LOAD(nb_) : -1;                     \
        const int pc_ = ((unsigned)nc_ < (unsigned)num_nodes)                  \
                            ? FF50K_PART_LOAD(nc_) : -1;                     \
        FF50K_FE_ACC(p0_);                                                        \
        FF50K_FE_ACC(pa_);                                                        \
        FF50K_FE_ACC(pb_);                                                        \
        FF50K_FE_ACC(pc_);                                                        \
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
// come from the enclosing FF50K_MV_BODY scope.
//
// `rel` is kmicro3's, unchanged: node_degree <= 256 => used_degree ==
// node_degree and (j*d)/d == j exactly; node_degree > 256 => the divisor is
// the literal 256 and the division is an exact >> 8 (both operands are
// non-negative and j <= 255, node_degree <= num_hyperedges, so the product is
// < 2^23 and it is computed in long long anyway).
#define FF50K_MV_CSR(G) do {                                                      \
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

// Exact bit-sliced argmax: availability and one-bit balance bonus masks.
// Every thread participates. All decisions use the same pre-round part sizes.
static __device__ __forceinline__ void exact_masks_50k(
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

#define FF50K_MV_BODY(G, SH, INCM, GETM, RED)                                     \
{                                                                              \
    const int lg_ = (int)(lane & (unsigned)((G) - 1));                         \
    const int node = im_.x;                                                    \
    const int nstart = im_.y;                                                  \
    const int node_degree = im_.z;                                             \
    int outkey = FF50K_SENT;                                                      \
    const int used_degree = node_degree > 256 ? 256 : node_degree;             \
    const bool capped_ = (node_degree > 256);                                  \
    const bool go_ = (node_degree > 0) && (lg_ < used_degree);                 \
    int j_ = lg_;                                                              \
    int h0_ = 0, h1_ = 0, h2_ = 0, h3_ = 0;                                    \
    bool v1_ = false, v2_ = false, v3_ = false;                                \
    if (go_) { FF50K_MV_CSR(G); }                                                 \
    const int current_part = FF50K_PART_LOAD(node);                          \
    if ((unsigned)current_part < (unsigned)num_parts &&                        \
        shared_nodes_in_part[current_part] > 1 &&                              \
        node_degree > 0) {                                                     \
        const unsigned long long current_bit = 1ULL << current_part;           \
        FF50K_PC_DECL                                                             \
        unsigned long long cand_mask = 0ULL;                                   \
        int count_current_present = 0;                                         \
        int crit = 0;                                                          \
        if (go_) {                                                             \
            for (;;) {                                                         \
                ulonglong2 f1_; f1_.x = 0ULL; f1_.y = 0ULL;                    \
                ulonglong2 f2_; f2_.x = 0ULL; f2_.y = 0ULL;                    \
                ulonglong2 f3_; f3_.x = 0ULL; f3_.y = 0ULL;                    \
                const ulonglong2 f0_ = ff50k_ldcg_u2(&edge_flags_pair[h0_]);      \
                if (v1_) f1_ = ff50k_ldcg_u2(&edge_flags_pair[h1_]);              \
                if (v2_) f2_ = ff50k_ldcg_u2(&edge_flags_pair[h2_]);              \
                if (v3_) f3_ = ff50k_ldcg_u2(&edge_flags_pair[h3_]);              \
                FF50K_MV_PIN(f0_, INCM);                                          \
                if (v1_) { FF50K_MV_PIN(f1_, INCM); }                             \
                if (v2_) { FF50K_MV_PIN(f2_, INCM); }                             \
                if (v3_) { FF50K_MV_PIN(f3_, INCM); }                             \
                j_ += 4 * (G);                                                 \
                if (j_ >= used_degree) break;                                  \
                FF50K_MV_CSR(G);                                                  \
            }                                                                  \
        }                                                                      \
        RED                                                                    \
        if (crit > 255) crit = 255;                                            \
        int degree_weight = node_degree > 255 ? 255 : node_degree;             \
        int rank_byte = degree_weight + crit;                                  \
        if (rank_byte > 255) rank_byte = 255;                                  \
        int best_gain = -999999;                                               \
        int best_target = current_part;                                        \
        unsigned long long active_ = ~current_bit & exact_free; \
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
        } \
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
        if (outkey != FF50K_SENT) {                                               \
            const int g_ = outkey >> 16;                                       \
            if (g_ > blk_max) blk_max = g_;                                    \
        }                                                                      \
    }                                                                          \
}

extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_filter_50k(
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
    __shared__ unsigned long long s_ab[32];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_50k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

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
        max_gain_buf[0] = FF50K_MG_INIT;
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
             k0 += (int)blockDim.x * FF50K_GIANT_BATCH) {
            int gn_[FF50K_GIANT_BATCH];
            int gp_[FF50K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                FF50K_FE_ACC(gp_[m]);
            }
        }
        FF50K_FE_RED_32
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
            case 0:  FF50K_FE_BODY(1,  0, FF50K_FE_RED_1)  break;
            case 1:  FF50K_FE_BODY(2,  1, FF50K_FE_RED_2)  break;
            case 2:  FF50K_FE_BODY(4,  2, FF50K_FE_RED_4)  break;
            case 3:  FF50K_FE_BODY(8,  3, FF50K_FE_RED_8)  break;
            case 4:  FF50K_FE_BODY(16, 4, FF50K_FE_RED_16) break;
            default: FF50K_FE_BODY(32, 5, FF50K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF50K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF50K_MV_BODY(1,  0, FF50K_PC_INC3, FF50K_PC_GET3, FF50K_MV_RED_1)  break;
            case 1:  FF50K_MV_BODY(2,  1, FF50K_PC_INC3, FF50K_PC_GET4, FF50K_MV_RED_2)  break;
            case 2:  FF50K_MV_BODY(4,  2, FF50K_PC_INC3, FF50K_PC_GET5, FF50K_MV_RED_4)  break;
            case 3:  FF50K_MV_BODY(8,  3, FF50K_PC_INC3, FF50K_PC_GET6, FF50K_MV_RED_8)  break;
            case 4:  FF50K_MV_BODY(16, 4, FF50K_PC_INC3, FF50K_PC_GET7, FF50K_MV_RED_16) break;
            default: FF50K_MV_BODY(32, 5, FF50K_PC_INC4, FF50K_PC_GET9, FF50K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff50k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF50K_MG_INIT if there are none
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
        // value is never used when key == FF50K_SENT and tabu_until[node] is in
        // range for every node < num_nodes.
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;

        int key = 0, gain = 0, elig = 0, posval = 0;
        if (node < num_nodes) {
            key = key_ld;
            if (key != FF50K_SENT) {
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
        // ff50k_block_sum in pass A and the rank with this scan in pass B1; the
        // scan already returns both).
        int etot = 0;
        const int erank = ff50k_block_scan(elig, s_warp, &etot);
        if (act && threadIdx.x == 0) blk_elig[tile] = etot;

        // ---- barrier (was barrier 3) --------------------------------------
        btgt_ += (unsigned int)gridDim.x;
        FF50K_GRID_BARRIER(grid_barrier, btgt_);

        int partsum = 0;
        for (int i = (int)threadIdx.x; i < plim; i += (int)blockDim.x) {
            partsum += __ldcg(&blk_elig[i]);
        }
        const int baseE = ff50k_block_sum(partsum, s_warp);

        int outkey = key;
        int accept = 0;
        if (elig) {
            // the host's rng_state is advanced ONCE per eligible node in node
            // order and tested AFTER the advance => the r-th (0-based) eligible
            // node tests f^(r+1)(x0). Tiles are consecutive node ranges and the
            // intra-tile rank is a threadIdx-ordered ballot prefix, so
            // baseE + erank IS the global 0-based eligible rank.
            const unsigned long long n = (unsigned long long)(baseE + erank) + 1ULL;
            const unsigned long long x = ff50k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain); // gain in [-3,0]
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key & 0xFFFF;      // host: (0 << 16) | (key & 0xFFFF)
            }
        }

        const int valid = (posval | accept);

        int vtot = 0;
        const int vrank = ff50k_block_scan(valid, s_warp, &vtot);
        if (act && threadIdx.x == 0) tile_valid[tile] = vtot;

        // ---- barrier (was barrier 4) --------------------------------------
        btgt_ += (unsigned int)gridDim.x;
        FF50K_GRID_BARRIER(grid_barrier, btgt_);

        int partsum2 = 0;
        for (int i = (int)threadIdx.x; i < plim; i += (int)blockDim.x) {
            partsum2 += __ldcg(&tile_valid[i]);
        }
        const int base = ff50k_block_sum(partsum2, s_warp);

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
        const int t = ff50k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = t;
    }
}


// ===========================================================================
// wave13_parity: the FLAT filter kernels.
//
// Body: `fused_flags_moves_filter_50k`'s prologue VERBATIM (phase 0, the
// giant hyperedges, mark list A, phase 1b, barrier 1, mark list B, phase 2b,
// the per-block max-gain atomicMax, barrier 2 and the aspiration) followed by
// the FLAT filter instead of wave6's fused tile loop.  Everything above the
// filter -- including every per-track move semantic -- is character for
// character the shipped kernel, because this script copies it out of it.
//
//   ..._flat_50k   __launch_bounds__(128, 2)   host `fused_mode = 1`
//   ..._flatw_50k  __launch_bounds__(256, 2)   host `fused_mode = 2`
//
// s_ab[16] is 2 entries per warp and 256 threads is 8 warps, so it is exactly
// full at 256 and its "blockDim.x <= 256" bound still holds.  s_warp[32]
// covers up to 1024 threads.
// ===========================================================================

extern "C" __global__ __launch_bounds__(128, 2) void fused_flags_moves_filter_flat_50k(
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
    __shared__ unsigned long long s_ab[32];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_50k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

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
        max_gain_buf[0] = FF50K_MG_INIT;
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
             k0 += (int)blockDim.x * FF50K_GIANT_BATCH) {
            int gn_[FF50K_GIANT_BATCH];
            int gp_[FF50K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                FF50K_FE_ACC(gp_[m]);
            }
        }
        FF50K_FE_RED_32
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
            case 0:  FF50K_FE_BODY(1,  0, FF50K_FE_RED_1)  break;
            case 1:  FF50K_FE_BODY(2,  1, FF50K_FE_RED_2)  break;
            case 2:  FF50K_FE_BODY(4,  2, FF50K_FE_RED_4)  break;
            case 3:  FF50K_FE_BODY(8,  3, FF50K_FE_RED_8)  break;
            case 4:  FF50K_FE_BODY(16, 4, FF50K_FE_RED_16) break;
            default: FF50K_FE_BODY(32, 5, FF50K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF50K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF50K_MV_BODY(1,  0, FF50K_PC_INC3, FF50K_PC_GET3, FF50K_MV_RED_1)  break;
            case 1:  FF50K_MV_BODY(2,  1, FF50K_PC_INC3, FF50K_PC_GET4, FF50K_MV_RED_2)  break;
            case 2:  FF50K_MV_BODY(4,  2, FF50K_PC_INC3, FF50K_PC_GET5, FF50K_MV_RED_4)  break;
            case 3:  FF50K_MV_BODY(8,  3, FF50K_PC_INC3, FF50K_PC_GET6, FF50K_MV_RED_8)  break;
            case 4:  FF50K_MV_BODY(16, 4, FF50K_PC_INC3, FF50K_PC_GET7, FF50K_MV_RED_16) break;
            default: FF50K_MV_BODY(32, 5, FF50K_PC_INC4, FF50K_PC_GET9, FF50K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff50k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF50K_MG_INIT if there are none
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
    // L2-resident i32 loads per node per pass and one extra ff50k_lcg_advance per
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
        if (node < num_nodes && key_ld != FF50K_SENT) {
            const int gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu && gain <= 0 && gain >= -3) elig = 1;
        }
        blkE += ff50k_block_sum(elig, s_warp);
    }
    if (threadIdx.x == 0) blk_elig[blockIdx.x] = blkE;

    // ---- barrier 3 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 3u * gridDim.x);

    int psE_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psE_ += __ldcg(&blk_elig[i]);
    }
    const int baseE_block = ff50k_block_sum(psE_, s_warp);

    // ---- pass B: the lottery, and this block's total number of valid nodes --
    int runE = 0;
    int blkV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF50K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff50k_block_scan(elig, s_warp, &etot);
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE + erank) + 1ULL;
            const unsigned long long x = ff50k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) accept = 1;
        }
        const int valid = (posval | accept);
        int vtot = 0;
        (void)ff50k_block_scan(valid, s_warp, &vtot);
        runE += etot;
        blkV += vtot;
    }
    if (threadIdx.x == 0) tile_valid[blockIdx.x] = blkV;

    // ---- barrier 4 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 4u * gridDim.x);

    int psV_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psV_ += __ldcg(&tile_valid[i]);
    }
    const int baseV_block = ff50k_block_sum(psV_, s_warp);

    // ---- pass C: emit the candidates ---------------------------------------
    int runE2 = 0;
    int runV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF50K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff50k_block_scan(elig, s_warp, &etot);
        int outkey = key_ld;
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE2 + erank) + 1ULL;
            const unsigned long long x = ff50k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key_ld & 0xFFFF;   // host: (0 << 16) | (key & 0xFFFF)
            }
        }
        const int valid = (posval | accept);
        int vtot = 0;
        const int vrank = ff50k_block_scan(valid, s_warp, &vtot);
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
        const int tot = ff50k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = tot;
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void fused_flags_moves_filter_flatw_50k(
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
    __shared__ unsigned long long s_ab[32];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_50k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

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
        max_gain_buf[0] = FF50K_MG_INIT;
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
             k0 += (int)blockDim.x * FF50K_GIANT_BATCH) {
            int gn_[FF50K_GIANT_BATCH];
            int gp_[FF50K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                FF50K_FE_ACC(gp_[m]);
            }
        }
        FF50K_FE_RED_32
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
            case 0:  FF50K_FE_BODY(1,  0, FF50K_FE_RED_1)  break;
            case 1:  FF50K_FE_BODY(2,  1, FF50K_FE_RED_2)  break;
            case 2:  FF50K_FE_BODY(4,  2, FF50K_FE_RED_4)  break;
            case 3:  FF50K_FE_BODY(8,  3, FF50K_FE_RED_8)  break;
            case 4:  FF50K_FE_BODY(16, 4, FF50K_FE_RED_16) break;
            default: FF50K_FE_BODY(32, 5, FF50K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF50K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF50K_MV_BODY(1,  0, FF50K_PC_INC3, FF50K_PC_GET3, FF50K_MV_RED_1)  break;
            case 1:  FF50K_MV_BODY(2,  1, FF50K_PC_INC3, FF50K_PC_GET4, FF50K_MV_RED_2)  break;
            case 2:  FF50K_MV_BODY(4,  2, FF50K_PC_INC3, FF50K_PC_GET5, FF50K_MV_RED_4)  break;
            case 3:  FF50K_MV_BODY(8,  3, FF50K_PC_INC3, FF50K_PC_GET6, FF50K_MV_RED_8)  break;
            case 4:  FF50K_MV_BODY(16, 4, FF50K_PC_INC3, FF50K_PC_GET7, FF50K_MV_RED_16) break;
            default: FF50K_MV_BODY(32, 5, FF50K_PC_INC4, FF50K_PC_GET9, FF50K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff50k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF50K_MG_INIT if there are none
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
    // L2-resident i32 loads per node per pass and one extra ff50k_lcg_advance per
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
        if (node < num_nodes && key_ld != FF50K_SENT) {
            const int gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu && gain <= 0 && gain >= -3) elig = 1;
        }
        blkE += ff50k_block_sum(elig, s_warp);
    }
    if (threadIdx.x == 0) blk_elig[blockIdx.x] = blkE;

    // ---- barrier 3 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 3u * gridDim.x);

    int psE_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psE_ += __ldcg(&blk_elig[i]);
    }
    const int baseE_block = ff50k_block_sum(psE_, s_warp);

    // ---- pass B: the lottery, and this block's total number of valid nodes --
    int runE = 0;
    int blkV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF50K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff50k_block_scan(elig, s_warp, &etot);
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE + erank) + 1ULL;
            const unsigned long long x = ff50k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) accept = 1;
        }
        const int valid = (posval | accept);
        int vtot = 0;
        (void)ff50k_block_scan(valid, s_warp, &vtot);
        runE += etot;
        blkV += vtot;
    }
    if (threadIdx.x == 0) tile_valid[blockIdx.x] = blkV;

    // ---- barrier 4 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 4u * gridDim.x);

    int psV_ = 0;
    for (int i = (int)threadIdx.x; i < (int)blockIdx.x; i += (int)blockDim.x) {
        psV_ += __ldcg(&tile_valid[i]);
    }
    const int baseV_block = ff50k_block_sum(psV_, s_warp);

    // ---- pass C: emit the candidates ---------------------------------------
    int runE2 = 0;
    int runV = 0;
    for (int t = t0; t < t1; t++) {
        const int node = t * TILE + (int)threadIdx.x;
        const int key_ld = (node < num_nodes) ? __ldcg(&move_priorities[node]) : 0;
        const int tu_ld = (node < num_nodes) ? __ldcg(&tabu_until[node]) : 0;
        int gain = 0, elig = 0, posval = 0;
        if (node < num_nodes && key_ld != FF50K_SENT) {
            gain = key_ld >> 16;
            const bool is_tabu = (tu_ld > round_idx) && (gain < aspiration);
            if (!is_tabu) {
                if (gain > 0) posval = 1;
                else if (gain >= -3) elig = 1;
            }
        }
        int etot = 0;
        const int erank = ff50k_block_scan(elig, s_warp, &etot);
        int outkey = key_ld;
        int accept = 0;
        if (elig) {
            const unsigned long long n =
                (unsigned long long)(baseE_block + runE2 + erank) + 1ULL;
            const unsigned long long x = ff50k_lcg_advance(lcg_x0, n);
            const unsigned long long penalty = (unsigned long long)(-gain);
            const unsigned long long den = prob_den_base * (1ULL + penalty * 5ULL);
            if (den > 0ULL && (x % den) < prob_num) {
                accept = 1;
                outkey = key_ld & 0xFFFF;   // host: (0 << 16) | (key & 0xFFFF)
            }
        }
        const int valid = (posval | accept);
        int vtot = 0;
        const int vrank = ff50k_block_scan(valid, s_warp, &vtot);
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
        const int tot = ff50k_block_sum(s, s_warp);
        if (threadIdx.x == 0) cand[0] = tot;
    }
}

// Exact bounded workspace. Sorting and application use disjoint phases.
// Exact bounded workspace. Sorting and application use disjoint phases.
// Exact bounded workspace. Sorting and application use disjoint phases.
// Exact bounded workspace. Sorting and application use disjoint phases.
union ExactScratch_50k {
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
static __device__ __forceinline__ int exact_apply_dag_50k(
    int M, int nn, int np, int cap, int *partition, int *part_sizes,
    const int *ordered, int *tabu, int round, int tenure,
    int mark_base, int mark_mult, ExactScratch_50k *work, int *reduce)
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
    int accepted = ff50k_block_sum(local_accepted, reduce);
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
static __device__ __forceinline__ int exact_apply_50k(
    int M, int nn, int np, int cap, int *partition, int *part_sizes,
    const int *ordered, int *tabu, int round, int tenure,
    int mark_base, int mark_mult, ExactScratch_50k *work, int *reduce)
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
        return exact_apply_dag_50k(M,nn,np,cap,partition,part_sizes,ordered,
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
            int prefix=ff50k_block_sum(local_accepted,reduce);
            int suffix=exact_apply_dag_50k(M-base,nn,np,cap,partition,part_sizes,ordered+base,
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
    int total=ff50k_block_sum(local_accepted,reduce);
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
static __device__ __forceinline__ int keyed_apply_50k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work)
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
// Solve only the sparse rejection system, without sorting every critical
// event. Dependencies use strictly smaller total keys. A stable iterate is
// the unique sequential replay; a bounded nonconvergent attempt changes no
// solver state and returns to the existing exact replay.
static __device__ __forceinline__ int sparse_reject_50k(
    int M,int nn,int np,int cap,int *partition,int *sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work,const int *data,const ulonglong2 *events){
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
        ulonglong2 event=ff50k_ldcg_u2(events+tid);
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

static __device__ __forceinline__ int parallel_keyed_apply_50k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work,const int *parallel_data,const ulonglong2 *parallel_events)
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
            unsigned long long key=ff50k_ldcg_u2(parallel_events+p*8+j).x;
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


static __device__ __forceinline__ int exact_apply_critical_50k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const int *ordered,int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work,int *reduce,const int *critical_input,const int *keeps)
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
        return exact_apply_50k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
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
        return exact_apply_50k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
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
        return exact_apply_50k(M,nn,np,cap,partition,part_sizes,ordered,tabu,
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

static __device__ __forceinline__ void exact_sort_prefix_50k(
    unsigned long long *keys,int n,int keep,ExactScratch_50k *work,int *ctl)
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

#undef FF50K_PART_LOAD
#define FF50K_PART_LOAD(n) __ldcg(&partition[(n)])
// f^n(x)=A[n]*x+C[n] (mod 2^64), precomputed once for every
// possible lottery rank. No RNG draws are added, skipped, or reordered.
static __device__ __forceinline__ unsigned long long exact_lcg_cached_50k(
    unsigned long long x,unsigned long long n,const ulonglong2 *table) {
    ulonglong2 ac=__ldg(&table[n]);return ac.x*x+ac.y;
}

static __device__ __forceinline__ int grouped_bucket_50k(int *counts,int target) {
    unsigned active=__activemask();
    unsigned same=__match_any_sync(active,target);
    int lane=threadIdx.x&31,leader=__ffs(same)-1,base=0;
    if(lane==leader)base=atomicAdd(counts+target,__popc(same));
    base=__shfl_sync(active,base,leader);
    return base+__popc(same&((1u<<lane)-1u));
}
// CTA barriers publish each block's writes to its leader and distribute the
// leader's acquire to its peers. Device-scope acq_rel RMWs form a release chain
// across block leaders; polling the completed epoch acquires that chain.
static __device__ __forceinline__ unsigned acqbar_load_50k(const unsigned *p){
    unsigned value;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(p):"memory");return value;
}
static __device__ __forceinline__ void acqbar_50k(unsigned *gb,unsigned target){
    __syncthreads();
    if(threadIdx.x==0){
        unsigned previous;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;":"=r"(previous):"l"(gb):"memory");
        while(acqbar_load_50k(gb)<target)__nanosleep(20);
    }
    __syncthreads();
}

#undef FF50K_GRID_BARRIER
#define FF50K_GRID_BARRIER(g,t) acqbar_50k((g),(t))

// Slot allocation order is immaterial: selection later restores the total key order.
static __device__ __forceinline__ int sparse_slot_50k(int *counts,int target){
    unsigned active=__activemask(),group=__match_any_sync(active,target);
    int lane=threadIdx.x&31,leader=__ffs(group)-1,base=0;
    if(lane==leader)base=atomicAdd(counts+target,__popc(group));
    base=__shfl_sync(active,base,leader);
    return base+__popc(group&((1u<<lane)-1u));
}

// An edge of at most 255 pins needs 8-bit counters, at most 65535 pins
// needs 16-bit counters; all others retain full 32-bit counters. Updates
// are separated into monotone removal/addition phases, so no field can
// underflow or carry into its neighboring packed field.
static __device__ __forceinline__ int packed_count_50k(
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

// Dense two-event replay with deferred partition/tabu scatter.
static __device__ __forceinline__ int dense_deferred128_50k(
    int M,int nn,int np,int cap,int *partition,int *sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work,const int *data,const ulonglong2 *events, int *reject_flags){
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
        ulonglong2 event=ff50k_ldcg_u2(events+target*8+local);
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
    // Publish only the decision. The next round's distributed membership
    // pass applies partition/tabu writes before its original flags barrier.
    if(reject){int index=offsets[target]+max(0,cap-work->keyed.initial[target])+local;reject_flags[index]=1;}
    return accepted;
}
static __device__ __forceinline__ int deferred_sparse_50k(
    int M,int nn,int np,int cap,int *partition,int *sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work,const int *data,const ulonglong2 *events, int *reject_flags){
    int fast=dense_deferred128_50k(M,nn,np,cap,partition,sizes,selected,keeps,offsets,
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
        ulonglong2 event=ff50k_ldcg_u2(events+tid);
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
    // Publish only the decision. The next round's distributed membership
    // pass applies partition/tabu writes before its original flags barrier.
    if(reject){int index=offsets[target]+max(0,cap-work->keyed.initial[target])+local;reject_flags[index]=1;}
    return accepted;
}
static __device__ __forceinline__ int deferred_parallel_50k(
    int M,int nn,int np,int cap,int *partition,int *part_sizes,
    const unsigned long long *selected,const int *keeps,const int *offsets,
    int *tabu,int round,int tenure,int mark_base,int mark_mult,
    ExactScratch_50k *work,const int *parallel_data,const ulonglong2 *parallel_events, int *reject_flags)
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
            unsigned long long key=ff50k_ldcg_u2(parallel_events+p*8+j).x;
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
    if(marks!=0 && marks!=M)return -1;
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
    for(int j=tid;j<S;j+=blockDim.x)if(work->keyed.rejected[j]){
        int target=work->keyed.packed[j]&63;
        int index=offsets[target]+max(0,cap-work->keyed.initial[target])+j-work->keyed.offsets[target];
        reject_flags[index]=1;
    }
    return accepted;
}
extern "C" __global__ __launch_bounds__(512, 1) void exact_round_loop_50k(
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
    __shared__ ExactScratch_50k exact_work;
    __shared__ int exact_bucket_starts[65],exact_radix_ctl[4];
    __shared__ int exact_keeps[64], exact_offsets[65], exact_meta[4];
    unsigned int nbar = 0;
    int rounds_done = 0, reason = 0, total_moves = 0;
    int stagnant = initial_stagnant, zero_streak = initial_zero;
    int zero_count = 0, max_zero = initial_zero;
    bool counts_removed=false,export_prepared=false;
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
    __shared__ unsigned long long s_ab[32];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64) {
        shared_nodes_in_part[threadIdx.x] = threadIdx.x < num_parts ? __ldcg(&nodes_in_part[threadIdx.x]) : 0;
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_50k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

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

    if (gtid < 64) bucket_counts[gtid] = 0;
    for(int j=gtid;j<577;j+=stride)parallel_data[j]=0;
    // ---- phase 0: reset max-gain -------------------------------------------
    if (gtid == 0) {
        max_gain_buf[0] = FF50K_MG_INIT; cand[0]=0;
    }

    // Exact membership updates from integer part counts. Separate removals
    // and additions so each cell traverses a fixed monotone count interval.
    // XOR toggles commute, and the number of threshold crossings is fixed.
    if(step==0){
        for(int j=gtid;j<count_words;j+=stride)part_counts[j]=0;
        for(int h=gtid;h<count_nh;h+=stride)edge_flags_pair[h]=make_ulonglong2(0,0);
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        for(int h=warp_global;h<count_nh;h+=warp_stride){
            int begin=__ldg(&count_edge_offsets[h]),end=__ldg(&count_edge_offsets[h+1]);
            for(int j=begin+(int)lane;j<end;j+=32){
                int n=__ldg(&hyperedge_nodes[j]);
                int part=(unsigned)n<(unsigned)num_nodes?__ldcg(&partition[n]):-1;
                if((unsigned)part<64u){
                    int old=packed_count_50k(part_counts,count_meta,h,part,false);
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
                int old=packed_count_50k(part_counts,count_meta,h,now,false);
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

    // ---- barrier 1 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);

    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF50K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF50K_MV_BODY(1,  0, FF50K_PC_INC3, FF50K_PC_GET3, FF50K_MV_RED_1)  break;
            case 1:  FF50K_MV_BODY(2,  1, FF50K_PC_INC3, FF50K_PC_GET4, FF50K_MV_RED_2)  break;
            case 2:  FF50K_MV_BODY(4,  2, FF50K_PC_INC3, FF50K_PC_GET5, FF50K_MV_RED_4)  break;
            case 3:  FF50K_MV_BODY(8,  3, FF50K_PC_INC3, FF50K_PC_GET6, FF50K_MV_RED_8)  break;
            case 4:  FF50K_MV_BODY(16, 4, FF50K_PC_INC3, FF50K_PC_GET7, FF50K_MV_RED_16) break;
            default: FF50K_MV_BODY(32, 5, FF50K_PC_INC4, FF50K_PC_GET9, FF50K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff50k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF50K_MG_INIT if there are none
    // (then no node passes the filter and the host breaks out, so the
    // aspiration value is never observed).

    const int mg=__ldcg(max_gain_buf),aspiration=(mg*3)/4;
    const int tiles_per_block=(ntiles+(int)gridDim.x-1)/(int)gridDim.x;
    const int first_tile=(int)blockIdx.x*tiles_per_block;
    const int last_tile=min(ntiles,first_tile+tiles_per_block);
    int local_eligible=0;
    for(int tile=first_tile;tile<last_tile;++tile){
        int node=tile*(int)blockDim.x+(int)threadIdx.x;
        if(node<num_nodes){
            int key=__ldcg(&move_priorities[node]),tu=__ldcg(&tabu_until[node]);
            if(key!=FF50K_SENT){int gain=key>>16;bool is_tabu=(tu > round_idx) && (gain < aspiration);
                if(!is_tabu && gain <= 0 && gain >= -3)++local_eligible;
            }
        }
    }
    int block_eligible=ff50k_block_sum(local_eligible,s_warp);
    if(threadIdx.x==0)blk_elig[blockIdx.x]=block_eligible;
    FF50K_GRID_BARRIER(grid_barrier,barrier_base+3u*gridDim.x);
    int prefix=0;
    for(int block=(int)threadIdx.x;block<(int)blockIdx.x;block+=(int)blockDim.x)
        prefix+=__ldcg(&blk_elig[block]);
    const int base_eligible=ff50k_block_sum(prefix,s_warp);
    int run_eligible=0,local_valid=0;
    for(int tile=first_tile;tile<last_tile;++tile){
        const int node=tile*(int)blockDim.x+(int)threadIdx.x;
        int key=node<num_nodes?__ldcg(&move_priorities[node]):FF50K_SENT;
        int tu=node<num_nodes?__ldcg(&tabu_until[node]):0;
        int gain=key>>16,eligible=0,positive=0;
        if(node<num_nodes && key!=FF50K_SENT){
            bool is_tabu=(tu > round_idx) && (gain < aspiration);
            positive=!is_tabu && gain>0;
            eligible=(!is_tabu && gain <= 0 && gain >= -3);
        }
        int eligible_count=0;
        int rank=ff50k_block_scan(eligible,s_warp,&eligible_count);
        int outkey=positive?key:-1;
        if(eligible){
            unsigned long long index=(unsigned long long)(base_eligible+run_eligible+rank)+1ULL;
            unsigned long long x=exact_lcg_cached_50k(lcg_x0,index,lcg_table);
            unsigned long long den=prob_den_base*(1ULL+5ULL*(unsigned long long)(-gain));
            if(den>0ULL && x%den<prob_num)outkey=key&0xffff;
        }
        // Sparse, node-indexed fallback scratch. Normal rounds need no global
        // candidate ordering; only the lottery eligibility index is ordered.
        if(node<num_nodes)cand[1+node]=outkey;
        if(outkey>=0){
            ++local_valid;
            int target=outkey&63,slot=sparse_slot_50k(bucket_counts,target);
            if(slot<4096)bucket_keys[target*4096+slot]=
                ((unsigned long long)((unsigned int)outkey^0x7fffffffu)<<32)|(unsigned int)node;
        }
        run_eligible+=eligible_count;
    }
    int block_valid=ff50k_block_sum(local_valid,s_warp);
    if(threadIdx.x==0)atomicAdd(cand,block_valid);
    nbar+=3;
    ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

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
        if(gtid==0)control[13]=M;
        if (N == 0) { reason = 1; break; }
        int adaptive_limit = round_idx < 50 ? move_limit / 2 :
                            (round_idx < 200 ? (move_limit * 3) / 4 : move_limit / 4);
        // Even when the original global k_base clips N, quota selection is
        // identical if its candidate window still contains ALL N entries and
        // the union of per-target prefixes does not exceed k_base.
        bool full_quota_equivalent = N <= adaptive_limit ||
            (N <= adaptive_limit + 57344 && M <= adaptive_limit);
        if (!full_quota_equivalent || exact_meta[2] || M <= 0 || num_nodes<256) { reason = 4; break; }


        if(exact_meta[3]){
            if(gtid<64)bucket_counts[64+gtid]=0;
            ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
            for(int i=gtid;i<num_nodes;i+=stride){
                int key=__ldcg(&cand[1+i]);
                if(key<0)continue;
                int node=i,target=key&63;
                int pos=grouped_bucket_50k(bucket_counts + 64,target);
                bucket_keys[exact_bucket_starts[target]+pos]=
                    ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
            }
            ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        }
        for(int p=blockIdx.x;p<num_parts && p<64;p+=gridDim.x){
            int n=__ldcg(&bucket_counts[p]),keep=exact_keeps[p],off=exact_offsets[p];
            int start=exact_bucket_starts[p];
            if(keep==0)continue;
            exact_sort_prefix_50k(&bucket_keys[start],n,keep,&exact_work,exact_radix_ctl);
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
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

        // All target prefixes are now published. Distribute the exact
        // predecessor tallies across the grid instead of serializing them
        // on the single replay CTA.
        for(int i=gtid;i<M;i+=stride){
            ordered_moves[i]=0;
            unsigned long long key=__ldcg(selected_keys+i);
            int source=__ldcg(selected_old_parts+i);
            if((unsigned)source>=(unsigned)num_parts)continue;
            int risk=min(8,max(0,exact_keeps[source]-max(0,max_part_size-shared_nodes_in_part[source])));
            for(int j=0;j<risk;++j){
                unsigned long long event=ff50k_ldcg_u2(parallel_events+source*8+j).x;
                if(key<event)atomicAdd(parallel_data+64+source*8+j,1);
            }
        }
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        // Try a key-ordered sparse replay without materializing the
        // globally merged list. Failure leaves all solver state untouched.
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
                int old=packed_count_50k(part_counts,count_meta,h,old_part,true);
                if(old==1)atomicXor(&edge_flags_pair[h].x,1ULL<<old_part);
                if(old==2)atomicXor(&edge_flags_pair[h].y,1ULL<<old_part);
            }
        }
        }
        if(blockIdx.x==0) {
            int result=deferred_sparse_50k(M,num_nodes,num_parts,max_part_size,
                partition,nodes_in_part,selected_keys,exact_keeps,exact_offsets,
                tabu_until,round_idx,tabu_tenure,tabu_mark_base,tabu_mark_mult,&exact_work,parallel_data,parallel_events,ordered_moves);
            if(result<0){
            result=deferred_parallel_50k(M,num_nodes,num_parts,max_part_size,
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
                int old=packed_count_50k(part_counts,count_meta,h,old_part,true);
                if(old==1)atomicXor(&edge_flags_pair[h].x,1ULL<<old_part);
                if(old==2)atomicXor(&edge_flags_pair[h].y,1ULL<<old_part);
            }
        }
        }
        counts_removed=true;
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
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
        ++nbar; FF50K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        if (blockIdx.x == 0) {
            int accepted = exact_apply_critical_50k(M, num_nodes, num_parts, max_part_size,
                partition, nodes_in_part, ordered_moves, tabu_until, round_idx,
                tabu_tenure, tabu_mark_base, tabu_mark_mult, &exact_work, s_warp, cand+1, exact_keeps);
            if (threadIdx.x == 0) control[15] = accepted;
        }
        ++nbar; FF50K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        }
        int accepted = __ldcg(&control[15]);
        deferred_pending=__ldcg(control+14)!=0;
        deferred_mark_pending=accepted>0 && (tabu_mark_base>0 || tabu_mark_mult>0);
        deferred_until=round_idx+tabu_tenure;

        // A zero-execution clipped round must run the original secondary
        // tail fallback (and failed tabu marks). No partition or tabu entry
        // changed when accepted==0, so replay it unchanged on the host.
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
                int old=packed_count_50k(part_counts,count_meta,h,now,false);
                if(old==0)atomicXor(&edge_flags_pair[h].x,1ULL<<now);
                if(old==1)atomicXor(&edge_flags_pair[h].y,1ULL<<now);
            }
        }
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        control[0] = reason; control[1] = rounds_done; control[2] = (int)nbar;
        control[3] = total_moves; control[4] = stagnant; control[5] = zero_streak;
        control[7] = zero_count; control[8] = max_zero;
    }
}

// Exact tail kernel: the shared engine's degree-classed flag/move evaluator,
// without lottery or selection. The original tail's sampling and tie breaks
// are retained; one grid barrier and the original 128-thread tail geometry.
#undef FF50K_PART_LOAD
#define FF50K_PART_LOAD(n) __ldg(&partition[(n)])
extern "C" __global__ __launch_bounds__(128, 2) void exact_tail_moves_50k(
    const int num_hyperedges,const int num_nodes,const int num_parts,const int max_part_size,
    const int *hyperedge_nodes,const int *node_hyperedges,
    const int *partition,const int *nodes_in_part,
    const int4 *fe_slots,const int n_fe_tasks,const int4 *fe_giant,const int n_fe_giant,
    const int4 *mv_slots,const int n_mv_tasks,ulonglong2 *edge_flags_pair,
    unsigned long long *edge_flags_all,unsigned long long *edge_flags_double,
    int *move_priorities,unsigned int *grid_barrier,const unsigned int barrier_target
) {

    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64];
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[32];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_50k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

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
             k0 += (int)blockDim.x * FF50K_GIANT_BATCH) {
            int gn_[FF50K_GIANT_BATCH];
            int gp_[FF50K_GIANT_BATCH];
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                const int k = k0 + m * (int)blockDim.x;
                gn_[m] = (k < hsz) ? __ldg(&hyperedge_nodes[hstart + k]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                gp_[m] = ((unsigned)gn_[m] < (unsigned)num_nodes)
                             ? __ldg(&partition[gn_[m]]) : -1;
            }
#pragma unroll
            for (int m = 0; m < FF50K_GIANT_BATCH; m++) {
                FF50K_FE_ACC(gp_[m]);
            }
        }
        FF50K_FE_RED_32
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
            case 0:  FF50K_FE_BODY(1,  0, FF50K_FE_RED_1)  break;
            case 1:  FF50K_FE_BODY(2,  1, FF50K_FE_RED_2)  break;
            case 2:  FF50K_FE_BODY(4,  2, FF50K_FE_RED_4)  break;
            case 3:  FF50K_FE_BODY(8,  3, FF50K_FE_RED_8)  break;
            case 4:  FF50K_FE_BODY(16, 4, FF50K_FE_RED_16) break;
            default: FF50K_FE_BODY(32, 5, FF50K_FE_RED_32) break;
        }
        fim_ = nx_;
    }

    // ---- barrier 1 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_target);

    // Preserve the original tail's separate flag-array outputs as well.
    for(int edge=gtid;edge<num_hyperedges;edge+=stride){
        ulonglong2 value=ff50k_ldcg_u2(&edge_flags_pair[edge]);
        edge_flags_all[edge]=value.x;edge_flags_double[edge]=value.y;
    }
    int blk_max = FF50K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF50K_MV_BODY(1,  0, FF50K_PC_INC3, FF50K_PC_GET3, FF50K_MV_RED_1)  break;
            case 1:  FF50K_MV_BODY(2,  1, FF50K_PC_INC3, FF50K_PC_GET4, FF50K_MV_RED_2)  break;
            case 2:  FF50K_MV_BODY(4,  2, FF50K_PC_INC3, FF50K_PC_GET5, FF50K_MV_RED_4)  break;
            case 3:  FF50K_MV_BODY(8,  3, FF50K_PC_INC3, FF50K_PC_GET6, FF50K_MV_RED_8)  break;
            case 4:  FF50K_MV_BODY(16, 4, FF50K_PC_INC3, FF50K_PC_GET7, FF50K_MV_RED_16) break;
            default: FF50K_MV_BODY(32, 5, FF50K_PC_INC4, FF50K_PC_GET9, FF50K_MV_RED_32) break;
        }
        mim_ = nx_;
    }


}


// Device-resident tail refinement. Same flags, move keys, quota order and
// sequential execution as the original tail. The LCG is RESET on every
// round, with increment1 rather than the main loop's increment.
#undef FF50K_PART_LOAD
#define FF50K_PART_LOAD(n) __ldcg(&partition[(n)])
static __device__ __forceinline__ unsigned long long exact_tail_lcg_50k(
    unsigned long long x,unsigned long long n,const ulonglong2 *table) {
    ulonglong2 ac=__ldg(&table[n]);return ac.x*x+ac.y*16613134689194145199ULL;
}
extern "C" __global__ __launch_bounds__(512, 1) void exact_tail_loop_50k(
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
    int *selected_old_parts, int *parallel_data,ulonglong2 *parallel_events, const unsigned *count_meta, const int count_words, const int num_hyperedges,
    unsigned long long *tail_flags_all, unsigned long long *tail_flags_double
)  {
    __shared__ ExactScratch_50k exact_work;
    __shared__ int exact_bucket_starts[65],exact_radix_ctl[4];
    __shared__ int exact_keeps[64], exact_offsets[65], exact_meta[4];
    unsigned int nbar = 0;
    int rounds_done = 0, reason = 0, total_moves = 0;
    int stagnant = initial_stagnant, zero_streak = initial_zero;
    int zero_count = 0, max_zero = initial_zero;
    bool counts_removed=false,export_prepared=false;
    bool deferred_pending=false,deferred_mark_pending=false;int deferred_until=0;
    for (int step = 0; step < (int)batch_rounds && round_start + step < round_end; ++step) {
        const int round_idx = round_start + step;
        const int n_mark_a = step == 0 ? initial_mark_a : 0;
        const int n_mark_b = step == 0 ? initial_mark_b : 0;
        unsigned int barrier_base = initial_barrier_base + nbar * gridDim.x;
        const unsigned long long lcg_x0=run_off;
        const unsigned long long prob_num=(unsigned long long)schedule_exp;

    __shared__ int shared_nodes_in_part[64];
    __shared__ unsigned long long exact_free, exact_bonus[64];
    __shared__ int s_warp[32];
    __shared__ unsigned long long s_ab[32];   // 2 per warp, blockDim.x <= 256

    if (threadIdx.x < 64) {
        shared_nodes_in_part[threadIdx.x] = threadIdx.x < num_parts ? __ldcg(&nodes_in_part[threadIdx.x]) : 0;
    }

    const int stride = gridDim.x * blockDim.x;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const int warps_per_block = (int)(blockDim.x >> 5);
    const int warp_global = (int)blockIdx.x * warps_per_block + (int)(threadIdx.x >> 5);
    const int warp_stride = (int)gridDim.x * warps_per_block;
    exact_masks_50k(num_parts, max_part_size, shared_nodes_in_part, &exact_free, exact_bonus);

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

    if (gtid < 64) bucket_counts[gtid] = 0;
    for(int j=gtid;j<577;j+=stride)parallel_data[j]=0;
    // ---- phase 0: reset max-gain -------------------------------------------
    if (gtid == 0) {
        max_gain_buf[0] = FF50K_MG_INIT; cand[0]=0;
    }

    // Exact membership updates from integer part counts. Separate removals
    // and additions so each cell traverses a fixed monotone count interval.
    // XOR toggles commute, and the number of threshold crossings is fixed.
    if(step==0){
        for(int j=gtid;j<count_words;j+=stride)part_counts[j]=0;
        for(int h=gtid;h<count_nh;h+=stride)edge_flags_pair[h]=make_ulonglong2(0,0);
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        for(int h=warp_global;h<count_nh;h+=warp_stride){
            int begin=__ldg(&count_edge_offsets[h]),end=__ldg(&count_edge_offsets[h+1]);
            for(int j=begin+(int)lane;j<end;j+=32){
                int n=__ldg(&hyperedge_nodes[j]);
                int part=(unsigned)n<(unsigned)num_nodes?__ldcg(&partition[n]):-1;
                if((unsigned)part<64u){
                    int old=packed_count_50k(part_counts,count_meta,h,part,false);
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
                int old=packed_count_50k(part_counts,count_meta,h,now,false);
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

    // ---- barrier 1 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + gridDim.x);
    if(step+1==(int)batch_rounds || round_idx+1==round_end){
        for(int h=gtid;h<num_hyperedges;h+=stride){
            ulonglong2 v=ff50k_ldcg_u2(&edge_flags_pair[h]);
            tail_flags_all[h]=v.x;tail_flags_double[h]=v.y;
        }
        export_prepared=true;
    }


    // ---- phase 2a: scatter tabu mark list B --------------------------------
    // Ordered strictly after list A by barrier 1 (last-writer-wins, as on the
    // host). Node ids inside one list are unique => no intra-list conflict.
    for (int i = gtid; i < n_mark_b; i += stride) {
        const int n = mark_nodes[n_mark_a + i];
        if (n >= 0 && n < num_nodes) tabu_until[n] = until_b;
    }

    int blk_max = FF50K_MG_INIT;

    // ---- phase 2b: moves, lane groups over warp tasks ----------------------
    for (int t = warp_global; t < n_mv_tasks; t += warp_stride) {
        const int tn_ = t + warp_stride;
        int4 nx_; nx_.x = 0; nx_.y = 0; nx_.z = 0; nx_.w = 0;
        if (tn_ < n_mv_tasks) {
            nx_ = __ldg(&mv_slots[tn_ * 32 + (int)lane]);
        }
        const int4 im_ = mim_;
        switch (im_.w) {
            case 0:  FF50K_MV_BODY(1,  0, FF50K_PC_INC3, FF50K_PC_GET3, FF50K_MV_RED_1)  break;
            case 1:  FF50K_MV_BODY(2,  1, FF50K_PC_INC3, FF50K_PC_GET4, FF50K_MV_RED_2)  break;
            case 2:  FF50K_MV_BODY(4,  2, FF50K_PC_INC3, FF50K_PC_GET5, FF50K_MV_RED_4)  break;
            case 3:  FF50K_MV_BODY(8,  3, FF50K_PC_INC3, FF50K_PC_GET6, FF50K_MV_RED_8)  break;
            case 4:  FF50K_MV_BODY(16, 4, FF50K_PC_INC3, FF50K_PC_GET7, FF50K_MV_RED_16) break;
            default: FF50K_MV_BODY(32, 5, FF50K_PC_INC4, FF50K_PC_GET9, FF50K_MV_RED_32) break;
        }
        mim_ = nx_;
    }

    {
        const int m = ff50k_block_max(blk_max, s_warp);
        if (threadIdx.x == 0) atomicMax(max_gain_buf, m);
    }

    // ---- barrier 2 ---------------------------------------------------------
    FF50K_GRID_BARRIER(grid_barrier, barrier_base + 2u * gridDim.x);

    // max over non-sentinel keys of (key >> 16); FF50K_MG_INIT if there are none
    // (then no node passes the filter and the host breaks out, so the
    // aspiration value is never observed).

    const int mg=__ldcg(max_gain_buf),aspiration=(mg*3)/4;
    const int tiles_per_block=(ntiles+(int)gridDim.x-1)/(int)gridDim.x;
    const int first_tile=(int)blockIdx.x*tiles_per_block;
    const int last_tile=min(ntiles,first_tile+tiles_per_block);
    int local_eligible=0;
    for(int tile=first_tile;tile<last_tile;++tile){
        int node=tile*(int)blockDim.x+(int)threadIdx.x;
        if(node<num_nodes){
            int key=__ldcg(&move_priorities[node]),tu=__ldcg(&tabu_until[node]);
            if(key!=FF50K_SENT){int gain=key>>16;bool is_tabu=false;
                if(prob_den_base>0ULL && gain==0)++local_eligible;
            }
        }
    }
    int block_eligible=ff50k_block_sum(local_eligible,s_warp);
    if(threadIdx.x==0)blk_elig[blockIdx.x]=block_eligible;
    FF50K_GRID_BARRIER(grid_barrier,barrier_base+3u*gridDim.x);
    int prefix=0;
    for(int block=(int)threadIdx.x;block<(int)blockIdx.x;block+=(int)blockDim.x)
        prefix+=__ldcg(&blk_elig[block]);
    const int base_eligible=ff50k_block_sum(prefix,s_warp);
    int run_eligible=0,local_valid=0;
    for(int tile=first_tile;tile<last_tile;++tile){
        const int node=tile*(int)blockDim.x+(int)threadIdx.x;
        int key=node<num_nodes?__ldcg(&move_priorities[node]):FF50K_SENT;
        int tu=node<num_nodes?__ldcg(&tabu_until[node]):0;
        int gain=key>>16,eligible=0,positive=0;
        if(node<num_nodes && key!=FF50K_SENT){
            bool is_tabu=false;
            positive=!is_tabu && gain>0;
            eligible=(prob_den_base>0ULL && gain==0);
        }
        int eligible_count=0;
        int rank=ff50k_block_scan(eligible,s_warp,&eligible_count);
        int outkey=positive?key:-1;
        if(eligible){
            unsigned long long index=(unsigned long long)(base_eligible+run_eligible+rank)+1ULL;
            unsigned long long x=exact_tail_lcg_50k(lcg_x0,index,lcg_table);
            unsigned long long den=prob_den_base*(1ULL+5ULL*(unsigned long long)(-gain));
            if(den>0ULL && x%den<prob_num)outkey=key&0xffff;
        }
        // Sparse, node-indexed fallback scratch. Normal rounds need no global
        // candidate ordering; only the lottery eligibility index is ordered.
        if(node<num_nodes)cand[1+node]=outkey;
        if(outkey>=0){
            ++local_valid;
            int target=outkey&63,slot=sparse_slot_50k(bucket_counts,target);
            if(slot<4096)bucket_keys[target*4096+slot]=
                ((unsigned long long)((unsigned int)outkey^0x7fffffffu)<<32)|(unsigned int)node;
        }
        run_eligible+=eligible_count;
    }
    int block_valid=ff50k_block_sum(local_valid,s_warp);
    if(threadIdx.x==0)atomicAdd(cand,block_valid);
    nbar+=3;
    ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

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
        if(gtid==0)control[13]=M;
        if (N == 0) { reason = 1; break; }
        int adaptive_limit=move_limit;
        // Even when the original global k_base clips N, quota selection is
        // identical if its candidate window still contains ALL N entries and
        // the union of per-target prefixes does not exceed k_base.
        bool full_quota_equivalent = N <= adaptive_limit ||
            (N <= adaptive_limit + perturb_end && M <= adaptive_limit);
        if (!full_quota_equivalent || exact_meta[2] || M <= 0 || num_nodes<256) { reason = 4; break; }


        if(exact_meta[3]){
            if(gtid<64)bucket_counts[64+gtid]=0;
            ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
            for(int i=gtid;i<num_nodes;i+=stride){
                int key=__ldcg(&cand[1+i]);
                if(key<0)continue;
                int node=i,target=key&63;
                int pos=grouped_bucket_50k(bucket_counts + 64,target);
                bucket_keys[exact_bucket_starts[target]+pos]=
                    ((unsigned long long)((unsigned int)key^0x7fffffffu)<<32)|(unsigned int)node;
            }
            ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        }
        for(int p=blockIdx.x;p<num_parts && p<64;p+=gridDim.x){
            int n=__ldcg(&bucket_counts[p]),keep=exact_keeps[p],off=exact_offsets[p];
            int start=exact_bucket_starts[p];
            if(keep==0)continue;
            exact_sort_prefix_50k(&bucket_keys[start],n,keep,&exact_work,exact_radix_ctl);
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
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);

        // All target prefixes are now published. Distribute the exact
        // predecessor tallies across the grid instead of serializing them
        // on the single replay CTA.
        for(int i=gtid;i<M;i+=stride){
            ordered_moves[i]=0;
            unsigned long long key=__ldcg(selected_keys+i);
            int source=__ldcg(selected_old_parts+i);
            if((unsigned)source>=(unsigned)num_parts)continue;
            int risk=min(8,max(0,exact_keeps[source]-max(0,max_part_size-shared_nodes_in_part[source])));
            for(int j=0;j<risk;++j){
                unsigned long long event=ff50k_ldcg_u2(parallel_events+source*8+j).x;
                if(key<event)atomicAdd(parallel_data+64+source*8+j,1);
            }
        }
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
        // Try a key-ordered sparse replay without materializing the
        // globally merged list. Failure leaves all solver state untouched.
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
                int old=packed_count_50k(part_counts,count_meta,h,old_part,true);
                if(old==1)atomicXor(&edge_flags_pair[h].x,1ULL<<old_part);
                if(old==2)atomicXor(&edge_flags_pair[h].y,1ULL<<old_part);
            }
        }
        }
        if(blockIdx.x==0) {
            int result=deferred_sparse_50k(M,num_nodes,num_parts,max_part_size,
                partition,nodes_in_part,selected_keys,exact_keeps,exact_offsets,
                tabu_until,round_idx,tabu_tenure,tabu_mark_base,tabu_mark_mult,&exact_work,parallel_data,parallel_events,ordered_moves);
            if(result<0){
            result=deferred_parallel_50k(M,num_nodes,num_parts,max_part_size,
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
                int old=packed_count_50k(part_counts,count_meta,h,old_part,true);
                if(old==1)atomicXor(&edge_flags_pair[h].x,1ULL<<old_part);
                if(old==2)atomicXor(&edge_flags_pair[h].y,1ULL<<old_part);
            }
        }
        }
        counts_removed=true;
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
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
        ++nbar; FF50K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        if (blockIdx.x == 0) {
            int accepted = exact_apply_critical_50k(M, num_nodes, num_parts, max_part_size,
                partition, nodes_in_part, ordered_moves, tabu_until, round_idx,
                tabu_tenure, tabu_mark_base, tabu_mark_mult, &exact_work, s_warp, cand+1, exact_keeps);
            if (threadIdx.x == 0) control[15] = accepted;
        }
        ++nbar; FF50K_GRID_BARRIER(grid_barrier, initial_barrier_base + nbar * gridDim.x);
        }
        int accepted = __ldcg(&control[15]);
        deferred_pending=__ldcg(control+14)!=0;
        deferred_mark_pending=accepted>0 && (tabu_mark_base>0 || tabu_mark_mult>0);
        deferred_until=round_idx+tabu_tenure;

        // A zero-execution clipped round must run the original secondary
        // tail fallback (and failed tabu marks). No partition or tabu entry
        // changed when accepted==0, so replay it unchanged on the host.
        if(accepted==0 && N>adaptive_limit){reason=4;break;}
        total_moves += accepted;
        ++rounds_done;
        if(accepted==0){reason=3;break;}

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
                int old=packed_count_50k(part_counts,count_meta,h,now,false);
                if(old==0)atomicXor(&edge_flags_pair[h].x,1ULL<<now);
                if(old==1)atomicXor(&edge_flags_pair[h].y,1ULL<<now);
            }
        }
        ++nbar; FF50K_GRID_BARRIER(grid_barrier,initial_barrier_base+nbar*gridDim.x);
    }
    if(!export_prepared){
    for(int h=(int)blockIdx.x*blockDim.x+threadIdx.x;h<num_hyperedges;h+=(int)gridDim.x*blockDim.x){
        ulonglong2 value=ff50k_ldcg_u2(&edge_flags_pair[h]);
        tail_flags_all[h]=value.x;tail_flags_double[h]=value.y;
    }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        control[0] = reason; control[1] = rounds_done; control[2] = (int)nbar;
        control[3] = total_moves; control[4] = stagnant; control[5] = zero_streak;
        control[7] = zero_count; control[8] = max_zero;
    }
}
