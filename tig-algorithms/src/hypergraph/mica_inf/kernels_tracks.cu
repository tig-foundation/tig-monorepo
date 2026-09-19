// Device kernels of the 20k/50k/100k/200k tracks: one body per kernel, one extern "C" entry per track.
// Invariants: 1 <= num_parts <= 64 (64-bit block masks), blocks of num_parts..=128 threads (a multiple
// of 32 for the warp kernels), no inter-block wait.

#include <stdint.h>
#include <cuda_runtime.h>

namespace track {

// Per hyperedge, the elite whose partition cuts it the least (ties: earliest in elite_order).
__device__ __forceinline__ void choose_elite_per_hyperedge(
    const int num_hyperedges, const int num_nodes, const int num_parts, const int num_elites,
    const int *elite_partitions, const int *elite_order, const int *hyperedge_offsets,
    const int *hyperedge_nodes, int *hedge_choice_elite)
{
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

// Per node, the block voted by the elites chosen for its hyperedges; ties go to the block
// present in the most of the node's hyperedges under edge_flags_best.
__device__ __forceinline__ void assign_from_elite_votes(
    const int num_nodes, const int num_parts, const int *elite_partitions,
    const int *hedge_choice_elite, const int *node_hyperedges, const int *node_offsets,
    const ulonglong2 *edge_flags_best, int *partition_out)
{
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
                    score += ((__ldg(edge_flags_best + hedge).x & bit) != 0ULL);
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

// Hashes each hyperedge into one of num_clusters buckets, stratified by hyperedge size.
__device__ __forceinline__ void hyperedge_clustering(
    const int num_hyperedges, const int num_clusters, const int *hyperedge_offsets,
    const int *hyperedge_nodes, int *hyperedge_clusters)
{
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

// Per node, the preferred block (from its majority cluster) and an assignment priority.
__device__ __forceinline__ void compute_node_preferences(
    const int num_nodes, const int num_parts, const int num_hedge_clusters,
    const int *node_hyperedges, const int *node_offsets, const int *hyperedge_clusters,
    const int *hyperedge_offsets, int *pref_parts, int *pref_priorities)
{
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

// Single-thread greedy assignment in priority order; the first num_parts nodes seed the blocks.
__device__ __forceinline__ void execute_node_assignments(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int *sorted_nodes, const int *sorted_parts, int *partition, int *nodes_in_part)
{
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

// Per node, its three best target blocks with gain >= -(neg_gain_thresh + degree/20),
// packed as (gain << 16) | target; 0 means no entry, so a zero gain towards block 0 is dropped.
// part_counts is a 64-entry per-thread scratch.
__device__ __forceinline__ void compute_swap_gains_extended(
    const int num_nodes, const int num_parts, const int neg_gain_thresh,
    const int *node_hyperedges, const int *node_offsets, const int *partition,
    const ulonglong2 *edge_flags, int *swap_gains, unsigned short *part_counts)
{
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

    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;
    const int sample_step = node_degree / used_degree;
    const int sample_rem = node_degree - sample_step * used_degree;
    int sample_rel = 0;
    int sample_acc = 0;

    for (int j = 0; j < used_degree; j++) {
        int hyperedge = __ldg(node_hyperedges + start + sample_rel);

        const ulonglong2 ef = __ldg(edge_flags + hyperedge);
        unsigned long long flags_all = ef.x;
        unsigned long long flags_double = ef.y;
        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);

        if (mask & current_bit) count_current_present++;

        unsigned long long flags = mask & ~current_bit;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            unsigned long long b = 1ULL << bit;
            if (cand_mask & b) { part_counts[bit]++; }
            else { part_counts[bit] = 1; cand_mask |= b; }
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

// Per hyperedge, (blocks touched - 1).
__device__ __forceinline__ void compute_connectivity(
    const int num_hyperedges, const int *hyperedge_nodes, const int *hyperedge_offsets,
    const int *partition, int *connectivity)
{
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

// Block-wise sum of the connectivity array (two elements per thread per stride); the host adds the blocks.
__device__ __forceinline__ void reduce_connectivity_sum(
    const int num_hyperedges, const int *connectivity, int *total_connectivity_blocks)
{
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

// Single-thread repair: fills blocks below min_part_size, then drains blocks above max_part_size.
__device__ __forceinline__ void balance_final(
    const int num_nodes, const int num_parts, const int min_part_size, const int max_part_size,
    int *partition, int *nodes_in_part)
{
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

// Total order on candidate moves: gain, then smaller target block size, then a per-node hash,
// then lower block index.
__device__ __forceinline__ bool move_better(
    int g, int t, int gb, int tb, const int *snip, int node)
{
    if (tb < 0)      return true;
    if (g  > gb)     return true;
    if (g  < gb)     return false;
    int ts = snip[t], bs = snip[tb];
    if (ts < bs)     return true;
    if (ts > bs)     return false;
    int ht = (t * 17 + node) & 63, hb = (tb * 17 + node) & 63;
    if (ht < hb)     return true;
    if (ht > hb)     return false;
    return t < tb;
}

// Best move per node, eight lanes per node (each lane counts one byte of the 64-block mask).
// Key: (gain << 16) | (rank << 8) | target; 0x80000000 means no move.
__device__ __forceinline__ void compute_refinement_moves_warp8(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int *node_hyperedges, const int *node_offsets, const int *partition,
    const int *nodes_in_part, const ulonglong2 *edge_flags, int *move_priorities,
    const int *node_order)
{
    __shared__ int shared_nodes_in_part[64];
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    const int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int task = gid >> 3;
    const int lane8 = gid & 7;
    if (task >= num_nodes) return;
    const int node = node_order[task];

    const unsigned lw   = threadIdx.x & 31;
    const unsigned grp  = 0xFFu << (lw & ~7);
    const unsigned base = lw & ~7;

    if (lane8 == 0) move_priorities[node] = 0x80000000;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;
    if (shared_nodes_in_part[current_part] <= 1) return;

    int start = node_offsets[node];
    int end   = node_offsets[node + 1];
    int node_degree = end - start;
    int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) return;

    const unsigned long long current_bit = 1ULL << current_part;
    const unsigned long long lane_slice  = 0xFFULL << (lane8 * 8);

    unsigned short c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;
    int crit = 0;

    const int sample_step = node_degree / used_degree;
    const int sample_rem  = node_degree - sample_step * used_degree;
    int sample_rel = 0, sample_acc = 0;

    for (int j = 0; j < used_degree; j++) {
        int hyperedge = node_hyperedges[start + sample_rel];
        const ulonglong2 ef = __ldg(edge_flags + hyperedge);
        unsigned long long flags_all    = ef.x;
        unsigned long long flags_double = ef.y;

        if (lane8 == 0) {
            if ((flags_all & current_bit) != 0ULL && (flags_double & current_bit) == 0ULL) {
                int parts = __popcll(flags_all);
                if (parts > 1) { crit += (parts - 1); if (crit > 255) crit = 255; }
            }
        }

        unsigned long long mask = (flags_all & ~current_bit) | (flags_double & current_bit);
        if (lane8 == 0 && (mask & current_bit)) count_current_present++;

        unsigned long long flags = mask & ~current_bit & lane_slice;
        cand_mask |= flags;
        const unsigned int octet =
            (unsigned int)(flags >> (lane8 * 8));
        c0 +=  octet       & 1u;
        c1 += (octet >> 1) & 1u;
        c2 += (octet >> 2) & 1u;
        c3 += (octet >> 3) & 1u;
        c4 += (octet >> 4) & 1u;
        c5 += (octet >> 5) & 1u;
        c6 += (octet >> 6) & 1u;
        c7 += (octet >> 7) & 1u;

        sample_rel += sample_step;
        sample_acc += sample_rem;
        if (sample_acc >= used_degree) { sample_acc -= used_degree; sample_rel++; }
    }

    crit                  = __shfl_sync(grp, crit,                  base);
    count_current_present = __shfl_sync(grp, count_current_present, base);

    int best_gain = -999999, best_target = -1;
    unsigned long long tmp = cand_mask;
    while (tmp) {
        int target_part = __ffsll(tmp) - 1;
        tmp &= (tmp - 1);
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int cnt;
        switch (target_part & 7) {
            case 0: cnt = c0; break; case 1: cnt = c1; break;
            case 2: cnt = c2; break; case 3: cnt = c3; break;
            case 4: cnt = c4; break; case 5: cnt = c5; break;
            case 6: cnt = c6; break; default: cnt = c7; break;
        }
        int basic_gain    = cnt - count_current_present;
        int current_size  = shared_nodes_in_part[current_part];
        int target_size   = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 2) ? 1 : 0;
        int total_gain    = basic_gain + balance_bonus;

        if (move_better(total_gain, target_part, best_gain, best_target,
                        shared_nodes_in_part, node)) {
            best_gain = total_gain; best_target = target_part;
        }
    }

    for (int off = 4; off > 0; off >>= 1) {
        int og = __shfl_down_sync(grp, best_gain,   off);
        int ot = __shfl_down_sync(grp, best_target, off);
        if ((lane8 < off) && ot >= 0 &&
            move_better(og, ot, best_gain, best_target, shared_nodes_in_part, node)) {
            best_gain = og; best_target = ot;
        }
    }

    if (lane8 != 0) return;
    if (best_target < 0 || best_target == current_part) return;

    int degree_weight = node_degree > 255 ? 255 : node_degree;
    int rank_byte = degree_weight + crit;
    if (rank_byte > 255) rank_byte = 255;

    int bg = best_gain > 32767 ? 32767 : best_gain;
    if (bg < -32768) bg = -32768;
    unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
    move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
}

// Best move per node, one thread per node with per-block counters in shared memory
// (blocks of at most 128 threads). zgm_mode selects the tie-break: 0 lowest target,
// 1 most pins in target, 2 most free room, 3 smallest target then hash.
__device__ __forceinline__ void compute_refinement_moves_smem(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int *node_hyperedges, const int *node_offsets, const int *partition,
    const int *nodes_in_part, const ulonglong2 *edge_flags, int *move_priorities,
    const int zgm_mode, const int *node_order)
{
    __shared__ int shared_nodes_in_part[64];
    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= num_nodes) return;
    const int node = node_order[task];

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

    __shared__ unsigned short part_counts[64][128];

    unsigned long long cand_mask = 0ULL;
    int count_current_present = 0;
    const int sample_step = node_degree / used_degree;
    const int sample_rem = node_degree - sample_step * used_degree;
    int sample_rel = 0;
    int sample_acc = 0;
    int crit = 0;

    for (int j = 0; j < used_degree; j++) {
        int hyperedge = __ldg(node_hyperedges + start + sample_rel);

        const ulonglong2 ef = __ldg(edge_flags + hyperedge);
        unsigned long long flags_all = ef.x;
        unsigned long long flags_double = ef.y;

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
            unsigned long long b = 1ULL << bit;
            if (cand_mask & b) { part_counts[bit][threadIdx.x]++; }
            else { part_counts[bit][threadIdx.x] = 1; cand_mask |= b; }
        }

        sample_rel += sample_step;
        sample_acc += sample_rem;
        if (sample_acc >= used_degree) {
            sample_acc -= used_degree;
            sample_rel++;
        }
    }

    int degree_weight = node_degree > 255 ? 255 : node_degree;
    int rank_byte = degree_weight + crit;
    if (rank_byte > 255) rank_byte = 255;

    int best_gain = -999999;
    int best_target = current_part;
    int best_tb = -2147483647;

    while (cand_mask) {
        int target_part = __ffsll(cand_mask) - 1;
        cand_mask &= (cand_mask - 1);
        if ((unsigned)target_part >= (unsigned)num_parts) continue;
        if (shared_nodes_in_part[target_part] >= max_part_size) continue;

        int basic_gain = (int)part_counts[target_part][threadIdx.x] - count_current_present;

        int current_size = shared_nodes_in_part[current_part];
        int target_size = shared_nodes_in_part[target_part];
        int balance_bonus = (current_size > target_size + 2) ? 1 : 0;

        int total_gain = basic_gain + balance_bonus;

        if (zgm_mode == 3) {
            bool better3 = (total_gain > best_gain);
            if (!better3 && total_gain == best_gain) {
                int best_target_size = shared_nodes_in_part[best_target];
                if (target_size < best_target_size) {
                    better3 = true;
                } else if (target_size == best_target_size) {
                    int hash_tgt = (target_part * 17 + node) & 63;
                    int hash_best = (best_target * 17 + node) & 63;
                    if (hash_tgt < hash_best) better3 = true;
                }
            }
            if (better3) {
                best_gain = total_gain;
                best_target = target_part;
            }
        } else if (zgm_mode == 0) {
            if (total_gain > best_gain || (total_gain == best_gain && target_part < best_target)) {
                best_gain = total_gain;
                best_target = target_part;
            }
        } else {
            int tb = (zgm_mode == 1) ? (int)part_counts[target_part][threadIdx.x]
                                     : (max_part_size - target_size);
            if (total_gain > best_gain) {
                best_gain = total_gain;
                best_tb = tb;
                best_target = target_part;
            } else if (total_gain == best_gain &&
                       (tb > best_tb || (tb == best_tb && target_part < best_target))) {
                best_tb = tb;
                best_target = target_part;
            }
        }
    }

    if (best_target != current_part) {
        int bg = best_gain > 32767 ? 32767 : best_gain;
        if (bg < -32768) bg = -32768;
        unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
        move_priorities[node] = ((int)(short)bg << 16) | (rank_byte << 8) | t16;
    }
}

// Block masks of one hyperedge: x = blocks present, y = blocks present at least twice.
__device__ __forceinline__ ulonglong2 edge_flags_of(
    const int start, const int end, const int num_nodes,
    const int *hyperedge_nodes, const int *partition)
{
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
    return make_ulonglong2(flags_all, flags_double);
}

// One thread per hyperedge.
__device__ __forceinline__ void precompute_edge_flags(
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags)
{
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        edge_flags[hedge] = edge_flags_of(start, end, num_nodes, hyperedge_nodes, partition);
    }
}

// One thread per listed hyperedge (duplicates in the list rewrite the same value).
__device__ __forceinline__ void precompute_edge_flags_hlist(
    const int n_list, const int *hedge_list, const int num_hyperedges, const int num_nodes,
    const int *hyperedge_nodes, const int *hyperedge_offsets, const int *partition,
    ulonglong2 *edge_flags)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_list) return;
    int hedge = hedge_list[t];
    if (hedge < 0 || hedge >= num_hyperedges) return;
    int start = hyperedge_offsets[hedge];
    int end   = hyperedge_offsets[hedge + 1];
    edge_flags[hedge] = edge_flags_of(start, end, num_nodes, hyperedge_nodes, partition);
}

// Warp-cooperative flags of one hyperedge (pins start..end); lane 0 writes the result.
__device__ __forceinline__ void edge_flags_warp(
    const int h, const int start, const int end, const int lane, const int num_nodes,
    const int *hyperedge_nodes, const int *partition, ulonglong2 *edge_flags)
{
    unsigned long long fa = 0ULL, fd = 0ULL;
    for (int k = start + lane; k < end; k += 32) {
        int node = hyperedge_nodes[k];
        if (node >= 0 && node < num_nodes) {
            int part = partition[node];
            if (part >= 0 && part < 64) {
                unsigned long long bit = 1ULL << part;
                fd |= (fa & bit);
                fa |= bit;
            }
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long oa = __shfl_down_sync(0xffffffffu, fa, off);
        unsigned long long od = __shfl_down_sync(0xffffffffu, fd, off);
        fd = fd | od | (fa & oa);
        fa = fa | oa;
    }
    if (lane == 0) edge_flags[h] = make_ulonglong2(fa, fd);
}

// One thread per hyperedge of at most 32 pins.
__device__ __forceinline__ void edge_flags_small(
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_hyperedges) {
        const int start = hyperedge_offsets[gid];
        const int end   = hyperedge_offsets[gid + 1];
        if (end - start <= 32) {
            unsigned long long fa = 0ULL, fd = 0ULL;
            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                if (node >= 0 && node < num_nodes) {
                    int part = partition[node];
                    if (part >= 0 && part < 64) {
                        unsigned long long bit = 1ULL << part;
                        fd |= (fa & bit);
                        fa |= bit;
                    }
                }
            }
            edge_flags[gid] = make_ulonglong2(fa, fd);
        }
    }
}

// Small hyperedges per thread, then every large hyperedge per warp (grid-stride scan).
__device__ __forceinline__ void precompute_edge_flags_warp_scan(
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags)
{
    edge_flags_small(num_hyperedges, num_nodes, hyperedge_nodes, hyperedge_offsets, partition, edge_flags);

    const int lane = threadIdx.x & 31;
    const int wpb  = blockDim.x >> 5;
    const int gw   = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int tw   = gridDim.x * wpb;

    for (int h = gw; h < num_hyperedges; h += tw) {
        const int start = hyperedge_offsets[h];
        const int end   = hyperedge_offsets[h + 1];
        if (end - start <= 32) continue;
        edge_flags_warp(h, start, end, lane, num_nodes, hyperedge_nodes, partition, edge_flags);
    }
}

// Small hyperedges per thread, then the listed hyperedges (all of more than 32 pins) per warp.
__device__ __forceinline__ void precompute_edge_flags_warp_list(
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags,
    const int *big_ids, const int n_big)
{
    edge_flags_small(num_hyperedges, num_nodes, hyperedge_nodes, hyperedge_offsets, partition, edge_flags);

    const int lane = threadIdx.x & 31;
    const int wpb  = blockDim.x >> 5;
    const int gw   = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int tw   = gridDim.x * wpb;

    for (int i = gw; i < n_big; i += tw) {
        const int h     = big_ids[i];
        const int start = hyperedge_offsets[h];
        const int end   = hyperedge_offsets[h + 1];
        edge_flags_warp(h, start, end, lane, num_nodes, hyperedge_nodes, partition, edge_flags);
    }
}

} // namespace track

// Exported entries. The names carry the track suffix that params.rs selects.
#define TRACK_KERNELS_COMMON(S)                                                                     \
extern "C" __global__ void choose_elite_per_hyperedge_##S(                                          \
    const int num_hyperedges, const int num_nodes, const int num_parts, const int num_elites,        \
    const int *elite_partitions, const int *elite_order, const int *hyperedge_offsets,              \
    const int *hyperedge_nodes, int *hedge_choice_elite)                                            \
{                                                                                                   \
    track::choose_elite_per_hyperedge(num_hyperedges, num_nodes, num_parts, num_elites,             \
        elite_partitions, elite_order, hyperedge_offsets, hyperedge_nodes, hedge_choice_elite);      \
}                                                                                                   \
extern "C" __global__ void assign_from_elite_votes_##S(                                             \
    const int num_nodes, const int num_parts, const int num_elites, const int *elite_partitions,    \
    const int *hedge_choice_elite, const int *node_hyperedges, const int *node_offsets,             \
    const ulonglong2 *edge_flags_best, int *partition_out)                                          \
{                                                                                                   \
    (void)num_elites;                                                                               \
    track::assign_from_elite_votes(num_nodes, num_parts, elite_partitions, hedge_choice_elite,      \
        node_hyperedges, node_offsets, edge_flags_best, partition_out);                             \
}                                                                                                   \
extern "C" __global__ void hyperedge_clustering_##S(                                                \
    const int num_hyperedges, const int num_clusters, const int *hyperedge_offsets,                 \
    const int *hyperedge_nodes, int *hyperedge_clusters)                                            \
{                                                                                                   \
    track::hyperedge_clustering(num_hyperedges, num_clusters, hyperedge_offsets, hyperedge_nodes,   \
        hyperedge_clusters);                                                                        \
}                                                                                                   \
extern "C" __global__ void compute_node_preferences_##S(                                            \
    const int num_nodes, const int num_parts, const int num_hedge_clusters,                         \
    const int *node_hyperedges, const int *node_offsets, const int *hyperedge_clusters,             \
    const int *hyperedge_offsets, int *pref_parts, int *pref_priorities)                            \
{                                                                                                   \
    track::compute_node_preferences(num_nodes, num_parts, num_hedge_clusters, node_hyperedges,      \
        node_offsets, hyperedge_clusters, hyperedge_offsets, pref_parts, pref_priorities);          \
}                                                                                                   \
extern "C" __global__ void execute_node_assignments_##S(                                            \
    const int num_nodes, const int num_parts, const int max_part_size,                              \
    const int *sorted_nodes, const int *sorted_parts, int *partition, int *nodes_in_part)           \
{                                                                                                   \
    track::execute_node_assignments(num_nodes, num_parts, max_part_size, sorted_nodes,              \
        sorted_parts, partition, nodes_in_part);                                                    \
}                                                                                                   \
extern "C" __global__ void compute_swap_gains_extended_##S(                                         \
    const int num_nodes, const int num_parts, const int neg_gain_thresh,                            \
    const int *node_hyperedges, const int *node_offsets, const int *partition,                      \
    const ulonglong2 *edge_flags, int *swap_gains)                                                  \
{                                                                                                   \
    unsigned short part_counts[64];                                                                 \
    track::compute_swap_gains_extended(num_nodes, num_parts, neg_gain_thresh, node_hyperedges,      \
        node_offsets, partition, edge_flags, swap_gains, part_counts);                              \
}                                                                                                   \
extern "C" __global__ void compute_connectivity_##S(                                                \
    const int num_hyperedges, const int *hyperedge_nodes, const int *hyperedge_offsets,             \
    const int *partition, int *connectivity)                                                        \
{                                                                                                   \
    track::compute_connectivity(num_hyperedges, hyperedge_nodes, hyperedge_offsets, partition,      \
        connectivity);                                                                              \
}                                                                                                   \
extern "C" __global__ void reduce_connectivity_sum_##S(                                             \
    const int num_hyperedges, const int *connectivity, int *total_connectivity_blocks)              \
{                                                                                                   \
    track::reduce_connectivity_sum(num_hyperedges, connectivity, total_connectivity_blocks);        \
}                                                                                                   \
extern "C" __global__ void balance_final_##S(                                                       \
    const int num_nodes, const int num_parts, const int min_part_size, const int max_part_size,     \
    int *partition, int *nodes_in_part)                                                             \
{                                                                                                   \
    track::balance_final(num_nodes, num_parts, min_part_size, max_part_size, partition,             \
        nodes_in_part);                                                                             \
}

#define TRACK_MOVES_WARP8(S)                                                                        \
extern "C" __global__ void compute_refinement_moves_warp8_##S(                                      \
    const int num_nodes, const int num_parts, const int max_part_size,                              \
    const int *node_hyperedges, const int *node_offsets, const int *partition,                      \
    const int *nodes_in_part, const ulonglong2 *edge_flags, int *move_priorities,                   \
    const int zgm_mode, const int *node_order)                                                      \
{                                                                                                   \
    (void)zgm_mode;                                                                                 \
    track::compute_refinement_moves_warp8(num_nodes, num_parts, max_part_size, node_hyperedges,     \
        node_offsets, partition, nodes_in_part, edge_flags, move_priorities, node_order);           \
}

#define TRACK_MOVES_SMEM(S)                                                                         \
extern "C" __global__ void compute_refinement_moves_smem_##S(                                       \
    const int num_nodes, const int num_parts, const int max_part_size,                              \
    const int *node_hyperedges, const int *node_offsets, const int *partition,                      \
    const int *nodes_in_part, const ulonglong2 *edge_flags, int *move_priorities,                   \
    const int zgm_mode, const int *node_order)                                                      \
{                                                                                                   \
    track::compute_refinement_moves_smem(num_nodes, num_parts, max_part_size, node_hyperedges,      \
        node_offsets, partition, nodes_in_part, edge_flags, move_priorities, zgm_mode, node_order); \
}

#define TRACK_EDGE_FLAGS_WARP_SCAN(S)                                                               \
extern "C" __global__ void precompute_edge_flags_wpe_##S(                                           \
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,                      \
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags)                     \
{                                                                                                   \
    track::precompute_edge_flags_warp_scan(num_hyperedges, num_nodes, hyperedge_nodes,              \
        hyperedge_offsets, partition, edge_flags);                                                  \
}

#define TRACK_EDGE_FLAGS_WARP_LIST(S)                                                               \
extern "C" __global__ void precompute_edge_flags_wpe_##S(                                           \
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,                      \
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags,                     \
    const int *big_ids, const int n_big)                                                            \
{                                                                                                   \
    track::precompute_edge_flags_warp_list(num_hyperedges, num_nodes, hyperedge_nodes,              \
        hyperedge_offsets, partition, edge_flags, big_ids, n_big);                                  \
}

#define TRACK_EDGE_FLAGS_PLAIN(S)                                                                   \
extern "C" __global__ void precompute_edge_flags_##S(                                               \
    const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,                      \
    const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags)                     \
{                                                                                                   \
    track::precompute_edge_flags(num_hyperedges, num_nodes, hyperedge_nodes, hyperedge_offsets,     \
        partition, edge_flags);                                                                     \
}                                                                                                   \
extern "C" __global__ void precompute_edge_flags_hlist_##S(                                         \
    const int n_list, const int *hedge_list, const int num_hyperedges, const int num_nodes,         \
    const int *hyperedge_nodes, const int *hyperedge_offsets, const int *partition,                 \
    ulonglong2 *edge_flags)                                                                         \
{                                                                                                   \
    track::precompute_edge_flags_hlist(n_list, hedge_list, num_hyperedges, num_nodes,               \
        hyperedge_nodes, hyperedge_offsets, partition, edge_flags);                                 \
}

TRACK_KERNELS_COMMON(20k)
TRACK_MOVES_WARP8(20k)
TRACK_EDGE_FLAGS_WARP_SCAN(20k)

TRACK_KERNELS_COMMON(50k)
TRACK_MOVES_WARP8(50k)
TRACK_EDGE_FLAGS_WARP_LIST(50k)

TRACK_KERNELS_COMMON(100k)
TRACK_MOVES_SMEM(100k)
TRACK_EDGE_FLAGS_PLAIN(100k)

TRACK_KERNELS_COMMON(200k)
TRACK_MOVES_WARP8(200k)
TRACK_EDGE_FLAGS_PLAIN(200k)

#undef TRACK_KERNELS_COMMON
#undef TRACK_MOVES_WARP8
#undef TRACK_MOVES_SMEM
#undef TRACK_EDGE_FLAGS_WARP_SCAN
#undef TRACK_EDGE_FLAGS_WARP_LIST
#undef TRACK_EDGE_FLAGS_PLAIN
