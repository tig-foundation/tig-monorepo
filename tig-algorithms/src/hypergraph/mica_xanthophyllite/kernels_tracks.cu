
#include <stdint.h>
#include <cuda_runtime.h>

namespace track {

__device__ __forceinline__ unsigned long long other_parts(int part) {
    const unsigned int low = part < 32 ? ~(1u << part) : 0xFFFFFFFFu;
    const unsigned int high = part < 32 ? 0xFFFFFFFFu : ~(1u << (part - 32));
    return ((unsigned long long)high << 32) | low;
}

__device__ __forceinline__ void
choose_elite_per_hyperedge(const int num_hyperedges, const int num_nodes, const int num_parts,
                           const int num_elites, const int *elite_partitions,
                           const int *elite_order, const int *hyperedge_offsets,
                           const int *hyperedge_nodes, int *hedge_choice_elite) {
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

__device__ __forceinline__ void
assign_from_elite_votes(const int num_nodes, const int num_parts, const int *elite_partitions,
                        const int *hedge_choice_elite, const int *node_hyperedges,
                        const int *node_offsets, const ulonglong2 *edge_flags_best,
                        int *partition_out) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int np = (num_parts < 64) ? num_parts : 64;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int deg = end - start;
    int used_degree = (deg > 256) ? 256 : deg;

    unsigned short votes[64];
    for (int p = 0; p < np; p++)
        votes[p] = 0;

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

__device__ __forceinline__ void
hyperedge_clustering(const int num_hyperedges, const int num_clusters, const int *hyperedge_offsets,
                     const int *hyperedge_nodes, int *hyperedge_clusters) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        int hedge_size = end - start;
        int quarter_clusters = num_clusters >> 2;
        int bucket = (hedge_size > 8) ? 3 : (hedge_size > 4) ? 2 : (hedge_size > 2) ? 1 : 0;
        int first = (hedge_size > 0) ? hyperedge_nodes[start] : hedge;
        int last = (hedge_size > 0) ? hyperedge_nodes[end - 1] : hedge;
        unsigned int h = (unsigned int)first * 1103515245u + (unsigned int)last * 12345u +
                         (unsigned int)hedge_size * 2654435761u + (unsigned int)hedge;
        int cluster = bucket * quarter_clusters + (int)(h % (unsigned int)quarter_clusters);
        hyperedge_clusters[hedge] = cluster;
    }
}

__device__ __forceinline__ void
compute_node_preferences(const int num_nodes, const int num_parts, const int num_hedge_clusters,
                         const int *node_hyperedges, const int *node_offsets,
                         const int *hyperedge_clusters, const int *hyperedge_offsets,
                         int *pref_parts, int *pref_priorities) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < num_nodes) {
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;
        int cluster_votes[64];
        int max_clusters = min(num_hedge_clusters, 64);
        for (int i = 0; i < max_clusters; i++)
            cluster_votes[i] = 0;
        int max_votes = 0;
        int best_cluster = 0;
        for (int j = start; j < end; j++) {
            int hyperedge = node_hyperedges[j];
            int cluster = hyperedge_clusters[hyperedge] % max_clusters;
            if (cluster >= 0) {
                int hedge_start = hyperedge_offsets[hyperedge];
                int hedge_end = hyperedge_offsets[hyperedge + 1];
                int hedge_size = hedge_end - hedge_start;
                int weight = (hedge_size <= 2)   ? 6
                             : (hedge_size <= 4) ? 4
                             : (hedge_size <= 8) ? 2
                                                 : 1;
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
        pref_priorities[node] =
            (mv << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
    }
}

__device__ __forceinline__ void execute_node_assignments(const int num_nodes, const int num_parts,
                                                         const int max_part_size,
                                                         const int *sorted_nodes,
                                                         const int *sorted_parts, int *partition,
                                                         int *nodes_in_part) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < num_nodes; i++) {
            int node = sorted_nodes[i];
            int preferred_part = sorted_parts[i];
            if (node >= 0 && node < num_nodes && preferred_part >= 0 &&
                preferred_part < num_parts) {
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

__device__ __forceinline__ void
compute_swap_gains_extended(const int num_nodes, const int num_parts, const int neg_gain_thresh,
                            const int *node_hyperedges, const int *node_offsets,
                            const int *partition, const ulonglong2 *edge_flags, int *swap_gains,
                            unsigned short *part_counts) {
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
    const unsigned long long others = other_parts(current_part);

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
        unsigned long long mask = (flags_all & others) | (flags_double & current_bit);

        if (mask & current_bit) count_current_present++;

        unsigned long long flags = mask & others;
        while (flags) {
            int bit = __ffsll(flags) - 1;
            flags &= (flags - 1);
            unsigned long long b = 1ULL << bit;
            if (cand_mask & b) {
                part_counts[bit]++;
            } else {
                part_counts[bit] = 1;
                cand_mask |= b;
            }
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
            top_gain[2] = top_gain[1];
            top_part[2] = top_part[1];
            top_gain[1] = top_gain[0];
            top_part[1] = top_part[0];
            top_gain[0] = basic_gain;
            top_part[0] = target_part;
        } else if (basic_gain > top_gain[1] ||
                   (basic_gain == top_gain[1] && target_part < top_part[1])) {
            top_gain[2] = top_gain[1];
            top_part[2] = top_part[1];
            top_gain[1] = basic_gain;
            top_part[1] = target_part;
        } else if (basic_gain > top_gain[2] ||
                   (basic_gain == top_gain[2] && target_part < top_part[2])) {
            top_gain[2] = basic_gain;
            top_part[2] = target_part;
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

__device__ __forceinline__ void compute_connectivity(const int num_hyperedges,
                                                     const int *hyperedge_nodes,
                                                     const int *hyperedge_offsets,
                                                     const int *partition, int *connectivity) {
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

__device__ __forceinline__ void reduce_connectivity_sum(const int num_hyperedges,
                                                        const int *connectivity,
                                                        int *total_connectivity_blocks) {
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

__device__ __forceinline__ void balance_final(const int num_nodes, const int num_parts,
                                              const int min_part_size, const int max_part_size,
                                              int *partition, int *nodes_in_part) {
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

__device__ __forceinline__ bool move_better(int g, int t, int gb, int tb, const int *snip,
                                            int node) {
    if (tb < 0) return true;
    if (g > gb) return true;
    if (g < gb) return false;
    int ts = snip[t], bs = snip[tb];
    if (ts < bs) return true;
    if (ts > bs) return false;
    int ht = (t * 17 + node) & 63, hb = (tb * 17 + node) & 63;
    if (ht < hb) return true;
    if (ht > hb) return false;
    return t < tb;
}

template <int ZGM>
__device__ __forceinline__ bool cand_better(int g, int t, int tb, int gb, int tbb, int tbest,
                                            const int *snip, int node) {
    if (ZGM == 1) {
        if (tbest < 0) return true;
        if (g > gb) return true;
        if (g < gb) return false;
        if (tb > tbb) return true;
        if (tb < tbb) return false;
        return t < tbest;
    }
    return move_better(g, t, gb, tbest, snip, node);
}

struct PartCounts {
    unsigned int b0, b1, b2, b3, b4, b5, b6, b7, b8;

    __device__ __forceinline__ void add(unsigned int mask) {
        unsigned int carry;
        carry = b0 & mask;
        b0 ^= mask;
        mask = carry;
        carry = b1 & mask;
        b1 ^= mask;
        mask = carry;
        carry = b2 & mask;
        b2 ^= mask;
        mask = carry;
        carry = b3 & mask;
        b3 ^= mask;
        mask = carry;
        carry = b4 & mask;
        b4 ^= mask;
        mask = carry;
        carry = b5 & mask;
        b5 ^= mask;
        mask = carry;
        carry = b6 & mask;
        b6 ^= mask;
        mask = carry;
        carry = b7 & mask;
        b7 ^= mask;
        mask = carry;
        b8 ^= mask;
    }

    __device__ __forceinline__ int get(int bit) const {
        return ((b0 >> bit) & 1u) | (((b1 >> bit) & 1u) << 1) | (((b2 >> bit) & 1u) << 2) |
               (((b3 >> bit) & 1u) << 3) | (((b4 >> bit) & 1u) << 4) | (((b5 >> bit) & 1u) << 5) |
               (((b6 >> bit) & 1u) << 6) | (((b7 >> bit) & 1u) << 7) | (((b8 >> bit) & 1u) << 8);
    }
};

__device__ __forceinline__ void keep_max_plane(unsigned int pl, unsigned int ph,
                                               unsigned int &lo, unsigned int &hi) {
    const unsigned int l = lo & pl, h = hi & ph;
    if (l | h) {
        lo = l;
        hi = h;
    }
}

__device__ __forceinline__ int best_count_target(const PartCounts &cl, const PartCounts &ch,
                                                  unsigned int lo, unsigned int hi,
                                                  const ulonglong2 choice) {
    lo &= (unsigned int)choice.x;
    hi &= (unsigned int)(choice.x >> 32);
    keep_max_plane(cl.b8, ch.b8, lo, hi);
    keep_max_plane(cl.b7, ch.b7, lo, hi);
    keep_max_plane(cl.b6, ch.b6, lo, hi);
    keep_max_plane(cl.b5, ch.b5, lo, hi);
    keep_max_plane(cl.b4, ch.b4, lo, hi);
    keep_max_plane(cl.b3, ch.b3, lo, hi);
    keep_max_plane(cl.b2, ch.b2, lo, hi);
    keep_max_plane(cl.b1, ch.b1, lo, hi);
    keep_max_plane(cl.b0, ch.b0, lo, hi);
    keep_max_plane((unsigned int)choice.y, (unsigned int)(choice.y >> 32), lo, hi);
    return lo ? __ffs(lo) - 1 : hi ? 31 + __ffs(hi) : -1;
}

__device__ __forceinline__ int best_gain_target(PartCounts cl, PartCounts ch,
                                                 unsigned int lo, unsigned int hi,
                                                 const ulonglong2 choice, const int *sizes,
                                                 const int node) {
    lo &= (unsigned int)choice.x;
    hi &= (unsigned int)(choice.x >> 32);
    if (!(lo | hi)) return -1;
    cl.add((unsigned int)choice.y);
    ch.add((unsigned int)(choice.y >> 32));
    keep_max_plane(cl.b8, ch.b8, lo, hi);
    keep_max_plane(cl.b7, ch.b7, lo, hi);
    keep_max_plane(cl.b6, ch.b6, lo, hi);
    keep_max_plane(cl.b5, ch.b5, lo, hi);
    keep_max_plane(cl.b4, ch.b4, lo, hi);
    keep_max_plane(cl.b3, ch.b3, lo, hi);
    keep_max_plane(cl.b2, ch.b2, lo, hi);
    keep_max_plane(cl.b1, ch.b1, lo, hi);
    keep_max_plane(cl.b0, ch.b0, lo, hi);
    unsigned long long winners = (unsigned long long)lo | ((unsigned long long)hi << 32);
    int best = __ffsll(winners) - 1;
    winners &= winners - 1;
    while (winners) {
        const int target = __ffsll(winners) - 1;
        winners &= winners - 1;
        if (move_better(0, target, 0, best, sizes, node)) best = target;
    }
    return best;
}

__device__ __forceinline__ void prepare_count_choices(int np, int cap, const int *sizes,
                                                       ulonglong2 *choices) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int low = lane < np ? sizes[lane] : cap;
    const int high = lane + 32 < np ? sizes[lane + 32] : cap;
    const unsigned int fl = __ballot_sync(0xffffffffu, lane < np && low < cap);
    const unsigned int fh = __ballot_sync(0xffffffffu, lane + 32 < np && high < cap);
    for (int source = warp; source < np; source += blockDim.x / 32) {
        const int size = sizes[source];
        const unsigned int bl = __ballot_sync(0xffffffffu, lane < np && size > low + 2);
        const unsigned int bh = __ballot_sync(0xffffffffu, lane + 32 < np && size > high + 2);
        if (lane == 0)
            choices[source] = make_ulonglong2(
                (unsigned long long)fl | ((unsigned long long)fh << 32),
                (unsigned long long)bl | ((unsigned long long)bh << 32));
    }
}

template <int ZGM>
__device__ __forceinline__ void
move_of_node(const int node, const int num_parts, const int max_part_size,
             const int *node_hyperedges, const int *node_offsets, const int *partition,
             const int *shared_nodes_in_part, const ulonglong2 *edge_flags, int *move_priorities,
             const ulonglong2 *choices) {
    move_priorities[node] = 0x80000000;

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;
    if (shared_nodes_in_part[current_part] <= 1) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) return;

    const unsigned long long current_bit = 1ULL << current_part;
    const unsigned long long others = other_parts(current_part);

    PartCounts counts_lo = {}, counts_hi = {};
    unsigned int cand_lo = 0u, cand_hi = 0u;
    int count_current_present = 0;
    int crit = 0;

    const int sample_step = node_degree / used_degree;
    const int sample_rem = node_degree - sample_step * used_degree;
    int sample_rel = 0, sample_acc = 0;

    for (int j = 0; j < used_degree; j++) {
        int hyperedge = node_hyperedges[start + sample_rel];
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
        if (flags_double & current_bit) count_current_present++;

        const unsigned long long flags = flags_all & others;
        cand_lo |= (unsigned int)flags;
        cand_hi |= (unsigned int)(flags >> 32);
        counts_lo.add((unsigned int)flags);
        counts_hi.add((unsigned int)(flags >> 32));

        sample_rel += sample_step;
        sample_acc += sample_rem;
        if (sample_acc >= used_degree) {
            sample_acc -= used_degree;
            sample_rel++;
        }
    }

    int best_gain = -999999, best_target = -1;
    best_target = ZGM == 1
        ? best_count_target(counts_lo, counts_hi, cand_lo, cand_hi, choices[current_part])
        : best_gain_target(counts_lo, counts_hi, cand_lo, cand_hi, choices[current_part],
                           shared_nodes_in_part, node);
    if (best_target >= 0) {
        const int cnt =
            best_target < 32 ? counts_lo.get(best_target) : counts_hi.get(best_target - 32);
        best_gain = cnt - count_current_present +
                    (shared_nodes_in_part[current_part] > shared_nodes_in_part[best_target] + 2);
    }

    if (best_target < 0 || best_target == current_part) return;

    int degree_weight = node_degree > 255 ? 255 : node_degree;
    int rank_byte = degree_weight + crit;
    if (rank_byte > 255) rank_byte = 255;

    int bg = best_gain > 32767 ? 32767 : best_gain;
    if (bg < -32768) bg = -32768;
    unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
    const unsigned int packed = ((unsigned int)(unsigned short)bg << 16) |
                                ((unsigned int)rank_byte << 8) | (unsigned int)t16;
    move_priorities[node] = (int)packed;
}

template <int ZGM>
__device__ __forceinline__ void
compute_refinement_moves_scalar(const int num_nodes, const int num_parts, const int max_part_size,
                                const int *node_hyperedges, const int *node_offsets,
                                const int *partition, const ulonglong2 *edge_flags,
                                int *move_priorities, const int *node_order, const int scalar_begin,
                                const int block_base, const int *shared_nodes_in_part,
                                const ulonglong2 *choices) {
    const int gid = (blockIdx.x - block_base) * blockDim.x + threadIdx.x;
    const int stride = (gridDim.x - block_base) * blockDim.x;
    for (int task = gid + scalar_begin; task < num_nodes; task += stride) {
        move_of_node<ZGM>(node_order[task], num_parts, max_part_size, node_hyperedges, node_offsets,
                          partition, shared_nodes_in_part, edge_flags, move_priorities, choices);
    }
}

template <int ZGM, int R>
__device__ __forceinline__ void
compute_refinement_moves_scalar_ilp(
    const int num_nodes, const int num_parts, const int max_part_size,
    const int *__restrict__ node_hyperedges, const int *__restrict__ node_offsets,
    const int *__restrict__ partition, const ulonglong2 *__restrict__ edge_flags,
    int *__restrict__ move_priorities, const int *__restrict__ node_order,
    const int scalar_begin, const int block_base, const int *__restrict__ shared_nodes_in_part,
    const ulonglong2 *__restrict__ choices) {
    static_assert(R >= 1 && R <= 4, "unsupported move interleave");
    const int gid = (blockIdx.x - block_base) * blockDim.x + threadIdx.x;
    const int stride = (gridDim.x - block_base) * blockDim.x;
    for (int task = gid + scalar_begin; task < num_nodes; task += R * stride) {
        int node[R], current_part[R], start[R], node_degree[R], used_degree[R] = {};
        unsigned long long current_bit[R], others[R];
        PartCounts counts_lo[R] = {}, counts_hi[R] = {};
        unsigned int cand_lo[R] = {}, cand_hi[R] = {};
        int count_current_present[R] = {}, crit[R] = {};
        int sample_step[R], sample_rem[R], sample_rel[R] = {}, sample_acc[R] = {};
        int max_degree = 0;

        #pragma unroll
        for (int r = 0; r < R; r++) {
            node[r] = task + r * stride < num_nodes ? node_order[task + r * stride] : -1;
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (node[r] < 0) continue;
            move_priorities[node[r]] = 0x80000000;
            current_part[r] = partition[node[r]];
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (node[r] < 0) continue;
            if ((unsigned)current_part[r] >= (unsigned)num_parts) continue;
            if (shared_nodes_in_part[current_part[r]] <= 1) continue;

            start[r] = node_offsets[node[r]];
            int end = node_offsets[node[r] + 1];
            node_degree[r] = end - start[r];
            used_degree[r] = node_degree[r] > 256 ? 256 : node_degree[r];
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (used_degree[r] <= 0) continue;

            current_bit[r] = 1ULL << current_part[r];
            others[r] = other_parts(current_part[r]);
            sample_step[r] = node_degree[r] / used_degree[r];
            sample_rem[r] = node_degree[r] - sample_step[r] * used_degree[r];
            if (used_degree[r] > max_degree) max_degree = used_degree[r];
        }

        for (int j = 0; j < max_degree; j++) {
            int hyperedge[R];
            ulonglong2 ef[R];
            #pragma unroll
            for (int r = 0; r < R; r++) {
                if (j < used_degree[r])
                    hyperedge[r] = node_hyperedges[start[r] + sample_rel[r]];
            }
            #pragma unroll
            for (int r = 0; r < R; r++) {
                if (j < used_degree[r]) ef[r] = __ldg(edge_flags + hyperedge[r]);
            }
            #pragma unroll
            for (int r = 0; r < R; r++) {
                if (j >= used_degree[r]) continue;
                unsigned long long flags_all = ef[r].x;
                unsigned long long flags_double = ef[r].y;

                if ((flags_all & current_bit[r]) != 0ULL &&
                    (flags_double & current_bit[r]) == 0ULL) {
                    int parts = __popcll(flags_all);
                    if (parts > 1) {
                        crit[r] += (parts - 1);
                        if (crit[r] > 255) crit[r] = 255;
                    }
                }
                if (flags_double & current_bit[r]) count_current_present[r]++;

                const unsigned long long flags = flags_all & others[r];
                cand_lo[r] |= (unsigned int)flags;
                cand_hi[r] |= (unsigned int)(flags >> 32);
                counts_lo[r].add((unsigned int)flags);
                counts_hi[r].add((unsigned int)(flags >> 32));

                sample_rel[r] += sample_step[r];
                sample_acc[r] += sample_rem[r];
                if (sample_acc[r] >= used_degree[r]) {
                    sample_acc[r] -= used_degree[r];
                    sample_rel[r]++;
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (used_degree[r] <= 0) continue;
            int best_gain = -999999, best_target = -1;
            best_target = ZGM == 1
                ? best_count_target(counts_lo[r], counts_hi[r], cand_lo[r], cand_hi[r],
                                    choices[current_part[r]])
                : best_gain_target(counts_lo[r], counts_hi[r], cand_lo[r], cand_hi[r],
                                   choices[current_part[r]], shared_nodes_in_part, node[r]);
            if (best_target >= 0) {
                const int cnt = best_target < 32 ? counts_lo[r].get(best_target)
                                                : counts_hi[r].get(best_target - 32);
                best_gain = cnt - count_current_present[r] +
                            (shared_nodes_in_part[current_part[r]] >
                             shared_nodes_in_part[best_target] + 2);
            }

            if (best_target < 0 || best_target == current_part[r]) continue;

            int degree_weight = node_degree[r] > 255 ? 255 : node_degree[r];
            int rank_byte = degree_weight + crit[r];
            if (rank_byte > 255) rank_byte = 255;

            int bg = best_gain > 32767 ? 32767 : best_gain;
            if (bg < -32768) bg = -32768;
            unsigned short t16 = (unsigned short)(best_target & 63) | ((node[r] & 3) << 6);
            const unsigned int packed = ((unsigned int)(unsigned short)bg << 16) |
                                        ((unsigned int)rank_byte << 8) | (unsigned int)t16;
            move_priorities[node[r]] = (int)packed;
        }
    }
}

__device__ __forceinline__ unsigned int plane_sum(unsigned int &a, const unsigned int b,
                                                  const unsigned int c) {
    const unsigned int x = a ^ b;
    const unsigned int carry = (a & b) | (c & x);
    a = x ^ c;
    return carry;
}

__device__ __forceinline__ void planes_add(PartCounts &a, const PartCounts &b) {
    unsigned int c = 0u;
    c = plane_sum(a.b0, b.b0, c);
    c = plane_sum(a.b1, b.b1, c);
    c = plane_sum(a.b2, b.b2, c);
    c = plane_sum(a.b3, b.b3, c);
    c = plane_sum(a.b4, b.b4, c);
    c = plane_sum(a.b5, b.b5, c);
    c = plane_sum(a.b6, b.b6, c);
    c = plane_sum(a.b7, b.b7, c);
    a.b8 = a.b8 ^ b.b8 ^ c;
}

template <int L>
__device__ __forceinline__ PartCounts planes_shfl_xor(const PartCounts &a, const unsigned int mask,
                                                      const int o) {
    PartCounts r;
    r.b0 = __shfl_xor_sync(mask, a.b0, o, L);
    r.b1 = __shfl_xor_sync(mask, a.b1, o, L);
    r.b2 = __shfl_xor_sync(mask, a.b2, o, L);
    r.b3 = __shfl_xor_sync(mask, a.b3, o, L);
    r.b4 = __shfl_xor_sync(mask, a.b4, o, L);
    r.b5 = __shfl_xor_sync(mask, a.b5, o, L);
    r.b6 = __shfl_xor_sync(mask, a.b6, o, L);
    r.b7 = __shfl_xor_sync(mask, a.b7, o, L);
    r.b8 = __shfl_xor_sync(mask, a.b8, o, L);
    return r;
}

template <int ZGM, int L>
__device__ __forceinline__ void
move_of_node_lanes(const int node, const int lane, const unsigned int gmask, const int num_parts,
                   const int max_part_size, const int *node_hyperedges, const int *node_offsets,
                   const int *partition, const int *shared_nodes_in_part,
                   const ulonglong2 *edge_flags, int *move_priorities, const ulonglong2 *choices) {
    if (lane == 0) move_priorities[node] = 0x80000000;

    const int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;
    if (shared_nodes_in_part[current_part] <= 1) return;

    const int start = node_offsets[node];
    const int node_degree = node_offsets[node + 1] - start;
    const int used_degree = node_degree > 256 ? 256 : node_degree;
    if (used_degree <= 0) return;

    const unsigned long long current_bit = 1ULL << current_part;
    const unsigned long long others = other_parts(current_part);

    PartCounts counts_lo = {}, counts_hi = {};
    unsigned int cand_lo = 0u, cand_hi = 0u;
    int count_current_present = 0;
    int crit = 0;

    const int step_q = (L * node_degree) / used_degree;
    const int step_r = L * node_degree - step_q * used_degree;
    int rel = (lane * node_degree) / used_degree;
    int rem = lane * node_degree - rel * used_degree;

    for (int j = lane; j < used_degree; j += L) {
        const int hyperedge = node_hyperedges[start + rel];
        const ulonglong2 ef = __ldg(edge_flags + hyperedge);
        const unsigned long long flags_all = ef.x;
        const unsigned long long flags_double = ef.y;

        if ((flags_all & current_bit) != 0ULL && (flags_double & current_bit) == 0ULL) {
            const int parts = __popcll(flags_all);
            if (parts > 1) crit += (parts - 1);
        }
        if (flags_double & current_bit) count_current_present++;

        const unsigned long long flags = flags_all & others;
        cand_lo |= (unsigned int)flags;
        cand_hi |= (unsigned int)(flags >> 32);
        counts_lo.add((unsigned int)flags);
        counts_hi.add((unsigned int)(flags >> 32));

        rel += step_q;
        rem += step_r;
        if (rem >= used_degree) {
            rem -= used_degree;
            rel++;
        }
    }

    for (int o = 1; o < L; o <<= 1) {
        planes_add(counts_lo, planes_shfl_xor<L>(counts_lo, gmask, o));
        planes_add(counts_hi, planes_shfl_xor<L>(counts_hi, gmask, o));
        cand_lo |= __shfl_xor_sync(gmask, cand_lo, o, L);
        cand_hi |= __shfl_xor_sync(gmask, cand_hi, o, L);
        count_current_present += __shfl_xor_sync(gmask, count_current_present, o, L);
        crit += __shfl_xor_sync(gmask, crit, o, L);
    }
    if (crit > 255) crit = 255;

    const unsigned long long lane_slice = ((1ULL << (64 / L)) - 1ULL) << (lane * (64 / L));
    const int current_size = shared_nodes_in_part[current_part];
    int best_gain = -999999, best_target = -1, best_tb = 0;
    unsigned long long tmp = (cand_lo | ((unsigned long long)cand_hi << 32)) & lane_slice;
    best_target = ZGM == 1
        ? best_count_target(counts_lo, counts_hi, (unsigned int)tmp,
                            (unsigned int)(tmp >> 32), choices[current_part])
        : best_gain_target(counts_lo, counts_hi, (unsigned int)tmp,
                           (unsigned int)(tmp >> 32), choices[current_part],
                           shared_nodes_in_part, node);
    if (best_target >= 0) {
        best_tb =
            best_target < 32 ? counts_lo.get(best_target) : counts_hi.get(best_target - 32);
        best_gain = best_tb - count_current_present +
                    (current_size > shared_nodes_in_part[best_target] + 2);
    }

    for (int off = L / 2; off > 0; off >>= 1) {
        const int og = __shfl_down_sync(gmask, best_gain, off, L);
        const int ot = __shfl_down_sync(gmask, best_target, off, L);
        const int otb = __shfl_down_sync(gmask, best_tb, off, L);
        if ((lane < off) && ot >= 0 &&
            cand_better<ZGM>(og, ot, otb, best_gain, best_tb, best_target, shared_nodes_in_part,
                             node)) {
            best_gain = og;
            best_target = ot;
            best_tb = otb;
        }
    }

    if (lane != 0) return;
    if (best_target < 0 || best_target == current_part) return;

    const int degree_weight = node_degree > 255 ? 255 : node_degree;
    int rank_byte = degree_weight + crit;
    if (rank_byte > 255) rank_byte = 255;

    int bg = best_gain > 32767 ? 32767 : best_gain;
    if (bg < -32768) bg = -32768;
    unsigned short t16 = (unsigned short)(best_target & 63) | ((node & 3) << 6);
    const unsigned int packed = ((unsigned int)(unsigned short)bg << 16) |
                                ((unsigned int)rank_byte << 8) | (unsigned int)t16;
    move_priorities[node] = (int)packed;
}

template <int ZGM, int L>
__device__ __forceinline__ void compute_refinement_moves_lanes(
    const int block, const int task_begin, const int task_end, const int num_parts,
    const int max_part_size, const int *node_hyperedges, const int *node_offsets,
    const int *partition, const int *shared_nodes_in_part, const ulonglong2 *edge_flags,
    int *move_priorities, const int *node_order, const ulonglong2 *choices) {
    const int task = task_begin + block * (int)(blockDim.x / L) + (int)(threadIdx.x / L);
    if (task >= task_end) return;
    const int lane = (int)(threadIdx.x & (L - 1));
    const unsigned int gmask = (0xFFFFFFFFu >> (32 - L)) << (threadIdx.x & 31 & ~(L - 1));
    move_of_node_lanes<ZGM, L>(node_order[task], lane, gmask, num_parts, max_part_size,
                               node_hyperedges, node_offsets, partition, shared_nodes_in_part,
                               edge_flags, move_priorities, choices);
}

__device__ __forceinline__ ulonglong2 edge_flags_of(const int start, const int end,
                                                    const int num_nodes, const int *hyperedge_nodes,
                                                    const int *partition) {
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

__device__ __forceinline__ void
precompute_edge_flags(const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,
                      const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags) {
    const int stride = gridDim.x * blockDim.x;
    for (int hedge = blockIdx.x * blockDim.x + threadIdx.x; hedge < num_hyperedges;
         hedge += stride) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        edge_flags[hedge] = edge_flags_of(start, end, num_nodes, hyperedge_nodes, partition);
    }
}

__device__ __forceinline__ void expand_node_hedges(const int n_pairs, const int *pairs,
                                                   const int *node_hyperedges,
                                                   const int *node_offsets, int *hedge_list) {
    const int stride = gridDim.x * blockDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n_pairs; t += stride) {
        const int v = pairs[2 * t];
        int dst = pairs[2 * t + 1];
        const int a = node_offsets[v];
        const int b = node_offsets[v + 1];
        for (int i = a; i < b; i++)
            hedge_list[dst++] = node_hyperedges[i];
    }
}

__device__ __forceinline__ void
precompute_edge_flags_hlist(const int n_list, const int *hedge_list, const int stamp,
                            int *edge_stamp, const int num_hyperedges, const int num_nodes,
                            const int *hyperedge_nodes, const int *hyperedge_offsets,
                            const int *partition, ulonglong2 *edge_flags) {
    const int stride = gridDim.x * blockDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n_list; t += stride) {
        int hedge = hedge_list[t];
        if (hedge < 0 || hedge >= num_hyperedges) continue;
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        if (end - start > 32 && atomicExch(edge_stamp + hedge, stamp) == stamp) continue;
        edge_flags[hedge] = edge_flags_of(start, end, num_nodes, hyperedge_nodes, partition);
    }
}

__device__ __forceinline__ void edge_flags_warp(const int h, const int start, const int end,
                                                const int lane, const int num_nodes,
                                                const int *hyperedge_nodes, const int *partition,
                                                ulonglong2 *edge_flags) {
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

__device__ __forceinline__ void edge_flags_small(const int num_hyperedges, const int num_nodes,
                                                 const int *hyperedge_nodes,
                                                 const int *hyperedge_offsets, const int *partition,
                                                 ulonglong2 *edge_flags) {
    const int stride = gridDim.x * blockDim.x;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < num_hyperedges; gid += stride) {
        const int start = hyperedge_offsets[gid];
        const int end = hyperedge_offsets[gid + 1];
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

__device__ __forceinline__ void
precompute_edge_flags_warp_list(const int num_hyperedges, const int num_nodes,
                                const int *hyperedge_nodes, const int *hyperedge_offsets,
                                const int *partition, ulonglong2 *edge_flags, const int *big_ids,
                                const int n_big) {
    edge_flags_small(num_hyperedges, num_nodes, hyperedge_nodes, hyperedge_offsets, partition,
                     edge_flags);

    const int lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int gw = blockIdx.x * wpb + (threadIdx.x >> 5);
    const int tw = gridDim.x * wpb;

    for (int i = gw; i < n_big; i += tw) {
        const int h = big_ids[i];
        const int start = hyperedge_offsets[h];
        const int end = hyperedge_offsets[h + 1];
        edge_flags_warp(h, start, end, lane, num_nodes, hyperedge_nodes, partition, edge_flags);
    }
}

}

#define TRACK_KERNELS_COMMON(S)                                                                    \
    extern "C" __global__ void choose_elite_per_hyperedge_##S(                                     \
        const int num_hyperedges, const int num_nodes, const int num_parts, const int num_elites,  \
        const int *elite_partitions, const int *elite_order, const int *hyperedge_offsets,         \
        const int *hyperedge_nodes, int *hedge_choice_elite) {                                     \
        track::choose_elite_per_hyperedge(num_hyperedges, num_nodes, num_parts, num_elites,        \
                                          elite_partitions, elite_order, hyperedge_offsets,        \
                                          hyperedge_nodes, hedge_choice_elite);                    \
    }                                                                                              \
    extern "C" __global__ void assign_from_elite_votes_##S(                                        \
        const int num_nodes, const int num_parts, const int *elite_partitions,                     \
        const int *hedge_choice_elite, const int *node_hyperedges, const int *node_offsets,        \
        const ulonglong2 *edge_flags_best, int *partition_out) {                                   \
        track::assign_from_elite_votes(num_nodes, num_parts, elite_partitions, hedge_choice_elite, \
                                       node_hyperedges, node_offsets, edge_flags_best,             \
                                       partition_out);                                             \
    }                                                                                              \
    extern "C" __global__ void hyperedge_clustering_##S(                                           \
        const int num_hyperedges, const int num_clusters, const int *hyperedge_offsets,            \
        const int *hyperedge_nodes, int *hyperedge_clusters) {                                     \
        track::hyperedge_clustering(num_hyperedges, num_clusters, hyperedge_offsets,               \
                                    hyperedge_nodes, hyperedge_clusters);                          \
    }                                                                                              \
    extern "C" __global__ void compute_node_preferences_##S(                                       \
        const int num_nodes, const int num_parts, const int num_hedge_clusters,                    \
        const int *node_hyperedges, const int *node_offsets, const int *hyperedge_clusters,        \
        const int *hyperedge_offsets, int *pref_parts, int *pref_priorities) {                     \
        track::compute_node_preferences(num_nodes, num_parts, num_hedge_clusters, node_hyperedges, \
                                        node_offsets, hyperedge_clusters, hyperedge_offsets,       \
                                        pref_parts, pref_priorities);                              \
    }                                                                                              \
    extern "C" __global__ void execute_node_assignments_##S(                                       \
        const int num_nodes, const int num_parts, const int max_part_size,                         \
        const int *sorted_nodes, const int *sorted_parts, int *partition, int *nodes_in_part) {    \
        track::execute_node_assignments(num_nodes, num_parts, max_part_size, sorted_nodes,         \
                                        sorted_parts, partition, nodes_in_part);                   \
    }                                                                                              \
    extern "C" __global__ void compute_swap_gains_extended_##S(                                    \
        const int num_nodes, const int num_parts, const int neg_gain_thresh,                       \
        const int *node_hyperedges, const int *node_offsets, const int *partition,                 \
        const ulonglong2 *edge_flags, int *swap_gains) {                                           \
        unsigned short part_counts[64];                                                            \
        track::compute_swap_gains_extended(num_nodes, num_parts, neg_gain_thresh, node_hyperedges, \
                                           node_offsets, partition, edge_flags, swap_gains,        \
                                           part_counts);                                           \
    }                                                                                              \
    extern "C" __global__ void compute_connectivity_##S(                                           \
        const int num_hyperedges, const int *hyperedge_nodes, const int *hyperedge_offsets,        \
        const int *partition, int *connectivity) {                                                 \
        track::compute_connectivity(num_hyperedges, hyperedge_nodes, hyperedge_offsets, partition, \
                                    connectivity);                                                 \
    }                                                                                              \
    extern "C" __global__ void reduce_connectivity_sum_##S(                                        \
        const int num_hyperedges, const int *connectivity, int *total_connectivity_blocks) {       \
        track::reduce_connectivity_sum(num_hyperedges, connectivity, total_connectivity_blocks);   \
    }                                                                                              \
    extern "C" __global__ void balance_final_##S(const int num_nodes, const int num_parts,         \
                                                 const int min_part_size, const int max_part_size, \
                                                 int *partition, int *nodes_in_part) {             \
        track::balance_final(num_nodes, num_parts, min_part_size, max_part_size, partition,        \
                             nodes_in_part);                                                       \
    }

#define TRACK_MOVES_WARP8(S, ZGM)                                                              \
    extern "C" __global__ void __launch_bounds__(256) compute_refinement_moves_warp8_##S(      \
        const int num_nodes, const int num_parts, const int max_part_size,                     \
        const int *node_hyperedges, const int *node_offsets, const int *partition,             \
        const int *nodes_in_part, const ulonglong2 *edge_flags, int *move_priorities,          \
        const int *node_order, const int n8, const int n4, const int n2,                       \
        const unsigned long long *gr) {                                                        \
        if (gr && gr[0] != 0) return;                                                          \
        __shared__ int shared_nodes_in_part[64];                                               \
        __shared__ ulonglong2 choices[64];                                                     \
        if (threadIdx.x < 64 && threadIdx.x < num_parts) {                                     \
            shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];                    \
        }                                                                                      \
        __syncthreads();                                                                       \
        track::prepare_count_choices(num_parts, max_part_size, shared_nodes_in_part, choices); \
        __syncthreads();                                                                       \
        const int per8 = (int)(blockDim.x >> 3), per4 = (int)(blockDim.x >> 2);                \
        const int per2 = (int)(blockDim.x >> 1);                                               \
        const int b8 = (n8 + per8 - 1) / per8;                                                 \
        const int b4 = b8 + (n4 - n8 + per4 - 1) / per4;                                       \
        const int b2 = b4 + (n2 - n4 + per2 - 1) / per2;                                       \
        const int b = (int)blockIdx.x;                                                         \
        if (b < b8) {                                                                          \
            track::compute_refinement_moves_lanes<ZGM, 8>(                                     \
                b, 0, n8, num_parts, max_part_size, node_hyperedges, node_offsets, partition,  \
                shared_nodes_in_part, edge_flags, move_priorities, node_order, choices);       \
        } else if (b < b4) {                                                                   \
            track::compute_refinement_moves_lanes<ZGM, 4>(                                     \
                b - b8, n8, n4, num_parts, max_part_size, node_hyperedges, node_offsets,       \
                partition, shared_nodes_in_part, edge_flags, move_priorities, node_order, choices); \
        } else if (b < b2) {                                                                   \
            track::compute_refinement_moves_lanes<ZGM, 2>(                                     \
                b - b4, n4, n2, num_parts, max_part_size, node_hyperedges, node_offsets,       \
                partition, shared_nodes_in_part, edge_flags, move_priorities, node_order, choices); \
        } else {                                                                               \
            track::compute_refinement_moves_scalar<ZGM>(                                       \
                num_nodes, num_parts, max_part_size, node_hyperedges, node_offsets, partition, \
                edge_flags, move_priorities, node_order, n2, b2, shared_nodes_in_part, choices); \
        }                                                                                      \
    }

#define TRACK_MOVES_WARP8_ILP(S, ZGM, R)                                                        \
    extern "C" __global__ void __launch_bounds__(256) compute_refinement_moves_warp8_##S##_ilp##R( \
        const int num_nodes, const int num_parts, const int max_part_size,                     \
        const int *__restrict__ node_hyperedges, const int *__restrict__ node_offsets,        \
        const int *__restrict__ partition, const int *__restrict__ nodes_in_part,             \
        const ulonglong2 *__restrict__ edge_flags, int *__restrict__ move_priorities,          \
        const int *__restrict__ node_order, const int n8, const int n4, const int n2,           \
        const unsigned long long *gr) {                                                        \
        if (gr && gr[0] != 0) return;                                                          \
        __shared__ int shared_nodes_in_part[64];                                               \
        __shared__ ulonglong2 choices[64];                                                     \
        if (threadIdx.x < 64 && threadIdx.x < num_parts) {                                     \
            shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];                    \
        }                                                                                      \
        __syncthreads();                                                                       \
        track::prepare_count_choices(num_parts, max_part_size, shared_nodes_in_part, choices); \
        __syncthreads();                                                                       \
        const int per8 = (int)(blockDim.x >> 3), per4 = (int)(blockDim.x >> 2);                \
        const int per2 = (int)(blockDim.x >> 1);                                               \
        const int b8 = (n8 + per8 - 1) / per8;                                                 \
        const int b4 = b8 + (n4 - n8 + per4 - 1) / per4;                                       \
        const int b2 = b4 + (n2 - n4 + per2 - 1) / per2;                                       \
        const int b = (int)blockIdx.x;                                                         \
        if (b < b8) {                                                                          \
            track::compute_refinement_moves_lanes<ZGM, 8>(                                     \
                b, 0, n8, num_parts, max_part_size, node_hyperedges, node_offsets, partition,  \
                shared_nodes_in_part, edge_flags, move_priorities, node_order, choices);       \
        } else if (b < b4) {                                                                   \
            track::compute_refinement_moves_lanes<ZGM, 4>(                                     \
                b - b8, n8, n4, num_parts, max_part_size, node_hyperedges, node_offsets,       \
                partition, shared_nodes_in_part, edge_flags, move_priorities, node_order, choices); \
        } else if (b < b2) {                                                                   \
            track::compute_refinement_moves_lanes<ZGM, 2>(                                     \
                b - b4, n4, n2, num_parts, max_part_size, node_hyperedges, node_offsets,       \
                partition, shared_nodes_in_part, edge_flags, move_priorities, node_order, choices); \
        } else {                                                                               \
            track::compute_refinement_moves_scalar_ilp<ZGM, R>(                                \
                num_nodes, num_parts, max_part_size, node_hyperedges, node_offsets, partition, \
                edge_flags, move_priorities, node_order, n2, b2, shared_nodes_in_part, choices); \
        }                                                                                      \
    }

#define TRACK_EDGE_FLAGS_WARP_LIST(S)                                                             \
    extern "C" __global__ void precompute_edge_flags_wpe_##S(                                     \
        const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,                \
        const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags,               \
        const int *big_ids, const int n_big, const unsigned long long *gr) {                      \
        if (gr && (gr[0] != 0 || gr[8] != 0)) return;                                             \
        track::precompute_edge_flags_warp_list(num_hyperedges, num_nodes, hyperedge_nodes,        \
                                               hyperedge_offsets, partition, edge_flags, big_ids, \
                                               n_big);                                            \
    }

#define TRACK_EDGE_FLAGS_PLAIN(S)                                                                 \
    extern "C" __global__ void precompute_edge_flags_##S(                                         \
        const int num_hyperedges, const int num_nodes, const int *hyperedge_nodes,                \
        const int *hyperedge_offsets, const int *partition, ulonglong2 *edge_flags,               \
        const unsigned long long *gr) {                                                           \
        if (gr && (gr[0] != 0 || gr[8] != 0)) return;                                             \
        track::precompute_edge_flags(num_hyperedges, num_nodes, hyperedge_nodes,                  \
                                     hyperedge_offsets, partition, edge_flags);                   \
    }                                                                                             \
    extern "C" __global__ void precompute_edge_flags_hlist_##S(                                   \
        int n_list, const int *hedge_list, int stamp, int *edge_stamp, const int num_hyperedges,  \
        const int num_nodes, const int *hyperedge_nodes, const int *hyperedge_offsets,            \
        const int *partition, ulonglong2 *edge_flags, const unsigned long long *gr) {             \
        if (gr) {                                                                                 \
            if (gr[0] != 0) return;                                                               \
            if (gr[30] && gr[8] == 0) {                                                           \
                track::precompute_edge_flags(num_hyperedges, num_nodes, hyperedge_nodes,          \
                                             hyperedge_offsets, partition, edge_flags);           \
                return;                                                                           \
            }                                                                                     \
            if (gr[8] == 0 || gr[6] == 0) return;                                                 \
            n_list = (int)gr[6];                                                                  \
            stamp = (int)gr[1] + 1;                                                               \
        }                                                                                         \
        track::precompute_edge_flags_hlist(n_list, hedge_list, stamp, edge_stamp, num_hyperedges, \
                                           num_nodes, hyperedge_nodes, hyperedge_offsets,         \
                                           partition, edge_flags);                                \
    }                                                                                             \
    extern "C" __global__ void expand_node_hedges_##S(                                            \
        int n_pairs, const int *pairs, const int *node_hyperedges, const int *node_offsets,       \
        int *hedge_list, const unsigned long long *gr) {                                          \
        if (gr) {                                                                                 \
            if (gr[0] != 0 || gr[8] == 0 || gr[6] == 0) return;                                   \
            n_pairs = (int)gr[5];                                                                 \
        }                                                                                         \
        track::expand_node_hedges(n_pairs, pairs, node_hyperedges, node_offsets, hedge_list);     \
    }

TRACK_KERNELS_COMMON(20k)
TRACK_MOVES_WARP8(20k, 0)
TRACK_MOVES_WARP8_ILP(20k, 0, 1)
TRACK_MOVES_WARP8_ILP(20k, 0, 2)
TRACK_MOVES_WARP8_ILP(20k, 0, 3)
TRACK_MOVES_WARP8_ILP(20k, 0, 4)
TRACK_EDGE_FLAGS_WARP_LIST(20k)

TRACK_KERNELS_COMMON(50k)
TRACK_MOVES_WARP8(50k, 0)
TRACK_MOVES_WARP8_ILP(50k, 0, 1)
TRACK_MOVES_WARP8_ILP(50k, 0, 2)
TRACK_MOVES_WARP8_ILP(50k, 0, 3)
TRACK_MOVES_WARP8_ILP(50k, 0, 4)
TRACK_EDGE_FLAGS_WARP_LIST(50k)

TRACK_KERNELS_COMMON(100k)
TRACK_MOVES_WARP8(100k, 1)
TRACK_MOVES_WARP8_ILP(100k, 1, 1)
TRACK_MOVES_WARP8_ILP(100k, 1, 2)
TRACK_MOVES_WARP8_ILP(100k, 1, 3)
TRACK_MOVES_WARP8_ILP(100k, 1, 4)
TRACK_EDGE_FLAGS_PLAIN(100k)

TRACK_KERNELS_COMMON(200k)
TRACK_MOVES_WARP8(200k, 0)
TRACK_MOVES_WARP8_ILP(200k, 0, 1)
TRACK_MOVES_WARP8_ILP(200k, 0, 2)
TRACK_MOVES_WARP8_ILP(200k, 0, 3)
TRACK_MOVES_WARP8_ILP(200k, 0, 4)
TRACK_EDGE_FLAGS_PLAIN(200k)

#undef TRACK_KERNELS_COMMON
#undef TRACK_MOVES_WARP8
#undef TRACK_MOVES_WARP8_ILP
#undef TRACK_EDGE_FLAGS_WARP_LIST
#undef TRACK_EDGE_FLAGS_PLAIN

template <bool FLAGS>
__device__ __forceinline__ void update_moved_edge_counts_100k(
    const unsigned long long *h, int num_hyperedges, int num_nodes, const int *pairs,
    const int *node_hyperedges, const int *node_offsets, const int *partition,
    unsigned int *counts, ulonglong2 *edge_flags) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total = (int)h[6], moves = (int)h[5];
    const int chunk = (total + stride - 1) / stride;
    int at = tid * chunk;
    const int stop = min(total, at + chunk);
    if (at >= stop) return;
    int low = 0, high = moves;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (pairs[2 * mid + 1] <= at)
            low = mid + 1;
        else
            high = mid;
    }
    int i = low - 1;
    while (at < stop && i < moves) {
        const int v = pairs[2 * i], first = pairs[2 * i + 1];
        const int end = i + 1 < moves ? pairs[2 * (i + 1) + 1] : total;
        const int last = min(stop, end);
        const int s = pairs[2 * num_nodes + i], t = partition[v];
        for (; at < last; ++at) {
            const int edge = node_hyperedges[node_offsets[v] + at - first];
            if (edge < 0 || edge >= num_hyperedges) continue;
            if (FLAGS) {
                for (int side = 0; side < 2; ++side) {
                    const int p = side ? t : s;
                    const unsigned int word = counts[32 * edge + (p >> 1)];
                    const unsigned int count = (word >> ((p & 1) * 16)) & 65535u;
                    const unsigned int bit = 1u << (p & 31);
                    unsigned int *mask =
                        reinterpret_cast<unsigned int *>(edge_flags + edge) + (p >> 5);
                    if (count >= 1u)
                        atomicOr(mask, bit);
                    else
                        atomicAnd(mask, ~bit);
                    if (count >= 2u)
                        atomicOr(mask + 2, bit);
                    else
                        atomicAnd(mask + 2, ~bit);
                }
            } else {
                atomicAdd(counts + 32 * edge + (t >> 1), 1u << ((t & 1) * 16));
                atomicAdd(counts + 32 * edge + (s >> 1), 0u - (1u << ((s & 1) * 16)));
            }
        }
        ++i;
    }
}

extern "C" __global__ void __launch_bounds__(128) update_edge_pin_counts_100k(
    const unsigned long long *h, int num_hyperedges, int num_nodes, const int *pairs,
    const int *node_hyperedges, const int *node_offsets, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *partition, unsigned int *counts,
    ulonglong2 *edge_flags) {
    if (h[0]) return;
    if (h[8]) {
        update_moved_edge_counts_100k<false>(h, num_hyperedges, num_nodes, pairs,
                                           node_hyperedges, node_offsets, partition,
                                           counts, edge_flags);
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = blockDim.x / 32;
    __shared__ int histogram[4 * 64];
    int *wc = histogram + warp * 64;
    for (int edge = blockIdx.x * warps + warp; edge < num_hyperedges;
         edge += gridDim.x * warps) {
        const int first = hyperedge_offsets[edge], last = hyperedge_offsets[edge + 1];
        wc[lane] = 0;
        wc[lane + 32] = 0;
        __syncwarp();
        for (int pos = first; pos < last; pos += 32) {
            int p = -1;
            if (pos + lane < last) {
                const int node = hyperedge_nodes[pos + lane];
                if (node >= 0 && node < num_nodes) p = partition[node];
            }
            if (p >= 0 && p < 64) atomicAdd(wc + p, 1);
        }
        __syncwarp();
        counts[32 * edge + lane] =
            (unsigned int)wc[2 * lane] | ((unsigned int)wc[2 * lane + 1] << 16);
        const int lo = wc[lane], hi = wc[lane + 32];
        const unsigned int all_lo = __ballot_sync(0xffffffffu, lo > 0);
        const unsigned int all_hi = __ballot_sync(0xffffffffu, hi > 0);
        const unsigned int double_lo = __ballot_sync(0xffffffffu, lo > 1);
        const unsigned int double_hi = __ballot_sync(0xffffffffu, hi > 1);
        if (lane == 0) {
            edge_flags[edge] = make_ulonglong2(
                (unsigned long long)all_lo | ((unsigned long long)all_hi << 32),
                (unsigned long long)double_lo | ((unsigned long long)double_hi << 32));
        }
        __syncwarp();
    }
}

extern "C" __global__ void __launch_bounds__(128) update_edge_flags_from_counts_100k(
    const unsigned long long *h, int num_hyperedges, int num_nodes, const int *pairs,
    const int *node_hyperedges, const int *node_offsets, const int *partition,
    unsigned int *counts, ulonglong2 *edge_flags) {
    if (h[0] || !h[8]) return;
    update_moved_edge_counts_100k<true>(h, num_hyperedges, num_nodes, pairs,
                                      node_hyperedges, node_offsets, partition,
                                      counts, edge_flags);
}
