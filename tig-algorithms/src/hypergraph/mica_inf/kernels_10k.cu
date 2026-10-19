// Device kernels of the 10k track (and the fallback for any other size).
// Invariants: 1 <= num_parts <= 64 (64-bit block masks), num_nodes <= 65536 and node degrees < 65536
// (balance keys and counters), blocks of num_parts..=128 threads (a multiple of 32), no inter-block wait.
// The extern "C" entries keep their unqualified names inside the namespace.

#include <stdint.h>
#include <cuda_runtime.h>

namespace track10k {

// Per hyperedge, the elite whose partition cuts it the least (ties: earliest in elite_order).
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

// Per node, the block voted by the elites chosen for its (at most 768 sampled) hyperedges;
// ties go to the block present in the most of them under edge_flags_all_best.
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

// Hashes each hyperedge into one of num_clusters buckets, stratified by hyperedge size.
extern "C" __global__ void hyperedge_clustering_10k(
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
}

// Per node, the preferred block (from its majority cluster, votes jittered by restart_id and
// random_seed) and an assignment priority.
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

// Writes the host-computed assignment, one thread per node.
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

// Per hyperedge, (blocks touched - 1).
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

// Single-thread random perturbation: perturb_strength percent of the nodes move to a random
// block that has room.
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

// Per node, the parent block that agrees with more (weighted) neighbours, then the parent whose
// hyperedges are cut less.
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

    int part_a = partition_a[node];
    int part_b = partition_b[node];

    int chosen_part;
    if (part_a == part_b) {
        chosen_part = part_a;
    } else {
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int degree = end - start;
        int used_deg = degree > 64 ? 64 : degree;

        int cost_a = 0, cost_b = 0;
        int agreed_a = 0, agreed_b = 0;

        for (int j = 0; j < used_deg; j++) {
            int rel = (int)(((long long)j * degree) / used_deg);
            int hedge = node_hyperedges[start + rel];

            int h_start = hyperedge_offsets[hedge];
            int h_end = hyperedge_offsets[hedge + 1];
            int hedge_size = h_end - h_start;
            int weight = 100 / (hedge_size + 1);

            if (hedge_size <= 64) {
                for (int k = h_start; k < h_end; k++) {
                    int nbr = hyperedge_nodes[k];
                    if (nbr != node) {
                        int pa = partition_a[nbr];
                        int pb = partition_b[nbr];
                        if (pa == pb) {
                            if (pa == part_a) agreed_a += weight;
                            else if (pa == part_b) agreed_b += weight;
                        }
                    }
                }
            }

            unsigned long long fa = edge_flags_a[hedge];
            unsigned long long fb = edge_flags_b[hedge];

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

// Per hyperedge spanning 2..8 blocks, up to four moves of its least-populated block (at most 4
// pins) towards its most-populated block, encoded ((node + 1) << 6) | target.
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

    unsigned long long flags_all = edge_flags_all[hedge];
    int span = __popcll(flags_all);

    if (span >= 2 && span <= 8) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];

        int part_counts[64];

        unsigned long long temp_flags = flags_all;
        while (temp_flags) {
            int p = __ffsll(temp_flags) - 1;
            temp_flags &= temp_flags - 1;
            part_counts[p] = 0;
        }

        for (int k = start; k < end; k++) {
            int node = hyperedge_nodes[k];
            int p = partition[node];
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
                int node = hyperedge_nodes[k];
                if (partition[node] == min_p) {
                    hedge_moves[base_idx + move_idx] = ((node + 1) << 6) | max_p;
                    move_idx++;
                    if (move_idx >= 4) break;
                }
            }
        }
    }
}

// Per node, its four best target blocks (gain scaled to the full degree when sampled; tabu nodes
// need a positive gain), packed as (gain << 16) | target; 0 means no entry, so a zero gain
// towards block 0 is dropped.
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

    int current_part = partition[node];
    if ((unsigned)current_part >= (unsigned)num_parts) return;

    bool is_tabu = (node_tabu_until[node] > global_round);

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
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
        int hyperedge = __ldg(node_hyperedges + start + sample_rel);

        unsigned long long flags_all = __ldg(edge_flags_all + hyperedge);
        unsigned long long flags_double = __ldg(edge_flags_double + hyperedge);
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

// Single-thread path relinking: moves a seeded fraction of the nodes that differ from
// best_partition back to their best block.
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

        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int start_node = (int)(state % (unsigned long long)num_nodes);

        for (int i = 0; i < num_nodes && moves_made < target_moves; i++) {
            int node = start_node + i;
            if (node >= num_nodes) node -= num_nodes;

            int cur_part = partition[node];
            int best_part = best_partition[node];

            if (cur_part == best_part) continue;

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

// Single-thread perturbation of the listed hyperedges: a quarter of their minority pins join
// the majority block.
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

// Single-thread ruin and recreate of one random block: half its nodes are reassigned to the
// neighbouring block with the most pins, or to a random block with room.
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

// Single-thread perturbation of sampled high-degree nodes and a quarter of their small-hyperedge
// neighbours, at most num_nodes / 20 moves.
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

// Block masks per hyperedge (all = blocks present, double = blocks present at least twice):
// one thread per hyperedge of at most a warp of pins, then one warp per larger hyperedge.
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
        int start = hyperedge_offsets[tid];
        int end = hyperedge_offsets[tid + 1];
        int hedge_size = end - start;

        if (hedge_size <= warpSize) {
            unsigned long long flags_all = 0ULL;
            unsigned long long flags_double = 0ULL;

            for (int k = start; k < end; k++) {
                int node = hyperedge_nodes[k];
                if ((unsigned)node < (unsigned)num_nodes) {
                    int part = partition[node];
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

    for (int hedge = global_warp; hedge < num_hyperedges; hedge += total_warps) {
        int start = 0;
        int end = 0;
        int hedge_size = 0;

        if (lane == 0) {
            start = hyperedge_offsets[hedge];
            end = hyperedge_offsets[hedge + 1];
            hedge_size = end - start;
        }

        start = __shfl_sync(0xffffffffu, start, 0);
        end = __shfl_sync(0xffffffffu, end, 0);
        hedge_size = __shfl_sync(0xffffffffu, hedge_size, 0);

        if (hedge_size <= warpSize) {
            continue;
        }

        unsigned long long local_all = 0ULL;
        unsigned long long local_double = 0ULL;

        for (int k = start + lane; k < end; k += warpSize) {
            int node = hyperedge_nodes[k];
            if ((unsigned)node < (unsigned)num_nodes) {
                int part = partition[node];
                if ((unsigned)part < 64u) {
                    unsigned long long bit = 1ULL << part;
                    local_double |= (local_all & bit);
                    local_all |= bit;
                }
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            unsigned long long other_all = __shfl_xor_sync(0xffffffffu, local_all, offset);
            unsigned long long other_double = __shfl_xor_sync(0xffffffffu, local_double, offset);
            local_double |= other_double | (local_all & other_all);
            local_all |= other_all;
        }

        if (lane == 0) {
            edge_flags_all[hedge] = local_all;
            edge_flags_double[hedge] = local_double;
        }
    }
}

// Thread capacity of the balance kernel's block-wide reduction.
constexpr int BALANCE_THREADS = 1024;

// Argmin key of the balance kernel: 22 bits of (2^21 - gain), 20 bits of degree, 22 bits of index.
__device__ __forceinline__ unsigned long long balance_key(int gain, int deg, int idx) {
    unsigned long long gq = (unsigned long long)(2097152 - gain);
    return (gq << 42) | ((unsigned long long)deg << 22) | (unsigned long long)idx;
}

// Total order on candidate moves: gain, then pins already doubled in the target, then smaller
// target block size, then a per-node hash, then lower block index.
__device__ __forceinline__ bool move_better(
    int g, int t, int gd, int gb, int tb, int gbd,
    const int *snip, int node)
{
    if (tb < 0)  return true;
    if (g > gb)  return true;
    if (g < gb)  return false;
    if (gd > gbd) return true;
    if (gd < gbd) return false;
    int ts = snip[t], bs = snip[tb];
    if (ts < bs) return true;
    if (ts > bs) return false;
    int ht = (t * 17 + node) & 63, hb = (tb * 17 + node) & 63;
    if (ht < hb) return true;
    if (ht > hb) return false;
    return t < tb;
}

// Block-parallel repair: fills blocks below min_part_size then drains blocks above
// max_part_size, one best-gain move at a time (block-wide argmin, deterministic ties).
extern "C" __global__ void balance_final_par_10k(
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
    __shared__ unsigned long long s_key[BALANCE_THREADS];
    __shared__ int s_nip[64];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const unsigned long long NONE = 0xFFFFFFFFFFFFFFFFULL;

    for (int i = tid; i < BALANCE_THREADS; i += nt) s_key[i] = NONE;
    for (int p = tid; p < num_parts && p < 64; p += nt) s_nip[p] = nodes_in_part[p];
    __syncthreads();

    for (int part = 0; part < num_parts; part++) {
        while (s_nip[part] < min_part_size) {
            unsigned long long best = NONE;
            unsigned long long tgt_bit = 1ULL << part;
            for (int node = tid; node < num_nodes; node += nt) {
                int p = partition[node];
                if (p == part || s_nip[p] <= min_part_size) continue;
                int start = node_offsets[node];
                int end = node_offsets[node + 1];
                int gain = 0;
                unsigned long long p_bit = 1ULL << p;
                for (int j = start; j < end; j++) {
                    int hedge = node_hyperedges[j];
                    unsigned long long flags = edge_flags_all[hedge];
                    unsigned long long dbl = edge_flags_double[hedge];
                    gain += (((dbl & p_bit) == 0) ? 1 : 0) - (((flags & tgt_bit) != 0) ? 0 : 1);
                }
                unsigned long long k = balance_key(gain, end - start, node);
                if (k < best) best = k;
            }
            s_key[tid] = best;
            __syncthreads();
            for (int s = BALANCE_THREADS >> 1; s > 0; s >>= 1) {
                if (tid < s && s_key[tid + s] < s_key[tid]) s_key[tid] = s_key[tid + s];
                __syncthreads();
            }
            unsigned long long win = s_key[0];
            __syncthreads();
            s_key[tid] = NONE;
            if (win == NONE) break;
            if (tid == 0) {
                int node = (int)(win & 0x3FFFFFULL);
                int p = partition[node];
                partition[node] = part;
                s_nip[p]--;
                s_nip[part]++;
            }
            __syncthreads();
        }
    }

    unsigned short cnt[64];
    int np = (num_parts < 64) ? num_parts : 64;
    for (int p = 0; p < np; p++) cnt[p] = 0;

    for (int part = 0; part < num_parts; part++) {
        unsigned long long p_bit = 1ULL << part;
        while (s_nip[part] > max_part_size) {
            unsigned long long best = NONE;
            for (int node = tid; node < num_nodes; node += nt) {
                if (partition[node] != part) continue;
                int start = node_offsets[node];
                int end = node_offsets[node + 1];
                int deg = end - start;
                int alone = 0;
                unsigned long long touched = 0ULL;
                for (int j = start; j < end; j++) {
                    int hedge = node_hyperedges[j];
                    unsigned long long flags = edge_flags_all[hedge];
                    unsigned long long dbl = edge_flags_double[hedge];
                    alone += ((dbl & p_bit) == 0) ? 1 : 0;
                    unsigned long long f = flags;
                    touched |= f;
                    while (f) { int b = __ffsll(f) - 1; f &= (f - 1ULL); cnt[b]++; }
                }
                int base = alone - deg;
                for (int tgt = 0; tgt < num_parts; tgt++) {
                    if (tgt == part || s_nip[tgt] >= max_part_size) continue;
                    unsigned long long k = balance_key(base + (int)cnt[tgt], deg, node * 64 + tgt);
                    if (k < best) best = k;
                }
                while (touched) { int b = __ffsll(touched) - 1; touched &= (touched - 1ULL); cnt[b] = 0; }
            }
            s_key[tid] = best;
            __syncthreads();
            for (int s = BALANCE_THREADS >> 1; s > 0; s >>= 1) {
                if (tid < s && s_key[tid + s] < s_key[tid]) s_key[tid] = s_key[tid + s];
                __syncthreads();
            }
            unsigned long long win = s_key[0];
            __syncthreads();
            s_key[tid] = NONE;
            if (win == NONE) break;
            if (tid == 0) {
                int idx = (int)(win & 0x3FFFFFULL);
                partition[idx >> 6] = idx & 63;
                s_nip[part]--;
                s_nip[idx & 63]++;
            }
            __syncthreads();
        }
    }

    __syncthreads();
    for (int p = tid; p < num_parts && p < 64; p += nt) nodes_in_part[p] = s_nip[p];
}

// Nodes of degree <= hub_degree, eight lanes per node.
// Key: ((gain + 1000) << 16) | (degree << 8) | target; 0 means no move.
__device__ __forceinline__ void moves_main(
    const int block,
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
    const int hub_degree
) {
    __shared__ int shared_nodes_in_part[64];

    if (threadIdx.x < 64 && threadIdx.x < num_parts) {
        shared_nodes_in_part[threadIdx.x] = nodes_in_part[threadIdx.x];
    }
    __syncthreads();

    const int gid  = block * blockDim.x + threadIdx.x;
    const int node = gid >> 3;
    const int lane8 = gid & 7;

    const unsigned lw   = threadIdx.x & 31;
    const unsigned grp  = 0xFFu << (lw & ~7);
    const unsigned base = lw & ~7;

    if (node < num_nodes) {
        int start = node_offsets[node];
        int end   = node_offsets[node + 1];
        int node_degree = end - start;
        if (node_degree > hub_degree) return;
        if (lane8 == 0) move_priorities[node] = 0;

        int current_part = partition[node];
        if ((unsigned)current_part < (unsigned)num_parts
            && shared_nodes_in_part[current_part] > 1) {

            if (node_degree > 0) {
                int degree_weight = node_degree > 255 ? 255 : node_degree;
                const unsigned long long current_bit = 1ULL << current_part;
                const unsigned long long lane_slice  = 0xFFULL << (lane8 * 8);

                unsigned short c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
                unsigned short d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                // The eight lanes of a node load one hyperedge each and exchange them.
                for (int j0 = 0; j0 < node_degree; j0 += 8) {
                    const int jj = j0 + lane8;
                    unsigned long long my_all = 0ULL, my_dbl = 0ULL;
                    if (jj < node_degree) {
                        const int hyperedge = __ldg(node_hyperedges + start + jj);
                        my_all = edge_flags_all[hyperedge];
                        my_dbl = edge_flags_double[hyperedge];
                    }
                  for (int t = 0; t < 8 && j0 + t < node_degree; t++) {
                    unsigned long long flags_all    = __shfl_sync(grp, my_all, base + t);
                    unsigned long long flags_double = __shfl_sync(grp, my_dbl, base + t);
                    unsigned long long mask =
                        (flags_all & ~current_bit) | (flags_double & current_bit);

                    if (lane8 == 0 && (mask & current_bit)) count_current_present++;

                    unsigned long long f_all = mask & ~current_bit & lane_slice;
                    unsigned long long f_dbl = flags_double & ~current_bit & lane_slice;

                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        cand_mask |= 1ULL << bit;
                        switch (bit & 7) {
                            case 0: c0++; break; case 1: c1++; break;
                            case 2: c2++; break; case 3: c3++; break;
                            case 4: c4++; break; case 5: c5++; break;
                            case 6: c6++; break; default: c7++; break;
                        }
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        switch (bit & 7) {
                            case 0: d0++; break; case 1: d1++; break;
                            case 2: d2++; break; case 3: d3++; break;
                            case 4: d4++; break; case 5: d5++; break;
                            case 6: d6++; break; default: d7++; break;
                        }
                    }

                  }
                }

                count_current_present = __shfl_sync(grp, count_current_present, base);

                int best_gain = -999999, best_target = -1, best_double = 0;
                unsigned long long tmp = cand_mask;
                while (tmp) {
                    int target_part = __ffsll(tmp) - 1;
                    tmp &= (tmp - 1);
                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count, p_double;
                    switch (target_part & 7) {
                        case 0: p_count=c0; p_double=d0; break;
                        case 1: p_count=c1; p_double=d1; break;
                        case 2: p_count=c2; p_double=d2; break;
                        case 3: p_count=c3; p_double=d3; break;
                        case 4: p_count=c4; p_double=d4; break;
                        case 5: p_count=c5; p_double=d5; break;
                        case 6: p_count=c6; p_double=d6; break;
                        default: p_count=c7; p_double=d7; break;
                    }

                    int basic_gain = p_count - count_current_present;
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size  = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    if (move_better(total_gain, target_part, p_double,
                                     best_gain, best_target, best_double,
                                     shared_nodes_in_part, node)) {
                        best_gain = total_gain; best_target = target_part; best_double = p_double;
                    }
                }

                for (int off = 4; off > 0; off >>= 1) {
                    int og = __shfl_down_sync(grp, best_gain,   off);
                    int ot = __shfl_down_sync(grp, best_target, off);
                    int od = __shfl_down_sync(grp, best_double, off);
                    if ((lane8 < off) && ot >= 0 &&
                        move_better(og, ot, od, best_gain, best_target, best_double,
                                     shared_nodes_in_part, node)) {
                        best_gain = og; best_target = ot; best_double = od;
                    }
                }

                if (lane8 == 0 && best_gain >= -1 && best_target >= 0
                    && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] =
                        (bg << 16) | (degree_weight << 8)
                        | ((best_target & 63) | ((node & 3) << 6));
                }
            }
        }
    }
}

// Nodes of degree > hub_degree, one warp per node over at most 768 sampled hyperedges.
__device__ __forceinline__ void moves_hub(
    const int block,
    const int num_hubs,
    const int *hub_nodes,
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

    const int hub  = (block * blockDim.x + threadIdx.x) >> 5;
    const unsigned lw = threadIdx.x & 31;
    const int lane8 = lw & 7;
    const int sub  = lw >> 3;
    const unsigned grp  = 0xFFu;

    if (hub < num_hubs) {
        const int node = hub_nodes[hub];
        if (lw == 0) move_priorities[node] = 0;

        int current_part = partition[node];
        if ((unsigned)current_part < (unsigned)num_parts
            && shared_nodes_in_part[current_part] > 1) {

            int start = node_offsets[node];
            int end   = node_offsets[node + 1];
            int node_degree = end - start;
            int used_degree = node_degree > 768 ? 768 : node_degree;

            if (used_degree > 0) {
                const int sample_step = node_degree / used_degree;
                const int sample_rem  = node_degree - sample_step * used_degree;

                int degree_weight = node_degree > 255 ? 255 : node_degree;
                const unsigned long long current_bit = 1ULL << current_part;
                const unsigned long long lane_slice  = 0xFFULL << (lane8 * 8);

                unsigned short c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
                unsigned short d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
                unsigned long long cand_mask = 0ULL;
                int count_current_present = 0;

                // Four groups of eight lanes split the hyperedges; each group loads eight at a time.
                const unsigned grp8 = 0xFFu << (sub * 8);
                for (int j0 = sub * 8; j0 < used_degree; j0 += 32) {
                    const int jj = j0 + lane8;
                    unsigned long long my_all = 0ULL, my_dbl = 0ULL;
                    if (jj < used_degree) {
                        const int rel = jj * sample_step + (jj * sample_rem) / used_degree;
                        const int hyperedge = __ldg(node_hyperedges + start + rel);
                        my_all = edge_flags_all[hyperedge];
                        my_dbl = edge_flags_double[hyperedge];
                    }
                  for (int t = 0; t < 8 && j0 + t < used_degree; t++) {
                    unsigned long long flags_all    = __shfl_sync(grp8, my_all, sub * 8 + t);
                    unsigned long long flags_double = __shfl_sync(grp8, my_dbl, sub * 8 + t);
                    unsigned long long mask =
                        (flags_all & ~current_bit) | (flags_double & current_bit);

                    if (lane8 == 0 && (mask & current_bit)) count_current_present++;

                    unsigned long long f_all = mask & ~current_bit & lane_slice;
                    unsigned long long f_dbl = flags_double & ~current_bit & lane_slice;

                    while (f_all) {
                        int bit = __ffsll(f_all) - 1;
                        f_all &= (f_all - 1);
                        cand_mask |= 1ULL << bit;
                        switch (bit & 7) {
                            case 0: c0++; break; case 1: c1++; break;
                            case 2: c2++; break; case 3: c3++; break;
                            case 4: c4++; break; case 5: c5++; break;
                            case 6: c6++; break; default: c7++; break;
                        }
                    }
                    while (f_dbl) {
                        int bit = __ffsll(f_dbl) - 1;
                        f_dbl &= (f_dbl - 1);
                        switch (bit & 7) {
                            case 0: d0++; break; case 1: d1++; break;
                            case 2: d2++; break; case 3: d3++; break;
                            case 4: d4++; break; case 5: d5++; break;
                            case 6: d6++; break; default: d7++; break;
                        }
                    }

                  }
                }

                for (int o = 8; o <= 16; o <<= 1) {
                    c0 += __shfl_xor_sync(0xffffffffu, c0, o); c1 += __shfl_xor_sync(0xffffffffu, c1, o);
                    c2 += __shfl_xor_sync(0xffffffffu, c2, o); c3 += __shfl_xor_sync(0xffffffffu, c3, o);
                    c4 += __shfl_xor_sync(0xffffffffu, c4, o); c5 += __shfl_xor_sync(0xffffffffu, c5, o);
                    c6 += __shfl_xor_sync(0xffffffffu, c6, o); c7 += __shfl_xor_sync(0xffffffffu, c7, o);
                    d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o);
                    d2 += __shfl_xor_sync(0xffffffffu, d2, o); d3 += __shfl_xor_sync(0xffffffffu, d3, o);
                    d4 += __shfl_xor_sync(0xffffffffu, d4, o); d5 += __shfl_xor_sync(0xffffffffu, d5, o);
                    d6 += __shfl_xor_sync(0xffffffffu, d6, o); d7 += __shfl_xor_sync(0xffffffffu, d7, o);
                    cand_mask |= __shfl_xor_sync(0xffffffffu, cand_mask, o);
                    count_current_present += __shfl_xor_sync(0xffffffffu, count_current_present, o);
                }
                count_current_present = __shfl_sync(0xffffffffu, count_current_present, 0);
                if (sub != 0) return;

                int best_gain = -999999, best_target = -1, best_double = 0;
                unsigned long long tmp = cand_mask;
                while (tmp) {
                    int target_part = __ffsll(tmp) - 1;
                    tmp &= (tmp - 1);
                    if ((unsigned)target_part >= (unsigned)num_parts) continue;
                    if (shared_nodes_in_part[target_part] >= max_part_size) continue;

                    int p_count, p_double;
                    switch (target_part & 7) {
                        case 0: p_count=c0; p_double=d0; break;
                        case 1: p_count=c1; p_double=d1; break;
                        case 2: p_count=c2; p_double=d2; break;
                        case 3: p_count=c3; p_double=d3; break;
                        case 4: p_count=c4; p_double=d4; break;
                        case 5: p_count=c5; p_double=d5; break;
                        case 6: p_count=c6; p_double=d6; break;
                        default: p_count=c7; p_double=d7; break;
                    }

                    int basic_gain = p_count - count_current_present;
                    if (used_degree < node_degree) {
                        basic_gain = (basic_gain * node_degree) / used_degree;
                    }
                    int current_size = shared_nodes_in_part[current_part];
                    int target_size  = shared_nodes_in_part[target_part];
                    int balance_bonus = (current_size > target_size + 1) ? 4 : 0;
                    int total_gain = basic_gain + balance_bonus;

                    if (move_better(total_gain, target_part, p_double,
                                     best_gain, best_target, best_double,
                                     shared_nodes_in_part, node)) {
                        best_gain = total_gain; best_target = target_part; best_double = p_double;
                    }
                }

                for (int off = 4; off > 0; off >>= 1) {
                    int og = __shfl_down_sync(grp, best_gain,   off);
                    int ot = __shfl_down_sync(grp, best_target, off);
                    int od = __shfl_down_sync(grp, best_double, off);
                    if ((lane8 < off) && ot >= 0 &&
                        move_better(og, ot, od, best_gain, best_target, best_double,
                                     shared_nodes_in_part, node)) {
                        best_gain = og; best_target = ot; best_double = od;
                    }
                }

                if (lane8 == 0 && best_gain >= -1 && best_target >= 0
                    && best_target != current_part) {
                    int bg = best_gain + 1000;
                    if (bg > 32767) bg = 32767;
                    if (bg < 0) bg = 0;
                    move_priorities[node] =
                        (bg << 16) | (degree_weight << 8)
                        | ((best_target & 63) | ((node & 3) << 6));
                }
            }
        }
    }
}

// The first main_blocks blocks cover the nodes, the remaining blocks the hub list.
extern "C" __global__ void compute_refinement_moves_10k(
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
    const int hub_degree,
    const int main_blocks,
    const int num_hubs,
    const int *hub_nodes
) {
    if ((int)blockIdx.x < main_blocks) {
        moves_main(blockIdx.x, num_nodes, num_parts, max_part_size, node_hyperedges, node_offsets,
                       partition, nodes_in_part, edge_flags_all, edge_flags_double, move_priorities, hub_degree);
    } else {
        moves_hub(blockIdx.x - main_blocks, num_hubs, hub_nodes, num_nodes, num_parts, max_part_size,
                      node_hyperedges, node_offsets, partition, nodes_in_part, edge_flags_all, edge_flags_double,
                      move_priorities);
    }
}

} // namespace track10k
