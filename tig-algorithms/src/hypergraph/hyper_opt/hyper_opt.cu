#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void hyperedge_clustering(
    const int num_hyperedges,
    const int num_clusters,
    const int *hyperedge_offsets,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        int hedge_size = end - start;
        int quarter_clusters = num_clusters >> 2;
        int cluster_mask = quarter_clusters - 1;
        int bucket = (hedge_size > 8) ? 3 :
                     (hedge_size > 4) ? 2 :
                     (hedge_size > 2) ? 1 : 0;
        hyperedge_clusters[hedge] = bucket * quarter_clusters + (hedge & cluster_mask);
    }
}

extern "C" __global__ void compute_node_preferences(
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
            int cluster = hyperedge_clusters[hyperedge];
            if (cluster >= 0 && cluster < max_clusters) {
                int hedge_start = hyperedge_offsets[hyperedge];
                int hedge_end = hyperedge_offsets[hyperedge + 1];
                int hedge_size = hedge_end - hedge_start;
                int weight = (hedge_size <= 2) ? 6 :
                             (hedge_size <= 4) ? 4 :
                             (hedge_size <= 8) ? 2 : 1;
                cluster_votes[cluster] += weight;
                if (cluster_votes[cluster] > max_votes ||
                    (cluster_votes[cluster] == max_votes && cluster < best_cluster)) {
                    max_votes = cluster_votes[cluster];
                    best_cluster = cluster;
                }
            }
        }

        int base_part = (num_parts > 0) ? (best_cluster % num_parts) : 0;
        pref_parts[node] = base_part;
        int degree_weight = node_degree > 255 ? 255 : node_degree;
        pref_priorities[node] = (max_votes << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
    }
}

extern "C" __global__ void execute_node_assignments(
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
                bool assigned = false;
                for (int attempt = 0; attempt < num_parts; attempt++) {
                    int try_part = (preferred_part + attempt) % num_parts;
                    if (nodes_in_part[try_part] < max_part_size) {
                        partition[node] = try_part;
                        nodes_in_part[try_part]++;
                        assigned = true;
                        break;
                    }
                }
                if (!assigned) {
                    partition[node] = node % num_parts;
                    nodes_in_part[node % num_parts]++;
                }
            }
        }
    }
}

extern "C" __global__ void compute_moves_v2(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int balance_weight,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *hyperedge_nodes,
    const int *hyperedge_offsets,
    const int *partition,
    const int *nodes_in_part,
    int *move_targets,
    int *move_gains,
    int *num_valid_moves
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int current_part = partition[node];
    move_targets[node] = current_part;
    move_gains[node] = 0;

    if (current_part < 0 || current_part >= num_parts) return;
    if (nodes_in_part[current_part] <= 1) return;

    int conn[64];
    for (int i = 0; i < 64; i++) conn[i] = 0;

    int edge_start = node_offsets[node];
    int edge_end = node_offsets[node + 1];

    for (int j = edge_start; j < edge_end; j++) {
        int hedge = node_hyperedges[j];
        int h_start = hyperedge_offsets[hedge];
        int h_end = hyperedge_offsets[hedge + 1];

        unsigned long long seen = 0;
        for (int k = h_start; k < h_end; k++) {
            int u = hyperedge_nodes[k];
            if (u == node) continue;
            int p = partition[u];
            if (p >= 0 && p < 64) {
                unsigned long long bit = 1ULL << p;
                if (!(seen & bit)) {
                    seen |= bit;
                    conn[p]++;
                }
            }
        }
    }

    int my_conn = conn[current_part];
    int best_gain = 0;
    int best_target = current_part;
    int current_size = nodes_in_part[current_part];

    for (int p = 0; p < num_parts; p++) {
        if (p == current_part) continue;
        if (nodes_in_part[p] >= max_part_size) continue;

        int gain = conn[p] - my_conn;
        int balance_bonus = 0;
        if (balance_weight > 0 && current_size > nodes_in_part[p] + 1)
            balance_bonus = balance_weight;
        int total_gain = gain + balance_bonus;

        if (total_gain > best_gain ||
            (total_gain == best_gain && total_gain > 0 && p < best_target)) {
            best_gain = total_gain;
            best_target = p;
        }
    }

    if (best_gain > 0 && best_target != current_part) {
        move_targets[node] = best_target;
        int degree_weight = (edge_end - edge_start);
        if (degree_weight > 255) degree_weight = 255;
        move_gains[node] = (best_gain << 16) + (degree_weight << 8) + (num_parts - (node % num_parts));
        atomicAdd(num_valid_moves, 1);
    }
}

extern "C" __global__ void execute_refinement_moves(
    const int num_valid_moves,
    const int *sorted_nodes,
    const int *sorted_parts,
    const int max_part_size,
    int *partition,
    int *nodes_in_part,
    int *moves_executed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < num_valid_moves; i++) {
            int node = sorted_nodes[i];
            int target_part = sorted_parts[i];
            if (node >= 0 && target_part >= 0) {
                int current_part = partition[node];
                if (current_part >= 0 &&
                    nodes_in_part[target_part] < max_part_size &&
                    nodes_in_part[current_part] > 1 &&
                    partition[node] == current_part) {
                    partition[node] = target_part;
                    nodes_in_part[current_part]--;
                    nodes_in_part[target_part]++;
                    (*moves_executed)++;
                }
            }
        }
    }
}

extern "C" __global__ void balance_final(
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
                for (int other = 0; other < num_parts && !moved; other++) {
                    if (other != part && nodes_in_part[other] > min_part_size) {
                        for (int node = 0; node < num_nodes; node++) {
                            if (partition[node] == other) {
                                partition[node] = part;
                                nodes_in_part[other]--;
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
                for (int other = 0; other < num_parts && !moved; other++) {
                    if (other != part && nodes_in_part[other] < max_part_size) {
                        for (int node = 0; node < num_nodes; node++) {
                            if (partition[node] == part) {
                                partition[node] = other;
                                nodes_in_part[part]--;
                                nodes_in_part[other]++;
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
