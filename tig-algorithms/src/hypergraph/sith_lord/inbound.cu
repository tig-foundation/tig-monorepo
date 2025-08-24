/*!
Copyright 2025 Rootz

Identity of Submitter Rootz

UAI null

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/
#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void hyperedge_clustering(
    const int num_hyperedges,
    const int num_clusters,
    const int * __restrict__ hyperedge_offsets,
    int * __restrict__ hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;

    if (hedge < num_hyperedges) {
        int start = hyperedge_offsets[hedge];
        int end = hyperedge_offsets[hedge + 1];
        int hedge_size = end - start;

        int quarter_clusters = max(1, num_clusters >> 2);
        int cluster_mask = quarter_clusters - 1;

        int cluster;
        if (hedge_size <= 2) {
            cluster = hedge & cluster_mask;
        } else if (hedge_size <= 4) {
            cluster = quarter_clusters + (hedge & cluster_mask);
        } else if (hedge_size <= 8) {
            cluster = (quarter_clusters << 1) + (hedge & cluster_mask);
        } else {
            cluster = (quarter_clusters * 3) + (hedge & cluster_mask);
        }

        hyperedge_clusters[hedge] = cluster;
    }
}

extern "C" __global__ void compute_node_preferences(
    const int num_nodes,
    const int num_parts,
    const int num_hedge_clusters,
    const int * __restrict__ node_hyperedges,
    const int * __restrict__ node_offsets,
    const int * __restrict__ hyperedge_clusters,
    const int * __restrict__ hyperedge_offsets,
    int * __restrict__ pref_parts,
    int * __restrict__ pref_priorities
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < num_nodes) {
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int node_degree = end - start;

        unsigned short cluster_votes[256];
        int max_clusters = min(num_hedge_clusters, 256);
        for (int i = 0; i < max_clusters; i++) {
            cluster_votes[i] = 0;
        }

        int max_votes = 0;
        int best_cluster = 0;

        for (int j = start; j < end; j++) {
            int hyperedge = node_hyperedges[j];
            int cluster = hyperedge_clusters[hyperedge];

            if (cluster >= 0 && cluster < max_clusters) {
                int hedge_start = hyperedge_offsets[hyperedge];
                int hedge_end = hyperedge_offsets[hyperedge + 1];
                int hedge_size = hedge_end - hedge_start;
                int weight = (hedge_size <= 3) ? 3 : (hedge_size <= 6) ? 2 : 1;

                cluster_votes[cluster] += weight;

                if (cluster_votes[cluster] > max_votes ||
                    (cluster_votes[cluster] == max_votes && cluster < best_cluster)) {
                    max_votes = cluster_votes[cluster];
                    best_cluster = cluster;
                }
            }
        }

        int target_partition;
        if (node_degree <= 3) {
            target_partition = (best_cluster + node) % num_parts;
        } else if (node_degree <= 8) {
            target_partition = (best_cluster + node_degree + node) % num_parts;
        } else {
            target_partition = (best_cluster * 2 + node_degree + node) % num_parts;
        }

        pref_parts[node] = target_partition;
        pref_priorities[node] = (max_votes << 16) + (num_parts - (node % num_parts));
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
                    int fallback_part = node % num_parts;
                    partition[node] = fallback_part;
                    nodes_in_part[fallback_part]++;
                }
            }
        }
    }
}

extern "C" __global__ void compute_refinement_moves_batched(
    const int batch_start,
    const int batch_size,
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int num_hyperedges,
    const int * __restrict__ node_hyperedges,
    const int * __restrict__ node_offsets,
    const int * __restrict__ hyperedge_nodes,
    const int * __restrict__ hyperedge_offsets,
    const int * __restrict__ partition,
    const int * __restrict__ nodes_in_part,
    int * __restrict__ move_parts,
    int * __restrict__ move_gains,
    int * __restrict__ move_priorities,
    int * __restrict__ num_valid_moves,
    const int round
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int node = batch_start + idx;
    if (idx >= batch_size || node >= num_nodes) return;

    move_parts[node] = partition[node];
    move_gains[node] = 0;
    move_priorities[node] = 0;

    int current_part = partition[node];
    if (current_part < 0 || current_part >= num_parts || nodes_in_part[current_part] <= 1) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];
    int node_degree = end - start;
    if (node_degree <= 0) return;

    const int MAX_FLAGS = 64;
    int eff_degree = (node_degree < MAX_FLAGS) ? node_degree : MAX_FLAGS;
    unsigned long long edge_flags[MAX_FLAGS];
    int max_p = min(num_parts, 64);

    #pragma unroll
    for (int j = 0; j < eff_degree; j++) {
        edge_flags[j] = 0ULL;
        int hyperedge = node_hyperedges[start + j];
        int hedge_start = hyperedge_offsets[hyperedge];
        int hedge_end = hyperedge_offsets[hyperedge + 1];

        #pragma unroll 4
        for (int k = hedge_start; k < hedge_end; k++) {
            int other_node = hyperedge_nodes[k];
            if (other_node != node && other_node >= 0 && other_node < num_nodes) {
                int part = partition[other_node];
                if (part >= 0 && part < max_p) {
                    edge_flags[j] |= 1ULL << (part & 63);
                }
            }
        }
    }

    int cp = current_part & 63;
    int original_cost = 0;
    #pragma unroll 4
    for (int j = 0; j < eff_degree; j++) {
        int lambda = __popcll(edge_flags[j] | (1ULL << cp));
        if (lambda > 1) original_cost += (lambda - 1);
    }

    int best_gain = 0;
    int best_target = current_part;
    int current_size = nodes_in_part[current_part];
    int balance_threshold = (num_hyperedges < 50000) ? 2 : 4;

    for (int offset = 0; offset < num_parts; offset++) {
        int target_part = (node + round + offset) % num_parts;
        if (target_part == current_part) continue;

        int target_size = nodes_in_part[target_part];
        if (target_size >= max_part_size) continue;

        int tp = target_part & 63;
        int new_cost = 0;
        #pragma unroll 8
        for (int j = 0; j < eff_degree; j++) {
            int lambda = __popcll(edge_flags[j] | (1ULL << tp));
            if (lambda > 1) new_cost += (lambda - 1);
        }

        int basic_gain = original_cost - new_cost;
        int balance_bonus = (current_size > target_size + 2) ? balance_threshold : 0;
        int total_gain = basic_gain + balance_bonus;

        if (total_gain > best_gain || (total_gain == best_gain && target_part < best_target)) {
            best_gain = total_gain;
            best_target = target_part;
        }
    }

    if (best_gain > 0 && best_target != current_part) {
        move_parts[node] = best_target;
        move_gains[node] = best_gain;
        move_priorities[node] = (best_gain << 16) + (num_parts - (node % num_parts));
        atomicAdd(num_valid_moves, 1);
    }
}

extern "C" __global__ void execute_refinement_moves(
    const int num_valid_moves,
    const int * __restrict__ sorted_nodes,
    const int * __restrict__ sorted_parts,
    const int max_part_size,
    int * __restrict__ partition,
    int * __restrict__ nodes_in_part,
    int * __restrict__ moves_executed
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
