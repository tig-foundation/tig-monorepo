#include <stdint.h>
#include <cuda_runtime.h>

// ============================================================================
// hyper_ada_v1: Ultra-Fast Bitmask Hypergraph Partitioning Kernels
// Target Hardware: NVIDIA Ada Lovelace (sm_89) / Pascal through Blackwell (sm_70+)
// ============================================================================

extern "C" __global__ void ada_hyperedge_clustering_10k(
    const int num_hyperedges,
    const int *hyperedge_sizes,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int num_clusters,
    const unsigned int random_seed,
    int *hyperedge_clusters
) {
    int hedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (hedge >= num_hyperedges) return;

    int size = hyperedge_sizes[hedge];
    int start = hyperedge_offsets[hedge];
    
    // Hash-based fast pseudo-community initial clustering
    unsigned int h = (unsigned int)hedge * 1664525u + random_seed;
    if (size > 0) {
        h ^= (unsigned int)hyperedge_nodes[start] * 1013904223u;
        h = (h ^ (h >> 16)) * 0x85ebca6bu;
    }
    hyperedge_clusters[hedge] = (int)(h % (unsigned int)num_clusters);
}

extern "C" __global__ void ada_compute_initial_preferences_10k(
    const int num_nodes,
    const int num_parts,
    const int num_clusters,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *hyperedge_clusters,
    const int *hyperedge_offsets,
    int *pref_parts
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int start = node_offsets[node];
    int end = node_offsets[node + 1];

    int votes[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) votes[i] = 0;

    int max_clusters = (num_clusters < 64) ? num_clusters : 64;
    for (int j = start; j < end; ++j) {
        int hedge = node_hyperedges[j];
        int cluster = hyperedge_clusters[hedge] % max_clusters;
        if (cluster >= 0) {
            int hedge_size = hyperedge_offsets[hedge + 1] - hyperedge_offsets[hedge];
            int weight = (hedge_size <= 2) ? 6 : (hedge_size <= 4) ? 4 : (hedge_size <= 8) ? 2 : 1;
            votes[cluster] += weight;
        }
    }

    int best_cluster = 0;
    int max_v = -1;
    for (int c = 0; c < max_clusters; ++c) {
        if (votes[c] > max_v) {
            max_v = votes[c];
            best_cluster = c;
        }
    }

    int part = (best_cluster * num_parts) / max_clusters;
    if (part >= num_parts) part = num_parts - 1;
    pref_parts[node] = part;
}

extern "C" __global__ void ada_compute_hyperedge_masks_10k(
    const int num_hyperedges,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int *partition,
    uint64_t *hyperedge_masks
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_hyperedges) return;

    int start = hyperedge_offsets[e];
    int end = hyperedge_offsets[e + 1];

    uint64_t mask = 0ULL;
    for (int i = start; i < end; ++i) {
        int node = hyperedge_nodes[i];
        int part = partition[node];
        if (part >= 0 && part < 64) {
            mask |= (1ULL << part);
        }
    }
    hyperedge_masks[e] = mask;
}

extern "C" __global__ void ada_find_best_moves_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_offsets,
    const int *node_hyperedges,
    const uint64_t *hyperedge_masks,
    const int *partition,
    const int *part_sizes,
    int *best_target_parts,
    int *best_gains
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int curr_part = partition[node];
    int start = node_offsets[node];
    int end = node_offsets[node + 1];

    int16_t part_touches[64];
    #pragma unroll
    for (int p = 0; p < 64; ++p) {
        part_touches[p] = 0;
    }

    for (int i = start; i < end; ++i) {
        int e = node_hyperedges[i];
        uint64_t mask = hyperedge_masks[e];

        #pragma unroll 4
        for (int p = 0; p < 64; ++p) {
            if (mask & (1ULL << p)) {
                part_touches[p]++;
            }
        }
    }

    int best_p = curr_part;
    int best_overlap = -1;

    for (int p = 0; p < num_parts; ++p) {
        if (p == curr_part) continue;
        if (part_sizes[p] >= max_part_size) continue;

        int overlap = part_touches[p];
        if (overlap > best_overlap) {
            best_overlap = overlap;
            best_p = p;
        }
    }

    int gain = best_overlap - part_touches[curr_part];
    best_target_parts[node] = best_p;
    best_gains[node] = gain;
}

extern "C" __global__ void ada_apply_moves_10k(
    const int num_nodes,
    const int max_part_size,
    const int *best_target_parts,
    const int *best_gains,
    int *partition,
    int *part_sizes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int gain = best_gains[node];
    int target_part = best_target_parts[node];
    int curr_part = partition[node];

    if (gain > 0 && target_part != curr_part) {
        int old_size = atomicAdd(&part_sizes[target_part], 1);
        if (old_size < max_part_size) {
            partition[node] = target_part;
            atomicSub(&part_sizes[curr_part], 1);
        } else {
            atomicSub(&part_sizes[target_part], 1);
        }
    }
}

extern "C" __global__ void ada_enforce_balance_10k(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    int *partition,
    int *part_sizes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int curr_p = partition[node];
    if (part_sizes[curr_p] > max_part_size) {
        // Find under-filled part
        for (int p = 0; p < num_parts; ++p) {
            if (part_sizes[p] < max_part_size) {
                int old_p = atomicAdd(&part_sizes[p], 1);
                if (old_p < max_part_size) {
                    partition[node] = p;
                    atomicSub(&part_sizes[curr_p], 1);
                    break;
                } else {
                    atomicSub(&part_sizes[p], 1);
                }
            }
        }
    }
}
