/*!
Copyright 2025 Uncharted Trading

Licensed under the TIG Open Data License v2.0 or (at your option) any later version
(the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

#include <curand_kernel.h>
#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ void sort_nodes_by_degrees(
    const int num_nodes,
    const int *node_degrees,
    int *sorted_nodes
) 
{
    // compare each vertex with all others to find its sort idx
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < num_nodes; i += blockDim.x * gridDim.x) {
        int pos = 0;
        for (int j = 0; j < num_nodes; j++) {
            if (i == j) {
                continue;
            }
            if (
                node_degrees[i] < node_degrees[j] || 
                (node_degrees[i] == node_degrees[j] && i > j)
            ) {
                pos++;
            }
        }
        sorted_nodes[pos] = i;
    }
}

extern "C" __global__ void my_greedy_bipartition(
    const int level,
    const int num_nodes,
    const int num_edges,
    const int *node_edges,
    const int *node_offsets,
    const int *sorted_nodes,
    const int *node_degrees,
    const int *curr_partition,
    int *partition,
    unsigned long long *left_edge_flags,
    unsigned long long *right_edge_flags
) {
    int p = (1 << level) + blockIdx.x - 1;

    __shared__ int count;
    if (threadIdx.x == 0) {
        count = 0;
    }
    __syncthreads();
    for (int v = threadIdx.x; v < num_nodes; v += blockDim.x) {
        if (curr_partition[v] == p) {
            atomicAdd(&count, 1);
        }
    }
    __syncthreads();
    
    if (count > 0) {
        int size_left = count / 2;
        int size_right = count - size_left;

        __shared__ int left_count;
        __shared__ int right_count;
        __shared__ float connections_left;
        __shared__ float connections_right;
        if (threadIdx.x == 0) {
            left_count = 0;
            right_count = 0;
        }
        __syncthreads();

        int num_flags = (num_edges + 63) / 64;
        unsigned long long *left_flags = left_edge_flags + blockIdx.x * num_flags;
        unsigned long long *right_flags = right_edge_flags + blockIdx.x * num_flags;

        for (int idx = 0; idx < num_nodes; idx++) {
            int v = sorted_nodes[idx];
            if (curr_partition[v] != p) continue;
            
            // Get range of edges for this node
            int start_pos = node_offsets[v];
            int end_pos = node_offsets[v+1];

            int left_child = p * 2 + 1;
            int right_child = p * 2 + 2;

            bool assign_left;
            if (left_count >= size_left) {
                assign_left = false;
            } else if (right_count >= size_right) {
                assign_left = true;
            } else {
                // Loop through this node's edges
                if (threadIdx.x == 0) {
                    connections_left = 0;
                    connections_right = 0;
                }
                __syncthreads();

                for (int pos = start_pos + threadIdx.x; pos < end_pos; pos += blockDim.x) {
                    int edge_idx = node_edges[pos];
                    if (left_flags[edge_idx / 64] & (1ULL << (edge_idx % 64))) atomicAdd(&connections_left, 1);
                    if (right_flags[edge_idx / 64] & (1ULL << (edge_idx % 64))) atomicAdd(&connections_right, 1);
                }
                __syncthreads();
                if (connections_left == connections_right) {
                    assign_left = left_count < right_count;
                } else {
                    assign_left = connections_left > connections_right;
                }
            }

            if (threadIdx.x == 0) {
                if (assign_left) {
                    partition[v] = left_child;
                    atomicAdd(&left_count, 1);
                } else {
                    partition[v] = right_child;
                    atomicAdd(&right_count, 1);
                }
            }
            unsigned long long *edge_flags = assign_left ? left_flags : right_flags;
            for (int e = start_pos + threadIdx.x; e < end_pos; e += blockDim.x) {
                int edge_idx = node_edges[e];
                atomicOr(&edge_flags[edge_idx / 64], 1ULL << (edge_idx % 64));
            }

            __syncthreads();
        }
    }
}

extern "C" __global__ void my_finalize_bipartition(
    const int num_nodes,
    const int num_parts,
    int *partition
) {    
    for (int v = threadIdx.x; v < num_nodes; v += blockDim.x) {
        if (partition[v] != -1) {
            partition[v] -= (num_parts - 1);
        }
    }
}

extern "C" __global__ void my_count_nodes_in_part(
    const int num_nodes,
    const int num_parts,
    const int *partition,
    int *nodes_in_part
) {
    for (int node_idx = threadIdx.x + blockIdx.x * blockDim.x; node_idx < num_nodes; node_idx += blockDim.x * gridDim.x) {
        int part = partition[node_idx];        
        atomicAdd(&nodes_in_part[part], 1);
    }
}

extern "C" __global__ void local_search(
    const int num_nodes,
    const int num_parts,
    const int max_part_size,
    const int *node_edges,
    const int *node_offsets,
    const int *edge_nodes,
    const int *edge_offsets,
    const int *partition,
    const int *nodes_in_part,
    uint64_t *edge_flags,
    int *best_part,
    int *best_diff
)
{
    for (int node = threadIdx.x + blockIdx.x * blockDim.x; node < num_nodes; node += blockDim.x * gridDim.x) {
        if (nodes_in_part[partition[node]] == 1) {
            continue;
        }
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        for (int j = start; j < end; j++) {
            int edge = node_edges[j];
            int start2 = edge_offsets[edge];
            int end2 = edge_offsets[edge + 1];
            for (int k = start2; k < end2; k++) {
                int node2 = edge_nodes[k];
                if (node2 == node) {
                    continue;
                }
                int part = partition[node2];
                edge_flags[j] |= 1ULL << part;
            }
        }

        int original_score = 0;
        for (int j = start; j < end; j++) {
            original_score += __popcll(edge_flags[j] | (1ULL << partition[node]));
        }
        best_part[node] = partition[node];

        for (int part = 0; part < num_parts; part++) {
            if (part == partition[node] || nodes_in_part[part] == max_part_size) {
                continue;
            }
            int score = 0;
            for (int j = start; j < end; j++) {
                score += __popcll(edge_flags[j] | (1ULL << part));
            }
            int diff = original_score - score;
            if (diff > best_diff[node]) {
                best_part[node] = part;
                best_diff[node] = diff;
            }
        }
    }
}

extern "C" __global__ void update_part(
    const int node,
    const int best_part,
    int *partition,
    int *nodes_in_part
) 
{
    int old_part = partition[node];
    partition[node] = best_part;
    nodes_in_part[old_part]--;
    nodes_in_part[best_part]++;
}
