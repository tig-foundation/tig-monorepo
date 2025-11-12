#include <curand_kernel.h>
#include <stdint.h>
#include <cuda_runtime.h>


__device__ int select_level_based_on_weights(
    const int num_levels, 
    const float* level_weights, 
    curandState* state
)
{
    float total_weight = 0.0f;
    for (int idx = 0; idx < num_levels; idx++)
    {
        total_weight += level_weights[idx];
    }

    float random_value = curand_uniform(state) * total_weight;
    float cumulative = 0.0f;

    for (int idx = 0; idx < num_levels; idx++)
    {
        cumulative += level_weights[idx];
        if (random_value <= cumulative)
        {
            return idx;
        }
    }

    return num_levels - 1;
}

__device__ void select_group(
    const int level, 
    const int num_nodes,
    int *group,
    int *num_groups,
    curandState* state
)
{
    *num_groups = 1 << level;  // 2^level
    *group = curand(state) % *num_groups;
}

__device__ void get_group_bounds(
    const int num_nodes, 
    const int num_groups, 
    const int group, 
    int *start_idx, 
    int *end_idx
)
{
    int s = num_nodes / num_groups;
    int r = num_nodes % num_groups;
    if (group < r)
    {
        *start_idx = (s + 1) * group;
        *end_idx = (s + 1) * (group + 1);
    }
    else
    {
        *start_idx = (s + 1) * r + s * (group - r);
        *end_idx = (s + 1) * r + s * (group + 1 - r);
    }
}

extern "C" __global__ void generate_hyperedge_sizes(
    const uint8_t *seed,
    const int num_hyperedges,
    const uint min_hyperedge_size,
    const uint max_hyperedge_size,
    const float alpha,
    int *hyperedge_sizes
)
{
    for (int hyperedge_idx = threadIdx.x + blockIdx.x * blockDim.x; hyperedge_idx < num_hyperedges; hyperedge_idx += blockDim.x * gridDim.x) 
    {   
        curandState state;
        curand_init(((uint64_t *)(seed))[0], hyperedge_idx, 0, &state);
        
        float y = curand_uniform(&state);

        float c1 = powf((float)min_hyperedge_size, alpha);
        float c2 = powf((float)max_hyperedge_size, alpha) - c1;
        float x = powf(c2 * y + c1, 1.0f / alpha);

        uint sample = (uint)floorf(x);
        if (sample < min_hyperedge_size)
        {
            hyperedge_sizes[hyperedge_idx] = min_hyperedge_size;
        }
        else if (sample > max_hyperedge_size)
        {
            hyperedge_sizes[hyperedge_idx] = max_hyperedge_size;
        }
        else
        {
            hyperedge_sizes[hyperedge_idx] = sample;
        }
    }
}

extern "C" __global__ void generate_node_weights(
    const uint8_t *seed,
    const int num_nodes,
    const float min_node_weight,
    const float max_node_weight,
    const float alpha,
    float *node_weights
)
{    
    for (int node_idx = threadIdx.x + blockIdx.x * blockDim.x; node_idx < num_nodes; node_idx += blockDim.x * gridDim.x) 
    {
        curandState state;
        curand_init(((uint64_t *)(seed))[1], node_idx, 0, &state);
     
        float y = curand_uniform(&state);

        float c1 = powf(min_node_weight, alpha);
        float c2 = powf(max_node_weight, alpha) - c1;
        float x = powf(c2 * y + c1, 1.0f / alpha);

        float sample = floorf(x);
        if (sample < min_node_weight)
        {
            node_weights[node_idx] = min_node_weight;
        }
        else if (sample > max_node_weight)
        {
            node_weights[node_idx] = max_node_weight;
        }
        else
        {
            node_weights[node_idx] = sample;
        }
    }
}

extern "C" __global__ void finalize_hyperedge_sizes(
    const int num_hyperedges,
    const int *hyperedge_sizes,
    int *hyperedge_offsets,
    uint *total_connections
)
{
    hyperedge_offsets[0] = 0;
    for (int idx = 0; idx < num_hyperedges; idx++)
    {
        hyperedge_offsets[idx+1] = hyperedge_offsets[idx] + hyperedge_sizes[idx];
    }
    *total_connections = hyperedge_offsets[num_hyperedges];
}

typedef struct
{
    int node_idx;
    float key;
} TrackedNode;

__device__ void swap_nodes(TrackedNode *a, TrackedNode *b)
{
    TrackedNode temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void swap_nodes(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void heapify(TrackedNode *arr, const int idx, const int size)
{
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
            
    if (left < size && arr[left].key < arr[smallest].key)
    {
        smallest = left;
    }
            
    if (right < size && arr[right].key < arr[smallest].key)
    {
        smallest = right;
    }
            
    if (smallest != idx)
    {
        swap_nodes(&arr[idx], &arr[smallest]);
        heapify(arr, smallest, size);
    }
}

__device__ void heapify(int *arr, const int idx, const int size)
{
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
            
    if (left < size && arr[left] > arr[largest])
    {
        largest = left;
    }
            
    if (right < size && arr[right] > arr[largest])
    {
        largest = right;
    }
            
    if (largest != idx)
    {
        swap_nodes(&arr[idx], &arr[largest]);
        heapify(arr, largest, size);
    }
}

__device__ void build_min_heap(TrackedNode *arr, const int size)
{
    for (int idx = size / 2 - 1; idx >= 0; idx--)
    {
        heapify(arr, idx, size);
    }
}

__device__ int binary_search(
    const int *arr,
    const int size,
    const int target
)
{
    int left = 0;
    int right = size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

extern "C" __global__ void generate_hyperedges(
    const uint8_t *seed,
    const int num_nodes,
    const int num_hyperedges,
    const uint total_connections,
    const int *hyperedge_sizes,
    const int *hyperedge_offsets,
    const float *node_weights,
    const float *level_weights,
    int *hyperedge_nodes,
    int *node_degrees
)
{
    for (int hyperedge_idx = threadIdx.x + blockIdx.x * blockDim.x; hyperedge_idx < num_hyperedges; hyperedge_idx += blockDim.x * gridDim.x) 
    {
        curandState state;
        curand_init(((uint64_t *)(seed))[2], hyperedge_idx, 0, &state);

        int hyperhyperedge_size = hyperedge_sizes[hyperedge_idx];
        int hyperedge_offset = hyperedge_offsets[hyperedge_idx];

        int num_levels = (int)log2f((float)num_nodes / hyperhyperedge_size) + 1;
        int level = select_level_based_on_weights(num_levels, level_weights, &state);
        
        int group, num_groups;
        select_group(level, num_nodes, &group, &num_groups, &state);
        
        int start_idx, end_idx;
        get_group_bounds(num_nodes, num_groups, group, &start_idx, &end_idx);
        
        TrackedNode reservoir[2000];
        int group_size = end_idx - start_idx;

        if (hyperhyperedge_size < 16)
        {
            for (int idx = 0; idx < group_size; idx++)
            {
                int node_idx = start_idx + idx;
                float weight = node_weights[node_idx];
                float key = powf(curand_uniform(&state), 1.0f / weight);
                
                if (idx < hyperhyperedge_size)
                {
                    reservoir[idx].node_idx = node_idx;
                    reservoir[idx].key = key;
                }
                else
                {
                    int min_idx = 0;
                    float min_key = reservoir[0].key;
                    
                    for (int j = 1; j < hyperhyperedge_size; j++)
                    {
                        if (reservoir[j].key < min_key)
                        {
                            min_key = reservoir[j].key;
                            min_idx = j;
                        }
                    }
                    
                    if (key > min_key)
                    {
                        reservoir[min_idx].node_idx = node_idx;
                        reservoir[min_idx].key = key;
                    }
                }
            }
        }
        else
        {
            for (int idx = 0; idx < hyperhyperedge_size; idx++)
            {
                int node_idx = start_idx + idx;
                float weight = node_weights[node_idx];
                float key = powf(curand_uniform(&state), 1.0f / weight);
                    
                reservoir[idx].node_idx = node_idx;
                reservoir[idx].key = key;
            }
            
            build_min_heap(reservoir, hyperhyperedge_size);
            for (int idx = hyperhyperedge_size; idx < group_size; idx++)
            {
                int node_idx = start_idx + idx;
                float weight = node_weights[node_idx];
                float key = powf(curand_uniform(&state), 1.0f / weight);
                
                if (key > reservoir[0].key)
                {
                    reservoir[0].node_idx = node_idx;
                    reservoir[0].key = key;
                    
                    heapify(reservoir, 0, hyperhyperedge_size);
                }
            }
        }

        // Sort nodes for this hyperedge
        for (int idx = 0; idx < hyperhyperedge_size; idx++)
        {
            int node_idx = reservoir[idx].node_idx;
            hyperedge_nodes[hyperedge_offset + idx] = node_idx;
            atomicAdd(&node_degrees[node_idx], 1);
        }
        for (int i = hyperhyperedge_size / 2 - 1; i >= 0; i--)
        {
            heapify(&hyperedge_nodes[hyperedge_offset], i, hyperhyperedge_size);
        }

        for (int i = hyperhyperedge_size - 1; i > 0; i--) 
        {
            swap_nodes(&hyperedge_nodes[hyperedge_offset], &hyperedge_nodes[hyperedge_offset + i]);
            heapify(&hyperedge_nodes[hyperedge_offset], 0, i);
        }
    }
}


extern "C" __global__ void finalize_hyperedges(
    const int num_nodes,
    const int num_hyperedges,
    const int *hyperedge_sizes,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int *node_degrees,
    int *node_hyperedges,
    int *node_offsets
)
{
    // Compute prefix sum for offsets
    int running_sum = 0;
    for (int v = 0; v < num_nodes; v++) 
    {
        if (threadIdx.x == 0) {
            node_offsets[v] = running_sum;
        }
        running_sum += node_degrees[v];
    }
    if (threadIdx.x == 0) {
        node_offsets[num_nodes] = running_sum;
    }
    __syncthreads();
    
    for (int node_idx = threadIdx.x + blockIdx.x * blockDim.x; node_idx < num_nodes; node_idx += blockDim.x * gridDim.x) 
    {
        int offset = 0;
        for (int hyperedge_idx = 0; hyperedge_idx < num_hyperedges; hyperedge_idx++)
        {
            int start = hyperedge_offsets[hyperedge_idx];
            if (binary_search(&hyperedge_nodes[start], hyperedge_sizes[hyperedge_idx], node_idx) != -1)
            {
                int insert_pos = node_offsets[node_idx] + offset;
                node_hyperedges[insert_pos] = hyperedge_idx;
                offset++;
            }
        }
    }
}

extern "C" __global__ void initialize_partitioning(
    const int num_nodes,
    const int *node_degrees,
    int *partition,
    int *sorted_nodes
) 
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Use level 1 as initial partition
        int start, end;
        get_group_bounds(num_nodes, 2, 1, &start, &end);
        for (int idx = 0; idx < num_nodes; idx++) 
        {
            if (node_degrees[idx] == 0) {
                partition[idx] = -1;
            } else {
                partition[idx] = idx < start ? 1 : 2;
            }
        }
    }

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

extern "C" __global__ void greedy_bipartition(
    const int level,
    const int num_nodes,
    const int num_hyperedges,
    const int *node_hyperedges,
    const int *node_offsets,
    const int *sorted_nodes,
    const int *node_degrees,
    const int *curr_partition,
    int *partition,
    unsigned long long *left_hyperedge_flags,
    unsigned long long *right_hyperedge_flags
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
        __shared__ int connections_left;
        __shared__ int connections_right;
        if (threadIdx.x == 0) {
            left_count = 0;
            right_count = 0;
        }
        __syncthreads();

        int num_flags = (num_hyperedges + 63) / 64;
        unsigned long long *left_flags = left_hyperedge_flags + blockIdx.x * num_flags;
        unsigned long long *right_flags = right_hyperedge_flags + blockIdx.x * num_flags;

        for (int idx = 0; idx < num_nodes; idx++) {
            int v = sorted_nodes[idx];
            if (curr_partition[v] != p) continue;
            
            // Get range of hyperedges for this node
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
                // Loop through this node's hyperedges
                if (threadIdx.x == 0) {
                    connections_left = 0;
                    connections_right = 0;
                }
                __syncthreads();

                for (int pos = start_pos + threadIdx.x; pos < end_pos; pos += blockDim.x) {
                    int hyperedge_idx = node_hyperedges[pos];
                    if (left_flags[hyperedge_idx / 64] & (1ULL << (hyperedge_idx % 64))) atomicAdd(&connections_left, 1);
                    if (right_flags[hyperedge_idx / 64] & (1ULL << (hyperedge_idx % 64))) atomicAdd(&connections_right, 1);
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
            unsigned long long *hyperedge_flags = assign_left ? left_flags : right_flags;
            for (int e = start_pos + threadIdx.x; e < end_pos; e += blockDim.x) {
                int hyperedge_idx = node_hyperedges[e];
                atomicOr(&hyperedge_flags[hyperedge_idx / 64], 1ULL << (hyperedge_idx % 64));
            }

            __syncthreads();
        }
    }
}

extern "C" __global__ void finalize_bipartition(
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

extern "C" __global__ void shuffle_nodes(
    const uint8_t *seed,
    const int num_nodes,
    const int *partition,
    const int *hyperedge_sizes,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int *node_degrees,
    const int *node_hyperedges,
    const int *node_offsets,
    const float *node_weights,
    const int *sorted_nodes,
    float *rand_weights,
    int *shuffled_partition,
    int *shuffled_hyperedge_nodes,
    float *shuffled_node_weights,
    int *shuffled_node_degrees,
    uint *num_prune
) {
    curandState state;
    curand_init(((uint64_t *)(seed))[3], 0, 0, &state);
    for (int idx = 0; idx < num_nodes; idx++) {
        rand_weights[idx] = curand_uniform(&state);
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (node_degrees[sorted_nodes[num_nodes - *num_prune - 1]] == 0) {
            (*num_prune)++;
        }
    }

    for (int node_idx = threadIdx.x + blockIdx.x * blockDim.x; node_idx < num_nodes; node_idx += blockDim.x * gridDim.x) {
        if (node_degrees[node_idx] == 0) {
            continue;
        }

        int pos = 0;
        for (int j = 0; j < num_nodes; j++) {
            if (node_idx == j || node_degrees[j] == 0) {
                continue;
            }
            if (
                (rand_weights[node_idx] > rand_weights[j]) || 
                (rand_weights[node_idx] == rand_weights[j] && node_idx > j)
            ) {
                pos++;
            }
        }
        shuffled_node_weights[pos] = node_weights[node_idx];
        shuffled_partition[pos] = partition[node_idx];
        shuffled_node_degrees[pos] = node_degrees[node_idx];
        for (int i = 0; i < node_degrees[node_idx]; i++) {
            int hyperedge_idx = node_hyperedges[node_offsets[node_idx] + i];
            int offset = hyperedge_offsets[hyperedge_idx];
            int pos2 = binary_search(&hyperedge_nodes[offset], hyperedge_sizes[hyperedge_idx], node_idx);
            shuffled_hyperedge_nodes[offset + pos2] = pos;
        }
    }
}

extern "C" __global__ void finalize_shuffle(
    const uint8_t *seed,
    const int num_hyperedges,
    const int *hyperedge_sizes,
    const int *hyperedge_offsets,
    int *shuffled_hyperedge_nodes
) {
    for (int hyperedge_idx = threadIdx.x + blockIdx.x * blockDim.x; hyperedge_idx < num_hyperedges; hyperedge_idx += blockDim.x * gridDim.x) {
        int hyperedge_size = hyperedge_sizes[hyperedge_idx];
        int hyperedge_offset = hyperedge_offsets[hyperedge_idx];

        for (int i = hyperedge_size / 2 - 1; i >= 0; i--)
        {
            heapify(&shuffled_hyperedge_nodes[hyperedge_offset], i, hyperedge_size);
        }

        for (int i = hyperedge_size - 1; i > 0; i--) 
        {
            swap_nodes(&shuffled_hyperedge_nodes[hyperedge_offset], &shuffled_hyperedge_nodes[hyperedge_offset + i]);
            heapify(&shuffled_hyperedge_nodes[hyperedge_offset], 0, i);
        }
    }
}

extern "C" __global__ void validate_partition(
    const int num_nodes,
    const int num_parts,
    const int *partition,
    unsigned int *errorflag
) {
    for (int node_idx = threadIdx.x; node_idx < num_nodes; node_idx += blockDim.x) {
        int part = partition[node_idx];
        
        // Validate partition (redundant but keeping for safety)
        if (part < 0 || part >= num_parts) {
            atomicOr(errorflag, 1u);
            return;
        }
    }
}

extern "C" __global__ void calc_connectivity_metric(
    const int num_hyperedges,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int *partition,
    uint *connectivity_metric
) {
    for (int hyperedge_idx = threadIdx.x + blockIdx.x * blockDim.x; hyperedge_idx < num_hyperedges; hyperedge_idx += blockDim.x * gridDim.x) {
        int start = hyperedge_offsets[hyperedge_idx];
        int end = hyperedge_offsets[hyperedge_idx + 1];
        
        // Count unique parts for this hyperedge
        uint64_t hyperedge_part_flags = 0;
        for (int pos = start; pos < end; pos++) {
            int node = hyperedge_nodes[pos];
            int part = partition[node];
            
            hyperedge_part_flags |= (1ULL << part);
        }
        
        // Add to connectivity sum
        int connectivity = __popcll(hyperedge_part_flags);
        atomicAdd(connectivity_metric, connectivity - 1);
    }
}

extern "C" __global__ void count_nodes_in_part(
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
