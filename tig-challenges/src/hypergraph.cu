#include <curand_kernel.h>
#include <stdint.h>
#include <cuda_runtime.h>

struct Hypergraph
{
    int num_nodes;
    int num_edges;
    int *edge_sizes;
    int *edge_offsets;
    int *edge_nodes;
    float *node_weights;
};

__device__ int select_level_based_on_weights(
    float* level_weights, 
    int num_levels, 
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

__device__ void calculate_level_weights(
    float *level_weights, 
    int num_nodes, 
    int num_levels
)
{
    #if 0
    // Hard-coded level fractions as in the Python code
    const float level_fractions[20] = {
        12.0f, 13.0f, 10.0f, 20.0f, 400.f, 1842.0f, 31335.0f, 2440.0f,
        1040.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    // Get top 6 fractions (reversed)
    float top_fractions[6];
    for (int idx = 0; idx < 6; idx++)
    {
        top_fractions[idx] = level_fractions[19 - idx];
    }

    // Get lower fractions (reversed)
    float lower_fractions[32];  // Max support for 32 levels (adjust if needed)
    float lower_sum = 0.0f;

    for (int idx = 0; idx < num_levels - 6; idx++)
    {
        if (idx < 14)  // Ensure we don't go out of bounds in level_fractions
        {
            lower_fractions[idx] = level_fractions[19 - 6 - idx];
            lower_sum += lower_fractions[idx];
        }
        else
        {
            lower_fractions[idx] = 0.0f;
        }
    }

    // Calculate pre-six total
    float pre_six_total = 0.0f;
    for (int idx = 0; idx < 14; idx++)
    {
        pre_six_total += level_fractions[idx];
    }

    // Normalize lower fractions
    if (lower_sum > 0.0f)
    {
        for (int idx = 0; idx < num_levels - 6; idx++)
        {
            lower_fractions[idx] *= pre_six_total / lower_sum;
        }
    }

    // Create padded arrays
    float padded_top[32] = {0.0f};    // Initialize all to 0
    float padded_lower[32] = {0.0f};  // Initialize all to 0

    // Fill padded arrays
    for (int idx = 0; idx < 6 && idx < num_levels; idx++)
    {
        padded_top[idx] = top_fractions[idx];
    }

    for (int idx = 6; idx < num_levels; idx++)
    {
        padded_lower[idx] = lower_fractions[idx - 6];
    }

    // Combine arrays
    float combined[32];
    float combined_sum = 0.0f;

    for (int idx = 0; idx < num_levels; idx++)
    {
        combined[idx] = padded_top[idx] + padded_lower[idx];
        combined_sum += combined[idx];
    }

    // Normalize combined array
    float normalized[32];
    for (int idx = 0; idx < num_levels; idx++)
    {
        normalized[idx] = combined[idx] / combined_sum;
    }

    // Reverse the array for final output
    for (int idx = 0; idx < num_levels; idx++)
    {
        level_weights[idx] = normalized[num_levels - 1 - idx];
    }
    #elif 1
    const int ARRAY_SIZE = 20;
    int arr[ARRAY_SIZE] = { 741, 776, 675, 752, 767, 771, 765, 736, 680, 720, 729, 688, 776, 783, 323699, 1744389, 91813, 0, 0, 0 };

    int desired_length = (int)log2f((float)num_nodes);
    if (ARRAY_SIZE > desired_length) 
    {
        int offset = ARRAY_SIZE - desired_length;
        for (int i = 0; i < desired_length; i++) 
        {
            arr[i] = arr[i + offset];
        }

        for (int i = desired_length; i < ARRAY_SIZE; i++) 
        {
            arr[i] = 0;
        }
    }

    for (int idx = 0; idx < ARRAY_SIZE; idx++)
    {
        level_weights[idx] = (float)arr[idx];
    }
    #else
    for (int idx = 0; idx < num_levels; idx++)
    {
        level_weights[idx] = 1.0f;
    }
    #endif
}

__device__ float generate_node_weight(
    curandState* state
)
{
    #if 1
    float y = curand_uniform(state);
    const float alpha = 2.457f;
    const int node_min = 1;
    const int node_max = 846;

    float alpha1 = 1.0f - alpha;
    float c1 = powf((float)node_min, alpha1);
    float c2 = powf((float)node_max, alpha1) - c1;
    float x = powf(c2 * y + c1, 1.0f / alpha1);

    int sample = (int)floorf(x);
    if (sample < node_min)
    {
        sample = node_min;
    }

    return (float)sample;
    #else
    return 1.0f;
    #endif
}

__device__ int generate_edge_size(
    curandState* state
)
{
    #if 1
    float y = curand_uniform(state);
    const float alpha = 4.353f;
    const int edge_min = 2;
    const int edge_max = 280;

    float alpha1 = 1.0f - alpha;
    float c1 = powf((float)edge_min, alpha1);
    float c2 = powf((float)edge_max, alpha1) - c1;
    float x = powf(c2 * y + c1, 1.0f / alpha1);

    int sample = (int)floorf(x);
    if (sample < edge_min)
    {
        sample = edge_min;
    }

    return sample;
    #else
    return 3;
    #endif
}

__device__ int select_level(
    curandState* state, 
    int num_nodes,  
    int hyperedge_size
)
{
    int num_levels = (int)log2f((float)num_nodes/hyperedge_size) + 1;
    return curand(state) % num_levels; 
}

__device__ int select_group(
    curandState* state, 
    int level, 
    int num_nodes
)
{
    int num_groups = 1 << level;  // 2^level
    int group = curand(state) % num_groups;
    return group;
}

__device__ void get_group_bounds(
    int group, 
    int level, 
    int num_nodes, 
    int *start_idx, 
    int *end_idx
)
{
    int num_groups = 1 << level;
    int nodes_per_group = num_nodes / num_groups;
    *start_idx = group * nodes_per_group;
    *end_idx = (group == num_groups - 1) ? num_nodes : (*start_idx + nodes_per_group);
}

//todo: decide whether to use this and enforce launch params or init new random states for each node/edge, that way any launch params will works
extern "C" __global__ void initialize_instance_kernel(
    uint8_t *seed,
    Hypergraph *hypergraph
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    curandState node_state;
    unsigned long long node_offset = (unsigned long long)idx * (hypergraph->num_nodes / total_threads + 1);
    curand_init(((uint64_t *)(seed))[0], 0, node_offset, &node_state);

    for (int node_idx = idx; node_idx < hypergraph->num_nodes; node_idx += total_threads) 
    {
        hypergraph->node_weights[node_idx] = generate_node_weight(&node_state);
    }

    curandState edge_state;
    unsigned long long edge_offset = (unsigned long long)idx * (hypergraph->num_edges / total_threads + 1);
    curand_init(((uint64_t *)(seed))[0], 1, edge_offset, &edge_state);

    for (int edge_idx = idx; edge_idx < hypergraph->num_edges; edge_idx += total_threads) 
    {
        hypergraph->edge_sizes[edge_idx] = generate_edge_size(&edge_state);
    }
}

//todoo: optimize
extern "C" __global__ void initialize_edge_offsets_kernel(
    Hypergraph *hypergraph
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        hypergraph->edge_offsets[0] = 0;
        for (int idx = 0; idx < hypergraph->num_edges; idx++)
        {
            hypergraph->edge_offsets[idx+1] = hypergraph->edge_offsets[idx] + hypergraph->edge_sizes[idx];
        }
    }
}

extern "C" __global__ void generate_edges_kernel(
    uint8_t *seed,
    Hypergraph *hypergraph
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hypergraph->num_edges)
    {
        curandState state;
        curand_init(((uint64_t *)(seed))[1], idx, 0, &state);

        int hyperedge_size = hypergraph->edge_sizes[idx];
        int edge_offset = hypergraph->edge_offsets[idx];

        #if 1 // weighted
        int num_levels = (int)log2f((float)hypergraph->num_nodes/(float)hyperedge_size) + 1;
        float level_weights[32];
        calculate_level_weights(level_weights, hypergraph->num_nodes, num_levels);
        int level = select_level_based_on_weights(level_weights, num_levels, &state);
        #elif 0 // uniform
        int level = select_level(&state, hypergraph->num_nodes, hyperedge_size);
        #elif 0 // minimum for edge
        int level = (int)log2f((float)hypergraph->num_nodes/(float)hyperedge_size);
        #elif 0 // minimum for edge * 2
        int level = (int)log2f((float)hypergraph->num_nodes/(float)(hyperedge_size * 2));
        #else // fixed level
        int level = 0;
        if (level > (int)log2f((float)hypergraph->num_nodes/(float)hyperedge_size))
        {
            level = (int)log2f((float)hypergraph->num_nodes/(float)hyperedge_size);
        }
        #endif
        
        int group = select_group(&state, level, hypergraph->num_nodes);
        
        int start_idx, end_idx;
        get_group_bounds(group, level, hypergraph->num_nodes, &start_idx, &end_idx);
  
        int selected_count = 0;
        int consecutive_failures = 0;
        int groups_tried = 0;
        const int MAX_FAILURES = 100;
        
        while (selected_count < hyperedge_size)
        {
            bool already_selected = false;
            int node_idx = -1;
            #if 0 // weighted (slow)
            float total_weight = 0.0f;
            for (int idx = start_idx; idx < end_idx; idx++) 
            {
                bool already_selected2 = false;
                for (int jdx = 0; jdx < selected_count; jdx++)
                {
                    if (hypergraph->edge_nodes[edge_offset + jdx] == idx)
                    {
                        already_selected2 = true;
                        break;
                    }
                }
                if (!already_selected2)
                {
                    total_weight += hypergraph->node_weights[idx];
                }
            }
            
            if (total_weight > 0.0f)
            {
                // Generate random value in [0, total_weight)
                float rand_weight = curand_uniform(&state) * total_weight;
                
                // Find node corresponding to this weight
                float cumulative_weight = 0.0f;
                for (int idx = start_idx; idx < end_idx; idx++) 
                {
                    bool already_selected3 = false;
                    for (int jdx = 0; jdx < selected_count; jdx++)
                    {
                        if (hypergraph->edge_nodes[edge_offset + jdx] == idx)
                        {
                            already_selected3 = true;
                            break;
                        }
                    }
                    if (!already_selected3)
                    {
                        cumulative_weight += hypergraph->node_weights[idx];
                        if (cumulative_weight > rand_weight) 
                        {
                            node_idx = idx;
                            break;
                        }
                    }
                }
            }

            /*for (int idx = 0; idx < selected_count; idx++)
            {
                if (hypergraph->edge_nodes[edge_offset + idx] == node_idx)
                {
                    already_selected = true;
                    break;
                }
            }*/
            #elif 0 // random
            node_idx = start_idx + (curand(&state) % (end_idx - start_idx));

            for (int idx = 0; idx < selected_count; idx++)
            {
                if (hypergraph->edge_nodes[edge_offset + idx] == node_idx)
                {
                    already_selected = true;
                    break;
                }
            }
            #else // weighted (fast), use this!!
            float total_weight = 0.0f;
            for (int idx = start_idx; idx < end_idx; idx++)
            {
                total_weight += hypergraph->node_weights[idx];
            }
            
            float rand_weight = curand_uniform(&state) * total_weight;
            
            float cumulative_weight = 0.0f;
            node_idx = start_idx;
            for (int idx = start_idx; idx < end_idx; idx++) 
            {
                cumulative_weight += hypergraph->node_weights[idx];
                if (cumulative_weight > rand_weight) 
                {
                    node_idx = idx;
                    break;
                }
            }

            for (int idx = 0; idx < selected_count; idx++)
            {
                if (hypergraph->edge_nodes[edge_offset + idx] == node_idx)
                {
                    already_selected = true;
                    break;
                }
            }
            #endif
            
            if (!already_selected && node_idx != -1)
            {
                hypergraph->edge_nodes[edge_offset + selected_count] = node_idx;
                selected_count++;
                consecutive_failures = 0;
            }
            else
            {
                if (++consecutive_failures >= MAX_FAILURES)
                {
                    int old_group = group;
                    consecutive_failures = 0;
                    while((group = select_group(&state, level, hypergraph->num_nodes)) == old_group)
                    {
                        // do not move to a new level, just try a new group
                        /*if (++consecutive_failures >= MAX_FAILURES)
                        {
                            // select a new level if we can't find a new group
                            level = max(0, level - 1);
                            group = select_group(&state, level, hypergraph->num_nodes);
                            break;
                        }*/
                    }
                    get_group_bounds(group, level, hypergraph->num_nodes, &start_idx, &end_idx);
                    consecutive_failures = 0;
                    groups_tried++;
                }
            }
        }
    }
}

extern "C" __global__ void verify_partitioning_kernel(
    Hypergraph *hypergraph,
    int *partitions,
    int *cut_edges,
    unsigned int *errorflag
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx < hypergraph->num_edges)
    {
        int start = hypergraph->edge_offsets[edge_idx];
        int end = hypergraph->edge_offsets[edge_idx + 1];

        // Track which partitions are used by this edge
        bool partition_used[64] = {false};
        int num_partitions_used = 0;

        // Check each node in the edge
        for (int pos = start; pos < end; pos++)
        {
            int node = hypergraph->edge_nodes[pos];
            int partition = partitions[node];

            // Validate partition assignment
            if (partition < 0 || partition >= 64)
            {
                atomicOr(errorflag, 1u);
                return;
            }

            // Count unique partitions
            if (!partition_used[partition])
            {
                partition_used[partition] = true;
                num_partitions_used++;
            }
        }

        // Add to cut edge count
        if (num_partitions_used > 1)
        {
            atomicAdd(cut_edges, 1);
        }
    }
}

extern "C" __global__ void count_nodes_per_partition_kernel(
    Hypergraph *hypergraph,
    int *partitions,
    int *partition_counts, // array of size NUM_PARTITIONS (64)
    int num_partitions,
    unsigned int *errorflag
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node_idx < hypergraph->num_nodes) {
        int partition = partitions[node_idx];
        
        // Validate partition range
        if (partition < 0 || partition >= num_partitions) {
            atomicOr(errorflag, 1u);
            return;
        }
        
        // Count nodes in each partition
        atomicAdd(&partition_counts[partition], 1);
    }
}

extern "C" __global__ void check_partition_balance_kernel(
    Hypergraph *hypergraph,
    int *partition_counts,  // array of size NUM_PARTITIONS (64)
    int num_partitions,
    unsigned int *errorflag
) {
    int partition_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (partition_idx < num_partitions) { // NUM_PARTITIONS = 64
        float avg_partition_size = (float)hypergraph->num_nodes / num_partitions;
        float epsilon = 0.5f;  // Same tolerance as in Rust code
        int count = partition_counts[partition_idx];
        
        // Check if partition is balanced
        if (fabsf((float)count - avg_partition_size) > epsilon * avg_partition_size) {
            atomicOr(errorflag, 2u); // Use different bit for different error types
        }
    }
}

extern "C" __global__ void calculate_connectivity_kernel(
    Hypergraph *hypergraph,
    int *partitions,
    int *connectivity_sum,  // Single value to store total connectivity
    unsigned int *errorflag
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx < hypergraph->num_edges) {
        int start = hypergraph->edge_offsets[edge_idx];
        int end = hypergraph->edge_offsets[edge_idx + 1];
        
        // Track unique partitions for this edge
        bool partition_used[64] = {false};
        int partitions_in_edge = 0;
        
        // Count unique partitions for this edge
        for (int pos = start; pos < end; pos++) {
            int node = hypergraph->edge_nodes[pos];
            int partition = partitions[node];
            
            // Validate partition (redundant but keeping for safety)
            if (partition < 0 || partition >= 64) {
                atomicOr(errorflag, 1u);
                return;
            }
            
            // Count unique partitions
            if (!partition_used[partition]) {
                partition_used[partition] = true;
                partitions_in_edge++;
            }
        }
        
        // Add to connectivity sum
        atomicAdd(connectivity_sum, partitions_in_edge);
        
        // Also track cut edges (for compatibility with original kernel)
        if (partitions_in_edge > 1) {
            // Note: If you have a cut_edges counter, uncomment this line
            // atomicAdd(cut_edges, 1);
        }
    }
}

extern "C" __global__ void check_connectivity_limit_kernel(
    int *connectivity_sum,
    Hypergraph *hypergraph,
    unsigned int *errorflag,
    float better_than_baseline
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Calculate baseline connectivity and maximum allowed
        int baseline_connectivity = hypergraph->num_edges;
        int maximum_connectivity = (int)(baseline_connectivity * better_than_baseline);
        
        // Check if connectivity is within limit
        if (*connectivity_sum > maximum_connectivity) {
            atomicOr(errorflag, 4u); // Use different bit for different error type
        }
    }
}

extern "C" __global__ void verify_partitioning_full_kernel(
    Hypergraph *hypergraph,
    int *partitions,
    int *cut_edges,
    int *connectivity_sum,
    unsigned int *errorflag,
    float better_than_baseline
) {
    // Reset error flags and counters
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *errorflag = 0;
        *cut_edges = 0;
        *connectivity_sum = 0;
    }
    __syncthreads();
    
    // Create shared memory for partition counts 
    __shared__ int partition_counts[64];
    
    // Initialize partition counts
    if (blockIdx.x == 0 && threadIdx.x < 64) {
        partition_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Step 1: Count nodes per partition and validate assignments
    // Block 0 will be responsible for all node counting to ensure counts are in one shared memory
    if (blockIdx.x == 0) {
        for (int node_idx = threadIdx.x; node_idx < hypergraph->num_nodes; node_idx += blockDim.x) {
            int partition = partitions[node_idx];
            
            // Validate partition range
            if (partition < 0 || partition >= 64) {
                atomicOr(errorflag, 1u);
            } else {
                // Count nodes in each partition using shared memory
                atomicAdd(&partition_counts[partition], 1);
            }
        }
        __syncthreads();
        
        // Step 2: Check partition balance (only block 0 does this)
        if (threadIdx.x < 64) {
            float avg_partition_size = (float)hypergraph->num_nodes / 64.0f;
            float epsilon = 0.5f;  // Same tolerance as in Rust code
            int count = partition_counts[threadIdx.x];
            
            // Check if partition is balanced
            if (fabsf((float)count - avg_partition_size) > epsilon * avg_partition_size) {
                atomicOr(errorflag, 2u);
            }
        }
        __syncthreads();
    }
    
    // All blocks wait for block 0 to complete node counting and balance check
    __threadfence();
    
    // Step 3: Calculate connectivity for edges - all blocks participate
    for (int edge_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         edge_idx < hypergraph->num_edges; 
         edge_idx += blockDim.x * gridDim.x) 
    {
        if (edge_idx < hypergraph->num_edges) {
            int start = hypergraph->edge_offsets[edge_idx];
            int end = hypergraph->edge_offsets[edge_idx + 1];
            
            // Local arrays for tracking partitions
            bool partition_used[64] = {false};
            int partitions_in_edge = 0;
            
            // Count unique partitions for this edge
            for (int pos = start; pos < end; pos++) {
                int node = hypergraph->edge_nodes[pos];
                int partition = partitions[node];
                
                // Skip invalid partitions (already checked)
                if (partition >= 0 && partition < 64) {
                    // Count unique partitions
                    if (!partition_used[partition]) {
                        partition_used[partition] = true;
                        partitions_in_edge++;
                    }
                }
            }
            
            // Add to connectivity sum
            atomicAdd(connectivity_sum, partitions_in_edge);
            
            // Also track cut edges
            if (partitions_in_edge > 1) {
                atomicAdd(cut_edges, 1);
            }
        }
    }
    __syncthreads();
    
    // Step 4: Check connectivity limit (single thread in block 0)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Calculate baseline connectivity and maximum allowed
        int baseline_connectivity = hypergraph->num_edges;
        int maximum_connectivity = (int)(baseline_connectivity * better_than_baseline);
        
        // Check if connectivity is within limit
        if (*connectivity_sum > maximum_connectivity) {
            atomicOr(errorflag, 4u); // Use different bit for different error type
        }
    }
}

extern "C" __global__ void solve_greedy_bipartition(
    Hypergraph *hypergraph,
    int *partitions,
    int *vertex_to_edges_values, // The actual edge IDs (flat array)
    int *vertex_to_edges_offsets, // Where each vertex's edges start (prefix sum)
    int *vertex_list, int *vertex_degrees,
    int *left_edge_flags, int *right_edge_flags,
    volatile int *barrier_counter,
    // Global memory pointers
    int *left_count_g,
    int *right_count_g,
    int *count_in_partition_g,
    int *size_left_g,
    int *size_right_g,
    int *done_g
) {
    // Hardcoded values for partitioning
    const int FIXED_NUM_PARTITIONS = 64;
    const int FIXED_DEPTH = 6;  // log2(64) = 6
    
    int *left_count_s = left_count_g;   // Now pointing to Global Memory
    int *right_count_s = right_count_g;  // Now pointing to Global Memory
    int *count_in_partition_s = count_in_partition_g; // Now pointing to Global Memory
    int *size_left_s = size_left_g;     // Now pointing to Global Memory
    int *size_right_s = size_right_g;    // Now pointing to Global Memory
    int *done_s = done_g;             // Now pointing to Global Memory

    // Initialization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Initializing: nodes=%d, edges=%d, partitions=%d, depth=%d\n",
               hypergraph->num_nodes, hypergraph->num_edges, FIXED_NUM_PARTITIONS, FIXED_DEPTH);
        
        // Initialize partitions, degrees, etc.
        for (int idx = 0; idx < hypergraph->num_nodes; idx++) {
            partitions[idx] = 1;
            vertex_degrees[idx] = 0;
            vertex_list[idx] = idx;
            // Initialize the offsets array (all zeros at first)
            if (idx < hypergraph->num_nodes) {
                vertex_to_edges_offsets[idx] = 0;
            }
        }
        vertex_to_edges_offsets[hypergraph->num_nodes] = 0; // Sentinel value
        
        // First pass: count degrees
        for (int edge_idx = 0; edge_idx < hypergraph->num_edges; edge_idx++) {
            int start = hypergraph->edge_offsets[edge_idx];
            int end = hypergraph->edge_offsets[edge_idx + 1];
            for (int pos = start; pos < end; pos++) {
                int vertex = hypergraph->edge_nodes[pos];
                vertex_degrees[vertex]++;
            }
        }
        
        // Compute prefix sum for offsets
        int running_sum = 0;
        for (int v = 0; v < hypergraph->num_nodes; v++) {
            vertex_to_edges_offsets[v] = running_sum;
            running_sum += vertex_degrees[v];
        }
        vertex_to_edges_offsets[hypergraph->num_nodes] = running_sum;
        
        // Reset degrees (will use them as counters in the next step)
        for (int v = 0; v < hypergraph->num_nodes; v++) {
            vertex_degrees[v] = 0;
        }
        
        // Fill in edge IDs using vertex_to_edges_offsets
        for (int edge_idx = 0; edge_idx < hypergraph->num_edges; edge_idx++) {
            int start = hypergraph->edge_offsets[edge_idx];
            int end = hypergraph->edge_offsets[edge_idx + 1];
            for (int pos = start; pos < end; pos++) {
                int vertex = hypergraph->edge_nodes[pos];
                int insert_pos = vertex_to_edges_offsets[vertex] + vertex_degrees[vertex];
                vertex_to_edges_values[insert_pos] = edge_idx;
                vertex_degrees[vertex]++;
            }
        }
    }
    __syncthreads();

    // Bitonic sort
    if (blockIdx.x == 0) {
        for (int k = 2; k <= hypergraph->num_nodes; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                for (int i = threadIdx.x; i < hypergraph->num_nodes; i += blockDim.x) {
                    int ixj = i ^ j;
                    if (ixj > i && ixj < hypergraph->num_nodes) {
                        int v1 = vertex_list[i];
                        int v2 = vertex_list[ixj];
                        if ((i & k) == 0) {
                            if (vertex_degrees[v1] < vertex_degrees[v2]) {
                                vertex_list[i] = v2;
                                vertex_list[ixj] = v1;
                            }
                        } else {
                            if (vertex_degrees[v1] > vertex_degrees[v2]) {
                                vertex_list[i] = v2;
                                vertex_list[ixj] = v1;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
        if (threadIdx.x == 0) {
            printf("Vertices sorted by degree\n");
        }
    }
    __syncthreads();

    // Process levels - now hardcoded for 6 levels (for 64 partitions)
    for (int level = 0; level < FIXED_DEPTH; level++) {
        int num_partitions_at_level = 1 << level;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Level %d: Processing %d partitions\n", level, num_partitions_at_level);
        }

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            barrier_counter[0] = 0;
        }
        __threadfence();
        __syncthreads();

        if (blockIdx.x < num_partitions_at_level) {
            int p = (1 << level) + blockIdx.x;
            if (threadIdx.x == 0) {
                // Initialize in GLOBAL memory now
                left_count_s[blockIdx.x] = 0;
                right_count_s[blockIdx.x] = 0;
                count_in_partition_s[blockIdx.x] = 0;
                done_s[blockIdx.x] = 0;
            }
            __syncthreads();

            for (int v = threadIdx.x; v < hypergraph->num_nodes; v += blockDim.x) {
                if (partitions[v] == p) {
                    atomicAdd(&count_in_partition_s[blockIdx.x], 1);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                size_left_s[blockIdx.x] = count_in_partition_s[blockIdx.x] / 2;
                size_right_s[blockIdx.x] = count_in_partition_s[blockIdx.x] - size_left_s[blockIdx.x];
            }
            for (int e = threadIdx.x; e < hypergraph->num_edges; e += blockDim.x) {
                left_edge_flags[e] = 0;
                right_edge_flags[e] = 0;
            }
            __syncthreads();

            if (count_in_partition_s[blockIdx.x] > 0) {
                for (int idx = 0; idx < hypergraph->num_nodes; idx++) {
                    int v = vertex_list[idx];
                    if (partitions[v] != p) continue;

                    if (threadIdx.x == 0) {
                        int connections_left = 0;
                        int connections_right = 0;
                        
                        // Get range of edges for this vertex
                        int start_pos = vertex_to_edges_offsets[v];
                        int end_pos = vertex_to_edges_offsets[v+1];
                        
                        // Loop through this vertex's edges
                        for (int pos = start_pos; pos < end_pos; pos++) {
                            int edge_idx = vertex_to_edges_values[pos];
                            if (left_edge_flags[edge_idx]) connections_left++;
                            if (right_edge_flags[edge_idx]) connections_right++;
                        }

                        int left_child = p * 2;
                        int right_child = p * 2 + 1;

                        if (size_left_s[blockIdx.x] <= left_count_s[blockIdx.x]) {
                            partitions[v] = right_child;
                            atomicAdd(&right_count_s[blockIdx.x], 1);
                            for (int e = start_pos; e < end_pos; e++) {
                                int edge_idx = vertex_to_edges_values[e];
                                right_edge_flags[edge_idx] = 1;
                            }
                        } else if (size_right_s[blockIdx.x] <= right_count_s[blockIdx.x]) {
                            partitions[v] = left_child;
                            atomicAdd(&left_count_s[blockIdx.x], 1);
                            for (int e = start_pos; e < end_pos; e++) {
                                int edge_idx = vertex_to_edges_values[e];
                                left_edge_flags[edge_idx] = 1;
                            }
                        } else if (connections_left > connections_right) {
                            partitions[v] = left_child;
                            atomicAdd(&left_count_s[blockIdx.x], 1);
                            for (int e = start_pos; e < end_pos; e++) {
                                int edge_idx = vertex_to_edges_values[e];
                                left_edge_flags[edge_idx] = 1;
                            }
                        } else {
                            partitions[v] = right_child;
                            atomicAdd(&right_count_s[blockIdx.x], 1);
                            for (int e = start_pos; e < end_pos; e++) {
                                int edge_idx = vertex_to_edges_values[e];
                                right_edge_flags[edge_idx] = 1;
                            }
                        }

                        if (size_left_s[blockIdx.x] + size_right_s[blockIdx.x] <= (left_count_s[blockIdx.x] + right_count_s[blockIdx.x]) && !done_s[blockIdx.x]) {
                            printf("Split partition %d: left=%d vertices, right=%d vertices\n",
                                   p, left_count_s[blockIdx.x], right_count_s[blockIdx.x]);
                            done_s[blockIdx.x] = 1;
                        }
                    }
                    __syncthreads();
                }
            }

            if (threadIdx.x == 0) {
                atomicAdd((int*)barrier_counter, 1);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            while (barrier_counter[0] < num_partitions_at_level) {
                // Spin
            }
        }
        __syncthreads();
    }

    // Final processing with validity check
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Iterative bipartitioning complete\n");

        // Check if all vertices are in leaf partitions
        int num_invalid = 0;
        for (int v = 0; v < hypergraph->num_nodes; v++) {
            int tree_id = partitions[v];
            if (tree_id < FIXED_NUM_PARTITIONS || tree_id >= 2 * FIXED_NUM_PARTITIONS) {
                num_invalid++;
            }
        }
        if (num_invalid == 0) {
            printf("All vertices are in valid leaf partitions.\n");
        } else {
            printf("Found %d vertices not in valid leaf partitions.\n", num_invalid);
        }

        // Map tree IDs to sequential IDs
        for (int v = 0; v < hypergraph->num_nodes; v++) {
            int tree_id = partitions[v];
            int level_of_node = (int)log2f(tree_id);
            int position_in_level = tree_id - (1 << level_of_node);
            partitions[v] = position_in_level;
        }

        printf("Final partition distribution:\n");
        int counts[64] = {0}; // Hardcoded for 64 partitions
        for (int idx = 0; idx < hypergraph->num_nodes; idx++) {
            counts[partitions[idx]]++;
        }
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 16; c++) {
                printf("%d ", counts[r * 16 + c]);
            }
            printf("\n");
        }
    }
}

extern "C" __global__ void setup_hypergraph_kernel(
    Hypergraph *graph,
    int num_nodes,
    int num_edges,
    int *edge_sizes,
    int *edge_offsets,
    int *edge_nodes,
    float *node_weights
)
{
    // Single-threaded kernel to set up the struct
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        graph->num_nodes = num_nodes;
        graph->num_edges = num_edges;
        graph->edge_sizes = edge_sizes;
        graph->edge_offsets = edge_offsets;
        graph->edge_nodes = edge_nodes;
        graph->node_weights = node_weights;
    }
}