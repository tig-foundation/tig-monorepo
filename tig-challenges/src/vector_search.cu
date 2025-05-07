#include <curand_kernel.h>
#include <stdint.h>

extern "C" __global__ void generate_clusters(
    const uint8_t *seed,
    const float avg_weight,
    const int vector_dims,
    const float var,
    const float alpha,
    const int num_clusters,
    float *cluster_means,
    float *cluster_stds,
    float *cluster_weights
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_clusters; i += blockDim.x * gridDim.x) 
    {
        curandState state;
        curand_init(((uint64_t *)(seed))[i % 4], i, 0, &state);
    
        cluster_weights[i] = expf(avg_weight + curand_normal(&state) * sqrtf(var));
        float *means = cluster_means + i * vector_dims;
        float *stds = cluster_stds + i * vector_dims;
        float sigma = curand_uniform(&state) * 2 * alpha + 1.05 - alpha;
        float epsilon = curand_uniform(&state) * alpha;
        for (int j = 0; j < vector_dims; ++j)
        {
            means[j] = curand_uniform(&state) * 2.0 - 1.0;
            stds[j] = curand_uniform(&state) * 2 * epsilon + sigma - epsilon;
        }
    }
}

__device__ int binary_search(
    const float *arr,
    const float target,
    const int size
)
{
    int left = 0;
    int right = size - 1;
    int result = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

__device__ float truncated_normal(
    curandState *state,
    const float mean,
    const float std,
    const float lower,
    const float upper
)
{
    float a = (lower - mean) / std;
    float b = (upper - mean) / std;

    // Uniform sample from truncated CDF range
    float u = curand_uniform(state);
    float p = 0.5f * (1.0f + erff(a / sqrtf(2.0f))) + u * (0.5f * (1.0f + erff(b / sqrtf(2.0f))) - 0.5f * (1.0f + erff(a / sqrtf(2.0f))));

    // Invert CDF using inverse error function
    return mean + std * sqrtf(2.0f) * erfinvf(2.0f * p - 1.0f);
}

extern "C" __global__ void generate_vectors(
    const uint8_t *seed,
    const int database_size,
    const int query_size,
    const int vector_dims,
    const int num_clusters,
    const float *cluster_cum_prob,
    const float *cluster_means,
    const float *cluster_stds,
    float *database_vectors,
    float *query_vectors
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < database_size; i += blockDim.x * gridDim.x) 
    {   
        curandState state;
        curand_init(((uint64_t *)(seed))[i % 4], i, 0, &state);
        
        int cluster_idx = binary_search(cluster_cum_prob, curand_uniform(&state), num_clusters);
        const float *means = cluster_means + cluster_idx * vector_dims;
        const float *stds = cluster_stds + cluster_idx * vector_dims;

        float *vector = database_vectors + i * vector_dims;
        for (int j = 0; j < vector_dims; ++j) 
        {
            vector[j] = truncated_normal(
                &state, 
                means[j], 
                stds[j],
                -1.0f, 
                1.0f
            );
        }
    }

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < query_size; i += blockDim.x * gridDim.x) 
    {   
        curandState state;
        curand_init(((uint64_t *)(seed))[i % 4], database_size + i, 0, &state);
        
        int cluster_idx = binary_search(cluster_cum_prob, curand_uniform(&state), num_clusters);
        const float *means = cluster_means + cluster_idx * vector_dims;
        const float *stds = cluster_stds + cluster_idx * vector_dims;

        float *vector = query_vectors + i * vector_dims;
        for (int j = 0; j < vector_dims; ++j) 
        {
            vector[j] = truncated_normal(
                &state, 
                means[j], 
                stds[j],
                -1.0f, 
                1.0f
            );
        }
    }
}

extern "C" __global__ void calc_total_distance(
    const uint32_t vector_dims,
    const uint32_t database_size,
    const uint32_t num_queries,
    const float *query_vectors,
    const float *database_vectors,
    const size_t *solution_indexes,
    float *total_distance,
    int *error_flag
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_queries) {
        size_t search_index = solution_indexes[idx];

        if (search_index >= database_size) {
            *error_flag = 1;
            return;
        }

        const float *search = database_vectors + search_index * vector_dims;
        const float *query = query_vectors + idx * vector_dims;
        
        float dist = 0.0f;
        for (int i = 0; i < vector_dims; ++i) {
            float diff = query[i] - search[i];
            dist += diff * diff;
        }
        dist = sqrtf(dist);

        atomicAdd(total_distance, dist);
    }
}