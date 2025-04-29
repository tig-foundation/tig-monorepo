#include <curand_kernel.h>
#include <stdint.h>
#include <math.h>
#include <float.h>


extern "C" __global__ void simple_search(
    const uint8_t *seeds,
    const uint32_t vector_dims,
    const uint32_t database_size,
    const uint32_t num_queries,
    uint32_t *solution_indexes
) {
    int search_idx = blockIdx.x;
    if (search_idx >= num_queries) {
        return;
    }
    
    __shared__ float query_vector[250];
    if (threadIdx.x == 0) {
        int id = database_size + search_idx;
        curandState state;
        curand_init(((uint64_t *)(seeds))[id % 4], id, 0, &state);
    
        for (int j = 0; j < vector_dims; j++) {
            query_vector[j] = curand_uniform(&state); // Random float in [0, 1]
        }
    }
    __syncthreads();

    
    float closest_dist = FLT_MAX;
    __shared__ float distance[1024]; // Adjust size as needed
    for (int i = 0; i < database_size; i += blockDim.x) {
        distance[threadIdx.x] = 0.0f;
        __syncthreads();

        curandState state;
        curand_init(((uint64_t *)(seeds))[i % 4], i, 0, &state);

        float dist = 0.0f;
        for (int j = 0; j < vector_dims; ++j) {
            float diff = curand_uniform(&state) - query_vector[j];
            dist += diff * diff;
        }
        distance[threadIdx.x] = dist;
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int j = 0; j < blockDim.x; j++) {
                if (distance[j] < closest_dist) {
                    closest_dist = distance[j];
                    solution_indexes[search_idx] = i + j;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        printf("Closest distance for query %d: %f\n", search_idx, closest_dist);
    }
}