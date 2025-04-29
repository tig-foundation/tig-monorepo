//------------------------------------------------------------------------------
//------------ Generate Instance Code Begins Here ------------------------------
//------------------------------------------------------------------------------

#include <curand_kernel.h>
#include <stdint.h>


__device__ void random_vector(
    uint8_t *seeds,
    uint64_t id,
    uint32_t vector_dims,
    float *vector
) {
    curandState state;
    curand_init(((uint64_t *)(seeds))[id % 4], id, 0, &state);

    for (int j = 0; j < vector_dims; j++) {
        vector[j] = curand_uniform(&state); // Random float in [0, 1]
    }
}

//
// CUDA kernel to calculate the total distance of a solution
// 
extern "C" __global__ void calc_total_distance(
    uint8_t *seeds,
    uint32_t vector_dims,       // Length of vectors
    uint32_t database_size,     // Number of database vectors
    uint32_t num_queries,       // Number of query vectors
    float *query_vectors,       // Output: query vectors
    float *database_vectors,    // Output: database vectors
    size_t *solution_indexes,   // Input: solution indexes
    float *total_distance,     // Output: total distance for all blocks
    int *error_flag          // Output: != 0 means error occurred
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_queries) {
        size_t search_index = solution_indexes[idx];

        if (search_index >= database_size) {
            *error_flag = 1;
            return;
        }

        // use all bits of the seed
        float *search = database_vectors + idx * vector_dims;
        float *query = query_vectors + idx * vector_dims;
        random_vector(seeds, search_index, vector_dims, search);
        random_vector(seeds, database_size + idx, vector_dims, query);

        float dist = 0.0f;
        for (int i = 0; i < vector_dims; ++i) {
            float diff = query[i] - search[i];
            dist += diff * diff;
        }
        dist = sqrtf(dist);

        atomicAdd(total_distance, dist);
    }
}