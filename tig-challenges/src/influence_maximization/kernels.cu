#include <curand_kernel.h>

extern "C" __global__ void rmat_kernel(
    int32_t *from_nodes,
    int32_t *to_nodes,
    int scale,
    float a,
    float b,
    float c,
    float d,
    unsigned long long seed,
    int num_edges
) {
    curandState state;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x) {
        curand_init(seed, idx, 0, &state);

        int32_t src = 0;
        int32_t dst = 0;
        // For each bit position high->low
        for (int bit = scale - 1; bit >= 0; --bit) {
            float r = curand_uniform(&state); // (0,1]
            // Decide quadrant
            if (r <= a) {
                // top-left: nothing
            } else if (r <= a + b) {
                // top-right: set dst bit
                dst |= (1u << bit);
            } else if (r <= a + b + c) {
                // bottom-left: set src bit
                src |= (1u << bit);
            } else {
                // bottom-right: set both bits
                src |= (1u << bit);
                dst |= (1u << bit);
            }
        }

        if (src == dst) {
            from_nodes[idx] = -1;
            to_nodes[idx] = -1;
        } else {
            from_nodes[idx] = src;
            to_nodes[idx] = dst;
        }
    }
}

extern "C" __global__ void count_degrees_kernel(int* from_nodes, int* degrees, int num_edges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges && from_nodes[i] != -1) {
        atomicAdd(&degrees[from_nodes[i]], 1);
    }
}

extern "C" __global__ void cascade(
    int32_t *from_nodes,
    int32_t *to_nodes,
    bool *activated_nodes,
    bool *cascade_nodes,
    bool *next_cascade_nodes,
    int32_t *num_new_activations,
    float activation_prob,
    unsigned long long seed,
    int num_edges    
) {
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x) {
        int32_t src = from_nodes[idx];
        int32_t dst = to_nodes[idx];

        if (src == -1) {
            continue;
        }

        if (cascade_nodes[src] && !activated_nodes[dst]) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            float r = curand_uniform(&state); // (0,1]
            if (curand_uniform(&state) < activation_prob) {
                activated_nodes[dst] = true;
                next_cascade_nodes[dst] = true;
                atomicAdd(num_new_activations, 1);
            }
        }
    }
}
