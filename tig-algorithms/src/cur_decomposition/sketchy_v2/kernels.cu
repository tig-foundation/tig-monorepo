#include <curand_kernel.h>
#include <stdint.h>

// Fill `size` elements with iid N(0, scale) values. Challenge-owned kernels
// provide extraction; this is the only custom kernel needed by sketchy_v2.
extern "C" __global__ void standard_gaussian_kernel(
    float *mat, int size, float scale, uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState state;
    curand_init((unsigned long long)seed, (unsigned long long)idx, 0, &state);
    mat[idx] = curand_normal(&state) * scale;
}
