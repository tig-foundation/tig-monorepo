#include <curand_kernel.h>
#include <stdint.h>

// Fill `size` elements with iid N(0, scale) values.
extern "C" __global__ void standard_gaussian_kernel(
    float *mat, int size, float scale, uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState state;
    curand_init((unsigned long long)seed, (unsigned long long)idx, 0, &state);
    mat[idx] = curand_normal(&state) * scale;
}

// Scale row i of a column-major (rows x cols) matrix by scales[i].
extern "C" __global__ void scale_rows_kernel(
    float *mat, const float *scales, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int i = idx % rows;
    mat[idx] *= scales[i];
}

// Scale col j of a column-major (rows x cols) matrix by scales[j].
extern "C" __global__ void scale_cols_kernel(
    float *mat, const float *scales, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int j = idx / rows;
    mat[idx] *= scales[j];
}
