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

// Squared L2 norm of every column of a (rows x cols) column-major matrix.
// out[j] = sum_i  mat[i + j*rows]^2
extern "C" __global__ void col_sq_norms_kernel(
    const float *mat, float *out, int rows, int cols
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float sum = 0.0f;
    const float *col = mat + (long long)j * rows;
    for (int i = 0; i < rows; i++) {
        float v = col[i];
        sum += v * v;
    }
    out[j] = sum;
}

// Squared L2 norm of every row of a (rows x cols) column-major matrix.
// out[i] = sum_j  mat[i + j*rows]^2
extern "C" __global__ void row_sq_norms_kernel(
    const float *mat, float *out, int rows, int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float v = mat[i + (long long)j * rows];
        sum += v * v;
    }
    out[i] = sum;
}
