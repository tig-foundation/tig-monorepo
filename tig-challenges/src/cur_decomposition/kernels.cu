// CUDA kernel to generate a random matrix with values between 0 and 1
#include <curand_kernel.h>

extern "C" __global__ void gaussian_matrix_kernel(
    float* matrix,
    int num_rows,
    int num_cols,
    float max_d,
    unsigned long long seed
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_rows * num_cols) {
        int col = idx / num_rows; // column major
        float d = max_d - (max_d - 1.0) * float(col) / float(num_cols - 1);

        curandState state;
        curand_init(seed, idx, 0, &state);
        matrix[idx] = curand_normal(&state) * d;
    }
}


extern "C" __global__ void scale_columns_kernel(
    float* matrix,
    int num_rows,
    int num_cols,
    float* singular_values
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_rows * num_cols) {
        int col = idx / num_rows; // column major
        matrix[idx] *= singular_values[col];
    }
}
