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


extern "C" __global__ void extract_columns_kernel(
    float* source_matrix,
    float* dest_matrix,
    int num_rows,
    int num_cols,
    int num_dest_cols,
    int* extract_col_idxs
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_rows * num_dest_cols) {
        int row = idx % num_rows;
        int col = extract_col_idxs[idx / num_rows];
        dest_matrix[idx] = source_matrix[row + col * num_rows];
    }
}


extern "C" __global__ void extract_rows_kernel(
    float* source_matrix,
    float* dest_matrix,
    int num_rows,
    int num_cols,
    int num_dest_rows,
    int* extract_row_idxs
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_dest_rows * num_cols) {
        int row = extract_row_idxs[idx % num_dest_rows];
        int col = idx / num_dest_rows;
        dest_matrix[idx] = source_matrix[row + col * num_rows];
    }
}