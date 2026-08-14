#include <curand_kernel.h>

// Legacy kernel retained for compatibility with existing TIG PTX artifacts.
extern "C" __global__ void gaussian_matrix_kernel(
    float* matrix,
    int num_rows,
    int num_cols,
    float max_d,
    unsigned long long seed
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_rows * num_cols) {
        int col = idx / num_rows;
        float d = max_d - (max_d - 1.0f) * float(col) / float(num_cols - 1);
        curandState state;
        curand_init(seed, idx, 0, &state);
        matrix[idx] = curand_normal(&state) * d;
    }
}

// Shared-basis production generator. A bounded portable grid initializes one
// counter-based Philox state per thread and each thread generates many batches
// of four independent N(0, 1) values. This avoids the former
// O(num_rows * num_cols) curand_init calls.
extern "C" __global__ void gaussian_matrix_kernel_philox(
    float* matrix,
    int num_rows,
    int num_cols,
    float max_d,
    unsigned long long seed
) {
    const unsigned long long total =
        (unsigned long long)num_rows * (unsigned long long)num_cols;
    const unsigned long long thread_idx =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long thread_count =
        (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long idx = thread_idx * 4ULL;
    if (idx >= total) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_idx, 0ULL, &state);
    const unsigned long long stride = thread_count * 4ULL;

    while (idx < total) {
        const float4 values = curand_normal4(&state);
        const float batch[4] = {values.x, values.y, values.z, values.w};
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const unsigned long long output_idx = idx + (unsigned long long)lane;
            if (output_idx < total) {
                const int col = (int)(output_idx / (unsigned long long)num_rows);
                const float d = max_d -
                    (max_d - 1.0f) * (float)col / (float)(num_cols - 1);
                matrix[output_idx] = batch[lane] * d;
            }
        }
        idx += stride;
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
