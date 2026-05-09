#include <cuda_fp16.h>
#include <float.h>

extern "C" __global__ void there_v7_compute_norms_250(
    const float* __restrict__ vectors,
    float* __restrict__ norms,
    int count
) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= count) return;

    const float* __restrict__ v = vectors + (long long)row * 250ll;
    float sum = 0.0f;

    #pragma unroll
    for (int d = 0; d < 250; ++d) {
        float x = v[d];
        sum = fmaf(x, x, sum);
    }

    norms[row] = sum;
}

extern "C" __global__ void there_v7_pack_fp16_250_to_256(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int rows,
    int src_stride,
    int dst_stride
) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const float* __restrict__ src_row = src + (long long)row * src_stride;
    unsigned short* __restrict__ dst_row = dst + (long long)row * dst_stride;

    #pragma unroll
    for (int d = 0; d < 250; ++d) {
        dst_row[d] = __half_as_ushort(__float2half(src_row[d]));
    }

    #pragma unroll
    for (int d = 250; d < 256; ++d) {
        dst_row[d] = 0;
    }
}

extern "C" __global__ void there_v7_pack_fp16_linear(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = __half_as_ushort(__float2half(src[i]));
    }
}

extern "C" __global__ void there_v7_reduce_gemm_tile_f32(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const float* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int db_start,
    int tile_len,
    int num_queries,
    int first_tile
) {
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;
    float query_norm = query_norms[q];
    float current_best = first_tile ? FLT_MAX : best_dist[q];
    int current_idx = first_tile ? -1 : best_idx[q];

    float local_best = FLT_MAX;
    int local_idx = -1;
    const float* __restrict__ row = dot_tile + (long long)q * tile_len;

    for (int j = lane; j < tile_len; j += 32) {
        float dist = query_norm + database_norms[db_start + j] + row[j];
        int idx = db_start + j;
        if (dist < local_best || (dist == local_best && idx < local_idx)) {
            local_best = dist;
            local_idx = idx;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFFu, local_best, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFFu, local_idx, offset);
        if (other_dist < local_best || (other_dist == local_best && other_idx < local_idx)) {
            local_best = other_dist;
            local_idx = other_idx;
        }
    }

    if (lane == 0 && (local_best < current_best ||
        (local_best == current_best && (current_idx < 0 || local_idx < current_idx)))) {
        best_dist[q] = local_best;
        best_idx[q] = local_idx;
    }
}

extern "C" __global__ void there_v7_reduce_gemm_tile_fp16(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const unsigned short* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int db_start,
    int tile_len,
    int num_queries,
    int first_tile
) {
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;
    float query_norm = query_norms[q];
    float current_best = first_tile ? FLT_MAX : best_dist[q];
    int current_idx = first_tile ? -1 : best_idx[q];

    float local_best = FLT_MAX;
    int local_idx = -1;
    const unsigned short* __restrict__ row = dot_tile + (long long)q * tile_len;

    for (int j = lane; j < tile_len; j += 32) {
        float dot = __half2float(__ushort_as_half(row[j]));
        float dist = query_norm + database_norms[db_start + j] + dot;
        int idx = db_start + j;
        if (dist < local_best || (dist == local_best && idx < local_idx)) {
            local_best = dist;
            local_idx = idx;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFFu, local_best, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFFu, local_idx, offset);
        if (other_dist < local_best || (other_dist == local_best && other_idx < local_idx)) {
            local_best = other_dist;
            local_idx = other_idx;
        }
    }

    if (lane == 0 && (local_best < current_best ||
        (local_best == current_best && (current_idx < 0 || local_idx < current_idx)))) {
        best_dist[q] = local_best;
        best_idx[q] = local_idx;
    }
}
