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

    float query_norm = lane == 0 ? query_norms[q] : 0.0f;
    query_norm = __shfl_sync(0xFFFFFFFFu, query_norm, 0);

    float current_best = FLT_MAX;
    int current_idx = -1;
    if (!first_tile && lane == 0) {
        current_best = best_dist[q];
        current_idx = best_idx[q];
    }
    current_best = __shfl_sync(0xFFFFFFFFu, current_best, 0);
    current_idx = __shfl_sync(0xFFFFFFFFu, current_idx, 0);

    float local_best = FLT_MAX;
    int local_idx = -1;
    const float* __restrict__ row = dot_tile + (long long)q * tile_len;
    const float* __restrict__ db_norm_row = database_norms + db_start;

    for (int j = lane; j < tile_len; j += 32) {
        float dist = query_norm + db_norm_row[j] + row[j];
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
    int q_base,
    int db_start,
    int tile_len,
    int num_queries,
    int first_tile
) {    
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;
    int q_idx = q_base + q;

    float query_norm = lane == 0 ? query_norms[q_idx] : 0.0f;
    query_norm = __shfl_sync(0xFFFFFFFFu, query_norm, 0);

    float current_best = FLT_MAX;
    int current_idx = -1;
    if (!first_tile && lane == 0) {
        current_best = best_dist[q_idx];
        current_idx = best_idx[q_idx];
    }
    current_best = __shfl_sync(0xFFFFFFFFu, current_best, 0);
    current_idx = __shfl_sync(0xFFFFFFFFu, current_idx, 0);

    float local_best = FLT_MAX;
    int local_idx = -1;
    const unsigned short* __restrict__ row = dot_tile + (long long)q * tile_len;
    const float* __restrict__ db_norm_row = database_norms + db_start;
    const half* __restrict__ row_h = reinterpret_cast<const half*>(row);

    const int vec_end = tile_len & ~1;
    const bool paired_path = ((reinterpret_cast<unsigned long long>(row_h) & 0x3ull) == 0ull);

    if (paired_path) {
        for (int j = lane << 1; j < vec_end; j += 64) {
            half2 dots_h2 = *reinterpret_cast<const half2*>(row_h + j);
            float2 dots = __half22float2(dots_h2);

            int idx0 = db_start + j;
            float dist0 = query_norm + db_norm_row[j] + dots.x;
            if (dist0 < local_best || (dist0 == local_best && idx0 < local_idx)) {
                local_best = dist0;
                local_idx = idx0;
            }

            int idx1 = idx0 + 1;
            float dist1 = query_norm + db_norm_row[j + 1] + dots.y;
            if (dist1 < local_best || (dist1 == local_best && idx1 < local_idx)) {
                local_best = dist1;
                local_idx = idx1;
            }
        }

        int j = vec_end + lane;
        if (j < tile_len) {
            float dot = __half2float(row_h[j]);
            float dist = query_norm + db_norm_row[j] + dot;
            int idx = db_start + j;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
        }
    } else {
        for (int j = lane; j < tile_len; j += 32) {
            float dot = __half2float(__ushort_as_half(row[j]));
            float dist = query_norm + db_norm_row[j] + dot;
            int idx = db_start + j;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
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
        best_dist[q_idx] = local_best;
        best_idx[q_idx] = local_idx;
    }
}
