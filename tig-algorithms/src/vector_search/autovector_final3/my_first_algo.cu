

#include <float.h>
#include <stdint.h>
#include <cuda_fp16.h>

extern "C" __global__ void compute_norms_v122(
    const float* __restrict__ vecs,
    float*       __restrict__ norms,
    int count, int dims
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float* row = vecs + (long long)i * dims;
    float sum = 0.0f;
    #pragma unroll 10
    for (int d = 0; d < 250; d++) {
        float v = row[d];
        sum = __fmaf_rn(v, v, sum);
    }
    norms[i] = sum;
}

extern "C" __global__ void find_best_v122(
    const float* __restrict__ q_norms,
    const float* __restrict__ db_norms,
    const float* __restrict__ d_dot,
    int*   best_idx,
    float* best_dist,
    int db_start,
    int db_tile_len,
    int num_queries,
    int first_tile
) {
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;

    float qn       = q_norms[q];
    float cur_best = first_tile ? FLT_MAX : best_dist[q];
    int   cur_idx  = first_tile ? -1 : best_idx[q];

    float local_dx = FLT_MAX;
    int   local_ix = -1;

    const float* row = d_dot + (long long)q * db_tile_len;
    for (int j = lane; j < db_tile_len; j += 32) {
        float dist = qn + db_norms[db_start + j] + row[j];
        if (dist < local_dx) { local_dx = dist; local_ix = db_start + j; }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float od = __shfl_down_sync(0xFFFFFFFF, local_dx, off);
        int   oi = __shfl_down_sync(0xFFFFFFFF, local_ix, off);
        if (od < local_dx || (od == local_dx && oi < local_ix)) {
            local_dx = od;
            local_ix = oi;
        }
    }

    if (lane == 0 &&
        (local_dx < cur_best || (local_dx == cur_best && local_ix < cur_idx))) {
        best_dist[q] = local_dx;
        best_idx[q]  = local_ix;
    }
}

extern "C" __global__ void convert_fp32_to_fp16_padded(
    const float*    __restrict__ in,
    unsigned short* __restrict__ out,
    int count_rows,
    int in_stride,
    int out_stride
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count_rows) return;
    const float*    row_in  = in  + (long long)i * in_stride;
    unsigned short* row_out = out + (long long)i * out_stride;
    #pragma unroll 10
    for (int d = 0; d < 250; d++) {
        row_out[d] = __half_as_ushort(__float2half(row_in[d]));
    }

    #pragma unroll
    for (int d = 250; d < 256; d++) {
        row_out[d] = 0;
    }
}

extern "C" __global__ void find_best_fp16_t19(
    const float*          __restrict__ q_norms,
    const float*          __restrict__ db_norms,
    const unsigned short* __restrict__ d_dot_u16,
    int*   best_idx,
    float* best_dist,
    int db_start,
    int db_tile_len,
    int num_queries,
    int first_tile
) {
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;

    float qn       = q_norms[q];
    float cur_best = first_tile ? FLT_MAX : best_dist[q];
    int   cur_idx  = first_tile ? -1      : best_idx[q];

    float local_dx = FLT_MAX;
    int   local_ix = -1;

    const unsigned short* row = d_dot_u16 + (long long)q * db_tile_len;
    for (int j = lane; j < db_tile_len; j += 32) {
        float dot_val = __half2float(__ushort_as_half(row[j]));
        float dist = qn + db_norms[db_start + j] + dot_val;
        if (dist < local_dx) { local_dx = dist; local_ix = db_start + j; }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float od = __shfl_down_sync(0xFFFFFFFF, local_dx, off);
        int   oi = __shfl_down_sync(0xFFFFFFFF, local_ix, off);
        if (od < local_dx || (od == local_dx && oi < local_ix)) {
            local_dx = od;
            local_ix = oi;
        }
    }

    if (lane == 0 &&
        (local_dx < cur_best || (local_dx == cur_best && local_ix < cur_idx))) {
        best_dist[q] = local_dx;
        best_idx[q]  = local_ix;
    }
}

extern "C" __global__ void convert_fp32_to_fp16(
    const float* __restrict__ in,
    unsigned short* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half_as_ushort(__float2half(in[i]));
    }
}

extern "C" __global__ void find_best_fp16_t20(
    const float*          __restrict__ q_norms,
    const float*          __restrict__ db_norms,
    const unsigned short* __restrict__ d_dot_u16,
    int*   best_idx,
    float* best_dist,
    int db_start,
    int db_tile_len,
    int num_queries,
    int first_tile
) {
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;

    float qn       = q_norms[q];
    float cur_best = first_tile ? FLT_MAX : best_dist[q];
    int   cur_idx  = first_tile ? -1 : best_idx[q];

    float local_dx = FLT_MAX;
    int   local_ix = -1;

    const unsigned short* row = d_dot_u16 + (long long)q * db_tile_len;
    for (int j = lane; j < db_tile_len; j += 32) {

        float dot_val = __half2float(__ushort_as_half(row[j]));
        float dist = qn + db_norms[db_start + j] + dot_val;
        if (dist < local_dx) { local_dx = dist; local_ix = db_start + j; }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float od = __shfl_down_sync(0xFFFFFFFF, local_dx, off);
        int   oi = __shfl_down_sync(0xFFFFFFFF, local_ix, off);
        if (od < local_dx || (od == local_dx && oi < local_ix)) {
            local_dx = od;
            local_ix = oi;
        }
    }

    if (lane == 0 &&
        (local_dx < cur_best || (local_dx == cur_best && local_ix < cur_idx))) {
        best_dist[q] = local_dx;
        best_idx[q]  = local_ix;
    }
}
