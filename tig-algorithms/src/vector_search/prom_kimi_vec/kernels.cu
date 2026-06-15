// Copyright (c) 2026 NVX — autovector_final2 t16_i122
// cuBLAS on TIG stream via CudaBlas::new(stream.clone()).
// d_dot layout (col-major from cuBLAS, m=tile_len, n=nq, ldc=tile_len):
//   d_dot[j + q*tile_len] = -2*dot(DB_tile[j], Q[q])
// first_tile=1 on tile 0: use FLT_MAX for cur_best (not alloc_zeros 0.0).
#include <float.h>
#include <stdint.h>

extern "C" __global__ void compute_norms_v122_alt(
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

// Each warp (32 lanes) handles one query q.
// lane = threadIdx.x (0..31), q = blockIdx.y * blockDim.y + threadIdx.y
extern "C" __global__ void find_best_v122_alt(
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

    // Warp reduce: find minimum across 32 lanes
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