#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

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



extern "C" __global__ void there_v9_pack_norm_fp16_250_to_256(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    float* __restrict__ norms,
    int rows,
    int src_stride,
    int dst_stride
) {    
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const float* __restrict__ src_row = src + (long long)row * src_stride;
    int row_tile = row >> 4;
    int row_in_tile = row & 15;
    unsigned short* __restrict__ dst_base =
        dst + (long long)row_tile * (16ll * dst_stride) + (long long)row_in_tile * 16ll;
    float sum = 0.0f;

    #pragma unroll
    for (int kt = 0; kt < 15; ++kt) {
        const float* __restrict__ s = src_row + kt * 16;
        unsigned short* __restrict__ d = dst_base + kt * 256;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            float x = s[c];
            sum = fmaf(x, x, sum);
            d[c] = __half_as_ushort(__float2half(x));
        }
    }

    const float* __restrict__ s = src_row + 240;
    unsigned short* __restrict__ d = dst_base + 15 * 256;
    #pragma unroll
    for (int c = 0; c < 10; ++c) {
        float x = s[c];
        sum = fmaf(x, x, sum);
        d[c] = __half_as_ushort(__float2half(x));
    }

    norms[row] = sum;
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

extern "C" __global__ void there_v9_wmma_chunk_best(
    const unsigned short* __restrict__ q_half_u16,
    const unsigned short* __restrict__ db_half_u16,
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    float* __restrict__ chunk_best_dist,
    int* __restrict__ chunk_best_idx,
    int num_queries,
    int database_size,
    int query_stride,
    int chunk_db
) {    
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int WK = 16;
    constexpr int DPAD = 256;
    constexpr int KTILES = DPAD / WK;
    constexpr int WARP_TILES = 4;
    constexpr int TILE_ELEMS = WM * WK;
    constexpr int PACKED_TILE_STRIDE = KTILES * TILE_ELEMS;
    constexpr int THREADS_PER_BLOCK = 32 * WARP_TILES;
    constexpr int TILE_VEC4 = PACKED_TILE_STRIDE / 8;

    int chunk_id = (int)blockIdx.y;
    int chunk_start = chunk_id * chunk_db;
    int chunk_len = database_size - chunk_start;
    if (chunk_len <= 0) return;
    if (chunk_len > chunk_db) chunk_len = chunk_db;

    int lane = (int)threadIdx.x;
    int warp_id = (int)threadIdx.y;
    int tid = warp_id * 32 + lane;
    int q_tile_id = (int)blockIdx.x * WARP_TILES + warp_id;
    int q_base = q_tile_id * WM;
    bool warp_active = q_base < query_stride;
    int q = q_base + lane;

    __shared__ float dot_tiles[WARP_TILES][WM * WN];
    __shared__ __align__(16) int4 db_tile_vec[TILE_VEC4];
    __shared__ float db_norm_tile[WN];
    float* __restrict__ dots = dot_tiles[warp_id];

    const half* __restrict__ q_half = reinterpret_cast<const half*>(q_half_u16);
    const half* __restrict__ db_half = reinterpret_cast<const half*>(db_half_u16);

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[KTILES];
    if (warp_active) {
        const half* __restrict__ q_tile_base = q_half + (long long)q_tile_id * PACKED_TILE_STRIDE;
        #pragma unroll
        for (int k = 0; k < KTILES; ++k) {
            wmma::load_matrix_sync(a[k], q_tile_base + (long long)k * TILE_ELEMS, WN);
        }
    }

    float best_dist = FLT_MAX;
    int best_idx = -1;
    float q_norm = (warp_active && lane < WM && q < num_queries) ? query_norms[q] : 0.0f;

    int db_tile_start = chunk_start >> 4;
    int db_tiles = (chunk_len + WN - 1) / WN;

    for (int tile = 0; tile < db_tiles; ++tile) {
        int db_local = tile * WN;
        int valid = chunk_len - db_local;
        if (valid > WN) valid = WN;

        const half* __restrict__ db_tile_base =
            db_half + (long long)(db_tile_start + tile) * PACKED_TILE_STRIDE;
        const int4* __restrict__ db_tile_base_vec =
            reinterpret_cast<const int4*>(db_tile_base);

        #pragma unroll
        for (int i = tid; i < TILE_VEC4; i += THREADS_PER_BLOCK) {
            db_tile_vec[i] = db_tile_base_vec[i];
        }
        if (tid < WN) {
            db_norm_tile[tid] = (tid < valid) ? database_norms[chunk_start + db_local + tid] : 0.0f;
        }
        __syncthreads();

        if (warp_active) {
            const half* __restrict__ db_tile_shared =
                reinterpret_cast<const half*>(db_tile_vec);

            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k = 0; k < KTILES; ++k) {
                wmma::load_matrix_sync(b, db_tile_shared + (long long)k * TILE_ELEMS, WN);
                wmma::mma_sync(acc, a[k], b, acc);
            }

            wmma::store_matrix_sync(dots, acc, WN, wmma::mem_row_major);
        }
        __syncthreads();

        if (warp_active && lane < WM && q < num_queries) {
            const float* __restrict__ dot_row = dots + lane * WN;
            #pragma unroll
            for (int j = 0; j < WN; ++j) {
                if (j < valid) {
                    int db_idx = chunk_start + db_local + j;
                    float dist = q_norm + db_norm_tile[j] - 2.0f * dot_row[j];
                    if (dist < best_dist || (dist == best_dist && db_idx < best_idx)) {
                        best_dist = dist;
                        best_idx = db_idx;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (warp_active && lane < WM && q < num_queries) {
        long long out = (long long)chunk_id * query_stride + q;
        chunk_best_dist[out] = best_dist;
        chunk_best_idx[out] = best_idx;
    }
}

extern "C" __global__ void there_v9_reduce_chunk_bests(
    const float* __restrict__ chunk_best_dist,
    const int* __restrict__ chunk_best_idx,
    float* __restrict__ best_dist,
    int* __restrict__ best_idx,
    int num_queries,
    int num_chunks,
    int query_stride
) {
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= num_queries) return;

    float best = FLT_MAX;
    int idx = -1;
    for (int c = 0; c < num_chunks; ++c) {
        long long off = (long long)c * query_stride + q;
        float cand = chunk_best_dist[off];
        int cand_idx = chunk_best_idx[off];
        if (cand_idx >= 0 && (cand < best || (cand == best && (idx < 0 || cand_idx < idx)))) {
            best = cand;
            idx = cand_idx;
        }
    }

    best_dist[q] = best;
    best_idx[q] = idx;
}
