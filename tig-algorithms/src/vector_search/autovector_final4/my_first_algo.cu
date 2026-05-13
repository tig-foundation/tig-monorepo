

#include <float.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

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

// Fused WMMA L2 search for track T20.
// One warp (32 threads) handles 16 queries and sweeps all DB vectors.
// Eliminates cuBLAS GEMM, intermediate d_dot buffer, and refinement pass.
// Tensor-core dot products accumulated in FP32 → exact L2 distance in one pass.
//
// Grid:  (NQ_PAD/16, 1, 1)    one block = one warp = 16 queries
// Block: (32, 1, 1)
// Shared: 16*16 floats (1 KB) for accumulator staging
//
// Layout:
//   d_q_fp16  [NQ_PAD × DIMS_PAD] row-major (query tile as matrix_a, row_major)
//   d_db_fp16 [DB_PAD × DIMS_PAD] row-major (DB tile accessed as matrix_b, col_major)
//   C[m][n] = dot(q[q_base+m], db[db0+n])  after K_TILES wmma::mma_sync rounds
extern "C" __global__ void fused_wmma_search_t20(
    const unsigned short* __restrict__ d_q_fp16,
    const unsigned short* __restrict__ d_db_fp16,
    const float*          __restrict__ q_norms,
    const float*          __restrict__ db_norms,
    float* __restrict__ best_dist,
    int*   __restrict__ best_idx,
    int num_queries,
    int database_size
) {
    const int WMMA_M  = 16;
    const int WMMA_N  = 16;
    const int WMMA_K  = 16;
    const int DPAD    = 256;
    const int K_TILES = 16;   // DPAD / WMMA_K

    int q_base = (int)blockIdx.x * WMMA_M;
    if (q_base >= num_queries) return;

    int lane = (int)threadIdx.x;
    int my_q = q_base + lane;

    float my_qn   = (lane < WMMA_M && my_q < num_queries) ? q_norms[my_q] : 0.0f;
    float my_best = FLT_MAX;
    int   my_idx  = -1;

    __shared__ float dot_smem[WMMA_M * WMMA_N];

    const half* q_ptr  = (const half*)d_q_fp16  + (long long)q_base * DPAD;
    const half* db_ptr = (const half*)d_db_fp16;

    int n_tiles = (database_size + WMMA_N - 1) / WMMA_N;

    for (int dt = 0; dt < n_tiles; dt++) {
        int db0  = dt * WMMA_N;
        int tlen = (db0 + WMMA_N <= database_size) ? WMMA_N : (database_size - db0);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        #pragma unroll
        for (int k = 0; k < K_TILES; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, q_ptr  + k * WMMA_K,                         DPAD);
            wmma::load_matrix_sync(b_frag, db_ptr + (long long)db0 * DPAD + k * WMMA_K, DPAD);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        wmma::store_matrix_sync(dot_smem, acc, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        if (lane < WMMA_M && my_q < num_queries) {
            for (int n = 0; n < tlen; n++) {
                float dot_val = dot_smem[lane * WMMA_N + n];
                float dist    = my_qn + db_norms[db0 + n] - 2.0f * dot_val;
                if (dist < my_best) {
                    my_best = dist;
                    my_idx  = db0 + n;
                }
            }
        }
    }

    if (lane < WMMA_M && my_q < num_queries) {
        best_dist[my_q] = my_best;
        best_idx[my_q]  = my_idx;
    }
}

// ---------------------------------------------------------------------------
// Two-phase chunked WMMA search for T20 (i26).
// Split DB into chunks so each warp processes a chunk → higher occupancy.
// Phase 1: per-(query,chunk) best stored in chunk arrays.
// Phase 2: reduce across chunks to final per-query best.
// ---------------------------------------------------------------------------

extern "C" __global__ void fused_wmma_chunk_search_t20(
    const unsigned short* __restrict__ d_q_fp16,
    const unsigned short* __restrict__ d_db_fp16,
    const float*          __restrict__ q_norms,
    const float*          __restrict__ db_norms,
    float* __restrict__  chunk_dists,
    int*   __restrict__  chunk_idxs,
    int num_queries,
    int database_size,
    int NQ_PAD,
    int CHUNK_DB
) {
    const int WMMA_M  = 16;
    const int WMMA_N  = 16;
    const int WMMA_K  = 16;
    const int DPAD    = 256;
    const int K_TILES = 16;

    int q_base = (int)blockIdx.x * WMMA_M;
    if (q_base >= num_queries) return;

    int lane = (int)threadIdx.x;
    int my_q = q_base + lane;

    float my_qn = (lane < WMMA_M && my_q < num_queries) ? q_norms[my_q] : 0.0f;
    float best  = FLT_MAX;
    int   best_idx = -1;

    __shared__ float dot_smem[WMMA_M * WMMA_N];
    // <<< shared cache for the entire query tile (16 x DPAD halfs = 8 KB)
    __shared__ half  q_shared[WMMA_M * DPAD];

    int chunk_id    = (int)blockIdx.y;
    int chunk_start = chunk_id * CHUNK_DB;
    int chunk_len   = (chunk_start + CHUNK_DB <= database_size) ? CHUNK_DB : (database_size - chunk_start);
    if (chunk_len <= 0) return;

    // <<< load query tile into shared memory once per chunk
    {
        const half* q_global = (const half*)d_q_fp16 + (long long)q_base * DPAD;
        for (int idx = lane; idx < WMMA_M * DPAD; idx += 32) {
            q_shared[idx] = q_global[idx];
        }
    }
    __syncthreads();

    const half* db_ptr = (const half*)d_db_fp16 + (long long)chunk_start * DPAD;

    int n_tiles = (chunk_len + WMMA_N - 1) / WMMA_N;

    for (int dt = 0; dt < n_tiles; dt++) {
        int db0  = dt * WMMA_N;
        int tlen = (db0 + WMMA_N <= chunk_len) ? WMMA_N : (chunk_len - db0);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        #pragma unroll
        for (int k = 0; k < K_TILES; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            // <<< load query tile from shared memory instead of global
            wmma::load_matrix_sync(a_frag, q_shared + k * WMMA_K, DPAD);
            wmma::load_matrix_sync(b_frag, db_ptr + (long long)db0 * DPAD + k * WMMA_K, DPAD);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        wmma::store_matrix_sync(dot_smem, acc, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        if (lane < WMMA_M && my_q < num_queries) {
            for (int n = 0; n < tlen; n++) {
                float dot_val = dot_smem[lane * WMMA_N + n];
                float dist    = my_qn + db_norms[chunk_start + db0 + n] - 2.0f * dot_val;
                if (dist < best) {
                    best = dist;
                    best_idx = chunk_start + db0 + n;
                }
            }
        }
    }

    if (lane < WMMA_M && my_q < num_queries) {
        chunk_dists[chunk_id * NQ_PAD + my_q] = best;
        chunk_idxs[chunk_id * NQ_PAD + my_q]  = best_idx;
    }
}

extern "C" __global__ void reduce_chunk_bests_t20(
    const float* __restrict__ chunk_dists,
    const int*   __restrict__ chunk_idxs,
    float* __restrict__ best_dist,
    int*   __restrict__ best_idx,
    int num_queries,
    int num_chunks,
    int NQ_PAD
) {
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= num_queries) return;

    float best_d = FLT_MAX;
    int   best_i = -1;

    for (int c = 0; c < num_chunks; c++) {
        float d = chunk_dists[c * NQ_PAD + q];
        int   i = chunk_idxs[c * NQ_PAD + q];
        if (i >= 0 && d < best_d) {
            best_d = d;
            best_i = i;
        }
    }

    best_dist[q] = best_d;
    best_idx[q]  = best_i;
}
