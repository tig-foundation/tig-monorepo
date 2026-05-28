#include <float.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void compute_norms_t17(
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

extern "C" __global__ void convert_fp32_to_fp16_padded_t17(
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

extern "C" __global__ void fused_wmma_chunk_search_t17(
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
    __shared__ half  q_shared[WMMA_M * DPAD];

    int chunk_id    = (int)blockIdx.y;
    int chunk_start = chunk_id * CHUNK_DB;
    int chunk_len   = (chunk_start + CHUNK_DB <= database_size) ? CHUNK_DB : (database_size - chunk_start);
    if (chunk_len <= 0) return;

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

extern "C" __global__ void reduce_chunk_bests_t17(
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
