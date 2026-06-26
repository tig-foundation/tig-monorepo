
#include <float.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define DIMS            250
#define DIMS_PAD        256
#define RABIT_WORDS     8
#define RABIT_THRESH    200

extern "C" __global__ void convert_fp32_to_fp16_padded_t18(
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
    for (int d = 0; d < DIMS; d++)
        row_out[d] = __half_as_ushort(__float2half(row_in[d]));
    #pragma unroll
    for (int d = DIMS; d < DIMS_PAD; d++)
        row_out[d] = 0;
}

extern "C" __global__ void compute_norms_fp16_t18(
    const half* __restrict__ d_vecs,
    float*      __restrict__ d_norms,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const half* v = d_vecs + (long long)i * DIMS_PAD;
    float norm = 0.0f;
    #pragma unroll 10
    for (int d = 0; d < DIMS; d++) {
        float x = __half2float(v[d]);
        norm = __fmaf_rn(x, x, norm);
    }
    d_norms[i] = norm;
}

extern "C" __global__ void encode_rabitq_flat_t18(
    const float* __restrict__ vecs,
    uint32_t*    __restrict__ codes,
    int count, int dims
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float* row = vecs + (long long)i * dims;
    uint32_t*    cod = codes + (long long)i * RABIT_WORDS;
    #pragma unroll
    for (int w = 0; w < RABIT_WORDS; w++) {
        uint32_t bword = 0u;
        int base = w * 32;
        #pragma unroll
        for (int b = 0; b < 32; b++) {
            int d = base + b;
            if (d < dims && row[d] > 0.0f) bword |= (1u << b);
        }
        cod[w] = bword;
    }
}

extern "C" __global__ void bruteforce_gemm_argmin_t18(
    const half*  __restrict__ d_q_fp16,
    const half*  __restrict__ d_db_fp16,
    const float* __restrict__ d_norms_q,
    const float* __restrict__ d_norms_db,
    const uint32_t* __restrict__ d_q_rabitq,
    const uint32_t* __restrict__ d_db_rabitq,
    int*         __restrict__ d_best_idx,
    int nq, int db_size
) {
    const int WMMA_M  = 8;
    const int WMMA_N  = 32;
    const int WMMA_K  = 16;
    const int K_TILES = DIMS_PAD / WMMA_K;
    const int N_WARPS = 16;
    const int N_BLOCKS = 82;
    const int N_WARPS_TOTAL = N_BLOCKS * N_WARPS;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    __shared__ half  s_b  [K_TILES * WMMA_K * WMMA_N];
    __shared__ float s_dot[N_WARPS * WMMA_M * WMMA_N];
    __shared__ int   s_process;
    
    __shared__ uint32_t s_rabitq[RABIT_WORDS * WMMA_N];

    const int n_q_batches = (nq + WMMA_M - 1) / WMMA_M;
    const int extra       = n_q_batches - N_WARPS_TOTAL;
    const int n_db_tiles  = (db_size + WMMA_N - 1) / WMMA_N;

    const int block_extra_start = (int)blockIdx.x * N_WARPS;
    const int block_n_passes    = (extra > 0 && block_extra_start < extra) ? 2 : 1;

    for (int pass = 0; pass < block_n_passes; pass++) {

        int warp_gid = (pass == 0)
            ? (int)(blockIdx.x * N_WARPS + warp_id)
            : (N_WARPS_TOTAL + (int)(blockIdx.x * N_WARPS + warp_id));
        int q_base = warp_gid * WMMA_M;
        int active = (q_base < nq);

        
        uint32_t q0_rabitq[RABIT_WORDS];
        if (active && lane == 0) {
            const uint32_t* q_rc = d_q_rabitq + (long long)q_base * RABIT_WORDS;
            #pragma unroll
            for (int w = 0; w < RABIT_WORDS; w++)
                q0_rabitq[w] = q_rc[w];
        }
        
        #pragma unroll
        for (int w = 0; w < RABIT_WORDS; w++)
            q0_rabitq[w] = __shfl_sync(0xffffffff, q0_rabitq[w], 0);

        
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frags[K_TILES];
        if (active) {
            const half* q_ptr = d_q_fp16 + (long long)q_base * DIMS_PAD;
            #pragma unroll
            for (int k = 0; k < K_TILES; k++)
                wmma::load_matrix_sync(a_frags[k], q_ptr + k * WMMA_K, DIMS_PAD);
        }

        float my_norm_q    = (active && lane < WMMA_M && q_base + lane < nq)
                              ? d_norms_q[q_base + lane] : 0.0f;
        float my_best_dist = FLT_MAX;
        int   my_best_j    = 0;

        
        for (int dt = 0; dt < n_db_tiles; dt++) {
            int db0 = dt * WMMA_N;
            int actual_tile_len = (db0 + WMMA_N <= db_size) ? WMMA_N : (db_size - db0);

            
            for (int idx = threadIdx.x; idx < RABIT_WORDS * WMMA_N; idx += 512) {
                int db_local = idx % WMMA_N;
                int word     = idx / WMMA_N;
                int db_global = db0 + db_local;
                s_rabitq[idx] = d_db_rabitq[(long long)db_global * RABIT_WORDS + word];
            }
            __syncthreads();

            
            s_process = 0;
            int lane_close = 0;

            if (active) {
                for (int db_offset = lane; db_offset < actual_tile_len; db_offset += 32) {
                    int ham = 0;
                    const uint32_t* s_rc = s_rabitq + (long long)db_offset * RABIT_WORDS;
                    #pragma unroll
                    for (int w = 0; w < RABIT_WORDS; w++)
                        ham += __popc(q0_rabitq[w] ^ s_rc[w]);
                    if (ham < RABIT_THRESH) {
                        lane_close = 1;
                        goto skip_scan;
                    }
                }
            }
        skip_scan:

            uint32_t wf = __ballot_sync(0xffffffff, lane_close != 0);
            if (lane == 0 && wf != 0)
                atomicOr(&s_process, 1);
            __syncthreads();

            if (s_process) {
                
                for (int k = 0; k < K_TILES; k++) {
                    half* b_dst = s_b + k * WMMA_K * WMMA_N;
                    int idx = threadIdx.x;
                    b_dst[idx] = d_db_fp16[(long long)(db0 + idx / WMMA_K) * DIMS_PAD
                                            + k * WMMA_K + idx % WMMA_K];
                }
                __syncthreads();

                
                if (active) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
                    wmma::fill_fragment(acc, 0.0f);
                    #pragma unroll
                    for (int k = 0; k < K_TILES; k++) {
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                        wmma::load_matrix_sync(b_frag, s_b + k * WMMA_K * WMMA_N, WMMA_K);
                        wmma::mma_sync(acc, a_frags[k], b_frag, acc);
                    }
                    float* my_sdot = s_dot + warp_id * WMMA_M * WMMA_N;
                    wmma::store_matrix_sync(my_sdot, acc, WMMA_N, wmma::mem_row_major);

                    if (lane < WMMA_M && q_base + lane < nq) {
                        for (int n = 0; n < actual_tile_len; n++) {
                            float dist = -2.0f * my_sdot[lane * WMMA_N + n]
                                         + my_norm_q + d_norms_db[db0 + n];
                            if (dist < my_best_dist) { my_best_dist = dist; my_best_j = db0 + n; }
                        }
                    }
                }
                __syncthreads();
            } else {
                __syncthreads();
            }
        }

        
        if (active && lane < WMMA_M && q_base + lane < nq)
            d_best_idx[q_base + lane] = my_best_j;

    } 
}
