// helix_search kernels — WMMA fp16 chunked distance scan.
// Per chunk: pack/normalise vectors to half2, Tensor-Core GEMM for squared
// distances, keep top candidates in fp16, then exact fp32 re-rank + filter.

#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void hlx_pack_norm_fp16_250_to_256(
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

    const unsigned long long src_addr = (unsigned long long)src_row;
    const unsigned long long dst_addr = (unsigned long long)dst_base;
    const bool vec16 = ((src_addr & 0xFULL) == 0ull) && ((dst_addr & 0x3ull) == 0ull);
    const bool vec8 = ((src_addr & 0x7ull) == 0ull) && ((dst_addr & 0x3ull) == 0ull);

    const float* __restrict__ s_block = src_row;
    unsigned short* __restrict__ d_block = dst_base;

    if (vec16) {
        #pragma unroll
        for (int kt = 0; kt < 15; ++kt) {
            const float4* __restrict__ s4 = reinterpret_cast<const float4*>(s_block);
            float4 v0 = s4[0];
            float4 v1 = s4[1];
            float4 v2 = s4[2];
            float4 v3 = s4[3];

            sum = fmaf(v0.x, v0.x, sum);
            sum = fmaf(v0.y, v0.y, sum);
            sum = fmaf(v0.z, v0.z, sum);
            sum = fmaf(v0.w, v0.w, sum);
            sum = fmaf(v1.x, v1.x, sum);
            sum = fmaf(v1.y, v1.y, sum);
            sum = fmaf(v1.z, v1.z, sum);
            sum = fmaf(v1.w, v1.w, sum);
            sum = fmaf(v2.x, v2.x, sum);
            sum = fmaf(v2.y, v2.y, sum);
            sum = fmaf(v2.z, v2.z, sum);
            sum = fmaf(v2.w, v2.w, sum);
            sum = fmaf(v3.x, v3.x, sum);
            sum = fmaf(v3.y, v3.y, sum);
            sum = fmaf(v3.z, v3.z, sum);
            sum = fmaf(v3.w, v3.w, sum);

            half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);
            d2[0] = __floats2half2_rn(v0.x, v0.y);
            d2[1] = __floats2half2_rn(v0.z, v0.w);
            d2[2] = __floats2half2_rn(v1.x, v1.y);
            d2[3] = __floats2half2_rn(v1.z, v1.w);
            d2[4] = __floats2half2_rn(v2.x, v2.y);
            d2[5] = __floats2half2_rn(v2.z, v2.w);
            d2[6] = __floats2half2_rn(v3.x, v3.y);
            d2[7] = __floats2half2_rn(v3.z, v3.w);

            s_block += 16;
            d_block += 256;
        }

        const float4* __restrict__ s4 = reinterpret_cast<const float4*>(s_block);
        float4 v0 = s4[0];
        float4 v1 = s4[1];
        float2 v2 = *reinterpret_cast<const float2*>(s_block + 8);

        sum = fmaf(v0.x, v0.x, sum);
        sum = fmaf(v0.y, v0.y, sum);
        sum = fmaf(v0.z, v0.z, sum);
        sum = fmaf(v0.w, v0.w, sum);
        sum = fmaf(v1.x, v1.x, sum);
        sum = fmaf(v1.y, v1.y, sum);
        sum = fmaf(v1.z, v1.z, sum);
        sum = fmaf(v1.w, v1.w, sum);
        sum = fmaf(v2.x, v2.x, sum);
        sum = fmaf(v2.y, v2.y, sum);

        half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);
        d2[0] = __floats2half2_rn(v0.x, v0.y);
        d2[1] = __floats2half2_rn(v0.z, v0.w);
        d2[2] = __floats2half2_rn(v1.x, v1.y);
        d2[3] = __floats2half2_rn(v1.z, v1.w);
        d2[4] = __floats2half2_rn(v2.x, v2.y);
    } else if (vec8) {
        #pragma unroll
        for (int kt = 0; kt < 15; ++kt) {
            const float2* __restrict__ s2 = reinterpret_cast<const float2*>(s_block);
            half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);

            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 v = s2[p];
                sum = fmaf(v.x, v.x, sum);
                sum = fmaf(v.y, v.y, sum);
                d2[p] = __floats2half2_rn(v.x, v.y);
            }

            s_block += 16;
            d_block += 256;
        }

        const float2* __restrict__ s2 = reinterpret_cast<const float2*>(s_block);
        half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);

        #pragma unroll
        for (int p = 0; p < 5; ++p) {
            float2 v = s2[p];
            sum = fmaf(v.x, v.x, sum);
            sum = fmaf(v.y, v.y, sum);
            d2[p] = __floats2half2_rn(v.x, v.y);
        }
    } else {
        #pragma unroll
        for (int kt = 0; kt < 15; ++kt) {
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float x = s_block[c];
                sum = fmaf(x, x, sum);
                d_block[c] = __half_as_ushort(__float2half(x));
            }
            s_block += 16;
            d_block += 256;
        }

        #pragma unroll
        for (int c = 0; c < 10; ++c) {
            float x = s_block[c];
            sum = fmaf(x, x, sum);
            d_block[c] = __half_as_ushort(__float2half(x));
        }
    }

    norms[row] = sum;
}

#define AVPF_ROWS 128
#define AVPF_TILE 32

extern "C" __global__ __launch_bounds__(AVPF_ROWS) void hlx_pack_norm_fast(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    float* __restrict__ norms,
    int rows, int src_stride, int dst_stride)
{
    __shared__ float sh[AVPF_ROWS * (AVPF_TILE + 1)];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = AVPF_ROWS / 32;
    const int row_base = (int)blockIdx.x * AVPF_ROWS;
    const int rows_here = min(AVPF_ROWS, rows - row_base);
    if (rows_here <= 0) return;

    const int row = row_base + tid;
    const bool active = (tid < rows_here);

    int row_tile = row >> 4;
    int row_in_tile = row & 15;
    unsigned short* __restrict__ d_row =
        dst + (long long)row_tile * (16ll * dst_stride) + (long long)row_in_tile * 16ll;

    float sum = 0.0f;

    for (int t = 0; t < 256; t += AVPF_TILE) {
        const int tlen = min(AVPF_TILE, 250 - t);
        if (tlen <= 0) break;
        __syncthreads();
        for (int r = warp; r < rows_here; r += nwarps) {
            const float* __restrict__ s = src + (long long)(row_base + r) * src_stride + t;
            if (lane < tlen) sh[r * (AVPF_TILE + 1) + lane] = s[lane];
        }
        __syncthreads();
        if (active) {
            const float* __restrict__ p = &sh[tid * (AVPF_TILE + 1)];
            if (tlen == AVPF_TILE) {
                #pragma unroll
                for (int g = 0; g < AVPF_TILE / 16; ++g) {
                    __align__(16) unsigned short b[16];
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        float x = p[g * 16 + j];
                        sum = fmaf(x, x, sum);
                        b[j] = __half_as_ushort(__float2half(x));
                    }
                    unsigned short* __restrict__ o = d_row + ((t >> 4) + g) * 256;
                    reinterpret_cast<uint4*>(o)[0] = reinterpret_cast<const uint4*>(b)[0];
                    reinterpret_cast<uint4*>(o)[1] = reinterpret_cast<const uint4*>(b)[1];
                }
            } else {
                #pragma unroll 1
                for (int j = 0; j < tlen; ++j) {
                    float x = p[j];
                    sum = fmaf(x, x, sum);
                    int d = t + j;
                    d_row[(d >> 4) * 256 + (d & 15)] = __half_as_ushort(__float2half(x));
                }
            }
        }
    }
    if (active) norms[row] = sum;
}

__device__ __forceinline__ unsigned long long hlx_pack_key(float dist, int idx) {
    unsigned int b = __float_as_uint(dist);
    b ^= (unsigned int)(((int)b) >> 31) | 0x80000000u;
    return ((unsigned long long)b << 32) | (unsigned long long)(unsigned int)idx;
}
__device__ __forceinline__ float hlx_key_dist(unsigned long long key) {
    unsigned int b = (unsigned int)(key >> 32);
    b ^= ((b >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(b);
}

template <int WARP_TILES>
__device__ __forceinline__ void hlx_wmma_chunk_best_impl(
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

    unsigned long long best_key = hlx_pack_key(FLT_MAX, -1);
    float q_norm = (warp_active && lane < WM && q < num_queries) ? query_norms[q] : 0.0f;

    int db_tile_start = chunk_start >> 4;
    int db_tiles = (chunk_len + WN - 1) / WN;

    const int full_tiles = chunk_len / WN;
    for (int tile = 0; tile < full_tiles; ++tile) {
        int db_local = tile * WN;

        const half* __restrict__ db_tile_base =
            db_half + (long long)(db_tile_start + tile) * PACKED_TILE_STRIDE;
        const int4* __restrict__ db_tile_base_vec =
            reinterpret_cast<const int4*>(db_tile_base);

        #pragma unroll
        for (int i = tid; i < TILE_VEC4; i += THREADS_PER_BLOCK) {
            db_tile_vec[i] = db_tile_base_vec[i];
        }
        if (tid < WN) {
            db_norm_tile[tid] = database_norms[chunk_start + db_local + tid];
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
        __syncwarp();

        if (warp_active) {
            const int row  = lane & (WM - 1);
            const int half = lane >> 4;
            const int q_row = q_base + row;
            const float q_norm_row = __shfl_sync(0xFFFFFFFFu, q_norm, row);
            const float* __restrict__ dot_row = dots + row * WN;

            unsigned long long k = hlx_pack_key(FLT_MAX, -1);
            #pragma unroll
            for (int t = 0; t < WN / 2; ++t) {
                const int j = half * (WN / 2) + t;
                const int db_idx = chunk_start + db_local + j;
                const float dist = q_norm_row + db_norm_tile[j] - 2.0f * dot_row[j];
                k = min(k, hlx_pack_key(dist, db_idx));
            }
            const unsigned long long other = __shfl_down_sync(0xFFFFFFFFu, k, WM);
            if (lane < WM && q_row < num_queries) {
                best_key = min(best_key, min(k, other));
            }
        }
        __syncthreads();
    }

    for (int tile = full_tiles; tile < db_tiles; ++tile) {
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
        __syncwarp();

        if (warp_active && lane < WM && q < num_queries) {
            const float* __restrict__ dot_row = dots + lane * WN;
            #pragma unroll
            for (int j = 0; j < WN; ++j) {
                if (j < valid) {
                    int db_idx = chunk_start + db_local + j;
                    float dist = q_norm + db_norm_tile[j] - 2.0f * dot_row[j];
                    best_key = min(best_key, hlx_pack_key(dist, db_idx));
                }
            }
        }
        __syncthreads();
    }

    if (warp_active && lane < WM && q < num_queries) {
        long long out = (long long)chunk_id * query_stride + q;
        chunk_best_dist[out] = hlx_key_dist(best_key);
        chunk_best_idx[out] = (int)(unsigned int)(best_key & 0xFFFFFFFFull);
    }
}

extern "C" __global__ void hlx_wmma_chunk_best(
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
    hlx_wmma_chunk_best_impl<4>(q_half_u16, db_half_u16, query_norms, database_norms, chunk_best_dist, chunk_best_idx, num_queries, database_size, query_stride, chunk_db);
}

extern "C" __global__ void hlx_wmma_chunk_best_w2(
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
    hlx_wmma_chunk_best_impl<2>(q_half_u16, db_half_u16, query_norms, database_norms, chunk_best_dist, chunk_best_idx, num_queries, database_size, query_stride, chunk_db);
}

extern "C" __global__ void hlx_wmma_chunk_best_w8(
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
    hlx_wmma_chunk_best_impl<8>(q_half_u16, db_half_u16, query_norms, database_norms, chunk_best_dist, chunk_best_idx, num_queries, database_size, query_stride, chunk_db);
}

extern "C" __global__ void hlx_reduce_chunk_bests(
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

template <int WARP_TILES>
__device__ __forceinline__ void hlx_top2_f16acc_impl(
    const unsigned short* __restrict__ q_half_u16,
    const unsigned short* __restrict__ db_half_u16,
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    float* __restrict__ c_d1, int* __restrict__ c_i1,
    float* __restrict__ c_d2, int* __restrict__ c_i2,
    int num_queries, int database_size, int query_stride, int chunk_db)
{
    constexpr int WM = 16, WN = 16, WK = 16, DPAD = 256;
    constexpr int KTILES = DPAD / WK;
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

    __shared__ __half dot_tiles[WARP_TILES][WM * WN];
    __shared__ __align__(16) int4 db_tile_vec[TILE_VEC4];
    __shared__ float db_norm_tile[WN];
    __half* __restrict__ dots = dot_tiles[warp_id];

    const half* __restrict__ q_half = reinterpret_cast<const half*>(q_half_u16);
    const half* __restrict__ db_half = reinterpret_cast<const half*>(db_half_u16);

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[KTILES];
    if (warp_active) {
        const half* __restrict__ qb = q_half + (long long)q_tile_id * PACKED_TILE_STRIDE;
        #pragma unroll
        for (int k = 0; k < KTILES; ++k)
            wmma::load_matrix_sync(a[k], qb + (long long)k * TILE_ELEMS, WN);
    }

    unsigned long long best_key = hlx_pack_key(FLT_MAX, -1);
    unsigned long long scnd_key = hlx_pack_key(FLT_MAX, -1);
    float q_norm = (warp_active && lane < WM && q < num_queries) ? query_norms[q] : 0.0f;

    const int row   = lane & (WM - 1);
    const int hlf   = lane >> 4;
    const int q_row = q_base + row;
    const float q_norm_row = __shfl_sync(0xFFFFFFFFu, q_norm, row);

    int db_tile_start = chunk_start >> 4;
    int db_tiles = (chunk_len + WN - 1) / WN;
    const int full_tiles = chunk_len / WN;

    for (int tile = 0; tile < db_tiles; ++tile) {
        int db_local = tile * WN;
        const bool full = (tile < full_tiles);
        int valid = chunk_len - db_local; if (valid > WN) valid = WN;

        const half* __restrict__ dbb = db_half + (long long)(db_tile_start + tile) * PACKED_TILE_STRIDE;
        const int4* __restrict__ dbv4 = reinterpret_cast<const int4*>(dbb);
        #pragma unroll
        for (int i = tid; i < TILE_VEC4; i += THREADS_PER_BLOCK) db_tile_vec[i] = dbv4[i];
        if (tid < WN)
            db_norm_tile[tid] = (full || tid < valid) ? database_norms[chunk_start + db_local + tid] : 0.0f;
        __syncthreads();

        if (warp_active) {
            const half* __restrict__ dbs = reinterpret_cast<const half*>(db_tile_vec);
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, WM, WN, WK, __half> acc;
            wmma::fill_fragment(acc, __float2half(0.0f));
            #pragma unroll
            for (int k = 0; k < KTILES; ++k) {
                wmma::load_matrix_sync(b, dbs + (long long)k * TILE_ELEMS, WN);
                wmma::mma_sync(acc, a[k], b, acc);
            }
            wmma::store_matrix_sync(dots, acc, WN, wmma::mem_row_major);
        }
        __syncwarp();

        if (warp_active) {
            const __half* __restrict__ dot_row = dots + row * WN;
            #pragma unroll
            for (int t = 0; t < WN / 2; ++t) {
                const int j = hlf * (WN / 2) + t;
                if (full || j < valid) {
                    const int db_idx = chunk_start + db_local + j;
                    const float dist = q_norm_row + db_norm_tile[j] - 2.0f * __half2float(dot_row[j]);
                    const unsigned long long kk = hlx_pack_key(dist, db_idx);
                    if (kk < best_key)      { scnd_key = best_key; best_key = kk; }
                    else if (kk < scnd_key) { scnd_key = kk; }
                }
            }
        }
        __syncthreads();
    }

    const unsigned long long o1 = __shfl_down_sync(0xFFFFFFFFu, best_key, WM);
    const unsigned long long o2 = __shfl_down_sync(0xFFFFFFFFu, scnd_key, WM);
    if (warp_active && lane < WM && q_row < num_queries) {
        unsigned long long m1, m2;
        if (best_key < o1) { m1 = best_key; m2 = (scnd_key < o1) ? scnd_key : o1; }
        else               { m1 = o1;       m2 = (o2 < best_key) ? o2 : best_key; }
        long long out = (long long)chunk_id * query_stride + q_row;
        c_d1[out] = hlx_key_dist(m1); c_i1[out] = (int)(unsigned int)(m1 & 0xFFFFFFFFull);
        c_d2[out] = hlx_key_dist(m2); c_i2[out] = (int)(unsigned int)(m2 & 0xFFFFFFFFull);
    }
}

#define HLX_TOP2_ENTRY(NAME, W)                                                     \
extern "C" __global__ void NAME(                                                    \
    const unsigned short* __restrict__ q_half_u16,                                  \
    const unsigned short* __restrict__ db_half_u16,                                 \
    const float* __restrict__ query_norms,                                          \
    const float* __restrict__ database_norms,                                       \
    float* __restrict__ c_d1, int* __restrict__ c_i1,                               \
    float* __restrict__ c_d2, int* __restrict__ c_i2,                               \
    int num_queries, int database_size, int query_stride, int chunk_db)             \
{                                                                                   \
    hlx_top2_f16acc_impl<W>(q_half_u16, db_half_u16, query_norms, database_norms,   \
                            c_d1, c_i1, c_d2, c_i2,                                 \
                            num_queries, database_size, query_stride, chunk_db);    \
}
HLX_TOP2_ENTRY(hlx_top2_f16acc,    4)
HLX_TOP2_ENTRY(hlx_top2_f16acc_w2, 2)
HLX_TOP2_ENTRY(hlx_top2_f16acc_w8, 8)
#undef HLX_TOP2_ENTRY

extern "C" __global__ void hlx_reduce_filter_rerank(
    const float* __restrict__ c_d1, const int* __restrict__ c_i1,
    const float* __restrict__ c_d2, const int* __restrict__ c_i2,
    const float* __restrict__ qv, const float* __restrict__ dbv,
    int num_queries, int num_chunks, int query_stride, int dims, int dbsz,
    float delta, int* __restrict__ out)
{
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= num_queries) return;

    float dmin = FLT_MAX;
    for (int c = 0; c < num_chunks; ++c) {
        long long off = (long long)c * query_stride + q;
        float a = c_d1[off];
        if (c_i1[off] >= 0 && a < dmin) dmin = a;
    }
    const float thr = dmin + delta;

    const float* __restrict__ qa = qv + (long long)q * dims;
    float best = 3.402823466e+38f; int bi = -1;
    for (int c = 0; c < num_chunks; ++c) {
        long long off = (long long)c * query_stride + q;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int   idx = (s == 0) ? c_i1[off] : c_i2[off];
            float df  = (s == 0) ? c_d1[off] : c_d2[off];
            if (idx < 0 || idx >= dbsz) continue;
            if (df > thr) continue;
            const float* __restrict__ bb = dbv + (long long)idx * dims;
            float sum = 0.0f;
            for (int j = 0; j < dims; ++j) { float t = qa[j] - bb[j]; sum += t * t; }
            if (sum < best || (sum == best && idx < bi)) { best = sum; bi = idx; }
        }
    }
    out[q] = bi;
}
