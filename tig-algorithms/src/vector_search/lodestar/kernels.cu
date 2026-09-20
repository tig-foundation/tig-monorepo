// lodestar kernels - WMMA fp16 chunked distance scan for exact nearest-neighbour search.

#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

__device__ __forceinline__ void lds_write_tailpad(unsigned short* __restrict__ d_row,
                                                  int ktile_stride, float value) {
    const __half hi = __float2half(value);
    const __half lo = __float2half(value - __half2float(hi));
    d_row[15 * ktile_stride + 10] = __half_as_ushort(hi);
    d_row[15 * ktile_stride + 11] = __half_as_ushort(lo);
}

#define AVPF_ROWS 128
#define LDS_KTILE_STRIDE dst_stride
#define AVPF_TILE 32

extern "C" __global__ __launch_bounds__(AVPF_ROWS) void lds_pack_norm_fast(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    float* __restrict__ norms,
    int rows, int src_stride, int dst_stride, int tail_mode)
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
    if (active) {
        norms[row] = sum;
        lds_write_tailpad(d_row, LDS_KTILE_STRIDE, (tail_mode != 0) ? (-0.5f * sum) : 1.0f);
    }
}

__device__ __forceinline__ unsigned long long lds_pack_key(float dist, int idx) {
    unsigned int b = __float_as_uint(dist);
    b ^= (unsigned int)(((int)b) >> 31) | 0x80000000u;
    return ((unsigned long long)b << 32) | (unsigned long long)(unsigned int)idx;
}
__device__ __forceinline__ float lds_key_dist(unsigned long long key) {
    unsigned int b = (unsigned int)(key >> 32);
    b ^= ((b >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(b);
}

template <int WARP_TILES>
__device__ __forceinline__ void lds_top2_f16acc_impl(
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

    unsigned long long best_key = lds_pack_key(FLT_MAX, -1);
    unsigned long long scnd_key = lds_pack_key(FLT_MAX, -1);
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
                    const float dist = -__half2float(dot_row[j]);
                    const unsigned long long kk = lds_pack_key(dist, db_idx);
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
        c_d1[out] = lds_key_dist(m1); c_i1[out] = (int)(unsigned int)(m1 & 0xFFFFFFFFull);
        c_d2[out] = lds_key_dist(m2); c_i2[out] = (int)(unsigned int)(m2 & 0xFFFFFFFFull);
    }
}

extern "C" __global__ void lds_top2_f16acc(
    const unsigned short* __restrict__ q_half_u16,
    const unsigned short* __restrict__ db_half_u16,
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    float* __restrict__ c_d1, int* __restrict__ c_i1,
    float* __restrict__ c_d2, int* __restrict__ c_i2,
    int num_queries, int database_size, int query_stride, int chunk_db)
{
    lds_top2_f16acc_impl<4>(q_half_u16, db_half_u16, query_norms, database_norms,
                            c_d1, c_i1, c_d2, c_i2,
                            num_queries, database_size, query_stride, chunk_db);
}

extern "C" __global__ void lds_reduce_filter_rerank(
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

#undef LDS_KTILE_STRIDE
#define LDS_KTILE_STRIDE (2 * dst_stride)
#define LDSQ_ROWS 128
#define LDSQ_TILE 32

extern "C" __global__ __launch_bounds__(LDSQ_ROWS) void lds_pack_norm_fast_q32(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    float* __restrict__ norms,
    int rows, int src_stride, int dst_stride, int tail_mode)
{
    __shared__ float sh[LDSQ_ROWS * (LDSQ_TILE + 1)];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = LDSQ_ROWS / 32;
    const int row_base = (int)blockIdx.x * LDSQ_ROWS;
    const int rows_here = min(LDSQ_ROWS, rows - row_base);
    if (rows_here <= 0) return;
    const int row = row_base + tid;
    const bool active = (tid < rows_here);

    const int row_tile = row >> 5;
    const int row_in_tile = row & 31;
    unsigned short* __restrict__ d_row =
        dst + (long long)row_tile * (32ll * dst_stride) + (long long)row_in_tile * 16ll;

    float sum = 0.0f;
    for (int t = 0; t < 256; t += LDSQ_TILE) {
        const int tlen = min(LDSQ_TILE, 250 - t);
        if (tlen <= 0) break;
        __syncthreads();
        for (int r = warp; r < rows_here; r += nwarps) {
            const float* __restrict__ s = src + (long long)(row_base + r) * src_stride + t;
            if (lane < tlen) sh[r * (LDSQ_TILE + 1) + lane] = s[lane];
        }
        __syncthreads();
        if (active) {
            const float* __restrict__ p = &sh[tid * (LDSQ_TILE + 1)];
            if (tlen == LDSQ_TILE) {
                #pragma unroll
                for (int g = 0; g < LDSQ_TILE / 16; ++g) {
                    __align__(16) unsigned short b[16];
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        float x = p[g * 16 + j];
                        sum = fmaf(x, x, sum);
                        b[j] = __half_as_ushort(__float2half(x));
                    }
                    unsigned short* __restrict__ o = d_row + ((t >> 4) + g) * 512;
                    reinterpret_cast<uint4*>(o)[0] = reinterpret_cast<const uint4*>(b)[0];
                    reinterpret_cast<uint4*>(o)[1] = reinterpret_cast<const uint4*>(b)[1];
                }
            } else {
                #pragma unroll 1
                for (int j = 0; j < tlen; ++j) {
                    float x = p[j];
                    sum = fmaf(x, x, sum);
                    int d = t + j;
                    d_row[(d >> 4) * 512 + (d & 15)] = __half_as_ushort(__float2half(x));
                }
            }
        }
    }
    if (active) {
        norms[row] = sum;
        lds_write_tailpad(d_row, LDS_KTILE_STRIDE, (tail_mode != 0) ? (-0.5f * sum) : 1.0f);
    }
}

#define LDS_G328_MAX_WARPS 8

extern "C" __global__ void lds_top2_g328(
    const unsigned short* __restrict__ q_half_u16,
    const unsigned short* __restrict__ db_half_u16,
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    float* __restrict__ c_d1, int* __restrict__ c_i1,
    float* __restrict__ c_d2, int* __restrict__ c_i2,
    int num_queries, int database_size, int query_stride, int chunk_db)
{
    constexpr int WM = 32, WN = 8, WK = 16, DPAD = 256;
    constexpr int KTILES    = DPAD / WK;
    constexpr int QROWTILE  = 32 * DPAD;
    constexpr int QSUB      = 32 * WK;
    constexpr int DBROWTILE = 16 * DPAD;
    constexpr int DBSUB     = 16 * WK;
    constexpr int SH_VEC4   = (KTILES * 16 * WK) / 8;

    const int chunk_id = (int)blockIdx.y;
    const int chunk_start = chunk_id * chunk_db;
    int chunk_len = database_size - chunk_start;
    if (chunk_len <= 0) return;
    if (chunk_len > chunk_db) chunk_len = chunk_db;

    const int lane = (int)threadIdx.x;
    const int warp_id = (int)threadIdx.y;
    const int nwarps = (int)blockDim.y;
    const int tid = warp_id * 32 + lane;
    const int threads = 32 * nwarps;

    const int q_tile = (int)blockIdx.x * nwarps + warp_id;
    const int q_base = q_tile * WM;
    const bool warp_active = q_base < query_stride;

    __shared__ __align__(16) int4 db_sh[SH_VEC4];
    __shared__ __half dot_tiles[LDS_G328_MAX_WARPS][2][WM * WN];
    __half* __restrict__ dots0 = dot_tiles[warp_id][0];
    __half* __restrict__ dots1 = dot_tiles[warp_id][1];

    const half* __restrict__ q_half  = reinterpret_cast<const half*>(q_half_u16);
    const half* __restrict__ db_half = reinterpret_cast<const half*>(db_half_u16);

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[KTILES];
    if (warp_active) {
        const half* __restrict__ qb = q_half + (long long)q_tile * QROWTILE;
        #pragma unroll
        for (int k = 0; k < KTILES; ++k)
            wmma::load_matrix_sync(a[k], qb + (long long)k * QSUB, WK);
    }

    unsigned long long best_key = lds_pack_key(FLT_MAX, -1);
    unsigned long long scnd_key = lds_pack_key(FLT_MAX, -1);

    const int q_row = q_base + lane;
    (void)query_norms;

    const int rt_start = chunk_start >> 4;
    const int n_tiles  = (chunk_len + 15) / 16;
    const int full16   = chunk_len / 16;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int db_local = tile * 16;
        const bool full = (tile < full16);
        int valid = chunk_len - db_local; if (valid > 16) valid = 16;

        const int4* __restrict__ s0 =
            reinterpret_cast<const int4*>(db_half + (long long)(rt_start + tile) * DBROWTILE);
        for (int i = tid; i < SH_VEC4; i += threads) db_sh[i] = s0[i];
        __syncthreads();

        if (warp_active) {
            const half* __restrict__ dbs = reinterpret_cast<const half*>(db_sh);
            {
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b0, b1;
                wmma::fragment<wmma::accumulator, WM, WN, WK, __half> acc0, acc1;
                wmma::fill_fragment(acc0, __float2half(0.0f));
                wmma::fill_fragment(acc1, __float2half(0.0f));
                #pragma unroll
                for (int k = 0; k < KTILES; ++k) {
                    const half* __restrict__ kb = dbs + (long long)k * DBSUB;
                    wmma::load_matrix_sync(b0, kb, WK);
                    wmma::load_matrix_sync(b1, kb + (WN * WK), WK);
                    wmma::mma_sync(acc0, a[k], b0, acc0);
                    wmma::mma_sync(acc1, a[k], b1, acc1);
                }
                wmma::store_matrix_sync(dots0, acc0, WN, wmma::mem_row_major);
                wmma::store_matrix_sync(dots1, acc1, WN, wmma::mem_row_major);
                __syncwarp();
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    const __half* __restrict__ dot_row =
                        (sub == 0 ? dots0 : dots1) + lane * WN;
                    #pragma unroll
                    for (int t = 0; t < WN; ++t) {
                        const int j = sub * WN + t;
                        if (full || j < valid) {
                            const int db_idx = chunk_start + db_local + j;
                            const float dist = -__half2float(dot_row[t]);
                            const unsigned long long kk = lds_pack_key(dist, db_idx);
                            if (kk < best_key)      { scnd_key = best_key; best_key = kk; }
                            else if (kk < scnd_key) { scnd_key = kk; }
                        }
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    if (warp_active && q_row < num_queries) {
        const long long out = (long long)chunk_id * query_stride + q_row;
        c_d1[out] = lds_key_dist(best_key); c_i1[out] = (int)(unsigned int)(best_key & 0xFFFFFFFFull);
        c_d2[out] = lds_key_dist(scnd_key); c_i2[out] = (int)(unsigned int)(scnd_key & 0xFFFFFFFFull);
    }
}
