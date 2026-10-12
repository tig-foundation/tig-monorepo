#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

#define V11_DIMS 250
#define V11_DPAD 256

__device__ __forceinline__ unsigned there_v11_pack2(float a, float b) {
    return (unsigned)__half_as_ushort(__float2half(a)) | ((unsigned)__half_as_ushort(__float2half(b)) << 16);
}

extern "C" __global__ void __launch_bounds__(256) there_v11_pack_fold(
    const float* __restrict__ q_src,
    const float* __restrict__ db_src,
    unsigned short* __restrict__ q_dst,
    unsigned short* __restrict__ db_dst,
    int nq,
    int nq_pad,
    int ndb,
    int ndb_pad
) {
    const int lane = (int)(threadIdx.x & 31);
    const int warps_total = (int)((gridDim.x * blockDim.x) >> 5);
    const int total_rows = nq_pad + ndb_pad;
    const int kt = lane >> 1;               
    const int off = (lane & 1) * 8;         

    for (int gw = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5); gw < total_rows; gw += warps_total) {
        const bool is_query = gw < nq_pad;
        const int row = is_query ? gw : gw - nq_pad;
        const int QT = is_query ? 32 : 16;
        const int rows = is_query ? nq : ndb;
        const float* __restrict__ src = is_query ? q_src : db_src;
        uint4* __restrict__ dst4 = reinterpret_cast<uint4*>(is_query ? q_dst : db_dst);

        float v[8];
        if (row >= rows) {
            #pragma unroll
            for (int c = 0; c < 8; ++c) v[c] = 0.0f;
            if (!is_query && lane == 31) v[2] = 30000.0f;   
        } else {
            const float2* __restrict__ s2 = reinterpret_cast<const float2*>(src + (long long)row * V11_DIMS) + lane * 4;
            float2 p0 = s2[0];
            float2 p1 = make_float2(0.0f, 0.0f), p2 = p1, p3 = p1;
            if (lane < 31) { p1 = s2[1]; p2 = s2[2]; p3 = s2[3]; }
            v[0] = p0.x; v[1] = p0.y; v[2] = p1.x; v[3] = p1.y;
            v[4] = p2.x; v[5] = p2.y; v[6] = p3.x; v[7] = p3.y;
            float sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 8; ++c) sum = fmaf(v[c], v[c], sum);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            float half_norm = 0.5f * sum;
            half hi = __float2half(half_norm);
            half lo = __float2half(half_norm - __half2float(hi));
            if (lane == 31) {
                if (is_query) { v[2] = -1.0f; v[3] = -1.0f; v[4] = __half2float(hi); v[5] = __half2float(lo); }
                else          { v[2] = __half2float(hi); v[3] = __half2float(lo); v[4] = -1.0f; v[5] = -1.0f; }
                v[6] = 0.0f; v[7] = 0.0f;
            }
        }

        long long idx4 = (long long)(row / QT) * (QT * V11_DPAD / 8)
                       + (long long)kt * (QT * 2)
                       + (long long)(row % QT) * 2
                       + (off >> 3);
        uint4 pk;
        pk.x = there_v11_pack2(v[0], v[1]);
        pk.y = there_v11_pack2(v[2], v[3]);
        pk.z = there_v11_pack2(v[4], v[5]);
        pk.w = there_v11_pack2(v[6], v[7]);
        dst4[idx4] = pk;
    }
}

__device__ __forceinline__ void there_v11_top2_update(
    float v, int idx, float& s1, int& i1, float& s2, int& i2
) {
    bool gt1 = v > s1;
    bool gt2 = v > s2;
    float ns2 = gt1 ? s1 : (gt2 ? v : s2);
    int   ni2 = gt1 ? i1 : (gt2 ? idx : i2);
    s1 = gt1 ? v : s1;
    i1 = gt1 ? idx : i1;
    s2 = ns2;
    i2 = ni2;
}

extern "C" __global__ void __launch_bounds__(256, 1) there_v11_chunk_top2(
    const unsigned short* __restrict__ q_half_u16,
    const unsigned short* __restrict__ db_half_u16,
    float* __restrict__ out_s1,
    int* __restrict__ out_i1,
    float* __restrict__ out_s2,
    int* __restrict__ out_i2,
    int database_size,
    int query_stride,   
    int chunk_db        
) {
    constexpr int WARPS = 8;
    constexpr int QT = 32;
    constexpr int NT = 32;
    constexpr int KTILES = V11_DPAD / 16;
    constexpr int NSUB = NT / 8;
    constexpr int Q_TILE_HALVES = QT * V11_DPAD;
    constexpr int DB_TILE16_HALVES = 16 * V11_DPAD;
    constexpr int THREADS = WARPS * 32;
    constexpr int NT_INT4 = NT * V11_DPAD / 8;
    constexpr int INT4_PER_THREAD = NT_INT4 / THREADS;

    int chunk_id = (int)blockIdx.y;
    int chunk_start = chunk_id * chunk_db;
    int chunk_len = min(database_size - chunk_start, chunk_db);
    if (chunk_len <= 0) return;

    int lane = (int)(threadIdx.x & 31);
    int warp = (int)(threadIdx.x >> 5);
    int tid = (int)threadIdx.x;
    int q_tile = (int)blockIdx.x * WARPS + warp;
    bool warp_active = q_tile * QT < query_stride;
    bool late = (warp & 4) != 0;   

    __shared__ __align__(128) int4 db_s[2][NT_INT4];
    __shared__ __align__(128) half dot_s[WARPS][QT * NT];

    const half* __restrict__ q_half = reinterpret_cast<const half*>(q_half_u16);
    const half* __restrict__ db_half = reinterpret_cast<const half*>(db_half_u16);

    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a[KTILES];
    if (warp_active) {
        const half* __restrict__ qt = q_half + (long long)q_tile * Q_TILE_HALVES;
        #pragma unroll
        for (int k = 0; k < KTILES; ++k) wmma::load_matrix_sync(a[k], qt + k * (QT * 16), 16);
    }

    float s1 = -3.0e38f, s2 = -3.0e38f;
    int i1 = -1, i2 = -1;

    int steps = (chunk_len + NT - 1) / NT;
    const int4* __restrict__ gsrc =
        reinterpret_cast<const int4*>(db_half + (long long)(chunk_start >> 4) * DB_TILE16_HALVES);

    int4 pre[INT4_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < INT4_PER_THREAD; ++i) pre[i] = gsrc[tid + i * THREADS];
    #pragma unroll
    for (int i = 0; i < INT4_PER_THREAD; ++i) db_s[0][tid + i * THREADS] = pre[i];
    __syncthreads();

    wmma::fragment<wmma::accumulator, 32, 8, 16, half> acc[NSUB];
    int prev_base = -1;

    auto do_mma = [&](int buf) {
        const half* __restrict__ dbs = reinterpret_cast<const half*>(db_s[buf]);
        #pragma unroll
        for (int n = 0; n < NSUB; ++n) wmma::fill_fragment(acc[n], __float2half(0.0f));
        #pragma unroll
        for (int k = 0; k < KTILES; ++k) {
            wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> b;
            #pragma unroll
            for (int n = 0; n < NSUB; ++n) {
                const half* __restrict__ bp = dbs + (n >> 1) * DB_TILE16_HALVES + k * 256 + (n & 1) * 128;
                wmma::load_matrix_sync(b, bp, 16);
                wmma::mma_sync(acc[n], a[k], b, acc[n]);
            }
        }
    };
    auto do_epilogue = [&](int base_idx) {
        #pragma unroll
        for (int n = 0; n < NSUB; ++n) wmma::store_matrix_sync(&dot_s[warp][n * 8], acc[n], NT, wmma::mem_row_major);
        __syncwarp();
        const half2* __restrict__ row = reinterpret_cast<const half2*>(&dot_s[warp][lane * NT]);
        #pragma unroll
        for (int j = 0; j < NT / 2; ++j) {
            float2 v = __half22float2(row[j]);
            int idx = base_idx + 2 * j;
            there_v11_top2_update(v.x, idx, s1, i1, s2, i2);
            there_v11_top2_update(v.y, idx + 1, s1, i1, s2, i2);
        }
        __syncwarp();
    };

    for (int step = 0; step < steps; ++step) {
        int buf = step & 1;
        bool has_next = step + 1 < steps;
        if (has_next) {
            const int4* __restrict__ nsrc = gsrc + (long long)(step + 1) * NT_INT4;
            #pragma unroll
            for (int i = 0; i < INT4_PER_THREAD; ++i) pre[i] = nsrc[tid + i * THREADS];
        }
        int base_idx = chunk_start + step * NT;
        if (warp_active) {
            if (late) {
                if (prev_base >= 0) do_epilogue(prev_base);
                do_mma(buf);
                prev_base = base_idx;
            } else {
                do_mma(buf);
                do_epilogue(base_idx);
            }
        }
        if (has_next) {
            #pragma unroll
            for (int i = 0; i < INT4_PER_THREAD; ++i) db_s[buf ^ 1][tid + i * THREADS] = pre[i];
        }
        __syncthreads();
    }

    if (warp_active) {
        if (late && prev_base >= 0) do_epilogue(prev_base);
        long long o = (long long)chunk_id * query_stride + (long long)q_tile * QT + lane;
        out_s1[o] = s1;
        out_i1[o] = i1;
        out_s2[o] = s2;
        out_i2[o] = i2;
    }
}

__device__ __forceinline__ float there_v11_exact_d2(
    const float* __restrict__ x, const float qs[8], int lane
) {
    float d = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int k = lane * 8 + c;
        if (k < V11_DIMS) {
            float df = qs[c] - x[k];
            d = fmaf(df, df, d);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    return d;
}

__device__ __forceinline__ void there_v11_exact_d2_x4(
    const float* __restrict__ dbv, int j, const float qs[8], int lane, float d[4]
) {
    float2 p[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float2* __restrict__ s2 = reinterpret_cast<const float2*>(dbv + (long long)(j + r) * V11_DIMS) + lane * 4;
        p[r][0] = s2[0];
        p[r][1] = make_float2(0.0f, 0.0f); p[r][2] = p[r][1]; p[r][3] = p[r][1];
        if (lane < 31) { p[r][1] = s2[1]; p[r][2] = s2[2]; p[r][3] = s2[3]; }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        float acc = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int k = lane * 8 + 2 * c;
            if (k < V11_DIMS) {
                float df = qs[2 * c] - p[r][c].x;
                acc = fmaf(df, df, acc);
                float dg = qs[2 * c + 1] - p[r][c].y;
                acc = fmaf(dg, dg, acc);
            }
        }
        d[r] = acc;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) d[r] += __shfl_xor_sync(0xffffffffu, d[r], o);
    }
}

__device__ __forceinline__ void there_v11_take_min(float d, int idx, float& best, int& best_idx) {
    if (d < best || (d == best && idx < best_idx)) {
        best = d;
        best_idx = idx;
    }
}

extern "C" __global__ void there_v11_merge_rerank(
    const float* __restrict__ s1,
    const int* __restrict__ i1,
    const float* __restrict__ s2,
    const int* __restrict__ i2,
    const float* __restrict__ qv,
    const float* __restrict__ dbv,
    int* __restrict__ out,
    int nq,
    int nchunks,
    int query_stride,
    int chunk_db,
    int database_size,
    float tau_half    
) {
    int q = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (q >= nq) return;

    float m = -3.0e38f;
    for (int c = lane; c < nchunks; c += 32) m = fmaxf(m, s1[(long long)c * query_stride + q]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float thr = m - tau_half;

    const float* __restrict__ qr = qv + (long long)q * V11_DIMS;
    float qs[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int k = lane * 8 + c;
        qs[c] = k < V11_DIMS ? qr[k] : 0.0f;
    }

    float best = FLT_MAX;
    int best_idx = 0x7fffffff;

    for (int c = 0; c < nchunks; ++c) {
        long long o = (long long)c * query_stride + q;
        float sc1 = 0.0f, sc2 = 0.0f;
        int c1 = -1;
        if (lane == 0) { sc1 = s1[o]; sc2 = s2[o]; c1 = i1[o]; }
        sc1 = __shfl_sync(0xffffffffu, sc1, 0);
        sc2 = __shfl_sync(0xffffffffu, sc2, 0);
        c1 = __shfl_sync(0xffffffffu, c1, 0);

        if (sc2 >= thr) {
            int start = c * chunk_db;
            int end = min(start + chunk_db, database_size);
            int j = start;
            for (; j + 4 <= end; j += 4) {
                float d[4];
                there_v11_exact_d2_x4(dbv, j, qs, lane, d);
                #pragma unroll
                for (int r = 0; r < 4; ++r) there_v11_take_min(d[r], j + r, best, best_idx);
            }
            for (; j < end; ++j) {
                float d = there_v11_exact_d2(dbv + (long long)j * V11_DIMS, qs, lane);
                there_v11_take_min(d, j, best, best_idx);
            }
        } else if (c1 >= 0 && c1 < database_size && sc1 >= thr) {
            float d = there_v11_exact_d2(dbv + (long long)c1 * V11_DIMS, qs, lane);
            there_v11_take_min(d, c1, best, best_idx);
        }
    }

    if (lane == 0) out[q] = (best_idx == 0x7fffffff) ? 0 : best_idx;
}
