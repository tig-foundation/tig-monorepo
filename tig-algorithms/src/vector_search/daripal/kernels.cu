#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

#define SW_DIMS 250
#define SW_DPAD 256

__device__ __forceinline__ unsigned daripal_pack2(float a, float b) {
    return (unsigned)__half_as_ushort(__float2half(a)) | ((unsigned)__half_as_ushort(__float2half(b)) << 16);
}

extern "C" __global__ void __launch_bounds__(256) daripal_pack_fold(
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
            const float2* __restrict__ s2 = reinterpret_cast<const float2*>(src + (long long)row * SW_DIMS) + lane * 4;
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

        long long idx4 = (long long)(row / QT) * (QT * SW_DPAD / 8)
                       + (long long)kt * (QT * 2)
                       + (long long)(row % QT) * 2
                       + (off >> 3);
        uint4 pk;
        pk.x = daripal_pack2(v[0], v[1]);
        pk.y = daripal_pack2(v[2], v[3]);
        pk.z = daripal_pack2(v[4], v[5]);
        pk.w = daripal_pack2(v[6], v[7]);
        dst4[idx4] = pk;
    }
}
__device__ __forceinline__ float daripal_exact_d2(
    const float* __restrict__ x, const float qs[8], int lane
) {
    float d = 0.0f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int k = lane * 8 + c;
        if (k < SW_DIMS) {
            float df = qs[c] - x[k];
            d = fmaf(df, df, d);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    return d;
}

__device__ __forceinline__ void daripal_exact_d2_x4(
    const float* __restrict__ dbv, int j, const float qs[8], int lane, float d[4]
) {
    float2 p[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float2* __restrict__ s2 = reinterpret_cast<const float2*>(dbv + (long long)(j + r) * SW_DIMS) + lane * 4;
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
            if (k < SW_DIMS) {
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

__device__ __forceinline__ void daripal_take_min(float d, int idx, float& best, int& best_idx) {
    if (d < best || (d == best && idx < best_idx)) {
        best = d;
        best_idx = idx;
    }
}


typedef wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> daripal_fa;
typedef wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> daripal_fb;
typedef wmma::fragment<wmma::accumulator, 32, 8, 16, half> daripal_fc;

#define SW_NEG2 0xFC00FC00u
#define SW_WARPS 8

__device__ __forceinline__ unsigned& daripal_pair(daripal_fc& f, int j) { return reinterpret_cast<unsigned*>(f.x)[j]; }

__device__ __forceinline__ unsigned daripal_gt2(unsigned a, unsigned b) {
    unsigned m;
    asm("set.gt.u32.f16x2 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
    return m;
}

__device__ __forceinline__ unsigned daripal_sel(unsigned m, unsigned a, unsigned b) { return (a & m) | (b & ~m); }

__device__ __forceinline__ void daripal_top2(unsigned v, unsigned idx, unsigned& s1, unsigned& s2, unsigned& i1) {
    unsigned m = daripal_gt2(v, s1);
    unsigned t = daripal_sel(m, s1, v);
    s1 = daripal_sel(m, v, s1);
    i1 = daripal_sel(m, idx, i1);
    unsigned m2 = daripal_gt2(t, s2);
    s2 = daripal_sel(m2, t, s2);
}

extern "C" __global__ void __launch_bounds__(256, 1) daripal_scan(
    const unsigned short* __restrict__ q_u16, const unsigned short* __restrict__ db_u16,
    float* __restrict__ out_s1, int* __restrict__ out_i1, float* __restrict__ out_s2,
    int database_size, int query_stride, int chunk_db, int nqb, int nchunks) {
    constexpr int NT = 32, LD = 24, KT = 16, THREADS = 256;
    constexpr int GROUP_HALVES = NT * KT * LD;
    constexpr int PER_T = NT * 32 / THREADS;
    const half* __restrict__ q_half = reinterpret_cast<const half*>(q_u16);
    const half* __restrict__ db_half = reinterpret_cast<const half*>(db_u16);

    extern __shared__ int4 daripal_dyn[];
    half* bufs = reinterpret_cast<half*>(daripal_dyn);

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    half* ms0 = bufs + warp * 768;
    half* ms1 = ms0 + 256;
    unsigned short* ms2 = reinterpret_cast<unsigned short*>(ms0 + 512);
    for (int e = lane; e < 256; e += 32) ms0[e] = __int2half_rn(e & 7);
    __syncwarp();
    unsigned cp[4];
    {
        daripal_fc cf;
        wmma::load_matrix_sync(cf, ms0, 8, wmma::mem_row_major);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            cp[j] = (unsigned)__half2int_rn(cf.x[2 * j]) | ((unsigned)__half2int_rn(cf.x[2 * j + 1]) << 16);
    }

    int soff[PER_T];
    #pragma unroll
    for (int i = 0; i < PER_T; ++i) {
        int gi = tid + i * THREADS;
        int r = (gi >> 9) * 16 + ((gi & 31) >> 1);
        int kt = (gi & 511) >> 5;
        soff[i] = (kt * NT + r) * (LD / 8) + (gi & 1);
    }

    const int total = nqb * nchunks;
    for (int item = (int)blockIdx.x; item < total; item += (int)gridDim.x) {
        const int qb = item % nqb;
        const int chunk = item / nqb;
        const int chunk_start = chunk * chunk_db;
        const int chunk_len = min(database_size - chunk_start, chunk_db);
        const int q_tile = qb * SW_WARPS + warp;
        const bool active = q_tile * 32 < query_stride;

        daripal_fa a[KT];
        if (active) {
            const half* __restrict__ qt = q_half + (long long)q_tile * (32 * SW_DPAD);
            #pragma unroll
            for (int k = 0; k < KT; ++k) wmma::load_matrix_sync(a[k], qt + k * (32 * 16), 16);
        } else {
            #pragma unroll
            for (int k = 0; k < KT; ++k) wmma::fill_fragment(a[k], __float2half(0.0f));
        }

        unsigned s1[4], s2[4], i1[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { s1[j] = SW_NEG2; s2[j] = SW_NEG2; i1[j] = 0u; }

        const int ng = ((chunk_len + NT - 1) / NT + 1) & ~1;
        const int4* __restrict__ gsrc = reinterpret_cast<const int4*>(db_half + (long long)chunk_start * SW_DPAD);

        int4 pre[PER_T];
        auto load_group = [&](int g) {
            const int4* __restrict__ src = gsrc + (long long)min(g, ng - 1) * (NT * 32);
            #pragma unroll
            for (int i = 0; i < PER_T; ++i) pre[i] = src[tid + i * THREADS];
        };
        auto store_group = [&](half* buf) {
            int4* dst = reinterpret_cast<int4*>(buf);
            #pragma unroll
            for (int i = 0; i < PER_T; ++i) dst[soff[i]] = pre[i];
        };

        daripal_fc accA[4], accB[4];
        auto mma = [&](daripal_fc (&acc)[4], const half* __restrict__ dbs) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[n], __float2half(0.0f));
            #pragma unroll
            for (int k = 0; k < KT; ++k) {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    daripal_fb b;
                    wmma::load_matrix_sync(b, dbs + (k * NT + n * 8) * LD, LD);
                    wmma::mma_sync(acc[n], a[k], b, acc[n]);
                }
            }
        };
        auto epi = [&](daripal_fc (&acc)[4], int g) {
            const unsigned base = (unsigned)(g * 32) * 0x10001u;
            #pragma unroll
            for (int n = 0; n < 4; ++n)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    daripal_top2(daripal_pair(acc[n], j), base + (unsigned)(n * 8) * 0x10001u + cp[j], s1[j], s2[j], i1[j]);
        };

        #pragma unroll
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(accB[n], __ushort_as_half((unsigned short)0xFC00));

        half* b0 = bufs;
        half* b1 = bufs + GROUP_HALVES;
        half* b2 = bufs + 2 * GROUP_HALVES;
        half* b3 = bufs + 3 * GROUP_HALVES;
        auto pair = [&](int g, half* ra, half* rb, half* wa, half* wb) {
            load_group(g + 2);
            mma(accA, ra);
            epi(accB, g - 1);
            store_group(wa);
            load_group(g + 3);
            mma(accB, rb);
            epi(accA, g);
            store_group(wb);
            __syncthreads();
        };

        __syncthreads();
        load_group(0);
        store_group(b0);
        load_group(1);
        store_group(b1);
        __syncthreads();
        int g = 0;
        for (; g + 4 <= ng; g += 4) {
            pair(g, b0, b1, b2, b3);
            pair(g + 2, b2, b3, b0, b1);
        }
        if (g < ng) pair(g, b0, b1, b2, b3);
        epi(accB, ng - 1);

        if (active) {
            daripal_fc f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) daripal_pair(f, j) = s1[j];
            wmma::store_matrix_sync(ms0, f, 8, wmma::mem_row_major);
            #pragma unroll
            for (int j = 0; j < 4; ++j) daripal_pair(f, j) = s2[j];
            wmma::store_matrix_sync(ms1, f, 8, wmma::mem_row_major);
            #pragma unroll
            for (int j = 0; j < 4; ++j) daripal_pair(f, j) = i1[j];
            wmma::store_matrix_sync(reinterpret_cast<half*>(ms2), f, 8, wmma::mem_row_major);
            __syncwarp();
            float b1 = -FLT_MAX, b2 = -FLT_MAX;
            int bi = 0x7fffffff;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                float v1 = __half2float(ms0[lane * 8 + c]);
                float v2 = __half2float(ms1[lane * 8 + c]);
                int ix = (int)ms2[lane * 8 + c];
                b2 = fmaxf(b2, v2);
                bool better = v1 > b1 || (v1 == b1 && ix < bi);
                b2 = fmaxf(b2, better ? b1 : v1);
                b1 = better ? v1 : b1;
                bi = better ? ix : bi;
            }
            long long o = (long long)chunk * query_stride + (long long)q_tile * 32 + lane;
            out_s1[o] = b1;
            out_i1[o] = chunk_start + bi;
            out_s2[o] = b2;
        }
    }
}

__device__ __forceinline__ unsigned long long daripal_key(float d, int idx) {
    return ((unsigned long long)__float_as_uint(d) << 32) | (unsigned)idx;
}

extern "C" __global__ void daripal_merge(
    const float* __restrict__ s1,
    const int* __restrict__ i1,
    const float* __restrict__ s2,
    const float* __restrict__ qv,
    const float* __restrict__ dbv,
    unsigned long long* __restrict__ best_out,
    int2* __restrict__ rescan,
    int* __restrict__ rescan_count,
    int nq,
    int nchunks,
    int query_stride,
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

    const float* __restrict__ qr = qv + (long long)q * SW_DIMS;
    float qs[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int k = lane * 8 + c;
        qs[c] = k < SW_DIMS ? qr[k] : 0.0f;
    }

    float best = FLT_MAX;
    int best_idx = 0x7fffffff;

    for (int base = 0; base < nchunks; base += 32) {
        int c = base + lane;
        float a1 = -3.0e38f, a2 = -3.0e38f;
        int ci = -1;
        if (c < nchunks) {
            long long o = (long long)c * query_stride + q;
            a1 = s1[o];
            a2 = s2[o];
            ci = i1[o];
        }
        bool full = a2 >= thr;
        if (full) rescan[atomicAdd(rescan_count, 1)] = make_int2(q, c);
        bool one = !full && a1 >= thr && ci >= 0 && ci < database_size;
        unsigned mask = __ballot_sync(0xffffffffu, one);
        while (mask) {
            int src = __ffs(mask) - 1;
            mask &= mask - 1;
            int j = __shfl_sync(0xffffffffu, ci, src);
            float d = daripal_exact_d2(dbv + (long long)j * SW_DIMS, qs, lane);
            daripal_take_min(d, j, best, best_idx);
        }
    }

    if (lane == 0) best_out[q] = daripal_key(best, best_idx);
}

extern "C" __global__ void __launch_bounds__(256) daripal_rescan(
    const int2* __restrict__ rescan,
    const int* __restrict__ rescan_count,
    const float* __restrict__ qv,
    const float* __restrict__ dbv,
    unsigned long long* __restrict__ best_out,
    int chunk_db,
    int database_size
) {
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)(threadIdx.x >> 5);
    const int warps = (int)(blockDim.x >> 5);
    const int n = *rescan_count;
    for (int e = (int)blockIdx.x; e < n; e += (int)gridDim.x) {
        int2 p = rescan[e];
        const float* __restrict__ qr = qv + (long long)p.x * SW_DIMS;
        float qs[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            int k = lane * 8 + c;
            qs[c] = k < SW_DIMS ? qr[k] : 0.0f;
        }
        int start = p.y * chunk_db;
        int end = min(start + chunk_db, database_size);
        float best = FLT_MAX;
        int best_idx = 0x7fffffff;
        int j = start + warp * 4;
        for (; j + 4 <= end; j += warps * 4) {
            float d[4];
            daripal_exact_d2_x4(dbv, j, qs, lane, d);
            #pragma unroll
            for (int r = 0; r < 4; ++r) daripal_take_min(d[r], j + r, best, best_idx);
        }
        for (; j < end; ++j) {
            float d = daripal_exact_d2(dbv + (long long)j * SW_DIMS, qs, lane);
            daripal_take_min(d, j, best, best_idx);
        }
        if (lane == 0 && best_idx != 0x7fffffff) atomicMin(&best_out[p.x], daripal_key(best, best_idx));
    }
}
