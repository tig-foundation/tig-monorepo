// Fused FP16 tensor-core nearest-neighbour kernel for TIG vector_search (c004), no cuBLAS.
//
// dist^2(q,d) = ||q||^2 + ||d||^2 - 2<q,d>;  ||q||^2 is constant per query and dropped.
// <q,d> runs as a block GEMM (BM queries x BN database rows, K = 256 = 250 dims + zero pad)
// on the tensor cores (WMMA m16n16k16, FP16 inputs). The distance matrix is never written:
// the epilogue reduces every tile straight to (min, argmin) per query in registers.
//
// Submission constraint (tig-binary/scripts/build_ptx): nvcc -arch compute_70 -code sm_70
// --use_fast_math. cp.async (sm_80), ldmatrix (sm_75) and mma.sync m16n8k16 (sm_80) are
// therefore unavailable -> WMMA API + register-staged double buffering
// (global -> registers -> shared), one __syncthreads per K step.
//
// Design (REVIEW-c004-kernel-2026-09-09 §3.1, Stufe B):
//   * BM x BN block tile (128x128), WM x WN warp tile (32x32 -> 16 warps, 512 threads).
//   * The query tile (128 x 256 halfs = 64 KB) stays resident in shared memory for a whole
//     work item; only the database tiles stream through a double-buffered stage of BK=32
//     columns. Shared rows are padded (+8 halfs) so fragment loads are bank-conflict free.
//   * Persistent CTAs walk work items (query tile, database chunk) in chunk-major order, so
//     all resident CTAs read the same database chunk at the same time: the database is read
//     from DRAM once and served from L2.
//   * Every work item writes its chunk winner per query (no atomics); `rerank` then re-evaluates all
//     chunk winners within `delta` of the best FP16 value with exact FP32 distances (FP16 near-ties
//     between chunks are resolved exactly; ties inside one 4096-row chunk remain FP16-ranked).
//   * Epilogue in registers: the accumulator fragment layout of sm_70+ (element e of lane l
//     sits at row (l>>2) + 8*((e>>1)&1), column 8*(e>>2) + 2*(l&3) + (e&1)) lets every lane
//     reduce its 4 columns per row locally, then 4 lanes merge via __shfl. At the end of a
//     chunk the warps merge through shared memory and write a packed
//     (ordered(val) << 32 | index) per (chunk, query).
//   * FP16 accumulation (full tensor rate on GeForce Ada, which halves the FP32-accumulate
//     rate) in a single accumulator set over K (NSPLIT = 1); the exact re-ranking absorbs the
//     larger FP16 rounding (measured: identical answers with re-rank windows 0.5 and 8).
#include <mma.h>
#include <cuda_fp16.h>
#include <stdint.h>
using namespace nvcuda;

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef WM
#define WM 32
#endif
#ifndef WN
#define WN 32
#endif
#define NSPLIT 1                     // one FP16 accumulator set over K (2 = split-K, more precise, ~8 % slower)
#ifndef KUNROLL
#define KUNROLL 8                    // K loop fully unrolled (KSTEPS = 8); NSPLIT > 1 needs static split index
#endif
#define FUSED_PRAGMA(x) _Pragma(#x)
#define FUSED_UNROLL(n) FUSED_PRAGMA(unroll n)
#define KDIM 256
#define LDS (BK + 8)                 // shared row stride of the database stage in halfs (padded)
#define LDQ (KDIM + 8)               // shared row stride of the resident query tile (528 B: conflict free)
#define WARPS_M (BM / WM)
#define WARPS_N (BN / WN)
#define NWARPS (WARPS_M * WARPS_N)
#define NT (NWARPS * 32)
#define FM (WM / 16)
#define FN (WN / 16)
#define KSTEPS (KDIM / BK)
#define CPR (BK / 8)                 // 16-byte chunks per row and K step
#define B_CHUNKS (BN * CPR)
#define B_PT (B_CHUNKS / NT)
#define Q_CHUNKS (BM * KDIM / 8)
#define Q_PT (Q_CHUNKS / NT)
#define Q_BYTES (BM * LDQ * 2)
#define STAGE_BYTES (BN * LDS * 2)
#define STAGE_OFF (Q_BYTES)
#define NRM_OFF (STAGE_OFF + 2 * STAGE_BYTES)
#define SMEM_BYTES (NRM_OFF + 2 * BN * 4)
static_assert(B_CHUNKS % NT == 0 && Q_CHUNKS % NT == 0, "chunk distribution");
static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0, "tiling");
static_assert(KDIM % BK == 0 && BK % 16 == 0 && (KDIM / BK) % 2 == 0, "K step");
static_assert(KDIM % NSPLIT == 0 && (KDIM / NSPLIT) % 16 == 0, "split-K");
static_assert((STAGE_BYTES % 32) == 0 && (STAGE_OFF % 32) == 0, "WMMA alignment 32 B");
static_assert(2 * STAGE_BYTES >= BM * WARPS_N * 8, "merge buffer fits in the stage buffers");
static_assert(BN <= NT, "norm load");

// FP16 accumulation (full tensor-core rate on GeForce Ada); precision near ties comes from `rerank`.
typedef half acc_t;
#define ACC_ZERO __float2half(0.0f)

// Meta kernel: hands the compile-time constants to the host (no host/device mismatch possible).
extern "C" __global__ void fused_info(unsigned* out){
    if (threadIdx.x == 0){
        out[0] = BM; out[1] = BN; out[2] = NT; out[3] = SMEM_BYTES; out[4] = BK; out[5] = WM; out[6] = WN;
        out[7] = 16;                 // accumulator width in bits
        out[8] = NSPLIT;
    }
}

// Cast f32 -> f16 with row padding to KDIM columns and ||d^||^2 from the f16 value (one warp per
// row, grid-stride over rows). Rows [n_rows, n_rows_pad) become zero, their norm +INF (never a
// minimum). norms == nullptr for the queries.
//
// Launch with a SMALL grid (a few blocks per SM), not one warp per row: the official build_ptx
// instrumentation ends every warp with three same-address atomics (fuel/signature), which
// serialise in L2 at ~20 ns each. One warp per row = 700k warps @7000 = ~40 ms of atomics
// (measured: fbench 87 vs 47 ms with/without the instrumentation); a persistent grid of a few
// thousand warps makes that negligible.
extern "C" __global__ void cast_norm_pad(const float* __restrict__ in, __half* __restrict__ out,
        float* __restrict__ norms, unsigned n_rows, unsigned n_rows_pad, unsigned in_dim){
    const unsigned lane = threadIdx.x & 31;
    const unsigned nwarps = (gridDim.x * blockDim.x) >> 5;
    for (unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < n_rows_pad; row += nwarps){
        __half* dst = out + (size_t)row * KDIM;
        if (row >= n_rows){
            for (unsigned c = lane; c < KDIM; c += 32) dst[c] = __float2half(0.0f);
            if (lane == 0 && norms) norms[row] = INFINITY;
            continue;
        }
        const float* src = in + (size_t)row * in_dim;
        float s = 0.0f;
        for (unsigned c = lane; c < KDIM; c += 32){
            float v = (c < in_dim) ? src[c] : 0.0f;
            __half h = __float2half(v);
            dst[c] = h;
            float f = __half2float(h);
            s = fmaf(f, f, s);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0 && norms) norms[row] = s;
    }
}

__device__ __forceinline__ unsigned f2ord(float f){
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned o){
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}

// Q: [nq_tiles*BM][KDIM] f16, D: [ndb_tiles*BN][KDIM] f16, dn: [ndb_tiles*BN] f32 (INF in the pad).
// cand[chunk * nq_pad + q]: packed (f2ord(val) << 32 | idx) of the best row of `chunk` for query q.
// Every (chunk, query) slot is written by exactly one work item: no atomics, no initialisation.
extern "C" __global__ void __launch_bounds__(NT, 1)
fused_nn(const __half* __restrict__ Q, const __half* __restrict__ D, const float* __restrict__ dn,
         unsigned nq_tiles, unsigned ndb_tiles, unsigned tiles_per_chunk,
         unsigned long long* __restrict__ cand){
    extern __shared__ __align__(128) unsigned char smem[];
    __half* Qs = (__half*)smem;                                          // [BM][LDQ]
    float* nrm_sh = (float*)(smem + NRM_OFF);                            // [2][BN]
    unsigned long long* red = (unsigned long long*)(smem + STAGE_OFF);   // [BM][WARPS_N], reuses the stage buffers after a chunk
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int t = lane & 3;

    const unsigned nchunks = (ndb_tiles + tiles_per_chunk - 1) / tiles_per_chunk;
    const unsigned nitems  = nq_tiles * nchunks;

    for (unsigned w = blockIdx.x; w < nitems; w += gridDim.x){
        const unsigned chunk = w / nq_tiles, qt = w - chunk * nq_tiles;
        const unsigned t0 = chunk * tiles_per_chunk;
        const unsigned t1 = min(t0 + tiles_per_chunk, ndb_tiles);
        const unsigned ntiles = t1 - t0;
        const __half* Qb = Q + (size_t)qt * BM * KDIM;

        // Running (min, argmin): lane holds rows g = lane>>2 and g+8 of every fragment row i.
        float bv[FM][2]; unsigned bi[FM][2];
        #pragma unroll
        for (int i = 0; i < FM; i++){ bv[i][0] = bv[i][1] = INFINITY; bi[i][0] = bi[i][1] = 0u; }

        wmma::fragment<wmma::accumulator, 16, 16, 16, acc_t> acc[NSPLIT][FM][FN];
        uint4 rb[B_PT]; float rn = 0.0f;

        // Query tile once per work item (64 KB, ~1.5 % of the item's traffic).
        {
            uint4 rq[Q_PT];
            #pragma unroll
            for (int p = 0; p < Q_PT; p++){
                const unsigned c = tid + p * NT, row = c / (KDIM / 8), kc = c % (KDIM / 8);
                rq[p] = __ldg((const uint4*)(Qb + (size_t)row * KDIM + kc * 8));
            }
            #pragma unroll
            for (int p = 0; p < Q_PT; p++){
                const unsigned c = tid + p * NT, row = c / (KDIM / 8), kc = c % (KDIM / 8);
                *(uint4*)(Qs + row * LDQ + kc * 8) = rq[p];
            }
        }

        // Load step (tile, ks) of the database: global -> registers.
        auto load_regs = [&](unsigned tile, unsigned ks){
            const unsigned k0 = ks * BK;
            const __half* Db = D + (size_t)tile * BN * KDIM;
            #pragma unroll
            for (int p = 0; p < B_PT; p++){
                const unsigned c = tid + p * NT, row = c / CPR, kc = c % CPR;
                rb[p] = __ldg((const uint4*)(Db + (size_t)row * KDIM + k0 + kc * 8));
            }
            if (ks == 0 && tid < BN) rn = __ldg(dn + (size_t)tile * BN + tid);
        };
        // Registers -> shared stage buffer `buf`.
        auto store_regs = [&](unsigned tile, unsigned ks, unsigned buf){
            __half* Bs = (__half*)(smem + STAGE_OFF + buf * STAGE_BYTES);
            #pragma unroll
            for (int p = 0; p < B_PT; p++){
                const unsigned c = tid + p * NT, row = c / CPR, kc = c % CPR;
                *(uint4*)(Bs + row * LDS + kc * 8) = rb[p];
            }
            if (ks == 0 && tid < BN) nrm_sh[(tile & 1) * BN + tid] = rn;
        };

        load_regs(t0, 0);
        store_regs(t0, 0, 0);
        __syncthreads();

        for (unsigned ti = 0; ti < ntiles; ti++){
            const unsigned tile = t0 + ti;
            const bool last_tile = (ti + 1 == ntiles);
            #pragma unroll
            for (int sp = 0; sp < NSPLIT; sp++)
                #pragma unroll
                for (int i = 0; i < FM; i++)
                    #pragma unroll
                    for (int j = 0; j < FN; j++) wmma::fill_fragment(acc[sp][i][j], ACC_ZERO);

            FUSED_UNROLL(KUNROLL)
            for (int ks = 0; ks < KSTEPS; ks++){
                const int buf = ks & 1;
                // Prefetch of the next step overlaps the MMAs below. At the last step of a chunk a
                // valid tile is re-loaded instead of branching (10 KB wasted, harmless): the tile loop
                // body stays free of data-dependent branches (the official build_ptx instruments every
                // PTX basic block, so fewer blocks = fewer counter updates).
                const unsigned ntile = (ks + 1 < KSTEPS) ? tile : (last_tile ? tile : tile + 1);
                const unsigned nks   = (ks + 1 < KSTEPS) ? (unsigned)(ks + 1) : 0u;
                load_regs(ntile, nks);

                const __half* Bs = (const __half*)(smem + STAGE_OFF + buf * STAGE_BYTES);
                const __half* As = Qs + ks * BK;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += 16){
                    const int sp = (ks * BK + kk) / (KDIM / NSPLIT);
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[FN];
                    #pragma unroll
                    for (int i = 0; i < FM; i++) wmma::load_matrix_sync(a[i], As + (wm * WM + i * 16) * LDQ + kk, LDQ);
                    #pragma unroll
                    for (int j = 0; j < FN; j++) wmma::load_matrix_sync(b[j], Bs + (wn * WN + j * 16) * LDS + kk, LDS);
                    #pragma unroll
                    for (int i = 0; i < FM; i++)
                        #pragma unroll
                        for (int j = 0; j < FN; j++) wmma::mma_sync(acc[sp][i][j], a[i], b[j], acc[sp][i][j]);
                }

                store_regs(ntile, nks, buf ^ 1);
                __syncthreads();
            }
            {
                {
                    // Register epilogue: min over the lane's 4 columns per row, then over the 4
                    // lanes sharing that row via __shfl (ties -> smaller column, i.e. smaller index).
                    // All compares use non-short-circuit `|`/`&` so nvcc emits predicates, not branches.
                    const float* nr = nrm_sh + (tile & 1) * BN + wn * WN;
                    #pragma unroll
                    for (int j = 0; j < FN; j++){
                        const float2 n0 = *(const float2*)(nr + j * 16 + 2 * t);
                        const float2 n1 = *(const float2*)(nr + j * 16 + 8 + 2 * t);
                        const float nrm[4] = { n0.x, n0.y, n1.x, n1.y };   // columns 2t, 2t+1, 8+2t, 9+2t
                        #pragma unroll
                        for (int i = 0; i < FM; i++){
                            float sv[8];
                            #pragma unroll
                            for (int e = 0; e < 8; e++) sv[e] = 0.0f;
                            #pragma unroll
                            for (int sp = 0; sp < NSPLIT; sp++)
                                #pragma unroll
                                for (int e = 0; e < 8; e++) sv[e] += (float)acc[sp][i][j].x[e];
                            #pragma unroll
                            for (int hh = 0; hh < 2; hh++){          // hh = 0: row g, hh = 1: row g+8
                                float lv = INFINITY; unsigned lc = 0u;
                                #pragma unroll
                                for (int q = 0; q < 4; q++){          // column slot q -> e = (q>>1)*4 + hh*2 + (q&1)
                                    const int e = (q >> 1) * 4 + hh * 2 + (q & 1);
                                    const unsigned col = (unsigned)((q >> 1) * 8 + 2 * t + (q & 1));
                                    const float v = fmaf(-2.0f, sv[e], nrm[q]);
                                    const bool better = v < lv;
                                    lv = better ? v : lv; lc = better ? col : lc;
                                }
                                #pragma unroll
                                for (int o = 1; o <= 2; o <<= 1){
                                    const float ov = __shfl_xor_sync(0xffffffffu, lv, o);
                                    const unsigned oc = __shfl_xor_sync(0xffffffffu, lc, o);
                                    const bool take = (ov < lv) | ((ov == lv) & (oc < lc));
                                    lv = take ? ov : lv; lc = take ? oc : lc;
                                }
                                const unsigned gi = tile * BN + wn * WN + j * 16 + lc;
                                const bool takeb = (lv < bv[i][hh]) | ((lv == bv[i][hh]) & (gi < bi[i][hh]));
                                bv[i][hh] = takeb ? lv : bv[i][hh]; bi[i][hh] = takeb ? gi : bi[i][hh];
                            }
                        }
                    }
                }
            }
        }

        // End of chunk: merge across the warps of a row (wn) and write the chunk winner.
        if (t == 0){
            #pragma unroll
            for (int i = 0; i < FM; i++)
                #pragma unroll
                for (int hh = 0; hh < 2; hh++){
                    const unsigned row = wm * WM + i * 16 + (lane >> 2) + 8 * hh;
                    red[row * WARPS_N + wn] = ((unsigned long long)f2ord(bv[i][hh]) << 32) | bi[i][hh];
                }
        }
        __syncthreads();
        if (tid < BM){
            unsigned long long m = red[tid * WARPS_N];
            #pragma unroll
            for (int x = 1; x < WARPS_N; x++) m = min(m, red[tid * WARPS_N + x]);
            cand[(size_t)chunk * nq_tiles * BM + (size_t)qt * BM + tid] = m;
        }
        __syncthreads();   // red[] aliases the stage buffers of the next work item
    }
}

// Exact re-ranking: for every query, all chunk winners whose FP16 value lies within `delta` of the
// best FP16 value are re-evaluated with the exact FP32 squared distance on the original vectors.
// One warp per query, grid-stride over queries with a small grid (instrumentation atomics, see above).
// The host uses delta = 0.5 (squared distances ~100); windows 0.5 and 8 gave identical answers on
// 40 instances across all five tracks.
extern "C" __global__ void rerank(const float* __restrict__ qv, const float* __restrict__ dv,
        const unsigned long long* __restrict__ cand, unsigned nq, unsigned nq_pad, unsigned nchunks,
        unsigned dims, float delta, unsigned* __restrict__ out){
    const unsigned lane = threadIdx.x & 31;
    const unsigned nwarps = (gridDim.x * blockDim.x) >> 5;
    const unsigned ncand = nchunks;
    for (unsigned q = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; q < nq; q += nwarps){
        unsigned long long m = ~0ull;
        for (unsigned c = lane; c < ncand; c += 32)
            m = min(m, __ldg(cand + (size_t)c * nq_pad + q));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = min(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float v0 = ord2f((unsigned)(m >> 32));
        const float thr = v0 + delta;
        float qr[8];
        #pragma unroll
        for (int k = 0; k < 8; k++){ const unsigned d = lane + 32 * k; qr[k] = (d < dims) ? __ldg(qv + (size_t)q * dims + d) : 0.0f; }
        float bd = INFINITY; unsigned bi = (unsigned)m;
        for (unsigned base = 0; base < ncand; base += 32){
            const unsigned c = base + lane;
            unsigned long long x = ~0ull;
            if (c < ncand) x = __ldg(cand + (size_t)c * nq_pad + q);
            const float v = ord2f((unsigned)(x >> 32));
            unsigned bal = __ballot_sync(0xffffffffu, (c < ncand) & (v <= thr));
            while (bal){
                const int src = __ffs(bal) - 1; bal &= bal - 1;
                const unsigned di = __shfl_sync(0xffffffffu, (unsigned)x, src);
                float s = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++){
                    const unsigned d = lane + 32 * k;
                    const float e = (d < dims) ? qr[k] - __ldg(dv + (size_t)di * dims + d) : 0.0f;
                    s = fmaf(e, e, s);
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
                const bool take = (s < bd) | ((s == bd) & (di < bi));
                bd = take ? s : bd; bi = take ? di : bi;
            }
        }
        if (lane == 0) out[q] = bi;
    }
}
