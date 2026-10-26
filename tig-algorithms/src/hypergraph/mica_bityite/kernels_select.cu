
namespace filt {

constexpr int S_THREADS = 1024;

__device__ __forceinline__ int chunk_of(int n, int grid) { return (n + grid - 1) / grid; }

__device__ __forceinline__ bool signed_key_valid(int key) {
    return (unsigned int)key != 0x80000000u;
}
__device__ __forceinline__ int signed_key_gain(int key) { return key >> 16; }
__device__ __forceinline__ bool biased_key_valid(int key) { return key > 0; }
__device__ __forceinline__ int biased_key_gain(int key) { return (key >> 16) - 1000; }

template <int T> __device__ __forceinline__ int blk_scan(int v, int *sh, int *total) {
    constexpr int W = T / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = (lane < W) ? sh[lane] : 0;
        for (int o = 1; o < W; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < W) sh[lane] = w;
    }
    __syncthreads();
    int base = (warp > 0) ? sh[warp - 1] : 0;
    *total = sh[W - 1];
    __syncthreads();
    return base + x - v;
}

template <int T> __device__ __forceinline__ int blk_max(int v, int *sh) {
    constexpr int W = T / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 16; o > 0; o >>= 1)
        x = max(x, __shfl_xor_sync(0xffffffffu, x, o));
    if (lane == 0) sh[warp] = x;
    __syncthreads();
    int r = sh[0];
    for (int i = 1; i < W; i++)
        r = max(r, sh[i]);
    __syncthreads();
    return r;
}

template <int T> __device__ __forceinline__ int blk_sum(int v, int *sh) {
    constexpr int W = T / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 16; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) sh[warp] = x;
    __syncthreads();
    int r = 0;
    for (int i = 0; i < W; i++)
        r += sh[i];
    __syncthreads();
    return r;
}

template <int FILT_THREADS>
__device__ __forceinline__ void filt_stats(const int *__restrict__ keys,
                                           const int *__restrict__ tabu_until, int n, int round,
                                           int lottery_min, int *__restrict__ block_max,
                                           int *__restrict__ stats, const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        round = (int)gr[1];
    }
    __shared__ int sh[32];
    int chunk = chunk_of(n, gridDim.x);
    int lo = blockIdx.x * chunk;
    int hi = min(n, lo + chunk);
    int m = -2147483647 - 1;
    int nt = 0, tb0 = 0, tb1 = 0, tb2 = 0, tb3 = 0;
    for (int i = lo + threadIdx.x; i < hi; i += FILT_THREADS) {
        int key = keys[i];
        if (signed_key_valid(key)) {
            int g = signed_key_gain(key);
            if (g > m) m = g;
            if (g <= 0 && g >= lottery_min) {
                if (tabu_until[i] > round) {
                    tb0 += (g == -3);
                    tb1 += (g == -2);
                    tb2 += (g == -1);
                    tb3 += (g == 0);
                } else {
                    nt++;
                }
            }
        }
    }
    m = blk_max<FILT_THREADS>(m, sh);
    nt = blk_sum<FILT_THREADS>(nt, sh);
    tb0 = blk_sum<FILT_THREADS>(tb0, sh);
    tb1 = blk_sum<FILT_THREADS>(tb1, sh);
    tb2 = blk_sum<FILT_THREADS>(tb2, sh);
    tb3 = blk_sum<FILT_THREADS>(tb3, sh);
    if (threadIdx.x == 0) {
        block_max[blockIdx.x] = m;
        stats[blockIdx.x * 5 + 0] = nt;
        stats[blockIdx.x * 5 + 1] = tb0;
        stats[blockIdx.x * 5 + 2] = tb1;
        stats[blockIdx.x * 5 + 3] = tb2;
        stats[blockIdx.x * 5 + 4] = tb3;
    }
}

__device__ __forceinline__ int lottery_count(const int *stats, int b, int asp) {
    int c = stats[b * 5];
    if (-3 >= asp) c += stats[b * 5 + 1];
    if (-2 >= asp) c += stats[b * 5 + 2];
    if (-1 >= asp) c += stats[b * 5 + 3];
    if (0 >= asp) c += stats[b * 5 + 4];
    return c;
}

__device__ __forceinline__ unsigned long long
lcg_jump(unsigned long long s0, unsigned long long steps, unsigned long long c) {
    unsigned long long A = 1ull, C = 0ull;
    unsigned long long a = 6364136223846793005ull;
    while (steps) {
        if (steps & 1ull) {
            C = C * a + c;
            A = A * a;
        }
        c = c * a + c;
        a = a * a;
        steps >>= 1;
    }
    return A * s0 + C;
}

__device__ __forceinline__ int s_digit(unsigned long long w, int p) {
    return (int)((w >> (32 + 8 * p)) & 255ull);
}

__device__ __forceinline__ void s_hist_add(int *h4, unsigned long long w, bool valid,
                                           unsigned int lanemask_lt) {
    if (valid) {
        for (int p = 0; p < 4; p++) {
            int d = s_digit(w, p);
            atomicAdd(&h4[p * 256 + d], 1);
        }
    }
}

template <int FILT_THREADS>
__device__ __forceinline__ void
filt_pick4(const int *__restrict__ keys, const int *__restrict__ tabu_until, int n, int round,
                int lottery_min, const int *__restrict__ block_max, const int *__restrict__ stats,
                unsigned long long rng0, unsigned long long lcg_c, unsigned long long num,
                unsigned long long den0, unsigned long long den1, unsigned long long den2,
                unsigned long long den3, unsigned long long *__restrict__ out,
                int *__restrict__ block_sel, int *__restrict__ h4, const unsigned long long *gr,
                int *__restrict__ seg_hist = nullptr) {
    if (gr) {
        if (gr[0] != 0) return;
        round = (int)gr[1];
        rng0 = 123456789ull + gr[1];
        num = gr[16];
        den0 = gr[17];
        den1 = gr[18];
        den2 = gr[19];
        den3 = gr[20];
    }
    __shared__ int sh[32];
    __shared__ int carry_lottery;
    __shared__ int carry_sel;
    __shared__ int hist[1024];
    const unsigned int lanemask_lt = (1u << (threadIdx.x & 31)) - 1u;
    const int G = gridDim.x;
    int m = -2147483647 - 1;
    for (int b = threadIdx.x; b < G; b += FILT_THREADS)
        m = max(m, block_max[b]);
    int max_gain = blk_max<FILT_THREADS>(m, sh);
    if (max_gain == (-2147483647 - 1)) max_gain = 0;
    int aspiration = (max_gain * 3) / 4;
    int lp = 0;
    for (int b = threadIdx.x; b < (int)blockIdx.x; b += FILT_THREADS)
        lp += lottery_count(stats, b, aspiration);
    int lottery_prefix = blk_sum<FILT_THREADS>(lp, sh);

    int chunk = chunk_of(n, G);
    int lo = blockIdx.x * chunk;
    int hi = min(n, lo + chunk);
    for (int i = threadIdx.x; i < 1024; i += FILT_THREADS)
        hist[i] = 0;
    if (threadIdx.x == 0) {
        carry_lottery = 0;
        carry_sel = 0;
    }
    __syncthreads();
    for (int base = lo; base < hi; base += FILT_THREADS * 4) {
        int key4[4];
        int f4[4];
        int lot_local = 0;
        for (int j = 0; j < 4; j++) {
            const int i = base + threadIdx.x * 4 + j;
            int key = 0x80000000;
            int f = 0;
            if (i < hi) {
                key = keys[i];
                if (signed_key_valid(key)) {
                    int gain = signed_key_gain(key);
                    bool is_tabu = (tabu_until[i] > round) && (gain < aspiration);
                    if (!is_tabu) {
                        if (gain > 0)
                            f = 1;
                        else if (gain >= lottery_min)
                            f = 2;
                    }
                }
            }
            key4[j] = key;
            f4[j] = f;
            lot_local += (f == 2) ? 1 : 0;
        }
        int total_lottery;
        int rank_base = blk_scan<FILT_THREADS>(lot_local, sh, &total_lottery);
        int s4[4];
        int ok4[4];
        int sel_local = 0;
        int lr = 0;
        for (int j = 0; j < 4; j++) {
            int s = 0, ok = 0;
            if (f4[j] == 1) {
                s = 1;
                ok = key4[j];
            } else if (f4[j] == 2) {
                int gain = key4[j] >> 16;
                int penalty = (gain < 0) ? (-gain) : 0;
                unsigned long long den = (penalty & 3) == 0   ? den0
                                         : (penalty & 3) == 1 ? den1
                                         : (penalty & 3) == 2 ? den2
                                                              : den3;
                unsigned long long st = lcg_jump(
                    rng0, (unsigned long long)(lottery_prefix + carry_lottery + rank_base + lr + 1),
                    lcg_c);
                lr++;
                if (den > 0ull && (st % den) < num) {
                    s = 1;
                    ok = key4[j] & 0xFFFF;
                }
            }
            s4[j] = s;
            ok4[j] = ok;
            sel_local += s;
        }
        int total_sel;
        int pos_base = blk_scan<FILT_THREADS>(sel_local, sh, &total_sel);
        int p = 0;
        for (int j = 0; j < 4; j++) {
            const int i = base + threadIdx.x * 4 + j;
            if (s4[j]) {
                out[lo + carry_sel + pos_base + p] = ((unsigned long long)(unsigned int)i << 32) |
                                                     (unsigned long long)(unsigned int)ok4[j];
                p++;
            }
            unsigned long long w = ((unsigned long long)(~(unsigned int)ok4[j]) << 32) |
                                   (unsigned long long)(unsigned int)i;
            s_hist_add(hist, w, s4[j] != 0, lanemask_lt);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            carry_lottery += total_lottery;
            carry_sel += total_sel;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sel[blockIdx.x] = carry_sel;
    for (int i = threadIdx.x; i < 1024; i += FILT_THREADS) {
        const int count = hist[i];
        if (seg_hist) seg_hist[blockIdx.x * 1024 + i] = count;
        if (count != 0) atomicAdd(&h4[i], count);
    }
}

#define FILTER_KERNELS(NAME, THREADS) \
extern "C" __global__ void NAME##_stats( \
    const int *keys, const int *tabu, int n, int round, int lottery_min, \
    int *block_max, int *stats, const unsigned long long *gr) { \
    filt_stats<THREADS>(keys, tabu, n, round, lottery_min, block_max, stats, gr); \
} \
extern "C" __global__ void NAME##_pick4( \
    const int *keys, const int *tabu, int n, int round, int lottery_min, \
    const int *block_max, const int *stats, unsigned long long rng0, \
    unsigned long long lcg_c, unsigned long long num, unsigned long long den0, \
    unsigned long long den1, unsigned long long den2, unsigned long long den3, \
    unsigned long long *out, int *block_sel, int *h4, const unsigned long long *gr) { \
    filt_pick4<THREADS>(keys, tabu, n, round, lottery_min, block_max, stats, rng0, \
                       lcg_c, num, den0, den1, den2, den3, out, block_sel, h4, gr); \
}
FILTER_KERNELS(mica_filt, 256)
FILTER_KERNELS(mica_sx, 64)
#undef FILTER_KERNELS

#define FILTER_MB_KERNEL(NAME, THREADS) \
extern "C" __global__ void NAME##_pick4_mb( \
    const int *keys, const int *tabu, int n, int round, int lottery_min, \
    const int *block_max, const int *stats, unsigned long long rng0, \
    unsigned long long lcg_c, unsigned long long num, unsigned long long den0, \
    unsigned long long den1, unsigned long long den2, unsigned long long den3, \
    unsigned long long *out, int *block_sel, int *h4, const unsigned long long *gr, \
    int *seg_hist, int *tile_hist, int *tgt_hist, int tiles) { \
    if (gr && gr[0] != 0) return; \
    filt_pick4<THREADS>(keys, tabu, n, round, lottery_min, block_max, stats, rng0, \
                       lcg_c, num, den0, den1, den2, den3, out, block_sel, h4, gr, seg_hist); \
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < tiles * 3 * 256; \
         i += gridDim.x * THREADS) tile_hist[i] = 0; \
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < tiles * 64; \
         i += gridDim.x * THREADS) tgt_hist[i] = 0; \
}
FILTER_MB_KERNEL(mica_filt, 256)
FILTER_MB_KERNEL(mica_sx, 64)
#undef FILTER_MB_KERNEL

__device__ __forceinline__ unsigned long long pick_word(unsigned long long w) {
    const unsigned int low = __byte_perm((unsigned int)w, 0u, 0x3210);
    return ((unsigned long long)low << 32) | (unsigned long long)(~(unsigned int)(w >> 32));
}

constexpr int SE = 8;
constexpr int STILE = S_THREADS * SE;

struct __align__(16) SShared {
    unsigned short wcount[32 * 256];
    int sh[32];
    int h4[4 * 256];
    int run[256];
    int ttot[256];
    int quota[64];
    int wsum[32];
    int skipf[4];
    int carry;
};

__device__ void s_sort_select(SShared &S, int m, unsigned long long *__restrict__ buf,
                              unsigned long long *__restrict__ sorted,
                              const int *__restrict__ nodes_in_part, int num_parts,
                              int max_part_size, int slack, int limit, int extra,
                              unsigned long long *__restrict__ picks) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;
    if (tid < 4) S.skipf[tid] = 0;
    const int hv = S.h4[tid];
    int tot;
    const int hoff = blk_scan<S_THREADS>(hv, S.sh, &tot) - (tid >> 8) * m;
    S.h4[tid] = hoff;
    if (hv == m && m > 0) S.skipf[tid >> 8] = 1;
    __syncthreads();
    const int L = (!S.skipf[0]) + (!S.skipf[1]) + (!S.skipf[2]) + (!S.skipf[3]);
    unsigned long long *cur = sorted;
    if ((L & 1) && m > STILE) {
        for (int i = tid; i < m; i += S_THREADS)
            buf[i] = sorted[i];
        __syncthreads();
        cur = buf;
    }
    int j = 0;
    for (int p = 0; p < 4; p++) {
        if (S.skipf[p]) continue;
        unsigned long long *dst = ((L - 1 - j) & 1) ? buf : sorted;
        if (tid < 256) S.run[tid] = S.h4[p * 256 + tid];
        for (int base = 0; base < m; base += STILE) {
            const int wb = base + warp * (32 * SE) + lane;
            unsigned long long x[SE];
#pragma unroll
            for (int e = 0; e < SE; e++) {
                const int i = wb + e * 32;
                x[e] = (i < m) ? cur[i] : 0ull;
            }
            reinterpret_cast<uint4 *>(S.wcount)[tid] = make_uint4(0u, 0u, 0u, 0u);
            __syncthreads();
            unsigned short *wc = S.wcount + warp * 256;
            int rk[SE];
#pragma unroll
            for (int e = 0; e < SE; e++) {
                const int d = (wb + e * 32 < m) ? s_digit(x[e], p) : 256;
                const unsigned int peers = __match_any_sync(0xffffffffu, d);
                const int r = (d < 256) ? (int)wc[d] : 0;
                __syncwarp();
                if (d < 256 && (peers & lanemask_lt) == 0u)
                    wc[d] = (unsigned short)(r + __popc(peers));
                __syncwarp();
                rk[e] = r + __popc(peers & lanemask_lt);
            }
            __syncthreads();
            if (tid < 256) {
                int a = 0;
#pragma unroll
                for (int ww = 0; ww < 32; ww++) {
                    const int c = S.wcount[ww * 256 + tid];
                    S.wcount[ww * 256 + tid] = (unsigned short)a;
                    a += c;
                }
                S.ttot[tid] = a;
            }
            __syncthreads();
#pragma unroll
            for (int e = 0; e < SE; e++) {
                if (wb + e * 32 < m) {
                    const int d = s_digit(x[e], p);
                    dst[S.run[d] + (int)wc[d] + rk[e]] = x[e];
                }
            }
            __syncthreads();
            if (tid < 256) S.run[tid] += S.ttot[tid];
        }
        cur = dst;
        j++;
    }

    const int k_base = min(m, limit);
    const int kc = min(m, k_base + extra);
    if (tid < 64) {
        int q = 0;
        if (tid < num_parts) {
            int fr = max_part_size - nodes_in_part[tid];
            if (fr < 0) fr = 0;
            q = max(1, fr + slack);
        }
        S.quota[tid] = q;
        S.run[tid] = 0;
    }
    int carry = 0;
    for (int base = 0; base < kc; base += STILE) {
        const int wb = base + warp * (32 * SE) + lane;
        unsigned long long x[SE];
#pragma unroll
        for (int e = 0; e < SE; e++) {
            const int i = wb + e * 32;
            x[e] = (i < kc) ? sorted[i] : 0ull;
        }
        reinterpret_cast<unsigned int *>(S.wcount)[tid] = 0u;
        __syncthreads();
        unsigned short *wc = S.wcount + warp * 64;
        int rk[SE];
#pragma unroll
        for (int e = 0; e < SE; e++) {
            const int t = (wb + e * 32 < kc) ? ((int)(~(unsigned int)(x[e] >> 32)) & 63) : 64;
            const unsigned int peers = __match_any_sync(0xffffffffu, t);
            const int r = (t < 64) ? (int)wc[t] : 0;
            __syncwarp();
            if (t < 64 && (peers & lanemask_lt) == 0u) wc[t] = (unsigned short)(r + __popc(peers));
            __syncwarp();
            rk[e] = r + __popc(peers & lanemask_lt);
        }
        __syncthreads();
        if (tid < 64) {
            int a = 0;
#pragma unroll
            for (int ww = 0; ww < 32; ww++) {
                const int c = S.wcount[ww * 64 + tid];
                S.wcount[ww * 64 + tid] = (unsigned short)a;
                a += c;
            }
            S.ttot[tid] = a;
        }
        __syncthreads();
        unsigned int bal[SE];
        int wtot = 0;
#pragma unroll
        for (int e = 0; e < SE; e++) {
            int acc = 0;
            if (wb + e * 32 < kc) {
                const int t = (int)(~(unsigned int)(x[e] >> 32)) & 63;
                if (t < num_parts) acc = (S.run[t] + (int)wc[t] + rk[e] < S.quota[t]) ? 1 : 0;
            }
            bal[e] = __ballot_sync(0xffffffffu, acc);
            wtot += __popc(bal[e]);
        }
        if (lane == 0) S.wsum[warp] = wtot;
        __syncthreads();
        const int ws = S.wsum[lane];
        int inc = ws;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += y;
        }
        int pos = carry + __shfl_sync(0xffffffffu, inc - ws, warp);
        const int ttl = __shfl_sync(0xffffffffu, inc, 31);
#pragma unroll
        for (int e = 0; e < SE; e++) {
            if ((bal[e] >> lane) & 1u) {
                const int ps = pos + __popc(bal[e] & lanemask_lt);
                if (ps < k_base) picks[2 + ps] = pick_word(x[e]);
            }
            pos += __popc(bal[e]);
        }
        carry += ttl;
        __syncthreads();
        if (tid < 64) S.run[tid] += S.ttot[tid];
    }
    const int total_acc = carry;
    if (total_acc == 0) {
        int take = min(k_base, kc);
        for (int i = tid; i < take; i += S_THREADS) {
            unsigned long long w = sorted[i];
            picks[2 + i] = pick_word(w);
        }
        if (tid == 0) {
            picks[0] = (unsigned long long)take | (1ull << 32);
            picks[1] = (unsigned long long)m;
        }
    } else if (tid == 0) {
        picks[0] = (unsigned long long)min(total_acc, k_base);
        picks[1] = (unsigned long long)m;
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    mica_sortsel(const unsigned long long *__restrict__ segs, const int *__restrict__ block_sel,
                 int G, int chunk, unsigned long long *__restrict__ buf,
                 unsigned long long *__restrict__ sorted, const int *__restrict__ nodes_in_part,
                 int num_parts, int max_part_size, int slack, int limit, int extra,
                 unsigned long long *__restrict__ picks, int *__restrict__ h4,
                 const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        limit = (int)gr[14];
        slack = (int)gr[15];
    }
    __shared__ SShared S;
    __shared__ int seg_off[257];
    const int tid = threadIdx.x;

    int v = (tid < G) ? block_sel[tid] : 0;
    int m;
    int off = blk_scan<S_THREADS>(v, S.sh, &m);
    if (tid < G) seg_off[tid] = off;
    if (tid == 0) seg_off[G] = m;
    for (int i = tid; i < 1024; i += S_THREADS)
        S.h4[i] = h4[i];
    __syncthreads();
    for (int i = tid; i < 1024; i += S_THREADS)
        h4[i] = 0;
    for (int base = 0; base < m; base += S_THREADS) {
        int i = base + tid;
        bool valid = i < m;
        unsigned long long w = 0ull;
        if (valid) {
            int a = 0, b = G;
            while (b - a > 1) {
                int mid = (a + b) >> 1;
                if (seg_off[mid] <= i)
                    a = mid;
                else
                    b = mid;
            }
            unsigned long long o = segs[a * chunk + (i - seg_off[a])];
            unsigned int key = (unsigned int)(o & 0xFFFFFFFFull);
            unsigned int node = (unsigned int)(o >> 32);
            w = ((unsigned long long)(~key) << 32) | (unsigned long long)node;
            sorted[i] = w;
        }
    }
    __syncthreads();
    s_sort_select(S, m, buf, sorted, nodes_in_part, num_parts, max_part_size, slack, limit, extra,
                  picks);
}

constexpr int MB_THREADS = 256;
constexpr int MB_TILE = 2048;
constexpr int MB_E = MB_TILE / MB_THREADS;
constexpr int MB_WARPS = MB_THREADS / 32;
static_assert(MB_TILE == MB_THREADS * MB_E && MB_TILE < 65536, "Invalid radix tile");

template <int STRIDE>
__device__ __forceinline__ int mb_hist_sum(const int *hist, int begin, int end, int digit) {
    int sums[8] = {};
    for (int t = begin; t < end; t += 8) {
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (t + k < end) sums[k] += hist[(t + k) * STRIDE + digit];
    }
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
           ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

extern "C" __global__ void __launch_bounds__(MB_THREADS)
    mica_sortmb_pass(const unsigned long long *__restrict__ segs,
                     const int *__restrict__ block_sel, int G, int chunk, int tiles,
                     unsigned long long *__restrict__ buf,
                     unsigned long long *__restrict__ sorted, const int *__restrict__ h4,
                     const int *__restrict__ seg_hist, int *__restrict__ tile_hist,
                     int *__restrict__ tgt_hist, int p, int slack, int limit,
                     const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        limit = (int)gr[14];
        slack = (int)gr[15];
    }
    __shared__ int sh[MB_WARPS];
    __shared__ int skip[4];
    __shared__ int run[256];
    __shared__ int total[256];
    __shared__ unsigned short wcount[MB_WARPS * 256];
    const int tid = threadIdx.x, b = blockIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;
    const int m = blk_sum<MB_THREADS>(tid < G ? block_sel[tid] : 0, sh);
    if (m == 0) return;
    if (tid < 4) skip[tid] = 0;
    __syncthreads();
    for (int d = 0; d < 4; d++)
        if (h4[d * 256 + tid] == m) skip[d] = 1;
    __syncthreads();
    int active = 0;
    for (int d = 0; d < 4; d++)
        if (!skip[d]) active |= 1 << d;
    if (active == 0) {
        if (p != 0 || b >= G) return;
        int sum;
        run[tid] = blk_scan<MB_THREADS>(tid < G ? block_sel[tid] : 0, sh, &sum);
        __syncthreads();
        for (int t = b; t < G; t += gridDim.x) {
            const int off = run[t];
            for (int i = tid; i < block_sel[t]; i += MB_THREADS) {
                const unsigned long long o = segs[t * chunk + i];
                const unsigned int key = (unsigned int)o;
                sorted[off + i] = ((unsigned long long)(~key) << 32) | (o >> 32);
                atomicAdd(&tgt_hist[((off + i) / MB_TILE) * 64 + (key & 63u)], 1);
            }
        }
        return;
    }
    if (!(active & (1 << p))) return;
    const int j = __popc((unsigned int)(active & ((1 << p) - 1)));
    const int L = __popc((unsigned int)active);
    int next = p + 1;
    while (next < 4 && !(active & (1 << next))) next++;
    const bool first = j == 0;
    const int count_tiles = first ? G : (m + MB_TILE - 1) / MB_TILE;
    if (b >= count_tiles) return;
    int sum;
    int off = blk_scan<MB_THREADS>(h4[p * 256 + tid], sh, &sum);
    int prev = 0;
    unsigned long long *dst = ((L - 1 - j) & 1) ? buf : sorted;
    const unsigned long long *src = ((L - 1 - j) & 1) ? sorted : buf;
    for (int t = b; t < count_tiles; t += gridDim.x) {
        off += first ? mb_hist_sum<1024>(seg_hist + p * 256, prev, t, tid)
                     : mb_hist_sum<256>(tile_hist + (p - 1) * tiles * 256, prev, t, tid);
        prev = t;
        const int count = first ? block_sel[t] : min(MB_TILE, m - t * MB_TILE);
        if (count == 0) continue;
        run[tid] = off;
        for (int base = 0; base < count; base += MB_TILE) {
            const int wb = base + warp * (32 * MB_E) + lane;
            unsigned long long x[MB_E];
#pragma unroll
            for (int e = 0; e < MB_E; e++) {
                const int i = wb + e * 32;
                unsigned long long w = 0ull;
                if (i < count) {
                    if (first) {
                        const unsigned long long o = segs[t * chunk + i];
                        w = ((unsigned long long)(~(unsigned int)o) << 32) | (o >> 32);
                    } else {
                        w = src[t * MB_TILE + i];
                    }
                }
                x[e] = w;
            }
            for (int i = tid; i < MB_WARPS * 256; i += MB_THREADS) wcount[i] = 0;
            __syncthreads();
            unsigned short *wc = wcount + warp * 256;
            int rk[MB_E];
#pragma unroll
            for (int e = 0; e < MB_E; e++) {
                const int d = wb + e * 32 < count ? s_digit(x[e], p) : 256;
                const unsigned int peers = __match_any_sync(0xffffffffu, d);
                const int r = d < 256 ? (int)wc[d] : 0;
                __syncwarp();
                if (d < 256 && (peers & lanemask_lt) == 0u)
                    wc[d] = (unsigned short)(r + __popc(peers));
                __syncwarp();
                rk[e] = r + __popc(peers & lanemask_lt);
            }
            __syncthreads();
            int a = 0;
#pragma unroll
            for (int ww = 0; ww < MB_WARPS; ww++) {
                const int c = wcount[ww * 256 + tid];
                wcount[ww * 256 + tid] = (unsigned short)a;
                a += c;
            }
            total[tid] = a;
            __syncthreads();
#pragma unroll
            for (int e = 0; e < MB_E; e++) {
                if (wb + e * 32 < count) {
                    const int d = s_digit(x[e], p);
                    const int pos = run[d] + (int)wc[d] + rk[e];
                    dst[pos] = x[e];
                    if (next < 4)
                        atomicAdd(&tile_hist[((next - 1) * tiles + pos / MB_TILE) * 256 +
                                             s_digit(x[e], next)], 1);
                    else
                        atomicAdd(&tgt_hist[(pos / MB_TILE) * 64 +
                                            ((~(unsigned int)(x[e] >> 32)) & 63u)], 1);
                }
            }
            __syncthreads();
            run[tid] += total[tid];
        }
    }
}

extern "C" __global__ void __launch_bounds__(MB_THREADS)
    mica_sortmb_select(const int *__restrict__ block_sel, int G,
                       const unsigned long long *__restrict__ sorted,
                       const int *__restrict__ nodes_in_part, int num_parts,
                       int max_part_size, int slack, int limit, int extra,
                       unsigned long long *__restrict__ picks, int *__restrict__ h4,
                       const int *__restrict__ tgt_hist, const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        limit = (int)gr[14];
        slack = (int)gr[15];
    }
    __shared__ int sh[MB_WARPS];
    __shared__ int tail[64];
    __shared__ int quota[64];
    __shared__ int prefix[64];
    __shared__ int wsum[MB_WARPS];
    __shared__ unsigned short wcount[MB_WARPS * 64];
    const int tid = threadIdx.x, b = blockIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;
    const int m = blk_sum<MB_THREADS>(tid < G ? block_sel[tid] : 0, sh);
    for (int i = b * MB_THREADS + tid; i < 1024; i += gridDim.x * MB_THREADS) h4[i] = 0;
    const int k_base = min(m, limit), kc = min(m, k_base + extra);
    if (kc == 0) {
        if (b == 0 && tid == 0) {
            picks[0] = 1ull << 32;
            picks[1] = (unsigned long long)m;
        }
        return;
    }
    if (b * MB_TILE >= kc) return;
    const int full = kc / MB_TILE;
    if (tid < 64) {
        tail[tid] = 0;
        int q = 0;
        if (tid < num_parts) {
            const int fr = max(0, max_part_size - nodes_in_part[tid]);
            q = max(1, fr + slack);
        }
        quota[tid] = q;
    }
    __syncthreads();
    for (int i = full * MB_TILE + tid; i < kc; i += MB_THREADS) {
        const int t = (int)(~(unsigned int)(sorted[i] >> 32)) & 63;
        atomicAdd(&tail[t], 1);
    }
    __syncthreads();
    int prev = 0, scanned = 0;
    for (int t = b; t * MB_TILE < kc; t += gridDim.x) {
        int before = 0;
        if (tid < 64) {
            prev += mb_hist_sum<64>(tgt_hist, scanned, t, tid);
            prefix[tid] = prev;
            before = min(prev, quota[tid]);
        }
        scanned = t;
        const int A = blk_sum<MB_THREADS>(before, sh);
        const int base = t * MB_TILE;
        const int wb = base + warp * (32 * MB_E) + lane;
        unsigned long long x[MB_E];
#pragma unroll
        for (int e = 0; e < MB_E; e++)
            x[e] = wb + e * 32 < kc ? sorted[wb + e * 32] : 0ull;
        for (int i = tid; i < MB_WARPS * 64; i += MB_THREADS) wcount[i] = 0;
        __syncthreads();
        unsigned short *wc = wcount + warp * 64;
        int rk[MB_E];
#pragma unroll
        for (int e = 0; e < MB_E; e++) {
            const int x_tgt = wb + e * 32 < kc ? ((int)(~(unsigned int)(x[e] >> 32)) & 63) : 64;
            const unsigned int peers = __match_any_sync(0xffffffffu, x_tgt);
            const int r = x_tgt < 64 ? (int)wc[x_tgt] : 0;
            __syncwarp();
            if (x_tgt < 64 && (peers & lanemask_lt) == 0u)
                wc[x_tgt] = (unsigned short)(r + __popc(peers));
            __syncwarp();
            rk[e] = r + __popc(peers & lanemask_lt);
        }
        __syncthreads();
        if (tid < 64) {
            int a = 0;
#pragma unroll
            for (int ww = 0; ww < MB_WARPS; ww++) {
                const int c = wcount[ww * 64 + tid];
                wcount[ww * 64 + tid] = (unsigned short)a;
                a += c;
            }
        }
        __syncthreads();
        unsigned int ballots[MB_E];
        int wtot = 0;
#pragma unroll
        for (int e = 0; e < MB_E; e++) {
            bool acc = false;
            if (wb + e * 32 < kc) {
                const int x_tgt = (int)(~(unsigned int)(x[e] >> 32)) & 63;
                acc = x_tgt < num_parts && prefix[x_tgt] + (int)wc[x_tgt] + rk[e] < quota[x_tgt];
            }
            ballots[e] = __ballot_sync(0xffffffffu, acc);
            wtot += __popc(ballots[e]);
        }
        if (lane == 0) wsum[warp] = wtot;
        __syncthreads();
        const int ws = lane < MB_WARPS ? wsum[lane] : 0;
        int inc = ws;
        for (int o = 1; o < MB_WARPS; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += y;
        }
        int pos = A + __shfl_sync(0xffffffffu, inc - ws, warp);
#pragma unroll
        for (int e = 0; e < MB_E; e++) {
            if ((ballots[e] >> lane) & 1u) {
                const int ps = pos + __popc(ballots[e] & lanemask_lt);
                if (ps < k_base) picks[2 + ps] = pick_word(x[e]);
            }
            pos += __popc(ballots[e]);
        }
        __syncthreads();
    }
    int accepted = 0;
    if (tid < 64) {
        const int c = prev + mb_hist_sum<64>(tgt_hist, scanned, full, tid) + tail[tid];
        accepted = min(c, quota[tid]);
    }
    const int total_acc = blk_sum<MB_THREADS>(accepted, sh);
    if (b == 0 && tid == 0) {
        picks[0] = total_acc == 0
                       ? ((unsigned long long)min(k_base, kc) | (1ull << 32))
                       : (unsigned long long)min(total_acc, k_base);
        picks[1] = (unsigned long long)m;
    }
    if (total_acc == 0) {
        for (int t = b; t * MB_TILE < min(k_base, kc); t += gridDim.x)
            for (int i = t * MB_TILE + tid; i < min((t + 1) * MB_TILE, min(k_base, kc));
                 i += MB_THREADS)
                picks[2 + i] = pick_word(sorted[i]);
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    mica_pick1(const int *__restrict__ keys, const int *__restrict__ tabu_until, int n, int round,
               int gmin, unsigned long long *__restrict__ buf,
               unsigned long long *__restrict__ sorted, const int *__restrict__ nodes_in_part,
               int num_parts, int max_part_size, int slack, int limit, int extra,
               unsigned long long *__restrict__ picks, const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        round = (int)gr[2];
        int r = (int)gr[1], L = (int)gr[14];
        limit = r < 50 ? L / 2 : r < 200 ? L : L / 3;
        slack = r < 64 ? 8 : r < 256 ? 4 : 2;
        if (gr[15]) {
            round = 2147483647;
            gmin = 0;
            limit = L;
            slack = (int)gr[3];
            extra = (int)gr[7];
        }
    }
    __shared__ SShared S;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;

    int mx = -2147483647 - 1;
    for (int i = tid; i < n; i += S_THREADS) {
        int key = keys[i];
        if (biased_key_valid(key) && biased_key_gain(key) >= gmin)
            mx = max(mx, biased_key_gain(key));
    }
    int max_gain = blk_max<S_THREADS>(mx, S.sh);
    if (max_gain == (-2147483647 - 1)) max_gain = 0;
    int aspiration = max(1, (max_gain * 3) / 4);
    for (int i = tid; i < 1024; i += S_THREADS)
        S.h4[i] = 0;
    if (tid == 0) S.carry = 0;
    __syncthreads();

    for (int base = 0; base < n; base += S_THREADS) {
        int i = base + tid;
        int key = 0x80000000;
        int sel = 0;
        if (i < n) {
            key = keys[i];
            if (biased_key_valid(key) && biased_key_gain(key) >= gmin) {
                int gain = biased_key_gain(key);
                sel = !((tabu_until[i] > round) && (gain < aspiration));
            }
        }
        int ok = sel ? key : 0;
        int total_sel;
        int pos = blk_scan<S_THREADS>(sel, S.sh, &total_sel);
        unsigned long long w =
            ((unsigned long long)(~(unsigned int)ok) << 32) | (unsigned long long)(unsigned int)i;
        if (sel) sorted[S.carry + pos] = w;
        s_hist_add(S.h4, w, sel != 0, lanemask_lt);
        __syncthreads();
        if (tid == 0) S.carry += total_sel;
        __syncthreads();
    }
    const int m = S.carry;
    __syncthreads();
    s_sort_select(S, m, buf, sorted, nodes_in_part, num_parts, max_part_size, slack, limit, extra,
                  picks);
}

}


namespace grouped {
using U = unsigned long long;
__device__ __forceinline__ U satmul(U a, U b) { return __umul64hi(a, b) ? ~0ull : a * b; }
__device__ __forceinline__ void schedule(U *h, const U *c) {
    U r = h[1], left = c[5] > r ? c[5] - r : 0, span = c[5];
    U num = left, den = span;
    if (c[6] >= 2) {
        num = satmul(num, left);
        den = satmul(den, span);
    }
    if (c[6] == 3) {
        num = satmul(num, left);
        den = satmul(den, span);
    }
    den = satmul(den, 100) / c[7];
    h[16] = num;
    h[17] = den;
    h[18] = satmul(den, 6);
    h[19] = satmul(den, 11);
    h[20] = satmul(den, 16);
    h[14] = r < 50 ? c[9] / 2 : r < 200 ? c[9] * 3 / 4 : c[9] / 4;
    h[15] = (r < 64 ? 8 : r < 256 ? 4 : 2) * c[11];
    h[8] = c[16] && !h[4] && h[6] < c[3] ? 1 : 0;
}
}

__device__ __forceinline__ int scan256(int v, int *sh, int *total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o *= 2) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) sh[warp] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int w = 0; w < 8; ++w) {
            int t = sh[w];
            sh[w] = a;
            a += t;
        }
        *total = a;
    }
    __syncthreads();
    int out = sh[warp] + x - v;
    __syncthreads();
    return out;
}

__device__ __forceinline__ void scan256_pair(int v, int d, int *sh, int &pos, int &off, int &ntotal,
                                             int &dtotal) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v, y = d;
    for (int step = 1; step < 32; step *= 2) {
        int a = __shfl_up_sync(0xffffffffu, x, step);
        int b = __shfl_up_sync(0xffffffffu, y, step);
        if (lane >= step) {
            x += a;
            y += b;
        }
    }
    if (lane == 31) {
        sh[warp] = x;
        sh[8 + warp] = y;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0, b = 0;
        for (int w = 0; w < 8; ++w) {
            int t = sh[w], u = sh[8 + w];
            sh[w] = a;
            sh[8 + w] = b;
            a += t;
            b += u;
        }
        sh[16] = a;
        sh[17] = b;
    }
    __syncthreads();
    ntotal = sh[16];
    dtotal = sh[17];
    pos = sh[warp] + x - v;
    off = sh[8 + warp] + y - d;
    __syncthreads();
}

extern "C" __global__ void grouped_resume(unsigned long long *h) {
    if (threadIdx.x == 0 && h[0] == 1) {
        h[0] = 0;
    }
}

__device__ __forceinline__ int grouped_warp_arbitrate(int s, int t, int length, int cap,
                                                     int &lo, int &hi) {
    const int lane = threadIdx.x & 31;
    const bool valid = lane < length;
    const unsigned int before = (1u << lane) - 1u;
    __shared__ unsigned int inmask[64], outmask[64];
    inmask[lane] = inmask[lane + 32] = 0u;
    outmask[lane] = outmask[lane + 32] = 0u;
    __syncwarp();
    const unsigned int active = __ballot_sync(0xffffffffu, valid);
    const unsigned int moving = __ballot_sync(0xffffffffu, valid && s != t);
    const unsigned int incoming = __match_any_sync(0xffffffffu, t) & moving;
    const unsigned int outgoing = __match_any_sync(0xffffffffu, s) & moving;
    if (valid && s != t) {
        if ((incoming & before) == 0u) inmask[t] = incoming;
        if ((outgoing & before) == 0u) outmask[s] = outgoing;
    }
    __syncwarp();
    const unsigned int si = inmask[s] & before, so = outmask[s] & before;
    const unsigned int ti = inmask[t] & before, to = outmask[t] & before;
    const int sl = __shfl_sync(0xffffffffu, lo, s & 31);
    const int sh = __shfl_sync(0xffffffffu, hi, s & 31);
    const int tl = __shfl_sync(0xffffffffu, lo, t & 31);
    const int th = __shfl_sync(0xffffffffu, hi, t & 31);
    const int cs = s < 32 ? sl : sh, ts = t < 32 ? tl : th;
    unsigned int decisions = active;
    for (int iteration = 0; iteration < length; ++iteration) {
        const int size_s = cs + __popc(si & decisions) - __popc(so & decisions);
        const int size_t = ts + __popc(ti & decisions) - __popc(to & decisions);
        const unsigned int next = __ballot_sync(0xffffffffu, valid && size_t < cap && size_s > 1);
        if (next == decisions) break;
        decisions = next;
    }
    lo += __popc(inmask[lane] & decisions) - __popc(outmask[lane] & decisions);
    hi += __popc(inmask[lane + 32] & decisions) - __popc(outmask[lane + 32] & decisions);
    __syncwarp();
    return (decisions >> lane) & 1u;
}

template <bool COUNTS>
__device__ __forceinline__ void grouped_finish_impl(
    unsigned long long *h, const unsigned long long *c, const unsigned long long *picks,
    const unsigned long long *sorted, int *part, int *sizes_out, int *tabu,
    const int *offsets, int *pairs) {
    __shared__ int active;
    if (threadIdx.x == 0) active = h[0] == 0;
    __syncthreads();
    if (!active) return;
    const int tid = threadIdx.x, lane = tid & 31;
    const int n = (int)c[0], np = (int)c[1], cap = (int)c[2], r = (int)h[1];
    const int m = min((unsigned int)picks[1], (unsigned int)n);
    const int k = min((unsigned int)picks[0], (unsigned int)m);
    const int kb = min(m, (int)h[14]), kc = min(m, kb + (int)c[10]);
    __shared__ int node[256], source[256], target[256], take[256];
    __shared__ int incoming[64], outgoing[64], sizes[64], ok;
    __shared__ int sh[18], total, exec, listed;
    __shared__ int unsafe_part[64], seq[256];
    if (tid == 0) {
        h[4] = 0;
    }
    if (m == 0) {
        if (tid == 0) {
            h[0] = 2;
            h[1] = r + 1;
            grouped::schedule(h, c);
        }
        return;
    }
    if (tid == 0) {
        exec = 0;
        listed = 0;
    }
    int lo = 0, hi = 0;
    if (tid < 32) {
        lo = lane < np ? sizes_out[lane] : 0;
        hi = lane + 32 < np ? sizes_out[lane + 32] : 0;
    }
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
        if (pass && (exec != 0 || kc <= kb)) break;
        if (pass) {
            for (int i = tid; i < min(k, (int)c[13]); i += 256) {
                int v = (int)(unsigned int)(picks[2 + i] >> 32);
                if (v >= 0 && v < n) tabu[v] = r + (int)c[14];
            }
            __syncthreads();
        }
        int count = pass ? min(kc - kb, kb) : k;
        for (int base = 0; base < count; base += 256) {
            int length = min(256, count - base);
            {
                int v = -1, t = -1, s = -1;
                if (tid < length) {
                    unsigned long long word =
                        pass ? sorted[kb + base + tid] : picks[2 + base + tid];
                    v = pass ? (int)(unsigned int)word : (int)(unsigned int)(word >> 32);
                    t = pass ? (int)(~(unsigned int)(word >> 32) & 63u)
                             : (int)((unsigned int)word & 63u);
                    if (v >= 0 && v < n && t < np) {
                        s = part[v];
                        if (s < 0 || s >= np) s = -1;
                    }
                }
                node[tid] = v;
                target[tid] = t;
                source[tid] = s;
                take[tid] = 0;
            }
            if (tid < 64) {
                incoming[tid] = 0;
                outgoing[tid] = 0;
            }
            if (tid < 32) {
                sizes[lane] = lo;
                sizes[lane + 32] = hi;
            }
            if (tid == 0) ok = 1;
            __syncthreads();
            if (tid < length && source[tid] >= 0) {
                atomicAdd(incoming + target[tid], 1);
                atomicAdd(outgoing + source[tid], 1);
            }
            __syncthreads();
            if (tid < 64 && (incoming[tid] || outgoing[tid]) &&
                (incoming[tid] > cap - sizes[tid] || outgoing[tid] > sizes[tid] - 1))
                atomicExch(&ok, 0);
            __syncthreads();
            if (ok) {
                if (tid < length) take[tid] = source[tid] >= 0;
                if (tid < 32) {
                    lo += incoming[lane] - outgoing[lane];
                    hi += incoming[lane + 32] - outgoing[lane + 32];
                }
            } else {
                __syncthreads();
                if (tid < 64) {
                    unsafe_part[tid] =
                        (incoming[tid] || outgoing[tid]) &&
                        (incoming[tid] > cap - sizes[tid] || outgoing[tid] > sizes[tid] - 1);
                }
                __syncthreads();
                int crit = 0;
                if (tid < length && source[tid] >= 0) {
                    crit = unsafe_part[source[tid]] | unsafe_part[target[tid]];
                    if (!crit) take[tid] = 1;
                }
                int cpos = scan256(crit, sh, &total);
                const int ccount = total;
                if (crit) seq[cpos] = source[tid] | (target[tid] << 6);
                __syncthreads();
                if (tid < 32) {
                    for (int j = 0; j < ccount; j += 32) {
                        const int length = min(32, ccount - j);
                        const int move = lane < length ? seq[j + lane] : 0;
                        const int yes = grouped_warp_arbitrate(move & 63, move >> 6,
                                                              length, cap, lo, hi);
                        if (lane < length) seq[j + lane] = yes;
                    }
                    sizes[lane] = lo;
                    sizes[lane + 32] = hi;
                }
                if (tid < 64) {
                    incoming[tid] = 0;
                    outgoing[tid] = 0;
                }
                __syncthreads();
                if (crit) take[tid] = seq[cpos];
                if (tid < length && source[tid] >= 0 && !crit) {
                    atomicAdd(incoming + target[tid], 1);
                    atomicAdd(outgoing + source[tid], 1);
                }
                __syncthreads();
                if (tid < 32) {
                    lo = sizes[lane] + incoming[lane] - outgoing[lane];
                    hi = sizes[lane + 32] + incoming[lane + 32] - outgoing[lane + 32];
                }
            }
            __syncthreads();
            int yes = tid < length ? take[tid] : 0;
            int pos = 0, ntotal = 0, off = 0, dtotal = 0;
            if (c[16]) {
                int degree = yes ? offsets[node[tid] + 1] - offsets[node[tid]] : 0;
                scan256_pair(yes, degree, sh, pos, off, ntotal, dtotal);
            } else {
                pos = scan256(yes, sh, &total);
                ntotal = total;
            }
            if (yes) {
                int v = node[tid];
                part[v] = target[tid];
                tabu[v] = r + (int)c[12];
                if (c[16]) {
                    pairs[2 * (exec + pos)] = v;
                    pairs[2 * (exec + pos) + 1] = listed + off;
                    if (COUNTS) pairs[2 * n + exec + pos] = source[tid];
                }
            }
            __syncthreads();
            if (tid == 0) {
                exec += ntotal;
                listed += dtotal;
            }
            __syncthreads();
        }
    }
    if (tid < 32) {
        sizes[lane] = lo;
        sizes[lane + 32] = hi;
    }
    __syncthreads();
    if (tid < np) sizes_out[tid] = sizes[tid];
    __syncthreads();
    if (tid == 0) {
        h[5] = exec;
        h[6] = listed;
        if (exec)
            h[7] = 0;
        else {
            ++h[7];
            if (h[7] >= 3 && (unsigned long long)r < (c[8] > 50 ? c[8] - 50 : 0)) {
                unsigned long long state = 987654321ull + (unsigned long long)r * 123456789ull;
                int made = 0, goal = n * 3 / 100;
                for (int attempt = 0; attempt < n && made < goal; ++attempt) {
                    state = state * 6364136223846793005ull + 1442695040888963407ull;
                    int v = (int)(state % (unsigned long long)n), s = part[v];
                    if (s < 0 || s >= np || sizes[s] <= 1) continue;
                    state = state * 6364136223846793005ull + 1442695040888963407ull;
                    int t = (int)(state % (unsigned long long)np);
                    if (t != s && sizes[t] < cap) {
                        part[v] = t;
                        --sizes[s];
                        ++sizes[t];
                        ++made;
                    }
                }
                for (int b = 0; b < np; ++b)
                    sizes_out[b] = sizes[b];
                h[4] = 1;
                h[7] = 0;
            } else if (h[7] > c[15]) {
                h[0] = 2;
            }
        }
        h[1] = r + 1;
        if (h[0] != 2 && h[1] >= c[4]) {
            h[0] = 2;
        }
        grouped::schedule(h, c);
    }
}

extern "C" __global__ void __launch_bounds__(256)
    grouped_finish(unsigned long long *h, const unsigned long long *c,
                   const unsigned long long *picks, const unsigned long long *sorted, int *part,
                   int *sizes_out, int *tabu, const int *offsets, int *pairs) {
    grouped_finish_impl<false>(h, c, picks, sorted, part, sizes_out, tabu, offsets, pairs);
}

extern "C" __global__ void __launch_bounds__(256)
    grouped_finish_counts_100k(unsigned long long *h, const unsigned long long *c,
                              const unsigned long long *picks, const unsigned long long *sorted,
                              int *part, int *sizes_out, int *tabu, const int *offsets, int *pairs) {
    grouped_finish_impl<true>(h, c, picks, sorted, part, sizes_out, tabu, offsets, pairs);
}


#include <cuda_runtime.h>
__device__ __forceinline__ void grouped_small_finish_impl(
    unsigned long long *h, const unsigned long long *c, const unsigned long long *picks,
    const unsigned long long *sorted, int *part, int *sizes_out, int *tabu) {
    if (h[0]) return;
    int tid = threadIdx.x, lane = tid & 31, n = (int)c[0], np = (int)c[1], cap = (int)c[2];
    int r = (int)h[1], global = (int)h[2];
    int phase = (int)h[15];
    int limit = phase ? (int)c[4] : (r < 50 ? (int)c[4] / 2 : r < 200 ? (int)c[4] : (int)c[4] / 3);
    int m = min((unsigned)picks[1], (unsigned)n), k = min((unsigned)picks[0], (unsigned)m);
    int kb = min(m, limit), kc = min(m, kb + 16384);
    __shared__ int node[256], target[256], source[256], take[256], in[64], out[64], sizes[64], ok,
        executed;
    __shared__ int unsafe_part[64], seq[256], sh[8], total;
    __syncthreads();
    if (!tid) {
        ++h[1];
        executed = 0;
    }
    if (!m) {
        if (!tid) {
            h[0] = 2;
        }
        return;
    }
    int lo = 0, hi = 0;
    if (tid < 32) {
        lo = lane < np ? sizes_out[lane] : 0;
        hi = lane + 32 < np ? sizes_out[lane + 32] : 0;
    }
    __syncthreads();
    for (int pass = 0; pass < (phase ? 1 : 2); ++pass) {
        if (pass && (executed || kc <= kb)) break;
        int count = pass ? min(kc - kb, kb) : k;
        for (int base = 0; base < count; base += 256) {
            int length = min(256, count - base);
            {
                int v = -1, t = -1, s = -1;
                if (tid < length) {
                    unsigned long long w = pass ? sorted[kb + base + tid] : picks[2 + base + tid];
                    v = pass ? (int)(unsigned)w : (int)(unsigned)(w >> 32);
                    t = pass ? (~(unsigned)(w >> 32)) & 63 : (unsigned)w & 63;
                    if ((unsigned)v < (unsigned)n && t < np) {
                        s = part[v];
                        if ((unsigned)s >= (unsigned)np) s = -1;
                    }
                }
                node[tid] = v;
                target[tid] = t;
                source[tid] = s;
                take[tid] = 0;
            }
            if (tid < 64) in[tid] = out[tid] = 0;
            if (tid < 32) {
                sizes[lane] = lo;
                sizes[lane + 32] = hi;
            }
            if (!tid) ok = 1;
            __syncthreads();
            if (tid < length && source[tid] >= 0) {
                atomicAdd(in + target[tid], 1);
                atomicAdd(out + source[tid], 1);
            }
            __syncthreads();
            if (tid < 64 && (in[tid] || out[tid]) &&
                (in[tid] > cap - sizes[tid] || out[tid] > sizes[tid] - 1))
                atomicExch(&ok, 0);
            __syncthreads();
            if (ok) {
                if (tid < length) take[tid] = source[tid] >= 0;
                if (tid < 32) {
                    lo += in[lane] - out[lane];
                    hi += in[lane + 32] - out[lane + 32];
                }
            } else {
                __syncthreads();
                if (tid < 64)
                    unsafe_part[tid] = (in[tid] || out[tid]) &&
                                       (in[tid] > cap - sizes[tid] || out[tid] > sizes[tid] - 1);
                __syncthreads();
                int crit = 0;
                if (tid < length && source[tid] >= 0) {
                    crit = unsafe_part[source[tid]] | unsafe_part[target[tid]];
                    if (!crit) take[tid] = 1;
                }
                int x = crit;
                for (int o = 1; o < 32; o *= 2) {
                    int y = __shfl_up_sync(0xffffffffu, x, o);
                    if (lane >= o) x += y;
                }
                if (lane == 31) sh[tid >> 5] = x;
                __syncthreads();
                if (!tid) {
                    int a = 0;
                    for (int w = 0; w < 8; ++w) {
                        int c = sh[w];
                        sh[w] = a;
                        a += c;
                    }
                    total = a;
                }
                __syncthreads();
                const int ccount = total;
                if (crit) seq[sh[tid >> 5] + x - crit] = tid;
                __syncthreads();
                if (tid < 32)
                    for (int j = 0; j < ccount; j += 32) {
                        const int length = min(32, ccount - j);
                        const int i = lane < length ? seq[j + lane] : 0;
                        const int s = lane < length ? source[i] : 0;
                        const int t = lane < length ? target[i] : 0;
                        const int yes = grouped_warp_arbitrate(s, t, length, cap, lo, hi);
                        if (lane < length) take[i] = yes;
                    }
                if (tid < 64) in[tid] = out[tid] = 0;
                __syncthreads();
                if (tid < length && source[tid] >= 0 && !crit) {
                    atomicAdd(in + target[tid], 1);
                    atomicAdd(out + source[tid], 1);
                }
                __syncthreads();
                if (tid < 32) {
                    lo += in[lane] - out[lane];
                    hi += in[lane + 32] - out[lane + 32];
                }
            }
            __syncthreads();
            int mine = tid < length && take[tid];
            if (mine) part[node[tid]] = target[tid];
            unsigned int votes = __ballot_sync(0xffffffffu, mine);
            if (!lane && votes) atomicAdd(&executed, __popc(votes));
            __syncthreads();
        }
        for (int i = tid; i < executed; i += 256) {
            unsigned long long w = pass ? sorted[kb + i] : picks[2 + i];
            int v = pass ? (int)(unsigned)w : (int)(unsigned)(w >> 32);
            tabu[v] = global + (int)c[5];
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (lane < np) sizes_out[lane] = lo;
        if (lane + 32 < np) sizes_out[lane + 32] = hi;
    }
    __syncthreads();
    if (!tid) {
        if (!executed) {
            if (phase == 1) {
                h[0] = 2;
            } else {
                h[0] = 1;
            }
        } else {
            if (!phase) h[3] = 0;
            if (h[1] >= c[3]) {
                h[0] = 2;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256)
    grouped_small_finish(unsigned long long *h, const unsigned long long *c,
                         const unsigned long long *picks, const unsigned long long *sorted,
                         int *part, int *sizes_out, int *tabu) {
    grouped_small_finish_impl(h, c, picks, sorted, part, sizes_out, tabu);
}

__device__ __forceinline__ void persistent_counts_100k(
    bool refresh, int grid, const unsigned long long *h, int ne, int n, const int *pairs,
    const int *node_hyperedges, const int *node_offsets, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *part, unsigned int *counts, ulonglong2 *flags);
template <int ZGM>
__device__ __forceinline__ void persistent_moves(
    int grid, int n, int np, int cap, const int *node_hyperedges, const int *node_offsets,
    const int *part, const int *sizes_out, const ulonglong2 *flags, int *priorities,
    const int *order, int n8, int n4, int n2);

namespace persistent100k {
using namespace filt;
constexpr int BT = 256;
constexpr int TILES = S_THREADS / BT;
static_assert(TILES == 4 && S_THREADS == 1024 && STILE == 8192, "Invalid logical tile geometry");
static_assert(sizeof(SShared) == 23072, "Unexpected selection storage");
static_assert(BT == 4 * 64 && BT == MB_THREADS, "Invalid persistent block geometry");

__device__ __forceinline__ void barrier(unsigned int *counter, unsigned int &generation) {
    const unsigned int target = ++generation * gridDim.x;
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile(
            "{\n\t"
            ".reg .u32 arrived;\n\t"
            "atom.acq_rel.gpu.global.add.u32 arrived, [%0], 1;\n\t"
            "}\n"
            :: "l"(counter) : "memory");
        unsigned int value;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                         : "=r"(value) : "l"(counter) : "memory");
            if (value < target) __nanosleep(20);
        } while (value < target);
    }
    __syncthreads();
}

__device__ __forceinline__ void stats_phase(int block, int grid, const int *keys,
                                           const int *tabu_until, int n, int round,
                                           int lottery_min, int *block_max,
                                           int *stats, const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        round = (int)gr[1];
    }
    __shared__ int sh[32];
    int chunk = chunk_of(n, grid);
    int lo = block * chunk;
    int hi = min(n, lo + chunk);
    int m = -2147483647 - 1;
    int nt = 0, tb0 = 0, tb1 = 0, tb2 = 0, tb3 = 0;
    for (int i = lo + threadIdx.x; threadIdx.x < 64 && i < hi; i += 64) {
        int key = keys[i];
        if (signed_key_valid(key)) {
            int g = signed_key_gain(key);
            if (g > m) m = g;
            if (g <= 0 && g >= lottery_min) {
                if (tabu_until[i] > round) {
                    tb0 += (g == -3);
                    tb1 += (g == -2);
                    tb2 += (g == -1);
                    tb3 += (g == 0);
                } else {
                    nt++;
                }
            }
        }
    }
    m = blk_max<64>(m, sh);
    nt = blk_sum<64>(nt, sh);
    tb0 = blk_sum<64>(tb0, sh);
    tb1 = blk_sum<64>(tb1, sh);
    tb2 = blk_sum<64>(tb2, sh);
    tb3 = blk_sum<64>(tb3, sh);
    if (threadIdx.x == 0) {
        block_max[block] = m;
        stats[block * 5 + 0] = nt;
        stats[block * 5 + 1] = tb0;
        stats[block * 5 + 2] = tb1;
        stats[block * 5 + 3] = tb2;
        stats[block * 5 + 4] = tb3;
    }
}

__device__ __forceinline__ void
pick_phase(int block, int grid, const int *keys, const int *tabu_until, int n, int round,
                int lottery_min, const int *block_max, const int *stats,
                unsigned long long rng0, unsigned long long lcg_c, unsigned long long num,
                unsigned long long den0, unsigned long long den1, unsigned long long den2,
                unsigned long long den3, unsigned long long *out,
                int *block_sel, int *h4, const unsigned long long *gr, int *seg_hist) {
    if (gr) {
        if (gr[0] != 0) return;
        round = (int)gr[1];
        rng0 = 123456789ull + gr[1];
        num = gr[16];
        den0 = gr[17];
        den1 = gr[18];
        den2 = gr[19];
        den3 = gr[20];
    }
    __shared__ int sh[32];
    __shared__ int carry_lottery;
    __shared__ int carry_sel;
    __shared__ int hist[1024];
    const unsigned int lanemask_lt = (1u << (threadIdx.x & 31)) - 1u;
    const int G = grid;
    int m = -2147483647 - 1;
    for (int b = threadIdx.x; b < G; b += 256)
        m = max(m, block_max[b]);
    int max_gain = blk_max<256>(m, sh);
    if (max_gain == (-2147483647 - 1)) max_gain = 0;
    int aspiration = (max_gain * 3) / 4;
    int lp = 0;
    for (int b = threadIdx.x; b < (int)block; b += 256)
        lp += lottery_count(stats, b, aspiration);
    int lottery_prefix = blk_sum<256>(lp, sh);

    int chunk = chunk_of(n, G);
    int lo = block * chunk;
    int hi = min(n, lo + chunk);
    for (int i = threadIdx.x; i < 1024; i += 256)
        hist[i] = 0;
    if (threadIdx.x == 0) {
        carry_lottery = 0;
        carry_sel = 0;
    }
    __syncthreads();
    for (int base = lo; base < hi; base += 256 * 4) {
        int key4[4];
        int f4[4];
        int lot_local = 0;
        for (int j = 0; j < 4; j++) {
            const int i = base + threadIdx.x * 4 + j;
            int key = 0x80000000;
            int f = 0;
            if (i < hi) {
                key = keys[i];
                if (signed_key_valid(key)) {
                    int gain = signed_key_gain(key);
                    bool is_tabu = (tabu_until[i] > round) && (gain < aspiration);
                    if (!is_tabu) {
                        if (gain > 0)
                            f = 1;
                        else if (gain >= lottery_min)
                            f = 2;
                    }
                }
            }
            key4[j] = key;
            f4[j] = f;
            lot_local += (f == 2) ? 1 : 0;
        }
        int total_lottery;
        int rank_base = blk_scan<256>(lot_local, sh, &total_lottery);
        int s4[4];
        int ok4[4];
        int sel_local = 0;
        int lr = 0;
        for (int j = 0; j < 4; j++) {
            int s = 0, ok = 0;
            if (f4[j] == 1) {
                s = 1;
                ok = key4[j];
            } else if (f4[j] == 2) {
                int gain = key4[j] >> 16;
                int penalty = (gain < 0) ? (-gain) : 0;
                unsigned long long den = (penalty & 3) == 0   ? den0
                                         : (penalty & 3) == 1 ? den1
                                         : (penalty & 3) == 2 ? den2
                                                              : den3;
                unsigned long long st = lcg_jump(
                    rng0, (unsigned long long)(lottery_prefix + carry_lottery + rank_base + lr + 1),
                    lcg_c);
                lr++;
                if (den > 0ull && (st % den) < num) {
                    s = 1;
                    ok = key4[j] & 0xFFFF;
                }
            }
            s4[j] = s;
            ok4[j] = ok;
            sel_local += s;
        }
        int total_sel;
        int pos_base = blk_scan<256>(sel_local, sh, &total_sel);
        int p = 0;
        for (int j = 0; j < 4; j++) {
            const int i = base + threadIdx.x * 4 + j;
            if (s4[j]) {
                out[lo + carry_sel + pos_base + p] = ((unsigned long long)(unsigned int)i << 32) |
                                                     (unsigned long long)(unsigned int)ok4[j];
                p++;
            }
            unsigned long long w = ((unsigned long long)(~(unsigned int)ok4[j]) << 32) |
                                   (unsigned long long)(unsigned int)i;
            s_hist_add(hist, w, s4[j] != 0, lanemask_lt);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            carry_lottery += total_lottery;
            carry_sel += total_sel;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sel[block] = carry_sel;
    for (int i = threadIdx.x; i < 1024; i += 256) {
        const int count = hist[i];
        if (seg_hist) seg_hist[block * 1024 + i] = count;
        if (count != 0) atomicAdd(&h4[i], count);
    }
}

__device__ __forceinline__ void sortmb_pass_phase(int block, int grid, const unsigned long long *segs,
                     const int *block_sel, int G, int chunk, int tiles,
                     unsigned long long *buf,
                     unsigned long long *sorted, const int *h4,
                     const int *seg_hist, int *tile_hist,
                     int *tgt_hist, int p, int slack, int limit,
                     const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        limit = (int)gr[14];
        slack = (int)gr[15];
    }
    __shared__ int sh[MB_WARPS];
    __shared__ int skip[4];
    __shared__ int run[256];
    __shared__ int total[256];
    __shared__ unsigned short wcount[MB_WARPS * 256];
    const int tid = threadIdx.x, b = block;
    const int warp = tid >> 5, lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;
    const int m = blk_sum<MB_THREADS>(tid < G ? block_sel[tid] : 0, sh);
    if (m == 0) return;
    if (tid < 4) skip[tid] = 0;
    __syncthreads();
    for (int d = 0; d < 4; d++)
        if (h4[d * 256 + tid] == m) skip[d] = 1;
    __syncthreads();
    int active = 0;
    for (int d = 0; d < 4; d++)
        if (!skip[d]) active |= 1 << d;
    if (active == 0) {
        if (p != 0 || b >= G) return;
        int sum;
        run[tid] = blk_scan<MB_THREADS>(tid < G ? block_sel[tid] : 0, sh, &sum);
        __syncthreads();
        for (int t = b; t < G; t += grid) {
            const int off = run[t];
            for (int i = tid; i < block_sel[t]; i += MB_THREADS) {
                const unsigned long long o = segs[t * chunk + i];
                const unsigned int key = (unsigned int)o;
                sorted[off + i] = ((unsigned long long)(~key) << 32) | (o >> 32);
                atomicAdd(&tgt_hist[((off + i) / MB_TILE) * 64 + (key & 63u)], 1);
            }
        }
        return;
    }
    if (!(active & (1 << p))) return;
    const int j = __popc((unsigned int)(active & ((1 << p) - 1)));
    const int L = __popc((unsigned int)active);
    int next = p + 1;
    while (next < 4 && !(active & (1 << next))) next++;
    const bool first = j == 0;
    const int count_tiles = first ? G : (m + MB_TILE - 1) / MB_TILE;
    if (b >= count_tiles) return;
    int sum;
    int off = blk_scan<MB_THREADS>(h4[p * 256 + tid], sh, &sum);
    int prev = 0;
    unsigned long long *dst = ((L - 1 - j) & 1) ? buf : sorted;
    const unsigned long long *src = ((L - 1 - j) & 1) ? sorted : buf;
    for (int t = b; t < count_tiles; t += grid) {
        off += first ? mb_hist_sum<1024>(seg_hist + p * 256, prev, t, tid)
                     : mb_hist_sum<256>(tile_hist + (p - 1) * tiles * 256, prev, t, tid);
        prev = t;
        const int count = first ? block_sel[t] : min(MB_TILE, m - t * MB_TILE);
        if (count == 0) continue;
        run[tid] = off;
        for (int base = 0; base < count; base += MB_TILE) {
            const int wb = base + warp * (32 * MB_E) + lane;
            unsigned long long x[MB_E];
#pragma unroll
            for (int e = 0; e < MB_E; e++) {
                const int i = wb + e * 32;
                unsigned long long w = 0ull;
                if (i < count) {
                    if (first) {
                        const unsigned long long o = segs[t * chunk + i];
                        w = ((unsigned long long)(~(unsigned int)o) << 32) | (o >> 32);
                    } else {
                        w = src[t * MB_TILE + i];
                    }
                }
                x[e] = w;
            }
            for (int i = tid; i < MB_WARPS * 256; i += MB_THREADS) wcount[i] = 0;
            __syncthreads();
            unsigned short *wc = wcount + warp * 256;
            int rk[MB_E];
#pragma unroll
            for (int e = 0; e < MB_E; e++) {
                const int d = wb + e * 32 < count ? s_digit(x[e], p) : 256;
                const unsigned int peers = __match_any_sync(0xffffffffu, d);
                const int r = d < 256 ? (int)wc[d] : 0;
                __syncwarp();
                if (d < 256 && (peers & lanemask_lt) == 0u)
                    wc[d] = (unsigned short)(r + __popc(peers));
                __syncwarp();
                rk[e] = r + __popc(peers & lanemask_lt);
            }
            __syncthreads();
            int a = 0;
#pragma unroll
            for (int ww = 0; ww < MB_WARPS; ww++) {
                const int c = wcount[ww * 256 + tid];
                wcount[ww * 256 + tid] = (unsigned short)a;
                a += c;
            }
            total[tid] = a;
            __syncthreads();
#pragma unroll
            for (int e = 0; e < MB_E; e++) {
                if (wb + e * 32 < count) {
                    const int d = s_digit(x[e], p);
                    const int pos = run[d] + (int)wc[d] + rk[e];
                    dst[pos] = x[e];
                    if (next < 4)
                        atomicAdd(&tile_hist[((next - 1) * tiles + pos / MB_TILE) * 256 +
                                             s_digit(x[e], next)], 1);
                    else
                        atomicAdd(&tgt_hist[(pos / MB_TILE) * 64 +
                                            ((~(unsigned int)(x[e] >> 32)) & 63u)], 1);
                }
            }
            __syncthreads();
            run[tid] += total[tid];
        }
    }
}

__device__ __forceinline__ void sortmb_select_phase(int block, int grid, const int *block_sel, int G,
                       const unsigned long long *sorted,
                       const int *nodes_in_part, int num_parts,
                       int max_part_size, int slack, int limit, int extra,
                       unsigned long long *picks, int *h4,
                       const int *tgt_hist, const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        limit = (int)gr[14];
        slack = (int)gr[15];
    }
    __shared__ int sh[MB_WARPS];
    __shared__ int tail[64];
    __shared__ int quota[64];
    __shared__ int prefix[64];
    __shared__ int wsum[MB_WARPS];
    __shared__ unsigned short wcount[MB_WARPS * 64];
    const int tid = threadIdx.x, b = block;
    const int warp = tid >> 5, lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;
    const int m = blk_sum<MB_THREADS>(tid < G ? block_sel[tid] : 0, sh);
    for (int i = b * MB_THREADS + tid; i < 1024; i += grid * MB_THREADS) h4[i] = 0;
    const int k_base = min(m, limit), kc = min(m, k_base + extra);
    if (kc == 0) {
        if (b == 0 && tid == 0) {
            picks[0] = 1ull << 32;
            picks[1] = (unsigned long long)m;
        }
        return;
    }
    if (b * MB_TILE >= kc) return;
    const int full = kc / MB_TILE;
    if (tid < 64) {
        tail[tid] = 0;
        int q = 0;
        if (tid < num_parts) {
            const int fr = max(0, max_part_size - nodes_in_part[tid]);
            q = max(1, fr + slack);
        }
        quota[tid] = q;
    }
    __syncthreads();
    for (int i = full * MB_TILE + tid; i < kc; i += MB_THREADS) {
        const int t = (int)(~(unsigned int)(sorted[i] >> 32)) & 63;
        atomicAdd(&tail[t], 1);
    }
    __syncthreads();
    int prev = 0, scanned = 0;
    for (int t = b; t * MB_TILE < kc; t += grid) {
        int before = 0;
        if (tid < 64) {
            prev += mb_hist_sum<64>(tgt_hist, scanned, t, tid);
            prefix[tid] = prev;
            before = min(prev, quota[tid]);
        }
        scanned = t;
        const int A = blk_sum<MB_THREADS>(before, sh);
        const int base = t * MB_TILE;
        const int wb = base + warp * (32 * MB_E) + lane;
        unsigned long long x[MB_E];
#pragma unroll
        for (int e = 0; e < MB_E; e++)
            x[e] = wb + e * 32 < kc ? sorted[wb + e * 32] : 0ull;
        for (int i = tid; i < MB_WARPS * 64; i += MB_THREADS) wcount[i] = 0;
        __syncthreads();
        unsigned short *wc = wcount + warp * 64;
        int rk[MB_E];
#pragma unroll
        for (int e = 0; e < MB_E; e++) {
            const int x_tgt = wb + e * 32 < kc ? ((int)(~(unsigned int)(x[e] >> 32)) & 63) : 64;
            const unsigned int peers = __match_any_sync(0xffffffffu, x_tgt);
            const int r = x_tgt < 64 ? (int)wc[x_tgt] : 0;
            __syncwarp();
            if (x_tgt < 64 && (peers & lanemask_lt) == 0u)
                wc[x_tgt] = (unsigned short)(r + __popc(peers));
            __syncwarp();
            rk[e] = r + __popc(peers & lanemask_lt);
        }
        __syncthreads();
        if (tid < 64) {
            int a = 0;
#pragma unroll
            for (int ww = 0; ww < MB_WARPS; ww++) {
                const int c = wcount[ww * 64 + tid];
                wcount[ww * 64 + tid] = (unsigned short)a;
                a += c;
            }
        }
        __syncthreads();
        unsigned int ballots[MB_E];
        int wtot = 0;
#pragma unroll
        for (int e = 0; e < MB_E; e++) {
            bool acc = false;
            if (wb + e * 32 < kc) {
                const int x_tgt = (int)(~(unsigned int)(x[e] >> 32)) & 63;
                acc = x_tgt < num_parts && prefix[x_tgt] + (int)wc[x_tgt] + rk[e] < quota[x_tgt];
            }
            ballots[e] = __ballot_sync(0xffffffffu, acc);
            wtot += __popc(ballots[e]);
        }
        if (lane == 0) wsum[warp] = wtot;
        __syncthreads();
        const int ws = lane < MB_WARPS ? wsum[lane] : 0;
        int inc = ws;
        for (int o = 1; o < MB_WARPS; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += y;
        }
        int pos = A + __shfl_sync(0xffffffffu, inc - ws, warp);
#pragma unroll
        for (int e = 0; e < MB_E; e++) {
            if ((ballots[e] >> lane) & 1u) {
                const int ps = pos + __popc(ballots[e] & lanemask_lt);
                if (ps < k_base) picks[2 + ps] = pick_word(x[e]);
            }
            pos += __popc(ballots[e]);
        }
        __syncthreads();
    }
    int accepted = 0;
    if (tid < 64) {
        const int c = prev + mb_hist_sum<64>(tgt_hist, scanned, full, tid) + tail[tid];
        accepted = min(c, quota[tid]);
    }
    const int total_acc = blk_sum<MB_THREADS>(accepted, sh);
    if (b == 0 && tid == 0) {
        picks[0] = total_acc == 0
                       ? ((unsigned long long)min(k_base, kc) | (1ull << 32))
                       : (unsigned long long)min(total_acc, k_base);
        picks[1] = (unsigned long long)m;
    }
    if (total_acc == 0) {
        for (int t = b; t * MB_TILE < min(k_base, kc); t += grid)
            for (int i = t * MB_TILE + tid; i < min((t + 1) * MB_TILE, min(k_base, kc));
                 i += MB_THREADS)
                picks[2 + i] = pick_word(sorted[i]);
    }
}

__device__ __forceinline__ void scan1024(int (&values)[TILES], int *sh, int &total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
        int x = values[tile];
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) sh[warp + tile * 8] = x;
        values[tile] = x - values[tile];
    }
    __syncthreads();
    if (warp == 0) {
        int x = sh[lane];
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, x, o);
            if (lane >= o) x += y;
        }
        sh[lane] = x;
    }
    __syncthreads();
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
        const int w = warp + tile * 8;
        values[tile] += w ? sh[w - 1] : 0;
    }
    total = sh[31];
    __syncthreads();
}

__device__ __forceinline__ void sort_select(
    SShared &S, int m, unsigned long long *buf, unsigned long long *sorted,
    const int *sizes, int np, int cap, int slack, int limit, int extra,
    unsigned long long *picks) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned int before = (1u << lane) - 1u;
    if (tid < 4) S.skipf[tid] = 0;
    int hv[TILES], hoff[TILES], total;
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile)
        hoff[tile] = hv[tile] = S.h4[tid + tile * BT];
    scan1024(hoff, S.sh, total);
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
        const int logical = tid + tile * BT;
        S.h4[logical] = hoff[tile] - (logical >> 8) * m;
        if (hv[tile] == m && m > 0) S.skipf[logical >> 8] = 1;
    }
    __syncthreads();
    const int L = (!S.skipf[0]) + (!S.skipf[1]) + (!S.skipf[2]) + (!S.skipf[3]);
    unsigned long long *cur = sorted;
    if ((L & 1) && m > STILE) {
        for (int tile = 0; tile < TILES; ++tile)
            for (int i = tid + tile * BT; i < m; i += S_THREADS) buf[i] = sorted[i];
        __syncthreads();
        cur = buf;
    }
    int j = 0;
    for (int p = 0; p < 4; ++p) {
        if (S.skipf[p]) continue;
        unsigned long long *dst = ((L - 1 - j) & 1) ? buf : sorted;
        S.run[tid] = S.h4[p * 256 + tid];
        for (int base = 0; base < m; base += STILE) {
            unsigned long long x[TILES][SE];
            int rank[TILES][SE];
#pragma unroll
            for (int tile = 0; tile < TILES; ++tile) {
                const int wb = base + (warp + tile * 8) * (32 * SE) + lane;
#pragma unroll
                for (int e = 0; e < SE; ++e) x[tile][e] = wb + e * 32 < m ? cur[wb + e * 32] : 0ull;
                reinterpret_cast<uint4 *>(S.wcount)[tid + tile * BT] = make_uint4(0u, 0u, 0u, 0u);
            }
            __syncthreads();
#pragma unroll
            for (int tile = 0; tile < TILES; ++tile) {
                const int wb = base + (warp + tile * 8) * (32 * SE) + lane;
                unsigned short *wc = S.wcount + (warp + tile * 8) * 256;
#pragma unroll
                for (int e = 0; e < SE; ++e) {
                    const int d = wb + e * 32 < m ? s_digit(x[tile][e], p) : 256;
                    const unsigned int peers = __match_any_sync(0xffffffffu, d);
                    const int r = d < 256 ? (int)wc[d] : 0;
                    __syncwarp();
                    if (d < 256 && (peers & before) == 0u)
                        wc[d] = (unsigned short)(r + __popc(peers));
                    __syncwarp();
                    rank[tile][e] = r + __popc(peers & before);
                }
            }
            __syncthreads();
            int a = 0;
#pragma unroll
            for (int w = 0; w < 32; ++w) {
                const int c = S.wcount[w * 256 + tid];
                S.wcount[w * 256 + tid] = (unsigned short)a;
                a += c;
            }
            S.ttot[tid] = a;
            __syncthreads();
#pragma unroll
            for (int tile = 0; tile < TILES; ++tile) {
                const int wb = base + (warp + tile * 8) * (32 * SE) + lane;
                const unsigned short *wc = S.wcount + (warp + tile * 8) * 256;
#pragma unroll
                for (int e = 0; e < SE; ++e) {
                    if (wb + e * 32 < m) {
                        const int d = s_digit(x[tile][e], p);
                        dst[S.run[d] + (int)wc[d] + rank[tile][e]] = x[tile][e];
                    }
                }
            }
            __syncthreads();
            S.run[tid] += S.ttot[tid];
        }
        cur = dst;
        ++j;
    }
    const int k_base = min(m, limit), kc = min(m, k_base + extra);
    if (tid < 64) {
        int q = 0;
        if (tid < np) q = max(1, max(0, cap - sizes[tid]) + slack);
        S.quota[tid] = q;
        S.run[tid] = 0;
    }
    int carry = 0;
    for (int base = 0; base < kc; base += STILE) {
        unsigned long long x[TILES][SE];
        int rank[TILES][SE];
#pragma unroll
        for (int tile = 0; tile < TILES; ++tile) {
            const int wb = base + (warp + tile * 8) * (32 * SE) + lane;
#pragma unroll
            for (int e = 0; e < SE; ++e) x[tile][e] = wb + e * 32 < kc ? sorted[wb + e * 32] : 0ull;
            reinterpret_cast<unsigned int *>(S.wcount)[tid + tile * BT] = 0u;
        }
        __syncthreads();
#pragma unroll
        for (int tile = 0; tile < TILES; ++tile) {
            const int wb = base + (warp + tile * 8) * (32 * SE) + lane;
            unsigned short *wc = S.wcount + (warp + tile * 8) * 64;
#pragma unroll
            for (int e = 0; e < SE; ++e) {
                const int t = wb + e * 32 < kc ? ((int)(~(unsigned int)(x[tile][e] >> 32)) & 63) : 64;
                const unsigned int peers = __match_any_sync(0xffffffffu, t);
                const int r = t < 64 ? (int)wc[t] : 0;
                __syncwarp();
                if (t < 64 && (peers & before) == 0u)
                    wc[t] = (unsigned short)(r + __popc(peers));
                __syncwarp();
                rank[tile][e] = r + __popc(peers & before);
            }
        }
        __syncthreads();
        if (tid < 64) {
            int a = 0;
#pragma unroll
            for (int w = 0; w < 32; ++w) {
                const int c = S.wcount[w * 64 + tid];
                S.wcount[w * 64 + tid] = (unsigned short)a;
                a += c;
            }
            S.ttot[tid] = a;
        }
        __syncthreads();
        unsigned int ballots[TILES][SE];
#pragma unroll
        for (int tile = 0; tile < TILES; ++tile) {
            const int wb = base + (warp + tile * 8) * (32 * SE) + lane;
            const unsigned short *wc = S.wcount + (warp + tile * 8) * 64;
            int wtot = 0;
#pragma unroll
            for (int e = 0; e < SE; ++e) {
                int acc = 0;
                if (wb + e * 32 < kc) {
                    const int t = (int)(~(unsigned int)(x[tile][e] >> 32)) & 63;
                    if (t < np) acc = S.run[t] + (int)wc[t] + rank[tile][e] < S.quota[t];
                }
                ballots[tile][e] = __ballot_sync(0xffffffffu, acc);
                wtot += __popc(ballots[tile][e]);
            }
            if (lane == 0) S.wsum[warp + tile * 8] = wtot;
        }
        __syncthreads();
        const int ws = S.wsum[lane];
        int inc = ws;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += y;
        }
        const int ttl = __shfl_sync(0xffffffffu, inc, 31);
#pragma unroll
        for (int tile = 0; tile < TILES; ++tile) {
            int pos = carry + __shfl_sync(0xffffffffu, inc - ws, warp + tile * 8);
#pragma unroll
            for (int e = 0; e < SE; ++e) {
                const unsigned int bal = ballots[tile][e];
                if ((bal >> lane) & 1u) {
                    const int ps = pos + __popc(bal & before);
                    if (ps < k_base) picks[2 + ps] = pick_word(x[tile][e]);
                }
                pos += __popc(bal);
            }
        }
        carry += ttl;
        __syncthreads();
        if (tid < 64) S.run[tid] += S.ttot[tid];
    }
    if (carry == 0) {
        const int take = min(k_base, kc);
        for (int tile = 0; tile < TILES; ++tile)
            for (int i = tid + tile * BT; i < take; i += S_THREADS)
                picks[2 + i] = pick_word(sorted[i]);
        if (tid == 0) {
            picks[0] = (unsigned long long)take | (1ull << 32);
            picks[1] = (unsigned long long)m;
        }
    } else if (tid == 0) {
        picks[0] = (unsigned long long)min(carry, k_base);
        picks[1] = (unsigned long long)m;
    }
}

__device__ __forceinline__ void sort_phase(
    const unsigned long long *segs, const int *block_sel, int G, int chunk,
    unsigned long long *buf, unsigned long long *sorted, const int *sizes, int np, int cap,
    int extra, unsigned long long *picks, int *h4, const unsigned long long *h) {
    __shared__ SShared S;
    __shared__ int seg_off[257];
    const int tid = threadIdx.x;
    int offsets[TILES], m;
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
        const int logical = tid + tile * BT;
        offsets[tile] = logical < G ? block_sel[logical] : 0;
    }
    scan1024(offsets, S.sh, m);
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) {
        const int logical = tid + tile * BT;
        if (logical < G) seg_off[logical] = offsets[tile];
        S.h4[logical] = h4[logical];
    }
    if (tid == 0) seg_off[G] = m;
    __syncthreads();
    for (int tile = 0; tile < TILES; ++tile) h4[tid + tile * BT] = 0;
    for (int base = 0; base < m; base += S_THREADS) {
        for (int tile = 0; tile < TILES; ++tile) {
            const int i = base + tid + tile * BT;
            if (i < m) {
                int a = 0, b = G;
                while (b - a > 1) {
                    const int mid = (a + b) >> 1;
                    if (seg_off[mid] <= i) a = mid;
                    else b = mid;
                }
                const unsigned long long o = segs[a * chunk + i - seg_off[a]];
                const unsigned int key = (unsigned int)o, node = (unsigned int)(o >> 32);
                sorted[i] = ((unsigned long long)(~key) << 32) | (unsigned long long)node;
            }
        }
    }
    __syncthreads();
    sort_select(S, m, buf, sorted, sizes, np, cap, (int)h[15], (int)h[14], extra, picks);
}
}

#define PERSISTENT_ROUNDS(ZGM) \
extern "C" __global__ void __launch_bounds__(256, 1) persistent_rounds_zgm##ZGM( \
    unsigned long long *h, const unsigned long long *c, unsigned int *counter, unsigned int rounds, \
    int counts_grid, int moves_grid, int filter_grid, const int *node_hyperedges, \
    const int *node_offsets, const int *hyperedge_nodes, const int *hyperedge_offsets, \
    int *part, int *sizes, int *tabu, unsigned int *counts, ulonglong2 *flags, int *priorities, \
    const int *order, int n8, int n4, int n2, int *pairs, int *block_max, int *stats, \
    int *block_sel, int *h4, unsigned long long *filtered, unsigned long long *buf, \
    unsigned long long *sorted, unsigned long long *picks, int sort_grid, int tiles, \
    int *seg_hist, int *tile_hist, int *tgt_hist) { \
    unsigned int generation = 0; \
    persistent100k::barrier(counter, generation); \
    __shared__ int active; \
    const int n = (int)c[0], np = (int)c[1], cap = (int)c[2], ne = (int)c[3]; \
    const int chunk = filt::chunk_of(n, filter_grid); \
    for (unsigned int round = 0; round < rounds; ++round) { \
        if (threadIdx.x == 0) active = ((const volatile unsigned long long *)h)[0] == 0; \
        __syncthreads(); \
        if (!active) break; \
        persistent_counts_100k(false, counts_grid, h, ne, n, pairs, node_hyperedges, \
            node_offsets, hyperedge_nodes, hyperedge_offsets, part, counts, flags); \
        persistent100k::barrier(counter, generation); \
        persistent_counts_100k(true, counts_grid, h, ne, n, pairs, node_hyperedges, \
            node_offsets, hyperedge_nodes, hyperedge_offsets, part, counts, flags); \
        persistent100k::barrier(counter, generation); \
        persistent_moves<ZGM>(moves_grid, n, np, cap, node_hyperedges, node_offsets, \
            part, sizes, flags, priorities, order, n8, n4, n2); \
        persistent100k::barrier(counter, generation); \
        for (int block = blockIdx.x; block < filter_grid; block += gridDim.x) { \
            persistent100k::stats_phase(block, filter_grid, priorities, tabu, n, 0, -3, block_max, stats, h); \
            __syncthreads(); \
        } \
        persistent100k::barrier(counter, generation); \
        for (int block = blockIdx.x; block < filter_grid; block += gridDim.x) { \
            persistent100k::pick_phase(block, filter_grid, priorities, tabu, n, 0, -3, \
                block_max, stats, 0ull, 1442695040888963407ull, 0ull, 0ull, 0ull, 0ull, \
                0ull, filtered, block_sel, h4, h, sort_grid > 0 ? seg_hist : nullptr); \
            if (sort_grid > 0) { \
                for (int i = block * 256 + threadIdx.x; i < tiles * 3 * 256; \
                     i += filter_grid * 256) tile_hist[i] = 0; \
                for (int i = block * 256 + threadIdx.x; i < tiles * 64; \
                     i += filter_grid * 256) tgt_hist[i] = 0; \
            } \
            __syncthreads(); \
        } \
        persistent100k::barrier(counter, generation); \
        if (sort_grid > 0) { \
            for (int p = 0; p < 4; ++p) { \
                for (int block = blockIdx.x; block < sort_grid; block += gridDim.x) { \
                    persistent100k::sortmb_pass_phase(block, sort_grid, filtered, block_sel, \
                        filter_grid, chunk, tiles, buf, sorted, h4, seg_hist, tile_hist, \
                        tgt_hist, p, 0, 0, h); \
                    __syncthreads(); \
                } \
                persistent100k::barrier(counter, generation); \
            } \
            for (int block = blockIdx.x; block < sort_grid; block += gridDim.x) { \
                persistent100k::sortmb_select_phase(block, sort_grid, block_sel, filter_grid, \
                    sorted, sizes, np, cap, 0, 0, (int)c[10], picks, h4, tgt_hist, h); \
                __syncthreads(); \
            } \
        } else if (blockIdx.x == 0) { \
            persistent100k::sort_phase(filtered, block_sel, filter_grid, chunk, buf, sorted, \
                sizes, np, cap, (int)c[10], picks, h4, h); \
        } \
        persistent100k::barrier(counter, generation); \
        if (blockIdx.x == 0) \
            grouped_finish_impl<true>(h, c, picks, sorted, part, sizes, tabu, node_offsets, pairs); \
        persistent100k::barrier(counter, generation); \
    } \
}

PERSISTENT_ROUNDS(0)
PERSISTENT_ROUNDS(1)

namespace track10k {
__device__ __forceinline__ void persistent_flags_10k(
    int block, int grid, int num_hyperedges, int num_nodes, const int *hyperedge_nodes,
    const int *hyperedge_offsets, const int *partition, unsigned long long *edge_flags_all,
    unsigned long long *edge_flags_double);
__device__ __forceinline__ void persistent_moves_10k(
    int grid, int num_nodes, int num_parts, int max_part_size, const int *node_hyperedges,
    const int *node_offsets, const int *partition, const int *nodes_in_part,
    const unsigned long long *edge_flags_all, const unsigned long long *edge_flags_double,
    int *move_priorities, int wide_count, int main_blocks, int lanes_count, const int *node_order);
}

namespace persistent10k {
using namespace persistent100k;
static_assert(BT == 256 && TILES == 4 && S_THREADS == 1024, "Invalid 10k tile geometry");

__device__ __forceinline__ void pick_phase(
    const int *keys, const int *tabu_until, int n, unsigned long long *buf,
    unsigned long long *sorted, const int *sizes, int np, int cap,
    unsigned long long *picks, const unsigned long long *h) {
    int round = (int)h[2], gmin = -2147483647 - 1;
    const int r = (int)h[1], L = (int)h[14];
    int limit = r < 50 ? L / 2 : r < 200 ? L : L / 3;
    int slack = r < 64 ? 8 : r < 256 ? 4 : 2;
    int extra = 16384;
    if (h[15]) {
        round = 2147483647;
        gmin = 0;
        limit = L;
        slack = (int)h[3];
        extra = (int)h[7];
    }
    __shared__ SShared S;
    const int tid = threadIdx.x;
    const unsigned int before = (1u << (tid & 31)) - 1u;
    int mx = -2147483647 - 1;
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile)
        for (int i = tid + tile * BT; i < n; i += S_THREADS) {
            const int key = keys[i];
            if (biased_key_valid(key) && biased_key_gain(key) >= gmin)
                mx = max(mx, biased_key_gain(key));
        }
    int max_gain = blk_max<BT>(mx, S.sh);
    if (max_gain == (-2147483647 - 1)) max_gain = 0;
    const int aspiration = max(1, (max_gain * 3) / 4);
#pragma unroll
    for (int tile = 0; tile < TILES; ++tile) S.h4[tid + tile * BT] = 0;
    if (tid == 0) S.carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += S_THREADS) {
        int selected[TILES], positions[TILES], total;
        unsigned long long words[TILES];
#pragma unroll
        for (int tile = 0; tile < TILES; ++tile) {
            const int i = base + tid + tile * BT;
            int key = 0x80000000, sel = 0;
            if (i < n) {
                key = keys[i];
                if (biased_key_valid(key) && biased_key_gain(key) >= gmin)
                    sel = !((tabu_until[i] > round) && (biased_key_gain(key) < aspiration));
            }
            selected[tile] = positions[tile] = sel;
            const int ok = sel ? key : 0;
            words[tile] = ((unsigned long long)(~(unsigned int)ok) << 32) |
                          (unsigned long long)(unsigned int)i;
        }
        scan1024(positions, S.sh, total);
#pragma unroll
        for (int tile = 0; tile < TILES; ++tile) {
            if (selected[tile]) sorted[S.carry + positions[tile]] = words[tile];
            s_hist_add(S.h4, words[tile], selected[tile] != 0, before);
        }
        __syncthreads();
        if (tid == 0) S.carry += total;
        __syncthreads();
    }
    const int m = S.carry;
    __syncthreads();
    sort_select(S, m, buf, sorted, sizes, np, cap, slack, limit, extra, picks);
}
}

extern "C" __global__ void __launch_bounds__(256, 1) persistent_rounds_10k(
    unsigned long long *h, const unsigned long long *c, unsigned int *counter, unsigned int rounds,
    int ne, int flags_grid, int moves_grid, const int *node_hyperedges, const int *node_offsets,
    const int *hyperedge_nodes, const int *hyperedge_offsets, int *part, int *sizes, int *tabu,
    unsigned long long *flags_all, unsigned long long *flags_double, int *priorities,
    const int *order, int wide_count, int main_blocks, int lanes_count, unsigned long long *buf,
    unsigned long long *sorted, unsigned long long *picks) {
    unsigned int generation = 0;
    persistent100k::barrier(counter, generation);
    __shared__ int active;
    const int n = (int)c[0], np = (int)c[1], cap = (int)c[2];
    for (unsigned int round = 0; round < rounds; ++round) {
        if (threadIdx.x == 0) active = ((const volatile unsigned long long *)h)[0] == 0;
        __syncthreads();
        if (!active) break;
        if (blockIdx.x == 0 && threadIdx.x == 0) ++h[2];
        if (threadIdx.x < 128)
            for (int block = blockIdx.x; block < flags_grid; block += gridDim.x)
                track10k::persistent_flags_10k(block, flags_grid, ne, n, hyperedge_nodes,
                    hyperedge_offsets, part, flags_all, flags_double);
        persistent100k::barrier(counter, generation);
        track10k::persistent_moves_10k(moves_grid, n, np, cap, node_hyperedges, node_offsets,
            part, sizes, flags_all, flags_double, priorities, wide_count, main_blocks, lanes_count,
            order);
        persistent100k::barrier(counter, generation);
        if (blockIdx.x == 0)
            persistent10k::pick_phase(priorities, tabu, n, buf, sorted, sizes, np, cap, picks, h);
        persistent100k::barrier(counter, generation);
        if (blockIdx.x == 0)
            grouped_small_finish_impl(h, c, picks, sorted, part, sizes, tabu);
        persistent100k::barrier(counter, generation);
    }
}

#define gl_m1    ((int)0x80000000)
#define gl_m2 (-1048576)

static __device__ __forceinline__ int gl_a1(int gl_v1, int *gl_v2)
{
    for (int gl_v3 = 16; gl_v3 > 0; gl_v3 >>= 1) gl_v1 += __shfl_xor_sync(0xffffffffu, gl_v1, gl_v3);
    const unsigned gl_v4 = threadIdx.x & 31u;
    const unsigned gl_v5 = threadIdx.x >> 5;
    if (gl_v4 == 0u) gl_v2[gl_v5] = gl_v1;
    __syncthreads();
    const int gl_v6 = (int)(blockDim.x >> 5);
    int gl_v7 = 0;
    for (int w = 0; w < gl_v6; w++) gl_v7 += gl_v2[w];
    __syncthreads();
    return gl_v7;
}

static __device__ __forceinline__ int gl_a2(int gl_v1, int *gl_v2)
{
    for (int gl_v3 = 16; gl_v3 > 0; gl_v3 >>= 1) {
        int gl_v8 = __shfl_xor_sync(0xffffffffu, gl_v1, gl_v3);
        if (gl_v8 > gl_v1) gl_v1 = gl_v8;
    }
    const unsigned gl_v4 = threadIdx.x & 31u;
    const unsigned gl_v5 = threadIdx.x >> 5;
    if (gl_v4 == 0u) gl_v2[gl_v5] = gl_v1;
    __syncthreads();
    const int gl_v6 = (int)(blockDim.x >> 5);
    int gl_v9 = gl_v2[0];
    for (int w = 1; w < gl_v6; w++) { int gl_v10 = gl_v2[w]; if (gl_v10 > gl_v9) gl_v9 = gl_v10; }
    __syncthreads();
    return gl_v9;
}

static __device__ __forceinline__ int gl_a3(int gl_v11, int *gl_v2, int *gl_v12)
{
    const unsigned gl_v4 = threadIdx.x & 31u;
    const unsigned gl_v5 = threadIdx.x >> 5;
    const unsigned gl_v9 = __ballot_sync(0xffffffffu, gl_v11 != 0);
    const int gl_v13 = __popc(gl_v9 & ((1u << gl_v4) - 1u));
    if (gl_v4 == 0u) gl_v2[gl_v5] = __popc(gl_v9);
    __syncthreads();
    const int gl_v6 = (int)(blockDim.x >> 5);
    int gl_v3 = 0, gl_v14 = 0;
    for (int w = 0; w < gl_v6; w++) {
        const int gl_v10 = gl_v2[w];
        if (w < (int)gl_v5) gl_v3 += gl_v10;
        gl_v14 += gl_v10;
    }
    __syncthreads();
    *gl_v12 = gl_v14;
    return gl_v3 + gl_v13;
}
static __device__ __forceinline__ ulonglong2 gl_a4(const ulonglong2 *gl_v15)
{
    ulonglong2 gl_v16;
    asm volatile("ld.global.cg.v2.u64 {%0,%1}, [%2];"
                 : "=l"(gl_v16.x), "=l"(gl_v16.y)
                 : "l"(gl_v15)
                 : "memory");
    return gl_v16;
}

#define gl_m3 \
    unsigned long long gl_v17 = 0ULL, gl_v18 = 0ULL, gl_v19 = 0ULL, gl_v20 = 0ULL, gl_v21 = 0ULL, \
                       gl_v22 = 0ULL, gl_v23 = 0ULL, gl_v24 = 0ULL, gl_v25 = 0ULL;

#define gl_m4(gl_v26) do { \
    unsigned long long gl_v10 = (gl_v26), gl_v27; \
    gl_v27 = gl_v17 & gl_v10; gl_v17 ^= gl_v10; gl_v10 = gl_v27; \
    gl_v27 = gl_v18 & gl_v10; gl_v18 ^= gl_v10; gl_v10 = gl_v27; \
    gl_v19 ^= gl_v10; \
} while (0)

#define gl_m5(gl_v26) do { \
    unsigned long long gl_v10 = (gl_v26), gl_v27; \
    gl_v27 = gl_v17 & gl_v10; gl_v17 ^= gl_v10; gl_v10 = gl_v27; \
    gl_v27 = gl_v18 & gl_v10; gl_v18 ^= gl_v10; gl_v10 = gl_v27; \
    gl_v27 = gl_v19 & gl_v10; gl_v19 ^= gl_v10; gl_v10 = gl_v27; \
    gl_v20 ^= gl_v10; \
} while (0)

#define gl_m6(gl_v3) do { \
    unsigned long long gl_v28, gl_v29, gl_v30, gl_v31; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v17, (gl_v3)); \
    gl_v30 = gl_v17 ^ gl_v28; gl_v29 = gl_v17 & gl_v28; gl_v17 = gl_v30; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v18, (gl_v3)); \
    gl_v30 = gl_v18 ^ gl_v28; gl_v31 = gl_v18 & gl_v28; gl_v18 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v19, (gl_v3)); \
    gl_v30 = gl_v19 ^ gl_v28; gl_v31 = gl_v19 & gl_v28; gl_v19 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v20, (gl_v3)); \
    gl_v30 = gl_v20 ^ gl_v28; gl_v20 = gl_v30 ^ gl_v29; \
} while (0)

#define gl_m7(gl_v3) do { \
    unsigned long long gl_v28, gl_v29, gl_v30, gl_v31; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v17, (gl_v3)); \
    gl_v30 = gl_v17 ^ gl_v28; gl_v29 = gl_v17 & gl_v28; gl_v17 = gl_v30; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v18, (gl_v3)); \
    gl_v30 = gl_v18 ^ gl_v28; gl_v31 = gl_v18 & gl_v28; gl_v18 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v19, (gl_v3)); \
    gl_v30 = gl_v19 ^ gl_v28; gl_v31 = gl_v19 & gl_v28; gl_v19 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v20, (gl_v3)); \
    gl_v30 = gl_v20 ^ gl_v28; gl_v31 = gl_v20 & gl_v28; gl_v20 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v21, (gl_v3)); \
    gl_v30 = gl_v21 ^ gl_v28; gl_v21 = gl_v30 ^ gl_v29; \
} while (0)

#define gl_m8(gl_v3) do { \
    unsigned long long gl_v28, gl_v29, gl_v30, gl_v31; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v17, (gl_v3)); \
    gl_v30 = gl_v17 ^ gl_v28; gl_v29 = gl_v17 & gl_v28; gl_v17 = gl_v30; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v18, (gl_v3)); \
    gl_v30 = gl_v18 ^ gl_v28; gl_v31 = gl_v18 & gl_v28; gl_v18 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v19, (gl_v3)); \
    gl_v30 = gl_v19 ^ gl_v28; gl_v31 = gl_v19 & gl_v28; gl_v19 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v20, (gl_v3)); \
    gl_v30 = gl_v20 ^ gl_v28; gl_v31 = gl_v20 & gl_v28; gl_v20 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v21, (gl_v3)); \
    gl_v30 = gl_v21 ^ gl_v28; gl_v31 = gl_v21 & gl_v28; gl_v21 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v22, (gl_v3)); \
    gl_v30 = gl_v22 ^ gl_v28; gl_v22 = gl_v30 ^ gl_v29; \
} while (0)

#define gl_m9(gl_v3) do { \
    unsigned long long gl_v28, gl_v29, gl_v30, gl_v31; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v17, (gl_v3)); \
    gl_v30 = gl_v17 ^ gl_v28; gl_v29 = gl_v17 & gl_v28; gl_v17 = gl_v30; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v18, (gl_v3)); \
    gl_v30 = gl_v18 ^ gl_v28; gl_v31 = gl_v18 & gl_v28; gl_v18 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v19, (gl_v3)); \
    gl_v30 = gl_v19 ^ gl_v28; gl_v31 = gl_v19 & gl_v28; gl_v19 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v20, (gl_v3)); \
    gl_v30 = gl_v20 ^ gl_v28; gl_v31 = gl_v20 & gl_v28; gl_v20 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v21, (gl_v3)); \
    gl_v30 = gl_v21 ^ gl_v28; gl_v31 = gl_v21 & gl_v28; gl_v21 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v22, (gl_v3)); \
    gl_v30 = gl_v22 ^ gl_v28; gl_v31 = gl_v22 & gl_v28; gl_v22 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v23, (gl_v3)); \
    gl_v30 = gl_v23 ^ gl_v28; gl_v23 = gl_v30 ^ gl_v29; \
} while (0)

#define gl_m10(gl_v3) do { \
    unsigned long long gl_v28, gl_v29, gl_v30, gl_v31; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v17, (gl_v3)); \
    gl_v30 = gl_v17 ^ gl_v28; gl_v29 = gl_v17 & gl_v28; gl_v17 = gl_v30; \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v18, (gl_v3)); \
    gl_v30 = gl_v18 ^ gl_v28; gl_v31 = gl_v18 & gl_v28; gl_v18 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v19, (gl_v3)); \
    gl_v30 = gl_v19 ^ gl_v28; gl_v31 = gl_v19 & gl_v28; gl_v19 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v20, (gl_v3)); \
    gl_v30 = gl_v20 ^ gl_v28; gl_v31 = gl_v20 & gl_v28; gl_v20 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v21, (gl_v3)); \
    gl_v30 = gl_v21 ^ gl_v28; gl_v31 = gl_v21 & gl_v28; gl_v21 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v22, (gl_v3)); \
    gl_v30 = gl_v22 ^ gl_v28; gl_v31 = gl_v22 & gl_v28; gl_v22 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v23, (gl_v3)); \
    gl_v30 = gl_v23 ^ gl_v28; gl_v31 = gl_v23 & gl_v28; gl_v23 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v24, (gl_v3)); \
    gl_v30 = gl_v24 ^ gl_v28; gl_v31 = gl_v24 & gl_v28; gl_v24 = gl_v30 ^ gl_v29; gl_v29 = gl_v31 | (gl_v30 & gl_v29); \
    gl_v28 = __shfl_xor_sync(0xffffffffu, gl_v25, (gl_v3)); \
    gl_v30 = gl_v25 ^ gl_v28; gl_v25 = gl_v30 ^ gl_v29; \
} while (0)

#define gl_m11(gl_v32) ( \
    ((int)((gl_v17 >> (gl_v32)) & 1ULL) << 0) | \
    ((int)((gl_v18 >> (gl_v32)) & 1ULL) << 1) | \
    ((int)((gl_v19 >> (gl_v32)) & 1ULL) << 2) )

#define gl_m12(gl_v32) ( \
    ((int)((gl_v17 >> (gl_v32)) & 1ULL) << 0) | \
    ((int)((gl_v18 >> (gl_v32)) & 1ULL) << 1) | \
    ((int)((gl_v19 >> (gl_v32)) & 1ULL) << 2) | \
    ((int)((gl_v20 >> (gl_v32)) & 1ULL) << 3) )

#define gl_m13(gl_v32) ( \
    ((int)((gl_v17 >> (gl_v32)) & 1ULL) << 0) | \
    ((int)((gl_v18 >> (gl_v32)) & 1ULL) << 1) | \
    ((int)((gl_v19 >> (gl_v32)) & 1ULL) << 2) | \
    ((int)((gl_v20 >> (gl_v32)) & 1ULL) << 3) | \
    ((int)((gl_v21 >> (gl_v32)) & 1ULL) << 4) )

#define gl_m14(gl_v32) ( \
    ((int)((gl_v17 >> (gl_v32)) & 1ULL) << 0) | \
    ((int)((gl_v18 >> (gl_v32)) & 1ULL) << 1) | \
    ((int)((gl_v19 >> (gl_v32)) & 1ULL) << 2) | \
    ((int)((gl_v20 >> (gl_v32)) & 1ULL) << 3) | \
    ((int)((gl_v21 >> (gl_v32)) & 1ULL) << 4) | \
    ((int)((gl_v22 >> (gl_v32)) & 1ULL) << 5) )

#define gl_m15(gl_v32) ( \
    ((int)((gl_v17 >> (gl_v32)) & 1ULL) << 0) | \
    ((int)((gl_v18 >> (gl_v32)) & 1ULL) << 1) | \
    ((int)((gl_v19 >> (gl_v32)) & 1ULL) << 2) | \
    ((int)((gl_v20 >> (gl_v32)) & 1ULL) << 3) | \
    ((int)((gl_v21 >> (gl_v32)) & 1ULL) << 4) | \
    ((int)((gl_v22 >> (gl_v32)) & 1ULL) << 5) | \
    ((int)((gl_v23 >> (gl_v32)) & 1ULL) << 6) )

#define gl_m16(gl_v32) ( \
    ((int)((gl_v17 >> (gl_v32)) & 1ULL) << 0) | \
    ((int)((gl_v18 >> (gl_v32)) & 1ULL) << 1) | \
    ((int)((gl_v19 >> (gl_v32)) & 1ULL) << 2) | \
    ((int)((gl_v20 >> (gl_v32)) & 1ULL) << 3) | \
    ((int)((gl_v21 >> (gl_v32)) & 1ULL) << 4) | \
    ((int)((gl_v22 >> (gl_v32)) & 1ULL) << 5) | \
    ((int)((gl_v23 >> (gl_v32)) & 1ULL) << 6) | \
    ((int)((gl_v24 >> (gl_v32)) & 1ULL) << 7) | \
    ((int)((gl_v25 >> (gl_v32)) & 1ULL) << 8) )

#define gl_m17(gl_v33, gl_v3) do {                                             \
    gl_v34 |= __shfl_xor_sync(0xffffffffu, gl_v34, (gl_v3));               \
    gl_v35 +=                                                   \
        __shfl_xor_sync(0xffffffffu, gl_v35, (gl_v3));            \
    gl_v36 += __shfl_xor_sync(0xffffffffu, gl_v36, (gl_v3));                         \
    gl_v33(gl_v3);                                                                 \
} while (0)

#define gl_m18
#define gl_m19   gl_m17(gl_m6, 1);
#define gl_m20   gl_m17(gl_m7, 2); gl_m17(gl_m7, 1);
#define gl_m21   gl_m17(gl_m8, 4); gl_m17(gl_m8, 2); \
                      gl_m17(gl_m8, 1);
#define gl_m22  gl_m17(gl_m9, 8); gl_m17(gl_m9, 4); \
                      gl_m17(gl_m9, 2); gl_m17(gl_m9, 1);
#define gl_m23  gl_m17(gl_m10, 16); gl_m17(gl_m10, 8); \
                      gl_m17(gl_m10, 4); gl_m17(gl_m10, 2); \
                      gl_m17(gl_m10, 1);


#define gl_m24(gl_v37, gl_v38) do {                                               \
    const unsigned long long gl_v39 = (gl_v37).x;                                     \
    const unsigned long long gl_v40 = (gl_v37).y;                                     \
    const unsigned long long gl_v41 = gl_v42 >= 64 ? ~0ULL : ((1ULL << gl_v42) - 1ULL); \
    if (gl_v42 <= 64 && gl_v39 == gl_v41 && (gl_v40 & gl_v43) != 0ULL) { \
        gl_v34 |= gl_v39 & ~gl_v43; \
    } else { \
    if ((gl_v39 & gl_v43) != 0ULL &&                                         \
        (gl_v40 & gl_v43) == 0ULL) {                                         \
        const int gl_v44 = __popcll(gl_v39);                                      \
        if (gl_v44 > 1) gl_v36 += (gl_v44 - 1);                                  \
    }                                                                          \
    const unsigned long long gl_v45 =                                           \
        (gl_v39 & ~gl_v43) | (gl_v40 & gl_v43);                            \
    if (gl_v45 & gl_v43) gl_v35++;                          \
    const unsigned long long gl_v46 = gl_v45 & ~gl_v43;                       \
    gl_v34 |= gl_v46;                                                          \
    gl_v38(gl_v46);                                                                 \
    } \
} while (0)


#define gl_m25(gl_v47) __ldg(&gl_v48[(gl_v47)])

#define gl_m26(gl_v49) do {                                                      \
    const int gl_v50 = gl_v51 + (gl_v49), gl_v52 = gl_v51 + 2 * (gl_v49), gl_v53 = gl_v51 + 3 * (gl_v49);          \
    gl_v54 = (gl_v50 < gl_v55);                                                 \
    gl_v56 = (gl_v52 < gl_v55);                                                 \
    gl_v57 = (gl_v53 < gl_v55);                                                 \
    const int gl_v58 = gl_v59 ? (int)(((long long)gl_v51  * gl_v60) >> 8) : gl_v51;  \
    const int gl_v61 = gl_v59 ? (int)(((long long)gl_v50 * gl_v60) >> 8) : gl_v50; \
    const int gl_v62 = gl_v59 ? (int)(((long long)gl_v52 * gl_v60) >> 8) : gl_v52; \
    const int gl_v63 = gl_v59 ? (int)(((long long)gl_v53 * gl_v60) >> 8) : gl_v53; \
    gl_v64 = __ldg(&gl_v65[gl_v66 + gl_v58]);                               \
    gl_v67 = gl_v54 ? __ldg(&gl_v65[gl_v66 + gl_v61]) : 0;                     \
    gl_v68 = gl_v56 ? __ldg(&gl_v65[gl_v66 + gl_v62]) : 0;                     \
    gl_v69 = gl_v57 ? __ldg(&gl_v65[gl_v66 + gl_v63]) : 0;                     \
} while (0)


static __device__ __forceinline__ void gl_a5(
    const int gl_v70, const int gl_v71, const int *gl_v72,
    unsigned long long *gl_v73, unsigned long long *gl_v74)
{
    __syncthreads();
    const int gl_v4 = threadIdx.x & 31;
    const int gl_v5 = threadIdx.x >> 5;
    const int gl_v6 = blockDim.x >> 5;
    if (gl_v5 == 0) {
        unsigned gl_v75 = __ballot_sync(0xffffffffu, gl_v4 < gl_v70 && gl_v72[gl_v4] < gl_v71);
        unsigned gl_v76 = __ballot_sync(0xffffffffu, gl_v4 + 32 < gl_v70 && gl_v72[gl_v4 + 32] < gl_v71);
        if (gl_v4 == 0) *gl_v73 = (unsigned long long)gl_v75 | ((unsigned long long)gl_v76 << 32);
    }
    for (int gl_v10 = gl_v5; gl_v10 < 64; gl_v10 += gl_v6) {
        int gl_v77 = gl_v10 < gl_v70 ? gl_v72[gl_v10] : 0;
        unsigned gl_v75 = __ballot_sync(0xffffffffu, gl_v4 < gl_v70 && gl_v77 > gl_v72[gl_v4] + 2);
        unsigned gl_v76 = __ballot_sync(0xffffffffu, gl_v4 + 32 < gl_v70 && gl_v77 > gl_v72[gl_v4 + 32] + 2);
        if (gl_v4 == 0) gl_v74[gl_v10] = (unsigned long long)gl_v75 | ((unsigned long long)gl_v76 << 32);
    }
    __syncthreads();
}

#define gl_m27(gl_v49, gl_v78, gl_v38, gl_v79, gl_v80)                                     \
{                                                                              \
    const int gl_v81 = (int)(gl_v4 & (unsigned)((gl_v49) - 1));                         \
    const int gl_v82 = gl_v83.x;                                                    \
    const int gl_v66 = gl_v83.y;                                                  \
    const int gl_v60 = gl_v83.z;                                             \
    int gl_v84 = gl_m1;                                                      \
    const int gl_v55 = gl_v60 > 256 ? 256 : gl_v60;             \
    const bool gl_v59 = (gl_v60 > 256);                                  \
    const bool gl_v85 = (gl_v60 > 0) && (gl_v81 < gl_v55);                 \
    int gl_v51 = gl_v81;                                                              \
    int gl_v64 = 0, gl_v67 = 0, gl_v68 = 0, gl_v69 = 0;                                    \
    bool gl_v54 = false, gl_v56 = false, gl_v57 = false;                                \
    if (gl_v85) { gl_m26(gl_v49); }                                                 \
    const int gl_v86 = gl_m25(gl_v82);                          \
    const bool gl_v87 = (unsigned)gl_v86 < (unsigned)gl_v42 &&     \
        gl_v88[gl_v86] > 1 &&                              \
        gl_v60 > 0;                                                       \
    const unsigned long long gl_v43 = gl_v87 ? (1ULL << gl_v86) : 0ULL; \
    gl_m3                                                                \
    unsigned long long gl_v34 = 0ULL;                                       \
    int gl_v35 = 0;                                             \
    int gl_v36 = 0;                                                              \
    if (gl_v87 && gl_v85) {                                                    \
            for (;;) {                                                         \
                ulonglong2 gl_v89; gl_v89.x = 0ULL; gl_v89.y = 0ULL;                    \
                ulonglong2 gl_v90; gl_v90.x = 0ULL; gl_v90.y = 0ULL;                    \
                ulonglong2 gl_v91; gl_v91.x = 0ULL; gl_v91.y = 0ULL;                    \
                const ulonglong2 gl_v92 = gl_a4(&gl_v93[gl_v64]);      \
                if (gl_v54) gl_v89 = gl_a4(&gl_v93[gl_v67]);              \
                if (gl_v56) gl_v90 = gl_a4(&gl_v93[gl_v68]);              \
                if (gl_v57) gl_v91 = gl_a4(&gl_v93[gl_v69]);              \
                gl_m24(gl_v92, gl_v38);                                          \
                if (gl_v54) { gl_m24(gl_v89, gl_v38); }                             \
                if (gl_v56) { gl_m24(gl_v90, gl_v38); }                             \
                if (gl_v57) { gl_m24(gl_v91, gl_v38); }                             \
                gl_v51 += 4 * (gl_v49);                                                 \
                if (gl_v51 >= gl_v55) break;                                  \
                gl_m26(gl_v49);                                                  \
            }                                                                  \
        }                                                                      \
        gl_v80                                                                    \
        if (gl_v36 > 255) gl_v36 = 255;                                            \
        int gl_v94 = gl_v60 > 255 ? 255 : gl_v60;             \
        int gl_v95 = gl_v94 + gl_v36;                                  \
        if (gl_v95 > 255) gl_v95 = 255;                                  \
        int gl_v96 = -999999;                                               \
        int gl_v97 = gl_v86;                                        \
        unsigned long long gl_v98 = gl_v34 & gl_v99; \
        if (gl_v98) { \
            unsigned long long gl_v29 = gl_v100[gl_v86], gl_v101; \
            gl_v101 = gl_v17 & gl_v29; gl_v17 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v18 & gl_v29; gl_v18 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v19 & gl_v29; gl_v19 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v20 & gl_v29; gl_v20 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v21 & gl_v29; gl_v21 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v22 & gl_v29; gl_v22 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v23 & gl_v29; gl_v23 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v24 & gl_v29; gl_v24 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v101 = gl_v25 & gl_v29; gl_v25 ^= gl_v29; gl_v29 = gl_v101; \
            gl_v96 = 0; \
            { unsigned long long gl_v102 = gl_v98 & gl_v25; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 256; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v24; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 128; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v23; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 64; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v22; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 32; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v21; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 16; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v20; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 8; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v19; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 4; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v18; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 2; } } \
            { unsigned long long gl_v102 = gl_v98 & gl_v17; \
              if (gl_v102) { gl_v98 = gl_v102; gl_v96 |= 1; } } \
            gl_v97 = __ffsll(gl_v98) - 1; \
            gl_v96 -= gl_v35; \
            if (gl_v59) { \
                const int gl_v103 = (int)((gl_v100[gl_v86] >> gl_v97) & 1ULL); \
                const int gl_v104 = gl_v96 - gl_v103; \
                gl_v96 = (int)(((long long)gl_v104 * gl_v60) / 256) + gl_v103; \
            } \
        } \
        if (gl_v97 != gl_v86) {                                     \
            int gl_v105 = gl_v96 > 32767 ? 32767 : gl_v96;                    \
            if (gl_v105 < -32768) gl_v105 = -32768;                                      \
            unsigned short gl_v106 =                                               \
                (unsigned short)(gl_v97 & 63) | ((gl_v82 & 3) << 6);        \
            gl_v84 = ((int)(short)gl_v105 << 16) | (gl_v95 << 8) | gl_v106;          \
        }                                                                      \
    if (gl_v81 == 0) {                                                            \
        gl_v107[gl_v82] = gl_v84;                                        \
        if (gl_v84 != gl_m1) {                                               \
            const int gl_v108 = gl_v84 >> 16;                                       \
            if (gl_v108 > gl_v109) gl_v109 = gl_v108;                                    \
        }                                                                      \
    }                                                                          \
}

union gl_s1 {
    struct {
        int gl_v110[64],gl_v111[64],gl_v112[64],gl_v113[65],gl_v114[64];
        int gl_v115[512],gl_v116[512],gl_v117[512];
        unsigned long long gl_v118[512],gl_v119[512],gl_v120;
        int gl_v121[512],gl_v122[64],gl_v123;
        int gl_v124;
    } gl_v125;

    struct {
        int gl_v110[64],gl_v111[64],gl_v112[64],gl_v113[65];
        int gl_v126[512],gl_v115[512],gl_v117[512];
        unsigned gl_v127[512];unsigned char gl_v116[512];int gl_v124;
    } gl_v128;

    unsigned long long gl_v129[4096];
    unsigned int gl_v130[264];
    struct {
        int gl_v117[1024];
        unsigned char gl_v131[1024],gl_v132[1024];
        int gl_v133[64],gl_v111[64],gl_v134[64];
        unsigned int gl_v135[32*64];
        unsigned short gl_v136[32*64],gl_v137[2048];
        int gl_v138[65];
    } gl_v139;
    struct {
        int gl_v117[1024];
        short gl_v140[1024];
        short gl_v141[1024];
        int gl_v142[1024];
        int gl_v133[64];
        int gl_v143[64];
    } gl_v144;
};

static __device__ __forceinline__ int gl_a6(
    int gl_v145, int gl_v146, int gl_v70, int gl_v71, int *gl_v48, int *gl_v147,
    const int *gl_v148, int *gl_v149, int gl_v150, int gl_v151,
    int gl_v152, int gl_v153, gl_s1 *gl_v154, int *gl_v155)
{
    int gl_v156 = threadIdx.x;
    int gl_v157 = 0;
    if (gl_v156 < 64) gl_v154->gl_v144.gl_v133[gl_v156] = gl_v156 < gl_v70 ? __ldcg(&gl_v147[gl_v156]) : 0;
    __syncthreads();
    for (int gl_v158 = 0; gl_v158 < gl_v145; gl_v158 += 1024) {
        int gl_v159 = min(1024, gl_v145 - gl_v158);
        if (gl_v156 < 64) gl_v154->gl_v144.gl_v143[gl_v156] = -1;
        for (int gl_v160 = gl_v156; gl_v160 < gl_v159; gl_v160 += blockDim.x) {
            int gl_v161 = __ldcg(&gl_v148[gl_v158+gl_v160]);
            int gl_v82 = gl_v161 >> 6, gl_v162 = gl_v161 & 63;
            int gl_v163 = (unsigned)gl_v82 < (unsigned)gl_v146 ? __ldcg(&gl_v48[gl_v82]) : -1;
            gl_v154->gl_v144.gl_v117[gl_v160] = ((unsigned)gl_v163 < (unsigned)gl_v70 && gl_v162 < gl_v70)
                ? ((gl_v82 << 12) | (gl_v163 << 6) | gl_v162) : -1;
            gl_v154->gl_v144.gl_v140[gl_v160] = -1;
            gl_v154->gl_v144.gl_v141[gl_v160] = -1;
            gl_v154->gl_v144.gl_v142[gl_v160] = -1;
        }
        __syncthreads();
        if (gl_v156 < 32) {
            for (int gl_v164 = 0; gl_v164 < gl_v159; gl_v164 += 16) {
                int gl_v160 = gl_v164 + (gl_v156 >> 1);
                int gl_v117 = gl_v160 < gl_v159 ? gl_v154->gl_v144.gl_v117[gl_v160] : -1;
                int gl_v165 = gl_v117 < 0 ? (64 + gl_v156) : ((gl_v156 & 1) ? (gl_v117 & 63) : ((gl_v117 >> 6) & 63));
                unsigned gl_v166 = __match_any_sync(0xffffffffu, gl_v165);
                unsigned gl_v167 = gl_v166 & ((1u << (gl_v156 & ~1)) - 1u);
                int gl_v11 = -1;
                if (gl_v117 >= 0) {
                    gl_v11 = gl_v167 ? gl_v164 + ((31 - __clz(gl_v167)) >> 1) : gl_v154->gl_v144.gl_v143[gl_v165];
                    if (gl_v156 & 1) gl_v154->gl_v144.gl_v141[gl_v160] = (short)gl_v11;
                    else gl_v154->gl_v144.gl_v140[gl_v160] = (short)gl_v11;
                }
                __syncwarp(0xffffffffu);
                if (gl_v117 >= 0 && gl_v156 == 31 - __clz(gl_v166)) gl_v154->gl_v144.gl_v143[gl_v165] = gl_v160;
                __syncwarp(0xffffffffu);
            }
        }
        __syncthreads();
        int gl_v160 = gl_v156;
        for (int gl_v168 = 0; ; ++gl_v168) {
            if (gl_v160 < gl_v159) {
                int gl_v169 = gl_v154->gl_v144.gl_v140[gl_v160], gl_v170 = gl_v154->gl_v144.gl_v141[gl_v160];
                int gl_v171 = gl_v169 < 0 ? -2 : atomicAdd(&gl_v154->gl_v144.gl_v142[gl_v169], 0);
                int gl_v172 = gl_v170 < 0 ? -2 : atomicAdd(&gl_v154->gl_v144.gl_v142[gl_v170], 0);
                bool gl_v173 = (gl_v169 < 0 || (gl_v171 >= 0 && gl_v171 < gl_v168)) &&
                             (gl_v170 < 0 || (gl_v172 >= 0 && gl_v172 < gl_v168));
                if (gl_v173) {
                    int gl_v117 = gl_v154->gl_v144.gl_v117[gl_v160];
                    if (gl_v117 >= 0) {
                        int gl_v174 = (gl_v117 >> 6) & 63, gl_v162 = gl_v117 & 63;
                        if (gl_v154->gl_v144.gl_v133[gl_v162] < gl_v71 && gl_v154->gl_v144.gl_v133[gl_v174] > 1) {
                            --gl_v154->gl_v144.gl_v133[gl_v174];
                            ++gl_v154->gl_v144.gl_v133[gl_v162];
                            gl_v48[gl_v117 >> 12] = gl_v162;
                            ++gl_v157;
                        }
                    }
                    atomicExch(&gl_v154->gl_v144.gl_v142[gl_v160], gl_v168);
                    gl_v160 += blockDim.x;
                }
            }
            if (__syncthreads_count(gl_v160 < gl_v159) == 0) break;
        }
    }
    if (gl_v156 < gl_v70 && gl_v156 < 64) gl_v147[gl_v156] = gl_v154->gl_v144.gl_v133[gl_v156];
    int gl_v124 = gl_a1(gl_v157, gl_v155);
    if (gl_v124 > 0) {
        int gl_v175 = min(gl_v145, max(gl_v152, gl_v124 * gl_v153));
        for (int gl_v176 = gl_v156; gl_v176 < gl_v175; gl_v176 += blockDim.x) {
            int gl_v82 = __ldcg(&gl_v148[gl_v176]) >> 6;
            if ((unsigned)gl_v82 < (unsigned)gl_v146) gl_v149[gl_v82] = gl_v150 + gl_v151;
        }
    }
    return gl_v124;
}

static __device__ __forceinline__ int gl_a7(
    int gl_v145, int gl_v146, int gl_v70, int gl_v71, int *gl_v48, int *gl_v147,
    const int *gl_v148, int *gl_v149, int gl_v150, int gl_v151,
    int gl_v152, int gl_v153, gl_s1 *gl_v154, int *gl_v155)
{
    int gl_v156=threadIdx.x;
    if(gl_v156<64){
        gl_v154->gl_v139.gl_v133[gl_v156]=gl_v156<gl_v70?__ldcg(&gl_v147[gl_v156]):0;
        gl_v154->gl_v139.gl_v111[gl_v156]=0;
    }
    __syncthreads();
    for(int gl_v160=gl_v156;gl_v160<gl_v145;gl_v160+=blockDim.x){
        int gl_v161=__ldcg(&gl_v148[gl_v160]), gl_v82=gl_v161>>6, gl_v162=gl_v161&63;
        if((unsigned)gl_v82<(unsigned)gl_v146 && gl_v162<gl_v70){
            int gl_v174=__ldcg(&gl_v48[gl_v82]);
            if((unsigned)gl_v174<(unsigned)gl_v70)atomicAdd(&gl_v154->gl_v139.gl_v111[gl_v174],1);
        }
    }
    __syncthreads();
    int gl_v177=__syncthreads_or(gl_v156<gl_v70 && gl_v156<64 &&
        (gl_v154->gl_v139.gl_v133[gl_v156]-gl_v154->gl_v139.gl_v111[gl_v156]<1 || gl_v154->gl_v139.gl_v133[gl_v156]>gl_v71));
    if(gl_v177)
        return gl_a6(gl_v145,gl_v146,gl_v70,gl_v71,gl_v48,gl_v147,gl_v148,
                                    gl_v149,gl_v150,gl_v151,gl_v152,gl_v153,gl_v154,gl_v155);
    int gl_v157=0;
    for(int gl_v158=0;gl_v158<gl_v145;gl_v158+=1024){
        int gl_v159=min(1024,gl_v145-gl_v158);
        for(int gl_v160=gl_v156;gl_v160<64*32;gl_v160+=blockDim.x)gl_v154->gl_v139.gl_v135[gl_v160]=0;
        __syncthreads();
        for(int gl_v160=gl_v156;gl_v160<gl_v159;gl_v160+=blockDim.x){
            int gl_v161=__ldcg(&gl_v148[gl_v158+gl_v160]), gl_v82=gl_v161>>6, gl_v162=gl_v161&63;
            int gl_v174=(unsigned)gl_v82<(unsigned)gl_v146?__ldcg(&gl_v48[gl_v82]):-1;
            int gl_v117=((unsigned)gl_v174<(unsigned)gl_v70 && gl_v162<gl_v70)
                ?((gl_v82<<12)|(gl_v174<<6)|gl_v162):-1;
            gl_v154->gl_v139.gl_v117[gl_v160]=gl_v117;
            gl_v154->gl_v139.gl_v131[gl_v160]=(gl_v117>=0);
            gl_v154->gl_v139.gl_v132[gl_v160]=0;
            if(gl_v117>=0){
                atomicOr(&gl_v154->gl_v139.gl_v135[(gl_v160>>5)*64+gl_v174],1u<<(gl_v160&31));
                if(gl_v174!=gl_v162)atomicOr(&gl_v154->gl_v139.gl_v135[(gl_v160>>5)*64+gl_v162],1u<<(gl_v160&31));
            }
        }
        __syncthreads();
        if(gl_v156<64){
            int gl_v178=0;
            for(int w=0;w<((gl_v159+31)>>5);++w){
                gl_v154->gl_v139.gl_v136[w*64+gl_v156]=(unsigned short)gl_v178;
                gl_v178+=__popc(gl_v154->gl_v139.gl_v135[w*64+gl_v156]);
            }
            int gl_v179=gl_v178,gl_v4=gl_v156&31;
            for(int gl_v3=1;gl_v3<32;gl_v3<<=1){
                int gl_v1=__shfl_up_sync(0xffffffffu,gl_v179,gl_v3);
                if(gl_v4>=gl_v3)gl_v179+=gl_v1;
            }
            gl_v154->gl_v139.gl_v138[gl_v156+1]=gl_v179;
        }
        if(gl_v156==0)gl_v154->gl_v139.gl_v138[0]=0;
        __syncthreads();
        if(gl_v156>=32 && gl_v156<64)gl_v154->gl_v139.gl_v138[gl_v156+1]+=gl_v154->gl_v139.gl_v138[32];
        __syncthreads();
        for(int gl_v160=gl_v156;gl_v160<gl_v159;gl_v160+=blockDim.x){
            int gl_v117=gl_v154->gl_v139.gl_v117[gl_v160];
            if(gl_v117>=0){
                int gl_v174=(gl_v117>>6)&63,gl_v162=gl_v117&63;
                unsigned gl_v115=(1u<<(gl_v160&31))-1u;
                int gl_v180=gl_v160>>5;
                int gl_v181=gl_v154->gl_v139.gl_v138[gl_v174]+gl_v154->gl_v139.gl_v136[gl_v180*64+gl_v174]
                       +__popc(gl_v154->gl_v139.gl_v135[gl_v180*64+gl_v174]&gl_v115);
                gl_v154->gl_v139.gl_v137[gl_v181]=(unsigned short)gl_v160;
                if(gl_v174!=gl_v162){
                    gl_v181=gl_v154->gl_v139.gl_v138[gl_v162]+gl_v154->gl_v139.gl_v136[gl_v180*64+gl_v162]
                       +__popc(gl_v154->gl_v139.gl_v135[gl_v180*64+gl_v162]&gl_v115);
                    gl_v154->gl_v139.gl_v137[gl_v181]=(unsigned short)gl_v160;
                }
            }
        }
        __syncthreads();
        int gl_v182=0;
        unsigned char *gl_v124=gl_v154->gl_v139.gl_v131;
        for(int gl_v183=0;gl_v183<8;++gl_v183){
            const unsigned char *gl_v184=(gl_v183&1)?gl_v154->gl_v139.gl_v132:gl_v154->gl_v139.gl_v131;
            unsigned char *gl_v185=(gl_v183&1)?gl_v154->gl_v139.gl_v131:gl_v154->gl_v139.gl_v132;
            int gl_v186=0;
            int gl_v4=gl_v156&31,gl_v5=gl_v156>>5,gl_v187=blockDim.x>>5;
            for(int gl_v165=gl_v5;gl_v165<gl_v70 && gl_v165<64;gl_v165+=gl_v187){
                int gl_v188=gl_v71-gl_v154->gl_v139.gl_v133[gl_v165];
                int gl_v189=gl_v154->gl_v139.gl_v138[gl_v165],gl_v190=gl_v154->gl_v139.gl_v138[gl_v165+1];
                for(int gl_v164=gl_v189;gl_v164<gl_v190;gl_v164+=32){
                    int gl_v191=gl_v164+gl_v4;
                    int gl_v160=gl_v191<gl_v190?(int)gl_v154->gl_v139.gl_v137[gl_v191]:-1;
                    int gl_v117=gl_v160>=0?gl_v154->gl_v139.gl_v117[gl_v160]:-1;
                    int gl_v192=gl_v117>=0 && (gl_v117&63)==gl_v165;
                    int gl_v193=gl_v117>=0 && ((gl_v117>>6)&63)==gl_v165?(int)gl_v184[gl_v160]:0;
                    int gl_v194=gl_v193-gl_v192,gl_v179=gl_v194;
                    for(int gl_v3=1;gl_v3<32;gl_v3<<=1){
                        int gl_v195=__shfl_up_sync(0xffffffffu,gl_v179,gl_v3);
                        if(gl_v4>=gl_v3)gl_v179+=gl_v195;
                    }
                    int gl_v196=gl_v188+gl_v179-gl_v194-gl_v192;
                    int gl_v197=gl_v196;
                    for(int gl_v3=1;gl_v3<32;gl_v3<<=1){
                        int gl_v195=__shfl_up_sync(0xffffffffu,gl_v197,gl_v3);
                        if(gl_v4>=gl_v3)gl_v197=min(gl_v197,gl_v195);
                    }
                    int gl_v198=__shfl_up_sync(0xffffffffu,gl_v197,1);
                    if(gl_v4==0)gl_v198=0;
                    if(gl_v192){
                        int gl_v199=gl_v196-min(0,gl_v198)>=0;
                        gl_v185[gl_v160]=(unsigned char)gl_v199;
                        gl_v186|=gl_v199!=(int)gl_v184[gl_v160];
                    }
                    gl_v188+=__shfl_sync(0xffffffffu,gl_v179,31)
                          -min(0,__shfl_sync(0xffffffffu,gl_v197,31));
                }
                if(gl_v4==0)gl_v154->gl_v139.gl_v134[gl_v165]=gl_v71-gl_v188;
            }
            int gl_v200=__syncthreads_or(gl_v186);
            if(!gl_v200){gl_v182=1;gl_v124=gl_v185;break;}
        }
        if(!gl_v182){
            if(gl_v156<gl_v70 && gl_v156<64)gl_v147[gl_v156]=gl_v154->gl_v139.gl_v133[gl_v156];
            __syncthreads();
            int gl_v178=gl_a1(gl_v157,gl_v155);
            int gl_v201=gl_a6(gl_v145-gl_v158,gl_v146,gl_v70,gl_v71,gl_v48,gl_v147,gl_v148+gl_v158,
                                           gl_v149,gl_v150,gl_v151,0,0,gl_v154,gl_v155);
            int gl_v202=gl_v178+gl_v201;
            if(gl_v202>0){
                int gl_v175=min(gl_v145,max(gl_v152,gl_v202*gl_v153));
                for(int gl_v176=gl_v156;gl_v176<gl_v175;gl_v176+=blockDim.x){
                    int gl_v82=__ldcg(&gl_v148[gl_v176])>>6;
                    if((unsigned)gl_v82<(unsigned)gl_v146)gl_v149[gl_v82]=gl_v150+gl_v151;
                }
            }
            return gl_v202;
        }
        if(gl_v156<gl_v70 && gl_v156<64)gl_v154->gl_v139.gl_v133[gl_v156]=gl_v154->gl_v139.gl_v134[gl_v156];
        for(int gl_v160=gl_v156;gl_v160<gl_v159;gl_v160+=blockDim.x){
            if(gl_v124[gl_v160]){
                int gl_v117=gl_v154->gl_v139.gl_v117[gl_v160];
                if(gl_v117>=0){gl_v48[gl_v117>>12]=gl_v117&63;++gl_v157;}
            }
        }
        __syncthreads();
    }
    if(gl_v156<gl_v70 && gl_v156<64)gl_v147[gl_v156]=gl_v154->gl_v139.gl_v133[gl_v156];
    int gl_v202=gl_a1(gl_v157,gl_v155);
    if(gl_v202>0){
        int gl_v175=min(gl_v145,max(gl_v152,gl_v202*gl_v153));
        for(int gl_v176=gl_v156;gl_v176<gl_v175;gl_v176+=blockDim.x){
            int gl_v82=__ldcg(&gl_v148[gl_v176])>>6;
            if((unsigned)gl_v82<(unsigned)gl_v146)gl_v149[gl_v82]=gl_v150+gl_v151;
        }
    }
    return gl_v202;
}

static __device__ __forceinline__ int gl_a8(
    int gl_v145,int gl_v146,int gl_v70,int gl_v71,int *gl_v48,int *gl_v147,
    const int *gl_v148,int *gl_v149,int gl_v150,int gl_v151,int gl_v152,int gl_v153,
    gl_s1 *gl_v154,int *gl_v155,const int *gl_v203,const int *gl_v204)
{
    int gl_v156=threadIdx.x,gl_v4=gl_v156&31;
    int gl_v205=0;
    if(gl_v156<64){
        int gl_v10=gl_v156<gl_v70?__ldcg(&gl_v147[gl_v156]):0;
        int gl_v112=gl_v156<gl_v70?max(0,gl_v204[gl_v156]-max(0,gl_v71-gl_v10)):0;
        gl_v154->gl_v128.gl_v110[gl_v156]=gl_v10;gl_v154->gl_v128.gl_v111[gl_v156]=0;
        gl_v154->gl_v128.gl_v112[gl_v156]=gl_v112;
        gl_v205|=gl_v112>8 || (gl_v156<gl_v70 && gl_v10>gl_v71);
        int gl_v206=gl_v112;
        for(int gl_v3=1;gl_v3<32;gl_v3<<=1){int gl_v1=__shfl_up_sync(0xffffffffu,gl_v206,gl_v3);if(gl_v4>=gl_v3)gl_v206+=gl_v1;}
        gl_v154->gl_v128.gl_v113[gl_v156+1]=gl_v206;
    }
    if(gl_v156==0)gl_v154->gl_v128.gl_v113[0]=0;
    if(__syncthreads_or(gl_v205))
        return gl_a7(gl_v145,gl_v146,gl_v70,gl_v71,gl_v48,gl_v147,gl_v148,gl_v149,
                                 gl_v150,gl_v151,gl_v152,gl_v153,gl_v154,gl_v155);
    if(gl_v156>=32 && gl_v156<64)gl_v154->gl_v128.gl_v113[gl_v156+1]+=gl_v154->gl_v128.gl_v113[32];
    __syncthreads();
    int gl_v207=gl_v154->gl_v128.gl_v113[64];
    for(int gl_v208=gl_v156;gl_v208<512;gl_v208+=blockDim.x){
        int gl_v15=gl_v208>>3,gl_v176=gl_v208&7;
        if(gl_v176<gl_v154->gl_v128.gl_v112[gl_v15]){
            int gl_v191=gl_v154->gl_v128.gl_v113[gl_v15]+gl_v176;
            int gl_v126=__ldcg(&gl_v203[gl_v208]);
            gl_v154->gl_v128.gl_v126[gl_v191]=gl_v126;
            gl_v154->gl_v128.gl_v115[gl_v191]=0;gl_v154->gl_v128.gl_v116[gl_v191]=0;
            gl_v154->gl_v128.gl_v127[gl_v191]=((unsigned)gl_v126<<9)|(unsigned)gl_v191;
            if((unsigned)gl_v126>=(unsigned)gl_v145){gl_v205=1;continue;}
            int gl_v161=__ldcg(&gl_v148[gl_v126]),gl_v82=gl_v161>>6,gl_v162=gl_v161&63;
            int gl_v174=(unsigned)gl_v82<(unsigned)gl_v146?__ldcg(&gl_v48[gl_v82]):-1;
            if((unsigned)gl_v174>=(unsigned)gl_v70 || gl_v162!=gl_v15){gl_v205=1;continue;}
            gl_v154->gl_v128.gl_v117[gl_v191]=(gl_v82<<12)|(gl_v174<<6)|gl_v162;
        }
    }
    if(__syncthreads_or(gl_v205))
        return gl_a7(gl_v145,gl_v146,gl_v70,gl_v71,gl_v48,gl_v147,gl_v148,gl_v149,
                                 gl_v150,gl_v151,gl_v152,gl_v153,gl_v154,gl_v155);
    for(int gl_v160=gl_v156;gl_v160<gl_v145;gl_v160+=blockDim.x){
        int gl_v161=__ldcg(&gl_v148[gl_v160]),gl_v82=gl_v161>>6,gl_v162=gl_v161&63;
        int gl_v174=(unsigned)gl_v82<(unsigned)gl_v146?__ldcg(&gl_v48[gl_v82]):-1;
        if((unsigned)gl_v174>=(unsigned)gl_v70 || gl_v162>=gl_v70){gl_v205=1;continue;}
        atomicAdd(&gl_v154->gl_v128.gl_v111[gl_v174],1);
        int gl_v189=gl_v154->gl_v128.gl_v113[gl_v174],gl_v190=gl_v154->gl_v128.gl_v113[gl_v174+1];
        for(int gl_v176=gl_v189;gl_v176<gl_v190;++gl_v176)if(gl_v160<gl_v154->gl_v128.gl_v126[gl_v176])atomicAdd(&gl_v154->gl_v128.gl_v115[gl_v176],1);
    }
    __syncthreads();
    if(gl_v156<gl_v70 && gl_v156<64)
        gl_v205|=gl_v154->gl_v128.gl_v111[gl_v156]>0 && gl_v154->gl_v128.gl_v110[gl_v156]-gl_v154->gl_v128.gl_v111[gl_v156]<1;
    if(__syncthreads_or(gl_v205))
        return gl_a7(gl_v145,gl_v146,gl_v70,gl_v71,gl_v48,gl_v147,gl_v148,gl_v149,
                                 gl_v150,gl_v151,gl_v152,gl_v153,gl_v154,gl_v155);
    int gl_v209=1;while(gl_v209<gl_v207)gl_v209<<=1;
    if(gl_v209<=(int)blockDim.x){
        unsigned gl_v210=gl_v156<gl_v207?gl_v154->gl_v128.gl_v127[gl_v156]:~0u;
        __syncthreads();
        for(int gl_v211=2;gl_v211<=gl_v209;gl_v211<<=1)for(int gl_v212=gl_v211>>1;gl_v212;gl_v212>>=1){
            unsigned gl_v195;
            if(gl_v212<32)gl_v195=__shfl_xor_sync(0xffffffffu,gl_v210,gl_v212);
            else{
                gl_v154->gl_v128.gl_v127[gl_v156]=gl_v210;__syncthreads();
                gl_v195=gl_v154->gl_v128.gl_v127[gl_v156^gl_v212];__syncthreads();
            }
            bool gl_v213=((gl_v156&gl_v212)==0)==((gl_v156&gl_v211)==0);
            gl_v210=gl_v213?min(gl_v210,gl_v195):max(gl_v210,gl_v195);
        }
        if(gl_v156<gl_v207)gl_v154->gl_v128.gl_v127[gl_v156]=gl_v210;
        __syncthreads();
    }else{
        for(int gl_v160=gl_v156;gl_v160<gl_v209;gl_v160+=blockDim.x)if(gl_v160>=gl_v207)gl_v154->gl_v128.gl_v127[gl_v160]=~0u;
        __syncthreads();
        for(int gl_v211=2;gl_v211<=gl_v209;gl_v211<<=1)for(int gl_v212=gl_v211>>1;gl_v212;gl_v212>>=1){
            for(int gl_v160=gl_v156;gl_v160<gl_v209;gl_v160+=blockDim.x){int gl_v214=gl_v160^gl_v212;if(gl_v214>gl_v160){
                unsigned gl_v215=gl_v154->gl_v128.gl_v127[gl_v160],gl_v216=gl_v154->gl_v128.gl_v127[gl_v214];
                if((gl_v160&gl_v211)==0?gl_v215>gl_v216:gl_v215<gl_v216){gl_v154->gl_v128.gl_v127[gl_v160]=gl_v216;gl_v154->gl_v128.gl_v127[gl_v214]=gl_v215;}
            }}
            __syncthreads();
        }
    }
    if(gl_v156<32){
        int gl_v217=0,gl_v218=0,gl_v116=0;
        for(int gl_v176=0;gl_v176<gl_v207;++gl_v176){
            int gl_v191=(int)(gl_v154->gl_v128.gl_v127[gl_v176]&511u);
            int gl_v117=gl_v154->gl_v128.gl_v117[gl_v191];
            int gl_v162=gl_v117&63,gl_v174=(gl_v117>>6)&63;
            int gl_v219=gl_v191-gl_v154->gl_v128.gl_v113[gl_v162];
            int gl_v220=gl_v154->gl_v128.gl_v115[gl_v191]-gl_v219;
            int gl_v194=__shfl_sync(0xffffffffu,gl_v162<32?gl_v217:gl_v218,gl_v162&31);
            if(gl_v220<=gl_v194){
                gl_v217+=(int)(gl_v174==gl_v156)-(int)(gl_v162==gl_v156);
                gl_v218+=(int)(gl_v174==gl_v156+32)-(int)(gl_v162==gl_v156+32);
                ++gl_v116;if(gl_v156==0)gl_v154->gl_v128.gl_v116[gl_v191]=1;
            }
        }
        if(gl_v156<gl_v70)gl_v147[gl_v156]=gl_v154->gl_v128.gl_v110[gl_v156]-gl_v154->gl_v128.gl_v111[gl_v156]+gl_v204[gl_v156]+gl_v217;
        if(gl_v156+32<gl_v70)gl_v147[gl_v156+32]=gl_v154->gl_v128.gl_v110[gl_v156+32]-gl_v154->gl_v128.gl_v111[gl_v156+32]+gl_v204[gl_v156+32]+gl_v218;
        if(gl_v156==0)gl_v154->gl_v128.gl_v124=gl_v145-gl_v116;
    }
    __syncthreads();
    for(int gl_v160=gl_v156;gl_v160<gl_v145;gl_v160+=blockDim.x){int gl_v161=__ldcg(&gl_v148[gl_v160]);gl_v48[gl_v161>>6]=gl_v161&63;}
    __syncthreads();
    for(int gl_v176=gl_v156;gl_v176<gl_v207;gl_v176+=blockDim.x)if(gl_v154->gl_v128.gl_v116[gl_v176]){
        int gl_v117=gl_v154->gl_v128.gl_v117[gl_v176];gl_v48[gl_v117>>12]=(gl_v117>>6)&63;
    }
    int gl_v124=gl_v154->gl_v128.gl_v124;
    if(gl_v124>0){
        int gl_v175=min(gl_v145,max(gl_v152,gl_v124*gl_v153));
        for(int gl_v160=gl_v156;gl_v160<gl_v175;gl_v160+=blockDim.x){int gl_v82=__ldcg(&gl_v148[gl_v160])>>6;gl_v149[gl_v82]=gl_v150+gl_v151;}
    }
    return gl_v124;
}

static __device__ __forceinline__ void gl_a9(
    unsigned long long *gl_v127,int gl_v47,int gl_v221,gl_s1 *gl_v154,int *gl_v222)
{
    int gl_v156=threadIdx.x, gl_v223=blockDim.x, gl_v4=gl_v156&31;
    int gl_v224=32;
    while(gl_v224<gl_v221 && gl_v224<4096)gl_v224<<=1;
    gl_v224=min(4096,gl_v224*2);
    int gl_v225=gl_v47;
    if(gl_v47>gl_v224){
        unsigned long long gl_v226=~0ULL,gl_v227=0ULL;
        for(int gl_v176=gl_v156;gl_v176<gl_v47;gl_v176+=gl_v223){
            unsigned long long gl_v228=__ldcg(&gl_v127[gl_v176]);
            gl_v226=min(gl_v226,gl_v228);gl_v227=max(gl_v227,gl_v228);
        }
        for(int gl_v3=16;gl_v3>0;gl_v3>>=1){
            gl_v226=min(gl_v226,__shfl_xor_sync(0xffffffffu,gl_v226,gl_v3));
            gl_v227=max(gl_v227,__shfl_xor_sync(0xffffffffu,gl_v227,gl_v3));
        }
        if(gl_v4==0){gl_v154->gl_v129[gl_v156>>5]=gl_v226;gl_v154->gl_v129[16+(gl_v156>>5)]=gl_v227;}
        __syncthreads();
        gl_v226=~0ULL;gl_v227=0ULL;
        for(int w=0;w<(gl_v223>>5);++w){gl_v226=min(gl_v226,gl_v154->gl_v129[w]);gl_v227=max(gl_v227,gl_v154->gl_v129[16+w]);}
        __syncthreads();
        int gl_v229=max(0,63-__clzll(gl_v226^gl_v227)-7);
        unsigned long long gl_v230=gl_v229==56?0ULL:(~0ULL<<(gl_v229+8));
        unsigned long long gl_v178=gl_v226&gl_v230,gl_v231=~0ULL;
        int gl_v126=gl_v221,gl_v232=0;
        for(int gl_v233=gl_v229;gl_v233>=0;gl_v233=(gl_v233>8?gl_v233-8:(gl_v233>0?0:-1))){
            for(int gl_v176=gl_v156;gl_v176<256;gl_v176+=gl_v223)gl_v154->gl_v130[gl_v176]=0;
            __syncthreads();
            for(int gl_v158=0;gl_v158<gl_v47;gl_v158+=gl_v223){
                int gl_v176=gl_v158+gl_v156;
                unsigned long long gl_v228=gl_v176<gl_v47?__ldcg(&gl_v127[gl_v176]):0ULL;
                int gl_v234=(gl_v176<gl_v47 && (gl_v228&gl_v230)==gl_v178)?(int)((gl_v228>>gl_v233)&255):256;
                unsigned gl_v235=__match_any_sync(0xffffffffu,gl_v234);
                if(gl_v234<256 && gl_v4==__ffs(gl_v235)-1)atomicAdd(&gl_v154->gl_v130[gl_v234],__popc(gl_v235));
            }
            __syncthreads();
            if(gl_v156<32){
                for(int gl_v10=0;gl_v10<8;++gl_v10){
                    unsigned gl_v210=gl_v154->gl_v130[gl_v10*32+gl_v4];
                    for(int gl_v3=16;gl_v3>0;gl_v3>>=1)gl_v210+=__shfl_xor_sync(0xffffffffu,gl_v210,gl_v3);
                    if(gl_v4==0)gl_v154->gl_v130[256+gl_v10]=gl_v210;
                }
                __syncwarp(0xffffffffu);
                if(gl_v4==0){
                    int gl_v206=0,gl_v236=0;
                    while(gl_v236<7 && gl_v206+(int)gl_v154->gl_v130[256+gl_v236]<gl_v126){gl_v206+=(int)gl_v154->gl_v130[256+gl_v236];++gl_v236;}
                    gl_v222[0]=gl_v236;gl_v222[1]=gl_v126-gl_v206;gl_v222[2]=gl_v232+gl_v206;
                }
                __syncwarp(0xffffffffu);
                int gl_v236=gl_v222[0],gl_v16=gl_v222[1];
                unsigned gl_v210=gl_v154->gl_v130[gl_v236*32+gl_v4],gl_v179=gl_v210;
                for(int gl_v3=1;gl_v3<32;gl_v3<<=1){
                    unsigned gl_v195=__shfl_up_sync(0xffffffffu,gl_v179,gl_v3);
                    if(gl_v4>=gl_v3)gl_v179+=gl_v195;
                }
                unsigned gl_v237=__ballot_sync(0xffffffffu,(int)gl_v179>=gl_v16 && (int)(gl_v179-gl_v210)<gl_v16);
                int gl_v238=__ffs(gl_v237)-1;
                int gl_v115=(int)__shfl_sync(0xffffffffu,gl_v179-gl_v210,gl_v238);
                int gl_v239=(int)__shfl_sync(0xffffffffu,gl_v210,gl_v238);
                __syncwarp(0xffffffffu);
                if(gl_v4==0){gl_v222[0]=gl_v236*32+gl_v238;gl_v222[1]=gl_v16-gl_v115;gl_v222[2]+=gl_v115;gl_v222[3]=gl_v239;}
            }
            __syncthreads();
            int gl_v240=gl_v222[0];gl_v126=gl_v222[1];gl_v232=gl_v222[2];
            gl_v178|=(unsigned long long)(unsigned)gl_v240<<gl_v233;
            gl_v230|=255ULL<<gl_v233;
            gl_v231=gl_v178|(gl_v233?((1ULL<<gl_v233)-1ULL):0ULL);
            if(gl_v232+gl_v222[3]<=gl_v224)break;
        }
        __syncthreads();
        if(gl_v156==0)gl_v222[3]=0;
        __syncthreads();
        for(int gl_v176=gl_v156;gl_v176<gl_v47;gl_v176+=gl_v223){
            unsigned long long gl_v228=__ldcg(&gl_v127[gl_v176]);
            if(gl_v228<=gl_v231){int gl_v241=atomicAdd(&gl_v222[3],1);gl_v154->gl_v129[gl_v241]=gl_v228;}
        }
        __syncthreads();
        gl_v225=gl_v222[3];
    }else{
        for(int gl_v176=gl_v156;gl_v176<gl_v47;gl_v176+=gl_v223)gl_v154->gl_v129[gl_v176]=__ldcg(&gl_v127[gl_v176]);
        __syncthreads();
    }
    if(gl_v225<=gl_v223){
        unsigned long long gl_v210=gl_v156<gl_v225?gl_v154->gl_v129[gl_v156]:~0ULL;
        __syncthreads();
        int gl_v242=1;while(gl_v242<gl_v225)gl_v242<<=1;
        for(int gl_v211=2;gl_v211<=gl_v242;gl_v211<<=1){
            for(int gl_v243=gl_v211>>1;gl_v243>0;gl_v243>>=1){
                unsigned long long gl_v195;
                if(gl_v243<32){
                    gl_v195=__shfl_xor_sync(0xffffffffu,gl_v210,gl_v243);
                }else{
                    gl_v154->gl_v129[gl_v156]=gl_v210;
                    __syncthreads();
                    gl_v195=gl_v154->gl_v129[gl_v156^gl_v243];
                    __syncthreads();
                }
                bool gl_v213=((gl_v156&gl_v243)==0)==((gl_v156&gl_v211)==0);
                gl_v210=gl_v213?min(gl_v210,gl_v195):max(gl_v210,gl_v195);
            }
        }
        if(gl_v156<gl_v221)gl_v127[gl_v156]=gl_v210;
        __syncthreads();
    }else{
        int gl_v209=1;while(gl_v209<gl_v225)gl_v209<<=1;
        for(int gl_v176=gl_v156;gl_v176<gl_v209;gl_v176+=gl_v223)if(gl_v176>=gl_v225)gl_v154->gl_v129[gl_v176]=~0ULL;
        __syncthreads();
        for(int gl_v211=2;gl_v211<=gl_v209;gl_v211<<=1){
            for(int gl_v243=gl_v211>>1;gl_v243>0;gl_v243>>=1){
                for(int gl_v176=gl_v156;gl_v176<gl_v209;gl_v176+=gl_v223){
                    int gl_v214=gl_v176^gl_v243;
                    if(gl_v214>gl_v176){
                        unsigned long long gl_v215=gl_v154->gl_v129[gl_v176],gl_v216=gl_v154->gl_v129[gl_v214];
                        if((gl_v176&gl_v211)==0?gl_v215>gl_v216:gl_v215<gl_v216){gl_v154->gl_v129[gl_v176]=gl_v216;gl_v154->gl_v129[gl_v214]=gl_v215;}
                    }
                }
                __syncthreads();
            }
        }
        for(int gl_v176=gl_v156;gl_v176<gl_v221;gl_v176+=gl_v223)gl_v127[gl_v176]=gl_v154->gl_v129[gl_v176];
        __syncthreads();
    }
}

#undef gl_m25
#define gl_m25(gl_v47) __ldcg(&gl_v48[(gl_v47)])
static __device__ __forceinline__ unsigned long long gl_a10(
    unsigned long long x,unsigned long long gl_v47,const ulonglong2 *gl_v244) {
    ulonglong2 gl_v245=__ldg(&gl_v244[gl_v47]);return gl_v245.x*x+gl_v245.y;
}

static __device__ __forceinline__ int gl_a11(int *gl_v133,int gl_v162) {
    unsigned gl_v246=__activemask();
    unsigned gl_v247=__match_any_sync(gl_v246,gl_v162);
    int gl_v4=threadIdx.x&31,gl_v248=__ffs(gl_v247)-1,gl_v158=0;
    if(gl_v4==gl_v248)gl_v158=atomicAdd(gl_v133+gl_v162,__popc(gl_v247));
    gl_v158=__shfl_sync(gl_v246,gl_v158,gl_v248);
    return gl_v158+__popc(gl_v247&((1u<<gl_v4)-1u));
}
static __device__ __forceinline__ unsigned gl_a12(const unsigned *gl_v15){
    unsigned gl_v210;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(gl_v210):"l"(gl_v15):"memory");return gl_v210;
}
static __device__ __forceinline__ void gl_a13(unsigned *gl_v249,unsigned gl_v162){
    __syncthreads();
    if(threadIdx.x==0){
        asm volatile(
            "{\n\t"
            ".reg .u32 _prev;\n\t"
            "atom.acq_rel.gpu.global.add.u32 _prev, [%0], 1;\n\t"
            "}\n"
            :: "l"(gl_v249) : "memory");
        while(gl_a12(gl_v249)<gl_v162)__nanosleep(20);
    }
    __syncthreads();
}

#define gl_m28(gl_v250,gl_v27) gl_a13((gl_v250),(gl_v27))

static __device__ __forceinline__ int gl_a14(int *gl_v133,int gl_v162){
    unsigned gl_v246=__activemask(),gl_v235=__match_any_sync(gl_v246,gl_v162);
    int gl_v4=threadIdx.x&31,gl_v248=__ffs(gl_v235)-1,gl_v158=0;
    if(gl_v4==gl_v248)gl_v158=atomicAdd(gl_v133+gl_v162,__popc(gl_v235));
    gl_v158=__shfl_sync(gl_v246,gl_v158,gl_v248);
    return gl_v158+__popc(gl_v235&((1u<<gl_v4)-1u));
}

static __device__ __forceinline__ int gl_a15(
    int *gl_v133,const unsigned *gl_v251,int gl_v252,int gl_v165,bool gl_v253){
    const unsigned gl_v254=__ldg(gl_v251+gl_v252),gl_v255=gl_v254&3u,gl_v158=gl_v254>>2;
    const unsigned gl_v256=8u<<gl_v255,gl_v257=2u-gl_v255;
    const unsigned gl_v233=((unsigned)gl_v165&((1u<<gl_v257)-1u))*gl_v256;
    unsigned *gl_v180=(unsigned*)(gl_v133+gl_v158+((unsigned)gl_v165>>gl_v257));
    const unsigned gl_v258=1u<<gl_v233;
    const unsigned gl_v259=atomicAdd(gl_v180,gl_v253?0u-gl_v258:gl_v258);
    const unsigned gl_v230=gl_v255==2u?~0u:((1u<<gl_v256)-1u);
    return (int)((gl_v259>>gl_v233)&gl_v230);
}

static __device__ __forceinline__ int gl_a16(
    int gl_v145,int gl_v146,int gl_v70,int gl_v71,int *gl_v48,int *gl_v72,
    const unsigned long long *gl_v260,const int *gl_v204,const int *gl_v113,
    int *gl_v149,int gl_v150,int gl_v151,int gl_v152,int gl_v153,
    gl_s1 *gl_v154,const int *gl_v261,const ulonglong2 *gl_v135, int *gl_v262){
    const int gl_v156=threadIdx.x,gl_v4=gl_v156&31;
    if(blockDim.x<128 || __ldcg(gl_v261+576))return -1;
    int gl_v205=0;
    if(gl_v156<64){
        int gl_v110=gl_v156<gl_v70?__ldcg(gl_v72+gl_v156):0,gl_v263=__ldcg(gl_v261+gl_v156);
        int gl_v112=gl_v156<gl_v70?max(0,gl_v204[gl_v156]-max(0,gl_v71-gl_v110)):0;
        gl_v154->gl_v125.gl_v110[gl_v156]=gl_v110;gl_v154->gl_v125.gl_v111[gl_v156]=gl_v263;gl_v154->gl_v125.gl_v112[gl_v156]=gl_v112;
        gl_v154->gl_v125.gl_v114[gl_v156]=gl_v110-gl_v263+gl_v204[gl_v156];
        gl_v205|=gl_v112>2 || gl_v110>gl_v71 || gl_v110<0 || (gl_v263>0 && gl_v110-gl_v263<1);
    }
    if(__syncthreads_or(gl_v205))return -1;
    const int gl_v162=gl_v156>>1,gl_v264=gl_v156&1;
    const bool gl_v265=gl_v156<128 && gl_v162<gl_v70 && gl_v264<gl_v154->gl_v125.gl_v112[gl_v162];
    unsigned long long gl_v228=0;int gl_v117=-1,gl_v266=0;
    if(gl_v265){
        ulonglong2 gl_v267=gl_a4(gl_v135+gl_v162*8+gl_v264);
        gl_v228=gl_v267.x;gl_v117=(int)(unsigned)gl_v267.y;
        gl_v266=__ldcg(gl_v261+64+gl_v162*8+gl_v264)-gl_v264;
        gl_v205|=(gl_v117&63)!=gl_v162 || ((gl_v117>>6)&63)>=gl_v70 || (unsigned)(gl_v117>>12)>=(unsigned)gl_v146;
    }
    if(gl_v156<128){gl_v154->gl_v125.gl_v118[gl_v156]=gl_v228;gl_v154->gl_v125.gl_v117[gl_v156]=gl_v117;}
    if(__syncthreads_or(gl_v205))return -1;
    bool gl_v268=gl_v265 && gl_v266<=0,gl_v182=false;int gl_v269=0;
    for(int gl_v183=0;gl_v183<6;++gl_v183){
        unsigned gl_v270=__ballot_sync(0xffffffffu,gl_v268);
        if(gl_v4==0 && gl_v156<128)gl_v154->gl_v125.gl_v122[gl_v156>>5]=(int)gl_v270;
        gl_v269=__syncthreads_count(gl_v268);
        if(gl_v269>64)return -1;
        bool gl_v185=false;
        if(gl_v265){
            int gl_v194=0;
            for(int gl_v180=0;gl_v180<4;++gl_v180){
                unsigned gl_v271=(unsigned)gl_v154->gl_v125.gl_v122[gl_v180];
                while(gl_v271){
                    int gl_v272=gl_v180*32+__ffs(gl_v271)-1;gl_v271&=gl_v271-1;
                    if(gl_v154->gl_v125.gl_v118[gl_v272]<gl_v228){
                        int gl_v267=gl_v154->gl_v125.gl_v117[gl_v272];
                        gl_v194+=(((gl_v267>>6)&63)==gl_v162)-((gl_v267&63)==gl_v162);
                    }
                }
            }
            gl_v185=gl_v266<=gl_v194;
        }
        bool gl_v186=__syncthreads_or(gl_v185!=gl_v268);
        gl_v268=gl_v185;
        if(!gl_v186){gl_v182=true;break;}
    }
    if(!gl_v182)return -1;
    const int gl_v124=gl_v145-gl_v269;
    int gl_v175=gl_v124>0?min(gl_v145,max(gl_v152,gl_v124*gl_v153)):0;
    if(gl_v175!=0 && gl_v175!=gl_v145)return -1;
    if(gl_v268){
        int gl_v174=(gl_v117>>6)&63;
        atomicAdd(gl_v154->gl_v125.gl_v114+gl_v174,1);
        atomicSub(gl_v154->gl_v125.gl_v114+gl_v162,1);
    }
    __syncthreads();
    if(gl_v156<gl_v70 && gl_v156<64)gl_v72[gl_v156]=gl_v154->gl_v125.gl_v114[gl_v156];
    if(gl_v268){int gl_v191=gl_v113[gl_v162]+max(0,gl_v71-gl_v154->gl_v125.gl_v110[gl_v162])+gl_v264;gl_v262[gl_v191]=1;}
    return gl_v124;
}
static __device__ __forceinline__ int gl_a17(
    int gl_v145,int gl_v146,int gl_v70,int gl_v71,int *gl_v48,int *gl_v72,
    const unsigned long long *gl_v260,const int *gl_v204,const int *gl_v113,
    int *gl_v149,int gl_v150,int gl_v151,int gl_v152,int gl_v153,
    gl_s1 *gl_v154,const int *gl_v261,const ulonglong2 *gl_v135, int *gl_v262){
    int gl_v273=gl_a16(gl_v145,gl_v146,gl_v70,gl_v71,gl_v48,gl_v72,gl_v260,gl_v204,gl_v113,
        gl_v149,gl_v150,gl_v151,gl_v152,gl_v153,gl_v154,gl_v261,gl_v135,gl_v262);
    if(gl_v273>=0)return gl_v273;

    const int gl_v156=threadIdx.x,gl_v4=gl_v156&31;
    if(blockDim.x<512 || __ldcg(gl_v261+576))return -1;
    int gl_v205=0;
    if(gl_v156<64){
        int gl_v110=gl_v156<gl_v70?__ldcg(gl_v72+gl_v156):0,gl_v263=__ldcg(gl_v261+gl_v156);
        int gl_v112=gl_v156<gl_v70?max(0,gl_v204[gl_v156]-max(0,gl_v71-gl_v110)):0;
        gl_v154->gl_v125.gl_v110[gl_v156]=gl_v110;gl_v154->gl_v125.gl_v111[gl_v156]=gl_v263;gl_v154->gl_v125.gl_v112[gl_v156]=gl_v112;
        gl_v154->gl_v125.gl_v114[gl_v156]=gl_v110-gl_v263+gl_v204[gl_v156];
        gl_v205|=gl_v112>8 || gl_v110>gl_v71 || gl_v110<0 || (gl_v263>0 && gl_v110-gl_v263<1);
    }
    if(__syncthreads_or(gl_v205))return -1;
    const int gl_v162=gl_v156>>3,gl_v264=gl_v156&7;
    const bool gl_v265=gl_v156<512 && gl_v162<gl_v70 && gl_v264<gl_v154->gl_v125.gl_v112[gl_v162];
    unsigned long long gl_v228=0;int gl_v117=-1,gl_v266=0;
    if(gl_v265){
        ulonglong2 gl_v267=gl_a4(gl_v135+gl_v156);
        gl_v228=gl_v267.x;gl_v117=(int)(unsigned)gl_v267.y;
        gl_v266=__ldcg(gl_v261+64+gl_v156)-gl_v264;
        gl_v205|=(gl_v117&63)!=gl_v162 || ((gl_v117>>6)&63)>=gl_v70 || (unsigned)(gl_v117>>12)>=(unsigned)gl_v146;
    }
    if(gl_v156<512){gl_v154->gl_v125.gl_v118[gl_v156]=gl_v228;gl_v154->gl_v125.gl_v117[gl_v156]=gl_v117;}
    if(__syncthreads_or(gl_v205))return -1;
    bool gl_v268=gl_v265 && gl_v266<=0,gl_v182=false;int gl_v269=0;
    for(int gl_v183=0;gl_v183<6;++gl_v183){
        unsigned gl_v270=__ballot_sync(0xffffffffu,gl_v268);
        if(gl_v4==0)gl_v154->gl_v125.gl_v122[gl_v156>>5]=(int)gl_v270;
        gl_v269=__syncthreads_count(gl_v268);
        if(gl_v269>64)return -1;
        bool gl_v185=false;
        if(gl_v265){
            int gl_v194=0;
            for(int gl_v180=0;gl_v180<16;++gl_v180){
                unsigned gl_v271=(unsigned)gl_v154->gl_v125.gl_v122[gl_v180];
                while(gl_v271){
                    int gl_v272=gl_v180*32+__ffs(gl_v271)-1;gl_v271&=gl_v271-1;
                    if(gl_v154->gl_v125.gl_v118[gl_v272]<gl_v228){
                        int gl_v267=gl_v154->gl_v125.gl_v117[gl_v272];
                        gl_v194+=(((gl_v267>>6)&63)==gl_v162)-((gl_v267&63)==gl_v162);
                    }
                }
            }
            gl_v185=gl_v266<=gl_v194;
        }
        bool gl_v186=__syncthreads_or(gl_v185!=gl_v268);
        gl_v268=gl_v185;
        if(!gl_v186){gl_v182=true;break;}
    }
    if(!gl_v182)return -1;
    const int gl_v124=gl_v145-gl_v269;
    int gl_v175=gl_v124>0?min(gl_v145,max(gl_v152,gl_v124*gl_v153)):0;
    if(gl_v175!=0 && gl_v175!=gl_v145)return -1;
    if(gl_v268){
        int gl_v174=(gl_v117>>6)&63;
        atomicAdd(gl_v154->gl_v125.gl_v114+gl_v174,1);
        atomicSub(gl_v154->gl_v125.gl_v114+gl_v162,1);
    }
    __syncthreads();
    if(gl_v156<gl_v70 && gl_v156<64)gl_v72[gl_v156]=gl_v154->gl_v125.gl_v114[gl_v156];
    if(gl_v268){int gl_v191=gl_v113[gl_v162]+max(0,gl_v71-gl_v154->gl_v125.gl_v110[gl_v162])+gl_v264;gl_v262[gl_v191]=1;}
    return gl_v124;
}
static __device__ __forceinline__ int gl_a18(
    int gl_v145,int gl_v146,int gl_v70,int gl_v71,int *gl_v48,int *gl_v147,
    const unsigned long long *gl_v260,const int *gl_v204,const int *gl_v113,
    int *gl_v149,int gl_v150,int gl_v151,int gl_v152,int gl_v153,
    gl_s1 *gl_v154,const int *gl_v274,const ulonglong2 *gl_v275, int *gl_v262)
{
    const int gl_v156=threadIdx.x,gl_v4=gl_v156&31;
    if(__ldcg(gl_v274+576))return -1;
    int gl_v205=0;
    if(gl_v156<64){
        int gl_v10=gl_v156<gl_v70?__ldcg(&gl_v147[gl_v156]):0;
        int gl_v112=gl_v156<gl_v70?max(0,gl_v204[gl_v156]-max(0,gl_v71-gl_v10)):0;
        gl_v154->gl_v125.gl_v110[gl_v156]=gl_v10;gl_v154->gl_v125.gl_v111[gl_v156]=__ldcg(gl_v274+gl_v156);
        gl_v154->gl_v125.gl_v112[gl_v156]=gl_v112;
        gl_v205|=gl_v112>8 || (gl_v156<gl_v70 && gl_v10>gl_v71);
        int gl_v206=gl_v112;
        for(int gl_v3=1;gl_v3<32;gl_v3<<=1){int gl_v1=__shfl_up_sync(0xffffffffu,gl_v206,gl_v3);if(gl_v4>=gl_v3)gl_v206+=gl_v1;}
        gl_v154->gl_v125.gl_v113[gl_v156+1]=gl_v206;
    }
    if(gl_v156==0)gl_v154->gl_v125.gl_v113[0]=0;
    if(__syncthreads_or(gl_v205))return -1;
    if(gl_v156>=32 && gl_v156<64)gl_v154->gl_v125.gl_v113[gl_v156+1]+=gl_v154->gl_v125.gl_v113[32];
    __syncthreads();
    const int gl_v207=gl_v154->gl_v125.gl_v113[64];
    for(int gl_v208=gl_v156;gl_v208<512;gl_v208+=blockDim.x){
        int gl_v15=gl_v208>>3,gl_v176=gl_v208&7;
        if(gl_v176<gl_v154->gl_v125.gl_v112[gl_v15]){
            int gl_v191=gl_v154->gl_v125.gl_v113[gl_v15]+gl_v176;
            int gl_v276=gl_v113[gl_v15]+max(0,gl_v71-gl_v154->gl_v125.gl_v110[gl_v15])+gl_v176;
            unsigned long long gl_v228=gl_a4(gl_v275+gl_v15*8+gl_v176).x;
            int gl_v82=(int)(unsigned)gl_v228,gl_v162=(((unsigned)(gl_v228>>32))&63)^63;
            int gl_v174=(unsigned)gl_v82<(unsigned)gl_v146?__ldcg(&gl_v48[gl_v82]):-1;
            gl_v154->gl_v125.gl_v118[gl_v191]=gl_v228;
            gl_v154->gl_v125.gl_v115[gl_v191]=__ldcg(gl_v274+64+gl_v15*8+gl_v176);gl_v154->gl_v125.gl_v116[gl_v191]=0;gl_v154->gl_v125.gl_v121[gl_v191]=1;
            if((unsigned)gl_v174>=(unsigned)gl_v70 || gl_v162!=gl_v15){gl_v205=1;continue;}
            gl_v154->gl_v125.gl_v117[gl_v191]=(gl_v82<<12)|(gl_v174<<6)|gl_v162;
        }
    }
    if(__syncthreads_or(gl_v205))return -1;
    if(gl_v156<gl_v70 && gl_v156<64)
        gl_v205|=gl_v154->gl_v125.gl_v111[gl_v156]>0 && gl_v154->gl_v125.gl_v110[gl_v156]-gl_v154->gl_v125.gl_v111[gl_v156]<1;
    if(__syncthreads_or(gl_v205))return -1;

    for(int gl_v183=0;gl_v183<3;++gl_v183){
        if(gl_v156<64)gl_v154->gl_v125.gl_v122[gl_v156]=0;
        __syncthreads();
        for(int gl_v176=gl_v156;gl_v176<gl_v207;gl_v176+=blockDim.x)if(gl_v154->gl_v125.gl_v121[gl_v176]){
            int gl_v174=(gl_v154->gl_v125.gl_v117[gl_v176]>>6)&63;
            atomicAdd(&gl_v154->gl_v125.gl_v122[gl_v174],1);
        }
        __syncthreads();
        int gl_v186=0;
        for(int gl_v176=gl_v156;gl_v176<gl_v207;gl_v176+=blockDim.x)if(gl_v154->gl_v125.gl_v121[gl_v176]){
            int gl_v162=gl_v154->gl_v125.gl_v117[gl_v176]&63;
            int gl_v266=gl_v154->gl_v125.gl_v115[gl_v176]-(gl_v176-gl_v154->gl_v125.gl_v113[gl_v162]);
            if(gl_v266>gl_v154->gl_v125.gl_v122[gl_v162]){gl_v154->gl_v125.gl_v121[gl_v176]=0;gl_v186=1;}
        }
        if(!__syncthreads_or(gl_v186))break;
    }
    if(gl_v156==0)gl_v154->gl_v125.gl_v123=0;
    __syncthreads();
    for(int gl_v176=gl_v156;gl_v176<gl_v207;gl_v176+=blockDim.x)if(gl_v154->gl_v125.gl_v121[gl_v176]){
        unsigned long long gl_v228=gl_v154->gl_v125.gl_v118[gl_v176];
        unsigned long long gl_v277=(gl_v228&0xffffffff00000000ULL)|((unsigned long long)(unsigned)gl_v228<<9)|(unsigned)gl_v176;
        int gl_v241=atomicAdd(&gl_v154->gl_v125.gl_v123,1);gl_v154->gl_v125.gl_v119[gl_v241]=gl_v277;
    }
    __syncthreads();
    const int gl_v278=gl_v154->gl_v125.gl_v123;
    int gl_v209=1;while(gl_v209<gl_v278)gl_v209<<=1;
    if(gl_v209<=(int)blockDim.x){
        unsigned long long gl_v210=gl_v156<gl_v278?gl_v154->gl_v125.gl_v119[gl_v156]:~0ULL;
        __syncthreads();
        for(int gl_v211=2;gl_v211<=gl_v209;gl_v211<<=1)for(int gl_v212=gl_v211>>1;gl_v212;gl_v212>>=1){
            unsigned long long gl_v195;
            if(gl_v212<32)gl_v195=__shfl_xor_sync(0xffffffffu,gl_v210,gl_v212);
            else{gl_v154->gl_v125.gl_v119[gl_v156]=gl_v210;__syncthreads();gl_v195=gl_v154->gl_v125.gl_v119[gl_v156^gl_v212];__syncthreads();}
            bool gl_v213=((gl_v156&gl_v212)==0)==((gl_v156&gl_v211)==0);
            gl_v210=gl_v213?min(gl_v210,gl_v195):max(gl_v210,gl_v195);
        }
        if(gl_v156<gl_v278)gl_v154->gl_v125.gl_v119[gl_v156]=gl_v210;
        __syncthreads();
    }else{
        for(int gl_v160=gl_v156;gl_v160<gl_v209;gl_v160+=blockDim.x)if(gl_v160>=gl_v278)gl_v154->gl_v125.gl_v119[gl_v160]=~0ULL;
        __syncthreads();
        for(int gl_v211=2;gl_v211<=gl_v209;gl_v211<<=1)for(int gl_v212=gl_v211>>1;gl_v212;gl_v212>>=1){
            for(int gl_v160=gl_v156;gl_v160<gl_v209;gl_v160+=blockDim.x){int gl_v214=gl_v160^gl_v212;if(gl_v214>gl_v160){
                unsigned long long gl_v215=gl_v154->gl_v125.gl_v119[gl_v160],gl_v216=gl_v154->gl_v125.gl_v119[gl_v214];
                if((gl_v160&gl_v211)==0?gl_v215>gl_v216:gl_v215<gl_v216){gl_v154->gl_v125.gl_v119[gl_v160]=gl_v216;gl_v154->gl_v125.gl_v119[gl_v214]=gl_v215;}
            }}__syncthreads();
        }
    }
    if(gl_v156<32){
        int gl_v217=0,gl_v218=0,gl_v116=0;
        for(int gl_v176=0;gl_v176<gl_v278;++gl_v176){
            int gl_v191=(int)(gl_v154->gl_v125.gl_v119[gl_v176]&511ULL);
            int gl_v117=gl_v154->gl_v125.gl_v117[gl_v191],gl_v174=(gl_v117>>6)&63,gl_v162=gl_v117&63;
            int gl_v279=gl_v191-gl_v154->gl_v125.gl_v113[gl_v162];
            int gl_v220=gl_v154->gl_v125.gl_v115[gl_v191]-gl_v279;
            int gl_v194=__shfl_sync(0xffffffffu,gl_v162<32?gl_v217:gl_v218,gl_v162&31);
            if(gl_v220<=gl_v194){
                gl_v217+=(gl_v174==gl_v156)-(gl_v162==gl_v156);
                gl_v218+=(gl_v174==gl_v156+32)-(gl_v162==gl_v156+32);
                ++gl_v116;if(gl_v156==0)gl_v154->gl_v125.gl_v116[gl_v191]=1;
            }
        }
        gl_v154->gl_v125.gl_v114[gl_v156]=gl_v154->gl_v125.gl_v110[gl_v156]-gl_v154->gl_v125.gl_v111[gl_v156]+gl_v204[gl_v156]+gl_v217;
        gl_v154->gl_v125.gl_v114[gl_v156+32]=gl_v154->gl_v125.gl_v110[gl_v156+32]-gl_v154->gl_v125.gl_v111[gl_v156+32]+gl_v204[gl_v156+32]+gl_v218;
        if(gl_v156==0){gl_v154->gl_v125.gl_v124=gl_v145-gl_v116;gl_v154->gl_v125.gl_v120=~0ULL;}
    }
    __syncthreads();
    int gl_v124=gl_v154->gl_v125.gl_v124;
    int gl_v175=gl_v124>0?min(gl_v145,max(gl_v152,gl_v124*gl_v153)):0;
    if(gl_v175!=0 && gl_v175!=gl_v145)return -1;
    int gl_v280=gl_v145-gl_v175;
    if(gl_v175>0 && gl_v280>64)return -1;
    if(gl_v175>0 && gl_v280>0 && gl_v156<32){
        int gl_v281=gl_v204[gl_v156],gl_v282=gl_v204[gl_v156+32];
        for(int gl_v283=0;gl_v283<=gl_v280;++gl_v283){
            unsigned long long gl_v215=gl_v281>0?__ldcg(&gl_v260[gl_v113[gl_v156]+gl_v281-1]):0;
            unsigned long long gl_v216=gl_v282>0?__ldcg(&gl_v260[gl_v113[gl_v156+32]+gl_v282-1]):0;
            unsigned long long gl_v284=max(gl_v215,gl_v216);
            for(int gl_v3=16;gl_v3>0;gl_v3>>=1)gl_v284=max(gl_v284,__shfl_xor_sync(0xffffffffu,gl_v284,gl_v3));
            if(gl_v283==gl_v280){if(gl_v156==0)gl_v154->gl_v125.gl_v120=gl_v284;}
            else{if(gl_v215==gl_v284)--gl_v281;if(gl_v216==gl_v284)--gl_v282;}
        }
    }
    __syncthreads();
    if(gl_v156<gl_v70 && gl_v156<64)gl_v147[gl_v156]=gl_v154->gl_v125.gl_v114[gl_v156];
    for(int gl_v176=gl_v156;gl_v176<gl_v207;gl_v176+=blockDim.x)if(gl_v154->gl_v125.gl_v116[gl_v176]){
        int gl_v162=gl_v154->gl_v125.gl_v117[gl_v176]&63;
        int gl_v191=gl_v113[gl_v162]+max(0,gl_v71-gl_v154->gl_v125.gl_v110[gl_v162])+gl_v176-gl_v154->gl_v125.gl_v113[gl_v162];
        gl_v262[gl_v191]=1;
    }
    return gl_v124;
}
extern "C" __global__ __launch_bounds__(512, 1) void gl_k1(
    const int gl_v285,
    const int gl_v42,
    const int gl_v286,
    const int *__restrict__ gl_v287,
    const int *__restrict__ gl_v65,
    int *gl_v48,
    int *gl_v288,
    const int4 *__restrict__ gl_v289,
    const int gl_v290,
    ulonglong2 *gl_v93,
    int *gl_v107,
    unsigned int *gl_v291,
    const unsigned int gl_v292,
    int *gl_v293,
    const int *__restrict__ gl_v294,
    const int gl_v295,
    const int gl_v296,
    const int gl_v297,
    const int gl_v298,
    const int gl_v299,
    const unsigned long long gl_v300,
    const unsigned long long gl_v301,
    const unsigned long long gl_v302,
    int *gl_v303,
    int *gl_v304,
    const int gl_v305,
    int *gl_v306,
    const int gl_v307, const int gl_v308, const int gl_v309,
    const int gl_v310, const int gl_v311, const int gl_v312, const int gl_v313,
    const int gl_v314, const int gl_v315, const int gl_v316,
    const int gl_v317, const int gl_v318, const int gl_v319, const int gl_v320,
    unsigned long long *gl_v321, int *gl_v322,
    unsigned long long *gl_v323, int *gl_v324, int *gl_v325, const ulonglong2 *gl_v326, const int *gl_v327, const int *gl_v328,
    const int gl_v329, int *gl_v330, int *gl_v331,
    int *gl_v332, int *gl_v274,ulonglong2 *gl_v275, const unsigned *gl_v333, const int gl_v334
)  {
    __shared__ gl_s1 gl_v335;
    __shared__ int gl_v336[65],gl_v337[4];
    __shared__ int gl_v338[64], gl_v339[65], gl_v340[4];
    unsigned int gl_v341 = 0;
    int gl_v342 = 0, gl_v343 = 0, gl_v344 = 0;
    int gl_v345 = gl_v319, gl_v346 = gl_v320;
    int gl_v347 = 0, gl_v348 = gl_v320;
    bool gl_v349=false;
    bool gl_v350=false,gl_v351=false;int gl_v352=0;
    for (int gl_v283 = 0; gl_v283 < (int)gl_v301 && gl_v299 + gl_v283 < gl_v307; ++gl_v283) {
        const int gl_v353 = gl_v299 + gl_v283;
        const int gl_v354 = gl_v283 == 0 ? gl_v295 : 0;
        const int gl_v355 = gl_v283 == 0 ? gl_v297 : 0;
        unsigned int gl_v356 = gl_v292 + gl_v341 * gridDim.x;
        const unsigned long long gl_v357 = 123456789ULL + (unsigned long long)gl_v353 + gl_v300;
        const unsigned long long gl_v358 = (unsigned long long)max(0, gl_v308 - gl_v353);
        unsigned long long gl_v359 = 1ULL;
        for (int gl_v360 = 0; gl_v360 < gl_v309; ++gl_v360) gl_v359 *= gl_v358;

    __shared__ int gl_v88[64];
    __shared__ unsigned long long gl_v99, gl_v100[64];
    __shared__ int gl_v2[32];

    if (threadIdx.x < 64) {
        gl_v88[threadIdx.x] = threadIdx.x < gl_v42 ? __ldcg(&gl_v288[threadIdx.x]) : 0;
    }

    const int gl_v361 = gridDim.x * blockDim.x;
    const int gl_v362 = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gl_v4 = threadIdx.x & 31u;
    const int gl_v363 = (int)(blockDim.x >> 5);
    const int gl_v364 = (int)blockIdx.x * gl_v363 + (int)(threadIdx.x >> 5);
    const int gl_v365 = (int)gridDim.x * gl_v363;
    if(gl_v329>=130){
        if(blockIdx.x==0){
            gl_a5(gl_v42,gl_v286,gl_v88,&gl_v99,gl_v100);
            unsigned long long *gl_v366=(unsigned long long*)gl_v331;
            if(threadIdx.x<64)gl_v366[1+threadIdx.x]=gl_v100[threadIdx.x];
            if(threadIdx.x==0)gl_v366[0]=gl_v99;
        }
    }else{
        gl_a5(gl_v42,gl_v286,gl_v88,&gl_v99,gl_v100);
    }

    int4 gl_v367; gl_v367.x = 0; gl_v367.y = 0; gl_v367.z = 0; gl_v367.w = 0;
    if (gl_v364 < gl_v290) {
        gl_v367 = __ldg(&gl_v289[gl_v364 * 32 + (int)gl_v4]);
    }

    if (gl_v362 < 64) gl_v322[gl_v362] = 0;
    for(int gl_v176=gl_v362;gl_v176<577;gl_v176+=gl_v361)gl_v274[gl_v176]=0;
    if (gl_v362 == 0) {
        gl_v303[0] = gl_m2; gl_v306[0]=0;
    }

    if(gl_v283==0){
        for(int gl_v176=gl_v362;gl_v176<gl_v334;gl_v176+=gl_v361)gl_v330[gl_v176]=0;
        for(int gl_v368=gl_v362;gl_v368<gl_v329;gl_v368+=gl_v361)gl_v93[gl_v368]=make_ulonglong2(0,0);
        ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);
        for(int gl_v368=gl_v364;gl_v368<gl_v329;gl_v368+=gl_v365){
            int gl_v369=__ldg(&gl_v328[gl_v368]),gl_v190=__ldg(&gl_v328[gl_v368+1]);
            for(int gl_v176=gl_v369+(int)gl_v4;gl_v176<gl_v190;gl_v176+=32){
                int gl_v47=__ldg(&gl_v287[gl_v176]);
                int gl_v165=(unsigned)gl_v47<(unsigned)gl_v285?__ldcg(&gl_v48[gl_v47]):-1;
                if((unsigned)gl_v165<64u){
                    int gl_v259=gl_a15(gl_v330,gl_v333,gl_v368,gl_v165,false);
                    if(gl_v259==0)atomicXor(&gl_v93[gl_v368].x,1ULL<<gl_v165);
                    if(gl_v259==1)atomicXor(&gl_v93[gl_v368].y,1ULL<<gl_v165);
                }
            }
        }
    }else{
        const int gl_v272=__ldcg(&gl_v325[13]);
        const int gl_v164=(blockIdx.x*blockDim.x+threadIdx.x)/8;
        const int gl_v370=gridDim.x*blockDim.x/8,gl_v371=(int)threadIdx.x&7;
        for(int gl_v160=gl_v164;gl_v160<gl_v272;gl_v160+=gl_v370){
            unsigned long long gl_v228=__ldcg(&gl_v323[gl_v160]);
            int gl_v82=(int)(unsigned)gl_v228,gl_v372;
            if(gl_v350){
                gl_v372=__ldcg(gl_v324+gl_v160)?__ldcg(gl_v332+gl_v160):(int)((((unsigned)(gl_v228>>32))&63)^63);
                if(gl_v371==0){gl_v48[gl_v82]=gl_v372;if(gl_v351)gl_v293[gl_v82]=gl_v352;}
            }else gl_v372=__ldcg(&gl_v48[gl_v82]);
            int gl_v369=__ldg(&gl_v327[gl_v82]),gl_v190=__ldg(&gl_v327[gl_v82+1]);
            if((unsigned)gl_v372<64u)for(int gl_v176=gl_v369+gl_v371;gl_v176<gl_v190;gl_v176+=8){
                int gl_v368=__ldg(&gl_v65[gl_v176]);
                int gl_v259=gl_a15(gl_v330,gl_v333,gl_v368,gl_v372,false);
                if(gl_v259==0)atomicXor(&gl_v93[gl_v368].x,1ULL<<gl_v372);
                if(gl_v259==1)atomicXor(&gl_v93[gl_v368].y,1ULL<<gl_v372);
            }
        }
    }
    gl_v349=false;
    gl_v350=false;gl_v351=false;
    for(int gl_v160=gl_v362;gl_v160<gl_v354;gl_v160+=gl_v361){
        int gl_v82=gl_v294[gl_v160];if((unsigned)gl_v82<(unsigned)gl_v285)gl_v293[gl_v82]=gl_v296;
    }
    gl_v356=gl_v292+gl_v341*gridDim.x;

    gl_m28(gl_v291, gl_v356 + gridDim.x);
    if(gl_v329>=130){
        const unsigned long long *gl_v366=(const unsigned long long*)gl_v331;
        if(threadIdx.x<64)gl_v100[threadIdx.x]=__ldcg(gl_v366+1+threadIdx.x);
        if(threadIdx.x==0)gl_v99=__ldcg(gl_v366);
        __syncthreads();
    }


    for (int gl_v160 = gl_v362; gl_v160 < gl_v355; gl_v160 += gl_v361) {
        const int gl_v47 = gl_v294[gl_v354 + gl_v160];
        if (gl_v47 >= 0 && gl_v47 < gl_v285) gl_v293[gl_v47] = gl_v298;
    }

    int gl_v109 = gl_m2;

    for (int gl_v27 = gl_v364; gl_v27 < gl_v290; gl_v27 += gl_v365) {
        const int gl_v373 = gl_v27 + gl_v365;
        int4 gl_v374; gl_v374.x = 0; gl_v374.y = 0; gl_v374.z = 0; gl_v374.w = 0;
        if (gl_v373 < gl_v290) {
            gl_v374 = __ldg(&gl_v289[gl_v373 * 32 + (int)gl_v4]);
        }
        const int4 gl_v83 = gl_v367;
        switch (gl_v83.w) {
            case 0:  gl_m27(1,  0, gl_m4, gl_m11, gl_m18)  break;
            case 1:  gl_m27(2,  1, gl_m4, gl_m12, gl_m19)  break;
            case 2:  gl_m27(4,  2, gl_m4, gl_m13, gl_m20)  break;
            case 3:  gl_m27(8,  3, gl_m4, gl_m14, gl_m21)  break;
            case 4:  gl_m27(16, 4, gl_m4, gl_m15, gl_m22) break;
            default: gl_m27(32, 5, gl_m5, gl_m16, gl_m23) break;
        }
        gl_v367 = gl_v374;
    }

    {
        const int gl_v9 = gl_a2(gl_v109, gl_v2);
        if (threadIdx.x == 0) atomicMax(gl_v303, gl_v9);
    }

    gl_m28(gl_v291, gl_v356 + 2u * gridDim.x);

    const int gl_v378=__ldcg(gl_v303),gl_v379=(gl_v378*3)/4;
    const int gl_v380=(gl_v305+(int)gridDim.x-1)/(int)gridDim.x;
    const int gl_v381=(int)blockIdx.x*gl_v380;
    const int gl_v382=min(gl_v305,gl_v381+gl_v380);
    int gl_v383=0;
    for(int gl_v384=gl_v381;gl_v384<gl_v382;++gl_v384){
        int gl_v82=gl_v384*(int)blockDim.x+(int)threadIdx.x;
        if(gl_v82<gl_v285){
            int gl_v228=__ldcg(&gl_v107[gl_v82]),gl_v385=__ldcg(&gl_v293[gl_v82]);
            if(gl_v228!=gl_m1){int gl_v386=gl_v228>>16;bool gl_v387=(gl_v385 > gl_v353) && (gl_v386 < gl_v379);
                if(!gl_v387 && gl_v386 <= 0 && gl_v386 >= -3)++gl_v383;
            }
        }
    }
    int gl_v388=gl_a1(gl_v383,gl_v2);
    if(threadIdx.x==0)gl_v304[blockIdx.x]=gl_v388;
    gl_m28(gl_v291,gl_v356+3u*gridDim.x);
    int gl_v178=0;
    for(int gl_v389=(int)threadIdx.x;gl_v389<(int)blockIdx.x;gl_v389+=(int)blockDim.x)
        gl_v178+=__ldcg(&gl_v304[gl_v389]);
    const int gl_v390=gl_a1(gl_v178,gl_v2);
    int gl_v391=0,gl_v392=0;
    for(int gl_v384=gl_v381;gl_v384<gl_v382;++gl_v384){
        const int gl_v82=gl_v384*(int)blockDim.x+(int)threadIdx.x;
        int gl_v228=gl_v82<gl_v285?__ldcg(&gl_v107[gl_v82]):gl_m1;
        int gl_v385=gl_v82<gl_v285?__ldcg(&gl_v293[gl_v82]):0;
        int gl_v386=gl_v228>>16,gl_v393=0,gl_v394=0;
        if(gl_v82<gl_v285 && gl_v228!=gl_m1){
            bool gl_v387=(gl_v385 > gl_v353) && (gl_v386 < gl_v379);
            gl_v394=!gl_v387 && gl_v386>0;
            gl_v393=(!gl_v387 && gl_v386 <= 0 && gl_v386 >= -3);
        }
        int gl_v395=0;
        int gl_v126=gl_a3(gl_v393,gl_v2,&gl_v395);
        int gl_v84=gl_v394?gl_v228:-1;
        if(gl_v393){
            unsigned long long gl_v191=(unsigned long long)(gl_v390+gl_v391+gl_v126)+1ULL;
            unsigned long long x=gl_a10(gl_v357,gl_v191,gl_v326);
            unsigned long long gl_v396=gl_v302*(1ULL+5ULL*(unsigned long long)(-gl_v386));
            if(gl_v396>0ULL && x%gl_v396<gl_v359)gl_v84=gl_v228&0xffff;
        }
        if(gl_v82<gl_v285)gl_v306[1+gl_v82]=gl_v84;
        if(gl_v84>=0){
            ++gl_v392;
            int gl_v162=gl_v84&63,gl_v397=gl_a14(gl_v322,gl_v162);
            if(gl_v397<4096)gl_v321[gl_v162*4096+gl_v397]=
                ((unsigned long long)((unsigned int)gl_v84^0x7fffffffu)<<32)|(unsigned int)gl_v82;
        }
        gl_v391+=gl_v395;
    }
    int gl_v398=gl_a1(gl_v392,gl_v2);
    if(threadIdx.x==0)atomicAdd(gl_v306,gl_v398);
    gl_v341+=3;
    ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);

        int gl_v399=threadIdx.x<64?__ldcg(&gl_v322[threadIdx.x]):0;
        int gl_v400=0;
        if(threadIdx.x<64){
            int gl_v15=threadIdx.x;
            int gl_v401=gl_v353<64?gl_v311:(gl_v353<256?gl_v312:gl_v313);
            int gl_v402=max(1,max(0,gl_v286-gl_v88[gl_v15])+gl_v401);
            gl_v400=gl_v15<gl_v42?min(gl_v399,gl_v402):0;
            gl_v338[gl_v15]=gl_v400;
            int gl_v403=gl_v399,gl_v404=gl_v400;
            for(int gl_v3=1;gl_v3<32;gl_v3<<=1){
                int gl_v215=__shfl_up_sync(0xffffffffu,gl_v403,gl_v3),gl_v216=__shfl_up_sync(0xffffffffu,gl_v404,gl_v3);
                if((gl_v15&31)>=gl_v3){gl_v403+=gl_v215;gl_v404+=gl_v216;}
            }
            gl_v336[gl_v15+1]=gl_v403;gl_v339[gl_v15+1]=gl_v404;
        }
        if(threadIdx.x==0){gl_v336[0]=0;gl_v339[0]=0;}
        __syncthreads();
        if(threadIdx.x>=32 && threadIdx.x<64){
            gl_v336[threadIdx.x+1]+=gl_v336[32];
            gl_v339[threadIdx.x+1]+=gl_v339[32];
        }
        int gl_v405=__syncthreads_or(threadIdx.x<64 && gl_v399>4096);
        int gl_v406=__syncthreads_or(threadIdx.x<64 && gl_v400>4096);
        if(threadIdx.x==0){
            gl_v340[0]=__ldcg(&gl_v306[0]);gl_v340[1]=gl_v339[64];
            gl_v340[2]=gl_v406;gl_v340[3]=gl_v405;
        }
        if(!gl_v405 && threadIdx.x<65)gl_v336[threadIdx.x]=threadIdx.x*4096;
        __syncthreads();
        int gl_v407 = gl_v340[0], gl_v145 = gl_v340[1];
        if(gl_v362==0)gl_v325[13]=gl_v145;
        if (gl_v407 == 0) { gl_v343 = 1; break; }
        int gl_v408 = gl_v353 < 50 ? gl_v310 / 2 :
                            (gl_v353 < 200 ? (gl_v310 * 3) / 4 : gl_v310 / 4);
        bool gl_v409 = gl_v407 <= gl_v408 ||
            (gl_v407 <= gl_v408 + 61440 && gl_v145 <= gl_v408);
        if (!gl_v409 || gl_v340[2] || gl_v145 <= 0 || gl_v285<256) { gl_v343 = 4; break; }


        if(gl_v340[3]){
            if(gl_v362<64)gl_v322[64+gl_v362]=0;
            ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);
            for(int gl_v160=gl_v362;gl_v160<gl_v285;gl_v160+=gl_v361){
                int gl_v228=__ldcg(&gl_v306[1+gl_v160]);
                if(gl_v228<0)continue;
                int gl_v82=gl_v160,gl_v162=gl_v228&63;
                int gl_v241=gl_a11(gl_v322 + 64,gl_v162);
                gl_v321[gl_v336[gl_v162]+gl_v241]=
                    ((unsigned long long)((unsigned int)gl_v228^0x7fffffffu)<<32)|(unsigned int)gl_v82;
            }
            ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);
        }
        for(int gl_v15=blockIdx.x;gl_v15<gl_v42 && gl_v15<64;gl_v15+=gridDim.x){
            int gl_v47=__ldcg(&gl_v322[gl_v15]),gl_v221=gl_v338[gl_v15],gl_v3=gl_v339[gl_v15];
            int gl_v189=gl_v336[gl_v15];
            if(gl_v221==0)continue;
            gl_a9(&gl_v321[gl_v189],gl_v47,gl_v221,&gl_v335,gl_v337);
            for(int gl_v176=threadIdx.x;gl_v176<gl_v221;gl_v176+=blockDim.x){
                unsigned long long gl_v228=__ldcg(&gl_v321[gl_v189+gl_v176]);
                int gl_v82=(int)(unsigned)gl_v228;
                int gl_v174=__ldcg(&gl_v48[gl_v82]);
                gl_v323[gl_v3+gl_v176]=gl_v228;gl_v332[gl_v3+gl_v176]=gl_v174;
                if((unsigned)gl_v174<(unsigned)gl_v42)atomicAdd(gl_v274+gl_v174,1);
                else atomicExch(gl_v274+576,1);
                int gl_v188=max(0,gl_v286-gl_v88[gl_v15]),gl_v112=gl_v176-gl_v188;
                if(gl_v112>=0 && gl_v112<8)gl_v275[gl_v15*8+gl_v112]=make_ulonglong2(gl_v228,(unsigned)((gl_v82<<12)|(gl_v174<<6)|gl_v15));

            }
        }
        ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);

        for(int gl_v160=gl_v362;gl_v160<gl_v145;gl_v160+=gl_v361){
            gl_v324[gl_v160]=0;
            unsigned long long gl_v228=__ldcg(gl_v323+gl_v160);
            int gl_v174=__ldcg(gl_v332+gl_v160);
            if((unsigned)gl_v174>=(unsigned)gl_v42)continue;
            int gl_v112=min(8,max(0,gl_v338[gl_v174]-max(0,gl_v286-gl_v88[gl_v174])));
            for(int gl_v176=0;gl_v176<gl_v112;++gl_v176){
                unsigned long long gl_v267=gl_a4(gl_v275+gl_v174*8+gl_v176).x;
                if(gl_v228<gl_v267)atomicAdd(gl_v274+64+gl_v174*8+gl_v176,1);
            }
        }
        ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);
        if(gridDim.x==1){
        const int gl_v410=(gridDim.x>1?((int)gridDim.x-1):(int)gridDim.x)*(int)blockDim.x/8;
        const int gl_v411=((gridDim.x>1?((int)blockIdx.x-1):(int)blockIdx.x)*(int)blockDim.x+(int)threadIdx.x)/8;
        const int gl_v371=(int)threadIdx.x&7;
        for(int gl_v160=gl_v411;gl_v160<gl_v145;gl_v160+=gl_v410){
            unsigned long long gl_v228=__ldcg(&gl_v323[gl_v160]);
            int gl_v82=(int)(unsigned)gl_v228,gl_v412=__ldcg(&gl_v332[gl_v160]);
            int gl_v369=__ldg(&gl_v327[gl_v82]),gl_v190=__ldg(&gl_v327[gl_v82+1]);
            if((unsigned)gl_v412<64u)for(int gl_v176=gl_v369+gl_v371;gl_v176<gl_v190;gl_v176+=8){
                int gl_v368=__ldg(&gl_v65[gl_v176]);
                int gl_v259=gl_a15(gl_v330,gl_v333,gl_v368,gl_v412,true);
                if(gl_v259==1)atomicXor(&gl_v93[gl_v368].x,1ULL<<gl_v412);
                if(gl_v259==2)atomicXor(&gl_v93[gl_v368].y,1ULL<<gl_v412);
            }
        }
        }
        if(blockIdx.x==0) {
            int gl_v134=gl_a17(gl_v145,gl_v285,gl_v42,gl_v286,
                gl_v48,gl_v288,gl_v323,gl_v338,gl_v339,
                gl_v293,gl_v353,gl_v314,gl_v315,gl_v316,&gl_v335,gl_v274,gl_v275,gl_v324);
            if(gl_v134<0){
            gl_v134=gl_a18(gl_v145,gl_v285,gl_v42,gl_v286,
                gl_v48,gl_v288,gl_v323,gl_v338,gl_v339,
                gl_v293,gl_v353,gl_v314,gl_v315,gl_v316,&gl_v335,gl_v274,gl_v275,gl_v324);
            }
            if(threadIdx.x==0){gl_v325[15]=gl_v134;gl_v325[14]=(gl_v134>=0);}
        } else {
        const int gl_v410=(gridDim.x>1?((int)gridDim.x-1):(int)gridDim.x)*(int)blockDim.x/8;
        const int gl_v411=((gridDim.x>1?((int)blockIdx.x-1):(int)blockIdx.x)*(int)blockDim.x+(int)threadIdx.x)/8;
        const int gl_v371=(int)threadIdx.x&7;
        for(int gl_v160=gl_v411;gl_v160<gl_v145;gl_v160+=gl_v410){
            unsigned long long gl_v228=__ldcg(&gl_v323[gl_v160]);
            int gl_v82=(int)(unsigned)gl_v228,gl_v412=__ldcg(&gl_v332[gl_v160]);
            int gl_v369=__ldg(&gl_v327[gl_v82]),gl_v190=__ldg(&gl_v327[gl_v82+1]);
            if((unsigned)gl_v412<64u)for(int gl_v176=gl_v369+gl_v371;gl_v176<gl_v190;gl_v176+=8){
                int gl_v368=__ldg(&gl_v65[gl_v176]);
                int gl_v259=gl_a15(gl_v330,gl_v333,gl_v368,gl_v412,true);
                if(gl_v259==1)atomicXor(&gl_v93[gl_v368].x,1ULL<<gl_v412);
                if(gl_v259==2)atomicXor(&gl_v93[gl_v368].y,1ULL<<gl_v412);
            }
        }
        }
        gl_v349=true;
        ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);
        if(__ldcg(&gl_v325[15])<0) {
        const bool gl_v413 = gl_v145 <= 4096;
        if(gl_v413){
            for(int gl_v176=threadIdx.x;gl_v176<gl_v145;gl_v176+=blockDim.x)gl_v335.gl_v129[gl_v176]=__ldcg(&gl_v323[gl_v176]);
            __syncthreads();
        }
        for (int gl_v160 = gl_v364; gl_v160 < gl_v145; gl_v160 += gl_v365) {
            unsigned long long gl_v228 = gl_v413 ? gl_v335.gl_v129[gl_v160] : __ldcg(&gl_v323[gl_v160]);
            int gl_v126 = 0;
            for (int gl_v15 = gl_v4; gl_v15 < gl_v42 && gl_v15 < 64; gl_v15 += 32) {
                int gl_v75 = 0, gl_v76 = gl_v338[gl_v15];
                while (gl_v75 < gl_v76) {
                    int gl_v414 = (gl_v75 + gl_v76) >> 1;
                    unsigned long long gl_v195 = gl_v413 ? gl_v335.gl_v129[gl_v339[gl_v15]+gl_v414] : __ldcg(&gl_v321[gl_v336[gl_v15] + gl_v414]);
                    if (gl_v195 < gl_v228) gl_v75 = gl_v414 + 1; else gl_v76 = gl_v414;
                }
                gl_v126 += gl_v75;
            }
            for (int gl_v415 = 16; gl_v415 > 0; gl_v415 >>= 1) gl_v126 += __shfl_xor_sync(0xffffffffu, gl_v126, gl_v415);
            if (gl_v4 == 0) {
                int gl_v82 = (int)(unsigned int)gl_v228;
                int gl_v162 = (((unsigned int)(gl_v228 >> 32)) & 63) ^ 63;
                gl_v324[gl_v126] = (gl_v82 << 6) | gl_v162;
                int gl_v188=max(0,gl_v286-gl_v88[gl_v162]);
                int gl_v219=gl_v160-gl_v339[gl_v162]-gl_v188;
                if(gl_v219>=0 && gl_v219<8) gl_v306[1+gl_v162*8+gl_v219]=gl_v126;
            }
        }
        ++gl_v341; gl_m28(gl_v291, gl_v292 + gl_v341 * gridDim.x);
        if (blockIdx.x == 0) {
            int gl_v124 = gl_a8(gl_v145, gl_v285, gl_v42, gl_v286,
                gl_v48, gl_v288, gl_v324, gl_v293, gl_v353,
                gl_v314, gl_v315, gl_v316, &gl_v335, gl_v2, gl_v306+1, gl_v338);
            if (threadIdx.x == 0) gl_v325[15] = gl_v124;
        }
        ++gl_v341; gl_m28(gl_v291, gl_v292 + gl_v341 * gridDim.x);
        }
        int gl_v124 = __ldcg(&gl_v325[15]);
        gl_v350=__ldcg(gl_v325+14)!=0;
        gl_v351=gl_v124>0 && (gl_v315>0 || gl_v316>0);
        gl_v352=gl_v353+gl_v314;

        if(gl_v124==0 && gl_v407>gl_v408){gl_v343=4;break;}
        gl_v344 += gl_v124;
        ++gl_v342;
        if (gl_v124 == 0) {
            ++gl_v345;
            ++gl_v346;
            ++gl_v347;
            gl_v348 = max(gl_v348, gl_v346);
            if (gl_v345 >= 3 && gl_v353 < max(0, gl_v317 - 50)) { gl_v343 = 2; break; }
            if (gl_v345 > 30) { gl_v343 = 3; break; }
        } else { gl_v345 = 0; gl_v346 = 0; }
        if (gl_v318 > 0 && gl_v346 >= gl_v318) { gl_v343 = 3; break; }

    }
    if(gl_v349){
        const int gl_v272=__ldcg(&gl_v325[13]);
        const int gl_v164=(blockIdx.x*blockDim.x+threadIdx.x)/8;
        const int gl_v370=gridDim.x*blockDim.x/8,gl_v371=(int)threadIdx.x&7;
        for(int gl_v160=gl_v164;gl_v160<gl_v272;gl_v160+=gl_v370){
            unsigned long long gl_v228=__ldcg(&gl_v323[gl_v160]);
            int gl_v82=(int)(unsigned)gl_v228,gl_v372;
            if(gl_v350){
                gl_v372=__ldcg(gl_v324+gl_v160)?__ldcg(gl_v332+gl_v160):(int)((((unsigned)(gl_v228>>32))&63)^63);
                if(gl_v371==0){gl_v48[gl_v82]=gl_v372;if(gl_v351)gl_v293[gl_v82]=gl_v352;}
            }else gl_v372=__ldcg(&gl_v48[gl_v82]);
            int gl_v369=__ldg(&gl_v327[gl_v82]),gl_v190=__ldg(&gl_v327[gl_v82+1]);
            if((unsigned)gl_v372<64u)for(int gl_v176=gl_v369+gl_v371;gl_v176<gl_v190;gl_v176+=8){
                int gl_v368=__ldg(&gl_v65[gl_v176]);
                int gl_v259=gl_a15(gl_v330,gl_v333,gl_v368,gl_v372,false);
                if(gl_v259==0)atomicXor(&gl_v93[gl_v368].x,1ULL<<gl_v372);
                if(gl_v259==1)atomicXor(&gl_v93[gl_v368].y,1ULL<<gl_v372);
            }
        }
        ++gl_v341; gl_m28(gl_v291,gl_v292+gl_v341*gridDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        gl_v325[0] = gl_v343; gl_v325[1] = gl_v342; gl_v325[2] = (int)gl_v341;
        gl_v325[3] = gl_v344; gl_v325[4] = gl_v345; gl_v325[5] = gl_v346;
        gl_v325[7] = gl_v347; gl_v325[8] = gl_v348;
    }
}

#undef gl_m1
#undef gl_m2
#undef gl_m3
#undef gl_m4
#undef gl_m5
#undef gl_m6
#undef gl_m7
#undef gl_m8
#undef gl_m9
#undef gl_m10
#undef gl_m11
#undef gl_m12
#undef gl_m13
#undef gl_m14
#undef gl_m15
#undef gl_m16
#undef gl_m17
#undef gl_m18
#undef gl_m19
#undef gl_m20
#undef gl_m21
#undef gl_m22
#undef gl_m23
#undef gl_m24
#undef gl_m25
#undef gl_m26
#undef gl_m27
#undef gl_m28

#define gm_m1 (-1048576)

#define gm_m2(gm_v1,gm_v2) gl_a13((gm_v1),(gm_v2))

#define gm_m3(gm_v3, gm_v4, gm_v5, gm_v6, gm_v7)                                     \
{                                                                              \
    const int gm_v8 = (int)(gm_v9 & (unsigned)((gm_v3) - 1));                         \
    const int gm_v10 = gm_v11.x;                                                    \
    const int gm_v12 = gm_v11.y;                                                  \
    const int gm_v13 = gm_v11.z;                                             \
    int gm_v14 = gm_m18;                                                      \
    const int gm_v15 = gm_v13 > 256 ? 256 : gm_v13;             \
    const bool gm_v16 = (gm_v13 > 256);                                  \
    const bool gm_v17 = (gm_v13 > 0) && (gm_v8 < gm_v15);                 \
    int gm_v18 = gm_v8;                                                              \
    int gm_v19 = 0, gm_v20 = 0, gm_v21 = 0, gm_v22 = 0;                                    \
    bool gm_v23 = false, gm_v24 = false, gm_v25 = false;                                \
    if (gm_v17) { gm_m19(gm_v3); }                                                 \
    const int gm_v26 = gm_m20(gm_v10);                          \
    const bool gm_v27 = (unsigned)gm_v26 < (unsigned)gm_v28 &&     \
        gm_v29[gm_v26] > 1 &&                              \
        gm_v13 > 0;                                                       \
    const unsigned long long gm_v30 = gm_v27 ? (1ULL << gm_v26) : 0ULL; \
    gm_m21                                                                \
    unsigned long long gm_v31 = 0ULL;                                       \
    int gm_v32 = 0;                                             \
    int gm_v33 = 0;                                                              \
    if (gm_v27 && gm_v17) {                                                    \
            for (;;) {                                                         \
                ulonglong2 gm_v34; gm_v34.x = 0ULL; gm_v34.y = 0ULL;                    \
                ulonglong2 gm_v35; gm_v35.x = 0ULL; gm_v35.y = 0ULL;                    \
                ulonglong2 gm_v36; gm_v36.x = 0ULL; gm_v36.y = 0ULL;                    \
                const ulonglong2 gm_v37 = gl_a4(&gm_v38[gm_v19]);      \
                if (gm_v23) gm_v34 = gl_a4(&gm_v38[gm_v20]);              \
                if (gm_v24) gm_v35 = gl_a4(&gm_v38[gm_v21]);              \
                if (gm_v25) gm_v36 = gl_a4(&gm_v38[gm_v22]);              \
                gm_m22(gm_v37, gm_v5);                                          \
                if (gm_v23) { gm_m22(gm_v34, gm_v5); }                             \
                if (gm_v24) { gm_m22(gm_v35, gm_v5); }                             \
                if (gm_v25) { gm_m22(gm_v36, gm_v5); }                             \
                gm_v18 += 4 * (gm_v3);                                                 \
                if (gm_v18 >= gm_v15) break;                                  \
                gm_m19(gm_v3);                                                  \
            }                                                                  \
        }                                                                      \
        gm_v7                                                                    \
        if (gm_v33 > 255) gm_v33 = 255;                                            \
        int gm_v39 = gm_v13 > 255 ? 255 : gm_v13;             \
        int gm_v40 = gm_v39 + gm_v33;                                  \
        if (gm_v40 > 255) gm_v40 = 255;                                  \
        int gm_v41 = -999999;                                               \
        int gm_v42 = gm_v26;                                        \
        unsigned long long gm_v43 = gm_v31 & gm_v44; \
        if (gm_v43) { \
            unsigned long long gm_v45 = gm_v46[gm_v26], gm_v47; \
            gm_v47 = gm_v48 & gm_v45; gm_v48 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v49 & gm_v45; gm_v49 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v50 & gm_v45; gm_v50 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v51 & gm_v45; gm_v51 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v52 & gm_v45; gm_v52 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v53 & gm_v45; gm_v53 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v54 & gm_v45; gm_v54 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v55 & gm_v45; gm_v55 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v47 = gm_v56 & gm_v45; gm_v56 ^= gm_v45; gm_v45 = gm_v47; \
            gm_v41 = 0; \
            { unsigned long long gm_v57 = gm_v43 & gm_v56; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 256; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v55; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 128; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v54; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 64; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v53; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 32; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v52; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 16; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v51; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 8; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v50; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 4; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v49; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 2; } } \
            { unsigned long long gm_v57 = gm_v43 & gm_v48; \
              if (gm_v57) { gm_v43 = gm_v57; gm_v41 |= 1; } } \
            gm_v42 = __ffsll(gm_v43) - 1; \
            gm_v41 -= gm_v32; \
        } \
        if (gm_v42 != gm_v26) {                                     \
            int gm_v58 = gm_v41 > 32767 ? 32767 : gm_v41;                    \
            if (gm_v58 < -32768) gm_v58 = -32768;                                      \
            unsigned short gm_v59 =                                               \
                (unsigned short)(gm_v42 & 63) | ((gm_v10 & 3) << 6);        \
            gm_v14 = ((int)(short)gm_v58 << 16) | (gm_v40 << 8) | gm_v59;          \
        }                                                                      \
    if (gm_v8 == 0) {                                                            \
        gm_v60[gm_v10] = gm_v14;                                        \
        if (gm_v14 != gm_m18) {                                               \
            const int gm_v61 = gm_v14 >> 16;                                       \
            if (gm_v61 > gm_v62) gm_v62 = gm_v61;                                    \
        }                                                                      \
    }                                                                          \
 \
    const int gm_v63=(gm_v3)==32?-1:__shfl_up_sync(0xffffffffu,gm_v10,(gm_v3)); \
    if(gm_v64 && gm_v8==0 && ((int)gm_v9<(gm_v3) || gm_v63!=gm_v10) \
       && gm_v14!=gm_m18){ \
        int gm_v65=gm_v14>>16; \
        int gm_v66=-1; \
        if(gm_v67){ \
            if(gm_v68>0ULL && gm_v65==0)gm_v66=0; \
        }else if(gm_v65>=-3 && gm_v65<=0){ \
            int gm_v69=__ldcg(&gm_v70[gm_v10]); \
            if(gm_v69<=gm_v71)gm_v66=0; \
            else if(gm_v65==0)gm_v66=1; \
        } \
        if(gm_v66>=0){ \
            int gm_v72=(unsigned)gm_v10>>(unsigned)gm_v73; \
            int gm_v74=gm_v75==1?gm_v72:gm_v72/gm_v75; \
            int gm_v76=gm_v74*2+gm_v66; \
            unsigned gm_v43=__activemask(); \
            unsigned gm_v77=__match_any_sync(gm_v43,gm_v76); \
            if((int)gm_v9==__ffs(gm_v77)-1) \
                atomicAdd(((unsigned long long*)gm_v78)+gm_v74, \
                    (unsigned long long)__popc(gm_v77)<<(32*gm_v66)); \
        } \
    } \
}

#define gm_m4(gm_v79) do { \
    unsigned long long gm_v80 = (gm_v79), gm_v2; \
    gm_v2 = gm_v48 & gm_v80; gm_v48 ^= gm_v80; gm_v80 = gm_v2; \
    gm_v2 = gm_v49 & gm_v80; gm_v49 ^= gm_v80; gm_v80 = gm_v2; \
    gm_v50 ^= gm_v80; \
} while (0)

#define gm_m5(gm_v81) ( \
    ((int)((gm_v48 >> (gm_v81)) & 1ULL) << 0) | \
    ((int)((gm_v49 >> (gm_v81)) & 1ULL) << 1) | \
    ((int)((gm_v50 >> (gm_v81)) & 1ULL) << 2) )

#define gm_m6

#define gm_m7(gm_v81) ( \
    ((int)((gm_v48 >> (gm_v81)) & 1ULL) << 0) | \
    ((int)((gm_v49 >> (gm_v81)) & 1ULL) << 1) | \
    ((int)((gm_v50 >> (gm_v81)) & 1ULL) << 2) | \
    ((int)((gm_v51 >> (gm_v81)) & 1ULL) << 3) )

#define gm_m8   gm_m23(gm_m24, 1);

#define gm_m9(gm_v81) ( \
    ((int)((gm_v48 >> (gm_v81)) & 1ULL) << 0) | \
    ((int)((gm_v49 >> (gm_v81)) & 1ULL) << 1) | \
    ((int)((gm_v50 >> (gm_v81)) & 1ULL) << 2) | \
    ((int)((gm_v51 >> (gm_v81)) & 1ULL) << 3) | \
    ((int)((gm_v52 >> (gm_v81)) & 1ULL) << 4) )

#define gm_m10   gm_m23(gm_m25, 2); gm_m23(gm_m25, 1);

#define gm_m11(gm_v81) ( \
    ((int)((gm_v48 >> (gm_v81)) & 1ULL) << 0) | \
    ((int)((gm_v49 >> (gm_v81)) & 1ULL) << 1) | \
    ((int)((gm_v50 >> (gm_v81)) & 1ULL) << 2) | \
    ((int)((gm_v51 >> (gm_v81)) & 1ULL) << 3) | \
    ((int)((gm_v52 >> (gm_v81)) & 1ULL) << 4) | \
    ((int)((gm_v53 >> (gm_v81)) & 1ULL) << 5) )

#define gm_m12   gm_m23(gm_m26, 4); gm_m23(gm_m26, 2); \
                      gm_m23(gm_m26, 1);

#define gm_m13(gm_v81) ( \
    ((int)((gm_v48 >> (gm_v81)) & 1ULL) << 0) | \
    ((int)((gm_v49 >> (gm_v81)) & 1ULL) << 1) | \
    ((int)((gm_v50 >> (gm_v81)) & 1ULL) << 2) | \
    ((int)((gm_v51 >> (gm_v81)) & 1ULL) << 3) | \
    ((int)((gm_v52 >> (gm_v81)) & 1ULL) << 4) | \
    ((int)((gm_v53 >> (gm_v81)) & 1ULL) << 5) | \
    ((int)((gm_v54 >> (gm_v81)) & 1ULL) << 6) )

#define gm_m14  gm_m23(gm_m27, 8); gm_m23(gm_m27, 4); \
                      gm_m23(gm_m27, 2); gm_m23(gm_m27, 1);

#define gm_m15(gm_v79) do { \
    unsigned long long gm_v80 = (gm_v79), gm_v2; \
    gm_v2 = gm_v48 & gm_v80; gm_v48 ^= gm_v80; gm_v80 = gm_v2; \
    gm_v2 = gm_v49 & gm_v80; gm_v49 ^= gm_v80; gm_v80 = gm_v2; \
    gm_v2 = gm_v50 & gm_v80; gm_v50 ^= gm_v80; gm_v80 = gm_v2; \
    gm_v51 ^= gm_v80; \
} while (0)

#define gm_m16(gm_v81) ( \
    ((int)((gm_v48 >> (gm_v81)) & 1ULL) << 0) | \
    ((int)((gm_v49 >> (gm_v81)) & 1ULL) << 1) | \
    ((int)((gm_v50 >> (gm_v81)) & 1ULL) << 2) | \
    ((int)((gm_v51 >> (gm_v81)) & 1ULL) << 3) | \
    ((int)((gm_v52 >> (gm_v81)) & 1ULL) << 4) | \
    ((int)((gm_v53 >> (gm_v81)) & 1ULL) << 5) | \
    ((int)((gm_v54 >> (gm_v81)) & 1ULL) << 6) | \
    ((int)((gm_v55 >> (gm_v81)) & 1ULL) << 7) | \
    ((int)((gm_v56 >> (gm_v81)) & 1ULL) << 8) )

#define gm_m17  gm_m23(gm_m28, 16); gm_m23(gm_m28, 8); \
                      gm_m23(gm_m28, 4); gm_m23(gm_m28, 2); \
                      gm_m23(gm_m28, 1);

#define gm_m18    ((int)0x80000000)

#define gm_m19(gm_v3) do {                                                      \
    const int gm_v82 = gm_v18 + (gm_v3), gm_v83 = gm_v18 + 2 * (gm_v3), gm_v84 = gm_v18 + 3 * (gm_v3);          \
    gm_v23 = (gm_v82 < gm_v15);                                                 \
    gm_v24 = (gm_v83 < gm_v15);                                                 \
    gm_v25 = (gm_v84 < gm_v15);                                                 \
    const int gm_v85 = gm_v16 ? (int)(((long long)gm_v18  * gm_v13) >> 8) : gm_v18;  \
    const int gm_v86 = gm_v16 ? (int)(((long long)gm_v82 * gm_v13) >> 8) : gm_v82; \
    const int gm_v87 = gm_v16 ? (int)(((long long)gm_v83 * gm_v13) >> 8) : gm_v83; \
    const int gm_v88 = gm_v16 ? (int)(((long long)gm_v84 * gm_v13) >> 8) : gm_v84; \
    gm_v19 = __ldg(&gm_v89[gm_v12 + gm_v85]);                               \
    gm_v20 = gm_v23 ? __ldg(&gm_v89[gm_v12 + gm_v86]) : 0;                     \
    gm_v21 = gm_v24 ? __ldg(&gm_v89[gm_v12 + gm_v87]) : 0;                     \
    gm_v22 = gm_v25 ? __ldg(&gm_v89[gm_v12 + gm_v88]) : 0;                     \
} while (0)

#define gm_m20(gm_v90) __ldcg(&gm_v91[(gm_v90)])

#define gm_m21 \
    unsigned long long gm_v48 = 0ULL, gm_v49 = 0ULL, gm_v50 = 0ULL, gm_v51 = 0ULL, gm_v52 = 0ULL, \
                       gm_v53 = 0ULL, gm_v54 = 0ULL, gm_v55 = 0ULL, gm_v56 = 0ULL;

#define gm_m22(gm_v92, gm_v5) do {                                               \
    const unsigned long long gm_v93 = (gm_v92).x;                                     \
    const unsigned long long gm_v94 = (gm_v92).y;                                     \
     \
    const unsigned long long gm_v95 = gm_v28 >= 64 ? ~0ULL : ((1ULL << gm_v28) - 1ULL); \
    if (gm_v28 <= 64 && gm_v93 == gm_v95 && (gm_v94 & gm_v30) != 0ULL) { \
        gm_v31 |= gm_v93 & ~gm_v30; \
    } else { \
    if ((gm_v93 & gm_v30) != 0ULL &&                                         \
        (gm_v94 & gm_v30) == 0ULL) {                                         \
        const int gm_v96 = __popcll(gm_v93);                                      \
        if (gm_v96 > 1) gm_v33 += (gm_v96 - 1);                                  \
    }                                                                          \
    const unsigned long long gm_v97 =                                           \
        (gm_v93 & ~gm_v30) | (gm_v94 & gm_v30);                            \
    if (gm_v97 & gm_v30) gm_v32++;                          \
    const unsigned long long gm_v98 = gm_v97 & ~gm_v30;                       \
    gm_v31 |= gm_v98;                                                          \
    gm_v5(gm_v98);                                                                 \
    } \
} while (0)

#define gm_m23(gm_v99, gm_v100) do {                                             \
    gm_v31 |= __shfl_xor_sync(0xffffffffu, gm_v31, (gm_v100));               \
    gm_v32 +=                                                   \
        __shfl_xor_sync(0xffffffffu, gm_v32, (gm_v100));            \
    gm_v33 += __shfl_xor_sync(0xffffffffu, gm_v33, (gm_v100));                         \
    gm_v99(gm_v100);                                                                 \
} while (0)

#define gm_m24(gm_v100) do { \
    unsigned long long gm_v74, gm_v45, gm_v101, gm_v102; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v48, (gm_v100)); \
    gm_v101 = gm_v48 ^ gm_v74; gm_v45 = gm_v48 & gm_v74; gm_v48 = gm_v101; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v49, (gm_v100)); \
    gm_v101 = gm_v49 ^ gm_v74; gm_v102 = gm_v49 & gm_v74; gm_v49 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v50, (gm_v100)); \
    gm_v101 = gm_v50 ^ gm_v74; gm_v102 = gm_v50 & gm_v74; gm_v50 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v51, (gm_v100)); \
    gm_v101 = gm_v51 ^ gm_v74; gm_v51 = gm_v101 ^ gm_v45; \
} while (0)

#define gm_m25(gm_v100) do { \
    unsigned long long gm_v74, gm_v45, gm_v101, gm_v102; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v48, (gm_v100)); \
    gm_v101 = gm_v48 ^ gm_v74; gm_v45 = gm_v48 & gm_v74; gm_v48 = gm_v101; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v49, (gm_v100)); \
    gm_v101 = gm_v49 ^ gm_v74; gm_v102 = gm_v49 & gm_v74; gm_v49 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v50, (gm_v100)); \
    gm_v101 = gm_v50 ^ gm_v74; gm_v102 = gm_v50 & gm_v74; gm_v50 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v51, (gm_v100)); \
    gm_v101 = gm_v51 ^ gm_v74; gm_v102 = gm_v51 & gm_v74; gm_v51 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v52, (gm_v100)); \
    gm_v101 = gm_v52 ^ gm_v74; gm_v52 = gm_v101 ^ gm_v45; \
} while (0)

#define gm_m26(gm_v100) do { \
    unsigned long long gm_v74, gm_v45, gm_v101, gm_v102; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v48, (gm_v100)); \
    gm_v101 = gm_v48 ^ gm_v74; gm_v45 = gm_v48 & gm_v74; gm_v48 = gm_v101; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v49, (gm_v100)); \
    gm_v101 = gm_v49 ^ gm_v74; gm_v102 = gm_v49 & gm_v74; gm_v49 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v50, (gm_v100)); \
    gm_v101 = gm_v50 ^ gm_v74; gm_v102 = gm_v50 & gm_v74; gm_v50 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v51, (gm_v100)); \
    gm_v101 = gm_v51 ^ gm_v74; gm_v102 = gm_v51 & gm_v74; gm_v51 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v52, (gm_v100)); \
    gm_v101 = gm_v52 ^ gm_v74; gm_v102 = gm_v52 & gm_v74; gm_v52 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v53, (gm_v100)); \
    gm_v101 = gm_v53 ^ gm_v74; gm_v53 = gm_v101 ^ gm_v45; \
} while (0)

#define gm_m27(gm_v100) do { \
    unsigned long long gm_v74, gm_v45, gm_v101, gm_v102; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v48, (gm_v100)); \
    gm_v101 = gm_v48 ^ gm_v74; gm_v45 = gm_v48 & gm_v74; gm_v48 = gm_v101; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v49, (gm_v100)); \
    gm_v101 = gm_v49 ^ gm_v74; gm_v102 = gm_v49 & gm_v74; gm_v49 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v50, (gm_v100)); \
    gm_v101 = gm_v50 ^ gm_v74; gm_v102 = gm_v50 & gm_v74; gm_v50 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v51, (gm_v100)); \
    gm_v101 = gm_v51 ^ gm_v74; gm_v102 = gm_v51 & gm_v74; gm_v51 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v52, (gm_v100)); \
    gm_v101 = gm_v52 ^ gm_v74; gm_v102 = gm_v52 & gm_v74; gm_v52 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v53, (gm_v100)); \
    gm_v101 = gm_v53 ^ gm_v74; gm_v102 = gm_v53 & gm_v74; gm_v53 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v54, (gm_v100)); \
    gm_v101 = gm_v54 ^ gm_v74; gm_v54 = gm_v101 ^ gm_v45; \
} while (0)

#define gm_m28(gm_v100) do { \
    unsigned long long gm_v74, gm_v45, gm_v101, gm_v102; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v48, (gm_v100)); \
    gm_v101 = gm_v48 ^ gm_v74; gm_v45 = gm_v48 & gm_v74; gm_v48 = gm_v101; \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v49, (gm_v100)); \
    gm_v101 = gm_v49 ^ gm_v74; gm_v102 = gm_v49 & gm_v74; gm_v49 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v50, (gm_v100)); \
    gm_v101 = gm_v50 ^ gm_v74; gm_v102 = gm_v50 & gm_v74; gm_v50 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v51, (gm_v100)); \
    gm_v101 = gm_v51 ^ gm_v74; gm_v102 = gm_v51 & gm_v74; gm_v51 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v52, (gm_v100)); \
    gm_v101 = gm_v52 ^ gm_v74; gm_v102 = gm_v52 & gm_v74; gm_v52 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v53, (gm_v100)); \
    gm_v101 = gm_v53 ^ gm_v74; gm_v102 = gm_v53 & gm_v74; gm_v53 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v54, (gm_v100)); \
    gm_v101 = gm_v54 ^ gm_v74; gm_v102 = gm_v54 & gm_v74; gm_v54 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v55, (gm_v100)); \
    gm_v101 = gm_v55 ^ gm_v74; gm_v102 = gm_v55 & gm_v74; gm_v55 = gm_v101 ^ gm_v45; gm_v45 = gm_v102 | (gm_v101 & gm_v45); \
    gm_v74 = __shfl_xor_sync(0xffffffffu, gm_v56, (gm_v100)); \
    gm_v101 = gm_v56 ^ gm_v74; gm_v56 = gm_v101 ^ gm_v45; \
} while (0)

extern "C" __global__ __launch_bounds__(512, 1) void gm_k1(
    const int gm_v103,
    const int gm_v28,
    const int gm_v104,
    const int *__restrict__ gm_v105,
    const int *__restrict__ gm_v89,
    int *gm_v91,
    int *gm_v106,
    const int4 *__restrict__ gm_v107,
    const int gm_v108,
    ulonglong2 *gm_v38,
    int *gm_v60,
    unsigned int *gm_v109,
    const unsigned int gm_v110,
    int *gm_v70,
    const int gm_v111,
    const unsigned long long gm_v112,
    const unsigned long long gm_v68,
    int *gm_v113,
    int *gm_v114,
    const int gm_v115,
    int *gm_v116,
    const int gm_v117, const int gm_v118, const int gm_v119,
    const int gm_v120, const int gm_v121, const int gm_v122, const int gm_v123,
    const int gm_v124, const int gm_v125, const int gm_v126,
    const int gm_v127, const int gm_v128, const int gm_v129,
    unsigned long long *gm_v130, int *gm_v131,
    unsigned long long *gm_v132, int *gm_v133, int *gm_v134, const ulonglong2 *gm_v135, const int *gm_v136, const int *gm_v137,
    const int gm_v138, int *gm_v139, int *gm_v78,
    int *gm_v140, int *gm_v141,ulonglong2 *gm_v142, const unsigned *gm_v143, const int gm_v144
)  {
    __shared__ gl_s1 gm_v145;
    __shared__ int gm_v146[65],gm_v147[4];
    __shared__ int gm_v148[64], gm_v149[65], gm_v150[4];
    unsigned int gm_v151 = 0;
    int gm_v152 = 0, gm_v153 = 0, gm_v154 = 0;
    int gm_v155 = gm_v128, gm_v156 = gm_v129;
    int gm_v157 = 0, gm_v158 = gm_v129;
    bool gm_v159=false;
    bool gm_v160=false,gm_v161=false;int gm_v162=0;
    for (int gm_v163 = 0; gm_v163 < (int)gm_v112 && gm_v111 + gm_v163 < gm_v117; ++gm_v163) {
        const int gm_v71 = gm_v111 + gm_v163;
        const bool gm_v67=false;
        const bool gm_v64=gm_v138>=2*(int)gridDim.x;
        const int gm_v75=(gm_v115+(int)gridDim.x-1)/(int)gridDim.x;
        const int gm_v73=__ffs((unsigned)blockDim.x)-1;

        unsigned int gm_v164 = gm_v110 + gm_v151 * gridDim.x;
        const unsigned long long gm_v165 = 123456789ULL + (unsigned long long)gm_v71;
        const unsigned long long gm_v166 = (unsigned long long)max(0, gm_v118 - gm_v71);
        unsigned long long gm_v167 = 1ULL;
        for (int gm_v168 = 0; gm_v168 < gm_v119; ++gm_v168) gm_v167 *= gm_v166;

    __shared__ int gm_v29[64];
    __shared__ unsigned long long gm_v44, gm_v46[64];
    __shared__ int gm_v169[32];

    if (threadIdx.x < 64) {
        gm_v29[threadIdx.x] = threadIdx.x < gm_v28 ? __ldcg(&gm_v106[threadIdx.x]) : 0;
    }

    const int gm_v170 = gridDim.x * blockDim.x;
    const int gm_v171 = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gm_v9 = threadIdx.x & 31u;
    const int gm_v172 = (int)(blockDim.x >> 5);
    const int gm_v173 = (int)blockIdx.x * gm_v172 + (int)(threadIdx.x >> 5);
    const int gm_v174 = (int)gridDim.x * gm_v172;
    gl_a5(gm_v28, gm_v104, gm_v29, &gm_v44, gm_v46);

    int4 gm_v175; gm_v175.x = 0; gm_v175.y = 0; gm_v175.z = 0; gm_v175.w = 0;
    if (gm_v173 < gm_v108) {
        gm_v175 = __ldg(&gm_v107[gm_v173 * 32 + (int)gm_v9]);
    }

    if (gm_v171 < 64) gm_v131[gm_v171] = 0;
    for(int gm_v176=gm_v171;gm_v176<577;gm_v176+=gm_v170)gm_v141[gm_v176]=0;
    if(gm_v64)for(int gm_v176=gm_v171;gm_v176<2*(int)gridDim.x;gm_v176+=gm_v170)gm_v78[gm_v176]=0;
    if (gm_v171 == 0) {
        gm_v113[0] = gm_m1; gm_v116[0]=0;
    }

    if(gm_v163==0){
        for(int gm_v176=gm_v171;gm_v176<gm_v144;gm_v176+=gm_v170)gm_v139[gm_v176]=0;
        for(int gm_v177=gm_v171;gm_v177<gm_v138;gm_v177+=gm_v170)gm_v38[gm_v177]=make_ulonglong2(0,0);
        ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);
        for(int gm_v177=gm_v173;gm_v177<gm_v138;gm_v177+=gm_v174){
            int gm_v178=__ldg(&gm_v137[gm_v177]),gm_v179=__ldg(&gm_v137[gm_v177+1]);
            for(int gm_v176=gm_v178+(int)gm_v9;gm_v176<gm_v179;gm_v176+=32){
                int gm_v90=__ldg(&gm_v105[gm_v176]);
                int gm_v180=(unsigned)gm_v90<(unsigned)gm_v103?__ldcg(&gm_v91[gm_v90]):-1;
                if((unsigned)gm_v180<64u){
                    int gm_v181=gl_a15(gm_v139,gm_v143,gm_v177,gm_v180,false);
                    if(gm_v181==0)atomicXor(&gm_v38[gm_v177].x,1ULL<<gm_v180);
                    if(gm_v181==1)atomicXor(&gm_v38[gm_v177].y,1ULL<<gm_v180);
                }
            }
        }
    }else{
        const int gm_v182=__ldcg(&gm_v134[13]);
        const int gm_v183=(blockIdx.x*blockDim.x+threadIdx.x)/8;
        const int gm_v184=gridDim.x*blockDim.x/8,gm_v185=(int)threadIdx.x&7;
        for(int gm_v186=gm_v183;gm_v186<gm_v182;gm_v186+=gm_v184){
            unsigned long long gm_v187=__ldcg(&gm_v132[gm_v186]);
            int gm_v10=(int)(unsigned)gm_v187,gm_v188;
            if(gm_v160){
                gm_v188=__ldcg(gm_v133+gm_v186)?__ldcg(gm_v140+gm_v186):(int)((((unsigned)(gm_v187>>32))&63)^63);
                if(gm_v185==0){gm_v91[gm_v10]=gm_v188;if(gm_v161)gm_v70[gm_v10]=gm_v162;}
            }else gm_v188=__ldcg(&gm_v91[gm_v10]);
            int gm_v178=__ldg(&gm_v136[gm_v10]),gm_v179=__ldg(&gm_v136[gm_v10+1]);
            if((unsigned)gm_v188<64u)for(int gm_v176=gm_v178+gm_v185;gm_v176<gm_v179;gm_v176+=8){
                int gm_v177=__ldg(&gm_v89[gm_v176]);
                int gm_v181=gl_a15(gm_v139,gm_v143,gm_v177,gm_v188,false);
                if(gm_v181==0)atomicXor(&gm_v38[gm_v177].x,1ULL<<gm_v188);
                if(gm_v181==1)atomicXor(&gm_v38[gm_v177].y,1ULL<<gm_v188);
            }
        }
    }
    gm_v159=false;
    gm_v160=false;gm_v161=false;
    gm_v164=gm_v110+gm_v151*gridDim.x;

    gm_m2(gm_v109, gm_v164 + gridDim.x);


    int gm_v62 = gm_m1;

    for (int gm_v2 = gm_v173; gm_v2 < gm_v108; gm_v2 += gm_v174) {
        const int gm_v189 = gm_v2 + gm_v174;
        int4 gm_v190; gm_v190.x = 0; gm_v190.y = 0; gm_v190.z = 0; gm_v190.w = 0;
        if (gm_v189 < gm_v108) {
            gm_v190 = __ldg(&gm_v107[gm_v189 * 32 + (int)gm_v9]);
        }
        const int4 gm_v11 = gm_v175;
        switch (gm_v11.w) {
            case 0:  gm_m3(1,  0, gm_m4, gm_m5, gm_m6)  break;
            case 1:  gm_m3(2,  1, gm_m4, gm_m7, gm_m8)  break;
            case 2:  gm_m3(4,  2, gm_m4, gm_m9, gm_m10)  break;
            case 3:  gm_m3(8,  3, gm_m4, gm_m11, gm_m12)  break;
            case 4:  gm_m3(16, 4, gm_m4, gm_m13, gm_m14) break;
            default: gm_m3(32, 5, gm_m15, gm_m16, gm_m17) break;
        }
        gm_v175 = gm_v190;
    }

    {
        const int gm_v191 = gl_a2(gm_v62, gm_v169);
        if (threadIdx.x == 0) atomicMax(gm_v113, gm_v191);
    }

    gm_m2(gm_v109, gm_v164 + 2u * gridDim.x);

    const int gm_v192=__ldcg(gm_v113),gm_v193=(gm_v192*3)/4;
    const int gm_v194=(gm_v115+(int)gridDim.x-1)/(int)gridDim.x;
    const int gm_v195=(int)blockIdx.x*gm_v194;
    const int gm_v196=min(gm_v115,gm_v195+gm_v194);
    if(!gm_v64){
    int gm_v197=0;
    for(int gm_v198=gm_v195;gm_v198<gm_v196;++gm_v198){
        int gm_v10=gm_v198*(int)blockDim.x+(int)threadIdx.x;
        if(gm_v10<gm_v103){
            int gm_v187=__ldcg(&gm_v60[gm_v10]),gm_v199=__ldcg(&gm_v70[gm_v10]);
            if(gm_v187!=gm_m18){int gm_v200=gm_v187>>16;bool gm_v201=(gm_v199 > gm_v71) && (gm_v200 < gm_v193);
                if(!gm_v201 && gm_v200 <= 0 && gm_v200 >= -3)++gm_v197;
            }
        }
    }
    int gm_v202=gl_a1(gm_v197,gm_v169);
    if(threadIdx.x==0)gm_v114[blockIdx.x]=gm_v202;
    gm_m2(gm_v109,gm_v164+3u*gridDim.x);
    }
    int gm_v203=0;
    for(int gm_v204=(int)threadIdx.x;gm_v204<(int)blockIdx.x;gm_v204+=(int)blockDim.x){
        if(gm_v64){
            unsigned long long gm_v205=__ldcg(((const unsigned long long*)gm_v78)+gm_v204);
            gm_v203+=(int)(unsigned)gm_v205;
            if(!gm_v67 && gm_v192<=1)gm_v203+=(int)(unsigned)(gm_v205>>32);
        }else gm_v203+=__ldcg(&gm_v114[gm_v204]);
    }
    const int gm_v206=gl_a1(gm_v203,gm_v169);
    int gm_v207=0,gm_v208=0;
    for(int gm_v198=gm_v195;gm_v198<gm_v196;++gm_v198){
        const int gm_v10=gm_v198*(int)blockDim.x+(int)threadIdx.x;
        int gm_v187=gm_v10<gm_v103?__ldcg(&gm_v60[gm_v10]):gm_m18;
        int gm_v199=gm_v10<gm_v103?__ldcg(&gm_v70[gm_v10]):0;
        int gm_v200=gm_v187>>16,gm_v209=0,gm_v210=0;
        if(gm_v10<gm_v103 && gm_v187!=gm_m18){
            bool gm_v201=(gm_v199 > gm_v71) && (gm_v200 < gm_v193);
            gm_v210=!gm_v201 && gm_v200>0;
            gm_v209=(!gm_v201 && gm_v200 <= 0 && gm_v200 >= -3);
        }
        int gm_v211=0;
        int gm_v212=gl_a3(gm_v209,gm_v169,&gm_v211);
        int gm_v14=gm_v210?gm_v187:-1;
        if(gm_v209){
            unsigned long long gm_v213=(unsigned long long)(gm_v206+gm_v207+gm_v212)+1ULL;
            unsigned long long x=gl_a10(gm_v165,gm_v213,gm_v135);
            unsigned long long gm_v214=gm_v68*(1ULL+5ULL*(unsigned long long)(-gm_v200));
            if(gm_v214>0ULL && x%gm_v214<gm_v167)gm_v14=gm_v187&0xffff;
        }
        if(gm_v10<gm_v103)gm_v116[1+gm_v10]=gm_v14;
        if(gm_v14>=0){
            ++gm_v208;
            int gm_v215=gm_v14&63,gm_v216=gl_a14(gm_v131,gm_v215);
            if(gm_v216<4096)gm_v130[gm_v215*4096+gm_v216]=
                ((unsigned long long)((unsigned int)gm_v14^0x7fffffffu)<<32)|(unsigned int)gm_v10;
        }
        gm_v207+=gm_v211;
    }
    int gm_v217=gl_a1(gm_v208,gm_v169);
    if(threadIdx.x==0)atomicAdd(gm_v116,gm_v217);
    gm_v151+=gm_v64?2:3;
    ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);

        int gm_v218=threadIdx.x<64?__ldcg(&gm_v131[threadIdx.x]):0;
        int gm_v219=0;
        if(threadIdx.x<64){
            int gm_v220=threadIdx.x;
            int gm_v221=gm_v71<64?gm_v121:(gm_v71<256?gm_v122:gm_v123);
            int gm_v222=max(1,max(0,gm_v104-gm_v29[gm_v220])+gm_v221);
            gm_v219=gm_v220<gm_v28?min(gm_v218,gm_v222):0;
            gm_v148[gm_v220]=gm_v219;
            int gm_v223=gm_v218,gm_v224=gm_v219;
            for(int gm_v100=1;gm_v100<32;gm_v100<<=1){
                int gm_v225=__shfl_up_sync(0xffffffffu,gm_v223,gm_v100),gm_v226=__shfl_up_sync(0xffffffffu,gm_v224,gm_v100);
                if((gm_v220&31)>=gm_v100){gm_v223+=gm_v225;gm_v224+=gm_v226;}
            }
            gm_v146[gm_v220+1]=gm_v223;gm_v149[gm_v220+1]=gm_v224;
        }
        if(threadIdx.x==0){gm_v146[0]=0;gm_v149[0]=0;}
        __syncthreads();
        if(threadIdx.x>=32 && threadIdx.x<64){
            gm_v146[threadIdx.x+1]+=gm_v146[32];
            gm_v149[threadIdx.x+1]+=gm_v149[32];
        }
        int gm_v227=__syncthreads_or(threadIdx.x<64 && gm_v218>4096);
        int gm_v228=__syncthreads_or(threadIdx.x<64 && gm_v219>4096);
        if(threadIdx.x==0){
            gm_v150[0]=__ldcg(&gm_v116[0]);gm_v150[1]=gm_v149[64];
            gm_v150[2]=gm_v228;gm_v150[3]=gm_v227;
        }
        if(!gm_v227 && threadIdx.x<65)gm_v146[threadIdx.x]=threadIdx.x*4096;
        __syncthreads();
        int gm_v229 = gm_v150[0], gm_v230 = gm_v150[1];
        if(gm_v171==0)gm_v134[13]=gm_v230;
        if (gm_v229 == 0) { gm_v153 = 1; break; }
        int gm_v231 = gm_v71 < 50 ? gm_v120 / 2 :
                            (gm_v71 < 200 ? (gm_v120 * 3) / 4 : gm_v120 / 4);
        bool gm_v232 = gm_v229 <= gm_v231 ||
            (gm_v229 <= gm_v231 + 49152 && gm_v230 <= gm_v231);
        if (!gm_v232 || gm_v150[2] || gm_v230 <= 0 || gm_v103<256) { gm_v153 = 4; break; }


        if(gm_v150[3]){
            if(gm_v171<64)gm_v131[64+gm_v171]=0;
            ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);
            for(int gm_v186=gm_v171;gm_v186<gm_v103;gm_v186+=gm_v170){
                int gm_v187=__ldcg(&gm_v116[1+gm_v186]);
                if(gm_v187<0)continue;
                int gm_v10=gm_v186,gm_v215=gm_v187&63;
                int gm_v233=gl_a11(gm_v131 + 64,gm_v215);
                gm_v130[gm_v146[gm_v215]+gm_v233]=
                    ((unsigned long long)((unsigned int)gm_v187^0x7fffffffu)<<32)|(unsigned int)gm_v10;
            }
            ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);
        }
        for(int gm_v220=blockIdx.x;gm_v220<gm_v28 && gm_v220<64;gm_v220+=gridDim.x){
            int gm_v90=__ldcg(&gm_v131[gm_v220]),gm_v234=gm_v148[gm_v220],gm_v100=gm_v149[gm_v220];
            int gm_v235=gm_v146[gm_v220];
            if(gm_v234==0)continue;
            gl_a9(&gm_v130[gm_v235],gm_v90,gm_v234,&gm_v145,gm_v147);
            for(int gm_v176=threadIdx.x;gm_v176<gm_v234;gm_v176+=blockDim.x){
                unsigned long long gm_v187=__ldcg(&gm_v130[gm_v235+gm_v176]);
                int gm_v10=(int)(unsigned)gm_v187;
                int gm_v236=__ldcg(&gm_v91[gm_v10]);
                gm_v132[gm_v100+gm_v176]=gm_v187;gm_v140[gm_v100+gm_v176]=gm_v236;
                if((unsigned)gm_v236<(unsigned)gm_v28)atomicAdd(gm_v141+gm_v236,1);
                else atomicExch(gm_v141+576,1);
                int gm_v237=max(0,gm_v104-gm_v29[gm_v220]),gm_v238=gm_v176-gm_v237;
                if(gm_v238>=0 && gm_v238<8)gm_v142[gm_v220*8+gm_v238]=make_ulonglong2(gm_v187,(unsigned)((gm_v10<<12)|(gm_v236<<6)|gm_v220));

            }
        }
        ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);

        for(int gm_v186=gm_v171;gm_v186<gm_v230;gm_v186+=gm_v170){
            gm_v133[gm_v186]=0;
            unsigned long long gm_v187=__ldcg(gm_v132+gm_v186);
            int gm_v236=__ldcg(gm_v140+gm_v186);
            if((unsigned)gm_v236>=(unsigned)gm_v28)continue;
            int gm_v238=min(8,max(0,gm_v148[gm_v236]-max(0,gm_v104-gm_v29[gm_v236])));
            for(int gm_v176=0;gm_v176<gm_v238;++gm_v176){
                unsigned long long gm_v239=gl_a4(gm_v142+gm_v236*8+gm_v176).x;
                if(gm_v187<gm_v239)atomicAdd(gm_v141+64+gm_v236*8+gm_v176,1);
            }
        }
        ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);
        if(gridDim.x==1){
        const int gm_v240=(gridDim.x>1?((int)gridDim.x-1):(int)gridDim.x)*(int)blockDim.x/8;
        const int gm_v241=((gridDim.x>1?((int)blockIdx.x-1):(int)blockIdx.x)*(int)blockDim.x+(int)threadIdx.x)/8;
        const int gm_v185=(int)threadIdx.x&7;
        for(int gm_v186=gm_v241;gm_v186<gm_v230;gm_v186+=gm_v240){
            unsigned long long gm_v187=__ldcg(&gm_v132[gm_v186]);
            int gm_v10=(int)(unsigned)gm_v187,gm_v242=__ldcg(&gm_v140[gm_v186]);
            int gm_v178=__ldg(&gm_v136[gm_v10]),gm_v179=__ldg(&gm_v136[gm_v10+1]);
            if((unsigned)gm_v242<64u)for(int gm_v176=gm_v178+gm_v185;gm_v176<gm_v179;gm_v176+=8){
                int gm_v177=__ldg(&gm_v89[gm_v176]);
                int gm_v181=gl_a15(gm_v139,gm_v143,gm_v177,gm_v242,true);
                if(gm_v181==1)atomicXor(&gm_v38[gm_v177].x,1ULL<<gm_v242);
                if(gm_v181==2)atomicXor(&gm_v38[gm_v177].y,1ULL<<gm_v242);
            }
        }
        }
        if(blockIdx.x==0) {
            int gm_v243=gl_a17(gm_v230,gm_v103,gm_v28,gm_v104,
                gm_v91,gm_v106,gm_v132,gm_v148,gm_v149,
                gm_v70,gm_v71,gm_v124,gm_v125,gm_v126,&gm_v145,gm_v141,gm_v142,gm_v133);
            if(gm_v243<0){
            gm_v243=gl_a18(gm_v230,gm_v103,gm_v28,gm_v104,
                gm_v91,gm_v106,gm_v132,gm_v148,gm_v149,
                gm_v70,gm_v71,gm_v124,gm_v125,gm_v126,&gm_v145,gm_v141,gm_v142,gm_v133);
            }
            if(threadIdx.x==0){gm_v134[15]=gm_v243;gm_v134[14]=(gm_v243>=0);}
        } else {
        const int gm_v240=(gridDim.x>1?((int)gridDim.x-1):(int)gridDim.x)*(int)blockDim.x/8;
        const int gm_v241=((gridDim.x>1?((int)blockIdx.x-1):(int)blockIdx.x)*(int)blockDim.x+(int)threadIdx.x)/8;
        const int gm_v185=(int)threadIdx.x&7;
        for(int gm_v186=gm_v241;gm_v186<gm_v230;gm_v186+=gm_v240){
            unsigned long long gm_v187=__ldcg(&gm_v132[gm_v186]);
            int gm_v10=(int)(unsigned)gm_v187,gm_v242=__ldcg(&gm_v140[gm_v186]);
            int gm_v178=__ldg(&gm_v136[gm_v10]),gm_v179=__ldg(&gm_v136[gm_v10+1]);
            if((unsigned)gm_v242<64u)for(int gm_v176=gm_v178+gm_v185;gm_v176<gm_v179;gm_v176+=8){
                int gm_v177=__ldg(&gm_v89[gm_v176]);
                int gm_v181=gl_a15(gm_v139,gm_v143,gm_v177,gm_v242,true);
                if(gm_v181==1)atomicXor(&gm_v38[gm_v177].x,1ULL<<gm_v242);
                if(gm_v181==2)atomicXor(&gm_v38[gm_v177].y,1ULL<<gm_v242);
            }
        }
        }
        gm_v159=true;
        ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);
        if(__ldcg(&gm_v134[15])<0) {
        const bool gm_v244 = gm_v230 <= 4096;
        if(gm_v244){
            for(int gm_v176=threadIdx.x;gm_v176<gm_v230;gm_v176+=blockDim.x)gm_v145.gl_v129[gm_v176]=__ldcg(&gm_v132[gm_v176]);
            __syncthreads();
        }
        for (int gm_v186 = gm_v173; gm_v186 < gm_v230; gm_v186 += gm_v174) {
            unsigned long long gm_v187 = gm_v244 ? gm_v145.gl_v129[gm_v186] : __ldcg(&gm_v132[gm_v186]);
            int gm_v212 = 0;
            for (int gm_v220 = gm_v9; gm_v220 < gm_v28 && gm_v220 < 64; gm_v220 += 32) {
                int gm_v245 = 0, gm_v246 = gm_v148[gm_v220];
                while (gm_v245 < gm_v246) {
                    int gm_v247 = (gm_v245 + gm_v246) >> 1;
                    unsigned long long gm_v248 = gm_v244 ? gm_v145.gl_v129[gm_v149[gm_v220]+gm_v247] : __ldcg(&gm_v130[gm_v146[gm_v220] + gm_v247]);
                    if (gm_v248 < gm_v187) gm_v245 = gm_v247 + 1; else gm_v246 = gm_v247;
                }
                gm_v212 += gm_v245;
            }
            for (int gm_v249 = 16; gm_v249 > 0; gm_v249 >>= 1) gm_v212 += __shfl_xor_sync(0xffffffffu, gm_v212, gm_v249);
            if (gm_v9 == 0) {
                int gm_v10 = (int)(unsigned int)gm_v187;
                int gm_v215 = (((unsigned int)(gm_v187 >> 32)) & 63) ^ 63;
                gm_v133[gm_v212] = (gm_v10 << 6) | gm_v215;
                int gm_v237=max(0,gm_v104-gm_v29[gm_v215]);
                int gm_v250=gm_v186-gm_v149[gm_v215]-gm_v237;
                if(gm_v250>=0 && gm_v250<8) gm_v116[1+gm_v215*8+gm_v250]=gm_v212;
            }
        }
        ++gm_v151; gm_m2(gm_v109, gm_v110 + gm_v151 * gridDim.x);
        if (blockIdx.x == 0) {
            int gm_v251 = gl_a8(gm_v230, gm_v103, gm_v28, gm_v104,
                gm_v91, gm_v106, gm_v133, gm_v70, gm_v71,
                gm_v124, gm_v125, gm_v126, &gm_v145, gm_v169, gm_v116+1, gm_v148);
            if (threadIdx.x == 0) gm_v134[15] = gm_v251;
        }
        ++gm_v151; gm_m2(gm_v109, gm_v110 + gm_v151 * gridDim.x);
        }
        int gm_v251 = __ldcg(&gm_v134[15]);
        gm_v160=__ldcg(gm_v134+14)!=0;
        gm_v161=gm_v251>0 && (gm_v125>0 || gm_v126>0);
        gm_v162=gm_v71+gm_v124;

        if(gm_v251==0 && gm_v229>gm_v231){gm_v153=4;break;}
        gm_v154 += gm_v251;
        ++gm_v152;
        if (gm_v251 == 0) {
            ++gm_v155;
            ++gm_v156;
            ++gm_v157;
            gm_v158 = max(gm_v158, gm_v156);
            if (gm_v155 >= 3 && gm_v71 < max(0, gm_v127 - 50)) { gm_v153 = 2; break; }
            if (gm_v155 > 30) { gm_v153 = 3; break; }
        } else { gm_v155 = 0; gm_v156 = 0; }

    }
    if(gm_v159){
        const int gm_v182=__ldcg(&gm_v134[13]);
        const int gm_v183=(blockIdx.x*blockDim.x+threadIdx.x)/8;
        const int gm_v184=gridDim.x*blockDim.x/8,gm_v185=(int)threadIdx.x&7;
        for(int gm_v186=gm_v183;gm_v186<gm_v182;gm_v186+=gm_v184){
            unsigned long long gm_v187=__ldcg(&gm_v132[gm_v186]);
            int gm_v10=(int)(unsigned)gm_v187,gm_v188;
            if(gm_v160){
                gm_v188=__ldcg(gm_v133+gm_v186)?__ldcg(gm_v140+gm_v186):(int)((((unsigned)(gm_v187>>32))&63)^63);
                if(gm_v185==0){gm_v91[gm_v10]=gm_v188;if(gm_v161)gm_v70[gm_v10]=gm_v162;}
            }else gm_v188=__ldcg(&gm_v91[gm_v10]);
            int gm_v178=__ldg(&gm_v136[gm_v10]),gm_v179=__ldg(&gm_v136[gm_v10+1]);
            if((unsigned)gm_v188<64u)for(int gm_v176=gm_v178+gm_v185;gm_v176<gm_v179;gm_v176+=8){
                int gm_v177=__ldg(&gm_v89[gm_v176]);
                int gm_v181=gl_a15(gm_v139,gm_v143,gm_v177,gm_v188,false);
                if(gm_v181==0)atomicXor(&gm_v38[gm_v177].x,1ULL<<gm_v188);
                if(gm_v181==1)atomicXor(&gm_v38[gm_v177].y,1ULL<<gm_v188);
            }
        }
        ++gm_v151; gm_m2(gm_v109,gm_v110+gm_v151*gridDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        gm_v134[0] = gm_v153; gm_v134[1] = gm_v152; gm_v134[2] = (int)gm_v151;
        gm_v134[3] = gm_v154; gm_v134[4] = gm_v155; gm_v134[5] = gm_v156;
        gm_v134[7] = gm_v157; gm_v134[8] = gm_v158;
    }
}

#undef gm_m1
#undef gm_m2
#undef gm_m3
#undef gm_m4
#undef gm_m5
#undef gm_m6
#undef gm_m7
#undef gm_m8
#undef gm_m9
#undef gm_m10
#undef gm_m11
#undef gm_m12
#undef gm_m13
#undef gm_m14
#undef gm_m15
#undef gm_m16
#undef gm_m17
#undef gm_m18
#undef gm_m19
#undef gm_m20
#undef gm_m21
#undef gm_m22
#undef gm_m23
#undef gm_m24
#undef gm_m25
#undef gm_m26
#undef gm_m27
#undef gm_m28
