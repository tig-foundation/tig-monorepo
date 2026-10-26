
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
extern "C" __global__ void __launch_bounds__(256)
    grouped_small_finish(unsigned long long *h, const unsigned long long *c,
                         const unsigned long long *picks, const unsigned long long *sorted,
                         int *part, int *sizes_out, int *tabu) {
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
