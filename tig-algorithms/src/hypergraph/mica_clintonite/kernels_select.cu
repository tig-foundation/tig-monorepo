
namespace filt {

constexpr int FILT_THREADS = 256;
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

extern "C" __global__ void mica_filt_stats(const int *__restrict__ keys,
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

extern "C" __global__ void
mica_filt_pick4(const int *__restrict__ keys, const int *__restrict__ tabu_until, int n, int round,
                int lottery_min, const int *__restrict__ block_max, const int *__restrict__ stats,
                unsigned long long rng0, unsigned long long lcg_c, unsigned long long num,
                unsigned long long den0, unsigned long long den1, unsigned long long den2,
                unsigned long long den3, unsigned long long *__restrict__ out,
                int *__restrict__ block_sel, const unsigned long long *gr) {
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
            if (s4[j]) {
                const int i = base + threadIdx.x * 4 + j;
                out[lo + carry_sel + pos_base + p] = ((unsigned long long)(unsigned int)i << 32) |
                                                     (unsigned long long)(unsigned int)ok4[j];
                p++;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            carry_lottery += total_lottery;
            carry_sel += total_sel;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sel[blockIdx.x] = carry_sel;
}

__device__ __forceinline__ int s_digit(unsigned long long w, int p) {
    return (int)((w >> (32 + 8 * p)) & 255ull);
}

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

__device__ __forceinline__ void s_hist_add(int *h4, unsigned long long w, bool valid,
                                           unsigned int lanemask_lt) {
    unsigned int vm = __ballot_sync(0xffffffffu, valid);
    if (valid) {
        for (int p = 0; p < 4; p++) {
            int d = s_digit(w, p);
            unsigned int peers = __match_any_sync(vm, d);
            if ((peers & lanemask_lt) == 0u) atomicAdd(&h4[p * 256 + d], __popc(peers));
        }
    }
}

extern "C" __global__ void __launch_bounds__(1024)
    mica_sortsel(const unsigned long long *__restrict__ segs, const int *__restrict__ block_sel,
                 int G, int chunk, unsigned long long *__restrict__ buf,
                 unsigned long long *__restrict__ sorted, const int *__restrict__ nodes_in_part,
                 int num_parts, int max_part_size, int slack, int limit, int extra,
                 unsigned long long *__restrict__ picks, const unsigned long long *gr) {
    if (gr) {
        if (gr[0] != 0) return;
        limit = (int)gr[14];
        slack = (int)gr[15];
    }
    __shared__ SShared S;
    __shared__ int seg_off[257];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;

    int v = (tid < G) ? block_sel[tid] : 0;
    int m;
    int off = blk_scan<S_THREADS>(v, S.sh, &m);
    if (tid < G) seg_off[tid] = off;
    if (tid == 0) seg_off[G] = m;
    for (int i = tid; i < 1024; i += S_THREADS)
        S.h4[i] = 0;
    __syncthreads();
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
        s_hist_add(S.h4, w, valid, lanemask_lt);
    }
    __syncthreads();
    s_sort_select(S, m, buf, sorted, nodes_in_part, num_parts, max_part_size, slack, limit, extra,
                  picks);
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

} // namespace filt


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
} // namespace grouped

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

extern "C" __global__ void __launch_bounds__(256)
    grouped_finish(unsigned long long *h, const unsigned long long *c,
                   const unsigned long long *picks, const unsigned long long *sorted, int *part,
                   int *sizes_out, int *tabu, const int *offsets, int *pairs) {
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
                if (crit) seq[cpos] = tid;
                __syncthreads();
                if (tid < 32) {
                    for (int j = 0; j < ccount; ++j) {
                        int i = seq[j], s = source[i], t = target[i];
                        int cs = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
                        int ts = __shfl_sync(0xffffffffu, t < 32 ? lo : hi, t & 31);
                        int yes = ts < cap && cs > 1;
                        lo += (int)(yes && t == lane) - (int)(yes && s == lane);
                        hi += (int)(yes && t == lane + 32) - (int)(yes && s == lane + 32);
                        if (lane == 0) take[i] = yes;
                    }
                }
                if (tid < 64) {
                    incoming[tid] = 0;
                    outgoing[tid] = 0;
                }
                __syncthreads();
                if (tid < length && source[tid] >= 0 && !crit) {
                    atomicAdd(incoming + target[tid], 1);
                    atomicAdd(outgoing + source[tid], 1);
                }
                __syncthreads();
                if (tid < 32) {
                    lo += incoming[lane] - outgoing[lane];
                    hi += incoming[lane + 32] - outgoing[lane + 32];
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
                h[0] = 1;
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
                    for (int j = 0; j < ccount; ++j) {
                        int i = seq[j], s = source[i], t = target[i];
                        int cs = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
                        int ts = __shfl_sync(0xffffffffu, t < 32 ? lo : hi, t & 31);
                        int yes = ts < cap && cs > 1;
                        lo += (int)(yes && t == lane) - (int)(yes && s == lane);
                        hi += (int)(yes && t == lane + 32) - (int)(yes && s == lane + 32);
                        if (!lane) take[i] = yes;
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
