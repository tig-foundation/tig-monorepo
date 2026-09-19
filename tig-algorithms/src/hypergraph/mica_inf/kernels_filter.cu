// Device-side move selection: per-block statistics, tabu/lottery filtering, one-block radix
// sort and per-target quota. Invariant: no block ever waits for another one.
// The extern "C" entries keep their unqualified names inside the namespace.

namespace filt {

// Block sizes of the filter kernels and of the single-block sort kernels.
constexpr int FILT_THREADS = 256;
constexpr int S_THREADS = 1024;

__device__ __forceinline__ int chunk_of(int n, int grid) { return (n + grid - 1) / grid; }

// Two key encodings: mode 0 is (gain << 16 | ...) with 0x80000000 for "no move",
// mode 1 is ((gain + 1000) << 16 | ...) with 0 for "no move".
__device__ __forceinline__ bool key_valid(int key, int mode) {
    return mode == 0 ? ((unsigned int)key != 0x80000000u) : (key > 0);
}
__device__ __forceinline__ int key_gain(int key, int mode) { return (key >> 16) - (mode == 0 ? 0 : 1000); }

// Block-wide primitives on one int per thread: warp shuffles, then one pass over the warp results.
template <int T>
__device__ __forceinline__ int blk_scan(int v, int* sh, int* total) {
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

template <int T>
__device__ __forceinline__ int blk_max(int v, int* sh) {
    constexpr int W = T / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 16; o > 0; o >>= 1) x = max(x, __shfl_xor_sync(0xffffffffu, x, o));
    if (lane == 0) sh[warp] = x;
    __syncthreads();
    int r = sh[0];
    for (int i = 1; i < W; i++) r = max(r, sh[i]);
    __syncthreads();
    return r;
}

template <int T>
__device__ __forceinline__ int blk_sum(int v, int* sh) {
    constexpr int W = T / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) sh[warp] = x;
    __syncthreads();
    int r = 0;
    for (int i = 0; i < W; i++) r += sh[i];
    __syncthreads();
    return r;
}

// Per-block statistics: best gain, and the counts of lottery candidates (gain in
// [lottery_min, 0]) split by tabu status and gain.
extern "C" __global__ void mica_filt_stats(
    const int* __restrict__ keys, const int* __restrict__ tabu_until, int n, int round, int lottery_min, int mode, int gmin,
    int* __restrict__ block_max, int* __restrict__ stats)
{
    __shared__ int sh[32];
    int chunk = chunk_of(n, gridDim.x);
    int lo = blockIdx.x * chunk;
    int hi = min(n, lo + chunk);
    int m = -2147483647 - 1;
    int nt = 0, tb0 = 0, tb1 = 0, tb2 = 0, tb3 = 0;
    for (int i = lo + threadIdx.x; i < hi; i += FILT_THREADS) {
        int key = keys[i];
        if (key_valid(key, mode) && key_gain(key, mode) >= gmin) {
            int g = key_gain(key, mode);
            if (g > m) m = g;
            if (mode == 0 && g <= 0 && g >= lottery_min) {
                if (tabu_until[i] > round) {
                    tb0 += (g == -3); tb1 += (g == -2); tb2 += (g == -1); tb3 += (g == 0);
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

// Lottery candidates of block b that are not tabu, or tabu with gain >= aspiration.
__device__ __forceinline__ int lottery_count(const int* stats, int b, int asp) {
    int c = stats[b * 5];
    if (-3 >= asp) c += stats[b * 5 + 1];
    if (-2 >= asp) c += stats[b * 5 + 2];
    if (-1 >= asp) c += stats[b * 5 + 3];
    if (0 >= asp) c += stats[b * 5 + 4];
    return c;
}

// LCG state after `steps` draws.
__device__ __forceinline__ unsigned long long lcg_jump(unsigned long long s0, unsigned long long steps, unsigned long long c) {
    unsigned long long A = 1ull, C = 0ull;
    unsigned long long a = 6364136223846793005ull;
    while (steps) {
        if (steps & 1ull) { C = C * a + c; A = A * a; }
        c = c * a + c;
        a = a * a;
        steps >>= 1;
    }
    return A * s0 + C;
}

// Candidate selection: positive gains pass, gains in [lottery_min, 0] pass a seeded lottery
// with odds num/den[penalty]; each block writes its own segment of `out`.
extern "C" __global__ void mica_filt_pick(
    const int* __restrict__ keys, const int* __restrict__ tabu_until, int n, int round, int lottery_min, int mode, int gmin,
    const int* __restrict__ block_max, const int* __restrict__ stats,
    unsigned long long rng0, unsigned long long lcg_c, unsigned long long num,
    unsigned long long den0, unsigned long long den1, unsigned long long den2, unsigned long long den3,
    unsigned long long* __restrict__ out, int* __restrict__ block_sel)
{
    __shared__ int sh[32];
    __shared__ int carry_lottery;
    __shared__ int carry_sel;
    const int G = gridDim.x;
    int m = -2147483647 - 1;
    for (int b = threadIdx.x; b < G; b += FILT_THREADS) m = max(m, block_max[b]);
    int max_gain = blk_max<FILT_THREADS>(m, sh);
    if (max_gain == (-2147483647 - 1)) max_gain = 0;
    int aspiration = (max_gain * 3) / 4;
    if (mode != 0) aspiration = max(1, aspiration);
    int lp = 0;
    for (int b = threadIdx.x; b < (int)blockIdx.x; b += FILT_THREADS) lp += lottery_count(stats, b, aspiration);
    int lottery_prefix = blk_sum<FILT_THREADS>(lp, sh);

    int chunk = chunk_of(n, G);
    int lo = blockIdx.x * chunk;
    int hi = min(n, lo + chunk);
    if (threadIdx.x == 0) { carry_lottery = 0; carry_sel = 0; }
    __syncthreads();
    for (int base = lo; base < hi; base += FILT_THREADS) {
        int i = base + threadIdx.x;
        int key = 0x80000000;
        int f = 0;
        if (i < hi) {
            key = keys[i];
            if (key_valid(key, mode) && key_gain(key, mode) >= gmin) {
                int gain = key_gain(key, mode);
                bool is_tabu = (tabu_until[i] > round) && (gain < aspiration);
                if (!is_tabu) {
                    if (mode != 0 || gain > 0) f = 1;
                    else if (gain >= lottery_min) f = 2;
                }
            }
        }
        int total_lottery;
        int rank = blk_scan<FILT_THREADS>(f == 2 ? 1 : 0, sh, &total_lottery);
        int s = 0;
        int ok = 0;
        if (f == 1) { s = 1; ok = key; }
        else if (f == 2) {
            int gain = key >> 16;
            int penalty = (gain < 0) ? (-gain) : 0;
            unsigned long long den = (penalty & 3) == 0 ? den0 : (penalty & 3) == 1 ? den1 : (penalty & 3) == 2 ? den2 : den3;
            unsigned long long st = lcg_jump(rng0, (unsigned long long)(lottery_prefix + carry_lottery + rank + 1), lcg_c);
            if (den > 0ull && (st % den) < num) { s = 1; ok = key & 0xFFFF; }
        }
        int total_sel;
        int pos = blk_scan<FILT_THREADS>(s, sh, &total_sel);
        if (s) {
            out[lo + carry_sel + pos] = ((unsigned long long)(unsigned int)i << 32) | (unsigned long long)(unsigned int)ok;
        }
        __syncthreads();
        if (threadIdx.x == 0) { carry_lottery += total_lottery; carry_sel += total_sel; }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sel[blockIdx.x] = carry_sel;
}

// One-block ordering and quota selection. picks[0] = count | fallback << 32, picks[1] = candidates,
// picks[2..] = (node << 32) | key ; the ordered list is left in `sorted`.
__device__ __forceinline__ int s_digit(unsigned long long w, int p) { return (int)((w >> (32 + 8 * p)) & 255ull); }

struct SShared {
    int sh[32];
    int h4[4 * 256];
    unsigned short wcount[32 * 256];
    int run[256];
    int ttot[256];
    int quota[64];
    int skipf[4];
    int carry;
};

// Whole-block sort of sorted[0..m] then quota selection.
__device__ void s_sort_select(SShared& S, int m,
    unsigned long long* __restrict__ buf, unsigned long long* __restrict__ sorted,
    const int* __restrict__ nodes_in_part, int num_parts, int max_part_size, int slack,
    int limit, int extra, unsigned long long* __restrict__ picks)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;
    if (tid < 4) {
        int sk = 0;
        for (int d = 0; d < 256; d++) if (S.h4[tid * 256 + d] == m) { sk = 1; break; }
        S.skipf[tid] = sk;
    }
    __syncthreads();
    int L = (!S.skipf[0]) + (!S.skipf[1]) + (!S.skipf[2]) + (!S.skipf[3]);
    const unsigned long long* src = sorted;
    unsigned long long* dst = buf;
    if (L & 1) {
        for (int i = tid; i < m; i += S_THREADS) buf[i] = sorted[i];
        __syncthreads();
        src = buf; dst = sorted;
    }
    for (int p = 0; p < 4; p++) {
        if (S.skipf[p]) continue;
        int tot;
        int r = blk_scan<S_THREADS>(tid < 256 ? S.h4[p * 256 + tid] : 0, S.sh, &tot);
        if (tid < 256) S.run[tid] = r;
        __syncthreads();
        for (int base = 0; base < m; base += S_THREADS) {
            const int i = base + tid;
            const bool valid = i < m;
            unsigned long long w = valid ? src[i] : 0ull;
            const int d = s_digit(w, p);
            for (int q = tid; q < 32 * 256; q += S_THREADS) S.wcount[q] = 0;
            __syncthreads();
            unsigned int vm = __ballot_sync(0xffffffffu, valid);
            unsigned int peers = 0u; int rk = 0;
            if (valid) {
                peers = __match_any_sync(vm, d);
                rk = __popc(peers & lanemask_lt);
                if (rk == 0) S.wcount[warp * 256 + d] = (unsigned short)__popc(peers);
            }
            __syncthreads();
            if (tid < 256) {
                int a = 0;
                for (int ww = 0; ww < 32; ww++) { int c = S.wcount[ww * 256 + tid]; S.wcount[ww * 256 + tid] = (unsigned short)a; a += c; }
                S.ttot[tid] = a;
            }
            __syncthreads();
            if (valid) dst[S.run[d] + (int)S.wcount[warp * 256 + d] + rk] = w;
            __syncthreads();
            if (tid < 256) S.run[tid] += S.ttot[tid];
            __syncthreads();
        }
        const unsigned long long* t = src; src = dst; dst = (unsigned long long*)t;
    }

    const int k_base = min(m, limit);
    const int kc = min(m, k_base + extra);
    if (tid < 64) {
        int q = 0;
        if (tid < num_parts) { int fr = max_part_size - nodes_in_part[tid]; if (fr < 0) fr = 0; q = max(1, fr + slack); }
        S.quota[tid] = q;
        S.run[tid] = 0;
    }
    if (tid == 0) S.carry = 0;
    __syncthreads();
    for (int base = 0; base < kc; base += S_THREADS) {
        const int i = base + tid;
        const bool valid = i < kc;
        unsigned long long w = valid ? sorted[i] : 0ull;
        const int tgt = (int)(~(unsigned int)(w >> 32)) & 63;
        for (int q = tid; q < 32 * 64; q += S_THREADS) S.wcount[q] = 0;
        __syncthreads();
        unsigned int vm = __ballot_sync(0xffffffffu, valid);
        unsigned int peers = 0u; int rk = 0;
        if (valid) {
            peers = __match_any_sync(vm, tgt);
            rk = __popc(peers & lanemask_lt);
            if (rk == 0) S.wcount[warp * 64 + tgt] = (unsigned short)__popc(peers);
        }
        __syncthreads();
        if (tid < 64) {
            int a = 0;
            for (int ww = 0; ww < 32; ww++) { int c = S.wcount[ww * 64 + tid]; S.wcount[ww * 64 + tid] = (unsigned short)a; a += c; }
            S.ttot[tid] = a;
        }
        __syncthreads();
        int acc = 0;
        if (valid && tgt < num_parts) {
            int rank_in_part = S.run[tgt] + (int)S.wcount[warp * 64 + tgt] + rk;
            acc = (rank_in_part < S.quota[tgt]) ? 1 : 0;
        }
        int tot;
        int pos = blk_scan<S_THREADS>(acc, S.sh, &tot) + S.carry;
        if (acc && pos < k_base) picks[2 + pos] = (w << 32) | (unsigned long long)(~(unsigned int)(w >> 32));
        __syncthreads();
        if (tid < 64) S.run[tid] += S.ttot[tid];
        if (tid == 0) S.carry += tot;
        __syncthreads();
    }
    int total_acc = S.carry;
    if (total_acc == 0) {
        int take = min(k_base, kc);
        for (int i = tid; i < take; i += S_THREADS) {
            unsigned long long w = sorted[i];
            picks[2 + i] = (w << 32) | (unsigned long long)(~(unsigned int)(w >> 32));
        }
        if (tid == 0) { picks[0] = (unsigned long long)take | (1ull << 32); picks[1] = (unsigned long long)m; }
    } else if (tid == 0) {
        picks[0] = (unsigned long long)min(total_acc, k_base); picks[1] = (unsigned long long)m;
    }
}

// Warp-aggregated digit histograms.
__device__ __forceinline__ void s_hist_add(int* h4, unsigned long long w, bool valid, unsigned int lanemask_lt) {
    unsigned int vm = __ballot_sync(0xffffffffu, valid);
    if (valid) {
        for (int p = 0; p < 4; p++) {
            int d = s_digit(w, p);
            unsigned int peers = __match_any_sync(vm, d);
            if ((peers & lanemask_lt) == 0u) atomicAdd(&h4[p * 256 + d], __popc(peers));
        }
    }
}

// Gathers the per-block segments, then sorts and selects.
extern "C" __global__ void __launch_bounds__(1024) mica_sortsel(
    const unsigned long long* __restrict__ segs, const int* __restrict__ block_sel, int G, int chunk,
    unsigned long long* __restrict__ buf, unsigned long long* __restrict__ sorted,
    const int* __restrict__ nodes_in_part, int num_parts, int max_part_size, int slack,
    int limit, int extra,
    unsigned long long* __restrict__ picks)
{
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
    for (int i = tid; i < 1024; i += S_THREADS) S.h4[i] = 0;
    __syncthreads();
    for (int base = 0; base < m; base += S_THREADS) {
        int i = base + tid;
        bool valid = i < m;
        unsigned long long w = 0ull;
        if (valid) {
            int a = 0, b = G;
            while (b - a > 1) { int mid = (a + b) >> 1; if (seg_off[mid] <= i) a = mid; else b = mid; }
            unsigned long long o = segs[a * chunk + (i - seg_off[a])];
            unsigned int key = (unsigned int)(o & 0xFFFFFFFFull);
            unsigned int node = (unsigned int)(o >> 32);
            w = ((unsigned long long)(~key) << 32) | (unsigned long long)node;
            sorted[i] = w;
        }
        s_hist_add(S.h4, w, valid, lanemask_lt);
    }
    __syncthreads();
    s_sort_select(S, m, buf, sorted, nodes_in_part, num_parts, max_part_size, slack, limit, extra, picks);
}

// Single-block form of mica_filt_stats + mica_filt_pick + mica_sortsel.
extern "C" __global__ void __launch_bounds__(1024) mica_pick1(
    const int* __restrict__ keys, const int* __restrict__ tabu_until, int n, int round, int lottery_min, int mode, int gmin,
    unsigned long long rng0, unsigned long long lcg_c, unsigned long long num,
    unsigned long long den0, unsigned long long den1, unsigned long long den2, unsigned long long den3,
    unsigned long long* __restrict__ buf, unsigned long long* __restrict__ sorted,
    const int* __restrict__ nodes_in_part, int num_parts, int max_part_size, int slack,
    int limit, int extra,
    unsigned long long* __restrict__ picks)
{
    __shared__ SShared S;
    __shared__ int carry_lottery;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned int lanemask_lt = (1u << lane) - 1u;

    int mx = -2147483647 - 1;
    for (int i = tid; i < n; i += S_THREADS) {
        int key = keys[i];
        if (key_valid(key, mode) && key_gain(key, mode) >= gmin) mx = max(mx, key_gain(key, mode));
    }
    int max_gain = blk_max<S_THREADS>(mx, S.sh);
    if (max_gain == (-2147483647 - 1)) max_gain = 0;
    int aspiration = (max_gain * 3) / 4;
    if (mode != 0) aspiration = max(1, aspiration);
    for (int i = tid; i < 1024; i += S_THREADS) S.h4[i] = 0;
    if (tid == 0) { carry_lottery = 0; S.carry = 0; }
    __syncthreads();

    for (int base = 0; base < n; base += S_THREADS) {
        int i = base + tid;
        int key = 0x80000000;
        int f = 0;
        if (i < n) {
            key = keys[i];
            if (key_valid(key, mode) && key_gain(key, mode) >= gmin) {
                int gain = key_gain(key, mode);
                bool is_tabu = (tabu_until[i] > round) && (gain < aspiration);
                if (!is_tabu) {
                    if (mode != 0 || gain > 0) f = 1;
                    else if (gain >= lottery_min) f = 2;
                }
            }
        }
        int total_lottery;
        int rank = blk_scan<S_THREADS>(f == 2 ? 1 : 0, S.sh, &total_lottery);
        int sel = 0;
        int ok = 0;
        if (f == 1) { sel = 1; ok = key; }
        else if (f == 2) {
            int gain = key >> 16;
            int penalty = (gain < 0) ? (-gain) : 0;
            unsigned long long den = (penalty & 3) == 0 ? den0 : (penalty & 3) == 1 ? den1 : (penalty & 3) == 2 ? den2 : den3;
            unsigned long long st = lcg_jump(rng0, (unsigned long long)(carry_lottery + rank + 1), lcg_c);
            if (den > 0ull && (st % den) < num) { sel = 1; ok = key & 0xFFFF; }
        }
        int total_sel;
        int pos = blk_scan<S_THREADS>(sel, S.sh, &total_sel);
        unsigned long long w = ((unsigned long long)(~(unsigned int)ok) << 32) | (unsigned long long)(unsigned int)i;
        if (sel) sorted[S.carry + pos] = w;
        s_hist_add(S.h4, w, sel != 0, lanemask_lt);
        __syncthreads();
        if (tid == 0) { carry_lottery += total_lottery; S.carry += total_sel; }
        __syncthreads();
    }
    const int m = S.carry;
    __syncthreads();
    s_sort_select(S, m, buf, sorted, nodes_in_part, num_parts, max_part_size, slack, limit, extra, picks);
}

} // namespace filt
