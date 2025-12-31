/*!
Copyright 2025 The Granite Labs LLC

Identity of Submitter Granite Labs LLC

Identity of Creator of Algorithmic Method Granite Labs LLC

UAI c004_a072

Licensed under the TIG Inbound Game License v3.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

//
// stat_filter
//
// Filtering based on Median Absolute Deviation (MAD):
// We compute the median of all L2 norms, then calculate the MAD (median of
// absolute deviations from the median). The threshold is set to:
//      norm_threshold = scale_factor × MAD × 1.4826
// The factor 1.4826 scales MAD to match the standard deviation for normally
// distributed data. This makes the filter more robust to outliers compared to
// filtering methods based on mean and standard deviation, which are more
// sensitive to extreme values.
//
// Reference:
// - NIST Engineering Statistics Handbook:
//   https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
// - See also: https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm
//

#include <float.h>
#include <math_constants.h>   // defines CUDART_INF_F, CUDART_NAN_F, etc.



//-------------------- Dimension Stats --------------------------

__device__ inline void atomicMaxFloat(float* addr, float val) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int  old    = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ inline void atomicMinFloat(float* addr, float val) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int  old    = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

// Initialize out_min/out_max
extern "C" __global__ void init_minmax_kernel(
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    int dims,
    float min_init,   // e.g., +INF
    float max_init)   // e.g., -INF (or 0 if you know data is >=0)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dims) {
        out_min[d] = min_init;
        out_max[d] = max_init;
    }
}

// Compute per-dim min and max over all vectors
extern "C" __global__ void compute_dim_stats_kernel(
    const float* __restrict__ db,   // [num_vecs * dims]
    float* __restrict__ out_min,    // [dims]
    float* __restrict__ out_max,    // [dims]
    int num_vecs,
    int dims)
{
    int d = threadIdx.x;
    for (int v = 0; v < num_vecs; ++v) {
        float x = db[(size_t)v * dims + d];
        atomicMinFloat(&out_min[d], x);
        atomicMaxFloat(&out_max[d], x);
    }
}



//-------------------- Calculate Dimension Divisors -------------

// Build per-dimension divisors from min/max. 
// Scale the min/max down so we throw away outliers.  

#ifndef FRAC_OF_MIN_MAX
//#define FRAC_OF_MIN_MAX 0.90f
#define FRAC_OF_MIN_MAX 0.80f
#endif

extern "C" __global__ void build_u4_divisors_from_minmax_kernel(
    float* __restrict__ dim_min,  // [dims]
    float* __restrict__ dim_max,  // [dims]
    float* __restrict__ s,        // [dims]
    int    dims)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dims) return;

    float mn = dim_min[d];
    float mx = dim_max[d];

    float range = mx - mn;
    if (!isfinite(range) || range <= 0.0f) {
        // Constant or degenerate dim: mark with s[d] = 0 to signal "constant"
        s[d] = 0.0f;
        return;
    }

    // Shrink to the central FRAC_OF_MIN_MAX of the range
    float mid  = 0.5f * (mx + mn);
    float half = 0.5f * FRAC_OF_MIN_MAX * range;

    mn = mid - half;
    mx = mid + half;

    // Write back the trimmed bounds so quantization uses them too
    dim_min[d] = mn;
    dim_max[d] = mx;

    // Normal scale: map (trimmed) range into ~16 buckets
    float trimmed_range = mx - mn;      // == FRAC_OF_MIN_MAX * original range
    float step = trimmed_range / 16.0f;
    s[d] = step;
}

extern "C" __global__ void build_u2_divisors_from_minmax_kernel(
    float* __restrict__ dim_min,  // [dims]
    float* __restrict__ dim_max,  // [dims]
    float* __restrict__ s,        // [dims]
    int    dims)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dims) return;

    float mn = dim_min[d];
    float mx = dim_max[d];

    float range = mx - mn;
    if (!isfinite(range) || range <= 0.0f) {
        // Constant or degenerate dim: mark with s[d] = 0 to signal "constant"
        s[d] = 0.0f;
        return;
    }

    // Same symmetric shrink
    float mid  = 0.5f * (mx + mn);
    float half = 0.5f * FRAC_OF_MIN_MAX * range;

    mn = mid - half;
    mx = mid + half;

    dim_min[d] = mn;
    dim_max[d] = mx;

    float trimmed_range = mx - mn;      // FRAC_OF_MIN_MAX * original
    float step = trimmed_range / 4.0f;  // 4 levels for u2
    s[d] = step;
}



//-------------------- Dimension Aware Conversion ---------------


// Packs two 4-bit codes per byte: even dim -> low nibble, odd dim -> high nibble.
// out size per row = (dims + 1) >> 1 bytes.
extern "C" __global__ void f32_to_u4_packed_perdim_kernel(
    const float*  __restrict__ in,       // [num_vecs * dims], original floats
    const float*  __restrict__ dim_min,  // [dims], per-dim min
    const float*  __restrict__ s,        // [dims], per-dim step = range/16 (or 0)
    uint8_t*      __restrict__ out,      // [num_vecs * ((dims+1)>>1)], packed u4
    int num_vecs,
    int dims)
{
    int row_bytes   = (dims + 1) >> 1;          // 2 dims per byte
    int bi          = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bytes = num_vecs * row_bytes;
    if (bi >= total_bytes) return;

    int v  = bi / row_bytes;                    // vector id
    int b  = bi % row_bytes;                    // byte index within row
    int j0 = (b << 1);                          // even dim
    int j1 = j0 + 1;                            // odd dim

    const float* vin = in + (size_t)v * dims;

    // ---- Dim j0 -> low nibble ----
    int q0 = 0;
    if (j0 < dims) {
        float x0  = vin[j0];                    // original value (can be negative)
        float mn0 = dim_min[j0];
        float sj0 = s[j0];                      // step = (max-min)/16 or 0

        if (sj0 <= 0.0f || !isfinite(sj0)) {
            // Degenerate / constant dimension: treat as uninformative, code = 0
            q0 = 0;
        } else {
            float t0 = (x0 - mn0) / sj0;        // in ~[0,16]
            int q0_lin = __float2int_rn(t0);    // 4-bit linear bin
            q0 = max(0, min(15, q0_lin));
        }
    }

    // ---- Dim j1 -> high nibble ----
    int q1 = 0;
    if (j1 < dims) {
        float x1  = vin[j1];
        float mn1 = dim_min[j1];
        float sj1 = s[j1];

        if (sj1 <= 0.0f || !isfinite(sj1)) {
            q1 = 0;
        } else {
            float t1 = (x1 - mn1) / sj1;
            int q1_lin = __float2int_rn(t1);
            q1 = max(0, min(15, q1_lin));
        }
    }

    uint8_t nibble0 = (uint8_t)(q0 & 0x0F);     // low nibble
    uint8_t nibble1 = (uint8_t)(q1 & 0x0F);     // high nibble

    out[(size_t)v * row_bytes + b] = (uint8_t)((nibble1 << 4) | nibble0);
}


// Packs four 2-bit codes per byte: dims j0..j3 -> bits [1:0], [3:2], [5:4], [7:6].
// out size per row = (dims + 3) >> 2 bytes.
extern "C" __global__ void f32_to_u2_packed_perdim_kernel(
    const float*  __restrict__ in,       // [num_vecs * dims], original floats
    const float*  __restrict__ dim_min,  // [dims], per-dim min
    const float*  __restrict__ s,        // [dims], per-dim step = range/4 or 0
    uint8_t*      __restrict__ out,      // [num_vecs * ((dims+3)>>2)], packed u2
    int num_vecs,
    int dims)
{
    int row_bytes   = (dims + 3) >> 2;          // 4 dims per byte
    int bi          = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bytes = num_vecs * row_bytes;
    if (bi >= total_bytes) return;

    int v  = bi / row_bytes;                    // vector id
    int b  = bi % row_bytes;                    // byte index within row
    int j0 = (b << 2);                          // first dim for this byte

    const float* vin = in + (size_t)v * dims;

    uint8_t packed = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int j = j0 + k;
        int q = 0;

        if (j < dims) {
            float x  = vin[j];                  // original value
            float mn = dim_min[j];
            float sj = s[j];                    // step = (max-min)/4 or 0

            if (sj <= 0.0f || !isfinite(sj)) {
                // Degenerate scale: constant dimension -> uninformative
                q = 0;
            } else {
                float t  = (x - mn) / sj;       // ~[0,4]
                int q_lin = __float2int_rn(t);  // 0..3
                q = max(0, min(3, q_lin));
            }
        }

        packed |= (uint8_t)((q & 0x3) << (2 * k));   // 2 bits each
    }

    out[(size_t)v * row_bytes + b] = packed;
}






//----------------- Vector Stats Before Conversion ---------------


extern "C" __global__ void compute_vector_stats_kernel(
    const float* vectors,
    float* norm_l2,
    float* norm_l2_squared,
    int num_vectors,
    const int vector_size
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double norm_sq = 0.0;

    if (i < num_vectors) {
        int idx = i * vector_size;
        for (int j = 0; j < vector_size; j++) {
            double v = vectors[idx + j];
            norm_sq = fmaf(v, v, norm_sq);
        }
        norm_l2_squared[i] = norm_sq;
        norm_l2[i] = sqrt(norm_sq);
    }
}


//----------------- Vector Stats After Conversion ---------------

extern "C" __global__ void compute_vector_stats_u4_packed_kernel(
    const uint8_t* __restrict__ vectors_packed,  // [num_vecs * ((dims+1)>>1)]
    float* __restrict__ norm_l2,                 // [num_vecs]
    float* __restrict__ norm_l2_squared,         // [num_vecs]
    int num_vecs,
    int dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vecs) return;

    const int row_bytes = (dims + 1) >> 1; // 2 dims per byte
    const uint8_t* row = vectors_packed + (size_t)i * row_bytes;

    double acc = 0.0;
    int j = 0;

    // Process full bytes
    for (int by = 0; by < row_bytes; ++by) {
        uint8_t b = row[by];

        // low nibble -> dim j
        if (j < dims) {
            double v = (double)(b & 0x0Fu);
            acc = fma(v, v, acc);
            ++j;
        }

        // high nibble -> dim j
        if (j < dims) {
            double v = (double)(b >> 4);
            acc = fma(v, v, acc);
            ++j;
        }
    }

    float accf = (float)acc;
    norm_l2_squared[i] = accf;
    norm_l2[i]         = sqrtf(accf);
}

extern "C" __global__ void compute_vector_stats_u2_packed_kernel(
    const uint8_t* __restrict__ vectors_packed,  // [num_vecs * ((dims+3)>>2)]
    float* __restrict__ norm_l2,                 // [num_vecs]
    float* __restrict__ norm_l2_squared,         // [num_vecs]
    int num_vecs,
    int dims)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vecs) return;

    const int row_bytes = (dims + 3) >> 2; // 4 dims per byte
    const uint8_t* row = vectors_packed + (size_t)i * row_bytes;

    double acc = 0.0;
    int j = 0;

    for (int by = 0; by < row_bytes; ++by) {
        uint8_t b = row[by];

        // dim j (bits 1:0)
        if (j < dims) {
            double v = (double)((b      ) & 0x3u);
            acc = fma(v, v, acc);
            ++j;
        }
        // dim j+1 (bits 3:2)
        if (j < dims) {
            double v = (double)((b >> 2) & 0x3u);
            acc = fma(v, v, acc);
            ++j;
        }
        // dim j+2 (bits 5:4)
        if (j < dims) {
            double v = (double)((b >> 4) & 0x3u);
            acc = fma(v, v, acc);
            ++j;
        }
        // dim j+3 (bits 7:6)
        if (j < dims) {
            double v = (double)((b >> 6) & 0x3u);
            acc = fma(v, v, acc);
            ++j;
        }
    }

    float accf = (float)acc;
    norm_l2_squared[i] = accf;
    norm_l2[i]         = sqrtf(accf);
}



//
//----------------- Nearest Neighbor Search ---------------------
//

#ifndef KMAX
#define KMAX 64
#endif

__device__ __forceinline__ void topk_try_insert(float d, int i, float* best_d, int* best_i, int K) {
    if (d >= best_d[K-1]) return;
    int pos = K-1;
    while (pos > 0 && d < best_d[pos-1]) {
        best_d[pos] = best_d[pos-1];
        best_i[pos] = best_i[pos-1];
        --pos;
    }
    best_d[pos] = d; best_i[pos] = i;
}



//------------------- 4-BIT bit-sliced -------------------------

extern "C" __global__ void u4_packed_to_bitplanes_rowwise(
    const uint8_t* __restrict__ packed,   // [num_vecs][(D+1)>>1] ; 2 dims/byte (lo nibble, hi nibble)
    unsigned long long* __restrict__ out_b0, // [num_vecs][W] ; bit 0 plane
    unsigned long long* __restrict__ out_b1, // [num_vecs][W] ; bit 1 plane
    unsigned long long* __restrict__ out_b2, // [num_vecs][W] ; bit 2 plane
    unsigned long long* __restrict__ out_b3, // [num_vecs][W] ; bit 3 plane (MSB)
    int num_vecs, int D, int W)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vecs) return;

    const uint8_t* row = packed + (size_t)v * ((D + 1) >> 1);

    for (int w = 0; w < W; ++w) {
        unsigned long long b0 = 0ULL, b1 = 0ULL, b2 = 0ULL, b3 = 0ULL;
        int j_base = w << 6; // 64 dims per 64b word
        #pragma unroll
        for (int t = 0; t < 64; ++t) {
            int j = j_base + t;
            if (j >= D) break;

            int by = j >> 1;                 // 2 dims per byte
            uint8_t code = (j & 1)
                ? (row[by] >> 4) & 0xF       // high nibble
                : (row[by]      ) & 0xF;     // low nibble

            if (code & 0x1) b0 |= (1ULL << t);
            if (code & 0x2) b1 |= (1ULL << t);
            if (code & 0x4) b2 |= (1ULL << t);
            if (code & 0x8) b3 |= (1ULL << t);
        }
        out_b0[(size_t)v * W + w] = b0;
        out_b1[(size_t)v * W + w] = b1;
        out_b2[(size_t)v * W + w] = b2;
        out_b3[(size_t)v * W + w] = b3;
    }
}


// 4-bit bit-sliced Top-K kernel
extern "C" __global__ void find_topk_neighbors_u4_bitsliced_kernel(
    const unsigned long long* __restrict__ q0,   // [M][W]
    const unsigned long long* __restrict__ q1,   // [M][W]
    const unsigned long long* __restrict__ q2,   // [M][W]
    const unsigned long long* __restrict__ q3,   // [M][W]
    const unsigned long long* __restrict__ x0,   // [N][W]
    const unsigned long long* __restrict__ x1,   // [N][W]
    const unsigned long long* __restrict__ x2,   // [N][W]
    const unsigned long long* __restrict__ x3,   // [N][W]
    const float*   __restrict__ norm_l2,               // [N] (bin-space)
    const float*   __restrict__ norm_l2_squared,       // [N] (bin-space)
    int*           __restrict__ topk_indices,          // [M*K]
    //float*         __restrict__ topk_distances,        // [M*K] // Not needed in upper level
    const int K,
    const float max_distance,
    const int   vector_database_len,   // N
    const int   query_vectors_len,     // M
    const int   vector_size,           // D
    const float precomputed_threshold,
    const float* __restrict__ query_norm_l2,           // [M] (bin-space)
    const float* __restrict__ query_norm_l2_squared,   // [M] (bin-space)
    const int   W                                         // words per plane
)
{
    int q = blockIdx.x;
    if (q >= query_vectors_len) return;
    if (K > KMAX) return;

    // shared: per-thread heaps + query planes
    extern __shared__ unsigned char smem[];
    int*   sm_idx  = (int*)smem;
    float* sm_dist = (float*)(sm_idx + blockDim.x * K);
    unsigned long long* sm_q0 = (unsigned long long*)(sm_dist + blockDim.x * K);
    unsigned long long* sm_q1 = sm_q0 + W;
    unsigned long long* sm_q2 = sm_q1 + W;
    unsigned long long* sm_q3 = sm_q2 + W;

    __shared__ float norm_threshold;
    __shared__ float query_norm, query_norm_sq;
    __shared__ unsigned long long tail_mask;

    if (threadIdx.x == 0) {
        norm_threshold = precomputed_threshold;
        query_norm_sq  = query_norm_l2_squared[q];
        query_norm     = query_norm_l2[q];
        int tail_bits  = vector_size & 63;
        tail_mask = (tail_bits == 0) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << tail_bits) - 1ULL);
    }
    __syncthreads();

    // load query bitplanes into shared
    const unsigned long long* Q0 = q0 + (size_t)q * W;
    const unsigned long long* Q1 = q1 + (size_t)q * W;
    const unsigned long long* Q2 = q2 + (size_t)q * W;
    const unsigned long long* Q3 = q3 + (size_t)q * W;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        unsigned long long m = (w == W-1) ? tail_mask : 0xFFFFFFFFFFFFFFFFULL;
        sm_q0[w] = Q0[w] & m;
        sm_q1[w] = Q1[w] & m;
        sm_q2[w] = Q2[w] & m;
        sm_q3[w] = Q3[w] & m;
    }
    __syncthreads();

    // thread-local top-K
    float tk_dist[KMAX];
    int   tk_idx[KMAX];
    #pragma unroll
    for (int t = 0; t < K; ++t) { tk_dist[t] = CUDART_INF_F; tk_idx[t] = -1; }

    // scan DB rows owned by this thread
    for (int i = threadIdx.x; i < vector_database_len; i += blockDim.x) {
        float norm_diff = fabsf(norm_l2[i] - query_norm);
        if (norm_diff > norm_threshold) continue;

        const unsigned long long* X0 = x0 + (size_t)i * W;
        const unsigned long long* X1 = x1 + (size_t)i * W;
        const unsigned long long* X2 = x2 + (size_t)i * W;
        const unsigned long long* X3 = x3 + (size_t)i * W;

        int c00=0,c01=0,c02=0,c03=0,
            c10=0,c11=0,c12=0,c13=0,
            c20=0,c21=0,c22=0,c23=0,
            c30=0,c31=0,c32=0,c33=0;

        #pragma unroll
        for (int w = 0; w < W; ++w) {
            unsigned long long m = (w == W-1) ? tail_mask : 0xFFFFFFFFFFFFFFFFULL;

            unsigned long long q0w = sm_q0[w];
            unsigned long long q1w = sm_q1[w];
            unsigned long long q2w = sm_q2[w];
            unsigned long long q3w = sm_q3[w];

            unsigned long long x0w = X0[w] & m;
            unsigned long long x1w = X1[w] & m;
            unsigned long long x2w = X2[w] & m;
            unsigned long long x3w = X3[w] & m;

            c00 += __popcll(q0w & x0w);
            c01 += __popcll(q0w & x1w);
            c02 += __popcll(q0w & x2w);
            c03 += __popcll(q0w & x3w);

            c10 += __popcll(q1w & x0w);
            c11 += __popcll(q1w & x1w);
            c12 += __popcll(q1w & x2w);
            c13 += __popcll(q1w & x3w);

            c20 += __popcll(q2w & x0w);
            c21 += __popcll(q2w & x1w);
            c22 += __popcll(q2w & x2w);
            c23 += __popcll(q2w & x3w);

            c30 += __popcll(q3w & x0w);
            c31 += __popcll(q3w & x1w);
            c32 += __popcll(q3w & x2w);
            c33 += __popcll(q3w & x3w);
        }

        // dot = Σ_{i=0..3} Σ_{j=0..3} 2^(i+j) * cij
        int dot_i =
              (1  * c00)
            + (2  * (c01 + c10))
            + (4  * (c02 + c20 + c11))
            + (8  * (c03 + c30 + c12 + c21))
            + (16 * (c13 + c31 + c22))
            + (32 * (c23 + c32))
            + (64 *  c33);

        float dot = (float)dot_i;

        float d2 = query_norm_sq + norm_l2_squared[i] - 2.0f * dot;
        d2 = fmaxf(d2, 0.0f);
        topk_try_insert(d2, i, tk_dist, tk_idx, K);
    }

    // spill & merge per-thread candidates
    int base = threadIdx.x * K;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        sm_idx [base + t] = tk_idx[t];
        sm_dist[base + t] = tk_dist[t];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_d[KMAX];
        int   best_i[KMAX];
        #pragma unroll
        for (int t = 0; t < K; ++t) { best_d[t] = CUDART_INF_F; best_i[t] = -1; }

        int Nspill = blockDim.x * K;
        for (int n = 0; n < Nspill; ++n) {
            float d = sm_dist[n];
            int   i = sm_idx[n];
            if (i >= 0 && isfinite(d)) topk_try_insert(d, i, best_d, best_i, K);
        }
        for (int a = 0; a < K-1; ++a)
            for (int b = a+1; b < K; ++b)
                if (best_d[b] < best_d[a]) {
                    float td=best_d[a]; best_d[a]=best_d[b]; best_d[b]=td;
                    int   ti=best_i[a]; best_i[a]=best_i[b]; best_i[b]=ti;
                }

        int out = q * K;
        for (int t = 0; t < K; ++t) {
            topk_indices  [out + t] = best_i[t];
            //topk_distances[out + t] = best_d[t];
        }
    }
}


//------------------- 2-BIT bit-sliced -------------------------

// packed: 4 dims per byte, low→high 2b fields
extern "C" __global__ void u2_packed_to_bitplanes_rowwise(
    const uint8_t*  __restrict__ packed,   // [num_vecs][(D+3)>>2]
    unsigned long long* __restrict__ out_b0, // [num_vecs][W]
    unsigned long long* __restrict__ out_b1, // [num_vecs][W]
    int num_vecs, int D, int W)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vecs) return;

    const uint8_t* row = packed + (size_t)v * ((D + 3) >> 2);
    unsigned long long* b0 = out_b0 + (size_t)v * W;
    unsigned long long* b1 = out_b1 + (size_t)v * W;

    for (int w = 0; w < W; ++w) {
        unsigned long long word0 = 0ULL, word1 = 0ULL;
        int j_base = w << 6; // 64 dims per word
        for (int t = 0; t < 64; ++t) {
            int j = j_base + t;
            if (j >= D) break;
            int by = j >> 2;          // 4 dims per byte
            int off = (j & 3) << 1;   // 2 bits per dim
            uint8_t code = (row[by] >> off) & 0x3;
            if (code & 0x1) word0 |= (1ULL << t);
            if (code & 0x2) word1 |= (1ULL << t);
        }
        b0[w] = word0;
        b1[w] = word1;
    }
}



// Each vector uses two bitplanes:
//  - plane 0 (LSB): B0, plane 1 (MSB): B1
// Layout per set (queries or DB): [num_vecs][words_per_plane],
// where words_per_plane = ceil(vector_size / 64.0).
// For SIFT-128 → words_per_plane = 2.

extern "C" __global__ void find_topk_neighbors_u2_bitsliced_kernel(
    const unsigned long long* __restrict__ query_b0,   // [M][W]
    const unsigned long long* __restrict__ query_b1,   // [M][W]
    const unsigned long long* __restrict__ db_b0,      // [N][W]
    const unsigned long long* __restrict__ db_b1,      // [N][W]
    const float*   __restrict__ norm_l2,               // [N]  (bin-space norms, sqrt)
    const float*   __restrict__ norm_l2_squared,       // [N]  (bin-space norms^2)
    int*           __restrict__ topk_indices,          // [M*K]
    //float*         __restrict__ topk_distances,        // [M*K]
    const int K,
    const float max_distance,
    const int   vector_database_len,   // N
    const int   query_vectors_len,     // M
    const int   vector_size,           // D
    const float precomputed_threshold,
    const float* __restrict__ query_norm_l2,           // [M]   (bin-space)
    const float* __restrict__ query_norm_l2_squared,   // [M]   (bin-space)
    const int   words_per_plane                           // W = (D+63)>>6
)
{
    const int q = blockIdx.x;     // one query per block
    //int q = blockIdx.x;
    if (q >= query_vectors_len) return;
    if (K > KMAX) return; // or assert

    // Shared: per-thread candidate spill + query bitplanes
    extern __shared__ unsigned char smem[];
    int*   sm_idx  = (int*)smem;
    float* sm_dist = (float*)(sm_idx + blockDim.x * K);
    // Place query planes after the per-thread scratch:
    unsigned long long* sm_q0 = (unsigned long long*)(sm_dist + blockDim.x * K);
    unsigned long long* sm_q1 = sm_q0 + words_per_plane;

    __shared__ float norm_threshold;
    __shared__ float query_norm_sq;
    __shared__ float query_norm;
    __shared__ unsigned long long tail_mask;

    if (threadIdx.x == 0) {
        norm_threshold = precomputed_threshold;
        query_norm_sq  = query_norm_l2_squared[q];
        query_norm     = query_norm_l2[q];

        int tail_bits = vector_size & 63;  // D % 64
        tail_mask = (tail_bits == 0) ? 0xFFFFFFFFFFFFFFFFULL
                                     : ((1ULL << tail_bits) - 1ULL);
    }
    __syncthreads();

    // Load query bitplanes into shared (rowwise)
    const unsigned long long* q0 = query_b0 + (size_t)q * words_per_plane;
    const unsigned long long* q1 = query_b1 + (size_t)q * words_per_plane;

    for (int w = threadIdx.x; w < words_per_plane; w += blockDim.x) {
        unsigned long long q0w = q0[w];
        unsigned long long q1w = q1[w];
        // Mask tail word to ensure no stray bits are counted
        if (w == words_per_plane - 1) {
            q0w &= tail_mask;
            q1w &= tail_mask;
        }
        sm_q0[w] = q0w;
        sm_q1[w] = q1w;
    }
    __syncthreads();


    // Thread-local Top-K
    float tk_dist[KMAX];
    int   tk_idx[KMAX];
    #pragma unroll
    for (int t = 0; t < K; ++t) { tk_dist[t] = CUDART_INF_F; tk_idx[t] = -1; }

    // Scan DB rows owned by this thread
    for (int i = threadIdx.x; i < vector_database_len; i += blockDim.x) {

        // -- Rowwise --

        // Norm prefilter in the SAME (bin) space
        float norm_diff = fabsf(norm_l2[i] - query_norm);
        if (norm_diff > norm_threshold) continue;

        const unsigned long long* x0 = db_b0 + (size_t)i * words_per_plane;
        const unsigned long long* x1 = db_b1 + (size_t)i * words_per_plane;

        int c00 = 0, c01 = 0, c10 = 0, c11 = 0;

        // Two 64b words for SIFT-128 (general W supported)
        #pragma unroll
        for (int w = 0; w < words_per_plane; ++w) {
            // Mask tail for the last word
            unsigned long long mask = (w == words_per_plane - 1) ? tail_mask : 0xFFFFFFFFFFFFFFFFULL;

            unsigned long long qw0 = sm_q0[w];
            unsigned long long qw1 = sm_q1[w];
            unsigned long long xw0 = x0[w] & mask;
            unsigned long long xw1 = x1[w] & mask;

            c00 += __popcll(qw0 & xw0);
            c01 += __popcll(qw0 & xw1);
            c10 += __popcll(qw1 & xw0);
            c11 += __popcll(qw1 & xw1);
        }

        // 2-bit dot product in bin space
        int dot_i = c00 + 2 * (c01 + c10) + 4 * c11;
        float dot = (float)dot_i;

        float d2 = query_norm_sq + norm_l2_squared[i] - 2.0f * dot;
        d2 = fmaxf(d2, 0.0f); // robust to tiny underflow

        topk_try_insert(d2, i, tk_dist, tk_idx, K);


    } // end for i

    // Spill per-thread candidates
    int base = threadIdx.x * K;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        sm_idx [base + t] = tk_idx[t];
        sm_dist[base + t] = tk_dist[t];
    }
    __syncthreads();

    // Merge to block top-K
    if (threadIdx.x == 0) {
        float best_d[KMAX];
        int   best_i[KMAX];
        #pragma unroll
        for (int t = 0; t < K; ++t) { best_d[t] = CUDART_INF_F; best_i[t] = -1; }

        int Nspill = blockDim.x * K;
        for (int n = 0; n < Nspill; ++n) {
            float d = sm_dist[n];
            int   i = sm_idx[n];
            if (i >= 0 && isfinite(d)) topk_try_insert(d, i, best_d, best_i, K);
        }

        // Optional small sort for tidy output
        for (int a = 0; a < K-1; ++a)
            for (int b = a+1; b < K; ++b)
                if (best_d[b] < best_d[a]) {
                    float td=best_d[a]; best_d[a]=best_d[b]; best_d[b]=td;
                    int   ti=best_i[a]; best_i[a]=best_i[b]; best_i[b]=ti;
                }

        int out_base = q * K;
        for (int t = 0; t < K; ++t) {
            topk_indices  [out_base + t] = best_i[t];
            //topk_distances[out_base + t] = best_d[t];
        }
    }
}



//-------------------- Rerank Top K --------------------------

extern "C" __global__ void refine_topk_rerank_kernel(
    const float* __restrict__ query_vectors,    // [num_queries * dim]
    const float* __restrict__ db_vectors,       // [db_len * dim]
    const int*   __restrict__ candidates,       // [num_queries * K]
    int*         __restrict__ out_index,        // [num_queries]
    float*       __restrict__ out_distance,     // [num_queries] (squared L2)
    const int num_queries,
    const int dim,
    const int K
)
{
    int q = blockIdx.x;
    if (q >= num_queries) return;

    extern __shared__ unsigned char shared[];
    float* sm_q = reinterpret_cast<float*>(shared);
    float* red  = sm_q + dim;       // reduction buffer, length = blockDim.x

    // Cache query vector into shared memory
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        sm_q[j] = query_vectors[q * dim + j];
    }
    __syncthreads();

    float best_d = FLT_MAX;
    int   best_i = -1;

    // For each candidate, compute exact squared L2 distance in parallel
    for (int t = 0; t < K; ++t) {
        int db_idx = candidates[q * K + t];
        if (db_idx < 0) continue;

        const float* db = &db_vectors[db_idx * dim];

        // Partial sum over dimensions (strided by thread)
        float sum = 0.0f;
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            float diff = sm_q[j] - db[j];
            sum = fmaf(diff, diff, sum);
        }

        // Block-wide reduction into red[0]
        red[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float d = red[0];
            if (d < best_d) { best_d = d; best_i = db_idx; }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_index[q]    = best_i;
        out_distance[q] = best_d;
    }
}
