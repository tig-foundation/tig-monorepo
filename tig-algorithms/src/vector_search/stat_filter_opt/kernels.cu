/*!
Copyright 2025 The Innovation Game

Identity of Submitter OptimusMaximus

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

// tig_challenges::vector_search

#include <float.h>
#include <math_constants.h>   // defines CUDART_INF_F, CUDART_NAN_F, etc.

//#ifndef CLIP_MAX
//#define CLIP_MAX 255.0f
//#endif


//-------------------- Misc Conversion Test ---------------------

extern "C" __global__
void shift_fp32_to_positive(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int n,
    const float shift_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        out[i] = in[i] + shift_val;
    }
}


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
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vecs) return;

    const float* row = db + (size_t)v * dims;
    for (int d = 0; d < dims; ++d) {
        float x = row[d];
        atomicMinFloat(&out_min[d], x);
        atomicMaxFloat(&out_max[d], x);
    }
}



//-------------------- Calculate Dimension Divisors -------------

// Build per-dimension divisors from max.  Scale the max down so
// we throw away outliers.  For example:
//     s[d] = max(FRAC_OF_MAX * max[d] / LEVELS, MIN_STEP)

#ifndef FRAC_OF_MAX
//#define FRAC_OF_MAX 1.00f
#define FRAC_OF_MAX 0.90f
//#define FRAC_OF_MAX 0.80f
#endif

#ifndef MIN_STEP
// This allows us to divide by the result ... no zeros
#define MIN_STEP 1.0f
#endif

extern "C" __global__ void build_u4_divisors_from_max_kernel(
    const float* __restrict__ dim_max, // [dims]
    float* __restrict__ s,             // [dims] (output... pre-allocated)
    int dims,
    float shift_val)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dims) return;

    float mx = fmaxf(0.0f, shift_val + dim_max[d]);      // guard negatives/NaN-ish
    float sd = FRAC_OF_MAX * mx / 16.0f;                 // example: 0.90 * max / 16
    s[d] = fmaxf(sd, MIN_STEP);                          // floor at 1.0
}

extern "C" __global__ void build_u2_divisors_from_max_kernel(
    const float* __restrict__ dim_max, // [dims]
    float* __restrict__ s,             // [dims] (output... pre-allocated)
    int dims)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dims) return;

    float mx = fmaxf(0.0f, dim_max[d]);                  // guard negatives/NaN-ish
    float sd = FRAC_OF_MAX * mx / 4.0f;                  // example: 0.90 * max / 4
    s[d] = fmaxf(sd, MIN_STEP);                          // floor at 1.0
}




//-------------------- Dimension Aware Conversion ---------------



// Packs two 4-bit codes per byte: even dim -> low nibble, odd dim -> high nibble.
// out size per row = (dims + 1) >> 1 bytes.
extern "C" __global__ void f32_to_u4_packed_perdim_kernel(
    const float*  __restrict__ in,   // [num_vecs * dims], original floats
    const float*  __restrict__ s,    // [dims], per-dim divisors (>= 1)
    uint8_t*      __restrict__ out,  // [num_vecs * ((dims+1)>>1)], packed u4
    int num_vecs,
    int dims)
{
    int row_bytes = (dims + 1) >> 1;            // 2 dims per byte
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bytes = num_vecs * row_bytes;
    if (bi >= total_bytes) return;

    int v = bi / row_bytes;                     // vector id
    int b = bi % row_bytes;                     // byte index within row
    int j0 = (b << 1);                          // even dim
    int j1 = j0 + 1;                            // odd dim

    const float* vin = in + (size_t)v * dims;
    const float* ss  = s;

    // Dim j0 -> low nibble
    float x0 = (j0 < dims) ? vin[j0] : 0.0f;
    //float y0 = fminf(fmaxf(x0, 0.0f), CLIP_MAX);
    float y0 = fmaxf(x0, 0.0f);
    float sj0 = ss[j0 < dims ? j0 : 0];         // safe even if j0>=dims
    int   q0  = (y0 <= 0.0f) ? 0 : __float2int_rn(y0 / sj0);
    q0 = max(0, min(15, q0));

    // Dim j1 -> high nibble (or 0 if odd dim does not exist)
    int q1 = 0;
    if (j1 < dims) {
        float x1 = vin[j1];
        //float y1 = fminf(fmaxf(x1, 0.0f), CLIP_MAX);
        float y1 = fmaxf(x1, 0.0f);
        float sj1 = ss[j1];
        q1 = (y1 <= 0.0f) ? 0 : __float2int_rn(y1 / sj1);
        q1 = max(0, min(15, q1));
    }

    out[(size_t)v * row_bytes + b] = (uint8_t)((q1 << 4) | (q0 & 0x0F));
}


// Packs four 2-bit codes per byte: dims j0..j3 -> bits [1:0], [3:2], [5:4], [7:6].
// out size per row = (dims + 3) >> 2 bytes.
extern "C" __global__ void f32_to_u2_packed_perdim_kernel(
    const float*  __restrict__ in,   // [num_vecs * dims], original floats
    const float*  __restrict__ s,    // [dims], per-dim divisors (>= 1)
    uint8_t*      __restrict__ out,  // [num_vecs * ((dims+3)>>2)], packed u2
    int num_vecs,
    int dims)
{
    int row_bytes = (dims + 3) >> 2;            // 4 dims per byte
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    int total_bytes = num_vecs * row_bytes;
    if (bi >= total_bytes) return;

    int v = bi / row_bytes;                     // vector id
    int b = bi % row_bytes;                     // byte index within row
    int j0 = (b << 2);                          // 4 dims starting here

    const float* vin = in + (size_t)v * dims;
    const float* ss  = s;

    uint8_t packed = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int j = j0 + k;
        int q = 0;
        if (j < dims) {
            float x = vin[j];
            //float y = fminf(fmaxf(x, 0.0f), CLIP_MAX);
            float y = fmaxf(x, 0.0f);
            float sj = ss[j];
            q = (y <= 0.0f) ? 0 : __float2int_rn(y / sj);
            q = max(0, min(3, q));
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

// ===============================
// 4-bit packed (two dims / byte)
// ===============================
extern "C" __global__ void find_topk_neighbors_u4_packed_kernel(
    const uint8_t* __restrict__ query_vectors_packed, // [M][(D+1)>>1]
    const uint8_t* __restrict__ vector_database_packed, // [N][(D+1)>>1]
    const float*   __restrict__ norm_l2,              // [N]
    const float*   __restrict__ norm_l2_squared,      // [N]
    int*           __restrict__ topk_indices,         // [M*K]
    float*         __restrict__ topk_distances,       // [M*K]
    const int K,
    const float max_distance,
    const int   vector_database_len,   // N
    const int   query_vectors_len,     // M
    const int   vector_size,           // D
    const float precomputed_threshold,
    const float* __restrict__ query_norm_l2,          // [M]
    const float* __restrict__ query_norm_l2_squared   // [M]
)
{
    int q = blockIdx.x;
    if (q >= query_vectors_len) return;
    if (K > KMAX) return;

    __shared__ float norm_threshold;
    __shared__ float query_norm_sq;
    __shared__ float query_norm;

    extern __shared__ char shared_memory[];
    int*   sm_idx  = (int*)shared_memory;
    float* sm_dist = (float*)(sm_idx + blockDim.x * K);
    uint8_t* sm_qv = (uint8_t*)(sm_dist + blockDim.x * K); // unpacked query codes [D]

    if (threadIdx.x == 0) {
        norm_threshold = precomputed_threshold;
        query_norm_sq  = query_norm_l2_squared[q];
        query_norm     = query_norm_l2[q];
    }

    // --- Unpack query (u4) into sm_qv as bytes (0..15) ---
    int row_bytes = (vector_size + 1) >> 1;
    const uint8_t* qrow = query_vectors_packed + (size_t)q * row_bytes;
    for (int by = threadIdx.x; by < row_bytes; by += blockDim.x) {
        uint8_t b = qrow[by];
        int j0 = by << 1;
        if (j0 < vector_size)     sm_qv[j0] = (uint8_t)(b & 0x0F);
        if (j0 + 1 < vector_size) sm_qv[j0 + 1] = (uint8_t)(b >> 4);
    }
    __syncthreads();

    // --- Thread-local Top-K ---
    float tk_dist[KMAX];
    int   tk_idx[KMAX];
    #pragma unroll
    for (int t = 0; t < K; ++t) { tk_dist[t] = CUDART_INF_F; tk_idx[t] = -1; }

    // --- Scan DB rows owned by this thread ---
    for (int i = threadIdx.x; i < vector_database_len; i += blockDim.x) {
        float norm_diff = fabsf(norm_l2[i] - query_norm);
        if (norm_diff > norm_threshold) continue;

        const uint8_t* drow = vector_database_packed + (size_t)i * row_bytes;


#if __CUDA_ARCH__ >= 610
        // Integer accumulator (exact for 0..15 * 0..15 sums up to D*225)
        int acc_i = 0;

        // Process 4 dims per iteration using 2 packed DB bytes → 4 values
        int j = 0;
        int row_bytes = (vector_size + 1) >> 1;
        int by = 0;

        // Fast path: handle pairs of bytes (covers 4 dims)
        #pragma unroll 16
        //for (; by + 1 < row_bytes && j + 3 < vector_size; by += 2, j += 4) {
        for (; by < row_bytes; by += 2) {  // ONLY if vector_size%4 == 0
            int j = by << 1;          // j = 2*by

            uint8_t b0 = drow[by];
            uint8_t b1 = drow[by + 1];

            // Expand two bytes into four 8-bit codes
            // d0 = low nibble of b0, d1 = high of b0, d2 = low of b1, d3 = high of b1
            int db_pack =
                  (int)( b0        & 0x0F)
                | (int)((b0 >> 4)  & 0x0F) << 8
                | (int)( b1        & 0x0F) << 16
                | (int)((b1 >> 4)  & 0x0F) << 24;

            // Pack query bytes for the same 4 dims
            int q_pack =
                  (int)sm_qv[j]
                | (int)sm_qv[j + 1] << 8
                | (int)sm_qv[j + 2] << 16
                | (int)sm_qv[j + 3] << 24;

            acc_i = __dp4a(q_pack, db_pack, acc_i);
        }

        // Tail (≤3 dims / ≤1 byte)
        if (j < vector_size) {
            uint8_t b = drow[by];
            // dim j
            int d0 = (b & 0x0F);
            acc_i += (int)sm_qv[j] * d0;
            ++j;
            if (j < vector_size) {
                int d1 = (b >> 4);
                acc_i += (int)sm_qv[j] * d1;
                ++j;
            }
            // If vector_size - j == 2 or 3, we'd need one more byte; but since
            // row_bytes = ceil(D/2), this only happens if D%2==0 and we already handled it above.
        }

        float dot = (float)acc_i;
#else
#  error 'unsupported CUDA architecture'
        // ----- Fallback: original scalar FMAs (no dp4a) -----
        float dot = 0.0f;

        // Accumulate dot using packed bytes
        int j = 0;
        #pragma unroll 1
        for (int by = 0; by < ((vector_size + 1) >> 1); ++by) {
            uint8_t b = drow[by];
            if (j < vector_size) { dot = fmaf((float)sm_qv[j], (float)(b & 0x0F), dot); ++j; }
            if (j < vector_size) { dot = fmaf((float)sm_qv[j], (float)(b >> 4),    dot); ++j; }
        }
#endif

        float d2 = query_norm_sq + norm_l2_squared[i] - 2.0f * dot;
        d2 = fmaxf(d2, 0.0f);
        if (max_distance <= 0.0f || d2 <= max_distance) {
            topk_try_insert(d2, i, tk_dist, tk_idx, K);
        }
    }

    // Spill per-thread candidates
    int base = threadIdx.x * K;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        sm_idx[base + t]  = tk_idx[t];
        sm_dist[base + t] = tk_dist[t];
    }
    __syncthreads();

    // Merge to block top-K
    if (threadIdx.x == 0) {
        float best_d[KMAX];
        int   best_i[KMAX];
        #pragma unroll
        for (int t = 0; t < K; ++t) { best_d[t] = CUDART_INF_F; best_i[t] = -1; }
        int N = blockDim.x * K;
        for (int n = 0; n < N; ++n) {
            float d = sm_dist[n];
            int   i = sm_idx[n];
            if (i >= 0 && isfinite(d)) topk_try_insert(d, i, best_d, best_i, K);
        }
        // Stable-ish sort for pretty output (optional)
        for (int a = 0; a < K-1; ++a)
            for (int b = a+1; b < K; ++b)
                if (best_d[b] < best_d[a]) { float td=best_d[a]; best_d[a]=best_d[b]; best_d[b]=td;
                                             int ti=best_i[a]; best_i[a]=best_i[b]; best_i[b]=ti; }
        int out_base = q * K;
        for (int t = 0; t < K; ++t) {
            topk_indices[out_base + t]   = best_i[t];
            topk_distances[out_base + t] = best_d[t];
        }
    }
}

// ===============================
// 2-bit packed (four dims / byte)
// ===============================
extern "C" __global__ void find_topk_neighbors_u2_packed_kernel(
    const uint8_t* __restrict__ query_vectors_packed,   // [M][(D+3)>>2]
    const uint8_t* __restrict__ vector_database_packed, // [N][(D+3)>>2]
    const float*   __restrict__ norm_l2,              // [N]
    const float*   __restrict__ norm_l2_squared,      // [N]
    int*           __restrict__ topk_indices,         // [M*K]
    float*         __restrict__ topk_distances,       // [M*K]
    const int K,
    const float max_distance,
    const int   vector_database_len,   // N
    const int   query_vectors_len,     // M
    const int   vector_size,           // D
    const float precomputed_threshold,
    const float* __restrict__ query_norm_l2,          // [M]
    const float* __restrict__ query_norm_l2_squared   // [M]
)
{
    int q = blockIdx.x;
    if (q >= query_vectors_len) return;
    if (K > KMAX) return;

    __shared__ float norm_threshold;
    __shared__ float query_norm_sq;
    __shared__ float query_norm;

    extern __shared__ char shared_memory[];
    int*   sm_idx  = (int*)shared_memory;
    float* sm_dist = (float*)(sm_idx + blockDim.x * K);
    uint8_t* sm_qv = (uint8_t*)(sm_dist + blockDim.x * K); // unpacked query codes [D]

    if (threadIdx.x == 0) {
        norm_threshold = precomputed_threshold;
        query_norm_sq  = query_norm_l2_squared[q];
        query_norm     = query_norm_l2[q];
    }

    // --- Unpack query (u2) into sm_qv as bytes (0..3) ---
    int row_bytes = (vector_size + 3) >> 2;
    const uint8_t* qrow = query_vectors_packed + (size_t)q * row_bytes;
    for (int by = threadIdx.x; by < row_bytes; by += blockDim.x) {
        uint8_t b = qrow[by];
        int j0 = by << 2;
        if (j0 < vector_size)     sm_qv[j0    ] = (uint8_t)( b        & 0x3);
        if (j0+1 < vector_size)   sm_qv[j0 + 1] = (uint8_t)((b >> 2) & 0x3);
        if (j0+2 < vector_size)   sm_qv[j0 + 2] = (uint8_t)((b >> 4) & 0x3);
        if (j0+3 < vector_size)   sm_qv[j0 + 3] = (uint8_t)((b >> 6) & 0x3);
    }
    __syncthreads();

    // --- Thread-local Top-K ---
    float tk_dist[KMAX];
    int   tk_idx[KMAX];
    #pragma unroll
    for (int t = 0; t < K; ++t) { tk_dist[t] = CUDART_INF_F; tk_idx[t] = -1; }

    // --- Scan DB rows owned by this thread ---
    for (int i = threadIdx.x; i < vector_database_len; i += blockDim.x) {
        float norm_diff = fabsf(norm_l2[i] - query_norm);
        if (norm_diff > norm_threshold) continue;

        const uint8_t* drow = vector_database_packed + (size_t)i * row_bytes;

// == replace ONLY the inner dot loop in your 2-bit kernel ==
#if __CUDA_ARCH__ >= 610
        int acc_i = 0;

        int j = 0;
        int row_bytes = (vector_size + 3) >> 2;

        // Each byte holds 4 codes → perfect for one dp4a per byte
        #pragma unroll 4
        for (int by = 0; by < row_bytes && j < vector_size; ++by, j += 4) {
            uint8_t b = drow[by];

            // Expand 4 two-bit fields into 4 bytes 0..3
            int db_pack =
                  (int)((b      ) & 0x03)
                | (int)((b >>  2) & 0x03) << 8
                | (int)((b >>  4) & 0x03) << 16
                | (int)((b >>  6) & 0x03) << 24;

            // Pack 4 query codes
            // (Bounds-safe: if D not multiple of 4, we may read 1–3 valid here;
            //  the kernel's unpack step already filled sm_qv[missing] with 0.)
            int q_pack =
                  (int)sm_qv[j]
                | (int)sm_qv[j + 1] << 8
                | (int)sm_qv[j + 2] << 16
                | (int)sm_qv[j + 3] << 24;

            acc_i = __dp4a(q_pack, db_pack, acc_i);
        }

        float dot = (float)acc_i;
#else
#  error 'CUDA dev kit too old'
        // ----- Fallback: original scalar FMAs (no dp4a) -----
        float dot = 0.0f;
        // Accumulate dot using packed bytes (4 fields/byte)
        int j = 0;
        #pragma unroll 1
        for (int by = 0; by < ((vector_size + 3) >> 2); ++by) {
            uint8_t b = drow[by];
            if (j < vector_size) { dot = fmaf((float)sm_qv[j], (float)((b      ) & 0x3), dot); ++j; }
            if (j < vector_size) { dot = fmaf((float)sm_qv[j], (float)((b >> 2) & 0x3), dot); ++j; }
            if (j < vector_size) { dot = fmaf((float)sm_qv[j], (float)((b >> 4) & 0x3), dot); ++j; }
            if (j < vector_size) { dot = fmaf((float)sm_qv[j], (float)((b >> 6) & 0x3), dot); ++j; }
        }
#endif



        float d2 = query_norm_sq + norm_l2_squared[i] - 2.0f * dot;
        d2 = fmaxf(d2, 0.0f);
        if (max_distance <= 0.0f || d2 <= max_distance) {
            topk_try_insert(d2, i, tk_dist, tk_idx, K);
        }
    }

    // Spill per-thread candidates
    int base = threadIdx.x * K;
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        sm_idx[base + t]  = tk_idx[t];
        sm_dist[base + t] = tk_dist[t];
    }
    __syncthreads();

    // Merge to block top-K
    if (threadIdx.x == 0) {
        float best_d[KMAX];
        int   best_i[KMAX];
        #pragma unroll
        for (int t = 0; t < K; ++t) { best_d[t] = CUDART_INF_F; best_i[t] = -1; }
        int N = blockDim.x * K;
        for (int n = 0; n < N; ++n) {
            float d = sm_dist[n];
            int   i = sm_idx[n];
            if (i >= 0 && isfinite(d)) topk_try_insert(d, i, best_d, best_i, K);
        }
        for (int a = 0; a < K-1; ++a)
            for (int b = a+1; b < K; ++b)
                if (best_d[b] < best_d[a]) { float td=best_d[a]; best_d[a]=best_d[b]; best_d[b]=td;
                                             int ti=best_i[a]; best_i[a]=best_i[b]; best_i[b]=ti; }
        int out_base = q * K;
        for (int t = 0; t < K; ++t) {
            topk_indices[out_base + t]   = best_i[t];
            topk_distances[out_base + t] = best_d[t];
        }
    }
}


//------------------- 4-BIT bit-sliced -------------------------

extern "C" __global__ void u4_packed_to_bitplanes(
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
    float*         __restrict__ topk_distances,        // [M*K]
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
//        if (max_distance <= 0.0f || d2 <= max_distance) {
            topk_try_insert(d2, i, tk_dist, tk_idx, K);
//        }
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
            topk_distances[out + t] = best_d[t];
        }
    }
}


//------------------- 2-BIT bit-sliced -------------------------

// packed: 4 dims per byte, low→high 2b fields
extern "C" __global__ void u2_packed_to_bitplanes(
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
    float*         __restrict__ topk_distances,        // [M*K]
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
    int q = blockIdx.x;
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

    // Load query bitplanes into shared
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
        // Norm prefilter in the SAME (bin) space
        float norm_diff = fabsf(norm_l2[i] - query_norm);
        if (norm_diff > norm_threshold) continue;

        const unsigned long long* x0 = db_b0 + (size_t)i * words_per_plane;
        const unsigned long long* x1 = db_b1 + (size_t)i * words_per_plane;

        int c00 = 0, c01 = 0, c10 = 0, c11 = 0;

        // Two 64b words for SIFT-128 (general W supported)
        #pragma unroll
        for (int w = 0; w < 4; ++w) {} // hint for better unroll on small W
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

//        if (max_distance <= 0.0f || d2 <= max_distance) {
            topk_try_insert(d2, i, tk_dist, tk_idx, K);
//        }
    }

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
            topk_distances[out_base + t] = best_d[t];
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


