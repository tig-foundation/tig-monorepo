#include <cuda_fp16.h>
#include <float.h>
#include <mma.h>

using namespace nvcuda;

extern "C" __global__ void there_v7_compute_norms_250(
    const float* __restrict__ vectors,
    float* __restrict__ norms,
    int count
) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= count) return;

    const float* __restrict__ v = vectors + (long long)row * 250ll;
    float sum = 0.0f;

    #pragma unroll
    for (int d = 0; d < 250; ++d) {
        float x = v[d];
        sum = fmaf(x, x, sum);
    }

    norms[row] = sum;
}



extern "C" __global__ void there_v9_pack_norm_fp16_250_to_256(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    float* __restrict__ norms,
    int rows,
    int src_stride,
    int dst_stride
) {    
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    const float* __restrict__ src_row = src + (long long)row * src_stride;
    int row_tile = row >> 4;
    int row_in_tile = row & 15;
    unsigned short* __restrict__ dst_base =
        dst + (long long)row_tile * (16ll * dst_stride) + (long long)row_in_tile * 16ll;
    float sum = 0.0f;

    const unsigned long long src_addr = (unsigned long long)src_row;
    const unsigned long long dst_addr = (unsigned long long)dst_base;
    const bool vec16 = ((src_addr & 0xFULL) == 0ull) && ((dst_addr & 0x3ull) == 0ull);
    const bool vec8 = ((src_addr & 0x7ull) == 0ull) && ((dst_addr & 0x3ull) == 0ull);

    const float* __restrict__ s_block = src_row;
    unsigned short* __restrict__ d_block = dst_base;

    if (vec16) {
        #pragma unroll
        for (int kt = 0; kt < 15; ++kt) {
            const float4* __restrict__ s4 = reinterpret_cast<const float4*>(s_block);
            float4 v0 = s4[0];
            float4 v1 = s4[1];
            float4 v2 = s4[2];
            float4 v3 = s4[3];

            sum = fmaf(v0.x, v0.x, sum);
            sum = fmaf(v0.y, v0.y, sum);
            sum = fmaf(v0.z, v0.z, sum);
            sum = fmaf(v0.w, v0.w, sum);
            sum = fmaf(v1.x, v1.x, sum);
            sum = fmaf(v1.y, v1.y, sum);
            sum = fmaf(v1.z, v1.z, sum);
            sum = fmaf(v1.w, v1.w, sum);
            sum = fmaf(v2.x, v2.x, sum);
            sum = fmaf(v2.y, v2.y, sum);
            sum = fmaf(v2.z, v2.z, sum);
            sum = fmaf(v2.w, v2.w, sum);
            sum = fmaf(v3.x, v3.x, sum);
            sum = fmaf(v3.y, v3.y, sum);
            sum = fmaf(v3.z, v3.z, sum);
            sum = fmaf(v3.w, v3.w, sum);

            half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);
            d2[0] = __floats2half2_rn(v0.x, v0.y);
            d2[1] = __floats2half2_rn(v0.z, v0.w);
            d2[2] = __floats2half2_rn(v1.x, v1.y);
            d2[3] = __floats2half2_rn(v1.z, v1.w);
            d2[4] = __floats2half2_rn(v2.x, v2.y);
            d2[5] = __floats2half2_rn(v2.z, v2.w);
            d2[6] = __floats2half2_rn(v3.x, v3.y);
            d2[7] = __floats2half2_rn(v3.z, v3.w);

            s_block += 16;
            d_block += 256;
        }

        const float4* __restrict__ s4 = reinterpret_cast<const float4*>(s_block);
        float4 v0 = s4[0];
        float4 v1 = s4[1];
        float2 v2 = *reinterpret_cast<const float2*>(s_block + 8);

        sum = fmaf(v0.x, v0.x, sum);
        sum = fmaf(v0.y, v0.y, sum);
        sum = fmaf(v0.z, v0.z, sum);
        sum = fmaf(v0.w, v0.w, sum);
        sum = fmaf(v1.x, v1.x, sum);
        sum = fmaf(v1.y, v1.y, sum);
        sum = fmaf(v1.z, v1.z, sum);
        sum = fmaf(v1.w, v1.w, sum);
        sum = fmaf(v2.x, v2.x, sum);
        sum = fmaf(v2.y, v2.y, sum);

        half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);
        d2[0] = __floats2half2_rn(v0.x, v0.y);
        d2[1] = __floats2half2_rn(v0.z, v0.w);
        d2[2] = __floats2half2_rn(v1.x, v1.y);
        d2[3] = __floats2half2_rn(v1.z, v1.w);
        d2[4] = __floats2half2_rn(v2.x, v2.y);
    } else if (vec8) {
        #pragma unroll
        for (int kt = 0; kt < 15; ++kt) {
            const float2* __restrict__ s2 = reinterpret_cast<const float2*>(s_block);
            half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);

            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 v = s2[p];
                sum = fmaf(v.x, v.x, sum);
                sum = fmaf(v.y, v.y, sum);
                d2[p] = __floats2half2_rn(v.x, v.y);
            }

            s_block += 16;
            d_block += 256;
        }

        const float2* __restrict__ s2 = reinterpret_cast<const float2*>(s_block);
        half2* __restrict__ d2 = reinterpret_cast<half2*>(d_block);

        #pragma unroll
        for (int p = 0; p < 5; ++p) {
            float2 v = s2[p];
            sum = fmaf(v.x, v.x, sum);
            sum = fmaf(v.y, v.y, sum);
            d2[p] = __floats2half2_rn(v.x, v.y);
        }
    } else {
        #pragma unroll
        for (int kt = 0; kt < 15; ++kt) {
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float x = s_block[c];
                sum = fmaf(x, x, sum);
                d_block[c] = __half_as_ushort(__float2half(x));
            }
            s_block += 16;
            d_block += 256;
        }

        #pragma unroll
        for (int c = 0; c < 10; ++c) {
            float x = s_block[c];
            sum = fmaf(x, x, sum);
            d_block[c] = __half_as_ushort(__float2half(x));
        }
    }

    norms[row] = sum;
}

extern "C" __global__ void there_v7_pack_fp16_linear(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int n
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = __half_as_ushort(__float2half(src[i]));
    }
}

__device__ __forceinline__ void there_v10_update_best_f32(
    float dist,
    int idx,
    float& best,
    int& best_idx
) {
    if (dist < best || (dist == best && idx < best_idx)) {
        best = dist;
        best_idx = idx;
    }
}

__device__ __forceinline__ void there_v10_warp_reduce_best_f32(
    float& best,
    int& best_idx
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFFu, best, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFFu, best_idx, offset);
        if (other_dist < best || (other_dist == best && other_idx < best_idx)) {
            best = other_dist;
            best_idx = other_idx;
        }
    }
}

template <bool LOAD_CURRENT>
__device__ __forceinline__ void there_v10_reduce_gemm_tile_f32_full_impl(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const float* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int db_start,
    int num_queries
) {
    constexpr int FULL_TILE = 4096;
    constexpr int VEC_GROUPS = FULL_TILE / 4;

    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;

    float query_norm = lane == 0 ? query_norms[q] : 0.0f;
    query_norm = __shfl_sync(0xFFFFFFFFu, query_norm, 0);

    const float4* __restrict__ row4 =
        reinterpret_cast<const float4*>(dot_tile + (long long)q * FULL_TILE);
    const float4* __restrict__ db4 =
        reinterpret_cast<const float4*>(database_norms + db_start);

    float local_best = FLT_MAX;
    int local_idx = -1;

    #pragma unroll
    for (int it = 0, v = lane; it < VEC_GROUPS / 32; ++it, v += 32) {
        int j = v << 2;
        float4 dots = row4[v];
        float4 norms = db4[v];

        int idx = db_start + j;
        there_v10_update_best_f32(query_norm + norms.x + dots.x, idx, local_best, local_idx);

        ++idx;
        there_v10_update_best_f32(query_norm + norms.y + dots.y, idx, local_best, local_idx);

        ++idx;
        there_v10_update_best_f32(query_norm + norms.z + dots.z, idx, local_best, local_idx);

        ++idx;
        there_v10_update_best_f32(query_norm + norms.w + dots.w, idx, local_best, local_idx);
    }

    there_v10_warp_reduce_best_f32(local_best, local_idx);

    if (lane == 0) {
        if (LOAD_CURRENT) {
            float current_best = best_dist[q];
            int current_idx = best_idx[q];
            if (local_best < current_best ||
                (local_best == current_best && (current_idx < 0 || local_idx < current_idx))) {
                best_dist[q] = local_best;
                best_idx[q] = local_idx;
            }
        } else {
            best_dist[q] = local_best;
            best_idx[q] = local_idx;
        }
    }
}

extern "C" __global__ void there_v10_reduce_gemm_tile_f32_full_first(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const float* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int db_start,
    int num_queries
) {
    there_v10_reduce_gemm_tile_f32_full_impl<false>(
        query_norms,
        database_norms,
        dot_tile,
        best_idx,
        best_dist,
        db_start,
        num_queries
    );
}

extern "C" __global__ void there_v10_reduce_gemm_tile_f32_full_next(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const float* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int db_start,
    int num_queries
) {
    there_v10_reduce_gemm_tile_f32_full_impl<true>(
        query_norms,
        database_norms,
        dot_tile,
        best_idx,
        best_dist,
        db_start,
        num_queries
    );
}

extern "C" __global__ void there_v7_reduce_gemm_tile_f32(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const float* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int db_start,
    int tile_len,
    int num_queries,
    int first_tile
) {
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;

    float query_norm = lane == 0 ? query_norms[q] : 0.0f;
    query_norm = __shfl_sync(0xFFFFFFFFu, query_norm, 0);

    float current_best = FLT_MAX;
    int current_idx = -1;
    if (!first_tile && lane == 0) {
        current_best = best_dist[q];
        current_idx = best_idx[q];
    }
    current_best = __shfl_sync(0xFFFFFFFFu, current_best, 0);
    current_idx = __shfl_sync(0xFFFFFFFFu, current_idx, 0);

    float local_best = FLT_MAX;
    int local_idx = -1;
    const float* __restrict__ row = dot_tile + (long long)q * tile_len;
    const float* __restrict__ db_norm_row = database_norms + db_start;

    const unsigned long long row_addr = reinterpret_cast<unsigned long long>(row);
    const unsigned long long db_addr = reinterpret_cast<unsigned long long>(db_norm_row);
    const bool vec_path = ((row_addr & 0xFULL) == 0ull) && ((db_addr & 0xFULL) == 0ull);
    const int vec_groups = tile_len >> 2;
    const int vec_end = vec_groups << 2;

    if (vec_path) {
        const float4* __restrict__ row4 = reinterpret_cast<const float4*>(row);
        const float4* __restrict__ db4 = reinterpret_cast<const float4*>(db_norm_row);

        for (int v = lane; v < vec_groups; v += 32) {
            int j = v << 2;
            float4 dots = row4[v];
            float4 norms = db4[v];

            int idx = db_start + j;
            float dist = query_norm + db_norm_row[j] + row[j];
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }

            ++idx;
            dist = query_norm + norms.y + dots.y;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }

            ++idx;
            dist = query_norm + norms.z + dots.z;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }

            ++idx;
            dist = query_norm + norms.w + dots.w;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
        }

        for (int j = vec_end + lane; j < tile_len; j += 32) {
            float dist = query_norm + db_norm_row[j] + row[j];
            int idx = db_start + j;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
        }
    } else {
        for (int j = lane; j < tile_len; j += 32) {
            float dist = query_norm + db_norm_row[j] + row[j];
            int idx = db_start + j;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFFu, local_best, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFFu, local_idx, offset);
        if (other_dist < local_best || (other_dist == local_best && other_idx < local_idx)) {
            local_best = other_dist;
            local_idx = other_idx;
        }
    }

    if (lane == 0 && (local_best < current_best ||
        (local_best == current_best && (current_idx < 0 || local_idx < current_idx)))) {
        best_dist[q] = local_best;
        best_idx[q] = local_idx;
    }
}

extern "C" __global__ void there_v7_reduce_gemm_tile_fp16(
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    const unsigned short* __restrict__ dot_tile,
    int* __restrict__ best_idx,
    float* __restrict__ best_dist,
    int q_base,
    int db_start,
    int tile_len,
    int num_queries,
    int first_tile
) {    
    int q = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (q >= num_queries) return;

    int lane = (int)threadIdx.x;
    int q_idx = q_base + q;

    float query_norm = lane == 0 ? query_norms[q_idx] : 0.0f;
    query_norm = __shfl_sync(0xFFFFFFFFu, query_norm, 0);

    float current_best = FLT_MAX;
    int current_idx = -1;
    if (!first_tile && lane == 0) {
        current_best = best_dist[q_idx];
        current_idx = best_idx[q_idx];
    }
    current_best = __shfl_sync(0xFFFFFFFFu, current_best, 0);
    current_idx = __shfl_sync(0xFFFFFFFFu, current_idx, 0);

    float local_best = FLT_MAX;
    int local_idx = -1;
    const unsigned short* __restrict__ row = dot_tile + (long long)q * tile_len;
    const float* __restrict__ db_norm_row = database_norms + db_start;
    const half* __restrict__ row_h = reinterpret_cast<const half*>(row);

    const int vec_end = tile_len & ~1;
    const bool paired_path = ((reinterpret_cast<unsigned long long>(row_h) & 0x3ull) == 0ull);

    if (paired_path) {
        for (int j = lane << 1; j < vec_end; j += 64) {
            half2 dots_h2 = *reinterpret_cast<const half2*>(row_h + j);
            float2 dots = __half22float2(dots_h2);

            int idx0 = db_start + j;
            float dist0 = query_norm + db_norm_row[j] + dots.x;
            if (dist0 < local_best || (dist0 == local_best && idx0 < local_idx)) {
                local_best = dist0;
                local_idx = idx0;
            }

            int idx1 = idx0 + 1;
            float dist1 = query_norm + db_norm_row[j + 1] + dots.y;
            if (dist1 < local_best || (dist1 == local_best && idx1 < local_idx)) {
                local_best = dist1;
                local_idx = idx1;
            }
        }

        int j = vec_end + lane;
        if (j < tile_len) {
            float dot = __half2float(row_h[j]);
            float dist = query_norm + db_norm_row[j] + dot;
            int idx = db_start + j;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
        }
    } else {
        for (int j = lane; j < tile_len; j += 32) {
            float dot = __half2float(__ushort_as_half(row[j]));
            float dist = query_norm + db_norm_row[j] + dot;
            int idx = db_start + j;
            if (dist < local_best || (dist == local_best && idx < local_idx)) {
                local_best = dist;
                local_idx = idx;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFFu, local_best, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFFu, local_idx, offset);
        if (other_dist < local_best || (other_dist == local_best && other_idx < local_idx)) {
            local_best = other_dist;
            local_idx = other_idx;
        }
    }

    if (lane == 0 && (local_best < current_best ||
        (local_best == current_best && (current_idx < 0 || local_idx < current_idx)))) {
        best_dist[q_idx] = local_best;
        best_idx[q_idx] = local_idx;
    }
}

extern "C" __global__ void there_v9_wmma_chunk_best(
    const unsigned short* __restrict__ q_half_u16,
    const unsigned short* __restrict__ db_half_u16,
    const float* __restrict__ query_norms,
    const float* __restrict__ database_norms,
    float* __restrict__ chunk_best_dist,
    int* __restrict__ chunk_best_idx,
    int num_queries,
    int database_size,
    int query_stride,
    int chunk_db
) {
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int WK = 16;
    constexpr int DPAD = 256;
    constexpr int KTILES = DPAD / WK;
    constexpr int WARP_TILES = 4;
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

    __shared__ float dot_tiles[WARP_TILES][WM * WN];
    __shared__ __align__(16) int4 db_tile_vec[TILE_VEC4];
    __shared__ float db_norm_tile[WN];
    float* __restrict__ dots = dot_tiles[warp_id];

    const half* __restrict__ q_half = reinterpret_cast<const half*>(q_half_u16);
    const half* __restrict__ db_half = reinterpret_cast<const half*>(db_half_u16);

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a[KTILES];
    if (warp_active) {
        const half* __restrict__ q_tile_base = q_half + (long long)q_tile_id * PACKED_TILE_STRIDE;
        #pragma unroll
        for (int k = 0; k < KTILES; ++k) {
            wmma::load_matrix_sync(a[k], q_tile_base + (long long)k * TILE_ELEMS, WN);
        }
    }

    float best_dist = FLT_MAX;
    int best_idx = -1;
    float q_norm = (warp_active && lane < WM && q < num_queries) ? query_norms[q] : 0.0f;

    int db_tile_start = chunk_start >> 4;
    int db_tiles = (chunk_len + WN - 1) / WN;

    for (int tile = 0; tile < db_tiles; ++tile) {
        int db_local = tile * WN;
        int valid = chunk_len - db_local;
        if (valid > WN) valid = WN;

        const half* __restrict__ db_tile_base =
            db_half + (long long)(db_tile_start + tile) * PACKED_TILE_STRIDE;
        const int4* __restrict__ db_tile_base_vec =
            reinterpret_cast<const int4*>(db_tile_base);

        #pragma unroll
        for (int i = tid; i < TILE_VEC4; i += THREADS_PER_BLOCK) {
            db_tile_vec[i] = db_tile_base_vec[i];
        }
        if (tid < WN) {
            db_norm_tile[tid] = (tid < valid) ? database_norms[chunk_start + db_local + tid] : 0.0f;
        }
        __syncthreads();

        if (warp_active) {
            const half* __restrict__ db_tile_shared =
                reinterpret_cast<const half*>(db_tile_vec);

            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k = 0; k < KTILES; ++k) {
                wmma::load_matrix_sync(b, db_tile_shared + (long long)k * TILE_ELEMS, WN);
                wmma::mma_sync(acc, a[k], b, acc);
            }

            wmma::store_matrix_sync(dots, acc, WN, wmma::mem_row_major);
        }
        __syncthreads();

        if (warp_active && lane < WM && q < num_queries) {
            const float* __restrict__ dot_row = dots + lane * WN;
            #pragma unroll
            for (int j = 0; j < WN; ++j) {
                if (j < valid) {
                    int db_idx = chunk_start + db_local + j;
                    float dist = q_norm + db_norm_tile[j] - 2.0f * dot_row[j];
                    if (dist < best_dist || (dist == best_dist && db_idx < best_idx)) {
                        best_dist = dist;
                        best_idx = db_idx;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (warp_active && lane < WM && q < num_queries) {
        long long out = (long long)chunk_id * query_stride + q;
        chunk_best_dist[out] = best_dist;
        chunk_best_idx[out] = best_idx;
    }
}

extern "C" __global__ void there_v9_reduce_chunk_bests(
    const float* __restrict__ chunk_best_dist,
    const int* __restrict__ chunk_best_idx,
    float* __restrict__ best_dist,
    int* __restrict__ best_idx,
    int num_queries,
    int num_chunks,
    int query_stride
) {
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= num_queries) return;

    float best = FLT_MAX;
    int idx = -1;
    for (int c = 0; c < num_chunks; ++c) {
        long long off = (long long)c * query_stride + q;
        float cand = chunk_best_dist[off];
        int cand_idx = chunk_best_idx[off];
        if (cand_idx >= 0 && (cand < best || (cand == best && (idx < 0 || cand_idx < idx)))) {
            best = cand;
            idx = cand_idx;
        }
    }

    best_dist[q] = best;
    best_idx[q] = idx;
}
