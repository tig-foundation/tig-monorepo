// Copyright (c) 2026 NVX
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define QUERIES_PER_BLOCK 16
#define DB_TILE 32
#define DIMS 250
#define BLOCK_SIZE (QUERIES_PER_BLOCK * WARP_SIZE)

extern "C" {

__global__ void search_v12(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    int* __restrict__ results,
    float* __restrict__ best_dists,
    int num_queries,
    int database_size
) {
    __shared__ float s_query[QUERIES_PER_BLOCK][DIMS];
    __shared__ float s_db[DIMS][DB_TILE + 1];

    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;
    const int tid = threadIdx.x;

    const int q_global = blockIdx.x * QUERIES_PER_BLOCK + ty;
    const bool valid_query = q_global < num_queries;

    // === Load queries into shared memory ===
    // 16 * 250 = 4000 floats, 512 threads, ~8 iterations
    // Stride 512: 512/250 = 2 rem 12 => idx += 2, dim += 12
    {
        int q_idx = tid / DIMS;
        int q_dim = tid - q_idx * DIMS;

        #pragma unroll 1
        for (int i = tid; i < QUERIES_PER_BLOCK * DIMS; i += BLOCK_SIZE) {
            int g_q = blockIdx.x * QUERIES_PER_BLOCK + q_idx;
            if (g_q < num_queries) {
                s_query[q_idx][q_dim] = query_vectors[(size_t)g_q * DIMS + q_dim];
            }
            q_idx += 2;
            q_dim += 12;
            if (q_dim >= DIMS) {
                q_dim -= DIMS;
                q_idx++;
            }
        }
    }

    __syncthreads();

    float my_best_dist = FLT_MAX;
    int my_best_idx = 0;

    // Precompute DB loading initial state
    const int db_init_idx = tid / DIMS;
    const int db_init_dim = tid - db_init_idx * DIMS;

    const int full_chunks = database_size / DB_TILE;
    const int remainder = database_size - full_chunks * DB_TILE;

    // =========================================================
    // FULL CHUNKS — no bounds checking on load or compute
    // =========================================================
    #pragma unroll 1
    for (int chunk = 0; chunk < full_chunks; chunk++) {
        const int chunk_start = chunk * DB_TILE;

        // Load DB tile: 32 * 250 = 8000 floats, 512 threads, ~16 iters
        {
            int cur_idx = db_init_idx;
            int cur_dim = db_init_dim;
            const float* p = database_vectors + (size_t)chunk_start * DIMS + tid;

            #pragma unroll 1
            for (int i = tid; i < DB_TILE * DIMS; i += BLOCK_SIZE) {
                s_db[cur_dim][cur_idx] = *p;
                p += BLOCK_SIZE;
                cur_idx += 2;
                cur_dim += 12;
                if (cur_dim >= DIMS) { cur_dim -= DIMS; cur_idx++; }
            }
        }

        __syncthreads();

        if (valid_query) {
            float dist = 0.0f;
            const float limit = my_best_dist;

            #pragma unroll 25
            for (int b = 0; b < 25; b++) {
                const int d = b * 10;
                float d0 = s_query[ty][d+0] - s_db[d+0][tx];
                float d1 = s_query[ty][d+1] - s_db[d+1][tx];
                float d2 = s_query[ty][d+2] - s_db[d+2][tx];
                float d3 = s_query[ty][d+3] - s_db[d+3][tx];
                float d4 = s_query[ty][d+4] - s_db[d+4][tx];
                float d5 = s_query[ty][d+5] - s_db[d+5][tx];
                float d6 = s_query[ty][d+6] - s_db[d+6][tx];
                float d7 = s_query[ty][d+7] - s_db[d+7][tx];
                float d8 = s_query[ty][d+8] - s_db[d+8][tx];
                float d9 = s_query[ty][d+9] - s_db[d+9][tx];
                dist += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4
                      + d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9;
                if (dist >= limit) { dist = FLT_MAX; break; }
            }

            float w_dist = dist;
            int w_idx = chunk_start + tx;

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float od = __shfl_down_sync(0xFFFFFFFF, w_dist, off);
                int oi = __shfl_down_sync(0xFFFFFFFF, w_idx, off);
                if (od < w_dist) { w_dist = od; w_idx = oi; }
            }

            w_dist = __shfl_sync(0xFFFFFFFF, w_dist, 0);
            w_idx = __shfl_sync(0xFFFFFFFF, w_idx, 0);

            if (w_dist < my_best_dist) {
                my_best_dist = w_dist;
                my_best_idx = w_idx;
            }
        }

        __syncthreads();
    }

    // =========================================================
    // PARTIAL CHUNK (last, with bounds checking)
    // =========================================================
    if (remainder > 0) {
        const int chunk_start = full_chunks * DB_TILE;

        {
            int cur_idx = db_init_idx;
            int cur_dim = db_init_dim;
            const float* p = database_vectors + (size_t)chunk_start * DIMS + tid;

            #pragma unroll 1
            for (int i = tid; i < DB_TILE * DIMS; i += BLOCK_SIZE) {
                if (cur_idx < remainder) {
                    s_db[cur_dim][cur_idx] = *p;
                }
                p += BLOCK_SIZE;
                cur_idx += 2;
                cur_dim += 12;
                if (cur_dim >= DIMS) { cur_dim -= DIMS; cur_idx++; }
            }
        }

        __syncthreads();

        if (valid_query) {
            float dist = FLT_MAX;
            if (tx < remainder) {
                dist = 0.0f;
                const float limit = my_best_dist;

                #pragma unroll 25
                for (int b = 0; b < 25; b++) {
                    const int d = b * 10;
                    float d0 = s_query[ty][d+0] - s_db[d+0][tx];
                    float d1 = s_query[ty][d+1] - s_db[d+1][tx];
                    float d2 = s_query[ty][d+2] - s_db[d+2][tx];
                    float d3 = s_query[ty][d+3] - s_db[d+3][tx];
                    float d4 = s_query[ty][d+4] - s_db[d+4][tx];
                    float d5 = s_query[ty][d+5] - s_db[d+5][tx];
                    float d6 = s_query[ty][d+6] - s_db[d+6][tx];
                    float d7 = s_query[ty][d+7] - s_db[d+7][tx];
                    float d8 = s_query[ty][d+8] - s_db[d+8][tx];
                    float d9 = s_query[ty][d+9] - s_db[d+9][tx];
                    dist += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4
                          + d5*d5 + d6*d6 + d7*d7 + d8*d8 + d9*d9;
                    if (dist >= limit) { dist = FLT_MAX; break; }
                }
            }

            float w_dist = dist;
            int w_idx = chunk_start + tx;

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float od = __shfl_down_sync(0xFFFFFFFF, w_dist, off);
                int oi = __shfl_down_sync(0xFFFFFFFF, w_idx, off);
                if (od < w_dist) { w_dist = od; w_idx = oi; }
            }

            w_dist = __shfl_sync(0xFFFFFFFF, w_dist, 0);
            w_idx = __shfl_sync(0xFFFFFFFF, w_idx, 0);

            if (w_dist < my_best_dist) {
                my_best_dist = w_dist;
                my_best_idx = w_idx;
            }
        }
    }

    // Write results
    if (valid_query && tx == 0) {
        results[q_global] = my_best_idx;
        best_dists[q_global] = my_best_dist;
    }
}

} // extern "C"
