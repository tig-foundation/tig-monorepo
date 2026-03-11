// Copyright (c) 2026 NVX
#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define QUERIES_PER_BLOCK 8
#define DB_VECS_PER_STEP 32
#define DIMS 250
#define MAX_FLOAT FLT_MAX

extern "C" {

__global__ void search_coalesced_v11(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    int* __restrict__ results,
    float* __restrict__ best_dists,
    int num_queries,
    int database_size
) {
    __shared__ float s_query[QUERIES_PER_BLOCK][DIMS];
    __shared__ float s_db[DIMS][DB_VECS_PER_STEP + 1];

    int tx = threadIdx.x % WARP_SIZE;
    int ty = threadIdx.x / WARP_SIZE;
    int tid = threadIdx.x;

    int q_global_idx = blockIdx.x * QUERIES_PER_BLOCK + ty;
    bool valid_query = q_global_idx < num_queries;

    int q_load_idx = tid / DIMS;
    int q_load_dim = tid - q_load_idx * DIMS;

    #pragma unroll 1
    for (int i = tid; i < QUERIES_PER_BLOCK * DIMS; i += 256) {
        int g_q = blockIdx.x * QUERIES_PER_BLOCK + q_load_idx;
        if (g_q < num_queries) {
            s_query[q_load_idx][q_load_dim] = query_vectors[(size_t)g_q * DIMS + q_load_dim];
        }
        q_load_idx += 1;
        q_load_dim += 6;
        if (q_load_dim >= DIMS) {
            q_load_dim -= DIMS;
            q_load_idx += 1;
        }
    }

    __syncthreads();

    float my_best_dist = MAX_FLOAT;
    int my_best_idx = 0;

    int num_chunks = (database_size + DB_VECS_PER_STEP - 1) / DB_VECS_PER_STEP;

    int db_load_idx = tid / DIMS;
    int db_load_dim = tid - db_load_idx * DIMS;

    #pragma unroll 1
    for (int chunk = 0; chunk < num_chunks; chunk++) {

        int chunk_start = chunk * DB_VECS_PER_STEP;
        int vecs_in_chunk = chunk_start + DB_VECS_PER_STEP <= database_size
            ? DB_VECS_PER_STEP : database_size - chunk_start;

        int cur_db_load_idx = db_load_idx;
        int cur_db_load_dim = db_load_dim;

        const float* chunk_ptr = database_vectors + (size_t)chunk_start * DIMS + tid;
        #pragma unroll 1
        for (int i = tid; i < DB_VECS_PER_STEP * DIMS; i += 256) {
            if (cur_db_load_idx < vecs_in_chunk) {
                s_db[cur_db_load_dim][cur_db_load_idx] = *chunk_ptr;
            }
            chunk_ptr += 256;
            cur_db_load_idx += 1;
            cur_db_load_dim += 6;
            if (cur_db_load_dim >= DIMS) {
                cur_db_load_dim -= DIMS;
                cur_db_load_idx += 1;
            }
        }

        __syncthreads();

        float dist = 0.0f;
        float limit = my_best_dist;

        if (valid_query && tx < vecs_in_chunk) {
            #pragma unroll 25
            for (int b = 0; b < 25; b++) {
                int d_base = b * 10;

                float diff0 = s_query[ty][d_base + 0] - s_db[d_base + 0][tx];
                float diff1 = s_query[ty][d_base + 1] - s_db[d_base + 1][tx];
                float diff2 = s_query[ty][d_base + 2] - s_db[d_base + 2][tx];
                float diff3 = s_query[ty][d_base + 3] - s_db[d_base + 3][tx];
                float diff4 = s_query[ty][d_base + 4] - s_db[d_base + 4][tx];
                float diff5 = s_query[ty][d_base + 5] - s_db[d_base + 5][tx];
                float diff6 = s_query[ty][d_base + 6] - s_db[d_base + 6][tx];
                float diff7 = s_query[ty][d_base + 7] - s_db[d_base + 7][tx];
                float diff8 = s_query[ty][d_base + 8] - s_db[d_base + 8][tx];
                float diff9 = s_query[ty][d_base + 9] - s_db[d_base + 9][tx];

                dist += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3 + diff4*diff4
                      + diff5*diff5 + diff6*diff6 + diff7*diff7 + diff8*diff8 + diff9*diff9;

                if (dist >= limit) {
                    dist = MAX_FLOAT;
                    break;
                }
            }
        } else {
            dist = MAX_FLOAT;
        }

        float warp_best_dist = dist;
        int warp_best_idx = chunk_start + tx;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_dist = __shfl_down_sync(0xFFFFFFFF, warp_best_dist, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, warp_best_idx, offset);
            if (other_dist < warp_best_dist) {
                warp_best_dist = other_dist;
                warp_best_idx = other_idx;
            }
        }

        warp_best_dist = __shfl_sync(0xFFFFFFFF, warp_best_dist, 0);
        warp_best_idx = __shfl_sync(0xFFFFFFFF, warp_best_idx, 0);

        if (warp_best_dist < my_best_dist) {
            my_best_dist = warp_best_dist;
            my_best_idx = warp_best_idx;
        }

        __syncthreads();
    }

    if (valid_query && tx == 0) {
        results[q_global_idx] = my_best_idx;
        best_dists[q_global_idx] = my_best_dist;
    }
}

} // extern "C"
