#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define QUERIES_PER_BLOCK 32
#define DB_TILE 64
#define DIMS 250
#define BLOCK_SIZE 512

extern "C" {

__global__ void search_v18(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    int* __restrict__ results,
    float* __restrict__ best_dists,
    int num_queries,
    int database_size
) {
    extern __shared__ float smem[];
    float (*s_query)[DIMS] = (float (*)[DIMS])(smem);
    float (*s_db)[DB_TILE + 1] = (float (*)[DB_TILE + 1])(smem + QUERIES_PER_BLOCK * DIMS);

    const int tid = threadIdx.x;
    const int ty = tid >> 5;
    const int tx = tid & 31;

    const int qA_idx = ty * 2;
    const int qB_idx = ty * 2 + 1;
    const int g_qA = blockIdx.x * QUERIES_PER_BLOCK + qA_idx;
    const int g_qB = blockIdx.x * QUERIES_PER_BLOCK + qB_idx;

    const bool validA = (g_qA < num_queries);
    const bool validB = (g_qB < num_queries);
    const bool valid_any = validA || validB;

    // --- PHASE 1 : Float4 Zero-Fuel Vectorized Query Loader ---
    const float4* q_f4 = reinterpret_cast<const float4*>(query_vectors);
    const size_t max_q_f4 = (size_t)num_queries * 250 / 4;

    int q_base = tid * 4;
    int cur_q = q_base / 250;
    int cur_qd = q_base % 250;

    #pragma unroll 4
    for(int step = 0; step < 4; step++) {
        int idx = step * 512 + tid;
        if (idx < 2000) {
            size_t global_f4_idx = (size_t)blockIdx.x * 2000 + idx;
            if (global_f4_idx < max_q_f4) {
                float4 val = __ldg(&q_f4[global_f4_idx]);
                s_query[cur_q][cur_qd] = val.x;
                int d1 = cur_qd + 1; int q1 = cur_q; if (d1 == 250) { d1 = 0; q1++; }
                s_query[q1][d1] = val.y;
                int d2 = d1 + 1;     int q2 = q1;    if (d2 == 250) { d2 = 0; q2++; }
                s_query[q2][d2] = val.z;
                int d3 = d2 + 1;     int q3 = q2;    if (d3 == 250) { d3 = 0; q3++; }
                s_query[q3][d3] = val.w;
            }
        }
        cur_q += 8;
        cur_qd += 48;
        if (cur_qd >= 250) { cur_qd -= 250; cur_q++; }
    }

    float limitA = validA ? FLT_MAX : -1.0f; int best_idxA = 0;
    float limitB = validB ? FLT_MAX : -1.0f; int best_idxB = 0;

    const int full_chunks = (database_size + DB_TILE - 1) / DB_TILE;
    const float4* db_f4 = reinterpret_cast<const float4*>(database_vectors);
    const size_t max_db_f4 = (size_t)database_size * 250 / 4;

    int db_base = tid * 4;
    int db_local_init = db_base / 250;
    int db_d_init = db_base % 250;
    size_t base_global_f4 = 0;

    // --- PHASE 2 : Master Scanning ---
    #pragma unroll 1
    for (int chunk = 0; chunk < full_chunks; chunk++) {
        const int chunk_start = chunk * DB_TILE;

        int cur_db_local = db_local_init;
        int cur_db_d = db_d_init;

        __syncthreads();

        // --- Float4 Cooperative DB Load ---
        #pragma unroll 8
        for(int step = 0; step < 8; step++) {
            int idx = step * 512 + tid;
            if (idx < 4000) {
                size_t global_f4_idx = base_global_f4 + idx;
                if (global_f4_idx < max_db_f4) {
                    float4 val = __ldg(&db_f4[global_f4_idx]);
                    s_db[cur_db_d][cur_db_local] = val.x;
                    int d1 = cur_db_d + 1; int db1 = cur_db_local; if (d1 == 250) { d1 = 0; db1++; }
                    s_db[d1][db1] = val.y;
                    int d2 = d1 + 1;       int db2 = db1;          if (d2 == 250) { d2 = 0; db2++; }
                    s_db[d2][db2] = val.z;
                    int d3 = d2 + 1;       int db3 = db2;          if (d3 == 250) { d3 = 0; db3++; }
                    s_db[d3][db3] = val.w;
                } else {
                    s_db[cur_db_d][cur_db_local] = 10000.0f;
                    int d1 = cur_db_d + 1; int db1 = cur_db_local; if (d1 == 250) { d1 = 0; db1++; }
                    s_db[d1][db1] = 10000.0f;
                    int d2 = d1 + 1;       int db2 = db1;          if (d2 == 250) { d2 = 0; db2++; }
                    s_db[d2][db2] = 10000.0f;
                    int d3 = d2 + 1;       int db3 = db2;          if (d3 == 250) { d3 = 0; db3++; }
                    s_db[d3][db3] = 10000.0f;
                }
            }
            cur_db_local += 8;
            cur_db_d += 48;
            if (cur_db_d >= 250) { cur_db_d -= 250; cur_db_local++; }
        }

        __syncthreads();

        // --- PHASE 3 : 2x2 Outer Product Compute ---
        if (valid_any) {
            float distA0 = 0.0f, distA1 = 0.0f;
            float distB0 = 0.0f, distB1 = 0.0f;

            #pragma unroll 5
            for (int b = 0; b < 25; b++) {
                const int d_base = b * 10;

                #pragma unroll
                for (int i = 0; i < 10; i++) {
                    const int d = d_base + i;

                    float qa = s_query[qA_idx][d];
                    float qb = s_query[qB_idx][d];

                    float db0 = s_db[d][tx];
                    float db1 = s_db[d][tx + 32];

                    float diffA0 = qa - db0;
                    float diffB0 = qb - db0;
                    float diffA1 = qa - db1;
                    float diffB1 = qb - db1;

                    distA0 = __fmaf_rn(diffA0, diffA0, distA0);
                    distB0 = __fmaf_rn(diffB0, diffB0, distB0);
                    distA1 = __fmaf_rn(diffA1, diffA1, distA1);
                    distB1 = __fmaf_rn(diffB1, diffB1, distB1);
                }

                if (distA0 >= limitA && distA1 >= limitA && distB0 >= limitB && distB1 >= limitB) {
                    break;
                }
            }

            if (chunk_start + tx >= database_size) { distA0 = FLT_MAX; distB0 = FLT_MAX; }
            if (chunk_start + tx + 32 >= database_size) { distA1 = FLT_MAX; distB1 = FLT_MAX; }

            float local_distA; int local_idxA;
            if (distA1 < distA0) { local_distA = distA1; local_idxA = chunk_start + tx + 32; }
            else { local_distA = distA0; local_idxA = chunk_start + tx; }

            float local_distB; int local_idxB;
            if (distB1 < distB0) { local_distB = distB1; local_idxB = chunk_start + tx + 32; }
            else { local_distB = distB0; local_idxB = chunk_start + tx; }

            // --- Warp Reduction Query A ---
            if (validA) {
                unsigned int any_validA = __ballot_sync(0xFFFFFFFF, local_distA < limitA);
                if (any_validA) {
                    float w_dist = local_distA;
                    int w_idx = local_idxA;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        float od = __shfl_down_sync(0xFFFFFFFF, w_dist, off);
                        int oi = __shfl_down_sync(0xFFFFFFFF, w_idx, off);
                        if (od < w_dist) { w_dist = od; w_idx = oi; }
                        else if (od == w_dist && oi < w_idx) { w_idx = oi; }
                    }
                    w_dist = __shfl_sync(0xFFFFFFFF, w_dist, 0);
                    w_idx = __shfl_sync(0xFFFFFFFF, w_idx, 0);
                    if (w_dist < limitA) {
                        limitA = w_dist;
                        best_idxA = w_idx;
                    } else if (w_dist == limitA && w_idx < best_idxA) {
                        best_idxA = w_idx;
                    }
                }
            }

            // --- Warp Reduction Query B ---
            if (validB) {
                unsigned int any_validB = __ballot_sync(0xFFFFFFFF, local_distB < limitB);
                if (any_validB) {
                    float w_dist = local_distB;
                    int w_idx = local_idxB;
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        float od = __shfl_down_sync(0xFFFFFFFF, w_dist, off);
                        int oi = __shfl_down_sync(0xFFFFFFFF, w_idx, off);
                        if (od < w_dist) { w_dist = od; w_idx = oi; }
                        else if (od == w_dist && oi < w_idx) { w_idx = oi; }
                    }
                    w_dist = __shfl_sync(0xFFFFFFFF, w_dist, 0);
                    w_idx = __shfl_sync(0xFFFFFFFF, w_idx, 0);
                    if (w_dist < limitB) {
                        limitB = w_dist;
                        best_idxB = w_idx;
                    } else if (w_dist == limitB && w_idx < best_idxB) {
                        best_idxB = w_idx;
                    }
                }
            }
        }

        base_global_f4 += 4000;
    }

    // --- PHASE 5 : Coalesced Global Write ---
    __syncthreads();

    int* s_results = (int*)smem;
    float* s_dists = (float*)(s_results + QUERIES_PER_BLOCK);

    if (tx == 0) {
        if (validA) {
            s_dists[qA_idx] = limitA;
            s_results[qA_idx] = best_idxA;
        }
        if (validB) {
            s_dists[qB_idx] = limitB;
            s_results[qB_idx] = best_idxB;
        }
    }

    __syncthreads();

    if (ty == 0) {
        int q_idx = blockIdx.x * QUERIES_PER_BLOCK + tx;
        if (q_idx < num_queries) {
            best_dists[q_idx] = s_dists[tx];
            results[q_idx] = s_results[tx];
        }
    }
}

} // extern "C"
