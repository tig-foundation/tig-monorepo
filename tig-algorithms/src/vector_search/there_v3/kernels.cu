
#ifndef MAX_FLOAT
#define MAX_FLOAT 3.402823466e+38f
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void cp_async4_predicated(void* smem_dst, const void* gmem_src, int pred) {
    unsigned int dst = __cvta_generic_to_shared(smem_dst);
    unsigned long long src = (unsigned long long)gmem_src;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], 4, p;\n"
        "}\n"
        :
        : "r"(dst), "l"(src), "r"(pred)
    );
}
#endif

__device__ __forceinline__ float euclidean_distance_high_bounded_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    int dims,
    float limit
) {
    float sum = 0.0f;
    int i = 0;
    const float* bp = b_lane;

    if (dims >= 4) {
        float b0 = *bp; bp += 16;
        float b1 = *bp; bp += 16;
        float b2 = *bp; bp += 16;
        float b3 = *bp; bp += 16;

        float d0 = a[0] - b0;
        float d1 = a[1] - b1;
        float d2 = a[2] - b2;
        float d3 = a[3] - b3;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, d3 * d3)));
        if (sum > limit) return sum;
        i = 4;
    }

    for (; i < dims - 31; i += 32) {
        const float* ap = a + i;

        float b0 = *bp; bp += 16;
        float b1 = *bp; bp += 16;
        float b2 = *bp; bp += 16;
        float b3 = *bp; bp += 16;
        float b4 = *bp; bp += 16;
        float b5 = *bp; bp += 16;
        float b6 = *bp; bp += 16;
        float b7 = *bp; bp += 16;
        float b8 = *bp; bp += 16;
        float b9 = *bp; bp += 16;
        float b10 = *bp; bp += 16;
        float b11 = *bp; bp += 16;
        float b12 = *bp; bp += 16;
        float b13 = *bp; bp += 16;
        float b14 = *bp; bp += 16;
        float b15 = *bp; bp += 16;

        float d0 = ap[0] - b0;
        float d1 = ap[1] - b1;
        float d2 = ap[2] - b2;
        float d3 = ap[3] - b3;
        float d4 = ap[4] - b4;
        float d5 = ap[5] - b5;
        float d6 = ap[6] - b6;
        float d7 = ap[7] - b7;
        float d8 = ap[8] - b8;
        float d9 = ap[9] - b9;
        float d10 = ap[10] - b10;
        float d11 = ap[11] - b11;
        float d12 = ap[12] - b12;
        float d13 = ap[13] - b13;
        float d14 = ap[14] - b14;
        float d15 = ap[15] - b15;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
              fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7,
              fmaf(d8, d8, fmaf(d9, d9, fmaf(d10, d10, fmaf(d11, d11,
              fmaf(d12, d12, fmaf(d13, d13, fmaf(d14, d14, fmaf(d15, d15, sum))))))))))))))));
        if (sum > limit) return sum;

        float b16 = *bp; bp += 16;
        float b17 = *bp; bp += 16;
        float b18 = *bp; bp += 16;
        float b19 = *bp; bp += 16;
        float b20 = *bp; bp += 16;
        float b21 = *bp; bp += 16;
        float b22 = *bp; bp += 16;
        float b23 = *bp; bp += 16;
        float b24 = *bp; bp += 16;
        float b25 = *bp; bp += 16;
        float b26 = *bp; bp += 16;
        float b27 = *bp; bp += 16;
        float b28 = *bp; bp += 16;
        float b29 = *bp; bp += 16;
        float b30 = *bp; bp += 16;
        float b31 = *bp; bp += 16;

        float e16 = ap[16] - b16;
        float e17 = ap[17] - b17;
        float e18 = ap[18] - b18;
        float e19 = ap[19] - b19;
        float e20 = ap[20] - b20;
        float e21 = ap[21] - b21;
        float e22 = ap[22] - b22;
        float e23 = ap[23] - b23;
        float e24 = ap[24] - b24;
        float e25 = ap[25] - b25;
        float e26 = ap[26] - b26;
        float e27 = ap[27] - b27;
        float e28 = ap[28] - b28;
        float e29 = ap[29] - b29;
        float e30 = ap[30] - b30;
        float e31 = ap[31] - b31;

        sum = fmaf(e16, e16, fmaf(e17, e17, fmaf(e18, e18, fmaf(e19, e19,
              fmaf(e20, e20, fmaf(e21, e21, fmaf(e22, e22, fmaf(e23, e23,
              fmaf(e24, e24, fmaf(e25, e25, fmaf(e26, e26, fmaf(e27, e27,
              fmaf(e28, e28, fmaf(e29, e29, fmaf(e30, e30, fmaf(e31, e31, sum))))))))))))))));
        if (sum > limit) return sum;
    }

    for (; i < dims - 15; i += 16) {
        const float* ap = a + i;

        float b0 = *bp; bp += 16;
        float b1 = *bp; bp += 16;
        float b2 = *bp; bp += 16;
        float b3 = *bp; bp += 16;
        float b4 = *bp; bp += 16;
        float b5 = *bp; bp += 16;
        float b6 = *bp; bp += 16;
        float b7 = *bp; bp += 16;
        float b8 = *bp; bp += 16;
        float b9 = *bp; bp += 16;
        float b10 = *bp; bp += 16;
        float b11 = *bp; bp += 16;
        float b12 = *bp; bp += 16;
        float b13 = *bp; bp += 16;
        float b14 = *bp; bp += 16;
        float b15 = *bp; bp += 16;

        float d0 = ap[0] - b0;
        float d1 = ap[1] - b1;
        float d2 = ap[2] - b2;
        float d3 = ap[3] - b3;
        float d4 = ap[4] - b4;
        float d5 = ap[5] - b5;
        float d6 = ap[6] - b6;
        float d7 = ap[7] - b7;
        float d8 = ap[8] - b8;
        float d9 = ap[9] - b9;
        float d10 = ap[10] - b10;
        float d11 = ap[11] - b11;
        float d12 = ap[12] - b12;
        float d13 = ap[13] - b13;
        float d14 = ap[14] - b14;
        float d15 = ap[15] - b15;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
              fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7,
              fmaf(d8, d8, fmaf(d9, d9, fmaf(d10, d10, fmaf(d11, d11,
              fmaf(d12, d12, fmaf(d13, d13, fmaf(d14, d14, fmaf(d15, d15, sum))))))))))))))));
        if (sum > limit) return sum;
    }

    for (; i < dims - 7; i += 8) {
        const float* ap = a + i;

        float b0 = *bp; bp += 16;
        float b1 = *bp; bp += 16;
        float b2 = *bp; bp += 16;
        float b3 = *bp; bp += 16;
        float b4 = *bp; bp += 16;
        float b5 = *bp; bp += 16;
        float b6 = *bp; bp += 16;
        float b7 = *bp; bp += 16;

        float d0 = ap[0] - b0;
        float d1 = ap[1] - b1;
        float d2 = ap[2] - b2;
        float d3 = ap[3] - b3;
        float d4 = ap[4] - b4;
        float d5 = ap[5] - b5;
        float d6 = ap[6] - b6;
        float d7 = ap[7] - b7;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
              fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7, sum))))))));
        if (sum > limit) return sum;
    }

    for (; i < dims - 3; i += 4) {
        const float* ap = a + i;

        float b0 = *bp; bp += 16;
        float b1 = *bp; bp += 16;
        float b2 = *bp; bp += 16;
        float b3 = *bp; bp += 16;

        float d0 = ap[0] - b0;
        float d1 = ap[1] - b1;
        float d2 = ap[2] - b2;
        float d3 = ap[3] - b3;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3, sum))));
        if (sum > limit) return sum;
    }

    for (; i < dims; i++) {
        float diff = a[i] - *bp;
        bp += 16;
        sum = fmaf(diff, diff, sum);
        if (sum > limit) return sum;
    }

    return sum;
}

extern "C" __global__ void transform_database_blocked(
    const float* __restrict__ src_database_vectors,
    float* __restrict__ dst_database_blocked,
    int database_size,
    int vector_dims
) {
    int db_block = (int)blockIdx.x;  
    int lane16   = (int)(threadIdx.x & 15);
    int d0       = (int)(threadIdx.x >> 4);

    int src_idx = db_block * 16 + lane16;
    const bool valid = (src_idx < database_size);

    const float* __restrict__ src_base = valid
        ? (src_database_vectors + (size_t)src_idx * (size_t)vector_dims)
        : (const float*)0;

    if (valid) {
        for (int d = d0; d < vector_dims; d += 16) {
            float v = src_base[d];
            dst_database_blocked[((size_t)db_block * (size_t)vector_dims + (size_t)d) * 16ull + (size_t)lane16] = v;
        }
    } else {
        for (int d = d0; d < vector_dims; d += 16) {
            dst_database_blocked[((size_t)db_block * (size_t)vector_dims + (size_t)d) * 16ull + (size_t)lane16] = 0.0f;
        }
    }
}

extern "C" __global__ void init_best_dists(
    float* __restrict__ best_dists,
    int num_queries
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries) {
        best_dists[idx] = MAX_FLOAT;
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void batched_search(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    int* __restrict__ results,
    float* __restrict__ best_dists,
    int num_queries,
    int vector_dims,
    int batch_start,
    int batch_count
) {
    extern __shared__ float s_queries[];
    __shared__ unsigned int s_bounds_u32[16];

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int half = lane >> 4;
    int lane16 = lane & 15;

    int local_query = warp_id * 2 + half;
    int query_idx = (int)(blockIdx.x * 16 + local_query);

    if (batch_count <= 0) return;

    int smem_size = 16 * vector_dims;
    bool use_smem = (smem_size > 0 && smem_size <= 12288);

    if (use_smem) {
        int block_query_base = (int)(blockIdx.x * 16);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        for (int q_local = 0; q_local < 16; ++q_local) {
            int q_global = block_query_base + q_local;
            int valid_q = (q_global < num_queries) ? 1 : 0;
            size_t q_off = valid_q ? ((size_t)q_global * (size_t)vector_dims) : 0ull;
            const float* q_src_base = query_vectors + q_off;

            for (int d = threadIdx.x; d < vector_dims; d += 256) {
                cp_async4_predicated(
                    (void*)(s_queries + (size_t)q_local * (size_t)vector_dims + (size_t)d),
                    (const void*)(q_src_base + (size_t)d),
                    valid_q
                );
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
#else
        for (int q_local = 0; q_local < 16; ++q_local) {
            int q_global = block_query_base + q_local;
            const float* q_ptr = (q_global < num_queries)
                ? (query_vectors + (size_t)q_global * (size_t)vector_dims)
                : (const float*)0;

            for (int d = threadIdx.x; d < vector_dims; d += 256) {
                s_queries[(size_t)q_local * (size_t)vector_dims + (size_t)d] = q_ptr ? q_ptr[d] : 0.0f;
            }
        }
        __syncthreads();
#endif
    }

    if (query_idx >= num_queries) return;

    const size_t q_stride = (size_t)vector_dims;

    const float* query;
    if (use_smem) {
        query = s_queries + (size_t)local_query * (size_t)vector_dims;
    } else {
        query = query_vectors + (size_t)query_idx * q_stride;
    }

    float lane_min_dist = best_dists[query_idx];
    int lane_best_idx = results[query_idx];

    unsigned mask = half ? 0xFFFF0000u : 0x0000FFFFu;

    if (lane16 == 0) {
        s_bounds_u32[local_query] = __float_as_uint(lane_min_dist);
    }
    __syncwarp(mask);

    volatile unsigned int* vbounds = (volatile unsigned int*)s_bounds_u32;

    const size_t db_block_stride = (size_t)vector_dims * 16ull;

    for (int i = lane16; i < batch_count; i += 16) {
        float bound = __uint_as_float(vbounds[local_query]);

        int db_idx = batch_start + i;
        int db_block = db_idx >> 4;
        int db_lane  = db_idx & 15;

        const float* db_lane_ptr = database_vectors + (size_t)db_block * db_block_stride + (size_t)db_lane;

        float dist = euclidean_distance_high_bounded_blocked(query, db_lane_ptr, vector_dims, bound);

        if (dist < lane_min_dist) {
            lane_min_dist = dist;
            lane_best_idx = db_idx;
        }

        unsigned int votes = __ballot_sync(mask, lane_min_dist < bound);
        if (votes) {
            float best = (lane_min_dist < bound) ? lane_min_dist : MAX_FLOAT;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                float other = __shfl_xor_sync(mask, best, offset);
                if (other < best) best = other;
            }
            if (lane16 == 0) {
                atomicMin((unsigned int*)&s_bounds_u32[local_query], __float_as_uint(best));
            }
            __syncwarp(mask);
        }
    }

    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        float other_dist = __shfl_xor_sync(mask, lane_min_dist, offset);
        int other_idx = __shfl_xor_sync(mask, lane_best_idx, offset);
        if (other_dist < lane_min_dist) {
            lane_min_dist = other_dist;
            lane_best_idx = other_idx;
        }
    }

    if (lane16 == 0) {
        best_dists[query_idx] = lane_min_dist;
        results[query_idx] = lane_best_idx;
    }
}
