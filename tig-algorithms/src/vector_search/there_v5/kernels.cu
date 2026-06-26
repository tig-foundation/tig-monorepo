#ifndef MAX_FLOAT
#define MAX_FLOAT 3.402823466e+38f
#endif

#ifndef DB_TILE_DIMS
#define DB_TILE_DIMS 32
#endif

__device__ __forceinline__ void load_database_segment_blocked(
    const float* __restrict__ database_vectors,
    float* __restrict__ s_db_blocked,
    int vector_dims,
    int block_base,
    int valid_lanes,
    int dim_base,
    int segment_dims,
    int local_query,
    int lane16
) {
    int db_idx = block_base + local_query;
    const float* __restrict__ db_row = (local_query < valid_lanes)
        ? (database_vectors + (size_t)db_idx * (size_t)vector_dims + (size_t)dim_base)
        : (const float*)0;

    for (int d = lane16; d < segment_dims; d += 16) {
        s_db_blocked[(size_t)d * 16ull + (size_t)local_query] = db_row ? db_row[d] : 0.0f;
    }
}

__device__ __forceinline__ float accumulate_first4_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    float limit
) {
    const float* bp = b_lane;

    float b0 = *bp; bp += 16;
    float b1 = *bp; bp += 16;
    float b2 = *bp; bp += 16;
    float b3 = *bp; bp += 16;

    float d0 = a[0] - b0;
    float d1 = a[1] - b1;
    float d2 = a[2] - b2;
    float d3 = a[3] - b3;

    float sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, d3 * d3)));
    if (sum > limit) return sum;
    return sum;
}

__device__ __forceinline__ float accumulate32_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    float sum,
    float limit
) {
    const float* bp = b_lane;

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

    float d0 = a[0] - b0;
    float d1 = a[1] - b1;
    float d2 = a[2] - b2;
    float d3 = a[3] - b3;
    float d4 = a[4] - b4;
    float d5 = a[5] - b5;
    float d6 = a[6] - b6;
    float d7 = a[7] - b7;
    float d8 = a[8] - b8;
    float d9 = a[9] - b9;
    float d10 = a[10] - b10;
    float d11 = a[11] - b11;
    float d12 = a[12] - b12;
    float d13 = a[13] - b13;
    float d14 = a[14] - b14;
    float d15 = a[15] - b15;

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

    float e16 = a[16] - b16;
    float e17 = a[17] - b17;
    float e18 = a[18] - b18;
    float e19 = a[19] - b19;
    float e20 = a[20] - b20;
    float e21 = a[21] - b21;
    float e22 = a[22] - b22;
    float e23 = a[23] - b23;
    float e24 = a[24] - b24;
    float e25 = a[25] - b25;
    float e26 = a[26] - b26;
    float e27 = a[27] - b27;
    float e28 = a[28] - b28;
    float e29 = a[29] - b29;
    float e30 = a[30] - b30;
    float e31 = a[31] - b31;

    sum = fmaf(e16, e16, fmaf(e17, e17, fmaf(e18, e18, fmaf(e19, e19,
          fmaf(e20, e20, fmaf(e21, e21, fmaf(e22, e22, fmaf(e23, e23,
          fmaf(e24, e24, fmaf(e25, e25, fmaf(e26, e26, fmaf(e27, e27,
          fmaf(e28, e28, fmaf(e29, e29, fmaf(e30, e30, fmaf(e31, e31, sum))))))))))))))));
    if (sum > limit) return sum;

    return sum;
}

__device__ __forceinline__ float accumulate16_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    float sum,
    float limit
) {
    const float* bp = b_lane;

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

    float d0 = a[0] - b0;
    float d1 = a[1] - b1;
    float d2 = a[2] - b2;
    float d3 = a[3] - b3;
    float d4 = a[4] - b4;
    float d5 = a[5] - b5;
    float d6 = a[6] - b6;
    float d7 = a[7] - b7;
    float d8 = a[8] - b8;
    float d9 = a[9] - b9;
    float d10 = a[10] - b10;
    float d11 = a[11] - b11;
    float d12 = a[12] - b12;
    float d13 = a[13] - b13;
    float d14 = a[14] - b14;
    float d15 = a[15] - b15;

    sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
          fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7,
          fmaf(d8, d8, fmaf(d9, d9, fmaf(d10, d10, fmaf(d11, d11,
          fmaf(d12, d12, fmaf(d13, d13, fmaf(d14, d14, fmaf(d15, d15, sum))))))))))))))));
    if (sum > limit) return sum;

    return sum;
}

__device__ __forceinline__ float accumulate8_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    float sum,
    float limit
) {
    const float* bp = b_lane;

    float b0 = *bp; bp += 16;
    float b1 = *bp; bp += 16;
    float b2 = *bp; bp += 16;
    float b3 = *bp; bp += 16;
    float b4 = *bp; bp += 16;
    float b5 = *bp; bp += 16;
    float b6 = *bp; bp += 16;
    float b7 = *bp; bp += 16;

    float d0 = a[0] - b0;
    float d1 = a[1] - b1;
    float d2 = a[2] - b2;
    float d3 = a[3] - b3;
    float d4 = a[4] - b4;
    float d5 = a[5] - b5;
    float d6 = a[6] - b6;
    float d7 = a[7] - b7;

    sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
          fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7, sum))))))));
    if (sum > limit) return sum;

    return sum;
}

__device__ __forceinline__ float accumulate4_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    float sum,
    float limit
) {
    const float* bp = b_lane;

    float b0 = *bp; bp += 16;
    float b1 = *bp; bp += 16;
    float b2 = *bp; bp += 16;
    float b3 = *bp; bp += 16;

    float d0 = a[0] - b0;
    float d1 = a[1] - b1;
    float d2 = a[2] - b2;
    float d3 = a[3] - b3;

    sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3, sum))));
    if (sum > limit) return sum;

    return sum;
}

__device__ __forceinline__ float accumulate_scalar_blocked(
    const float* __restrict__ a,
    const float* __restrict__ b_lane,
    int dims,
    float sum,
    float limit
) {
    const float* bp = b_lane;

    for (int i = 0; i < dims; i++) {
        float diff = a[i] - *bp;
        bp += 16;
        sum = fmaf(diff, diff, sum);
        if (sum > limit) return sum;
    }

    return sum;
}

__device__ __forceinline__ float4 load_db_chunk4(
    const float4* __restrict__ b_chunks,
    int chunk,
    int lane16
) {
    return b_chunks[(size_t)chunk * 16ull + (size_t)lane16];
}

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

__device__ __forceinline__ float euclidean_distance_high_bounded_chunked4(
    const float* __restrict__ a,
    const float* __restrict__ b_block,
    int db_lane,
    int dims,
    float limit
) {
    float sum = 0.0f;
    int i = 0;
    int chunk = 0;

    const int full_chunks = dims >> 2;
    const int tail_base_dim = full_chunks << 2;
    const float4* __restrict__ b_chunks = reinterpret_cast<const float4*>(b_block);

    if (dims >= 4) {
        float4 b0 = load_db_chunk4(b_chunks, chunk, db_lane);

        float d0 = a[0] - b0.x;
        float d1 = a[1] - b0.y;
        float d2 = a[2] - b0.z;
        float d3 = a[3] - b0.w;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, d3 * d3)));
        if (sum > limit) return sum;

        i = 4;
        chunk = 1;
    }

    for (; i < dims - 31; i += 32, chunk += 8) {
        const float* ap = a + i;

        float4 b0 = load_db_chunk4(b_chunks, chunk + 0, db_lane);
        float4 b1 = load_db_chunk4(b_chunks, chunk + 1, db_lane);
        float4 b2 = load_db_chunk4(b_chunks, chunk + 2, db_lane);
        float4 b3 = load_db_chunk4(b_chunks, chunk + 3, db_lane);

        float d0 = ap[0] - b0.x;
        float d1 = ap[1] - b0.y;
        float d2 = ap[2] - b0.z;
        float d3 = ap[3] - b0.w;
        float d4 = ap[4] - b1.x;
        float d5 = ap[5] - b1.y;
        float d6 = ap[6] - b1.z;
        float d7 = ap[7] - b1.w;
        float d8 = ap[8] - b2.x;
        float d9 = ap[9] - b2.y;
        float d10 = ap[10] - b2.z;
        float d11 = ap[11] - b2.w;
        float d12 = ap[12] - b3.x;
        float d13 = ap[13] - b3.y;
        float d14 = ap[14] - b3.z;
        float d15 = ap[15] - b3.w;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
              fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7,
              fmaf(d8, d8, fmaf(d9, d9, fmaf(d10, d10, fmaf(d11, d11,
              fmaf(d12, d12, fmaf(d13, d13, fmaf(d14, d14, fmaf(d15, d15, sum))))))))))))))));
        if (sum > limit) return sum;

        float4 b4 = load_db_chunk4(b_chunks, chunk + 4, db_lane);
        float4 b5 = load_db_chunk4(b_chunks, chunk + 5, db_lane);
        float4 b6 = load_db_chunk4(b_chunks, chunk + 6, db_lane);
        float4 b7 = load_db_chunk4(b_chunks, chunk + 7, db_lane);

        float e16 = ap[16] - b4.x;
        float e17 = ap[17] - b4.y;
        float e18 = ap[18] - b4.z;
        float e19 = ap[19] - b4.w;
        float e20 = ap[20] - b5.x;
        float e21 = ap[21] - b5.y;
        float e22 = ap[22] - b5.z;
        float e23 = ap[23] - b5.w;
        float e24 = ap[24] - b6.x;
        float e25 = ap[25] - b6.y;
        float e26 = ap[26] - b6.z;
        float e27 = ap[27] - b6.w;
        float e28 = ap[28] - b7.x;
        float e29 = ap[29] - b7.y;
        float e30 = ap[30] - b7.z;
        float e31 = ap[31] - b7.w;

        sum = fmaf(e16, e16, fmaf(e17, e17, fmaf(e18, e18, fmaf(e19, e19,
              fmaf(e20, e20, fmaf(e21, e21, fmaf(e22, e22, fmaf(e23, e23,
              fmaf(e24, e24, fmaf(e25, e25, fmaf(e26, e26, fmaf(e27, e27,
              fmaf(e28, e28, fmaf(e29, e29, fmaf(e30, e30, fmaf(e31, e31, sum))))))))))))))));
        if (sum > limit) return sum;
    }

    for (; i < dims - 15; i += 16, chunk += 4) {
        const float* ap = a + i;

        float4 b0 = load_db_chunk4(b_chunks, chunk + 0, db_lane);
        float4 b1 = load_db_chunk4(b_chunks, chunk + 1, db_lane);
        float4 b2 = load_db_chunk4(b_chunks, chunk + 2, db_lane);
        float4 b3 = load_db_chunk4(b_chunks, chunk + 3, db_lane);

        float d0 = ap[0] - b0.x;
        float d1 = ap[1] - b0.y;
        float d2 = ap[2] - b0.z;
        float d3 = ap[3] - b0.w;
        float d4 = ap[4] - b1.x;
        float d5 = ap[5] - b1.y;
        float d6 = ap[6] - b1.z;
        float d7 = ap[7] - b1.w;
        float d8 = ap[8] - b2.x;
        float d9 = ap[9] - b2.y;
        float d10 = ap[10] - b2.z;
        float d11 = ap[11] - b2.w;
        float d12 = ap[12] - b3.x;
        float d13 = ap[13] - b3.y;
        float d14 = ap[14] - b3.z;
        float d15 = ap[15] - b3.w;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
              fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7,
              fmaf(d8, d8, fmaf(d9, d9, fmaf(d10, d10, fmaf(d11, d11,
              fmaf(d12, d12, fmaf(d13, d13, fmaf(d14, d14, fmaf(d15, d15, sum))))))))))))))));
        if (sum > limit) return sum;
    }

    for (; i < dims - 7; i += 8, chunk += 2) {
        const float* ap = a + i;

        float4 b0 = load_db_chunk4(b_chunks, chunk + 0, db_lane);
        float4 b1 = load_db_chunk4(b_chunks, chunk + 1, db_lane);

        float d0 = ap[0] - b0.x;
        float d1 = ap[1] - b0.y;
        float d2 = ap[2] - b0.z;
        float d3 = ap[3] - b0.w;
        float d4 = ap[4] - b1.x;
        float d5 = ap[5] - b1.y;
        float d6 = ap[6] - b1.z;
        float d7 = ap[7] - b1.w;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3,
              fmaf(d4, d4, fmaf(d5, d5, fmaf(d6, d6, fmaf(d7, d7, sum))))))));
        if (sum > limit) return sum;
    }

    for (; i < dims - 3; i += 4, chunk += 1) {
        const float* ap = a + i;
        float4 b0 = load_db_chunk4(b_chunks, chunk, db_lane);

        float d0 = ap[0] - b0.x;
        float d1 = ap[1] - b0.y;
        float d2 = ap[2] - b0.z;
        float d3 = ap[3] - b0.w;

        sum = fmaf(d0, d0, fmaf(d1, d1, fmaf(d2, d2, fmaf(d3, d3, sum))));
        if (sum > limit) return sum;
    }

    if (i < dims) {
        const float* __restrict__ b_tail = b_block + (size_t)full_chunks * 64ull;
        for (int tail_idx = i - tail_base_dim; i < dims; ++i, ++tail_idx) {
            float diff = a[i] - b_tail[(size_t)tail_idx * 16ull + (size_t)db_lane];
            sum = fmaf(diff, diff, sum);
            if (sum > limit) return sum;
        }
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
    int group    = (int)(threadIdx.x >> 4);

    int src_idx = db_block * 16 + lane16;
    const bool valid = (src_idx < database_size);

    const float* __restrict__ src_base = valid
        ? (src_database_vectors + (size_t)src_idx * (size_t)vector_dims)
        : (const float*)0;

    int full_chunks = vector_dims >> 2;
    int tail_dims = vector_dims & 3;
    int tail_base_dim = full_chunks << 2;

    float* __restrict__ dst_block =
        dst_database_blocked + (size_t)db_block * (size_t)vector_dims * 16ull;
    float4* __restrict__ dst_chunks = reinterpret_cast<float4*>(dst_block);

    for (int c = group; c < full_chunks; c += 16) {
        int d = c << 2;
        float4 v;
        if (valid) {
            v.x = src_base[d + 0];
            v.y = src_base[d + 1];
            v.z = src_base[d + 2];
            v.w = src_base[d + 3];
        } else {
            v.x = 0.0f;
            v.y = 0.0f;
            v.z = 0.0f;
            v.w = 0.0f;
        }
        dst_chunks[(size_t)c * 16ull + (size_t)lane16] = v;
    }

    float* __restrict__ dst_tail = dst_block + (size_t)full_chunks * 64ull;

    if (valid) {
        for (int t = group; t < tail_dims; t += 16) {
            dst_tail[(size_t)t * 16ull + (size_t)lane16] = src_base[tail_base_dim + t];
        }
    } else {
        for (int t = group; t < tail_dims; t += 16) {
            dst_tail[(size_t)t * 16ull + (size_t)lane16] = 0.0f;
        }
    }
}

extern "C" __global__ void transform_query_blocked(
    const float* __restrict__ src_query_vectors,
    float* __restrict__ dst_query_blocked,
    int num_queries,
    int vector_dims
) {
    int query_block = (int)blockIdx.x;
    int lane16      = (int)(threadIdx.x & 15);
    int group       = (int)(threadIdx.x >> 4);

    int src_idx = query_block * 16 + lane16;
    const bool valid = (src_idx < num_queries);

    const float* __restrict__ src_base = valid
        ? (src_query_vectors + (size_t)src_idx * (size_t)vector_dims)
        : (const float*)0;

    int full_chunks = vector_dims >> 2;
    int tail_dims = vector_dims & 3;
    int tail_base_dim = full_chunks << 2;

    float* __restrict__ dst_block =
        dst_query_blocked + (size_t)query_block * (size_t)vector_dims * 16ull;
    float4* __restrict__ dst_chunks = reinterpret_cast<float4*>(dst_block);

    for (int c = group; c < full_chunks; c += 16) {
        int d = c << 2;
        float4 v;
        if (valid) {
            v.x = src_base[d + 0];
            v.y = src_base[d + 1];
            v.z = src_base[d + 2];
            v.w = src_base[d + 3];
        } else {
            v.x = 0.0f;
            v.y = 0.0f;
            v.z = 0.0f;
            v.w = 0.0f;
        }
        dst_chunks[(size_t)c * 16ull + (size_t)lane16] = v;
    }

    float* __restrict__ dst_tail = dst_block + (size_t)full_chunks * 64ull;

    if (valid) {
        for (int t = group; t < tail_dims; t += 16) {
            dst_tail[(size_t)t * 16ull + (size_t)lane16] = src_base[tail_base_dim + t];
        }
    } else {
        for (int t = group; t < tail_dims; t += 16) {
            dst_tail[(size_t)t * 16ull + (size_t)lane16] = 0.0f;
        }
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

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int half = lane >> 4;
    int lane16 = lane & 15;

    int local_query = warp_id * 2 + half;
    int query_idx = (int)(blockIdx.x * 16 + local_query);

    if (batch_count <= 0) return;

    const int smem_size = 16 * vector_dims;
    const bool use_smem = (smem_size > 0 && smem_size <= 12288);
    const size_t q_stride = (size_t)vector_dims;

    if (use_smem) {
        int block_query_base = (int)(blockIdx.x * 16);
        for (int q_local = 0; q_local < 16; ++q_local) {
            int q_global = block_query_base + q_local;
            const float* q_ptr = (q_global < num_queries)
                ? (query_vectors + (size_t)q_global * q_stride)
                : (const float*)0;

            for (int d = threadIdx.x; d < vector_dims; d += 256) {
                s_queries[(size_t)q_local * q_stride + (size_t)d] = q_ptr ? q_ptr[d] : 0.0f;
            }
        }
        __syncthreads();
    }

    if (query_idx >= num_queries) return;

    const float* query = use_smem
        ? (s_queries + (size_t)local_query * q_stride)
        : (query_vectors + (size_t)query_idx * q_stride);

    float lane_min_dist = (batch_start == 0) ? MAX_FLOAT : best_dists[query_idx];
    int lane_best_idx = (batch_start == 0) ? 0 : results[query_idx];
    float group_bound = lane_min_dist;

    unsigned mask = half ? 0xFFFF0000u : 0x0000FFFFu;

    const size_t db_block_stride = (size_t)vector_dims * 16ull;
    int db_idx = batch_start + lane16;
    const int db_lane = db_idx & 15;
    const float* db_block_ptr =
        database_vectors + (size_t)(db_idx >> 4) * db_block_stride;

    for (int i = lane16; i < batch_count; i += 16, db_idx += 16, db_block_ptr += db_block_stride) {
        float dist = euclidean_distance_high_bounded_chunked4(
            query,
            db_block_ptr,
            db_lane,
            vector_dims,
            group_bound
        );

        if (dist < lane_min_dist) {
            lane_min_dist = dist;
            lane_best_idx = db_idx;
        }

        unsigned int votes = __ballot_sync(mask, lane_min_dist < group_bound);
        if (votes) {
            float best = (lane_min_dist < group_bound) ? lane_min_dist : MAX_FLOAT;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                float other = __shfl_xor_sync(mask, best, offset);
                if (other < best) best = other;
            }
            group_bound = best;
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