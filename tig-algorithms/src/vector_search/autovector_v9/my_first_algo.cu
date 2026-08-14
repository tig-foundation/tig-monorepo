#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32
#define MAX_FLOAT FLT_MAX

extern "C" {

__device__ __forceinline__ float squared_l2_bounded(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int dims,
    float limit
) {
    float sum = 0.0f;
    int i = 0;

    if (dims >= 4) {
        float d0 = a[0]-b[0], d1 = a[1]-b[1], d2 = a[2]-b[2], d3 = a[3]-b[3];
        sum = d0*d0 + d1*d1 + d2*d2 + d3*d3;
        if (sum > limit) return sum;
        i = 4;
    }

    for (; i < dims - 31; i += 32) {
        float d0=a[i]-b[i],     d1=a[i+1]-b[i+1],   d2=a[i+2]-b[i+2],   d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5],   d6=a[i+6]-b[i+6],   d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9],   d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        sum += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7
             + d8*d8+d9*d9+d10*d10+d11*d11+d12*d12+d13*d13+d14*d14+d15*d15;
        if (sum > limit) return sum;

        float d16=a[i+16]-b[i+16], d17=a[i+17]-b[i+17], d18=a[i+18]-b[i+18], d19=a[i+19]-b[i+19];
        float d20=a[i+20]-b[i+20], d21=a[i+21]-b[i+21], d22=a[i+22]-b[i+22], d23=a[i+23]-b[i+23];
        float d24=a[i+24]-b[i+24], d25=a[i+25]-b[i+25], d26=a[i+26]-b[i+26], d27=a[i+27]-b[i+27];
        float d28=a[i+28]-b[i+28], d29=a[i+29]-b[i+29], d30=a[i+30]-b[i+30], d31=a[i+31]-b[i+31];
        sum += d16*d16+d17*d17+d18*d18+d19*d19+d20*d20+d21*d21+d22*d22+d23*d23
             + d24*d24+d25*d25+d26*d26+d27*d27+d28*d28+d29*d29+d30*d30+d31*d31;
        if (sum > limit) return sum;
    }

    for (; i < dims - 15; i += 16) {
        float d0=a[i]-b[i],     d1=a[i+1]-b[i+1],   d2=a[i+2]-b[i+2],   d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5],   d6=a[i+6]-b[i+6],   d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9],   d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        sum += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7
             + d8*d8+d9*d9+d10*d10+d11*d11+d12*d12+d13*d13+d14*d14+d15*d15;
        if (sum > limit) return sum;
    }

    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i],     d1=a[i+1]-b[i+1],   d2=a[i+2]-b[i+2],   d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5],   d6=a[i+6]-b[i+6],   d7=a[i+7]-b[i+7];
        sum += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7;
        if (sum > limit) return sum;
    }

    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        sum += d0*d0+d1*d1+d2*d2+d3*d3;
        if (sum > limit) return sum;
    }

    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
        if (sum > limit) return sum;
    }

    return sum;
}

__device__ __forceinline__ void warp_reduce_min(float& dist, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, dist, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_dist < dist) {
            dist = other_dist;
            idx = other_idx;
        }
    }
}

__global__ void search_warp_per_query(
    const float* __restrict__ query_vectors,
    const float* __restrict__ database_vectors,
    int* __restrict__ results,
    float* __restrict__ best_dists,
    int num_queries,
    int vector_dims,
    int batch_start,
    int batch_count,
    int is_first_batch
) {
    int query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    int lane = threadIdx.x;
    const size_t dims = (size_t)vector_dims;
    const float* query = query_vectors + (size_t)query_idx * dims;

    float my_best_dist = MAX_FLOAT;
    int my_best_idx = 0;

    if (is_first_batch && batch_count > 0) {
        const float* first = database_vectors + (size_t)batch_start * dims;
        my_best_dist = squared_l2_bounded(query, first, vector_dims, MAX_FLOAT);
        my_best_idx = batch_start;
    } else if (!is_first_batch) {
        my_best_dist = best_dists[query_idx];
        my_best_idx = results[query_idx];
    }

    int scan_start = is_first_batch ? 1 : 0;

    for (int i = scan_start + lane; i < batch_count; i += WARP_SIZE) {
        int vec_idx = batch_start + i;
        const float* db_vec = database_vectors + (size_t)vec_idx * dims;

        float dist = squared_l2_bounded(query, db_vec, vector_dims, my_best_dist);
        if (dist < my_best_dist) {
            my_best_dist = dist;
            my_best_idx = vec_idx;
        }
    }

    float final_dist = my_best_dist;
    int final_idx = my_best_idx;
    warp_reduce_min(final_dist, final_idx);

    if (lane == 0) {
        best_dists[query_idx] = final_dist;
        results[query_idx] = final_idx;
    }
}

} // extern "C"
