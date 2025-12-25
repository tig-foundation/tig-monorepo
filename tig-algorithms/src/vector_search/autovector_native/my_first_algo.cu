#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

extern "C" {

// Simple vectorized distance (FAST - no cross-warp reduction overhead)
__device__ __forceinline__ float squared_l2_distance_vectorized(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    unsigned int dim
) {
    float dist = 0.0f;
    
    if (dim >= 4 && (dim % 4 == 0)) {
        const float4* a4 = reinterpret_cast<const float4*>(a);
        const float4* b4 = reinterpret_cast<const float4*>(b);
        unsigned int n4 = dim / 4;
        
        #pragma unroll 4
        for (unsigned int i = 0; i < n4; i++) {
            float4 va = a4[i];
            float4 vb = b4[i];
            
            float dx = va.x - vb.x;
            float dy = va.y - vb.y;
            float dz = va.z - vb.z;
            float dw = va.w - vb.w;
            
            dist += dx*dx + dy*dy + dz*dz + dw*dw;
        }
    } else {
        #pragma unroll 8
        for (unsigned int i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
    }
    
    return dist;
}

// Atomic minimum for float with index tracking
__device__ void atomicMinFloat(float* addr, int* idx_addr, float val, int val_idx) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    
    while (true) {
        assumed = old;
        float current_val = __int_as_float(assumed);
        
        if (val >= current_val) break;
        
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
        
        if (assumed == old) {
            *idx_addr = val_idx;
            break;
        }
    }
}

// Centroid assignment kernel
__global__ void assign_to_nearest_centroid(
    const float* __restrict__ vectors,
    const float* __restrict__ centroids,
    int* __restrict__ assignments,
    unsigned int num_vectors,
    unsigned int num_centroids,
    unsigned int dim
) {
    unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    const float* vec = vectors + vec_idx * dim;
    float min_dist = FLT_MAX;
    int best_centroid = 0;
    
    for (unsigned int c = 0; c < num_centroids; c++) {
        const float* cent = centroids + c * dim;
        float dist = squared_l2_distance_vectorized(vec, cent, dim);
        
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = c;
        }
    }
    
    assignments[vec_idx] = best_centroid;
}

// FIXED: Multi-query search kernel with proper coalesced distance and synchronization
__global__ void search_coalesced_multiquery(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    const float* __restrict__ centroids,
    const int* __restrict__ cluster_indices,
    const int* __restrict__ cluster_sizes,
    const int* __restrict__ cluster_offsets,
    const int* __restrict__ probe_lists,  // [batch_size * num_probes]
    int* __restrict__ results,
    int query_offset,          // Starting query index for this batch
    int batch_size,            // Number of queries in this batch
    int total_vectors,
    int dim,
    int num_centroids,
    int num_probes,
    int use_shared_mem
) {
    extern __shared__ float shared_centroids[];
    
    int qid_local = blockIdx.x;  // Query within batch
    if (qid_local >= batch_size) return;
    
    int qid_global = query_offset + qid_local;
    const float* query = queries + qid_global * dim;
    
    // Load centroids into shared memory if enabled
    if (use_shared_mem) {
        int total_elements = num_centroids * dim;
        for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
            shared_centroids[i] = centroids[i];
        }
        __syncthreads();
    }
    
    // Get probe list for this query
    const int* probe_list = probe_lists + qid_local * num_probes;
    
    // Shared memory for block-level reduction
    __shared__ float shared_best_dist;
    __shared__ int shared_best_idx;
    
    if (threadIdx.x == 0) {
        shared_best_dist = FLT_MAX;
        shared_best_idx = -1;
    }
    __syncthreads();
    
    float local_best = FLT_MAX;
    int local_idx = -1;
    
    // Search all probed clusters
    for (int p = 0; p < num_probes; p++) {
        int cluster_id = probe_list[p];
        if (cluster_id < 0 || cluster_id >= num_centroids) continue;
        
        int offset = cluster_offsets[cluster_id];
        int size = cluster_sizes[cluster_id];
        
        // Each thread searches its own vectors
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            int vec_idx = cluster_indices[offset + i];
            if (vec_idx >= total_vectors) continue;
            
            const float* vec = database + vec_idx * dim;
            
            // Simple vectorized distance (fast!)
            float dist = squared_l2_distance_vectorized(query, vec, dim);
            
            if (dist < local_best) {
                local_best = dist;
                local_idx = vec_idx;
            }
        }
    }
    
    // Warp-level reduction
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    float reduced_dist = local_best;
    int reduced_idx = local_idx;
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, reduced_dist, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, reduced_idx, offset);
        
        if (other_dist < reduced_dist) {
            reduced_dist = other_dist;
            reduced_idx = other_idx;
        }
    }
    
    // Store warp results
    __shared__ float warp_mins[8];  // Max 256/32 = 8 warps
    __shared__ int warp_ids[8];
    
    if (lane == 0) {
        warp_mins[warp_id] = reduced_dist;
        warp_ids[warp_id] = reduced_idx;
    }
    __syncthreads();
    
    // Block-level reduction (thread 0 only)
    if (threadIdx.x == 0) {
        float block_best = FLT_MAX;
        int block_idx = -1;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int i = 0; i < num_warps; i++) {
            if (warp_mins[i] < block_best) {
                block_best = warp_mins[i];
                block_idx = warp_ids[i];
            }
        }
        
        if (block_idx != -1 && block_best < shared_best_dist) {
            shared_best_dist = block_best;
            shared_best_idx = block_idx;
        }
    }
    __syncthreads();
    
    // Write final result
    if (threadIdx.x == 0) {
        results[qid_global] = shared_best_idx;
    }
}

// GPU kernel to copy selected vectors as centroids
__global__ void select_centroids_strided(
    const float* __restrict__ vectors,
    float* __restrict__ centroids,
    unsigned int num_vectors,
    unsigned int num_centroids,
    unsigned int dim
) {
    unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c_idx >= num_centroids * dim) return;
    
    unsigned int centroid_id = c_idx / dim;
    unsigned int feature_id = c_idx % dim;
    
    // Strided sampling: every N-th vector
    unsigned int step = num_vectors / num_centroids;
    if (step == 0) step = 1;
    unsigned int vec_idx = (centroid_id * step) % num_vectors;
    
    centroids[c_idx] = vectors[vec_idx * dim + feature_id];
}

// GPU kernel to count vectors per cluster (pass 1)
__global__ void count_cluster_sizes(
    const int* __restrict__ assignments,
    int* __restrict__ cluster_sizes,
    unsigned int num_vectors
) {
    unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    int cluster_id = assignments[vec_idx];
    atomicAdd(&cluster_sizes[cluster_id], 1);
}

// GPU kernel to build cluster indices (pass 2)
__global__ void build_cluster_indices(
    const int* __restrict__ assignments,
    const int* __restrict__ cluster_offsets,
    int* __restrict__ cluster_indices,
    int* __restrict__ cluster_positions,  // Temp array for atomic counters
    unsigned int num_vectors
) {
    unsigned int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    int cluster_id = assignments[vec_idx];
    int offset = cluster_offsets[cluster_id];
    int pos = atomicAdd(&cluster_positions[cluster_id], 1);
    
    cluster_indices[offset + pos] = vec_idx;
}

} // extern "C"
