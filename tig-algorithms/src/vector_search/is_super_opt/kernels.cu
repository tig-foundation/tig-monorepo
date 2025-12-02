/*!
Copyright 2025 Optimus_Maximus

Identity of Submitter Optimus_Maximus

UAI null

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

#include <cuda_runtime.h>
#include <float.h>

#define MAX_FLOAT 3.402823466e+38F
#define NUM_REFERENCE_VECTORS 8
#define TOP_K_CLUSTERS 3
#define HISTOGRAM_BINS 100

__device__ float euclidean_distance_squared(const float* a, const float* b, int dims) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < dims - 3; i += 4) {
        float diff0 = a[i] - b[i];
        float diff1 = a[i+1] - b[i+1];
        float diff2 = a[i+2] - b[i+2];
        float diff3 = a[i+3] - b[i+3];
        sum = fmaf(diff0, diff0, sum);
        sum = fmaf(diff1, diff1, sum);
        sum = fmaf(diff2, diff2, sum);
        sum = fmaf(diff3, diff3, sum);
    }
    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        sum = fmaf(diff, diff, sum);
    }
    return sum;  // Return squared distance (no sqrt)
}

// Step 1: Select reference vectors
extern "C" __global__ void select_reference_vectors(
    const float* database_vectors,
    float* reference_vectors,
    int database_size,
    int vector_dims,
    int num_references
) {
    int ref_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ref_idx >= num_references) return;
    
    // Deterministic selection using prime-based hashing
    int db_idx = ((ref_idx * 73856093LL + 19349663LL) % (long long)database_size);
    
    const float* src = database_vectors + db_idx * vector_dims;
    float* dst = reference_vectors + ref_idx * vector_dims;
    
    for (int d = 0; d < vector_dims; d++) {
        dst[d] = src[d];
    }
}

// Step 2: Cluster database vectors to reference vectors
extern "C" __global__ void cluster_database_vectors(
    const float* database_vectors,
    const float* reference_vectors,
    int* cluster_assignments,
    int* cluster_sizes,
    int database_size,
    int vector_dims,
    int num_references
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= database_size) return;
    
    const float* vector = database_vectors + vec_idx * vector_dims;
    
    float min_dist = MAX_FLOAT;
    int best_cluster = 0;
    
    for (int ref = 0; ref < num_references; ref++) {
        const float* ref_vector = reference_vectors + ref * vector_dims;
        float dist = euclidean_distance_squared(vector, ref_vector, vector_dims);
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = ref;
        }
    }
    
    cluster_assignments[vec_idx] = best_cluster;
    atomicAdd(&cluster_sizes[best_cluster], 1);
}

// Step 3: Assign queries to top 3 clusters
extern "C" __global__ void assign_queries_to_clusters(
    const float* query_vectors,
    const float* reference_vectors,
    int* query_cluster_assignments,  // TOP_K_CLUSTERS per query
    float* query_cluster_distances,   // TOP_K_CLUSTERS per query
    int num_queries,
    int vector_dims,
    int num_references
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    const float* query = query_vectors + query_idx * vector_dims;
    
    // Initialize top K
    float top_dists[TOP_K_CLUSTERS];
    int top_clusters[TOP_K_CLUSTERS];
    for (int k = 0; k < TOP_K_CLUSTERS; k++) {
        top_dists[k] = MAX_FLOAT;
        top_clusters[k] = -1;
    }
    
    // Find top K closest reference vectors
    for (int ref = 0; ref < num_references; ref++) {
        const float* ref_vector = reference_vectors + ref * vector_dims;
        float dist = euclidean_distance_squared(query, ref_vector, vector_dims);
        
        // Insert into top K
        for (int k = 0; k < TOP_K_CLUSTERS; k++) {
            if (dist < top_dists[k]) {
                // Shift down
                for (int j = TOP_K_CLUSTERS - 1; j > k; j--) {
                    top_dists[j] = top_dists[j-1];
                    top_clusters[j] = top_clusters[j-1];
                }
                top_dists[k] = dist;
                top_clusters[k] = ref;
                break;
            }
        }
    }
    
    // Store results
    int base_idx = query_idx * TOP_K_CLUSTERS;
    for (int k = 0; k < TOP_K_CLUSTERS; k++) {
        query_cluster_assignments[base_idx + k] = top_clusters[k];
        query_cluster_distances[base_idx + k] = top_dists[k];
    }
}

// Step 4: Compute cluster-query-means
extern "C" __global__ void compute_cluster_query_means(
    const float* query_vectors,
    const int* query_cluster_assignments,
    float* cluster_query_means,
    int* queries_per_cluster,
    int num_queries,
    int vector_dims,
    int num_references
) {
    int cluster_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (cluster_idx >= num_references || dim_idx >= vector_dims) return;
    
    extern __shared__ float shared_sum[];
    shared_sum[dim_idx] = 0.0f;
    __syncthreads();
    
    // Sum queries assigned to this cluster
    int count = 0;
    for (int q = 0; q < num_queries; q++) {
        for (int k = 0; k < TOP_K_CLUSTERS; k++) {
            if (query_cluster_assignments[q * TOP_K_CLUSTERS + k] == cluster_idx) {
                shared_sum[dim_idx] += query_vectors[q * vector_dims + dim_idx];
                if (dim_idx == 0) count++;
                break;  // Only count query once per cluster
            }
        }
    }
    __syncthreads();
    
    // Compute mean
    if (count > 0) {
        cluster_query_means[cluster_idx * vector_dims + dim_idx] = shared_sum[dim_idx] / count;
    }
    
    if (dim_idx == 0) {
        queries_per_cluster[cluster_idx] = count;
    }
}

// Step 5: Build histogram of distances for each cluster in parallel
// Now takes min_distance and max_distance as parameters
extern "C" __global__ void build_distance_histograms(
    const float* database_vectors,
    const float* cluster_query_means,
    const int* cluster_assignments,
    const int* queries_per_cluster,
    int* histograms,  // num_references * HISTOGRAM_BINS
    int database_size,
    int vector_dims,
    int num_references,
    float min_distance,
    float max_distance
) {
    extern __shared__ int shared_histogram[];
    
    int cluster_idx = blockIdx.x;
    if (cluster_idx >= num_references) return;
    
    // Skip clusters with no queries
    if (queries_per_cluster[cluster_idx] == 0) return;
    
    // Initialize shared histogram
    int tid = threadIdx.x;
    if (tid < HISTOGRAM_BINS) {
        shared_histogram[tid] = 0;
    }
    __syncthreads();
    
    const float* cluster_mean = cluster_query_means + cluster_idx * vector_dims;
    float bin_width = (max_distance - min_distance) / HISTOGRAM_BINS;
    
    // Process database vectors in this cluster
    for (int vec_idx = tid; vec_idx < database_size; vec_idx += blockDim.x) {
        if (cluster_assignments[vec_idx] == cluster_idx) {
            const float* vector = database_vectors + vec_idx * vector_dims;
            float dist_squared = euclidean_distance_squared(vector, cluster_mean, vector_dims);
            float dist = sqrtf(dist_squared);
            
            // Calculate bin index
            int bin = (int)((dist - min_distance) / bin_width);
            bin = max(0, min(HISTOGRAM_BINS - 1, bin));
            
            atomicAdd(&shared_histogram[bin], 1);
        }
    }
    __syncthreads();
    
    // Write shared histogram to global memory
    if (tid < HISTOGRAM_BINS) {
        histograms[cluster_idx * HISTOGRAM_BINS + tid] = shared_histogram[tid];
    }
}

// Step 6: Calculate percentile cutoffs based on difficulty
extern "C" __global__ void calculate_percentile_cutoffs(
    const int* histograms,
    const int* cluster_sizes,
    const int* queries_per_cluster,
    float* cluster_cutoff_distances,
    int num_references,
    float keep_percentage,
    float min_distance,
    float max_distance
) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_idx >= num_references) return;
    
    // Skip clusters with no queries - set cutoff to max
    if (queries_per_cluster[cluster_idx] == 0) {
        cluster_cutoff_distances[cluster_idx] = max_distance * max_distance;
        return;
    }
    
    const int* cluster_histogram = histograms + cluster_idx * HISTOGRAM_BINS;
    int cluster_size = cluster_sizes[cluster_idx];
    int target_count = (int)(cluster_size * keep_percentage);
    
    float bin_width = (max_distance - min_distance) / HISTOGRAM_BINS;
    
    // Find cutoff bin
    int cumsum = 0;
    int cutoff_bin = HISTOGRAM_BINS - 1;
    for (int bin = 0; bin < HISTOGRAM_BINS; bin++) {
        cumsum += cluster_histogram[bin];
        if (cumsum >= target_count) {
            cutoff_bin = bin;
            break;
        }
    }
    
    // Convert bin to distance (add 1 to include the entire bin)
    float cutoff_distance = min_distance + (cutoff_bin + 1) * bin_width;
    cluster_cutoff_distances[cluster_idx] = cutoff_distance * cutoff_distance;  // Store squared
}

// Step 7: Filter database vectors using adaptive cutoffs
extern "C" __global__ void filter_database_vectors_adaptive(
    const float* database_vectors,
    const float* cluster_query_means,
    const int* cluster_assignments,
    const float* cluster_cutoff_distances,
    int* vector_valid_flags,
    int database_size,
    int vector_dims,
    int num_references
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= database_size) return;
    
    int cluster = cluster_assignments[vec_idx];
    const float* vector = database_vectors + vec_idx * vector_dims;
    const float* cluster_mean = cluster_query_means + cluster * vector_dims;
    
    float dist_squared = euclidean_distance_squared(vector, cluster_mean, vector_dims);
    
    // Keep vector if within adaptive cutoff for its cluster
    vector_valid_flags[vec_idx] = (dist_squared <= cluster_cutoff_distances[cluster]) ? 1 : 0;
}

// Step 8: Search with filtering and early stopping (parallelized)
extern "C" __global__ void filtered_search(
    const float* query_vectors,
    const float* database_vectors,
    const int* query_cluster_assignments,
    const int* cluster_assignments,
    const int* vector_valid_flags,
    int* results,
    float* result_distances,
    int num_queries,
    int database_size,
    int vector_dims,
    int num_references,
    float distance_threshold
) {
    int query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    const float* query = query_vectors + query_idx * vector_dims;
    
    // Shared memory for reduction
    extern __shared__ char shared_mem[];
    float* shared_dists = (float*)shared_mem;
    int* shared_indices = (int*)(shared_mem + block_size * sizeof(float));
    
    float local_min_dist = MAX_FLOAT;
    int local_best_idx = -1;
    
    // Search in top 3 clusters first
    for (int k = 0; k < TOP_K_CLUSTERS; k++) {
        int target_cluster = query_cluster_assignments[query_idx * TOP_K_CLUSTERS + k];
        if (target_cluster == -1) continue;
        
        // Parallel search in this cluster
        for (int vec_idx = tid; vec_idx < database_size; vec_idx += block_size) {
            if (cluster_assignments[vec_idx] == target_cluster && vector_valid_flags[vec_idx]) {
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance_squared(query, db_vector, vector_dims);
                if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_best_idx = vec_idx;
                }
            }
        }
        
        // Early exit if found good solution
        if (local_min_dist <= distance_threshold) {
            break;
        }
    }
    
    // If not found in top 3, search remaining clusters
    if (local_min_dist > distance_threshold) {
        for (int cluster = 0; cluster < num_references; cluster++) {
            // Skip if already searched
            bool already_searched = false;
            for (int k = 0; k < TOP_K_CLUSTERS; k++) {
                if (query_cluster_assignments[query_idx * TOP_K_CLUSTERS + k] == cluster) {
                    already_searched = true;
                    break;
                }
            }
            if (already_searched) continue;
            
            // Parallel search in this cluster
            for (int vec_idx = tid; vec_idx < database_size; vec_idx += block_size) {
                if (cluster_assignments[vec_idx] == cluster && vector_valid_flags[vec_idx]) {
                    const float* db_vector = database_vectors + vec_idx * vector_dims;
                    float dist = euclidean_distance_squared(query, db_vector, vector_dims);
                    if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_best_idx = vec_idx;
                    }
                }
            }
            
            // Early exit if found good solution
            if (local_min_dist <= distance_threshold) {
                break;
            }
        }
    }
    
    // Store local results in shared memory
    shared_dists[tid] = local_min_dist;
    shared_indices[tid] = local_best_idx;
    __syncthreads();
    
    // Parallel reduction to find minimum across block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_dists[tid + stride] < shared_dists[tid]) {
                shared_dists[tid] = shared_dists[tid + stride];
                shared_indices[tid] = shared_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final result
    if (tid == 0) {
        // If no valid solution found, do unfiltered search
        if (shared_indices[0] == -1) {
            float min_dist = MAX_FLOAT;
            int best_idx = -1;
            for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance_squared(query, db_vector, vector_dims);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                }
            }
            results[query_idx] = best_idx;
            result_distances[query_idx] = min_dist;
        } else {
            results[query_idx] = shared_indices[0];
            result_distances[query_idx] = shared_dists[0];
        }
    }
}