/*!Copyright 2025 Rootz

Identity of Submitter Rootz

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

__device__ float euclidean_distance(const float* a, const float* b, int dims) {
    float sum = 0.0f;
    for (int i = 0; i < dims; i += 4) {
        float diff0 = a[i] - b[i];
        float diff1 = a[i+1] - b[i+1];
        float diff2 = a[i+2] - b[i+2];
        float diff3 = a[i+3] - b[i+3];
        sum = fmaf(diff0, diff0, sum);
        sum = fmaf(diff1, diff1, sum);
        sum = fmaf(diff2, diff2, sum);
        sum = fmaf(diff3, diff3, sum);
    }
    return sum;
}

extern "C" __global__ void deterministic_clustering(
    const float* database_vectors,
    float* cluster_centers,
    int* cluster_assignments,
    int* cluster_sizes,
    int database_size,
    int vector_dims,
    int num_clusters
) {
    int cluster_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (cluster_idx >= num_clusters) return;
    
    extern __shared__ float shared_mem[];
    float* center = shared_mem;
    
    for (int d = tid; d < vector_dims; d += blockDim.x) {
        center[d] = 0.0f;
    }
    __syncthreads();
    
    int seed_idx = ((cluster_idx * 982451653LL + 1566083941LL) % (long long)database_size);
    const float* seed_vector = database_vectors + seed_idx * vector_dims;
    
    for (int d = tid; d < vector_dims; d += blockDim.x) {
        center[d] = seed_vector[d];
        cluster_centers[cluster_idx * vector_dims + d] = seed_vector[d];
    }
    
    if (tid == 0) {
        cluster_sizes[cluster_idx] = 0;
    }
    __syncthreads();
    
    for (int vec_idx = tid; vec_idx < database_size; vec_idx += blockDim.x) {
        const float* vector = database_vectors + vec_idx * vector_dims;
        
        float min_dist = MAX_FLOAT;
        int best_cluster = 0;
        
        for (int c = 0; c < num_clusters; c++) {
            const float* c_center = cluster_centers + c * vector_dims;
            float dist = euclidean_distance(vector, c_center, vector_dims);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        
        cluster_assignments[vec_idx] = best_cluster;
        if (best_cluster == cluster_idx) {
            atomicAdd(&cluster_sizes[cluster_idx], 1);
        }
    }
}

extern "C" __global__ void cluster_search(
    const float* query_vectors,
    const float* database_vectors,
    const float* cluster_centers,
    const int* cluster_assignments,
    const int* cluster_sizes,
    int* results,
    int num_queries,
    int database_size,
    int vector_dims,
    int num_clusters
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    const float* query = query_vectors + query_idx * vector_dims;
    
    extern __shared__ float shared_mem[];
    float* cluster_dists = shared_mem;
    int* cluster_indices = (int*)&shared_mem[num_clusters];
    
    if (threadIdx.x < num_clusters) {
        cluster_dists[threadIdx.x] = MAX_FLOAT;
        cluster_indices[threadIdx.x] = -1;
    }
    
    float best_dist[2] = {MAX_FLOAT, MAX_FLOAT};
    int best_clusters[2] = {-1, -1};
    
    for (int cluster = 0; cluster < num_clusters; cluster++) {
        const float* center = cluster_centers + cluster * vector_dims;
        float dist = euclidean_distance(query, center, vector_dims);
        
        if (dist < best_dist[0]) {
            best_dist[1] = best_dist[0];
            best_clusters[1] = best_clusters[0];
            best_dist[0] = dist;
            best_clusters[0] = cluster;
        } else if (dist < best_dist[1]) {
            best_dist[1] = dist;
            best_clusters[1] = cluster;
        }
        
        if (cluster < num_clusters && threadIdx.x == 0) {
            cluster_dists[cluster] = dist;
        }
    }
    
    float min_dist = MAX_FLOAT;
    int best_idx = -1;
    
    int target_cluster = best_clusters[0];
    if (target_cluster != -1 && cluster_sizes[target_cluster] > 0) {
        for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
            if (cluster_assignments[vec_idx] == target_cluster) {
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance(query, db_vector, vector_dims);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                }
            }
        }
    }
    
    if (min_dist == MAX_FLOAT && best_clusters[1] != -1 && cluster_sizes[best_clusters[1]] > 0) {
        target_cluster = best_clusters[1];
        for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
            if (cluster_assignments[vec_idx] == target_cluster) {
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance(query, db_vector, vector_dims);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                }
            }
        }
    }
    
    if (min_dist == MAX_FLOAT) {
        float search_radius = cluster_dists[0] * 2.0f;
        
        for (int cluster = 0; cluster < num_clusters; cluster++) {
            if (cluster == best_clusters[0] || cluster == best_clusters[1]) continue;
            if (cluster_dists[cluster] >= search_radius) continue;
            if (cluster_sizes[cluster] == 0) continue;
            
            for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
                if (cluster_assignments[vec_idx] == cluster) {
                    const float* db_vector = database_vectors + vec_idx * vector_dims;
                    float dist = euclidean_distance(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = vec_idx;
                    }
                }
            }
        }
    }
    
    results[query_idx] = best_idx;
}
