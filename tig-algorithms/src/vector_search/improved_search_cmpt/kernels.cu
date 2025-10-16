/*!Copyright 2025 CodeAlchemist

Identity of Submitter CodeAlchemist

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
    int i;
    
    for (i = 0; i < dims - 7; i += 8) {
        float diff0 = a[i] - b[i];
        float diff1 = a[i+1] - b[i+1];
        float diff2 = a[i+2] - b[i+2];
        float diff3 = a[i+3] - b[i+3];
        float diff4 = a[i+4] - b[i+4];
        float diff5 = a[i+5] - b[i+5];
        float diff6 = a[i+6] - b[i+6];
        float diff7 = a[i+7] - b[i+7];
        
        sum = fmaf(diff0, diff0, sum);
        sum = fmaf(diff1, diff1, sum);
        sum = fmaf(diff2, diff2, sum);
        sum = fmaf(diff3, diff3, sum);
        sum = fmaf(diff4, diff4, sum);
        sum = fmaf(diff5, diff5, sum);
        sum = fmaf(diff6, diff6, sum);
        sum = fmaf(diff7, diff7, sum);
    }
    
    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        sum = fmaf(diff, diff, sum);
    }
    return sum;
}

__device__ float euclidean_distance_high(const float* a, const float* b, int dims) {
    float sum = 0.0f;
    
    for (int i = 0; i < dims - 7; i += 8) {
        float diff0 = a[i] - b[i];
        float diff1 = a[i+1] - b[i+1];
        float diff2 = a[i+2] - b[i+2];
        float diff3 = a[i+3] - b[i+3];
        float diff4 = a[i+4] - b[i+4];
        float diff5 = a[i+5] - b[i+5];
        float diff6 = a[i+6] - b[i+6];
        float diff7 = a[i+7] - b[i+7];
        
        sum = fmaf(diff0, diff0, sum);
        sum = fmaf(diff1, diff1, sum);
        sum = fmaf(diff2, diff2, sum);
        sum = fmaf(diff3, diff3, sum);
        sum = fmaf(diff4, diff4, sum);
        sum = fmaf(diff5, diff5, sum);
        sum = fmaf(diff6, diff6, sum);
        sum = fmaf(diff7, diff7, sum);
    }
    
    for (int i = dims & ~7; i < dims - 3; i += 4) {
        float diff0 = a[i] - b[i];
        float diff1 = a[i+1] - b[i+1];
        float diff2 = a[i+2] - b[i+2];
        float diff3 = a[i+3] - b[i+3];
        sum = fmaf(diff0, diff0, sum);
        sum = fmaf(diff1, diff1, sum);
        sum = fmaf(diff2, diff2, sum);
        sum = fmaf(diff3, diff3, sum);
    }
    
    for (int i = dims & ~3; i < dims; i++) {
        float diff = a[i] - b[i];
        sum = fmaf(diff, diff, sum);
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
    int num_clusters,
    int num_queries
) {
    int cluster_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (cluster_idx >= num_clusters) return;
    
    extern __shared__ float shared_center[];
    
    for (int d = tid; d < vector_dims; d += blockDim.x) {
        shared_center[d] = 0.0f;
    }
    __syncthreads();
    
    int seed_idx = ((cluster_idx * 982451653LL + 1566083941LL) % (long long)database_size);
    const float* seed_vector = database_vectors + seed_idx * vector_dims;
    
    for (int d = tid; d < vector_dims; d += blockDim.x) {
        shared_center[d] = seed_vector[d];
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
            float dist = (num_queries <= 4000) ? 
                euclidean_distance(vector, c_center, vector_dims) :
                euclidean_distance_high(vector, c_center, vector_dims);
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
    if (num_queries <= 4000) {
        int query_idx = blockIdx.x;
        if (query_idx >= num_queries) return;
        
        const float* query = query_vectors + query_idx * vector_dims;
        
        float cluster_dist0 = MAX_FLOAT, cluster_dist1 = MAX_FLOAT, cluster_dist2 = MAX_FLOAT, cluster_dist3 = MAX_FLOAT;
        float cluster_dist4 = MAX_FLOAT, cluster_dist5 = MAX_FLOAT, cluster_dist6 = MAX_FLOAT, cluster_dist7 = MAX_FLOAT;
        
        float best_dist0 = MAX_FLOAT, best_dist1 = MAX_FLOAT, best_dist2 = MAX_FLOAT;
        int best_cluster0 = -1, best_cluster1 = -1, best_cluster2 = -1;
        
        for (int cluster = 0; cluster < num_clusters; cluster++) {
            const float* center = cluster_centers + cluster * vector_dims;
            float dist = euclidean_distance(query, center, vector_dims);
            
            switch(cluster) {
                case 0: cluster_dist0 = dist; break;
                case 1: cluster_dist1 = dist; break;
                case 2: cluster_dist2 = dist; break;
                case 3: cluster_dist3 = dist; break;
                case 4: cluster_dist4 = dist; break;
                case 5: cluster_dist5 = dist; break;
                case 6: cluster_dist6 = dist; break;
                case 7: cluster_dist7 = dist; break;
            }
            
            if (dist < best_dist0) {
                best_dist2 = best_dist1;
                best_cluster2 = best_cluster1;
                best_dist1 = best_dist0;
                best_cluster1 = best_cluster0;
                best_dist0 = dist;
                best_cluster0 = cluster;
            } else if (dist < best_dist1) {
                best_dist2 = best_dist1;
                best_cluster2 = best_cluster1;
                best_dist1 = dist;
                best_cluster1 = cluster;
            } else if (dist < best_dist2) {
                best_dist2 = dist;
                best_cluster2 = cluster;
            }
        }
        
        float min_dist = MAX_FLOAT;
        int best_idx = -1;
        
        int target_cluster = best_cluster0;
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
        
        if (best_cluster1 != -1 && cluster_sizes[best_cluster1] > 0) {
            target_cluster = best_cluster1;
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
        
        if (best_cluster2 != -1 && cluster_sizes[best_cluster2] > 0) {
            target_cluster = best_cluster2;
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
        
        for (int cluster = 0; cluster < num_clusters; cluster++) {
            if (cluster == best_cluster0 || cluster == best_cluster1 || cluster == best_cluster2) continue;
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
        
        results[query_idx] = best_idx;
    } else {
        int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (query_idx >= num_queries) return;
        
        const float* query = query_vectors + query_idx * vector_dims;
        
        float best_dist0 = MAX_FLOAT, best_dist1 = MAX_FLOAT;
        int best_cluster0 = -1, best_cluster1 = -1;
        float cluster_dist0 = MAX_FLOAT;
        
        for (int cluster = 0; cluster < num_clusters; cluster++) {
            const float* center = cluster_centers + cluster * vector_dims;
            float dist = euclidean_distance_high(query, center, vector_dims);
            
            if (cluster == 0) {
                cluster_dist0 = dist;
            }
            
            if (dist < best_dist0) {
                best_dist1 = best_dist0;
                best_cluster1 = best_cluster0;
                best_dist0 = dist;
                best_cluster0 = cluster;
            } else if (dist < best_dist1) {
                best_dist1 = dist;
                best_cluster1 = cluster;
            }
        }
        
        float min_dist = MAX_FLOAT;
        int best_idx = -1;
        
        int target_cluster = best_cluster0;
        if (target_cluster != -1 && cluster_sizes[target_cluster] > 0) {
            for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
                if (cluster_assignments[vec_idx] == target_cluster) {
                    const float* db_vector = database_vectors + vec_idx * vector_dims;
                    float dist = euclidean_distance_high(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = vec_idx;
                    }
                }
            }
        }
        
        if (min_dist == MAX_FLOAT && best_cluster1 != -1 && cluster_sizes[best_cluster1] > 0) {
            target_cluster = best_cluster1;
            for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
                if (cluster_assignments[vec_idx] == target_cluster) {
                    const float* db_vector = database_vectors + vec_idx * vector_dims;
                    float dist = euclidean_distance_high(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = vec_idx;
                    }
                }
            }
        }
        
        if (min_dist == MAX_FLOAT) {
            float search_radius = cluster_dist0 * 2.0f;
            
            for (int cluster = 0; cluster < num_clusters; cluster++) {
                if (cluster == best_cluster0 || cluster == best_cluster1) continue;
                if (cluster_sizes[cluster] == 0) continue;
                
                const float* center = cluster_centers + cluster * vector_dims;
                float cluster_dist = euclidean_distance_high(query, center, vector_dims);
                if (cluster_dist >= search_radius) continue;
                
                for (int vec_idx = 0; vec_idx < database_size; vec_idx++) {
                    if (cluster_assignments[vec_idx] == cluster) {
                        const float* db_vector = database_vectors + vec_idx * vector_dims;
                        float dist = euclidean_distance_high(query, db_vector, vector_dims);
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
}
