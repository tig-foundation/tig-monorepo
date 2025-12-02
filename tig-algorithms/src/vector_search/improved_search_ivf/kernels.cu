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
#define N_PROBE 3

__device__ float euclidean_distance(const float* a, const float* b, int dims) {
    float sum = 0.0f;
    int i = 0;
    for (; i < dims - 7; i += 8) {
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

extern "C" __global__ void initialize_centers(
    const float* database_vectors,
    float* cluster_centers,
    int database_size,
    int vector_dims,
    int num_clusters
) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_idx >= num_clusters) return;

    int seed_idx = cluster_idx * (database_size / num_clusters);
    const float* seed_vector = database_vectors + seed_idx * vector_dims;
    float* center_vector = cluster_centers + cluster_idx * vector_dims;

    for (int i = 0; i < vector_dims; ++i) {
        center_vector[i] = seed_vector[i];
    }
}

extern "C" __global__ void assign_clusters_and_count_sizes(
    const float* database_vectors,
    const float* cluster_centers,
    int* cluster_assignments,
    int* cluster_sizes,
    int database_size,
    int vector_dims,
    int num_clusters
) {
    extern __shared__ float shared_centers[];

    int tid_in_block = threadIdx.x;
    for (int i = tid_in_block; i < num_clusters * vector_dims; i += blockDim.x) {
        shared_centers[i] = cluster_centers[i];
    }
    __syncthreads();

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    for (; vec_idx < database_size; vec_idx += grid_stride) {
        const float* vector = database_vectors + vec_idx * vector_dims;
        float min_dist = MAX_FLOAT;
        int best_cluster = -1;

        for (int c = 0; c < num_clusters; ++c) {
            const float* center = shared_centers + c * vector_dims;
            float dist = euclidean_distance(vector, center, vector_dims);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        if (best_cluster != -1) {
            cluster_assignments[vec_idx] = best_cluster;
            atomicAdd(&cluster_sizes[best_cluster], 1);
        }
    }
}

extern "C" __global__ void build_inverted_index(
    const int* cluster_assignments,
    const int* cluster_offsets,
    int* inverted_indices,
    int* temp_cluster_heads,
    int database_size
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    for (; vec_idx < database_size; vec_idx += grid_stride) {
        int cluster_id = cluster_assignments[vec_idx];
        if (cluster_id != -1) {
            int write_pos = atomicAdd(&temp_cluster_heads[cluster_id], 1);
            inverted_indices[write_pos] = vec_idx;
        }
    }
}

extern "C" __global__ void search_ivf(
    const float* query_vectors,
    const float* database_vectors,
    const float* cluster_centers,
    const int* inverted_indices,
    const int* cluster_offsets,
    const int* cluster_sizes,
    int* results,
    int num_queries,
    int vector_dims,
    int num_clusters,
    int database_size
) {
    extern __shared__ float shared_centers[];

    int tid_in_block = threadIdx.x;
    for (int i = tid_in_block; i < num_clusters * vector_dims; i += blockDim.x) {
        shared_centers[i] = cluster_centers[i];
    }
    __syncthreads();

    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    for (; query_idx < num_queries; query_idx += grid_stride) {
        const float* query = query_vectors + query_idx * vector_dims;

        int best_clusters[N_PROBE];
        float best_dists[N_PROBE];
        for(int i = 0; i < N_PROBE; ++i) {
            best_dists[i] = MAX_FLOAT;
            best_clusters[i] = -1;
        }

        for (int c = 0; c < num_clusters; ++c) {
            const float* center = shared_centers + c * vector_dims;
            float dist = euclidean_distance(query, center, vector_dims);
            for(int k = 0; k < N_PROBE; ++k) {
                if (dist < best_dists[k]) {
                    for(int m = N_PROBE - 1; m > k; --m) {
                        best_dists[m] = best_dists[m-1];
                        best_clusters[m] = best_clusters[m-1];
                    }
                    best_dists[k] = dist;
                    best_clusters[k] = c;
                    break;
                }
            }
        }

        float min_dist = MAX_FLOAT;
        int best_idx = -1;

        for (int k = 0; k < N_PROBE; ++k) {
            int cluster_id = best_clusters[k];
            if (cluster_id == -1 || cluster_sizes[cluster_id] == 0) continue;

            int start_idx = cluster_offsets[cluster_id];
            int end_idx = start_idx + cluster_sizes[cluster_id];

            for (int i = start_idx; i < end_idx; ++i) {
                int db_vec_original_idx = inverted_indices[i];
                const float* db_vector = database_vectors + db_vec_original_idx * vector_dims;
                float dist = euclidean_distance(query, db_vector, vector_dims);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = db_vec_original_idx;
                }
            }
        }
        
        if (best_idx == -1) {
            best_idx = query_idx % database_size;
        }

        results[query_idx] = best_idx;
    }
}