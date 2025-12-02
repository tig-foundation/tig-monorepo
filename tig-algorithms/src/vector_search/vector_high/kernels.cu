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
    float c = 0.0f;
    int i;
    
    for (i = 0; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];

        float s0 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
        float s1 = d4*d4 + d5*d5 + d6*d6 + d7*d7;
        float s2 = d8*d8 + d9*d9 + d10*d10 + d11*d11;
        float s3 = d12*d12 + d13*d13 + d14*d14 + d15*d15;

        float partial = s0 + s1 + s2 + s3;
        float y = partial - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];

        float values[8] = {d0*d0, d1*d1, d2*d2, d3*d3, d4*d4, d5*d5, d6*d6, d7*d7};

        for (int j = 0; j < 8; j++) {
            float y = values[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }

    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];

        float values[4] = {d0*d0, d1*d1, d2*d2, d3*d3};

        for (int j = 0; j < 4; j++) {
            float y = values[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }

    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        float squared = diff * diff;
        float y = squared - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__device__ float euclidean_distance_high(const float* a, const float* b, int dims) {
    float sum = 0.0f;
    float c = 0.0f;
    int i;

    for (i = 0; i < dims - 31; i += 32) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];
        float d16=a[i+16]-b[i+16], d17=a[i+17]-b[i+17], d18=a[i+18]-b[i+18], d19=a[i+19]-b[i+19];
        float d20=a[i+20]-b[i+20], d21=a[i+21]-b[i+21], d22=a[i+22]-b[i+22], d23=a[i+23]-b[i+23];
        float d24=a[i+24]-b[i+24], d25=a[i+25]-b[i+25], d26=a[i+26]-b[i+26], d27=a[i+27]-b[i+27];
        float d28=a[i+28]-b[i+28], d29=a[i+29]-b[i+29], d30=a[i+30]-b[i+30], d31=a[i+31]-b[i+31];

        float values[32] = {d0*d0, d1*d1, d2*d2, d3*d3, d4*d4, d5*d5, d6*d6, d7*d7,
                           d8*d8, d9*d9, d10*d10, d11*d11, d12*d12, d13*d13, d14*d14, d15*d15,
                           d16*d16, d17*d17, d18*d18, d19*d19, d20*d20, d21*d21, d22*d22, d23*d23,
                           d24*d24, d25*d25, d26*d26, d27*d27, d28*d28, d29*d29, d30*d30, d31*d31};

        for (int j = 0; j < 32; j++) {
            float y = values[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }

    for (; i < dims - 15; i += 16) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];
        float d8=a[i+8]-b[i+8], d9=a[i+9]-b[i+9], d10=a[i+10]-b[i+10], d11=a[i+11]-b[i+11];
        float d12=a[i+12]-b[i+12], d13=a[i+13]-b[i+13], d14=a[i+14]-b[i+14], d15=a[i+15]-b[i+15];

        float values[16] = {d0*d0, d1*d1, d2*d2, d3*d3, d4*d4, d5*d5, d6*d6, d7*d7,
                           d8*d8, d9*d9, d10*d10, d11*d11, d12*d12, d13*d13, d14*d14, d15*d15};

        for (int j = 0; j < 16; j++) {
            float y = values[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }

    for (; i < dims - 7; i += 8) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];
        float d4=a[i+4]-b[i+4], d5=a[i+5]-b[i+5], d6=a[i+6]-b[i+6], d7=a[i+7]-b[i+7];

        float values[8] = {d0*d0, d1*d1, d2*d2, d3*d3, d4*d4, d5*d5, d6*d6, d7*d7};

        for (int j = 0; j < 8; j++) {
            float y = values[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }

    for (; i < dims - 3; i += 4) {
        float d0=a[i]-b[i], d1=a[i+1]-b[i+1], d2=a[i+2]-b[i+2], d3=a[i+3]-b[i+3];

        float values[4] = {d0*d0, d1*d1, d2*d2, d3*d3};

        for (int j = 0; j < 4; j++) {
            float y = values[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
    }

    for (; i < dims; i++) {
        float diff = a[i] - b[i];
        float squared = diff * diff;
        float y = squared - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
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

    long long seed_idx = ((long long)cluster_idx * 982451653LL + 1566083941LL) % (long long)database_size;
    const float* seed_vector = database_vectors + seed_idx * vector_dims;

    for (int d = tid; d < vector_dims; d += blockDim.x) {
        cluster_centers[cluster_idx * vector_dims + d] = seed_vector[d];
    }

    if (tid == 0) {
        cluster_sizes[cluster_idx] = 0;
    }
}

extern "C" __global__ void assign_clusters(
    const float* database_vectors,
    const float* cluster_centers,
    int* cluster_assignments,
    int* cluster_sizes,
    int database_size,
    int vector_dims,
    int num_clusters,
    int num_queries
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const bool use_high = (num_queries > 4500);

    for (int vec_idx = tid; vec_idx < database_size; vec_idx += stride) {
        const float* vector = database_vectors + vec_idx * vector_dims;

        float min_dist = MAX_FLOAT;
        int best_cluster = 0;

        for (int c = 0; c < num_clusters; c++) {
            const float* c_center = cluster_centers + c * vector_dims;
            float dist = use_high ? euclidean_distance_high(vector, c_center, vector_dims)
                                  : euclidean_distance(vector, c_center, vector_dims);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        cluster_assignments[vec_idx] = best_cluster;
        atomicAdd(&cluster_sizes[best_cluster], 1);
    }
}

extern "C" __global__ void exclusive_scan_sizes(
    const int* cluster_sizes,
    int* cluster_offsets,
    int* write_offsets,
    int num_clusters
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int acc = 0;
        for (int i = 0; i < num_clusters; i++) {
            cluster_offsets[i] = acc;
            write_offsets[i] = acc;
            acc += cluster_sizes[i];
        }
    }
}

extern "C" __global__ void build_cluster_index(
    const int* cluster_assignments,
    int* write_offsets,
    int* cluster_indices,
    int database_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < database_size; i += stride) {
        int c = cluster_assignments[i];
        int pos = atomicAdd(&write_offsets[c], 1);
        cluster_indices[pos] = i;
    }
}

extern "C" __global__ void cluster_search(
    const float* query_vectors,
    const float* database_vectors,
    const float* cluster_centers,
    const int* cluster_assignments,
    const int* cluster_sizes,
    const int* cluster_indices,
    const int* cluster_offsets,
    int* results,
    int num_queries,
    int database_size,
    int vector_dims,
    int num_clusters
) {
    if (num_queries <= 3000) {
        int query_idx = blockIdx.x;
        if (query_idx >= num_queries) return;

        const float* query = query_vectors + query_idx * vector_dims;

        float cluster_dists[8];
        int cluster_order[8];

        for (int cluster = 0; cluster < num_clusters; cluster++) {
            const float* center = cluster_centers + cluster * vector_dims;
            cluster_dists[cluster] = euclidean_distance(query, center, vector_dims);
            cluster_order[cluster] = cluster;
        }

        for (int i = 0; i < num_clusters - 1; i++) {
            for (int j = i + 1; j < num_clusters; j++) {
                if (cluster_dists[cluster_order[i]] > cluster_dists[cluster_order[j]]) {
                    int temp = cluster_order[i];
                    cluster_order[i] = cluster_order[j];
                    cluster_order[j] = temp;
                }
            }
        }

        float min_dist = MAX_FLOAT;
        int best_idx = -1;

        int clusters_to_search = (num_queries <= 1000) ? num_clusters :
                                (num_queries <= 2000) ? min(num_clusters, (num_clusters * 3) / 4) :
                                (num_queries <= 2800) ? min(num_clusters, (num_clusters * 2) / 3) :
                                min(num_clusters, num_clusters / 2);

        for (int c_idx = 0; c_idx < clusters_to_search; c_idx++) {
            int target_cluster = cluster_order[c_idx];
            if (cluster_sizes[target_cluster] <= 0) continue;

            int start = cluster_offsets[target_cluster];
            int end = start + cluster_sizes[target_cluster];
            for (int p = start; p < end; p++) {
                int vec_idx = cluster_indices[p];
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance(query, db_vector, vector_dims);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            int base_stride = max(5, database_size / 2000);
            int max_checks = min(database_size / base_stride, 2000);

            for (int phase = 0; phase < 2; phase++) {
                int offset = phase * (base_stride / 2);
                for (int i = 0; i < max_checks / 2; i++) {
                    int db_idx = (offset + i * base_stride) % database_size;

                    const float* db_vector = database_vectors + db_idx * vector_dims;
                    float dist = euclidean_distance(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = db_idx;
                    }
                }
            }

            if (best_idx != -1) {
                int radius = min(25, base_stride);
                int start_local = max(0, best_idx - radius);
                int end_local = min(database_size, best_idx + radius + 1);

                for (int i = start_local; i < end_local; i++) {
                    if (i == best_idx) continue;
                    const float* db_vector = database_vectors + i * vector_dims;
                    float dist = euclidean_distance(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = i;
                    }
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            best_idx = 0;
        }

        results[query_idx] = best_idx;
    } else {
        int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (query_idx >= num_queries) return;

        const float* query = query_vectors + query_idx * vector_dims;

        float cluster_dists[8];
        int cluster_order[8];

        for (int cluster = 0; cluster < num_clusters; cluster++) {
            const float* center = cluster_centers + cluster * vector_dims;
            cluster_dists[cluster] = euclidean_distance_high(query, center, vector_dims);
            cluster_order[cluster] = cluster;
        }

        for (int i = 0; i < num_clusters - 1; i++) {
            for (int j = i + 1; j < num_clusters; j++) {
                if (cluster_dists[cluster_order[i]] > cluster_dists[cluster_order[j]]) {
                    int temp = cluster_order[i];
                    cluster_order[i] = cluster_order[j];
                    cluster_order[j] = temp;
                }
            }
        }

        float min_dist = MAX_FLOAT;
        int best_idx = -1;

        int clusters_to_search = (num_queries <= 3500) ? min(num_clusters, 4) :
                                (num_queries <= 5500) ? min(num_clusters, 3) :
                                (num_queries <= 8000) ? min(num_clusters, 2) :
                                1;

        for (int c_idx = 0; c_idx < clusters_to_search; c_idx++) {
            int target_cluster = cluster_order[c_idx];
            if (cluster_sizes[target_cluster] <= 0) continue;
            
            if (c_idx > 0 && cluster_dists[target_cluster] > min_dist * 1.5f) break;

            int start = cluster_offsets[target_cluster];
            int end = start + cluster_sizes[target_cluster];
            for (int p = start; p < end; p++) {
                int vec_idx = cluster_indices[p];
                const float* db_vector = database_vectors + vec_idx * vector_dims;
                float dist = euclidean_distance_high(query, db_vector, vector_dims);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = vec_idx;
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            int base_stride = max(9, database_size / 1200);
            int max_checks = min(database_size / base_stride, 1200);

            for (int phase = 0; phase < 2; phase++) {
                int offset = phase * (base_stride / 3);
                int phase_checks = max_checks / 2;

                for (int i = 0; i < phase_checks; i++) {
                    int db_idx = (offset + i * base_stride) % database_size;

                    const float* db_vector = database_vectors + db_idx * vector_dims;
                    float dist = euclidean_distance_high(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = db_idx;
                    }
                }
            }

            if (best_idx != -1) {
                int radius = min(18, base_stride / 2);
                int start_local = max(0, best_idx - radius);
                int end_local = min(database_size, best_idx + radius + 1);

                for (int i = start_local; i < end_local; i++) {
                    if (i == best_idx) continue;
                    const float* db_vector = database_vectors + i * vector_dims;
                    float dist = euclidean_distance_high(query, db_vector, vector_dims);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_idx = i;
                    }
                }
            }
        }

        if (min_dist == MAX_FLOAT) {
            best_idx = 0;
        }

        results[query_idx] = best_idx;
    }
}
