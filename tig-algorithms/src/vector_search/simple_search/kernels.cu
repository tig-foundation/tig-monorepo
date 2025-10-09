/*!
Copyright 2025 Uncharted Trading

Identity of Submitter Uncharted Trading

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

extern "C" __global__ void simple_search(
    const int vector_dims,
    const int database_size,
    const int num_queries,
    const float *database_vectors,
    const float *query_vectors,
    const int query,
    float *distances
) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < database_size; i += blockDim.x * gridDim.x) 
    {
        const float *db_vector = database_vectors + i * vector_dims;
        const float *query_vector = query_vectors + query * vector_dims;

        float dist = 0.0f;
        for (int j = 0; j < vector_dims; ++j) 
        {
            float diff = query_vector[j] - db_vector[j];
            dist += diff * diff;
        }

        distances[i] = dist;
    }
}
