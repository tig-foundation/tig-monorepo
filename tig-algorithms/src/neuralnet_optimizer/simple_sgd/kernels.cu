/*!
Copyright 2025 Uncharted Trading

Licensed under the TIG Open Data License v2.0 or (at your option) any later version
(the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

extern "C" __global__ void sgd(
    const float* gradients,
    const int n,
    const float learning_rate,
    float* updates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        updates[idx] = fminf(fmaxf(gradients[idx], -1.0f), 1.0f) * -learning_rate;
    }
}
