/*!
Copyright 2025 jinx

Identity of Submitter jinx

Identity of Creator of Algorithmic Method jinx

UAI null

Licensed under the TIG Inbound Game License v3.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

extern "C" __global__ void adam(
    const float* gradients,
    const int n,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float grad_clip_min,
    const float grad_clip_max,
    const int step_count,
    float* momentum,
    float* velocity,
    float* updates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Clip gradient using configurable range for stability
        float grad = fminf(fmaxf(gradients[idx], grad_clip_min), grad_clip_max);

        // Update biased first moment estimate (momentum)
        momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate (velocity)
        velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected moments
        float m_hat = momentum[idx] / (1.0f - powf(beta1, step_count));
        float v_hat = velocity[idx] / (1.0f - powf(beta2, step_count));

        // Adam update: adaptive learning rate based on momentum and velocity
        updates[idx] = -learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}
