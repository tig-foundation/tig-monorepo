/*
-----------------------------------------------------------------------
Copyright 2025 Opti

Identity of Submitter Opti

Identity of Creator of Algorithmic Method null

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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__device__ __forceinline__ unsigned int mix_bits(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float hash_to_uniform(unsigned int x) {
    // Maps to (0, 1] using 2^-32 scaling.
    return (x + 1.0f) * 2.3283064365386963e-10f;
}

__device__ __forceinline__ float deterministic_standard_normal(
    unsigned long long seed,
    int idx,
    int step_count
) {
    unsigned int seed_lo = static_cast<unsigned int>(seed);
    unsigned int seed_hi = static_cast<unsigned int>(seed >> 32);
    unsigned int base = seed_lo ^ seed_hi ^ static_cast<unsigned int>(idx * 0x9E3779B9u)
        ^ static_cast<unsigned int>(step_count * 0x85EBCA6Bu);

    unsigned int h1 = mix_bits(base);
    unsigned int h2 = mix_bits(base ^ 0xC2B2AE35u);

    float u1 = fmaxf(hash_to_uniform(h1), 1e-7f);
    float u2 = hash_to_uniform(h2);

    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

extern "C" __global__ void adamw(
    const float* params,
    const float* gradients,
    const int n,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float l1_decay,
    const float grad_clip_min,
    const float grad_clip_max,
    const int step_count,
    const int set_update_clipping,
    const float update_clip_min,
    const float update_clip_max,
    const float gradient_noise_std,
    const unsigned long long noise_seed,
    float* momentum,
    float* velocity,
    float* updates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float grad = fminf(fmaxf(gradients[idx], grad_clip_min), grad_clip_max);

    if (gradient_noise_std > 0.0f) {
        grad += gradient_noise_std * deterministic_standard_normal(noise_seed, idx, step_count);
    }

    momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grad;
    velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;

    float m_hat = momentum[idx] / (1.0f - powf(beta1, step_count));
    float v_hat = velocity[idx] / (1.0f - powf(beta2, step_count));

    float adam_step = m_hat / (sqrtf(v_hat) + epsilon);
    float sign_param = (params[idx] > 0.0f) - (params[idx] < 0.0f);
    float reg_step = weight_decay * params[idx] + l1_decay * sign_param;

    float update = -learning_rate * (adam_step + reg_step);

    if (set_update_clipping != 0) {
        update = fminf(fmaxf(update, update_clip_min), update_clip_max);
    }

    updates[idx] = update;
}
