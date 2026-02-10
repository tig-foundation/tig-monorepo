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
extern "C" __global__ void adam(
    const float* gradients,
    const int n,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float grad_clip_min,
    const float grad_clip_max,
    const float bias_correction1,
    const float bias_correction2,
    const float clip_coef,
    float* momentum,
    float* velocity,
    float* updates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Apply global gradient clipping first (for adaptive clipping)
        float grad = gradients[idx] * clip_coef;
        
        // Then apply element-wise gradient clipping for stability
        grad = fminf(fmaxf(grad, grad_clip_min), grad_clip_max);

        // Update biased first moment estimate (momentum)
        momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate (velocity)
        velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected moments using precomputed factors
        float m_hat = momentum[idx] / bias_correction1;
        float v_hat = velocity[idx] / bias_correction2;

        // Adam update: adaptive learning rate based on momentum and velocity
        updates[idx] = -learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

extern "C" __global__ void adam_vectorized(
    const float* gradients,
    const int n_vec4,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float grad_clip_min,
    const float grad_clip_max,
    const float bias_correction1,
    const float bias_correction2,
    const float clip_coef,
    float* momentum,
    float* velocity,
    float* updates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        // Process 4 floats at a time
        float4 grad4 = reinterpret_cast<const float4*>(gradients)[idx];
        float4 mom4 = reinterpret_cast<float4*>(momentum)[idx];
        float4 vel4 = reinterpret_cast<float4*>(velocity)[idx];
        float4 upd4;
        
        // Process each element
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float grad;
            float mom;
            float vel;
            
            // Extract values
            if (i == 0) { grad = grad4.x; mom = mom4.x; vel = vel4.x; }
            else if (i == 1) { grad = grad4.y; mom = mom4.y; vel = vel4.y; }
            else if (i == 2) { grad = grad4.z; mom = mom4.z; vel = vel4.z; }
            else { grad = grad4.w; mom = mom4.w; vel = vel4.w; }
            
            // Apply global gradient clipping first
            grad *= clip_coef;
            
            // Apply element-wise gradient clipping
            grad = fminf(fmaxf(grad, grad_clip_min), grad_clip_max);
            
            // Update momentum
            mom = beta1 * mom + (1.0f - beta1) * grad;
            
            // Update velocity
            vel = beta2 * vel + (1.0f - beta2) * grad * grad;
            
            // Compute bias-corrected moments
            float m_hat = mom / bias_correction1;
            float v_hat = vel / bias_correction2;
            
            // Adam update
            float upd = -learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
            
            // Store back
            if (i == 0) { mom4.x = mom; vel4.x = vel; upd4.x = upd; }
            else if (i == 1) { mom4.y = mom; vel4.y = vel; upd4.y = upd; }
            else if (i == 2) { mom4.z = mom; vel4.z = vel; upd4.z = upd; }
            else { mom4.w = mom; vel4.w = vel; upd4.w = upd; }
        }
        
        // Write back vectorized results
        reinterpret_cast<float4*>(momentum)[idx] = mom4;
        reinterpret_cast<float4*>(velocity)[idx] = vel4;
        reinterpret_cast<float4*>(updates)[idx] = upd4;
    }
}

__device__ __forceinline__ unsigned int mix_hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

extern "C" __global__ void add_gradient_noise(
    float* gradients,
    const int n,
    const float noise_scale,
    const unsigned int seed,
    const unsigned int epoch,
    const unsigned int step,
    const unsigned int tensor_id,
    const int noise_mode,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (noise_mode <= 0) {
            return;
        }
        unsigned int h = seed ^ (epoch * 0x9e3779b9U) ^ (step * 0x85ebca6bU) ^ (tensor_id * 0xc2b2ae35U) ^ (unsigned int)idx;
        h = mix_hash(h);
        float u = (float)(h & 0x00ffffffU) * (1.0f / 16777216.0f);
        float noise = (u * 2.0f) - 1.0f;
        float g = gradients[idx];
        if (noise_mode == 1) {
            gradients[idx] = g + noise * noise_scale;
        } else if (noise_mode == 2) {
            gradients[idx] = g * (1.0f + noise * noise_scale);
        } else if (noise_mode == 3) {
            gradients[idx] = g + noise * noise_scale * (fabsf(g) + eps);
        } else {
            gradients[idx] = g;
        }
    }
}

extern "C" __global__ void add_gradient_noise_vectorized(
    float* gradients,
    const int n_vec4,
    const float noise_scale,
    const unsigned int seed,
    const unsigned int epoch,
    const unsigned int step,
    const unsigned int tensor_id,
    const int noise_mode,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        if (noise_mode <= 0) {
            return;
        }
        
        // Process 4 floats at a time
        float4 grad4 = reinterpret_cast<float4*>(gradients)[idx];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int element_idx = idx * 4 + i;
            unsigned int h = seed ^ (epoch * 0x9e3779b9U) ^ (step * 0x85ebca6bU) ^ (tensor_id * 0xc2b2ae35U) ^ (unsigned int)element_idx;
            h = mix_hash(h);
            float u = (float)(h & 0x00ffffffU) * (1.0f / 16777216.0f);
            float noise = (u * 2.0f) - 1.0f;
            
            float g;
            if (i == 0) g = grad4.x;
            else if (i == 1) g = grad4.y;
            else if (i == 2) g = grad4.z;
            else g = grad4.w;
            
            float result;
            if (noise_mode == 1) {
                result = g + noise * noise_scale;
            } else if (noise_mode == 2) {
                result = g * (1.0f + noise * noise_scale);
            } else if (noise_mode == 3) {
                result = g + noise * noise_scale * (fabsf(g) + eps);
            } else {
                result = g;
            }
            
            if (i == 0) grad4.x = result;
            else if (i == 1) grad4.y = result;
            else if (i == 2) grad4.z = result;
            else grad4.w = result;
        }
        
        // Write back vectorized results
        reinterpret_cast<float4*>(gradients)[idx] = grad4;
    }
}
