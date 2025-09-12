#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__device__ float box_muller(curandState* state, float& z1) {
    float u1, u2;
    // Prevent u1 from being 0 to avoid log(0) -> -inf
    do {
        u1 = curand_uniform(state);
    } while (u1 == 0.0f);
    u2 = curand_uniform(state);
    
    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * M_PI * u2);
    z1 = mag * sinf(2.0f * M_PI * u2);
    return z0;
}

extern "C" __global__ void generate_rff_params(
    unsigned char* seed,
    int output_dims,
    int input_dims,
    int k_rff,
    float lengthscale,
    float* a_params, // (output_dims, k_rff)
    float* b_params, // (output_dims, k_rff)
    float* w_params  // (output_dims, k_rff, input_dims)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_dims * k_rff) return;

    curandState state;
    curand_init(
        *((unsigned long long*)seed) + idx,
        0, 0, &state
    );

    int out_dim = idx / k_rff;
    int k_idx = idx % k_rff;

    // Box-muller generates two samples, cache one
    float z1;
    a_params[idx] = box_muller(&state, z1);
    // Note: this is not perfectly efficient, could be improved by having half threads write z1
    
    b_params[idx] = curand_uniform(&state) * 2.0f * M_PI;

    float lengthscale_inv_sq = 1.0f / (lengthscale * lengthscale);

    for(int in_dim = 0; in_dim < input_dims; ++in_dim) {
        int w_idx = out_dim * k_rff * input_dims + k_idx * input_dims + in_dim;
        float z_w1;
        w_params[w_idx] = lengthscale_inv_sq * box_muller(&state, z_w1);
    }
}


extern "C" __global__ void generate_dataset(
    unsigned char* seed,
    int num_samples,
    int input_dims,
    int output_dims,
    int k_rff,
    float scaling_factor,
    float noise_std,
    const float* a_params,
    const float* b_params,
    const float* w_params,
    float* out_inputs,         // (num_samples, input_dims)
    float* out_targets_noisy,  // (num_samples, output_dims)
    float* out_targets_true    // (num_samples, output_dims)
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= num_samples) return;

    curandState state;
    curand_init(
        *((unsigned long long*)seed) + sample_idx + num_samples, // Offset seed
        0, 0, &state
    );
    
    // Generate input sample
    for (int i = 0; i < input_dims; ++i) {
        out_inputs[sample_idx * input_dims + i] = curand_uniform(&state) * 2.0f - 1.0f;
    }

    // Generate targets
    for (int out_dim = 0; out_dim < output_dims; ++out_dim) {
        float f_val = 0.0f;
        for (int k_idx = 0; k_idx < k_rff; ++k_idx) {
            float wx_sum = 0.0f;
            for (int in_dim = 0; in_dim < input_dims; ++in_dim) {
                float w = w_params[out_dim * k_rff * input_dims + k_idx * input_dims + in_dim];
                float x = out_inputs[sample_idx * input_dims + in_dim];
                wx_sum += w * x;
            }
            float b = b_params[out_dim * k_rff + k_idx];
            float a = a_params[out_dim * k_rff + k_idx];
            f_val += a * cosf(wx_sum + b);
        }
        f_val *= scaling_factor;
        
        float z_noise1;
        float noise = noise_std * box_muller(&state, z_noise1);

        out_targets_true[sample_idx * output_dims + out_dim] = f_val;
        out_targets_noisy[sample_idx * output_dims + out_dim] = f_val + noise;
    }
}