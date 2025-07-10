#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ===================================================================
// UTILITY FUNCTIONS
// ===================================================================

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

__device__ float relu(float x) {
    return fmaxf(x, 0.0f);
}
__device__ float relu_grad(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ float sigmoid_from_out(float s_x) {
    return s_x * (1.0f - s_x);
}

// ===================================================================
// DATA GENERATION KERNELS
// ===================================================================

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

// ===================================================================
// NEURAL NETWORK KERNELS
// ===================================================================

extern "C" __global__ void init_linear_layer(
    unsigned long long seed,
    int out_features,
    int in_features,
    float* weights, // (out_features, in_features)
    float* biases   // (out_features)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_features * in_features) return;

    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    float fan_in = (float)in_features;
    float fan_out = (float)out_features;
    float limit = sqrtf(2.0f / (fan_in + fan_out)) * 0.5f;

    weights[idx] = curand_uniform(&state) * 2.0f * limit - limit;

    // Initialize biases to zero (can be done by one thread or memset)
    if (idx < out_features) {
        biases[idx] = 0.0f;
    }
}

extern "C" __global__ void zero_out(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

extern "C" __global__ void add_bias_forward(float* output, const float* bias, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features) return;
    
    int feature_idx = idx % features;
    output[idx] += bias[feature_idx];
}

extern "C" __global__ void activation_forward(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = relu(data[idx]);
}

extern "C" __global__ void loss_mse(
    const float* output,     // (batch_size, out_features)
    const float* target,     // (batch_size, out_features)
    int batch_size,
    int out_features,
    float* grad_loss,        // (batch_size, out_features)
    float* total_loss_out    // single element
) {
    extern __shared__ float s_loss[]; // size = blockDim.x
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float element_loss_sum = 0.0f;
    int n = batch_size * out_features;

    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        float diff = output[i] - target[i];
        grad_loss[i] = 2.0f * diff;// / batch_size;
        element_loss_sum += diff * diff;
    }
    s_loss[tid] = element_loss_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_loss[tid] += s_loss[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(total_loss_out, s_loss[0] / n);
    }
}

extern "C" __global__ void activation_backward(
    const float* grad_in, 
    const float* pre_act_vals, // Input to the activation function from forward pass
    int n, 
    float* grad_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // If pre-activation value was positive, gradient passes through. Otherwise, it's zero.
        grad_out[i] = pre_act_vals[i] > 0.0f ? grad_in[i] : 0.0f;
    }
}

extern "C" __global__ void backward_bias(
    const float* grad_output, // (batch_size, out_features)
    float* bias_grad,         // (out_features)
    int batch_size,
    int out_features
) {
    extern __shared__ float s_grad_sum[]; // size = out_features
    int feature_idx = blockIdx.x; // Each block responsible for one feature
    
    if (feature_idx >= out_features) return;

    float sum = 0.0f;
    for(int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        sum += grad_output[i * out_features + feature_idx];
    }
    s_grad_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_grad_sum[threadIdx.x] += s_grad_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        atomicAdd(&bias_grad[feature_idx], s_grad_sum[0]);
    }
}

extern "C" __global__ void apply_parameter_updates(
    float* params,
    const float* updates,
    int n,
    float learning_rate // Simple SGD example
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= updates[idx] * learning_rate;
    }
}

// ===================================================================
// OPTIMIZER KERNELS
// ===================================================================

extern "C" __global__ void apply_parameter_updates_direct(
    float* params,
    const float* updates,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] += updates[idx]; // Direct addition (updates already scaled)
    }
}

extern "C" __global__ void scale_tensor(
    float* data,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

extern "C" __global__ void copy_tensor(
    float* dst,
    const float* src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}


extern "C" __global__ void batch_norm_backward(
    const float* input,
    const float* grad_output,
    float* grad_input,
    const float* saved_mean,
    const float* saved_inv_variance,
    const float* weight,
    float* weight_grad,
    float* bias_grad,
    int batch_size,
    int num_features,
    float eps
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx >= num_features) return;

    float mean = saved_mean[feature_idx];
    float inv_var = saved_inv_variance[feature_idx];
    float w = weight[feature_idx];
    
    // Step 1: Compute normalized values and sums needed for gradients
    float sum_grad_output = 0.0f;
    float sum_grad_output_times_normalized = 0.0f;
    
    // First pass: compute sums
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int idx = batch_idx * num_features + feature_idx;
        float x = input[idx];
        float grad_out = grad_output[idx];
        float normalized = (x - mean) * inv_var;  // This is x_hat
        
        sum_grad_output += grad_out;
        sum_grad_output_times_normalized += grad_out * normalized;
    }
    
    // Step 2: Compute weight and bias gradients (same as old Rust implementation)
    weight_grad[feature_idx] += sum_grad_output_times_normalized;
    bias_grad[feature_idx] += sum_grad_output;
    
    // Step 3: Compute grad_x_hat (gradient w.r.t. normalized input)
    // grad_x_hat = grad_output * weight (element-wise)
    
    // Step 4: Compute input gradients using the old Rust formula
    float batch_size_f = (float)batch_size;
    
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int idx = batch_idx * num_features + feature_idx;
        float x = input[idx];
        float grad_out = grad_output[idx];
        float normalized = (x - mean) * inv_var;  // x_hat
        
        // grad_x_hat = grad_output * weight
        float grad_x_hat = grad_out * w;
        
        // Corrected gradient calculation for grad_input
        float term1 = grad_x_hat;
        float term2 = sum_grad_output * w / batch_size_f; // sum(grad_x_hat) / batch_size
        float term3 = normalized * (sum_grad_output_times_normalized * w) / batch_size_f; // normalized * sum(grad_x_hat * x_hat) / batch_size
        
        float grad_input_val = (term1 - term2 - term3) * inv_var;
        
        grad_input[idx] = grad_input_val;
    }
}

extern "C" __global__ void transpose_simple(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        output[c * rows + r] = input[idx];
    }
}