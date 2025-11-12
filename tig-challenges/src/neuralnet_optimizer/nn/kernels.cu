#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cmath>


__device__ float relu(float x) {
    return fmaxf(x, 0.0f);
}

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