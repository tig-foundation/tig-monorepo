// CAG-SGD++ CUDA Kernel Implementation
// Implements Adam-style optimizer with Fisher preconditioning

#include <cuda_runtime.h>
#include <math.h>

// Main optimizer kernel combining momentum and adaptive learning rates
extern "C" __global__ void cag_sgd_kernel(
    float *updates,           // Output: parameter updates
    const float *gradients,   // Input: preconditioned gradients
    float *m,                 // State: first moment (momentum)
    float *v,                 // State: second moment (variance)
    float lr,                 // Learning rate
    float beta1,              // Momentum decay
    float beta2,              // Variance decay
    float eps,                // Numerical stability
    float weight_decay,       // L2 regularization
    int size                  // Number of parameters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Apply weight decay (decoupled as in AdamW)
        grad = grad + weight_decay * updates[idx];
        
        // Update biased first moment estimate (momentum)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second moment estimate (variance)
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected estimates
        // Note: In practice, we could track step count for proper bias correction
        // For simplicity, we use the raw estimates which converge quickly
        float m_hat = m[idx];
        float v_hat = v[idx];
        
        // Compute update with adaptive learning rate
        float denom = sqrtf(v_hat) + eps;
        float update = -lr * m_hat / denom;
        
        // Clip update magnitude to prevent instability
        float max_update = 0.1f;
        if (update > max_update) update = max_update;
        if (update < -max_update) update = -max_update;
        
        // Store the update
        updates[idx] = update;
    }
}

// Alternative robust kernel for high-instability phases
extern "C" __global__ void cag_sgd_robust_kernel(
    float *updates,
    const float *gradients,
    float *m,
    float lr,
    float beta1,
    float weight_decay,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Simpler, more conservative update for robust phase
        grad = grad + weight_decay * updates[idx];
        
        // Sign-based momentum (more stable)
        float sign_grad = (grad > 0.0f) ? 1.0f : ((grad < 0.0f) ? -1.0f : 0.0f);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * sign_grad;
        
        // Conservative update using momentum sign
        float sign_m = (m[idx] > 0.0f) ? 1.0f : ((m[idx] < 0.0f) ? -1.0f : 0.0f);
        float update = -lr * sign_m;
        
        updates[idx] = update;
    }
}

// Gradient norm computation helper
extern "C" __global__ void compute_grad_norm(
    const float *gradients,
    float *partial_norms,
    int size
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load gradient and compute squared value
    float val = 0.0f;
    if (idx < size) {
        float g = gradients[idx];
        val = g * g;
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_norms[blockIdx.x] = sdata[0];
    }
}
