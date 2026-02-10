// GAS_FX CUDA Kernel Implementation
// v2.0: Adam-style optimizer with Fisher preconditioning
// v3.0: Fused kernels for SMPE + DHS + optimizer updates
// v4.0: GAS-aware kernel selection, CONE multi-objective optimization, VTL audit hooks

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// ============================================================================
// v2.0 KERNELS
// ============================================================================

// Main optimizer kernel combining momentum and adaptive learning rates
// Simplified for maximum speed while maintaining quality
extern "C" __global__ __launch_bounds__(256, 4) void gas_fx_kernel(
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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int n = (unsigned int)size;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float grad = gradients[idx];

        // Update biased first moment estimate (momentum)
        float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate (variance)
        float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute adaptive learning rate using fast rsqrt
        float inv_denom = rsqrtf(v_new + eps);
        float update = -lr * m_new * inv_denom;

        // Apply weight decay
        update -= weight_decay * lr * grad;

        m[idx] = m_new;
        v[idx] = v_new;
        updates[idx] = update;
    }
}

// Alternative robust kernel for high-instability phases
extern "C" __global__ void gas_fx_robust_kernel(
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
        
        // Sign-based momentum (more stable)
        float sign_grad = (grad > 0.0f) ? 1.0f : ((grad < 0.0f) ? -1.0f : 0.0f);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * sign_grad;
        
        // Conservative update using momentum sign
        float sign_m = (m[idx] > 0.0f) ? 1.0f : ((m[idx] < 0.0f) ? -1.0f : 0.0f);
        float update = -lr * sign_m;
        
        updates[idx] = update;
    }
}

// Phase 2: Warp-level gradient norm reduction (no shared memory)
extern "C" __global__ __launch_bounds__(256, 4) void compute_grad_norm(
    const float *gradients,
    float *partial_norms,
    int size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int stride = blockDim.x * gridDim.x;
    
    // Each thread accumulates multiple elements
    float local_sum = 0.0f;
    #pragma unroll 4
    for (unsigned int idx = tid; idx < size; idx += stride) {
        float g = gradients[idx];
        local_sum += g * g;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    
    // First thread in each warp writes to global memory
    if (lane == 0) {
        atomicAdd(partial_norms, local_sum);
    }
}

// ============================================================================
// v3.0 FUSED SMPE + ADAM KERNEL
// ============================================================================

/// Fused kernel combining SMPE policy application with Adam update
extern "C" __global__ void fused_smpe_adam_kernel(
    float *updates,
    const float *gradients,
    float *m,
    float *v,
    const float *lr_scales,
    const float *preconditioner,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        float lr_scale = lr_scales[idx];
        if (preconditioner != NULL) {
            grad = grad * preconditioner[idx];
        }
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        float effective_lr = base_lr * lr_scale;
        float denom = sqrtf(v[idx]) + eps;
        float update = -effective_lr * m[idx] / denom;
        updates[idx] = update;
    }
}

// ============================================================================
// v4.1 WEIGHT-AWARE KERNEL (v0.0.5: Leverage model weights for adaptive LR)
// ============================================================================

/// Weight-aware optimization kernel with adaptive learning rate scaling
/// Features: Uses model parameter magnitudes for adaptive learning rate per-parameter
/// v0.0.5: Enables weight-informed CONE negotiation and adaptive regularization
extern "C" __global__ __launch_bounds__(256, 4) void gas_fx_weight_aware_kernel(
    float *updates,
    const float *gradients,
    const float *weights,                 // v0.0.5: Model parameters (NEW)
    float *m,
    float *v,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int n = (unsigned int)size;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float grad = gradients[idx];
        float w = weights[idx];
        
        // Adaptive learning rate based on weight magnitude
        // Intuition: larger magnitude weights → more stable → higher LR
        // Smaller magnitude weights → more sensitive → lower LR (conservative)
        float weight_scale = 1.0f / (1.0f + 0.1f * fabsf(w));
        
        // Update momentum
        float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update variance
        float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute adaptive learning rate with weight-aware scaling
        float inv_denom = rsqrtf(v_new + eps);
        float update = -base_lr * weight_scale * m_new * inv_denom;
        
        // Weight decay with weight magnitude awareness
        update -= weight_decay * base_lr * w;
        
        m[idx] = m_new;
        v[idx] = v_new;
        updates[idx] = update;
    }
}

// ============================================================================
// v3.0 DHS LAPLACIAN UPDATE KERNEL
// ============================================================================

extern "C" __global__ void dhs_laplacian_kernel(
    float *laplacian,
    const int *csr_row_ptr,
    const int *csr_col_idx,
    const float *csr_values,
    int num_nodes
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_nodes) {
        int row_start = csr_row_ptr[row];
        int row_end = csr_row_ptr[row + 1];
        float degree = 0.0f;
        
        for (int i = row_start; i < row_end; i++) {
            degree += csr_values[i];
        }
        laplacian[row * num_nodes + row] = degree;
        for (int i = row_start; i < row_end; i++) {
            int col = csr_col_idx[i];
            float val = csr_values[i];
            laplacian[row * num_nodes + col] = -val;
        }
    }
}

// ============================================================================
// v3.0 DHS COHERENCE KERNEL
// ============================================================================

extern "C" __global__ void dhs_coherence_kernel(
    float *coherence,
    const float *gradients_a,
    const float *gradients_b,
    int size
) {
    __shared__ float shared_dot[256];
    __shared__ float shared_norm_a[256];
    __shared__ float shared_norm_b[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    if (idx < size) {
        float ga = gradients_a[idx];
        float gb = gradients_b[idx];
        dot = ga * gb;
        norm_a = ga * ga;
        norm_b = gb * gb;
    }
    
    shared_dot[tid] = dot;
    shared_norm_a[tid] = norm_a;
    shared_norm_b[tid] = norm_b;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_dot[tid] += shared_dot[tid + s];
            shared_norm_a[tid] += shared_norm_a[tid + s];
            shared_norm_b[tid] += shared_norm_b[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float total_dot = shared_dot[0];
        float total_norm_a = sqrtf(shared_norm_a[0]);
        float total_norm_b = sqrtf(shared_norm_b[0]);
        float cosine_sim = total_dot / (total_norm_a * total_norm_b + 1e-8f);
        coherence[blockIdx.x] = cosine_sim;
    }
}

// ============================================================================
// v3.0 RPC STATE DIGEST KERNEL
// ============================================================================

extern "C" __global__ void rpc_state_digest_kernel(
    unsigned char *digest,
    const float *parameters,
    const float *moments_m,
    const float *moments_v,
    int size
) {
    __shared__ unsigned int hash[8];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int local_hash = 0;
    if (idx < size) {
        unsigned int param_bits = __float_as_uint(parameters[idx]);
        unsigned int m_bits = __float_as_uint(moments_m[idx]);
        unsigned int v_bits = __float_as_uint(moments_v[idx]);
        local_hash = param_bits ^ m_bits ^ v_bits;
    }
    
    __shared__ unsigned int shared_hash[256];
    shared_hash[tid] = local_hash;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_hash[tid] ^= shared_hash[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 8 && blockIdx.x == 0) {
        hash[tid] = shared_hash[tid];
    }
    __syncthreads();
    
    if (tid == 0 && blockIdx.x == 0) {
        for (int i = 0; i < 8; i++) {
            digest[i * 4 + 0] = (hash[i] >> 24) & 0xFF;
            digest[i * 4 + 1] = (hash[i] >> 16) & 0xFF;
            digest[i * 4 + 2] = (hash[i] >> 8) & 0xFF;
            digest[i * 4 + 3] = hash[i] & 0xFF;
        }
    }
}

// ============================================================================
// v4.0 GAS-AWARE KERNEL (Accuracy Mode for Vision/Transformers)
// ============================================================================

/// GAS-selected kernel for high-accuracy models (vision transformers, language models)
/// Features: High curvature sensitivity, layer-specific adaptive scheduling
extern "C" __global__ void gas_accuracy_kernel(
    float *updates,
    const float *gradients,
    float *m,
    float *v,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float layer_scale,      // GAS-determined layer scaling factor
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Apply weight decay with GAS-computed layer scale
        grad = grad + weight_decay * layer_scale * updates[idx];
        
        // Momentum update
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Variance update with aggressive curvature tracking
        float grad_sq = grad * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad_sq;
        
        // Curvature-aware adaptation: scaled by layer importance
        float curvature_scale = 1.0f + (v[idx] - grad_sq) / (grad_sq + eps);
        curvature_scale = fminf(fmaxf(curvature_scale, 0.5f), 2.0f); // Clamp [0.5, 2.0]
        
        // Compute adaptive learning rate
        float effective_lr = base_lr * layer_scale * curvature_scale;
        float denom = sqrtf(v[idx]) + eps;
        float update = -effective_lr * m[idx] / denom;
        
        updates[idx] = update;
    }
}

// ============================================================================
// v4.0 GAS-AWARE KERNEL (Throughput Mode for Recommendation/Sparse Models)
// ============================================================================

/// GAS-selected kernel for high-throughput models (sparse embeddings, MoE)
/// Features: Pruning-aware updates, aggressive momentum for convergence
extern "C" __global__ void gas_throughput_kernel(
    float *updates,
    const float *gradients,
    float *m,
    float *v,
    const unsigned char *pruning_mask,  // Sparsity indicator from GAS
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Check pruning mask: if 0, skip this parameter (pruned)
        if (pruning_mask != NULL && pruning_mask[idx / 8] & (1 << (idx % 8)) == 0) {
            updates[idx] = 0.0f;
            return;
        }
        
        float grad = gradients[idx];
        
        // Aggressive momentum for fast convergence
        float beta1_aggressive = 0.95f; // Higher momentum
        m[idx] = beta1_aggressive * m[idx] + (1.0f - beta1_aggressive) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute update
        float denom = sqrtf(v[idx]) + eps;
        float update = -base_lr * m[idx] / denom;
        
        updates[idx] = update;
    }
}

// ============================================================================
// v4.0 CONE MULTI-OBJECTIVE KERNEL
// ============================================================================

/// CONE-optimized kernel balancing speed, accuracy, memory, and energy
/// Uses Pareto-frontier-derived weights per parameter group
extern "C" __global__ void cone_balanced_kernel(
    float *updates,
    const float *gradients,
    float *m,
    float *v,
    const float *group_weights,         // CONE-computed allocation weights
    const int *group_assignment,        // Which parameter group for each idx
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int size,
    int num_groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Determine parameter group and get allocation weight from CONE
        int group = (group_assignment != NULL) ? group_assignment[idx] : 0;
        group = group % num_groups;
        float group_weight = group_weights[group];
        
        // CONE-aware weight decay: scale by group allocation
        grad = grad + weight_decay * group_weight * updates[idx];
        
        // Momentum update with group-aware scaling
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Multi-objective-aware learning rate
        float effective_lr = base_lr * group_weight;
        float denom = sqrtf(v[idx]) + eps;
        float update = -effective_lr * m[idx] / denom;
        
        updates[idx] = update;
    }
}

// ============================================================================
// v4.0 VTL AUDIT LOGGING HELPER KERNEL
// ============================================================================

/// Compute state digests for VTL audit trail (Sigma II compliant hashing)
/// Creates deterministic snapshots for causal ledger entries
extern "C" __global__ void vtl_audit_digest_kernel(
    unsigned char *out_digest,           // Output: 32-byte SHA256-style hash
    const float *params_before,          // Parameter snapshot before update
    const float *params_after,           // Parameter snapshot after update
    const float *gradients,              // Gradient values (justification)
    int size
) {
    __shared__ unsigned int hash_state[8];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize hash state in thread 0
    if (tid == 0) {
        hash_state[0] = 0x6a09e667;
        hash_state[1] = 0xbb67ae85;
        hash_state[2] = 0x3c6ef372;
        hash_state[3] = 0xa54ff53a;
        hash_state[4] = 0x510e527f;
        hash_state[5] = 0x9b05688c;
        hash_state[6] = 0x1f83d9ab;
        hash_state[7] = 0x5be0cd19;
    }
    __syncthreads();
    
    // Accumulate XOR hash of all parameters
    unsigned int local_hash = 0;
    if (idx < size) {
        unsigned int before_bits = __float_as_uint(params_before[idx]);
        unsigned int after_bits = __float_as_uint(params_after[idx]);
        unsigned int grad_bits = __float_as_uint(gradients[idx]);
        
        // Deterministic mixing: XOR for distribution
        local_hash = before_bits ^ after_bits ^ grad_bits;
    }
    
    __shared__ unsigned int block_hash[256];
    block_hash[tid] = local_hash;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_hash[tid] ^= block_hash[tid + s];
        }
        __syncthreads();
    }
    
    // Write final digest
    if (tid == 0 && blockIdx.x == 0) {
        unsigned int final_hash = block_hash[0];
        
        // Expand to 32 bytes (simple repetition for demo)
        for (int i = 0; i < 8; i++) {
            unsigned int mixed = final_hash ^ hash_state[i];
            out_digest[i * 4 + 0] = (mixed >> 24) & 0xFF;
            out_digest[i * 4 + 1] = (mixed >> 16) & 0xFF;
            out_digest[i * 4 + 2] = (mixed >> 8) & 0xFF;
            out_digest[i * 4 + 3] = mixed & 0xFF;
        }
    }
}

// ============================================================================
// v4.0 PERFORMANCE MONITORING KERNEL
// ============================================================================

/// Monitor CONE objectives per-step: speed, accuracy gradient, memory footprint
extern "C" __global__ void cone_metrics_kernel(
    const float *gradients,
    float *out_grad_norm,               // Output: L2 norm
    float *out_grad_mean,               // Output: mean abs value
    float *out_sparsity,                // Output: sparsity (% zeros)
    int size
) {
    extern __shared__ float shared_metrics[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute local gradient statistics
    float val = 0.0f;
    float abs_val = 0.0f;
    int is_sparse = 0;
    
    if (idx < size) {
        val = gradients[idx];
        abs_val = fabsf(val);
        is_sparse = (val == 0.0f) ? 1 : 0;
    }
    
    // Store in shared memory
    shared_metrics[tid] = val * val;                          // For norm
    shared_metrics[blockDim.x + tid] = abs_val;               // For mean
    shared_metrics[2 * blockDim.x + tid] = (float)is_sparse;  // For sparsity
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_metrics[tid] += shared_metrics[tid + s];
            shared_metrics[blockDim.x + tid] += shared_metrics[blockDim.x + tid + s];
            shared_metrics[2 * blockDim.x + tid] += shared_metrics[2 * blockDim.x + tid + s];
        }
        __syncthreads();
    }
    
    // Write results
    if (tid == 0) {
        float norm_sq = shared_metrics[0];
        float total_abs = shared_metrics[blockDim.x];
        float total_sparse = shared_metrics[2 * blockDim.x];
        
        out_grad_norm[blockIdx.x] = sqrtf(norm_sq);
        out_grad_mean[blockIdx.x] = total_abs / size;
        out_sparsity[blockIdx.x] = total_sparse / size;
    }
}


// ============================================================================
// STAGE 1: PER-LAYER KERNEL WITH OPTIMIZED GRID SIZING
// ============================================================================

// Per-layer Adam kernel with Phase 2b optimizations:
// - FP16 variance storage (halves bandwidth)
// - Scalar-only processing (no vectorization complexity)
// - unroll 4 for loop optimization
// - __launch_bounds__(256, 4) for SM occupancy
// Per-layer launches with (layer_size + 255) / 256 grid sizing
extern "C" __global__ __launch_bounds__(256, 4) void gas_fx_per_layer_kernel(
    float *updates,
    const float *grads,
    float *m,
    unsigned short *v_fp16,
    int layer_size,
    float lr,
    float beta1, float beta2, float eps, float weight_decay
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    
    #pragma unroll 4
    for (int i = idx; i < layer_size; i += stride) {
        float grad = grads[i];
        
        // Update momentum (FP32)
        float m_val = m[i];
        float m_new = beta1 * m_val + one_minus_beta1 * grad;
        
        // Load variance from FP16, compute in FP32
        half v_half = __ushort_as_half(v_fp16[i]);
        float v_val = __half2float(v_half);
        float v_new = beta2 * v_val + one_minus_beta2 * grad * grad;
        
        // Store variance back as FP16
        v_fp16[i] = __half_as_ushort(__float2half(v_new));
        
        // Compute update with weight decay
        float inv_denom = rsqrtf(v_new + eps);
        float update = -lr * m_new * inv_denom - weight_decay * lr * grad;
        
        m[i] = m_new;
        updates[i] = update;
    }
}

// ============================================================================
// LEGACY: MEGA-FUSED KERNEL (replaced by per-layer approach)
// ============================================================================

extern "C" __global__ __launch_bounds__(256, 4) void gas_fx_mega_fused_kernel(
    const unsigned long long *layer_updates_ptrs,
    const unsigned long long *layer_grads_ptrs,
    const unsigned long long *layer_m_ptrs,
    const unsigned long long *layer_v_fp16_ptrs,
    const int *layer_sizes,
    const float *layer_lrs,
    int num_layers,
    float beta1, float beta2, float eps, float weight_decay
) {
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int lane = threadIdx.x % 32;
    
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    
    // Process each layer
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        int layer_size = layer_sizes[layer_idx];
        float lr = layer_lrs[layer_idx];
        
        float *updates = (float *)layer_updates_ptrs[layer_idx];
        const float *grads = (const float *)layer_grads_ptrs[layer_idx];
        float *m = (float *)layer_m_ptrs[layer_idx];
        unsigned short *v_fp16 = (unsigned short *)layer_v_fp16_ptrs[layer_idx];
        
        // Simple scalar path for all elements
        #pragma unroll 4
        for (int idx = global_tid; idx < layer_size; idx += total_threads) {
            float grad = grads[idx];
            
            // Update momentum (FP32)
            float m_val = m[idx];
            float m_new = beta1 * m_val + one_minus_beta1 * grad;
            
            // Load variance from FP16, compute in FP32
            half v_half = __ushort_as_half(v_fp16[idx]);
            float v_val = __half2float(v_half);
            float v_new = beta2 * v_val + one_minus_beta2 * grad * grad;
            
            // Store variance back as FP16
            v_fp16[idx] = __half_as_ushort(__float2half(v_new));
            
            // Compute update
            float inv_denom = rsqrtf(v_new + eps);
            float update = -lr * m_new * inv_denom - weight_decay * lr * grad;
            
            m[idx] = m_new;
            updates[idx] = update;
        }
    }
}

// ============================================================================
// v4.1 Compute-Only Fused Kernel (audit hooks removed)
// ============================================================================

// Fully compute-only path: momentum + variance + update
// No debug counters, no audit hooks, minimal fences
extern "C" __global__ __launch_bounds__(256, 4) void gas_fx_compute_only_kernel(
    float *updates,
    const float *gradients,
    float *m,
    float *v,
    float base_lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int n = (unsigned int)size;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float grad = gradients[idx];
        float m_new = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_new = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        float inv_denom = rsqrtf(v_new + eps);
        m[idx] = m_new;
        v[idx] = v_new;
        updates[idx] = -base_lr * m_new * inv_denom;
    }
}

// ============================================================================
// v4.1 Optional Audit Metadata Kernel (off hot path)
// ============================================================================

// Computes a lightweight digest of pre/post parameters and gradients.
// Intended to be launched asynchronously when audit modes require.
extern "C" __global__ void gas_fx_audit_metadata_kernel(
    unsigned char *out_digest,
    const float *params_before,
    const float *params_after,
    const float *gradients,
    int size
) {
    __shared__ unsigned int block_hash[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local = 0;
    if (idx < size) {
        unsigned int a = __float_as_uint(params_before[idx]);
        unsigned int b = __float_as_uint(params_after[idx]);
        unsigned int g = __float_as_uint(gradients[idx]);
        local = a ^ b ^ g;
    }
    block_hash[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_hash[tid] ^= block_hash[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0 && blockIdx.x == 0) {
        unsigned int h = block_hash[0];
        for (int i = 0; i < 8; i++) {
            unsigned int mixed = h ^ (0x9e3779b9u * (i + 1));
            out_digest[i * 4 + 0] = (mixed >> 24) & 0xFF;
            out_digest[i * 4 + 1] = (mixed >> 16) & 0xFF;
            out_digest[i * 4 + 2] = (mixed >> 8) & 0xFF;
            out_digest[i * 4 + 3] = mixed & 0xFF;
        }
    }
}

