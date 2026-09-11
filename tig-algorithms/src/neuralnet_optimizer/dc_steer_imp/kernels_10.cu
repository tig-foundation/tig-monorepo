__device__ __forceinline__ unsigned int t26_tensor_of_block(
    const int* __restrict__ meta,
    const unsigned int num_tensors,
    const unsigned int b
) {
    unsigned int lo = 0u, hi = num_tensors - 1u;
    while (lo < hi) {
        const unsigned int mid = (lo + hi) >> 1;
        if ((unsigned int)meta[mid + 1] <= b) lo = mid + 1u; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ void sign_directional_consensus_body_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ prev_grad,
    float* __restrict__ parameter_average,
    float* __restrict__ fisher_diag,
    float* __restrict__ slow_update,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float eps,
    const float weight_decay,
    const float rel_update_cap,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi,
    const unsigned int average_select,
    const unsigned int average_update,
    const unsigned int average_initialized,
    const unsigned int tid,
    const unsigned int stride
) {
    if (tid >= n) return;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 0.025f;
    const float wd_lr = lr * weight_decay;
    const float abs_floor = 1.0e-3f;
    const float min_step = 0.18f * lr;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fisher_diag[idx];
        const float g = gradients[idx];
        const float w = params[idx];
        const float fd_std = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.2f * fd_std), 4.2f * fd_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        const float rms = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_n = g_clipped / rms;
        const float u_quant = -lr * copysignf(1.0f, g_n);
        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * u_quant;
        const float final_update = one_minus_la_alpha * u_quant + lookahead_alpha * su;
        const float target = lr * fabsf(g_n);
        const float uabs = fabsf(final_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        float adj_update = final_update * scale;
        adj_update -= wd_lr * w;
        const float max_step = fmaxf(rel_update_cap * (fabsf(w) + abs_floor), min_step);
        adj_update = fminf(fmaxf(adj_update, -max_step), max_step);
        float snapshot = parameter_average[idx];
        if (!average_initialized || average_select) snapshot = w;
        if (average_update && average_initialized) adj_update = snapshot - w;
        parameter_average[idx] = snapshot;
        fisher_diag[idx] = fd;
        slow_update[idx] = su;
        updates[idx] = adj_update;
    }
}

__device__ __forceinline__ void dual_consensus_fisher_body_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ velocity,
    float* __restrict__ prev_grad,
    float* __restrict__ prev_update,
    float* __restrict__ slow_update,
    float* __restrict__ fisher_diag,
    float* __restrict__ parameter_average,
    float* __restrict__ updates,
    float* __restrict__ adaptive_targets,
    const unsigned int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const float blend_adam,
    const float blend_norm,
    const float blend_sign,
    const float nesterov_gamma,
    const float bb_blend,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi,
    const unsigned int average_select,
    const unsigned int average_update,
    const unsigned int average_initialized,
    float* smem,
    const unsigned int tid,
    const unsigned int stride
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int n_warps = (blockDim.x + 31) >> 5;
    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 0.02f;
    const float ortho_mix = fminf(fmaxf(0.12f + 0.42f * blend_sign, 0.0f), 0.58f);

    float local_sq_sum = 0.0f;
    float local_count = 0.0f;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        local_sq_sum += g * g;
        local_count += 1.0f;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    if (lane == 0) {
        smem[warp_id] = local_sq_sum;
        smem[4 + warp_id] = local_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total_sq_sum = lane < n_warps ? smem[lane] : 0.0f;
        float total_count = lane < n_warps ? smem[4 + lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total_sq_sum += __shfl_down_sync(0xffffffff, total_sq_sum, offset);
            total_count += __shfl_down_sync(0xffffffff, total_count, offset);
        }
        if (lane == 0) smem[0] = sqrtf(fmaxf(total_sq_sum / fmaxf(total_count, 1.0f), 0.0f)) + eps;
    }
    __syncthreads();
    const float inv_block_rms = 1.0f / fmaxf(smem[0], 1.0e-8f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];
        const float w = params[idx];
        const float gamma_local = g * pg >= 0.0f ? nesterov_gamma : 0.20f * nesterov_gamma;
        const float g_pred = g + gamma_local * (g - pg);
        float m = momentum[idx];
        float v = velocity[idx];
        m = beta1 * m + one_minus_beta1 * g_pred;
        const float err = g_pred - m;
        v = beta2 * v + one_minus_beta2 * err * err;
        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;
        const float sqrt_v = sqrtf(fmaxf(v_hat, 0.0f));
        const float adaptive_eps = eps * (1.0f + 0.06f * sqrt_v);
        const float inv_denom = 1.0f / fmaxf(sqrt_v + adaptive_eps, 1.0e-12f);
        const float adam_update = -lr * (m_hat * inv_denom + weight_decay * g_pred);
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update = -lr * g_over_denom;
        const float sign_update = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update + blend_norm * norm_update + blend_sign * sign_update;
        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float ortho_update = -lr * ((g_pred - overlap) * inv_denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;
        const float s_pu = prev_update[idx];
        const float bb_scale = fabsf(s_pu) > 1e-6f ? fminf(fabsf(s_pu) * 2.0f, 2.5f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;
        float fd = fisher_diag[idx];
        const float grad_std = sqrtf(fmaxf(v, 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g_pred, -5.0f * grad_std), 5.0f * grad_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        const float fisher_rms = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float robust_track = 0.5f * sign_update + 0.5f * (-lr * (g_pred / fisher_rms));
        const float flip = g * pg < 0.0f ? 1.0f : 0.0f;
        const float vol = fminf(sqrt_v * 0.33333334f, 1.0f);
        const float agree = base_update * robust_track >= 0.0f ? 1.0f : 0.0f;
        const float grad_mom_align = g_pred * m_hat >= 0.0f ? 1.0f : 0.0f;
        const float stability = grad_mom_align * (1.0f - flip);
        const float curvature = fabsf(g - pg) / fmaxf(fabsf(g) + fabsf(pg) + eps, 1.0e-8f);
        float consensus_mix = 0.18f * fminf(curvature, 1.0f) + 0.32f * (1.0f - agree) + 0.22f * vol + 0.22f * blend_sign + 0.12f * (1.0f - stability);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;
        const float align_strength = grad_mom_align * (1.0f - flip) * (1.0f - vol);
        chosen_update *= (1.0f + 0.12f * align_strength) / (1.0f + 0.60f * vol + 0.60f * flip);
        const float elem_rms = fabsf(g) * inv_block_rms;
        if (elem_rms > 1.8f) chosen_update *= 1.8f / elem_rms;
        const float target = lr * fabsf(g_over_denom);
        chosen_update *= fminf(fmaxf(target / fmaxf(fabsf(chosen_update), 1.0e-12f), gate_lo), gate_hi);
        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * chosen_update;
        float final_update = one_minus_la_alpha * chosen_update + lookahead_alpha * su;
        float snapshot = parameter_average[idx];
        if (!average_initialized || average_select) snapshot = w;
        if (average_update && average_initialized) final_update = snapshot - w;
        momentum[idx] = m; velocity[idx] = v; prev_grad[idx] = g; prev_update[idx] = final_update;
        slow_update[idx] = su; fisher_diag[idx] = fd; parameter_average[idx] = snapshot; updates[idx] = final_update;
        adaptive_targets[idx] = adam_update;
    }
}

extern "C" __global__ __launch_bounds__(128, 6) void sign_directional_consensus_kernel_10(
    const float* gradients, const float* params, float* prev_grad, float* directional_consistency,
    float* fisher_diag, float* slow_update, float* updates, const unsigned int n, const float lr,
    const float eps, const float weight_decay, const float rel_update_cap, const float lookahead_alpha,
    const float lookahead_tau, const float gate_lo, const float gate_hi
) {
    sign_directional_consensus_body_10(gradients, params, prev_grad, directional_consistency, fisher_diag, slow_update, updates, n, lr, eps, weight_decay, rel_update_cap, lookahead_alpha, lookahead_tau, gate_lo, gate_hi, 0u, 0u, 0u, blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x);
}

extern "C" __global__ __launch_bounds__(128, 6) void dual_consensus_fisher_kernel_10(
    const float* gradients, const float* params, float* momentum, float* velocity, float* prev_grad,
    float* prev_update, float* slow_update, float* fisher_diag, float* directional_consistency,
    float* updates, float* adaptive_targets, const unsigned int n, const float lr, const float beta1, const float beta2,
    const float eps, const float weight_decay, const float bias_correction1, const float bias_correction2,
    const float blend_adam, const float blend_norm, const float blend_sign, const float nesterov_gamma,
    const float bb_blend, const float lookahead_alpha, const float lookahead_tau, const float gate_lo,
    const float gate_hi, const unsigned int average_select, const unsigned int average_update,
    const unsigned int average_initialized
) {
    __shared__ float smem[8];
    dual_consensus_fisher_body_10(gradients, params, momentum, velocity, prev_grad, prev_update, slow_update, fisher_diag, directional_consistency, updates, adaptive_targets, n, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau, gate_lo, gate_hi, average_select, average_update, average_initialized, smem, blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x);
}

extern "C" __global__ __launch_bounds__(128, 6) void sign_directional_consensus_fused_t26(
    const unsigned long long* all_ptrs, const float* scal, const int* meta, const unsigned int num_tensors,
    const float eps, const float weight_decay, const float lookahead_alpha, const float lookahead_tau,
    const float gate_lo, const float gate_hi, const unsigned int average_select,
    const unsigned int average_update, const unsigned int average_initialized
) {
    const unsigned int b = blockIdx.x, t = t26_tensor_of_block(meta, num_tensors, b);
    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk = (unsigned int)meta[t + 1] - first;
    const unsigned int n = (unsigned int)meta[num_tensors + 1u + t];
    sign_directional_consensus_body_10((const float*)all_ptrs[t], (const float*)all_ptrs[num_tensors + t], (float*)all_ptrs[4u * num_tensors + t], (float*)all_ptrs[8u * num_tensors + t], (float*)all_ptrs[7u * num_tensors + t], (float*)all_ptrs[6u * num_tensors + t], (float*)all_ptrs[9u * num_tensors + t], n, scal[t], eps, weight_decay, scal[num_tensors + t], lookahead_alpha, lookahead_tau, gate_lo, gate_hi, average_select, average_update, average_initialized, (b - first) * blockDim.x + threadIdx.x, blockDim.x * nblk);
}

extern "C" __global__ __launch_bounds__(128, 6) void dual_consensus_fisher_fused_t26(
    const unsigned long long* all_ptrs, const float* scal, const int* meta, const unsigned int num_tensors,
    const float beta1, const float beta2, const float eps, const float weight_decay,
    const float bias_correction1, const float bias_correction2, const float blend_adam,
    const float blend_norm, const float blend_sign, const float nesterov_gamma, const float bb_blend,
    const float lookahead_alpha, const float lookahead_tau, const float gate_lo, const float gate_hi,
    const unsigned int average_select, const unsigned int average_update, const unsigned int average_initialized
) {
    __shared__ float smem[8];
    const unsigned int b = blockIdx.x, t = t26_tensor_of_block(meta, num_tensors, b);
    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk = (unsigned int)meta[t + 1] - first;
    const unsigned int n = (unsigned int)meta[num_tensors + 1u + t];
    dual_consensus_fisher_body_10((const float*)all_ptrs[t], (const float*)all_ptrs[num_tensors + t], (float*)all_ptrs[2u * num_tensors + t], (float*)all_ptrs[3u * num_tensors + t], (float*)all_ptrs[4u * num_tensors + t], (float*)all_ptrs[5u * num_tensors + t], (float*)all_ptrs[6u * num_tensors + t], (float*)all_ptrs[7u * num_tensors + t], (float*)all_ptrs[8u * num_tensors + t], (float*)all_ptrs[9u * num_tensors + t], (float*)all_ptrs[10u * num_tensors + t], n, scal[t], beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2, blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend, lookahead_alpha, lookahead_tau, gate_lo, gate_hi, average_select, average_update, average_initialized, smem, (b - first) * blockDim.x + threadIdx.x, blockDim.x * nblk);
}

extern "C" __global__ __launch_bounds__(128, 6) void update_energy_reduce_fused_t26(
    const unsigned long long* all_ptrs,
    const int* meta,
    const unsigned int num_tensors,
    float* provisional_partials,
    float* target_partials
) {
    __shared__ float smem[8];
    const unsigned int b = blockIdx.x, t = t26_tensor_of_block(meta, num_tensors, b);
    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk = (unsigned int)meta[t + 1] - first;
    const unsigned int n = (unsigned int)meta[num_tensors + 1u + t];
    const unsigned int tid = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;
    const float* updates = (const float*)all_ptrs[9u * num_tensors + t];
    const float* targets = (const float*)all_ptrs[10u * num_tensors + t];

    float local_provisional = 0.0f;
    float local_target = 0.0f;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float update = updates[idx];
        const float target = targets[idx];
        local_provisional += update * update;
        local_target += target * target;
    }

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_provisional += __shfl_down_sync(0xffffffff, local_provisional, offset);
        local_target += __shfl_down_sync(0xffffffff, local_target, offset);
    }
    if (lane == 0u) {
        smem[warp] = local_provisional;
        smem[4u + warp] = local_target;
    }
    __syncthreads();

    if (warp == 0u) {
        float block_provisional = lane < 4u ? smem[lane] : 0.0f;
        float block_target = lane < 4u ? smem[4u + lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_provisional += __shfl_down_sync(0xffffffff, block_provisional, offset);
            block_target += __shfl_down_sync(0xffffffff, block_target, offset);
        }
        if (lane == 0u) {
            provisional_partials[b] = block_provisional;
            target_partials[b] = block_target;
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 6) void update_energy_commit_fused_t26(
    const unsigned long long* all_ptrs,
    const float* energy_scales,
    const int* meta,
    const unsigned int num_tensors
) {
    const unsigned int b = blockIdx.x, t = t26_tensor_of_block(meta, num_tensors, b);
    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk = (unsigned int)meta[t + 1] - first;
    const unsigned int n = (unsigned int)meta[num_tensors + 1u + t];
    const unsigned int tid = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;
    float* updates = (float*)all_ptrs[9u * num_tensors + t];
    const float scale = energy_scales[t];
    for (unsigned int idx = tid; idx < n; idx += stride) {
        updates[idx] *= scale;
    }
}

extern "C" __global__ void scale_additive_kernel_t26(const float* params, float* updates, const unsigned int n, const float scale_minus_1) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) updates[idx] += scale_minus_1 * params[idx];
}

extern "C" __global__ void dc_axpy_t26(float* dst, const float* src, const int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) dst[i] += src[i];
}
