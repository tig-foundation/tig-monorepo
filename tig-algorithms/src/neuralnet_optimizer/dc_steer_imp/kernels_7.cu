extern "C" __global__ __launch_bounds__(256, 4) void grad_norm_reduction_kernel_7(
    const float* __restrict__ gradients,
    float* __restrict__ norm_out,
    const unsigned int n
) {
    __shared__ float smem[8]; 
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;

    float local_sq = 0.0f;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        local_sq += g * g;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
    if (lane == 0) smem[warp_id] = local_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(norm_out, sqrtf(fmaxf(val, 0.0f)));
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void grad_norm_fused_1blk_7(
    const unsigned long long* __restrict__ g_ptrs,   
    const int* __restrict__ lens,                    
    const int* __restrict__ slots,                   
    float* __restrict__ norm_out,
    const unsigned int num_tensors
) {
    const unsigned int t = blockIdx.x;
    if (t >= num_tensors) return;
    const float* __restrict__ gradients = (const float*)g_ptrs[t];
    const unsigned int n = (unsigned int)lens[t];

    __shared__ float smem[8];
    const unsigned int tid = threadIdx.x;              
    const unsigned int stride = blockDim.x;            
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;

    float local_sq = 0.0f;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        local_sq += g * g;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
    if (lane == 0) smem[warp_id] = local_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(norm_out + slots[t], sqrtf(fmaxf(val, 0.0f)));
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void svrg_variance_reduce_kernel_7(
    const float* __restrict__ gradients,
    const float* __restrict__ ef_residual,
    float* __restrict__ ef_mean_out,
    float* __restrict__ vr_grad_out,
    const unsigned int n,
    const float blend
) {
    __shared__ float smem[8];
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;

    float local_sum = 0.0f;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        local_sum += ef_residual[idx];
        local_count++;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    if (lane == 0) smem[warp_id] = local_sum / fmaxf((float)local_count, 1.0f);
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) smem[0] = val / 8.0f;
    }
    __syncthreads();
    const float mean_ef = smem[0];

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float ef = ef_residual[idx];
        vr_grad_out[idx] = g - blend * (ef - mean_ef);
    }
}

__device__ __forceinline__ unsigned int t30_tensor_of_block(
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

__device__ __forceinline__ void svrg_body_7(
    const float* __restrict__ gradients,
    const float* __restrict__ ef_residual,
    float* __restrict__ vr_grad_out,
    const unsigned int n,
    const float blend,
    const unsigned int tid,
    const unsigned int stride
) {
    __shared__ float smem[8];
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;

    float local_sum = 0.0f;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        local_sum += ef_residual[idx];
        local_count++;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    if (lane == 0) smem[warp_id] = local_sum / fmaxf((float)local_count, 1.0f);
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) smem[0] = val / 8.0f;
    }
    __syncthreads();
    const float mean_ef = smem[0];

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float ef = ef_residual[idx];
        vr_grad_out[idx] = g - blend * (ef - mean_ef);
    }
}

extern "C" __global__ __launch_bounds__(256, 4) void svrg_variance_reduce_fused_7(
    const unsigned long long* __restrict__ all_ptrs,   
    const int* __restrict__ meta,                      
    const unsigned int num_tensors,
    const float blend
) {
    const unsigned int b     = blockIdx.x;
    const unsigned int t     = t30_tensor_of_block(meta, num_tensors, b);
    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];
    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;
    svrg_body_7((const float*)all_ptrs[0u * num_tensors + t],
                (const float*)all_ptrs[1u * num_tensors + t],
                (float*)      all_ptrs[2u * num_tensors + t],
                n, blend, tid, stride);
}


__device__ __forceinline__ void sign_ef_consensus_body_7(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ fisher_diag,
    float* __restrict__ ef_residual,
    float* __restrict__ slow_update,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float eps,
    const float weight_decay,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi,
    const unsigned int tid,
    const unsigned int stride
) {
    __shared__ float smem[24];

    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.95f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;

    float local_gradient_sum = 0.0f;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        local_gradient_sum += gradients[idx];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_gradient_sum += __shfl_down_sync(0xffffffff, local_gradient_sum, offset);
    }
    if (lane == 0) smem[warp_id] = local_gradient_sum;
    __syncthreads();

    if (warp_id == 0) {
        float gradient_sum = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, offset);
        }
        if (lane == 0) smem[0] = gradient_sum;
    }
    __syncthreads();
    const float shared_direction = smem[0];

    float local_reservoir = 0.0f;
    unsigned int local_disagreement_count = 0;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fmaxf(fisher_diag[idx], 1.0e-8f);
        const float g = gradients[idx];
        const float fd_std = sqrtf(fd) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.0f * fd_std), 4.0f * fd_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        fd = fmaxf(fd, 1.0e-8f);
        fisher_diag[idx] = fd;

        const float g_n = g / (sqrtf(fd) + eps);
        const float u_quant = -lr * copysignf(1.0f, g_n);
        const float residual = (-lr * g_n + ef_residual[idx]) - u_quant;
        ef_residual[idx] = residual;
        updates[idx] = u_quant;

        local_reservoir += residual;
        local_disagreement_count += (g * shared_direction < 0.0f);
        local_count++;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_reservoir += __shfl_down_sync(0xffffffff, local_reservoir, offset);
        local_disagreement_count +=
            __shfl_down_sync(0xffffffff, local_disagreement_count, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    if (lane == 0) {
        smem[warp_id] = local_reservoir;
        smem[8 + warp_id] = (float)local_disagreement_count;
        smem[16 + warp_id] = (float)local_count;
    }
    __syncthreads();

    if (warp_id == 0) {
        float reservoir = (lane < 8) ? smem[lane] : 0.0f;
        float disagreement_count = (lane < 8) ? smem[8 + lane] : 0.0f;
        float total_count = (lane < 8) ? smem[16 + lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            reservoir += __shfl_down_sync(0xffffffff, reservoir, offset);
            disagreement_count += __shfl_down_sync(0xffffffff, disagreement_count, offset);
            total_count += __shfl_down_sync(0xffffffff, total_count, offset);
        }
        if (lane == 0) {
            const float recipient_count =
                disagreement_count > 0.0f ? disagreement_count : total_count;
            smem[0] = reservoir / fmaxf(recipient_count, 1.0f);
            smem[8] = disagreement_count;
        }
    }
    __syncthreads();

    const float reservoir_share = smem[0];
    const bool has_disagreement = smem[8] > 0.0f;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const bool receives_reservoir =
            has_disagreement ? (g * shared_direction < 0.0f) : true;
        const float correction = receives_reservoir ? reservoir_share : 0.0f;
        ef_residual[idx] -= correction;

        const float g_n = g / (sqrtf(fmaxf(fisher_diag[idx], 1.0e-8f)) + eps);
        const float corrected_quantized = updates[idx] + correction;
        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * corrected_quantized;
        const float final_update =
            one_minus_la_alpha * corrected_quantized + lookahead_alpha * su;
        const float target = lr * fabsf(g_n);
        const float scale = fminf(
            fmaxf(target / fmaxf(fabsf(final_update), 1.0e-12f), gate_lo),
            gate_hi
        );
        slow_update[idx] = su;
        updates[idx] = final_update * scale - (lr * weight_decay) * params[idx];
    }
}

__device__ __forceinline__ void dual_consensus_fisher_body_7(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ velocity,
    float* __restrict__ prev_grad,
    float* __restrict__ prev_update,
    float* __restrict__ slow_update,
    float* __restrict__ fisher_diag,
    float* __restrict__ updates,
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
    const float tr_hi,
    const float bns_mix,
    const int innovation_only,
    float* __restrict__ block_partials,
    const unsigned int partial_slot,
    const unsigned int tid,
    const unsigned int stride
) {
    __shared__ float smem[56];

    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;

    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.95f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float ortho_mix = fminf(fmaxf(0.2f + 0.4f * blend_sign, 0.0f), 0.6f);

    float local_sq_sum = 0.0f;
    float local_signed_sum = 0.0f;
    float local_abs_sum = 0.0f;
    float local_param_dot = local_sq_sum - local_sq_sum;
    float local_param_sq = local_sq_sum - local_sq_sum;
    float local_update_sq = 0.0f;
    float local_param_update_dot = 0.0f;
    float local_gradient_update_dot = 0.0f;
    float local_gradient_sq = 0.0f;
    float local_update_sum = 0.0f;
    float local_gradient_sum = 0.0f;
    float local_parameter_sum = 0.0f;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float p = params[idx];
        local_sq_sum += g * g;
        local_signed_sum += g;
        local_abs_sum += fabsf(g);
        local_param_dot += p * g;
        local_param_sq += p * p;
        local_count++;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
        local_signed_sum += __shfl_down_sync(0xffffffff, local_signed_sum, offset);
        local_abs_sum += __shfl_down_sync(0xffffffff, local_abs_sum, offset);
        local_param_dot += __shfl_down_sync(0xffffffff, local_param_dot, offset);
        local_param_sq += __shfl_down_sync(0xffffffff, local_param_sq, offset);
        local_count  += __shfl_down_sync(0xffffffff, local_count,  offset);
    }
    if (lane == 0) {
        smem[warp_id] = local_sq_sum / fmaxf((float)local_count, 1.0f);
        smem[8 + warp_id] = local_signed_sum;
        smem[16 + warp_id] = local_abs_sum;
        smem[24 + warp_id] = local_param_dot;
        smem[32 + warp_id] = local_param_sq;
    }
    __syncthreads();

    float block_grad_rms = 0.0f;
    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        float signed_sum = (lane < 8) ? smem[8 + lane] : 0.0f;
        float abs_sum = (lane < 8) ? smem[16 + lane] : 0.0f;
        float param_dot = (lane < 8) ? smem[24 + lane] : val - val;
        float param_sq = (lane < 8) ? smem[32 + lane] : val - val;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
            signed_sum += __shfl_down_sync(0xffffffff, signed_sum, offset);
            abs_sum += __shfl_down_sync(0xffffffff, abs_sum, offset);
            param_dot += __shfl_down_sync(0xffffffff, param_dot, offset);
            param_sq += __shfl_down_sync(0xffffffff, param_sq, offset);
        }
        if (lane == 0) {
            smem[0] = sqrtf(fmaxf(val / 8.0f, 0.0f)) + eps;
            smem[8] = fabsf(signed_sum) / fmaxf(abs_sum, eps);
            smem[24] = param_dot / fmaxf(param_sq, eps);
            smem[32] = param_sq;
        }
    }
    __syncthreads();
    block_grad_rms = smem[0];
    const float block_directional_consensus = smem[8];
    const float radial_gradient = smem[24];
    const float block_parameter_sq = smem[32];
    const float inv_block_rms = 1.0f / fmaxf(block_grad_rms, 1.0e-8f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx] - params[idx] * radial_gradient;
        const float pg = prev_grad[idx];
        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.25f * nesterov_gamma);
        const float g_pred = innovation_only ? g : g + gamma_local * (g - pg);
        float m = innovation_only ? g * bias_correction1 : momentum[idx];
        float v = velocity[idx];
        if (!innovation_only) m = beta1 * m + one_minus_beta1 * g_pred;
        const float err = g_pred - m;
        v = beta2 * v + one_minus_beta2 * err * err;
        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;
        const float sqrt_v = sqrtf(fmaxf(v_hat, 0.0f));
        const float adaptive_eps = eps * (1.0f + 0.1f * sqrt_v);
        const float denom = sqrt_v + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);
        const float adam_update = -lr * (m_hat * inv_denom);
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update = -lr * g_over_denom;
        const float sign_update = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update + blend_norm * norm_update + blend_sign * sign_update;
        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float g_ortho = g_pred - overlap;
        const float ortho_update = -lr * (g_ortho / denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;
        const float s = prev_update[idx];
        const float s_mag = fabsf(s);
        const float bb_scale = (s_mag > 1e-6f) ? fminf(s_mag * 2.0f, 2.5f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;
        float fd = fmaxf(fisher_diag[idx], 1.0e-8f);
        const float sqrt_v_for_clip = sqrtf(fmaxf(v, 0.0f));
        const float grad_std = sqrt_v_for_clip + eps;
        const float g_clipped = fminf(fmaxf(g_pred, -5.0f * grad_std), 5.0f * grad_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        fd = fmaxf(fd, 1.0e-8f);
        const float fisher_rms = sqrtf(fd) + eps;
        const float fisher_norm_update = -lr * (g_pred / fisher_rms);
        const float robust_track = 0.45f * sign_update + 0.55f * fisher_norm_update;
        const float vol = fminf(sqrt_v * 0.33333334f, 1.0f);
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        const float g_abs = fabsf(g);
        const float pg_abs = fabsf(pg);
        const float curvature = fabsf(g - pg) / fmaxf(g_abs + pg_abs + eps, 1.0e-8f);
        const float curvature_clamped = fminf(curvature, 1.0f);
        float consensus_mix = 0.3f * curvature_clamped + 0.2f * vol + 0.2f * blend_sign + 0.15f * (1.0f - block_directional_consensus) + 0.15f * (1.0f - agree);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;
        const float elem_rms = fabsf(g) * inv_block_rms;
        const float elem_rms_eff = bns_mix * elem_rms + (1.0f - bns_mix) * 0.61735f;
        const float block_norm_scale = 1.0f / fmaxf(0.5f + 0.5f * elem_rms_eff, 1.0e-8f);
        chosen_update *= block_norm_scale;
        const float target = lr * fabsf(g_over_denom);
        const float uabs = fabsf(chosen_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        chosen_update *= scale;
        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * chosen_update;
        const float final_update = one_minus_la_alpha * chosen_update + lookahead_alpha * su;
        const float local_gradient = gradients[idx];
        local_update_sq += final_update * final_update;
        local_param_update_dot += params[idx] * final_update;
        local_gradient_update_dot += local_gradient * final_update;
        local_gradient_sq += local_gradient * local_gradient;
        local_update_sum += final_update;
        local_gradient_sum += local_gradient;
        local_parameter_sum += params[idx];
        momentum[idx] = m;
        velocity[idx] = v;
        prev_grad[idx] = g;
        prev_update[idx] = final_update;
        slow_update[idx] = su;
        fisher_diag[idx] = fd;
        updates[idx] = final_update;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_update_sq += __shfl_down_sync(0xffffffff, local_update_sq, offset);
        local_param_update_dot += __shfl_down_sync(0xffffffff, local_param_update_dot, offset);
        local_gradient_update_dot += __shfl_down_sync(0xffffffff, local_gradient_update_dot, offset);
        local_gradient_sq += __shfl_down_sync(0xffffffff, local_gradient_sq, offset);
        local_update_sum += __shfl_down_sync(0xffffffff, local_update_sum, offset);
        local_gradient_sum += __shfl_down_sync(0xffffffff, local_gradient_sum, offset);
        local_parameter_sum += __shfl_down_sync(0xffffffff, local_parameter_sum, offset);
    }
    if (lane == 0) {
        smem[warp_id] = local_update_sq;
        smem[8 + warp_id] = local_param_update_dot;
        smem[16 + warp_id] = local_gradient_update_dot;
        smem[24 + warp_id] = local_gradient_sq;
        smem[32 + warp_id] = local_update_sum;
        smem[40 + warp_id] = local_gradient_sum;
        smem[48 + warp_id] = local_parameter_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float update_sq = (lane < 8) ? smem[lane] : 0.0f;
        float parameter_update_dot = (lane < 8) ? smem[8 + lane] : 0.0f;
        float gradient_update_dot = (lane < 8) ? smem[16 + lane] : 0.0f;
        float gradient_sq = (lane < 8) ? smem[24 + lane] : 0.0f;
        float update_sum = (lane < 8) ? smem[32 + lane] : 0.0f;
        float gradient_sum = (lane < 8) ? smem[40 + lane] : 0.0f;
        float parameter_sum = (lane < 8) ? smem[48 + lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            update_sq += __shfl_down_sync(0xffffffff, update_sq, offset);
            parameter_update_dot += __shfl_down_sync(0xffffffff, parameter_update_dot, offset);
            gradient_update_dot += __shfl_down_sync(0xffffffff, gradient_update_dot, offset);
            gradient_sq += __shfl_down_sync(0xffffffff, gradient_sq, offset);
            update_sum += __shfl_down_sync(0xffffffff, update_sum, offset);
            gradient_sum += __shfl_down_sync(0xffffffff, gradient_sum, offset);
            parameter_sum += __shfl_down_sync(0xffffffff, parameter_sum, offset);
        }
        if (lane == 0 && block_partials != 0) {
            block_partials[8u * partial_slot] = block_parameter_sq;
            block_partials[8u * partial_slot + 1u] = update_sq;
            block_partials[8u * partial_slot + 2u] = parameter_update_dot;
            block_partials[8u * partial_slot + 3u] = gradient_update_dot;
            block_partials[8u * partial_slot + 4u] = gradient_sq;
            block_partials[8u * partial_slot + 5u] = update_sum;
            block_partials[8u * partial_slot + 6u] = gradient_sum;
            block_partials[8u * partial_slot + 7u] = parameter_sum;
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 3) void sign_ef_consensus_kernel_7(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ fisher_diag,
    float* __restrict__ ef_residual,
    float* __restrict__ slow_update,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float eps,
    const float weight_decay,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    sign_ef_consensus_body_7(gradients, params, fisher_diag, ef_residual, slow_update,
                             updates, n, lr, eps, weight_decay, lookahead_alpha,
                             lookahead_tau, gate_lo, gate_hi, tid, stride);
}

extern "C" __global__ __launch_bounds__(256, 3) void dual_consensus_fisher_kernel_7(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ velocity,
    float* __restrict__ prev_grad,
    float* __restrict__ prev_update,
    float* __restrict__ slow_update,
    float* __restrict__ fisher_diag,
    float* __restrict__ updates,
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
    const float gate_hi
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    dual_consensus_fisher_body_7(gradients, params, momentum, velocity, prev_grad,
                                 prev_update, slow_update, fisher_diag, updates,
                                 n, lr, beta1, beta2, eps, weight_decay,
                                 bias_correction1, bias_correction2, blend_adam,
                                 blend_norm, blend_sign, nesterov_gamma, bb_blend,
                                 lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
                                 2.0f, 1.0f, false, 0, 0u, tid, stride);
}

extern "C" __global__ __launch_bounds__(256, 3) void sign_ef_consensus_fused_7(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ scal,
    const int* __restrict__ meta,
    const unsigned int num_tensors,
    const float eps,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    const unsigned int b = blockIdx.x;
    const unsigned int t = t30_tensor_of_block(meta, num_tensors, b);

    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];

    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr           = scal[0u * num_tensors + t];
    const float weight_decay = scal[1u * num_tensors + t];

    sign_ef_consensus_body_7(
        (const float*)all_ptrs[0u * num_tensors + t],   
        (const float*)all_ptrs[1u * num_tensors + t],   
        (float*)all_ptrs[7u * num_tensors + t],         
        (float*)all_ptrs[8u * num_tensors + t],         
        (float*)all_ptrs[6u * num_tensors + t],         
        (float*)all_ptrs[9u * num_tensors + t],         
        n, lr, eps, weight_decay, lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
        tid, stride);
}

extern "C" __global__ __launch_bounds__(256, 3) void dual_consensus_fisher_fused_7(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ scal,
    const int* __restrict__ meta,
    const unsigned int num_tensors,
    const float beta1,
    const float beta2,
    const float eps,
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
    const float tr_hi,
    const float bns_mix,
    float* __restrict__ block_partials
) {
    const unsigned int b = blockIdx.x;
    const unsigned int t = t30_tensor_of_block(meta, num_tensors, b);

    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];

    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr           = scal[0u * num_tensors + t];
    const float weight_decay = scal[1u * num_tensors + t];

    dual_consensus_fisher_body_7(
        (const float*)all_ptrs[0u * num_tensors + t],   
        (const float*)all_ptrs[1u * num_tensors + t],   
        (float*)all_ptrs[2u * num_tensors + t],         
        (float*)all_ptrs[3u * num_tensors + t],         
        (float*)all_ptrs[4u * num_tensors + t],         
        (float*)all_ptrs[5u * num_tensors + t],         
        (float*)all_ptrs[6u * num_tensors + t],         
        (float*)all_ptrs[7u * num_tensors + t],         
        (float*)all_ptrs[9u * num_tensors + t],         
        n, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
        blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend,
        lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
        tr_hi, bns_mix, (int)scal[2u * num_tensors + t],
        block_partials, b, tid, stride);
}

extern "C" __global__ __launch_bounds__(256, 3) void tensorwise_rayleigh_scale_fused_7(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ scal,
    const int* __restrict__ meta,
    const float* __restrict__ rayleigh_scales,
    const unsigned int num_tensors
) {
    const unsigned int b = blockIdx.x;
    const unsigned int t = t30_tensor_of_block(meta, num_tensors, b);
    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk = (unsigned int)meta[t + 1] - first;
    const unsigned int n = (unsigned int)meta[num_tensors + 1u + t];
    const unsigned int tid = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;
    const float lr = scal[t];
    const float weight_decay = scal[num_tensors + t];
    const float differential_scale = rayleigh_scales[t];
    const float common_mode = rayleigh_scales[num_tensors + t];
    const float radial_projection = rayleigh_scales[2u * num_tensors + t];
    const float affine_bias_transport = rayleigh_scales[3u * num_tensors + t];
    const float* __restrict__ params = (const float*)all_ptrs[1u * num_tensors + t];
    float* __restrict__ prev_update = (float*)all_ptrs[5u * num_tensors + t];
    float* __restrict__ updates = (float*)all_ptrs[9u * num_tensors + t];

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float differential_update = updates[idx] - common_mode;
        const float tangent_update =
            differential_update - radial_projection * params[idx];
        const float final_update =
            differential_scale * tangent_update + common_mode -
            (lr * weight_decay) * params[idx] - affine_bias_transport;
        updates[idx] = final_update;
        prev_update[idx] = final_update;
    }
}

extern "C" __global__ void dc_axpy_t30(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] += src[i];
    }
}