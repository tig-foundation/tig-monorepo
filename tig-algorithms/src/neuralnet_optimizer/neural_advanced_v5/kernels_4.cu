extern "C" __global__ __launch_bounds__(256, 3) void tensor_sqnorms_block_kernel(
    const float* __restrict__ params,
    const float* __restrict__ updates,
    float* __restrict__ block_sqnorms,
    const unsigned int n
) {
    const unsigned int tid = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);

    __shared__ float sh_param[256];
    __shared__ float sh_update[256];

    float local_param_sq = 0.0f;
    float local_update_sq = 0.0f;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float p = params[idx];
        const float u = updates[idx];
        local_param_sq += p * p;
        local_update_sq += u * u;
    }

    sh_param[threadIdx.x]  = local_param_sq;
    sh_update[threadIdx.x] = local_update_sq;
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh_param[threadIdx.x]  += sh_param[threadIdx.x + offset];
            sh_update[threadIdx.x] += sh_update[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_sqnorms[2u * blockIdx.x]      = sh_param[0];
        block_sqnorms[2u * blockIdx.x + 1u] = sh_update[0];
    }
}

extern "C" __global__ __launch_bounds__(256, 3) void scale_updates_kernel(
    float* __restrict__ updates,
    const unsigned int n,
    const float scale
) {    
    const unsigned int tid = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    if (tid >= n) return;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        updates[idx] *= scale;
    }
}

extern "C" __global__ __launch_bounds__(256, 3) void sign_ef_consensus_kernel(
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
    const unsigned int tid = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);

    __shared__ float sh_signed[256];
    __shared__ float sh_mass[256];

    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;

    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;

    const float lr_safe = fmaxf(lr, 1.0e-12f);
    const float inv_lr = 1.0f / lr_safe;
    const float ef_decay = 0.990f;

    float local_signed = 0.0f;
    float local_mass = 0.0f;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float fd = fisher_diag[idx];
        const float fd_floor = fmaxf(fd, 1.0e-4f);
        const float fd_std = sqrtf(fd_floor) + eps;
        const float g_clip = fminf(fmaxf(g, -4.0f * fd_std), 4.0f * fd_std);
        const float rms = sqrtf(fd_floor) + eps;
        const float g_n = g_clip / rms;
        const float drive = g_n + ef_residual[idx] * inv_lr;
        const float drive_soft = drive / (1.0f + fabsf(drive));
        local_signed += drive_soft;
        local_mass += fabsf(drive_soft);
    }

    sh_signed[threadIdx.x] = local_signed;
    sh_mass[threadIdx.x] = local_mass;
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh_signed[threadIdx.x] += sh_signed[threadIdx.x + offset];
            sh_mass[threadIdx.x] += sh_mass[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const float block_coherence =
        fabsf(sh_signed[0]) / fmaxf(sh_mass[0], 1.0e-12f);
    const float sign_blend = fminf(fmaxf(block_coherence, 0.0f), 1.0f);

    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fisher_diag[idx];
        const float g = gradients[idx];

        const float fd_floor = fmaxf(fd, 1.0e-4f);
        const float fd_std = sqrtf(fd_floor) + eps;
        const float g_clip = fminf(fmaxf(g, -4.0f * fd_std), 4.0f * fd_std);

        fd = fisher_beta * fd + one_minus_fisher_beta * (g_clip * g_clip);
        fisher_diag[idx] = fd;

        const float rms = sqrtf(fmaxf(fd, 1.0e-12f)) + eps;
        const float g_n = g_clip / rms;

        const float ef_old = ef_residual[idx];
        const float drive = g_n + ef_old * inv_lr;
        const float u_desired = -lr * g_n + ef_old;
        const float u_quant = -lr * copysignf(1.0f, drive);
        const float u_consensus =
            sign_blend * u_quant + (1.0f - sign_blend) * u_desired;

        float ef_delta = u_desired - u_consensus;
        const float ef_cap = 10.0f * lr_safe;
        ef_delta = ef_delta / (1.0f + fabsf(ef_delta) / fmaxf(ef_cap, 1.0e-12f));
        ef_residual[idx] = ef_decay * ef_old + ef_delta;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * u_consensus;
        float u = one_minus_la_alpha * u_consensus + lookahead_alpha * su;

        const float target = lr * fabsf(g_n);
        const float uabs = fabsf(u);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        u *= scale;

        u -= (lr * weight_decay) * params[idx];

        slow_update[idx] = su;
        updates[idx] = u;
    }
}

extern "C" __global__ __launch_bounds__(256, 3) void dual_consensus_fisher_kernel(
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
    const unsigned int tid = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    if (tid >= n) return;

    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-12f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-12f);

    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;

    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;

    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;

    const float lr_abs = fmaxf(fabsf(lr), 1.0e-12f);

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];
        const float g_pred = g + nesterov_gamma * (g - pg);

        float m = momentum[idx];
        float v = velocity[idx];

        m = beta1 * m + one_minus_beta1 * g_pred;
        const float innovation = g_pred - m;
        v = beta2 * v + one_minus_beta2 * (innovation * innovation);

        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;

        const float denom = sqrtf(fmaxf(v_hat, 0.0f)) + eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        const float g_slow = 0.97f * pg + 0.03f * g;
        const float g_consensus = m_hat + 0.25f * (m_hat - g_slow * inv_denom);

        const float adam_u = -lr * (g_consensus * inv_denom);
        const float norm_u = -lr * (g_pred * inv_denom);
        const float sign_u = -lr * copysignf(1.0f, m_hat);

        const float base_u = blend_adam * adam_u + blend_norm * norm_u + blend_sign * sign_u;

        float fd = fisher_diag[idx];
        const float fd_floor = fmaxf(fd, 1.0e-4f);
        const float fd_std = sqrtf(fd_floor) + eps;
        const float g_clip = fminf(fmaxf(g_pred, -5.0f * fd_std), 5.0f * fd_std);

        fd = fisher_beta * fd + one_minus_fisher_beta * (g_clip * g_clip);
        fisher_diag[idx] = fd;

        const float fisher_rms = sqrtf(fmaxf(fd, 1.0e-12f)) + eps;
        const float fisher_u = -lr * (g_pred / fisher_rms);
        const float robust_u = 0.55f * sign_u + 0.45f * fisher_u;

        const float agree = (base_u * robust_u >= 0.0f) ? 1.0f : 0.0f;
        const float vol = fminf(sqrtf(fmaxf(v_hat, 0.0f)) * 0.5f, 1.0f);
        float mix = (1.0f - agree) * (0.25f + 0.25f * vol) + 0.20f * blend_sign;
        mix = fminf(fmaxf(mix, 0.0f), 1.0f);
        float u = (1.0f - mix) * base_u + mix * robust_u;

        const float s = prev_update[idx];
        const float bb_scale = fminf(2.0f, 1.0f + fabsf(s) / lr_abs);
        u *= (1.0f - 0.18f * bb_blend) + (0.18f * bb_blend) * bb_scale;

        const float sign_agree = (g_pred * pg >= 0.0f) ? 1.0f : -1.0f;
        const float sign_trust = 1.0f + 0.12f * sign_agree;
        u *= sign_trust;

        const float target = lr * fabsf(g_pred * inv_denom);
        const float uabs = fabsf(u);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        u *= scale;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * u;
        float final_u = one_minus_la_alpha * u + lookahead_alpha * su;

        final_u -= (lr * weight_decay) * params[idx];

        momentum[idx] = m;
        velocity[idx] = v;
        prev_grad[idx] = g;
        prev_update[idx] = final_u;
        slow_update[idx] = su;
        updates[idx] = final_u;
    }
}