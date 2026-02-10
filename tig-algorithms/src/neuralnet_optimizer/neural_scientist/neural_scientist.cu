extern "C" __global__ __launch_bounds__(512, 4) void sign_ef_consensus_kernel(
    const float* __restrict__ gradients,
    float* __restrict__ fisher_diag,
    float* __restrict__ ef_residual,
    float* __restrict__ slow_update,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float eps,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const unsigned int stride = blockDim.x * gridDim.x;
    
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float inv_lr = 1.0f / fmaxf(lr, 1.0e-8f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fisher_diag[idx];
        const float g = gradients[idx];
        fd = fisher_beta * fd + one_minus_fisher_beta * g * g;
        fisher_diag[idx] = fd;

        const float rms = sqrtf(fd) + eps;
        const float g_n = g / rms;

        const float ef_old = ef_residual[idx];
        const float u_desired = -lr * g_n + ef_old;

        const float u_quant = -lr * copysignf(1.0f, g_n + ef_old * inv_lr);

        float ef_delta = u_desired - u_quant;
        const float ef_cap = 6.0f * lr;
        ef_delta = ef_delta / (1.0f + fabsf(ef_delta) / fmaxf(ef_cap, 1.0e-8f));
        const float ef_new = 0.98f * ef_old + ef_delta;
        ef_residual[idx] = ef_new;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * u_quant;
        const float final_update = one_minus_la_alpha * u_quant + lookahead_alpha * su;

        const float target = lr * fabsf(g_n);
        const float uabs = fabsf(final_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        const float adj_update = final_update * scale;

        slow_update[idx] = su;
        updates[idx] = adj_update;
    }
}

extern "C" __global__ __launch_bounds__(512, 4) void dual_consensus_fisher_kernel(
    const float* __restrict__ gradients,
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
    
    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float ortho_mix = fminf(fmaxf(0.2f + 0.4f * blend_sign, 0.0f), 0.6f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];
        
        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.25f * nesterov_gamma);
        const float g_pred = g + gamma_local * (g - pg);

        float m = momentum[idx];
        float v = velocity[idx];
        
        m = beta1 * m + one_minus_beta1 * g_pred;
        
        const float err = g_pred - m;
        v = beta2 * v + one_minus_beta2 * err * err;

        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;

        const float sqrt_v = sqrtf(fmaxf(v_hat, 0.0f));
        const float adaptive_eps = eps * (1.0f + 0.1f * sqrt_v);
        const float denom = sqrt_v + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);
        
        const float adam_update = -lr * (m_hat * inv_denom + weight_decay * g_pred);
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update = -lr * g_over_denom;
        const float sign_update = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update
                          + blend_norm * norm_update
                          + blend_sign * sign_update;
        
        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float g_ortho = g_pred - overlap;
        const float ortho_update = -lr * (g_ortho / denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;

        const float y = g - pg;                   
        const float s = prev_update[idx];         
        const float bb_raw = fabsf(s) / (fabsf(y) + 1.0e-8f);
        const float bb_scale = fminf(fmaxf(bb_raw, 0.2f), 5.0f);
        base_update *= (1.0f - bb_blend) + bb_blend * bb_scale;
        
        float fd = fisher_diag[idx];
        fd = fisher_beta * fd + one_minus_fisher_beta * g_pred * g_pred;
        const float fisher_rms = sqrtf(fd) + eps;
        
        const float fisher_norm_update = -lr * (g_pred / fisher_rms);
        const float robust_track = 0.5f * sign_update + 0.5f * fisher_norm_update;

        const float flip = (g * pg < 0.0f) ? 1.0f : 0.0f;     
        const float vol = fminf(sqrt_v * 0.33333334f, 1.0f);         
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        float consensus_mix = 0.4f * (1.0f - agree) + 0.3f * vol + 0.3f * blend_sign;
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float mixed_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;
        
        const float s1 = g_pred * mixed_update;
        const float s2 = g_pred * robust_track;
        float chosen_update = (s2 < s1) ? robust_track : mixed_update;
        
        float trust = 1.0f / (1.0f + 0.5f * vol + 0.5f * flip);
        chosen_update *= trust;

        const float target = lr * fabsf(g_over_denom);
        const float uabs = fabsf(chosen_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        chosen_update *= scale;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * chosen_update;
        const float final_update = one_minus_la_alpha * chosen_update + lookahead_alpha * su;

        momentum[idx] = m;
        velocity[idx] = v;
        prev_grad[idx] = g;               
        prev_update[idx] = final_update;  
        slow_update[idx] = su;            
        fisher_diag[idx] = fd;            
        updates[idx] = final_update;
    }
}
