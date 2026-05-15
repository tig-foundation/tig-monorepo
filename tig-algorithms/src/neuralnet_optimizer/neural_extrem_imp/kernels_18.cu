
extern "C" __global__ __launch_bounds__(128, 6) void sign_ef_consensus_kernel_18(
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
    const float rel_update_cap,
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
    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 0.025f;
    const float inv_lr = 1.0f / fmaxf(lr, 1.0e-8f);
    const float wd_lr = lr * weight_decay;
    const float abs_floor = 1.0e-3f;
    const float min_step = 0.18f * lr;
    const float ef_cap = 6.0f * lr;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fisher_diag[idx];
        const float g = gradients[idx];
        const float w = params[idx];

        // Tighter clipping for deep net stability
        const float fd_std = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g, -3.5f * fd_std), 3.5f * fd_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        fisher_diag[idx] = fd;

        const float rms = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_n = g_clipped / rms;

        float ef_old = ef_residual[idx];
        ef_old = fminf(fmaxf(ef_old, -ef_cap), ef_cap);
        const float combined = g_n + ef_old * inv_lr;
        const float u_quant = -lr * copysignf(1.0f, combined);
        const float ef_new = ef_old - u_quant;
        ef_residual[idx] = fminf(fmaxf(ef_new, -ef_cap), ef_cap);

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

        slow_update[idx] = su;
        updates[idx] = adj_update;
    }
}

extern "C" __global__ __launch_bounds__(128, 6) void dual_consensus_fisher_kernel_18(
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
    const float rel_update_cap,
    const float mom_decay
) {
    __shared__ float smem[16];

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int n_warps = (blockDim.x + 31) >> 5;

    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 0.025f;
    // Deeper net (18 hidden): slightly more ortho to stabilize
    const float ortho_mix = fminf(fmaxf(0.24f + 0.38f * blend_sign, 0.0f), 0.60f);

    // Pass 1: per-block gradient RMS and weight RMS for block-wise LARS adaptation
    float local_g_sq = 0.0f;
    float local_w_sq = 0.0f;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float w = params[idx];
        local_g_sq += g * g;
        local_w_sq += w * w;
        local_count++;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_g_sq += __shfl_down_sync(0xffffffff, local_g_sq, offset);
        local_w_sq += __shfl_down_sync(0xffffffff, local_w_sq, offset);
        local_count  += __shfl_down_sync(0xffffffff, local_count,  offset);
    }
    if (lane == 0) {
        smem[warp_id] = (local_count > 0) ? (local_g_sq / (float)local_count) : 0.0f;
        smem[warp_id + 8] = (local_count > 0) ? (local_w_sq / (float)local_count) : 0.0f;
    }
    __syncthreads();

    float block_grad_rms = 0.0f;
    float block_weight_rms = 0.0f;
    if (warp_id == 0) {
        float val_g = (lane < n_warps) ? smem[lane] : 0.0f;
        float val_w = (lane < n_warps) ? smem[lane + 8] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            val_g += __shfl_down_sync(0xffffffff, val_g, offset);
            val_w += __shfl_down_sync(0xffffffff, val_w, offset);
        }
        if (lane == 0) {
            smem[0] = sqrtf(fmaxf(val_g / fmaxf((float)n_warps, 1.0f), 0.0f)) + eps;
            smem[1] = sqrtf(fmaxf(val_w / fmaxf((float)n_warps, 1.0f), 0.0f)) + eps;
        }
    }
    __syncthreads();
    block_grad_rms = smem[0];
    block_weight_rms = smem[1];
    const float inv_block_rms = 1.0f / fmaxf(block_grad_rms, 1.0e-8f);
    
    // Scale the base learning rate by the weight magnitude (Block-wise LARS-lite)
    // Avoids dividing by gradient norm since Adam already handles gradient scaling intrinsically
    const float lars_lr = lr * fminf(fmaxf(block_weight_rms, 0.4f), 2.5f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];
        const float w = params[idx];

        // Nesterov look-ahead with sign-aware gamma; deep nets need tighter damping on flip
        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.18f * nesterov_gamma);
        const float g_pred = g + gamma_local * (g - pg);

        float m = momentum[idx] * mom_decay;
        float v = velocity[idx];

        m = beta1 * m + one_minus_beta1 * g_pred;
        const float err = g_pred - m;
        v = beta2 * v + one_minus_beta2 * err * err;

        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;

        const float sqrt_v = sqrtf(fmaxf(v_hat, 0.0f));
        const float adaptive_eps = eps * (1.0f + 0.08f * sqrt_v);
        const float denom = sqrt_v + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        // Apply true AdamW decoupled weight decay instead of L2 regularization on gradients
        const float adam_update = -lars_lr * (m_hat * inv_denom);
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update = -lars_lr * g_over_denom;
        const float sign_update = -lars_lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update + blend_norm * norm_update + blend_sign * sign_update;

        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float g_ortho = g_pred - overlap;
        const float ortho_update = -lars_lr * (g_ortho * inv_denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;

        const float s_pu = prev_update[idx];
        const float s_mag = fabsf(s_pu);
        const float bb_scale = (s_mag > 1e-6f) ? fminf(s_mag * 2.0f, 2.2f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;

        // Fisher diagonal with raw-velocity-based clipping
        float fd = fisher_diag[idx];
        const float sqrt_v_for_clip = sqrtf(fmaxf(v, 0.0f));
        const float grad_std = sqrt_v_for_clip + eps;
        // Slightly tighter clip for deep net (4.5 vs 5)
        const float g_clipped = fminf(fmaxf(g_pred, -4.5f * grad_std), 4.5f * grad_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        const float fisher_rms = sqrtf(fmaxf(fd, 0.0f)) + eps;

        const float fisher_norm_update = -lars_lr * (g_pred / fisher_rms);
        // Deeper net: push even harder toward fisher-normalized track (0.35/0.65)
        const float robust_track = 0.35f * sign_update + 0.65f * fisher_norm_update;

        const float flip = (g * pg < 0.0f) ? 1.0f : 0.0f;
        const float vol = fminf(sqrt_v * 0.33333334f, 1.0f);
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        const float grad_mom_align = (g_pred * m_hat >= 0.0f) ? 1.0f : 0.0f;
        const float stability = grad_mom_align * (1.0f - flip);

        // Curvature-aware consensus
        const float g_abs = fabsf(g);
        const float pg_abs = fabsf(pg);
        const float curvature = fabsf(g - pg) / fmaxf(g_abs + pg_abs + eps, 1.0e-8f);
        const float curvature_clamped = fminf(curvature, 1.0f);

        float consensus_mix = 0.28f * curvature_clamped
                            + 0.25f * (1.0f - agree)
                            + 0.20f * vol
                            + 0.18f * blend_sign
                            + 0.12f * (1.0f - stability);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;

        // Enhanced trust with alignment boost
        const float align_strength = grad_mom_align * (1.0f - flip) * (1.0f - vol);
        const float trust = (1.0f + 0.10f * align_strength) / (1.0f + 0.55f * vol + 0.55f * flip);
        chosen_update *= trust;

        // Block-level element normalization: dampen big outliers vs block avg
        const float elem_rms = g_abs * inv_block_rms;
        const float block_norm_scale = 1.0f / fmaxf(0.6f + 0.4f * elem_rms, 1.0e-8f);
        chosen_update *= block_norm_scale;

        // Target gate against adam's norm_update magnitude
        const float target = lars_lr * fabsf(g_over_denom);
        const float uabs = fabsf(chosen_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        chosen_update *= scale;

        float su = slow_update[idx] * mom_decay;
        su = one_minus_la_tau * su + lookahead_tau * chosen_update;
        float final_update = one_minus_la_alpha * chosen_update + lookahead_alpha * su;

        // Decoupled weight decay (AdamW)
        final_update -= lars_lr * weight_decay * w;

        // LAMB-style relative update capping
        const float abs_floor = 1.0e-3f;
        const float min_step = 0.18f * lars_lr;
        const float max_step = fmaxf(rel_update_cap * (fabsf(w) + abs_floor), min_step);
        final_update = fminf(fmaxf(final_update, -max_step), max_step);

        momentum[idx] = m;
        velocity[idx] = v;
        prev_grad[idx] = g;
        prev_update[idx] = final_update;
        slow_update[idx] = su;
        fisher_diag[idx] = fd;
        updates[idx] = final_update;
    }
}
