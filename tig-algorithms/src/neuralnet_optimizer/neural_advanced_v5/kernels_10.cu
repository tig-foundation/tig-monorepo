extern "C" __global__ __launch_bounds__(512, 4) void sign_ef_consensus_kernel_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ prev_grad,
    float* __restrict__ prev_step,
    float* __restrict__ fisher_diag,
    float* __restrict__ slow_fisher_diag,
    float* __restrict__ ef_residual,
    float* __restrict__ slow_update,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float slow_beta,
    const float eps,
    const float weight_decay,
    const float rel_update_cap,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    /* Robust path: rollback-capable Rprop with error feedback.
       prev_step stores the signed control step; on a gradient sign flip the kernel emits
       a corrective pulse that undoes the prior step, while the residual tracks any mismatch. */
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float one_minus_slow_beta = 1.0f - slow_beta;
    const float grow = fmaxf(gate_hi, 1.0f);
    const float shrink = fminf(gate_lo, 1.0f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fisher_diag[idx];
        float sd = slow_fisher_diag[idx];
        const float g = gradients[idx];
        const float pg = prev_grad[idx];
        const float w = params[idx];
        const float wabs = fabsf(w);
        const float su_prev = slow_update[idx];
        const float prev_signed_step = prev_step[idx];

        const float curv_std = sqrtf(fmaxf(fmaxf(fd, sd), 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.0f * curv_std), 4.0f * curv_std);

        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        sd = slow_beta * sd + one_minus_slow_beta * g_clipped * g_clipped;
        fisher_diag[idx] = fd;
        slow_fisher_diag[idx] = sd;

        const float fast_rms = sqrtf(fd) + eps;
        const float curvature_ratio = sqrtf((fd + eps) / (sd + eps));
        const float sharp_ratio = fmaxf(curvature_ratio, 1.0f);
        const float sharp_trust = 1.0f / sharp_ratio;

        const float g_n = g_clipped / fast_rms;
        const float target_update = -lr * g_n;

        const float abs_floor = 1.0e-3f;
        const float min_cap = 0.18f * lr;
        const float max_step = fmaxf(rel_update_cap * (wabs + abs_floor), min_cap);
        const float target_step = fminf(fabsf(target_update) * sharp_trust + eps, max_step);

        float step_state = fabsf(prev_signed_step);
        if (!(step_state > 0.0f)) {
            step_state = fmaxf(fminf(lr * sharp_trust, max_step), target_step);
        }

        const float sign_prod = g * pg;
        const bool rollback = (sign_prod < 0.0f) && (prev_signed_step != 0.0f);

        if (sign_prod > 0.0f) {
            step_state = fminf(step_state * grow, max_step);
        } else if (sign_prod < 0.0f) {
            step_state = fmaxf(step_state * shrink, target_step);
        } else {
            step_state = fminf(fmaxf(step_state, target_step), max_step);
        }

        const float sharp_cap = fmaxf(target_step, max_step * sharp_trust);
        step_state = fminf(fmaxf(step_state, target_step), sharp_cap);

        const float ef_scale = rollback ? fmaxf(fabsf(prev_signed_step), fmaxf(step_state, target_step))
                                        : fmaxf(step_state, target_step);
        const float ef_clip = 6.0f * ef_scale;
        float ef_old = ef_residual[idx];
        ef_old = fminf(fmaxf(ef_old, -ef_clip), ef_clip);

        const float corrected_target = target_update + ef_old;
        float u_quant = 0.0f;
        float stored_signed_step = 0.0f;
        float su = 0.0f;
        float adj_update = 0.0f;

        if (rollback) {
            u_quant = -prev_signed_step;
            stored_signed_step = copysignf(step_state, u_quant);

            const float ef_new = corrected_target - u_quant;
            ef_residual[idx] = fminf(fmaxf(ef_new, -ef_clip), ef_clip);

            su = one_minus_la_tau * su_prev;
            const float rollback_cap = fmaxf(fabsf(u_quant), sharp_cap);
            adj_update = u_quant - (lr * weight_decay) * w;
            adj_update = fminf(fmaxf(adj_update, -rollback_cap), rollback_cap);
        } else {
            if (corrected_target > 0.0f) {
                u_quant = step_state;
            } else if (corrected_target < 0.0f) {
                u_quant = -step_state;
            }

            const bool quant_cross = (w != 0.0f) && (w * (w + u_quant) < 0.0f);
            const bool established_weight = wabs > fmaxf(abs_floor, 0.5f * step_state);
            const bool strong_cross_evidence = fabsf(target_update) >= wabs;
            const bool persistent_grad_support = (g * w > 0.0f) && (pg * w > 0.0f);
            const bool persistent_update_support = (su_prev * u_quant > 0.0f) && (fabsf(su_prev) >= 0.5f * wabs);
            const bool repeated_cross_support = persistent_grad_support || persistent_update_support;

            if (quant_cross && established_weight && !strong_cross_evidence && !repeated_cross_support) {
                u_quant = -copysignf(0.5f * wabs, w);
            }

            stored_signed_step = u_quant;

            const float ef_new = corrected_target - u_quant;
            ef_residual[idx] = fminf(fmaxf(ef_new, -ef_clip), ef_clip);

            su = one_minus_la_tau * su_prev + lookahead_tau * u_quant;
            const float final_update = one_minus_la_alpha * u_quant + lookahead_alpha * su;

            const float target = fmaxf(fabsf(target_update) * sharp_trust, step_state);
            const float uabs = fabsf(final_update);
            const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
            adj_update = final_update * scale;

            adj_update -= (lr * weight_decay) * w;
            adj_update = fminf(fmaxf(adj_update, -sharp_cap), sharp_cap);

            const bool final_cross = (w != 0.0f) && (w * (w + adj_update) < 0.0f);
            if (final_cross && established_weight && !strong_cross_evidence && !repeated_cross_support) {
                adj_update = -copysignf(fminf(0.5f * wabs, sharp_cap), w);
            }
        }

        prev_grad[idx] = rollback ? 0.0f : g;
        prev_step[idx] = stored_signed_step;
        slow_update[idx] = su;
        updates[idx] = adj_update;
    }
}

extern "C" __global__ __launch_bounds__(512, 4) void dual_consensus_fisher_kernel_10(
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

    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float ortho_mix = fminf(fmaxf(0.18f + 0.42f * blend_sign, 0.0f), 0.58f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];

        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.20f * nesterov_gamma);
        const float g_pred = g + gamma_local * (g - pg);

        float m = momentum[idx];
        float slow_fd = velocity[idx];
        float fast_fd = fisher_diag[idx];

        const float slow_hat_pre = slow_fd * inv_bc2;
        const float clip_rms = sqrtf(fmaxf(fmaxf(slow_hat_pre, fast_fd), 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g_pred, -5.0f * clip_rms), 5.0f * clip_rms);

        m = beta1 * m + one_minus_beta1 * g_clipped;
        slow_fd = beta2 * slow_fd + one_minus_beta2 * g_clipped * g_clipped;
        fast_fd = fisher_beta * fast_fd + one_minus_fisher_beta * g_clipped * g_clipped;

        const float m_hat = m * inv_bc1;
        const float slow_hat = slow_fd * inv_bc2;
        const float slow_rms = sqrtf(fmaxf(slow_hat, 0.0f));
        const float fast_rms = sqrtf(fmaxf(fast_fd, 0.0f));
        const float adaptive_eps = eps * (1.0f + 0.08f * fast_rms);
        const float denom = slow_rms + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        const float curvature_ratio = sqrtf((fast_fd + eps) / (slow_fd + eps));
        const float sharp_ratio = fmaxf(curvature_ratio, 1.0f);
        const float sharpness = 1.0f - (1.0f / sharp_ratio);

        const float adam_update = -lr * (m_hat * inv_denom + weight_decay * g_clipped);
        const float g_over_denom = g_clipped * inv_denom;
        const float norm_update = -lr * g_over_denom;
        const float sign_update = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update
                          + blend_norm * norm_update
                          + blend_sign * sign_update;

        const float overlap = copysignf(fminf(fabsf(g_clipped), fabsf(m_hat)), m_hat);
        const float g_ortho = g_clipped - overlap;
        const float ortho_update = -lr * (g_ortho / denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;

        const float s_prev = prev_update[idx];
        const float s_mag = fabsf(s_prev);
        const float bb_scale = (s_mag > 1e-6f) ? fminf(s_mag * 2.0f, 2.5f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;

        const float fisher_norm_update = -lr * (g_clipped / (fast_rms + eps));
        const float robust_track = 0.5f * sign_update + 0.5f * fisher_norm_update;

        const float flip = (g * pg < 0.0f) ? 1.0f : 0.0f;
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        const float grad_mom_align = (g_clipped * m_hat >= 0.0f) ? 1.0f : 0.0f;
        const float stability = grad_mom_align * (1.0f - flip);

        float consensus_mix = 0.35f * (1.0f - agree) + 0.25f * sharpness + 0.25f * blend_sign + 0.15f * (1.0f - stability);
        consensus_mix = consensus_mix + (1.0f - consensus_mix) * sharpness;
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;

        const float trust = 1.0f / (1.0f + sharpness + 0.5f * flip);
        chosen_update *= trust;

        const float target = lr * fabsf(g_over_denom) * (1.0f / sharp_ratio);
        const float uabs = fabsf(chosen_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        chosen_update *= scale;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * chosen_update;
        const float final_update = one_minus_la_alpha * chosen_update + lookahead_alpha * su;

        momentum[idx] = m;
        velocity[idx] = slow_fd;
        prev_grad[idx] = g;
        prev_update[idx] = final_update;
        slow_update[idx] = su;
        fisher_diag[idx] = fast_fd;
        updates[idx] = final_update;
    }
}
