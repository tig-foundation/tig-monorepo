// T26 kernels — V4_F : SOTA kernel base + 4 innovations greffées
//
// Innovation A : Fisher-ASAM — dénominateur ASAM normalisé par F_diag (immunité vs beta1 adaptatif)
// Innovation B : Trust-Gated WD — WD post-gate modulé par trust (régularisation bayésienne)
// Innovation C : Jacobian Trust Damping — S_l depth-adaptive dans trust (fondation stable, tête agile)
// Innovation 2 : Pont de Fisher — v_stable = max(v_adabelief, alpha * F_diag)
// Innovation 5 : WD Post-Gate dans sign kernel

// Innovation A — asam_topo_kernel_26 (Fisher-Normalized ASAM)
// Direction : m_hat / (sqrt(F_diag) + eps)  au lieu de  m_hat / sqrt(v_hat)
// Garantie : |dir| ≤ 1  car  |m_hat| ≤ sqrt(F_diag)  par définition de l'EMA de Fisher
// → perturbation strictement bornée quelle que soit la dynamique de beta1 adaptatif
extern "C" __global__ __launch_bounds__(256) void asam_topo_kernel_26(
    const float* __restrict__ params,
    const float* __restrict__ m,
    const float* __restrict__ fisher_diag,
    float* __restrict__ p_out,
    const unsigned int n,
    const float rho,
    const float eps,
    const float bc1
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float inv_bc1 = 1.0f / fmaxf(bc1, 1.0e-12f);

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float p = params[idx];
        const float m_hat = m[idx] * inv_bc1;
        // Dénominateur Fisher : toujours ≥ eps, immunisé contre l'effondrement de v_hat
        const float denom_f = sqrtf(fmaxf(fisher_diag[idx], 0.0f)) + eps;
        const float dir = m_hat / denom_f;
        const float clipped_dir = fminf(fmaxf(dir, -10.0f), 10.0f);
        const float p_mag = fmaxf(fabsf(p), 0.05f);
        p_out[idx] = p + rho * p_mag * clipped_dir;
    }
}

// dual_consensus_fisher_kernel_10 — Phase rapide
//
// Base : SOTA dual_consensus_fisher_kernel (binaire flip/agree, ortho_mix SOTA, gamma 0.25)
//
// Innovations greffées vs SOTA :
//   Innovation 2 : Pont de Fisher (v_stable = max(v_hat, 0.05 * F_diag))
//   Innovation 5 : WD retiré d'adam_update (déplacé en post-gate)
//   Innovation B : Trust-Gated WD — final = candidate - (lr * wd * trust) * p
//   Innovation C : Jacobian Trust Damping — trust = 1/(1 + s_l*(vol+flip)), s_l passé par couche
//
// flip/agree : BINAIRES (retour SOTA — C1 cosine abandonné, over-damping en 256 dims)
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
    const float gate_hi,
    const float s_l       // Innovation C : coefficient Jacobian Trust par couche [0.2, 0.8]
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau   = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    // ortho_mix SOTA
    const float ortho_mix = fminf(fmaxf(0.2f + 0.4f * blend_sign, 0.0f), 0.6f);
    // Innovation 2 : alpha Pont de Fisher
    const float alpha_fisher = 0.05f;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g  = gradients[idx];
        const float pg = prev_grad[idx];
        const float p  = params[idx];

        // Nesterov SOTA (gamma fallback = 0.25)
        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.25f * nesterov_gamma);
        const float g_pred = g + gamma_local * (g - pg);

        float mk = momentum[idx];
        float vk = velocity[idx];

        mk = beta1 * mk + one_minus_beta1 * g_pred;
        const float err = g_pred - mk;
        vk = beta2 * vk + one_minus_beta2 * err * err;

        const float m_hat = mk * inv_bc1;
        const float v_hat = vk * inv_bc2;

        // Fisher diag update identique SOTA
        float fd = fisher_diag[idx];
        const float sqrt_v_raw = sqrtf(fmaxf(vk, 0.0f));
        const float grad_std = sqrt_v_raw + eps;
        const float g_clipped_f = fminf(fmaxf(g_pred, -5.0f * grad_std), 5.0f * grad_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped_f * g_clipped_f;

        // Innovation 2 : Pont de Fisher — plancher dynamique sur v_hat
        const float v_stable = fmaxf(v_hat, alpha_fisher * fd);
        const float sqrt_stable = sqrtf(fmaxf(v_stable, 0.0f));
        // adaptive_eps SOTA (coefficient 0.1)
        const float adaptive_eps = eps * (1.0f + 0.1f * sqrt_stable);
        const float denom = sqrt_stable + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        // Innovation 5 : WD hors adam_update
        const float adam_update = -lr * m_hat * inv_denom;
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update  = -lr * g_over_denom;
        const float sign_update  = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update
                          + blend_norm * norm_update
                          + blend_sign * sign_update;

        // Ortho correction SOTA
        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float g_ortho = g_pred - overlap;
        const float ortho_update = -lr * (g_ortho / denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;

        // BB scaling SOTA
        const float s_prev = prev_update[idx];
        const float s_mag = fabsf(s_prev);
        const float bb_scale = (s_mag > 1e-6f) ? fminf(s_mag * 2.0f, 2.5f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;

        // Robust track SOTA
        const float fisher_rms = sqrtf(fd) + eps;
        const float fisher_norm_update = -lr * (g_pred / fisher_rms);
        const float robust_track = 0.5f * sign_update + 0.5f * fisher_norm_update;

        // Consensus BINAIRE — SOTA (flip/agree binaires, pas cosine C1)
        const float flip  = (g * pg < 0.0f) ? 1.0f : 0.0f;
        const float vol   = fminf(sqrt_stable * 0.33333334f, 1.0f);
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        const float grad_mom_align = (g_pred * m_hat >= 0.0f) ? 1.0f : 0.0f;
        const float stability = grad_mom_align * (1.0f - flip);

        float consensus_mix = 0.35f * (1.0f - agree) + 0.25f * vol
                            + 0.25f * blend_sign + 0.15f * (1.0f - stability);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;

        // Innovation C : Jacobian Trust Damping
        // s_l ∈ [0.2, 0.8] passé par couche depuis Rust
        // trust = 1 / (1 + s_l * (vol + flip))
        // Couche input (s_l=0.8) : freinage fort si bruit/flip → fondation stable
        // Couche output (s_l=0.2) : freinage léger → tête agile
        const float trust = 1.0f / (1.0f + s_l * (vol + flip));
        chosen *= trust;

        // Gate clipping SOTA
        const float target = lr * fabsf(g_over_denom);
        const float uabs = fabsf(chosen);
        const float scale = fminf(fmaxf(__fdividef(target, fmaxf(uabs, 1.0e-12f)), gate_lo), gate_hi);
        chosen *= scale;

        // Lookahead SOTA
        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * chosen;
        const float candidate = one_minus_la_alpha * chosen + lookahead_alpha * su;

        // Innovation B : Trust-Gated WD Post-Gate
        // WD modulé par trust : fort si signal clair, éteint si bruit pur (bayésien)
        const float final_update = candidate - (lr * weight_decay * trust) * p;

        momentum[idx]    = mk;
        velocity[idx]    = vk;
        prev_grad[idx]   = g;
        prev_update[idx] = final_update;
        slow_update[idx] = su;
        fisher_diag[idx] = fd;
        updates[idx]     = final_update;
    }
}

// sign_ef_consensus_kernel_10 — Phase robuste
// Base : SOTA sign_ef_consensus_kernel avec soft-clip ef_delta restauré
// + WD Post-Gate (Innovation 5) — non trust-gaté dans la phase robuste
extern "C" __global__ __launch_bounds__(512, 4) void sign_ef_consensus_kernel_10(
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
    if (tid >= n) return;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float one_minus_la_tau   = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.98f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float inv_lr = 1.0f / fmaxf(lr, 1.0e-8f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float p = params[idx];
        float fd = fisher_diag[idx];
        const float g = gradients[idx];

        const float fd_std = sqrtf(fd) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.0f * fd_std), 4.0f * fd_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        fisher_diag[idx] = fd;

        const float rms = sqrtf(fd) + eps;
        const float g_n = g_clipped / rms;

        // Soft-clip ef_delta identique SOTA
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
        const float candidate = one_minus_la_alpha * u_quant + lookahead_alpha * su;

        const float target = lr * fabsf(g_n);
        const float uabs = fabsf(candidate);
        const float scale = fminf(fmaxf(__fdividef(target, fmaxf(uabs, 1.0e-12f)), gate_lo), gate_hi);
        const float adj = candidate * scale;

        // Innovation 5 : WD Post-Gate (phase robuste)
        const float final_update = adj - lr * weight_decay * p;

        slow_update[idx] = su;
        updates[idx] = final_update;
    }
}
