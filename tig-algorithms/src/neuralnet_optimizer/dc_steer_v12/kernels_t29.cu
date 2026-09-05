// ─────────────────────────────────────────────────────────────────────────────
// P6 — grouped multi-tensor launch (NVIDIA Apex `multi_tensor_apply` pattern).
//
//   https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh
//   https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh
//
// The per-element arithmetic lives in ONE place -- the two `*_body_t29` helpers below.
// `sign_ef_consensus_kernel` / `dual_consensus_fisher_kernel` (per-tensor, unchanged
// signatures) and `sign_ef_consensus_fused_t29` / `dual_consensus_fisher_fused_t29`
// (one launch for all tensors) both `__forceinline__` the SAME text, so the fused path
// cannot drift from the reference implementation.
//
// Table layout (T = num_tensors, all indices 0-based):
//   all_ptrs[role * T + t]      role 0 g, 1 p, 2 m, 3 v, 4 prev_g, 5 prev_u,
//                               role 6 slow_u, 7 f (fisher), 8 ef, 9 upd
//   scal[0 * T + t] = effective_lr_t
//   scal[1 * T + t] = wd_eff_t
//   meta[0 .. T]        = exclusive block prefix sum, meta[T] == gridDim.x
//   meta[T + 1 + t]     = n_t                        (both halves are STATIC)
//
// Block -> tensor: binary search over meta[0..T] (uniform across the block, <= 4 loads
// from L1, no shared memory, no __syncthreads -- so the early `return` inside the body
// can never deadlock). Apex resolves this with an explicit `block_to_tensor[]` array;
// with T <= 12 the search is cheaper than uploading that array per step.
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ unsigned int t29_tensor_of_block(
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

// ── body #1: sign / error-feedback consensus ────────────────────────────────
// VERBATIM copy of the original `sign_ef_consensus_kernel` body (kernels_t29.cu:21-67).
// The only edit is that `tid` and `stride` arrive as parameters instead of being
// derived from blockIdx/blockDim/gridDim. DO NOT EDIT WITHOUT RE-RUNNING THE GATE.
__device__ __forceinline__ void sign_ef_consensus_body_t29(
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
    if (tid >= n) return;

    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;

    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;

    const float lr_safe = fmaxf(lr, 1.0e-12f);
    const float inv_lr = 1.0f / lr_safe;
    const float ef_decay = 0.990f;

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
        const float u_desired = -lr * g_n + ef_old;
        const float u_quant = -lr * copysignf(1.0f, g_n + ef_old * inv_lr);

        float ef_delta = u_desired - u_quant;
        const float ef_cap = 10.0f * lr_safe;
        ef_delta = ef_delta / (1.0f + fabsf(ef_delta) / fmaxf(ef_cap, 1.0e-12f));
        ef_residual[idx] = ef_decay * ef_old + ef_delta;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * u_quant;
        float u = one_minus_la_alpha * u_quant + lookahead_alpha * su;

        const float target = lr * fabsf(g_n);
        const float uabs = fabsf(u);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        u *= scale;

        u -= (lr * weight_decay) * params[idx];

        slow_update[idx] = su;
        updates[idx] = u;
    }
}

// ── body #2: dual consensus + Fisher ────────────────────────────────────────
// VERBATIM copy of the original `dual_consensus_fisher_kernel` body
// (kernels_t29.cu:102-180), same single edit (tid/stride as parameters).
__device__ __forceinline__ void dual_consensus_fisher_body_t29(
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
    const unsigned int tid,
    const unsigned int stride
) {
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
        // Adaptive Nesterov: if sign disagree, reduce gamma to avoid overshoot
        const float sign_agree = (g * pg >= 0.0f) ? 1.0f : 0.35f;
        const float g_pred = g + (nesterov_gamma * sign_agree) * (g - pg);

        float m = momentum[idx];
        float v = velocity[idx];

        m = beta1 * m + one_minus_beta1 * g_pred;
        v = beta2 * v + one_minus_beta2 * (g_pred * g_pred);

        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;

        const float denom = sqrtf(fmaxf(v_hat, 0.0f)) + eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        const float adam_u = -lr * (m_hat * inv_denom);
        const float norm_u = -lr * (g_pred * inv_denom);
        const float sign_u = -lr * copysignf(1.0f, m_hat);

        const float base_u = blend_adam * adam_u + blend_norm * norm_u + blend_sign * sign_u;

        float fd = fisher_diag[idx];
        const float fd_floor = fmaxf(fd, 1.0e-4f);
        const float fd_std = sqrtf(fd_floor) + eps;
        const float g_clip = fminf(fmaxf(g_pred, -6.0f * fd_std), 6.0f * fd_std);

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

// ── per-tensor entry points (UNCHANGED external signatures) ─────────────────
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
    sign_ef_consensus_body_t29(gradients, params, fisher_diag, ef_residual, slow_update,
                               updates, n, lr, eps, weight_decay, lookahead_alpha,
                               lookahead_tau, gate_lo, gate_hi, tid, stride);
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
    dual_consensus_fisher_body_t29(gradients, params, momentum, velocity, prev_grad,
                                   prev_update, slow_update, fisher_diag, updates,
                                   n, lr, beta1, beta2, eps, weight_decay,
                                   bias_correction1, bias_correction2, blend_adam,
                                   blend_norm, blend_sign, nesterov_gamma, bb_blend,
                                   lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
                                   tid, stride);
}

// ── fused entry points: ONE launch for all live tensors ────────────────────
extern "C" __global__ __launch_bounds__(256, 3) void sign_ef_consensus_fused_t29(
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
    const unsigned int t = t29_tensor_of_block(meta, num_tensors, b);

    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];

    // Reproduces EXACTLY the (tid, stride) the per-tensor launch gave this element:
    // that launch had gridDim.x == nblk and this block was its blockIdx.x == b - first.
    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr           = scal[0u * num_tensors + t];
    const float weight_decay = scal[1u * num_tensors + t];

    sign_ef_consensus_body_t29(
        (const float*)all_ptrs[0u * num_tensors + t],   // gradients
        (const float*)all_ptrs[1u * num_tensors + t],   // params
        (float*)all_ptrs[7u * num_tensors + t],         // fisher_diag  <- s.f
        (float*)all_ptrs[8u * num_tensors + t],         // ef_residual  <- s.ef
        (float*)all_ptrs[6u * num_tensors + t],         // slow_update  <- s.slow_u
        (float*)all_ptrs[9u * num_tensors + t],         // updates      <- fresh upd
        n, lr, eps, weight_decay, lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
        tid, stride);
}

extern "C" __global__ __launch_bounds__(256, 3) void dual_consensus_fisher_fused_t29(
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
    const float gate_hi
) {
    const unsigned int b = blockIdx.x;
    const unsigned int t = t29_tensor_of_block(meta, num_tensors, b);

    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];

    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr           = scal[0u * num_tensors + t];
    const float weight_decay = scal[1u * num_tensors + t];

    dual_consensus_fisher_body_t29(
        (const float*)all_ptrs[0u * num_tensors + t],   // gradients
        (const float*)all_ptrs[1u * num_tensors + t],   // params
        (float*)all_ptrs[2u * num_tensors + t],         // momentum     <- s.m
        (float*)all_ptrs[3u * num_tensors + t],         // velocity     <- s.v
        (float*)all_ptrs[4u * num_tensors + t],         // prev_grad    <- s.prev_g
        (float*)all_ptrs[5u * num_tensors + t],         // prev_update  <- s.prev_u
        (float*)all_ptrs[6u * num_tensors + t],         // slow_update  <- s.slow_u
        (float*)all_ptrs[7u * num_tensors + t],         // fisher_diag  <- s.f
        (float*)all_ptrs[9u * num_tensors + t],         // updates      <- fresh upd
        n, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
        blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend,
        lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
        tid, stride);
}

// ── DC steering: in-place add of a small host-supplied vector into an update ──
// Launched at most twice per epoch (pulse + restore) and only when the DC
// feature is enabled; the fused step kernel is untouched.
extern "C" __global__ void dc_axpy_t29(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] += src[i];
    }
}
