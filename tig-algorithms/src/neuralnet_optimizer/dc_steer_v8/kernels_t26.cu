// ─────────────────────────────────────────────────────────────────────────────
// P6 (t26 port) — grouped multi-tensor launch (NVIDIA Apex `multi_tensor_apply`).
//
//   https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh
//   https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultiTensorApply.cuh
//
// Same design as kernels_t29.cu. The per-element arithmetic lives in ONE place --
// the two `*_body_10` helpers below. `sign_ef_consensus_kernel_10` /
// `dual_consensus_fisher_kernel_10` (per-tensor, unchanged signatures) and
// `sign_ef_consensus_fused_t26` / `dual_consensus_fisher_fused_t26` (one launch for
// all tensors) both `__forceinline__` the SAME text, so `--use_fast_math` cannot
// contract FMAs differently between the two paths.
//
// Table layout (T = num_tensors, all indices 0-based):
//   all_ptrs[role * T + t]      role 0 g, 1 p, 2 m, 3 v, 4 prev_g, 5 prev_u,
//                               role 6 slow_u, 7 f (fisher), 8 ef, 9 upd
//   scal[0 * T + t] = effective_lr_t
//   scal[1 * T + t] = rel_update_cap_t
//   meta[0 .. T]        = exclusive block prefix sum, meta[T] == gridDim.x
//   meta[T + 1 + t]     = n_t                        (both halves are STATIC)
//
// Block -> tensor: binary search over meta[0..T] (uniform across the block). Every
// thread of a block therefore maps to the SAME tensor, which keeps the block-level
// __syncthreads() reduction in the fast body legal and keeps the early `return`
// in the robust body deadlock-free.
//
// The block partition reproduces (tid, stride) EXACTLY as the per-tensor launch
// produced them, so bit-identity is structural rather than hoped-for.
// ─────────────────────────────────────────────────────────────────────────────

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

// ── body #1: sign / error-feedback consensus (medium depth, n_hidden=10) ─────
// VERBATIM copy of the original `sign_ef_consensus_kernel_10` body. The only edit
// is that `tid` and `stride` arrive as parameters instead of being derived from
// blockIdx/blockDim/gridDim. DO NOT EDIT WITHOUT RE-RUNNING THE GATE.
__device__ __forceinline__ void sign_ef_consensus_body_10(
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
    const float gate_hi,
    const unsigned int tid,
    const unsigned int stride
) {
    if (tid >= n) return;

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

        // Medium depth (10): clip slightly looser than deep (14/18), tighter than shallow
        const float fd_std = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.2f * fd_std), 4.2f * fd_std);
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

// ── body #2: dual consensus + Fisher (medium depth, n_hidden=10) ────────────
// VERBATIM copy of the original `dual_consensus_fisher_kernel_10` body. Two edits:
// `tid`/`stride` arrive as parameters, and the 4-float `__shared__` scratch used by
// the block-level gradient-RMS reduction is allocated by the caller and passed in
// (identical storage in both paths). DO NOT EDIT WITHOUT RE-RUNNING THE GATE.
//
// The reduction is purely intra-block and only ever touches the elements this block
// visits via (tid, stride); since the fused mapping reproduces (tid, stride) exactly,
// `block_grad_rms` is bit-identical to the per-tensor launch.
__device__ __forceinline__ void dual_consensus_fisher_body_10(
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
    // Medium-depth (10): match original t26's ortho recipe
    const float ortho_mix = fminf(fmaxf(0.12f + 0.42f * blend_sign, 0.0f), 0.58f);

    // Pass 1: per-block gradient RMS (for block-level normalization)
    float local_sq_sum = 0.0f;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        local_sq_sum += g * g;
        local_count++;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
        local_count  += __shfl_down_sync(0xffffffff, local_count,  offset);
    }
    if (lane == 0) smem[warp_id] = (local_count > 0) ? (local_sq_sum / (float)local_count) : 0.0f;
    __syncthreads();

    float block_grad_rms = 0.0f;
    if (warp_id == 0) {
        float val = (lane < n_warps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) smem[0] = sqrtf(fmaxf(val / fmaxf((float)n_warps, 1.0f), 0.0f)) + eps;
    }
    __syncthreads();
    block_grad_rms = smem[0];
    const float inv_block_rms = 1.0f / fmaxf(block_grad_rms, 1.0e-8f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];
        const float w = params[idx];

        // Nesterov look-ahead with sign-aware gamma
        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.20f * nesterov_gamma);
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
        const float denom = sqrt_v + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        const float adam_update = -lr * (m_hat * inv_denom + weight_decay * g_pred);
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update = -lr * g_over_denom;
        const float sign_update = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update + blend_norm * norm_update + blend_sign * sign_update;

        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float g_ortho = g_pred - overlap;
        const float ortho_update = -lr * (g_ortho * inv_denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;

        const float s_pu = prev_update[idx];
        const float s_mag = fabsf(s_pu);
        const float bb_scale = (s_mag > 1e-6f) ? fminf(s_mag * 2.0f, 2.5f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;

        // Fisher diagonal with raw-velocity-based clipping
        float fd = fisher_diag[idx];
        const float sqrt_v_for_clip = sqrtf(fmaxf(v, 0.0f));
        const float grad_std = sqrt_v_for_clip + eps;
        const float g_clipped = fminf(fmaxf(g_pred, -5.0f * grad_std), 5.0f * grad_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        const float fisher_rms = sqrtf(fmaxf(fd, 0.0f)) + eps;

        const float fisher_norm_update = -lr * (g_pred / fisher_rms);
        // Medium-depth: match original t26 balance
        const float robust_track = 0.5f * sign_update + 0.5f * fisher_norm_update;

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

        float consensus_mix = 0.18f * curvature_clamped
                            + 0.32f * (1.0f - agree)
                            + 0.22f * vol
                            + 0.22f * blend_sign
                            + 0.12f * (1.0f - stability);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;

        // Enhanced trust with alignment boost (original t26 used 0.12)
        const float align_strength = grad_mom_align * (1.0f - flip) * (1.0f - vol);
        const float trust = (1.0f + 0.12f * align_strength) / (1.0f + 0.60f * vol + 0.60f * flip);
        chosen_update *= trust;

        // Block-level outlier-only clamp: only attenuate elements much larger than block avg
        const float elem_rms = g_abs * inv_block_rms;
        if (elem_rms > 1.8f) {
            chosen_update *= (1.8f / elem_rms);
        }

        // Target gate against adam's norm_update magnitude
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

// ── per-tensor entry points (UNCHANGED external signatures) ─────────────────
extern "C" __global__ __launch_bounds__(128, 6) void sign_ef_consensus_kernel_10(
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
    const unsigned int stride = blockDim.x * gridDim.x;
    sign_ef_consensus_body_10(gradients, params, fisher_diag, ef_residual, slow_update,
                              updates, n, lr, eps, weight_decay, rel_update_cap,
                              lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
                              tid, stride);
}

extern "C" __global__ __launch_bounds__(128, 6) void dual_consensus_fisher_kernel_10(
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
    __shared__ float smem[4];

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    dual_consensus_fisher_body_10(gradients, params, momentum, velocity, prev_grad,
                                  prev_update, slow_update, fisher_diag, updates,
                                  n, lr, beta1, beta2, eps, weight_decay,
                                  bias_correction1, bias_correction2, blend_adam,
                                  blend_norm, blend_sign, nesterov_gamma, bb_blend,
                                  lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
                                  smem, tid, stride);
}

// ── fused entry points: ONE launch for all live tensors ────────────────────
extern "C" __global__ __launch_bounds__(128, 6) void sign_ef_consensus_fused_t26(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ scal,
    const int* __restrict__ meta,
    const unsigned int num_tensors,
    const float eps,
    const float weight_decay,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    const unsigned int b = blockIdx.x;
    const unsigned int t = t26_tensor_of_block(meta, num_tensors, b);

    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];

    // Reproduces EXACTLY the (tid, stride) the per-tensor launch gave this element:
    // that launch had gridDim.x == nblk and this block was its blockIdx.x == b - first.
    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr             = scal[0u * num_tensors + t];
    const float rel_update_cap = scal[1u * num_tensors + t];

    sign_ef_consensus_body_10(
        (const float*)all_ptrs[0u * num_tensors + t],   // gradients
        (const float*)all_ptrs[1u * num_tensors + t],   // params
        (float*)all_ptrs[7u * num_tensors + t],         // fisher_diag  <- s.f
        (float*)all_ptrs[8u * num_tensors + t],         // ef_residual  <- s.ef
        (float*)all_ptrs[6u * num_tensors + t],         // slow_update  <- s.slow_u
        (float*)all_ptrs[9u * num_tensors + t],         // updates      <- fresh upd
        n, lr, eps, weight_decay, rel_update_cap,
        lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
        tid, stride);
}

extern "C" __global__ __launch_bounds__(128, 6) void dual_consensus_fisher_fused_t26(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ scal,
    const int* __restrict__ meta,
    const unsigned int num_tensors,
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
    __shared__ float smem[4];

    const unsigned int b = blockIdx.x;
    const unsigned int t = t26_tensor_of_block(meta, num_tensors, b);

    const unsigned int first = (unsigned int)meta[t];
    const unsigned int nblk  = (unsigned int)meta[t + 1] - first;
    const unsigned int n     = (unsigned int)meta[num_tensors + 1u + t];

    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr = scal[0u * num_tensors + t];

    dual_consensus_fisher_body_10(
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
        smem, tid, stride);
}

// Adds (scale_minus_1 * params[i]) to updates[i] in-place.
// Used at step 0 to shift hidden weights from W_0 → init_scale * W_0.
extern "C" __global__ void scale_additive_kernel_t26(
    const float* __restrict__ params,
    float* __restrict__ updates,
    const unsigned int n,
    const float scale_minus_1
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        updates[idx] += scale_minus_1 * params[idx];
    }
}

// ── DC steering: in-place add of a small host-supplied vector into an update ──
// Launched at most twice per epoch (pulse + restore) and only when the DC
// feature is enabled; the fused step kernel is untouched.
extern "C" __global__ void dc_axpy_t26(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] += src[i];
    }
}
