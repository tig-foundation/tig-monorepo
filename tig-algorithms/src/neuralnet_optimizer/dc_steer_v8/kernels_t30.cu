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

// ── T30-NORMFUSE: one launch for every SINGLE-BLOCK grad-norm reduction ──────
// 19 of the 24 live tensors on nh7 get gridDim.x == 1 from
// `min(ceil(n/256), sm_blocks).max(1)` (track_t30.rs:502-505), i.e. 19 launches that each
// occupy 256 of ~10,496 CUDA cores (2.4% of the GPU) and, because the runtime fuel-checks
// after every launch, are STRICTLY SERIALISED with respect to one another.
//
// BIT-IDENTITY IS STRUCTURAL, not statistical:
//   * one block per tensor => blockIdx.x is the tensor index, so tid == threadIdx.x and
//     stride == blockDim.x -- exactly what the per-tensor kernel computes when gridDim.x==1
//     (tid = 0*blockDim + threadIdx, stride = blockDim*1). Same elements, same order,
//     therefore the same float accumulation tree.
//   * each tensor's slot receives exactly ONE atomicAdd, into a slot pre-zeroed by
//     alloc_zeros, so arrival order is irrelevant and atomicAdd(0, v) == v exactly.
// The 5 multi-block tensors are NOT fused -- several blocks atomicAdd into one slot there,
// so their arrival order is observable and fusing them would not be bit-identical.
extern "C" __global__ __launch_bounds__(256, 4) void grad_norm_fused_1blk_7(
    const unsigned long long* __restrict__ g_ptrs,   // [T] gradient base pointer per tensor
    const int* __restrict__ lens,                    // [T] element count per tensor
    const int* __restrict__ slots,                   // [T] destination slot in norm_out
    float* __restrict__ norm_out,
    const unsigned int num_tensors
) {
    const unsigned int t = blockIdx.x;
    if (t >= num_tensors) return;
    const float* __restrict__ gradients = (const float*)g_ptrs[t];
    const unsigned int n = (unsigned int)lens[t];

    // ---- verbatim body of grad_norm_reduction_kernel_7 from here down ----
    __shared__ float smem[8];
    const unsigned int tid = threadIdx.x;              // == blockIdx.x*blockDim.x+threadIdx.x when gridDim.x==1
    const unsigned int stride = blockDim.x;            // == blockDim.x*gridDim.x   when gridDim.x==1
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

// ─────────────────────────────────────────────────────────────────────────────
// P6 (t30 port) — grouped multi-tensor launch, NVIDIA Apex `multi_tensor_apply`
// pattern, identical in structure to the measured t29 implementation
// (`kernels_t29.cu`).  Design notes: research/P6_DIFF.md.
//
// The per-element arithmetic lives in ONE place — the two `*_body_7` helpers.
// The per-tensor kernels (`sign_ef_consensus_kernel_7`,
// `dual_consensus_fisher_kernel_7`, external signatures unchanged) and the fused
// kernels (`sign_ef_consensus_fused_7`, `dual_consensus_fisher_fused_7`) both
// `__forceinline__` the SAME text, so `--use_fast_math` cannot contract FMAs
// differently between the two paths.
//
// Table layout (T = num_tensors, all indices 0-based):
//   all_ptrs[role * T + t]   role 0 g, 1 p, 2 m, 3 v, 4 prev_g, 5 prev_u,
//                            role 6 slow_u, 7 f (fisher), 8 ef, 9 upd
//   scal[0 * T + t] = effective_lr_t
//   scal[1 * T + t] = wd_eff_t
//   meta[0 .. T]        = exclusive block prefix sum, meta[T] == gridDim.x
//   meta[T + 1 + t]     = n_t                        (both halves are STATIC)
//
// Block -> tensor: binary search over meta[0..T], uniform across the block.  The
// grid is BLOCK-partitioned (not element-partitioned): block `b` of tensor `t`
// reconstructs the exact `(tid, stride)` the per-tensor launch gave it, because
// that launch had `gridDim.x == nblk_t` and this block was its `blockIdx.x`.
// That is what makes bit-identity structural.  It also keeps `__syncthreads()`
// inside `dual_consensus_fisher_body_7` block-uniform: every block belongs to
// exactly one tensor and no thread returns early before the barrier.
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// H1 — grouped multi-tensor SVRG.  Collapses the per-tensor
// `svrg_variance_reduce_kernel_7` launches into ONE launch on the P6 grid.
//
// BIT-IDENTITY, structurally (same argument already proved for
// `dual_consensus_fisher_fused_7` in this file):
//   * the per-tensor launch used `gridDim.x == nblk_t` and this block was its
//     `blockIdx.x`, so reconstructing `tid = (b-first)*blockDim + threadIdx` and
//     `stride = blockDim * nblk_t` gives every element the identical (tid, stride)
//     and the identical trip count;
//   * `mean_ef` is a PER-BLOCK statistic over exactly that block's strided slice,
//     so it is unchanged;
//   * `__syncthreads()` stays block-uniform: every block belongs to exactly one
//     tensor and no thread returns early before a barrier.
// The body below is a VERBATIM copy of `svrg_variance_reduce_kernel_7`'s body
// (the ONLY edits are that tid/stride are parameters and the never-written
// `ef_mean_out` argument is dropped).  DO NOT EDIT WITHOUT RE-RUNNING THE GATE.
// ─────────────────────────────────────────────────────────────────────────────
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
    const unsigned long long* __restrict__ all_ptrs,   // role 0 = g, 1 = ef, 2 = vr_out
    const int* __restrict__ meta,                      // [0..T] block prefix, [T+1..2T] lengths
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


// ── body #1: sign / error-feedback consensus ────────────────────────────────
// VERBATIM copy of the original `sign_ef_consensus_kernel_7` body.  The only
// edit is that `tid` and `stride` arrive as parameters instead of being derived
// from blockIdx/blockDim/gridDim.  DO NOT EDIT WITHOUT RE-RUNNING THE GATE.
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
    if (tid >= n) return;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.95f;
    const float one_minus_fisher_beta = 1.0f - fisher_beta;
    const float inv_lr = 1.0f / fmaxf(lr, 1.0e-8f);
    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fmaxf(fisher_diag[idx], 1.0e-8f);
        const float g = gradients[idx];
        const float fd_std = sqrtf(fd) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.0f * fd_std), 4.0f * fd_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        fd = fmaxf(fd, 1.0e-8f);
        fisher_diag[idx] = fd;
        const float rms = sqrtf(fd) + eps;
        const float g_n = g / rms;
        const float ef_old = ef_residual[idx];
        const float u_quant = -lr * copysignf(1.0f, g_n + ef_old * inv_lr);
        float ef_delta = (-lr * g_n + ef_old) - u_quant;
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
        slow_update[idx] = su;
        updates[idx] = final_update * scale - (lr * weight_decay) * params[idx];
    }
}

// ── body #2: dual consensus + Fisher ────────────────────────────────────────
// VERBATIM copy of the original `dual_consensus_fisher_kernel_7` body, same
// single edit (tid/stride as parameters).  The `__shared__ float smem[8]` and
// the two `__syncthreads()` stay inside the body: it is inlined exactly once per
// kernel, and every thread of the block reaches both barriers.
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
    // P-TR probe (40B): upper clamp of the per-element LARS trust ratio.
    // Default 2.0f == the shipped literal, so the default path is unchanged.
    const float tr_hi,
    // 41A-H1: 1.0f == shipped; 0.0f collapses block_norm_scale to the constant E[c].
    const float bns_mix,
    const unsigned int tid,
    const unsigned int stride
) {
    __shared__ float smem[8];

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
    if (lane == 0) smem[warp_id] = local_sq_sum / fmaxf((float)local_count, 1.0f);
    __syncthreads();

    float block_grad_rms = 0.0f;
    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) smem[0] = sqrtf(fmaxf(val / 8.0f, 0.0f)) + eps;
    }
    __syncthreads();
    block_grad_rms = smem[0];
    const float inv_block_rms = 1.0f / fmaxf(block_grad_rms, 1.0e-8f);

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
        const float flip = (g * pg < 0.0f) ? 1.0f : 0.0f;
        const float vol = fminf(sqrt_v * 0.33333334f, 1.0f);
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        const float grad_mom_align = (g_pred * m_hat >= 0.0f) ? 1.0f : 0.0f;
        const float stability = grad_mom_align * (1.0f - flip);
        const float g_abs = fabsf(g);
        const float pg_abs = fabsf(pg);
        const float curvature = fabsf(g - pg) / fmaxf(g_abs + pg_abs + eps, 1.0e-8f);
        const float curvature_clamped = fminf(curvature, 1.0f);
        float consensus_mix = 0.3f * curvature_clamped + 0.2f * vol + 0.2f * blend_sign + 0.15f * (1.0f - stability) + 0.15f * (1.0f - agree);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;
        const float param_mag = fabsf(params[idx]);
        const float mom_mag = fabsf(m_hat) + eps;
        const float trust_radius = fminf(fmaxf(param_mag / mom_mag, 0.5f), tr_hi);
        const float trust_centered = trust_radius / 1.0f;
        const float trust = trust_centered / (1.0f + 0.5f * vol + 0.5f * flip);
        chosen_update *= trust;
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
        const float final_wd = final_update - (lr * weight_decay) * params[idx];
        momentum[idx] = m;
        velocity[idx] = v;
        prev_grad[idx] = g;
        prev_update[idx] = final_wd;
        slow_update[idx] = su;
        fisher_diag[idx] = fd;
        updates[idx] = final_wd;
    }
}

// ── per-tensor entry points (UNCHANGED external signatures) ─────────────────
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
                                 2.0f, 1.0f, tid, stride);
}

// ── fused entry points: ONE launch for all live tensors ────────────────────
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

    // Reproduces EXACTLY the (tid, stride) the per-tensor launch gave this element:
    // that launch had gridDim.x == nblk and this block was its blockIdx.x == b - first.
    const unsigned int tid    = (b - first) * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * nblk;

    const float lr           = scal[0u * num_tensors + t];
    const float weight_decay = scal[1u * num_tensors + t];

    sign_ef_consensus_body_7(
        (const float*)all_ptrs[0u * num_tensors + t],   // gradients
        (const float*)all_ptrs[1u * num_tensors + t],   // params
        (float*)all_ptrs[7u * num_tensors + t],         // fisher_diag  <- s.f
        (float*)all_ptrs[8u * num_tensors + t],         // ef_residual  <- s.ef
        (float*)all_ptrs[6u * num_tensors + t],         // slow_update  <- s.slow_u
        (float*)all_ptrs[9u * num_tensors + t],         // updates      <- s.upd
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
    const float bns_mix
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
        (const float*)all_ptrs[0u * num_tensors + t],   // gradients (or SVRG vr_grad)
        (const float*)all_ptrs[1u * num_tensors + t],   // params
        (float*)all_ptrs[2u * num_tensors + t],         // momentum     <- s.m
        (float*)all_ptrs[3u * num_tensors + t],         // velocity     <- s.v
        (float*)all_ptrs[4u * num_tensors + t],         // prev_grad    <- s.prev_g
        (float*)all_ptrs[5u * num_tensors + t],         // prev_update  <- s.prev_u
        (float*)all_ptrs[6u * num_tensors + t],         // slow_update  <- s.slow_u
        (float*)all_ptrs[7u * num_tensors + t],         // fisher_diag  <- s.f
        (float*)all_ptrs[9u * num_tensors + t],         // updates      <- s.upd
        n, lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
        blend_adam, blend_norm, blend_sign, nesterov_gamma, bb_blend,
        lookahead_alpha, lookahead_tau, gate_lo, gate_hi,
        tr_hi, bns_mix, tid, stride);
}

// ── DC steering: in-place add of a small host-supplied vector into an update ──
// Launched at most twice per epoch (pulse + restore) and only when the DC
// feature is enabled; the fused step kernel is untouched.
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
