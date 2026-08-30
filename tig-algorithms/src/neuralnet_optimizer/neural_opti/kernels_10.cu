#include <cuda_runtime.h>
#include <math.h>

#ifndef NEURAL_OPTI_S3_MATH
#define NEURAL_OPTI_S3_MATH

__device__ __forceinline__ float s3_canonicalize_rn(float x, unsigned int keep_bits) {
    if (keep_bits >= 23u || x == 0.0f) return x;
    if (keep_bits < 1u) keep_bits = 1u;
    if (keep_bits > 23u) keep_bits = 23u;

    unsigned int bits = __float_as_uint(x);
    const unsigned int sign = bits & 0x80000000u;
    unsigned int mag = bits & 0x7fffffffu;
    const unsigned int exp = mag & 0x7f800000u;
    if (exp == 0x7f800000u) return x;

    const unsigned int drop = 23u - keep_bits;
    if (drop == 0u) return x;

    const unsigned int mask = (1u << drop) - 1u;
    const unsigned int half = 1u << (drop - 1u);
    const unsigned int rem = mag & mask;
    const unsigned int lsb = (mag >> drop) & 1u;

    mag &= ~mask;
    if (rem > half || (rem == half && lsb != 0u)) {
        mag += 1u << drop;
    }
    if (mag >= 0x7f800000u) {
        mag = 0x7f7fffffu & ~mask;
    }
    return __uint_as_float(sign | mag);
}

__device__ __forceinline__ float s3_cbrt_canonical(float x, unsigned int keep_bits) {
    const float root = cbrtf(fmaxf(x, 0.0f));
    return s3_canonicalize_rn(root, keep_bits);
}

#endif

extern "C" __global__ void s3_p3_base_update_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float one_minus_beta = __fsub_rn(1.0f, beta);

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(__ldg(gradients + idx), arch_mantissa_bits);
        const float g2 = __fmul_rn(g, g);

        float m = momentum[idx];
        float s = power_momentum[idx];

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float s_term = __fmul_rn(one_minus_beta, g2);
        s = __fmaf_ieee_rn(beta, s, s_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_sec = __fmaf_ieee_rn(beta, s, s_term);
        const float root = s3_canonicalize_rn(__fsqrt_rn(fmaxf(nesterov_sec, 0.0f)), arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        momentum[idx] = m;
        power_momentum[idx] = s;
        const float with_decay = __fmaf_ieee_rn(weight_decay, __ldg(params + idx), normalized);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_p3_fused_dual_loss_update_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const float progress_credit,
    const float headroom_flag,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const bool allow_headroom = (headroom_flag > 0.0f) && (progress_credit > 0.0f);
    const float gain = allow_headroom
        ? fminf(fmaxf(__fmul_rn(coherence_gain, progress_credit), 0.0f), 1.0f)
        : 0.0f;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float ag = fabsf(g);
        const float g2 = __fmul_rn(g, g);

        float m = momentum[idx];
        float s = power_momentum[idx];

        float confidence = 0.0f;
        if (allow_headroom) {
            const bool aligned = ((g > 0.0f && m > 0.0f) || (g < 0.0f && m < 0.0f));
            if (aligned && s > 0.0f && ag > 0.0f) {
                const float hist_scale = s3_canonicalize_rn(__fsqrt_rn(fmaxf(s, 0.0f)), arch_mantissa_bits);
                const float hi = fmaxf(ag, hist_scale);
                const float lo = fminf(ag, hist_scale);
                const float q = __fdiv_rn(lo, fmaxf(hi, eps));
                confidence = __fmul_rn(q, q);
            }
        }

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float s_term = __fmul_rn(one_minus_beta, g2);
        s = __fmaf_ieee_rn(beta, s, s_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_sec = __fmaf_ieee_rn(beta, s, s_term);
        const float root = s3_canonicalize_rn(__fsqrt_rn(fmaxf(nesterov_sec, 0.0f)), arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        float final_u = normalized;
        if (allow_headroom && gain > 0.0f) {
            const float base_mag = fminf(fabsf(normalized), 1.0f);
            const float release_shape = __fsqrt_rn(base_mag);
            const float headroom = __fsub_rn(1.0f, base_mag);
            const float gc = __fmul_rn(gain, confidence);
            const float shaped_headroom = __fmul_rn(release_shape, headroom);
            const float extra = __fmul_rn(gc, shaped_headroom);
            const float boosted_mag = __fadd_rn(base_mag, extra);
            final_u = copysignf(fminf(boosted_mag, 1.0f), normalized);
        }

        momentum[idx] = m;
        power_momentum[idx] = s;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], final_u);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_reduce_sum_sq_10(
    const float* __restrict__ x,
    float* __restrict__ partial_outs,
    const unsigned int n,
    const unsigned int out_offset
) {
    extern __shared__ float smem[];
    const unsigned int tid = threadIdx.x;
    const unsigned int global = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (unsigned int i = global; i < n; i += stride) {
        const float v = __ldg(x + i);
        sum = __fmaf_ieee_rn(v, v, sum);
    }

    const unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    if (lane == 0u) {
        smem[warp] = sum;
    }
    __syncthreads();

    const unsigned int num_warps = (blockDim.x + 31u) >> 5;
    if (warp == 0u) {
        float block_sum = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0u) {
            partial_outs[out_offset + blockIdx.x] = block_sum;
        }
    }
}

extern "C" __global__ void s3_reduce_sum_sq_dual_10(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ partial_outs,
    const unsigned int n,
    const unsigned int partial_stride
) {
    extern __shared__ float smem[];
    const unsigned int tid = threadIdx.x;
    const unsigned int global = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (unsigned int i = global; i < n; i += stride) {
        const float va = __ldg(a + i);
        const float vb = __ldg(b + i);
        sum_a = __fmaf_ieee_rn(va, va, sum_a);
        sum_b = __fmaf_ieee_rn(vb, vb, sum_b);
    }

    const unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_a += __shfl_down_sync(mask, sum_a, offset);
        sum_b += __shfl_down_sync(mask, sum_b, offset);
    }

    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int num_warps = (blockDim.x + 31u) >> 5;
    if (lane == 0u) {
        smem[warp] = sum_a;
        smem[warp + num_warps] = sum_b;
    }
    __syncthreads();

    if (warp == 0u) {
        float block_a = (lane < num_warps) ? smem[lane] : 0.0f;
        float block_b = (lane < num_warps) ? smem[lane + num_warps] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_a += __shfl_down_sync(mask, block_a, offset);
            block_b += __shfl_down_sync(mask, block_b, offset);
        }
        if (lane == 0u) {
            partial_outs[blockIdx.x] = block_a;
            partial_outs[blockIdx.x + partial_stride] = block_b;
        }
    }
}

extern "C" __global__ void s3_scale_updates_10(
    float* __restrict__ updates,
    const unsigned int n,
    const float scale,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float u = updates[idx];
        const float scaled = __fmul_rn(u, scale);
        updates[idx] = s3_canonicalize_rn(scaled, arch_mantissa_bits);
    }
}

__device__ __forceinline__ float s3_sam_apply_to_update(
    float u,
    float g,
    float pg,
    float inv_norm,
    float lr,
    float do_corr,
    unsigned int arch_mantissa_bits
) {
    const float g_hat = __fmul_rn(g, inv_norm);
    const float diff = __fsub_rn(g, pg);
    const float hvp_proxy = __fmul_rn(diff, inv_norm);
    const float ascent_corr = __fmul_rn(__fmul_rn(-lr, hvp_proxy), do_corr);
    const float g_aligned = __fmul_rn(hvp_proxy, g_hat);
    const float buffer_corr = __fmul_rn(__fmul_rn(-lr, g_aligned), do_corr);
    const float u_new = __fadd_rn(u, __fadd_rn(ascent_corr, buffer_corr));
    return s3_canonicalize_rn(u_new, arch_mantissa_bits);
}

extern "C" __global__ void s3_p3_base_update_sam_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ updates,
    float* __restrict__ prev_grad,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float g_norm,
    const float apply_corr,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const float inv_norm = __fdiv_rn(1.0f, __fadd_rn(g_norm, eps));
    const float do_corr = (apply_corr > 0.0f) ? 1.0f : 0.0f;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(__ldg(gradients + idx), arch_mantissa_bits);
        const float g2 = __fmul_rn(g, g);

        float m = momentum[idx];
        float s = power_momentum[idx];

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float s_term = __fmul_rn(one_minus_beta, g2);
        s = __fmaf_ieee_rn(beta, s, s_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_sec = __fmaf_ieee_rn(beta, s, s_term);
        const float root = s3_canonicalize_rn(__fsqrt_rn(fmaxf(nesterov_sec, 0.0f)), arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        momentum[idx] = m;
        power_momentum[idx] = s;
        const float with_decay = __fmaf_ieee_rn(weight_decay, __ldg(params + idx), normalized);
        float update = __fmul_rn(-lr, with_decay);
        update = s3_canonicalize_rn(update, arch_mantissa_bits);

        const float pg = prev_grad[idx];
        update = s3_sam_apply_to_update(update, g, pg, inv_norm, lr, do_corr, arch_mantissa_bits);
        updates[idx] = update;
        prev_grad[idx] = g;
    }
}

extern "C" __global__ void s3_p3_fused_dual_loss_update_sam_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ updates,
    float* __restrict__ prev_grad,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const float progress_credit,
    const float headroom_flag,
    const float g_norm,
    const float apply_corr,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const bool allow_headroom = (headroom_flag > 0.0f) && (progress_credit > 0.0f);
    const float gain = allow_headroom
        ? fminf(fmaxf(__fmul_rn(coherence_gain, progress_credit), 0.0f), 1.0f)
        : 0.0f;
    const float inv_norm = __fdiv_rn(1.0f, __fadd_rn(g_norm, eps));
    const float do_corr = (apply_corr > 0.0f) ? 1.0f : 0.0f;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float ag = fabsf(g);
        const float g2 = __fmul_rn(g, g);

        float m = momentum[idx];
        float s = power_momentum[idx];

        float confidence = 0.0f;
        if (allow_headroom) {
            const bool aligned = ((g > 0.0f && m > 0.0f) || (g < 0.0f && m < 0.0f));
            if (aligned && s > 0.0f && ag > 0.0f) {
                const float hist_scale = s3_canonicalize_rn(__fsqrt_rn(fmaxf(s, 0.0f)), arch_mantissa_bits);
                const float hi = fmaxf(ag, hist_scale);
                const float lo = fminf(ag, hist_scale);
                const float q = __fdiv_rn(lo, fmaxf(hi, eps));
                confidence = __fmul_rn(q, q);
            }
        }

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float s_term = __fmul_rn(one_minus_beta, g2);
        s = __fmaf_ieee_rn(beta, s, s_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_sec = __fmaf_ieee_rn(beta, s, s_term);
        const float root = s3_canonicalize_rn(__fsqrt_rn(fmaxf(nesterov_sec, 0.0f)), arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        float final_u = normalized;
        if (allow_headroom && gain > 0.0f) {
            const float base_mag = fminf(fabsf(normalized), 1.0f);
            const float release_shape = __fsqrt_rn(base_mag);
            const float headroom = __fsub_rn(1.0f, base_mag);
            const float gc = __fmul_rn(gain, confidence);
            const float shaped_headroom = __fmul_rn(release_shape, headroom);
            const float extra = __fmul_rn(gc, shaped_headroom);
            const float boosted_mag = __fadd_rn(base_mag, extra);
            final_u = copysignf(fminf(boosted_mag, 1.0f), normalized);
        }

        momentum[idx] = m;
        power_momentum[idx] = s;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], final_u);
        float update = __fmul_rn(-lr, with_decay);
        update = s3_canonicalize_rn(update, arch_mantissa_bits);

        const float pg = prev_grad[idx];
        update = s3_sam_apply_to_update(update, g, pg, inv_norm, lr, do_corr, arch_mantissa_bits);
        updates[idx] = update;
        prev_grad[idx] = g;
    }
}

extern "C" __global__ void s3_p3_fused_dual_loss_update_sam_row_trust_10(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ updates,
    float* __restrict__ prev_grad,
    const unsigned int n_rows,
    const unsigned int n_cols,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const float progress_credit,
    const float headroom_flag,
    const float g_norm,
    const float apply_corr,
    const unsigned int arch_mantissa_bits
) {
    extern __shared__ float smem[];
    const unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int cols = n_cols;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const bool allow_headroom = (headroom_flag > 0.0f) && (progress_credit > 0.0f);
    const float gain = allow_headroom
        ? fminf(fmaxf(__fmul_rn(coherence_gain, progress_credit), 0.0f), 1.0f)
        : 0.0f;
    const float inv_norm = __fdiv_rn(1.0f, __fadd_rn(g_norm, eps));
    const float do_corr = (apply_corr > 0.0f) ? 1.0f : 0.0f;

    float p_acc = 0.0f;
    float u_acc = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        const unsigned int idx = row * cols + c;
        const float g = s3_canonicalize_rn(__ldg(gradients + idx), arch_mantissa_bits);
        const float ag = fabsf(g);
        const float g2 = __fmul_rn(g, g);

        float m = momentum[idx];
        float s = power_momentum[idx];

        float confidence = 0.0f;
        if (allow_headroom) {
            const bool aligned = ((g > 0.0f && m > 0.0f) || (g < 0.0f && m < 0.0f));
            if (aligned && s > 0.0f && ag > 0.0f) {
                const float hist_scale = s3_canonicalize_rn(__fsqrt_rn(fmaxf(s, 0.0f)), arch_mantissa_bits);
                const float hi = fmaxf(ag, hist_scale);
                const float lo = fminf(ag, hist_scale);
                const float q = __fdiv_rn(lo, fmaxf(hi, eps));
                confidence = __fmul_rn(q, q);
            }
        }

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float s_term = __fmul_rn(one_minus_beta, g2);
        s = __fmaf_ieee_rn(beta, s, s_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_sec = __fmaf_ieee_rn(beta, s, s_term);
        const float root = s3_canonicalize_rn(__fsqrt_rn(fmaxf(nesterov_sec, 0.0f)), arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        float final_u = normalized;
        if (allow_headroom && gain > 0.0f) {
            const float base_mag = fminf(fabsf(normalized), 1.0f);
            const float release_shape = __fsqrt_rn(base_mag);
            const float headroom = __fsub_rn(1.0f, base_mag);
            const float gc = __fmul_rn(gain, confidence);
            const float shaped_headroom = __fmul_rn(release_shape, headroom);
            const float extra = __fmul_rn(gc, shaped_headroom);
            const float boosted_mag = __fadd_rn(base_mag, extra);
            final_u = copysignf(fminf(boosted_mag, 1.0f), normalized);
        }

        momentum[idx] = m;
        power_momentum[idx] = s;
        const float p = __ldg(params + idx);
        const float with_decay = __fmaf_ieee_rn(weight_decay, p, final_u);
        float update = __fmul_rn(-lr, with_decay);
        update = s3_canonicalize_rn(update, arch_mantissa_bits);

        const float pg = prev_grad[idx];
        update = s3_sam_apply_to_update(update, g, pg, inv_norm, lr, do_corr, arch_mantissa_bits);
        updates[idx] = update;
        prev_grad[idx] = g;

        p_acc = __fmaf_ieee_rn(p, p, p_acc);
        u_acc = __fmaf_ieee_rn(update, update, u_acc);
    }

    const unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        p_acc += __shfl_down_sync(mask, p_acc, offset);
        u_acc += __shfl_down_sync(mask, u_acc, offset);
    }

    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int num_warps = (blockDim.x + 31u) >> 5;
    if (lane == 0u) {
        smem[warp] = p_acc;
        smem[warp + num_warps] = u_acc;
    }
    __syncthreads();

    if (warp == 0u) {
        float p_block = (lane < num_warps) ? smem[lane] : 0.0f;
        float u_block = (lane < num_warps) ? smem[lane + num_warps] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            p_block += __shfl_down_sync(mask, p_block, offset);
            u_block += __shfl_down_sync(mask, u_block, offset);
        }
        if (lane == 0u) {
            float scale = 1.0f;
            const float p_norm = sqrtf(p_block);
            const float u_norm = sqrtf(u_block);
            if (p_norm > 0.0f && u_norm > 0.0f && lr > 0.0f) {
                const float v_norm = __fdiv_rn(u_norm, lr);
                scale = __fdiv_rn(p_norm, __fadd_rn(v_norm, eps));
            }
            smem[0] = scale;
        }
    }
    __syncthreads();

    const float scale = smem[0];
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        const unsigned int idx = row * cols + c;
        const float scaled = __fmul_rn(updates[idx], scale);
        updates[idx] = s3_canonicalize_rn(scaled, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_sam_sharpness_modify_10(
    const float* __restrict__ gradients,
    float* __restrict__ prev_grad,
    float* __restrict__ updates,
    const unsigned int n,
    const float g_norm,
    const float lr,
    const float eps,
    const float apply_corr,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float inv_norm = __fdiv_rn(1.0f, __fadd_rn(g_norm, eps));
    const float do_corr = (apply_corr > 0.0f) ? 1.0f : 0.0f;

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float pg = prev_grad[idx];
        const float u = updates[idx];
        updates[idx] = s3_sam_apply_to_update(u, g, pg, inv_norm, lr, do_corr, arch_mantissa_bits);
        prev_grad[idx] = g;
    }
}

extern "C" __global__ void s3_row_trust_scale_10(
    const float* __restrict__ params,
    float* __restrict__ updates,
    const unsigned int n_rows,
    const unsigned int n_cols,
    const float lr,
    const float eps,
    const unsigned int arch_mantissa_bits
) {
    extern __shared__ float smem[];
    const unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int cols = n_cols;

    float p_acc = 0.0f;
    float u_acc = 0.0f;
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        const unsigned int idx = row * cols + c;
        const float p = params[idx];
        const float u = updates[idx];
        p_acc = __fmaf_ieee_rn(p, p, p_acc);
        u_acc = __fmaf_ieee_rn(u, u, u_acc);
    }

    const unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        p_acc += __shfl_down_sync(mask, p_acc, offset);
        u_acc += __shfl_down_sync(mask, u_acc, offset);
    }

    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int num_warps = (blockDim.x + 31u) >> 5;
    if (lane == 0u) {
        smem[warp] = p_acc;
        smem[warp + num_warps] = u_acc;
    }
    __syncthreads();

    if (warp == 0u) {
        float p_block = (lane < num_warps) ? smem[lane] : 0.0f;
        float u_block = (lane < num_warps) ? smem[lane + num_warps] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            p_block += __shfl_down_sync(mask, p_block, offset);
            u_block += __shfl_down_sync(mask, u_block, offset);
        }
        if (lane == 0u) {
            float scale = 1.0f;
            const float p_norm = sqrtf(p_block);
            const float u_norm = sqrtf(u_block);
            if (p_norm > 0.0f && u_norm > 0.0f && lr > 0.0f) {
                const float v_norm = __fdiv_rn(u_norm, lr);
                scale = __fdiv_rn(p_norm, __fadd_rn(v_norm, eps));
            }
            smem[0] = scale;
        }
    }
    __syncthreads();

    const float scale = smem[0];
    for (unsigned int c = tid; c < cols; c += blockDim.x) {
        const unsigned int idx = row * cols + c;
        const float scaled = __fmul_rn(updates[idx], scale);
        updates[idx] = s3_canonicalize_rn(scaled, arch_mantissa_bits);
    }
}

