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

extern "C" __global__ void s3_p3_base_update_7(
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
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float ag = fabsf(g);
        const float gp2 = __fmul_rn(ag, ag);
        const float gp3 = __fmul_rn(gp2, ag);

        float m = momentum[idx];
        float s = power_momentum[idx];

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float p_term = __fmul_rn(one_minus_beta, gp3);
        s = __fmaf_ieee_rn(beta, s, p_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_pow = __fmaf_ieee_rn(beta, s, p_term);
        const float root = s3_cbrt_canonical(nesterov_pow, arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        momentum[idx] = m;
        power_momentum[idx] = s;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], normalized);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_p3_headroom_sqrt_update_7(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ prev_grad,
    float* __restrict__ prev_grad2,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const unsigned int arch_mantissa_bits,
    const int enable_consensus_denom_mix
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const float gain = fminf(fmaxf(coherence_gain, 0.0f), 1.0f);

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float ag = fabsf(g);
        const float gp2 = __fmul_rn(ag, ag);
        const float gp3 = __fmul_rn(gp2, ag);

        float m = momentum[idx];
        float s = power_momentum[idx];
        const float g1 = prev_grad[idx];
        const float g2 = prev_grad2[idx];
        const float a1 = fabsf(g1);
        const float a2 = fabsf(g2);

        float confidence = 0.0f;
        const bool pos = (g > 0.0f && g1 > 0.0f && g2 > 0.0f);
        const bool neg = (g < 0.0f && g1 < 0.0f && g2 < 0.0f);
        const bool consensus = (pos || neg) && ag > 0.0f;
        if (consensus) {
            const float hi = fmaxf(ag, fmaxf(a1, a2));
            const float lo = fminf(ag, fminf(a1, a2));
            const float q = __fdiv_rn(lo, fmaxf(hi, eps));
            const float q2 = __fmul_rn(q, q);
            confidence = __fmul_rn(q2, q2);
        }

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float p_term = __fmul_rn(one_minus_beta, gp3);
        s = __fmaf_ieee_rn(beta, s, p_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float nesterov_pow = __fmaf_ieee_rn(beta, s, p_term);
        const float root = s3_cbrt_canonical(nesterov_pow, arch_mantissa_bits);
        float denom_core = root;
        if (enable_consensus_denom_mix != 0 && consensus) {
            const float mean_abs = __fmul_rn(
                __fadd_rn(ag, __fadd_rn(a1, a2)),
                0.333333343f
            );
            const float harm_num = __fmul_rn(__fmul_rn(2.0f, root), mean_abs);
            const float harm_den = __fadd_rn(__fadd_rn(root, mean_abs), eps);
            denom_core = __fdiv_rn(harm_num, harm_den);
        }
        const float denom = __fadd_rn(denom_core, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        const float base_mag = fminf(fabsf(normalized), 1.0f);
        const float release_shape = __fsqrt_rn(base_mag);
        const float headroom = __fsub_rn(1.0f, base_mag);
        const float gc = __fmul_rn(gain, confidence);
        const float shaped_headroom = __fmul_rn(release_shape, headroom);
        const float extra = __fmul_rn(gc, shaped_headroom);
        const float boosted_mag = __fadd_rn(base_mag, extra);
        const float bounded_normalized = copysignf(fminf(boosted_mag, 1.0f), normalized);

        momentum[idx] = m;
        power_momentum[idx] = s;        
        prev_grad2[idx] = g1;
        prev_grad[idx] = g;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], bounded_normalized);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_p3_bn_affine_update_7(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ power_momentum,
    float* __restrict__ ema_g2,
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
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float ag = fabsf(g);
        const float gp2 = __fmul_rn(ag, ag);
        const float gp3 = __fmul_rn(gp2, ag);
        const float g2 = __fmul_rn(g, g);

        float s = power_momentum[idx];
        float v = ema_g2[idx];

        const float p_term = __fmul_rn(one_minus_beta, gp3);
        s = __fmaf_ieee_rn(beta, s, p_term);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        const float v_term = __fmul_rn(one_minus_beta, g2);
        v = __fmaf_ieee_rn(beta, v, v_term);
        v = s3_canonicalize_rn(v, arch_mantissa_bits);

        const float root = s3_cbrt_canonical(s, arch_mantissa_bits);
        const float denom_s = __fadd_rn(root, eps);
        float normalized = __fdiv_rn(g, denom_s);

        const float rms = __fsqrt_rn(fmaxf(v, 0.0f));
        const float denom_v = __fadd_rn(rms, eps);
        normalized = __fdiv_rn(normalized, denom_v);

        power_momentum[idx] = s;
        ema_g2[idx] = v;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], normalized);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}
