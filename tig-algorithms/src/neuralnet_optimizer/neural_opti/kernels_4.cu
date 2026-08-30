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

extern "C" __global__ void s3_p3_base_update_4(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ monotone_cubic_envelope,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float nesterov_bias_correction,
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

        float envelope = monotone_cubic_envelope[idx];
        envelope = s3_canonicalize_rn(fmaxf(envelope, s), arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float root = s3_cbrt_canonical(envelope, arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        momentum[idx] = m;
        power_momentum[idx] = s;
        monotone_cubic_envelope[idx] = envelope;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], normalized);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_p3_headroom_sqrt_update_4(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ monotone_cubic_envelope,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const unsigned int arch_mantissa_bits
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

        float confidence = 0.0f;
        const bool aligned = ((g > 0.0f && m > 0.0f) || (g < 0.0f && m < 0.0f));
        if (aligned && s > 0.0f) {
            const float hist_scale = s3_cbrt_canonical(s, arch_mantissa_bits);
            const float directional = __fdiv_rn(fabsf(m), fmaxf(hist_scale, eps));
            const float bounded_directional = fminf(directional, 1.0f);
            confidence = __fmul_rn(bounded_directional, bounded_directional);
        }

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float p_term = __fmul_rn(one_minus_beta, gp3);
        s = __fmaf_ieee_rn(beta, s, p_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);

        float envelope = monotone_cubic_envelope[idx];
        envelope = s3_canonicalize_rn(fmaxf(envelope, s), arch_mantissa_bits);

        const float nesterov_num = __fmaf_ieee_rn(beta, m, g_term);
        const float root = s3_cbrt_canonical(envelope, arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
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
        monotone_cubic_envelope[idx] = envelope;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], bounded_normalized);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_p3_affine_row_update_4(
    const float* __restrict__ weight_gradients,
    const float* __restrict__ weights,
    const float* __restrict__ bias_gradients,
    const float* __restrict__ biases,
    float* __restrict__ weight_momentum,
    float* __restrict__ bias_momentum,
    float* __restrict__ weight_directional_momentum,
    float* __restrict__ bias_directional_momentum,
    float* __restrict__ row_power_momentum,
    float* __restrict__ weight_updates,
    float* __restrict__ bias_updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int row = blockIdx.x;
    const unsigned int column = threadIdx.x;
    const unsigned int idx = row * blockDim.x + column;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const float gain = fminf(fmaxf(coherence_gain, 0.0f), 1.0f);
    __shared__ float affine_cubic[256];
    __shared__ float s3_affine_completed_4[256];

    const float g = s3_canonicalize_rn(weight_gradients[idx], arch_mantissa_bits);
    const float ag = fabsf(g);
    const float gp3 = __fmul_rn(__fmul_rn(ag, ag), ag);
    float m = weight_momentum[idx];
    float persistent = weight_directional_momentum[idx];

    const float historical_root =
        s3_cbrt_canonical(row_power_momentum[row], arch_mantissa_bits);
    float confidence = 0.0f;
    if (((g > 0.0f && m > 0.0f) || (g < 0.0f && m < 0.0f))
        && row_power_momentum[row] > 0.0f) {
        const float directional = __fdiv_rn(fabsf(m), fmaxf(historical_root, eps));
        const float bounded_directional = fminf(directional, 1.0f);
        confidence = __fmul_rn(bounded_directional, bounded_directional);
    }

    affine_cubic[column] = gp3;
    if (column == 0u) {
        const float bg = s3_canonicalize_rn(bias_gradients[row], arch_mantissa_bits);
        const float abg = fabsf(bg);
        const float bgp3 = __fmul_rn(__fmul_rn(abg, abg), abg);
        affine_cubic[0] = __fadd_rn(affine_cubic[0], bgp3);
    }
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (column < offset) {
            affine_cubic[column] =
                __fadd_rn(affine_cubic[column], affine_cubic[column + offset]);
        }
        __syncthreads();
    }

    if (column == 0u) {
        const float augmented_mean =
            __fdiv_rn(affine_cubic[0], __uint2float_rn(blockDim.x + 1u));
        const float previous = row_power_momentum[row];
        row_power_momentum[row] = s3_canonicalize_rn(
            __fmaf_ieee_rn(beta, previous, __fmul_rn(one_minus_beta, augmented_mean)),
            arch_mantissa_bits
        );
        affine_cubic[0] = row_power_momentum[row];
    }
    __syncthreads();

    const float g_term = __fmul_rn(one_minus_beta, g);
    m = s3_canonicalize_rn(__fmaf_ieee_rn(beta, m, g_term), arch_mantissa_bits);
    persistent = s3_canonicalize_rn(
        __fmaf_ieee_rn(beta, persistent, __fmul_rn(one_minus_beta, m)),
        arch_mantissa_bits
    );
    const float transient = __fsub_rn(m, persistent);
    const float directional_num = __fmaf_ieee_rn(one_minus_beta, transient, persistent);
    const float nesterov_num = __fadd_rn(directional_num, g_term);
    const float row_root = s3_cbrt_canonical(affine_cubic[0], arch_mantissa_bits);
    const float normalized = __fdiv_rn(nesterov_num, __fadd_rn(row_root, eps));
    const float base_mag = fminf(fabsf(normalized), 1.0f);
    const float extra = __fmul_rn(
        __fmul_rn(gain, confidence),
        __fmul_rn(__fsqrt_rn(base_mag), __fsub_rn(1.0f, base_mag))
    );
    const float bounded_normalized =
        copysignf(fminf(__fadd_rn(base_mag, extra), 1.0f), normalized);

    weight_momentum[idx] = m;
    weight_directional_momentum[idx] = persistent;
    const float completed_weight_update = s3_canonicalize_rn(
        __fmul_rn(-lr, __fmaf_ieee_rn(weight_decay, weights[idx], bounded_normalized)),
        arch_mantissa_bits
    );

    float completed_bias_update = 0.0f;
    s3_affine_completed_4[column] = completed_weight_update;

    if (column == 0u) {
        const float bg = s3_canonicalize_rn(bias_gradients[row], arch_mantissa_bits);
        float bm = bias_momentum[row];
        float bias_persistent = bias_directional_momentum[row];
        float bias_confidence = 0.0f;
        if (((bg > 0.0f && bm > 0.0f) || (bg < 0.0f && bm < 0.0f))
            && affine_cubic[0] > 0.0f) {
            const float directional = __fdiv_rn(fabsf(bm), fmaxf(historical_root, eps));
            const float bounded_directional = fminf(directional, 1.0f);
            bias_confidence = __fmul_rn(bounded_directional, bounded_directional);
        }

        const float bg_term = __fmul_rn(one_minus_beta, bg);
        bm = s3_canonicalize_rn(__fmaf_ieee_rn(beta, bm, bg_term), arch_mantissa_bits);
        bias_persistent = s3_canonicalize_rn(
            __fmaf_ieee_rn(beta, bias_persistent, __fmul_rn(one_minus_beta, bm)),
            arch_mantissa_bits
        );
        const float bias_transient = __fsub_rn(bm, bias_persistent);
        const float bias_directional_num = __fmaf_ieee_rn(
            one_minus_beta,
            bias_transient,
            bias_persistent
        );
        const float bias_num = __fadd_rn(bias_directional_num, bg_term);
        const float bias_normalized = __fdiv_rn(bias_num, __fadd_rn(row_root, eps));
        const float bias_base_mag = fminf(fabsf(bias_normalized), 1.0f);
        const float bias_extra = __fmul_rn(
            __fmul_rn(gain, bias_confidence),
            __fmul_rn(__fsqrt_rn(bias_base_mag), __fsub_rn(1.0f, bias_base_mag))
        );
        const float bounded_bias =
            copysignf(fminf(__fadd_rn(bias_base_mag, bias_extra), 1.0f), bias_normalized);

        bias_momentum[row] = bm;
        bias_directional_momentum[row] = bias_persistent;
        completed_bias_update = s3_canonicalize_rn(
            __fmul_rn(-lr, bounded_bias),
            arch_mantissa_bits
        );
        s3_affine_completed_4[0] =
            __fadd_rn(completed_weight_update, completed_bias_update);
    }
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1u; offset > 0u; offset >>= 1u) {
        if (column < offset) {
            s3_affine_completed_4[column] = __fadd_rn(
                s3_affine_completed_4[column],
                s3_affine_completed_4[column + offset]
            );
        }
        __syncthreads();
    }

    const float affine_shared_component = __fdiv_rn(
        s3_affine_completed_4[0],
        __uint2float_rn(blockDim.x + 1u)
    );
    weight_updates[idx] = s3_canonicalize_rn(
        __fsub_rn(completed_weight_update, affine_shared_component),
        arch_mantissa_bits
    );

    if (column == 0u) {
        bias_updates[row] = s3_canonicalize_rn(
            __fsub_rn(completed_bias_update, affine_shared_component),
            arch_mantissa_bits
        );
    }
}
