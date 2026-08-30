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

extern "C" __global__ void s3_p3_base_dispatch_update_14(
    const unsigned long long* __restrict__ descriptors,
    const unsigned int descriptor_count,
    const float lr,
    const float beta,
    const unsigned int arch_mantissa_bits
) {
    const unsigned int tile_index = blockIdx.x;
    if (tile_index >= descriptor_count) return;

    const unsigned long long* tile = descriptors + (unsigned long long)tile_index * 6ull;
    const float* __restrict__ gradients = (const float*)(uintptr_t)tile[0];
    const float* __restrict__ params = (const float*)(uintptr_t)tile[1];
    float* __restrict__ momentum = (float*)(uintptr_t)tile[2];
    float* __restrict__ updates = (float*)(uintptr_t)tile[3];
    const unsigned int n = (unsigned int)tile[4];
    const unsigned long long packed = tile[5];
    const unsigned int idx = (unsigned int)packed + threadIdx.x;
    const float weight_decay = __uint_as_float((unsigned int)(packed >> 32));
    const float one_minus_beta = __fsub_rn(1.0f, beta);

    if (idx < n) {
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float g_term = __fmul_rn(one_minus_beta, g);

        float m = momentum[idx];
        m = __fmaf_ieee_rn(beta, m, g_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);

        const float predicted = s3_canonicalize_rn(
            __fmaf_ieee_rn(beta, m, g_term),
            arch_mantissa_bits
        );
        const float direction = predicted > 0.0f
            ? 1.0f
            : (predicted < 0.0f ? -1.0f : 0.0f);

        momentum[idx] = m;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], direction);
        const float update = __fmul_rn(-lr, with_decay);
        updates[idx] = s3_canonicalize_rn(update, arch_mantissa_bits);
    }
}

extern "C" __global__ void s3_p3_headroom_sqrt_update_14(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ power_momentum,
    float* __restrict__ max_power_momentum,
    float* __restrict__ previous_gradients,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta,
    const float eps,
    const float weight_decay,
    const float coherence_gain,
    const unsigned int arch_mantissa_bits,
    const unsigned int diversification,
    const unsigned int seed_key,
    const unsigned int tensor_index
) {
    __shared__ float warp_dots[8];
    __shared__ float warp_norms[8];
    __shared__ float warp_alignment[8];
    __shared__ float warp_gradient_norms[8];
    __shared__ float warp_momentum_norms[8];
    __shared__ float warp_pattern_sums[8];
    __shared__ float warp_pattern_dots[8];
    __shared__ float warp_update_norms[8];
    __shared__ float warp_pattern_norms[8];

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int lane = threadIdx.x % warpSize;
    const unsigned int warp = threadIdx.x / warpSize;
    const float one_minus_beta = __fsub_rn(1.0f, beta);
    const float gain = fminf(fmaxf(coherence_gain, 0.0f), 1.0f);

    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = s3_canonicalize_rn(gradients[idx], arch_mantissa_bits);
        const float ag = fabsf(g);
        const float gp2 = __fmul_rn(ag, ag);
        const float gp3 = __fmul_rn(gp2, ag);

        float m = momentum[idx];
        float s = power_momentum[idx];
        float max_s = max_power_momentum[idx];

        float row_alignment = __fmul_rn(g, m);
        float row_gradient_norm = __fmul_rn(g, g);
        float row_momentum_norm = __fmul_rn(m, m);
        for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
            row_alignment = __fadd_rn(row_alignment, __shfl_down_sync(0xffffffffu, row_alignment, offset));
            row_gradient_norm = __fadd_rn(row_gradient_norm, __shfl_down_sync(0xffffffffu, row_gradient_norm, offset));
            row_momentum_norm = __fadd_rn(row_momentum_norm, __shfl_down_sync(0xffffffffu, row_momentum_norm, offset));
        }
        if (lane == 0) {
            warp_alignment[warp] = row_alignment;
            warp_gradient_norms[warp] = row_gradient_norm;
            warp_momentum_norms[warp] = row_momentum_norm;
        }
        __syncthreads();

        if (warp == 0) {
            row_alignment = lane < blockDim.x / warpSize ? warp_alignment[lane] : 0.0f;
            row_gradient_norm = lane < blockDim.x / warpSize ? warp_gradient_norms[lane] : 0.0f;
            row_momentum_norm = lane < blockDim.x / warpSize ? warp_momentum_norms[lane] : 0.0f;
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                row_alignment = __fadd_rn(row_alignment, __shfl_down_sync(0xffffffffu, row_alignment, offset));
                row_gradient_norm = __fadd_rn(row_gradient_norm, __shfl_down_sync(0xffffffffu, row_gradient_norm, offset));
                row_momentum_norm = __fadd_rn(row_momentum_norm, __shfl_down_sync(0xffffffffu, row_momentum_norm, offset));
            }
            if (lane == 0) {
                warp_alignment[0] = row_alignment;
                warp_gradient_norms[0] = row_gradient_norm;
                warp_momentum_norms[0] = row_momentum_norm;
            }
        }
        __syncthreads();

        const float row_scale = __fmul_rn(
            __fsqrt_rn(warp_gradient_norms[0]),
            __fsqrt_rn(warp_momentum_norms[0])
        );
        const float row_coherence = fmaxf(
            __fdiv_rn(warp_alignment[0], __fadd_rn(row_scale, eps)),
            0.0f
        );

        const float g_term = __fmul_rn(one_minus_beta, g);
        m = __fmaf_ieee_rn(beta, m, g_term);
        const float p_term = __fmul_rn(one_minus_beta, gp3);
        s = __fmaf_ieee_rn(beta, s, p_term);
        m = s3_canonicalize_rn(m, arch_mantissa_bits);
        s = s3_canonicalize_rn(s, arch_mantissa_bits);
        max_s = s3_canonicalize_rn(fmaxf(max_s, s), arch_mantissa_bits);

        const float optimistic_gradient = __fadd_rn(
            g,
            __fsub_rn(g, previous_gradients[idx])
        );
        const float optimistic_term = __fmul_rn(one_minus_beta, optimistic_gradient);
        const float nesterov_num = __fmaf_ieee_rn(beta, m, optimistic_term);
        const float root = s3_cbrt_canonical(max_s, arch_mantissa_bits);
        const float denom = __fadd_rn(root, eps);
        const float normalized = __fdiv_rn(nesterov_num, denom);

        const float base_mag = fminf(fabsf(normalized), 1.0f);
        const float release_shape = __fsqrt_rn(base_mag);
        const float headroom = __fsub_rn(1.0f, base_mag);
        const float gc = __fmul_rn(gain, row_coherence);
        const float shaped_headroom = __fmul_rn(release_shape, headroom);
        const float extra = __fmul_rn(gc, shaped_headroom);
        const float boosted_mag = __fadd_rn(base_mag, extra);
        const float bounded_normalized = copysignf(fminf(boosted_mag, 1.0f), normalized);

        float row_dot = __fmul_rn(bounded_normalized, params[idx]);
        float row_norm = __fmul_rn(params[idx], params[idx]);
        for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
            row_dot = __fadd_rn(row_dot, __shfl_down_sync(0xffffffffu, row_dot, offset));
            row_norm = __fadd_rn(row_norm, __shfl_down_sync(0xffffffffu, row_norm, offset));
        }
        if (lane == 0) {
            warp_dots[warp] = row_dot;
            warp_norms[warp] = row_norm;
        }
        __syncthreads();

        if (warp == 0) {
            row_dot = lane < blockDim.x / warpSize ? warp_dots[lane] : 0.0f;
            row_norm = lane < blockDim.x / warpSize ? warp_norms[lane] : 0.0f;
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                row_dot = __fadd_rn(row_dot, __shfl_down_sync(0xffffffffu, row_dot, offset));
                row_norm = __fadd_rn(row_norm, __shfl_down_sync(0xffffffffu, row_norm, offset));
            }
            if (lane == 0) {
                warp_dots[0] = row_dot;
                warp_norms[0] = row_norm;
            }
        }
        __syncthreads();

        const float projection = __fdiv_rn(
            warp_dots[0],
            __fadd_rn(warp_norms[0], eps)
        );
        const float tangent_normalized = s3_canonicalize_rn(
            __fmaf_ieee_rn(-projection, params[idx], bounded_normalized),
            arch_mantissa_bits
        );

        momentum[idx] = m;
        power_momentum[idx] = s;
        max_power_momentum[idx] = max_s;
        previous_gradients[idx] = g;
        const float with_decay = __fmaf_ieee_rn(weight_decay, params[idx], tangent_normalized);
        const float update = __fmul_rn(-lr, with_decay);

        float diversification_update = 0.0f;
        if (diversification != 0u) {
            unsigned int hash = seed_key;
            hash ^= tensor_index * 0x9e3779b9u;
            hash ^= blockIdx.x * 0x85ebca6bu;
            hash ^= threadIdx.x * 0xc2b2ae35u;
            hash ^= hash >> 16;
            hash *= 0x7feb352du;
            hash ^= hash >> 15;
            hash *= 0x846ca68bu;
            hash ^= hash >> 16;
            const float raw_pattern = (hash & 1u) != 0u ? 1.0f : -1.0f;

            float pattern_sum = raw_pattern;
            float pattern_dot = __fmul_rn(raw_pattern, params[idx]);
            float update_norm = __fmul_rn(update, update);
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                pattern_sum = __fadd_rn(pattern_sum, __shfl_down_sync(0xffffffffu, pattern_sum, offset));
                pattern_dot = __fadd_rn(pattern_dot, __shfl_down_sync(0xffffffffu, pattern_dot, offset));
                update_norm = __fadd_rn(update_norm, __shfl_down_sync(0xffffffffu, update_norm, offset));
            }
            if (lane == 0) {
                warp_pattern_sums[warp] = pattern_sum;
                warp_pattern_dots[warp] = pattern_dot;
                warp_update_norms[warp] = update_norm;
            }
            __syncthreads();

            if (warp == 0) {
                pattern_sum = lane < blockDim.x / warpSize ? warp_pattern_sums[lane] : 0.0f;
                pattern_dot = lane < blockDim.x / warpSize ? warp_pattern_dots[lane] : 0.0f;
                update_norm = lane < blockDim.x / warpSize ? warp_update_norms[lane] : 0.0f;
                for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                    pattern_sum = __fadd_rn(pattern_sum, __shfl_down_sync(0xffffffffu, pattern_sum, offset));
                    pattern_dot = __fadd_rn(pattern_dot, __shfl_down_sync(0xffffffffu, pattern_dot, offset));
                    update_norm = __fadd_rn(update_norm, __shfl_down_sync(0xffffffffu, update_norm, offset));
                }
                if (lane == 0) {
                    warp_pattern_sums[0] = pattern_sum;
                    warp_pattern_dots[0] = pattern_dot;
                    warp_update_norms[0] = update_norm;
                }
            }
            __syncthreads();

            const float row_width = (float)n;
            const float parameter_sum = warp_dots[0];
            const float parameter_norm = warp_norms[0];
            const float determinant = __fsub_rn(
                __fmul_rn(row_width, parameter_norm),
                __fmul_rn(parameter_sum, parameter_sum)
            );

            float tangent_pattern = 0.0f;
            if (determinant != 0.0f) {
                const float mean_coefficient = __fdiv_rn(
                    __fsub_rn(
                        __fmul_rn(warp_pattern_sums[0], parameter_norm),
                        __fmul_rn(warp_pattern_dots[0], parameter_sum)
                    ),
                    determinant
                );
                const float radial_coefficient = __fdiv_rn(
                    __fsub_rn(
                        __fmul_rn(row_width, warp_pattern_dots[0]),
                        __fmul_rn(parameter_sum, warp_pattern_sums[0])
                    ),
                    determinant
                );
                tangent_pattern = __fsub_rn(
                    __fsub_rn(raw_pattern, mean_coefficient),
                    __fmul_rn(radial_coefficient, params[idx])
                );
            }

            float tangent_norm = __fmul_rn(tangent_pattern, tangent_pattern);
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                tangent_norm = __fadd_rn(
                    tangent_norm,
                    __shfl_down_sync(0xffffffffu, tangent_norm, offset)
                );
            }
            if (lane == 0) {
                warp_pattern_norms[warp] = tangent_norm;
            }
            __syncthreads();

            if (warp == 0) {
                tangent_norm = lane < blockDim.x / warpSize ? warp_pattern_norms[lane] : 0.0f;
                for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                    tangent_norm = __fadd_rn(
                        tangent_norm,
                        __shfl_down_sync(0xffffffffu, tangent_norm, offset)
                    );
                }
                if (lane == 0) {
                    warp_pattern_norms[0] = tangent_norm;
                }
            }
            __syncthreads();

            const float diversification_scale = __fdiv_rn(
                __fsqrt_rn(warp_update_norms[0]),
                __fadd_rn(__fsqrt_rn(warp_pattern_norms[0]), eps)
            );
            diversification_update = __fmul_rn(diversification_scale, tangent_pattern);
        }
        updates[idx] = s3_canonicalize_rn(
            __fadd_rn(update, diversification_update),
            arch_mantissa_bits
        );
    }
}

