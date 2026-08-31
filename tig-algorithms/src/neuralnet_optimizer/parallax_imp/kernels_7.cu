#include <cuda_runtime.h>
#include <math.h>

template<int FIRST_STEP, int APPLY_RB>
__device__ __forceinline__ float sk_adan_one_t(
    float g, float p, float gp, float mo, float vo, float nso,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    float* __restrict__ mo_out, float* __restrict__ vo_out,
    float* __restrict__ nso_out, float* __restrict__ gp_out
) {
    float gd = FIRST_STEP ? 0.0f : (g - gp);
    *gp_out = g;
    float mi = (1.0f - b1) * g + b1 * mo;
    float vi = (1.0f - b2) * gd + b2 * vo;
    float comb = g + b2 * gd;
    float ni = (1.0f - b3) * comb * comb + b3 * nso;
    *mo_out = mi;
    *vo_out = vi;
    *nso_out = ni;
    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1.0f && upd * g > 0.0f) upd *= cautious;
    float np = (p + upd) / (1.0f + lr * wd);
    return np - p;
}

template<int FIRST_STEP, int APPLY_RB>
__device__ __forceinline__ void sk_adan_f4_chunk_t(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ previous,
    int i4,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious
) {
    float4 g4  = *reinterpret_cast<const float4*>(grad + i4);
    float4 p4  = *reinterpret_cast<const float4*>(param + i4);
    float4 m4  = *reinterpret_cast<float4*>(m + i4);
    float4 v4  = *reinterpret_cast<float4*>(v + i4);
    float4 n4  = *reinterpret_cast<float4*>(nsq + i4);
    float4 gp4 = *reinterpret_cast<float4*>(gprev + i4);

    float d0 = sk_adan_one_t<FIRST_STEP, APPLY_RB>(g4.x, p4.x, gp4.x, m4.x, v4.x, n4.x, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, &m4.x, &v4.x, &n4.x, &gp4.x);
    float d1 = sk_adan_one_t<FIRST_STEP, APPLY_RB>(g4.y, p4.y, gp4.y, m4.y, v4.y, n4.y, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, &m4.y, &v4.y, &n4.y, &gp4.y);
    float d2 = sk_adan_one_t<FIRST_STEP, APPLY_RB>(g4.z, p4.z, gp4.z, m4.z, v4.z, n4.z, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, &m4.z, &v4.z, &n4.z, &gp4.z);
    float d3 = sk_adan_one_t<FIRST_STEP, APPLY_RB>(g4.w, p4.w, gp4.w, m4.w, v4.w, n4.w, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, &m4.w, &v4.w, &n4.w, &gp4.w);

    *reinterpret_cast<float4*>(m + i4) = m4;
    *reinterpret_cast<float4*>(v + i4) = v4;
    *reinterpret_cast<float4*>(nsq + i4) = n4;
    *reinterpret_cast<float4*>(gprev + i4) = gp4;

    if (APPLY_RB) {
        if (FIRST_STEP) {
            float4 a4 = make_float4(d0, d1, d2, d3);
            *reinterpret_cast<float4*>(delta + i4) = a4;
            *reinterpret_cast<float4*>(previous + i4) = a4;
        } else {
            float4 pr4 = *reinterpret_cast<float4*>(previous + i4);
            float a0 = (d0 * pr4.x < 0.0f) ? -pr4.x : d0;
            float a1 = (d1 * pr4.y < 0.0f) ? -pr4.y : d1;
            float a2 = (d2 * pr4.z < 0.0f) ? -pr4.z : d2;
            float a3 = (d3 * pr4.w < 0.0f) ? -pr4.w : d3;
            float4 a4 = make_float4(a0, a1, a2, a3);
            *reinterpret_cast<float4*>(delta + i4) = a4;
            *reinterpret_cast<float4*>(previous + i4) = a4;
        }
    } else {
        *reinterpret_cast<float4*>(delta + i4) = make_float4(d0, d1, d2, d3);
    }
}

template<int FIRST_STEP, int APPLY_RB>
__device__ __forceinline__ void sk_adan_body(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ previous,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i8 = tid << 3;

    if (i8 + 7 < n) {
        sk_adan_f4_chunk_t<FIRST_STEP, APPLY_RB>(grad, param, m, v, nsq, gprev, delta, previous, i8,
            lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
        sk_adan_f4_chunk_t<FIRST_STEP, APPLY_RB>(grad, param, m, v, nsq, gprev, delta, previous, i8 + 4,
            lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
        return;
    }

    if (i8 + 3 < n) {
        sk_adan_f4_chunk_t<FIRST_STEP, APPLY_RB>(grad, param, m, v, nsq, gprev, delta, previous, i8,
            lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
        i8 += 4;
    }

    for (int i = i8; i < n; ++i) {
        float g = grad[i];
        float p = param[i];
        float d = sk_adan_one_t<FIRST_STEP, APPLY_RB>(g, p, gprev[i], m[i], v[i], nsq[i], lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, &m[i], &v[i], &nsq[i], &gprev[i]);
        if (APPLY_RB) {
            if (FIRST_STEP) {
                delta[i] = d;
                previous[i] = d;
            } else {
                float prior = previous[i];
                float applied = (d * prior < 0.0f) ? -prior : d;
                delta[i] = applied;
                previous[i] = applied;
            }
        } else {
            delta[i] = d;
        }
    }
}

extern "C" __global__ void sk_adan_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    float* __restrict__ previous, int n
) {
    sk_adan_body<0, 1>(grad, param, m, v, nsq, gprev, delta, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, n);
}
extern "C" __global__ void sk_adan_fs_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    float* __restrict__ previous, int n
) {
    sk_adan_body<1, 1>(grad, param, m, v, nsq, gprev, delta, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, n);
}
extern "C" __global__ void sk_adan_nr_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    float* __restrict__ previous, int n
) {
    sk_adan_body<0, 0>(grad, param, m, v, nsq, gprev, delta, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, n);
}
extern "C" __global__ void sk_adan_fs_nr_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    float* __restrict__ previous, int n
) {
    sk_adan_body<1, 0>(grad, param, m, v, nsq, gprev, delta, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious, n);
}

extern "C" __global__ void sk_kink_7(
    float* __restrict__ delta,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float spread,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = spread * (-1.0f + 2.0f * ((float)i + 0.5f) / (float)n);
    delta[i] = -w[i] * t - b[i];
}

extern "C" __global__ void sk_kink2_7(
    float* __restrict__ delta,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float f_lin,
    float span,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int n_lin = (int)(f_lin * (float)n);
    if (n_lin > n) n_lin = n;
    float t;
    if (i < n_lin) {
        t = (w[i] >= 0.0f) ? -4.0f : 4.0f;
    } else {
        int m = n - n_lin;
        int j = i - n_lin;
        t = (m > 0) ? span * (-1.0f + 2.0f * ((float)j + 0.5f) / (float)m) : 0.0f;
    }
    delta[i] = -w[i] * t - b[i];
}

extern "C" __global__ void sk_offset_7(
    float* __restrict__ delta,
    float amount,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] += amount;
}

extern "C" __global__ void sk_scale_7(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] = factor * src[i];
}

extern "C" __global__ void sk_rollback_7(
    float* __restrict__ delta,
    float* __restrict__ previous,
    int first_step,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float current = delta[i];
    float prior = previous[i];
    float applied = (first_step == 0 && current * prior < 0.0f) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_dual_axis_rms_7(
    const float* __restrict__ grad,
    float* __restrict__ row_rms,
    float* __restrict__ col_rms,
    int rows,
    int cols
) {
    const int tid = threadIdx.x; 
    const unsigned mask = 0xffffffffu;
    float sum = 0.0f;
    int count;
    float* out;
    int group;

    if ((int)blockIdx.x < rows) {
        group = (int)blockIdx.x;
        count = cols;
        out = row_rms;
        const float* row_ptr = grad + (size_t)group * (size_t)cols;
        const int cols4 = cols & ~3;
        for (int j = tid * 4; j < cols4; j += 128) { 
            float4 g4 = *reinterpret_cast<const float4*>(row_ptr + j);
            sum += g4.x * g4.x + g4.y * g4.y + g4.z * g4.z + g4.w * g4.w;
        }
        for (int j = cols4 + tid; j < cols; j += 32) {
            float g = __ldg(row_ptr + j);
            sum += g * g;
        }
    } else {
        group = (int)blockIdx.x - rows;
        if (group >= cols) return;
        count = rows;
        out = col_rms;
        for (int j = tid; j < rows; j += 64) {
            float g0 = __ldg(grad + (size_t)j * (size_t)cols + group);
            sum += g0 * g0;
            int j1 = j + 32;
            if (j1 < rows) {
                float g1 = __ldg(grad + (size_t)j1 * (size_t)cols + group);
                sum += g1 * g1;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    if (tid == 0) out[group] = sqrtf(sum / (float)count);
}

template<int FIRST_STEP, int APPLY_RB>
__device__ __forceinline__ void sk_adan_precond_tiled_body(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    const float* __restrict__ row_rms,
    const float* __restrict__ col_rms,
    int rows,
    int cols,
    float* __restrict__ previous,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious
) {
    constexpr int TILE_R = 8;
    constexpr int TILE_C = 32;
    __shared__ float s_row[TILE_R];
    __shared__ float s_col[TILE_C];

    const int row_base = (int)blockIdx.y * TILE_R;
    const int col_base = (int)blockIdx.x * TILE_C;
    const int tid = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;

    if (tid < TILE_R) {
        int r = row_base + tid;
        s_row[tid] = (r < rows) ? __ldg(row_rms + r) : 0.0f;
    }
    if (tid < TILE_C) {
        int c = col_base + tid;
        s_col[tid] = (c < cols) ? __ldg(col_rms + c) : 0.0f;
    }
    __syncthreads();

    const int tr = (int)threadIdx.y;
    const int tc = (int)threadIdx.x;
    const int row = row_base + tr;
    const int col = col_base + tc;
    if (row >= rows || col >= cols) return;

    const int i = row * cols + col;
    float g = __ldg(grad + i);
    float p = __ldg(param + i);
    float d = sk_adan_one_t<FIRST_STEP, APPLY_RB>(
        g, p, gprev[i], m[i], v[i], nsq[i],
        lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious,
        &m[i], &v[i], &nsq[i], &gprev[i]);
    d /= sqrtf(s_row[tr] * s_col[tc] + eps);

    if (APPLY_RB) {
        if (FIRST_STEP) {
            delta[i] = d;
            previous[i] = d;
        } else {
            float prior = previous[i];
            float applied = (d * prior < 0.0f) ? -prior : d;
            delta[i] = applied;
            previous[i] = applied;
        }
    } else {
        delta[i] = d;
    }
}

extern "C" __global__ void sk_adan_precond_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    const float* __restrict__ row_rms, const float* __restrict__ col_rms,
    int rows, int cols,
    float* __restrict__ previous
) {
    sk_adan_precond_tiled_body<0, 1>(grad, param, m, v, nsq, gprev, delta, row_rms, col_rms, rows, cols, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
}
extern "C" __global__ void sk_adan_precond_fs_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    const float* __restrict__ row_rms, const float* __restrict__ col_rms,
    int rows, int cols,
    float* __restrict__ previous
) {
    sk_adan_precond_tiled_body<1, 1>(grad, param, m, v, nsq, gprev, delta, row_rms, col_rms, rows, cols, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
}
extern "C" __global__ void sk_adan_precond_nr_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    const float* __restrict__ row_rms, const float* __restrict__ col_rms,
    int rows, int cols,
    float* __restrict__ previous
) {
    sk_adan_precond_tiled_body<0, 0>(grad, param, m, v, nsq, gprev, delta, row_rms, col_rms, rows, cols, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
}
extern "C" __global__ void sk_adan_precond_fs_nr_7(
    const float* __restrict__ grad, const float* __restrict__ param,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ nsq,
    float* __restrict__ gprev, float* __restrict__ delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3, float cautious,
    const float* __restrict__ row_rms, const float* __restrict__ col_rms,
    int rows, int cols,
    float* __restrict__ previous
) {
    sk_adan_precond_tiled_body<1, 0>(grad, param, m, v, nsq, gprev, delta, row_rms, col_rms, rows, cols, previous, lr, b1, b2, b3, eps, wd, bc1, bc2, bc3, cautious);
}
