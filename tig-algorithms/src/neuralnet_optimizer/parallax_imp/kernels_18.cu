#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void sk_adan_18(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ previous,
    float lr,
    float b1,
    float b2,
    float b3,
    float eps,
    float wd,
    float bc1,
    float bc2,
    float bc3,
    int first_step,
    float cautious,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float p = param[i];
    float gd = (first_step != 0) ? 0.0f : (g - gprev[i]);
    gprev[i] = g;
    float mi = (1.0f - b1) * g + b1 * m[i];
    float vi = (1.0f - b2) * gd + b2 * v[i];
    float comb = g + b2 * gd;
    float ni = (1.0f - b3) * comb * comb + b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;
    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1.0f && upd * g > 0.0f) upd *= cautious;
    float np = (p + upd) / (1.0f + lr * wd);
    float current = np - p;
    float prior = previous[i];
    current = (first_step == 0 && current * prior < 0.0f) ? -prior : current;
    previous[i] = current;
    delta[i] = current;
}

extern "C" __global__ void sk_adan_no_rollback_18(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float lr,
    float b1,
    float b2,
    float b3,
    float eps,
    float wd,
    float bc1,
    float bc2,
    float bc3,
    int first_step,
    float cautious,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float p = param[i];
    float gd = (first_step != 0) ? 0.0f : (g - gprev[i]);
    gprev[i] = g;
    float mi = (1.0f - b1) * g + b1 * m[i];
    float vi = (1.0f - b2) * gd + b2 * v[i];
    float comb = g + b2 * gd;
    float ni = (1.0f - b3) * comb * comb + b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;
    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1.0f && upd * g > 0.0f) upd *= cautious;
    float np = (p + upd) / (1.0f + lr * wd);
    delta[i] = np - p;
}

extern "C" __global__ void sk_adan_matrix_18(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ previous,
    const float* __restrict__ row_rms,
    const float* __restrict__ col_rms,
    float lr,
    float b1,
    float b2,
    float b3,
    float eps,
    float wd,
    float bc1,
    float bc2,
    float bc3,
    int first_step,
    float cautious,
    int cols,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float p = param[i];
    float gd = (first_step != 0) ? 0.0f : (g - gprev[i]);
    gprev[i] = g;
    float mi = (1.0f - b1) * g + b1 * m[i];
    float vi = (1.0f - b2) * gd + b2 * v[i];
    float comb = g + b2 * gd;
    float ni = (1.0f - b3) * comb * comb + b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;
    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1.0f && upd * g > 0.0f) upd *= cautious;
    float np = (p + upd) / (1.0f + lr * wd);
    float current = np - p;
    int row = i / cols;
    int col = i - row * cols;
    float denom = sqrtf(row_rms[row] * col_rms[col] + eps);
    current = current / denom;
    float prior = previous[i];
    float applied = (first_step == 0 && current * prior < 0.0f) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_kink_18(
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

extern "C" __global__ void sk_kink2_18(
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

extern "C" __global__ void sk_offset_18(
    float* __restrict__ delta,
    float amount,
    float* __restrict__ previous,
    int first_step,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float current = delta[i] + amount;
    float prior = previous[i];
    float applied = (first_step == 0 && current * prior < 0.0f) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_scale_18(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] = factor * src[i];
}

extern "C" __global__ void sk_rollback_18(
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

extern "C" __global__ void sk_axis_rms_18(
    const float* __restrict__ grad,
    float* __restrict__ row_rms,
    float* __restrict__ col_rms,
    int rows,
    int cols
) {
    int group = blockIdx.x;
    int axis = blockIdx.y;
    int tid = threadIdx.x;
    int count = (axis == 0) ? cols : rows;
    if (group >= ((axis == 0) ? rows : cols)) return;
    float sum = 0.0f;
    for (int j = tid; j < count; j += blockDim.x) {
        int idx = (axis == 0) ? group * cols + j : j * cols + group;
        float g = grad[idx];
        sum += g * g;
    }
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }
    __shared__ float warp_sums[32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        int warp_count = (blockDim.x + 31) >> 5;
        sum = (tid < warp_count) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (tid == 0) {
            if (axis == 0) row_rms[group] = sqrtf(sum / (float)count);
            else col_rms[group] = sqrtf(sum / (float)count);
        }
    }
}

extern "C" __global__ void sk_matrix_precond_18(
    float* __restrict__ delta,
    const float* __restrict__ src,
    const float* __restrict__ row_rms,
    const float* __restrict__ col_rms,
    int cols,
    float eps,
    int n,
    float* __restrict__ previous,
    int first_step
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = i / cols;
    int col = i - row * cols;
    float denom = sqrtf(row_rms[row] * col_rms[col] + eps);
    float current = src[i] / denom;
    float prior = previous[i];
    float applied = (first_step == 0 && current * prior < 0.0f) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}
