#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void sk_adan_14(
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

extern "C" __global__ void sk_kink_14(
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

extern "C" __global__ void sk_kink2_14(
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

extern "C" __global__ void sk_offset_rollback_14(
    float* __restrict__ delta,
    float* __restrict__ previous,
    float amount,
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

extern "C" __global__ void sk_scale_14(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] = factor * src[i];
}

struct AdanDescriptor {
    unsigned long long grad;
    unsigned long long param;
    unsigned long long m;
    unsigned long long v;
    unsigned long long nsq;
    unsigned long long gprev;
    unsigned long long delta;
    unsigned long long previous;
    float lr;
    float b1;
    float b2;
    float b3;
    float eps;
    float wd;
    float bc1;
    float bc2;
    float bc3;
    float cautious;
    int first_step;
    int n;
    int block_start;
    int block_end;
};

extern "C" __global__ void sk_batched_adan_rollback_14(
    const AdanDescriptor* __restrict__ descriptors,
    int descriptor_count
) {
    int block = blockIdx.x;
    int descriptor_index = 0;
    while (descriptor_index < descriptor_count &&
           block >= descriptors[descriptor_index].block_end) {
        ++descriptor_index;
    }
    if (descriptor_index >= descriptor_count) return;

    const AdanDescriptor d = descriptors[descriptor_index];
    int i = (block - d.block_start) * blockDim.x + threadIdx.x;
    if (i >= d.n) return;

    const float* grad = (const float*)(uintptr_t)d.grad;
    const float* param = (const float*)(uintptr_t)d.param;
    float* m = (float*)(uintptr_t)d.m;
    float* v = (float*)(uintptr_t)d.v;
    float* nsq = (float*)(uintptr_t)d.nsq;
    float* gprev = (float*)(uintptr_t)d.gprev;
    float* delta = (float*)(uintptr_t)d.delta;
    float* previous = (float*)(uintptr_t)d.previous;

    float g = grad[i];
    float p = param[i];
    float gd = (d.first_step != 0) ? 0.0f : (g - gprev[i]);
    gprev[i] = g;

    float mi = (1.0f - d.b1) * g + d.b1 * m[i];
    float vi = (1.0f - d.b2) * gd + d.b2 * v[i];
    float comb = g + d.b2 * gd;
    float ni = (1.0f - d.b3) * comb * comb + d.b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;

    float step = (mi / d.bc1 + d.b2 * vi / d.bc2) / (sqrtf(ni / d.bc3) + d.eps);
    float upd = -d.lr * step;
    if (d.cautious < 1.0f && upd * g > 0.0f) upd *= d.cautious;
    float np = (p + upd) / (1.0f + d.lr * d.wd);
    float current = np - p;
    float prior = previous[i];
    float applied = (d.first_step == 0 && current * prior < 0.0f) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_matrix_adan_precond_rollback_14(
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
    int cols,
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
    float raw = np - p;

    int row = i / cols;
    int col = i - row * cols;
    float current = raw / sqrtf(row_rms[row] * col_rms[col] + eps);
    float prior = previous[i];
    float applied = (first_step == 0 && current * prior < 0.0f) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_rollback_14(
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

extern "C" __global__ void sk_axis_rms_14(
    const float* __restrict__ grad,
    float* __restrict__ row_rms,
    float* __restrict__ col_rms,
    int rows,
    int cols
) {
    int block = blockIdx.x;
    int is_col = block >= rows;
    int group = is_col ? block - rows : block;
    int tid = threadIdx.x;
    int count = is_col ? rows : cols;
    float sum = 0.0f;

    for (int j = tid; j < count; j += blockDim.x) {
        int idx = is_col ? j * cols + group : group * cols + j;
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
            if (is_col) {
                col_rms[group] = sqrtf(sum / (float)count);
            } else {
                row_rms[group] = sqrtf(sum / (float)count);
            }
        }
    }
}

extern "C" __global__ void sk_matrix_precond_14(
    float* __restrict__ delta,
    const float* __restrict__ row_rms,
    const float* __restrict__ col_rms,
    int cols,
    float eps,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float src = delta[i];
    int row = i / cols;
    int col = i - row * cols;
    float denom = sqrtf(row_rms[row] * col_rms[col] + eps);
    delta[i] = src / denom;
}
