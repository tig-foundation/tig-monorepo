#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ void adan_body_10(
    int i,
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ last_delta,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3,
    int first_step, float cautious, int apply_rb
) {
    float g = __ldg(&grad[i]);
    float p = __ldg(&param[i]);
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
    float d = np - p;

    if (apply_rb != 0) {
        float prior = last_delta[i];
        float applied = (first_step == 0 && d * prior < 0.0f) ? -prior : d;
        delta[i] = applied;
        last_delta[i] = applied;
    } else {
        delta[i] = d;
    }
}

__device__ __forceinline__ void adan_precond_body_10(
    int i,
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ last_delta,
    const float* __restrict__ row_rms,
    const float* __restrict__ col_rms,
    float lr, float b1, float b2, float b3, float eps, float wd,
    float bc1, float bc2, float bc3,
    int first_step, float cautious, int cols, int apply_rb
) {
    float g = __ldg(&grad[i]);
    float p = __ldg(&param[i]);
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
    float d = np - p;

    int row = i / cols;
    int col = i - row * cols;
    float denom = sqrtf(__ldg(&row_rms[row]) * __ldg(&col_rms[col]) + eps);
    d = d / denom;

    if (apply_rb != 0) {
        float prior = last_delta[i];
        float applied = (first_step == 0 && d * prior < 0.0f) ? -prior : d;
        delta[i] = applied;
        last_delta[i] = applied;
    } else {
        delta[i] = d;
    }
}

extern "C" __global__ void sk_adan_multi_10(
    const unsigned long long* __restrict__ grad_ptrs,
    const unsigned long long* __restrict__ param_ptrs,
    const unsigned long long* __restrict__ m_ptrs,
    const unsigned long long* __restrict__ v_ptrs,
    const unsigned long long* __restrict__ nsq_ptrs,
    const unsigned long long* __restrict__ gprev_ptrs,
    const unsigned long long* __restrict__ delta_ptrs,
    const unsigned long long* __restrict__ last_delta_ptrs,
    const int* __restrict__ starts,
    const int* __restrict__ ns,
    const float* __restrict__ lrs,
    const float* __restrict__ wds,
    const int* __restrict__ apply_rbs,
    int num_tensors,
    int total_n,
    float b1,
    float b2,
    float b3,
    float eps,
    float bc1,
    float bc2,
    float bc3,
    int   first_step,
    float cautious
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int tid = tid0; tid < total_n; tid += stride) {
        int lo = 0;
        int hi = num_tensors - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (starts[mid] <= tid) lo = mid;
            else hi = mid - 1;
        }
        const int t = lo;
        const int i = tid - starts[t];
        if ((unsigned)i >= (unsigned)ns[t]) continue;

        adan_body_10(
            i,
            (const float*)(uintptr_t)grad_ptrs[t],
            (const float*)(uintptr_t)param_ptrs[t],
            (float*)(uintptr_t)m_ptrs[t],
            (float*)(uintptr_t)v_ptrs[t],
            (float*)(uintptr_t)nsq_ptrs[t],
            (float*)(uintptr_t)gprev_ptrs[t],
            (float*)(uintptr_t)delta_ptrs[t],
            (float*)(uintptr_t)last_delta_ptrs[t],
            lrs[t], b1, b2, b3, eps, wds[t],
            bc1, bc2, bc3, first_step, cautious, apply_rbs[t]
        );
    }
}

extern "C" __global__ void sk_adan_precond_10(
    const float* __restrict__ grad,
    const float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ nsq,
    float* __restrict__ gprev,
    float* __restrict__ delta,
    float* __restrict__ last_delta,
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
    int   first_step,
    float cautious,
    int   cols,
    int   apply_rb,
    int   n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int n4 = n >> 2;

    for (int vi = tid; vi < n4; vi += stride) {
        const int i0 = vi << 2;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            adan_precond_body_10(i0 + k, grad, param, m, v, nsq, gprev, delta, last_delta,
                                row_rms, col_rms,
                                lr, b1, b2, b3, eps, wd, bc1, bc2, bc3,
                                first_step, cautious, cols, apply_rb);
        }
    }
    for (int i = (n4 << 2) + tid; i < n; i += stride) {
        adan_precond_body_10(i, grad, param, m, v, nsq, gprev, delta, last_delta,
                             row_rms, col_rms,
                             lr, b1, b2, b3, eps, wd, bc1, bc2, bc3,
                             first_step, cautious, cols, apply_rb);
    }
}

extern "C" __global__ void sk_kink_rb_10(
    float* __restrict__ delta,
    float* __restrict__ last_delta,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float spread,
    int first_step,
    int n
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid0; i < n; i += stride) {
        float t = spread * (-1.0f + 2.0f * ((float)i + 0.5f) / (float)n);
        float d = -w[i] * t - b[i];
        float prior = last_delta[i];
        float applied = (first_step == 0 && d * prior < 0.0f) ? -prior : d;
        delta[i] = applied;
        last_delta[i] = applied;
    }
}

extern "C" __global__ void sk_kink2_rb_10(
    float* __restrict__ delta,
    float* __restrict__ last_delta,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float f_lin,
    float span,
    int first_step,
    int n
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int n_lin = (int)(f_lin * (float)n);
    if (n_lin > n) n_lin = n;
    for (int i = tid0; i < n; i += stride) {
        float t;
        if (i < n_lin) {
            t = (w[i] >= 0.0f) ? -4.0f : 4.0f;
        } else {
            int m = n - n_lin;
            int j = i - n_lin;
            t = (m > 0) ? span * (-1.0f + 2.0f * ((float)j + 0.5f) / (float)m) : 0.0f;
        }
        float d = -w[i] * t - b[i];
        float prior = last_delta[i];
        float applied = (first_step == 0 && d * prior < 0.0f) ? -prior : d;
        delta[i] = applied;
        last_delta[i] = applied;
    }
}

extern "C" __global__ void sk_offset_rb_10(
    float* __restrict__ delta,
    float* __restrict__ last_delta,
    float amount,
    int first_step,
    int n
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid0; i < n; i += stride) {
        float d = delta[i] + amount;
        float prior = last_delta[i];
        float applied = (first_step == 0 && d * prior < 0.0f) ? -prior : d;
        delta[i] = applied;
        last_delta[i] = applied;
    }
}

extern "C" __global__ void sk_scale_10(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    const int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid0; i < n; i += stride) {
        delta[i] = factor * src[i];
    }
}

extern "C" __global__ void sk_axis_rms_both_10(
    const float* __restrict__ grad,
    float* __restrict__ row_rms,
    float* __restrict__ col_rms,
    int rows,
    int cols
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int axis = (bid < rows) ? 0 : 1;
    int group = (axis == 0) ? bid : (bid - rows);
    int count = (axis == 0) ? cols : rows;
    float sum = 0.0f;

    for (int j = tid; j < count; j += blockDim.x) {
        int idx = (axis == 0) ? group * cols + j : j * cols + group;
        float g = __ldg(&grad[idx]);
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
            float r = sqrtf(sum / (float)count);
            if (axis == 0) row_rms[group] = r;
            else           col_rms[group] = r;
        }
    }
}