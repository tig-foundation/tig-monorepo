#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void sk_adan_4(
    const float* grad,
    const float* param,
    float* m,
    float* v,
    float* nsq,
    float* gprev,
    float* delta,
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
    float gd = first_step ? 0.0f : g - gprev[i];
    gprev[i] = g;

    float mi = (1 - b1) * g + b1 * m[i];
    float vi = (1 - b2) * gd + b2 * v[i];
    float comb = g + b2 * gd;
    float ni = (1 - b3) * comb * comb + b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;

    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1 && upd * g > 0) upd *= cautious;
    delta[i] = (p + upd) / (1 + lr * wd) - p;
}

extern "C" __global__ void sk_rollback_4(
    float* delta,
    float* previous,
    int first_step,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float current = delta[i];
    float prior = previous[i];
    float applied = (!first_step && current * prior < 0) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_nonmatrix_adan_rollback_batch_4(
    const unsigned long long* descriptors,
    float b1,
    float b2,
    float b3,
    float eps,
    float bc1,
    float bc2,
    float bc3,
    int first_step,
    float cautious
) {
    const unsigned long long* d = descriptors + (size_t)blockIdx.x * 12;
    int local = threadIdx.x;
    int count = (int)d[11];
    if (local >= count) return;

    int i = (int)d[10] + local;
    const float* grad = reinterpret_cast<const float*>(d[0]);
    const float* param = reinterpret_cast<const float*>(d[1]);
    float* m = reinterpret_cast<float*>(d[2]);
    float* v = reinterpret_cast<float*>(d[3]);
    float* nsq = reinterpret_cast<float*>(d[4]);
    float* gprev = reinterpret_cast<float*>(d[5]);
    float* delta = reinterpret_cast<float*>(d[6]);
    float* previous = reinterpret_cast<float*>(d[7]);
    float lr = __uint_as_float((unsigned int)d[8]);
    float wd = __uint_as_float((unsigned int)d[9]);

    float g = grad[i];
    float p = param[i];
    float gd = first_step ? 0.0f : g - gprev[i];
    gprev[i] = g;

    float mi = (1 - b1) * g + b1 * m[i];
    float vi = (1 - b2) * gd + b2 * v[i];
    float comb = g + b2 * gd;
    float ni = (1 - b3) * comb * comb + b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;

    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1 && upd * g > 0) upd *= cautious;

    float current = (p + upd) / (1 + lr * wd) - p;
    float prior = previous[i];
    float applied = (!first_step && current * prior < 0) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}

extern "C" __global__ void sk_kink_4(
    float* delta,
    const float* w,
    const float* b,
    float spread,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = spread * (-1 + 2 * ((float)i + 0.5f) / (float)n);
    delta[i] = -w[i] * t - b[i];
}

extern "C" __global__ void sk_kink2_4(
    float* delta,
    const float* w,
    const float* b,
    float f_lin,
    float span,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int nl = (int)(f_lin * n);
    if (nl > n) nl = n;

    float t;
    if (i < nl) {
        t = w[i] >= 0 ? -4 : 4;
    } else {
        int m = n - nl;
        int j = i - nl;
        t = m > 0 ? span * (-1 + 2 * ((float)j + 0.5f) / m) : 0;
    }
    delta[i] = -w[i] * t - b[i];
}

extern "C" __global__ void sk_offset_4(float* delta, float amount, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) delta[i] += amount;
}

extern "C" __global__ void sk_scale_4(float* delta, const float* src, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) delta[i] = factor * src[i];
}

extern "C" __global__ void sk_axis_rms_4(
    const float* grad,
    float* rms,
    int rows,
    int cols,
    int axis
) {
    int group = blockIdx.x;
    int tid = threadIdx.x;
    int count = axis == 0 ? cols : rows;
    float sum = 0;

    for (int j = tid; j < count; j += blockDim.x) {
        int idx = axis == 0 ? group * cols + j : j * cols + group;
        float g = grad[idx];
        sum += g * g;
    }

    unsigned mask = 0xffffffff;
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_down_sync(mask, sum, o);
    }

    __shared__ float ws[32];
    if ((tid & 31) == 0) ws[tid >> 5] = sum;
    __syncthreads();

    if (tid < 32) {
        int wc = (blockDim.x + 31) >> 5;
        sum = tid < wc ? ws[tid] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_down_sync(mask, sum, o);
        }
        if (tid == 0) rms[group] = sqrtf(sum / count);
    }
}

extern "C" __global__ void sk_matrix_adan_precond_rollback_4(
    const float* grad,
    const float* param,
    float* m,
    float* v,
    float* nsq,
    float* gprev,
    float* delta,
    const float* row_rms,
    const float* col_rms,
    float* previous,
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
    float gd = first_step ? 0 : g - gprev[i];
    gprev[i] = g;

    float mi = (1 - b1) * g + b1 * m[i];
    float vi = (1 - b2) * gd + b2 * v[i];
    float comb = g + b2 * gd;
    float ni = (1 - b3) * comb * comb + b3 * nsq[i];
    m[i] = mi;
    v[i] = vi;
    nsq[i] = ni;

    float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
    float upd = -lr * step;
    if (cautious < 1 && upd * g > 0) upd *= cautious;

    float raw = (p + upd) / (1 + lr * wd) - p;
    int row = i / cols;
    int col = i - row * cols;
    float current = raw / sqrtf(row_rms[row] * col_rms[col] + eps);
    float prior = previous[i];
    float applied = (!first_step && current * prior < 0) ? -prior : current;
    delta[i] = applied;
    previous[i] = applied;
}
