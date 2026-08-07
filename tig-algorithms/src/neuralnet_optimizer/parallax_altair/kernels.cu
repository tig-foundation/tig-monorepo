// parallax_altair, CUDA kernels

// Adan (Xie et al. 2022), decoupled weight decay, cautious damping.
extern "C" __global__ void sk_adan(
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
    int   first_step,
    float cautious,
    int   n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float p = param[i];
    // No previous gradient on the first step.
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

// Spread the first layer's ReLU breakpoints. The challenge has a scalar input
// and zero-initialised biases, so all 256 breakpoints sit at x = 0 and the
// layer spans only two dimensions. Unit i breaks at x = -b_i/w_i, so
// b_i = -w_i*t_i places it at t_i.
extern "C" __global__ void sk_kink(
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

// Two populations. A breakpoint outside the input range makes the unit linear
// or dead depending on sign(w); f_lin of them are sent outside on the side that
// keeps them active, the rest are spaced evenly.
extern "C" __global__ void sk_kink2(
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

// Constant offset added to an update.
extern "C" __global__ void sk_offset(
    float* __restrict__ delta,
    float amount,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] += amount;
}

// delta = factor * src, used to force a parameter to a chosen value.
extern "C" __global__ void sk_scale(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] = factor * src[i];
}
