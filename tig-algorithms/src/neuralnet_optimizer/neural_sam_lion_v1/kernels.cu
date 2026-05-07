// neural_sam_lion_v1 — 3 kernels:
//   1. perturb_sign_sam — produce θ + ε_sam · sign(g_last)  for SAM probe
//   2. capture_grad     — copy current grad into g_last buffer
//   3. lion_step        — same Lion sign-momentum step as baseline + per-track variants

extern "C" __global__ void perturb_sign_sam(
    float* __restrict__ out,
    const float* __restrict__ theta,
    const float* __restrict__ g_last,
    const float eps_sam,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gi = g_last[i];
    float s;
    if (isnan(gi) || isinf(gi))   s = 0.0f;
    else if (gi > 0.0f)            s = 1.0f;
    else if (gi < 0.0f)            s = -1.0f;
    else                            s = 0.0f;
    out[i] = theta[i] + eps_sam * s;
}

extern "C" __global__ void capture_grad(
    float* __restrict__ dst,
    const float* __restrict__ grad,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = grad[i];
}

extern "C" __global__ void lion_step(
    float* __restrict__ update,
    float* __restrict__ m,
    const float* __restrict__ theta,
    const float* __restrict__ grad,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float gi = grad[i];
    const float mi = m[i];
    const float ti = theta[i];

    const float blend = beta1 * mi + (1.0f - beta1) * gi;

    float s;
    if (isnan(blend) || isinf(blend))     s = 0.0f;
    else if (blend > 0.0f)                s = 1.0f;
    else if (blend < 0.0f)                s = -1.0f;
    else                                   s = 0.0f;

    update[i] = lr * (s + weight_decay * ti);
    m[i]      = beta2 * mi + (1.0f - beta2) * gi;
}
