// neural_lion_per_track_v1 — same lion_step kernel as neural_baseline_lion_v1.
// Per-track logic lives entirely in mod.rs hyperparameter selection.

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
