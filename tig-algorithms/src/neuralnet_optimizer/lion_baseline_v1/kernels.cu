// neural_baseline_lion_v1 — CUDA kernel for the Lion sign-momentum update.
//
// One kernel, ~30 lines, element-wise. No shared memory, no atomics.
// Bounded step magnitude (sign() ∈ {-1, 0, +1}) makes NaN-propagation
// near-impossible: we explicitly clamp the sign of NaN to 0.
//
// Convention (verified against tig-challenges training_loop):
//   The harness applies the returned `update` as theta_new = theta - update.
//   So we return the POSITIVE delta to subtract: lr * (sign_blend + wd*theta).
//
// Math:
//   blend  = beta1 * m + (1 - beta1) * grad
//   s      = sign(blend)            // clamp NaN/Inf to 0
//   update = lr * (s + weight_decay * theta)
//   m      = beta2 * m + (1 - beta2) * grad

extern "C" __global__ void lion_step(
    float* __restrict__ update,        // OUT: per-element update (to be subtracted from theta)
    float* __restrict__ m,             // IN/OUT: per-element momentum buffer
    const float* __restrict__ theta,   // IN: current parameter values
    const float* __restrict__ grad,    // IN: current gradient values
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

    // Sign-momentum blend.
    const float blend = beta1 * mi + (1.0f - beta1) * gi;

    // Clamped sign — the only place a NaN could leak in (isnan(blend) → 0).
    float s;
    if (isnan(blend) || isinf(blend)) {
        s = 0.0f;
    } else if (blend > 0.0f) {
        s = 1.0f;
    } else if (blend < 0.0f) {
        s = -1.0f;
    } else {
        s = 0.0f;
    }

    // Update to subtract from theta. Bounded magnitude: |update[i]| ≤ lr * (1 + |wd| * |theta|).
    update[i] = lr * (s + weight_decay * ti);

    // EMA-style momentum buffer for the next step.
    m[i] = beta2 * mi + (1.0f - beta2) * gi;
}
