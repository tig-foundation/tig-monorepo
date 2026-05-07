// neural_sophia_g_v1 — Sophia-G (GNB diagonal preconditioner + per-element clip).
//
// Sophia paper: Liu et al. 2023, https://arxiv.org/abs/2305.14342
// Gauss-Newton-Bartlett variant: H estimated as g², no HVP needed.
//
// Math (per element):
//   m       = β1 · m_old + (1 − β1) · g
//   h       = β2 · h_old + (1 − β2) · g²
//   precond = m / max(h, ε)
//   clipped = clamp(precond, −ρ, +ρ)
//   update  = lr · (clipped + λ · theta)
//
// Convention: harness applies update as theta_new = theta - update.
// Numerical safety: clamp NaN/Inf in clipped to 0; max(h, ε) prevents
// division blow-up; clip(·, ρ) bounds step magnitude regardless of m and h.

extern "C" __global__ void sophia_g_step(
    float* __restrict__ update,        // OUT: per-element update (to be subtracted)
    float* __restrict__ m,             // IN/OUT: first-moment momentum
    float* __restrict__ h,             // IN/OUT: GNB diagonal curvature estimate
    const float* __restrict__ theta,   // IN: current parameter values
    const float* __restrict__ grad,    // IN: current gradient values
    const float lr,
    const float beta1,
    const float beta2,
    const float rho,
    const float epsilon,
    const float weight_decay,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float gi    = grad[i];
    const float mi    = m[i];
    const float hi    = h[i];
    const float ti    = theta[i];

    // EMA updates for momentum and GNB diagonal.
    const float m_new = beta1 * mi + (1.0f - beta1) * gi;
    const float h_new = beta2 * hi + (1.0f - beta2) * gi * gi;
    m[i] = m_new;
    h[i] = h_new;

    // Preconditioned step: m / max(h, ε).
    const float denom   = fmaxf(h_new, epsilon);
    const float precond = m_new / denom;

    // Clipped update — bounds per-element step magnitude to ρ.
    float clipped = fminf(rho, fmaxf(-rho, precond));

    // Final NaN/Inf safety net (paranoia layer; clip already handles most cases).
    if (isnan(clipped) || isinf(clipped)) {
        clipped = 0.0f;
    }

    update[i] = lr * (clipped + weight_decay * ti);
}
