// neural_sophia_h_v1 — CUDA kernels for Sophia with Hutchinson HVP.
//
// 4 kernels:
//   1. perturb_params_rademacher — generate θ ± ε·v with Rademacher v
//      (used by query_at_params on probe steps 0 and 1)
//   2. capture_grad — copy grad into a saved buffer (grad_plus or grad_minus)
//   3. hvp_update_h — compute h_hessian = (grad_plus − grad_minus)/(2ε) ⊙ v
//      and EMA-blend into h
//   4. sophia_h_step — the regular Sophia update applied every step
//
// Rademacher vector v is generated deterministically from (cycle_seed, idx)
// via a fast PRNG mix so probe step 0 and probe step 1 agree on v.

// xorshift mix used to deterministically derive a Rademacher value from
// (cycle_seed, idx). Returns +1.0 or -1.0.
__device__ __forceinline__ float rademacher(unsigned int cycle_seed, int idx) {
    unsigned int x = cycle_seed * 2654435761u ^ ((unsigned int)idx * 40503u);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return (x & 1u) ? 1.0f : -1.0f;
}

// ----- Kernel 1: produce θ ± ε·v (called by query_at_params on probe steps).
extern "C" __global__ void perturb_params_rademacher(
    float* __restrict__ out_perturbed,
    const float* __restrict__ theta,
    const float eps_hvp,
    const float sign,            // +1.0 for probe phase 0, -1.0 for phase 1
    const unsigned int cycle_seed,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = rademacher(cycle_seed, i);
    out_perturbed[i] = theta[i] + sign * eps_hvp * v;
}

// ----- Kernel 2: capture grad into a saved buffer.
extern "C" __global__ void capture_grad(
    float* __restrict__ dst,
    const float* __restrict__ grad,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = grad[i];
}

// ----- Kernel 3: compute h_hessian = (gp - gm) / (2ε) * v, EMA into h.
extern "C" __global__ void hvp_update_h(
    float* __restrict__ h,
    const float* __restrict__ grad_plus,
    const float* __restrict__ grad_minus,
    const float eps_hvp,
    const float beta_h,
    const unsigned int cycle_seed,
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = rademacher(cycle_seed, i);
    const float h_est_raw = (grad_plus[i] - grad_minus[i]) / (2.0f * eps_hvp) * v;
    // Clamp the raw HVP estimate to a reasonable magnitude — Hutchinson is
    // unbiased but high-variance; clipping prevents single-batch noise from
    // poisoning the EMA. Bound: 100 (per-element diagonal Hessian rarely
    // exceeds this on the c006 MLP).
    float h_est = fminf(100.0f, fmaxf(-100.0f, h_est_raw));
    if (isnan(h_est) || isinf(h_est)) h_est = 0.0f;
    // We track |h_diag| since Sophia uses h as a positive denominator.
    h_est = fabsf(h_est);
    h[i] = beta_h * h[i] + (1.0f - beta_h) * h_est;
}

// ----- Kernel 4: Sophia-H step (same shape as Sophia-G, h is HVP-derived
// when fresh and falls back to last cycle's h between probe pairs).
extern "C" __global__ void sophia_h_step(
    float* __restrict__ update,
    float* __restrict__ m,
    float* __restrict__ h,
    const float* __restrict__ theta,
    const float* __restrict__ grad,
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

    const float gi = grad[i];
    const float ti = theta[i];
    const float mi = m[i];
    const float hi = h[i];

    // Update first moment.
    const float m_new = beta1 * mi + (1.0f - beta1) * gi;
    m[i] = m_new;

    // We do NOT touch h here — h is updated by hvp_update_h on probe step 2
    // of each cycle. (Optionally one could also blend g² into h here as a
    // safety net for early steps before any HVP has fired; we use an
    // EMA-from-zero start instead, since Sophia paper warns against mixing.)

    // Sophia-style preconditioned + clipped update.
    const float denom = fmaxf(hi, epsilon);
    const float precond = m_new / denom;
    float clipped = fminf(rho, fmaxf(-rho, precond));
    if (isnan(clipped) || isinf(clipped)) clipped = 0.0f;

    update[i] = lr * (clipped + weight_decay * ti);
}
