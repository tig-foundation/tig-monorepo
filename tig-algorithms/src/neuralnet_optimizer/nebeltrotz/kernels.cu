// nebeltrotz — clean-room optimizer step kernel for TIG c006 neuralnet_optimizer.
//
// Fused AdamW update with decoupled weight decay and optional per-element gradient
// clipping. The host computes the scheduled learning rate (warmup + cosine decay) and
// the per-tensor weight-decay coefficient and passes them in; this kernel only does the
// per-parameter arithmetic. Written from scratch against the public optimizer API of the
// challenge (params += update; update returned by optimizer_step). No third-party code.
//
// Determinism: every thread writes exactly one, non-overlapping element of m/v/update.

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void nebeltrotz_step(
    const float* __restrict__ grad,     // gradient for this tensor
    const float* __restrict__ param,    // current parameter values (for decoupled decay)
    const int    n,
    const float  lr,                    // scheduled learning rate for this step
    const float  beta1,
    const float  beta2,
    const float  eps,
    const float  weight_decay,          // decoupled; 0 for biases / batch-norm tensors
    const float  clip,                  // per-element grad clip magnitude; <=0 disables
    const int    step,                  // 1-indexed global step (for bias correction)
    float* __restrict__ m,              // 1st moment (persistent)
    float* __restrict__ v,              // 2nd moment (persistent)
    float* __restrict__ update          // output delta: param += update
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    if (clip > 0.0f) {
        g = fminf(fmaxf(g, -clip), clip);
    }

    // Exponential moving averages of gradient and its square.
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    // Bias correction (Kingma & Ba).
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;

    float adam_dir = m_hat / (sqrtf(v_hat) + eps);

    // AdamW: decoupled weight decay is applied to the parameter itself, not the gradient,
    // and scaled by the same learning rate. Fused into the single returned delta.
    update[idx] = -lr * (adam_dir + weight_decay * param[idx]);
}

// Nesterov-style preview point for optimizer_query_at_params: where the current momentum
// would carry the parameters with the NEXT step's learning rate (scaled by `nesterov`).
extern "C" __global__ void nebeltrotz_preview(
    const float* __restrict__ param,
    const float* __restrict__ m,
    const float* __restrict__ v,
    const int    n,
    const float  lr_scaled,
    const float  beta1,
    const float  beta2,
    const float  eps,
    const int    step,                  // number of updates already applied to m/v (>=1)
    float* __restrict__ out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    float m_hat = m[idx] / bc1;
    float v_hat = v[idx] / bc2;
    out[idx] = param[idx] - lr_scaled * (m_hat / (sqrtf(v_hat) + eps));
}

// Lookahead (Zhang, Lucas, Ba, Hinton 2019) slow-weight synchronisation, fused with the
// returned delta: fast = param + update; slow += alpha * (fast - slow); update = slow - param.
extern "C" __global__ void nebeltrotz_lookahead_sync(
    const float* __restrict__ param,
    const int    n,
    const float  alpha,
    float* __restrict__ slow,
    float* __restrict__ update
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float fast = param[idx] + update[idx];
    float s = slow[idx] + alpha * (fast - slow[idx]);
    slow[idx] = s;
    update[idx] = s - param[idx];
}

// ---------------------------------------------------------------------------
// Weight averaging ("Abendmittel").
//
// Why this exists: the challenge scores the model against the NOISY test targets, so the
// quality ceiling is 1 - 1/8 = 0.875 and everything above the noise floor is estimation
// error. On 1000 training points that error is dominated by variance, and an average over
// the tail of the trajectory removes variance without touching bias. The average is only
// made visible to the harness at validation time: it is swapped into the parameters on the
// LAST step of an epoch (validation follows immediately, and save_solution stores exactly
// those parameters) and swapped back out on the FIRST step of the next epoch, so the
// training trajectory itself never sees it. Same mechanism the DC probe already uses.
//
// Determinism: one thread per element, no cross-thread reads.
// ---------------------------------------------------------------------------

// ema = decay*ema + (1-decay)*(param+update);  init != 0 seeds it with param+update.
extern "C" __global__ void nebeltrotz_ema(
    const float* __restrict__ param,
    const float* __restrict__ update,
    const int    n,
    const float  decay,
    const int    init,
    float* __restrict__ ema
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float w = param[idx] + update[idx];
    ema[idx] = init ? w : (decay * ema[idx] + (1.0f - decay) * w);
}

// Swap the average in: stash the trajectory point, then steer the model onto the average.
extern "C" __global__ void nebeltrotz_ema_swap(
    const float* __restrict__ param,
    const float* __restrict__ ema,
    const int    n,
    float* __restrict__ backup,
    float* __restrict__ update
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    backup[idx] = param[idx] + update[idx];
    update[idx] = ema[idx] - param[idx];
}

// Swap the average back out: fold the return to the stashed trajectory point into the delta.
extern "C" __global__ void nebeltrotz_ema_restore(
    const float* __restrict__ param,
    const float* __restrict__ backup,
    const int    n,
    float* __restrict__ update
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    update[idx] += backup[idx] - param[idx];
}

// ---------------------------------------------------------------------------
// Orthogonalised momentum for the weight MATRICES ("Rautenschliff").
//
// Why this exists: the fuel budget of c006 is 5e12 and the AdamW step uses 1.9-4.5 % of it.
// Architecture, batch size and training loop are fixed, so the only way to spend that
// headroom is a heavier update rule. AdamW rescales every weight element on its own; a
// dense layer's weight is a matrix, and the informative structure sits in its singular
// values. Replacing the momentum matrix M by the nearest semi-orthogonal matrix (all
// singular values pulled to 1) spends compute to make each step use every direction of
// the layer equally, instead of following the few dominant ones.
//
// The orthogonalisation is a Newton-Schulz iteration of the odd quintic
// p(X) = a*X + b*X(X^T X) + c*X(X^T X)^2 on the Frobenius-normalised matrix. It needs only
// matrix products, so it runs without cuBLAS/cuSOLVER (neither is available in this build).
//
// Determinism: every matmul thread computes exactly one output element and accumulates the
// K dimension in a fixed tile order; the norm reduction uses a fixed block count, a fixed
// strided load order, a shared-memory tree and a serial final sum. No atomics anywhere.
// ---------------------------------------------------------------------------

#define MUON_TILE 16
#define MUON_NORM_BLOCKS 64
#define MUON_NORM_THREADS 256

// m = beta*m + g;  dir = nesterov ? g + beta*m : m.   Per-element clipping as in AdamW.
extern "C" __global__ void nebeltrotz_muon_mom(
    const float* __restrict__ grad,
    const int    n,
    const float  clip,                  // per-element grad clip magnitude; <=0 disables
    const float  beta,
    const float  nesterov,              // 0 = plain heavy ball, != 0 = Nesterov flavour
    float* __restrict__ m,              // 1st moment (persistent, separate from the AdamW moments)
    float* __restrict__ dir             // output: direction matrix to orthogonalise
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    if (clip > 0.0f) {
        g = fminf(fmaxf(g, -clip), clip);
    }
    float mn = beta * m[idx] + g;
    m[idx] = mn;
    dir[idx] = (nesterov != 0.0f) ? (g + beta * mn) : mn;
}

// Partial sums of squares; fixed grid of MUON_NORM_BLOCKS blocks so the split never
// depends on n and the result is bit-reproducible across runs.
extern "C" __global__ void nebeltrotz_sqsum(
    const float* __restrict__ x,
    const int    n,
    float* __restrict__ partial
) {
    __shared__ float s[MUON_NORM_THREADS];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = blockIdx.x * MUON_NORM_THREADS + tid;
         i < n;
         i += MUON_NORM_BLOCKS * MUON_NORM_THREADS) {
        acc += x[i] * x[i];
    }
    s[tid] = acc;
    __syncthreads();
    for (int off = MUON_NORM_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

// Serial final sum -> 1/(||x||_F + eps). One thread: fixed summation order.
extern "C" __global__ void nebeltrotz_norm_inv(
    const float* __restrict__ partial,
    const int    nb,
    const float  eps,
    float* __restrict__ inv
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float s = 0.0f;
    for (int i = 0; i < nb; i++) s += partial[i];
    inv[0] = 1.0f / (sqrtf(s) + eps);
}

// dst = src * inv[0]
extern "C" __global__ void nebeltrotz_scale_by(
    const float* __restrict__ src,
    const int    n,
    const float* __restrict__ inv,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx] * inv[0];
}

// out = cp*p + cq*q   (out may alias p; every thread touches one index only)
extern "C" __global__ void nebeltrotz_lincomb(
    float* __restrict__ out,
    const float* __restrict__ p,
    const float* __restrict__ q,
    const int    n,
    const float  cp,
    const float  cq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = cp * p[idx] + cq * q[idx];
}

// y = a*y + x   (in place; avoids aliasing the same buffer as both operands)
extern "C" __global__ void nebeltrotz_axpy(
    float* __restrict__ y,
    const float* __restrict__ x,
    const int    n,
    const float  a
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = a * y[idx] + x[idx];
}

// C[M x N] = op(A) * op(B), row-major.  ta/tb != 0 transposes that operand, so the Gram
// matrix can be formed on whichever side is smaller without ever materialising a transpose.
// A is M x K (ta == 0) or K x M (ta != 0);  B is K x N (tb == 0) or N x K (tb != 0).
// A and B are deliberately NOT __restrict__: the Gram product passes the same buffer twice.
extern "C" __global__ void nebeltrotz_mm(
    const float* A,
    const float* B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int ta,
    const int tb
) {
    __shared__ float As[MUON_TILE][MUON_TILE];
    __shared__ float Bs[MUON_TILE][MUON_TILE];

    int row = blockIdx.y * MUON_TILE + threadIdx.y;
    int col = blockIdx.x * MUON_TILE + threadIdx.x;
    float acc = 0.0f;

    int tiles = (K + MUON_TILE - 1) / MUON_TILE;
    for (int t = 0; t < tiles; t++) {
        int ak = t * MUON_TILE + threadIdx.x;
        int bk = t * MUON_TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? (ta ? A[(size_t)ak * M + row] : A[(size_t)row * K + ak]) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (col < N && bk < K) ? (tb ? B[(size_t)col * K + bk] : B[(size_t)bk * N + col]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < MUON_TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// update = -lr * (scale * o + weight_decay * param)   -- same convention as nebeltrotz_step.
extern "C" __global__ void nebeltrotz_muon_update(
    const float* __restrict__ o,
    const float* __restrict__ param,
    const int    n,
    const float  lr,
    const float  scale,                 // RMS match to AdamW: sqrt(max(rows, cols)) * muon_lr_mult
    const float  weight_decay,
    float* __restrict__ update
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    update[idx] = -lr * (scale * o[idx] + weight_decay * param[idx]);
}
