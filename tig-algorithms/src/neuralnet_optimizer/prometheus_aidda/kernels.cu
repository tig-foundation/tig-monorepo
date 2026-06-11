#include <math.h>

// Cautious Adan update with decoupled (AdamW-style) weight decay.
// Updates m, v, s (reused as Adan's n) and prev_grad in place. Writes the
// parameter delta (to be added) into `delta`.
//
//   g_diff = first_step ? 0 : (g - g_prev)
//   m_t = 0.98 * m_{t-1} + 0.02 * g                      (EMA of g)
//   v_t = 0.92 * v_{t-1} + 0.08 * g_diff                 (EMA of g_diff)
//   n_t = 0.99 * n_{t-1} + 0.01 * (g + 0.92 * g_diff)^2  (second moment)
//   adan_dir = (m_t + 0.92 * v_t) / (sqrt(n_t) + eps)
//   wd_gate  = 1.0 if adan_dir * theta >= 0 else 0.0   (cautious weight decay)
//   dir      = adan_dir + wd_gate * wd * theta
//   keep     = 1.0 if dir * g > 0 else 0.25   (cautious sign mask)
//   delta    = -lr * keep * dir
extern "C" __global__ void adabelief_update_kernel(
    const float* __restrict__ grad,
    const float* __restrict__ params,
    float* m,
    float* v,
    float* s,
    float* prev_grad,
    float lr,
    float eps,
    float wd,
    int first_step,
    float* delta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g     = grad[idx];
        float theta = params[idx];
        float g_diff = (first_step != 0) ? 0.0f : (g - prev_grad[idx]);

        float m_new = 0.98f * m[idx] + 0.02f * g;
        float v_new = 0.92f * v[idx] + 0.08f * g_diff;
        float g_n   = g + 0.92f * g_diff;
        float n_new = 0.99f * s[idx] + 0.01f * g_n * g_n;

        m[idx] = m_new;
        v[idx] = v_new;
        s[idx] = n_new;
        prev_grad[idx] = g;

        // Cautious weight decay (CWD): apply decay only on coordinates where
        // the pre-damping Adan direction and the parameter share a sign, i.e.
        // where the step already moves the coordinate toward zero. Groups with
        // wd == 0 (biases, BN affine, running stats) are unaffected.
        float adan_dir = (m_new + 0.92f * v_new) / (sqrtf(n_new) + eps);
        float wd_gate  = (adan_dir * theta >= 0.0f) ? 1.0f : 0.0f;
        // Cautious mask: damp coordinates whose update direction disagrees
        // with the raw gradient; damped (not zeroed), no mean-rescale.
        float dir  = adan_dir + wd_gate * wd * theta;
        float keep = (dir * g > 0.0f) ? 1.0f : 0.25f;
        delta[idx] = -lr * keep * dir;
    }
}

// Nesterov-style lookahead: position the gradient query one (scaled) momentum
// step ahead of the current params, using last step's belief-adapted direction.
// Unused while query_at_params is disabled; kept so the module's kernel set
// stays stable.
//
//   m_hat = m / bc1,  s_hat = s / bc2
//   step  = lr * m_hat / (sqrt(s_hat) + eps)
//   out   = params - beta * step
extern "C" __global__ void nesterov_lookahead_kernel(
    const float* __restrict__ params,
    const float* __restrict__ m,
    const float* __restrict__ s,
    float lr,
    float beta,
    float bc1,
    float bc2,
    float eps,
    float* out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m_hat = m[idx] / bc1;
        float s_hat = s[idx] / bc2;
        float step  = lr * m_hat / (sqrtf(s_hat) + eps);
        out[idx] = params[idx] - beta * step;
    }
}
