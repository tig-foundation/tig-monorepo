// T29 i10: pure Lion (no EF) + momentum-SAM (rho=0.004).
// Lion: update = sign(beta1*m + (1-beta1)*g), m updated with same beta.
// No EF buffer → no residual corruption when SAM provides perturbed grads.
// SAM perturbs params by rho * sign(m) along ascent direction.

extern "C" __global__ __launch_bounds__(256, 4)
void lion_kernel(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta1,
    const float weight_decay
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = gradients[idx];
    float m = momentum[idx];

    float interp = beta1 * m + (1.0f - beta1) * g;
    float direction = (interp > 0.0f) ? 1.0f : ((interp < 0.0f) ? -1.0f : 0.0f);

    updates[idx] = -lr * direction - lr * weight_decay * params[idx];

    momentum[idx] = beta1 * m + (1.0f - beta1) * g;
}

extern "C" __global__ __launch_bounds__(256, 4)
void momentum_sam_kernel(
    const float* __restrict__ params,
    const float* __restrict__ momentum,
    float* __restrict__ perturbed,
    const unsigned int n,
    const float rho
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float m = momentum[idx];
    float sign_m = (m > 0.0f) ? 1.0f : ((m < 0.0f) ? -1.0f : 0.0f);
    perturbed[idx] = params[idx] + rho * sign_m;
}
