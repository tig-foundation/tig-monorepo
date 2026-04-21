
extern "C" __global__ __launch_bounds__(256) void nitt_restore_kernel_30(
    const float* __restrict__ params, const float* __restrict__ best_w, float* __restrict__ updates,
    float* __restrict__ m, float* __restrict__ v, float* __restrict__ slow_u, const unsigned int n
) {
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        updates[idx] = best_w[idx] - params[idx];
        m[idx] *= 0.25f;
        v[idx] *= 0.25f;
        slow_u[idx] = 0.0f;
    }
}
