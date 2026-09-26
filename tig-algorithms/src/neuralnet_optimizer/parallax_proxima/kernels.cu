extern "C" __global__ void orth_same(const float* const* weights, unsigned int* flags) {
    __shared__ unsigned int same[256];
    int t = threadIdx.x;
    int m = blockIdx.z;
    int i = blockIdx.x * 256 + t;
    same[t] = __float_as_uint(weights[m][i]) == __float_as_uint(weights[0][i]);
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) same[t] &= same[t + stride];
        __syncthreads();
    }
    if (t == 0) flags[m * 256 + blockIdx.x] = same[0];
}

extern "C" __global__ void orth_same_reduce(const unsigned int* flags, unsigned int* result) {
    __shared__ unsigned int same[256];
    int t = threadIdx.x;
    int m = blockIdx.z;
    same[t] = flags[m * 256 + t];
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) same[t] &= same[t + stride];
        __syncthreads();
    }
    if (t == 0) result[m] = same[0];
}

extern "C" __global__ void orth_pack(const float* const* weights, float* x) {
    int m = blockIdx.z;
    int i = blockIdx.x * 256 + threadIdx.x;
    x[m * 65536 + i] = weights[m][i];
}

extern "C" __global__ void orth_norm(const float* x, float* norms) {
    __shared__ float sums[256];
    int t = threadIdx.x;
    int base = blockIdx.z * 65536;
    float sum = 0.0f;
    for (int i = t; i < 65536; i += 256) {
        float v = x[base + i];
        sum += v * v;
    }
    sums[t] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    if (t == 0) norms[blockIdx.z] = sqrtf(sums[0]);
}

extern "C" __global__ void orth_normalize(float* x, const float* norms) {
    int m = blockIdx.z;
    int i = m * 65536 + blockIdx.x * 256 + threadIdx.x;
    float n = norms[m];
    x[i] = n > 0.0f ? x[i] / n : 0.0f;
}

extern "C" __global__ void orth_gemm(
    const float* a, const float* b, float* out, int transpose_b
) {
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    int base = blockIdx.z * 65536;
    float sum = 0.0f;
    for (int tile = 0; tile < 16; ++tile) {
        int q = tile * 16;
        tile_a[ty][tx] = a[base + row * 256 + q + tx];
        tile_b[ty][tx] = transpose_b
            ? b[base + col * 256 + q + ty]
            : b[base + (q + ty) * 256 + col];
        __syncthreads();
        for (int k = 0; k < 16; ++k) sum += tile_a[ty][k] * tile_b[k][tx];
        __syncthreads();
    }
    out[base + row * 256 + col] = sum;
}

extern "C" __global__ void orth_combine(const float* x, float* y, float a, float b) {
    int i = blockIdx.z * 65536 + blockIdx.x * 256 + threadIdx.x;
    y[i] = a * x[i] + b * y[i];
}

extern "C" __global__ void orth_finish(
    const float* const* weights, float* const* updates, const float* x,
    const float* norm_w, const float* norm_x, float scale, int shared
) {
    int m = blockIdx.z;
    int i = blockIdx.x * 256 + threadIdx.x;
    int src = shared ? 0 : m;
    float ratio = norm_x[src] > 0.0f ? scale * norm_w[src] / norm_x[src] : 0.0f;
    updates[m][i] = ratio * x[src * 65536 + i] - weights[m][i];
}

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

extern "C" __global__ void sk_offset_vec(
    float* __restrict__ delta,
    const float* __restrict__ v,
    float g,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] += g * v[i];
}

extern "C" __global__ void sk_adan_fused(
    const float* const* __restrict__ grad,
    const float* const* __restrict__ param,
    float* const* __restrict__ m,
    float* const* __restrict__ v,
    float* const* __restrict__ nsq,
    float* const* __restrict__ gprev,
    float* const* __restrict__ delta,
    const int*   __restrict__ prefix,
    const float* __restrict__ lrs,
    const float* __restrict__ wds,
    int   n_tensors,
    float b1, float b2, float b3, float eps,
    float bc1, float bc2, float bc3,
    int   first_step,
    float cautious,
    float ab,
    int   wd_gate,
    int   total
) {
    const int stride = blockDim.x * gridDim.x;
    for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total; gid += stride) {
        int lo = 0, hi = n_tensors - 1, t = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (prefix[mid] <= gid) { t = mid; lo = mid + 1; } else { hi = mid - 1; }
        }
        int i = gid - prefix[t];

        float lr = lrs[t];
        float wd = wds[t];
        float g = grad[t][i];
        float p = param[t][i];
        float gp = gprev[t][i];
        float gd = (first_step != 0) ? 0.0f : (g - gp);
        gprev[t][i] = g;

        float mi = (1.0f - b1) * g + b1 * m[t][i];
        float vi = (1.0f - b2) * gd + b2 * v[t][i];
        float comb = g + b2 * gd;
        float dev = comb - mi;
        float sq = ab * comb * comb + (1.0f - ab) * (dev * dev + 1e-16f);
        float ni = (1.0f - b3) * sq + b3 * nsq[t][i];
        m[t][i] = mi;
        v[t][i] = vi;
        nsq[t][i] = ni;

        float step = (mi / bc1 + b2 * vi / bc2) / (sqrtf(ni / bc3) + eps);
        float upd = -lr * step;
        if (cautious < 1.0f && upd * g > 0.0f) upd *= cautious;
        float np;
        if (wd_gate != 0) {
            float d = (upd * p <= 0.0f) ? (wd * p) : 0.0f;
            np = p + upd - lr * d;
        } else {
            np = (p + upd) / (1.0f + lr * wd);
        }
        delta[t][i] = np - p;
    }
}

extern "C" __global__ void gp_adam(
    float* __restrict__ p,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ lrm,
    float lr, float b1, float b2, float bc1, float bc2, float eps,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    p[i] -= lr * lrm[i] * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}
