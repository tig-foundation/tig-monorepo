// parallax_vega, CUDA kernels

// First-layer initialisation stage.
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

// delta = factor * src.
extern "C" __global__ void sk_scale(
    float* __restrict__ delta,
    const float* __restrict__ src,
    float factor,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    delta[i] = factor * src[i];
}

// Per-element offset added to an update.
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

// One launch for every trainable tensor instead of one per tensor. Each thread
// finds its tensor by binary search over the cumulative element counts.
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
    int   gsign,
    int   total
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

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
    if (gsign != 0 && g * gp < 0.0f) gd *= 0.25f;
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

