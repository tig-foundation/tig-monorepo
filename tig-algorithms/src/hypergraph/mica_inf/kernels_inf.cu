// Device kernels of the inferred starting partition: hyperedge means, node steps, recentring,
// threshold counting and the split of a set into two.

extern "C" __global__ void inf_edge_sums(
    const int num_hyperedges,
    const int *hyperedge_offsets,
    const int *hyperedge_nodes,
    const int *set_of,
    const float *y,
    const float *mu,
    int *cnt,
    float *z
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_hyperedges) return;
    int c0 = 0, c1 = 0;
    float z0 = 0.0f, z1 = 0.0f;
    for (int k = hyperedge_offsets[e]; k < hyperedge_offsets[e + 1]; k++) {
        int v = hyperedge_nodes[k];
        int s = set_of[v];
        if (s == 0) { c0++; z0 += y[v] - mu[0]; }
        else if (s == 1) { c1++; z1 += y[v] - mu[1]; }
    }
    cnt[2 * e] = c0;
    cnt[2 * e + 1] = c1;
    z[2 * e] = z0;
    z[2 * e + 1] = z1;
}

extern "C" __global__ void inf_node_deg(
    const int num_nodes,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *set_of,
    const int *cnt,
    const float tau,
    float *wdeg
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    int s = set_of[v];
    if (s < 0 || s > 1) { wdeg[v] = 0.0f; return; }
    int d = 0;
    for (int k = node_offsets[v]; k < node_offsets[v + 1]; k++) {
        if (cnt[2 * node_hyperedges[k] + s] >= 2) d++;
    }
    wdeg[v] = (float)d + tau;
}

extern "C" __global__ void inf_node_step(
    const int num_nodes,
    const int *node_offsets,
    const int *node_hyperedges,
    const int *set_of,
    const int *cnt,
    const float *z,
    const float *mu,
    const float *wdeg,
    const float *y_in,
    float *y_out
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    int s = set_of[v];
    if (s < 0 || s > 1) { y_out[v] = 0.0f; return; }
    float xv = y_in[v] - mu[s];
    float acc = 0.0f;
    for (int k = node_offsets[v]; k < node_offsets[v + 1]; k++) {
        int e = node_hyperedges[k];
        int c = cnt[2 * e + s];
        if (c >= 2) acc += (z[2 * e + s] - xv) / (float)(c - 1);
    }
    float wd = wdeg[v];
    y_out[v] = (wd > 0.0f) ? acc / wd : 0.0f;
}

extern "C" __global__ void inf_reduce2(
    const int num_nodes,
    const int *set_of,
    const float *a,
    const float *b,
    const int use_b,
    float *out
) {
    __shared__ float s0[1024];
    __shared__ float s1[1024];
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int v = threadIdx.x; v < num_nodes; v += blockDim.x) {
        int s = set_of[v];
        if (s < 0 || s > 1) continue;
        float val = use_b ? a[v] * b[v] : a[v];
        if (s == 0) acc0 += val; else acc1 += val;
    }
    s0[threadIdx.x] = acc0;
    s1[threadIdx.x] = acc1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s0[threadIdx.x] += s0[threadIdx.x + stride];
            s1[threadIdx.x] += s1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[0] = s0[0]; out[1] = s1[0]; }
}

extern "C" __global__ void inf_mu(const float *sum, const float *w, float *mu) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        mu[0] = (w[0] > 0.0f) ? sum[0] / w[0] : 0.0f;
        mu[1] = (w[1] > 0.0f) ? sum[1] / w[1] : 0.0f;
    }
}

extern "C" __global__ void inf_count_below(
    const int num_nodes,
    const int *set_of,
    const float *y,
    const float *mu,
    const int s,
    const float t,
    const float t2,
    const int vt,
    int *count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (set_of[v] != s) return;
    float x = y[v] - mu[s];
    if (x < t || (x < t2 && v < vt)) atomicAdd(count, 1);
}

extern "C" __global__ void inf_split(
    const int num_nodes,
    const int *set_of,
    const float *y,
    const float *mu,
    const int s,
    const float t,
    const float t2,
    const int vt,
    int *new_set_of
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (set_of[v] != s) return;
    float x = y[v] - mu[s];
    new_set_of[v] = (x < t || (x < t2 && v < vt)) ? 2 * s : 2 * s + 1;
}
