#include <curand_kernel.h>
#include <stdint.h>
#include <math.h>

// All reductions use exactly 256 threads and fixed reduction trees. Each
// block owns one point; there are no floating-point atomics or racing writes.
__device__ double gc_sum(double value, double *scratch) {
    const int t = threadIdx.x;
    scratch[t] = value;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride) scratch[t] += scratch[t + stride];
        __syncthreads();
    }
    return scratch[0];
}

// CPQR uses ||x_j||^2. Conditional selection uses the exact greedy gain
// ||x_j^T Y||^2 / ||x_j||^2 for the current residual dictionary and target.
extern "C" __global__ void gc_scores(
    const float *x, const float *cross, const int *used,
    double *scores, int dim, int points, int targets
) {
    const int p = blockIdx.x;
    if (p >= points) return;
    __shared__ double scratch[256];
    double norm = 0.0;
    for (int d = threadIdx.x; d < dim; d += 256) {
        double v = x[d + p * dim];
        norm += v * v;
    }
    norm = gc_sum(norm, scratch);
    double gain = norm;
    if (targets) {
        double energy = 0.0;
        for (int d = threadIdx.x; d < targets; d += 256) {
            double v = cross[d + p * targets];
            energy += v * v;
        }
        __syncthreads();
        energy = gc_sum(energy, scratch);
        gain = norm > 1e-30 ? energy / norm : 0.0;
    }
    if (threadIdx.x == 0)
        scores[p] = used[p] ? -1.0 : (isfinite(gain) ? gain : 0.0);
}

extern "C" __global__ void gc_pick(
    const double *scores, int *used, int *indices, int points, int step
) {
    __shared__ double best[256];
    __shared__ int index[256];
    const int t = threadIdx.x;
    double value = -2.0;
    int chosen = points;
    for (int p = t; p < points; p += 256) {
        if (scores[p] > value || (scores[p] == value && p < chosen)) {
            value = scores[p];
            chosen = p;
        }
    }
    best[t] = value;
    index[t] = chosen;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride && (best[t + stride] > best[t] ||
            (best[t + stride] == best[t] && index[t + stride] < index[t]))) {
            best[t] = best[t + stride];
            index[t] = index[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        indices[step] = index[0];
        used[index[0]] = 1;
    }
}

extern "C" __global__ void gc_direction(
    const float *x, const int *indices, float *q, int dim, int step
) {
    __shared__ double scratch[256];
    const int p = indices[step];
    double norm = 0.0;
    for (int d = threadIdx.x; d < dim; d += 256) {
        double v = x[d + p * dim];
        norm += v * v;
    }
    norm = gc_sum(norm, scratch);
    double scale = norm > 1e-30 ? 1.0 / sqrt(norm) : 0.0;
    for (int d = threadIdx.x; d < dim; d += 256)
        q[d] = float(double(x[d + p * dim]) * scale);
}

// Reorthogonalized modified Gram-Schmidt with double-precision reductions.
extern "C" __global__ void gc_deflate(
    float *x, const float *q, int dim, int points
) {
    const int p = blockIdx.x;
    if (p >= points) return;
    __shared__ double scratch[256];
    double dot = 0.0;
    for (int d = threadIdx.x; d < dim; d += 256)
        dot += double(q[d]) * double(x[d + p * dim]);
    dot = gc_sum(dot, scratch);
    for (int d = threadIdx.x; d < dim; d += 256)
        x[d + p * dim] = float(double(x[d + p * dim]) - double(q[d]) * dot);
}

// Fill `size` elements with iid N(0, scale) values.
extern "C" __global__ void standard_gaussian_kernel(
    float *mat, int size, float scale, uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    curandState state;
    curand_init((unsigned long long)seed, (unsigned long long)idx, 0, &state);
    mat[idx] = curand_normal(&state) * scale;
}

// Scale row i of a column-major (rows x cols) matrix by scales[i].
extern "C" __global__ void scale_rows_kernel(
    float *mat, const float *scales, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int i = idx % rows;
    mat[idx] *= scales[i];
}

// Scale col j of a column-major (rows x cols) matrix by scales[j].
extern "C" __global__ void scale_cols_kernel(
    float *mat, const float *scales, int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int j = idx / rows;
    mat[idx] *= scales[j];
}
