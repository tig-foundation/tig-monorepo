#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void adabelief_update_kernel_18(
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

        float adan_dir = (m_new + 0.92f * v_new) / (sqrtf(n_new) + eps);
        float wd_gate  = (adan_dir * theta >= 0.0f) ? 1.0f : 0.0f;
        float dir  = adan_dir + wd_gate * wd * theta;
        float keep = (dir * g > 0.0f) ? 1.0f : 0.25f;
        delta[idx] = -lr * keep * dir;
    }
}

extern "C" __global__ void adabelief_fused_kernel_18(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ lr_wd,
    const int* __restrict__ offsets,
    float eps,
    int first_step,
    int total_elems,
    int num_tensors)
{
    extern __shared__ int s_off[];
    if ((int)threadIdx.x <= num_tensors) {
        s_off[threadIdx.x] = offsets[threadIdx.x];
    }
    __syncthreads();

    int gidx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gidx >= total_elems) return;

    int lo = 0, hi = num_tensors - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (s_off[mid + 1] <= gidx) lo = mid + 1;
        else hi = mid;
    }
    int t = lo;
    int lidx = gidx - s_off[t];

    const float* grad     = (const float*)all_ptrs[0 * num_tensors + t];
    const float* param    = (const float*)all_ptrs[1 * num_tensors + t];
    float*       m_arr    = (float*)all_ptrs[2 * num_tensors + t];
    float*       v_arr    = (float*)all_ptrs[3 * num_tensors + t];
    float*       s_arr    = (float*)all_ptrs[4 * num_tensors + t];
    float*       pg_arr   = (float*)all_ptrs[5 * num_tensors + t];
    float*       delta    = (float*)all_ptrs[6 * num_tensors + t];
    float lr_i = lr_wd[2 * t];
    float wd_i = lr_wd[2 * t + 1];

    float g     = grad[lidx];
    float theta = param[lidx];
    float g_diff = (first_step != 0) ? 0.0f : (g - pg_arr[lidx]);

    float m_new = 0.98f * m_arr[lidx] + 0.02f * g;
    float v_new = 0.92f * v_arr[lidx] + 0.08f * g_diff;
    float g_n   = g + 0.92f * g_diff;
    float n_new = 0.99f * s_arr[lidx] + 0.01f * g_n * g_n;

    m_arr[lidx]  = m_new;
    v_arr[lidx]  = v_new;
    s_arr[lidx]  = n_new;
    pg_arr[lidx] = g;

    float adan_dir = (m_new + 0.92f * v_new) / (sqrtf(n_new) + eps);
    float wd_gate  = (adan_dir * theta >= 0.0f) ? 1.0f : 0.0f;
    float dir  = adan_dir + wd_gate * wd_i * theta;
    float keep = (dir * g > 0.0f) ? 1.0f : 0.25f;
    delta[lidx] = -lr_i * keep * dir;
}
