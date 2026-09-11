extern "C" __global__ void adabelief_update_kernel_t27(
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

extern "C" __global__ void adabelief_fused_kernel_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const float* __restrict__ lr_wd,
    const int* __restrict__ offsets,
    float eps,
    int first_step,
    int total_elems,
    int num_tensors,
    float ema_gamma,
    float ema_beta,
    float qh_nu,
    float ab_blend,
    float sophia_rho)
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
    float*       ema_arr     = (float*)all_ptrs[6 * num_tensors + t];
    float*       delta       = (float*)all_ptrs[7 * num_tensors + t];
    float*       envelope    = (float*)all_ptrs[8 * num_tensors + t];
    const float* prev_update = (const float*)all_ptrs[9 * num_tensors + t];
    float lr_i = lr_wd[2 * t];
    float wd_i = lr_wd[2 * t + 1];

    float g     = grad[lidx];
    float theta = param[lidx];
    float raw_g_diff = (first_step != 0) ? 0.0f : (g - pg_arr[lidx]);
    float secant_consistent =
        (first_step != 0 || raw_g_diff * prev_update[lidx] >= 0.0f) ? 1.0f : 0.0f;
    float g_diff = secant_consistent * raw_g_diff;

    float m_new   = 0.98f * m_arr[lidx] + 0.02f * g;
    float v_new   = secant_consistent * (0.92f * v_arr[lidx] + 0.08f * g_diff);
    float g_adan  = g + 0.92f * g_diff;
    float n_tgt;
    if (ab_blend < 0.9999f) {
        float g_ab_e = g - m_new;
        float n_ab   = g_ab_e * g_ab_e + 1e-16f;
        n_tgt = ab_blend * g_adan * g_adan + (1.0f - ab_blend) * n_ab;
    } else {
        n_tgt = g_adan * g_adan;
    }
    float n_new   = 0.99f * s_arr[lidx] + 0.01f * n_tgt;
    float envelope_new = fmaxf(envelope[lidx], n_new);

    m_arr[lidx]    = m_new;
    v_arr[lidx]    = v_new;
    s_arr[lidx]    = n_new;
    envelope[lidx] = envelope_new;
    pg_arr[lidx]   = g;

    float m_qh = qh_nu * m_new + (1.0f - qh_nu) * g;
    float adan_dir;
    if (sophia_rho > 1e-7f) {
        float raw_ratio = (m_qh + 0.92f * v_new) / fmaxf(envelope_new, eps);
        float clip_b    = 1.0f / sophia_rho;
        adan_dir = copysignf(fminf(fabsf(raw_ratio), clip_b), raw_ratio);
    } else {
        adan_dir = (m_qh + 0.92f * v_new) / (sqrtf(envelope_new) + eps);
    }
    float wd_gate  = (adan_dir * theta >= 0.0f) ? 1.0f : 0.0f;
    float dir  = adan_dir + wd_gate * wd_i * theta;
    float keep = (dir * g > 0.0f) ? 1.0f : 0.25f;
    float d_adan = -lr_i * keep * dir;

    float b_old = ema_arr[lidx];
    float b_new = ema_gamma * b_old + (1.0f - ema_gamma) * d_adan;
    ema_arr[lidx] = b_new;
    delta[lidx] = d_adan + ema_beta * (b_new - b_old);
}

extern "C" __global__ void hidden_matrix_biaxial_energy_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const int* __restrict__ geometry_mask,
    float* __restrict__ matrix_energies,
    int num_tensors)
{
    int packed_axis = (int)blockIdx.x;
    int t = packed_axis / 512;
    int axis_slot = packed_axis - t * 512;
    if (t >= num_tensors || geometry_mask[t] == 0) return;

    bool is_column = axis_slot >= 256;
    int axis = is_column ? axis_slot - 256 : axis_slot;
    int other = (int)threadIdx.x;
    float* delta = (float*)all_ptrs[7 * num_tensors + t];
    int idx = is_column ? other * 256 + axis : axis * 256 + other;
    float d = delta[idx];

    __shared__ float sums[256];
    sums[other] = d * d;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (other < stride) {
            sums[other] += sums[other + stride];
        }
        __syncthreads();
    }
    if (other == 0) {
        matrix_energies[packed_axis] = sums[0];
    }
}

extern "C" __global__ void hidden_matrix_biaxial_balance_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const int* __restrict__ geometry_mask,
    const float* __restrict__ matrix_scales,
    int num_tensors)
{
    int packed_row = (int)blockIdx.x;
    int t = packed_row / 256;
    int row = packed_row - t * 256;
    if (t >= num_tensors || geometry_mask[t] == 0) return;

    int col = (int)threadIdx.x;
    float* delta = (float*)all_ptrs[7 * num_tensors + t];
    int scale_base = t * 512;
    delta[row * 256 + col] *=
        matrix_scales[scale_base + row] *
        matrix_scales[scale_base + 256 + col];
}

extern "C" __global__ void ema_tensor_agreement_stats_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const int* __restrict__ offsets,
    unsigned long long stats_ptr,
    int num_tensors)
{
    int t = (int)blockIdx.x;
    if (t >= num_tensors) return;

    float* delta = (float*)all_ptrs[7 * num_tensors + t];
    float* ema = (float*)all_ptrs[6 * num_tensors + t];
    float* stats = (float*)stats_ptr;
    int start = offsets[t];
    int n = offsets[t + 1] - start;
    int tid = (int)threadIdx.x;

    float d2 = 0.0f;
    float e2 = 0.0f;
    float de = 0.0f;
    for (int i = tid; i < n; i += (int)blockDim.x) {
        float d = delta[i];
        float e = ema[i];
        d2 += d * d;
        e2 += e * e;
        de += d * e;
    }

    __shared__ float d2_sum[256];
    __shared__ float e2_sum[256];
    __shared__ float de_sum[256];
    d2_sum[tid] = d2;
    e2_sum[tid] = e2;
    de_sum[tid] = de;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            d2_sum[tid] += d2_sum[tid + stride];
            e2_sum[tid] += e2_sum[tid + stride];
            de_sum[tid] += de_sum[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        stats[3 * t] = d2_sum[0];
        stats[3 * t + 1] = e2_sum[0];
        stats[3 * t + 2] = de_sum[0];
    }
}

extern "C" __global__ void ema_tensor_conservative_route_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const int* __restrict__ offsets,
    const int* __restrict__ routes,
    float ema_gamma,
    float ema_beta,
    int num_tensors)
{
    int t = (int)blockIdx.x;
    if (t >= num_tensors || routes[t] == 0) return;

    float* delta = (float*)all_ptrs[7 * num_tensors + t];
    float* ema = (float*)all_ptrs[6 * num_tensors + t];
    int n = offsets[t + 1] - offsets[t];
    int tid = (int)threadIdx.x;

    if (ema_gamma == 0.0f) {
        for (int i = tid; i < n; i += (int)blockDim.x) {
            delta[i] = ema[i];
        }
    } else {
        float extrapolation =
            ema_beta * (1.0f - ema_gamma) / ema_gamma;
        for (int i = tid; i < n; i += (int)blockDim.x) {
            delta[i] = (delta[i] + extrapolation * ema[i]) /
                       (1.0f + extrapolation);
        }
    }
}

#define NS_DIM  256
#define NS_TILE 16

extern "C" __global__ void ns_frob_kernel(
    unsigned long long m_ptr,
    unsigned long long partial_ptr,
    int n)
{
    const float* m = (const float*)m_ptr;
    float* partials = (float*)partial_ptr;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    float value = (gid < n) ? m[gid] * m[gid] : 0.0f;

    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    __shared__ float warp_sums[8];

    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = (lane < 8) ? warp_sums[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, delta);
        }
        if (lane == 0) partials[blockIdx.x] = value;
    }
}

extern "C" __global__ void ns_copy_normalize(
    unsigned long long m_ptr,
    unsigned long long dst_ptr,
    unsigned long long sq_ptr,
    float eps,
    int n)
{
    const float* m      = (const float*)m_ptr;
    float*       dst    = (float*)dst_ptr;
    const float* sq_sum = (const float*)sq_ptr;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < n) dst[gid] = m[gid] / sqrtf(*sq_sum + eps);
}

extern "C" __global__ void ns_atb_256(
    unsigned long long A_ptr,
    unsigned long long B_ptr,
    unsigned long long C_ptr)
{
    const float* A = (const float*)A_ptr;
    const float* B = (const float*)B_ptr;
    float*       C = (float*)C_ptr;
    __shared__ float sA[NS_TILE][NS_TILE + 1];
    __shared__ float sB[NS_TILE][NS_TILE + 1];
    int row = (int)(blockIdx.y * NS_TILE + threadIdx.y);
    int col = (int)(blockIdx.x * NS_TILE + threadIdx.x);
    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < NS_DIM / NS_TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = A[(t * NS_TILE + (int)threadIdx.x) * NS_DIM + row];
        sB[threadIdx.y][threadIdx.x] = B[(t * NS_TILE + (int)threadIdx.y) * NS_DIM + col];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < NS_TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    C[row * NS_DIM + col] = sum;
}

extern "C" __global__ void ns_ab_256(
    unsigned long long A_ptr,
    unsigned long long B_ptr,
    unsigned long long C_ptr)
{
    const float* A = (const float*)A_ptr;
    const float* B = (const float*)B_ptr;
    float*       C = (float*)C_ptr;
    __shared__ float sA[NS_TILE][NS_TILE + 1];
    __shared__ float sB[NS_TILE][NS_TILE + 1];
    int row = (int)(blockIdx.y * NS_TILE + threadIdx.y);
    int col = (int)(blockIdx.x * NS_TILE + threadIdx.x);
    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < NS_DIM / NS_TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = A[row * NS_DIM + t * NS_TILE + (int)threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t * NS_TILE + (int)threadIdx.y) * NS_DIM + col];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < NS_TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    C[row * NS_DIM + col] = sum;
}

extern "C" __global__ void ns_poly_256(
    unsigned long long X_ptr,
    unsigned long long X2_ptr,
    unsigned long long poly_ptr,
    float a, float b, float c,
    int n)
{
    const float* X    = (const float*)X_ptr;
    const float* X2   = (const float*)X2_ptr;
    float*       poly = (float*)poly_ptr;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < n) {
        int row = gid / NS_DIM;
        int col = gid % NS_DIM;
        float ident = (row == col) ? a : 0.0f;
        poly[gid] = ident + b * X[gid] + c * X2[gid];
    }
}

extern "C" __global__ void muon_ema_delta_256(
    unsigned long long ns_dir_ptr,
    unsigned long long muon_ema_ptr,
    unsigned long long delta_ptr,
    float muon_lr,
    float ema_gamma,
    float ema_beta,
    int n)
{
    const float* ns_dir   = (const float*)ns_dir_ptr;
    float*       muon_ema = (float*)muon_ema_ptr;
    float*       delta    = (float*)delta_ptr;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < n) {
        float d_muon = -muon_lr * ns_dir[gid];
        float b_old  = muon_ema[gid];
        float b_new  = ema_gamma * b_old + (1.0f - ema_gamma) * d_muon;
        muon_ema[gid] = b_new;
        delta[gid]    = d_muon + ema_beta * (b_new - b_old);
    }
}

extern "C" __global__ void adan_muon_candidate_stats_256(
    unsigned long long grad_ptr,
    unsigned long long param_ptr,
    unsigned long long adan_ptr,
    unsigned long long muon_ptr,
    unsigned long long stats_ptr,
    int n)
{
    const float* grad = (const float*)grad_ptr;
    const float* param = (const float*)param_ptr;
    const float* adan = (const float*)adan_ptr;
    const float* muon = (const float*)muon_ptr;
    float* stats = (float*)stats_ptr;
    int tid = (int)threadIdx.x;

    float g2 = 0.0f, p2 = 0.0f, a2 = 0.0f, u2 = 0.0f;
    float ga = 0.0f, gu = 0.0f, pa = 0.0f, pu = 0.0f;
    for (int i = tid; i < n; i += (int)blockDim.x) {
        float g = grad[i];
        float p = param[i];
        float a = adan[i];
        float u = muon[i];
        g2 += g * g;
        p2 += p * p;
        a2 += a * a;
        u2 += u * u;
        ga += g * a;
        gu += g * u;
        pa += p * a;
        pu += p * u;
    }

    __shared__ float sg2[256], sp2[256], sa2[256], su2[256];
    __shared__ float sga[256], sgu[256], spa[256], spu[256];
    sg2[tid] = g2; sp2[tid] = p2; sa2[tid] = a2; su2[tid] = u2;
    sga[tid] = ga; sgu[tid] = gu; spa[tid] = pa; spu[tid] = pu;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sg2[tid] += sg2[tid + stride];
            sp2[tid] += sp2[tid + stride];
            sa2[tid] += sa2[tid + stride];
            su2[tid] += su2[tid + stride];
            sga[tid] += sga[tid + stride];
            sgu[tid] += sgu[tid + stride];
            spa[tid] += spa[tid + stride];
            spu[tid] += spu[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        stats[0] = sg2[0];
        stats[1] = sp2[0];
        stats[2] = sa2[0];
        stats[3] = su2[0];
        stats[4] = sga[0];
        stats[5] = sgu[0];
        stats[6] = spa[0];
        stats[7] = spu[0];
    }
}

extern "C" __global__ void muon_candidate_commit_256(
    unsigned long long muon_ptr,
    unsigned long long delta_ptr,
    int n)
{
    const float* muon = (const float*)muon_ptr;
    float* delta = (float*)delta_ptr;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid < n) delta[gid] = muon[gid];
}

extern "C" __global__ __launch_bounds__(128, 6) void sign_ef_consensus_kernel_14(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ fisher_diag,
    float* __restrict__ ef_residual,
    float* __restrict__ slow_update,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float eps,
    const float weight_decay,
    const float rel_update_cap,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const unsigned int stride = blockDim.x * gridDim.x;

    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 0.025f;
    const float inv_lr = 1.0f / fmaxf(lr, 1.0e-8f);
    const float wd_lr = lr * weight_decay;
    const float abs_floor = 1.0e-3f;
    const float min_step = 0.18f * lr;
    const float ef_cap = 6.0f * lr;

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        float fd = fisher_diag[idx];
        const float g = gradients[idx];
        const float w = params[idx];

        const float fd_std = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_clipped = fminf(fmaxf(g, -4.0f * fd_std), 4.0f * fd_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        fisher_diag[idx] = fd;

        const float rms = sqrtf(fmaxf(fd, 0.0f)) + eps;
        const float g_n = g_clipped / rms;

        float ef_old = ef_residual[idx];
        ef_old = fminf(fmaxf(ef_old, -ef_cap), ef_cap);
        const float combined = g_n + ef_old * inv_lr;
        const float u_quant = -lr * copysignf(1.0f, combined);
        const float ef_new = ef_old - u_quant;
        ef_residual[idx] = fminf(fmaxf(ef_new, -ef_cap), ef_cap);

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * u_quant;
        const float final_update = one_minus_la_alpha * u_quant + lookahead_alpha * su;

        const float target = lr * fabsf(g_n);
        const float uabs = fabsf(final_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        float adj_update = final_update * scale;

        adj_update -= wd_lr * w;

        const float max_step = fmaxf(rel_update_cap * (fabsf(w) + abs_floor), min_step);
        adj_update = fminf(fmaxf(adj_update, -max_step), max_step);

        slow_update[idx] = su;
        updates[idx] = adj_update;
    }
}

extern "C" __global__ __launch_bounds__(128, 6) void dual_consensus_fisher_kernel_14(
    const float* __restrict__ gradients,
    const float* __restrict__ params,
    float* __restrict__ momentum,
    float* __restrict__ velocity,
    float* __restrict__ prev_grad,
    float* __restrict__ prev_update,
    float* __restrict__ slow_update,
    float* __restrict__ fisher_diag,
    float* __restrict__ updates,
    const unsigned int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const float blend_adam,
    const float blend_norm,
    const float blend_sign,
    const float nesterov_gamma,
    const float bb_blend,
    const float lookahead_alpha,
    const float lookahead_tau,
    const float gate_lo,
    const float gate_hi
) {
    (void)params;
    __shared__ float smem[4];

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int n_warps = (blockDim.x + 31) >> 5;

    const float inv_bc1 = 1.0f / fmaxf(bias_correction1, 1.0e-8f);
    const float inv_bc2 = 1.0f / fmaxf(bias_correction2, 1.0e-8f);
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float one_minus_la_tau = 1.0f - lookahead_tau;
    const float one_minus_la_alpha = 1.0f - lookahead_alpha;
    const float fisher_beta = 0.975f;
    const float one_minus_fisher_beta = 0.025f;
    const float ortho_mix = fminf(fmaxf(0.22f + 0.38f * blend_sign, 0.0f), 0.58f);

    float local_sq_sum = 0.0f;
    unsigned int local_count = 0;
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        local_sq_sum += g * g;
        local_count++;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
        local_count  += __shfl_down_sync(0xffffffff, local_count,  offset);
    }
    if (lane == 0) smem[warp_id] = (local_count > 0) ? (local_sq_sum / (float)local_count) : 0.0f;
    __syncthreads();

    float block_grad_rms = 0.0f;
    if (warp_id == 0) {
        float val = (lane < n_warps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) smem[0] = sqrtf(fmaxf(val / fmaxf((float)n_warps, 1.0f), 0.0f)) + eps;
    }
    __syncthreads();
    block_grad_rms = smem[0];
    const float inv_block_rms = 1.0f / fmaxf(block_grad_rms, 1.0e-8f);

    #pragma unroll 2
    for (unsigned int idx = tid; idx < n; idx += stride) {
        const float g = gradients[idx];
        const float pg = prev_grad[idx];

        const float gamma_local = (g * pg >= 0.0f) ? nesterov_gamma : (0.20f * nesterov_gamma);
        const float g_pred = g + gamma_local * (g - pg);

        float m = momentum[idx];
        float v = velocity[idx];

        m = beta1 * m + one_minus_beta1 * g_pred;
        const float err = g_pred - m;
        v = beta2 * v + one_minus_beta2 * err * err;

        const float m_hat = m * inv_bc1;
        const float v_hat = v * inv_bc2;

        const float sqrt_v = sqrtf(fmaxf(v_hat, 0.0f));
        const float adaptive_eps = eps * (1.0f + 0.08f * sqrt_v);
        const float denom = sqrt_v + adaptive_eps;
        const float inv_denom = 1.0f / fmaxf(denom, 1.0e-12f);

        const float adam_update = -lr * (m_hat * inv_denom + weight_decay * g_pred);
        const float g_over_denom = g_pred * inv_denom;
        const float norm_update = -lr * g_over_denom;
        const float sign_update = -lr * copysignf(1.0f, m_hat);
        float base_update = blend_adam * adam_update + blend_norm * norm_update + blend_sign * sign_update;

        const float overlap = copysignf(fminf(fabsf(g_pred), fabsf(m_hat)), m_hat);
        const float g_ortho = g_pred - overlap;
        const float ortho_update = -lr * (g_ortho * inv_denom);
        base_update = (1.0f - ortho_mix) * base_update + ortho_mix * ortho_update;

        const float s_pu = prev_update[idx];
        const float s_mag = fabsf(s_pu);
        const float bb_scale = (s_mag > 1e-6f) ? fminf(s_mag * 2.0f, 2.2f) : 1.0f;
        base_update *= (1.0f - bb_blend * 0.3f) + (bb_blend * 0.3f) * bb_scale;

        float fd = fisher_diag[idx];
        const float sqrt_v_for_clip = sqrtf(fmaxf(v, 0.0f));
        const float grad_std = sqrt_v_for_clip + eps;
        const float g_clipped = fminf(fmaxf(g_pred, -5.0f * grad_std), 5.0f * grad_std);
        fd = fisher_beta * fd + one_minus_fisher_beta * g_clipped * g_clipped;
        const float fisher_rms = sqrtf(fmaxf(fd, 0.0f)) + eps;

        const float fisher_norm_update = -lr * (g_pred / fisher_rms);
        const float robust_track = 0.4f * sign_update + 0.6f * fisher_norm_update;

        const float flip = (g * pg < 0.0f) ? 1.0f : 0.0f;
        const float vol = fminf(sqrt_v * 0.33333334f, 1.0f);
        const float agree = (base_update * robust_track >= 0.0f) ? 1.0f : 0.0f;
        const float grad_mom_align = (g_pred * m_hat >= 0.0f) ? 1.0f : 0.0f;
        const float stability = grad_mom_align * (1.0f - flip);

        const float g_abs = fabsf(g);
        const float pg_abs = fabsf(pg);
        const float curvature = fabsf(g - pg) / fmaxf(g_abs + pg_abs + eps, 1.0e-8f);
        const float curvature_clamped = fminf(curvature, 1.0f);

        float consensus_mix = 0.25f * curvature_clamped
                            + 0.25f * (1.0f - agree)
                            + 0.20f * vol
                            + 0.18f * blend_sign
                            + 0.12f * (1.0f - stability);
        consensus_mix = fminf(fmaxf(consensus_mix, 0.0f), 1.0f);
        float chosen_update = (1.0f - consensus_mix) * base_update + consensus_mix * robust_track;

        const float align_strength = grad_mom_align * (1.0f - flip) * (1.0f - vol);
        const float trust = (1.0f + 0.10f * align_strength) / (1.0f + 0.55f * vol + 0.55f * flip);
        chosen_update *= trust;

        const float elem_rms = g_abs * inv_block_rms;
        const float block_norm_scale = 1.0f / fmaxf(0.6f + 0.4f * elem_rms, 1.0e-8f);
        chosen_update *= block_norm_scale;

        const float target = lr * fabsf(g_over_denom);
        const float uabs = fabsf(chosen_update);
        const float scale = fminf(fmaxf(target / fmaxf(uabs, 1.0e-12f), gate_lo), gate_hi);
        chosen_update *= scale;

        float su = slow_update[idx];
        su = one_minus_la_tau * su + lookahead_tau * chosen_update;
        const float final_update = one_minus_la_alpha * chosen_update + lookahead_alpha * su;

        momentum[idx] = m;
        velocity[idx] = v;
        prev_grad[idx] = g;
        prev_update[idx] = final_update;
        slow_update[idx] = su;
        fisher_diag[idx] = fd;
        updates[idx] = final_update;
    }
}

extern "C" __global__ void grokfast_ema_kernel(
    unsigned long long g_raw,
    unsigned long long mu_raw,
    unsigned long long g_aug_raw,
    float alpha,
    float lambda,
    int n)
{
    const float* g = (const float*)g_raw;
    float* mu    = (float*)mu_raw;
    float* g_aug = (float*)g_aug_raw;
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float gi = g[i];
    float mi = alpha * mu[i] + (1.0f - alpha) * gi;
    mu[i]    = mi;
    g_aug[i] = gi + lambda * mi;
}

extern "C" __global__ void mars_tensor_drift_stats_kernel(
    unsigned long long g_raw,
    unsigned long long g_prev_raw,
    unsigned long long stats_raw,
    int n)
{
    const float* g = (const float*)g_raw;
    const float* g_prev = (const float*)g_prev_raw;
    float* stats = (float*)stats_raw;
    __shared__ float g_sq[256];
    __shared__ float prev_sq[256];
    __shared__ float dot[256];

    int tid = (int)threadIdx.x;
    float local_g_sq = 0.0f;
    float local_prev_sq = 0.0f;
    float local_dot = 0.0f;
    for (int i = tid; i < n; i += (int)blockDim.x) {
        float gi = g[i];
        float pi = g_prev[i];
        local_g_sq += gi * gi;
        local_prev_sq += pi * pi;
        local_dot += gi * pi;
    }
    g_sq[tid] = local_g_sq;
    prev_sq[tid] = local_prev_sq;
    dot[tid] = local_dot;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            g_sq[tid] += g_sq[tid + stride];
            prev_sq[tid] += prev_sq[tid + stride];
            dot[tid] += dot[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        stats[0] = g_sq[0];
        stats[1] = prev_sq[0];
        stats[2] = dot[0];
    }
}

extern "C" __global__ void mars_correction_kernel(
    unsigned long long g_raw,
    unsigned long long g_prev_raw,
    unsigned long long c_raw,
    float gamma,
    float mars_clip_scale,
    int route_correction,
    int first_step,
    int n)
{
    const float* g   = (const float*)g_raw;
    float* g_prev    = (float*)g_prev_raw;
    float* c         = (float*)c_raw;
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float eps = 1e-8f;
    float gi = g[i];
    float g_diff = (first_step != 0) ? 0.0f : (gi - g_prev[i]);
    float corr = (gamma * 49.0f) * g_diff;
    float cap  = mars_clip_scale * fabsf(gi) + eps;
    corr = fminf(fmaxf(corr, -cap), cap);
    c[i]      = (route_correction != 0) ? (gi + corr) : gi;
    g_prev[i] = gi;
}

extern "C" __global__ void matrix_update_polarize_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const int* __restrict__ offsets,
    const int* __restrict__ rows,
    const int* __restrict__ column_offsets,
    int total_columns,
    int num_tensors)
{
    int column_index = (int)blockIdx.x;
    if (column_index >= total_columns) return;

    int lo = 0, hi = num_tensors - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (column_offsets[mid + 1] <= column_index) lo = mid + 1;
        else hi = mid;
    }
    int t = lo;
    int local_column = column_index - column_offsets[t];
    int row_count = rows[t];
    int width = (offsets[t + 1] - offsets[t]) / row_count;
    float* delta = (float*)all_ptrs[7 * num_tensors + t];

    float sum = 0.0f;
    for (int row = (int)threadIdx.x; row < row_count; row += (int)blockDim.x) {
        sum += delta[row * width + local_column];
    }

    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = lane < 8 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) warp_sums[0] = sum / (float)row_count;
    }
    __syncthreads();

    float common_motion = warp_sums[0];
    for (int row = (int)threadIdx.x; row < row_count; row += (int)blockDim.x) {
        delta[row * width + local_column] -= common_motion;
    }
}

extern "C" __global__ void record_applied_update_fused_t27(
    const unsigned long long* __restrict__ all_ptrs,
    const int* __restrict__ offsets,
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

    float* prev_update = (float*)all_ptrs[9 * num_tensors + t];
    const float* delta = (const float*)all_ptrs[7 * num_tensors + t];
    prev_update[lidx] = delta[lidx];
}

extern "C" __global__ void dc_axpy_t27(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int n
) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] += src[i];
    }
}