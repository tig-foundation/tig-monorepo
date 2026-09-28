// QR/SVD helpers derived from sketchy.
use anyhow::{anyhow, Result};
use core::ffi::c_int;
use cudarc::{
    cublas::{
        sys::{self as cublas_sys, cublasOperation_t},
        CudaBlas,
    },
    cusolver::{sys as cusolver_sys, DnHandle},
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
};
use std::sync::Arc;

/// In-place QR decomposition: d_mat (m×n, col-major) → Q (m×n, orthonormal cols).
pub(super) fn gpu_qr(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<()> {
    let min_mn = m.min(n);
    let mut lwork = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSgeqrf_bufferSize(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            &mut lwork,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgeqrf_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut d_tau = stream.alloc_zeros::<f32>(min_mn as usize)?;
    unsafe {
        if cusolver_sys::cusolverDnSgeqrf(
            cusolver.cu(),
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *mut f32,
            d_work.device_ptr_mut(stream).0 as *mut f32,
            lwork,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgeqrf failed"));
        }
    }
    stream.synchronize()?;
    let info = stream.memcpy_dtov(&d_info)?;
    if info[0] != 0 {
        return Err(anyhow!("GPU QR factorization failed with info={}", info[0]));
    }
    let mut lwork_q = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSorgqr_bufferSize(
            cusolver.cu(),
            m,
            n,
            min_mn,
            d_mat.device_ptr_mut(stream).0 as *const f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            &mut lwork_q,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSorgqr_bufferSize failed"));
        }
    }
    let mut d_work_q = stream.alloc_zeros::<f32>((lwork_q as usize).max(1))?;
    unsafe {
        if cusolver_sys::cusolverDnSorgqr(
            cusolver.cu(),
            m,
            n,
            min_mn,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_tau.device_ptr_mut(stream).0 as *const f32,
            d_work_q.device_ptr_mut(stream).0 as *mut f32,
            lwork_q,
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSorgqr failed"));
        }
    }
    stream.synchronize()?;
    let info = stream.memcpy_dtov(&d_info)?;
    if info[0] != 0 {
        return Err(anyhow!("GPU QR failed with info={}", info[0]));
    }
    Ok(())
}

/// Thin SVD of d_mat (m×n). Returns (d_u, s_cpu, d_vt).
/// d_u: m×p (GPU), s_cpu: Vec<f32> of p (CPU), d_vt: p×n (GPU), where p = min(m,n).
/// d_mat is destroyed by this call.
/// cusolverDnSgesvd requires m >= n; when m < n we transpose, compute SVD of the
/// tall matrix, then recover U and Vt for the original via U_orig = Vt_T^T, Vt_orig = U_T^T.
pub(super) fn gpu_svd_thin(
    cusolver: &DnHandle,
    stream: &Arc<CudaStream>,
    d_mat: &mut CudaSlice<f32>,
    m: c_int,
    n: c_int,
) -> Result<(CudaSlice<f32>, Vec<f32>, CudaSlice<f32>)> {
    if m < n {
        // cuSOLVER's GESVD requires a tall matrix. Transpose entirely on the
        // GPU, factor A^T, then transpose the factors back. Keeping this path
        // on-device avoids two large host transfers for the k-by-n projected
        // matrices used by the selector.
        let m_sz = m as usize;
        let n_sz = n as usize;
        let cublas = CudaBlas::new(stream.clone())?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let mut d_at = stream.alloc_zeros::<f32>(m_sz * n_sz)?;
        unsafe {
            cublas_sys::cublasSgeam(
                *cublas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                n,
                m,
                &alpha,
                d_mat.device_ptr(stream).0 as *const f32,
                m,
                &beta,
                d_mat.device_ptr(stream).0 as *const f32,
                m,
                d_at.device_ptr_mut(stream).0 as *mut f32,
                n,
            )
            .result()?;
        }

        // SVD of A^T (n×m, n >= m): returns (U_T: n×m, s: m, Vt_T: m×m).
        let (d_u_t, s, d_vt_t) = gpu_svd_thin(cusolver, stream, &mut d_at, n, m)?;
        drop(d_at);

        let mut d_u_a = stream.alloc_zeros::<f32>(m_sz * m_sz)?;
        let mut d_vt_a = stream.alloc_zeros::<f32>(m_sz * n_sz)?;
        unsafe {
            cublas_sys::cublasSgeam(
                *cublas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                m,
                m,
                &alpha,
                d_vt_t.device_ptr(stream).0 as *const f32,
                m,
                &beta,
                d_vt_t.device_ptr(stream).0 as *const f32,
                m,
                d_u_a.device_ptr_mut(stream).0 as *mut f32,
                m,
            )
            .result()?;
            cublas_sys::cublasSgeam(
                *cublas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                m,
                n,
                &alpha,
                d_u_t.device_ptr(stream).0 as *const f32,
                n,
                &beta,
                d_u_t.device_ptr(stream).0 as *const f32,
                n,
                d_vt_a.device_ptr_mut(stream).0 as *mut f32,
                m,
            )
            .result()?;
        }

        return Ok((d_u_a, s, d_vt_a));
    }

    let p = m.min(n);
    let p_sz = p as usize;
    let mut d_u = stream.alloc_zeros::<f32>(m as usize * p_sz)?;
    let mut d_s = stream.alloc_zeros::<f32>(p_sz)?;
    let mut d_vt = stream.alloc_zeros::<f32>(p_sz * n as usize)?;
    let mut d_info = stream.alloc_zeros::<i32>(1)?;
    let mut lwork = 0i32;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd_bufferSize(cusolver.cu(), m, n, &mut lwork)
            != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgesvd_bufferSize failed"));
        }
    }
    let mut d_work = stream.alloc_zeros::<f32>((lwork as usize).max(1))?;
    let jobu = b'S' as i8;
    let jobvt = b'S' as i8;
    unsafe {
        if cusolver_sys::cusolverDnSgesvd(
            cusolver.cu(),
            jobu,
            jobvt,
            m,
            n,
            d_mat.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_s.device_ptr_mut(stream).0 as *mut f32,
            d_u.device_ptr_mut(stream).0 as *mut f32,
            m,
            d_vt.device_ptr_mut(stream).0 as *mut f32,
            p,
            d_work.device_ptr_mut(stream).0 as *mut f32,
            lwork,
            std::ptr::null_mut(),
            d_info.device_ptr_mut(stream).0 as *mut i32,
        ) != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS
        {
            return Err(anyhow!("cusolverDnSgesvd failed"));
        }
    }
    stream.synchronize()?;
    // Check d_info: >0 means SVD didn't converge; U/Vt may contain NaN.
    let info_vec = stream.memcpy_dtov(&d_info)?;
    if info_vec[0] != 0 {
        return Err(anyhow!(
            "cusolverDnSgesvd did not converge (info={})",
            info_vec[0]
        ));
    }
    let s_cpu = stream.memcpy_dtov(&d_s)?;
    Ok((d_u, s_cpu, d_vt))
}
