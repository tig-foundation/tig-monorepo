use anyhow::Result;
use cudarc::cudnn::{self, result::CudnnError};
use std::ops::Deref;

pub struct CudnnTensorDescriptor(cudnn::sys::cudnnTensorDescriptor_t);

impl CudnnTensorDescriptor {
    pub fn new() -> Result<Self, CudnnError> {
        let mut desc = std::ptr::null_mut();
        unsafe {
            match cudnn::sys::cudnnCreateTensorDescriptor(&mut desc) {
                cudnn::sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(Self(desc)),
                e => Err(CudnnError(e)),
            }
        }
    }

    pub fn set_4d(
        &mut self,
        format: cudnn::sys::cudnnTensorFormat_t,
        data_type: cudnn::sys::cudnnDataType_t,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<(), CudnnError> {
        unsafe {
            match cudnn::sys::cudnnSetTensor4dDescriptor(self.0, format, data_type, n, c, h, w) {
                cudnn::sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
                e => Err(CudnnError(e)),
            }
        }
    }
}

impl Deref for CudnnTensorDescriptor {
    type Target = cudnn::sys::cudnnTensorDescriptor_t;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for CudnnTensorDescriptor {
    fn drop(&mut self) {
        unsafe {
            cudnn::sys::cudnnDestroyTensorDescriptor(self.0);
        }
    }
}
