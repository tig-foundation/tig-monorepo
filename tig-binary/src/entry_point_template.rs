use anyhow::{anyhow, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tig_algorithms::{CHALLENGE}::{ALGORITHM};
use tig_challenges::{CHALLENGE}::*;

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;


#[cfg(not(feature = "cuda"))]
#[unsafe(no_mangle)]
pub fn entry_point(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        {ALGORITHM}::solve_challenge(challenge, save_solution)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling solve_challenge"))
    })
}


#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub fn entry_point(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Option<Solution>, String>
{
    catch_unwind(AssertUnwindSafe(|| {
        {ALGORITHM}::solve_challenge(challenge, module, stream, prop)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling solve_challenge"))
    })
}