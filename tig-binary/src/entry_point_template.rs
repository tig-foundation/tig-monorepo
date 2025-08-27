use serde_json::{Map, Value};
use std::panic::catch_unwind;
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
pub extern "C" fn entry_point(challenge: &Challenge, hyperparameters: &Option<Map<String, Value>>) -> Result<Option<Solution>, String>
{
    return catch_unwind(|| {
        {ALGORITHM}::solve_challenge(challenge, hyperparameters).map_err(|e| e.to_string())
    }).unwrap_or_else(|_| {
        Err("Panic occurred calling solve_challenge".to_string())
    });
}


#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(
    challenge: &Challenge,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Option<Solution>, String>
{
    return catch_unwind(|| {
        {ALGORITHM}::solve_challenge(challenge, hyperparameters, module, stream, prop).map_err(|e| e.to_string())
    }).unwrap_or_else(|_| {
        Err("Panic occurred calling solve_challenge".to_string())
    });
}