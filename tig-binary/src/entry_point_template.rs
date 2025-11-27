use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
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
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: Option<String>,
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::solve_challenge(challenge, save_solution, &hyperparameters)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling solve_challenge"))
    })
}


#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub fn entry_point(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: Option<String>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::solve_challenge(challenge, save_solution, &hyperparameters, module, stream, prop)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling solve_challenge"))
    })
}

#[no_mangle]
pub extern "C" fn help() {
    {ALGORITHM}::help();
}
