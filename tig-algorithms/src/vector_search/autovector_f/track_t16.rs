
use anyhow::Result;
use cudarc::driver::{
    safe::{CudaModule, CudaStream, LaunchConfig},
    PushKernelArg,
};
use cudarc::runtime::sys::cudaDeviceProp;
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
const RABIT_WORDS: usize = 8;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> Result<()> {
    let nq  = challenge.num_queries   as usize;
    let db  = challenge.database_size as usize;
    let nq_i32   = nq as i32;
    let db_i32   = db as i32;
    let dims_i32     = DIMS     as i32;
    let dims_pad_i32 = DIMS_PAD as i32;

    let nq_pad = (nq + 7) / 8 * 8;
    let db_pad = (db + 31) / 32 * 32;

    
    let conv_fn    = module.load_function("convert_fp32_to_fp16_padded_t16")?;
    let norms_fn   = module.load_function("compute_norms_fp16_t16")?;
    let rabit_fn   = module.load_function("encode_rabitq_flat_t16")?;
    let bf_gemm_fn = module.load_function("bruteforce_gemm_argmin_t16")?;

    
    let mut d_q_fp16  = stream.alloc_zeros::<u16>(nq_pad * DIMS_PAD)?;
    let mut d_db_fp16 = stream.alloc_zeros::<u16>(db_pad * DIMS_PAD)?;

    let q_blk  = ((nq as u32) + 255) / 256;
    let db_blk = ((db as u32) + 255) / 256;

    unsafe {
        stream.launch_builder(&conv_fn)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_q_fp16)
            .arg(&nq_i32).arg(&dims_i32).arg(&dims_pad_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&conv_fn)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_db_fp16)
            .arg(&db_i32).arg(&dims_i32).arg(&dims_pad_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    
    let mut d_norms_q  = stream.alloc_zeros::<f32>(nq)?;
    let mut d_norms_db = stream.alloc_zeros::<f32>(db)?;
    unsafe {
        stream.launch_builder(&norms_fn)
            .arg(&d_q_fp16)
            .arg(&mut d_norms_q)
            .arg(&nq_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&norms_fn)
            .arg(&d_db_fp16)
            .arg(&mut d_norms_db)
            .arg(&db_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    
    let mut d_rabitq_q  = stream.alloc_zeros::<u32>(nq * RABIT_WORDS)?;
    let mut d_rabitq_db = stream.alloc_zeros::<u32>(db * RABIT_WORDS)?;
    unsafe {
        stream.launch_builder(&rabit_fn)
            .arg(&challenge.d_query_vectors)
            .arg(&mut d_rabitq_q)
            .arg(&nq_i32).arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (q_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        stream.launch_builder(&rabit_fn)
            .arg(&challenge.d_database_vectors)
            .arg(&mut d_rabitq_db)
            .arg(&db_i32).arg(&dims_i32)
            .launch(LaunchConfig { grid_dim: (db_blk, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
    }

    
    let mut d_best_idx = stream.alloc_zeros::<i32>(nq)?;
    unsafe {
        stream.launch_builder(&bf_gemm_fn)
            .arg(&d_q_fp16)
            .arg(&d_db_fp16)
            .arg(&d_norms_q)
            .arg(&d_norms_db)
            .arg(&d_rabitq_q)
            .arg(&d_rabitq_db)
            .arg(&mut d_best_idx)
            .arg(&nq_i32)
            .arg(&db_i32)
            .launch(LaunchConfig {
                grid_dim: (82, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    stream.synchronize()?;

    let result_indices: Vec<i32> = stream.memcpy_dtov(&d_best_idx)?;
    let indexes: Vec<usize> = result_indices
        .iter()
        .map(|&idx| if idx < 0 || idx >= db_i32 { 0 } else { idx as usize })
        .collect();

    save_solution(&Solution { indexes })?;
    Ok(())
}

