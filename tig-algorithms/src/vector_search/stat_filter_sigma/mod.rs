// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

// when launching kernels, you should not exceed this const or else it may not be deterministic
//const MAX_THREADS_PER_BLOCK: u32 = 1024;

//
// stat_filter
//
// Filtering based on Median Absolute Deviation (MAD):
// We compute the median of all L2 norms, then calculate the MAD (median of
// absolute deviations from the median). The threshold is set to:
//      norm_threshold = scale_factor × MAD × 1.4826
// The factor 1.4826 scales MAD to match the standard deviation for normally
// distributed data. This makes the filter more robust to outliers compared to
// filtering methods based on mean and standard deviation, which are more
// sensitive to extreme values.
//
// Reference:
// - NIST Engineering Statistics Handbook:
//   https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
// - See also: https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm
//

//use crate::{seeded_hasher, HashMap, HashSet};

use anyhow::{anyhow, Result};
use cudarc::driver::PushKernelArg;
use cudarc::{
    driver::{CudaModule, CudaStream, LaunchConfig},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::vector_search::*;
// use std::env;

const MAD_SCALE_NORMAL: f32 = 1.4826;
const MAX_THREADS_PER_BLOCK: u32 = 1024;

// Default K for Top-K retrieval (must be <= kernel KMAX)
pub const DEFAULT_TOP_K: usize = 10;
// Maximum K supported by the CUDA kernels (must match KMAX in kernels.cu).
const TOPK_MAX: usize = 64;
//pub const DEFAULT_TOP_K: usize = 25;
//pub const DEFAULT_TOP_K: usize = 40;

// Default bit mode (4 or 2)
//pub const DEFAULT_BIT_MODE: usize = 4;

#[derive(Copy, Clone, Debug)]
enum BitMode {
    U2,
    U4,
}

impl BitMode {
//    #[inline]
//    fn from_env() -> Self {
//        match std::env::var("STATFILT_BIT_MODE").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
//            Some(2) => BitMode::U2,
//            _       => BitMode::U4, // default
//        }
//    }
    #[inline]
    fn bits(self) -> usize {
        match self {
            BitMode::U2 => 2,
            BitMode::U4 => 4,
        }
    }
    #[inline]
    fn planes(self) -> usize {
        self.bits()
    } // one bit-plane per bit
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            BitMode::U2 => "2",
            BitMode::U4 => "4",
        }
    }
    #[inline]
    fn k(self, template: &str) -> String {
        // Replace "{b}" with "2" or "4" in a kernel name template.
        template.replace("{b}", self.as_str())
    }
}

//
// Hyperparameter configuration passed from TIG (MAD scale, bit mode, internal top-k).
//
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct Hyperparameters {
    /// Internal top-k used during the bitsliced search and refinement.
    top_k: usize,
    /// Number of bits per dimension to use (2 or 4).
    bit_mode: usize,
    /// MAD scale factor. Values in (0, 5) enable MAD filtering,
    /// values >= 5 disable it. A value of 0.0 enables "legacy auto-MAD"
    /// mode, which computes the scale dynamically based on num_queries.
    mad_scale: f32,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K, // 10
            bit_mode: 2,          // 2-bit
            // 5.0 effectively disables MAD filtering with the current threshold logic.
            mad_scale: 5.0,
        }
    }
}

//
// ================ Solve Challenge Function ================

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    // Parse hyperparameters (if provided) or fall back to defaults.
    let hparams: Hyperparameters = match hyperparameters {
        Some(hyperparameters) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => Hyperparameters::default(),
    };

    // Validate bit_mode; we only support 2-bit and 4-bit variants.
    if hparams.bit_mode != 2 && hparams.bit_mode != 4 {
        return Err(anyhow!(
            "Invalid bit_mode: {}. Must be 2 or 4.",
            hparams.bit_mode
        ));
    }

    println!("Searching {} DB vectors of length {} for {} queries",challenge.database_size,challenge.vector_dims,challenge.difficulty.num_queries);

    // let start_time_total = std::time::Instant::now();

    // Get top-k value to use (hyperparameter, clamped to what the kernels support)
    let mut topk = hparams.top_k.max(1);
    let max_topk = std::cmp::min(challenge.database_size as usize, TOPK_MAX);
    if topk > max_topk {
        topk = max_topk;
    }

    // Get bit mode value to use from hyperparameters (2-bit or 4-bit)
    let mode = match hparams.bit_mode {
        2 => BitMode::U2,
        4 => BitMode::U4,
        _ => unreachable!(), // validated above
    };

    //println!("mode = {} bits; topk = {}", mode.bits(), topk);

    // Allocations for dimension statistics
    let mut d_db_dim_min = stream.alloc_zeros::<f32>(challenge.vector_dims as usize)?;
    let mut d_db_dim_max = stream.alloc_zeros::<f32>(challenge.vector_dims as usize)?;
    let d_s = stream.alloc_zeros::<f32>(challenge.vector_dims as usize)?;

    // Allocations for norms
    let d_db_norm_l2 = stream.alloc_zeros::<f32>(challenge.database_size as usize)?;
    let d_db_norm_l2_squared = stream.alloc_zeros::<f32>(challenge.database_size as usize)?;
    let d_query_norm_l2 = stream.alloc_zeros::<f32>(challenge.difficulty.num_queries as usize)?;
    let d_query_norm_l2_squared =
        stream.alloc_zeros::<f32>(challenge.difficulty.num_queries as usize)?;

    // Total number of elements in DB and queries
    //let num_db_el = challenge.database_size * challenge.vector_dims;
    //let num_qv_el = challenge.difficulty.num_queries * challenge.vector_dims;


    // ---------- Packed buffers ----------

    let dims = challenge.vector_dims as usize;
    let n_db = challenge.database_size as usize;
    let n_q = challenge.difficulty.num_queries as usize;

    // Packed bytes per row for N-bit values = ceil(dims * bits / 8)
    let row_bytes = (dims * mode.bits() + 7) >> 3;
    let num_db_bytes = n_db * row_bytes;
    let num_qv_bytes = n_q * row_bytes;

    // Allocate packed outputs
    let d_db_packed = stream.alloc_zeros::<u8>(num_db_bytes)?;
    let d_qv_packed = stream.alloc_zeros::<u8>(num_qv_bytes)?;
/*
    let d_db_packed = alloc_uninit::<u8>(&stream, num_db_bytes)?;
    let d_qv_packed = alloc_uninit::<u8>(&stream, num_qv_bytes)?;
*/

    // load kernels
    let init_minmax_kernel = module.load_function("init_minmax_kernel")?;
    let compute_dim_stats_kernel = module.load_function("compute_dim_stats_kernel")?;

    // launch config (use counts of VECTORS for stats kernels)
    let threads_db: u32 = 256;
    let blocks_db: u32 = ((challenge.database_size as u32) + threads_db - 1) / threads_db;

    let threads_init: u32 = 256;
    let blocks_init: u32 = ((challenge.vector_dims as u32) + threads_init - 1) / threads_init;

    // initialize min/max arrays on device
    let min_init: f32 = f32::INFINITY;
    let max_init: f32 = f32::NEG_INFINITY; // or 0.0 if values are known >= 0

    unsafe {
        stream
            .launch_builder(&init_minmax_kernel)
            .arg(&mut d_db_dim_min)
            .arg(&mut d_db_dim_max)
            .arg(&challenge.vector_dims)
            .arg(&min_init)
            .arg(&max_init)
            .launch(LaunchConfig {
                grid_dim: (blocks_init, 1, 1),
                block_dim: (threads_init, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    // compute per-dim min & max... scan original data
    unsafe {
        stream
            .launch_builder(&compute_dim_stats_kernel)
            .arg(&challenge.d_database_vectors) // const float* db
            .arg(&mut d_db_dim_min) // float* out_min
            .arg(&mut d_db_dim_max) // float* out_max
            .arg(&challenge.database_size) // num_vecs
            .arg(&challenge.vector_dims) // dims
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (challenge.vector_dims as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    //stream.synchronize()?;


    //
    // ---------- Compute Dimensional Stats ----------
    //

    let threads_db: u32 = 256;
    //let blocks_db: u32 = ((num_db_el as u32) + threads_db - 1) / threads_db;
    let blocks_db: u32 = ((challenge.vector_dims as u32) + threads_db - 1) / threads_db;

    let threads_qv: u32 = 256;
    //let blocks_qv: u32 = ((num_qv_el as u32) + threads_qv - 1) / threads_qv;
    let blocks_qv: u32 = ((challenge.vector_dims as u32) + threads_qv - 1) / threads_qv;

    // Calculate the per-dim divisors based on min/max
    let build_divisors_from_minmax_kernel =
        module.load_function(&mode.k("build_u{b}_divisors_from_minmax_kernel"))?;

    let cfg_db_dm = LaunchConfig {
        grid_dim: (blocks_db, 1, 1),
        block_dim: (threads_db, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&build_divisors_from_minmax_kernel)
            .arg(&d_db_dim_min)
            .arg(&d_db_dim_max)
            .arg(&d_s)
            .arg(&challenge.vector_dims)
            .launch(cfg_db_dm)?;
    }

    //stream.synchronize()?;

    //
    // ---------- Convert input data by packing into bits ----------
    //

    let f32_to_packed_perdim_kernel =
        module.load_function(&mode.k("f32_to_u{b}_packed_perdim_kernel"))?;

    // DB
    let threads_db: u32 = 256;
    let blocks_db: u32 = ((num_db_bytes as u32) + threads_db - 1) / threads_db;
    let cfg_db = LaunchConfig {
        grid_dim: (blocks_db, 1, 1),
        block_dim: (threads_db, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&f32_to_packed_perdim_kernel)
            .arg(&challenge.d_database_vectors) // const float* in   [num_db * D]
            .arg(&d_db_dim_min) // float* in_min
            .arg(&d_s) // const float* s    [D]
            .arg(&d_db_packed) // uint8_t* out      [num_db * ((D+1)>>1)]
            .arg(&challenge.database_size) // num_vecs
            .arg(&challenge.vector_dims) // dims
            .launch(cfg_db)?;
    }

    // Queries
    let threads_qv: u32 = 256;
    let blocks_qv: u32 = ((num_qv_bytes as u32) + threads_qv - 1) / threads_qv;
    let cfg_qv = LaunchConfig {
        grid_dim: (blocks_qv, 1, 1),
        block_dim: (threads_qv, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&f32_to_packed_perdim_kernel)
            .arg(&challenge.d_query_vectors) // const float* in   [num_query * D]
            .arg(&d_db_dim_min) // float* in_min
            .arg(&d_s)
            .arg(&d_qv_packed)
            .arg(&challenge.difficulty.num_queries)
            .arg(&challenge.vector_dims)
            .launch(cfg_qv)?;
    }

    //stream.synchronize()?;


    //
    // ---------- Compute Vector Stats ----------
    //

    let compute_vector_stats_packed_kernel =
        module.load_function(&mode.k("compute_vector_stats_u{b}_packed_kernel"))?;

    let threads_per_block_stats = prop.maxThreadsPerBlock as u32;
    let num_blocks_db =
        (challenge.database_size + threads_per_block_stats - 1) / threads_per_block_stats;

    let cfg_stats = LaunchConfig {
        grid_dim: (num_blocks_db, 1, 1),
        block_dim: (threads_per_block_stats, 1, 1),
        shared_mem_bytes: 0,
    };

    // DB norms
    unsafe {
        stream
            .launch_builder(&compute_vector_stats_packed_kernel)
            .arg(&d_db_packed) // const uint8_t* packed [num_db * ((D+1)>>1)]
            .arg(&d_db_norm_l2) // float* norm_l2        [num_db]
            .arg(&d_db_norm_l2_squared) // float* norm_l2_sq     [num_db]
            .arg(&challenge.database_size) // num_vecs
            .arg(&challenge.vector_dims) // dims
            .launch(cfg_stats)?;
    }

    // Query norms
    let num_blocks_qv =
        (challenge.difficulty.num_queries + threads_per_block_stats - 1) / threads_per_block_stats;

    let cfg_stats_qv = LaunchConfig {
        grid_dim: (num_blocks_qv, 1, 1),
        block_dim: (threads_per_block_stats, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&compute_vector_stats_packed_kernel)
            .arg(&d_qv_packed)
            .arg(&d_query_norm_l2)
            .arg(&d_query_norm_l2_squared)
            .arg(&challenge.difficulty.num_queries)
            .arg(&challenge.vector_dims)
            .launch(cfg_stats_qv)?;
    }

    //stream.synchronize()?;

    // let elapsed_time_ms_1 = start_time_total.elapsed().as_micros() as f32 / 1000.0;

    //
    // ---------- Compute MAD Stats ----------
    //

    let mut norm_threshold: f32 = f32::MAX;
    // Determine MAD scale: 0.0 means "legacy auto-MAD" (compute from num_queries)
    let scale = if hparams.mad_scale == 0.0 {
        scale_factor(challenge.difficulty.num_queries as usize)
    } else {
        hparams.mad_scale
    };
    //println!("stat_filter scale: {}", scale);

    // Only compute and apply MAD if within range
    if scale > 0.0 && scale < 5.0 {
        // MAD threshold on DB norms (unchanged logic)
        let mut h_norms = stream.memcpy_dtov(&d_db_norm_l2)?;
        h_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = h_norms.len() / 2;
        let median = if h_norms.len() % 2 == 0 {
            (h_norms[mid - 1] + h_norms[mid]) / 2.0
        } else {
            h_norms[mid]
        };

        let mut deviations: Vec<f32> = h_norms.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if deviations.len() % 2 == 0 {
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[mid]
        };

        norm_threshold = scale * mad * MAD_SCALE_NORMAL;
    }

    // let elapsed_time_ms_2 = start_time_total.elapsed().as_micros() as f32 / 1000.0;

    //
    // ---------- Search ----------
    //

    // --- TopK outputs ---
    let mut d_topk_indices =
        stream.alloc_zeros::<i32>((challenge.difficulty.num_queries as usize) * topk)?;

    // Save some memory -- we don't use this output
    //let mut d_topk_dist =
    //    stream.alloc_zeros::<f32>((challenge.difficulty.num_queries as usize) * topk)?;

    // --- Geometry ---
    let words_per_plane = ((dims + 63) >> 6) as usize; // W
    let words_per_plane_i32 = words_per_plane as i32;

    // --- Shared memory sizing for Top-K ---

    // Per-thread spill for heap:
    let per_thread_bytes = topk * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>());
    // 4 planes * W words * 8B per word
    //let base_query_bytes = 4 * words_per_plane * std::mem::size_of::<u64>();
    let base_query_bytes = mode.planes() * words_per_plane * std::mem::size_of::<u64>();

    let smem_limit = prop.sharedMemPerBlock as usize;
    let mut threads_per_block: usize = 256;
    while base_query_bytes + threads_per_block * per_thread_bytes > smem_limit
        && threads_per_block > 32
    {
        threads_per_block >>= 1;
    }
    if base_query_bytes + threads_per_block * per_thread_bytes > smem_limit {
        return Err(anyhow!(
            "Insufficient shared memory for topk={} with dims={} (need ~{}B, have {}B)",
            topk,
            challenge.vector_dims,
            base_query_bytes + threads_per_block * per_thread_bytes,
            smem_limit
        ));
    }
    let threads_per_block = threads_per_block as u32;

    let shared_mem_bytes =
        (base_query_bytes + (threads_per_block as usize) * per_thread_bytes) as u32;

    let cfg_topk = LaunchConfig {
        grid_dim: (challenge.difficulty.num_queries, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: shared_mem_bytes,
    };

    let k_i32: i32 = topk as i32;

    // --- Convert packed -> bitplanes ---
    let packed_to_bitplanes_rowwise = module.load_function(&mode.k("u{b}_packed_to_bitplanes_rowwise"))?;
    let blk_conv = (256u32, 1u32, 1u32);
    let grd_db = (((n_db as u32) + 255) / 256, 1, 1);
    let grd_q = (((n_q as u32) + 255) / 256, 1, 1);

    let find_topk_neighbors_bitsliced_kernel =
        module.load_function(&mode.k("find_topk_neighbors_u{b}_bitsliced_kernel"))?;

    // let mut elapsed_time_ms_3 = start_time_total.elapsed().as_micros() as f32 / 1000.0;

    match mode {
        BitMode::U2 => {
            let mut d_db_b0 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
            let mut d_db_b1 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
            let mut d_q_b0 = stream.alloc_zeros::<u64>(n_q * words_per_plane)?;
            let mut d_q_b1 = stream.alloc_zeros::<u64>(n_q * words_per_plane)?;
/*
            let mut d_db_b0 = alloc_uninit::<u64>(&stream, n_db * words_per_plane)?;
            let mut d_db_b1 = alloc_uninit::<u64>(&stream, n_db * words_per_plane)?;
            let mut d_q_b0  = alloc_uninit::<u64>(&stream, n_q * words_per_plane)?;
            let mut d_q_b1  = alloc_uninit::<u64>(&stream, n_q * words_per_plane)?;
*/

            unsafe {
                // DB
                stream
                    .launch_builder(&packed_to_bitplanes_rowwise)
                    .arg(&d_db_packed)
                    .arg(&mut d_db_b0)
                    .arg(&mut d_db_b1)
                    .arg(&(challenge.database_size))
                    .arg(&(challenge.vector_dims))
                    .arg(&words_per_plane_i32)
                    .launch(LaunchConfig {
                        grid_dim: grd_db,
                        block_dim: blk_conv,
                        shared_mem_bytes: 0,
                    })?;
                // Q
                stream
                    .launch_builder(&packed_to_bitplanes_rowwise)
                    .arg(&d_qv_packed)
                    .arg(&mut d_q_b0)
                    .arg(&mut d_q_b1)
                    .arg(&(challenge.difficulty.num_queries))
                    .arg(&(challenge.vector_dims))
                    .arg(&words_per_plane_i32)
                    .launch(LaunchConfig {
                        grid_dim: grd_q,
                        block_dim: blk_conv,
                        shared_mem_bytes: 0,
                    })?;
            }

            // elapsed_time_ms_3 = start_time_total.elapsed().as_micros() as f32 / 1000.0;

            // launch top-k with 2 plane args
            unsafe {
                stream
                    .launch_builder(&find_topk_neighbors_bitsliced_kernel)
                    .arg(&d_q_b0)
                    .arg(&d_q_b1)
                    .arg(&d_db_b0)
                    .arg(&d_db_b1)
                    .arg(&d_db_norm_l2)
                    .arg(&d_db_norm_l2_squared)
                    .arg(&mut d_topk_indices)
                    //.arg(&mut d_topk_dist) // Save some memory -- we don't use this output
                    .arg(&k_i32)
                    .arg(&challenge.max_distance)
                    .arg(&challenge.database_size)
                    .arg(&challenge.difficulty.num_queries)
                    .arg(&challenge.vector_dims)
                    .arg(&norm_threshold)
                    .arg(&d_query_norm_l2)
                    .arg(&d_query_norm_l2_squared)
                    .arg(&words_per_plane_i32)
                    .launch(cfg_topk)?;
            }
        }
        BitMode::U4 => {
            let mut d_db_b0 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
            let mut d_db_b1 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
            let mut d_db_b2 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
            let mut d_db_b3 = stream.alloc_zeros::<u64>(n_db * words_per_plane)?;
            let mut d_q_b0 = stream.alloc_zeros::<u64>(n_q * words_per_plane)?;
            let mut d_q_b1 = stream.alloc_zeros::<u64>(n_q * words_per_plane)?;
            let mut d_q_b2 = stream.alloc_zeros::<u64>(n_q * words_per_plane)?;
            let mut d_q_b3 = stream.alloc_zeros::<u64>(n_q * words_per_plane)?;
/*
            let mut d_db_b0 = alloc_uninit::<u64>(&stream, n_db * words_per_plane)?;
            let mut d_db_b1 = alloc_uninit::<u64>(&stream, n_db * words_per_plane)?;
            let mut d_db_b2 = alloc_uninit::<u64>(&stream, n_db * words_per_plane)?;
            let mut d_db_b3 = alloc_uninit::<u64>(&stream, n_db * words_per_plane)?;
            let mut d_q_b0  = alloc_uninit::<u64>(&stream, n_q * words_per_plane)?;
            let mut d_q_b1  = alloc_uninit::<u64>(&stream, n_q * words_per_plane)?;
            let mut d_q_b2  = alloc_uninit::<u64>(&stream, n_q * words_per_plane)?;
            let mut d_q_b3  = alloc_uninit::<u64>(&stream, n_q * words_per_plane)?;
*/

            unsafe {
                // DB
                stream
                    .launch_builder(&packed_to_bitplanes_rowwise)
                    .arg(&d_db_packed)
                    .arg(&mut d_db_b0)
                    .arg(&mut d_db_b1)
                    .arg(&mut d_db_b2)
                    .arg(&mut d_db_b3)
                    .arg(&(challenge.database_size))
                    .arg(&(challenge.vector_dims))
                    .arg(&words_per_plane_i32)
                    .launch(LaunchConfig {
                        grid_dim: grd_db,
                        block_dim: blk_conv,
                        shared_mem_bytes: 0,
                    })?;
                // Q
                stream
                    .launch_builder(&packed_to_bitplanes_rowwise)
                    .arg(&d_qv_packed)
                    .arg(&mut d_q_b0)
                    .arg(&mut d_q_b1)
                    .arg(&mut d_q_b2)
                    .arg(&mut d_q_b3)
                    .arg(&(challenge.difficulty.num_queries))
                    .arg(&(challenge.vector_dims))
                    .arg(&words_per_plane_i32)
                    .launch(LaunchConfig {
                        grid_dim: grd_q,
                        block_dim: blk_conv,
                        shared_mem_bytes: 0,
                    })?;
            }

            // elapsed_time_ms_3 = start_time_total.elapsed().as_micros() as f32 / 1000.0;

            // launch top-k with 4 plane args
            unsafe {
                stream
                    .launch_builder(&find_topk_neighbors_bitsliced_kernel)
                    .arg(&d_q_b0)
                    .arg(&d_q_b1)
                    .arg(&d_q_b2)
                    .arg(&d_q_b3)
                    .arg(&d_db_b0)
                    .arg(&d_db_b1)
                    .arg(&d_db_b2)
                    .arg(&d_db_b3)
                    .arg(&d_db_norm_l2)
                    .arg(&d_db_norm_l2_squared)
                    .arg(&mut d_topk_indices)
                    //.arg(&mut d_topk_dist) // Save some memory -- we don't use this output
                    .arg(&k_i32)
                    .arg(&challenge.max_distance)
                    .arg(&challenge.database_size)
                    .arg(&challenge.difficulty.num_queries)
                    .arg(&challenge.vector_dims)
                    .arg(&norm_threshold)
                    .arg(&d_query_norm_l2)
                    .arg(&d_query_norm_l2_squared)
                    .arg(&words_per_plane_i32)
                    .launch(cfg_topk)?;
            }
        }
    }

    // Pull back top-K indices, build Top-1 for the Solution, and compute Recall@K if provided
    let h_topk: Vec<i32> = stream.memcpy_dtov(&d_topk_indices)?;
    let mut top1 = Vec::<usize>::with_capacity(challenge.difficulty.num_queries as usize);
    for q in 0..(challenge.difficulty.num_queries as usize) {
        let base = q * topk;
        top1.push(h_topk[base] as usize); // assuming kernel writes sorted asc by distance
    }

    // let elapsed_time_ms_4 = start_time_total.elapsed().as_micros() as f32 / 1000.0;

    //
    // === Re-rank Top-K on FP32 ===
    //
    // NOTE: We only return the best match, not an array.  This is an "internal" top-k.
    //

    let refine_fn = module.load_function("refine_topk_rerank_kernel")?;

    let threads_refine: u32 = 128;
    let grid_refine = challenge.difficulty.num_queries;
    let shared_refine = (challenge.vector_dims as usize * std::mem::size_of::<f32>()
        + threads_refine as usize * std::mem::size_of::<f32>()) as u32;

    let mut d_refined_index =
        stream.alloc_zeros::<i32>(challenge.difficulty.num_queries as usize)?;
    let mut d_refined_distance =
        stream.alloc_zeros::<f32>(challenge.difficulty.num_queries as usize)?;
    let k_i32: i32 = topk as i32;

    let cfg_refine = LaunchConfig {
        grid_dim: (grid_refine, 1, 1),
        block_dim: (threads_refine, 1, 1),
        shared_mem_bytes: shared_refine,
    };

    unsafe {
        stream
            .launch_builder(&refine_fn)
            .arg(&challenge.d_query_vectors) // Original FP32 queries
            .arg(&challenge.d_database_vectors) // Original FP32 DB
            .arg(&d_topk_indices) // [num_queries * K] (i32)
            .arg(&mut d_refined_index) // OUT best index per query
            .arg(&mut d_refined_distance) // OUT best distance per query
            .arg(&challenge.difficulty.num_queries) // num_queries
            .arg(&challenge.vector_dims) // original vector dim
            .arg(&k_i32) // K
            .launch(cfg_refine)?;
    }
    //stream.synchronize()?;

    // Use refined Top-1 as the final Solution
    let top1_refined: Vec<i32> = stream.memcpy_dtov(&d_refined_index)?;
    let mut final_idxs = Vec::<usize>::with_capacity(top1_refined.len());
    for &idx in &top1_refined {
        final_idxs.push(idx as usize);
    }

    // let elapsed_time_ms = start_time_total.elapsed().as_micros() as f32 / 1000.0;

    // Internal timing statistics

    // println!("===== stat_filter bitslice {}-bit ( Top-{} ) =====", mode.bits(), topk);
    // println!(
    //     "Time for nonce: {:.3} ms (sum+stats: {:.3} ms + mad_sort: {:.3} ms + slice: {:.3} ms + search: {:.3} ms + rerank {:.3} ms)",
    //     elapsed_time_ms,
    //     elapsed_time_ms_1,
    //     elapsed_time_ms_2 - elapsed_time_ms_1,
    //     elapsed_time_ms_3 - elapsed_time_ms_2,
    //     elapsed_time_ms_4 - elapsed_time_ms_3,
    //     elapsed_time_ms - elapsed_time_ms_4
    // );

    let _ = save_solution(&Solution {
        indexes: final_idxs,
    });
    return Ok(());
}

//------------ MAD Scale Factor Adjustment -------------

fn scale_factor(num_queries: usize) -> f32 {
    match num_queries {
        q if q <= 700 => 0.20,
        q if q <= 1000 => 0.20 + (q as f32 - 700.0) * (0.10 / 300.0), // 0.30 at 1000
        q if q <= 1500 => 0.30 + (q as f32 - 1000.0) * (0.20 / 500.0), // 0.50 at 1500
        q if q <= 2000 => 0.50 + (q as f32 - 1500.0) * (0.44 / 500.0), // 0.94 at 2000
        q if q <= 2500 => 0.94 + (q as f32 - 2000.0) * (1.08 / 500.0), // 2.02 at 2500
        _ => 1.00,
    }
}

//----------------- Env Variables -------------------

// fn read_topk() -> usize {
//     env::var("STATFILT_TOP_K")
//         .ok()
//         .and_then(|s| s.trim().parse::<usize>().ok())
//         .filter(|&v| v > 0)
//         .unwrap_or(DEFAULT_TOP_K)
// }

//----------------- Alloc Not Zeroed ----------------

/*
use cudarc::driver::CudaSlice;
use cudarc::driver::DriverError;

fn alloc_uninit<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>, DriverError> {
    unsafe { stream.alloc::<T>(len) }
}
*/



