//! Seed-selectable audit benchmark for a polynomial CUR track.
//!
//! It measures the complete eight-subinstance generator, the high-quality
//! block-Krylov/max-volume solver, protocol-equivalent verification, exact JSON
//! solution size, and every continuous CUR sub-score.

use anyhow::{anyhow, Context, Result};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
    runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
};
use serde::Serialize;
use std::{cell::RefCell, fs, path::PathBuf, sync::Arc, time::Instant};
use tig_challenges::cur_decomposition::*;

#[path = "../src/cur_decomposition/sketchy_v2/mod.rs"]
mod high_quality;

const DEFAULT_M: i32 = 8000;
const DEFAULT_N: i32 = 8000;

#[derive(Serialize)]
struct GenerationReport {
    measured_wall_ms: f64,
    internal_wall_ms: f64,
    gaussian_u_ms: f64,
    qr_u_ms: f64,
    gaussian_v_ms: f64,
    qr_v_ms: f64,
    basis_extract_and_scale_ms: f64,
    matrix_multiply_ms: f64,
    qr_plus_matrix_multiply_ms: f64,
}

#[derive(Serialize)]
struct SubInstanceReport {
    sub_idx: usize,
    true_rank: i32,
    target_k: i32,
    k_over_true_rank: f64,
    solve_ms: f64,
    verification_ms: f64,
    raw_index_bytes: usize,
    serialized_solution_bytes: usize,
    cur_fnorm: f64,
    optimal_svd_fnorm: f64,
    error_ratio: f64,
    score: f64,
}

#[derive(Serialize)]
struct BenchmarkReport {
    algorithm: &'static str,
    algorithm_parameters: high_quality::Hyperparameters,
    gpu: String,
    m: i32,
    n: i32,
    poly: bool,
    seed_index: u64,
    seed_hex: String,
    generation: GenerationReport,
    solve_total_ms: f64,
    verification_total_ms: f64,
    measured_total_ms: f64,
    raw_index_bytes: usize,
    raw_solution_payload_bytes: usize,
    serialized_solution_bytes: usize,
    average_score: f64,
    protocol_quality: i32,
    sub_instances: Vec<SubInstanceReport>,
}

fn make_seed(index: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&index.to_le_bytes());
    seed[8..16].copy_from_slice(&index.wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());
    seed[16..24].copy_from_slice(&index.wrapping_add(0xD1B5_4A32_D192_ED03).to_le_bytes());
    seed[24..32].copy_from_slice(&index.wrapping_mul(0x94D0_49BB_1331_11EB).to_le_bytes());
    seed
}

fn seed_hex(seed: &[u8; 32]) -> String {
    seed.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn gpu_name(prop: &cudaDeviceProp) -> String {
    let bytes: Vec<u8> = prop
        .name
        .iter()
        .copied()
        .take_while(|character| *character != 0)
        .map(|character| character as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn load_cuda(ptx_path: &str) -> Result<(Arc<CudaModule>, Arc<CudaStream>, cudaDeviceProp)> {
    let ptx_source = fs::read_to_string(ptx_path)
        .with_context(|| format!("failed to read PTX from {ptx_path}"))?
        .replace("0xdeadbeefdeadbeef", "0xffffffffffffffff");
    if CudaContext::device_count()? == 0 {
        return Err(anyhow!("no CUDA device found"));
    }
    let context = CudaContext::new(0)?;
    context.set_blocking_synchronize()?;
    let module = context.load_module(Ptx::from_src(ptx_source))?;
    let stream = context.default_stream();
    let prop = get_device_prop(0)?;
    Ok((module, stream, prop))
}

fn design_config(m: i32, n: i32) -> DesignGenerationConfig {
    DesignGenerationConfig {
        m,
        n,
        delta: DESIGN_DELTA,
        poly: true,
        spectrum_a: DESIGN_SPECTRUM_A,
    }
}

fn solve_one(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Solution> {
    let saved = RefCell::new(None);
    let save = |solution: &Solution| -> Result<()> {
        *saved.borrow_mut() = Some(solution.clone());
        Ok(())
    };
    high_quality::solve_challenge(challenge, &save, &None, module, stream, prop)?
        .or_else(|| saved.into_inner())
        .ok_or_else(|| anyhow!("high-quality solver produced no solution"))
}

/// Warm CUDA libraries, PTX kernels, and the canonical fast-U path without
/// contaminating the measured 8000x8000 nonce.
fn warm_up(module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp) -> Result<()> {
    let warm_seed = [0xA5u8; 32];
    let warm = Challenge::generate_design_instance(
        &warm_seed,
        &design_config(384, 384),
        module.clone(),
        stream.clone(),
        prop,
    )?;
    for sub in &warm.sub_instances {
        let solution = solve_one(&sub.challenge, module.clone(), stream.clone(), prop)?;
        sub.challenge.evaluate_fast_fnorm(
            &solution.c_idxs,
            &solution.r_idxs,
            module.clone(),
            stream.clone(),
            prop,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 2
        || arguments.len() > 6
        || arguments.iter().any(|argument| argument == "--help")
    {
        eprintln!(
            "Usage: {} <PTX_PATH> [OUTPUT_JSON] [M] [N] [SEED_INDEX]",
            arguments[0]
        );
        std::process::exit((arguments.len() < 2) as i32);
    }
    let m = arguments
        .get(3)
        .map(|value| value.parse::<i32>().context("M must be a positive integer"))
        .transpose()?
        .unwrap_or(DEFAULT_M);
    let n = arguments
        .get(4)
        .map(|value| value.parse::<i32>().context("N must be a positive integer"))
        .transpose()?
        .unwrap_or(DEFAULT_N);
    if m < 2 || n < 2 {
        return Err(anyhow!("matrix dimensions must both be at least 2"));
    }
    let seed_index = arguments
        .get(5)
        .map(|value| {
            value
                .parse::<u64>()
                .context("SEED_INDEX must be a non-negative integer")
        })
        .transpose()?
        .unwrap_or(0);
    let output_path = arguments
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("cur_quality_{m}x{n}_report.json")));
    let (module, stream, prop) = load_cuda(&arguments[1])?;
    println!("GPU: {}", gpu_name(&prop));
    println!("Warming CUDA and canonical fast-U verification on 384x384...");
    warm_up(module.clone(), stream.clone(), &prop)?;

    let seed = make_seed(seed_index);
    stream.synchronize()?;
    let instance_started = Instant::now();
    let generation_started = Instant::now();
    let design = Challenge::generate_design_instance(
        &seed,
        &design_config(m, n),
        module.clone(),
        stream.clone(),
        &prop,
    )?;
    stream.synchronize()?;
    let measured_generation_ms = generation_started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Generated eight {}x{} subinstances in {:.3} ms",
        m, n, measured_generation_ms
    );

    let mut solutions = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    let mut solve_times = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    stream.synchronize()?;
    let solve_started = Instant::now();
    for sub in &design.sub_instances {
        stream.synchronize()?;
        let sub_started = Instant::now();
        let solution = solve_one(&sub.challenge, module.clone(), stream.clone(), &prop)
            .with_context(|| format!("solver failed at subinstance {}", sub.metadata.sub_idx))?;
        stream.synchronize()?;
        let solve_ms = sub_started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Solved sub {}: rank={} k={} in {:.3} ms",
            sub.metadata.sub_idx, sub.metadata.true_rank, sub.challenge.target_k, solve_ms
        );
        solve_times.push(solve_ms);
        solutions.push(solution);
    }
    stream.synchronize()?;
    let solve_total_ms = solve_started.elapsed().as_secs_f64() * 1000.0;

    let serialized_solution = serde_json::to_vec(&solutions)?;
    let raw_index_bytes = solutions
        .iter()
        .map(|solution| (solution.c_idxs.len() + solution.r_idxs.len()) * size_of::<i32>())
        .sum::<usize>();
    let mut sub_reports = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    let mut scores = Vec::with_capacity(DESIGN_NUM_SUB_INSTANCES);
    stream.synchronize()?;
    let verification_started = Instant::now();
    for ((sub, solution), &solve_ms) in design
        .sub_instances
        .iter()
        .zip(&solutions)
        .zip(&solve_times)
    {
        stream.synchronize()?;
        let sub_verify_started = Instant::now();
        let fnorm = sub.challenge.evaluate_fast_fnorm(
            &solution.c_idxs,
            &solution.r_idxs,
            module.clone(),
            stream.clone(),
            &prop,
        )?;
        let optimal = sub.challenge.optimal_fnorm();
        let score = score_from_errors(fnorm as f64, optimal as f64, sub.challenge.target_k)?;
        stream.synchronize()?;
        let verification_ms = sub_verify_started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "Verified sub {} in {:.3} ms: ratio={:.6} score={:.9}",
            sub.metadata.sub_idx,
            verification_ms,
            fnorm as f64 / optimal as f64,
            score
        );
        scores.push(score);
        sub_reports.push(SubInstanceReport {
            sub_idx: sub.metadata.sub_idx,
            true_rank: sub.metadata.true_rank,
            target_k: sub.challenge.target_k,
            k_over_true_rank: sub.challenge.target_k as f64 / sub.metadata.true_rank as f64,
            solve_ms,
            verification_ms,
            raw_index_bytes: (solution.c_idxs.len() + solution.r_idxs.len()) * size_of::<i32>(),
            serialized_solution_bytes: serde_json::to_vec(solution)?.len(),
            cur_fnorm: fnorm as f64,
            optimal_svd_fnorm: optimal as f64,
            error_ratio: fnorm as f64 / optimal as f64,
            score,
        });
    }
    stream.synchronize()?;
    let verification_total_ms = verification_started.elapsed().as_secs_f64() * 1000.0;
    let measured_total_ms = instance_started.elapsed().as_secs_f64() * 1000.0;
    let average_score = scores.iter().sum::<f64>() / scores.len() as f64;
    let generation = &design.generation;
    let report = BenchmarkReport {
        algorithm: "sketchy_v2_quality_portfolio_fast_u",
        algorithm_parameters: high_quality::Hyperparameters::default(),
        gpu: gpu_name(&prop),
        m,
        n,
        poly: true,
        seed_index,
        seed_hex: seed_hex(&seed),
        generation: GenerationReport {
            measured_wall_ms: measured_generation_ms,
            internal_wall_ms: generation.wall_ms,
            gaussian_u_ms: generation.gaussian_u_ms,
            qr_u_ms: generation.qr_u_ms,
            gaussian_v_ms: generation.gaussian_v_ms,
            qr_v_ms: generation.qr_v_ms,
            basis_extract_and_scale_ms: generation
                .sub_instances
                .iter()
                .map(|timing| timing.basis_extract_and_scale_ms)
                .sum(),
            matrix_multiply_ms: generation
                .sub_instances
                .iter()
                .map(|timing| timing.matrix_multiply_ms)
                .sum(),
            qr_plus_matrix_multiply_ms: generation.qr_plus_matrix_multiply_ms,
        },
        solve_total_ms,
        verification_total_ms,
        measured_total_ms,
        raw_index_bytes,
        raw_solution_payload_bytes: raw_index_bytes,
        serialized_solution_bytes: serialized_solution.len(),
        average_score,
        protocol_quality: aggregate_sub_scores(&scores)?,
        sub_instances: sub_reports,
    };

    let report_json = serde_json::to_vec_pretty(&report)?;
    fs::write(&output_path, report_json)?;
    println!("\n=== TOTALS ===");
    println!("generation_ms={:.3}", measured_generation_ms);
    println!("solve_ms={:.3}", solve_total_ms);
    println!("verification_ms={:.3}", verification_total_ms);
    println!("measured_total_ms={:.3}", measured_total_ms);
    println!("solution_json_bytes={}", serialized_solution.len());
    println!("raw_solution_payload_bytes={}", raw_index_bytes);
    println!("average_score={:.9}", average_score);
    println!("protocol_quality={}", aggregate_sub_scores(&scores)?);
    println!("report={}", output_path.display());
    Ok(())
}
