//! Paired benchmark on the official eight-subinstance generator.
//! cargo run --release -p tig-algorithms --features cur_decomposition \
//!   --example good_cur_compare -- <good_cur_alg.ptx> 'm=2000,n=3000,poly=false' 3
use anyhow::{anyhow, Result};
use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};
use std::{cell::RefCell, fs, time::Instant};
use tig_algorithms::cur_decomposition::good_cur_alg;
use tig_challenges::cur_decomposition::*;

#[path = "../src/cur_decomposition/sketchy/mod.rs"]
mod sketchy;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        return Err(anyhow!(
            "Usage: good_cur_compare <PTX> <m=2000,n=3000,poly=false> [seeds=3]"
        ));
    }
    let track: Track = serde_json::from_value(serde_json::Value::String(args[2].clone()))?;
    let count = args
        .get(3)
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(3);
    if count == 0 {
        return Err(anyhow!("seeds must be positive"));
    }
    let ptx = fs::read_to_string(&args[1])?.replace("0xdeadbeefdeadbeef", "0xffffffffffffffff");
    let context = CudaContext::new(0)?;
    context.set_blocking_synchronize()?;
    let module = context.load_module(Ptx::from_src(ptx))?;
    let stream = context.default_stream();
    let prop = get_device_prop(0)?;
    let config = DesignGenerationConfig {
        m: track.m,
        n: track.n,
        poly: track.poly,
        delta: DESIGN_DELTA,
        spectrum_a: DESIGN_SPECTRUM_A,
    };
    println!("seed,sub,k,sketchy_error,good_error,sketchy_score,good_score,sketchy_ms,good_ms");
    for seed_index in 0..count {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&seed_index.to_le_bytes());
        seed[8..16].copy_from_slice(
            &seed_index
                .wrapping_add(1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .to_le_bytes(),
        );
        let instance = Challenge::generate_design_instance(
            &seed,
            &config,
            module.clone(),
            stream.clone(),
            &prop,
        )?;
        let mut scores = [Vec::new(), Vec::new()];
        for sub in &instance.sub_instances {
            let mut results = Vec::new();
            for use_good in [false, true] {
                let saved = RefCell::new(None);
                let history = RefCell::new(Vec::new());
                let save = |solution: &Solution| -> Result<()> {
                    history.borrow_mut().push(solution.clone());
                    *saved.borrow_mut() = Some(solution.clone());
                    Ok(())
                };
                stream.synchronize()?;
                let start = Instant::now();
                let returned = if use_good {
                    good_cur_alg::solve_challenge(
                        &sub.challenge,
                        &save,
                        &None,
                        module.clone(),
                        stream.clone(),
                        &prop,
                    )?;
                    None
                } else {
                    sketchy::solve_challenge(
                        &sub.challenge,
                        &save,
                        &None,
                        module.clone(),
                        stream.clone(),
                        &prop,
                    )?
                };
                let solution = returned
                    .or_else(|| saved.into_inner())
                    .ok_or_else(|| anyhow!("no solution"))?;
                stream.synchronize()?;
                let ms = start.elapsed().as_secs_f64() * 1000.0;
                let error = sub.challenge.evaluate_fast_fnorm(
                    &solution.c_idxs,
                    &solution.r_idxs,
                    module.clone(),
                    stream.clone(),
                    &prop,
                )?;
                let score = score_from_errors(
                    error as f64,
                    sub.challenge.optimal_fnorm() as f64,
                    sub.challenge.target_k,
                )?;
                let mut previous = f32::INFINITY;
                for candidate in history.into_inner() {
                    let next = sub.challenge.evaluate_fast_fnorm(
                        &candidate.c_idxs,
                        &candidate.r_idxs,
                        module.clone(),
                        stream.clone(),
                        &prop,
                    )?;
                    if !next.is_finite() || next > previous * (1.0 + 1e-5) {
                        return Err(anyhow!(
                            "saved candidate regressed or had a non-finite residual"
                        ));
                    }
                    previous = next;
                }
                results.push((error, score, ms));
            }
            let (old, new) = (results[0], results[1]);
            if new.0 > old.0 * (1.0 + 1e-5) {
                return Err(anyhow!("good_cur_alg regressed against sketchy"));
            }
            println!(
                "{},{},{},{:.9},{:.9},{:.9},{:.9},{:.3},{:.3}",
                seed_index,
                sub.metadata.sub_idx,
                sub.challenge.target_k,
                old.0,
                new.0,
                old.1,
                new.1,
                old.2,
                new.2
            );
            scores[0].push(old.1);
            scores[1].push(new.1);
        }
        eprintln!(
            "seed {seed_index}: sketchy quality={}, good_cur_alg quality={}",
            aggregate_sub_scores(&scores[0])?,
            aggregate_sub_scores(&scores[1])?
        );
    }
    Ok(())
}
