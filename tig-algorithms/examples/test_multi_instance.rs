// Large benchmark for CUR decomposition.
//
// Algorithms : fastest_algo | leverage (1t+cheap) | leverage (15t) | sketchy
// Sizes      : 6 rectangular configs (m always 2^p+1, n > m)
// Seeds      : 25 real + 1 warmup per (algo, size) pair
//
// Score      : fnorm / optimal  (1.0 = perfect, lower is better)
// Per seed   : avg_score = mean(score) over sub-instances of that seed
// Reported   : avg/min/max of per-seed avg_scores over the 25 seeds
//
// Outputs:
//   examples/large_benchmark_summary.txt  — summary table
//   examples/large_benchmark_scores.csv   — every (algo,m,n,seed,sub,k,score)
//
// Usage:
//   cargo run --release --example test_multi_instance --features cur_decomposition \
//       -- <PTX_PATH> [--seeds N] [--gpu N]

use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
    runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
};
use std::{cell::RefCell, fmt::Write as FmtWrite, fs, sync::Arc, time::Instant};
use tig_challenges::cur_decomposition::*;

#[path = "../src/cur_decomposition/leverage/mod.rs"]
mod leverage;

#[path = "../src/cur_decomposition/fastest_algo/mod.rs"]
mod fastest_algo;

#[path = "../src/cur_decomposition/sketchy/mod.rs"]
mod sketchy;

// ─── Stats ────────────────────────────────────────────────────────────────────

struct Stats {
    values: Vec<f64>,
}

impl Stats {
    fn new() -> Self { Self { values: Vec::new() } }
    fn push(&mut self, v: f64) { self.values.push(v); }
    fn is_empty(&self) -> bool { self.values.is_empty() }
    fn mean(&self) -> f64 { self.values.iter().sum::<f64>() / self.values.len() as f64 }
    fn min(&self) -> f64 { self.values.iter().cloned().fold(f64::INFINITY, f64::min) }
    fn max(&self) -> f64 { self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
}

// ─── Config & result types ────────────────────────────────────────────────────

struct AlgoConfig {
    label: &'static str,
    algo: &'static str,
    hyperparameters: Option<serde_json::Map<String, serde_json::Value>>,
}

struct SizeConfig {
    m: i32,
    n: i32,
}

struct RunResult {
    algo: String,
    m: i32,
    n: i32,
    avg_solve_ms: f64,
    avg_gen_verify_ms: f64,
    avg_score: f64,       // mean of per-seed avg_scores
    min_seed_score: f64,  // min of per-seed avg_scores
    max_seed_score: f64,  // max of per-seed avg_scores
}

struct ScoreRecord {
    algo: String,
    m: i32,
    n: i32,
    seed: usize,
    sub_idx: usize,
    target_k: i32,
    score: f64,
}

// ─── Seed helper ──────────────────────────────────────────────────────────────

fn make_seed(index: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&index.to_le_bytes());
    seed
}

// ─── Solve dispatch ───────────────────────────────────────────────────────────

fn solve(
    algo: &str,
    challenge: &Challenge,
    save_fn: &dyn Fn(&Solution) -> Result<()>,
    hp: &Option<serde_json::Map<String, serde_json::Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Option<Solution>> {
    match algo {
        "fastest_algo" => fastest_algo::solve_challenge(challenge, save_fn, hp, module, stream, prop),
        "sketchy"      => sketchy::solve_challenge(challenge, save_fn, hp, module, stream, prop),
        _              => leverage::solve_challenge(challenge, save_fn, hp, module, stream, prop),
    }
}

// ─── Per (algo, size) benchmark ───────────────────────────────────────────────

fn run_algo(
    cfg: &AlgoConfig,
    size: &SizeConfig,
    num_seeds: usize,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<(RunResult, Vec<ScoreRecord>)> {
    let m = size.m;
    let n = size.n;
    let track = Track { m, n };

    let mut all_gen_ms    = Stats::new();
    let mut all_solve_ms  = Stats::new();
    let mut all_verify_ms = Stats::new();
    let mut seed_avgs     = Stats::new(); // one entry per real seed
    let mut records: Vec<ScoreRecord> = Vec::new();

    println!(
        "\n╔══ {} │ M={} N={} │ {} seeds + 1 warmup ══",
        cfg.label, m, n, num_seeds
    );
    println!(
        "║ {:>5}  {:>9}  {:>9}  {:>9}  {:>10}  {:>8}  {:>8}",
        "seed", "gen_ms", "solve_ms", "vrfy_ms", "avg_score", "min", "max"
    );
    println!("╟{}", "─".repeat(74));

    // seed_idx == 0  →  warmup (distinct seed, results discarded)
    // seed_idx  > 0  →  real seeds 0 .. num_seeds-1
    for seed_idx in 0..=(num_seeds) {
        let is_warmup = seed_idx == 0;
        let si = if is_warmup { num_seeds } else { seed_idx - 1 };
        let seed = make_seed(si as u64);

        // ── Generate sub-instances ────────────────────────────────────────────
        let t_gen = Instant::now();
        let challenges = match Challenge::generate_multiple_instances(
            &seed, &track, module.clone(), stream.clone(), prop,
        ) {
            Ok(cs) => cs,
            Err(e) => {
                let lbl = if is_warmup { "warm".to_string() } else { si.to_string() };
                println!("║ {:>5}  generate error: {}", lbl, e);
                continue;
            }
        };
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

        // ── Solve + verify each sub-instance ──────────────────────────────────
        let mut total_solve_ms  = 0.0f64;
        let mut total_verify_ms = 0.0f64;
        let mut sub_scores: Vec<f64> = Vec::new();

        for (sub_idx, challenge) in challenges.iter().enumerate() {
            let optimal = challenge.optimal_fnorm() as f64;
            let best: RefCell<Option<Solution>> = RefCell::new(None);

            let t_solve = Instant::now();
            let result = {
                let save_fn = |sol: &Solution| -> Result<()> {
                    *best.borrow_mut() = Some(sol.clone());
                    Ok(())
                };
                solve(cfg.algo, challenge, &save_fn, &cfg.hyperparameters,
                      module.clone(), stream.clone(), prop)
            };
            total_solve_ms += t_solve.elapsed().as_secs_f64() * 1000.0;

            let solution = match result {
                Ok(sol) => sol.or_else(|| best.into_inner()),
                Err(e) => {
                    eprintln!("  solve error (k={}): {}", challenge.target_k, e);
                    None
                }
            };

            if let Some(sol) = solution {
                let t_verify = Instant::now();
                match challenge.evaluate_fnorm(&sol, module.clone(), stream.clone(), prop) {
                    Ok(fnorm) => {
                        total_verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
                        let score = fnorm as f64 / optimal;
                        sub_scores.push(score);
                        if !is_warmup {
                            records.push(ScoreRecord {
                                algo: cfg.label.to_string(),
                                m, n,
                                seed: si,
                                sub_idx,
                                target_k: challenge.target_k,
                                score,
                            });
                        }
                    }
                    Err(e) => eprintln!("  verify error (k={}): {}", challenge.target_k, e),
                }
            }
        }

        let (seed_avg, seed_min, seed_max) = if sub_scores.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let mean = sub_scores.iter().sum::<f64>() / sub_scores.len() as f64;
            let min  = sub_scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let max  = sub_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mean, min, max)
        };

        let lbl = if is_warmup { "warm".to_string() } else { si.to_string() };
        println!(
            "║ {:>5}  {:>9.1}  {:>9.1}  {:>9.1}  {:>10.4}  {:>8.4}  {:>8.4}{}",
            lbl, gen_ms, total_solve_ms, total_verify_ms,
            seed_avg, seed_min, seed_max,
            if is_warmup { "  (warmup)" } else { "" }
        );

        if is_warmup { continue; }

        all_gen_ms.push(gen_ms);
        all_solve_ms.push(total_solve_ms);
        all_verify_ms.push(total_verify_ms);
        if seed_avg.is_finite() { seed_avgs.push(seed_avg); }
    }

    println!("╟{}", "─".repeat(74));
    let avg_solve      = if all_solve_ms.is_empty() { f64::NAN } else { all_solve_ms.mean() };
    let avg_gen_verify = if all_gen_ms.is_empty() { f64::NAN }
                         else { all_gen_ms.mean() + all_verify_ms.mean() };
    let (avg_score, min_seed_score, max_seed_score) = if seed_avgs.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (seed_avgs.mean(), seed_avgs.min(), seed_avgs.max())
    };
    println!(
        "║ avg    {:>9.1}  {:>9.1}  {:>9.1}  {:>10.4}  {:>8.4}  {:>8.4}",
        all_gen_ms.mean(), avg_solve, all_verify_ms.mean(),
        avg_score, min_seed_score, max_seed_score
    );
    println!("╚{}", "═".repeat(74));

    Ok((RunResult { algo: cfg.label.to_string(), m, n,
                    avg_solve_ms: avg_solve, avg_gen_verify_ms: avg_gen_verify,
                    avg_score, min_seed_score, max_seed_score },
        records))
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <PTX_PATH> [--seeds N] [--gpu N]", args[0]);
        std::process::exit(1);
    }

    let ptx_path    = &args[1];
    let mut num_seeds  = 25usize;
    let mut gpu_device = 0usize;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => { i += 1; num_seeds  = args[i].parse()?; }
            "--gpu"   => { i += 1; gpu_device = args[i].parse()?; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
        i += 1;
    }

    // ── CUDA setup ─────────────────────────────────────────────────────────────
    let ptx_src = std::fs::read_to_string(ptx_path)
        .map_err(|e| anyhow!("Failed to read PTX '{}': {}", ptx_path, e))?;
    let ptx_src = ptx_src.replace("0xdeadbeefdeadbeef", "0xffffffffffffffff");
    let ptx = Ptx::from_src(ptx_src);

    let num_gpus = CudaContext::device_count()?;
    if num_gpus == 0 { return Err(anyhow!("No CUDA devices found")); }

    let ctx    = CudaContext::new(gpu_device)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(ptx)?;
    let stream = ctx.default_stream();
    let prop   = get_device_prop(gpu_device as i32)?;

    // ── Algorithm configs ──────────────────────────────────────────────────────
    let hp_1t_cheap = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(1));
        m.insert("cheap_u".into(),    serde_json::json!(true));
        m
    };
    let hp_15t = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(15));
        m
    };

    let algos: Vec<AlgoConfig> = vec![
        AlgoConfig { label: "fastest_algo",       algo: "fastest_algo", hyperparameters: None },
        AlgoConfig { label: "leverage (1t+cheap)", algo: "leverage",     hyperparameters: Some(hp_1t_cheap) },
        AlgoConfig { label: "leverage (15t)",      algo: "leverage",     hyperparameters: Some(hp_15t) },
        AlgoConfig { label: "sketchy",             algo: "sketchy",      hyperparameters: None },
    ];

    // ── Size configs ───────────────────────────────────────────────────────────
    let sizes: Vec<SizeConfig> = vec![
        SizeConfig { m: 1025, n: 1500 },
        SizeConfig { m: 1025, n: 2000 },
        SizeConfig { m: 2049, n: 2150 },
        SizeConfig { m: 2049, n: 3000 },
        SizeConfig { m: 4097, n: 4150 },
        SizeConfig { m: 4097, n: 5000 },
    ];

    println!("=== CUR Decomposition Large Benchmark ===");
    println!("PTX    : {}", ptx_path);
    println!("GPU    : device {} of {}", gpu_device, num_gpus);
    println!("Seeds  : {} real + 1 warmup per (algo, size)", num_seeds);
    println!("Score  : fnorm / optimal  (1.0 = perfect, lower is better)");
    println!("Algos  : {}", algos.len());
    println!("Sizes  : {}", sizes.len());

    // ── Run all (algo, size) combos ────────────────────────────────────────────
    let mut all_results: Vec<RunResult> = Vec::new();
    let mut all_records: Vec<ScoreRecord> = Vec::new();

    for algo in &algos {
        for size in &sizes {
            let (result, records) =
                run_algo(algo, size, num_seeds, module.clone(), stream.clone(), &prop)?;
            all_results.push(result);
            all_records.extend(records);
        }
    }

    // ── Summary table ──────────────────────────────────────────────────────────
    let mut summary = String::new();
    writeln!(summary, "CUR Decomposition Large Benchmark Results").unwrap();
    writeln!(summary, "PTX   : {}", ptx_path).unwrap();
    writeln!(summary, "Seeds : {} real + 1 warmup per (algo, size)", num_seeds).unwrap();
    writeln!(summary, "Score : fnorm / optimal  (1.0 = perfect, lower is better)").unwrap();
    writeln!(summary).unwrap();
    writeln!(summary,
        "{:<24} {:>6} {:>6} {:>10} {:>14} {:>10} {:>12} {:>12}",
        "algorithm", "m", "n", "solve_ms", "gen+verify_ms",
        "avg_score", "min_seed_sc", "max_seed_sc"
    ).unwrap();
    writeln!(summary, "{}", "─".repeat(100)).unwrap();

    let mut prev_algo = String::new();
    for r in &all_results {
        if r.algo != prev_algo && !prev_algo.is_empty() {
            writeln!(summary).unwrap();
        }
        writeln!(summary,
            "{:<24} {:>6} {:>6} {:>10.1} {:>14.1} {:>10.4} {:>12.4} {:>12.4}",
            r.algo, r.m, r.n,
            r.avg_solve_ms, r.avg_gen_verify_ms,
            r.avg_score, r.min_seed_score, r.max_seed_score
        ).unwrap();
        prev_algo = r.algo.clone();
    }

    println!("\n{}", summary);

    let manifest = env!("CARGO_MANIFEST_DIR");
    let summary_path = format!("{}/examples/large_benchmark_summary.txt", manifest);
    fs::write(&summary_path, &summary)
        .map_err(|e| anyhow!("Failed to write summary: {}", e))?;
    println!("Summary saved to {}", summary_path);

    // ── Detailed scores CSV ────────────────────────────────────────────────────
    let mut csv = String::new();
    writeln!(csv, "algo,m,n,seed,sub_idx,target_k,score").unwrap();
    for rec in &all_records {
        writeln!(csv, "{},{},{},{},{},{},{:.6}",
            rec.algo, rec.m, rec.n, rec.seed, rec.sub_idx, rec.target_k, rec.score
        ).unwrap();
    }

    let scores_path = format!("{}/examples/large_benchmark_scores.csv", manifest);
    fs::write(&scores_path, &csv)
        .map_err(|e| anyhow!("Failed to write scores CSV: {}", e))?;
    println!("Detailed scores saved to {}", scores_path);

    Ok(())
}
