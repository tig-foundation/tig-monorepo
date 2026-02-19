// Benchmark for CUR decomposition: generate_multiple_instances + solve + verify.
//
// Runs a fixed set of algorithm configurations over N seeds (plus 1 warmup seed
// whose results are discarded). Writes a summary table to stdout and to
// examples/benchmark_results.txt.
//
// Score = fnorm / optimal  (1.0 = perfect, lower is better)
//
// Usage:
//   cargo run --release --example test_multi_instance --features cur_decomposition \
//       -- <PTX_PATH> [--m N] [--n N] [--seeds N] [--gpu N]

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

// ─── Stats ───────────────────────────────────────────────────────────────────

struct Stats {
    values: Vec<f64>,
}

impl Stats {
    fn new() -> Self { Self { values: Vec::new() } }
    fn push(&mut self, v: f64) { self.values.push(v); }
    fn is_empty(&self) -> bool { self.values.is_empty() }
    fn mean(&self) -> f64 { self.values.iter().sum::<f64>() / self.values.len() as f64 }
    fn variance(&self) -> f64 {
        let m = self.mean();
        self.values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.values.len() as f64
    }
    fn min(&self) -> f64 { self.values.iter().cloned().fold(f64::INFINITY, f64::min) }
    fn max(&self) -> f64 { self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
}

// ─── Config & result types ────────────────────────────────────────────────────

struct RunConfig {
    label: &'static str,
    algo: &'static str,
    hyperparameters: Option<serde_json::Map<String, serde_json::Value>>,
}

struct BenchmarkResult {
    label: String,
    avg_solve_ms: f64,
    avg_gen_verify_ms: f64,
    mean_score: f64,
    var_score: f64,
    min_score: f64,
    max_score: f64,
}

// ─── Seed helper ─────────────────────────────────────────────────────────────

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
        _              => leverage::solve_challenge(challenge, save_fn, hp, module, stream, prop),
    }
}

// ─── Per-algorithm benchmark ──────────────────────────────────────────────────

fn run_algo(
    cfg: &RunConfig,
    m: i32,
    n: i32,
    num_seeds: usize,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<BenchmarkResult> {
    let track = Track { m, n };

    let mut all_solve_ms  = Stats::new();
    let mut all_gen_ms    = Stats::new();
    let mut all_verify_ms = Stats::new();
    let mut all_scores    = Stats::new(); // fnorm/optimal across ALL instances × seeds

    println!(
        "\n╔══ {} │ M={} N={} │ {} seeds + 1 warmup ══",
        cfg.label, m, n, num_seeds
    );
    println!(
        "║ {:>5}  {:>9}  {:>9}  {:>9}  {:>10}  {:>8}  {:>8}",
        "seed", "gen_ms", "solve_ms", "vrfy_ms", "mean_score", "min", "max"
    );
    println!("╟{}", "─".repeat(74));

    // seed_idx == 0  →  warmup (results discarded)
    // seed_idx  > 0  →  real seeds 0 .. num_seeds-1
    for seed_idx in 0..=(num_seeds) {
        let is_warmup = seed_idx == 0;
        // warmup uses a distinct seed (num_seeds itself) to avoid influencing real seeds
        let si = if is_warmup { num_seeds } else { seed_idx - 1 };
        let seed = make_seed(si as u64);

        // ── Generate 13 sub-instances ────────────────────────────────────────
        let t_gen = Instant::now();
        let challenges = match Challenge::generate_multiple_instances(
            &seed, &track, module.clone(), stream.clone(), prop,
        ) {
            Ok(cs) => cs,
            Err(e) => {
                let label = if is_warmup { "warm".to_string() } else { si.to_string() };
                println!("║ {:>5}  generate error: {}", label, e);
                continue;
            }
        };
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

        // ── Solve + verify each sub-instance ────────────────────────────────
        let mut seed_solve_ms  = 0.0f64;
        let mut seed_verify_ms = 0.0f64;
        let mut seed_scores: Vec<f64> = Vec::new();

        for challenge in &challenges {
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
            seed_solve_ms += t_solve.elapsed().as_secs_f64() * 1000.0;

            let solution = match result {
                Ok(sol) => sol.or_else(|| best.into_inner()),
                Err(e) => { eprintln!("  solve error (k={}): {}", challenge.target_k, e); continue; }
            };

            if let Some(sol) = solution {
                let t_verify = Instant::now();
                let fnorm = match challenge.evaluate_fnorm(&sol, module.clone(), stream.clone(), prop) {
                    Ok(f) => f,
                    Err(e) => { eprintln!("  verify error (k={}): {}", challenge.target_k, e); continue; }
                };
                seed_verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;

                seed_scores.push(fnorm as f64 / optimal);
            }
        }

        let (seed_mean, seed_min, seed_max) = if seed_scores.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let mean = seed_scores.iter().sum::<f64>() / seed_scores.len() as f64;
            let min  = seed_scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let max  = seed_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mean, min, max)
        };

        let label = if is_warmup { "warm".to_string() } else { si.to_string() };
        println!(
            "║ {:>5}  {:>9.1}  {:>9.1}  {:>9.1}  {:>10.4}  {:>8.4}  {:>8.4}{}",
            label, gen_ms, seed_solve_ms, seed_verify_ms,
            seed_mean, seed_min, seed_max,
            if is_warmup { "  (warmup)" } else { "" }
        );

        if is_warmup { continue; }

        all_gen_ms.push(gen_ms);
        all_solve_ms.push(seed_solve_ms);
        all_verify_ms.push(seed_verify_ms);
        for &score in &seed_scores {
            all_scores.push(score);
        }
    }

    println!("╟{}", "─".repeat(74));
    let avg_gen_verify = if all_gen_ms.is_empty() { f64::NAN }
                         else { all_gen_ms.mean() + all_verify_ms.mean() };
    let avg_solve = if all_solve_ms.is_empty() { f64::NAN } else { all_solve_ms.mean() };
    let (mean_score, var_score, min_score, max_score) = if all_scores.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        (all_scores.mean(), all_scores.variance(), all_scores.min(), all_scores.max())
    };

    println!("║ avg    {:>9.1}  {:>9.1}  {:>9.1}  {:>10.4}  {:>8.4}  {:>8.4}  var={:.6}",
        all_gen_ms.mean(), avg_solve, all_verify_ms.mean(),
        mean_score, min_score, max_score, var_score);
    println!("╚{}", "═".repeat(74));

    Ok(BenchmarkResult {
        label: cfg.label.to_string(),
        avg_solve_ms: avg_solve,
        avg_gen_verify_ms: avg_gen_verify,
        mean_score,
        var_score,
        min_score,
        max_score,
    })
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <PTX_PATH> [--m N] [--n N] [--seeds N] [--gpu N]", args[0]);
        std::process::exit(1);
    }

    let ptx_path    = &args[1];
    let mut m: i32        = 1025;
    let mut n: i32        = 1025;
    let mut num_seeds     = 20usize;
    let mut gpu_device    = 0usize;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--m"     => { i += 1; m          = args[i].parse()?; }
            "--n"     => { i += 1; n          = args[i].parse()?; }
            "--seeds" => { i += 1; num_seeds  = args[i].parse()?; }
            "--gpu"   => { i += 1; gpu_device = args[i].parse()?; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
        i += 1;
    }

    // ── CUDA setup ────────────────────────────────────────────────────────
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

    // ── Algorithm configurations ──────────────────────────────────────────
    let hp_1t = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(1));
        m
    };
    let hp_1t_cheap = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(1));
        m.insert("cheap_u".into(), serde_json::json!(true));
        m
    };
    let hp_12t = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(12));
        m
    };

    let configs: Vec<RunConfig> = vec![
        RunConfig { label: "leverage (1t)",       algo: "leverage",      hyperparameters: Some(hp_1t) },
        RunConfig { label: "leverage (1t+cheap)", algo: "leverage",      hyperparameters: Some(hp_1t_cheap) },
        RunConfig { label: "leverage (12t)",      algo: "leverage",      hyperparameters: Some(hp_12t) },
        RunConfig { label: "fastest_algo",        algo: "fastest_algo",  hyperparameters: None },
    ];

    println!("=== CUR Decomposition Multi-Instance Benchmark ===");
    println!("PTX    : {}", ptx_path);
    println!("GPU    : device {} of {}", gpu_device, num_gpus);
    println!("Matrix : {}×{}  (max_rank={})", m, n, m.min(n));
    println!("Seeds  : {} real + 1 warmup per algorithm", num_seeds);
    println!("Score  : fnorm / optimal  (1.0 = perfect, lower is better)");
    println!("Configs: {}", configs.len());

    // ── Run all configs ───────────────────────────────────────────────────
    let mut results: Vec<BenchmarkResult> = Vec::new();
    for cfg in &configs {
        let r = run_algo(cfg, m, n, num_seeds, module.clone(), stream.clone(), &prop)?;
        results.push(r);
    }

    // ── Summary table ─────────────────────────────────────────────────────
    let mut out = String::new();
    writeln!(out, "CUR Decomposition Benchmark Results").unwrap();
    writeln!(out, "Matrix : {}×{}  max_rank={}", m, n, m.min(n)).unwrap();
    writeln!(out, "Seeds  : {} (+ 1 warmup per algorithm, warmup excluded)", num_seeds).unwrap();
    writeln!(out, "PTX    : {}", ptx_path).unwrap();
    writeln!(out, "Score  : fnorm / optimal  (1.0 = perfect, lower is better)").unwrap();
    writeln!(out).unwrap();
    writeln!(out,
        "{:<24} {:>10} {:>14} {:>10} {:>12} {:>10} {:>10}",
        "algorithm", "solve_ms", "gen+verify_ms", "mean_score", "var_score", "min_score", "max_score"
    ).unwrap();
    writeln!(out, "{}", "─".repeat(96)).unwrap();
    for r in &results {
        writeln!(out,
            "{:<24} {:>10.1} {:>14.1} {:>10.4} {:>12.6} {:>10.4} {:>10.4}",
            r.label, r.avg_solve_ms, r.avg_gen_verify_ms,
            r.mean_score, r.var_score, r.min_score, r.max_score
        ).unwrap();
    }

    println!("\n{}", out);

    let out_path = format!("{}/examples/benchmark_results.txt", env!("CARGO_MANIFEST_DIR"));
    fs::write(&out_path, &out)
        .map_err(|e| anyhow!("Failed to write results to '{}': {}", out_path, e))?;
    println!("Results saved to {}", out_path);

    Ok(())
}
