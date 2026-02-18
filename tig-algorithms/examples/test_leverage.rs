use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
    runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
};
use std::{cell::RefCell, sync::Arc, time::Instant};
use tig_challenges::cur_decomposition::*;

#[path = "../src/cur_decomposition/leverage/mod.rs"]
mod leverage;

// ─── Stats helper ────────────────────────────────────────────────────────────

struct Stats {
    values: Vec<f64>,
}

impl Stats {
    fn new() -> Self {
        Self { values: Vec::new() }
    }
    fn push(&mut self, v: f64) {
        self.values.push(v);
    }
    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    fn mean(&self) -> f64 {
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }
    fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn max(&self) -> f64 {
        self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    fn std(&self) -> f64 {
        let m = self.mean();
        (self.values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.values.len() as f64)
            .sqrt()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// The singular values used in challenge generation are strictly decreasing
/// (exp(-l5*sqrt(j+1)/sqrt(T)) for j=0..T), so their sorted order is fixed
/// regardless of the seed shuffle. optimal_fnorm is purely a function of T and K.
fn compute_optimal_fnorm(true_rank: i32, target_rank: i32) -> f32 {
    let l5: f32 = 2.0;
    (0..true_rank)
        .skip(target_rank as usize)
        .map(|j| {
            let s = (-l5 * ((j + 1) as f32).sqrt() / (true_rank as f32).sqrt()).exp();
            s * s
        })
        .sum::<f32>()
        .sqrt()
}

fn make_seed(index: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&index.to_le_bytes());
    seed
}

// ─── Per-config benchmark ─────────────────────────────────────────────────────

fn run_config(
    m: i32,
    n: i32,
    true_rank: i32,
    target_rank: i32,
    num_seeds: usize,
    hyperparameters: &Option<serde_json::Map<String, serde_json::Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    println!("\n╔══ M={} N={} T={} K={} ══", m, n, true_rank, target_rank);
    println!(
        "║ {:>5}  {:>10}  {:>10}  {:>9}  {:>9}",
        "seed", "gen_ms", "algo_ms", "quality", "fnorm/opt"
    );
    println!("╟{}", "─".repeat(52));

    let track = Track { m, n };
    let optimal = compute_optimal_fnorm(true_rank, target_rank);
    let baseline = 50.0 * optimal;

    let mut gen_ms = Stats::new();
    let mut algo_ms = Stats::new();
    let mut quality_stats = Stats::new();
    let mut ratio_stats = Stats::new();
    let mut passes = 0usize;
    let mut attempted = 0usize;

    for s in 0..num_seeds {
        let seed = make_seed(s as u64);

        // ── Generation ─────────────────────────────────────────────────────
        let t0 = Instant::now();
        let challenge = match Challenge::generate_single_instance(
            &seed,
            &track,
            true_rank,
            target_rank,
            module.clone(),
            stream.clone(),
            prop,
        ) {
            Ok(c) => c,
            Err(e) => {
                println!("║ {:>5}  gen error: {}", s, e);
                continue;
            }
        };
        let g_ms = t0.elapsed().as_secs_f64() * 1000.0;
        gen_ms.push(g_ms);

        // ── Algorithm ──────────────────────────────────────────────────────
        let best: RefCell<Option<Solution>> = RefCell::new(None);
        let t1 = Instant::now();
        let result = {
            let save_fn = |sol: &Solution| -> Result<()> {
                *best.borrow_mut() = Some(sol.clone());
                Ok(())
            };
            leverage::solve_challenge(
                &challenge,
                &save_fn,
                hyperparameters,
                module.clone(),
                stream.clone(),
                prop,
            )
        };
        let a_ms = t1.elapsed().as_secs_f64() * 1000.0;
        algo_ms.push(a_ms);
        attempted += 1;

        let solution = match result {
            Ok(sol) => sol.or_else(|| best.into_inner()),
            Err(e) => {
                println!("║ {:>5}  {:>10.1}  algo error: {}", s, g_ms, e);
                continue;
            }
        };

        match solution {
            None => {
                println!(
                    "║ {:>5}  {:>10.1}  {:>10.1}  no solution",
                    s, g_ms, a_ms
                );
            }
            Some(sol) => {
                let fnorm =
                    challenge.evaluate_fnorm(&sol, module.clone(), stream.clone(), prop)?;
                let q = (baseline - fnorm) / (baseline - optimal);
                let ratio = fnorm / optimal;

                quality_stats.push(q as f64);
                ratio_stats.push(ratio as f64);
                if q > 0.0 {
                    passes += 1;
                }

                println!(
                    "║ {:>5}  {:>10.1}  {:>10.1}  {:>9.4}  {:>9.4}{}",
                    s,
                    g_ms,
                    a_ms,
                    q,
                    ratio,
                    if q <= 0.0 { "  ← FAIL" } else { "" }
                );
            }
        }
    }

    println!("╟{}", "─".repeat(52));

    if !gen_ms.is_empty() {
        println!(
            "║ gen_ms   avg={:8.1}  min={:8.1}  max={:8.1}",
            gen_ms.mean(),
            gen_ms.min(),
            gen_ms.max()
        );
    }
    if !algo_ms.is_empty() {
        println!(
            "║ algo_ms  avg={:8.1}  min={:8.1}  max={:8.1}",
            algo_ms.mean(),
            algo_ms.min(),
            algo_ms.max()
        );
    }
    if !quality_stats.is_empty() {
        println!(
            "║ quality  avg={:8.4}  min={:8.4}  max={:8.4}  std={:.4}  pass={}/{}",
            quality_stats.mean(),
            quality_stats.min(),
            quality_stats.max(),
            quality_stats.std(),
            passes,
            attempted
        );
        println!(
            "║ f/opt    avg={:8.4}  min={:8.4}  max={:8.4}  std={:.4}",
            ratio_stats.mean(),
            ratio_stats.min(),
            ratio_stats.max(),
            ratio_stats.std()
        );
    }
    println!("╚{}", "═".repeat(52));

    Ok(())
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn print_usage(prog: &str) {
    eprintln!(
        "Usage: {} <PTX_PATH> <M> <N> <T> <K> [<M> <N> <T> <K> ...] [OPTIONS]",
        prog
    );
    eprintln!();
    eprintln!("  PTX_PATH       Path to compiled leverage.ptx");
    eprintln!("                 Build with:");
    eprintln!("                   CHALLENGE=cur_decomposition python3 tig-binary/scripts/build_ptx leverage");
    eprintln!();
    eprintln!("  M N T K        Matrix dimensions and ranks. Multiple groups run sequentially.");
    eprintln!("                   M = rows, N = cols, T = true rank, K = target rank (K <= T)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --seeds  N     Number of seeds to test per config  (default: 5)");
    eprintln!("  --trials N     Leverage score trials per solve      (default: algorithm default)");
    eprintln!("  --gpu    N     GPU device index                     (default: 0)");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 6 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let ptx_path = &args[1];

    // Parse positional M N T K groups and named options
    let mut configs: Vec<(i32, i32, i32, i32)> = Vec::new();
    let mut num_seeds: usize = 5;
    let mut num_trials: Option<usize> = None;
    let mut gpu_device: usize = 0;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                i += 1;
                num_seeds = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--seeds requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--seeds must be a positive integer"))?;
            }
            "--trials" => {
                i += 1;
                num_trials = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow!("--trials requires a value"))?
                        .parse()
                        .map_err(|_| anyhow!("--trials must be a positive integer"))?,
                );
            }
            "--gpu" => {
                i += 1;
                gpu_device = args
                    .get(i)
                    .ok_or_else(|| anyhow!("--gpu requires a value"))?
                    .parse()
                    .map_err(|_| anyhow!("--gpu must be a non-negative integer"))?;
            }
            _ => {
                // Expect a group of 4 integers: M N T K
                if i + 3 >= args.len() {
                    eprintln!(
                        "Expected M N T K group at position {}, but not enough arguments.",
                        i
                    );
                    print_usage(&args[0]);
                    std::process::exit(1);
                }
                let parse = |s: &str, name: &str| -> Result<i32> {
                    s.parse()
                        .map_err(|_| anyhow!("{} must be a positive integer, got '{}'", name, s))
                };
                let m = parse(&args[i], "M")?;
                let n = parse(&args[i + 1], "N")?;
                let t = parse(&args[i + 2], "T")?;
                let k = parse(&args[i + 3], "K")?;
                configs.push((m, n, t, k));
                i += 3;
            }
        }
        i += 1;
    }

    if configs.is_empty() {
        eprintln!("No M N T K configs provided.");
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let hyperparameters = num_trials.map(|t| {
        let mut map = serde_json::Map::new();
        map.insert("num_trials".to_string(), serde_json::json!(t));
        map
    });

    // ── CUDA setup (shared across all configs) ────────────────────────────
    let ptx_src = std::fs::read_to_string(ptx_path)
        .map_err(|e| anyhow!("Failed to read PTX '{}': {}", ptx_path, e))?;
    // Replace the sentinel fuel-limit with u64::MAX so kernels never abort.
    let ptx_src = ptx_src.replace("0xdeadbeefdeadbeef", "0xffffffffffffffff");
    let ptx = Ptx::from_src(ptx_src);

    let num_gpus = CudaContext::device_count()?;
    if num_gpus == 0 {
        return Err(anyhow!("No CUDA devices found"));
    }

    println!("=== CUR Decomposition Benchmark (leverage) ===");
    println!("PTX    : {}", ptx_path);
    println!("GPU    : device {} of {}", gpu_device, num_gpus);
    println!("Seeds  : {}", num_seeds);
    if let Some(t) = num_trials {
        println!("Trials : {}", t);
    }
    println!("Configs: {}", configs.len());

    let ctx = CudaContext::new(gpu_device)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(ptx)?;
    let stream = ctx.default_stream();
    let prop = get_device_prop(gpu_device as i32)?;

    // ── Run each config ───────────────────────────────────────────────────
    for (m, n, true_rank, target_rank) in &configs {
        run_config(
            *m,
            *n,
            *true_rank,
            *target_rank,
            num_seeds,
            &hyperparameters,
            module.clone(),
            stream.clone(),
            &prop,
        )?;
    }

    Ok(())
}
