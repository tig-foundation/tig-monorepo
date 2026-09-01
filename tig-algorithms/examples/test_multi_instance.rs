// Large benchmark for CUR decomposition.
//
// Algorithms : fastest_algo | leverage (1t) | leverage (10t) | leverage (15t) | sketchy
// Sizes      : official CUR tracks from tig-challenges
// Seeds      : N real + 1 warmup per (algo, size)
//
// Score      : 1/(1+log(max(q,1))/log(k+1)), q=fnorm/optimal
// Per seed   : avg_score = mean(score) over sub-instances of that seed
// Reported   : avg/min/max of per-seed avg_scores over the N seeds
//
// Outputs:
//   examples/large_benchmark_summary.txt  — summary table
//   examples/large_benchmark_scores.csv   — every (algo,m,n,seed,sub,true_rank,k,score)
//
// Usage:
//   cargo run --release --example test_multi_instance --features cur_decomposition \
//       -- <PTX_PATH> [--seeds N] [--gpu N] [--algos name1,name2]

use anyhow::{anyhow, Result};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
    runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
};
use std::{
    cell::RefCell, collections::HashMap, fmt::Write as FmtWrite, fs, sync::Arc, time::Instant,
};
use tig_challenges::cur_decomposition::*;

#[path = "../src/cur_decomposition/leverage/mod.rs"]
mod leverage;

#[path = "../src/cur_decomposition/fastest_algo/mod.rs"]
mod fastest_algo;

#[path = "../src/cur_decomposition/sketchy/mod.rs"]
mod sketchy;

// ─── Null score (do-nothing baseline) ─────────────────────────────────────────

fn design_config(m: i32, n: i32, poly: bool) -> DesignGenerationConfig {
    DesignGenerationConfig {
        m,
        n,
        delta: DESIGN_DELTA,
        poly,
        spectrum_a: DESIGN_SPECTRUM_A,
    }
}

/// Apply the protocol score analytically to the do-nothing CUR=0 baseline.
fn null_score_for_sub(seed: &[u8; 32], m: i32, n: i32, poly: bool, sub_idx: usize) -> Result<f64> {
    let metadata = sample_design_metadata(seed, &design_config(m, n, poly))?;
    let sub = &metadata[sub_idx];
    let mut singular_values: Vec<f64> = sub
        .singular_values
        .iter()
        .map(|value| *value as f64)
        .collect();
    singular_values.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
    let target_k = sub.target_k as usize;
    let a_fnorm = singular_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let optimal_fnorm = singular_values[target_k..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    score_from_errors(a_fnorm, optimal_fnorm, sub.target_k)
}

// ─── Stats ────────────────────────────────────────────────────────────────────

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
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
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
    poly: bool,
}

struct RunResult {
    algo: String,
    m: i32,
    n: i32,
    poly: bool,
    avg_solve_ms: f64,
    avg_gen_verify_ms: f64,
    avg_score: f64,
    min_seed_score: f64,
    max_seed_score: f64,
}

struct ScoreRecord {
    algo: String,
    m: i32,
    n: i32,
    poly: bool,
    seed: usize,
    sub_idx: usize,
    true_rank: i32,
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
        "fastest_algo" => {
            fastest_algo::solve_challenge(challenge, save_fn, hp, module, stream, prop)
        }
        "sketchy" => sketchy::solve_challenge(challenge, save_fn, hp, module, stream, prop),
        _ => leverage::solve_challenge(challenge, save_fn, hp, module, stream, prop),
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
    let poly = size.poly;
    let track = Track { m, n, poly };
    let mut all_gen_ms = Stats::new();
    let mut all_solve_ms = Stats::new();
    let mut all_verify_ms = Stats::new();
    let mut seed_avgs = Stats::new();
    let mut records: Vec<ScoreRecord> = Vec::new();

    let decay_label = if poly { "poly" } else { "exp" };
    println!(
        "\n╔══ {} │ M={} N={} ({}) │ {} seeds + 1 warmup ══",
        cfg.label, m, n, decay_label, num_seeds
    );
    println!(
        "║ {:>5}  {:>9}  {:>9}  {:>9}  {:>10}  {:>8}  {:>8}",
        "seed", "gen_ms", "solve_ms", "vrfy_ms", "avg_score", "min", "max"
    );
    println!("╟{}", "─".repeat(74));

    for seed_idx in 0..=(num_seeds) {
        let is_warmup = seed_idx == 0;
        let si = if is_warmup { num_seeds } else { seed_idx - 1 };
        let seed = make_seed(si as u64);
        let metadata = sample_design_metadata(&seed, &design_config(m, n, poly))?;

        let t_gen = Instant::now();
        let challenges = match Challenge::generate_multiple_instances(
            &seed,
            &track,
            module.clone(),
            stream.clone(),
            prop,
        ) {
            Ok(cs) => cs,
            Err(e) => {
                let lbl = if is_warmup {
                    "warm".to_string()
                } else {
                    si.to_string()
                };
                println!("║ {:>5}  generate error: {}", lbl, e);
                continue;
            }
        };
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

        let mut total_solve_ms = 0.0f64;
        let mut total_verify_ms = 0.0f64;
        let mut sub_scores: Vec<f64> = Vec::new();

        for (sub_idx, challenge) in challenges.iter().enumerate() {
            let optimal = challenge.optimal_fnorm() as f64;

            let score = if cfg.algo == "zero" {
                // Analytical: CUR=0 gives error = ||A||_F, no solver needed
                Some(null_score_for_sub(&seed, m, n, poly, sub_idx)?)
            } else {
                let best: RefCell<Option<Solution>> = RefCell::new(None);

                let t_solve = Instant::now();
                let result = {
                    let save_fn = |sol: &Solution| -> Result<()> {
                        *best.borrow_mut() = Some(sol.clone());
                        Ok(())
                    };
                    solve(
                        cfg.algo,
                        challenge,
                        &save_fn,
                        &cfg.hyperparameters,
                        module.clone(),
                        stream.clone(),
                        prop,
                    )
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
                    let fnorm_result = challenge.evaluate_fast_fnorm(
                        &sol.c_idxs,
                        &sol.r_idxs,
                        module.clone(),
                        stream.clone(),
                        prop,
                    );
                    match fnorm_result {
                        Ok(fnorm) => {
                            total_verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
                            Some(score_from_errors(
                                fnorm as f64,
                                optimal,
                                challenge.target_k,
                            )?)
                        }
                        Err(e) => {
                            eprintln!("  verify error (k={}): {}", challenge.target_k, e);
                            None
                        }
                    }
                } else {
                    None
                }
            };

            if let Some(score) = score {
                sub_scores.push(score);
                if !is_warmup {
                    let true_rank = metadata[sub_idx].true_rank;
                    records.push(ScoreRecord {
                        algo: cfg.label.to_string(),
                        m,
                        n,
                        poly,
                        seed: si,
                        sub_idx,
                        true_rank,
                        target_k: challenge.target_k,
                        score,
                    });
                }
            }
        }

        let (seed_avg, seed_min, seed_max) = if sub_scores.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            let mean = sub_scores.iter().sum::<f64>() / sub_scores.len() as f64;
            let min = sub_scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = sub_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mean, min, max)
        };

        let lbl = if is_warmup {
            "warm".to_string()
        } else {
            si.to_string()
        };
        println!(
            "║ {:>5}  {:>9.1}  {:>9.1}  {:>9.1}  {:>10.4}  {:>8.4}  {:>8.4}{}",
            lbl,
            gen_ms,
            total_solve_ms,
            total_verify_ms,
            seed_avg,
            seed_min,
            seed_max,
            if is_warmup { "  (warmup)" } else { "" }
        );

        if is_warmup {
            continue;
        }

        all_gen_ms.push(gen_ms);
        all_solve_ms.push(total_solve_ms);
        all_verify_ms.push(total_verify_ms);
        if seed_avg.is_finite() {
            seed_avgs.push(seed_avg);
        }
    }

    println!("╟{}", "─".repeat(74));
    let avg_solve = if all_solve_ms.is_empty() {
        f64::NAN
    } else {
        all_solve_ms.mean()
    };
    let avg_gen_verify = if all_gen_ms.is_empty() {
        f64::NAN
    } else {
        all_gen_ms.mean() + all_verify_ms.mean()
    };
    let (avg_score, min_seed_score, max_seed_score) = if seed_avgs.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (seed_avgs.mean(), seed_avgs.min(), seed_avgs.max())
    };
    println!(
        "║ avg    {:>9.1}  {:>9.1}  {:>9.1}  {:>10.4}  {:>8.4}  {:>8.4}",
        all_gen_ms.mean(),
        avg_solve,
        all_verify_ms.mean(),
        avg_score,
        min_seed_score,
        max_seed_score
    );
    println!("╚{}", "═".repeat(74));

    Ok((
        RunResult {
            algo: cfg.label.to_string(),
            m,
            n,
            poly,
            avg_solve_ms: avg_solve,
            avg_gen_verify_ms: avg_gen_verify,
            avg_score,
            min_seed_score,
            max_seed_score,
        },
        records,
    ))
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <PTX_PATH> [--seeds N] [--gpu N] [--algos name1,name2]",
            args[0]
        );
        std::process::exit(1);
    }

    let ptx_path = &args[1];
    let mut num_seeds = 25usize;
    let mut gpu_device = 0usize;
    let mut algo_filter: Vec<String> = Vec::new();
    let mut poly_filter: Option<bool> = None; // None = both, Some(true) = poly only, Some(false) = exp only
    let mut size_filter: Vec<(i32, i32)> = Vec::new(); // empty = all sizes

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                i += 1;
                num_seeds = args[i].parse()?;
            }
            "--gpu" => {
                i += 1;
                gpu_device = args[i].parse()?;
            }
            "--algos" => {
                i += 1;
                algo_filter = args[i].split(',').map(|s| s.trim().to_string()).collect();
            }
            "--poly" => {
                poly_filter = Some(true);
            }
            "--no-poly" => {
                poly_filter = Some(false);
            }
            "--sizes" => {
                i += 1;
                for s in args[i].split(',') {
                    let s = s.trim();
                    let mut parts = s.splitn(2, 'x');
                    let m: i32 = parts
                        .next()
                        .ok_or_else(|| anyhow!("bad --sizes format"))?
                        .parse()?;
                    let n: i32 = parts
                        .next()
                        .ok_or_else(|| anyhow!("bad --sizes format"))?
                        .parse()?;
                    size_filter.push((m, n));
                }
            }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // ── CUDA setup ─────────────────────────────────────────────────────────────
    let ptx_src = std::fs::read_to_string(ptx_path)
        .map_err(|e| anyhow!("Failed to read PTX '{}': {}", ptx_path, e))?;
    let ptx_src = ptx_src.replace("0xdeadbeefdeadbeef", "0xffffffffffffffff");
    let ptx = Ptx::from_src(ptx_src);

    let num_gpus = CudaContext::device_count()?;
    if num_gpus == 0 {
        return Err(anyhow!("No CUDA devices found"));
    }

    let ctx = CudaContext::new(gpu_device)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(ptx)?;
    let stream = ctx.default_stream();
    let prop = get_device_prop(gpu_device as i32)?;

    // ── Algorithm configs ──────────────────────────────────────────────────────
    let hp_1t = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(1));
        m
    };
    let hp_10t = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(10));
        m
    };
    let hp_15t = {
        let mut m = serde_json::Map::new();
        m.insert("num_trials".into(), serde_json::json!(15));
        m
    };

    let all_algos: Vec<AlgoConfig> = vec![
        AlgoConfig {
            label: "fastest_algo",
            algo: "fastest_algo",
            hyperparameters: None,
        },
        AlgoConfig {
            label: "leverage (1t)",
            algo: "leverage",
            hyperparameters: Some(hp_1t),
        },
        AlgoConfig {
            label: "leverage (10t)",
            algo: "leverage",
            hyperparameters: Some(hp_10t),
        },
        AlgoConfig {
            label: "leverage (15t)",
            algo: "leverage",
            hyperparameters: Some(hp_15t),
        },
        AlgoConfig {
            label: "sketchy",
            algo: "sketchy",
            hyperparameters: None,
        },
    ];
    let algos: Vec<AlgoConfig> = if algo_filter.is_empty() {
        all_algos
    } else {
        all_algos
            .into_iter()
            .filter(|a| algo_filter.iter().any(|f| a.label.contains(f.as_str())))
            .collect()
    };

    // ── Size configs ───────────────────────────────────────────────────────────
    let sizes: Vec<SizeConfig> = TRACKS
        .iter()
        .filter(|t| poly_filter.map_or(true, |p| t.poly == p))
        .filter(|t| size_filter.is_empty() || size_filter.contains(&(t.m, t.n)))
        .map(|t| SizeConfig {
            m: t.m,
            n: t.n,
            poly: t.poly,
        })
        .collect();

    println!("=== CUR Decomposition Large Benchmark ===");
    println!("PTX    : {}", ptx_path);
    println!("GPU    : device {} of {}", gpu_device, num_gpus);
    println!("Seeds  : {} real + 1 warmup per (algo, size)", num_seeds);
    println!("Score  : 1/(1+log(max(q,1))/log(k+1)), q=fnorm/optimal");
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

    // ── Collect algo labels in insertion order ─────────────────────────────────
    let algo_labels: Vec<String> = {
        let mut seen: Vec<String> = Vec::new();
        for r in &all_results {
            if !seen.contains(&r.algo) {
                seen.push(r.algo.clone());
            }
        }
        seen
    };

    // ── Per-(m,n) breakdown tables ─────────────────────────────────────────────
    // For each (m,n): rows = (true_rank, target_k, seed), cols = algos

    // Index records: (algo, m, n, poly, sub_idx, seed) -> score
    let mut score_map: HashMap<(String, i32, i32, bool, usize, usize), f64> = HashMap::new();
    for rec in &all_records {
        score_map.insert(
            (
                rec.algo.clone(),
                rec.m,
                rec.n,
                rec.poly,
                rec.sub_idx,
                rec.seed,
            ),
            rec.score,
        );
    }

    // Collect sorted seed indices seen per (m, n, poly)
    let mut seed_map: HashMap<(i32, i32, bool), Vec<usize>> = HashMap::new();
    for rec in &all_records {
        let v = seed_map.entry((rec.m, rec.n, rec.poly)).or_default();
        if !v.contains(&rec.seed) {
            v.push(rec.seed);
        }
    }
    for v in seed_map.values_mut() {
        v.sort();
    }

    // Index RunResults: (algo, m, n, poly) -> RunResult
    let mut result_map: HashMap<(String, i32, i32, bool), &RunResult> = HashMap::new();
    for r in &all_results {
        result_map.insert((r.algo.clone(), r.m, r.n, r.poly), r);
    }

    let col_w = 13usize; // column width per algorithm

    let mut breakdown = String::new();
    writeln!(
        breakdown,
        "\n=== Score Breakdown by (m, n, true_rank, target_k) ==="
    )
    .unwrap();
    writeln!(
        breakdown,
        "Score = 1/(1+log(max(q,1))/log(k+1)), q=fnorm/optimal"
    )
    .unwrap();
    writeln!(
        breakdown,
        "null  = the same scoring rule applied to the do-nothing CUR=0 baseline"
    )
    .unwrap();
    writeln!(breakdown, "Values are averages over {} seed(s).", num_seeds).unwrap();

    let sizes_list: Vec<(i32, i32, bool)> = {
        let mut v: Vec<(i32, i32, bool)> = sizes.iter().map(|s| (s.m, s.n, s.poly)).collect();
        v.dedup();
        v
    };

    for &(m, n, poly) in &sizes_list {
        let max_rank = m.min(n);
        let decay_label = if poly { "poly" } else { "exp" };

        writeln!(breakdown).unwrap();
        writeln!(
            breakdown,
            "m={m}  n={n}  max_rank={max_rank}  decay={decay_label}"
        )
        .unwrap();

        // Header  (null column comes first, before the real algos)
        let ratio_w = 7usize;
        let null_w = 8usize;
        let header_row = format!(
            "  {:>10}  {:>9}  {:>ratio_w$}  {:>4}  {:>null_w$}  {}",
            "true_rank",
            "target_k",
            "k/rank",
            "seed",
            "null",
            algo_labels
                .iter()
                .map(|l| format!("{:>col_w$}", l))
                .collect::<Vec<_>>()
                .join("  "),
            ratio_w = ratio_w,
            null_w = null_w,
        );
        writeln!(breakdown, "{}", header_row).unwrap();
        writeln!(breakdown, "  {}", "─".repeat(header_row.len() - 2)).unwrap();

        let seeds = seed_map.get(&(m, n, poly)).cloned().unwrap_or_default();

        // One row per (sub-instance, seed)
        for sub_idx in 0..NUM_SUB_INSTANCES {
            for &seed in &seeds {
                let seed_bytes = make_seed(seed as u64);
                let metadata = sample_design_metadata(&seed_bytes, &design_config(m, n, poly))?;
                let true_rank = metadata[sub_idx].true_rank;
                let target_k = metadata[sub_idx].target_k;
                let ratio = target_k as f64 / true_rank as f64;
                let null_sc = null_score_for_sub(&seed_bytes, m, n, poly, sub_idx)?;
                let scores_str: String = algo_labels
                    .iter()
                    .map(|lbl| {
                        let key = (lbl.clone(), m, n, poly, sub_idx, seed);
                        if let Some(&sc) = score_map.get(&key) {
                            format!("{:>col_w$.4}", sc)
                        } else {
                            format!("{:>col_w$}", "N/A")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("  ");
                writeln!(
                    breakdown,
                    "  {:>10}  {:>9}  {:>ratio_w$.4}  {:>4}  {:>null_w$.4}  {}",
                    true_rank,
                    target_k,
                    ratio,
                    seed,
                    null_sc,
                    scores_str,
                    ratio_w = ratio_w,
                    null_w = null_w
                )
                .unwrap();
            }
        }

        writeln!(breakdown, "  {}", "─".repeat(header_row.len() - 2)).unwrap();

        // Average row (averaged over all seeds and sub-instances)
        let mut null_scores = Vec::new();
        for &seed in &seeds {
            let seed_bytes = make_seed(seed as u64);
            for sub_idx in 0..NUM_SUB_INSTANCES {
                null_scores.push(null_score_for_sub(&seed_bytes, m, n, poly, sub_idx)?);
            }
        }
        let avg_null_sc = null_scores.iter().sum::<f64>() / null_scores.len() as f64;
        let avg_scores_str: String = algo_labels
            .iter()
            .map(|lbl| {
                let vs: Vec<f64> = (0..NUM_SUB_INSTANCES)
                    .flat_map(|si| {
                        seeds
                            .iter()
                            .filter_map(|&seed| {
                                score_map.get(&(lbl.clone(), m, n, poly, si, seed)).copied()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                if vs.is_empty() {
                    format!("{:>col_w$}", "N/A")
                } else {
                    format!("{:>col_w$.4}", vs.iter().sum::<f64>() / vs.len() as f64)
                }
            })
            .collect::<Vec<_>>()
            .join("  ");
        writeln!(
            breakdown,
            "  {:>10}  {:>9}  {:>ratio_w$}  {:>4}  {:>null_w$.4}  {}",
            "average",
            "",
            "",
            "",
            avg_null_sc,
            avg_scores_str,
            ratio_w = ratio_w,
            null_w = null_w
        )
        .unwrap();

        // Solve_ms row
        let solve_ms_str: String = algo_labels
            .iter()
            .map(|lbl| {
                let key = (lbl.clone(), m, n, poly);
                if let Some(r) = result_map.get(&key) {
                    format!("{:>col_w$.1}", r.avg_solve_ms)
                } else {
                    format!("{:>col_w$}", "N/A")
                }
            })
            .collect::<Vec<_>>()
            .join("  ");
        writeln!(
            breakdown,
            "  {:>10}  {:>9}  {:>ratio_w$}  {:>4}  {:>null_w$}  {}",
            "solve_ms",
            "",
            "",
            "",
            "N/A",
            solve_ms_str,
            ratio_w = ratio_w,
            null_w = null_w
        )
        .unwrap();

        // gen+verify_ms row
        let gv_ms_str: String = algo_labels
            .iter()
            .map(|lbl| {
                let key = (lbl.clone(), m, n, poly);
                if let Some(r) = result_map.get(&key) {
                    format!("{:>col_w$.1}", r.avg_gen_verify_ms)
                } else {
                    format!("{:>col_w$}", "N/A")
                }
            })
            .collect::<Vec<_>>()
            .join("  ");
        writeln!(
            breakdown,
            "  {:>10}  {:>9}  {:>ratio_w$}  {:>4}  {:>null_w$}  {}",
            "gen+vrfy_ms",
            "",
            "",
            "",
            "N/A",
            gv_ms_str,
            ratio_w = ratio_w,
            null_w = null_w
        )
        .unwrap();
    }

    println!("{}", breakdown);

    // ── Classic summary table ──────────────────────────────────────────────────
    let mut summary = String::new();
    writeln!(summary, "CUR Decomposition Large Benchmark Results").unwrap();
    writeln!(summary, "PTX   : {}", ptx_path).unwrap();
    writeln!(
        summary,
        "Seeds : {} real + 1 warmup per (algo, size)",
        num_seeds
    )
    .unwrap();
    writeln!(
        summary,
        "Score : 1/(1+log(max(q,1))/log(k+1)), q=fnorm/optimal"
    )
    .unwrap();
    writeln!(summary).unwrap();
    writeln!(
        summary,
        "{:<24} {:>6} {:>6} {:>10} {:>14} {:>10} {:>12} {:>12}",
        "algorithm",
        "m",
        "n",
        "solve_ms",
        "gen+verify_ms",
        "avg_score",
        "min_seed_sc",
        "max_seed_sc"
    )
    .unwrap();
    writeln!(summary, "{}", "─".repeat(100)).unwrap();

    let mut prev_algo = String::new();
    for r in &all_results {
        if r.algo != prev_algo && !prev_algo.is_empty() {
            writeln!(summary).unwrap();
        }
        writeln!(
            summary,
            "{:<24} {:>6} {:>6} {:>10.1} {:>14.1} {:>10.4} {:>12.4} {:>12.4}",
            r.algo,
            r.m,
            r.n,
            r.avg_solve_ms,
            r.avg_gen_verify_ms,
            r.avg_score,
            r.min_seed_score,
            r.max_seed_score
        )
        .unwrap();
        prev_algo = r.algo.clone();
    }

    writeln!(summary).unwrap();
    writeln!(summary, "{}", breakdown).unwrap();

    let manifest = env!("CARGO_MANIFEST_DIR");
    let summary_path = format!("{}/examples/large_benchmark_summary.txt", manifest);
    fs::write(&summary_path, &summary).map_err(|e| anyhow!("Failed to write summary: {}", e))?;
    println!("Summary saved to {}", summary_path);

    // ── Detailed scores CSV ────────────────────────────────────────────────────
    let mut csv = String::new();
    writeln!(csv, "algo,m,n,poly,seed,sub_idx,true_rank,target_k,score").unwrap();
    for rec in &all_records {
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{:.6}",
            rec.algo,
            rec.m,
            rec.n,
            rec.poly,
            rec.seed,
            rec.sub_idx,
            rec.true_rank,
            rec.target_k,
            rec.score
        )
        .unwrap();
    }

    let scores_path = format!("{}/examples/large_benchmark_scores.csv", manifest);
    fs::write(&scores_path, &csv).map_err(|e| anyhow!("Failed to write scores CSV: {}", e))?;
    println!("Detailed scores saved to {}", scores_path);

    Ok(())
}
