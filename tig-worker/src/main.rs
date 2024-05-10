mod worker;
use clap::{arg, Command};
use std::{fs, path::PathBuf, time::Instant};
use tig_structs::core::BenchmarkSettings;
use tig_utils::dejsonify;

fn cli() -> Command {
    Command::new("tig_performance_tester")
        .about("Tests performance of a TIG algorithm")
        .arg_required_else_help(true)
        .arg(arg!(<SETTINGS> "Path to a settings file").value_parser(clap::value_parser!(PathBuf)))
        .arg(arg!(<WASM> "Path to a wasm file").value_parser(clap::value_parser!(PathBuf)))
        .arg(
            arg!(--nonce "Starting nonce")
                .default_value("0")
                .value_parser(clap::value_parser!(u32)),
        )
        .arg(
            arg!(--fuel [FUEL] "Optional maximum fuel parameter for WASM VM")
                .default_value("1000000000")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            arg!(--mem [MEM] "Optional maximum memory parameter for WASM VM")
                .default_value("1000000000")
                .value_parser(clap::value_parser!(u64)),
        )
}

fn main() {
    let matches = cli().get_matches();

    let settings_path = matches.get_one::<PathBuf>("SETTINGS").unwrap();
    let wasm_path = matches.get_one::<PathBuf>("WASM").unwrap();
    let nonce = *matches.get_one::<u32>("nonce").unwrap();
    let max_fuel = *matches.get_one::<u64>("fuel").unwrap();
    let max_memory = *matches.get_one::<u64>("mem").unwrap();

    test_performance(settings_path, wasm_path, nonce, max_memory, max_fuel);
}

fn test_performance(
    settings_path: &PathBuf,
    wasm_path: &PathBuf,
    nonce: u32,
    max_memory: u64,
    max_fuel: u64,
) {
    let settings = dejsonify::<BenchmarkSettings>(
        &fs::read_to_string(settings_path).expect("Failed to read settings file"),
    )
    .expect("Failed to dejsonify settings file");

    let wasm = fs::read(wasm_path).expect("Failed to read wasm file");

    println!("Algorithm: {:?}", wasm_path);
    println!("Settings: {:?}", settings);
    let start = Instant::now();
    let mut num_solutions = 0u32;
    let mut num_errors = 0u32;
    let mut i = 0;
    let mut last_update = 0f64;
    loop {
        match worker::compute_solution(&settings, nonce + i, wasm.as_slice(), max_memory, max_fuel)
        {
            Ok(worker::ComputeResult::ValidSolution(_)) => {
                num_solutions += 1;
            }
            Err(_) => {
                num_errors += 1;
            }
            _ => {}
        }
        i += 1;
        let elapsed = start.elapsed().as_micros() as f64 / 1000.0;
        if elapsed >= last_update + 1000.0 {
            let num_invalid_solutions = i - num_solutions - num_errors;
            last_update = elapsed;
            println!(
                "#instances: {}, #solutions: {} ({:.1}%), #invalid_solutions: {} ({:.1}%), #errors: {} ({:.1}%), avg_time_per_solution: {}ms",
                i,
                num_solutions,
                num_solutions as f64 / i as f64 * 100.0,
                num_invalid_solutions,
                num_invalid_solutions as f64 / i as f64 * 100.0,
                num_errors,
                num_errors as f64 / i as f64 * 100.0,
                if num_solutions == 0 { 0f64 } else { elapsed / num_solutions as f64 }
            );
        }
    }
}
