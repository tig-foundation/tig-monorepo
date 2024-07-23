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
            arg!(<timer> "Time taken testing each algorithm")
                .default_value("30000")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            arg!(<name> "Name of the algorithm being tested")
                .default_value("Error, no name")
                .value_parser(clap::value_parser!(String)),
        )
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
    let max_time = *matches.get_one::<f64>("timer").unwrap();
    let name = matches.get_one::<String>("name").unwrap();

    test_performance(settings_path, wasm_path, nonce, max_memory, max_fuel, max_time, name);
}

fn test_performance(
    settings_path: &PathBuf,
    wasm_path: &PathBuf,
    nonce: u32,
    max_memory: u64,
    max_fuel: u64,
    max_time: f64,
    name: &String,
) {
    let settings = dejsonify::<BenchmarkSettings>(
        &fs::read_to_string(settings_path).expect("Failed to read settings file"),
    )
    .expect("Failed to dejsonify settings file");

    let wasm = fs::read(wasm_path).expect("Failed to read wasm file");

    let start = Instant::now();
    let mut num_solutions = 0u32;
    let mut num_errors = 0u32;
    let mut i = 0;

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

        if elapsed > max_time {
            let num_invalid_solutions = i - num_solutions - num_errors;
            println!("Algorithm: {}, Instances: {}, solutions: {}, invalid_solutions: {}, errors: {}, overall_time: {}ms, avg_time_per_solution: {}ms", name, i, num_solutions, num_invalid_solutions, num_errors, elapsed, if num_solutions == 0 { 0f64 } else { elapsed / num_solutions as f64});
            break;
        }
    }
}
