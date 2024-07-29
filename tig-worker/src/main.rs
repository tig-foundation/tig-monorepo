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
            arg!(<INSTANCE> "Current Instance")
                .default_value("0")
                .value_parser(clap::value_parser!(u32))
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
    let max_fuel = *matches.get_one::<u64>("fuel").unwrap();
    let max_memory = *matches.get_one::<u64>("mem").unwrap();
    let i = *matches.get_one::<u32>("INSTANCE").unwrap();

    test_performance(settings_path, wasm_path, max_memory, max_fuel, i);
}

fn test_performance(
    settings_path: &PathBuf,
    wasm_path: &PathBuf,
    max_memory: u64,
    max_fuel: u64,
    i: u32,
) {
    let settings = dejsonify::<BenchmarkSettings>(
        &fs::read_to_string(settings_path).expect("Failed to read settings file"),
    )
    .expect("Failed to dejsonify settings file");

    let wasm = fs::read(wasm_path).expect("Failed to read wasm file");

    match worker::compute_solution(&settings, i, wasm.as_slice(), max_memory, max_fuel)
    {
        Ok(worker::ComputeResult::ValidSolution(_)) => {
            std::process::exit(0)
        }
        Err(_) => {
            std::process::exit(2)
        }
        _ => {
            std::process::exit(1)
        }
    }
    
}
