mod worker;
use clap::{arg, Command};
use std::{fs, path::PathBuf};
use tig_structs::core::BenchmarkSettings;
use tig_utils::{dejsonify, jsonify};

fn cli() -> Command {
    Command::new("tig-worker")
        .about("Computes or verifies a solution")
        .arg_required_else_help(true)
        .subcommand(
            Command::new("compute_solution")
                .about("Computes a solution")
                .arg(
                    arg!(<SETTINGS> "Settings json string or path to json file")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(arg!(<NONCE> "Nonce value").value_parser(clap::value_parser!(u64)))
                .arg(arg!(<WASM> "Path to a wasm file").value_parser(clap::value_parser!(PathBuf)))
                .arg(
                    arg!(--fuel [FUEL] "Optional maximum fuel parameter for WASM VM")
                        .default_value("2000000000")
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--mem [MEM] "Optional maximum memory parameter for WASM VM")
                        .default_value("1000000000")
                        .value_parser(clap::value_parser!(u64)),
                ),
        )
        .subcommand(
            Command::new("verify_solution")
                .about("Verifies a solution")
                .arg(
                    arg!(<SETTINGS> "Settings json string or path to json file")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(arg!(<NONCE> "Nonce value").value_parser(clap::value_parser!(u64)))
                .arg(
                    arg!(<SOLUTION> "Solution json string or path to json file")
                        .value_parser(clap::value_parser!(String)),
                ),
        )
}

fn main() {
    let matches = cli().get_matches();

    match matches.subcommand() {
        Some(("compute_solution", sub_m)) => compute_solution(
            sub_m.get_one::<String>("SETTINGS").unwrap().clone(),
            *sub_m.get_one::<u64>("NONCE").unwrap(),
            sub_m.get_one::<PathBuf>("WASM").unwrap().clone(),
            *sub_m.get_one::<u64>("mem").unwrap(),
            *sub_m.get_one::<u64>("fuel").unwrap(),
        ),
        Some(("verify_solution", sub_m)) => verify_solution(
            sub_m.get_one::<String>("SETTINGS").unwrap().clone(),
            *sub_m.get_one::<u64>("NONCE").unwrap(),
            sub_m.get_one::<String>("SOLUTION").unwrap().clone(),
        ),
        _ => {}
    }
}

fn compute_solution(
    mut settings: String,
    nonce: u64,
    wasm_path: PathBuf,
    max_memory: u64,
    max_fuel: u64,
) {
    if settings.ends_with(".json") {
        settings = fs::read_to_string(&settings).unwrap_or_else(|_| {
            eprintln!("Failed to read settings file: {}", settings);
            std::process::exit(1);
        });
    }
    let settings = dejsonify::<BenchmarkSettings>(&settings).unwrap_or_else(|_| {
        eprintln!("Failed to parse settings");
        std::process::exit(1);
    });

    let wasm = fs::read(&wasm_path).unwrap_or_else(|_| {
        eprintln!("Failed to read wasm file: {}", wasm_path.display());
        std::process::exit(1);
    });

    match worker::compute_solution(&settings, nonce, wasm.as_slice(), max_memory, max_fuel) {
        Ok(Some(solution_data)) => {
            println!("{}", jsonify(&solution_data));
            if solution_data.solution.len() == 0 {
                eprintln!("No solution found");
                std::process::exit(1);
            }
            match worker::verify_solution(&settings, nonce, &solution_data.solution) {
                Ok(()) => {
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("Invalid solution: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Ok(None) => {
            eprintln!("No solution found");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn verify_solution(mut settings: String, nonce: u64, mut solution: String) {
    if settings.ends_with(".json") {
        settings = fs::read_to_string(&settings).unwrap_or_else(|_| {
            eprintln!("Failed to read settings file: {}", settings);
            std::process::exit(1);
        });
    }
    let settings = dejsonify::<BenchmarkSettings>(&settings).unwrap_or_else(|_| {
        eprintln!("Failed to parse settings");
        std::process::exit(1);
    });

    if solution.ends_with(".json") {
        solution = fs::read_to_string(&solution).unwrap_or_else(|_| {
            eprintln!("Failed to read solution file: {}", solution);
            std::process::exit(1);
        });
    }
    let solution = dejsonify::<worker::Solution>(&solution).unwrap_or_else(|_| {
        eprintln!("Failed to parse solution");
        std::process::exit(1);
    });

    match worker::verify_solution(&settings, nonce, &solution) {
        Ok(()) => {
            println!("Solution is valid");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Invalid solution: {}", e);
            std::process::exit(1);
        }
    }
}
