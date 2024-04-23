mod worker;
use anyhow::{anyhow, Result};
use clap::{arg, Command};
use std::{fs, path::PathBuf, process::exit};
use tig_structs::core::{BenchmarkSettings, SolutionData};
use tig_utils::{dejsonify, jsonify};

fn cli() -> Command {
    Command::new("rust_cli_app")
        .about("CLI app to compute or verify solutions")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("compute_solution")
                .about("Computes a solution")
                .arg(
                    arg!(<SETTINGS> "Path to a settings file")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(arg!(<NONCE> "A u32 nonce").value_parser(clap::value_parser!(u32)))
                .arg(arg!(<WASM> "Path to a wasm file").value_parser(clap::value_parser!(PathBuf)))
                .arg(
                    arg!(--endless "Optional flag to compute solutions continuously")
                        .action(clap::ArgAction::SetTrue),
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
                .arg(
                    arg!(--debug "Optional flag to print debug messages")
                        .action(clap::ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("verify_solution")
                .about("Verifies a solution")
                .arg(
                    arg!(<SETTINGS> "Path to a settings file")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(<SOLUTION> "Path to a solution file")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(--debug "Optional flag to print debug messages")
                        .action(clap::ArgAction::SetTrue),
                ),
        )
}

fn main() {
    let matches = cli().get_matches();

    let result = match matches.subcommand() {
        Some(("compute_solution", sub_matches)) => {
            let settings_path = sub_matches.get_one::<PathBuf>("SETTINGS").unwrap();
            let nonce = *sub_matches.get_one::<u32>("NONCE").unwrap();
            let wasm_path = sub_matches.get_one::<PathBuf>("WASM").unwrap();
            let endless = sub_matches.get_flag("endless");
            let max_fuel = *sub_matches.get_one::<u64>("fuel").unwrap();
            let max_memory = *sub_matches.get_one::<u64>("mem").unwrap();
            let debug = sub_matches.get_flag("debug");

            compute_solution(
                settings_path,
                nonce,
                wasm_path,
                max_memory,
                max_fuel,
                endless,
                debug,
            )
        }
        Some(("verify_solution", sub_matches)) => {
            let settings_path = sub_matches.get_one::<PathBuf>("SETTINGS").unwrap();
            let solution_path = sub_matches.get_one::<PathBuf>("SOLUTION").unwrap();
            let debug = sub_matches.get_flag("debug");

            verify_solution(settings_path, solution_path, debug)
        }
        _ => unreachable!("The CLI should prevent getting here"),
    };
    match result {
        Ok(_) => exit(0),
        Err(e) => {
            println!("Error: {}", e);
            exit(1);
        }
    };
}

fn compute_solution(
    settings_path: &PathBuf,
    nonce: u32,
    wasm_path: &PathBuf,
    max_memory: u64,
    max_fuel: u64,
    endless: bool,
    debug: bool,
) -> Result<()> {
    let settings = dejsonify::<BenchmarkSettings>(
        &fs::read_to_string(settings_path)
            .map_err(|e| anyhow!("Failed to read settings file: {}", e))?,
    )
    .map_err(|e| anyhow!("Failed to dejsonify settings file: {}", e))?;

    let wasm = fs::read(wasm_path).map_err(|e| anyhow!("Failed to read wasm file: {}", e))?;

    let mut i = 0;
    loop {
        let result =
            worker::compute_solution(&settings, nonce + i, wasm.as_slice(), max_memory, max_fuel)?;
        match result {
            Ok(solution_data) => {
                println!("{}", jsonify(&solution_data));
            }
            Err(e) => {
                if debug {
                    println!("Nonce {}, no solution: {}", nonce + i, e);
                }
            }
        }
        i += 1;
        if !endless {
            break;
        }
    }

    Ok(())
}

fn verify_solution(settings_path: &PathBuf, solution_path: &PathBuf, debug: bool) -> Result<()> {
    let settings = dejsonify::<BenchmarkSettings>(
        &fs::read_to_string(settings_path)
            .map_err(|e| anyhow!("Failed to read settings file: {}", e))?,
    )
    .map_err(|e| anyhow!("Failed to dejsonify settings file: {}", e))?;
    let solution_data = dejsonify::<SolutionData>(
        fs::read_to_string(solution_path)
            .map_err(|e| anyhow!("Failed to read solution file: {}", e))?
            .as_str(),
    )
    .map_err(|e| anyhow!("Failed to dejsonify solution: {}", e))?;
    let result = worker::verify_solution(&settings, solution_data.nonce, &solution_data.solution)?;
    if debug {
        if let Err(e) = result {
            println!("Solution is invalid: {}", e);
        } else {
            println!("Solution is valid");
        }
    }
    Ok(())
}
