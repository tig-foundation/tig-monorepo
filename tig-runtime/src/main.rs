use anyhow::{anyhow, Result};
use clap::{arg, ArgAction, Command};
use libloading::Library;
use std::{fs, panic, path::PathBuf};
use tig_challenges::*;
use tig_structs::core::{BenchmarkSettings, OutputData, Solution};
use tig_utils::{compress_obj, dejsonify, jsonify};

fn cli() -> Command {
    Command::new("tig-native-runtime")
        .about("Computes or verifies solutions")
        .arg_required_else_help(true)
        .subcommand(
            Command::new("compute_solution")
                .about("Computes a solution")
                .arg(
                    arg!(<SETTINGS> "Settings json string or path to json file")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(
                    arg!(<RAND_HASH> "A string used in seed generation")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(arg!(<NONCE> "Nonce value").value_parser(clap::value_parser!(u64)))
                .arg(
                    arg!(<BINARY> "Path to a native binary")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(--fuel [FUEL] "Optional maximum fuel parameter")
                        .default_value("2000000000")
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--output [OUTPUT_FILE] "If set, the output data will be saved to this file path (default json")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(--compress [COMPRESS] "If output file is set, the output data will be compressed as zlib")
                    .action(ArgAction::SetTrue)
                ),
        )
}

fn main() {
    let matches = cli().get_matches();

    if let Err(e) = match matches.subcommand() {
        Some(("compute_solution", sub_m)) => compute_solution(
            sub_m.get_one::<String>("SETTINGS").unwrap().clone(),
            sub_m.get_one::<String>("RAND_HASH").unwrap().clone(),
            *sub_m.get_one::<u64>("NONCE").unwrap(),
            sub_m.get_one::<PathBuf>("BINARY").unwrap().clone(),
            *sub_m.get_one::<u64>("fuel").unwrap(),
            sub_m.get_one::<PathBuf>("output").cloned(),
            sub_m.get_one::<bool>("compress").unwrap().clone(),
        ),
        _ => Err(anyhow!("Invalid subcommand")),
    } {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

pub fn compute_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    library_path: PathBuf,
    max_fuel: u64,
    output_file: Option<PathBuf>,
    compress: bool,
) -> Result<()> {
    let settings = load_settings(&settings);

    let library = load_module(&library_path)?;
    let fuel_remaining_ptr = unsafe { *library.get::<*mut u64>(b"__fuel_remaining")? };
    unsafe { *fuel_remaining_ptr = max_fuel };
    let seed = settings.calc_seed(&rand_hash, nonce);

    let mut solution = Solution::new();
    let mut err_msg = Option::<String>::None;

    macro_rules! dispatch_challenges {
        ( $( $c:ident ),+ $(,)? ) => {{
            match settings.challenge_id.as_str() {
                $(
                    stringify!($c) => {
                        let solve_challenge_fn = unsafe {
                            library.get::<fn($c::Challenge) -> Result<Option<$c::Solution>, String>>(
                                b"entry_point",
                            )?
                        };

                        let challenge = $c::Challenge::generate_instance_from_vec(
                            seed,
                            &settings.difficulty,
                        ).unwrap();

                        match solve_challenge_fn(challenge) {
                            Ok(Some(s)) => {
                                solution = serde_json::to_value(s)
                                    .unwrap()
                                    .as_object()
                                    .unwrap()
                                    .to_owned();
                            }
                            Ok(None) => {}
                            Err(e)    => err_msg = Some(e),
                        }
                    }
                )+
                _ => panic!("Unsupported challenge"),
            }
        }};
    }
    // fix
    dispatch_challenges!(c001, c002, c003);

    let fuel_remaining = unsafe { **library.get::<*const u64>(b"__fuel_remaining")? };
    let runtime_signature = unsafe { **library.get::<*const u64>(b"__runtime_signature")? };

    let output_data = OutputData {
        nonce,
        runtime_signature,
        fuel_consumed: max_fuel - fuel_remaining,
        solution,
    };
    if let Some(path) = output_file {
        if compress {
            fs::write(&path, compress_obj(&output_data))?;
        } else {
            fs::write(&path, jsonify(&output_data))?;
        }
        println!("output_data written to: {:?}", path);
    } else {
        println!("{}", jsonify(&output_data));
    }
    if let Some(err_msg) = err_msg {
        eprintln!("Runtime error: {}", err_msg);
        std::process::exit(86);
    } else if output_data.solution.len() == 0 {
        eprintln!("No solution found");
        std::process::exit(85);
    }
    Ok(())
}

fn load_settings(settings: &str) -> BenchmarkSettings {
    let settings = if settings.ends_with(".json") {
        fs::read_to_string(settings).unwrap_or_else(|_| {
            eprintln!("Failed to read settings file: {}", settings);
            std::process::exit(1);
        })
    } else {
        settings.to_string()
    };

    dejsonify::<BenchmarkSettings>(&settings).unwrap_or_else(|_| {
        eprintln!("Failed to parse settings");
        std::process::exit(1);
    })
}

pub fn load_module(path: &PathBuf) -> Result<Library> {
    let res = panic::catch_unwind(|| unsafe { Library::new(path) });

    match res {
        Ok(lib_result) => lib_result.map_err(|e| anyhow!(e.to_string())),
        Err(_) => Err(anyhow!("Failed to load module")),
    }
}
