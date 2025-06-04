use anyhow::{anyhow, Result};
use clap::{arg, Command};
use std::{fs, io::Read, panic, path::PathBuf};
use tig_challenges::*;
use tig_structs::core::{BenchmarkSettings, Solution};
use tig_utils::dejsonify;

#[cfg(feature = "cuda")]
use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};

fn cli() -> Command {
    Command::new("tig-verifier")
        .about("Verifies a solution or merkle proof")
        .arg_required_else_help(true)
        .subcommand(
            Command::new("verify_solution")
                .about("Verifies a solution")
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
                    arg!(<SOLUTION> "Solution json string, path to json file, or '-' for stdin")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(
                    arg!(--ptx [PTX] "Path to a CUDA ptx file")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(--gpu [GPU] "Which GPU device to use")
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("verify_merkle_proof")
                .about("Verifies a merkle proof")
                .arg(arg!(<ROOT> "Merkle root").value_parser(clap::value_parser!(String)))
                .arg(
                    arg!(<PROOF> "Merkle proof json string, path to json file, or '-' for stdin")
                        .value_parser(clap::value_parser!(String)),
                ),
        )
}

fn main() {
    let matches = cli().get_matches();

    if let Err(e) = match matches.subcommand() {
        Some(("verify_solution", sub_m)) => verify_solution(
            sub_m.get_one::<String>("SETTINGS").unwrap().clone(),
            sub_m.get_one::<String>("RAND_HASH").unwrap().clone(),
            *sub_m.get_one::<u64>("NONCE").unwrap(),
            sub_m.get_one::<String>("SOLUTION").unwrap().clone(),
            sub_m.get_one::<PathBuf>("ptx").cloned(),
            sub_m.get_one::<usize>("gpu").cloned(),
        ),
        Some(("verify_merkle_proof", sub_m)) => verify_merkle_proof(
            sub_m.get_one::<String>("ROOT").unwrap().clone(),
            sub_m.get_one::<String>("PROOF").unwrap().clone(),
        ),
        _ => Err(anyhow!("Invalid subcommand")),
    } {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

pub fn verify_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    solution_path: String,
    ptx_path: Option<PathBuf>,
    gpu_device: Option<usize>,
) -> Result<()> {
    let settings = load_settings(&settings);
    let solution = load_solution(&solution_path);
    let seed = settings.calc_seed(&rand_hash, nonce);

    let mut err_msg = Option::<String>::None;

    macro_rules! dispatch_challenge {
        ($c:ident, cpu) => {{
            let challenge =
                $c::Challenge::generate_instance(&seed, &settings.difficulty.into()).unwrap();

            match $c::Solution::try_from(solution) {
                Ok(solution) => match challenge.verify_solution(&solution) {
                    Ok(_) => println!("Solution is valid"),
                    Err(e) => err_msg = Some(format!("Invalid solution: {}", e)),
                },
                Err(_) => {
                    err_msg = Some(format!(
                        "Invalid solution. Cannot convert to {}::Solution",
                        stringify!($c)
                    ))
                }
            }
        }};

        ($c:ident, gpu) => {{
            if ptx_path.is_none() {
                panic!("PTX file is required for GPU challenges.");
            }

            let num_gpus = CudaContext::device_count()?;
            if num_gpus == 0 {
                panic!("No CUDA devices found");
            }
            let gpu_device = gpu_device.unwrap_or((nonce % num_gpus as u64) as usize);
            let ptx = Ptx::from_file(ptx_path.unwrap());
            let ctx = CudaContext::new(gpu_device).unwrap();
            ctx.set_blocking_synchronize()?;
            let module = ctx.load_module(ptx).unwrap();
            let stream = ctx.default_stream();
            let prop = get_device_prop(gpu_device as i32).unwrap();

            let challenge = $c::Challenge::generate_instance(
                &seed,
                &settings.difficulty.into(),
                module.clone(),
                stream.clone(),
                &prop,
            )
            .unwrap();

            match $c::Solution::try_from(solution) {
                Ok(solution) => {
                    match challenge.verify_solution(
                        &solution,
                        module.clone(),
                        stream.clone(),
                        &prop,
                    ) {
                        Ok(_) => {
                            stream.synchronize()?;
                            ctx.synchronize()?;
                            println!("Solution is valid");
                        }
                        Err(e) => err_msg = Some(format!("Invalid solution: {}", e)),
                    }
                }
                Err(_) => {
                    err_msg = Some(format!(
                        "Invalid solution. Cannot convert to {}::Solution",
                        stringify!($c)
                    ))
                }
            }
        }};
    }

    match settings.challenge_id.as_str() {
        "c001" => {
            #[cfg(not(feature = "c001"))]
            panic!("tig-verifier was not compiled with '--features c001'");
            #[cfg(feature = "c001")]
            dispatch_challenge!(c001, cpu)
        }
        "c002" => {
            #[cfg(not(feature = "c002"))]
            panic!("tig-verifier was not compiled with '--features c002'");
            #[cfg(feature = "c002")]
            dispatch_challenge!(c002, cpu)
        }
        "c003" => {
            #[cfg(not(feature = "c003"))]
            panic!("tig-verifier was not compiled with '--features c003'");
            #[cfg(feature = "c003")]
            dispatch_challenge!(c003, cpu)
        }
        "c004" => {
            #[cfg(not(feature = "c004"))]
            panic!("tig-verifier was not compiled with '--features c004'");
            #[cfg(feature = "c004")]
            dispatch_challenge!(c004, gpu)
        }
        "c005" => {
            #[cfg(not(feature = "c005"))]
            panic!("tig-verifier was not compiled with '--features c005'");
            #[cfg(feature = "c005")]
            dispatch_challenge!(c005, gpu)
        }
        _ => panic!("Unsupported challenge"),
    }

    if let Some(err_msg) = err_msg {
        eprintln!("Verification error: {}", err_msg);
        std::process::exit(1);
    }

    Ok(())
}

pub fn verify_merkle_proof(_merkle_root: String, _merkle_proof: String) -> Result<()> {
    // TODO
    Err(anyhow!("Merkle proof verification is not implemented yet"))
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

fn load_solution(solution: &str) -> Solution {
    let solution = if solution == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .unwrap_or_else(|_| {
                eprintln!("Failed to read solution from stdin");
                std::process::exit(1);
            });
        buffer
    } else if solution.ends_with(".json") {
        fs::read_to_string(&solution).unwrap_or_else(|_| {
            eprintln!("Failed to read solution file: {}", solution);
            std::process::exit(1);
        })
    } else {
        solution.to_string()
    };

    dejsonify::<Solution>(&solution).unwrap_or_else(|_| {
        eprintln!("Failed to parse solution");
        std::process::exit(1);
    })
}
