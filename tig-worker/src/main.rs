mod worker;
use anyhow::{anyhow, Result};
use clap::{arg, Command};
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};
use tig_structs::core::{BenchmarkSettings, MerkleProof};
use tig_utils::{dejsonify, jsonify, MerkleHash, MerkleTree};
use tokio::runtime::Runtime;

fn cli() -> Command {
    Command::new("tig-worker")
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
                .arg(arg!(<WASM> "Path to a wasm file").value_parser(clap::value_parser!(PathBuf)))
                .arg(
                    arg!(--fuel [FUEL] "Optional maximum fuel parameter for WASM VM")
                        .default_value("2000000000")
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--interval [INTERVAL] "Optional amount of fuel between signatures")
                        .default_value("200000000")
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
                .arg(
                    arg!(<RAND_HASH> "A string used in seed generation")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(arg!(<NONCE> "Nonce value").value_parser(clap::value_parser!(u64)))
                .arg(
                    arg!(<SOLUTION> "Solution json string or path to json file")
                        .value_parser(clap::value_parser!(String)),
                ),
        )
        .subcommand(
            Command::new("compute_batch")
                .about("Computes batch of nonces and generates Merkle proofs")
                .arg(
                    arg!(<SETTINGS> "Settings json string or path to json file")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(
                    arg!(<RAND_HASH> "A string used in seed generation")
                        .value_parser(clap::value_parser!(String)),
                )
                .arg(arg!(<START_NONCE> "Starting nonce").value_parser(clap::value_parser!(u64)))
                .arg(
                    arg!(<NUM_NONCES> "Number of nonces to compute")
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(<BATCH_SIZE> "Batch size for Merkle tree")
                        .value_parser(clap::value_parser!(u64)),
                )
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
                )
                .arg(
                    arg!(--sampled [SAMPLED_NONCES] "Sampled nonces for which to generate proofs")
                        .num_args(1..)
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--workers [WORKERS] "Number of worker threads")
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize)),
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
            sub_m.get_one::<PathBuf>("WASM").unwrap().clone(),
            *sub_m.get_one::<u64>("mem").unwrap(),
            *sub_m.get_one::<u64>("fuel").unwrap(),
        ),
        Some(("verify_solution", sub_m)) => verify_solution(
            sub_m.get_one::<String>("SETTINGS").unwrap().clone(),
            sub_m.get_one::<String>("RAND_HASH").unwrap().clone(),
            *sub_m.get_one::<u64>("NONCE").unwrap(),
            sub_m.get_one::<String>("SOLUTION").unwrap().clone(),
        ),

        Some(("compute_batch", sub_m)) => compute_batch(
            sub_m.get_one::<String>("SETTINGS").unwrap().clone(),
            sub_m.get_one::<String>("RAND_HASH").unwrap().clone(),
            *sub_m.get_one::<u64>("START_NONCE").unwrap(),
            *sub_m.get_one::<u64>("NUM_NONCES").unwrap(),
            *sub_m.get_one::<u64>("BATCH_SIZE").unwrap(),
            sub_m.get_one::<PathBuf>("WASM").unwrap().clone(),
            *sub_m.get_one::<u64>("mem").unwrap(),
            *sub_m.get_one::<u64>("fuel").unwrap(),
            sub_m
                .get_many::<u64>("sampled")
                .map(|values| values.cloned().collect())
                .unwrap_or_default(),
            *sub_m.get_one::<usize>("workers").unwrap(),
        ),
        _ => Err(anyhow!("Invalid subcommand")),
    } {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn compute_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    wasm_path: PathBuf,
    max_memory: u64,
    max_fuel: u64,
) -> Result<()> {
    let settings = load_settings(&settings);
    let wasm = load_wasm(&wasm_path);

    let (output_data, err_msg) = worker::compute_solution(
        &settings,
        &rand_hash,
        nonce,
        wasm.as_slice(),
        max_memory,
        max_fuel,
    )?;
    println!("{}", jsonify(&output_data));
    if let Some(err_msg) = err_msg {
        return Err(anyhow!("Runtime error: {}", err_msg));
    } else if output_data.solution.len() == 0 {
        return Err(anyhow!("No solution found"));
    }
    worker::verify_solution(&settings, &rand_hash, nonce, &output_data.solution)
        .map_err(|e| anyhow!("Invalid solution: {}", e))
}

fn verify_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    solution: String,
) -> Result<()> {
    let settings = load_settings(&settings);
    let solution = load_solution(&solution);

    match worker::verify_solution(&settings, &rand_hash, nonce, &solution) {
        Ok(()) => {
            println!("Solution is valid");
            Ok(())
        }
        Err(e) => Err(anyhow!("Invalid solution: {}", e)),
    }
}

fn compute_batch(
    settings: String,
    rand_hash: String,
    start_nonce: u64,
    num_nonces: u64,
    batch_size: u64,
    wasm_path: PathBuf,
    max_memory: u64,
    max_fuel: u64,
    sampled_nonces: Vec<u64>,
    num_workers: usize,
) -> Result<()> {
    if num_nonces == 0 || batch_size < num_nonces {
        return Err(anyhow!(
            "Invalid number of nonces. Must be non-zero and less than batch size"
        ));
    }
    let end_nonce = start_nonce + num_nonces;
    for &nonce in &sampled_nonces {
        if nonce < start_nonce || nonce >= end_nonce {
            return Err(anyhow!(
                "Sampled nonce {} is out of range [{}, {})",
                nonce,
                start_nonce,
                end_nonce
            ));
        }
    }
    if batch_size == 0 || (batch_size & (batch_size - 1)) != 0 {
        return Err(anyhow!("Batch size must be a power of 2"));
    }

    let settings = Arc::new(load_settings(&settings));
    let wasm = Arc::new(load_wasm(&wasm_path));

    let runtime = Runtime::new()?;

    runtime.block_on(async {
        let mut output_data_map = HashMap::new();
        let mut hashes = vec![MerkleHash::null(); num_nonces as usize];
        let mut solution_nonces = Vec::new();

        // Create a stream of nonces and process them concurrently
        let results = stream::iter(start_nonce..end_nonce)
            .map(|nonce| {
                let settings = Arc::clone(&settings);
                let wasm = Arc::clone(&wasm);
                let rand_hash = rand_hash.clone();
                let sampled_nonces = sampled_nonces.clone();
                tokio::spawn(async move {
                    let (output_data, err_msg) = worker::compute_solution(
                        &settings,
                        &rand_hash,
                        nonce,
                        wasm.as_slice(),
                        max_memory,
                        max_fuel,
                    )?;
                    let is_solution = err_msg.is_none()
                        && worker::verify_solution(
                            &settings,
                            &rand_hash,
                            nonce,
                            &output_data.solution,
                        )
                        .is_ok();
                    let hash = MerkleHash::from(output_data.clone());
                    // only keep the data if required
                    let output_data = if sampled_nonces.contains(&nonce) {
                        Some(output_data)
                    } else {
                        None
                    };
                    Ok::<(u64, Option<worker::OutputData>, MerkleHash, bool), anyhow::Error>((
                        nonce,
                        output_data,
                        hash,
                        is_solution,
                    ))
                })
            })
            .buffer_unordered(num_workers)
            .collect::<Vec<_>>()
            .await;

        for result in results {
            let (nonce, output_data, hash, is_solution) = result??;
            if let Some(output_data) = output_data {
                output_data_map.insert(nonce, output_data);
            }
            if is_solution {
                solution_nonces.push(nonce);
            }
            *hashes.get_mut((nonce - start_nonce) as usize).unwrap() = hash;
        }

        let tree = MerkleTree::new(hashes, batch_size as usize)?;
        let merkle_root = tree.calc_merkle_root();

        let mut merkle_proofs = Vec::new();
        for (nonce, output_data) in output_data_map {
            merkle_proofs.push(MerkleProof {
                leaf: output_data,
                branch: tree.calc_merkle_branch((nonce - start_nonce) as usize)?,
            });
        }

        let result = json!({
            "merkle_root": merkle_root,
            "merkle_proofs": merkle_proofs,
            "solution_nonces": solution_nonces,
        });

        println!("{}", jsonify(&result));
        Ok(())
    })
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

fn load_solution(solution: &str) -> worker::Solution {
    let solution = if solution.ends_with(".json") {
        fs::read_to_string(&solution).unwrap_or_else(|_| {
            eprintln!("Failed to read solution file: {}", solution);
            std::process::exit(1);
        })
    } else {
        solution.to_string()
    };

    dejsonify::<worker::Solution>(&solution).unwrap_or_else(|_| {
        eprintln!("Failed to parse solution");
        std::process::exit(1);
    })
}

fn load_wasm(wasm_path: &PathBuf) -> Vec<u8> {
    fs::read(wasm_path).unwrap_or_else(|_| {
        eprintln!("Failed to read wasm file: {}", wasm_path.display());
        std::process::exit(1);
    })
}
