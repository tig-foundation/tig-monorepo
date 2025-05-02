use anyhow::{anyhow, Result};
use clap::{arg, Command};
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::{fs, path::PathBuf};
use tempfile::NamedTempFile;
use tig_structs::core::*;
use tig_utils::{compress_obj, decompress_obj, dejsonify, jsonify, MerkleHash, MerkleTree};
use tokio::runtime::Runtime;

fn cli() -> Command {
    Command::new("tig-worker")
        .about("Computes batch of nonces and generates Merkle proofs")
        .arg_required_else_help(true)
        .arg(arg!(<RUNTIME> "Path to tig-runtime executable").value_parser(clap::value_parser!(PathBuf)))
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
        .arg(
            arg!(<BINARY> "Path to a shared object (*.so) file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--ptx [PTX] "Path to a CUDA ptx file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--fuel [FUEL] "Optional maximum fuel parameter for runtime")
                .default_value("2000000000")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            arg!(--workers [WORKERS] "Number of worker threads")
                .default_value("1")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            arg!(--output [OUTPUT_FOLDER] "If set, the data for nonce will be saved as '<nonce>.json' in this folder")
                .value_parser(clap::value_parser!(PathBuf)),
        )
}

fn main() {
    let matches = cli().get_matches();

    if let Err(e) = compute_batch(
        matches.get_one::<PathBuf>("RUNTIME").unwrap().clone(),
        matches.get_one::<String>("SETTINGS").unwrap().clone(),
        matches.get_one::<String>("RAND_HASH").unwrap().clone(),
        *matches.get_one::<u64>("START_NONCE").unwrap(),
        *matches.get_one::<u64>("NUM_NONCES").unwrap(),
        *matches.get_one::<u64>("BATCH_SIZE").unwrap(),
        matches.get_one::<PathBuf>("BINARY").unwrap().clone(),
        matches.get_one::<PathBuf>("ptx").cloned(),
        *matches.get_one::<u64>("fuel").unwrap(),
        *matches.get_one::<usize>("workers").unwrap(),
        matches.get_one::<PathBuf>("output").cloned(),
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn compute_batch(
    runtime_path: PathBuf,
    settings: String,
    rand_hash: String,
    start_nonce: u64,
    num_nonces: u64,
    batch_size: u64,
    binary_path: PathBuf,
    ptx_path: Option<PathBuf>,
    max_fuel: u64,
    num_workers: usize,
    output_folder: Option<PathBuf>,
) -> Result<()> {
    if num_nonces == 0 || batch_size < num_nonces {
        return Err(anyhow!(
            "Invalid number of nonces. Must be non-zero and less than batch size"
        ));
    }
    if batch_size == 0 || (batch_size & (batch_size - 1)) != 0 {
        return Err(anyhow!("Batch size must be a power of 2"));
    }

    if let Some(path) = &output_folder {
        fs::create_dir_all(path)?;
    }

    let settings = load_settings(&settings);
    match settings.challenge_id.as_str() {
        "c004" => {
            #[cfg(not(feature = "cuda"))]
            panic!("tig-worker was not compiled with '--features cuda'");

            #[cfg(feature = "cuda")]
            if ptx_path.is_none() {
                return Err(anyhow!(
                    "PTX file is required for challenge {}",
                    settings.challenge_id
                ));
            }
        }
        _ => {
            if ptx_path.is_some() {
                return Err(anyhow!(
                    "PTX file is not required for challenge {}",
                    settings.challenge_id
                ));
            }
        }
    }
    let settings = jsonify(&settings);

    let runtime = Runtime::new()?;

    runtime.block_on(async {
        let mut hashes = vec![MerkleHash::null(); num_nonces as usize];
        let mut solution_nonces = Vec::new();

        // Create a stream of nonces and process them concurrently
        let results = stream::iter(start_nonce..(start_nonce + num_nonces))
            .map(|nonce| {
                let runtime_path = runtime_path.clone();
                let settings = settings.clone();
                let rand_hash = rand_hash.clone();
                let binary_path = binary_path.clone();
                let output_folder = output_folder.clone();
                #[cfg(feature = "cuda")]
                let ptx_path = ptx_path.clone();
                #[cfg(feature = "cuda")]
                let num_gpus = cudarc::runtime::result::device::get_count().unwrap() as u64;

                tokio::spawn(async move {
                    let temp_file = NamedTempFile::new()?;
                    let mut cmd = std::process::Command::new(runtime_path);
                    cmd.arg("compute_solution")
                        .arg(settings)
                        .arg(rand_hash)
                        .arg(nonce.to_string())
                        .arg(binary_path)
                        .arg("--output")
                        .arg(temp_file.path())
                        .arg("--compress");

                    #[cfg(feature = "cuda")]
                    if let Some(ptx_path) = ptx_path {
                        cmd.arg("--ptx")
                            .arg(ptx_path)
                            .arg("--gpu")
                            .arg((nonce % num_gpus).to_string());
                    }

                    let output = cmd.output().unwrap();

                    let exit_code = output.status.code();
                    let is_solution = output.status.success();
                    if exit_code == Some(87) {
                        // out of fuel
                        // let mut runtime_signature = 0;
                        // let stdout = String::from_utf8_lossy(&output.stdout);
                        // let mut lines = stdout.lines().rev();
                        // while let Some(line) = lines.next() {
                        //     if line.starts_with("Runtime signature: ") {
                        //         if let Some(sig) = line.strip_prefix("Runtime signature: ") {
                        //             if let Ok(sig) = sig.trim().parse::<u64>() {
                        //                 runtime_signature = sig;
                        //                 break;
                        //             }
                        //         }
                        //     }
                        // }

                        let output_data = OutputData {
                            nonce,
                            solution: Solution::new(),
                            fuel_consumed: max_fuel + 1,
                            runtime_signature: 0,
                        };
                        let hash = MerkleHash::from(output_data.clone());
                        Ok::<(u64, MerkleHash, bool, Option<OutputData>), anyhow::Error>((
                            nonce,
                            hash,
                            is_solution,
                            output_folder.is_some().then(|| output_data),
                        ))
                    } else if is_solution || exit_code == Some(86) || exit_code == Some(85) {
                        let bytes = fs::read(temp_file.path())?;
                        let output_data: OutputData = decompress_obj(&bytes)?;
                        let hash = MerkleHash::from(output_data.clone());
                        Ok::<(u64, MerkleHash, bool, Option<OutputData>), anyhow::Error>((
                            nonce,
                            hash,
                            is_solution,
                            output_folder.is_some().then(|| output_data),
                        ))
                    } else {
                        Err(anyhow!(
                            "Failed to compute nonce {}: {:?}",
                            nonce,
                            output.status
                        ))
                    }
                })
            })
            .buffer_unordered(num_workers)
            .collect::<Vec<_>>()
            .await;

        let mut dump = Vec::new();
        for result in results {
            let (nonce, hash, is_solution, output_data) = result??;
            if is_solution {
                solution_nonces.push(nonce);
            }
            if let Some(output_data) = output_data {
                dump.push(output_data);
            }
            *hashes.get_mut((nonce - start_nonce) as usize).unwrap() = hash;
        }
        if let Some(path) = output_folder {
            dump.sort_by_key(|data| data.nonce);
            fs::write(&path.join("data.zlib"), compress_obj(&dump))?;
            fs::write(&path.join("hashes.zlib"), compress_obj(&hashes))?;
        }

        let tree = MerkleTree::new(hashes, batch_size as usize)?;
        let merkle_root = tree.calc_merkle_root();

        let result = json!({
            "merkle_root": merkle_root,
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
