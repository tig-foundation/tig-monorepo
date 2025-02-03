mod worker;
use anyhow::{anyhow, Result};
use clap::{arg, Command};
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::{fs, path::PathBuf, sync::Arc};
use tig_structs::core::*;
use tig_utils::{compress_obj, dejsonify, jsonify, MerkleHash, MerkleTree};
use tig_worker::OutputData;
use tokio::runtime::Runtime;
use tig_challenges::knapsack::{Challenge as KnapsackChallenge};
use tig_challenges::satisfiability::{Challenge as SatisfiabilityChallenge};
use tig_challenges::vehicle_routing::{Challenge as VehicleRoutingChallenge};
use tig_challenges::vector_search::{Challenge as VectorSearchChallenge};
use tig_challenges::ChallengeTrait;

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
                .arg(
                    arg!(--wasm <PATH> "Path to a wasm file")
                        .value_parser(clap::value_parser!(PathBuf))
                        .conflicts_with("native")
                        .required_unless_present("native"),
                )
                .arg(
                    arg!(--native <PATH> "Path to a native binary")
                        .value_parser(clap::value_parser!(PathBuf))
                        .conflicts_with("wasm")
                        .required_unless_present("wasm"),
                )
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
                .arg(
                    arg!(--wasm <PATH> "Path to a wasm file")
                        .value_parser(clap::value_parser!(PathBuf))
                        .conflicts_with("native")
                        .required_unless_present("native"),
                )
                .arg(
                    arg!(--native <PATH> "Path to a native binary")
                        .value_parser(clap::value_parser!(PathBuf))
                        .conflicts_with("wasm")
                        .required_unless_present("wasm"),
                )
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
                    arg!(--workers [WORKERS] "Number of worker threads")
                        .default_value("1")
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    arg!(--output [OUTPUT_FOLDER] "If set, the data for nonce will be saved as '<nonce>.json' in this folder")
                        .value_parser(clap::value_parser!(PathBuf)),
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
            match (sub_m.get_one::<PathBuf>("wasm"), sub_m.get_one::<PathBuf>("native")) {
                (Some(wasm), None) => Binary::Wasm(wasm.clone()),
                (None, Some(native)) => Binary::Native(native.clone()),
                _ => unreachable!("clap ensures one of wasm or native is present")
            },
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
            match (sub_m.get_one::<PathBuf>("wasm"), sub_m.get_one::<PathBuf>("native")) {
                (Some(wasm), None) => Binary::Wasm(wasm.clone()),
                (None, Some(native)) => Binary::Native(native.clone()),
                _ => unreachable!("clap ensures one of wasm or native is present")
            },
            *sub_m.get_one::<u64>("mem").unwrap(),
            *sub_m.get_one::<u64>("fuel").unwrap(),
            *sub_m.get_one::<usize>("workers").unwrap(),
            sub_m.get_one::<PathBuf>("output").cloned(),
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
    binary: Binary,
    max_memory: u64,
    max_fuel: u64,
) -> Result<()> {
    let settings = load_settings(&settings);
    if let Binary::Wasm(wasm_path) = binary {
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
        } else if output_data.solution.as_ref().unwrap().len() == 0 {
            return Err(anyhow!("No solution found"));
        }
        return worker::verify_solution(&settings, &rand_hash, nonce, &output_data.solution.as_ref().unwrap())
            .map_err(|e| anyhow!("Invalid solution: {}", e))
    }

    if let Binary::Native(native_path) = binary {
        macro_rules! challenge_match {
            ($seed:expr, $difficulty:expr, [$(($id:literal, $challenge:ident)),*]) => {
                match settings.challenge_id.as_str() {
                    $($id => {
                        let challenge = $challenge::generate_instance_from_vec($seed, $difficulty)
                            .map_err(|e| anyhow!("Failed to generate challenge: {}", e))?;
                        let type_name = stringify!($challenge).to_lowercase();
                        let type_name = type_name.strip_suffix("challenge").unwrap_or(&type_name);
                        (type_name.to_string(), jsonify(&challenge))
                    },)*
                    _ => return Err(anyhow!("Unknown challenge type"))
                }
            }
        }

        let seed = settings.calc_seed(&rand_hash, nonce);
        let (challenge_type, challenge_json) = challenge_match!(seed, &settings.difficulty, [
            ("c001", SatisfiabilityChallenge),
            ("c002", VehicleRoutingChallenge),
            ("c003", KnapsackChallenge),
            ("c004", VectorSearchChallenge)
        ]);

        let wrapper_path = std::env::current_exe()?
            .parent()
            .ok_or_else(|| anyhow!("Failed to get executable directory"))?
            .join("tig-native-wrapper");

        let stdlib_path = get_rust_stdlib_path("nightly-2025-01-16")?;

        let mut process = std::process::Command::new(wrapper_path)
            .env("LD_LIBRARY_PATH", format!("{}:{}", stdlib_path.display(), std::env::var("LD_LIBRARY_PATH").unwrap_or_default()))
            .arg(native_path)
            .arg(challenge_type)
            .arg(max_fuel.to_string())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        if let Some(mut stdin) = process.stdin.take() {
            use std::io::Write;
            stdin.write_all(challenge_json.as_bytes())?;
        }

        let output = process.wait_with_output()?;

        if !output.status.success() {
            if output.status.code() == Some(87) {
                let mut rt_sig = 0;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let mut lines = stdout.lines().rev();
                while let Some(line) = lines.next() {
                    if line.starts_with("Runtime signature: ") {
                        if let Some(sig) = line.strip_prefix("Runtime signature: ") {
                            if let Ok(sig) = sig.trim().parse::<u64>() {
                                rt_sig = sig;
                                break
                            }
                        }
                    }
                }

                let output_data = OutputData {
                    nonce,
                    solution: None,
                    fuel_consumed: max_fuel,
                    runtime_signature: rt_sig,
                };

                println!("{}", jsonify(&output_data));
                
                return Err(anyhow!("Ran out of fuel, runtime signature: {}", rt_sig));
            }

            return Err(anyhow!("Native wrapper failed: {}", String::from_utf8_lossy(&output.stderr)));
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        let mut output_data: OutputData = dejsonify(&output_str)?;
        output_data.nonce = nonce;
        println!("{}", jsonify(&output_data));

        return worker::verify_solution(&settings, &rand_hash, nonce, &output_data.solution.as_ref().unwrap())
            .map_err(|e| anyhow!("Invalid solution: {}", e));
    }

    Err(anyhow!("Invalid binary type"))
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

pub enum Binary
{
    Native(PathBuf),
    Wasm(PathBuf),
}

fn compute_batch(
    settings: String,
    rand_hash: String,
    start_nonce: u64,
    num_nonces: u64,
    batch_size: u64,
    binary: Binary,
    max_memory: u64,
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

    let settings = Arc::new(load_settings(&settings));
    if let Binary::Wasm(wasm_path) = binary {
        let wasm = Arc::new(load_wasm(&wasm_path));
        let runtime = Runtime::new()?;

        return runtime.block_on(async {
            let mut hashes = vec![MerkleHash::null(); num_nonces as usize];
            let mut solution_nonces = Vec::new();

            let results = stream::iter(start_nonce..(start_nonce + num_nonces))
                .map(|nonce| {
                    let settings = Arc::clone(&settings);
                    let wasm = Arc::clone(&wasm);
                    let rand_hash = rand_hash.clone();
                    let output_folder = output_folder.clone();

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
                                &output_data.solution.as_ref().unwrap(),
                            )
                            .is_ok();
                        let hash = MerkleHash::from(output_data.clone());

                        Ok::<(u64, MerkleHash, bool, Option<OutputData>), anyhow::Error>((
                            nonce,
                            hash,
                            is_solution,
                            output_folder.is_some().then(|| output_data),
                        ))
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
        });
    }

    if let Binary::Native(path) = binary {
        macro_rules! challenge_match {
            ($seed:expr, $difficulty:expr, [$(($id:literal, $challenge:ident)),*]) => {
                match settings.challenge_id.as_str() {
                    $($id => {
                        let challenge = $challenge::generate_instance_from_vec($seed, $difficulty)
                            .map_err(|e| anyhow!("Failed to generate challenge: {}", e))?;
                        let type_name = stringify!($challenge).to_lowercase();
                        let type_name = type_name.strip_suffix("challenge").unwrap_or(&type_name);
                        (type_name.to_string(), jsonify(&challenge))
                    },)*
                    _ => return Err(anyhow!("Unknown challenge type"))
                }
            }
        }
        
        let mut hashes = vec![MerkleHash::null(); num_nonces as usize];
        let mut solution_nonces = Vec::new();
        let mut dump = Vec::new();

        let wrapper_path = std::env::current_exe()?
            .parent()
            .ok_or_else(|| anyhow!("Failed to get executable directory"))?
            .join("tig-native-wrapper");

        let stdlib_path = get_rust_stdlib_path("nightly-2025-01-16")?;

        let mut processes = Vec::new();
        let mut current_nonce = start_nonce;

        while current_nonce < start_nonce + num_nonces || !processes.is_empty() 
        {
            while processes.len() < num_workers && current_nonce < start_nonce + num_nonces 
            {
                let seed = settings.calc_seed(&rand_hash, current_nonce);
                let (challenge_type, challenge_json) = challenge_match!(seed, &settings.difficulty, [
                    ("c001", SatisfiabilityChallenge),
                    ("c002", VehicleRoutingChallenge),
                    ("c003", KnapsackChallenge),
                    ("c004", VectorSearchChallenge)
                ]);

                let mut process = std::process::Command::new(&wrapper_path)
                    .env("LD_LIBRARY_PATH", format!("{}:{}", stdlib_path.display(), std::env::var("LD_LIBRARY_PATH").unwrap_or_default()))
                    .arg(&path)
                    .arg(&challenge_type)
                    .arg(max_fuel.to_string())
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()?;

                if let Some(mut stdin) = process.stdin.take() {
                    use std::io::Write;
                    stdin.write_all(challenge_json.as_bytes())?;
                }

                processes.push((current_nonce, process));
                current_nonce += 1
            }

            let mut finished_idx = None;
            for (idx, (_, process)) in processes.iter_mut().enumerate() 
            {
                match process.try_wait() 
                {
                    Ok(Some(_)) => {
                        finished_idx = Some(idx);
                        break
                    }
                    Ok(None) => continue,
                    Err(e) => return Err(anyhow!("Failed to wait on process: {}", e))
                }
            }

            if let Some(idx) = finished_idx 
            {
                let (nonce, mut process) = processes.remove(idx);
                let output = process.wait_with_output()?;

                if !output.status.success() 
                {
                    if output.status.code() == Some(87) 
                    {
                        let mut rt_sig = 0;
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let mut lines = stdout.lines().rev();
                        while let Some(line) = lines.next() 
                        {
                            if line.starts_with("Runtime signature: ") 
                            {
                                if let Some(sig) = line.strip_prefix("Runtime signature: ") 
                                {
                                    if let Ok(sig) = sig.trim().parse::<u64>() 
                                    {
                                        rt_sig = sig;
                                        break
                                    }
                                }
                            }
                        }

                        let output_data = OutputData {
                            nonce,
                            solution: None,
                            fuel_consumed: max_fuel,
                            runtime_signature: rt_sig,
                        };

                        let hash = MerkleHash::from(output_data.clone());
                        hashes[(nonce - start_nonce) as usize] = hash;

                        if output_folder.is_some() 
                        {
                            dump.push(output_data)
                        }
                        
                        println!("Ran out of fuel, runtime signature: {}", rt_sig);
                        continue
                    }

                    return Err(anyhow!("Native wrapper failed: {}", String::from_utf8_lossy(&output.stderr)))
                }

                let output_str = String::from_utf8_lossy(&output.stdout);
                let mut output_data: OutputData = match dejsonify(&output_str) 
                {
                    Ok(data) => data,
                    Err(e) => {
                        println!("Failed to parse output: {}", e);
                        continue
                    }
                };

                output_data.nonce = nonce;

                if output_data.solution.is_some() 
                {
                    let is_solution = worker::verify_solution(
                        &settings,
                        &rand_hash,
                        nonce,
                        &output_data.solution.as_ref().unwrap(),
                    ).is_ok();

                    if is_solution 
                    {
                        solution_nonces.push(nonce)
                    }
                }

                let hash = MerkleHash::from(output_data.clone());
                hashes[(nonce - start_nonce) as usize] = hash;

                if output_folder.is_some() 
                {
                    dump.push(output_data)
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(10))
            }
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
        return Ok(());
    }

    Err(anyhow!("Invalid binary type"))
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

fn get_rust_stdlib_path(toolchain: &str) -> Result<PathBuf> {
    let output = std::process::Command::new("rustc")
        .arg(format!("+{}", toolchain))
        .arg("--print")
        .arg("target-libdir")
        .output()?;

    if !output.status.success()
    {
        return Err(anyhow!("Failed to get rust stdlib path"));
    }

    let path = String::from_utf8(output.stdout)?;
    Ok(PathBuf::from(path.trim()))
}