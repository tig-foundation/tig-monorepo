use anyhow::{anyhow, Result};
use clap::{arg, ArgAction, Command};
use libloading::Library;
use std::{fs, panic, path::PathBuf};
use tig_challenges::*;
use tig_structs::core::{BenchmarkSettings, CPUArchitecture, OutputData, Solution};
use tig_utils::{compress_obj, dejsonify, jsonify};
#[cfg(feature = "cuda")]
use {
    cudarc::{
        driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
        runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
    },
    std::sync::Arc,
};

fn cli() -> Command {
    Command::new("tig-runtime")
        .about("Executes an algorithm on a single challenge instance")
        .arg_required_else_help(true)
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
            arg!(<BINARY> "Path to a shared object (*.so) file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--ptx [PTX] "Path to a CUDA ptx file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--fuel [FUEL] "Optional maximum fuel parameter")
                .default_value("2000000000")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            arg!(--output [OUTPUT_FILE] "If set, the output data will be saved to this file path (default json)")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--compress [COMPRESS] "If output file is set, the output data will be compressed as zlib")
            .action(ArgAction::SetTrue)
        )
        .arg(
            arg!(--gpu [GPU] "Which GPU device to use")
                .default_value("0")
                .value_parser(clap::value_parser!(usize)),
        )
}

fn main() {
    let matches = cli().get_matches();

    if let Err(e) = compute_solution(
        matches.get_one::<String>("SETTINGS").unwrap().clone(),
        matches.get_one::<String>("RAND_HASH").unwrap().clone(),
        *matches.get_one::<u64>("NONCE").unwrap(),
        matches.get_one::<PathBuf>("BINARY").unwrap().clone(),
        matches.get_one::<PathBuf>("ptx").cloned(),
        *matches.get_one::<u64>("fuel").unwrap(),
        matches.get_one::<PathBuf>("output").cloned(),
        matches.get_one::<bool>("compress").unwrap().clone(),
        matches.get_one::<usize>("gpu").unwrap().clone(),
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

pub fn compute_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    library_path: PathBuf,
    ptx_path: Option<PathBuf>,
    max_fuel: u64,
    output_file: Option<PathBuf>,
    compress: bool,
    gpu_device: usize,
) -> Result<()> {
    let settings = load_settings(&settings);

    let library = load_module(&library_path)?;
    let fuel_remaining_ptr = unsafe { *library.get::<*mut u64>(b"__fuel_remaining")? };
    unsafe { *fuel_remaining_ptr = max_fuel };
    let seed = settings.calc_seed(&rand_hash, nonce);

    let mut solution = Solution::new();
    let mut err_msg = Option::<String>::None;
    let mut fuel_consumed = 0;
    let mut runtime_signature = 0;

    macro_rules! dispatch_challenges {
        ( $( ($c:ident, $cpu_or_gpu:tt) ),+ $(,)? ) => {{
            match settings.challenge_id.as_str() {
                $(
                    stringify!($c) => {
                        dispatch_challenges!(@expand $c, $cpu_or_gpu);
                    }
                )+
                _ => panic!("Unsupported challenge"),
            }
        }};

        (@expand $c:ident, cpu) => {{
            let solve_challenge_fn = unsafe {
                library.get::<fn(&$c::Challenge) -> Result<Option<$c::Solution>, String>>(b"entry_point")?
            };

            let challenge = $c::Challenge::generate_instance(
                &seed,
                &settings.difficulty.into(),
            ).unwrap();

            match solve_challenge_fn(&challenge) {
                Ok(Some(s)) => {
                    match challenge.verify_solution(&s) {
                        Ok(_) => {
                            solution = serde_json::to_value(s)
                                .unwrap()
                                .as_object()
                                .unwrap()
                                .to_owned();
                        }
                        Err(e) => err_msg = Some(e.to_string()),
                    }
                }
                Ok(None) => {}
                Err(e) => err_msg = Some(e),
            }
        }};

        (@expand $c:ident, gpu) => {{
            #[cfg(not(feature = "cuda"))]
            panic!("tig-runtime was not compiled with '--features cuda'");

            #[cfg(feature = "cuda")]
            {
                if ptx_path.is_none() {
                    panic!("PTX file is required for GPU challenges.");
                }
                let ptx_path = ptx_path.unwrap();
                let solve_challenge_fn = unsafe {
                    library.get::<fn(
                        &$c::Challenge,
                        Arc<CudaModule>,
                        Arc<CudaStream>,
                        &cudaDeviceProp
                    ) -> Result<Option<$c::Solution>, String>>(
                        b"entry_point",
                    )?
                };

                let ptx_content = std::fs::read_to_string(&ptx_path)
                    .map_err(|e| anyhow!("Failed to read PTX file: {}", e))?;
                let max_fuel_hex = format!("0x{:016x}", max_fuel);
                let modified_ptx = ptx_content.replace("0xdeadbeefdeadbeef", &max_fuel_hex);

                let ptx = Ptx::from_src(modified_ptx);
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
                ).unwrap();

                let initialize_kernel = module.load_function("initialize_kernel")?;

                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };

                let mut builder = stream.launch_builder(&initialize_kernel);
                unsafe { builder.launch(cfg)?; }

                match solve_challenge_fn(&challenge, module.clone(), stream.clone(), &prop) {
                    Ok(Some(s)) => {
                        stream.synchronize()?;
                        ctx.synchronize()?;

                        let mut fuel_usage = stream.alloc_zeros::<u64>(1)?;
                        let mut signature = stream.alloc_zeros::<u64>(1)?;
                        let mut error_stat = stream.alloc_zeros::<u64>(1)?;

                        let finalize_kernel = module.load_function("finalize_kernel")?;

                        let cfg = LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        };

                        let mut builder = stream.launch_builder(&finalize_kernel);
                        unsafe {
                            builder
                                .arg(&mut fuel_usage)
                                .arg(&mut signature)
                                .arg(&mut error_stat)
                                .launch(cfg)?;
                        }

                        if stream.memcpy_dtov(&error_stat)?[0] != 0 {
                            fuel_consumed = max_fuel + 1;
                            runtime_signature = 0;
                        } else {
                            fuel_consumed = stream.memcpy_dtov(&fuel_usage)?[0];
                            runtime_signature = stream.memcpy_dtov(&signature)?[0];
                        }
                        match challenge.verify_solution(&s, module.clone(), stream.clone(), &prop) {
                            Ok(_) => {
                                solution = serde_json::to_value(s)
                                    .unwrap()
                                    .as_object()
                                    .unwrap()
                                    .to_owned();
                            }
                            Err(e) => err_msg = Some(e.to_string()),
                        }
                    }
                    Ok(None) => {}
                    Err(e) => err_msg = Some(e),
                }
            }
        }};
    }
    dispatch_challenges!(
        (c001, cpu),
        (c002, cpu),
        (c003, cpu),
        (c004, gpu),
        (c005, gpu)
    );

    fuel_consumed += max_fuel - unsafe { **library.get::<*const u64>(b"__fuel_remaining")? };
    if fuel_consumed > max_fuel {
        fuel_consumed = max_fuel + 1;
        runtime_signature = 0;
    } else {
        runtime_signature ^= unsafe { **library.get::<*const u64>(b"__runtime_signature")? };
    }

    let output_data = OutputData {
        nonce,
        runtime_signature,
        fuel_consumed,
        solution,
        #[cfg(target_arch = "x86_64")]
        cpu_arch: CPUArchitecture::AMD64,
        #[cfg(target_arch = "aarch64")]
        cpu_arch: CPUArchitecture::AARCH64,
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
        eprintln!("Error: {}", err_msg);
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
