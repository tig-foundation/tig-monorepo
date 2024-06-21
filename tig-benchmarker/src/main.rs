// #[cfg(any(not(feature = "standalone"), feature = "browser"))]
// compile_error!("to build the binary use `--no-default-features --features standalone`");

mod benchmarker;
mod future_utils;
use benchmarker::{Job, NonceIterator};
use clap::{value_parser, Arg, Command};
use future_utils::{sleep, Mutex};
use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};
use tig_structs::core::*;
use tig_utils::{dejsonify, get, jsonify, post};
use warp::Filter;

fn cli() -> Command {
    Command::new("TIG Benchmarker")
        .about("Standalone benchmarker")
        .arg_required_else_help(true)
        .arg(
            Arg::new("PLAYER_ID")
                .help("Your wallet address")
                .required(true)
                .value_parser(value_parser!(String)),
        )
        .arg(
            Arg::new("API_KEY")
                .help("Your API Key")
                .required(true)
                .value_parser(value_parser!(String)),
        )
        .arg(
            Arg::new("ALGORITHMS_SELECTION")
                .help("Path to json file with your algorithm selection")
                .required(true)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("workers")
                .long("workers")
                .help("(Optional) Set number of workers")
                .default_value("4")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("duration")
                .long("duration")
                .help("(Optional) Set duration of a benchmark in milliseconds")
                .default_value("7500")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("api")
                .long("api")
                .help("(Optional) Set api_url")
                .default_value("https://mainnet-api.tig.foundation")
                .value_parser(value_parser!(String)),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .help("(Optional) Set port for cluster communication")
                .default_value("5115")
                .value_parser(value_parser!(u16)),
        )
        .arg(
            Arg::new("master")
                .long("master")
                .help("(Optional) Set hostname for master node to connect to")
                .value_parser(value_parser!(String)),
        )
        .arg(
            Arg::new("offset")
                .long("offset")
                .help("(Optional) Set nonce offset for each slave")
                .default_value("5000000")
                .value_parser(value_parser!(u32)),
        )
}

#[tokio::main]
async fn main() {
    let matches = cli().get_matches();

    let algorithms_path = matches.get_one::<PathBuf>("ALGORITHMS_SELECTION").unwrap();
    let num_workers = *matches.get_one::<u32>("workers").unwrap();
    let port = *matches.get_one::<u16>("port").unwrap();
    let duration = *matches.get_one::<u32>("duration").unwrap();
    let api_url = matches.get_one::<String>("api").unwrap().clone();
    let api_key = matches.get_one::<String>("API_KEY").unwrap().clone();
    let player_id = matches.get_one::<String>("PLAYER_ID").unwrap().clone();
    let nonce_offset = matches.get_one::<u32>("offset").unwrap().clone();
    if let Some(master) = matches.get_one::<String>("master") {
        slave_node(master, port, num_workers).await;
    } else {
        master_node(
            api_url,
            api_key,
            player_id,
            num_workers,
            duration,
            algorithms_path,
            port,
            nonce_offset,
        )
        .await
    }
}

async fn slave_node(master: &String, port: u16, num_workers: u32) {
    let master_url = format!("http://{}:{}", master, port);
    let mut job: Option<Job> = None;
    let mut nonce_iters: Vec<Arc<Mutex<NonceIterator>>> = Vec::new();
    let solutions_data = Arc::new(Mutex::new(Vec::<SolutionData>::new()));
    let solutions_count = Arc::new(Mutex::new(0u32));
    let mut num_solutions = 0;
    loop {
        let next_job = match get::<String>(&format!("{}/job", master_url), None).await {
            Ok(resp) => dejsonify::<Option<Job>>(&resp).unwrap(),
            Err(e) => {
                println!("Error getting job: {:?}", e);
                sleep(5000).await;
                continue;
            }
        };

        if job != next_job {
            println!("Ending job");

            for nonce_iter in nonce_iters.iter() {
                (*(*nonce_iter).lock().await).empty();
            }
            nonce_iters.clear();
            {
                (*solutions_data).lock().await.clear();
                *(*solutions_count).lock().await = 0;
                num_solutions = 0;
            }
            if next_job
                .as_ref()
                .is_some_and(|x| x.sampled_nonces.is_none())
            {
                let job = next_job.as_ref().unwrap();
                println!("Starting new job: {:?}", job);
                println!(
                    "Downloading algorithm {}",
                    job.download_url.split("/").last().unwrap()
                );
                let wasm = match benchmarker::download_wasm::execute(job).await {
                    Ok(wasm) => wasm,
                    Err(e) => {
                        println!("Error downloading wasm: {:?}", e);
                        sleep(5000).await;
                        continue;
                    }
                };

                println!("Getting nonce offset from master");
                let offset = match get::<String>(
                    &format!("{}/nonce_offset/{:?}", master_url, hostname::get().unwrap()),
                    None,
                )
                .await
                {
                    Ok(resp) => dejsonify::<u32>(&resp).unwrap(),
                    Err(e) => {
                        println!("Error getting nonce offset: {:?}", e);
                        sleep(5000).await;
                        continue;
                    }
                };
                println!("Got nonce offset: {}", offset);

                // variables that are shared by workers
                nonce_iters = (0..num_workers)
                    .into_iter()
                    .map(|x| {
                        Arc::new(Mutex::new(NonceIterator::from_u32(
                            offset + u32::MAX / num_workers * x,
                        )))
                    })
                    .collect();
                println!("Starting benchmark");
                benchmarker::run_benchmark::execute(
                    nonce_iters.iter().cloned().collect(),
                    job,
                    &wasm,
                    solutions_data.clone(),
                    solutions_count.clone(),
                )
                .await;
            }

            job = next_job;
        }
        if job.as_ref().is_some_and(|x| x.sampled_nonces.is_none()) {
            let job = job.as_ref().unwrap();
            let mut solutions_data = solutions_data.lock().await;
            let n = solutions_data.len();
            if n > 0 {
                num_solutions += n as u32;
                let data: Vec<SolutionData> = solutions_data.drain(..).collect();
                println!("Posting {} solutions", n);
                if let Err(e) = post::<String>(
                    &format!("{}/solutions_data/{}", master_url, job.benchmark_id),
                    &jsonify(&data),
                    Some(vec![(
                        "Content-Type".to_string(),
                        "application/json".to_string(),
                    )]),
                )
                .await
                {
                    println!("Error posting solutions data: {:?}", e);
                    sleep(5000).await;
                    continue;
                }
            }
            let mut num_attempts = 0;
            for nonce_iter in nonce_iters.iter().cloned() {
                let nonce_iter = (*nonce_iter).lock().await;
                num_attempts += nonce_iter.attempts();
            }
            println!(
                "Computed {} solutions out of {} instances",
                num_solutions, num_attempts
            );
            sleep(200).await;
        } else {
            println!("No job, sleeping 1s");
            sleep(1000).await;
        }
    }
}

async fn master_node(
    api_url: String,
    api_key: String,
    player_id: String,
    num_workers: u32,
    duration: u32,
    algorithms_path: &PathBuf,
    port: u16,
    nonce_offset: u32,
) {
    benchmarker::setup(api_url, api_key, player_id).await;
    benchmarker::start(num_workers, duration).await;
    future_utils::spawn(async move {
        let offsets = Arc::new(Mutex::new(HashMap::new()));
        let get_nonce_offset = warp::path!("nonce_offset" / String)
            .and(warp::get())
            .and(warp::any().map(move || offsets.clone()))
            .and_then(
                move |slave_id: String, offsets: Arc<Mutex<HashMap<String, u32>>>| async move {
                    let offsets = &mut (*offsets).lock().await;
                    let len = offsets.len() as u32;
                    let o = offsets
                        .entry(slave_id)
                        .or_insert_with(|| (len + 1) * nonce_offset);
                    Ok::<_, warp::Rejection>(warp::reply::json(&o))
                },
            );
        let get_job = warp::path("job").and(warp::get()).and_then(|| async {
            let state = (*benchmarker::state()).lock().await;
            Ok::<_, warp::Rejection>(warp::reply::json(&state.job))
        });
        let post_solutions_data = warp::path!("solutions_data" / String)
            .and(warp::post())
            .and(warp::body::json())
            .and_then(
                |benchmark_id: String, mut solutions_data: Vec<SolutionData>| async move {
                    benchmarker::drain_solutions(&benchmark_id, &mut solutions_data).await;
                    Ok::<_, warp::Rejection>(warp::reply::with_status(
                        "SolutionsData received",
                        warp::http::StatusCode::OK,
                    ))
                },
            );
        warp::serve(get_nonce_offset.or(get_job).or(post_solutions_data))
            .run(([127, 0, 0, 1], port))
            .await;
    });
    loop {
        let selection = serde_json::from_str::<HashMap<String, String>>(
            &fs::read_to_string(algorithms_path).unwrap(),
        )
        .unwrap();
        for (challenge_id, algorithm_id) in selection {
            benchmarker::select_algorithm(challenge_id, algorithm_id).await;
        }
        future_utils::sleep(10000).await;
    }
}
