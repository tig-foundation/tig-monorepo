mod benchmarker;
mod utils;
use benchmarker::{Job, State};
use clap::{value_parser, Arg, Command};
use std::collections::HashMap;
use tig_structs::core::*;
use tig_utils::{dejsonify, get, jsonify, post};
use tokio::{spawn, task::yield_now};
use utils::{sleep, time, Result};
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
            Arg::new("SELECTED_ALGORITHMS")
                .help("Json string with your algorithm selection")
                .required(true)
                .value_parser(value_parser!(String)),
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
                .help("(Optional) Set duration (in milliseconds) of a benchmark")
                .default_value("10000")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("delay")
                .long("delay")
                .help("(Optional) Set delay (in milliseconds) between benchmark finishing and submitting")
                .default_value("5000")
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
}

#[tokio::main]
async fn main() {
    let matches = cli().get_matches();

    let selected_algorithms = matches
        .get_one::<String>("SELECTED_ALGORITHMS")
        .unwrap()
        .clone();
    let num_workers = *matches.get_one::<u32>("workers").unwrap();
    let port = *matches.get_one::<u16>("port").unwrap();
    let duration = *matches.get_one::<u32>("duration").unwrap();
    let delay = *matches.get_one::<u32>("delay").unwrap();
    let api_url = matches.get_one::<String>("api").unwrap().clone();
    let api_key = matches.get_one::<String>("API_KEY").unwrap().clone();
    let player_id = matches.get_one::<String>("PLAYER_ID").unwrap().clone();
    if let Some(master) = matches.get_one::<String>("master") {
        slave(
            api_url,
            api_key,
            player_id,
            master,
            port,
            num_workers,
            selected_algorithms,
        )
        .await;
    } else {
        master(
            api_url,
            api_key,
            player_id,
            num_workers,
            duration,
            delay,
            selected_algorithms,
            port,
        )
        .await
    }
}

async fn slave(
    api_url: String,
    api_key: String,
    player_id: String,
    master: &String,
    port: u16,
    num_workers: u32,
    selected_algorithms: String,
) {
    println!("[slave] setting up");
    benchmarker::setup(api_url, api_key, player_id).await;
    println!(
        "[slave] parsing selected algorithms from: {:?}",
        selected_algorithms
    );
    let selected_algorithms = serde_json::from_str::<HashMap<String, String>>(&selected_algorithms)
        .unwrap_or_else(|err| panic!("Failed to parse {:?}: {}", selected_algorithms, err));

    async fn do_work(
        master_url: &String,
        num_workers: u32,
        selected_algorithms: &HashMap<String, String>,
    ) -> Result<()> {
        println!("[slave] fetching jobs from master at: {}", master_url);
        let available_jobs = match get::<String>(&format!("{}/jobs", master_url), None).await {
            Ok(resp) => dejsonify::<HashMap<String, Job>>(&resp).unwrap(),
            Err(e) => {
                return Err(format!("failed to fetch jobs from master: {:?}", e));
            }
        };
        println!("[slave] fetched {} jobs", available_jobs.len());
        for (i, job) in available_jobs.values().enumerate() {
            println!(
                "[slave] job {}: {:?}, weight: {}",
                i, job.settings, job.weight
            );
        }
        println!(
            "[slave] filtering jobs that match selected_algorithms: {:?}",
            selected_algorithms
        );
        let now = time();
        let filtered_jobs: HashMap<String, Job> = available_jobs
            .into_iter()
            .filter(|(benchmark_id, job)| {
                now < job.timestamps.end
                    && selected_algorithms
                        .iter()
                        .any(|(challenge_name, algorithm_name)| {
                            benchmark_id
                                .starts_with(&format!("{}_{}", challenge_name, algorithm_name))
                        })
            })
            .collect();

        if filtered_jobs.len() == 0 {
            return Err("no jobs matching selected_algorithms".to_string());
        }

        let job = benchmarker::select_job::execute(&filtered_jobs).await?;
        println!(
            "[slave]: downloading algorithm {}",
            job.download_url.split("/").last().unwrap()
        );
        let wasm = benchmarker::download_wasm::execute(&job).await?;

        println!("[slave]: starting benchmark {:?}", job.settings);
        let nonce_iterators = job.create_nonce_iterators(num_workers);
        benchmarker::run_benchmark::execute(nonce_iterators.clone(), &job, &wasm).await;
        let start = time();
        while time() < job.timestamps.end {
            {
                let mut num_attempts = 0;
                for nonce_iterator in &nonce_iterators {
                    num_attempts += nonce_iterator.lock().await.num_attempts();
                }
                let num_solutions = job.solutions_data.lock().await.len() as u32;
                let elapsed = time() - job.timestamps.start;
                println!(
                    "[slave]: #solutions: {}, #instances: {}, elapsed: {}ms",
                    num_solutions, num_attempts, elapsed
                );
            }
            sleep(500).await;
        }
        let elapsed = time() - start;
        if elapsed < 2500 {
            println!("[slave]: sleeping for {}ms", 2500 - elapsed);
            sleep(2500 - elapsed).await;
        }

        let mut num_attempts = 0;
        for nonce_iterator in &nonce_iterators {
            num_attempts += nonce_iterator.lock().await.num_attempts();
        }
        if num_attempts > 0 {
            let solutions_data = &*job.solutions_data.lock().await;
            println!(
                "[slave]: posting {} solutions to master",
                solutions_data.len()
            );
            post::<String>(
                &format!("{}/solutions_data/{}", master_url, job.benchmark_id),
                &jsonify(&solutions_data),
                Some(vec![(
                    "Content-Type".to_string(),
                    "application/json".to_string(),
                )]),
            )
            .await
            .map_err(|e| format!("failed to post solutions to master: {:?}", e))?;
        }

        Ok(())
    }

    let master_url = format!("http://{}:{}", master, port);
    loop {
        if let Err(e) = do_work(&master_url, num_workers, &selected_algorithms).await {
            println!("[slave]: error: {:?}", e);
            println!("[slave]: sleeping 5s");
            sleep(5000).await;
        }
        yield_now().await;
    }
}

async fn master(
    api_url: String,
    api_key: String,
    player_id: String,
    num_workers: u32,
    duration: u32,
    delay: u32,
    selected_algorithms: String,
    port: u16,
) {
    println!(
        "[master] parsing selected algorithms from: {:?}",
        selected_algorithms
    );
    let selected_algorithms = serde_json::from_str::<HashMap<String, String>>(&selected_algorithms)
        .unwrap_or_else(|err| panic!("Failed to parse {:?}: {}", selected_algorithms, err));
    println!("[master] setting up");
    benchmarker::setup(api_url, api_key, player_id).await;
    println!("[master] starting data_fetcher");
    spawn(async {
        benchmarker::data_fetcher().await;
    });
    println!("[master] starting benchmark_submitter");
    spawn(async {
        benchmarker::benchmark_submitter().await;
    });
    println!("[master] starting proof_submitter");
    spawn(async {
        benchmarker::proof_submitter().await;
    });
    println!("[master] starting benchmarker");
    spawn(async move {
        benchmarker::benchmarker(
            selected_algorithms,
            num_workers,
            duration as u64,
            delay as u64,
        )
        .await;
    });
    println!("[master] starting webserver on port {}", port);
    spawn(async move {
        let get_jobs = warp::path("jobs").and(warp::get()).and_then(|| async {
            println!("[master] slave node fetching jobs",);
            let state = (*benchmarker::state()).lock().await;
            Ok::<_, warp::Rejection>(warp::reply::json(&state.available_jobs))
        });
        let post_solutions_data = warp::path!("solutions_data" / String)
            .and(warp::post())
            .and(warp::body::json())
            .and_then(
                |benchmark_id: String, solutions_data: HashMap<u64, SolutionData>| async move {
                    {
                        let num_solutions = solutions_data.len() as u32;
                        println!("[master] received {} solutions from slave", num_solutions,);
                        let mut state = (*benchmarker::state()).lock().await;
                        let State {
                            available_jobs,
                            pending_benchmark_jobs,
                            difficulty_samplers,
                            ..
                        } = &mut *state;
                        if let Some(job) = available_jobs
                            .get_mut(&benchmark_id)
                            .or_else(|| pending_benchmark_jobs.get_mut(&benchmark_id))
                        {
                            println!("[master] adding solutions to benchmark {:?}", job.settings,);
                            difficulty_samplers
                                .get_mut(&job.settings.challenge_id)
                                .unwrap()
                                .update_with_solutions(&job.settings.difficulty, num_solutions);
                            job.solutions_data.lock().await.extend(solutions_data);
                        } else {
                            println!("[master] failed to find benchmark to add solutions to",);
                        }
                    }
                    Ok::<_, warp::Rejection>(warp::reply::with_status(
                        "solutions received",
                        warp::http::StatusCode::OK,
                    ))
                },
            );
        warp::serve(get_jobs.or(post_solutions_data))
            .run(([0, 0, 0, 0], port))
            .await;
    });
    loop {
        sleep(30000).await;
    }
}
