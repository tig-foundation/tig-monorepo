// #[cfg(any(not(feature = "standalone"), feature = "browser"))]
// compile_error!("to build the binary use `--no-default-features --features standalone`");

mod benchmarker;
mod future_utils;
use clap::{value_parser, Arg, Command};
use std::{collections::HashMap, fs, path::PathBuf};

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
}

#[tokio::main]
async fn main() {
    let matches = cli().get_matches();

    let algorithms_path = matches.get_one::<PathBuf>("ALGORITHMS_SELECTION").unwrap();
    let num_workers = *matches.get_one::<u32>("workers").unwrap();
    let duration = *matches.get_one::<u32>("duration").unwrap();
    benchmarker::setup(
        matches.get_one::<String>("api").unwrap().clone(),
        matches.get_one::<String>("API_KEY").unwrap().clone(),
        matches.get_one::<String>("PLAYER_ID").unwrap().clone(),
    )
    .await;
    benchmarker::start(num_workers, duration).await;
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
