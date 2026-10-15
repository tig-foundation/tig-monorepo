use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

#[path = "types.rs"]
mod types;
#[path = "preprocess.rs"]
mod preprocess;
#[path = "infra_shared.rs"]
mod infra_shared;
#[path = "flow_shop.rs"]
mod flow_shop;
#[path = "job_shop.rs"]
mod job_shop;
#[path = "ember_medium.rs"]
mod ember_medium;
#[path = "fjsp_medium.rs"]
mod fjsp_medium;
#[path = "ember_high.rs"]
mod ember_high;

fn track(hyperparameters: &Option<Map<String, Value>>) -> &str {
    hyperparameters
        .as_ref()
        .and_then(|map| map.get("track"))
        .and_then(Value::as_str)
        .unwrap_or("fjsp_high")
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    match track(hyperparameters) {
        "flow_shop" | "flow" => {
            let pre = preprocess::build_pre(challenge)?;
            flow_shop::solve(
                challenge,
                save_solution,
                &pre,
                &types::EffortConfig::default_effort(),
            )
        }
        "job_shop" | "job" => {
            let pre = preprocess::build_pre(challenge)?;
            job_shop::solve(
                challenge,
                save_solution,
                &pre,
                &types::EffortConfig::default_effort(),
            )
        }
        "fjsp_medium" | "medium" => {
            let pre = preprocess::build_pre(challenge)?;
            fjsp_medium::solve(
                challenge,
                save_solution,
                &pre,
                &types::EffortConfig::default_effort(),
            )
        }
        _ => ember_high::solve(challenge, save_solution),
    }
}
