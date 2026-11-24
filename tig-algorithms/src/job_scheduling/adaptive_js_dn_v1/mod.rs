mod invariant_core;
mod submission;

pub fn solve_challenge(
    challenge: &tig_challenges::job_scheduling::Challenge,
    save_solution: &dyn Fn(&tig_challenges::job_scheduling::Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<serde_json::Map<String, serde_json::Value>>,
) -> anyhow::Result<()> {
    submission::solve_challenge(challenge, save_solution, hyperparameters)
}

pub fn help() { submission::help(); }
