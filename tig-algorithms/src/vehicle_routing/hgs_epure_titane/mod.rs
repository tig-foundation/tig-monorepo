use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::vehicle_routing::*;

thread_local! {
    static SAMPLED_DIVERSITY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static COMPACT_REUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SEED_STALL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

mod part_a;
mod part_b;

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    SAMPLED_DIVERSITY.with(|enabled| {
        let previous = enabled.replace(challenge.num_nodes == 800);
        let result = COMPACT_REUSE.with(|reuse| SEED_STALL.with(|stall| {
            let previous_reuse = reuse.replace(challenge.num_nodes == 1000);
            let previous_stall = stall.replace(0);
            let result = match challenge.num_nodes {
                700 => {
                    let latest = std::cell::RefCell::new(None::<Solution>);
                    let baseline = std::cell::Cell::new(None::<u32>);
                    let save_seed = |solution: &Solution| -> Result<()> {
                        let result = save_solution(solution);
                        *latest.borrow_mut() = Some(solution.clone());
                        let greedy = match baseline.get() {
                            Some(distance) => distance,
                            None => {
                                let distance =
                                    part_a::quality::greedy_total_distance(challenge)?;
                                baseline.set(Some(distance));
                                distance
                            }
                        };
                        let distance = challenge.evaluate_total_distance(solution)?;
                        println!(
                            "DIAG quality={}",
                            part_a::quality::quality_of(greedy, distance)
                        );
                        result
                    };
                    part_b::solve_challenge(challenge, &save_seed, hyperparameters)
                        .and_then(|()| {
                            match latest.into_inner() {
                                Some(seed) => {
                                    let greedy =
                                        part_a::quality::greedy_total_distance(challenge)?;
                                    part_a::solve_band_from_seed(
                                        challenge,
                                        seed,
                                        greedy,
                                        save_solution,
                                        hyperparameters,
                                    )
                                }
                                None => Ok(()),
                            }
                        })
                }
                900 => part_a::solve_challenge(challenge, save_solution, hyperparameters),
                _ => part_b::solve_challenge(challenge, save_solution, hyperparameters),
            };
            stall.set(previous_stall);
            reuse.set(previous_reuse);
            result
        }));
        enabled.set(previous);
        result
    })
}

#[allow(dead_code)]
pub fn help() {
    part_a::help();
    part_b::help();
}
