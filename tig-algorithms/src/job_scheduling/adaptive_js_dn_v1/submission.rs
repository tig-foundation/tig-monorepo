//! Complete the invariant donor search, then apply bounded monotone polish.
use anyhow::Result;
use serde_json::{Map,Value};
use std::cell::RefCell;
use tig_challenges::job_scheduling::{Challenge,Solution};
use super::invariant_core;

#[path = "submission_polish.rs"]
mod polish;

const MAX_POLISH_TRIALS: usize = 1024;

fn completion_time(c: &Challenge,s: &Solution) -> Option<u32> {
    if s.job_schedule.len()!=c.num_jobs { return None; }
    let mut job = 0;
    let mut makespan = 0;
    for (product,&count) in c.jobs_per_product.iter().enumerate() {
        let processing = &c.product_processing_times[product];
        for _ in 0..count {
            let row = s.job_schedule.get(job)?;
            if row.len()!=processing.len() { return None; }
            if let Some(&(machine,start)) = row.last() {
                let duration = *processing.last()?.get(&machine)?;
                makespan = makespan.max(start.checked_add(duration)?);
            }
            job += 1;
        }
    }
    if job!=c.num_jobs { return None; }
    Some(makespan)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution)->Result<()>,
    hyperparameters: &Option<Map<String,Value>>,
) -> Result<()> {
    let incumbent: RefCell<Option<(u32,Solution)>> = RefCell::new(None);
    let forward = |s: &Solution| -> Result<()> {
        // Forward every donor save unchanged and in its original order.
        save_solution(s)?;
        if let Some(objective) = completion_time(challenge,s) {
            let mut best = incumbent.borrow_mut();
            if best.as_ref().map_or(true,|(value,_)|objective<*value) {
                *best = Some((objective,s.clone()));
            }
        }
        Ok(())
    };
    invariant_core::solve_challenge(challenge,&forward,hyperparameters)?;
    // This phase starts only after the complete, unchanged donor trajectory.
    // The polish validator permits a save only for a strictly better schedule.
    if let Some((_,best)) = incumbent.into_inner() {
        if let Some(candidate) = polish::polish(challenge,&best,MAX_POLISH_TRIALS) {
            save_solution(&candidate)?;
        }
    }
    Ok(())
}

pub fn help() {
    println!("job_scheduling: invariant core with bounded critical-block polish");
}
