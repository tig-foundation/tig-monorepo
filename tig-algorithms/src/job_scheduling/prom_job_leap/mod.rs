use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {}

use tig_challenges::job_scheduling::{Challenge, Solution};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use rand::SeedableRng;
use rand::Rng;
use rand::rngs::SmallRng;

struct JobState {
    product: usize,
    next_op: usize,
    ready: u32, // time the previous operation of this job completes
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);

    let mut jobs: Vec<JobState> = Vec::with_capacity(challenge.num_jobs);
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(challenge.num_jobs);
    let mut total_ops: usize = 0;
    for (product, &count) in challenge.jobs_per_product.iter().enumerate() {
        let n_ops = challenge.product_processing_times[product].len();
        for _ in 0..count {
            jobs.push(JobState { product, next_op: 0, ready: 0 });
            job_schedule.push(Vec::with_capacity(n_ops));
            total_ops += n_ops;
        }
    }

    let mut machine_avail: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::new();

    let mut scheduled = 0usize;
    while scheduled < total_ops {
        let mut candidates: Vec<(usize, usize, u32, u32)> = Vec::new();
        for (ji, j) in jobs.iter().enumerate() {
            let ops = &challenge.product_processing_times[j.product];
            if j.next_op >= ops.len() {
                continue;
            }
            let op = &ops[j.next_op];
            for (&machine, &dur) in op.iter() {
                let avail = *machine_avail.get(&machine).unwrap_or(&0);
                let start = j.ready.max(avail);
                let finish = start.saturating_add(dur);
                candidates.push((ji, machine, start, finish));
            }
        }

        candidates.sort_unstable_by(|a, b| {
            if a.3 == b.3 {
                a.2.cmp(&b.2)
            } else {
                a.3.cmp(&b.3)
            }
        });

        if let Some((ji, machine, start, dur)) = candidates.first() {
            job_schedule[*ji].push((*machine, *start));
            machine_avail.insert(*machine, start.saturating_add(*dur));
            jobs[*ji].ready = start.saturating_add(*dur);
            jobs[*ji].next_op += 1;
            scheduled += 1;
        } else {
            return Err(anyhow!("no ready operation could be scheduled"));
        }

        if rng.gen_range(0.0f64..1.0f64) < 0.1f64 {
            let ji = rng.gen_range(0..jobs.len());
            if jobs[ji].next_op < challenge.product_processing_times[jobs[ji].product].len() {
                let ops = &challenge.product_processing_times[jobs[ji].product];
                let op = &ops[jobs[ji].next_op];
                let mut machines: Vec<usize> = op.keys().cloned().collect();
                machines.sort_unstable();
                let machine = machines[rng.gen_range(0..machines.len())];
                let dur = op[&machine];
                let avail = *machine_avail.get(&machine).unwrap_or(&0);
                let start = jobs[ji].ready.max(avail);
                job_schedule[ji].push((machine, start));
                machine_avail.insert(machine, start.saturating_add(dur));
                jobs[ji].ready = start.saturating_add(dur);
                jobs[ji].next_op += 1;
                scheduled += 1;
            }
        }
    }

    save_solution(&Solution { job_schedule })?;
    Ok(())
}

pub fn help() {
    println!("Prometheus solver");
}
