use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng, seq::SliceRandom};
use tig_challenges::job_scheduling::*;
use super::types::GreedyRule;

pub fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt {
            job_products.push(p);
        }
    }

    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len())
        .collect();

    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| {
                let avg: f64 = op.values().sum::<u32>() as f64 / op.len().max(1) as f64;
                avg
            })
            .sum()
    }).collect();

    let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::LeastFlex, GreedyRule::ShortestProc, GreedyRule::LongestProc];
    let mut best_mk = u32::MAX;
    let mut best_sol: Option<Solution> = None;

    for rule in rules {
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, None)?;
        if mk < best_mk {
            best_mk = mk;
            best_sol = Some(sol);
        }
    }

    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>();
        let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5);
        let mut local_rng = SmallRng::seed_from_u64(seed);

        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk {
            best_mk = mk;
            best_sol = Some(sol);
        }
    }

    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_greedy_rule(
    challenge: &Challenge,
    job_products: &[usize],
    job_ops_len: &[usize],
    job_total_work: &[f64],
    rule: GreedyRule,
    mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_next_op = vec![0usize; num_jobs];
    let mut job_ready = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter()
        .map(|&len| Vec::with_capacity(len))
        .collect();
    let mut job_work_left = job_total_work.to_vec();

    let mut remaining = job_ops_len.iter().sum::<usize>();
    let mut time = 0u32;
    let eps = 1e-9;

    while remaining > 0 {
        let mut available_machines: Vec<usize> = (0..num_machines)
            .filter(|&m| machine_avail[m] <= time)
            .collect();
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k {
            available_machines.shuffle(*rng);
        }

        for &m in &available_machines {
            #[derive(Clone)]
            struct Candidate {
                job: usize,
                priority: f64,
                end: u32,
                pt: u32,
                flex: usize,
            }

            let mut candidates: Vec<Candidate> = Vec::new();

            for j in 0..num_jobs {
                if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time {
                    continue;
                }

                let product = job_products[j];
                let op_idx = job_next_op[j];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let pt = match op_times.get(&m) {
                    Some(&v) => v,
                    None => continue,
                };

                let earliest = op_times.iter()
                    .map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt)
                    .min().unwrap_or(u32::MAX);
                let this_end = time.max(machine_avail[m]) + pt;
                if this_end != earliest {
                    continue;
                }

                let flex = op_times.len();
                let ops_left = job_ops_len[j] - job_next_op[j];
                let priority = match rule {
                    GreedyRule::MostWork => job_work_left[j],
                    GreedyRule::MostOps => ops_left as f64,
                    GreedyRule::LeastFlex => -(flex as f64),
                    GreedyRule::ShortestProc => -(pt as f64),
                    GreedyRule::LongestProc => pt as f64,
                };

                candidates.push(Candidate { job: j, priority, end: this_end, pt, flex });
            }

            if candidates.is_empty() {
                continue;
            }

            let best_job = if let Some((top_k, ref mut rng)) = random_top_k {
                candidates.sort_by(|a, b| {
                    if (b.priority - a.priority).abs() > eps {
                        b.priority.partial_cmp(&a.priority).unwrap()
                    } else if a.end != b.end {
                        a.end.cmp(&b.end)
                    } else if a.pt != b.pt {
                        a.pt.cmp(&b.pt)
                    } else if a.flex != b.flex {
                        a.flex.cmp(&b.flex)
                    } else {
                        a.job.cmp(&b.job)
                    }
                });
                let top = candidates.len().min(top_k);
                candidates[rng.gen_range(0..top)].job
            } else {
                let mut best: Option<Candidate> = None;
                for cand in candidates {
                    let better = if let Some(ref b) = best {
                        if (cand.priority - b.priority).abs() > eps {
                            cand.priority > b.priority
                        } else if cand.end != b.end {
                            cand.end < b.end
                        } else if cand.pt != b.pt {
                            cand.pt < b.pt
                        } else if cand.flex != b.flex {
                            cand.flex < b.flex
                        } else {
                            cand.job < b.job
                        }
                    } else {
                        true
                    };
                    if better {
                        best = Some(cand);
                    }
                }
                best.ok_or_else(|| anyhow!("No candidate"))?.job
            };

            let product = job_products[best_job];
            let op_idx = job_next_op[best_job];
            let op_times = &challenge.product_processing_times[product][op_idx];
            let pt = op_times[&m];
            let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;

            let st = time.max(machine_avail[m]);
            let end = st + pt;

            job_schedule[best_job].push((m, st));
            job_next_op[best_job] += 1;
            job_ready[best_job] = end;
            machine_avail[m] = end;
            job_work_left[best_job] -= avg_pt;
            if job_work_left[best_job] < 0.0 {
                job_work_left[best_job] = 0.0;
            }
            remaining -= 1;
        }

        if remaining == 0 {
            break;
        }

        let mut next = u32::MAX;
        for &t in &machine_avail {
            if t > time && t < next {
                next = t;
            }
        }
        for j in 0..num_jobs {
            if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next {
                next = job_ready[j];
            }
        }

        if next == u32::MAX {
            return Err(anyhow!("Greedy baseline stuck"));
        }
        time = next;
    }

    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}
