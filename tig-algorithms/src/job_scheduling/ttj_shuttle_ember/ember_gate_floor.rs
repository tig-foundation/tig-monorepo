// Validity floor: guarantees a savable solution before any search.
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::Pre;

fn gate_dispatch(
    pre: &Pre,
    rule: u8,
    random_top_k: usize,
    rng: Option<&mut SmallRng>,
) -> Option<Solution> {
    let nj = pre.num_jobs;
    let nm = pre.num_machines;
    let mut job_next = vec![0usize; nj];
    let mut job_ready = vec![0u32; nj];
    let mut mach_avail = vec![0u32; nm];
    let mut job_schedule: Vec<Vec<(usize, u32)>> =
        pre.job_nops.iter().map(|&k| Vec::with_capacity(k)).collect();

    let op_w = |j: usize, oi: usize| -> f64 { pre.op_w[pre.job_off[j] + oi] };
    let mut rem_work: Vec<f64> = (0..nj)
        .map(|j| (0..pre.job_nops[j]).map(|oi| op_w(j, oi)).sum())
        .collect();

    let mut remaining: usize = pre.job_nops.iter().sum();
    let mut time = 0u32;
    let eps = 1e-9f64;
    let use_random = random_top_k > 1 && rng.is_some();
    let mut rng = rng;
    let mut cands: Vec<(f64, u32, u32, usize, usize)> = Vec::new();

    while remaining > 0 {
        let mut machines: Vec<usize> = (0..nm).filter(|&m| mach_avail[m] <= time).collect();
        if use_random {
            machines.shuffle(rng.as_mut().unwrap());
        }
        for &machine in machines.iter() {
            cands.clear();
            for job in 0..nj {
                let oi = job_next[job];
                if oi >= pre.job_nops[job] || job_ready[job] > time {
                    continue;
                }
                let op = &pre.job_ops[job][oi];
                let Some(&(_, pt)) = op.iter().find(|&&(m, _)| m == machine) else {
                    continue;
                };
                let mut earliest_end = u32::MAX;
                for &(m2, p2) in op.iter() {
                    let e = time.max(mach_avail[m2]) + p2;
                    if e < earliest_end {
                        earliest_end = e;
                    }
                }
                let machine_end = time.max(mach_avail[machine]) + pt;
                if machine_end != earliest_end {
                    continue;
                }
                let flex = op.len();
                let pr = match rule {
                    0 => rem_work[job],
                    1 => (pre.job_nops[job] - oi) as f64,
                    2 => -(flex as f64),
                    3 => -(pt as f64),
                    _ => pt as f64,
                };
                cands.push((pr, machine_end, pt, flex, job));
            }
            if cands.is_empty() {
                continue;
            }
            let chosen = if use_random {
                cands.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap()
                        .then(a.1.cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                        .then(a.3.cmp(&b.3))
                        .then(a.4.cmp(&b.4))
                });
                let k = random_top_k.min(cands.len());
                cands[rng.as_mut().unwrap().gen_range(0..k)]
            } else {
                let mut best = cands[0];
                for &c in cands.iter().skip(1) {
                    let better = if c.0 > best.0 + eps {
                        true
                    } else if (c.0 - best.0).abs() <= eps {
                        (c.1, c.2, c.3, c.4) < (best.1, best.2, best.3, best.4)
                    } else {
                        false
                    };
                    if better {
                        best = c;
                    }
                }
                best
            };
            let (_, _, pt, _, job) = chosen;
            let oi = job_next[job];
            let start = time.max(mach_avail[machine]);
            let end = start + pt;
            job_schedule[job].push((machine, start));
            job_next[job] += 1;
            job_ready[job] = end;
            mach_avail[machine] = end;
            rem_work[job] -= op_w(job, oi);
            if rem_work[job] < 0.0 {
                rem_work[job] = 0.0;
            }
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        let mut next: Option<u32> = None;
        for &t in mach_avail.iter() {
            if t > time {
                next = Some(next.map_or(t, |b: u32| b.min(t)));
            }
        }
        for job in 0..nj {
            if job_next[job] < pre.job_nops[job] && job_ready[job] > time {
                let t = job_ready[job];
                next = Some(next.map_or(t, |b: u32| b.min(t)));
            }
        }
        time = next?;
    }
    Some(Solution { job_schedule })
}

pub fn gate_floor(
    pre: &Pre,
    seed: &[u8; 32],
    extra_restarts: usize,
    guarded: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    for rule in 0..5u8 {
        if let Some(s) = gate_dispatch(pre, rule, 0, None) {
            guarded(&s)?;
        }
    }
    let mut rng = SmallRng::from_seed(*seed);
    for _ in 1..=(10 + extra_restarts) {
        let local_seed = rng.r#gen::<u64>();
        let rule = rng.gen_range(0..5usize) as u8;
        let top_k = rng.gen_range(2..=5usize);
        let mut local = SmallRng::seed_from_u64(local_seed);
        if let Some(s) = gate_dispatch(pre, rule, top_k, Some(&mut local)) {
            guarded(&s)?;
        }
    }
    Ok(())
}
