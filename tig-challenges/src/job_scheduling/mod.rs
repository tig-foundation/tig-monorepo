use crate::QUALITY_PRECISION;
mod baselines;
use anyhow::{anyhow, Result};
use rand::{
    distributions::Distribution,
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

pub struct FlowConfig {
    pub avg_op_flexibility: f32,
    pub reentrance_level: f32,
    pub flow_structure: f32,
    pub product_mix_ratio: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flow {
    STRICT,
    PARALLEL,
    RANDOM,
    COMPLEX,
    CHAOTIC,
}

impl From<Flow> for FlowConfig {
    fn from(flow: Flow) -> Self {
        match flow {
            Flow::STRICT => FlowConfig {
                avg_op_flexibility: 1.0,
                reentrance_level: 0.2,
                flow_structure: 0.0,
                product_mix_ratio: 0.5,
            },
            Flow::PARALLEL => FlowConfig {
                avg_op_flexibility: 3.0,
                reentrance_level: 0.2,
                flow_structure: 0.0,
                product_mix_ratio: 0.5,
            },
            Flow::RANDOM => FlowConfig {
                avg_op_flexibility: 1.0,
                reentrance_level: 0.0,
                flow_structure: 0.4,
                product_mix_ratio: 1.0,
            },
            Flow::COMPLEX => FlowConfig {
                avg_op_flexibility: 3.0,
                reentrance_level: 0.2,
                flow_structure: 0.4,
                product_mix_ratio: 1.0,
            },
            Flow::CHAOTIC => FlowConfig {
                avg_op_flexibility: 10.0,
                reentrance_level: 0.0,
                flow_structure: 1.0,
                product_mix_ratio: 1.0,
            },
        }
    }
}

impl std::fmt::Display for Flow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Flow::STRICT => write!(f, "strict"),
            Flow::PARALLEL => write!(f, "parallel"),
            Flow::RANDOM => write!(f, "random"),
            Flow::COMPLEX => write!(f, "complex"),
            Flow::CHAOTIC => write!(f, "chaotic"),
        }
    }
}

impl std::str::FromStr for Flow {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "strict" => Ok(Flow::STRICT),
            "parallel" => Ok(Flow::PARALLEL),
            "random" => Ok(Flow::RANDOM),
            "complex" => Ok(Flow::COMPLEX),
            "chaotic" => Ok(Flow::CHAOTIC),
            _ => Err(anyhow::anyhow!("Invalid flow type: {}", s)),
        }
    }
}

impl_kv_string_serde! {
    Track {
        n: usize,
        m: usize,
        o: usize,
        flow: Flow
    }
}

impl_base64_serde! {
    Solution {
        job_schedule: Vec<Vec<(usize, u32)>>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            job_schedule: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub num_jobs: usize,
    pub num_machines: usize,
    pub num_operations: usize,
    pub jobs_per_product: Vec<usize>,
    // each product has a sequence of operations, and each operation has a map of eligible machines to processing times
    pub product_processing_times: Vec<Vec<HashMap<usize, u32>>>,
}

impl Challenge {
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed.clone()).r#gen());
        let FlowConfig {
            avg_op_flexibility,
            reentrance_level,
            flow_structure,
            product_mix_ratio,
        } = track.flow.clone().into();
        let n_jobs = track.n;
        let n_machines = track.m;
        let n_op_types = track.o;
        let n_products = 1.max((product_mix_ratio * n_jobs as f32) as usize);
        let n_routes = 1.max((flow_structure * n_jobs as f32) as usize);
        let min_eligible_machines = 1;
        let flexibility_std_dev = 0.5;
        let base_proc_time_min = 1;
        let base_proc_time_max = 200;
        let min_speed_factor = 0.8;
        let max_speed_factor = 1.2;

        // random product for each job, only keep products that have at least one job
        let mut map = HashMap::new();
        let jobs_per_product = (0..n_jobs).fold(Vec::new(), |mut acc, _| {
            let map_len = map.len();
            let product = *map
                .entry(rng.gen_range(0..n_products))
                .or_insert_with(|| map_len);
            if product >= acc.len() {
                acc.push(0);
            }
            acc[product] += 1;
            acc
        });
        // actual number of products (some products may have zero jobs)
        let n_products = jobs_per_product.len();

        // random route for each product, only keep routes that are used
        let mut map = HashMap::new();
        let product_route = (0..n_products)
            .map(|_| {
                let map_len = map.len();
                *map.entry(rng.gen_range(0..n_routes))
                    .or_insert_with(|| map_len)
            })
            .collect::<Vec<usize>>();
        // actual number of routes
        let n_routes = map.len();

        // generate operation sequence for each route
        let routes = (0..n_routes)
            .map(|_| {
                let seq_len = n_op_types;
                let mut base_sequence: Vec<usize> = (0..n_op_types).collect();
                let mut steps = Vec::new();

                // randomly build op sequence
                for _ in 0..seq_len {
                    let next_op_idx = if rng.r#gen::<f32>() < flow_structure {
                        // Job Shop Logic: Random permutation
                        rng.gen_range(0..base_sequence.len())
                    } else {
                        // Flow Shop Logic: Pick next sequential op
                        0
                    };

                    let op_id = base_sequence.remove(next_op_idx);
                    steps.push(op_id);
                }

                for step_idx in (2..steps.len()).rev() {
                    // Reentrance Logic
                    if rng.r#gen::<f32>() < reentrance_level {
                        // assuming reentrance_level of 0.1
                        let op_id = steps[rng.gen_range(0..step_idx - 1)];
                        steps.insert(step_idx, op_id);
                    }
                }

                steps
            })
            .collect::<Vec<Vec<usize>>>();

        // generate machine eligibility and base processing time for each operation
        let normal = Normal::new(avg_op_flexibility, flexibility_std_dev).unwrap();
        let all_machines = (0..n_machines).collect::<HashSet<usize>>();
        let op_eligible_machines = (0..n_op_types)
            .map(|i| {
                if avg_op_flexibility as usize >= n_machines {
                    (0..n_machines).collect::<HashSet<usize>>()
                } else {
                    let mut eligible = HashSet::<usize>::from([if i < n_machines {
                        i
                    } else {
                        rng.gen_range(0..n_machines)
                    }]);
                    if avg_op_flexibility > 1.0 {
                        let target_flex = min_eligible_machines
                            .max(normal.sample(&mut rng) as usize)
                            .min(n_machines);
                        let mut remaining = all_machines
                            .difference(&eligible)
                            .cloned()
                            .collect::<Vec<usize>>();
                        let num_to_add = (target_flex - 1).min(remaining.len());
                        for j in 0..num_to_add {
                            let idx = rng.gen_range(j..remaining.len());
                            remaining.swap(j, idx);
                        }
                        eligible.extend(remaining[..num_to_add].iter().cloned());
                    }
                    eligible
                }
            })
            .collect::<Vec<_>>();
        let base_proc_times = (0..n_op_types)
            .map(|_| rng.gen_range(base_proc_time_min..=base_proc_time_max))
            .collect::<Vec<u32>>();

        // generate processing times for each product according to its route
        let product_processing_times = product_route
            .iter()
            .map(|&r_idx| {
                let route = &routes[r_idx];
                route
                    .iter()
                    .map(|&op_id| {
                        let machines = &op_eligible_machines[op_id];
                        let base_time = base_proc_times[op_id];
                        machines
                            .iter()
                            .map(|&m_id| {
                                (
                                    m_id,
                                    1.max(
                                        (base_time as f32
                                            * (min_speed_factor
                                                + (max_speed_factor - min_speed_factor)
                                                    * rng.r#gen::<f32>()))
                                            as u32,
                                    ),
                                )
                            })
                            .collect::<HashMap<usize, u32>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Ok(Challenge {
            seed: seed.clone(),
            num_jobs: n_jobs,
            num_machines: n_machines,
            num_operations: n_op_types,
            jobs_per_product,
            product_processing_times,
        })
    }

    pub fn evaluate_makespan(&self, solution: &Solution) -> Result<u32> {
        if solution.job_schedule.len() != self.num_jobs {
            return Err(anyhow!(
                "Expecting solution to have {} jobs. Got {}",
                self.num_jobs,
                solution.job_schedule.len(),
            ));
        }
        let mut job = 0;
        let mut machine_usage = HashMap::<usize, Vec<(u32, u32)>>::new();
        let mut makespan = 0u32;
        for (product, num_jobs) in self.jobs_per_product.iter().enumerate() {
            for _ in 0..*num_jobs {
                let schedule = &solution.job_schedule[job];
                let processing_times = &self.product_processing_times[product];
                if schedule.len() != processing_times.len() {
                    return Err(anyhow!(
                        "Job {} of product {} expecting {} operations. Got {}",
                        job,
                        product,
                        processing_times.len(),
                        schedule.len(),
                    ));
                }
                let mut min_start_time = 0;
                for (op_idx, &(machine, start_time)) in schedule.iter().enumerate() {
                    let eligible_machines = &processing_times[op_idx];
                    if !eligible_machines.contains_key(&machine) {
                        return Err(anyhow!("Job {} schedule contains ineligible machine", job,));
                    }
                    if start_time < min_start_time {
                        return Err(anyhow!(
                            "Job {} schedule contains operation starting before previous is complete",
                            job,
                        ));
                    }
                    let finish_time = start_time + eligible_machines[&machine];
                    machine_usage
                        .entry(machine)
                        .or_default()
                        .push((start_time, finish_time));
                    min_start_time = finish_time;
                }
                // min_start_time is the finish time of the job
                if min_start_time > makespan {
                    makespan = min_start_time;
                }
                job += 1;
            }
        }

        for (machine, usage) in machine_usage.iter_mut() {
            usage.sort_by_key(|&(start, _)| start);
            for i in 1..usage.len() {
                if usage[i].0 < usage[i - 1].1 {
                    return Err(anyhow!(
                        "Machine {} is scheduled with overlapping jobs",
                        machine,
                    ));
                }
            }
        }

        Ok(makespan)
    }

    conditional_pub!(
        fn compute_greedy_baseline(&self) -> Result<Solution> {
            let solution = RefCell::new(Solution::new());
            let save_solution_fn = |s: &Solution| -> Result<()> {
                *solution.borrow_mut() = s.clone();
                Ok(())
            };
            baselines::dispatching_rules::solve_challenge(self, &save_solution_fn, &None)?;
            Ok(solution.into_inner())
        }
    );

    conditional_pub!(
        fn compute_sota_baseline(&self) -> Result<Solution> {
            Err(anyhow!("Not implemented yet"))
        }
    );

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let makespan = self.evaluate_makespan(solution)?;
            let greedy_solution = self.compute_greedy_baseline()?;
            let greedy_makespan = self.evaluate_makespan(&greedy_solution)?;
            // TODO: implement SOTA baseline
            let quality = (greedy_makespan as f64 - makespan as f64) / greedy_makespan as f64;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}
