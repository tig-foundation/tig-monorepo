mod difficulty_sampler;
pub mod download_wasm;
mod find_benchmark_to_submit;
mod find_proof_to_recompute;
mod find_proof_to_submit;
mod query_data;
pub mod select_job;
mod setup_jobs;
mod submit_benchmark;
mod submit_proof;

#[cfg(not(feature = "cuda"))]
pub mod run_benchmark;
#[cfg(feature = "cuda")]
#[path = "cuda_run_benchmark.rs"]
pub mod run_benchmark;

use crate::utils::{sleep, time, Result};
use difficulty_sampler::DifficultySampler;
use once_cell::sync::OnceCell;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tig_api::Api;
use tig_structs::{
    config::{MinMaxDifficulty, WasmVMConfig},
    core::*,
};
use tokio::{sync::Mutex, task::yield_now};

#[derive(Serialize, Clone, Debug)]
pub struct QueryData {
    pub latest_block: Block,
    pub player_data: Option<PlayerBlockData>,
    pub challenges: Vec<Challenge>,
    pub download_urls: HashMap<String, String>,
    pub algorithms_by_challenge: HashMap<String, Vec<Algorithm>>,
    pub benchmarks: HashMap<String, Benchmark>,
    pub proofs: HashMap<String, Proof>,
    pub frauds: HashMap<String, Fraud>,
}

pub struct NonceIterator {
    start: u64,
    current: u64,
    end: u64,
}

impl NonceIterator {
    pub fn new(start: u64, end: u64) -> Self {
        NonceIterator {
            start,
            current: start,
            end,
        }
    }

    pub fn num_attempts(&self) -> u64 {
        self.current - self.start
    }
}

impl Iterator for NonceIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let result = Some(self.current);
            self.current += 1;
            result
        } else {
            None
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Timestamps {
    pub start: u64,
    pub end: u64,
    pub submit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Job {
    pub download_url: String,
    pub benchmark_id: String,
    pub settings: BenchmarkSettings,
    pub solution_signature_threshold: u32,
    pub sampled_nonces: Option<Vec<u64>>,
    pub wasm_vm_config: WasmVMConfig,
    pub weight: f64,
    pub timestamps: Timestamps,
    #[serde(skip)]
    pub solutions_data: Arc<Mutex<HashMap<u64, SolutionData>>>,
}

impl Job {
    pub fn create_nonce_iterators(&self, num_workers: u32) -> Vec<Arc<Mutex<NonceIterator>>> {
        match &self.sampled_nonces {
            Some(sampled_nonces) => sampled_nonces
                .iter()
                .map(|&n| Arc::new(Mutex::new(NonceIterator::new(n, n + 1))))
                .collect(),
            None => {
                let mut rng = StdRng::seed_from_u64(time());
                let offset = u64::MAX / (num_workers as u64 + 1);
                let random_offset = rng.gen_range(0..offset);
                (0..num_workers)
                    .map(|i| {
                        let start = random_offset + offset * i as u64;
                        let end = start + offset;
                        Arc::new(Mutex::new(NonceIterator::new(start, end)))
                    })
                    .collect()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub query_data: QueryData,
    pub available_jobs: HashMap<String, Job>,
    pub pending_benchmark_jobs: HashMap<String, Job>,
    pub pending_proof_jobs: HashMap<String, Job>,
    pub submitted_proof_ids: HashSet<String>,
    pub difficulty_samplers: HashMap<String, DifficultySampler>,
}

static STATE: OnceCell<Mutex<State>> = OnceCell::new();
static API: OnceCell<Api> = OnceCell::new();
static PLAYER_ID: OnceCell<String> = OnceCell::new();

pub fn api() -> &'static Api {
    API.get().expect("API should be initialised")
}

pub fn player_id() -> &'static String {
    PLAYER_ID.get().expect("PLAYER_ID should be initialised")
}

pub fn state() -> &'static Mutex<State> {
    STATE.get().expect("STATE should be initialised")
}

pub async fn proof_submitter() {
    async fn do_work() -> Result<()> {
        println!("[proof_submitter]: checking for any proofs to submit");
        if let Some(mut job) = find_proof_to_submit::execute().await? {
            println!(
                "[proof_submitter]: submitting proof for benchmark {}",
                job.benchmark_id
            );
            {
                let mut state = (*state()).lock().await;
                state.submitted_proof_ids.insert(job.benchmark_id.clone());
            }
            {
                let solutions_data = job.solutions_data.lock().await;
                if !job
                    .sampled_nonces
                    .as_ref()
                    .unwrap()
                    .iter()
                    .all(|n| solutions_data.contains_key(n))
                {
                    return Err(format!("failed to find solutions for every sampled nonces"));
                }
            }
            match submit_proof::execute(&job).await {
                Ok(_) => {
                    println!(
                        "[proof_submitter]: successfully submitted proof for benchmark {}",
                        job.benchmark_id
                    );
                }
                Err(e) => {
                    println!("[proof_submitter]: failed to submit proof: {:?}", e);
                    // FIXME hacky way to check for 4xx status
                    if !e.contains("status: 4") {
                        println!("[proof_submitter]: re-queueing proof for another submit attempt 10s later");
                        job.timestamps.submit = time() + 10000;
                        let mut state = (*state()).lock().await;
                        state.submitted_proof_ids.remove(&job.benchmark_id);
                        state
                            .pending_proof_jobs
                            .insert(job.benchmark_id.clone(), job);
                    }
                }
            }
        } else {
            println!("[proof_submitter]: no proofs to submit");
        }
        Ok(())
    }

    loop {
        if let Err(e) = do_work().await {
            println!("[proof_submitter]: error: {:?}", e);
        }
        println!("[proof_submitter]: sleeping 5s");
        sleep(5000).await;
    }
}

pub async fn benchmark_submitter() {
    async fn do_work() -> Result<()> {
        println!("[benchmark_submitter]: checking for any benchmarks to submit");
        if let Some(mut job) = find_benchmark_to_submit::execute().await? {
            let num_solutions = {
                let solutions_data = job.solutions_data.lock().await;
                solutions_data.len()
            };
            println!(
                "[benchmark_submitter]: submitting benchmark {:?} with {} solutions",
                job.settings, num_solutions
            );
            match submit_benchmark::execute(&job).await {
                Ok(benchmark_id) => {
                    job.benchmark_id = benchmark_id.clone();
                    job.timestamps.submit = time();
                    println!(
                        "[benchmark_submitter]: successfully submitted benchmark {}",
                        benchmark_id
                    );
                    {
                        let mut state = (*state()).lock().await;
                        state.pending_proof_jobs.insert(benchmark_id, job);
                    }
                }
                Err(e) => {
                    println!("[benchmark_submitter]: failed to submit benchmark: {:?}", e);
                    // FIXME hacky way to check for 4xx status
                    if !e.contains("status: 4") {
                        println!("[benchmark_submitter]: re-queueing benchmark for another submit attempt 10s later");
                        job.timestamps.submit = time() + 10000;
                        let mut state = (*state()).lock().await;
                        state
                            .pending_benchmark_jobs
                            .insert(job.benchmark_id.clone(), job);
                    }
                }
            }
        } else {
            println!("[benchmark_submitter]: no benchmarks to submit");
        }
        Ok(())
    }

    loop {
        if let Err(e) = do_work().await {
            println!("[benchmark_submitter]: error: {:?}", e);
        }
        println!("[benchmark_submitter]: sleeping 5s");
        sleep(5000).await;
    }
}

pub async fn data_fetcher() {
    async fn do_work() -> Result<()> {
        println!("[data_fetcher]: fetching latest data");
        let new_query_data = query_data::execute().await?;
        if {
            let state = (*state()).lock().await;
            state.query_data.latest_block.id != new_query_data.latest_block.id
        } {
            println!(
                "[data_fetcher]: got new block {} @ {}",
                new_query_data.latest_block.id, new_query_data.latest_block.details.height
            );
            {
                let mut state = (*state()).lock().await;
                (*state).query_data = new_query_data;
            }

            println!("[data_fetcher]: updating difficulty samplers");
            {
                let mut state = state().lock().await;
                let State {
                    query_data,
                    difficulty_samplers,
                    ..
                } = &mut (*state);
                for challenge in query_data.challenges.iter() {
                    let difficulty_sampler = difficulty_samplers
                        .entry(challenge.id.clone())
                        .or_insert_with(|| DifficultySampler::new());
                    let min_difficulty = query_data.latest_block.config().difficulty.parameters
                        [&challenge.id]
                        .min_difficulty();
                    difficulty_sampler
                        .update_with_block_data(&min_difficulty, challenge.block_data());
                }
            }
        } else {
            println!("[data_fetcher]: no new data");
        }
        Ok(())
    }

    loop {
        if let Err(e) = do_work().await {
            println!("[data_fetcher]: error: {:?}", e);
        }
        println!("[data_fetcher]: sleeping 10s");
        sleep(10000).await;
    }
}

pub async fn benchmarker(
    selected_algorithms: HashMap<String, String>,
    num_workers: u32,
    benchmark_duration_ms: u64,
    submit_delay_ms: u64,
) {
    async fn do_work(
        selected_algorithms: &HashMap<String, String>,
        num_workers: u32,
        benchmark_duration_ms: u64,
        submit_delay_ms: u64,
    ) -> Result<()> {
        println!("[benchmarker]: setting up jobs");
        let jobs = setup_jobs::execute(selected_algorithms, benchmark_duration_ms, submit_delay_ms)
            .await?;
        for (i, job) in jobs.values().enumerate() {
            println!(
                "[benchmarker]: job {}: {:?}, weight: {}",
                i, job.settings, job.weight
            );
        }
        {
            let mut state = state().lock().await;
            state.available_jobs = jobs.clone();
        }

        println!("[benchmarker]: finding proofs to re-compute");
        let job = {
            if let Some(job) =
                find_proof_to_recompute::execute(benchmark_duration_ms + 5000).await?
            {
                println!(
                    "[benchmarker]: found proof to recompute: {}",
                    job.benchmark_id
                );
                let mut state = state().lock().await;
                state.query_data.proofs.insert(
                    job.benchmark_id.clone(),
                    Proof {
                        benchmark_id: job.benchmark_id.clone(),
                        state: None,
                        solutions_data: Some(Vec::new()),
                    },
                );
                job
            } else {
                println!("[benchmarker]: no proofs to recompute");
                println!("[benchmarker]: weighted sampling one of the available jobs");
                select_job::execute(&jobs).await?
            }
        };

        println!(
            "[benchmarker]: downloading algorithm {}",
            job.download_url.split("/").last().unwrap()
        );
        let wasm = download_wasm::execute(&job).await?;

        println!("[benchmarker]: starting benchmark {:?}", job.settings);
        let nonce_iterators = job.create_nonce_iterators(num_workers);
        run_benchmark::execute(nonce_iterators.clone(), &job, &wasm).await;
        loop {
            {
                let mut num_attempts = 0;
                for nonce_iterator in &nonce_iterators {
                    num_attempts += nonce_iterator.lock().await.num_attempts();
                }
                let num_solutions = job.solutions_data.lock().await.len();
                let elapsed = time() - job.timestamps.start;
                if num_workers > 0 || job.sampled_nonces.is_some() {
                    println!(
                        "[benchmarker]: #solutions: {}, #instances: {}, elapsed: {}ms",
                        num_solutions, num_attempts, elapsed
                    );
                }
                if time() >= job.timestamps.end
                    || job
                        .sampled_nonces
                        .as_ref()
                        .is_some_and(|sampled_nonces| num_solutions == sampled_nonces.len())
                {
                    break;
                }
            }
            sleep(500).await;
        }
        {
            let mut num_attempts = 0;
            for nonce_iterator in &nonce_iterators {
                num_attempts += nonce_iterator.lock().await.num_attempts();
            }
            let mut state = state().lock().await;
            let num_solutions = job.solutions_data.lock().await.len() as u32;
            if job.sampled_nonces.is_some() {
                state
                    .pending_proof_jobs
                    .insert(job.benchmark_id.clone(), job);
            } else if num_attempts > 0 {
                state
                    .difficulty_samplers
                    .get_mut(&job.settings.challenge_id)
                    .unwrap()
                    .update_with_solutions(&job.settings.difficulty, num_solutions);
            }

            let jobs = state.available_jobs.drain().collect::<HashMap<_, _>>();
            state.pending_benchmark_jobs.extend(jobs);
        }
        Ok(())
    }

    loop {
        if let Err(e) = do_work(
            &selected_algorithms,
            num_workers,
            benchmark_duration_ms,
            submit_delay_ms,
        )
        .await
        {
            println!("[benchmarker]: error: {:?}", e);
            println!("[benchmarker]: sleeping 5s");
            sleep(5000).await;
        } else {
            yield_now().await;
        }
    }
}

pub async fn setup(api_url: String, api_key: String, player_id: String) {
    API.get_or_init(|| Api::new(api_url, api_key));
    PLAYER_ID.get_or_init(|| player_id);
    let query_data = query_data::execute().await.expect("Failed to query data");
    let mut difficulty_samplers = HashMap::new();
    for challenge in query_data.challenges.iter() {
        let difficulty_sampler = difficulty_samplers
            .entry(challenge.id.clone())
            .or_insert_with(|| DifficultySampler::new());
        let min_difficulty =
            query_data.latest_block.config().difficulty.parameters[&challenge.id].min_difficulty();
        difficulty_sampler.update_with_block_data(&min_difficulty, challenge.block_data());
    }
    STATE.get_or_init(|| {
        Mutex::new(State {
            query_data,
            difficulty_samplers,
            available_jobs: HashMap::new(),
            pending_benchmark_jobs: HashMap::new(),
            pending_proof_jobs: HashMap::new(),
            submitted_proof_ids: HashSet::new(),
        })
    });
}
