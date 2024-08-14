mod difficulty_sampler;
pub mod download_wasm;
mod find_proof_to_submit;
mod query_data;
mod setup_job;
mod submit_benchmark;
mod submit_proof;

#[cfg(not(feature = "cuda"))]
pub mod run_benchmark;
#[cfg(feature = "cuda")]
#[path = "cuda_run_benchmark.rs"]
pub mod run_benchmark;

use crate::future_utils::{sleep, spawn, time, Mutex};
use difficulty_sampler::DifficultySampler;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tig_api::Api;
use tig_structs::{
    config::{MinMaxDifficulty, WasmVMConfig},
    core::*,
};

pub type Result<T> = std::result::Result<T, String>;

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

#[derive(Serialize, Clone, Debug)]
pub struct Timer {
    pub start: u64,
    pub end: u64,
    pub now: u64,
}
impl Timer {
    fn new(ms: u64) -> Self {
        let now = time();
        Timer {
            start: now,
            end: now + ms,
            now,
        }
    }
    fn update(&mut self) -> &Self {
        self.now = time();
        self
    }
    fn finished(&self) -> bool {
        self.now >= self.end
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Job {
    pub download_url: String,
    pub benchmark_id: String,
    pub settings: BenchmarkSettings,
    pub solution_signature_threshold: u32,
    pub sampled_nonces: Option<Vec<u32>>,
    pub wasm_vm_config: WasmVMConfig,
}

#[derive(Serialize, Debug, Clone)]
pub struct NonceIterator {
    nonces: Option<Vec<u32>>,
    current: u32,
    attempts: u32,
}

impl NonceIterator {
    pub fn from_vec(nonces: Vec<u32>) -> Self {
        Self {
            nonces: Some(nonces),
            current: 0,
            attempts: 0,
        }
    }
    pub fn from_u32(start: u32) -> Self {
        Self {
            nonces: None,
            current: start,
            attempts: 0,
        }
    }
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
    pub fn is_empty(&self) -> bool {
        self.nonces.as_ref().is_some_and(|x| x.is_empty()) || self.current == u32::MAX
    }
    pub fn empty(&mut self) {
        if let Some(nonces) = self.nonces.as_mut() {
            nonces.clear();
        }
        self.current = u32::MAX;
    }
}
impl Iterator for NonceIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(nonces) = self.nonces.as_mut() {
            let value = nonces.pop();
            self.attempts += value.is_some() as u32;
            value
        } else if self.current < u32::MAX {
            let value = Some(self.current);
            self.attempts += 1;
            self.current += 1;
            value
        } else {
            None
        }
    }
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub enum Status {
    Starting,
    Running(String),
    Stopping,
    Stopped,
}
#[derive(Serialize, Debug, Clone)]
pub struct State {
    pub status: Status,
    pub timer: Option<Timer>,
    pub query_data: QueryData,
    pub selected_algorithms: HashMap<String, String>,
    pub job: Option<Job>,
    pub submission_errors: HashMap<String, String>,
    #[serde(skip_serializing)]
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

async fn update_status(status: &str) {
    let mut state = state().lock().await;
    if let Status::Running(_) = state.status {
        state.status = Status::Running(status.to_string());
        println!("{}", status);
        #[cfg(feature = "browser")]
        web_sys::console::log_1(&status.to_string().into());
    }
}

async fn run_once(num_workers: u32, ms_per_benchmark: u32) -> Result<()> {
    {
        let mut state = (*state()).lock().await;
        state.job = None;
        state.timer = None;
    }
    update_status("Querying latest data").await;
    // retain only benchmarks that are within the lifespan period
    // preserves solution_meta_data and solution_data
    let mut new_query_data = query_data::execute().await?;
    if {
        let state = (*state()).lock().await;
        state.query_data.latest_block.id != new_query_data.latest_block.id
    } {
        {
            let mut state = (*state()).lock().await;
            let block_started_cutoff = new_query_data.latest_block.details.height.saturating_sub(
                new_query_data
                    .latest_block
                    .config()
                    .benchmark_submissions
                    .lifespan_period,
            );
            let mut latest_benchmarks = state.query_data.benchmarks.clone();
            latest_benchmarks.retain(|_, x| x.details.block_started >= block_started_cutoff);
            latest_benchmarks.extend(new_query_data.benchmarks.drain());

            let mut latest_proofs = state.query_data.proofs.clone();
            latest_proofs.retain(|id, _| latest_benchmarks.contains_key(id));
            latest_proofs.extend(new_query_data.proofs.drain());

            let mut latest_frauds = state.query_data.frauds.clone();
            latest_frauds.retain(|id, _| latest_benchmarks.contains_key(id));
            latest_frauds.extend(new_query_data.frauds.drain());

            (*state)
                .submission_errors
                .retain(|id, _| latest_benchmarks.contains_key(id));
            new_query_data.benchmarks = latest_benchmarks;
            new_query_data.proofs = latest_proofs;
            new_query_data.frauds = latest_frauds;
            (*state).query_data = new_query_data;
        }

        update_status("Updating difficulty sampler with query data").await;
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
                difficulty_sampler.update_with_block_data(&min_difficulty, challenge.block_data());
            }
        }
    }

    update_status("Finding proof to submit").await;
    match find_proof_to_submit::execute().await? {
        Some((benchmark_id, solutions_data)) => {
            update_status(&format!("Submitting proof for {}", benchmark_id)).await;
            if let Err(e) = submit_proof::execute(benchmark_id.clone(), solutions_data).await {
                let mut state = state().lock().await;
                state.submission_errors.insert(benchmark_id, e.clone());
                return Err(e);
            }
            update_status(&format!("Success. Proof {} submitted", benchmark_id)).await;
        }
        None => {
            update_status("No proof to submit").await;
        }
    }
    // creates a benchmark & proof with job.benchmark_id
    update_status("Selecting settings to benchmark").await;
    setup_job::execute().await?;
    let job = {
        let state = state().lock().await;
        state.job.clone().unwrap()
    };
    update_status(&format!("{:?}", job.settings)).await;

    update_status(&format!(
        "Downloading algorithm {}",
        job.download_url.split("/").last().unwrap()
    ))
    .await;
    let wasm = download_wasm::execute(&job).await?;

    // variables that are shared by workers
    let nonce_iters = match &job.sampled_nonces {
        Some(nonces) => vec![Arc::new(Mutex::new(NonceIterator::from_vec(
            nonces.clone(),
        )))],
        None => (0..num_workers)
            .into_iter()
            .map(|x| {
                Arc::new(Mutex::new(NonceIterator::from_u32(
                    u32::MAX / num_workers * x,
                )))
            })
            .collect(),
    };
    let solutions_data = Arc::new(Mutex::new(Vec::<SolutionData>::new()));
    let solutions_count = Arc::new(Mutex::new(0u32));
    update_status("Starting benchmark").await;
    run_benchmark::execute(
        nonce_iters.iter().cloned().collect(),
        &job,
        &wasm,
        solutions_data.clone(),
        solutions_count.clone(),
    )
    .await;
    {
        let mut state = state().lock().await;
        (*state).timer = Some(Timer::new(ms_per_benchmark as u64));
    }
    loop {
        {
            // transfers solutions computed by workers to benchmark state
            let num_solutions =
                drain_solutions(&job.benchmark_id, &mut *(*solutions_data).lock().await).await;
            let mut finished = true;
            let mut num_attempts = 0;
            for nonce_iter in nonce_iters.iter().cloned() {
                let nonce_iter = (*nonce_iter).lock().await;
                num_attempts += nonce_iter.attempts();
                finished &= nonce_iter.is_empty();
            }
            update_status(&format!(
                "Computed {} solutions out of {} instances",
                num_solutions, num_attempts
            ))
            .await;
            let State {
                status,
                timer: time_left,
                ..
            } = &mut (*state().lock().await);
            if time_left.as_mut().unwrap().update().finished()
                || (finished && num_solutions == num_attempts) // nonce_iter is only empty if recomputing
                || *status == Status::Stopping
            {
                break;
            }
        }
        sleep(200).await;
    }
    for nonce_iter in nonce_iters {
        (*(*nonce_iter).lock().await).empty();
    }

    // transfers solutions computed by workers to benchmark state
    let num_solutions =
        drain_solutions(&job.benchmark_id, &mut *(*solutions_data).lock().await).await;
    if let Some(sampled_nonces) = job.sampled_nonces.as_ref() {
        if num_solutions != sampled_nonces.len() as u32 {
            let mut state = (*state()).lock().await;
            (*state)
                .query_data
                .proofs
                .get_mut(&job.benchmark_id)
                .unwrap()
                .solutions_data
                .take();
            return Err(format!(
                "Failed to recompute solutions for {}",
                job.benchmark_id
            ));
        } else {
            update_status(&format!(
                "Finished. Recompute solutions for {}",
                job.benchmark_id
            ))
            .await;
            sleep(5000).await;
        }
    } else {
        update_status("Updating difficulty sampler with solutions").await;
        {
            let num_solutions = *solutions_count.lock().await;
            let mut state = state().lock().await;
            state
                .difficulty_samplers
                .get_mut(&job.settings.challenge_id)
                .unwrap()
                .update_with_solutions(&job.settings.difficulty, num_solutions);
        }

        if num_solutions == 0 {
            update_status("Finished. No solutions to submit").await;
        } else {
            update_status(&format!("Finished. Submitting {} solutions", num_solutions,)).await;
            let benchmark_id = match submit_benchmark::execute(&job).await {
                Ok(benchmark_id) => benchmark_id,
                Err(e) => {
                    let mut state = (*state()).lock().await;
                    state
                        .submission_errors
                        .insert(job.benchmark_id.clone(), e.clone());
                    return Err(e);
                }
            };
            update_status(&format!("Success. Benchmark {} submitted", benchmark_id)).await;
            let mut state = (*state()).lock().await;
            let QueryData {
                benchmarks, proofs, ..
            } = &mut (*state).query_data;
            let mut benchmark = benchmarks.remove(&job.benchmark_id).unwrap();
            let mut proof = proofs.remove(&job.benchmark_id).unwrap();
            benchmark.id = benchmark_id.clone();
            proof.benchmark_id = benchmark_id.clone();
            benchmarks.insert(benchmark_id.clone(), benchmark);
            proofs.insert(benchmark_id.clone(), proof);
        }
    }
    Ok(())
}

pub async fn drain_solutions(benchmark_id: &String, solutions_data: &mut Vec<SolutionData>) -> u32 {
    let mut state = (*state()).lock().await;
    let QueryData {
        benchmarks, proofs, ..
    } = &mut (*state).query_data;
    if let Some(benchmark) = benchmarks.get_mut(benchmark_id) {
        let proof = proofs.get_mut(benchmark_id).unwrap();
        if let Some(x) = benchmark.solutions_meta_data.as_mut() {
            x.extend(
                solutions_data
                    .iter()
                    .map(|x| SolutionMetaData::from(x.clone())),
            );
            benchmark.details.num_solutions = x.len() as u32;
        }
        let to_update = proof.solutions_data.as_mut().unwrap();
        to_update.extend(solutions_data.drain(..));
        to_update.len() as u32
    } else {
        0
    }
}
pub async fn start(num_workers: u32, ms_per_benchmark: u32) {
    {
        let mut state = (*state()).lock().await;
        if state.status != Status::Stopped {
            return;
        }
        state.status = Status::Starting;
    }
    spawn(async move {
        {
            let mut state = (*state()).lock().await;
            state.status = Status::Running("Starting".to_string());
        }
        loop {
            {
                let mut state = (*state()).lock().await;
                if state.status == Status::Stopping {
                    state.status = Status::Stopped;
                }
            }
            if let Err(e) = run_once(num_workers, ms_per_benchmark).await {
                update_status(&format!("Error: {:?}", e)).await;
                sleep(5000).await;
            }
        }
    });
}
pub async fn stop() {
    let mut state = (*state()).lock().await;
    match state.status {
        Status::Running(_) => {
            state.status = Status::Stopping;
        }
        _ => {}
    }
}
pub async fn select_algorithm(challenge_name: String, algorithm_name: String) {
    let mut state = (*state()).lock().await;
    state
        .selected_algorithms
        .insert(challenge_name, algorithm_name);
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
            status: Status::Stopped,
            timer: None,
            query_data,
            difficulty_samplers,
            selected_algorithms: HashMap::new(),
            job: None,
            submission_errors: HashMap::new(),
        })
    });
}
