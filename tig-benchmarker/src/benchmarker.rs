use crate::future_utils::{self, time, Mutex};
use once_cell::sync::OnceCell;
use rand::{
    distributions::{Alphanumeric, DistString, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
use rand_distr::Distribution;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use tig_api::*;
use tig_structs::{config::WasmVMConfig, core::*};
use tig_utils::*;
use tig_worker::compute_solution;

type Result<T> = std::result::Result<T, String>;

#[cfg_attr(feature = "browser", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Serialize, Clone, Debug)]
pub struct Duration {
    pub start: u64,
    pub end: u64,
    pub now: u64,
}

#[cfg_attr(feature = "browser", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Serialize, Debug, Clone)]
pub struct Job {
    benchmark_id: String,
    settings: BenchmarkSettings,
    duration: Duration,
    solution_signature_threshold: u32,
    nonce_iter: NonceIterator,
    wasm_vm_config: WasmVMConfig,
}

#[cfg_attr(feature = "browser", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Serialize, Debug, Clone)]
pub struct State {
    running: bool,
    status: HashMap<String, String>,
    latest_block: Option<Block>,
    benchmarker_data: Option<PlayerBlockData>,
    challenges: Vec<Challenge>,
    download_urls: HashMap<String, String>,
    algorithms_by_challenge: HashMap<String, Vec<Algorithm>>,
    selected_algorithms: HashMap<String, String>,
    benchmarks: HashMap<String, Benchmark>,
    proofs: HashMap<String, Proof>,
    frauds: HashMap<String, Fraud>,
    job: Option<Job>,
}

impl State {
    pub fn new() -> State {
        State {
            running: false,
            status: HashMap::new(),
            latest_block: None,
            algorithms_by_challenge: HashMap::new(),
            selected_algorithms: HashMap::new(),
            benchmarks: HashMap::new(),
            proofs: HashMap::new(),
            frauds: HashMap::new(),
            benchmarker_data: None,
            challenges: Vec::new(),
            download_urls: HashMap::new(),
            job: None,
        }
    }
}

#[cfg_attr(feature = "browser", wasm_bindgen::prelude::wasm_bindgen)]
#[derive(Serialize, Debug, Clone)]
pub struct NonceIterator {
    nonces: Option<Vec<u32>>,
    current: u32,
    attempts: u32,
}

impl NonceIterator {
    pub fn new(nonces: Option<Vec<u32>>) -> Self {
        Self {
            nonces,
            current: 0,
            attempts: 0,
        }
    }
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
    pub fn is_recompute(&self) -> bool {
        self.nonces.is_some()
    }
    pub fn is_finished(&self) -> bool {
        self.nonces.as_ref().is_some_and(|x| x.is_empty()) || self.current == u32::MAX
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

static STATE: OnceCell<Mutex<State>> = OnceCell::new();
static BLOBS: OnceCell<Mutex<HashMap<String, Vec<u8>>>> = OnceCell::new();
static API: OnceCell<Api> = OnceCell::new();
static PLAYER_ID: OnceCell<String> = OnceCell::new();
const BLOCK_DATA_POLLER_ID: &'static str = "Block data poller";
const WORKER_ID: &'static str = "Benchmark worker";
const MANAGER_ID: &'static str = "Benchmark manager";
const B_SUBMITTER_ID: &'static str = "Benchmark submitter";
const P_SUBMITTER_ID: &'static str = "Proof submitter";

pub fn mutex() -> &'static Mutex<State> {
    STATE.get().unwrap()
}

pub async fn start() {
    let mut state = mutex().lock().await;
    (*state).running = true;
}

pub async fn stop() {
    let mut state = mutex().lock().await;
    (*state).running = false;
}

pub async fn select_algorithm(challenge_id: String, algorithm_id: String) {
    let mut state = mutex().lock().await;
    (*state)
        .selected_algorithms
        .insert(challenge_id, algorithm_id);
}

pub async fn setup(api_url: String, api_key: String, player_id: String, num_workers: u32) {
    BLOBS.get_or_init(|| Mutex::new(HashMap::new()));
    STATE.get_or_init(|| Mutex::new(State::new()));
    API.get_or_init(|| Api::new(api_url, api_key));
    PLAYER_ID.get_or_init(|| player_id);

    update_block_data().await.unwrap();
    future_utils::spawn(async {
        update_status(BLOCK_DATA_POLLER_ID, "Running").await;
        loop {
            future_utils::sleep(30000).await;
            if let Err(e) = update_block_data().await {
                update_status(BLOCK_DATA_POLLER_ID, &e).await;
            }
        }
    });

    for _ in 0..num_workers {
        future_utils::spawn(async {
            update_status(WORKER_ID, "Stopped").await;
            let mut curr_running: bool = false;
            loop {
                let next_running = { mutex().lock().await.running };
                if curr_running != next_running {
                    curr_running = next_running;
                    if curr_running {
                        update_status(WORKER_ID, "Starting").await;
                    } else {
                        update_status(WORKER_ID, "Stopped").await;
                    }
                }
                if !curr_running {
                    future_utils::sleep(5000).await;
                    continue;
                }
                if let Err(e) = do_benchmark().await {
                    update_status(WORKER_ID, &e.to_string()).await;
                }
                future_utils::sleep(1000).await;
            }
        });
    }
    future_utils::spawn(async {
        update_status(MANAGER_ID, "Stopped").await;
        let mut curr_running: bool = false;
        loop {
            let next_running = { mutex().lock().await.running };
            if curr_running != next_running {
                curr_running = next_running;
                if curr_running {
                    update_status(MANAGER_ID, "Starting").await;
                } else {
                    update_status(MANAGER_ID, "Stopped").await;
                }
            }
            if !curr_running {
                future_utils::sleep(5000).await;
                continue;
            }
            if let Err(e) = do_manage_benchmark().await {
                update_status(MANAGER_ID, &e.to_string()).await;
                future_utils::sleep(5000).await;
            }
        }
    });
    future_utils::spawn(async {
        update_status(B_SUBMITTER_ID, "Stopped").await;
        let mut curr_running: bool = false;
        loop {
            let next_running = { mutex().lock().await.running };
            if curr_running != next_running {
                curr_running = next_running;
                if curr_running {
                    update_status(B_SUBMITTER_ID, "Starting").await;
                } else {
                    update_status(B_SUBMITTER_ID, "Stopped").await;
                }
            }
            if !curr_running {
                future_utils::sleep(5000).await;
                continue;
            }
            if let Err(e) = do_submit_benchmark().await {
                update_status(B_SUBMITTER_ID, &e.to_string()).await;
            }
            future_utils::sleep(5000).await;
        }
    });
    future_utils::spawn(async {
        update_status(P_SUBMITTER_ID, "Stopped").await;
        let mut curr_running: bool = false;
        loop {
            let next_running = { mutex().lock().await.running };
            if curr_running != next_running {
                curr_running = next_running;
                if curr_running {
                    update_status(P_SUBMITTER_ID, "Starting").await;
                } else {
                    update_status(P_SUBMITTER_ID, "Stopped").await;
                }
            }
            if !curr_running {
                future_utils::sleep(5000).await;
                continue;
            }
            if let Err(e) = do_submit_proof().await {
                update_status(P_SUBMITTER_ID, &e.to_string()).await;
            }
            future_utils::sleep(5000).await;
        }
    });
}

async fn get_latest_block_id() -> String {
    let state = mutex().lock().await;
    state.latest_block.as_ref().unwrap().id.clone()
}

async fn update_status(id: &str, status: &str) {
    let s = format!("[{}]: {}", id, status);
    println!("{}", s);
    #[cfg(feature = "browser")]
    web_sys::console::log_1(&s.into());
    let mut state = mutex().lock().await;
    (*state).status.insert(id.to_string(), status.to_string());
}

async fn get_latest_block() -> Result<Block> {
    let GetBlockResp { block, .. } = API
        .get()
        .unwrap()
        .get_block(GetBlockReq {
            id: None,
            round: None,
            height: None,
            include_data: false,
        })
        .await
        .map_err(|e| format!("Failed to get latest block: {:?}", e))?;
    Ok(block.unwrap())
}

async fn get_benchmarks() -> Result<(Vec<Benchmark>, Vec<Proof>, Vec<Fraud>)> {
    let GetBenchmarksResp {
        benchmarks,
        proofs,
        frauds,
        ..
    } = API
        .get()
        .unwrap()
        .get_benchmarks(GetBenchmarksReq {
            block_id: get_latest_block_id().await,
            player_id: PLAYER_ID.get().unwrap().clone(),
        })
        .await
        .map_err(|e| format!("Failed to get benchmarks: {:?}", e))?;
    Ok((benchmarks, proofs, frauds))
}

async fn get_benchmarker_data() -> Result<Option<PlayerBlockData>> {
    let GetPlayersResp { players, .. } = API
        .get()
        .unwrap()
        .get_players(GetPlayersReq {
            block_id: get_latest_block_id().await,
            player_type: PlayerType::Benchmarker,
        })
        .await
        .map_err(|e| format!("Failed to get players: {:?}", e))?;
    Ok(players
        .into_iter()
        .find(|x| x.id == *PLAYER_ID.get().unwrap())
        .map(|x| x.block_data.unwrap()))
}

async fn get_challenges() -> Result<Vec<Challenge>> {
    let GetChallengesResp { challenges, .. } = API
        .get()
        .unwrap()
        .get_challenges(GetChallengesReq {
            block_id: get_latest_block_id().await,
        })
        .await
        .map_err(|e| format!("Failed to get challenges: {:?}", e))?;
    Ok(challenges)
}

async fn get_algorithms() -> Result<(HashMap<String, Vec<Algorithm>>, HashMap<String, String>)> {
    let GetAlgorithmsResp {
        algorithms, wasms, ..
    } = API
        .get()
        .unwrap()
        .get_algorithms(GetAlgorithmsReq {
            block_id: get_latest_block_id().await,
        })
        .await
        .map_err(|e| format!("Failed to get algorithms: {:?}", e))?;
    let algorithms_by_challenge: HashMap<String, Vec<Algorithm>> =
        algorithms.into_iter().fold(HashMap::new(), |mut acc, x| {
            acc.entry(x.details.challenge_id.clone())
                .or_default()
                .push(x.clone());
            acc
        });
    let download_urls = wasms
        .into_iter()
        .filter(|x| x.details.download_url.is_some())
        .map(|x| (x.algorithm_id, x.details.download_url.unwrap()))
        .collect();
    Ok((algorithms_by_challenge, download_urls))
}

async fn update_block_data() -> Result<()> {
    let block = get_latest_block().await?;
    {
        let mut state = mutex().lock().await;
        (*state).latest_block = Some(block.clone());
    }
    let results = future_utils::join(
        get_algorithms(),
        get_benchmarker_data(),
        get_benchmarks(),
        get_challenges(),
    )
    .await?;
    let mut state = mutex().lock().await;
    let (algorithms_by_challenge, download_urls) = results.0?;
    (*state).algorithms_by_challenge = algorithms_by_challenge;
    (*state).download_urls = download_urls;

    let benchmarker_data = results.1?;
    (*state).benchmarker_data = benchmarker_data;

    let (benchmarks, proofs, frauds) = results.2?;
    (*state).benchmarks.retain(|_, x| {
        x.details.block_started
            >= block
                .details
                .height
                .saturating_sub(block.config().benchmark_submissions.lifespan_period)
    });
    let keys_to_keep: HashSet<String> = state.benchmarks.keys().cloned().collect();
    (*state)
        .proofs
        .retain(|_, x| keys_to_keep.contains(&x.benchmark_id));
    (*state)
        .frauds
        .retain(|_, x| keys_to_keep.contains(&x.benchmark_id));
    for x in benchmarks {
        (*state).benchmarks.insert(x.id.clone(), x);
    }
    for x in proofs {
        (*state).proofs.insert(x.benchmark_id.clone(), x);
    }
    for x in frauds {
        (*state).frauds.insert(x.benchmark_id.clone(), x);
    }

    let challenges = results.3?;
    (*state).challenges = challenges;
    Ok(())
}

async fn find_settings_to_recompute() -> Option<Job> {
    let state = mutex().lock().await;
    for (benchmark_id, benchmark) in state.benchmarks.iter() {
        if !state.proofs.contains_key(benchmark_id) && benchmark.state.is_some() {
            let sampled_nonces = benchmark.state().sampled_nonces.clone().unwrap();
            return Some(Job {
                benchmark_id: benchmark.id.clone(),
                settings: benchmark.settings.clone(),
                duration: Duration {
                    start: time(),
                    end: time() + 5000,
                    now: time(),
                },
                solution_signature_threshold: u32::MAX, // is fine unless the player has committed fraud
                nonce_iter: NonceIterator::new(Some(sampled_nonces)),
                wasm_vm_config: WasmVMConfig {
                    max_memory: u64::MAX,
                    max_fuel: u64::MAX,
                },
            });
        }
    }
    None
}

async fn pick_settings_to_benchmark() -> Job {
    let block_id = get_latest_block_id().await;
    let state = mutex().lock().await;
    let num_qualifiers_by_challenge = match &state.benchmarker_data {
        Some(data) => data.num_qualifiers_by_challenge.clone().unwrap(),
        None => HashMap::new(),
    };
    let percent_qualifiers_by_challenge: HashMap<String, f64> = state
        .challenges
        .iter()
        .map(|c| {
            let player_num_qualifiers = *num_qualifiers_by_challenge.get(&c.id).unwrap_or(&0);
            let challenge_num_qualifiers = *c.block_data().num_qualifiers();
            let percent = if player_num_qualifiers == 0 || challenge_num_qualifiers == 0 {
                0f64
            } else {
                (player_num_qualifiers as f64) / (challenge_num_qualifiers as f64)
            };
            (c.id.clone(), percent)
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(time() as u64);
    let challenge_weights: Vec<(String, f64)> = state
        .selected_algorithms
        .keys()
        .map(|challenge_id| {
            (
                challenge_id.clone(),
                1f64 - percent_qualifiers_by_challenge[challenge_id] + 1e-10f64,
            )
        })
        .collect();
    let dist = WeightedIndex::new(
        &challenge_weights
            .iter()
            .map(|w| w.1.clone())
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    let index = dist.sample(&mut rng);

    let random_challenge_id = challenge_weights[index].0.clone();
    let selected_algorithm_id = state
        .selected_algorithms
        .get(&random_challenge_id)
        .unwrap()
        .clone();

    let challenge = state
        .challenges
        .iter()
        .find(|c| c.id == random_challenge_id)
        .unwrap();
    let min_difficulty = challenge.details.min_difficulty();
    let max_difficulty = challenge.details.max_difficulty();
    let block_data = &challenge.block_data();
    let random_difficulty = block_data.base_frontier().sample(&mut rng).scale(
        &min_difficulty,
        &max_difficulty,
        *block_data.scaling_factor(),
    );
    Job {
        benchmark_id: Alphanumeric.sample_string(&mut rng, 32),
        settings: BenchmarkSettings {
            player_id: PLAYER_ID.get().unwrap().clone(),
            block_id,
            challenge_id: random_challenge_id,
            algorithm_id: selected_algorithm_id,
            difficulty: random_difficulty,
        },
        duration: Duration {
            start: time(),
            end: time() + 30000,
            now: time(),
        },
        solution_signature_threshold: *block_data.solution_signature_threshold(),
        nonce_iter: NonceIterator::new(None),
        wasm_vm_config: state
            .latest_block
            .as_ref()
            .unwrap()
            .config()
            .wasm_vm
            .clone(),
    }
}

async fn download_wasm_blob(algorithm_id: &String) -> Result<Vec<u8>> {
    // lock it so that only do 1 download
    let mut blobs = BLOBS.get().unwrap().lock().await;
    if let Some(wasm_blob) = blobs.get(algorithm_id) {
        Ok(wasm_blob.clone())
    } else {
        let state = mutex().lock().await;
        let download_url = state
            .download_urls
            .get(algorithm_id.as_str())
            .ok_or_else(|| format!("Algorithm {} does not have wasm download_url", algorithm_id))?;
        let wasm = get::<Vec<u8>>(download_url.as_str(), None)
            .await
            .map_err(|e| format!("Failed to download wasm from {}: {:?}", download_url, e))?;
        (*blobs).insert(algorithm_id.clone(), wasm.clone());
        Ok(wasm)
    }
}

async fn do_benchmark() -> Result<()> {
    let mut last_algorithm_id = "None".to_string();
    let mut blob = Vec::new();
    while let Some((job, Some(nonce))) = {
        let mut state = mutex().lock().await;
        if state.running {
            (*state)
                .job
                .as_mut()
                .map(|x| (x.clone(), x.nonce_iter.next()))
        } else {
            None
        }
    } {
        if last_algorithm_id != job.settings.algorithm_id {
            blob = download_wasm_blob(&job.settings.algorithm_id).await?;
            last_algorithm_id = job.settings.algorithm_id.clone();
        }
        if let Ok(solution_data) = compute_solution(
            &job.settings,
            nonce,
            blob.as_slice(),
            job.wasm_vm_config.max_memory,
            job.wasm_vm_config.max_fuel,
        )
        .map_err(|e| e.to_string())?
        {
            if solution_data.calc_solution_signature() <= job.solution_signature_threshold {
                let mut state = mutex().lock().await;
                if let Some(Some(solutions_meta_data)) = (*state)
                    .benchmarks
                    .get_mut(&job.benchmark_id)
                    .map(|x| x.solutions_meta_data.as_mut())
                {
                    solutions_meta_data.push(solution_data.clone().into());
                }
                if let Some(Some(solutions_data)) = (*state)
                    .proofs
                    .get_mut(&job.benchmark_id)
                    .map(|x| x.solutions_data.as_mut())
                {
                    solutions_data.push(solution_data);
                    if !job.nonce_iter.is_recompute() {
                        (*state)
                            .benchmarks
                            .get_mut(&job.benchmark_id)
                            .unwrap()
                            .details
                            .num_solutions += 1;
                    }
                } else {
                    return Ok(());
                }
            }
        }
        future_utils::sleep(1).await;
    }
    update_status(WORKER_ID, "Finished job").await;
    Ok(())
}

async fn do_manage_benchmark() -> Result<()> {
    update_status(MANAGER_ID, "Checking for any benchmarks to recompute").await;
    let job = if let Some(x) = find_settings_to_recompute().await {
        update_status(MANAGER_ID, "Found benchmark to recompute").await;
        x
    } else {
        update_status(MANAGER_ID, "Picking new settings to benchmark").await;
        pick_settings_to_benchmark().await
    };
    update_status(MANAGER_ID, &format!("{:?}", job.settings)).await;

    update_status(
        MANAGER_ID,
        &format!("Downloading algorithm: {}", job.settings.algorithm_id),
    )
    .await;
    download_wasm_blob(&job.settings.algorithm_id).await?;

    update_status(MANAGER_ID, &format!("Setting up benchmark")).await;
    let benchmark_id = job.benchmark_id.clone();
    let mut state = mutex().lock().await;
    if !job.nonce_iter.is_recompute() {
        let block_started = state.latest_block.as_ref().unwrap().details.height.clone();
        (*state).benchmarks.insert(
            benchmark_id.clone(),
            Benchmark {
                id: benchmark_id.clone(),
                settings: job.settings.clone(),
                details: BenchmarkDetails {
                    block_started,
                    num_solutions: 0,
                },
                state: None,
                solutions_meta_data: Some(Vec::new()),
                solution_data: None,
            },
        );
    }
    (*state).proofs.insert(
        benchmark_id.clone(),
        Proof {
            benchmark_id: benchmark_id.clone(),
            state: None,
            solutions_data: Some(Vec::new()),
        },
    );
    (*state).job = Some(job);
    drop(state);

    loop {
        let mut state = mutex().lock().await;
        if !state.running {
            break;
        }
        let job = (*state).job.as_mut().unwrap();
        job.duration.now = time();
        if job.duration.now > job.duration.end {
            break;
        }

        if job.nonce_iter.is_finished() {
            break;
        }
        let num_attempts = job.nonce_iter.attempts();
        let num_solutions = state
            .proofs
            .get(&benchmark_id)
            .unwrap()
            .solutions_data()
            .len();
        drop(state);
        update_status(
            MANAGER_ID,
            &format!(
                "Computed {} solutions out of {} instances",
                num_solutions, num_attempts
            ),
        )
        .await;

        future_utils::sleep(200).await;
    }

    let mut state = mutex().lock().await;
    let num_solutions = state
        .proofs
        .get(&benchmark_id)
        .unwrap()
        .solutions_data()
        .len();
    let num_attempts = state.job.as_ref().unwrap().nonce_iter.attempts();
    (*state).job = None;
    if num_solutions == 0 {
        (*state).benchmarks.remove(&benchmark_id);
        (*state).proofs.remove(&benchmark_id);
    }
    drop(state);
    update_status(
        MANAGER_ID,
        &format!(
            "Finished. Computed {} solutions out of {} instances",
            num_solutions, num_attempts
        ),
    )
    .await;
    Ok(())
}

async fn do_submit_benchmark() -> Result<()> {
    update_status(B_SUBMITTER_ID, "Finding benchmark to submit").await;
    let benchmark_to_submit = {
        let mut state = mutex().lock().await;
        let State {
            ref mut benchmarks,
            ref proofs,
            ref job,
            ..
        } = *state;
        let job_benchmark_id = job.as_ref().map(|x| &x.benchmark_id);
        benchmarks
            .values_mut()
            .find(|benchmark| {
                job_benchmark_id != Some(&benchmark.id) && benchmark.solutions_meta_data.is_some()
            })
            .map(|benchmark| {
                (
                    benchmark.id.clone(),
                    benchmark.settings.clone(),
                    benchmark.solutions_meta_data.take().unwrap(),
                    proofs
                        .get(&benchmark.id)
                        .unwrap()
                        .solutions_data()
                        .first()
                        .unwrap()
                        .clone(),
                )
            })
    };
    if let Some((old_benchmark_id, settings, solutions_meta_data, solution_data)) =
        benchmark_to_submit
    {
        update_status(
            B_SUBMITTER_ID,
            &format!("Submitting benchmark: {:?}", settings),
        )
        .await;
        let resp = API
            .get()
            .unwrap()
            .submit_benchmark(SubmitBenchmarkReq {
                settings,
                solutions_meta_data,
                solution_data,
            })
            .await
            .map_err(|e| format!("Failed to submit benchmark: {:?}", e))?;
        update_status(B_SUBMITTER_ID, &format!("{:?}", resp)).await;
        if resp.benchmark_id != old_benchmark_id {
            let mut state = mutex().lock().await;
            let mut benchmark = (*state).benchmarks.remove(&old_benchmark_id).unwrap();
            let mut proof = (*state).proofs.remove(&old_benchmark_id).unwrap();
            benchmark.id = resp.benchmark_id.clone();
            proof.benchmark_id = resp.benchmark_id.clone();
            (*state)
                .benchmarks
                .insert(resp.benchmark_id.clone(), benchmark);
            (*state).proofs.insert(resp.benchmark_id.clone(), proof);
        }
    } else {
        update_status(B_SUBMITTER_ID, "No benchmark to submit").await;
    }
    Ok(())
}

async fn do_submit_proof() -> Result<()> {
    update_status(P_SUBMITTER_ID, "Finding proof to submit").await;
    let proof_to_submit = {
        let mut state = mutex().lock().await;
        let State {
            ref benchmarks,
            ref mut proofs,
            ref job,
            ..
        } = *state;
        let job_benchmark_id = job.as_ref().map(|x| &x.benchmark_id);
        proofs
            .values_mut()
            .find(|x| {
                job_benchmark_id != Some(&x.benchmark_id)
                    && x.solutions_data.is_some()
                    && benchmarks
                        .get(&x.benchmark_id)
                        .is_some_and(|x| x.state.is_some())
            })
            .map(|x| {
                let state = benchmarks
                    .get(&x.benchmark_id)
                    .unwrap()
                    .state
                    .as_ref()
                    .unwrap();
                let sampled_nonces: HashSet<u32> =
                    state.sampled_nonces.clone().unwrap().into_iter().collect();
                let mut solutions_data = x.solutions_data.take().unwrap();
                solutions_data.retain(|x| sampled_nonces.contains(&x.nonce));
                (x.benchmark_id.clone(), solutions_data)
            })
    };
    if let Some((benchmark_id, solutions_data)) = proof_to_submit {
        update_status(
            P_SUBMITTER_ID,
            &format!("Submitting proof for benchmark {}", benchmark_id),
        )
        .await;
        let resp = API
            .get()
            .unwrap()
            .submit_proof(SubmitProofReq {
                benchmark_id: benchmark_id.clone(),
                solutions_data: solutions_data.into(),
            })
            .await
            .map_err(|e| {
                format!(
                    "Failed to submit proof for benchmark {}: {:?}",
                    benchmark_id, e
                )
            })?;
        update_status(P_SUBMITTER_ID, &format!("{:?}", resp)).await;
    } else {
        update_status(P_SUBMITTER_ID, "No proof to submit").await;
    }
    Ok(())
}
