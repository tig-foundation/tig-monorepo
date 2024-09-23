use crate::context::*;
use logging_timer::time;
use rand::{prelude::SliceRandom, rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use tig_structs::{config::*, core::*};
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(ctx: &T) -> String {
    let (mut block, mut cache) = create_block(ctx).await;
    confirm_mempool_challenges(&block, &mut cache).await;
    confirm_mempool_algorithms(&block, &mut cache).await;
    confirm_mempool_precommits(&mut block, &mut cache).await;
    confirm_mempool_benchmarks(&block, &mut cache).await;
    confirm_mempool_proofs(&block, &mut cache).await;
    confirm_mempool_frauds(&block, &mut cache).await;
    confirm_mempool_topups(&block, &mut cache).await;
    confirm_mempool_wasms(&block, &mut cache).await;
    update_deposits(ctx, &block, &mut cache).await;
    update_cutoffs(&block, &mut cache).await;
    update_qualifiers(&block, &mut cache).await;
    update_frontiers(&block, &mut cache).await;
    update_solution_signature_thresholds(&block, &mut cache).await;
    update_fees(&block, &mut cache).await;
    update_influence(&block, &mut cache).await;
    update_adoption(&mut cache).await;
    update_innovator_rewards(&block, &mut cache).await;
    update_benchmarker_rewards(&block, &mut cache).await;
    update_merge_points(&block, &mut cache).await;
    update_merges(&block, &mut cache).await;
    commit_changes(ctx, &block, &mut cache).await;
    block.id
}

struct AddBlockCache {
    pub mempool_challenges: Vec<Challenge>,
    pub mempool_algorithms: Vec<Algorithm>,
    pub mempool_benchmarks: Vec<Benchmark>,
    pub mempool_precommits: Vec<Precommit>,
    pub mempool_proofs: Vec<Proof>,
    pub mempool_frauds: Vec<Fraud>,
    pub mempool_topups: Vec<TopUp>,
    pub mempool_wasms: Vec<Wasm>,
    pub confirmed_precommits: HashMap<String, Precommit>,
    pub active_challenges: HashMap<String, Challenge>,
    pub active_algorithms: HashMap<String, Algorithm>,
    pub active_solutions: HashMap<String, (BenchmarkSettings, u32)>,
    pub active_players: HashMap<String, Player>,
    pub active_fee_players: HashMap<String, Player>,
    pub prev_challenges: HashMap<String, Challenge>,
    pub prev_algorithms: HashMap<String, Algorithm>,
    pub prev_players: HashMap<String, Player>,
}

#[time]
async fn setup_cache<T: Context>(
    ctx: &T,
    details: &BlockDetails,
    config: &ProtocolConfig,
) -> AddBlockCache {
    let from_block_started = details
        .height
        .saturating_sub(config.benchmark_submissions.lifespan_period);
    let mut confirmed_precommits = HashMap::new();
    for precommit in ctx
        .get_precommits(PrecommitsFilter::Confirmed { from_block_started })
        .await
        .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
    {
        confirmed_precommits.insert(precommit.benchmark_id.clone(), precommit);
    }
    let mut mempool_challenges = Vec::new();
    for mut challenge in ctx
        .get_challenges(ChallengesFilter::Mempool, None)
        .await
        .unwrap_or_else(|e| panic!("get_challenges error: {:?}", e))
    {
        challenge.state = Some(ChallengeState {
            block_confirmed: None,
            round_active: None,
        });
        mempool_challenges.push(challenge);
    }
    let mut mempool_algorithms = Vec::new();
    for mut algorithm in ctx
        .get_algorithms(AlgorithmsFilter::Mempool, None, false)
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
    {
        algorithm.state = Some(AlgorithmState {
            block_confirmed: None,
            round_submitted: None,
            round_pushed: None,
            round_merged: None,
            banned: false,
        });
        mempool_algorithms.push(algorithm);
    }
    let mut mempool_benchmarks = Vec::new();
    for mut benchmark in ctx
        .get_benchmarks(BenchmarksFilter::Mempool { from_block_started }, true)
        .await
        .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e))
    {
        if !confirmed_precommits.contains_key(&benchmark.id) {
            continue;
        }
        benchmark.state = Some(BenchmarkState {
            block_confirmed: None,
            sampled_nonces: None,
        });
        mempool_benchmarks.push(benchmark);
    }
    let mut mempool_precommits = Vec::new();
    for mut precommit in ctx
        .get_precommits(PrecommitsFilter::Mempool { from_block_started })
        .await
        .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
    {
        precommit.state = Some(PrecommitState {
            block_confirmed: None,
            rand_hash: None,
        });
        mempool_precommits.push(precommit);
    }
    let mut mempool_proofs = Vec::new();
    for mut proof in ctx
        .get_proofs(ProofsFilter::Mempool { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("mempool_proofs error: {:?}", e))
    {
        if !confirmed_precommits.contains_key(&proof.benchmark_id) {
            continue;
        }
        proof.state = Some(ProofState {
            block_confirmed: None,
            submission_delay: None,
        });
        mempool_proofs.push(proof);
    }
    let mut mempool_frauds = Vec::new();
    for mut fraud in ctx
        .get_frauds(FraudsFilter::Mempool { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("mempool_frauds error: {:?}", e))
    {
        if !confirmed_precommits.contains_key(&fraud.benchmark_id) {
            continue;
        }
        fraud.state = Some(FraudState {
            block_confirmed: None,
        });
        mempool_frauds.push(fraud);
    }
    let mut mempool_topups = Vec::new();
    for mut topup in ctx
        .get_topups(TopUpsFilter::Mempool)
        .await
        .unwrap_or_else(|e| panic!("get_topups error: {:?}", e))
    {
        topup.state = Some(TopUpState {
            block_confirmed: None,
        });
        mempool_topups.push(topup);
    }
    let mut mempool_wasms = Vec::new();
    for mut wasm in ctx
        .get_wasms(WasmsFilter::Mempool)
        .await
        .unwrap_or_else(|e| panic!("mempool_wasms error: {:?}", e))
    {
        wasm.state = Some(WasmState {
            block_confirmed: None,
        });
        mempool_wasms.push(wasm);
    }
    let mut active_challenges = HashMap::new();
    for mut challenge in ctx
        .get_challenges(ChallengesFilter::Confirmed, None)
        .await
        .unwrap_or_else(|e| panic!("get_challenges error: {:?}", e))
    {
        if challenge
            .state
            .as_ref()
            .unwrap()
            .round_active
            .is_some_and(|r| r <= details.round)
        {
            challenge.block_data = Some(ChallengeBlockData {
                num_qualifiers: None,
                solution_signature_threshold: None,
                scaled_frontier: None,
                base_frontier: None,
                scaling_factor: None,
                qualifier_difficulties: None,
                base_fee: None,
                per_nonce_fee: None,
            });
            active_challenges.insert(challenge.id.clone(), challenge);
        }
    }
    let algorithms = ctx
        .get_algorithms(AlgorithmsFilter::Confirmed, None, false)
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e));
    let challenges_with_algorithms = algorithms
        .iter()
        .filter(|a| a.state().round_pushed.is_some())
        .map(|a| a.details.challenge_id.clone())
        .collect::<HashSet<String>>();
    let mut active_algorithms = HashMap::new(); // FIXME round_pushed 1 for new challenges
    for mut algorithm in algorithms {
        let state = algorithm.state.as_mut().unwrap();
        let round_pushed = *state.round_submitted()
            + if challenges_with_algorithms.contains(&algorithm.details.challenge_id) {
                config.algorithm_submissions.push_delay
            } else {
                1
            };
        let wasm = ctx
            .get_wasms(WasmsFilter::AlgorithmId(algorithm.id.clone()))
            .await
            .unwrap_or_else(|e| panic!("get_wasms error: {:?}", e));
        if !state.banned
            && details.round >= *state.round_pushed.as_ref().unwrap_or(&round_pushed)
            && wasm.first().is_some_and(|w| w.details.compile_success)
        {
            algorithm.block_data = Some(AlgorithmBlockData {
                reward: None,
                adoption: None,
                merge_points: None,
                num_qualifiers_by_player: None,
                round_earnings: None,
            });
            if state.round_pushed.is_none() {
                state.round_pushed = Some(round_pushed);
            }
            active_algorithms.insert(algorithm.id.clone(), algorithm);
        }
    }
    let mut active_solutions = HashMap::new();
    for proof in ctx
        .get_proofs(ProofsFilter::Confirmed { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
    {
        let benchmark_id = &proof.benchmark_id;
        let fraud = ctx
            .get_frauds(FraudsFilter::BenchmarkId(benchmark_id.clone()), false)
            .await
            .unwrap_or_else(|e| panic!("get_frauds error: {:?}", e))
            .pop();
        if fraud.is_some_and(|f| f.state.is_some()) {
            continue;
        }
        let state = proof.state();
        let submission_delay = *state.submission_delay();
        let block_confirmed = *state.block_confirmed();
        let block_active = block_confirmed
            + (submission_delay as f64 * config.benchmark_submissions.submission_delay_multiplier)
                as u32;
        if block_active > details.height {
            continue;
        }
        let benchmark = ctx
            .get_benchmarks(BenchmarksFilter::Id(benchmark_id.clone()), false)
            .await
            .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
            .pop()
            .unwrap();
        let settings = &confirmed_precommits[benchmark_id].settings;
        active_solutions.insert(
            benchmark_id.clone(),
            (settings.clone(), benchmark.details.num_solutions),
        );
    }
    let mut active_players = HashMap::new();
    for (settings, _) in active_solutions.values() {
        let mut player = ctx
            .get_players(PlayersFilter::Id(settings.player_id.clone()), None)
            .await
            .unwrap()
            .pop()
            .unwrap();
        player.block_data = Some(PlayerBlockData {
            reward: None,
            influence: None,
            cutoff: None,
            imbalance: None,
            imbalance_penalty: None,
            num_qualifiers_by_challenge: None,
            round_earnings: None,
            deposit: None,
            rolling_deposit: None,
            qualifying_percent_rolling_deposit: None,
        });
        active_players.insert(player.id.clone(), player);
    }
    let mut active_fee_players = HashMap::new();
    for topup in mempool_topups.iter() {
        let mut player = ctx
            .get_players(PlayersFilter::Id(topup.details.player_id.clone()), None)
            .await
            .unwrap()
            .pop()
            .unwrap();
        if player.state.is_none() {
            player.state = Some(PlayerState {
                total_fees_paid: Some(PreciseNumber::from(0)),
                available_fee_balance: Some(PreciseNumber::from(0)),
            });
        }
        active_fee_players.insert(player.id.clone(), player);
    }
    for precommit in mempool_precommits.iter() {
        let mut player = ctx
            .get_players(
                PlayersFilter::Id(precommit.settings.player_id.clone()),
                None,
            )
            .await
            .unwrap()
            .pop()
            .unwrap();
        if player.state.is_none() {
            player.state = Some(PlayerState {
                total_fees_paid: Some(PreciseNumber::from(0)),
                available_fee_balance: Some(PreciseNumber::from(0)),
            });
        }
        active_fee_players.insert(player.id.clone(), player);
    }
    let mut prev_players = HashMap::<String, Player>::new();
    for player_id in active_players.keys() {
        let player = ctx
            .get_players(
                PlayersFilter::Id(player_id.clone()),
                Some(BlockFilter::Id(details.prev_block_id.clone())),
            )
            .await
            .unwrap()
            .pop()
            .unwrap();
        prev_players.insert(player.id.clone(), player);
    }
    let mut prev_algorithms = HashMap::<String, Algorithm>::new();
    for algorithm_id in active_algorithms.keys() {
        let algorithm = ctx
            .get_algorithms(
                AlgorithmsFilter::Id(algorithm_id.clone()),
                Some(BlockFilter::Id(details.prev_block_id.clone())),
                false,
            )
            .await
            .unwrap()
            .pop()
            .unwrap();
        prev_algorithms.insert(algorithm_id.clone(), algorithm);
    }
    let mut prev_challenges = HashMap::<String, Challenge>::new();
    for challenge_id in active_challenges.keys() {
        let challenge = ctx
            .get_challenges(
                ChallengesFilter::Id(challenge_id.clone()),
                Some(BlockFilter::Id(details.prev_block_id.clone())),
            )
            .await
            .unwrap()
            .pop()
            .unwrap();
        prev_challenges.insert(challenge_id.clone(), challenge);
    }
    AddBlockCache {
        mempool_challenges,
        mempool_algorithms,
        mempool_benchmarks,
        mempool_precommits,
        mempool_proofs,
        mempool_frauds,
        mempool_topups,
        mempool_wasms,
        confirmed_precommits,
        active_challenges,
        active_algorithms,
        active_solutions,
        active_players,
        active_fee_players,
        prev_challenges,
        prev_algorithms,
        prev_players,
    }
}

#[time]
async fn create_block<T: Context>(ctx: &T) -> (Block, AddBlockCache) {
    let latest_block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("No latest block found");
    let config = ctx
        .get_config()
        .await
        .unwrap_or_else(|e| panic!("get_config error: {:?}", e));
    let height = latest_block.details.height + 1;
    let mut details = BlockDetails {
        prev_block_id: latest_block.id.clone(),
        height,
        round: height / config.rounds.blocks_per_round + 1,
        eth_block_num: Some(ctx.get_latest_eth_block_num().await.unwrap()),
        fees_paid: Some(PreciseNumber::from(0)),
        num_confirmed_challenges: None,
        num_confirmed_algorithms: None,
        num_confirmed_benchmarks: None,
        num_confirmed_precommits: None,
        num_confirmed_proofs: None,
        num_confirmed_frauds: None,
        num_confirmed_topups: None,
        num_confirmed_wasms: None,
        num_active_challenges: None,
        num_active_algorithms: None,
        num_active_benchmarks: None,
        num_active_players: None,
    };
    let cache = setup_cache(ctx, &details, &config).await;
    details.num_confirmed_challenges = Some(cache.mempool_challenges.len() as u32);
    details.num_confirmed_algorithms = Some(cache.mempool_algorithms.len() as u32);
    details.num_confirmed_benchmarks = Some(cache.mempool_benchmarks.len() as u32);
    details.num_confirmed_precommits = Some(cache.mempool_precommits.len() as u32);
    details.num_confirmed_proofs = Some(cache.mempool_proofs.len() as u32);
    details.num_confirmed_frauds = Some(cache.mempool_frauds.len() as u32);
    details.num_confirmed_topups = Some(cache.mempool_topups.len() as u32);
    details.num_confirmed_wasms = Some(cache.mempool_wasms.len() as u32);
    details.num_active_challenges = Some(cache.active_challenges.len() as u32);
    details.num_active_algorithms = Some(cache.active_algorithms.len() as u32);
    details.num_active_benchmarks = Some(cache.active_solutions.len() as u32);
    details.num_active_players = Some(cache.active_players.len() as u32);

    let data = BlockData {
        confirmed_challenge_ids: cache
            .mempool_challenges
            .iter()
            .map(|c| c.id.clone())
            .collect(),
        confirmed_algorithm_ids: cache
            .mempool_algorithms
            .iter()
            .map(|a| a.id.clone())
            .collect(),
        confirmed_benchmark_ids: cache
            .mempool_benchmarks
            .iter()
            .map(|b| b.id.clone())
            .collect(),
        confirmed_fraud_ids: cache
            .mempool_frauds
            .iter()
            .map(|f| f.benchmark_id.clone())
            .collect(),
        confirmed_precommit_ids: cache
            .mempool_precommits
            .iter()
            .map(|p| p.benchmark_id.clone())
            .collect(),
        confirmed_proof_ids: cache
            .mempool_proofs
            .iter()
            .map(|p| p.benchmark_id.clone())
            .collect(),
        confirmed_topup_ids: cache.mempool_topups.iter().map(|t| t.id.clone()).collect(),
        confirmed_wasm_ids: cache
            .mempool_wasms
            .iter()
            .map(|w| w.algorithm_id.clone())
            .collect(),
        active_challenge_ids: cache.active_challenges.keys().cloned().collect(),
        active_algorithm_ids: cache.active_algorithms.keys().cloned().collect(),
        active_benchmark_ids: cache.active_solutions.keys().cloned().collect(),
        active_player_ids: cache.active_players.keys().cloned().collect(),
    };

    let block_id = ctx
        .add_block(details.clone(), data.clone(), config.clone())
        .await
        .unwrap_or_else(|e| panic!("add_block error: {:?}", e));

    (
        Block {
            id: block_id,
            config: Some(config.clone()),
            details,
            data: Some(data),
        },
        cache,
    )
}

#[time]
async fn confirm_mempool_challenges(block: &Block, cache: &mut AddBlockCache) {
    for challenge in cache.mempool_challenges.iter_mut() {
        let state = challenge.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_algorithms(block: &Block, cache: &mut AddBlockCache) {
    for algorithm in cache.mempool_algorithms.iter_mut() {
        let state = algorithm.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);
        state.round_submitted = Some(block.details.round);
    }
}

#[time]
async fn confirm_mempool_precommits(block: &mut Block, cache: &mut AddBlockCache) {
    for precommit in cache.mempool_precommits.iter_mut() {
        let state = precommit.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);
        state.rand_hash = Some(block.id.clone());

        let player_state = cache
            .active_fee_players
            .get_mut(&precommit.settings.player_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        let fee_paid = *precommit.details.fee_paid.as_ref().unwrap();
        *player_state.available_fee_balance.as_mut().unwrap() -= fee_paid;
        *player_state.total_fees_paid.as_mut().unwrap() += fee_paid;
        *block.details.fees_paid.as_mut().unwrap() += fee_paid;
    }
}

#[time]
async fn confirm_mempool_benchmarks(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    for benchmark in cache.mempool_benchmarks.iter_mut() {
        let seed = u64s_from_str(format!("{:?}|{:?}", block.id, benchmark.id).as_str())[0];
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sampled_nonces = HashSet::new();
        let mut solution_nonces = benchmark
            .solution_nonces
            .as_ref()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<u64>>();
        if solution_nonces.len() > 0 {
            solution_nonces.shuffle(&mut rng);
            for nonce in solution_nonces
                .iter()
                .take(config.benchmark_submissions.max_samples)
            {
                sampled_nonces.insert(*nonce);
            }
        }
        let precommit = &cache.confirmed_precommits[&benchmark.id];
        let solution_nonces = benchmark.solution_nonces.as_ref().unwrap();
        let num_nonces = *precommit.details.num_nonces.as_ref().unwrap() as usize;
        if num_nonces > solution_nonces.len() {
            if num_nonces > solution_nonces.len() * 2 {
                // use rejection sampling
                let stop_length = config
                    .benchmark_submissions
                    .max_samples
                    .min(num_nonces - solution_nonces.len())
                    + sampled_nonces.len();
                while sampled_nonces.len() < stop_length {
                    let nonce = rng.gen_range(0..num_nonces as u64);
                    if sampled_nonces.contains(&nonce) || solution_nonces.contains(&nonce) {
                        continue;
                    }
                    sampled_nonces.insert(nonce);
                }
            } else {
                let mut non_solution_nonces: Vec<u64> = (0..num_nonces as u64)
                    .filter(|n| !solution_nonces.contains(n))
                    .collect();
                non_solution_nonces.shuffle(&mut rng);
                for nonce in non_solution_nonces
                    .iter()
                    .take(config.benchmark_submissions.max_samples)
                {
                    sampled_nonces.insert(*nonce);
                }
            }
        }

        let state = benchmark.state.as_mut().unwrap();
        state.sampled_nonces = Some(sampled_nonces);
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_proofs(block: &Block, cache: &mut AddBlockCache) {
    for proof in cache.mempool_proofs.iter_mut() {
        let precommit = &cache.confirmed_precommits[&proof.benchmark_id];
        let state = proof.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);
        state.submission_delay = Some(block.details.height - precommit.details.block_started);
    }
}

#[time]
async fn confirm_mempool_frauds(block: &Block, cache: &mut AddBlockCache) {
    // Future Todo: slash player's rewards from past day
    for fraud in cache.mempool_frauds.iter_mut() {
        let state = fraud.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_topups(block: &Block, cache: &mut AddBlockCache) {
    for topup in cache.mempool_topups.iter_mut() {
        let state = topup.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);

        let player_state = cache
            .active_fee_players
            .get_mut(&topup.details.player_id)
            .unwrap()
            .state
            .as_mut()
            .unwrap();
        *player_state.available_fee_balance.as_mut().unwrap() += topup.details.amount;
    }
}

#[time]
async fn confirm_mempool_wasms(block: &Block, cache: &mut AddBlockCache) {
    for wasm in cache.mempool_wasms.iter_mut() {
        let state = wasm.state.as_mut().unwrap();
        state.block_confirmed = Some(block.details.height);
    }
}

#[time]
async fn update_deposits<T: Context>(ctx: &T, block: &Block, cache: &mut AddBlockCache) {
    let decay = match &block
        .config()
        .optimisable_proof_of_work
        .rolling_deposit_decay
    {
        Some(decay) => PreciseNumber::from_f64(*decay),
        None => return, // Proof of deposit not implemented for these blocks
    };
    let eth_block_num = block.details.eth_block_num();
    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    for player in cache.active_players.values_mut() {
        let rolling_deposit = match &cache.prev_players.get(&player.id).unwrap().block_data {
            Some(data) => data.rolling_deposit.clone(),
            None => None,
        }
        .unwrap_or_else(|| zero.clone());

        let data = player.block_data.as_mut().unwrap();
        let deposit = ctx
            .get_player_deposit(eth_block_num, &player.id)
            .await
            .unwrap()
            .unwrap_or_else(|| zero.clone());
        data.rolling_deposit = Some(decay * rolling_deposit + (one - decay) * deposit);
        data.deposit = Some(deposit);
        data.qualifying_percent_rolling_deposit = Some(zero.clone());
    }
}

#[time]
async fn update_cutoffs(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    let mut phase_in_challenge_ids: HashSet<String> =
        cache.active_challenges.keys().cloned().collect();
    for algorithm in cache.active_algorithms.values() {
        if algorithm
            .state()
            .round_pushed
            .is_some_and(|r| r + 1 <= block.details.round)
        {
            phase_in_challenge_ids.remove(&algorithm.details.challenge_id);
        }
    }

    let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    for (settings, num_solutions) in cache.active_solutions.values() {
        *num_solutions_by_player_by_challenge
            .entry(settings.player_id.clone())
            .or_default()
            .entry(settings.challenge_id.clone())
            .or_default() += *num_solutions;
    }

    for (player_id, num_solutions_by_challenge) in num_solutions_by_player_by_challenge.iter() {
        let data = cache
            .active_players
            .get_mut(player_id)
            .unwrap()
            .block_data
            .as_mut()
            .unwrap();
        let phase_in_start = (block.details.round - 1) * config.rounds.blocks_per_round;
        let phase_in_period = config.qualifiers.cutoff_phase_in_period.unwrap();
        let phase_in_end = phase_in_start + phase_in_period;
        let min_cutoff = config.qualifiers.min_cutoff.clone().unwrap();
        let min_num_solutions = cache
            .active_challenges
            .keys()
            .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
            .min()
            .unwrap();
        let mut cutoff = min_cutoff
            .max((min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil() as u32);
        if phase_in_challenge_ids.len() > 0 && phase_in_end > block.details.height {
            let phase_in_min_num_solutions = cache
                .active_challenges
                .keys()
                .filter(|&id| !phase_in_challenge_ids.contains(id))
                .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
                .min()
                .unwrap();
            let phase_in_cutoff = min_cutoff.max(
                (phase_in_min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil()
                    as u32,
            );
            let phase_in_weight =
                (phase_in_end - block.details.height) as f64 / phase_in_period as f64;
            cutoff = (phase_in_cutoff as f64 * phase_in_weight
                + cutoff as f64 * (1.0 - phase_in_weight)) as u32;
        }
        data.cutoff = Some(cutoff);
    }
}

#[time]
async fn update_solution_signature_thresholds(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let confirmed_proof_ids = &block.data().confirmed_proof_ids;
    let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    let mut new_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    for (benchmark_id, (settings, num_solutions)) in cache.active_solutions.iter() {
        *num_solutions_by_player_by_challenge
            .entry(settings.player_id.clone())
            .or_default()
            .entry(settings.challenge_id.clone())
            .or_default() += *num_solutions;
        if confirmed_proof_ids.contains(benchmark_id) {
            *new_solutions_by_player_by_challenge
                .entry(settings.player_id.clone())
                .or_default()
                .entry(settings.challenge_id.clone())
                .or_default() += *num_solutions;
        }
    }

    let mut solutions_rate_by_challenge = HashMap::<String, u32>::new();
    for (player_id, new_solutions_by_challenge) in new_solutions_by_player_by_challenge.iter() {
        let cutoff = *cache
            .active_players
            .get(player_id)
            .unwrap()
            .block_data()
            .cutoff();
        for (challenge_id, new_solutions) in new_solutions_by_challenge.iter() {
            let num_solutions =
                num_solutions_by_player_by_challenge[player_id][challenge_id].clone();
            *solutions_rate_by_challenge
                .entry(challenge_id.clone())
                .or_default() +=
                new_solutions.saturating_sub(num_solutions - cutoff.min(num_solutions));
        }
    }

    for challenge in cache.active_challenges.values_mut() {
        let max_threshold = u32::MAX as f64;
        let current_threshold = match &cache.prev_challenges.get(&challenge.id).unwrap().block_data
        {
            Some(data) => *data.solution_signature_threshold() as f64,
            None => max_threshold,
        };
        let current_rate = *solutions_rate_by_challenge.get(&challenge.id).unwrap_or(&0) as f64;

        let equilibrium_rate = config.qualifiers.total_qualifiers_threshold as f64
            / config.benchmark_submissions.lifespan_period as f64;
        let target_rate = config.solution_signature.equilibrium_rate_multiplier * equilibrium_rate;
        let target_threshold = if current_rate == 0.0 {
            max_threshold
        } else {
            (current_threshold * target_rate / current_rate).clamp(0.0, max_threshold)
        };

        let threshold_decay = config.solution_signature.threshold_decay.unwrap_or(0.99);
        let block_data = challenge.block_data.as_mut().unwrap();
        block_data.solution_signature_threshold = Some(
            (current_threshold * threshold_decay + target_threshold * (1.0 - threshold_decay))
                .clamp(0.0, max_threshold) as u32,
        );
    }
}

#[time]
async fn update_fees(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    let PrecommitSubmissionsConfig {
        min_base_fee,
        min_per_nonce_fee,
        target_num_precommits,
        max_fee_percentage_delta,
        ..
    } = config.precommit_submissions();
    let num_precommits_by_challenge = cache.mempool_precommits.iter().fold(
        HashMap::<String, u32>::new(),
        |mut map, precommit| {
            *map.entry(precommit.settings.challenge_id.clone())
                .or_default() += 1;
            map
        },
    );
    let target_num_precommits = PreciseNumber::from(*target_num_precommits);
    let max_fee_percent_delta = PreciseNumber::from_f64(*max_fee_percentage_delta);
    let one = PreciseNumber::from(1);
    let zero = PreciseNumber::from(0);
    for challenge in cache.active_challenges.values_mut() {
        let num_precommits = PreciseNumber::from(
            num_precommits_by_challenge
                .get(&challenge.id)
                .unwrap_or(&0)
                .clone(),
        );
        let mut percent_delta = num_precommits / target_num_precommits;
        if num_precommits >= target_num_precommits {
            percent_delta = percent_delta - one;
        } else {
            percent_delta = one - percent_delta;
        }
        if percent_delta > max_fee_percent_delta {
            percent_delta = max_fee_percent_delta;
        }
        let current_base_fee =
            match &cache.prev_challenges.get(&challenge.id).unwrap().block_data {
                Some(data) => data.base_fee.as_ref().unwrap_or(&zero),
                None => &zero,
            }
            .clone();
        let mut base_fee = if num_precommits >= target_num_precommits {
            current_base_fee * (one + percent_delta)
        } else {
            current_base_fee * (one - percent_delta)
        };
        if base_fee < *min_base_fee {
            base_fee = *min_base_fee;
        }
        let block_data = challenge.block_data.as_mut().unwrap();
        block_data.base_fee = Some(base_fee);
        block_data.per_nonce_fee = Some(min_per_nonce_fee.clone());
    }
}

fn find_smallest_range_dimension(points: &Frontier) -> usize {
    (0..2)
        .min_by_key(|&d| {
            let (min, max) = points
                .iter()
                .map(|p| p[d])
                .fold((i32::MAX, i32::MIN), |(min, max), val| {
                    (min.min(val), max.max(val))
                });
            max - min
        })
        .unwrap()
}

fn pareto_algorithm(points: Frontier, only_one: bool) -> Vec<Frontier> {
    if points.is_empty() {
        return Vec::new();
    }
    let dimension = find_smallest_range_dimension(&points);
    let sort_dimension = 1 - dimension;

    let mut buckets: HashMap<i32, Vec<Point>> = HashMap::new();
    for point in points {
        buckets.entry(point[dimension]).or_default().push(point);
    }
    for (_, group) in buckets.iter_mut() {
        // sort descending
        group.sort_unstable_by(|a, b| b[sort_dimension].cmp(&a[sort_dimension]));
    }
    let mut result = Vec::new();
    while !buckets.is_empty() {
        let points: HashSet<Point> = buckets.values().map(|group| group[0].clone()).collect();
        let frontier = points.pareto_frontier();
        for point in frontier.iter() {
            let bucket = buckets.get_mut(&point[dimension]).unwrap();
            bucket.remove(0);
            if bucket.is_empty() {
                buckets.remove(&point[dimension]);
            }
        }
        result.push(frontier);
        if only_one {
            break;
        }
    }
    result
}

#[time]
async fn update_qualifiers(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let mut solutions_by_challenge = HashMap::<String, Vec<(&BenchmarkSettings, &u32)>>::new();
    for (settings, num_solutions) in cache.active_solutions.values() {
        solutions_by_challenge
            .entry(settings.challenge_id.clone())
            .or_default()
            .push((settings, num_solutions));
    }

    let mut max_qualifiers_by_player = HashMap::<String, u32>::new();
    for challenge in cache.active_challenges.values_mut() {
        let block_data = challenge.block_data.as_mut().unwrap();
        block_data.num_qualifiers = Some(0);
        block_data.qualifier_difficulties = Some(HashSet::new());
    }
    for algorithm in cache.active_algorithms.values_mut() {
        let block_data = algorithm.block_data.as_mut().unwrap();
        block_data.num_qualifiers_by_player = Some(HashMap::new());
    }
    for player in cache.active_players.values_mut() {
        let block_data = player.block_data.as_mut().unwrap();
        max_qualifiers_by_player.insert(player.id.clone(), *block_data.cutoff());
        block_data.num_qualifiers_by_challenge = Some(HashMap::new());
    }

    for (challenge_id, challenge) in cache.active_challenges.iter_mut() {
        if !solutions_by_challenge.contains_key(challenge_id) {
            continue;
        }
        let solutions = solutions_by_challenge.get_mut(challenge_id).unwrap();
        let points = solutions
            .iter()
            .map(|(settings, _)| settings.difficulty.clone())
            .collect::<Frontier>();
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        for (frontier_index, frontier) in pareto_algorithm(points, false).into_iter().enumerate() {
            for point in frontier {
                frontier_indexes.insert(point, frontier_index);
            }
        }
        solutions.sort_by(|(a_settings, _), (b_settings, _)| {
            let a_index = frontier_indexes[&a_settings.difficulty];
            let b_index = frontier_indexes[&b_settings.difficulty];
            a_index.cmp(&b_index)
        });

        let mut max_qualifiers_by_player = max_qualifiers_by_player.clone();
        let mut curr_frontier_index = 0;
        let challenge_data = challenge.block_data.as_mut().unwrap();
        for (settings, &num_solutions) in solutions.iter() {
            let BenchmarkSettings {
                player_id,
                algorithm_id,
                challenge_id,
                difficulty,
                ..
            } = settings;

            if curr_frontier_index != frontier_indexes[difficulty]
                && *challenge_data.num_qualifiers() > config.qualifiers.total_qualifiers_threshold
            {
                break;
            }
            let difficulty_parameters = &config.difficulty.parameters[challenge_id];
            let min_difficulty = difficulty_parameters.min_difficulty();
            let max_difficulty = difficulty_parameters.max_difficulty();
            if (0..difficulty.len())
                .into_iter()
                .any(|i| difficulty[i] < min_difficulty[i] || difficulty[i] > max_difficulty[i])
            {
                continue;
            }
            curr_frontier_index = frontier_indexes[difficulty];
            let player_data = cache
                .active_players
                .get_mut(player_id)
                .unwrap()
                .block_data
                .as_mut()
                .unwrap();
            let algorithm_data = cache
                .active_algorithms
                .get_mut(algorithm_id)
                .unwrap()
                .block_data
                .as_mut()
                .unwrap();

            let max_qualifiers = max_qualifiers_by_player.get(player_id).unwrap().clone();
            let num_qualifiers = num_solutions.min(max_qualifiers);
            max_qualifiers_by_player.insert(player_id.clone(), max_qualifiers - num_qualifiers);

            if num_qualifiers > 0 {
                *player_data
                    .num_qualifiers_by_challenge
                    .as_mut()
                    .unwrap()
                    .entry(challenge_id.clone())
                    .or_default() += num_qualifiers;
                *algorithm_data
                    .num_qualifiers_by_player
                    .as_mut()
                    .unwrap()
                    .entry(player_id.clone())
                    .or_default() += num_qualifiers;
                *challenge_data.num_qualifiers.as_mut().unwrap() += num_qualifiers;
            }
            challenge_data
                .qualifier_difficulties
                .as_mut()
                .unwrap()
                .insert(difficulty.clone());
        }
    }
}

#[time]
async fn update_frontiers(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    for challenge in cache.active_challenges.values_mut() {
        let block_data = challenge.block_data.as_mut().unwrap();

        let difficulty_parameters = &config.difficulty.parameters[&challenge.id];
        let min_difficulty = difficulty_parameters.min_difficulty();
        let max_difficulty = difficulty_parameters.max_difficulty();

        let points = block_data
            .qualifier_difficulties()
            .iter()
            .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>();
        let (base_frontier, scaling_factor, scaled_frontier) = if points.len() == 0 {
            let base_frontier: Frontier = vec![min_difficulty.clone()].into_iter().collect();
            let scaling_factor = 0.0;
            let scaled_frontier = base_frontier.clone();
            (base_frontier, scaling_factor, scaled_frontier)
        } else {
            let base_frontier = pareto_algorithm(points, true)
                .pop()
                .unwrap()
                .into_iter()
                .map(|d| d.into_iter().map(|x| -x).collect())
                .collect::<Frontier>() // mirror the points back;
                .extend(&min_difficulty, &max_difficulty);
            let scaling_factor = (*block_data.num_qualifiers() as f64
                / config.qualifiers.total_qualifiers_threshold as f64)
                .min(config.difficulty.max_scaling_factor);
            let scaled_frontier = base_frontier
                .scale(&min_difficulty, &max_difficulty, scaling_factor)
                .extend(&min_difficulty, &max_difficulty);
            (base_frontier, scaling_factor, scaled_frontier)
        };

        block_data.base_frontier = Some(base_frontier);
        block_data.scaled_frontier = Some(scaled_frontier);
        block_data.scaling_factor = Some(scaling_factor);
    }
}

#[time]
async fn update_influence(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    let active_player_ids = &block.data().active_player_ids;
    if active_player_ids.len() == 0 {
        return;
    }

    let mut num_qualifiers_by_challenge = HashMap::<String, u32>::new();
    for challenge in cache.active_challenges.values() {
        num_qualifiers_by_challenge.insert(
            challenge.id.clone(),
            *challenge.block_data().num_qualifiers(),
        );
    }

    let total_deposit = cache
        .active_players
        .values()
        .map(|p| p.block_data().deposit.clone().unwrap())
        .sum::<PreciseNumber>();

    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    let imbalance_multiplier =
        PreciseNumber::from_f64(config.optimisable_proof_of_work.imbalance_multiplier);
    let num_challenges = PreciseNumber::from(cache.active_challenges.len());

    let mut weights = Vec::<PreciseNumber>::new();
    for player_id in active_player_ids.iter() {
        let data = cache
            .active_players
            .get_mut(player_id)
            .unwrap()
            .block_data
            .as_mut()
            .unwrap();

        let mut percent_qualifiers = Vec::<PreciseNumber>::new();
        for challenge_id in cache.active_challenges.keys() {
            let num_qualifiers = num_qualifiers_by_challenge[challenge_id];
            let num_qualifiers_by_player = *data
                .num_qualifiers_by_challenge()
                .get(challenge_id)
                .unwrap_or(&0);

            percent_qualifiers.push(if num_qualifiers_by_player == 0 {
                PreciseNumber::from(0)
            } else {
                PreciseNumber::from(num_qualifiers_by_player) / PreciseNumber::from(num_qualifiers)
            });
        }
        let OptimisableProofOfWorkConfig {
            avg_percent_qualifiers_multiplier,
            enable_proof_of_deposit,
            ..
        } = &config.optimisable_proof_of_work;
        if enable_proof_of_deposit.is_some_and(|x| x) {
            let max_percent_rolling_deposit =
                PreciseNumber::from_f64(avg_percent_qualifiers_multiplier.clone().unwrap())
                    * percent_qualifiers.arithmetic_mean();
            let percent_rolling_deposit = if total_deposit == zero {
                zero.clone()
            } else {
                data.deposit.clone().unwrap() / total_deposit
            };
            let qualifying_percent_rolling_deposit =
                if percent_rolling_deposit > max_percent_rolling_deposit {
                    max_percent_rolling_deposit.clone()
                } else {
                    percent_rolling_deposit
                };
            percent_qualifiers.push(qualifying_percent_rolling_deposit.clone());
            data.qualifying_percent_rolling_deposit = Some(qualifying_percent_rolling_deposit);
        }

        let mean = percent_qualifiers.arithmetic_mean();
        let variance = percent_qualifiers.variance();
        let cv_sqr = if mean == zero {
            zero.clone()
        } else {
            variance / (mean * mean)
        };

        let imbalance = cv_sqr / (num_challenges - one);
        let imbalance_penalty =
            one - PreciseNumber::approx_inv_exp(imbalance_multiplier * imbalance);

        weights.push(mean * (one - imbalance_penalty));

        data.imbalance = Some(imbalance);
        data.imbalance_penalty = Some(imbalance_penalty);
    }

    let influences = weights.normalise();
    for (player_id, &influence) in active_player_ids.iter().zip(influences.iter()) {
        let data = cache
            .active_players
            .get_mut(player_id)
            .unwrap()
            .block_data
            .as_mut()
            .unwrap();
        data.influence = Some(influence);
    }
}

#[time]
async fn update_adoption(cache: &mut AddBlockCache) {
    let mut algorithms_by_challenge = HashMap::<String, Vec<&mut Algorithm>>::new();
    for algorithm in cache.active_algorithms.values_mut() {
        algorithms_by_challenge
            .entry(algorithm.details.challenge_id.clone())
            .or_default()
            .push(algorithm);
    }

    for challenge_id in cache.active_challenges.keys() {
        let algorithms = algorithms_by_challenge.get_mut(challenge_id);
        if algorithms.is_none() {
            continue;
        }
        let algorithms = algorithms.unwrap();

        let mut weights = Vec::<PreciseNumber>::new();
        for algorithm in algorithms.iter() {
            let mut weight = PreciseNumber::from(0);
            for (player_id, &num_qualifiers) in
                algorithm.block_data().num_qualifiers_by_player().iter()
            {
                let num_qualifiers = PreciseNumber::from(num_qualifiers);
                let player_data = cache.active_players.get(player_id).unwrap().block_data();
                let influence = player_data.influence.unwrap();
                let player_num_qualifiers = PreciseNumber::from(
                    *player_data
                        .num_qualifiers_by_challenge
                        .as_ref()
                        .unwrap()
                        .get(challenge_id)
                        .unwrap(),
                );

                weight = weight + influence * num_qualifiers / player_num_qualifiers;
            }
            weights.push(weight);
        }

        let adoption = weights.normalise();
        for (algorithm, adoption) in algorithms.iter_mut().zip(adoption) {
            algorithm.block_data.as_mut().unwrap().adoption = Some(adoption);
        }
    }
}

#[time]
async fn update_innovator_rewards(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    let zero = PreciseNumber::from(0);
    let mut eligible_algorithms_by_challenge = HashMap::<String, Vec<&mut Algorithm>>::new();
    for algorithm in cache.active_algorithms.values_mut() {
        let is_merged = algorithm.state().round_merged.is_some();
        let data = algorithm.block_data.as_mut().unwrap();
        data.reward = Some(zero.clone());

        if *data.adoption() >= adoption_threshold || (is_merged && *data.adoption() > zero) {
            eligible_algorithms_by_challenge
                .entry(algorithm.details.challenge_id.clone())
                .or_default()
                .push(algorithm);
        }
    }
    if eligible_algorithms_by_challenge.len() == 0 {
        return;
    }

    let reward_pool_per_challenge = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.optimisations)
        / PreciseNumber::from(eligible_algorithms_by_challenge.len());

    let zero = PreciseNumber::from(0);
    for algorithms in eligible_algorithms_by_challenge.values_mut() {
        let mut total_adoption = zero.clone();
        for algorithm in algorithms.iter() {
            total_adoption = total_adoption + algorithm.block_data().adoption();
        }

        for algorithm in algorithms.iter_mut() {
            let data = algorithm.block_data.as_mut().unwrap();
            let adoption = *data.adoption();
            data.reward = Some(reward_pool_per_challenge * adoption / total_adoption);
        }
    }
}

#[time]
async fn update_benchmarker_rewards(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let reward_pool = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

    for player in cache.active_players.values_mut() {
        let data = player.block_data.as_mut().unwrap();
        let influence = *data.influence();
        data.reward = Some(influence * reward_pool);
    }
}

#[time]
async fn update_merge_points(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    for algorithm in cache.active_algorithms.values_mut() {
        let is_merged = algorithm.state().round_merged.is_some();
        let data = algorithm.block_data.as_mut().unwrap();

        // first block of the round
        let prev_merge_points = if block.details.height % config.rounds.blocks_per_round == 0 {
            0
        } else {
            match &cache.prev_algorithms.get(&algorithm.id).unwrap().block_data {
                Some(data) => *data.merge_points(),
                None => 0,
            }
        };
        data.merge_points = Some(if is_merged || *data.adoption() < adoption_threshold {
            prev_merge_points
        } else {
            prev_merge_points + 1
        });
    }
}

#[time]
async fn update_merges(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    // last block of the round
    if (block.details.height + 1) % config.rounds.blocks_per_round != 0 {
        return;
    }

    let mut algorithm_to_merge_by_challenge = HashMap::<String, &mut Algorithm>::new();
    for algorithm in cache.active_algorithms.values_mut() {
        let challenge_id = algorithm.details.challenge_id.clone();
        let data = algorithm.block_data();

        if algorithm.state().round_merged.is_some()
            || *data.merge_points() < config.algorithm_submissions.merge_points_threshold
        {
            continue;
        }
        if !algorithm_to_merge_by_challenge.contains_key(&challenge_id)
            || algorithm_to_merge_by_challenge[&challenge_id]
                .block_data()
                .merge_points
                < data.merge_points
        {
            algorithm_to_merge_by_challenge.insert(challenge_id, algorithm);
        }
    }

    let round_merged = block.details.round + 1;
    for algorithm in algorithm_to_merge_by_challenge.values_mut() {
        let state = algorithm.state.as_mut().unwrap();
        state.round_merged = Some(round_merged);
    }
}

#[time]
async fn commit_changes<T: Context>(ctx: &T, block: &Block, cache: &mut AddBlockCache) {
    for precommit in cache.mempool_precommits.drain(..) {
        ctx.update_precommit_state(&precommit.benchmark_id, precommit.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_precommit_state error: {:?}", e));
    }
    for algorithm in cache.mempool_algorithms.drain(..) {
        ctx.update_algorithm_state(&algorithm.id, algorithm.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
    }
    for challenge in cache.mempool_challenges.drain(..) {
        ctx.update_challenge_state(&challenge.id, challenge.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_challenge_state error: {:?}", e));
    }
    for benchmark in cache.mempool_benchmarks.drain(..) {
        ctx.update_benchmark_state(&benchmark.id, benchmark.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_benchmark_state error: {:?}", e));
    }
    for fraud in cache.mempool_frauds.drain(..) {
        ctx.update_fraud_state(&fraud.benchmark_id, fraud.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_fraud_state error: {:?}", e));
    }
    for proof in cache.mempool_proofs.drain(..) {
        ctx.update_proof_state(&proof.benchmark_id, proof.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_proof_state error: {:?}", e));
    }
    for wasm in cache.mempool_wasms.drain(..) {
        ctx.update_wasm_state(&wasm.algorithm_id, wasm.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_wasm_state error: {:?}", e));
    }
    for topup in cache.mempool_topups.drain(..) {
        ctx.update_topup_state(&topup.id, topup.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_topup_state error: {:?}", e));
    }
    for (player_id, player) in cache.active_fee_players.drain() {
        ctx.update_player_state(&player_id, player.state.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_player_state error: {:?}", e));
    }
    for (_, algorithm) in cache.active_algorithms.drain() {
        let state = algorithm.state.unwrap();
        if state.round_pushed.is_some_and(|r| r == block.details.round) {
            ctx.update_algorithm_state(&algorithm.id, state)
                .await
                .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
        }
        ctx.update_algorithm_block_data(&algorithm.id, &block.id, algorithm.block_data.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
    }
    for (_, challenge) in cache.active_challenges.drain() {
        ctx.update_challenge_block_data(&challenge.id, &block.id, challenge.block_data.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_challenge_block_data error: {:?}", e));
    }
    for (_, player) in cache.active_players.drain() {
        ctx.update_player_block_data(&player.id, &block.id, player.block_data.unwrap())
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }
}

fn get_block_reward(block: &Block) -> f64 {
    let config = block.config();

    config
        .rewards
        .schedule
        .iter()
        .filter(|s| s.round_start <= block.details.round)
        .last()
        .unwrap_or_else(|| {
            panic!(
                "get_block_reward error: Expecting a reward schedule for round {}",
                block.details.round
            )
        })
        .block_reward
}
