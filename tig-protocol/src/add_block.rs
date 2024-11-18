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
        if details.round >= *state.round_pushed.as_ref().unwrap_or(&round_pushed)
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
        if benchmark.details.num_solutions == 0 {
            continue;
        }
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
        .get_block(BlockFilter::LastConfirmed, false)
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
    details.fees_paid = Some(
        cache
            .mempool_precommits
            .iter()
            .map(|p| p.details.fee_paid().clone())
            .sum(),
    );
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
