use crate::context::*;
use logging_timer::time;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::{
    collections::{HashMap, HashSet},
    ops::Mul,
};
use tig_structs::{config::*, core::*};
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(ctx: &mut T) -> String {
    let block = create_block(ctx).await;
    confirm_mempool_algorithms(ctx, &block).await;
    confirm_mempool_benchmarks(ctx, &block).await;
    confirm_mempool_proofs(ctx, &block).await;
    confirm_mempool_frauds(ctx, &block).await;
    confirm_mempool_wasms(ctx, &block).await;
    update_deposits(ctx, &block).await;
    update_cutoffs(ctx, &block).await;
    update_solution_signature_thresholds(ctx, &block).await;
    update_qualifiers(ctx, &block).await;
    update_frontiers(ctx, &block).await;
    update_influence(ctx, &block).await;
    update_adoption(ctx, &block).await;
    update_innovator_rewards(ctx, &block).await;
    update_benchmarker_rewards(ctx, &block).await;
    update_merge_points(ctx, &block).await;
    update_merges(ctx, &block).await;
    block.id
}

#[time]
async fn create_block<T: Context>(ctx: &mut T) -> Block {
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
    let details = BlockDetails {
        prev_block_id: latest_block.id.clone(),
        height,
        round: height / config.rounds.blocks_per_round + 1,
        eth_block_num: Some(ctx.get_latest_eth_block_num().await.unwrap()),
    };
    let from_block_started = details
        .height
        .saturating_sub(config.benchmark_submissions.lifespan_period);
    let mut data = BlockData {
        mempool_algorithm_ids: HashSet::<String>::new(),
        mempool_benchmark_ids: HashSet::<String>::new(),
        mempool_fraud_ids: HashSet::<String>::new(),
        mempool_proof_ids: HashSet::<String>::new(),
        mempool_wasm_ids: HashSet::<String>::new(),
        active_challenge_ids: HashSet::<String>::new(),
        active_algorithm_ids: HashSet::<String>::new(),
        active_benchmark_ids: HashSet::<String>::new(),
        active_player_ids: HashSet::<String>::new(),
    };
    for algorithm in ctx
        .get_algorithms(AlgorithmsFilter::Mempool, None, false)
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
        .iter()
    {
        data.mempool_algorithm_ids.insert(algorithm.id.clone());
    }
    for benchmark in ctx
        .get_benchmarks(BenchmarksFilter::Mempool { from_block_started }, true)
        .await
        .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e))
        .iter()
    {
        data.mempool_benchmark_ids.insert(benchmark.id.clone());
    }
    for proof in ctx
        .get_proofs(ProofsFilter::Mempool { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
        .iter()
    {
        data.mempool_proof_ids.insert(proof.benchmark_id.clone());
    }
    for fraud in ctx
        .get_frauds(FraudsFilter::Mempool { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("get_frauds error: {:?}", e))
        .iter()
    {
        data.mempool_fraud_ids.insert(fraud.benchmark_id.clone());
    }
    for wasm in ctx
        .get_wasms(WasmsFilter::Mempool, false)
        .await
        .unwrap_or_else(|e| panic!("get_wasms error: {:?}", e))
        .iter()
    {
        data.mempool_wasm_ids.insert(wasm.algorithm_id.clone());
    }

    data.active_challenge_ids
        .extend(config.difficulty.parameters.keys().cloned());
    let wasms: HashMap<String, Wasm> = ctx
        .get_wasms(WasmsFilter::Confirmed, false)
        .await
        .unwrap_or_else(|e| panic!("get_wasms error: {:?}", e))
        .into_iter()
        .map(|x| (x.algorithm_id.clone(), x))
        .collect();
    for algorithm in ctx
        .get_algorithms(AlgorithmsFilter::Confirmed, None, false)
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
    {
        let mut state = algorithm.state.unwrap();
        let round_pushed = state
            .round_pushed
            .unwrap_or(state.round_submitted() + config.algorithm_submissions.push_delay);
        if !state.banned
            && details.round >= round_pushed
            && wasms
                .get(&algorithm.id)
                .is_some_and(|w| w.details.compile_success)
        {
            data.active_algorithm_ids.insert(algorithm.id.clone());
            if state.round_pushed.is_none() {
                state.round_pushed = Some(round_pushed);
                ctx.update_algorithm_state(&algorithm.id, &state)
                    .await
                    .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
            }
        }
    }
    let confirmed_proofs = ctx
        .get_proofs(ProofsFilter::Confirmed { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
        .into_iter()
        .map(|x| (x.benchmark_id.clone(), x))
        .collect::<HashMap<String, Proof>>();
    let confirmed_frauds = ctx
        .get_frauds(FraudsFilter::Confirmed { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("get_frauds error: {:?}", e))
        .into_iter()
        .map(|x| (x.benchmark_id.clone(), x))
        .collect::<HashMap<String, Fraud>>();
    for benchmark in ctx
        .get_benchmarks(BenchmarksFilter::Confirmed { from_block_started }, false)
        .await
        .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e))
    {
        let proof = confirmed_proofs.get(&benchmark.id);
        if proof.is_none() || confirmed_frauds.contains_key(&benchmark.id) {
            continue;
        }
        // TODO check player state
        let _player = ctx
            .get_players(
                PlayersFilter::Id(benchmark.settings.player_id.clone()),
                None,
            )
            .await
            .unwrap_or_else(|e| panic!("get_players error: {:?}", e))
            .pop();
        let proof_state = proof.unwrap().state();
        let submission_delay = proof_state.submission_delay();
        let block_confirmed = proof_state.block_confirmed();
        let block_active = block_confirmed
            + submission_delay * config.benchmark_submissions.submission_delay_multiplier;
        if details.height >= block_active {
            data.active_benchmark_ids.insert(benchmark.id.clone());
            data.active_player_ids
                .insert(benchmark.settings.player_id.clone());
        }
    }

    let block_id = ctx
        .add_block(&details, &data, &config)
        .await
        .unwrap_or_else(|e| panic!("add_block error: {:?}", e));
    for algorithm_id in data.mempool_algorithm_ids.iter() {
        let state = AlgorithmState {
            block_confirmed: None,
            round_submitted: None,
            round_pushed: None,
            round_merged: None,
            banned: false,
        };
        ctx.update_algorithm_state(algorithm_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
    }
    for benchmark_id in data.mempool_benchmark_ids.iter() {
        let state = BenchmarkState {
            block_confirmed: None,
            sampled_nonces: None,
        };
        ctx.update_benchmark_state(benchmark_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_benchmark_state error: {:?}", e));
    }
    for proof_id in data.mempool_proof_ids.iter() {
        let state = ProofState {
            block_confirmed: None,
            submission_delay: None,
        };
        ctx.update_proof_state(proof_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_proof_state error: {:?}", e));
    }
    for fraud_id in data.mempool_fraud_ids.iter() {
        let state = FraudState {
            block_confirmed: None,
        };
        ctx.update_fraud_state(fraud_id, &state)
            .await
            .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e));
    }

    for challenge_id in data.active_challenge_ids.iter() {
        let data = ChallengeBlockData {
            num_qualifiers: None,
            solution_signature_threshold: None,
            scaled_frontier: None,
            base_frontier: None,
            scaling_factor: None,
            qualifier_difficulties: None,
        };
        ctx.update_challenge_block_data(challenge_id, &block_id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_challenge_block_data error: {:?}", e));
    }
    for algorithm_id in data.active_algorithm_ids.iter() {
        let data = AlgorithmBlockData {
            reward: None,
            adoption: None,
            merge_points: None,
            num_qualifiers_by_player: None,
            round_earnings: None,
        };
        ctx.update_algorithm_block_data(algorithm_id, &block_id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
    }
    for player_id in data.active_player_ids.iter() {
        let data = PlayerBlockData {
            reward: None,
            influence: None,
            cutoff: None,
            imbalance: None,
            imbalance_penalty: None,
            num_qualifiers_by_challenge: None,
            round_earnings: None,
            deposit: None,
            rolling_deposit: None,
        };
        ctx.update_player_block_data(player_id, &block_id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }

    Block {
        id: block_id,
        config: Some(config.clone()),
        details,
        data: Some(data),
    }
}

#[time]
async fn confirm_mempool_algorithms<T: Context>(ctx: &mut T, block: &Block) {
    for algorithm_id in block.data().mempool_algorithm_ids.iter() {
        let algorithm = get_algorithm_by_id(ctx, algorithm_id, None)
            .await
            .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e));
        let mut state = algorithm.state().clone();
        state.block_confirmed = Some(block.details.height);
        state.round_submitted = Some(block.details.round);
        ctx.update_algorithm_state(algorithm_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
    }
}

#[time]
async fn confirm_mempool_benchmarks<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    for benchmark_id in block.data().mempool_benchmark_ids.iter() {
        let benchmark = get_benchmark_by_id(ctx, benchmark_id, true)
            .await
            .unwrap_or_else(|e| panic!("get_benchmark_by_id error: {:?}", e));

        let seed = u32_from_str(format!("{:?}|{:?}", block.id, benchmark_id).as_str());
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let solutions_meta_data = benchmark.solutions_meta_data();
        let mut indexes: Vec<usize> = (0..solutions_meta_data.len()).collect();
        indexes.shuffle(&mut rng);

        let mut state = benchmark.state().clone();
        state.sampled_nonces = Some(
            indexes
                .into_iter()
                .take(config.benchmark_submissions.max_samples)
                .map(|i| solutions_meta_data[i].nonce)
                .collect(),
        );
        state.block_confirmed = Some(block.details.height);

        ctx.update_benchmark_state(&benchmark_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_benchmark_state error: {:?}", e));
    }
}

#[time]
async fn confirm_mempool_proofs<T: Context>(ctx: &mut T, block: &Block) {
    for benchmark_id in block.data().mempool_proof_ids.iter() {
        let benchmark = get_benchmark_by_id(ctx, &benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_benchmark_by_id error: {:?}", e));
        let proof = get_proof_by_benchmark_id(ctx, &benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_proof_by_benchmark_id error: {:?}", e));
        let mut state = proof.state().clone();
        state.block_confirmed = Some(block.details.height);
        state.submission_delay = Some(block.details.height - benchmark.details.block_started);
        ctx.update_proof_state(benchmark_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_proof_state error: {:?}", e));
    }
}

#[time]
async fn confirm_mempool_frauds<T: Context>(ctx: &mut T, block: &Block) {
    // Future Todo: slash player's rewards from past day
    for benchmark_id in block.data().mempool_fraud_ids.iter() {
        let fraud = get_fraud_by_id(ctx, &benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_fraud_by_id error: {:?}", e));
        let mut state = fraud.state().clone();
        state.block_confirmed = Some(block.details.height);
        ctx.update_fraud_state(benchmark_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_fraud_state error: {:?}", e));
    }
}

#[time]
async fn confirm_mempool_wasms<T: Context>(ctx: &mut T, block: &Block) {
    for algorithm_id in block.data().mempool_wasm_ids.iter() {
        let wasm = get_wasm_by_id(ctx, &algorithm_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e));
        let mut state = wasm.state().clone();
        state.block_confirmed = Some(block.details.height);
        ctx.update_wasm_state(algorithm_id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_wasm_state error: {:?}", e));
    }
}

#[time]
async fn update_deposits<T: Context>(ctx: &mut T, block: &Block) {
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
    for player_id in block.data().active_player_ids.iter() {
        let rolling_deposit =
            match get_player_by_id(ctx, player_id, Some(&block.details.prev_block_id))
                .await
                .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e))
                .block_data
            {
                Some(data) => data.rolling_deposit,
                None => None,
            }
            .unwrap_or_else(|| zero.clone());

        let player = get_player_by_id(ctx, player_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e));
        let mut data = player.block_data().clone();

        let deposit = ctx
            .get_player_deposit(eth_block_num, player_id)
            .await
            .unwrap()
            .unwrap_or_else(|| zero.clone());
        data.rolling_deposit = Some(decay * rolling_deposit + (one - decay) * deposit);
        data.deposit = Some(deposit);

        ctx.update_player_block_data(player_id, &block.id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }
}

#[time]
async fn update_cutoffs<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();
    let num_challenges = block.data().active_challenge_ids.len() as f64;

    let mut total_solutions_by_player = HashMap::<String, f64>::new();
    for benchmark_id in block.data().active_benchmark_ids.iter() {
        let benchmark = get_benchmark_by_id(ctx, benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_benchmark_by_id error: {:?}", e));
        *total_solutions_by_player
            .entry(benchmark.settings.player_id.clone())
            .or_default() += benchmark.details.num_solutions as f64;
    }

    for (player_id, total_solutions) in total_solutions_by_player.iter() {
        let player = &get_player_by_id(ctx, player_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e));
        let mut data = player.block_data().clone();

        data.cutoff =
            Some((total_solutions / num_challenges * config.qualifiers.cutoff_multiplier) as u32);

        ctx.update_player_block_data(player_id, &block.id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }
}

#[time]
async fn update_solution_signature_thresholds<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    let mut num_new_solutions_by_challenge = HashMap::<String, u32>::new();
    for benchmark_id in block.data().mempool_proof_ids.iter() {
        let benchmark = get_benchmark_by_id(ctx, benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_benchmark_by_id error: {:?}", e));
        *num_new_solutions_by_challenge
            .entry(benchmark.settings.challenge_id.clone())
            .or_default() += benchmark.details.num_solutions;
    }

    for challenge_id in block.data().active_challenge_ids.iter() {
        let num_new_solutions = *num_new_solutions_by_challenge
            .get(challenge_id)
            .unwrap_or(&0) as f64;
        let equilibrium_rate = config.qualifiers.total_qualifiers_threshold as f64
            / config.benchmark_submissions.lifespan_period as f64;
        let percentage_error = 1f64
            - num_new_solutions
                / (config.solution_signature.equilibrium_rate_multiplier * equilibrium_rate);
        let max_threshold = u32::MAX as f64;
        let percent_delta = (percentage_error * config.solution_signature.percent_error_multiplier)
            .abs()
            .clamp(0f64, config.solution_signature.max_percent_delta)
            .mul(if percentage_error < 0f64 { -1f64 } else { 1f64 });

        let prev_solution_signature_threshold =
            match get_challenge_by_id(ctx, challenge_id, Some(&block.details.prev_block_id))
                .await
                .unwrap_or_else(|e| panic!("get_challenge_by_id error: {:?}", e))
                .block_data
            {
                Some(data) => *data.solution_signature_threshold() as f64,
                None => max_threshold,
            };
        let mut block_data = get_challenge_by_id(ctx, challenge_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_challenge_by_id error: {:?}", e))
            .block_data()
            .clone();
        block_data.solution_signature_threshold = Some(
            (prev_solution_signature_threshold + percent_delta * max_threshold)
                .clamp(0f64, max_threshold) as u32,
        );

        ctx.update_challenge_block_data(challenge_id, &block.id, &block_data)
            .await
            .unwrap_or_else(|e| panic!("update_challenge_block_data error: {:?}", e));
    }
}

#[time]
async fn update_qualifiers<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();
    let BlockData {
        active_benchmark_ids,
        active_algorithm_ids,
        active_challenge_ids,
        active_player_ids,
        ..
    } = block.data();

    let mut benchmarks_by_challenge = HashMap::<String, Vec<Benchmark>>::new();
    for benchmark_id in active_benchmark_ids.iter() {
        let benchmark = get_benchmark_by_id(ctx, benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_benchmark_by_id error: {:?}", e));
        benchmarks_by_challenge
            .entry(benchmark.settings.challenge_id.clone())
            .or_default()
            .push(benchmark);
    }

    let mut data_by_challenge = HashMap::<String, ChallengeBlockData>::new();
    let mut data_by_player = HashMap::<String, PlayerBlockData>::new();
    let mut data_by_algorithm = HashMap::<String, AlgorithmBlockData>::new();
    let mut max_qualifiers_by_player = HashMap::<String, u32>::new();
    for challenge_id in active_challenge_ids.iter() {
        let mut block_data = get_challenge_by_id(ctx, challenge_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_challenge_by_id error: {:?}", e))
            .block_data()
            .clone();
        block_data.num_qualifiers = Some(0);
        block_data.qualifier_difficulties = Some(HashSet::new());
        data_by_challenge.insert(challenge_id.clone(), block_data);
    }
    for algorithm_id in active_algorithm_ids.iter() {
        let mut block_data = get_algorithm_by_id(ctx, algorithm_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e))
            .block_data()
            .clone();
        block_data.num_qualifiers_by_player = Some(HashMap::new());
        data_by_algorithm.insert(algorithm_id.clone(), block_data);
    }
    for player_id in active_player_ids.iter() {
        let mut block_data = get_player_by_id(ctx, player_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e))
            .block_data()
            .clone();
        max_qualifiers_by_player.insert(player_id.clone(), block_data.cutoff().clone());
        block_data.num_qualifiers_by_challenge = Some(HashMap::new());
        data_by_player.insert(player_id.clone(), block_data);
    }

    for challenge_id in active_challenge_ids.iter() {
        if !benchmarks_by_challenge.contains_key(challenge_id) {
            continue;
        }
        let benchmarks = benchmarks_by_challenge.get_mut(challenge_id).unwrap();
        let mut points = benchmarks
            .iter()
            .map(|b| b.settings.difficulty.clone())
            .collect::<Frontier>();
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        let mut frontier_index = 0;
        while !points.is_empty() {
            let frontier = points.pareto_frontier();
            points = points.difference(&frontier).cloned().collect();
            frontier.iter().for_each(|p| {
                frontier_indexes.insert(p.clone(), frontier_index);
            });
            frontier_index += 1;
        }
        benchmarks.sort_by(|a, b| {
            let a_index = frontier_indexes[&a.settings.difficulty];
            let b_index = frontier_indexes[&b.settings.difficulty];
            a_index.cmp(&b_index)
        });

        let mut max_qualifiers_by_player = max_qualifiers_by_player.clone();
        let mut curr_frontier_index = 0;
        let challenge_data = data_by_challenge.get_mut(challenge_id).unwrap();
        for benchmark in benchmarks.iter() {
            let BenchmarkSettings {
                player_id,
                algorithm_id,
                challenge_id,
                difficulty,
                ..
            } = &benchmark.settings;

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
            let player_data = data_by_player.get_mut(player_id).unwrap();
            let algorithm_data = data_by_algorithm.get_mut(algorithm_id).unwrap();

            let max_qualifiers = max_qualifiers_by_player.get(player_id).unwrap().clone();
            let num_qualifiers = benchmark.details.num_solutions.min(max_qualifiers);
            max_qualifiers_by_player.insert(player_id.clone(), max_qualifiers - num_qualifiers);

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
            challenge_data
                .qualifier_difficulties
                .as_mut()
                .unwrap()
                .insert(difficulty.clone());
        }
    }

    for (id, data) in data_by_challenge.iter() {
        ctx.update_challenge_block_data(id, &block.id, data)
            .await
            .unwrap_or_else(|e| panic!("update_challenge_block_data error: {:?}", e));
    }
    for (id, data) in data_by_algorithm.iter() {
        ctx.update_algorithm_block_data(id, &block.id, data)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
    }
    for (id, data) in data_by_player.iter() {
        ctx.update_player_block_data(id, &block.id, data)
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }
}

#[time]
async fn update_frontiers<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    for challenge_id in block.data().active_challenge_ids.iter() {
        let challenge = get_challenge_by_id(ctx, challenge_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_challenge_by_id error: {:?}", e));
        let mut block_data = challenge.block_data().clone();

        let difficulty_parameters = &config.difficulty.parameters[&challenge.id];
        let min_difficulty = difficulty_parameters.min_difficulty();
        let max_difficulty = difficulty_parameters.max_difficulty();

        let base_frontier = block_data
            .qualifier_difficulties()
            .iter()
            .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>()
            .pareto_frontier()
            .iter()
            .map(|d| d.iter().map(|x| -x).collect())
            .collect::<Frontier>() // mirror the points back;
            .extend(&min_difficulty, &max_difficulty);

        let mut scaling_factor = (*block_data.num_qualifiers() as f64
            / config.qualifiers.total_qualifiers_threshold as f64)
            .clamp(0.0, config.difficulty.max_scaling_factor);
        if let Some(scaling_factor_decay) = config.difficulty.scaling_factor_decay {
            let prev_scaling_factor =
                get_challenge_by_id(ctx, challenge_id, Some(&block.details.prev_block_id))
                    .await
                    .unwrap_or_else(|e| panic!("get_challenge_by_id error: {:?}", e))
                    .block_data()
                    .scaling_factor
                    .unwrap_or(1.0)
                    .clamp(0.0, config.difficulty.max_scaling_factor);
            scaling_factor = scaling_factor_decay * prev_scaling_factor
                + (1.0 - scaling_factor_decay) * scaling_factor;
        }
        let scaled_frontier = base_frontier
            .scale(&min_difficulty, &max_difficulty, scaling_factor)
            .extend(&min_difficulty, &max_difficulty);

        block_data.base_frontier = Some(base_frontier);
        block_data.scaled_frontier = Some(scaled_frontier);
        block_data.scaling_factor = Some(scaling_factor);

        ctx.update_challenge_block_data(challenge_id, &block.id, &block_data)
            .await
            .unwrap_or_else(|e| panic!("update_challenge_block_data error: {:?}", e));
    }
}

#[time]
async fn update_influence<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();
    let BlockData {
        active_challenge_ids,
        active_player_ids,
        ..
    } = block.data();

    if active_player_ids.len() == 0 {
        return;
    }

    let mut num_qualifiers_by_challenge = HashMap::<String, u32>::new();
    for challenge_id in active_challenge_ids.iter() {
        num_qualifiers_by_challenge.insert(
            challenge_id.clone(),
            *get_challenge_by_id(ctx, challenge_id, Some(&block.id))
                .await
                .unwrap_or_else(|e| panic!("get_challenge_by_id error: {:?}", e))
                .block_data()
                .num_qualifiers(),
        );
    }

    let mut player_data = HashMap::<String, PlayerBlockData>::new();
    for player_id in active_player_ids.iter() {
        player_data.insert(
            player_id.clone(),
            get_player_by_id(ctx, player_id, Some(&block.id))
                .await
                .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e))
                .block_data()
                .clone(),
        );
    }
    let total_deposit = player_data
        .values()
        .map(|data| data.deposit.clone().unwrap())
        .sum::<PreciseNumber>();

    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    let imbalance_multiplier =
        PreciseNumber::from_f64(config.optimisable_proof_of_work.imbalance_multiplier);
    let num_challenges = PreciseNumber::from(active_challenge_ids.len());

    let mut weights = Vec::<PreciseNumber>::new();
    for player_id in active_player_ids.iter() {
        let data = player_data.get_mut(player_id).unwrap();

        let mut percent_qualifiers = Vec::<PreciseNumber>::new();
        for challenge_id in active_challenge_ids.iter() {
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
            rolling_deposit_decay,
            enable_proof_of_deposit,
            ..
        } = &config.optimisable_proof_of_work;
        if rolling_deposit_decay.is_some() && enable_proof_of_deposit.is_some_and(|x| x) {
            percent_qualifiers.push(if total_deposit == zero {
                zero.clone()
            } else {
                data.deposit.clone().unwrap() / total_deposit
            });
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
        let data = player_data.get_mut(player_id).unwrap();

        data.influence = Some(influence);

        ctx.update_player_block_data(player_id, &block.id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }
}

#[time]
async fn update_adoption<T: Context>(ctx: &mut T, block: &Block) {
    let BlockData {
        active_algorithm_ids,
        active_challenge_ids,
        ..
    } = block.data();

    let mut algorithms_by_challenge = HashMap::<String, Vec<&String>>::new();
    for algorithm_id in active_algorithm_ids.iter() {
        algorithms_by_challenge
            .entry(
                get_algorithm_by_id(ctx, algorithm_id, Some(&block.id))
                    .await
                    .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e))
                    .details
                    .challenge_id
                    .clone(),
            )
            .or_default()
            .push(algorithm_id);
    }

    for challenge_id in active_challenge_ids.iter() {
        if !algorithms_by_challenge.contains_key(challenge_id) {
            continue;
        }

        let mut algorithm_data = HashMap::<&String, AlgorithmBlockData>::new();
        for algorithm_id in algorithms_by_challenge[challenge_id].iter() {
            algorithm_data.insert(
                algorithm_id,
                get_algorithm_by_id(ctx, algorithm_id, Some(&block.id))
                    .await
                    .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e))
                    .block_data()
                    .clone(),
            );
        }

        let mut weights = Vec::<PreciseNumber>::new();
        for (_, data) in algorithm_data.iter() {
            let mut weight = PreciseNumber::from(0);
            for (player_id, &num_qualifiers) in data.num_qualifiers_by_player().iter() {
                let num_qualifiers = PreciseNumber::from(num_qualifiers);
                let influence = *get_player_by_id(ctx, player_id, Some(&block.id))
                    .await
                    .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e))
                    .block_data()
                    .influence();

                weight = weight + influence * num_qualifiers;
            }
            weights.push(weight);
        }

        let adoption = weights.normalise();
        for ((algorithm_id, data), adoption) in algorithm_data.iter_mut().zip(adoption) {
            data.adoption = Some(adoption);
            ctx.update_algorithm_block_data(algorithm_id, &block.id, data)
                .await
                .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
        }
    }
}

#[time]
async fn update_innovator_rewards<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    let zero = PreciseNumber::from(0);
    let mut eligible_algorithms_by_challenge = HashMap::<String, Vec<Algorithm>>::new();
    for algorithm_id in block.data().active_algorithm_ids.iter() {
        let algorithm = get_algorithm_by_id(ctx, algorithm_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e));
        let mut data = algorithm.block_data().clone();

        if *data.adoption() >= adoption_threshold
            || (algorithm.state().round_merged.is_some() && *data.adoption() > zero)
        {
            eligible_algorithms_by_challenge
                .entry(algorithm.details.challenge_id.clone())
                .or_default()
                .push(algorithm);
        }

        data.reward = Some(zero.clone());
        ctx.update_algorithm_block_data(algorithm_id, &block.id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
    }
    if eligible_algorithms_by_challenge.len() == 0 {
        return;
    }

    let reward_pool_per_challenge = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.optimisations)
        / PreciseNumber::from(eligible_algorithms_by_challenge.len());

    let zero = PreciseNumber::from(0);
    for (_, algorithms) in eligible_algorithms_by_challenge.iter() {
        let mut total_adoption = zero.clone();
        for algorithm in algorithms.iter() {
            total_adoption = total_adoption + algorithm.block_data().adoption();
        }

        for algorithm in algorithms.iter() {
            let mut data = algorithm.block_data().clone();
            let adoption = *data.adoption();

            data.reward = Some(reward_pool_per_challenge * adoption / total_adoption);

            ctx.update_algorithm_block_data(&algorithm.id, &block.id, &data)
                .await
                .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
        }
    }
}

#[time]
async fn update_benchmarker_rewards<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    let reward_pool = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

    for player_id in block.data().active_player_ids.iter() {
        let mut data = get_player_by_id(ctx, player_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_player_by_id error: {:?}", e))
            .block_data()
            .clone();
        let influence = *data.influence();

        data.reward = Some(influence * reward_pool);

        ctx.update_player_block_data(player_id, &block.id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_player_block_data error: {:?}", e));
    }
}

#[time]
async fn update_merge_points<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    for algorithm_id in block.data().active_algorithm_ids.iter() {
        let algorithm = get_algorithm_by_id(ctx, algorithm_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e));
        let mut data = algorithm.block_data().clone();

        // first block of the round
        let prev_merge_points = if block.details.height % config.rounds.blocks_per_round == 0 {
            0
        } else {
            match get_algorithm_by_id(ctx, algorithm_id, Some(&block.details.prev_block_id))
                .await
                .unwrap_or_else(|e| panic!("update_merge_points error: {:?}", e))
                .block_data
            {
                Some(data) => *data.merge_points(),
                None => 0,
            }
        };
        data.merge_points = Some(
            if algorithm.state().round_merged.is_some() || *data.adoption() < adoption_threshold {
                prev_merge_points
            } else {
                prev_merge_points + 1
            },
        );
        ctx.update_algorithm_block_data(algorithm_id, &block.id, &data)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_block_data error: {:?}", e));
    }
}

#[time]
async fn update_merges<T: Context>(ctx: &mut T, block: &Block) {
    let config = block.config();

    // last block of the round
    if (block.details.height + 1) % config.rounds.blocks_per_round != 0 {
        return;
    }

    let mut merge_algorithm_by_challenge = HashMap::<String, Algorithm>::new();
    for algorithm_id in block.data().active_algorithm_ids.iter() {
        let algorithm = get_algorithm_by_id(ctx, algorithm_id, Some(&block.id))
            .await
            .unwrap_or_else(|e| panic!("get_algorithm_by_id error: {:?}", e));
        let challenge_id = algorithm.details.challenge_id.clone();
        let data = algorithm.block_data();

        if algorithm.state().round_merged.is_some()
            || *data.merge_points() < config.algorithm_submissions.merge_points_threshold
        {
            continue;
        }
        if !merge_algorithm_by_challenge.contains_key(&challenge_id)
            || merge_algorithm_by_challenge[&challenge_id]
                .block_data()
                .merge_points
                < data.merge_points
        {
            merge_algorithm_by_challenge.insert(challenge_id, algorithm);
        }
    }

    let round_merged = block.details.round + 1;
    for (_, algorithm) in merge_algorithm_by_challenge.iter() {
        let mut state = algorithm.state().clone();

        state.round_merged = Some(round_merged);

        ctx.update_algorithm_state(&algorithm.id, &state)
            .await
            .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
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

async fn get_player_by_id<T: Context>(
    ctx: &mut T,
    player_id: &String,
    block_id: Option<&String>,
) -> anyhow::Result<Player> {
    Ok(ctx
        .get_players(
            PlayersFilter::Id(player_id.clone()),
            match block_id {
                Some(block_id) => Some(BlockFilter::Id(block_id.clone())),
                None => None,
            },
        )
        .await?
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting player {} to exist", player_id).as_str()))
}

async fn get_proof_by_benchmark_id<T: Context>(
    ctx: &mut T,
    benchmark_id: &String,
    include_data: bool,
) -> anyhow::Result<Proof, String> {
    Ok(ctx
        .get_proofs(
            ProofsFilter::BenchmarkId(benchmark_id.clone()),
            include_data,
        )
        .await
        .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting proof for benchmark {} to exist", benchmark_id).as_str()))
}

async fn get_benchmark_by_id<T: Context>(
    ctx: &mut T,
    benchmark_id: &String,
    include_data: bool,
) -> anyhow::Result<Benchmark> {
    Ok(ctx
        .get_benchmarks(BenchmarksFilter::Id(benchmark_id.clone()), include_data)
        .await?
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting benchmark {} to exist", benchmark_id).as_str()))
}

async fn get_fraud_by_id<T: Context>(
    ctx: &mut T,
    benchmark_id: &String,
    include_data: bool,
) -> anyhow::Result<Fraud> {
    Ok(ctx
        .get_frauds(
            FraudsFilter::BenchmarkId(benchmark_id.clone()),
            include_data,
        )
        .await?
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting fraud {} to exist", benchmark_id).as_str()))
}

async fn get_wasm_by_id<T: Context>(
    ctx: &mut T,
    algorithm_id: &String,
    include_data: bool,
) -> anyhow::Result<Wasm> {
    Ok(ctx
        .get_wasms(WasmsFilter::AlgorithmId(algorithm_id.clone()), include_data)
        .await?
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting wasm {} to exist", algorithm_id).as_str()))
}

async fn get_algorithm_by_id<T: Context>(
    ctx: &mut T,
    algorithm_id: &String,
    block_id: Option<&String>,
) -> anyhow::Result<Algorithm> {
    Ok(ctx
        .get_algorithms(
            AlgorithmsFilter::Id(algorithm_id.clone()),
            match block_id {
                Some(block_id) => Some(BlockFilter::Id(block_id.clone())),
                None => None,
            },
            false,
        )
        .await?
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting algorithm {} to exist", algorithm_id).as_str()))
}

async fn get_challenge_by_id<T: Context>(
    ctx: &mut T,
    challenge_id: &String,
    block_id: Option<&String>,
) -> anyhow::Result<Challenge> {
    Ok(ctx
        .get_challenges(
            ChallengesFilter::Id(challenge_id.clone()),
            match block_id {
                Some(block_id) => Some(BlockFilter::Id(block_id.clone())),
                None => None,
            },
        )
        .await?
        .first()
        .map(|x| x.to_owned())
        .expect(format!("Expecting challenge {} to exist", challenge_id).as_str()))
}
