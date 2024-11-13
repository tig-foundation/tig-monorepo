use
{
    crate::
    {
        contracts::Contracts, ctx::Context
    }, config::ProtocolConfig, logging_timer::time, rand::
    {
        rngs::StdRng, seq::SliceRandom, Rng, SeedableRng
    }, rayon::prelude::*, std::
    {
        collections::{HashMap, HashSet},
        sync::
        {
            Arc,
            RwLock,
        },
    }, tig_structs::
    {
        core::*, *
    }, tig_utils::u64s_from_str
};

pub struct AddBlockCache 
{
    //pub mempool_challenges:             RwLock<Vec<Challenge>>,
    //pub mempool_algorithms:             RwLock<Vec<Algorithm>>,
    //pub mempool_benchmarks:             RwLock<Vec<Benchmark>>,
    //pub mempool_precommits:             RwLock<Vec<Precommit>>,
    //pub mempool_proofs:                 RwLock<Vec<Proof>>,
    //pub mempool_frauds:                 RwLock<Vec<Fraud>>,
    //pub mempool_topups:                 RwLock<Vec<TopUp>>,
    //pub mempool_wasms:                  RwLock<Vec<Wasm>>,
    pub confirmed_precommits:           RwLock<HashMap<String, BenchmarkSettings>>,
    pub active_challenges:              RwLock<HashSet<String>>,
    pub active_algorithms:              RwLock<HashSet<String>>,
    pub active_solutions:               RwLock<HashMap<String, u32>>,
    pub active_players:                 RwLock<HashSet<String>>,
    //pub active_fee_players:             RwLock<HashMap<String, Player>>,
    //pub prev_challenges:                RwLock<HashMap<String, Challenge>>,
    //pub prev_algorithms:                RwLock<HashMap<String, Algorithm>>,
    //pub prev_players:                   RwLock<HashMap<String, Player>>,
    
    // more optimizedf prev fields
    //pub active_solutions:                           RwLock<Vec<(Arc<BenchmarkSettings>, u32)>>,
    // new fields we use to keep track of data to commit    
    pub commit_algorithms_adoption:                 RwLock<HashMap<String, (String, PreciseNumber)>>, // (challenge_id, (algorithm_id, adoption))
    pub commit_algorithms_merge_points:             RwLock<HashMap<String, u32>>,                    // (challenge_id, (algorithm_id, merge_points))
    pub commit_algorithms_merges:                   RwLock<HashMap<String, u32>>,                   // (challenge_id, algorithm_id)

    pub commit_challenges_solution_sig_thresholds:  RwLock<HashMap<String, u32>>,                            // (challenge_id, threshold)
    pub commit_challenges_fees:                     RwLock<HashMap<String, (PreciseNumber, PreciseNumber)>>, // (challenge_id, (base_fee, per_nonce_fee))

    pub commit_innovator_rewards:                   RwLock<HashMap<String, (String, PreciseNumber)>>,     // (challenge_id, reward)
    pub commit_benchmarker_rewards:                 RwLock<HashMap<String, PreciseNumber>>,               // (player_id, reward)

    pub commit_opow_cutoffs:                        RwLock<HashMap<String, u32>>,                           // acitve players[player_id].cutoff 
    pub commit_opow_add_qualifiers:                 RwLock<HashMap<String, (String, String, u32, Point)>>, // (challenge_id, algorithm_id, num_qualifiers, difficulty)
    pub commit_opow_frontiers:                      RwLock<HashMap<String, (Frontier, f64, Frontier)>>,     // (base_frontier, scaling_factor, scaled_frontier)
    pub commit_opow_influence:                      RwLock<HashMap<String, PreciseNumber>>,                 // (player_id, influence)
    pub commit_opow_qualifying_percent_rolling_deposit: RwLock<HashMap<String, PreciseNumber>>, // (player_id, qualifying_percent_rolling_deposit)
    pub commit_opow_player_imbalance:               RwLock<HashMap<String, (PreciseNumber, PreciseNumber)>>, // (player_id, (imbalance, imbalance_penalty))

    pub commit_players_deposits:                    RwLock<HashMap<String, (PreciseNumber, PreciseNumber, PreciseNumber)>>, // (player_id, (rolling_deposit, deposit, qualifying_percent_rolling_deposit))
}


#[time]
pub async fn create_block<T: Context>(
    ctx:                    &T,
    config:                 &ProtocolConfig,
    block_id:               &String
)                                   -> (Block, Arc<AddBlockCache>)
{
    let cache                           = setup_cache(ctx, &config, &block_id).await;
    let block                           = Block 
    {
        id                              : block_id.clone(),
        config                          : None,
        details                         : BlockDetails
        {
            height                      : 0,
            eth_block_num               : None,
            prev_block_id               : "".to_string(),
            round                       : 0,
            fees_paid                   : None,
            num_confirmed_challenges    : None,
            num_confirmed_algorithms    : None,
            num_confirmed_benchmarks    : None,
            num_confirmed_precommits    : None,
            num_confirmed_proofs        : None,
            num_confirmed_frauds        : None,
            num_confirmed_topups        : None,
            num_confirmed_wasms         : None,
            num_active_challenges       : None,
            num_active_algorithms       : None,
            num_active_benchmarks       : None,
            num_active_players          : None,
        },
        data                            : None,
    };

    return (block, cache);
}

#[time]
async fn setup_cache<T: Context>(
    ctx:                    &T,
    config:                 &ProtocolConfig,
    block_id:               &String
)                                   -> Arc<AddBlockCache>
{
    let latest_block = ctx.get_block_details(&block_id).unwrap();
    let cache = Arc::new(AddBlockCache 
    {
        //mempool_challenges              : RwLock::new(vec![]),
        //mempool_algorithms              : RwLock::new(vec![]),
        //mempool_benchmarks              : RwLock::new(vec![]),
        //mempool_precommits              : RwLock::new(vec![]),
        //mempool_proofs                  : RwLock::new(vec![]),
        //mempool_frauds                  : RwLock::new(vec![]),
        //mempool_topups                  : RwLock::new(vec![]),
        //mempool_wasms                   : RwLock::new(vec![]),
        confirmed_precommits            : RwLock::new(HashMap::new()),
        //active_fee_players              : RwLock::new(HashMap::new()),
        active_challenges               : RwLock::new(HashSet::new()),
        active_algorithms               : RwLock::new(HashSet::new()),
        active_solutions                : RwLock::new(HashMap::new()),
        //prev_players                    : RwLock::new(HashMap::new()),
        //prev_challenges                 : RwLock::new(HashMap::new()),
        //prev_algorithms                 : RwLock::new(HashMap::new()),
        active_players                  : RwLock::new(HashSet::new()),
        // new fields
        commit_algorithms_adoption                  : RwLock::new(HashMap::new()),
        commit_algorithms_merge_points              : RwLock::new(HashMap::new()),
        commit_algorithms_merges                    : RwLock::new(HashMap::new()),

        commit_challenges_solution_sig_thresholds   : RwLock::new(HashMap::new()),
        commit_challenges_fees                      : RwLock::new(HashMap::new()),

        commit_innovator_rewards                    : RwLock::new(HashMap::new()),
        commit_benchmarker_rewards                  : RwLock::new(HashMap::new()),

        commit_opow_cutoffs                         : RwLock::new(HashMap::new()),
        commit_opow_add_qualifiers                  : RwLock::new(HashMap::new()),
        commit_opow_frontiers                       : RwLock::new(HashMap::new()),
        commit_opow_influence                       : RwLock::new(HashMap::new()),
        commit_opow_qualifying_percent_rolling_deposit: RwLock::new(HashMap::new()),
        commit_opow_player_imbalance                : RwLock::new(HashMap::new()),

        commit_players_deposits                     : RwLock::new(HashMap::new()),
    });

    // grab challenges
    // mempool
    let mut challenges = HashSet::new();
    {
        for challenge in ctx.get_confirmed_challenges()
        {
            if challenge
                .state.as_ref().unwrap()
                .round_active
                .is_some_and(|r| r <= latest_block.round)
            {
                challenges.insert(challenge.id.clone());
            }
        }
    }

    // grab algorithms
    let mut algorithms: HashSet<String> = HashSet::new();
    {
        let confirmed_algorithms        = ctx.get_confirmed_algorithms();
        let challenges_with_algorithms  = confirmed_algorithms
            .iter()
            .filter(|a| a.state().round_pushed.is_some())
            .map(|a| a.details.challenge_id.clone())
            .collect::<HashSet<String>>();

        for algorithm in confirmed_algorithms
        {
            let state           = algorithm.state.as_ref().unwrap();
            let round_pushed    = *state.round_submitted()
                + if challenges_with_algorithms.contains(&algorithm.details.challenge_id) 
                {
                    config.algorithm_submissions.push_delay
                } else {
                    1
                };

            let wasm = ctx
                .get_wasm_by_algorithm_id(&algorithm.id);

            if latest_block.round >= *state.round_pushed.as_ref().unwrap_or(&round_pushed)
                && wasm.is_some_and(|w| w.details.compile_success)
            {
                /*let mut algorithm   = algorithm.clone();
                *algorithm.block_data.as_mut().unwrap() = AlgorithmBlockData 
                {
                    reward: None,
                    adoption: None,
                    merge_points: None,
                    num_qualifiers_by_player: None,
                    round_earnings: None,
                };

                if algorithm.state().round_pushed.is_none() 
                {
                    algorithm.state.as_mut().unwrap().round_pushed = Some(round_pushed);
                }*/

                algorithms.insert(algorithm.id.clone());
            }
        }
    }

    // grab solutions
    let mut solutions = HashMap::new();
    {
        for proof in ctx.get_confirmed_proofs_by_height_started(0)
        {
            let benchmark_id = &proof.benchmark_id;
            let fraud = ctx
                .get_frauds_by_benchmark_id(benchmark_id)
                .pop();

            if fraud.is_some_and(|f| f.state.is_some()) 
            {
                continue;
            }
            
            let state               = proof.state();
            let submission_delay    = *state.submission_delay();
            let block_confirmed     = *state.block_confirmed();
            let block_active        = block_confirmed
                                        + (submission_delay as f64 * config.benchmark_submissions.submission_delay_multiplier) as u32;

            if block_active > latest_block.height 
            {
                continue;
            }

            let benchmark_details = ctx
                .get_benchmark_details(benchmark_id)
                .unwrap();

            if benchmark_details.num_solutions == 0 
            {
                continue;
            }

            solutions.insert(
                benchmark_id.clone(),
                benchmark_details.num_solutions,
            );
        }

        // grab precommits
        let mut precommits: HashMap<String, BenchmarkSettings> = HashMap::new();
        for precommit in ctx.get_confirmed_precommits_by_height_started(0)
        {
            precommits.insert(precommit.benchmark_id.clone(), precommit.settings.clone());
        }

        //grab players
        let mut players = HashSet::new();
        for (benchmark_id, _) in solutions.iter() 
        {
            players.insert(precommits[benchmark_id].player_id.clone());
        }

        /*let mut active_fee_players = HashMap::<String, PlayerState>::new();
        for topup in topups.iter() 
        {
            let player_id = &topup.details.player_id;
            let mut player_state = match ctx.get_player_state(player_id) 
            {
                Some(state) => state.clone(),
                None        => PlayerState {
                    total_fees_paid: Some(PreciseNumber::from(0)),
                    available_fee_balance: Some(PreciseNumber::from(0))
                }
            };

            active_fee_players.insert(player_id.clone(), player_state);
        }*/

        *cache.confirmed_precommits.write().unwrap()    = precommits;
        *cache.active_challenges.write().unwrap()       = challenges;
        *cache.active_algorithms.write().unwrap()       = algorithms;
        *cache.active_solutions.write().unwrap()        = solutions;
        *cache.active_players.write().unwrap()          = players;
    }

    /*
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
    }*/

    return cache;
}