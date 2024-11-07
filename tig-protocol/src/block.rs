use
{
    crate::
    {
        ctx::Context,
        contracts::Contracts,
    },
    std::
    {
        collections::HashMap,
        collections::HashSet,
        sync::
        {
            Arc,
            RwLock,
        },
    },
    tig_structs::
    {
        *,
        core::*,
    },
    tig_utils::
    {
        u64s_from_str,
    },
    rand::
    {
        Rng,
        SeedableRng,
        rngs::StdRng,
        seq::SliceRandom,
    },
    logging_timer::time,
    rayon::prelude::*,
};

struct AddBlockCache 
{
    pub mempool_challenges:     RwLock<Vec<Challenge>>,
    pub mempool_algorithms:     RwLock<Vec<Algorithm>>,
    pub mempool_benchmarks:     RwLock<Vec<Benchmark>>,
    pub mempool_precommits:     RwLock<Vec<Precommit>>,
    pub mempool_proofs:         RwLock<Vec<Proof>>,
    pub mempool_frauds:         RwLock<Vec<Fraud>>,
    pub mempool_topups:         RwLock<Vec<TopUp>>,
    pub mempool_wasms:          RwLock<Vec<Wasm>>,
    pub confirmed_precommits:   RwLock<HashMap<String, Precommit>>,
    pub active_challenges:      RwLock<HashMap<String, Challenge>>,
    pub active_algorithms:      RwLock<HashMap<String, Algorithm>>,
    pub active_solutions:       RwLock<HashMap<String, (BenchmarkSettings, u32)>>,
    pub active_players:         RwLock<HashMap<String, Player>>,
    pub active_fee_players:     RwLock<HashMap<String, Player>>,
    pub prev_players:           RwLock<HashMap<String, Player>>,
}

#[time]
pub async fn create_block<T: Context>(
    ctx:                    &RwLock<T>
)                                   -> (Block, Arc<AddBlockCache>)
{
    let cache                           = setup_cache(ctx).await;
    let block                           = Block 
    {
        id                              : "".to_string(),
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
    ctx:                    &RwLock<T>,
)                                   -> Arc<AddBlockCache>
{
    return Arc::new(AddBlockCache 
    {
        mempool_challenges              : RwLock::new(vec![]),
        mempool_algorithms              : RwLock::new(vec![]),
        mempool_benchmarks              : RwLock::new(vec![]),
        mempool_precommits              : RwLock::new(vec![]),
        mempool_proofs                  : RwLock::new(vec![]),
        mempool_frauds                  : RwLock::new(vec![]),
        mempool_topups                  : RwLock::new(vec![]),
        mempool_wasms                   : RwLock::new(vec![]),
        confirmed_precommits            : RwLock::new(HashMap::new()),
        active_fee_players              : RwLock::new(HashMap::new()),
        active_challenges               : RwLock::new(HashMap::new()),
        active_algorithms               : RwLock::new(HashMap::new()),
        active_solutions                : RwLock::new(HashMap::new()),
        prev_players                    : RwLock::new(HashMap::new()),
        active_players                  : RwLock::new(HashMap::new()),
    });
}

#[time]
pub async fn add_block<T: Context + std::marker::Send + std::marker::Sync>(
    ctx:                    Arc<RwLock<T>>,
    contracts:              Arc<Contracts<T>>,
)                                   -> String
{
    let (mut block, mut cache)          = create_block(&Arc::into_inner(ctx.clone()).unwrap()).await;

    // confirm mempool items
    async 
    {
        rayon::scope(|s|
        {
            s.spawn(|_| futures::executor::block_on(confirm_mempool_challenges(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_algorithms(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_precommits(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_benchmarks(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_proofs(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_frauds(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_topups(&block, &Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(confirm_mempool_wasms(&block, &Arc::into_inner(cache.clone()).unwrap())));
        });
    }.await;

    //update block details
    async 
    {
        rayon::scope(|s|
        {
            s.spawn(|_| futures::executor::block_on(update_deposits(&Arc::into_inner(ctx.clone()).unwrap(), &block, &mut Arc::into_inner(cache.clone()).unwrap())));
            s.spawn(|_| futures::executor::block_on(update_cutoffs(&block, &mut Arc::into_inner(cache.clone()).unwrap())));
        });
    }.await;

    // commit changes
    async
    {
        rayon::scope(|s|
        {
            // commit precommits
            s.spawn(|_|
            {
                for precommit in cache.mempool_precommits.write().unwrap().drain(..) 
                {
                    futures::executor::block_on(ctx.read().unwrap().update_precommit_state(&precommit.benchmark_id, &precommit.state.unwrap()))
                        .unwrap_or_else(|e| panic!("update_precommit_state error: {:?}", e));
                }
            });

            // commit algorithm states
            s.spawn(|_|
            {
                for algorithm in cache.mempool_algorithms.write().unwrap().drain(..) 
                {
                    futures::executor::block_on(ctx.read().unwrap().update_algorithm_state(&algorithm.id, &algorithm.state.unwrap()))
                        .unwrap_or_else(|e| panic!("update_algorithm_state error: {:?}", e));
                }
            });
        });
    }.await;

    return block.id;
}

#[time]
async fn confirm_mempool_challenges(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for challenge in cache.mempool_challenges.write().unwrap().iter_mut() 
    {
        let state                       = challenge.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_algorithms(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for algorithm in cache.mempool_algorithms.write().unwrap().iter_mut() 
    {
        let state                       = algorithm.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);
        state.round_submitted           = Some(block.details.round);
    }
}

#[time]
async fn confirm_mempool_precommits(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for precommit in cache.mempool_precommits.write().unwrap().iter_mut() 
    {
        let state                       = precommit.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);
        state.rand_hash                 = Some(block.id.clone());

        let fee_paid                    = *precommit.details.fee_paid.as_ref().unwrap();
        
        *cache
            .active_fee_players.write().unwrap().get_mut(&precommit.settings.player_id).unwrap()
            .state.as_mut().unwrap()
            .available_fee_balance.as_mut().unwrap()   -= fee_paid;

        *cache
            .active_fee_players.write().unwrap().get_mut(&precommit.settings.player_id).unwrap()
            .state.as_mut().unwrap()
            .total_fees_paid.as_mut().unwrap()   -= fee_paid;
    }
}

#[time]
async fn confirm_mempool_benchmarks(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    let config                          = block.config();
    for benchmark in cache.mempool_benchmarks.write().unwrap().iter_mut() 
    {
        let seed                        = u64s_from_str(format!("{:?}|{:?}", block.id, benchmark.id).as_str())[0];
        let mut rng                     = StdRng::seed_from_u64(seed);
        let mut sampled_nonces          = HashSet::new();
        let mut solution_nonces         = benchmark.solution_nonces.as_ref().unwrap().iter().cloned().collect::<Vec<u64>>();

        if solution_nonces.len() > 0 
        {
            solution_nonces.shuffle(&mut rng);
            for nonce in solution_nonces
                .iter()
                .take(config.benchmark_submissions.max_samples)
            {
                sampled_nonces.insert(*nonce);
            }
        }

        let solution_nonces             = benchmark.solution_nonces.as_ref().unwrap();
        let num_nonces                  = *cache.confirmed_precommits.read().unwrap().get(&benchmark.id).unwrap().details.num_nonces.as_ref().unwrap() as usize;
        if num_nonces > solution_nonces.len() 
        {
            if num_nonces > solution_nonces.len() * 2 
            {
                // use rejection sampling
                let stop_length         = config.benchmark_submissions.max_samples.min(num_nonces - solution_nonces.len())
                                            + sampled_nonces.len();

                while sampled_nonces.len() < stop_length 
                {
                    let nonce           = rng.gen_range(0..num_nonces as u64);
                    if sampled_nonces.contains(&nonce) || solution_nonces.contains(&nonce) 
                    {
                        continue;
                    }

                    sampled_nonces.insert(nonce);
                }
            } 
            else 
            {
                let mut non_solution_nonces: Vec<u64> = (0..num_nonces as u64)
                    .filter(|n| !solution_nonces.contains(n))
                    .collect();

                non_solution_nonces.shuffle(&mut rng);
                for nonce in non_solution_nonces.iter().take(config.benchmark_submissions.max_samples)
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
async fn confirm_mempool_proofs(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for proof in cache.mempool_proofs.write().unwrap().iter_mut() 
    {
        let state                       = proof.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);
        state.submission_delay          = Some(block.details.height 
                                            - cache.confirmed_precommits.read().unwrap().get(&proof.benchmark_id).unwrap().details.block_started
        );
    }
}

#[time]
async fn confirm_mempool_frauds(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for fraud in cache.mempool_frauds.write().unwrap().iter_mut() 
    {
        let state                       = fraud.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);
    }
}

#[time]
async fn confirm_mempool_topups(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for topup in cache.mempool_topups.write().unwrap().iter_mut() 
    {
        let state                       = topup.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);

        *cache
            .active_fee_players.write().unwrap().get_mut(&topup.details.player_id).unwrap()
            .state.as_mut().unwrap()
            .available_fee_balance.as_mut().unwrap()    += topup.details.amount;
    }
}

#[time]
async fn confirm_mempool_wasms(
    block:                  &Block,
    cache:                  &AddBlockCache,
)
{
    for wasm in cache.mempool_wasms.write().unwrap().iter_mut() 
    {
        let state                       = wasm.state.as_mut().unwrap();
        state.block_confirmed           = Some(block.details.height);
    }
}

#[time]
async fn update_deposits<T: Context>(
    ctx:                    &RwLock<T>,
    block:                  &Block,
    cache:                  &mut AddBlockCache,
)
{
    let decay                           = match &block.config().optimisable_proof_of_work.rolling_deposit_decay
    {
        Some(decay)                     => PreciseNumber::from_f64(*decay),
        None                            => return, // Proof of deposit not implemented for these blocks
    };

    let eth_block_num                   = block.details.eth_block_num();
    let zero                            = PreciseNumber::from(0);
    let one                             = PreciseNumber::from(1);
    for player in cache.active_fee_players.write().unwrap().values_mut() 
    {
        let rolling_deposit             = match &cache.prev_players.read().unwrap().get(&player.id).unwrap().block_data 
        {
            Some(data)                  => data.rolling_deposit.clone(),
            None                        => None,
        }
        .unwrap_or_else(|| zero.clone());

        let data                        = player.block_data.as_mut().unwrap();
        let deposit                     = ctx
            .read()
            .unwrap()
            .get_player_deposit(eth_block_num, &player.id)
            .await
            .unwrap_or_else(|| zero.clone());

        data.rolling_deposit            = Some(decay * rolling_deposit + (one - decay) * deposit);
        data.deposit                    = Some(deposit);
        data.qualifying_percent_rolling_deposit = Some(zero.clone());
    }
}
#[time]
async fn update_cutoffs(
    block:                  &Block,
    cache:                  &mut AddBlockCache,
)
{
    let config                          = block.config();
    let mut phase_in_challenge_ids      = HashSet::<String>::new();
    phase_in_challenge_ids              = cache.active_challenges.read().unwrap().keys().cloned().collect();

    for algorithm in cache.active_algorithms.read().unwrap().values() 
    {
        if algorithm.state().round_pushed.is_some_and(|r| r + 1 <= block.details.round)
        {
            phase_in_challenge_ids.remove(&algorithm.details.challenge_id);
        }
    }

    let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    for (settings, num_solutions) in cache.active_solutions.read().unwrap().values() 
    {
        *num_solutions_by_player_by_challenge
            .entry(settings.player_id.clone()).or_default()
            .entry(settings.challenge_id.clone()).or_default()         
                                        += *num_solutions;
    }

    for (player_id, num_solutions_by_challenge) in num_solutions_by_player_by_challenge.iter() 
    {
        let phase_in_start              = (block.details.round - 1) * config.rounds.blocks_per_round;
        let phase_in_period             = config.qualifiers.cutoff_phase_in_period.unwrap();
        let phase_in_end                = phase_in_start + phase_in_period;
        let min_cutoff                  = config.qualifiers.min_cutoff.unwrap();
        let min_num_solutions           = cache.active_challenges.read().unwrap().keys()
            .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0))
            .min().unwrap();

        let mut cutoff                  = min_cutoff.max((*min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil() as u32);
        if phase_in_challenge_ids.len() > 0 && phase_in_end > block.details.height 
        {
            let phase_in_min_num_solutions = cache
                .active_challenges.read().unwrap()
                .keys().filter(|&id| !phase_in_challenge_ids.contains(id))
                .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
                .min().unwrap();

            let phase_in_cutoff         = min_cutoff.max(
                (phase_in_min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil() as u32
            );

            let phase_in_weight         = (phase_in_end - block.details.height) as f64 / phase_in_period as f64;
            cutoff                      = (phase_in_cutoff as f64 * phase_in_weight + cutoff as f64 * (1.0 - phase_in_weight)) as u32;
        }

        cache.active_players.write().unwrap()
            .get_mut(player_id).unwrap()
            .block_data.as_mut().unwrap()
            .cutoff                     = Some(cutoff);
    }
}