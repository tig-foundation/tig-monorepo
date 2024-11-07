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
    pub active_fee_players:     RwLock<HashMap<String, Player>>,
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
    });
}

#[time]
pub async fn add_block<T: Context>(
    ctx:                    Arc<RwLock<T>>,
    contracts:              Arc<Contracts<T>>,
)                                   -> String
{
    let (mut block, mut cache)          = create_block(&Arc::into_inner(ctx.clone()).unwrap()).await;

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

    return "".to_string();
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