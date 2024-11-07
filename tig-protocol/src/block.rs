use
{
    crate::
    {
        ctx::Context,
        contracts::Contracts,
    },
    std::
    {
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
    });
}

#[time]
pub async fn add_block<T: Context>(
    ctx:                    Arc<RwLock<T>>,
    contracts:              Arc<Contracts<T>>,
)                                   -> String
{
    let (mut block, mut cache)          = create_block(&Arc::into_inner(ctx.clone()).unwrap()).await;

    rayon::scope(|s|
    {
        s.spawn(|_| futures::executor::block_on(confirm_mempool_challenges(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_algorithms(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_challenges(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_algorithms(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_challenges(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_algorithms(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_challenges(&block, &Arc::into_inner(cache.clone()).unwrap())));
        s.spawn(|_| futures::executor::block_on(confirm_mempool_algorithms(&block, &Arc::into_inner(cache.clone()).unwrap()))); 
    });

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
