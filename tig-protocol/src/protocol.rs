use
{
    crate::
    {
        cache::
        {
            Cache
        },
        ctx::
        {
            Context,
            ContextResult,
            ContextError,
        },
        contracts::
        {
            Contracts,
        },
    },
    std::
    {
        sync::
        {
            RwLock,
            Arc
        }
    },
};

pub struct Protocol<T: Context>
{
    ctx:                        T,
    cache:                      Arc<Cache<T>>,
    contracts:                  Contracts<T>,
}

impl<T: Context> Protocol<T>
{
    pub fn new(ctx: T)          -> Self
    {
        let cache                   = Arc::new(Cache::new());
        let mut new                 = Self 
        { 
            ctx                     : ctx,
            cache                   : cache.clone(),
            contracts               : Contracts::new(cache.clone()),
        };

        return new;
    }

    async fn run_block()
    {
    }
}
