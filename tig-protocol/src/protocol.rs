use
{
    crate::
    {
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
    contracts:                  Contracts<T>,
}

impl<T: Context> Protocol<T>
{
    pub fn new(ctx: T)          -> Self
    {
        let mut new                 = Self 
        { 
            ctx                     : ctx,
            contracts               : Contracts::new(),
        };

        return new;
    }

    async fn run_block()
    {
    }
}
