use
{
    crate::
    {
        ctx::
        {
            Context,
        },
        err::
        {
            ProtocolError,
            ContractResult,
        },
    },
    std::
    {
        sync::
        {
            RwLock,
            Arc
        },
        marker::
        {
            PhantomData,
        },
    },
    tig_structs::
    {
        core::
        {
            *
        }
    },
    logging_timer::time
};

pub struct PrecommitsContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> PrecommitsContract<T>
{
    pub fn new()                                -> Self
    {
        return Self 
        { 
            phantom                                 : PhantomData
        };
    }

    #[time]
    async fn get_precommit_by_id<'a>(
        ctx:                            &T,
        benchmark_id:                   &'a String,
    )                                           -> ContractResult<'a, Precommit> 
    {
        return ctx.get_precommits_by_benchmark_id(benchmark_id)
            .await
            .unwrap_or_else(|e| panic!("get_precommits_by_benchmark_id error: {:?}", e))
            .pop()
            .filter(|p| p.state.is_some())
            .ok_or_else(|| ProtocolError::InvalidPrecommit 
            {
                benchmark_id                        : benchmark_id,
            });
    }
}
