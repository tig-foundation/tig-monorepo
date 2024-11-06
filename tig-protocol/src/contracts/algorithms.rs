use
{
    crate::
    {
        ctx::
        {
            Context,
            ContextResult,
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
    logging_timer::time,
};

pub struct AlgorithmsContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> AlgorithmsContract<T>
{
    pub fn new()                                -> Self
    {
        return Self 
        { 
            phantom                                 : PhantomData
        };
    }

    #[time]
    pub async fn fetch_algorithm<'a>(
        &self,
        ctx:                    &T,
        id:                     &'a String,
        include_data:           bool
    )                                   -> ContractResult<'a, Algorithm>
    {
        return Ok(ctx.get_algorithm_by_id(id, include_data).await.unwrap().expect("Algorithm not found"));
    }

    #[time]
    pub async fn verify_algorithm<'b>(
        &self,
        ctx:                            &T,
        algorithm_id:                   &'b String,
        block:                          &Block,
    )                                           -> ContractResult<'b, ()> 
    {
        let algorithm                   = self.fetch_algorithm(ctx, algorithm_id, false).await;

        if !algorithm.unwrap().state.as_ref().is_some_and(|s| !s.banned)
        {
            return Err(ProtocolError::InvalidAlgorithm 
            {
                algorithm_id                        : algorithm_id,
            });
        }

        if !block.data().active_algorithm_ids.contains(algorithm_id)
        {
            return Err(ProtocolError::InvalidAlgorithm 
            {
                algorithm_id                        : algorithm_id,
            });
        }

        return Ok(());
    }
}