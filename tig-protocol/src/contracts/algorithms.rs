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
            ProtocolResult,
        },
        cache::
        {
            Cache
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
    }
};

pub struct AlgorithmsContract<T: Context>
{
    cache:                              Arc<Cache<T>>,
    phantom:                            PhantomData<T>,
}   

impl<T: Context> AlgorithmsContract<T>
{
    pub fn new(
        cache:                          Arc<Cache<T>>
    )                                           -> Self
    {
        return Self 
        { 
            cache                                   : cache, 
            phantom                                 : PhantomData
        };
    }

    pub async fn verify_algorithm<'b>(
        &self,
        ctx:                            &T,
        algorithm_id:                   &'b String,
        block:                          &Block,
    )                                           -> ProtocolResult<'b, ()> 
    {
        if !self.cache.algorithms.read().unwrap().has_algorithm(algorithm_id)
        {
            let _ = self.cache
                .algorithms
                .write()
                .unwrap()
                .fetch_algorithm(ctx, algorithm_id, false)
                .await;
        }

        if !self.cache.algorithms.read().unwrap().get_algorithm(ctx, algorithm_id).unwrap().state.as_ref().is_some_and(|s| !s.banned)
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