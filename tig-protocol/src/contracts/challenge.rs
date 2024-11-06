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

pub struct ChallengeContract<T: Context>
{
    cache:                              Arc<Cache<T>>,
    phantom:                            PhantomData<T>,
}   

impl<T: Context> ChallengeContract<T>
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
}
