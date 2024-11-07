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

pub struct ChallengeContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> ChallengeContract<T>
{
    pub fn new()                                -> Self
    {
        return Self 
        { 
            phantom                                 : PhantomData
        };
    }
}
