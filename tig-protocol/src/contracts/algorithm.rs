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

pub struct AlgorithmContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> AlgorithmContract<T>
{
    pub fn new()                                -> Self
    {
        return Self 
        { 
            phantom                                 : PhantomData
        };
    }
}
