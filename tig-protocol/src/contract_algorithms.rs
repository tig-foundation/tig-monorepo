use
{
    crate::
    {
        AlgorithmsFilter,
        Block,
        BlockFilter,
        Context,
        ProtocolResult,
        ProtocolError
    },
    std::
    {
        future::
        {
            Future
        },
        pin::
        {
            Pin
        }
    }
};

pub struct AlgorithmsContract<T: Context>
{
    _phantom: std::marker::PhantomData<T>
}

impl<T: Context> AlgorithmsContract<T>
{
    pub fn new()                                -> Self 
    {
        return Self { _phantom: std::marker::PhantomData };
    }

    pub async fn verify_algorithm<'a>(
        &'a self, 
        ctx:                            &'a T,
        algorithm_id:                   &'a String,
        block:                          &'a Block
    )                                           -> ProtocolResult<()>
    {
        if !ctx
            .get_algorithms(AlgorithmsFilter::Id(algorithm_id.clone()), None, false)
            .await
            .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
            .pop()
            .is_some_and(|a| a.state.is_some_and(|s| !s.banned))
        {
            return Err(ProtocolError::InvalidAlgorithm 
            {
                algorithm_id                    : algorithm_id.clone(),
            });
        }

        if !block.data().active_algorithm_ids.contains(algorithm_id) 
        {
            return Err(ProtocolError::InvalidAlgorithm 
            {
                algorithm_id                    : algorithm_id.clone(),
            });
        }

        return Ok(());  
    }
}