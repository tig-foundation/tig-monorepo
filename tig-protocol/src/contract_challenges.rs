use
{
    crate::
    {
        Challenge,
        ChallengesFilter,
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

pub struct ChallengesContract<T: Context>
{
    _phantom: std::marker::PhantomData<T>
}

impl<T: Context> ChallengesContract<T>
{
    pub fn new()                                -> Self 
    {
        return Self { _phantom: std::marker::PhantomData };
    }

    pub async fn get_challenge_by_id<'a>(
        &'a self,
        ctx:                            &'a T,
        challenge_id:                   &'a String,
        block:                          &'a Block,
    )                                           -> ProtocolResult<Challenge>
    {
        if !block.data().active_challenge_ids.contains(challenge_id) 
        {
            return Err(ProtocolError::InvalidChallenge 
            {
                challenge_id                    : challenge_id.clone(),
            });
        }
    
        let challenge = ctx
            .get_challenges(ChallengesFilter::Id(challenge_id.clone()), Some(BlockFilter::Id(block.id.clone())))
            .await
            .unwrap_or_else(|e| panic!("get_challenges error: {:?}", e))
            .first()
            .map(|x| x.to_owned())
            .ok_or_else(|| ProtocolError::InvalidChallenge 
            {
                challenge_id                    : challenge_id.clone(),
            })?;
    
        return Ok(challenge);
    }
}