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

pub struct ChallengesContext;
impl Context for ChallengesContext {}

pub trait IChallengesContract
{
    fn get_challenge_by_id(
        &self,
        ctx:                            &ChallengesContext,
        challenge_id:                   &String,
        block:                          &Block,
    )                                           -> Pin<Box<dyn Future<Output = ProtocolResult<Challenge>>>>;
}

impl dyn IChallengesContract
{
    fn get_challenge_by_id(
        &self,
        ctx:                            &ChallengesContext,
        challenge_id:                   &String,
        block:                          &Block,
    )                                           -> Pin<Box<dyn Future<Output = ProtocolResult<Challenge>>>>
    {
        Box::pin(async move 
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
        })
    }
}