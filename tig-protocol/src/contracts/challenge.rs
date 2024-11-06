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

    #[time]
    pub async fn fetch_challenge_by_id_and_block<'a>(
        &mut self,
        ctx:                    &T,
        id:                     &'a String,
        block:                  &Block,
        include_data:           bool
    )                                   -> ContractResult<'a, Challenge>
    {
        if !block.data().active_challenge_ids.contains(id) 
        {
            return Err(ProtocolError::InvalidChallenge 
            {
                challenge_id                : id,
            });
        }

        return Ok(
            ctx.get_challenge_by_id_and_height(id, block.details.height as u64, include_data)
                .await
                .unwrap()
                .expect("Challenge not found")
        );
    }

    #[time]
    async fn verify_challenge_exists<'a>(
        &self,
        ctx:                    &T,
        details:                &'a AlgorithmDetails,
    )                                   -> ContractResult<'a, ()> 
    {
        let latest_block = ctx
            .get_block_by_height(-1, false)
            .await
            .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
            .expect("Expecting latest block to exist");

        if !ctx
            .get_challenges_by_id(&details.challenge_id)
            .await
            .unwrap_or_else(|e| panic!("get_challenges error: {:?}", e))
            .first()
            .is_some_and(|c| {
                c.state()
                    .round_active
                    .as_ref()
                    .is_some_and(|r| *r <= latest_block.details.round)
            })
        {
            return Err(ProtocolError::InvalidChallenge 
            {
                challenge_id                : &details.challenge_id,
            });
        }
        Ok(())
    }
}
