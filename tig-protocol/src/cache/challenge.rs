use 
{
    crate::
    {
        err::
        {
            ContractResult,
            ProtocolError,
        },
        ctx::
        {
            Context,
        },
    },
    std::
    {
        marker::
        {
            PhantomData,
        },
        collections::
        {
            HashMap,
        },
    },
    tig_structs::
    {
        *,
        core::
        {
            *
        }
    }
};

pub struct ChallengeCache<T: Context>
{
    challenges:                 HashMap<String, HashMap<u64, Challenge>>,
    ctx:                        PhantomData<T>,
}

impl<T: Context> ChallengeCache<T>
{
    pub fn new() -> Self
    {
        return Self
        { 
            challenges                      : HashMap::new(),
            ctx                             : PhantomData
        };
    }

    pub async fn fetch_challenge_by_id_and_height<'a>(
        &mut self,
        ctx:                    &T,
        id:                     &'a String,
        block:                  &Block,
        include_data:           bool
    )                                   -> ContractResult<'a, &Challenge>
    {
        if !block.data().active_challenge_ids.contains(id) 
        {
            return Err(ProtocolError::InvalidChallenge 
            {
                challenge_id                : id,
            });
        }

        if !self.challenges.contains_key(id) 
        {
            self.challenges.insert(id.clone(), HashMap::new());
        }

        let block_height                    = block.details.height as u64;
        let block_challenges                = self.challenges.get_mut(id).unwrap();
        if !block_challenges.contains_key(&block_height)
        {
            let challenge                   = ctx.get_challenge_by_id_and_height(id, block_height, include_data).await;
            block_challenges.insert(block_height, challenge.expect("Challenge not found").unwrap());
        }

        return Ok(block_challenges.get(&block_height).unwrap());
    }

    pub fn get_challenge_by_id_and_height(
        &self,
        id:                     &String,
        block_height:           u64
    )                                   -> Option<&Challenge>
    {
        return self.challenges.get(id).and_then(|blocks| blocks.get(&block_height));
    }

    pub fn has_challenge_by_id_and_height(
        &self,
        id:                     &String,
        block_height:           u64
    )                                   -> bool
    {
        return self.challenges.get(id).map_or(false, |blocks| blocks.contains_key(&block_height));
    }
}
