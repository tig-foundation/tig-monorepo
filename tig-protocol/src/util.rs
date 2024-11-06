use 
{
    crate::
    {
        cache::
        {
            Cache,
        },
        ctx::
        {
            Context,
        },
        err::
        {
            ProtocolError,
            ProtocolResult
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

pub(crate) fn verify_num_nonces(
    num_nonces:                             u32
)                                                   -> ProtocolResult<'static, ()> 
{
    if num_nonces == 0 
    {
        return Err(ProtocolError::InvalidNumNonces { num_nonces });
    }

    return Ok(());
}

pub(crate) async fn verify_sufficient_lifespan<'a, T: Context>(
    ctx:                                    &'a T,
    cache:                                  &'a Cache<T>,
    block:                                  &'a Block
)                                                   -> ProtocolResult<'a, ()> 
{
    let _                                               = cache
        .blocks
        .write()
        .unwrap()
        .fetch_block(ctx, -1);

    let block_cache                                     = cache
        .blocks
        .read()
        .unwrap();

    let latest_block                                    = block_cache
        .get_block(ctx, -1)
        .await
        .expect("Expecting latest block to exist");

    let config                                          = block.config();
    let submission_delay                                = latest_block.details.height - block.details.height + 1;
    if (submission_delay as f64 * (config.benchmark_submissions.submission_delay_multiplier + 1.0))
        as u32
        >= config.benchmark_submissions.lifespan_period
    {
        return Err(ProtocolError::InsufficientLifespan);
    }

    return Ok(());
}

pub(crate) async fn get_fee_paid<'a>(
    player:                                 &'a Player, 
    num_nonces:                             u32, 
    challenge:                              &'a Challenge
)                                                   -> ProtocolResult<'a, PreciseNumber>
{
    let num_nonces                                      = PreciseNumber::from(num_nonces);
    let fee_paid                                        = challenge.block_data().base_fee().clone()
                                                            + challenge.block_data().per_nonce_fee().clone() * num_nonces;
    if !player
        .state
        .as_ref()
        .is_some_and(|s| *s.available_fee_balance.as_ref().unwrap() >= fee_paid)
    {
        return Err(ProtocolError::InsufficientFeeBalance 
        {
            fee_paid,
            available_fee_balance                       : player
                .state
                .as_ref()
                .map(|s| s.available_fee_balance().clone())
                .unwrap_or(PreciseNumber::from(0)),
        });
    }

    return Ok(fee_paid);
}