use crate::{context::*, error::*};
use logging_timer::time;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn get_block_by_id<T: Context>(ctx: &T, block_id: &String) -> ProtocolResult<Block> {
    ctx.get_block(BlockFilter::Id(block_id.clone()), true)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .ok_or_else(|| ProtocolError::InvalidBlock {
            block_id: block_id.clone(),
        })
}

#[time]
pub async fn verify_sufficient_lifespan<T: Context>(
    ctx:                            &T,
    block:                          &Block
)                                           -> ProtocolResult<()>
{
    let latest_block                        = ctx
    .get_block(BlockFilter::Latest, false)
    .await
    .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
    .expect("Expecting latest block to exist");

    let config                              = block.config();
    let submission_delay                    = latest_block.details.height - block.details.height + 1;
    if (submission_delay as f64 * (config.benchmark_submissions.submission_delay_multiplier + 1.0))
        as u32
        >= config.benchmark_submissions.lifespan_period
    {
        return Err(ProtocolError::InsufficientLifespan);
    }

    return Ok(());
}

#[time]
pub fn verify_num_nonces(
    num_nonces:                     u32
)                                           -> ProtocolResult<()>
{
    if num_nonces == 0 
    {
        return Err(ProtocolError::InvalidNumNonces
        { 
            num_nonces
        });
    }

    return Ok(());
}

#[time]
pub(crate) fn get_fee_paid(
    player: &Player,
    num_nonces: u32,
    challenge: &Challenge,
) -> ProtocolResult<PreciseNumber> {
    let num_nonces = PreciseNumber::from(num_nonces);
    let fee_paid = challenge.block_data().base_fee().clone()
        + challenge.block_data().per_nonce_fee().clone() * num_nonces;
    if !player
        .state
        .as_ref()
        .is_some_and(|s| *s.available_fee_balance.as_ref().unwrap() >= fee_paid)
    {
        return Err(ProtocolError::InsufficientFeeBalance {
            fee_paid,
            available_fee_balance: player
                .state
                .as_ref()
                .map(|s| s.available_fee_balance().clone())
                .unwrap_or(PreciseNumber::from(0)),
        });
    }
    Ok(fee_paid)
}
