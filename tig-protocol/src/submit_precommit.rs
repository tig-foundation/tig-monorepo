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
