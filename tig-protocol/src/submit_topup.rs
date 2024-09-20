use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    tx_hash: String,
) -> ProtocolResult<()> {
    let topup_amount = verify_topup_tx(ctx, player, &tx_hash).await?;
    ctx.add_topup_to_mempool(
        &tx_hash,
        TopUpDetails {
            player_id: player.id.clone(),
            amount: topup_amount,
        },
    )
    .await
    .unwrap_or_else(|e| panic!("add_topup_to_mempool error: {:?}", e));
    Ok(())
}

#[time]
async fn verify_topup_tx<T: Context>(
    ctx: &T,
    player: &Player,
    tx_hash: &String,
) -> ProtocolResult<PreciseNumber> {
    let block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("No latest block found");

    if ctx
        .get_topups(TopUpsFilter::Id(tx_hash.clone()))
        .await
        .unwrap_or_else(|e| panic!("get_topups error: {:?}", e))
        .first()
        .is_some()
    {
        return Err(ProtocolError::DuplicateTransaction {
            tx_hash: tx_hash.clone(),
        });
    }
    let mut valid_senders = HashSet::<String>::new();
    valid_senders.insert(player.id.clone());
    if player.details.is_multisig {
        let multisig_owners = ctx
            .get_multisig_owners(&player.id)
            .await
            .unwrap_or_else(|e| panic!("get_multisig_owners error: {:?}", e));
        valid_senders.extend(multisig_owners.into_iter());
    }

    let transaction =
        ctx.get_transaction(&tx_hash)
            .await
            .map_err(|_| ProtocolError::InvalidTransaction {
                tx_hash: tx_hash.clone(),
            })?;
    if !valid_senders.contains(&transaction.sender) {
        return Err(ProtocolError::InvalidTransactionSender {
            tx_hash: tx_hash.clone(),
            expected_sender: player.id.clone(),
            actual_sender: transaction.sender.clone(),
        });
    }
    let burn_address = block.config().erc20.burn_address.clone();
    if transaction.receiver != burn_address {
        return Err(ProtocolError::InvalidTransactionReceiver {
            tx_hash: tx_hash.clone(),
            expected_receiver: burn_address,
            actual_receiver: transaction.receiver.clone(),
        });
    }

    let expected_amount = block.config().precommit_submissions.topup_amount.clone();
    if transaction.amount != expected_amount {
        return Err(ProtocolError::InvalidTransactionAmount {
            tx_hash: tx_hash.clone(),
            expected_amount: jsonify(&expected_amount),
            actual_amount: jsonify(&transaction.amount),
        });
    }
    Ok(expected_amount)
}
