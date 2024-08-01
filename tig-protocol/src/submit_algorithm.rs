use crate::{context::*, error::*};
use logging_timer::time;
use std::collections::HashSet;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    details: &AlgorithmDetails,
    code: &String,
) -> ProtocolResult<String> {
    verify_challenge_exists(ctx, details).await?;
    verify_submission_fee(ctx, player, details).await?;
    let algorithm_id = ctx
        .add_algorithm_to_mempool(details, code)
        .await
        .unwrap_or_else(|e| panic!("add_algorithm_to_mempool error: {:?}", e));
    Ok(algorithm_id)
}

#[time]
async fn verify_challenge_exists<T: Context>(
    ctx: &T,
    details: &AlgorithmDetails,
) -> ProtocolResult<()> {
    let latest_block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("Expecting latest block to exist");
    if !latest_block
        .data()
        .active_challenge_ids
        .contains(&details.challenge_id)
    {
        return Err(ProtocolError::InvalidChallenge {
            challenge_id: details.challenge_id.clone(),
        });
    }
    Ok(())
}

#[time]
async fn verify_submission_fee<T: Context>(
    ctx: &T,
    player: &Player,
    details: &AlgorithmDetails,
) -> ProtocolResult<()> {
    let block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("No latest block found");

    if ctx
        .get_algorithms(
            AlgorithmsFilter::TxHash(details.tx_hash.clone()),
            None,
            false,
        )
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
        .first()
        .is_some()
    {
        return Err(ProtocolError::DuplicateSubmissionFeeTx {
            tx_hash: details.tx_hash.clone(),
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

    let transaction = ctx.get_transaction(&details.tx_hash).await.map_err(|_| {
        ProtocolError::InvalidTransaction {
            tx_hash: details.tx_hash.clone(),
        }
    })?;
    if !valid_senders.contains(&transaction.sender) {
        return Err(ProtocolError::InvalidSubmissionFeeSender {
            tx_hash: details.tx_hash.clone(),
            expected_sender: player.id.clone(),
            actual_sender: transaction.sender.clone(),
        });
    }
    let burn_address = block.config().erc20.burn_address.clone();
    if transaction.receiver != burn_address {
        return Err(ProtocolError::InvalidSubmissionFeeReceiver {
            tx_hash: details.tx_hash.clone(),
            expected_receiver: burn_address,
            actual_receiver: transaction.receiver.clone(),
        });
    }

    let expected_amount = block.config().algorithm_submissions.submission_fee;
    if transaction.amount != expected_amount {
        return Err(ProtocolError::InvalidSubmissionFeeAmount {
            tx_hash: details.tx_hash.clone(),
            expected_amount: jsonify(&expected_amount),
            actual_amount: jsonify(&transaction.amount),
        });
    }
    Ok(())
}
