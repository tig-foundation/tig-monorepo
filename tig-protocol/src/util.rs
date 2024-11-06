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
    },
    std::collections::HashSet,
    logging_timer::time,
};

#[time]
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

#[time]
pub(crate) async fn verify_sufficient_lifespan<'a, T: Context>(
    ctx:                                    &'a T,
    block:                                  &'a Block
)                                                   -> ProtocolResult<'a, ()> 
{
    let latest_block                                    = ctx
        .get_block_by_height(-1, false)
        .await
        .unwrap()
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

#[time]
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

#[time]
pub(crate) fn verify_sampled_nonces<'a>(
    benchmark:                              &'a Benchmark,
    merkle_proofs:                          &Vec<MerkleProof>,
)                                                   -> ProtocolResult<'a, ()>
{
    let sampled_nonces                                  = benchmark.state().sampled_nonces().clone();
    let proof_nonces                                    : HashSet<u64> = merkle_proofs.iter().map(|p| p.leaf.nonce).collect();

    if sampled_nonces != proof_nonces || sampled_nonces.len() != merkle_proofs.len()
    {
        return Err(ProtocolError::InvalidProofNonces
        {
            submitted_nonces                        : merkle_proofs.iter().map(|p| p.leaf.nonce).collect(),
            expected_nonces                         : benchmark.state().sampled_nonces(),
        });
    }

    return Ok(());
}

#[time]
async fn verify_solutions_are_valid<'a, T: Context>(
    ctx:                                    &'a T,
    precommit:                              &'a Precommit,
    merkle_proofs:                          &'a Vec<MerkleProof>,
)                                                   -> ProtocolResult<'a, ()>
{
    for p in merkle_proofs.iter()
    {
        if ctx
            .verify_solution(&precommit.settings, p.leaf.nonce, &p.leaf.solution)
            .await
            .unwrap_or_else(|e| panic!("verify_solution error: {:?}", e))
            .is_err()
        {
            return Err(ProtocolError::InvalidSolution 
            {
                nonce                                   : p.leaf.nonce,
            });
        }
    }

    return Ok(());
}

#[time]
pub(crate) async fn verify_submission_fee<'a, T: Context>(
    ctx:                                    &'a T,
    player:                                 &'a Player,
    details:                                &'a AlgorithmDetails,
)                                                   -> ProtocolResult<'a, ()>
{
    let block                                          = ctx
        .get_block_by_height(-1, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("No latest block found");

    if ctx
        .get_algorithm_by_tx_hash(
            &details.tx_hash,
            false,
        )
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
        .is_some()
    {
        return Err(ProtocolError::DuplicateTransaction 
        {
            tx_hash                                    : &details.tx_hash,
        });
    }

    let transaction                                    = ctx
        .get_transaction(&details.tx_hash)
        .await
        .map_err(|_| ProtocolError::InvalidTransaction 
        {
            tx_hash                                    : &details.tx_hash,
        })?;

    if player.id != transaction.sender 
    {
        return Err(ProtocolError::InvalidTransactionSender 
        {
            tx_hash                                    : &details.tx_hash,
            expected_sender                            : player.id.clone(),
            actual_sender                              : transaction.sender,
        });
    }

    if transaction.receiver != *block.config().erc20.burn_address 
    {
        return Err(ProtocolError::InvalidTransactionReceiver 
        {
            tx_hash                                    : &details.tx_hash,
            expected_receiver                          : block.config().erc20.burn_address.clone(),
            actual_receiver                            : transaction.receiver,
        });
    }

    let expected_amount                                = &block.config().algorithm_submissions.submission_fee;
    if transaction.amount != *expected_amount 
    {
        return Err(ProtocolError::InvalidTransactionAmount 
        {
            tx_hash                                    : &details.tx_hash,
            expected_amount                            : *expected_amount,
            actual_amount                              : transaction.amount,
        });
    }

    return Ok(());
}

#[time]
pub(crate) async fn verify_solutions_with_algorithm<'a, T: Context>(
    ctx:                                    &'a T,
    precommit:                              &'a Precommit,
    proof:                                  &'a Proof,
)                                                   -> ProtocolResult<'a, ()>
{
    let settings                                        = &precommit.settings;
    let wasm_vm_config                                 = ctx
        .get_block_by_id(&settings.block_id, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect(format!("Expecting block {} to exist", settings.block_id).as_str())
        .config
        .unwrap()
        .wasm_vm;

    for merkle_proof in proof.merkle_proofs()
    {
        if let Ok(actual_solution_data)                 = ctx
            .compute_solution(settings, merkle_proof.leaf.nonce, &wasm_vm_config)
            .await
            .unwrap_or_else(|e| panic!("compute_solution error: {:?}", e))
        {
            if actual_solution_data == merkle_proof.leaf
            {
                continue;
            }
        }

        return Err(ProtocolError::InvalidSolutionData
        {
            algorithm_id                                : &settings.algorithm_id,
            nonce                                       : merkle_proof.leaf.nonce,
        });
    }

    return Ok(());
}