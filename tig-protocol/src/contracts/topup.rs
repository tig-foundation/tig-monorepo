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
    tig_utils::
    {
        jsonify,
    },
    logging_timer::time,
};

pub struct TopUpContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> TopUpContract<T>
{
    pub fn new()                                -> Self
    {
        return Self 
        { 
            phantom                                 : PhantomData
        };
    }

    #[time]
    async fn verify_topup_tx<'a>(
        &self,
        ctx:                            &'a T,
        player:                         &'a Player,
        tx_hash:                        &'a String,
    )                                           -> ContractResult<'a, PreciseNumber> 
    {
        let block = ctx
            .get_block_by_height(-1, false)
            .await
            .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
            .expect("No latest block found");

        if ctx
            .get_topups_by_txid(tx_hash)
            .await
            .unwrap_or_else(|e| panic!("get_topups error: {:?}", e))
            .first()
            .is_some()
        {
            return Err(ProtocolError::DuplicateTransaction 
            {
                tx_hash                         : tx_hash,
            });
        }

        let transaction                         = ctx.get_transaction(&tx_hash)
            .await
            .map_err(|_| ProtocolError::InvalidTransaction 
            {
                tx_hash                         : tx_hash,
            })?;

        if player.id != transaction.sender 
        {
            return Err(ProtocolError::InvalidTransactionSender 
            {
                tx_hash                         : tx_hash,
                expected_sender                 : player.id.clone(),
                actual_sender                   : transaction.sender.clone(),
            });
        }

        let burn_address                        = &block.config().erc20.burn_address;
        if transaction.receiver != *burn_address 
        {
            return Err(ProtocolError::InvalidTransactionReceiver 
            {
                tx_hash                         : tx_hash,
                expected_receiver               : burn_address.to_string(),
                actual_receiver                 : transaction.receiver,
            });
        }

        let expected_amount                     = block.config().precommit_submissions().topup_amount;
        if transaction.amount != expected_amount 
        {
            return Err(ProtocolError::InvalidTransactionAmount 
            {
                tx_hash                         : tx_hash,
                expected_amount                 : expected_amount,
                actual_amount                   : transaction.amount,
            });
        }

        return Ok(expected_amount);
    }
}
