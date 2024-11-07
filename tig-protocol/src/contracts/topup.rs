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
    async fn verify_topup_tx(
        &self,
        ctx:                            &T,
        player:                         &Player,
        tx_hash:                        &String,
    )                                           -> ContractResult<PreciseNumber> 
    {
        let block                             = ctx
            .get_block_by_height(-1)
            .await
            .expect("No latest block found");

        if ctx
            .get_topups_by_tx_hash(tx_hash)
            .await
            .first()
            .is_some()
        {
            return Err(format!("Duplicate transaction: {}", tx_hash));
        }

        let transaction                         = ctx.get_transaction(&tx_hash)
            .await
            .expect(&format!("Invalid transaction: {}", tx_hash));

        if player.id != transaction.sender 
        {
            return Err(format!("Invalid transaction sender: expected {}, actual {}", player.id, transaction.sender));
        }

        let burn_address                        = &block.config().erc20.burn_address;
        if transaction.receiver != *burn_address 
        {
            return Err(format!("Invalid transaction receiver: expected {}, actual {}", burn_address, transaction.receiver));
        }

        let expected_amount                     = block.config().precommit_submissions().topup_amount;
        if transaction.amount != expected_amount 
        {
            return Err(format!("Invalid transaction amount: expected {}, actual {}", expected_amount, transaction.amount));
        }

        return Ok(expected_amount);
    }
}
