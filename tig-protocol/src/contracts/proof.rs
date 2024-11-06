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

pub struct ProofsContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> ProofsContract<T>
{
    pub fn new()                                -> Self
    {
        return Self 
        { 
            phantom                                 : PhantomData
        };
    }

    #[time]
    async fn verify_proof_not_already_submitted<'a>(
        ctx:                            &T,
        benchmark_id:                   &'a String,
    )                                           -> ContractResult<'a, ()> 
    {
        if ctx
            .get_proofs_by_benchmark_id(benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
            .first()
            .is_some()
        {
            return Err(ProtocolError::DuplicateProof 
            {
                benchmark_id                        : benchmark_id,
            });
        }

        return Ok(());
    }
    
    #[time]
    fn verify_merkle_proofs<'a>(
        precommit:               &'a Precommit,
        benchmark:               &'a Benchmark,
        merkle_proofs:           &'a Vec<MerkleProof>,
    )                                   -> ContractResult<'a, ()>
    {
        let max_branch_len                  = (64 - (*precommit.details.num_nonces.as_ref().unwrap() - 1).leading_zeros()) as usize;
        let expected_merkle_root            = benchmark.details.merkle_root.clone().unwrap();

        for merkle_proof in merkle_proofs.iter()
        {
            let branch                      = merkle_proof.branch.as_ref().unwrap();

            if branch.0.len() > max_branch_len || branch.0.iter().any(|(d, _)| *d as usize > max_branch_len)
            {
                return Err(ProtocolError::InvalidMerkleProof 
                {
                    nonce                       : merkle_proof.leaf.nonce.clone(),
                });
            }

            let output_meta_data            = OutputMetaData::from(merkle_proof.leaf.clone());
            let hash                        = MerkleHash::from(output_meta_data);
            let result                      = merkle_proof
                .branch
                .as_ref()
                .unwrap()
                .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);

            if !result.is_ok_and(|actual_merkle_root| actual_merkle_root == expected_merkle_root)
            {
                return Err(ProtocolError::InvalidMerkleProof 
                {
                    nonce                       : merkle_proof.leaf.nonce.clone(),
                });
            }
        }

        return Ok(());
    }

    #[time]
    async fn get_proof_by_benchmark_id<'a>(
        ctx:                            &T,
        benchmark_id:                   &'a String,
    )                                           -> ContractResult<'a, Proof> 
    {
        return Ok(ctx
            .get_proofs_by_benchmark_id(benchmark_id, true)
            .await
            .unwrap_or_else(|e| panic!("get_proofs error: {:?}", e))
            .first()
            .map(|x| x.to_owned())
            .expect(format!("Expecting proof for benchmark {} to exist", benchmark_id).as_str())
        )
    }
}
