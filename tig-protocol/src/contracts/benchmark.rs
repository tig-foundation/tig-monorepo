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
            ContractResult,
        },
    },
    std::
    {
        marker::PhantomData,
        collections::HashSet,
        sync::RwLock,
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
        *
    },
    logging_timer::time
};

pub struct BenchmarkContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> BenchmarkContract<T>
{
    pub fn new()                        -> Self
    {
        return Self 
        {
            phantom                         : PhantomData,
        };
    }

    pub async fn submit_precommit(
        &self,
        ctx:                    &RwLock<T>,
        player:                 &Player,
        settings:               &BenchmarkSettings,
        num_nonces:             u32,
    )                                   -> ContractResult<String>
    {
        //verify that the player owns the benchmark
        if player.id != settings.player_id 
        {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //verify that the num nonces is greater than 0
        if num_nonces == 0 
        {
            return Err("Invalid num nonces".to_string());
        }

        //make sure that the submission delay is within the lifespan period
        let benchmark_block                 = ctx.read().unwrap().get_block_by_id(&settings.block_id).await.expect(&format!("Expecting benchmark block to exist: {}", settings.block_id));
        let latest_block                    = ctx.read().unwrap().get_block_by_height(-1).await.expect("Expecting latest block to exist");

        let config                          = benchmark_block.config();
        let submission_delay                = latest_block.details.height - benchmark_block.details.height + 1;
        if (submission_delay as f64 * (config.benchmark_submissions.submission_delay_multiplier + 1.0)) as u32 >= config.benchmark_submissions.lifespan_period
        {
            return Err(format!("Insufficient lifespan"));
        }

        if !benchmark_block.data().active_challenge_ids.contains(&settings.challenge_id) 
        {
            return Err(format!("Invalid challenge: {}", settings.challenge_id));
        }

        let challenge                       = ctx.read().unwrap().get_challenge_by_id_and_block_id(&settings.challenge_id, &benchmark_block.id).await
            .expect(&format!("Invalid challenge: {}", settings.challenge_id));

        //verify that the algorithm is not banned
        let algorithm                       = ctx.read().unwrap().get_algorithm_by_id(&settings.algorithm_id).await.unwrap();
        if !algorithm.state.as_ref().is_some_and(|s| !s.banned)
        {
            return Err(format!("Invalid algorithm: {}", settings.algorithm_id));
        }
    
        if !benchmark_block.data().active_algorithm_ids.contains(&settings.algorithm_id)
        {
            return Err(format!("Invalid algorithm: {}", settings.algorithm_id));
        }

        // verify that benchmark settings are unique
        if ctx.read().unwrap().get_precommits_by_settings(settings).await.first().is_some()
        {
            return Err(format!("Duplicate benchmark settings"));
        }

        //verify benchmark difficulty
        let difficulty                      = &settings.difficulty;
        let difficulty_parameters           = &config.difficulty.parameters[&challenge.id];
        if difficulty.len() != difficulty_parameters.len()
            || difficulty.iter()
                .zip(difficulty_parameters.iter())
                .any(|(d, p)| *d < p.min_value || *d > p.max_value)
        {
            return Err(format!("Invalid difficulty: {}", 
                difficulty.iter().zip(difficulty_parameters.iter()).map(|(d, p)| format!("{}: {}..{}", p.name, p.min_value, p.max_value)).collect::<Vec<String>>().join(", "))
            );
        }

        let challenge_data                  = challenge.block_data();
        let (lower_frontier, upper_frontier) = if *challenge_data.scaling_factor() > 1f64 {
            (
                challenge_data.base_frontier(),
                challenge_data.scaled_frontier(),
            )
        } else {
            (
                challenge_data.scaled_frontier(),
                challenge_data.base_frontier(),
            )
        };
        
        match difficulty.within(lower_frontier, upper_frontier) 
        {
            PointCompareFrontiers::Above => 
            {
                return Err(format!("Difficulty above hardest frontier: {}", 
                    difficulty.iter().zip(difficulty_parameters.iter()).map(|(d, p)| format!("{}: {}..{}", p.name, p.min_value, p.max_value)).collect::<Vec<String>>().join(", "))
                );
            }
            PointCompareFrontiers::Below => 
            {
                return Err(format!("Difficulty below easiest frontier: {}", 
                    difficulty.iter().zip(difficulty_parameters.iter()).map(|(d, p)| format!("{}: {}..{}", p.name, p.min_value, p.max_value)).collect::<Vec<String>>().join(", "))
                );
            }
            PointCompareFrontiers::Within => {}
        }

        //verify fee paid
        let fee_paid                        = challenge.block_data().base_fee()
                                                + challenge.block_data().per_nonce_fee() * PreciseNumber::from(num_nonces);

        if !player.state.as_ref().is_some_and(|s| *s.available_fee_balance.as_ref().unwrap() >= fee_paid)
        {
            return Err(format!("Insufficient fee balance: expected {}, actual {}", fee_paid, player
                .state.as_ref()
                .map(|s| s.available_fee_balance())
                .unwrap_or(&PreciseNumber::from(0)),
            ));
        }

        //submit precommit
        let benchmark_id = ctx.write().unwrap().add_precommit_to_mempool(settings, &PrecommitDetails 
        {
            block_started                   : benchmark_block.details.height,
            num_nonces                      : Some(num_nonces),
            fee_paid                        : Some(fee_paid),
        }).await
        .unwrap_or_else(|e| panic!("add_precommit_to_mempool error: {:?}", e));
        
        return Ok(benchmark_id);    
    }

    pub async fn submit_benchmark(
        &self,
        ctx:                    &RwLock<T>,
        player:                 &Player,
        benchmark_id:           &String,
        merkle_root:            &MerkleHash,
        solution_nonces:        &HashSet<u64>,
    )                                   -> ContractResult<()>
    {
        //verify that the benchmark is not already submitted
        if ctx.read().unwrap().get_benchmarks_by_id(benchmark_id).await.first().is_some()
        {
            return Err(format!("Duplicate benchmark: {}", benchmark_id));
        }
    
        //fetch the precommit
        let precommit                       = ctx.read().unwrap().get_precommits_by_benchmark_id(benchmark_id).await
            .pop()
            .filter(|p| p.state.is_some())
            .ok_or_else(|| format!("Invalid precommit: {}", benchmark_id))?;

        //verify that the player owns the precommit
        if player.id != precommit.settings.player_id
        {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //verify that the solution nonces are valid
        let num_nonces                      = *precommit.details.num_nonces.as_ref().unwrap() as u64;
        for n in solution_nonces.iter() 
        {
            if *n >= num_nonces 
            {
                return Err(format!("Invalid solution nonce: {}", n));
            }
        }

        ctx.write().unwrap().add_benchmark_to_mempool(benchmark_id, merkle_root, solution_nonces).await
            .unwrap_or_else(|e| panic!("add_benchmark_to_mempool error: {:?}", e));

        return Ok(());
    }

    pub async fn submit_proof(
        &self,
        ctx:                    &RwLock<T>,
        player:                 &Player,
        benchmark_id:           &String,
        merkle_proofs:          &Vec<MerkleProof>,
    )                                   -> ContractResult<Result<(), String>>
    {
        //verify that the proof is not already submitted
        if ctx.read().unwrap().get_proofs_by_benchmark_id(benchmark_id).await.first().is_some()
        {
            return Err(format!("Duplicate proof: {}", benchmark_id));
        }
        
        //fetch the precommit
        let precommit                       = ctx.read().unwrap().get_precommits_by_benchmark_id(benchmark_id).await
            .pop()
            .filter(|p| p.state.is_some())
            .ok_or_else(|| format!("Invalid precommit: {}", benchmark_id))?;

        //verify that the player owns the benchmark
        if player.id != precommit.settings.player_id 
        {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //fetch the benchmark
        let benchmark                       = ctx.read().unwrap().get_benchmarks_by_id(benchmark_id).await
            .pop()
            .filter(|b| b.state.is_some())
            .ok_or_else(|| format!("Invalid benchmark: {}", benchmark_id))?;


        //verify the sampled nonces
        let sampled_nonces                  = benchmark.state().sampled_nonces();
        let proof_nonces                    : HashSet<u64> = merkle_proofs.iter().map(|p| p.leaf.nonce).collect();
    
        if *sampled_nonces != proof_nonces || sampled_nonces.len() != merkle_proofs.len() 
        {
            return Err(format!("Invalid proof nonces"));
        }

        //verify the merkle proofs
        let max_branch_len                  = (64 - (*precommit.details.num_nonces.as_ref().unwrap() - 1).leading_zeros()) as usize;
        let expected_merkle_root            = benchmark.details.merkle_root.as_ref().unwrap();

        let mut is_fraudulent               = None;
        for merkle_proof in merkle_proofs.iter()
        {
            let branch                      = merkle_proof.branch.as_ref().unwrap();
            if branch.0.len() > max_branch_len || branch.0.iter().any(|(d, _)| *d as usize > max_branch_len)
            {
                is_fraudulent               = Some(format!("Invalid merkle proof: {}", merkle_proof.leaf.nonce));

                break;
            }

            let output_meta_data            = OutputMetaData::from(merkle_proof.leaf.clone());
            let hash                        = MerkleHash::from(output_meta_data);
            let result                      = merkle_proof
                .branch
                .as_ref()
                .unwrap()
                .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);

            if !result.is_ok_and(|actual_merkle_root| actual_merkle_root == *expected_merkle_root)
            {
                is_fraudulent               = Some(format!("Invalid merkle proof: {}", merkle_proof.leaf.nonce));
                
                break;
            }
        }

        //verify the solutions
        if !is_fraudulent.is_some()
        {
            for p in merkle_proofs.iter() 
            {
                if ctx.read().unwrap().verify_solution(&precommit.settings, p.leaf.nonce, &p.leaf.solution).await.unwrap().is_err()
                {
                    is_fraudulent           = Some(format!("Invalid solution: {}", p.leaf.nonce));

                    break;
                }
            }
        }

        //add the proof to the mempool
        ctx.write().unwrap().add_proof_to_mempool(benchmark_id, &merkle_proofs).await.unwrap_or_else(|e| panic!("add_proof_to_mempool error: {:?}", e));

        //add fraud to the mempool if the proof is fraudulent
        if is_fraudulent.is_some()
        {
            ctx.write().unwrap().add_fraud_to_mempool(benchmark_id, &is_fraudulent.clone().unwrap().to_string()).await.unwrap_or_else(|e| panic!("add_fraud_to_mempool error: {:?}", e));

            return Ok(Err(is_fraudulent.unwrap()));
        }

        return Ok(Ok(()));
    }
}