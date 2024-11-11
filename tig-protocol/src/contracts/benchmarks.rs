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
        core::*
    },
    tig_utils::*,
    logging_timer::time,
    rayon::prelude::*,
};

pub struct BenchmarkContract<T: Context>
{
    phantom: PhantomData<T>,
}   

impl<T: Context> BenchmarkContract<T> 
{
    pub fn new()                        -> Self
    {
        return Self { phantom: PhantomData };
    }

    pub async fn submit_precommit(
        &self,
        ctx:                    &T,
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

        let next_block_id               = ctx.get_next_block_id();
        //make sure that the submission delay is within the lifespan period
        let benchmark_block_details     = ctx.get_block_details(&settings.block_id).unwrap();
        let benchmark_block             = ctx.get_block_data(&settings.block_id).expect(&format!("Expecting benchmark block to exist: {}", settings.block_id));

        let latest_block_details        = ctx.get_block_details(next_block_id).unwrap();

        let config                      = ctx.get_block_config(&settings.block_id).unwrap();
        let submission_delay            = latest_block_details.height - benchmark_block_details.height + 1;
        if (submission_delay as f64 * (config.benchmark_submissions.submission_delay_multiplier + 1.0)) as u32 >= config.benchmark_submissions.lifespan_period
        {
            return Err(format!("Insufficient lifespan"));
        }

        if !benchmark_block.confirmed_challenge_ids.contains(&settings.challenge_id) 
        {
            return Err(format!("Invalid challenge: {}", settings.challenge_id));
        }

        //verify that the algorithm is not banned
        let algorithm = ctx.get_algorithm_state(&settings.algorithm_id, &settings.block_id).unwrap();
        if !algorithm.banned
        {
            return Err(format!("Invalid algorithm: {}", settings.algorithm_id));
        }
    
        if !benchmark_block.confirmed_algorithm_ids.contains(&settings.algorithm_id)
        {
            return Err(format!("Invalid algorithm: {}", settings.algorithm_id));
        }

        // verify that benchmark settings are unique
        if ctx.get_precommit_details(ctx.calc_benchmark_id(settings)).is_some()
        {
            return Err(format!("Duplicate benchmark settings"));
        }

        //verify benchmark difficulty
        let difficulty              = &settings.difficulty;
        let difficulty_parameters   = &config.difficulty.parameters[&settings.challenge_id];
        if difficulty.len() != difficulty_parameters.len()
            || difficulty.iter()
                .zip(difficulty_parameters.iter())
                .any(|(d, p)| *d < p.min_value || *d > p.max_value)
        {
            return Err(format!("Invalid difficulty: {}", 
                difficulty.iter().zip(difficulty_parameters.iter()).map(|(d, p)| format!("{}: {}..{}", p.name, p.min_value, p.max_value)).collect::<Vec<String>>().join(", "))
            );
        }

        let challenge_data = ctx.get_challenge_data(&settings.challenge_id, &settings.block_id).unwrap();
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
        let fee_paid = challenge_data.base_fee()
                        + challenge_data.per_nonce_fee() * PreciseNumber::from(num_nonces);

        if !player.state.as_ref().is_some_and(|s| *s.available_fee_balance.as_ref().unwrap() >= fee_paid)
        {
            return Err(format!("Insufficient fee balance: expected {}, actual {}", fee_paid, player
                .state.as_ref()
                .map(|s| s.available_fee_balance())
                .unwrap_or(&PreciseNumber::from(0)),
            ));
        }

        //submit precommit
        /*let benchmark_id = ctx.write().unwrap().add_precommit_to_mempool(settings, &PrecommitDetails 
        {
            block_started                   : benchmark_block.details.height,
            num_nonces                      : Some(num_nonces),
            fee_paid                        : Some(fee_paid),
        }).await
        .unwrap_or_else(|e| panic!("add_precommit_to_mempool error: {:?}", e));*/

        let result = ctx.add_precommit(settings, &PrecommitDetails 
        {
            block_started                   : benchmark_block_details.height,
            num_nonces                      : Some(num_nonces),
            fee_paid                        : Some(fee_paid),
        });

        if result.is_err()
        {
            panic!("add_precommit error: {:?}", result.err().unwrap());
        }

        /*ctx.get_player_state_mut(&player.id).unwrap().add_precommit(settings, &PrecommitDetails 
        {
            block_started                   : benchmark_block_details.height,
            num_nonces                      : Some(num_nonces),
            fee_paid                        : Some(fee_paid),
        });*/

        // add_precommit 
        // get_player_state_mut
        // add precommit to player state
        // 
        
        return Ok(String::new());    
    }

    pub async fn submit_benchmark(&self,
        ctx:                &T,
        player:             &Player,
        benchmark_id:       &String,
        merkle_root:        &MerkleHash,
        solution_nonces:    &HashSet<u64>,
    ) -> ContractResult<()>
    {
        //verify that the benchmark is not already submitted
        if ctx.get_benchmark_state(benchmark_id).is_some() 
        {
            return Err(format!("Duplicate benchmark: {}", benchmark_id));
        }
    
        //fetch the precommit
        let precommit_details = ctx.get_precommit_details(benchmark_id)
            .ok_or_else(|| format!("Invalid precommit: {}", benchmark_id))?;
        
        let settings = ctx.get_benchmark_settings(benchmark_id)
            .ok_or_else(|| format!("Invalid benchmark settings: {}", benchmark_id))?;

        //verify that the player owns the precommit
        if player.id != settings.player_id 
        {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //verify that the solution nonces are valid
        let num_nonces = precommit_details.num_nonces.unwrap() as u64;
        for n in solution_nonces.iter() 
        {
            if *n >= num_nonces 
            {
                return Err(format!("Invalid solution nonce: {}", n));
            }
        }

        // TODO: Add benchmark to mempool using appropriate context method
        // ctx.add_benchmark_to_mempool(benchmark_id, merkle_root, solution_nonces)?;

        return Ok(());
    }

    pub async fn submit_proof(
        &self,
        ctx:                    &T,
        player:                 &Player,
        benchmark_id:           &String,
        merkle_proofs:          &Vec<MerkleProof>,
    )                                   -> ContractResult<Result<(), String>>
    {
        //verify that the proof is not already submitted
        if ctx.get_proof_state(benchmark_id).is_some()
        {
            return Err(format!("Duplicate proof: {}", benchmark_id));
        }
        
        //fetch the precommit
        let precommit_details = ctx.get_precommit_details(benchmark_id)
            .ok_or_else(|| format!("Invalid precommit: {}", benchmark_id))?;

        let settings = ctx.get_benchmark_settings(benchmark_id)
            .ok_or_else(|| format!("Invalid benchmark settings: {}", benchmark_id))?;

        //verify that the player owns the benchmark
        if player.id != settings.player_id 
        {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //fetch the benchmark
        let benchmark_details = ctx.get_benchmark_details(benchmark_id)
            .ok_or_else(|| format!("Invalid benchmark: {}", benchmark_id))?;

        let benchmark_state = ctx.get_benchmark_state(benchmark_id)
            .ok_or_else(|| format!("Invalid benchmark state: {}", benchmark_id))?;

        //verify the sampled nonces
        let sampled_nonces  = &benchmark_state.sampled_nonces;
        let proof_nonces    : HashSet<u64> = merkle_proofs.iter().map(|p| p.leaf.nonce).collect();
    
        if sampled_nonces.as_ref().is_some_and(|s| s != &proof_nonces) || sampled_nonces.as_ref().is_some_and(|s| s.len() != merkle_proofs.len()) 
        {
            return Err(format!("Invalid proof nonces"));
        }

        //verify the merkle proofs
        let max_branch_len          = (64 - (precommit_details.num_nonces.unwrap() - 1).leading_zeros()) as usize;
        let expected_merkle_root    = benchmark_details.merkle_root.as_ref().unwrap();
        let mut is_fraudulent       = None;
        
        for merkle_proof in merkle_proofs.iter()
        {
            let branch = merkle_proof.branch.as_ref().unwrap();
            if branch.0.len() > max_branch_len || branch.0.iter().any(|(d, _)| *d as usize > max_branch_len)
            {
                is_fraudulent = Some(format!("Invalid merkle proof: {}", merkle_proof.leaf.nonce));
                break;
            }

            let output_meta_data    = OutputMetaData::from(merkle_proof.leaf.clone());
            let hash                = MerkleHash::from(output_meta_data);
            let result              = merkle_proof
                .branch
                .as_ref()
                .unwrap()
                .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);

            if !result.is_ok_and(|actual_merkle_root| actual_merkle_root == *expected_merkle_root)
            {
                is_fraudulent = Some(format!("Invalid merkle proof: {}", merkle_proof.leaf.nonce));
                break;
            }
        }

        //verify the solutions
        if !is_fraudulent.is_some()
        {
            for p in merkle_proofs.iter() 
            {
                if ctx.verify_solution(&settings, p.leaf.nonce, &p.leaf.solution).is_err()
                {
                    is_fraudulent = Some(format!("Invalid solution: {}", p.leaf.nonce));
                    break;
                }
            }
        }

        //add the proof to the mempool
        //ctx.add_proof_to_mempool(benchmark_id, merkle_proofs)?;

        //add fraud to the mempool if the proof is fraudulent
        if is_fraudulent.is_some()
        {
            //ctx.add_fraud_to_mempool(benchmark_id, &is_fraudulent.clone().unwrap())?;
            return Ok(Err(is_fraudulent.unwrap()));
        }
        
        return Ok(Ok(()));
    }
}