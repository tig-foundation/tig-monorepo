use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
        block::AddBlockCache,
    },
    logging_timer::time,
    std::{
        marker::PhantomData, 
        sync::RwLock,
        collections::HashMap,
    },
    tig_structs::core::*,
    tig_utils::PreciseNumberOps,
    rayon::prelude::*,
};

pub struct AlgorithmContract<T: Context + Send + Sync>
{
    phantom: PhantomData<T>,
}

impl<T: Context + Send + Sync> AlgorithmContract<T>
{
    pub fn new() -> Self {
        return Self { phantom: PhantomData };
    }

    async fn submit_algorithm(ctx: &T, player: &Player, details: &AlgorithmDetails, code: &String) -> ContractResult<String> 
    {
        if !ctx
            .get_challenge_state(&details.challenge_id, &ctx.get_latest_block_id())
            .is_some_and(|c| 
        {
            c.round_active
                .as_ref()
                .is_some_and(|r| *r <= ctx.get_block_details(&ctx.get_latest_block_id()).unwrap().round)
        })
        {
            return Err(format!("Invalid challenge: {}", details.challenge_id));
        }

        let block_config = ctx
            .get_block_config(&ctx.get_latest_block_id())
            .expect("No latest block found");

        if ctx
            .get_algorithm_details(&details.tx_hash)
            .is_some()
        {
            return Err(format!("Duplicate transaction: {}", details.tx_hash));
        }

        let transaction = ctx
            .get_transaction(&details.tx_hash)
            .expect("No transaction found");

        if player.id != transaction.sender 
        {
            return Err(format!("Invalid transaction sender: expected {}, actual {}", player.id, transaction.sender));
        }

        let burn_address = &block_config.erc20.burn_address;
        if transaction.receiver != *burn_address 
        {
            return Err(format!("Invalid transaction receiver: expected {}, actual {}", burn_address, transaction.receiver));
        }

        let expected_amount = block_config.algorithm_submissions.submission_fee;
        if transaction.amount != expected_amount 
        {
            return Err(format!("Invalid transaction amount: expected {}, actual {}", expected_amount, transaction.amount));
        };

        return Ok(ctx.add_algorithm(details, code).unwrap());

        /*
        {
            let player_state = ctx.get_player_state_mut(&player.id).unwrap();

            *player_state.available_fee_balance.as_mut().unwrap()   -= fee_paid;
            *player_state.total_fees_paid.as_mut().unwrap()         += fee_paid;
        }*/
    }  

    async fn submit_wasm() {
        // FIXME
    }

    pub fn update(&self, ctx: &T, cache: &AddBlockCache, block: &Block, prev_block_id: &String)
    {
        // update adoption
        {
            let mut algorithms_by_challenge = HashMap::<String, Vec<&String>>::new();
            let algorithms = cache.active_algorithms.read().unwrap();
            
            // First collect all algorithms into a map keyed by challenge_id
            {
                for algorithm in algorithms.iter() 
                {
                    let algorithm_details = ctx.get_algorithm_details(algorithm).unwrap();

                    algorithms_by_challenge
                        .entry(algorithm_details.challenge_id.clone())
                        .or_default()
                        .push(algorithm);
                }
            }

            for challenge_id in cache.active_challenges.read().unwrap().iter() 
            {
                let algorithms = algorithms_by_challenge.get_mut(challenge_id);
                if algorithms.is_none() 
                {
                    continue;
                }

                let algorithms  = algorithms.unwrap();
                let mut weights = Vec::<PreciseNumber>::new();
                for algorithm in algorithms.iter() 
                {
                    let algorithm_data = ctx.get_algorithm_data(algorithm, &block.id).unwrap();

                    let mut weight = PreciseNumber::from(0);
                    for (player_id, &num_qualifiers) in algorithm_data.num_qualifiers_by_player().iter()
                    {
                        let player_block_data = ctx.get_player_block_data(player_id).unwrap();

                        let num_qualifiers = PreciseNumber::from(num_qualifiers);
                        let influence = player_block_data.influence.unwrap();

                        let player_num_qualifiers = PreciseNumber::from(
                            *player_block_data
                                .num_qualifiers_by_challenge.as_ref().unwrap()
                                .get(challenge_id).unwrap(),
                        );

                        weight = weight + influence * num_qualifiers / player_num_qualifiers;
                    }
                    weights.push(weight);
                }

                let adoption = weights.normalise();
                
                // Update the adoptions back in the cache
                for (algorithm, adoption) in algorithms.iter().zip(adoption) 
                {
                    //algorithm.block_data.as_mut().unwrap().adoption = Some(adoption);
                    let algorithm_details = ctx.get_algorithm_details(algorithm).unwrap();

                    cache
                        .commit_algorithms_adoption.write().unwrap()
                        .insert(algorithm_details.challenge_id.clone(), (algorithm.to_string(), adoption));
                }
            }
        }

        // update merge points
        {
            let config = block.config();
            let adoption_threshold =PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
            cache.active_algorithms.read().unwrap().iter().for_each(|algorithm_id| 
            {
                let algorithm_state   = ctx.get_algorithm_state(algorithm_id, &block.id).unwrap();
                let algorithm_data    = ctx.get_algorithm_data(algorithm_id, &block.id).unwrap();

                let is_merged   = algorithm_state.round_merged.is_some();
                let data        = algorithm_data;

                // first block of the round
                let prev_merge_points = if block.details.height % config.rounds.blocks_per_round == 0 
                {
                    0
                } else 
                {
                    match ctx.get_algorithm_data(&algorithm_id, &prev_block_id)
                    {
                        Some(data)  => *data.merge_points(),
                        None        => 0,
                    }
                };

                /*data.merge_points = Some(if is_merged || *data.adoption() < adoption_threshold 
                {
                    prev_merge_points
                } 
                else 
                {
                    prev_merge_points + 1
                });*/

                let merge_points = Some(if is_merged || *data.adoption() < adoption_threshold 
                {
                    prev_merge_points
                } 
                else 
                {
                    prev_merge_points + 1
                }).unwrap();

                cache
                    .commit_algorithms_merge_points.write().unwrap()
                    .insert(algorithm_id.clone(), merge_points);
            });
        }

        //update merges
        {
            let config = block.config();

            // last block of the round
            if (block.details.height + 1) % config.rounds.blocks_per_round != 0 
            {
                return;
            }
        
            let mut algorithm_to_merge_by_challenge = HashMap::<String, (&String, &AlgorithmBlockData)>::new();
            let mut algorithms                      = cache.active_algorithms.read().unwrap();
            for algorithm_id in algorithms.iter() 
            {
                let algorithm_details = ctx.get_algorithm_details(algorithm_id).unwrap();
                let algorithm_data    = ctx.get_algorithm_data(algorithm_id, &block.id).unwrap();
                let algorithm_state   = ctx.get_algorithm_state(algorithm_id, &block.id).unwrap();
                let challenge_id      = algorithm_details.challenge_id.clone();
        
                if algorithm_state.round_merged.is_some()
                    || *algorithm_data.merge_points() < config.algorithm_submissions.merge_points_threshold
                {
                    continue;
                }
                if !algorithm_to_merge_by_challenge.contains_key(&challenge_id)
                    || algorithm_to_merge_by_challenge[&challenge_id]
                        .1.merge_points()
                        < algorithm_data.merge_points()
                {
                    algorithm_to_merge_by_challenge.insert(challenge_id, (algorithm_id, algorithm_data));
                }
            }
        
            let round_merged = block.details.round + 1;
            for (algorithm_id, algorithm_data) in algorithm_to_merge_by_challenge.values()
            {
                //let state           = algorithm.state.as_mut().unwrap();
                //state.round_merged  = Some(round_merged);

                cache
                    .commit_algorithms_merges.write().unwrap()
                    .insert(algorithm_id.to_string(), round_merged);
            }
        }
    }

    pub fn commit_update(&self, ctx: &T, cache: &AddBlockCache, block: &Block) -> ContractResult<()>
    {
        // commit adoption
        for (challenge_id, (algorithm_id, adoption)) in cache.commit_algorithms_adoption.read().unwrap().iter()
        {
            let algorithm_data = ctx.get_algorithm_data_mut(algorithm_id, &block.id).unwrap();

            algorithm_data.adoption = Some(adoption.to_owned());
        }

        // commit merge points
        for (algorithm_id, merge_points) in cache.commit_algorithms_merge_points.read().unwrap().iter()
        {
            let algorithm_data = ctx.get_algorithm_data_mut(algorithm_id, &block.id).unwrap();

            algorithm_data.merge_points = Some(merge_points.to_owned());
        }

        // commit merges
        for (algorithm_id, round_merged) in cache.commit_algorithms_merges.read().unwrap().iter()
        {
            let algorithm_state = ctx.get_algorithm_state_mut(algorithm_id, &block.id).unwrap();

            algorithm_state.round_merged = Some(round_merged.to_owned());
        }

        Ok(())
    }

    // FUTURE submit_brekthrough
    // FUTURE rename wasm -> binary
    // FUTURE update breakthrough adoption
    // FUTURE update breakthrough merge points
    // FUTURE update breakthrough merges
}
