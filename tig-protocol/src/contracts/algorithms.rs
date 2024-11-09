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
};

pub struct AlgorithmContract
{
}

impl AlgorithmContract {
    pub fn new() -> Self {
        return Self {};
    }

    async fn submit_algorithm() {
        // FIXME
    }

    async fn submit_wasm() {
        // FIXME
    }

    pub fn update(&self, cache: &AddBlockCache, block: &Block)
    {
        // update adoption
        {
            let mut algorithms_by_challenge = HashMap::<String, Vec<Algorithm>>::new();
            
            // First collect all algorithms into a map keyed by challenge_id
            {
                let algorithms = cache.active_algorithms.read().unwrap();
                for algorithm in algorithms.values() {
                    algorithms_by_challenge
                        .entry(algorithm.details.challenge_id.clone())
                        .or_default()
                        .push(algorithm.clone());
                }
            }

            for challenge_id in cache.active_challenges.read().unwrap().keys() 
            {
                let algorithms = algorithms_by_challenge.get_mut(challenge_id);
                if algorithms.is_none() 
                {
                    continue;
                }

                let algorithms = algorithms.unwrap();
                let mut weights = Vec::<PreciseNumber>::new();
                for algorithm in algorithms.iter() 
                {
                    let mut weight = PreciseNumber::from(0);
                    for (player_id, &num_qualifiers) in algorithm.block_data().num_qualifiers_by_player().iter()
                    {
                        let num_qualifiers = PreciseNumber::from(num_qualifiers);
                        let influence = cache
                            .active_players.read().unwrap()
                            .get(player_id).unwrap()
                            .block_data.as_ref().unwrap()
                            .influence.unwrap();

                        let player_num_qualifiers = PreciseNumber::from(
                            *cache
                                .active_players.read().unwrap()
                                .get(player_id).unwrap()
                                .block_data.as_ref().unwrap()
                                .num_qualifiers_by_challenge.as_ref().unwrap()
                                .get(challenge_id).unwrap(),
                        );

                        weight = weight + influence * num_qualifiers / player_num_qualifiers;
                    }
                    weights.push(weight);
                }

                let adoption = weights.normalise();
                
                // Update the adoptions back in the cache
                for (algorithm, adoption) in algorithms.iter_mut().zip(adoption) {
                    algorithm.block_data.as_mut().unwrap().adoption = Some(adoption);
                }
            }
        }

        // update merge points
        {
            let config = block.config();
            let adoption_threshold =PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
            for algorithm in cache.active_algorithms.write().unwrap().values_mut() 
            {
                let is_merged   = algorithm.state().round_merged.is_some();
                let data        = algorithm.block_data.as_mut().unwrap();

                // first block of the round
                let prev_merge_points = if block.details.height % config.rounds.blocks_per_round == 0 
                {
                    0
                } else 
                {
                    match cache.prev_algorithms.read().unwrap().get(&algorithm.id)
                    {
                        Some(data)  => *data.block_data.as_ref().unwrap().merge_points(),
                        None        => 0,
                    }
                };

                data.merge_points = Some(if is_merged || *data.adoption() < adoption_threshold 
                {
                    prev_merge_points
                } 
                else 
                {
                    prev_merge_points + 1
                });
            }
        }

        //update merges
        {
            let config = block.config();

            // last block of the round
            if (block.details.height + 1) % config.rounds.blocks_per_round != 0 
            {
                return;
            }
        
            let mut algorithm_to_merge_by_challenge = HashMap::<String, &mut Algorithm>::new();
            let mut algorithms                      = cache.active_algorithms.write().unwrap();
            for algorithm in algorithms.values_mut() 
            {
                let challenge_id    = algorithm.details.challenge_id.clone();
                let data            = algorithm.block_data();
        
                if algorithm.state().round_merged.is_some()
                    || *data.merge_points() < config.algorithm_submissions.merge_points_threshold
                {
                    continue;
                }
                if !algorithm_to_merge_by_challenge.contains_key(&challenge_id)
                    || algorithm_to_merge_by_challenge[&challenge_id]
                        .block_data()
                        .merge_points
                        < data.merge_points
                {
                    algorithm_to_merge_by_challenge.insert(challenge_id, algorithm);
                }
            }
        
            let round_merged = block.details.round + 1;
            for algorithm in algorithm_to_merge_by_challenge.values_mut() 
            {
                let state           = algorithm.state.as_mut().unwrap();
                state.round_merged  = Some(round_merged);
            }
        }
    }

    // FUTURE submit_brekthrough
    // FUTURE rename wasm -> binary
    // FUTURE update breakthrough adoption
    // FUTURE update breakthrough merge points
    // FUTURE update breakthrough merges
}
