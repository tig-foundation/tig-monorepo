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

    fn update(cache: &AddBlockCache, block: &Block)
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
    }

    // update (call after opow.update)
    // update_adoption
    // update_merge_points
    // update_merges

    // FUTURE submit_brekthrough
    // FUTURE rename wasm -> binary
    // FUTURE update breakthrough adoption
    // FUTURE update breakthrough merge points
    // FUTURE update breakthrough merges
}
