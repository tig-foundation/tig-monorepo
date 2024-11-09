use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
    },
    logging_timer::time,
    std::{
        marker::PhantomData,
        sync::{Arc, RwLock},
        collections::{HashSet, HashMap},
    },
    tig_structs::core::*,
    rayon::prelude::*,
};

pub struct OPoWContract
{
}

impl OPoWContract {
    pub fn new() -> Self {
        return Self 
        {
        };
    }

    async fn submit_delegation_share() {
        // FIXME
    }

    fn update(cache: &mut crate::block::AddBlockCache, block: &Block)
    {
        // update cutoffs
        {
            let config                      = block.config();
            let mut phase_in_challenge_ids  : HashSet<String> = cache.active_challenges.read().unwrap().keys().cloned().collect();
            for algorithm in cache.active_algorithms.read().unwrap().values() 
            {
                if algorithm.state()
                    .round_pushed.is_some_and(|r| r + 1 <= block.details.round)
                {
                    phase_in_challenge_ids.remove(&algorithm.details.challenge_id);
                }
            }
        
            let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
            for (settings, num_solutions) in cache.active_solutions.read().unwrap().values() 
            {
                *num_solutions_by_player_by_challenge
                    .entry(settings.player_id.clone()).or_default()
                    .entry(settings.challenge_id.clone()).or_default() += *num_solutions;
            }
        
            num_solutions_by_player_by_challenge.par_iter().for_each(|(player_id, num_solutions_by_challenge)| 
            {
                let phase_in_start      = (block.details.round - 1) * config.rounds.blocks_per_round;
                let phase_in_period     = config.qualifiers.cutoff_phase_in_period.unwrap();
                let phase_in_end        = phase_in_start + phase_in_period;
                let min_cutoff          = config.qualifiers.min_cutoff.clone().unwrap();
                let min_num_solutions   = cache
                    .active_challenges.read().unwrap()
                    .keys()
                    .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0))
                    .min()
                    .unwrap();

                let mut cutoff = min_cutoff.max(
                    (*min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil() as u32
                );

                if phase_in_challenge_ids.len() > 0 && phase_in_end > block.details.height 
                {
                    let phase_in_min_num_solutions = cache
                        .active_challenges.read().unwrap()
                        .keys().filter(|&id| !phase_in_challenge_ids.contains(id))
                        .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0))
                        .min().unwrap();

                    let phase_in_cutoff = min_cutoff.max(
                        (*phase_in_min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil()
                            as u32,
                    );
                    
                    let phase_in_weight = (phase_in_end - block.details.height) as f64 / phase_in_period as f64;
                    cutoff              = (phase_in_cutoff as f64 * phase_in_weight + cutoff as f64 * (1.0 - phase_in_weight)) as u32;
                }
                
                cache
                    .active_players.write().unwrap()
                    .get_mut(player_id).unwrap()
                    .block_data.as_mut().unwrap()
                    .cutoff = Some(cutoff);
            });
        }
    }

    // update (order of ops)
    // 1. update_cutoffs
    // 2. update_qualifiers
    // 3. update_frontiers
    // 4. update_influence
}
