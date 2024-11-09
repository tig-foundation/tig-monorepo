use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
        block::AddBlockCache,
    },
    logging_timer::time,
    std::{
        marker::PhantomData,
        sync::{Arc, RwLock},
        collections::HashMap,
    },
    tig_structs::core::*,
    rayon::prelude::*,
};

pub struct RewardsContract<T: Context>
{
    phantom: PhantomData<T>,
}

impl<T: Context> RewardsContract<T> {
    pub fn new() -> Self {
        return Self { phantom: PhantomData };
    }

    pub fn update(&self, ctx: &T, cache: &AddBlockCache, block: &Block)
    {
        let config          = block.config();
        let block_reward    = config
            .rewards
            .schedule.iter()
            .filter(|s| s.round_start <= block.details.round)
            .last().unwrap_or_else(|| {
                panic!(
                    "get_block_reward error: Expecting a reward schedule for round {}",
                    block.details.round
                )
        }).block_reward;

        // update innovator rewards
        {
            let adoption_threshold                      = PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
            let zero                                    = PreciseNumber::from(0);
            let eligible_algorithms_by_challenge        = Arc::new(RwLock::new(HashMap::<String, Vec<&mut Algorithm>>::new()));

            let mut algorithms = cache.active_algorithms.write().unwrap();
            algorithms.par_iter_mut().for_each(|(_, algorithm)|
            {
                let is_merged   = algorithm.state().round_merged.is_some();
                let is_banned   = algorithm.state().banned.clone();
                let data        = algorithm.block_data.as_mut().unwrap();
                data.reward     = Some(zero.clone());

                if !is_banned 
                    && (*data.adoption() >= adoption_threshold || (is_merged && *data.adoption() > zero))
                {
                    eligible_algorithms_by_challenge
                        .write().unwrap()
                        .entry(algorithm.details.challenge_id.clone()).or_default()
                        .push(algorithm);
                }
            });

            if eligible_algorithms_by_challenge.read().unwrap().len() == 0 
            {
                return;
            }

            let reward_pool_per_challenge = PreciseNumber::from_f64(block_reward)
                * PreciseNumber::from_f64(config.rewards.distribution.optimisations)
                / PreciseNumber::from(eligible_algorithms_by_challenge.read().unwrap().len());

            eligible_algorithms_by_challenge.write().unwrap().par_iter_mut().for_each(|(_, algorithms)| 
            {
                let total_adoption = algorithms.iter()
                    .fold(zero.clone(), |acc, algorithm| 
                    {
                        acc + algorithm.block_data().adoption()
                    });

                algorithms.par_iter_mut().for_each(|algorithm| 
                {
                    let data        = algorithm.block_data.as_mut().unwrap();
                    let adoption    = *data.adoption();
                    data.reward     = Some(reward_pool_per_challenge * adoption / total_adoption);
                });
            });
        }

        // update benchmarker rewards
        {
            let config      = block.config();
            let reward_pool = PreciseNumber::from_f64(block_reward) * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

            for player in cache.active_players.write().unwrap().values_mut() 
            {
                let data        = player.block_data.as_mut().unwrap();
                let influence   = *data.influence();
                data.reward     = Some(influence * reward_pool);
            }
        }
    }
}
