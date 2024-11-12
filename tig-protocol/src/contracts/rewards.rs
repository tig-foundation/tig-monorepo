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
            let eligible_algorithms_by_challenge        = Arc::new(RwLock::new(HashMap::<String, Vec<&Algorithm>>::new()));
            let algorithms                              = cache.active_algorithms.read().unwrap();
            algorithms.par_iter().for_each(|(_, algorithm)|
            {
                let is_merged   = algorithm.state().round_merged.is_some();
                let is_banned   = algorithm.state().banned.clone();
                let data        = algorithm.block_data.as_ref().unwrap();
                //data.reward     = Some(zero.clone());

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

            eligible_algorithms_by_challenge.read().unwrap().par_iter().for_each(|(_, algorithms)| 
            {
                let total_adoption = algorithms.iter()
                    .fold(zero.clone(), |acc, algorithm| 
                    {
                        acc + algorithm.block_data().adoption()
                    });

                algorithms.par_iter().for_each(|algorithm| 
                {
                    let data        = algorithm.block_data.as_ref().unwrap();
                    let adoption    = *data.adoption();
                    //data.reward     = Some(reward_pool_per_challenge * adoption / total_adoption);

                    cache
                        .commit_innovator_rewards.write().unwrap()
                        .insert(algorithm.details.challenge_id.clone(), (algorithm.id.clone(), reward_pool_per_challenge * adoption / total_adoption));
                });
            });
        }

        // update benchmarker rewards
        {
            let config      = block.config();
            let reward_pool = PreciseNumber::from_f64(block_reward) * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

            /*for player in ctx.get_active_players().iter()
            {
                let data        = player.block_data.as_ref().unwrap();
                let influence   = *data.influence();

                cache
                    .commit_benchmarker_rewards.write().unwrap()
                    .insert(player.id.clone(), influence * reward_pool);

                //data.reward     = Some(influence * reward_pool);
            }*/
        }
    }
}
