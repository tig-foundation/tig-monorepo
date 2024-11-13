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
            let eligible_algorithms_by_challenge        = Arc::new(RwLock::new(HashMap::<String, Vec<(&String, &AlgorithmDetails, &AlgorithmBlockData)>>::new()));
            let algorithms                              = cache.active_algorithms.read().unwrap();
            algorithms.iter().for_each(|algorithm_id|
            {
                let algorithm_state     = ctx.get_algorithm_state(algorithm_id, &block.id).unwrap();
                let algorithm_details   = ctx.get_algorithm_details(algorithm_id).unwrap();
                let algorithm_data      = ctx.get_algorithm_data(algorithm_id, &block.id).unwrap();
                let is_merged           = algorithm_state.round_merged.is_some();
                let is_banned           = algorithm_state.banned.clone();
                //data.reward         = Some(zero.clone());

                if !is_banned 
                    && (*algorithm_data.adoption() >= adoption_threshold || (is_merged && *algorithm_data.adoption() > zero))
                {
                    eligible_algorithms_by_challenge
                        .write().unwrap()
                        .entry(algorithm_details.challenge_id.clone()).or_default()
                        .push((algorithm_id, algorithm_details, algorithm_data));
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
                        acc + algorithm.2.adoption()
                    });

                algorithms.iter().for_each(|(algorithm_id, algorithm_details, algorithm_data)| 
                {
                    let adoption    = algorithm_data.adoption();
                    //data.reward     = Some(reward_pool_per_challenge * adoption / total_adoption);

                    cache
                        .commit_innovator_rewards.write().unwrap()
                        .insert(algorithm_details.challenge_id.clone(), (algorithm_id.to_string(), reward_pool_per_challenge * adoption / total_adoption));
                });
            });
        }

        // update benchmarker rewards
        {
            let config      = block.config();
            let reward_pool = PreciseNumber::from_f64(block_reward) * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

            for player_id in cache.active_players.read().unwrap().iter()
            {
                let data        = ctx.get_player_block_data(player_id).unwrap();
                let influence   = *data.influence();

                cache
                    .commit_benchmarker_rewards.write().unwrap()
                    .insert(player_id.clone(), influence * reward_pool);

                //data.reward     = Some(influence * reward_pool);
            }
        }
    }

    pub fn commit_updates(&self, ctx: &T, cache: &AddBlockCache, block: &Block) -> ContractResult<()>
    {
        for (_, (algorithm_id, reward)) in cache.commit_innovator_rewards.read().unwrap().iter()
        {
            ctx.get_algorithm_data_mut(algorithm_id, &block.id).unwrap().reward = Some(reward.clone());
        }

        for (player_id, reward) in cache.commit_benchmarker_rewards.read().unwrap().iter()
        {
            ctx.get_player_block_data_mut(player_id).unwrap().reward = Some(reward.clone());
        }

        return Ok(());
    }
}
