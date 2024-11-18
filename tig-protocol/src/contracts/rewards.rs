#[time]
async fn update_innovator_rewards(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let adoption_threshold =
        PreciseNumber::from_f64(config.algorithm_submissions.adoption_threshold);
    let zero = PreciseNumber::from(0);
    let mut eligible_algorithms_by_challenge = HashMap::<String, Vec<&mut Algorithm>>::new();
    for algorithm in cache.active_algorithms.values_mut() {
        let is_merged = algorithm.state().round_merged.is_some();
        let is_banned = algorithm.state().banned.clone();
        let data = algorithm.block_data.as_mut().unwrap();
        data.reward = Some(zero.clone());

        if !is_banned
            && (*data.adoption() >= adoption_threshold || (is_merged && *data.adoption() > zero))
        {
            eligible_algorithms_by_challenge
                .entry(algorithm.details.challenge_id.clone())
                .or_default()
                .push(algorithm);
        }
    }
    if eligible_algorithms_by_challenge.len() == 0 {
        return;
    }

    let reward_pool_per_challenge = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.optimisations)
        / PreciseNumber::from(eligible_algorithms_by_challenge.len());

    let zero = PreciseNumber::from(0);
    for algorithms in eligible_algorithms_by_challenge.values_mut() {
        let mut total_adoption = zero.clone();
        for algorithm in algorithms.iter() {
            total_adoption = total_adoption + algorithm.block_data().adoption();
        }

        for algorithm in algorithms.iter_mut() {
            let data = algorithm.block_data.as_mut().unwrap();
            let adoption = *data.adoption();
            data.reward = Some(reward_pool_per_challenge * adoption / total_adoption);
        }
    }
}

#[time]
async fn update_benchmarker_rewards(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let reward_pool = PreciseNumber::from_f64(get_block_reward(block))
        * PreciseNumber::from_f64(config.rewards.distribution.benchmarkers);

    for player in cache.active_players.values_mut() {
        let data = player.block_data.as_mut().unwrap();
        let influence = *data.influence();
        data.reward = Some(influence * reward_pool);
    }
}

/*
delegator rewards
breakthrough rewards
*/
